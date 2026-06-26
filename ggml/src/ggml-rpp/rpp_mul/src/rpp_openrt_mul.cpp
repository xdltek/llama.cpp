#include "rpp_mul/rpp_mul.h"

#if GGML_RPP_USE_RT

static int ggml_rpp_mul_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_set_io_bingdings_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_mul_bingdings");
    auto rpp_node = static_cast<rpp_openrt_mul *>(rpp_base_node);
    // clear buffers
    rpp_node->binding_io_buffers.clear();
    for (int i = 0; i < rpp_node->engine->getNbBindings(); i++) {
        ggml_tensor * io_tensor = rpp_node->binding_io_tensors[i];
        void *        io_buffer = io_tensor->data;
        if (rpp_node->engine->bindingIsInput(i)) {
            rpp_node->binding_i_buffers[io_tensor] = io_buffer;
        } else {
            rpp_node->binding_o_buffers[io_tensor] = io_buffer;
        }
        rpp_node->binding_io_buffers.emplace_back(io_buffer);
        ctx.rpp_io_buffers.emplace(io_tensor, io_buffer);
    }
    return true;
}

static bool ggml_rpp_mul_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
    bool          dims_is_same              = true;
    ggml_tensor * dst_vec[GGML_MAX_SRC + 1] = { dst };
    auto          dst_iter =
        std::copy_if(dst->src, dst->src + GGML_MAX_SRC, dst_vec + 1, [](ggml_tensor * ptr) { return ptr != nullptr; });

    ggml_tensor * src_vec[GGML_MAX_SRC + 1] = { src };
    auto          src_iter =
        std::copy_if(src->src, src->src + GGML_MAX_SRC, src_vec + 1, [](ggml_tensor * ptr) { return ptr != nullptr; });
    int dst_size = dst_iter - dst_vec;
    int src_size = src_iter - src_vec;
    if (dst_size != src_size || rpp_node->ggml_node_properties.size() != src_size) {
        return false;
    }
    for (int i = 0; i < dst_size; i++) {
        auto & property = rpp_node->ggml_node_properties[src_vec[i]];
        auto & cur_dst  = dst_vec[i];
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (cur_dst->ne[i] != property.ne[i]) {
                dims_is_same = false;
                break;
            }
        }
        if (!dims_is_same) {
            break;
        }
    }
    return dims_is_same;
}

static bool ggml_rpp_create_engine_mul_ubatch(ggml_backend_rpp_context & ctx,
                                              ggml_rpp_node *            rpp_base_node,
                                              ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto          rpp_node                     = static_cast<rpp_openrt_mul *>(rpp_base_node);
    auto          rpp_network                  = rpp_node->network.get();
    // get n dims
    ggml_tensor * cur_tensor[GGML_MAX_SRC + 1] = { dst };
    auto          end_iter                     = std::copy_if(dst->src, dst->src + GGML_MAX_SRC, cur_tensor + 1,
                                                              [](ggml_tensor * ptr) { return ptr != nullptr; });
    int           ndims                        = ggml_rpp_n_dims(cur_tensor);

    // create inputs
    infer1::ITensor * inputs[2] = { nullptr, nullptr };
    for (size_t i = 0; i < 2; i++) {
        GGML_ASSERT(ggml_is_contiguous(dst->src[i]));
        if (is_mul_weight(dst->src[i])) {
            inputs[i]         = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, ndims);
            infer1::Dims dims = ggml_rpp_dims_mapping(dst->src[i], ndims);
            if (!inputs[i]) {
                GGML_LOG_ERROR("%s: creat input%ld failed for mul, %s (%s)\n", __func__, i, dst->name,
                               ggml_op_name(dst->op));
                return false;
            }
        } else {
            infer1::Dims dims = ggml_rpp_dims_mapping(dst->src[i], ndims, rpp_node);
            inputs[i]         = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, dims);
            if (!inputs[i]) {
                GGML_LOG_ERROR("%s: creat input%ld failed for mul, %s (%s)\n", __func__, i, dst->name,
                               ggml_op_name(dst->op));
                return false;
            }
        }
    }

    // create add operator
    auto opt_mul = rpp_network->addElementWise(*inputs[0], *inputs[1], infer1::ElementWiseOperation::kPROD);
    if (!opt_mul) {
        GGML_LOG_ERROR("%s: addElementWise mul faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = opt_mul->getOutput(0);
    ort_tensor->setName(dst->name);
    rpp_network->markOutput(*ort_tensor);

    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed,  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_mul(ggml_backend_rpp_context & ctx,
                                       ggml_rpp_node *            rpp_base_node,
                                       ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto              rpp_node                     = static_cast<rpp_openrt_mul *>(rpp_base_node);
    auto              rpp_network                  = rpp_node->network.get();
    // get n dims
    ggml_tensor *     cur_tensor[GGML_MAX_SRC + 1] = { dst };
    auto              end_iter                     = std::copy_if(dst->src, dst->src + GGML_MAX_SRC, cur_tensor + 1,
                                                                  [](ggml_tensor * ptr) { return ptr != nullptr; });
    int               ndims                        = ggml_rpp_n_dims(cur_tensor);
    // create inputs
    infer1::ITensor * inputs[2]                    = { nullptr, nullptr };
    for (size_t i = 0; i < 2; i++) {
        GGML_ASSERT(ggml_is_contiguous(dst->src[i]));
        if (is_mul_weight(dst->src[i])) {
            inputs[i] = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, ndims);
            if (!inputs[i]) {
                GGML_LOG_ERROR("%s: creat input%ld failed for mul, %s (%s)\n", __func__, i, dst->name,
                               ggml_op_name(dst->op));
                return false;
            }
        } else {
            inputs[i] = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, ndims);
            if (!inputs[i]) {
                GGML_LOG_ERROR("%s: creat input%ld failed for mul, %s (%s)\n", __func__, i, dst->name,
                               ggml_op_name(dst->op));
                return false;
            }
        }
    }

    // create add operator
    auto opt_mul = rpp_network->addElementWise(*inputs[0], *inputs[1], infer1::ElementWiseOperation::kPROD);
    if (!opt_mul) {
        GGML_LOG_ERROR("%s: addElementWise mul faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = opt_mul->getOutput(0);
    ort_tensor->setName(dst->name);
    rpp_network->markOutput(*ort_tensor);

    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed,  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_openrt_mul *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_mul_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }
    std::string save_path = STRINGIZE(GGML_RPP_SAVE_ENGINE);
    std::string load_path = STRINGIZE(GGML_RPP_LOAD_ENGINE);
    // load engine
    if (!load_path.empty()) {
        char engine_path[128];
        sprintf(engine_path, "%s/upatch%d_%s_%s.engine", load_path.c_str(), rpp_node->n_ubatch, ggml_op_name(dst->op),
                dst->name);
        ret = ggml_rpp_load_enigne(rpp_node, engine_path);
    }
    // create engine
    if (!ret) {
        if (ctx.use_ubatch && ggml_rpp_mul_seq_len(dst, rpp_node) > 1) {
            ret = ggml_rpp_create_engine_mul_ubatch(ctx, rpp_node, dst);
        } else {
            ret = ggml_rpp_create_engine_mul(ctx, rpp_node, dst);
        }
        if (ret && !save_path.empty()) {
            char engine_path[128];
            sprintf(engine_path, "%s/upatch%d_%s_%s.engine", save_path.c_str(), rpp_node->n_ubatch,
                    ggml_op_name(dst->op), dst->name);
            ggml_rpp_save_enigne(rpp_node, engine_path);
        }
    }
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        ggml_tensor * cur_tensor[GGML_MAX_SRC + 1] = { dst };
        auto          end_iter                     = std::copy_if(dst->src, dst->src + GGML_MAX_SRC, cur_tensor + 1,
                                                                  [](ggml_tensor * ptr) { return ptr != nullptr; });
        for (int i = 0; i < rpp_node->engine->getNbBindings(); i++) {
            std::string str = rpp_node->engine->getBindingName(i);
            for (int j = 0; j < end_iter - cur_tensor; j++) {
                if (std::string(cur_tensor[j]->name) == str) {
                    rpp_node->binding_io_tensors.emplace_back(cur_tensor[j]);
                    break;
                }
            }
        }
    }
    return ret;
}

bool ggml_rpp_op_openrt_mul(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    // static int rms_num = 0;
    rpp_openrt_mul * rpp_node = nullptr;
    auto             iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_mul");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                // use batch node, add inputs padding 0
                if (ctx.use_ubatch && ggml_rpp_mul_seq_len(dst, cur_node) > 1) {
                    if (cur_node->n_ubatch == ctx.n_ubatch) {
                        rpp_node = (rpp_openrt_mul *) cur_node;
                        break;
                    }
                } else {
                    if (ggml_rpp_node_has_matching_properties(iter_node->first, cur_node)) {
                        rpp_node = (rpp_openrt_mul *) cur_node;
                        break;
                    }
                }
            }
        }

        // find rpp_node from other graph, because of rpp_node can be shared if have same dims
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_mul");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                if (!rpp_graph_tmp) {
                    continue;
                }
                for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter.first == dst) {
                        continue;
                    }
                    if (node_iter.first->op == GGML_OP_MUL && ggml_rpp_dims_is_same(node_iter.first, dst)) {
                        auto & node_vec = node_iter.second;
                        for (size_t i = 0; i < node_vec.size(); i++) {
                            auto cur_node = node_vec[i].get();
                            if (ggml_rpp_mul_dims_is_same(dst, node_iter.first, cur_node)) {
                                auto new_node = std::make_unique<rpp_openrt_mul>(dst, cur_node);
                                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                                rpp_node = (rpp_openrt_mul *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                                // set io tensor
                                rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
                                rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
                                rpp_node->binding_io_tensors.emplace_back(dst);
                                // set node properties
                                ggml_rpp_node_set_properties(rpp_node, dst);
                                // set ubatch
                                rpp_node->n_ubatch      = cur_node->n_ubatch;
                                rpp_node->seq_len_index = cur_node->seq_len_index;
                                break;
                            }
                        }
                    }
                    if (rpp_node) {
                        break;
                    }
                }
                if (rpp_node) {
                    break;
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_openrt_mul");
            auto new_node = std::make_unique<rpp_openrt_mul>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_openrt_mul *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            if (!(ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ggml_rpp_set_io_bingdings_device(ctx, rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_openrt_mul *) (iter->second);
        if (!ctx.rpp_io_buffers.count(dst)) {
            ctx.rpp_io_buffers[dst] = rpp_node->binding_o_buffers[dst];
        }
    }
    // compute add operator
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "execute_openrt_mul");
        RPP_EXECUTE_CONTEXT(rpp_node->context, 1, rpp_node->binding_io_buffers.data(), ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }
    return true;
}
#endif  // __linux__
