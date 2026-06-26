#include "rpp_mul_mat/rpp_mul_mat.h"

#if GGML_RPP_USE_RT

static int ggml_rpp_mat_mul_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_mat_mul_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
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

static bool ggml_rpp_set_io_bingdings_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_mul_bingdings");
    auto rpp_node = static_cast<rpp_openrt_mul_mat *>(rpp_base_node);
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

static void ggml_cpu_transpose_2d(const ggml_tensor * src, void * dst) {
    const int64_t n0        = src->ne[0];
    const int64_t n1        = src->ne[1];
    const size_t  type_size = ggml_type_size(src->type);

    const char * src_data = (const char *) src->data;
    char *       dst_data = (char *) dst;

    for (int64_t i1 = 0; i1 < n1; i1++) {
        for (int64_t i0 = 0; i0 < n0; i0++) {
            const char * src_ptr = src_data + (i0 + i1 * n0) * type_size;
            char *       dst_ptr = dst_data + (i1 + i0 * n1) * type_size;
            memcpy(dst_ptr, src_ptr, type_size);
        }
    }
}

static bool ggml_rpp_create_engine_mul_mat_ubatch(ggml_backend_rpp_context & ctx,
                                                  ggml_rpp_node *            rpp_base_node,
                                                  ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_openrt_mul_mat *>(rpp_base_node);
    // notice!!!  rt requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);

    // create weights
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    auto              rpp_network = rpp_node->network.get();
    ggml_tensor *     g_weights   = dst->src[0];
    infer1::ITensor * rt_weights  = nullptr;
    if (is_matmul_weight(g_weights)) {
        // notice!!!  rt requires the weigths must be 2-dimensional
        infer1::Dims dims;
        dims.nbDims = 2;
        dims.d[0]   = g_weights->ne[0];
        dims.d[1]   = g_weights->ne[1];
        for (int i = 0; i < dims.nbDims; i++) {
            dims.type[i] = (infer1::DimensionType) 0;
        }
        rt_weights = rpp_node->network->addInput(g_weights->name, ggml_rpp_dtype_mapping(g_weights->type), dims);

        if (!rt_weights) {
            GGML_LOG_ERROR("%s: create rt weigths for mulmat failed, name: %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
        }
    } else {
        GGML_LOG_INFO("%s: ggml tensor is not mul mat weigths, name: %s (%s)\n", __func__, g_weights->name,
                      ggml_op_name(g_weights->op));
        // notice!!!  rt requires the inputs must be 2-dimensional
        infer1::Dims dims;
        dims.nbDims = 2;
        dims.d[0]   = g_weights->ne[1];
        dims.d[1]   = g_weights->ne[0];
        for (int i = 0; i < dims.nbDims; i++) {
            dims.type[i] = (infer1::DimensionType) 0;
        }

        rt_weights = rpp_node->network->addInput(g_weights->name, ggml_rpp_dtype_mapping(g_weights->type), dims);
        if (!rt_weights) {
            GGML_LOG_ERROR("%s: create rt inputs for mulmat failed, name: %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
        }
        return false;
    }

    // create inputs
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    ggml_tensor *     g_inputs  = dst->src[1];
    infer1::ITensor * rt_inputs = nullptr;
    if (ggml_is_contiguous(g_inputs)) {
        // notice!!!  rt requires the inputs must be 2-dimensional
        infer1::Dims dims;
        dims.nbDims = 2;
        dims.d[0]   = ctx.n_ubatch;
        dims.d[1]   = g_inputs->ne[0];
        for (int i = 0; i < dims.nbDims; i++) {
            dims.type[i] = (infer1::DimensionType) 0;
        }

        rt_inputs = rpp_node->network->addInput(g_inputs->name, ggml_rpp_dtype_mapping(g_inputs->type), dims);
        if (!rt_inputs) {
            GGML_LOG_ERROR("%s: create rt inputs for mulmat failed, name: %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
        }
    } else {
        GGML_LOG_ERROR("%s: the inputs is not contiguous for mulmat, name: %s (%s)\n", __func__, g_inputs->name,
                       ggml_op_name(g_inputs->op));
        return false;
    }

    // create add operator
    auto mul_mat_layer = rpp_network->addMatrixMultiply(*rt_inputs, infer1::MatrixOperation::kNONE, *rt_weights,
                                                        infer1::MatrixOperation::kNONE);
    if (!mul_mat_layer) {
        GGML_LOG_ERROR("%s: addMatrixMultiply faied  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = mul_mat_layer->getOutput(0);
    ort_tensor->setName(dst->name);
    rpp_network->markOutput(*ort_tensor);

    // rpp_node->config->setFlag(infer1::BuilderFlag::kBF16);
    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_mul_mat(ggml_backend_rpp_context & ctx,
                                           ggml_rpp_node *            rpp_base_node,
                                           ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_openrt_mul_mat *>(rpp_base_node);
    // notice!!!  rt requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);

    // create weights
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    auto              rpp_network = rpp_node->network.get();
    ggml_tensor *     g_weights   = dst->src[0];
    infer1::ITensor * rt_weights  = nullptr;
    if (is_matmul_weight(g_weights)) {
        // notice!!!  rt requires the weigths must be 2-dimensional
        infer1::Dims dims;
        dims.nbDims = 2;
        dims.d[0]   = g_weights->ne[0];
        dims.d[1]   = g_weights->ne[1];
        for (int i = 0; i < dims.nbDims; i++) {
            dims.type[i] = (infer1::DimensionType) 0;
        }
        rt_weights = rpp_node->network->addInput(g_weights->name, ggml_rpp_dtype_mapping(g_weights->type), dims);
        if (!rt_weights) {
            GGML_LOG_ERROR("%s: create rt weigths for mulmat failed, name: %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
        }
    } else {
        GGML_LOG_INFO("%s: ggml tensor is not mul mat weigths, create input, name: %s (%s)\n", __func__,
                      g_weights->name, ggml_op_name(g_weights->op));
        // notice!!!  rt requires the inputs must be 2-dimensional
        infer1::Dims dims;
        dims.nbDims = 2;
        dims.d[0]   = g_weights->ne[1];
        dims.d[1]   = g_weights->ne[0];
        for (int i = 0; i < dims.nbDims; i++) {
            dims.type[i] = (infer1::DimensionType) 0;
        }

        rt_weights = rpp_node->network->addInput(g_weights->name, ggml_rpp_dtype_mapping(g_weights->type), dims);
        if (!rt_weights) {
            GGML_LOG_ERROR("%s: create rt inputs for mulmat failed, name: %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
        }
        return false;
    }

    // create inputs
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    ggml_tensor *     g_inputs  = dst->src[1];
    infer1::ITensor * rt_inputs = nullptr;
    if (ggml_is_contiguous(g_inputs)) {
        // notice!!!  rt requires the inputs must be 2-dimensional
        infer1::Dims dims;
        dims.nbDims = 2;
        dims.d[0]   = g_inputs->ne[1];
        dims.d[1]   = g_inputs->ne[0];
        for (int i = 0; i < dims.nbDims; i++) {
            dims.type[i] = (infer1::DimensionType) 0;
        }

        rt_inputs = rpp_node->network->addInput(g_inputs->name, ggml_rpp_dtype_mapping(g_inputs->type), dims);
        if (!rt_inputs) {
            GGML_LOG_ERROR("%s: create rt inputs for mulmat failed, name: %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
        }
    } else {
        GGML_LOG_ERROR("%s: the inputs is not contiguous for mulmat, name: %s (%s)\n", __func__, g_inputs->name,
                       ggml_op_name(g_inputs->op));
        return false;
    }

    // create add operator
    auto mul_mat_layer = rpp_network->addMatrixMultiply(*rt_inputs, infer1::MatrixOperation::kNONE, *rt_weights,
                                                        infer1::MatrixOperation::kNONE);
    if (!mul_mat_layer) {
        GGML_LOG_ERROR("%s: addMatrixMultiply faied  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = mul_mat_layer->getOutput(0);
    ort_tensor->setName(dst->name);
    rpp_network->markOutput(*ort_tensor);

    // rpp_node->config->setFlag(infer1::BuilderFlag::kBF16);
    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_openrt_mul_mat *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }

    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_mat_mul_seq_len(dst, rpp_node) > 1) {
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
        if (ctx.use_ubatch && ggml_rpp_mat_mul_seq_len(dst, rpp_node) > 1) {
            ret = ggml_rpp_create_engine_mul_mat_ubatch(ctx, rpp_node, dst);
        } else {
            ret = ggml_rpp_create_engine_mul_mat(ctx, rpp_node, dst);
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

bool ggml_rpp_op_openrt_mul_mat(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_openrt_mul_mat * rpp_node = nullptr;
    auto                 iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_mul_mat");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (ctx.use_ubatch && ggml_rpp_mat_mul_seq_len(dst, cur_node) > 1) {
                    if (cur_node->n_ubatch == ctx.n_ubatch) {
                        rpp_node = (rpp_openrt_mul_mat *) cur_node;
                        break;
                    }
                } else {
                    if (ggml_rpp_node_has_matching_properties(iter_node->first, cur_node)) {
                        rpp_node = (rpp_openrt_mul_mat *) cur_node;
                        break;
                    }
                }
            }
        }
        // find rpp_node from other graph, because of rpp_node can be shared if have same dims
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_mul_mat");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                if (!rpp_graph_tmp) {
                    continue;
                }
                for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter.first == dst) {
                        continue;
                    }
                    if (node_iter.first->op == GGML_OP_MUL_MAT && ggml_rpp_dims_is_same(node_iter.first, dst)) {
                        auto & node_vec = node_iter.second;
                        for (size_t i = 0; i < node_vec.size(); i++) {
                            auto cur_node = node_vec[i].get();
                            if (ggml_rpp_mat_mul_dims_is_same(dst, node_iter.first, cur_node)) {
                                auto new_node = std::make_unique<rpp_openrt_mul_mat>(dst, cur_node);
                                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                                rpp_node = (rpp_openrt_mul_mat *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
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
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_openrt_mul_mat");
            auto new_node = std::make_unique<rpp_openrt_mul_mat>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_openrt_mul_mat *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            if (!(ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ggml_rpp_set_io_bingdings_device(ctx, rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_openrt_mul_mat *) (iter->second);
        if (!ctx.rpp_io_buffers.count(dst)) {
            ctx.rpp_io_buffers[dst] = rpp_node->binding_o_buffers[dst];
        }
    }

    // compute add operator
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "execute_openrt_mul_mat");
        RPP_EXECUTE_CONTEXT(rpp_node->context, 1, rpp_node->binding_io_buffers.data(), ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }

    return true;
}
#endif
