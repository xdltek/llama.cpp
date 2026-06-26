#include "rpp_set_rows/rpp_set_rows.h"

#if GGML_RPP_USE_RT

static int ggml_rpp_mul_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->src[0]->ne[node->seq_len_index];
}

static bool ggml_rpp_set_rows_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
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

static bool ggml_rpp_create_engine_set_rows(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node    = static_cast<rpp_openrt_set_rows *>(rpp_base_node);
    auto rpp_network = rpp_node->network.get();
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // notice!!! create inputs, inputs/outputs must be 2 dims, and index must be 1 dims
    // now only support set_rows for 2d tensor, and type is bf16 or f32, not support f16/int8/int4 quantization
    infer1::ITensor * inputs[3] = { nullptr, nullptr, nullptr };
    inputs[0]                   = ggml_rpp_create_input_tensor(dst->src[0], rpp_node, 2);
    inputs[1]                   = ggml_rpp_create_input_tensor(dst->src[1], rpp_node, 1);
    inputs[2]                   = ggml_rpp_create_input_tensor(dst, rpp_node, 2);

    // create add operator
    auto opt_set_rows = rpp_network->addSetRows(*inputs[0], *inputs[1], *inputs[2]);
    if (!opt_set_rows) {
        GGML_LOG_ERROR("%s: addElementWise mul faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = opt_set_rows->getOutput(0);
    ort_tensor->setName(dst->name);
    if (dst->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16) {
        ort_tensor->setType(infer1::DataType::kBF);
    }
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

static bool ggml_rpp_set_io_bingdings_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_set_rows_bingdings");
    auto rpp_node = static_cast<rpp_openrt_set_rows *>(rpp_base_node);
    // clear buffers
    rpp_node->binding_io_buffers.clear();
    for (auto iter : rpp_node->pool_buffers) {
        ctx.pool().free(iter);
    }
    rpp_node->pool_buffers.clear();
    for (int i = 0; i < rpp_node->engine->getNbBindings(); i++) {
        ggml_tensor * io_tensor = rpp_node->binding_io_tensors[i];
        void *        io_buffer = io_tensor->data;
        if (rpp_node->engine->bindingIsInput(i)) {
            // because ggml use int64_t for index, but rpp only support uint32_t index, so need convert int64_t to uint32_t
            // if seq_len == 1, no need convert, uint32_t and int64_t is same
            if (io_tensor->type == GGML_TYPE_I64 && ggml_rpp_mul_seq_len(rpp_node->cur_ggml_tensor, rpp_node) > 1) {
                auto iter = ctx.rpp_io_buffers.find(io_tensor);
                if (iter != ctx.rpp_io_buffers.end()) {
                    io_buffer = ctx.rpp_io_buffers[io_tensor];
                } else {
                    // convert uint64_t to uint32_t
                    size_t               size = ggml_nbytes(io_tensor);
                    size_t               nele = ggml_nelements(io_tensor);
                    std::vector<int64_t> host_index_64(nele);
                    RPP_CHECK(rtMemcpy(host_index_64.data(), io_tensor->data, size, rtMemcpyDeviceToHost));
                    std::vector<uint32_t> host_index_32(nele);
                    for (int j = 0; j < nele; j++) {
                        host_index_32[j] = static_cast<uint32_t>(host_index_64[j]);
                    }
                    io_buffer = ctx.pool().alloc(nele * sizeof(uint32_t));
                    RPP_CHECK(rtMemcpy(io_buffer, host_index_32.data(), nele * sizeof(uint32_t), rtMemcpyHostToDevice));
                    rpp_node->pool_buffers.emplace(io_buffer);
                }
            }
            rpp_node->binding_i_buffers[io_tensor] = io_buffer;
        } else {
            rpp_node->binding_o_buffers[io_tensor] = io_buffer;
        }
        rpp_node->binding_io_buffers.emplace_back(io_buffer);
        ctx.rpp_io_buffers.emplace(io_tensor, io_buffer);
    }
    return true;
}

static bool ggml_rpp_create_engine_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_openrt_set_rows *>(rpp_base_node);
    bool ret      = false;

    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst->src[0]);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }

    std::string save_path = STRINGIZE(GGML_RPP_SAVE_ENGINE);
    std::string load_path = STRINGIZE(GGML_RPP_LOAD_ENGINE);
    // load engine
    if (!load_path.empty()) {
        char engine_path[128];
        sprintf(engine_path, "%s/upatch%ld_%s_%s.engine", load_path.c_str(), rpp_node->n_ubatch, ggml_op_name(dst->op),
                dst->name);
        ret = ggml_rpp_load_enigne(rpp_node, engine_path);
    }
    // create engine
    if (!ret) {
        ret = ggml_rpp_create_engine_set_rows(ctx, rpp_node, dst);
        if (ret && !save_path.empty()) {
            char engine_path[128];
            sprintf(engine_path, "%s/upatch%ld_%s_%s.engine", save_path.c_str(), rpp_node->n_ubatch,
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

bool ggml_rpp_op_openrt_set_rows(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_openrt_set_rows * rpp_node = nullptr;
    auto                  iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_set_rows");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (ggml_rpp_node_has_matching_properties(iter_node->first, cur_node)) {
                    rpp_node = (rpp_openrt_set_rows *) cur_node;
                    break;
                }
            }
        }

        // find rpp_node from other graph, because of rpp_node can be shared if have same dims
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_set_rows");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                if (!rpp_graph_tmp) {
                    continue;
                }
                for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter.first == dst) {
                        continue;
                    }
                    if (node_iter.first->op == GGML_OP_SET_ROWS && ggml_rpp_dims_is_same(node_iter.first, dst)) {
                        auto & node_vec = node_iter.second;
                        for (size_t i = 0; i < node_vec.size(); i++) {
                            auto cur_node = node_vec[i].get();
                            if (ggml_rpp_set_rows_dims_is_same(dst, node_iter.first, cur_node)) {
                                auto new_node = std::make_unique<rpp_openrt_set_rows>(dst, cur_node);
                                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                                rpp_node = (rpp_openrt_set_rows *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                                // set io tensor
                                rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
                                rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
                                rpp_node->binding_io_tensors.emplace_back(dst);
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
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_openrt_set_rows");
            auto new_node = std::make_unique<rpp_openrt_set_rows>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_openrt_set_rows *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            if (!(ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ggml_rpp_set_io_bingdings_device(ctx, rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_openrt_set_rows *) (iter->second);
        if (!ctx.rpp_io_buffers.count(dst)) {
            ctx.rpp_io_buffers[dst] = rpp_node->binding_o_buffers[dst];
        }
    }

    // compute add operator
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "execute_openrt_set_rows");
        RPP_EXECUTE_CONTEXT(rpp_node->context, 1, rpp_node->binding_io_buffers.data(), ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }
    return true;
}

#    if 0
bool ggml_rpp_op_openrt_set_rows(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (!src0 || !src1) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->type != GGML_TYPE_I64) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t nc   = src0->ne[0];
    const int64_t nr   = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne1  = dst->ne[1];
    const int64_t ne12 = src1->ne[1];
    const int64_t ne13 = src1->ne[2];

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(dst->ne[2] == ne02);
    GGML_ASSERT(dst->ne[3] == ne03);
    GGML_ASSERT(ne02 % ne12 == 0);
    GGML_ASSERT(ne03 % ne13 == 0);

    const ggml_from_float_t from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        const int64_t i13 = i03 % ne13;
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            const int64_t i12 = i02 % ne12;
            for (int64_t i = 0; i < nr; ++i) {
                const int64_t dst_row_i =
                    *(int64_t *) ((char *) src1->data + i * src1->nb[0] + i12 * src1->nb[1] + i13 * src1->nb[2]);
                if (dst_row_i < 0 || dst_row_i >= ne1) {
                    continue;
                }

                const float * src_row =
                    (const float *) ((char *) src0->data + i * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);

                char * dst_row = (char *) dst->data + dst_row_i * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3];
                from_float(src_row, dst_row, nc);
            }
        }
    }

    return true;
}
#    endif
#endif  // __linux__
