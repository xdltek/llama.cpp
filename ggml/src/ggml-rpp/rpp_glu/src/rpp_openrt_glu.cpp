
#include "rpp_glu/rpp_glu.h"

#if GGML_RPP_USE_RT

static int ggml_rpp_glu_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_glu_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
    bool dims_is_same = true;
    if (rpp_node->ggml_node_properties.count(src) && dims_is_same) {
        auto & property = rpp_node->ggml_node_properties[src];
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (dst->ne[i] != property.ne[i]) {
                dims_is_same = false;
                break;
            }
        }
    } else {
        dims_is_same = false;
    }

    return dims_is_same;
}

static bool ggml_rpp_create_engine_glu_ubatch(ggml_backend_rpp_context & ctx,
                                              ggml_rpp_node *            rpp_base_node,
                                              ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto         rpp_node    = static_cast<rpp_openrt_glu *>(rpp_base_node);
    auto         rpp_network = rpp_node->network.get();
    infer1::Dims dims;
    dims = ggml_rpp_dims_mapping(dst->src[0]);
    for (int i = 0; i < dims.nbDims; i++) {
        dims.d[i] = dst->src[0]->ne[dims.nbDims - i - 1];
    }
    dims[2] = ctx.n_ubatch;
    for (int i = 0; i < infer1::Dims::MAX_DIMS; i++) {
        dims.type[i] = (infer1::DimensionType) 0;
    }

    // create inputs
    infer1::ITensor * inputs[2] = { nullptr, nullptr };
    inputs[0]                   = ggml_rpp_create_input_tensor(dst->src[0], rpp_node, dims);
    if (!inputs[0]) {
        GGML_LOG_ERROR("%s: input0 is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    // src1
    const bool has_src1 = (dst->src[1] != nullptr);
    if (has_src1) {
        inputs[1] = ggml_rpp_create_input_tensor(dst->src[1], rpp_node, dims);
        if (!inputs[1]) {
            GGML_LOG_ERROR("%s: input1 is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
            return false;
        }
    } else {
        inputs[1] = nullptr;
    }

    // -------------------------
    // build SWIGLU:
    // 1) gate = swish(gate_in)
    // 2) out  = gate * value
    // -------------------------

    infer1::ITensor * gate_in  = nullptr;
    infer1::ITensor * value_in = nullptr;
    infer1::ITensor * first    = nullptr;
    infer1::ITensor * second   = nullptr;

    if (has_src1) {
        // two models：gate from src0 or src1？
        // swish(inputs[0]) as gate，inputs[1] as value
        gate_in  = inputs[0];
        value_in = inputs[1];
    } else {
        // one input model：inputs[0] last dimenession is 2*nc
        // It needs to be split in half, and which half is gate/value is determined by the "swapped" parameter in ggml op params.

        // ggml swiglu kernel read op_params_i32(dst, 1)
        const int32_t swapped = ggml_get_op_params_i32(dst, 1);

        // nc = dst->ne[0]；src0->ne[0] should be 2*nc
        const int64_t nc = dst->ne[0];
        GGML_ASSERT(nc > 0);

        const int    split_axis = dims.nbDims - 1;
        // slice0: [0:nc), slice1: [nc:2*nc)
        infer1::Dims start0     = dims;
        infer1::Dims size0      = dims;
        infer1::Dims stride     = dims;
        for (int d = 0; d < dims.nbDims; ++d) {
            start0.d[d] = 0;
            size0.d[d]  = dims.d[d];
            stride.d[d] = 1;
        }
        start0.d[split_axis] = 0;
        size0.d[split_axis]  = (int) nc;

        infer1::Dims start1  = start0;
        infer1::Dims size1   = size0;
        start1.d[split_axis] = (int) nc;

        // build two Slice layer
        auto slice0 = rpp_network->addSlice(*inputs[0], start0, size0, stride);
        if (!slice0) {
            GGML_LOG_ERROR("%s: addSlice(0) failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
            return false;
        }
        auto slice1 = rpp_network->addSlice(*inputs[0], start1, size1, stride);
        if (!slice1) {
            GGML_LOG_ERROR("%s: addSlice(1) failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
            return false;
        }

        first  = slice0->getOutput(0);
        second = slice1->getOutput(0);

        // if (!src1) { src0_p += swapped ? nc : 0; src1_p += swapped ? 0 : nc; }
        // swapped==0: src0=A(first), src1=B(second)  -> silu(A) * B
        // swapped==1: src0=A(second), src1=B(first)  -> A * silu(B)
        // value_in = A, gate_in = B
        if (!swapped) {
            gate_in  = first;
            value_in = second;
        } else {
            value_in = first;
            gate_in  = second;
        }
    }

    // gate：swish(silu)
    auto opt_glu_activation = rpp_network->addActivation(*gate_in, infer1::ActivationType::kSWISH);
    if (!opt_glu_activation) {
        GGML_LOG_ERROR("%s: addActivation failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    infer1::ITensor * kswish_out = opt_glu_activation->getOutput(0);

    // swish(gate) * value
    auto opt_glu_gate = rpp_network->addElementWise(*kswish_out, *value_in, infer1::ElementWiseOperation::kPROD);
    if (!opt_glu_gate) {
        GGML_LOG_ERROR("%s: addElementWise failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    infer1::ITensor * kswich_glu_output = opt_glu_gate->getOutput(0);

    kswich_glu_output->setName(dst->name);
    rpp_network->markOutput(*kswich_glu_output);

    // build engine
    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_glu(ggml_backend_rpp_context & ctx,
                                       ggml_rpp_node *            rpp_base_node,
                                       ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto         rpp_node    = static_cast<rpp_openrt_glu *>(rpp_base_node);
    auto         rpp_network = rpp_node->network.get();
    infer1::Dims dims;
    dims = ggml_rpp_dims_mapping(dst->src[0]);

    // create inputs
    infer1::ITensor * inputs[2] = { nullptr, nullptr };
    inputs[0]                   = ggml_rpp_create_input_tensor(dst->src[0], rpp_node);
    if (!inputs[0]) {
        GGML_LOG_ERROR("%s: input0 is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    // src1
    const bool has_src1 = (dst->src[1] != nullptr);
    if (has_src1) {
        inputs[1] = ggml_rpp_create_input_tensor(dst->src[1], rpp_node);
        if (!inputs[1]) {
            GGML_LOG_ERROR("%s: input1 is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
            return false;
        }
    } else {
        inputs[1] = nullptr;
    }

    // -------------------------
    // build SWIGLU:
    // 1) gate = swish(gate_in)
    // 2) out  = gate * value
    // -------------------------

    infer1::ITensor * gate_in  = nullptr;
    infer1::ITensor * value_in = nullptr;
    infer1::ITensor * first    = nullptr;
    infer1::ITensor * second   = nullptr;

    if (has_src1) {
        // two models：gate from src0 or src1？
        // swish(inputs[0]) as gate，inputs[1] as value
        gate_in  = inputs[0];
        value_in = inputs[1];
    } else {
        // one input model：inputs[0] last dimenession is 2*nc
        // It needs to be split in half, and which half is gate/value is determined by the "swapped" parameter in ggml op params.

        // ggml swiglu kernel read op_params_i32(dst, 1)
        const int32_t swapped = ggml_get_op_params_i32(dst, 1);

        // nc = dst->ne[0]；src0->ne[0] should be 2*nc
        const int64_t nc = dst->ne[0];
        GGML_ASSERT(nc > 0);

        const int    split_axis = dims.nbDims - 1;
        // slice0: [0:nc), slice1: [nc:2*nc)
        infer1::Dims start0     = dims;
        infer1::Dims size0      = dims;
        infer1::Dims stride     = dims;
        for (int d = 0; d < dims.nbDims; ++d) {
            start0.d[d] = 0;
            size0.d[d]  = dims.d[d];
            stride.d[d] = 1;
        }
        start0.d[split_axis] = 0;
        size0.d[split_axis]  = (int) nc;

        infer1::Dims start1  = start0;
        infer1::Dims size1   = size0;
        start1.d[split_axis] = (int) nc;

        // build two Slice layer
        auto slice0 = rpp_network->addSlice(*inputs[0], start0, size0, stride);
        if (!slice0) {
            GGML_LOG_ERROR("%s: addSlice(0) failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
            return false;
        }
        auto slice1 = rpp_network->addSlice(*inputs[0], start1, size1, stride);
        if (!slice1) {
            GGML_LOG_ERROR("%s: addSlice(1) failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
            return false;
        }

        first  = slice0->getOutput(0);
        second = slice1->getOutput(0);

        // if (!src1) { src0_p += swapped ? nc : 0; src1_p += swapped ? 0 : nc; }
        // swapped==0: src0=A(first), src1=B(second)  -> silu(A) * B
        // swapped==1: src0=A(second), src1=B(first)  -> A * silu(B)
        // value_in = A, gate_in = B
        if (!swapped) {
            gate_in  = first;
            value_in = second;
        } else {
            value_in = first;
            gate_in  = second;
        }
    }

    // gate：swish(silu)
    auto opt_glu_activation = rpp_network->addActivation(*gate_in, infer1::ActivationType::kSWISH);
    if (!opt_glu_activation) {
        GGML_LOG_ERROR("%s: addActivation failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    infer1::ITensor * kswish_out = opt_glu_activation->getOutput(0);

    // swish(gate) * value
    auto opt_glu_gate = rpp_network->addElementWise(*kswish_out, *value_in, infer1::ElementWiseOperation::kPROD);
    if (!opt_glu_gate) {
        GGML_LOG_ERROR("%s: addElementWise failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    infer1::ITensor * kswich_glu_output = opt_glu_gate->getOutput(0);

    kswich_glu_output->setName(dst->name);
    rpp_network->markOutput(*kswich_glu_output);

    // build engine
    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
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
    auto rpp_node = static_cast<rpp_openrt_glu *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_glu_seq_len(dst, rpp_node) > 1) {
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
        if (ctx.use_ubatch && ggml_rpp_glu_seq_len(dst, rpp_node) > 1) {
            ret = ggml_rpp_create_engine_glu_ubatch(ctx, rpp_node, dst);
        } else {
            ret = ggml_rpp_create_engine_glu(ctx, rpp_node, dst);
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

static bool ggml_rpp_set_io_bingdings_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_glu_bingdings");
    auto rpp_node = static_cast<rpp_openrt_glu *>(rpp_base_node);
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

bool ggml_rpp_op_openrt_glu(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_openrt_glu * rpp_node = nullptr;
    auto             iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_glu");
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (ctx.use_ubatch && ggml_rpp_glu_seq_len(dst, cur_node) > 1) {
                    if (cur_node->n_ubatch == ctx.n_ubatch) {
                        rpp_node = (rpp_openrt_glu *) cur_node;
                        break;
                    }
                } else {
                    if (ggml_rpp_node_has_matching_properties(iter_node->first, cur_node)) {
                        rpp_node = (rpp_openrt_glu *) cur_node;
                        break;
                    }
                }
            }
        }

        // find rpp_node from other graph, because of rpp_node can be shared if have same dims
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_glu");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter.first == dst) {
                        continue;
                    }
                    if (node_iter.first->op == GGML_OP_GLU && ggml_rpp_dims_is_same(node_iter.first, dst)) {
                        auto & node_vec = node_iter.second;
                        for (size_t i = 0; i < node_vec.size(); i++) {
                            auto cur_node = node_vec[i].get();
                            if (ggml_rpp_glu_dims_is_same(dst, node_iter.first, cur_node)) {
                                auto new_node = std::make_unique<rpp_openrt_glu>(dst, cur_node);
                                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                                rpp_node = (rpp_openrt_glu *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                                // set io tensor
                                rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
                                if (dst->src[1]) {
                                    rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
                                }
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
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_openrt_glu");
            auto new_node = std::make_unique<rpp_openrt_glu>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_openrt_glu *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            if (!(ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ggml_rpp_set_io_bingdings_device(ctx, rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_openrt_glu *) (iter->second);
        if (!ctx.rpp_io_buffers.count(dst)) {
            ctx.rpp_io_buffers[dst] = rpp_node->binding_o_buffers[dst];
        }
    }

    // compute add operator
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "execute_openrt_glu");
        RPP_EXECUTE_CONTEXT(rpp_node->context, 1, rpp_node->binding_io_buffers.data(), ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }

    return true;
}
#endif  // __linux__
