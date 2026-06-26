#include "rpp_soft_max/rpp_soft_max.h"

#if GGML_RPP_USE_RT

static int ggml_rpp_soft_max_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_create_engine_soft_max_ubatch(ggml_backend_rpp_context & ctx,
                                                   ggml_rpp_node *            base_rpp_node,
                                                   ggml_tensor *              dst) {
    GGML_ASSERT(base_rpp_node);
    auto          rpp_node                     = (rpp_openrt_soft_max *) base_rpp_node;
    auto          rpp_network                  = rpp_node->network.get();
    // get n dims
    ggml_tensor * cur_tensor[GGML_MAX_SRC + 1] = { dst };
    auto          end_iter                     = std::copy_if(dst->src, dst->src + GGML_MAX_SRC, cur_tensor + 1,
                                                              [](ggml_tensor * ptr) { return ptr != nullptr; });
    void *        out_buffer                   = nullptr;
    size_t        out_size                     = 0;
    auto          rpp_graph                    = ctx.cur_rpp_graph;

    //data preprocessed
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    assert(ggml_is_contiguous(dst));
    assert(ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    // const int ith = params->ith;

    infer1::DataType ort_dtype = ggml_rpp_dtype_mapping(src0->type);
    infer1::Dims     dims      = ggml_rpp_dims_mapping(src0);

    // scale weights
    int nb_dims     = GGML_MAX_DIMS;
    dims.nbDims     = nb_dims;
    dims.d[2]       = ctx.n_ubatch;
    int element_num = 1;
    for (int i = 0; i < nb_dims; i++) {
        element_num *= dims.d[i];
        dims.type[i] = (infer1::DimensionType) 0;
    }

    // create inputs
    infer1::ITensor * inputs[2] = { nullptr, nullptr };
    for (size_t i = 0; i < 2; i++) {
        inputs[i] = ggml_rpp_create_input_tensor(dst->src[0], rpp_node, dims);
        if (!inputs[i]) {
            GGML_LOG_ERROR("%s: creat input%ld failed for add, %s (%s)\n", __func__, i, dst->name,
                           ggml_op_name(dst->op));
            return false;
        }
    }

    std::vector<float> scale_values(element_num, scale);
    infer1::Weights    weights;
    weights.type                   = ort_dtype;
    weights.values                 = scale_values.data();
    weights.count                  = element_num;
    auto              constant     = rpp_node->network->addConstant(dims, weights);
    infer1::ITensor * scale_tensor = constant->getOutput(0);
    auto opt_scale_mul = rpp_network->addElementWise(*inputs[0], *scale_tensor, infer1::ElementWiseOperation::kPROD);
    if (!opt_scale_mul) {
        GGML_LOG_ERROR("%s: addElementWise mul faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * scale_out = opt_scale_mul->getOutput(0);
    auto              opt_add = rpp_network->addElementWise(*scale_out, *inputs[1], infer1::ElementWiseOperation::kSUM);
    if (!opt_add) {
        GGML_LOG_ERROR("%s: addElementWise mul faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * add_out = opt_add->getOutput(0);

    auto opt_soft_max = rpp_network->addSoftMax(*add_out);
    if (!opt_soft_max) {
        GGML_LOG_ERROR("%s: addElementWise softmax faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    int axis = 3;
    opt_soft_max->setAxes(1 << axis);
    infer1::ITensor * ort_tensor = opt_soft_max->getOutput(0);

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
        GGML_LOG_ERROR("%s: create execution context failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_soft_max(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            base_rpp_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(base_rpp_node);
    auto          rpp_node                     = (rpp_openrt_soft_max *) base_rpp_node;
    auto          rpp_network                  = rpp_node->network.get();
    // get n dims
    ggml_tensor * cur_tensor[GGML_MAX_SRC + 1] = { dst };
    auto          end_iter                     = std::copy_if(dst->src, dst->src + GGML_MAX_SRC, cur_tensor + 1,
                                                              [](ggml_tensor * ptr) { return ptr != nullptr; });
    void *        out_buffer                   = nullptr;
    size_t        out_size                     = 0;
    auto          rpp_graph                    = ctx.cur_rpp_graph;
    // ggml_rpp_node * rpp_node   = nullptr;

    //data preprocessed
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    assert(ggml_is_contiguous(dst));
    assert(ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));
    // const int ith = params->ith;

    infer1::DataType ort_dtype = ggml_rpp_dtype_mapping(src0->type);
    infer1::Dims     dims      = ggml_rpp_dims_mapping(src0);

    // scale weights
    int nb_dims     = GGML_MAX_DIMS;
    dims.nbDims     = nb_dims;
    int element_num = 1;
    for (int i = 0; i < nb_dims; i++) {
        element_num *= dims.d[i];
        dims.type[i] = (infer1::DimensionType) 0;
    }

    // create inputs
    infer1::ITensor * inputs[2] = { nullptr, nullptr };
    for (size_t i = 0; i < 2; i++) {
        inputs[i] = ggml_rpp_create_input_tensor(dst->src[0], rpp_node);
        if (!inputs[i]) {
            GGML_LOG_ERROR("%s: creat input%ld failed for add, %s (%s)\n", __func__, i, dst->name,
                           ggml_op_name(dst->op));
            return false;
        }
    }

    std::vector<float> scale_values(element_num, scale);
    infer1::Weights    weights;
    weights.type                   = ort_dtype;
    weights.values                 = scale_values.data();
    weights.count                  = element_num;
    auto              constant     = rpp_node->network->addConstant(dims, weights);
    infer1::ITensor * scale_tensor = constant->getOutput(0);
    auto opt_scale_mul = rpp_network->addElementWise(*inputs[0], *scale_tensor, infer1::ElementWiseOperation::kPROD);
    if (!opt_scale_mul) {
        GGML_LOG_ERROR("%s: addElementWise mul faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * scale_out = opt_scale_mul->getOutput(0);
    auto              opt_add = rpp_network->addElementWise(*scale_out, *inputs[1], infer1::ElementWiseOperation::kSUM);
    if (!opt_add) {
        GGML_LOG_ERROR("%s: addElementWise mul faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * add_out = opt_add->getOutput(0);

    auto opt_soft_max = rpp_network->addSoftMax(*add_out);
    if (!opt_soft_max) {
        GGML_LOG_ERROR("%s: addElementWise softmax faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    int axis = 3;
    opt_soft_max->setAxes(1 << axis);
    infer1::ITensor * ort_tensor = opt_soft_max->getOutput(0);

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
        GGML_LOG_ERROR("%s: create execution context failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_soft_max_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
    bool dims_is_same = true;
    // beacuse one inputs is const value, so the number of inputs is 1, add 1 outputs = 2
    if (rpp_node->binding_io_tensors.size() == 2) {
        dims_is_same = false;
    }
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

static bool ggml_rpp_create_engine_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            base_rpp_node,
                                            ggml_tensor *              dst) {
    auto rpp_node = (rpp_openrt_soft_max *) base_rpp_node;
    GGML_ASSERT(rpp_node);
    bool ret = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_soft_max_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
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
        if (ctx.use_ubatch && ggml_rpp_soft_max_seq_len(dst, rpp_node) > 1) {
            ret = ggml_rpp_create_engine_soft_max_ubatch(ctx, rpp_node, dst);
        } else {
            ret = ggml_rpp_create_engine_soft_max(ctx, rpp_node, dst);
        }
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
        rpp_node->init_softmax_input_tensor(dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(rpp_node->input1.get());
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return true;
}

static void ggml_cpu_softmax_get_input1_data(ggml_rpp_node * base_rpp_node) {
    auto          rpp_node = (rpp_openrt_soft_max *) base_rpp_node;
    ggml_tensor * dst      = rpp_node->cur_ggml_tensor;
    ggml_tensor * src0     = dst->src[0];
    ggml_tensor * src1     = dst->src[1];
    infer1::Dims  dims     = rpp_node->engine->getBindingDimensions(0);
    size_t        io_size  = ggml_rpp_nbytes(dims, ggml_type_size(src0->type));

    int element_num = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        element_num *= dims.d[i];
    }
    void * src1_data = rpp_node->input1_data;
    RPP_CHECK(rtMemcpy(src1_data, src1->data, io_size, rtMemcpyDeviceToHost));
    // att
    std::vector<float> att_vec(element_num, -INFINITY);
    float *            mp_f32 = nullptr;
    size_t             offset = 0;
    for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
        for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += 1) {
            mp_f32 = src1 ? (float *) ((char *) src1_data + i01 * src1->nb[1]) : NULL;
            // att_vec.insert(att_vec.end(), mp_f32, mp_f32 + 256);
            std::copy(mp_f32, mp_f32 + src0->ne[0], att_vec.begin() + offset);
            offset += src0->ne[0];
        }
    }
    RPP_CHECK(rtMemcpy(rpp_node->input1->data, att_vec.data(), io_size, rtMemcpyHostToDevice));
}

static bool ggml_rpp_set_io_datas_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_node) {
    GGML_ASSERT(rpp_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_set_io_datas");
    if (!rpp_node->ori_rpp_node) {
        ggml_cpu_softmax_get_input1_data(rpp_node);
    }
}

static bool ggml_rpp_set_io_bingdings_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node) {
    GGML_ASSERT(base_rpp_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_set_rows_bingdings");
    // clear buffers
    rpp_openrt_soft_max * rpp_node = (rpp_openrt_soft_max *) base_rpp_node;
    // clear buffers
    rpp_node->binding_io_buffers.clear();
    if (!rpp_node->ori_rpp_node) {
        infer1::Dims dims    = rpp_node->engine->getBindingDimensions(0);
        size_t       io_size = ggml_rpp_nbytes(dims, ggml_type_size(rpp_node->cur_ggml_tensor->type));
        ggml_cpu_softmax_get_input1_data(rpp_node);
    }
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

bool ggml_rpp_op_openrt_soft_max(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_openrt_soft_max * rpp_node = nullptr;
    auto                  iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_soft_max");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                // use batch node, add inputs padding 0
                if (ctx.use_ubatch && ggml_rpp_soft_max_seq_len(dst, cur_node) > 1) {
                    if (cur_node->n_ubatch == ctx.n_ubatch) {
                        rpp_node = (rpp_openrt_soft_max *) cur_node;
                        break;
                    }
                } else {
                    if (ggml_rpp_node_has_matching_properties(iter_node->first, cur_node)) {
                        rpp_node = (rpp_openrt_soft_max *) cur_node;
                        break;
                    }
                }
            }
        }

        // find rpp_node from other graph, because of rpp_node can be shared if have same dims
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_soft_max");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter.first == dst) {
                        continue;
                    }
                    if (node_iter.first->op == GGML_OP_SOFT_MAX && ggml_rpp_dims_is_same(node_iter.first, dst)) {
                        auto & node_vec = node_iter.second;
                        for (size_t i = 0; i < node_vec.size(); i++) {
                            auto cur_node = (rpp_openrt_soft_max *) node_vec[i].get();
                            if (ggml_rpp_soft_max_dims_is_same(dst, node_iter.first, cur_node)) {
                                auto new_node = std::make_unique<rpp_openrt_soft_max>(dst, cur_node);
                                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                                rpp_node = (rpp_openrt_soft_max *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                                // set io tensor
                                rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
                                rpp_node->binding_io_tensors.emplace_back(cur_node->input1.get());
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
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_openrt_soft_max");
            auto new_node = std::make_unique<rpp_openrt_soft_max>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_openrt_soft_max *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            if (!(ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ggml_rpp_set_io_bingdings_device(ctx, rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_openrt_soft_max *) (iter->second);
        ggml_rpp_set_io_datas_device(ctx, rpp_node);
        if (!ctx.rpp_io_buffers.count(dst)) {
            ctx.rpp_io_buffers[dst] = rpp_node->binding_o_buffers[dst];
        }
    }

    // compute add operator
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "execute_openrt_soft_max");
        RPP_EXECUTE_CONTEXT(rpp_node->context, 1, rpp_node->binding_io_buffers.data(), ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }
    return true;
}
#endif  // __linux__
