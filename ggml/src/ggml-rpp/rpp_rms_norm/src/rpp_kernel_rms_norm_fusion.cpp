#include "rpp_rms_norm/rpp_rms_norm.h"
#include "rpp_rms_norm/src/rpp_kernel_build.h"

static int ggml_rpp_rms_norm_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static ggml_rpp_node * ggml_rpp_find_rms_norm_node(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_RMS_NORM) {
                auto & node_vec = node_iter.second;
                for (size_t i = 0; i < node_vec.size(); i++) {
                    auto cur_node = node_vec[i].get();
                    return cur_node;
                }
            }
        }
    }
    return nullptr;
}

static bool ggml_rpp_rms_norm_properties_is_same(ggml_backend_rpp_context & ctx,
                                                 ggml_tensor *              dst,
                                                 ggml_rpp_node *            rpp_node) {
    GGML_ASSERT(rpp_node);
    if (dst != rpp_node->cur_ggml_tensor) {
        return false;
    }
    if (!rpp_node->ggml_node_properties.size()) {
        return false;
    }
    if (!rpp_node->ggml_node_properties.count(dst)) {
        return false;
    }

    auto & node                  = dst;
    auto & graph_node_properties = rpp_node->ggml_node_properties[node];
    if (node->data != graph_node_properties.node_address && node->op != GGML_OP_CPY && node->op != GGML_OP_VIEW) {
        return false;
    }
    if (node->op != graph_node_properties.node_op) {
        return false;
    }
    if (rpp_node->n_ubatch == 1) {
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (node->ne[i] != graph_node_properties.ne[i]) {
                return false;
            }
            if (node->nb[i] != graph_node_properties.nb[i]) {
                return false;
            }
        }
    } else {
        if (dst->ne[rpp_node->seq_len_index] == 1) {
            return false;
        }
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] && node->src[i]->data != graph_node_properties.src_address[i] && node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW) {
            return false;
        }
    }
    if (node->op == GGML_OP_SCALE &&
        memcmp(graph_node_properties.op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }
    // for (auto item : rpp_node->binding_i_buffers) {
    //     if (ctx.rpp_io_buffers.count(item.first)) {
    //         void * new_buffer = ctx.rpp_io_buffers[item.first];
    //         void * old_buffer = rpp_node->binding_i_buffers[item.first];
    //         if (new_buffer != old_buffer) {
    //             return false;
    //         }
    //     }
    // }
    return true;
}

static bool ggml_rpp_create_kernel_rms_norm_mul_fusion(ggml_backend_rpp_context & ctx,
                                                       ggml_rpp_node *            rpp_base_node,
                                                       ggml_tensor *              dst,
                                                       ggml_tensor *              mul_tensor) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_rms_norm_mul_fusion *>(rpp_base_node);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    float epsilon = 1e-6f;
    if (dst->op_params[0] != 0) {
        epsilon = *reinterpret_cast<float *>(&dst->op_params[0]);
    }

    const int seq_len = ggml_rpp_rms_norm_seq_len(dst, rpp_node);
    // need to merge rows for rms_norm kernel
    int       rows    = 1;
    for (int i = GGML_MAX_DIMS - 1; i > 0; i--) {
        int cur_dim = dst->ne[i];
        if (i == rpp_node->seq_len_index && ctx.use_ubatch && seq_len > 1) {
            cur_dim = rpp_node->n_ubatch;
        }
        rows *= cur_dim;
    }
    const int     cols    = dst->ne[0];
    ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        return false;
    }

    const int i_type_size    = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size    = ggml_rpp_get_io_type_size(ctx, mul_tensor, 1);
    const int mul_i_type_size = ggml_rpp_get_io_type_size(ctx, mul_src, 0);

    RPPdeviceptr mul_src_dev = (RPPdeviceptr) (mul_src->data);
    if (mul_i_type_size != i_type_size) {
        const size_t cols_elems = (size_t) cols;
        if (mul_i_type_size == (int) sizeof(float) && i_type_size == (int) sizeof(ggml_bf16_t)) {
            ggml_rpp_pool_alloc<float>      mul_host_f32(ctx.pool_host_leg(), cols_elems);
            ggml_rpp_pool_alloc<ggml_bf16_t> mul_host_bf16(ctx.pool_host_leg(), cols_elems);
            RPP_CHECK(rtMemcpy(mul_host_f32.ptr, mul_src->data, cols_elems * sizeof(float), rtMemcpyDeviceToHost));
            ggml_rpp_fp32_to_bf16(mul_host_bf16.ptr, mul_host_f32.ptr, cols_elems);
            void * mul_dev_bf16 = ctx.pool().alloc(cols_elems * sizeof(ggml_bf16_t));
            RPP_CHECK(rtMemcpy(mul_dev_bf16, mul_host_bf16.ptr, cols_elems * sizeof(ggml_bf16_t), rtMemcpyHostToDevice));
            mul_src_dev = (RPPdeviceptr) mul_dev_bf16;
            rpp_node->pool_buffers.emplace(mul_dev_bf16);
        } else if (mul_i_type_size == (int) sizeof(ggml_bf16_t) && i_type_size == (int) sizeof(float)) {
            ggml_rpp_pool_alloc<ggml_bf16_t> mul_host_bf16(ctx.pool_host_leg(), cols_elems);
            ggml_rpp_pool_alloc<float>      mul_host_f32(ctx.pool_host_leg(), cols_elems);
            RPP_CHECK(
                rtMemcpy(mul_host_bf16.ptr, mul_src->data, cols_elems * sizeof(ggml_bf16_t), rtMemcpyDeviceToHost));
            ggml_rpp_bf16_to_fp32(mul_host_bf16.ptr, mul_host_f32.ptr, cols_elems);
            void * mul_dev_f32 = ctx.pool().alloc(cols_elems * sizeof(float));
            RPP_CHECK(rtMemcpy(mul_dev_f32, mul_host_f32.ptr, cols_elems * sizeof(float), rtMemcpyHostToDevice));
            mul_src_dev = (RPPdeviceptr) mul_dev_f32;
            rpp_node->pool_buffers.emplace(mul_dev_f32);
        } else {
            GGML_LOG_ERROR("%s: unsupported rms_norm mul fusion type mix, rms_in=%d mul_in=%d\n", __func__, i_type_size,
                           mul_i_type_size);
            return false;
        }
    }

    // kernel inputs
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (dst->src[0]->data));
    rpp_node->kernel_ctx->dev_in.emplace_back(mul_src_dev);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (mul_tensor->data));
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_i_buffers.emplace(mul_src, (void *) mul_src_dev);
    rpp_node->binding_o_buffers.emplace(mul_tensor, mul_tensor->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back((void *) mul_src_dev);
    rpp_node->binding_io_buffers.emplace_back(mul_tensor->data);
    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_rms_norm_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_rms_norm_mul_fusion *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    int       mode        = 1;
    // build rms_norm kernel
    rpp_rmsnorm_build(*(rpp_node->kernel_ctx.get()), rows, cols, epsilon, i_type_size, o_type_size, mode,
                      rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch_rms_norm_mul_fusion(ggml_backend_rpp_context & ctx,
                                                                ggml_rpp_node *            rpp_base_node,
                                                                ggml_tensor *              dst,
                                                                ggml_tensor *              mul_tensor) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_rms_norm_mul_fusion *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_rms_norm_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }
    ret = ggml_rpp_create_kernel_rms_norm_mul_fusion(ctx, rpp_node, dst, mul_tensor);
    GGML_ASSERT(ret);
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

bool ggml_rpp_op_kernel_rms_norm_mul_fusion(ggml_backend_rpp_context & ctx,
                                            ggml_tensor *              dst,
                                            ggml_tensor *              mul_tensor,
                                            int                        is_instantial,
                                            int                        is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_rms_norm_mul_fusion * rpp_node = nullptr;
    auto                             iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_rms_norm_mul_fusion");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_rms_norm_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_rms_norm_mul_fusion *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_rms_norm_mul_fusion");
            auto new_node = std::make_unique<rpp_kernel_rms_norm_mul_fusion>(dst, mul_tensor);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_kernel_rms_norm_mul_fusion *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch_rms_norm_mul_fusion(ctx, rpp_node, dst, mul_tensor))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_rms_norm_mul_fusion *) (iter->second);
    }

    if (is_launch) {
        // compute rms_norm_mul_fusion operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_rms_norm_mul_fusion");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return true;
}
