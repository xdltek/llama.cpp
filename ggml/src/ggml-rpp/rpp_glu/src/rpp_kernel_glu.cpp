#include "rpp_glu/kernel_geglu_erf/rpp_kernel_build.h"
#include "rpp_glu/kernel_swiglu/rpp_kernel_build.h"
#include "rpp_glu/rpp_glu.h"

static int ggml_rpp_glu_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static ggml_rpp_node * ggml_rpp_find_glu_node(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_GLU) {
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

static bool ggml_rpp_create_kernel_swiglu(ggml_backend_rpp_context & ctx,
                                          ggml_rpp_node *            rpp_base_node,
                                          ggml_tensor *              dst,
                                          int                        use_sram_direct = 0) {
    GGML_ASSERT(rpp_base_node);
    auto                rpp_node = static_cast<rpp_kernel_glu *>(rpp_base_node);
    const ggml_tensor * src0     = dst->src[0];
    const ggml_tensor * src1     = dst->src[1];
    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    // mode:
    // 0:y = GELU(x)
    // 1:y = x1 * GELU(x0)
    // 2:x split along split axis;y= GELU(x0)*x1 (one-input packed x)
    int mode = 2;
    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
        mode = 1;
    }
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const int     seq_len = ggml_rpp_glu_seq_len(dst, rpp_node);
    const int C = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 2) ? rpp_node->n_ubatch : src0->ne[2];
    const int H = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 1) ? rpp_node->n_ubatch : src0->ne[1];
    const int W = src0->ne[0];
    // const int     C       = src0->ne[2];
    // const int     H       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : src0->ne[1];
    // const int     W       = src0->ne[0];

    // set io buffer info to rpp_node
    void * inputs_0_buf = src0->data;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));
    rpp_node->binding_i_buffers.emplace(dst->src[0], inputs_0_buf);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(dst->data);
    if (src1) {
        void * inputs_1_buf = src1->data;
        rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_1_buf);
        rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_1_buf);
        rpp_node->binding_io_buffers.emplace_back(inputs_1_buf);
    }

    // find first glu_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_glu_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_glu_node = static_cast<rpp_kernel_glu *>(ori_rpp_node);
        rpp_node->init_workspace(ori_glu_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }
    // build kernel
    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (use_sram_direct) {
        kernel_swiglu::silu_sram_io * io = new kernel_swiglu::silu_sram_io;
        kernel_swiglu::silu_prepare_sram_io(*(rpp_node->kernel_ctx.get()), *io, mode, C, H, W, 2, i_type_size,
                                            o_type_size);
        rpp_node->sram_io = (void *) io;
    }
    kernel_swiglu::rpp_silu_build(*(rpp_node->kernel_ctx.get()), mode, C, H, W, 2, i_type_size, o_type_size,
                                  use_sram_direct, rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_geglu_erf(ggml_backend_rpp_context & ctx,
                                             ggml_rpp_node *            rpp_base_node,
                                             ggml_tensor *              dst,
                                             int                        use_sram_direct = 0) {
    GGML_ASSERT(rpp_base_node);
    auto                rpp_node = static_cast<rpp_kernel_glu *>(rpp_base_node);
    const ggml_tensor * src0     = dst->src[0];
    const ggml_tensor * src1     = dst->src[1];
    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    // mode:
    // 0:y = GELU(x)
    // 1:y = x1 * GELU(x0)
    // 2:x split along split axis;y= GELU(x0)*x1 (one-input packed x)
    int mode = 2;
    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
        mode = 1;
    }

    const int seq_len = ggml_rpp_glu_seq_len(dst, rpp_node);
    const int C = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 2) ? rpp_node->n_ubatch : src0->ne[2];
    const int H = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 1) ? rpp_node->n_ubatch : src0->ne[1];
    const int W = src1 ? src0->ne[0] : src0->ne[0] / 2;
    // const int C       = src0->ne[2];
    // const int H       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : src0->ne[1];

    // set io buffer info to rpp_node
    void * inputs_0_buf = src0->data;
    void * inputs_1_buf = src1->data;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));
    rpp_node->binding_i_buffers.emplace(dst->src[0], inputs_0_buf);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(dst->data);
    if (src1) {
        rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_1_buf);
        rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_1_buf);
        rpp_node->binding_io_buffers.emplace_back(inputs_1_buf);
    }

    // find first glu_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_glu_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_glu_node = static_cast<rpp_kernel_glu *>(ori_rpp_node);
        rpp_node->init_workspace(ori_glu_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }
    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build kernel
    kernel_geglu_erf::rpp_gelu_build(*(rpp_node->kernel_ctx.get()), mode, C, H, W, 1, i_type_size, o_type_size,
                                     rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_glu *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // GGML_ASSERT(rpp_node->seq_len_index == 1);
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_glu_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_SWIGLU:
            ret = ggml_rpp_create_kernel_swiglu(ctx, rpp_node, dst);
            break;
        case GGML_GLU_OP_GEGLU_ERF:
            ret = ggml_rpp_create_kernel_geglu_erf(ctx, rpp_node, dst);
            break;
        default:
            ret = false;
    }
    GGML_ASSERT(ret);

    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        if (dst->src[1]) {
            rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
        }
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

static bool ggml_rpp_glu_properties_is_same(ggml_backend_rpp_context & ctx,
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
    return true;
}

static bool ggml_rpp_launch_kernel(ggml_backend_rpp_context & ctx, ggml_tensor * dst, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_glu *>(rpp_base_node);
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_SWIGLU:
            {
                // sram model launch kernel
                if (rpp_node->sram_io) {
                    auto io     = (kernel_swiglu::silu_sram_io *) (rpp_node->sram_io);
                    auto stream = ctx.stream();
                    kernel_swiglu::silu_cdma_d2s_async(io->sram_in0, (RPPdeviceptr) dst->src[0]->data, io->size_in0,
                                                       stream);
                    if (dst->src[1]) {
                        kernel_swiglu::silu_cdma_d2s_async(io->sram_in1, (RPPdeviceptr) dst->src[1]->data, io->size_in1,
                                                           stream);
                    }
                    try {
                        RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, stream);
                    } catch (const std::exception & e) {
                        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name,
                                       ggml_op_name(dst->op), e.what());
                    }
                    kernel_swiglu::silu_cdma_s2d_async((RPPdeviceptr) dst->data, io->sram_out_final, io->size_out,
                                                       stream);
                } else {
                    try {
                        RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
                    } catch (const std::exception & e) {
                        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name,
                                       ggml_op_name(dst->op), e.what());
                    }
                }
            }
            break;
        case GGML_GLU_OP_GEGLU_ERF:
            {
                try {
                    RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
                } catch (const std::exception & e) {
                    GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                                   e.what());
                }
            }
            break;
        default:
            GGML_LOG_ERROR("%s: infer failed, %s (%s), unknown glu op: %s\n", __func__, dst->name,
                           ggml_op_name(dst->op), ggml_glu_op_name(ggml_get_glu_op(dst)));
            return false;
    }
    return true;
}

bool ggml_rpp_op_kernel_glu(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_glu * rpp_node = nullptr;
    auto             iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_glu");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_glu_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_glu *) cur_node;
                    break;
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_glu");
            auto new_node = std::make_unique<rpp_kernel_glu>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_glu *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_glu *) (iter->second);
    }

    bool ret = true;
    if (is_launch) {
        TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_glu");
        ret = ggml_rpp_launch_kernel(ctx, dst, rpp_node);
    }
    return ret;
}
