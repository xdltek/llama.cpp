#include "rpp_con2d/rpp_con2d.h"

bool ggml_rpp_can_fuse_con2d(const ggml_cgraph * cgraph,
                             int &               node_idx,
                             ggml_tensor *&      im2col,
                             ggml_tensor *&      mul_mat,
                             ggml_tensor *&      cont) {
    im2col  = nullptr;
    mul_mat = nullptr;
    cont    = nullptr;

    const int idx = node_idx;
    if (cgraph == nullptr || idx < 0 || idx + 9 >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * const im2col_node = cgraph->nodes[idx + 0];
    if (im2col_node == nullptr || im2col_node->op != GGML_OP_IM2COL) {
        return false;
    }

    ggml_tensor * const im2col_reshape        = cgraph->nodes[idx + 1];
    ggml_tensor * const weight_reshape        = cgraph->nodes[idx + 2];
    ggml_tensor * const mul_mat_node          = cgraph->nodes[idx + 3];
    ggml_tensor * const mul_mat_reshape       = cgraph->nodes[idx + 4];
    ggml_tensor * const mul_mat_permute       = cgraph->nodes[idx + 5];
    ggml_tensor * const mul_mat_cont          = cgraph->nodes[idx + 6];
    ggml_tensor * const output_reshape        = cgraph->nodes[idx + 7];
    ggml_tensor * const output_transpose      = cgraph->nodes[idx + 8];
    ggml_tensor * const output_transpose_cont = cgraph->nodes[idx + 9];

    // Gemma vision patch embedding is lowered as:
    // IM2COL -> RESHAPE, weight RESHAPE -> MUL_MAT -> RESHAPE -> PERMUTE -> CONT -> RESHAPE -> TRANSPOSE -> CONT.
    if (im2col_reshape == nullptr || im2col_reshape->op != GGML_OP_RESHAPE || weight_reshape == nullptr ||
        weight_reshape->op != GGML_OP_RESHAPE || mul_mat_node == nullptr || mul_mat_node->op != GGML_OP_MUL_MAT ||
        mul_mat_reshape == nullptr || mul_mat_reshape->op != GGML_OP_RESHAPE || mul_mat_permute == nullptr ||
        mul_mat_permute->op != GGML_OP_PERMUTE || mul_mat_cont == nullptr || mul_mat_cont->op != GGML_OP_CONT ||
        output_reshape == nullptr || output_reshape->op != GGML_OP_RESHAPE || output_transpose == nullptr ||
        output_transpose->op != GGML_OP_TRANSPOSE || output_transpose_cont == nullptr ||
        output_transpose_cont->op != GGML_OP_CONT) {
        return false;
    }

    if (im2col_node->src[0] == nullptr || im2col_node->src[1] == nullptr) {
        return false;
    }

    if (im2col_reshape->src[0] != im2col_node || weight_reshape->src[0] == nullptr) {
        return false;
    }

    const bool mul_src0_is_im2col = mul_mat_node->src[0] == im2col_reshape;
    const bool mul_src1_is_im2col = mul_mat_node->src[1] == im2col_reshape;
    const bool mul_src0_is_weight = mul_mat_node->src[0] == weight_reshape;
    const bool mul_src1_is_weight = mul_mat_node->src[1] == weight_reshape;
    if (!((mul_src0_is_im2col && mul_src1_is_weight) || (mul_src1_is_im2col && mul_src0_is_weight))) {
        return false;
    }

    if (mul_mat_reshape->src[0] != mul_mat_node || mul_mat_permute->src[0] != mul_mat_reshape ||
        mul_mat_cont->src[0] != mul_mat_permute || output_reshape->src[0] != mul_mat_cont ||
        output_transpose->src[0] != output_reshape || output_transpose_cont->src[0] != output_transpose) {
        return false;
    }

    for (int i = 0; i < 9; ++i) {
        const ggml_tensor * node = cgraph->nodes[idx + i];
        if (ggml_node_get_use_count(cgraph, idx + i) != 1 || node->view_src != nullptr ||
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
    }

    im2col   = im2col_node;
    mul_mat  = mul_mat_node;
    cont     = output_transpose_cont;
    node_idx = idx + 9;
    return true;
}

static ggml_tensor * ggml_rpp_fused_con2d_get_weight(const ggml_tensor * im2col, const ggml_tensor * mul_mat) {
    if (im2col == nullptr || mul_mat == nullptr || mul_mat->op != GGML_OP_MUL_MAT) {
        return nullptr;
    }

    for (int i = 0; i < 2; ++i) {
        const ggml_tensor * src = mul_mat->src[i];
        if (src == nullptr || src->op != GGML_OP_RESHAPE || src->src[0] == nullptr) {
            continue;
        }

        if (src->src[0] != im2col) {
            return src->src[0];
        }
    }

    return nullptr;
}

static bool ggml_rpp_fused_con2d_properties_is_same(ggml_backend_rpp_context & ctx,
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
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties.ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties.nb[i]) {
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

static bool ggml_rpp_create_kernel_fused_con2d(ggml_backend_rpp_context & ctx,
                                               ggml_rpp_node *            rpp_base_node,
                                               ggml_tensor *              im2col,
                                               ggml_tensor *              mul_mat,
                                               ggml_tensor *              cont) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_fused_con2d *>(rpp_base_node);
    if (im2col == nullptr || mul_mat == nullptr || cont == nullptr || im2col->op != GGML_OP_IM2COL ||
        mul_mat->op != GGML_OP_MUL_MAT || cont->op != GGML_OP_CONT) {
        GGML_LOG_ERROR("%s: invalid fused_con2d tensors\n", __func__);
        return false;
    }
    GGML_ASSERT(ggml_is_contiguous(im2col->src[1]));
    GGML_ASSERT(ggml_is_contiguous(mul_mat->src[0]));
    GGML_ASSERT(ggml_is_contiguous(mul_mat->src[1]));
    GGML_ASSERT(ggml_is_contiguous(cont));

    ggml_tensor * input  = im2col->src[1];
    ggml_tensor * weight = ggml_rpp_fused_con2d_get_weight(im2col, mul_mat);
    if (input == nullptr || weight == nullptr) {
        GGML_LOG_ERROR("%s: failed to parse fused_con2d input/weight from %s and %s\n", __func__, im2col->name,
                       mul_mat->name);
        return false;
    }

    rpp_node->im2col  = im2col;
    rpp_node->mul_mat = mul_mat;
    rpp_node->input   = input;
    rpp_node->weight  = weight;
    rpp_node->output  = cont;

    rpp_node->params.stride_x   = ggml_get_op_params_i32(im2col, 0);
    rpp_node->params.stride_y   = ggml_get_op_params_i32(im2col, 1);
    rpp_node->params.padding_x  = ggml_get_op_params_i32(im2col, 2);
    rpp_node->params.padding_y  = ggml_get_op_params_i32(im2col, 3);
    rpp_node->params.dilation_x = ggml_get_op_params_i32(im2col, 4);
    rpp_node->params.dilation_y = ggml_get_op_params_i32(im2col, 5);
    rpp_node->params.is_2d      = ggml_get_op_params_i32(im2col, 6) != 0;
    rpp_node->params.input_w    = input->ne[0];
    rpp_node->params.input_h    = input->ne[1];
    rpp_node->params.input_c    = input->ne[2];
    rpp_node->params.input_n    = input->ne[3];
    rpp_node->params.kernel_w   = weight->ne[0];
    rpp_node->params.kernel_h   = weight->ne[1];
    rpp_node->params.kernel_c   = weight->ne[2];
    rpp_node->params.kernel_n   = weight->ne[3];
    rpp_node->params.output_w   = cont->ne[0];
    rpp_node->params.output_h   = cont->ne[1];
    rpp_node->params.output_c   = cont->ne[2];
    rpp_node->params.output_n   = cont->ne[3];

    GGML_ASSERT(rpp_node->params.is_2d);
    GGML_ASSERT(input->data != nullptr);
    GGML_ASSERT(weight->data != nullptr);
    GGML_ASSERT(cont->data != nullptr);

    // kernel inputs: image tensor and patch embedding weight
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) input->data);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) weight->data);
    // kernel outputs: final contiguous tensor after the fused reshape/permute/transpose chain
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) cont->data);
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(input, input->data);
    rpp_node->binding_i_buffers.emplace(weight, weight->data);
    rpp_node->binding_o_buffers.emplace(cont, cont->data);
    rpp_node->binding_io_buffers.emplace_back(input->data);
    rpp_node->binding_io_buffers.emplace_back(weight->data);
    rpp_node->binding_io_buffers.emplace_back(cont->data);

    const int input_type_size  = ggml_rpp_get_io_type_size(ctx, input, 0);
    const int output_type_size = ggml_rpp_get_io_type_size(ctx, cont, 1);
    // build fused_con2d kernel
    // rpp_fused_con2d_build(*(rpp_node->kernel_ctx.get()), rpp_node->params, input, weight, cont,
    //                        rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              im2col,
                                            ggml_tensor *              mul_mat,
                                            ggml_tensor *              cont) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_fused_con2d *>(rpp_base_node);
    bool ret      = false;
    GGML_ASSERT(ctx.use_ubatch == false);
    ret = ggml_rpp_create_kernel_fused_con2d(ctx, rpp_node, im2col, mul_mat, cont);
    GGML_ASSERT(ret);
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, cont);
        ggml_rpp_node_set_properties(rpp_node, rpp_node->im2col);
        ggml_rpp_node_set_properties(rpp_node, rpp_node->mul_mat);
        rpp_node->binding_io_tensors.emplace_back(rpp_node->input);
        rpp_node->binding_io_tensors.emplace_back(rpp_node->weight);
        rpp_node->binding_io_tensors.emplace_back(cont);
    }
    return ret;
}

bool ggml_rpp_op_kernel_fused_con2d(ggml_backend_rpp_context & ctx,
                                    ggml_tensor *              im2col,
                                    ggml_tensor *              mul_mat,
                                    ggml_tensor *              cont,
                                    int                        is_instantial,
                                    int                        is_launch) {
    if (!im2col || !mul_mat || !cont) {
        GGML_LOG_ERROR("%s: fused_con2d tensor is nullptr\n", __func__);
        return false;
    }
    rpp_kernel_fused_con2d * rpp_node = nullptr;
    auto                     iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(cont);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(cont);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_fused_con2d");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[cont];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    cur_node->op == ggml_rpp_node::RPP_OP_FUSED_CON2D &&
                    ggml_rpp_fused_con2d_properties_is_same(ctx, cont, cur_node)) {
                    rpp_node = (rpp_kernel_fused_con2d *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_fused_con2d");
            auto new_node = std::make_unique<rpp_kernel_fused_con2d>(cont);
            ctx.cur_rpp_graph->rpp_nodes[cont].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_fused_con2d *) (ctx.cur_rpp_graph->rpp_nodes[cont].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, im2col, mul_mat, cont))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[cont] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_fused_con2d *) (iter->second);
    }

    if (is_launch) {
        // compute fused_con2d operator
        if (rpp_node->kernel_ctx->graphexec == nullptr) {
            return true;
        }
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_fused_con2d");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, cont->name, ggml_op_name(cont->op),
                           e.what());
        }
    }
    return true;
}
