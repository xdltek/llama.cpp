#include "rpp_reduce_sum/rpp_reduce_sum.h"
#include "rpp_reduce_sum/src/rpp_kernel_build.h"

static int ggml_rpp_reduce_sum_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_create_kernel_reduce_sum(ggml_backend_rpp_context & ctx,
                                              ggml_rpp_node *            rpp_base_node,
                                              ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_reduce_sum *>(rpp_base_node);

    std::vector<ggml_tensor *> src_tensors;
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        if (dst->src[i]->op != GGML_OP_VIEW) {
            GGML_LOG_ERROR("%s: add_fusion src type is: %s, inputs i: %d\n", __func__,
                           ggml_type_name(dst->src[i]->type), i);
            return false;
        }
        if (i > 0 && dst->src[i - 1]->view_src != dst->src[i]->view_src) {
            GGML_LOG_ERROR("%s: add_fusion src[%d]->view_src != src[%d]->view_src\n", __func__, i - 1, i);
            return false;
        }
        src_tensors.emplace_back(dst->src[i]);
    }
    if (src_tensors.size() < 2) {
        GGML_LOG_ERROR("%s: add_fusion expects >=2 inputs, got %zu for %s\n", __func__, src_tensors.size(), dst->name);
        return false;
    }

    ggml_tensor * view_src = dst->src[0]->view_src;
    const int     seq_len  = ggml_rpp_reduce_sum_seq_len(dst, rpp_node);
    const int     C        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : view_src->ne[2];
    const int     H        = view_src->ne[1];
    const int     W        = view_src->ne[0];
    GGML_ASSERT(seq_len == view_src->ne[2]);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (view_src->data));
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        rpp_node->binding_i_buffers.emplace(dst->src[i], dst->src[i]->data);
        rpp_node->binding_io_buffers.emplace_back(dst->src[i]->data);
    }
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // axis=1: input CxHxW -> output Cx1xW (sum over H)
    const int axis        = 1;
    const int i_type_size = ggml_rpp_get_io_type_size(ctx, view_src, 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    rpp_reduce_sum_build(*(rpp_node->kernel_ctx.get()), C, H, W, axis, i_type_size, o_type_size,
                         rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_engine_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_reduce_sum *>(rpp_base_node);
    bool ret      = false;

    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    GGML_ASSERT(rpp_node->seq_len_index == 1);
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_reduce_sum_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    ret = ggml_rpp_create_kernel_reduce_sum(ctx, rpp_node, dst);
    GGML_ASSERT(ret);

    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        for (int i = 0; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
            rpp_node->binding_io_tensors.emplace_back(dst->src[i]);
        }
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

static bool ggml_rpp_reduce_sum_properties_is_same(ggml_backend_rpp_context & ctx,
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

bool ggml_rpp_op_kernel_reduce_sum(ggml_backend_rpp_context & ctx,
                                   ggml_tensor *              dst,
                                   int                        is_instantial,
                                   int                        is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_reduce_sum * rpp_node = nullptr;
    auto                    iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_reduce_sum");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_reduce_sum_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_reduce_sum *) cur_node;
                    break;
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_reduce_sum");
            auto new_node = std::make_unique<rpp_kernel_reduce_sum>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_reduce_sum *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_reduce_sum *) (iter->second);
    }

    if (is_launch) {
        // compute reduce_sum operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_reduce_sum");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return true;
}
