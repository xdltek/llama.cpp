#include "rpp_glu/kernel_geglu_erf/rpp_kernel_build.h"
#include "rpp_glu/kernel_swiglu/rpp_kernel_build.h"
#include "rpp_unary/rpp_unary.h"

static int ggml_rpp_unary_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static ggml_rpp_node * ggml_rpp_find_unary_node(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_UNARY) {
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

static bool ggml_rpp_create_kernel_gelu(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst,
                                        int                        use_sram_direct = 0) {
    GGML_ASSERT(rpp_base_node);
    auto                rpp_node = static_cast<rpp_kernel_unary *>(rpp_base_node);
    const ggml_tensor * src0     = dst->src[0];
    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    GGML_ASSERT(dst->src[1] == nullptr);

    // mode:
    // 0:y = GELU(x)
    // 1:y = x1 * GELU(x0)
    // 2:x split along split axis;y= GELU(x0)*x1 (one-input packed x)
    GGML_ASSERT(dst->op == GGML_OP_UNARY);
    const auto unary_op = ggml_get_unary_op(dst);
    GGML_ASSERT(unary_op == GGML_UNARY_OP_GELU || unary_op == GGML_UNARY_OP_GELU_ERF);
    const int mode = 0;

    const int seq_len = ggml_rpp_unary_seq_len(dst, rpp_node);
    const int C = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 2) ? rpp_node->n_ubatch : src0->ne[2];
    const int H = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 1) ? rpp_node->n_ubatch : src0->ne[1];
    const int W = src0->ne[0];

    // set io buffer info to rpp_node
    void * inputs_0_buf = src0->data;
    void * inputs_1_buf = nullptr;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));
    rpp_node->binding_i_buffers.emplace(dst->src[0], inputs_0_buf);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(dst->data);
    
    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build kernel, can use geglu kernel for unary glu
    kernel_geglu_erf::rpp_gelu_build(*(rpp_node->kernel_ctx.get()), mode, C, H, W, 1, i_type_size, o_type_size,
                                     rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_unary *>(rpp_base_node);
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
    if (ctx.use_ubatch && ggml_rpp_unary_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
            ret = ggml_rpp_create_kernel_gelu(ctx, rpp_node, dst);
            break;
        default:
            ret = false;
    }
    GGML_ASSERT(ret);

    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

static bool ggml_rpp_unary_properties_is_same(ggml_backend_rpp_context & ctx,
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
    auto rpp_node = static_cast<rpp_kernel_unary *>(rpp_base_node);
    switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
            {
                try {
                    RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
                } catch (const std::exception & e) {
                    GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                                   e.what());
                    return false;
                }
            }
            break;
        default:
            GGML_LOG_ERROR("%s: infer failed, %s (%s), unknown unary op: %s\n", __func__, dst->name,
                           ggml_op_name(dst->op), ggml_unary_op_name(ggml_get_unary_op(dst)));
            return false;
    }
    return true;
}

bool ggml_rpp_op_kernel_unary(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_unary * rpp_node = nullptr;
    auto               iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_unary");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_unary_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_unary *) cur_node;
                    break;
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_unary");
            auto new_node = std::make_unique<rpp_kernel_unary>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_unary *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_unary *) (iter->second);
    }

    bool ret = true;
    if (is_launch) {
        TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_unary");
        ret = ggml_rpp_launch_kernel(ctx, dst, rpp_node);
    }
    return ret;
}
