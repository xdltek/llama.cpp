#include "rpp_expert_routing/rpp_expert_routing.h"
#include "rpp_expert_routing/src/rpp_kernel_expert_routing_build.hpp"

static bool ggml_rpp_expert_routing_node_matches(const ggml_cgraph * cgraph, int node_idx, ggml_op op) {
    return cgraph != nullptr && node_idx >= 0 && node_idx < cgraph->n_nodes && cgraph->nodes[node_idx] != nullptr &&
           cgraph->nodes[node_idx]->op == op;
}

static bool ggml_rpp_validate_router_expert_judgment_fusion(const ggml_rpp_router_expert_judgment_fusion_desc & desc) {
    if (desc.softmax == nullptr || desc.logits == nullptr || desc.argsort == nullptr || desc.ids_view == nullptr ||
        desc.get_rows == nullptr || desc.sum_rows == nullptr || desc.div == nullptr || desc.weights_out == nullptr) {
        return false;
    }

    if (desc.softmax->type != GGML_TYPE_F32 || desc.logits->type != GGML_TYPE_F32 || desc.div->type != GGML_TYPE_F32 ||
        desc.weights_out->type != GGML_TYPE_F32 || desc.ids_view->type != GGML_TYPE_I32) {
        return false;
    }

    if (desc.logits->ne[2] != 1 || desc.logits->ne[3] != 1 || desc.ids_view->ne[2] != 1 || desc.ids_view->ne[3] != 1 ||
        desc.div->ne[2] != 1 || desc.div->ne[3] != 1) {
        return false;
    }

    if (desc.logits->nb[0] != (int64_t) sizeof(float) || desc.div->nb[0] != (int64_t) sizeof(float) ||
        desc.ids_view->nb[0] != (int64_t) sizeof(int32_t)) {
        return false;
    }

    if (desc.logits->ne[1] != desc.ids_view->ne[1] || desc.ids_view->ne[1] != desc.div->ne[1]) {
        return false;
    }

    if (desc.div->ne[0] != desc.ids_view->ne[0]) {
        return false;
    }

    const int64_t n_experts     = desc.logits->ne[0];
    const int64_t n_expert_used = desc.div->ne[0];

    if (n_experts <= 0 || n_expert_used <= 0 || n_expert_used > n_experts) {
        return false;
    }

    if ((n_experts & (n_experts - 1)) != 0 || n_experts > 512) {
        return false;
    }

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale, (const float *) desc.softmax->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) desc.softmax->op_params + 1, sizeof(float));

    if (scale != 1.0f || max_bias != 0.0f || desc.softmax->src[1] != nullptr || desc.softmax->src[2] != nullptr) {
        return false;
    }

    const enum ggml_sort_order order = (enum ggml_sort_order) desc.argsort->op_params[0];
    if (order != GGML_SORT_ORDER_DESC) {
        return false;
    }

    if (desc.weights_out != desc.div && desc.weights_out->data != desc.div->data) {
        return false;
    }

    for (ggml_tensor * t : desc.compute_nodes_to_skip) {
        if (t && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
    }

    return true;
}

static bool ggml_rpp_match_router_expert_judgment_fusion(const ggml_cgraph *                           cgraph,
                                                         int                                           softmax_idx,
                                                         ggml_rpp_router_expert_judgment_fusion_desc & out) {
    if (!ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx, GGML_OP_SOFT_MAX) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 1, GGML_OP_ARGSORT) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 2, GGML_OP_VIEW)) {
        return false;
    }

    ggml_tensor * softmax  = cgraph->nodes[softmax_idx + 0];
    ggml_tensor * argsort  = cgraph->nodes[softmax_idx + 1];
    ggml_tensor * ids_view = cgraph->nodes[softmax_idx + 2];

    if (argsort->src[0] != softmax || ids_view->src[0] != argsort) {
        return false;
    }

    int get_rows_idx = -1;
    for (int i = softmax_idx + 3; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_GET_ROWS && node->src[1] == ids_view) {
            get_rows_idx = i;
            break;
        }
    }
    if (get_rows_idx < 0 || !ggml_rpp_expert_routing_node_matches(cgraph, get_rows_idx - 1, GGML_OP_RESHAPE)) {
        return false;
    }

    ggml_tensor * probs_reshape = cgraph->nodes[get_rows_idx - 1];
    ggml_tensor * get_rows      = cgraph->nodes[get_rows_idx + 0];
    if (probs_reshape->src[0] != softmax || get_rows->src[0] != probs_reshape || get_rows->src[1] != ids_view) {
        return false;
    }

    int           cursor           = get_rows_idx + 1;
    ggml_tensor * numerator        = get_rows;
    ggml_tensor * get_rows_reshape = nullptr;
    if (ggml_rpp_expert_routing_node_matches(cgraph, cursor, GGML_OP_RESHAPE) &&
        cgraph->nodes[cursor]->src[0] == get_rows) {
        get_rows_reshape = cgraph->nodes[cursor];
        numerator        = get_rows_reshape;
        ++cursor;
    }

    if (!ggml_rpp_expert_routing_node_matches(cgraph, cursor, GGML_OP_SUM_ROWS) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, cursor + 1, GGML_OP_DIV)) {
        return false;
    }

    ggml_tensor * sum_rows = cgraph->nodes[cursor + 0];
    ggml_tensor * div      = cgraph->nodes[cursor + 1];
    if (sum_rows->src[0] != numerator || div->src[0] != numerator || div->src[1] != sum_rows) {
        return false;
    }

    ggml_tensor * weights_out = div;
    if (ggml_rpp_expert_routing_node_matches(cgraph, cursor + 2, GGML_OP_RESHAPE) &&
        cgraph->nodes[cursor + 2]->src[0] == div) {
        weights_out = cgraph->nodes[cursor + 2];
    }

    out                  = {};
    out.softmax          = softmax;
    out.logits           = softmax->src[0];
    out.argsort          = argsort;
    out.ids_view         = ids_view;
    out.probs_reshape    = probs_reshape;
    out.get_rows         = get_rows;
    out.get_rows_reshape = get_rows_reshape;
    out.sum_rows         = sum_rows;
    out.div              = div;
    out.weights_out      = weights_out;

    out.compute_nodes_to_skip.emplace(argsort);
    out.compute_nodes_to_skip.emplace(get_rows);
    out.compute_nodes_to_skip.emplace(sum_rows);
    out.compute_nodes_to_skip.emplace(div);

    return ggml_rpp_validate_router_expert_judgment_fusion(out);
}

static bool ggml_rpp_match_router_expert_judgment_fusion_with_clamp(const ggml_cgraph * cgraph,
                                                                    int                 softmax_idx,
                                                                    ggml_rpp_router_expert_judgment_fusion_desc & out) {
    if (!ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 0, GGML_OP_SOFT_MAX) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 1, GGML_OP_RESHAPE) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 2, GGML_OP_ARGSORT) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 3, GGML_OP_VIEW) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 4, GGML_OP_GET_ROWS) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 5, GGML_OP_RESHAPE) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 6, GGML_OP_SUM_ROWS) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 7, GGML_OP_CLAMP) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 8, GGML_OP_DIV) ||
        !ggml_rpp_expert_routing_node_matches(cgraph, softmax_idx + 9, GGML_OP_RESHAPE)) {
        return false;
    }

    ggml_tensor * softmax          = cgraph->nodes[softmax_idx + 0];
    ggml_tensor * probs_reshape    = cgraph->nodes[softmax_idx + 1];
    ggml_tensor * argsort          = cgraph->nodes[softmax_idx + 2];
    ggml_tensor * ids_view         = cgraph->nodes[softmax_idx + 3];
    ggml_tensor * get_rows         = cgraph->nodes[softmax_idx + 4];
    ggml_tensor * get_rows_reshape = cgraph->nodes[softmax_idx + 5];
    ggml_tensor * sum_rows         = cgraph->nodes[softmax_idx + 6];
    ggml_tensor * clamp            = cgraph->nodes[softmax_idx + 7];
    ggml_tensor * div              = cgraph->nodes[softmax_idx + 8];
    ggml_tensor * weights_out      = cgraph->nodes[softmax_idx + 9];

    if (probs_reshape->src[0] != softmax || (argsort->src[0] != softmax && argsort->src[0] != probs_reshape) ||
        ids_view->src[0] != argsort || get_rows->src[0] != probs_reshape || get_rows->src[1] != ids_view ||
        get_rows_reshape->src[0] != get_rows || sum_rows->src[0] != get_rows_reshape || clamp->src[0] != sum_rows ||
        div->src[0] != get_rows_reshape || div->src[1] != clamp || weights_out->src[0] != div) {
        return false;
    }

    out                  = {};
    out.softmax          = softmax;
    out.logits           = softmax->src[0];
    out.argsort          = argsort;
    out.ids_view         = ids_view;
    out.probs_reshape    = probs_reshape;
    out.get_rows         = get_rows;
    out.get_rows_reshape = get_rows_reshape;
    out.sum_rows         = sum_rows;
    out.div              = div;
    out.weights_out      = weights_out;

    out.compute_nodes_to_skip.emplace(argsort);
    out.compute_nodes_to_skip.emplace(get_rows);
    out.compute_nodes_to_skip.emplace(sum_rows);
    out.compute_nodes_to_skip.emplace(clamp);
    out.compute_nodes_to_skip.emplace(div);

    return ggml_rpp_validate_router_expert_judgment_fusion(out);
}

bool ggml_rpp_is_fuse_expert_routing(const ggml_cgraph *                           cgraph,
                                     int &                                         node_idx,
                                     ggml_rpp_router_expert_judgment_fusion_desc & fusion,
                                     const std::unordered_set<ggml_tensor *> &     runtime_skips) {
    fusion = {};

    if (cgraph == nullptr || node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return false;
    }

    const bool with_clamp = ggml_rpp_expert_routing_node_matches(cgraph, node_idx + 1, GGML_OP_RESHAPE);
    const int  start_idx  = node_idx;
    if (with_clamp) {
        if (!ggml_rpp_match_router_expert_judgment_fusion_with_clamp(cgraph, node_idx, fusion)) {
            return false;
        }
    } else {
        if (!ggml_rpp_match_router_expert_judgment_fusion(cgraph, node_idx, fusion)) {
            return false;
        }
    }

    for (ggml_tensor * t : fusion.compute_nodes_to_skip) {
        if (runtime_skips.count(t) > 0) {
            return false;
        }
    }

    if (with_clamp) {
        node_idx = start_idx + 9;
    }
    return true;
}

static int ggml_rpp_expert_routing_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static ggml_rpp_node * ggml_rpp_find_expert_routing_node(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_SOFT_MAX) {
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

static bool ggml_rpp_expert_routing_properties_is_same(ggml_backend_rpp_context & ctx,
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

static bool ggml_rpp_create_kernel_expert_routing_fusion(ggml_backend_rpp_context &                    ctx,
                                                         ggml_rpp_node *                               rpp_base_node,
                                                         ggml_rpp_router_expert_judgment_fusion_desc & fusion) {
    GGML_ASSERT(rpp_base_node);
    auto          rpp_node = static_cast<rpp_kernel_expert_routing_fusion *>(rpp_base_node);
    ggml_tensor * dst      = fusion.softmax;
    GGML_ASSERT(ggml_is_contiguous(fusion.softmax->src[0]));

    const int seq_len = ggml_rpp_expert_routing_seq_len(dst, rpp_node);

    // kernel inputs
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (dst->src[0]->data));
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (fusion.argsort->data));
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (fusion.div->data));
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(fusion.argsort, fusion.argsort->data);
    rpp_node->binding_o_buffers.emplace(fusion.div, fusion.div->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(fusion.argsort->data);
    rpp_node->binding_io_buffers.emplace_back(fusion.div->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_expert_routing_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_soft_max_node = static_cast<rpp_kernel_expert_routing_fusion *>(ori_rpp_node);
        rpp_node->init_workspace(ori_soft_max_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size  = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size  = ggml_rpp_get_io_type_size(ctx, fusion.div, 1);
    const int num_tokens   = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->ne[1];
    const int n_experts    = fusion.softmax->ne[0];
    const int experts_used = fusion.div->ne[0];
    // build expert routing kernel
    rpp_expert_routing_build(*(rpp_node->kernel_ctx.get()), num_tokens, n_experts, experts_used, i_type_size,
                             o_type_size, rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch_expert_routing_fusion(
    ggml_backend_rpp_context &                    ctx,
    ggml_rpp_node *                               rpp_base_node,
    ggml_rpp_router_expert_judgment_fusion_desc & fusion) {
    GGML_ASSERT(rpp_base_node);
    auto          rpp_node = static_cast<rpp_kernel_expert_routing_fusion *>(rpp_base_node);
    bool          ret      = false;
    ggml_tensor * dst      = fusion.softmax;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[fusion.softmax].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_expert_routing_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }
    ret = ggml_rpp_create_kernel_expert_routing_fusion(ctx, rpp_node, fusion);
    GGML_ASSERT(ret);
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

bool ggml_rpp_op_kernel_expert_routing_fusion(ggml_backend_rpp_context &                    ctx,
                                              ggml_rpp_router_expert_judgment_fusion_desc & fusion,
                                              int                                           is_instantial,
                                              int                                           is_launch) {
    ggml_tensor * dst = fusion.softmax;
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_expert_routing_fusion * rpp_node = nullptr;
    auto                               iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_rms_norm_mul_fusion");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_expert_routing_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_expert_routing_fusion *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_expert_routing_fusion");
            auto new_node = std::make_unique<rpp_kernel_expert_routing_fusion>(dst, fusion);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_kernel_expert_routing_fusion *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch_expert_routing_fusion(ctx, rpp_node, fusion))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_expert_routing_fusion *) (iter->second);
    }

    if (is_launch) {
        // compute add operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_expert_routing_fusion");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return true;
}
