#include "rpp_clamp/rpp_clamp.h"
#include "rpp_clamp/src/rpp_kernel_build.h"

#include <algorithm>
#include <cstring>

static bool ggml_rpp_clamp_properties_is_same(ggml_backend_rpp_context & ctx,
                                              ggml_tensor *              dst,
                                              ggml_rpp_node *            rpp_node) {
    GGML_UNUSED(ctx);
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
    if (node->op == GGML_OP_CLAMP &&
        memcmp(graph_node_properties.op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

static bool ggml_rpp_create_kernel_clamp(ggml_backend_rpp_context & ctx,
                                         ggml_rpp_node *            rpp_base_node,
                                         ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_clamp *>(rpp_base_node);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst));

    float min_v = 0.0f;
    float max_v = 0.0f;
    memcpy(&min_v, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_v, (float *) dst->op_params + 1, sizeof(float));

    const int seq_len = dst->ne[rpp_node->seq_len_index];
    const int C = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 2) ? rpp_node->n_ubatch : dst->ne[2];
    const int H = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 1) ? rpp_node->n_ubatch : dst->ne[1];
    const int W = dst->ne[0];

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (dst->src[0]->data));
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    kernel_clamp::rpp_clamp_build(*(rpp_node->kernel_ctx.get()), C, H, W, min_v, max_v, i_type_size, o_type_size,
                                  rpp_node->is_instantial);

    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_clamp *>(rpp_base_node);

    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    if (ctx.use_ubatch && dst->ne[rpp_node->seq_len_index] > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    const bool ret = ggml_rpp_create_kernel_clamp(ctx, rpp_node, dst);
    GGML_ASSERT(ret);

    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }

    return ret;
}

bool ggml_rpp_op_kernel_clamp(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr\n", __func__);
        return false;
    }

    rpp_kernel_clamp * rpp_node = nullptr;
    auto               iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_clamp_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_clamp *) cur_node;
                    break;
                }
            }
        }

        if (!rpp_node) {
            auto new_node = std::make_unique<rpp_kernel_clamp>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_clamp *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_clamp *) (iter->second);
    }

    if (is_launch) {
        try {
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
            return false;
        }
    }

    return true;
}
