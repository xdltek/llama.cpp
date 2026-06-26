#ifndef RPP_EXPERT_ROUTING
#define RPP_EXPERT_ROUTING

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

struct ggml_rpp_router_expert_judgment_fusion_desc {
    ggml_tensor * softmax          = nullptr;  // anchor node
    ggml_tensor * logits           = nullptr;  // softmax input
    ggml_tensor * argsort          = nullptr;
    ggml_tensor * ids_view         = nullptr;  // consumed by GET_ROWS and later MUL_MAT_ID
    ggml_tensor * probs_reshape    = nullptr;  // reshape(softmax)
    ggml_tensor * get_rows         = nullptr;
    ggml_tensor * get_rows_reshape = nullptr;  // optional reshape(get_rows)
    ggml_tensor * sum_rows         = nullptr;
    ggml_tensor * div              = nullptr;
    ggml_tensor * weights_out      = nullptr;  // final normalized weights (div or reshape(div))

    // Compute nodes that can be skipped after the fused op runs at the softmax anchor.
    std::unordered_set<ggml_tensor *> compute_nodes_to_skip;
};

bool ggml_rpp_is_fuse_expert_routing(const ggml_cgraph *                           cgraph,
                                     int &                                         node_idx,
                                     ggml_rpp_router_expert_judgment_fusion_desc & fusion,
                                     const std::unordered_set<ggml_tensor *> &     runtime_skips);

bool ggml_rpp_op_kernel_expert_routing_fusion(ggml_backend_rpp_context &                    ctx,
                                              ggml_rpp_router_expert_judgment_fusion_desc & fusion,
                                              int                                           is_instantial = 1,
                                              int                                           is_launch     = 1);

struct rpp_kernel_expert_routing_fusion : public rpp_node_kernel {
    explicit rpp_kernel_expert_routing_fusion(ggml_tensor *                                 tensor,
                                              ggml_rpp_router_expert_judgment_fusion_desc & fusion) :
        rpp_node_kernel(tensor),
        fusion_node(fusion) {
        op = RPP_OP_EXPERT_ROUTING;
    }

    explicit rpp_kernel_expert_routing_fusion(ggml_tensor *                                 tensor,
                                              ggml_rpp_node *                               rpp_node,
                                              ggml_rpp_router_expert_judgment_fusion_desc & fusion) :
        rpp_node_kernel(tensor, rpp_node),
        fusion_node(fusion) {
        op = RPP_OP_EXPERT_ROUTING;
    }

    void init_workspace(ggml_backend_rpp_context & ctx) {
        this->kernel_ctx->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc((65536 * 2) * (int) sizeof(uint16_t)));
    }

    void init_workspace(RPPdeviceptr dev_workspace) { this->kernel_ctx->dev_workspace = dev_workspace; }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_expert_routing_fusion(ctx, fusion_node, is_instantial, is_launch);
    }

    ggml_rpp_router_expert_judgment_fusion_desc fusion_node;
};

inline bool ggml_rpp_op_expert_routing_fusion(ggml_backend_rpp_context &                    ctx,
                                              ggml_rpp_router_expert_judgment_fusion_desc & fusion,
                                              int                                           is_instantial = 1,
                                              int                                           is_launch     = 1) {
    return ggml_rpp_op_kernel_expert_routing_fusion(ctx, fusion, is_instantial, is_launch);
}

inline bool ggml_rpp_can_fuse_expert_routing(const ggml_cgraph *                           cgraph,
                                             int &                                         node_idx,
                                             ggml_rpp_router_expert_judgment_fusion_desc & fusion,
                                             const std::unordered_set<ggml_tensor *> &     runtime_skips) {
    return ggml_rpp_is_fuse_expert_routing(cgraph, node_idx, fusion, runtime_skips);
}

#endif
