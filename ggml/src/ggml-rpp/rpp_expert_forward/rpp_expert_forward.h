#ifndef RPP_EXPERT_FORWARD
#define RPP_EXPERT_FORWARD

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"
#include "rpp_expert_forward/kernel/rpp_kernel_build.h"

#include <functional>
#include <unordered_set>

bool ggml_rpp_is_fuse_expert_forward(const ggml_cgraph * cgraph,
                                     int &               node_idx,
                                     ggml_tensor *&      mul_gate,
                                     ggml_tensor *&      mul_up,
                                     ggml_tensor *&      mul_down,
                                     ggml_tensor *&      div,
                                     ggml_tensor *&      add);

struct expert_forward_vxm_sram_io {
    uint32_t     size_act_gate{ 0 };
    uint32_t     size_act_up{ 0 };
    uint32_t     size_weight_gate{ 0 };
    uint32_t     size_weight_up{ 0 };
    uint32_t     size_weight_down{ 0 };
    uint32_t     size_out_gate{ 0 };
    uint32_t     size_out_up{ 0 };
    uint32_t     size_out_glu{ 0 };
    uint32_t     size_out_down{ 0 };
    uint32_t     size_out_div{ 0 };
    uint32_t     size_out_add{ 0 };
    uint32_t     total_sram_bytes{ 0 };
    RPPdeviceptr sram_base{ 0 };
    RPPdeviceptr sram_act_gate{ 0 };
    RPPdeviceptr sram_act_up{ 0 };
    RPPdeviceptr sram_weight_gate{ 0 };
    RPPdeviceptr sram_weight_up{ 0 };
    RPPdeviceptr sram_weight_down{ 0 };
    RPPdeviceptr sram_out_gate{ 0 };
    RPPdeviceptr sram_out_up{ 0 };
    RPPdeviceptr sram_out_glu{ 0 };
    RPPdeviceptr sram_out_down{ 0 };
    RPPdeviceptr sram_out_div{ 0 };
    RPPdeviceptr sram_out_add{ 0 };
};

bool ggml_rpp_op_kernel_export_forward(ggml_backend_rpp_context & ctx,
                                       ggml_tensor *              gate,
                                       ggml_tensor *              up,
                                       ggml_tensor *              down,
                                       ggml_tensor *              div,
                                       ggml_tensor *              add,
                                       int                        is_instantial = 1,
                                       int                        is_launch     = 1);

struct rpp_kernel_export_forward : public rpp_node_kernel {
    explicit rpp_kernel_export_forward(ggml_tensor * gate,
                                       ggml_tensor * up,
                                       ggml_tensor * down,
                                       ggml_tensor * div,
                                       ggml_tensor * add) :
        rpp_node_kernel(gate),
        gate(gate),
        up(up),
        down(down),
        div(div),
        add(add) {
        op = RPP_OP_EXPERT_FORWARD;
    }

    explicit rpp_kernel_export_forward(ggml_tensor *   gate,
                                       ggml_rpp_node * rpp_node,
                                       ggml_tensor *   up,
                                       ggml_tensor *   down,
                                       ggml_tensor *   div,
                                       ggml_tensor *   add) :
        rpp_node_kernel(gate, rpp_node),
        gate(gate),
        up(up),
        down(down),
        div(div),
        add(add) {
        auto ori_node       = static_cast<rpp_kernel_export_forward *>(rpp_node);
        dev_expert_info     = ori_node->dev_expert_info;
        host_expert_info    = ori_node->host_expert_info;
        token_ids_size      = ori_node->token_ids_size;
        expert_counts_size  = ori_node->expert_counts_size;
        expert_offsets_size = ori_node->expert_offsets_size;
        topk_slots_size     = ori_node->topk_slots_size;
        plan                = ori_node->plan;
        kernel_ctx_next     = ori_node->kernel_ctx_next;
        plan_next           = ori_node->plan_next;
        op                  = RPP_OP_EXPERT_FORWARD;
    }

    ~rpp_kernel_export_forward() {
        if (!ori_rpp_node) {
            if (kernel_ctx_next) {
                rpp_destroy_kernel_ctx(*(kernel_ctx_next.get()));
                kernel_expert_forward::destroy_fusion_graph_bundle(plan_next.graphs);
                kernel_expert_forward::destroy_fusion_graph_bundle(plan.graphs);
            }
        }
        if (kernel_ctx_up) {
            rpp_destroy_kernel_ctx(*(kernel_ctx_up.get()));
        }
        if (kernel_ctx_down) {
            rpp_destroy_kernel_ctx(*(kernel_ctx_down.get()));
        }
        if (kernel_ctx_glu) {
            rpp_destroy_kernel_ctx(*(kernel_ctx_glu.get()));
        }
        if (kernel_ctx_topk) {
            rpp_destroy_kernel_ctx(*(kernel_ctx_topk.get()));
        }
        if (dev_expert_id) {
            RPP_CHECK(rppGraphResourceFree(dev_expert_id, RPP_GRAPH_RESOURCE_CDMA_DESC));
            dev_expert_id = 0;
        }
    }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_export_forward(ctx, gate, up, down, div, add, is_instantial, is_launch);
    }

    ggml_tensor * gate = nullptr;
    ggml_tensor * up   = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * div  = nullptr;
    ggml_tensor * add  = nullptr;

    // for seq_len > 1
    RPPdeviceptr dev_expert_info     = 0;
    void *       host_expert_info    = nullptr;
    size_t       token_ids_size      = 0;
    size_t       expert_counts_size  = 0;
    size_t       expert_offsets_size = 0;
    size_t       topk_slots_size     = 0;
    size_t       workspace_size      = (64 * 1024) * (int) sizeof(uint16_t);

    kernel_expert_forward::fusion_runtime_plan plan;
    kernel_expert_forward::fusion_runtime_plan plan_next;
    std::shared_ptr<rpp_kernel_context>        kernel_ctx_next{ nullptr };

    // for seq_len == 1
    RPPdeviceptr                                dev_expert_id = 0;
    std::shared_ptr<rpp_kernel_context>         kernel_ctx_up{ nullptr };
    std::shared_ptr<rpp_kernel_context>         kernel_ctx_down{ nullptr };
    std::shared_ptr<rpp_kernel_context>         kernel_ctx_glu{ nullptr };
    std::shared_ptr<rpp_kernel_context>         kernel_ctx_topk{ nullptr };
    std::shared_ptr<expert_forward_vxm_sram_io> fusion_sram_io{ nullptr };
};

inline bool ggml_rpp_op_export_forward(ggml_backend_rpp_context & ctx,
                                       ggml_tensor *              gate,
                                       ggml_tensor *              up,
                                       ggml_tensor *              down,
                                       ggml_tensor *              div,
                                       ggml_tensor *              add,
                                       int                        is_instantial = 1,
                                       int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_export_forward(ctx, gate, up, down, div, add, is_instantial, is_launch);
}

inline bool ggml_rpp_can_fuse_expert_forward(const ggml_cgraph * cgraph,
                                             int &               node_idx,
                                             ggml_tensor *&      mul_gate,
                                             ggml_tensor *&      mul_up,
                                             ggml_tensor *&      mul_down,
                                             ggml_tensor *&      div,
                                             ggml_tensor *&      add) {
    return ggml_rpp_is_fuse_expert_forward(cgraph, node_idx, mul_gate, mul_up, mul_down, div, add);
}

#endif
