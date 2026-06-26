#include "rpp_expert_forward/rpp_expert_forward.h"
#include "rpp_glu/kernel_swiglu/rpp_kernel_build.h"
#include "rpp_glu/kernel_swiglu/rpp_kernel_param.h"
#include "rpp_mul_mat/kernel_q2_s_vxm_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_xs_vxm_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q3_xxs_vxm_nolut/rpp_kernel_build.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

constexpr int kQuantQ2s   = 0;
constexpr int kQuantQ2xs  = 1;
constexpr int kQuantQ3xxs = 2;
constexpr int kQkK        = 256;
constexpr int kTopkPadded = 8;

static bool ggml_rpp_expert_forward_node_matches(const ggml_cgraph * cgraph, int node_idx, ggml_op op) {
    return cgraph != nullptr && node_idx >= 0 && node_idx < cgraph->n_nodes && cgraph->nodes[node_idx] != nullptr &&
           cgraph->nodes[node_idx]->op == op;
}

static bool ggml_rpp_match_expert_forward_moe_ffn(const ggml_cgraph * cgraph, int mul_gate_idx, int mul_anchor_idx) {
    if (cgraph == nullptr || mul_gate_idx < 0 || mul_gate_idx + 3 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * mul_gate = cgraph->nodes[mul_gate_idx + 0];
    const ggml_tensor * mul_up   = cgraph->nodes[mul_gate_idx + 1];
    const ggml_tensor * swiglu   = cgraph->nodes[mul_gate_idx + 2];
    const ggml_tensor * mul_down = cgraph->nodes[mul_gate_idx + 3];
    if (mul_gate == nullptr || mul_up == nullptr || swiglu == nullptr || mul_down == nullptr ||
        mul_gate->op != GGML_OP_MUL_MAT_ID || mul_up->op != GGML_OP_MUL_MAT_ID || swiglu->op != GGML_OP_GLU ||
        ggml_get_glu_op(swiglu) != GGML_GLU_OP_SWIGLU || mul_down->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    const ggml_tensor * expert_id = mul_down->src[2];
    if (expert_id == nullptr) {
        return false;
    }

    if (mul_gate->src[1] == nullptr || mul_up->src[1] == nullptr || mul_gate->src[2] == nullptr ||
        mul_up->src[2] == nullptr) {
        return false;
    }

    if (mul_gate->src[1] != mul_up->src[1]) {
        return false;
    }

    if (mul_gate->src[2] != expert_id || mul_up->src[2] != expert_id) {
        return false;
    }

    if (mul_gate->type != GGML_TYPE_F32 || mul_up->type != GGML_TYPE_F32 || swiglu->type != GGML_TYPE_F32 ||
        mul_down->type != GGML_TYPE_F32 || mul_gate->src[1]->type != GGML_TYPE_F32 ||
        expert_id->type != GGML_TYPE_I32) {
        return false;
    }

    if ((swiglu->src[0] != mul_gate && swiglu->src[1] != mul_gate) ||
        (swiglu->src[0] != mul_up && swiglu->src[1] != mul_up) || mul_down->src[1] != swiglu) {
        return false;
    }

    const int mul_up_idx   = mul_gate_idx + 1;
    const int swiglu_idx   = mul_gate_idx + 2;
    const int mul_down_idx = mul_gate_idx + 3;
    if (mul_down_idx >= mul_anchor_idx) {
        return false;
    }

    if (!ggml_node_has_n_uses(cgraph, mul_gate_idx, 1) || !ggml_node_has_n_uses(cgraph, mul_up_idx, 1) ||
        !ggml_node_has_n_uses(cgraph, swiglu_idx, 1) || !ggml_node_has_n_uses(cgraph, mul_down_idx, 1)) {
        return false;
    }

    return true;
}

static bool ggml_rpp_match_residual_combine_fusion(const ggml_cgraph * cgraph, int node_idx) {
    if (cgraph == nullptr || node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * mul = cgraph->nodes[node_idx];
    if (mul == nullptr || mul->op != GGML_OP_MUL || mul->src[0] == nullptr || mul->src[1] == nullptr) {
        return false;
    }

    const int view_begin = node_idx + 1;
    if (!ggml_rpp_expert_forward_node_matches(cgraph, view_begin, GGML_OP_VIEW)) {
        return false;
    }

    static constexpr int k_max_view_count = 8;
    int                  view_count       = 0;
    const ggml_tensor *  first_view       = cgraph->nodes[view_begin];
    for (; view_count < k_max_view_count; ++view_count) {
        const int view_idx = view_begin + view_count;
        if (view_idx >= cgraph->n_nodes) {
            break;
        }
        const ggml_tensor * view = cgraph->nodes[view_idx];
        if (view->op != GGML_OP_VIEW || view->src[0] != mul) {
            break;
        }
        if (view_count > 0 && !ggml_are_same_layout(first_view, view)) {
            return false;
        }
    }

    if (view_begin + view_count < cgraph->n_nodes) {
        const ggml_tensor * next = cgraph->nodes[view_begin + view_count];
        if (next->op == GGML_OP_VIEW && next->src[0] == mul) {
            return false;
        }
    }

    if (view_count < 3) {
        return false;
    }

    if (!ggml_node_has_n_uses(cgraph, node_idx, view_count)) {
        return false;
    }

    const int reduce_sum_idx = view_begin + view_count;
    if (!ggml_rpp_expert_forward_node_matches(cgraph, reduce_sum_idx, GGML_OP_ADD)) {
        return false;
    }

    int                  n_fuse       = 0;
    static const ggml_op k_add_ops[8] = {
        GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD,
    };

    for (; n_fuse <= 6; ++n_fuse) {
        if (!ggml_can_fuse(cgraph, reduce_sum_idx + n_fuse, k_add_ops + n_fuse, 2)) {
            break;
        }
        if (cgraph->nodes[reduce_sum_idx + n_fuse] != cgraph->nodes[reduce_sum_idx + n_fuse + 1]->src[0]) {
            break;
        }
        if (!ggml_are_same_layout(cgraph->nodes[reduce_sum_idx + n_fuse]->src[1],
                                  cgraph->nodes[reduce_sum_idx + n_fuse + 1]->src[1])) {
            break;
        }
    }
    ++n_fuse;

    if (n_fuse <= 1 || view_count != n_fuse + 1) {
        return false;
    }

    for (int j = 0; j < n_fuse; ++j) {
        const ggml_tensor * add          = cgraph->nodes[reduce_sum_idx + j];
        const ggml_tensor * expected_rhs = cgraph->nodes[view_begin + j + 1];

        if (add->src[1] != expected_rhs) {
            return false;
        }
        if (expected_rhs->op != GGML_OP_VIEW || expected_rhs->src[0] != mul) {
            return false;
        }
        if (j == 0 && add->src[0] != cgraph->nodes[view_begin]) {
            return false;
        }
    }

    const int reduce_sum_out_idx = reduce_sum_idx + n_fuse - 1;
    const int residual_add_idx   = reduce_sum_out_idx + 1;
    if (!ggml_rpp_expert_forward_node_matches(cgraph, residual_add_idx, GGML_OP_ADD)) {
        return false;
    }

    const ggml_tensor * reduce_sum_out = cgraph->nodes[reduce_sum_out_idx];
    const ggml_tensor * residual_add   = cgraph->nodes[residual_add_idx];
    if (residual_add->src[0] != reduce_sum_out) {
        return false;
    }
    if (residual_add->src[1] == nullptr || residual_add->src[1] == reduce_sum_out) {
        return false;
    }

    if (!ggml_node_has_n_uses(cgraph, reduce_sum_out_idx, 1)) {
        return false;
    }

    return true;
}

bool ggml_rpp_is_fuse_expert_forward(const ggml_cgraph * cgraph,
                                     int &               node_idx,
                                     ggml_tensor *&      mul_gate,
                                     ggml_tensor *&      mul_up,
                                     ggml_tensor *&      mul_down,
                                     ggml_tensor *&      div,
                                     ggml_tensor *&      add) {
    if (cgraph == nullptr || node_idx < 3 || node_idx + 10 >= cgraph->n_nodes) {
        return false;
    }

    int           mul_idx     = 0;
    ggml_tensor * swiglu      = nullptr;
    ggml_tensor * mul         = nullptr;
    ggml_tensor * div_reshape = nullptr;

    if (ggml_rpp_expert_forward_node_matches(cgraph, node_idx - 3, GGML_OP_DIV) &&
        ggml_rpp_expert_forward_node_matches(cgraph, node_idx + 4, GGML_OP_MUL)) {
        if (node_idx + 19 >= cgraph->n_nodes) {
            return false;
        }
        mul_gate    = cgraph->nodes[node_idx + 0];
        mul_up      = cgraph->nodes[node_idx + 1];
        swiglu      = cgraph->nodes[node_idx + 2];
        mul_down    = cgraph->nodes[node_idx + 3];
        mul         = cgraph->nodes[node_idx + 4];
        div_reshape = cgraph->nodes[node_idx - 2];
        div         = cgraph->nodes[node_idx - 3];
        add         = cgraph->nodes[node_idx + 19];
        mul_idx     = node_idx + 4;
    } else if (ggml_rpp_expert_forward_node_matches(cgraph, node_idx + 8, GGML_OP_DIV) &&
               ggml_rpp_expert_forward_node_matches(cgraph, node_idx + 10, GGML_OP_MUL)) {
        if (node_idx + 25 >= cgraph->n_nodes) {
            return false;
        }
        mul_gate    = cgraph->nodes[node_idx + 0];
        mul_up      = cgraph->nodes[node_idx + 1];
        swiglu      = cgraph->nodes[node_idx + 2];
        mul_down    = cgraph->nodes[node_idx + 3];
        div         = cgraph->nodes[node_idx + 8];
        div_reshape = cgraph->nodes[node_idx + 9];
        mul         = cgraph->nodes[node_idx + 10];
        add         = cgraph->nodes[node_idx + 25];
        mul_idx     = node_idx + 10;
    } else {
        return false;
    }

    if (mul_gate == nullptr || mul_up == nullptr || swiglu == nullptr || mul_down == nullptr || mul == nullptr ||
        div_reshape == nullptr || div == nullptr || add == nullptr) {
        return false;
    }

    if (mul_gate->op != GGML_OP_MUL_MAT_ID || mul_up->op != GGML_OP_MUL_MAT_ID || swiglu->op != GGML_OP_GLU ||
        ggml_get_glu_op(swiglu) != GGML_GLU_OP_SWIGLU || mul_down->op != GGML_OP_MUL_MAT_ID || mul->op != GGML_OP_MUL ||
        div_reshape->op != GGML_OP_RESHAPE || div->op != GGML_OP_DIV || add->op != GGML_OP_ADD) {
        return false;
    }

    if (mul->src[0] != mul_down || mul->src[1] != div_reshape) {
        return false;
    }

    if (!ggml_rpp_match_expert_forward_moe_ffn(cgraph, node_idx, mul_idx)) {
        return false;
    }

    if (!ggml_rpp_match_residual_combine_fusion(cgraph, mul_idx)) {
        return false;
    }

    node_idx = mul_idx + 15;
    return true;
}

struct quant_layout_desc {
    int      quant_kind        = -1;
    uint32_t codebook_bytes    = 0;
    uint32_t scales_bytes      = 0;
    uint32_t sign_bytes        = 0;
    uint32_t super_scale_bytes = 0;
    uint32_t per_expert_bytes  = 0;
};

static bool get_quant_layout(ggml_type weight_type, int K, int N, quant_layout_desc & out) {
    if (K <= 0 || N <= 0) {
        return false;
    }
    if (K % kQkK != 0) {
        return false;
    }

    out = {};
    switch (weight_type) {
        case GGML_TYPE_IQ2_S:
            out.quant_kind        = kQuantQ2s;
            out.codebook_bytes    = (uint32_t) ((uint64_t) K * (uint64_t) N / 8ull * sizeof(uint16_t));
            out.scales_bytes      = (uint32_t) ((uint64_t) K * (uint64_t) N / 64ull * sizeof(uint16_t));
            out.sign_bytes        = (uint32_t) ((uint64_t) K * (uint64_t) N / 16ull * sizeof(uint16_t));
            out.super_scale_bytes = (uint32_t) ((uint64_t) (K / kQkK) * (uint64_t) N * sizeof(uint16_t));
            break;
        case GGML_TYPE_IQ2_XS:
            out.quant_kind        = kQuantQ2xs;
            out.codebook_bytes    = (uint32_t) ((uint64_t) K * (uint64_t) N / 8ull * sizeof(uint16_t));
            out.scales_bytes      = (uint32_t) ((uint64_t) K * (uint64_t) N / 64ull * sizeof(uint16_t));
            out.sign_bytes        = (uint32_t) ((uint64_t) K * (uint64_t) N / 16ull * sizeof(uint16_t));
            out.super_scale_bytes = (uint32_t) ((uint64_t) (K / kQkK) * (uint64_t) N * sizeof(uint16_t));
            break;
        case GGML_TYPE_IQ3_XXS:
            out.quant_kind        = kQuantQ3xxs;
            out.codebook_bytes    = (uint32_t) (((uint64_t) K / 16ull) * 3ull * (uint64_t) N * sizeof(uint16_t));
            out.scales_bytes      = (uint32_t) ((uint64_t) K * (uint64_t) N / 128ull * sizeof(uint16_t));
            out.sign_bytes        = (uint32_t) ((uint64_t) K * (uint64_t) N / 16ull * sizeof(uint16_t));
            out.super_scale_bytes = (uint32_t) ((uint64_t) (K / kQkK) * (uint64_t) N * sizeof(uint16_t));
            break;
        default:
            return false;
    }
    out.per_expert_bytes = out.codebook_bytes + out.scales_bytes + out.sign_bytes + out.super_scale_bytes;
    return out.per_expert_bytes > 0;
}

static bool prepare_stage_weights(ggml_tensor *       weight,
                                  int                 nr_of_experts,
                                  quant_layout_desc & layout,
                                  RPPdeviceptr &      dev_codebook,
                                  RPPdeviceptr &      dev_scales,
                                  RPPdeviceptr &      dev_sign,
                                  RPPdeviceptr &      dev_super_scale) {
    if (weight == nullptr || !ggml_is_contiguous(weight) || nr_of_experts <= 0) {
        return false;
    }
    if (!get_quant_layout(weight->type, (int) weight->ne[0], (int) weight->ne[1], layout)) {
        return false;
    }
    RPPdeviceptr dev_weights = (RPPdeviceptr) weight->data;
    dev_codebook             = dev_weights;
    dev_scales               = dev_weights + layout.codebook_bytes;
    dev_sign                 = dev_weights + layout.codebook_bytes + layout.scales_bytes;
    dev_super_scale          = dev_weights + layout.codebook_bytes + layout.scales_bytes + layout.sign_bytes;
    if (dev_codebook == 0 || dev_scales == 0 || dev_sign == 0 || dev_super_scale == 0) {
        return false;
    }
    return true;
}

static int ggml_rpp_export_forward_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_export_forward_properties_is_same(ggml_tensor * dst, ggml_rpp_node * rpp_node) {
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

static bool ggml_rpp_expert_forward_tensor_signature_is_same(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return lhs == rhs;
    }
    if (lhs->op != rhs->op || lhs->type != rhs->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs->ne[i] != rhs->ne[i] || lhs->nb[i] != rhs->nb[i]) {
            return false;
        }
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * lhs_src = lhs->src[i];
        const ggml_tensor * rhs_src = rhs->src[i];
        if (lhs_src == nullptr || rhs_src == nullptr) {
            if (lhs_src != rhs_src) {
                return false;
            }
            continue;
        }
        if (lhs_src->type != rhs_src->type) {
            return false;
        }
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            if (lhs_src->ne[j] != rhs_src->ne[j] || lhs_src->nb[j] != rhs_src->nb[j]) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_rpp_export_forward_dims_is_same(ggml_tensor *   gate,
                                                 ggml_tensor *   up,
                                                 ggml_tensor *   down,
                                                 ggml_tensor *   div,
                                                 ggml_tensor *   add,
                                                 ggml_rpp_node * rpp_node) {
    GGML_ASSERT(rpp_node);
    if (rpp_node->op != ggml_rpp_node::RPP_OP_EXPERT_FORWARD) {
        return false;
    }
    auto ref_node = static_cast<rpp_kernel_export_forward *>(rpp_node);
    if (!ggml_rpp_expert_forward_tensor_signature_is_same(gate, ref_node->gate) ||
        !ggml_rpp_expert_forward_tensor_signature_is_same(up, ref_node->up) ||
        !ggml_rpp_expert_forward_tensor_signature_is_same(down, ref_node->down) ||
        !ggml_rpp_expert_forward_tensor_signature_is_same(div, ref_node->div) ||
        !ggml_rpp_expert_forward_tensor_signature_is_same(add, ref_node->add)) {
        return false;
    }
    return true;
}

static RPPdeviceptr ggml_rpp_prepare_quant_workspace(rpp_kernel_context & ctx, ggml_tensor * weight) {
    if (weight == nullptr) {
        return 0;
    }
    switch (weight->type) {
        case GGML_TYPE_IQ2_S:
            return kernel_q2_s_nolut::q2s_nolut_prepare_lut_workspace(ctx);
        case GGML_TYPE_IQ2_XS:
            return kernel_q2_xs_nolut::q2xs_nolut_prepare_lut_workspace(ctx);
        case GGML_TYPE_IQ3_XXS:
            return kernel_q3_xxs_nolut::q3xxs_nolut_prepare_lut_workspace(ctx);
        default:
            GGML_LOG_ERROR("%s: unsupported weight type: %s\n", __func__, ggml_type_name(weight->type));
            GGML_ASSERT(false);
            return 0;
    }
}

static RPPdeviceptr ggml_rpp_prepare_silu_workspace(rpp_kernel_context & ctx) {
    return kernel_swiglu::silu_prepare_lut_workspace(ctx);
}

static bool ggml_rpp_create_kernel_export_forward(ggml_backend_rpp_context & ctx,
                                                  ggml_rpp_node *            rpp_base_node,
                                                  ggml_tensor *              gate,
                                                  ggml_tensor *              up,
                                                  ggml_tensor *              down,
                                                  ggml_tensor *              div,
                                                  ggml_tensor *              add) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_export_forward *>(rpp_base_node);
    if (gate == nullptr || up == nullptr || down == nullptr || div == nullptr || add == nullptr) {
        return false;
    }
    if (gate->src[0] == nullptr || gate->src[1] == nullptr || gate->src[2] == nullptr || up->src[0] == nullptr ||
        up->src[1] == nullptr || up->src[2] == nullptr || down->src[0] == nullptr || down->src[1] == nullptr ||
        down->src[2] == nullptr) {
        return false;
    }

    ggml_tensor * ids = gate->src[2];
    if (ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(gate->src[0]) || !ggml_is_contiguous(up->src[0]) ||
        !ggml_is_contiguous(down->src[0])) {
        return false;
    }

    const int B             = ctx.use_ubatch ? rpp_node->n_ubatch : ids->ne[1];
    const int topk          = (int) ids->ne[0];
    const int K             = (int) gate->src[0]->ne[0];
    const int N             = (int) gate->src[0]->ne[1];
    const int nr_of_experts = (int) gate->src[0]->ne[2];
    if (B <= 0 || topk <= 0 || topk > kTopkPadded || K <= 0 || N <= 0 || nr_of_experts <= 0) {
        return false;
    }
    if ((int) up->src[0]->ne[0] != K || (int) up->src[0]->ne[1] != N || (int) up->src[0]->ne[2] != nr_of_experts) {
        return false;
    }
    if ((int) down->src[0]->ne[0] != N || (int) down->src[0]->ne[1] != K ||
        (int) down->src[0]->ne[2] != nr_of_experts) {
        return false;
    }

    const int in_bytes_per_element  = ggml_rpp_get_io_type_size(ctx, gate->src[1], 0);
    const int out_bytes_per_element = ggml_rpp_get_io_type_size(ctx, add, 1);
    if (in_bytes_per_element != 2 && in_bytes_per_element != 4) {
        return false;
    }
    if (out_bytes_per_element != 2 && out_bytes_per_element != 4) {
        return false;
    }

    ggml_tensor * dst  = gate;
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * src2 = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS;
    const int64_t n_expert_used = ids->ne[0];
    GGML_ASSERT(n_expert_used == topk);
    const int64_t n_experts = ne02;
    GGML_ASSERT(n_experts == nr_of_experts);

    const size_t token_ids_size      = (size_t) B * (size_t) n_expert_used * sizeof(uint16_t);
    const size_t expert_counts_size  = (size_t) n_experts * sizeof(uint32_t);
    const size_t expert_offsets_size = ((size_t) n_experts + 1) * sizeof(uint32_t);
    const size_t topk_slots_size     = (size_t) B * (size_t) n_expert_used * sizeof(uint16_t);
    const size_t expert_info_size    = token_ids_size + expert_counts_size + expert_offsets_size + topk_slots_size;

    if (!rpp_node->dev_expert_info || !rpp_node->host_expert_info) {
        rpp_node->token_ids_size      = token_ids_size;
        rpp_node->expert_counts_size  = expert_counts_size;
        rpp_node->expert_offsets_size = expert_offsets_size;
        rpp_node->topk_slots_size     = topk_slots_size;
        rpp_node->dev_expert_info     = (RPPdeviceptr) ctx.pool().alloc(expert_info_size);
        rpp_node->host_expert_info    = (void *) ctx.pool_host().alloc(expert_info_size);
    } else {
        GGML_ASSERT(rpp_node->token_ids_size == token_ids_size);
        GGML_ASSERT(rpp_node->expert_counts_size == expert_counts_size);
        GGML_ASSERT(rpp_node->expert_offsets_size == expert_offsets_size);
        GGML_ASSERT(rpp_node->topk_slots_size == topk_slots_size);
    }

    quant_layout_desc gate_layout{};
    quant_layout_desc up_layout{};
    quant_layout_desc down_layout{};
    RPPdeviceptr      dev_token_ids       = rpp_node->dev_expert_info;
    RPPdeviceptr      dev_expert_counts   = dev_token_ids + rpp_node->token_ids_size;
    RPPdeviceptr      dev_expert_offsets  = dev_expert_counts + rpp_node->expert_counts_size;
    RPPdeviceptr      dev_gate_codebook   = 0;
    RPPdeviceptr      dev_gate_scales     = 0;
    RPPdeviceptr      dev_gate_sign       = 0;
    RPPdeviceptr      dev_gate_super      = 0;
    RPPdeviceptr      dev_up_codebook     = 0;
    RPPdeviceptr      dev_up_scales       = 0;
    RPPdeviceptr      dev_up_sign         = 0;
    RPPdeviceptr      dev_up_super        = 0;
    RPPdeviceptr      dev_down_codebook   = 0;
    RPPdeviceptr      dev_down_scales     = 0;
    RPPdeviceptr      dev_down_sign       = 0;
    RPPdeviceptr      dev_down_super      = 0;
    RPPdeviceptr      dev_topk_slots      = dev_expert_offsets + rpp_node->expert_offsets_size;
    RPPdeviceptr      dev_routing_weights = (RPPdeviceptr) div->data;
    RPPdeviceptr      dev_final_out       = (RPPdeviceptr) add->data;

    if (!prepare_stage_weights(gate->src[0], nr_of_experts, gate_layout, dev_gate_codebook, dev_gate_scales,
                               dev_gate_sign, dev_gate_super) ||
        !prepare_stage_weights(up->src[0], nr_of_experts, up_layout, dev_up_codebook, dev_up_scales, dev_up_sign,
                               dev_up_super) ||
        !prepare_stage_weights(down->src[0], nr_of_experts, down_layout, dev_down_codebook, dev_down_scales,
                               dev_down_sign, dev_down_super)) {
        return false;
    }

    if (rpp_node->kernel_ctx->dev_workspace == 0) {
        rpp_node->kernel_ctx->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc(rpp_node->workspace_size * 4));
        GGML_ASSERT(rpp_node->kernel_ctx->dev_workspace != 0);
        auto &             cur_kernel_ctx     = *(rpp_node->kernel_ctx.get());
        const RPPdeviceptr dev_workspace_base = cur_kernel_ctx.dev_workspace;
        const size_t       workspace_offset   = (64 * 1024) * sizeof(uint16_t);
        cur_kernel_ctx.dev_workspace          = ggml_rpp_prepare_quant_workspace(cur_kernel_ctx, gate->src[0]);
        GGML_ASSERT(cur_kernel_ctx.dev_workspace != 0);
        cur_kernel_ctx.dev_workspace += workspace_offset;
        cur_kernel_ctx.dev_workspace = ggml_rpp_prepare_quant_workspace(cur_kernel_ctx, up->src[0]);
        GGML_ASSERT(cur_kernel_ctx.dev_workspace != 0);
        cur_kernel_ctx.dev_workspace += workspace_offset;
        cur_kernel_ctx.dev_workspace = ggml_rpp_prepare_quant_workspace(cur_kernel_ctx, down->src[0]);
        GGML_ASSERT(cur_kernel_ctx.dev_workspace != 0);
        cur_kernel_ctx.dev_workspace += workspace_offset;
        cur_kernel_ctx.dev_workspace = ggml_rpp_prepare_silu_workspace(cur_kernel_ctx);
        GGML_ASSERT(cur_kernel_ctx.dev_workspace != 0);
        cur_kernel_ctx.dev_workspace = dev_workspace_base;
    }

    rpp_node->kernel_ctx->dev_in.clear();
    rpp_node->kernel_ctx->dev_out.clear();
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) gate->src[1]->data);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_token_ids);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_expert_counts);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_expert_offsets);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_gate_codebook);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_gate_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_gate_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_gate_super);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_up_codebook);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_up_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_up_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_up_super);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_down_codebook);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_down_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_down_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_down_super);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_topk_slots);
    rpp_node->kernel_ctx->dev_in.emplace_back(dev_routing_weights);
    rpp_node->kernel_ctx->dev_out.emplace_back(dev_final_out);
    kernel_expert_forward::rpp_matmul_id_fusion_build(
        *(rpp_node->kernel_ctx.get()), rpp_node->plan, B, K, N, nr_of_experts, in_bytes_per_element,
        out_bytes_per_element, gate_layout.quant_kind, up_layout.quant_kind, down_layout.quant_kind);

    // for next kernel ctx and plan
    if (!rpp_node->kernel_ctx_next) {
        rpp_node->kernel_ctx_next = std::make_shared<rpp_kernel_context>();
        rpp_init_kernel_ctx(*(rpp_node->kernel_ctx_next.get()));
    }
    rpp_node->kernel_ctx_next->dev_workspace = rpp_node->kernel_ctx->dev_workspace;
    rpp_node->kernel_ctx_next->dev_in        = rpp_node->kernel_ctx->dev_in;
    rpp_node->kernel_ctx_next->dev_out       = rpp_node->kernel_ctx->dev_out;
    kernel_expert_forward::rpp_matmul_id_fusion_build(
        *(rpp_node->kernel_ctx_next.get()), rpp_node->plan_next, B, K, N, nr_of_experts, in_bytes_per_element,
        out_bytes_per_element, gate_layout.quant_kind, up_layout.quant_kind, down_layout.quant_kind);
    return true;
}

struct mul_mat_id_build_info {
    uint32_t     size_in{ 0 };
    uint32_t     size_weight{ 0 };
    RPPdeviceptr sram_weight{ 0 };
    uint32_t     size_out{ 0 };
    RPPdeviceptr sram_out{ 0 };
};

static uint32_t ggml_rpp_mul_mat_id_vxm_required_sram(rpp_kernel_context * kernel_ctx,
                                                      ggml_tensor *        dst,
                                                      int                  M,
                                                      int                  K,
                                                      int                  N,
                                                      int                  experts,
                                                      int                  i_type_size,
                                                      int                  o_type_size) {
    GGML_ASSERT(kernel_ctx);
    GGML_ASSERT(dst && dst->src[0] && dst->src[1]);
    uint32_t total_sram_bytes = 0;
    try {
        switch (dst->src[0]->type) {
            case GGML_TYPE_IQ2_S:
                {
                    kernel_q2_s_vxm_nolut::q2s_vxm_nolut_sram_io io{};
                    kernel_q2_s_vxm_nolut::q2s_vxm_nolut_prepare_sram_io(*kernel_ctx, io, M, K, N, QK_K, i_type_size,
                                                                         o_type_size, experts, false);
                    total_sram_bytes = io.total_sram_bytes;
                }
                break;
            case GGML_TYPE_IQ2_XS:
                {
                    kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_sram_io io{};
                    kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_prepare_sram_io(*kernel_ctx, io, M, K, N, QK_K, i_type_size,
                                                                           o_type_size, experts, false);
                    total_sram_bytes = io.total_sram_bytes;
                }
                break;
            case GGML_TYPE_IQ3_XXS:
                {
                    kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_sram_io io{};
                    kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_prepare_sram_io(*kernel_ctx, io, M, K, N, QK_K,
                                                                             i_type_size, o_type_size, experts, false);
                    total_sram_bytes = io.total_sram_bytes;
                }
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported mul_mat_id weight type %d in fusion (%s)\n", __func__,
                               (int) dst->src[0]->type, dst->name);
                break;
        }
    } catch (const std::exception & ex) {
        GGML_LOG_ERROR("%s: prepare mul_mat_id sram failed (%s): %s\n", __func__, dst->name, ex.what());
    }
    GGML_ASSERT(total_sram_bytes > 0);
    return total_sram_bytes;
}

static bool ggml_rpp_mul_mat_id_vxm_prepare_sram(rpp_kernel_context *    kernel_ctx,
                                                 ggml_tensor *           dst,
                                                 int                     M,
                                                 int                     K,
                                                 int                     N,
                                                 int                     experts,
                                                 RPPdeviceptr            sram_input,
                                                 int                     i_type_size,
                                                 int                     o_type_size,
                                                 mul_mat_id_build_info & out) {
    GGML_ASSERT(kernel_ctx);
    GGML_ASSERT(dst && dst->src[0] && dst->src[1]);
    try {
        switch (dst->src[0]->type) {
            case GGML_TYPE_IQ2_S:
                {
                    kernel_q2_s_vxm_nolut::q2s_vxm_nolut_sram_io io;
                    kernel_q2_s_vxm_nolut::q2s_vxm_nolut_prepare_sram_io(*kernel_ctx, io, M, K, N, QK_K, i_type_size,
                                                                         o_type_size, experts, true);
                    kernel_ctx->dev_in[0] = sram_input;  // allow op1 to reuse op0 input buffer
                    out.size_in           = io.sizeA;
                    out.sram_weight       = io.sramB_codebook_nolut;
                    out.size_weight       = io.size_weights_expert;
                    if (o_type_size == (int) sizeof(float)) {
                        out.sram_out = io.sramC1;
                        out.size_out = io.sizeC32;
                    } else {
                        out.sram_out = io.sramC;
                        out.size_out = io.sizeC;
                    }

                    kernel_q2_s_vxm_nolut::q2s_vxm_nolut_prepare_lut_workspace(*kernel_ctx);
                    return true;
                }
            case GGML_TYPE_IQ2_XS:
                {
                    kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_sram_io io;
                    kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_prepare_sram_io(*kernel_ctx, io, M, K, N, QK_K, i_type_size,
                                                                           o_type_size, experts, true);
                    kernel_ctx->dev_in[0] = sram_input;
                    out.size_in           = io.sizeA;
                    out.sram_weight       = io.sramB_codebook_nolut;
                    out.size_weight       = io.size_weights_expert;
                    if (o_type_size == (int) sizeof(float)) {
                        out.sram_out = io.sramC1;
                        out.size_out = io.sizeC32;
                    } else {
                        out.sram_out = io.sramC;
                        out.size_out = io.sizeC;
                    }
                    kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_prepare_lut_workspace(*kernel_ctx);
                    return true;
                }
            case GGML_TYPE_IQ3_XXS:
                {
                    kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_sram_io io{};
                    kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_prepare_sram_io(*kernel_ctx, io, M, K, N, QK_K,
                                                                             i_type_size, o_type_size, experts, true);
                    kernel_ctx->dev_in[0] = sram_input;
                    out.size_in           = io.sizeA;
                    out.sram_weight       = io.sramB_codebook_nolut;
                    out.size_weight       = io.size_weights_expert;
                    if (o_type_size == (int) sizeof(float)) {
                        out.sram_out = io.sramC1;
                        out.size_out = io.sizeC32;
                    } else {
                        out.sram_out = io.sramC;
                        out.size_out = io.sizeC;
                    }
                    kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_prepare_lut_workspace(*kernel_ctx);
                    return true;
                }
            default:
                GGML_LOG_ERROR("%s: unsupported mul_mat_id weight type %d in fusion (%s)\n", __func__,
                               (int) dst->src[0]->type, dst->name);
                return false;
        }
    } catch (const std::exception & ex) {
        GGML_LOG_ERROR("%s: build mul_mat_id kernel failed (%s): %s\n", __func__, dst->name, ex.what());
        return false;
    }
}

static bool ggml_rpp_mul_mat_id_vxm_build_kernel(rpp_kernel_context & ctx,
                                                 ggml_tensor *        dst,
                                                 int                  M,
                                                 int                  K,
                                                 int                  N,
                                                 int                  in_bytes_per_element,
                                                 int                  out_bytes_per_element,
                                                 int                  experts,
                                                 int                  is_instantial = 1,
                                                 int                  is_capture    = 1) {
    GGML_ASSERT(dst && dst->src[0] && dst->src[1]);
    try {
        switch (dst->src[0]->type) {
            case GGML_TYPE_IQ2_S:
                {
                    kernel_q2_s_vxm_nolut::rpp_matmul_q2s_vxm_nolut_sram(ctx, M, K, N, QK_K, in_bytes_per_element,
                                                                         out_bytes_per_element, experts, is_instantial,
                                                                         is_capture);
                    return true;
                }
            case GGML_TYPE_IQ2_XS:
                {
                    kernel_q2_xs_vxm_nolut::rpp_matmul_q2xs_vxm_nolut_sram(ctx, M, K, N, QK_K, in_bytes_per_element,
                                                                           out_bytes_per_element, experts,
                                                                           is_instantial, is_capture);
                    return true;
                }
            case GGML_TYPE_IQ3_XXS:
                {
                    kernel_q3_xxs_vxm_nolut::rpp_matmul_q3xxs_vxm_nolut_sram(ctx, M, K, N, QK_K, in_bytes_per_element,
                                                                             out_bytes_per_element, experts,
                                                                             is_instantial, is_capture);
                    return true;
                }
            default:
                GGML_LOG_ERROR("%s: unsupported mul_mat_id weight type %d in fusion (%s)\n", __func__,
                               (int) dst->src[0]->type, dst->name);
                return false;
        }
    } catch (const std::exception & ex) {
        GGML_LOG_ERROR("%s: build mul_mat_id kernel failed (%s): %s\n", __func__, dst->name, ex.what());
        return false;
    }
}

struct expert_forward_weight_update_binding {
    size_t                            update_ordinal{ 0 };
    RPPdeviceptr                      target_src{ 0 };
    RPPdeviceptr                      target_dst{ 0 };
    RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params{};
};

static RPPgraphNode ggml_rpp_find_nth_indirect_update_node(RPPgraph graph, size_t update_ordinal) {
    RPPgraphNode nodes[4096];
    size_t       num_nodes = 0;
    RPP_CHECK(rppGraphGetNodes(graph, nodes, &num_nodes));

    size_t seen_updates = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        RPPgraphNodeType type = RPP_GRAPH_NODE_TYPE_EMPTY;
        RPP_CHECK(rppGraphNodeGetType(nodes[i], &type));
        if (type != RPP_GRAPH_NODE_TYPE_MEMCPY_INDIRECT_UPDATE) {
            continue;
        }
        if (seen_updates == update_ordinal) {
            return nodes[i];
        }
        ++seen_updates;
    }

    throw std::runtime_error("expert_forward weight io-update node not found");
}

static RPPgraphNode ggml_rpp_find_memcpy_node_by_src_dst(RPPgraph graph, RPPdeviceptr src, RPPdeviceptr dst) {
    RPPgraphNode nodes[4096];
    size_t       num_nodes = 0;
    RPP_CHECK(rppGraphGetNodes(graph, nodes, &num_nodes));

    for (size_t i = 0; i < num_nodes; ++i) {
        RPPgraphNodeType type = RPP_GRAPH_NODE_TYPE_EMPTY;
        RPP_CHECK(rppGraphNodeGetType(nodes[i], &type));
        if (type != RPP_GRAPH_NODE_TYPE_MEMCPY) {
            continue;
        }

        RPP_MEMCPY3D memcpy_params;
        std::memset(&memcpy_params, 0, sizeof(memcpy_params));
        RPP_CHECK(rppGraphMemcpyNodeGetParams(nodes[i], &memcpy_params));
        if (memcpy_params.src == src && memcpy_params.dst == dst) {
            return nodes[i];
        }
    }

    throw std::runtime_error("expert_forward target weight memcpy node not found");
}

static void ggml_rpp_bind_weight_update_node(RPPgraph graph, const expert_forward_weight_update_binding & binding) {
    RPP_MEMCPY_INDIRECT_UPDATE_NODE_PARAMS node_params;
    std::memset(&node_params, 0, sizeof(node_params));
    node_params.targetNode = ggml_rpp_find_memcpy_node_by_src_dst(graph, binding.target_src, binding.target_dst);
    std::memcpy(&node_params.updateParams, &binding.update_params, sizeof(binding.update_params));

    RPP_CHECK(rppGraphMemcpyIndirectUpdateNodeSetParams(
        graph, ggml_rpp_find_nth_indirect_update_node(graph, binding.update_ordinal), &node_params));
}

static RPP_MEMCPY_INDIRECT_UPDATE_PARAMS ggml_rpp_make_weight_base_offset_update(RPPdeviceptr index_addr,
                                                                                 RPPdeviceptr base_addr,
                                                                                 RPPdeviceptr max_addr,
                                                                                 uint32_t     block_size) {
    RPPdeviceptr min_phy_addr = 0;
    RPP_CHECK(rppMemGetPhyAddr(&min_phy_addr, base_addr));
    RPPdeviceptr max_phy_addr = 0;
    RPP_CHECK(rppMemGetPhyAddr(&max_phy_addr, max_addr));
    auto isAddrRangeChanged = [&](RPPdeviceptr addr1, RPPdeviceptr addr2) -> bool {
        return (addr1 >> 32) != (addr2 >> 32);
    };
    size_t element_size = sizeof(uint32_t);
    if (isAddrRangeChanged(min_phy_addr, max_phy_addr)) {
        element_size = sizeof(uint64_t);
    }

    RPP_MEMCPY_INDIRECT_UPDATE_PARAMS params;
    std::memset(&params, 0, sizeof(params));
    params.inputType                    = RPP_MEMCPY_INDIRECT_INPUT_TYPE_BASE_OFFSET;
    params.input.baseOffset.indexAddr   = index_addr;
    params.input.baseOffset.baseAddr    = base_addr;
    params.input.baseOffset.elementSize = element_size;
    params.input.baseOffset.blockSize   = block_size;
    params.input.baseOffset.offset      = 0;
    params.target                       = RPP_MEMCPY_INDIRECT_TARGET_SRC_ADDR;
    return params;
}

static bool ggml_rpp_weigths_io_update_graph(rpp_kernel_context &                                ctx,
                                             rpp_kernel_export_forward *                         fusion_node,
                                             ggml_tensor *                                       weight,
                                             RPPdeviceptr                                        sram_weight,
                                             uint32_t                                            block_size,
                                             int                                                 topk,
                                             std::vector<expert_forward_weight_update_binding> & bindings,
                                             RPPstream update_stream = nullptr,
                                             RPPstream copy_stream   = nullptr,
                                             RPPevent  update_done   = nullptr,
                                             RPPevent  copy_done     = nullptr) {
    GGML_ASSERT(fusion_node);
    GGML_ASSERT(weight);
    GGML_ASSERT(fusion_node->dev_expert_id != 0);
    GGML_ASSERT(topk > 0);
    if (update_stream == nullptr) {
        update_stream = ctx.kernelStream;
    }
    if (copy_stream == nullptr) {
        copy_stream = update_stream;
    }

    if (weight->data == nullptr || sram_weight == 0 || block_size == 0 || weight->ne[2] <= 0) {
        GGML_LOG_ERROR("%s: invalid weight stage io\n", __func__);
        return false;
    }

    RPPcontext current_ctx = nullptr;
    RPP_CHECK(rppCtxGetCurrent(&current_ctx));

    const RPPdeviceptr base_addr = (RPPdeviceptr) weight->data;
    const RPPdeviceptr max_addr  = base_addr + (RPPdeviceptr) ((uint64_t) weight->ne[2] * (uint64_t) block_size - 1ull);
    for (int slot = 0; slot < topk; ++slot) {
        const RPPdeviceptr index_addr =
            fusion_node->dev_expert_id + (RPPdeviceptr) ((uint64_t) slot * (uint64_t) sizeof(uint32_t));
        const RPPdeviceptr dst_addr = sram_weight + (RPPdeviceptr) ((uint64_t) slot * (uint64_t) block_size);
        const RPP_MEMCPY_INDIRECT_UPDATE_PARAMS update_params =
            ggml_rpp_make_weight_base_offset_update(index_addr, base_addr, max_addr, block_size);
        RPP_CHECK(
            rppGraphMemcpyNodeSetIndirectParamsAsync(ctx.graph, NULL, &update_params, current_ctx, update_stream));

        expert_forward_weight_update_binding binding;
        binding.update_ordinal = bindings.size();
        binding.target_src     = base_addr;
        binding.target_dst     = dst_addr;
        std::memcpy(&binding.update_params, &update_params, sizeof(update_params));
        bindings.emplace_back(binding);
    }

    if (update_stream != copy_stream) {
        GGML_ASSERT(update_done != nullptr);
        RPP_CHECK(rppEventRecord(update_done, update_stream));
        RPP_CHECK(rppStreamWaitEvent(copy_stream, update_done, 0));
    }

    for (int slot = 0; slot < topk; ++slot) {
        const RPPdeviceptr dst_addr = sram_weight + (RPPdeviceptr) ((uint64_t) slot * (uint64_t) block_size);
        RPP_CHECK(
            rtMemcpyAsync((void *) dst_addr, (const void *) base_addr, block_size, rtMemcpyDeviceToSram, copy_stream));
    }
    if (copy_done != nullptr) {
        RPP_CHECK(rppEventRecord(copy_done, copy_stream));
    }

    return true;
}

static void ggml_rpp_launch_moe_topk_combine_slot_major(RPPdeviceptr packed_input_bf16,
                                                        RPPdeviceptr topk_weights,
                                                        int          topk_weights_bytes_per_element,
                                                        RPPdeviceptr merged_out_bf16,
                                                        int          cols,
                                                        int          topk,
                                                        RPPmodule    module,
                                                        RPPstream    stream) {
    if (cols <= 0 || topk <= 0) {
        return;
    }

    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    threadsPerBlock.x = (uint32_t) cols;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = 1;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;

    params.emplace_back((uint32_t) packed_input_bf16);
    params.emplace_back((uint32_t) topk_weights);
    params.emplace_back((uint32_t) (cols * (int) sizeof(uint16_t)));
    params.emplace_back((uint32_t) merged_out_bf16);
    params.emplace_back((uint32_t) topk);

    const char * kernel_name =
        (topk_weights_bytes_per_element == (int) sizeof(uint16_t)) ? "moe_topk_comb_fp16" : "moe_topk_comb_fp32";
    launchWrapperAysnc(kernel_name, blocksPerGrid, threadsPerBlock, params, module, stream);
}

struct expert_forward_vxm_config {
    int          mode{ 1 };
    int          C{ 0 };
    int          H{ 0 };
    int          W{ 0 };
    int          M0{ 0 };
    int          K0{ 0 };
    int          N0{ 0 };
    int          K1{ 0 };
    int          N1{ 0 };
    int          K3{ 0 };
    int          N3{ 0 };
    int          experts{ 0 };
    int          gate_i_type_size{ 0 };
    int          gate_o_type_size{ 0 };
    int          up_i_type_size{ 0 };
    int          up_o_type_size{ 0 };
    int          glu_i_type_size0{ 0 };
    int          glu_o_type_size{ 0 };
    int          down_i_type_size{ 0 };
    int          down_o_type_size{ 0 };
    int          div_o_type_size{ 0 };
    int          add_o_type_size{ 0 };
    uint32_t     size_act{ 0 };
    RPPdeviceptr fusion_sram_base{ 0 };
};

static bool ggml_rpp_prepare_expert_forward_vxm_sram(rpp_kernel_export_forward *       fusion_node,
                                                     const expert_forward_vxm_config & cfg,
                                                     rpp_kernel_context *              ctx0,
                                                     rpp_kernel_context *              ctx1,
                                                     rpp_kernel_context *              ctx2,
                                                     rpp_kernel_context *              ctx3,
                                                     ggml_tensor *                     mul_gate,
                                                     ggml_tensor *                     mul_up,
                                                     ggml_tensor *                     mul_down,
                                                     ggml_tensor *                     div,
                                                     ggml_tensor *                     add) {
    const size_t size_op0 = ggml_rpp_mul_mat_id_vxm_required_sram(ctx0, mul_gate, cfg.M0, cfg.K0, cfg.N0, cfg.experts,
                                                                  cfg.gate_i_type_size, cfg.gate_o_type_size);
    const size_t size_op1 = ggml_rpp_mul_mat_id_vxm_required_sram(ctx1, mul_up, cfg.M0, cfg.K1, cfg.N1, cfg.experts,
                                                                  cfg.up_i_type_size, cfg.up_o_type_size);
    const size_t size_op2 = kernel_swiglu::silu_prepare_sram_io(*ctx2, cfg.mode, cfg.C, cfg.H, cfg.W, 2,
                                                                cfg.glu_i_type_size0, cfg.glu_o_type_size, false)
                                .total_sram_bytes;
    const size_t size_op3 = ggml_rpp_mul_mat_id_vxm_required_sram(ctx3, mul_down, cfg.M0, cfg.K3, cfg.N3, cfg.experts,
                                                                  cfg.down_i_type_size, cfg.down_o_type_size);
    const size_t size_op4 =
        round_up(ggml_nelements(div) * cfg.div_o_type_size) + round_up(ggml_nelements(add) * cfg.add_o_type_size);

    const size_t op_offset0        = 0;
    const size_t op_offset1        = size_op0;
    const size_t op_offset2        = op_offset1 + size_op1;
    const size_t op_offset3        = op_offset2 + size_op2;
    const size_t op_offset4        = op_offset3 + size_op3;
    const size_t total_sram_needed = op_offset4 + size_op4;
    const size_t k_virt_sram_max   = 22u * 1024u * 1024u;
    if (total_sram_needed > k_virt_sram_max) {
        GGML_LOG_WARN("%s: fusion SRAM layout %zu exceeds 22MB, skipping fusion\n", __func__, total_sram_needed);
        return false;
    }
    ctx0->virtual_sram_base = cfg.fusion_sram_base + (RPPdeviceptr) op_offset0;
    ctx1->virtual_sram_base = cfg.fusion_sram_base + (RPPdeviceptr) op_offset1;
    ctx2->virtual_sram_base = cfg.fusion_sram_base + (RPPdeviceptr) op_offset2;
    ctx3->virtual_sram_base = cfg.fusion_sram_base + (RPPdeviceptr) op_offset3;

    if (!fusion_node->fusion_sram_io) {
        fusion_node->fusion_sram_io = std::make_shared<expert_forward_vxm_sram_io>();
    }

    auto io              = fusion_node->fusion_sram_io.get();
    io->total_sram_bytes = (uint32_t) total_sram_needed;
    io->sram_base        = cfg.fusion_sram_base;
    io->sram_act_gate    = cfg.fusion_sram_base + (RPPdeviceptr) op_offset0;
    io->sram_act_up      = cfg.fusion_sram_base + (RPPdeviceptr) op_offset1;

    mul_mat_id_build_info info_gate{}, info_up{}, info_down{};
    if (!ggml_rpp_mul_mat_id_vxm_prepare_sram(ctx0, mul_gate, cfg.M0, cfg.K0, cfg.N0, cfg.experts, io->sram_act_gate,
                                              cfg.gate_i_type_size, cfg.gate_o_type_size, info_gate)) {
        return false;
    }
    io->size_act_gate = info_gate.size_in;
    GGML_ASSERT(io->size_act_gate == cfg.size_act);
    io->size_weight_gate = info_gate.size_weight;
    io->sram_weight_gate = info_gate.sram_weight;
    io->sram_out_gate    = info_gate.sram_out;
    io->size_out_gate    = info_gate.size_out;

    if (!ggml_rpp_mul_mat_id_vxm_prepare_sram(ctx1, mul_up, cfg.M0, cfg.K1, cfg.N1, cfg.experts, io->sram_act_gate,
                                              cfg.up_i_type_size, cfg.up_o_type_size, info_up)) {
        return false;
    }
    io->size_act_up = info_up.size_in;
    GGML_ASSERT(io->size_act_up == cfg.size_act);
    GGML_ASSERT(io->size_act_up == io->size_act_gate);
    io->size_weight_up = info_up.size_weight;
    io->sram_weight_up = info_up.sram_weight;
    io->sram_out_up    = info_up.sram_out;
    io->size_out_up    = info_up.size_out;

    auto silu_io = kernel_swiglu::silu_prepare_sram_io(*ctx2, cfg.mode, cfg.C, cfg.H, cfg.W, 2, cfg.glu_i_type_size0,
                                                       cfg.glu_o_type_size);
    GGML_ASSERT(io->size_out_gate == silu_io.size_in0);
    GGML_ASSERT(io->size_out_up == silu_io.size_in1);
    // Rebind GLU inputs before graph capture so op2 consumes op0/op1 SRAM outputs.
    ctx2->dev_in[0] = io->sram_out_gate;
    ctx2->dev_in[1] = io->sram_out_up;
    if (cfg.glu_i_type_size0 != (int) sizeof(float)) {
        ctx2->dev_in[3] = io->sram_out_gate;
        ctx2->dev_in[4] = io->sram_out_up;
    }
    io->size_out_glu = (cfg.glu_o_type_size == (int) sizeof(float)) ? silu_io.size_out : silu_io.size_out_bf16;
    io->sram_out_glu = silu_io.sram_out_final;

    if (!ggml_rpp_mul_mat_id_vxm_prepare_sram(ctx3, mul_down, cfg.M0, cfg.K3, cfg.N3, cfg.experts, io->sram_out_glu,
                                              cfg.down_i_type_size, cfg.down_o_type_size, info_down)) {
        return false;
    }
    GGML_ASSERT(info_down.size_in == io->size_out_glu);
    io->size_weight_down = info_down.size_weight;
    io->sram_weight_down = info_down.sram_weight;
    io->sram_out_down    = info_down.sram_out;
    io->size_out_down    = info_down.size_out;

    io->size_out_div = ggml_nelements(div) * cfg.div_o_type_size;
    io->sram_out_div = cfg.fusion_sram_base + (RPPdeviceptr) op_offset4;
    io->size_out_add = ggml_nelements(add) * cfg.add_o_type_size;
    io->sram_out_add = io->sram_out_div + round_up(io->size_out_div);
    return true;
}

static bool ggml_rpp_build_expert_forward_vxm_graph(rpp_kernel_export_forward *       fusion_node,
                                                    const expert_forward_vxm_config & cfg,
                                                    rpp_kernel_context *              ctx0,
                                                    rpp_kernel_context *              ctx1,
                                                    rpp_kernel_context *              ctx2,
                                                    rpp_kernel_context *              ctx3,
                                                    rpp_kernel_context *              ctx4,
                                                    ggml_tensor *                     mul_gate,
                                                    ggml_tensor *                     mul_up,
                                                    ggml_tensor *                     mul_down,
                                                    ggml_tensor *                     div,
                                                    ggml_tensor *                     add) {
    auto io = fusion_node->fusion_sram_io.get();
    GGML_ASSERT(io);

    std::vector<expert_forward_weight_update_binding> weight_update_bindings;
    weight_update_bindings.reserve((size_t) cfg.experts * 3u);

    const RPPstream old_ctx1_stream = ctx1->kernelStream;
    const RPPstream old_ctx2_stream = ctx2->kernelStream;
    const RPPstream old_ctx3_stream = ctx3->kernelStream;
    const RPPstream old_ctx4_stream = ctx4->kernelStream;
    ctx1->kernelStream              = ctx0->kernelStream;
    ctx2->kernelStream              = ctx0->kernelStream;
    ctx3->kernelStream              = ctx0->kernelStream;
    ctx4->kernelStream              = ctx0->kernelStream;

    auto restore_streams = [&]() {
        ctx1->kernelStream = old_ctx1_stream;
        ctx2->kernelStream = old_ctx2_stream;
        ctx3->kernelStream = old_ctx3_stream;
        ctx4->kernelStream = old_ctx4_stream;
    };

    const ggml_tensor * act = mul_gate->src[1];
    GGML_ASSERT(act == mul_up->src[1]);
    GGML_ASSERT(ggml_type_size(act->type) % (size_t) cfg.gate_i_type_size == 0);
    const size_t  act_scale      = ggml_type_size(act->type) / (size_t) cfg.gate_i_type_size;
    const size_t  src1_row_bytes = (size_t) act->nb[1] / act_scale;
    const int64_t ne11           = act->ne[1];
    if (ne11 != 1 && ne11 != cfg.experts) {
        GGML_LOG_ERROR("%s: fusion activation rows mismatch (%s): ne[1]=%lld, experts=%d\n", __func__, mul_gate->name,
                       (long long) ne11, cfg.experts);
        restore_streams();
        return false;
    }
    const size_t act_total_bytes = (size_t) cfg.experts * src1_row_bytes;
    if (act_total_bytes != (size_t) io->size_act_gate || act_total_bytes != (size_t) io->size_act_up) {
        GGML_LOG_ERROR(
            "%s: fusion activation SRAM size mismatch (%s): io.size_act_gate=%u, io.size_act_up=%u, expected=%zu\n",
            __func__, mul_gate->name, io->size_act_gate, io->size_act_up, act_total_bytes);
        restore_streams();
        return false;
    }

    auto vector_to_clear = [&](std::vector<RPPdeviceptr> & dst_sram, std::vector<RPPdeviceptr> & src_devi,
                               std::vector<size_t> & byte_count) {
        dst_sram.clear();
        src_devi.clear();
        byte_count.clear();
    };

    auto copy_activation_to_sram = [&](RPPdeviceptr sram_act_base, std::vector<RPPdeviceptr> & dst_sram,
                                       std::vector<RPPdeviceptr> & src_devi, std::vector<size_t> & byte_count) {
        if (ne11 == cfg.experts) {
            dst_sram.emplace_back(sram_act_base);
            src_devi.emplace_back((RPPdeviceptr) act->data);
            byte_count.emplace_back(act_total_bytes);
        } else {
            for (int64_t e = 0; e < cfg.experts; ++e) {
                const auto src1_cur  = (RPPdeviceptr) act->data;
                const auto sramA_cur = sram_act_base + (RPPdeviceptr) e * (RPPdeviceptr) src1_row_bytes;
                dst_sram.emplace_back(sramA_cur);
                src_devi.emplace_back(src1_cur);
                byte_count.emplace_back(src1_row_bytes);
            }
        }
    };

    auto copy_workspace_to_sram = [&](rpp_kernel_context * kernel_ctx, ggml_tensor * cur_tensor,
                                      std::vector<RPPdeviceptr> & dst_sram, std::vector<RPPdeviceptr> & src_devi,
                                      std::vector<size_t> & byte_count) {
        GGML_ASSERT(kernel_ctx);
        GGML_ASSERT(cur_tensor && cur_tensor->src[0] && cur_tensor->src[1]);
        switch (cur_tensor->src[0]->type) {
            case GGML_TYPE_IQ2_S:
                {
                    using namespace kernel_q2_s_vxm_nolut;
                    const RPPdeviceptr base_ptr         = kernel_ctx->dev_workspace;
                    const size_t       qscale_lut_bytes = q2s_vxm_nolut_lut_workspace::qscale_lut_bytes;
                    const size_t       mag_lut_bytes    = q2s_vxm_nolut_lut_workspace::mag_lut_bytes;

                    dst_sram.emplace_back(kernel_ctx->dev_in[5]);
                    dst_sram.emplace_back(kernel_ctx->dev_in[6]);
                    src_devi.emplace_back(base_ptr);
                    src_devi.emplace_back(base_ptr + (RPPdeviceptr) qscale_lut_bytes);
                    byte_count.emplace_back(qscale_lut_bytes);
                    byte_count.emplace_back(mag_lut_bytes);
                }
                break;
            case GGML_TYPE_IQ2_XS:
                {
                    const RPPdeviceptr base_ptr         = kernel_ctx->dev_workspace;
                    constexpr size_t   qscale_lut_bytes = 16u * sizeof(uint16_t);
                    constexpr size_t   mag_lut_bytes    = 4u * sizeof(uint16_t);

                    dst_sram.emplace_back(kernel_ctx->dev_in[5]);
                    dst_sram.emplace_back(kernel_ctx->dev_in[6]);
                    src_devi.emplace_back(base_ptr);
                    src_devi.emplace_back(base_ptr + (RPPdeviceptr) qscale_lut_bytes);
                    byte_count.emplace_back(qscale_lut_bytes);
                    byte_count.emplace_back(mag_lut_bytes);
                }
                break;
            case GGML_TYPE_IQ3_XXS:
                {
                    using namespace kernel_q3_xxs_vxm_nolut;
                    const RPPdeviceptr base_ptr     = kernel_ctx->dev_workspace;
                    const size_t       qscale_bytes = q3xxs_vxm_nolut_lut_workspace::qscale_lut_bytes;
                    const size_t       mag_bytes    = q3xxs_vxm_nolut_lut_workspace::mag_lut_bytes;
                    const size_t       mat_bytes    = q3xxs_vxm_nolut_lut_workspace::mat_lut_bytes;
                    const size_t       off_qscale   = 0;
                    const size_t       off_mag      = kernel_q3_xxs_vxm_nolut::align_up(off_qscale + qscale_bytes, 64);
                    const size_t       off_mat      = kernel_q3_xxs_vxm_nolut::align_up(off_mag + mag_bytes, 64);

                    dst_sram.emplace_back(kernel_ctx->dev_in[5]);
                    dst_sram.emplace_back(kernel_ctx->dev_in[6]);
                    dst_sram.emplace_back(kernel_ctx->dev_in[7]);
                    src_devi.emplace_back(base_ptr + (RPPdeviceptr) off_qscale);
                    src_devi.emplace_back(base_ptr + (RPPdeviceptr) off_mag);
                    src_devi.emplace_back(base_ptr + (RPPdeviceptr) off_mat);
                    byte_count.emplace_back(qscale_bytes);
                    byte_count.emplace_back(mag_bytes);
                    byte_count.emplace_back(mat_bytes);
                }
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported mul_mat_id weight type %d in fusion (%s)\n", __func__,
                               (int) cur_tensor->src[0]->type, cur_tensor->name);
                break;
        }
    };

    RPP_CHECK(rppStreamBeginCapture(ctx0->kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL));
    std::vector<RPPdeviceptr> dst_sram;
    std::vector<RPPdeviceptr> src_devi;
    std::vector<size_t>       byte_count;

    auto copy_sram_batch = [&](RPPstream stream) {
        if (!dst_sram.empty()) {
            RPP_CHECK(
                rppMemcpyLinkDtoSAsync(dst_sram.data(), src_devi.data(), byte_count.data(), dst_sram.size(), stream));
        }
    };

    const RPPevent capture_ready     = ctx0->kernel_done_ping[0];
    const RPPevent gate_update_done  = ctx0->mpu_done_ping[1];
    const RPPevent up_update_done    = ctx0->mpu_done_ping[0];
    const RPPevent down_update_done  = ctx0->dma_aux_done_ping[0];
    const RPPevent gate_weight_ready = ctx0->dma_done_ping[0];
    const RPPevent up_weight_ready   = ctx0->dma_done_ping[1];
    const RPPevent down_weight_ready = ctx0->dma_aux_done_ping[1];
    // Reuse after down_update_done has already been consumed by dmaStream.
    const RPPevent topk_input_ready  = ctx0->dma_aux_done_ping[0];
    const RPPevent topk_done         = ctx0->kernel_done_ping[1];
    // Reuse after gate_weight_ready has already been consumed by kernelStream.
    const RPPevent output_ready      = ctx0->dma_done_ping[0];

    RPP_CHECK(rppEventRecord(capture_ready, ctx0->kernelStream));
    RPP_CHECK(rppStreamWaitEvent(ctx0->dmaStream, capture_ready, 0));
    RPP_CHECK(rppStreamWaitEvent(ctx0->mpuStream, capture_ready, 0));

    RPP_CHECK(rppMemcpyDtoDAsync(fusion_node->dev_expert_id, (RPPdeviceptr) mul_gate->src[2]->data,
                                 sizeof(uint32_t) * cfg.experts, ctx0->mpuStream));

    vector_to_clear(dst_sram, src_devi, byte_count);
    copy_activation_to_sram(io->sram_act_gate, dst_sram, src_devi, byte_count);
    copy_workspace_to_sram(ctx0, mul_gate, dst_sram, src_devi, byte_count);
    copy_sram_batch(ctx0->dmaStream);

    if (!ggml_rpp_weigths_io_update_graph(*ctx0, fusion_node, mul_gate->src[0], io->sram_weight_gate,
                                          io->size_weight_gate, cfg.experts, weight_update_bindings, ctx0->mpuStream,
                                          ctx0->dmaStream, gate_update_done, gate_weight_ready)) {
        restore_streams();
        return false;
    }
    vector_to_clear(dst_sram, src_devi, byte_count);
    copy_workspace_to_sram(ctx1, mul_up, dst_sram, src_devi, byte_count);
    copy_sram_batch(ctx0->dmaStream);
    if (!ggml_rpp_weigths_io_update_graph(*ctx0, fusion_node, mul_up->src[0], io->sram_weight_up, io->size_weight_up,
                                          cfg.experts, weight_update_bindings, ctx0->mpuStream, ctx0->dmaStream,
                                          up_update_done, up_weight_ready)) {
        restore_streams();
        return false;
    }
    vector_to_clear(dst_sram, src_devi, byte_count);
    copy_workspace_to_sram(ctx3, mul_down, dst_sram, src_devi, byte_count);
    copy_sram_batch(ctx0->dmaStream);
    if (!ggml_rpp_weigths_io_update_graph(*ctx0, fusion_node, mul_down->src[0], io->sram_weight_down,
                                          io->size_weight_down, cfg.experts, weight_update_bindings, ctx0->mpuStream,
                                          ctx0->dmaStream, down_update_done, down_weight_ready)) {
        restore_streams();
        return false;
    }
    RPP_CHECK(rtMemcpyAsync((void *) io->sram_out_div, (const void *) div->data, io->size_out_div, rtMemcpyDeviceToSram,
                            ctx0->dmaStream));
    RPP_CHECK(rppEventRecord(topk_input_ready, ctx0->dmaStream));

    RPP_CHECK(rppStreamWaitEvent(ctx0->kernelStream, gate_weight_ready, 0));
    ggml_rpp_mul_mat_id_vxm_build_kernel(*ctx0, mul_gate, cfg.M0, cfg.K0, cfg.N0, cfg.gate_i_type_size,
                                         cfg.gate_o_type_size, cfg.experts, 0, 0);
    RPP_CHECK(rppStreamWaitEvent(ctx0->kernelStream, up_weight_ready, 0));
    ggml_rpp_mul_mat_id_vxm_build_kernel(*ctx1, mul_up, cfg.M0, cfg.K1, cfg.N1, cfg.up_i_type_size, cfg.up_o_type_size,
                                         cfg.experts, 0, 0);
    kernel_swiglu::rpp_silu_build(*ctx2, cfg.mode, cfg.C, cfg.H, cfg.W, 2, cfg.glu_i_type_size0, cfg.glu_o_type_size, 1,
                                  0, 0);
    RPP_CHECK(rppStreamWaitEvent(ctx0->kernelStream, down_weight_ready, 0));
    ggml_rpp_mul_mat_id_vxm_build_kernel(*ctx3, mul_down, cfg.M0, cfg.K3, cfg.N3, cfg.down_i_type_size,
                                         cfg.down_o_type_size, cfg.experts, 0, 0);

    if (ctx4->rppBinMod == 0) {
        RPP_CHECK(rppModuleLoad(&ctx4->rppBinMod, "rpp_kernel/topk_combine.o"));
    }
    RPP_CHECK(rppStreamWaitEvent(ctx0->kernelStream, topk_input_ready, 0));
    ggml_rpp_launch_moe_topk_combine_slot_major(io->sram_out_down, io->sram_out_div, cfg.div_o_type_size,
                                                io->sram_out_add, cfg.N3, cfg.experts, ctx4->rppBinMod,
                                                ctx0->kernelStream);
    RPP_CHECK(rppEventRecord(topk_done, ctx0->kernelStream));
    RPP_CHECK(rppStreamWaitEvent(ctx0->dmaStream, topk_done, 0));
    RPP_CHECK(rtMemcpyAsync((void *) add->data, (const void *) io->sram_out_add, io->size_out_add, rtMemcpySramToDevice,
                            ctx0->dmaStream));
    RPP_CHECK(rppEventRecord(output_ready, ctx0->dmaStream));
    RPP_CHECK(rppStreamWaitEvent(ctx0->kernelStream, output_ready, 0));

    RPP_CHECK(rppStreamEndCapture(ctx0->kernelStream, &ctx0->graph));
    restore_streams();

    for (const expert_forward_weight_update_binding & binding : weight_update_bindings) {
        ggml_rpp_bind_weight_update_node(ctx0->graph, binding);
    }

    RPP_CHECK(rppGraphInstantiate(&ctx0->graphexec, ctx0->graph, NULL, NULL, 0));
    return true;
}

static bool ggml_rpp_create_kernel_export_forward_vxm(ggml_backend_rpp_context & ctx,
                                                      ggml_rpp_node *            rpp_base_node,
                                                      ggml_tensor *              mul_gate,
                                                      ggml_tensor *              mul_up,
                                                      ggml_tensor *              mul_down,
                                                      ggml_tensor *              div,
                                                      ggml_tensor *              add) {
    auto fusion_node = static_cast<rpp_kernel_export_forward *>(rpp_base_node);
    GGML_ASSERT(mul_gate && mul_up && mul_down);
    GGML_ASSERT(mul_gate->src[0] && mul_gate->src[1] && mul_gate->src[2]);
    GGML_ASSERT(mul_up->src[0] && mul_up->src[1]);
    ggml_tensor * swiglu = mul_down->src[1];
    GGML_ASSERT(swiglu);
    GGML_ASSERT(swiglu->src[0] && swiglu->src[1]);
    GGML_ASSERT(mul_down->src[0] && mul_down->src[1] && mul_down->src[2]);
    GGML_ASSERT(ggml_get_glu_op(swiglu) == GGML_GLU_OP_SWIGLU);
    GGML_ASSERT(mul_gate->src[1] == mul_up->src[1]);
    GGML_ASSERT(swiglu->src[0] == mul_gate);
    GGML_ASSERT(swiglu->src[1] == mul_up);
    GGML_ASSERT(mul_down->src[1] == swiglu);
    GGML_ASSERT(mul_gate->src[2] == mul_up->src[2]);
    GGML_ASSERT(mul_gate->src[2] == mul_down->src[2]);

    const int M0               = (int) mul_gate->src[1]->ne[2];
    const int K0               = (int) mul_gate->src[0]->ne[0];
    const int N0               = (int) mul_gate->src[0]->ne[1];
    const int K1               = (int) mul_up->src[0]->ne[0];
    const int N1               = (int) mul_up->src[0]->ne[1];
    const int K3               = (int) mul_down->src[0]->ne[0];
    const int N3               = (int) mul_down->src[0]->ne[1];
    const int experts          = (int) mul_gate->src[2]->ne[0];
    const int nr_of_experts    = (int) mul_gate->src[0]->ne[2];
    const int gate_i_type_size = ggml_rpp_get_io_type_size(ctx, mul_gate->src[1], 0);
    const int gate_o_type_size = ggml_rpp_get_io_type_size(ctx, mul_gate, 1);
    const int up_i_type_size   = ggml_rpp_get_io_type_size(ctx, mul_up->src[1], 0);
    const int up_o_type_size   = ggml_rpp_get_io_type_size(ctx, mul_up, 1);
    const int glu_i_type_size0 = ggml_rpp_get_io_type_size(ctx, swiglu->src[0], 0);
    const int glu_i_type_size1 = ggml_rpp_get_io_type_size(ctx, swiglu->src[1], 0);
    const int glu_o_type_size  = ggml_rpp_get_io_type_size(ctx, swiglu, 1);
    const int down_i_type_size = ggml_rpp_get_io_type_size(ctx, mul_down->src[1], 0);
    const int down_o_type_size = ggml_rpp_get_io_type_size(ctx, mul_down, 1);
    const int div_o_type_size  = ggml_rpp_get_io_type_size(ctx, div, 1);
    const int add_o_type_size  = ggml_rpp_get_io_type_size(ctx, add, 1);
    GGML_ASSERT(M0 == 1);
    GGML_ASSERT((int) mul_up->src[1]->ne[2] == M0);
    GGML_ASSERT((int) mul_down->src[1]->ne[2] == M0);
    GGML_ASSERT((int) mul_gate->src[2]->ne[1] == M0);
    GGML_ASSERT((int) mul_up->src[2]->ne[1] == M0);
    GGML_ASSERT((int) mul_down->src[2]->ne[1] == M0);
    GGML_ASSERT(gate_i_type_size == up_i_type_size);
    GGML_ASSERT(glu_i_type_size0 == glu_i_type_size1);
    GGML_ASSERT(gate_o_type_size == glu_i_type_size0);
    GGML_ASSERT(up_o_type_size == glu_i_type_size1);
    GGML_ASSERT(glu_o_type_size == down_i_type_size);

    const int mode = 1;
    const int C    = (int) swiglu->ne[2];
    const int H    = (int) swiglu->ne[1];
    const int W    = (int) swiglu->ne[0];
    GGML_ASSERT(C == 1);

    const uint32_t     size_act = (uint32_t) ((size_t) experts * (size_t) M0 * (size_t) K0 * (size_t) gate_i_type_size);
    const RPPdeviceptr fusion_sram_base = fusion_node->kernel_ctx->virtual_sram_base;
    if (!fusion_node->kernel_ctx->dev_workspace) {
        fusion_node->kernel_ctx->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc(fusion_node->workspace_size));
        GGML_ASSERT(fusion_node->kernel_ctx->dev_workspace != 0);
    }
    if (!fusion_node->kernel_ctx_up) {
        fusion_node->kernel_ctx_up = std::make_shared<rpp_kernel_context>();
        rpp_init_kernel_ctx(*(fusion_node->kernel_ctx_up.get()));
        fusion_node->kernel_ctx_up->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc(fusion_node->workspace_size));
        GGML_ASSERT(fusion_node->kernel_ctx_up->dev_workspace != 0);
    }
    if (!fusion_node->kernel_ctx_down) {
        fusion_node->kernel_ctx_down = std::make_shared<rpp_kernel_context>();
        rpp_init_kernel_ctx(*(fusion_node->kernel_ctx_down.get()));
        fusion_node->kernel_ctx_down->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc(fusion_node->workspace_size));
        GGML_ASSERT(fusion_node->kernel_ctx_down->dev_workspace != 0);
    }
    if (!fusion_node->kernel_ctx_glu) {
        fusion_node->kernel_ctx_glu = std::make_shared<rpp_kernel_context>();
        rpp_init_kernel_ctx(*(fusion_node->kernel_ctx_glu.get()));
        fusion_node->kernel_ctx_glu->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc(fusion_node->workspace_size));
        GGML_ASSERT(fusion_node->kernel_ctx_glu->dev_workspace != 0);
    }
    if (!fusion_node->kernel_ctx_topk) {
        fusion_node->kernel_ctx_topk = std::make_shared<rpp_kernel_context>();
        rpp_init_kernel_ctx(*(fusion_node->kernel_ctx_topk.get()));
    }

    if (!fusion_node->dev_expert_id) {
        RPPdeviceptr phy_addr = 0;
        RPP_CHECK(rppGraphResourceAlloc(&phy_addr, sizeof(uint32_t) * nr_of_experts, RPP_GRAPH_RESOURCE_CDMA_DESC));
        rppMemGetVirtAddr(&fusion_node->dev_expert_id, RPP_MEMORYTYPE_GRAPH_DESC, phy_addr);
    }

    rpp_kernel_context * ctx0 = fusion_node->kernel_ctx.get();
    rpp_kernel_context * ctx1 = fusion_node->kernel_ctx_up.get();
    rpp_kernel_context * ctx2 = fusion_node->kernel_ctx_glu.get();
    rpp_kernel_context * ctx3 = fusion_node->kernel_ctx_down.get();
    rpp_kernel_context * ctx4 = fusion_node->kernel_ctx_topk.get();

    expert_forward_vxm_config vxm_cfg{};
    vxm_cfg.mode             = mode;
    vxm_cfg.C                = C;
    vxm_cfg.H                = H;
    vxm_cfg.W                = W;
    vxm_cfg.M0               = M0;
    vxm_cfg.K0               = K0;
    vxm_cfg.N0               = N0;
    vxm_cfg.K1               = K1;
    vxm_cfg.N1               = N1;
    vxm_cfg.K3               = K3;
    vxm_cfg.N3               = N3;
    vxm_cfg.experts          = experts;
    vxm_cfg.gate_i_type_size = gate_i_type_size;
    vxm_cfg.gate_o_type_size = gate_o_type_size;
    vxm_cfg.up_i_type_size   = up_i_type_size;
    vxm_cfg.up_o_type_size   = up_o_type_size;
    vxm_cfg.glu_i_type_size0 = glu_i_type_size0;
    vxm_cfg.glu_o_type_size  = glu_o_type_size;
    vxm_cfg.down_i_type_size = down_i_type_size;
    vxm_cfg.down_o_type_size = down_o_type_size;
    vxm_cfg.div_o_type_size  = div_o_type_size;
    vxm_cfg.add_o_type_size  = add_o_type_size;
    vxm_cfg.size_act         = size_act;
    vxm_cfg.fusion_sram_base = fusion_sram_base;

    if (!ggml_rpp_prepare_expert_forward_vxm_sram(fusion_node, vxm_cfg, ctx0, ctx1, ctx2, ctx3, mul_gate, mul_up,
                                                  mul_down, div, add)) {
        return false;
    }
    return ggml_rpp_build_expert_forward_vxm_graph(fusion_node, vxm_cfg, ctx0, ctx1, ctx2, ctx3, ctx4, mul_gate, mul_up,
                                                   mul_down, div, add);
}

static bool ggml_rpp_update_expert_forward_plan_inputs(rpp_kernel_export_forward *                  rpp_node,
                                                       kernel_expert_forward::fusion_runtime_plan & plan,
                                                       ggml_tensor *                                gate,
                                                       ggml_tensor *                                up,
                                                       ggml_tensor *                                down,
                                                       ggml_tensor *                                div,
                                                       ggml_tensor *                                add) {
    GGML_ASSERT(rpp_node);
    GGML_ASSERT(rpp_node->kernel_ctx && rpp_node->kernel_ctx_next);
    if (gate == nullptr || up == nullptr || down == nullptr || div == nullptr || add == nullptr) {
        return false;
    }

    const int         nr_of_experts = (int) gate->src[0]->ne[2];
    quant_layout_desc gate_layout{};
    quant_layout_desc up_layout{};
    quant_layout_desc down_layout{};
    RPPdeviceptr      dev_gate_codebook = 0;
    RPPdeviceptr      dev_gate_scales   = 0;
    RPPdeviceptr      dev_gate_sign     = 0;
    RPPdeviceptr      dev_gate_super    = 0;
    RPPdeviceptr      dev_up_codebook   = 0;
    RPPdeviceptr      dev_up_scales     = 0;
    RPPdeviceptr      dev_up_sign       = 0;
    RPPdeviceptr      dev_up_super      = 0;
    RPPdeviceptr      dev_down_codebook = 0;
    RPPdeviceptr      dev_down_scales   = 0;
    RPPdeviceptr      dev_down_sign     = 0;
    RPPdeviceptr      dev_down_super    = 0;

    if (!prepare_stage_weights(gate->src[0], nr_of_experts, gate_layout, dev_gate_codebook, dev_gate_scales,
                               dev_gate_sign, dev_gate_super) ||
        !prepare_stage_weights(up->src[0], nr_of_experts, up_layout, dev_up_codebook, dev_up_scales, dev_up_sign,
                               dev_up_super) ||
        !prepare_stage_weights(down->src[0], nr_of_experts, down_layout, dev_down_codebook, dev_down_scales,
                               dev_down_sign, dev_down_super)) {
        return false;
    }

    plan.inputs.sparse_act       = (RPPdeviceptr) gate->src[1]->data;
    plan.inputs.routing_weights  = (RPPdeviceptr) div->data;
    plan.inputs.gate.codebook    = dev_gate_codebook;
    plan.inputs.gate.scales      = dev_gate_scales;
    plan.inputs.gate.sign        = dev_gate_sign;
    plan.inputs.gate.super_scale = dev_gate_super;
    plan.inputs.up.codebook      = dev_up_codebook;
    plan.inputs.up.scales        = dev_up_scales;
    plan.inputs.up.sign          = dev_up_sign;
    plan.inputs.up.super_scale   = dev_up_super;
    plan.inputs.down.codebook    = dev_down_codebook;
    plan.inputs.down.scales      = dev_down_scales;
    plan.inputs.down.sign        = dev_down_sign;
    plan.inputs.down.super_scale = dev_down_super;
    return true;
}

static bool ggml_rpp_get_expert_forward_route_info(rpp_kernel_export_forward * rpp_node,
                                                   const char *                ids_host,
                                                   int64_t                     n_tokens,
                                                   int64_t                     n_expert_used,
                                                   int64_t                     n_experts,
                                                   int64_t                     ids_nb0,
                                                   int64_t                     ids_nb1,
                                                   std::vector<uint32_t> &     tmp_tokens_per_expert) {
    GGML_ASSERT(rpp_node->dev_expert_info != 0);
    GGML_ASSERT(rpp_node->host_expert_info != nullptr);

    char *     host_expert_info   = static_cast<char *>(rpp_node->host_expert_info);
    uint16_t * ids_to_sorted_host = reinterpret_cast<uint16_t *>(host_expert_info);
    uint32_t * tokens_per_expert  = reinterpret_cast<uint32_t *>(host_expert_info + rpp_node->token_ids_size);
    uint32_t * expert_offsets =
        reinterpret_cast<uint32_t *>(host_expert_info + rpp_node->token_ids_size + rpp_node->expert_counts_size);
    uint16_t * ids_from_sorted_host = reinterpret_cast<uint16_t *>(
        host_expert_info + rpp_node->token_ids_size + rpp_node->expert_counts_size + rpp_node->expert_offsets_size);
    std::fill_n(tokens_per_expert, n_experts, 0);
    std::fill_n(expert_offsets, n_experts + 1, 0);

    tmp_tokens_per_expert.assign(n_experts, 0);
    std::vector<int64_t> last_token_for_expert(n_experts, -1);
    size_t               ids_to_sorted_host_size = 0;
    for (int64_t i12 = 0; i12 < n_tokens; ++i12) {
        const char * ids_token_ptr = ids_host + i12 * ids_nb1;
        for (int64_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t expert_to_use = *(const int32_t *) (ids_token_ptr + iex * ids_nb0);
            assert(expert_to_use >= 0 && expert_to_use < n_experts);
            if (last_token_for_expert[expert_to_use] != i12) {
                last_token_for_expert[expert_to_use] = i12;
                ids_to_sorted_host_size++;
                tmp_tokens_per_expert[expert_to_use]++;
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host_size == size_t(n_tokens * n_expert_used));

    std::vector<uint32_t> tmp_expert_offsets(n_experts + 1, 0);
    for (int64_t i = 0; i < n_experts; ++i) {
        tmp_expert_offsets[i + 1] = tmp_expert_offsets[i] + tmp_tokens_per_expert[i];
    }

    std::vector<uint32_t> next_write_offset(tmp_expert_offsets.begin(), tmp_expert_offsets.end() - 1);
    for (int64_t i12 = 0; i12 < n_tokens; ++i12) {
        const char * ids_token_ptr = ids_host + i12 * ids_nb1;
        for (int64_t iex = 0; iex < n_expert_used; ++iex) {
            const int32_t  expert_to_use = *(const int32_t *) (ids_token_ptr + iex * ids_nb0);
            const uint32_t packed_row    = next_write_offset[expert_to_use]++;
            assert(packed_row < (uint32_t) (n_tokens * n_expert_used));
            ids_to_sorted_host[packed_row]   = (uint16_t) i12;
            ids_from_sorted_host[packed_row] = (uint16_t) iex;
        }
    }

    size_t active_expert_idx = 0;
    for (int64_t i = 0; i < n_experts; ++i) {
        if (tmp_tokens_per_expert[i] == 0) {
            continue;
        }
        tokens_per_expert[active_expert_idx]  = tmp_tokens_per_expert[i];
        expert_offsets[active_expert_idx]     = tmp_expert_offsets[i];
        expert_offsets[active_expert_idx + 1] = tmp_expert_offsets[i + 1];
        active_expert_idx++;
    }
    return true;
}

static bool ggml_rpp_copy_expert_forward_inputs_to_sram(rpp_kernel_context &                               kernel_ctx,
                                                        const kernel_expert_forward::fusion_runtime_plan & plan,
                                                        RPPstream                                          stream) {
    std::vector<RPPdeviceptr> src_ddr;
    std::vector<RPPdeviceptr> dst_sram;
    std::vector<size_t>       byte_count;

    auto enqueue_d2s_copy = [&](RPPdeviceptr src, RPPdeviceptr dst, size_t bytes) {
        src_ddr.push_back(src);
        dst_sram.push_back(dst);
        byte_count.push_back(bytes);
    };

    const RPPdeviceptr dev_workspace_base = kernel_ctx.dev_workspace;
    const size_t       workspace_offset   = (64 * 1024) * sizeof(uint16_t);

    auto copy_sparse_input_to_sram = [&]() {
        const size_t sparse_f32_bytes  = (size_t) plan.B * (size_t) plan.K * sizeof(float);
        const size_t sparse_bf16_bytes = (size_t) plan.B * (size_t) plan.K * sizeof(uint16_t);
        if (plan.in_bytes_per_element == kernel_expert_forward::kFloatBytes) {
            enqueue_d2s_copy(plan.inputs.sparse_act, plan.io.sparse_input, sparse_f32_bytes);
        } else {
            enqueue_d2s_copy(plan.inputs.sparse_act, plan.io.sparse_bf16, sparse_bf16_bytes);
        }
    };
    auto copy_quant_lut_to_sram = [&](RPPdeviceptr                                         dev_lut_workspace,
                                      const kernel_expert_forward::quant_lut_sram_window & lut, int quant) {
        const size_t qscale_bytes = kernel_expert_forward::quant_qscale_lut_bytes(quant);
        const size_t mag_bytes    = kernel_expert_forward::quant_mag_lut_bytes(quant);
        enqueue_d2s_copy(dev_lut_workspace, lut.qscale, qscale_bytes);
        enqueue_d2s_copy(dev_lut_workspace + qscale_bytes, lut.mag, mag_bytes);
    };
    auto copy_silu_lut_to_sram = [&](RPPdeviceptr dev_lut_workspace) {
        enqueue_d2s_copy(dev_lut_workspace, plan.io.silu_lut, kernel_expert_forward::kSiluLutBytes);
    };

    copy_sparse_input_to_sram();
    copy_quant_lut_to_sram(dev_workspace_base, plan.io.gate_lut, plan.mat0_quant);
    copy_quant_lut_to_sram(dev_workspace_base + workspace_offset, plan.io.up_lut, plan.mat1_quant);
    if (plan.has_down_stage) {
        copy_quant_lut_to_sram(dev_workspace_base + workspace_offset * 2, plan.io.down_lut, plan.mat2_quant);
    }
    copy_silu_lut_to_sram(dev_workspace_base + workspace_offset * 3);
    if (!dst_sram.empty()) {
        rppMemcpyLinkDtoSAsync(dst_sram.data(), src_ddr.data(), byte_count.data(), dst_sram.size(), stream);
    }
    return true;
}

static bool ggml_rpp_copy_expert_forward_route_to_sram(const kernel_expert_forward::fusion_runtime_plan & plan,
                                                       RPPstream                                          stream) {
    std::vector<RPPdeviceptr> src_ddr;
    std::vector<RPPdeviceptr> dst_sram;
    std::vector<size_t>       byte_count;

    auto enqueue_d2s_copy = [&](RPPdeviceptr src, RPPdeviceptr dst, size_t bytes) {
        src_ddr.push_back(src);
        dst_sram.push_back(dst);
        byte_count.push_back(bytes);
    };

    const size_t token_id_bytes = (size_t) plan.B * (size_t) kernel_expert_forward::kDefaultTopK * sizeof(uint16_t);
    const size_t routing_weights_bytes = token_id_bytes;
    enqueue_d2s_copy(plan.inputs.token_ids, plan.io.token_ids, token_id_bytes);
    if (plan.has_topk_merge) {
        enqueue_d2s_copy(plan.inputs.slot_ids, plan.io.slot_ids, token_id_bytes);
        enqueue_d2s_copy(plan.inputs.routing_weights, plan.io.routing_weights, routing_weights_bytes);
    }

    if (!dst_sram.empty()) {
        rppMemcpyLinkDtoSAsync(dst_sram.data(), src_ddr.data(), byte_count.data(), dst_sram.size(), stream);
    }
    return true;
}

static bool ggml_rpp_run_kernel_export_forward(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_export_forward *>(rpp_base_node);
    if (rpp_node->gate == nullptr || rpp_node->up == nullptr || rpp_node->down == nullptr || rpp_node->div == nullptr ||
        rpp_node->add == nullptr) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, rpp_node->gate->name,
                       ggml_op_name(rpp_node->gate->op));
        return false;
    }
    rtStream_t    stream = ctx.stream();
    ggml_tensor * ids    = rpp_node->gate->src[2];
    ggml_tensor * dst    = rpp_node->gate;
    ggml_tensor * src0   = dst->src[0];
    ggml_tensor * src1   = dst->src[1];
    ggml_tensor * src2   = dst->src[2];
    GGML_TENSOR_BINARY_OP_LOCALS;
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_experts     = ne02;
    const int64_t ids_nb0       = ids->nb[0];
    const int64_t ids_nb1       = ids->nb[1];

    if (ggml_rpp_export_forward_seq_len(rpp_node->gate, rpp_node) == 1) {
        auto & kernel_ctx_gate = *(rpp_node->kernel_ctx.get());
        RPP_CHECK(rppGraphLaunch(kernel_ctx_gate.graphexec, stream));
        return true;
    }

    ggml_rpp_pool_alloc<char> ids_host(ctx.pool_host_leg(), ggml_nbytes(ids));
    RPP_CHECK(rtMemcpyAsync(ids_host.ptr, ids->data, ggml_nbytes(ids), rtMemcpyDeviceToHost, stream));
    RPP_CHECK(rtStreamSynchronize(stream));

    // copy inputs to sram
    auto & kernel_ctx      = *(rpp_node->kernel_ctx.get());
    auto & kernel_ctx_next = *(rpp_node->kernel_ctx_next.get());

    const kernel_expert_forward::fusion_runtime_plan & plan       = rpp_node->plan;
    const kernel_expert_forward::fusion_runtime_plan & plan_next  = rpp_node->plan_next;
    const int                                          final_cols = plan.has_down_stage ? plan.K : plan.N;
    if (!ggml_rpp_copy_expert_forward_inputs_to_sram(kernel_ctx, plan, stream)) {
        return false;
    }
    // get route info
    std::vector<uint32_t> tmp_tokens_per_expert;
    if (!ggml_rpp_get_expert_forward_route_info(rpp_node, ids_host.ptr, ne12, n_expert_used, n_experts, ids_nb0,
                                                ids_nb1, tmp_tokens_per_expert)) {
        return false;
    }
    // copy route info to device and sram
    const size_t expert_info_size = rpp_node->token_ids_size + rpp_node->expert_counts_size +
                                    rpp_node->expert_offsets_size + rpp_node->topk_slots_size;
    RPP_CHECK(rtMemcpyAsync((void *) rpp_node->dev_expert_info, rpp_node->host_expert_info, expert_info_size,
                            rtMemcpyHostToDevice, stream));
    if (!ggml_rpp_copy_expert_forward_route_to_sram(plan, stream)) {
        return false;
    }
    // launch experts input graph
    if (plan.graphs.experts_input_exec != 0) {
        RPP_CHECK(rppGraphLaunch(plan.graphs.experts_input_exec, stream));
    }
    size_t single_expert_exec_count = 0;
    for (size_t expert_id = 0; expert_id < tmp_tokens_per_expert.size(); ++expert_id) {
        const uint32_t token_count = tmp_tokens_per_expert[expert_id];
        if (!token_count) {
            continue;
        }
        assert(token_count <= (uint32_t) ctx.n_ubatch &&
               "matmul_id_fusion graph mode requires token_count <= 255 per expert; "
               "future work: chunk oversized experts into multiple launches");
        if (token_count > (uint32_t) ctx.n_ubatch) {
            throw std::runtime_error(
                "matmul_id_fusion graph mode requires token_count <= 255 per expert; "
                "future work: chunk oversized experts into multiple launches");
        }

        if (single_expert_exec_count % 2 == 0) {
            preload_single_expert_weights(kernel_ctx, plan, (int) expert_id, stream);
            RPP_CHECK(rppGraphLaunch(plan.graphs.single_expert_exec, stream));
        } else {
            preload_single_expert_weights(kernel_ctx_next, plan_next, (int) expert_id, stream);
            RPP_CHECK(rppGraphLaunch(plan_next.graphs.single_expert_exec, stream));
        }
        single_expert_exec_count++;
    }
    if (plan.graphs.experts_output_exec != 0) {
        RPP_CHECK(rppGraphLaunch(plan.graphs.experts_output_exec, stream));
    }
    const uint64_t     bytes    = (uint64_t) plan.B * (uint64_t) final_cols * (uint64_t) plan.out_bytes_per_element;
    const RPPdeviceptr sram_out = (plan.out_bytes_per_element == kernel_expert_forward::kFloatBytes) ?
                                      plan.io.final_chw_out :
                                      plan.io.merged_chw_bf16;
    RPP_CHECK(rtMemcpyAsync((void *) rpp_node->add->data, (const void *) sram_out, (size_t) bytes, rtMemcpySramToDevice,
                            stream));
    return true;
}

static bool ggml_rpp_create_kernel_dispatch_export_forward(ggml_backend_rpp_context & ctx,
                                                           ggml_rpp_node *            rpp_base_node,
                                                           ggml_tensor *              gate,
                                                           ggml_tensor *              up,
                                                           ggml_tensor *              down,
                                                           ggml_tensor *              div,
                                                           ggml_tensor *              add) {
    GGML_ASSERT(rpp_base_node);
    auto          rpp_node = static_cast<rpp_kernel_export_forward *>(rpp_base_node);
    bool          ret      = false;
    ggml_tensor * dst      = gate;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 2 ? 2 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    const int seq_len = ggml_rpp_export_forward_seq_len(dst, rpp_node);
    if (ctx.use_ubatch && seq_len > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }
    if (seq_len > 1) {
        ret = ggml_rpp_create_kernel_export_forward(ctx, rpp_node, gate, up, down, div, add);
    } else {
        ret = ggml_rpp_create_kernel_export_forward_vxm(ctx, rpp_node, gate, up, down, div, add);
    }
    GGML_ASSERT(ret);
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        ggml_rpp_node_set_properties(rpp_node, up);
        ggml_rpp_node_set_properties(rpp_node, down);
        ggml_rpp_node_set_properties(rpp_node, div);
        ggml_rpp_node_set_properties(rpp_node, add);
    }
    return ret;
}

bool ggml_rpp_op_kernel_export_forward(ggml_backend_rpp_context & ctx,
                                       ggml_tensor *              gate,
                                       ggml_tensor *              up,
                                       ggml_tensor *              down,
                                       ggml_tensor *              div,
                                       ggml_tensor *              add,
                                       int                        is_instantial,
                                       int                        is_launch) {
    ggml_tensor * dst = gate;
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_export_forward * rpp_node = nullptr;
    auto                        iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_rms_norm_mul_fusion");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    cur_node->op == ggml_rpp_node::RPP_OP_EXPERT_FORWARD &&
                    ggml_rpp_export_forward_properties_is_same(dst, cur_node)) {
                    rpp_node = (rpp_kernel_export_forward *) cur_node;
                    break;
                }
            }
        }
        const int seq_len = dst->ne[2];
        if (!rpp_node && seq_len > 1) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_export_forward_shared");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                if (!rpp_graph_tmp) {
                    continue;
                }
                for (auto & node_iter_other : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter_other.first == dst) {
                        continue;
                    }
                    if (node_iter_other.first->op != dst->op || !ggml_rpp_dims_is_same(node_iter_other.first, dst)) {
                        continue;
                    }
                    auto & node_vec = node_iter_other.second;
                    for (size_t i = 0; i < node_vec.size(); ++i) {
                        auto cur_node = node_vec[i].get();
                        if (cur_node->rpp_type != ggml_rpp_node::RPP_NODE_TYPE_KERNEL ||
                            cur_node->op != ggml_rpp_node::RPP_OP_EXPERT_FORWARD ||
                            !ggml_rpp_export_forward_dims_is_same(gate, up, down, div, add, cur_node)) {
                            continue;
                        }
                        auto new_node = std::make_unique<rpp_kernel_export_forward>(gate, cur_node, up, down, div, add);
                        ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                        rpp_node =
                            static_cast<rpp_kernel_export_forward *>(ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                        ggml_rpp_node_set_properties(rpp_node, dst);
                        ggml_rpp_node_set_properties(rpp_node, up);
                        ggml_rpp_node_set_properties(rpp_node, down);
                        ggml_rpp_node_set_properties(rpp_node, div);
                        ggml_rpp_node_set_properties(rpp_node, add);
                        ggml_rpp_update_expert_forward_plan_inputs(rpp_node, rpp_node->plan, gate, up, down, div, add);
                        ggml_rpp_update_expert_forward_plan_inputs(rpp_node, rpp_node->plan_next, gate, up, down, div,
                                                                   add);
                        rpp_node->n_ubatch      = cur_node->n_ubatch;
                        rpp_node->seq_len_index = cur_node->seq_len_index;
                        rpp_node->is_instantial = is_instantial;
                        break;
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
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_export_forward");
            auto new_node = std::make_unique<rpp_kernel_export_forward>(gate, up, down, div, add);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_export_forward *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch_export_forward(ctx, rpp_node, gate, up, down, div, add))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_export_forward *) (iter->second);
    }

    bool ret = true;
    if (is_launch) {
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_export_forward");
            ret = ggml_rpp_run_kernel_export_forward(ctx, rpp_node);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return ret;
}
