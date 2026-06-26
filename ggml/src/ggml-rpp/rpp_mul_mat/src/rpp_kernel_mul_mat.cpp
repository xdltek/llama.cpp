#include "rpp_mul_mat/kernel_bf16/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_s/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_s_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_s_vxm/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_s_vxm_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_xs/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_xs_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_xs_vxm/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_xs_vxm_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q3_xxs/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q3_xxs_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q3_xxs_vxm/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q3_xxs_vxm_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q4_1/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q4_1_vxm/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q4_k/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q4_k_vxm/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q5_k/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q5_k_vxm/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q6_k/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q6_k_vxm/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q8_0/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q8_0_vxm/rpp_kernel_build.h"
#include "rpp_mul_mat/rpp_mul_mat.h"

static int ggml_rpp_mat_mul_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static ggml_rpp_node * ggml_rpp_find_mul_mat_node(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_MUL_MAT &&
                node_iter.first->src[0]->type == dst->src[0]->type) {
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

static bool ggml_rpp_mat_mul_properties_is_same(ggml_backend_rpp_context & ctx,
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

static bool ggml_rpp_create_kernel_q4_1(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    // notice!!!  requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    // get B M N G
    const int seq_len       = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M             = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K             = dst->src[0]->ne[0];
    const int N             = dst->src[0]->ne[1];
    const int weights_group = 32;                 // now is const,32
    const int G             = N / weights_group;  // of Q4_1 blocks per row
    const int CK            = K / 4;              // of uint16 packs along K

    size_t q4_weights_size = CK * N * sizeof(uint16_t);
    size_t q4_scales_size  = K * G * sizeof(uint16_t);
    size_t q4_zeros_size   = K * G * sizeof(uint16_t);
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q4_weights_size + q4_scales_size + q4_zeros_size);

    // kernel inputs
    RPPdeviceptr inputs_0_buf   = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q4_weights_buf = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q4_scales_buf  = (RPPdeviceptr) dst->src[0]->data + q4_weights_size;
    RPPdeviceptr q4_zeros_buf   = (RPPdeviceptr) dst->src[0]->data + q4_weights_size + q4_scales_size;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_weights_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_scales_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_zeros_buf);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build q4_1 kernel
    if (M == 1) {
        kernel_q4_1_vxm::rpp_matmul_q4_1_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, weights_group, i_type_size,
                                                   o_type_size, 1, rpp_node->is_instantial);
    } else {
        kernel_q4_1::rpp_matmul_q4_1_build(*(rpp_node->kernel_ctx.get()), M, K, N, weights_group, i_type_size,
                                           o_type_size, 1, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_q8_0(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    // notice!!!  requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    // get B M N G
    const int seq_len       = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M             = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K             = dst->src[0]->ne[0];
    const int N             = dst->src[0]->ne[1];
    const int weights_group = 32;  // now is const,32

    size_t q8_weights_size = K / 2 * N * sizeof(uint16_t);
    size_t q8_scales_size  = K / weights_group * N * sizeof(uint16_t);
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q8_weights_size + q8_scales_size);

    // kernel inputs
    RPPdeviceptr inputs_0_buf   = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q8_weights_buf = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q8_scales_buf  = (RPPdeviceptr) dst->src[0]->data + q8_weights_size;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q8_weights_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q8_scales_buf);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build q8_0 kernel
    if (M == 1) {
        kernel_q8_0_vxm::rpp_matmul_q80_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, weights_group, i_type_size,
                                                  o_type_size, 1, rpp_node->is_instantial);
    } else {
        kernel_q8_0::rpp_matmul_q80_build(*(rpp_node->kernel_ctx.get()), M, K, N, weights_group, i_type_size,
                                          o_type_size, 1, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_q6_k(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    // notice!!!  requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    // get B M N G
    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    size_t q6_weights_ql_size   = K / 4 * N * sizeof(uint16_t);
    size_t q6_weights_qh_size   = K / 8 * N * sizeof(uint16_t);
    size_t q6_scales_size       = K / 32 * N * sizeof(uint16_t);
    size_t q6_super_scales_size = K / QK_K * N * sizeof(uint16_t);
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N ==
                q6_weights_ql_size + q6_weights_qh_size + q6_scales_size + q6_super_scales_size);

    // kernel inputs
    RPPdeviceptr inputs_0_buf        = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q6_weights_ql_buf   = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q6_weights_qh_buf   = q6_weights_ql_buf + q6_weights_ql_size;
    RPPdeviceptr q6_scales_buf       = q6_weights_qh_buf + q6_weights_qh_size;
    RPPdeviceptr q6_super_scales_buf = q6_scales_buf + q6_scales_size;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q6_weights_ql_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q6_weights_qh_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q6_scales_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q6_super_scales_buf);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build q6_k kernel
    if (M == 1) {
        kernel_q6_k_vxm::rpp_matmul_q6k_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                  o_type_size, 1, rpp_node->is_instantial);
    } else {
        kernel_q6_k::rpp_matmul_q6k_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size, o_type_size, 1,
                                          rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_q4_k(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    // notice!!! requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    const size_t q4_weights_q_size   = (size_t) K / 4 * (size_t) N * sizeof(uint16_t);
    const size_t q4_scale_lsb_size   = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q4_zero_lsb_size    = q4_scale_lsb_size;
    const size_t q4_scale_msb_size   = (size_t) K / 256 * (size_t) N * sizeof(uint16_t);
    const size_t q4_zero_msb_size    = q4_scale_msb_size;
    const size_t q4_super_scale_size = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q4_super_zero_size  = q4_super_scale_size;
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q4_weights_q_size + q4_scale_lsb_size + q4_zero_lsb_size +
                                                               q4_scale_msb_size + q4_zero_msb_size +
                                                               q4_super_scale_size + q4_super_zero_size);

    // kernel inputs
    RPPdeviceptr inputs_0_buf       = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q4_weights_q_buf   = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q4_scale_lsb_buf   = q4_weights_q_buf + q4_weights_q_size;
    RPPdeviceptr q4_zero_lsb_buf    = q4_scale_lsb_buf + q4_scale_lsb_size;
    RPPdeviceptr q4_scale_msb_buf   = q4_zero_lsb_buf + q4_zero_lsb_size;
    RPPdeviceptr q4_zero_msb_buf    = q4_scale_msb_buf + q4_scale_msb_size;
    RPPdeviceptr q4_super_scale_buf = q4_zero_msb_buf + q4_zero_msb_size;
    RPPdeviceptr q4_super_zero_buf  = q4_super_scale_buf + q4_super_scale_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_weights_q_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_scale_lsb_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_zero_lsb_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_scale_msb_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_zero_msb_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_super_scale_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q4_super_zero_buf);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build q4_k kernel
    if (M == 1) {
        kernel_q4_k_vxm::rpp_matmul_q4k_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                  o_type_size, 1, rpp_node->is_instantial);
    } else {
        kernel_q4_k::rpp_matmul_q4k_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size, o_type_size, 1,
                                          rpp_node->is_instantial);
    }

    return true;
}

static bool ggml_rpp_create_kernel_q5_k(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    const size_t q5_weights_lsb_size   = (size_t) K / 4 * (size_t) N * sizeof(uint16_t);
    const size_t q5_weights_msb_size   = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q5_scale_lsb_size     = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q5_zero_lsb_size      = q5_scale_lsb_size;
    const size_t q5_scale_msb_size     = (size_t) K / 256 * (size_t) N * sizeof(uint16_t);
    const size_t q5_zero_msb_size      = q5_scale_msb_size;
    const size_t q5_super_scale_size   = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q5_super_zero_size    = q5_super_scale_size;
    const size_t q5_transformed_nbytes = q5_weights_lsb_size + q5_weights_msb_size + q5_scale_lsb_size +
                                         q5_zero_lsb_size + q5_scale_msb_size + q5_zero_msb_size + q5_super_scale_size +
                                         q5_super_zero_size;
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q5_transformed_nbytes);

    // kernel inputs
    RPPdeviceptr inputs_0_buf   = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q5_weights_lsb = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q5_weights_msb = q5_weights_lsb + q5_weights_lsb_size;
    RPPdeviceptr q5_scale_lsb   = q5_weights_msb + q5_weights_msb_size;
    RPPdeviceptr q5_zero_lsb    = q5_scale_lsb + q5_scale_lsb_size;
    RPPdeviceptr q5_scale_msb   = q5_zero_lsb + q5_zero_lsb_size;
    RPPdeviceptr q5_zero_msb    = q5_scale_msb + q5_scale_msb_size;
    RPPdeviceptr q5_super_scale = q5_zero_msb + q5_zero_msb_size;
    RPPdeviceptr q5_super_zero  = q5_super_scale + q5_super_scale_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_weights_lsb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_weights_msb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_scale_lsb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_zero_lsb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_scale_msb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_zero_msb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_super_scale);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_super_zero);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q5_k_vxm::rpp_matmul_q5k_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                  o_type_size, 1, rpp_node->is_instantial);
    } else {
        kernel_q5_k::rpp_matmul_q5k_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size, o_type_size, 1,
                                          rpp_node->is_instantial);
    }

    return true;
}

static bool ggml_rpp_create_kernel_iq3_xxs(ggml_backend_rpp_context & ctx,
                                           ggml_rpp_node *            rpp_base_node,
                                           ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    const size_t q3_codebook_size    = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q3_scales_size      = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q3_sign_size        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q3_super_scale_size = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);

    // kernel inputs
    RPPdeviceptr inputs_0_buf   = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q3_codebook    = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q3_scales      = q3_codebook + q3_codebook_size;
    RPPdeviceptr q3_sign        = q3_scales + q3_scales_size;
    RPPdeviceptr q3_super_scale = q3_sign + q3_sign_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_codebook);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_super_scale);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q3_xxs_vxm::rpp_matmul_q3xxs_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                      o_type_size, 1, 0, 0, rpp_node->is_instantial);
    } else {
        kernel_q3_xxs::rpp_matmul_q3xxs_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size, o_type_size,
                                              rpp_node->is_instantial);
    }

    return true;
}

static bool ggml_rpp_create_kernel_iq3_xxs_nolut(ggml_backend_rpp_context & ctx,
                                                 ggml_rpp_node *            rpp_base_node,
                                                 ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    const size_t q3_codebook_size    = (size_t) (K / 16) * 3u * (size_t) N * sizeof(uint16_t);
    const size_t q3_scales_size      = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q3_sign_size        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q3_super_scale_size = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);

    // kernel inputs
    RPPdeviceptr inputs_0_buf   = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q3_codebook    = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q3_scales      = q3_codebook + q3_codebook_size;
    RPPdeviceptr q3_sign        = q3_scales + q3_scales_size;
    RPPdeviceptr q3_super_scale = q3_sign + q3_sign_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_codebook);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_super_scale);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q3_xxs_vxm_nolut::rpp_matmul_q3xxs_vxm_nolut_build(
            *(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size, o_type_size, 1, 0, 0, rpp_node->is_instantial);
    } else {
        kernel_q3_xxs_nolut::rpp_matmul_q3xxs_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                          o_type_size, 0, rpp_node->is_instantial);
    }

    return true;
}

static bool ggml_rpp_create_kernel_iq2_s(ggml_backend_rpp_context & ctx,
                                         ggml_rpp_node *            rpp_base_node,
                                         ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    const size_t q2_codebook_lsb_size = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_codebook_msb_size = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_size       = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_sign_size         = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_size  = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_nbytes =
        q2_codebook_lsb_size + q2_codebook_msb_size + q2_scales_size + q2_sign_size + q2_super_scale_size;
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q2_transformed_nbytes);

    // kernel inputs
    RPPdeviceptr inputs_0_buf    = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q2_codebook_lsb = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q2_codebook_msb = q2_codebook_lsb + q2_codebook_lsb_size;
    RPPdeviceptr q2_scales       = q2_codebook_msb + q2_codebook_msb_size;
    RPPdeviceptr q2_sign         = q2_scales + q2_scales_size;
    RPPdeviceptr q2_super_scale  = q2_sign + q2_sign_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_codebook_lsb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_codebook_msb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_super_scale);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);

    if (M == 1) {
        kernel_q2_s_vxm::rpp_matmul_q2s_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                  o_type_size, 1, rpp_node->is_instantial);
    } else {
        kernel_q2_s::rpp_matmul_q2s_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size, o_type_size,
                                          rpp_node->is_instantial);
    }

    return true;
}

static bool ggml_rpp_create_kernel_iq2_s_nolut(ggml_backend_rpp_context & ctx,
                                               ggml_rpp_node *            rpp_base_node,
                                               ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    const size_t q2_codebook_nolut_size = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_size         = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_sign_size           = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_size    = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_nbytes  = q2_codebook_nolut_size + q2_scales_size + q2_sign_size + q2_super_scale_size;
    const int    q2_weights_group       = QK_K;

    RPPdeviceptr inputs_0_buf      = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q2_codebook_nolut = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q2_scales         = q2_codebook_nolut + q2_codebook_nolut_size;
    RPPdeviceptr q2_sign           = q2_scales + q2_scales_size;
    RPPdeviceptr q2_super_scale    = q2_sign + q2_sign_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_codebook_nolut);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_super_scale);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q2_s_vxm_nolut::rpp_matmul_q2s_vxm_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                              i_type_size, o_type_size, 1, 0, 0,
                                                              rpp_node->is_instantial);
    } else {
        kernel_q2_s_nolut::rpp_matmul_q2s_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                      i_type_size, o_type_size, 0, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_iq2_xs(ggml_backend_rpp_context & ctx,
                                          ggml_rpp_node *            rpp_base_node,
                                          ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    const size_t q2_qs_size            = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_size        = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_size   = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_nbytes = q2_qs_size + q2_scales_size + q2_super_scale_size;
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q2_transformed_nbytes);

    RPPdeviceptr inputs_0_buf   = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q2_qs          = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q2_scales      = q2_qs + q2_qs_size;
    RPPdeviceptr q2_super_scale = q2_scales + q2_scales_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_qs);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_super_scale);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);

    if (M == 1) {
        kernel_q2_xs_vxm::rpp_matmul_q2xs_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                    o_type_size, 1, rpp_node->is_instantial);
    } else {
        kernel_q2_xs::rpp_matmul_q2xs_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size, o_type_size,
                                            rpp_node->is_instantial);
    }

    return true;
}

static bool ggml_rpp_create_kernel_iq2_xs_nolut(ggml_backend_rpp_context & ctx,
                                                ggml_rpp_node *            rpp_base_node,
                                                ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];

    const size_t q2_codebook_nolut_size = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_size         = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_sign_size           = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_size    = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_nbytes  = q2_codebook_nolut_size + q2_scales_size + q2_sign_size + q2_super_scale_size;
    const int    q2_weights_group       = QK_K;

    RPPdeviceptr inputs_0_buf      = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr q2_codebook_nolut = (RPPdeviceptr) dst->src[0]->data;
    RPPdeviceptr q2_scales         = q2_codebook_nolut + q2_codebook_nolut_size;
    RPPdeviceptr q2_sign           = q2_scales + q2_scales_size;
    RPPdeviceptr q2_super_scale    = q2_sign + q2_sign_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_codebook_nolut);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_super_scale);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q2_xs_vxm_nolut::rpp_matmul_q2xs_vxm_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N,
                                                                q2_weights_group, i_type_size, o_type_size, 1, 0, 0,
                                                                rpp_node->is_instantial);
    } else {
        kernel_q2_xs_nolut::rpp_matmul_q2xs_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                        i_type_size, o_type_size, 0, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_bf16(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    // notice!!!  requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 3);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 3);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    // get B M N G
    const int seq_len = ggml_rpp_mat_mul_seq_len(dst, rpp_node);
    const int M       = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[1];
    const int K       = dst->src[0]->ne[0];
    const int N       = dst->src[0]->ne[1];
    const int w_group = 32;

    // kernel inputs
    RPPdeviceptr inputs_0_buf     = (RPPdeviceptr) dst->src[1]->data;
    RPPdeviceptr bf16_weights_buf = (RPPdeviceptr) dst->src[0]->data;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) bf16_weights_buf);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build q6_k kernel
    kernel_bf16::rpp_matmul_bf16_build(*(rpp_node->kernel_ctx.get()), M, K, N, w_group, i_type_size, o_type_size,
                                       rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_mat_mul_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }
    // create kernel according to type, now only support q4_1 for kernel model
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            ret = ggml_rpp_create_kernel_bf16(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q4_1:
            ret = ggml_rpp_create_kernel_q4_1(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q8_0:
            ret = ggml_rpp_create_kernel_q8_0(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q6_K:
            ret = ggml_rpp_create_kernel_q6_k(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q4_K:
            ret = ggml_rpp_create_kernel_q4_k(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q5_K:
            ret = ggml_rpp_create_kernel_q5_k(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_IQ3_XXS:
            // ret = ggml_rpp_create_kernel_iq3_xxs(ctx, rpp_node, dst);
            ret = ggml_rpp_create_kernel_iq3_xxs_nolut(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_IQ2_S:
            // ret = ggml_rpp_create_kernel_iq2_s(ctx, rpp_node, dst);
            ret = ggml_rpp_create_kernel_iq2_s_nolut(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_IQ2_XS:
            // ret = ggml_rpp_create_kernel_iq2_xs(ctx, rpp_node, dst);
            ret = ggml_rpp_create_kernel_iq2_xs_nolut(ctx, rpp_node, dst);
            break;
        default:
            break;
    }
    GGML_ASSERT(ret);
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

bool ggml_rpp_op_kernel_mul_mat(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    // void * src1_data = malloc(ggml_nbytes(dst->src[1]));
    // RPP_MEMCPY_DEV_AND_HOST(src1_data, dst->src[1]->data, ggml_nbytes(dst->src[1]), rtMemcpyDeviceToHost, ctx.stream(),
    //                         1);
    // char src1_name[128];
    // // sprintf(src1_name, "./cpu_data/%s_src1.bin", dst->src[0]->name);
    // sprintf(src1_name, "./cpu_data/%s_src1.bin", "blk.0.attn_q.weight");
    // float mse_src = calculate_mse(src1_name, (float *) src1_data, ggml_nelements(dst->src[1]));
    // printf("xxxxxxxxxxxxxxxxxxxxxx %s mse: %f\n", src1_name, mse_src);
    // free(src1_data);
    rpp_kernel_mul_mat * rpp_node = nullptr;
    auto                 iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_mul_mat");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_mat_mul_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_mul_mat *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_mul_mat");
            auto new_node = std::make_unique<rpp_kernel_mul_mat>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_mul_mat *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_mul_mat *) (iter->second);
    }

    if (is_launch) {
        // compute add operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_mul_mat");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }

    // void * dst_data = malloc(ggml_nbytes(dst));
    // RPP_MEMCPY_DEV_AND_HOST(dst_data, dst->data, ggml_nbytes(dst), rtMemcpyDeviceToHost, ctx.stream(), 1);
    // char dst_name[128];
    // // sprintf(dst_name, "./cpu_data/%s_dst.bin", dst->src[0]->name);
    // sprintf(dst_name, "./cpu_data/%s_dst.bin", "blk.0.attn_q.weight");
    // float mse_dst = calculate_mse(dst_name, (float *) dst_data, ggml_nelements(dst));
    // printf("xxxxxxxxxxxxxxxxxxxxxx %s mse: %f\n", dst_name, mse_dst);
    // free(dst_data);
    return true;
}
