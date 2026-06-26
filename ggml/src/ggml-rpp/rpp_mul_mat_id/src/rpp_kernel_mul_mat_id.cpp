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
#include "rpp_mul_mat_id/rpp_mul_mat_id.h"

enum kernel_type {
    NORMAL,
    LUT,
    LUT_SRAM,
    NOLUT,
    NOLUT_SRAM,
};

static int ggml_rpp_mat_mul_id_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_mat_mul_id_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
    bool          dims_is_same              = true;
    ggml_tensor * dst_vec[GGML_MAX_SRC + 1] = { dst };
    auto          dst_iter =
        std::copy_if(dst->src, dst->src + GGML_MAX_SRC, dst_vec + 1, [](ggml_tensor * ptr) { return ptr != nullptr; });

    ggml_tensor * src_vec[GGML_MAX_SRC + 1] = { src };
    auto          src_iter =
        std::copy_if(src->src, src->src + GGML_MAX_SRC, src_vec + 1, [](ggml_tensor * ptr) { return ptr != nullptr; });
    int dst_size = dst_iter - dst_vec;
    int src_size = src_iter - src_vec;
    if (dst_size != src_size || rpp_node->ggml_node_properties.size() != src_size) {
        return false;
    }
    for (int i = 0; i < dst_size; i++) {
        auto & property = rpp_node->ggml_node_properties[src_vec[i]];
        auto & cur_dst  = dst_vec[i];
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (cur_dst->ne[i] != property.ne[i]) {
                dims_is_same = false;
                break;
            }
        }
        if (!dims_is_same) {
            break;
        }
    }
    return dims_is_same;
}

static void ggml_rpp_get_rows_cpy(const void *    src,
                                  const int32_t * ids_host,
                                  void *          dst,
                                  int64_t         nrows,
                                  size_t          src_row_size,
                                  size_t          dst_row_size,
                                  rtStream_t      stream) {
    GGML_ASSERT(src_row_size == dst_row_size);
    const char * src_bytes = static_cast<const char *>(src);
    char *       dst_bytes = static_cast<char *>(dst);

    for (int64_t i = 0; i < nrows; ++i) {
        const int32_t row = ids_host[i];
        RPP_CHECK(rtMemcpyAsync(dst_bytes + i * dst_row_size, src_bytes + row * src_row_size, src_row_size,
                                rtMemcpyDeviceToDevice, stream));
    }
    // RPP_CHECK(rppMemcpyLinkDtoDAsync(...));
}

static ggml_rpp_node * ggml_rpp_find_mul_mat_id_node(ggml_backend_rpp_context & ctx,
                                                     ggml_rpp_node *            rpp_node,
                                                     ggml_tensor *              dst) {
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_MUL_MAT_ID &&
                node_iter.first->src[0]->type == dst->src[0]->type) {
                auto & node_vec = node_iter.second;
                for (size_t i = 0; i < node_vec.size(); i++) {
                    auto cur_node = node_vec[i].get();
                    // beacuse of this type prefill and decode kernel share the not same workspace, so we need to check the n_ubatch
                    if (dst->src[0]->type == GGML_TYPE_IQ2_XS || dst->src[0]->type == GGML_TYPE_IQ2_S ||
                        dst->src[0]->type == GGML_TYPE_IQ3_XXS) {
                        auto cur_rpp_node = static_cast<rpp_kernel_mul_mat_id *>(cur_node);
                        if (rpp_node->n_ubatch == cur_rpp_node->n_ubatch) {
                            return cur_node;
                        }
                    } else {
                        return cur_node;
                    }
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
    return true;
}

static bool ggml_rpp_create_kernel_q4_1(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    // notice!!!  requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[2]));

    // get B M N G
    const int seq_len       = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert      = dst->src[2]->ne[0];
    const int M             = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K             = dst->src[0]->ne[0];
    const int N             = dst->src[0]->ne[1];
    const int weights_group = 32;                 // now is const,32
    const int G             = N / weights_group;  // of Q4_1 blocks per row
    const int CK            = K / 4;              // of uint16 packs along K

    size_t inputs_0_size   = M * K * ggml_type_size(dst->src[1]->type);
    size_t outputs_size    = M * N * n_expert * ggml_type_size(dst->type);
    size_t q4_weights_size = CK * N * sizeof(uint16_t);
    size_t q4_scales_size  = K * G * sizeof(uint16_t);
    size_t q4_zeros_size   = K * G * sizeof(uint16_t);
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q4_weights_size + q4_scales_size + q4_zeros_size);

    // kernel inputs
    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q4_weights_size + q4_scales_size + q4_zeros_size);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) weights_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) weights_buf + q4_weights_size);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) weights_buf + q4_weights_size + q4_scales_size);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], (void *) inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], (void *) weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back((void *) inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back((void *) weights_buf);
    rpp_node->binding_io_buffers.emplace_back((void *) outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    // find first rms_node, and all mul_mat_id kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
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
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    // notice!!!  requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[2]));

    // get B M N G
    const int seq_len       = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert      = dst->src[2]->ne[0];
    const int M             = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K             = dst->src[0]->ne[0];
    const int N             = dst->src[0]->ne[1];
    const int weights_group = 32;  // now is const,32

    size_t inputs_0_size   = M * K * ggml_type_size(dst->src[1]->type);
    size_t outputs_size    = M * N * n_expert * ggml_type_size(dst->type);
    size_t q8_weights_size = K / 2 * N * sizeof(uint16_t);
    size_t q8_scales_size  = K / weights_group * N * sizeof(uint16_t);
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q8_weights_size + q8_scales_size);

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q8_weights_size + q8_scales_size);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);
    // kernel inputs
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) weights_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) weights_buf + q8_weights_size);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    // find first rms_node, and all mul_mat_id kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
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
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    // notice!!!  requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[2]));

    // get B M N G
    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    size_t inputs_0_size        = M * K * ggml_type_size(dst->src[1]->type);
    size_t outputs_size         = M * N * n_expert * ggml_type_size(dst->type);
    size_t q6_weights_ql_size   = K / 4 * N * sizeof(uint16_t);
    size_t q6_weights_qh_size   = K / 8 * N * sizeof(uint16_t);
    size_t q6_scales_size       = K / 32 * N * sizeof(uint16_t);
    size_t q6_super_scales_size = K / QK_K * N * sizeof(uint16_t);
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N ==
                q6_weights_ql_size + q6_weights_qh_size + q6_scales_size + q6_super_scales_size);

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf =
        ctx.pool().alloc(q6_weights_ql_size + q6_weights_qh_size + q6_scales_size + q6_super_scales_size);
    void *       outputs_buf         = ctx.pool().alloc(outputs_size);
    // kernel inputs
    RPPdeviceptr q6_weights_ql_buf   = (RPPdeviceptr) weights_buf;
    RPPdeviceptr q6_weights_qh_buf   = (RPPdeviceptr) (q6_weights_ql_buf + q6_weights_ql_size);
    RPPdeviceptr q6_scales_buf       = (RPPdeviceptr) (q6_weights_qh_buf + q6_weights_qh_size);
    RPPdeviceptr q6_super_scales_buf = (RPPdeviceptr) (q6_scales_buf + q6_scales_size);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q6_weights_ql_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q6_weights_qh_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q6_scales_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q6_super_scales_buf);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    // find first rms_node, and all mul_mat_id kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
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
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[2]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    const size_t inputs_0_size       = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size        = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q4_weights_q_size   = (size_t) K / 4 * (size_t) N * sizeof(uint16_t);
    const size_t q4_scale_lsb_size   = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q4_zero_lsb_size    = q4_scale_lsb_size;
    const size_t q4_scale_msb_size   = (size_t) K / 256 * (size_t) N * sizeof(uint16_t);
    const size_t q4_zero_msb_size    = q4_scale_msb_size;
    const size_t q4_super_scale_size = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q4_super_zero_size  = q4_super_scale_size;
    const size_t q4_transformed_size = q4_weights_q_size + q4_scale_lsb_size + q4_zero_lsb_size + q4_scale_msb_size +
                                       q4_zero_msb_size + q4_super_scale_size + q4_super_zero_size;
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q4_transformed_size);

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q4_transformed_size);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);

    RPPdeviceptr q4_weights_q_buf   = (RPPdeviceptr) weights_buf;
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
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);

    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
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
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[2]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    const size_t inputs_0_size     = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size      = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q5_weights_lsb_sz = (size_t) K / 4 * (size_t) N * sizeof(uint16_t);
    const size_t q5_weights_msb_sz = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q5_scale_lsb_sz   = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q5_zero_lsb_sz    = q5_scale_lsb_sz;
    const size_t q5_scale_msb_sz   = (size_t) K / 256 * (size_t) N * sizeof(uint16_t);
    const size_t q5_zero_msb_sz    = q5_scale_msb_sz;
    const size_t q5_super_scale_sz = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q5_super_zero_sz  = q5_super_scale_sz;
    const size_t q5_transformed_sz = q5_weights_lsb_sz + q5_weights_msb_sz + q5_scale_lsb_sz + q5_zero_lsb_sz +
                                     q5_scale_msb_sz + q5_zero_msb_sz + q5_super_scale_sz + q5_super_zero_sz;
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q5_transformed_sz);

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q5_transformed_sz);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);

    RPPdeviceptr q5_weights_lsb = (RPPdeviceptr) weights_buf;
    RPPdeviceptr q5_weights_msb = q5_weights_lsb + q5_weights_lsb_sz;
    RPPdeviceptr q5_scale_lsb   = q5_weights_msb + q5_weights_msb_sz;
    RPPdeviceptr q5_zero_lsb    = q5_scale_lsb + q5_scale_lsb_sz;
    RPPdeviceptr q5_scale_msb   = q5_zero_lsb + q5_zero_lsb_sz;
    RPPdeviceptr q5_zero_msb    = q5_scale_msb + q5_scale_msb_sz;
    RPPdeviceptr q5_super_scale = q5_zero_msb + q5_zero_msb_sz;
    RPPdeviceptr q5_super_zero  = q5_super_scale + q5_super_scale_sz;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_weights_lsb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_weights_msb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_scale_lsb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_zero_lsb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_scale_msb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_zero_msb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_super_scale);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q5_super_zero);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);

    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
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

static bool ggml_rpp_create_kernel_iq3_xxs_lut(ggml_backend_rpp_context & ctx,
                                               ggml_rpp_node *            rpp_base_node,
                                               ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    // GGML_ASSERT(ggml_is_contiguous(dst->src[2]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    const size_t inputs_0_size       = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size        = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q3_codebook_size    = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q3_scales_size      = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q3_sign_size        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q3_super_scale_size = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q3_transformed_size = q3_codebook_size + q3_scales_size + q3_sign_size + q3_super_scale_size;
    // GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q3_transformed_size);

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q3_transformed_size);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);

    RPPdeviceptr q3_codebook    = (RPPdeviceptr) weights_buf;
    RPPdeviceptr q3_scales      = q3_codebook + q3_codebook_size;
    RPPdeviceptr q3_sign        = q3_scales + q3_scales_size;
    RPPdeviceptr q3_super_scale = q3_sign + q3_sign_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_codebook);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_super_scale);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);

    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q3_xxs_vxm::rpp_matmul_q3xxs_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                      o_type_size, 1, rpp_node->is_instantial);
    } else {
        kernel_q3_xxs::rpp_matmul_q3xxs_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size, o_type_size,
                                              rpp_node->is_instantial);
    }

    return true;
}

static bool ggml_rpp_create_kernel_iq3_xxs_lut_sram(ggml_backend_rpp_context & ctx,
                                                    ggml_rpp_node *            rpp_base_node,
                                                    ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    // now noly support seq_len == 1 for nolut_sram
    if (seq_len > 1) {
        return ggml_rpp_create_kernel_iq3_xxs_lut(ctx, rpp_base_node, dst);
    }
    const int n_expert    = dst->src[2]->ne[0];
    const int M           = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K           = dst->src[0]->ne[0];
    const int N           = dst->src[0]->ne[1];
    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);

    const size_t inputs_0_size       = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size        = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q3_codebook_size    = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q3_scales_size      = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q3_sign_size        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q3_super_scale_size = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q3_transformed_size = q3_codebook_size + q3_scales_size + q3_sign_size + q3_super_scale_size;

    kernel_q3_xxs_vxm::q3xxs_vxm_sram_io * io = new kernel_q3_xxs_vxm::q3xxs_vxm_sram_io;
    kernel_q3_xxs_vxm::q3xxs_vxm_prepare_sram_io(*(rpp_node->kernel_ctx.get()), *io, M, K, N, QK_K, i_type_size,
                                                 o_type_size, n_expert, true);
    if ((size_t) io->sizeC != outputs_size) {
        GGML_LOG_ERROR("%s: SRAM output size mismatch: io.sizeC=%s sizeC_all=%u\n", __func__, dst->name, io->sizeC,
                       outputs_size);
        return false;
    }
    rpp_node->sram_io = (void *) io;

    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);
    rpp_node->pool_buffers.emplace(dst->src[1]->data);
    rpp_node->pool_buffers.emplace(dst->src[0]->data);
    rpp_node->pool_buffers.emplace(dst->data);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    if (M == 1) {
        kernel_q3_xxs_vxm::rpp_matmul_q3xxs_vxm_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                      o_type_size, 1, 1, n_expert, rpp_node->is_instantial);
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
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    // GGML_ASSERT(ggml_is_contiguous(dst->src[2]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    const size_t inputs_0_size       = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size        = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q3_codebook_size    = (size_t) (K / 16) * 3u * (size_t) N * sizeof(uint16_t);
    const size_t q3_scales_size      = (size_t) (K / 128) * (size_t) N * sizeof(uint16_t);
    const size_t q3_sign_size        = (size_t) (K / 16) * (size_t) N * sizeof(uint16_t);
    const size_t q3_super_scale_size = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q3_transformed_size = q3_codebook_size + q3_scales_size + q3_sign_size + q3_super_scale_size;
    // GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q3_transformed_size);

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q3_transformed_size);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);

    RPPdeviceptr q3_codebook    = (RPPdeviceptr) weights_buf;
    RPPdeviceptr q3_scales      = q3_codebook + q3_codebook_size;
    RPPdeviceptr q3_sign        = q3_scales + q3_scales_size;
    RPPdeviceptr q3_super_scale = q3_sign + q3_sign_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_codebook);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q3_super_scale);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);

    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q3_xxs_vxm_nolut::rpp_matmul_q3xxs_vxm_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K,
                                                                  i_type_size, o_type_size, 1, 0, n_expert,
                                                                  rpp_node->is_instantial);
    } else {
        kernel_q3_xxs_nolut::rpp_matmul_q3xxs_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                          o_type_size, 0, rpp_node->is_instantial);
    }

    return true;
}

static bool ggml_rpp_create_kernel_iq3_xxs_nolut_sram(ggml_backend_rpp_context & ctx,
                                                      ggml_rpp_node *            rpp_base_node,
                                                      ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    // now noly support seq_len == 1 for nolut_sram
    if (seq_len > 1) {
        return ggml_rpp_create_kernel_iq3_xxs_nolut(ctx, rpp_base_node, dst);
    }
    const int n_expert    = dst->src[2]->ne[0];
    const int M           = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K           = dst->src[0]->ne[0];
    const int N           = dst->src[0]->ne[1];
    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);

    const size_t inputs_0_size       = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size        = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q3_codebook_size    = (size_t) (K / 16) * 3u * (size_t) N * sizeof(uint16_t);
    const size_t q3_scales_size      = (size_t) K / 128 * (size_t) N * sizeof(uint16_t);
    const size_t q3_sign_size        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q3_super_scale_size = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q3_transformed_size = q3_codebook_size + q3_scales_size + q3_sign_size + q3_super_scale_size;

    kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_sram_io * io = new kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_sram_io;
    kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_prepare_sram_io(*(rpp_node->kernel_ctx.get()), *io, M, K, N, QK_K,
                                                             i_type_size, o_type_size, n_expert, true);
    if ((size_t) io->sizeC != outputs_size) {
        GGML_LOG_ERROR("%s: SRAM output size mismatch: io.sizeC=%s sizeC_all=%u\n", __func__, dst->name, io->sizeC,
                       outputs_size);
        return false;
    }
    rpp_node->sram_io = (void *) io;

    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);
    rpp_node->pool_buffers.emplace(dst->src[1]->data);
    rpp_node->pool_buffers.emplace(dst->src[0]->data);
    rpp_node->pool_buffers.emplace(dst->data);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    if (M == 1) {
        kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_prepare_lut_workspace(*(rpp_node->kernel_ctx.get()));
        kernel_q3_xxs_vxm_nolut::rpp_matmul_q3xxs_vxm_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K,
                                                                  i_type_size, o_type_size, 1, 1, n_expert,
                                                                  rpp_node->is_instantial);
    } else {
        kernel_q3_xxs_nolut::rpp_matmul_q3xxs_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, QK_K, i_type_size,
                                                          o_type_size, 0, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_iq3_xxs(ggml_backend_rpp_context & ctx,
                                           ggml_rpp_node *            rpp_base_node,
                                           ggml_tensor *              dst,
                                           kernel_type                type = kernel_type::LUT_SRAM) {
    switch (type) {
        case LUT:
            return ggml_rpp_create_kernel_iq3_xxs_lut(ctx, rpp_base_node, dst);
        case NOLUT:
            return ggml_rpp_create_kernel_iq3_xxs_nolut(ctx, rpp_base_node, dst);
        case LUT_SRAM:
            return ggml_rpp_create_kernel_iq3_xxs_lut_sram(ctx, rpp_base_node, dst);
        case NOLUT_SRAM:
            return ggml_rpp_create_kernel_iq3_xxs_nolut_sram(ctx, rpp_base_node, dst);
        default:
            GGML_LOG_ERROR("%s: call create_kernel_iq3_xxs error, unkown type: %d:\n", __func__, type);
            break;
    }
    return false;
}

static bool ggml_rpp_create_kernel_iq2_s_lut(ggml_backend_rpp_context & ctx,
                                             ggml_rpp_node *            rpp_base_node,
                                             ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    // GGML_ASSERT(ggml_is_contiguous(dst->src[2]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    const size_t inputs_0_size      = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size       = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q2_codebook_lsb_sz = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_codebook_msb_sz = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_sz       = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_sign_sz         = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_sz  = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_size =
        q2_codebook_lsb_sz + q2_codebook_msb_sz + q2_scales_sz + q2_sign_sz + q2_super_scale_sz;
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q2_transformed_size);

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q2_transformed_size);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);

    RPPdeviceptr q2_codebook_lsb = (RPPdeviceptr) weights_buf;
    RPPdeviceptr q2_codebook_msb = q2_codebook_lsb + q2_codebook_lsb_sz;
    RPPdeviceptr q2_scales       = q2_codebook_msb + q2_codebook_msb_sz;
    RPPdeviceptr q2_sign         = q2_scales + q2_scales_sz;
    RPPdeviceptr q2_super_scale  = q2_sign + q2_sign_sz;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_codebook_lsb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_codebook_msb);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_sign);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_super_scale);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);

    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
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
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    const size_t inputs_0_size     = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size      = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q2_codebook_nolut = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_sz      = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_sign_sz        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_sz = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_sz = q2_codebook_nolut + q2_scales_sz + q2_sign_sz + q2_super_scale_sz;
    const int    q2_weights_group  = QK_K;

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q2_transformed_sz);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);

    RPPdeviceptr q2_codebook_nolut_buf = (RPPdeviceptr) weights_buf;
    RPPdeviceptr q2_scales_buf         = q2_codebook_nolut_buf + q2_codebook_nolut;
    RPPdeviceptr q2_sign_buf           = q2_scales_buf + q2_scales_sz;
    RPPdeviceptr q2_super_scale_buf    = q2_sign_buf + q2_sign_sz;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_codebook_nolut_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_scales_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_sign_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_super_scale_buf);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);

    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q2_s_vxm_nolut::rpp_matmul_q2s_vxm_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                              i_type_size, o_type_size, 1, 0, n_expert,
                                                              rpp_node->is_instantial);
    } else {
        kernel_q2_s_nolut::rpp_matmul_q2s_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                      i_type_size, o_type_size, 0, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_iq2_s_nolut_sram(ggml_backend_rpp_context & ctx,
                                                    ggml_rpp_node *            rpp_base_node,
                                                    ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    // now noly support seq_len == 1 for nolut_sram
    if (seq_len > 1) {
        return ggml_rpp_create_kernel_iq2_s_nolut(ctx, rpp_base_node, dst);
    }
    const int n_expert    = dst->src[2]->ne[0];
    const int M           = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K           = dst->src[0]->ne[0];
    const int N           = dst->src[0]->ne[1];
    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);

    const size_t inputs_0_size     = M * K * i_type_size;
    const size_t outputs_size      = M * N * n_expert * o_type_size;
    const size_t q2_codebook_nolut = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_sz      = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_sign_sz        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_sz = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_sz = q2_codebook_nolut + q2_scales_sz + q2_sign_sz + q2_super_scale_sz;
    const int    q2_weights_group  = QK_K;

    kernel_q2_s_vxm_nolut::q2s_vxm_nolut_sram_io * io = new kernel_q2_s_vxm_nolut::q2s_vxm_nolut_sram_io;
    kernel_q2_s_vxm_nolut::q2s_vxm_nolut_prepare_sram_io(*(rpp_node->kernel_ctx.get()), *io, M, K, N, QK_K, i_type_size,
                                                         o_type_size, n_expert, true);
    if ((size_t) io->sizeC != outputs_size) {
        GGML_LOG_ERROR("%s: SRAM output size mismatch: io.sizeC=%s sizeC_all=%d\n", __func__, io->sizeC, outputs_size);
        return false;
    }
    rpp_node->sram_io = (void *) io;

    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);
    rpp_node->pool_buffers.emplace(dst->src[1]->data);
    rpp_node->pool_buffers.emplace(dst->src[0]->data);
    rpp_node->pool_buffers.emplace(dst->data);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    if (M == 1) {
        kernel_q2_s_vxm_nolut::q2s_vxm_nolut_prepare_lut_workspace(*(rpp_node->kernel_ctx.get()));
        kernel_q2_s_vxm_nolut::rpp_matmul_q2s_vxm_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                              i_type_size, o_type_size, 1, 1, n_expert,
                                                              rpp_node->is_instantial);
    } else {
        kernel_q2_s_nolut::rpp_matmul_q2s_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                      i_type_size, o_type_size, 0, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_iq2_s(ggml_backend_rpp_context & ctx,
                                         ggml_rpp_node *            rpp_base_node,
                                         ggml_tensor *              dst,
                                         kernel_type                type = kernel_type::NOLUT_SRAM) {
    switch (type) {
        case LUT:
            return ggml_rpp_create_kernel_iq2_s_lut(ctx, rpp_base_node, dst);
        case NOLUT:
            return ggml_rpp_create_kernel_iq2_s_nolut(ctx, rpp_base_node, dst);
        case NOLUT_SRAM:
            return ggml_rpp_create_kernel_iq2_s_nolut_sram(ctx, rpp_base_node, dst);
        default:
            GGML_LOG_ERROR("%s: call create_kernel_iq2_s error, unkown type: %d:\n", __func__, type);
            break;
    }
    return false;
}

static bool ggml_rpp_create_kernel_iq2_xs_lut(ggml_backend_rpp_context & ctx,
                                              ggml_rpp_node *            rpp_base_node,
                                              ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    const size_t inputs_0_size = (size_t) M * (size_t) K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size  = (size_t) M * (size_t) N * (size_t) n_expert * ggml_type_size(dst->type);

    const size_t q2_qs_size            = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_size        = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_size   = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_nbytes = q2_qs_size + q2_scales_size + q2_super_scale_size;
    GGML_ASSERT(ggml_row_size(dst->src[0]->type, K) * N == q2_transformed_nbytes);

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q2_transformed_nbytes);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);

    RPPdeviceptr q2_qs          = (RPPdeviceptr) weights_buf;
    RPPdeviceptr q2_scales      = q2_qs + q2_qs_size;
    RPPdeviceptr q2_super_scale = q2_scales + q2_scales_size;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_qs);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_scales);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_super_scale);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);

    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
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
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];

    const size_t inputs_0_size     = M * K * ggml_type_size(dst->src[1]->type);
    const size_t outputs_size      = M * N * n_expert * ggml_type_size(dst->type);
    const size_t q2_codebook_nolut = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_sz      = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_sign_sz        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_sz = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_sz = q2_codebook_nolut + q2_scales_sz + q2_sign_sz + q2_super_scale_sz;
    const int    q2_weights_group  = QK_K;

    void * inputs_0_buf = ctx.pool().alloc(inputs_0_size);
    void * weights_buf  = ctx.pool().alloc(q2_transformed_sz);
    void * outputs_buf  = ctx.pool().alloc(outputs_size);

    RPPdeviceptr q2_codebook_nolut_buf = (RPPdeviceptr) weights_buf;
    RPPdeviceptr q2_scales_buf         = q2_codebook_nolut_buf + q2_codebook_nolut;
    RPPdeviceptr q2_sign_buf           = q2_scales_buf + q2_scales_sz;
    RPPdeviceptr q2_super_scale_buf    = q2_sign_buf + q2_sign_sz;

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_codebook_nolut_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_scales_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_sign_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) q2_super_scale_buf);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);

    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);
    rpp_node->pool_buffers.emplace(inputs_0_buf);
    rpp_node->pool_buffers.emplace(weights_buf);
    rpp_node->pool_buffers.emplace(outputs_buf);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    if (M == 1) {
        kernel_q2_xs_vxm_nolut::rpp_matmul_q2xs_vxm_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N,
                                                                q2_weights_group, i_type_size, o_type_size, 1, 0,
                                                                n_expert, rpp_node->is_instantial);
    } else {
        kernel_q2_xs_nolut::rpp_matmul_q2xs_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                        i_type_size, o_type_size, 0, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_iq2_xs_nolut_sram(ggml_backend_rpp_context & ctx,
                                                     ggml_rpp_node *            rpp_base_node,
                                                     ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    // now noly support seq_len == 1 for nolut_sram
    if (seq_len > 1) {
        return ggml_rpp_create_kernel_iq2_xs_nolut(ctx, rpp_base_node, dst);
    }
    const int n_expert    = dst->src[2]->ne[0];
    const int M           = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K           = dst->src[0]->ne[0];
    const int N           = dst->src[0]->ne[1];
    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);

    const size_t inputs_0_size     = M * K * i_type_size;
    const size_t outputs_size      = M * N * n_expert * o_type_size;
    const size_t q2_codebook_nolut = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
    const size_t q2_scales_sz      = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
    const size_t q2_sign_sz        = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
    const size_t q2_super_scale_sz = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
    const size_t q2_transformed_sz = q2_codebook_nolut + q2_scales_sz + q2_sign_sz + q2_super_scale_sz;
    const int    q2_weights_group  = QK_K;

    kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_sram_io * io = new kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_sram_io;
    kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_prepare_sram_io(*(rpp_node->kernel_ctx.get()), *io, M, K, N, QK_K,
                                                           i_type_size, o_type_size, n_expert, true);
    if ((size_t) io->sizeC != outputs_size) {
        GGML_LOG_ERROR("%s: SRAM output size mismatch: io.sizeC=%s sizeC_all=%u\n", __func__, dst->name, io->sizeC,
                       outputs_size);
        return false;
    }
    rpp_node->sram_io = (void *) io;

    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);
    rpp_node->pool_buffers.emplace(dst->src[1]->data);
    rpp_node->pool_buffers.emplace(dst->src[0]->data);
    rpp_node->pool_buffers.emplace(dst->data);

    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    if (M == 1) {
        kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_prepare_lut_workspace(*(rpp_node->kernel_ctx.get()));
        kernel_q2_xs_vxm_nolut::rpp_matmul_q2xs_vxm_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N,
                                                                q2_weights_group, i_type_size, o_type_size, 1, 1,
                                                                n_expert, rpp_node->is_instantial);
    } else {
        kernel_q2_xs_nolut::rpp_matmul_q2xs_nolut_build(*(rpp_node->kernel_ctx.get()), M, K, N, q2_weights_group,
                                                        i_type_size, o_type_size, 0, rpp_node->is_instantial);
    }
    return true;
}

static bool ggml_rpp_create_kernel_iq2_xs(ggml_backend_rpp_context & ctx,
                                          ggml_rpp_node *            rpp_base_node,
                                          ggml_tensor *              dst,
                                          kernel_type                type = kernel_type::NOLUT_SRAM) {
    switch (type) {
        case LUT:
            return ggml_rpp_create_kernel_iq2_xs_lut(ctx, rpp_base_node, dst);
        case NOLUT:
            return ggml_rpp_create_kernel_iq2_xs_nolut(ctx, rpp_base_node, dst);
        case NOLUT_SRAM:
            return ggml_rpp_create_kernel_iq2_xs_nolut_sram(ctx, rpp_base_node, dst);
        default:
            GGML_LOG_ERROR("%s: call create_kernel_iq2_xs error, unkown type: %d:\n", __func__, type);
            break;
    }
    return false;
}

static bool ggml_rpp_create_kernel_bf16(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    // notice!!!  requires the inputs must be 2-dimensional
    GGML_ASSERT(ggml_n_dims(dst->src[1]) < 4);
    GGML_ASSERT(ggml_n_dims(dst->src[0]) < 4);
    GGML_ASSERT(is_matmul_weight(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));

    const int seq_len  = ggml_rpp_mat_mul_id_seq_len(dst, rpp_node);
    const int n_expert = dst->src[2]->ne[0];
    const int M        = (ctx.use_ubatch && seq_len > 1) ? rpp_node->n_ubatch : dst->src[1]->ne[2];
    const int K        = dst->src[0]->ne[0];
    const int N        = dst->src[0]->ne[1];
    const int w_group  = 32;

    const size_t inputs_0_size = (size_t) M * (size_t) K * ggml_type_size(dst->src[1]->type);
    const size_t weights_size  = ggml_row_size(dst->src[0]->type, K) * N;
    const size_t outputs_size  = (size_t) M * (size_t) N * (size_t) n_expert * ggml_type_size(dst->type);
    void *       inputs_0_buf  = ctx.pool().alloc(inputs_0_size);
    void *       weights_buf   = ctx.pool().alloc(weights_size);
    void *       outputs_buf   = ctx.pool().alloc(outputs_size);

    // kernel inputs
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) weights_buf);
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) outputs_buf);
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
    rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
    rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(weights_buf);
    rpp_node->binding_io_buffers.emplace_back(outputs_buf);

    // find first rms_node, and all rms_norm kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_mul_mat_id_node(ctx, rpp_node, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_mul_mat_id *>(ori_rpp_node);
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
    auto rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 2 ? 2 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_mat_mul_id_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }
    // create kernel according to quantized weight type
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            ret = ggml_rpp_create_kernel_bf16(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q4_1:
            ret = ggml_rpp_create_kernel_q4_1(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q8_0:
            ret = ggml_rpp_create_kernel_q8_0(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q4_K:
            ret = ggml_rpp_create_kernel_q4_k(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q5_K:
            ret = ggml_rpp_create_kernel_q5_k(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_Q6_K:
            ret = ggml_rpp_create_kernel_q6_k(ctx, rpp_node, dst);
            break;
        case GGML_TYPE_IQ3_XXS:
            ret = ggml_rpp_create_kernel_iq3_xxs(ctx, rpp_node, dst, kernel_type::NOLUT_SRAM);
            break;
        case GGML_TYPE_IQ2_S:
            ret = ggml_rpp_create_kernel_iq2_s(ctx, rpp_node, dst, kernel_type::NOLUT_SRAM);
            break;
        case GGML_TYPE_IQ2_XS:
            ret = ggml_rpp_create_kernel_iq2_xs(ctx, rpp_node, dst, kernel_type::NOLUT_SRAM);
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

void * ggml_rpp_get_quantized_data(const ggml_tensor * src, int64_t expert_id, int64_t & quantized_size) {
    GGML_ASSERT(src != nullptr);
    GGML_ASSERT(ggml_is_quantized(src->type));

    const int64_t K = src->ne[0];
    const int64_t N = src->ne[1];
    GGML_ASSERT(K > 0 && N > 0);

    switch (src->type) {
        case GGML_TYPE_F32:
            {
                //because the weights is convert to bf16, so the quantized size is half of f32
                quantized_size = ggml_row_size(src->type, K) * (size_t) N / 2;
            }
            break;
        case GGML_TYPE_IQ3_XXS:
            {
                // this for lut
                // const size_t q3_codebook_sz = (size_t) (K / 8) * (size_t) N * sizeof(uint16_t);
                // this for no lut
                const size_t q3_codebook_sz = (size_t) (K / 16) * 3u * (size_t) N * sizeof(uint16_t);
                const size_t q3_scales_sz   = (size_t) (K / 128) * (size_t) N * sizeof(uint16_t);
                const size_t q3_sign_sz     = (size_t) (K / 16) * (size_t) N * sizeof(uint16_t);
                const size_t q3_super_sz    = (size_t) (K / QK_K) * (size_t) N * sizeof(bfloat16_u16);
                quantized_size              = q3_codebook_sz + q3_scales_sz + q3_sign_sz + q3_super_sz;
            }
            break;
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_XS:
            {
                const size_t q2_codebook_nolut_size = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
                const size_t q2_scales_size         = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
                const size_t q2_sign_size           = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
                const size_t q2_super_scale_size    = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
                quantized_size = q2_codebook_nolut_size + q2_scales_size + q2_sign_size + q2_super_scale_size;
            }
            break;
        default:
            {
                // Per-expert quantized bytes for a [K, N] weight matrix.
                quantized_size = ggml_row_size(src->type, K) * (size_t) N;
            }
            break;
    }
    const int64_t n_experts = src->ne[2] > 0 ? src->ne[2] : 1;
    GGML_ASSERT(expert_id >= 0 && expert_id < n_experts);

    if (n_experts == 1) {
        GGML_ASSERT(quantized_size <= ggml_nbytes(src));
        return src->data;
    }

    return (char *) src->data + (size_t) expert_id * quantized_size;
}

bool ggml_rpp_launch_kernel_ddr(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node, ggml_tensor * dst) {
    GGML_ASSERT(rpp_base_node);
    auto                rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    const ggml_tensor * src0     = dst->src[0];
    const ggml_tensor * src1     = dst->src[1];
    const ggml_tensor * ids      = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    // GGML_ASSERT(!ggml_rpp_backend_buft_is_rpp_split(src0->buffer->buft) && "mul_mat_id does not support split buffers");

    GGML_TENSOR_BINARY_OP_LOCALS

    rtStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2 % nb1 == 0);

    const int src1_i_type_size = ggml_rpp_get_io_type_size(ctx, (ggml_tensor *) src1, 0);
    const int dst_o_type_size  = ggml_rpp_get_io_type_size(ctx, dst, 1);
    GGML_ASSERT(src1_i_type_size > 0 && dst_o_type_size > 0);
    GGML_ASSERT(ggml_type_size(src1->type) % (size_t) src1_i_type_size == 0);
    GGML_ASSERT(ggml_type_size(dst->type) % (size_t) dst_o_type_size == 0);
    const size_t  src1_scale          = ggml_type_size(src1->type) / (size_t) src1_i_type_size;
    const size_t  dst_scale           = ggml_type_size(dst->type) / (size_t) dst_o_type_size;
    const size_t  src1_nb11_internal  = (size_t) nb11 / src1_scale;
    const size_t  dst_nb1_internal    = (size_t) nb1 / dst_scale;
    const size_t  src1_row_copy_bytes = (size_t) ne10 * (size_t) src1_i_type_size;
    const size_t  dst_row_copy_bytes  = (size_t) ne0 * (size_t) dst_o_type_size;
    const int64_t n_expert_used       = ids->ne[0];

    ggml_rpp_pool_alloc<int32_t> ids_host(ctx.pool_host_leg(), ggml_nelements(ids));
    RPP_CHECK(rtMemcpyAsync(ids_host.ptr, ids->data, ggml_nbytes(ids), rtMemcpyDeviceToHost, stream));
    RPP_CHECK(rtStreamSynchronize(stream));

    GGML_ASSERT(ne11 == 1 || ne11 == n_expert_used);
    char * dst_data_cur = (char *) dst->data;
    for (size_t i = 0; i < n_expert_used; i++) {
        const int64_t src1_row      = (ne11 == 1) ? 0 : (int64_t) i;
        char * const  src1_data_cur = (char *) src1->data + src1_row * src1_nb11_internal;

        int64_t expert_stride = 0;
        void *  weigths_ptr   = ggml_rpp_get_quantized_data(src0, ids_host.ptr[i], expert_stride);
        // GGML_ASSERT(expert_stride == nb02);
        RPP_MEMCPY_DEV_AND_HOST(rpp_node->binding_io_buffers[1], weigths_ptr, expert_stride, rtMemcpyDeviceToDevice,
                                stream);

        RPP_MEMCPY_DEV_AND_HOST(rpp_node->binding_io_buffers[0], src1_data_cur, src1_row_copy_bytes,
                                rtMemcpyDeviceToDevice, stream);

        // compute add operator
        try {
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }

        RPP_MEMCPY_DEV_AND_HOST(dst_data_cur, rpp_node->binding_io_buffers[2], dst_row_copy_bytes, rtMemcpyDeviceToDevice,
                                stream, 1);
        dst_data_cur += dst_nb1_internal;
    }
    return true;
}

bool ggml_rpp_launch_kernel_sram(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node, ggml_tensor * dst) {
    GGML_ASSERT(rpp_base_node);
    auto                rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    const ggml_tensor * src0     = dst->src[0];
    const ggml_tensor * src1     = dst->src[1];
    const ggml_tensor * ids      = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS
    const rtStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2 % nb1 == 0);

    const int src1_i_type_size = ggml_rpp_get_io_type_size(ctx, (ggml_tensor *) src1, 0);
    GGML_ASSERT(src1_i_type_size > 0);
    GGML_ASSERT(ggml_type_size(src1->type) % (size_t) src1_i_type_size == 0);
    const size_t src1_scale         = ggml_type_size(src1->type) / (size_t) src1_i_type_size;
    const size_t src1_row_bytes     = (size_t) nb11 / src1_scale;
    const int64_t n_expert_used     = ids->ne[0];
    const size_t  act_total_bytes   = (size_t) n_expert_used * src1_row_bytes;

    ggml_rpp_pool_alloc<int32_t> ids_host(ctx.pool_host_leg(), ggml_nelements(ids));
    RPP_CHECK(rtMemcpyAsync(ids_host.ptr, ids->data, ggml_nbytes(ids), rtMemcpyDeviceToHost, stream));
    RPP_CHECK(rtStreamSynchronize(stream));

    std::vector<RPPdeviceptr> dst_sram;
    std::vector<RPPdeviceptr> src_devi;
    std::vector<size_t>       byte_count;

    auto copy_workspace_to_sram = [&](rpp_kernel_context * kernel_ctx, ggml_tensor * cur_tensor) {
        GGML_ASSERT(kernel_ctx);
        GGML_ASSERT(cur_tensor && cur_tensor->src[0] && cur_tensor->src[1]);
        try {
            switch (cur_tensor->src[0]->type) {
                case GGML_TYPE_IQ2_S:
                    {
                        using namespace kernel_q2_s_vxm_nolut;
                        RPPdeviceptr base_ptr         = kernel_ctx->dev_workspace;
                        const size_t qscale_lut_bytes = q2s_vxm_nolut_lut_workspace::qscale_lut_bytes;
                        const size_t mag_lut_bytes    = q2s_vxm_nolut_lut_workspace::mag_lut_bytes;

                        const RPPdeviceptr sramB_qscale_lut = kernel_ctx->dev_in[5];
                        const RPPdeviceptr sramB_mag_lut    = kernel_ctx->dev_in[6];

                        dst_sram.emplace_back(sramB_qscale_lut);
                        dst_sram.emplace_back(sramB_mag_lut);
                        src_devi.emplace_back(base_ptr);
                        src_devi.emplace_back(base_ptr + qscale_lut_bytes);
                        byte_count.emplace_back(qscale_lut_bytes);
                        byte_count.emplace_back(mag_lut_bytes);
                    }
                case GGML_TYPE_IQ2_XS:
                    {
                        RPPdeviceptr       base_ptr         = kernel_ctx->dev_workspace;
                        constexpr uint32_t qscale_lut_bytes = 16u * (uint32_t) sizeof(uint16_t);
                        constexpr uint32_t mag_lut_bytes    = 4u * (uint32_t) sizeof(uint16_t);

                        const RPPdeviceptr sramB_qscale_lut = kernel_ctx->dev_in[5];
                        const RPPdeviceptr sramB_mag_lut    = kernel_ctx->dev_in[6];

                        dst_sram.emplace_back(sramB_qscale_lut);
                        dst_sram.emplace_back(sramB_mag_lut);
                        src_devi.emplace_back(base_ptr);
                        src_devi.emplace_back(base_ptr + qscale_lut_bytes);
                        byte_count.emplace_back(qscale_lut_bytes);
                        byte_count.emplace_back(mag_lut_bytes);
                    }
                    break;
                case GGML_TYPE_IQ3_XXS:
                    {
                        using namespace kernel_q3_xxs_vxm_nolut;
                        RPPdeviceptr base_ptr     = kernel_ctx->dev_workspace;
                        const size_t qscale_bytes = q3xxs_vxm_nolut_lut_workspace::qscale_lut_bytes;
                        const size_t mag_bytes    = q3xxs_vxm_nolut_lut_workspace::mag_lut_bytes;
                        const size_t mat_bytes    = q3xxs_vxm_nolut_lut_workspace::mat_lut_bytes;
                        const size_t off_qscale   = 0;
                        const size_t off_mag      = kernel_q3_xxs_vxm_nolut::align_up(off_qscale + qscale_bytes, 64);
                        const size_t off_mat      = kernel_q3_xxs_vxm_nolut::align_up(off_mag + mag_bytes, 64);

                        const RPPdeviceptr deviB_qscale_lut = base_ptr + (RPPdeviceptr) off_qscale;
                        const RPPdeviceptr deviB_mag_lut    = base_ptr + (RPPdeviceptr) off_mag;
                        const RPPdeviceptr deviB_mat_lut    = base_ptr + (RPPdeviceptr) off_mat;

                        const RPPdeviceptr sramB_qscale_lut = kernel_ctx->dev_in[5];
                        const RPPdeviceptr sramB_mag_lut    = kernel_ctx->dev_in[6];
                        const RPPdeviceptr sramB_mat_lut    = kernel_ctx->dev_in[7];

                        dst_sram.emplace_back(sramB_qscale_lut);
                        dst_sram.emplace_back(sramB_mag_lut);
                        dst_sram.emplace_back(sramB_mat_lut);
                        src_devi.emplace_back(deviB_qscale_lut);
                        src_devi.emplace_back(deviB_mag_lut);
                        src_devi.emplace_back(deviB_mat_lut);
                        byte_count.emplace_back(qscale_bytes);
                        byte_count.emplace_back(mag_bytes);
                        byte_count.emplace_back(mat_bytes);
                    }
                    break;
                default:
                    GGML_LOG_ERROR("%s: unsupported mul_mat_id weight type %d in fusion (%s)\n", __func__,
                                   (int) cur_tensor->src[0]->type, cur_tensor->name);
            }
        } catch (const std::exception & ex) {
            GGML_LOG_ERROR("%s: build mul_mat_id kernel failed (%s): %s\n", __func__, cur_tensor->name, ex.what());
        }
    };

    auto copy_activation_to_sram = [&](RPPdeviceptr sram_act_base) {
        if (ne11 == n_expert_used) {
            dst_sram.push_back(sram_act_base);
            src_devi.push_back((RPPdeviceptr) dst->src[1]->data);
            byte_count.push_back(act_total_bytes);
            return;
        }
        for (int64_t e = 0; e < n_expert_used; ++e) {
            const int64_t src1_row = (ne11 == 1) ? 0 : e;
            const auto    src1_cur =
                (RPPdeviceptr) dst->src[1]->data + (RPPdeviceptr) src1_row * (RPPdeviceptr) src1_row_bytes;
            const auto sramA_cur = sram_act_base + (RPPdeviceptr) e * (RPPdeviceptr) src1_row_bytes;
            dst_sram.push_back(sramA_cur);
            src_devi.push_back(src1_cur);
            byte_count.push_back(src1_row_bytes);
        }
    };

    auto copy_expert_weights_to_sram = [&](ggml_tensor * src_weight, RPPdeviceptr sram_weight_base,
                                           uint32_t weight_bytes) {
        for (int e = 0; e < n_expert_used; ++e) {
            const RPPdeviceptr sram_dst = sram_weight_base + (RPPdeviceptr) e * (RPPdeviceptr) weight_bytes;
            const RPPdeviceptr ddr_src =
                (RPPdeviceptr) src_weight->data + (RPPdeviceptr) ids_host.ptr[e] * (RPPdeviceptr) weight_bytes;
            dst_sram.push_back(sram_dst);
            src_devi.push_back(ddr_src);
            byte_count.push_back(weight_bytes);
        }
    };

    GGML_ASSERT(ne11 == 1 || ne11 == n_expert_used);
    char * dst_data_cur = (char *) dst->data;
    switch (src0->type) {
        case GGML_TYPE_IQ2_S:
            {
                auto         io_ptr = static_cast<kernel_q2_s_vxm_nolut::q2s_vxm_nolut_sram_io *>(rpp_node->sram_io);
                auto         io     = *io_ptr;
                if ((size_t) io.sizeA != (size_t) n_expert_used * src1_row_bytes) {
                    GGML_LOG_ERROR("%s: IQ2_S SRAM input size mismatch dst=%s io.sizeA=%u expected=%zu\n", __func__,
                                   dst->name, io.sizeA, (size_t) n_expert_used * src1_row_bytes);
                    return false;
                }
                copy_activation_to_sram(io.sramA);
                copy_workspace_to_sram(rpp_node->kernel_ctx.get(), dst);
                copy_expert_weights_to_sram(dst->src[0], io.sramB_codebook_nolut, io.size_weights_expert);
                rppMemcpyLinkDtoSAsync(dst_sram.data(), src_devi.data(), byte_count.data(), dst_sram.size(), stream);
                // for (int e = 0; e < n_expert_used; ++e) {
                //     const int64_t src1_row = (ne11 == 1) ? 0 : (int64_t) e;
                //     const auto    src1_cur =
                //         (RPPdeviceptr) src1->data + (RPPdeviceptr) src1_row * (RPPdeviceptr) src1_row_bytes;
                //     const auto sramA_cur = io.sramA + (RPPdeviceptr) e * (RPPdeviceptr) src1_row_bytes;
                //     kernel_q2_s_vxm_nolut::q2s_vxm_nolut_cdma_d2s_async(sramA_cur, src1_cur, src1_row_bytes, stream);
                //     kernel_q2_s_vxm_nolut::q2s_vxm_nolut_cdma_copy_expert_weights_to_sram(io, (RPPdeviceptr) src0->data,
                //                                                                           e, ids_host.ptr[e], stream);
                // }
            }
            break;
        case GGML_TYPE_IQ2_XS:
            {
                auto         io_ptr = static_cast<kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_sram_io *>(rpp_node->sram_io);
                auto         io     = *io_ptr;
                if ((size_t) io.sizeA != (size_t) n_expert_used * src1_row_bytes) {
                    GGML_LOG_ERROR("%s: IQ2_XS SRAM input size mismatch dst=%s io.sizeA=%u expected=%zu\n", __func__,
                                   dst->name, io.sizeA, (size_t) n_expert_used * src1_row_bytes);
                    return false;
                }

                copy_activation_to_sram(io.sramA);
                copy_workspace_to_sram(rpp_node->kernel_ctx.get(), dst);
                copy_expert_weights_to_sram(dst->src[0], io.sramB_codebook_nolut, io.size_weights_expert);
                rppMemcpyLinkDtoSAsync(dst_sram.data(), src_devi.data(), byte_count.data(), dst_sram.size(), stream);
                // for (int e = 0; e < n_expert_used; ++e) {
                //     const int64_t src1_row = (ne11 == 1) ? 0 : (int64_t) e;
                //     const auto    src1_cur =
                //         (RPPdeviceptr) src1->data + (RPPdeviceptr) src1_row * (RPPdeviceptr) src1_row_bytes;
                //     const auto sramA_cur = io.sramA + (RPPdeviceptr) e * (RPPdeviceptr) src1_row_bytes;
                //     kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_cdma_d2s_async(sramA_cur, src1_cur, src1_row_bytes, stream);
                //     kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_cdma_copy_expert_weights_to_sram(
                //         io, (RPPdeviceptr) src0->data, e, ids_host.ptr[e], stream);
                // }
            }
            break;
        case GGML_TYPE_IQ3_XXS:
            {
                // this is for IQ3_XXS nolut sram
                auto io_ptr = static_cast<kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_sram_io *>(rpp_node->sram_io);
                auto io     = *io_ptr;
                if ((size_t) io.sizeA != (size_t) n_expert_used * src1_row_bytes) {
                    GGML_LOG_ERROR("%s: IQ3_XXS SRAM input size mismatch dst=%s io.sizeA=%u expected=%zu\n", __func__,
                                   dst->name, io.sizeA, (size_t) n_expert_used * src1_row_bytes);
                    return false;
                }
                copy_activation_to_sram(io.sramA);
                copy_workspace_to_sram(rpp_node->kernel_ctx.get(), dst);
                copy_expert_weights_to_sram(dst->src[0], io.sramB_codebook_nolut, io.size_weights_expert);
                rppMemcpyLinkDtoSAsync(dst_sram.data(), src_devi.data(), byte_count.data(), dst_sram.size(), stream);
                // for (int e = 0; e < n_expert_used; ++e) {
                //     const int64_t src1_row = (ne11 == 1) ? 0 : (int64_t) e;
                //     const auto    src1_cur =
                //         (RPPdeviceptr) src1->data + (RPPdeviceptr) src1_row * (RPPdeviceptr) src1_row_bytes;
                //     const auto sramA_cur = io.sramA + (RPPdeviceptr) e * (RPPdeviceptr) src1_row_bytes;
                //     kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_cdma_d2s_async(sramA_cur, src1_cur, src1_row_bytes,
                //                                                             stream);
                //     kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_cdma_copy_expert_weights_to_sram(
                //         io, (RPPdeviceptr) src0->data, e, ids_host.ptr[e], stream);
                // }
            }
            break;
        default:
            GGML_LOG_ERROR("%s: , unkown type: %s, con not copy expert_weights/inputs from ddr to sram:\n", __func__,
                           ggml_type_name(src0->type));
            return false;
    }

    // compute add operator
    try {
        RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, stream);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }

    switch (src0->type) {
        case GGML_TYPE_IQ2_S:
            {
                auto io_ptr = static_cast<kernel_q2_s_vxm_nolut::q2s_vxm_nolut_sram_io *>(rpp_node->sram_io);
                auto io     = *io_ptr;
                kernel_q2_s_vxm_nolut::q2s_vxm_nolut_cdma_copy_output_to_ddr(io, (RPPdeviceptr) dst->data, stream);
            }
            break;
        case GGML_TYPE_IQ2_XS:
            {
                auto io_ptr = static_cast<kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_sram_io *>(rpp_node->sram_io);
                auto io     = *io_ptr;
                kernel_q2_xs_vxm_nolut::q2xs_vxm_nolut_cdma_copy_output_to_ddr(io, (RPPdeviceptr) dst->data, stream);
            }
            break;
        case GGML_TYPE_IQ3_XXS:
            {
                auto io_ptr = static_cast<kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_sram_io *>(rpp_node->sram_io);
                auto io     = *io_ptr;
                kernel_q3_xxs_vxm_nolut::q3xxs_vxm_nolut_cdma_copy_output_to_ddr(io, (RPPdeviceptr) dst->data, stream);
            }
            break;
        default:
            GGML_LOG_ERROR("%s: , unkown type: %s, con not copy outputs from sram to ddr:\n", __func__,
                           ggml_type_name(src0->type));
            return false;
    }
    return true;
}

bool ggml_rpp_launch_kernel(ggml_backend_rpp_context & ctx,
                            ggml_rpp_node *            rpp_base_node,
                            ggml_tensor *              dst,
                            int                        use_sram_direct = 0) {
    const int seq_len = ggml_rpp_mat_mul_id_seq_len(dst, rpp_base_node);
    // noly support seq_len == 1 for nolut_sram
    if (use_sram_direct && seq_len == 1) {
        switch (dst->src[0]->type) {
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ3_XXS:
                return ggml_rpp_launch_kernel_sram(ctx, rpp_base_node, dst);
            default:
                return ggml_rpp_launch_kernel_ddr(ctx, rpp_base_node, dst);
        }
    } else {
        return ggml_rpp_launch_kernel_ddr(ctx, rpp_base_node, dst);
    }
}

bool ggml_rpp_launch_kernel_ubatch(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node, ggml_tensor * dst) {
    GGML_ASSERT(rpp_base_node);
    auto                rpp_node = static_cast<rpp_kernel_mul_mat_id *>(rpp_base_node);
    const ggml_tensor * src0     = dst->src[0];
    const ggml_tensor * src1     = dst->src[1];
    const ggml_tensor * ids      = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    rtStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2 % nb1 == 0);

    const int src1_i_type_size = ggml_rpp_get_io_type_size(ctx, (ggml_tensor *) src1, 0);
    const int dst_o_type_size  = ggml_rpp_get_io_type_size(ctx, dst, 1);
    GGML_ASSERT(src1_i_type_size > 0 && dst_o_type_size > 0);
    GGML_ASSERT(ggml_type_size(src1->type) % (size_t) src1_i_type_size == 0);
    GGML_ASSERT(ggml_type_size(dst->type) % (size_t) dst_o_type_size == 0);
    const size_t src1_scale         = ggml_type_size(src1->type) / (size_t) src1_i_type_size;
    const size_t dst_scale          = ggml_type_size(dst->type) / (size_t) dst_o_type_size;
    const size_t src1_nb11_internal = (size_t) nb11 / src1_scale;
    const size_t dst_nb1_internal   = (size_t) nb1 / dst_scale;
    const ggml_type type_src1_sorted = (src1_i_type_size == (int) sizeof(float)) ? GGML_TYPE_F32 : GGML_TYPE_BF16;
    const ggml_type type_dst_sorted  = (dst_o_type_size == (int) sizeof(float)) ? GGML_TYPE_F32 : GGML_TYPE_BF16;
    const size_t    ts_src1_sorted   = (size_t) src1_i_type_size;
    const size_t    ts_dst_sorted    = (size_t) dst_o_type_size;

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows   = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2 * ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);
    // now is on cpu
    // ggml_rpp_pool_alloc<int32_t> ids_buf_dev(ctx.pool_leg(), 2 * ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_rpp_pool_alloc<char> src1_sorted(ctx.pool_leg(), ne12 * n_expert_used * ne10 * ts_src1_sorted);
    ggml_rpp_pool_alloc<char> dst_sorted(ctx.pool_leg(), ne2 * n_expert_used * ne0 * ts_dst_sorted);
    // std::vector<char> ids_host(ggml_nbytes(ids));
    // RPP_CHECK(rtMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), rtMemcpyDeviceToHost, stream));
    ggml_rpp_pool_alloc<char> ids_host(ctx.pool_host_leg(), ggml_nbytes(ids));
    RPP_CHECK(rtMemcpyAsync(ids_host.ptr, ids->data, ggml_nbytes(ids), rtMemcpyDeviceToHost, stream));
    RPP_CHECK(rtStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) {      // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) {  // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *) (ids_host.ptr + i12 * ids->nb[1] + iex * ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12 * n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12 * ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    // now is on cpu
    // RPP_CHECK(rtMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2 * ne_get_rows * sizeof(int32_t),
    //                         rtMemcpyHostToDevice, stream));
    // RPP_CHECK(rtStreamSynchronize(stream));

    // const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0 * ne_get_rows;
    // const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1 * ne_get_rows;

    ggml_rpp_get_rows_cpy(src1->data, ids_to_sorted_host.data(), src1_sorted.ptr, ne_get_rows, src1_nb11_internal,
                          ne10 * ts_src1_sorted, stream);
    RPP_CHECK(rtGetLastError());
    GGML_ASSERT(ne11 == 1 || ne11 == n_expert_used);
    char * src1_data_cur = (char *) src1_sorted.ptr;
    char * dst_data_cur  = (char *) dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }
        int64_t     expert_stride = 0;
        ggml_tensor src0_slice    = *src0;
        src0_slice.ne[2]          = 1;
        src0_slice.nb[3]          = src0_slice.nb[2];
        src0_slice.op             = GGML_OP_VIEW;
        src0_slice.view_src       = dst->src[0];  // non-const pointer to src0
        src0_slice.data           = ggml_rpp_get_quantized_data(src0, i02, expert_stride);
        // GGML_ASSERT(expert_stride == nb02);
        RPP_MEMCPY_DEV_AND_HOST(rpp_node->binding_io_buffers[1], src0_slice.data, expert_stride, rtMemcpyDeviceToDevice,
                                stream);

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;
        RPP_MEMCPY_DEV_AND_HOST(rpp_node->binding_io_buffers[0], src1_data_cur, src1_slice.nb[2],
                                rtMemcpyDeviceToDevice, stream);

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        // compute add operator
        try {
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }

        RPP_MEMCPY_DEV_AND_HOST(dst_data_cur, rpp_node->binding_io_buffers[2], dst_slice.nb[2], rtMemcpyDeviceToDevice,
                                stream, 1);

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur += dst_slice.nb[2];
    }

    ggml_rpp_get_rows_cpy(dst_sorted.ptr, ids_from_sorted_host.data(), dst->data, ne_get_rows, ne0 * ts_dst_sorted,
                          dst_nb1_internal,
                          stream);
    RPP_CHECK(rtStreamSynchronize(stream));
    return true;
}

bool ggml_rpp_op_kernel_mul_mat_id(ggml_backend_rpp_context & ctx,
                                   ggml_tensor *              dst,
                                   int                        is_instantial,
                                   int                        is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_mul_mat_id * rpp_node = nullptr;
    auto                    iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_mul_mat_id");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_mat_mul_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_mul_mat_id *) cur_node;
                    break;
                }
            }
        }

        // find rpp_node from other graph, because of rpp_node can be shared if have same dims
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_mul_mat");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                if (!rpp_graph_tmp) {
                    continue;
                }
                for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter.first == dst) {
                        continue;
                    }
                    if (node_iter.first->op == GGML_OP_MUL_MAT_ID && ggml_rpp_dims_is_same(node_iter.first, dst)) {
                        auto & node_vec = node_iter.second;
                        for (size_t i = 0; i < node_vec.size(); i++) {
                            auto cur_node = node_vec[i].get();
                            if (ggml_rpp_mat_mul_id_dims_is_same(dst, node_iter.first, cur_node) &&
                                cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                                cur_node->cur_ggml_tensor->src[0]->type == dst->src[0]->type) {
                                auto new_node = std::make_unique<rpp_kernel_mul_mat_id>(dst, cur_node);
                                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                                rpp_node = (rpp_kernel_mul_mat_id *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                                // set io tensor, src[0] is weights, src[1] is inputs_0
                                void * inputs_0_buf = cur_node->binding_io_buffers[0];
                                void * weights_buf  = cur_node->binding_io_buffers[1];
                                void * outputs_buf  = cur_node->binding_io_buffers[2];
                                rpp_node->binding_i_buffers.emplace(dst->src[1], inputs_0_buf);
                                rpp_node->binding_i_buffers.emplace(dst->src[0], weights_buf);
                                rpp_node->binding_o_buffers.emplace(dst, outputs_buf);
                                rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
                                rpp_node->binding_io_buffers.emplace_back(weights_buf);
                                rpp_node->binding_io_buffers.emplace_back(outputs_buf);
                                // set node properties
                                ggml_rpp_node_set_properties(rpp_node, dst);
                                // set ubatch
                                rpp_node->n_ubatch      = cur_node->n_ubatch;
                                rpp_node->seq_len_index = cur_node->seq_len_index;
                                rpp_node->sram_io       = ((rpp_kernel_mul_mat_id *) cur_node)->sram_io;
                                rpp_node->is_instantial = is_instantial;
                                break;
                            }
                        }
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
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_mul_mat_id");
            auto new_node = std::make_unique<rpp_kernel_mul_mat_id>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_mul_mat_id *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_mul_mat_id *) (iter->second);
    }
    bool r_launch = true;
    if (is_launch) {
        if (ggml_rpp_mat_mul_id_seq_len(dst, rpp_node) > 1) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_mul_mat_id_ubatch");
            r_launch = ggml_rpp_launch_kernel_ubatch(ctx, rpp_node, dst);
        } else {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_mul_mat_id");
            r_launch = ggml_rpp_launch_kernel(ctx, rpp_node, dst, 1);
        }
    }
    return r_launch;
}
