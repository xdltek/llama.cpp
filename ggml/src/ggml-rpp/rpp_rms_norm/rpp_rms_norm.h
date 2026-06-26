#ifndef RPP_RMS_NORM
#define RPP_RMS_NORM

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

#if GGML_RPP_USE_RT

bool ggml_rpp_op_openrt_rms_norm(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

struct rpp_openrt_rms_norm : public rpp_node_openrt {
    explicit rpp_openrt_rms_norm(ggml_tensor * tensor) : rpp_node_openrt(tensor) {}

    explicit rpp_openrt_rms_norm(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_openrt(tensor, rpp_node) {}

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_openrt_rms_norm(ctx, dst);
    }
};
#endif
bool ggml_rpp_op_kernel_rms_norm(ggml_backend_rpp_context & ctx,
                                 struct ggml_tensor *       dst,
                                 int                        is_instantial = 1,
                                 int                        is_launch     = 1);

struct rpp_kernel_rms_norm : public rpp_node_kernel {
    explicit rpp_kernel_rms_norm(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_rms_norm(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {}

    // all rms_norm kernel will shared the same workspace,
    void init_workspace(ggml_backend_rpp_context & ctx) {
        this->kernel_ctx->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc((64 * 1024) * (int) sizeof(uint16_t)));
    }

    void init_workspace(RPPdeviceptr dev_workspace) { this->kernel_ctx->dev_workspace = dev_workspace; }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_rms_norm(ctx, dst, is_instantial, is_launch);
    }
};

inline bool ggml_rpp_op_rms_norm(ggml_backend_rpp_context & ctx,
                                 struct ggml_tensor *       dst,
                                 int                        is_instantial = 1,
                                 int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_rms_norm(ctx, dst, is_instantial, is_launch);
}

bool ggml_rpp_op_kernel_rms_norm_mul_fusion(ggml_backend_rpp_context & ctx,
                                            struct ggml_tensor *       rms_tensor,
                                            struct ggml_tensor *       mul_tensor,
                                            int                        is_instantial = 1,
                                            int                        is_launch     = 1);

struct rpp_kernel_rms_norm_mul_fusion : public rpp_node_kernel {
    explicit rpp_kernel_rms_norm_mul_fusion(ggml_tensor * tensor, ggml_tensor * mul_tensor) :
        rpp_node_kernel(tensor),
        mul_tensor(mul_tensor) {
        op = RPP_OP_RMS_MUL;
    }

    explicit rpp_kernel_rms_norm_mul_fusion(ggml_tensor * tensor, ggml_rpp_node * rpp_node, ggml_tensor * mul_tensor) :
        rpp_node_kernel(tensor, rpp_node),
        mul_tensor(mul_tensor) {
        op = RPP_OP_RMS_MUL;
    }

    // Keep consistent with rms_norm workspace policy.
    void init_workspace(ggml_backend_rpp_context & ctx) {
        this->kernel_ctx->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc((64 * 1024) * (int) sizeof(uint16_t)));
    }

    void init_workspace(RPPdeviceptr dev_workspace) { this->kernel_ctx->dev_workspace = dev_workspace; }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_rms_norm_mul_fusion(ctx, dst, mul_tensor, is_instantial, is_launch);
    }

    ggml_tensor * mul_tensor{ nullptr };
};

inline bool ggml_rpp_op_rms_norm_mul_fusion(ggml_backend_rpp_context & ctx,
                                            struct ggml_tensor *       rms_tensor,
                                            struct ggml_tensor *       mul_tensor,
                                            int                        is_instantial = 1,
                                            int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_rms_norm_mul_fusion(ctx, rms_tensor, mul_tensor, is_instantial, is_launch);
}

#endif
