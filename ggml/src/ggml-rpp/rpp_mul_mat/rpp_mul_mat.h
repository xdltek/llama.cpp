#ifndef RPP_MUL_MAT
#define RPP_MUL_MAT

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

#if GGML_RPP_USE_RT

bool ggml_rpp_op_openrt_mul_mat(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

struct rpp_openrt_mul_mat : public rpp_node_openrt {
    explicit rpp_openrt_mul_mat(ggml_tensor * tensor) : rpp_node_openrt(tensor) {}

    explicit rpp_openrt_mul_mat(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_openrt(tensor, rpp_node) {}

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_openrt_mul_mat(ctx, dst);
    }
};
#endif

bool ggml_rpp_op_kernel_mul_mat(ggml_backend_rpp_context & ctx,
                                struct ggml_tensor *       dst,
                                int                        is_instantial = 1,
                                int                        is_launch     = 1);

struct rpp_kernel_mul_mat : public rpp_node_kernel {
    explicit rpp_kernel_mul_mat(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_mul_mat(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {}

    // all rms_norm kernel will shared the same workspace,
    void init_workspace(ggml_backend_rpp_context & ctx) {
        // Shared LUT workspace for quantized matmul kernels.
        // Largest current user is IQ2_S (~8 KB), keep headroom for future kernels.
        this->kernel_ctx->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc(64 * 1024));
    }

    void init_workspace(RPPdeviceptr dev_workspace) { this->kernel_ctx->dev_workspace = dev_workspace; }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_mul_mat(ctx, dst, is_instantial, is_launch);
    }
};

inline bool ggml_rpp_op_mul_mat(ggml_backend_rpp_context & ctx,
                                struct ggml_tensor *       dst,
                                int                        is_instantial = 1,
                                int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_mul_mat(ctx, dst, is_instantial, is_launch);
}

#endif
