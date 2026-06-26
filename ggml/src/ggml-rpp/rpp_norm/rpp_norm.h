#ifndef RPP_NORM
#define RPP_NORM

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

bool ggml_rpp_op_kernel_norm(ggml_backend_rpp_context & ctx,
                             struct ggml_tensor *       dst,
                             int                        is_instantial = 1,
                             int                        is_launch     = 1);

struct rpp_kernel_norm : public rpp_node_kernel {
    explicit rpp_kernel_norm(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_norm(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {}

    void init_workspace(ggml_backend_rpp_context & ctx) {
        this->kernel_ctx->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc((64 * 1024) * (int) sizeof(uint16_t)));
    }

    void init_workspace(RPPdeviceptr dev_workspace) { this->kernel_ctx->dev_workspace = dev_workspace; }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_norm(ctx, dst, is_instantial, is_launch);
    }
};

inline bool ggml_rpp_op_norm(ggml_backend_rpp_context & ctx,
                             struct ggml_tensor *       dst,
                             int                        is_instantial = 1,
                             int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_norm(ctx, dst, is_instantial, is_launch);
}

#endif  // RPP_NORM_H
