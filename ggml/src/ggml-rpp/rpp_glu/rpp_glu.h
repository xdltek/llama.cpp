#ifndef RPP_GLU
#define RPP_GLU

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

#if GGML_RPP_USE_RT

bool ggml_rpp_op_openrt_glu(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

struct rpp_openrt_glu : public rpp_node_openrt {
    explicit rpp_openrt_glu(ggml_tensor * tensor) : rpp_node_openrt(tensor) {}

    explicit rpp_openrt_glu(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_openrt(tensor, rpp_node) {}

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_openrt_glu(ctx, dst);
    }
};

#endif

bool ggml_rpp_op_kernel_glu(ggml_backend_rpp_context & ctx,
                            struct ggml_tensor *       dst,
                            int                        is_instantial = 1,
                            int                        is_launch     = 1);

struct rpp_kernel_glu : public rpp_node_kernel {
    explicit rpp_kernel_glu(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_glu(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {}

    ~rpp_kernel_glu() {
        if (sram_io) {
            delete sram_io;
            sram_io = nullptr;
        }
    }

    // all glu kernel will shared the same workspace,
    void init_workspace(ggml_backend_rpp_context & ctx) {
        this->kernel_ctx->dev_workspace = (RPPdeviceptr) (ctx.pool().alloc((64 * 1024) * (int) sizeof(uint16_t)));
    }

    void init_workspace(RPPdeviceptr dev_workspace) { this->kernel_ctx->dev_workspace = dev_workspace; }

    void * sram_io{ nullptr };

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_glu(ctx, dst, is_instantial, is_launch);
    }
};

inline bool ggml_rpp_op_glu(ggml_backend_rpp_context & ctx,
                            struct ggml_tensor *       dst,
                            int                        is_instantial = 1,
                            int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_glu(ctx, dst, is_instantial, is_launch);
}

#endif  // RPP_GLU_H
