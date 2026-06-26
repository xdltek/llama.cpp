#ifndef RPP_FLASH_ATTN_EXT
#define RPP_FLASH_ATTN_EXT

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

#if GGML_RPP_USE_RT

bool ggml_rpp_op_openrt_flash_attn_ext(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

struct rpp_openrt_flash_attn_ext : public rpp_node_openrt {
    explicit rpp_openrt_flash_attn_ext(ggml_tensor * tensor) : rpp_node_openrt(tensor) {}

    explicit rpp_openrt_flash_attn_ext(ggml_tensor * tensor, ggml_rpp_node * rpp_node) :
        rpp_node_openrt(tensor, rpp_node) {}

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_openrt_flash_attn_ext(ctx, dst);
    }
};
#endif

bool ggml_rpp_op_kernel_flash_attn_ext(ggml_backend_rpp_context & ctx,
                                       struct ggml_tensor *       dst,
                                       int                        is_instantial = 1,
                                       int                        is_launch     = 1);

struct rpp_kernel_flash_attn_ext : public rpp_node_kernel {
    explicit rpp_kernel_flash_attn_ext(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_flash_attn_ext(ggml_tensor * tensor, ggml_rpp_node * rpp_node) :
        rpp_node_kernel(tensor, rpp_node) {}

    // all flash attn ext kernel will shared the same workspace,
    void init_workspace(ggml_backend_rpp_context & ctx) {
        this->kernel_ctx->dev_workspace =
            (RPPdeviceptr) (ctx.pool().alloc((64 * 1024 + 64 * 1024) * (int) sizeof(uint16_t)));
    }

    void init_workspace(RPPdeviceptr dev_workspace) { this->kernel_ctx->dev_workspace = dev_workspace; }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_flash_attn_ext(ctx, dst, is_instantial, is_launch);
    }

    RPPdeviceptr dev_shared_kparas{ 0 };

    size_t shared_kpara_size{ 0 };

    uint32_t    kv_length{ 0 };

    ~rpp_kernel_flash_attn_ext() {
        if (dev_shared_kparas != 0) {
            RPP_CHECK(rppGraphResourceFree(dev_shared_kparas, RPP_GRAPH_RESOURCE_KPARA));
        }
    }
};

inline bool ggml_rpp_op_flash_attn_ext(ggml_backend_rpp_context & ctx,
                                       struct ggml_tensor *       dst,
                                       int                        is_instantial = 1,
                                       int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_flash_attn_ext(ctx, dst, is_instantial, is_launch);
}
#endif  // RPP_FLASH_ATTN_EXT
