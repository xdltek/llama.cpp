#ifndef RPP_CPY
#define RPP_CPY

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

bool ggml_rpp_op_cpu_cpy(ggml_backend_rpp_context & ctx, ggml_tensor * dst);

bool ggml_rpp_op_kernel_cpy(ggml_backend_rpp_context & ctx,
                            ggml_tensor *              dst,
                            int                        is_instantial = 1,
                            int                        is_launch     = 1);

struct rpp_kernel_cpy : public rpp_node_kernel {
    explicit rpp_kernel_cpy(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_cpy(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {}

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_cpy(ctx, dst, is_instantial, is_launch);
    }

    bool use_cpu_fallback{ false };
};

inline bool ggml_rpp_op_cpy(ggml_backend_rpp_context & ctx,
                            struct ggml_tensor *       dst,
                            int                        is_instantial = 1,
                            int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_cpy(ctx, dst, is_instantial, is_launch);
}

#endif
