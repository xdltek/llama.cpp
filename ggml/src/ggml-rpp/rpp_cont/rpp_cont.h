#ifndef RPP_CONT
#define RPP_CONT

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>
bool ggml_rpp_op_memcpy_cont(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

bool ggml_rpp_op_kernel_cont(ggml_backend_rpp_context & ctx,
                             struct ggml_tensor *       dst,
                             int                        is_instantial = 1,
                             int                        is_launch     = 1);

bool ggml_rpp_cont_supports_op(const struct ggml_tensor * dst);

enum rpp_cont_exec_mode {
    RPP_CONT_EXEC_DMA = 0,
    RPP_CONT_EXEC_MEMCPY_ALIGN = 1,
};

struct rpp_kernel_cont : public rpp_node_kernel {
    explicit rpp_kernel_cont(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_cont(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {}

    rpp_cont_exec_mode exec_mode = RPP_CONT_EXEC_DMA;

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_cont(ctx, dst, is_instantial, is_launch);
    }
};

inline bool ggml_rpp_op_cont(ggml_backend_rpp_context & ctx,
                             struct ggml_tensor *       dst,
                             int                        is_instantial = 1,
                             int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_cont(ctx, dst, is_instantial, is_launch);
}

#endif
