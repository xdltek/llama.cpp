#ifndef RPP_GET_ROWS
#define RPP_GET_ROWS

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

#if GGML_RPP_USE_RT

bool ggml_rpp_op_openrt_get_rows(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

bool ggml_rpp_op_cpy_get_rows(ggml_backend_rpp_context & ctx, ggml_tensor * dst);

struct rpp_openrt_get_rows : public rpp_node_openrt {
    explicit rpp_openrt_get_rows(ggml_tensor * tensor) : rpp_node_openrt(tensor) {}

    explicit rpp_openrt_get_rows(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_openrt(tensor, rpp_node) {}

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_openrt_get_rows(ctx, dst);
    }
};

#endif

bool ggml_rpp_op_kernel_get_rows(ggml_backend_rpp_context & ctx,
                                 struct ggml_tensor *       dst,
                                 int                        is_instantial = 1,
                                 int                        is_launch     = 1);

struct rpp_kernel_get_rows : public rpp_node_kernel {
    explicit rpp_kernel_get_rows(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_get_rows(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {}

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_get_rows(ctx, dst, is_instantial, is_launch);
    }
};

inline bool ggml_rpp_op_get_rows(ggml_backend_rpp_context & ctx,
                                 struct ggml_tensor *       dst,
                                 int                        is_instantial = 1,
                                 int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_get_rows(ctx, dst, is_instantial, is_launch);
}
#endif
