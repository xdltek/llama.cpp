#ifndef RPP_REDUCE_SUM_H
#define RPP_REDUCE_SUM_H

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

bool ggml_rpp_op_kernel_reduce_sum(ggml_backend_rpp_context & ctx,
                                   struct ggml_tensor *       dst,
                                   int                        is_instantial = 1,
                                   int                        is_launch     = 1);

struct rpp_kernel_reduce_sum : public rpp_node_kernel {
    explicit rpp_kernel_reduce_sum(ggml_tensor * tensor) : rpp_node_kernel(tensor) { op = RPP_OP_REDUCE_SUM; }

    explicit rpp_kernel_reduce_sum(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {
        op = RPP_OP_REDUCE_SUM;
    }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_reduce_sum(ctx, dst, is_instantial, is_launch);
    }
};

inline bool ggml_rpp_op_reduce_sum(ggml_backend_rpp_context & ctx,
                                   struct ggml_tensor *       dst,
                                   int                        is_instantial = 1,
                                   int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_reduce_sum(ctx, dst, is_instantial, is_launch);
}

#endif  // RPP_REDUCE_SUM_H
