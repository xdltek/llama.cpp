#ifndef RPP_PERMUTE
#define RPP_PERMUTE

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

#if GGML_RPP_USE_RT

bool ggml_rpp_op_openrt_permute(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

struct rpp_openrt_permute : public rpp_node_openrt {
    explicit rpp_openrt_permute(ggml_tensor * tensor) : rpp_node_openrt(tensor) {}

    explicit rpp_openrt_permute(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_openrt(tensor, rpp_node) {}

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_openrt_permute(ctx, dst);
    }
};
#endif

inline bool ggml_rpp_op_permute(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst) {
#if GGML_RPP_USE_RT
    return ggml_rpp_op_openrt_permute(ctx, dst);
#else
    GGML_LOG_ERROR("%s: RPP backend only supports Linux and Windows systems.", __func__);
    return false;
#endif
}

#endif
