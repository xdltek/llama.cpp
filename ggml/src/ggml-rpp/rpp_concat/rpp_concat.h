#ifndef RPP_CONCAT
#define RPP_CONCAT

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

bool ggml_rpp_op_kernel_concat(ggml_backend_rpp_context & ctx,
                               ggml_tensor *              dst,
                               int                        is_instantial = 1,
                               int                        is_launch     = 1);

bool ggml_rpp_concat_supports_op(const ggml_tensor * dst);

inline bool ggml_rpp_op_concat(ggml_backend_rpp_context & ctx,
                               struct ggml_tensor *       dst,
                               int                        is_instantial = 1,
                               int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_concat(ctx, dst, is_instantial, is_launch);
}

#endif
