#include "rpp_cont/rpp_cont.h"

bool ggml_rpp_op_memcpy_cont(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst){
    TRACE_SCOPE_GUARD(ctx.trace_id, "execute_memcpy_cont");
    ggml_rpp_pack_tensor_to_contiguous(ctx, dst->src[0], dst->data, dst->src[0]->view_offs);
    return false;
}