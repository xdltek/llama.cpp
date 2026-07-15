#include "ggml-cpu/ops.h"
#include "rpp_concat/rpp_concat.h"

#include <limits>

struct ggml_compute_params {
    int ith, nth;
    size_t wsize;
    void * wdata;
    struct ggml_threadpool * threadpool;
};

static bool ggml_rpp_concat_check_shape(const ggml_tensor * dst) {
    if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type != src1->type || src0->type != dst->type) {
        return false;
    }

    const int32_t dim = ggml_get_op_params_i32(dst, 0);
    if (dim < 0 || dim >= GGML_MAX_DIMS) {
        return false;
    }

    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d == dim) {
            if (dst->ne[d] != src0->ne[d] + src1->ne[d]) {
                return false;
            }
            continue;
        }
        if (src0->ne[d] != src1->ne[d] || dst->ne[d] != src0->ne[d]) {
            return false;
        }
    }

    return true;
}

bool ggml_rpp_concat_supports_op(const ggml_tensor * dst) {
    if (!ggml_rpp_concat_check_shape(dst)) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // Keep current RPP concat scope narrow: contiguous tensors with identical dtype.
    return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
}

static bool ggml_rpp_op_cpu_concat(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst != nullptr);
    GGML_ASSERT(dst->src[0] != nullptr);
    GGML_ASSERT(dst->src[1] != nullptr);

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    const size_t src0_size = ggml_nbytes(src0);
    const size_t src1_size = ggml_nbytes(src1);
    const size_t dst_size  = ggml_nbytes(dst);

    void * src0_host = ctx.pool_host().alloc(src0_size);
    void * src1_host = ctx.pool_host().alloc(src1_size);
    void * dst_host  = ctx.pool_host().alloc(dst_size);

    RPP_MEMCPY_DEV_AND_HOST(src0_host, src0->data, src0_size, rtMemcpyDeviceToHost, ctx.stream(), 1);
    RPP_MEMCPY_DEV_AND_HOST(src1_host, src1->data, src1_size, rtMemcpyDeviceToHost, ctx.stream(), 1);

    const void * src0_dev = src0->data;
    const void * src1_dev = src1->data;
    void *       dst_dev  = dst->data;

    src0->data = src0_host;
    src1->data = src1_host;
    dst->data  = dst_host;

    struct ggml_compute_params params = {
        /*.ith=*/0,
        /*.nth=*/1,
        /*.wsize=*/0,
        /*.wdata=*/nullptr,
        /*.threadpool=*/nullptr,
    };
    ggml_compute_forward_concat(&params, dst);

    src0->data = const_cast<void *>(src0_dev);
    src1->data = const_cast<void *>(src1_dev);
    dst->data  = dst_dev;

    RPP_MEMCPY_DEV_AND_HOST(dst_dev, dst_host, dst_size, rtMemcpyHostToDevice, ctx.stream(), 0);

    ctx.pool_host().free(src0_host);
    ctx.pool_host().free(src1_host);
    ctx.pool_host().free(dst_host);

    return true;
}

static bool ggml_rpp_op_device_concat(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int32_t dim = ggml_get_op_params_i32(dst, 0);
    const size_t  ts  = ggml_type_size(dst->type);

    uint64_t inner = 1;
    uint64_t outer = 1;
    for (int d = 0; d < dim; ++d) {
        inner *= (uint64_t) dst->ne[d];
    }
    for (int d = dim + 1; d < GGML_MAX_DIMS; ++d) {
        outer *= (uint64_t) dst->ne[d];
    }

    const uint64_t chunk0 = (uint64_t) src0->ne[dim] * inner * (uint64_t) ts;
    const uint64_t chunk1 = (uint64_t) src1->ne[dim] * inner * (uint64_t) ts;
    const uint64_t pitch  = (uint64_t) dst->ne[dim] * inner * (uint64_t) ts;

    if (chunk0 > std::numeric_limits<size_t>::max() || chunk1 > std::numeric_limits<size_t>::max() ||
        pitch > std::numeric_limits<size_t>::max() || outer > std::numeric_limits<size_t>::max()) {
        GGML_LOG_ERROR("%s: concat size overflow\n", __func__);
        return false;
    }

    for (uint64_t o = 0; o < outer; ++o) {
        const size_t src0_off = (size_t) (o * chunk0);
        const size_t src1_off = (size_t) (o * chunk1);
        const size_t dst_off  = (size_t) (o * pitch);

        const char * src0_ptr = (const char *) src0->data + src0_off;
        const char * src1_ptr = (const char *) src1->data + src1_off;
        char *       dst_ptr  = (char *) dst->data + dst_off;

        RPP_MEMCPY_DEV_AND_HOST(dst_ptr, src0_ptr, (size_t) chunk0, rtMemcpyDeviceToDevice, ctx.stream(), 0);
        RPP_MEMCPY_DEV_AND_HOST(dst_ptr + chunk0, src1_ptr, (size_t) chunk1, rtMemcpyDeviceToDevice, ctx.stream(), 0);
    }

    return true;
}

bool ggml_rpp_op_kernel_concat(ggml_backend_rpp_context & ctx,
                               ggml_tensor *              dst,
                               int                        is_instantial,
                               int                        is_launch) {
    GGML_UNUSED(is_instantial);

    if (dst == nullptr) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr\n", __func__);
        return false;
    }

    if (!is_launch) {
        return true;
    }

    if (!ggml_rpp_concat_check_shape(dst)) {
        GGML_LOG_ERROR("%s: invalid concat shape for %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    if (ggml_rpp_concat_supports_op(dst)) {
        return ggml_rpp_op_device_concat(ctx, dst);
    }

    return ggml_rpp_op_cpu_concat(ctx, dst);
}
