#pragma once

#include "ggml-rpp/rpp_kernel_ctx.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>

static inline void rpp_cont_memcpy_align_build(rpp_kernel_context & ctx,
                                               uint32_t             oD0,
                                               uint32_t             oD1,
                                               uint32_t             oD2,
                                               uint32_t             in_stride_x,
                                               uint32_t             in_stride_y,
                                               uint32_t             in_stride_z,
                                               size_t               input_span_bytes,
                                               uint32_t             bytes_per_element,
                                               int                  is_instantial = 1) {
    (void) input_span_bytes;
    const size_t width_bytes = (size_t) bytes_per_element;
    const size_t src_pitch   = (size_t) in_stride_x * (size_t) bytes_per_element;

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    for (uint32_t z = 0; z < oD0; ++z) {
        for (uint32_t y = 0; y < oD1; ++y) {
            uint8_t * src = (uint8_t *) ctx.dev_in[0] +
                            ((size_t) z * (size_t) in_stride_z + (size_t) y * (size_t) in_stride_y) *
                                (size_t) bytes_per_element;
            uint8_t * dst = (uint8_t *) ctx.dev_out[0] +
                            ((size_t) z * (size_t) oD1 * (size_t) oD2 + (size_t) y * (size_t) oD2) *
                                (size_t) bytes_per_element;
            rtMemcpy2DAsync((void *) dst, width_bytes, (const void *) src, src_pitch, width_bytes, oD2,
                            rtMemcpyDeviceToDevice, ctx.kernelStream);
        }
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
