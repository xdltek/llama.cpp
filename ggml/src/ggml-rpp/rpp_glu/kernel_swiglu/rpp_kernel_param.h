
#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"

#include <assert.h>
#include <math.h>
#include <rpp_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace kernel_swiglu {
static inline uint32_t silu_round_up_512(uint32_t x) {
    return (x + 511u) / 512u * 512u + 512u;
}

static inline void silu_mode2_output_shape(int C, int H, int W, int split_axis, int & Co, int & Ho, int & Wo) {
    Co = C;
    Ho = H;
    Wo = W;
    if (split_axis == 0) {
        Co = C / 2;
    } else if (split_axis == 1) {
        Ho = H / 2;
    } else if (split_axis == 2) {
        Wo = W / 2;
    }
}

struct silu_sram_io {
    int          mode                  = 0;
    int          split_axis            = 0;
    int          C                     = 0;
    int          H                     = 0;
    int          W                     = 0;
    int          Co                    = 0;
    int          Ho                    = 0;
    int          Wo                    = 0;
    int          in_bytes_per_element  = 0;
    int          out_bytes_per_element = 0;
    uint32_t     tile_in_elements      = 0;
    uint32_t     tile_out_elements     = 0;
    uint32_t     lut_bytes             = 64u * 1024u * (uint32_t) sizeof(uint16_t);
    uint32_t     size_in0              = 0;
    uint32_t     size_in1              = 0;
    uint32_t     size_in0_bf16         = 0;
    uint32_t     size_in1_bf16         = 0;
    uint32_t     size_out_bf16         = 0;
    uint32_t     size_out              = 0;
    uint32_t     total_sram_bytes      = 0;
    RPPdeviceptr sram_base             = 0;
    RPPdeviceptr sram_lut              = 0;
    RPPdeviceptr sram_in0              = 0;
    RPPdeviceptr sram_in1              = 0;
    RPPdeviceptr sram_in0_bf16         = 0;
    RPPdeviceptr sram_in1_bf16         = 0;
    RPPdeviceptr sram_out_bf16         = 0;
    RPPdeviceptr sram_out              = 0;
    RPPdeviceptr sram_out_final        = 0;
};

static inline silu_sram_io silu_prepare_sram_io(rpp_kernel_context & ctx,
                                                int                  mode,
                                                int                  C,
                                                int                  H,
                                                int                  W,
                                                int                  split_axis,
                                                int                  in_bytes_per_element,
                                                int                  out_bytes_per_element,
                                                bool                 bind_ctx_io = true) {
    silu_sram_io io;
    io.mode                  = mode;
    io.split_axis            = split_axis;
    io.C                     = C;
    io.H                     = H;
    io.W                     = W;
    io.Co                    = C;
    io.Ho                    = H;
    io.Wo                    = W;
    io.in_bytes_per_element  = in_bytes_per_element;
    io.out_bytes_per_element = out_bytes_per_element;

    if (mode == 2) {
        silu_mode2_output_shape(C, H, W, split_axis, io.Co, io.Ho, io.Wo);
    }

    io.tile_in_elements  = (uint32_t) ((uint64_t) C * (uint64_t) H * (uint64_t) W);
    io.tile_out_elements = (uint32_t) ((uint64_t) io.Co * (uint64_t) io.Ho * (uint64_t) io.Wo);
    io.size_in0          = io.tile_in_elements * (uint32_t) in_bytes_per_element;
    io.size_in1          = (mode == 1) ? io.size_in0 : 0u;
    io.size_in0_bf16     = io.tile_in_elements * (uint32_t) sizeof(uint16_t);
    io.size_in1_bf16     = (mode == 1) ? io.size_in0_bf16 : 0u;
    io.size_out_bf16     = io.tile_out_elements * (uint32_t) sizeof(uint16_t);
    io.size_out          = io.tile_out_elements * (uint32_t) out_bytes_per_element;

    io.sram_base        = ctx.virtual_sram_base;
    io.sram_lut         = io.sram_base;
    io.sram_in0         = io.sram_lut + (RPPdeviceptr) silu_round_up_512(io.lut_bytes);
    io.sram_in1         = io.sram_in0 + (RPPdeviceptr) silu_round_up_512(io.size_in0);
    io.sram_in0_bf16    = io.sram_in1 + (RPPdeviceptr) silu_round_up_512(io.size_in1);
    io.sram_in1_bf16    = io.sram_in0_bf16 + (RPPdeviceptr) silu_round_up_512(io.size_in0_bf16);
    io.sram_out_bf16    = io.sram_in1_bf16 + (RPPdeviceptr) silu_round_up_512(io.size_in1_bf16);
    io.sram_out         = io.sram_out_bf16 + (RPPdeviceptr) silu_round_up_512(io.size_out_bf16);
    io.sram_out_final   = (out_bytes_per_element == (int) sizeof(float)) ? io.sram_out : io.sram_out_bf16;
    io.total_sram_bytes = (uint32_t) (io.sram_out + (RPPdeviceptr) silu_round_up_512(io.size_out) - io.sram_base);

    const uint32_t SRAM_LIMIT = 22u * 1024u * 1024u;
    if (io.total_sram_bytes > SRAM_LIMIT) {
        std::fprintf(stderr, "SiLU SRAM overflow: need=%u, limit=%u\n", io.total_sram_bytes, SRAM_LIMIT);
        std::abort();
    }

    if (bind_ctx_io) {
        ctx.dev_in.clear();
        ctx.dev_out.clear();
        ctx.dev_in.emplace_back(io.sram_in0);
        ctx.dev_in.emplace_back((mode == 1) ? io.sram_in1 : 0);
        ctx.dev_in.emplace_back(io.sram_lut);
        ctx.dev_in.emplace_back(io.sram_in0_bf16);
        ctx.dev_in.emplace_back((mode == 1) ? io.sram_in1_bf16 : 0);
        ctx.dev_out.emplace_back(io.sram_out_final);
        ctx.dev_out.emplace_back(io.sram_out_bf16);
    }
    return io;
}

static inline void silu_prepare_sram_io(rpp_kernel_context & ctx,
                                        silu_sram_io &       io,
                                        int                  mode,
                                        int                  C,
                                        int                  H,
                                        int                  W,
                                        int                  split_axis,
                                        int                  in_bytes_per_element,
                                        int                  out_bytes_per_element,
                                        bool                 bind_ctx_io = true) {
    io.mode                  = mode;
    io.split_axis            = split_axis;
    io.C                     = C;
    io.H                     = H;
    io.W                     = W;
    io.Co                    = C;
    io.Ho                    = H;
    io.Wo                    = W;
    io.in_bytes_per_element  = in_bytes_per_element;
    io.out_bytes_per_element = out_bytes_per_element;

    if (mode == 2) {
        silu_mode2_output_shape(C, H, W, split_axis, io.Co, io.Ho, io.Wo);
    }

    io.tile_in_elements  = (uint32_t) ((uint64_t) C * (uint64_t) H * (uint64_t) W);
    io.tile_out_elements = (uint32_t) ((uint64_t) io.Co * (uint64_t) io.Ho * (uint64_t) io.Wo);
    io.size_in0          = io.tile_in_elements * (uint32_t) in_bytes_per_element;
    io.size_in1          = (mode == 1) ? io.size_in0 : 0u;
    io.size_in0_bf16     = io.tile_in_elements * (uint32_t) sizeof(uint16_t);
    io.size_in1_bf16     = (mode == 1) ? io.size_in0_bf16 : 0u;
    io.size_out_bf16     = io.tile_out_elements * (uint32_t) sizeof(uint16_t);
    io.size_out          = io.tile_out_elements * (uint32_t) out_bytes_per_element;

    io.sram_base        = ctx.virtual_sram_base;
    io.sram_lut         = io.sram_base;
    io.sram_in0         = io.sram_lut + (RPPdeviceptr) silu_round_up_512(io.lut_bytes);
    io.sram_in1         = io.sram_in0 + (RPPdeviceptr) silu_round_up_512(io.size_in0);
    io.sram_in0_bf16    = io.sram_in1 + (RPPdeviceptr) silu_round_up_512(io.size_in1);
    io.sram_in1_bf16    = io.sram_in0_bf16 + (RPPdeviceptr) silu_round_up_512(io.size_in0_bf16);
    io.sram_out_bf16    = io.sram_in1_bf16 + (RPPdeviceptr) silu_round_up_512(io.size_in1_bf16);
    io.sram_out         = io.sram_out_bf16 + (RPPdeviceptr) silu_round_up_512(io.size_out_bf16);
    io.sram_out_final   = (out_bytes_per_element == (int) sizeof(float)) ? io.sram_out : io.sram_out_bf16;
    io.total_sram_bytes = (uint32_t) (io.sram_out + (RPPdeviceptr) silu_round_up_512(io.size_out) - io.sram_base);

    const uint32_t SRAM_LIMIT = 22u * 1024u * 1024u;
    if (io.total_sram_bytes > SRAM_LIMIT) {
        std::fprintf(stderr, "SiLU SRAM overflow: need=%u, limit=%u\n", io.total_sram_bytes, SRAM_LIMIT);
        std::abort();
    }

    if (bind_ctx_io) {
        ctx.dev_in.clear();
        ctx.dev_out.clear();
        ctx.dev_in.emplace_back(io.sram_in0);
        ctx.dev_in.emplace_back((mode == 1) ? io.sram_in1 : 0);
        ctx.dev_in.emplace_back(io.sram_lut);
        ctx.dev_in.emplace_back(io.sram_in0_bf16);
        ctx.dev_in.emplace_back((mode == 1) ? io.sram_in1_bf16 : 0);
        ctx.dev_out.emplace_back(io.sram_out_final);
        ctx.dev_out.emplace_back(io.sram_out_bf16);
    }
}

static inline RPPdeviceptr silu_prepare_lut_workspace(rpp_kernel_context & ctx) {
    constexpr uint32_t lut_elements = 64u * 1024u;
    constexpr uint32_t lut_bytes    = lut_elements * (uint32_t) sizeof(uint16_t);

    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        rtMalloc((void **) &dev_lut_workspace, lut_bytes);
        ctx.dev_workspace = dev_lut_workspace;
    }

    std::vector<uint16_t> silu_table(lut_elements);
    for (uint32_t i = 0; i < lut_elements; ++i) {
        uint32_t x0   = i << 16;
        float    x    = *(float *) &x0;
        float    y    = x / (1.0f + std::exp(-x));
        silu_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }
    rtMemcpy((void *) dev_lut_workspace, silu_table.data(), lut_bytes, rtMemcpyHostToDevice);
    return dev_lut_workspace;
}

static inline void silu_copy_lut_workspace_to_sram(const silu_sram_io & io,
                                                   RPPdeviceptr         dev_lut_workspace,
                                                   RPPstream            stream) {
    rtMemcpyAsync((void *) io.sram_lut, (const void *) dev_lut_workspace, io.lut_bytes, rtMemcpyDeviceToSram, stream);
}

static inline void silu_cdma_d2s_async(RPPdeviceptr sram_dst, RPPdeviceptr ddr_src, size_t bytes, RPPstream stream) {
    rtMemcpyAsync((void *) sram_dst, (const void *) ddr_src, bytes, rtMemcpyDeviceToSram, stream);
}

static inline void silu_cdma_s2d_async(RPPdeviceptr ddr_dst, RPPdeviceptr sram_src, size_t bytes, RPPstream stream) {
    rtMemcpyAsync((void *) ddr_dst, (const void *) sram_src, bytes, rtMemcpySramToDevice, stream);
}
}  // namespace kernel_swiglu
