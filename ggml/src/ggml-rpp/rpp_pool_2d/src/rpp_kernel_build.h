#pragma once

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

inline int rpp_pool_2d_round_up(int a) {
    return (a + 511) / 512 * 512 + 512;
}

inline void rpp_pool_2d_build(rpp_kernel_context & ctx,
                              int                  in_w,
                              int                  in_h,
                              int                  channels,
                              int                  out_w,
                              int                  out_h,
                              int                  k0,
                              int                  k1,
                              int                  s0,
                              int                  s1,
                              int                  in_bytes_per_element,
                              int                  out_bytes_per_element,
                              int                  is_instantial = 1) {
    if (in_w <= 0 || in_h <= 0 || channels <= 0 || out_w <= 0 || out_h <= 0) {
        throw std::runtime_error("POOL_2D invalid tensor dimensions");
    }
    if (k0 != 2 || k1 != 2 || s0 != 2 || s1 != 2) {
        throw std::runtime_error("POOL_2D RPP kernel currently supports AVG 2x2 stride 2 only");
    }
    if ((channels % 32) != 0) {
        throw std::runtime_error("POOL_2D RPP HWC32 kernel requires channels to be a multiple of 32");
    }
    if (in_w != 32 || in_h != 32 || channels != 1152 || out_w != 16 || out_h != 16) {
        throw std::runtime_error("POOL_2D RPP HWC32 kernel currently mirrors the rpprt Phi-4 shape only");
    }
    if (in_bytes_per_element != (int) sizeof(float) || out_bytes_per_element != (int) sizeof(float)) {
        throw std::runtime_error("POOL_2D RPP kernel currently supports F32 input/output only");
    }
    if (ctx.dev_in.empty() || ctx.dev_out.empty()) {
        throw std::runtime_error("POOL_2D requires ctx.dev_in[0] and ctx.dev_out[0]");
    }

    RPPdeviceptr dev_in  = ctx.dev_in[0];
    RPPdeviceptr dev_out = ctx.dev_out[0];

    const int in_plane_elems  = in_w * in_h;
    const int out_plane_elems = out_w * out_h;
    const int in_elems        = channels * in_plane_elems;
    const int out_elems       = channels * out_plane_elems;
    const int size_in         = in_elems * in_bytes_per_element;
    const int size_out        = out_elems * out_bytes_per_element;

    const int size_in16       = in_elems * (int) sizeof(uint16_t);
    const int size_out16      = out_elems * (int) sizeof(uint16_t);
    const int channel_groups  = (channels + 31) / 32;
    const int size_in_hwc32   = channel_groups * 32 * in_plane_elems * (int) sizeof(uint16_t);
    const int size_out_hwc32  = channel_groups * 32 * out_plane_elems * (int) sizeof(uint16_t);
    const int size_out_hwc33  = channel_groups * 33 * out_plane_elems * (int) sizeof(uint16_t);

    RPPdeviceptr sram_base      = ctx.virtual_sram_base;
    RPPdeviceptr sram_in_f32    = sram_base;
    RPPdeviceptr sram_in16      = sram_in_f32 + rpp_pool_2d_round_up(size_in);
    RPPdeviceptr sram_in_hwc32  = sram_in16 + rpp_pool_2d_round_up(size_in16);
    RPPdeviceptr sram_out_hwc32 = sram_in_hwc32 + rpp_pool_2d_round_up(size_in_hwc32);
    RPPdeviceptr sram_out_hwc33 = sram_out_hwc32 + rpp_pool_2d_round_up(size_out_hwc32);
    RPPdeviceptr sram_out16     = sram_out_hwc33 + rpp_pool_2d_round_up(size_out_hwc33);
    RPPdeviceptr sram_out_f32   = sram_out16 + rpp_pool_2d_round_up(size_out16);

    const int total_sram_bytes = (int) (sram_out_f32 + rpp_pool_2d_round_up(size_out) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "POOL_2D SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                  << " bytes\n";
        std::abort();
    }

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/pool_2d.o");

    rtMemcpyAsync((void *) sram_in_f32, (const void *) dev_in, size_in, rtMemcpyDeviceToSram, ctx.kernelStream);

    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    calc_tbdim_flattern(1, in_elems, threadsPerBlock, blocksPerGrid);
    params.clear();
    cvt_kernel_param_init(threadsPerBlock, (uint32_t) sram_in_f32, (uint32_t) sram_in16, kFLOAT, kBF16, params);
    launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    for (int cg = 0; cg < channel_groups; ++cg) {
        const uint32_t input_group  = (uint32_t) (sram_in16 + cg * 32 * in_plane_elems * sizeof(uint16_t));
        const uint32_t output_group = (uint32_t) (sram_in_hwc32 + cg * 32 * in_plane_elems * sizeof(uint16_t));

        threadsPerBlock.x = 32;
        threadsPerBlock.y = 255;
        threadsPerBlock.z = 1;
        blocksPerGrid.x   = 4;
        blocksPerGrid.y   = 1;
        blocksPerGrid.z   = 1;
        params.clear();
        params.reserve(7);
        params.push_back(input_group);
        params.push_back(output_group);
        params.push_back((uint32_t) (threadsPerBlock.y * sizeof(uint16_t)));
        params.push_back((uint32_t) (32 * threadsPerBlock.y * sizeof(uint16_t)));
        params.push_back((uint32_t) (in_plane_elems * sizeof(uint16_t)));
        params.push_back(32);
        params.push_back(0);
        launchWrapperAysnc("opt_chw2hwc32_16b", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        threadsPerBlock.x = 32;
        threadsPerBlock.y = 4;
        threadsPerBlock.z = 1;
        blocksPerGrid.x   = 1;
        blocksPerGrid.y   = 1;
        blocksPerGrid.z   = 1;
        params.clear();
        params.reserve(7);
        params.push_back((uint32_t) (input_group + 1020 * sizeof(uint16_t)));
        params.push_back((uint32_t) (output_group + 32 * 1020 * sizeof(uint16_t)));
        params.push_back((uint32_t) (threadsPerBlock.y * sizeof(uint16_t)));
        params.push_back((uint32_t) (32 * threadsPerBlock.y * sizeof(uint16_t)));
        params.push_back((uint32_t) (in_plane_elems * sizeof(uint16_t)));
        params.push_back(32);
        params.push_back(0);
        launchWrapperAysnc("opt_chw2hwc32_16b", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    const float avg_scale = 1.0f / (float) (k0 * k1);
    uint32_t    avg_scale_bits;
    std::memcpy(&avg_scale_bits, &avg_scale, sizeof(avg_scale_bits));

    const int pool_bytes_per_element = (int) sizeof(uint16_t);
    threadsPerBlock.x = 32;
    threadsPerBlock.y = 16;
    threadsPerBlock.z = 8;
    blocksPerGrid.x   = (uint32_t) channel_groups;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 2;
    params.clear();
    params.reserve(22);
    params.push_back((uint32_t) sram_in_hwc32);                                  // input
    params.push_back((uint32_t) sram_out_hwc32);                                 // out
    params.push_back((uint32_t) (32 * pool_bytes_per_element));                  // inStrideY
    params.push_back((uint32_t) 32);                                             // outStrideY
    params.push_back((uint32_t) (in_w * 32 * pool_bytes_per_element));           // inStrideZ
    params.push_back((uint32_t) (out_w * 32));                                   // outStrideZ
    params.push_back((uint32_t) (in_plane_elems * 32 * pool_bytes_per_element)); // inBlockSize
    params.push_back((uint32_t) (out_plane_elems * 32 * pool_bytes_per_element));// outBlockSize
    params.push_back((uint32_t) (32 * pool_bytes_per_element));                  // offset0
    params.push_back((uint32_t) ((in_w - k0) * 32 * pool_bytes_per_element));    // offset1
    params.push_back((uint32_t) k0);                                             // rpt0
    params.push_back(1);                                                         // Un
    params.push_back((uint32_t) (threadsPerBlock.z * s1 * in_w * 32 * pool_bytes_per_element)); // inUnStride
    params.push_back((uint32_t) (threadsPerBlock.z * out_w * 32 * pool_bytes_per_element));     // outUnStride
    params.push_back(1);                                                         // Bn
    params.push_back((uint32_t) k1);                                             // rpt_m1
    params.push_back(avg_scale_bits);                                            // ap_scale
    params.push_back((uint32_t) (threadsPerBlock.y * s0 * 32 * pool_bytes_per_element));        // inBlockYStride
    params.push_back((uint32_t) (threadsPerBlock.y * 32 * pool_bytes_per_element));             // outBlockYStride
    params.push_back((uint32_t) (threadsPerBlock.z * s1 * in_w * 32 * pool_bytes_per_element)); // inBlockZStride
    params.push_back((uint32_t) (threadsPerBlock.z * out_w * 32 * pool_bytes_per_element));     // outBlockZStride
    params.push_back(0);                                                         // tail_block_y
    launchWrapperAysnc("ap_f16_f32_f16_hwc32", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 144;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = 64;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;
    params.clear();
    params.reserve(9);
    params.push_back((uint32_t) sram_out_hwc32);           // input
    params.push_back((uint32_t) sram_out_hwc33);           // output
    params.push_back((uint32_t) (threadsPerBlock.x * threadsPerBlock.y * sizeof(uint16_t)));
    params.push_back(1);
    params.push_back((uint32_t) ((threadsPerBlock.x + 1) * threadsPerBlock.y * sizeof(uint16_t)));
    params.push_back(32);
    params.push_back(0);
    params.push_back(33);
    params.push_back(0);
    launchWrapperAysnc("hwc322hwc128_f16_f32_f16_all", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 32;
    threadsPerBlock.z = 4;
    blocksPerGrid.x   = (uint32_t) channel_groups;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;
    params.clear();
    params.reserve(7);
    params.push_back((uint32_t) sram_out_hwc33);                              // input
    params.push_back((uint32_t) sram_out16);                                  // output
    params.push_back((uint32_t) (33 * sizeof(uint16_t)));                     // src_x_stribe
    params.push_back((uint32_t) out_plane_elems);                             // store_jump_size
    params.push_back(2);                                                       // loop_n
    params.push_back((uint32_t) (out_plane_elems * 33 * sizeof(uint16_t)));    // in_block_size
    params.push_back((uint32_t) (out_plane_elems * 32 * sizeof(uint16_t)));    // out_block_size
    launchWrapperAysnc("hwc322chw_f16_f32_f16_all_opt1", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    params.clear();
    calc_tbdim_flattern(1, out_elems * 2, threadsPerBlock, blocksPerGrid);
    cvt_kernel_param_init_opt(threadsPerBlock, (uint32_t) sram_out16, (uint32_t) sram_out_f32, kBF16, kFLOAT, params);
    launchWrapperAysnc("opt_vector_cvt_f16_f32_v2", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    rtMemcpyAsync((void *) dev_out, (const void *) sram_out_f32, size_out, rtMemcpySramToDevice, ctx.kernelStream);

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        if (rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0) != RPP_SUCCESS) {
            throw std::runtime_error("POOL_2D rppGraphInstantiate failed");
        }
    }
}
