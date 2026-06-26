
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

namespace kernel_q2_xs_vxm_nolut {
static void q2xs_super_scale_params(uint32_t                in_scale,
                                    uint32_t                in_super_scale,
                                    uint32_t                in_lut,
                                    uint32_t                out_scale,
                                    uint32_t                K,
                                    uint32_t                N,
                                    uint32_t                super_group,
                                    uint32_t                q_group,
                                    dim3 &                  blocksPerGrid,
                                    dim3 &                  threadsPerBlock,
                                    std::vector<uint32_t> & params) {
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    //----------------------------------------------------------------------------------------------------
    // in_super_scale   [K/256]  |     | [N]
    // in_super_scale   [sg]     |     | [N]
    // in_super_scale   [z]      | [1] | [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // out_scale   [K/256]     | [16]     |  [N]
    // out_scale   [z]         | [16]     |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // in_scale   [K/256]  | [4]          |  [4][N]
    // in_scale   [K/256]  | [4]          |  [4][N]
    // in_scale   [z]      | [unroll]     |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    uint32_t inStrideY          = N;
    uint32_t inStrideZ          = 4 * N;
    uint32_t outStrideY         = N;
    uint32_t outStrideZ         = 16 * N;
    uint32_t inUnRollStride     = N * sizeof(short);
    uint32_t outUnRollStride    = N * sizeof(short);
    uint32_t stride_per_block_x = threadsPerBlock.x * sizeof(short);

    uint32_t blockX = blocksPerGrid.x;
    blocksPerGrid.x = 1;
    params.emplace_back(in_scale);
    params.emplace_back(in_super_scale);
    params.emplace_back(out_scale);
    params.emplace_back(in_lut);
    params.emplace_back(inStrideY);
    params.emplace_back(inStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(inUnRollStride);
    params.emplace_back(outUnRollStride);
    params.emplace_back(stride_per_block_x);
    params.emplace_back(blockX);
}

static void matmul_weights_q2xs_kernel_params(dim3 &                  blocksPerGrid,
                                              dim3 &                  threadsPerBlock,
                                              uint32_t                in_act,
                                              uint32_t                in_wq,
                                              uint32_t                in_scale,
                                              uint32_t                in_sign,
                                              uint32_t                out_addr,
                                              uint32_t                lut_addr,
                                              uint32_t                input_acc_addr,
                                              uint32_t                input_acc_addr_hi,
                                              uint32_t                N,
                                              uint32_t                hilo_stride,
                                              uint32_t                weights_group,
                                              uint32_t                combine,
                                              std::vector<uint32_t> & params) {
    //----------------------------------------------------------------------------------------------------
    // scale      [K/256]     |  [4]     |  [4]                |  [N]
    // scale      [grid.z]    |  [loop]  |  [unroll0]          |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // qsign      [K/256]     |  [4]     |  [4]                |  [16][N]
    // qsign      [grid.z]    |  [loop]  |  [unroll0]          |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // codebook   [K/256]      |  [4]     |  [4]       |  [2]  |  [8][N]
    // codebook   [grid.z]     |  [loop]  |  [unroll0] |       |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    (void) blocksPerGrid;
    uint32_t inUnrollStride  = N * sizeof(short);
    uint32_t blockXSize      = threadsPerBlock.x * sizeof(short);
    uint32_t scaleLoopStride = 4 * N * sizeof(short);

    uint32_t inLoopStride0        = 4 * 2 * N * sizeof(short);
    uint32_t inLoopStride1        = 4 * N * sizeof(short);
    uint32_t in_wq_blockz_size    = 4 * inLoopStride0;
    uint32_t in_sign_blockz_size  = 4 * inLoopStride1;
    uint32_t in_scale_blockz_size = 4 * inLoopStride1;
    uint32_t in_a_blockz_size     = weights_group * sizeof(short);
    uint32_t loop                 = 4;

    params.emplace_back(in_act);
    params.emplace_back(in_wq);
    params.emplace_back(in_sign);
    params.emplace_back(in_scale);
    params.emplace_back(out_addr);
    params.emplace_back(lut_addr);
    params.emplace_back(inUnrollStride);
    params.emplace_back(inLoopStride0);
    params.emplace_back(inLoopStride1);
    params.emplace_back(scaleLoopStride);
    params.emplace_back(blockXSize);
    params.emplace_back(in_wq_blockz_size);
    params.emplace_back(in_sign_blockz_size);
    params.emplace_back(in_scale_blockz_size);
    params.emplace_back(in_a_blockz_size);
    params.emplace_back(input_acc_addr);
    params.emplace_back(input_acc_addr_hi);
    params.emplace_back(hilo_stride);
    params.emplace_back(loop);
    params.emplace_back(combine);
}

static void matmul_weights_q2xs_nolut_batch_params(dim3 &                  blocksPerGrid,
                                                   dim3 &                  threadsPerBlock,
                                                   uint32_t                in_act,
                                                   uint32_t                in_wq,
                                                   uint32_t                in_scale,
                                                   uint32_t                in_sign,
                                                   uint32_t                out_addr,
                                                   uint32_t                lut_addr,
                                                   uint32_t                input_acc_addr,
                                                   uint32_t                input_acc_addr_hi,
                                                   uint32_t                N,
                                                   uint32_t                hilo_stride,
                                                   uint32_t                weights_group,
                                                   uint32_t                in_wq_expert_stride_bytes,
                                                   uint32_t                in_sign_expert_stride_bytes,
                                                   uint32_t                in_scale_expert_stride_bytes,
                                                   uint32_t                in_act_expert_stride_bytes,
                                                   uint32_t                combine,
                                                   std::vector<uint32_t> & params) {
    //----------------------------------------------------------------------------------------------------
    // scale      [K/256]     |  [4]     |  [4]                |  [N]
    // scale      [grid.z]    |  [loop]  |  [unroll0]          |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // qsign      [K/256]     |  [4]     |  [4]                |  [16][N]
    // qsign      [grid.z]    |  [loop]  |  [unroll0]          |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // codebook   [K/256]      |  [4]     |  [4]       |  [2]  |  [8][N]
    // codebook   [grid.z]     |  [loop]  |  [unroll0] |       |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    (void) blocksPerGrid;
    uint32_t inUnrollStride  = N * sizeof(short);
    uint32_t blockXSize      = threadsPerBlock.x * sizeof(short);
    uint32_t scaleLoopStride = 4 * N * sizeof(short);

    uint32_t inLoopStride0        = 4 * 2 * N * sizeof(short);
    uint32_t inLoopStride1        = 4 * N * sizeof(short);
    uint32_t in_wq_blockz_size    = 4 * inLoopStride0;
    uint32_t in_sign_blockz_size  = 4 * inLoopStride1;
    uint32_t in_scale_blockz_size = 4 * inLoopStride1;

    uint32_t in_wq_stridez    = in_wq_expert_stride_bytes / 2;
    uint32_t in_sign_stridez  = in_sign_expert_stride_bytes / 2;
    uint32_t in_scale_stridez = in_scale_expert_stride_bytes / 2;

    uint32_t in_a_blockz_size = weights_group * sizeof(short);
    uint32_t loop             = 4;

    params.emplace_back(in_act);
    params.emplace_back(in_wq);
    params.emplace_back(in_sign);
    params.emplace_back(in_scale);
    params.emplace_back(out_addr);
    params.emplace_back(lut_addr);
    params.emplace_back(inUnrollStride);
    params.emplace_back(inLoopStride0);
    params.emplace_back(inLoopStride1);
    params.emplace_back(scaleLoopStride);
    params.emplace_back(blockXSize);
    params.emplace_back(in_wq_blockz_size);
    params.emplace_back(in_sign_blockz_size);
    params.emplace_back(in_scale_blockz_size);
    params.emplace_back(in_a_blockz_size);
    params.emplace_back(input_acc_addr);
    params.emplace_back(input_acc_addr_hi);
    params.emplace_back(hilo_stride);

    params.emplace_back(in_wq_stridez);
    params.emplace_back(in_sign_stridez);
    params.emplace_back(in_scale_stridez);
    params.emplace_back(in_act_expert_stride_bytes);

    params.emplace_back(loop);
    params.emplace_back(combine);
}

static inline int q2xs_round_up(int a) {
    return (a + 511) / 512 * 512 + 512;
}

struct q2xs_vxm_nolut_sram_io {
    int M                     = 0;
    int K                     = 0;
    int N                     = 0;
    int experts               = 1;
    int weights_group         = 0;
    int in_bytes_per_element  = 0;
    int out_bytes_per_element = 0;

    uint32_t sizeA                     = 0;
    uint32_t sizeC32                   = 0;
    uint32_t sizeC                     = 0;
    uint32_t size_codebook_tile        = 0;
    uint32_t size_scales_tile          = 0;
    uint32_t size_sign_tile            = 0;
    uint32_t size_super_scale_tile     = 0;
    uint32_t size_weights_expert       = 0;
    uint32_t size_weights_total        = 0;
    uint32_t off_weights_scales        = 0;
    uint32_t off_weights_sign          = 0;
    uint32_t off_weights_super         = 0;
    uint32_t sizeB_scale_scratch       = 0;
    uint32_t sizeB_scale_scratch_total = 0;
    uint32_t sizeA_acc_scratch         = 0;
    uint32_t total_sram_bytes          = 0;

    RPPdeviceptr sram_base            = 0;
    RPPdeviceptr sramA                = 0;
    RPPdeviceptr sramC                = 0;
    RPPdeviceptr sramC1               = 0;
    RPPdeviceptr sramB_codebook_nolut = 0;
    RPPdeviceptr sramB_scales         = 0;
    RPPdeviceptr sramB_sign           = 0;
    RPPdeviceptr sramB_super_scale    = 0;
    RPPdeviceptr sramB_qscale_lut     = 0;
    RPPdeviceptr sramB_mag_lut        = 0;
    RPPdeviceptr sramB_scale          = 0;
    RPPdeviceptr sramA_acc            = 0;
};

static void q2xs_vxm_nolut_prepare_sram_io(rpp_kernel_context &     ctx,
                                           q2xs_vxm_nolut_sram_io & io,
                                           int                      M,
                                           int                      K,
                                           int                      N,
                                           int                      weights_group,
                                           int                      in_bytes_per_element,
                                           int                      out_bytes_per_element,
                                           int                      experts,
                                           bool                     bind_ctx_io = true) {
    io.M                     = M;
    io.K                     = K;
    io.N                     = N;
    io.experts               = experts;
    io.weights_group         = weights_group;
    io.in_bytes_per_element  = in_bytes_per_element;
    io.out_bytes_per_element = out_bytes_per_element;

    const int Ktile = K;
    io.sizeA        = (uint32_t) ((uint64_t) M * (uint64_t) K * (uint64_t) experts * (uint64_t) in_bytes_per_element);
    io.sizeC32      = (uint32_t) ((uint64_t) N * (uint64_t) experts * sizeof(float));
    io.sizeC        = (uint32_t) ((uint64_t) N * (uint64_t) experts * (uint64_t) out_bytes_per_element);

    io.size_codebook_tile    = (uint32_t) (Ktile * N / 4);
    io.size_scales_tile      = (uint32_t) (Ktile * N / 32);
    io.size_sign_tile        = (uint32_t) (Ktile * N / 8);
    io.size_super_scale_tile = (uint32_t) ((Ktile / weights_group) * N * (int) sizeof(uint16_t));

    io.off_weights_scales  = io.size_codebook_tile;
    io.off_weights_sign    = io.off_weights_scales + io.size_scales_tile;
    io.off_weights_super   = io.off_weights_sign + io.size_sign_tile;
    io.size_weights_expert = io.off_weights_super + io.size_super_scale_tile;
    io.size_weights_total  = (uint32_t) ((uint64_t) io.size_weights_expert * (uint64_t) experts);

    io.sizeB_scale_scratch       = (uint32_t) ((Ktile * N / 16) * (int) sizeof(uint16_t));
    io.sizeB_scale_scratch_total = (uint32_t) ((uint64_t) io.sizeB_scale_scratch * (uint64_t) experts);
    io.sizeA_acc_scratch         = (uint32_t) q2xs_round_up(weights_group * 4);

    constexpr uint32_t qscale_lut_bytes = 16u * (uint32_t) sizeof(uint16_t);
    constexpr uint32_t mag_lut_bytes    = 4u * (uint32_t) sizeof(uint16_t);

    io.sram_base            = ctx.virtual_sram_base;
    io.sramA                = io.sram_base;
    io.sramC                = io.sramA + (RPPdeviceptr) q2xs_round_up((int) io.sizeA);
    io.sramC1               = io.sramC + (RPPdeviceptr) q2xs_round_up((int) io.sizeC32);
    io.sramB_codebook_nolut = io.sramC1 + (RPPdeviceptr) q2xs_round_up((int) io.sizeC);
    io.sramB_scales         = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_scales;
    io.sramB_sign           = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_sign;
    io.sramB_super_scale    = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_super;
    io.sramB_qscale_lut     = io.sramB_codebook_nolut + (RPPdeviceptr) q2xs_round_up((int) io.size_weights_total);
    io.sramB_mag_lut        = io.sramB_qscale_lut + (RPPdeviceptr) q2xs_round_up((int) qscale_lut_bytes);
    io.sramB_scale          = io.sramB_mag_lut + (RPPdeviceptr) q2xs_round_up((int) mag_lut_bytes);
    io.sramA_acc            = io.sramB_scale + (RPPdeviceptr) q2xs_round_up((int) io.sizeB_scale_scratch_total);
    io.total_sram_bytes     = (uint32_t) (io.sramA_acc + (RPPdeviceptr) io.sizeA_acc_scratch - io.sram_base);

    const uint32_t SRAM_LIMIT = 22u * 1024u * 1024u;
    if (io.total_sram_bytes > SRAM_LIMIT) {
        std::fprintf(stderr, "Q2XS SRAM overflow: need=%u, limit=%u\n", io.total_sram_bytes, SRAM_LIMIT);
        std::abort();
    }

    if (bind_ctx_io) {
        ctx.dev_in.clear();
        ctx.dev_out.clear();
        ctx.dev_in.emplace_back(io.sramA);
        ctx.dev_in.emplace_back(io.sramB_codebook_nolut);
        ctx.dev_in.emplace_back(io.sramB_scales);
        ctx.dev_in.emplace_back(io.sramB_sign);
        ctx.dev_in.emplace_back(io.sramB_super_scale);
        ctx.dev_in.emplace_back(io.sramB_qscale_lut);
        ctx.dev_in.emplace_back(io.sramB_mag_lut);
        ctx.dev_in.emplace_back(io.sramB_scale);
        ctx.dev_in.emplace_back(io.sramA_acc);

        if (out_bytes_per_element == (int) sizeof(float)) {
            ctx.dev_out.emplace_back(io.sramC1);
            ctx.dev_out.emplace_back(io.sramC);
        } else {
            ctx.dev_out.emplace_back(io.sramC);
        }
    }
}

static RPPdeviceptr q2xs_vxm_nolut_prepare_lut_workspace(rpp_kernel_context & ctx) {
    constexpr uint32_t qscale_lut_bytes = 16u * (uint32_t) sizeof(uint16_t);
    constexpr uint32_t mag_lut_bytes    = 4u * (uint32_t) sizeof(uint16_t);
    constexpr uint32_t total_bytes      = qscale_lut_bytes + mag_lut_bytes;

    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        rtMalloc((void **) &dev_lut_workspace, total_bytes);
        ctx.dev_workspace = dev_lut_workspace;
    }

    std::array<uint16_t, 16> qscale_lut = {};
    for (uint32_t i = 0; i < qscale_lut.size(); ++i) {
        const float lut_val = (0.5f + (float) i) * 0.25f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }
    std::array<uint16_t, 4> mag_lut = {};
    const float             mags[4] = { 8.0f, 25.0f, 43.0f, 0.0f };
    for (int i = 0; i < 4; ++i) {
        mag_lut[(size_t) i] = float_to_bf16_rne(mags[i]);
    }

    rtMemcpy((void *) dev_lut_workspace, qscale_lut.data(), qscale_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) (dev_lut_workspace + qscale_lut_bytes), mag_lut.data(), mag_lut_bytes, rtMemcpyHostToDevice);
    return dev_lut_workspace;
}

static inline void q2xs_vxm_nolut_copy_lut_workspace_to_sram(RPPdeviceptr sram_qscale_lut,
                                                             RPPdeviceptr sram_mag_lut,
                                                             RPPdeviceptr dev_lut_workspace,
                                                             RPPstream    stream) {
    constexpr uint32_t qscale_lut_bytes = 16u * (uint32_t) sizeof(uint16_t);
    constexpr uint32_t mag_lut_bytes    = 4u * (uint32_t) sizeof(uint16_t);
    rtMemcpyAsync((void *) sram_qscale_lut, (const void *) dev_lut_workspace, qscale_lut_bytes, rtMemcpyDeviceToSram,
                  stream);
    rtMemcpyAsync((void *) sram_mag_lut, (const void *) (dev_lut_workspace + qscale_lut_bytes), mag_lut_bytes,
                  rtMemcpyDeviceToSram, stream);
}

static inline void q2xs_vxm_nolut_copy_lut_workspace_to_sram(const q2xs_vxm_nolut_sram_io & io,
                                                             RPPdeviceptr                   dev_lut_workspace,
                                                             RPPstream                      stream) {
    q2xs_vxm_nolut_copy_lut_workspace_to_sram(io.sramB_qscale_lut, io.sramB_mag_lut, dev_lut_workspace, stream);
}

static inline void q2xs_vxm_nolut_cdma_d2s_async(RPPdeviceptr sram_dst,
                                                 RPPdeviceptr ddr_src,
                                                 size_t       bytes,
                                                 RPPstream    stream) {
    rtMemcpyAsync((void *) sram_dst, (const void *) ddr_src, bytes, rtMemcpyDeviceToSram, stream);
}

static inline void q2xs_vxm_nolut_cdma_s2d_async(RPPdeviceptr ddr_dst,
                                                 RPPdeviceptr sram_src,
                                                 size_t       bytes,
                                                 RPPstream    stream) {
    rtMemcpyAsync((void *) ddr_dst, (const void *) sram_src, bytes, rtMemcpySramToDevice, stream);
}

// Copy one expert packed weights chunk [codebook|scales|sign|super_scale] from DDR to SRAM.
static inline void q2xs_vxm_nolut_cdma_copy_expert_weights_to_sram(const q2xs_vxm_nolut_sram_io & io,
                                                                   RPPdeviceptr dev_expert_weights_base,
                                                                   int          sram_idx,
                                                                   int          expert_idx,
                                                                   RPPstream    stream) {
    if (sram_idx < 0 || sram_idx >= io.experts) {
        std::cerr << "q2xs_vxm_nolut_cdma_copy_expert_weights_to_sram expert_idx out of range\n";
        std::abort();
    }
    const RPPdeviceptr sram_dst =
        io.sramB_codebook_nolut + (RPPdeviceptr) sram_idx * (RPPdeviceptr) io.size_weights_expert;
    const RPPdeviceptr ddr_src =
        dev_expert_weights_base + (RPPdeviceptr) expert_idx * (RPPdeviceptr) io.size_weights_expert;
    q2xs_vxm_nolut_cdma_d2s_async(sram_dst, ddr_src, (size_t) io.size_weights_expert, stream);
}

static inline void q2xs_vxm_nolut_cdma_copy_output_to_ddr(const q2xs_vxm_nolut_sram_io & io,
                                                          RPPdeviceptr                   devC,
                                                          RPPstream                      stream) {
    const RPPdeviceptr sram_out = (io.out_bytes_per_element == (int) sizeof(float)) ? io.sramC1 : io.sramC;
    q2xs_vxm_nolut_cdma_s2d_async(devC, sram_out, io.sizeC, stream);
}
}  // namespace kernel_q2_xs_vxm_nolut
