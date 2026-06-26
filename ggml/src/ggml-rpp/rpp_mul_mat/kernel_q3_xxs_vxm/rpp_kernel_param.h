
#pragma once
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

namespace kernel_q3_xxs_vxm {

static void q3xxs_super_scale_params(uint32_t                in_scale,
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
    // out_scale   [K/256]     | [8]                  |  [N]
    // out_scale   [z]         | [8]                  |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // qscale_per_word = 4
    // in_scale_lsb   [K/256]  | [2]                  |  [qscale_per_word][N]
    // in_scale_lsb   [K/256]  | [2]                  |  [qscale_per_word][N]
    // in_scale_lsb   [z]      | [unroll]             |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    uint32_t inStrideY          = N;
    uint32_t inStrideZ          = 2 * N;
    uint32_t outStrideY         = N;
    uint32_t outStrideZ         = 8 * N;
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

static void matmul_weights_q3xxs_kernel_params(dim3 &                  blocksPerGrid,
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
    //-----------------------------------------------------------------------------------------------------
    // in_scale   [K/256]   | [8]       | [N]
    // in_scale   [grid.z]  | [loop]    | [x]
    //-----------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------
    // codebook   [K/256]  | [8]    |  [2]       |  [2]      | [2][N]
    // codebook   [grid.z] | [loop] |  [unroll0] |  [unroll] | [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // qsign      [K/256]   |  [8]    |  [2]        |  [16][N]
    // qsign      [grid.z]  |  [loop] |  [unroll0]  |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    uint32_t inUnrollStride  = N * sizeof(short);
    uint32_t blockXSize      = threadsPerBlock.x * sizeof(short);
    uint32_t scaleLoopStride = N * sizeof(short);

    uint32_t inLoopStride0        = 2 * 2 * N * sizeof(short);
    uint32_t inLoopStride1        = 2 * N * sizeof(short);
    uint32_t in_wq_blockz_size    = 8 * inLoopStride0;
    // qsign block-z stride for one K/256 tile.
    uint32_t in_sign_blockz_size  = 8 * inLoopStride1;
    uint32_t in_scale_blockz_size = 8 * scaleLoopStride;
    uint32_t in_a_blockz_size     = weights_group * sizeof(short);
    uint32_t loop                 = 8;

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

static void matmul_weights_q3xxs_batch_params(dim3 &                  blocksPerGrid,
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
                                              uint32_t                in_wq_stridez,
                                              uint32_t                in_sign_stridez,
                                              uint32_t                in_scale_stridez,
                                              uint32_t                in_a_stridez,
                                              uint32_t                combine,
                                              std::vector<uint32_t> & params) {
    (void) blocksPerGrid;
    //-----------------------------------------------------------------------------------------------------
    // in_scale   [K/256]   | [8]       | [N]
    // in_scale   [grid.z]  | [loop]    | [x]
    //-----------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------
    // codebook   [K/256]  | [8]    |  [2]       |  [2]      | [2][N]
    // codebook   [grid.z] | [loop] |  [unroll0] |  [unroll] | [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // qsign      [K/256]   |  [8]    |  [2]        |  [16][N]
    // qsign      [grid.z]  |  [loop] |  [unroll0]  |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    uint32_t inUnrollStride  = N * sizeof(short);
    uint32_t blockXSize      = threadsPerBlock.x * sizeof(short);
    uint32_t scaleLoopStride = N * sizeof(short);

    uint32_t inLoopStride0        = 2 * 2 * N * sizeof(short);
    uint32_t inLoopStride1        = 2 * N * sizeof(short);
    uint32_t in_wq_blockz_size    = 8 * inLoopStride0;
    // qsign block-z stride for one K/256 tile.
    uint32_t in_sign_blockz_size  = 8 * inLoopStride1;
    uint32_t in_scale_blockz_size = 8 * scaleLoopStride;
    uint32_t in_a_blockz_size     = weights_group * sizeof(short);

    uint32_t loop                 = 8;
    // LOADALN.Z32 expects z-stride in 16-bit lanes, not bytes.
    uint32_t in_wq_stridez_u16    = in_wq_stridez / 2;
    uint32_t in_sign_stridez_u16  = in_sign_stridez / 2;
    uint32_t in_scale_stridez_u16 = in_scale_stridez / 2;

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

    params.emplace_back(in_wq_stridez_u16);
    params.emplace_back(in_sign_stridez_u16);
    params.emplace_back(in_scale_stridez_u16);
    params.emplace_back(in_a_stridez);

    params.emplace_back(loop);
    params.emplace_back(combine);
}

static inline int q3xxs_round_up(int a) {
    return (a + 511) / 512 * 512 + 512;
}

struct q3xxs_vxm_sram_io {
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
    uint32_t size_q4_tile              = 0;
    uint32_t size_qscale_tile          = 0;
    uint32_t size_qsign_tile           = 0;
    uint32_t size_super_scale_tile     = 0;
    uint32_t size_weights_expert       = 0;
    uint32_t size_weights_total        = 0;
    uint32_t off_weights_qscale        = 0;
    uint32_t off_weights_qsign         = 0;
    uint32_t off_weights_super         = 0;
    uint32_t sizeB_scale_scratch       = 0;
    uint32_t sizeB_scale_scratch_total = 0;
    uint32_t sizeA_acc_scratch         = 0;
    uint32_t total_sram_bytes          = 0;

    RPPdeviceptr sram_base         = 0;
    RPPdeviceptr sramA             = 0;
    RPPdeviceptr sramC             = 0;
    RPPdeviceptr sramC1            = 0;
    RPPdeviceptr sramB_q4          = 0;
    RPPdeviceptr sramB_qscale      = 0;
    RPPdeviceptr sramB_qsign       = 0;
    RPPdeviceptr sramB_super_scale = 0;
    RPPdeviceptr sramB_qscale_lut  = 0;
    RPPdeviceptr sramB_grid_lut    = 0;
    RPPdeviceptr sramB_scale       = 0;
    RPPdeviceptr sramA_acc         = 0;
};

static q3xxs_vxm_sram_io q3xxs_vxm_prepare_sram_io(rpp_kernel_context & ctx,
                                                   q3xxs_vxm_sram_io &  io,
                                                   int                  M,
                                                   int                  K,
                                                   int                  N,
                                                   int                  weights_group,
                                                   int                  in_bytes_per_element,
                                                   int                  out_bytes_per_element,
                                                   int                  experts,
                                                   bool                 bind_ctx_io = true) {
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

    io.size_q4_tile          = (uint32_t) (Ktile * N / 4);
    io.size_qscale_tile      = (uint32_t) (Ktile * N / 64);
    io.size_qsign_tile       = (uint32_t) (Ktile * N / 8);
    io.size_super_scale_tile = (uint32_t) ((Ktile / weights_group) * N * (int) sizeof(uint16_t));

    io.off_weights_qscale  = io.size_q4_tile;
    io.off_weights_qsign   = io.off_weights_qscale + io.size_qscale_tile;
    io.off_weights_super   = io.off_weights_qsign + io.size_qsign_tile;
    io.size_weights_expert = io.off_weights_super + io.size_super_scale_tile;
    io.size_weights_total  = (uint32_t) ((uint64_t) io.size_weights_expert * (uint64_t) experts);

    io.sizeB_scale_scratch       = (uint32_t) ((Ktile * N / 16) * (int) sizeof(uint16_t));
    io.sizeB_scale_scratch_total = (uint32_t) ((uint64_t) io.sizeB_scale_scratch * (uint64_t) experts);
    io.sizeA_acc_scratch         = (uint32_t) q3xxs_round_up(weights_group * 4);

    constexpr uint32_t qscale_lut_bytes = 16u * (uint32_t) sizeof(uint16_t);
    constexpr uint32_t grid_lut_bytes   = 256u * (uint32_t) sizeof(uint32_t);

    io.sram_base         = ctx.virtual_sram_base;
    io.sramA             = io.sram_base;
    io.sramC             = io.sramA + (RPPdeviceptr) q3xxs_round_up((int) io.sizeA);
    io.sramC1            = io.sramC + (RPPdeviceptr) q3xxs_round_up((int) io.sizeC32);
    io.sramB_q4          = io.sramC1 + (RPPdeviceptr) q3xxs_round_up((int) io.sizeC);
    io.sramB_qscale      = io.sramB_q4 + (RPPdeviceptr) io.off_weights_qscale;
    io.sramB_qsign       = io.sramB_q4 + (RPPdeviceptr) io.off_weights_qsign;
    io.sramB_super_scale = io.sramB_q4 + (RPPdeviceptr) io.off_weights_super;
    io.sramB_qscale_lut  = io.sramB_q4 + (RPPdeviceptr) q3xxs_round_up((int) io.size_weights_total);
    io.sramB_grid_lut    = io.sramB_qscale_lut + (RPPdeviceptr) q3xxs_round_up((int) qscale_lut_bytes);
    io.sramB_scale       = io.sramB_grid_lut + (RPPdeviceptr) q3xxs_round_up((int) grid_lut_bytes);
    io.sramA_acc         = io.sramB_scale + (RPPdeviceptr) q3xxs_round_up((int) io.sizeB_scale_scratch_total);
    io.total_sram_bytes  = (uint32_t) (io.sramA_acc + (RPPdeviceptr) io.sizeA_acc_scratch - io.sram_base);

    const uint32_t SRAM_LIMIT = 22u * 1024u * 1024u;
    if (io.total_sram_bytes > SRAM_LIMIT) {
        std::fprintf(stderr, "Q3XXS SRAM overflow: need=%u, limit=%u\n", io.total_sram_bytes, SRAM_LIMIT);
        std::abort();
    }

    if (bind_ctx_io) {
        ctx.dev_in.clear();
        ctx.dev_out.clear();
        ctx.dev_in.emplace_back(io.sramA);
        ctx.dev_in.emplace_back(io.sramB_q4);
        ctx.dev_in.emplace_back(io.sramB_qscale);
        ctx.dev_in.emplace_back(io.sramB_qsign);
        ctx.dev_in.emplace_back(io.sramB_super_scale);
        ctx.dev_in.emplace_back(io.sramB_qscale_lut);
        ctx.dev_in.emplace_back(io.sramB_grid_lut);
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

static RPPdeviceptr q3xxs_vxm_prepare_lut_workspace(rpp_kernel_context & ctx) {
    constexpr uint32_t qscale_lut_bytes = 16u * (uint32_t) sizeof(uint16_t);
    constexpr uint32_t grid_lut_bytes   = 256u * (uint32_t) sizeof(uint32_t);
    constexpr uint32_t total_bytes      = qscale_lut_bytes + grid_lut_bytes;

    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        rtMalloc((void **) &dev_lut_workspace, total_bytes);
        ctx.dev_workspace = dev_lut_workspace;
    }

    std::array<uint16_t, 16> qscale_lut = {};
    for (uint32_t i = 0; i < qscale_lut.size(); ++i) {
        const float lut_val = (0.5f + (float) i) * 0.5f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }
    std::array<uint32_t, 256> grid_lut = {};
    std::memcpy(grid_lut.data(), iq3xxs_grid_local, grid_lut_bytes);

    rtMemcpy((void *) dev_lut_workspace, qscale_lut.data(), qscale_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) (dev_lut_workspace + qscale_lut_bytes), grid_lut.data(), grid_lut_bytes, rtMemcpyHostToDevice);
    return dev_lut_workspace;
}

static inline void q3xxs_vxm_copy_lut_workspace_to_sram(RPPdeviceptr sram_qscale_lut,
                                                        RPPdeviceptr sram_grid_lut,
                                                        RPPdeviceptr dev_lut_workspace,
                                                        RPPstream    stream) {
    constexpr uint32_t qscale_lut_bytes = 16u * (uint32_t) sizeof(uint16_t);
    constexpr uint32_t grid_lut_bytes   = 256u * (uint32_t) sizeof(uint32_t);
    rtMemcpyAsync((void *) sram_qscale_lut, (const void *) dev_lut_workspace, qscale_lut_bytes, rtMemcpyDeviceToSram,
                  stream);
    rtMemcpyAsync((void *) sram_grid_lut, (const void *) (dev_lut_workspace + qscale_lut_bytes), grid_lut_bytes,
                  rtMemcpyDeviceToSram, stream);
}

static inline void q3xxs_vxm_copy_lut_workspace_to_sram(const q3xxs_vxm_sram_io & io,
                                                        RPPdeviceptr              dev_lut_workspace,
                                                        RPPstream                 stream) {
    q3xxs_vxm_copy_lut_workspace_to_sram(io.sramB_qscale_lut, io.sramB_grid_lut, dev_lut_workspace, stream);
}

static inline void q3xxs_vxm_cdma_d2s_async(RPPdeviceptr sram_dst,
                                            RPPdeviceptr ddr_src,
                                            size_t       bytes,
                                            RPPstream    stream) {
    rtMemcpyAsync((void *) sram_dst, (const void *) ddr_src, bytes, rtMemcpyDeviceToSram, stream);
}

static inline void q3xxs_vxm_cdma_s2d_async(RPPdeviceptr ddr_dst,
                                            RPPdeviceptr sram_src,
                                            size_t       bytes,
                                            RPPstream    stream) {
    rtMemcpyAsync((void *) ddr_dst, (const void *) sram_src, bytes, rtMemcpySramToDevice, stream);
}

static inline void q3xxs_vxm_cdma_copy_expert_weights_to_sram(const q3xxs_vxm_sram_io & io,
                                                              RPPdeviceptr              dev_expert_weights_base,
                                                              int                       sram_idx,
                                                              int                       expert_idx,
                                                              RPPstream                 stream) {
    if (sram_idx < 0 || sram_idx >= io.experts) {
        std::cerr << "q2xs_vxm_nolut_cdma_copy_expert_weights_to_sram expert_idx out of range\n";
        std::abort();
    }
    const RPPdeviceptr sram_dst = io.sramB_q4 + (RPPdeviceptr) sram_idx * (RPPdeviceptr) io.size_weights_expert;
    const RPPdeviceptr ddr_src =
        dev_expert_weights_base + (RPPdeviceptr) expert_idx * (RPPdeviceptr) io.size_weights_expert;
    q3xxs_vxm_cdma_d2s_async(sram_dst, ddr_src, (size_t) io.size_weights_expert, stream);
}

static inline void q3xxs_vxm_cdma_copy_output_to_ddr(const q3xxs_vxm_sram_io & io,
                                                     RPPdeviceptr              devC,
                                                     RPPstream                 stream) {
    const RPPdeviceptr sram_out = (io.out_bytes_per_element == (int) sizeof(float)) ? io.sramC1 : io.sramC;
    q3xxs_vxm_cdma_s2d_async(devC, sram_out, io.sizeC, stream);
}
}  // namespace kernel_q3_xxs_vxm
