
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

namespace kernel_q3_xxs_vxm_nolut {
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
    uint32_t in_sign_blockz_size  = 8 * 2 * inLoopStride1;
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

constexpr int                  kQ3xxsWeightsGroup = 256;
constexpr int                  kQ3xxsQGroup       = 32;
constexpr uint32_t             kSramLimitBytes    = 22u * 1024u * 1024u;
constexpr std::array<float, 8> kQ3xxsMagValues    = {
    4.0f, 12.0f, 20.0f, 28.0f, 36.0f, 44.0f, 52.0f, 62.0f,
};

inline size_t align_up(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

inline int q3xxs_vxm_nolut_codebook_rows(int K) {
    assert((K % 16) == 0);
    return (K / 16) * 3;
}

inline size_t q3xxs_vxm_nolut_codebook_bytes(int K, int N) {
    return (size_t) q3xxs_vxm_nolut_codebook_rows(K) * (size_t) N * sizeof(uint16_t);
}

struct q3xxs_vxm_nolut_lut_workspace {
    static constexpr uint32_t qscale_lut_elems        = 16;
    static constexpr uint32_t mag_lut_elems           = (uint32_t) kQ3xxsMagValues.size();
    static constexpr uint32_t mat_lut_codes_per_entry = 4;
    static constexpr uint32_t mat_lut_rows            = 1u << (3u * mat_lut_codes_per_entry);
    static constexpr uint32_t mat_lut_cols            = mat_lut_codes_per_entry;
    static constexpr uint32_t qscale_lut_bytes        = qscale_lut_elems * (uint32_t) sizeof(uint16_t);
    static constexpr uint32_t mag_lut_bytes           = mag_lut_elems * (uint32_t) sizeof(uint16_t);
    static constexpr uint32_t mat_lut_bytes           = mat_lut_rows * mat_lut_cols * (uint32_t) sizeof(uint16_t);
};

struct q3xxs_vxm_nolut_sram_io {
    int M                     = 0;
    int K                     = 0;
    int Ktile                 = 0;
    int N                     = 0;
    int experts               = 1;
    int weights_group         = 0;
    int in_bytes_per_element  = 0;
    int out_bytes_per_element = 0;

    uint32_t sizeA                     = 0;
    uint32_t sizeC32                   = 0;
    uint32_t sizeC                     = 0;
    uint32_t size_codebook_nolut_tile  = 0;
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
    RPPdeviceptr sramB_mat_lut        = 0;
    RPPdeviceptr sramB_scale          = 0;
};

struct q3xxs_vxm_nolut_workspace_ptrs {
    RPPdeviceptr base        = 0;
    RPPdeviceptr qscale_lut  = 0;
    RPPdeviceptr mag_lut     = 0;
    RPPdeviceptr mat_lut     = 0;
    uint32_t     total_bytes = 0;
};

static q3xxs_vxm_nolut_workspace_ptrs q3xxs_vxm_nolut_prepare_lut_workspace(rpp_kernel_context & ctx) {
    q3xxs_vxm_nolut_workspace_ptrs out{};
    const size_t                   off_qscale = 0;
    const size_t off_mag     = align_up(off_qscale + (size_t) q3xxs_vxm_nolut_lut_workspace::qscale_lut_bytes, 64);
    const size_t off_mat     = align_up(off_mag + (size_t) q3xxs_vxm_nolut_lut_workspace::mag_lut_bytes, 64);
    const size_t total_bytes = align_up(off_mat + (size_t) q3xxs_vxm_nolut_lut_workspace::mat_lut_bytes, 64);

    if (ctx.dev_workspace == 0) {
        if (rtMalloc((void **) &ctx.dev_workspace, total_bytes) != rtSuccess) {
            throw std::runtime_error("Q3XXS NoLUT LUT workspace allocation failed");
        }
    }

    out.base        = ctx.dev_workspace;
    out.qscale_lut  = out.base + (RPPdeviceptr) off_qscale;
    out.mag_lut     = out.base + (RPPdeviceptr) off_mag;
    out.mat_lut     = out.base + (RPPdeviceptr) off_mat;
    out.total_bytes = (uint32_t) total_bytes;

    std::array<uint16_t, q3xxs_vxm_nolut_lut_workspace::qscale_lut_elems> qscale_lut = {};
    for (uint32_t i = 0; i < qscale_lut.size(); ++i) {
        const float lut_val = (0.5f + (float) i) * 0.5f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }

    std::array<uint16_t, q3xxs_vxm_nolut_lut_workspace::mag_lut_elems> mag_lut = {};
    for (uint32_t i = 0; i < mag_lut.size(); ++i) {
        mag_lut[i] = float_to_bf16_rne(kQ3xxsMagValues[i]);
    }

    std::vector<uint16_t> mat_lut(
        (size_t) q3xxs_vxm_nolut_lut_workspace::mat_lut_rows * (size_t) q3xxs_vxm_nolut_lut_workspace::mat_lut_cols, 0);
    for (uint32_t packed = 0; packed < q3xxs_vxm_nolut_lut_workspace::mat_lut_rows; ++packed) {
        for (uint32_t u = 0; u < q3xxs_vxm_nolut_lut_workspace::mat_lut_cols; ++u) {
            const uint8_t code = (uint8_t) ((packed >> (3u * u)) & 0x7u);
            mat_lut[(size_t) packed * (size_t) q3xxs_vxm_nolut_lut_workspace::mat_lut_cols + (size_t) u] =
                float_to_bf16_rne(kQ3xxsMagValues[code]);
        }
    }

    rtMemcpy((void *) out.qscale_lut, qscale_lut.data(), q3xxs_vxm_nolut_lut_workspace::qscale_lut_bytes,
             rtMemcpyHostToDevice);
    rtMemcpy((void *) out.mag_lut, mag_lut.data(), q3xxs_vxm_nolut_lut_workspace::mag_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) out.mat_lut, mat_lut.data(), q3xxs_vxm_nolut_lut_workspace::mat_lut_bytes, rtMemcpyHostToDevice);
    return out;
}

static inline void q3xxs_vxm_nolut_copy_lut_workspace_to_sram(RPPdeviceptr                           sram_qscale_lut,
                                                              RPPdeviceptr                           sram_mag_lut,
                                                              RPPdeviceptr                           sram_mat_lut,
                                                              const q3xxs_vxm_nolut_workspace_ptrs & lut_ws,
                                                              RPPstream                              stream) {
    rtMemcpyAsync((void *) sram_qscale_lut, (const void *) lut_ws.qscale_lut,
                  q3xxs_vxm_nolut_lut_workspace::qscale_lut_bytes, rtMemcpyDeviceToSram, stream);
    rtMemcpyAsync((void *) sram_mag_lut, (const void *) lut_ws.mag_lut, q3xxs_vxm_nolut_lut_workspace::mag_lut_bytes,
                  rtMemcpyDeviceToSram, stream);
    rtMemcpyAsync((void *) sram_mat_lut, (const void *) lut_ws.mat_lut, q3xxs_vxm_nolut_lut_workspace::mat_lut_bytes,
                  rtMemcpyDeviceToSram, stream);
}

static void q3xxs_vxm_nolut_prepare_sram_io(rpp_kernel_context &      ctx,
                                            q3xxs_vxm_nolut_sram_io & io,
                                            int                       M,
                                            int                       K,
                                            int                       N,
                                            int                       weights_group,
                                            int                       in_bytes_per_element,
                                            int                       out_bytes_per_element,
                                            int                       experts,
                                            bool                      bind_ctx_io) {
    if (weights_group != kQ3xxsWeightsGroup || (K % weights_group) != 0) {
        throw std::runtime_error("Matmul Q3XXS NoLUT SRAM expects weights_group == 256 and K % 256 == 0");
    }
    if (experts <= 0) {
        throw std::runtime_error("Matmul Q3XXS NoLUT SRAM expects experts > 0");
    }

    io.M                     = M;
    io.K                     = K;
    io.Ktile                 = K;
    io.N                     = N;
    io.experts               = experts;
    io.weights_group         = weights_group;
    io.in_bytes_per_element  = in_bytes_per_element;
    io.out_bytes_per_element = out_bytes_per_element;

    io.sizeA   = (uint32_t) ((uint64_t) M * (uint64_t) K * (uint64_t) experts * (uint64_t) in_bytes_per_element);
    io.sizeC32 = (uint32_t) ((uint64_t) N * (uint64_t) experts * sizeof(float));
    io.sizeC   = (uint32_t) ((uint64_t) N * (uint64_t) experts * (uint64_t) out_bytes_per_element);

    io.size_codebook_nolut_tile = (uint32_t) q3xxs_vxm_nolut_codebook_bytes(K, N);
    io.size_scales_tile         = (uint32_t) ((uint64_t) (K / 128) * (uint64_t) N * sizeof(uint16_t));
    io.size_sign_tile           = (uint32_t) ((uint64_t) (K / 16) * (uint64_t) N * sizeof(uint16_t));
    io.size_super_scale_tile    = (uint32_t) ((uint64_t) (K / weights_group) * (uint64_t) N * sizeof(uint16_t));

    io.off_weights_scales  = io.size_codebook_nolut_tile;
    io.off_weights_sign    = io.off_weights_scales + io.size_scales_tile;
    io.off_weights_super   = io.off_weights_sign + io.size_sign_tile;
    io.size_weights_expert = io.off_weights_super + io.size_super_scale_tile;
    io.size_weights_total  = (uint32_t) ((uint64_t) io.size_weights_expert * (uint64_t) experts);

    io.sizeB_scale_scratch = (uint32_t) ((uint64_t) K * (uint64_t) N / (uint64_t) kQ3xxsQGroup * sizeof(uint16_t));
    io.sizeB_scale_scratch_total = (uint32_t) ((uint64_t) io.sizeB_scale_scratch * (uint64_t) experts);

    io.sram_base            = ctx.virtual_sram_base;
    io.sramA                = io.sram_base;
    io.sramC                = io.sramA + (RPPdeviceptr) round_up((int) io.sizeA);
    io.sramC1               = io.sramC + (RPPdeviceptr) round_up((int) io.sizeC32);
    io.sramB_codebook_nolut = io.sramC1 + (RPPdeviceptr) round_up((int) io.sizeC);
    io.sramB_scales         = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_scales;
    io.sramB_sign           = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_sign;
    io.sramB_super_scale    = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_super;
    io.sramB_qscale_lut     = io.sramB_codebook_nolut + (RPPdeviceptr) round_up((int) io.size_weights_total);
    io.sramB_mag_lut =
        io.sramB_qscale_lut + (RPPdeviceptr) round_up((int) q3xxs_vxm_nolut_lut_workspace::qscale_lut_bytes);
    io.sramB_mat_lut = io.sramB_mag_lut + (RPPdeviceptr) round_up((int) q3xxs_vxm_nolut_lut_workspace::mag_lut_bytes);
    io.sramB_scale   = io.sramB_mat_lut + (RPPdeviceptr) round_up((int) q3xxs_vxm_nolut_lut_workspace::mat_lut_bytes);
    io.total_sram_bytes =
        (uint32_t) ((io.sramB_scale + (RPPdeviceptr) round_up((int) io.sizeB_scale_scratch_total)) - io.sram_base);

    if (io.total_sram_bytes > kSramLimitBytes) {
        throw std::runtime_error("Matmul Q3XXS NoLUT SRAM layout exceeds the 22MB SRAM budget");
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
        ctx.dev_in.emplace_back(io.sramB_mat_lut);
        ctx.dev_in.emplace_back(io.sramB_scale);

        if (out_bytes_per_element == (int) sizeof(float)) {
            ctx.dev_out.emplace_back(io.sramC1);
            ctx.dev_out.emplace_back(io.sramC);
        } else {
            ctx.dev_out.emplace_back(io.sramC);
        }
    }
}

static inline void q3xxs_vxm_nolut_cdma_d2s_async(RPPdeviceptr sram_dst,
                                                  RPPdeviceptr ddr_src,
                                                  size_t       bytes,
                                                  RPPstream    stream) {
    rtMemcpyAsync((void *) sram_dst, (const void *) ddr_src, bytes, rtMemcpyDeviceToSram, stream);
}

static inline void q3xxs_vxm_nolut_cdma_s2d_async(RPPdeviceptr ddr_dst,
                                                  RPPdeviceptr sram_src,
                                                  size_t       bytes,
                                                  RPPstream    stream) {
    rtMemcpyAsync((void *) ddr_dst, (const void *) sram_src, bytes, rtMemcpySramToDevice, stream);
}

static inline void q3xxs_vxm_nolut_cdma_copy_expert_weights_to_sram(const q3xxs_vxm_nolut_sram_io & io,
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
    q3xxs_vxm_nolut_cdma_d2s_async(sram_dst, ddr_src, (size_t) io.size_weights_expert, stream);
}

static inline void q3xxs_vxm_nolut_cdma_copy_output_to_ddr(const q3xxs_vxm_nolut_sram_io & io,
                                                           RPPdeviceptr                    devC,
                                                           RPPstream                       stream) {
    const RPPdeviceptr sram_out = (io.out_bytes_per_element == (int) sizeof(float)) ? io.sramC1 : io.sramC;
    q3xxs_vxm_nolut_cdma_s2d_async(devC, sram_out, io.sizeC, stream);
}
}  // namespace kernel_q3_xxs_vxm_nolut
