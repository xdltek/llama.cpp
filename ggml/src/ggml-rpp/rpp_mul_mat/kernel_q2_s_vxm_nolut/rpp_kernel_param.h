
#pragma once
#include "ggml-rpp/rpp_kernel_utils.h"
#include "ggml-rpp/rpp_kernel_ctx.h"
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
#include <array>

namespace kernel_q2_s_vxm_nolut {
static void q2s_super_scale_params(uint32_t                in_scale,
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

static void matmul_weights_q2s_nolut_kernel_params(dim3 &                  blocksPerGrid,
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

static inline int q2s_vxm_nolut_round_up(int a) {
    return (a + 511) / 512 * 512 + 512;
}

struct q2s_vxm_nolut_tiles_info {
    int nr_of_tiles     = 0;
    int groups_per_tile = 0;
    int nr_of_ns        = 0;
    int Ns              = 0;
    int NsTail          = 0;
};

static inline q2s_vxm_nolut_tiles_info q2s_vxm_nolut_get_tiles_info(int N, int K, int weights_group) {
    q2s_vxm_nolut_tiles_info info{};

    if (weights_group == 32) {
        if (K % 512 == 0) {
            info.nr_of_tiles     = K / 512;
            info.groups_per_tile = 16;
        } else if (K % 256 == 0) {
            info.nr_of_tiles     = K / 256;
            info.groups_per_tile = 8;
        } else if (K % 128 == 0) {
            info.nr_of_tiles     = K / 128;
            info.groups_per_tile = 4;
        } else if (K % 64 == 0) {
            info.nr_of_tiles     = K / 64;
            info.groups_per_tile = 2;
        } else {
            info.nr_of_tiles     = K / 32;
            info.groups_per_tile = 1;
        }

        if (N >= 32768 * 4) {
            info.nr_of_tiles     = info.nr_of_tiles * 8;
            info.groups_per_tile = info.groups_per_tile / 8;
        } else if (N >= 32768 * 2) {
            info.nr_of_tiles     = info.nr_of_tiles * 4;
            info.groups_per_tile = info.groups_per_tile / 4;
        } else if (N >= 32768) {
            info.nr_of_tiles     = info.nr_of_tiles * 2;
            info.groups_per_tile = info.groups_per_tile / 2;
        }
    } else {
        info.nr_of_tiles     = K / weights_group;
        info.groups_per_tile = 1;
    }

    if (N * weights_group * info.groups_per_tile > (16 * 1024 * 1024)) {
        info.Ns       = N / 4;
        info.NsTail   = info.Ns;
        info.nr_of_ns = 4;
    } else if (N * weights_group * info.groups_per_tile > (8 * 1024 * 1024)) {
        info.Ns       = N / 2;
        info.NsTail   = info.Ns;
        info.nr_of_ns = 2;
    } else {
        info.Ns       = N;
        info.NsTail   = info.Ns;
        info.nr_of_ns = 1;
    }

    if (N == 151936) {
        info.Ns       = 600 * 32;
        info.nr_of_ns = 8;
        info.NsTail   = 548 * 32;
    } else if (N == 200064) {
        info.Ns       = 640 * 32;
        info.nr_of_ns = 10;
        info.NsTail   = 492 * 32;
    }

    const int NS_SAFE = 4096;
    if (info.Ns > NS_SAFE) {
        info.nr_of_ns = (N + NS_SAFE - 1) / NS_SAFE;
        info.Ns       = NS_SAFE;
        info.NsTail   = N - (info.nr_of_ns - 1) * info.Ns;
    }

    assert(K == info.nr_of_tiles * weights_group * info.groups_per_tile);
    assert(N == ((info.nr_of_ns - 1) * info.Ns + info.NsTail));
    assert(info.NsTail % 32 == 0);
    assert(info.Ns % 32 == 0);
    assert(info.Ns >= info.NsTail);

    return info;
}

static inline uint32_t q2s_vxm_nolut_estimate_moe_pipa_sram_bytes(int N,
                                                                  int weights_group,
                                                                  int groups_per_tile,
                                                                  int in_bytes_per_element,
                                                                  int out_bytes_per_element,
                                                                  int experts = 1) {
    if (experts <= 0) {
        throw std::runtime_error("q2s_vxm_nolut_estimate_moe_pipa_sram_bytes: experts must be > 0");
    }
    const int Ktile = groups_per_tile * weights_group;
    const int Ns    = N;

    const uint32_t sizeA   = (uint32_t) ((uint64_t) Ktile * (uint64_t) in_bytes_per_element);
    const uint32_t sizeC32 = (uint32_t) ((uint64_t) N * (uint64_t) experts * sizeof(float));
    const uint32_t sizeC   = (uint32_t) ((uint64_t) N * (uint64_t) experts * (uint64_t) out_bytes_per_element);

    const uint32_t size_codebook_tile = (uint32_t) ((uint64_t) Ktile * (uint64_t) Ns / 4ull);
    const uint32_t size_scales_tile   = (uint32_t) ((uint64_t) Ktile * (uint64_t) Ns / 32ull);
    const uint32_t size_sign_tile     = (uint32_t) ((uint64_t) Ktile * (uint64_t) Ns / 8ull);
    const uint32_t size_super_scale_tile =
        (uint32_t) ((uint64_t) (Ktile / weights_group) * (uint64_t) Ns * sizeof(uint16_t));
    const uint32_t size_expert_weights = (uint32_t) ((uint64_t) size_codebook_tile + (uint64_t) size_scales_tile +
                                                     (uint64_t) size_sign_tile + (uint64_t) size_super_scale_tile);
    const uint32_t size_weights_total  = (uint32_t) ((uint64_t) size_expert_weights * (uint64_t) experts);

    const uint32_t sizeB_scale_scratch =
        (uint32_t) ((uint64_t) groups_per_tile * (uint64_t) weights_group * (uint64_t) Ns / 16ull * sizeof(uint16_t));
    const uint32_t sizeB_scale_scratch_total = (uint32_t) ((uint64_t) sizeB_scale_scratch * (uint64_t) experts);
    const uint32_t sizeA_acc_scratch         = (uint32_t) q2s_vxm_nolut_round_up(weights_group * 4);

    constexpr uint32_t qscale_lut_bytes = 16u * (uint32_t) sizeof(uint16_t);
    constexpr uint32_t mag_lut_bytes    = 4u * (uint32_t) sizeof(uint16_t);

    uint64_t ping_end = 0;
    ping_end += (uint64_t) q2s_vxm_nolut_round_up((int) sizeA);
    ping_end += (uint64_t) q2s_vxm_nolut_round_up((int) sizeC32);
    ping_end += (uint64_t) q2s_vxm_nolut_round_up((int) sizeC);
    ping_end += (uint64_t) q2s_vxm_nolut_round_up((int) size_weights_total);
    ping_end += (uint64_t) q2s_vxm_nolut_round_up((int) qscale_lut_bytes);
    ping_end += (uint64_t) q2s_vxm_nolut_round_up((int) mag_lut_bytes);
    ping_end += (uint64_t) q2s_vxm_nolut_round_up((int) sizeB_scale_scratch_total);
    ping_end += (uint64_t) q2s_vxm_nolut_round_up((int) sizeA_acc_scratch);

    return (uint32_t) ping_end;
}

static inline q2s_vxm_nolut_tiles_info q2s_vxm_nolut_get_tiles_info_sram(
    int      N,
    int      K,
    int      weights_group,
    int      in_bytes_per_element  = (int) sizeof(uint16_t),
    int      out_bytes_per_element = (int) sizeof(uint16_t),
    int      experts               = 1,
    uint32_t sram_limit_bytes      = 22u * 1024u * 1024u) {
    q2s_vxm_nolut_tiles_info out{};
    if (weights_group <= 0 || K <= 0 || N <= 0) {
        throw std::runtime_error("q2s_vxm_nolut_get_tiles_info_sram: invalid shape/weights_group");
    }
    if (experts <= 0) {
        throw std::runtime_error("q2s_vxm_nolut_get_tiles_info_sram: experts must be > 0");
    }
    if (K % weights_group != 0) {
        throw std::runtime_error(
            "q2s_vxm_nolut_get_tiles_info_sram: K must be divisible by weights_group for 1-tile SRAM-direct");
    }

    const int total_groups = K / weights_group;
    out.nr_of_tiles        = 1;
    out.groups_per_tile    = total_groups;
    out.nr_of_ns           = 1;
    out.Ns                 = N;
    out.NsTail             = N;

    const uint32_t needed = q2s_vxm_nolut_estimate_moe_pipa_sram_bytes(
        N, weights_group, out.groups_per_tile, in_bytes_per_element, out_bytes_per_element, experts);
    if (needed > sram_limit_bytes) {
        throw std::runtime_error(
            "q2s_vxm_nolut_get_tiles_info_sram: 1-tile SRAM-direct requirement not met (SRAM overflow)");
    }

    assert(K == out.nr_of_tiles * weights_group * out.groups_per_tile);
    assert(N == ((out.nr_of_ns - 1) * out.Ns + out.NsTail));
    assert(out.NsTail % 32 == 0);
    assert(out.Ns % 32 == 0);
    assert(out.Ns >= out.NsTail);
    return out;
}

struct q2s_vxm_nolut_lut_workspace {
    static constexpr uint32_t qscale_lut_elems = 16;
    static constexpr uint32_t mag_lut_elems    = 4;
    static constexpr uint32_t qscale_lut_bytes = qscale_lut_elems * (uint32_t) sizeof(uint16_t);
    static constexpr uint32_t mag_lut_bytes    = mag_lut_elems * (uint32_t) sizeof(uint16_t);
    static constexpr uint32_t total_bytes      = qscale_lut_bytes + mag_lut_bytes;
};

struct q2s_vxm_nolut_sram_io {
    int M                     = 0;
    int K                     = 0;  // full K
    int Ktile                 = 0;  // one-tile K = groups_per_tile * weights_group
    int N                     = 0;
    int experts               = 1;
    int weights_group         = 0;
    int in_bytes_per_element  = 0;
    int out_bytes_per_element = 0;

    int nr_of_tiles     = 0;
    int groups_per_tile = 0;
    int nr_of_ns        = 1;
    int Ns              = 0;
    int NsTail          = 0;

    // One-expert chunk sizes for one full-K/full-N tile.
    uint32_t size_codebook_tile    = 0;
    uint32_t size_scales_tile      = 0;
    uint32_t size_sign_tile        = 0;
    uint32_t size_super_scale_tile = 0;
    uint32_t size_weights_expert   = 0;
    uint32_t size_weights_total    = 0;
    uint32_t off_weights_scales    = 0;
    uint32_t off_weights_sign      = 0;
    uint32_t off_weights_super     = 0;
    uint32_t sizeB_scale_scratch   = 0;

    uint32_t sizeA                      = 0;
    uint32_t sizeC32                    = 0;
    uint32_t sizeC                      = 0;
    uint32_t sizeB_codebook_nolut_total = 0;
    uint32_t sizeB_scales_total         = 0;
    uint32_t sizeB_sign_total           = 0;
    uint32_t sizeB_super_scale_total    = 0;
    uint32_t sizeB_scale_scratch_total  = 0;
    uint32_t sizeA_acc_scratch          = 0;
    uint32_t total_sram_bytes           = 0;

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

// Application-side helper: plan SRAM layout and bind ctx.dev_in/dev_out as SRAM addresses.
// dev_in mapping:
//   [0]=sramA, [1]=expert0 codebook, [2]=expert0 scales, [3]=expert0 sign, [4]=expert0 super_scale,
//      where all experts are packed as contiguous per-expert chunks:
//      [codebook|scales|sign|super_scale]expert0 [codebook|scales|sign|super_scale]expert1 ...
//   [5]=qscale_lut, [6]=mag_lut, [7]=scale_scratch, [8]=input_acc_scratch
// dev_out mapping:
//   BF16 output: [0]=sramC
//   FP32 output: [0]=sramC1 (final fp32), [1]=sramC (bf16 temp)
static inline q2s_vxm_nolut_sram_io q2s_vxm_nolut_prepare_sram_io(rpp_kernel_context & ctx,
                                                                  int                  M,
                                                                  int                  K,
                                                                  int                  N,
                                                                  int                  weights_group,
                                                                  int                  in_bytes_per_element,
                                                                  int                  out_bytes_per_element,
                                                                  int                  experts     = 1,
                                                                  bool                 bind_ctx_io = true) {
    q2s_vxm_nolut_sram_io io{};
    io.M                     = M;
    io.K                     = K;
    io.N                     = N;
    io.experts               = experts;
    io.weights_group         = weights_group;
    io.in_bytes_per_element  = in_bytes_per_element;
    io.out_bytes_per_element = out_bytes_per_element;

    const q2s_vxm_nolut_tiles_info tiles =
        q2s_vxm_nolut_get_tiles_info_sram(N, K, weights_group, in_bytes_per_element, out_bytes_per_element, experts);
    io.nr_of_tiles     = tiles.nr_of_tiles;
    io.groups_per_tile = tiles.groups_per_tile;
    io.nr_of_ns        = tiles.nr_of_ns;
    io.Ns              = tiles.Ns;
    io.NsTail          = tiles.NsTail;

    const int Ktile = io.groups_per_tile * weights_group;
    io.Ktile        = Ktile;

    io.sizeA   = (uint32_t) ((uint64_t) M * (uint64_t) K * (uint64_t) experts * (uint64_t) in_bytes_per_element);
    io.sizeC32 = (uint32_t) ((uint64_t) N * (uint64_t) experts * (uint64_t) sizeof(float));
    io.sizeC   = (uint32_t) ((uint64_t) N * (uint64_t) experts * (uint64_t) out_bytes_per_element);

    io.size_codebook_tile    = (uint32_t) (Ktile * io.Ns / 4);
    io.size_scales_tile      = (uint32_t) (Ktile * io.Ns / 32);
    io.size_sign_tile        = (uint32_t) (Ktile * io.Ns / 8);
    io.size_super_scale_tile = (uint32_t) ((Ktile / weights_group) * io.Ns * (int) sizeof(short));
    io.off_weights_scales    = io.size_codebook_tile;
    io.off_weights_sign      = io.off_weights_scales + io.size_scales_tile;
    io.off_weights_super     = io.off_weights_sign + io.size_sign_tile;
    io.size_weights_expert   = io.off_weights_super + io.size_super_scale_tile;
    io.size_weights_total    = (uint32_t) ((uint64_t) io.size_weights_expert * (uint64_t) experts);
    io.sizeB_scale_scratch   = (uint32_t) ((io.groups_per_tile * weights_group * io.Ns / 16) * (int) sizeof(short));

    io.sizeB_codebook_nolut_total = (uint32_t) ((uint64_t) io.size_codebook_tile * (uint64_t) experts);
    io.sizeB_scales_total         = (uint32_t) ((uint64_t) io.size_scales_tile * (uint64_t) experts);
    io.sizeB_sign_total           = (uint32_t) ((uint64_t) io.size_sign_tile * (uint64_t) experts);
    io.sizeB_super_scale_total    = (uint32_t) ((uint64_t) io.size_super_scale_tile * (uint64_t) experts);
    io.sizeB_scale_scratch_total  = (uint32_t) ((uint64_t) io.sizeB_scale_scratch * (uint64_t) experts);
    io.sizeA_acc_scratch          = (uint32_t) q2s_vxm_nolut_round_up(weights_group * 4);

    io.sram_base            = ctx.virtual_sram_base;
    io.sramA                = io.sram_base;
    io.sramC                = io.sramA + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.sizeA);
    io.sramC1               = io.sramC + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.sizeC32);
    io.sramB_codebook_nolut = io.sramC1 + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.sizeC);
    io.sramB_scales         = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_scales;
    io.sramB_sign           = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_sign;
    io.sramB_super_scale    = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_super;
    io.sramB_qscale_lut = io.sramB_codebook_nolut + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.size_weights_total);
    io.sramB_mag_lut    = io.sramB_qscale_lut +
                       (RPPdeviceptr) q2s_vxm_nolut_round_up((int) q2s_vxm_nolut_lut_workspace::qscale_lut_bytes);
    io.sramB_scale =
        io.sramB_mag_lut + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) q2s_vxm_nolut_lut_workspace::mag_lut_bytes);
    io.sramA_acc        = io.sramB_scale + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.sizeB_scale_scratch_total);
    io.total_sram_bytes = (uint32_t) (io.sramA_acc + (RPPdeviceptr) io.sizeA_acc_scratch - io.sram_base);

    const uint32_t SRAM_LIMIT = 22u * 1024u * 1024u;
    if (io.total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << io.total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                  << " bytes\n";
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
            ctx.dev_out.emplace_back(io.sramC1);  // final fp32
            ctx.dev_out.emplace_back(io.sramC);   // bf16 temp
        } else {
            ctx.dev_out.emplace_back(io.sramC);   // final bf16
        }
    }

    return io;
}

static inline void q2s_vxm_nolut_prepare_sram_io(rpp_kernel_context &    ctx,
                                                 q2s_vxm_nolut_sram_io & io,
                                                 int                     M,
                                                 int                     K,
                                                 int                     N,
                                                 int                     weights_group,
                                                 int                     in_bytes_per_element,
                                                 int                     out_bytes_per_element,
                                                 int                     experts     = 1,
                                                 bool                    bind_ctx_io = true) {
    io.M                     = M;
    io.K                     = K;
    io.N                     = N;
    io.experts               = experts;
    io.weights_group         = weights_group;
    io.in_bytes_per_element  = in_bytes_per_element;
    io.out_bytes_per_element = out_bytes_per_element;

    const q2s_vxm_nolut_tiles_info tiles =
        q2s_vxm_nolut_get_tiles_info_sram(N, K, weights_group, in_bytes_per_element, out_bytes_per_element, experts);
    io.nr_of_tiles     = tiles.nr_of_tiles;
    io.groups_per_tile = tiles.groups_per_tile;
    io.nr_of_ns        = tiles.nr_of_ns;
    io.Ns              = tiles.Ns;
    io.NsTail          = tiles.NsTail;

    const int Ktile = io.groups_per_tile * weights_group;
    io.Ktile        = Ktile;

    io.sizeA   = (uint32_t) ((uint64_t) M * (uint64_t) K * (uint64_t) experts * (uint64_t) in_bytes_per_element);
    io.sizeC32 = (uint32_t) ((uint64_t) N * (uint64_t) experts * (uint64_t) sizeof(float));
    io.sizeC   = (uint32_t) ((uint64_t) N * (uint64_t) experts * (uint64_t) out_bytes_per_element);

    io.size_codebook_tile    = (uint32_t) (Ktile * io.Ns / 4);
    io.size_scales_tile      = (uint32_t) (Ktile * io.Ns / 32);
    io.size_sign_tile        = (uint32_t) (Ktile * io.Ns / 8);
    io.size_super_scale_tile = (uint32_t) ((Ktile / weights_group) * io.Ns * (int) sizeof(short));
    io.off_weights_scales    = io.size_codebook_tile;
    io.off_weights_sign      = io.off_weights_scales + io.size_scales_tile;
    io.off_weights_super     = io.off_weights_sign + io.size_sign_tile;
    io.size_weights_expert   = io.off_weights_super + io.size_super_scale_tile;
    io.size_weights_total    = (uint32_t) ((uint64_t) io.size_weights_expert * (uint64_t) experts);
    io.sizeB_scale_scratch   = (uint32_t) ((io.groups_per_tile * weights_group * io.Ns / 16) * (int) sizeof(short));

    io.sizeB_codebook_nolut_total = (uint32_t) ((uint64_t) io.size_codebook_tile * (uint64_t) experts);
    io.sizeB_scales_total         = (uint32_t) ((uint64_t) io.size_scales_tile * (uint64_t) experts);
    io.sizeB_sign_total           = (uint32_t) ((uint64_t) io.size_sign_tile * (uint64_t) experts);
    io.sizeB_super_scale_total    = (uint32_t) ((uint64_t) io.size_super_scale_tile * (uint64_t) experts);
    io.sizeB_scale_scratch_total  = (uint32_t) ((uint64_t) io.sizeB_scale_scratch * (uint64_t) experts);
    io.sizeA_acc_scratch          = (uint32_t) q2s_vxm_nolut_round_up(weights_group * 4);

    io.sram_base            = ctx.virtual_sram_base;
    io.sramA                = io.sram_base;
    io.sramC                = io.sramA + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.sizeA);
    io.sramC1               = io.sramC + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.sizeC32);
    io.sramB_codebook_nolut = io.sramC1 + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.sizeC);
    io.sramB_scales         = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_scales;
    io.sramB_sign           = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_sign;
    io.sramB_super_scale    = io.sramB_codebook_nolut + (RPPdeviceptr) io.off_weights_super;
    io.sramB_qscale_lut = io.sramB_codebook_nolut + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.size_weights_total);
    io.sramB_mag_lut    = io.sramB_qscale_lut +
                       (RPPdeviceptr) q2s_vxm_nolut_round_up((int) q2s_vxm_nolut_lut_workspace::qscale_lut_bytes);
    io.sramB_scale =
        io.sramB_mag_lut + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) q2s_vxm_nolut_lut_workspace::mag_lut_bytes);
    io.sramA_acc        = io.sramB_scale + (RPPdeviceptr) q2s_vxm_nolut_round_up((int) io.sizeB_scale_scratch_total);
    io.total_sram_bytes = (uint32_t) (io.sramA_acc + (RPPdeviceptr) io.sizeA_acc_scratch - io.sram_base);

    const uint32_t SRAM_LIMIT = 22u * 1024u * 1024u;
    if (io.total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << io.total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                  << " bytes\n";
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
            ctx.dev_out.emplace_back(io.sramC1);  // final fp32
            ctx.dev_out.emplace_back(io.sramC);   // bf16 temp
        } else {
            ctx.dev_out.emplace_back(io.sramC);   // final bf16
        }
    }
}

// Host-side helper for full tensor tiling metadata.
static inline q2s_vxm_nolut_tiles_info q2s_vxm_nolut_get_full_tiles_info(
    int      N,
    int      K,
    int      weights_group,
    int      in_bytes_per_element  = (int) sizeof(uint16_t),
    int      out_bytes_per_element = (int) sizeof(uint16_t),
    uint32_t sram_limit_bytes      = 22u * 1024u * 1024u) {
    return q2s_vxm_nolut_get_tiles_info_sram(N, K, weights_group, in_bytes_per_element, out_bytes_per_element, 1,
                                             sram_limit_bytes);
}

// ----------------------------
// CDMA helper functions (DDR <-> SRAM)
// ----------------------------
static inline void q2s_vxm_nolut_cdma_d2s_async(RPPdeviceptr sram_dst,
                                                RPPdeviceptr ddr_src,
                                                size_t       bytes,
                                                RPPstream    stream) {
    rtMemcpyAsync((void *) sram_dst, (const void *) ddr_src, bytes, rtMemcpyDeviceToSram, stream);
}

static inline void q2s_vxm_nolut_cdma_s2d_async(RPPdeviceptr ddr_dst,
                                                RPPdeviceptr sram_src,
                                                size_t       bytes,
                                                RPPstream    stream) {
    rtMemcpyAsync((void *) ddr_dst, (const void *) sram_src, bytes, rtMemcpySramToDevice, stream);
}

// Copy one expert packed weights chunk [codebook|scales|sign|super_scale] from DDR to SRAM.
static inline void q2s_vxm_nolut_cdma_copy_expert_weights_to_sram(const q2s_vxm_nolut_sram_io & io,
                                                                  RPPdeviceptr                  dev_expert_weights_base,
                                                                  int                           sram_idx,
                                                                  int                           expert_idx,
                                                                  RPPstream                     stream) {
    if (sram_idx < 0 || sram_idx >= io.experts) {
        std::cerr << "q2s_vxm_nolut_cdma_copy_expert_weights_to_sram expert_idx out of range\n";
        std::abort();
    }
    const RPPdeviceptr sram_dst =
        io.sramB_codebook_nolut + (RPPdeviceptr) sram_idx * (RPPdeviceptr) io.size_weights_expert;
    const RPPdeviceptr ddr_src =
        dev_expert_weights_base + (RPPdeviceptr) expert_idx * (RPPdeviceptr) io.size_weights_expert;
    q2s_vxm_nolut_cdma_d2s_async(sram_dst, ddr_src, (size_t) io.size_weights_expert, stream);
}

// Copy packed A for all experts from DDR to SRAM.
// Layout is contiguous per expert: [A_expert0][A_expert1]...
static inline void q2s_vxm_nolut_cdma_copy_a_full_to_sram(const q2s_vxm_nolut_sram_io & io,
                                                          RPPdeviceptr                  devA,
                                                          RPPstream                     stream) {
    if (io.M != 1) {
        std::cerr << "q2s_vxm_nolut_cdma_copy_a_full_to_sram supports M==1 only\n";
        std::abort();
    }
    q2s_vxm_nolut_cdma_d2s_async(io.sramA, devA, (size_t) io.sizeA, stream);
}

// Copy one A tile [M, tile_k] from DDR to SRAM.
// This helper is intended for VxM path (M==1).
static inline void q2s_vxm_nolut_cdma_copy_a_tile_to_sram(const q2s_vxm_nolut_sram_io & io,
                                                          RPPdeviceptr                  devA,
                                                          int                           tile_idx,
                                                          RPPstream                     stream,
                                                          int                           K_full = 0) {
    if (io.M != 1) {
        std::cerr << "q2s_vxm_nolut_cdma_copy_a_tile_to_sram supports M==1 only\n";
        std::abort();
    }
    const int K_total = (K_full > 0) ? K_full : io.K;
    if ((tile_idx + 1) * io.Ktile > K_total) {
        std::cerr << "q2s_vxm_nolut_cdma_copy_a_tile_to_sram tile out of range, tile_idx=" << tile_idx
                  << ", tile_k=" << io.Ktile << ", K_full=" << K_total << "\n";
        std::abort();
    }
    const size_t       tile_bytes = (size_t) io.Ktile * (size_t) io.in_bytes_per_element;
    const RPPdeviceptr src_tile   = devA + (RPPdeviceptr) tile_idx * (RPPdeviceptr) tile_bytes;
    q2s_vxm_nolut_cdma_d2s_async(io.sramA, src_tile, tile_bytes, stream);
}

// Copy one IQ tile from DDR to SRAM compact layout expected by sram-direct graph.
static inline void q2s_vxm_nolut_cdma_copy_iq_tile_to_sram(const q2s_vxm_nolut_sram_io & io,
                                                           RPPdeviceptr                  devB_codebook_nolut,
                                                           RPPdeviceptr                  devB_scales,
                                                           RPPdeviceptr                  devB_sign,
                                                           RPPdeviceptr                  devB_super_scale,
                                                           int                           tile_idx,
                                                           RPPstream                     stream) {
    const int Ktile              = io.Ktile;
    const int codebook_stride    = (Ktile / 8) * io.N * (int) sizeof(short);
    const int scales_stride      = (Ktile / 64) * io.N * (int) sizeof(short);
    const int sign_stride        = (Ktile / 16) * io.N * (int) sizeof(short);
    const int super_scale_stride = (Ktile / io.weights_group) * io.N * (int) sizeof(short);

    const RPPdeviceptr ddr_codebook_tile = devB_codebook_nolut + (RPPdeviceptr) tile_idx * codebook_stride;
    const RPPdeviceptr ddr_scales_tile   = devB_scales + (RPPdeviceptr) tile_idx * scales_stride;
    const RPPdeviceptr ddr_sign_tile     = devB_sign + (RPPdeviceptr) tile_idx * sign_stride;
    const RPPdeviceptr ddr_super_tile    = devB_super_scale + (RPPdeviceptr) tile_idx * super_scale_stride;

    q2s_vxm_nolut_cdma_d2s_async(io.sramB_codebook_nolut, ddr_codebook_tile, io.size_codebook_tile, stream);
    q2s_vxm_nolut_cdma_d2s_async(io.sramB_scales, ddr_scales_tile, io.size_scales_tile, stream);
    q2s_vxm_nolut_cdma_d2s_async(io.sramB_sign, ddr_sign_tile, io.size_sign_tile, stream);
    q2s_vxm_nolut_cdma_d2s_async(io.sramB_super_scale, ddr_super_tile, io.size_super_scale_tile, stream);
}

static inline void q2s_vxm_nolut_cdma_copy_output_to_ddr(const q2s_vxm_nolut_sram_io & io,
                                                         RPPdeviceptr                  devC,
                                                         RPPstream                     stream) {
    const RPPdeviceptr sram_out = (io.out_bytes_per_element == (int) sizeof(float)) ? io.sramC1 : io.sramC;
    q2s_vxm_nolut_cdma_s2d_async(devC, sram_out, io.sizeC, stream);
}

// ----------------------------
// LUT helper functions
// ----------------------------
static inline RPPdeviceptr q2s_vxm_nolut_prepare_lut_workspace(rpp_kernel_context & ctx) {
    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        rtMalloc((void **) &dev_lut_workspace, q2s_vxm_nolut_lut_workspace::total_bytes);
        ctx.dev_workspace = dev_lut_workspace;
    }

    std::array<uint16_t, q2s_vxm_nolut_lut_workspace::qscale_lut_elems> qscale_lut = {};
    for (uint32_t i = 0; i < q2s_vxm_nolut_lut_workspace::qscale_lut_elems; ++i) {
        const float scale4  = (float) i;
        const float lut_val = (0.5f + scale4) * 0.25f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }

    constexpr std::array<float, q2s_vxm_nolut_lut_workspace::mag_lut_elems> mag_lut_values = { 8.0f, 25.0f, 43.0f,
                                                                                               0.0f };
    std::array<uint16_t, q2s_vxm_nolut_lut_workspace::mag_lut_elems>        mag_lut        = {};
    for (uint32_t i = 0; i < q2s_vxm_nolut_lut_workspace::mag_lut_elems; ++i) {
        mag_lut[i] = float_to_bf16_rne(mag_lut_values[i]);
    }

    rtMemcpy((void *) dev_lut_workspace, qscale_lut.data(), q2s_vxm_nolut_lut_workspace::qscale_lut_bytes,
             rtMemcpyHostToDevice);
    rtMemcpy((void *) (dev_lut_workspace + q2s_vxm_nolut_lut_workspace::qscale_lut_bytes), mag_lut.data(),
             q2s_vxm_nolut_lut_workspace::mag_lut_bytes, rtMemcpyHostToDevice);

    return dev_lut_workspace;
}

static inline void q2s_vxm_nolut_copy_lut_workspace_to_sram(const q2s_vxm_nolut_sram_io & io,
                                                            RPPdeviceptr                  dev_lut_workspace,
                                                            RPPstream                     stream) {
    q2s_vxm_nolut_cdma_d2s_async(io.sramB_qscale_lut, dev_lut_workspace, q2s_vxm_nolut_lut_workspace::qscale_lut_bytes,
                                 stream);
    q2s_vxm_nolut_cdma_d2s_async(io.sramB_mag_lut, dev_lut_workspace + q2s_vxm_nolut_lut_workspace::qscale_lut_bytes,
                                 q2s_vxm_nolut_lut_workspace::mag_lut_bytes, stream);
}

static void matmul_weights_q2s_nolut_batch_params(dim3 &                  blocksPerGrid,
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
                                                  std::vector<uint32_t> & params) {
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
    uint32_t in_wq_stridez        = in_wq_expert_stride_bytes / 2;
    uint32_t in_sign_stridez      = in_sign_expert_stride_bytes / 2;
    uint32_t in_scale_stridez     = in_scale_expert_stride_bytes / 2;
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
    params.emplace_back(in_wq_stridez);
    params.emplace_back(in_sign_stridez);
    params.emplace_back(in_scale_stridez);
    params.emplace_back(in_act_expert_stride_bytes);
    params.emplace_back(loop);
    params.emplace_back(0);
}
}  // namespace kernel_q2_s_vxm_nolut
