#pragma once
// rpp_matmul_q4k.cpp
// q4k backend: q6k-style scale/min expansion + q4 dequant/matmul kernels.

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q4_k/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q4_k/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q4_k {

inline int get_tn(int N) {
    if ((N % 128) == 0) {
        return 4;
    } else if ((N % 96) == 0) {
        return 3;
    } else if ((N % 64) == 0) {
        return 2;
    } else {
        return 1;
    }
}

inline int get_tiles(int N) {
    return (N + MATMUL_NS - 1) / MATMUL_NS;
}

inline int get_ns(int N, bool is_tail) {
    if (is_tail) {
        return N - (get_tiles(N) - 1) * MATMUL_NS;
    } else {
        return MATMUL_NS;
    }
}

inline std::string get_matmul_kernel(int N) {
    if ((N % 128) == 0) {
        return "matmul_tn4_f16_f32_f16";
    } else if ((N % 96) == 0) {
        return "matmul_tn3_f16_f32_f16";
    } else if ((N % 64) == 0) {
        return "matmul_tn2_f16_f32_f16";
    } else {
        return "matmul_tn1_f16_f32_f16";
    }
}

static void rpp_matmul_q4k(rpp_kernel_context & ctx,
                           int                  M,
                           int                  K,
                           int                  N,
                           int                  weights_group,
                           int                  in_bytes_per_element,
                           int                  out_bytes_per_element,
                           int                  is_instantial);

inline int choose_m_tile(int M, int K, int in_bytes_per_element, bool use_pipeline) {
    const int q_group     = 32;
    const int super_group = 256;
    const int SRAM_LIMIT  = 22 * 1024 * 1024;

    const int sizeB_q4         = (K * MATMUL_NS / 2);
    const int sizeB_qscale_lsb = (K * MATMUL_NS / 2 / q_group);
    const int sizeB_qscale_msb = sizeB_qscale_lsb / 2;
    const int sizeB_qzero_lsb  = sizeB_qscale_lsb;
    const int sizeB_qzero_msb  = sizeB_qscale_msb;
    const int sizeB_super      = (K * MATMUL_NS / super_group) * (int) sizeof(short);
    const int sizeB_scale      = (K * MATMUL_NS / q_group) * (int) sizeof(short);
    const int sizeB            = (K * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int b_ping_buffers   = use_pipeline ? 2 : 1;

    auto calc_total_sram_bytes = [&](int m_rows) -> int64_t {
        const int sizeA0  = m_rows * K * in_bytes_per_element;
        const int sizeA1  = m_rows * K * (int) sizeof(rpp::bfloat16);
        const int sizeCr  = m_rows * MATMUL_NS * (int) sizeof(rpp::bfloat16);
        const int sizeC32 = m_rows * MATMUL_NS * (int) sizeof(float);

        int64_t total = 0;
        total += round_up(sizeA0);
        total += round_up(sizeA1);
        total += b_ping_buffers * (round_up(sizeB_q4) + round_up(sizeB_qscale_lsb) + round_up(sizeB_qzero_lsb) +
                                   round_up(sizeB_qscale_msb) + round_up(sizeB_qzero_msb) + round_up(sizeB_super) +
                                   round_up(sizeB_super));
        total += round_up(sizeB_scale);
        total += round_up(sizeB_scale);
        total += round_up(sizeB);
        total += round_up(sizeCr);
        total += round_up(sizeCr);
        total += round_up(sizeC32);
        return total;
    };

    int Mtile = M;
    if (calc_total_sram_bytes(Mtile) > SRAM_LIMIT) {
        int lo = 1;
        int hi = M;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (calc_total_sram_bytes(mid) <= SRAM_LIMIT) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Mtile = lo;
        if (Mtile > 32) {
            Mtile = (Mtile / 32) * 32;
        }
        if (Mtile <= 0) {
            Mtile = lo;
        }
    }

    return Mtile;
}

static inline RPPdeviceptr stage_a_tile(rpp_kernel_context &    ctx,
                                        RPPdeviceptr            devA_base,
                                        int                     m_start,
                                        int                     m_rows,
                                        int                     K,
                                        int                     in_bytes_per_element,
                                        RPPdeviceptr            sramA_in,
                                        RPPdeviceptr            sramA_out,
                                        dim3 &                  threadsPerBlock,
                                        dim3 &                  blocksPerGrid,
                                        std::vector<uint32_t> & params) {
    rtMemcpyAsync((void *) sramA_in, (const void *) (devA_base + (RPPdeviceptr) m_start * K * in_bytes_per_element),
                  m_rows * K * in_bytes_per_element, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(m_rows, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA_out, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        chw2chw32_blocks(1, m_rows, K, threadsPerBlock, blocksPerGrid);
        chw2chw32_align_params((int) sramA_out, (int) sramA_in, m_rows, K, 0, (int) threadsPerBlock.x,
                               (int) threadsPerBlock.y, (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params,
                               false);
        launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                           ctx.rppBinMod, ctx.kernelStream);

        return sramA_in;
    }

    params.clear();
    chw2chw32_blocks(1, m_rows, K, threadsPerBlock, blocksPerGrid);
    chw2chw32_align_params((int) sramA_in, (int) sramA_out, m_rows, K, 0, (int) threadsPerBlock.x,
                           (int) threadsPerBlock.y, (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params,
                           false);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    return sramA_out;
}

static void rpp_matmul_q4k_tiled(rpp_kernel_context & ctx,
                                 int                  M,
                                 int                  K,
                                 int                  N,
                                 int                  weights_group,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  is_instantial = 1) {
    const int q_group     = 32;
    const int super_group = 256;
    const int group       = super_group / q_group;
    if (K % super_group != 0) {
        throw std::runtime_error("Q4K requires K % 256 == 0");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA_base            = ctx.dev_in[0];
    RPPdeviceptr devB_q4_base         = ctx.dev_in[1];
    RPPdeviceptr devB_qscale_lsb_base = ctx.dev_in[2];
    RPPdeviceptr devB_qzero_lsb_base  = ctx.dev_in[3];
    RPPdeviceptr devB_qscale_msb_base = ctx.dev_in[4];
    RPPdeviceptr devB_qzero_msb_base  = ctx.dev_in[5];
    RPPdeviceptr devB_super_scale_base = ctx.dev_in[6];
    RPPdeviceptr devB_super_zero_base  = ctx.dev_in[7];
    RPPdeviceptr devC_base             = ctx.dev_out[0];

    const int Mtile = choose_m_tile(M, K, in_bytes_per_element, false);
    if (Mtile >= M) {
        rpp_matmul_q4k(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
        return;
    }

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q4k.o");

    const int sizeA0 = Mtile * K * in_bytes_per_element;
    const int sizeA1 = Mtile * K * (int) sizeof(rpp::bfloat16);

    const int sizeB_q4         = (K * MATMUL_NS / 2);
    const int sizeB_qscale_lsb = (K * MATMUL_NS / 2 / q_group);
    const int sizeB_qscale_msb = sizeB_qscale_lsb / 2;
    const int sizeB_qzero_lsb  = sizeB_qscale_lsb;
    const int sizeB_qzero_msb  = sizeB_qscale_msb;

    const int sizeB_super_scale = (K * MATMUL_NS / super_group) * (int) sizeof(short);
    const int sizeB_super_zero  = sizeB_super_scale;

    const int sizeB_scale = (K * MATMUL_NS / q_group) * (int) sizeof(short);
    const int sizeB_zero  = sizeB_scale;

    const int sizeB   = (K * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeCr  = (Mtile * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeC32 = (Mtile * MATMUL_NS) * (int) sizeof(float);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);

    RPPdeviceptr sramB_q4         = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_qscale_lsb = sramB_q4 + round_up(sizeB_q4);
    RPPdeviceptr sramB_qzero_lsb  = sramB_qscale_lsb + round_up(sizeB_qscale_lsb);

    RPPdeviceptr sramB_qscale_msb = sramB_qzero_lsb + round_up(sizeB_qzero_lsb);
    RPPdeviceptr sramB_qzero_msb  = sramB_qscale_msb + round_up(sizeB_qscale_msb);

    RPPdeviceptr sramB_super_scale = sramB_qzero_msb + round_up(sizeB_qzero_msb);
    RPPdeviceptr sramB_super_zero  = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_scale       = sramB_super_zero + round_up(sizeB_super_zero);
    RPPdeviceptr sramB_zero        = sramB_scale + round_up(sizeB_scale);

    RPPdeviceptr sramB_out      = sramB_zero + round_up(sizeB_zero);
    RPPdeviceptr sramC_hw32     = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw      = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32 = sramC_chw + round_up(sizeCr);

    const int total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    const int m_tiles = (M + Mtile - 1) / Mtile;
    for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
        const int          m_start  = m_tile * Mtile;
        const int          m_rows   = std::min(Mtile, M - m_start);
        const RPPdeviceptr matmul_A = stage_a_tile(ctx, devA_base, m_start, m_rows, K, in_bytes_per_element, sramA_in,
                                                   sramA_out, threadsPerBlock, blocksPerGrid, params);

        for (int i = 0; i < get_tiles(N); ++i) {
            const bool is_tail = (i == (get_tiles(N) - 1));
            const int  ns      = get_ns(N, is_tail);
            const int  n_start = i * MATMUL_NS;

            rtMemcpy2DAsync((void *) sramB_q4, ns * (int) sizeof(short),
                            (const void *) (devB_q4_base + (RPPdeviceptr) n_start * sizeof(short)),
                            N * (int) sizeof(short), ns * (int) sizeof(short), K / 4, rtMemcpyDeviceToSram,
                            ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_qscale_lsb, ns * (int) sizeof(short),
                            (const void *) (devB_qscale_lsb_base + (RPPdeviceptr) n_start * sizeof(short)),
                            N * (int) sizeof(short), ns * (int) sizeof(short), K / q_group / 4, rtMemcpyDeviceToSram,
                            ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_qzero_lsb, ns * (int) sizeof(short),
                            (const void *) (devB_qzero_lsb_base + (RPPdeviceptr) n_start * sizeof(short)),
                            N * (int) sizeof(short), ns * (int) sizeof(short), K / q_group / 4, rtMemcpyDeviceToSram,
                            ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_qscale_msb, ns * (int) sizeof(short),
                            (const void *) (devB_qscale_msb_base + (RPPdeviceptr) n_start * sizeof(short)),
                            N * (int) sizeof(short), ns * (int) sizeof(short), K / q_group / 8, rtMemcpyDeviceToSram,
                            ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_qzero_msb, ns * (int) sizeof(short),
                            (const void *) (devB_qzero_msb_base + (RPPdeviceptr) n_start * sizeof(short)),
                            N * (int) sizeof(short), ns * (int) sizeof(short), K / q_group / 8, rtMemcpyDeviceToSram,
                            ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_super_scale, ns * (int) sizeof(short),
                            (const void *) (devB_super_scale_base + (RPPdeviceptr) n_start * sizeof(short)),
                            N * (int) sizeof(short), ns * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                            ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_super_zero, ns * (int) sizeof(short),
                            (const void *) (devB_super_zero_base + (RPPdeviceptr) n_start * sizeof(short)),
                            N * (int) sizeof(short), ns * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                            ctx.kernelStream);

            params.clear();
            q4k_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
            q4k_super_scale_params(sramB_qscale_lsb, sramB_qscale_msb, sramB_super_scale, sramB_scale, K, ns,
                                   super_group, q_group, 0, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            params.clear();
            q4k_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
            q4k_super_scale_params(sramB_qzero_lsb, sramB_qzero_msb, sramB_super_zero, sramB_zero, K, ns, super_group,
                                   q_group, 1, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            params.clear();
            q4k_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
            q4k_dequant_params(blocksPerGrid, threadsPerBlock, sramB_q4, sramB_scale, sramB_zero, sramB_out, 0, ns, K,
                               (int) sizeof(short), (int) sizeof(short), params);
            launchWrapperAysnc("matrix_mul_q4k_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            params.clear();
            const int tn = get_tn(ns);
            matmul_chw32_blocks(1, m_rows, ns, threadsPerBlock, blocksPerGrid, tn, 0);
            matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) sramB_out, 0,
                                     (uint32_t) sramC_hw32, m_rows, K, K, ns, 1, 1, 1, 0, tn,
                                     (int) sizeof(rpp::bfloat16), (int) sizeof(rpp::bfloat16), params, false, false);
            launchWrapperAysnc(get_matmul_kernel(ns), blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            params.clear();
            chw322chw_blocks(1, m_rows, ns, threadsPerBlock, blocksPerGrid);
            chw322chw_align_params((int) sramC_hw32, (int) sramC_chw, m_rows, ns, 0, threadsPerBlock.x,
                                   threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
            launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                               ctx.rppBinMod, ctx.kernelStream);

            if (out_bytes_per_element == (int) sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(m_rows, ns * 2, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);

                rtMemcpy2DAsync((void *) (devC_base + ((RPPdeviceptr) m_start * N + n_start) * (int) sizeof(float)),
                                N * (int) sizeof(float), (void *) sramC_chw_fp32, ns * (int) sizeof(float),
                                ns * (int) sizeof(float), m_rows, rtMemcpySramToDevice, ctx.kernelStream);
            } else {
                rtMemcpy2DAsync((void *) (devC_base + ((RPPdeviceptr) m_start * N + n_start) * (int) sizeof(short)),
                                N * (int) sizeof(short), (void *) sramC_chw, ns * (int) sizeof(short),
                                ns * (int) sizeof(short), m_rows, rtMemcpySramToDevice, ctx.kernelStream);
            }
        }
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

// ctx.dev_in:
//  [0] A
//  [1] weights_q4 (uint16, [K/4][N])
//  [2] qscale_i8_packed (uint16, [K/64][N])
//  [3] qmin_i8_packed   (uint16, [K/64][N])
//  [4] super_scale (bf16, [K/256][N])
//  [5] super_zero  (bf16, [K/256][N])
static void rpp_matmul_q4k(rpp_kernel_context & ctx,
                           int                  M,
                           int                  K,
                           int                  N,
                           int                  weights_group,
                           int                  in_bytes_per_element,
                           int                  out_bytes_per_element,
                           int                  is_instantial = 1) {
    const int q_group     = 32;
    const int super_group = 256;
    const int group       = super_group / q_group;
    if (K % super_group != 0) {
        throw std::runtime_error("Q4K requires K % 256 == 0");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA             = ctx.dev_in[0];
    RPPdeviceptr devB_q4          = ctx.dev_in[1];
    RPPdeviceptr devB_qscale_lsb  = ctx.dev_in[2];
    RPPdeviceptr devB_qzero_lsb   = ctx.dev_in[3];
    RPPdeviceptr devB_qscale_msb  = ctx.dev_in[4];
    RPPdeviceptr devB_qzero_msb   = ctx.dev_in[5];
    RPPdeviceptr devB_super_scale = ctx.dev_in[6];
    RPPdeviceptr devB_super_zero  = ctx.dev_in[7];
    RPPdeviceptr devC             = ctx.dev_out[0];

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q4k.o");

    const int sizeA0 = M * K * in_bytes_per_element;
    const int sizeA1 = M * K * (int) sizeof(rpp::bfloat16);
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // q_group = 32
    // wq_per_group = 32
    // in_wq     [K/256]      |  [8]  |  [32/4]    |  [4][N]
    // in_wq     [z]          |  [y]  |  [unroll]  |  [x]
    //           [grid.y]*[z] |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------

    const int sizeB_q4         = (K * MATMUL_NS / 2);
    const int sizeB_qscale_lsb = (K * MATMUL_NS / 2 / q_group);
    const int sizeB_qscale_msb = sizeB_qscale_lsb / 2;
    const int sizeB_qzero_lsb  = sizeB_qscale_lsb;
    const int sizeB_qzero_msb  = sizeB_qscale_msb;

    const int sizeB_super_scale = (K * MATMUL_NS / super_group) * (int) sizeof(short);
    const int sizeB_super_zero  = sizeB_super_scale;

    const int sizeB_scale = (K * MATMUL_NS / q_group) * (int) sizeof(short);
    const int sizeB_zero  = sizeB_scale;

    const int sizeB   = (K * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeCr  = (M * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeC32 = (M * MATMUL_NS) * (int) sizeof(float);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);

    RPPdeviceptr sramB_q4         = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_qscale_lsb = sramB_q4 + round_up(sizeB_q4);
    RPPdeviceptr sramB_qzero_lsb  = sramB_qscale_lsb + round_up(sizeB_qscale_lsb);

    RPPdeviceptr sramB_qscale_msb = sramB_qzero_lsb + round_up(sizeB_qzero_lsb);
    RPPdeviceptr sramB_qzero_msb  = sramB_qscale_msb + round_up(sizeB_qscale_msb);

    RPPdeviceptr sramB_super_scale = sramB_qzero_msb + round_up(sizeB_qzero_msb);
    RPPdeviceptr sramB_super_zero  = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_scale       = sramB_super_zero + round_up(sizeB_super_zero);
    RPPdeviceptr sramB_zero        = sramB_scale + round_up(sizeB_scale);

    RPPdeviceptr sramB_out      = sramB_zero + round_up(sizeB_zero);
    RPPdeviceptr sramC_hw32     = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw      = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32 = sramC_chw + round_up(sizeCr);

    const int total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    RPPdeviceptr matmul_A = sramA_out;
    RPPdeviceptr matmul_B = sramB_out;
    RPPdeviceptr matmul_C = sramC_hw32;

    rtMemcpyAsync((void *) sramA_in, (const void *) devA, sizeA0, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, M * K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA_in, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    params.clear();
    chw2chw32_blocks(1, M, K, threadsPerBlock, blocksPerGrid);
    chw2chw32_align_params((int) sramA_in, (int) sramA_out, M, K, 0, (int) threadsPerBlock.x, (int) threadsPerBlock.y,
                           (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params, false);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    for (int i = 0; i < get_tiles(N); i++) {
        bool is_tail = (i == (get_tiles(N) - 1));
        int  ns      = get_ns(N, is_tail);

        rtMemcpy2DAsync((void *) sramB_q4, ns * (int) sizeof(short), (const void *) devB_q4, N * (int) sizeof(short),
                        ns * (int) sizeof(short), K / 4, rtMemcpyDeviceToSram, ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_qscale_lsb, ns * (int) sizeof(short), (const void *) devB_qscale_lsb,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / q_group / 4, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_qzero_lsb, ns * (int) sizeof(short), (const void *) devB_qzero_lsb,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / q_group / 4, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_qscale_msb, ns * (int) sizeof(short), (const void *) devB_qscale_msb,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / q_group / 8, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_qzero_msb, ns * (int) sizeof(short), (const void *) devB_qzero_msb,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / q_group / 8, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_super_scale, ns * (int) sizeof(short), (const void *) devB_super_scale,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_super_zero, ns * (int) sizeof(short), (const void *) devB_super_zero,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        // qscale_i6 * super_scale -> bf16 scales
        params.clear();
        q4k_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
        q4k_super_scale_params(sramB_qscale_lsb, sramB_qscale_msb, sramB_super_scale, sramB_scale, K, ns, super_group,
                               q_group, 0, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        // qmin_i6 * super_zero -> bf16 mins
        // qmin can reuse same kernel of qscale
        params.clear();
        q4k_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
        q4k_super_scale_params(sramB_qzero_lsb, sramB_qzero_msb, sramB_super_zero, sramB_zero, K, ns, super_group,
                               q_group, 1, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        //jsq4k
        params.clear();
        q4k_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
        q4k_dequant_params(blocksPerGrid, threadsPerBlock, sramB_q4, sramB_scale, sramB_zero, sramB_out, 0, ns, K,
                           (int) sizeof(short), (int) sizeof(short), params);
        launchWrapperAysnc("matrix_mul_q4k_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        matmul_chw32_blocks(1, M, ns, threadsPerBlock, blocksPerGrid, get_tn(ns), 0);
        matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                                 (uint32_t) matmul_C, M, K, K, ns, 1, 1, 1, 0, get_tn(ns), (int) sizeof(rpp::bfloat16),
                                 (int) sizeof(rpp::bfloat16), params, false, false);
        launchWrapperAysnc(get_matmul_kernel(ns), blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        chw322chw_blocks(1, M, ns, threadsPerBlock, blocksPerGrid);
        chw322chw_align_params((int) matmul_C, (int) sramC_chw, M, ns, 0, threadsPerBlock.x, threadsPerBlock.y,
                               threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
        launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                           ctx.rppBinMod, ctx.kernelStream);

        if (out_bytes_per_element == (int) sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, M * ns * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpy2DAsync((void *) devC, N * (int) sizeof(float), (void *) sramC_chw_fp32, ns * (int) sizeof(float),
                            ns * (int) sizeof(float), M, rtMemcpySramToDevice, ctx.kernelStream);

            devC += ns * (int) sizeof(float);
        } else {
            rtMemcpy2DAsync((void *) devC, N * (int) sizeof(short), (void *) sramC_chw, ns * (int) sizeof(short),
                            ns * (int) sizeof(short), M, rtMemcpySramToDevice, ctx.kernelStream);
            devC += ns * (int) sizeof(short);
        }

        devB_q4 += ns * (int) sizeof(short);
        devB_qscale_lsb += ns * (int) sizeof(short);
        devB_qzero_lsb += ns * (int) sizeof(short);
        devB_qscale_msb += ns * (int) sizeof(short);
        devB_qzero_msb += ns * (int) sizeof(short);
        devB_super_scale += ns * (int) sizeof(short);
        devB_super_zero += ns * (int) sizeof(short);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q4k_pipeline(rpp_kernel_context & ctx,
                                    int                  M,
                                    int                  K,
                                    int                  N,
                                    int                  weights_group,
                                    int                  in_bytes_per_element,
                                    int                  out_bytes_per_element,
                                    int                  is_instantial = 1) {
    const int q_group     = 32;
    const int super_group = 256;
    const int group       = super_group / q_group;
    if (K % super_group != 0) {
        throw std::runtime_error("Q4K requires K % 256 == 0");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA             = ctx.dev_in[0];
    RPPdeviceptr devB_q4          = ctx.dev_in[1];
    RPPdeviceptr devB_qscale_lsb  = ctx.dev_in[2];
    RPPdeviceptr devB_qzero_lsb   = ctx.dev_in[3];
    RPPdeviceptr devB_qscale_msb  = ctx.dev_in[4];
    RPPdeviceptr devB_qzero_msb   = ctx.dev_in[5];
    RPPdeviceptr devB_super_scale = ctx.dev_in[6];
    RPPdeviceptr devB_super_zero  = ctx.dev_in[7];
    RPPdeviceptr devC             = ctx.dev_out[0];

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q4k.o");

    const int sizeA0           = M * K * in_bytes_per_element;
    const int sizeA1           = M * K * (int) sizeof(rpp::bfloat16);
    const int sizeB_q4         = (K * MATMUL_NS / 2);
    const int sizeB_qscale_lsb = (K * MATMUL_NS / 2 / q_group);
    const int sizeB_qscale_msb = sizeB_qscale_lsb / 2;
    const int sizeB_qzero_lsb  = sizeB_qscale_lsb;
    const int sizeB_qzero_msb  = sizeB_qscale_msb;

    const int sizeB_super_scale = (K * MATMUL_NS / super_group) * (int) sizeof(short);
    const int sizeB_super_zero  = sizeB_super_scale;

    const int sizeB_scale = (K * MATMUL_NS / q_group) * (int) sizeof(short);
    const int sizeB_zero  = sizeB_scale;

    const int sizeB   = (K * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeCr  = (M * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeC32 = (M * MATMUL_NS) * (int) sizeof(float);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);

    // --- ping-pong buffer for B weights ---
    RPPdeviceptr sramB_q4_0          = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_qscale_lsb_0  = sramB_q4_0 + round_up(sizeB_q4);
    RPPdeviceptr sramB_qzero_lsb_0   = sramB_qscale_lsb_0 + round_up(sizeB_qscale_lsb);
    RPPdeviceptr sramB_qscale_msb_0  = sramB_qzero_lsb_0 + round_up(sizeB_qzero_lsb);
    RPPdeviceptr sramB_qzero_msb_0   = sramB_qscale_msb_0 + round_up(sizeB_qscale_msb);
    RPPdeviceptr sramB_super_scale_0 = sramB_qzero_msb_0 + round_up(sizeB_qzero_msb);
    RPPdeviceptr sramB_super_zero_0  = sramB_super_scale_0 + round_up(sizeB_super_scale);

    RPPdeviceptr sramB_q4_1          = sramB_super_zero_0 + round_up(sizeB_super_zero);
    RPPdeviceptr sramB_qscale_lsb_1  = sramB_q4_1 + round_up(sizeB_q4);
    RPPdeviceptr sramB_qzero_lsb_1   = sramB_qscale_lsb_1 + round_up(sizeB_qscale_lsb);
    RPPdeviceptr sramB_qscale_msb_1  = sramB_qzero_lsb_1 + round_up(sizeB_qzero_lsb);
    RPPdeviceptr sramB_qzero_msb_1   = sramB_qscale_msb_1 + round_up(sizeB_qscale_msb);
    RPPdeviceptr sramB_super_scale_1 = sramB_qzero_msb_1 + round_up(sizeB_qzero_msb);
    RPPdeviceptr sramB_super_zero_1  = sramB_super_scale_1 + round_up(sizeB_super_scale);

    RPPdeviceptr sramB_scale = sramB_super_zero_1 + round_up(sizeB_super_zero);
    RPPdeviceptr sramB_zero  = sramB_scale + round_up(sizeB_scale);

    RPPdeviceptr sramB_out      = sramB_zero + round_up(sizeB_zero);
    RPPdeviceptr sramC_hw32     = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw      = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32 = sramC_chw + round_up(sizeCr);

    const int total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    RPPdeviceptr matmul_A = sramA_out;
    RPPdeviceptr matmul_B = sramB_out;
    RPPdeviceptr matmul_C = sramC_hw32;

    // Mark ping buffers as free at graph start
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);
    rtMemcpyAsync((void *) sramA_in, (const void *) devA, sizeA0, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, M * K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA_in, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    params.clear();
    chw2chw32_blocks(1, M, K, threadsPerBlock, blocksPerGrid);
    chw2chw32_align_params((int) sramA_in, (int) sramA_out, M, K, 0, (int) threadsPerBlock.x, (int) threadsPerBlock.y,
                           (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params, false);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    // --- B weights ---
    auto sramB_q4 = [&](int ping) {
        return ping ? sramB_q4_1 : sramB_q4_0;
    };
    auto sramB_qscale_lsb = [&](int ping) {
        return ping ? sramB_qscale_lsb_1 : sramB_qscale_lsb_0;
    };
    auto sramB_qzero_lsb = [&](int ping) {
        return ping ? sramB_qzero_lsb_1 : sramB_qzero_lsb_0;
    };
    auto sramB_qscale_msb = [&](int ping) {
        return ping ? sramB_qscale_msb_1 : sramB_qscale_msb_0;
    };
    auto sramB_qzero_msb = [&](int ping) {
        return ping ? sramB_qzero_msb_1 : sramB_qzero_msb_0;
    };
    auto sramB_super_scale = [&](int ping) {
        return ping ? sramB_super_scale_1 : sramB_super_scale_0;
    };
    auto sramB_super_zero = [&](int ping) {
        return ping ? sramB_super_zero_1 : sramB_super_zero_0;
    };

    const int q4_dma_rows_per_chunk = 128;

    auto schedule_dma_aux = [&](int ping, int tile_idx, int NsSeg) {
        const RPPdeviceptr tile_col_offset = (RPPdeviceptr) tile_idx * MATMUL_NS * (int) sizeof(short);

        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

        rtMemcpy2DAsync((void *) sramB_qscale_lsb(ping), NsSeg * (int) sizeof(short),
                        (const void *) (devB_qscale_lsb + tile_col_offset), N * (int) sizeof(short),
                        NsSeg * (int) sizeof(short), K / q_group / 4, rtMemcpyDeviceToSram, ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_qzero_lsb(ping), NsSeg * (int) sizeof(short),
                        (const void *) (devB_qzero_lsb + tile_col_offset), N * (int) sizeof(short),
                        NsSeg * (int) sizeof(short), K / q_group / 4, rtMemcpyDeviceToSram, ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_qscale_msb(ping), NsSeg * (int) sizeof(short),
                        (const void *) (devB_qscale_msb + tile_col_offset), N * (int) sizeof(short),
                        NsSeg * (int) sizeof(short), K / q_group / 8, rtMemcpyDeviceToSram, ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_qzero_msb(ping), NsSeg * (int) sizeof(short),
                        (const void *) (devB_qzero_msb + tile_col_offset), N * (int) sizeof(short),
                        NsSeg * (int) sizeof(short), K / q_group / 8, rtMemcpyDeviceToSram, ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_super_scale(ping), NsSeg * (int) sizeof(short),
                        (const void *) (devB_super_scale + tile_col_offset), N * (int) sizeof(short),
                        NsSeg * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram, ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_super_zero(ping), NsSeg * (int) sizeof(short),
                        (const void *) (devB_super_zero + tile_col_offset), N * (int) sizeof(short),
                        NsSeg * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram, ctx.dmaStream);

        rppEventRecord(ctx.dma_aux_done_ping[ping], ctx.dmaStream);
    };

    auto schedule_dma_q4 = [&](int ping, int tile_idx, int NsSeg) {
        const size_t       src_pitch       = N * (int) sizeof(short);
        const size_t       dst_pitch       = NsSeg * (int) sizeof(short);
        const int          q4_rows         = K / 4;
        const RPPdeviceptr src_q4_tile     = devB_q4 + (RPPdeviceptr) tile_idx * MATMUL_NS * (int) sizeof(short);

        for (int row_start = 0; row_start < q4_rows; row_start += q4_dma_rows_per_chunk) {
            const int          chunk_rows = std::min(q4_dma_rows_per_chunk, q4_rows - row_start);
            const RPPdeviceptr dst_chunk  = sramB_q4(ping) + (RPPdeviceptr) row_start * dst_pitch;
            const RPPdeviceptr src_chunk  = src_q4_tile + (RPPdeviceptr) row_start * src_pitch;

            rtMemcpy2DAsync((void *) dst_chunk, dst_pitch, (const void *) src_chunk, src_pitch, dst_pitch, chunk_rows,
                            rtMemcpyDeviceToSram, ctx.dmaStream);
        }

        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    const int nr_of_tiles = get_tiles(N);

    {
        int  ping    = 0;
        bool is_tail = (0 == (nr_of_tiles - 1));
        int  NsSeg   = get_ns(N, is_tail);
        schedule_dma_aux(ping, 0, NsSeg);
        schedule_dma_q4(ping, 0, NsSeg);
    }

    for (int i = 0; i < nr_of_tiles; ++i) {
        const bool is_tail = (i == (nr_of_tiles - 1));
        const int  NsSeg   = get_ns(N, is_tail);
        const int  ping    = (i & 1);

        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_aux_done_ping[ping], 0);

        if (i + 1 < nr_of_tiles) {
            const int next_i       = i + 1;
            const int next_is_tail = (next_i == (nr_of_tiles - 1));
            const int next_NsSeg   = get_ns(N, next_is_tail);
            const int next_ping    = next_i & 1;
            schedule_dma_aux(next_ping, next_i, next_NsSeg);
            schedule_dma_q4(next_ping, next_i, next_NsSeg);
        }

        // qscale_i6 * super_scale -> bf16 scales
        params.clear();
        q4k_super_scale_blocks(K, super_group, q_group, NsSeg, threadsPerBlock, blocksPerGrid);
        q4k_super_scale_params(sramB_qscale_lsb(ping), sramB_qscale_msb(ping), sramB_super_scale(ping), sramB_scale, K,
                               NsSeg, super_group, q_group, 0, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        // qmin_i6 * super_zero -> bf16 mins
        params.clear();
        q4k_super_scale_blocks(K, super_group, q_group, NsSeg, threadsPerBlock, blocksPerGrid);
        q4k_super_scale_params(sramB_qzero_lsb(ping), sramB_qzero_msb(ping), sramB_super_zero(ping), sramB_zero, K,
                               NsSeg, super_group, q_group, 1, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

        params.clear();
        q4k_dequant_blocks(1, K, NsSeg, threadsPerBlock, blocksPerGrid, weights_group);
        q4k_dequant_params(blocksPerGrid, threadsPerBlock, sramB_q4(ping), sramB_scale, sramB_zero, sramB_out, 0, NsSeg,
                           K, (int) sizeof(short), (int) sizeof(short), params);
        launchWrapperAysnc("matrix_mul_q4k_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        const int tn = get_tn(NsSeg);
        matmul_chw32_blocks(1, M, NsSeg, threadsPerBlock, blocksPerGrid, tn, 0);
        matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                                 (uint32_t) matmul_C, M, K, K, NsSeg, 1, 1, 1, 0, tn, (int) sizeof(rpp::bfloat16),
                                 (int) sizeof(rpp::bfloat16), params, false, false);
        launchWrapperAysnc(get_matmul_kernel(NsSeg), blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        chw322chw_blocks(1, M, NsSeg, threadsPerBlock, blocksPerGrid);
        chw322chw_align_params((int) matmul_C, (int) sramC_chw, M, NsSeg, 0, threadsPerBlock.x, threadsPerBlock.y,
                               threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
        launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                           ctx.rppBinMod, ctx.kernelStream);

        if (out_bytes_per_element == (int) sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, M * NsSeg * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpy2DAsync((void *) devC, N * (int) sizeof(float), (void *) sramC_chw_fp32,
                            NsSeg * (int) sizeof(float), NsSeg * (int) sizeof(float), M, rtMemcpySramToDevice,
                            ctx.kernelStream);

            devC += NsSeg * (int) sizeof(float);
        } else {
            rtMemcpy2DAsync((void *) devC, N * (int) sizeof(short), (void *) sramC_chw, NsSeg * (int) sizeof(short),
                            NsSeg * (int) sizeof(short), M, rtMemcpySramToDevice, ctx.kernelStream);
            devC += NsSeg * (int) sizeof(short);
        }
        rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q4k_build(rpp_kernel_context & ctx,
                                 int                  M,
                                 int                  K,
                                 int                  N,
                                 int                  weights_group,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  use_pipeline  = 0,
                                 int                  is_instantial = 1) {
    if (choose_m_tile(M, K, in_bytes_per_element, use_pipeline != 0) < M) {
        rpp_matmul_q4k_tiled(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
        return;
    }

    if (use_pipeline) {
        rpp_matmul_q4k_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                is_instantial);
    } else {
        rpp_matmul_q4k(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    }
}
}  // namespace kernel_q4_k
