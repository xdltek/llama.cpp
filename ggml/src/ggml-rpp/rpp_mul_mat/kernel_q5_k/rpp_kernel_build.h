#pragma once
// rpp_matmul_q4k.cpp
// q4k backend: q6k-style scale/min expansion + q4 dequant/matmul kernels.

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q5_k/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q5_k/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q5_k {

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

static void rpp_matmul_q5k(rpp_kernel_context & ctx,
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
    RPPdeviceptr devB_q4_lsb      = ctx.dev_in[1];
    RPPdeviceptr devB_q4_msb      = ctx.dev_in[2];
    RPPdeviceptr devB_qscale_lsb  = ctx.dev_in[3];
    RPPdeviceptr devB_qzero_lsb   = ctx.dev_in[4];
    RPPdeviceptr devB_qscale_msb  = ctx.dev_in[5];
    RPPdeviceptr devB_qzero_msb   = ctx.dev_in[6];
    RPPdeviceptr devB_super_scale = ctx.dev_in[7];
    RPPdeviceptr devB_super_zero  = ctx.dev_in[8];
    RPPdeviceptr devC             = ctx.dev_out[0];

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q5k.o");

    const int sizeA0           = M * K * in_bytes_per_element;
    const int sizeA1           = M * K * (int) sizeof(rpp::bfloat16);
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // q_group = 32
    // wq_per_group = 32
    // in_wq_lsb     [K/256]      |  [8]  |  [32/16] | [4] |  [4][N]
    // in_wq_lsb     [z]          |  [y]  |  [loop0] |     |  [x]
    //               [grid.y]*[z] |  [y]                   |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // in_wq_msb     [K/256]      |  [8]  |  [32/16]       |  [16][N]
    // in_wq_msb     [z]          |  [y]  |  [loop0]       |  [x]
    //               [grid.y]*[z] |  [y]                   |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    const int sizeB_q4_lsb     = (K * MATMUL_NS / 2);
    const int sizeB_q4_msb     = sizeB_q4_lsb / 4;
    const int sizeB_qscale_lsb = sizeB_q4_lsb / q_group;
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

    RPPdeviceptr sramB_q4_lsb     = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_q4_msb     = sramB_q4_lsb + round_up(sizeB_q4_lsb);
    RPPdeviceptr sramB_qscale_lsb = sramB_q4_msb + round_up(sizeB_q4_msb);
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
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
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

        rtMemcpy2DAsync((void *) sramB_q4_lsb, ns * (int) sizeof(short), (const void *) devB_q4_lsb,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 4, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_q4_msb, ns * (int) sizeof(short), (const void *) devB_q4_msb,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 4 / 4, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

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
        q5k_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
        q5k_super_scale_params(sramB_qscale_lsb, sramB_qscale_msb, sramB_super_scale, sramB_scale, K, ns, super_group,
                               q_group, 0, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q5k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        // qmin_i6 * super_zero -> bf16 mins
        // qmin can reuse same kernel of qscale
        params.clear();
        q5k_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
        q5k_super_scale_params(sramB_qzero_lsb, sramB_qzero_msb, sramB_super_zero, sramB_zero, K, ns, super_group,
                               q_group, 1, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q5k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        //jsq5k
        params.clear();
        q5k_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
        q5k_dequant_params(blocksPerGrid, threadsPerBlock, sramB_q4_lsb, sramB_q4_msb, sramB_scale, sramB_zero,
                           sramB_out, 0, ns, K, (int) sizeof(short), (int) sizeof(short), params);
        launchWrapperAysnc("matrix_mul_q5k_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
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
            calc_tbdim_flattern(M, ns * 2, threadsPerBlock, blocksPerGrid);
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

        devB_q4_lsb += ns * (int) sizeof(short);
        devB_q4_msb += ns * (int) sizeof(short);
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

static void rpp_matmul_q5k_pipeline(rpp_kernel_context & ctx,
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
    RPPdeviceptr devB_q4_lsb      = ctx.dev_in[1];
    RPPdeviceptr devB_q4_msb      = ctx.dev_in[2];
    RPPdeviceptr devB_qscale_lsb  = ctx.dev_in[3];
    RPPdeviceptr devB_qzero_lsb   = ctx.dev_in[4];
    RPPdeviceptr devB_qscale_msb  = ctx.dev_in[5];
    RPPdeviceptr devB_qzero_msb   = ctx.dev_in[6];
    RPPdeviceptr devB_super_scale = ctx.dev_in[7];
    RPPdeviceptr devB_super_zero  = ctx.dev_in[8];
    RPPdeviceptr devC             = ctx.dev_out[0];

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q5k.o");

    const int sizeA0           = M * K * in_bytes_per_element;
    const int sizeA1           = M * K * (int) sizeof(rpp::bfloat16);
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // q_group = 32
    // wq_per_group = 32
    // in_wq_lsb     [K/256]      |  [8]  |  [32/16] | [4] |  [4][N]
    // in_wq_lsb     [z]          |  [y]  |  [loop0] |     |  [x]
    //               [grid.y]*[z] |  [y]                   |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // in_wq_msb     [K/256]      |  [8]  |  [32/16]       |  [16][N]
    // in_wq_msb     [z]          |  [y]  |  [loop0]       |  [x]
    //               [grid.y]*[z] |  [y]                   |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    const int sizeB_q4_lsb     = (K * MATMUL_NS / 2);
    const int sizeB_q4_msb     = sizeB_q4_lsb / 4;
    const int sizeB_qscale_lsb = sizeB_q4_lsb / q_group;
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
    RPPdeviceptr sramB_q4_lsb_0      = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_q4_msb_0      = sramB_q4_lsb_0 + round_up(sizeB_q4_lsb);
    RPPdeviceptr sramB_qscale_lsb_0  = sramB_q4_msb_0 + round_up(sizeB_q4_msb);
    RPPdeviceptr sramB_qzero_lsb_0   = sramB_qscale_lsb_0 + round_up(sizeB_qscale_lsb);
    RPPdeviceptr sramB_qscale_msb_0  = sramB_qzero_lsb_0 + round_up(sizeB_qzero_lsb);
    RPPdeviceptr sramB_qzero_msb_0   = sramB_qscale_msb_0 + round_up(sizeB_qscale_msb);
    RPPdeviceptr sramB_super_scale_0 = sramB_qzero_msb_0 + round_up(sizeB_qzero_msb);
    RPPdeviceptr sramB_super_zero_0  = sramB_super_scale_0 + round_up(sizeB_super_scale);

    RPPdeviceptr sramB_q4_lsb_1      = sramB_super_zero_0 + round_up(sizeB_super_zero);
    RPPdeviceptr sramB_q4_msb_1      = sramB_q4_lsb_1 + round_up(sizeB_q4_lsb);
    RPPdeviceptr sramB_qscale_lsb_1  = sramB_q4_msb_1 + round_up(sizeB_q4_msb);
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
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
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
    auto sramB_q4_lsb = [&](int ping) {
        return ping ? sramB_q4_lsb_1 : sramB_q4_lsb_0;
    };
    auto sramB_q4_msb = [&](int ping) {
        return ping ? sramB_q4_msb_1 : sramB_q4_msb_0;
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

    auto schedule_dma = [&](int ping, int NsSeg) {
        // Ensure previous kernel using this ping buffer has finished
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

        rtMemcpy2DAsync((void *) sramB_q4_lsb(ping), NsSeg * (int) sizeof(short), (const void *) devB_q4_lsb,
                        N * (int) sizeof(short), NsSeg * (int) sizeof(short), K / 4, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_q4_msb(ping), NsSeg * (int) sizeof(short), (const void *) devB_q4_msb,
                        N * (int) sizeof(short), NsSeg * (int) sizeof(short), K / 4 / 4, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_qscale_lsb(ping), NsSeg * (int) sizeof(short), (const void *) devB_qscale_lsb,
                        N * (int) sizeof(short), NsSeg * (int) sizeof(short), K / q_group / 4, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_qzero_lsb(ping), NsSeg * (int) sizeof(short), (const void *) devB_qzero_lsb,
                        N * (int) sizeof(short), NsSeg * (int) sizeof(short), K / q_group / 4, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_qscale_msb(ping), NsSeg * (int) sizeof(short), (const void *) devB_qscale_msb,
                        N * (int) sizeof(short), NsSeg * (int) sizeof(short), K / q_group / 8, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_qzero_msb(ping), NsSeg * (int) sizeof(short), (const void *) devB_qzero_msb,
                        N * (int) sizeof(short), NsSeg * (int) sizeof(short), K / q_group / 8, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_super_scale(ping), NsSeg * (int) sizeof(short), (const void *) devB_super_scale,
                        N * (int) sizeof(short), NsSeg * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_super_zero(ping), NsSeg * (int) sizeof(short), (const void *) devB_super_zero,
                        N * (int) sizeof(short), NsSeg * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        devB_q4_lsb += NsSeg * (int) sizeof(short);
        devB_q4_msb += NsSeg * (int) sizeof(short);
        devB_qscale_lsb += NsSeg * (int) sizeof(short);
        devB_qzero_lsb += NsSeg * (int) sizeof(short);
        devB_qscale_msb += NsSeg * (int) sizeof(short);
        devB_qzero_msb += NsSeg * (int) sizeof(short);
        devB_super_scale += NsSeg * (int) sizeof(short);
        devB_super_zero += NsSeg * (int) sizeof(short);

        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    const int nr_of_tiles = get_tiles(N);

    // --- prefetch first tile ---
    {
        int  ping    = 0;
        bool is_tail = (0 == (nr_of_tiles - 1));
        int  NsSeg   = get_ns(N, is_tail);
        schedule_dma(ping, NsSeg);
    }

    for (int i = 0; i < nr_of_tiles; ++i) {
        const bool is_tail = (i == (nr_of_tiles - 1));
        const int  NsSeg   = get_ns(N, is_tail);
        const int  ping    = (i & 1);

        // --- wait for current DMA (ping-specific) ---
        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

        // --- prefetch next segment ---
        if (i + 1 < nr_of_tiles) {
            const int next_i       = i + 1;
            const int next_is_tail = (next_i == (nr_of_tiles - 1));
            const int next_NsSeg   = get_ns(N, next_is_tail);
            const int next_ping    = next_i & 1;
            schedule_dma(next_ping, next_NsSeg);
        }

        // qscale_i6 * super_scale -> bf16 scales
        params.clear();
        q5k_super_scale_blocks(K, super_group, q_group, NsSeg, threadsPerBlock, blocksPerGrid);
        q5k_super_scale_params(sramB_qscale_lsb(ping), sramB_qscale_msb(ping), sramB_super_scale(ping), sramB_scale, K,
                               NsSeg, super_group, q_group, 0, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q5k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        // qmin_i6 * super_zero -> bf16 mins
        // qmin can reuse same kernel of qscale
        params.clear();
        q5k_super_scale_blocks(K, super_group, q_group, NsSeg, threadsPerBlock, blocksPerGrid);
        q5k_super_scale_params(sramB_qzero_lsb(ping), sramB_qzero_msb(ping), sramB_super_zero(ping), sramB_zero, K,
                               NsSeg, super_group, q_group, 1, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q5k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        //jsq5k
        params.clear();
        q5k_dequant_blocks(1, K, NsSeg, threadsPerBlock, blocksPerGrid, weights_group);
        q5k_dequant_params(blocksPerGrid, threadsPerBlock, sramB_q4_lsb(ping), sramB_q4_msb(ping), sramB_scale,
                           sramB_zero, sramB_out, 0, NsSeg, K, (int) sizeof(short), (int) sizeof(short), params);
        launchWrapperAysnc("matrix_mul_q5k_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
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
            calc_tbdim_flattern(M, NsSeg * 2, threadsPerBlock, blocksPerGrid);
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
        // Signal this ping buffer is safe to reuse
        rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q5k_build(rpp_kernel_context & ctx,
                                 int                  M,
                                 int                  K,
                                 int                  N,
                                 int                  weights_group,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  use_pipeline  = 0,
                                 int                  is_instantial = 1) {
    // if (use_pipeline) {
    //     rpp_matmul_q5k_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    // } else {
    //     rpp_matmul_q5k(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    // }
    rpp_matmul_q5k(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
}
}  // namespace kernel_q5_k
