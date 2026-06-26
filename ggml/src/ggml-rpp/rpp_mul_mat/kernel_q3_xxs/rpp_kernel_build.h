#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q3_xxs/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q3_xxs/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q3_xxs {

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

static void rpp_matmul_q3xxs_build(rpp_kernel_context & ctx,
                                   int                  M,
                                   int                  K,
                                   int                  N,
                                   int                  weights_group,
                                   int                  in_bytes_per_element,
                                   int                  out_bytes_per_element,
                                   int                  is_instantial = 1) {
    const int q_group             = 32;
    const int super_group         = 256;
    const int qscale_lut_elems    = 16;
    const int grid_lut_elems      = 256;
    const int qscale_lut_bytes    = qscale_lut_elems * (int) sizeof(uint16_t);
    const int grid_lut_bytes      = grid_lut_elems * (int) sizeof(uint32_t);
    const int lut_workspace_bytes = qscale_lut_bytes + grid_lut_bytes;
    if (K % super_group != 0) {
        throw std::runtime_error("Q3XXS requires K % 256 == 0");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA              = ctx.dev_in[0];
    RPPdeviceptr devB_q4           = ctx.dev_in[1];
    RPPdeviceptr devB_qscale       = ctx.dev_in[2];
    RPPdeviceptr devB_qsign        = ctx.dev_in[3];
    RPPdeviceptr devB_super_scale  = ctx.dev_in[4];
    RPPdeviceptr devC              = ctx.dev_out[0];
    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;

    RPPdeviceptr devB_qscale_lut = dev_lut_workspace;
    RPPdeviceptr devB_grid_lut   = devB_qscale_lut + qscale_lut_bytes;

    std::array<uint16_t, qscale_lut_elems> qscale_lut = {};
    for (int i = 0; i < qscale_lut_elems; ++i) {
        const float scale4  = (float) i;
        const float lut_val = (0.5f + scale4) * 0.5f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }
    std::array<uint32_t, grid_lut_elems> grid_lut = {};
    std::memcpy(grid_lut.data(), iq3xxs_grid_local, grid_lut_bytes);

    rtMemcpy((void *) devB_qscale_lut, qscale_lut.data(), qscale_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_grid_lut, grid_lut.data(), grid_lut_bytes, rtMemcpyHostToDevice);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q3xxs.o");

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

    // q3xxs packed inputs:
    // q4/codebook:   [K/8][N]   (uint16)
    // qscale:        [K/128][N] (uint16, packed 4-bit scales)
    // qsign:         [K/16][N]  (uint16, packed sign bits)
    // super_scale:   [K/256][N] (bf16)
    const int sizeB_q4          = (K * MATMUL_NS / 4);
    const int sizeB_qscale      = (K * MATMUL_NS / 64);
    const int sizeB_qsign       = (K * MATMUL_NS / 8);
    const int sizeB_super_scale = (K * MATMUL_NS / super_group) * (int) sizeof(short);
    const int sizeB_qscale_lut  = qscale_lut_bytes;
    // 256-entry iq3xxs grid decode table, each entry is packed 4x uint8 magnitudes.
    const int sizeB_grid_lut    = grid_lut_bytes;

    const int sizeB_scale = (K * MATMUL_NS / q_group) * (int) sizeof(short);

    const int sizeB   = (K * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeCr  = (M * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeC32 = (M * MATMUL_NS) * (int) sizeof(float);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);

    RPPdeviceptr sramB_q4          = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_qscale      = sramB_q4 + round_up(sizeB_q4);
    RPPdeviceptr sramB_qsign       = sramB_qscale + round_up(sizeB_qscale);
    RPPdeviceptr sramB_super_scale = sramB_qsign + round_up(sizeB_qsign);
    RPPdeviceptr sramB_qscale_lut  = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_grid_lut    = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_scale       = sramB_grid_lut + round_up(sizeB_grid_lut);
    RPPdeviceptr sramB_out         = sramB_scale + round_up(sizeB_scale);
    RPPdeviceptr sramC_hw32        = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw         = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32    = sramC_chw + round_up(sizeCr);

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
    rtMemcpyAsync((void *) sramB_qscale_lut, (const void *) devB_qscale_lut, sizeB_qscale_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_grid_lut, (const void *) devB_grid_lut, sizeB_grid_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

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

        rtMemcpy2DAsync((void *) sramB_q4, ns * (int) sizeof(short), (const void *) devB_q4, N * (int) sizeof(short),
                        ns * (int) sizeof(short), K / 8, rtMemcpyDeviceToSram, ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_qscale, ns * (int) sizeof(short), (const void *) devB_qscale,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 128, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_qsign, ns * (int) sizeof(short), (const void *) devB_qsign,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 16, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_super_scale, ns * (int) sizeof(short), (const void *) devB_super_scale,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        // qscale_i4 * super_scale -> bf16 scales (via LUT)
        params.clear();
        q3xxs_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
        q3xxs_super_scale_params(sramB_qscale, sramB_super_scale, sramB_qscale_lut, sramB_scale, K, ns, super_group,
                                 q_group, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        //jsq3xxs
        params.clear();
        q3xxs_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
        q3xxs_dequant_params(blocksPerGrid, threadsPerBlock, sramB_q4, sramB_qsign, sramB_scale, sramB_out,
                             sramB_grid_lut, ns, K, (int) sizeof(short), (int) sizeof(short), params);
        launchWrapperAysnc("matrix_mul_q3xxs_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
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

        devB_q4 += ns * (int) sizeof(short);
        devB_qscale += ns * (int) sizeof(short);
        devB_qsign += ns * (int) sizeof(short);
        devB_super_scale += ns * (int) sizeof(short);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
}  // namespace kernel_q3_xxs
