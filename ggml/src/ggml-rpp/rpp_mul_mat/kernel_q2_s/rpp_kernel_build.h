#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q2_s/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q2_s/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q2_s {

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

static void rpp_matmul_q2s_build(rpp_kernel_context & ctx,
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
    const int grid_lut_elems      = 1024;
    const int qscale_lut_bytes    = qscale_lut_elems * (int) sizeof(uint16_t);
    const int grid_lut_bytes      = grid_lut_elems * (int) sizeof(uint64_t);
    const int lut_workspace_bytes = qscale_lut_bytes + grid_lut_bytes;
    if (ctx.dev_in.size() < 6 || ctx.dev_out.empty()) {
        throw std::runtime_error("Q2S requires 6 input buffers and 1 output buffer");
    }
    if (K % super_group != 0) {
        throw std::runtime_error("Q2S requires K % 256 == 0");
    }
    if (weights_group != super_group) {
        throw std::runtime_error("Q2S requires weights_group == 256");
    }
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA              = ctx.dev_in[0];
    RPPdeviceptr devB_codebook_lsb = ctx.dev_in[1];
    RPPdeviceptr devB_codebook_msb = ctx.dev_in[2];
    RPPdeviceptr devB_scales       = ctx.dev_in[3];
    RPPdeviceptr devB_sign         = ctx.dev_in[4];
    RPPdeviceptr devB_super_scale  = ctx.dev_in[5];
    RPPdeviceptr devC              = ctx.dev_out[0];
    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        if (rtMalloc((void **) &dev_lut_workspace, lut_workspace_bytes) != rtSuccess) {
            throw std::runtime_error("Q2S rtMalloc failed for LUT workspace");
        }
        ctx.dev_workspace = dev_lut_workspace;
    }

    RPPdeviceptr devB_qscale_lut = dev_lut_workspace;
    RPPdeviceptr devB_grid_lut   = devB_qscale_lut + qscale_lut_bytes;

    std::array<uint16_t, qscale_lut_elems> qscale_lut = {};
    for (int i = 0; i < qscale_lut_elems; ++i) {
        const float scale4  = (float) i;
        const float lut_val = (0.5f + scale4) * 0.25f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }
    std::array<uint64_t, grid_lut_elems> grid_lut = {};
    std::memcpy(grid_lut.data(), iq2s_grid_local, grid_lut_bytes);

    rtMemcpy((void *) devB_qscale_lut, qscale_lut.data(), qscale_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_grid_lut, grid_lut.data(), grid_lut_bytes, rtMemcpyHostToDevice);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q2s.o");

    const int sizeA0 = M * K * in_bytes_per_element;
    const int sizeA1 = M * K * (int) sizeof(rpp::bfloat16);

    // q2s packed inputs:
    // codebook_lsb:  [K/16][N]  (uint16, 2x byte per word)
    // codebook_msb:  [K/64][N]  (uint16, 2x byte per word)
    // scales:        [K/64][N]  (uint16, 2x byte per word)
    // sign:          [K/16][N]  (uint16, 2x byte per word)
    // super_scale:   [K/256][N] (bf16)
    const int sizeB_codebook_lsb = (K * MATMUL_NS / 8);
    const int sizeB_codebook_msb = (K * MATMUL_NS / 32);
    const int sizeB_scales       = (K * MATMUL_NS / 32);
    const int sizeB_sign         = (K * MATMUL_NS / 8);
    const int sizeB_super_scale  = (K * MATMUL_NS / super_group) * (int) sizeof(short);
    const int sizeB_qscale_lut   = qscale_lut_bytes;
    // 1024-entry iq2s grid decode table, each entry is packed 8x uint8 magnitudes.
    const int sizeB_grid_lut     = grid_lut_bytes;

    // q2s dequant consumes one bf16 scale per 16 weights (K/16 rows),
    // not per q_group(=32). Using /32 under-allocates and can overlap
    // the dequant output buffer in SRAM.
    const int sizeB_scale = (K * MATMUL_NS / 16) * (int) sizeof(short);

    const int sizeB   = (K * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeCr  = (M * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeC32 = (M * MATMUL_NS) * (int) sizeof(float);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);

    RPPdeviceptr sramB_codebook_lsb = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_codebook_msb = sramB_codebook_lsb + round_up(sizeB_codebook_lsb);
    RPPdeviceptr sramB_scales       = sramB_codebook_msb + round_up(sizeB_codebook_msb);
    RPPdeviceptr sramB_sign         = sramB_scales + round_up(sizeB_scales);
    RPPdeviceptr sramB_super_scale  = sramB_sign + round_up(sizeB_sign);
    RPPdeviceptr sramB_qscale_lut   = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_grid_lut     = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_scale        = sramB_grid_lut + round_up(sizeB_grid_lut);
    RPPdeviceptr sramB_out          = sramB_scale + round_up(sizeB_scale);
    RPPdeviceptr sramC_hw32         = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw          = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32     = sramC_chw + round_up(sizeCr);
    const int    total_sram_bytes   = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int    SRAM_LIMIT         = 22 * 1024 * 1024;
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
        bool              is_tail       = (i == (get_tiles(N) - 1));
        int               ns            = get_ns(N, is_tail);
        const int         tn            = get_tn(ns);
        const std::string matmul_kernel = get_matmul_kernel(ns);

        rtMemcpy2DAsync((void *) sramB_codebook_lsb, ns * (int) sizeof(short), (const void *) devB_codebook_lsb,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 16, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_codebook_msb, ns * (int) sizeof(short), (const void *) devB_codebook_msb,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 64, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_scales, ns * (int) sizeof(short), (const void *) devB_scales,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 64, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_sign, ns * (int) sizeof(short), (const void *) devB_sign,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 16, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_super_scale, ns * (int) sizeof(short), (const void *) devB_super_scale,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        // scale_i4 * super_scale -> bf16 scales (via LUT)
        params.clear();
        q2s_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
        q2s_super_scale_params(sramB_scales, sramB_super_scale, sramB_qscale_lut, sramB_scale, K, ns, super_group,
                               q_group, blocksPerGrid, threadsPerBlock, params);
        // Kernel symbol names are still q3xxs_* in matmul_q2s kernel_cfg.
        launchWrapperAysnc("matrix_mul_q2s_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        // codebook_msb is already staged in SRAM for q2s dequant kernels.
        params.clear();
        q2s_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
        q2s_dequant_params(blocksPerGrid, threadsPerBlock, sramB_codebook_lsb, sramB_codebook_msb, sramB_sign,
                           sramB_scale, sramB_out, sramB_grid_lut, ns, K, (int) sizeof(short), (int) sizeof(short),
                           params);
        // Kernel symbol names are still q3xxs_* in matmul_q2s kernel_cfg.
        launchWrapperAysnc("matrix_mul_q2s_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        matmul_chw32_blocks(1, M, ns, threadsPerBlock, blocksPerGrid, tn, 0);
        matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                                 (uint32_t) matmul_C, M, K, K, ns, 1, 1, 1, 0, tn, (int) sizeof(rpp::bfloat16),
                                 (int) sizeof(rpp::bfloat16), params, false, false);
        launchWrapperAysnc(matmul_kernel, blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);

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

        devB_codebook_lsb += ns * (int) sizeof(short);
        devB_codebook_msb += ns * (int) sizeof(short);
        devB_scales += ns * (int) sizeof(short);
        devB_sign += ns * (int) sizeof(short);
        devB_super_scale += ns * (int) sizeof(short);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
}  // namespace kernel_q2_s
