#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q2_xs/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q2_xs/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q2_xs {
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

static inline uint8_t q2xs_reverse_bits_u8(uint8_t x) {
    x = (uint8_t) (((x & 0xF0u) >> 4) | ((x & 0x0Fu) << 4));
    x = (uint8_t) (((x & 0xCCu) >> 2) | ((x & 0x33u) << 2));
    x = (uint8_t) (((x & 0xAAu) >> 1) | ((x & 0x55u) << 1));
    return x;
}

static void q2xs_prepare_lookup_tables(rpp_kernel_context & ctx,
                                       int                  qscale_lut_bytes,
                                       int                  codebook_lut_bytes,
                                       int                  sign_lut_bytes,
                                       RPPdeviceptr &       devB_qscale_lut,
                                       RPPdeviceptr &       devB_codebook_lut,
                                       RPPdeviceptr &       devB_sign_lut) {
    auto align_up = [](size_t x, size_t a) -> size_t {
        return ((x + a - 1) / a) * a;
    };

    const size_t off_qscale_lut   = 0;
    const size_t off_codebook_lut = align_up(off_qscale_lut + (size_t) qscale_lut_bytes, 64);
    const size_t off_sign_lut     = align_up(off_codebook_lut + (size_t) codebook_lut_bytes, 64);
    const size_t workspace_bytes  = align_up(off_sign_lut + (size_t) sign_lut_bytes, 64);

    // Workspace holds q2xs LUTs generated on host:
    // qscale LUT (for super-scale), codebook LUT (iq2xs_grid), and sign LUT (reversed-bit ksigns_iq2xs).
    // Reuse the shared workspace to avoid invalidating pointers captured by other kernels.
    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        if (rtMalloc((void **) &dev_lut_workspace, workspace_bytes) != rtSuccess) {
            throw std::runtime_error("Q2XS rtMalloc failed for LUT workspace");
        }
        ctx.dev_workspace = dev_lut_workspace;
    }

    RPPdeviceptr base = dev_lut_workspace;
    devB_qscale_lut   = base + (RPPdeviceptr) off_qscale_lut;
    devB_codebook_lut = base + (RPPdeviceptr) off_codebook_lut;
    devB_sign_lut     = base + (RPPdeviceptr) off_sign_lut;

    std::array<uint16_t, 16> qscale_lut = {};
    for (int i = 0; i < (int) qscale_lut.size(); ++i) {
        const float lut_val = (0.5f + (float) i) * 0.25f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }

    std::array<uint64_t, 512> codebook_lut = {};
    // q2xs uses its dedicated 512-entry iq2xs grid table (same packed 8x-u8 format).
    std::memcpy(codebook_lut.data(), iq2xs_grid_local, sizeof(codebook_lut));

    std::array<uint8_t, 128> sign_lut = {};
    for (int i = 0; i < (int) sign_lut.size(); ++i) {
        sign_lut[i] = q2xs_reverse_bits_u8(ksigns_iq2xs_local[i]);
    }

    rtMemcpy((void *) devB_qscale_lut, qscale_lut.data(), qscale_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_codebook_lut, codebook_lut.data(), codebook_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_sign_lut, sign_lut.data(), sign_lut_bytes, rtMemcpyHostToDevice);
}

static void rpp_matmul_q2xs_build(rpp_kernel_context & ctx,
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
    const int codebook_lut_elems  = 512;
    const int sign_lut_elems      = 128;
    const int qscale_lut_bytes    = qscale_lut_elems * (int) sizeof(uint16_t);
    const int codebook_lut_bytes  = codebook_lut_elems * (int) sizeof(uint64_t);
    const int sign_lut_bytes      = sign_lut_elems * (int) sizeof(uint8_t);
    const int lut_workspace_bytes = qscale_lut_bytes + codebook_lut_bytes + sign_lut_bytes;
    (void) lut_workspace_bytes;
    if (ctx.dev_in.size() < 4 || ctx.dev_out.empty()) {
        throw std::runtime_error("Q2XS requires 4 input buffers (A, qs, scales, super_scale) and 1 output buffer");
    }
    if (K % super_group != 0) {
        throw std::runtime_error("Q2XS requires K % 256 == 0");
    }
    if (weights_group != super_group) {
        throw std::runtime_error("Q2XS requires weights_group == 256");
    }
    const char * force_tn1_env = std::getenv("RPP_Q2XS_FORCE_TN1");
    const bool   force_tn1     = (force_tn1_env != nullptr) && (std::atoi(force_tn1_env) != 0);
    if (force_tn1) {
        // std::cout << "[Q2XS DBG] RPP_Q2XS_FORCE_TN1=1, forcing tn1 for all tiles" << std::endl;
    }
    const char * force_ss_bx32_env = std::getenv("RPP_Q2XS_SS_BX32");
    const bool   force_ss_bx32     = (force_ss_bx32_env != nullptr) && (std::atoi(force_ss_bx32_env) != 0);
    if (force_ss_bx32) {
        // std::cout << "[Q2XS DBG] RPP_Q2XS_SS_BX32=1, forcing q2xs_super_scale block_x=32" << std::endl;
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA              = ctx.dev_in[0];
    RPPdeviceptr devB_qs           = ctx.dev_in[1];
    RPPdeviceptr devB_scales       = ctx.dev_in[2];
    RPPdeviceptr devB_super_scale  = ctx.dev_in[3];
    RPPdeviceptr devC              = ctx.dev_out[0];
    RPPdeviceptr devB_qscale_lut   = 0;
    RPPdeviceptr devB_codebook_lut = 0;
    RPPdeviceptr devB_sign_lut     = 0;
    q2xs_prepare_lookup_tables(ctx, qscale_lut_bytes, codebook_lut_bytes, sign_lut_bytes, devB_qscale_lut,
                               devB_codebook_lut, devB_sign_lut);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    // Native q2xs kernels + q2xs local block/param helpers.
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q2xs.o");

    const int sizeA0 = M * K * in_bytes_per_element;
    const int sizeA1 = M * K * (int) sizeof(rpp::bfloat16);

    // q2xs packed inputs:
    // qs:            [K/8][N]   (uint16, merged codebook+sign word)
    // scales:        [K/64][N]  (uint16, 2x byte per word)
    // super_scale:   [K/256][N] (bf16)
    const int sizeB_qs           = (K * MATMUL_NS / 4);
    const int sizeB_scales       = (K * MATMUL_NS / 32);
    const int sizeB_super_scale  = (K * MATMUL_NS / super_group) * (int) sizeof(short);
    const int sizeB_qscale_lut   = qscale_lut_bytes;
    const int sizeB_codebook_lut = codebook_lut_bytes;
    const int sizeB_sign_lut     = sign_lut_bytes;

    // q2xs dequant scale tensor consumes one bf16 scale per 16 weights (K/16 rows),
    // not per q_group(=32). Using /32 under-allocates and can overlap
    // the dequant output buffer in SRAM.
    const int sizeB_scale = (K * MATMUL_NS / 16) * (int) sizeof(short);

    const int sizeB   = (K * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeCr  = (M * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeC32 = (M * MATMUL_NS) * (int) sizeof(float);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);

    RPPdeviceptr sramB_qs           = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_scales       = sramB_qs + round_up(sizeB_qs);
    RPPdeviceptr sramB_super_scale  = sramB_scales + round_up(sizeB_scales);
    RPPdeviceptr sramB_qscale_lut   = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_codebook_lut = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_sign_lut     = sramB_codebook_lut + round_up(sizeB_codebook_lut);
    RPPdeviceptr sramB_scale        = sramB_sign_lut + round_up(sizeB_sign_lut);
    RPPdeviceptr sramB_out          = sramB_scale + round_up(sizeB_scale);
    RPPdeviceptr sramC_hw32         = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw          = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32     = sramC_chw + round_up(sizeCr);

    const int total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    const bool   a_is_f32 = (in_bytes_per_element == (int) sizeof(float));
    RPPdeviceptr matmul_A = a_is_f32 ? sramA_in : sramA_out;
    RPPdeviceptr matmul_B = sramB_out;
    RPPdeviceptr matmul_C = sramC_hw32;

    rtMemcpyAsync((void *) sramA_in, (const void *) devA, sizeA0, rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_qscale_lut, (const void *) devB_qscale_lut, sizeB_qscale_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_codebook_lut, (const void *) devB_codebook_lut, sizeB_codebook_lut,
                  rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_sign_lut, (const void *) devB_sign_lut, sizeB_sign_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

    if (a_is_f32) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        // HW workaround: avoid in-place CVT_32_16 on SRAM buffer.
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA_out, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    params.clear();
    chw2chw32_blocks(1, M, K, threadsPerBlock, blocksPerGrid);
    // If A was f32, source is converted bf16 in sramA_out and destination is sramA_in.
    // Otherwise source is bf16 in sramA_in and destination is sramA_out (original behavior).
    chw2chw32_align_params((int) (a_is_f32 ? sramA_out : sramA_in), (int) (a_is_f32 ? sramA_in : sramA_out), M, K, 0,
                           (int) threadsPerBlock.x, (int) threadsPerBlock.y, (int) threadsPerBlock.z,
                           (int) sizeof(rpp::bfloat16), params, false);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    for (int i = 0; i < get_tiles(N); i++) {
        bool              is_tail       = (i == (get_tiles(N) - 1));
        int               ns            = get_ns(N, is_tail);
        const int         tn            = force_tn1 ? 1 : get_tn(ns);
        const std::string matmul_kernel = force_tn1 ? "matmul_tn1_f16_f32_f16" : get_matmul_kernel(ns);
        if (i == 0) {
            // std::cout << "[Q2XS DBG] tile ns=" << ns << " tn=" << tn << " kernel=" << matmul_kernel << std::endl;
        }

        rtMemcpy2DAsync((void *) sramB_qs, ns * (int) sizeof(short), (const void *) devB_qs, N * (int) sizeof(short),
                        ns * (int) sizeof(short), K / 8, rtMemcpyDeviceToSram, ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_scales, ns * (int) sizeof(short), (const void *) devB_scales,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 64, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_super_scale, ns * (int) sizeof(short), (const void *) devB_super_scale,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        // scale_i4 * super_scale -> bf16 scales (via LUT)
        params.clear();
        q2xs_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
        if (force_ss_bx32 && (ns % 32 == 0)) {
            // Debug override: avoid multi-warp x-dimension in super-scale kernel.
            threadsPerBlock.x = 32;
            blocksPerGrid.x   = (uint32_t) (ns / 32);
            if (i == 0) {
                // std::cout << "[Q2XS DBG] super_scale override: block_x=" << threadsPerBlock.x
                //           << " grid_x=" << blocksPerGrid.x << std::endl;
            }
        }
        q2xs_super_scale_params(sramB_scales, sramB_super_scale, sramB_qscale_lut, sramB_scale, K, ns, super_group,
                                q_group, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q2xs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        q2xs_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
        q2xs_dequant_params(blocksPerGrid, threadsPerBlock, (uint32_t) sramB_qs, (uint32_t) sramB_scale,
                            (uint32_t) sramB_out, (uint32_t) sramB_codebook_lut, (uint32_t) sramB_sign_lut, ns, K,
                            (int) sizeof(short), (int) sizeof(short), params);
        launchWrapperAysnc("matrix_mul_q2xs_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
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

        devB_qs += ns * (int) sizeof(short);
        devB_scales += ns * (int) sizeof(short);
        devB_super_scale += ns * (int) sizeof(short);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

}  // namespace kernel_q2_xs
