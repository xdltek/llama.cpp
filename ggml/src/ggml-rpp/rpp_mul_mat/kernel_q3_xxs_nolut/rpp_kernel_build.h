#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q3_xxs_nolut/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q3_xxs_nolut/rpp_kernel_param.h"

#include <assert.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q3_xxs_nolut {

constexpr std::array<float, 8> kQ3xxsNoLutMagValues = {
    4.0f, 12.0f, 20.0f, 28.0f, 36.0f, 44.0f, 52.0f, 62.0f,
};

inline int q3xxs_nolut_codebook_rows(int K) {
    if ((K % 16) != 0) {
        throw std::runtime_error("Q3XXS NoLUT requires K % 16 == 0 for codebook packing");
    }
    return (K / 16) * 3;
}

inline int q3xxs_nolut_codebook_bytes(int K, int N) {
    return q3xxs_nolut_codebook_rows(K) * N * (int) sizeof(uint16_t);
}

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

static void rpp_matmul_q3xxs_nolut(rpp_kernel_context & ctx,
                                   int                  M,
                                   int                  K,
                                   int                  N,
                                   int                  weights_group,
                                   int                  in_bytes_per_element,
                                   int                  out_bytes_per_element,
                                   int                  is_instantial = 1) {
    const int q_group                 = 32;
    const int super_group             = 256;
    const int qscale_lut_elems        = 16;
    const int mag_lut_elems           = (int) kQ3xxsNoLutMagValues.size();
    const int mat_lut_codes_per_entry = 4;
    const int mat_lut_rows            = 1 << (3 * mat_lut_codes_per_entry);
    const int mat_lut_cols            = mat_lut_codes_per_entry;
    const int qscale_lut_bytes        = qscale_lut_elems * (int) sizeof(uint16_t);
    const int mag_lut_bytes           = mag_lut_elems * (int) sizeof(uint16_t);
    const int mat_lut_bytes           = mat_lut_rows * mat_lut_cols * (int) sizeof(uint16_t);
    const int lut_workspace_bytes     = qscale_lut_bytes + mag_lut_bytes + mat_lut_bytes;
    if (ctx.dev_in.size() < 5 || ctx.dev_out.empty()) {
        throw std::runtime_error("Q3XXS NoLUT requires 5 input buffers and 1 output buffer");
    }
    if (K % super_group != 0) {
        throw std::runtime_error("Q3XXS NoLUT requires K % 256 == 0");
    }
    if (weights_group != super_group) {
        throw std::runtime_error("Q3XXS NoLUT requires weights_group == 256");
    }
    const char * force_tn1_env = std::getenv("RPP_Q3XXS_NOLUT_FORCE_TN1");
    const bool   force_tn1     = (force_tn1_env != nullptr) && (std::atoi(force_tn1_env) != 0);
    if (force_tn1) {
        // std::cout << "[Q3XXS_NOLUT DBG] RPP_Q3XXS_NOLUT_FORCE_TN1=1, forcing tn1 for all tiles" << std::endl;
    }
    const char * force_ss_bx32_env = std::getenv("RPP_Q3XXS_NOLUT_SS_BX32");
    const bool   force_ss_bx32     = (force_ss_bx32_env != nullptr) && (std::atoi(force_ss_bx32_env) != 0);
    if (force_ss_bx32) {
        // std::cout << "[Q3XXS_NOLUT DBG] RPP_Q3XXS_NOLUT_SS_BX32=1, forcing q3xxs_nolut_super_scale block_x=32"
        //           << std::endl;
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA                = ctx.dev_in[0];
    RPPdeviceptr devB_codebook_nolut = ctx.dev_in[1];
    RPPdeviceptr devB_scales         = ctx.dev_in[2];
    RPPdeviceptr devB_sign           = ctx.dev_in[3];
    RPPdeviceptr devB_super_scale    = ctx.dev_in[4];
    RPPdeviceptr devC                = ctx.dev_out[0];
    RPPdeviceptr dev_lut_workspace   = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        rtMalloc((void **) &dev_lut_workspace, lut_workspace_bytes);
        ctx.dev_workspace = dev_lut_workspace;
    }

    RPPdeviceptr devB_qscale_lut = dev_lut_workspace;
    RPPdeviceptr devB_mag_lut    = devB_qscale_lut + qscale_lut_bytes;
    RPPdeviceptr devB_mat_lut    = devB_mag_lut + mag_lut_bytes;

    std::array<uint16_t, qscale_lut_elems> qscale_lut = {};
    for (int i = 0; i < qscale_lut_elems; ++i) {
        const float scale4  = (float) i;
        const float lut_val = (0.5f + scale4) * 0.5f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }

    std::array<uint16_t, mag_lut_elems> mag_lut = {};
    for (int i = 0; i < mag_lut_elems; ++i) {
        mag_lut[i] = float_to_bf16_rne(kQ3xxsNoLutMagValues[(size_t) i]);
    }

    std::vector<uint16_t> mat_lut((size_t) mat_lut_rows * (size_t) mat_lut_cols, 0);
    for (int packed = 0; packed < mat_lut_rows; ++packed) {
        for (int u = 0; u < mat_lut_cols; ++u) {
            const uint8_t code = (uint8_t) ((packed >> (3 * u)) & 0x7);
            mat_lut[(size_t) packed * (size_t) mat_lut_cols + (size_t) u] =
                float_to_bf16_rne(kQ3xxsNoLutMagValues[(size_t) code]);
        }
    }

    rtMemcpy((void *) devB_qscale_lut, qscale_lut.data(), qscale_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_mag_lut, mag_lut.data(), mag_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_mat_lut, mat_lut.data(), mat_lut_bytes, rtMemcpyHostToDevice);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q3xxs_nolut.o");

    const int sizeA0 = M * K * in_bytes_per_element;
    const int sizeA1 = M * K * (int) sizeof(rpp::bfloat16);

    // q3xxs_nolut packed inputs:
    // codebook_nolut: [K/16 * 3][N]  (uint16, three words per 16 weights)
    // scales:         [K/128][N]     (uint16, packed 4-bit scales)
    // sign:           [K/16][N]      (uint16, packed sign bits)
    // super_scale:    [K/256][N]     (bf16)
    const int codebook_rows        = q3xxs_nolut_codebook_rows(K);
    const int sizeB_codebook_nolut = q3xxs_nolut_codebook_bytes(K, MATMUL_NS);
    const int sizeB_scales         = (K * MATMUL_NS / 64);
    const int sizeB_sign           = (K * MATMUL_NS / 8);
    const int sizeB_super_scale    = (K * MATMUL_NS / super_group) * (int) sizeof(short);
    const int sizeB_qscale_lut     = qscale_lut_bytes;
    const int sizeB_mag_lut        = mag_lut_bytes;
    const int sizeB_mat_lut        = mat_lut_bytes;

    // q3xxs dequant consumes one bf16 scale per 32 weights.
    const int sizeB_scale = (K * MATMUL_NS / q_group) * (int) sizeof(short);

    const int sizeB   = (K * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeCr  = (M * MATMUL_NS) * (int) sizeof(rpp::bfloat16);
    const int sizeC32 = (M * MATMUL_NS) * (int) sizeof(float);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);

    RPPdeviceptr sramB_codebook_nolut = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_scales         = sramB_codebook_nolut + round_up(sizeB_codebook_nolut);
    RPPdeviceptr sramB_sign           = sramB_scales + round_up(sizeB_scales);
    RPPdeviceptr sramB_super_scale    = sramB_sign + round_up(sizeB_sign);
    RPPdeviceptr sramB_qscale_lut     = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_mag_lut        = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_mat_lut        = sramB_mag_lut + round_up(sizeB_mag_lut);
    RPPdeviceptr sramB_scale          = sramB_mat_lut + round_up(sizeB_mat_lut);
    RPPdeviceptr sramB_out            = sramB_scale + round_up(sizeB_scale);
    RPPdeviceptr sramC_hw32           = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw            = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32       = sramC_chw + round_up(sizeCr);

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
    rtMemcpyAsync((void *) sramB_mag_lut, (const void *) devB_mag_lut, sizeB_mag_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_mat_lut, (const void *) devB_mat_lut, sizeB_mat_lut, rtMemcpyDeviceToSram,
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
    // Otherwise source is bf16 in sramA_in and destination is sramA_out.
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
            // std::cout << "[Q3XXS_NOLUT DBG] tile ns=" << ns << " tn=" << tn << " kernel=" << matmul_kernel << std::endl;
        }

        rtMemcpy2DAsync((void *) sramB_codebook_nolut, ns * (int) sizeof(short), (const void *) devB_codebook_nolut,
                        N * (int) sizeof(short), ns * (int) sizeof(short), codebook_rows, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_scales, ns * (int) sizeof(short), (const void *) devB_scales,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 128, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_sign, ns * (int) sizeof(short), (const void *) devB_sign,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / 16, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_super_scale, ns * (int) sizeof(short), (const void *) devB_super_scale,
                        N * (int) sizeof(short), ns * (int) sizeof(short), K / super_group, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        params.clear();
        q3xxs_nolut_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
        if (force_ss_bx32 && (ns % 32 == 0)) {
            threadsPerBlock.x = 32;
            blocksPerGrid.x   = (uint32_t) (ns / 32);
            if (i == 0) {
                // std::cout << "[Q3XXS_NOLUT DBG] super_scale override: block_x=" << threadsPerBlock.x
                //           << " grid_x=" << blocksPerGrid.x << std::endl;
            }
        }
        q3xxs_nolut_super_scale_params(sramB_scales, sramB_super_scale, sramB_qscale_lut, sramB_scale, K, ns,
                                       super_group, q_group, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        q3xxs_nolut_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
        q3xxs_nolut_dequant_params(blocksPerGrid, threadsPerBlock, sramB_codebook_nolut, sramB_sign, sramB_scale,
                                   sramB_out, sramB_mag_lut, ns, K, (int) sizeof(short), (int) sizeof(short), params);
        launchWrapperAysnc("matrix_mul_q3xxs_deqaunt_nolut_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
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
            calc_tbdim_flattern(1, M * 2 * ns, threadsPerBlock, blocksPerGrid);
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

        devB_codebook_nolut += ns * (int) sizeof(short);
        devB_scales += ns * (int) sizeof(short);
        devB_sign += ns * (int) sizeof(short);
        devB_super_scale += ns * (int) sizeof(short);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q3xxs_nolut_sram(rpp_kernel_context & ctx,
                                        int                  M,
                                        int                  K,
                                        int                  N,
                                        int                  weights_group,
                                        int                  in_bytes_per_element,
                                        int                  out_bytes_per_element,
                                        int                  is_instantial = 1) {
    const int q_group     = 32;
    const int super_group = 256;
    if (weights_group != super_group) {
        throw std::runtime_error("Q3XXS NoLUT SRAM requires weights_group == 256");
    }

    const char * force_tn1_env = std::getenv("RPP_Q3XXS_NOLUT_FORCE_TN1");
    const bool   force_tn1     = (force_tn1_env != nullptr) && (std::atoi(force_tn1_env) != 0);
    if (force_tn1) {
        // std::cout << "[Q3XXS_NOLUT DBG] RPP_Q3XXS_NOLUT_FORCE_TN1=1, forcing tn1 for SRAM-direct full-N launch"
        //           << std::endl;
    }
    const char * force_ss_bx32_env = std::getenv("RPP_Q3XXS_NOLUT_SS_BX32");
    const bool   force_ss_bx32     = (force_ss_bx32_env != nullptr) && (std::atoi(force_ss_bx32_env) != 0);
    if (force_ss_bx32) {
        // std::cout << "[Q3XXS_NOLUT DBG] RPP_Q3XXS_NOLUT_SS_BX32=1, forcing q3xxs_nolut_super_scale block_x=32"
        //           << std::endl;
    }

    const q3xxs_nolut_sram_io io =
        q3xxs_nolut_prepare_sram_io(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element);

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q3xxs_nolut.o");

    const bool         a_is_f32 = (in_bytes_per_element == (int) sizeof(float));
    const RPPdeviceptr matmul_A = a_is_f32 ? io.sramA_in : io.sramA_out;
    const RPPdeviceptr matmul_B = io.sramB_out;
    const RPPdeviceptr matmul_C = io.sramC_hw32;

    if (a_is_f32) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, io.sramA_in, io.sramA_out, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    params.clear();
    chw2chw32_blocks(1, M, K, threadsPerBlock, blocksPerGrid);
    chw2chw32_align_params((int) (a_is_f32 ? io.sramA_out : io.sramA_in), (int) (a_is_f32 ? io.sramA_in : io.sramA_out),
                           M, K, 0, (int) threadsPerBlock.x, (int) threadsPerBlock.y, (int) threadsPerBlock.z,
                           (int) sizeof(rpp::bfloat16), params, false);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    const int         ns            = N;
    const int         tn            = force_tn1 ? 1 : get_tn(ns);
    const std::string matmul_kernel = force_tn1 ? "matmul_tn1_f16_f32_f16" : get_matmul_kernel(ns);
    // std::cout << "[Q3XXS_NOLUT DBG] SRAM-direct full-N ns=" << ns << " tn=" << tn << " kernel=" << matmul_kernel
    //           << std::endl;

    params.clear();
    q3xxs_nolut_super_scale_blocks(K, super_group, q_group, ns, threadsPerBlock, blocksPerGrid);
    if (force_ss_bx32 && (ns % 32) == 0) {
        threadsPerBlock.x = 32;
        blocksPerGrid.x   = (uint32_t) (ns / 32);
        // std::cout << "[Q3XXS_NOLUT DBG] SRAM-direct super_scale override: block_x=" << threadsPerBlock.x
        //           << " grid_x=" << blocksPerGrid.x << std::endl;
    }
    q3xxs_nolut_super_scale_params((uint32_t) io.sramB_scales, (uint32_t) io.sramB_super_scale,
                                   (uint32_t) io.sramB_qscale_lut, (uint32_t) io.sramB_scale, K, ns, super_group,
                                   q_group, blocksPerGrid, threadsPerBlock, params);
    launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    params.clear();
    q3xxs_nolut_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
    q3xxs_nolut_dequant_params(blocksPerGrid, threadsPerBlock, (uint32_t) io.sramB_codebook_nolut,
                               (uint32_t) io.sramB_sign, (uint32_t) io.sramB_scale, (uint32_t) io.sramB_out,
                               (uint32_t) io.sramB_mag_lut, ns, K, (int) sizeof(short), (int) sizeof(short), params);
    launchWrapperAysnc("matrix_mul_q3xxs_deqaunt_nolut_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    params.clear();
    matmul_chw32_blocks(1, M, ns, threadsPerBlock, blocksPerGrid, tn, 0);
    matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                             (uint32_t) matmul_C, M, K, K, ns, 1, 1, 1, 0, tn, (int) sizeof(rpp::bfloat16),
                             (int) sizeof(rpp::bfloat16), params, false, false);
    launchWrapperAysnc(matmul_kernel, blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);

    params.clear();
    chw322chw_blocks(1, M, ns, threadsPerBlock, blocksPerGrid);
    chw322chw_align_params((int) matmul_C, (int) io.sramC_chw, M, ns, 0, threadsPerBlock.x, threadsPerBlock.y,
                           threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, M * 2 * ns, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, io.sramC_chw, io.sramC_chw_fp32, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q3xxs_nolut_build(rpp_kernel_context & ctx,
                                         int                  M,
                                         int                  K,
                                         int                  N,
                                         int                  weights_group,
                                         int                  in_bytes_per_element,
                                         int                  out_bytes_per_element,
                                         int                  use_sram_direct = 0,
                                         int                  is_instantial   = 1) {
    if (!use_sram_direct) {
        rpp_matmul_q3xxs_nolut(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    } else {
        rpp_matmul_q3xxs_nolut_sram(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                    is_instantial);
    }
}

}  // namespace kernel_q3_xxs_nolut
