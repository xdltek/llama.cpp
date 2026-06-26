#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q8_0/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q8_0/rpp_kernel_param.h"

#include <assert.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_q8_0 {
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
    std::string kernel_name;
    if ((N % 128) == 0) {
        kernel_name = "matmul_tn4_f16_f32_f16";
    } else if ((N % 96) == 0) {
        kernel_name = "matmul_tn3_f16_f32_f16";
    } else if ((N % 64) == 0) {
        kernel_name = "matmul_tn2_f16_f32_f16";
    } else {
        kernel_name = "matmul_tn1_f16_f32_f16";
    }
    return kernel_name;
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_q80(rpp_kernel_context & ctx,
                           int                  M,
                           int                  K,
                           int                  N,
                           int                  weights_group,
                           int                  in_bytes_per_element,
                           int                  out_bytes_per_element,
                           int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    int                   full_pitch, sub_pitch, offset;
    RPPdeviceptr          devA       = ctx.dev_in[0];
    RPPdeviceptr          devB_q8    = ctx.dev_in[1];
    RPPdeviceptr          devB_scale = ctx.dev_in[2];
    // RPPdeviceptr          devB_zero  = ctx.dev_in[3];
    RPPdeviceptr          devC       = ctx.dev_out[0];

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q80.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int sizeA0 = (int) (M) * (K) *in_bytes_per_element;
    const int sizeA1 = (int) (M) * (K) * sizeof(rpp::bfloat16);

    const int sizeB_q8  = (int) ((K) * (MATMUL_NS) * sizeof(int8_t));
    const int sizeScale = (int) ((K) * (MATMUL_NS) / weights_group * sizeof(rpp::bfloat16));
    const int sizeZero  = (int) ((K) * (MATMUL_NS) / weights_group * sizeof(rpp::bfloat16));
    const int sizeLut   = (int) (32);
    const int sizeB     = (int) ((K) * (MATMUL_NS) * sizeof(rpp::bfloat16));
    const int sizeCr    = (int) ((M) * (MATMUL_NS) * sizeof(rpp::bfloat16));
    const int sizeC32   = (int) ((M) * (MATMUL_NS) * sizeof(float));

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    // Linear in + HW32 out for A/B
    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);

    RPPdeviceptr sramB_q8    = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_scale = sramB_q8 + round_up(sizeB_q8);
    RPPdeviceptr sramB_zero  = sramB_scale + round_up(sizeScale);
    RPPdeviceptr sramB_lut   = sramB_zero + round_up(sizeZero);

    RPPdeviceptr sramB_out  = sramB_lut + round_up(sizeLut);
    // Dedicated HW32 output
    RPPdeviceptr sramC_hw32 = sramB_out + round_up(sizeB);

    // Reuse a buffer as final CHW output (linear) before DDR copy:
    // (You can dedicate another buffer if you prefer; this is SRAM-minimal.)
    RPPdeviceptr sramC_chw      = sramC_hw32 + round_up(sizeCr);  // reuse Ar area after matmul if safe in your pipeline
    RPPdeviceptr sramC_chw_fp32 = sramC_chw + round_up(sizeCr);   // reuse Ar area after matmul if safe in your pipeline

    const int total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    // Matmul uses HW32 outputs as inputs
    RPPdeviceptr matmul_A = sramA_out;
    RPPdeviceptr matmul_B = sramB_out;
    RPPdeviceptr matmul_C = sramC_hw32;

    // -------------------------
    // (1) DDR -> SRAM async copies on dmaStream
    // -------------------------
    rtMemcpyAsync((void *) sramA_in, (const void *) devA, sizeA0, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA_in, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    // -------------------------
    //with given split factor, launch kernel one by one
    // -------------------------

    // -------------------------
    // (2) Linear -> HW32 for A
    // -------------------------
    params.clear();
    chw2chw32_blocks(1, M, K, threadsPerBlock, blocksPerGrid);
    chw2chw32_align_params((int) sramA_in, (int) sramA_out, M, K, 0, (int) threadsPerBlock.x, (int) threadsPerBlock.y,
                           (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params, false);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);
    for (int i = 0; i < get_tiles(N); i++) {
        bool is_tail = (i == (get_tiles(N) - 1));
        int  ns      = get_ns(N, is_tail);
        rtMemcpy2DAsync((void *) sramB_q8, get_ns(N, is_tail) * sizeof(short), (const void *) devB_q8,
                        N * sizeof(short), get_ns(N, is_tail) * sizeof(short), K / 2, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramB_scale, get_ns(N, is_tail) * sizeof(short), (const void *) devB_scale,
                        N * sizeof(short), get_ns(N, is_tail) * sizeof(short), K / weights_group, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        //rtMemcpy2DAsync((void *)sramB_zero, get_ns(N, is_tail) * sizeof(short),
        //    (const void *)devB_zero, N * sizeof(short),
        //    get_ns(N, is_tail) * sizeof(short), K / weights_group, rtMemcpyDeviceToSram, ctx.kernelStream);

        // -------------------------
        // (3) dequant B from INT4 to Bfloat16
        // -------------------------
        params.clear();
        matmul_q80_dequant_blocks(1, K, get_ns(N, is_tail), threadsPerBlock, blocksPerGrid, weights_group);

        matmul_dequant_params(sramB_q8, sramB_scale, sramB_zero, sramB_out, sramB_lut, get_ns(N, is_tail), K,
                              threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, blocksPerGrid.x, blocksPerGrid.y,
                              blocksPerGrid.z, sizeof(short), sizeof(short), params);
        blocksPerGrid.x = 1;
        blocksPerGrid.y = 1;

        launchWrapperAysnc("matrix_mul_i8_deqaunt_f16_no_lut", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        // -------------------------
        // (4) Matmul HW32
        // -------------------------
        params.clear();

        matmul_chw32_blocks(1, M, get_ns(N, is_tail), threadsPerBlock, blocksPerGrid, get_tn(get_ns(N, is_tail)), 0);

        int         tn        = get_tn(get_ns(N, is_tail));
        std::string kern_name = get_matmul_kernel(get_ns(N, is_tail));
        matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                                 (uint32_t) matmul_C, M, K, K, get_ns(N, is_tail), 1, 1, 1, 0,
                                 get_tn(get_ns(N, is_tail)), (int) sizeof(rpp::bfloat16), (int) sizeof(rpp::bfloat16),
                                 params, false, false);

        launchWrapperAysnc(get_matmul_kernel(get_ns(N, is_tail)), blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        // -------------------------
        // (5) HW32 -> CHW (linear) into sramC_chw
        // -------------------------
        params.clear();
        chw322chw_blocks(1, M, get_ns(N, is_tail), threadsPerBlock, blocksPerGrid);
        chw322chw_align_params((int) matmul_C, (int) sramC_chw, M, get_ns(N, is_tail), 0, threadsPerBlock.x,
                               threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
        launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                           ctx.rppBinMod, ctx.kernelStream);

        if (out_bytes_per_element == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(M,ns * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpy2DAsync((void *) devC, N * sizeof(float), (void *) sramC_chw_fp32,
                            get_ns(N, is_tail) * sizeof(float), get_ns(N, is_tail) * sizeof(float), M,
                            rtMemcpySramToDevice, ctx.kernelStream);

            devC += get_ns(N, is_tail) * sizeof(float);
        } else {
            rtMemcpy2DAsync((void *) devC, N * sizeof(short), (void *) sramC_chw, get_ns(N, is_tail) * sizeof(short),
                            get_ns(N, is_tail) * sizeof(short), M, rtMemcpySramToDevice, ctx.kernelStream);
            devC += get_ns(N, is_tail) * sizeof(short);
        }

        devB_q8 += get_ns(N, is_tail) * sizeof(short);
        devB_scale += get_ns(N, is_tail) * sizeof(short);
        //devB_zero += get_ns(N, is_tail) * sizeof(short);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q80_pipeline(rpp_kernel_context & ctx,
                                    int                  M,
                                    int                  K,
                                    int                  N,
                                    int                  weights_group,
                                    int                  in_bytes_per_element,
                                    int                  out_bytes_per_element,
                                    int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    int                   full_pitch, sub_pitch, offset;
    RPPdeviceptr          devA       = ctx.dev_in[0];
    RPPdeviceptr          devB_q8    = ctx.dev_in[1];
    RPPdeviceptr          devB_scale = ctx.dev_in[2];
    // RPPdeviceptr          devB_zero  = ctx.dev_in[3];
    RPPdeviceptr          devC       = ctx.dev_out[0];

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q80.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int sizeA0 = (int) (M) * (K) * in_bytes_per_element;
    const int sizeA1 = (int) (M) * (K) * sizeof(rpp::bfloat16);

    const int sizeB_q8  = (int) ((K) * (MATMUL_NS) * sizeof(int8_t));
    const int sizeScale = (int) ((K) * (MATMUL_NS) / weights_group * sizeof(rpp::bfloat16));
    const int sizeZero  = (int) ((K) * (MATMUL_NS) / weights_group * sizeof(rpp::bfloat16));
    const int sizeLut   = (int) (32);
    const int sizeB     = (int) ((K) * (MATMUL_NS) * sizeof(rpp::bfloat16));
    const int sizeCr    = (int) ((M) * (MATMUL_NS) * sizeof(rpp::bfloat16));
    const int sizeC32   = (int) ((M) * (MATMUL_NS) * sizeof(float));

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    // Linear in + HW32 out for A/B
    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA_out = sramA_in + round_up(sizeA0);
    // --- ping-pong buffer for B weights ---
    RPPdeviceptr sramB_q8_0    = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramB_scale_0 = sramB_q8_0 + round_up(sizeB_q8);
    RPPdeviceptr sramB_q8_1    = sramB_scale_0 + round_up(sizeScale);
    RPPdeviceptr sramB_scale_1 = sramB_q8_1 + round_up(sizeB_q8);

    RPPdeviceptr sramB_zero  = sramB_scale_1 + round_up(sizeScale);
    RPPdeviceptr sramB_lut   = sramB_zero + round_up(sizeZero);

    RPPdeviceptr sramB_out  = sramB_lut + round_up(sizeLut);
    // Dedicated HW32 output
    RPPdeviceptr sramC_hw32 = sramB_out + round_up(sizeB);

    // Reuse a buffer as final CHW output (linear) before DDR copy:
    // (You can dedicate another buffer if you prefer; this is SRAM-minimal.)
    RPPdeviceptr sramC_chw      = sramC_hw32 + round_up(sizeCr);  // reuse Ar area after matmul if safe in your pipeline
    RPPdeviceptr sramC_chw_fp32 = sramC_chw + round_up(sizeCr);   // reuse Ar area after matmul if safe in your pipeline

    const int total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    // Matmul uses HW32 outputs as inputs
    RPPdeviceptr matmul_A = sramA_out;
    RPPdeviceptr matmul_B = sramB_out;
    RPPdeviceptr matmul_C = sramC_hw32;

    // -------------------------
    // (1) DDR -> SRAM async copies on dmaStream
    // -------------------------
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);
    rtMemcpyAsync((void *) sramA_in, (const void *) devA, sizeA0, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA_in, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    // -------------------------
    // (2) Linear -> HW32 for A
    // -------------------------
    params.clear();
    chw2chw32_blocks(1, M, K, threadsPerBlock, blocksPerGrid);
    chw2chw32_align_params((int) sramA_in, (int) sramA_out, M, K, 0, (int) threadsPerBlock.x, (int) threadsPerBlock.y,
                           (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params, false);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    auto sramB_q8 = [&](int ping) {
        return ping ? sramB_q8_1 : sramB_q8_0;
    };
    auto sramB_scale = [&](int ping) {
        return ping ? sramB_scale_1 : sramB_scale_0;
    };

    auto schedule_dma = [&](int ping, int NsSeg) {
        // Ensure previous kernel using this ping buffer has finished
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);
        
        rtMemcpy2DAsync((void *) sramB_q8(ping), NsSeg * sizeof(short), (const void *) devB_q8,
                        N * sizeof(short), NsSeg * sizeof(short), K / 2, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        rtMemcpy2DAsync((void *) sramB_scale(ping), NsSeg * sizeof(short), (const void *) devB_scale,
                        N * sizeof(short), NsSeg * sizeof(short), K / weights_group, rtMemcpyDeviceToSram,
                        ctx.dmaStream);

        // --- update data addr ---
        devB_q8 += NsSeg * sizeof(short);
        devB_scale += NsSeg * sizeof(short);

        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    const int nr_of_tiles = get_tiles(N);

    // --- prefetch data for tile 0 ---
    {
        int ping = 0;
        bool is_tail = (0 == (nr_of_tiles - 1));
        int  ns      = get_ns(N, is_tail);
        schedule_dma(ping, ns);
    }

    for (int i = 0; i < nr_of_tiles; ++ i) {
        const bool is_tail = (i == (nr_of_tiles - 1));
        const int  NsSeg   = get_ns(N, is_tail);
        const int  ping    = (i & 1);

        // --- wait for current DMA (ping-specific) ---
        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

        // --- prefetch next segment (ping-specific) ---
        if (i + 1 < nr_of_tiles) {
            const bool next_is_tail    = ((i + 1) == (nr_of_tiles - 1));
            const int  next_NsSeg      = get_ns(N, next_is_tail);
            const int  next_ping       = ((i + 1) & 1);
            schedule_dma(next_ping, next_NsSeg);
        }
        // -------------------------
        // (3) dequant B from INT4 to Bfloat16
        // -------------------------
        params.clear();
        matmul_q80_dequant_blocks(1, K, NsSeg, threadsPerBlock, blocksPerGrid, weights_group);

        matmul_dequant_params(sramB_q8(ping), sramB_scale(ping), sramB_zero, sramB_out, sramB_lut, NsSeg, K,
                              threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, blocksPerGrid.x, blocksPerGrid.y,
                              blocksPerGrid.z, sizeof(short), sizeof(short), params);
        blocksPerGrid.x = 1;
        blocksPerGrid.y = 1;

        launchWrapperAysnc("matrix_mul_i8_deqaunt_f16_no_lut", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        // -------------------------
        // (4) Matmul HW32
        // -------------------------
        params.clear();
        int         tn        = get_tn(NsSeg);
        std::string kern_name = get_matmul_kernel(NsSeg);
        matmul_chw32_blocks(1, M, NsSeg, threadsPerBlock, blocksPerGrid, tn, 0);
        matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                                 (uint32_t) matmul_C, M, K, K, NsSeg, 1, 1, 1, 0, tn, (int) sizeof(rpp::bfloat16), (int) sizeof(rpp::bfloat16),
                                 params, false, false);
        launchWrapperAysnc(kern_name, blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        // -------------------------
        // (5) HW32 -> CHW (linear) into sramC_chw
        // -------------------------
        params.clear();
        chw322chw_blocks(1, M, NsSeg, threadsPerBlock, blocksPerGrid);
        chw322chw_align_params((int) matmul_C, (int) sramC_chw, M, NsSeg, 0, threadsPerBlock.x,
                               threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
        launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                           ctx.rppBinMod, ctx.kernelStream);

        if (out_bytes_per_element == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(M, NsSeg * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpy2DAsync((void *) devC, N * sizeof(float), (void *) sramC_chw_fp32,
                            NsSeg * sizeof(float), NsSeg * sizeof(float), M,
                            rtMemcpySramToDevice, ctx.kernelStream);

            devC += NsSeg * sizeof(float);
        } else {
            rtMemcpy2DAsync((void *) devC, N * sizeof(short), (void *) sramC_chw, NsSeg * sizeof(short),
                            NsSeg * sizeof(short), M, rtMemcpySramToDevice, ctx.kernelStream);
            devC += NsSeg * sizeof(short);
        }
        // Signal this ping buffer is safe to reuse
        rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q80_build(rpp_kernel_context & ctx,
                                 int                  M,
                                 int                  K,
                                 int                  N,
                                 int                  weights_group,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  use_pipeline = 0,
                                 int                  is_instantial = 1) {
    // if (use_pipeline) {
    //     rpp_matmul_q80_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    // } else {
    //     rpp_matmul_q80(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    // }
    rpp_matmul_q80(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
}
}  // namespace kernel_q8_0
