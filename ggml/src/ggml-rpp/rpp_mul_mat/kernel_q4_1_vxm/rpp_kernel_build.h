
#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q4_1_vxm/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q4_1_vxm/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_q4_1_vxm {
inline void get_tiles_info(int N, int K, int weights_group, int & nr_of_tiles, int & groups_per_tile) {
    if (weights_group == 32) {
        if (K % 512 == 0) {
            nr_of_tiles     = K / 512;
            groups_per_tile = 16;
        } else if (K % 256 == 0) {
            nr_of_tiles     = K / 256;
            groups_per_tile = 8;
        } else if (K % 128 == 0) {
            nr_of_tiles     = K / 128;
            groups_per_tile = 4;
        } else if (K % 64 == 0) {
            nr_of_tiles     = K / 64;
            groups_per_tile = 2;
        } else {
            nr_of_tiles     = K / 32;
            groups_per_tile = 1;
        }
        if (N >= 32768 * 4) {
            nr_of_tiles     = nr_of_tiles * 8;
            groups_per_tile = groups_per_tile / 8;
        } else if (N >= 32768 * 2) {
            nr_of_tiles     = nr_of_tiles * 4;
            groups_per_tile = groups_per_tile / 4;
        } else if (N >= 32768) {
            nr_of_tiles     = nr_of_tiles * 2;
            groups_per_tile = groups_per_tile / 2;
        }
    } else {
        nr_of_tiles     = K / weights_group;
        groups_per_tile = 1;
    }
    assert(K == nr_of_tiles * weights_group * groups_per_tile);
}

static void rpp_matmul_q4_1_vxm(rpp_kernel_context & ctx,
                                int                  M,
                                int                  K,
                                int                  N,
                                int                  weights_group,
                                int                  in_bytes_per_element,
                                int                  out_bytes_per_element,
                                int                  is_instantial = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q4 Configure Paramter Invalid");
    }
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    int                   full_pitch, sub_pitch, offset;
    RPPdeviceptr          devA       = ctx.dev_in[0];
    RPPdeviceptr          devB_q4    = ctx.dev_in[1];
    RPPdeviceptr          devB_scale = ctx.dev_in[2];
    RPPdeviceptr          devB_zero  = ctx.dev_in[3];
    RPPdeviceptr          devC       = ctx.dev_out[0];

    // build exp table
    int             lut_elements = 16;
    rpp::bfloat16 * quant_table  = (rpp::bfloat16 *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < lut_elements; i++) {
        quant_table[i] = (rpp::bfloat16) i;
    }
    RPPdeviceptr dev_lut = ctx.dev_workspace;
    rtMemcpy((void *) dev_lut, (const void *) quant_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q4_vxm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    int nr_of_tiles, groups_per_tile;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile);

    const int sizeA     = (K) *in_bytes_per_element;
    // weights_group
    const int sizeB_q4  = groups_per_tile * weights_group * N / 2 * sizeof(int8_t);
    const int sizeScale = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeZero  = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeLut   = lut_elements * sizeof(rpp::bfloat16);
    const int sizeC     = N * out_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA     = sram_base;
    RPPdeviceptr sramC     = sramA + round_up(sizeA);
    RPPdeviceptr sramC1    = sramC + round_up(sizeC);

    RPPdeviceptr sramB_q4    = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_scale = sramB_q4 + round_up(sizeB_q4);
    RPPdeviceptr sramB_zero  = sramB_scale + round_up(sizeScale);
    RPPdeviceptr sramB_lut   = sramB_zero + round_up(sizeZero);
    RPPdeviceptr sramA_acc   = sramB_lut + round_up(sizeLut);

    const int total_sram_bytes = (int) (sramA_acc + round_up(weights_group * 4) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    // -------------------------
    // (1) DDR -> SRAM async copies on dmaStream
    // -------------------------
    rtMemcpyAsync((void *) sramA, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_lut, (const void *) dev_lut, sizeLut, rtMemcpyDeviceToSram, ctx.kernelStream);
    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    int Nx = K / weights_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    int            hilo_offset = (Nx * Nz + 31) / 32 * 32 * sizeof(short);
    int            combine;
    RppTaskElement task_in_acc;
    task_in_acc.taskName   = "asym_quant_input_acc";
    task_in_acc.blockDim.x = Nx;
    task_in_acc.blockDim.y = 1;
    task_in_acc.blockDim.z = Nz;
    task_in_acc.gridDim.x  = 1;
    task_in_acc.gridDim.y  = 1;
    task_in_acc.gridDim.z  = 1;
    task_in_acc.params.kernelList.clear();
    task_in_acc.params.kernelList.emplace_back(sramA);
    task_in_acc.params.kernelList.emplace_back(sramA_acc);
    task_in_acc.params.kernelList.emplace_back(sramA_acc + hilo_offset);
    task_in_acc.params.kernelList.emplace_back(weights_group);
    launchWrapperAysnc(task_in_acc.taskName, task_in_acc.gridDim, task_in_acc.blockDim, task_in_acc.params.kernelList,
                       ctx.rppBinMod, ctx.kernelStream);

    // -------------------------
    // with given split factor, launch kernel one by one
    // -------------------------
    for (int i = 0; i < nr_of_tiles; i++) {
        rtMemcpyAsync((void *) sramB_q4, (const void *) devB_q4, sizeB_q4, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpyAsync((void *) sramB_scale, (const void *) devB_scale, sizeScale, rtMemcpyDeviceToSram,
                      ctx.kernelStream);
        rtMemcpyAsync((void *) sramB_zero, (const void *) devB_zero, sizeZero, rtMemcpyDeviceToSram, ctx.kernelStream);

        devB_q4 += sizeB_q4;
        devB_scale += sizeScale;
        devB_zero += sizeZero;

        if (i == 0) {
            combine = 0;
        } else {
            combine = 1;
        }

        RppTaskElement task;
        uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
        kerneldim_calc_matmul_linear(1, M, N, block_x, block_y, block_z, grid_x, grid_y, grid_z);
        task.blockDim.x = block_x;
        task.blockDim.y = block_y;
        task.blockDim.z = block_z;
        task.gridDim.x  = grid_x;
        task.gridDim.y  = grid_y;
        task.gridDim.z  = grid_z;

        task.gridDim.z          = groups_per_tile;
        uint32_t stride_ina     = weights_group * sizeof(short);
        uint32_t stride_inb     = weights_group * N / 2;
        uint32_t stride_scale   = N * sizeof(short);
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * sizeof(short);
        task.taskName           = "matrix_mul_vxM_f16_i4_f16_asym_opt";
        task.params.kernelList.clear();
        matmul_weights_i4_kernel_params(sramA + i * stride_ina * groups_per_tile, sramB_q4, sramB_scale, sramC,
                                        sramB_lut, sramB_zero, input_acc_addr, input_acc_addr + hilo_offset, 1, 1, N,
                                        weights_group, combine, stride_ina, stride_inb, stride_scale,
                                        task.params.kernelList);

        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);

        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpyAsync((void *) devC, (const void *) sramC1, sizeC, rtMemcpySramToDevice, ctx.kernelStream);

    } else {
        rtMemcpyAsync((void *) devC, (const void *) sramC, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }

    free(quant_table);
}

static void rpp_matmul_q4_1_vxm_pipeline(rpp_kernel_context & ctx,
                                         int                  M,
                                         int                  K,
                                         int                  N,
                                         int                  weights_group,
                                         int                  in_bytes_per_element,
                                         int                  out_bytes_per_element,
                                         int                  is_instantial = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q4 Configure Parameter Invalid");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA       = ctx.dev_in[0];
    RPPdeviceptr devB_q4    = ctx.dev_in[1];
    RPPdeviceptr devB_scale = ctx.dev_in[2];
    RPPdeviceptr devB_zero  = ctx.dev_in[3];
    RPPdeviceptr devC       = ctx.dev_out[0];

    // -------------------------
    // LUT build
    // -------------------------
    const int       lut_elements = 16;
    rpp::bfloat16 * quant_table  = (rpp::bfloat16 *) malloc(lut_elements * sizeof(uint16_t));
    for (int i = 0; i < lut_elements; ++i) {
        quant_table[i] = (rpp::bfloat16) i;
    }

    RPPdeviceptr dev_lut = ctx.dev_workspace;
    rtMemcpy((void *) dev_lut, (const void *) quant_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);

    // Load module
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q4_vxm.o");

    // -------------------------
    // SRAM allocation planning
    // -------------------------
    int nr_of_tiles, groups_per_tile;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile);

    const int sizeA = (K) *in_bytes_per_element;

    // weights for one tile (groups_per_tile * weights_group portion)
    const int sizeB_q4  = groups_per_tile * weights_group * N / 2 * sizeof(int8_t);
    const int sizeScale = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeZero  = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeLut   = lut_elements * sizeof(rpp::bfloat16);
    const int sizeC     = N * out_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA  = sram_base;
    RPPdeviceptr sramC  = sramA + round_up(sizeA);
    RPPdeviceptr sramC1 = sramC + round_up(sizeC);

    // --- ping-pong buffers for B weights ---
    RPPdeviceptr sramB_q4_0    = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_scale_0 = sramB_q4_0 + round_up(sizeB_q4);
    RPPdeviceptr sramB_zero_0  = sramB_scale_0 + round_up(sizeScale);

    RPPdeviceptr sramB_q4_1    = sramB_zero_0 + round_up(sizeZero);
    RPPdeviceptr sramB_scale_1 = sramB_q4_1 + round_up(sizeB_q4);
    RPPdeviceptr sramB_zero_1  = sramB_scale_1 + round_up(sizeScale);

    RPPdeviceptr sramB_lut = sramB_zero_1 + round_up(sizeZero);
    RPPdeviceptr sramA_acc = sramB_lut + round_up(sizeLut);

    const int total_sram_bytes = (int) (sramA_acc + round_up(weights_group * 4) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    // -------------------------
    // Capture (optional)
    // -------------------------
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    // Mark ping buffers as free at graph start
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);
    // -------------------------
    // (A) Preload A and LUT (DMA stream)
    // -------------------------
    rtMemcpyAsync((void *) sramA, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_lut, (const void *) dev_lut, sizeLut, rtMemcpyDeviceToSram, ctx.kernelStream);

    // If input is fp32, convert vector to bf16 in-place in SRAM (kernel stream)
    if (in_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    // input asym quant acc (kernel stream)
    {
        int Nx = K / weights_group;
        int Nz = 1;
        while (Nx * Nz <= 32) {
            Nz++;
        }
        int hilo_offset = (Nx * Nz + 31) / 32 * 32 * sizeof(short);

        RppTaskElement task_in_acc;
        task_in_acc.taskName   = "asym_quant_input_acc";
        task_in_acc.blockDim.x = Nx;
        task_in_acc.blockDim.y = 1;
        task_in_acc.blockDim.z = Nz;
        task_in_acc.gridDim.x  = 1;
        task_in_acc.gridDim.y  = 1;
        task_in_acc.gridDim.z  = 1;

        task_in_acc.params.kernelList.clear();
        task_in_acc.params.kernelList.emplace_back(sramA);
        task_in_acc.params.kernelList.emplace_back(sramA_acc);
        task_in_acc.params.kernelList.emplace_back(sramA_acc + hilo_offset);
        task_in_acc.params.kernelList.emplace_back(weights_group);

        launchWrapperAysnc(task_in_acc.taskName, task_in_acc.gridDim, task_in_acc.blockDim,
                           task_in_acc.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
    }

    // -------------------------
    // (B) Tile loop with ping-pong:
    //     DMA(tile i) on dmaStream while KERNEL(tile i-1) on kernelStream
    // -------------------------

    auto sramB_q4 = [&](int ping) {
        return ping ? sramB_q4_1 : sramB_q4_0;
    };
    auto sramB_scale = [&](int ping) {
        return ping ? sramB_scale_1 : sramB_scale_0;
    };
    auto sramB_zero = [&](int ping) {
        return ping ? sramB_zero_1 : sramB_zero_0;
    };

    // per-tile strides in DDR
    const uint32_t stride_ina   = weights_group * sizeof(short);
    const uint32_t stride_inb   = weights_group * N / 2;
    const uint32_t stride_scale = N * sizeof(short);

    // for matmul kernel offsets
    int Nx = K / weights_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    const int hilo_offset = (Nx * Nz + 31) / 32 * 32 * sizeof(short);
#if 1
    auto schedule_dma = [&](int ping) {
        // Ensure previous kernel using this ping buffer has finished
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

        rtMemcpyAsync((void *) sramB_q4(ping), (const void *) devB_q4, sizeB_q4, rtMemcpyDeviceToSram, ctx.dmaStream);
        rtMemcpyAsync((void *) sramB_scale(ping), (const void *) devB_scale, sizeScale, rtMemcpyDeviceToSram,
                      ctx.dmaStream);
        rtMemcpyAsync((void *) sramB_zero(ping), (const void *) devB_zero, sizeZero, rtMemcpyDeviceToSram,
                      ctx.dmaStream);

        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    // ---- prefetch tile 0 ----
    {
        int ping = 0;
        schedule_dma(ping);
    }

    // advance DDR pointers after scheduling tile0 DMA
    devB_q4 += sizeB_q4;
    devB_scale += sizeScale;
    devB_zero += sizeZero;

    for (int i = 0; i < nr_of_tiles; ++i) {
        const int ping = (i & 1);

        // wait for tile i DMA (ping-specific)
        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

        // ---- schedule prefetch for tile (i+1) ASAP on dmaStream ----
        if (i + 1 < nr_of_tiles) {
            const int next_ping = ((i + 1) & 1);

            schedule_dma(next_ping);

            // update DDR pointers for next prefetch
            devB_q4 += sizeB_q4;
            devB_scale += sizeScale;
            devB_zero += sizeZero;

            // dma_done_ping recorded in schedule_dma
        }

        // ---- kernel for tile i (kernelStream) ----
        int combine = (i == 0) ? 0 : 1;

        RppTaskElement task;
        uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
        kerneldim_calc_matmul_linear(1, M, N, block_x, block_y, block_z, grid_x, grid_y, grid_z);

        task.blockDim.x = block_x;
        task.blockDim.y = block_y;
        task.blockDim.z = block_z;
        task.gridDim.x  = grid_x;
        task.gridDim.y  = grid_y;
        task.gridDim.z  = groups_per_tile;

        const uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * sizeof(short);

        task.taskName = "matrix_mul_vxM_f16_i4_f16_asym_opt";
        task.params.kernelList.clear();

        matmul_weights_i4_kernel_params(sramA + i * stride_ina * groups_per_tile, sramB_q4(ping), sramB_scale(ping),
                                        sramC, sramB_lut, sramB_zero(ping), input_acc_addr,
                                        input_acc_addr + hilo_offset, 1, 1, N, weights_group, combine, stride_ina,
                                        stride_inb, stride_scale, task.params.kernelList);

        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
        // Signal this ping buffer is safe to reuse
        rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
    }

    const int last_ping = (nr_of_tiles - 1) & 1;
    // -------------------------
    // (C) Writeback C (DMA stream), optional BF16->F32 conversion (kernel stream)
    // -------------------------
    if (out_bytes_per_element == (int) sizeof(float)) {
        // convert BF16(sramC) -> F32(sramC1) in kernelStream, then DMA writeback on dmaStream
        params.clear();
        calc_tbdim_flattern(1, N * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        // fence: wait conversion done before DMA out
        rppEventRecord(ctx.kernel_done_ping[last_ping], ctx.kernelStream);
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[last_ping], 0);

        rtMemcpyAsync((void *) devC, (const void *) sramC1, sizeC, rtMemcpySramToDevice, ctx.dmaStream);
    } else {
        // output bf16 directly
        // fence: ensure matmul finished before DMA out
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[last_ping], 0);

        rtMemcpyAsync((void *) devC, (const void *) sramC, sizeC, rtMemcpySramToDevice, ctx.dmaStream);
    }
#endif
    // end capture / instantiate graph
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }

    free(quant_table);
}

static void rpp_matmul_q4_1_vxm_build(rpp_kernel_context & ctx,
                                      int                  M,
                                      int                  K,
                                      int                  N,
                                      int                  weights_group,
                                      int                  in_bytes_per_element,
                                      int                  out_bytes_per_element,
                                      int                  use_pipeline  = 0,
                                      int                  is_instantial = 1) {
    if (use_pipeline) {
        rpp_matmul_q4_1_vxm_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                     is_instantial);
    } else {
        rpp_matmul_q4_1_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    }
}
}  // namespace kernel_q4_1_vxm
