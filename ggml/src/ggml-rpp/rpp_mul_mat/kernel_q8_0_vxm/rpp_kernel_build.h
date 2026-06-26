
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q8_0_vxm/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q8_0_vxm/rpp_kernel_param.h"

#include <assert.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_q8_0_vxm {

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

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_q80_vxm(rpp_kernel_context & ctx,
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
    RPPdeviceptr          devB_q80   = ctx.dev_in[1];
    RPPdeviceptr          devB_scale = ctx.dev_in[2];
    RPPdeviceptr          devC       = ctx.dev_out[0];

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q80_vxm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    int nr_of_tiles, groups_per_tile;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile);

    const int sizeA     = (K) *in_bytes_per_element;
    // weights_group
    const int sizeB_q80 = groups_per_tile * weights_group * N * sizeof(int8_t);
    const int sizeScale = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeZero  = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeC     = N * out_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA     = sram_base;
    RPPdeviceptr sramC     = sramA + round_up(sizeA);
    RPPdeviceptr sramC1    = sramC + round_up(sizeC);

    RPPdeviceptr sramB_q80   = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_scale = sramB_q80 + round_up(sizeB_q80);
    RPPdeviceptr sramA_acc   = sramB_scale + round_up(sizeScale);

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
        rtMemcpyAsync((void *) sramB_q80, (const void *) devB_q80, sizeB_q80, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpyAsync((void *) sramB_scale, (const void *) devB_scale, sizeScale, rtMemcpyDeviceToSram,
                      ctx.kernelStream);

        devB_q80 += sizeB_q80;
        devB_scale += sizeScale;

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
        uint32_t stride_inb     = weights_group * N;
        uint32_t stride_scale   = N * sizeof(short);
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * sizeof(short);
        task.taskName           = "matrix_mul_vxM_f16_i80_f16_asym_opt";
        task.params.kernelList.clear();
        matmul_weights_i80_kernel_params(sramA + i * stride_ina * groups_per_tile, sramB_q80, sramB_scale, sramC, 0, 0,
                                         input_acc_addr, input_acc_addr + hilo_offset, 1, 1, N, weights_group, combine,
                                         stride_ina, stride_inb, stride_scale, task.params.kernelList);

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

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q80_vxm_pipeline(rpp_kernel_context & ctx,
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
    RPPdeviceptr          devB_q80   = ctx.dev_in[1];
    RPPdeviceptr          devB_scale = ctx.dev_in[2];
    RPPdeviceptr          devC       = ctx.dev_out[0];

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q80_vxm.o");

    int nr_of_tiles, groups_per_tile;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile);

    const int sizeA     = (K) *in_bytes_per_element;
    const int sizeB_q80 = groups_per_tile * weights_group * N * sizeof(int8_t);
    const int sizeScale = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeZero  = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeC     = N * out_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA     = sram_base;
    RPPdeviceptr sramC     = sramA + round_up(sizeA);
    RPPdeviceptr sramC1    = sramC + round_up(sizeC);

    RPPdeviceptr sramB_q80_0   = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_scale_0 = sramB_q80_0 + round_up(sizeB_q80);
    RPPdeviceptr sramB_q80_1   = sramB_scale_0 + round_up(sizeScale);
    RPPdeviceptr sramB_scale_1 = sramB_q80_1 + round_up(sizeB_q80);
    RPPdeviceptr sramA_acc     = sramB_scale_1 + round_up(sizeScale);

    const int total_sram_bytes = (int) (sramA_acc + round_up(weights_group * 4) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    // Mark ping buffers as free at graph start.
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);

    rtMemcpyAsync((void *) sramA, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
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

    auto sramB_q80 = [&](int ping) {
        return ping ? sramB_q80_1 : sramB_q80_0;
    };
    auto sramB_scale = [&](int ping) {
        return ping ? sramB_scale_1 : sramB_scale_0;
    };
    auto schedule_dma = [&](int ping) {
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);
        rtMemcpyAsync((void *) sramB_q80(ping), (const void *) devB_q80, sizeB_q80, rtMemcpyDeviceToSram,
                      ctx.dmaStream);
        rtMemcpyAsync((void *) sramB_scale(ping), (const void *) devB_scale, sizeScale, rtMemcpyDeviceToSram,
                      ctx.dmaStream);
        devB_q80 += sizeB_q80;
        devB_scale += sizeScale;
        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    // Prefetch first tile.
    schedule_dma(0);

    for (int i = 0; i < nr_of_tiles; i++) {
        const int ping = (i & 1);
        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

        if (i + 1 < nr_of_tiles) {
            schedule_dma((i + 1) & 1);
        }

        combine = (i == 0) ? 0 : 1;

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
        uint32_t stride_inb     = weights_group * N;
        uint32_t stride_scale   = N * sizeof(short);
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * sizeof(short);
        task.taskName           = "matrix_mul_vxM_f16_i80_f16_asym_opt";
        task.params.kernelList.clear();
        matmul_weights_i80_kernel_params(sramA + i * stride_ina * groups_per_tile, sramB_q80(ping), sramB_scale(ping),
                                         sramC, 0, 0, input_acc_addr, input_acc_addr + hilo_offset, 1, 1, N,
                                         weights_group, combine, stride_ina, stride_inb, stride_scale,
                                         task.params.kernelList);

        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);

        rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
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

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q80_vxm_build(rpp_kernel_context & ctx,
                                     int                  M,
                                     int                  K,
                                     int                  N,
                                     int                  weights_group,
                                     int                  in_bytes_per_element,
                                     int                  out_bytes_per_element,
                                     int                  use_pipeline  = 0,
                                     int                  is_instantial = 1) {
    // if (use_pipeline) {
    //     rpp_matmul_q80_vxm_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    // } else {
    //     rpp_matmul_q80_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    // }
    rpp_matmul_q80_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
}
}  // namespace kernel_q8_0_vxm
