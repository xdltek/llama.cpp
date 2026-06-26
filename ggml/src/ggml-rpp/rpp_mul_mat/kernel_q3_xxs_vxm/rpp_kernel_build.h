#pragma once
// rpp_matmul_q3xxs_vxm.cpp (v1 overlap: dmaStream + kernelStream + pingpong SRAM)

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q3_xxs_vxm/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q3_xxs_vxm/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <array>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_q3_xxs_vxm {

inline void get_tiles_info(int   N,
                           int   K,
                           int   weights_group,
                           int & nr_of_tiles,
                           int & groups_per_tile,
                           int & nr_of_ns,
                           int & Ns,
                           int & NsTail) {
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
        // jsq3xxs
        nr_of_tiles     = K / weights_group;
        groups_per_tile = 1;
        // nr_of_tiles = 1;
        // groups_per_tile = K / weights_group;
    }
    // jsq3xxs
    if (N * weights_group * groups_per_tile > (16 * 1024 * 1024)) {
        Ns       = N / 4;
        NsTail   = Ns;
        nr_of_ns = 4;
    } else if (N * weights_group * groups_per_tile > (8 * 1024 * 1024)) {
        Ns       = N / 2;
        NsTail   = Ns;
        nr_of_ns = 2;
    } else {
        Ns       = N;
        NsTail   = Ns;
        nr_of_ns = 1;
    }

    // special handle for 151936 to make sure 32bytes alignment
    if (N == 151936) {
        Ns       = 600 * 32;
        nr_of_ns = 8;
        NsTail   = 548 * 32;
    } else if (N == 200064) {
        Ns       = 640 * 32;
        nr_of_ns = 10;
        NsTail   = 492 * 32;
        // Ns = 600 * 32;
        // nr_of_ns = 11;
        // NsTail = 252 * 32;
        // Ns = 600 * 32;
        // nr_of_ns = 1;
        // NsTail = 600 * 32;
    }

    // Workaround: keep each Ns chunk <= 4096 to avoid super-scale boundary issue.
    const int NS_SAFE = 4096;
    if (Ns > NS_SAFE) {
        nr_of_ns = (N + NS_SAFE - 1) / NS_SAFE;
        Ns       = NS_SAFE;
        NsTail   = N - (nr_of_ns - 1) * Ns;
    }

    assert(K == nr_of_tiles * weights_group * groups_per_tile);
    assert(N == ((nr_of_ns - 1) * Ns + NsTail));
    assert(NsTail % 32 == 0);
    assert(Ns % 32 == 0);
    assert(Ns >= NsTail);
}

static void matmul_weights_q3xxs_batch_params_local(dim3 &                  blocksPerGrid,
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
                                                    uint32_t                in_a_expert_stride_bytes,
                                                    uint32_t                combine,
                                                    std::vector<uint32_t> & params) {
    (void) blocksPerGrid;
    uint32_t inUnrollStride  = N * (uint32_t) sizeof(short);
    uint32_t blockXSize      = threadsPerBlock.x * (uint32_t) sizeof(short);
    uint32_t scaleLoopStride = N * (uint32_t) sizeof(short);

    uint32_t inLoopStride0        = 2 * 2 * N * (uint32_t) sizeof(short);
    uint32_t inLoopStride1        = 2 * N * (uint32_t) sizeof(short);
    uint32_t in_wq_blockz_size    = 8 * inLoopStride0;
    uint32_t in_sign_blockz_size  = 8 * inLoopStride1;
    uint32_t in_scale_blockz_size = 8 * scaleLoopStride;
    uint32_t in_a_blockz_size     = weights_group * (uint32_t) sizeof(short);
    uint32_t in_wq_stridez        = in_wq_expert_stride_bytes / 2;
    uint32_t in_sign_stridez      = in_sign_expert_stride_bytes / 2;
    uint32_t in_scale_stridez     = in_scale_expert_stride_bytes / 2;
    uint32_t in_a_stridez         = in_a_expert_stride_bytes;
    uint32_t loop                 = 8;

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
    params.emplace_back(in_a_stridez);
    params.emplace_back(loop);
    params.emplace_back(combine);
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_q3xxs_vxm(rpp_kernel_context & ctx,
                                 int                  M,
                                 int                  K,
                                 int                  N,
                                 int                  weights_group,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  is_instantial = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q3XXS Configure Paramter Invalid");
    }
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    int                   full_pitch, sub_pitch, offset;
    const int             qscale_lut_elems    = 16;
    const int             grid_lut_elems      = 256;
    const int             qscale_lut_bytes    = qscale_lut_elems * (int) sizeof(uint16_t);
    const int             grid_lut_bytes      = grid_lut_elems * (int) sizeof(uint32_t);
    const int             lut_workspace_bytes = qscale_lut_bytes + grid_lut_bytes;

    RPPdeviceptr devA              = ctx.dev_in[0];
    RPPdeviceptr devB_q4           = ctx.dev_in[1];
    RPPdeviceptr devB_qscale       = ctx.dev_in[2];
    RPPdeviceptr devB_qsign        = ctx.dev_in[3];
    RPPdeviceptr devB_super_scale  = ctx.dev_in[4];
    RPPdeviceptr devC              = ctx.dev_out[0];
    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    assert(dev_lut_workspace != 0);

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

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q3xxs_vxm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    int nr_of_tiles, groups_per_tile;
    int nr_of_ns, Ns, NsTail;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);

    const int sizeA       = K * in_bytes_per_element;
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // q_group = 32
    // wq_per_group = 32
    // in_wq     [K/256]      |  [8]  |  [32/4]    |  [4][N]
    // in_wq     [z]          |  [y]  |  [unroll]  |  [x]
    //           [grid.y]*[z] |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    const int q_group     = 32;
    const int super_group = 256;
    const int group       = super_group / q_group;
    assert(super_group == weights_group);

    const int Ktile             = groups_per_tile * weights_group;
    const int sizeB_q4          = (Ktile * Ns / 4);
    const int sizeB_qscale      = (Ktile * Ns / 64);
    const int sizeB_qsign       = (Ktile * Ns / 8);
    const int sizeB_super_scale = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int sizeB_qscale_lut  = qscale_lut_bytes;
    const int sizeB_grid_lut    = grid_lut_bytes;
    const int sizeB_scale       = (groups_per_tile * weights_group * Ns / q_group) * (int) sizeof(short);

    // const int sizeZero = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeC32 = N * sizeof(float);
    const int sizeC   = N * out_bytes_per_element;

    RPPdeviceptr sram_base         = ctx.virtual_sram_base;
    RPPdeviceptr sramA             = sram_base;
    RPPdeviceptr sramC             = sramA + round_up(sizeA);
    RPPdeviceptr sramC1            = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_q4          = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_qscale      = sramB_q4 + round_up(sizeB_q4);
    RPPdeviceptr sramB_qsign       = sramB_qscale + round_up(sizeB_qscale);
    RPPdeviceptr sramB_super_scale = sramB_qsign + round_up(sizeB_qsign);
    RPPdeviceptr sramB_qscale_lut  = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_grid_lut    = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_scale       = sramB_grid_lut + round_up(sizeB_grid_lut);
    RPPdeviceptr sramA_acc         = sramB_scale + round_up(sizeB_scale);
    const int    total_sram_bytes  = (int) (sramA_acc + round_up(weights_group * 4) - sram_base);
    const int    SRAM_LIMIT        = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }
    // -------------------------
    // (1) DDR -> SRAM async copies on dmaStream
    // -------------------------
    // CDMA D2S
    rtMemcpyAsync((void *) sramA, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_qscale_lut, (const void *) devB_qscale_lut, sizeB_qscale_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_grid_lut, (const void *) devB_grid_lut, sizeB_grid_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    int Nx = K / q_group;
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
    task_in_acc.params.kernelList.emplace_back(q_group);
    launchWrapperAysnc(task_in_acc.taskName, task_in_acc.gridDim, task_in_acc.blockDim, task_in_acc.params.kernelList,
                       ctx.rppBinMod, ctx.kernelStream);

    // -------------------------
    // with given split factor, launch kernel one by one
    // Launch the norm Ns first
    // -------------------------
    for (int i = 0; i < nr_of_tiles; i++) {
        for (int j = 0; j < nr_of_ns - 1; j++) {
            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_q4, (const void *) devB_q4, sizeB_q4, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_qscale, (const void *) devB_qscale, sizeB_qscale, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_qsign, (const void *) devB_qsign, sizeB_qsign, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale,
                              rtMemcpyDeviceToSram, ctx.kernelStream);

                devB_q4 += sizeB_q4;
                devB_qscale += sizeB_qscale;
                devB_qsign += sizeB_qsign;
                devB_super_scale += sizeB_super_scale;
            } else {
                int       q4_stride          = (Ktile / 8 * N * (int) sizeof(short));
                int       qscale_stride      = q4_stride / 16;
                int       qsign_stride       = q4_stride / 2;
                int       super_scale_stride = Ktile / weights_group * N * (int) sizeof(short);
                const int tile_ns_offset     = j * Ns * (int) sizeof(short);

                rtMemcpy2DAsync((void *) sramB_q4, Ns * (int) sizeof(short),
                                (const void *) (devB_q4 + i * q4_stride + tile_ns_offset), N * (int) sizeof(short),
                                Ns * (int) sizeof(short), Ktile / 8, rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_qscale, Ns * (int) sizeof(short),
                                (const void *) (devB_qscale + i * qscale_stride + tile_ns_offset),
                                N * (int) sizeof(short), Ns * (int) sizeof(short), Ktile / 128, rtMemcpyDeviceToSram,
                                ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_qsign, Ns * (int) sizeof(short),
                                (const void *) (devB_qsign + i * qsign_stride + tile_ns_offset),
                                N * (int) sizeof(short), Ns * (int) sizeof(short), Ktile / 16, rtMemcpyDeviceToSram,
                                ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_super_scale, Ns * (int) sizeof(short),
                                (const void *) (devB_super_scale + i * super_scale_stride + tile_ns_offset),
                                N * (int) sizeof(short), Ns * (int) sizeof(short), Ktile / weights_group,
                                rtMemcpyDeviceToSram, ctx.kernelStream);
            }

            // merge super_scale with qscale
            params.clear();
            q3xxs_super_scale_blocks(Ktile, weights_group, group, Ns, threadsPerBlock, blocksPerGrid);
            q3xxs_super_scale_params(sramB_qscale, sramB_super_scale, sramB_qscale_lut, sramB_scale, Ktile, Ns,
                                     weights_group, q_group, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            if (i == 0) {
                combine = 0;
            } else {
                combine = 1;
            }

            RppTaskElement task;
            uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
            kerneldim_calc_matmul_linear(1, M, Ns, block_x, block_y, block_z, grid_x, grid_y, grid_z);
            task.blockDim.x = block_x;
            task.blockDim.y = block_y;
            task.blockDim.z = block_z;
            task.gridDim.x  = grid_x;
            task.gridDim.y  = grid_y;
            task.gridDim.z  = grid_z;

            task.gridDim.z          = groups_per_tile;
            uint32_t stride_ina     = weights_group * sizeof(short);
            uint32_t stride_inb     = weights_group * Ns;
            //---------------------------------------------------------------------------------------
            // InputAcc [K/256]      | [8]
            //         [grid.z]      | [loopidx]
            // accIdx = blockIdx.z * 8 + loopidx
            //----------------------------------------------------------------------------------------
            uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * sizeof(short);
            task.taskName           = "matrix_mul_vxM_f16_q3xxs_f16_asym_opt";
            task.params.kernelList.clear();
            // Use the actual matmul launch shape for blockXSize-sensitive params.
            dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
            dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
            matmul_weights_q3xxs_kernel_params(matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile,
                                               sramB_q4, sramB_scale, sramB_qsign, sramC + j * Ns * sizeof(short),
                                               sramB_grid_lut, input_acc_addr, input_acc_addr + hilo_offset, Ns,
                                               N * sizeof(short), weights_group, combine, task.params.kernelList);
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    // -------------------------
    // Launch with NsTail
    // -------------------------
    for (int i = 0; i < nr_of_tiles; i++) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_q4, (const void *) devB_q4, sizeB_q4, rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qscale, (const void *) devB_qscale, sizeB_qscale, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qsign, (const void *) devB_qsign, sizeB_qsign, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale,
                          rtMemcpyDeviceToSram, ctx.kernelStream);

            devB_q4 += sizeB_q4;
            devB_qscale += sizeB_qscale;
            devB_qsign += sizeB_qsign;
            devB_super_scale += sizeB_super_scale;
        } else {
            int       q4_stride          = (Ktile / 8 * N * (int) sizeof(short));
            int       qscale_stride      = q4_stride / 16;
            int       qsign_stride       = q4_stride / 2;
            int       super_scale_stride = Ktile / weights_group * N * (int) sizeof(short);
            const int tail_ns_offset     = (nr_of_ns - 1) * Ns * (int) sizeof(short);

            rtMemcpy2DAsync((void *) sramB_q4, NsTail * (int) sizeof(short),
                            (const void *) (devB_q4 + i * q4_stride + tail_ns_offset), N * (int) sizeof(short),
                            NsTail * (int) sizeof(short), Ktile / 8, rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_qscale, NsTail * (int) sizeof(short),
                            (const void *) (devB_qscale + i * qscale_stride + tail_ns_offset), N * (int) sizeof(short),
                            NsTail * (int) sizeof(short), Ktile / 128, rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_qsign, NsTail * (int) sizeof(short),
                            (const void *) (devB_qsign + i * qsign_stride + tail_ns_offset), N * (int) sizeof(short),
                            NsTail * (int) sizeof(short), Ktile / 16, rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_super_scale, NsTail * (int) sizeof(short),
                            (const void *) (devB_super_scale + i * super_scale_stride + tail_ns_offset),
                            N * (int) sizeof(short), NsTail * (int) sizeof(short), Ktile / weights_group,
                            rtMemcpyDeviceToSram, ctx.kernelStream);
        }

        // merge super_scale with qscale
        params.clear();
        q3xxs_super_scale_blocks(Ktile, weights_group, group, NsTail, threadsPerBlock, blocksPerGrid);
        q3xxs_super_scale_params(sramB_qscale, sramB_super_scale, sramB_qscale_lut, sramB_scale, Ktile, NsTail,
                                 weights_group, q_group, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        if (i == 0) {
            combine = 0;
        } else {
            combine = 1;
        }

        RppTaskElement task;
        uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
        kerneldim_calc_matmul_linear(1, M, NsTail, block_x, block_y, block_z, grid_x, grid_y, grid_z);
        task.blockDim.x = block_x;
        task.blockDim.y = block_y;
        task.blockDim.z = block_z;
        task.gridDim.x  = grid_x;
        task.gridDim.y  = grid_y;
        task.gridDim.z  = grid_z;

        task.gridDim.z          = groups_per_tile;
        uint32_t stride_ina     = weights_group * sizeof(short);
        uint32_t stride_inb     = weights_group * NsTail;
        //---------------------------------------------------------------------------------------
        // InputAcc [K/256]      | [8]
        //         [grid.z]     | [loopidx]
        // accIdx = blockIdx.z * 8 + loopidx
        //----------------------------------------------------------------------------------------
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * sizeof(short);
        task.taskName           = "matrix_mul_vxM_f16_q3xxs_f16_asym_opt";
        task.params.kernelList.clear();
        // Use the actual matmul launch shape for blockXSize-sensitive params.
        dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q3xxs_kernel_params(
            matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile, sramB_q4, sramB_scale, sramB_qsign,
            sramC + (nr_of_ns - 1) * Ns * sizeof(short), sramB_grid_lut, input_acc_addr, input_acc_addr + hilo_offset,
            NsTail, N * sizeof(short), weights_group, combine, task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);

        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        // CDMA S2D
        rtMemcpyAsync((void *) devC, (const void *) sramC1, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        // CDMA S2D
        rtMemcpyAsync((void *) devC, (const void *) sramC, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q3xxs_vxm_sram(rpp_kernel_context & ctx,
                                      int                  M,
                                      int                  K,
                                      int                  N,
                                      int                  weights_group,
                                      int                  in_bytes_per_element,
                                      int                  out_bytes_per_element,
                                      int                  experts,
                                      int                  is_instantial = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q3XXS SRAM Configure Parameter Invalid");
    }
    if (experts <= 0) {
        throw std::runtime_error("Matmul Q3XXS SRAM expects experts > 0");
    }
    if (weights_group != 256 || (K % 256) != 0) {
        throw std::runtime_error("Matmul Q3XXS SRAM expects weights_group == 256 and K % 256 == 0");
    }
    if (ctx.dev_in.size() < 9 || ctx.dev_out.empty()) {
        throw std::runtime_error("Matmul Q3XXS SRAM requires ctx.dev_in[0..8] and ctx.dev_out SRAM addresses");
    }
    if (out_bytes_per_element == (int) sizeof(float) && ctx.dev_out.size() < 2) {
        throw std::runtime_error(
            "Matmul Q3XXS SRAM fp32 output requires ctx.dev_out[0]=fp32_out and ctx.dev_out[1]=bf16_tmp");
    }
    const RPPdeviceptr lut_ws = q3xxs_vxm_prepare_lut_workspace(ctx);

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    const RPPdeviceptr sramA             = ctx.dev_in[0];
    const RPPdeviceptr sramB_q4          = ctx.dev_in[1];
    const RPPdeviceptr sramB_qscale      = ctx.dev_in[2];
    const RPPdeviceptr sramB_qsign       = ctx.dev_in[3];
    const RPPdeviceptr sramB_super_scale = ctx.dev_in[4];
    const RPPdeviceptr sramB_qscale_lut  = ctx.dev_in[5];
    const RPPdeviceptr sramB_grid_lut    = ctx.dev_in[6];
    const RPPdeviceptr sramB_scale       = ctx.dev_in[7];
    const RPPdeviceptr sramA_acc         = ctx.dev_in[8];

    RPPdeviceptr sramC   = ctx.dev_out[0];
    RPPdeviceptr sramOut = ctx.dev_out[0];
    if (out_bytes_per_element == (int) sizeof(float)) {
        sramC = ctx.dev_out[1];
    }

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q3xxs_vxm.o");

    if (K % weights_group != 0) {
        throw std::runtime_error("Matmul Q3XXS SRAM expects K % weights_group == 0");
    }
    const int groups_per_tile = K / weights_group;
    const int Ktile           = groups_per_tile * weights_group;
    assert(K == Ktile);

    const int q_group     = 32;
    const int super_group = 256;
    const int group       = super_group / q_group;
    assert(super_group == weights_group);

    const int      sizeB_q4          = (Ktile * N / 4);
    const int      sizeB_qscale      = (Ktile * N / 64);
    const int      sizeB_qsign       = (Ktile * N / 8);
    const int      sizeB_super_scale = (int) ((Ktile / weights_group) * N * (int) sizeof(short));
    const int      sizeB_scale       = (groups_per_tile * weights_group * N / q_group) * (int) sizeof(short);
    const uint32_t expert_weights_stride =
        (uint32_t) sizeB_q4 + (uint32_t) sizeB_qscale + (uint32_t) sizeB_qsign + (uint32_t) sizeB_super_scale;

    const RPPdeviceptr expected_sramB_qscale      = sramB_q4 + (RPPdeviceptr) sizeB_q4;
    const RPPdeviceptr expected_sramB_qsign       = expected_sramB_qscale + (RPPdeviceptr) sizeB_qscale;
    const RPPdeviceptr expected_sramB_super_scale = expected_sramB_qsign + (RPPdeviceptr) sizeB_qsign;
    if (sramB_qscale != expected_sramB_qscale || sramB_qsign != expected_sramB_qsign ||
        sramB_super_scale != expected_sramB_super_scale) {
        throw std::runtime_error(
            "Matmul Q3XXS SRAM expects per-expert packed weight chunk layout [q4|qscale|qsign|super]");
    }
    q3xxs_vxm_copy_lut_workspace_to_sram(sramB_qscale_lut, sramB_grid_lut, lut_ws, ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M * experts, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    int Nx = K / q_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    const int      hilo_offset          = (Nx * Nz + 31) / 32 * 32 * (int) sizeof(short);
    const uint32_t input_acc_addr       = (uint32_t) sramA_acc;
    const uint32_t in_act_expert_stride = (uint32_t) K * (uint32_t) sizeof(short);

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
    task_in_acc.params.kernelList.emplace_back(q_group);
    launchWrapperAysnc(task_in_acc.taskName, task_in_acc.gridDim, task_in_acc.blockDim, task_in_acc.params.kernelList,
                       ctx.rppBinMod, ctx.kernelStream);

    for (int e = 0; e < experts; ++e) {
        const RPPdeviceptr expert_base           = sramB_q4 + (RPPdeviceptr) e * (RPPdeviceptr) expert_weights_stride;
        const RPPdeviceptr sramB_qscale_cur      = expert_base + (RPPdeviceptr) sizeB_q4;
        const RPPdeviceptr sramB_super_scale_cur = expert_base + (RPPdeviceptr) (sizeB_q4 + sizeB_qscale + sizeB_qsign);
        const RPPdeviceptr sramB_scale_cur       = sramB_scale + (RPPdeviceptr) e * (RPPdeviceptr) sizeB_scale;

        params.clear();
        q3xxs_super_scale_blocks(Ktile, weights_group, group, N, threadsPerBlock, blocksPerGrid);
        q3xxs_super_scale_params((uint32_t) sramB_qscale_cur, (uint32_t) sramB_super_scale_cur,
                                 (uint32_t) sramB_qscale_lut, (uint32_t) sramB_scale_cur, Ktile, N, weights_group,
                                 q_group, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    RppTaskElement task;
    uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
    kerneldim_calc_matmul_linear(1, M, N, block_x, block_y, block_z, grid_x, grid_y, grid_z);
    (void) block_z;
    (void) grid_z;

    const uint32_t out_expert_stride = (uint32_t) N * (uint32_t) sizeof(short);

    uint32_t tx                 = 1;
    uint32_t experts_per_launch = (uint32_t) experts;
    // Keep same Tx split policy as q2xs SRAM path and always launch batch kernel.
    if (block_x >= 2048) {
        tx                 = experts_per_launch;
        experts_per_launch = 1;
    } else {
        while (experts_per_launch > 1 && experts_per_launch * block_x >= 8192) {
            experts_per_launch >>= 1;
            tx <<= 1;
        }
    }
    if ((uint32_t) experts != tx * experts_per_launch) {
        throw std::runtime_error("Matmul Q3XXS SRAM expects experts to be power-of-2 for Tx tiling");
    }

    task.blockDim.x = block_x;
    task.blockDim.y = block_y;
    task.blockDim.z = experts_per_launch;
    task.gridDim.x  = grid_x;
    task.gridDim.y  = grid_y;
    task.gridDim.z  = groups_per_tile;
    task.taskName   = "matrix_mul_vxM_f16_q3xxs_f16_asym_batch";
    task.params.kernelList.clear();

    for (uint32_t t = 0; t < tx; ++t) {
        const uint32_t expert_begin = t * experts_per_launch;
        const uint32_t in_act_base =
            (uint32_t) (sramA + (RPPdeviceptr) expert_begin * (RPPdeviceptr) in_act_expert_stride);
        const RPPdeviceptr in_wq_base   = sramB_q4 + (RPPdeviceptr) expert_begin * (RPPdeviceptr) expert_weights_stride;
        const RPPdeviceptr in_sign_base = in_wq_base + (RPPdeviceptr) (sizeB_q4 + sizeB_qscale);
        const RPPdeviceptr in_scale_base = sramB_scale + (RPPdeviceptr) expert_begin * (RPPdeviceptr) sizeB_scale;
        const RPPdeviceptr out_base      = sramC + (RPPdeviceptr) expert_begin * (RPPdeviceptr) out_expert_stride;

        task.params.kernelList.clear();
        dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q3xxs_batch_params_local(
            matmulBlocks, matmulThreads, in_act_base, (uint32_t) in_wq_base, (uint32_t) in_scale_base,
            (uint32_t) in_sign_base, (uint32_t) out_base, (uint32_t) sramB_grid_lut, input_acc_addr,
            input_acc_addr + (uint32_t) hilo_offset, (uint32_t) N,
            (uint32_t) experts * (uint32_t) N * (uint32_t) sizeof(short), (uint32_t) weights_group,
            (uint32_t) expert_weights_stride, (uint32_t) expert_weights_stride, (uint32_t) sizeB_scale,
            in_act_expert_stride, 0, task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * experts * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramOut, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    } else if (sramOut != sramC) {
        throw std::runtime_error(
            "Matmul Q3XXS SRAM bf16 output requires ctx.dev_out[0] to point to matmul bf16 output SRAM");
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q3xxs_vxm_build(rpp_kernel_context & ctx,
                                       int                  M,
                                       int                  K,
                                       int                  N,
                                       int                  weights_group,
                                       int                  in_bytes_per_element,
                                       int                  out_bytes_per_element,
                                       int                  use_pipeline    = 0,
                                       int                  use_sram_direct = 0,
                                       int                  experts         = 8,
                                       int                  is_instantial   = 1) {
    (void) use_pipeline;
    if (!use_sram_direct) {
        rpp_matmul_q3xxs_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    } else {
        rpp_matmul_q3xxs_vxm_sram(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, experts,
                                  is_instantial);
    }
}
}  // namespace kernel_q3_xxs_vxm
