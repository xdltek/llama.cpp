// rpp_matmul_q3xxs_vxm_nolut.cpp
#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q3_xxs_vxm_nolut/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q3_xxs_vxm_nolut/rpp_kernel_param.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q3_xxs_vxm_nolut {

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
        nr_of_tiles     = K / weights_group;
        groups_per_tile = 1;
    }

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

    if (N == 151936) {
        Ns       = 600 * 32;
        nr_of_ns = 8;
        NsTail   = 548 * 32;
    } else if (N == 200064) {
        Ns       = 640 * 32;
        nr_of_ns = 10;
        NsTail   = 492 * 32;
    }

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

static inline void kerneldim_calc_matmul_linear_legacy(uint32_t   loop,
                                                       uint32_t   row,
                                                       uint32_t   column,
                                                       uint32_t & block_x,
                                                       uint32_t & block_y,
                                                       uint32_t & block_z,
                                                       uint32_t & grid_x,
                                                       uint32_t & grid_y,
                                                       uint32_t & grid_z) {
    if (loop > 65535) {
        assert(0);
    }

    block_x       = 32;
    uint32_t cntX = column / 32;
    while (block_x < 4096) {
        if (cntX % 2 == 0) {
            block_x = block_x * 2;
            cntX    = cntX / 2;
        } else {
            break;
        }
    }

    block_y       = 1;
    uint32_t cntY = row;
    while (block_x * cntX * block_y < 4096) {
        if (cntY % 2 == 0) {
            block_y = block_y * 2;
            cntY    = cntY / 2;
        } else {
            break;
        }
    }

    grid_x  = cntX;
    grid_y  = cntY;
    block_z = RppHW32Fix(block_x, block_y, 1);
    grid_z  = loop;

    assert(block_x % 32 == 0);
    assert(block_x * grid_x == column);
}

static void matmul_weights_q3xxs_nolut_kernel_params_local(dim3 &                  blocksPerGrid,
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
                                                           uint32_t                combine,
                                                           std::vector<uint32_t> & params) {
    (void) blocksPerGrid;
    const uint32_t inUnrollStride       = N * (uint32_t) sizeof(uint16_t);
    const uint32_t blockXSize           = threadsPerBlock.x * (uint32_t) sizeof(uint16_t);
    const uint32_t scaleLoopStride      = N * (uint32_t) sizeof(uint16_t);
    const uint32_t inLoopStride0        = N * (uint32_t) sizeof(uint16_t);
    const uint32_t inLoopStride1        = 3 * N * (uint32_t) sizeof(uint16_t);
    const uint32_t in_wq_blockz_size    = (uint32_t) q3xxs_vxm_nolut_codebook_bytes((int) weights_group, (int) N);
    const uint32_t in_sign_blockz_size  = (weights_group / 16u) * N * (uint32_t) sizeof(uint16_t);
    const uint32_t in_scale_blockz_size = (weights_group / 32u) * N * (uint32_t) sizeof(uint16_t);
    const uint32_t in_a_blockz_size     = weights_group * (uint32_t) sizeof(uint16_t);
    const uint32_t loop                 = 8;

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
    params.emplace_back(loop);
    params.emplace_back(combine);
}

static void matmul_weights_q3xxs_nolut_batch_params(dim3 &                  blocksPerGrid,
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
                                                    uint32_t                in_act_expert_stride_bytes,
                                                    std::vector<uint32_t> & params) {
    (void) blocksPerGrid;
    const uint32_t inUnrollStride       = N * (uint32_t) sizeof(uint16_t);
    const uint32_t blockXSize           = threadsPerBlock.x * (uint32_t) sizeof(uint16_t);
    const uint32_t scaleLoopStride      = N * (uint32_t) sizeof(uint16_t);
    const uint32_t inLoopStride0        = N * (uint32_t) sizeof(uint16_t);
    const uint32_t inLoopStride1        = 3 * N * (uint32_t) sizeof(uint16_t);
    const uint32_t in_wq_blockz_size    = (uint32_t) q3xxs_vxm_nolut_codebook_bytes((int) weights_group, (int) N);
    const uint32_t in_sign_blockz_size  = (weights_group / 16u) * N * (uint32_t) sizeof(uint16_t);
    const uint32_t in_scale_blockz_size = (weights_group / 32u) * N * (uint32_t) sizeof(uint16_t);
    const uint32_t in_a_blockz_size     = weights_group * (uint32_t) sizeof(uint16_t);
    const uint32_t in_wq_stridez        = in_wq_expert_stride_bytes / (uint32_t) sizeof(uint16_t);
    const uint32_t in_sign_stridez      = in_sign_expert_stride_bytes / (uint32_t) sizeof(uint16_t);
    const uint32_t in_scale_stridez     = in_scale_expert_stride_bytes / (uint32_t) sizeof(uint16_t);
    const uint32_t loop                 = 8;

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
    params.emplace_back(in_act_expert_stride_bytes);
    params.emplace_back(loop);
    params.emplace_back(0);
}

static void rpp_matmul_q3xxs_vxm_nolut(rpp_kernel_context & ctx,
                                       int                  M,
                                       int                  K,
                                       int                  N,
                                       int                  weights_group,
                                       int                  in_bytes_per_element,
                                       int                  out_bytes_per_element,
                                       int                  is_instantial = 1,
                                       int                  is_capture    = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q3XXS NoLUT Configure Parameter Invalid");
    }
    if (weights_group != kQ3xxsWeightsGroup || (K % weights_group) != 0) {
        throw std::runtime_error("Matmul Q3XXS NoLUT requires weights_group == 256 and K % 256 == 0");
    }
    if (ctx.dev_in.size() < 5 || ctx.dev_out.empty()) {
        throw std::runtime_error(
            "Matmul Q3XXS NoLUT requires 5 inputs (A, codebook_nolut, scales, sign, super_scale) and 1 output");
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

    const q3xxs_vxm_nolut_workspace_ptrs lut_ws = q3xxs_vxm_nolut_prepare_lut_workspace(ctx);

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q3xxs_vxm_nolut.o");

    int nr_of_tiles, groups_per_tile;
    int nr_of_ns, Ns, NsTail;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);

    const int sizeA       = K * in_bytes_per_element;
    const int q_group     = kQ3xxsQGroup;
    const int super_group = kQ3xxsWeightsGroup;
    const int group       = super_group / q_group;
    assert(super_group == weights_group);

    const int Ktile                = groups_per_tile * weights_group;
    const int codebook_rows_tile   = q3xxs_vxm_nolut_codebook_rows(Ktile);
    const int sizeB_codebook_nolut = (int) q3xxs_vxm_nolut_codebook_bytes(Ktile, Ns);
    const int sizeB_scales         = (Ktile / 128) * Ns * (int) sizeof(uint16_t);
    const int sizeB_sign           = (Ktile / 16) * Ns * (int) sizeof(uint16_t);
    const int sizeB_super_scale    = (Ktile / weights_group) * Ns * (int) sizeof(uint16_t);
    const int sizeB_qscale_lut     = (int) q3xxs_vxm_nolut_lut_workspace::qscale_lut_bytes;
    const int sizeB_mag_lut        = (int) q3xxs_vxm_nolut_lut_workspace::mag_lut_bytes;
    const int sizeB_mat_lut        = (int) q3xxs_vxm_nolut_lut_workspace::mat_lut_bytes;
    const int sizeB_scale          = (groups_per_tile * weights_group * Ns / q_group) * (int) sizeof(uint16_t);

    const int sizeC32 = N * (int) sizeof(float);
    const int sizeC   = N * out_bytes_per_element;

    RPPdeviceptr sram_base            = ctx.virtual_sram_base;
    RPPdeviceptr sramA                = sram_base;
    RPPdeviceptr sramC                = sramA + round_up(sizeA);
    RPPdeviceptr sramC1               = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_codebook_nolut = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_scales         = sramB_codebook_nolut + round_up(sizeB_codebook_nolut);
    RPPdeviceptr sramB_sign           = sramB_scales + round_up(sizeB_scales);
    RPPdeviceptr sramB_super_scale    = sramB_sign + round_up(sizeB_sign);
    RPPdeviceptr sramB_qscale_lut     = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_mag_lut        = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_mat_lut        = sramB_mag_lut + round_up(sizeB_mag_lut);
    RPPdeviceptr sramB_scale          = sramB_mat_lut + round_up(sizeB_mat_lut);
    const int    total_sram_bytes     = (int) ((sramB_scale + round_up(sizeB_scale)) - sram_base);
    if ((uint32_t) total_sram_bytes > kSramLimitBytes) {
        throw std::runtime_error("Matmul Q3XXS NoLUT capture path exceeds the 22MB SRAM budget");
    }

    rtMemcpyAsync((void *) sramA, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
    q3xxs_vxm_nolut_copy_lut_workspace_to_sram(sramB_qscale_lut, sramB_mag_lut, sramB_mat_lut, lut_ws,
                                               ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
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
    const int hilo_offset = (Nx * Nz + 31) / 32 * 32 * (int) sizeof(uint16_t);

    int combine = 0;
    for (int i = 0; i < nr_of_tiles; i++) {
        for (int j = 0; j < nr_of_ns - 1; j++) {
            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_codebook_nolut, (const void *) devB_codebook_nolut, sizeB_codebook_nolut,
                              rtMemcpyDeviceToSram, ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_scales, (const void *) devB_scales, sizeB_scales, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_sign, (const void *) devB_sign, sizeB_sign, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale,
                              rtMemcpyDeviceToSram, ctx.kernelStream);

                devB_codebook_nolut += sizeB_codebook_nolut;
                devB_scales += sizeB_scales;
                devB_sign += sizeB_sign;
                devB_super_scale += sizeB_super_scale;
            } else {
                const int codebook_nolut_stride = codebook_rows_tile * N * (int) sizeof(uint16_t);
                const int scales_stride         = (Ktile / 128) * N * (int) sizeof(uint16_t);
                const int sign_stride           = (Ktile / 16) * N * (int) sizeof(uint16_t);
                const int super_scale_stride    = (Ktile / weights_group) * N * (int) sizeof(uint16_t);
                const int tile_ns_offset        = j * Ns * (int) sizeof(uint16_t);

                rtMemcpy2DAsync((void *) sramB_codebook_nolut, Ns * (int) sizeof(uint16_t),
                                (const void *) (devB_codebook_nolut + i * codebook_nolut_stride + tile_ns_offset),
                                N * (int) sizeof(uint16_t), Ns * (int) sizeof(uint16_t), codebook_rows_tile,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_scales, Ns * (int) sizeof(uint16_t),
                                (const void *) (devB_scales + i * scales_stride + tile_ns_offset),
                                N * (int) sizeof(uint16_t), Ns * (int) sizeof(uint16_t), Ktile / 128,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_sign, Ns * (int) sizeof(uint16_t),
                                (const void *) (devB_sign + i * sign_stride + tile_ns_offset),
                                N * (int) sizeof(uint16_t), Ns * (int) sizeof(uint16_t), Ktile / 16,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_super_scale, Ns * (int) sizeof(uint16_t),
                                (const void *) (devB_super_scale + i * super_scale_stride + tile_ns_offset),
                                N * (int) sizeof(uint16_t), Ns * (int) sizeof(uint16_t), Ktile / weights_group,
                                rtMemcpyDeviceToSram, ctx.kernelStream);
            }

            params.clear();
            q3xxs_super_scale_blocks(Ktile, weights_group, group, Ns, threadsPerBlock, blocksPerGrid);
            q3xxs_super_scale_params((uint32_t) sramB_scales, (uint32_t) sramB_super_scale, (uint32_t) sramB_qscale_lut,
                                     (uint32_t) sramB_scale, Ktile, Ns, weights_group, q_group, blocksPerGrid,
                                     threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            combine = (i == 0) ? 0 : 1;

            RppTaskElement task;
            uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
            kerneldim_calc_matmul_linear_legacy(1, M, Ns, block_x, block_y, block_z, grid_x, grid_y, grid_z);
            task.blockDim.x = block_x;
            task.blockDim.y = block_y;
            task.blockDim.z = block_z;
            task.gridDim.x  = grid_x;
            task.gridDim.y  = grid_y;
            task.gridDim.z  = groups_per_tile;
            task.taskName   = "matrix_mul_vxM_f16_q3xxs_f16_asym_nolut_opt";
            task.params.kernelList.clear();

            const uint32_t stride_ina = weights_group * (uint32_t) sizeof(uint16_t);
            dim3           matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
            dim3           matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
            matmul_weights_q3xxs_nolut_kernel_params_local(
                matmulBlocks, matmulThreads, (uint32_t) (sramA + i * stride_ina * groups_per_tile),
                (uint32_t) sramB_codebook_nolut, (uint32_t) sramB_scale, (uint32_t) sramB_sign,
                (uint32_t) (sramC + j * Ns * (uint32_t) sizeof(uint16_t)), (uint32_t) sramB_mag_lut, 0, 0,
                (uint32_t) Ns, (uint32_t) N * (uint32_t) sizeof(uint16_t), (uint32_t) weights_group, (uint32_t) combine,
                task.params.kernelList);
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    for (int i = 0; i < nr_of_tiles; i++) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_codebook_nolut, (const void *) devB_codebook_nolut, sizeB_codebook_nolut,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_scales, (const void *) devB_scales, sizeB_scales, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_sign, (const void *) devB_sign, sizeB_sign, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale,
                          rtMemcpyDeviceToSram, ctx.kernelStream);

            devB_codebook_nolut += sizeB_codebook_nolut;
            devB_scales += sizeB_scales;
            devB_sign += sizeB_sign;
            devB_super_scale += sizeB_super_scale;
        } else {
            const int codebook_nolut_stride = codebook_rows_tile * N * (int) sizeof(uint16_t);
            const int scales_stride         = (Ktile / 128) * N * (int) sizeof(uint16_t);
            const int sign_stride           = (Ktile / 16) * N * (int) sizeof(uint16_t);
            const int super_scale_stride    = (Ktile / weights_group) * N * (int) sizeof(uint16_t);
            const int tail_ns_offset        = (nr_of_ns - 1) * Ns * (int) sizeof(uint16_t);

            rtMemcpy2DAsync((void *) sramB_codebook_nolut, NsTail * (int) sizeof(uint16_t),
                            (const void *) (devB_codebook_nolut + i * codebook_nolut_stride + tail_ns_offset),
                            N * (int) sizeof(uint16_t), NsTail * (int) sizeof(uint16_t), codebook_rows_tile,
                            rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_scales, NsTail * (int) sizeof(uint16_t),
                            (const void *) (devB_scales + i * scales_stride + tail_ns_offset),
                            N * (int) sizeof(uint16_t), NsTail * (int) sizeof(uint16_t), Ktile / 128,
                            rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_sign, NsTail * (int) sizeof(uint16_t),
                            (const void *) (devB_sign + i * sign_stride + tail_ns_offset), N * (int) sizeof(uint16_t),
                            NsTail * (int) sizeof(uint16_t), Ktile / 16, rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_super_scale, NsTail * (int) sizeof(uint16_t),
                            (const void *) (devB_super_scale + i * super_scale_stride + tail_ns_offset),
                            N * (int) sizeof(uint16_t), NsTail * (int) sizeof(uint16_t), Ktile / weights_group,
                            rtMemcpyDeviceToSram, ctx.kernelStream);
        }

        params.clear();
        q3xxs_super_scale_blocks(Ktile, weights_group, group, NsTail, threadsPerBlock, blocksPerGrid);
        q3xxs_super_scale_params((uint32_t) sramB_scales, (uint32_t) sramB_super_scale, (uint32_t) sramB_qscale_lut,
                                 (uint32_t) sramB_scale, Ktile, NsTail, weights_group, q_group, blocksPerGrid,
                                 threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        combine = (i == 0) ? 0 : 1;

        RppTaskElement task;
        uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
        kerneldim_calc_matmul_linear_legacy(1, M, NsTail, block_x, block_y, block_z, grid_x, grid_y, grid_z);
        task.blockDim.x = block_x;
        task.blockDim.y = block_y;
        task.blockDim.z = block_z;
        task.gridDim.x  = grid_x;
        task.gridDim.y  = grid_y;
        task.gridDim.z  = groups_per_tile;
        task.taskName   = "matrix_mul_vxM_f16_q3xxs_f16_asym_nolut_opt";
        task.params.kernelList.clear();

        const uint32_t stride_ina = weights_group * (uint32_t) sizeof(uint16_t);
        dim3           matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3           matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q3xxs_nolut_kernel_params_local(
            matmulBlocks, matmulThreads, (uint32_t) (sramA + i * stride_ina * groups_per_tile),
            (uint32_t) sramB_codebook_nolut, (uint32_t) sramB_scale, (uint32_t) sramB_sign,
            (uint32_t) (sramC + (nr_of_ns - 1) * Ns * (uint32_t) sizeof(uint16_t)), (uint32_t) sramB_mag_lut, 0, 0,
            (uint32_t) NsTail, (uint32_t) N * (uint32_t) sizeof(uint16_t), (uint32_t) weights_group, (uint32_t) combine,
            task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpyAsync((void *) devC, (const void *) sramC1, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        rtMemcpyAsync((void *) devC, (const void *) sramC, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    }

    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q3xxs_vxm_nolut_pipeline(rpp_kernel_context & ctx,
                                                int                  M,
                                                int                  K,
                                                int                  N,
                                                int                  weights_group,
                                                int                  in_bytes_per_element,
                                                int                  out_bytes_per_element,
                                                int                  is_instantial = 1,
                                                int                  is_capture    = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q3XXS NoLUT Configure Parameter Invalid");
    }
    if (weights_group != kQ3xxsWeightsGroup || (K % weights_group) != 0) {
        throw std::runtime_error("Matmul Q3XXS NoLUT requires weights_group == 256 and K % 256 == 0");
    }
    if (ctx.dev_in.size() < 5 || ctx.dev_out.empty()) {
        throw std::runtime_error(
            "Matmul Q3XXS NoLUT requires 5 inputs (A, codebook_nolut, scales, sign, super_scale) and 1 output");
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

    const q3xxs_vxm_nolut_workspace_ptrs lut_ws = q3xxs_vxm_nolut_prepare_lut_workspace(ctx);

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q3xxs_vxm_nolut.o");

    int nr_of_tiles, groups_per_tile;
    int nr_of_ns, Ns, NsTail;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);

    const int sizeA       = K * in_bytes_per_element;
    const int q_group     = kQ3xxsQGroup;
    const int super_group = kQ3xxsWeightsGroup;
    const int group       = super_group / q_group;
    assert(super_group == weights_group);

    const int Ktile                = groups_per_tile * weights_group;
    const int codebook_rows_tile   = q3xxs_vxm_nolut_codebook_rows(Ktile);
    const int sizeB_codebook_nolut = (int) q3xxs_vxm_nolut_codebook_bytes(Ktile, Ns);
    const int sizeB_scales         = (Ktile / 128) * Ns * (int) sizeof(uint16_t);
    const int sizeB_sign           = (Ktile / 16) * Ns * (int) sizeof(uint16_t);
    const int sizeB_super_scale    = (Ktile / weights_group) * Ns * (int) sizeof(uint16_t);
    const int sizeB_qscale_lut     = (int) q3xxs_vxm_nolut_lut_workspace::qscale_lut_bytes;
    const int sizeB_mag_lut        = (int) q3xxs_vxm_nolut_lut_workspace::mag_lut_bytes;
    const int sizeB_mat_lut        = (int) q3xxs_vxm_nolut_lut_workspace::mat_lut_bytes;
    const int sizeB_scale          = (groups_per_tile * weights_group * Ns / q_group) * (int) sizeof(uint16_t);

    const int sizeC32 = N * (int) sizeof(float);
    const int sizeC   = N * out_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA     = sram_base;

    // Ping 0
    RPPdeviceptr sramB_codebook_nolut_0 = sramA + round_up(sizeA);
    RPPdeviceptr sramB_scales_0         = sramB_codebook_nolut_0 + round_up(sizeB_codebook_nolut);
    RPPdeviceptr sramB_sign_0           = sramB_scales_0 + round_up(sizeB_scales);
    RPPdeviceptr sramB_super_scale_0    = sramB_sign_0 + round_up(sizeB_sign);

    // Ping 1
    RPPdeviceptr sramB_codebook_nolut_1 = sramB_super_scale_0 + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_scales_1         = sramB_codebook_nolut_1 + round_up(sizeB_codebook_nolut);
    RPPdeviceptr sramB_sign_1           = sramB_scales_1 + round_up(sizeB_scales);
    RPPdeviceptr sramB_super_scale_1    = sramB_sign_1 + round_up(sizeB_sign);

    // Shared B intermediates and output C buffers
    RPPdeviceptr sramB_qscale_lut = sramB_super_scale_1 + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_mag_lut    = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_mat_lut    = sramB_mag_lut + round_up(sizeB_mag_lut);
    RPPdeviceptr sramB_scale      = sramB_mat_lut + round_up(sizeB_mat_lut);
    RPPdeviceptr sramC            = sramB_scale + round_up(sizeB_scale);
    RPPdeviceptr sramC1           = sramC + round_up(sizeC32);

    const int total_sram_bytes = (int) ((sramC1 + round_up(sizeC)) - sram_base);
    if ((uint32_t) total_sram_bytes > kSramLimitBytes) {
        throw std::runtime_error("Matmul Q3XXS NoLUT capture path exceeds the 22MB SRAM budget");
    }

    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);

    rtMemcpyAsync((void *) sramA, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
    q3xxs_vxm_nolut_copy_lut_workspace_to_sram(sramB_qscale_lut, sramB_mag_lut, sramB_mat_lut, lut_ws,
                                               ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
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
    const int hilo_offset = (Nx * Nz + 31) / 32 * 32 * (int) sizeof(uint16_t);
    (void) hilo_offset;

    auto sramB_codebook_nolut = [&](int ping) {
        return ping ? sramB_codebook_nolut_1 : sramB_codebook_nolut_0;
    };
    auto sramB_scales = [&](int ping) {
        return ping ? sramB_scales_1 : sramB_scales_0;
    };
    auto sramB_sign = [&](int ping) {
        return ping ? sramB_sign_1 : sramB_sign_0;
    };
    auto sramB_super_scale = [&](int ping) {
        return ping ? sramB_super_scale_1 : sramB_super_scale_0;
    };

    const int    codebook_nolut_stride   = codebook_rows_tile * N * (int) sizeof(uint16_t);
    const int    scales_stride           = (Ktile / 128) * N * (int) sizeof(uint16_t);
    const int    sign_stride             = (Ktile / 16) * N * (int) sizeof(uint16_t);
    const int    super_scale_stride      = (Ktile / weights_group) * N * (int) sizeof(uint16_t);
    const int    tile_ns_stride          = Ns * (int) sizeof(uint16_t);
    const int    total_jobs              = nr_of_tiles * nr_of_ns;
    RPPdeviceptr devB_codebook_nolut_dma = devB_codebook_nolut;
    RPPdeviceptr devB_scales_dma         = devB_scales;
    RPPdeviceptr devB_sign_dma           = devB_sign;
    RPPdeviceptr devB_super_scale_dma    = devB_super_scale;

    auto schedule_dma = [&](int ping, int tile_idx, int ns_idx, int NsSeg) {
        const int tile_ns_offset = ns_idx * tile_ns_stride;
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_codebook_nolut(ping), (const void *) devB_codebook_nolut_dma,
                          sizeB_codebook_nolut, rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpyAsync((void *) sramB_scales(ping), (const void *) devB_scales_dma, sizeB_scales,
                          rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpyAsync((void *) sramB_sign(ping), (const void *) devB_sign_dma, sizeB_sign, rtMemcpyDeviceToSram,
                          ctx.dmaStream);
            rtMemcpyAsync((void *) sramB_super_scale(ping), (const void *) devB_super_scale_dma, sizeB_super_scale,
                          rtMemcpyDeviceToSram, ctx.dmaStream);

            devB_codebook_nolut_dma += sizeB_codebook_nolut;
            devB_scales_dma += sizeB_scales;
            devB_sign_dma += sizeB_sign;
            devB_super_scale_dma += sizeB_super_scale;
        } else {
            rtMemcpy2DAsync((void *) sramB_codebook_nolut(ping), NsSeg * (int) sizeof(uint16_t),
                            (const void *) (devB_codebook_nolut + tile_idx * codebook_nolut_stride + tile_ns_offset),
                            N * (int) sizeof(uint16_t), NsSeg * (int) sizeof(uint16_t), codebook_rows_tile,
                            rtMemcpyDeviceToSram, ctx.dmaStream);

            rtMemcpy2DAsync((void *) sramB_scales(ping), NsSeg * (int) sizeof(uint16_t),
                            (const void *) (devB_scales + tile_idx * scales_stride + tile_ns_offset),
                            N * (int) sizeof(uint16_t), NsSeg * (int) sizeof(uint16_t), Ktile / 128,
                            rtMemcpyDeviceToSram, ctx.dmaStream);

            rtMemcpy2DAsync((void *) sramB_sign(ping), NsSeg * (int) sizeof(uint16_t),
                            (const void *) (devB_sign + tile_idx * sign_stride + tile_ns_offset),
                            N * (int) sizeof(uint16_t), NsSeg * (int) sizeof(uint16_t), Ktile / 16,
                            rtMemcpyDeviceToSram, ctx.dmaStream);

            rtMemcpy2DAsync((void *) sramB_super_scale(ping), NsSeg * (int) sizeof(uint16_t),
                            (const void *) (devB_super_scale + tile_idx * super_scale_stride + tile_ns_offset),
                            N * (int) sizeof(uint16_t), NsSeg * (int) sizeof(uint16_t), Ktile / weights_group,
                            rtMemcpyDeviceToSram, ctx.dmaStream);
        }

        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    {
        const int job0_i  = 0;
        const int job0_j  = 0;
        const int job0_Ns = (job0_j == nr_of_ns - 1) ? NsTail : Ns;
        schedule_dma(0, job0_i, job0_j, job0_Ns);
    }

    for (int job = 0; job < total_jobs; ++job) {
        const int ping   = job & 1;
        const int tile_i = job / nr_of_ns;
        const int ns_j   = job % nr_of_ns;
        const int NsSeg  = (ns_j == nr_of_ns - 1) ? NsTail : Ns;

        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

        if (job + 1 < total_jobs) {
            const int next_job  = job + 1;
            const int next_ping = next_job & 1;
            const int next_i    = next_job / nr_of_ns;
            const int next_j    = next_job % nr_of_ns;
            const int next_Ns   = (next_j == nr_of_ns - 1) ? NsTail : Ns;
            schedule_dma(next_ping, next_i, next_j, next_Ns);
        }

        params.clear();
        q3xxs_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
        q3xxs_super_scale_params((uint32_t) sramB_scales(ping), (uint32_t) sramB_super_scale(ping),
                                 (uint32_t) sramB_qscale_lut, (uint32_t) sramB_scale, Ktile, NsSeg, weights_group,
                                 q_group, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        const int combine = (tile_i == 0) ? 0 : 1;

        RppTaskElement task;
        uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
        kerneldim_calc_matmul_linear_legacy(1, M, NsSeg, block_x, block_y, block_z, grid_x, grid_y, grid_z);
        task.blockDim.x = block_x;
        task.blockDim.y = block_y;
        task.blockDim.z = block_z;
        task.gridDim.x  = grid_x;
        task.gridDim.y  = grid_y;
        task.gridDim.z  = groups_per_tile;
        task.taskName   = "matrix_mul_vxM_f16_q3xxs_f16_asym_nolut_opt";
        task.params.kernelList.clear();

        const uint32_t stride_ina = weights_group * (uint32_t) sizeof(uint16_t);
        dim3           matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3           matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q3xxs_nolut_kernel_params_local(
            matmulBlocks, matmulThreads, (uint32_t) (sramA + tile_i * stride_ina * groups_per_tile),
            (uint32_t) sramB_codebook_nolut(ping), (uint32_t) sramB_scale, (uint32_t) sramB_sign(ping),
            (uint32_t) (sramC + ns_j * tile_ns_stride), (uint32_t) sramB_mag_lut, 0, 0, (uint32_t) NsSeg,
            (uint32_t) N * (uint32_t) sizeof(uint16_t), (uint32_t) weights_group, (uint32_t) combine,
            task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);

        rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
    }

    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpyAsync((void *) devC, (const void *) sramC1, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        rtMemcpyAsync((void *) devC, (const void *) sramC, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    }

    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q3xxs_vxm_nolut_sram(rpp_kernel_context & ctx,
                                            int                  M,
                                            int                  K,
                                            int                  N,
                                            int                  weights_group,
                                            int                  in_bytes_per_element,
                                            int                  out_bytes_per_element,
                                            int                  experts,
                                            int                  is_instantial = 1,
                                            int                  is_capture    = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q3XXS NoLUT SRAM Configure Parameter Invalid");
    }
    if (weights_group != kQ3xxsWeightsGroup || (K % weights_group) != 0) {
        throw std::runtime_error("Matmul Q3XXS NoLUT SRAM expects weights_group == 256 and K % 256 == 0");
    }
    if (experts <= 0) {
        throw std::runtime_error("Matmul Q3XXS NoLUT SRAM expects experts > 0");
    }
    if (ctx.dev_in.size() < 9 || ctx.dev_out.empty()) {
        throw std::runtime_error("Matmul Q3XXS NoLUT SRAM requires ctx.dev_in[0..8] and ctx.dev_out SRAM addresses");
    }
    if (out_bytes_per_element == (int) sizeof(float) && ctx.dev_out.size() < 2) {
        throw std::runtime_error(
            "Matmul Q3XXS NoLUT SRAM fp32 output requires ctx.dev_out[0]=fp32_out and ctx.dev_out[1]=bf16_tmp");
    }

    q3xxs_vxm_nolut_sram_io io;
    q3xxs_vxm_nolut_prepare_sram_io(ctx, io, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                    experts, false);
    // const q3xxs_vxm_nolut_workspace_ptrs lut_ws = q3xxs_vxm_nolut_prepare_lut_workspace(ctx);

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    const RPPdeviceptr sramA                = ctx.dev_in[0];
    const RPPdeviceptr sramB_codebook_nolut = ctx.dev_in[1];
    const RPPdeviceptr sramB_scales         = ctx.dev_in[2];
    const RPPdeviceptr sramB_sign           = ctx.dev_in[3];
    const RPPdeviceptr sramB_super_scale    = ctx.dev_in[4];
    const RPPdeviceptr sramB_qscale_lut     = ctx.dev_in[5];
    const RPPdeviceptr sramB_mag_lut        = ctx.dev_in[6];
    const RPPdeviceptr sramB_mat_lut        = ctx.dev_in[7];
    const RPPdeviceptr sramB_scale          = ctx.dev_in[8];

    RPPdeviceptr sramC   = ctx.dev_out[0];
    RPPdeviceptr sramOut = ctx.dev_out[0];
    if (out_bytes_per_element == (int) sizeof(float)) {
        sramC = ctx.dev_out[1];
    }

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q3xxs_vxm_nolut.o");

    const int groups_per_tile = K / weights_group;
    const int Ktile           = io.Ktile;
    assert(K == Ktile);

    const int q_group     = kQ3xxsQGroup;
    const int super_group = kQ3xxsWeightsGroup;
    const int group       = super_group / q_group;
    assert(super_group == weights_group);

    const int          sizeB_codebook_nolut       = (int) io.size_codebook_nolut_tile;
    const int          sizeB_scales               = (int) io.size_scales_tile;
    const int          sizeB_sign                 = (int) io.size_sign_tile;
    const int          sizeB_super_scale          = (int) io.size_super_scale_tile;
    const int          sizeB_scale                = (int) io.sizeB_scale_scratch;
    const uint32_t     expert_weights_stride      = io.size_weights_expert;
    const RPPdeviceptr expected_sramB_scales      = sramB_codebook_nolut + (RPPdeviceptr) sizeB_codebook_nolut;
    const RPPdeviceptr expected_sramB_sign        = expected_sramB_scales + (RPPdeviceptr) sizeB_scales;
    const RPPdeviceptr expected_sramB_super_scale = expected_sramB_sign + (RPPdeviceptr) sizeB_sign;
    if (sramB_scales != expected_sramB_scales || sramB_sign != expected_sramB_sign ||
        sramB_super_scale != expected_sramB_super_scale) {
        throw std::runtime_error(
            "Matmul Q3XXS NoLUT SRAM expects packed weight chunk layout [codebook_nolut|scales|sign|super_scale]");
    }

    // q3xxs_vxm_nolut_copy_lut_workspace_to_sram(sramB_qscale_lut, sramB_mag_lut, sramB_mat_lut, lut_ws,
    //                                            ctx.kernelStream);

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
    const int hilo_offset = experts * N * (int) sizeof(uint16_t);

    for (int e = 0; e < experts; ++e) {
        const RPPdeviceptr sramB_scales_cur = sramB_codebook_nolut +
                                              (RPPdeviceptr) e * (RPPdeviceptr) expert_weights_stride +
                                              (RPPdeviceptr) sizeB_codebook_nolut;
        const RPPdeviceptr sramB_super_scale_cur = sramB_codebook_nolut +
                                                   (RPPdeviceptr) e * (RPPdeviceptr) expert_weights_stride +
                                                   (RPPdeviceptr) (sizeB_codebook_nolut + sizeB_scales + sizeB_sign);
        const RPPdeviceptr sramB_scale_cur = sramB_scale + (RPPdeviceptr) e * (RPPdeviceptr) sizeB_scale;

        params.clear();
        q3xxs_super_scale_blocks(Ktile, weights_group, group, N, threadsPerBlock, blocksPerGrid);
        q3xxs_super_scale_params((uint32_t) sramB_scales_cur, (uint32_t) sramB_super_scale_cur,
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

    uint32_t tx                 = 1;
    uint32_t experts_per_launch = (uint32_t) experts;
    while (experts_per_launch * block_x >= 8192 && experts_per_launch > 1) {
        experts_per_launch >>= 1;
        tx <<= 1;
    }
    if ((uint32_t) experts != tx * experts_per_launch) {
        throw std::runtime_error("Matmul Q3XXS NoLUT SRAM expects experts to be power-of-2 for Tx tiling");
    }

    task.blockDim.x = block_x;
    task.blockDim.y = block_y;
    task.blockDim.z = experts_per_launch;
    task.gridDim.x  = grid_x;
    task.gridDim.y  = grid_y;
    task.gridDim.z  = groups_per_tile;
    task.taskName   = "matrix_mul_vxM_q3xxs_nolut_asym_batch";
    task.params.kernelList.clear();

    const uint32_t out_expert_stride    = (uint32_t) N * (uint32_t) sizeof(uint16_t);
    const uint32_t in_act_expert_stride = (uint32_t) K * (uint32_t) sizeof(uint16_t);
    for (uint32_t t = 0; t < tx; ++t) {
        const uint32_t expert_begin = t * experts_per_launch;
        const uint32_t in_act_base =
            (uint32_t) (sramA + (RPPdeviceptr) expert_begin * (RPPdeviceptr) in_act_expert_stride);
        const uint32_t in_wq_base =
            (uint32_t) (sramB_codebook_nolut + (RPPdeviceptr) expert_begin * (RPPdeviceptr) expert_weights_stride);
        const uint32_t in_sign_base = in_wq_base + (uint32_t) sizeB_codebook_nolut + (uint32_t) sizeB_scales;
        const uint32_t in_scale_base =
            (uint32_t) (sramB_scale + (RPPdeviceptr) expert_begin * (RPPdeviceptr) sizeB_scale);
        const uint32_t out_base = (uint32_t) (sramC + (RPPdeviceptr) expert_begin * (RPPdeviceptr) out_expert_stride);

        task.params.kernelList.clear();
        dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q3xxs_nolut_batch_params(matmulBlocks, matmulThreads, in_act_base, in_wq_base, in_scale_base,
                                                in_sign_base, out_base, (uint32_t) sramB_mag_lut, 0, 0, (uint32_t) N,
                                                (uint32_t) hilo_offset, (uint32_t) weights_group, expert_weights_stride,
                                                expert_weights_stride, (uint32_t) sizeB_scale, in_act_expert_stride,
                                                task.params.kernelList);
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
            "Matmul Q3XXS NoLUT SRAM bf16 output requires ctx.dev_out[0] to point to matmul bf16 output SRAM");
    }

    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q3xxs_vxm_nolut_build(rpp_kernel_context & ctx,
                                             int                  M,
                                             int                  K,
                                             int                  N,
                                             int                  weights_group,
                                             int                  in_bytes_per_element,
                                             int                  out_bytes_per_element,
                                             int                  use_pipeline    = 0,
                                             int                  use_sram_direct = 0,
                                             int                  experts         = 8,
                                             int                  is_instantial   = 1,
                                             int                  is_capture      = 1) {
    use_pipeline = 0;
    if (!use_sram_direct) {
        if (use_pipeline) {
            rpp_matmul_q3xxs_vxm_nolut_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element,
                                                out_bytes_per_element, is_instantial, is_capture);
        } else {
            rpp_matmul_q3xxs_vxm_nolut(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                       is_instantial, is_capture);
        }
    } else {
        rpp_matmul_q3xxs_vxm_nolut_sram(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                        experts, is_instantial, is_capture);
    }
}
}  // namespace kernel_q3_xxs_vxm_nolut
