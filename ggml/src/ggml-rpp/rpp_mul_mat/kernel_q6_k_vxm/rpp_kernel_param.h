
#pragma once
#include "rpp_drv_api.h"

#include <assert.h>
#include <math.h>
#include <rpp_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace kernel_q6_k_vxm {
static void matmul_super_scale_params(uint32_t                in_scale,
                                      uint32_t                in_super_scale,
                                      uint32_t                out_scale,
                                      uint32_t                K,
                                      uint32_t                N,
                                      uint32_t                super_group,
                                      uint32_t                group,
                                      dim3 &                  blocksPerGrid,
                                      dim3 &                  threadsPerBlock,
                                      std::vector<uint32_t> & params) {
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 16
    //----------------------------------------------------------------------------------------------------
    // in_super_scale   [K/256]  |     | [N]
    // in_super_scale   [sg]     |     | [N]
    // in_super_scale   [z]      | [1] | [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // qscale_per_word = 2
    // in_scale   [K/256]  | [256/16/qscale_per_word]  |  [qscale_per_word][N]
    // in_scale   [K/256]  | [256/32]                  |  [qscale_per_word][N]
    // in_scale   [z]      | [unroll]                  |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // out_scale   [K/256]  | [256/16]                 |  [N]
    // out_scale   [z]      | [unroll]                 |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    uint32_t qscale_per_word = 2;
    uint32_t inStrideY       = N;
    uint32_t inStrideZ       = super_group / group / qscale_per_word * N;
    uint32_t outStrideY      = N;
    uint32_t outStrideZ      = super_group / group * N;

    uint32_t nr_of_unroll       = super_group / group / qscale_per_word;
    uint32_t stride_per_unroll  = N * sizeof(short);
    uint32_t stride_per_block_x = threadsPerBlock.x * sizeof(short);
    //stride_per_block_x -= (nr_of_unroll - 1)* stride_per_unroll;

    uint32_t blockX = blocksPerGrid.x;
    blocksPerGrid.x = 1;
    uint32_t blockY = group / 2;
    params.emplace_back(in_scale);
    params.emplace_back(in_super_scale);
    params.emplace_back(out_scale);
    params.emplace_back(inStrideY);
    params.emplace_back(inStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(stride_per_unroll);
    params.emplace_back(stride_per_block_x);
    params.emplace_back(blockX);
    params.emplace_back(blockY);
}

static void matmul_weights_q6k_kernel_params(dim3 &                  blocksPerGrid,
                                             dim3 &                  threadsPerBlock,
                                             uint32_t                in_act,
                                             uint32_t                in_lsb,
                                             uint32_t                in_msb,
                                             uint32_t                in_scale,
                                             uint32_t                out_addr,
                                             uint32_t                lut_addr,
                                             uint32_t                zero_addr,
                                             uint32_t                input_acc_addr,
                                             uint32_t                input_acc_addr_hi,
                                             uint32_t                N,
                                             uint32_t                hilo_stride,
                                             uint32_t                weights_group,
                                             uint32_t                combine,
                                             std::vector<uint32_t> & params) {
    //----------------------------------------------------------------------------------------------------
    // in_lsb  [K/256][256/16][4][4][N]
    // in_lsb  [K/256]      | [256/elements_per_thread] | [elements_per_thread/wqlsb_per_word] | [wqlsb_per_word][N]
    // in_lsb  [1]          | [loop]                    | [unroll]                             | [x]
    //         [grid.z]     | [loop]                                                           | [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------
    // in_msb  [K/256][256/16][2][8][N]
    // in_msb  [K/256]      | [256/elements_per_thread] | [elements_per_thread/wqmsb_per_word] | [wqmsb_per_word][N]
    // in_msb  [1]          | [loop]                    | [unroll]                             | [x]
    //         [grid.z]     | [loop]                                                           | [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 16
    // in_scale   [K/256]  | [256/16]                 |  [N]
    // in_scale   [z]      | [loop]                   |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    int elements_per_thread = 16;
    int bits_per_wqlsb      = 4;
    int bits_per_wqmsb      = 2;
    int bits_per_qscale     = 8;
    int wqlsb_per_word      = sizeof(short) * 8 / bits_per_wqlsb;
    int wqmsb_per_word      = sizeof(short) * 8 / bits_per_wqmsb;
    int qscale_per_word     = sizeof(short) * 8 / bits_per_qscale;

    uint32_t inUnrollStride       = N * sizeof(short);
    uint32_t inLoopStride0        = elements_per_thread / wqlsb_per_word * N * sizeof(short);
    uint32_t inLoopStride1        = elements_per_thread / wqmsb_per_word * N * sizeof(short);
    uint32_t scaleLoopStride      = N * sizeof(short);
    uint32_t blockXSize           = threadsPerBlock.x * sizeof(short);
    uint32_t in_lsb_blockz_size   = weights_group / elements_per_thread * inLoopStride0;
    uint32_t in_msb_blockz_size   = weights_group / elements_per_thread * inLoopStride1;
    uint32_t in_scale_blockz_size = weights_group / elements_per_thread * scaleLoopStride;
    uint32_t in_a_blockz_size     = weights_group * sizeof(short);
    uint32_t loop                 = weights_group / elements_per_thread;

    params.emplace_back(in_act);
    params.emplace_back(in_lsb);
    params.emplace_back(in_msb);
    params.emplace_back(in_scale);
    params.emplace_back(out_addr);
    params.emplace_back(lut_addr);
    params.emplace_back(inUnrollStride);
    params.emplace_back(inLoopStride0);
    params.emplace_back(inLoopStride1);
    params.emplace_back(scaleLoopStride);
    params.emplace_back(blockXSize);
    params.emplace_back(in_msb_blockz_size);
    params.emplace_back(in_lsb_blockz_size);
    params.emplace_back(in_scale_blockz_size);
    params.emplace_back(in_a_blockz_size);
    params.emplace_back(zero_addr);
    params.emplace_back(hilo_stride);
    params.emplace_back(loop);
    params.emplace_back(combine);
}
}  // namespace kernel_q6_k_vxm
