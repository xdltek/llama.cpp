
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

namespace kernel_q2_xs_vxm {
static void q2xs_super_scale_params(uint32_t                in_scale,
                                    uint32_t                in_super_scale,
                                    uint32_t                in_lut,
                                    uint32_t                out_scale,
                                    uint32_t                K,
                                    uint32_t                N,
                                    uint32_t                super_group,
                                    uint32_t                q_group,
                                    dim3 &                  blocksPerGrid,
                                    dim3 &                  threadsPerBlock,
                                    std::vector<uint32_t> & params) {
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    //----------------------------------------------------------------------------------------------------
    // in_super_scale   [K/256]  |     | [N]
    // in_super_scale   [sg]     |     | [N]
    // in_super_scale   [z]      | [1] | [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // out_scale   [K/256]     | [16]     |  [N]
    // out_scale   [z]         | [16]     |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // in_scale   [K/256]  | [4]          |  [4][N]
    // in_scale   [K/256]  | [4]          |  [4][N]
    // in_scale   [z]      | [unroll]     |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    uint32_t inStrideY          = N;
    uint32_t inStrideZ          = 4 * N;
    uint32_t outStrideY         = N;
    uint32_t outStrideZ         = 16 * N;
    uint32_t inUnRollStride     = N * sizeof(short);
    uint32_t outUnRollStride    = N * sizeof(short);
    uint32_t stride_per_block_x = threadsPerBlock.x * sizeof(short);

    uint32_t blockX = blocksPerGrid.x;
    blocksPerGrid.x = 1;
    params.emplace_back(in_scale);
    params.emplace_back(in_super_scale);
    params.emplace_back(out_scale);
    params.emplace_back(in_lut);
    params.emplace_back(inStrideY);
    params.emplace_back(inStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(inUnRollStride);
    params.emplace_back(outUnRollStride);
    params.emplace_back(stride_per_block_x);
    params.emplace_back(blockX);
}

static void matmul_weights_q2xs_kernel_params(dim3 &                  blocksPerGrid,
                                              dim3 &                  threadsPerBlock,
                                              uint32_t                in_act,
                                              uint32_t                in_wq,
                                              uint32_t                in_scale,
                                              uint32_t                out_addr,
                                              uint32_t                lut_codebook,
                                              uint32_t                lut_sign,
                                              uint32_t                input_acc_addr,
                                              uint32_t                input_acc_addr_hi,
                                              uint32_t                N,
                                              uint32_t                hilo_stride,
                                              uint32_t                weights_group,
                                              uint32_t                combine,
                                              std::vector<uint32_t> & params) {
    //-----------------------------------------------------------------------------------------------------
    // in_scale   [K/256]   | [16]       | [N]
    // in_scale   [grid.z]  | [loop]     | [x]
    //-----------------------------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------------------------
    // qs   [K/256]  | [16]   |  [2]       | [8][N]
    // qs   [grid.z] | [loop] |  [unroll0] | [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    uint32_t inUnrollStride  = N * sizeof(short);
    uint32_t blockXSize      = threadsPerBlock.x * sizeof(short);
    uint32_t scaleLoopStride = N * sizeof(short);
    uint32_t inLoopStride0   = 2 * N * sizeof(short);

    uint32_t in_wq_blockz_size    = 16 * inLoopStride0;
    uint32_t in_scale_blockz_size = 16 * scaleLoopStride;
    uint32_t in_a_blockz_size     = weights_group * sizeof(short);
    uint32_t loop                 = 16;

    params.emplace_back(in_act);
    params.emplace_back(in_wq);
    params.emplace_back(in_scale);
    params.emplace_back(out_addr);
    params.emplace_back(lut_codebook);
    params.emplace_back(lut_sign);
    params.emplace_back(inUnrollStride);
    params.emplace_back(inLoopStride0);
    params.emplace_back(scaleLoopStride);
    params.emplace_back(blockXSize);
    params.emplace_back(in_wq_blockz_size);
    params.emplace_back(in_scale_blockz_size);
    params.emplace_back(in_a_blockz_size);
    params.emplace_back(input_acc_addr);
    params.emplace_back(input_acc_addr_hi);
    params.emplace_back(hilo_stride);
    params.emplace_back(loop);
    params.emplace_back(combine);
}
}  // namespace kernel_q2_xs_vxm
