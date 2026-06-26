
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

namespace kernel_q4_k_vxm {

static void q4k_super_scale_params(uint32_t                in_scale_lsb,
                                   uint32_t                in_scale_msb,
                                   uint32_t                in_super_scale,
                                   uint32_t                out_scale,
                                   uint32_t                K,
                                   uint32_t                N,
                                   uint32_t                super_group,
                                   uint32_t                q_group,
                                   int                     is_zero,
                                   dim3 &                  blocksPerGrid,
                                   dim3 &                  threadsPerBlock,
                                   std::vector<uint32_t> & params) {
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    //----------------------------------------------------------------------------------------------------
    // in_super_scale   [K/256]  |     | [N]
    // in_super_scale   [sg]     |     | [N]
    // in_super_scale   [z]      | [1] | [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // out_scale   [K/256]     | [8]                  |  [N]
    // out_scale   [z]         | [8]                  |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // qscale_per_word = 4
    // in_scale_lsb   [K/256]  | [2]                  |  [qscale_per_word][N]
    // in_scale_lsb   [K/256]  | [2]                  |  [qscale_per_word][N]
    // in_scale_lsb   [z]      | [unroll]             |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // qscale_per_word = 8
    // in_scale_msb   [K/256]  | [1]                  |  [qscale_per_word][N]
    // in_scale_msb   [K/256]  | [1]                  |  [qscale_per_word][N]
    // in_scale_msb   [z]      | [1]                  |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    uint32_t inStrideY          = N;
    uint32_t inStrideZ          = 2 * N;
    uint32_t outStrideY         = N;
    uint32_t outStrideZ         = 8 * N;
    uint32_t inUnRollStride     = N * sizeof(short);
    uint32_t outUnRollStride    = N * sizeof(short);
    uint32_t stride_per_block_x = threadsPerBlock.x * sizeof(short);

    uint16_t sign = 0x3f80;
    if (is_zero) {
        sign = 0xbf80;
    }

    float    aaa    = -1.0f;
    // Let the kernel handle x-chunks internally via blockX. This matches the
    // standalone UT's fixed path for wide-N shapes such as N=12288.
    uint32_t blockX = blocksPerGrid.x;
    blocksPerGrid.x = 1;
    params.emplace_back(in_scale_lsb);
    params.emplace_back(in_scale_msb);
    params.emplace_back(in_super_scale);
    params.emplace_back(out_scale);
    params.emplace_back(inStrideY);
    params.emplace_back(inStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(inUnRollStride);
    params.emplace_back(outUnRollStride);
    params.emplace_back(stride_per_block_x);
    params.emplace_back(blockX);
    params.emplace_back(sign);
}

static void matmul_weights_q4k_kernel_params(dim3 &                  blocksPerGrid,
                                             dim3 &                  threadsPerBlock,
                                             uint32_t                in_act,
                                             uint32_t                in_lsb,
                                             uint32_t                in_scale,
                                             uint32_t                in_zero,
                                             uint32_t                out_addr,
                                             uint32_t                lut_addr,
                                             uint32_t                input_acc_addr,
                                             uint32_t                input_acc_addr_hi,
                                             uint32_t                N,
                                             uint32_t                hilo_stride,
                                             uint32_t                weights_group,
                                             uint32_t                combine,
                                             std::vector<uint32_t> & params) {
    //-----------------------------------------------------------------------------------------------------
    // in_scale   [K/256]   | [8]       | [N]
    // in_scale   [grid.z]  | [loop]    | [x]
    //-----------------------------------------------------------------------------------------------------
    // in_lsb  [K/256]      | [8]      | [32/4]         | [4][N]
    // in_lsb  [1]          | [loop]   | [unroll]       | [x]
    //         [grid.z]     | [loop]                    | [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    int      elements_per_thread = 32;
    int      bits_per_wqlsb      = 4;
    int      wqlsb_per_word      = sizeof(short) * 8 / bits_per_wqlsb;
    uint32_t inUnrollStride      = N * sizeof(short);
    uint32_t blockXSize          = threadsPerBlock.x * sizeof(short);
    uint32_t scaleLoopStride     = N * sizeof(short);

    uint32_t inLoopStride0        = elements_per_thread / wqlsb_per_word * N * sizeof(short);
    uint32_t in_lsb_blockz_size   = weights_group / elements_per_thread * inLoopStride0;
    uint32_t in_scale_blockz_size = weights_group / elements_per_thread * scaleLoopStride;
    uint32_t in_a_blockz_size     = weights_group * sizeof(short);
    uint32_t loop                 = weights_group / elements_per_thread;

    params.emplace_back(in_act);
    params.emplace_back(in_lsb);
    params.emplace_back(in_scale);
    params.emplace_back(in_zero);
    params.emplace_back(out_addr);
    params.emplace_back(lut_addr);
    params.emplace_back(inUnrollStride);
    params.emplace_back(inLoopStride0);
    ;
    params.emplace_back(scaleLoopStride);
    params.emplace_back(blockXSize);
    params.emplace_back(in_lsb_blockz_size);
    params.emplace_back(in_scale_blockz_size);
    params.emplace_back(in_a_blockz_size);
    params.emplace_back(input_acc_addr);
    params.emplace_back(input_acc_addr_hi);
    params.emplace_back(hilo_stride);
    params.emplace_back(loop);
    params.emplace_back(combine);
}

static void matmul_weights_q4k_step1_kernel_params(dim3 &                  blocksPerGrid,
                                                   dim3 &                  threadsPerBlock,
                                                   uint32_t                in_act,
                                                   uint32_t                in_lsb,
                                                   uint32_t                out_lsb,
                                                   uint32_t                out_msb,
                                                   uint32_t                input_acc_addr,
                                                   uint32_t                input_acc_addr_hi,
                                                   uint32_t                N,
                                                   uint32_t                hilo_offset,
                                                   uint32_t                weights_group,
                                                   uint32_t                combine,
                                                   std::vector<uint32_t> & params) {
    (void) blocksPerGrid;
    int      elements_per_thread = 32;
    int      bits_per_wqlsb      = 4;
    int      wqlsb_per_word      = sizeof(short) * 8 / bits_per_wqlsb;
    uint32_t inUnrollStride      = N * sizeof(short);
    uint32_t outStrideY          = N;
    uint32_t outStrideZ          = (weights_group / elements_per_thread) * N;
    uint32_t blockXSize          = threadsPerBlock.x * sizeof(short);
    uint32_t in_lsb_blockz_size  = (weights_group / elements_per_thread) *
                                  (elements_per_thread / wqlsb_per_word) * N * sizeof(short);
    uint32_t out_blockz_size     = weights_group / elements_per_thread * N * sizeof(short);
    uint32_t in_a_blockz_size    = weights_group * sizeof(short);
    uint32_t act_stridey         = (weights_group / (weights_group / elements_per_thread)) * sizeof(short);
    uint32_t in_lsb_stridey      = (elements_per_thread / wqlsb_per_word) * N;

    assert(act_stridey <= UINT16_MAX);
    assert(in_lsb_stridey <= UINT16_MAX);

    params.emplace_back(in_act);
    params.emplace_back(in_lsb);
    params.emplace_back(out_lsb);
    params.emplace_back(out_msb);
    params.emplace_back(inUnrollStride);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(in_lsb_blockz_size);
    params.emplace_back(out_blockz_size);
    params.emplace_back(in_a_blockz_size);
    params.emplace_back(input_acc_addr);
    params.emplace_back(input_acc_addr_hi);
    params.emplace_back(hilo_offset);
    params.emplace_back(blockXSize);
    params.emplace_back(static_cast<uint16_t>(act_stridey));
    params.emplace_back(static_cast<uint16_t>(in_lsb_stridey));
    params.emplace_back(static_cast<uint16_t>(combine));
}

static void matmul_weights_q4k_step2_kernel_params(dim3 &                  blocksPerGrid,
                                                   dim3 &                  threadsPerBlock,
                                                   uint32_t                in_step1_hi,
                                                   uint32_t                in_step1_lo,
                                                   uint32_t                in_scale,
                                                   uint32_t                in_zero,
                                                   uint32_t                out_addr,
                                                   uint32_t                input_acc_addr,
                                                   uint32_t                input_acc_addr_hi,
                                                   uint32_t                N,
                                                   uint32_t                hilo_stride,
                                                   uint32_t                weights_group,
                                                   uint32_t                combine,
                                                   std::vector<uint32_t> & params) {
    (void) blocksPerGrid;
    int      elements_per_thread  = 32;
    int      bits_per_wqlsb       = 4;
    int      wqlsb_per_word       = sizeof(short) * 8 / bits_per_wqlsb;
    uint32_t inUnrollStride       = N * sizeof(short);
    uint32_t scaleLoopStride      = N * sizeof(short);
    uint32_t inLoopStride0        = elements_per_thread / wqlsb_per_word * N * sizeof(short);
    uint32_t in_scale_blockz_size = weights_group / elements_per_thread * scaleLoopStride;
    uint16_t loop                 = static_cast<uint16_t>(weights_group / elements_per_thread);

    params.emplace_back(in_step1_hi);
    params.emplace_back(in_step1_lo);
    params.emplace_back(in_scale);
    params.emplace_back(in_zero);
    params.emplace_back(out_addr);
    params.emplace_back(inUnrollStride);
    params.emplace_back(inLoopStride0);
    params.emplace_back(scaleLoopStride);
    params.emplace_back(in_scale_blockz_size);
    params.emplace_back(input_acc_addr);
    params.emplace_back(input_acc_addr_hi);
    params.emplace_back(hilo_stride);
    params.emplace_back(loop);
    params.emplace_back(static_cast<uint16_t>(combine));
}
}  // namespace kernel_q4_k_vxm
