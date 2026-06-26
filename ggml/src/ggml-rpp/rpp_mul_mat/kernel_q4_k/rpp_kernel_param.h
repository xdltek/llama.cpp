
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

#define MATMUL_NS 128

namespace kernel_q4_k {

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

static void q4k_dequant_params(dim3 &                  blocksPerGrid,
                               dim3 &                  threadsPerBlock,
                               uint32_t                in_wq,
                               uint32_t                scale,
                               uint32_t                zero,
                               uint32_t                output,
                               uint32_t                lut_addr,
                               int32_t                 column,
                               int32_t                 row,
                               int                     in_type_of_bytes,
                               int                     out_type_of_bytes,
                               std::vector<uint32_t> & params) {
    const int32_t grid_x = (int32_t) blocksPerGrid.x;
    const int32_t grid_y = (int32_t) blocksPerGrid.y;
    const int32_t grid_z = (int32_t) blocksPerGrid.z;

    const int32_t block_x = (int32_t) threadsPerBlock.x;
    const int32_t block_y = (int32_t) threadsPerBlock.y;
    const int32_t block_z = (int32_t) threadsPerBlock.z;

    int super_group    = 256;
    int bits_per_wqlsb = 4;
    int wqlsb_per_word = sizeof(short) * 8 / bits_per_wqlsb;
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // in_wq     [K/256]      |  [8]  |  [32/4]    |  [4][N]
    // in_wq     [z]          |  [y]  |  [unroll]  |  [x]
    //           [grid.y]*[z] |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    int wq_per_word    = 4;
    int nr_of_unroll   = 32 / wq_per_word;
    int inStrideY0     = nr_of_unroll * column;
    int inStrideZ0     = 8 * inStrideY0;
    assert(inStrideZ0 == (super_group / 4 * column));
    int in_unroll_stride = column * sizeof(short);
    int inBlockXSize0    = block_x * sizeof(short);
    inBlockXSize0 -= nr_of_unroll * in_unroll_stride;

    int inBlockYSize0 = block_y * block_z * (nr_of_unroll) *column * sizeof(short);
    inBlockYSize0 -= grid_x * block_x * sizeof(short);
    //----------------------------------------------------------------------------------------------------
    // out_addr   [N/32]   | [K/256]      | [8]     | [32/4]    | [4]        | [32]
    // out_addr            | [z]          | [y]     | [unroll0] | [unroll1]  | [x]
    //            [grid.x] | [grid.y]*[z] | [y]                              | [x]
    //-----------------------------------------------------------------------------------------------------
    int outStrideY        = nr_of_unroll * 4 * block_x;
    int outStrideZ        = 8 * outStrideY;
    int out_unroll_stride = block_x * sizeof(short);
    int outBlockXSize     = block_x * row * sizeof(short);
    outBlockXSize -= 4 * nr_of_unroll * out_unroll_stride;
    int outBlockYSize = nr_of_unroll * 4 * block_x * block_y * block_z * sizeof(short);
    outBlockYSize -= grid_x * block_x * row * sizeof(short);

    //-----------------------------------------------------------------------------------------------------
    // in_scale   [K/256]      | [8]    | [N]
    // in_scale   [z]          | [y]    | [x]
    //            [grid.y]*[z] | [y]    | [grid.x]*[x]
    //-----------------------------------------------------------------------------------------------------
    int dequantStrideY    = column;
    int dequantStrideZ    = 8 * dequantStrideY;
    int deQuantBlockSizeX = block_x * sizeof(short);
    int deQuantBlockSizeY = column * block_z * block_y * sizeof(short);
    deQuantBlockSizeY -= grid_x * block_x * sizeof(short);
    blocksPerGrid.x = 1;
    blocksPerGrid.y = 1;

    params.emplace_back(in_wq);
    params.emplace_back(scale);
    params.emplace_back(zero);
    params.emplace_back(output);
    params.emplace_back(lut_addr);
    params.emplace_back(inStrideY0);
    params.emplace_back(inStrideZ0);
    params.emplace_back(dequantStrideY);
    params.emplace_back(dequantStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(in_unroll_stride);
    params.emplace_back(out_unroll_stride);
    params.emplace_back(inBlockXSize0);
    params.emplace_back(inBlockYSize0);
    params.emplace_back(deQuantBlockSizeX);
    params.emplace_back(deQuantBlockSizeY);
    params.emplace_back(outBlockXSize);
    params.emplace_back(outBlockYSize);
    params.emplace_back(grid_x);
    params.emplace_back(grid_y);
}

static void matmul_opt_kernel_params(const dim3 &            blocksPerGrid,
                                     const dim3 &            threadsPerBlock,
                                     uint32_t                input_a,
                                     uint32_t                input_b,
                                     uint32_t                postScale,
                                     uint32_t                out,
                                     uint32_t                in0_row,
                                     uint32_t                in0_col,
                                     uint32_t                in1_row,
                                     uint32_t                in1_col,
                                     uint32_t                loop_in0,
                                     uint32_t                loop_in1,
                                     uint32_t                loop_out,
                                     int32_t                 tn_offset,
                                     int32_t                 tn,
                                     int                     in_type_of_bytes,
                                     int                     out_type_of_bytes,
                                     std::vector<uint32_t> & params,
                                     bool                    use_hw_output,
                                     bool                    is_expand) {
    // Extract launch dims (kept for readability / minimal code changes)
    const int32_t grid_x = (int32_t) blocksPerGrid.x;
    const int32_t grid_y = (int32_t) blocksPerGrid.y;
    const int32_t grid_z = (int32_t) blocksPerGrid.z;

    const int32_t block_x = (int32_t) threadsPerBlock.x;
    const int32_t block_y = (int32_t) threadsPerBlock.y;
    const int32_t block_z = (int32_t) threadsPerBlock.z;

    (void) grid_x;
    (void) grid_y;
    (void) grid_z;
    (void) block_z;  // not all are used below

    uint32_t in0_col_round = (in0_col + 31) / 32 * 32;
    uint32_t in1_row_round = (in1_row + 31) / 32 * 32;
    if (in1_col == 32 && (in1_row % 32) > 0) {
        in1_row_round = in1_row;
    }
    uint32_t in1_col_round = (in1_col + 31) / 32 * 32;

    int gridx_inb_stride, gridx_out_stride;
    int gridz_ina_stride, gridz_inb_stride, gridz_out_stride;
    int gridy_ina_stride, gridy_out_stride;
    int cn;

    int inStrideY  = block_x * in_type_of_bytes;
    int outStrideY = block_x;
    if (use_hw_output) {
        outStrideY = (int) in1_col;
    }

    int inStrideZ  = 0;
    int outStrideZ = 0;
    int inSwitchSize;

    cn = (int) ((in0_col_round + 31) / 32) - 1;

    if (in_type_of_bytes == (int) sizeof(short)) {
        inSwitchSize = ((int) (in0_row - 1) * block_x + 8) * in_type_of_bytes;
    } else {
        inSwitchSize = ((int) (in0_row - 1) * block_x + 16) * in_type_of_bytes;
    }

    // Apply tn_offset to B and output pointer
    input_b += in1_row_round * block_x * in_type_of_bytes * (uint32_t) tn_offset;
    out += in0_row * block_x * out_type_of_bytes * (uint32_t) tn_offset;

    // Strides between grid_x tiles
    gridx_inb_stride = in1_row_round * block_x * in_type_of_bytes * tn;

    if (use_hw_output) {
        gridx_out_stride = block_x * out_type_of_bytes * tn;
    } else {
        gridx_out_stride = in0_row * block_x * out_type_of_bytes * tn;
    }

    // Strides between grid_y partitions
    gridy_ina_stride = block_x * block_y * in_type_of_bytes;
    gridy_out_stride = block_x * block_y * out_type_of_bytes;

    // Strides between grid_z loops
    if (loop_in0 == 1) {
        gridz_ina_stride = 0;
    } else {
        gridz_ina_stride = (int) (in0_row * in0_col_round * in_type_of_bytes);
    }

    if (loop_in1 == 1) {
        gridz_inb_stride = 0;
    } else {
        gridz_inb_stride = (int) (in1_row_round * in1_col_round * in_type_of_bytes);
    }

    if (loop_out == 1) {
        gridz_out_stride = 0;
    } else {
        gridz_out_stride = (int) (in0_row * in1_col_round * out_type_of_bytes);
    }

    if (is_expand) {
        gridz_inb_stride = 0;
    }

    uint16_t out_type = (uint16_t) (out_type_of_bytes == (int) sizeof(short));

    // Pack params
    params.emplace_back(input_a);
    params.emplace_back(input_b);
    params.emplace_back(out);
    params.emplace_back((uint32_t) cn);
    params.emplace_back((uint32_t) 0);
    params.emplace_back((uint32_t) inStrideY);
    params.emplace_back((uint32_t) outStrideY);
    params.emplace_back((uint32_t) inStrideZ);
    params.emplace_back((uint32_t) outStrideZ);
    params.emplace_back((uint32_t) inSwitchSize);
    params.emplace_back((uint32_t) gridx_inb_stride);
    params.emplace_back((uint32_t) gridx_out_stride);
    params.emplace_back((uint32_t) gridy_ina_stride);
    params.emplace_back((uint32_t) gridy_out_stride);
    params.emplace_back((uint32_t) gridz_ina_stride);
    params.emplace_back((uint32_t) gridz_inb_stride);
    params.emplace_back((uint32_t) gridz_out_stride);

    // Extra params when tn > 1
    if (tn > 1) {
        int outTnStride = (int) (in0_row * 32 * sizeof(short));
        if (use_hw_output) {
            outTnStride = (int) (32 * sizeof(short));
        }

        int filterOffset0 = (int) (in0_col_round * 32 * sizeof(short));
        int filterOffset1 = (int) (in0_col_round * 32 * sizeof(short) * (tn - 1) - 512);

        params.emplace_back((uint32_t) outTnStride);
        params.emplace_back((uint32_t) filterOffset0);
        params.emplace_back((uint32_t) filterOffset1);
    }
}
}  // namespace kernel_q4_k
