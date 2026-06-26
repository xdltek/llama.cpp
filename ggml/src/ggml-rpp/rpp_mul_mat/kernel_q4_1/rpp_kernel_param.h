
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
namespace kernel_q4_1 {
#define MATMUL_NS 128

static void matmul_dequant_params(uint32_t                weight_in_addr,
                                  uint32_t                dequant_addr,
                                  uint32_t                zero_point_addr,
                                  uint32_t                weights_out_addr,
                                  uint32_t                lut_addr,
                                  int32_t                 column,
                                  int32_t                 row,
                                  int32_t                 block_x,
                                  int32_t                 block_y,
                                  int32_t                 block_z,
                                  int32_t                 grid_x,
                                  int32_t                 grid_y,
                                  int32_t                 grid_z,
                                  int                     in_type_of_bytes,
                                  int                     out_type_of_bytes,
                                  std::vector<uint32_t> & params) {
    uint32_t inStrideY, inStrideZ;
    uint32_t deQuantInStrideY, deQuantInStrideZ;
    uint32_t outStrideY, outStrideZ;
    uint32_t inBlockXSize, inBlockYSize;
    uint32_t outBlockXSize, outBlockYSize;
    uint32_t deQuantBlockSizeX, deQuantBlockSizeY;
    uint32_t out_stride;

    //output format N [r,32], N = n/32
    //dim.x = 32 , dim.y = 16, dim.z = 8
    //grid.x = N, grid.y = r/(y*z) = y/128
    //out stride = column size of output  [r,32];
    out_stride = block_x * in_type_of_bytes;
    //4 bits
    inStrideY  = column;
    inStrideZ  = block_y * column;

    deQuantInStrideY = 0;
    deQuantInStrideZ = column;
    //each y will store into differnt block [r,32]
    outStrideY       = 4 * block_x;
    outStrideZ       = 4 * block_x * block_y;
    //input block stride
    inBlockXSize     = block_x * sizeof(short);
    inBlockYSize     = block_y * block_z * column * sizeof(short);

    // output block size, one input 4 outputs
    outBlockXSize     = block_x * row * in_type_of_bytes;
    outBlockYSize     = 4 * block_x * block_y * block_z * in_type_of_bytes;
    //bfloat 16
    deQuantBlockSizeX = block_x * sizeof(short);
    deQuantBlockSizeY = column * block_z * sizeof(short);

    outBlockXSize -= 3 * out_stride;
    inBlockYSize -= grid_x * inBlockXSize;
    deQuantBlockSizeY -= grid_x * deQuantBlockSizeX;
    outBlockYSize -= grid_x * (block_x * row * in_type_of_bytes);

    params.emplace_back(weight_in_addr);
    params.emplace_back(dequant_addr);
    params.emplace_back(weights_out_addr);
    params.emplace_back(out_stride);
    params.emplace_back(lut_addr);
    params.emplace_back(inStrideY);
    params.emplace_back(inStrideZ);
    params.emplace_back(deQuantInStrideY);
    params.emplace_back(deQuantInStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(inBlockXSize);
    params.emplace_back(inBlockYSize);
    params.emplace_back(deQuantBlockSizeX);
    params.emplace_back(deQuantBlockSizeY);
    params.emplace_back(outBlockXSize);
    params.emplace_back(outBlockYSize);
    params.emplace_back(zero_point_addr);
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
}