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

namespace kernel_bf16 {
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
    const int32_t grid_x = (int32_t) blocksPerGrid.x;
    const int32_t grid_y = (int32_t) blocksPerGrid.y;
    const int32_t grid_z = (int32_t) blocksPerGrid.z;

    const int32_t block_x = (int32_t) threadsPerBlock.x;
    const int32_t block_y = (int32_t) threadsPerBlock.y;
    const int32_t block_z = (int32_t) threadsPerBlock.z;

    (void) grid_x;
    (void) grid_y;
    (void) grid_z;
    (void) block_z;

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

    input_b += in1_row_round * block_x * in_type_of_bytes * (uint32_t) tn_offset;
    out += in0_row * block_x * out_type_of_bytes * (uint32_t) tn_offset;

    gridx_inb_stride = in1_row_round * block_x * in_type_of_bytes * tn;

    if (use_hw_output) {
        gridx_out_stride = block_x * out_type_of_bytes * tn;
    } else {
        gridx_out_stride = in0_row * block_x * out_type_of_bytes * tn;
    }

    gridy_ina_stride = block_x * block_y * in_type_of_bytes;
    gridy_out_stride = block_x * block_y * out_type_of_bytes;

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
}  // namespace kernel_bf16
