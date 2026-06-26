#pragma once
#include "rpp_drv_api.h"

#include <rpp_runtime.h>

#include <cstdint>
#include <vector>

#define MATMUL_NS 128

namespace kernel_q2_s {

static void q2s_super_scale_params(uint32_t                in_scale,
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

static void q2s_dequant_params(dim3 &                  blocksPerGrid,
                               dim3 &                  threadsPerBlock,
                               uint32_t                in_wq_lsb,
                               uint32_t                in_wq_msb,
                               uint32_t                in_sign,
                               uint32_t                scale,
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

    (void) grid_z;
    (void) in_type_of_bytes;
    (void) out_type_of_bytes;
    //----------------------------------------------------------------------------------------------------
    // codebook_lsb   [K/256]      |  [4]  |  [4]       |  [2][N]
    // codebook_lsb   [z]          |  [y]  |  [unroll]  |  [x]
    //                [grid.y]*[z] |  [y]  |            |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // codebook_msb   [K/256]      |  [4]  |    [1]     |  [8][N]
    // codebook_msb   [z]          |  [y]  |    [1]     |  [x]
    //                [grid.y]*[z] |  [y]  |            |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // qsign      [K/256]       |  [4]  |  [4]       |  [16][N]
    // qsign      [z]           |  [y]  |  [unroll]  |  [x]
    //            [grid.y]*[z]  |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // scale      [K/256]       |  [4]  |  [4]       |  [N]
    // scale      [z]           |  [y]  |  [unroll]  |  [x]
    //            [grid.y]*[z]  |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // out        [K/256]       |  [4]  |  [4]       |  [16]        |  [N]
    // out        [z]           |  [y]  |  [unroll0] |  [unroll1]   |  [x]
    //            [grid.y]*[z]  |  [y]                              |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------

    //qsign/codebook_lsb/scale share same strideY & strideZ
    int inStrideY0 = 4 * column;
    int inStrideZ0 = 4 * inStrideY0;
    int inStrideY1 = 1 * column;
    int inStrideZ1 = 4 * inStrideY1;

    int in_unroll_stride = column * sizeof(short);
    int inBlockXSize0    = block_x * sizeof(short);
    inBlockXSize0 -= (4) * in_unroll_stride;
    int inBlockXSize1 = block_x * sizeof(short);

    int inBlockYSize0 = block_y * block_z * inStrideY0 * sizeof(short);
    inBlockYSize0 -= grid_x * block_x * sizeof(short);

    int inBlockYSize1 = block_y * block_z * inStrideY1 * sizeof(short);
    inBlockYSize1 -= grid_x * block_x * sizeof(short);
    //----------------------------------------------------------------------------------------------------
    // out        [K/256]       |  [4]  |  [4]       |  [16]        |  [N]
    // out        [z]           |  [y]  |  [unroll0] |  [unroll1]   |  [x]
    //            [grid.y]*[z]  |  [y]                              |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    int outStrideY        = 4 * 16 * block_x;
    int outStrideZ        = 4 * outStrideY;
    int out_unroll_stride = block_x * sizeof(short);
    int outBlockXSize     = block_x * row * sizeof(short);
    outBlockXSize -= 64 * out_unroll_stride;
    int outBlockYSize = 64 * block_x * block_y * block_z * sizeof(short);
    outBlockYSize -= grid_x * block_x * row * sizeof(short);
    ;

    //----------------------------------------------------------------------------------------------------
    // scale      [K/256]       |  [4]  |  [4]       |  [N]
    // scale      [z]           |  [y]  |  [unroll]  |  [x]
    //            [grid.y]*[z]  |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    int dequantStrideY    = 4 * column;
    int dequantStrideZ    = 4 * dequantStrideY;
    int deQuantBlockSizeX = block_x * sizeof(short);
    deQuantBlockSizeX -= (4) * in_unroll_stride;
    int deQuantBlockSizeY = dequantStrideY * block_z * block_y * sizeof(short);
    deQuantBlockSizeY -= grid_x * block_x * (int) sizeof(short);
    blocksPerGrid.x = 1;
    blocksPerGrid.y = 1;

    params.emplace_back(in_wq_lsb);
    params.emplace_back(in_wq_msb);
    params.emplace_back(in_sign);
    params.emplace_back(scale);
    params.emplace_back(output);
    params.emplace_back(lut_addr);
    params.emplace_back(inStrideY0);
    params.emplace_back(inStrideZ0);
    params.emplace_back(inStrideY1);
    params.emplace_back(inStrideZ1);
    params.emplace_back(dequantStrideY);
    params.emplace_back(dequantStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(in_unroll_stride);
    params.emplace_back(out_unroll_stride);
    params.emplace_back(inBlockXSize0);
    params.emplace_back(inBlockYSize0);
    params.emplace_back(inBlockXSize1);
    params.emplace_back(inBlockYSize1);
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
    const int32_t block_x = (int32_t) threadsPerBlock.x;
    const int32_t block_y = (int32_t) threadsPerBlock.y;
    (void) blocksPerGrid;

    uint32_t in0_col_round = (in0_col + 31) / 32 * 32;
    uint32_t in1_row_round = (in1_row + 31) / 32 * 32;
    if (in1_col == 32 && (in1_row % 32) > 0) {
        in1_row_round = in1_row;
    }
    uint32_t in1_col_round = (in1_col + 31) / 32 * 32;

    int inStrideY  = block_x * in_type_of_bytes;
    int outStrideY = block_x;
    if (use_hw_output) {
        outStrideY = (int) in1_col;
    }

    int inStrideZ  = 0;
    int outStrideZ = 0;
    int cn         = (int) ((in0_col_round + 31) / 32) - 1;
    int inSwitchSize =
        ((int) (in0_row - 1) * block_x + (in_type_of_bytes == (int) sizeof(short) ? 8 : 16)) * in_type_of_bytes;

    input_b += in1_row_round * block_x * in_type_of_bytes * (uint32_t) tn_offset;
    out += in0_row * block_x * out_type_of_bytes * (uint32_t) tn_offset;

    int gridx_inb_stride = in1_row_round * block_x * in_type_of_bytes * tn;
    int gridx_out_stride =
        use_hw_output ? block_x * out_type_of_bytes * tn : in0_row * block_x * out_type_of_bytes * tn;
    int gridy_ina_stride = block_x * block_y * in_type_of_bytes;
    int gridy_out_stride = block_x * block_y * out_type_of_bytes;

    int gridz_ina_stride = (loop_in0 == 1) ? 0 : (int) (in0_row * in0_col_round * in_type_of_bytes);
    int gridz_inb_stride = (loop_in1 == 1) ? 0 : (int) (in1_row_round * in1_col_round * in_type_of_bytes);
    int gridz_out_stride = (loop_out == 1) ? 0 : (int) (in0_row * in1_col_round * out_type_of_bytes);
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
        int outTnStride   = use_hw_output ? (int) (32 * sizeof(short)) : (int) (in0_row * 32 * sizeof(short));
        int filterOffset0 = (int) (in0_col_round * 32 * sizeof(short));
        int filterOffset1 = (int) (in0_col_round * 32 * sizeof(short) * (tn - 1) - 512);
        params.emplace_back((uint32_t) outTnStride);
        params.emplace_back((uint32_t) filterOffset0);
        params.emplace_back((uint32_t) filterOffset1);
    }
}
}  // namespace kernel_q2_s
