
#pragma once
#include "rpp_drv_api.h"
#include "rpp_flash_attn_ext/src/rpp_kernel_block.h"

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

inline std::string get_matmul_kernel(int N) {
    std::string kernel_name;
    if ((N % 128) == 0) {
        kernel_name = "matmul_tn4_f16_f32_f16";
    } else if ((N % 96) == 0) {
        kernel_name = "matmul_tn3_f16_f32_f16";
    } else if ((N % 64) == 0) {
        kernel_name = "matmul_tn2_f16_f32_f16";
    } else {
        kernel_name = "matmul_tn1_f16_f32_f16";
    }
    return kernel_name;
}

inline std::string get_expand_batch_vxM_kernel(int expand) {
    switch (expand) {
        case 4:
            return "expand_matrix_batch_vxM_f16_f16_f32_dyn";
        case 2:
            return "expand_2_matrix_batch_vxM_f16_f16_f32_dyn";
        case 3:
            return "expand_3_matrix_batch_vxM_f16_f16_f32_dyn";
        case 6:
            return "expand_6_matrix_batch_vxM_f16_f16_f32_dyn";
        case 7:
            return "expand_7_matrix_batch_vxM_f16_f16_f32_dyn";
        case 8:
            return "expand_8_matrix_batch_vxM_f16_f16_f32_dyn";
        case 1:
            return "expand_1_matrix_batch_vxM_f16_f16_f32_dyn";
        default:
            throw std::runtime_error("No kernel function supported yet.");
    }
}

inline std::string get_expand_batch_hkv_vxM_kernel() {
    return "expand_matrix_batch_hkv_vxM_f16_f16_f32_dyn";
}

static void KParam_TransPadding(dim3 &                  threadsPerBlock,
                                dim3 &                  blocksPerGrid,
                                uint32_t                input0,
                                uint32_t                output0,
                                int                     w,
                                std::vector<uint32_t> & params) {
    params.clear();
    int nBlockX, nBlockY, padStride, inBlkStride, outYStride;
    nBlockX         = blocksPerGrid.x;
    nBlockY         = blocksPerGrid.y;
    blocksPerGrid.y = 1;
    blocksPerGrid.x = 1;
    inBlkStride     = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * 2;
    padStride       = threadsPerBlock.y * threadsPerBlock.z * 2;
    outYStride      = (w + 1);
    params.push_back((uint32_t) (input0));
    params.push_back((uint32_t) (output0));
    params.push_back(inBlkStride);
    params.push_back(padStride);
    params.push_back(outYStride);
    params.push_back(1);
    params.push_back(nBlockX);
    params.push_back(nBlockY);
    if (outYStride >= 0xffff) {
        throw std::runtime_error("Tranpose Padding is greater than 65535");
    }
    return;
}

static void KParam_Trans(dim3 &                  threadsPerBlock,
                         dim3 &                  blocksPerGrid,
                         uint32_t                input,
                         uint32_t                output0,
                         int                     h,
                         int                     w,
                         std::vector<uint32_t> & params) {
    params.clear();
    int colSegLenth = w;
    int rowSegLenth = h;
    int outputPad   = 0;
    int stride0     = 8;

    uint32_t src_x_stribe    = colSegLenth * sizeof(uint16_t) + 2;
    uint32_t src_y_stribe    = (rowSegLenth) *stride0;
    uint32_t load_jump_size1 = (threadsPerBlock.z - 1) * 8 * 2;
    uint32_t load_jump_size2 = (threadsPerBlock.x - 1) * src_x_stribe + (threadsPerBlock.z * 8 - 7) * 2;

    uint32_t store_jump_size = (threadsPerBlock.z * stride0 - stride0) * rowSegLenth * sizeof(uint16_t);
    uint32_t loop_row        = rowSegLenth / 32;
    uint32_t loop_block      = colSegLenth / (threadsPerBlock.z * stride0);
    if (loop_block == 0 || loop_row == 0) {
        throw std::runtime_error("Tranpose not support loop_block == 0 || loop_row == 0");
    }
    params.push_back((uint32_t) (input));
    params.push_back((uint32_t) (output0));
    params.push_back(src_x_stribe);
    params.push_back(src_y_stribe);
    params.push_back(load_jump_size1);
    params.push_back(load_jump_size2);
    params.push_back(store_jump_size);
    params.push_back(loop_row);
    params.push_back(loop_block);
    params.push_back(outputPad);

    return;
}

static void KParam_Hw2Hw32(dim3 &                  threadsPerBlock,
                           dim3 &                  blocksPerGrid,
                           uint32_t                input,
                           uint32_t                output,
                           int                     H,
                           int                     W,
                           std::vector<uint32_t> & params) {
    params.clear();
    params.push_back(input);
    params.push_back(output);
    params.push_back(W);
    params.push_back(threadsPerBlock.x);
    params.push_back(threadsPerBlock.x);
    params.push_back(H * threadsPerBlock.x);
    params.push_back(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(uint16_t));
    params.push_back(threadsPerBlock.x * threadsPerBlock.y * sizeof(uint16_t));
    return;
}

static void KParam_Scale(dim3 &                  threadsPerBlock,
                         dim3 &                  blocksPerGrid,
                         uint32_t                input,
                         uint32_t                output,
                         float                   scale,
                         std::vector<uint32_t> & params) {
    params.clear();
    params.push_back(input);
    params.push_back(*(uint32_t *) &scale);
    params.push_back(output);
    params.push_back(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(uint16_t));
    return;
}

static void matmul_chw32_blocks(uint32_t loop,
                                uint32_t row,
                                uint32_t column,
                                dim3 &   threadsPerBlock,
                                dim3 &   blocksPerGrid,
                                int      tn,
                                int      is_tail) {
    if (loop > 65535) {
        assert(0);
    }
    // Defaults
    uint32_t bx = DIM_X;
    uint32_t by = row;
    uint32_t bz = 1;
    uint32_t gy = 1;
    if (row > MAX_EXEC) {
        while (true) {
            by = (by + 1) / 2;
            gy *= 2;
            if (by < MAX_EXEC) {
                break;
            }
        }
        grid_dim_adjustment(gy, by, row);
    }
    int      nr_of_tn = (column + 31) / 32;
    uint32_t gx       = 0;
    if (is_tail == 0) {
        gx = (uint32_t) (nr_of_tn / tn);
    } else {
        gx = 1;
    }
    threadsPerBlock.x = bx;
    threadsPerBlock.y = by;
    threadsPerBlock.z = bz;
    blocksPerGrid.x   = gx;
    blocksPerGrid.y   = gy;
    blocksPerGrid.z   = loop;
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

static void create_mm_chw32_unroll(int                     a_row,
                                   int                     a_col,
                                   int                     b_row,
                                   int                     b_col,
                                   uint32_t                in_a_addr,
                                   uint32_t                in_b_addr,
                                   uint32_t                out_addr,
                                   std::string &           kernel_name,
                                   dim3 &                  threadsPerBlock,
                                   dim3 &                  blocksPerGrid,
                                   std::vector<uint32_t> & params) {
    int unroll = 1;
    int tn;
    if ((b_col % 128) == 0) {
        tn = 4;
    } else if ((b_col % 96) == 0) {
        tn = 3;
    } else if ((b_col % 64) == 0) {
        tn = 2;
    } else if ((b_col % 32) == 0) {
        tn = 1;
    } else {
        throw std::runtime_error("Gemm dimension not support");
    }
    kernel_name = get_matmul_kernel(b_col);
    matmul_chw32_blocks(1, a_row, b_col, threadsPerBlock, blocksPerGrid, tn, 0);

    int rnd_a_col = (a_col + 31) / 32 * 32;
    int rnd_b_row = (b_row + 31) / 32 * 32;
    int rnd_b_col = (b_col + 31) / 32 * 32;

    matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, in_a_addr, in_b_addr, 0, out_addr, a_row, rnd_a_col,
                             rnd_b_row, rnd_b_col, unroll, unroll, unroll, 0, tn, sizeof(short), sizeof(short), params,
                             false, false);
}

void expand_batch_vxM_kernel_params(uint32_t                input_a,
                                    uint32_t                input_b,
                                    uint32_t                postScale,
                                    uint32_t                out,
                                    uint32_t                batch,
                                    uint32_t                D,
                                    uint32_t                Tk,
                                    uint32_t                kv_page,
                                    uint32_t                expand,
                                    int                     isDequant,
                                    std::vector<uint32_t> & params) {
    params.clear();
    uint32_t in1ElementSize;
    uint32_t deQuantBase = 0;
    if (isDequant) {
        in1ElementSize = sizeof(char);
    } else {
        in1ElementSize = sizeof(short);
    }
    int N = 1;
    if (Tk < kv_page) {
        kv_page = Tk;
        N       = 1;
    } else {
        N = Tk / kv_page;
    }
    //in0 [batch][expand][D]  ==> query
    //in1 [batch][D][N][kv_page]  ==> past key & current key
    //out [batch][expand][N][kv_page]
    uint32_t in0BatchStride = expand * D * sizeof(short);
    uint32_t in1BatchStride = D * Tk * in1ElementSize;
    uint32_t outBatchStride = Tk * sizeof(short);
    uint32_t rowSize        = Tk * sizeof(short);
    uint32_t expandStride   = D * sizeof(short);

    uint32_t rollback  = 1 * expandStride - 2;
    uint32_t inStrideZ = kv_page;

    uint32_t outStrideZ = kv_page;

    params.push_back(input_a);
    params.push_back(input_b);
    params.push_back(out);
    params.push_back(in0BatchStride);
    params.push_back(in1BatchStride);
    params.push_back(outBatchStride);
    params.push_back(D);
    params.push_back(batch);
    params.push_back(deQuantBase);
    params.push_back(rowSize);
    params.push_back(expandStride);
    params.push_back(rollback);
    if (expand == 7) {
        uint32_t rollback1 = 2 * expandStride - 2;
        params.push_back(rollback1);
    }
    params.push_back(inStrideZ);
    params.push_back(outStrideZ);
    return;
}

void expand_batch_hkv_vxM_kernel_params(uint32_t                input_a,
                                        uint32_t                input_b,
                                        uint32_t                out_high,
                                        uint32_t                out_low,
                                        uint32_t                D,
                                        uint32_t                Tk,
                                        uint32_t                kv_tile,
                                        std::vector<uint32_t> & params) {
    params.clear();
    params.push_back(input_a);
    params.push_back(input_b);
    params.push_back(out_high);
    params.push_back(out_low);
    // LOADALN/STOREALN use element strides, not byte strides.
    params.push_back(Tk);
    params.push_back(sizeof(short));
    params.push_back(Tk * sizeof(short));
    params.push_back(D);
    params.push_back(kv_tile);
    params.push_back(kv_tile);
    params.push_back(D);
}

static void loop2_mid_add_bc_y_f32_f16_f32_params(uint32_t                input_high,
                                                  uint32_t                input_low,
                                                  uint32_t                input_mask,
                                                  uint32_t                output_high,
                                                  uint32_t                output_low,
                                                  uint32_t                n_loop_stride,
                                                  uint32_t                block_loop_stride,
                                                  uint32_t                row_stride_y,
                                                  uint32_t                mask_n_loop_stride,
                                                  uint32_t                block_repeat_num,
                                                  uint32_t                kv_page_repeat,
                                                  std::vector<uint32_t> & params) {
    params.clear();
    params.push_back(input_high);
    params.push_back(input_low);
    params.push_back(input_mask);
    params.push_back(output_high);
    params.push_back(output_low);
    params.push_back(n_loop_stride);
    params.push_back(block_loop_stride);
    params.push_back(row_stride_y);
    params.push_back(mask_n_loop_stride);
    params.push_back(block_repeat_num);
    params.push_back(kv_page_repeat);
}

static void loop2_mid_reduce_max_f32_params(uint32_t                input_high,
                                            uint32_t                input_low,
                                            uint32_t                output_high,
                                            uint32_t                output_low,
                                            uint32_t                n_loop_stride,
                                            uint32_t                block_loop_stride,
                                            uint32_t                input_row_stride_y,
                                            uint32_t                output_row_stride_y,
                                            uint32_t                block_repeat_num,
                                            uint32_t                kv_page_repeat,
                                            std::vector<uint32_t> & params) {
    params.clear();
    params.push_back(input_high);
    params.push_back(input_low);
    params.push_back(output_high);
    params.push_back(output_low);
    params.push_back(n_loop_stride);
    params.push_back(block_loop_stride);
    params.push_back(input_row_stride_y);
    params.push_back(output_row_stride_y);
    params.push_back(block_repeat_num);
    params.push_back(kv_page_repeat);
}

//input0: [expand][N][kv_step]
//input1: [N][kv_step][D]
//out:    [expand][D]
void expand_batch_vxM_group_kernel_params(uint32_t                input_a,
                                          uint32_t                input_b,
                                          uint32_t                postScale,
                                          uint32_t                out,
                                          uint32_t                batch,
                                          uint32_t                kv_page,
                                          uint32_t                kv_page_repeat,
                                          uint32_t                feature,
                                          uint32_t                expand,
                                          int                     isDequant,
                                          std::vector<uint32_t> & params) {
    params.clear();
    (void) postScale;
    uint32_t rowSize     = kv_page_repeat * kv_page * sizeof(short);
    uint32_t in1StrideZ  = kv_page * feature;
    //256K is reserved for lut table
    uint32_t deQuantBase = 0;

    uint32_t in1ElementSize;

    if (isDequant) {
        in1ElementSize = sizeof(char);
    } else {
        in1ElementSize = sizeof(short);
    }

    batch                   = 1;
    uint32_t in0Rollback    = 0;
    uint32_t in1Rollback    = (batch * kv_page * feature - kv_page * feature) * in1ElementSize;
    uint32_t outBatchStride = 0;

    params.push_back(input_a);
    params.push_back(input_b);
    params.push_back(out);
    params.push_back(in0Rollback);
    params.push_back(in1Rollback);
    params.push_back(outBatchStride);
    params.push_back(in1StrideZ);
    params.push_back(kv_page);
    params.push_back(batch);
    params.push_back(deQuantBase);
    params.push_back(rowSize);

    return;
}
