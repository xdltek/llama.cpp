
#pragma once
#include "ggml-rpp/rpp_kernel_utils.h"
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
#include <stdexcept>
#include <string>
#include <vector>

#define MAX_EXEC 255
#define DIM_X    32

namespace kernel_q4_k {

static void chw2chw32_blocks(uint32_t loop,
                             uint32_t row,
                             uint32_t column,
                             dim3 &   threadsPerBlock,
                             dim3 &   blocksPerGrid) {
    uint32_t gy = 1;
    uint32_t bx = 32;
    uint32_t by = row;

    if (bx * by > MAX_EXEC * DIM_X) {
        while (true) {
            by = (by + 1) / 2;
            gy *= 2;
            if (bx * by <= MAX_EXEC * DIM_X) {
                break;
            }
        }
        grid_dim_adjustment(gy, by, (int) row);
    }

    threadsPerBlock.x = bx;
    threadsPerBlock.y = by;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = (column + 31) / 32;
    blocksPerGrid.y = gy;
    blocksPerGrid.z = loop;
}

static void chw322chw_blocks(uint32_t loop,
                             uint32_t row,
                             uint32_t column,
                             dim3 &   threadsPerBlock,
                             dim3 &   blocksPerGrid) {
    uint32_t gy = 1;
    uint32_t bx = 32;
    uint32_t by = row;

    if (bx * by > MAX_EXEC * DIM_X) {
        while (true) {
            by = (by + 1) / 2;
            gy *= 2;
            if (bx * by <= MAX_EXEC * DIM_X) {
                break;
            }
        }
        grid_dim_adjustment(gy, by, (int) row);
    }

    threadsPerBlock.x = bx;
    threadsPerBlock.y = by;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = (column + 31) / 32;
    blocksPerGrid.y = gy;
    blocksPerGrid.z = loop;
}

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
// in_scale   [K/256]  | [256/8]                   |  [qscale_per_word][N]
// in_scale   [z]      | [unroll]                  |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
// out_scale   [K/256]  | [256/16]                 |  [N]
// out_scale   [z]      | [unroll]                 |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
static void q4k_super_scale_blocks(uint32_t K,
                                   uint32_t super_group,
                                   uint32_t group,
                                   uint32_t N,
                                   dim3 &   threadsPerBlock,
                                   dim3 &   blocksPerGrid) {
    int      sg      = super_group;
    int      g       = group;
    int      nsg     = K / super_group;
    uint32_t block_z = nsg;
    uint32_t block_x = N;
    uint32_t cntX    = 1;
    while (block_x * block_z >= 8192) {
        block_x = block_x / 2;
        cntX    = cntX * 2;
    }
    if (block_x * cntX * block_z != N * nsg) {
        throw std::runtime_error("Matmul Q6K TB Invalid");
    }

    threadsPerBlock.x = block_x;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = block_z;

    blocksPerGrid.x = cntX;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;
}

static void q4k_dequant_blocks(uint32_t loop,
                               uint32_t row,
                               uint32_t column,
                               dim3 &   threadsPerBlock,
                               dim3 &   blocksPerGrid,
                               int      weights_group) {
    uint32_t elements_per_thread = 32;
    uint32_t block_x             = DIM_X;
    uint32_t block_y             = weights_group / elements_per_thread;
    uint32_t block_z             = 1;
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // in_wq     [K/256]      |  [8]  |  [32/4]    |  [4][N]
    // in_wq     [z]          |  [y]  |  [unroll]  |  [x]
    //           [grid.y]*[z] |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    while (true) {
        // condition 1: block occupancy threshold
        if ((block_x * block_y * block_z) >= (128 * DIM_X)) {
            break;
        }

        // condition 2: do not map more rows than exist
        if ((block_y * block_z * elements_per_thread) >= row) {
            break;
        }

        block_z *= 2;
    }

    // Keep the chosen block shape but guarantee exact row tiling.
    const uint32_t base_rows_per_block_z1 = block_y * elements_per_thread;
    while (block_z > 1 && (row % (base_rows_per_block_z1 * block_z)) != 0) {
        block_z >>= 1;
    }

    const int sub_row_per_block = (int) (block_y * block_z * elements_per_thread);

    if (row < block_y * block_z) {
        throw -1;
    }

    if ((row % (uint32_t) sub_row_per_block) != 0) {
        throw std::runtime_error("q4k_dequant_blocks row is not divisible by block tile");
    }

    uint32_t grid_x = (column + DIM_X - 1) / DIM_X;  // column grid
    uint32_t grid_y = row / sub_row_per_block;

    uint32_t grid_z = 1;

    // ----------------------------------------------------------------
    // assign CUDA-style launch parameters
    // ----------------------------------------------------------------
    threadsPerBlock.x = block_x;
    threadsPerBlock.y = block_y;
    threadsPerBlock.z = block_z;

    blocksPerGrid.x = grid_x;
    blocksPerGrid.y = grid_y;
    blocksPerGrid.z = grid_z;
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
    // Adjust block_y / grid_y if row is too large (or exceeds per-block limit)
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

    // Compute grid_x based on tn packing and tail handling
    int      nr_of_tn = (column + 31) / 32;  // number of 32-wide tiles across N
    uint32_t gx       = 0;
    if (is_tail == 0) {
        gx = (uint32_t) (nr_of_tn / tn);
    } else {
        gx = 1;
    }
    // Fill dim3 outputs
    threadsPerBlock.x = bx;
    threadsPerBlock.y = by;
    threadsPerBlock.z = bz;
    blocksPerGrid.x   = gx;
    blocksPerGrid.y   = gy;
    blocksPerGrid.z   = loop;
}

static void kerneldim_calc_flatten_pmem(uint32_t   batch,
                                        int        dim,
                                        uint32_t & block_x,
                                        uint32_t & block_y,
                                        uint32_t & block_z,
                                        uint32_t & grid_x,
                                        uint32_t & grid_y,
                                        uint32_t & grid_z) {
    unsigned int size = dim;
    size *= batch;
    uint32_t bx, gx;
    if (size > 8160) {
        gx = (size + 8159) / 8160;  // would rather to reduce gridDim number as less as possible
        bx = (size + gx - 1) / gx;
        bx = (bx + 31) / 32 * 32;
        gx = (size + bx - 1) / bx;
    } else {
        bx = (size + 31) / 32 * 32;
        gx = 1;
    }
    assert(bx > 32);
    block_x = bx;
    grid_x  = gx;
    block_y = 1;
    grid_y  = 1;
    block_z = 1;
    grid_z  = 1;
}

static bool flatten_pmem_handler(dim3 & threadsPerBlock, dim3 & blocksPerGrid, dim3 & threadsPerBlockTail, int dim) {
    // Create uint32_t variables for the function calls
    uint32_t block_x = threadsPerBlock.x;
    uint32_t block_y = threadsPerBlock.y;
    uint32_t block_z = threadsPerBlock.z;
    uint32_t grid_x  = blocksPerGrid.x;
    uint32_t grid_y  = blocksPerGrid.y;
    uint32_t grid_z  = blocksPerGrid.z;

    kerneldim_calc_flatten_pmem(1, dim, block_x, block_y, block_z, grid_x, grid_y, grid_z);

    // Copy values back to task
    threadsPerBlock.x     = block_x;
    threadsPerBlock.y     = block_y;
    threadsPerBlock.z     = block_z;
    blocksPerGrid.x       = grid_x;
    blocksPerGrid.y       = grid_y;
    blocksPerGrid.z       = grid_z;
    unsigned int size     = dim;
    threadsPerBlockTail.x = threadsPerBlock.x;
    threadsPerBlockTail.y = threadsPerBlock.y;
    threadsPerBlockTail.z = threadsPerBlock.z;
    if (threadsPerBlock.x * blocksPerGrid.x > size && size > (255 * 32)) {
        threadsPerBlockTail.x = size - (blocksPerGrid.x - 1) * threadsPerBlock.x;
        if (threadsPerBlockTail.x <= 0) {
            assert(0);
        }
    }
    return true;
}
}  // namespace kernel_q4_k
