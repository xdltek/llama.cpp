#pragma once
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"

#include <rpp_runtime.h>

#include <cassert>
#include <stdexcept>
#include <vector>

#define MAX_EXEC 255
#define DIM_X    32

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
namespace kernel_q2_s_nolut {
static void q2s_nolut_super_scale_blocks(uint32_t K,
                                         uint32_t super_group,
                                         uint32_t group,
                                         uint32_t N,
                                         dim3 &   threadsPerBlock,
                                         dim3 &   blocksPerGrid) {
    (void) group;
    int      nsg     = K / super_group;
    uint32_t block_z = nsg;
    uint32_t block_x = N;
    uint32_t cntX    = 1;
    while (block_x * block_z >= 8192) {
        block_x = block_x / 2;
        cntX    = cntX * 2;
    }
    if (block_x * cntX * block_z != N * nsg) {
        throw std::runtime_error("Matmul Q2S TB Invalid");
    }

    threadsPerBlock.x = block_x;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = block_z;

    blocksPerGrid.x = cntX;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;
}

static void q2s_nolut_dequant_blocks(uint32_t loop,
                                     uint32_t row,
                                     uint32_t column,
                                     dim3 &   threadsPerBlock,
                                     dim3 &   blocksPerGrid,
                                     int      weights_group) {
    (void) loop;
    uint32_t elements_per_thread = 64;
    uint32_t block_x             = DIM_X;
    uint32_t block_y             = weights_group / elements_per_thread;
    uint32_t block_z             = 1;
    //----------------------------------------------------------------------------------------------------
    // out        [K/256]       |  [4]  |  [4]       |  [16]        |  [N]
    // out        [z]           |  [y]  |  [unroll0] |  [unroll1]   |  [x]
    //            [grid.y]*[z]  |  [y]                              |  [grid.x]*[x]
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
    // Some valid K (e.g. 3072) do not divide the first heuristic choice.
    const uint32_t base_rows_per_block_z1 = block_y * elements_per_thread;
    while (block_z > 1 && (row % (base_rows_per_block_z1 * block_z)) != 0) {
        block_z >>= 1;
    }

    const int sub_row_per_block = (int) (block_y * block_z * elements_per_thread);

    if (row < block_y * block_z) {
        throw -1;
    }

    if ((row % (uint32_t) sub_row_per_block) != 0) {
        throw std::runtime_error("q2s_nolut_dequant_blocks row is not divisible by block tile");
    }

    uint32_t grid_x = (column + DIM_X - 1) / DIM_X;  // column grid
    uint32_t grid_y = row / sub_row_per_block;

    if (grid_y == 0) {
        throw std::runtime_error("q2s_nolut_dequant_blocks grid_y became zero");
    }

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
}  // namespace kernel_q2_s_nolut
