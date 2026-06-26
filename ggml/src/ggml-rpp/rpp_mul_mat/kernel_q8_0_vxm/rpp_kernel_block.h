
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
#include <string>
#include <vector>

#define MAX_EXEC 255
#define DIM_X    32

namespace kernel_q8_0_vxm {
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

static void kerneldim_calc_matmul_linear(uint32_t   loop,
                                         uint32_t   row,
                                         uint32_t   column,
                                         uint32_t & block_x,
                                         uint32_t & block_y,
                                         uint32_t & block_z,
                                         uint32_t & grid_x,
                                         uint32_t & grid_y,
                                         uint32_t & grid_z) {
    if (loop > 65535) {
        assert(0);
    }
    uint32_t dims_x = column;
    block_y         = 0;
    if (column < KDC_H_MAX_THREAD_NUM) {
        while (1) {
            if (block_y >= row) {
                break;
            }
            block_y += 1;
            if (dims_x * block_y > KDC_H_MAX_THREAD_NUM) {
                block_y -= 1;
                break;
            }
        }
    } else {
        while (dims_x > KDC_H_MAX_THREAD_NUM) {
            dims_x = (dims_x + 1) / 2;
        }
        block_y = 1;
    }
    block_x = dims_x;
    block_x = (block_x + 31) / 32 * 32;
    grid_x  = (column + block_x - 1) / block_x;
    grid_y  = (row + block_y - 1) / block_y;
    block_z = RppHW32Fix(block_x, block_y, 1);
    grid_z  = loop;
}
}  // namespace kernel_q8_0_vxm
