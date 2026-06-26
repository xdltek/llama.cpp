
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
namespace kernel_q5_k_vxm {

static void q5k_super_scale_blocks(uint32_t K,
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
#if 0
    uint32_t dims_x = column;
    block_y = 0;
    if(column < KDC_H_MAX_THREAD_NUM){
        while(1)
        {
            if(block_y >= row)
                break;
            block_y += 1;
            if(dims_x * block_y > KDC_H_MAX_THREAD_NUM){
                block_y -= 1;
                break;
            }
        }
    }
    else
    {
        while(dims_x > KDC_H_MAX_THREAD_NUM){
            dims_x = (dims_x + 1)/2;
        }
        block_y = 1;
    }
    block_x = dims_x;
    block_x = (block_x + 31) / 32 * 32;
    grid_x = (column + block_x - 1) / block_x;
    grid_y = (row + block_y - 1) / block_y;
    block_z = RppHW32Fix(block_x, block_y, 1);
#endif
    block_x       = 32;
    uint32_t cntX = column / 32;
    while (block_x < 4096) {
        if (cntX % 2 == 0) {
            block_x = block_x * 2;
            cntX    = cntX / 2;
        } else {
            break;
        }
    }

    block_y       = 1;
    uint32_t cntY = row;
    while (block_x * cntX * block_y < 4096) {
        if (cntY % 2 == 0) {
            block_y = block_y * 2;
            cntY    = cntY / 2;
        } else {
            break;
        }
    }
    grid_x  = cntX;
    grid_y  = cntY;
    block_z = RppHW32Fix(block_x, block_y, 1);
    grid_z  = loop;

    assert(block_x % 32 == 0);
    assert(block_x * grid_x == column);
}
}  // namespace kernel_q5_k_vxm
