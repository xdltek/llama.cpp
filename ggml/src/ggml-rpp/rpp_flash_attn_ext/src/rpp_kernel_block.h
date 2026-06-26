
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

static void KShape_TransPadding(uint32_t h, uint32_t w, dim3 & threadsPerBlock, dim3 & blocksPerGrid) {
    const int max_threads = 4096;
    int       chosen_w0   = 0;
    int       chosen_h0   = 0;

    for (int w0 = (w < (uint32_t) max_threads ? (int) w : max_threads); w0 >= 1; --w0) {
        if ((w % (uint32_t) w0) != 0) {
            continue;
        }

        const int max_h0 = max_threads / w0;
        for (int h0 = (h < (uint32_t) max_h0 ? (int) h : max_h0); h0 >= 1; --h0) {
            if ((h % (uint32_t) h0) != 0) {
                continue;
            }

            chosen_w0 = w0;
            chosen_h0 = h0;
            break;
        }

        if (chosen_w0 != 0) {
            break;
        }
    }

    if (chosen_w0 == 0 || chosen_h0 == 0) {
        throw std::runtime_error("w is not divisible by w0 or h is not divisible by h0");
    }

    threadsPerBlock.x = (uint32_t) chosen_w0;
    threadsPerBlock.y = (uint32_t) chosen_h0;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = w / (uint32_t) chosen_w0;
    blocksPerGrid.y   = h / (uint32_t) chosen_h0;
    blocksPerGrid.z   = 1;
}

static void KShape_Trans(uint32_t h, uint32_t w, dim3 & threadsPerBlock, dim3 & blocksPerGrid) {
    int colSegLenth = w;
    int outPoints   = 8;

    if (colSegLenth >= 128 * outPoints) {
        threadsPerBlock.z = 128;
    } else {
        threadsPerBlock.z = colSegLenth / outPoints;
    }

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    blocksPerGrid.x   = 1;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;
    return;
}

static void KShape_Hw2Hw32(uint32_t H, uint32_t W, dim3 & threadsPerBlock, dim3 & blocksPerGrid) {
    int block_x = 32, block_y = 1, block_z = W / block_x, grid_x = H;
    while (block_x * block_y * block_z < 4096 && grid_x % 2 == 0) {
        block_y *= 2;
        grid_x /= 2;
    }
    if (grid_x * block_y != H) {
        throw std::runtime_error("flashAttn input Mask shape is illegal!");
    }
    if (block_x * block_z != W) {
        throw std::runtime_error("flashAttn input Mask shape is illegal!");
    }
    threadsPerBlock.x = block_x;
    threadsPerBlock.y = block_y;
    threadsPerBlock.z = block_z;
    blocksPerGrid.x   = grid_x;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;
}

static void KShape_Scale(uint32_t H, uint32_t W, dim3 & threadsPerBlock, dim3 & blocksPerGrid) {
    int mul_block_x = W, mul_block_y = 1, mul_grid_x = H;

    if (mul_block_x * mul_grid_x < 4096) {
        mul_block_y *= mul_grid_x;
        mul_grid_x = 1;
    } else {
        while ((mul_block_x * mul_block_y < 4096)) {
            mul_block_y *= 2;
            mul_grid_x /= 2;
        }
    }
    if (mul_grid_x * mul_block_y != H) {
        throw std::runtime_error("block and grid size is illegal");
    }
    threadsPerBlock.x = mul_block_x;
    threadsPerBlock.y = mul_block_y;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = mul_grid_x;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;
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

void expand_kerneldim_calc_batch_vxM(uint32_t   b,
                                     uint32_t   D,
                                     uint32_t   Tk,
                                     uint32_t   kv_page,
                                     uint32_t   e,
                                     uint32_t & block_x,
                                     uint32_t & block_y,
                                     uint32_t & block_z,
                                     uint32_t & grid_x,
                                     uint32_t & grid_y,
                                     uint32_t & grid_z) {
    (void) b;
    (void) D;
    (void) e;
    int N = 1;
    if (Tk < kv_page) {
        kv_page = Tk;
        N       = 1;
    } else {
        N = Tk / kv_page;
    }

    block_x = kv_page;
    block_y = 1;
    block_z = N;
    grid_x  = 1;
    grid_y  = 1;
    grid_z  = 1;

    if((kv_page % 32 != 0) || (N * kv_page != Tk) || (Tk > 16384))
    {
        throw std::runtime_error("Current Tk not support");
    }

    while (block_x * block_z >= 8192)
    {
        if ((block_x % 2) != 0)
        {
            throw std::runtime_error("Current Tk not support VXM tile split");
        }
        block_x /= 2;
        grid_x *= 2;
    }

    if ((block_x % 32) != 0 || block_x * grid_x != kv_page)
    {
        throw std::runtime_error("Current Tk not support VXM tile split");
    }
}

void expand_kerneldim_calc_batch_hkv_vxM(uint32_t   D,
                                         uint32_t   Tk,
                                         uint32_t   expand,
                                         uint32_t & block_x,
                                         uint32_t & block_y,
                                         uint32_t & block_z,
                                         uint32_t & grid_x,
                                         uint32_t & grid_y,
                                         uint32_t & grid_z) {
    (void) D;
    const uint32_t kv_tile = 128;
    if ((Tk % kv_tile) != 0) {
        throw std::runtime_error("Current Tk not support batch_hkv_vxM tile");
    }

    const uint32_t N = Tk / kv_tile;
    if (N == 0) {
        throw std::runtime_error("Current Tk not support batch_hkv_vxM tile");
    }

    const uint32_t max_block_y = 8191 / (kv_tile * expand);
    if (max_block_y < 1) {
        throw std::runtime_error("Thread Block greater than 8K");
    }

    uint32_t candidate_block_y = (N < max_block_y) ? N : max_block_y;
    while ((candidate_block_y > 1) && ((N % candidate_block_y) != 0)) {
        --candidate_block_y;
    }

    block_x = kv_tile;
    block_y = candidate_block_y;
    block_z = expand;
    grid_x  = N / block_y;
    grid_y  = 1;
    grid_z  = 1;
}

void expand_kerneldim_calc_batch_vxM_group(uint32_t   b,
                                           uint32_t   c,
                                           uint32_t   D,
                                           uint32_t   expand,
                                           uint32_t & block_x,
                                           uint32_t & block_y,
                                           uint32_t & block_z,
                                           uint32_t & grid_x,
                                           uint32_t & grid_y,
                                           uint32_t & grid_z) {
    block_x = D;
    block_y = expand;
    block_z = b;
    grid_x  = 1;
    grid_y  = 1;
    grid_z  = 1;

    if (expand * D >= 8192) {
        throw std::runtime_error("Thread Block greater than 8K");
    }
}
