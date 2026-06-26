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

namespace kernel_bf16 {
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
}  // namespace kernel_bf16
