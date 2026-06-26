
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

static inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// Rules:
// 1D:  B[D0]
//   blockDim.x -> D0, blockDim.y = 1
//   if D0 >= 8192, use gridDim.x
//
// 2D:  B[D0][D1]
//   blockDim.x -> D1, blockDim.y -> D0
//
// 3D:  B[D0][D1][D2]
//   blockDim.x -> D2, blockDim.y -> D1, blockDim.z = 1, gridDim.z = D0
//
// 4D: not supported (we only have 3 dims as input anyway)
static inline void calc_tbdim_align(uint32_t D0, uint32_t D1, uint32_t D2, dim3 & gridDim, dim3 & blockDim) {
    const uint32_t MAX_THREADS = 4096;

    // infer "rank"
    bool is3D = (D2 > 1);
    bool is2D = (!is3D && D1 > 1);
    bool is1D = (!is3D && !is2D);  // D1<=1 && D2<=1

    // Guard against weird zero sizes
    if (D0 == 0) {
        gridDim  = dim3(1, 1, 1);
        blockDim = dim3(1, 1, 1);
        return;
    }

    if (is1D) {
        // ----------------------------------------------------
        // 1D tensor: B[D0]
        // blockDim.x maps D0, blockDim.y = 1
        // if D0 >= 8192, use gridDim.x
        // ----------------------------------------------------
        uint32_t len = D0;

        uint32_t bx = (len < MAX_THREADS) ? len : MAX_THREADS;
        uint32_t by = 1;

        blockDim.x = bx;
        blockDim.y = by;
        blockDim.z = 1;

        gridDim.x = ceil_div(len, bx);
        gridDim.y = 1;
        gridDim.z = 1;

        return;
    }

    if (is2D) {
        // ----------------------------------------------------
        // 2D tensor: B[D0][D1]
        // blockDim.x maps D1, blockDim.y maps D0
        // Need blockDim.x * blockDim.y <= 8192
        // Use gridDim.x, gridDim.y for tiling if needed
        // ----------------------------------------------------
        uint32_t rows = D0;  // mapped to blockDim.y
        uint32_t cols = D1;  // mapped to blockDim.x

        // Start from "ideal" full coverage: bx = cols, by = rows
        uint32_t bx = cols;
        uint32_t by = rows;

        // Clamp to max threads per block
        if (bx == 0) {
            bx = 1;
        }
        if (by == 0) {
            by = 1;
        }

        // First clamp bx to not exceed MAX_THREADS
        if (bx > MAX_THREADS) {
            bx = MAX_THREADS;
        }

        // Now choose by so that bx * by <= MAX_THREADS
        if (bx * by > MAX_THREADS) {
            by = MAX_THREADS / bx;
            if (by == 0) {
                by = 1;
            }
        }

        blockDim.x = bx;
        blockDim.y = by;
        blockDim.z = 1;

        gridDim.x = ceil_div(cols, bx);
        gridDim.y = ceil_div(rows, by);
        gridDim.z = 1;

        return;
    }

    if (is3D) {
        // ----------------------------------------------------
        // 3D tensor: B[D0][D1][D2]
        // blockDim.x maps D2, blockDim.y maps D1, blockDim.z = 1
        // gridDim.z = D0
        // Need blockDim.x * blockDim.y <= 8192
        // ----------------------------------------------------
        uint32_t D0_out = D0;  // mapped to gridDim.z
        uint32_t D1_out = D1;  // mapped to blockDim.y
        uint32_t D2_out = D2;  // mapped to blockDim.x

        uint32_t bx = D2_out;
        uint32_t by = D1_out;

        if (bx == 0) {
            bx = 1;
        }
        if (by == 0) {
            by = 1;
        }

        // Clamp bx to MAX_THREADS first
        if (bx > MAX_THREADS) {
            bx = MAX_THREADS;
        }

        // Ensure bx * by <= MAX_THREADS
        if (bx * by > MAX_THREADS) {
            by = MAX_THREADS / bx;
            if (by == 0) {
                by = 1;
            }
        }

        blockDim.x = bx;
        blockDim.y = by;
        blockDim.z = 1;

        gridDim.x = ceil_div(D2_out, bx);
        gridDim.y = ceil_div(D1_out, by);
        gridDim.z = D0_out;

        return;
    }

    // If we ever got here (shouldn't with the above logic), treat as unsupported
    assert(false && "4D tensor or unsupported shape");
}
