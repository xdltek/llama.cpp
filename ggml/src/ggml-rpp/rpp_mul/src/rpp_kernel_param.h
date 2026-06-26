
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

#define MAX_TILE_SIZE (1024 * 1024)

// ----------------------------- broadcast axis detect ------------------------------
// Return -2 if unsupported.
static inline int rpp_broadcast_axis(int C0, int H0, int W0, int C1, int H1, int W1) {
    // no broadcast
    if (C0 == C1 && H0 == H1 && W0 == W1) {
        return -1;
    }

    // input1 broadcast cases (axis 0/1/2)
    if (C1 == 1 && H1 == H0 && W1 == W0 && C0 >= 1) {
        return 0;  // broadcast on C0
    }
    if (H1 == 1 && C1 == C0 && W1 == W0 && H0 >= 1) {
        return 1;  // broadcast on H0
    }
    if (W1 == 1 && C1 == C0 && H1 == H0 && W0 >= 1) {
        return 2;  // broadcast on W0
    }

    // input0 broadcast cases (axis 3/4/5)
    if (C0 == 1 && H0 == H1 && W0 == W1 && C1 >= 1) {
        return 3;  // broadcast on C1
    }
    if (H0 == 1 && C0 == C1 && W0 == W1 && H1 >= 1) {
        return 4;  // broadcast on H1
    }
    if (W0 == 1 && C0 == C1 && H0 == H1 && W1 >= 1) {
        return 5;  // broadcast on W1
    }

    // new: input1 broadcast on both C0 & H0 (input1 = 1 x 1 x W)
    if (C1 == 1 && H1 == 1 && W1 == W0 && C0 >= 1 && H0 >= 1) {
        return 6;
    }

    // new: input0 broadcast on both C1 & H1 (input0 = 1 x 1 x W)
    if (C0 == 1 && H0 == 1 && W0 == W1 && C1 >= 1 && H1 >= 1) {
        return 7;
    }

    return -2;  // unsupported pattern
}
