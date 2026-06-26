#pragma once

#include "rpp_drv_api.h"

#include <rpp_runtime.h>

#include <algorithm>
#include <string>
#include <vector>

inline const std::vector<std::string> g_large_power_kernels = {
    "matmul_tn1_f16_f32_f16",
    "matmul_tn1_i8_i32_f16",
    "matmul_tn1_i8_i32_i8",
    "matmul_tn2_f16_f32_f16",
    "matmul_tn3_f16_f32_f16",
    "matmul_tn4_f16_f32_f16",
    "matmul_tn4_f16_f32_f32",
};

inline int Kernel2Frequency(const std::string & name, int threads, int freq2, int freq1, int freq0) {
    (void) freq2;

    auto find_it = std::find(g_large_power_kernels.begin(), g_large_power_kernels.end(), name);
    if (find_it != g_large_power_kernels.end()) {
        if (threads >= 4096) {
            return freq1;
        }
        return freq0;
    }
    return freq0;
}

void rpp_reset_dfs_state(RPPstream kernel_stream);
