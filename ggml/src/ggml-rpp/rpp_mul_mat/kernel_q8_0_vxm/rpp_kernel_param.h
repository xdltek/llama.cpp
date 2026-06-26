
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

namespace kernel_q8_0_vxm {
static void matmul_weights_i80_kernel_params(uint32_t                input_a,
                                             uint32_t                input_b,
                                             uint32_t                postScale,
                                             uint32_t                out,
                                             uint32_t                lut_addr,
                                             uint32_t                zero_addr,
                                             uint32_t                input_acc_addr,
                                             uint32_t                input_acc_addr_hi,
                                             int                     use_asym_opt,
                                             uint32_t                Md,
                                             uint32_t                Nd,
                                             uint32_t                Kd,
                                             uint32_t                combine,
                                             uint32_t                stride_a,
                                             uint32_t                stride_b,
                                             uint32_t                stride_scale,
                                             std::vector<uint32_t> & params) {
    uint32_t b_Nd_size, c_Nd_size;
    uint32_t Kd_size;
    int      elements_per_thread = 2;
    Kd_size                      = Kd / elements_per_thread * sizeof(short);
    b_Nd_size                    = Nd * sizeof(short);

    params.emplace_back(input_a);
    params.emplace_back(input_b);
    params.emplace_back(out);
    params.emplace_back(Nd * sizeof(short));

    params.emplace_back(postScale);
    params.emplace_back(b_Nd_size);
    params.emplace_back(0);
    params.emplace_back(stride_a);
    params.emplace_back(stride_b);
    params.emplace_back(stride_scale);
    params.emplace_back(lut_addr);
    params.emplace_back(zero_addr);
    params.emplace_back(input_acc_addr);
    params.emplace_back(input_acc_addr_hi);

    params.emplace_back(0);
    params.emplace_back(0);
    params.emplace_back(Kd / elements_per_thread);
    params.emplace_back(Kd_size);
    params.emplace_back(combine);
}
}  // namespace kernel_q8_0_vxm
