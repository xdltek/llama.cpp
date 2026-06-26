
#pragma once
#include "ggml-rpp/rpp_common.h"
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

static uint32_t dyn_d0 = 0;

static inline void rpp_update_dyn_d0(uint32_t d0) {
#ifdef RPP_SIM_RT
    uint32_t * fptr = (uint32_t *) &fram[0x814 / 2];
    fptr[0]         = d0;
    fptr            = (uint32_t *) &fram[0x1814 / 2];
    fptr[0]         = d0;
#else
    // FILE* fp = fopen("./in1.bin", "wb");
    // fwrite(&d0, sizeof(uint32_t), 1, fp);
    // fclose(fp);

    // char pcie_cmd[1024] = {0};
    // sprintf(pcie_cmd, "%s",
    //     "/home/lab/rpp_drv_api/build/tools/pcie_tool/pcie_tool -m dw -a 0x1006000808 -f in1.bin");
    // assert(system(pcie_cmd) == 0);
    // memset(pcie_cmd, 0, sizeof(pcie_cmd));
    // sprintf(pcie_cmd, "%s",
    //     "/home/lab/rpp_drv_api/build/tools/pcie_tool/pcie_tool -m dw -a 0x1006001808 -f in1.bin");
    // assert(system(pcie_cmd) == 0);
    if (dyn_d0 != d0) {
        dyn_d0 = d0;
        RPP_CHECK(rtDeviceRegWrite32(0x1006000814, d0));
        RPP_CHECK(rtDeviceRegWrite32(0x1006001814, d0));
    }
#endif
    return;
}
