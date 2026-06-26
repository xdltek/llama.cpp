
#pragma once
#include "ggml-rpp/rpp_common.h"
#include "rpp_drv_api.h"
#include "rpp_runtime.h"

#include <assert.h>
#include <math.h>
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
#define SET_ROW_MAX_ELEM (12 * 1024 * 1024)
#ifdef RPP_SIM_RT
extern uint16_t * fram;
#endif

static uint32_t s_start_of_rows           = 0;
static uint32_t s_contious_number_of_rows = 0;

static inline void rpp_update_rowids(uint32_t start_of_rows, uint32_t contious_number_of_rows) {
#ifdef RPP_SIM_RT
    uint32_t * fptr = (uint32_t *) &fram[0x800 / 2];
    fptr[0]         = start_of_rows;
    fptr            = (uint32_t *) &fram[0x1800 / 2];
    fptr[0]         = start_of_rows;

    fptr    = (uint32_t *) &fram[0x804 / 2];
    fptr[0] = contious_number_of_rows;
    fptr    = (uint32_t *) &fram[0x1804 / 2];
    fptr[0] = contious_number_of_rows;
#else
    // FILE * fp = fopen("./in1.bin", "wb");
    // fwrite(&start_of_rows, sizeof(uint32_t), 1, fp);
    // fclose(fp);

    // fp = fopen("./in2.bin", "wb");
    // fwrite(&contious_number_of_rows, sizeof(uint32_t), 1, fp);
    // fclose(fp);
    // char pcie_cmd[1024] = { 0 };
    // sprintf(pcie_cmd, "%s", "/usr/local/rpp/bin/ae-smi-internal pcie -m dw -a 0x1006000800 -f in1.bin");
    // assert(system(pcie_cmd) == 0);
    // memset(pcie_cmd, 0, sizeof(pcie_cmd));
    // sprintf(pcie_cmd, "%s", "/usr/local/rpp/bin/ae-smi-internal pcie -m dw -a 0x1006001800 -f in1.bin");
    // assert(system(pcie_cmd) == 0);
    // memset(pcie_cmd, 0, sizeof(pcie_cmd));
    // sprintf(pcie_cmd, "%s", "/usr/local/rpp/bin/ae-smi-internal pcie -m dw -a 0x1006000804 -f in2.bin");
    // assert(system(pcie_cmd) == 0);
    // memset(pcie_cmd, 0, sizeof(pcie_cmd));
    // sprintf(pcie_cmd, "%s", "/usr/local/rpp/bin/ae-smi-internal pcie -m dw -a 0x1006001804 -f in2.bin");
    // assert(system(pcie_cmd) == 0);
    if (s_start_of_rows != start_of_rows || s_contious_number_of_rows != contious_number_of_rows) {
        s_start_of_rows           = start_of_rows;
        s_contious_number_of_rows = contious_number_of_rows;
        RPP_CHECK(rtDeviceRegWrite32(0x1006000800, start_of_rows));
        RPP_CHECK(rtDeviceRegWrite32(0x1006001800, start_of_rows));
        RPP_CHECK(rtDeviceRegWrite32(0x1006000804, contious_number_of_rows));
        RPP_CHECK(rtDeviceRegWrite32(0x1006001804, contious_number_of_rows));
    }
#endif
    return;
}
