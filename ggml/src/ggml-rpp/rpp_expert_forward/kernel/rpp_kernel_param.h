
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

#ifdef __cplusplus
extern "C" {
#endif
extern uint16_t * fram;
#ifdef __cplusplus
}
#endif

static inline void rpp_update_expert_tokens(uint32_t token_count, uint32_t token_begin_offset) {
#if defined(RPP_SIM_RT) || defined(_WIN32)
    uint32_t * fptr = (uint32_t *) &fram[0x818 / 2];
    fptr[0]         = token_count;
    fptr            = (uint32_t *) &fram[0x1818 / 2];
    fptr[0]         = token_count;

    fptr    = (uint32_t *) &fram[0x81c / 2];
    fptr[0] = token_begin_offset;
    fptr    = (uint32_t *) &fram[0x181c / 2];
    fptr[0] = token_begin_offset;
#else
#    if 0
    FILE * fp = fopen("./in1.bin", "wb");
    fwrite(&token_count, sizeof(uint32_t), 1, fp);
    fclose(fp);

    fp = fopen("./in2.bin", "wb");
    fwrite(&token_begin_offset, sizeof(uint32_t), 1, fp);
    fclose(fp);

    const char * pcie_tool_path = "/usr/local/rpp/bin/pcie_tool";
    char         pcie_cmd[1024] = {};

    sprintf(pcie_cmd, "%s %s", pcie_tool_path, "-m dw -a 0x1006000818 -f in1.bin");
    assert(system(pcie_cmd) == 0);
    memset(pcie_cmd, 0, sizeof(pcie_cmd));
    sprintf(pcie_cmd, "%s %s", pcie_tool_path, "-m dw -a 0x1006001818 -f in1.bin");
    assert(system(pcie_cmd) == 0);
    memset(pcie_cmd, 0, sizeof(pcie_cmd));
    sprintf(pcie_cmd, "%s %s", pcie_tool_path, "-m dw -a 0x100600081c -f in2.bin");
    assert(system(pcie_cmd) == 0);
    memset(pcie_cmd, 0, sizeof(pcie_cmd));
    sprintf(pcie_cmd, "%s %s", pcie_tool_path, "-m dw -a 0x100600181c -f in2.bin");
    assert(system(pcie_cmd) == 0);
#    else
    rtDeviceRegWrite32(0x1006000818, token_count);
    rtDeviceRegWrite32(0x1006001818, token_count);
    rtDeviceRegWrite32(0x100600081c, token_begin_offset);
    rtDeviceRegWrite32(0x100600181c, token_begin_offset);
#    endif
#endif
}
