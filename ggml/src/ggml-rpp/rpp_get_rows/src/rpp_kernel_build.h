// get_rows_rpp.cpp
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_get_rows/src/rpp_kernel_block.h"
#include "rpp_get_rows/src/rpp_kernel_param.h"

#include <assert.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

void rpp_get_rows_build(rpp_kernel_context & ctx,
                        int                  elements_per_row,
                        int                  in_bytes_per_element,
                        int                  out_bytes_per_element,
                        int                  bytes_per_rowid,
                        int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    //RPPdeviceptr dev_rowdata = ctx.dev_in[0];
    //RPPdeviceptr dev_rowid = ctx.dev_in[1];
    //RPPdeviceptr dev_out = ctx.dev_out[0];

    RPPdeviceptr dev_out     = ctx.dev_in[0];
    RPPdeviceptr dev_rowid   = ctx.dev_in[1];
    RPPdeviceptr dev_rowdata = ctx.dev_out[0];

    RPPdeviceptr phy_rowdata, phy_out, phy_rowid;
    rppMemGetPhyAddr(&phy_rowdata, dev_rowdata);
    rppMemGetPhyAddr(&phy_out, dev_out);
    rppMemGetPhyAddr(&phy_rowid, dev_rowid);
    RPPdeviceptr rowid_fram0 = 0x1006000810;
    RPPdeviceptr rowid_fram1 = 0x1006001810;

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/get_rows.o");

    //RPPdeviceptr sram_base = ctx.virtual_sram_base;
    //RPPdeviceptr sram_baseA = sram_base + SET_ROW_MAX_ELEM;

    //only enable at simulator
#ifdef RPP_SIM_RT
    rtMemcpyAsync((void *) (sram_base), (void *) dev_rowdata, 32, rtMemcpyDeviceToSram, ctx.kernelStream);
#endif
    if (in_bytes_per_element != out_bytes_per_element) {
        throw std::runtime_error("Get Row From BF16 To Float32 not Supportted");
    } else {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 1;
        threadsPerBlock.z = 1;
        blocksPerGrid.x   = 1;
        blocksPerGrid.y   = 1;
        blocksPerGrid.z   = 1;
        params.clear();
        params.push_back(phy_rowdata & 0xFFFFFFFF);
        params.push_back((phy_rowdata >> 32) & 0xFFFFFFFF);
        params.push_back(phy_out & 0xFFFFFFFF);
        params.push_back((phy_out >> 32) & 0xFFFFFFFF);
        params.push_back(phy_rowid & 0xFFFFFFFF);
        params.push_back((phy_rowid >> 32) & 0xFFFFFFFF);
        params.push_back(rowid_fram0 & 0xFFFFFFFF);
        params.push_back(rowid_fram1 & 0xFFFFFFFF);
        params.push_back((rowid_fram0 >> 32) & 0xFFFFFFFF);
        params.push_back(elements_per_row * in_bytes_per_element);
        params.push_back(bytes_per_rowid);
        launchWrapperAysnc("dma_ddr_to_row_ddr_non_cont", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
