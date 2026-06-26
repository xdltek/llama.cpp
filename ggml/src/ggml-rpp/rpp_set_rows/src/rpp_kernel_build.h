// memcpy_2d_rpp.cpp
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_set_rows/src/rpp_kernel_block.h"
#include "rpp_set_rows/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// -----------------------------
// Build graph once
// -----------------------------
void rpp_set_rows_build(rpp_kernel_context & ctx,
                        int                  elements_per_row,
                        int                  in_bytes_per_element,
                        int                  out_bytes_per_element,
                        int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          dev_rowdata = ctx.dev_in[0];
    RPPdeviceptr          dev_rowid   = ctx.dev_in[1];
    RPPdeviceptr          dev_out     = ctx.dev_out[0];

    RPPdeviceptr phy_rowdata, phy_out, phy_rowid;
    rppMemGetPhyAddr(&phy_rowdata, dev_rowdata);
    rppMemGetPhyAddr(&phy_out, dev_out);
    rppMemGetPhyAddr(&phy_rowid, dev_rowid);
    RPPdeviceptr rowid_fram0 = 0x1006000810;
    RPPdeviceptr rowid_fram1 = 0x1006001810;

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/set_rows.o");

    RPPdeviceptr sram_base  = ctx.virtual_sram_base;
    RPPdeviceptr sram_baseA = sram_base + SET_ROW_MAX_ELEM;

    //only enable at simulator
#ifdef RPP_SIM_RT
    rtMemcpyAsync((void *) (sram_base), (void *) dev_rowdata, 32, rtMemcpyDeviceToSram, ctx.kernelStream);
#endif
    if (in_bytes_per_element != out_bytes_per_element) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 1;
        threadsPerBlock.z = 1;
        blocksPerGrid.x   = 1;
        blocksPerGrid.y   = 1;
        blocksPerGrid.z   = 1;
        params.clear();
        params.push_back(phy_rowdata & 0xFFFFFFFF);
        params.push_back((phy_rowdata >> 32) & 0xFFFFFFFF);
        params.push_back(sram_base & 0xFFFFFFFF);
        params.push_back((sram_base >> 32) & 0xFFFFFFFF);
        params.push_back(elements_per_row * in_bytes_per_element);
        launchWrapperAysnc("dma_row_ddr_to_sram", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        if ((in_bytes_per_element == sizeof(float)) && (out_bytes_per_element == sizeof(rpp::bfloat16))) {
            threadsPerBlock.x = elements_per_row;
            threadsPerBlock.y = 1;
            threadsPerBlock.z = 1;
            blocksPerGrid.x   = 1;
            blocksPerGrid.y   = 1;
            blocksPerGrid.z   = 1;
            if (threadsPerBlock.x >= 8192) {
                throw std::runtime_error("Row Size 8K not Supportted");
            }
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sram_base, sram_baseA, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            threadsPerBlock.x = 32;
            threadsPerBlock.y = 1;
            threadsPerBlock.z = 1;
            blocksPerGrid.x   = 1;
            blocksPerGrid.y   = 1;
            blocksPerGrid.z   = 1;
            params.clear();
            params.push_back(sram_baseA & 0xFFFFFFFF);
            params.push_back((sram_baseA >> 32) & 0xFFFFFFFF);
            params.push_back(phy_out & 0xFFFFFFFF);
            params.push_back((phy_out >> 32) & 0xFFFFFFFF);
            params.push_back(phy_rowid & 0xFFFFFFFF);
            params.push_back((phy_rowid >> 32) & 0xFFFFFFFF);
            params.push_back(rowid_fram0 & 0xFFFFFFFF);
            params.push_back(rowid_fram1 & 0xFFFFFFFF);
            params.push_back((rowid_fram0 >> 32) & 0xFFFFFFFF);
            params.push_back(elements_per_row * out_bytes_per_element);

            launchWrapperAysnc("dma_row_sram_to_ddr", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        } else {
            throw std::runtime_error("Set Row From BF16 To Float32 not Supportted");
        }
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
        launchWrapperAysnc("dma_row_ddr_to_ddr", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
