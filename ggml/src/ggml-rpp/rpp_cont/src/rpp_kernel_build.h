// cont_rpp.cpp
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_cont/src/rpp_kernel_block.h"
#include "rpp_cont/src/rpp_kernel_param.h"
#include "rpp_drv_api.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

static inline bool rpp_cont_supports_dma(int use_dyn_d0,
                                         int oD0,
                                         int oD1,
                                         int oD2,
                                         int in_stride_x,
                                         int in_stride_y,
                                         int in_stride_z) {
    const bool cont_x = (in_stride_x == 1);
    const bool cont_y = (in_stride_y == oD2);
    const bool cont_z = (in_stride_z == oD1 * oD2);

    if (cont_x && cont_y && cont_z && !use_dyn_d0) {
        return true;
    }
    if (cont_x && (cont_y || (use_dyn_d0 && oD0 == 1 && oD1 > 1))) {
        return true;
    }
    if (cont_x) {
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Public entry: run cont on RPP without SRAM
// hostB = output, hostA = input
// oD0/oD1/oD2 = output dims
// -----------------------------------------------------------------------------
void rpp_cont_build(rpp_kernel_context & ctx,
                    int                  use_dyn_d0,
                    int                  oD0,
                    int                  oD1,
                    int                  oD2,
                    int                  in_stride_x,
                    int                  in_stride_y,
                    int                  in_stride_z,
                    int                  bytes_per_element,
                    int                  is_instantial = 1) {
    // -----------------------------------------------------------------
    // 1) Get variable from context
    // -----------------------------------------------------------------
    RPPdeviceptr devIn     = ctx.dev_in[0];
    RPPdeviceptr devOut    = ctx.dev_out[0];
    auto         dmaStream = ctx.kernelStream;

    // -----------------------------------------------------------------
    // 2) Capture graph: DMA only (DDR input/output)
    // -----------------------------------------------------------------
    const bool cont_x = (in_stride_x == 1);
    const bool cont_y = (in_stride_y == oD2);
    const bool cont_z = (in_stride_z == oD1 * oD2);

    rppStreamBeginCapture(dmaStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/cont.o");

    if (cont_x && cont_y && cont_z && !use_dyn_d0) {
        // Fully contiguous: 1D DMA for whole tensor
        const size_t size_bytes = (size_t) oD0 * (size_t) oD1 * (size_t) oD2 * (size_t) bytes_per_element;
        rtMemcpyAsync((void *) devOut, (void *) devIn, size_bytes, rtMemcpyDeviceToDevice, dmaStream);
    }
    // Allow dyn-d0 for D1 slices even if in_stride_y != oD2
    else if (cont_x && (cont_y || (use_dyn_d0 && oD0 == 1 && oD1 > 1))) {
        // Last two dims contiguous: 1D DMA per slice (dynamic or static highest dimension)
        size_t slice_bytes      = 0;
        size_t in_slice_stride  = 0;
        size_t out_slice_stride = 0;
        int    dyn_dim          = -1;

        if (oD0 > 1) {
            // highest dim is D0
            dyn_dim          = 0;
            slice_bytes      = (size_t) oD1 * (size_t) oD2 * (size_t) bytes_per_element;
            in_slice_stride  = (size_t) in_stride_z * (size_t) bytes_per_element;
            out_slice_stride = (size_t) oD1 * (size_t) oD2 * (size_t) bytes_per_element;
        } else if (oD1 > 1) {
            // highest dim is D1 (D0 == 1)
            dyn_dim          = 1;
            slice_bytes      = (size_t) oD2 * (size_t) bytes_per_element;
            in_slice_stride  = (size_t) in_stride_y * (size_t) bytes_per_element;
            out_slice_stride = (size_t) oD2 * (size_t) bytes_per_element;
        }

        if (use_dyn_d0) {
            if (dyn_dim < 0) {
                throw std::runtime_error("Not Support CONT with dynamic D0 when highest dim is D2");
            }

            dim3                  threadsPerBlock;
            dim3                  threadsPerBlockTail;
            dim3                  blocksPerGrid;
            std::vector<uint32_t> params;
            RPPdeviceptr          phy_in, phy_out;
            rppMemGetPhyAddr(&phy_in, devIn);
            rppMemGetPhyAddr(&phy_out, devOut);
            threadsPerBlock.x = 32;
            threadsPerBlock.y = 1;
            threadsPerBlock.z = 1;
            blocksPerGrid.x   = 1;
            blocksPerGrid.y   = 1;
            blocksPerGrid.z   = 1;
            params.clear();
            params.push_back(phy_in & 0xFFFFFFFF);
            params.push_back((phy_in >> 32) & 0xFFFFFFFF);
            params.push_back(phy_out & 0xFFFFFFFF);
            params.push_back((phy_out >> 32) & 0xFFFFFFFF);
            params.push_back(in_slice_stride);
            params.push_back(out_slice_stride);
            params.push_back(slice_bytes);
            launchWrapperAysnc("dma_dyn_d0_2d", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        } else {
            for (int z = 0; z < oD0; ++z) {
                void * src = (void *) ((uint8_t *) devIn + z * in_slice_stride);
                void * dst = (void *) ((uint8_t *) devOut + z * out_slice_stride);
                rtMemcpyAsync(dst, src, slice_bytes, rtMemcpyDeviceToDevice, dmaStream);
            }
        }
    } else if (cont_x) {
        // Last dim contiguous: 2D DMA per slice (Z)
        if (use_dyn_d0) {
            throw std::runtime_error("Not Support CONT with dynamic D0");
        }
        const size_t width_bytes      = (size_t) oD2 * (size_t) bytes_per_element;
        const size_t src_pitch        = (size_t) in_stride_y * (size_t) bytes_per_element;
        const size_t dst_pitch        = (size_t) oD2 * (size_t) bytes_per_element;
        const size_t in_slice_stride  = (size_t) in_stride_z * (size_t) bytes_per_element;
        const size_t out_slice_stride = (size_t) oD1 * (size_t) oD2 * (size_t) bytes_per_element;
        for (int z = 0; z < oD0; ++z) {
            void * src = (void *) ((uint8_t *) devIn + z * in_slice_stride);
            void * dst = (void *) ((uint8_t *) devOut + z * out_slice_stride);
            rtMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width_bytes, oD1, rtMemcpyDeviceToDevice, dmaStream);
        }
    } else {
        assert(false && "cont DMA: unsupported stride pattern");
    }

    rppStreamEndCapture(dmaStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
