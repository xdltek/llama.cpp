#pragma once

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace kernel_tanh {

static RPPdeviceptr tanh_prepare_lut_workspace(rpp_kernel_context & ctx, int lut_elements) {
    static std::mutex   mutex;
    static RPPdeviceptr kernel_lut_workspace = 0;

    std::lock_guard<std::mutex> lock(mutex);
    const int lut_bytes = lut_elements * (int) sizeof(uint16_t);

    if (kernel_lut_workspace == 0) {
        if (rtMalloc((void **) &kernel_lut_workspace, lut_bytes) != rtSuccess) {
            throw std::runtime_error("Tanh rtMalloc failed for shared LUT workspace");
        }

        std::vector<uint16_t> tanh_table(lut_elements);
        for (uint32_t i = 0; i < (uint32_t) lut_elements; i++) {
            uint32_t x0 = i;
            x0 <<= 16;
            float x;
            std::memcpy(&x, &x0, sizeof(float));
            tanh_table[i] = rpp::bfloat16::round_to_bfloat16(std::tanh(x)).value;
        }

        rtMemcpy((void *) kernel_lut_workspace, (const void *) tanh_table.data(), lut_bytes, rtMemcpyHostToDevice);
    }

    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        ctx.dev_workspace = kernel_lut_workspace;
        return kernel_lut_workspace;
    }
    if (dev_lut_workspace != kernel_lut_workspace) {
        rtMemcpy((void *) dev_lut_workspace, (const void *) kernel_lut_workspace, lut_bytes, rtMemcpyDeviceToDevice);
    }
    return dev_lut_workspace;
}

static void rpp_tanh_build(rpp_kernel_context & ctx,
                           int                  C,
                           int                  H,
                           int                  W,
                           int                  in_bytes_per_element,
                           int                  out_bytes_per_element,
                           int                  is_instantial = 1,
                           bool                 write_mid_workspace = false) {
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA = ctx.dev_in[0];
    RPPdeviceptr devB = ctx.dev_out[0];

    const int elements = C * H * W;
    const int norm_tiles = 1024 * 1024;
    int       num_of_tiles = 0;
    int       tail_tiles   = 0;
    get_linear_blocks(elements, norm_tiles, num_of_tiles, tail_tiles);

    const bool   has_mid_workspace = write_mid_workspace && ctx.dev_workspace != 0;
    const int    lut_elements      = 64 * 1024;
    RPPdeviceptr dev_tanh_lut      = tanh_prepare_lut_workspace(ctx, lut_elements);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/tanh.o");

    const int    lutSize         = 64 * 1024 * (int) sizeof(uint16_t);
    RPPdeviceptr sram_base       = ctx.virtual_sram_base;
    RPPdeviceptr tanh_table_addr = sram_base;
    RPPdeviceptr dev_tanh_mid    = has_mid_workspace ? (ctx.dev_workspace + lutSize) : 0;
    rtMemcpyAsync((void *) tanh_table_addr, (const void *) dev_tanh_lut, lutSize, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

    auto launch_tile = [&](int tile_elements, int element_offset) {
        const int input_offset  = element_offset * in_bytes_per_element;
        const int output_offset = element_offset * out_bytes_per_element;
        const int mid_offset    = element_offset * (int) sizeof(uint16_t);
        const int sizeA_raw  = tile_elements * in_bytes_per_element;
        const int sizeA_bf16 = tile_elements * (int) sizeof(uint16_t);
        const int sizeB_raw  = tile_elements * out_bytes_per_element;

        RPPdeviceptr sramA_raw     = tanh_table_addr + round_up(lutSize);
        RPPdeviceptr sramA_bf16    = sramA_raw + round_up(sizeA_raw);
        RPPdeviceptr sramOut_bf16  = (in_bytes_per_element == (int) sizeof(uint16_t)) ?
                                         (sramA_raw + round_up(sizeA_raw)) :
                                         (sramA_bf16 + round_up(sizeA_bf16));
        RPPdeviceptr sramOut_final = sramOut_bf16 + round_up(sizeA_bf16);

        const int total_sram_bytes = (int) ((out_bytes_per_element == (int) sizeof(float) ?
                                                 (sramOut_final + round_up(sizeB_raw)) :
                                                 (sramOut_bf16 + round_up(sizeA_bf16))) -
                                            sram_base);
        const int SRAM_LIMIT = 22 * 1024 * 1024;
        if (total_sram_bytes > SRAM_LIMIT) {
            std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                      << " bytes\n";
            std::abort();
        }

        rtMemcpyAsync((void *) sramA_raw, (const void *) (devA + input_offset), sizeA_raw, rtMemcpyDeviceToSram,
                      ctx.kernelStream);

        RPPdeviceptr tanh_input = sramA_raw;
        if (in_bytes_per_element == (int) sizeof(float)) {
            calc_tbdim_flattern(1, tile_elements, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA_raw, sramA_bf16, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            tanh_input = sramA_bf16;
        }

        calc_tbdim_flattern(1, tile_elements, threadsPerBlock, blocksPerGrid);
        params.clear();
        params.emplace_back(tanh_input);
        params.emplace_back(sramOut_bf16);
        params.emplace_back(tanh_table_addr);
        params.emplace_back(threadsPerBlock.x * (int) sizeof(uint16_t));
        launchWrapperAysnc("mish_f16_f32_f16_all", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        if (has_mid_workspace) {
            rtMemcpyAsync((void *) (dev_tanh_mid + mid_offset), (const void *) sramOut_bf16, sizeA_bf16,
                          rtMemcpySramToDevice, ctx.kernelStream);
        }

        if (out_bytes_per_element == (int) sizeof(float)) {
            calc_tbdim_flattern(1, tile_elements * 2, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init_opt(threadsPerBlock, sramOut_bf16, sramOut_final, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            rtMemcpyAsync((void *) (devB + output_offset), (const void *) sramOut_final, sizeB_raw,
                          rtMemcpySramToDevice, ctx.kernelStream);
        } else {
            rtMemcpyAsync((void *) (devB + output_offset), (const void *) sramOut_bf16, sizeB_raw,
                          rtMemcpySramToDevice, ctx.kernelStream);
        }
    };

    for (int i = 0; i < num_of_tiles - 1; i++) {
        launch_tile(norm_tiles, i * norm_tiles);
    }

    if (tail_tiles > 0) {
        const int tail_idx = num_of_tiles - 1;
        launch_tile(tail_tiles, tail_idx * norm_tiles);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

}  // namespace kernel_tanh
