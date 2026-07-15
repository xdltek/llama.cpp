#pragma once

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace kernel_clamp {

struct clamp_lut_state {
    uint32_t     min_bits;
    uint32_t     max_bits;
    RPPdeviceptr dev_lut;
};

static RPPdeviceptr clamp_prepare_lut_workspace(rpp_kernel_context & ctx, float clamp_min, float clamp_max) {
    constexpr int lut_elements = 64 * 1024;
    constexpr int lut_size     = lut_elements * (int) sizeof(uint16_t);

    uint32_t min_bits = 0;
    uint32_t max_bits = 0;
    std::memcpy(&min_bits, &clamp_min, sizeof(min_bits));
    std::memcpy(&max_bits, &clamp_max, sizeof(max_bits));

    static std::mutex                   mutex;
    static std::vector<clamp_lut_state> kernel_luts;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = std::find_if(kernel_luts.begin(), kernel_luts.end(), [&](const clamp_lut_state & state) {
        return state.min_bits == min_bits && state.max_bits == max_bits;
    });
    if (it == kernel_luts.end()) {
        RPPdeviceptr kernel_lut_workspace = 0;
        rtMalloc((void **) &kernel_lut_workspace, lut_size);

        std::vector<uint16_t> clamp_table(lut_elements);
        for (uint32_t i = 0; i < (uint32_t) lut_elements; ++i) {
            uint32_t x_bits = i << 16;
            float x;
            std::memcpy(&x, &x_bits, sizeof(float));
            const float y = std::max(std::min(x, clamp_max), clamp_min);
            clamp_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
        }

        rtMemcpy((void *) kernel_lut_workspace, (const void *) clamp_table.data(), lut_size, rtMemcpyHostToDevice);
        kernel_luts.push_back(clamp_lut_state{ min_bits, max_bits, kernel_lut_workspace });
        it = kernel_luts.end() - 1;
    }

    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        ctx.dev_workspace = it->dev_lut;
        return it->dev_lut;
    }
    if (dev_lut_workspace != it->dev_lut) {
        rtMemcpy((void *) dev_lut_workspace, (const void *) it->dev_lut, lut_size, rtMemcpyDeviceToDevice);
    }
    return dev_lut_workspace;
}

static void rpp_clamp_build(rpp_kernel_context & ctx,
                            int                  C,
                            int                  H,
                            int                  W,
                            float                clamp_min,
                            float                clamp_max,
                            int                  in_bytes_per_element,
                            int                  out_bytes_per_element,
                            int                  is_instantial = 1) {
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

    const int    lut_elements  = 64 * 1024;
    RPPdeviceptr dev_clamp_lut = clamp_prepare_lut_workspace(ctx, clamp_min, clamp_max);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/tanh.o");

    const int    lut_size       = lut_elements * (int) sizeof(uint16_t);
    RPPdeviceptr sram_base      = ctx.virtual_sram_base;
    RPPdeviceptr sram_lut       = sram_base;
    rtMemcpyAsync((void *) sram_lut, (const void *) dev_clamp_lut, lut_size, rtMemcpyDeviceToSram, ctx.kernelStream);

    auto launch_tile = [&](int tile_elements, int element_offset) {
        const int input_offset  = element_offset * in_bytes_per_element;
        const int output_offset = element_offset * out_bytes_per_element;
        const int size_in_raw   = tile_elements * in_bytes_per_element;
        const int size_in_bf16  = tile_elements * (int) sizeof(uint16_t);
        const int size_out_raw  = tile_elements * out_bytes_per_element;

        RPPdeviceptr sram_in_raw   = sram_lut + round_up(lut_size);
        RPPdeviceptr sram_in_bf16  = sram_in_raw + round_up(size_in_raw);
        RPPdeviceptr sram_out_bf16 = (in_bytes_per_element == (int) sizeof(uint16_t)) ?
                                         (sram_in_raw + round_up(size_in_raw)) :
                                         (sram_in_bf16 + round_up(size_in_bf16));
        RPPdeviceptr sram_out_f32  = sram_out_bf16 + round_up(size_in_bf16);

        const int total_sram_bytes = (int) ((out_bytes_per_element == (int) sizeof(float) ?
                                                 (sram_out_f32 + round_up(size_out_raw)) :
                                                 (sram_out_bf16 + round_up(size_in_bf16))) -
                                            sram_base);
        const int SRAM_LIMIT = 22 * 1024 * 1024;
        if (total_sram_bytes > SRAM_LIMIT) {
            std::abort();
        }

        rtMemcpyAsync((void *) sram_in_raw, (const void *) (devA + input_offset), size_in_raw, rtMemcpyDeviceToSram,
                      ctx.kernelStream);

        RPPdeviceptr clamp_input = sram_in_raw;
        if (in_bytes_per_element == (int) sizeof(float)) {
            calc_tbdim_flattern(1, tile_elements, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sram_in_raw, sram_in_bf16, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            clamp_input = sram_in_bf16;
        }

        calc_tbdim_flattern(1, tile_elements, threadsPerBlock, blocksPerGrid);
        params.clear();
        params.emplace_back(clamp_input);
        params.emplace_back(sram_out_bf16);
        params.emplace_back(sram_lut);
        params.emplace_back(threadsPerBlock.x * (int) sizeof(uint16_t));
        launchWrapperAysnc("mish_f16_f32_f16_all", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        if (out_bytes_per_element == (int) sizeof(float)) {
            calc_tbdim_flattern(1, tile_elements * 2, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init_opt(threadsPerBlock, sram_out_bf16, sram_out_f32, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            rtMemcpyAsync((void *) (devB + output_offset), (const void *) sram_out_f32, size_out_raw,
                          rtMemcpySramToDevice, ctx.kernelStream);
        } else {
            rtMemcpyAsync((void *) (devB + output_offset), (const void *) sram_out_bf16, size_out_raw,
                          rtMemcpySramToDevice, ctx.kernelStream);
        }
    };

    for (int i = 0; i < num_of_tiles - 1; ++i) {
        launch_tile(norm_tiles, i * norm_tiles);
    }

    if (tail_tiles > 0) {
        launch_tile(tail_tiles, (num_of_tiles - 1) * norm_tiles);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);

    uint32_t min_bits = 0;
    uint32_t max_bits = 0;
    memcpy(&min_bits, &clamp_min, sizeof(min_bits));
    memcpy(&max_bits, &clamp_max, sizeof(max_bits));
    // Clamp-local cache isolation: include runtime addresses in the graph key so
    // graph instantiation does not reuse captured LUT/SRAM bindings incorrectly.
    const uint64_t ctx_workspace_key = (uint64_t) ctx.dev_workspace;
    const uint64_t ctx_sram_key      = (uint64_t) ctx.virtual_sram_base;
    const uint64_t ctx_graph_key     = (uint64_t) (uintptr_t) ctx.graph;
    const std::string graph_key =
        rpp_join_function_name_and_args(__func__, C, H, W, min_bits, max_bits, in_bytes_per_element,
                                        out_bytes_per_element, ctx_workspace_key, ctx_sram_key, ctx_graph_key);
    if (rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial) != RPP_SUCCESS) {
        throw std::runtime_error("rpp_graph_instantiate failed.");
    }
}

}  // namespace kernel_clamp
