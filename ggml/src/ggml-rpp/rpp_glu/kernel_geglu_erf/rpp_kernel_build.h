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
#include <vector>

namespace kernel_geglu_erf {

static RPPdeviceptr gelu_prepare_lut_workspace_once(rpp_kernel_context & ctx, int gelu_variant, int lut_elements) {
    static std::mutex   mutex;
    static RPPdeviceptr kernel_lut_workspaces[3] = {};

    std::lock_guard<std::mutex> lock(mutex);
    assert(gelu_variant >= 0 && gelu_variant < 3);
    RPPdeviceptr & kernel_lut_workspace = kernel_lut_workspaces[gelu_variant];
    const size_t   lut_bytes            = (size_t) lut_elements * sizeof(uint16_t);

    if (kernel_lut_workspace == 0) {
        rtMalloc((void **) &kernel_lut_workspace, lut_bytes);

        std::vector<uint16_t> gelu_table(lut_elements);
        for (uint32_t i = 0; i < (uint32_t) lut_elements; i++) {
            uint32_t x0 = i;
            x0 <<= 16;
            float x;
            memcpy(&x, &x0, sizeof(x));
            float y;
            if (gelu_variant == 1) {
                const float inv_sqrt2 = 0.70710678118654752440f;
                y = 0.5f * x * (1.0f + std::erff(x * inv_sqrt2));
            } else if (gelu_variant == 2) {
                y = x / (1.0f + std::exp(-1.702f * x));
            } else {
                const float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
                const float gelu_coef_a    = 0.044715f;
                y = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * x * (1.0f + gelu_coef_a * x * x)));
            }
            gelu_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
        }

        rtMemcpy((void *) kernel_lut_workspace, (const void *) gelu_table.data(), lut_bytes, rtMemcpyHostToDevice);
    }

    RPPdeviceptr dev_gelu_lut = ctx.dev_workspace;
    if (dev_gelu_lut == 0) {
        ctx.dev_workspace = kernel_lut_workspace;
        return kernel_lut_workspace;
    }
    if (dev_gelu_lut != kernel_lut_workspace) {
        rtMemcpy((void *) dev_gelu_lut, (const void *) kernel_lut_workspace, lut_bytes, rtMemcpyDeviceToDevice);
    }
    return dev_gelu_lut;
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_gelu_build(rpp_kernel_context & ctx,
                           int                  mode,
                           int                  C,
                           int                  H,
                           int                  W,
                           int                  split_axis,  // used only by mode2; ignored otherwise
                           int                  in_bytes_per_element,
                           int                  out_bytes_per_element,
                           int                  gelu_variant, // 0=tanh, 1=erf, 2=quick
                           int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA, devA1;
    devA = ctx.dev_in[0];
    if (mode == 1) {
        devA1 = ctx.dev_in[1];
    }
    RPPdeviceptr devB = ctx.dev_out[0];
    int          num_of_tiles, norm_tiles, tail_tiles;
    int          elements = C * H * W;

    if (mode == 0) {
        norm_tiles = 2 * 1024 * 1024;
        get_linear_blocks(elements, norm_tiles, num_of_tiles, tail_tiles);
    } else if (mode == 1) {
        norm_tiles = 1024 * 1024;
        get_linear_blocks(elements, norm_tiles, num_of_tiles, tail_tiles);
    } else {
        throw std::runtime_error("Not Support");
        norm_tiles   = C * H * W / 2;
        tail_tiles   = C * H * W / 2;
        num_of_tiles = 1;
        if (C * H * W >= 4 * 1024 * 1024) {
            throw std::runtime_error("Gelu Mode2 Exceed 24M");
        }
    }

    // gelu_variant: 0 -> tanh GELU, 1 -> erf GELU, 2 -> quick GELU.
    const int    lut_elements = 64 * 1024;
    RPPdeviceptr dev_gelu_lut = gelu_prepare_lut_workspace_once(ctx, gelu_variant, lut_elements);
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/gelu.o");

    const int    lutSize         = 64 * 1024 * sizeof(short);
    RPPdeviceptr sram_base       = ctx.virtual_sram_base;
    RPPdeviceptr gelu_table_addr = sram_base;
    rtMemcpyAsync((void *) gelu_table_addr, (const void *) dev_gelu_lut, lutSize, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

    for (int i = 0; i < num_of_tiles - 1; i++) {
        // -------------------------
        // SRAM allocation planning
        // -------------------------
        int          sizeA, sizeB;
        RPPdeviceptr sramA, sramA1, sramB;
        if (mode == 0) {
            sizeA  = norm_tiles * in_bytes_per_element;
            sizeB  = norm_tiles * out_bytes_per_element;
            sramA  = gelu_table_addr + round_up(lutSize);
            sramA1 = 0;
            sramB  = sramA + round_up(sizeA);
        } else if (mode == 1) {
            sizeA  = norm_tiles * in_bytes_per_element;
            sizeB  = norm_tiles * out_bytes_per_element;
            sramA  = gelu_table_addr + round_up(lutSize);
            sramA1 = sramA + round_up(sizeA);
            sramB  = sramA1 + round_up(sizeA);
        }

        const int total_sram_bytes = (int) (sramB + round_up(sizeB) - sram_base);
        const int SRAM_LIMIT       = 22 * 1024 * 1024;
        if (total_sram_bytes > SRAM_LIMIT) {
            std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                      << " bytes\n";
            std::abort();
        }

        rtMemcpyAsync((void *) sramA, (const void *) (devA + i * sizeA), sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
        if (mode == 1) {
            rtMemcpyAsync((void *) sramA1, (const void *) (devA1 + i * sizeA), sizeA, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
        }

        if (in_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, norm_tiles, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            if (mode == 1) {
                calc_tbdim_flattern(1, norm_tiles, threadsPerBlock, blocksPerGrid);
                params.clear();
                cvt_kernel_param_init(threadsPerBlock, sramA1, sramA1, kFLOAT, kBF16, params);
                launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }

        int nr_of_blocks, Nt, Ns;
        Ns = 4096;
        // norm_tiles = (nr_of_blocks - 1) * Ns + Nt
        get_linear_blocks(norm_tiles, Ns, nr_of_blocks, Nt);

        int bytes_per_block = Ns * sizeof(rpp::bfloat16);
        if (nr_of_blocks > 1) {
            calc_tbdim_flattern(1, Ns, threadsPerBlock, blocksPerGrid);
            params.clear();
            params.push_back(sramA);
            params.push_back(sramA1);
            params.push_back(gelu_table_addr);
            params.push_back(sramA);
            params.push_back(bytes_per_block);
            params.push_back((nr_of_blocks - 1));
            params.push_back(mode);
            launchWrapperAysnc("llm_gelu", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
        }
        if (Nt > 0) {
            calc_tbdim_flattern(1, Nt, threadsPerBlock, blocksPerGrid);
            params.clear();
            params.push_back(sramA + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(sramA1 + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(gelu_table_addr);
            params.push_back(sramA + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(0);
            params.push_back(1);
            params.push_back(mode);
            launchWrapperAysnc("llm_gelu", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
        }
        if (out_bytes_per_element == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, norm_tiles * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramA, sramB, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpyAsync((void *) (devB + i * sizeB), (const void *) sramB, sizeB, rtMemcpySramToDevice,
                          ctx.kernelStream);
        } else {
            rtMemcpyAsync((void *) (devB + i * sizeB), (const void *) sramA, sizeB, rtMemcpySramToDevice,
                          ctx.kernelStream);
        }
    }

    if (tail_tiles > 0) {
        // launch for tail tiles
        const int sizeA                = tail_tiles * in_bytes_per_element;
        const int sizeB                = tail_tiles * out_bytes_per_element;
        int       tile_tail_offset_in  = (num_of_tiles - 1) * norm_tiles * in_bytes_per_element;
        int       tile_tail_offset_out = (num_of_tiles - 1) * norm_tiles * out_bytes_per_element;

        RPPdeviceptr sramA, sramB, sramA1;
        if (mode == 0) {
            sramA  = gelu_table_addr + round_up(lutSize);
            sramA1 = 0;
            sramB  = sramA + round_up(sizeA);
        } else if (mode == 1) {
            sramA  = gelu_table_addr + round_up(lutSize);
            sramA1 = sramA + round_up(sizeA);
            sramB  = sramA1 + round_up(sizeA);
        }
        const int total_sram_bytes = (int) (sramB + round_up(sizeB) - sram_base);
        const int SRAM_LIMIT       = 22 * 1024 * 1024;
        if (total_sram_bytes > SRAM_LIMIT) {
            std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                      << " bytes\n";
            std::abort();
        }
        rtMemcpyAsync((void *) sramA, (const void *) (devA + tile_tail_offset_in), sizeA, rtMemcpyDeviceToSram,
                      ctx.kernelStream);
        if (mode == 1) {
            rtMemcpyAsync((void *) sramA1, (const void *) (devA1 + tile_tail_offset_in), sizeA, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
        }

        if (in_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tail_tiles, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            if (mode == 1) {
                calc_tbdim_flattern(1, tail_tiles, threadsPerBlock, blocksPerGrid);
                params.clear();
                cvt_kernel_param_init(threadsPerBlock, sramA1, sramA1, kFLOAT, kBF16, params);
                launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }

        int nr_of_blocks, Nt, Ns;
        Ns = 4096;
        // norm_tiles = (nr_of_blocks - 1) * Ns + Nt
        get_linear_blocks(tail_tiles, Ns, nr_of_blocks, Nt);

        int bytes_per_block = Ns * sizeof(rpp::bfloat16);
        if (nr_of_blocks > 1) {
            calc_tbdim_flattern(1, Ns, threadsPerBlock, blocksPerGrid);
            params.clear();
            params.push_back(sramA);
            params.push_back(sramA1);
            params.push_back(gelu_table_addr);
            params.push_back(sramA);
            params.push_back(bytes_per_block);
            params.push_back((nr_of_blocks - 1));
            params.push_back(mode);
            launchWrapperAysnc("llm_gelu", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
        }
        if (Nt > 0) {
            calc_tbdim_flattern(1, Nt, threadsPerBlock, blocksPerGrid);
            params.clear();
            params.push_back(sramA + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(sramA1 + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(gelu_table_addr);
            params.push_back(sramA + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(0);
            params.push_back(1);
            params.push_back(mode);
            launchWrapperAysnc("llm_gelu", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
        }
        if (out_bytes_per_element == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, tail_tiles * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramA, sramB, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpyAsync((void *) (devB + tile_tail_offset_out), (const void *) sramB, sizeB, rtMemcpySramToDevice,
                          ctx.kernelStream);
        } else {
            rtMemcpyAsync((void *) (devB + tile_tail_offset_out), (const void *) sramA, sizeB, rtMemcpySramToDevice,
                          ctx.kernelStream);
        }
    }
    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    const std::string graph_key =
        rpp_join_function_name_and_args(__func__, mode, C, H, W, split_axis, in_bytes_per_element,
                                        out_bytes_per_element);
    if (rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial) != RPP_SUCCESS) {
        throw std::runtime_error("rpp_graph_instantiate failed.");
    }
}
}  // namespace kernel_geglu_erf
