#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_scale/src/rpp_kernel_block.h"
#include "rpp_scale/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// -----------------------------
// Build graph once
// -----------------------------
// jsgelu
void rpp_scale_build(rpp_kernel_context & ctx,
                     int                  elements,
                     float                scale,
                     float                bias,
                     int                  in_bytes_per_element,
                     int                  out_bytes_per_element,
                     int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA = ctx.dev_in[0];
    RPPdeviceptr          devB = ctx.dev_out[0];
    int                   num_of_tiles, norm_tiles, tail_tiles;
    norm_tiles = 1024 * 1024;
    get_linear_blocks(elements, norm_tiles, num_of_tiles, tail_tiles);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/scale.o");
    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    for (int i = 0; i < num_of_tiles - 1; i++) {
        // -------------------------
        // SRAM allocation planning
        // -------------------------
        const int    sizeA            = norm_tiles * in_bytes_per_element;
        const int    sizeB            = norm_tiles * out_bytes_per_element;
        RPPdeviceptr sramA            = sram_base;
        RPPdeviceptr sramB            = sramA + round_up(sizeA);
        const int    total_sram_bytes = (int) (sramB + round_up(sizeB) - sram_base);
        const int    SRAM_LIMIT       = 22 * 1024 * 1024;
        if (total_sram_bytes > SRAM_LIMIT) {
            std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                      << " bytes\n";
            std::abort();
        }

        rtMemcpyAsync((void *) sramA, (const void *) (devA + i * sizeA), sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);

        if (in_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, norm_tiles, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
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
            params.push_back(*(uint32_t *) &scale);
            params.push_back(*(uint32_t *) &bias);
            params.push_back(sramA);
            params.push_back(bytes_per_block);
            params.push_back((nr_of_blocks - 1));
            launchWrapperAysnc("llm_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
        }
        if (Nt > 0) {
            calc_tbdim_flattern(1, Nt, threadsPerBlock, blocksPerGrid);
            params.clear();
            params.push_back(sramA + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(*(uint32_t *) &scale);
            params.push_back(*(uint32_t *) &bias);
            params.push_back(sramA + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(0);
            params.push_back(1);
            launchWrapperAysnc("llm_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
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
        const int    sizeA                = tail_tiles * in_bytes_per_element;
        const int    sizeB                = tail_tiles * out_bytes_per_element;
        int          tile_tail_offset_in  = (num_of_tiles - 1) * norm_tiles * in_bytes_per_element;
        int          tile_tail_offset_out = (num_of_tiles - 1) * norm_tiles * out_bytes_per_element;
        RPPdeviceptr sramA                = sram_base;
        RPPdeviceptr sramB                = sramA + round_up(sizeA);
        const int    total_sram_bytes     = (int) (sramB + round_up(sizeB) - sram_base);
        const int    SRAM_LIMIT           = 22 * 1024 * 1024;
        if (total_sram_bytes > SRAM_LIMIT) {
            std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                      << " bytes\n";
            std::abort();
        }
        rtMemcpyAsync((void *) sramA, (const void *) (devA + tile_tail_offset_in), sizeA, rtMemcpyDeviceToSram,
                      ctx.kernelStream);

        if (in_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tail_tiles, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
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
            params.push_back(*(uint32_t *) &scale);
            params.push_back(*(uint32_t *) &bias);
            params.push_back(sramA);
            params.push_back(bytes_per_block);
            params.push_back((nr_of_blocks - 1));
            launchWrapperAysnc("llm_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
        }
        if (Nt > 0) {
            calc_tbdim_flattern(1, Nt, threadsPerBlock, blocksPerGrid);
            params.clear();
            params.push_back(sramA + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(*(uint32_t *) &scale);
            params.push_back(*(uint32_t *) &bias);
            params.push_back(sramA + (nr_of_blocks - 1) * bytes_per_block);
            params.push_back(0);
            params.push_back(1);
            launchWrapperAysnc("llm_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
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
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
