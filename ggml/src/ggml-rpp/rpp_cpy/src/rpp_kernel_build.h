#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_cpy/src/rpp_kernel_block.h"
#include "rpp_cpy/src/rpp_kernel_param.h"
#include "rpp_drv_api.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

inline int round_up_cpy(int a) {
    return (a + 511) / 512 * 512 + 512;
}

}  // namespace

void rpp_copy_build(rpp_kernel_context & ctx,
                    int                  elements,
                    RppDataType          inType,
                    RppDataType          outType,
                    int                  is_instantial = 1) {
    if (elements <= 0) {
        throw std::runtime_error("copy requires elements > 0");
    }
    if (ctx.dev_in.empty() || ctx.dev_out.empty()) {
        throw std::runtime_error("copy requires ctx.dev_in[0] and ctx.dev_out[0]");
    }

    const bool is_f32_to_bf16 = (inType == kFLOAT) && (outType == kBF16);
    const bool is_bf16_to_f32 = (inType == kBF16) && (outType == kFLOAT);
    if (!is_f32_to_bf16 && !is_bf16_to_f32) {
        throw std::runtime_error("copy currently only supports FLOAT->BF16 and BF16->FLOAT");
    }

    const int in_bytes_per_element  = (int) GetRppElementSize(inType);
    const int out_bytes_per_element = (int) GetRppElementSize(outType);
    const int norm_tiles            = 1024 * 1024;
    int       num_of_tiles          = 0;
    int       tail_tiles            = 0;
    get_linear_blocks(elements, norm_tiles, num_of_tiles, tail_tiles);

    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;
    const RPPdeviceptr    devA      = ctx.dev_in[0];
    const RPPdeviceptr    devB      = ctx.dev_out[0];
    const RPPdeviceptr    sram_base = ctx.virtual_sram_base;

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/copy.o");

    for (int i = 0; i < num_of_tiles; ++i) {
        const int tile_elems      = (i == num_of_tiles - 1) ? tail_tiles : norm_tiles;
        const int sizeA           = tile_elems * in_bytes_per_element;
        const int sizeB           = tile_elems * out_bytes_per_element;
        const int tile_in_offset  = i * norm_tiles * in_bytes_per_element;
        const int tile_out_offset = i * norm_tiles * out_bytes_per_element;

        RPPdeviceptr sramA            = sram_base;
        RPPdeviceptr sramB            = sramA + round_up_cpy(sizeA);
        const int    total_sram_bytes = (int) (sramB + round_up_cpy(sizeB) - sram_base);
        const int    SRAM_LIMIT       = 22 * 1024 * 1024;
        if (total_sram_bytes > SRAM_LIMIT) {
            std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                      << " bytes\n";
            std::abort();
        }

        rtMemcpyAsync((void *) sramA, (const void *) (devA + tile_in_offset), sizeA, rtMemcpyDeviceToSram,
                      ctx.kernelStream);

        if (is_f32_to_bf16) {
            params.clear();
            calc_tbdim_flattern(1, tile_elems, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init(threadsPerBlock, (uint32_t) sramA, (uint32_t) sramB, inType, outType, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpyAsync((void *) (devB + tile_out_offset), (const void *) sramB, sizeB, rtMemcpySramToDevice,
                          ctx.kernelStream);
        } else {
            params.clear();
            calc_tbdim_flattern(1, tile_elems * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, (uint32_t) sramA, (uint32_t) sramB, inType, outType, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpyAsync((void *) (devB + tile_out_offset), (const void *) sramB, sizeB, rtMemcpySramToDevice,
                          ctx.kernelStream);
        }
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        if (rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0) != RPP_SUCCESS) {
            throw std::runtime_error("rppGraphInstantiate failed.");
        }
    }    
}
