#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_reduce_sum/src/rpp_kernel_block.h"
#include "rpp_reduce_sum/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

// Build graph once
// axis=1: input CxHxW -> output Cx1xW (sum over H)
static void rpp_reduce_sum_build(rpp_kernel_context & ctx,
                                 int                  C,
                                 int                  H,
                                 int                  W,
                                 int                  axis,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  is_instantial = 1) {
    if (axis != 1) {
        throw std::runtime_error("ReduceSum currently supports axis=1 only");
    }
    if (C <= 0 || H <= 0 || W <= 0) {
        throw std::runtime_error("ReduceSum invalid input dims");
    }

    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA = ctx.dev_in[0];
    RPPdeviceptr devB = ctx.dev_out[0];

    const int in_elems  = C * H * W;
    const int out_elems = C * W;

    const int sizeA     = in_elems * in_bytes_per_element;
    const int sizeOut   = out_elems * out_bytes_per_element;
    const int sizeIn16  = in_elems * (int) sizeof(uint16_t);
    const int sizeOut16 = out_elems * (int) sizeof(uint16_t);
    const int sizeOut32 = out_elems * (int) sizeof(float);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/reduce_sum.o");

    // -------------------------
    // SRAM allocation planning
    // -------------------------
    RPPdeviceptr sram_base       = ctx.virtual_sram_base;
    RPPdeviceptr sramInput       = sram_base;
    RPPdeviceptr workspace0_addr = sramInput + round_up(sizeA);
    RPPdeviceptr workspace1_addr = workspace0_addr + round_up(sizeIn16);
    RPPdeviceptr workspace2_addr = workspace1_addr + round_up(sizeIn16);
    RPPdeviceptr sramOut16       = workspace2_addr + round_up(sizeIn16);
    RPPdeviceptr sramOut32       = sramOut16 + round_up(sizeOut16);

    const int total_sram_bytes =
        (int) ((out_bytes_per_element == (int) sizeof(float) ? (sramOut32 + round_up(sizeOut32)) :
                                                               (sramOut16 + round_up(sizeOut16))) -
               sram_base);

    const int SRAM_LIMIT = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    rtMemcpyAsync((void *) sramInput, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);

    RPPdeviceptr reduce_in_addr = sramInput;
    if (in_bytes_per_element == (int) sizeof(float)) {
        calc_tbdim_flattern(1, in_elems, threadsPerBlock, blocksPerGrid);
        params.clear();
        cvt_kernel_param_init(threadsPerBlock, sramInput, workspace2_addr, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        reduce_in_addr = workspace2_addr;
    }

    RPPdeviceptr reduce_out_addr = sramOut16;

    RppDims reduce_in_dims{}, reduce_out_dims{};
    reduce_in_dims.nbDims = 3;
    reduce_in_dims.d[0]   = C;
    reduce_in_dims.d[1]   = H;
    reduce_in_dims.d[2]   = W;

    reduce_out_dims.nbDims = 3;
    reduce_out_dims.d[0]   = C;
    reduce_out_dims.d[1]   = 1;
    reduce_out_dims.d[2]   = W;

    std::vector<RppDims> sub_reduce_io_dims;
    RppDims              middle_dims{};
    sub_reduce_io_dims.emplace_back(reduce_in_dims);
    sub_reduce_io_dims.emplace_back(reduce_out_dims);

    int io_cursor = 0;
    while (ReduceSpawnIO(axis, sub_reduce_io_dims[io_cursor], sub_reduce_io_dims[sub_reduce_io_dims.size() - 1],
                         middle_dims, true, false)) {
        sub_reduce_io_dims.insert(sub_reduce_io_dims.begin() + io_cursor + 1, middle_dims);
        io_cursor++;
    }

    uint32_t ping_pong_addr[2] = { workspace0_addr, workspace1_addr };
    for (size_t i = 0; i < sub_reduce_io_dims.size() - 1; i++) {
        RppDims  in_dims     = sub_reduce_io_dims[i];
        RppDims  out_dims    = sub_reduce_io_dims[i + 1];
        uint32_t input_addr  = (i == 0 ? reduce_in_addr : ping_pong_addr[i % 2]);
        uint32_t output_addr = (i == sub_reduce_io_dims.size() - 2 ? reduce_out_addr : ping_pong_addr[(i + 1) % 2]);

        std::vector<RppTaskElement> tasks = create_reduce_kernel_task(
            (size_t) axis, RppReduceOperation::kSUM, input_addr, output_addr, in_dims, out_dims, 0, true, false);

        for (RppTaskElement & task : tasks) {
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, out_elems * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramOut16, sramOut32, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpyAsync((void *) devB, (const void *) sramOut32, sizeOut, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        rtMemcpyAsync((void *) devB, (const void *) sramOut16, sizeOut, rtMemcpySramToDevice, ctx.kernelStream);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
