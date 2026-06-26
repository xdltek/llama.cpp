#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_norm/src/rpp_kernel_block.h"
#include "rpp_norm/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// -----------------------------
// Build graph once
// -----------------------------
void rpp_norm_build(rpp_kernel_context & ctx,
                    int                  rows,
                    int                  cols,
                    float                eps,
                    int                  in_bytes_per_element,
                    int                  out_bytes_per_element,
                    int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA = ctx.dev_in[0];
    RPPdeviceptr          devB = ctx.dev_out[0];
    int                   num_of_tiles;

    // build exp table
    int        lut_elements = 64 * 1024;
    uint16_t * rsqrt_table  = (uint16_t *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < lut_elements; i++) {
        uint32_t x = i;
        x <<= 16;
        float y        = 1.0f / std::sqrt((*(float *) &x));
        rsqrt_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }
    rsqrt_table[0]      = 0;
    rsqrt_table[0x8000] = 0;

    RPPdeviceptr dev_rsqrt_lut = ctx.dev_workspace;
    rtMemcpy((void *) dev_rsqrt_lut, (const void *) rsqrt_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/norm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int sizeA  = rows * cols * in_bytes_per_element;
    const int sizeB  = rows * cols * out_bytes_per_element;
    const int sizeIO = rows * cols * sizeof(float);
    const int size16 = rows * cols * sizeof(short);

    const int    sizeMean         = rows * sizeof(rpp::bfloat16);
    const int    sizeVar          = rows * sizeof(rpp::bfloat16);
    const int    lutSize          = 64 * 1024 * sizeof(short);
    RPPdeviceptr sram_base        = ctx.virtual_sram_base;
    RPPdeviceptr sramIO           = sram_base;
    RPPdeviceptr sramMean         = sramIO + round_up(sizeIO);
    RPPdeviceptr sramVar          = sramMean + round_up(sizeMean);
    RPPdeviceptr rsqrt_table_addr = sramVar + round_up(sizeVar);
    RPPdeviceptr workspace0_addr  = rsqrt_table_addr + round_up(lutSize);
    RPPdeviceptr workspace1_addr  = workspace0_addr + round_up(size16);
    RPPdeviceptr workspace2_addr  = workspace1_addr + round_up(size16);
    RPPdeviceptr workspace3_addr  = workspace2_addr + round_up(size16);

    const int total_sram_bytes = (int) (workspace3_addr + round_up(size16) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }
    rtMemcpyAsync((void *) sramIO, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) rsqrt_table_addr, (const void *) dev_rsqrt_lut, lutSize, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

    if (in_bytes_per_element == sizeof(float)) {
        calc_tbdim_flattern(1, rows * cols, threadsPerBlock, blocksPerGrid);
        params.clear();
        cvt_kernel_param_init(threadsPerBlock, sramIO, workspace2_addr, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    RppDims in_dims;
    in_dims.nbDims = 2;
    in_dims[0]     = rows;
    in_dims[1]     = cols;

    if (in_dims[1] >= 8192) {
        throw std::runtime_error("RMS current support columns greater than 8K");
    }

    {
        // for (int c = 0; c < cols; ++c) mean += (double)xr[c];
        // mean /= (double)cols;
        RPPdeviceptr reduce_in_addr = sramIO;
        if (in_bytes_per_element == sizeof(float)) {
            reduce_in_addr = workspace2_addr;
        }
        RPPdeviceptr reduce_out_addr = sramMean;
        RppDims      reduce_mean_in_dims{}, reduce_mean_out_dims{};
        reduce_mean_in_dims.nbDims  = 2;
        reduce_mean_in_dims.d[0]    = in_dims[0];
        reduce_mean_in_dims.d[1]    = in_dims[1];
        reduce_mean_out_dims.nbDims = 2;
        reduce_mean_out_dims.d[0]   = in_dims[0];
        reduce_mean_out_dims.d[1]   = 1;
        std::vector<RppDims> sub_reduce_io_dims;
        RppDims              middle_dims{};
        sub_reduce_io_dims.emplace_back(reduce_mean_in_dims);
        sub_reduce_io_dims.emplace_back(reduce_mean_out_dims);
        int io_cursor = 0;
        while (ReduceSpawnIO((int) 1, sub_reduce_io_dims[io_cursor], sub_reduce_io_dims[sub_reduce_io_dims.size() - 1],
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
                1, RppReduceOperation::kAVG, input_addr, output_addr, in_dims, out_dims, 0, true, false);
            for (RppTaskElement & task : tasks) {
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }
    }

    // var = (xr[c] - mean) *  (xr[c] - mean);
    RppTaskElement sub_pow_task{};
    sub_pow_task.blockDim.x = in_dims[1];
    sub_pow_task.blockDim.y = 1;
    sub_pow_task.blockDim.z = 1;
    sub_pow_task.gridDim.x  = in_dims[0];
    sub_pow_task.gridDim.y  = 1;
    sub_pow_task.gridDim.z  = 1;
    if (in_bytes_per_element == sizeof(float)) {
        sub_pow_task.params.kernelList.emplace_back(workspace2_addr);
    } else {
        sub_pow_task.params.kernelList.emplace_back(sramIO);
    }
    sub_pow_task.params.kernelList.emplace_back(sramMean);
    sub_pow_task.params.kernelList.emplace_back(workspace3_addr);
    sub_pow_task.params.kernelList.emplace_back(in_dims[1] * sizeof(uint16_t));
    sub_pow_task.params.kernelList.emplace_back(sizeof(uint16_t));
    sub_pow_task.taskName = "matrix_norm_sub_pow";
    launchWrapperAysnc(sub_pow_task.taskName, sub_pow_task.gridDim, sub_pow_task.blockDim,
                       sub_pow_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

    {
        // for (int c = 0; c < cols; ++c) var_mean += var;
        // var_mean /= (double)cols;
        RPPdeviceptr reduce_in_addr  = workspace3_addr;
        RPPdeviceptr reduce_out_addr = sramVar;
        RppDims      reduce_mean_in_dims{}, reduce_mean_out_dims{};
        reduce_mean_in_dims.nbDims  = 2;
        reduce_mean_in_dims.d[0]    = in_dims[0];
        reduce_mean_in_dims.d[1]    = in_dims[1];
        reduce_mean_out_dims.nbDims = 2;
        reduce_mean_out_dims.d[0]   = in_dims[0];
        reduce_mean_out_dims.d[1]   = 1;
        std::vector<RppDims> sub_reduce_io_dims;
        RppDims              middle_dims{};
        sub_reduce_io_dims.emplace_back(reduce_mean_in_dims);
        sub_reduce_io_dims.emplace_back(reduce_mean_out_dims);
        int io_cursor = 0;
        while (ReduceSpawnIO((int) 1, sub_reduce_io_dims[io_cursor], sub_reduce_io_dims[sub_reduce_io_dims.size() - 1],
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
                1, RppReduceOperation::kAVG, input_addr, output_addr, in_dims, out_dims, 0, true, false);
            for (RppTaskElement & task : tasks) {
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }
    }
    //inv_std = 1.0f / std::sqrt((float)var + eps);
    RppTaskElement rsqrt_task{};
    rsqrt_task.blockDim.x = in_dims.d[0] <= 32 ? 33 : in_dims.d[0];
    rsqrt_task.blockDim.y = 1;
    rsqrt_task.blockDim.z = 1;
    rsqrt_task.gridDim.x  = 1;
    rsqrt_task.gridDim.y  = 1;
    rsqrt_task.gridDim.z  = 1;
    rsqrt_task.params.kernelList.emplace_back(sramVar);
    rsqrt_task.params.kernelList.emplace_back(rsqrt_table_addr);
    rsqrt_task.params.kernelList.emplace_back(sramVar);
    rsqrt_task.params.kernelList.emplace_back(*(uint32_t *) &eps);
    rsqrt_task.taskName = "llm_add_rsqrt_lut";
    launchWrapperAysnc(rsqrt_task.taskName, rsqrt_task.gridDim, rsqrt_task.blockDim, rsqrt_task.params.kernelList,
                       ctx.rppBinMod, ctx.kernelStream);

    RppTaskElement sub_mul_task{};
    sub_mul_task.blockDim.x = in_dims[1];
    sub_mul_task.blockDim.y = 1;
    sub_mul_task.blockDim.z = 1;
    sub_mul_task.gridDim.x  = in_dims[0];
    sub_mul_task.gridDim.y  = 1;
    sub_mul_task.gridDim.z  = 1;
    if (in_bytes_per_element == sizeof(float)) {
        sub_mul_task.params.kernelList.emplace_back(workspace2_addr);
    } else {
        sub_mul_task.params.kernelList.emplace_back(sramIO);
    }
    sub_mul_task.params.kernelList.emplace_back(sramMean);
    sub_mul_task.params.kernelList.emplace_back(sramVar);
    sub_mul_task.params.kernelList.emplace_back(workspace2_addr);
    sub_mul_task.params.kernelList.emplace_back(in_dims[1] * sizeof(uint16_t));
    sub_mul_task.params.kernelList.emplace_back(sizeof(uint16_t));
    sub_mul_task.taskName = "matrix_norm_sub_mul";
    launchWrapperAysnc(sub_mul_task.taskName, sub_mul_task.gridDim, sub_mul_task.blockDim,
                       sub_mul_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, rows * cols * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, workspace2_addr, sramIO, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpyAsync((void *) devB, (const void *) sramIO, sizeB, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        rtMemcpyAsync((void *) devB, (const void *) workspace2_addr, sizeB, rtMemcpySramToDevice, ctx.kernelStream);
    }

    free(rsqrt_table);
    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
