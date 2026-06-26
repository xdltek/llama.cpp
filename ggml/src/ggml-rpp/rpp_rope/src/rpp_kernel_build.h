#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_rope/src/rpp_kernel_block.h"
#include "rpp_rope/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// -----------------------------
// Build graph once
// -----------------------------
void rpp_rope_build(rpp_kernel_context & ctx,
                    int                  T,
                    int                  H,
                    int                  D,
                    int                  Tstride,
                    int                  Hstride,
                    int                  Dstride,
                    int                  mode,
                    int                  n_rot,
                    int                  in0_bytes_per_element,
                    int                  in1_bytes_per_element,
                    int                  in2_bytes_per_element,
                    int                  out_bytes_per_element,
                    int                  is_instantial = 1) {
    if (((D / 2 % 32) != 0) || (n_rot / 2 <= 32) || (mode != 2)) {
        throw std::runtime_error("ROPE Parameter not Supportted");
    }

    // View-fusion stride contract for input0:
    // 1) contiguous THD or
    // 2) only T-stride differs (view on T), H/D are contiguous.
    const int expect_Dstride = in0_bytes_per_element;
    const int expect_Hstride = D * expect_Dstride;
    const int expect_Tstride = H * expect_Hstride;

    if (Dstride != expect_Dstride || Hstride != expect_Hstride) {
        throw std::runtime_error("ROPE view only supports Dstride==elem_bytes and Hstride==D*Dstride");
    }
    if (Tstride < expect_Tstride) {
        throw std::runtime_error("ROPE invalid Tstride: must be >= H*Hstride");
    }
    const bool has_view_on_t = (Tstride != expect_Tstride);

    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA    = ctx.dev_in[0];
    RPPdeviceptr          devTbl0 = ctx.dev_in[1];
    RPPdeviceptr          devTbl1 = ctx.dev_in[2];
    RPPdeviceptr          devB    = ctx.dev_out[0];
    int                   num_of_tiles;

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/rope.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int sizeA    = T * H * D * in0_bytes_per_element;
    const int sizeTbl0 = T * D * in1_bytes_per_element;
    const int sizeTbl1 = T * D * in2_bytes_per_element;
    const int sizeB    = T * H * D * out_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA     = sram_base;
    RPPdeviceptr sramA0    = sramA + round_up(sizeA);
    RPPdeviceptr sramTbl0  = sramA0 + round_up(sizeA);
    RPPdeviceptr sramTbl1  = sramTbl0 + round_up(sizeTbl0);
    RPPdeviceptr sramB     = sramTbl1 + round_up(sizeTbl1);

    const int total_sram_bytes = (int) (sramB + round_up(sizeB) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }
    if (!has_view_on_t) {
        rtMemcpyAsync((void *) sramA, (const void *) devA, sizeA, rtMemcpyDeviceToSram, ctx.kernelStream);
    } else {
        // Source uses strided T dimension; compact each [H*D] plane into SRAM.
        const int dense_t_bytes = H * D * in0_bytes_per_element;
        rtMemcpy2DAsync((void *) sramA, dense_t_bytes, (const void *) devA, Tstride, dense_t_bytes, T,
                        rtMemcpyDeviceToSram, ctx.kernelStream);
    }
    rtMemcpyAsync((void *) sramTbl0, (const void *) devTbl0, sizeTbl0, rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramTbl1, (const void *) devTbl1, sizeTbl1, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in0_bytes_per_element == sizeof(float)) {
        calc_tbdim_flattern(1, T * H * D, threadsPerBlock, blocksPerGrid);
        params.clear();
        cvt_kernel_param_init(threadsPerBlock, sramA, sramA0, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        sramA = sramA0;
    }
    if (in1_bytes_per_element == sizeof(float)) {
        calc_tbdim_flattern(1, T * D, threadsPerBlock, blocksPerGrid);
        params.clear();
        cvt_kernel_param_init(threadsPerBlock, sramTbl0, sramTbl0, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    if (in2_bytes_per_element == sizeof(float)) {
        calc_tbdim_flattern(1, T * D, threadsPerBlock, blocksPerGrid);
        params.clear();
        cvt_kernel_param_init(threadsPerBlock, sramTbl1, sramTbl1, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    RppTaskElement task;
    RppDims        in_out_dims;
    in_out_dims.nbDims = 3;
    in_out_dims.d[0]   = T;
    in_out_dims.d[1]   = H;
    in_out_dims.d[2]   = D;
    int bx             = 2;
    task.params.kernelList.clear();
    if (mode == 2 && n_rot == in_out_dims.d[2]) {
        task.taskName   = "llama3_loop1_pat0_fuse";
        task.blockDim.x = in_out_dims.d[2] / bx;
        if (bx * task.blockDim.x != in_out_dims.d[2]) {
            throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
        }
        if (in_out_dims.d[0] <= 64) {
            task.blockDim.y = in_out_dims.d[0];
        } else {
            task.blockDim.y = 64;
        }
        task.blockDim.z = 1;
        task.gridDim.x  = 1;
        task.gridDim.y  = in_out_dims.d[0] / task.blockDim.y;
        if (task.gridDim.y * task.blockDim.y != in_out_dims.d[0]) {
            throw std::runtime_error("ROPE Thread Block Y Dim Not Equal");
        }
        task.gridDim.z = in_out_dims.d[1];

        //in0 [by * 64][bz][bx * 64]
        //in1 [by * 64][bx * 64]
        //in2 [by * 64][bx * 64]
        //out [by * 64][bz][bx * 64]
        uint32_t in0StrideY = task.gridDim.z * bx * task.blockDim.x;
        uint32_t in1StrideY = bx * task.blockDim.x;
        uint32_t outStrideY = task.gridDim.z * bx * task.blockDim.x;

        uint32_t in0BlockYStride = task.blockDim.y * in0StrideY * sizeof(short);
        uint32_t in0BlockZStride = bx * task.blockDim.x * sizeof(short);

        uint32_t in1BlockYStride = task.blockDim.y * in1StrideY * sizeof(short);
        uint32_t outBlockYStride = task.blockDim.y * outStrideY * sizeof(short);

        uint32_t outBlockZStride = bx * task.blockDim.x * sizeof(short);

        if (in0StrideY > 0xffff) {
            throw std::runtime_error("ROPE in0StrideY Exceed");
        }
        task.params.kernelList.emplace_back(sramA);
        task.params.kernelList.emplace_back(sramTbl0);
        task.params.kernelList.emplace_back(sramTbl1);
        task.params.kernelList.emplace_back(sramA);
        task.params.kernelList.emplace_back(in0StrideY);
        task.params.kernelList.emplace_back(in1StrideY);
        task.params.kernelList.emplace_back(outStrideY);
        task.params.kernelList.emplace_back(in0BlockYStride);
        task.params.kernelList.emplace_back(in0BlockZStride);
        task.params.kernelList.emplace_back(in1BlockYStride);
        task.params.kernelList.emplace_back(outBlockYStride);
        task.params.kernelList.emplace_back(outBlockZStride);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    } else if (mode == 2 && n_rot < in_out_dims.d[2]) {
        task.blockDim.x = n_rot / bx;
        if ((task.blockDim.x % 32) == 0) {
            task.taskName = "rope_mode2_align_fuse";
        } else {
            task.taskName = "rope_mode2_gen_fuse";
        }

        if (bx * task.blockDim.x != n_rot) {
            throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
        }
        if (in_out_dims.d[0] <= 64) {
            task.blockDim.y = in_out_dims.d[0];
        } else {
            task.blockDim.y = 64;
        }
        task.blockDim.z = 1;
        task.gridDim.x  = 1;
        task.gridDim.y  = in_out_dims.d[0] / task.blockDim.y;
        if (task.gridDim.y * task.blockDim.y != in_out_dims.d[0]) {
            throw std::runtime_error("ROPE Thread Block Y Dim Not Equal");
        }
        task.gridDim.z = in_out_dims.d[1];

        //[T][H][96 + 32]
        //out [y][z][x]
        uint32_t in0StrideY = task.gridDim.z * in_out_dims.d[2];
        uint32_t in1StrideY = in_out_dims.d[2];
        uint32_t outStrideY = task.gridDim.z * in_out_dims.d[2];

        uint32_t in0BlockYStride = task.blockDim.y * task.gridDim.z * in_out_dims.d[2] * sizeof(short);
        uint32_t in0BlockZStride = in_out_dims.d[2] * sizeof(short);

        uint32_t in1BlockYStride = task.blockDim.y * in_out_dims.d[2] * sizeof(short);

        uint32_t outBlockYStride = task.blockDim.y * task.gridDim.z * in_out_dims.d[2] * sizeof(short);
        uint32_t outBlockZStride = in_out_dims.d[2] * sizeof(short);

        if (in0StrideY > 0xffff) {
            throw std::runtime_error("ROPE in0StrideY Exceed");
        }
        task.params.kernelList.clear();
        task.params.kernelList.emplace_back(sramA);
        task.params.kernelList.emplace_back(sramTbl0);
        task.params.kernelList.emplace_back(sramTbl1);
        task.params.kernelList.emplace_back(sramA);
        task.params.kernelList.emplace_back(in0StrideY);
        task.params.kernelList.emplace_back(in1StrideY);
        task.params.kernelList.emplace_back(outStrideY);
        task.params.kernelList.emplace_back(in0BlockYStride);
        task.params.kernelList.emplace_back(in0BlockZStride);
        task.params.kernelList.emplace_back(in1BlockYStride);
        task.params.kernelList.emplace_back(outBlockYStride);
        task.params.kernelList.emplace_back(outBlockZStride);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, T * H * D * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramA, sramB, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpyAsync((void *) devB, (const void *) sramB, sizeB, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        rtMemcpyAsync((void *) devB, (const void *) sramA, sizeB, rtMemcpySramToDevice, ctx.kernelStream);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

// Backward-compatible entry: contiguous layout (no view) by default.
void rpp_rope_build(rpp_kernel_context & ctx,
                    int                  T,
                    int                  H,
                    int                  D,
                    int                  mode,
                    int                  n_rot,
                    int                  in0_bytes_per_element,
                    int                  in1_bytes_per_element,
                    int                  in2_bytes_per_element,
                    int                  out_bytes_per_element,
                    int                  is_instantial = 1) {
    const int Dstride = in0_bytes_per_element;
    const int Hstride = D * Dstride;
    const int Tstride = H * Hstride;
    rpp_rope_build(ctx, T, H, D, Tstride, Hstride, Dstride, mode, n_rot, in0_bytes_per_element, in1_bytes_per_element,
                   in2_bytes_per_element, out_bytes_per_element, is_instantial);
}
