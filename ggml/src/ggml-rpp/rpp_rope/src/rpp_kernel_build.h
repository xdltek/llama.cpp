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
                    int                  out_Tstride,
                    int                  out_Hstride,
                    int                  out_Dstride,
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

    const int expect_Dstride = in0_bytes_per_element;
    const int expect_Hstride = D * expect_Dstride;
    const int expect_Tstride = H * expect_Hstride;

    if (Dstride != expect_Dstride) {
        throw std::runtime_error("ROPE view only supports Dstride==elem_bytes");
    }
    if (Hstride < expect_Hstride || Tstride < H * Hstride) {
        throw std::runtime_error("ROPE invalid input stride");
    }

    const int expect_out_Dstride = out_bytes_per_element;
    const int expect_out_Hstride = D * expect_out_Dstride;
    const int expect_out_Tstride = H * expect_out_Hstride;
    if (out_Dstride != expect_out_Dstride) {
        throw std::runtime_error("ROPE output view only supports Dstride==elem_bytes");
    }
    if (out_Hstride < expect_out_Hstride || out_Tstride < H * out_Hstride) {
        throw std::runtime_error("ROPE invalid output stride");
    }
    const bool input_contiguous  = Tstride == expect_Tstride && Hstride == expect_Hstride;
    const bool output_contiguous = out_Tstride == expect_out_Tstride && out_Hstride == expect_out_Hstride;

    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA    = ctx.dev_in[0];
    RPPdeviceptr          devTbl0 = ctx.dev_in[1];
    RPPdeviceptr          devTbl1 = ctx.dev_in[2];
    RPPdeviceptr          devB    = ctx.dev_out[0];
    auto max_block_y_for_x = [](uint32_t block_x) -> uint32_t {
        if (block_x >= 256) {
            return 16;
        }
        if (block_x >= 128) {
            return 32;
        }
        return 64;
    };

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rpp_module_load_once(ctx.rppBinMod, "rpp_kernel/rope.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int SRAM_LIMIT = 22 * 1024 * 1024;
    auto rope_sram_bytes = [&](int tile_T) -> int64_t {
        const int64_t sizeA    = (int64_t) tile_T * H * D * in0_bytes_per_element;
        const int64_t sizeTbl0 = (int64_t) tile_T * D * in1_bytes_per_element;
        const int64_t sizeTbl1 = (int64_t) tile_T * D * in2_bytes_per_element;
        const int64_t sizeB    = (int64_t) tile_T * H * D * out_bytes_per_element;

        int64_t total = round_up((int) sizeA);
        if (in0_bytes_per_element == sizeof(float)) {
            total += round_up((int) sizeA);
        }
        total += round_up((int) sizeTbl0);
        total += round_up((int) sizeTbl1);
        if (out_bytes_per_element == sizeof(float)) {
            total += round_up((int) sizeB);
        }
        return total;
    };

    int tile_T_max = 0;
    int lo         = 1;
    int hi         = T;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (rope_sram_bytes(mid) <= SRAM_LIMIT) {
            tile_T_max = mid;
            lo         = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (tile_T_max <= 0) {
        std::cerr << "SRAM overflow: even one ROPE T tile does not fit in " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    const int max_sizeA    = tile_T_max * H * D * in0_bytes_per_element;
    const int max_sizeTbl0 = tile_T_max * D * in1_bytes_per_element;
    const int max_sizeTbl1 = tile_T_max * D * in2_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA0    = sram_base;
    RPPdeviceptr sramA1    = sramA0 + round_up(max_sizeA);
    RPPdeviceptr sramTbl0  = sramA1 + (in0_bytes_per_element == sizeof(float) ? round_up(max_sizeA) : 0);
    RPPdeviceptr sramTbl1  = sramTbl0 + round_up(max_sizeTbl0);
    RPPdeviceptr sramB     = sramTbl1 + round_up(max_sizeTbl1);

    auto process_tile = [&](int t_offset, int tile_T) {
        const int sizeA    = tile_T * H * D * in0_bytes_per_element;
        const int sizeTbl0 = tile_T * D * in1_bytes_per_element;
        const int sizeTbl1 = tile_T * D * in2_bytes_per_element;
        const int sizeB    = tile_T * H * D * out_bytes_per_element;

        if (input_contiguous) {
            rtMemcpyAsync((void *) sramA0, (const void *) (devA + (RPPdeviceptr) t_offset * expect_Tstride), sizeA,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
        } else if (Hstride == expect_Hstride) {
            const int dense_t_bytes = H * D * in0_bytes_per_element;
            rtMemcpy2DAsync((void *) sramA0, dense_t_bytes, (const void *) (devA + (RPPdeviceptr) t_offset * Tstride),
                            Tstride, dense_t_bytes, tile_T, rtMemcpyDeviceToSram, ctx.kernelStream);
        } else {
            const int dense_h_bytes = D * in0_bytes_per_element;
            const int dense_t_bytes = H * dense_h_bytes;
            for (int t = 0; t < tile_T; ++t) {
                rtMemcpy2DAsync((void *) (sramA0 + (RPPdeviceptr) t * dense_t_bytes), dense_h_bytes,
                                (const void *) (devA + (RPPdeviceptr) (t_offset + t) * Tstride), Hstride,
                                dense_h_bytes, H, rtMemcpyDeviceToSram, ctx.kernelStream);
            }
        }
        rtMemcpyAsync((void *) sramTbl0, (const void *) (devTbl0 + (RPPdeviceptr) t_offset * D * in1_bytes_per_element),
                      sizeTbl0, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpyAsync((void *) sramTbl1, (const void *) (devTbl1 + (RPPdeviceptr) t_offset * D * in2_bytes_per_element),
                      sizeTbl1, rtMemcpyDeviceToSram, ctx.kernelStream);

        RPPdeviceptr sramA = sramA0;
        if (in0_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * H * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA0, sramA1, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            sramA = sramA1;
        }
        if (in1_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramTbl0, sramTbl0, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }
        if (in2_bytes_per_element == sizeof(float)) {
            calc_tbdim_flattern(1, tile_T * D, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramTbl1, sramTbl1, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

    RppTaskElement task;
    RppDims        in_out_dims;
    in_out_dims.nbDims = 3;
    in_out_dims.d[0]   = tile_T;
    in_out_dims.d[1]   = H;
    in_out_dims.d[2]   = D;
    int bx             = 2;
    task.params.kernelList.clear();
    if (mode == 2 && n_rot == in_out_dims.d[2]) {
        const uint32_t half_D = in_out_dims.d[2] / bx;
        uint32_t       block_x;
        const bool     xsplit = half_D > MAX_EXEC;
        if (half_D <= MAX_EXEC) {
            task.taskName = "llama3_loop1_pat0_fuse";
            block_x       = half_D;
        } else if ((half_D % 128) == 0) {
            task.taskName = "llama3_loop1_pat0_fuse_xsplit";
            block_x       = 128;
        } else if ((half_D % 64) == 0) {
            task.taskName = "llama3_loop1_pat0_fuse_xsplit";
            block_x       = 64;
        } else {
            task.taskName = "llama3_loop1_pat0_fuse_xsplit";
            block_x       = 32;
        }
        task.blockDim.x = block_x;
        if (half_D % task.blockDim.x != 0) {
            throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
        }
        const uint32_t max_block_y = max_block_y_for_x(task.blockDim.x);
        if ((uint32_t) in_out_dims.d[0] <= max_block_y) {
            task.blockDim.y = in_out_dims.d[0];
        } else {
            task.blockDim.y = max_block_y;
        }
        task.blockDim.z = 1;
        task.gridDim.x  = 1;
        task.gridDim.z = in_out_dims.d[1];

        //in0 [by * 64][bz][bx * 64]
        //in1 [by * 64][bx * 64]
        //in2 [by * 64][bx * 64]
        //out [by * 64][bz][bx * 64]
        uint32_t in0StrideY = task.gridDim.z * in_out_dims.d[2];
        uint32_t in1StrideY = in_out_dims.d[2];
        uint32_t outStrideY = task.gridDim.z * in_out_dims.d[2];

        uint32_t in0BlockYStride = task.blockDim.y * in0StrideY * sizeof(short);
        uint32_t in0BlockZStride = in_out_dims.d[2] * sizeof(short);

        uint32_t in1BlockYStride = task.blockDim.y * in1StrideY * sizeof(short);
        uint32_t outBlockYStride = task.blockDim.y * outStrideY * sizeof(short);

        uint32_t outBlockZStride = in_out_dims.d[2] * sizeof(short);

        if (in0StrideY > 0xffff) {
            throw std::runtime_error("ROPE in0StrideY Exceed");
        }
        const uint32_t nr_x_tiles = half_D / task.blockDim.x;
        const uint32_t full_block_y = task.blockDim.y;
        const uint32_t full_grid_y  = (uint32_t) in_out_dims.d[0] / full_block_y;
        const uint32_t tail_block_y = (uint32_t) in_out_dims.d[0] % full_block_y;
        auto launch_y_range = [&](uint32_t rows_done, uint32_t block_y, uint32_t grid_y) {
            if (block_y == 0 || grid_y == 0) {
                return;
            }
            task.blockDim.y = block_y;
            task.gridDim.y  = grid_y;

            const uint32_t in0TailOffset = rows_done * in0StrideY * sizeof(short);
            const uint32_t in1TailOffset = rows_done * in1StrideY * sizeof(short);
            const uint32_t outTailOffset = rows_done * outStrideY * sizeof(short);
            in0BlockYStride              = task.blockDim.y * in0StrideY * sizeof(short);
            in1BlockYStride              = task.blockDim.y * in1StrideY * sizeof(short);
            outBlockYStride              = task.blockDim.y * outStrideY * sizeof(short);

            for (uint32_t ix = 0; ix < nr_x_tiles; ++ix) {
                const uint32_t x_offset = ix * task.blockDim.x * sizeof(short);
                task.params.kernelList.clear();
                task.params.kernelList.emplace_back(sramA + in0TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramTbl0 + in1TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramTbl1 + in1TailOffset + x_offset);
                task.params.kernelList.emplace_back(sramA + outTailOffset + x_offset);
                task.params.kernelList.emplace_back(in0StrideY);
                task.params.kernelList.emplace_back(in1StrideY);
                task.params.kernelList.emplace_back(outStrideY);
                task.params.kernelList.emplace_back(in0BlockYStride);
                task.params.kernelList.emplace_back(in0BlockZStride);
                task.params.kernelList.emplace_back(in1BlockYStride);
                task.params.kernelList.emplace_back(outBlockYStride);
                task.params.kernelList.emplace_back(outBlockZStride);
                if (xsplit) {
                    task.params.kernelList.emplace_back(half_D * sizeof(short));
                }
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        };

        launch_y_range(0, full_block_y, full_grid_y);
        if (tail_block_y != 0) {
            launch_y_range(full_grid_y * full_block_y, tail_block_y, 1);
        }
    } else if (mode == 2 && n_rot < in_out_dims.d[2]) {
        task.blockDim.x = n_rot / bx;
        if ((task.blockDim.x % 32) == 0) {
            task.taskName = "rope_mode2_align_fuse";
        } else {
            task.taskName = "rope_mode2_gen_fuse";
        }

        if ((uint32_t) bx * task.blockDim.x != (uint32_t) n_rot) {
            throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
        }
        const uint32_t max_block_y = max_block_y_for_x(task.blockDim.x);
        if ((uint32_t) in_out_dims.d[0] <= max_block_y) {
            task.blockDim.y = (uint32_t) in_out_dims.d[0];
        } else {
            task.blockDim.y = max_block_y;
        }
        task.blockDim.z = 1;
        task.gridDim.x  = 1;
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
        const uint32_t full_block_y = task.blockDim.y;
        const uint32_t full_grid_y  = (uint32_t) in_out_dims.d[0] / full_block_y;
        const uint32_t tail_block_y = (uint32_t) in_out_dims.d[0] % full_block_y;
        auto launch_y_range = [&](uint32_t rows_done, uint32_t block_y, uint32_t grid_y) {
            if (block_y == 0 || grid_y == 0) {
                return;
            }
            task.blockDim.y = block_y;
            task.gridDim.y  = grid_y;

            const uint32_t in0TailOffset = rows_done * in0StrideY * sizeof(short);
            const uint32_t in1TailOffset = rows_done * in1StrideY * sizeof(short);
            const uint32_t outTailOffset = rows_done * outStrideY * sizeof(short);
            in0BlockYStride              = task.blockDim.y * in0StrideY * sizeof(short);
            in1BlockYStride              = task.blockDim.y * in1StrideY * sizeof(short);
            outBlockYStride              = task.blockDim.y * outStrideY * sizeof(short);

            task.params.kernelList.clear();
            task.params.kernelList.emplace_back(sramA + in0TailOffset);
            task.params.kernelList.emplace_back(sramTbl0 + in1TailOffset);
            task.params.kernelList.emplace_back(sramTbl1 + in1TailOffset);
            task.params.kernelList.emplace_back(sramA + outTailOffset);
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
        };

        launch_y_range(0, full_block_y, full_grid_y);
        if (tail_block_y != 0) {
            launch_y_range(full_grid_y * full_block_y, tail_block_y, 1);
        }
    }

    RPPdeviceptr sramOut = sramA;
    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, tile_T * H * D * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramA, sramB, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        sramOut = sramB;
    }

    if (output_contiguous) {
        rtMemcpyAsync((void *) (devB + (RPPdeviceptr) t_offset * expect_out_Tstride), (const void *) sramOut, sizeB,
                      rtMemcpySramToDevice, ctx.kernelStream);
    } else if (out_Hstride == expect_out_Hstride) {
        const int dense_t_bytes = H * D * out_bytes_per_element;
        rtMemcpy2DAsync((void *) (devB + (RPPdeviceptr) t_offset * out_Tstride), out_Tstride,
                        (const void *) sramOut, dense_t_bytes, dense_t_bytes, tile_T, rtMemcpySramToDevice,
                        ctx.kernelStream);
    } else {
        const int dense_h_bytes = D * out_bytes_per_element;
        const int dense_t_bytes = H * dense_h_bytes;
        for (int t = 0; t < tile_T; ++t) {
            rtMemcpy2DAsync((void *) (devB + (RPPdeviceptr) (t_offset + t) * out_Tstride), out_Hstride,
                            (const void *) (sramOut + (RPPdeviceptr) t * dense_t_bytes), dense_h_bytes,
                            dense_h_bytes, H, rtMemcpySramToDevice, ctx.kernelStream);
        }
    }
    };

    for (int t_offset = 0; t_offset < T; t_offset += tile_T_max) {
        const int tile_T = (T - t_offset) < tile_T_max ? (T - t_offset) : tile_T_max;
        process_tile(t_offset, tile_T);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    const std::string graph_key = rpp_join_function_name_and_args(
        __func__, T, H, D, Tstride, Hstride, Dstride, out_Tstride, out_Hstride, out_Dstride, mode, n_rot,
        in0_bytes_per_element, in1_bytes_per_element, in2_bytes_per_element, out_bytes_per_element);
    if (rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial) != RPP_SUCCESS) {
        throw std::runtime_error("rpp_graph_instantiate failed.");
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
    const int out_Dstride = out_bytes_per_element;
    const int out_Hstride = D * out_Dstride;
    const int out_Tstride = H * out_Hstride;
    rpp_rope_build(ctx, T, H, D, Tstride, Hstride, Dstride, out_Tstride, out_Hstride, out_Dstride, mode, n_rot,
                   in0_bytes_per_element, in1_bytes_per_element, in2_bytes_per_element, out_bytes_per_element,
                   is_instantial);
}
