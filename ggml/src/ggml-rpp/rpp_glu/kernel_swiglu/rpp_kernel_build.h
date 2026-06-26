#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_glu/kernel_swiglu/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_swiglu {

static inline void silu_check_sram_limit(RPPdeviceptr sram_base, RPPdeviceptr sram_end) {
    const int total_sram_bytes = (int) (sram_end - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }
}

static inline void silu_require_sram_direct_io(const rpp_kernel_context & ctx) {
    if (ctx.dev_in.size() < 5 || ctx.dev_out.size() < 2) {
        throw std::runtime_error("Silu SRAM direct mode requires pre-bound SRAM ctx.dev_in/dev_out");
    }
}

static inline bool silu_mode2_tile_fits(int rows, int W, int in_bytes_per_element, int out_bytes_per_element) {
    const uint64_t lutSize = 64ull * 1024ull * sizeof(uint16_t);
    const uint64_t sizeA   = (uint64_t) rows * (uint64_t) W * (uint64_t) in_bytes_per_element;
    const uint64_t sizeA0  = (uint64_t) rows * (uint64_t) W * (uint64_t) sizeof(rpp::bfloat16);
    const uint64_t sizeB   = (uint64_t) rows * (uint64_t) (W / 2) * (uint64_t) out_bytes_per_element;
    const uint64_t total =
        (uint64_t) silu_round_up_512((uint32_t) lutSize) + (uint64_t) silu_round_up_512((uint32_t) sizeA) +
        (uint64_t) silu_round_up_512((uint32_t) sizeA0) + (uint64_t) silu_round_up_512((uint32_t) sizeB);
    const uint64_t SRAM_LIMIT = 22ull * 1024ull * 1024ull;
    return total <= SRAM_LIMIT;
}

static inline int silu_mode2_rows_per_tile(int rows_total, int W, int in_bytes_per_element, int out_bytes_per_element) {
    int lo   = 1;
    int hi   = rows_total;
    int best = 0;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (silu_mode2_tile_fits(mid, W, in_bytes_per_element, out_bytes_per_element)) {
            best = mid;
            lo   = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return best;
}

static void rpp_silu_mode2_build(rpp_kernel_context & ctx,
                                 int                  C,
                                 int                  H,
                                 int                  W,
                                 int                  split_axis,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  is_instantial,
                                 int                  is_capture) {
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA       = ctx.dev_in[0];
    RPPdeviceptr          devB       = ctx.dev_out[0];
    const int             rows_total = C * H;

    if ((W % 2) != 0) {
        throw std::runtime_error("Silu Mode2 requires even W");
    }
    if (split_axis != 2) {
        throw std::runtime_error("Silu Only Support axis 2 Split");
    }

    const RPPdeviceptr dev_gelu_lut = silu_prepare_lut_workspace(ctx);

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/silu.o");

    const int    lutSize         = 64 * 1024 * (int) sizeof(uint16_t);
    RPPdeviceptr sram_base       = ctx.virtual_sram_base;
    RPPdeviceptr gelu_table_addr = sram_base;
    rtMemcpyAsync((void *) gelu_table_addr, (const void *) dev_gelu_lut, lutSize, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

    const int rows_per_tile = silu_mode2_rows_per_tile(rows_total, W, in_bytes_per_element, out_bytes_per_element);
    if (rows_per_tile <= 0) {
        throw std::runtime_error("Silu Mode2 even one row does not fit in SRAM");
    }

    const int full_tiles = rows_total / rows_per_tile;
    const int tail_rows  = rows_total % rows_per_tile;
    const int num_tiles  = full_tiles + (tail_rows > 0 ? 1 : 0);
    const int bx         = 2;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int tile_rows = (tile_idx < full_tiles) ? rows_per_tile : tail_rows;
        if (tile_rows <= 0) {
            break;
        }

        const int tile_elements   = tile_rows * W;
        const int output_elements = tile_elements / 2;
        const int sizeA           = tile_elements * in_bytes_per_element;
        const int sizeA0          = tile_elements * (int) sizeof(rpp::bfloat16);
        const int sizeB           = output_elements * out_bytes_per_element;
        const int input_offset    = tile_idx * rows_per_tile * W * in_bytes_per_element;
        const int output_offset   = tile_idx * rows_per_tile * (W / 2) * out_bytes_per_element;

        RPPdeviceptr sramA  = gelu_table_addr + silu_round_up_512(lutSize);
        RPPdeviceptr sramA0 = sramA + silu_round_up_512(sizeA);
        RPPdeviceptr sramB  = sramA0 + silu_round_up_512(sizeA0);
        silu_check_sram_limit(sram_base, sramB + silu_round_up_512(sizeB));

        rtMemcpyAsync((void *) sramA, (const void *) (devA + (RPPdeviceptr) input_offset), sizeA, rtMemcpyDeviceToSram,
                      ctx.kernelStream);

        RPPdeviceptr sramA_exec = sramA;
        if (in_bytes_per_element == (int) sizeof(float)) {
            calc_tbdim_flattern(1, output_elements * 2, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA, sramA0, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            sramA_exec = sramA0;
        }

        RppTaskElement task;
        RppDims        in_out_dims;
        in_out_dims.nbDims = 3;
        in_out_dims.d[0]   = 1;
        in_out_dims.d[1]   = tile_rows;
        in_out_dims.d[2]   = W;
        task.blockDim.y    = 1;
        task.blockDim.z    = 1;
        task.gridDim.z     = 1;
        task.gridDim.x     = 1;
        task.gridDim.y     = tile_rows;
        task.params.kernelList.clear();
        task.taskName   = "llm_split_gelu";
        task.blockDim.x = in_out_dims.d[2] / bx;
        while (task.blockDim.x >= 8192) {
            task.blockDim.x = task.blockDim.x >> 1;
            task.gridDim.x  = task.gridDim.x * 2;
        }
        if (bx * task.blockDim.x * task.gridDim.x != in_out_dims.d[2]) {
            throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
        }
        while (task.blockDim.y * task.blockDim.x < 4096) {
            if (task.gridDim.y % 2 == 0) {
                task.gridDim.y  = task.gridDim.y / 2;
                task.blockDim.y = task.blockDim.y * 2;
            } else {
                break;
            }
        }
        if (task.blockDim.y * task.blockDim.x <= 32 || task.blockDim.y * task.blockDim.x >= 8192) {
            throw std::runtime_error("ROPE Thread Block Size Invalid");
        }

        uint32_t in0StrideY      = W;
        uint32_t outStrideY      = W / bx;
        uint32_t in0BlockYStride = task.blockDim.y * in0StrideY * (uint32_t) sizeof(uint16_t);
        uint32_t outBlockYStride = task.blockDim.y * outStrideY * (uint32_t) sizeof(uint16_t);
        uint32_t in0BlockXStride = task.blockDim.x * (uint32_t) sizeof(uint16_t);
        uint32_t outBlockXStride = task.blockDim.x * (uint32_t) sizeof(uint16_t);

        if (in0StrideY > 0xffff) {
            throw std::runtime_error("ROPE in0StrideY Exceed");
        }
        task.params.kernelList.emplace_back(sramA_exec);
        task.params.kernelList.emplace_back(W / 2 * (int) sizeof(uint16_t));
        task.params.kernelList.emplace_back(gelu_table_addr);
        task.params.kernelList.emplace_back(sramA0);
        task.params.kernelList.emplace_back(in0StrideY);
        task.params.kernelList.emplace_back(outStrideY);
        task.params.kernelList.emplace_back(in0BlockXStride);
        task.params.kernelList.emplace_back(outBlockXStride);
        task.params.kernelList.emplace_back(in0BlockYStride);
        task.params.kernelList.emplace_back(outBlockYStride);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);

        if (out_bytes_per_element == (int) sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, output_elements * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramA0, sramB, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpyAsync((void *) (devB + (RPPdeviceptr) output_offset), (const void *) sramB, sizeB,
                          rtMemcpySramToDevice, ctx.kernelStream);
        } else {
            rtMemcpyAsync((void *) (devB + (RPPdeviceptr) output_offset), (const void *) sramA0, sizeB,
                          rtMemcpySramToDevice, ctx.kernelStream);
        }
    }

    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_silu_mode2_build_sram(rpp_kernel_context & ctx,
                                      int                  C,
                                      int                  H,
                                      int                  W,
                                      int                  split_axis,
                                      int                  in_bytes_per_element,
                                      int                  out_bytes_per_element,
                                      int                  is_instantial,
                                      int                  is_capture) {
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    const int             elements        = C * H * W;
    const int             output_elements = elements / 2;

    if (elements >= 4 * 1024 * 1024) {
        throw std::runtime_error("Silu SRAM Mode2 Exceed 24M");
    }
    if (split_axis != 2) {
        throw std::runtime_error("Silu SRAM Mode2 Only Support axis 2 Split");
    }

    silu_require_sram_direct_io(ctx);
    const RPPdeviceptr lut_ws = silu_prepare_lut_workspace(ctx);

    const RPPdeviceptr sramA_raw       = ctx.dev_in[0];
    const RPPdeviceptr gelu_table_addr = ctx.dev_in[2];
    const RPPdeviceptr sramA_exec      = (in_bytes_per_element == (int) sizeof(float)) ? ctx.dev_in[3] : sramA_raw;
    const RPPdeviceptr sramOutBf16 = (out_bytes_per_element == (int) sizeof(float)) ? ctx.dev_out[1] : ctx.dev_out[0];
    const RPPdeviceptr sramOut     = ctx.dev_out[0];

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/silu.o");
    const int lutSize = 64 * 1024 * (int) sizeof(uint16_t);
    rtMemcpyAsync((void *) gelu_table_addr, (const void *) lut_ws, lutSize, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        calc_tbdim_flattern(1, output_elements * 2, threadsPerBlock, blocksPerGrid);
        params.clear();
        cvt_kernel_param_init(threadsPerBlock, sramA_raw, sramA_exec, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    RppTaskElement task;
    RppDims        in_out_dims;
    in_out_dims.nbDims = 3;
    in_out_dims.d[0]   = C;
    in_out_dims.d[1]   = H;
    in_out_dims.d[2]   = W;
    task.blockDim.y    = 1;
    task.blockDim.z    = 1;
    task.gridDim.z     = 1;
    task.gridDim.x     = 1;
    task.gridDim.y     = C * H;
    int bx             = 2;
    task.params.kernelList.clear();
    task.taskName   = "llm_split_gelu";
    task.blockDim.x = in_out_dims.d[2] / bx;
    while (task.blockDim.x >= 8192) {
        task.blockDim.x = task.blockDim.x >> 1;
        task.gridDim.x  = task.gridDim.x * 2;
    }
    if (bx * task.blockDim.x * task.gridDim.x != in_out_dims.d[2]) {
        throw std::runtime_error("ROPE Thread Block X Dim Not Equal");
    }
    while (task.blockDim.y * task.blockDim.x < 4096) {
        if (task.gridDim.y % 2 == 0) {
            task.gridDim.y  = task.gridDim.y / 2;
            task.blockDim.y = task.blockDim.y * 2;
        } else {
            break;
        }
    }
    if (task.blockDim.y * task.blockDim.x <= 32 || task.blockDim.y * task.blockDim.x >= 8192) {
        throw std::runtime_error("ROPE Thread Block Size Invalid");
    }

    uint32_t in0StrideY      = W;
    uint32_t outStrideY      = W / bx;
    uint32_t in0BlockYStride = task.blockDim.y * in0StrideY * (uint32_t) sizeof(uint16_t);
    uint32_t outBlockYStride = task.blockDim.y * outStrideY * (uint32_t) sizeof(uint16_t);
    uint32_t in0BlockXStride = task.blockDim.x * (uint32_t) sizeof(uint16_t);
    uint32_t outBlockXStride = task.blockDim.x * (uint32_t) sizeof(uint16_t);

    if (in0StrideY > 0xffff) {
        throw std::runtime_error("ROPE in0StrideY Exceed");
    }
    task.params.kernelList.emplace_back(sramA_exec);
    task.params.kernelList.emplace_back(W / 2 * (int) sizeof(uint16_t));
    task.params.kernelList.emplace_back(gelu_table_addr);
    task.params.kernelList.emplace_back(sramOutBf16);
    task.params.kernelList.emplace_back(in0StrideY);
    task.params.kernelList.emplace_back(outStrideY);
    task.params.kernelList.emplace_back(in0BlockXStride);
    task.params.kernelList.emplace_back(outBlockXStride);
    task.params.kernelList.emplace_back(in0BlockYStride);
    task.params.kernelList.emplace_back(outBlockYStride);
    launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                       ctx.kernelStream);

    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, output_elements * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramOutBf16, sramOut, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_silu_mode01_build(rpp_kernel_context & ctx,
                                  int                  mode,
                                  int                  C,
                                  int                  H,
                                  int                  W,
                                  int                  in_bytes_per_element,
                                  int                  out_bytes_per_element,
                                  int                  is_instantial,
                                  int                  is_capture) {
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA         = ctx.dev_in[0];
    RPPdeviceptr          devA1        = (mode == 1) ? ctx.dev_in[1] : 0;
    RPPdeviceptr          devB         = ctx.dev_out[0];
    int                   num_of_tiles = 0;
    int                   norm_tiles   = 0;
    int                   tail_tiles   = 0;
    const int             elements     = C * H * W;

    if (mode == 0) {
        norm_tiles = 2 * 1024 * 1024;
        get_linear_blocks(elements, norm_tiles, num_of_tiles, tail_tiles);
    } else {
        norm_tiles = 1024 * 1024;
        get_linear_blocks(elements, norm_tiles, num_of_tiles, tail_tiles);
    }

    const RPPdeviceptr dev_gelu_lut = silu_prepare_lut_workspace(ctx);

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/silu.o");

    const int    lutSize         = 64 * 1024 * (int) sizeof(uint16_t);
    RPPdeviceptr sram_base       = ctx.virtual_sram_base;
    RPPdeviceptr gelu_table_addr = sram_base;
    rtMemcpyAsync((void *) gelu_table_addr, (const void *) dev_gelu_lut, lutSize, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

    for (int i = 0; i < num_of_tiles - 1; i++) {
        int          sizeA_cvt  = norm_tiles * (int) sizeof(uint16_t);
        int          sizeA      = norm_tiles * in_bytes_per_element;
        int          sizeB      = norm_tiles * out_bytes_per_element;
        RPPdeviceptr sramA_cvt  = gelu_table_addr + silu_round_up_512(lutSize);
        RPPdeviceptr sramA1_cvt = sramA_cvt + silu_round_up_512(sizeA_cvt);
        RPPdeviceptr sramA      = sramA1_cvt + silu_round_up_512(sizeA_cvt);
        RPPdeviceptr sramA1     = sramA + silu_round_up_512(sizeA);
        RPPdeviceptr sramB      = sramA1 + silu_round_up_512(sizeA);
        silu_check_sram_limit(sram_base, sramB + silu_round_up_512(sizeB));

        rtMemcpyAsync((void *) sramA, (const void *) (devA + (RPPdeviceptr) i * (RPPdeviceptr) sizeA), sizeA,
                      rtMemcpyDeviceToSram, ctx.kernelStream);
        if (mode == 1) {
            rtMemcpyAsync((void *) sramA1, (const void *) (devA1 + (RPPdeviceptr) i * (RPPdeviceptr) sizeA), sizeA,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
        }

        if (in_bytes_per_element == (int) sizeof(float)) {
            calc_tbdim_flattern(1, norm_tiles, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA, sramA_cvt, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            sramA = sramA_cvt;

            if (mode == 1) {
                calc_tbdim_flattern(1, norm_tiles, threadsPerBlock, blocksPerGrid);
                params.clear();
                cvt_kernel_param_init(threadsPerBlock, sramA1, sramA1_cvt, kFLOAT, kBF16, params);
                launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
                sramA1 = sramA1_cvt;
            }
        }

        int nr_of_blocks = 0;
        int Nt           = 0;
        int Ns           = 4096;
        get_linear_blocks(norm_tiles, Ns, nr_of_blocks, Nt);

        int bytes_per_block = Ns * (int) sizeof(rpp::bfloat16);
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
            params.push_back(sramA + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
            params.push_back(sramA1 + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
            params.push_back(gelu_table_addr);
            params.push_back(sramA + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
            params.push_back(0);
            params.push_back(1);
            params.push_back(mode);
            launchWrapperAysnc("llm_gelu", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
        }
        if (out_bytes_per_element == (int) sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, norm_tiles * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramA, sramB, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpyAsync((void *) (devB + (RPPdeviceptr) i * (RPPdeviceptr) sizeB), (const void *) sramB, sizeB,
                          rtMemcpySramToDevice, ctx.kernelStream);
        } else {
            rtMemcpyAsync((void *) (devB + (RPPdeviceptr) i * (RPPdeviceptr) sizeB), (const void *) sramA, sizeB,
                          rtMemcpySramToDevice, ctx.kernelStream);
        }
    }

    if (tail_tiles > 0) {
        int          sizeA_cvt  = tail_tiles * (int) sizeof(uint16_t);
        int          sizeA      = tail_tiles * in_bytes_per_element;
        int          sizeB      = tail_tiles * out_bytes_per_element;
        RPPdeviceptr sramA_cvt  = gelu_table_addr + silu_round_up_512(lutSize);
        RPPdeviceptr sramA1_cvt = sramA_cvt + silu_round_up_512(sizeA_cvt);
        RPPdeviceptr sramA      = sramA1_cvt + silu_round_up_512(sizeA_cvt);
        RPPdeviceptr sramA1     = sramA + silu_round_up_512(sizeA);
        RPPdeviceptr sramB      = sramA1 + silu_round_up_512(sizeA);
        silu_check_sram_limit(sram_base, sramB + silu_round_up_512(sizeB));

        int tile_tail_offset_in  = (num_of_tiles - 1) * norm_tiles * in_bytes_per_element;
        int tile_tail_offset_out = (num_of_tiles - 1) * norm_tiles * out_bytes_per_element;

        rtMemcpyAsync((void *) sramA, (const void *) (devA + (RPPdeviceptr) tile_tail_offset_in), sizeA,
                      rtMemcpyDeviceToSram, ctx.kernelStream);
        if (mode == 1) {
            rtMemcpyAsync((void *) sramA1, (const void *) (devA1 + (RPPdeviceptr) tile_tail_offset_in), sizeA,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
        }

        if (in_bytes_per_element == (int) sizeof(float)) {
            calc_tbdim_flattern(1, tail_tiles, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA, sramA_cvt, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            sramA = sramA_cvt;

            if (mode == 1) {
                calc_tbdim_flattern(1, tail_tiles, threadsPerBlock, blocksPerGrid);
                params.clear();
                cvt_kernel_param_init(threadsPerBlock, sramA1, sramA1_cvt, kFLOAT, kBF16, params);
                launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
                sramA1 = sramA1_cvt;
            }
        }

        int nr_of_blocks = 0;
        int Nt           = 0;
        int Ns           = 4096;
        get_linear_blocks(tail_tiles, Ns, nr_of_blocks, Nt);

        int bytes_per_block = Ns * (int) sizeof(rpp::bfloat16);
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
            params.push_back(sramA + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
            params.push_back(sramA1 + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
            params.push_back(gelu_table_addr);
            params.push_back(sramA + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
            params.push_back(0);
            params.push_back(1);
            params.push_back(mode);
            launchWrapperAysnc("llm_gelu", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
        }
        if (out_bytes_per_element == (int) sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, tail_tiles * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramA, sramB, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpyAsync((void *) (devB + (RPPdeviceptr) tile_tail_offset_out), (const void *) sramB, sizeB,
                          rtMemcpySramToDevice, ctx.kernelStream);
        } else {
            rtMemcpyAsync((void *) (devB + (RPPdeviceptr) tile_tail_offset_out), (const void *) sramA, sizeB,
                          rtMemcpySramToDevice, ctx.kernelStream);
        }
    }

    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_silu_mode01_build_sram(rpp_kernel_context & ctx,
                                       int                  mode,
                                       int                  C,
                                       int                  H,
                                       int                  W,
                                       int                  in_bytes_per_element,
                                       int                  out_bytes_per_element,
                                       int                  is_instantial,
                                       int                  is_capture) {
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    const int             elements = C * H * W;
    silu_require_sram_direct_io(ctx);
    const RPPdeviceptr lut_ws          = silu_prepare_lut_workspace(ctx);
    RPPdeviceptr       sramA_raw       = ctx.dev_in[0];
    RPPdeviceptr       sramA1_raw      = (mode == 1) ? ctx.dev_in[1] : 0;
    RPPdeviceptr       gelu_table_addr = ctx.dev_in[2];
    RPPdeviceptr       sramA           = (in_bytes_per_element == (int) sizeof(float)) ? ctx.dev_in[3] : sramA_raw;
    RPPdeviceptr       sramA1 =
        (mode == 1) ? ((in_bytes_per_element == (int) sizeof(float)) ? ctx.dev_in[4] : sramA1_raw) : 0;
    RPPdeviceptr sramOutBf16 = (out_bytes_per_element == (int) sizeof(float)) ? ctx.dev_out[1] : ctx.dev_out[0];
    RPPdeviceptr sramOut     = ctx.dev_out[0];

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/silu.o");
    const int lutSize = 64 * 1024 * (int) sizeof(uint16_t);
    rtMemcpyAsync((void *) gelu_table_addr, (const void *) lut_ws, lutSize, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        calc_tbdim_flattern(1, elements, threadsPerBlock, blocksPerGrid);
        params.clear();
        cvt_kernel_param_init(threadsPerBlock, sramA_raw, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        if (mode == 1) {
            calc_tbdim_flattern(1, elements, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramA1_raw, sramA1, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    int nr_of_blocks = 0;
    int Nt           = 0;
    int Ns           = 4096;
    get_linear_blocks(elements, Ns, nr_of_blocks, Nt);

    int bytes_per_block = Ns * (int) sizeof(rpp::bfloat16);
    if (nr_of_blocks > 1) {
        calc_tbdim_flattern(1, Ns, threadsPerBlock, blocksPerGrid);
        params.clear();
        params.push_back(sramA);
        params.push_back(sramA1);
        params.push_back(gelu_table_addr);
        params.push_back(sramOutBf16);
        params.push_back(bytes_per_block);
        params.push_back((nr_of_blocks - 1));
        params.push_back(mode);
        launchWrapperAysnc("llm_gelu", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
    }
    if (Nt > 0) {
        calc_tbdim_flattern(1, Nt, threadsPerBlock, blocksPerGrid);
        params.clear();
        params.push_back(sramA + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
        params.push_back(sramA1 + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
        params.push_back(gelu_table_addr);
        params.push_back(sramOutBf16 + (RPPdeviceptr) (nr_of_blocks - 1) * (RPPdeviceptr) bytes_per_block);
        params.push_back(0);
        params.push_back(1);
        params.push_back(mode);
        launchWrapperAysnc("llm_gelu", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
    }
    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, elements * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramOutBf16, sramOut, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_silu_build(rpp_kernel_context & ctx,
                           int                  mode,
                           int                  C,
                           int                  H,
                           int                  W,
                           int                  split_axis,
                           int                  in_bytes_per_element,
                           int                  out_bytes_per_element,
                           int                  use_sram_direct = 0,
                           int                  is_instantial   = 1,
                           int                  is_capture      = 1) {
    if (use_sram_direct) {
        if (mode == 0 || mode == 1) {
            rpp_silu_mode01_build_sram(ctx, mode, C, H, W, in_bytes_per_element, out_bytes_per_element, is_instantial, is_capture);
        } else {
            rpp_silu_mode2_build_sram(ctx, C, H, W, split_axis, in_bytes_per_element, out_bytes_per_element,
                                      is_instantial, is_capture);
        }
    } else {
        if (mode == 0 || mode == 1) {
            rpp_silu_mode01_build(ctx, mode, C, H, W, in_bytes_per_element, out_bytes_per_element, is_instantial, is_capture);
        } else {
            rpp_silu_mode2_build(ctx, C, H, W, split_axis, in_bytes_per_element, out_bytes_per_element, is_instantial, is_capture);
        }
    }
}

}  // namespace kernel_swiglu
