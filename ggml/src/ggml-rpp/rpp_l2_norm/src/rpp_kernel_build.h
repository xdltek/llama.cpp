#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_rms_norm/src/rpp_kernel_block.h"
#include "rpp_rms_norm/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

inline int round_up_to_32(int a) {
    return (a + 31) / 32 * 32;
}

inline int get_l2norm_rsqrt_max_rows() {
    return KDC_H_MAX_THREAD_NUM;
}

inline int get_l2norm_tile_sram_bytes(int tile_rows,
                                      int cols,
                                      int in_bytes_per_element,
                                      int out_bytes_per_element) {
    const int  size32               = tile_rows * cols * (int) sizeof(float);
    const int  size16               = tile_rows * cols * (int) sizeof(uint16_t);
    const int  sizeReduce           = tile_rows * (int) sizeof(uint16_t);
    const int  lutSize              = 64 * 1024 * (int) sizeof(uint16_t);
    const int  padded_cols          = round_up_to_32(cols);
    const bool use_row_tail_padding = tile_rows > 1 && padded_cols != cols;
    const int  size16_padded        = tile_rows * padded_cols * (int) sizeof(uint16_t);

    int total_sram_bytes = 0;
    total_sram_bytes += round_up(size32);
    total_sram_bytes += round_up(sizeReduce);
    total_sram_bytes += round_up(lutSize);
    total_sram_bytes += round_up(size16);
    total_sram_bytes += round_up(size16);
    if (use_row_tail_padding) {
        total_sram_bytes += round_up(size16_padded);
        total_sram_bytes += round_up(size16_padded);
    }
    return total_sram_bytes;
}

inline int get_max_l2norm_tile_rows(int rows,
                                    int cols,
                                    int in_bytes_per_element,
                                    int out_bytes_per_element,
                                    int sram_limit) {
    int lo = 0;
    int hi = rows;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        const int total_sram_bytes = get_l2norm_tile_sram_bytes(mid, cols, in_bytes_per_element, out_bytes_per_element);
        if (total_sram_bytes <= sram_limit) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if (lo <= 0) {
        std::cerr << "SRAM overflow: even one row does not fit in SRAM\n";
        std::abort();
    }
    return lo;
}

// Build graph once for L2 norm:
// y = x / max(sqrt(sum(x^2)), eps)
// Current kernel path approximates clamp via rsqrt(sum + eps^2).
static void rpp_l2norm_build(rpp_kernel_context & ctx,
                             int                  rows,
                             int                  cols,
                             float                eps,
                             int                  in_bytes_per_element,
                             int                  out_bytes_per_element,
                             int                  is_instantial = 1) {
    if (cols >= 8192) {
        throw std::runtime_error("L2 norm currently supports columns less than 8K");
    }

    const int  SRAM_LIMIT                = 22 * 1024 * 1024;
    const int  padded_cols               = round_up_to_32(cols);
    const int  rsqrt_max_rows            = get_l2norm_rsqrt_max_rows();
    const bool force_tiling_for_rsqrt    = rows > rsqrt_max_rows;
    const int  full_tensor_sram_bytes    = get_l2norm_tile_sram_bytes(rows, cols, in_bytes_per_element, out_bytes_per_element);
    const bool force_tiling_for_capacity = full_tensor_sram_bytes > SRAM_LIMIT;

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA = ctx.dev_in[0];
    RPPdeviceptr          devB = ctx.dev_out[0];

    int        lut_elements = 64 * 1024;
    uint16_t * rsqrt_table  = (uint16_t *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < (uint32_t) lut_elements; i++) {
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
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/rmsnorm.o");

    int max_tile_rows = rows;
    if (force_tiling_for_capacity || force_tiling_for_rsqrt) {
        max_tile_rows = get_max_l2norm_tile_rows(rows, cols, in_bytes_per_element, out_bytes_per_element, SRAM_LIMIT);
        max_tile_rows = std::min(max_tile_rows, rsqrt_max_rows);
    }

    for (int row_begin = 0; row_begin < rows; row_begin += max_tile_rows) {
        const int    tile_rows            = std::min(max_tile_rows, rows - row_begin);
        const int    sizeA                = tile_rows * cols * in_bytes_per_element;
        const int    sizeB                = tile_rows * cols * out_bytes_per_element;
        const int    size32               = tile_rows * cols * (int) sizeof(float);
        const int    size16               = tile_rows * cols * (int) sizeof(uint16_t);
        const int    sizeReduce           = tile_rows * (int) sizeof(uint16_t);
        const bool   use_row_tail_padding = tile_rows > 1 && padded_cols != cols;
        const int    row_bytes_bf16       = padded_cols * (int) sizeof(uint16_t);
        const int    size16_padded        = tile_rows * row_bytes_bf16;
        const int    lutSize              = 64 * 1024 * (int) sizeof(uint16_t);
        const size_t devA_offset          = (size_t) row_begin * (size_t) cols * (size_t) in_bytes_per_element;
        const size_t devB_offset          = (size_t) row_begin * (size_t) cols * (size_t) out_bytes_per_element;

        RPPdeviceptr sram_base          = ctx.virtual_sram_base;
        RPPdeviceptr sramIO             = sram_base;
        RPPdeviceptr sramReduce         = sramIO + round_up(size32);
        RPPdeviceptr rsqrt_table_addr   = sramReduce + round_up(sizeReduce);
        RPPdeviceptr workspace0_addr    = rsqrt_table_addr + round_up(lutSize);
        RPPdeviceptr workspace1_addr    = workspace0_addr + round_up(size16);
        RPPdeviceptr workspace2_addr    = workspace1_addr + round_up(size16);
        RPPdeviceptr padded_input_addr  = workspace2_addr + round_up(size16);
        RPPdeviceptr padded_output_addr = padded_input_addr + round_up(size16_padded);

        const int total_sram_bytes =
            (int) ((use_row_tail_padding ? (padded_output_addr + round_up(size16_padded)) :
                                          (workspace2_addr + round_up(size16))) -
                   sram_base);
        if (total_sram_bytes > SRAM_LIMIT) {
            std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                      << " bytes\n";
            std::abort();
        }

        rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[0], 0);
        rtMemcpyAsync((void *) sramIO, (const void *) (devA + (RPPdeviceptr) devA_offset), sizeA, rtMemcpyDeviceToSram,
                      ctx.dmaStream);
        rtMemcpyAsync((void *) rsqrt_table_addr, (const void *) dev_rsqrt_lut, lutSize, rtMemcpyDeviceToSram,
                      ctx.dmaStream);
        rppEventRecord(ctx.dma_done_ping[1], ctx.dmaStream);
        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[1], 0);

        if (in_bytes_per_element == (int) sizeof(float)) {
            calc_tbdim_flattern(1, tile_rows * cols, threadsPerBlock, blocksPerGrid);
            params.clear();
            cvt_kernel_param_init(threadsPerBlock, sramIO, workspace2_addr, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        const RPPdeviceptr compact_input_addr =
            (in_bytes_per_element == (int) sizeof(float)) ? workspace2_addr : sramIO;

        const int tensor_size               = tile_rows * cols;
        const int tensor_size_roundup_to_32 = round_up_to_32(tensor_size);
        dim3      pow_threads;
        dim3      pow_blocks;
        calc_tbdim_flattern(1, tensor_size, pow_threads, pow_blocks);
        if (pow_blocks.x > 1 && (pow_threads.x % 32) != 0) {
            calc_tbdim_flattern(1, tensor_size_roundup_to_32, pow_threads, pow_blocks);
        }

        params.clear();
        params.emplace_back(compact_input_addr);
        params.emplace_back(workspace0_addr);
        params.emplace_back(pow_threads.x * (int) sizeof(uint16_t));
        launchWrapperAysnc("matrix_rms_power_of_two", pow_blocks, pow_threads, params, ctx.rppBinMod, ctx.kernelStream);

        RppDims reduce_in_dims_2d{}, reduce_out_dims_2d{};
        reduce_in_dims_2d.nbDims  = 2;
        reduce_in_dims_2d.d[0]    = tile_rows;
        reduce_in_dims_2d.d[1]    = cols;
        reduce_out_dims_2d.nbDims = 2;
        reduce_out_dims_2d.d[0]   = tile_rows;
        reduce_out_dims_2d.d[1]   = 1;
        std::vector<RppDims> sub_reduce_io_dims;
        RppDims              middle_dims{};
        sub_reduce_io_dims.emplace_back(reduce_in_dims_2d);
        sub_reduce_io_dims.emplace_back(reduce_out_dims_2d);
        int io_cursor = 0;
        while (ReduceSpawnIO((int) 1, sub_reduce_io_dims[io_cursor], sub_reduce_io_dims[sub_reduce_io_dims.size() - 1],
                             middle_dims, true, false)) {
            sub_reduce_io_dims.insert(sub_reduce_io_dims.begin() + io_cursor + 1, middle_dims);
            io_cursor++;
        }

        uint32_t ping_pong_addr[2] = { workspace0_addr, workspace1_addr };
        for (size_t i = 0; i < sub_reduce_io_dims.size() - 1; i++) {
            RppDims  reduce_in_dims  = sub_reduce_io_dims[i];
            RppDims  reduce_out_dims = sub_reduce_io_dims[i + 1];
            uint32_t input_addr      = ping_pong_addr[i % 2];
            uint32_t output_addr     = (i == sub_reduce_io_dims.size() - 2 ? sramReduce : ping_pong_addr[(i + 1) % 2]);
            std::vector<RppTaskElement> tasks = create_reduce_kernel_task(
                1, RppReduceOperation::kSUM, input_addr, output_addr, reduce_in_dims, reduce_out_dims, 0, true, false);
            for (RppTaskElement & task : tasks) {
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }

        const float rsqrt_eps = eps * eps;
        params.clear();
        params.emplace_back(sramReduce);
        params.emplace_back(rsqrt_table_addr);
        params.emplace_back(workspace1_addr);
        params.emplace_back(*(uint32_t *) &rsqrt_eps);
        dim3 rsqrt_threads;
        rsqrt_threads.x = tile_rows <= 32 ? 33 : tile_rows;
        rsqrt_threads.y = 1;
        rsqrt_threads.z = 1;
        dim3 rsqrt_blocks;
        rsqrt_blocks.x = 1;
        rsqrt_blocks.y = 1;
        rsqrt_blocks.z = 1;
        launchWrapperAysnc("llm_add_rsqrt_lut", rsqrt_blocks, rsqrt_threads, params, ctx.rppBinMod, ctx.kernelStream);

        if (use_row_tail_padding) {
            if (in_bytes_per_element == (int) sizeof(float)) {
                for (int h = 0; h < tile_rows; ++h) {
                    calc_tbdim_flattern(1, cols, threadsPerBlock, blocksPerGrid);
                    params.clear();
                    cvt_kernel_param_init(threadsPerBlock, sramIO + (RPPdeviceptr) h * cols * (int) sizeof(float),
                                          padded_input_addr + (RPPdeviceptr) h * row_bytes_bf16, kFLOAT, kBF16, params);
                    launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                }
            } else {
                for (int h = 0; h < tile_rows; ++h) {
                    rtMemcpyAsync((void *) (padded_input_addr + (RPPdeviceptr) h * row_bytes_bf16),
                                  (const void *) (devA + (RPPdeviceptr) devA_offset +
                                                  (RPPdeviceptr) h * cols * (int) sizeof(uint16_t)),
                                  cols * (int) sizeof(uint16_t), rtMemcpyDeviceToSram, ctx.kernelStream);
                }
            }

            dim3 mul_threads;
            mul_threads.x = padded_cols;
            mul_threads.y = 1;
            mul_threads.z = 1;
            dim3 mul_blocks;
            mul_blocks.x = tile_rows;
            mul_blocks.y = 1;
            mul_blocks.z = 1;
            params.clear();
            params.emplace_back(padded_input_addr);
            params.emplace_back(workspace1_addr);
            params.emplace_back(padded_output_addr);
            params.emplace_back(row_bytes_bf16);
            params.emplace_back((int) sizeof(uint16_t));
            launchWrapperAysnc("matrix_rms_mul", mul_blocks, mul_threads, params, ctx.rppBinMod, ctx.kernelStream);
        } else {
            dim3 mul_threads;
            mul_threads.x = cols;
            mul_threads.y = 1;
            mul_threads.z = 1;
            dim3 mul_blocks;
            mul_blocks.x = tile_rows;
            mul_blocks.y = 1;
            mul_blocks.z = 1;
            params.clear();
            params.emplace_back(compact_input_addr);
            params.emplace_back(workspace1_addr);
            params.emplace_back(workspace2_addr);
            params.emplace_back(cols * (int) sizeof(uint16_t));
            params.emplace_back((int) sizeof(uint16_t));
            launchWrapperAysnc("matrix_rms_mul", mul_blocks, mul_threads, params, ctx.rppBinMod, ctx.kernelStream);
        }

        if (out_bytes_per_element == (int) sizeof(float)) {
            if (use_row_tail_padding) {
                for (int h = 0; h < tile_rows; ++h) {
                    calc_tbdim_flattern(1, cols * 2, threadsPerBlock, blocksPerGrid);
                    params.clear();
                    cvt_kernel_param_init_opt(threadsPerBlock, padded_output_addr + (RPPdeviceptr) h * row_bytes_bf16,
                                              sramIO + (RPPdeviceptr) h * cols * (int) sizeof(float), kBF16, kFLOAT,
                                              params);
                    launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params,
                                       ctx.rppBinMod, ctx.kernelStream);
                }
            } else {
                calc_tbdim_flattern(1, tile_rows * cols * 2, threadsPerBlock, blocksPerGrid);
                params.clear();
                cvt_kernel_param_init_opt(threadsPerBlock, workspace2_addr, sramIO, kBF16, kFLOAT, params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
            rtMemcpyAsync((void *) (devB + (RPPdeviceptr) devB_offset), (const void *) sramIO, sizeB,
                          rtMemcpySramToDevice, ctx.kernelStream);
        } else {
            if (use_row_tail_padding) {
                for (int h = 0; h < tile_rows; ++h) {
                    rtMemcpyAsync(
                        (void *) (devB + (RPPdeviceptr) devB_offset + (RPPdeviceptr) h * cols * (int) sizeof(uint16_t)),
                        (const void *) (padded_output_addr + (RPPdeviceptr) h * row_bytes_bf16),
                        cols * (int) sizeof(uint16_t), rtMemcpySramToDevice, ctx.kernelStream);
                }
            } else {
                rtMemcpyAsync((void *) (devB + (RPPdeviceptr) devB_offset), (const void *) workspace2_addr, sizeB,
                              rtMemcpySramToDevice, ctx.kernelStream);
            }
        }
    }

    free(rsqrt_table);
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
