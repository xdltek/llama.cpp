// memcpy_2d_rpp.cpp
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_flash_attn_ext/src/rpp_kernel_block.h"
#include "rpp_flash_attn_ext/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

inline int get_tn(int N) {
    if ((N % 128) == 0) {
        return 4;
    } else if ((N % 96) == 0) {
        return 3;
    } else if ((N % 64) == 0) {
        return 2;
    } else {
        return 1;
    }
}

static int round_up_32(int x) {
    return (x + 31) / 32 * 32;
}

static int choose_softmax_row_tile(int total_rows) {
    if (total_rows <= 1) {
        return total_rows;
    }

    // The pointwise softmax kernels use a fixed 32-lane X dimension, so Y must stay <= 255.
    int tile_rows = std::min(total_rows, 8191 / 32);
    while (tile_rows > 1 && (total_rows % tile_rows) == 1) {
        tile_rows--;
    }
    if (tile_rows <= 1) {
        throw std::runtime_error("Unable to tile softmax rows without creating a 1-row tail block");
    }
    return tile_rows;
}

static constexpr int kFlashAttnSramLimitBytes     = 22 * 1024 * 1024;
static constexpr int FLASH_VXM_SMALL_KV_THRESHOLD = 512;

static int64_t round_up_sram_bytes(int64_t bytes) {
    return (bytes + LOADALN_GUARD - 1) / LOADALN_GUARD * LOADALN_GUARD + LOADALN_GUARD;
}

static int scale_row_granularity(int Dp) {
    int granularity = 1;
    while (Dp * granularity < 4096) {
        granularity *= 2;
    }
    return granularity;
}

static bool is_valid_query_tile_rows(int rows, int Dp) {
    if (rows <= 0) {
        return false;
    }
    if (rows * Dp < 4096) {
        return true;
    }
    return (rows % scale_row_granularity(Dp)) == 0;
}

static int largest_valid_query_tile_rows(int rows, int Dp) {
    if (rows <= 0) {
        return 0;
    }
    if (is_valid_query_tile_rows(rows, Dp)) {
        return rows;
    }

    const int granularity = scale_row_granularity(Dp);
    const int rounded     = (rows / granularity) * granularity;
    if (rounded > 0) {
        return rounded;
    }
    return rows;
}

static int next_query_tile_rows(int remaining_rows, int tile_capacity, int Dp) {
    const int rows      = std::min(remaining_rows, tile_capacity);
    const int tile_rows = largest_valid_query_tile_rows(rows, Dp);
    if (tile_rows <= 0) {
        throw std::runtime_error("Unable to choose a valid flash attention Tq tile");
    }
    return tile_rows;
}

static int64_t flash_attn_full_path_sram_bytes(int Tq_tile,
                                               int Tk,
                                               int Dp,
                                               int kv_bytes_per_elem,
                                               int io_bytes_per_elem,
                                               int mask_bytes_per_elem) {
    const int64_t sizeQ        = (int64_t) Tq_tile * Dp * io_bytes_per_elem;
    const int64_t sizeQHw32    = (int64_t) Tq_tile * Dp * (int) sizeof(rpp::bfloat16);
    const int64_t sizeK        = (int64_t) Tk * Dp * kv_bytes_per_elem;
    const int64_t sizeKt       = (int64_t) Tk * (Dp + 1) * kv_bytes_per_elem;
    const int64_t sizeV        = (int64_t) Tk * Dp * kv_bytes_per_elem;
    const int64_t sizeMask     = (int64_t) Tq_tile * Tk * mask_bytes_per_elem;
    const int64_t sizeMaskHw32 = (int64_t) Tq_tile * Tk * (int) sizeof(rpp::bfloat16);
    const int64_t sizeOBf16    = (int64_t) Tq_tile * Dp * (int) sizeof(rpp::bfloat16);
    const int64_t sizeOIo      = (int64_t) Tq_tile * Dp * io_bytes_per_elem;
    const int64_t sizeScore    = (int64_t) Tq_tile * Tk * (int) sizeof(rpp::bfloat16);
    const int64_t lutSize      = 64 * 1024 * (int64_t) sizeof(short);

    int64_t total = 0;
    total += round_up_sram_bytes(sizeQ);
    total += round_up_sram_bytes(sizeK);
    total += round_up_sram_bytes(sizeKt);
    total += round_up_sram_bytes(sizeV);
    total += round_up_sram_bytes(sizeMask);
    total += round_up_sram_bytes(sizeOBf16);
    if (io_bytes_per_elem == (int) sizeof(float)) {
        total += round_up_sram_bytes(sizeOIo);
    }
    total += round_up_sram_bytes(sizeQHw32);
    total += round_up_sram_bytes(sizeK);
    total += round_up_sram_bytes(sizeV);
    total += round_up_sram_bytes(sizeMaskHw32);
    total += round_up_sram_bytes(sizeScore);
    total += round_up_sram_bytes(sizeScore);
    total += round_up_sram_bytes(lutSize);
    total += round_up_sram_bytes(lutSize);
    return total;
}

static int choose_query_tile_capacity(int Tq,
                                      int Tk,
                                      int Dp,
                                      int kv_bytes_per_elem,
                                      int io_bytes_per_elem,
                                      int mask_bytes_per_elem) {
    int lo   = 1;
    int hi   = Tq;
    int best = 0;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (flash_attn_full_path_sram_bytes(mid, Tk, Dp, kv_bytes_per_elem, io_bytes_per_elem, mask_bytes_per_elem) <=
            kFlashAttnSramLimitBytes) {
            best = mid;
            lo   = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    best = largest_valid_query_tile_rows(best, Dp);
    if (best <= 0) {
        throw std::runtime_error("flash attention SRAM is too small for even one Tq tile");
    }
    return best;
}

static void zero_fill_sram_16bit_aligned(RPPdeviceptr            addr,
                                         int                     bytes,
                                         dim3 &                  threadsPerBlock,
                                         dim3 &                  blocksPerGrid,
                                         std::vector<uint32_t> & params,
                                         RPPmodule &             module,
                                         rtStream_t              stream);

static bool is_vxm_kv_page_split_compatible(int Tk, int kv_page) {
    if (Tk <= 0 || kv_page <= 0 || (Tk % kv_page) != 0 || (kv_page % 32) != 0) {
        return false;
    }

    int block_x = kv_page;
    int block_z = Tk / kv_page;
    int grid_x  = 1;

    while (block_x * block_z >= 8192) {
        if ((block_x % 2) != 0) {
            return false;
        }
        block_x /= 2;
        grid_x *= 2;
    }

    return (block_x % 32) == 0 && (block_x * grid_x == kv_page);
}

static int select_vxm_kv_page(int Tk) {
    const int max_kv_page = 768;
    for (int kv_page = (Tk < max_kv_page ? Tk : max_kv_page); kv_page >= 32; kv_page -= 32) {
        if ((kv_page % 32) == 0 && (Tk % kv_page) == 0 && is_vxm_kv_page_split_compatible(Tk, kv_page)) {
            return kv_page;
        }
    }
    throw std::runtime_error("Current Tk not support VXM tile split");
}

static void KParam_Hw2Hw32_strided(dim3 &                  threadsPerBlock,
                                   uint32_t                input,
                                   uint32_t                output,
                                   int                     H,
                                   int                     input_stride_w,
                                   std::vector<uint32_t> & params) {
    params.clear();
    params.push_back(input);
    params.push_back(output);
    params.push_back(input_stride_w);
    params.push_back(threadsPerBlock.x);
    params.push_back(threadsPerBlock.x);
    params.push_back(H * threadsPerBlock.x);
    params.push_back(input_stride_w * threadsPerBlock.y * (int) sizeof(uint16_t));
    params.push_back(threadsPerBlock.x * threadsPerBlock.y * (int) sizeof(uint16_t));
}

static void launch_hw2hw32_transpose_f16(RPPdeviceptr            input,
                                         RPPdeviceptr            output,
                                         int                     H,
                                         int                     W,
                                         dim3 &                  threadsPerBlock,
                                         dim3 &                  blocksPerGrid,
                                         std::vector<uint32_t> & params,
                                         RPPmodule               cuMod,
                                         RPPstream               kernelStream) {
    if ((W % 32) != 0) {
        throw std::runtime_error("flashAttn input Mask shape is illegal!");
    }

    // Keep each transpose launch below the 8K-thread block limit while preserving the full source row stride.
    const int max_w_tile = (W < 8192) ? W : 4096;
    for (int w_start = 0; w_start < W;) {
        const int w_tile = std::min(max_w_tile, W - w_start);
        KShape_Hw2Hw32(H, w_tile, threadsPerBlock, blocksPerGrid);
        KParam_Hw2Hw32_strided(threadsPerBlock, (uint32_t) (input + (size_t) w_start * sizeof(uint16_t)),
                               (uint32_t) (output + (size_t) (w_start / 32) * (size_t) H * 32 * sizeof(uint16_t)), H, W,
                               params);
        launchWrapperAysnc("kv_transpose_f16", blocksPerGrid, threadsPerBlock, params, cuMod, kernelStream);
        w_start += w_tile;
    }
}

static void convert_reduce_task_to_split_f32(RppTaskElement & task,
                                             uint32_t         input_high_base,
                                             uint32_t         input_low_base,
                                             uint32_t         output_high_base,
                                             uint32_t         output_low_base) {
    const uint32_t task_input_high  = task.params.kernelList.at(0);
    const uint32_t task_output_high = task.params.kernelList.at(1);
    const uint32_t task_input_low   = input_low_base + (task_input_high - input_high_base);
    const uint32_t task_output_low  = output_low_base + (task_output_high - output_high_base);

    if (task.taskName == "opt_reduce_f16") {
        task.taskName = "opt_reduce_f32";
    } else if (task.taskName == "opt_gen_reduce_f16") {
        task.taskName = "opt_gen_reduce_f32";
    } else if (task.taskName == "opt_aln_reduce_f16") {
        task.taskName = "opt_aln_reduce_f32";
    } else if (task.taskName == "opt_aln_z32_reduce_f16") {
        task.taskName = "opt_aln_z32_reduce_f32";
    } else {
        throw std::runtime_error("No split-f32 reduce kernel supported for " + task.taskName);
    }

    task.params.kernelList.insert(task.params.kernelList.begin() + 2, task_input_low);
    task.params.kernelList.insert(task.params.kernelList.begin() + 3, task_output_low);
}

static void convert_reduce_task_to_reciprocal_f16(RppTaskElement & task, uint32_t reciprocal_table_addr) {
    if (task.taskName == "opt_gen_reduce_f16") {
        task.taskName = "opt_gen_reduce_reciprocal_f16";
    } else if (task.taskName == "opt_reduce_f16") {
        task.taskName = "opt_reduce_reciprocal_f16";
    } else if (task.taskName == "opt_aln_reduce_f16") {
        task.taskName = "opt_aln_reduce_reciprocal_f16";
    } else {
        throw std::runtime_error("No reciprocal BF16 reduce kernel supported for " + task.taskName);
    }
    task.params.kernelList.emplace_back(reciprocal_table_addr);
}

static void calc_scale_flat_launch(uint32_t total_elements, dim3 & threadsPerBlock, dim3 & blocksPerGrid) {
    if (total_elements == 0) {
        throw std::runtime_error("Scale launch requires non-zero element count");
    }

    const uint32_t max_threads    = 4096;
    const uint32_t x_candidates[] = { 256, 128, 64, 32, 16, 8, 4, 2, 1 };

    for (uint32_t block_x : x_candidates) {
        if (block_x > total_elements || (total_elements % block_x) != 0) {
            continue;
        }

        uint32_t max_block_y = total_elements / block_x;
        if (block_x * max_block_y > max_threads) {
            max_block_y = max_threads / block_x;
        }

        for (uint32_t block_y = max_block_y; block_y >= 1; --block_y) {
            const uint32_t block_elements = block_x * block_y;
            if ((total_elements % block_elements) != 0) {
                continue;
            }

            threadsPerBlock.x = block_x;
            threadsPerBlock.y = block_y;
            threadsPerBlock.z = 1;
            blocksPerGrid.x   = total_elements / block_elements;
            blocksPerGrid.y   = 1;
            blocksPerGrid.z   = 1;
            return;
        }
    }

    throw std::runtime_error("Unable to build flat scale launch");
}

struct VxmHeadBuffers {
    RPPdeviceptr devQGroup;
    RPPdeviceptr devKHead;
    RPPdeviceptr devVHead;
    RPPdeviceptr devOGroup;
    RPPdeviceptr sramQRaw;
    RPPdeviceptr sramQBf16;
    RPPdeviceptr sramQStage;
    RPPdeviceptr sramK;
    RPPdeviceptr sramKt;
    RPPdeviceptr sramV;
    RPPdeviceptr sramScoreHigh;
    RPPdeviceptr sramScoreLow;
    RPPdeviceptr sramSub;
    RPPdeviceptr sramProb;
    RPPdeviceptr sramMaxHigh;
    RPPdeviceptr sramMaxLow;
    RPPdeviceptr sramOutBf16;
    RPPdeviceptr sramOutIo;
    RPPdeviceptr sramOutStage;
};

struct VxmTileBank {
    RPPdeviceptr sramQRawAll;
    RPPdeviceptr sramQBf16All;
    RPPdeviceptr sramKPacked;
    RPPdeviceptr sramKtPacked;
    RPPdeviceptr sramKTransAll;
    RPPdeviceptr sramVAll;
    RPPdeviceptr sramScoreHighAll;
    RPPdeviceptr sramScoreLowAll;
    RPPdeviceptr sramSubAll;
    RPPdeviceptr sramProbAll;
    RPPdeviceptr sramMaxHighAll;
    RPPdeviceptr sramMaxLowAll;
    RPPdeviceptr sramReducePing0High;
    RPPdeviceptr sramReducePing0Low;
    RPPdeviceptr sramReducePing1High;
    RPPdeviceptr sramReducePing1Low;
    RPPdeviceptr sramOutBf16All;
    RPPdeviceptr sramOutIoAll;
    RPPdeviceptr end;
};

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_flash_atten_build_impl(rpp_kernel_context & ctx,
                                       int                  Tq,
                                       int                  Tk,
                                       int                  Hq,
                                       int                  Hkv,
                                       int                  D,
                                       float                scale,
                                       int                  kv_bytes_per_elem,
                                       int                  io_bytes_per_elem,
                                       int                  mask_bytes_per_elem,
                                       int                  is_instantial) {
    if (kv_bytes_per_elem != sizeof(rpp::bfloat16)) {
        throw std::runtime_error("Not Supportted KV Storage Type");
    }
    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devQ     = ctx.dev_in[0];
    RPPdeviceptr          devK     = ctx.dev_in[1];
    RPPdeviceptr          devV     = ctx.dev_in[2];
    RPPdeviceptr          devMask  = ctx.dev_in[3];
    RPPdeviceptr          devO     = ctx.dev_out[0];
    const RPPdeviceptr    devQBase = devQ;
    const RPPdeviceptr    devOBase = devO;

    // build exp table
    int        lut_elements = 64 * 1024;
    uint16_t * exp_table    = (uint16_t *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < lut_elements; i++) {
        uint32_t x = i;
        x <<= 16;
        float y      = std::exp(*(float *) &x);
        exp_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }

    // build rcp table
    uint16_t * div_table = (uint16_t *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < lut_elements; i++) {
        uint32_t x = i;
        x <<= 16;
        float y      = 1.0f / (*(float *) &x);
        div_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }
    // 0, -0
    div_table[0]      = 0;
    div_table[0x8000] = 0;

    RPPdeviceptr dev_exp_lut = ctx.dev_workspace;
    RPPdeviceptr dev_div_lut = dev_exp_lut + lut_elements * sizeof(short);

    rtMemcpy((void *) dev_exp_lut, (const void *) exp_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);
    rtMemcpy((void *) dev_div_lut, (const void *) div_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);

    const int  Dp           = round_up_32(D);
    const bool use_padded_d = (Dp != D);
    const int  Tq_tile_capacity =
        choose_query_tile_capacity(Tq, Tk, Dp, kv_bytes_per_elem, io_bytes_per_elem, mask_bytes_per_elem);
    std::vector<int> query_tile_rows;
    for (int remaining_rows = Tq; remaining_rows > 0;) {
        const int tile_rows = next_query_tile_rows(remaining_rows, Tq_tile_capacity, Dp);
        query_tile_rows.emplace_back(tile_rows);
        remaining_rows -= tile_rows;
    }

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/flash_atten.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int    sizeQ        = Tq_tile_capacity * Dp * io_bytes_per_elem;
    // Transposed/intermediate tensors in this path are BF16 even when the external IO is F32.
    const int    sizeQHw32    = Tq_tile_capacity * Dp * (int) sizeof(rpp::bfloat16);
    const int    sizeK        = Tk * Dp * kv_bytes_per_elem;
    // `blas_trans_padding` pads each K row to (D + 1) elements before transpose.
    const int    sizeKt       = Tk * (Dp + 1) * kv_bytes_per_elem;
    const int    sizeV        = Tk * Dp * kv_bytes_per_elem;
    const int    sizeMask     = Tq_tile_capacity * Tk * mask_bytes_per_elem;
    const int    sizeMaskHw32 = Tq_tile_capacity * Tk * (int) sizeof(rpp::bfloat16);
    const int    sizeOBf16    = Tq_tile_capacity * Dp * (int) sizeof(rpp::bfloat16);
    const int    sizeOIo      = Tq_tile_capacity * Dp * io_bytes_per_elem;
    const int    sizeScore    = Tq_tile_capacity * Tk * (int) sizeof(rpp::bfloat16);
    const int    lutSize      = 64 * 1024 * sizeof(short);
    RPPdeviceptr sram_base    = ctx.virtual_sram_base;
    RPPdeviceptr sramQ        = sram_base;
    RPPdeviceptr sramK        = sramQ + round_up(sizeQ);
    RPPdeviceptr sramKt       = sramK + round_up(sizeK);
    RPPdeviceptr sramV        = sramKt + round_up(sizeKt);
    RPPdeviceptr sramMask     = sramV + round_up(sizeV);
    RPPdeviceptr sramO        = sramMask + round_up(sizeMask);
    RPPdeviceptr sramO1       = sramO + round_up(sizeOBf16);

    RPPdeviceptr sramQr                = sramO1 + ((io_bytes_per_elem == (int) sizeof(float)) ? round_up(sizeOIo) : 0);
    RPPdeviceptr sramKr                = sramQr + round_up(sizeQHw32);
    RPPdeviceptr sramVr                = sramKr + round_up(sizeK);
    RPPdeviceptr sramMaskr             = sramVr + round_up(sizeV);
    RPPdeviceptr workspace0_addr       = sramMaskr + round_up(sizeMaskHw32);
    RPPdeviceptr workspace1_addr       = workspace0_addr + round_up(sizeScore);
    RPPdeviceptr exp_table_addr        = workspace1_addr + round_up(sizeScore);
    RPPdeviceptr reciprocal_table_addr = exp_table_addr + round_up(lut_elements * 2);

    int      tmp_space_size     = Tq_tile_capacity * 32 * sizeof(uint16_t);
    uint32_t out_tmp_space_addr = sramO;
    uint32_t in1_tmp_space_addr = sramQ;

    const int total_sram_bytes = (int) (reciprocal_table_addr + round_up(lut_elements * 2) - sram_base);
    const int SRAM_LIMIT       = kFlashAttnSramLimitBytes;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    // std::cout << "flash attention Tq tiling: Tq=" << Tq << " Tk=" << Tk << " D=" << D << " Dp=" << Dp
    //           << " IO=" << io_bytes_per_elem << " Mask=" << mask_bytes_per_elem << " tile_capacity=" << Tq_tile_capacity
    //           << " tile_count=" << query_tile_rows.size() << " tile_rows=[";
    // for (size_t i = 0; i < query_tile_rows.size(); i++) {
    //     if (i != 0) {
    //         std::cout << ",";
    //     }
    //     std::cout << query_tile_rows[i];
    // }
    // std::cout << "] sram=" << total_sram_bytes << "/" << SRAM_LIMIT << " bytes\n";

    rtMemcpyAsync((void *) exp_table_addr, (const void *) dev_exp_lut, lut_elements * 2, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) reciprocal_table_addr, (const void *) dev_div_lut, lut_elements * 2, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

    int          group          = Hq / Hkv;
    const size_t q_row_bytes    = (size_t) D * (size_t) io_bytes_per_elem;
    const size_t q_src_pitch    = q_row_bytes * (size_t) Hq;
    const size_t mask_row_bytes = (size_t) Tk * (size_t) mask_bytes_per_elem;
    const size_t out_row_bytes  = (size_t) D * (size_t) io_bytes_per_elem;
    const size_t out_dst_pitch  = out_row_bytes * (size_t) Hq;

    for (int h = 0; h < Hq; h++) {
        int kv_h = h / group;
        if (h == 0 || kv_h != ((h - 1) / group)) {
            RPPdeviceptr devK_h = devK + (size_t) kv_h * D * sizeof(rpp::bfloat16);
            RPPdeviceptr devV_h = devV + (size_t) kv_h * D * sizeof(rpp::bfloat16);
            if (use_padded_d) {
                zero_fill_sram_16bit_aligned(sramK, sizeK, threadsPerBlock, blocksPerGrid, params, ctx.rppBinMod,
                                             ctx.kernelStream);
                zero_fill_sram_16bit_aligned(sramV, sizeV, threadsPerBlock, blocksPerGrid, params, ctx.rppBinMod,
                                             ctx.kernelStream);
            }
            rtMemcpy2DAsync((void *) sramK, Dp * sizeof(rpp::bfloat16), (const void *) devK_h,
                            D * Hkv * sizeof(rpp::bfloat16), D * sizeof(rpp::bfloat16), Tk, rtMemcpyDeviceToSram,
                            ctx.kernelStream);
            rtMemcpy2DAsync((void *) sramV, Dp * sizeof(rpp::bfloat16), (const void *) devV_h,
                            D * Hkv * sizeof(rpp::bfloat16), D * sizeof(rpp::bfloat16), Tk, rtMemcpyDeviceToSram,
                            ctx.kernelStream);
            // K Tranpose from [Tk, D] to K[D, Tk]

            params.clear();
            KShape_TransPadding(Tk, Dp, threadsPerBlock, blocksPerGrid);
            KParam_TransPadding(threadsPerBlock, blocksPerGrid, sramK, sramKt, Dp, params);
            launchWrapperAysnc("blas_trans_padding", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            params.clear();
            KShape_Trans(Tk, Dp, threadsPerBlock, blocksPerGrid);
            KParam_Trans(threadsPerBlock, blocksPerGrid, sramKt, sramK, Tk, Dp, params);
            launchWrapperAysnc("blas_transpose_u16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            // K from HW to HW32
            launch_hw2hw32_transpose_f16(sramK, sramKr, Dp, Tk, threadsPerBlock, blocksPerGrid, params, ctx.rppBinMod,
                                         ctx.kernelStream);

            // V from HW to HW32
            launch_hw2hw32_transpose_f16(sramV, sramVr, Tk, Dp, threadsPerBlock, blocksPerGrid, params, ctx.rppBinMod,
                                         ctx.kernelStream);

            // V from HW to HW32
            params.clear();
            KShape_Hw2Hw32(Tk, Dp, threadsPerBlock, blocksPerGrid);
            KParam_Hw2Hw32(threadsPerBlock, blocksPerGrid, sramV, sramVr, Tk, Dp, params);
            launchWrapperAysnc("kv_transpose_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        const RPPdeviceptr devQ_h = devQBase + (size_t) h * q_row_bytes;
        const RPPdeviceptr devO_h = devOBase + (size_t) h * out_row_bytes;
        for (int q_start = 0; q_start < Tq;) {
            const int TqTile       = next_query_tile_rows(Tq - q_start, Tq_tile_capacity, Dp);
            const int sizeQTile    = TqTile * Dp * io_bytes_per_elem;
            const int sizeMaskTile = TqTile * Tk * mask_bytes_per_elem;
            const RPPdeviceptr sramMaskBf16Stage = (mask_bytes_per_elem == (int)sizeof(float)) ? sramMaskr : sramMask;
            const RPPdeviceptr sramMaskHw32Stage = (mask_bytes_per_elem == (int)sizeof(float)) ? sramMask : sramMaskr;
            const RPPdeviceptr sramQBf16Stage = (io_bytes_per_elem == (int)sizeof(float)) ? sramQr : sramQ;
            const RPPdeviceptr sramQHw32Stage = (io_bytes_per_elem == (int)sizeof(float)) ? sramQ : sramQr;
            
            rtMemcpyAsync((void *) sramMask, (const void *) (devMask + (size_t) q_start * mask_row_bytes), sizeMaskTile,
                          rtMemcpyDeviceToSram, ctx.kernelStream);

            if (mask_bytes_per_elem == sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(TqTile, Tk, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init(threadsPerBlock, sramMask, sramMask, kFLOAT, kBF16, params);
                launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }

            // Mask from HW to HW32
            launch_hw2hw32_transpose_f16(sramMaskBf16Stage, sramMaskHw32Stage, TqTile, Tk,
                threadsPerBlock, blocksPerGrid, params,
                ctx.rppBinMod, ctx.kernelStream);

            if (use_padded_d) {
                zero_fill_sram_16bit_aligned(sramQ, sizeQTile, threadsPerBlock, blocksPerGrid, params, ctx.rppBinMod,
                                             ctx.kernelStream);
            }
            rtMemcpy2DAsync((void *) sramQ, Dp * io_bytes_per_elem,
                            (const void *) (devQ_h + (size_t) q_start * q_src_pitch), q_src_pitch, q_row_bytes, TqTile,
                            rtMemcpyDeviceToSram, ctx.kernelStream);
            if (io_bytes_per_elem == sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(TqTile, Dp, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init(threadsPerBlock, sramQ, sramQ, kFLOAT, kBF16, params);
                launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }

            // Q from HW to HW32
            launch_hw2hw32_transpose_f16(sramQBf16Stage, sramQHw32Stage, TqTile, Dp,
                threadsPerBlock, blocksPerGrid, params,
                ctx.rppBinMod, ctx.kernelStream);

            // Q apply scale
            params.clear();
            KShape_Scale(TqTile, Dp, threadsPerBlock, blocksPerGrid);
            KParam_Scale(threadsPerBlock, blocksPerGrid, sramQr, sramQr, scale, params);
            launchWrapperAysnc("opt_scale_uniform_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            // First MM
            int         first_mm_a_row = TqTile;
            int         first_mm_a_col = Dp;
            int         first_mm_b_row = Dp;
            int         first_mm_b_col = Tk;
            std::string first_mm_kernel;
            params.clear();
            create_mm_chw32_unroll(first_mm_a_row, first_mm_a_col, first_mm_b_row, first_mm_b_col, sramQr, sramKr,
                                   workspace0_addr, first_mm_kernel, threadsPerBlock, blocksPerGrid, params);
            launchWrapperAysnc(first_mm_kernel, blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            // Add with Mask
            int add_block_len  = Tk * TqTile;
            int add_repeat_num = 1;
            while (add_block_len >= 8192) {
                add_block_len /= 2;
                add_repeat_num *= 2;
            }
            assert(add_repeat_num * add_block_len == Tk * TqTile);
            threadsPerBlock.x = add_block_len;
            threadsPerBlock.y = 1;
            threadsPerBlock.z = 1;
            blocksPerGrid.x   = 1;
            blocksPerGrid.y   = 1;
            blocksPerGrid.z   = 1;
            params.clear();
            params.push_back(sramMaskr);
            params.push_back(workspace0_addr);
            params.push_back(workspace1_addr);
            params.push_back(add_block_len * sizeof(uint16_t));
            params.push_back(add_repeat_num);
            launchWrapperAysnc("opt_binary_add_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            // Reduce Max for softmax
            // [Tk/32, TqTile, 32] --> [1, TqTile, 32]
            RppDims first_reduce_max_in_dims{}, first_reduce_max_out_dims{};
            first_reduce_max_in_dims.nbDims  = 3;
            first_reduce_max_in_dims.d[0]    = first_mm_b_col / 32;
            first_reduce_max_in_dims.d[1]    = first_mm_a_row;
            first_reduce_max_in_dims.d[2]    = 32;
            first_reduce_max_out_dims.nbDims = 3;
            first_reduce_max_out_dims.d[0]   = 1;
            first_reduce_max_out_dims.d[1]   = first_mm_a_row;
            first_reduce_max_out_dims.d[2]   = 32;
            std::vector<RppTaskElement> first_reduce_max_tasks =
                create_reduce_kernel_task(0, RppReduceOperation::kMAX, workspace1_addr, workspace0_addr,
                                          first_reduce_max_in_dims, first_reduce_max_out_dims, 0, true, false);
            launchWrapperAysnc(first_reduce_max_tasks[0].taskName, first_reduce_max_tasks[0].gridDim,
                               first_reduce_max_tasks[0].blockDim, first_reduce_max_tasks[0].params.kernelList,
                               ctx.rppBinMod, ctx.kernelStream);

            // [1, TqTile, 32] --> [1, TqTile, 1]
            RppDims second_reduce_max_in_dims{}, second_reduce_max_out_dims{};
            second_reduce_max_in_dims.nbDims  = 2;
            second_reduce_max_in_dims.d[0]    = first_mm_a_row;
            second_reduce_max_in_dims.d[1]    = 32;
            second_reduce_max_out_dims.nbDims = 2;
            second_reduce_max_out_dims.d[0]   = first_mm_a_row;
            second_reduce_max_out_dims.d[1]   = 1;
            std::vector<RppDims> second_reduce_max_io_dims;
            RppDims              second_reduce_max_middle_dims{};
            second_reduce_max_io_dims.emplace_back(second_reduce_max_in_dims);
            second_reduce_max_io_dims.emplace_back(second_reduce_max_out_dims);
            int second_reduce_max_io_cursor = 0;
            while (ReduceSpawnIO(1, second_reduce_max_io_dims[second_reduce_max_io_cursor],
                                 second_reduce_max_io_dims[second_reduce_max_io_dims.size() - 1],
                                 second_reduce_max_middle_dims, true, false)) {
                second_reduce_max_io_dims.insert(second_reduce_max_io_dims.begin() + second_reduce_max_io_cursor + 1,
                                                 second_reduce_max_middle_dims);
                second_reduce_max_io_cursor++;
            }
            if (second_reduce_max_io_dims[1].Length() * (int) sizeof(uint16_t) * 2 > tmp_space_size) {
                throw std::runtime_error("No enough workspace for Second Reduce Max.");
            }
            uint32_t second_reduce_max_ping_pong_addr[2] = { out_tmp_space_addr,
                                                             out_tmp_space_addr + tmp_space_size / 2 };
            for (size_t j = 0; j < second_reduce_max_io_dims.size() - 1; j++) {
                RppDims  reduce_in_dims  = second_reduce_max_io_dims[j];
                RppDims  reduce_out_dims = second_reduce_max_io_dims[j + 1];
                uint32_t input_addr      = (j == 0 ? workspace0_addr : second_reduce_max_ping_pong_addr[j % 2]);
                uint32_t output_addr =
                    (j == second_reduce_max_io_dims.size() - 2 ? in1_tmp_space_addr :
                                                                 second_reduce_max_ping_pong_addr[(j + 1) % 2]);
                std::vector<RppTaskElement> tasks =
                    create_reduce_kernel_task(1, RppReduceOperation::kMAX, input_addr, output_addr, reduce_in_dims,
                                              reduce_out_dims, 0, true, false);
                for (RppTaskElement & task : tasks) {
                    launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList,
                                       ctx.rppBinMod, ctx.kernelStream);
                }
            }

            // exp(x - max)
            if (first_mm_a_row == 1) {
                RppTaskElement sub_max_exp_task{};
                sub_max_exp_task.params.kernelList.clear();
                sub_max_exp_task.setBlockDim(33, first_mm_a_row, 1);
                sub_max_exp_task.setGridDim(1, 1, 1);
                sub_max_exp_task.appendParam(workspace1_addr);
                sub_max_exp_task.appendParam(in1_tmp_space_addr);
                sub_max_exp_task.appendParam(exp_table_addr);
                sub_max_exp_task.appendParam(workspace0_addr);
                sub_max_exp_task.appendParam(32 * sizeof(uint16_t));
                sub_max_exp_task.appendParam(0);
                sub_max_exp_task.appendParam(first_mm_b_col / 32);
                sub_max_exp_task.taskName = "llama3_sub_bc_x_exp_f16_opt_dmb";
                launchWrapperAysnc(sub_max_exp_task.taskName, sub_max_exp_task.gridDim, sub_max_exp_task.blockDim,
                                   sub_max_exp_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
            } else {
                const int      softmax_row_tile           = choose_softmax_row_tile(first_mm_a_row);
                const uint32_t softmax_row_bytes          = 32u * (uint32_t) sizeof(uint16_t);
                const uint32_t softmax_block_stride_bytes = (uint32_t) first_mm_a_row * softmax_row_bytes;
                for (int row_start = 0; row_start < first_mm_a_row; row_start += softmax_row_tile) {
                    const int row_tile = std::min(softmax_row_tile, first_mm_a_row - row_start);
                    if (row_tile <= 1) {
                        throw std::runtime_error("Unsupported softmax tail block height");
                    }

                    RppTaskElement sub_max_exp_task{};
                    sub_max_exp_task.params.kernelList.clear();
                    sub_max_exp_task.setBlockDim(32, row_tile, 1);
                    sub_max_exp_task.setGridDim(1, 1, 1);
                    sub_max_exp_task.appendParam(workspace1_addr + (uint32_t) row_start * softmax_row_bytes);
                    sub_max_exp_task.appendParam(in1_tmp_space_addr +
                                                 (uint32_t) row_start * (uint32_t) sizeof(uint16_t));
                    sub_max_exp_task.appendParam(exp_table_addr);
                    sub_max_exp_task.appendParam(workspace0_addr + (uint32_t) row_start * softmax_row_bytes);
                    sub_max_exp_task.appendParam(softmax_block_stride_bytes);
                    sub_max_exp_task.appendParam(0);
                    sub_max_exp_task.appendParam(first_mm_b_col / 32);
                    sub_max_exp_task.taskName = "llama3_sub_bc_x_exp_f16_opt";
                    launchWrapperAysnc(sub_max_exp_task.taskName, sub_max_exp_task.gridDim, sub_max_exp_task.blockDim,
                                       sub_max_exp_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
                }
            }

            // ReduceSum [Tk/32, TqTile, 32] -> [1, TqTile, 32]
            RppDims first_reduce_sum_in_dims{}, first_reduce_sum_out_dims{};
            first_reduce_sum_in_dims.nbDims  = 3;
            first_reduce_sum_in_dims.d[0]    = first_mm_b_col / 32;
            first_reduce_sum_in_dims.d[1]    = first_mm_a_row;
            first_reduce_sum_in_dims.d[2]    = 32;
            first_reduce_sum_out_dims.nbDims = 3;
            first_reduce_sum_out_dims.d[0]   = 1;
            first_reduce_sum_out_dims.d[1]   = first_mm_a_row;
            first_reduce_sum_out_dims.d[2]   = 32;
            std::vector<RppTaskElement> first_reduce_sum_tasks =
                create_reduce_kernel_task(0, RppReduceOperation::kSUM, workspace0_addr, workspace1_addr,
                                          first_reduce_sum_in_dims, first_reduce_sum_out_dims, 0, true, false);
            launchWrapperAysnc(first_reduce_sum_tasks[0].taskName, first_reduce_sum_tasks[0].gridDim,
                               first_reduce_sum_tasks[0].blockDim, first_reduce_sum_tasks[0].params.kernelList,
                               ctx.rppBinMod, ctx.kernelStream);

            // ReduceSum [1, TqTile, 32] -> [1, TqTile, 1]
            RppDims second_reduce_sum_in_dims{}, second_reduce_sum_out_dims{};
            second_reduce_sum_in_dims.nbDims  = 2;
            second_reduce_sum_in_dims.d[0]    = first_mm_a_row;
            second_reduce_sum_in_dims.d[1]    = 32;
            second_reduce_sum_out_dims.nbDims = 2;
            second_reduce_sum_out_dims.d[0]   = first_mm_a_row;
            second_reduce_sum_out_dims.d[1]   = 1;
            std::vector<RppDims> second_reduce_sum_io_dims;
            RppDims              second_reduce_sum_middle_dims{};
            second_reduce_sum_io_dims.emplace_back(second_reduce_sum_in_dims);
            second_reduce_sum_io_dims.emplace_back(second_reduce_sum_out_dims);
            int second_reduce_sum_io_cursor = 0;
            while (ReduceSpawnIO(1, second_reduce_sum_io_dims[second_reduce_sum_io_cursor],
                                 second_reduce_sum_io_dims[second_reduce_sum_io_dims.size() - 1],
                                 second_reduce_sum_middle_dims, true, false)) {
                second_reduce_sum_io_dims.insert(second_reduce_sum_io_dims.begin() + second_reduce_sum_io_cursor + 1,
                                                 second_reduce_sum_middle_dims);
                second_reduce_sum_io_cursor++;
            }
            uint32_t second_reduce_sum_ping_pong_addr[2] = { out_tmp_space_addr,
                                                             out_tmp_space_addr + tmp_space_size / 2 };
            for (size_t j = 0; j < second_reduce_sum_io_dims.size() - 1; j++) {
                RppDims  reduce_in_dims  = second_reduce_sum_io_dims[j];
                RppDims  reduce_out_dims = second_reduce_sum_io_dims[j + 1];
                uint32_t input_addr      = (j == 0 ? workspace1_addr : second_reduce_sum_ping_pong_addr[j % 2]);
                uint32_t output_addr =
                    (j == second_reduce_sum_io_dims.size() - 2 ? in1_tmp_space_addr :
                                                                 second_reduce_sum_ping_pong_addr[(j + 1) % 2]);
                std::vector<RppTaskElement> tasks =
                    create_reduce_kernel_task(1, RppReduceOperation::kSUM, input_addr, output_addr, reduce_in_dims,
                                              reduce_out_dims, 0, true, false);

                assert(tasks.size() == 1);
                for (RppTaskElement & task : tasks) {
                    // for last reduce task, apply div
                    if (j == second_reduce_sum_io_dims.size() - 2) {
                        if (task.taskName == "opt_gen_reduce_f16") {
                            task.taskName = "opt_gen_reduce_reciprocal_f16";
                        } else if (task.taskName == "opt_reduce_f16") {
                            task.taskName = "opt_reduce_reciprocal_f16";
                        } else {
                            assert(0);
                        }
                        task.params.kernelList.emplace_back(reciprocal_table_addr);
                        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList,
                                           ctx.rppBinMod, ctx.kernelStream);
                    } else {
                        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList,
                                           ctx.rppBinMod, ctx.kernelStream);
                    }
                }
            }
            // Softmax Div, [Tk/32, TqTile, 32] MUL [1, TqTile, 1]
            if (first_mm_a_row == 1) {
                RppTaskElement softmax_div_task{};
                softmax_div_task.params.kernelList.clear();
                softmax_div_task.setBlockDim(33, first_mm_a_row, 1);
                softmax_div_task.setGridDim(1, 1, 1);
                softmax_div_task.appendParam(workspace0_addr);
                softmax_div_task.appendParam(in1_tmp_space_addr);
                softmax_div_task.appendParam(workspace1_addr);
                softmax_div_task.appendParam(32 * sizeof(uint16_t));
                softmax_div_task.appendParam(first_mm_b_col / 32);
                softmax_div_task.taskName = "llama3_mul_bc_x_f16_dmb";
                launchWrapperAysnc(softmax_div_task.taskName, softmax_div_task.gridDim, softmax_div_task.blockDim,
                                   softmax_div_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
            } else {
                const int      softmax_row_tile           = choose_softmax_row_tile(first_mm_a_row);
                const uint32_t softmax_row_bytes          = 32u * (uint32_t) sizeof(uint16_t);
                const uint32_t softmax_block_stride_bytes = (uint32_t) first_mm_a_row * softmax_row_bytes;
                for (int row_start = 0; row_start < first_mm_a_row; row_start += softmax_row_tile) {
                    const int row_tile = std::min(softmax_row_tile, first_mm_a_row - row_start);
                    if (row_tile <= 1) {
                        throw std::runtime_error("Unsupported softmax tail block height");
                    }

                    RppTaskElement softmax_div_task{};
                    softmax_div_task.params.kernelList.clear();
                    softmax_div_task.setBlockDim(32, row_tile, 1);
                    softmax_div_task.setGridDim(1, 1, 1);
                    softmax_div_task.appendParam(workspace0_addr + (uint32_t) row_start * softmax_row_bytes);
                    softmax_div_task.appendParam(in1_tmp_space_addr +
                                                 (uint32_t) row_start * (uint32_t) sizeof(uint16_t));
                    softmax_div_task.appendParam(workspace1_addr + (uint32_t) row_start * softmax_row_bytes);
                    softmax_div_task.appendParam(softmax_block_stride_bytes);
                    softmax_div_task.appendParam(first_mm_b_col / 32);
                    softmax_div_task.taskName = "llama3_mul_bc_x_f16";
                    launchWrapperAysnc(softmax_div_task.taskName, softmax_div_task.gridDim, softmax_div_task.blockDim,
                                       softmax_div_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
                }
            }

            // second MM
            std::string second_mm_kernel;
            params.clear();
            int second_mm_a_row = TqTile;
            int second_mm_a_col = Tk;
            int second_mm_b_row = Tk;
            int second_mm_b_col = Dp;
            create_mm_chw32_unroll(second_mm_a_row, second_mm_a_col, second_mm_b_row, second_mm_b_col, workspace1_addr,
                                   sramVr, in1_tmp_space_addr, second_mm_kernel, threadsPerBlock, blocksPerGrid,
                                   params);
            launchWrapperAysnc(second_mm_kernel, blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            // -------------------------
            // (5) HW32 -> CHW (linear) into sramC_chw
            // -------------------------
            params.clear();
            chw322chw_blocks(1, TqTile, Dp, threadsPerBlock, blocksPerGrid);
            chw322chw_align_params((int) in1_tmp_space_addr, (int) sramO, TqTile, Dp, 0, threadsPerBlock.x,
                                   threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
            launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                               ctx.rppBinMod, ctx.kernelStream);

            if (io_bytes_per_elem == sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(1, TqTile * Dp * 2, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init_opt(threadsPerBlock, sramO, sramO1, kBF16, kFLOAT, params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);

                rtMemcpy2DAsync((void *) (devO_h + (size_t) q_start * out_dst_pitch), out_dst_pitch,
                                (const void *) sramO1, Dp * sizeof(float), D * sizeof(float), TqTile,
                                rtMemcpySramToDevice, ctx.kernelStream);
            } else {
                rtMemcpy2DAsync((void *) (devO_h + (size_t) q_start * out_dst_pitch), out_dst_pitch,
                                (const void *) sramO, Dp * sizeof(short), D * sizeof(short), TqTile,
                                rtMemcpySramToDevice, ctx.kernelStream);
            }

            q_start += TqTile;
        }
    }

    free(exp_table);
    free(div_table);
    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

void rpp_flash_atten_build_vxm(rpp_kernel_context & ctx,
                               int                  Tq,
                               int                  Tk,
                               int                  Hq,
                               int                  Hkv,
                               int                  D,
                               float                scale,
                               int                  kv_bytes_per_elem,
                               int                  io_bytes_per_elem,
                               int                  mask_bytes_per_elem,
                               int                  is_instantial) {
    if (Tq != 1) {
        throw std::runtime_error("rpp_flash_atten_build_vxm only supports Tq == 1");
    }
    if (kv_bytes_per_elem != sizeof(rpp::bfloat16)) {
        throw std::runtime_error("Not Supportted KV Storage Type");
    }
    if (Hkv <= 0 || (Hq % Hkv) != 0) {
        throw std::runtime_error("Hq must be divisible by Hkv for VXM");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devQ    = ctx.dev_in[0];
    RPPdeviceptr          devK    = ctx.dev_in[1];
    RPPdeviceptr          devV    = ctx.dev_in[2];
    RPPdeviceptr          devMask = ctx.dev_in[3];

    // build exp table
    const int  lut_elements = 64 * 1024;
    uint16_t * exp_table    = (uint16_t *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < lut_elements; i++) {
        uint32_t x = i;
        x <<= 16;
        float y      = std::exp(*(float *) &x);
        exp_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }

    // build rcp table
    uint16_t * div_table = (uint16_t *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < lut_elements; i++) {
        uint32_t x = i;
        x <<= 16;
        float y      = 1.0f / (*(float *) &x);
        div_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }
    // 0, -0
    div_table[0]      = 0;
    div_table[0x8000] = 0;

    RPPdeviceptr dev_exp_lut = ctx.dev_workspace;
    RPPdeviceptr dev_div_lut = dev_exp_lut + lut_elements * sizeof(short);

    rtMemcpy((void *) dev_exp_lut, (const void *) exp_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);
    rtMemcpy((void *) dev_div_lut, (const void *) div_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);

    //flashopt
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/flash_atten_vxm.o");

    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int group          = Hq / Hkv;
    int       kv_page        = select_vxm_kv_page(Tk);
    int       kv_page_repeat = Tk / kv_page;

    const int sizeMask        = Tk * mask_bytes_per_elem;
    const int sizeMaskBf16    = Tk * (int) sizeof(rpp::bfloat16);
    const int sizeQ           = group * D * io_bytes_per_elem;
    const int sizeQBf16       = group * D * (int) sizeof(rpp::bfloat16);
    const int sizeK           = Tk * D * (int) sizeof(rpp::bfloat16);
    // `blas_trans_padding` pads each K row to (D + 1) elements before transpose.
    const int sizeKt          = Tk * (D + 1) * (int) sizeof(rpp::bfloat16);
    const int sizeV           = Tk * D * (int) sizeof(rpp::bfloat16);
    const int sizeScore       = group * Tk * (int) sizeof(rpp::bfloat16);
    const int sizeReduce      = group * (int) sizeof(rpp::bfloat16);
    const int sizeReduceStage = group * kv_page * (int) sizeof(rpp::bfloat16);
    const int sizeOutBf16     = group * D * (int) sizeof(rpp::bfloat16);
    const int sizeOutIo       = group * D * io_bytes_per_elem;
    const int lutSize         = lut_elements * (int) sizeof(short);

    RPPdeviceptr sram_base             = ctx.virtual_sram_base;
    RPPdeviceptr sramMaskRaw           = sram_base;
    RPPdeviceptr sramMaskBf16          = sramMaskRaw + round_up(sizeMask);
    RPPdeviceptr sramQRaw              = sramMaskBf16 + round_up(sizeMaskBf16);
    RPPdeviceptr sramQBf16             = sramQRaw + round_up(sizeQ);
    RPPdeviceptr sramK                 = sramQBf16 + round_up(sizeQBf16);
    RPPdeviceptr sramKt                = sramK + round_up(sizeK);
    RPPdeviceptr sramV                 = sramKt + round_up(sizeKt);
    RPPdeviceptr sramScore             = sramV + round_up(sizeV);
    RPPdeviceptr sramScoreMasked       = sramScore + round_up(sizeScore);
    RPPdeviceptr sramExp               = sramScoreMasked + round_up(sizeScore);
    RPPdeviceptr sramProb              = sramExp + round_up(sizeScore);
    RPPdeviceptr sramMax               = sramProb + round_up(sizeScore);
    RPPdeviceptr sramSum               = sramMax + round_up(sizeReduce);
    RPPdeviceptr sramReducePing0High   = sramSum + round_up(sizeReduce);
    RPPdeviceptr sramReducePing0Low    = sramReducePing0High + round_up(sizeReduceStage);
    RPPdeviceptr sramReducePing1High   = sramReducePing0Low + round_up(sizeReduceStage);
    RPPdeviceptr sramReducePing1Low    = sramReducePing1High + round_up(sizeReduceStage);
    RPPdeviceptr sramOutBf16           = sramReducePing1Low + round_up(sizeReduceStage);
    RPPdeviceptr sramOutIo             = sramOutBf16 + round_up(sizeOutBf16);
    RPPdeviceptr exp_table_addr        = sramOutIo + round_up(sizeOutIo);
    RPPdeviceptr reciprocal_table_addr = exp_table_addr + round_up(lutSize);

    RPPdeviceptr sramMaskStage = (mask_bytes_per_elem == sizeof(float)) ? sramMaskBf16 : sramMaskRaw;
    RPPdeviceptr sramQStage    = (io_bytes_per_elem == sizeof(float)) ? sramQBf16 : sramQRaw;
    RPPdeviceptr sramOutStage  = (io_bytes_per_elem == sizeof(float)) ? sramOutIo : sramOutBf16;
    RPPdeviceptr sramScoreLow  = sramExp;

    const int total_sram_bytes = (int) (reciprocal_table_addr + round_up(lut_elements * 2) - sram_base);
    const int SRAM_LIMIT       = kFlashAttnSramLimitBytes;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    rtMemcpyAsync((void *) exp_table_addr, (const void *) dev_exp_lut, lut_elements * 2, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) reciprocal_table_addr, (const void *) dev_div_lut, lut_elements * 2, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramMaskRaw, (const void *) devMask, sizeMask, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (mask_bytes_per_elem == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, Tk, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramMaskRaw, sramMaskBf16, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    const size_t q_row_bytes   = (size_t) D * (size_t) io_bytes_per_elem;
    const size_t kv_row_bytes  = (size_t) D * sizeof(rpp::bfloat16);
    const size_t kv_src_pitch  = kv_row_bytes * (size_t) Hkv;
    const size_t out_row_bytes = (size_t) D * (size_t) io_bytes_per_elem;

    for (int kv_h = 0; kv_h < Hkv; kv_h++) {
        const int    q_group_head = kv_h * group;
        RPPdeviceptr devQ_group   = devQ + (size_t) q_group_head * q_row_bytes;
        RPPdeviceptr devK_h       = devK + (size_t) kv_h * kv_row_bytes;
        RPPdeviceptr devV_h       = devV + (size_t) kv_h * kv_row_bytes;
        RPPdeviceptr devO_group   = ctx.dev_out[0] + (size_t) q_group_head * out_row_bytes;

        rtMemcpy2DAsync((void *) sramQRaw, q_row_bytes, (const void *) devQ_group, q_row_bytes, q_row_bytes, group,
                        rtMemcpyDeviceToSram, ctx.kernelStream);
        if (io_bytes_per_elem == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(group, D, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init(threadsPerBlock, sramQRaw, sramQBf16, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        params.clear();
        KShape_Scale(group, D, threadsPerBlock, blocksPerGrid);
        KParam_Scale(threadsPerBlock, blocksPerGrid, sramQStage, sramQStage, scale, params);
        launchWrapperAysnc("opt_scale_uniform_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpy2DAsync((void *) sramK, kv_row_bytes, (const void *) devK_h, kv_src_pitch, kv_row_bytes, Tk,
                        rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramV, kv_row_bytes, (const void *) devV_h, kv_src_pitch, kv_row_bytes, Tk,
                        rtMemcpyDeviceToSram, ctx.kernelStream);

        // K transpose from [Tk, D] to [D, Tk].
        params.clear();
        KShape_TransPadding(Tk, D, threadsPerBlock, blocksPerGrid);
        KParam_TransPadding(threadsPerBlock, blocksPerGrid, sramK, sramKt, D, params);
        launchWrapperAysnc("blas_trans_padding", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        KShape_Trans(Tk, D, threadsPerBlock, blocksPerGrid);
        KParam_Trans(threadsPerBlock, blocksPerGrid, sramKt, sramK, Tk, D, params);
        launchWrapperAysnc("blas_transpose_u16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        RppTaskElement mm_task{};
        const int      expand                 = group;
        const uint32_t in2_addr               = (uint32_t) sramQStage;
        const uint32_t in1_addr               = (uint32_t) sramK;
        const uint32_t out1_addr              = (uint32_t) sramScore;
        const uint32_t mm_low_addr            = (uint32_t) sramScoreLow;
        // The expand_vxM kernel stores grouped rows with packed [group][Tk] layout.
        const uint32_t score_row_stride_bytes = (uint32_t) Tk * (uint32_t) sizeof(uint16_t);
        const int      score_row_stride_elems = Tk;
        uint32_t       block_x = 1, block_y = 1, block_z = 1;
        uint32_t       grid_x = 1, grid_y = 1, grid_z = 1;
        expand_kerneldim_calc_batch_vxM(1, D, Tk, kv_page, expand, block_x, block_y, block_z, grid_x, grid_y, grid_z);
        mm_task.blockDim.x = block_x;
        mm_task.blockDim.y = block_y;
        mm_task.blockDim.z = block_z;
        mm_task.gridDim.x  = grid_x;
        mm_task.gridDim.y  = grid_y;
        mm_task.gridDim.z  = grid_z;
        expand_batch_vxM_kernel_params(in2_addr, in1_addr, 0, out1_addr, 1, D, Tk, kv_page, expand, 0,
                                       mm_task.params.kernelList);
        mm_task.taskName = get_expand_batch_vxM_kernel(expand);
        // the low part of float result
        mm_task.params.kernelList.emplace_back(mm_low_addr);
        launchWrapperAysnc(mm_task.taskName, mm_task.gridDim, mm_task.blockDim, mm_task.params.kernelList,
                           ctx.rppBinMod, ctx.kernelStream);
        // Apply the BF16 mask page-by-page across the float32 score rows.
        // the input0 is In[G][N][kv_page]
        // the input1 is Mask[N][kv_page]
        // the output is Out[G][N][kv_page]
        RppTaskElement add_task{};
        const uint32_t in3_addr     = (uint32_t) sramMaskStage;
        int            add_block_x  = kv_page;
        int            add_block_y  = group;
        int            block_repeat = 1;
        while (add_block_x * add_block_y >= 8192) {
            add_block_y /= 2;
            block_repeat *= 2;
        }
        if (add_block_y == 0 || add_block_y * block_repeat != group) {
            throw std::runtime_error("VXM mask add does not support this group split");
        }
        if (block_repeat != 1) {
            throw std::runtime_error("VXM loop2_mid kernels require block_repeat == 1");
        }
        add_task.setBlockDim(add_block_x, add_block_y, 1);
        add_task.setGridDim(1, 1, 1);
        loop2_mid_add_bc_y_f32_f16_f32_params(out1_addr, mm_low_addr, in3_addr, out1_addr, mm_low_addr,
                                              kv_page * sizeof(uint16_t), score_row_stride_bytes,
                                              score_row_stride_elems, kv_page * sizeof(uint16_t), block_repeat,
                                              kv_page_repeat, add_task.params.kernelList);
        add_task.taskName = "llama3_loop2_mid_add_bc_y_f32_f16_f32";
        launchWrapperAysnc(add_task.taskName, add_task.gridDim, add_task.blockDim, add_task.params.kernelList,
                           ctx.rppBinMod, ctx.kernelStream);
        // reduce max
        // N == 1: reduce directly over the packed [G, Tk] row
        // N > 1: first collapse [G, N, kv_page] -> [G, kv_page], then reduce kv_page -> 1
        RppDims reduce1_in_dims{}, reduce1_out_dims{};
        reduce1_in_dims.nbDims  = 2;
        reduce1_in_dims.d[0]    = group;
        reduce1_out_dims.nbDims = 2;
        reduce1_out_dims.d[0]   = group;
        reduce1_out_dims.d[1]   = 1;

        uint32_t reduce1_input_high_addr = out1_addr;
        uint32_t reduce1_input_low_addr  = mm_low_addr;
        if (kv_page_repeat > 1) {
            RppTaskElement reduce1_stage0{};
            reduce1_stage0.setBlockDim(add_block_x, add_block_y, 1);
            reduce1_stage0.setGridDim(1, 1, 1);
            loop2_mid_reduce_max_f32_params(out1_addr, mm_low_addr, (uint32_t) sramReducePing0High,
                                            (uint32_t) sramReducePing0Low, kv_page * sizeof(uint16_t),
                                            score_row_stride_bytes, (uint32_t) score_row_stride_elems,
                                            (uint32_t) kv_page, (uint32_t) block_repeat, (uint32_t) kv_page_repeat,
                                            reduce1_stage0.params.kernelList);
            reduce1_stage0.taskName = "llama3_loop2_mid_reduce_max_f32";
            launchWrapperAysnc(reduce1_stage0.taskName, reduce1_stage0.gridDim, reduce1_stage0.blockDim,
                               reduce1_stage0.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

            reduce1_in_dims.d[1]    = kv_page;
            reduce1_input_high_addr = (uint32_t) sramReducePing0High;
            reduce1_input_low_addr  = (uint32_t) sramReducePing0Low;
        } else {
            reduce1_in_dims.d[1] = Tk;
        }

        std::vector<RppDims> reduce1_io_dims;
        reduce1_io_dims.emplace_back(reduce1_in_dims);
        reduce1_io_dims.emplace_back(reduce1_out_dims);
        RppDims reduce1_mid_dims{};
        int     reduce1_cursor = 0;
        while (ReduceSpawnIO(1, reduce1_io_dims[reduce1_cursor], reduce1_io_dims[reduce1_io_dims.size() - 1],
                             reduce1_mid_dims, true, false)) {
            reduce1_io_dims.insert(reduce1_io_dims.begin() + reduce1_cursor + 1, reduce1_mid_dims);
            reduce1_cursor++;
        }

        const uint32_t reduce1_ping_high_addr[2] = { (uint32_t) sramReducePing0High, (uint32_t) sramReducePing1High };
        const uint32_t reduce1_ping_low_addr[2]  = { (uint32_t) sramReducePing0Low, (uint32_t) sramReducePing1Low };

        for (size_t i = 0; i < reduce1_io_dims.size() - 1; i++) {
            RppDims        in_dims         = reduce1_io_dims[i];
            RppDims        out_dims        = reduce1_io_dims[i + 1];
            const uint32_t input_high_addr = (i == 0) ? reduce1_input_high_addr : reduce1_ping_high_addr[i % 2];
            const uint32_t input_low_addr  = (i == 0) ? reduce1_input_low_addr : reduce1_ping_low_addr[i % 2];
            const uint32_t output_high_addr =
                (i == reduce1_io_dims.size() - 2) ? (uint32_t) sramMax : reduce1_ping_high_addr[(i + 1) % 2];
            const uint32_t output_low_addr =
                (i == reduce1_io_dims.size() - 2) ? (uint32_t) sramSum : reduce1_ping_low_addr[(i + 1) % 2];

            std::vector<RppTaskElement> tasks = create_reduce_kernel_task(
                1, RppReduceOperation::kMAX, input_high_addr, output_high_addr, in_dims, out_dims, 0, true, false);
            for (RppTaskElement & task : tasks) {
                convert_reduce_task_to_split_f32(task, input_high_addr, input_low_addr, output_high_addr,
                                                 output_low_addr);
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }
        // exp(x - max)
        // score  : split-f32 [G, N, kv_page] with physical row stride Tk
        // max    : split-f32 [G, 1]
        // sub    : BF16      [G, N, kv_page]
        // exp    : BF16      [G, N, kv_page]
        const uint32_t sub_output_addr = (uint32_t) sramScoreMasked;
        const uint32_t exp_output_addr = (uint32_t) sramProb;

        RppTaskElement softmax_norm_sub{};
        softmax_norm_sub.setBlockDim(add_block_x, add_block_y, 1);
        softmax_norm_sub.setGridDim(1, 1, 1);
        softmax_norm_sub.appendParam(out1_addr);
        softmax_norm_sub.appendParam(mm_low_addr);
        softmax_norm_sub.appendParam((uint32_t) sramMax);
        softmax_norm_sub.appendParam((uint32_t) sramSum);
        softmax_norm_sub.appendParam(sub_output_addr);
        softmax_norm_sub.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
        softmax_norm_sub.appendParam(score_row_stride_bytes);
        softmax_norm_sub.appendParam(add_block_y * (uint32_t) sizeof(uint16_t));
        softmax_norm_sub.appendParam((uint32_t) score_row_stride_elems);
        softmax_norm_sub.appendParam((uint32_t) block_repeat);
        softmax_norm_sub.appendParam((uint32_t) kv_page_repeat);
        softmax_norm_sub.taskName = "llama3_loop2_mid_sub_bc_x_f32_f16";
        launchWrapperAysnc(softmax_norm_sub.taskName, softmax_norm_sub.gridDim, softmax_norm_sub.blockDim,
                           softmax_norm_sub.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

        RppTaskElement softmax_exp{};
        softmax_exp.setBlockDim(add_block_x, add_block_y, 1);
        softmax_exp.setGridDim(1, 1, 1);
        softmax_exp.appendParam(sub_output_addr);
        softmax_exp.appendParam((uint32_t) exp_table_addr);
        softmax_exp.appendParam(exp_output_addr);
        softmax_exp.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
        softmax_exp.appendParam(score_row_stride_bytes);
        softmax_exp.appendParam((uint32_t) score_row_stride_elems);
        softmax_exp.appendParam((uint32_t) block_repeat);
        softmax_exp.appendParam((uint32_t) kv_page_repeat);
        softmax_exp.taskName = "llama3_loop2_mid_mish_f16";
        launchWrapperAysnc(softmax_exp.taskName, softmax_exp.gridDim, softmax_exp.blockDim,
                           softmax_exp.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

        // reduce sum
        // N == 1: reduce directly over the packed [G, Tk] row
        // N > 1: first collapse [G, N, kv_page] -> [G, kv_page], then reduce kv_page -> 1
        RppDims reduce_sum1_in_dims{}, reduce_sum1_out_dims{};
        reduce_sum1_in_dims.nbDims  = 2;
        reduce_sum1_in_dims.d[0]    = group;
        reduce_sum1_out_dims.nbDims = 2;
        reduce_sum1_out_dims.d[0]   = group;
        reduce_sum1_out_dims.d[1]   = 1;

        uint32_t reduce_sum1_input_addr = exp_output_addr;
        if (kv_page_repeat > 1) {
            RppTaskElement reduce_sum1_stage0{};
            reduce_sum1_stage0.setBlockDim(add_block_x, add_block_y, 1);
            reduce_sum1_stage0.setGridDim(1, 1, 1);
            reduce_sum1_stage0.appendParam(exp_output_addr);
            reduce_sum1_stage0.appendParam((uint32_t) sramReducePing0High);
            reduce_sum1_stage0.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
            reduce_sum1_stage0.appendParam(score_row_stride_bytes);
            reduce_sum1_stage0.appendParam((uint32_t) score_row_stride_elems);
            reduce_sum1_stage0.appendParam((uint32_t) kv_page);
            reduce_sum1_stage0.appendParam((uint32_t) block_repeat);
            reduce_sum1_stage0.appendParam((uint32_t) kv_page_repeat);
            reduce_sum1_stage0.taskName = "llama3_loop2_mid_reduce_sum_f16";
            launchWrapperAysnc(reduce_sum1_stage0.taskName, reduce_sum1_stage0.gridDim, reduce_sum1_stage0.blockDim,
                               reduce_sum1_stage0.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

            reduce_sum1_in_dims.d[1] = kv_page;
            reduce_sum1_input_addr   = (uint32_t) sramReducePing0High;
        } else {
            reduce_sum1_in_dims.d[1] = Tk;
        }

        std::vector<RppDims> reduce_sum1_io_dims;
        reduce_sum1_io_dims.emplace_back(reduce_sum1_in_dims);
        reduce_sum1_io_dims.emplace_back(reduce_sum1_out_dims);
        RppDims reduce_sum1_mid_dims{};
        int     reduce_sum1_cursor = 0;
        while (ReduceSpawnIO(1, reduce_sum1_io_dims[reduce_sum1_cursor],
                             reduce_sum1_io_dims[reduce_sum1_io_dims.size() - 1], reduce_sum1_mid_dims, true, false)) {
            reduce_sum1_io_dims.insert(reduce_sum1_io_dims.begin() + reduce_sum1_cursor + 1, reduce_sum1_mid_dims);
            reduce_sum1_cursor++;
        }

        const uint32_t reduce_sum1_ping_addr[2] = { (uint32_t) sramReducePing0High, (uint32_t) sramReducePing1High };
        const uint32_t reciprocal_output_addr   = (uint32_t) sramSum;
        for (size_t i = 0; i < reduce_sum1_io_dims.size() - 1; i++) {
            RppDims        in_dims    = reduce_sum1_io_dims[i];
            RppDims        out_dims   = reduce_sum1_io_dims[i + 1];
            const uint32_t input_addr = (i == 0) ? reduce_sum1_input_addr : reduce_sum1_ping_addr[i % 2];
            const uint32_t output_addr =
                (i == reduce_sum1_io_dims.size() - 2) ? reciprocal_output_addr : reduce_sum1_ping_addr[(i + 1) % 2];

            std::vector<RppTaskElement> tasks = create_reduce_kernel_task(
                1, RppReduceOperation::kSUM, input_addr, output_addr, in_dims, out_dims, 0, true, false);
            if (i == reduce_sum1_io_dims.size() - 2 && tasks.size() != 1) {
                throw std::runtime_error("Unexpected multi-task final reduce-sum stage");
            }
            for (RppTaskElement & task : tasks) {
                if (i == reduce_sum1_io_dims.size() - 2) {
                    convert_reduce_task_to_reciprocal_f16(task, (uint32_t) reciprocal_table_addr);
                }
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }

        // apply the reduced 1/sum
        // input  is exp_output_addr      [G][N][kv_page]
        // input1 is reciprocal_output_addr [G][1]
        // output is prob                [G][N][kv_page]
        RppTaskElement softmax_mul{};
        softmax_mul.setBlockDim(add_block_x, add_block_y, 1);
        softmax_mul.setGridDim(1, 1, 1);
        softmax_mul.appendParam(exp_output_addr);
        softmax_mul.appendParam(reciprocal_output_addr);
        softmax_mul.appendParam(exp_output_addr);
        softmax_mul.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
        softmax_mul.appendParam(score_row_stride_bytes);
        softmax_mul.appendParam(add_block_y * (uint32_t) sizeof(uint16_t));
        softmax_mul.appendParam((uint32_t) score_row_stride_elems);
        softmax_mul.appendParam((uint32_t) block_repeat);
        softmax_mul.appendParam((uint32_t) kv_page_repeat);
        softmax_mul.taskName = "llama3_loop2_mid_mul_bc_x_f16";
        launchWrapperAysnc(softmax_mul.taskName, softmax_mul.gridDim, softmax_mul.blockDim,
                           softmax_mul.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
        // Launch the updated second MM:
        // input0 is prob [G][N][kv_page], input1 is V [N][kv_page][D], output is [G][D].

        RppTaskElement second_mm_task{};
        second_mm_task.taskName    = "expand_matrix_batch_vxM_group_f16_dyn";
        uint32_t second_mm_block_x = 1, second_mm_block_y = 1, second_mm_block_z = 1;
        uint32_t second_mm_grid_x = 1, second_mm_grid_y = 1, second_mm_grid_z = 1;
        expand_kerneldim_calc_batch_vxM_group(1, kv_page, D, group, second_mm_block_x, second_mm_block_y,
                                              second_mm_block_z, second_mm_grid_x, second_mm_grid_y, second_mm_grid_z);
        second_mm_task.blockDim.x = second_mm_block_x;
        second_mm_task.blockDim.y = second_mm_block_y;
        second_mm_task.blockDim.z = second_mm_block_z;
        second_mm_task.gridDim.x  = second_mm_grid_x;
        second_mm_task.gridDim.y  = second_mm_grid_y;
        second_mm_task.gridDim.z  = second_mm_grid_z;
        expand_batch_vxM_group_kernel_params(exp_output_addr, (uint32_t) sramV, 0, (uint32_t) sramOutBf16, 1, kv_page,
                                             kv_page_repeat, D, group, 0, second_mm_task.params.kernelList);
        // Keep the ABI slot populated even though the current kernel ignores shrNum.
        second_mm_task.appendParam(0);
        second_mm_task.appendParam((uint32_t) kv_page_repeat);
        launchWrapperAysnc(second_mm_task.taskName, second_mm_task.gridDim, second_mm_task.blockDim,
                           second_mm_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

        if (io_bytes_per_elem == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, group * D * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramOutBf16, sramOutIo, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        rtMemcpy2DAsync((void *) devO_group, out_row_bytes, (const void *) sramOutStage, out_row_bytes, out_row_bytes,
                        group, rtMemcpySramToDevice, ctx.kernelStream);
    }

    free(exp_table);
    free(div_table);
    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_flash_atten_build_vxm_v1_tiled_hkv(rpp_kernel_context & ctx,
                                                   int                  Tk,
                                                   int                  Hq,
                                                   int                  Hkv,
                                                   int                  D,
                                                   float                scale,
                                                   int                  io_bytes_per_elem,
                                                   int                  mask_bytes_per_elem,
                                                   RPPdeviceptr         devQ,
                                                   RPPdeviceptr         devK,
                                                   RPPdeviceptr         devV,
                                                   RPPdeviceptr         devMask,
                                                   RPPdeviceptr         dev_exp_lut,
                                                   RPPdeviceptr         dev_div_lut,
                                                   int                  lut_elements,
                                                   int                  hkv_tile_max,
                                                   int                  is_instantial = 1) {
    const int group          = Hq / Hkv;
    const int kv_page        = select_vxm_kv_page(Tk);
    const int kv_page_repeat = Tk / kv_page;
    const int num_hkv_tiles  = (Hkv + hkv_tile_max - 1) / hkv_tile_max;
    const int tile_hq_max    = group * hkv_tile_max;
    const int lutSize        = lut_elements * (int) sizeof(short);

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    const int sizeMask            = Tk * mask_bytes_per_elem;
    const int sizeMaskBf16        = Tk * (int) sizeof(rpp::bfloat16);
    const int sizeQGroupRaw       = group * D * io_bytes_per_elem;
    const int sizeQGroupBf16      = group * D * (int) sizeof(rpp::bfloat16);
    const int sizeK               = Tk * D * (int) sizeof(rpp::bfloat16);
    const int sizeKFusedTileMax   = Tk * D * hkv_tile_max * (int) sizeof(rpp::bfloat16);
    const int sizeKtFusedTileMax  = Tk * (D * hkv_tile_max + 1) * (int) sizeof(rpp::bfloat16);
    const int sizeV               = Tk * D * (int) sizeof(rpp::bfloat16);
    const int sizeScore           = group * Tk * (int) sizeof(rpp::bfloat16);
    const int sizeReduce          = group * (int) sizeof(rpp::bfloat16);
    const int sizeReduceStageTile = tile_hq_max * kv_page * (int) sizeof(rpp::bfloat16);
    const int sizeOutBf16         = group * D * (int) sizeof(rpp::bfloat16);
    const int sizeOutIo           = group * D * io_bytes_per_elem;

    const size_t q_row_bytes     = (size_t) D * (size_t) io_bytes_per_elem;
    const size_t q_group_bytes   = (size_t) sizeQGroupRaw;
    const size_t kv_row_bytes    = (size_t) D * sizeof(rpp::bfloat16);
    const size_t kv_src_pitch    = kv_row_bytes * (size_t) Hkv;
    const size_t out_row_bytes   = (size_t) D * (size_t) io_bytes_per_elem;
    const size_t out_group_bytes = (size_t) sizeOutIo;

    const size_t q_raw_stride =
        (io_bytes_per_elem == sizeof(float)) ? (size_t) round_up(sizeQGroupBf16) * 2 : (size_t) round_up(sizeQGroupRaw);
    const size_t q_bf16_stride    = (size_t) round_up(sizeQGroupBf16);
    const size_t q_bf16_total     = (io_bytes_per_elem == sizeof(float)) ? (size_t) hkv_tile_max * q_bf16_stride : 0;
    const size_t k_fused_stride   = (size_t) round_up(sizeKFusedTileMax);
    const size_t kt_fused_stride  = (size_t) round_up(sizeKtFusedTileMax);
    const size_t v_all_bytes      = (size_t) hkv_tile_max * (size_t) sizeV;
    const size_t score_all_bytes  = (size_t) hkv_tile_max * (size_t) sizeScore;
    const size_t reduce_all_bytes = (size_t) hkv_tile_max * (size_t) sizeReduce;
    const size_t out_bf16_stride  = (size_t) round_up(sizeOutBf16);
    const size_t out_io_stride =
        (io_bytes_per_elem == sizeof(float)) ? out_bf16_stride * 2 : (size_t) round_up(sizeOutIo);
    const size_t out_stage_stride_bytes =
        (size_t) ((io_bytes_per_elem == sizeof(float)) ? out_io_stride : out_bf16_stride);
    const size_t out_io_total = (io_bytes_per_elem == sizeof(float)) ? (size_t) hkv_tile_max * out_io_stride : 0;

    auto build_tile_bank = [&](RPPdeviceptr base) -> VxmTileBank {
        VxmTileBank  bank{};
        RPPdeviceptr cursor      = base;
        bank.sramQRawAll         = cursor;
        cursor                   = bank.sramQRawAll + (RPPdeviceptr) ((size_t) hkv_tile_max * q_raw_stride);
        bank.sramQBf16All        = cursor;
        cursor                   = bank.sramQBf16All + (RPPdeviceptr) q_bf16_total;
        bank.sramKPacked         = cursor;
        cursor                   = bank.sramKPacked + (RPPdeviceptr) k_fused_stride;
        bank.sramKtPacked        = cursor;
        cursor                   = bank.sramKtPacked + (RPPdeviceptr) kt_fused_stride;
        bank.sramKTransAll       = cursor;
        cursor                   = bank.sramKTransAll + (RPPdeviceptr) k_fused_stride;
        bank.sramVAll            = cursor;
        cursor                   = bank.sramVAll + round_up((int) v_all_bytes);
        bank.sramScoreHighAll    = cursor;
        cursor                   = bank.sramScoreHighAll + round_up((int) score_all_bytes);
        bank.sramScoreLowAll     = cursor;
        cursor                   = bank.sramScoreLowAll + round_up((int) score_all_bytes);
        bank.sramSubAll          = cursor;
        cursor                   = bank.sramSubAll + round_up((int) score_all_bytes);
        bank.sramProbAll         = cursor;
        cursor                   = bank.sramProbAll + round_up((int) score_all_bytes);
        bank.sramMaxHighAll      = cursor;
        cursor                   = bank.sramMaxHighAll + round_up((int) reduce_all_bytes);
        bank.sramMaxLowAll       = cursor;
        cursor                   = bank.sramMaxLowAll + round_up((int) reduce_all_bytes);
        bank.sramReducePing0High = cursor;
        cursor                   = bank.sramReducePing0High + round_up(sizeReduceStageTile);
        bank.sramReducePing0Low  = cursor;
        cursor                   = bank.sramReducePing0Low + round_up(sizeReduceStageTile);
        bank.sramReducePing1High = cursor;
        cursor                   = bank.sramReducePing1High + round_up(sizeReduceStageTile);
        bank.sramReducePing1Low  = cursor;
        cursor                   = bank.sramReducePing1Low + round_up(sizeReduceStageTile);
        bank.sramOutBf16All      = cursor;
        cursor                   = bank.sramOutBf16All + (RPPdeviceptr) ((size_t) hkv_tile_max * out_bf16_stride);
        bank.sramOutIoAll        = cursor;
        cursor                   = bank.sramOutIoAll + (RPPdeviceptr) out_io_total;
        bank.end                 = cursor;
        return bank;
    };

    RPPdeviceptr sram_base    = ctx.virtual_sram_base;
    RPPdeviceptr sramMaskRaw  = sram_base;
    RPPdeviceptr sramMaskBf16 = sramMaskRaw + round_up(sizeMask);
    VxmTileBank  tile_banks[2];
    tile_banks[0]                      = build_tile_bank(sramMaskBf16 + round_up(sizeMaskBf16));
    tile_banks[1]                      = build_tile_bank(tile_banks[0].end);
    RPPdeviceptr exp_table_addr        = tile_banks[1].end;
    RPPdeviceptr reciprocal_table_addr = exp_table_addr + round_up(lutSize);

    const int total_sram_bytes = (int) (reciprocal_table_addr + round_up(lut_elements * 2) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "VXM v1 tiled SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                  << " bytes\n";
        std::abort();
    }

    auto build_tile_heads = [&](const VxmTileBank & bank, int hkv_base, int tile_hkv,
                                std::vector<VxmHeadBuffers> & tile_heads) {
        tile_heads.resize((size_t) tile_hkv);
        for (int local_h = 0; local_h < tile_hkv; ++local_h) {
            VxmHeadBuffers & head         = tile_heads[(size_t) local_h];
            const int        global_h     = hkv_base + local_h;
            const int        q_group_head = global_h * group;
            head.devQGroup                = devQ + (size_t) q_group_head * q_row_bytes;
            head.devKHead                 = devK + (size_t) global_h * kv_row_bytes;
            head.devVHead                 = devV + (size_t) global_h * kv_row_bytes;
            head.devOGroup                = ctx.dev_out[0] + (size_t) q_group_head * out_row_bytes;
            head.sramQRaw                 = bank.sramQRawAll + (RPPdeviceptr) ((size_t) local_h * q_raw_stride);
            head.sramQBf16                = (io_bytes_per_elem == sizeof(float)) ?
                                                bank.sramQBf16All + (RPPdeviceptr) ((size_t) local_h * q_bf16_stride) :
                                                head.sramQRaw;
            head.sramQStage               = (io_bytes_per_elem == sizeof(float)) ? head.sramQBf16 : head.sramQRaw;
            head.sramK                    = bank.sramKTransAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeK);
            head.sramKt                   = bank.sramKtPacked;
            head.sramV                    = bank.sramVAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeV);
            head.sramScoreHigh = bank.sramScoreHighAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeScore);
            head.sramScoreLow  = bank.sramScoreLowAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeScore);
            head.sramSub       = bank.sramSubAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeScore);
            head.sramProb      = bank.sramProbAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeScore);
            head.sramMaxHigh   = bank.sramMaxHighAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeReduce);
            head.sramMaxLow    = bank.sramMaxLowAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeReduce);
            head.sramOutBf16   = bank.sramOutBf16All + (RPPdeviceptr) ((size_t) local_h * out_bf16_stride);
            head.sramOutIo     = (io_bytes_per_elem == sizeof(float)) ?
                                     bank.sramOutIoAll + (RPPdeviceptr) ((size_t) local_h * out_io_stride) :
                                     head.sramOutBf16;
            head.sramOutStage  = (io_bytes_per_elem == sizeof(float)) ? head.sramOutIo : head.sramOutBf16;
        }
    };

    uint32_t mm_block_x = 1, mm_block_y = 1, mm_block_z = 1;
    uint32_t mm_grid_x = 1, mm_grid_y = 1, mm_grid_z = 1;
    expand_kerneldim_calc_batch_vxM(1, D, Tk, kv_page, group, mm_block_x, mm_block_y, mm_block_z, mm_grid_x, mm_grid_y,
                                    mm_grid_z);
    const std::string mm_kernel_name = get_expand_batch_vxM_kernel(group);

    bool     use_hkv_mm_kernel = true;
    uint32_t hkv_mm_block_x = 1, hkv_mm_block_y = 1, hkv_mm_block_z = 1;
    uint32_t hkv_mm_grid_x = 1, hkv_mm_grid_y = 1, hkv_mm_grid_z = 1;
    if (use_hkv_mm_kernel) {
        expand_kerneldim_calc_batch_hkv_vxM(D, Tk, group, hkv_mm_block_x, hkv_mm_block_y, hkv_mm_block_z, hkv_mm_grid_x,
                                            hkv_mm_grid_y, hkv_mm_grid_z);
        use_hkv_mm_kernel = true;
    }

    uint32_t second_mm_block_x = 1, second_mm_block_y = 1, second_mm_block_z = 1;
    uint32_t second_mm_grid_x = 1, second_mm_grid_y = 1, second_mm_grid_z = 1;
    expand_kerneldim_calc_batch_vxM_group(1, kv_page, D, group, second_mm_block_x, second_mm_block_y, second_mm_block_z,
                                          second_mm_grid_x, second_mm_grid_y, second_mm_grid_z);

    auto schedule_tile_dma = [&](int tile_idx, int ping) {
        const int           hkv_base = tile_idx * hkv_tile_max;
        const int           tile_hkv = ((hkv_base + hkv_tile_max) <= Hkv) ? hkv_tile_max : (Hkv - hkv_base);
        const VxmTileBank & bank     = tile_banks[ping];

        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

        rtMemcpy2DAsync((void *) bank.sramQRawAll, q_raw_stride,
                        (const void *) (devQ + (size_t) (hkv_base * group) * q_row_bytes), q_group_bytes, q_group_bytes,
                        tile_hkv, rtMemcpyDeviceToSram, ctx.dmaStream);

        for (int local_h = 0; local_h < tile_hkv; ++local_h) {
            const int global_h = hkv_base + local_h;
            rtMemcpy2DAsync((void *) (bank.sramVAll + (RPPdeviceptr) ((size_t) local_h * (size_t) sizeV)), kv_row_bytes,
                            (const void *) (devV + (size_t) global_h * kv_row_bytes), kv_src_pitch, kv_row_bytes, Tk,
                            rtMemcpyDeviceToSram, ctx.dmaStream);
        }

        rtMemcpy2DAsync((void *) bank.sramKPacked, kv_row_bytes * (size_t) tile_hkv,
                        (const void *) (devK + (size_t) hkv_base * kv_row_bytes), kv_src_pitch,
                        kv_row_bytes * (size_t) tile_hkv, Tk, rtMemcpyDeviceToSram, ctx.dmaStream);
        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    auto launch_tile = [&](int tile_idx, int ping, RPPdeviceptr sramMaskStage) {
        const int                   hkv_base = tile_idx * hkv_tile_max;
        const int                   tile_hkv = ((hkv_base + hkv_tile_max) <= Hkv) ? hkv_tile_max : (Hkv - hkv_base);
        const int                   tile_hq  = group * tile_hkv;
        const int                   tile_fused_k_width     = D * tile_hkv;
        const uint32_t              score_row_stride_bytes = (uint32_t) Tk * (uint32_t) sizeof(uint16_t);
        const int                   score_row_stride_elems = Tk;
        const VxmTileBank &         bank                   = tile_banks[ping];
        std::vector<VxmHeadBuffers> tile_heads;
        build_tile_heads(bank, hkv_base, tile_hkv, tile_heads);

        if (io_bytes_per_elem == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, (uint32_t) ((size_t) tile_hkv * q_bf16_stride / sizeof(uint16_t)), threadsPerBlock,
                                blocksPerGrid);
            cvt_kernel_param_init(threadsPerBlock, (uint32_t) tile_heads[0].sramQRaw,
                                  (uint32_t) tile_heads[0].sramQBf16, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        {
            const uint32_t q_stage_stride_bytes =
                (uint32_t) ((io_bytes_per_elem == sizeof(float)) ? q_bf16_stride : q_raw_stride);
            const uint32_t q_stage_total_bytes    = (uint32_t) ((size_t) tile_hkv * (size_t) q_stage_stride_bytes);
            const uint32_t q_stage_total_elements = q_stage_total_bytes / (uint32_t) sizeof(uint16_t);
            calc_scale_flat_launch(q_stage_total_elements, threadsPerBlock, blocksPerGrid);
            params.clear();
            KParam_Scale(threadsPerBlock, blocksPerGrid, (uint32_t) tile_heads[0].sramQStage,
                         (uint32_t) tile_heads[0].sramQStage, scale, params);
            launchWrapperAysnc("opt_scale_uniform_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        {
            params.clear();
            KShape_TransPadding(Tk, tile_fused_k_width, threadsPerBlock, blocksPerGrid);
            KParam_TransPadding(threadsPerBlock, blocksPerGrid, bank.sramKPacked, bank.sramKtPacked, tile_fused_k_width,
                                params);
            launchWrapperAysnc("blas_trans_padding", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        {
            params.clear();
            KShape_Trans(Tk, tile_fused_k_width, threadsPerBlock, blocksPerGrid);
            KParam_Trans(threadsPerBlock, blocksPerGrid, bank.sramKtPacked, bank.sramKTransAll, Tk, tile_fused_k_width,
                         params);
            launchWrapperAysnc("blas_transpose_u16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        for (int local_h = 0; local_h < tile_hkv; ++local_h) {
            const VxmHeadBuffers & head = tile_heads[(size_t) local_h];
            RppTaskElement         mm_task{};
            if (use_hkv_mm_kernel) {
                mm_task.blockDim.x = hkv_mm_block_x;
                mm_task.blockDim.y = hkv_mm_block_y;
                mm_task.blockDim.z = hkv_mm_block_z;
                mm_task.gridDim.x  = hkv_mm_grid_x;
                mm_task.gridDim.y  = hkv_mm_grid_y;
                mm_task.gridDim.z  = hkv_mm_grid_z;
                expand_batch_hkv_vxM_kernel_params((uint32_t) head.sramQStage, (uint32_t) head.sramK,
                                                   (uint32_t) head.sramScoreHigh, (uint32_t) head.sramScoreLow, D, Tk,
                                                   hkv_mm_block_x, mm_task.params.kernelList);
                mm_task.taskName = get_expand_batch_hkv_vxM_kernel();
            } else {
                mm_task.blockDim.x = mm_block_x;
                mm_task.blockDim.y = mm_block_y;
                mm_task.blockDim.z = mm_block_z;
                mm_task.gridDim.x  = mm_grid_x;
                mm_task.gridDim.y  = mm_grid_y;
                mm_task.gridDim.z  = mm_grid_z;
                expand_batch_vxM_kernel_params((uint32_t) head.sramQStage, (uint32_t) head.sramK, 0,
                                               (uint32_t) head.sramScoreHigh, 1, D, Tk, kv_page, group, 0,
                                               mm_task.params.kernelList);
                mm_task.taskName = mm_kernel_name;
                mm_task.params.kernelList.emplace_back((uint32_t) head.sramScoreLow);
            }
            launchWrapperAysnc(mm_task.taskName, mm_task.gridDim, mm_task.blockDim, mm_task.params.kernelList,
                               ctx.rppBinMod, ctx.kernelStream);
        }

        const int reduce_block_x     = kv_page;
        int       reduce_stage0_rows = 8191 / reduce_block_x;
        if (reduce_stage0_rows < 1) {
            reduce_stage0_rows = 1;
        }
        if (reduce_stage0_rows > tile_hq) {
            reduce_stage0_rows = tile_hq;
        }

        int loop2_rows_per_launch = 8191 / kv_page;
        if (loop2_rows_per_launch < 1) {
            loop2_rows_per_launch = 1;
        }
        if (loop2_rows_per_launch > 255) {
            loop2_rows_per_launch = 255;
        }
        if (loop2_rows_per_launch > tile_hq) {
            loop2_rows_per_launch = tile_hq;
        }

        for (int row_base = 0; row_base < tile_hq; row_base += loop2_rows_per_launch) {
            const int tile_rows =
                (((row_base + loop2_rows_per_launch) <= tile_hq) ? loop2_rows_per_launch : (tile_hq - row_base));
            const uint32_t row_offset_bytes = (uint32_t) row_base * score_row_stride_bytes;

            RppTaskElement add_task{};
            add_task.setBlockDim(kv_page, tile_rows, 1);
            add_task.setGridDim(1, 1, 1);
            loop2_mid_add_bc_y_f32_f16_f32_params(
                (uint32_t) bank.sramScoreHighAll + row_offset_bytes, (uint32_t) bank.sramScoreLowAll + row_offset_bytes,
                (uint32_t) sramMaskStage, (uint32_t) bank.sramScoreHighAll + row_offset_bytes,
                (uint32_t) bank.sramScoreLowAll + row_offset_bytes, kv_page * sizeof(uint16_t),
                score_row_stride_bytes * (uint32_t) tile_rows, score_row_stride_elems, kv_page * sizeof(uint16_t), 1,
                (uint32_t) kv_page_repeat, add_task.params.kernelList);
            add_task.taskName = "llama3_loop2_mid_add_bc_y_f32_f16_f32";
            launchWrapperAysnc(add_task.taskName, add_task.gridDim, add_task.blockDim, add_task.params.kernelList,
                               ctx.rppBinMod, ctx.kernelStream);
        }

        {
            RppDims reduce1_in_dims{}, reduce1_out_dims{};
            reduce1_in_dims.nbDims  = 2;
            reduce1_in_dims.d[0]    = tile_hq;
            reduce1_out_dims.nbDims = 2;
            reduce1_out_dims.d[0]   = tile_hq;
            reduce1_out_dims.d[1]   = 1;

            uint32_t reduce1_input_high_addr = (uint32_t) bank.sramScoreHighAll;
            uint32_t reduce1_input_low_addr  = (uint32_t) bank.sramScoreLowAll;
            if (kv_page_repeat > 1) {
                const uint32_t reduce_stage0_output_row_bytes = (uint32_t) kv_page * (uint32_t) sizeof(uint16_t);
                for (int row_base = 0; row_base < tile_hq; row_base += reduce_stage0_rows) {
                    const int tile_rows =
                        (((row_base + reduce_stage0_rows) <= tile_hq) ? reduce_stage0_rows : (tile_hq - row_base));
                    const uint32_t input_row_offset  = (uint32_t) row_base * score_row_stride_bytes;
                    const uint32_t output_row_offset = (uint32_t) row_base * reduce_stage0_output_row_bytes;

                    RppTaskElement reduce1_stage0{};
                    reduce1_stage0.setBlockDim(reduce_block_x, tile_rows, 1);
                    reduce1_stage0.setGridDim(1, 1, 1);
                    loop2_mid_reduce_max_f32_params(
                        (uint32_t) bank.sramScoreHighAll + input_row_offset,
                        (uint32_t) bank.sramScoreLowAll + input_row_offset,
                        (uint32_t) bank.sramReducePing0High + output_row_offset,
                        (uint32_t) bank.sramReducePing0Low + output_row_offset, kv_page * sizeof(uint16_t),
                        score_row_stride_bytes * (uint32_t) tile_rows, (uint32_t) score_row_stride_elems,
                        (uint32_t) kv_page, 1, (uint32_t) kv_page_repeat, reduce1_stage0.params.kernelList);
                    reduce1_stage0.taskName = "llama3_loop2_mid_reduce_max_f32";
                    launchWrapperAysnc(reduce1_stage0.taskName, reduce1_stage0.gridDim, reduce1_stage0.blockDim,
                                       reduce1_stage0.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
                }

                reduce1_in_dims.d[1]    = kv_page;
                reduce1_input_high_addr = (uint32_t) bank.sramReducePing0High;
                reduce1_input_low_addr  = (uint32_t) bank.sramReducePing0Low;
            } else {
                reduce1_in_dims.d[1] = Tk;
            }

            std::vector<RppDims> reduce1_io_dims;
            reduce1_io_dims.emplace_back(reduce1_in_dims);
            reduce1_io_dims.emplace_back(reduce1_out_dims);
            RppDims reduce1_mid_dims{};
            int     reduce1_cursor = 0;
            while (ReduceSpawnIO(1, reduce1_io_dims[reduce1_cursor], reduce1_io_dims[reduce1_io_dims.size() - 1],
                                 reduce1_mid_dims, true, false)) {
                reduce1_io_dims.insert(reduce1_io_dims.begin() + reduce1_cursor + 1, reduce1_mid_dims);
                reduce1_cursor++;
            }

            const uint32_t reduce1_ping_high_addr[2] = { (uint32_t) bank.sramReducePing0High,
                                                         (uint32_t) bank.sramReducePing1High };
            const uint32_t reduce1_ping_low_addr[2]  = { (uint32_t) bank.sramReducePing0Low,
                                                         (uint32_t) bank.sramReducePing1Low };

            for (size_t i = 0; i < reduce1_io_dims.size() - 1; i++) {
                RppDims        in_dims          = reduce1_io_dims[i];
                RppDims        out_dims         = reduce1_io_dims[i + 1];
                const uint32_t input_high_addr  = (i == 0) ? reduce1_input_high_addr : reduce1_ping_high_addr[i % 2];
                const uint32_t input_low_addr   = (i == 0) ? reduce1_input_low_addr : reduce1_ping_low_addr[i % 2];
                const uint32_t output_high_addr = (i == reduce1_io_dims.size() - 2) ?
                                                      (uint32_t) bank.sramMaxHighAll :
                                                      reduce1_ping_high_addr[(i + 1) % 2];
                const uint32_t output_low_addr  = (i == reduce1_io_dims.size() - 2) ? (uint32_t) bank.sramMaxLowAll :
                                                                                      reduce1_ping_low_addr[(i + 1) % 2];

                std::vector<RppTaskElement> tasks = create_reduce_kernel_task(
                    1, RppReduceOperation::kMAX, input_high_addr, output_high_addr, in_dims, out_dims, 0, true, false);
                for (RppTaskElement & task : tasks) {
                    convert_reduce_task_to_split_f32(task, input_high_addr, input_low_addr, output_high_addr,
                                                     output_low_addr);
                    launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList,
                                       ctx.rppBinMod, ctx.kernelStream);
                }
            }
        }

        for (int row_base = 0; row_base < tile_hq; row_base += loop2_rows_per_launch) {
            const int tile_rows =
                (((row_base + loop2_rows_per_launch) <= tile_hq) ? loop2_rows_per_launch : (tile_hq - row_base));
            const uint32_t row_offset_bytes    = (uint32_t) row_base * score_row_stride_bytes;
            const uint32_t scalar_offset_bytes = (uint32_t) row_base * (uint32_t) sizeof(uint16_t);

            RppTaskElement softmax_norm_sub{};
            softmax_norm_sub.setBlockDim(kv_page, tile_rows, 1);
            softmax_norm_sub.setGridDim(1, 1, 1);
            softmax_norm_sub.appendParam((uint32_t) bank.sramScoreHighAll + row_offset_bytes);
            softmax_norm_sub.appendParam((uint32_t) bank.sramScoreLowAll + row_offset_bytes);
            softmax_norm_sub.appendParam((uint32_t) bank.sramMaxHighAll + scalar_offset_bytes);
            softmax_norm_sub.appendParam((uint32_t) bank.sramMaxLowAll + scalar_offset_bytes);
            softmax_norm_sub.appendParam((uint32_t) bank.sramSubAll + row_offset_bytes);
            softmax_norm_sub.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
            softmax_norm_sub.appendParam(score_row_stride_bytes * (uint32_t) tile_rows);
            softmax_norm_sub.appendParam((uint32_t) tile_rows * (uint32_t) sizeof(uint16_t));
            softmax_norm_sub.appendParam((uint32_t) score_row_stride_elems);
            softmax_norm_sub.appendParam(1);
            softmax_norm_sub.appendParam((uint32_t) kv_page_repeat);
            softmax_norm_sub.taskName = "llama3_loop2_mid_sub_bc_x_f32_f16";
            launchWrapperAysnc(softmax_norm_sub.taskName, softmax_norm_sub.gridDim, softmax_norm_sub.blockDim,
                               softmax_norm_sub.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
        }

        for (int row_base = 0; row_base < tile_hq; row_base += loop2_rows_per_launch) {
            const int tile_rows =
                (((row_base + loop2_rows_per_launch) <= tile_hq) ? loop2_rows_per_launch : (tile_hq - row_base));
            const uint32_t row_offset_bytes = (uint32_t) row_base * score_row_stride_bytes;

            RppTaskElement softmax_exp{};
            softmax_exp.setBlockDim(kv_page, tile_rows, 1);
            softmax_exp.setGridDim(1, 1, 1);
            softmax_exp.appendParam((uint32_t) bank.sramSubAll + row_offset_bytes);
            softmax_exp.appendParam((uint32_t) exp_table_addr);
            softmax_exp.appendParam((uint32_t) bank.sramProbAll + row_offset_bytes);
            softmax_exp.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
            softmax_exp.appendParam(score_row_stride_bytes * (uint32_t) tile_rows);
            softmax_exp.appendParam((uint32_t) score_row_stride_elems);
            softmax_exp.appendParam(1);
            softmax_exp.appendParam((uint32_t) kv_page_repeat);
            softmax_exp.taskName = "llama3_loop2_mid_mish_f16";
            launchWrapperAysnc(softmax_exp.taskName, softmax_exp.gridDim, softmax_exp.blockDim,
                               softmax_exp.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
        }

        {
            RppDims reduce_sum1_in_dims{}, reduce_sum1_out_dims{};
            reduce_sum1_in_dims.nbDims  = 2;
            reduce_sum1_in_dims.d[0]    = tile_hq;
            reduce_sum1_out_dims.nbDims = 2;
            reduce_sum1_out_dims.d[0]   = tile_hq;
            reduce_sum1_out_dims.d[1]   = 1;

            uint32_t reduce_sum1_input_addr = (uint32_t) bank.sramProbAll;
            if (kv_page_repeat > 1) {
                const uint32_t reduce_stage0_output_row_bytes = (uint32_t) kv_page * (uint32_t) sizeof(uint16_t);
                for (int row_base = 0; row_base < tile_hq; row_base += reduce_stage0_rows) {
                    const int tile_rows =
                        (((row_base + reduce_stage0_rows) <= tile_hq) ? reduce_stage0_rows : (tile_hq - row_base));
                    const uint32_t input_row_offset  = (uint32_t) row_base * score_row_stride_bytes;
                    const uint32_t output_row_offset = (uint32_t) row_base * reduce_stage0_output_row_bytes;

                    RppTaskElement reduce_sum1_stage0{};
                    reduce_sum1_stage0.setBlockDim(reduce_block_x, tile_rows, 1);
                    reduce_sum1_stage0.setGridDim(1, 1, 1);
                    reduce_sum1_stage0.appendParam((uint32_t) bank.sramProbAll + input_row_offset);
                    reduce_sum1_stage0.appendParam((uint32_t) bank.sramReducePing0High + output_row_offset);
                    reduce_sum1_stage0.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
                    reduce_sum1_stage0.appendParam(score_row_stride_bytes * (uint32_t) tile_rows);
                    reduce_sum1_stage0.appendParam((uint32_t) score_row_stride_elems);
                    reduce_sum1_stage0.appendParam((uint32_t) kv_page);
                    reduce_sum1_stage0.appendParam(1);
                    reduce_sum1_stage0.appendParam((uint32_t) kv_page_repeat);
                    reduce_sum1_stage0.taskName = "llama3_loop2_mid_reduce_sum_f16";
                    launchWrapperAysnc(reduce_sum1_stage0.taskName, reduce_sum1_stage0.gridDim,
                                       reduce_sum1_stage0.blockDim, reduce_sum1_stage0.params.kernelList, ctx.rppBinMod,
                                       ctx.kernelStream);
                }

                reduce_sum1_in_dims.d[1] = kv_page;
                reduce_sum1_input_addr   = (uint32_t) bank.sramReducePing0High;
            } else {
                reduce_sum1_in_dims.d[1] = Tk;
            }

            std::vector<RppDims> reduce_sum1_io_dims;
            reduce_sum1_io_dims.emplace_back(reduce_sum1_in_dims);
            reduce_sum1_io_dims.emplace_back(reduce_sum1_out_dims);
            RppDims reduce_sum1_mid_dims{};
            int     reduce_sum1_cursor = 0;
            while (ReduceSpawnIO(1, reduce_sum1_io_dims[reduce_sum1_cursor],
                                 reduce_sum1_io_dims[reduce_sum1_io_dims.size() - 1], reduce_sum1_mid_dims, true,
                                 false)) {
                reduce_sum1_io_dims.insert(reduce_sum1_io_dims.begin() + reduce_sum1_cursor + 1, reduce_sum1_mid_dims);
                reduce_sum1_cursor++;
            }

            const uint32_t reduce_sum1_ping_addr[2] = { (uint32_t) bank.sramReducePing0High,
                                                        (uint32_t) bank.sramReducePing1High };
            const uint32_t reciprocal_output_addr   = (uint32_t) bank.sramMaxLowAll;
            for (size_t i = 0; i < reduce_sum1_io_dims.size() - 1; i++) {
                RppDims        in_dims    = reduce_sum1_io_dims[i];
                RppDims        out_dims   = reduce_sum1_io_dims[i + 1];
                const uint32_t input_addr = (i == 0) ? reduce_sum1_input_addr : reduce_sum1_ping_addr[i % 2];
                const uint32_t output_addr =
                    (i == reduce_sum1_io_dims.size() - 2) ? reciprocal_output_addr : reduce_sum1_ping_addr[(i + 1) % 2];

                std::vector<RppTaskElement> tasks = create_reduce_kernel_task(
                    1, RppReduceOperation::kSUM, input_addr, output_addr, in_dims, out_dims, 0, true, false);
                if (i == reduce_sum1_io_dims.size() - 2 && tasks.size() != 1) {
                    throw std::runtime_error("Unexpected multi-task final reduce-sum stage");
                }
                for (RppTaskElement & task : tasks) {
                    if (i == reduce_sum1_io_dims.size() - 2) {
                        convert_reduce_task_to_reciprocal_f16(task, (uint32_t) reciprocal_table_addr);
                    }
                    launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList,
                                       ctx.rppBinMod, ctx.kernelStream);
                }
            }
        }

        for (int row_base = 0; row_base < tile_hq; row_base += loop2_rows_per_launch) {
            const int tile_rows =
                (((row_base + loop2_rows_per_launch) <= tile_hq) ? loop2_rows_per_launch : (tile_hq - row_base));
            const uint32_t row_offset_bytes    = (uint32_t) row_base * score_row_stride_bytes;
            const uint32_t scalar_offset_bytes = (uint32_t) row_base * (uint32_t) sizeof(uint16_t);

            RppTaskElement softmax_mul{};
            softmax_mul.setBlockDim(kv_page, tile_rows, 1);
            softmax_mul.setGridDim(1, 1, 1);
            softmax_mul.appendParam((uint32_t) bank.sramProbAll + row_offset_bytes);
            softmax_mul.appendParam((uint32_t) bank.sramMaxLowAll + scalar_offset_bytes);
            softmax_mul.appendParam((uint32_t) bank.sramProbAll + row_offset_bytes);
            softmax_mul.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
            softmax_mul.appendParam(score_row_stride_bytes * (uint32_t) tile_rows);
            softmax_mul.appendParam((uint32_t) tile_rows * (uint32_t) sizeof(uint16_t));
            softmax_mul.appendParam((uint32_t) score_row_stride_elems);
            softmax_mul.appendParam(1);
            softmax_mul.appendParam((uint32_t) kv_page_repeat);
            softmax_mul.taskName = "llama3_loop2_mid_mul_bc_x_f16";
            launchWrapperAysnc(softmax_mul.taskName, softmax_mul.gridDim, softmax_mul.blockDim,
                               softmax_mul.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
        }

        for (int local_h = 0; local_h < tile_hkv; ++local_h) {
            const VxmHeadBuffers & head = tile_heads[(size_t) local_h];
            RppTaskElement         second_mm_task{};
            second_mm_task.taskName   = "expand_matrix_batch_vxM_group_f16_dyn";
            second_mm_task.blockDim.x = second_mm_block_x;
            second_mm_task.blockDim.y = second_mm_block_y;
            second_mm_task.blockDim.z = second_mm_block_z;
            second_mm_task.gridDim.x  = second_mm_grid_x;
            second_mm_task.gridDim.y  = second_mm_grid_y;
            second_mm_task.gridDim.z  = second_mm_grid_z;
            expand_batch_vxM_group_kernel_params((uint32_t) head.sramProb, (uint32_t) head.sramV, 0,
                                                 (uint32_t) head.sramOutBf16, 1, kv_page, kv_page_repeat, D, group, 0,
                                                 second_mm_task.params.kernelList);
            second_mm_task.appendParam(0);
            second_mm_task.appendParam((uint32_t) kv_page_repeat);
            launchWrapperAysnc(second_mm_task.taskName, second_mm_task.gridDim, second_mm_task.blockDim,
                               second_mm_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
        }

        if (io_bytes_per_elem == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, (uint32_t) ((size_t) tile_hkv * out_bf16_stride), threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, (uint32_t) tile_heads[0].sramOutBf16,
                                      (uint32_t) tile_heads[0].sramOutIo, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        rtMemcpy2DAsync((void *) tile_heads[0].devOGroup, out_group_bytes, (const void *) tile_heads[0].sramOutStage,
                        out_stage_stride_bytes, out_group_bytes, tile_hkv, rtMemcpySramToDevice, ctx.kernelStream);
    };

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/flash_atten_vxm.o");
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);

    rtMemcpyAsync((void *) exp_table_addr, (const void *) dev_exp_lut, lut_elements * 2, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) reciprocal_table_addr, (const void *) dev_div_lut, lut_elements * 2, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramMaskRaw, (const void *) devMask, sizeMask, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (mask_bytes_per_elem == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, Tk, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramMaskRaw, sramMaskBf16, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    const RPPdeviceptr sramMaskStage = (mask_bytes_per_elem == sizeof(float)) ? sramMaskBf16 : sramMaskRaw;

    schedule_tile_dma(0, 0);
    for (int tile_idx = 0; tile_idx < num_hkv_tiles; ++tile_idx) {
        const int ping = tile_idx & 1;
        rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);
        if (tile_idx + 1 < num_hkv_tiles) {
            schedule_tile_dma(tile_idx + 1, (tile_idx + 1) & 1);
        }
        launch_tile(tile_idx, ping, sramMaskStage);
        rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

void rpp_flash_atten_build_vxm_v1(rpp_kernel_context & ctx,
                                  int                  Tq,
                                  int                  Tk,
                                  int                  Hq,
                                  int                  Hkv,
                                  int                  D,
                                  float                scale,
                                  int                  kv_bytes_per_elem,
                                  int                  io_bytes_per_elem,
                                  int                  mask_bytes_per_elem,
                                  int                  is_instantial = 1) {
    if (Tq != 1) {
        throw std::runtime_error("rpp_flash_atten_build_vxm_v1 only supports Tq == 1");
    }
    if (Tk > FLASH_VXM_SMALL_KV_THRESHOLD) {
        rpp_flash_atten_build_vxm(ctx, Tq, Tk, Hq, Hkv, D, scale, kv_bytes_per_elem, io_bytes_per_elem,
                                  mask_bytes_per_elem, is_instantial);
        return;
    }
    if (kv_bytes_per_elem != sizeof(rpp::bfloat16)) {
        throw std::runtime_error("Not Supportted KV Storage Type");
    }
    if (Hkv <= 0 || (Hq % Hkv) != 0) {
        throw std::runtime_error("Hq must be divisible by Hkv for VXM");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devQ    = ctx.dev_in[0];
    RPPdeviceptr          devK    = ctx.dev_in[1];
    RPPdeviceptr          devV    = ctx.dev_in[2];
    RPPdeviceptr          devMask = ctx.dev_in[3];

    const int  lut_elements = 64 * 1024;
    uint16_t * exp_table    = (uint16_t *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < lut_elements; i++) {
        uint32_t x = i;
        x <<= 16;
        float y      = std::exp(*(float *) &x);
        exp_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }

    uint16_t * div_table = (uint16_t *) malloc(lut_elements * sizeof(uint16_t));
    for (uint32_t i = 0; i < lut_elements; i++) {
        uint32_t x = i;
        x <<= 16;
        float y      = 1.0f / (*(float *) &x);
        div_table[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }
    div_table[0]      = 0;
    div_table[0x8000] = 0;

    RPPdeviceptr dev_exp_lut = ctx.dev_workspace;
    RPPdeviceptr dev_div_lut = dev_exp_lut + lut_elements * sizeof(short);

    rtMemcpy((void *) dev_exp_lut, (const void *) exp_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);
    rtMemcpy((void *) dev_div_lut, (const void *) div_table, lut_elements * sizeof(short), rtMemcpyHostToDevice);

    int hkv_tile_max = Hkv;
    if (const char * tile_env = std::getenv("GGML_FLASH_HKV_TILE_MAX")) {
        const int env_tile = std::atoi(tile_env);
        if (env_tile > 0) {
            hkv_tile_max = env_tile;
        }
    }
    if (hkv_tile_max > Hkv) {
        hkv_tile_max = Hkv;
    }
    if (hkv_tile_max < 1) {
        hkv_tile_max = 1;
    }

    if (((Hkv + hkv_tile_max - 1) / hkv_tile_max) > 1) {
        rpp_flash_atten_build_vxm_v1_tiled_hkv(ctx, Tk, Hq, Hkv, D, scale, io_bytes_per_elem, mask_bytes_per_elem, devQ,
                                               devK, devV, devMask, dev_exp_lut, dev_div_lut, lut_elements,
                                               hkv_tile_max);
        free(exp_table);
        free(div_table);
        return;
    }

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/flash_atten_vxm.o");
    // Join dmaStream into the capture graph before issuing staged DMA work.
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[0], 0);

    const int group          = Hq / Hkv;
    const int kv_page        = select_vxm_kv_page(Tk);
    const int kv_page_repeat = Tk / kv_page;

    const int sizeMask           = Tk * mask_bytes_per_elem;
    const int sizeMaskBf16       = Tk * (int) sizeof(rpp::bfloat16);
    const int sizeQGroupRaw      = group * D * io_bytes_per_elem;
    const int sizeQGroupBf16     = group * D * (int) sizeof(rpp::bfloat16);
    const int sizeK              = Tk * D * (int) sizeof(rpp::bfloat16);
    const int fused_k_width      = D * Hkv;
    const int sizeKFused         = Tk * fused_k_width * (int) sizeof(rpp::bfloat16);
    const int sizeKtFused        = Tk * (fused_k_width + 1) * (int) sizeof(rpp::bfloat16);
    const int sizeV              = Tk * D * (int) sizeof(rpp::bfloat16);
    const int sizeScore          = group * Tk * (int) sizeof(rpp::bfloat16);
    const int sizeReduce         = group * (int) sizeof(rpp::bfloat16);
    const int sizeReduceStageAll = Hq * kv_page * (int) sizeof(rpp::bfloat16);
    const int sizeOutBf16        = group * D * (int) sizeof(rpp::bfloat16);
    const int sizeOutIo          = group * D * io_bytes_per_elem;
    const int lutSize            = lut_elements * (int) sizeof(short);

    const size_t q_raw_stride =
        (io_bytes_per_elem == sizeof(float)) ? (size_t) round_up(sizeQGroupBf16) * 2 : (size_t) round_up(sizeQGroupRaw);
    const size_t q_bf16_stride      = (size_t) round_up(sizeQGroupBf16);
    const size_t k_fused_stride     = (size_t) round_up(sizeKFused);
    const size_t kt_fused_stride    = (size_t) round_up(sizeKtFused);
    const size_t v_stride           = (size_t) round_up(sizeV);
    const size_t score_slice_bytes  = (size_t) sizeScore;
    const size_t score_all_bytes    = (size_t) Hkv * score_slice_bytes;
    const size_t reduce_slice_bytes = (size_t) sizeReduce;
    const size_t reduce_all_bytes   = (size_t) Hkv * reduce_slice_bytes;
    const size_t out_bf16_stride    = (size_t) round_up(sizeOutBf16);
    const size_t out_io_stride =
        (io_bytes_per_elem == sizeof(float)) ? out_bf16_stride * 2 : (size_t) round_up(sizeOutIo);

    const size_t q_bf16_total = (io_bytes_per_elem == sizeof(float)) ? (size_t) Hkv * q_bf16_stride : 0;
    const size_t out_io_total = (io_bytes_per_elem == sizeof(float)) ? (size_t) Hkv * out_io_stride : 0;

    RPPdeviceptr sram_base             = ctx.virtual_sram_base;
    RPPdeviceptr sramMaskRaw           = sram_base;
    RPPdeviceptr sramMaskBf16          = sramMaskRaw + round_up(sizeMask);
    RPPdeviceptr sramQRawAll           = sramMaskBf16 + round_up(sizeMaskBf16);
    RPPdeviceptr sramQBf16All          = sramQRawAll + (RPPdeviceptr) ((size_t) Hkv * q_raw_stride);
    RPPdeviceptr sramKPacked           = sramQBf16All + (RPPdeviceptr) q_bf16_total;
    RPPdeviceptr sramKtPacked          = sramKPacked + (RPPdeviceptr) k_fused_stride;
    RPPdeviceptr sramKTransAll         = sramKtPacked + (RPPdeviceptr) kt_fused_stride;
    RPPdeviceptr sramVAll              = sramKTransAll + (RPPdeviceptr) k_fused_stride;
    RPPdeviceptr sramScoreHighAll      = sramVAll + (RPPdeviceptr) ((size_t) Hkv * v_stride);
    RPPdeviceptr sramScoreLowAll       = sramScoreHighAll + round_up((int) score_all_bytes);
    RPPdeviceptr sramSubAll            = sramScoreLowAll + round_up((int) score_all_bytes);
    RPPdeviceptr sramProbAll           = sramSubAll + round_up((int) score_all_bytes);
    RPPdeviceptr sramMaxHighAll        = sramProbAll + round_up((int) score_all_bytes);
    RPPdeviceptr sramMaxLowAll         = sramMaxHighAll + round_up((int) reduce_all_bytes);
    RPPdeviceptr sramReducePing0High   = sramMaxLowAll + round_up((int) reduce_all_bytes);
    RPPdeviceptr sramReducePing0Low    = sramReducePing0High + round_up(sizeReduceStageAll);
    RPPdeviceptr sramReducePing1High   = sramReducePing0Low + round_up(sizeReduceStageAll);
    RPPdeviceptr sramReducePing1Low    = sramReducePing1High + round_up(sizeReduceStageAll);
    RPPdeviceptr sramOutBf16All        = sramReducePing1Low + round_up(sizeReduceStageAll);
    RPPdeviceptr sramOutIoAll          = sramOutBf16All + (RPPdeviceptr) ((size_t) Hkv * out_bf16_stride);
    RPPdeviceptr exp_table_addr        = sramOutIoAll + (RPPdeviceptr) out_io_total;
    RPPdeviceptr reciprocal_table_addr = exp_table_addr + round_up(lutSize);

    const int total_sram_bytes = (int) (reciprocal_table_addr + round_up(lut_elements * 2) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "VXM v1 SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                  << " bytes\n";
        std::abort();
    }

    std::vector<VxmHeadBuffers> heads(Hkv);
    const size_t                q_row_bytes     = (size_t) D * (size_t) io_bytes_per_elem;
    const size_t                q_group_bytes   = (size_t) sizeQGroupRaw;
    const size_t                kv_row_bytes    = (size_t) D * sizeof(rpp::bfloat16);
    const size_t                kv_src_pitch    = kv_row_bytes * (size_t) Hkv;
    const size_t                out_row_bytes   = (size_t) D * (size_t) io_bytes_per_elem;
    const size_t                out_group_bytes = (size_t) sizeOutIo;
    const size_t                out_stage_stride_bytes =
        (size_t) ((io_bytes_per_elem == sizeof(float)) ? out_io_stride : out_bf16_stride);

    for (int kv_h = 0; kv_h < Hkv; kv_h++) {
        VxmHeadBuffers & head         = heads[kv_h];
        const int        q_group_head = kv_h * group;
        head.devQGroup                = devQ + (size_t) q_group_head * q_row_bytes;
        head.devKHead                 = devK + (size_t) kv_h * kv_row_bytes;
        head.devVHead                 = devV + (size_t) kv_h * kv_row_bytes;
        head.devOGroup                = ctx.dev_out[0] + (size_t) q_group_head * out_row_bytes;
        head.sramQRaw                 = sramQRawAll + (RPPdeviceptr) ((size_t) kv_h * q_raw_stride);
        head.sramQBf16                = (io_bytes_per_elem == sizeof(float)) ?
                                            sramQBf16All + (RPPdeviceptr) ((size_t) kv_h * q_bf16_stride) :
                                            head.sramQRaw;
        head.sramQStage               = (io_bytes_per_elem == sizeof(float)) ? head.sramQBf16 : head.sramQRaw;
        head.sramK                    = sramKTransAll + (RPPdeviceptr) ((size_t) kv_h * (size_t) sizeK);
        head.sramKt                   = sramKtPacked;
        head.sramV                    = sramVAll + (RPPdeviceptr) ((size_t) kv_h * v_stride);
        head.sramScoreHigh            = sramScoreHighAll + (RPPdeviceptr) ((size_t) kv_h * score_slice_bytes);
        head.sramScoreLow             = sramScoreLowAll + (RPPdeviceptr) ((size_t) kv_h * score_slice_bytes);
        head.sramSub                  = sramSubAll + (RPPdeviceptr) ((size_t) kv_h * score_slice_bytes);
        head.sramProb                 = sramProbAll + (RPPdeviceptr) ((size_t) kv_h * score_slice_bytes);
        head.sramMaxHigh              = sramMaxHighAll + (RPPdeviceptr) ((size_t) kv_h * reduce_slice_bytes);
        head.sramMaxLow               = sramMaxLowAll + (RPPdeviceptr) ((size_t) kv_h * reduce_slice_bytes);
        head.sramOutBf16              = sramOutBf16All + (RPPdeviceptr) ((size_t) kv_h * out_bf16_stride);
        head.sramOutIo                = (io_bytes_per_elem == sizeof(float)) ?
                                            sramOutIoAll + (RPPdeviceptr) ((size_t) kv_h * out_io_stride) :
                                            head.sramOutBf16;
        head.sramOutStage             = (io_bytes_per_elem == sizeof(float)) ? head.sramOutIo : head.sramOutBf16;
    }

    const RPPdeviceptr sramMaskStage = (mask_bytes_per_elem == sizeof(float)) ? sramMaskBf16 : sramMaskRaw;

    // Stage 1: K DMA.
    rtMemcpy2DAsync((void *) sramKPacked, kv_src_pitch, (const void *) devK, kv_src_pitch, kv_src_pitch, Tk,
                    rtMemcpyDeviceToSram, ctx.dmaStream);
    rppEventRecord(ctx.dma_done_ping[0], ctx.dmaStream);

    // Stage 2: once K is in SRAM, transpose it while Q DMA runs behind it.
    // Q head-groups are contiguous in device memory, so one 2D DMA is enough even
    // though SRAM keeps a padded stride between head-groups.
    rtMemcpy2DAsync((void *) sramQRawAll, q_raw_stride, (const void *) devQ, q_group_bytes, q_group_bytes, Hkv,
                    rtMemcpyDeviceToSram, ctx.dmaStream);
    rppEventRecord(ctx.dma_done_ping[1], ctx.dmaStream);

    // Stage 3: LUT/mask become live after the first MM starts.
    rtMemcpyAsync((void *) exp_table_addr, (const void *) dev_exp_lut, lut_elements * 2, rtMemcpyDeviceToSram,
                  ctx.dmaStream);
    rtMemcpyAsync((void *) reciprocal_table_addr, (const void *) dev_div_lut, lut_elements * 2, rtMemcpyDeviceToSram,
                  ctx.dmaStream);
    rtMemcpyAsync((void *) sramMaskRaw, (const void *) devMask, sizeMask, rtMemcpyDeviceToSram, ctx.dmaStream);
    rppEventRecord(ctx.kernel_done_ping[0], ctx.dmaStream);

    // Stage 4: V is only consumed by the second MM, so copy it while softmax/reduce runs.
    for (int kv_h = 0; kv_h < Hkv; kv_h++) {
        const VxmHeadBuffers & head = heads[kv_h];
        rtMemcpy2DAsync((void *) head.sramV, kv_row_bytes, (const void *) head.devVHead, kv_src_pitch, kv_row_bytes, Tk,
                        rtMemcpyDeviceToSram, ctx.dmaStream);
    }
    rppEventRecord(ctx.kernel_done_ping[1], ctx.dmaStream);

    rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[0], 0);

    {
        params.clear();
        KShape_TransPadding(Tk, fused_k_width, threadsPerBlock, blocksPerGrid);
        KParam_TransPadding(threadsPerBlock, blocksPerGrid, sramKPacked, sramKtPacked, fused_k_width, params);
        launchWrapperAysnc("blas_trans_padding", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    {
        params.clear();
        KShape_Trans(Tk, fused_k_width, threadsPerBlock, blocksPerGrid);
        KParam_Trans(threadsPerBlock, blocksPerGrid, sramKtPacked, sramKTransAll, Tk, fused_k_width, params);
        launchWrapperAysnc("blas_transpose_u16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[1], 0);

    if (io_bytes_per_elem == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, (uint32_t) ((size_t) Hkv * q_bf16_stride / sizeof(uint16_t)), threadsPerBlock,
                            blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, (uint32_t) heads[0].sramQRaw, (uint32_t) heads[0].sramQBf16, kFLOAT,
                              kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    {
        const uint32_t q_stage_stride_bytes =
            (uint32_t) ((io_bytes_per_elem == sizeof(float)) ? q_bf16_stride : q_raw_stride);
        const uint32_t q_stage_total_bytes    = (uint32_t) ((size_t) Hkv * (size_t) q_stage_stride_bytes);
        const uint32_t q_stage_total_elements = q_stage_total_bytes / (uint32_t) sizeof(uint16_t);
        calc_scale_flat_launch(q_stage_total_elements, threadsPerBlock, blocksPerGrid);
        params.clear();
        KParam_Scale(threadsPerBlock, blocksPerGrid, (uint32_t) heads[0].sramQStage, (uint32_t) heads[0].sramQStage,
                     scale, params);
        launchWrapperAysnc("opt_scale_uniform_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    const uint32_t score_row_stride_bytes = (uint32_t) Tk * (uint32_t) sizeof(uint16_t);
    const int      score_row_stride_elems = Tk;
    uint32_t       mm_block_x = 1, mm_block_y = 1, mm_block_z = 1;
    uint32_t       mm_grid_x = 1, mm_grid_y = 1, mm_grid_z = 1;
    expand_kerneldim_calc_batch_vxM(1, D, Tk, kv_page, group, mm_block_x, mm_block_y, mm_block_z, mm_grid_x, mm_grid_y,
                                    mm_grid_z);
    const std::string mm_kernel_name    = get_expand_batch_vxM_kernel(group);
    //jsmoe
    bool              use_hkv_mm_kernel = true;
    uint32_t          hkv_mm_block_x = 1, hkv_mm_block_y = 1, hkv_mm_block_z = 1;
    uint32_t          hkv_mm_grid_x = 1, hkv_mm_grid_y = 1, hkv_mm_grid_z = 1;
    if (use_hkv_mm_kernel) {
        expand_kerneldim_calc_batch_hkv_vxM(D, Tk, group, hkv_mm_block_x, hkv_mm_block_y, hkv_mm_block_z, hkv_mm_grid_x,
                                            hkv_mm_grid_y, hkv_mm_grid_z);
        use_hkv_mm_kernel = true;
    }

    for (int kv_h = 0; kv_h < Hkv; kv_h++) {
        const VxmHeadBuffers & head = heads[kv_h];
        RppTaskElement         mm_task{};
        if (use_hkv_mm_kernel) {
            mm_task.blockDim.x = hkv_mm_block_x;
            mm_task.blockDim.y = hkv_mm_block_y;
            mm_task.blockDim.z = hkv_mm_block_z;
            mm_task.gridDim.x  = hkv_mm_grid_x;
            mm_task.gridDim.y  = hkv_mm_grid_y;
            mm_task.gridDim.z  = hkv_mm_grid_z;
            expand_batch_hkv_vxM_kernel_params((uint32_t) head.sramQStage, (uint32_t) head.sramK,
                                               (uint32_t) head.sramScoreHigh, (uint32_t) head.sramScoreLow, D, Tk,
                                               hkv_mm_block_x, mm_task.params.kernelList);
            mm_task.taskName = get_expand_batch_hkv_vxM_kernel();
        } else {
            mm_task.blockDim.x = mm_block_x;
            mm_task.blockDim.y = mm_block_y;
            mm_task.blockDim.z = mm_block_z;
            mm_task.gridDim.x  = mm_grid_x;
            mm_task.gridDim.y  = mm_grid_y;
            mm_task.gridDim.z  = mm_grid_z;
            expand_batch_vxM_kernel_params((uint32_t) head.sramQStage, (uint32_t) head.sramK, 0,
                                           (uint32_t) head.sramScoreHigh, 1, D, Tk, kv_page, group, 0,
                                           mm_task.params.kernelList);
            mm_task.taskName = mm_kernel_name;
            mm_task.params.kernelList.emplace_back((uint32_t) head.sramScoreLow);
        }
        launchWrapperAysnc(mm_task.taskName, mm_task.gridDim, mm_task.blockDim, mm_task.params.kernelList,
                           ctx.rppBinMod, ctx.kernelStream);
    }

    rppStreamWaitEvent(ctx.kernelStream, ctx.kernel_done_ping[0], 0);
    if (mask_bytes_per_elem == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, Tk, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramMaskRaw, sramMaskBf16, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    const int reduce_block_x     = kv_page;
    // Stage-0 reduce kernels share one block-loop stride for input and output,
    // so row fusion is done with explicit launches instead of blockRepeat > 1.
    int       reduce_stage0_rows = 8191 / reduce_block_x;
    if (reduce_stage0_rows < 1) {
        reduce_stage0_rows = 1;
    }
    if (reduce_stage0_rows > Hq) {
        reduce_stage0_rows = Hq;
    }

    // The fused loop2-mid kernels become numerically unstable on some shapes
    // when using block_repeat > 1. Keep the launches explicit over row tiles
    // and always run the kernels with block_repeat == 1.
    int loop2_rows_per_launch = 8191 / kv_page;
    if (loop2_rows_per_launch < 1) {
        loop2_rows_per_launch = 1;
    }
    if (loop2_rows_per_launch > 255) {
        loop2_rows_per_launch = 255;
    }
    if (loop2_rows_per_launch > Hq) {
        loop2_rows_per_launch = Hq;
    }

    for (int row_base = 0; row_base < Hq; row_base += loop2_rows_per_launch) {
        const int tile_rows = (((row_base + loop2_rows_per_launch) <= Hq) ? loop2_rows_per_launch : (Hq - row_base));
        const uint32_t row_offset_bytes = (uint32_t) row_base * score_row_stride_bytes;

        RppTaskElement add_task{};
        add_task.setBlockDim(kv_page, tile_rows, 1);
        add_task.setGridDim(1, 1, 1);
        loop2_mid_add_bc_y_f32_f16_f32_params(
            (uint32_t) sramScoreHighAll + row_offset_bytes, (uint32_t) sramScoreLowAll + row_offset_bytes,
            (uint32_t) sramMaskStage, (uint32_t) sramScoreHighAll + row_offset_bytes,
            (uint32_t) sramScoreLowAll + row_offset_bytes, kv_page * sizeof(uint16_t),
            score_row_stride_bytes * (uint32_t) tile_rows, score_row_stride_elems, kv_page * sizeof(uint16_t), 1,
            (uint32_t) kv_page_repeat, add_task.params.kernelList);
        add_task.taskName = "llama3_loop2_mid_add_bc_y_f32_f16_f32";
        launchWrapperAysnc(add_task.taskName, add_task.gridDim, add_task.blockDim, add_task.params.kernelList,
                           ctx.rppBinMod, ctx.kernelStream);
    }

    {
        RppDims reduce1_in_dims{}, reduce1_out_dims{};
        reduce1_in_dims.nbDims  = 2;
        reduce1_in_dims.d[0]    = Hq;
        reduce1_out_dims.nbDims = 2;
        reduce1_out_dims.d[0]   = Hq;
        reduce1_out_dims.d[1]   = 1;

        uint32_t reduce1_input_high_addr = (uint32_t) sramScoreHighAll;
        uint32_t reduce1_input_low_addr  = (uint32_t) sramScoreLowAll;
        if (kv_page_repeat > 1) {
            const uint32_t reduce_stage0_output_row_bytes = (uint32_t) kv_page * (uint32_t) sizeof(uint16_t);
            for (int row_base = 0; row_base < Hq; row_base += reduce_stage0_rows) {
                const int tile_rows = (((row_base + reduce_stage0_rows) <= Hq) ? reduce_stage0_rows : (Hq - row_base));
                const uint32_t input_row_offset  = (uint32_t) row_base * score_row_stride_bytes;
                const uint32_t output_row_offset = (uint32_t) row_base * reduce_stage0_output_row_bytes;

                RppTaskElement reduce1_stage0{};
                reduce1_stage0.setBlockDim(reduce_block_x, tile_rows, 1);
                reduce1_stage0.setGridDim(1, 1, 1);
                loop2_mid_reduce_max_f32_params(
                    (uint32_t) sramScoreHighAll + input_row_offset, (uint32_t) sramScoreLowAll + input_row_offset,
                    (uint32_t) sramReducePing0High + output_row_offset,
                    (uint32_t) sramReducePing0Low + output_row_offset, kv_page * sizeof(uint16_t),
                    score_row_stride_bytes * (uint32_t) tile_rows, (uint32_t) score_row_stride_elems,
                    (uint32_t) kv_page, 1, (uint32_t) kv_page_repeat, reduce1_stage0.params.kernelList);
                reduce1_stage0.taskName = "llama3_loop2_mid_reduce_max_f32";
                launchWrapperAysnc(reduce1_stage0.taskName, reduce1_stage0.gridDim, reduce1_stage0.blockDim,
                                   reduce1_stage0.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
            }

            reduce1_in_dims.d[1]    = kv_page;
            reduce1_input_high_addr = (uint32_t) sramReducePing0High;
            reduce1_input_low_addr  = (uint32_t) sramReducePing0Low;
        } else {
            reduce1_in_dims.d[1] = Tk;
        }

        std::vector<RppDims> reduce1_io_dims;
        reduce1_io_dims.emplace_back(reduce1_in_dims);
        reduce1_io_dims.emplace_back(reduce1_out_dims);
        RppDims reduce1_mid_dims{};
        int     reduce1_cursor = 0;
        while (ReduceSpawnIO(1, reduce1_io_dims[reduce1_cursor], reduce1_io_dims[reduce1_io_dims.size() - 1],
                             reduce1_mid_dims, true, false)) {
            reduce1_io_dims.insert(reduce1_io_dims.begin() + reduce1_cursor + 1, reduce1_mid_dims);
            reduce1_cursor++;
        }

        const uint32_t reduce1_ping_high_addr[2] = { (uint32_t) sramReducePing0High, (uint32_t) sramReducePing1High };
        const uint32_t reduce1_ping_low_addr[2]  = { (uint32_t) sramReducePing0Low, (uint32_t) sramReducePing1Low };

        for (size_t i = 0; i < reduce1_io_dims.size() - 1; i++) {
            RppDims        in_dims         = reduce1_io_dims[i];
            RppDims        out_dims        = reduce1_io_dims[i + 1];
            const uint32_t input_high_addr = (i == 0) ? reduce1_input_high_addr : reduce1_ping_high_addr[i % 2];
            const uint32_t input_low_addr  = (i == 0) ? reduce1_input_low_addr : reduce1_ping_low_addr[i % 2];
            const uint32_t output_high_addr =
                (i == reduce1_io_dims.size() - 2) ? (uint32_t) sramMaxHighAll : reduce1_ping_high_addr[(i + 1) % 2];
            const uint32_t output_low_addr =
                (i == reduce1_io_dims.size() - 2) ? (uint32_t) sramMaxLowAll : reduce1_ping_low_addr[(i + 1) % 2];

            std::vector<RppTaskElement> tasks = create_reduce_kernel_task(
                1, RppReduceOperation::kMAX, input_high_addr, output_high_addr, in_dims, out_dims, 0, true, false);
            for (RppTaskElement & task : tasks) {
                convert_reduce_task_to_split_f32(task, input_high_addr, input_low_addr, output_high_addr,
                                                 output_low_addr);
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }
    }

    for (int row_base = 0; row_base < Hq; row_base += loop2_rows_per_launch) {
        const int tile_rows = (((row_base + loop2_rows_per_launch) <= Hq) ? loop2_rows_per_launch : (Hq - row_base));
        const uint32_t row_offset_bytes    = (uint32_t) row_base * score_row_stride_bytes;
        const uint32_t scalar_offset_bytes = (uint32_t) row_base * (uint32_t) sizeof(uint16_t);

        RppTaskElement softmax_norm_sub{};
        softmax_norm_sub.setBlockDim(kv_page, tile_rows, 1);
        softmax_norm_sub.setGridDim(1, 1, 1);
        softmax_norm_sub.appendParam((uint32_t) sramScoreHighAll + row_offset_bytes);
        softmax_norm_sub.appendParam((uint32_t) sramScoreLowAll + row_offset_bytes);
        softmax_norm_sub.appendParam((uint32_t) sramMaxHighAll + scalar_offset_bytes);
        softmax_norm_sub.appendParam((uint32_t) sramMaxLowAll + scalar_offset_bytes);
        softmax_norm_sub.appendParam((uint32_t) sramSubAll + row_offset_bytes);
        softmax_norm_sub.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
        softmax_norm_sub.appendParam(score_row_stride_bytes * (uint32_t) tile_rows);
        softmax_norm_sub.appendParam((uint32_t) tile_rows * (uint32_t) sizeof(uint16_t));
        softmax_norm_sub.appendParam((uint32_t) score_row_stride_elems);
        softmax_norm_sub.appendParam(1);
        softmax_norm_sub.appendParam((uint32_t) kv_page_repeat);
        softmax_norm_sub.taskName = "llama3_loop2_mid_sub_bc_x_f32_f16";
        launchWrapperAysnc(softmax_norm_sub.taskName, softmax_norm_sub.gridDim, softmax_norm_sub.blockDim,
                           softmax_norm_sub.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
    }

    for (int row_base = 0; row_base < Hq; row_base += loop2_rows_per_launch) {
        const int tile_rows = (((row_base + loop2_rows_per_launch) <= Hq) ? loop2_rows_per_launch : (Hq - row_base));
        const uint32_t row_offset_bytes = (uint32_t) row_base * score_row_stride_bytes;

        RppTaskElement softmax_exp{};
        softmax_exp.setBlockDim(kv_page, tile_rows, 1);
        softmax_exp.setGridDim(1, 1, 1);
        softmax_exp.appendParam((uint32_t) sramSubAll + row_offset_bytes);
        softmax_exp.appendParam((uint32_t) exp_table_addr);
        softmax_exp.appendParam((uint32_t) sramProbAll + row_offset_bytes);
        softmax_exp.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
        softmax_exp.appendParam(score_row_stride_bytes * (uint32_t) tile_rows);
        softmax_exp.appendParam((uint32_t) score_row_stride_elems);
        softmax_exp.appendParam(1);
        softmax_exp.appendParam((uint32_t) kv_page_repeat);
        softmax_exp.taskName = "llama3_loop2_mid_mish_f16";
        launchWrapperAysnc(softmax_exp.taskName, softmax_exp.gridDim, softmax_exp.blockDim,
                           softmax_exp.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
    }

    {
        RppDims reduce_sum1_in_dims{}, reduce_sum1_out_dims{};
        reduce_sum1_in_dims.nbDims  = 2;
        reduce_sum1_in_dims.d[0]    = Hq;
        reduce_sum1_out_dims.nbDims = 2;
        reduce_sum1_out_dims.d[0]   = Hq;
        reduce_sum1_out_dims.d[1]   = 1;

        uint32_t reduce_sum1_input_addr = (uint32_t) sramProbAll;
        if (kv_page_repeat > 1) {
            const uint32_t reduce_stage0_output_row_bytes = (uint32_t) kv_page * (uint32_t) sizeof(uint16_t);
            for (int row_base = 0; row_base < Hq; row_base += reduce_stage0_rows) {
                const int tile_rows = (((row_base + reduce_stage0_rows) <= Hq) ? reduce_stage0_rows : (Hq - row_base));
                const uint32_t input_row_offset  = (uint32_t) row_base * score_row_stride_bytes;
                const uint32_t output_row_offset = (uint32_t) row_base * reduce_stage0_output_row_bytes;

                RppTaskElement reduce_sum1_stage0{};
                reduce_sum1_stage0.setBlockDim(reduce_block_x, tile_rows, 1);
                reduce_sum1_stage0.setGridDim(1, 1, 1);
                reduce_sum1_stage0.appendParam((uint32_t) sramProbAll + input_row_offset);
                reduce_sum1_stage0.appendParam((uint32_t) sramReducePing0High + output_row_offset);
                reduce_sum1_stage0.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
                reduce_sum1_stage0.appendParam(score_row_stride_bytes * (uint32_t) tile_rows);
                reduce_sum1_stage0.appendParam((uint32_t) score_row_stride_elems);
                reduce_sum1_stage0.appendParam((uint32_t) kv_page);
                reduce_sum1_stage0.appendParam(1);
                reduce_sum1_stage0.appendParam((uint32_t) kv_page_repeat);
                reduce_sum1_stage0.taskName = "llama3_loop2_mid_reduce_sum_f16";
                launchWrapperAysnc(reduce_sum1_stage0.taskName, reduce_sum1_stage0.gridDim, reduce_sum1_stage0.blockDim,
                                   reduce_sum1_stage0.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
            }

            reduce_sum1_in_dims.d[1] = kv_page;
            reduce_sum1_input_addr   = (uint32_t) sramReducePing0High;
        } else {
            reduce_sum1_in_dims.d[1] = Tk;
        }

        std::vector<RppDims> reduce_sum1_io_dims;
        reduce_sum1_io_dims.emplace_back(reduce_sum1_in_dims);
        reduce_sum1_io_dims.emplace_back(reduce_sum1_out_dims);
        RppDims reduce_sum1_mid_dims{};
        int     reduce_sum1_cursor = 0;
        while (ReduceSpawnIO(1, reduce_sum1_io_dims[reduce_sum1_cursor],
                             reduce_sum1_io_dims[reduce_sum1_io_dims.size() - 1], reduce_sum1_mid_dims, true, false)) {
            reduce_sum1_io_dims.insert(reduce_sum1_io_dims.begin() + reduce_sum1_cursor + 1, reduce_sum1_mid_dims);
            reduce_sum1_cursor++;
        }

        const uint32_t reduce_sum1_ping_addr[2] = { (uint32_t) sramReducePing0High, (uint32_t) sramReducePing1High };
        const uint32_t reciprocal_output_addr   = (uint32_t) sramMaxLowAll;
        for (size_t i = 0; i < reduce_sum1_io_dims.size() - 1; i++) {
            RppDims        in_dims    = reduce_sum1_io_dims[i];
            RppDims        out_dims   = reduce_sum1_io_dims[i + 1];
            const uint32_t input_addr = (i == 0) ? reduce_sum1_input_addr : reduce_sum1_ping_addr[i % 2];
            const uint32_t output_addr =
                (i == reduce_sum1_io_dims.size() - 2) ? reciprocal_output_addr : reduce_sum1_ping_addr[(i + 1) % 2];

            std::vector<RppTaskElement> tasks = create_reduce_kernel_task(
                1, RppReduceOperation::kSUM, input_addr, output_addr, in_dims, out_dims, 0, true, false);
            if (i == reduce_sum1_io_dims.size() - 2 && tasks.size() != 1) {
                throw std::runtime_error("Unexpected multi-task final reduce-sum stage");
            }
            for (RppTaskElement & task : tasks) {
                if (i == reduce_sum1_io_dims.size() - 2) {
                    convert_reduce_task_to_reciprocal_f16(task, (uint32_t) reciprocal_table_addr);
                }
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }
    }

    for (int row_base = 0; row_base < Hq; row_base += loop2_rows_per_launch) {
        const int tile_rows = (((row_base + loop2_rows_per_launch) <= Hq) ? loop2_rows_per_launch : (Hq - row_base));
        const uint32_t row_offset_bytes    = (uint32_t) row_base * score_row_stride_bytes;
        const uint32_t scalar_offset_bytes = (uint32_t) row_base * (uint32_t) sizeof(uint16_t);

        RppTaskElement softmax_mul{};
        softmax_mul.setBlockDim(kv_page, tile_rows, 1);
        softmax_mul.setGridDim(1, 1, 1);
        softmax_mul.appendParam((uint32_t) sramProbAll + row_offset_bytes);
        softmax_mul.appendParam((uint32_t) sramMaxLowAll + scalar_offset_bytes);
        softmax_mul.appendParam((uint32_t) sramProbAll + row_offset_bytes);
        softmax_mul.appendParam(kv_page * (uint32_t) sizeof(uint16_t));
        softmax_mul.appendParam(score_row_stride_bytes * (uint32_t) tile_rows);
        softmax_mul.appendParam((uint32_t) tile_rows * (uint32_t) sizeof(uint16_t));
        softmax_mul.appendParam((uint32_t) score_row_stride_elems);
        softmax_mul.appendParam(1);
        softmax_mul.appendParam((uint32_t) kv_page_repeat);
        softmax_mul.taskName = "llama3_loop2_mid_mul_bc_x_f16";
        launchWrapperAysnc(softmax_mul.taskName, softmax_mul.gridDim, softmax_mul.blockDim,
                           softmax_mul.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
    }

    uint32_t second_mm_block_x = 1, second_mm_block_y = 1, second_mm_block_z = 1;
    uint32_t second_mm_grid_x = 1, second_mm_grid_y = 1, second_mm_grid_z = 1;
    expand_kerneldim_calc_batch_vxM_group(1, kv_page, D, group, second_mm_block_x, second_mm_block_y, second_mm_block_z,
                                          second_mm_grid_x, second_mm_grid_y, second_mm_grid_z);

    rppStreamWaitEvent(ctx.kernelStream, ctx.kernel_done_ping[1], 0);
    for (int kv_h = 0; kv_h < Hkv; kv_h++) {
        const VxmHeadBuffers & head = heads[kv_h];
        RppTaskElement         second_mm_task{};
        second_mm_task.taskName   = "expand_matrix_batch_vxM_group_f16_dyn";
        second_mm_task.blockDim.x = second_mm_block_x;
        second_mm_task.blockDim.y = second_mm_block_y;
        second_mm_task.blockDim.z = second_mm_block_z;
        second_mm_task.gridDim.x  = second_mm_grid_x;
        second_mm_task.gridDim.y  = second_mm_grid_y;
        second_mm_task.gridDim.z  = second_mm_grid_z;
        expand_batch_vxM_group_kernel_params((uint32_t) head.sramProb, (uint32_t) head.sramV, 0,
                                             (uint32_t) head.sramOutBf16, 1, kv_page, kv_page_repeat, D, group, 0,
                                             second_mm_task.params.kernelList);
        second_mm_task.appendParam(0);
        second_mm_task.appendParam((uint32_t) kv_page_repeat);
        launchWrapperAysnc(second_mm_task.taskName, second_mm_task.gridDim, second_mm_task.blockDim,
                           second_mm_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
    }

    if (io_bytes_per_elem == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, (uint32_t) ((size_t) Hkv * out_bf16_stride), threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, (uint32_t) heads[0].sramOutBf16, (uint32_t) heads[0].sramOutIo,
                                  kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    rtMemcpy2DAsync((void *) heads[0].devOGroup, out_group_bytes, (const void *) heads[0].sramOutStage,
                    out_stage_stride_bytes, out_group_bytes, Hkv, rtMemcpySramToDevice, ctx.kernelStream);

    free(exp_table);
    free(div_table);
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void zero_fill_sram_16bit_aligned(RPPdeviceptr            addr,
                                         int                     bytes,
                                         dim3 &                  threadsPerBlock,
                                         dim3 &                  blocksPerGrid,
                                         std::vector<uint32_t> & params,
                                         RPPmodule &             module,
                                         rtStream_t              stream) {
    if (bytes <= 0) {
        return;
    }

    const int count16 = bytes / (int) sizeof(uint16_t);
    if (count16 <= 0) {
        return;
    }

    params.clear();
    calc_tbdim_flattern(1, count16, threadsPerBlock, blocksPerGrid);
    fill_16bits_align_params((int) addr, (int) threadsPerBlock.x, 0, (int) sizeof(uint16_t), params);
    launchWrapperAysnc("fill_16bits_align", blocksPerGrid, threadsPerBlock, params, module, stream);
}

void rpp_flash_atten_build(rpp_kernel_context & ctx,
                           int                  Tq,
                           int                  Tk,
                           int                  Hq,
                           int                  Hkv,
                           int                  D,
                           float                scale,
                           int                  kv_bytes_per_elem,
                           int                  io_bytes_per_elem,
                           int                  mask_bytes_per_elem,
                           int                  is_instantial = 1) {
    if (Tq == 1) {
        if (Tk <= FLASH_VXM_SMALL_KV_THRESHOLD) {
            rpp_flash_atten_build_vxm_v1(ctx, Tq, Tk, Hq, Hkv, D, scale, kv_bytes_per_elem, io_bytes_per_elem,
                                         mask_bytes_per_elem, is_instantial);
            return;
        }

        rpp_flash_atten_build_vxm(ctx, Tq, Tk, Hq, Hkv, D, scale, kv_bytes_per_elem, io_bytes_per_elem,
                                  mask_bytes_per_elem, is_instantial);
        return;
    }

    rpp_flash_atten_build_impl(ctx, Tq, Tk, Hq, Hkv, D, scale, kv_bytes_per_elem, io_bytes_per_elem,
                               mask_bytes_per_elem, is_instantial);
}
