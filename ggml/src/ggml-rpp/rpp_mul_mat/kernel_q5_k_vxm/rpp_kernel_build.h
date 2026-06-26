// rpp_matmul_q5k_vxm.cpp (v1 overlap: dmaStream + kernelStream + pingpong SRAM)

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q5_k_vxm/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q5_k_vxm/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_q5_k_vxm {

inline void copy_output_segments_async(RPPdeviceptr dst_base,
                                       RPPdeviceptr src_base,
                                       int          nr_of_ns,
                                       int          Ns,
                                       int          NsTailValid,
                                       int          bytes_per_element,
                                       rtMemcpyKind kind,
                                       RPPstream    stream) {
    const size_t prefix_bytes = (size_t) (nr_of_ns - 1) * (size_t) Ns * (size_t) bytes_per_element;
    const size_t tail_bytes   = (size_t) NsTailValid * (size_t) bytes_per_element;

    if (prefix_bytes > 0) {
        rtMemcpyAsync((void *) dst_base, (const void *) src_base, prefix_bytes, kind, stream);
    }
    if (tail_bytes > 0) {
        rtMemcpyAsync((void *) (dst_base + prefix_bytes), (const void *) (src_base + prefix_bytes), tail_bytes, kind,
                      stream);
    }
}

inline void memcpy2d_chunked_async(RPPdeviceptr dst_base,
                                   int          dst_pitch,
                                   RPPdeviceptr src_base,
                                   int          src_pitch,
                                   int          width_bytes,
                                   int          height,
                                   rtMemcpyKind kind,
                                   RPPstream    stream,
                                   int          min_chunk_bytes    = 32 * 1024,
                                   int          target_chunk_bytes = 256 * 1024,
                                   int          max_chunk_rows     = 0) {

    if (height <= 1 || width_bytes <= 0) {
        rtMemcpy2DAsync((void *) dst_base, dst_pitch, (const void *) src_base, src_pitch, width_bytes, height, kind,
                        stream);
        return;
    }

    int min_rows = (min_chunk_bytes + width_bytes - 1) / width_bytes;
    if (min_rows < 1) {
        min_rows = 1;
    }

    int chunk_rows = target_chunk_bytes / width_bytes;
    if (chunk_rows < min_rows) {
        chunk_rows = min_rows;
    }
    if (max_chunk_rows > 0 && chunk_rows > max_chunk_rows) {
        chunk_rows = max_chunk_rows;
    }

    if (chunk_rows >= height) {
        rtMemcpy2DAsync((void *) dst_base, dst_pitch, (const void *) src_base, src_pitch, width_bytes, height, kind,
                        stream);
        return;
    }

    for (int row = 0; row < height; row += chunk_rows) {
        const int rows = ((row + chunk_rows) <= height) ? chunk_rows : (height - row);
        rtMemcpy2DAsync((void *) (dst_base + (size_t) row * (size_t) dst_pitch), dst_pitch,
                        (const void *) (src_base + (size_t) row * (size_t) src_pitch), src_pitch, width_bytes, rows,
                        kind, stream);
    }
}

inline void get_tiles_info(int   N,
                           int   K,
                           int   weights_group,
                           int & nr_of_tiles,
                           int & groups_per_tile,
                           int & nr_of_ns,
                           int & Ns,
                           int & NsTail) {
    if (weights_group == 32) {
        if (K % 512 == 0) {
            nr_of_tiles     = K / 512;
            groups_per_tile = 16;
        } else if (K % 256 == 0) {
            nr_of_tiles     = K / 256;
            groups_per_tile = 8;
        } else if (K % 128 == 0) {
            nr_of_tiles     = K / 128;
            groups_per_tile = 4;
        } else if (K % 64 == 0) {
            nr_of_tiles     = K / 64;
            groups_per_tile = 2;
        } else {
            nr_of_tiles     = K / 32;
            groups_per_tile = 1;
        }

        if (N >= 32768 * 4) {
            nr_of_tiles     = nr_of_tiles * 8;
            groups_per_tile = groups_per_tile / 8;
        } else if (N >= 32768 * 2) {
            nr_of_tiles     = nr_of_tiles * 4;
            groups_per_tile = groups_per_tile / 4;
        } else if (N >= 32768) {
            nr_of_tiles     = nr_of_tiles * 2;
            groups_per_tile = groups_per_tile / 2;
        }
    } else {
        // jsq4k
        nr_of_tiles     = K / weights_group;
        groups_per_tile = 1;
        // nr_of_tiles = 1;
        // groups_per_tile = K / weights_group;
    }
    // jsq6k
    if (N * weights_group * groups_per_tile > (16 * 1024 * 1024)) {
        Ns       = N / 4;
        NsTail   = Ns;
        nr_of_ns = 4;
    } else if (N * weights_group * groups_per_tile > (8 * 1024 * 1024)) {
        Ns       = N / 2;
        NsTail   = Ns;
        nr_of_ns = 2;
    } else {
        Ns       = N;
        NsTail   = Ns;
        nr_of_ns = 1;
    }

    // special handle for 151936 to make sure 32bytes alignment
    if (N == 151936) {
        Ns       = 600 * 32;
        nr_of_ns = 8;
        NsTail   = 548 * 32;
    } else if (N == 200064) {
        Ns       = 640 * 32;
        nr_of_ns = 10;
        NsTail   = 492 * 32;
        // Ns = 600 * 32;
        // nr_of_ns = 11;
        // NsTail = 252 * 32;
        // Ns = 600 * 32;
        // nr_of_ns = 1;
        // NsTail = 600 * 32;
    }

    // Workaround: keep each Ns chunk <= 4096 to avoid super-scale boundary issue.
    const int NS_SAFE = 4096;
    if (Ns > NS_SAFE) {
        nr_of_ns = (N + NS_SAFE - 1) / NS_SAFE;
        Ns       = NS_SAFE;
        NsTail   = N - (nr_of_ns - 1) * Ns;
    }

    assert(K == nr_of_tiles * weights_group * groups_per_tile);
    assert(N == ((nr_of_ns - 1) * Ns + NsTail));
    assert(NsTail % 32 == 0);
    assert(Ns % 32 == 0);
    assert(Ns >= NsTail);
}

inline void get_tiles_info_pipeline_opt(int   N,
                                        int   K,
                                        int   weights_group,
                                        int & nr_of_tiles,
                                        int & groups_per_tile,
                                        int & nr_of_ns,
                                        int & Ns,
                                        int & NsTail,
                                        int & NsTailValid,
                                        int & NAlloc) {
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);
    NsTailValid = NsTail;
    NAlloc      = N;

    if (N == 151936) {
        // Keep the last segment padded to 40960 columns so the tail uses the
        // same launch shape as the full segments. Valid columns are tracked
        // separately via NsTailValid and zero-filled on upload.
        Ns          = 1280 * 32;
        nr_of_ns    = 4;
        NsTail      = Ns;
        NsTailValid = N - (nr_of_ns - 1) * Ns;
        NAlloc      = nr_of_ns * Ns;
    }

    assert(N == ((nr_of_ns - 1) * Ns + NsTailValid));
    assert(NAlloc == ((nr_of_ns - 1) * Ns + NsTail));
    assert(NsTail >= NsTailValid);
}

inline size_t q5k_vxm_tail_zero_workspace_bytes(int Ktile,
                                                int NsTail,
                                                int NsTailValid,
                                                int wqlsb_per_word,
                                                int wqmsb_per_word,
                                                int q_group,
                                                int weights_group) {
    const int pad_cols = NsTail - NsTailValid;
    if (pad_cols <= 0) {
        return 0;
    }

    const size_t wqlsb_bytes       = (size_t) pad_cols * sizeof(short) * (size_t) (Ktile / wqlsb_per_word);
    const size_t wqmsb_bytes       = (size_t) pad_cols * sizeof(short) * (size_t) (Ktile / wqmsb_per_word);
    const size_t qscale_lsb_bytes  = (size_t) pad_cols * sizeof(short) * (size_t) (Ktile / 4 / q_group);
    const size_t qscale_msb_bytes  = qscale_lsb_bytes / 2;
    const size_t qzero_lsb_bytes   = qscale_lsb_bytes;
    const size_t qzero_msb_bytes   = qscale_msb_bytes;
    const size_t super_scale_bytes = (size_t) pad_cols * sizeof(short) * (size_t) (Ktile / weights_group);
    const size_t super_zero_bytes  = super_scale_bytes;

    return std::max(
        std::max(std::max(wqlsb_bytes, wqmsb_bytes), std::max(qscale_lsb_bytes, qscale_msb_bytes)),
        std::max(std::max(qzero_lsb_bytes, qzero_msb_bytes), std::max(super_scale_bytes, super_zero_bytes)));
}

inline RPPdeviceptr ensure_q5k_vxm_zero_workspace(rpp_kernel_context & ctx, size_t bytes) {
    if (bytes == 0) {
        return 0;
    }

    if (ctx.dev_aux_workspace != 0 && ctx.dev_aux_workspace_bytes >= bytes) {
        return ctx.dev_aux_workspace;
    }

    if (ctx.dev_aux_workspace != 0) {
        rtFree((void *) ctx.dev_aux_workspace);
        ctx.dev_aux_workspace       = 0;
        ctx.dev_aux_workspace_bytes = 0;
    }

    if (rtMalloc((void **) &ctx.dev_aux_workspace, bytes) != rtSuccess) {
        throw std::runtime_error("failed to allocate q5k_vxm zero workspace");
    }
    if (rtMemset((void *) ctx.dev_aux_workspace, 0, bytes) != rtSuccess) {
        rtFree((void *) ctx.dev_aux_workspace);
        ctx.dev_aux_workspace       = 0;
        ctx.dev_aux_workspace_bytes = 0;
        throw std::runtime_error("failed to initialize q5k_vxm zero workspace");
    }
    ctx.dev_aux_workspace_bytes = bytes;
    return ctx.dev_aux_workspace;
}

inline void copy_2d_tail_zero_async(RPPdeviceptr dst_base, int dst_pitch, RPPdeviceptr zero_src_base, int valid_cols,
                                    int total_cols, int height, RPPstream stream) {
    const int pad_cols = total_cols - valid_cols;
    if (pad_cols <= 0 || height <= 0) {
        return;
    }

    rtMemcpy2DAsync((void *) (dst_base + (size_t) valid_cols * sizeof(short)), dst_pitch, (const void *) zero_src_base,
                    pad_cols * (int) sizeof(short), pad_cols * (int) sizeof(short), height, rtMemcpyDeviceToSram,
                    stream);
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_q5k_vxm(rpp_kernel_context & ctx,
                               int                  M,
                               int                  K,
                               int                  N,
                               int                  weights_group,
                               int                  in_bytes_per_element,
                               int                  out_bytes_per_element,
                               int                  is_instantial = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q4 Configure Paramter Invalid");
    }
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA             = ctx.dev_in[0];
    RPPdeviceptr          devB_wq_lsb      = ctx.dev_in[1];
    RPPdeviceptr          devB_wq_msb      = ctx.dev_in[2];
    RPPdeviceptr          devB_qscale_lsb  = ctx.dev_in[3];
    RPPdeviceptr          devB_qzero_lsb   = ctx.dev_in[4];
    RPPdeviceptr          devB_qscale_msb  = ctx.dev_in[5];
    RPPdeviceptr          devB_qzero_msb   = ctx.dev_in[6];
    RPPdeviceptr          devB_super_scale = ctx.dev_in[7];
    RPPdeviceptr          devB_super_zero  = ctx.dev_in[8];
    RPPdeviceptr          devC             = ctx.dev_out[0];

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q5k_vxm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int logical_N = N;
    int       nr_of_tiles, groups_per_tile;
    int       nr_of_ns, Ns, NsTail;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);

    const int sizeA_raw      = K * in_bytes_per_element;
    const int sizeA_exec     = K * (int) sizeof(rpp::bfloat16);
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // q_group = 32
    // wq_per_group = 32
    // in_wq     [K/256]      |  [8]  |  [32/4]    |  [4][N]
    // in_wq     [z]          |  [y]  |  [unroll]  |  [x]
    //           [grid.y]*[z] |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    const int q_group        = 32;
    const int super_group    = 256;
    const int group          = super_group / q_group;
    int       bits_per_wqlsb = 4;
    int       bits_per_wqmsb = 1;
    int       wqlsb_per_word = sizeof(short) * 8 / bits_per_wqlsb;
    int       wqmsb_per_word = sizeof(short) * 8 / bits_per_wqmsb;
    assert(super_group == weights_group);
    const int sizeB_wq_lsb           = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqlsb_per_word);
    const int sizeB_wq_msb           = sizeB_wq_lsb / 4;
    const int sizeB_qscale_lsb_tile  = (sizeB_wq_lsb / q_group);
    const int sizeB_qscale_msb_tile  = sizeB_qscale_lsb_tile / 2;
    const int sizeB_qzero_lsb_tile   = sizeB_qscale_lsb_tile;
    const int sizeB_qzero_msb_tile   = sizeB_qscale_msb_tile;
    const int sizeB_super_scale_tile = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int sizeB_super_zero_tile  = sizeB_super_scale_tile;
    const int sizeB_qscale_lsb_full  = (int) (K * Ns * sizeof(short) / wqlsb_per_word / q_group);
    const int sizeB_qscale_msb_full  = sizeB_qscale_lsb_full / 2;
    const int sizeB_qzero_lsb_full   = sizeB_qscale_lsb_full;
    const int sizeB_qzero_msb_full   = sizeB_qscale_msb_full;
    const int sizeB_super_scale_full = (int) (K * Ns / weights_group * sizeof(short));
    const int sizeB_super_zero_full  = sizeB_super_scale_full;
    const int sizeB_scale            = (groups_per_tile * weights_group * Ns / q_group) * (int) sizeof(short);
    const int sizeB_zero             = sizeB_scale;

    // const int sizeZero = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeC32         = N * sizeof(float);
    const int sizeC           = N * out_bytes_per_element;
    const int SRAM_LIMIT      = 22 * 1024 * 1024;
    auto      calc_total_sram = [&](int sizeB_qscale_lsb_bytes, int sizeB_qzero_lsb_bytes, int sizeB_qscale_msb_bytes,
                               int sizeB_qzero_msb_bytes, int sizeB_super_scale_bytes,
                               int sizeB_super_zero_bytes) -> int {
        return ((in_bytes_per_element == (int) sizeof(float)) ? round_up(sizeA_raw) : 0) + round_up(sizeA_exec) +
               round_up(sizeC32) + round_up(sizeC) + round_up(sizeB_wq_lsb) + round_up(sizeB_wq_msb) +
               round_up(sizeB_qscale_lsb_bytes) + round_up(sizeB_qzero_lsb_bytes) + round_up(sizeB_qscale_msb_bytes) +
               round_up(sizeB_qzero_msb_bytes) + round_up(sizeB_super_scale_bytes) + round_up(sizeB_super_zero_bytes) +
               round_up(sizeB_scale) + round_up(sizeB_zero) + round_up(weights_group * 4);
    };
    const int total_sram_bytes_tiled =
        calc_total_sram(sizeB_qscale_lsb_tile, sizeB_qzero_lsb_tile, sizeB_qscale_msb_tile, sizeB_qzero_msb_tile,
                        sizeB_super_scale_tile, sizeB_super_zero_tile);
    const int total_sram_bytes_fullk_aux =
        calc_total_sram(sizeB_qscale_lsb_full, sizeB_qzero_lsb_full, sizeB_qscale_msb_full, sizeB_qzero_msb_full,
                        sizeB_super_scale_full, sizeB_super_zero_full);
    const bool use_fullk_aux          = (total_sram_bytes_fullk_aux <= SRAM_LIMIT);
    const int  sizeB_qscale_lsb_sram  = use_fullk_aux ? sizeB_qscale_lsb_full : sizeB_qscale_lsb_tile;
    const int  sizeB_qzero_lsb_sram   = use_fullk_aux ? sizeB_qzero_lsb_full : sizeB_qzero_lsb_tile;
    const int  sizeB_qscale_msb_sram  = use_fullk_aux ? sizeB_qscale_msb_full : sizeB_qscale_msb_tile;
    const int  sizeB_qzero_msb_sram   = use_fullk_aux ? sizeB_qzero_msb_full : sizeB_qzero_msb_tile;
    const int  sizeB_super_scale_sram = use_fullk_aux ? sizeB_super_scale_full : sizeB_super_scale_tile;
    const int  sizeB_super_zero_sram  = use_fullk_aux ? sizeB_super_zero_full : sizeB_super_zero_tile;
    const int  total_sram_bytes       = use_fullk_aux ? total_sram_bytes_fullk_aux : total_sram_bytes_tiled;

    RPPdeviceptr sram_base         = ctx.virtual_sram_base;
    RPPdeviceptr sramA_raw         = sram_base;
    RPPdeviceptr sramA             =
        (in_bytes_per_element == (int) sizeof(float)) ? (sramA_raw + round_up(sizeA_raw)) : sramA_raw;
    RPPdeviceptr sramC             = sramA + round_up(sizeA_exec);
    RPPdeviceptr sramC1            = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_wq_lsb      = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_wq_msb      = sramB_wq_lsb + round_up(sizeB_wq_lsb);
    RPPdeviceptr sramB_qscale_lsb  = sramB_wq_msb + round_up(sizeB_wq_msb);
    RPPdeviceptr sramB_qzero_lsb   = sramB_qscale_lsb + round_up(sizeB_qscale_lsb_sram);
    RPPdeviceptr sramB_qscale_msb  = sramB_qzero_lsb + round_up(sizeB_qzero_lsb_sram);
    RPPdeviceptr sramB_qzero_msb   = sramB_qscale_msb + round_up(sizeB_qscale_msb_sram);
    RPPdeviceptr sramB_super_scale = sramB_qzero_msb + round_up(sizeB_qzero_msb_sram);
    RPPdeviceptr sramB_super_zero  = sramB_super_scale + round_up(sizeB_super_scale_sram);
    RPPdeviceptr sramB_scale       = sramB_super_zero + round_up(sizeB_super_zero_sram);
    RPPdeviceptr sramB_zero        = sramB_scale + round_up(sizeB_scale);
    RPPdeviceptr sramA_acc         = sramB_zero + round_up(sizeB_zero);
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }
    // -------------------------
    // (1) DDR -> SRAM async copies on dmaStream
    // -------------------------
    rtMemcpyAsync((void *) sramA_raw, (const void *) devA, sizeA_raw, rtMemcpyDeviceToSram, ctx.kernelStream);
    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_raw, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    int Nx = K / q_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    int            hilo_offset = (Nx * Nz + 31) / 32 * 32 * sizeof(short);
    int            combine;
    RppTaskElement task_in_acc;
    task_in_acc.taskName   = "asym_quant_input_acc";
    task_in_acc.blockDim.x = Nx;
    task_in_acc.blockDim.y = 1;
    task_in_acc.blockDim.z = Nz;
    task_in_acc.gridDim.x  = 1;
    task_in_acc.gridDim.y  = 1;
    task_in_acc.gridDim.z  = 1;
    task_in_acc.params.kernelList.clear();
    task_in_acc.params.kernelList.emplace_back(sramA);
    task_in_acc.params.kernelList.emplace_back(sramA_acc);
    task_in_acc.params.kernelList.emplace_back(sramA_acc + hilo_offset);
    task_in_acc.params.kernelList.emplace_back(q_group);
    launchWrapperAysnc(task_in_acc.taskName, task_in_acc.gridDim, task_in_acc.blockDim, task_in_acc.params.kernelList,
                       ctx.rppBinMod, ctx.kernelStream);

    // -------------------------
    // with given split factor, launch kernel one Ns segment at a time
    // -------------------------
    int       Ktile                     = groups_per_tile * weights_group;
    const int wq_lsb_stride             = (Ktile / 4 * logical_N * sizeof(short));
    const int wq_msb_stride             = wq_lsb_stride / 4;
    const int qscale_lsb_stride         = wq_lsb_stride / q_group;
    const int qscale_msb_stride         = qscale_lsb_stride / 2;
    const int qzero_lsb_stride          = qscale_lsb_stride;
    const int qzero_msb_stride          = qscale_msb_stride;
    const int super_scale_stride        = Ktile / weights_group * logical_N * sizeof(short);
    const int super_zero_stride         = super_scale_stride;
    const int wq_lsb_rows_per_tile      = Ktile / 4;
    const int wq_msb_rows_per_tile      = wq_lsb_rows_per_tile / 4;
    const int qscale_lsb_rows_per_tile  = Ktile / 4 / q_group;
    const int qscale_msb_rows_per_tile  = qscale_lsb_rows_per_tile / 2;
    const int qzero_lsb_rows_per_tile   = qscale_lsb_rows_per_tile;
    const int qzero_msb_rows_per_tile   = qscale_msb_rows_per_tile;
    const int super_scale_rows_per_tile = Ktile / weights_group;
    const int super_zero_rows_per_tile  = super_scale_rows_per_tile;
    const int qscale_lsb_rows_full      = K / 4 / q_group;
    const int qscale_msb_rows_full      = qscale_lsb_rows_full / 2;
    const int qzero_lsb_rows_full       = qscale_lsb_rows_full;
    const int qzero_msb_rows_full       = qscale_msb_rows_full;
    const int super_scale_rows_full     = K / weights_group;
    const int super_zero_rows_full      = super_scale_rows_full;

    auto copy_wq_tile = [&](int i, int j, int NsSeg) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_wq_lsb, (const void *) (devB_wq_lsb + i * sizeB_wq_lsb), sizeB_wq_lsb,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_wq_msb, (const void *) (devB_wq_msb + i * sizeB_wq_msb), sizeB_wq_msb,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            return;
        }

        const int src_col_offset = j * Ns * sizeof(short);
        rtMemcpy2DAsync((void *) sramB_wq_lsb, NsSeg * sizeof(short),
                        (const void *) (devB_wq_lsb + i * wq_lsb_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), wq_lsb_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_wq_msb, NsSeg * sizeof(short),
                        (const void *) (devB_wq_msb + i * wq_msb_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), wq_msb_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
    };

    auto copy_aux_tile = [&](int i, int j, int NsSeg) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_qscale_lsb, (const void *) (devB_qscale_lsb + i * sizeB_qscale_lsb_tile),
                          sizeB_qscale_lsb_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qzero_lsb, (const void *) (devB_qzero_lsb + i * sizeB_qzero_lsb_tile),
                          sizeB_qzero_lsb_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qscale_msb, (const void *) (devB_qscale_msb + i * sizeB_qscale_msb_tile),
                          sizeB_qscale_msb_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qzero_msb, (const void *) (devB_qzero_msb + i * sizeB_qzero_msb_tile),
                          sizeB_qzero_msb_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_scale, (const void *) (devB_super_scale + i * sizeB_super_scale_tile),
                          sizeB_super_scale_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_zero, (const void *) (devB_super_zero + i * sizeB_super_zero_tile),
                          sizeB_super_zero_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
            return;
        }

        const int src_col_offset = j * Ns * sizeof(short);
        rtMemcpy2DAsync((void *) sramB_qscale_lsb, NsSeg * sizeof(short),
                        (const void *) (devB_qscale_lsb + i * qscale_lsb_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), qscale_lsb_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_qzero_lsb, NsSeg * sizeof(short),
                        (const void *) (devB_qzero_lsb + i * qzero_lsb_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), qzero_lsb_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_qscale_msb, NsSeg * sizeof(short),
                        (const void *) (devB_qscale_msb + i * qscale_msb_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), qscale_msb_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_qzero_msb, NsSeg * sizeof(short),
                        (const void *) (devB_qzero_msb + i * qzero_msb_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), qzero_msb_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_super_scale, NsSeg * sizeof(short),
                        (const void *) (devB_super_scale + i * super_scale_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), super_scale_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_super_zero, NsSeg * sizeof(short),
                        (const void *) (devB_super_zero + i * super_zero_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), super_zero_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
    };

    auto copy_aux_segment_fullk = [&](int j, int NsSeg) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_qscale_lsb, (const void *) devB_qscale_lsb, sizeB_qscale_lsb_full,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qzero_lsb, (const void *) devB_qzero_lsb, sizeB_qzero_lsb_full,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qscale_msb, (const void *) devB_qscale_msb, sizeB_qscale_msb_full,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qzero_msb, (const void *) devB_qzero_msb, sizeB_qzero_msb_full,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale_full,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_zero, (const void *) devB_super_zero, sizeB_super_zero_full,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            return;
        }

        const int src_col_offset = j * Ns * sizeof(short);
        rtMemcpy2DAsync((void *) sramB_qscale_lsb, NsSeg * sizeof(short),
                        (const void *) (devB_qscale_lsb + src_col_offset), N * sizeof(short), NsSeg * sizeof(short),
                        qscale_lsb_rows_full, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_qzero_lsb, NsSeg * sizeof(short),
                        (const void *) (devB_qzero_lsb + src_col_offset), N * sizeof(short), NsSeg * sizeof(short),
                        qzero_lsb_rows_full, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_qscale_msb, NsSeg * sizeof(short),
                        (const void *) (devB_qscale_msb + src_col_offset), N * sizeof(short), NsSeg * sizeof(short),
                        qscale_msb_rows_full, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_qzero_msb, NsSeg * sizeof(short),
                        (const void *) (devB_qzero_msb + src_col_offset), N * sizeof(short), NsSeg * sizeof(short),
                        qzero_msb_rows_full, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_super_scale, NsSeg * sizeof(short),
                        (const void *) (devB_super_scale + src_col_offset), N * sizeof(short), NsSeg * sizeof(short),
                        super_scale_rows_full, rtMemcpyDeviceToSram, ctx.kernelStream);
        rtMemcpy2DAsync((void *) sramB_super_zero, NsSeg * sizeof(short),
                        (const void *) (devB_super_zero + src_col_offset), N * sizeof(short), NsSeg * sizeof(short),
                        super_zero_rows_full, rtMemcpyDeviceToSram, ctx.kernelStream);
    };

    auto aux_tile_ptr = [&](RPPdeviceptr base, int rows_per_tile, int i, int NsSeg) -> RPPdeviceptr {
        return base + static_cast<RPPdeviceptr>(i) * rows_per_tile * NsSeg * sizeof(short);
    };

    for (int j = 0; j < nr_of_ns; ++j) {
        const int NsSeg = (j == nr_of_ns - 1) ? NsTail : Ns;
        if (use_fullk_aux) {
            // Keep non-Wq metadata resident for the whole K range of the current Ns slice.
            copy_aux_segment_fullk(j, NsSeg);
        }

        for (int i = 0; i < nr_of_tiles; ++i) {
            copy_wq_tile(i, j, NsSeg);
            if (!use_fullk_aux) {
                copy_aux_tile(i, j, NsSeg);
            }

            const RPPdeviceptr sramB_qscale_lsb_tile =
                use_fullk_aux ? aux_tile_ptr(sramB_qscale_lsb, qscale_lsb_rows_per_tile, i, NsSeg) : sramB_qscale_lsb;
            const RPPdeviceptr sramB_qzero_lsb_tile =
                use_fullk_aux ? aux_tile_ptr(sramB_qzero_lsb, qzero_lsb_rows_per_tile, i, NsSeg) : sramB_qzero_lsb;
            const RPPdeviceptr sramB_qscale_msb_tile =
                use_fullk_aux ? aux_tile_ptr(sramB_qscale_msb, qscale_msb_rows_per_tile, i, NsSeg) : sramB_qscale_msb;
            const RPPdeviceptr sramB_qzero_msb_tile =
                use_fullk_aux ? aux_tile_ptr(sramB_qzero_msb, qzero_msb_rows_per_tile, i, NsSeg) : sramB_qzero_msb;
            const RPPdeviceptr sramB_super_scale_tile =
                use_fullk_aux ? aux_tile_ptr(sramB_super_scale, super_scale_rows_per_tile, i, NsSeg) :
                                sramB_super_scale;
            const RPPdeviceptr sramB_super_zero_tile =
                use_fullk_aux ? aux_tile_ptr(sramB_super_zero, super_zero_rows_per_tile, i, NsSeg) : sramB_super_zero;

            params.clear();
            q5k_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            q5k_super_scale_params(sramB_qscale_lsb_tile, sramB_qscale_msb_tile, sramB_super_scale_tile, sramB_scale,
                                   Ktile, NsSeg, weights_group, q_group, 0, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q5k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            params.clear();
            q5k_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            q5k_super_scale_params(sramB_qzero_lsb_tile, sramB_qzero_msb_tile, sramB_super_zero_tile, sramB_zero, Ktile,
                                   NsSeg, weights_group, q_group, 1, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q5k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            combine = (i == 0) ? 0 : 1;

            RppTaskElement task;
            uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
            kerneldim_calc_matmul_linear(1, M, NsSeg, block_x, block_y, block_z, grid_x, grid_y, grid_z);
            task.blockDim.x = block_x;
            task.blockDim.y = block_y;
            task.blockDim.z = block_z;
            task.gridDim.x  = grid_x;
            task.gridDim.y  = grid_y;
            task.gridDim.z  = grid_z;

            task.gridDim.z          = groups_per_tile;
            uint32_t stride_ina     = weights_group * sizeof(short);
            //---------------------------------------------------------------------------------------
            // InputAcc [K/256]      | [8]
            //         [grid.z]      | [loopidx]
            // accIdx = blockIdx.z * 8 + loopidx
            //----------------------------------------------------------------------------------------
            uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * sizeof(short);
            task.taskName           = "matrix_mul_vxM_f16_q5k_f16_asym_opt";
            task.params.kernelList.clear();
            dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
            dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
            matmul_weights_q5k_kernel_params(
                matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile, sramB_wq_lsb, sramB_wq_msb,
                sramB_scale, sramB_zero, sramC + j * Ns * sizeof(short), 0, input_acc_addr,
                input_acc_addr + hilo_offset, NsSeg, N * sizeof(short), weights_group, combine, task.params.kernelList);
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);

        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpyAsync((void *) devC, (const void *) sramC1, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        rtMemcpyAsync((void *) devC, (const void *) sramC, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static bool rpp_matmul_q5k_vxm_pipeline(rpp_kernel_context & ctx,
                                        int                  M,
                                        int                  K,
                                        int                  N,
                                        int                  weights_group,
                                        int                  in_bytes_per_element,
                                        int                  out_bytes_per_element,
                                        int                  is_instantial = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q4 Configure Paramter Invalid");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr       devA                  = ctx.dev_in[0];
    const RPPdeviceptr devB_wq_lsb_base      = ctx.dev_in[1];
    const RPPdeviceptr devB_wq_msb_base      = ctx.dev_in[2];
    const RPPdeviceptr devB_qscale_lsb_base  = ctx.dev_in[3];
    const RPPdeviceptr devB_qzero_lsb_base   = ctx.dev_in[4];
    const RPPdeviceptr devB_qscale_msb_base  = ctx.dev_in[5];
    const RPPdeviceptr devB_qzero_msb_base   = ctx.dev_in[6];
    const RPPdeviceptr devB_super_scale_base = ctx.dev_in[7];
    const RPPdeviceptr devB_super_zero_base  = ctx.dev_in[8];
    RPPdeviceptr       devC                  = ctx.dev_out[0];

    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q5k_vxm.o");

    const int logical_N = N;
    int       nr_of_tiles, groups_per_tile;
    int       nr_of_ns, Ns, NsTail, NsTailValid, NAlloc;
    get_tiles_info_pipeline_opt(logical_N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail,
                                NsTailValid, NAlloc);

    const int sizeA_raw      = K * in_bytes_per_element;
    const int sizeA_exec     = K * (int) sizeof(rpp::bfloat16);
    const int q_group        = 32;
    const int super_group    = 256;
    const int group          = super_group / q_group;
    int       bits_per_wqlsb = 4;
    int       bits_per_wqmsb = 1;
    int       wqlsb_per_word = sizeof(short) * 8 / bits_per_wqlsb;
    int       wqmsb_per_word = sizeof(short) * 8 / bits_per_wqmsb;
    assert(super_group == weights_group);

    const int sizeB_wq_lsb           = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqlsb_per_word);
    const int sizeB_wq_msb           = sizeB_wq_lsb / 4;
    const int sizeB_qscale_lsb_tile  = (sizeB_wq_lsb / q_group);
    const int sizeB_qscale_msb_tile  = sizeB_qscale_lsb_tile / 2;
    const int sizeB_qzero_lsb_tile   = sizeB_qscale_lsb_tile;
    const int sizeB_qzero_msb_tile   = sizeB_qscale_msb_tile;
    const int sizeB_super_scale_tile = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int sizeB_super_zero_tile  = sizeB_super_scale_tile;
    const int sizeB_qscale_lsb_full  = (int) (K * Ns * sizeof(short) / wqlsb_per_word / q_group);
    const int sizeB_qscale_msb_full  = sizeB_qscale_lsb_full / 2;
    const int sizeB_qzero_lsb_full   = sizeB_qscale_lsb_full;
    const int sizeB_qzero_msb_full   = sizeB_qscale_msb_full;
    const int sizeB_super_scale_full = (int) (K * Ns / weights_group * sizeof(short));
    const int sizeB_super_zero_full  = sizeB_super_scale_full;
    const int sizeB_scale            = (groups_per_tile * weights_group * Ns / q_group) * (int) sizeof(short);
    const int sizeB_zero             = sizeB_scale;
    const int sizeC32                = NAlloc * sizeof(float);
    const int sizeC                  = NAlloc * out_bytes_per_element;
    const int SRAM_LIMIT             = 22 * 1024 * 1024;

    auto calc_pipeline_total_sram = [&](bool fullk_aux) -> int {
        const int qscale_lsb_bytes = fullk_aux ? round_up(sizeB_qscale_lsb_full) : 2 * round_up(sizeB_qscale_lsb_tile);
        const int qzero_lsb_bytes  = fullk_aux ? round_up(sizeB_qzero_lsb_full) : 2 * round_up(sizeB_qzero_lsb_tile);
        const int qscale_msb_bytes = fullk_aux ? round_up(sizeB_qscale_msb_full) : 2 * round_up(sizeB_qscale_msb_tile);
        const int qzero_msb_bytes  = fullk_aux ? round_up(sizeB_qzero_msb_full) : 2 * round_up(sizeB_qzero_msb_tile);
        const int super_scale_bytes =
            fullk_aux ? round_up(sizeB_super_scale_full) : 2 * round_up(sizeB_super_scale_tile);
        const int super_zero_bytes = fullk_aux ? round_up(sizeB_super_zero_full) : 2 * round_up(sizeB_super_zero_tile);

        return ((in_bytes_per_element == (int) sizeof(float)) ? round_up(sizeA_raw) : 0) + round_up(sizeA_exec) +
               round_up(sizeC32) + round_up(sizeC) + 2 * round_up(sizeB_wq_lsb) +
               2 * round_up(sizeB_wq_msb) + qscale_lsb_bytes + qzero_lsb_bytes + qscale_msb_bytes + qzero_msb_bytes +
               super_scale_bytes + super_zero_bytes + round_up(sizeB_scale) + round_up(sizeB_zero) +
               round_up(weights_group * 4);
    };

    const int total_sram_bytes_fullk_aux = calc_pipeline_total_sram(true);
    const int total_sram_bytes_tiled     = calc_pipeline_total_sram(false);
    const bool fullk_aux_fits            = (total_sram_bytes_fullk_aux <= SRAM_LIMIT);
    const bool tiled_aux_fits            = (total_sram_bytes_tiled <= SRAM_LIMIT);
    // When there is only one tile, full-K aux leaves no ping-pong work to overlap
    // across segments. Prefer tiled aux in that case so seg+1 DMA can overlap seg.
    const bool prefer_tiled_single_tile = (nr_of_tiles == 1) && (nr_of_ns > 1);
    const bool use_fullk_aux            = fullk_aux_fits && !prefer_tiled_single_tile;
    if (!use_fullk_aux && !tiled_aux_fits) {
        return false;
    }
    const int total_sram_bytes = use_fullk_aux ? total_sram_bytes_fullk_aux : total_sram_bytes_tiled;
    if (total_sram_bytes > SRAM_LIMIT) {
        return false;
    }

    RPPdeviceptr sram_base      = ctx.virtual_sram_base;
    RPPdeviceptr sramA_raw      = sram_base;
    RPPdeviceptr sramA          =
        (in_bytes_per_element == (int) sizeof(float)) ? (sramA_raw + round_up(sizeA_raw)) : sramA_raw;
    RPPdeviceptr sramC          = sramA + round_up(sizeA_exec);
    RPPdeviceptr sramC1         = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_wq_lsb_0 = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_wq_msb_0 = sramB_wq_lsb_0 + round_up(sizeB_wq_lsb);
    RPPdeviceptr sramB_wq_lsb_1 = sramB_wq_msb_0 + round_up(sizeB_wq_msb);
    RPPdeviceptr sramB_wq_msb_1 = sramB_wq_lsb_1 + round_up(sizeB_wq_lsb);
    RPPdeviceptr cursor         = sramB_wq_msb_1 + round_up(sizeB_wq_msb);

    RPPdeviceptr sramB_qscale_lsb  = 0;
    RPPdeviceptr sramB_qzero_lsb   = 0;
    RPPdeviceptr sramB_qscale_msb  = 0;
    RPPdeviceptr sramB_qzero_msb   = 0;
    RPPdeviceptr sramB_super_scale = 0;
    RPPdeviceptr sramB_super_zero  = 0;

    RPPdeviceptr sramB_qscale_lsb_0  = 0;
    RPPdeviceptr sramB_qzero_lsb_0   = 0;
    RPPdeviceptr sramB_qscale_msb_0  = 0;
    RPPdeviceptr sramB_qzero_msb_0   = 0;
    RPPdeviceptr sramB_super_scale_0 = 0;
    RPPdeviceptr sramB_super_zero_0  = 0;
    RPPdeviceptr sramB_qscale_lsb_1  = 0;
    RPPdeviceptr sramB_qzero_lsb_1   = 0;
    RPPdeviceptr sramB_qscale_msb_1  = 0;
    RPPdeviceptr sramB_qzero_msb_1   = 0;
    RPPdeviceptr sramB_super_scale_1 = 0;
    RPPdeviceptr sramB_super_zero_1  = 0;

    if (use_fullk_aux) {
        sramB_qscale_lsb  = cursor;
        sramB_qzero_lsb   = sramB_qscale_lsb + round_up(sizeB_qscale_lsb_full);
        sramB_qscale_msb  = sramB_qzero_lsb + round_up(sizeB_qzero_lsb_full);
        sramB_qzero_msb   = sramB_qscale_msb + round_up(sizeB_qscale_msb_full);
        sramB_super_scale = sramB_qzero_msb + round_up(sizeB_qzero_msb_full);
        sramB_super_zero  = sramB_super_scale + round_up(sizeB_super_scale_full);
        cursor            = sramB_super_zero + round_up(sizeB_super_zero_full);
    } else {
        sramB_qscale_lsb_0  = cursor;
        sramB_qzero_lsb_0   = sramB_qscale_lsb_0 + round_up(sizeB_qscale_lsb_tile);
        sramB_qscale_msb_0  = sramB_qzero_lsb_0 + round_up(sizeB_qzero_lsb_tile);
        sramB_qzero_msb_0   = sramB_qscale_msb_0 + round_up(sizeB_qscale_msb_tile);
        sramB_super_scale_0 = sramB_qzero_msb_0 + round_up(sizeB_qzero_msb_tile);
        sramB_super_zero_0  = sramB_super_scale_0 + round_up(sizeB_super_scale_tile);

        sramB_qscale_lsb_1  = sramB_super_zero_0 + round_up(sizeB_super_zero_tile);
        sramB_qzero_lsb_1   = sramB_qscale_lsb_1 + round_up(sizeB_qscale_lsb_tile);
        sramB_qscale_msb_1  = sramB_qzero_lsb_1 + round_up(sizeB_qzero_lsb_tile);
        sramB_qzero_msb_1   = sramB_qscale_msb_1 + round_up(sizeB_qscale_msb_tile);
        sramB_super_scale_1 = sramB_qzero_msb_1 + round_up(sizeB_qzero_msb_tile);
        sramB_super_zero_1  = sramB_super_scale_1 + round_up(sizeB_super_scale_tile);
        cursor              = sramB_super_zero_1 + round_up(sizeB_super_zero_tile);
    }

    RPPdeviceptr sramB_scale = cursor;
    RPPdeviceptr sramB_zero  = sramB_scale + round_up(sizeB_scale);
    RPPdeviceptr sramA_acc   = sramB_zero + round_up(sizeB_zero);

    auto sramB_wq_lsb_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_wq_lsb_1 : sramB_wq_lsb_0;
    };
    auto sramB_wq_msb_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_wq_msb_1 : sramB_wq_msb_0;
    };
    auto sramB_qscale_lsb_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_qscale_lsb_1 : sramB_qscale_lsb_0;
    };
    auto sramB_qzero_lsb_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_qzero_lsb_1 : sramB_qzero_lsb_0;
    };
    auto sramB_qscale_msb_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_qscale_msb_1 : sramB_qscale_msb_0;
    };
    auto sramB_qzero_msb_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_qzero_msb_1 : sramB_qzero_msb_0;
    };
    auto sramB_super_scale_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_super_scale_1 : sramB_super_scale_0;
    };
    auto sramB_super_zero_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_super_zero_1 : sramB_super_zero_0;
    };

    int                 Ktile = groups_per_tile * weights_group;
    const size_t        tail_zero_workspace_bytes =
        q5k_vxm_tail_zero_workspace_bytes(Ktile, NsTail, NsTailValid, wqlsb_per_word, wqmsb_per_word, q_group,
                                          weights_group);
    const RPPdeviceptr  zero_workspace = ensure_q5k_vxm_zero_workspace(ctx, tail_zero_workspace_bytes);

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);

    rtMemcpyAsync((void *) sramA_raw, (const void *) devA, sizeA_raw, rtMemcpyDeviceToSram, ctx.kernelStream);
    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_raw, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    int Nx = K / q_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    int            hilo_offset = (Nx * Nz + 31) / 32 * 32 * sizeof(short);
    RppTaskElement task_in_acc;
    task_in_acc.taskName   = "asym_quant_input_acc";
    task_in_acc.blockDim.x = Nx;
    task_in_acc.blockDim.y = 1;
    task_in_acc.blockDim.z = Nz;
    task_in_acc.gridDim.x  = 1;
    task_in_acc.gridDim.y  = 1;
    task_in_acc.gridDim.z  = 1;
    task_in_acc.params.kernelList.clear();
    task_in_acc.params.kernelList.emplace_back(sramA);
    task_in_acc.params.kernelList.emplace_back(sramA_acc);
    task_in_acc.params.kernelList.emplace_back(sramA_acc + hilo_offset);
    task_in_acc.params.kernelList.emplace_back(q_group);
    launchWrapperAysnc(task_in_acc.taskName, task_in_acc.gridDim, task_in_acc.blockDim, task_in_acc.params.kernelList,
                       ctx.rppBinMod, ctx.kernelStream);

    const int wq_lsb_stride             = (Ktile / 4 * logical_N * sizeof(short));
    const int wq_msb_stride             = wq_lsb_stride / 4;
    const int qscale_lsb_stride         = wq_lsb_stride / q_group;
    const int qscale_msb_stride         = qscale_lsb_stride / 2;
    const int qzero_lsb_stride          = qscale_lsb_stride;
    const int qzero_msb_stride          = qscale_msb_stride;
    const int super_scale_stride        = Ktile / weights_group * logical_N * sizeof(short);
    const int super_zero_stride         = super_scale_stride;
    const int wq_lsb_rows_per_tile      = Ktile / 4;
    const int wq_msb_rows_per_tile      = wq_lsb_rows_per_tile / 4;
    const int qscale_lsb_rows_per_tile  = Ktile / 4 / q_group;
    const int qscale_msb_rows_per_tile  = qscale_lsb_rows_per_tile / 2;
    const int qzero_lsb_rows_per_tile   = qscale_lsb_rows_per_tile;
    const int qzero_msb_rows_per_tile   = qscale_msb_rows_per_tile;
    const int super_scale_rows_per_tile = Ktile / weights_group;
    const int super_zero_rows_per_tile  = super_scale_rows_per_tile;
    const int qscale_lsb_rows_full      = K / 4 / q_group;
    const int qscale_msb_rows_full      = qscale_lsb_rows_full / 2;
    const int qzero_lsb_rows_full       = qscale_lsb_rows_full;
    const int qzero_msb_rows_full       = qscale_msb_rows_full;
    const int super_scale_rows_full     = K / weights_group;
    const int super_zero_rows_full      = super_scale_rows_full;

    auto segment_alloc_ns = [&](int seg) -> int {
        return (seg == nr_of_ns - 1) ? NsTail : Ns;
    };
    auto segment_valid_ns = [&](int seg) -> int {
        return (seg == nr_of_ns - 1) ? NsTailValid : Ns;
    };

    auto aux_tile_ptr = [&](RPPdeviceptr base, int rows_per_tile, int i, int NsSeg) -> RPPdeviceptr {
        return base + static_cast<RPPdeviceptr>(i) * rows_per_tile * NsSeg * sizeof(short);
    };

    auto launch_matmul_tile = [&](int i, int j, int NsSeg, RPPdeviceptr sramB_wq_lsb_tile,
                                  RPPdeviceptr sramB_wq_msb_tile, RPPdeviceptr sramB_qscale_lsb_tile,
                                  RPPdeviceptr sramB_qzero_lsb_tile, RPPdeviceptr sramB_qscale_msb_tile,
                                  RPPdeviceptr sramB_qzero_msb_tile, RPPdeviceptr sramB_super_scale_tile,
                                  RPPdeviceptr sramB_super_zero_tile) {
        params.clear();
        q5k_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
        q5k_super_scale_params(sramB_qscale_lsb_tile, sramB_qscale_msb_tile, sramB_super_scale_tile, sramB_scale, Ktile,
                               NsSeg, weights_group, q_group, 0, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q5k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        q5k_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
        q5k_super_scale_params(sramB_qzero_lsb_tile, sramB_qzero_msb_tile, sramB_super_zero_tile, sramB_zero, Ktile,
                               NsSeg, weights_group, q_group, 1, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q5k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        int            combine = (i == 0) ? 0 : 1;
        RppTaskElement task;
        uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
        kerneldim_calc_matmul_linear(1, M, NsSeg, block_x, block_y, block_z, grid_x, grid_y, grid_z);
        task.blockDim.x = block_x;
        task.blockDim.y = block_y;
        task.blockDim.z = block_z;
        task.gridDim.x  = grid_x;
        task.gridDim.y  = grid_y;
        task.gridDim.z  = grid_z;
        task.gridDim.z  = groups_per_tile;

        uint32_t stride_ina     = weights_group * sizeof(short);
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * sizeof(short);
        task.taskName           = "matrix_mul_vxM_f16_q5k_f16_asym_opt";
        task.params.kernelList.clear();
        dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q5k_kernel_params(
            matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile, sramB_wq_lsb_tile, sramB_wq_msb_tile,
            sramB_scale, sramB_zero, sramC + j * Ns * sizeof(short), 0, input_acc_addr, input_acc_addr + hilo_offset,
            NsSeg, logical_N * sizeof(short), weights_group, combine, task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    };

    if (use_fullk_aux) {
        auto copy_aux_segment_fullk_dma = [&](int seg, int NsSegAlloc, int NsSegValid) {
            const int aux_min_chunk_bytes = 32 * 1024;
            const int aux_target_chunk_bytes = 64 * 1024;
            const int aux_max_chunk_rows = 16;
            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_qscale_lsb, (const void *) devB_qscale_lsb_base, sizeB_qscale_lsb_full,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_qzero_lsb, (const void *) devB_qzero_lsb_base, sizeB_qzero_lsb_full,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_qscale_msb, (const void *) devB_qscale_msb_base, sizeB_qscale_msb_full,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_qzero_msb, (const void *) devB_qzero_msb_base, sizeB_qzero_msb_full,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale_base, sizeB_super_scale_full,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_super_zero, (const void *) devB_super_zero_base, sizeB_super_zero_full,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                return;
            }

            const int src_col_offset = seg * Ns * sizeof(short);
            memcpy2d_chunked_async(sramB_qscale_lsb, NsSegAlloc * sizeof(short),
                                   devB_qscale_lsb_base + src_col_offset, logical_N * sizeof(short),
                                   NsSegValid * sizeof(short), qscale_lsb_rows_full,
                                   rtMemcpyDeviceToSram, ctx.dmaStream,
                                   aux_min_chunk_bytes, aux_target_chunk_bytes, aux_max_chunk_rows);
            copy_2d_tail_zero_async(sramB_qscale_lsb, NsSegAlloc * sizeof(short), zero_workspace,
                                    NsSegValid, NsSegAlloc, qscale_lsb_rows_full, ctx.dmaStream);
            memcpy2d_chunked_async(sramB_qzero_lsb, NsSegAlloc * sizeof(short),
                                   devB_qzero_lsb_base + src_col_offset, logical_N * sizeof(short),
                                   NsSegValid * sizeof(short), qzero_lsb_rows_full,
                                   rtMemcpyDeviceToSram, ctx.dmaStream,
                                   aux_min_chunk_bytes, aux_target_chunk_bytes, aux_max_chunk_rows);
            copy_2d_tail_zero_async(sramB_qzero_lsb, NsSegAlloc * sizeof(short), zero_workspace,
                                    NsSegValid, NsSegAlloc, qzero_lsb_rows_full, ctx.dmaStream);
            memcpy2d_chunked_async(sramB_qscale_msb, NsSegAlloc * sizeof(short),
                                   devB_qscale_msb_base + src_col_offset, logical_N * sizeof(short),
                                   NsSegValid * sizeof(short), qscale_msb_rows_full,
                                   rtMemcpyDeviceToSram, ctx.dmaStream,
                                   aux_min_chunk_bytes, aux_target_chunk_bytes, aux_max_chunk_rows);
            copy_2d_tail_zero_async(sramB_qscale_msb, NsSegAlloc * sizeof(short), zero_workspace,
                                    NsSegValid, NsSegAlloc, qscale_msb_rows_full, ctx.dmaStream);
            memcpy2d_chunked_async(sramB_qzero_msb, NsSegAlloc * sizeof(short),
                                   devB_qzero_msb_base + src_col_offset, logical_N * sizeof(short),
                                   NsSegValid * sizeof(short), qzero_msb_rows_full,
                                   rtMemcpyDeviceToSram, ctx.dmaStream,
                                   aux_min_chunk_bytes, aux_target_chunk_bytes, aux_max_chunk_rows);
            copy_2d_tail_zero_async(sramB_qzero_msb, NsSegAlloc * sizeof(short), zero_workspace,
                                    NsSegValid, NsSegAlloc, qzero_msb_rows_full, ctx.dmaStream);
            memcpy2d_chunked_async(sramB_super_scale, NsSegAlloc * sizeof(short),
                                   devB_super_scale_base + src_col_offset, logical_N * sizeof(short),
                                   NsSegValid * sizeof(short), super_scale_rows_full,
                                   rtMemcpyDeviceToSram, ctx.dmaStream,
                                   aux_min_chunk_bytes, aux_target_chunk_bytes, aux_max_chunk_rows);
            copy_2d_tail_zero_async(sramB_super_scale, NsSegAlloc * sizeof(short), zero_workspace,
                                    NsSegValid, NsSegAlloc, super_scale_rows_full, ctx.dmaStream);
            memcpy2d_chunked_async(sramB_super_zero, NsSegAlloc * sizeof(short),
                                   devB_super_zero_base + src_col_offset, logical_N * sizeof(short),
                                   NsSegValid * sizeof(short), super_zero_rows_full,
                                   rtMemcpyDeviceToSram, ctx.dmaStream,
                                   aux_min_chunk_bytes, aux_target_chunk_bytes, aux_max_chunk_rows);
            copy_2d_tail_zero_async(sramB_super_zero, NsSegAlloc * sizeof(short), zero_workspace,
                                    NsSegValid, NsSegAlloc, super_zero_rows_full, ctx.dmaStream);
        };

        auto schedule_wq_dma = [&](int i, int seg, int NsSegAlloc, int NsSegValid, int ping) {
            const int weight_min_chunk_bytes = 32 * 1024;
            const int weight_target_chunk_bytes = 1024 * 1024;
            const int weight_max_chunk_rows = 32;
            rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);
            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_wq_lsb_ping(ping), (const void *) (devB_wq_lsb_base + i * sizeB_wq_lsb),
                              sizeB_wq_lsb, rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_wq_msb_ping(ping), (const void *) (devB_wq_msb_base + i * sizeB_wq_msb),
                              sizeB_wq_msb, rtMemcpyDeviceToSram, ctx.dmaStream);
            } else {
                const int src_col_offset = seg * Ns * sizeof(short);
                memcpy2d_chunked_async(sramB_wq_lsb_ping(ping), NsSegAlloc * sizeof(short),
                                       devB_wq_lsb_base + i * wq_lsb_stride + src_col_offset,
                                       logical_N * sizeof(short),
                                       NsSegValid * sizeof(short), wq_lsb_rows_per_tile,
                                       rtMemcpyDeviceToSram, ctx.dmaStream,
                                       weight_min_chunk_bytes, weight_target_chunk_bytes, weight_max_chunk_rows);
                copy_2d_tail_zero_async(sramB_wq_lsb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                        NsSegValid, NsSegAlloc, wq_lsb_rows_per_tile, ctx.dmaStream);
                memcpy2d_chunked_async(sramB_wq_msb_ping(ping), NsSegAlloc * sizeof(short),
                                       devB_wq_msb_base + i * wq_msb_stride + src_col_offset,
                                       logical_N * sizeof(short),
                                       NsSegValid * sizeof(short), wq_msb_rows_per_tile,
                                       rtMemcpyDeviceToSram, ctx.dmaStream,
                                       weight_min_chunk_bytes, weight_target_chunk_bytes, weight_max_chunk_rows);
                copy_2d_tail_zero_async(sramB_wq_msb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                        NsSegValid, NsSegAlloc, wq_msb_rows_per_tile, ctx.dmaStream);
            }
            rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
        };

        const int segment_tail_ping = (nr_of_tiles > 0) ? ((nr_of_tiles - 1) & 1) : 0;
        for (int seg = 0; seg < nr_of_ns; ++seg) {
            const int NsSegAlloc = segment_alloc_ns(seg);
            const int NsSegValid = segment_valid_ns(seg);
            rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[segment_tail_ping], 0);
            copy_aux_segment_fullk_dma(seg, NsSegAlloc, NsSegValid);
            schedule_wq_dma(0, seg, NsSegAlloc, NsSegValid, 0);

            for (int i = 0; i < nr_of_tiles; ++i) {
                const int ping = i & 1;
                rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);
                if (i + 1 < nr_of_tiles) {
                    schedule_wq_dma(i + 1, seg, NsSegAlloc, NsSegValid, (i + 1) & 1);
                }

                launch_matmul_tile(i, seg, NsSegAlloc,
                                   sramB_wq_lsb_ping(ping), sramB_wq_msb_ping(ping),
                                   aux_tile_ptr(sramB_qscale_lsb, qscale_lsb_rows_per_tile, i, NsSegAlloc),
                                   aux_tile_ptr(sramB_qzero_lsb, qzero_lsb_rows_per_tile, i, NsSegAlloc),
                                   aux_tile_ptr(sramB_qscale_msb, qscale_msb_rows_per_tile, i, NsSegAlloc),
                                   aux_tile_ptr(sramB_qzero_msb, qzero_msb_rows_per_tile, i, NsSegAlloc),
                                   aux_tile_ptr(sramB_super_scale, super_scale_rows_per_tile, i, NsSegAlloc),
                                   aux_tile_ptr(sramB_super_zero, super_zero_rows_per_tile, i, NsSegAlloc));
                rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
            }
        }
    } else {
        auto schedule_dma = [&](int i, int j, int NsSegAlloc, int NsSegValid, int ping) {
            rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_wq_lsb_ping(ping), (const void *) (devB_wq_lsb_base + i * sizeB_wq_lsb),
                              sizeB_wq_lsb, rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_wq_msb_ping(ping), (const void *) (devB_wq_msb_base + i * sizeB_wq_msb),
                              sizeB_wq_msb, rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_qscale_lsb_ping(ping),
                              (const void *) (devB_qscale_lsb_base + i * sizeB_qscale_lsb_tile), sizeB_qscale_lsb_tile,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_qzero_lsb_ping(ping),
                              (const void *) (devB_qzero_lsb_base + i * sizeB_qzero_lsb_tile), sizeB_qzero_lsb_tile,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_qscale_msb_ping(ping),
                              (const void *) (devB_qscale_msb_base + i * sizeB_qscale_msb_tile), sizeB_qscale_msb_tile,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_qzero_msb_ping(ping),
                              (const void *) (devB_qzero_msb_base + i * sizeB_qzero_msb_tile), sizeB_qzero_msb_tile,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_super_scale_ping(ping),
                              (const void *) (devB_super_scale_base + i * sizeB_super_scale_tile),
                              sizeB_super_scale_tile, rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpyAsync((void *) sramB_super_zero_ping(ping),
                              (const void *) (devB_super_zero_base + i * sizeB_super_zero_tile), sizeB_super_zero_tile,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
            } else {
                const int src_col_offset = j * Ns * sizeof(short);
                memcpy2d_chunked_async(sramB_wq_lsb_ping(ping), NsSegAlloc * sizeof(short),
                                       devB_wq_lsb_base + i * wq_lsb_stride + src_col_offset,
                                       logical_N * sizeof(short), NsSegValid * sizeof(short), wq_lsb_rows_per_tile,
                                       rtMemcpyDeviceToSram, ctx.dmaStream);
                copy_2d_tail_zero_async(sramB_wq_lsb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace, NsSegValid,
                                        NsSegAlloc, wq_lsb_rows_per_tile, ctx.dmaStream);

                memcpy2d_chunked_async(sramB_wq_msb_ping(ping), NsSegAlloc * sizeof(short),
                                       devB_wq_msb_base + i * wq_msb_stride + src_col_offset,
                                       logical_N * sizeof(short), NsSegValid * sizeof(short), wq_msb_rows_per_tile,
                                       rtMemcpyDeviceToSram, ctx.dmaStream);
                copy_2d_tail_zero_async(sramB_wq_msb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace, NsSegValid,
                                        NsSegAlloc, wq_msb_rows_per_tile, ctx.dmaStream);

                rtMemcpy2DAsync((void *) sramB_qscale_lsb_ping(ping), NsSegAlloc * sizeof(short),
                                (const void *) (devB_qscale_lsb_base + i * qscale_lsb_stride + src_col_offset),
                                logical_N * sizeof(short), NsSegValid * sizeof(short), qscale_lsb_rows_per_tile,
                                rtMemcpyDeviceToSram, ctx.dmaStream);
                copy_2d_tail_zero_async(sramB_qscale_lsb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                        NsSegValid, NsSegAlloc, qscale_lsb_rows_per_tile, ctx.dmaStream);

                rtMemcpy2DAsync((void *) sramB_qzero_lsb_ping(ping), NsSegAlloc * sizeof(short),
                                (const void *) (devB_qzero_lsb_base + i * qzero_lsb_stride + src_col_offset),
                                logical_N * sizeof(short), NsSegValid * sizeof(short), qzero_lsb_rows_per_tile,
                                rtMemcpyDeviceToSram,
                                ctx.dmaStream);
                copy_2d_tail_zero_async(sramB_qzero_lsb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                        NsSegValid, NsSegAlloc, qzero_lsb_rows_per_tile, ctx.dmaStream);

                rtMemcpy2DAsync((void *) sramB_qscale_msb_ping(ping), NsSegAlloc * sizeof(short),
                                (const void *) (devB_qscale_msb_base + i * qscale_msb_stride + src_col_offset),
                                logical_N * sizeof(short), NsSegValid * sizeof(short), qscale_msb_rows_per_tile,
                                rtMemcpyDeviceToSram, ctx.dmaStream);
                copy_2d_tail_zero_async(sramB_qscale_msb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                        NsSegValid, NsSegAlloc, qscale_msb_rows_per_tile, ctx.dmaStream);

                rtMemcpy2DAsync((void *) sramB_qzero_msb_ping(ping), NsSegAlloc * sizeof(short),
                                (const void *) (devB_qzero_msb_base + i * qzero_msb_stride + src_col_offset),
                                logical_N * sizeof(short), NsSegValid * sizeof(short), qzero_msb_rows_per_tile,
                                rtMemcpyDeviceToSram,
                                ctx.dmaStream);
                copy_2d_tail_zero_async(sramB_qzero_msb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                        NsSegValid, NsSegAlloc, qzero_msb_rows_per_tile, ctx.dmaStream);

                rtMemcpy2DAsync((void *) sramB_super_scale_ping(ping), NsSegAlloc * sizeof(short),
                                (const void *) (devB_super_scale_base + i * super_scale_stride + src_col_offset),
                                logical_N * sizeof(short), NsSegValid * sizeof(short), super_scale_rows_per_tile,
                                rtMemcpyDeviceToSram, ctx.dmaStream);
                copy_2d_tail_zero_async(sramB_super_scale_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                        NsSegValid, NsSegAlloc, super_scale_rows_per_tile, ctx.dmaStream);

                rtMemcpy2DAsync((void *) sramB_super_zero_ping(ping), NsSegAlloc * sizeof(short),
                                (const void *) (devB_super_zero_base + i * super_zero_stride + src_col_offset),
                                logical_N * sizeof(short), NsSegValid * sizeof(short), super_zero_rows_per_tile,
                                rtMemcpyDeviceToSram, ctx.dmaStream);
                copy_2d_tail_zero_async(sramB_super_zero_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                        NsSegValid, NsSegAlloc, super_zero_rows_per_tile, ctx.dmaStream);
            }

            rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
        };

        const int total_segments = nr_of_tiles * nr_of_ns;
        if (total_segments > 0) {
            schedule_dma(0, 0, segment_alloc_ns(0), segment_valid_ns(0), 0);

            for (int t = 0; t < total_segments; ++t) {
                const int i     = t / nr_of_ns;
                const int j     = t % nr_of_ns;
                const int NsSeg = segment_alloc_ns(j);
                const int ping  = t & 1;
                rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

                if (t + 1 < total_segments) {
                    const int next_t = t + 1;
                    const int next_i = next_t / nr_of_ns;
                    const int next_j = next_t % nr_of_ns;
                    schedule_dma(next_i, next_j, segment_alloc_ns(next_j), segment_valid_ns(next_j), next_t & 1);
                }

                launch_matmul_tile(i, j, NsSeg, sramB_wq_lsb_ping(ping), sramB_wq_msb_ping(ping),
                                   sramB_qscale_lsb_ping(ping), sramB_qzero_lsb_ping(ping), sramB_qscale_msb_ping(ping),
                                   sramB_qzero_msb_ping(ping), sramB_super_scale_ping(ping),
                                   sramB_super_zero_ping(ping));
                rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
            }
        }
    }

    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, NAlloc * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        copy_output_segments_async(devC, sramC1, nr_of_ns, Ns, NsTailValid, out_bytes_per_element,
                                   rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        copy_output_segments_async(devC, sramC, nr_of_ns, Ns, NsTailValid, out_bytes_per_element,
                                   rtMemcpySramToDevice, ctx.kernelStream);
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
    return true;
}

static void rpp_matmul_q5k_vxm_build(rpp_kernel_context & ctx,
                                     int                  M,
                                     int                  K,
                                     int                  N,
                                     int                  weights_group,
                                     int                  in_bytes_per_element,
                                     int                  out_bytes_per_element,
                                     int                  use_pipeline  = 0,
                                     int                  is_instantial = 1) {
    if (use_pipeline) {
        if (!rpp_matmul_q5k_vxm_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                         is_instantial)) {
            rpp_matmul_q5k_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                               is_instantial);
        }
    } else {
        rpp_matmul_q5k_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    }
}

}  // namespace kernel_q5_k_vxm
