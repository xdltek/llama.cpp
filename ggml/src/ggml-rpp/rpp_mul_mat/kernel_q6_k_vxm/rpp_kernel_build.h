// rpp_matmul_q6k_vxm.cpp (v1 overlap: dmaStream + kernelStream + pingpong SRAM)
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q6_k_vxm/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q6_k_vxm/rpp_kernel_param.h"
#include "rpp_runtime.h"

#include <assert.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_q6_k_vxm {

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
                                   RPPstream    stream) {
    const int min_chunk_bytes    = 32 * 1024;
    const int target_chunk_bytes = 1024 * 1024;

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
                           int & NsTail,
                           int & NsTailValid,
                           int & NAlloc,
                           bool  prefer_pipeline = false) {
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
        nr_of_tiles     = K / weights_group;
        groups_per_tile = 1;
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

    NsTailValid = NsTail;
    NAlloc      = N;

    // special handle for 151936:
    // non-pipeline: 2 x 77824 for minimal launch count.
    // pipeline: 4 x 40960 with a padded last segment so ping-pong B buffers
    // still fit SRAM while the tail keeps the same launch shape as full segments.
    if (N == 151936) {
        if (prefer_pipeline) {
            Ns          = 1280 * 32;
            nr_of_ns    = 4;
            NsTail      = Ns;
            NsTailValid = N - (nr_of_ns - 1) * Ns;
            NAlloc      = nr_of_ns * Ns;
        } else {
            Ns          = 2432 * 32;
            nr_of_ns    = 2;
            NsTail      = Ns;
            NsTailValid = N - Ns;
            NAlloc      = Ns + NsTail;
        }
    } else if (N == 200064) {
        Ns       = 600 * 32;
        nr_of_ns = 11;
        NsTail   = 252 * 32;
        NsTailValid = NsTail;
        NAlloc      = N;
    }

    assert(K == nr_of_tiles * weights_group * groups_per_tile);
    assert(N == ((nr_of_ns - 1) * Ns + NsTailValid));
    assert(NAlloc == ((nr_of_ns - 1) * Ns + NsTail));
    assert(NsTail % 32 == 0);
    assert(Ns % 32 == 0);
    assert(Ns >= NsTail);
    assert(NsTail >= NsTailValid);
}

inline bool q6k_vxm_pipeline_fits_sram(int K, int N, int weights_group, int in_bytes_per_element,
                                       int out_bytes_per_element) {
    int nr_of_tiles, groups_per_tile;
    int nr_of_ns, Ns, NsTail, NsTailValid, NAlloc;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail, NsTailValid, NAlloc,
                   true);

    const int group            = 16;
    const int bits_per_wqlsb   = 4;
    const int bits_per_wqmsb   = 2;
    const int bits_per_qscale  = 8;
    const int wqlsb_per_word   = sizeof(short) * 8 / bits_per_wqlsb;
    const int wqmsb_per_word   = sizeof(short) * 8 / bits_per_wqmsb;
    const int qscale_per_word  = sizeof(short) * 8 / bits_per_qscale;
    const int sizeA_raw        = K * in_bytes_per_element;
    const int sizeA_exec       = K * (int) sizeof(rpp::bfloat16);
    const int sizeB_wqlsb      = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqlsb_per_word);
    const int sizeB_wqmsb      = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqmsb_per_word);
    const int size_super_scale = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int size_qscale =
        (int) ((groups_per_tile * weights_group) * Ns / group * sizeof(short) / qscale_per_word);
    const int size_scale  = (int) ((groups_per_tile * weights_group) * Ns / group * sizeof(short));
    const int sizeC32     = NAlloc * sizeof(float);
    const int sizeCAlloc  = NAlloc * out_bytes_per_element;

    size_t total = 0;
    total += (in_bytes_per_element == (int) sizeof(float)) ? round_up(sizeA_raw) : 0;
    total += round_up(sizeA_exec);
    total += round_up(sizeC32);
    total += round_up(sizeCAlloc);

    const size_t one_ping = round_up(sizeB_wqlsb) + round_up(sizeB_wqmsb) + round_up(size_super_scale) +
                            round_up(size_qscale) + round_up(size_scale);
    total += one_ping * 2;
    total += round_up(weights_group * 4);

    const size_t SRAM_LIMIT = 22 * 1024 * 1024;
    return total <= SRAM_LIMIT;
}

inline size_t q6k_vxm_tail_zero_workspace_bytes(int Ktile, int NsTail, int NsTailValid, int wqlsb_per_word,
                                                int wqmsb_per_word, int group, int qscale_per_word,
                                                int weights_group) {
    const int pad_cols = NsTail - NsTailValid;
    if (pad_cols <= 0) {
        return 0;
    }

    const size_t wqlsb_bytes       = (size_t) pad_cols * sizeof(short) * (size_t) (Ktile / wqlsb_per_word);
    const size_t wqmsb_bytes       = (size_t) pad_cols * sizeof(short) * (size_t) (Ktile / wqmsb_per_word);
    const size_t qscale_bytes      = (size_t) pad_cols * sizeof(short) * (size_t) (Ktile / group / qscale_per_word);
    const size_t super_scale_bytes = (size_t) pad_cols * sizeof(short) * (size_t) (Ktile / weights_group);

    return std::max(std::max(wqlsb_bytes, wqmsb_bytes), std::max(qscale_bytes, super_scale_bytes));
}

inline RPPdeviceptr ensure_q6k_vxm_zero_workspace(rpp_kernel_context & ctx, size_t bytes) {
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
        throw std::runtime_error("failed to allocate q6k_vxm zero workspace");
    }
    if (rtMemset((void *) ctx.dev_aux_workspace, 0, bytes) != rtSuccess) {
        rtFree((void *) ctx.dev_aux_workspace);
        ctx.dev_aux_workspace       = 0;
        ctx.dev_aux_workspace_bytes = 0;
        throw std::runtime_error("failed to initialize q6k_vxm zero workspace");
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

    const int width_bytes = pad_cols * (int) sizeof(short);
    rtMemcpy2DAsync((void *) (dst_base + (size_t) valid_cols * sizeof(short)), dst_pitch, (const void *) zero_src_base,
                    width_bytes, width_bytes, height, rtMemcpyDeviceToSram, stream);
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_q6k_vxm(rpp_kernel_context & ctx,
                               int                  M,
                               int                  K,
                               int                  N,
                               int                  weights_group,
                               int                  in_bytes_per_element,
                               int                  out_bytes_per_element,
                               int                  is_instantial = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q4 Configure Parameter Invalid");
    }
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA             = ctx.dev_in[0];
    RPPdeviceptr          devB_wqlsb       = ctx.dev_in[1];
    RPPdeviceptr          devB_wqmsb       = ctx.dev_in[2];
    RPPdeviceptr          devB_scale       = ctx.dev_in[3];
    RPPdeviceptr          devB_super_scale = ctx.dev_in[4];
    RPPdeviceptr          devC             = ctx.dev_out[0];

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q6k_vxm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int logical_N = N;
    int       nr_of_tiles, groups_per_tile;
    int       nr_of_ns, Ns, NsTail, NsTailValid, NAlloc;
    get_tiles_info(logical_N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail, NsTailValid,
                   NAlloc, false);

    const int sizeA_raw  = K * in_bytes_per_element;
    const int sizeA_exec = K * (int) sizeof(rpp::bfloat16);

    //----------------------------------------------------------------------------------------------------
    // in_lsb        [K/4][N]
    // in_msb        [K/8][N]
    // in_scale      [K/16][N]
    // super_scale   [K/256][N]
    //-----------------------------------------------------------------------------------------------------
    int super_group         = 256;
    int group               = 16;
    int elements_per_thread = 16;
    int bits_per_wqlsb      = 4;
    int bits_per_wqmsb      = 2;
    int bits_per_qscale     = 8;
    int wqlsb_per_word      = sizeof(short) * 8 / bits_per_wqlsb;
    int wqmsb_per_word      = sizeof(short) * 8 / bits_per_wqmsb;
    int qscale_per_word     = sizeof(short) * 8 / bits_per_qscale;
    assert(super_group == weights_group);
    assert(group == elements_per_thread);

    const int sizeB_wqlsb      = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqlsb_per_word);
    const int sizeB_wqmsb      = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqmsb_per_word);
    const int size_super_scale = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int size_qscale = (int) ((groups_per_tile * weights_group) * Ns / group * sizeof(short) / qscale_per_word);
    const int size_scale  = (int) ((groups_per_tile * weights_group) * Ns / group * sizeof(short));
    const size_t tail_zero_workspace_bytes =
        q6k_vxm_tail_zero_workspace_bytes(groups_per_tile * weights_group, NsTail, NsTailValid, wqlsb_per_word,
                                          wqmsb_per_word, group, qscale_per_word, weights_group);
    const RPPdeviceptr zero_workspace = ensure_q6k_vxm_zero_workspace(ctx, tail_zero_workspace_bytes);
    // const int sizeZero = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeC32     = NAlloc * sizeof(float);
    const int sizeCAlloc  = NAlloc * out_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA_raw = sram_base;
    RPPdeviceptr sramA  = (in_bytes_per_element == (int) sizeof(float)) ? (sramA_raw + round_up(sizeA_raw)) : sramA_raw;
    RPPdeviceptr sramC  = sramA + round_up(sizeA_exec);
    RPPdeviceptr sramC1 = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_super_zero  = 0;
    RPPdeviceptr sramB_lut         = 0;
    RPPdeviceptr sramB_wqlsb       = sramC1 + round_up(sizeCAlloc);
    RPPdeviceptr sramB_wqmsb       = sramB_wqlsb + round_up(sizeB_wqlsb);
    RPPdeviceptr sramB_super_scale = sramB_wqmsb + round_up(sizeB_wqmsb);
    RPPdeviceptr sramB_qscale      = sramB_super_scale + round_up(size_super_scale);
    RPPdeviceptr sramB_scale       = sramB_qscale + round_up(size_qscale);
    RPPdeviceptr sramA_acc         = sramB_scale + round_up(size_scale);
    const int    total_sram_bytes  = (int) (sramA_acc + round_up(weights_group * 4) - sram_base);
    const int    SRAM_LIMIT        = 22 * 1024 * 1024;
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
        calc_tbdim_flattern(1, M * K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_raw, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    int Nx = K / weights_group;
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
    task_in_acc.params.kernelList.emplace_back(weights_group);
    launchWrapperAysnc(task_in_acc.taskName, task_in_acc.gridDim, task_in_acc.blockDim, task_in_acc.params.kernelList,
                       ctx.rppBinMod, ctx.kernelStream);

    // -------------------------
    // with given split factor, launch kernel one by one
    // Launch the norm Ns first
    // -------------------------
    int Ktile = groups_per_tile * weights_group;
    for (int i = 0; i < nr_of_tiles; i++) {
        for (int j = 0; j < nr_of_ns - 1; j++) {
            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_wqlsb, (const void *) devB_wqlsb, sizeB_wqlsb, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_wqmsb, (const void *) devB_wqmsb, sizeB_wqmsb, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_qscale, (const void *) devB_scale, size_qscale, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, size_super_scale,
                              rtMemcpyDeviceToSram, ctx.kernelStream);
                devB_wqlsb += sizeB_wqlsb;
                devB_wqmsb += sizeB_wqmsb;
                devB_scale += size_qscale;
                devB_super_scale += size_super_scale;
            } else {
                int wqlsb_stride       = (Ktile / wqlsb_per_word * logical_N * sizeof(short));
                int wqmsb_stride       = (Ktile / wqmsb_per_word * logical_N * sizeof(short));
                int qscale_stride      = (Ktile / group / qscale_per_word * logical_N * sizeof(short));
                int super_scale_stride = (Ktile / weights_group * logical_N * sizeof(short));
                rtMemcpy2DAsync((void *) sramB_wqlsb, Ns * sizeof(short),
                                (const char *) devB_wqlsb + i * wqlsb_stride + j * Ns * sizeof(short),
                                logical_N * sizeof(short), Ns * sizeof(short), Ktile / wqlsb_per_word,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_wqmsb, Ns * sizeof(short),
                                (const char *) devB_wqmsb + i * wqmsb_stride + j * Ns * sizeof(short),
                                logical_N * sizeof(short), Ns * sizeof(short), Ktile / wqmsb_per_word,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_qscale, Ns * sizeof(short),
                                (const char *) devB_scale + i * qscale_stride + j * Ns * sizeof(short),
                                logical_N * sizeof(short), Ns * sizeof(short), Ktile / group / qscale_per_word,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_super_scale, Ns * sizeof(short),
                                (const char *) devB_super_scale + i * super_scale_stride + j * Ns * sizeof(short),
                                logical_N * sizeof(short), Ns * sizeof(short), Ktile / weights_group,
                                rtMemcpyDeviceToSram, ctx.kernelStream);
            }

            // -------------------------
            // merge super_scale with scale
            // -------------------------
            params.clear();
            matmul_super_scale_blocks(Ktile, weights_group, group, Ns, threadsPerBlock, blocksPerGrid);
            matmul_super_scale_params(sramB_qscale, sramB_super_scale, sramB_scale, Ktile, Ns, weights_group, group,
                                      blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q6k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            if (i == 0) {
                combine = 0;
            } else {
                combine = 1;
            }

            RppTaskElement task;
            uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
            kerneldim_calc_matmul_linear(1, M, Ns, block_x, block_y, block_z, grid_x, grid_y, grid_z);
            task.blockDim.x = block_x;
            task.blockDim.y = block_y;
            task.blockDim.z = block_z;
            task.gridDim.x  = grid_x;
            task.gridDim.y  = grid_y;
            task.gridDim.z  = grid_z;

            task.gridDim.z          = groups_per_tile;
            uint32_t stride_ina     = weights_group * sizeof(short);
            uint32_t stride_inb     = weights_group * Ns;
            uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * sizeof(short);
            task.taskName           = "matrix_mul_vxM_f16_q6k_f16_asym_opt";
            task.params.kernelList.clear();
            // Use the actual matmul launch shape for blockXSize-sensitive params.
            dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
            dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
            matmul_weights_q6k_kernel_params(matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile,
                                             sramB_wqlsb, sramB_wqmsb, sramB_scale, sramC + j * Ns * sizeof(short),
                                             sramB_lut, sramB_super_zero, input_acc_addr, input_acc_addr + hilo_offset,
                                             Ns, logical_N * sizeof(short), weights_group, combine,
                                             task.params.kernelList);
            // jsq6k
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    // -------------------------
    // Launch with NsTail
    // -------------------------
    for (int i = 0; i < nr_of_tiles; i++) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_wqlsb, (const void *) devB_wqlsb, sizeB_wqlsb, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_wqmsb, (const void *) devB_wqmsb, sizeB_wqmsb, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_qscale, (const void *) devB_scale, size_qscale, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, size_super_scale,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            devB_wqlsb += sizeB_wqlsb;
            devB_wqmsb += sizeB_wqmsb;
            devB_scale += size_qscale;
            devB_super_scale += size_super_scale;
        } else {
            int wqlsb_stride       = (Ktile / wqlsb_per_word * logical_N * sizeof(short));
            int wqmsb_stride       = (Ktile / wqmsb_per_word * logical_N * sizeof(short));
            int qscale_stride      = (Ktile / group / qscale_per_word * logical_N * sizeof(short));
            int super_scale_stride = (Ktile / weights_group * logical_N * sizeof(short));
            rtMemcpy2DAsync((void *) sramB_wqlsb, NsTail * sizeof(short),
                            (const char *) devB_wqlsb + i * wqlsb_stride + (nr_of_ns - 1) * Ns * sizeof(short),
                            logical_N * sizeof(short), NsTailValid * sizeof(short), Ktile / wqlsb_per_word,
                            rtMemcpyDeviceToSram, ctx.kernelStream);
            copy_2d_tail_zero_async(sramB_wqlsb, NsTail * sizeof(short), zero_workspace, NsTailValid, NsTail,
                                    Ktile / wqlsb_per_word, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_wqmsb, NsTail * sizeof(short),
                            (const char *) devB_wqmsb + i * wqmsb_stride + (nr_of_ns - 1) * Ns * sizeof(short),
                            logical_N * sizeof(short), NsTailValid * sizeof(short), Ktile / wqmsb_per_word,
                            rtMemcpyDeviceToSram, ctx.kernelStream);
            copy_2d_tail_zero_async(sramB_wqmsb, NsTail * sizeof(short), zero_workspace, NsTailValid, NsTail,
                                    Ktile / wqmsb_per_word, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_qscale, NsTail * sizeof(short),
                            (const char *) devB_scale + i * qscale_stride + (nr_of_ns - 1) * Ns * sizeof(short),
                            logical_N * sizeof(short), NsTailValid * sizeof(short), Ktile / group / qscale_per_word,
                            rtMemcpyDeviceToSram, ctx.kernelStream);
            copy_2d_tail_zero_async(sramB_qscale, NsTail * sizeof(short), zero_workspace, NsTailValid, NsTail,
                                    Ktile / group / qscale_per_word, ctx.kernelStream);

            rtMemcpy2DAsync(
                (void *) sramB_super_scale, NsTail * sizeof(short),
                (const char *) devB_super_scale + i * super_scale_stride + (nr_of_ns - 1) * Ns * sizeof(short),
                logical_N * sizeof(short), NsTailValid * sizeof(short), Ktile / weights_group, rtMemcpyDeviceToSram,
                ctx.kernelStream);
            copy_2d_tail_zero_async(sramB_super_scale, NsTail * sizeof(short), zero_workspace, NsTailValid, NsTail,
                                    Ktile / weights_group, ctx.kernelStream);
        }

        // -------------------------
        // merge super_scale with scale
        // -------------------------
        params.clear();
        matmul_super_scale_blocks(Ktile, weights_group, group, NsTail, threadsPerBlock, blocksPerGrid);
        matmul_super_scale_params(sramB_qscale, sramB_super_scale, sramB_scale, Ktile, NsTail, weights_group, group,
                                  blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q6k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        if (i == 0) {
            combine = 0;
        } else {
            combine = 1;
        }

        RppTaskElement task;
        uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
        kerneldim_calc_matmul_linear(1, M, NsTail, block_x, block_y, block_z, grid_x, grid_y, grid_z);
        task.blockDim.x = block_x;
        task.blockDim.y = block_y;
        task.blockDim.z = block_z;
        task.gridDim.x  = grid_x;
        task.gridDim.y  = grid_y;
        task.gridDim.z  = grid_z;

        task.gridDim.z          = groups_per_tile;
        uint32_t stride_ina     = weights_group * sizeof(short);
        uint32_t stride_inb     = weights_group * NsTail;
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * sizeof(short);
        task.taskName           = "matrix_mul_vxM_f16_q6k_f16_asym_opt";
        task.params.kernelList.clear();
        // Use the actual matmul launch shape for blockXSize-sensitive params.
        dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q6k_kernel_params(
            matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile, sramB_wqlsb, sramB_wqmsb,
            sramB_scale, sramC + (nr_of_ns - 1) * Ns * sizeof(short), sramB_lut, sramB_super_zero, input_acc_addr,
            input_acc_addr + hilo_offset, NsTail, logical_N * sizeof(short), weights_group, combine,
            task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
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

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q6k_vxm_pipeline(rpp_kernel_context & ctx,
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
    RPPdeviceptr          devB_wqlsb       = ctx.dev_in[1];
    RPPdeviceptr          devB_wqmsb       = ctx.dev_in[2];
    RPPdeviceptr          devB_scale       = ctx.dev_in[3];
    RPPdeviceptr          devB_super_scale = ctx.dev_in[4];
    RPPdeviceptr          devC             = ctx.dev_out[0];

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q6k_vxm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    const int logical_N = N;
    int       nr_of_tiles, groups_per_tile;
    int       nr_of_ns, Ns, NsTail, NsTailValid, NAlloc;
    get_tiles_info(logical_N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail, NsTailValid,
                   NAlloc, true);

    const int sizeA_raw  = K * in_bytes_per_element;
    const int sizeA_exec = K * (int) sizeof(rpp::bfloat16);

    //----------------------------------------------------------------------------------------------------
    // in_lsb        [K/4][N]
    // in_msb        [K/8][N]
    // in_scale      [K/16][N]
    // super_scale   [K/256][N]
    //-----------------------------------------------------------------------------------------------------
    int super_group         = 256;
    int group               = 16;
    int elements_per_thread = 16;
    int bits_per_wqlsb      = 4;
    int bits_per_wqmsb      = 2;
    int bits_per_qscale     = 8;
    int wqlsb_per_word      = sizeof(short) * 8 / bits_per_wqlsb;
    int wqmsb_per_word      = sizeof(short) * 8 / bits_per_wqmsb;
    int qscale_per_word     = sizeof(short) * 8 / bits_per_qscale;
    assert(super_group == weights_group);
    assert(group == elements_per_thread);

    const int sizeB_wqlsb      = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqlsb_per_word);
    const int sizeB_wqmsb      = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqmsb_per_word);
    const int size_super_scale = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int size_qscale = (int) ((groups_per_tile * weights_group) * Ns / group * sizeof(short) / qscale_per_word);
    const int size_scale  = (int) ((groups_per_tile * weights_group) * Ns / group * sizeof(short));
    const size_t tail_zero_workspace_bytes =
        q6k_vxm_tail_zero_workspace_bytes(groups_per_tile * weights_group, NsTail, NsTailValid, wqlsb_per_word,
                                          wqmsb_per_word, group, qscale_per_word, weights_group);
    const RPPdeviceptr zero_workspace = ensure_q6k_vxm_zero_workspace(ctx, tail_zero_workspace_bytes);
    // const int sizeZero = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeC32     = NAlloc * sizeof(float);
    const int sizeCAlloc  = NAlloc * out_bytes_per_element;

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA_raw = sram_base;
    RPPdeviceptr sramA  = (in_bytes_per_element == (int) sizeof(float)) ? (sramA_raw + round_up(sizeA_raw)) : sramA_raw;
    RPPdeviceptr sramC  = sramA + round_up(sizeA_exec);
    RPPdeviceptr sramC1 = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_super_zero = 0;
    RPPdeviceptr sramB_lut        = 0;

    // ping-pong buffers for B
    RPPdeviceptr sramB_wqlsb_0       = sramC1 + round_up(sizeCAlloc);
    RPPdeviceptr sramB_wqmsb_0       = sramB_wqlsb_0 + round_up(sizeB_wqlsb);
    RPPdeviceptr sramB_super_scale_0 = sramB_wqmsb_0 + round_up(sizeB_wqmsb);
    RPPdeviceptr sramB_qscale_0      = sramB_super_scale_0 + round_up(size_super_scale);
    RPPdeviceptr sramB_scale_0       = sramB_qscale_0 + round_up(size_qscale);

    RPPdeviceptr sramB_wqlsb_1       = sramB_scale_0 + round_up(size_scale);
    RPPdeviceptr sramB_wqmsb_1       = sramB_wqlsb_1 + round_up(sizeB_wqlsb);
    RPPdeviceptr sramB_super_scale_1 = sramB_wqmsb_1 + round_up(sizeB_wqmsb);
    RPPdeviceptr sramB_qscale_1      = sramB_super_scale_1 + round_up(size_super_scale);
    RPPdeviceptr sramB_scale_1       = sramB_qscale_1 + round_up(size_qscale);

    RPPdeviceptr sramA_acc        = sramB_scale_1 + round_up(size_scale);
    const int    total_sram_bytes = (int) (sramA_acc + round_up(weights_group * 4) - sram_base);
    const int    SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    auto sramB_wqlsb_ping = [&](int ping) {
        return ping ? sramB_wqlsb_1 : sramB_wqlsb_0;
    };
    auto sramB_wqmsb_ping = [&](int ping) {
        return ping ? sramB_wqmsb_1 : sramB_wqmsb_0;
    };
    auto sramB_super_scale_ping = [&](int ping) {
        return ping ? sramB_super_scale_1 : sramB_super_scale_0;
    };
    auto sramB_qscale_ping = [&](int ping) {
        return ping ? sramB_qscale_1 : sramB_qscale_0;
    };
    auto sramB_scale_ping = [&](int ping) {
        return ping ? sramB_scale_1 : sramB_scale_0;
    };

    // -------------------------
    // (1) DDR -> SRAM async copies on kernelStream for A
    // -------------------------
    // Mark ping buffers as free at graph start
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);
    rtMemcpyAsync((void *) sramA_raw, (const void *) devA, sizeA_raw, rtMemcpyDeviceToSram, ctx.kernelStream);
    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, M * K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_raw, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    int Nx = K / weights_group;
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
    task_in_acc.params.kernelList.emplace_back(weights_group);
    launchWrapperAysnc(task_in_acc.taskName, task_in_acc.gridDim, task_in_acc.blockDim, task_in_acc.params.kernelList,
                       ctx.rppBinMod, ctx.kernelStream);

    // -------------------------
    // Pipeline B DMA (dmaStream) + kernel (kernelStream)
    // -------------------------
    int Ktile = groups_per_tile * weights_group;

    auto segment_alloc_ns = [&](int seg) {
        return (seg == nr_of_ns - 1) ? NsTail : Ns;
    };
    auto segment_valid_ns = [&](int seg) {
        return (seg == nr_of_ns - 1) ? NsTailValid : Ns;
    };

    auto schedule_dma = [&](int i, int seg, int NsSegAlloc, int NsSegValid, int ping) {
        // Ensure previous kernel using this ping buffer has finished
        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_wqlsb_ping(ping), (const void *) devB_wqlsb, sizeB_wqlsb, rtMemcpyDeviceToSram,
                          ctx.dmaStream);
            rtMemcpyAsync((void *) sramB_wqmsb_ping(ping), (const void *) devB_wqmsb, sizeB_wqmsb, rtMemcpyDeviceToSram,
                          ctx.dmaStream);
            rtMemcpyAsync((void *) sramB_qscale_ping(ping), (const void *) devB_scale, size_qscale,
                          rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpyAsync((void *) sramB_super_scale_ping(ping), (const void *) devB_super_scale, size_super_scale,
                          rtMemcpyDeviceToSram, ctx.dmaStream);
            devB_wqlsb += sizeB_wqlsb;
            devB_wqmsb += sizeB_wqmsb;
            devB_scale += size_qscale;
            devB_super_scale += size_super_scale;
        } else {
            int       wqlsb_stride       = (Ktile / wqlsb_per_word * logical_N * sizeof(short));
            int       wqmsb_stride       = (Ktile / wqmsb_per_word * logical_N * sizeof(short));
            int       qscale_stride      = (Ktile / group / qscale_per_word * logical_N * sizeof(short));
            int       super_scale_stride = (Ktile / weights_group * logical_N * sizeof(short));
            const int seg_offset         = seg * Ns * sizeof(short);

            memcpy2d_chunked_async(sramB_wqlsb_ping(ping), NsSegAlloc * sizeof(short),
                                   devB_wqlsb + i * wqlsb_stride + seg_offset, logical_N * sizeof(short),
                                   NsSegValid * sizeof(short), Ktile / wqlsb_per_word, rtMemcpyDeviceToSram,
                                   ctx.dmaStream);
            copy_2d_tail_zero_async(sramB_wqlsb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace, NsSegValid,
                                    NsSegAlloc, Ktile / wqlsb_per_word, ctx.dmaStream);

            memcpy2d_chunked_async(sramB_wqmsb_ping(ping), NsSegAlloc * sizeof(short),
                                   devB_wqmsb + i * wqmsb_stride + seg_offset, logical_N * sizeof(short),
                                   NsSegValid * sizeof(short), Ktile / wqmsb_per_word, rtMemcpyDeviceToSram,
                                   ctx.dmaStream);
            copy_2d_tail_zero_async(sramB_wqmsb_ping(ping), NsSegAlloc * sizeof(short), zero_workspace, NsSegValid,
                                    NsSegAlloc, Ktile / wqmsb_per_word, ctx.dmaStream);

            rtMemcpy2DAsync((void *) sramB_qscale_ping(ping), NsSegAlloc * sizeof(short),
                            (const char *) devB_scale + i * qscale_stride + seg_offset, logical_N * sizeof(short),
                            NsSegValid * sizeof(short), Ktile / group / qscale_per_word, rtMemcpyDeviceToSram,
                            ctx.dmaStream);
            copy_2d_tail_zero_async(sramB_qscale_ping(ping), NsSegAlloc * sizeof(short), zero_workspace, NsSegValid,
                                    NsSegAlloc, Ktile / group / qscale_per_word, ctx.dmaStream);

            rtMemcpy2DAsync((void *) sramB_super_scale_ping(ping), NsSegAlloc * sizeof(short),
                            (const char *) devB_super_scale + i * super_scale_stride + seg_offset,
                            logical_N * sizeof(short), NsSegValid * sizeof(short), Ktile / weights_group,
                            rtMemcpyDeviceToSram, ctx.dmaStream);
            copy_2d_tail_zero_async(sramB_super_scale_ping(ping), NsSegAlloc * sizeof(short), zero_workspace,
                                    NsSegValid, NsSegAlloc, Ktile / weights_group, ctx.dmaStream);
        }
        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    const int total_segments = nr_of_tiles * nr_of_ns;
    if (total_segments > 0) {
        schedule_dma(0, 0, segment_alloc_ns(0), segment_valid_ns(0), 0);

        for (int t = 0; t < total_segments; ++t) {
            const int i     = t / nr_of_ns;
            const int seg   = t % nr_of_ns;
            const int NsSeg = segment_alloc_ns(seg);
            const int ping  = t & 1;

            // wait for current DMA (ping-specific)
            rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

            // prefetch next segment
            if (t + 1 < total_segments) {
                const int next_t      = t + 1;
                const int next_i      = next_t / nr_of_ns;
                const int next_seg    = next_t % nr_of_ns;
                const int next_NsSeg  = segment_alloc_ns(next_seg);
                const int next_NsValid = segment_valid_ns(next_seg);
                const int next_ping   = next_t & 1;
                schedule_dma(next_i, next_seg, next_NsSeg, next_NsValid, next_ping);
            }

            // merge super_scale with scale
            params.clear();
            matmul_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            matmul_super_scale_params(sramB_qscale_ping(ping), sramB_super_scale_ping(ping), sramB_scale_ping(ping),
                                      Ktile, NsSeg, weights_group, group, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q6k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            if (i == 0) {
                combine = 0;
            } else {
                combine = 1;
            }

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
            uint32_t stride_inb     = weights_group * NsSeg;
            uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * sizeof(short);
            task.taskName           = "matrix_mul_vxM_f16_q6k_f16_asym_opt";
            task.params.kernelList.clear();
            // Use the actual matmul launch shape for blockXSize-sensitive params.
            dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
            dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
            matmul_weights_q6k_kernel_params(matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile,
                                             sramB_wqlsb_ping(ping), sramB_wqmsb_ping(ping), sramB_scale_ping(ping),
                                             sramC + seg * Ns * sizeof(short), sramB_lut, sramB_super_zero,
                                             input_acc_addr, input_acc_addr + hilo_offset, NsSeg,
                                             logical_N * sizeof(short), weights_group, combine,
                                             task.params.kernelList);
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
            // Signal this ping buffer is safe to reuse
            rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
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

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q6k_vxm_build(rpp_kernel_context & ctx,
                                     int                  M,
                                     int                  K,
                                     int                  N,
                                     int                  weights_group,
                                     int                  in_bytes_per_element,
                                     int                  out_bytes_per_element,
                                     int                  use_pipeline  = 0,
                                     int                  is_instantial = 1) {
    if (use_pipeline &&
        !q6k_vxm_pipeline_fits_sram(K, N, weights_group, in_bytes_per_element, out_bytes_per_element)) {
        use_pipeline = 0;
    }

    if (use_pipeline) {
        rpp_matmul_q6k_vxm_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                    is_instantial);
    } else {
        rpp_matmul_q6k_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    }
}
}  // namespace kernel_q6_k_vxm
