#pragma once
// rpp_matmul_q6k_vxm.cpp (v1 overlap: dmaStream + kernelStream + pingpong SRAM)

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q4_k_vxm/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q4_k_vxm/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_q4_k_vxm {

inline int get_q4k_vxm_ns_safe_limit() {
    // Default to no clamp now that the q4k super-scale blockX path is fixed.
    // Keep the env override so the old segmentation can be restored quickly if
    // another shape regresses.
    const char * env = std::getenv("GGML_Q4K_VXM_NS_SAFE");
    if (!env || env[0] == '\0') {
        return 0;
    }

    const int requested = std::atoi(env);
    if (requested <= 0) {
        return 0;
    }

    if (requested < 32) {
        return 32;
    }

    return (requested / 32) * 32;
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

    // Optional q4k N-segmentation clamp. Disabled by default now that the
    // scale kernel's internal blockX loop is fixed.
    //   GGML_Q4K_VXM_NS_SAFE=0     -> disable the clamp
    //   GGML_Q4K_VXM_NS_SAFE=4096  -> restore the old conservative behavior
    //   GGML_Q4K_VXM_NS_SAFE=12288 -> force this case to stay in one segment
    const int NS_SAFE = get_q4k_vxm_ns_safe_limit();
    if (NS_SAFE > 0 && Ns > NS_SAFE) {
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

inline int get_q4k_fullk_scale_chunk_k(int K, int N, int weights_group, int group) {
    if (K <= 0 || N <= 0 || weights_group <= 0 || (K % weights_group) != 0) {
        return 0;
    }

    for (int chunk_k = K; chunk_k >= weights_group; chunk_k -= weights_group) {
        if ((K % chunk_k) != 0) {
            continue;
        }

        dim3 chunk_threads;
        dim3 chunk_blocks;
        q4k_super_scale_blocks(chunk_k, weights_group, group, N, chunk_threads, chunk_blocks);
        if (chunk_blocks.x == 1) {
            return chunk_k;
        }
    }

    return 0;
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_q4k_vxm(rpp_kernel_context & ctx,
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
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    int                   full_pitch, sub_pitch, offset;
    RPPdeviceptr          devA             = ctx.dev_in[0];
    RPPdeviceptr          devB_wq          = ctx.dev_in[1];
    RPPdeviceptr          devB_qscale_lsb  = ctx.dev_in[2];
    RPPdeviceptr          devB_qzero_lsb   = ctx.dev_in[3];
    RPPdeviceptr          devB_qscale_msb  = ctx.dev_in[4];
    RPPdeviceptr          devB_qzero_msb   = ctx.dev_in[5];
    RPPdeviceptr          devB_super_scale = ctx.dev_in[6];
    RPPdeviceptr          devB_super_zero  = ctx.dev_in[7];
    RPPdeviceptr          devC             = ctx.dev_out[0];

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q4k_vxm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    int nr_of_tiles, groups_per_tile;
    int nr_of_ns, Ns, NsTail;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);

    const int sizeA_in        = K * in_bytes_per_element;
    const int sizeA           = K * (int) sizeof(short);
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // q_group = 32
    // wq_per_group = 32
    // in_wq     [K/256]      |  [8]  |  [32/4]    |  [4][N]
    // in_wq     [z]          |  [y]  |  [unroll]  |  [x]
    //           [grid.y]*[z] |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    const int q_group         = 32;
    const int super_group     = 256;
    const int group           = super_group / q_group;
    int       bits_per_wqlsb  = 4;
    int       bits_per_wqmsb  = 2;
    int       bits_per_qscale = 8;
    int       wqlsb_per_word  = sizeof(short) * 8 / bits_per_wqlsb;
    int       wqmsb_per_word  = sizeof(short) * 8 / bits_per_wqmsb;
    int       qscale_per_word = sizeof(short) * 8 / bits_per_qscale;
    assert(super_group == weights_group);
    const int sizeB_wq                = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqlsb_per_word);
    const int sizeB_qscale_lsb_tile   = (sizeB_wq / q_group);
    const int sizeB_qscale_msb_tile   = sizeB_qscale_lsb_tile / 2;
    const int sizeB_qzero_lsb_tile    = sizeB_qscale_lsb_tile;
    const int sizeB_qzero_msb_tile    = sizeB_qscale_msb_tile;
    const int sizeB_super_scale_tile  = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int sizeB_super_zero_tile   = sizeB_super_scale_tile;
    const int sizeB_qscale_lsb_full   = (int) (K * Ns * sizeof(short) / wqlsb_per_word / q_group);
    const int sizeB_qscale_msb_full   = sizeB_qscale_lsb_full / 2;
    const int sizeB_qzero_lsb_full    = sizeB_qscale_lsb_full;
    const int sizeB_qzero_msb_full    = sizeB_qscale_msb_full;
    const int sizeB_super_scale_full  = (int) (K * Ns / weights_group * sizeof(short));
    const int sizeB_super_zero_full   = sizeB_super_scale_full;
    const size_t sizeB_full_aux_bytes = (size_t) sizeB_qscale_lsb_full + (size_t) sizeB_qzero_lsb_full +
                                        (size_t) sizeB_qscale_msb_full + (size_t) sizeB_qzero_msb_full +
                                        (size_t) sizeB_super_scale_full + (size_t) sizeB_super_zero_full;
    const int  sizeB_scale                 = (groups_per_tile * weights_group * Ns / q_group) * (int) sizeof(short);
    const int  sizeB_zero                  = sizeB_scale;
    const int  sizeB_scale_full            = (K * Ns / q_group) * (int) sizeof(short);
    const int  sizeB_zero_full             = sizeB_scale_full;
    const int  fullk_scale_chunk_k         = get_q4k_fullk_scale_chunk_k(K, N, weights_group, group);
    const bool fullk_scale_shape_supported = (M == 1 && N <= 1024 && K <= 4096 && fullk_scale_chunk_k > 0);
    // step1/step2 currently use one tile-local [group][N] temp slab, so they
    // are only valid when a K tile contains exactly one weights_group block.
    const bool step_shape_supported        = (M == 1 && N <= 1024 && groups_per_tile == 1);
    int        step_kernel_mode            = -1;  // auto
    if (const char * env = std::getenv("GGML_Q4K_VXM_USE_STEP12")) {
        if (env[0] == '0') {
            step_kernel_mode = 0;
        } else if (env[0] == '1') {
            step_kernel_mode = 1;
        }
    }

    bool use_step_kernels = step_shape_supported;
    if (step_kernel_mode == 0) {
        use_step_kernels = false;
    } else if (step_kernel_mode == 1) {
        if (!step_shape_supported) {
            throw std::runtime_error("GGML_Q4K_VXM_USE_STEP12=1 requires M==1, N<=1024, and groups_per_tile==1");
        }
        use_step_kernels = true;
    }
    const int sizeStep1Temp = groups_per_tile * group * Ns * (int) sizeof(short);

    int Nx = K / q_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    const int hilo_offset = (Nx * Nz + 31) / 32 * 32 * sizeof(short);
    const int sizeAAcc    = 2 * hilo_offset;

    // const int sizeZero = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeC32         = N * sizeof(float);
    const int sizeC           = N * out_bytes_per_element;
    const int SRAM_LIMIT      = 22 * 1024 * 1024;
    auto      calc_total_sram = [&](int sizeB_qscale_lsb_bytes, int sizeB_qzero_lsb_bytes, int sizeB_qscale_msb_bytes,
                               int sizeB_qzero_msb_bytes, int sizeB_super_scale_bytes, int sizeB_super_zero_bytes,
                               bool fullk_scale) -> int {
        return round_up(sizeA_in) + round_up(sizeA) + round_up(sizeC32) + round_up(sizeC) + round_up(sizeB_wq) +
               round_up(sizeB_qscale_lsb_bytes) + round_up(sizeB_qzero_lsb_bytes) + round_up(sizeB_qscale_msb_bytes) +
               round_up(sizeB_qzero_msb_bytes) + round_up(sizeB_super_scale_bytes) + round_up(sizeB_super_zero_bytes) +
               round_up(fullk_scale ? sizeB_scale_full : sizeB_scale) +
               round_up(fullk_scale ? sizeB_zero_full : sizeB_zero) + round_up(sizeAAcc) +
               (use_step_kernels ? 2 * round_up(sizeStep1Temp) : 0);
    };
    const int total_sram_bytes_tiled =
        calc_total_sram(sizeB_qscale_lsb_tile, sizeB_qzero_lsb_tile, sizeB_qscale_msb_tile, sizeB_qzero_msb_tile,
                        sizeB_super_scale_tile, sizeB_super_zero_tile, false);
    const int total_sram_bytes_fullk_aux =
        calc_total_sram(sizeB_qscale_lsb_full, sizeB_qzero_lsb_full, sizeB_qscale_msb_full, sizeB_qzero_msb_full,
                        sizeB_super_scale_full, sizeB_super_zero_full, false);
    const int total_sram_bytes_fullk_aux_fullscale =
        calc_total_sram(sizeB_qscale_lsb_full, sizeB_qzero_lsb_full, sizeB_qscale_msb_full, sizeB_qzero_msb_full,
                        sizeB_super_scale_full, sizeB_super_zero_full, true);
    const bool use_fullk_scale =
        fullk_scale_shape_supported && (nr_of_ns == 1) && (total_sram_bytes_fullk_aux_fullscale <= SRAM_LIMIT);
    const bool use_fullk_aux          = use_fullk_scale || (total_sram_bytes_fullk_aux <= SRAM_LIMIT);
    const int  sizeB_qscale_lsb_sram  = use_fullk_aux ? sizeB_qscale_lsb_full : sizeB_qscale_lsb_tile;
    const int  sizeB_qzero_lsb_sram   = use_fullk_aux ? sizeB_qzero_lsb_full : sizeB_qzero_lsb_tile;
    const int  sizeB_qscale_msb_sram  = use_fullk_aux ? sizeB_qscale_msb_full : sizeB_qscale_msb_tile;
    const int  sizeB_qzero_msb_sram   = use_fullk_aux ? sizeB_qzero_msb_full : sizeB_qzero_msb_tile;
    const int  sizeB_super_scale_sram = use_fullk_aux ? sizeB_super_scale_full : sizeB_super_scale_tile;
    const int  sizeB_super_zero_sram  = use_fullk_aux ? sizeB_super_zero_full : sizeB_super_zero_tile;
    const int  sizeB_scale_sram       = use_fullk_scale ? sizeB_scale_full : sizeB_scale;
    const int  sizeB_zero_sram        = use_fullk_scale ? sizeB_zero_full : sizeB_zero;
    const int  total_sram_bytes       = use_fullk_scale ?
                                            total_sram_bytes_fullk_aux_fullscale :
                                            (use_fullk_aux ? total_sram_bytes_fullk_aux : total_sram_bytes_tiled);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;
    RPPdeviceptr sramA_in  = sram_base;
    RPPdeviceptr sramA     = sramA_in + round_up(sizeA_in);
    RPPdeviceptr sramC     = sramA + round_up(sizeA);
    RPPdeviceptr sramC1    = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_wq  = sramC1 + round_up(sizeC);

    RPPdeviceptr sramB_qscale_lsb = sramB_wq + round_up(sizeB_wq);
    RPPdeviceptr sramB_qzero_lsb  = sramB_qscale_lsb + round_up(sizeB_qscale_lsb_sram);
    RPPdeviceptr sramB_qscale_msb = sramB_qzero_lsb + round_up(sizeB_qzero_lsb_sram);
    RPPdeviceptr sramB_qzero_msb  = sramB_qscale_msb + round_up(sizeB_qscale_msb_sram);

    RPPdeviceptr       sramB_super_scale      = sramB_qzero_msb + round_up(sizeB_qzero_msb_sram);
    RPPdeviceptr       sramB_super_zero       = sramB_super_scale + round_up(sizeB_super_scale_sram);
    RPPdeviceptr       sramB_scale            = sramB_super_zero + round_up(sizeB_super_zero_sram);
    RPPdeviceptr       sramB_zero             = sramB_scale + round_up(sizeB_scale_sram);
    RPPdeviceptr       sramA_acc              = sramB_zero + round_up(sizeB_zero_sram);
    const int          sizeStep1TempAligned   = use_step_kernels ? round_up(sizeStep1Temp) : 0;
    const RPPdeviceptr sramStep1Temp          = use_step_kernels ? (sramA_acc + round_up(sizeAAcc)) : 0;
    const uint32_t     step1_temp_hilo_offset = use_step_kernels ? sizeStep1TempAligned : 0;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }
    (void) sramStep1Temp;
    (void) step1_temp_hilo_offset;
    // -------------------------
    // (1) DDR -> SRAM async copies on dmaStream
    // -------------------------
    rtMemcpyAsync((void *) (in_bytes_per_element == (int) sizeof(float) ? sramA_in : sramA), (const void *) devA,
                  sizeA_in, rtMemcpyDeviceToSram, ctx.kernelStream);
    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, M * K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

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
    const int wq_stride                 = (Ktile / 4 * N * sizeof(short));
    const int qscale_lsb_stride         = wq_stride / q_group;
    const int qscale_msb_stride         = qscale_lsb_stride / 2;
    const int qzero_lsb_stride          = qscale_lsb_stride;
    const int qzero_msb_stride          = qscale_msb_stride;
    const int super_scale_stride        = Ktile / weights_group * N * sizeof(short);
    const int super_zero_stride         = super_scale_stride;
    const int wq_rows_per_tile          = Ktile / 4;
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
    const int scale_rows_per_tile       = Ktile / q_group;
    const int zero_rows_per_tile        = scale_rows_per_tile;

    auto copy_wq_tile = [&](int i, int j, int NsSeg) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_wq, (const void *) (devB_wq + i * sizeB_wq), sizeB_wq, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            return;
        }

        const int src_col_offset = j * Ns * sizeof(short);
        rtMemcpy2DAsync((void *) sramB_wq, NsSeg * sizeof(short),
                        (const void *) (devB_wq + i * wq_stride + src_col_offset), N * sizeof(short),
                        NsSeg * sizeof(short), wq_rows_per_tile, rtMemcpyDeviceToSram, ctx.kernelStream);
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

    auto aux_row_ptr = [&](RPPdeviceptr base, int row_offset, int NsSeg) -> RPPdeviceptr {
        return base + static_cast<RPPdeviceptr>(row_offset) * NsSeg * sizeof(short);
    };

    auto launch_fullk_scale_chunks = [&](int NsSeg) {
        if (fullk_scale_chunk_k <= 0 || (K % fullk_scale_chunk_k) != 0) {
            throw std::runtime_error("invalid q4k full-K super-scale chunking");
        }

        for (int chunk_k_off = 0; chunk_k_off < K; chunk_k_off += fullk_scale_chunk_k) {
            const int qscale_lsb_row_off  = chunk_k_off / 4 / q_group;
            const int qscale_msb_row_off  = qscale_lsb_row_off / 2;
            const int qzero_lsb_row_off   = qscale_lsb_row_off;
            const int qzero_msb_row_off   = qscale_msb_row_off;
            const int super_scale_row_off = chunk_k_off / weights_group;
            const int super_zero_row_off  = super_scale_row_off;
            const int scale_row_off       = chunk_k_off / q_group;
            const int zero_row_off        = scale_row_off;

            q4k_super_scale_blocks(fullk_scale_chunk_k, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            params.clear();
            q4k_super_scale_params(aux_row_ptr(sramB_qscale_lsb, qscale_lsb_row_off, NsSeg),
                                   aux_row_ptr(sramB_qscale_msb, qscale_msb_row_off, NsSeg),
                                   aux_row_ptr(sramB_super_scale, super_scale_row_off, NsSeg),
                                   aux_row_ptr(sramB_scale, scale_row_off, NsSeg), fullk_scale_chunk_k, NsSeg,
                                   weights_group, q_group, 0, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            q4k_super_scale_blocks(fullk_scale_chunk_k, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            params.clear();
            q4k_super_scale_params(aux_row_ptr(sramB_qzero_lsb, qzero_lsb_row_off, NsSeg),
                                   aux_row_ptr(sramB_qzero_msb, qzero_msb_row_off, NsSeg),
                                   aux_row_ptr(sramB_super_zero, super_zero_row_off, NsSeg),
                                   aux_row_ptr(sramB_zero, zero_row_off, NsSeg), fullk_scale_chunk_k, NsSeg,
                                   weights_group, q_group, 1, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    };

    for (int j = 0; j < nr_of_ns; ++j) {
        const int NsSeg = (j == nr_of_ns - 1) ? NsTail : Ns;
        if (use_fullk_aux) {
            // Keep non-Wq metadata resident for the whole K range of the current Ns slice.
            copy_aux_segment_fullk(j, NsSeg);
            if (use_fullk_scale) {
                launch_fullk_scale_chunks(NsSeg);
            }
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
            const RPPdeviceptr sramB_scale_tile =
                use_fullk_scale ? aux_tile_ptr(sramB_scale, scale_rows_per_tile, i, NsSeg) : sramB_scale;
            const RPPdeviceptr sramB_zero_tile =
                use_fullk_scale ? aux_tile_ptr(sramB_zero, zero_rows_per_tile, i, NsSeg) : sramB_zero;

            if (!use_fullk_scale) {
                q4k_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
                params.clear();
                q4k_super_scale_params(sramB_qscale_lsb_tile, sramB_qscale_msb_tile, sramB_super_scale_tile,
                                       sramB_scale_tile, Ktile, NsSeg, weights_group, q_group, 0, blocksPerGrid,
                                       threadsPerBlock, params);
                launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);

                q4k_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
                params.clear();
                q4k_super_scale_params(sramB_qzero_lsb_tile, sramB_qzero_msb_tile, sramB_super_zero_tile,
                                       sramB_zero_tile, Ktile, NsSeg, weights_group, q_group, 1, blocksPerGrid,
                                       threadsPerBlock, params);
                launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }

            combine = (i == 0) ? 0 : 1;

            uint32_t stride_ina     = weights_group * sizeof(short);
            //---------------------------------------------------------------------------------------
            // InputAcc [K/256]      | [8]
            //         [grid.z]      | [loopidx]
            // accIdx = blockIdx.z * 8 + loopidx
            //----------------------------------------------------------------------------------------
            uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * sizeof(short);
            if (use_step_kernels) {
                RppTaskElement step1_task;
                uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
                kerneldim_calc_matmul_q4k_step1(groups_per_tile, NsSeg, block_x, block_y, block_z, grid_x, grid_y,
                                                grid_z);
                step1_task.blockDim.x = block_x;
                step1_task.blockDim.y = block_y;
                step1_task.blockDim.z = block_z;
                step1_task.gridDim.x  = grid_x;
                step1_task.gridDim.y  = grid_y;
                step1_task.gridDim.z  = grid_z;
                step1_task.taskName   = "matrix_mul_vxM_q4k_step1";
                step1_task.params.kernelList.clear();
                dim3 step1Threads(step1_task.blockDim.x, step1_task.blockDim.y, step1_task.blockDim.z);
                dim3 step1Blocks(step1_task.gridDim.x, step1_task.gridDim.y, step1_task.gridDim.z);
                matmul_weights_q4k_step1_kernel_params(
                    step1Blocks, step1Threads, sramA + i * stride_ina * groups_per_tile, sramB_wq, sramStep1Temp,
                    sramStep1Temp + step1_temp_hilo_offset, input_acc_addr, input_acc_addr + hilo_offset, NsSeg,
                    step1_temp_hilo_offset, weights_group, combine, step1_task.params.kernelList);
                launchWrapperAysnc(step1_task.taskName, step1_task.gridDim, step1_task.blockDim,
                                   step1_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

                RppTaskElement step2_task;
                kerneldim_calc_matmul_q4k_step2(groups_per_tile, NsSeg, block_x, block_y, block_z, grid_x, grid_y,
                                                grid_z);
                step2_task.blockDim.x = block_x;
                step2_task.blockDim.y = block_y;
                step2_task.blockDim.z = block_z;
                step2_task.gridDim.x  = grid_x;
                step2_task.gridDim.y  = grid_y;
                step2_task.gridDim.z  = grid_z;
                step2_task.taskName   = "matrix_mul_vxM_q4k_step2";
                step2_task.params.kernelList.clear();
                dim3 step2Threads(step2_task.blockDim.x, step2_task.blockDim.y, step2_task.blockDim.z);
                dim3 step2Blocks(step2_task.gridDim.x, step2_task.gridDim.y, step2_task.gridDim.z);
                matmul_weights_q4k_step2_kernel_params(
                    step2Blocks, step2Threads, sramStep1Temp, sramStep1Temp + step1_temp_hilo_offset, sramB_scale_tile,
                    sramB_zero_tile, sramC + j * Ns * sizeof(short), input_acc_addr, input_acc_addr + hilo_offset,
                    NsSeg, N * sizeof(short), weights_group, combine, step2_task.params.kernelList);
                launchWrapperAysnc(step2_task.taskName, step2_task.gridDim, step2_task.blockDim,
                                   step2_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
            } else {
                RppTaskElement task;
                uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
                kerneldim_calc_matmul_linear(1, M, NsSeg, block_x, block_y, block_z, grid_x, grid_y, grid_z);
                task.blockDim.x = block_x;
                task.blockDim.y = block_y;
                task.blockDim.z = block_z;
                task.gridDim.x  = grid_x;
                task.gridDim.y  = grid_y;
                task.gridDim.z  = grid_z;

                task.gridDim.z = groups_per_tile;
                task.taskName  = "matrix_mul_vxM_f16_q4k_f16_asym_opt";
                task.params.kernelList.clear();
                dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
                dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
                matmul_weights_q4k_kernel_params(
                    matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile, sramB_wq, sramB_scale_tile,
                    sramB_zero_tile, sramC + j * Ns * sizeof(short), 0, input_acc_addr, input_acc_addr + hilo_offset,
                    NsSeg, N * sizeof(short), weights_group, combine, task.params.kernelList);
                launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
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

static bool rpp_matmul_q4k_vxm_pipeline(rpp_kernel_context & ctx,
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
    const RPPdeviceptr devB_wq_base          = ctx.dev_in[1];
    const RPPdeviceptr devB_qscale_lsb_base  = ctx.dev_in[2];
    const RPPdeviceptr devB_qzero_lsb_base   = ctx.dev_in[3];
    const RPPdeviceptr devB_qscale_msb_base  = ctx.dev_in[4];
    const RPPdeviceptr devB_qzero_msb_base   = ctx.dev_in[5];
    const RPPdeviceptr devB_super_scale_base = ctx.dev_in[6];
    const RPPdeviceptr devB_super_zero_base  = ctx.dev_in[7];
    RPPdeviceptr       devC                  = ctx.dev_out[0];

    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q4k_vxm.o");

    int nr_of_tiles, groups_per_tile;
    int nr_of_ns, Ns, NsTail;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);

    const int sizeA_in        = K * in_bytes_per_element;
    const int sizeA           = K * (int) sizeof(short);
    const int q_group         = 32;
    const int super_group     = 256;
    const int group           = super_group / q_group;
    int       bits_per_wqlsb  = 4;
    int       bits_per_qscale = 8;
    int       wqlsb_per_word  = sizeof(short) * 8 / bits_per_wqlsb;
    int       qscale_per_word = sizeof(short) * 8 / bits_per_qscale;
    (void) qscale_per_word;
    assert(super_group == weights_group);

    const int sizeB_wq                = (int) ((groups_per_tile * weights_group) * Ns * sizeof(short) / wqlsb_per_word);
    const int sizeB_qscale_lsb_tile   = (sizeB_wq / q_group);
    const int sizeB_qscale_msb_tile   = sizeB_qscale_lsb_tile / 2;
    const int sizeB_qzero_lsb_tile    = sizeB_qscale_lsb_tile;
    const int sizeB_qzero_msb_tile    = sizeB_qscale_msb_tile;
    const int sizeB_super_scale_tile  = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int sizeB_super_zero_tile   = sizeB_super_scale_tile;
    const int sizeB_qscale_lsb_full   = (int) (K * Ns * sizeof(short) / wqlsb_per_word / q_group);
    const int sizeB_qscale_msb_full   = sizeB_qscale_lsb_full / 2;
    const int sizeB_qzero_lsb_full    = sizeB_qscale_lsb_full;
    const int sizeB_qzero_msb_full    = sizeB_qscale_msb_full;
    const int sizeB_super_scale_full  = (int) (K * Ns / weights_group * sizeof(short));
    const int sizeB_super_zero_full   = sizeB_super_scale_full;
    const size_t sizeB_full_aux_bytes = (size_t) sizeB_qscale_lsb_full + (size_t) sizeB_qzero_lsb_full +
                                        (size_t) sizeB_qscale_msb_full + (size_t) sizeB_qzero_msb_full +
                                        (size_t) sizeB_super_scale_full + (size_t) sizeB_super_zero_full;
    const int sizeB_scale      = (groups_per_tile * weights_group * Ns / q_group) * (int) sizeof(short);
    const int sizeB_zero       = sizeB_scale;
    const int sizeB_scale_full = (K * Ns / q_group) * (int) sizeof(short);
    const int sizeB_zero_full  = sizeB_scale_full;

    const bool step_shape_supported = (M == 1 && N <= 1024 && groups_per_tile == 1);
    int        step_kernel_mode     = -1;
    if (const char * env = std::getenv("GGML_Q4K_VXM_USE_STEP12")) {
        if (env[0] == '0') {
            step_kernel_mode = 0;
        } else if (env[0] == '1') {
            step_kernel_mode = 1;
        }
    }

    bool use_step_kernels = step_shape_supported;
    if (step_kernel_mode == 0) {
        use_step_kernels = false;
    } else if (step_kernel_mode == 1) {
        if (!step_shape_supported) {
            throw std::runtime_error("GGML_Q4K_VXM_USE_STEP12=1 requires M==1, N<=1024, and groups_per_tile==1");
        }
        use_step_kernels = true;
    }

    const int sizeStep1Temp = groups_per_tile * group * Ns * (int) sizeof(short);

    int Nx = K / q_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    const int hilo_offset = (Nx * Nz + 31) / 32 * 32 * sizeof(short);
    const int sizeAAcc    = 2 * hilo_offset;

    const int sizeC32    = N * sizeof(float);
    const int sizeC      = N * out_bytes_per_element;
    const int SRAM_LIMIT = 22 * 1024 * 1024;

    const int  fullk_scale_chunk_k         = get_q4k_fullk_scale_chunk_k(K, N, weights_group, group);
    const bool fullk_scale_shape_supported = (M == 1 && N <= 1024 && K <= 4096 && fullk_scale_chunk_k > 0);

    auto calc_pipeline_total_sram = [&](bool fullk_aux, bool fullk_scale) -> int {
        const int qscale_lsb_bytes = fullk_aux ? round_up(sizeB_qscale_lsb_full) : 2 * round_up(sizeB_qscale_lsb_tile);
        const int qzero_lsb_bytes  = fullk_aux ? round_up(sizeB_qzero_lsb_full) : 2 * round_up(sizeB_qzero_lsb_tile);
        const int qscale_msb_bytes = fullk_aux ? round_up(sizeB_qscale_msb_full) : 2 * round_up(sizeB_qscale_msb_tile);
        const int qzero_msb_bytes  = fullk_aux ? round_up(sizeB_qzero_msb_full) : 2 * round_up(sizeB_qzero_msb_tile);
        const int super_scale_bytes =
            fullk_aux ? round_up(sizeB_super_scale_full) : 2 * round_up(sizeB_super_scale_tile);
        const int super_zero_bytes = fullk_aux ? round_up(sizeB_super_zero_full) : 2 * round_up(sizeB_super_zero_tile);
        const int scale_bytes      = round_up(fullk_scale ? sizeB_scale_full : sizeB_scale);
        const int zero_bytes       = round_up(fullk_scale ? sizeB_zero_full : sizeB_zero);

        return round_up(sizeA_in) + round_up(sizeA) + round_up(sizeC32) + round_up(sizeC) +
               2 * round_up(sizeB_wq) + qscale_lsb_bytes +
               qzero_lsb_bytes + qscale_msb_bytes + qzero_msb_bytes + super_scale_bytes + super_zero_bytes +
               scale_bytes + zero_bytes + round_up(sizeAAcc) + (use_step_kernels ? 2 * round_up(sizeStep1Temp) : 0);
    };

    const int  total_sram_bytes_fullk_aux           = calc_pipeline_total_sram(true, false);
    const int  total_sram_bytes_fullk_aux_fullscale = calc_pipeline_total_sram(true, true);
    const int  total_sram_bytes_tiled               = calc_pipeline_total_sram(false, false);
    const bool use_fullk_scale =
        fullk_scale_shape_supported && (nr_of_ns == 1) && (total_sram_bytes_fullk_aux_fullscale <= SRAM_LIMIT);
    const bool use_fullk_aux    = (nr_of_ns == 1) && (use_fullk_scale || (total_sram_bytes_fullk_aux <= SRAM_LIMIT));
    const int  sizeB_scale_sram = use_fullk_scale ? sizeB_scale_full : sizeB_scale;
    const int  sizeB_zero_sram  = use_fullk_scale ? sizeB_zero_full : sizeB_zero;
    const int  total_sram_bytes = use_fullk_scale ?
                                      total_sram_bytes_fullk_aux_fullscale :
                                      (use_fullk_aux ? total_sram_bytes_fullk_aux : total_sram_bytes_tiled);
    if (total_sram_bytes > SRAM_LIMIT) {
        return false;
    }

    RPPdeviceptr sram_base  = ctx.virtual_sram_base;
    RPPdeviceptr sramA_in   = sram_base;
    RPPdeviceptr sramA      = sramA_in + round_up(sizeA_in);
    RPPdeviceptr sramC      = sramA + round_up(sizeA);
    RPPdeviceptr sramC1     = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_wq_0 = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_wq_1 = sramB_wq_0 + round_up(sizeB_wq);
    RPPdeviceptr cursor     = sramB_wq_1 + round_up(sizeB_wq);

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
        sramB_qzero_lsb   = sramB_qscale_lsb + sizeB_qscale_lsb_full;
        sramB_qscale_msb  = sramB_qzero_lsb + sizeB_qzero_lsb_full;
        sramB_qzero_msb   = sramB_qscale_msb + sizeB_qscale_msb_full;
        sramB_super_scale = sramB_qzero_msb + sizeB_qzero_msb_full;
        sramB_super_zero  = sramB_super_scale + sizeB_super_scale_full;
        cursor            = sramB_qscale_lsb + round_up((int) sizeB_full_aux_bytes);
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

    RPPdeviceptr       sramB_scale            = cursor;
    RPPdeviceptr       sramB_zero             = sramB_scale + round_up(sizeB_scale_sram);
    RPPdeviceptr       sramA_acc              = sramB_zero + round_up(sizeB_zero_sram);
    const int          sizeStep1TempAligned   = use_step_kernels ? round_up(sizeStep1Temp) : 0;
    const RPPdeviceptr sramStep1Temp          = use_step_kernels ? (sramA_acc + round_up(sizeAAcc)) : 0;
    const uint32_t     step1_temp_hilo_offset = use_step_kernels ? sizeStep1TempAligned : 0;

    const bool src_full_aux_contiguous =
        ((size_t) devB_qzero_lsb_base == (size_t) devB_qscale_lsb_base + (size_t) sizeB_qscale_lsb_full) &&
        ((size_t) devB_qscale_msb_base == (size_t) devB_qzero_lsb_base + (size_t) sizeB_qzero_lsb_full) &&
        ((size_t) devB_qzero_msb_base == (size_t) devB_qscale_msb_base + (size_t) sizeB_qscale_msb_full) &&
        ((size_t) devB_super_scale_base == (size_t) devB_qzero_msb_base + (size_t) sizeB_qzero_msb_full) &&
        ((size_t) devB_super_zero_base == (size_t) devB_super_scale_base + (size_t) sizeB_super_scale_full);
    const bool enable_packed_full_aux_dma = []() -> bool {
        const char * env = std::getenv("GGML_Q4K_VXM_PACKED_AUX_DMA");
        return !env || std::atoi(env) != 0;
    }();
    const bool use_packed_full_aux_dma = enable_packed_full_aux_dma && src_full_aux_contiguous;

    // std::cout << "[Q4K_VXM_PIPELINE] "
    //           << "use_fullk_aux=" << (use_fullk_aux ? 1 : 0) << " use_fullk_scale=" << (use_fullk_scale ? 1 : 0)
    //           << " packed_aux_dma=" << (use_packed_full_aux_dma ? 1 : 0)
    //           << " src_full_aux_contiguous=" << (src_full_aux_contiguous ? 1 : 0) << " nr_of_tiles=" << nr_of_tiles
    //           << " nr_of_ns=" << nr_of_ns << " use_step12=" << (use_step_kernels ? 1 : 0) << "\n";

    auto sramB_wq_ping = [&](int ping) -> RPPdeviceptr {
        return ping ? sramB_wq_1 : sramB_wq_0;
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

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);

    rtMemcpyAsync((void *) (in_bytes_per_element == (int) sizeof(float) ? sramA_in : sramA), (const void *) devA,
                  sizeA_in, rtMemcpyDeviceToSram, ctx.kernelStream);
    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, M * K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

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

    int       Ktile                     = groups_per_tile * weights_group;
    const int wq_stride                 = (Ktile / 4 * N * sizeof(short));
    const int qscale_lsb_stride         = wq_stride / q_group;
    const int qscale_msb_stride         = qscale_lsb_stride / 2;
    const int qzero_lsb_stride          = qscale_lsb_stride;
    const int qzero_msb_stride          = qscale_msb_stride;
    const int super_scale_stride        = Ktile / weights_group * N * sizeof(short);
    const int super_zero_stride         = super_scale_stride;
    const int wq_rows_per_tile          = Ktile / 4;
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
    const int scale_rows_per_tile       = Ktile / q_group;
    const int zero_rows_per_tile        = scale_rows_per_tile;

    auto segment_ns = [&](int seg) -> int {
        return (seg == nr_of_ns - 1) ? NsTail : Ns;
    };

    auto aux_tile_ptr = [&](RPPdeviceptr base, int rows_per_tile, int i, int NsSeg) -> RPPdeviceptr {
        return base + static_cast<RPPdeviceptr>(i) * rows_per_tile * NsSeg * sizeof(short);
    };

    auto aux_row_ptr = [&](RPPdeviceptr base, int row_offset, int NsSeg) -> RPPdeviceptr {
        return base + static_cast<RPPdeviceptr>(row_offset) * NsSeg * sizeof(short);
    };

    auto launch_fullk_scale_chunks = [&](int NsSeg) {
        if (fullk_scale_chunk_k <= 0 || (K % fullk_scale_chunk_k) != 0) {
            throw std::runtime_error("invalid q4k full-K super-scale chunking");
        }

        for (int chunk_k_off = 0; chunk_k_off < K; chunk_k_off += fullk_scale_chunk_k) {
            const int qscale_lsb_row_off  = chunk_k_off / 4 / q_group;
            const int qscale_msb_row_off  = qscale_lsb_row_off / 2;
            const int qzero_lsb_row_off   = qscale_lsb_row_off;
            const int qzero_msb_row_off   = qscale_msb_row_off;
            const int super_scale_row_off = chunk_k_off / weights_group;
            const int super_zero_row_off  = super_scale_row_off;
            const int scale_row_off       = chunk_k_off / q_group;
            const int zero_row_off        = scale_row_off;

            q4k_super_scale_blocks(fullk_scale_chunk_k, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            params.clear();
            q4k_super_scale_params(aux_row_ptr(sramB_qscale_lsb, qscale_lsb_row_off, NsSeg),
                                   aux_row_ptr(sramB_qscale_msb, qscale_msb_row_off, NsSeg),
                                   aux_row_ptr(sramB_super_scale, super_scale_row_off, NsSeg),
                                   aux_row_ptr(sramB_scale, scale_row_off, NsSeg), fullk_scale_chunk_k, NsSeg,
                                   weights_group, q_group, 0, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            q4k_super_scale_blocks(fullk_scale_chunk_k, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            params.clear();
            q4k_super_scale_params(aux_row_ptr(sramB_qzero_lsb, qzero_lsb_row_off, NsSeg),
                                   aux_row_ptr(sramB_qzero_msb, qzero_msb_row_off, NsSeg),
                                   aux_row_ptr(sramB_super_zero, super_zero_row_off, NsSeg),
                                   aux_row_ptr(sramB_zero, zero_row_off, NsSeg), fullk_scale_chunk_k, NsSeg,
                                   weights_group, q_group, 1, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    };

    auto launch_matmul_tile = [&](int i, int j, int NsSeg, RPPdeviceptr sramB_wq_tile,
                                  RPPdeviceptr sramB_qscale_lsb_tile, RPPdeviceptr sramB_qzero_lsb_tile,
                                  RPPdeviceptr sramB_qscale_msb_tile, RPPdeviceptr sramB_qzero_msb_tile,
                                  RPPdeviceptr sramB_super_scale_tile, RPPdeviceptr sramB_super_zero_tile,
                                  RPPdeviceptr sramB_scale_tile, RPPdeviceptr sramB_zero_tile, bool run_super_scale) {
        if (run_super_scale) {
            q4k_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            params.clear();
            q4k_super_scale_params(sramB_qscale_lsb_tile, sramB_qscale_msb_tile, sramB_super_scale_tile,
                                   sramB_scale_tile, Ktile, NsSeg, weights_group, q_group, 0, blocksPerGrid,
                                   threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            q4k_super_scale_blocks(Ktile, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
            params.clear();
            q4k_super_scale_params(sramB_qzero_lsb_tile, sramB_qzero_msb_tile, sramB_super_zero_tile, sramB_zero_tile,
                                   Ktile, NsSeg, weights_group, q_group, 1, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q4k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        int      combine        = (i == 0) ? 0 : 1;
        uint32_t stride_ina     = weights_group * sizeof(short);
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * sizeof(short);

        if (use_step_kernels) {
            RppTaskElement step1_task;
            uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
            kerneldim_calc_matmul_q4k_step1(groups_per_tile, NsSeg, block_x, block_y, block_z, grid_x, grid_y, grid_z);
            step1_task.blockDim.x = block_x;
            step1_task.blockDim.y = block_y;
            step1_task.blockDim.z = block_z;
            step1_task.gridDim.x  = grid_x;
            step1_task.gridDim.y  = grid_y;
            step1_task.gridDim.z  = grid_z;
            step1_task.taskName   = "matrix_mul_vxM_q4k_step1";
            step1_task.params.kernelList.clear();
            dim3 step1Threads(step1_task.blockDim.x, step1_task.blockDim.y, step1_task.blockDim.z);
            dim3 step1Blocks(step1_task.gridDim.x, step1_task.gridDim.y, step1_task.gridDim.z);
            matmul_weights_q4k_step1_kernel_params(
                step1Blocks, step1Threads, sramA + i * stride_ina * groups_per_tile, sramB_wq_tile, sramStep1Temp,
                sramStep1Temp + step1_temp_hilo_offset, input_acc_addr, input_acc_addr + hilo_offset, NsSeg,
                step1_temp_hilo_offset, weights_group, combine, step1_task.params.kernelList);
            launchWrapperAysnc(step1_task.taskName, step1_task.gridDim, step1_task.blockDim,
                               step1_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);

            RppTaskElement step2_task;
            kerneldim_calc_matmul_q4k_step2(groups_per_tile, NsSeg, block_x, block_y, block_z, grid_x, grid_y, grid_z);
            step2_task.blockDim.x = block_x;
            step2_task.blockDim.y = block_y;
            step2_task.blockDim.z = block_z;
            step2_task.gridDim.x  = grid_x;
            step2_task.gridDim.y  = grid_y;
            step2_task.gridDim.z  = grid_z;
            step2_task.taskName   = "matrix_mul_vxM_q4k_step2";
            step2_task.params.kernelList.clear();
            dim3 step2Threads(step2_task.blockDim.x, step2_task.blockDim.y, step2_task.blockDim.z);
            dim3 step2Blocks(step2_task.gridDim.x, step2_task.gridDim.y, step2_task.gridDim.z);
            matmul_weights_q4k_step2_kernel_params(
                step2Blocks, step2Threads, sramStep1Temp, sramStep1Temp + step1_temp_hilo_offset, sramB_scale_tile,
                sramB_zero_tile, sramC + j * Ns * sizeof(short), input_acc_addr, input_acc_addr + hilo_offset, NsSeg,
                N * sizeof(short), weights_group, combine, step2_task.params.kernelList);
            launchWrapperAysnc(step2_task.taskName, step2_task.gridDim, step2_task.blockDim,
                               step2_task.params.kernelList, ctx.rppBinMod, ctx.kernelStream);
        } else {
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
            task.taskName   = "matrix_mul_vxM_f16_q4k_f16_asym_opt";
            task.params.kernelList.clear();
            dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
            dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
            matmul_weights_q4k_kernel_params(
                matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile, sramB_wq_tile, sramB_scale_tile,
                sramB_zero_tile, sramC + j * Ns * sizeof(short), 0, input_acc_addr, input_acc_addr + hilo_offset, NsSeg,
                N * sizeof(short), weights_group, combine, task.params.kernelList);
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    };

    if (use_fullk_aux) {
        auto copy_aux_segment_fullk_dma = [&](int j, int NsSeg) {
            if (nr_of_ns == 1) {
                if (use_packed_full_aux_dma) {
                    rtMemcpyAsync((void *) sramB_qscale_lsb, (const void *) devB_qscale_lsb_base, sizeB_full_aux_bytes,
                                  rtMemcpyDeviceToSram, ctx.dmaStream);
                } else {
                    rtMemcpyAsync((void *) sramB_qscale_lsb, (const void *) devB_qscale_lsb_base, sizeB_qscale_lsb_full,
                                  rtMemcpyDeviceToSram, ctx.dmaStream);
                    rtMemcpyAsync((void *) sramB_qzero_lsb, (const void *) devB_qzero_lsb_base, sizeB_qzero_lsb_full,
                                  rtMemcpyDeviceToSram, ctx.dmaStream);
                    rtMemcpyAsync((void *) sramB_qscale_msb, (const void *) devB_qscale_msb_base, sizeB_qscale_msb_full,
                                  rtMemcpyDeviceToSram, ctx.dmaStream);
                    rtMemcpyAsync((void *) sramB_qzero_msb, (const void *) devB_qzero_msb_base, sizeB_qzero_msb_full,
                                  rtMemcpyDeviceToSram, ctx.dmaStream);
                    rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale_base,
                                  sizeB_super_scale_full, rtMemcpyDeviceToSram, ctx.dmaStream);
                    rtMemcpyAsync((void *) sramB_super_zero, (const void *) devB_super_zero_base, sizeB_super_zero_full,
                                  rtMemcpyDeviceToSram, ctx.dmaStream);
                }
                return;
            }

            const int src_col_offset = j * Ns * sizeof(short);
            rtMemcpy2DAsync((void *) sramB_qscale_lsb, NsSeg * sizeof(short),
                            (const void *) (devB_qscale_lsb_base + src_col_offset), N * sizeof(short),
                            NsSeg * sizeof(short), qscale_lsb_rows_full, rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpy2DAsync((void *) sramB_qzero_lsb, NsSeg * sizeof(short),
                            (const void *) (devB_qzero_lsb_base + src_col_offset), N * sizeof(short),
                            NsSeg * sizeof(short), qzero_lsb_rows_full, rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpy2DAsync((void *) sramB_qscale_msb, NsSeg * sizeof(short),
                            (const void *) (devB_qscale_msb_base + src_col_offset), N * sizeof(short),
                            NsSeg * sizeof(short), qscale_msb_rows_full, rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpy2DAsync((void *) sramB_qzero_msb, NsSeg * sizeof(short),
                            (const void *) (devB_qzero_msb_base + src_col_offset), N * sizeof(short),
                            NsSeg * sizeof(short), qzero_msb_rows_full, rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpy2DAsync((void *) sramB_super_scale, NsSeg * sizeof(short),
                            (const void *) (devB_super_scale_base + src_col_offset), N * sizeof(short),
                            NsSeg * sizeof(short), super_scale_rows_full, rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpy2DAsync((void *) sramB_super_zero, NsSeg * sizeof(short),
                            (const void *) (devB_super_zero_base + src_col_offset), N * sizeof(short),
                            NsSeg * sizeof(short), super_zero_rows_full, rtMemcpyDeviceToSram, ctx.dmaStream);
        };

        auto schedule_wq_dma = [&](int i, int j, int NsSeg, int ping) {
            rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_wq_ping(ping), (const void *) (devB_wq_base + i * sizeB_wq), sizeB_wq,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
            } else {
                const int src_col_offset = j * Ns * sizeof(short);
                rtMemcpy2DAsync((void *) sramB_wq_ping(ping), NsSeg * sizeof(short),
                                (const void *) (devB_wq_base + i * wq_stride + src_col_offset), N * sizeof(short),
                                NsSeg * sizeof(short), wq_rows_per_tile, rtMemcpyDeviceToSram, ctx.dmaStream);
            }

            rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
        };

        const int segment_tail_ping = (nr_of_tiles > 0) ? ((nr_of_tiles - 1) & 1) : 0;
        for (int j = 0; j < nr_of_ns; ++j) {
            const int NsSeg = segment_ns(j);
            rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[segment_tail_ping], 0);
            copy_aux_segment_fullk_dma(j, NsSeg);
            schedule_wq_dma(0, j, NsSeg, 0);
            bool prefetched_tile1 = false;
            if (use_fullk_scale && nr_of_tiles > 1) {
                schedule_wq_dma(1, j, NsSeg, 1);
                prefetched_tile1 = true;
            }

            for (int i = 0; i < nr_of_tiles; ++i) {
                const int ping = i & 1;
                rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

                if (use_fullk_scale && i == 0) {
                    launch_fullk_scale_chunks(NsSeg);
                }

                if (i + 1 < nr_of_tiles) {
                    if (!(use_fullk_scale && i == 0 && prefetched_tile1)) {
                        schedule_wq_dma(i + 1, j, NsSeg, (i + 1) & 1);
                    }
                }

                launch_matmul_tile(
                    i, j, NsSeg, sramB_wq_ping(ping),
                    aux_tile_ptr(sramB_qscale_lsb, qscale_lsb_rows_per_tile, i, NsSeg),
                    aux_tile_ptr(sramB_qzero_lsb, qzero_lsb_rows_per_tile, i, NsSeg),
                    aux_tile_ptr(sramB_qscale_msb, qscale_msb_rows_per_tile, i, NsSeg),
                    aux_tile_ptr(sramB_qzero_msb, qzero_msb_rows_per_tile, i, NsSeg),
                    aux_tile_ptr(sramB_super_scale, super_scale_rows_per_tile, i, NsSeg),
                    aux_tile_ptr(sramB_super_zero, super_zero_rows_per_tile, i, NsSeg),
                    use_fullk_scale ? aux_tile_ptr(sramB_scale, scale_rows_per_tile, i, NsSeg) : sramB_scale,
                    use_fullk_scale ? aux_tile_ptr(sramB_zero, zero_rows_per_tile, i, NsSeg) : sramB_zero,
                    !use_fullk_scale);

                rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
            }
        }
    } else {
        auto schedule_dma = [&](int i, int j, int NsSeg, int ping) {
            rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);

            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_wq_ping(ping), (const void *) (devB_wq_base + i * sizeB_wq), sizeB_wq,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
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
                rtMemcpy2DAsync((void *) sramB_wq_ping(ping), NsSeg * sizeof(short),
                                (const void *) (devB_wq_base + i * wq_stride + src_col_offset), N * sizeof(short),
                                NsSeg * sizeof(short), wq_rows_per_tile, rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpy2DAsync((void *) sramB_qscale_lsb_ping(ping), NsSeg * sizeof(short),
                                (const void *) (devB_qscale_lsb_base + i * qscale_lsb_stride + src_col_offset),
                                N * sizeof(short), NsSeg * sizeof(short), qscale_lsb_rows_per_tile,
                                rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpy2DAsync((void *) sramB_qzero_lsb_ping(ping), NsSeg * sizeof(short),
                                (const void *) (devB_qzero_lsb_base + i * qzero_lsb_stride + src_col_offset),
                                N * sizeof(short), NsSeg * sizeof(short), qzero_lsb_rows_per_tile, rtMemcpyDeviceToSram,
                                ctx.dmaStream);
                rtMemcpy2DAsync((void *) sramB_qscale_msb_ping(ping), NsSeg * sizeof(short),
                                (const void *) (devB_qscale_msb_base + i * qscale_msb_stride + src_col_offset),
                                N * sizeof(short), NsSeg * sizeof(short), qscale_msb_rows_per_tile,
                                rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpy2DAsync((void *) sramB_qzero_msb_ping(ping), NsSeg * sizeof(short),
                                (const void *) (devB_qzero_msb_base + i * qzero_msb_stride + src_col_offset),
                                N * sizeof(short), NsSeg * sizeof(short), qzero_msb_rows_per_tile, rtMemcpyDeviceToSram,
                                ctx.dmaStream);
                rtMemcpy2DAsync((void *) sramB_super_scale_ping(ping), NsSeg * sizeof(short),
                                (const void *) (devB_super_scale_base + i * super_scale_stride + src_col_offset),
                                N * sizeof(short), NsSeg * sizeof(short), super_scale_rows_per_tile,
                                rtMemcpyDeviceToSram, ctx.dmaStream);
                rtMemcpy2DAsync((void *) sramB_super_zero_ping(ping), NsSeg * sizeof(short),
                                (const void *) (devB_super_zero_base + i * super_zero_stride + src_col_offset),
                                N * sizeof(short), NsSeg * sizeof(short), super_zero_rows_per_tile,
                                rtMemcpyDeviceToSram, ctx.dmaStream);
            }

            rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
        };

        const int total_segments = nr_of_ns * nr_of_tiles;
        if (total_segments > 0) {
            schedule_dma(0, 0, segment_ns(0), 0);

            for (int t = 0; t < total_segments; ++t) {
                const int j     = t / nr_of_tiles;
                const int i     = t % nr_of_tiles;
                const int NsSeg = segment_ns(j);
                const int ping  = t & 1;
                rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

                if (t + 1 < total_segments) {
                    const int next_t = t + 1;
                    const int next_j = next_t / nr_of_tiles;
                    const int next_i = next_t % nr_of_tiles;
                    schedule_dma(next_i, next_j, segment_ns(next_j), next_t & 1);
                }

                launch_matmul_tile(i, j, NsSeg, sramB_wq_ping(ping), sramB_qscale_lsb_ping(ping),
                                   sramB_qzero_lsb_ping(ping), sramB_qscale_msb_ping(ping), sramB_qzero_msb_ping(ping),
                                   sramB_super_scale_ping(ping), sramB_super_zero_ping(ping), sramB_scale, sramB_zero,
                                   true);
                rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
            }
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

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
    return true;
}

static void rpp_matmul_q4k_vxm_build(rpp_kernel_context & ctx,
                                     int                  M,
                                     int                  K,
                                     int                  N,
                                     int                  weights_group,
                                     int                  in_bytes_per_element,
                                     int                  out_bytes_per_element,
                                     int                  use_pipeline  = 0,
                                     int                  is_instantial = 1) {
    if (use_pipeline) {
        rpp_matmul_q4k_vxm_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                    is_instantial);
    } else {
        rpp_matmul_q4k_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
    }
}
}  // namespace kernel_q4_k_vxm
