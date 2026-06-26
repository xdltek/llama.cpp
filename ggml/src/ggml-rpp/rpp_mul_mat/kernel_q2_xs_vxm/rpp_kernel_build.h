#pragma once
// rpp_matmul_q2xs_vxm.cpp (v1 overlap: dmaStream + kernelStream + pingpong SRAM)

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q2_xs_vxm/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q2_xs_vxm/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <array>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_q2_xs_vxm {
static inline uint8_t q2xs_reverse_bits_u8(uint8_t x) {
    x = (uint8_t) (((x & 0xF0u) >> 4) | ((x & 0x0Fu) << 4));
    x = (uint8_t) (((x & 0xCCu) >> 2) | ((x & 0x33u) << 2));
    x = (uint8_t) (((x & 0xAAu) >> 1) | ((x & 0x55u) << 1));
    return x;
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
        // q2xs path (weights_group != 32)
        nr_of_tiles     = K / weights_group;
        groups_per_tile = 1;
        // nr_of_tiles = 1;
        // groups_per_tile = K / weights_group;
    }
    // q2xs path
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

static void q2xs_prepare_lookup_tables(rpp_kernel_context & ctx,
                                       int                  qscale_lut_bytes,
                                       int                  codebook_lut_bytes,
                                       int                  sign_lut_bytes,
                                       RPPdeviceptr &       devB_qscale_lut,
                                       RPPdeviceptr &       devB_codebook_lut,
                                       RPPdeviceptr &       devB_sign_lut) {
    auto align_up = [](size_t x, size_t a) -> size_t {
        return ((x + a - 1) / a) * a;
    };

    const size_t off_qscale_lut   = 0;
    const size_t off_codebook_lut = align_up(off_qscale_lut + (size_t) qscale_lut_bytes, 64);
    const size_t off_sign_lut     = align_up(off_codebook_lut + (size_t) codebook_lut_bytes, 64);
    const size_t workspace_bytes  = align_up(off_sign_lut + (size_t) sign_lut_bytes, 64);

    // Workspace holds q2xs LUTs generated on host:
    // qscale LUT (for super-scale), codebook LUT (iq2xs_grid), and sign LUT (reversed-bit ksigns_iq2xs).
    // Reuse the shared workspace to avoid invalidating pointers captured by other kernels.
    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        if (rtMalloc((void **) &dev_lut_workspace, workspace_bytes) != rtSuccess) {
            throw std::runtime_error("Q2XS rtMalloc failed for LUT workspace");
        }
        ctx.dev_workspace = dev_lut_workspace;
    }

    RPPdeviceptr base = dev_lut_workspace;
    devB_qscale_lut   = base + (RPPdeviceptr) off_qscale_lut;
    devB_codebook_lut = base + (RPPdeviceptr) off_codebook_lut;
    devB_sign_lut     = base + (RPPdeviceptr) off_sign_lut;

    std::array<uint16_t, 16> qscale_lut = {};
    for (int i = 0; i < (int) qscale_lut.size(); ++i) {
        const float lut_val = (0.5f + (float) i) * 0.25f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }

    std::array<uint64_t, 512> codebook_lut = {};
    // q2xs uses its dedicated 512-entry iq2xs grid table (same packed 8x-u8 format).
    std::memcpy(codebook_lut.data(), iq2xs_grid_local, sizeof(codebook_lut));

    std::array<uint8_t, 128> sign_lut = {};
    for (int i = 0; i < (int) sign_lut.size(); ++i) {
        sign_lut[i] = q2xs_reverse_bits_u8(ksigns_iq2xs_local[i]);
    }

    rtMemcpy((void *) devB_qscale_lut, qscale_lut.data(), qscale_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_codebook_lut, codebook_lut.data(), codebook_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_sign_lut, sign_lut.data(), sign_lut_bytes, rtMemcpyHostToDevice);
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_q2xs_vxm(rpp_kernel_context & ctx,
                                int                  M,
                                int                  K,
                                int                  N,
                                int                  weights_group,
                                int                  in_bytes_per_element,
                                int                  out_bytes_per_element,
                                int                  is_instantial = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q2XS Configure Paramter Invalid");
    }
    if (ctx.dev_in.size() < 4 || ctx.dev_out.empty()) {
        throw std::runtime_error("Matmul Q2XS requires 4 inputs (A, qs, scales, super_scale) and 1 output");
    }
    if (weights_group != 256 || (K % 256) != 0) {
        throw std::runtime_error("Matmul Q2XS requires weights_group == 256 and K % 256 == 0");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    const int             qscale_lut_elems    = 16;
    const int             codebook_lut_elems  = 512;
    const int             sign_lut_elems      = 128;
    const int             qscale_lut_bytes    = qscale_lut_elems * (int) sizeof(uint16_t);
    const int             codebook_lut_bytes  = codebook_lut_elems * (int) sizeof(uint64_t);
    const int             sign_lut_bytes      = sign_lut_elems * (int) sizeof(uint8_t);
    const int             lut_workspace_bytes = qscale_lut_bytes + codebook_lut_bytes + sign_lut_bytes;
    (void) lut_workspace_bytes;

    RPPdeviceptr devA              = ctx.dev_in[0];
    RPPdeviceptr devB_qs           = ctx.dev_in[1];
    RPPdeviceptr devB_scales       = ctx.dev_in[2];
    RPPdeviceptr devB_super_scale  = ctx.dev_in[3];
    RPPdeviceptr devC              = ctx.dev_out[0];
    RPPdeviceptr devB_qscale_lut   = 0;
    RPPdeviceptr devB_codebook_lut = 0;
    RPPdeviceptr devB_sign_lut     = 0;
    q2xs_prepare_lookup_tables(ctx, qscale_lut_bytes, codebook_lut_bytes, sign_lut_bytes, devB_qscale_lut,
                               devB_codebook_lut, devB_sign_lut);

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    // Native q2xs-vxm kernels + q2xs local block/param helpers.
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q2xs_vxm.o");
    // -------------------------
    // SRAM allocation planning
    // -------------------------
    int nr_of_tiles, groups_per_tile;
    int nr_of_ns, Ns, NsTail;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);

    const int sizeA_in    = K * in_bytes_per_element;
    // HW workaround: keep CVT_32_16 out-of-place (raw input -> bf16 working buffer).
    const int sizeA       = K * (int) sizeof(short);
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    // group = 8
    // q_group = 32
    // wq_per_group = 32
    // in_wq     [K/256]      |  [8]  |  [32/4]    |  [4][N]
    // in_wq     [z]          |  [y]  |  [unroll]  |  [x]
    //           [grid.y]*[z] |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    const int q_group     = 32;
    const int super_group = 256;
    const int group       = super_group / q_group;
    assert(super_group == weights_group);

    const int Ktile              = groups_per_tile * weights_group;
    const int sizeB_qs           = (Ktile * Ns / 4);
    const int sizeB_scales       = (Ktile * Ns / 32);
    const int sizeB_super_scale  = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int sizeB_qscale_lut   = qscale_lut_bytes;
    const int sizeB_codebook_lut = codebook_lut_bytes;
    const int sizeB_sign_lut     = sign_lut_bytes;
    // Q2XS scale tensor after super-scale is one bf16 per 16 weights.
    const int sizeB_scale        = (groups_per_tile * weights_group * Ns / 16) * (int) sizeof(short);

    // const int sizeZero = groups_per_tile * N * sizeof(rpp::bfloat16);
    const int sizeC32 = N * sizeof(float);
    const int sizeC   = N * out_bytes_per_element;

    RPPdeviceptr sram_base          = ctx.virtual_sram_base;
    RPPdeviceptr sramA_in           = sram_base;
    RPPdeviceptr sramA              = sramA_in + round_up(sizeA_in);
    RPPdeviceptr sramC              = sramA + round_up(sizeA);
    RPPdeviceptr sramC1             = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_qs           = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_scales       = sramB_qs + round_up(sizeB_qs);
    RPPdeviceptr sramB_super_scale  = sramB_scales + round_up(sizeB_scales);
    RPPdeviceptr sramB_qscale_lut   = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_codebook_lut = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_sign_lut     = sramB_codebook_lut + round_up(sizeB_codebook_lut);
    RPPdeviceptr sramB_scale        = sramB_sign_lut + round_up(sizeB_sign_lut);
    RPPdeviceptr sramA_acc          = sramB_scale + round_up(sizeB_scale);
    const int    total_sram_bytes   = (int) (sramA_acc + round_up(weights_group * 4) - sram_base);
    const int    SRAM_LIMIT         = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }
    // -------------------------
    // (1) DDR -> SRAM async copies on dmaStream
    // -------------------------
    // CDMA D2S
    rtMemcpyAsync((void *) (in_bytes_per_element == (int) sizeof(float) ? sramA_in : sramA), (const void *) devA,
                  sizeA_in, rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_qscale_lut, (const void *) devB_qscale_lut, sizeB_qscale_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_codebook_lut, (const void *) devB_codebook_lut, sizeB_codebook_lut,
                  rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_sign_lut, (const void *) devB_sign_lut, sizeB_sign_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    if (in_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA, kFLOAT, kBF16, params);
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
    // with given split factor, launch kernel one by one
    // Launch the norm Ns first
    // -------------------------
    for (int i = 0; i < nr_of_tiles; i++) {
        for (int j = 0; j < nr_of_ns - 1; j++) {
            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_qs, (const void *) devB_qs, sizeB_qs, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_scales, (const void *) devB_scales, sizeB_scales, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale,
                              rtMemcpyDeviceToSram, ctx.kernelStream);

                devB_qs += sizeB_qs;
                devB_scales += sizeB_scales;
                devB_super_scale += sizeB_super_scale;
            } else {
                int       qs_stride          = (Ktile / 8 * N * (int) sizeof(short));
                int       scales_stride      = (Ktile / 64 * N * (int) sizeof(short));
                int       super_scale_stride = Ktile / weights_group * N * (int) sizeof(short);
                const int tile_ns_offset     = j * Ns * (int) sizeof(short);

                rtMemcpy2DAsync((void *) sramB_qs, Ns * (int) sizeof(short),
                                (const void *) (devB_qs + i * qs_stride + tile_ns_offset), N * (int) sizeof(short),
                                Ns * (int) sizeof(short), Ktile / 8, rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_scales, Ns * (int) sizeof(short),
                                (const void *) (devB_scales + i * scales_stride + tile_ns_offset),
                                N * (int) sizeof(short), Ns * (int) sizeof(short), Ktile / 64, rtMemcpyDeviceToSram,
                                ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_super_scale, Ns * (int) sizeof(short),
                                (const void *) (devB_super_scale + i * super_scale_stride + tile_ns_offset),
                                N * (int) sizeof(short), Ns * (int) sizeof(short), Ktile / weights_group,
                                rtMemcpyDeviceToSram, ctx.kernelStream);
            }

            // merge super_scale with qscale
            params.clear();
            q2xs_super_scale_blocks(Ktile, weights_group, group, Ns, threadsPerBlock, blocksPerGrid);
            q2xs_super_scale_params(sramB_scales, sramB_super_scale, sramB_qscale_lut, sramB_scale, Ktile, Ns,
                                    weights_group, q_group, blocksPerGrid, threadsPerBlock, params);
            launchWrapperAysnc("matrix_mul_q2xs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
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
            //---------------------------------------------------------------------------------------
            // InputAcc [K/256]      | [8]
            //         [grid.z]      | [loopidx]
            // accIdx = blockIdx.z * 8 + loopidx
            //----------------------------------------------------------------------------------------
            uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * sizeof(short);
            task.taskName           = "matrix_mul_vxM_f16_q2xs_f16_asym_opt";
            task.params.kernelList.clear();
            // Use actual matmul launch shape for blockXSize-sensitive params.
            dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
            dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
            matmul_weights_q2xs_kernel_params(matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile,
                                              sramB_qs, sramB_scale, sramC + j * Ns * sizeof(short), sramB_codebook_lut,
                                              sramB_sign_lut, input_acc_addr, input_acc_addr + hilo_offset, Ns,
                                              N * sizeof(short), weights_group, combine, task.params.kernelList);
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    // -------------------------
    // Launch with NsTail
    // -------------------------
    for (int i = 0; i < nr_of_tiles; i++) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_qs, (const void *) devB_qs, sizeB_qs, rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_scales, (const void *) devB_scales, sizeB_scales, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale,
                          rtMemcpyDeviceToSram, ctx.kernelStream);

            devB_qs += sizeB_qs;
            devB_scales += sizeB_scales;
            devB_super_scale += sizeB_super_scale;
        } else {
            int       qs_stride          = (Ktile / 8 * N * (int) sizeof(short));
            int       scales_stride      = (Ktile / 64 * N * (int) sizeof(short));
            int       super_scale_stride = Ktile / weights_group * N * (int) sizeof(short);
            const int tail_ns_offset     = (nr_of_ns - 1) * Ns * (int) sizeof(short);

            rtMemcpy2DAsync((void *) sramB_qs, NsTail * (int) sizeof(short),
                            (const void *) (devB_qs + i * qs_stride + tail_ns_offset), N * (int) sizeof(short),
                            NsTail * (int) sizeof(short), Ktile / 8, rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_scales, NsTail * (int) sizeof(short),
                            (const void *) (devB_scales + i * scales_stride + tail_ns_offset), N * (int) sizeof(short),
                            NsTail * (int) sizeof(short), Ktile / 64, rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_super_scale, NsTail * (int) sizeof(short),
                            (const void *) (devB_super_scale + i * super_scale_stride + tail_ns_offset),
                            N * (int) sizeof(short), NsTail * (int) sizeof(short), Ktile / weights_group,
                            rtMemcpyDeviceToSram, ctx.kernelStream);
        }

        // merge super_scale with qscale
        params.clear();
        q2xs_super_scale_blocks(Ktile, weights_group, group, NsTail, threadsPerBlock, blocksPerGrid);
        q2xs_super_scale_params(sramB_scales, sramB_super_scale, sramB_qscale_lut, sramB_scale, Ktile, NsTail,
                                weights_group, q_group, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q2xs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
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
        //---------------------------------------------------------------------------------------
        // InputAcc [K/256]      | [8]
        //         [grid.z]     | [loopidx]
        // accIdx = blockIdx.z * 8 + loopidx
        //----------------------------------------------------------------------------------------
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * sizeof(short);
        task.taskName           = "matrix_mul_vxM_f16_q2xs_f16_asym_opt";
        task.params.kernelList.clear();
        // Use actual matmul launch shape for blockXSize-sensitive params.
        dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q2xs_kernel_params(
            matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile, sramB_qs, sramB_scale,
            sramC + (nr_of_ns - 1) * Ns * sizeof(short), sramB_codebook_lut, sramB_sign_lut, input_acc_addr,
            input_acc_addr + hilo_offset, NsTail, N * sizeof(short), weights_group, combine, task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (out_bytes_per_element == sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        // CDMA S2D
        rtMemcpyAsync((void *) devC, (const void *) sramC1, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        // CDMA S2D
        rtMemcpyAsync((void *) devC, (const void *) sramC, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q2xs_vxm_build(rpp_kernel_context & ctx,
                                      int                  M,
                                      int                  K,
                                      int                  N,
                                      int                  weights_group,
                                      int                  in_bytes_per_element,
                                      int                  out_bytes_per_element,
                                      int                  use_pipeline  = 0,
                                      int                  is_instantial = 1) {
    rpp_matmul_q2xs_vxm(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, is_instantial);
}

}  // namespace kernel_q2_xs_vxm
