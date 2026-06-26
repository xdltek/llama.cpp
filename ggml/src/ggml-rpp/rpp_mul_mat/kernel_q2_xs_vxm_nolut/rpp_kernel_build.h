// rpp_matmul_q2xs_vxm_nolut.cpp

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q2_xs_vxm_nolut/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q2_xs_vxm_nolut/rpp_kernel_param.h"

#include <assert.h>

#include <array>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q2_xs_vxm_nolut {

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
        nr_of_tiles     = K / weights_group;
        groups_per_tile = 1;
    }

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

    if (N == 151936) {
        Ns       = 600 * 32;
        nr_of_ns = 8;
        NsTail   = 548 * 32;
    } else if (N == 200064) {
        Ns       = 640 * 32;
        nr_of_ns = 10;
        NsTail   = 492 * 32;
    }

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

static void matmul_weights_q2xs_nolut_batch_params_local(dim3 &                  blocksPerGrid,
                                                         dim3 &                  threadsPerBlock,
                                                         uint32_t                in_act,
                                                         uint32_t                in_wq,
                                                         uint32_t                in_scale,
                                                         uint32_t                in_sign,
                                                         uint32_t                out_addr,
                                                         uint32_t                lut_addr,
                                                         uint32_t                input_acc_addr,
                                                         uint32_t                input_acc_addr_hi,
                                                         uint32_t                N,
                                                         uint32_t                hilo_stride,
                                                         uint32_t                weights_group,
                                                         uint32_t                in_wq_expert_stride_bytes,
                                                         uint32_t                in_sign_expert_stride_bytes,
                                                         uint32_t                in_scale_expert_stride_bytes,
                                                         uint32_t                in_act_expert_stride_bytes,
                                                         uint32_t                combine,
                                                         std::vector<uint32_t> & params) {
    (void) blocksPerGrid;
    uint32_t inUnrollStride  = N * sizeof(short);
    uint32_t blockXSize      = threadsPerBlock.x * sizeof(short);
    uint32_t scaleLoopStride = 4 * N * sizeof(short);

    uint32_t inLoopStride0        = 4 * 2 * N * sizeof(short);
    uint32_t inLoopStride1        = 4 * N * sizeof(short);
    uint32_t in_wq_blockz_size    = 4 * inLoopStride0;
    uint32_t in_sign_blockz_size  = 4 * inLoopStride1;
    uint32_t in_scale_blockz_size = 4 * inLoopStride1;
    uint32_t in_a_blockz_size     = weights_group * sizeof(short);
    uint32_t in_wq_stridez        = in_wq_expert_stride_bytes / 2;
    uint32_t in_sign_stridez      = in_sign_expert_stride_bytes / 2;
    uint32_t in_scale_stridez     = in_scale_expert_stride_bytes / 2;
    uint32_t loop                 = 4;

    params.emplace_back(in_act);
    params.emplace_back(in_wq);
    params.emplace_back(in_sign);
    params.emplace_back(in_scale);
    params.emplace_back(out_addr);
    params.emplace_back(lut_addr);
    params.emplace_back(inUnrollStride);
    params.emplace_back(inLoopStride0);
    params.emplace_back(inLoopStride1);
    params.emplace_back(scaleLoopStride);
    params.emplace_back(blockXSize);
    params.emplace_back(in_wq_blockz_size);
    params.emplace_back(in_sign_blockz_size);
    params.emplace_back(in_scale_blockz_size);
    params.emplace_back(in_a_blockz_size);
    params.emplace_back(input_acc_addr);
    params.emplace_back(input_acc_addr_hi);
    params.emplace_back(hilo_stride);
    params.emplace_back(in_wq_stridez);
    params.emplace_back(in_sign_stridez);
    params.emplace_back(in_scale_stridez);
    params.emplace_back(in_act_expert_stride_bytes);
    params.emplace_back(loop);
    params.emplace_back(combine);
}

static void q2xs_vxm_nolut_prepare_lookup_tables(rpp_kernel_context & ctx,
                                                 int                  qscale_lut_bytes,
                                                 int                  mag_lut_bytes,
                                                 int                  mat_lut_bytes,
                                                 RPPdeviceptr &       devB_qscale_lut,
                                                 RPPdeviceptr &       devB_mag_lut,
                                                 RPPdeviceptr &       devB_mat_lut) {
    auto align_up = [](size_t x, size_t a) -> size_t {
        return ((x + a - 1) / a) * a;
    };

    const size_t off_qscale_lut  = 0;
    const size_t off_mag_lut     = align_up(off_qscale_lut + (size_t) qscale_lut_bytes, 64);
    const size_t off_mat_lut     = align_up(off_mag_lut + (size_t) mag_lut_bytes, 64);
    const size_t workspace_bytes = align_up(off_mat_lut + (size_t) mat_lut_bytes, 64);

    if (ctx.dev_workspace != 0) {
        rtFree((void *) ctx.dev_workspace);
        ctx.dev_workspace = 0;
    }

    if (rtMalloc((void **) &ctx.dev_workspace, workspace_bytes) != rtSuccess) {
        throw std::runtime_error("Q2XS NoLUT workspace allocation failed");
    }

    RPPdeviceptr base = ctx.dev_workspace;
    devB_qscale_lut   = base + (RPPdeviceptr) off_qscale_lut;
    devB_mag_lut      = base + (RPPdeviceptr) off_mag_lut;
    devB_mat_lut      = base + (RPPdeviceptr) off_mat_lut;

    std::array<uint16_t, 16> qscale_lut = {};
    for (int i = 0; i < (int) qscale_lut.size(); ++i) {
        const float lut_val = (0.5f + (float) i) * 0.25f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }

    constexpr std::array<float, 4> mag_lut_values = { 8.0f, 25.0f, 43.0f, 0.0f };
    std::array<uint16_t, 4>        mag_lut        = {};
    for (int i = 0; i < (int) mag_lut.size(); ++i) {
        mag_lut[i] = float_to_bf16_rne(mag_lut_values[i]);
    }

    // mat_lut: decode one packed 16-bit no-LUT word (8 x 2-bit codes)
    // into 8 bf16 magnitudes.
    constexpr int                  mat_lut_rows      = 1 << 16;
    constexpr int                  mat_lut_cols      = 8;
    constexpr std::array<float, 4> nolut_code_to_mag = { 8.0f, 25.0f, 43.0f, 0.0f };
    std::vector<uint16_t>          mat_lut((size_t) mat_lut_rows * (size_t) mat_lut_cols, 0);
    for (int packed = 0; packed < mat_lut_rows; ++packed) {
        for (int u = 0; u < mat_lut_cols; ++u) {
            const uint8_t code                                            = (uint8_t) ((packed >> (2 * u)) & 0x3u);
            mat_lut[(size_t) packed * (size_t) mat_lut_cols + (size_t) u] = float_to_bf16_rne(nolut_code_to_mag[code]);
        }
    }

    rtMemcpy((void *) devB_qscale_lut, qscale_lut.data(), qscale_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_mag_lut, mag_lut.data(), mag_lut_bytes, rtMemcpyHostToDevice);
    rtMemcpy((void *) devB_mat_lut, mat_lut.data(), mat_lut_bytes, rtMemcpyHostToDevice);
}

static void rpp_matmul_q2xs_vxm_nolut(rpp_kernel_context & ctx,
                                      int                  M,
                                      int                  K,
                                      int                  N,
                                      int                  weights_group,
                                      int                  in_bytes_per_element,
                                      int                  out_bytes_per_element,
                                      int                  is_instantial = 1,
                                      int                  is_capture    = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q2XS NoLUT Configure Parameter Invalid");
    }
    if (ctx.dev_in.size() < 5 || ctx.dev_out.empty()) {
        throw std::runtime_error(
            "Matmul Q2XS NoLUT requires 5 inputs (A, codebook_nolut, scales, sign, super_scale) and 1 output");
    }
    if (weights_group != 256 || (K % 256) != 0) {
        throw std::runtime_error("Matmul Q2XS NoLUT requires weights_group == 256 and K % 256 == 0");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    const int qscale_lut_elems = 16;
    const int mag_lut_elems    = 4;
    const int mat_lut_elems    = (1 << 16) * 8;
    const int qscale_lut_bytes = qscale_lut_elems * (int) sizeof(uint16_t);
    const int mag_lut_bytes    = mag_lut_elems * (int) sizeof(uint16_t);
    const int mat_lut_bytes    = mat_lut_elems * (int) sizeof(uint16_t);

    RPPdeviceptr devA                = ctx.dev_in[0];
    RPPdeviceptr devB_codebook_nolut = ctx.dev_in[1];
    RPPdeviceptr devB_scales         = ctx.dev_in[2];
    RPPdeviceptr devB_sign           = ctx.dev_in[3];
    RPPdeviceptr devB_super_scale    = ctx.dev_in[4];
    RPPdeviceptr devC                = ctx.dev_out[0];

    RPPdeviceptr devB_qscale_lut = 0;
    RPPdeviceptr devB_mag_lut    = 0;
    RPPdeviceptr devB_mat_lut    = 0;
    q2xs_vxm_nolut_prepare_lookup_tables(ctx, qscale_lut_bytes, mag_lut_bytes, mat_lut_bytes, devB_qscale_lut,
                                         devB_mag_lut, devB_mat_lut);

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q2xs_vxm_nolut.o");

    int nr_of_tiles, groups_per_tile;
    int nr_of_ns, Ns, NsTail;
    get_tiles_info(N, K, weights_group, nr_of_tiles, groups_per_tile, nr_of_ns, Ns, NsTail);

    const int sizeA_in = K * in_bytes_per_element;
    const int sizeA    = K * (int) sizeof(short);

    const int q_group     = 32;
    const int super_group = 256;
    const int group       = super_group / q_group;
    assert(super_group == weights_group);

    const int Ktile                = groups_per_tile * weights_group;
    const int sizeB_codebook_nolut = (Ktile * Ns / 4);
    const int sizeB_scales         = (Ktile * Ns / 32);
    const int sizeB_sign           = (Ktile * Ns / 8);
    const int sizeB_super_scale    = (int) ((groups_per_tile * weights_group) * Ns / weights_group * sizeof(short));
    const int sizeB_qscale_lut     = qscale_lut_bytes;
    const int sizeB_mag_lut        = mag_lut_bytes;
    const int sizeB_mat_lut        = mat_lut_bytes;
    const int sizeB_scale          = (groups_per_tile * weights_group * Ns / 16) * (int) sizeof(short);

    const int sizeC32 = N * (int) sizeof(float);
    const int sizeC   = N * out_bytes_per_element;

    RPPdeviceptr sram_base            = ctx.virtual_sram_base;
    RPPdeviceptr sramA_in             = sram_base;
    RPPdeviceptr sramA                = sramA_in + round_up(sizeA_in);
    RPPdeviceptr sramC                = sramA + round_up(sizeA);
    RPPdeviceptr sramC1               = sramC + round_up(sizeC32);
    RPPdeviceptr sramB_codebook_nolut = sramC1 + round_up(sizeC);
    RPPdeviceptr sramB_scales         = sramB_codebook_nolut + round_up(sizeB_codebook_nolut);
    RPPdeviceptr sramB_sign           = sramB_scales + round_up(sizeB_scales);
    RPPdeviceptr sramB_super_scale    = sramB_sign + round_up(sizeB_sign);
    RPPdeviceptr sramB_qscale_lut     = sramB_super_scale + round_up(sizeB_super_scale);
    RPPdeviceptr sramB_mag_lut        = sramB_qscale_lut + round_up(sizeB_qscale_lut);
    RPPdeviceptr sramB_mat_lut        = sramB_mag_lut + round_up(sizeB_mag_lut);
    RPPdeviceptr sramB_scale          = sramB_mat_lut + round_up(sizeB_mat_lut);
    RPPdeviceptr sramA_acc            = sramB_scale + round_up(sizeB_scale);

    const int total_sram_bytes = (int) (sramA_acc + round_up(weights_group * 4) - sram_base);
    const int SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    rtMemcpyAsync((void *) (in_bytes_per_element == (int) sizeof(float) ? sramA_in : sramA), (const void *) devA,
                  sizeA_in, rtMemcpyDeviceToSram, ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_qscale_lut, (const void *) devB_qscale_lut, sizeB_qscale_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_mag_lut, (const void *) devB_mag_lut, sizeB_mag_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);
    rtMemcpyAsync((void *) sramB_mat_lut, (const void *) devB_mat_lut, sizeB_mat_lut, rtMemcpyDeviceToSram,
                  ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M, K, threadsPerBlock, blocksPerGrid);
        // HW workaround: keep CVT_32_16 out-of-place.
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
    int combine;

    int Nx = K / q_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    int hilo_offset = (Nx * Nz + 31) / 32 * 32 * (int) sizeof(short);

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

    for (int i = 0; i < nr_of_tiles; i++) {
        for (int j = 0; j < nr_of_ns - 1; j++) {
            if (nr_of_ns == 1) {
                rtMemcpyAsync((void *) sramB_codebook_nolut, (const void *) devB_codebook_nolut, sizeB_codebook_nolut,
                              rtMemcpyDeviceToSram, ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_scales, (const void *) devB_scales, sizeB_scales, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_sign, (const void *) devB_sign, sizeB_sign, rtMemcpyDeviceToSram,
                              ctx.kernelStream);
                rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale,
                              rtMemcpyDeviceToSram, ctx.kernelStream);

                devB_codebook_nolut += sizeB_codebook_nolut;
                devB_scales += sizeB_scales;
                devB_sign += sizeB_sign;
                devB_super_scale += sizeB_super_scale;
            } else {
                const int codebook_nolut_stride = (Ktile / 8 * N * (int) sizeof(short));
                const int scales_stride         = (Ktile / 64 * N * (int) sizeof(short));
                const int sign_stride           = (Ktile / 16 * N * (int) sizeof(short));
                const int super_scale_stride    = Ktile / weights_group * N * (int) sizeof(short);
                const int tile_ns_offset        = j * Ns * (int) sizeof(short);

                rtMemcpy2DAsync((void *) sramB_codebook_nolut, Ns * (int) sizeof(short),
                                (const void *) (devB_codebook_nolut + i * codebook_nolut_stride + tile_ns_offset),
                                N * (int) sizeof(short), Ns * (int) sizeof(short), Ktile / 8, rtMemcpyDeviceToSram,
                                ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_scales, Ns * (int) sizeof(short),
                                (const void *) (devB_scales + i * scales_stride + tile_ns_offset),
                                N * (int) sizeof(short), Ns * (int) sizeof(short), Ktile / 64, rtMemcpyDeviceToSram,
                                ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_sign, Ns * (int) sizeof(short),
                                (const void *) (devB_sign + i * sign_stride + tile_ns_offset), N * (int) sizeof(short),
                                Ns * (int) sizeof(short), Ktile / 16, rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_super_scale, Ns * (int) sizeof(short),
                                (const void *) (devB_super_scale + i * super_scale_stride + tile_ns_offset),
                                N * (int) sizeof(short), Ns * (int) sizeof(short), Ktile / weights_group,
                                rtMemcpyDeviceToSram, ctx.kernelStream);
            }

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
            uint32_t stride_ina     = weights_group * (uint32_t) sizeof(short);
            uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * (uint32_t) sizeof(short);
            task.taskName           = "matrix_mul_vxM_q2xs_nolut_asym_opt";
            task.params.kernelList.clear();
            dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
            dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
            matmul_weights_q2xs_kernel_params(matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile,
                                              sramB_codebook_nolut, sramB_scale, sramB_sign,
                                              sramC + j * Ns * (uint32_t) sizeof(short), sramB_mag_lut, input_acc_addr,
                                              input_acc_addr + hilo_offset, Ns, N * (uint32_t) sizeof(short),
                                              weights_group, combine, task.params.kernelList);
            launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    for (int i = 0; i < nr_of_tiles; i++) {
        if (nr_of_ns == 1) {
            rtMemcpyAsync((void *) sramB_codebook_nolut, (const void *) devB_codebook_nolut, sizeB_codebook_nolut,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_scales, (const void *) devB_scales, sizeB_scales, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_sign, (const void *) devB_sign, sizeB_sign, rtMemcpyDeviceToSram,
                          ctx.kernelStream);
            rtMemcpyAsync((void *) sramB_super_scale, (const void *) devB_super_scale, sizeB_super_scale,
                          rtMemcpyDeviceToSram, ctx.kernelStream);

            devB_codebook_nolut += sizeB_codebook_nolut;
            devB_scales += sizeB_scales;
            devB_sign += sizeB_sign;
            devB_super_scale += sizeB_super_scale;
        } else {
            const int codebook_nolut_stride = (Ktile / 8 * N * (int) sizeof(short));
            const int scales_stride         = (Ktile / 64 * N * (int) sizeof(short));
            const int sign_stride           = (Ktile / 16 * N * (int) sizeof(short));
            const int super_scale_stride    = Ktile / weights_group * N * (int) sizeof(short);
            const int tail_ns_offset        = (nr_of_ns - 1) * Ns * (int) sizeof(short);

            rtMemcpy2DAsync((void *) sramB_codebook_nolut, NsTail * (int) sizeof(short),
                            (const void *) (devB_codebook_nolut + i * codebook_nolut_stride + tail_ns_offset),
                            N * (int) sizeof(short), NsTail * (int) sizeof(short), Ktile / 8, rtMemcpyDeviceToSram,
                            ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_scales, NsTail * (int) sizeof(short),
                            (const void *) (devB_scales + i * scales_stride + tail_ns_offset), N * (int) sizeof(short),
                            NsTail * (int) sizeof(short), Ktile / 64, rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_sign, NsTail * (int) sizeof(short),
                            (const void *) (devB_sign + i * sign_stride + tail_ns_offset), N * (int) sizeof(short),
                            NsTail * (int) sizeof(short), Ktile / 16, rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_super_scale, NsTail * (int) sizeof(short),
                            (const void *) (devB_super_scale + i * super_scale_stride + tail_ns_offset),
                            N * (int) sizeof(short), NsTail * (int) sizeof(short), Ktile / weights_group,
                            rtMemcpyDeviceToSram, ctx.kernelStream);
        }

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
        uint32_t stride_ina     = weights_group * (uint32_t) sizeof(short);
        uint32_t input_acc_addr = sramA_acc + i * groups_per_tile * group * (uint32_t) sizeof(short);
        task.taskName           = "matrix_mul_vxM_q2xs_nolut_asym_opt";
        task.params.kernelList.clear();
        dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q2xs_kernel_params(matmulBlocks, matmulThreads, sramA + i * stride_ina * groups_per_tile,
                                          sramB_codebook_nolut, sramB_scale, sramB_sign,
                                          sramC + (nr_of_ns - 1) * Ns * (uint32_t) sizeof(short), sramB_mag_lut,
                                          input_acc_addr, input_acc_addr + hilo_offset, NsTail,
                                          N * (uint32_t) sizeof(short), weights_group, combine, task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramC1, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        rtMemcpyAsync((void *) devC, (const void *) sramC1, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    } else {
        rtMemcpyAsync((void *) devC, (const void *) sramC, sizeC, rtMemcpySramToDevice, ctx.kernelStream);
    }

    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q2xs_vxm_nolut_sram(rpp_kernel_context & ctx,
                                           int                  M,
                                           int                  K,
                                           int                  N,
                                           int                  weights_group,
                                           int                  in_bytes_per_element,
                                           int                  out_bytes_per_element,
                                           int                  experts,
                                           int                  is_instantial = 1,
                                           int                  is_capture    = 1) {
    if ((M != 1) || (K / weights_group == 0) || (N % 32 != 0)) {
        throw std::runtime_error("Matmul Q2XS NoLUT SRAM Configure Parameter Invalid");
    }
    if (experts <= 0) {
        throw std::runtime_error("Matmul Q2XS NoLUT SRAM expects experts > 0");
    }
    if (weights_group != 256 || (K % 256) != 0) {
        throw std::runtime_error("Matmul Q2XS NoLUT SRAM expects weights_group == 256 and K % 256 == 0");
    }
    if (ctx.dev_in.size() < 9 || ctx.dev_out.empty()) {
        throw std::runtime_error("Matmul Q2XS NoLUT SRAM requires ctx.dev_in[0..8] and ctx.dev_out SRAM addresses");
    }
    if (out_bytes_per_element == (int) sizeof(float) && ctx.dev_out.size() < 2) {
        throw std::runtime_error(
            "Matmul Q2XS NoLUT SRAM fp32 output requires ctx.dev_out[0]=fp32_out and ctx.dev_out[1]=bf16_tmp");
    }
    // const RPPdeviceptr lut_ws = q2xs_vxm_nolut_prepare_lut_workspace(ctx);

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    const RPPdeviceptr sramA                = ctx.dev_in[0];
    const RPPdeviceptr sramB_codebook_nolut = ctx.dev_in[1];
    const RPPdeviceptr sramB_scales         = ctx.dev_in[2];
    const RPPdeviceptr sramB_sign           = ctx.dev_in[3];
    const RPPdeviceptr sramB_super_scale    = ctx.dev_in[4];
    const RPPdeviceptr sramB_qscale_lut     = ctx.dev_in[5];
    const RPPdeviceptr sramB_mag_lut        = ctx.dev_in[6];
    const RPPdeviceptr sramB_scale          = ctx.dev_in[7];
    const RPPdeviceptr sramA_acc            = ctx.dev_in[8];

    RPPdeviceptr sramC   = ctx.dev_out[0];
    RPPdeviceptr sramOut = ctx.dev_out[0];
    if (out_bytes_per_element == (int) sizeof(float)) {
        sramC = ctx.dev_out[1];
    }

    if (is_capture) {
        rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    }
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q2xs_vxm_nolut.o");

    if (K % weights_group != 0) {
        throw std::runtime_error("Matmul Q2XS NoLUT SRAM expects full-K one tile (K % weights_group == 0)");
    }
    const int groups_per_tile = K / weights_group;
    const int Ktile           = groups_per_tile * weights_group;
    assert(K == Ktile);

    const int q_group     = 32;
    const int super_group = 256;
    const int group       = super_group / q_group;
    assert(super_group == weights_group);

    const int      sizeB_codebook_nolut  = (Ktile * N / 4);
    const int      sizeB_scales          = (Ktile * N / 32);
    const int      sizeB_sign            = (Ktile * N / 8);
    const int      sizeB_super_scale     = (int) ((Ktile / weights_group) * N * (int) sizeof(short));
    const int      sizeB_scale           = (int) ((Ktile * N / 16) * (int) sizeof(short));
    const uint32_t expert_weights_stride = (uint32_t) sizeB_codebook_nolut + (uint32_t) sizeB_scales +
                                           (uint32_t) sizeB_sign + (uint32_t) sizeB_super_scale;

    const RPPdeviceptr expected_sramB_scales      = sramB_codebook_nolut + (RPPdeviceptr) sizeB_codebook_nolut;
    const RPPdeviceptr expected_sramB_sign        = expected_sramB_scales + (RPPdeviceptr) sizeB_scales;
    const RPPdeviceptr expected_sramB_super_scale = expected_sramB_sign + (RPPdeviceptr) sizeB_sign;
    if (sramB_scales != expected_sramB_scales || sramB_sign != expected_sramB_sign ||
        sramB_super_scale != expected_sramB_super_scale) {
        throw std::runtime_error(
            "Matmul Q2XS NoLUT SRAM expects per-expert packed weight chunk layout [codebook|scales|sign|super]");
    }
    // q2xs_vxm_nolut_copy_lut_workspace_to_sram(sramB_qscale_lut, sramB_mag_lut, lut_ws, ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(M * experts, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA, sramA, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    int Nx = K / q_group;
    int Nz = 1;
    while (Nx * Nz <= 32) {
        Nz++;
    }
    const int      hilo_offset          = (Nx * Nz + 31) / 32 * 32 * (int) sizeof(short);
    const uint32_t input_acc_addr       = (uint32_t) sramA_acc;
    const uint32_t in_act_expert_stride = (uint32_t) K * (uint32_t) sizeof(short);

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

    for (int e = 0; e < experts; ++e) {
        const RPPdeviceptr sramB_scales_cur = sramB_codebook_nolut +
                                              (RPPdeviceptr) e * (RPPdeviceptr) expert_weights_stride +
                                              (RPPdeviceptr) sizeB_codebook_nolut;
        const RPPdeviceptr sramB_super_scale_cur = sramB_codebook_nolut +
                                                   (RPPdeviceptr) e * (RPPdeviceptr) expert_weights_stride +
                                                   (RPPdeviceptr) (sizeB_codebook_nolut + sizeB_scales + sizeB_sign);
        const RPPdeviceptr sramB_scale_cur = sramB_scale + (RPPdeviceptr) e * (RPPdeviceptr) sizeB_scale;

        params.clear();
        q2xs_super_scale_blocks(Ktile, weights_group, group, N, threadsPerBlock, blocksPerGrid);
        q2xs_super_scale_params((uint32_t) sramB_scales_cur, (uint32_t) sramB_super_scale_cur,
                                (uint32_t) sramB_qscale_lut, (uint32_t) sramB_scale_cur, Ktile, N, weights_group,
                                q_group, blocksPerGrid, threadsPerBlock, params);
        launchWrapperAysnc("matrix_mul_q2xs_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    RppTaskElement task;
    uint32_t       block_x, block_y, block_z, grid_x, grid_y, grid_z;
    kerneldim_calc_matmul_linear(1, M, N, block_x, block_y, block_z, grid_x, grid_y, grid_z);
    (void) grid_z;

    (void) block_z;

    uint32_t tx                 = 1;
    uint32_t experts_per_launch = (uint32_t) experts;
    while (experts_per_launch > 1 && experts_per_launch * block_x >= 8192) {
        experts_per_launch >>= 1;
        tx <<= 1;
    }
    if ((uint32_t) experts != tx * experts_per_launch) {
        throw std::runtime_error("Matmul Q2XS NoLUT SRAM expects experts to be power-of-2 for Tx tiling");
    }

    task.blockDim.x = block_x;
    task.blockDim.y = block_y;
    task.blockDim.z = experts_per_launch;
    task.gridDim.x  = grid_x;
    task.gridDim.y  = grid_y;
    task.gridDim.z  = groups_per_tile;
    task.taskName   = "matrix_mul_vxM_q2xs_nolut_batch";
    task.params.kernelList.clear();

    const uint32_t out_expert_stride = (uint32_t) N * (uint32_t) sizeof(short);
    for (uint32_t t = 0; t < tx; ++t) {
        const uint32_t expert_begin = t * experts_per_launch;
        const uint32_t in_act_base =
            (uint32_t) (sramA + (RPPdeviceptr) expert_begin * (RPPdeviceptr) in_act_expert_stride);
        const uint32_t in_wq_base =
            (uint32_t) (sramB_codebook_nolut + (RPPdeviceptr) expert_begin * (RPPdeviceptr) expert_weights_stride);
        const uint32_t in_sign_base = in_wq_base + (uint32_t) sizeB_codebook_nolut + (uint32_t) sizeB_scales;
        const uint32_t in_scale_base =
            (uint32_t) (sramB_scale + (RPPdeviceptr) expert_begin * (RPPdeviceptr) sizeB_scale);
        const uint32_t out_base = (uint32_t) (sramC + (RPPdeviceptr) expert_begin * (RPPdeviceptr) out_expert_stride);

        task.params.kernelList.clear();
        dim3 matmulThreads(task.blockDim.x, task.blockDim.y, task.blockDim.z);
        dim3 matmulBlocks(task.gridDim.x, task.gridDim.y, task.gridDim.z);
        matmul_weights_q2xs_nolut_batch_params_local(
            matmulBlocks, matmulThreads, in_act_base, in_wq_base, in_scale_base, in_sign_base, out_base,
            (uint32_t) sramB_mag_lut, input_acc_addr, input_acc_addr + hilo_offset, (uint32_t) N,
            (uint32_t) experts * (uint32_t) N * (uint32_t) sizeof(short), (uint32_t) weights_group,
            expert_weights_stride, expert_weights_stride, (uint32_t) sizeB_scale, in_act_expert_stride, 0,
            task.params.kernelList);
        launchWrapperAysnc(task.taskName, task.gridDim, task.blockDim, task.params.kernelList, ctx.rppBinMod,
                           ctx.kernelStream);
    }

    if (out_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(1, N * experts * 2, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init_opt(threadsPerBlock, sramC, sramOut, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    } else if (sramOut != sramC) {
        throw std::runtime_error(
            "Matmul Q2XS NoLUT SRAM bf16 output requires ctx.dev_out[0] to point to matmul bf16 output SRAM");
    }
    if (is_capture) {
        rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    }
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q2xs_vxm_nolut_build(rpp_kernel_context & ctx,
                                            int                  M,
                                            int                  K,
                                            int                  N,
                                            int                  weights_group,
                                            int                  in_bytes_per_element,
                                            int                  out_bytes_per_element,
                                            int                  use_pipeline    = 0,
                                            int                  use_sram_direct = 0,
                                            int                  experts         = 8,
                                            int                  is_instantial   = 1,
                                            int                  is_capture      = 1) {
    (void) use_pipeline;
    if (!use_sram_direct) {
        rpp_matmul_q2xs_vxm_nolut(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                  is_instantial, is_capture);
    } else {
        rpp_matmul_q2xs_vxm_nolut_sram(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                       experts, is_instantial, is_capture);
    }
}
}  // namespace kernel_q2_xs_vxm_nolut
