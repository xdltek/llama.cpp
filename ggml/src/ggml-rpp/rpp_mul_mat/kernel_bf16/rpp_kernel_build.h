// rpp_matmul_bf16.cpp
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_bf16/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_bf16/rpp_kernel_param.h"

#include <assert.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernel_bf16 {

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

inline int get_tiles(int N) {
    return (N + MATMUL_NS - 1) / MATMUL_NS;
}

inline int get_ns(int N, bool is_tail) {
    if (is_tail) {
        return N - (get_tiles(N) - 1) * MATMUL_NS;
    } else {
        return MATMUL_NS;
    }
}

inline std::string get_matmul_kernel(int N) {
    std::string kernel_name;
    if ((N % 128) == 0) {
        kernel_name = "matmul_tn4_f16_f32_f16";
    } else if ((N % 96) == 0) {
        kernel_name = "matmul_tn3_f16_f32_f16";
    } else if ((N % 64) == 0) {
        kernel_name = "matmul_tn2_f16_f32_f16";
    } else {
        kernel_name = "matmul_tn1_f16_f32_f16";
    }
    return kernel_name;
}

static inline void launch_fill_16bits_zero(rpp_kernel_context &    ctx,
                                           RPPdeviceptr            addr,
                                           int                     bytes,
                                           dim3 &                  threadsPerBlock,
                                           dim3 &                  blocksPerGrid,
                                           std::vector<uint32_t> & params) {
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
    launchWrapperAysnc("fill_16bits_align", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod, ctx.kernelStream);
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_bf16_build(rpp_kernel_context & ctx,
                                  int                  M,
                                  int                  K,
                                  int                  N,
                                  int                  weights_group,
                                  int                  in_bytes_per_element,
                                  int                  out_bytes_per_element,
                                  int                  is_instantial = 1) {
    if (ctx.dev_in.size() < 2 || ctx.dev_out.empty()) {
        throw std::runtime_error("Matmul BF16 requires 2 inputs and 1 output");
    }
    if (weights_group != 32) {
        throw std::runtime_error("Matmul BF16 requires weights_group == 32");
    }
    const int Kr = (int) round_up_32((uint32_t) K);
    const int Nr = (int) round_up_32((uint32_t) N);

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA      = ctx.dev_in[0];
    RPPdeviceptr devB_wf16 = ctx.dev_in[1];
    RPPdeviceptr devC      = ctx.dev_out[0];

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_bf16.o");

    const int SRAM_LIMIT            = 22 * 1024 * 1024;
    const int sizeB_tile            = Kr * MATMUL_NS * (int) sizeof(uint16_t);
    auto      calc_total_sram_bytes = [&](int m_rows) -> int64_t {
        const int sizeA0_cur  = m_rows * Kr * in_bytes_per_element;
        const int sizeA1_cur  = m_rows * Kr * (int) sizeof(uint16_t);
        const int sizeCr_cur  = m_rows * MATMUL_NS * (int) sizeof(uint16_t);
        const int sizeC32_cur = m_rows * MATMUL_NS * (int) sizeof(float);

        int64_t total = 0;
        total += round_up(sizeA0_cur);
        total += round_up(sizeA1_cur);
        total += round_up(sizeB_tile);
        total += round_up(sizeCr_cur);
        total += round_up(sizeCr_cur);
        total += round_up(sizeC32_cur);
        return total;
    };

    int Mtile = M;
    if (calc_total_sram_bytes(Mtile) > SRAM_LIMIT) {
        int lo = 1;
        int hi = M;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (calc_total_sram_bytes(mid) <= SRAM_LIMIT) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Mtile = lo;
        if (Mtile > 32) {
            Mtile = (Mtile / 32) * 32;
        }
        if (Mtile <= 0) {
            Mtile = lo;
        }
    }

    const int sizeA0  = Mtile * Kr * in_bytes_per_element;
    const int sizeA1  = Mtile * Kr * (int) sizeof(uint16_t);
    const int sizeCr  = Mtile * MATMUL_NS * (int) sizeof(uint16_t);
    const int sizeC32 = Mtile * MATMUL_NS * (int) sizeof(float);

    RPPdeviceptr sram_base      = ctx.virtual_sram_base;
    RPPdeviceptr sramA_in       = sram_base;
    RPPdeviceptr sramA_out      = sramA_in + round_up(sizeA0);
    RPPdeviceptr sramB_out      = sramA_out + round_up(sizeA1);
    RPPdeviceptr sramC_hw32     = sramB_out + round_up(sizeB_tile);
    RPPdeviceptr sramC_chw      = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32 = sramC_chw + round_up(sizeCr);

    const int total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    const RPPdeviceptr devB_base = devB_wf16;
    const RPPdeviceptr devC_base = devC;
    const int          m_tiles   = (M + Mtile - 1) / Mtile;

    for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
        const int          m_start                        = m_tile * Mtile;
        const int          m_rows                         = std::min(Mtile, M - m_start);
        const bool         layout_identity_for_single_row = (m_rows == 1);
        const RPPdeviceptr matmul_A_cur                   = layout_identity_for_single_row ? sramA_in : sramA_out;

        if (Kr != K) {
            const int sizeA0_cur = m_rows * Kr * in_bytes_per_element;
            launch_fill_16bits_zero(ctx, sramA_in, sizeA0_cur, threadsPerBlock, blocksPerGrid, params);
        }

        rtMemcpy2DAsync((void *) sramA_in, Kr * in_bytes_per_element,
                        (const void *) (devA + (RPPdeviceptr) m_start * K * in_bytes_per_element),
                        K * in_bytes_per_element, K * in_bytes_per_element, m_rows, rtMemcpyDeviceToSram,
                        ctx.kernelStream);

        if (in_bytes_per_element == (int) sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(m_rows, Kr, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA_in, kFLOAT, kBF16, params);
            launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        if (!layout_identity_for_single_row) {
            params.clear();
            chw2chw32_blocks(1, m_rows, Kr, threadsPerBlock, blocksPerGrid);
            chw2chw32_align_params((int) sramA_in, (int) sramA_out, m_rows, Kr, 0, (int) threadsPerBlock.x,
                                   (int) threadsPerBlock.y, (int) threadsPerBlock.z, (int) sizeof(uint16_t), params,
                                   false);
            launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                               ctx.rppBinMod, ctx.kernelStream);
        }

        for (int i = 0; i < get_tiles(Nr); i++) {
            const int          tile_start = i * MATMUL_NS;
            const int          ns         = std::min(MATMUL_NS, Nr - tile_start);
            const int          valid_ns   = std::max(0, std::min(MATMUL_NS, N - tile_start));
            const int          sizeB_cur  = Kr * ns * (int) sizeof(uint16_t);
            const RPPdeviceptr devB_tile  = devB_base + (RPPdeviceptr) Kr * tile_start * (int) sizeof(uint16_t);

            rtMemcpyAsync((void *) sramB_out, (const void *) devB_tile, sizeB_cur, rtMemcpyDeviceToSram,
                          ctx.kernelStream);

            params.clear();
            matmul_chw32_blocks(1, m_rows, ns, threadsPerBlock, blocksPerGrid, get_tn(ns), 0);
            matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A_cur, (uint32_t) sramB_out, 0,
                                     (uint32_t) sramC_hw32, m_rows, Kr, Kr, ns, 1, 1, 1, 0, get_tn(ns),
                                     (int) sizeof(uint16_t), (int) sizeof(uint16_t), params, false, false);
            launchWrapperAysnc(get_matmul_kernel(ns), blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            const RPPdeviceptr linear_c_src = layout_identity_for_single_row ? sramC_hw32 : sramC_chw;
            if (!layout_identity_for_single_row) {
                params.clear();
                chw322chw_blocks(1, m_rows, ns, threadsPerBlock, blocksPerGrid);
                chw322chw_align_params((int) sramC_hw32, (int) sramC_chw, m_rows, ns, 0, threadsPerBlock.x,
                                       threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(uint16_t), params);
                launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                                   ctx.rppBinMod, ctx.kernelStream);
            }

            if (out_bytes_per_element == (int) sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(m_rows, ns * 2, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init_opt(threadsPerBlock, linear_c_src, sramC_chw_fp32, kBF16, kFLOAT, params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);

                if (valid_ns > 0) {
                    rtMemcpy2DAsync(
                        (void *) (devC_base + ((RPPdeviceptr) m_start * N + tile_start) * (int) sizeof(float)),
                        N * (int) sizeof(float), (void *) sramC_chw_fp32, ns * (int) sizeof(float),
                        valid_ns * (int) sizeof(float), m_rows, rtMemcpySramToDevice, ctx.kernelStream);
                }
            } else {
                if (valid_ns > 0) {
                    rtMemcpy2DAsync(
                        (void *) (devC_base + ((RPPdeviceptr) m_start * N + tile_start) * (int) sizeof(uint16_t)),
                        N * (int) sizeof(uint16_t), (void *) linear_c_src, ns * (int) sizeof(uint16_t),
                        valid_ns * (int) sizeof(uint16_t), m_rows, rtMemcpySramToDevice, ctx.kernelStream);
                }
            }
        }
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}
}  // namespace kernel_bf16
