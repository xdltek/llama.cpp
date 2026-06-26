#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul_mat/kernel_q6_k/rpp_kernel_block.h"
#include "rpp_mul_mat/kernel_q6_k/rpp_kernel_param.h"
#include "rpp_runtime.h"

#include <assert.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace kernel_q6_k {
static inline bool q6k_bypass_super_scale_cpu_enabled() {
    const char * v = getenv("GGML_RPP_Q6K_BYPASS_SUPER_SCALE_CPU");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

static inline bool q6k_bypass_dequant_cpu_enabled() {
    const char * v = getenv("GGML_RPP_Q6K_BYPASS_DEQUANT_CPU");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

static inline float q6k_bf16_to_float(uint16_t v) {
    uint32_t u = ((uint32_t) v) << 16;
    float    f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static void q6k_merge_scales_cpu(const uint16_t * qscale,
                                 const uint16_t * super_scale,
                                 int              K,
                                 int              N,
                                 uint16_t *       out_scale) {
    const int super_group = 256;
    const int group       = 16;

    const int n_super_groups        = K / super_group;
    const int qscale_rows_per_sg    = super_group / group / 2;  // 8 packed rows of int8 pairs
    const int out_scale_rows_per_sg = super_group / group;      // 16 bf16 rows

    for (int sg = 0; sg < n_super_groups; ++sg) {
        for (int n = 0; n < N; ++n) {
            const float d = q6k_bf16_to_float(super_scale[(size_t) sg * (size_t) N + (size_t) n]);
            for (int r = 0; r < qscale_rows_per_sg; ++r) {
                const uint16_t w  = qscale[(size_t) (sg * qscale_rows_per_sg + r) * (size_t) N + (size_t) n];
                const int8_t   s0 = (int8_t) (w & 0xFF);
                const int8_t   s1 = (int8_t) ((w >> 8) & 0xFF);

                const size_t out_row0                         = (size_t) (sg * out_scale_rows_per_sg + 2 * r + 0);
                const size_t out_row1                         = (size_t) (sg * out_scale_rows_per_sg + 2 * r + 1);
                out_scale[out_row0 * (size_t) N + (size_t) n] = float_to_bf16_rne((float) s0 * d);
                out_scale[out_row1 * (size_t) N + (size_t) n] = float_to_bf16_rne((float) s1 * d);
            }
        }
    }
}

// Decode packed q6k weights to BF16 using merged per-group scales.
// Layout:
//   wqlsb : [K/4][N] packed 4-bit lows
//   wqmsb : [K/8][N] packed 2-bit highs
//   scale : [K/16][N] BF16 merged scales (qscale * super_scale)
//   out_B : [K][N] BF16 decoded weights
static void q6k_dequant_bf16_cpu(const uint16_t * wqlsb,
                                 const uint16_t * wqmsb,
                                 const uint16_t * scale,
                                 int              K,
                                 int              N,
                                 uint16_t *       out_B) {
    const int group = 16;
    for (int k = 0; k < K; ++k) {
        const size_t ql_row   = (size_t) (k / 4) * (size_t) N;
        const size_t qh_row   = (size_t) (k / 8) * (size_t) N;
        const size_t sc_row   = (size_t) (k / group) * (size_t) N;
        const int    ql_shift = (k & 3) * 4;
        const int    qh_shift = (k & 7) * 2;
        for (int n = 0; n < N; ++n) {
            const uint16_t ql = wqlsb[ql_row + (size_t) n];
            const uint16_t qh = wqmsb[qh_row + (size_t) n];
            const int      lo = (int) ((ql >> ql_shift) & 0xF);
            const int      hi = (int) ((qh >> qh_shift) & 0x3);
            const int      q  = ((hi << 4) | lo) - 32;

            const float sc                              = q6k_bf16_to_float(scale[sc_row + (size_t) n]);
            out_B[(size_t) k * (size_t) N + (size_t) n] = float_to_bf16_rne((float) q * sc);
        }
    }
}

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

inline int get_tiles(int N, int tile_ns = MATMUL_NS) {
    return (N + tile_ns - 1) / tile_ns;
}

inline int get_ns(int N, bool is_tail, int tile_ns = MATMUL_NS) {
    if (is_tail) {
        return N - (get_tiles(N, tile_ns) - 1) * tile_ns;
    } else {
        return tile_ns;
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

static void rpp_matmul_q6k(rpp_kernel_context & ctx,
                           int                  M,
                           int                  K,
                           int                  N,
                           int                  weights_group,
                           int                  in_bytes_per_element,
                           int                  out_bytes_per_element,
                           int                  tile_ns,
                           int                  is_instantial);

inline int choose_m_tile(int M, int K, int in_bytes_per_element, bool use_pipeline, int tile_ns = MATMUL_NS) {
    const int SRAM_LIMIT = 22 * 1024 * 1024;

    const bool bypass_dequant_cpu = q6k_bypass_dequant_cpu_enabled();

    const int weights_group = 256;
    const int group         = 16;
    const int bits_per_wqlsb  = 4;
    const int bits_per_wqmsb  = 2;
    const int bits_per_qscale = 8;
    const int wqlsb_per_word  = (int) sizeof(short) * 8 / bits_per_wqlsb;
    const int wqmsb_per_word  = (int) sizeof(short) * 8 / bits_per_wqmsb;
    const int qscale_per_word = (int) sizeof(short) * 8 / bits_per_qscale;

    const int sizeB_wqlsb      = K * tile_ns * (int) sizeof(short) / wqlsb_per_word;
    const int sizeB_wqmsb      = K * tile_ns * (int) sizeof(short) / wqmsb_per_word;
    const int size_super_scale = K * tile_ns / weights_group * (int) sizeof(rpp::bfloat16);
    const int size_qscale      = K * tile_ns / group * (int) sizeof(short) / qscale_per_word;
    const int size_scale       = K * tile_ns / group * (int) sizeof(short);
    const int sizeB            = K * tile_ns * (int) sizeof(rpp::bfloat16);
    const int b_ping_buffers   = (!bypass_dequant_cpu && use_pipeline) ? 2 : 1;

    auto calc_total_sram_bytes = [&](int m_rows) -> int64_t {
        const int sizeA0  = m_rows * K * in_bytes_per_element;
        const int sizeA1  = m_rows * K * (int) sizeof(rpp::bfloat16);
        const int sizeCr  = m_rows * tile_ns * (int) sizeof(rpp::bfloat16);
        const int sizeC32 = m_rows * tile_ns * (int) sizeof(float);

        int64_t total = 0;
        total += round_up(sizeA0);
        total += round_up(sizeA1);
        if (!bypass_dequant_cpu) {
            total += b_ping_buffers * (round_up(sizeB_wqlsb) + round_up(sizeB_wqmsb) + round_up(size_super_scale) +
                                       round_up(size_qscale));
            total += round_up(size_scale);
            total += round_up(sizeB);
        } else {
            total += round_up(sizeB);
            total += round_up(sizeB);
        }
        total += round_up(sizeCr);
        total += round_up(sizeCr);
        total += round_up(sizeC32);
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

    return Mtile;
}

struct q6k_full_m_strategy {
    bool use_pipeline = false;
    int  tile_ns      = 0;
};

inline q6k_full_m_strategy choose_full_m_strategy(int M, int K, int in_bytes_per_element, bool prefer_pipeline) {
    const int candidates[] = {MATMUL_NS, 96, 64};

    for (const int tile_ns : candidates) {
        if (prefer_pipeline && choose_m_tile(M, K, in_bytes_per_element, true, tile_ns) >= M) {
            return {true, tile_ns};
        }
        if (choose_m_tile(M, K, in_bytes_per_element, false, tile_ns) >= M) {
            return {false, tile_ns};
        }
    }

    return {};
}

static inline RPPdeviceptr stage_a_tile(rpp_kernel_context &    ctx,
                                        RPPdeviceptr            devA_base,
                                        int                     m_start,
                                        int                     m_rows,
                                        int                     K,
                                        int                     in_bytes_per_element,
                                        RPPdeviceptr            sramA_in,
                                        RPPdeviceptr            sramA_out,
                                        dim3 &                  threadsPerBlock,
                                        dim3 &                  blocksPerGrid,
                                        std::vector<uint32_t> & params) {
    rtMemcpyAsync((void *) sramA_in, (const void *) (devA_base + (RPPdeviceptr) m_start * K * in_bytes_per_element),
                  m_rows * K * in_bytes_per_element, rtMemcpyDeviceToSram, ctx.kernelStream);

    if (in_bytes_per_element == (int) sizeof(float)) {
        params.clear();
        calc_tbdim_flattern(m_rows, K, threadsPerBlock, blocksPerGrid);
        cvt_kernel_param_init(threadsPerBlock, sramA_in, sramA_out, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);

        params.clear();
        chw2chw32_blocks(1, m_rows, K, threadsPerBlock, blocksPerGrid);
        chw2chw32_align_params((int) sramA_out, (int) sramA_in, m_rows, K, 0, (int) threadsPerBlock.x,
                               (int) threadsPerBlock.y, (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params,
                               false);
        launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                           ctx.rppBinMod, ctx.kernelStream);

        return sramA_in;
    }

    params.clear();
    chw2chw32_blocks(1, m_rows, K, threadsPerBlock, blocksPerGrid);
    chw2chw32_align_params((int) sramA_in, (int) sramA_out, m_rows, K, 0, (int) threadsPerBlock.x,
                           (int) threadsPerBlock.y, (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params,
                           false);
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                       ctx.kernelStream);

    return sramA_out;
}

static void rpp_matmul_q6k_tiled(rpp_kernel_context & ctx,
                                 int                  M,
                                 int                  K,
                                 int                  N,
                                 int                  weights_group,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  is_instantial = 1) {
    if (ctx.dev_in.size() < 5 || ctx.dev_out.empty()) {
        throw std::runtime_error("Matmul Q6K invalid IO buffers");
    }

    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;

    RPPdeviceptr devA_base             = ctx.dev_in[0];
    RPPdeviceptr devB_wqlsb_base       = ctx.dev_in[1];
    RPPdeviceptr devB_wqmsb_base       = ctx.dev_in[2];
    RPPdeviceptr devB_scale_base       = ctx.dev_in[3];
    RPPdeviceptr devB_super_scale_base = ctx.dev_in[4];
    RPPdeviceptr devC_base             = ctx.dev_out[0];

    const bool   bypass_dequant_cpu     = q6k_bypass_dequant_cpu_enabled();
    const bool   bypass_super_scale_cpu = q6k_bypass_super_scale_cpu_enabled() || bypass_dequant_cpu;
    RPPdeviceptr devB_scale_merged      = 0;
    RPPdeviceptr devB_dequant_merged    = 0;

    const int super_group         = 256;
    const int group               = 16;
    const int elements_per_thread = 16;
    const int bits_per_wqlsb      = 4;
    const int bits_per_wqmsb      = 2;
    const int bits_per_qscale     = 8;
    const int wqlsb_per_word      = (int) sizeof(short) * 8 / bits_per_wqlsb;
    const int wqmsb_per_word      = (int) sizeof(short) * 8 / bits_per_wqmsb;
    const int qscale_per_word     = (int) sizeof(short) * 8 / bits_per_qscale;

    assert(super_group == weights_group);
    assert(group == elements_per_thread);

    if (bypass_super_scale_cpu) {
        const size_t qscale_words_all = (size_t) K / (size_t) group / (size_t) qscale_per_word * (size_t) N;
        const size_t super_words_all  = (size_t) K / (size_t) weights_group * (size_t) N;
        const size_t scale_words_all  = (size_t) K / (size_t) group * (size_t) N;

        std::vector<uint16_t> host_qscale(qscale_words_all);
        std::vector<uint16_t> host_super(super_words_all);
        std::vector<uint16_t> host_scale(scale_words_all);

        rtMemcpyAsync((void *) host_qscale.data(), (const void *) devB_scale_base, qscale_words_all * sizeof(uint16_t),
                      rtMemcpyDeviceToHost, ctx.kernelStream);
        rtMemcpyAsync((void *) host_super.data(), (const void *) devB_super_scale_base,
                      super_words_all * sizeof(uint16_t), rtMemcpyDeviceToHost, ctx.kernelStream);
        rtStreamSynchronize(ctx.kernelStream);

        q6k_merge_scales_cpu(host_qscale.data(), host_super.data(), K, N, host_scale.data());

        if (!bypass_dequant_cpu) {
            if (rtMalloc((void **) &devB_scale_merged, scale_words_all * sizeof(uint16_t)) != rtSuccess) {
                throw std::runtime_error("q6k debug bypass: failed to allocate merged-scale device buffer");
            }
            ctx.dev_owned.emplace_back(devB_scale_merged);
            rtMemcpyAsync((void *) devB_scale_merged, (const void *) host_scale.data(),
                          scale_words_all * sizeof(uint16_t), rtMemcpyHostToDevice, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);
        } else {
            const size_t wqlsb_words_all   = (size_t) K / (size_t) wqlsb_per_word * (size_t) N;
            const size_t wqmsb_words_all   = (size_t) K / (size_t) wqmsb_per_word * (size_t) N;
            const size_t dequant_words_all = (size_t) K * (size_t) N;

            std::vector<uint16_t> host_wqlsb(wqlsb_words_all);
            std::vector<uint16_t> host_wqmsb(wqmsb_words_all);
            std::vector<uint16_t> host_B(dequant_words_all);

            rtMemcpyAsync((void *) host_wqlsb.data(), (const void *) devB_wqlsb_base,
                          wqlsb_words_all * sizeof(uint16_t), rtMemcpyDeviceToHost, ctx.kernelStream);
            rtMemcpyAsync((void *) host_wqmsb.data(), (const void *) devB_wqmsb_base,
                          wqmsb_words_all * sizeof(uint16_t), rtMemcpyDeviceToHost, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);

            q6k_dequant_bf16_cpu(host_wqlsb.data(), host_wqmsb.data(), host_scale.data(), K, N, host_B.data());

            if (rtMalloc((void **) &devB_dequant_merged, dequant_words_all * sizeof(uint16_t)) != rtSuccess) {
                throw std::runtime_error("q6k debug bypass: failed to allocate CPU-dequant device buffer");
            }
            ctx.dev_owned.emplace_back(devB_dequant_merged);
            rtMemcpyAsync((void *) devB_dequant_merged, (const void *) host_B.data(),
                          dequant_words_all * sizeof(uint16_t), rtMemcpyHostToDevice, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);
        }

        static bool printed_scale_once   = false;
        static bool printed_dequant_once = false;
        if (!bypass_dequant_cpu && !printed_scale_once) {
            std::cout << "[Q6K DBG] GGML_RPP_Q6K_BYPASS_SUPER_SCALE_CPU=1 (using CPU-merged scales)" << std::endl;
            printed_scale_once = true;
        }
        if (bypass_dequant_cpu && !printed_dequant_once) {
            std::cout << "[Q6K DBG] GGML_RPP_Q6K_BYPASS_DEQUANT_CPU=1 (using CPU dequantized B weights)" << std::endl;
            printed_dequant_once = true;
        }
    }

    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q6k.o");

    const int Mtile = choose_m_tile(M, K, in_bytes_per_element, false);

    const int sizeA0 = Mtile * K * in_bytes_per_element;
    const int sizeA1 = Mtile * K * (int) sizeof(rpp::bfloat16);

    const int sizeB_wqlsb      = K * MATMUL_NS * (int) sizeof(short) / wqlsb_per_word;
    const int sizeB_wqmsb      = K * MATMUL_NS * (int) sizeof(short) / wqmsb_per_word;
    const int size_super_scale = K * MATMUL_NS / weights_group * (int) sizeof(rpp::bfloat16);
    const int size_qscale      = K * MATMUL_NS / group * (int) sizeof(short) / qscale_per_word;
    const int size_scale       = K * MATMUL_NS / group * (int) sizeof(short);
    const int sizeB            = K * MATMUL_NS * (int) sizeof(rpp::bfloat16);
    const int sizeCr           = Mtile * MATMUL_NS * (int) sizeof(rpp::bfloat16);
    const int sizeC32          = Mtile * MATMUL_NS * (int) sizeof(float);

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    RPPdeviceptr sramA_in          = sram_base;
    RPPdeviceptr sramA_out         = sramA_in + round_up(sizeA0);
    RPPdeviceptr sramB_wqlsb       = 0;
    RPPdeviceptr sramB_wqmsb       = 0;
    RPPdeviceptr sramB_super_scale = 0;
    RPPdeviceptr sramB_qscale      = 0;
    RPPdeviceptr sramB_scale       = 0;
    RPPdeviceptr sramB_linear      = 0;
    RPPdeviceptr sramB_out         = 0;
    if (!bypass_dequant_cpu) {
        sramB_wqlsb       = sramA_out + round_up(sizeA1);
        sramB_wqmsb       = sramB_wqlsb + round_up(sizeB_wqlsb);
        sramB_super_scale = sramB_wqmsb + round_up(sizeB_wqmsb);
        sramB_qscale      = sramB_super_scale + round_up(size_super_scale);
        sramB_scale       = sramB_qscale + round_up(size_qscale);
        sramB_out         = sramB_scale + round_up(size_scale);
    } else {
        sramB_linear = sramA_out + round_up(sizeA1);
        sramB_out    = sramB_linear + round_up(sizeB);
    }
    RPPdeviceptr sramC_hw32       = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw        = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32   = sramC_chw + round_up(sizeCr);
    RPPdeviceptr sramB_super_zero = 0;
    RPPdeviceptr sramB_lut        = 0;
    const int    total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int    SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);

    const int m_tiles = (M + Mtile - 1) / Mtile;
    for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
        const int          m_start  = m_tile * Mtile;
        const int          m_rows   = std::min(Mtile, M - m_start);
        const RPPdeviceptr matmul_A = stage_a_tile(ctx, devA_base, m_start, m_rows, K, in_bytes_per_element, sramA_in,
                                                   sramA_out, threadsPerBlock, blocksPerGrid, params);

        for (int i = 0; i < get_tiles(N); ++i) {
            const bool is_tail = (i == (get_tiles(N) - 1));
            const int  ns      = get_ns(N, is_tail);
            const int  n_start = i * MATMUL_NS;

            if (!bypass_dequant_cpu) {
                rtMemcpy2DAsync((void *) sramB_wqlsb, ns * (int) sizeof(short),
                                (const void *) (devB_wqlsb_base + (RPPdeviceptr) n_start * sizeof(short)),
                                N * (int) sizeof(short), ns * (int) sizeof(short), K / wqlsb_per_word,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_wqmsb, ns * (int) sizeof(short),
                                (const void *) (devB_wqmsb_base + (RPPdeviceptr) n_start * sizeof(short)),
                                N * (int) sizeof(short), ns * (int) sizeof(short), K / wqmsb_per_word,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                if (!bypass_super_scale_cpu) {
                    rtMemcpy2DAsync((void *) sramB_qscale, ns * (int) sizeof(short),
                                    (const void *) (devB_scale_base + (RPPdeviceptr) n_start * sizeof(short)),
                                    N * (int) sizeof(short), ns * (int) sizeof(short), K / group / qscale_per_word,
                                    rtMemcpyDeviceToSram, ctx.kernelStream);

                    rtMemcpy2DAsync((void *) sramB_super_scale, ns * (int) sizeof(short),
                                    (const void *) (devB_super_scale_base + (RPPdeviceptr) n_start * sizeof(short)),
                                    N * (int) sizeof(short), ns * (int) sizeof(short), K / weights_group,
                                    rtMemcpyDeviceToSram, ctx.kernelStream);

                    params.clear();
                    matmul_super_scale_blocks(K, weights_group, group, ns, threadsPerBlock, blocksPerGrid);
                    matmul_super_scale_params(sramB_qscale, sramB_super_scale, sramB_scale, K, ns, weights_group,
                                              group, blocksPerGrid, threadsPerBlock, params);
                    launchWrapperAysnc("matrix_mul_q6k_super_scale", blocksPerGrid, threadsPerBlock, params,
                                       ctx.rppBinMod, ctx.kernelStream);
                } else {
                    rtMemcpy2DAsync((void *) sramB_scale, ns * (int) sizeof(short),
                                    (const void *) (devB_scale_merged + (RPPdeviceptr) n_start * sizeof(short)),
                                    N * (int) sizeof(short), ns * (int) sizeof(short), K / group,
                                    rtMemcpyDeviceToSram, ctx.kernelStream);
                }

                params.clear();
                matmul_q6k_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
                matmul_dequant_params(blocksPerGrid, threadsPerBlock, sramB_wqlsb, sramB_wqmsb, sramB_scale,
                                      sramB_super_zero, sramB_out, sramB_lut, ns, K, (int) sizeof(short),
                                      (int) sizeof(short), params);
                launchWrapperAysnc("matrix_mul_q6k_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            } else {
                rtMemcpy2DAsync((void *) sramB_linear, ns * (int) sizeof(short),
                                (const void *) (devB_dequant_merged + (RPPdeviceptr) n_start * sizeof(short)),
                                N * (int) sizeof(short), ns * (int) sizeof(short), K, rtMemcpyDeviceToSram,
                                ctx.kernelStream);

                params.clear();
                chw2chw32_blocks(1, K, ns, threadsPerBlock, blocksPerGrid);
                chw2chw32_align_params((int) sramB_linear, (int) sramB_out, K, ns, 0, (int) threadsPerBlock.x,
                                       (int) threadsPerBlock.y, (int) threadsPerBlock.z, (int) sizeof(rpp::bfloat16),
                                       params, false);
                launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                                   ctx.rppBinMod, ctx.kernelStream);
            }

            params.clear();
            const int tn = get_tn(ns);
            matmul_chw32_blocks(1, m_rows, ns, threadsPerBlock, blocksPerGrid, tn, 0);
            matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) sramB_out, 0,
                                     (uint32_t) sramC_hw32, m_rows, K, K, ns, 1, 1, 1, 0, tn,
                                     (int) sizeof(rpp::bfloat16), (int) sizeof(rpp::bfloat16), params, false, false);
            launchWrapperAysnc(get_matmul_kernel(ns), blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            params.clear();
            chw322chw_blocks(1, m_rows, ns, threadsPerBlock, blocksPerGrid);
            chw322chw_align_params((int) sramC_hw32, (int) sramC_chw, m_rows, ns, 0, threadsPerBlock.x,
                                   threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
            launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                               ctx.rppBinMod, ctx.kernelStream);

            if (out_bytes_per_element == (int) sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(m_rows, ns * 2, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);

                rtMemcpy2DAsync((void *) (devC_base + ((RPPdeviceptr) m_start * N + n_start) * (int) sizeof(float)),
                                N * (int) sizeof(float), (void *) sramC_chw_fp32, ns * (int) sizeof(float),
                                ns * (int) sizeof(float), m_rows, rtMemcpySramToDevice, ctx.kernelStream);
            } else {
                rtMemcpy2DAsync((void *) (devC_base + ((RPPdeviceptr) m_start * N + n_start) * (int) sizeof(short)),
                                N * (int) sizeof(short), (void *) sramC_chw, ns * (int) sizeof(short),
                                ns * (int) sizeof(short), m_rows, rtMemcpySramToDevice, ctx.kernelStream);
            }
        }
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_matmul_q6k_pipeline(rpp_kernel_context & ctx,
                                    int                  M,
                                    int                  K,
                                    int                  N,
                                    int                  weights_group,
                                    int                  in_bytes_per_element,
                                    int                  out_bytes_per_element,
                                    int                  tile_ns = MATMUL_NS,
                                    int                  is_instantial = 1) {
    if (ctx.dev_in.size() < 5 || ctx.dev_out.empty()) {
        throw std::runtime_error("Matmul Q6K invalid IO buffers");
    }

    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    int                   full_pitch, sub_pitch, offset;
    RPPdeviceptr          devA             = ctx.dev_in[0];
    RPPdeviceptr          devB_wqlsb       = ctx.dev_in[1];
    RPPdeviceptr          devB_wqmsb       = ctx.dev_in[2];
    RPPdeviceptr          devB_scale       = ctx.dev_in[3];
    RPPdeviceptr          devB_super_scale = ctx.dev_in[4];
    RPPdeviceptr          devB_zero        = 0;
    RPPdeviceptr          devC             = ctx.dev_out[0];
    (void) devB_zero;

    const bool   bypass_dequant_cpu     = q6k_bypass_dequant_cpu_enabled();
    const bool   bypass_super_scale_cpu = q6k_bypass_super_scale_cpu_enabled() || bypass_dequant_cpu;
    RPPdeviceptr devB_scale_merged      = 0;
    RPPdeviceptr devB_dequant_merged    = 0;

    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q6k.o");

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
    const int sizeA0 = (int) (M) * (K) *in_bytes_per_element;
    const int sizeA1 = (int) (M) * (K) * sizeof(rpp::bfloat16);

    const int sizeB_wqlsb      = (int) ((K) * (tile_ns) * sizeof(short) / wqlsb_per_word);
    const int sizeB_wqmsb      = (int) ((K) * (tile_ns) * sizeof(short) / wqmsb_per_word);
    const int size_super_scale = (int) ((K) * (tile_ns) / weights_group * sizeof(rpp::bfloat16));
    const int size_qscale      = (int) ((K) * (tile_ns) / group * sizeof(short) / qscale_per_word);
    const int size_scale       = (int) ((K) * (tile_ns) / group * sizeof(short));

    //const int sizeZero = (int)((K) * (MATMUL_NS) / weights_group * sizeof(rpp::bfloat16));
    //const int sizeLut = (int)(32);
    const int sizeB   = (int) ((K) * (tile_ns) * sizeof(rpp::bfloat16));
    const int sizeCr  = (int) ((M) * (tile_ns) * sizeof(rpp::bfloat16));
    const int sizeC32 = (int) ((M) * (tile_ns) * sizeof(float));

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    // Linear in + HW32 out for A/B
    RPPdeviceptr sramA_in            = sram_base;
    RPPdeviceptr sramA_out           = sramA_in + round_up(sizeA0);

    RPPdeviceptr sramB_wqlsb_0       = 0;
    RPPdeviceptr sramB_wqmsb_0       = 0;
    RPPdeviceptr sramB_super_scale_0 = 0;
    RPPdeviceptr sramB_qscale_0      = 0;
    RPPdeviceptr sramB_wqlsb_1       = 0;
    RPPdeviceptr sramB_wqmsb_1       = 0;
    RPPdeviceptr sramB_super_scale_1 = 0;
    RPPdeviceptr sramB_qscale_1      = 0;

    RPPdeviceptr sramB_scale         = 0;
    RPPdeviceptr sramB_linear        = 0;
    RPPdeviceptr sramB_out           = 0;
    if (!bypass_dequant_cpu) {
        // --- ping-pong buffer for B weights ---
        sramB_wqlsb_0       = sramA_out + round_up(sizeA1);
        sramB_wqmsb_0       = sramB_wqlsb_0 + round_up(sizeB_wqlsb);
        sramB_super_scale_0 = sramB_wqmsb_0 + round_up(sizeB_wqmsb);
        sramB_qscale_0      = sramB_super_scale_0 + round_up(size_super_scale);

        sramB_wqlsb_1       = sramB_qscale_0 + round_up(size_qscale);
        sramB_wqmsb_1       = sramB_wqlsb_1 + round_up(sizeB_wqlsb);
        sramB_super_scale_1 = sramB_wqmsb_1 + round_up(sizeB_wqmsb);
        sramB_qscale_1      = sramB_super_scale_1 + round_up(size_super_scale);

        sramB_scale       = sramB_qscale_1 + round_up(size_qscale);
        sramB_out         = sramB_scale + round_up(size_scale);
    } else {
        // Debug path: place CPU-dequantized linear B tile in SRAM, then convert to HW32.
        sramB_linear = sramA_out + round_up(sizeA1);
        sramB_out    = sramB_linear + round_up(sizeB);
    }
    RPPdeviceptr sramC_hw32       = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw        = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32   = sramC_chw + round_up(sizeCr);
    RPPdeviceptr sramB_super_zero = 0;
    RPPdeviceptr sramB_lut        = 0;
    const int    total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int    SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    if (bypass_super_scale_cpu) {
        const size_t qscale_words_all = (size_t) K / (size_t) group / (size_t) qscale_per_word * (size_t) N;
        const size_t super_words_all  = (size_t) K / (size_t) weights_group * (size_t) N;
        const size_t scale_words_all  = (size_t) K / (size_t) group * (size_t) N;

        std::vector<uint16_t> host_qscale(qscale_words_all);
        std::vector<uint16_t> host_super(super_words_all);
        std::vector<uint16_t> host_scale(scale_words_all);

        rtMemcpyAsync((void *) host_qscale.data(), (const void *) devB_scale, qscale_words_all * sizeof(uint16_t),
                      rtMemcpyDeviceToHost, ctx.kernelStream);
        rtMemcpyAsync((void *) host_super.data(), (const void *) devB_super_scale, super_words_all * sizeof(uint16_t),
                      rtMemcpyDeviceToHost, ctx.kernelStream);
        rtStreamSynchronize(ctx.kernelStream);

        q6k_merge_scales_cpu(host_qscale.data(), host_super.data(), K, N, host_scale.data());

        if (!bypass_dequant_cpu) {
            if (rtMalloc((void **) &devB_scale_merged, scale_words_all * sizeof(uint16_t)) != rtSuccess) {
                throw std::runtime_error("q6k debug bypass: failed to allocate merged-scale device buffer");
            }
            ctx.dev_owned.emplace_back(devB_scale_merged);
            rtMemcpyAsync((void *) devB_scale_merged, (const void *) host_scale.data(),
                          scale_words_all * sizeof(uint16_t), rtMemcpyHostToDevice, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);
        } else {
            const size_t wqlsb_words_all   = (size_t) K / (size_t) wqlsb_per_word * (size_t) N;
            const size_t wqmsb_words_all   = (size_t) K / (size_t) wqmsb_per_word * (size_t) N;
            const size_t dequant_words_all = (size_t) K * (size_t) N;

            std::vector<uint16_t> host_wqlsb(wqlsb_words_all);
            std::vector<uint16_t> host_wqmsb(wqmsb_words_all);
            std::vector<uint16_t> host_B(dequant_words_all);

            rtMemcpyAsync((void *) host_wqlsb.data(), (const void *) devB_wqlsb, wqlsb_words_all * sizeof(uint16_t),
                          rtMemcpyDeviceToHost, ctx.kernelStream);
            rtMemcpyAsync((void *) host_wqmsb.data(), (const void *) devB_wqmsb, wqmsb_words_all * sizeof(uint16_t),
                          rtMemcpyDeviceToHost, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);

            q6k_dequant_bf16_cpu(host_wqlsb.data(), host_wqmsb.data(), host_scale.data(), K, N, host_B.data());

            if (rtMalloc((void **) &devB_dequant_merged, dequant_words_all * sizeof(uint16_t)) != rtSuccess) {
                throw std::runtime_error("q6k debug bypass: failed to allocate CPU-dequant device buffer");
            }
            ctx.dev_owned.emplace_back(devB_dequant_merged);
            rtMemcpyAsync((void *) devB_dequant_merged, (const void *) host_B.data(),
                          dequant_words_all * sizeof(uint16_t), rtMemcpyHostToDevice, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);
        }

        static bool printed_scale_once   = false;
        static bool printed_dequant_once = false;
        if (!bypass_dequant_cpu && !printed_scale_once) {
            std::cout << "[Q6K DBG] GGML_RPP_Q6K_BYPASS_SUPER_SCALE_CPU=1 (using CPU-merged scales)" << std::endl;
            printed_scale_once = true;
        }
        if (bypass_dequant_cpu && !printed_dequant_once) {
            std::cout << "[Q6K DBG] GGML_RPP_Q6K_BYPASS_DEQUANT_CPU=1 (using CPU dequantized B weights)" << std::endl;
            printed_dequant_once = true;
        }
    }

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);

    // Matmul uses HW32 outputs as inputs
    RPPdeviceptr matmul_A = stage_a_tile(ctx, devA, 0, M, K, in_bytes_per_element, sramA_in, sramA_out,
                                         threadsPerBlock, blocksPerGrid, params);
    RPPdeviceptr matmul_B = sramB_out;
    RPPdeviceptr matmul_C = sramC_hw32;

    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);
    
    // -------------------------
    // for weights B
    // -------------------------
    const int nr_of_tiles = get_tiles(N, tile_ns);
    
    // --- NOTE: currently not enable pipeline for bypass_dequant_cpu == true ---
    if (bypass_dequant_cpu) {
        for (int i = 0; i < nr_of_tiles; ++ i) {
            bool is_tail = (i == (nr_of_tiles - 1));
            int  ns      = get_ns(N, is_tail, tile_ns);

            // CPU-dequantized B is linear [K][N]; convert to HW32 before matmul.
            rtMemcpy2DAsync((void *) sramB_linear, ns * sizeof(short),
                            (const void *) devB_dequant_merged, N * sizeof(short), ns * sizeof(short),
                            K, rtMemcpyDeviceToSram, ctx.kernelStream);

            params.clear();
            chw2chw32_blocks(1, K, ns, threadsPerBlock, blocksPerGrid);
            chw2chw32_align_params((int) sramB_linear, (int) sramB_out, K, ns, 0,
                                   (int) threadsPerBlock.x, (int) threadsPerBlock.y, (int) threadsPerBlock.z,
                                   (int) sizeof(rpp::bfloat16), params, false);
            launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                               ctx.rppBinMod, ctx.kernelStream);

            devB_dequant_merged += ns * sizeof(short);
            // -------------------------
            // (4) Matmul HW32
            // -------------------------
            params.clear();
            int         tn        = get_tn(ns);
            std::string kern_name = get_matmul_kernel(ns);
            matmul_chw32_blocks(1, M, ns, threadsPerBlock, blocksPerGrid, tn, 0);
            matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                                    (uint32_t) matmul_C, M, K, K, ns, 1, 1, 1, 0,
                                    tn, (int) sizeof(rpp::bfloat16), (int) sizeof(rpp::bfloat16),
                                    params, false, false);
            launchWrapperAysnc(kern_name, blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                            ctx.kernelStream);
            // -------------------------
            // (5) HW32 -> CHW (linear) into sramC_chw
            // -------------------------
            params.clear();
            chw322chw_blocks(1, M, ns, threadsPerBlock, blocksPerGrid);
            chw322chw_align_params((int) matmul_C, (int) sramC_chw, M, ns, 0, threadsPerBlock.x,
                                threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
            launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                            ctx.rppBinMod, ctx.kernelStream);

            if (out_bytes_per_element == sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(1, M * ns * 2, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                ctx.kernelStream);

                rtMemcpy2DAsync((void *) devC, N * sizeof(float), (void *) sramC_chw_fp32,
                                ns * sizeof(float), ns * sizeof(float), M,
                                rtMemcpySramToDevice, ctx.kernelStream);

                devC += ns * sizeof(float);
            } else {
                rtMemcpy2DAsync((void *) devC, N * sizeof(short), (void *) sramC_chw, ns * sizeof(short),
                                ns * sizeof(short), M, rtMemcpySramToDevice, ctx.kernelStream);
                devC += ns * sizeof(short);
            }
        }
    } else {
        auto sramB_wqlsb = [&](int ping) {
            return ping ? sramB_wqlsb_1 : sramB_wqlsb_0;
        };
        auto sramB_wqmsb = [&](int ping) {
            return ping ? sramB_wqmsb_1 : sramB_wqmsb_0;
        };
        auto sramB_qscale = [&](int ping) {
            return ping ? sramB_qscale_1 : sramB_qscale_0;
        };
        auto sramB_super_scale = [&](int ping) {
            return ping ? sramB_super_scale_1 : sramB_super_scale_0;
        };

        const int wqlsb_dma_rows_per_chunk = 128;
        const int wqmsb_dma_rows_per_chunk = 128;

        auto schedule_dma = [&](int ping, int NsSeg) {
            // Ensure previous kernel using this ping buffer has finished
            rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);
            {
                const size_t src_pitch = N * sizeof(short);
                const size_t dst_pitch = NsSeg * sizeof(short);
                const int    total_rows = K / wqlsb_per_word;

                for (int row_start = 0; row_start < total_rows; row_start += wqlsb_dma_rows_per_chunk) {
                    const int          chunk_rows = std::min(wqlsb_dma_rows_per_chunk, total_rows - row_start);
                    const RPPdeviceptr dst_chunk = sramB_wqlsb(ping) + (RPPdeviceptr) row_start * dst_pitch;
                    const RPPdeviceptr src_chunk = devB_wqlsb + (RPPdeviceptr) row_start * src_pitch;

                    rtMemcpy2DAsync((void *) dst_chunk, dst_pitch, (const void *) src_chunk, src_pitch, dst_pitch,
                                    chunk_rows, rtMemcpyDeviceToSram, ctx.dmaStream);
                }
            }

            {
                const size_t src_pitch = N * sizeof(short);
                const size_t dst_pitch = NsSeg * sizeof(short);
                const int    total_rows = K / wqmsb_per_word;

                for (int row_start = 0; row_start < total_rows; row_start += wqmsb_dma_rows_per_chunk) {
                    const int          chunk_rows = std::min(wqmsb_dma_rows_per_chunk, total_rows - row_start);
                    const RPPdeviceptr dst_chunk = sramB_wqmsb(ping) + (RPPdeviceptr) row_start * dst_pitch;
                    const RPPdeviceptr src_chunk = devB_wqmsb + (RPPdeviceptr) row_start * src_pitch;

                    rtMemcpy2DAsync((void *) dst_chunk, dst_pitch, (const void *) src_chunk, src_pitch, dst_pitch,
                                    chunk_rows, rtMemcpyDeviceToSram, ctx.dmaStream);
                }
            }
            if (!bypass_super_scale_cpu) {
                rtMemcpy2DAsync((void *) sramB_qscale(ping), NsSeg * sizeof(short), (const void *) devB_scale,
                                N * sizeof(short), NsSeg * sizeof(short), K / group / qscale_per_word,
                                rtMemcpyDeviceToSram, ctx.dmaStream);

                rtMemcpy2DAsync((void *) sramB_super_scale(ping), NsSeg * sizeof(short),
                                (const void *) devB_super_scale, N * sizeof(short), NsSeg * sizeof(short),
                                K / weights_group, rtMemcpyDeviceToSram, ctx.dmaStream);
            }
            // --- update data addr ---
            devB_wqlsb += NsSeg * sizeof(short);
            devB_wqmsb += NsSeg * sizeof(short);
            if (!bypass_super_scale_cpu) {
                devB_scale += NsSeg * sizeof(short);
                devB_super_scale += NsSeg * sizeof(short);
            }

            rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
        };
        
        // --- prefecth data for tile 0 ---
        {
            int  ping    = 0;
            bool is_tail = (0 == (nr_of_tiles - 1));
            int  ns      = get_ns(N, is_tail, tile_ns);
            schedule_dma(ping, ns);
        }

        for (int i = 0; i < nr_of_tiles; ++ i) {
            const bool is_tail = (i == (nr_of_tiles - 1));
            const int NsSeg    = get_ns(N, is_tail, tile_ns);
            const int ping     = (i & 1);

            // --- wait for current DMA (ping-specific) ---
            rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);

            // --- prefetch next segment (ping-specific) ---
            if (i + 1 < nr_of_tiles) {
                const bool next_is_tail    = ((i + 1) == (nr_of_tiles - 1));
                const int next_NsSeg       = get_ns(N, next_is_tail, tile_ns);
                const int next_ping        = ((i + 1) & 1);
                schedule_dma(next_ping, next_NsSeg);
            }
            
            // -------------------------
            // merge super_scale with scale
            // -------------------------
            if (!bypass_super_scale_cpu) {
                params.clear();
                matmul_super_scale_blocks(K, weights_group, group, NsSeg, threadsPerBlock, blocksPerGrid);
                matmul_super_scale_params(sramB_qscale(ping), sramB_super_scale(ping), sramB_scale, K, NsSeg,
                                          weights_group, group, blocksPerGrid, threadsPerBlock, params);
                launchWrapperAysnc("matrix_mul_q6k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            } else {
                rtMemcpy2DAsync((void *) sramB_scale, NsSeg * sizeof(short),
                                (const void *) devB_scale_merged, N * sizeof(short), NsSeg * sizeof(short),
                                K / group, rtMemcpyDeviceToSram, ctx.kernelStream);
                devB_scale_merged += NsSeg * sizeof(short);
            }
            // -------------------------
            // (3) dequant B from INT4 to Bfloat16
            // -------------------------
            params.clear();
            matmul_q6k_dequant_blocks(1, K, NsSeg, threadsPerBlock, blocksPerGrid, weights_group);
            matmul_dequant_params(blocksPerGrid, threadsPerBlock, sramB_wqlsb(ping), sramB_wqmsb(ping), sramB_scale,
                                  sramB_super_zero, sramB_out, sramB_lut, NsSeg, K, sizeof(short),
                                  sizeof(short), params);
            launchWrapperAysnc("matrix_mul_q6k_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            
            // -------------------------
            // (4) Matmul HW32
            // -------------------------
            params.clear();
            int         tn        = get_tn(NsSeg);
            std::string kern_name = get_matmul_kernel(NsSeg);
            matmul_chw32_blocks(1, M, NsSeg, threadsPerBlock, blocksPerGrid, tn, 0);
            matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                                    (uint32_t) matmul_C, M, K, K, NsSeg, 1, 1, 1, 0,
                                    tn, (int) sizeof(rpp::bfloat16), (int) sizeof(rpp::bfloat16),
                                    params, false, false);
            launchWrapperAysnc(kern_name, blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
            // -------------------------
            // (5) HW32 -> CHW (linear) into sramC_chw
            // -------------------------
            params.clear();
            chw322chw_blocks(1, M, NsSeg, threadsPerBlock, blocksPerGrid);
            chw322chw_align_params((int) matmul_C, (int) sramC_chw, M, NsSeg, 0, threadsPerBlock.x,
                                   threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
            launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                               ctx.rppBinMod, ctx.kernelStream);

            if (out_bytes_per_element == sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(1, M * NsSeg * 2, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                  ctx.kernelStream);

                rtMemcpy2DAsync((void *) devC, N * sizeof(float), (void *) sramC_chw_fp32,
                                NsSeg * sizeof(float), NsSeg * sizeof(float), M, rtMemcpySramToDevice, ctx.kernelStream);

                devC += NsSeg * sizeof(float);
            } else {
                rtMemcpy2DAsync((void *) devC, N * sizeof(short), (void *) sramC_chw, NsSeg * sizeof(short),
                                NsSeg * sizeof(short), M, rtMemcpySramToDevice, ctx.kernelStream);
                devC += NsSeg * sizeof(short);
            }
            
            // Signal this ping buffer is safe to reuse
            rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
        }
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q6k(rpp_kernel_context & ctx,
                          int                  M,
                          int                  K,
                          int                  N,
                          int                  weights_group,
                          int                  in_bytes_per_element,
                          int                  out_bytes_per_element,
                          int                  tile_ns = MATMUL_NS,
                          int                  is_instantial = 1) {
    if (ctx.dev_in.size() < 5 || ctx.dev_out.empty()) {
        throw std::runtime_error("Matmul Q6K invalid IO buffers");
    }

    dim3                  threadsPerBlock;
    dim3                  threadsPerBlockTail;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    int                   full_pitch, sub_pitch, offset;
    RPPdeviceptr          devA             = ctx.dev_in[0];
    RPPdeviceptr          devB_wqlsb       = ctx.dev_in[1];
    RPPdeviceptr          devB_wqmsb       = ctx.dev_in[2];
    RPPdeviceptr          devB_scale       = ctx.dev_in[3];
    RPPdeviceptr          devB_super_scale = ctx.dev_in[4];
    RPPdeviceptr          devB_zero        = 0;
    RPPdeviceptr          devC             = ctx.dev_out[0];
    (void) devB_zero;

    const bool   bypass_dequant_cpu     = q6k_bypass_dequant_cpu_enabled();
    const bool   bypass_super_scale_cpu = q6k_bypass_super_scale_cpu_enabled() || bypass_dequant_cpu;
    RPPdeviceptr devB_scale_merged      = 0;
    RPPdeviceptr devB_dequant_merged    = 0;

    RPPmodule cuMod;
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_q6k.o");

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
    const int sizeA0 = (int) (M) * (K) *in_bytes_per_element;
    const int sizeA1 = (int) (M) * (K) * sizeof(rpp::bfloat16);

    const int sizeB_wqlsb      = (int) ((K) * (tile_ns) * sizeof(short) / wqlsb_per_word);
    const int sizeB_wqmsb      = (int) ((K) * (tile_ns) * sizeof(short) / wqmsb_per_word);
    const int size_super_scale = (int) ((K) * (tile_ns) / weights_group * sizeof(rpp::bfloat16));
    const int size_qscale      = (int) ((K) * (tile_ns) / group * sizeof(short) / qscale_per_word);
    const int size_scale       = (int) ((K) * (tile_ns) / group * sizeof(short));

    //const int sizeZero = (int)((K) * (MATMUL_NS) / weights_group * sizeof(rpp::bfloat16));
    //const int sizeLut = (int)(32);
    const int sizeB   = (int) ((K) * (tile_ns) * sizeof(rpp::bfloat16));
    const int sizeCr  = (int) ((M) * (tile_ns) * sizeof(rpp::bfloat16));
    const int sizeC32 = (int) ((M) * (tile_ns) * sizeof(float));

    RPPdeviceptr sram_base = ctx.virtual_sram_base;

    // Linear in + HW32 out for A/B
    RPPdeviceptr sramA_in          = sram_base;
    RPPdeviceptr sramA_out         = sramA_in + round_up(sizeA0);
    RPPdeviceptr sramB_wqlsb       = 0;
    RPPdeviceptr sramB_wqmsb       = 0;
    RPPdeviceptr sramB_super_scale = 0;
    RPPdeviceptr sramB_qscale      = 0;
    RPPdeviceptr sramB_scale       = 0;
    RPPdeviceptr sramB_linear      = 0;
    RPPdeviceptr sramB_out         = 0;
    if (!bypass_dequant_cpu) {
        sramB_wqlsb       = sramA_out + round_up(sizeA1);
        sramB_wqmsb       = sramB_wqlsb + round_up(sizeB_wqlsb);
        sramB_super_scale = sramB_wqmsb + round_up(sizeB_wqmsb);
        sramB_qscale      = sramB_super_scale + round_up(size_super_scale);
        sramB_scale       = sramB_qscale + round_up(size_qscale);
        sramB_out         = sramB_scale + round_up(size_scale);
    } else {
        // Debug path: place CPU-dequantized linear B tile in SRAM, then convert to HW32.
        sramB_linear = sramA_out + round_up(sizeA1);
        sramB_out    = sramB_linear + round_up(sizeB);
    }
    RPPdeviceptr sramC_hw32       = sramB_out + round_up(sizeB);
    RPPdeviceptr sramC_chw        = sramC_hw32 + round_up(sizeCr);
    RPPdeviceptr sramC_chw_fp32   = sramC_chw + round_up(sizeCr);
    RPPdeviceptr sramB_super_zero = 0;
    RPPdeviceptr sramB_lut        = 0;
    const int    total_sram_bytes = (int) (sramC_chw_fp32 + round_up(sizeC32) - sram_base);
    const int    SRAM_LIMIT       = 22 * 1024 * 1024;
    if (total_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_sram_bytes << " bytes, but allocated " << SRAM_LIMIT << " bytes\n";
        std::abort();
    }

    if (bypass_super_scale_cpu) {
        const size_t qscale_words_all = (size_t) K / (size_t) group / (size_t) qscale_per_word * (size_t) N;
        const size_t super_words_all  = (size_t) K / (size_t) weights_group * (size_t) N;
        const size_t scale_words_all  = (size_t) K / (size_t) group * (size_t) N;

        std::vector<uint16_t> host_qscale(qscale_words_all);
        std::vector<uint16_t> host_super(super_words_all);
        std::vector<uint16_t> host_scale(scale_words_all);

        rtMemcpyAsync((void *) host_qscale.data(), (const void *) devB_scale, qscale_words_all * sizeof(uint16_t),
                      rtMemcpyDeviceToHost, ctx.kernelStream);
        rtMemcpyAsync((void *) host_super.data(), (const void *) devB_super_scale, super_words_all * sizeof(uint16_t),
                      rtMemcpyDeviceToHost, ctx.kernelStream);
        rtStreamSynchronize(ctx.kernelStream);

        q6k_merge_scales_cpu(host_qscale.data(), host_super.data(), K, N, host_scale.data());

        if (!bypass_dequant_cpu) {
            if (rtMalloc((void **) &devB_scale_merged, scale_words_all * sizeof(uint16_t)) != rtSuccess) {
                throw std::runtime_error("q6k debug bypass: failed to allocate merged-scale device buffer");
            }
            ctx.dev_owned.emplace_back(devB_scale_merged);
            rtMemcpyAsync((void *) devB_scale_merged, (const void *) host_scale.data(),
                          scale_words_all * sizeof(uint16_t), rtMemcpyHostToDevice, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);
        } else {
            const size_t wqlsb_words_all   = (size_t) K / (size_t) wqlsb_per_word * (size_t) N;
            const size_t wqmsb_words_all   = (size_t) K / (size_t) wqmsb_per_word * (size_t) N;
            const size_t dequant_words_all = (size_t) K * (size_t) N;

            std::vector<uint16_t> host_wqlsb(wqlsb_words_all);
            std::vector<uint16_t> host_wqmsb(wqmsb_words_all);
            std::vector<uint16_t> host_B(dequant_words_all);

            rtMemcpyAsync((void *) host_wqlsb.data(), (const void *) devB_wqlsb, wqlsb_words_all * sizeof(uint16_t),
                          rtMemcpyDeviceToHost, ctx.kernelStream);
            rtMemcpyAsync((void *) host_wqmsb.data(), (const void *) devB_wqmsb, wqmsb_words_all * sizeof(uint16_t),
                          rtMemcpyDeviceToHost, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);

            q6k_dequant_bf16_cpu(host_wqlsb.data(), host_wqmsb.data(), host_scale.data(), K, N, host_B.data());

            if (rtMalloc((void **) &devB_dequant_merged, dequant_words_all * sizeof(uint16_t)) != rtSuccess) {
                throw std::runtime_error("q6k debug bypass: failed to allocate CPU-dequant device buffer");
            }
            ctx.dev_owned.emplace_back(devB_dequant_merged);
            rtMemcpyAsync((void *) devB_dequant_merged, (const void *) host_B.data(),
                          dequant_words_all * sizeof(uint16_t), rtMemcpyHostToDevice, ctx.kernelStream);
            rtStreamSynchronize(ctx.kernelStream);
        }

        static bool printed_scale_once   = false;
        static bool printed_dequant_once = false;
        if (!bypass_dequant_cpu && !printed_scale_once) {
            std::cout << "[Q6K DBG] GGML_RPP_Q6K_BYPASS_SUPER_SCALE_CPU=1 (using CPU-merged scales)" << std::endl;
            printed_scale_once = true;
        }
        if (bypass_dequant_cpu && !printed_dequant_once) {
            std::cout << "[Q6K DBG] GGML_RPP_Q6K_BYPASS_DEQUANT_CPU=1 (using CPU dequantized B weights)" << std::endl;
            printed_dequant_once = true;
        }
    }

    // Capture on kernelStream (like CUDA graph capture pattern)
    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);

    // Matmul uses HW32 outputs as inputs
    RPPdeviceptr matmul_A = stage_a_tile(ctx, devA, 0, M, K, in_bytes_per_element, sramA_in, sramA_out,
                                         threadsPerBlock, blocksPerGrid, params);
    RPPdeviceptr matmul_B = sramB_out;
    RPPdeviceptr matmul_C = sramC_hw32;
    const int nr_of_tiles = get_tiles(N, tile_ns);
    for (int i = 0; i < nr_of_tiles; i++) {
        bool is_tail = (i == (nr_of_tiles - 1));
        int  ns      = get_ns(N, is_tail, tile_ns);

        if (!bypass_dequant_cpu) {
            rtMemcpy2DAsync((void *) sramB_wqlsb, ns * sizeof(short), (const void *) devB_wqlsb,
                            N * sizeof(short), ns * sizeof(short), K / wqlsb_per_word,
                            rtMemcpyDeviceToSram, ctx.kernelStream);

            rtMemcpy2DAsync((void *) sramB_wqmsb, ns * sizeof(short), (const void *) devB_wqmsb,
                            N * sizeof(short), ns * sizeof(short), K / wqmsb_per_word,
                            rtMemcpyDeviceToSram, ctx.kernelStream);

            if (!bypass_super_scale_cpu) {
                rtMemcpy2DAsync((void *) sramB_qscale, ns * sizeof(short), (const void *) devB_scale,
                                N * sizeof(short), ns * sizeof(short), K / group / qscale_per_word,
                                rtMemcpyDeviceToSram, ctx.kernelStream);

                rtMemcpy2DAsync((void *) sramB_super_scale, ns * sizeof(short),
                                (const void *) devB_super_scale, N * sizeof(short), ns * sizeof(short),
                                K / weights_group, rtMemcpyDeviceToSram, ctx.kernelStream);
                // -------------------------
                // merge super_scale with scale
                // -------------------------
                params.clear();
                matmul_super_scale_blocks(K, weights_group, group, ns, threadsPerBlock, blocksPerGrid);
                matmul_super_scale_params(sramB_qscale, sramB_super_scale, sramB_scale, K, ns,
                                          weights_group, group, blocksPerGrid, threadsPerBlock, params);
                launchWrapperAysnc("matrix_mul_q6k_super_scale", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            } else {
                rtMemcpy2DAsync((void *) sramB_scale, ns * sizeof(short),
                                (const void *) devB_scale_merged, N * sizeof(short), ns * sizeof(short),
                                K / group, rtMemcpyDeviceToSram, ctx.kernelStream);
            }
            // -------------------------
            // (3) dequant B from INT4 to Bfloat16
            // -------------------------
            params.clear();
            matmul_q6k_dequant_blocks(1, K, ns, threadsPerBlock, blocksPerGrid, weights_group);
            matmul_dequant_params(blocksPerGrid, threadsPerBlock, sramB_wqlsb, sramB_wqmsb, sramB_scale,
                                  sramB_super_zero, sramB_out, sramB_lut, ns, K, sizeof(short),
                                  sizeof(short), params);
            launchWrapperAysnc("matrix_mul_q6k_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);
        } else {
            // CPU-dequantized B is linear [K][N]; convert to HW32 before matmul.
            rtMemcpy2DAsync((void *) sramB_linear, ns * sizeof(short),
                            (const void *) devB_dequant_merged, N * sizeof(short), ns * sizeof(short),
                            K, rtMemcpyDeviceToSram, ctx.kernelStream);

            params.clear();
            chw2chw32_blocks(1, K, ns, threadsPerBlock, blocksPerGrid);
            chw2chw32_align_params((int) sramB_linear, (int) sramB_out, K, ns, 0,
                                   (int) threadsPerBlock.x, (int) threadsPerBlock.y, (int) threadsPerBlock.z,
                                   (int) sizeof(rpp::bfloat16), params, false);
            launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                               ctx.rppBinMod, ctx.kernelStream);
        }
        //jsq6k
        //return ;
        // -------------------------
        // (4) Matmul HW32
        // -------------------------
        params.clear();
        matmul_chw32_blocks(1, M, ns, threadsPerBlock, blocksPerGrid, get_tn(ns), 0);
        int         tn        = get_tn(ns);
        std::string kern_name = get_matmul_kernel(ns);
        matmul_opt_kernel_params(blocksPerGrid, threadsPerBlock, (uint32_t) matmul_A, (uint32_t) matmul_B, 0,
                                 (uint32_t) matmul_C, M, K, K, ns, 1, 1, 1, 0,
                                 get_tn(ns), (int) sizeof(rpp::bfloat16), (int) sizeof(rpp::bfloat16),
                                 params, false, false);

        launchWrapperAysnc(get_matmul_kernel(ns), blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        // -------------------------
        // (5) HW32 -> CHW (linear) into sramC_chw
        // -------------------------
        params.clear();
        chw322chw_blocks(1, M, ns, threadsPerBlock, blocksPerGrid);
        chw322chw_align_params((int) matmul_C, (int) sramC_chw, M, ns, 0, threadsPerBlock.x,
                               threadsPerBlock.y, threadsPerBlock.z, (int) sizeof(rpp::bfloat16), params);
        launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt", blocksPerGrid, threadsPerBlock, params,
                           ctx.rppBinMod, ctx.kernelStream);

        if (out_bytes_per_element == sizeof(float)) {
            params.clear();
            calc_tbdim_flattern(1, M * ns * 2, threadsPerBlock, blocksPerGrid);
            cvt_kernel_param_init_opt(threadsPerBlock, sramC_chw, sramC_chw_fp32, kBF16, kFLOAT, params);
            launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                               ctx.kernelStream);

            rtMemcpy2DAsync((void *) devC, N * sizeof(float), (void *) sramC_chw_fp32,
                            ns * sizeof(float), ns * sizeof(float), M,
                            rtMemcpySramToDevice, ctx.kernelStream);

            devC += ns * sizeof(float);
        } else {
            rtMemcpy2DAsync((void *) devC, N * sizeof(short), (void *) sramC_chw, ns * sizeof(short),
                            ns * sizeof(short), M, rtMemcpySramToDevice, ctx.kernelStream);
            devC += ns * sizeof(short);
        }

        if (!bypass_dequant_cpu) {
            devB_wqlsb += ns * sizeof(short);
            devB_wqmsb += ns * sizeof(short);
            if (!bypass_super_scale_cpu) {
                devB_scale += ns * sizeof(short);
                devB_super_scale += ns * sizeof(short);
            } else {
                devB_scale_merged += ns * sizeof(short);
            }
        } else {
            devB_dequant_merged += ns * sizeof(short);
        }
        //devB_zero += ns * sizeof(short);
    }

    // End capture after all enqueued work is defined
    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }
}

static void rpp_matmul_q6k_build(rpp_kernel_context & ctx,
                                 int                  M,
                                 int                  K,
                                 int                  N,
                                 int                  weights_group,
                                 int                  in_bytes_per_element,
                                 int                  out_bytes_per_element,
                                 int                  use_pipeline = 0,
                                 int                  is_instantial = 1) {
    const q6k_full_m_strategy full_m = choose_full_m_strategy(M, K, in_bytes_per_element, use_pipeline != 0);
    if (full_m.tile_ns > 0) {
        if (full_m.use_pipeline) {
            rpp_matmul_q6k_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                    full_m.tile_ns, is_instantial);
        } else {
            rpp_matmul_q6k(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                           full_m.tile_ns, is_instantial);
        }
        return;
    }

    if (choose_m_tile(M, K, in_bytes_per_element, use_pipeline != 0) < M) {
        rpp_matmul_q6k_tiled(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                             is_instantial);
        return;
    }

    if (use_pipeline) {
        rpp_matmul_q6k_pipeline(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element,
                                MATMUL_NS, is_instantial);
    } else {
        rpp_matmul_q6k(ctx, M, K, N, weights_group, in_bytes_per_element, out_bytes_per_element, MATMUL_NS,
                       is_instantial);
    }
}
}  // namespace kernel_q6_k
