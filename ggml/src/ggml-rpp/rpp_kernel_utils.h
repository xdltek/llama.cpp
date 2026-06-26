
#pragma once
#include "ggml.h"
#include "rpp_drv_api.h"
#include "rpp_half.h"
#include "rpp_quant_lut.h"

#include <assert.h>
#include <math.h>
#include <rpp_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifndef GGML_RPP_USE_DFS
#    define GGML_RPP_USE_DFS 0
#endif

#ifndef GGML_RPP_USE_DFS_FLEXIBLE
#    define GGML_RPP_USE_DFS_FLEXIBLE 0
#endif

#define KDC_H_DIM_X          32
#define KDC_H_EXT_DIM_X      128
#define KDC_H_MAX_EXECUT_NUM 255
#define KDC_H_MAX_THREAD_NUM 8160
#define LOADALN_GUARD        512

enum RppDataType : int {
    kFLOAT   = 0,
    kHALF    = 1,
    kBF16    = 2,
    kBF16x8  = 3,
    kINT16   = 4,
    kINT32   = 5,
    kINT8x16 = 6,
    kDOUBLE  = 7,
    kINT8x2  = 8,
    kINT8    = 9,
    INT8x4   = 10,
    kUINT8   = 11,
    kUINT16  = 12,
    kUINT32  = 13,
    kINT64   = 14,
    kBOOL    = 15,
};

static size_t GetRppElementSize(RppDataType dataType) {
    size_t dValue = 2;
    switch (dataType) {
        case RppDataType::kBF16:
        case RppDataType::kINT16:
        case RppDataType::kUINT16:
        case RppDataType::kHALF:
            break;
        case RppDataType::kINT8:
        case RppDataType::kUINT8:
        case RppDataType::kBOOL:
            dValue = 1;
            break;
        case RppDataType::kFLOAT:
        case RppDataType::kINT32:
        case RppDataType::kUINT32:
            dValue = 4;
            break;
        default:
            throw -1;
    }
    return dValue;
}

enum rppElementWiseType : int {
    RPP_ELEMWISE_MUL = 0,
    RPP_ELEMWISE_ADD = 1,
    RPP_ELEMWISE_DIV = 2,
};

inline int round_up(int a) {
    return (a + LOADALN_GUARD - 1) / LOADALN_GUARD * LOADALN_GUARD + LOADALN_GUARD;
}

static void grid_dim_adjustment(uint32_t & grid_dim, uint32_t block_dim, int size) {
    int  grid_size = (int) block_dim * (int) grid_dim;
    int  gap       = grid_size - size;
    auto ratio     = gap / (int) block_dim;
    if (ratio > 0) {
        grid_dim -= (uint32_t) ratio;
    }
}

// ggml_fp16_t from llama.cpp – we only need the raw 16 bits
using ggml_fp16_t = uint16_t;

// GGUF Q4_1 block layout (simplified from llama.cpp)
struct block_q4_1_dbg {
    ggml_fp16_t d;       // scale (ignored or used as-is, see below)
    ggml_fp16_t m;       // bias/min (ignored in this simple mapping)
    uint8_t     qs[16];  // 32 4-bit values packed (2 per byte)
};

// Simple bfloat16 representation as raw 16-bit
using bfloat16_u16 = uint16_t;

// Output container for RPP Q4 quant params
struct RppQ4Params {
    // RPP Q4 weights, packed 4x4-bit into uint16:
    // shape conceptually [K/4][N] flattened as weights[i * N + j]
    std::vector<uint16_t> weights_q4;

    // RPP Q4 scales: one bfloat16 scale per 32-element block along K per column.
    // We treat gguf's "d" (fp16) as the scale.
    // shape [K/32][N], flattened as scales[g * N + n]
    std::vector<bfloat16_u16> scales_bf16;

    // RPP Q4 zeros: int8 per group, shape [K/32][N], flattened as zeros[g * N + n].
    // For simplicity, we set all zeros to 0 (ignoring gguf's "m").
    std::vector<bfloat16_u16> zeros_bf16;

    // LUT: 16 entries 0..15 in bfloat16
    std::vector<bfloat16_u16> lut_bf16;
};

struct block_q8_0_dbg {
    uint16_t d_fp16;  // scale in fp16
    int8_t   qs[32];
};

struct RppQ8Params {
    // packed 2x int8 into uint16, conceptual shape [K/2][N]
    std::vector<uint16_t>     weights_q8;
    // bf16 scale per (K/32) group per column, shape [K/32][N]
    std::vector<bfloat16_u16> scales_bf16;
    // zeros (filled with 0), shape [K/32][N]
    std::vector<bfloat16_u16> zeros_bf16;
};

static const int QK_K = 256;
#pragma pack(push, 1)

struct block_q6_k_dbg {
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d;
};

#pragma pack(pop)

struct RppQ6kParams {
    // lower 4bits, shape conceptually [K/4][N] flattened as weights[i * N + j]
    std::vector<uint16_t> weights_ql;
    // upper 2bits, shape conceptually [K/8][N] flattened as weights[i * N + j]
    std::vector<uint16_t> weights_qh;

    // packed int8 scales: (k0,k1) -> uint16, shape [K/32][N]
    // layout: low byte = k0, high byte = k1
    std::vector<uint16_t> scales;

    // shape [K/256][N], flattened as super_scale[g * N + n]
    std::vector<bfloat16_u16> super_scale;
};

static inline size_t idx2(size_t r, size_t c, size_t ld) {
    return r * ld + c;
}

// -----------------------------------------------------------------------------
// Helpers: FP16 <-> FP32, FP32 -> BF16
// -----------------------------------------------------------------------------

// FP16 -> float32 (same logic as your half2float32)
static inline float half_to_float(uint16_t h) {
    // Reuse ggml's canonical conversion to keep behavior consistent
    // across all quant conversion paths (including subnormals).
    return ggml_fp16_to_fp32((ggml_fp16_t) h);
}

// float32 -> bfloat16 (round-to-nearest-even)
static inline bfloat16_u16 float_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    uint32_t lsb           = (x >> 16) & 1u;  // LSB of BF16
    uint32_t rounding_bias = 0x7FFFu + lsb;
    x += rounding_bias;

    return (bfloat16_u16) (x >> 16);
}

// -------------------------------------------------------------
// FP16 <-> FP32 helpers (simple IEEE-754 half format)
// -------------------------------------------------------------

// float32 -> fp16 (round-to-nearest-even)
static inline ggml_fp16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));

    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = (int32_t) ((x >> 23) & 0xFF) - 127 + 15;  // rebias exponent
    uint32_t mant = x & 0x7FFFFF;

    uint16_t h;

    if (exp <= 0) {
        // subnormal or zero
        if (exp < -10) {
            h = (uint16_t) (sign << 15);
        } else {
            mant |= 0x800000;                  // add implicit leading 1
            int32_t  shift        = 14 - exp;  // 24 - (exp + 10)
            uint32_t mant_rounded = mant >> shift;
            // round to nearest
            if (mant & (1u << (shift - 1))) {
                mant_rounded++;
            }
            h = (uint16_t) ((sign << 15) | (mant_rounded & 0x3FF));
        }
    } else if (exp >= 0x1F) {
        // overflow -> Inf
        h = (uint16_t) ((sign << 15) | (0x1F << 10));
    } else {
        // normal
        uint16_t half_exp  = (uint16_t) (exp & 0x1F);
        uint16_t half_mant = (uint16_t) (mant >> 13);
        // round to nearest
        if (mant & (1u << 12)) {
            half_mant++;
            if (half_mant == 0x400) {  // mant overflow
                half_mant = 0;
                half_exp++;
                if (half_exp >= 0x1F) {  // overflow -> Inf
                    half_exp  = 0x1F;
                    half_mant = 0;
                }
            }
        }
        h = (uint16_t) ((sign << 15) | (half_exp << 10) | (half_mant & 0x3FF));
    }
    return h;
}

static inline uint16_t float_to_bf16_rne(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    uint32_t lsb           = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;  // RNE
    u += rounding_bias;
    return (uint16_t) (u >> 16);
}

static inline float bf16_to_float(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float    x;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

static inline float fp16_to_float(uint16_t h) {
    // Reuse ggml's canonical conversion to avoid edge-case drift
    // (notably fp16 subnormal handling).
    return ggml_fp16_to_fp32((ggml_fp16_t) h);
}

// -----------------------------------------------------------------------------
// Main conversion: gguf Q4_1 -> RPP Q4
// -----------------------------------------------------------------------------

// Convert GGUF Q4_1 weights to RPP Q4 format (weights + scales + zeros + LUT).
//
// Matmul is A[M][K] * B[K][N] = C[M][N].
//
// GGUF layout:
//   - There are N "rows" (columns of B).
//   - Each row has (K / 32) blocks of type block_q4_1_dbg.
//   - Each block has d,m (fp16) + qs[16] holding 32 4-bit values.
//
// RPP layout we produce:
//   - weights_q4: packs 4 consecutive K-positions into uint16_t for each column.
//       let Ck = K / 4;
//       weights_q4[i * N + j] packs K indices {4*i+0, 4*i+1, 4*i+2, 4*i+3} at column j.
//   - scales_bf16: per 32-element block along K per column;
//       let G = K / 32;
//       scales_bf16[g * N + j] = bf16( half_to_float(src_row[g].d) )
//   - zeros_i8: same shape as scales_bf16; we simply set zeros_i8[...] = 0.
//   - lut_bf16: 16 entries 0..15 in bfloat16.
//
static void convert_q4_1_to_rpp(const block_q4_1_dbg * src,  // length N * (K/32)
                                int                    K,
                                int                    N,
                                RppQ4Params &          out) {
    assert(K % 32 == 0);
    assert(K % 4 == 0);
    if (K == 0 || N == 0) {
        out.weights_q4.clear();
        out.scales_bf16.clear();
        out.zeros_bf16.clear();
        out.lut_bf16.clear();
        return;
    }
    const int            blocks_per_row = K / 32;  // # of Q4_1 blocks per row
    const int            Ck             = K / 4;   // # of uint16 packs along K
    // tmp_int4[k * N + n] = q (0..15) for position (k,n)
    std::vector<uint8_t> tmp_int4((size_t) K * (size_t) N);

    for (int n = 0; n < N; ++n) {
        const block_q4_1_dbg * row_blocks = src + (size_t) n * blocks_per_row;

        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_1_dbg & blk = row_blocks[b];

            // ggml packing: qs[i] holds q[i] in low nibble and q[i+16] in high nibble
            for (int i = 0; i < 16; ++i) {
                uint8_t v  = blk.qs[i];
                uint8_t q0 = v & 0x0F;         // q[i]
                uint8_t q1 = (v >> 4) & 0x0F;  // q[i+16]

                int k0 = b * 32 + i;           // 0..15
                int k1 = b * 32 + i + 16;      // 16..31

                tmp_int4[(size_t) k0 * N + n] = q0;
                tmp_int4[(size_t) k1 * N + n] = q1;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2) Pack every 4 consecutive K-values into RPP Q4 uint16_t weights
    // -------------------------------------------------------------------------
    //
    // Matching your AWQ -> RPP packing:
    //   pack = (int4_0 & 0x000F)
    //        | ((int4_1 << 4) & 0x00F0)
    //        | ((int4_2 << 8) & 0x0F00)
    //        | ((int4_3 << 12) & 0xF000);
    //
    out.weights_q4.resize((size_t) Ck * (size_t) N);

    for (int i = 0; i < Ck; ++i) {
        int k0 = 4 * i + 0;
        int k1 = 4 * i + 1;
        int k2 = 4 * i + 2;
        int k3 = 4 * i + 3;

        for (int j = 0; j < N; ++j) {
            uint16_t int4_0 = (k0 < K) ? (tmp_int4[(size_t) k0 * N + j] & 0x0F) : 0;
            uint16_t int4_1 = (k1 < K) ? (tmp_int4[(size_t) k1 * N + j] & 0x0F) : 0;
            uint16_t int4_2 = (k2 < K) ? (tmp_int4[(size_t) k2 * N + j] & 0x0F) : 0;
            uint16_t int4_3 = (k3 < K) ? (tmp_int4[(size_t) k3 * N + j] & 0x0F) : 0;

            uint16_t pack = (int4_0) | (uint16_t(int4_1) << 4) | (uint16_t(int4_2) << 8) | (uint16_t(int4_3) << 12);

            out.weights_q4[(size_t) i * N + j] = pack;
        }
    }

    // -------------------------------------------------------------------------
    // 3) Build RPP Q4 scales from gguf Q4_1 "d" (scale) per 32-element block
    // -------------------------------------------------------------------------
    //
    // We treat each 32-element block along K as a "group":
    //   G = K / 32
    // For each (g, n), we read src[n * blocks_per_row + g].d (fp16),
    // convert to float, then to bf16.
    //
    const int G = blocks_per_row;
    out.scales_bf16.resize((size_t) G * (size_t) N);
    for (int n = 0; n < N; ++n) {
        const block_q4_1_dbg * row_blocks = src + n * blocks_per_row;
        for (int g = 0; g < G; ++g) {
            float d_f32                         = half_to_float(row_blocks[g].d);
            // printf("scale row=%d, g=%d val=%f, \n", n, g, d_f32);
            out.scales_bf16[(size_t) g * N + n] = float_to_bf16(d_f32);
        }
    }

    // -------------------------------------------------------------------------
    // 4) Build RPP Q4 zeros (int8) – simple version: set to 0
    // -------------------------------------------------------------------------
    //
    // Q4_1 also has "m" (min/bias), but to keep things simple (as requested),
    // we ignore it here and use zero-point = 0 for every group.
    // If you later want to reflect 'm', you can approximate:
    //   zeros[g * N + n] = round(-m / d)
    // or similar, depending on the exact dequantization formula.
    //
    // std::vector<int8_t> zeros_i8_tmp;
    out.zeros_bf16.resize((size_t) G * (size_t) N);
    for (int n = 0; n < N; ++n) {
        const block_q4_1_dbg * row_blocks = src + n * blocks_per_row;
        for (int g = 0; g < G; ++g) {
            float m_f32                        = half_to_float(row_blocks[g].m);
            // printf("zero row=%d, g=%d val=%f, \n", n, g, m_f32);
            out.zeros_bf16[(size_t) g * N + n] = float_to_bf16(m_f32);
            // out.zeros_i8[g * N + n] = m_f32;
            // printf("zero row=%d, g=%d lo=%f\n", n, g, m_f32);
        }
    }

    // -------------------------------------------------------------------------
    // 5) Build LUT (16 entries 0..15) in bfloat16
    // -------------------------------------------------------------------------
    out.lut_bf16.resize(16);
    for (int i = 0; i < 16; ++i) {
        out.lut_bf16[i] = float_to_bf16((float) i);
    }
}

static void convert_q4_1_to_rpp(const block_q4_1_dbg * src,  // length N * (K/32)
                                int                    K,
                                int                    N,
                                void *                 out) {
    assert(K % 32 == 0);
    assert(K % 4 == 0);

    if (out == nullptr) {
        std::fprintf(stderr, "convert_q4_1_to_rpp requires out != nullptr\n");
        std::abort();
    }
    if (K == 0 || N == 0) {
        return;
    }

    const int blocks_per_row = K / 32;  // # of Q4_1 blocks per row
    const int Ck             = K / 4;   // # of uint16 packs along K
    const int G              = blocks_per_row;

    uint8_t * base = static_cast<uint8_t *>(out);

    uint16_t *     weights_q4  = reinterpret_cast<uint16_t *>(base);
    bfloat16_u16 * scales_bf16 = reinterpret_cast<bfloat16_u16 *>(weights_q4 + (size_t) Ck * (size_t) N);
    bfloat16_u16 * zeros_bf16  = reinterpret_cast<bfloat16_u16 *>(scales_bf16 + (size_t) G * (size_t) N);
    // bfloat16_u16 * lut_bf16    = reinterpret_cast<bfloat16_u16 *>(zeros_bf16 + (size_t) G * (size_t) N);

    // -------------------------------------------------------------------------
    // 1) Unpack ggml q4_1 into tmp_int4[k * N + n] = q(0..15)
    // -------------------------------------------------------------------------
    std::vector<uint8_t> tmp_int4((size_t) K * (size_t) N);
    for (int n = 0; n < N; ++n) {
        const block_q4_1_dbg * row_blocks = src + (size_t) n * (size_t) blocks_per_row;

        for (int b = 0; b < blocks_per_row; ++b) {
            const block_q4_1_dbg & blk = row_blocks[b];

            // ggml packing: qs[i] holds q[i] in low nibble and q[i+16] in high nibble
            for (int i = 0; i < 16; ++i) {
                uint8_t v  = blk.qs[i];
                uint8_t q0 = v & 0x0F;         // q[i]
                uint8_t q1 = (v >> 4) & 0x0F;  // q[i+16]

                int k0 = b * 32 + i;           // 0..15 within this block
                int k1 = b * 32 + i + 16;      // 16..31

                tmp_int4[(size_t) k0 * (size_t) N + (size_t) n] = q0;
                tmp_int4[(size_t) k1 * (size_t) N + (size_t) n] = q1;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2) Pack every 4 consecutive K-values into RPP Q4 uint16_t weights
    // -------------------------------------------------------------------------
    //
    // Matching your AWQ -> RPP packing:
    //   pack = (int4_0 & 0x000F)
    //        | ((int4_1 << 4) & 0x00F0)
    //        | ((int4_2 << 8) & 0x0F00)
    //        | ((int4_3 << 12) & 0xF000);
    //
    for (int i = 0; i < Ck; ++i) {
        int k0 = 4 * i + 0;
        int k1 = 4 * i + 1;
        int k2 = 4 * i + 2;
        int k3 = 4 * i + 3;

        for (int j = 0; j < N; ++j) {
            uint16_t int4_0 = (k0 < K) ? (tmp_int4[(size_t) k0 * (size_t) N + (size_t) j] & 0x0F) : 0;
            uint16_t int4_1 = (k1 < K) ? (tmp_int4[(size_t) k1 * (size_t) N + (size_t) j] & 0x0F) : 0;
            uint16_t int4_2 = (k2 < K) ? (tmp_int4[(size_t) k2 * (size_t) N + (size_t) j] & 0x0F) : 0;
            uint16_t int4_3 = (k3 < K) ? (tmp_int4[(size_t) k3 * (size_t) N + (size_t) j] & 0x0F) : 0;

            uint16_t pack =
                (uint16_t) (int4_0) | (uint16_t) (int4_1 << 4) | (uint16_t) (int4_2 << 8) | (uint16_t) (int4_3 << 12);

            weights_q4[(size_t) i * (size_t) N + (size_t) j] = pack;
        }
    }

    // -------------------------------------------------------------------------
    // 3) Build RPP Q4 scales from gguf Q4_1 "d" (scale) per 32-element block
    // -------------------------------------------------------------------------
    //
    // We treat each 32-element block along K as a "group":
    //   G = K / 32
    // For each (g, n), we read src[n * blocks_per_row + g].d (fp16),
    // convert to float, then to bf16.
    //
    for (int n = 0; n < N; ++n) {
        const block_q4_1_dbg * row_blocks = src + (size_t) n * (size_t) blocks_per_row;
        for (int g = 0; g < G; ++g) {
            float d_f32                                       = half_to_float(row_blocks[g].d);
            scales_bf16[(size_t) g * (size_t) N + (size_t) n] = float_to_bf16(d_f32);
        }
    }

    // -------------------------------------------------------------------------
    // 4) Build RPP Q4 zeros (int8) – simple version: set to 0
    // -------------------------------------------------------------------------
    //
    // Q4_1 also has "m" (min/bias), but to keep things simple (as requested),
    // we ignore it here and use zero-point = 0 for every group.
    // If you later want to reflect 'm', you can approximate:
    //   zeros[g * N + n] = round(-m / d)
    // or similar, depending on the exact dequantization formula.
    //
    for (int n = 0; n < N; ++n) {
        const block_q4_1_dbg * row_blocks = src + (size_t) n * (size_t) blocks_per_row;
        for (int g = 0; g < G; ++g) {
            float m_f32                                      = half_to_float(row_blocks[g].m);
            zeros_bf16[(size_t) g * (size_t) N + (size_t) n] = float_to_bf16(m_f32);
        }
    }

    // -------------------------------------------------------------------------
    // 5) LUT (16 entries 0..15) in bfloat16
    // -------------------------------------------------------------------------
    // for (int i = 0; i < 16; ++i) {
    //     lut_bf16[i] = float_to_bf16((float) i);
    // }
}

static void convert_q8_0_to_rpp(const block_q8_0_dbg * q8_blocks, int K, int N, RppQ8Params & out) {
    if (K % 32 != 0) {
        std::fprintf(stderr, "convert_q8_0_to_rpp requires K %% 32 == 0\n");
        std::abort();
    }
    if (K % 2 != 0) {
        std::fprintf(stderr, "convert_q8_0_to_rpp requires K %% 2 == 0\n");
        std::abort();
    }

    const int ng = K / 32;
    out.weights_q8.assign((size_t) (K / 2) * (size_t) N, 0);
    out.scales_bf16.assign((size_t) ng * (size_t) N, bfloat16_u16{ 0 });
    out.zeros_bf16.assign((size_t) ng * (size_t) N, bfloat16_u16{ 0 });

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_q8_0_dbg & blk = q8_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            float                  d   = fp16_to_float(blk.d_fp16);
            out.scales_bf16[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(d);
            out.zeros_bf16[idx2((size_t) g, (size_t) n, (size_t) N)]  = bfloat16_u16{ 0 };

            for (int i = 0; i < 32; ++i) {
                int          k        = g * 32 + i;
                const int8_t q        = blk.qs[i];
                size_t       pack_row = (size_t) (k / 2);
                size_t       pack_idx = idx2(pack_row, (size_t) n, (size_t) N);
                uint16_t &   dst      = out.weights_q8[pack_idx];
                if ((k & 1) == 0) {
                    // low byte
                    dst = (dst & 0xFF00u) | (uint8_t) q;
                } else {
                    // high byte
                    dst = (dst & 0x00FFu) | ((uint16_t) (uint8_t) q << 8);
                }
            }
        }
    }
}

static void convert_q8_0_to_rpp(const block_q8_0_dbg * q8_blocks, int K, int N, void * out) {
    if (K % 32 != 0) {
        std::fprintf(stderr, "K %% 32 == 0 required\n");
        std::abort();
    }
    if (K % 2 != 0) {
        std::fprintf(stderr, "K %% 2 == 0 required\n");
        std::abort();
    }
    if (!out) {
        std::fprintf(stderr, "out != nullptr required\n");
        std::abort();
    }

    const int    ng          = K / 32;
    const size_t weights_cnt = (size_t) (K / 2) * (size_t) N;
    const size_t scales_cnt  = (size_t) ng * (size_t) N;

    uint8_t * base    = static_cast<uint8_t *>(out);
    auto *    weights = reinterpret_cast<uint16_t *>(base);
    auto *    scales  = reinterpret_cast<bfloat16_u16 *>(weights + weights_cnt);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_q8_0_dbg & blk                       = q8_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            float                  d                         = fp16_to_float(blk.d_fp16);
            scales[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(d);
            for (int i = 0; i < 32; ++i) {
                int          k = g * 32 + i;
                const int8_t q = blk.qs[i];

                size_t pack_row = (size_t) (k / 2);
                size_t pack_idx = idx2(pack_row, (size_t) n, (size_t) N);

                uint16_t & dst = weights[pack_idx];
                if ((k & 1) == 0) {
                    dst = (dst & 0xFF00u) | (uint8_t) q;
                } else {
                    dst = (dst & 0x00FFu) | ((uint16_t) (uint8_t) q << 8);
                }
            }
        }
    }
}

static const int K_SCALE_SIZE_Q4K_DBG = 12;

#pragma pack(push, 1)

struct block_q4_k_dbg {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE_Q4K_DBG];
    uint8_t  qs[QK_K / 2];
};

#pragma pack(pop)

static_assert(sizeof(block_q4_k_dbg) == 2 * sizeof(uint16_t) + K_SCALE_SIZE_Q4K_DBG + QK_K / 2,
              "wrong q4_k block size/padding");

static inline uint16_t pack_q4k_u6_lsb_word(const uint8_t v[8], int word_id) {
    const int base = word_id * 4;
    return (uint16_t) (((uint16_t) (v[base + 0] & 0x0F) << 0) | ((uint16_t) (v[base + 1] & 0x0F) << 4) |
                       ((uint16_t) (v[base + 2] & 0x0F) << 8) | ((uint16_t) (v[base + 3] & 0x0F) << 12));
}

static inline uint16_t pack_q4k_u6_msb_word(const uint8_t v[8]) {
    uint16_t out = 0;
    for (int i = 0; i < 8; ++i) {
        out |= (uint16_t) (((v[i] >> 4) & 0x03) << (2 * i));
    }
    return out;
}

static inline void get_scale_min_k4_local(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static inline void decode_q4k_block_q(const block_q4_k_dbg & blk, uint8_t * q_out) {
    const uint8_t * q = blk.qs;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; ++l) {
            q_out[j + l]      = (uint8_t) (q[l] & 0xF);
            q_out[j + l + 32] = (uint8_t) (q[l] >> 4);
        }
        q += 32;
    }
}

static void convert_q4_k_to_rpp(const block_q4_k_dbg * q4_blocks, int K, int N, void * out) {
    if (K % QK_K != 0 || K % 32 != 0 || K % 4 != 0) {
        std::fprintf(stderr, "convert_q4_k_to_rpp requires K %% 256 == 0, K %% 32 == 0 and K %% 4 == 0\n");
        std::abort();
    }
    if (!out) {
        std::fprintf(stderr, "out != nullptr required\n");
        std::abort();
    }

    const int    ng               = K / QK_K;
    const size_t weights_q_size   = (size_t) K / 4 * (size_t) N;
    const size_t scale_lsb_size   = (size_t) 2 * (size_t) ng * (size_t) N;
    const size_t zero_lsb_size    = scale_lsb_size;
    const size_t scale_msb_size   = (size_t) ng * (size_t) N;
    const size_t zero_msb_size    = scale_msb_size;
    const size_t super_scale_size = (size_t) ng * (size_t) N;
    const size_t super_zero_size  = super_scale_size;

    uint8_t * base        = static_cast<uint8_t *>(out);
    auto *    weights_q   = reinterpret_cast<uint16_t *>(base);
    auto *    scale_lsb   = reinterpret_cast<uint16_t *>(weights_q + weights_q_size);
    auto *    zero_lsb    = reinterpret_cast<uint16_t *>(scale_lsb + scale_lsb_size);
    auto *    scale_msb   = reinterpret_cast<uint16_t *>(zero_lsb + zero_lsb_size);
    auto *    zero_msb    = reinterpret_cast<uint16_t *>(scale_msb + scale_msb_size);
    auto *    super_scale = reinterpret_cast<bfloat16_u16 *>(zero_msb + zero_msb_size);
    auto *    super_zero  = reinterpret_cast<bfloat16_u16 *>(super_scale + super_scale_size);

    const size_t total_bytes =
        (weights_q_size + scale_lsb_size + zero_lsb_size + scale_msb_size + zero_msb_size) * sizeof(uint16_t) +
        (super_scale_size + super_zero_size) * sizeof(bfloat16_u16);
    // std::memset(out, 0, total_bytes);

    std::vector<uint8_t> qvals(QK_K, 0);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_q4_k_dbg & blk = q4_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];

            super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));
            super_zero[idx2((size_t) g, (size_t) n, (size_t) N)]  = float_to_bf16_rne(fp16_to_float(blk.dmin));

            uint8_t sc_u6[8] = { 0 };
            uint8_t mn_u6[8] = { 0 };
            for (int ib = 0; ib < 8; ++ib) {
                get_scale_min_k4_local(ib, blk.scales, &sc_u6[ib], &mn_u6[ib]);
            }

            scale_lsb[idx2((size_t) (2 * g + 0), (size_t) n, (size_t) N)] = pack_q4k_u6_lsb_word(sc_u6, 0);
            scale_lsb[idx2((size_t) (2 * g + 1), (size_t) n, (size_t) N)] = pack_q4k_u6_lsb_word(sc_u6, 1);
            zero_lsb[idx2((size_t) (2 * g + 0), (size_t) n, (size_t) N)]  = pack_q4k_u6_lsb_word(mn_u6, 0);
            zero_lsb[idx2((size_t) (2 * g + 1), (size_t) n, (size_t) N)]  = pack_q4k_u6_lsb_word(mn_u6, 1);
            scale_msb[idx2((size_t) g, (size_t) n, (size_t) N)]           = pack_q4k_u6_msb_word(sc_u6);
            zero_msb[idx2((size_t) g, (size_t) n, (size_t) N)]            = pack_q4k_u6_msb_word(mn_u6);

            decode_q4k_block_q(blk, qvals.data());
            for (int i = 0; i < QK_K; ++i) {
                const int     k     = g * QK_K + i;
                const uint8_t q     = qvals[i] & 0x0F;
                const size_t  q_row = (size_t) (k / 4);
                const size_t  q_idx = idx2(q_row, (size_t) n, (size_t) N);
                uint16_t &    word  = weights_q[q_idx];
                const int     shift = (k & 3) * 4;
                word                = (uint16_t) ((word & ~(0xFu << shift)) | ((uint16_t) q << shift));
            }
        }
    }
}

static const int K_SCALE_SIZE_Q5K_DBG = 12;

#pragma pack(push, 1)

struct block_q5_k_dbg {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE_Q5K_DBG];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
};

#pragma pack(pop)

static_assert(sizeof(block_q5_k_dbg) == 2 * sizeof(uint16_t) + K_SCALE_SIZE_Q5K_DBG + QK_K / 8 + QK_K / 2,
              "wrong q5_k block size/padding");

static inline void decode_q5k_block_q(const block_q5_k_dbg & blk, uint8_t * q_out) {
    const uint8_t * ql = blk.qs;
    const uint8_t * qh = blk.qh;

    uint8_t m1 = 1;
    uint8_t m2 = 2;
    for (int n = 0; n < QK_K; n += 64) {
        for (int j = 0; j < 32; ++j) {
            const uint8_t lo0 = (uint8_t) (ql[j] & 0x0F);
            const uint8_t lo1 = (uint8_t) (ql[j] >> 4);
            const uint8_t hi0 = (uint8_t) ((qh[j] & m1) ? 1 : 0);
            const uint8_t hi1 = (uint8_t) ((qh[j] & m2) ? 1 : 0);
            q_out[n + j]      = (uint8_t) (lo0 | (hi0 << 4));
            q_out[n + j + 32] = (uint8_t) (lo1 | (hi1 << 4));
        }
        ql += 32;
        m1 <<= 2;
        m2 <<= 2;
    }
}

static void convert_q5_k_to_rpp(const block_q5_k_dbg * q5_blocks, int K, int N, void * out) {
    if (K % QK_K != 0 || K % 32 != 0 || K % 16 != 0 || K % 4 != 0) {
        std::fprintf(stderr, "convert_q5_k_to_rpp requires K %% 256 == 0, K %% 32 == 0, K %% 16 == 0, K %% 4 == 0\n");
        std::abort();
    }
    if (!out) {
        std::fprintf(stderr, "out != nullptr required\n");
        std::abort();
    }

    const int    ng               = K / QK_K;
    const size_t weights_lsb_size = (size_t) K / 4 * (size_t) N;
    const size_t weights_msb_size = (size_t) K / 16 * (size_t) N;
    const size_t scale_lsb_size   = (size_t) 2 * (size_t) ng * (size_t) N;
    const size_t zero_lsb_size    = scale_lsb_size;
    const size_t scale_msb_size   = (size_t) ng * (size_t) N;
    const size_t zero_msb_size    = scale_msb_size;
    const size_t super_scale_size = (size_t) ng * (size_t) N;
    const size_t super_zero_size  = super_scale_size;

    uint8_t * base        = static_cast<uint8_t *>(out);
    auto *    weights_lsb = reinterpret_cast<uint16_t *>(base);
    auto *    weights_msb = reinterpret_cast<uint16_t *>(weights_lsb + weights_lsb_size);
    auto *    scale_lsb   = reinterpret_cast<uint16_t *>(weights_msb + weights_msb_size);
    auto *    zero_lsb    = reinterpret_cast<uint16_t *>(scale_lsb + scale_lsb_size);
    auto *    scale_msb   = reinterpret_cast<uint16_t *>(zero_lsb + zero_lsb_size);
    auto *    zero_msb    = reinterpret_cast<uint16_t *>(scale_msb + scale_msb_size);
    auto *    super_scale = reinterpret_cast<bfloat16_u16 *>(zero_msb + zero_msb_size);
    auto *    super_zero  = reinterpret_cast<bfloat16_u16 *>(super_scale + super_scale_size);

    const size_t total_bytes =
        (weights_lsb_size + weights_msb_size + scale_lsb_size + zero_lsb_size + scale_msb_size + zero_msb_size) *
            sizeof(uint16_t) +
        (super_scale_size + super_zero_size) * sizeof(bfloat16_u16);
    // std::memset(out, 0, total_bytes);

    std::vector<uint8_t> qvals(QK_K, 0);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_q5_k_dbg & blk = q5_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];

            super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));
            super_zero[idx2((size_t) g, (size_t) n, (size_t) N)]  = float_to_bf16_rne(fp16_to_float(blk.dmin));

            uint8_t sc_u6[8] = { 0 };
            uint8_t mn_u6[8] = { 0 };
            for (int ib = 0; ib < 8; ++ib) {
                get_scale_min_k4_local(ib, blk.scales, &sc_u6[ib], &mn_u6[ib]);
            }
            scale_lsb[idx2((size_t) (2 * g + 0), (size_t) n, (size_t) N)] = pack_q4k_u6_lsb_word(sc_u6, 0);
            scale_lsb[idx2((size_t) (2 * g + 1), (size_t) n, (size_t) N)] = pack_q4k_u6_lsb_word(sc_u6, 1);
            zero_lsb[idx2((size_t) (2 * g + 0), (size_t) n, (size_t) N)]  = pack_q4k_u6_lsb_word(mn_u6, 0);
            zero_lsb[idx2((size_t) (2 * g + 1), (size_t) n, (size_t) N)]  = pack_q4k_u6_lsb_word(mn_u6, 1);
            scale_msb[idx2((size_t) g, (size_t) n, (size_t) N)]           = pack_q4k_u6_msb_word(sc_u6);
            zero_msb[idx2((size_t) g, (size_t) n, (size_t) N)]            = pack_q4k_u6_msb_word(mn_u6);

            decode_q5k_block_q(blk, qvals.data());
            for (int i = 0; i < QK_K; ++i) {
                const int     k  = g * QK_K + i;
                const uint8_t q  = (uint8_t) (qvals[i] & 0x1F);
                const uint8_t lo = (uint8_t) (q & 0x0F);
                const uint8_t hi = (uint8_t) ((q >> 4) & 0x01);

                {
                    const size_t row   = (size_t) (k / 4);
                    const size_t idx   = idx2(row, (size_t) n, (size_t) N);
                    uint16_t &   word  = weights_lsb[idx];
                    const int    shift = (k & 3) * 4;
                    word               = (uint16_t) ((word & ~(0xFu << shift)) | ((uint16_t) lo << shift));
                }

                {
                    const size_t row   = (size_t) (k / 16);
                    const size_t idx   = idx2(row, (size_t) n, (size_t) N);
                    uint16_t &   word  = weights_msb[idx];
                    const int    shift = (k & 15);
                    word               = (uint16_t) ((word & ~(1u << shift)) | ((uint16_t) hi << shift));
                }
            }
        }
    }
}

#pragma pack(push, 1)

struct block_iq3_xxs_dbg {
    uint16_t d;
    uint8_t  qs[3 * QK_K / 8];
};

#pragma pack(pop)

static_assert(sizeof(block_iq3_xxs_dbg) == sizeof(uint16_t) + 3 * (QK_K / 8), "wrong iq3_xxs block size/padding");

static inline uint8_t reverse_bits_u8(uint8_t x) {
    x = (uint8_t) (((x & 0xF0u) >> 4) | ((x & 0x0Fu) << 4));
    x = (uint8_t) (((x & 0xCCu) >> 2) | ((x & 0x33u) << 2));
    x = (uint8_t) (((x & 0xAAu) >> 1) | ((x & 0x55u) << 1));
    return x;
}

static inline void rpp_store_packed_u8_u16(uint16_t * dst, size_t byte_row, size_t n, size_t N, uint8_t value) {
    uint16_t &     word  = dst[idx2(byte_row >> 1, n, N)];
    const uint16_t shift = (uint16_t) (8 * (byte_row & 1));
    word                 = (uint16_t) ((word & ~(0xFFu << shift)) | ((uint16_t) value << shift));
}

static inline uint8_t rpp_load_packed_u8_u16(const uint16_t * src, size_t byte_row, size_t n, size_t N) {
    const uint16_t word = src[idx2(byte_row >> 1, n, N)];
    return (uint8_t) ((word >> (8 * (byte_row & 1))) & 0xFFu);
}

struct RppQ3xxsParams {
    // shape [K/8][N], each word stores 2x 8-bit codebook indices:
    // low byte = even index, high byte = odd index
    std::vector<uint16_t>     codebook;
    // shape [K/128][N], each word packs 4x 4-bit scales, each scale covers 32 K elements
    std::vector<uint16_t>     scales;
    // shape [K/16][N], each word packs 16 sign bits.
    // Bit order is reversed per 16 weights: weight 0 -> bit15, weight 15 -> bit0.
    std::vector<uint16_t>     sign;
    // shape [K/256][N], bf16 super scale
    std::vector<bfloat16_u16> super_scale;
};

static void convert_iq3_xxs_to_rpp(const block_iq3_xxs_dbg * q3_blocks, int K, int N, RppQ3xxsParams & out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq3xxs_to_rpp requires K %% 256 == 0\n");
        std::abort();
    }

    const int ng = K / QK_K;
    // if ((int) q3_blocks.size() != ng * N) {
    //     std::fprintf(stderr, "convert_iq3xxs_to_rpp input block count mismatch\n");
    //     std::abort();
    // }
    out.codebook.assign((size_t) (K / 8) * (size_t) N, 0);
    out.scales.assign((size_t) (K / 128) * (size_t) N, 0);
    out.sign.assign((size_t) (K / 16) * (size_t) N, 0);
    out.super_scale.assign((size_t) ng * (size_t) N, bfloat16_u16{ 0 });

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq3_xxs_dbg & blk = q3_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            out.super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int i = 0; i < QK_K / 4; i += 2) {
                const uint16_t packed = (uint16_t) blk.qs[i] | ((uint16_t) blk.qs[i + 1] << 8);
                const size_t   row    = (size_t) (g * (QK_K / 8) + i / 2);
                out.codebook[idx2(row, (size_t) n, (size_t) N)] = packed;
            }

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                uint32_t aux = 0;
                std::memcpy(&aux, blk.qs + QK_K / 4 + 4 * ib32, sizeof(uint32_t));

                const uint8_t scale4     = (uint8_t) ((aux >> 28) & 0xF);
                const size_t  scale_row  = (size_t) (g * (QK_K / 128) + ib32 / 4);
                uint16_t &    scale_word = out.scales[idx2(scale_row, (size_t) n, (size_t) N)];
                scale_word               = (uint16_t) (scale_word | ((uint16_t) scale4 << (4 * (ib32 & 3))));

                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs8 = ksigns_iq2xs_local[(aux >> (7 * l)) & 127];
                    for (int b = 0; b < 8; ++b) {
                        if ((signs8 & (1u << b)) == 0) {
                            continue;
                        }
                        const int    local       = ib32 * 32 + l * 8 + b;
                        const size_t sign_row    = (size_t) (g * (QK_K / 16) + local / 16);
                        uint16_t &   sign_word   = out.sign[idx2(sign_row, (size_t) n, (size_t) N)];
                        const int    bit_in_word = 15 - (local & 15);
                        sign_word                = (uint16_t) (sign_word | (1u << bit_in_word));
                    }
                }
            }
        }
    }
}

static void convert_iq3_xxs_to_rpp(const block_iq3_xxs_dbg * q3_blocks, int K, int N, void * out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq3_xxs_to_rpp requires K %% 256 == 0\n");
        std::abort();
    }
    if (!out) {
        std::fprintf(stderr, "out != nullptr required\n");
        std::abort();
    }

    const int    ng             = K / QK_K;
    const size_t codebook_size  = (size_t) (K / 8) * (size_t) N;
    const size_t scales_size    = (size_t) (K / 128) * (size_t) N;
    const size_t sign_size      = (size_t) (K / 16) * (size_t) N;
    const size_t super_scale_sz = (size_t) ng * (size_t) N;

    uint8_t * base        = static_cast<uint8_t *>(out);
    auto *    codebook    = reinterpret_cast<uint16_t *>(base);
    auto *    scales      = reinterpret_cast<uint16_t *>(codebook + codebook_size);
    auto *    sign        = reinterpret_cast<uint16_t *>(scales + scales_size);
    auto *    super_scale = reinterpret_cast<bfloat16_u16 *>(sign + sign_size);

    const size_t total_bytes =
        (codebook_size + scales_size + sign_size) * sizeof(uint16_t) + super_scale_sz * sizeof(bfloat16_u16);
    // std::memset(out, 0, total_bytes);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq3_xxs_dbg & blk = q3_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int i = 0; i < QK_K / 4; i += 2) {
                const uint16_t packed                       = (uint16_t) blk.qs[i] | ((uint16_t) blk.qs[i + 1] << 8);
                const size_t   row                          = (size_t) (g * (QK_K / 8) + i / 2);
                codebook[idx2(row, (size_t) n, (size_t) N)] = packed;
            }

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                uint32_t aux = 0;
                std::memcpy(&aux, blk.qs + QK_K / 4 + 4 * ib32, sizeof(uint32_t));

                const uint8_t scale4     = (uint8_t) ((aux >> 28) & 0xF);
                const size_t  scale_row  = (size_t) (g * (QK_K / 128) + ib32 / 4);
                uint16_t &    scale_word = scales[idx2(scale_row, (size_t) n, (size_t) N)];
                scale_word               = (uint16_t) (scale_word | ((uint16_t) scale4 << (4 * (ib32 & 3))));

                for (int l = 0; l < 4; ++l) {
                    const uint8_t signs8 = ksigns_iq2xs_local[(aux >> (7 * l)) & 127];
                    for (int b = 0; b < 8; ++b) {
                        if ((signs8 & (1u << b)) == 0) {
                            continue;
                        }
                        const int    local       = ib32 * 32 + l * 8 + b;
                        const size_t sign_row    = (size_t) (g * (QK_K / 16) + local / 16);
                        uint16_t &   sign_word   = sign[idx2(sign_row, (size_t) n, (size_t) N)];
                        const int    bit_in_word = 15 - (local & 15);
                        sign_word                = (uint16_t) (sign_word | (1u << bit_in_word));
                    }
                }
            }
        }
    }
}

struct RppQ3xxsNoLutParams {
    // shape [K/16 * 3][N], packed into uint16 words.
    // Each 16-weight chunk becomes 48 bits = 3 uint16 words.
    // Code -> magnitude:
    //   0:0x04 1:0x0c 2:0x14 3:0x1c 4:0x24 5:0x2c 6:0x34 7:0x3e
    // For K=256, N=1 this is 16 chunks * 3 words * 2 bytes = 96 bytes.
    std::vector<uint16_t>     codebook_nolut;
    // same layout as q3xxs LUT path
    std::vector<uint16_t>     scales;
    std::vector<uint16_t>     sign;
    std::vector<bfloat16_u16> super_scale;
    // Total payload for K=256, N=1:
    //   codebook_nolut  96 bytes
    //   scales           4 bytes
    //   sign            32 bytes
    //   super_scale      2 bytes
    //   total          134 bytes
};

static inline uint8_t q3xxs_mag_to_nolut_code(uint8_t mag) {
    switch (mag) {
        case 0x04:
            return 0;
        case 0x0c:
            return 1;
        case 0x14:
            return 2;
        case 0x1c:
            return 3;
        case 0x24:
            return 4;
        case 0x2c:
            return 5;
        case 0x34:
            return 6;
        case 0x3e:
            return 7;
        default:
            std::fprintf(stderr, "q3xxs_mag_to_nolut_code: unsupported mag=0x%02X\n", (unsigned) mag);
            std::abort();
    }
}

static void convert_iq3_xxs_to_nolut(const block_iq3_xxs_dbg * q3_blocks, int K, int N, RppQ3xxsNoLutParams & out) {
    RppQ3xxsParams q3_rpp;
    convert_iq3_xxs_to_rpp(q3_blocks, K, N, q3_rpp);

    out.codebook_nolut.assign((size_t) (K / 16) * 3u * (size_t) N, 0);
    out.scales      = q3_rpp.scales;
    out.sign        = q3_rpp.sign;
    out.super_scale = q3_rpp.super_scale;

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < K / QK_K; ++g) {
            for (int chunk16 = 0; chunk16 < QK_K / 16; ++chunk16) {
                uint64_t packed = 0;
                for (int u16 = 0; u16 < 16; ++u16) {
                    const int local = chunk16 * 16 + u16;
                    const int ib32  = local / 32;
                    const int r     = local & 31;
                    const int l     = r / 8;
                    const int u     = r & 7;

                    const int      cb_idx  = ib32 * 8 + 2 * l + (u >= 4 ? 1 : 0);
                    const size_t   cb_row  = (size_t) (g * (QK_K / 8) + cb_idx / 2);
                    const uint16_t cb_word = q3_rpp.codebook[idx2(cb_row, (size_t) n, (size_t) N)];
                    const uint8_t  cb      = (uint8_t) ((cb_word >> (8 * (cb_idx & 1))) & 0xFFu);
                    const uint32_t grid    = iq3xxs_grid_local[cb];
                    const uint8_t  mag     = (uint8_t) ((grid >> (8 * (u & 3))) & 0xFFu);
                    const uint8_t  code    = q3xxs_mag_to_nolut_code(mag);
                    packed |= (uint64_t) code << (3 * u16);
                }

                const size_t row_base = (size_t) (g * (QK_K / 16 * 3) + chunk16 * 3);
                out.codebook_nolut[idx2(row_base + 0, (size_t) n, (size_t) N)] = (uint16_t) (packed & 0xFFFFu);
                out.codebook_nolut[idx2(row_base + 1, (size_t) n, (size_t) N)] = (uint16_t) ((packed >> 16) & 0xFFFFu);
                out.codebook_nolut[idx2(row_base + 2, (size_t) n, (size_t) N)] = (uint16_t) ((packed >> 32) & 0xFFFFu);
            }
        }
    }
}

static void convert_iq3_xxs_to_nolut(const block_iq3_xxs_dbg * q3_blocks, int K, int N, void * out) {
    if (K % QK_K != 0 || out == nullptr) {
        std::fprintf(stderr, "convert_iq3xxs_to_nolut: K %% %d == 0 and out != nullptr required\n", QK_K);
        std::abort();
    }
    const int    ng                   = K / QK_K;
    const size_t codebook_nolut_count = (size_t) (K / 16) * 3u * (size_t) N;
    const size_t scales_count         = (size_t) (K / 128) * (size_t) N;
    const size_t sign_count           = (size_t) (K / 16) * (size_t) N;
    const size_t super_scale_count    = (size_t) ng * (size_t) N;

    // LUT layout sizes (same as convert_iq3_xxs_to_rpp)
    const size_t lut_codebook_size = (size_t) (K / 8) * (size_t) N;
    const size_t lut_total_bytes =
        (lut_codebook_size + scales_count + sign_count) * sizeof(uint16_t) + super_scale_count * sizeof(bfloat16_u16);

    uint16_t *     p_codebook_nolut = static_cast<uint16_t *>(out);
    uint16_t *     p_scales         = p_codebook_nolut + codebook_nolut_count;
    uint16_t *     p_sign           = p_scales + scales_count;
    bfloat16_u16 * p_super_scale    = reinterpret_cast<bfloat16_u16 *>(p_sign + sign_count);

    // Use convert_iq3_xxs_to_rpp to fill LUT (codebook, scales, sign, super_scale) in a temp buffer
    std::vector<uint8_t> lut_buf(lut_total_bytes, 0);
    convert_iq3_xxs_to_rpp(q3_blocks, K, N, lut_buf.data());

    const uint16_t *     lut_codebook    = reinterpret_cast<const uint16_t *>(lut_buf.data());
    const uint16_t *     lut_scales      = lut_codebook + lut_codebook_size;
    const uint16_t *     lut_sign        = lut_scales + scales_count;
    const bfloat16_u16 * lut_super_scale = reinterpret_cast<const bfloat16_u16 *>(lut_sign + sign_count);

    std::memcpy(p_scales, lut_scales, scales_count * sizeof(uint16_t));
    std::memcpy(p_sign, lut_sign, sign_count * sizeof(uint16_t));
    std::memcpy(p_super_scale, lut_super_scale, super_scale_count * sizeof(bfloat16_u16));

    // LUT codebook -> codebook_nolut (pack 16x 3-bit codes into 48 bits = 3 uint16)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < K / QK_K; ++g) {
            for (int chunk16 = 0; chunk16 < QK_K / 16; ++chunk16) {
                uint64_t packed = 0;
                for (int u16 = 0; u16 < 16; ++u16) {
                    const int      local   = chunk16 * 16 + u16;
                    const int      ib32    = local / 32;
                    const int      r       = local & 31;
                    const int      l       = r / 8;
                    const int      u       = r & 7;
                    const int      cb_idx  = ib32 * 8 + 2 * l + (u >= 4 ? 1 : 0);
                    const size_t   cb_row  = (size_t) (g * (QK_K / 8) + cb_idx / 2);
                    const uint16_t cb_word = lut_codebook[idx2(cb_row, (size_t) n, (size_t) N)];
                    const uint8_t  cb      = (uint8_t) ((cb_word >> (8 * (cb_idx & 1))) & 0xFFu);
                    const uint32_t grid    = iq3xxs_grid_local[cb];
                    const uint8_t  mag     = (uint8_t) ((grid >> (8 * (u & 3))) & 0xFFu);
                    const uint8_t  code    = q3xxs_mag_to_nolut_code(mag);
                    packed |= (uint64_t) code << (3 * u16);
                }
                const size_t row_base = (size_t) (g * (QK_K / 16 * 3) + chunk16 * 3);
                p_codebook_nolut[idx2(row_base + 0, (size_t) n, (size_t) N)] = (uint16_t) (packed & 0xFFFFu);
                p_codebook_nolut[idx2(row_base + 1, (size_t) n, (size_t) N)] = (uint16_t) ((packed >> 16) & 0xFFFFu);
                p_codebook_nolut[idx2(row_base + 2, (size_t) n, (size_t) N)] = (uint16_t) ((packed >> 32) & 0xFFFFu);
            }
        }
    }
}

#pragma pack(push, 1)

struct block_iq2_xs_dbg {
    uint16_t d;
    uint16_t qs[QK_K / 8];
    uint8_t  scales[QK_K / 32];
};

#pragma pack(pop)

static_assert(sizeof(block_iq2_xs_dbg) == sizeof(uint16_t) + (QK_K / 8) * sizeof(uint16_t) + (QK_K / 32),
              "wrong iq2_xs block size/padding");

static inline uint8_t rpp_load_packed_u8(const std::vector<uint16_t> & src, size_t byte_row, size_t n, size_t N) {
    const uint16_t word = src[idx2(byte_row >> 1, n, N)];
    return (uint8_t) ((word >> (8 * (byte_row & 1))) & 0xFF);
}

static inline void rpp_store_packed_u8(std::vector<uint16_t> & dst,
                                       size_t                  byte_row,
                                       size_t                  n,
                                       size_t                  N,
                                       uint8_t                 value) {
    uint16_t &     word  = dst[idx2(byte_row >> 1, n, N)];
    const uint16_t shift = (uint16_t) (8 * (byte_row & 1));
    word                 = (uint16_t) ((word & ~(0xFFu << shift)) | ((uint16_t) value << shift));
}

static inline void rpp_store_packed_u8(uint16_t * dst, size_t byte_row, size_t n, size_t N, uint8_t value) {
    uint16_t &     word  = dst[idx2(byte_row >> 1, n, N)];
    const uint16_t shift = (uint16_t) (8 * (byte_row & 1));
    word                 = (uint16_t) ((word & ~(0xFFu << shift)) | ((uint16_t) value << shift));
}

static inline uint8_t q2s_mag_to_nolut_code(uint8_t mag) {
    if (mag == 8) {
        return 0;
    }
    if (mag == 25) {
        return 1;
    }
    if (mag == 43) {
        return 2;
    }
    std::fprintf(stderr, "q2s_mag_to_nolut_code got unsupported magnitude=%u\n", (unsigned) mag);
    std::abort();
}

static inline uint8_t q2s_nolut_code_to_mag(uint8_t code) {
    if (code == 0) {
        return 8;
    }
    if (code == 1) {
        return 25;
    }
    if (code == 2) {
        return 43;
    }
    std::fprintf(stderr, "q2s_nolut_code_to_mag got unsupported code=%u\n", (unsigned) code);
    std::abort();
}

struct RppQ2xsParams {
    // shape [K/8][N], one uint16 per 8-value subgroup (same packed word as ggml block_iq2_xs::qs):
    // bits [8:0] = iq2xs codebook index, bits [15:9] = 7-bit sign code (expanded by ksigns_iq2xs).
    std::vector<uint16_t>     qs;
    // shape [K/64][N], each word stores 2x packed scale bytes (same layout as RppQ2sParams::scales)
    std::vector<uint16_t>     scales;
    // shape [K/256][N], bf16 super scale
    std::vector<bfloat16_u16> super_scale;
};

static void convert_iq2xs_to_rpp(const block_iq2_xs_dbg * q2xs_blocks, int K, int N, RppQ2xsParams & out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq2xs_to_rpp requires K %% 256 == 0\n");
        std::abort();
    }

    const int ng = K / QK_K;
    // if ((int)q2xs_blocks.size() != ng * N) {
    //     std::fprintf(stderr, "convert_iq2xs_to_rpp input block count mismatch\n");
    //     std::abort();
    // }

    out.qs.assign((size_t) (K / 8) * (size_t) N, 0);
    out.scales.assign((size_t) (K / 64) * (size_t) N, 0);
    out.super_scale.assign((size_t) ng * (size_t) N, bfloat16_u16{ 0 });

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq2_xs_dbg & blk = q2xs_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            out.super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                const size_t byte_row_scale = (size_t) (g * (QK_K / 32) + ib32);
                rpp_store_packed_u8(out.scales, byte_row_scale, (size_t) n, (size_t) N, blk.scales[ib32]);

                for (int l = 0; l < 4; ++l) {
                    const size_t cb_row                          = (size_t) (g * (QK_K / 8) + ib32 * 4 + l);
                    out.qs[idx2(cb_row, (size_t) n, (size_t) N)] = blk.qs[ib32 * 4 + l];
                }
            }
        }
    }
}

static void convert_iq2_xs_to_rpp(const block_iq2_xs_dbg * q2_blocks, int K, int N, void * out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq2_xs_to_rpp requires K %% 256 == 0\n");
        std::abort();
    }
    if (!out) {
        std::fprintf(stderr, "out != nullptr required\n");
        std::abort();
    }

    const int    ng               = K / QK_K;
    const size_t qs_size          = (size_t) (K / 8) * (size_t) N;
    const size_t scales_size      = (size_t) (K / 64) * (size_t) N;
    const size_t super_scale_size = (size_t) ng * (size_t) N;

    uint8_t * base        = static_cast<uint8_t *>(out);
    auto *    qs          = reinterpret_cast<uint16_t *>(base);
    auto *    scales      = reinterpret_cast<uint16_t *>(qs + qs_size);
    auto *    super_scale = reinterpret_cast<bfloat16_u16 *>(scales + scales_size);

    const size_t total_bytes = (qs_size + scales_size) * sizeof(uint16_t) + super_scale_size * sizeof(bfloat16_u16);
    // std::memset(out, 0, total_bytes);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq2_xs_dbg & blk = q2_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int i = 0; i < QK_K / 8; ++i) {
                const size_t row                      = (size_t) (g * (QK_K / 8) + i);
                qs[idx2(row, (size_t) n, (size_t) N)] = blk.qs[i];
            }

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                const size_t byte_row = (size_t) (g * (QK_K / 32) + ib32);
                rpp_store_packed_u8_u16(scales, byte_row, (size_t) n, (size_t) N, blk.scales[ib32]);
            }
        }
    }
}

struct RppQ2xsNoLutParams {
    // shape [K/8][N], one uint16 per 8 weights, each weight uses 2-bit code.
    // code: 0->8, 1->25, 2->43
    std::vector<uint16_t>     codebook_nolut;
    // same layout as q2xs LUT path
    std::vector<uint16_t>     scales;
    std::vector<uint16_t>     sign;
    std::vector<bfloat16_u16> super_scale;
};

static void convert_iq2_xs_to_nolut(const block_iq2_xs_dbg * q2xs_blocks, int K, int N, RppQ2xsNoLutParams & out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq2xs_to_nolut requires K %% 256 == 0\n");
        std::abort();
    }

    const int ng = K / QK_K;
    // if ((int)q2xs_blocks.size() != ng * N) {
    //     std::fprintf(stderr, "convert_iq2xs_to_nolut input block count mismatch\n");
    //     std::abort();
    // }

    out.codebook_nolut.assign((size_t) (K / 8) * (size_t) N, 0);
    out.scales.assign((size_t) (K / 64) * (size_t) N, 0);
    out.sign.assign((size_t) (K / 16) * (size_t) N, 0);
    out.super_scale.assign((size_t) ng * (size_t) N, bfloat16_u16{ 0 });

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq2_xs_dbg & blk = q2xs_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            out.super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                const size_t row_scale = (size_t) (g * (QK_K / 32) + ib32);
                rpp_store_packed_u8(out.scales, row_scale, (size_t) n, (size_t) N, blk.scales[ib32]);

                for (int l = 0; l < 4; ++l) {
                    const size_t   row_nolut = (size_t) (g * (QK_K / 8) + ib32 * 4 + l);
                    const uint16_t q         = blk.qs[ib32 * 4 + l];
                    const uint16_t cb        = (uint16_t) (q & 0x1FFu);
                    const uint8_t  sign_code = (uint8_t) ((q >> 9) & 0x7Fu);
                    const uint64_t grid      = iq2xs_grid_local[cb];

                    uint16_t packed_codes = 0;
                    for (int u = 0; u < 8; ++u) {
                        const uint8_t mag  = (uint8_t) ((grid >> (8 * u)) & 0xFFu);
                        const uint8_t code = q2s_mag_to_nolut_code(mag);
                        packed_codes       = (uint16_t) (packed_codes | ((uint16_t) code << (2 * u)));
                    }
                    out.codebook_nolut[idx2(row_nolut, (size_t) n, (size_t) N)] = packed_codes;

                    const uint8_t sign_byte = ksigns_iq2xs_local[sign_code];
                    const size_t  row_sign  = row_nolut ^ 1u;
                    rpp_store_packed_u8(out.sign, row_sign, (size_t) n, (size_t) N, reverse_bits_u8(sign_byte));
                }
            }
        }
    }
}

static void convert_iq2_xs_to_nolut(const block_iq2_xs_dbg * q2xs_blocks, int K, int N, void * out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq2xs_to_nolut requires K %% 256 == 0\n");
        std::abort();
    }

    const int    ng                = K / QK_K;
    // if ((int)q2xs_blocks.size() != ng * N) {
    //     std::fprintf(stderr, "convert_iq2xs_to_nolut input block count mismatch\n");
    //     std::abort();
    // }
    const size_t codebook_nolut_sz = (size_t) (K / 8) * (size_t) N;
    const size_t scales_size       = (size_t) (K / 64) * (size_t) N;
    const size_t sign_size         = (size_t) (K / 16) * (size_t) N;
    const size_t super_scale_size  = (size_t) ng * (size_t) N;

    uint8_t * base           = static_cast<uint8_t *>(out);
    auto *    codebook_nolut = reinterpret_cast<uint16_t *>(base);
    auto *    scales         = reinterpret_cast<uint16_t *>(codebook_nolut + codebook_nolut_sz);
    auto *    sign           = reinterpret_cast<uint16_t *>(scales + scales_size);
    auto *    super_scale    = reinterpret_cast<bfloat16_u16 *>(sign + sign_size);

    const size_t total_bytes =
        (codebook_nolut_sz + scales_size + sign_size) * sizeof(uint16_t) + super_scale_size * sizeof(bfloat16_u16);
    // std::memset(out, 0, total_bytes);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq2_xs_dbg & blk = q2xs_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                const size_t row_scale = (size_t) (g * (QK_K / 32) + ib32);
                rpp_store_packed_u8(scales, row_scale, (size_t) n, (size_t) N, blk.scales[ib32]);

                for (int l = 0; l < 4; ++l) {
                    const size_t   row_nolut = (size_t) (g * (QK_K / 8) + ib32 * 4 + l);
                    const uint16_t q         = blk.qs[ib32 * 4 + l];
                    const uint16_t cb        = (uint16_t) (q & 0x1FFu);
                    const uint8_t  sign_code = (uint8_t) ((q >> 9) & 0x7Fu);
                    const uint64_t grid      = iq2xs_grid_local[cb];

                    uint16_t packed_codes = 0;
                    for (int u = 0; u < 8; ++u) {
                        const uint8_t mag  = (uint8_t) ((grid >> (8 * u)) & 0xFFu);
                        const uint8_t code = q2s_mag_to_nolut_code(mag);
                        packed_codes       = (uint16_t) (packed_codes | ((uint16_t) code << (2 * u)));
                    }
                    codebook_nolut[idx2(row_nolut, (size_t) n, (size_t) N)] = packed_codes;

                    const uint8_t sign_byte = ksigns_iq2xs_local[sign_code];
                    const size_t  row_sign  = row_nolut ^ 1u;
                    rpp_store_packed_u8(sign, row_sign, (size_t) n, (size_t) N, reverse_bits_u8(sign_byte));
                }
            }
        }
    }
}

#pragma pack(push, 1)

struct block_iq2_s_dbg {
    uint16_t d;
    uint8_t  qs[QK_K / 4];
    uint8_t  qh[QK_K / 32];
    uint8_t  scales[QK_K / 32];
};

#pragma pack(pop)

static_assert(sizeof(block_iq2_s_dbg) == sizeof(uint16_t) + QK_K / 4 + QK_K / 16, "wrong iq2_s block size/padding");

struct RppQ2sParams {
    // shape [K/16][N], each word stores 2x 8-bit low index bytes for iq2s grid lookup
    std::vector<uint16_t>     codebook_lsb;
    // shape [K/64][N], each word stores 2x packed qh bytes (2 high bits per 8-value sub-block)
    std::vector<uint16_t>     codebook_msb;
    // shape [K/64][N], each word stores 2x packed scale bytes (low/high nibble for 2x16 values)
    std::vector<uint16_t>     scales;
    // shape [K/16][N], each word stores 2x sign bytes (each byte controls one 8-value group)
    // Bit order in each byte is reversed for kernel MSB-first consumption:
    // local weight 0 -> bit7, ..., local weight 7 -> bit0.
    std::vector<uint16_t>     sign;
    // shape [K/256][N], bf16 super scale
    std::vector<bfloat16_u16> super_scale;
};

static void convert_iq2s_to_rpp(const std::vector<block_iq2_s_dbg> & q2_blocks, int K, int N, RppQ2sParams & out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq2s_to_rpp requires K %% 256 == 0\n");
        std::abort();
    }

    const int ng = K / QK_K;
    if ((int) q2_blocks.size() != ng * N) {
        std::fprintf(stderr, "convert_iq2s_to_rpp input block count mismatch\n");
        std::abort();
    }

    out.codebook_lsb.assign((size_t) (K / 16) * (size_t) N, 0);
    out.codebook_msb.assign((size_t) (K / 64) * (size_t) N, 0);
    out.scales.assign((size_t) (K / 64) * (size_t) N, 0);
    out.sign.assign((size_t) (K / 16) * (size_t) N, 0);
    out.super_scale.assign((size_t) ng * (size_t) N, bfloat16_u16{ 0 });

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq2_s_dbg & blk = q2_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            out.super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                const size_t byte_row_msb = (size_t) (g * (QK_K / 32) + ib32);
                rpp_store_packed_u8(out.codebook_msb, byte_row_msb, (size_t) n, (size_t) N, blk.qh[ib32]);
                rpp_store_packed_u8(out.scales, byte_row_msb, (size_t) n, (size_t) N, blk.scales[ib32]);

                for (int l = 0; l < 4; ++l) {
                    const size_t byte_row_lsb = (size_t) (g * (QK_K / 8) + ib32 * 4 + l);
                    rpp_store_packed_u8(out.codebook_lsb, byte_row_lsb, (size_t) n, (size_t) N, blk.qs[ib32 * 4 + l]);
                    const uint8_t sign_byte     = blk.qs[QK_K / 8 + ib32 * 4 + l];
                    // q2s dequant kernel consumes sign bytes with opposite byte parity inside each u16 word
                    // compared to codebook_lsb indexing (first 8 values read from high byte).
                    const size_t  byte_row_sign = byte_row_lsb ^ 1u;
                    rpp_store_packed_u8(out.sign, byte_row_sign, (size_t) n, (size_t) N, reverse_bits_u8(sign_byte));
                }
            }
        }
    }
}

static void convert_iq2_s_to_rpp(const block_iq2_s_dbg * q2_blocks, int K, int N, void * out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq2_s_to_rpp requires K %% 256 == 0\n");
        std::abort();
    }
    if (!out) {
        std::fprintf(stderr, "out != nullptr required\n");
        std::abort();
    }

    const int    ng                = K / QK_K;
    const size_t codebook_lsb_size = (size_t) (K / 16) * (size_t) N;
    const size_t codebook_msb_size = (size_t) (K / 64) * (size_t) N;
    const size_t scales_size       = (size_t) (K / 64) * (size_t) N;
    const size_t sign_size         = (size_t) (K / 16) * (size_t) N;
    const size_t super_scale_size  = (size_t) ng * (size_t) N;

    uint8_t * base         = static_cast<uint8_t *>(out);
    auto *    codebook_lsb = reinterpret_cast<uint16_t *>(base);
    auto *    codebook_msb = reinterpret_cast<uint16_t *>(codebook_lsb + codebook_lsb_size);
    auto *    scales       = reinterpret_cast<uint16_t *>(codebook_msb + codebook_msb_size);
    auto *    sign         = reinterpret_cast<uint16_t *>(scales + scales_size);
    auto *    super_scale  = reinterpret_cast<bfloat16_u16 *>(sign + sign_size);

    const size_t total_bytes = (codebook_lsb_size + codebook_msb_size + scales_size + sign_size) * sizeof(uint16_t) +
                               super_scale_size * sizeof(bfloat16_u16);
    // std::memset(out, 0, total_bytes);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq2_s_dbg & blk = q2_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                const size_t byte_row_msb = (size_t) (g * (QK_K / 32) + ib32);
                rpp_store_packed_u8_u16(codebook_msb, byte_row_msb, (size_t) n, (size_t) N, blk.qh[ib32]);
                rpp_store_packed_u8_u16(scales, byte_row_msb, (size_t) n, (size_t) N, blk.scales[ib32]);

                for (int l = 0; l < 4; ++l) {
                    const size_t byte_row_lsb = (size_t) (g * (QK_K / 8) + ib32 * 4 + l);
                    rpp_store_packed_u8_u16(codebook_lsb, byte_row_lsb, (size_t) n, (size_t) N, blk.qs[ib32 * 4 + l]);

                    const uint8_t sign_byte     = blk.qs[QK_K / 8 + ib32 * 4 + l];
                    // q2s dequant kernel consumes sign bytes with opposite byte parity inside each u16 word.
                    const size_t  byte_row_sign = byte_row_lsb ^ 1u;
                    rpp_store_packed_u8_u16(sign, byte_row_sign, (size_t) n, (size_t) N, reverse_bits_u8(sign_byte));
                }
            }
        }
    }
}

struct RppQ2sNoLutParams {
    // shape [K/8][N], one uint16 per 8 weights, each weight uses 2-bit code.
    // code: 0->8, 1->25, 2->43
    std::vector<uint16_t>     codebook_nolut;
    // same layout as q2s LUT path
    std::vector<uint16_t>     scales;
    std::vector<uint16_t>     sign;
    std::vector<bfloat16_u16> super_scale;
};

static void convert_iq2_s_to_nolut(const std::vector<block_iq2_s_dbg> & q2_blocks,
                                   int                                  K,
                                   int                                  N,
                                   RppQ2sNoLutParams &                  out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq2s_to_nolut requires K %% 256 == 0\n");
        std::abort();
    }

    const int ng = K / QK_K;
    if ((int) q2_blocks.size() != ng * N) {
        std::fprintf(stderr, "convert_iq2s_to_nolut input block count mismatch\n");
        std::abort();
    }

    out.codebook_nolut.assign((size_t) (K / 8) * (size_t) N, 0);
    out.scales.assign((size_t) (K / 64) * (size_t) N, 0);
    out.sign.assign((size_t) (K / 16) * (size_t) N, 0);
    out.super_scale.assign((size_t) ng * (size_t) N, bfloat16_u16{ 0 });

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq2_s_dbg & blk = q2_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            out.super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                const size_t byte_row_scale = (size_t) (g * (QK_K / 32) + ib32);
                rpp_store_packed_u8(out.scales, byte_row_scale, (size_t) n, (size_t) N, blk.scales[ib32]);

                const uint8_t qh = blk.qh[ib32];
                for (int l = 0; l < 4; ++l) {
                    const size_t   row_nolut = (size_t) (g * (QK_K / 8) + ib32 * 4 + l);
                    const uint8_t  ql        = blk.qs[ib32 * 4 + l];
                    const uint16_t idx10     = (uint16_t) (ql | (((qh >> (2 * l)) & 0x3u) << 8));
                    const uint64_t grid      = iq2s_grid_local[idx10];

                    uint16_t packed_codes = 0;
                    for (int u = 0; u < 8; ++u) {
                        const uint8_t mag  = (uint8_t) ((grid >> (8 * u)) & 0xFFu);
                        const uint8_t code = q2s_mag_to_nolut_code(mag);
                        packed_codes       = (uint16_t) (packed_codes | ((uint16_t) code << (2 * u)));
                    }
                    out.codebook_nolut[idx2(row_nolut, (size_t) n, (size_t) N)] = packed_codes;

                    const uint8_t sign_byte     = blk.qs[QK_K / 8 + ib32 * 4 + l];
                    const size_t  byte_row_sign = row_nolut ^ 1u;
                    rpp_store_packed_u8(out.sign, byte_row_sign, (size_t) n, (size_t) N, reverse_bits_u8(sign_byte));
                }
            }
        }
    }
}

static void convert_iq2_s_to_nolut(const block_iq2_s_dbg * q2_blocks, int K, int N, void * out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_iq2s_to_nolut requires K %% 256 == 0\n");
        std::abort();
    }
    if (!out) {
        std::fprintf(stderr, "out != nullptr required\n");
        std::abort();
    }

    const int    ng                = K / QK_K;
    const size_t codebook_nolut_sz = (size_t) (K / 8) * (size_t) N;
    const size_t scales_size       = (size_t) (K / 64) * (size_t) N;
    const size_t sign_size         = (size_t) (K / 16) * (size_t) N;
    const size_t super_scale_size  = (size_t) ng * (size_t) N;

    uint8_t * base           = static_cast<uint8_t *>(out);
    auto *    codebook_nolut = reinterpret_cast<uint16_t *>(base);
    auto *    scales         = reinterpret_cast<uint16_t *>(codebook_nolut + codebook_nolut_sz);
    auto *    sign           = reinterpret_cast<uint16_t *>(scales + scales_size);
    auto *    super_scale    = reinterpret_cast<bfloat16_u16 *>(sign + sign_size);

    const size_t total_bytes =
        (codebook_nolut_sz + scales_size + sign_size) * sizeof(uint16_t) + super_scale_size * sizeof(bfloat16_u16);
    // std::memset(out, 0, total_bytes);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_iq2_s_dbg & blk = q2_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(fp16_to_float(blk.d));

            for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                const size_t byte_row_scale = (size_t) (g * (QK_K / 32) + ib32);
                rpp_store_packed_u8_u16(scales, byte_row_scale, (size_t) n, (size_t) N, blk.scales[ib32]);

                const uint8_t qh = blk.qh[ib32];
                for (int l = 0; l < 4; ++l) {
                    const size_t   row_nolut = (size_t) (g * (QK_K / 8) + ib32 * 4 + l);
                    const uint8_t  ql        = blk.qs[ib32 * 4 + l];
                    const uint16_t idx10     = (uint16_t) (ql | (((qh >> (2 * l)) & 0x3u) << 8));
                    const uint64_t grid      = iq2s_grid_local[idx10];

                    uint16_t packed_codes = 0;
                    for (int u = 0; u < 8; ++u) {
                        const uint8_t mag  = (uint8_t) ((grid >> (8 * u)) & 0xFFu);
                        const uint8_t code = q2s_mag_to_nolut_code(mag);
                        packed_codes       = (uint16_t) (packed_codes | ((uint16_t) code << (2 * u)));
                    }
                    codebook_nolut[idx2(row_nolut, (size_t) n, (size_t) N)] = packed_codes;

                    const uint8_t sign_byte     = blk.qs[QK_K / 8 + ib32 * 4 + l];
                    const size_t  byte_row_sign = row_nolut ^ 1u;
                    rpp_store_packed_u8_u16(sign, byte_row_sign, (size_t) n, (size_t) N, reverse_bits_u8(sign_byte));
                }
            }
        }
    }
}

static void decode_q6k_block_to_L(const block_q6_k_dbg & blk, uint8_t * L_out) {
    const uint8_t * ql = blk.ql;
    const uint8_t * qh = blk.qh;
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t ql0 = (uint8_t) (ql[l + 0] & 0xF);
            const uint8_t ql2 = (uint8_t) (ql[l + 32] & 0xF);
            const uint8_t ql1 = (uint8_t) (ql[l + 0] >> 4);
            const uint8_t ql3 = (uint8_t) (ql[l + 32] >> 4);
            const uint8_t qh0 = (uint8_t) ((qh[l] >> 0) & 0x3);
            const uint8_t qh2 = (uint8_t) ((qh[l] >> 2) & 0x3);
            const uint8_t qh1 = (uint8_t) ((qh[l] >> 4) & 0x3);
            const uint8_t qh3 = (uint8_t) ((qh[l] >> 6) & 0x3);

            L_out[j + l + 0]  = (uint8_t) (ql0 | (qh0 << 4));
            L_out[j + l + 32] = (uint8_t) (ql2 | (qh2 << 4));
            L_out[j + l + 64] = (uint8_t) (ql1 | (qh1 << 4));
            L_out[j + l + 96] = (uint8_t) (ql3 | (qh3 << 4));
        }
        ql += 64;
        qh += 32;
    }
}

static inline bool rpp_env_flag_enabled(const char * name) {
    const char * v = getenv(name);
    if (!v || !v[0]) {
        return false;
    }
    if ((v[0] == '0' && v[1] == '\0') || std::strcmp(v, "false") == 0 || std::strcmp(v, "False") == 0 ||
        std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 || std::strcmp(v, "OFF") == 0) {
        return false;
    }
    return true;
}

static inline int q6k_swap_ql_src_index(int i) {
    // Swap ql low/high nibble lanes inside each 128-value half:
    // [0..31]<->[64..95], [32..63]<->[96..127].
    return (i & ~127) | ((i & 127) ^ 64);
}

// Convert ggml-like q6 blocks -> RPP Q6KParams (packed weights + scales + super_scale)
static void convert_q6_k_to_rpp(const block_q6_k_dbg * q6_blocks, int K, int N, RppQ6kParams & out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_q6k_to_rpp requires K %% 256 == 0\n");
        std::abort();
    }

    const int  ng              = K / QK_K;
    const bool swap_scale_pair = rpp_env_flag_enabled("GGML_RPP_Q6K_SWAP_SCALE_PAIR");
    const bool swap_ql         = rpp_env_flag_enabled("GGML_RPP_Q6K_SWAP_QL");
    if (swap_scale_pair || swap_ql) {
        static bool s_q6k_pack_dbg_once = false;
        if (!s_q6k_pack_dbg_once) {
            std::fprintf(stderr, "[RPP Q6K PACK] GGML_RPP_Q6K_SWAP_SCALE_PAIR=%d GGML_RPP_Q6K_SWAP_QL=%d\n",
                         (int) swap_scale_pair, (int) swap_ql);
            s_q6k_pack_dbg_once = true;
        }
    }

    out.weights_ql.assign((size_t) (K / 4) * (size_t) N, 0);
    out.weights_qh.assign((size_t) (K / 8) * (size_t) N, 0);
    out.scales.assign((size_t) (K / 32) * (size_t) N, 0);
    out.super_scale.assign((size_t) ng * (size_t) N, bfloat16_u16{ 0 });

    std::vector<uint8_t> L(QK_K);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_q6_k_dbg & blk = q6_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            decode_q6k_block_to_L(blk, L.data());

            const float d                                             = fp16_to_float(blk.d);
            out.super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(d);

            for (int ib = 0; ib < QK_K / 16; ib += 2) {
                const size_t  pack_row = (size_t) ((g * (QK_K / 16) + ib) / 2);
                const size_t  pack_idx = idx2(pack_row, (size_t) n, (size_t) N);
                const uint8_t s0       = (uint8_t) blk.scales[ib + 0];
                const uint8_t s1       = (uint8_t) blk.scales[ib + 1];
                out.scales[pack_idx] =
                    swap_scale_pair ? (uint16_t) (s1 | ((uint16_t) s0 << 8)) : (uint16_t) (s0 | ((uint16_t) s1 << 8));
            }

            for (int i = 0; i < QK_K; ++i) {
                const int     k    = g * QK_K + i;
                const uint8_t l_hi = L[i] & 0x3F;
                const uint8_t l_lo = L[swap_ql ? q6k_swap_ql_src_index(i) : i] & 0x3F;
                const uint8_t lo   = l_lo & 0xF;
                const uint8_t hi   = (l_hi >> 4) & 0x3;

                size_t     ql_row   = (size_t) (k / 4);
                size_t     ql_idx   = idx2(ql_row, (size_t) n, (size_t) N);
                uint16_t & ql_word  = out.weights_ql[ql_idx];
                const int  ql_shift = (k & 3) * 4;
                ql_word             = (uint16_t) ((ql_word & ~(0xFu << ql_shift)) | ((uint16_t) lo << ql_shift));

                size_t     qh_row   = (size_t) (k / 8);
                size_t     qh_idx   = idx2(qh_row, (size_t) n, (size_t) N);
                uint16_t & qh_word  = out.weights_qh[qh_idx];
                const int  qh_shift = (k & 7) * 2;
                qh_word             = (uint16_t) ((qh_word & ~(0x3u << qh_shift)) | ((uint16_t) hi << qh_shift));
            }
        }
    }
}

static void convert_q6_k_to_rpp(const block_q6_k_dbg * q6_blocks, int K, int N, void * out) {
    if (K % QK_K != 0) {
        std::fprintf(stderr, "convert_q6k_to_rpp requires K %% 256 == 0\n");
        std::abort();
    }
    if (!out) {
        std::fprintf(stderr, "out != nullptr required\n");
        std::abort();
    }

    const int  ng              = K / QK_K;
    const bool swap_scale_pair = rpp_env_flag_enabled("GGML_RPP_Q6K_SWAP_SCALE_PAIR");
    const bool swap_ql         = rpp_env_flag_enabled("GGML_RPP_Q6K_SWAP_QL");
    if (swap_scale_pair || swap_ql) {
        static bool s_q6k_pack_dbg_once2 = false;
        if (!s_q6k_pack_dbg_once2) {
            std::fprintf(stderr, "[RPP Q6K PACK] GGML_RPP_Q6K_SWAP_SCALE_PAIR=%d GGML_RPP_Q6K_SWAP_QL=%d\n",
                         (int) swap_scale_pair, (int) swap_ql);
            s_q6k_pack_dbg_once2 = true;
        }
    }

    const size_t weights_ql_size   = K / 4 * N;
    const size_t weights_qh_size   = K / 8 * N;
    const size_t scales_size       = K / 32 * N;
    const size_t super_scales_size = ng * N;
    (void) super_scales_size;

    uint8_t * base        = static_cast<uint8_t *>(out);
    auto *    weights_ql  = reinterpret_cast<uint16_t *>(base);
    auto *    weights_qh  = reinterpret_cast<uint16_t *>(weights_ql + weights_ql_size);
    auto *    scales      = reinterpret_cast<uint16_t *>(weights_qh + weights_qh_size);
    auto *    super_scale = reinterpret_cast<bfloat16_u16 *>(scales + scales_size);

    std::vector<uint8_t> L(QK_K);

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < ng; ++g) {
            const block_q6_k_dbg & blk = q6_blocks[idx2((size_t) n, (size_t) g, (size_t) ng)];
            decode_q6k_block_to_L(blk, L.data());

            const float d                                         = fp16_to_float(blk.d);
            super_scale[idx2((size_t) g, (size_t) n, (size_t) N)] = float_to_bf16_rne(d);

            for (int ib = 0; ib < QK_K / 16; ib += 2) {
                const size_t  pack_row = (size_t) ((g * (QK_K / 16) + ib) / 2);
                const size_t  pack_idx = idx2(pack_row, (size_t) n, (size_t) N);
                const uint8_t s0       = (uint8_t) blk.scales[ib + 0];
                const uint8_t s1       = (uint8_t) blk.scales[ib + 1];
                scales[pack_idx] =
                    swap_scale_pair ? (uint16_t) (s1 | ((uint16_t) s0 << 8)) : (uint16_t) (s0 | ((uint16_t) s1 << 8));
            }

            for (int i = 0; i < QK_K; ++i) {
                const int     k    = g * QK_K + i;
                const uint8_t l_hi = L[i] & 0x3F;
                const uint8_t l_lo = L[swap_ql ? q6k_swap_ql_src_index(i) : i] & 0x3F;
                const uint8_t lo   = l_lo & 0xF;
                const uint8_t hi   = (l_hi >> 4) & 0x3;

                size_t     ql_row   = (size_t) (k / 4);
                size_t     ql_idx   = idx2(ql_row, (size_t) n, (size_t) N);
                uint16_t & ql_word  = weights_ql[ql_idx];
                const int  ql_shift = (k & 3) * 4;
                ql_word             = (uint16_t) ((ql_word & ~(0xFu << ql_shift)) | ((uint16_t) lo << ql_shift));

                size_t     qh_row   = (size_t) (k / 8);
                size_t     qh_idx   = idx2(qh_row, (size_t) n, (size_t) N);
                uint16_t & qh_word  = weights_qh[qh_idx];
                const int  qh_shift = (k & 7) * 2;
                qh_word             = (uint16_t) ((qh_word & ~(0x3u << qh_shift)) | ((uint16_t) hi << qh_shift));
            }
        }
    }
}

struct RppBf16Params {
    // format is [N/32][K][32]
    std::vector<uint16_t> wf16;
};

// Convert FP32 gguf weight Wf[N][K] into RPP BF16 packed format [N/32][K][32].
static void convert_fp32_to_rpp_bf16(const std::vector<float> & wf_nk, int N, int K, RppBf16Params & out) {
    if (N % 32 != 0) {
        std::fprintf(stderr, "convert_fp32_to_rpp_bf16 requires N %% 32 == 0\n");
        std::abort();
    }
    if ((int) wf_nk.size() != N * K) {
        std::fprintf(stderr, "convert_fp32_to_rpp_bf16 input shape mismatch\n");
        std::abort();
    }

    out.wf16.assign((size_t) N * (size_t) K, 0);
    for (int n = 0; n < N; ++n) {
        const int n_blk  = n / 32;
        const int n_lane = n & 31;
        for (int k = 0; k < K; ++k) {
            const size_t src = idx2((size_t) n, (size_t) k, (size_t) K);  // Wf[N][K]
            const size_t dst = ((size_t) n_blk * (size_t) K + (size_t) k) * 32u + (size_t) n_lane;
            out.wf16[dst]    = float_to_bf16_rne(wf_nk[src]);
        }
    }
}

static void convert_fp32_to_rpp_bf16(const float * wf_nk, int K, int N, void * out) {
    if (wf_nk == nullptr) {
        std::fprintf(stderr, "convert_fp32_to_rpp_bf16 requires wf_nk != nullptr\n");
        std::abort();
    }
    if (out == nullptr) {
        std::fprintf(stderr, "convert_fp32_to_rpp_bf16 requires out != nullptr\n");
        std::abort();
    }

    const int Kp = (int) (((uint32_t) K + 31u) & ~31u);
    const int Np = (int) (((uint32_t) N + 31u) & ~31u);
    uint16_t * out_wf16 = reinterpret_cast<uint16_t *>(out);
    std::memset(out_wf16, 0, (size_t) Kp * (size_t) Np * sizeof(uint16_t));
    for (int n = 0; n < N; ++n) {
        const int n_blk  = n / 32;
        const int n_lane = n & 31;
        for (int k = 0; k < K; ++k) {
            const size_t src = idx2((size_t) n, (size_t) k, (size_t) K);  // Wf[N][K]
            const size_t dst = ((size_t) n_blk * (size_t) Kp + (size_t) k) * 32u + (size_t) n_lane;
            out_wf16[dst]    = float_to_bf16_rne(wf_nk[src]);
        }
    }
}

static void convert_f16_to_rpp_bf16(const ggml_fp16_t * wf_nk, int K, int N, void * out) {
    if (wf_nk == nullptr) {
        std::fprintf(stderr, "convert_f16_to_rpp_bf16 requires wf_nk != nullptr\n");
        std::abort();
    }
    if (out == nullptr) {
        std::fprintf(stderr, "convert_f16_to_rpp_bf16 requires out != nullptr\n");
        std::abort();
    }

    const int Kp = (int) (((uint32_t) K + 31u) & ~31u);
    const int Np = (int) (((uint32_t) N + 31u) & ~31u);
    uint16_t * out_wf16 = reinterpret_cast<uint16_t *>(out);
    std::memset(out_wf16, 0, (size_t) Kp * (size_t) Np * sizeof(uint16_t));
    for (int n = 0; n < N; ++n) {
        const int n_blk  = n / 32;
        const int n_lane = n & 31;
        for (int k = 0; k < K; ++k) {
            const size_t src = idx2((size_t) n, (size_t) k, (size_t) K);  // Wf[N][K]
            const size_t dst = ((size_t) n_blk * (size_t) Kp + (size_t) k) * 32u + (size_t) n_lane;
            out_wf16[dst]    = float_to_bf16_rne(fp16_to_float(wf_nk[src]));
        }
    }
}

static void convert_bf16_to_rpp_bf16(const ggml_bf16_t * wf_nk, int K, int N, void * out) {
    if (wf_nk == nullptr) {
        std::fprintf(stderr, "convert_bf16_to_rpp_bf16 requires wf_nk != nullptr\n");
        std::abort();
    }
    if (out == nullptr) {
        std::fprintf(stderr, "convert_bf16_to_rpp_bf16 requires out != nullptr\n");
        std::abort();
    }

    const int Kp = (int) (((uint32_t) K + 31u) & ~31u);
    const int Np = (int) (((uint32_t) N + 31u) & ~31u);
    uint16_t * out_wf16 = reinterpret_cast<uint16_t *>(out);
    std::memset(out_wf16, 0, (size_t) Kp * (size_t) Np * sizeof(uint16_t));
    for (int n = 0; n < N; ++n) {
        const int n_blk  = n / 32;
        const int n_lane = n & 31;
        for (int k = 0; k < K; ++k) {
            const size_t src = idx2((size_t) n, (size_t) k, (size_t) K);  // Wf[N][K]
            const size_t dst = ((size_t) n_blk * (size_t) Kp + (size_t) k) * 32u + (size_t) n_lane;
            out_wf16[dst]    = wf_nk[src].bits;
        }
    }
}

enum RppReduceOperation : int { kSUM = 0, kPROD = 1, kMAX = 2, kMIN = 3, kAVG = 4 };

class RppDims {
  public:
    int & operator[](int i) {
        auto index = i >= 0 ? i : nbDims + i;
        return d[index];
    }

    static const int MAX_DIMS = 8;  //!< The maximum number of dimensions supported for a tensor.
    int              nbDims;        //!< The number of dimensions.
    int              d[MAX_DIMS];   //!< The extent of each dimension.

    int Length() {
        if (nbDims <= 0) {
            return 0;
        }
        int length = 1;
        for (int i = 0; i < nbDims && i < MAX_DIMS; i++) {
            length *= d[i];
        }
        return length;
    }
};

struct RppTaskElement {
    RppTaskElement() {}

    std::string taskName;

    // to be Modified
    struct Params {
        std::vector<uint32_t> kernelList;
    } params;

    dim3 gridDim;      // only if type is kKERNEL, gridDim is a kernel parameter
    dim3 blockDim;     // only if type is kKERNEL, blockDim is a kernel parameter
    dim3 subBlockDim;  // conv speciality

    void setBlockDim(uint32_t x, uint32_t y, uint32_t z) {
        if (x * y * z > 8191 || x * y * z <= 32) {
            throw -1;
        }
        blockDim.x    = x;
        blockDim.y    = y;
        blockDim.z    = z;
        subBlockDim.x = x;
        subBlockDim.y = y;
        subBlockDim.z = z;
    }

    uint32_t getBlockLen() const { return blockDim.x * blockDim.y * blockDim.z; }

    void setGridDim(uint32_t x, uint32_t y, uint32_t z) {
        if (x == 0 || x > 65535 || y == 0 || y > 65535 || z == 0 || z > 65535) {
            throw -1;
        }
        gridDim.x = x;
        gridDim.y = y;
        gridDim.z = z;
    }

    uint32_t getGridLen() const { return gridDim.x * gridDim.y * gridDim.z; }

    void appendParam(uint32_t param) { params.kernelList.emplace_back(param); }
};

static inline void get_linear_blocks(int N, int Ns, int & n, int & Nt) {
    assert(N > 0);
    assert(Ns > 0);

    // number of blocks (must be >= 1)
    n = (N + Ns - 1) / Ns;  // ceil_div

    // size of the tail block
    Nt = N - (n - 1) * Ns;

    // safety
    assert(n >= 1);
    assert(Nt >= 0 && Nt <= Ns);
}

void launchWrapperAysnc(std::string             kernName,
                        dim3                    blocksPerGrid,
                        dim3                    threadsPerBlock,
                        std::vector<uint32_t> & kparams,
                        RPPmodule               cuMod,
                        RPPstream               kernelStream);

void launchWrapperAysnc(std::string             kernName,
                        dim3                    blocksPerGrid,
                        dim3                    threadsPerBlock,
                        dim3                    threadsPerBlockTail,
                        std::vector<uint32_t> & kparams,
                        RPPmodule               cuMod,
                        RPPstream               kernelStream);

bool tryCustomKernelLaunch(const std::string &     kernName,
                           dim3                    blocksPerGrid,
                           dim3                    threadsPerBlock,
                           dim3                    threadsPerBlockTail,
                           std::vector<uint32_t> & kparams,
                           RPPmodule               cuMod,
                           RPPstream               kernelStream);

void adjustKernelLaunchConfig(const std::string &     kernName,
                              dim3 &                  blocksPerGrid,
                              dim3 &                  threadsPerBlock,
                              dim3 &                  threadsPerBlockTail,
                              std::vector<uint32_t> & kparams);

std::vector<RppTaskElement> create_reduce_kernel_task(size_t             axis_index,
                                                      RppReduceOperation operation,
                                                      uint32_t           input_addr,
                                                      uint32_t           output_addr,
                                                      RppDims &          input_dims_ori,
                                                      RppDims &          output_dims_ori,
                                                      uint32_t           mean_addr,
                                                      bool               for_reduce_task,
                                                      bool               is_input_chw32);
bool                        ReduceSpawnIO(int       axis,
                                          RppDims & in_dims,
                                          RppDims & out_dims,
                                          RppDims & mid_dims,
                                          bool      for_reduce_task,
                                          bool      for_chw32);

void fill_16bits_align_params(int                     output,
                              int                     block_x,
                              int16_t                 value,
                              int                     type_of_bytes,
                              std::vector<uint32_t> & params);

void cvt_kernel_param_init(const dim3 &            threadsPerBlock,
                           uint32_t                inAddr,
                           uint32_t                outAddr,
                           RppDataType             inDataType,
                           RppDataType             outDataType,
                           std::vector<uint32_t> & params);

void cvt_kernel_param_init_opt(const dim3 &            threadsPerBlock,
                               uint32_t                inAddr,
                               uint32_t                outAddr,
                               RppDataType             inDataType,
                               RppDataType             outDataType,
                               std::vector<uint32_t> & params);

void chw2chw32_align_params(int                     input,
                            int                     output,
                            int                     row,
                            int                     column,
                            int                     grid_y_tail,
                            int                     block_x,
                            int                     block_y,
                            int                     block_z,
                            int                     type_of_bytes,
                            std::vector<uint32_t> & params,
                            bool                    is_row_round);

void chw322chw_align_params(int                     input,
                            int                     output,
                            int                     row,
                            int                     column,
                            int                     grid_y_tail,
                            int                     block_x,
                            int                     block_y,
                            int                     block_z,
                            int                     type_of_bytes,
                            std::vector<uint32_t> & params);

void calc_tbdim_flattern(uint32_t D0, uint32_t D1, dim3 & threadsPerBlock, dim3 & blocksPerGrid);

uint32_t RppHW32Fix(uint32_t block_x, uint32_t block_y, uint32_t block_z);
