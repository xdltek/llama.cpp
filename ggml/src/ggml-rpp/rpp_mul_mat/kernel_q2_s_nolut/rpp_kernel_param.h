#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"

#include <rpp_runtime.h>

#include <cstdint>
#include <vector>

#define MATMUL_NS 128

namespace kernel_q2_s_nolut {
static void q2s_nolut_super_scale_params(uint32_t                in_scale,
                                         uint32_t                in_super_scale,
                                         uint32_t                in_lut,
                                         uint32_t                out_scale,
                                         uint32_t                K,
                                         uint32_t                N,
                                         uint32_t                super_group,
                                         uint32_t                q_group,
                                         dim3 &                  blocksPerGrid,
                                         dim3 &                  threadsPerBlock,
                                         std::vector<uint32_t> & params) {
    //----------------------------------------------------------------------------------------------------
    // super group = 256
    //----------------------------------------------------------------------------------------------------
    // in_super_scale   [K/256]  |     | [N]
    // in_super_scale   [sg]     |     | [N]
    // in_super_scale   [z]      | [1] | [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // out_scale   [K/256]     | [16]     |  [N]
    // out_scale   [z]         | [16]     |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    // in_scale   [K/256]  | [4]          |  [4][N]
    // in_scale   [K/256]  | [4]          |  [4][N]
    // in_scale   [z]      | [unroll]     |  [grid.x] * [x]
    //----------------------------------------------------------------------------------------------------
    uint32_t inStrideY          = N;
    uint32_t inStrideZ          = 4 * N;
    uint32_t outStrideY         = N;
    uint32_t outStrideZ         = 16 * N;
    uint32_t inUnRollStride     = N * sizeof(short);
    uint32_t outUnRollStride    = N * sizeof(short);
    uint32_t stride_per_block_x = threadsPerBlock.x * sizeof(short);

    uint32_t blockX = blocksPerGrid.x;
    blocksPerGrid.x = 1;
    params.emplace_back(in_scale);
    params.emplace_back(in_super_scale);
    params.emplace_back(out_scale);
    params.emplace_back(in_lut);
    params.emplace_back(inStrideY);
    params.emplace_back(inStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(inUnRollStride);
    params.emplace_back(outUnRollStride);
    params.emplace_back(stride_per_block_x);
    params.emplace_back(blockX);
}

static void q2s_nolut_dequant_params(dim3 &                  blocksPerGrid,
                                     dim3 &                  threadsPerBlock,
                                     uint32_t                in_wq,
                                     uint32_t                in_sign,
                                     uint32_t                scale,
                                     uint32_t                output,
                                     uint32_t                lut_addr,
                                     int32_t                 column,
                                     int32_t                 row,
                                     int                     in_type_of_bytes,
                                     int                     out_type_of_bytes,
                                     std::vector<uint32_t> & params) {
    const int32_t grid_x = (int32_t) blocksPerGrid.x;
    const int32_t grid_y = (int32_t) blocksPerGrid.y;
    const int32_t grid_z = (int32_t) blocksPerGrid.z;

    const int32_t block_x = (int32_t) threadsPerBlock.x;
    const int32_t block_y = (int32_t) threadsPerBlock.y;
    const int32_t block_z = (int32_t) threadsPerBlock.z;

    (void) grid_z;
    (void) in_type_of_bytes;
    (void) out_type_of_bytes;
    //----------------------------------------------------------------------------------------------------
    // codebook_nolut   [K/256]      |  [4]  |  [4]       | [2] |  [8][N]
    // codebook_nolut   [z]          |  [y]  |  [unroll]        |  [x]
    //                  [grid.y]*[z] |  [y]  |                  |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // qsign      [K/256]       |  [4]  |  [4]       |  [16][N]
    // qsign      [z]           |  [y]  |  [unroll]  |  [x]
    //            [grid.y]*[z]  |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // scale      [K/256]       |  [4]  |  [4]       |  [N]
    // scale      [z]           |  [y]  |  [unroll]  |  [x]
    //            [grid.y]*[z]  |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    // out        [K/256]       |  [4]  |  [4]       |  [16]        |  [N]
    // out        [z]           |  [y]  |  [unroll0] |  [unroll1]   |  [x]
    //            [grid.y]*[z]  |  [y]                              |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------

    //qsign/codebook_lsb/scale share same strideY & strideZ
    int inStrideY0 = 4 * 2 * column;
    int inStrideZ0 = 4 * inStrideY0;
    int inStrideY1 = 4 * column;
    int inStrideZ1 = 4 * inStrideY1;

    int in_unroll_stride = column * sizeof(short);
    int inBlockXSize0    = block_x * sizeof(short);
    inBlockXSize0 -= (4 * 2) * in_unroll_stride;
    int inBlockXSize1 = block_x * sizeof(short);

    int inBlockYSize0 = block_y * block_z * inStrideY0 * sizeof(short);
    inBlockYSize0 -= grid_x * block_x * sizeof(short);

    int inBlockYSize1 = block_y * block_z * inStrideY1 * sizeof(short);
    inBlockYSize1 -= grid_x * block_x * sizeof(short);
    //----------------------------------------------------------------------------------------------------
    // out        [K/256]       |  [4]  |  [4]       |  [16]        |  [N]
    // out        [z]           |  [y]  |  [unroll0] |  [unroll1]   |  [x]
    //            [grid.y]*[z]  |  [y]                              |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    int outStrideY        = 4 * 16 * block_x;
    int outStrideZ        = 4 * outStrideY;
    int out_unroll_stride = block_x * sizeof(short);
    int outBlockXSize     = block_x * row * sizeof(short);
    outBlockXSize -= 64 * out_unroll_stride;
    int outBlockYSize = 64 * block_x * block_y * block_z * sizeof(short);
    outBlockYSize -= grid_x * block_x * row * sizeof(short);
    ;

    //----------------------------------------------------------------------------------------------------
    // scale      [K/256]       |  [4]  |  [4]       |  [N]
    // scale      [z]           |  [y]  |  [unroll]  |  [x]
    //            [grid.y]*[z]  |  [y]               |  [grid.x]*[x]
    //----------------------------------------------------------------------------------------------------
    int dequantStrideY    = 4 * column;
    int dequantStrideZ    = 4 * dequantStrideY;
    int deQuantBlockSizeX = block_x * sizeof(short);
    deQuantBlockSizeX -= (4) * in_unroll_stride;
    // Scale tensor uses dequantStrideY (= 4 * column), so Y-step must be based on
    // that full per-row stride. Also subtract the actual net X advance per tile
    // (block_x * sizeof(short)), not deQuantBlockSizeX (which is a corrective delta).
    int deQuantBlockSizeY = dequantStrideY * block_z * block_y * sizeof(short);
    deQuantBlockSizeY -= grid_x * block_x * (int) sizeof(short);
    blocksPerGrid.x = 1;
    blocksPerGrid.y = 1;

    params.emplace_back(in_wq);
    params.emplace_back(in_sign);
    params.emplace_back(scale);
    params.emplace_back(output);
    params.emplace_back(lut_addr);
    params.emplace_back(inStrideY0);
    params.emplace_back(inStrideZ0);
    params.emplace_back(inStrideY1);
    params.emplace_back(inStrideZ1);
    params.emplace_back(dequantStrideY);
    params.emplace_back(dequantStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(in_unroll_stride);
    params.emplace_back(out_unroll_stride);
    params.emplace_back(inBlockXSize0);
    params.emplace_back(inBlockYSize0);
    params.emplace_back(inBlockXSize1);
    params.emplace_back(inBlockYSize1);
    params.emplace_back(deQuantBlockSizeX);
    params.emplace_back(deQuantBlockSizeY);
    params.emplace_back(outBlockXSize);
    params.emplace_back(outBlockYSize);
    params.emplace_back(grid_x);
    params.emplace_back(grid_y);
}

static void matmul_opt_kernel_params(const dim3 &            blocksPerGrid,
                                     const dim3 &            threadsPerBlock,
                                     uint32_t                input_a,
                                     uint32_t                input_b,
                                     uint32_t                postScale,
                                     uint32_t                out,
                                     uint32_t                in0_row,
                                     uint32_t                in0_col,
                                     uint32_t                in1_row,
                                     uint32_t                in1_col,
                                     uint32_t                loop_in0,
                                     uint32_t                loop_in1,
                                     uint32_t                loop_out,
                                     int32_t                 tn_offset,
                                     int32_t                 tn,
                                     int                     in_type_of_bytes,
                                     int                     out_type_of_bytes,
                                     std::vector<uint32_t> & params,
                                     bool                    use_hw_output,
                                     bool                    is_expand) {
    const int32_t block_x = (int32_t) threadsPerBlock.x;
    const int32_t block_y = (int32_t) threadsPerBlock.y;
    (void) blocksPerGrid;

    uint32_t in0_col_round = (in0_col + 31) / 32 * 32;
    uint32_t in1_row_round = (in1_row + 31) / 32 * 32;
    if (in1_col == 32 && (in1_row % 32) > 0) {
        in1_row_round = in1_row;
    }
    uint32_t in1_col_round = (in1_col + 31) / 32 * 32;

    int inStrideY  = block_x * in_type_of_bytes;
    int outStrideY = block_x;
    if (use_hw_output) {
        outStrideY = (int) in1_col;
    }

    int inStrideZ  = 0;
    int outStrideZ = 0;
    int cn         = (int) ((in0_col_round + 31) / 32) - 1;
    int inSwitchSize =
        ((int) (in0_row - 1) * block_x + (in_type_of_bytes == (int) sizeof(short) ? 8 : 16)) * in_type_of_bytes;

    input_b += in1_row_round * block_x * in_type_of_bytes * (uint32_t) tn_offset;
    out += in0_row * block_x * out_type_of_bytes * (uint32_t) tn_offset;

    int gridx_inb_stride = in1_row_round * block_x * in_type_of_bytes * tn;
    int gridx_out_stride =
        use_hw_output ? block_x * out_type_of_bytes * tn : in0_row * block_x * out_type_of_bytes * tn;
    int gridy_ina_stride = block_x * block_y * in_type_of_bytes;
    int gridy_out_stride = block_x * block_y * out_type_of_bytes;

    int gridz_ina_stride = (loop_in0 == 1) ? 0 : (int) (in0_row * in0_col_round * in_type_of_bytes);
    int gridz_inb_stride = (loop_in1 == 1) ? 0 : (int) (in1_row_round * in1_col_round * in_type_of_bytes);
    int gridz_out_stride = (loop_out == 1) ? 0 : (int) (in0_row * in1_col_round * out_type_of_bytes);
    if (is_expand) {
        gridz_inb_stride = 0;
    }

    params.emplace_back(input_a);
    params.emplace_back(input_b);
    params.emplace_back(out);
    params.emplace_back((uint32_t) cn);
    params.emplace_back((uint32_t) 0);
    params.emplace_back((uint32_t) inStrideY);
    params.emplace_back((uint32_t) outStrideY);
    params.emplace_back((uint32_t) inStrideZ);
    params.emplace_back((uint32_t) outStrideZ);
    params.emplace_back((uint32_t) inSwitchSize);
    params.emplace_back((uint32_t) gridx_inb_stride);
    params.emplace_back((uint32_t) gridx_out_stride);
    params.emplace_back((uint32_t) gridy_ina_stride);
    params.emplace_back((uint32_t) gridy_out_stride);
    params.emplace_back((uint32_t) gridz_ina_stride);
    params.emplace_back((uint32_t) gridz_inb_stride);
    params.emplace_back((uint32_t) gridz_out_stride);

    if (tn > 1) {
        int outTnStride   = use_hw_output ? (int) (32 * sizeof(short)) : (int) (in0_row * 32 * sizeof(short));
        int filterOffset0 = (int) (in0_col_round * 32 * sizeof(short));
        int filterOffset1 = (int) (in0_col_round * 32 * sizeof(short) * (tn - 1) - 512);
        params.emplace_back((uint32_t) outTnStride);
        params.emplace_back((uint32_t) filterOffset0);
        params.emplace_back((uint32_t) filterOffset1);
    }
}

static inline int q2s_nolut_round_up(int a) {
    return (a + 511) / 512 * 512;
}

struct q2s_nolut_lut_workspace {
    static constexpr uint32_t qscale_lut_elems = 16;
    static constexpr uint32_t mag_lut_elems    = 4;
    static constexpr uint32_t qscale_lut_bytes = qscale_lut_elems * (uint32_t) sizeof(uint16_t);
    static constexpr uint32_t mag_lut_bytes    = mag_lut_elems * (uint32_t) sizeof(uint16_t);
    static constexpr uint32_t total_bytes      = qscale_lut_bytes + mag_lut_bytes;
};

struct q2s_nolut_sram_io {
    int M                     = 0;
    int K                     = 0;
    int N                     = 0;
    int weights_group         = 0;
    int in_bytes_per_element  = 0;
    int out_bytes_per_element = 0;

    uint32_t sizeA_in             = 0;
    uint32_t sizeA_bf16           = 0;
    uint32_t sizeB_codebook_nolut = 0;
    uint32_t sizeB_scales         = 0;
    uint32_t sizeB_sign           = 0;
    uint32_t sizeB_super_scale    = 0;
    uint32_t sizeB_scale_scratch  = 0;
    uint32_t sizeB_dequant        = 0;
    uint32_t sizeC_bf16           = 0;
    uint32_t sizeC32              = 0;
    uint32_t sizeC                = 0;
    uint32_t total_sram_bytes     = 0;

    RPPdeviceptr sram_base            = 0;
    RPPdeviceptr sramA_in             = 0;
    RPPdeviceptr sramA_out            = 0;
    RPPdeviceptr sramB_codebook_nolut = 0;
    RPPdeviceptr sramB_scales         = 0;
    RPPdeviceptr sramB_sign           = 0;
    RPPdeviceptr sramB_super_scale    = 0;
    RPPdeviceptr sramB_qscale_lut     = 0;
    RPPdeviceptr sramB_mag_lut        = 0;
    RPPdeviceptr sramB_scale          = 0;
    RPPdeviceptr sramB_out            = 0;
    RPPdeviceptr sramC_hw32           = 0;
    RPPdeviceptr sramC_chw            = 0;
    RPPdeviceptr sramC_chw_fp32       = 0;
    RPPdeviceptr sram_out             = 0;
};

// Full-tensor SRAM-direct layout for q2s_nolut:
// 1. Copy raw A to sramA_in.
// 2. Copy packed weights to sramB_codebook_nolut using the contiguous layout
//    [codebook_nolut | scales | sign | super_scale].
// 3. Copy LUT workspace to SRAM via q2s_nolut_copy_lut_workspace_to_sram().
// 4. Launch rpp_matmul_q2s_nolut_sram().
// 5. Read final output from sram_out or use q2s_nolut_cdma_copy_output_to_ddr().
static inline q2s_nolut_sram_io q2s_nolut_prepare_sram_io(rpp_kernel_context & ctx,
                                                          int                  M,
                                                          int                  K,
                                                          int                  N,
                                                          int                  weights_group,
                                                          int                  in_bytes_per_element,
                                                          int                  out_bytes_per_element) {
    constexpr uint32_t kSramLimitBytes = 22u * 1024u * 1024u;
    constexpr int      kSuperGroup     = 256;

    if (M <= 0 || K <= 0 || N <= 0) {
        throw std::runtime_error("Q2S NoLUT SRAM expects positive M/K/N");
    }
    if ((N % 32) != 0) {
        throw std::runtime_error("Q2S NoLUT SRAM currently expects N % 32 == 0");
    }
    if (weights_group != kSuperGroup || (K % weights_group) != 0) {
        throw std::runtime_error("Q2S NoLUT SRAM expects weights_group == 256 and K % 256 == 0");
    }

    q2s_nolut_sram_io io{};
    io.M                     = M;
    io.K                     = K;
    io.N                     = N;
    io.weights_group         = weights_group;
    io.in_bytes_per_element  = in_bytes_per_element;
    io.out_bytes_per_element = out_bytes_per_element;

    io.sizeA_in             = (uint32_t) ((uint64_t) M * (uint64_t) K * (uint64_t) in_bytes_per_element);
    io.sizeA_bf16           = (uint32_t) ((uint64_t) M * (uint64_t) K * sizeof(uint16_t));
    io.sizeB_codebook_nolut = (uint32_t) ((uint64_t) K * (uint64_t) N / 4ull);
    io.sizeB_scales         = (uint32_t) ((uint64_t) K * (uint64_t) N / 32ull);
    io.sizeB_sign           = (uint32_t) ((uint64_t) K * (uint64_t) N / 8ull);
    io.sizeB_super_scale    = (uint32_t) ((uint64_t) (K / weights_group) * (uint64_t) N * sizeof(uint16_t));
    io.sizeB_scale_scratch  = (uint32_t) ((uint64_t) K * (uint64_t) N / 16ull * sizeof(uint16_t));
    io.sizeB_dequant        = (uint32_t) ((uint64_t) K * (uint64_t) N * sizeof(uint16_t));
    io.sizeC_bf16           = (uint32_t) ((uint64_t) M * (uint64_t) N * sizeof(uint16_t));
    io.sizeC32              = (uint32_t) ((uint64_t) M * (uint64_t) N * sizeof(float));
    io.sizeC                = (uint32_t) ((uint64_t) M * (uint64_t) N * (uint64_t) out_bytes_per_element);

    io.sram_base            = ctx.virtual_sram_base;
    io.sramA_in             = io.sram_base;
    io.sramA_out            = io.sramA_in + (RPPdeviceptr) q2s_nolut_round_up((int) io.sizeA_in);
    io.sramB_codebook_nolut = io.sramA_out + (RPPdeviceptr) q2s_nolut_round_up((int) io.sizeA_bf16);
    io.sramB_scales         = io.sramB_codebook_nolut + (RPPdeviceptr) io.sizeB_codebook_nolut;
    io.sramB_sign           = io.sramB_scales + (RPPdeviceptr) io.sizeB_scales;
    io.sramB_super_scale    = io.sramB_sign + (RPPdeviceptr) io.sizeB_sign;
    const uint32_t sizeB_weights_packed =
        io.sizeB_codebook_nolut + io.sizeB_scales + io.sizeB_sign + io.sizeB_super_scale;
    io.sramB_qscale_lut = io.sramB_codebook_nolut + (RPPdeviceptr) q2s_nolut_round_up((int) sizeB_weights_packed);
    io.sramB_mag_lut =
        io.sramB_qscale_lut + (RPPdeviceptr) q2s_nolut_round_up((int) q2s_nolut_lut_workspace::qscale_lut_bytes);
    io.sramB_scale = io.sramB_mag_lut + (RPPdeviceptr) q2s_nolut_round_up((int) q2s_nolut_lut_workspace::mag_lut_bytes);
    io.sramB_out   = io.sramB_scale + (RPPdeviceptr) q2s_nolut_round_up((int) io.sizeB_scale_scratch);
    io.sramC_hw32  = io.sramB_out + (RPPdeviceptr) q2s_nolut_round_up((int) io.sizeB_dequant);
    io.sramC_chw   = io.sramC_hw32 + (RPPdeviceptr) q2s_nolut_round_up((int) io.sizeC_bf16);
    io.sramC_chw_fp32 = io.sramC_chw + (RPPdeviceptr) q2s_nolut_round_up((int) io.sizeC_bf16);
    io.sram_out       = (out_bytes_per_element == (int) sizeof(float)) ? io.sramC_chw_fp32 : io.sramC_chw;
    io.total_sram_bytes =
        (uint32_t) ((io.sramC_chw_fp32 + (RPPdeviceptr) q2s_nolut_round_up((int) io.sizeC32)) - io.sram_base);

    if (io.total_sram_bytes > kSramLimitBytes) {
        std::cerr << "SRAM overflow: need " << io.total_sram_bytes << " bytes, but allocated " << kSramLimitBytes
                  << " bytes\n";
        std::abort();
    }

    return io;
}

static inline RPPdeviceptr q2s_nolut_prepare_lut_workspace(rpp_kernel_context & ctx) {
    RPPdeviceptr dev_lut_workspace = ctx.dev_workspace;
    if (dev_lut_workspace == 0) {
        rtMalloc((void **) &dev_lut_workspace, q2s_nolut_lut_workspace::total_bytes);
        ctx.dev_workspace = dev_lut_workspace;
    }

    std::array<uint16_t, q2s_nolut_lut_workspace::qscale_lut_elems> qscale_lut = {};
    for (uint32_t i = 0; i < q2s_nolut_lut_workspace::qscale_lut_elems; ++i) {
        const float scale4  = (float) i;
        const float lut_val = (0.5f + scale4) * 0.25f;
        qscale_lut[i]       = float_to_bf16_rne(lut_val);
    }

    constexpr std::array<float, q2s_nolut_lut_workspace::mag_lut_elems> mag_lut_values = { 8.0f, 25.0f, 43.0f, 0.0f };
    std::array<uint16_t, q2s_nolut_lut_workspace::mag_lut_elems>        mag_lut        = {};
    for (uint32_t i = 0; i < q2s_nolut_lut_workspace::mag_lut_elems; ++i) {
        mag_lut[i] = float_to_bf16_rne(mag_lut_values[i]);
    }

    rtMemcpy((void *) dev_lut_workspace, qscale_lut.data(), q2s_nolut_lut_workspace::qscale_lut_bytes,
             rtMemcpyHostToDevice);
    rtMemcpy((void *) (dev_lut_workspace + q2s_nolut_lut_workspace::qscale_lut_bytes), mag_lut.data(),
             q2s_nolut_lut_workspace::mag_lut_bytes, rtMemcpyHostToDevice);
    return dev_lut_workspace;
}

static inline void q2s_nolut_cdma_d2s_async(RPPdeviceptr sram_dst,
                                            RPPdeviceptr ddr_src,
                                            size_t       bytes,
                                            RPPstream    stream) {
    rtMemcpyAsync((void *) sram_dst, (const void *) ddr_src, bytes, rtMemcpyDeviceToSram, stream);
}

static inline void q2s_nolut_cdma_s2d_async(RPPdeviceptr ddr_dst,
                                            RPPdeviceptr sram_src,
                                            size_t       bytes,
                                            RPPstream    stream) {
    rtMemcpyAsync((void *) ddr_dst, (const void *) sram_src, bytes, rtMemcpySramToDevice, stream);
}

static inline void q2s_nolut_copy_lut_workspace_to_sram(const q2s_nolut_sram_io & io,
                                                        RPPdeviceptr              dev_lut_workspace,
                                                        RPPstream                 stream) {
    q2s_nolut_cdma_d2s_async(io.sramB_qscale_lut, dev_lut_workspace, q2s_nolut_lut_workspace::qscale_lut_bytes, stream);
    q2s_nolut_cdma_d2s_async(io.sramB_mag_lut, dev_lut_workspace + q2s_nolut_lut_workspace::qscale_lut_bytes,
                             q2s_nolut_lut_workspace::mag_lut_bytes, stream);
}

static inline void q2s_nolut_cdma_copy_output_to_ddr(const q2s_nolut_sram_io & io,
                                                     RPPdeviceptr              devC,
                                                     RPPstream                 stream) {
    q2s_nolut_cdma_s2d_async(devC, io.sram_out, (size_t) io.sizeC, stream);
}

}  // namespace kernel_q2_s_nolut
