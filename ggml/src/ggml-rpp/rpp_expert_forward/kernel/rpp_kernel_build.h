#pragma once
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "rpp_drv_api.h"
#include "rpp_glu/kernel_swiglu/rpp_kernel_build.h"
#include "rpp_glu/kernel_swiglu/rpp_kernel_param.h"
#include "rpp_kernel_param.h"
#include "rpp_mul_mat/kernel_q2_s_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q2_xs_nolut/rpp_kernel_build.h"
#include "rpp_mul_mat/kernel_q3_xxs_nolut/rpp_kernel_build.h"

#include <rpp_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define MATID_GRAPH
#if defined(RPP_SIM_RT) || defined(WIN_X86_SIMDRV_ISS)
#    include "iss_mem_sim.h"
#endif

namespace kernel_expert_forward {

constexpr int      kDefaultTopK                     = 8;
constexpr int      kBf16Bytes                       = (int) sizeof(uint16_t);
constexpr int      kFloatBytes                      = (int) sizeof(float);
constexpr int      kQ2WeightsGroup                  = 256;
constexpr int      kQGroup                          = 32;
constexpr int      kGateTn                          = 4;
constexpr int      kDynamicMatmulBlockY             = 128;
constexpr int      kQ2FamilyFusionElementsPerThread = 32;
constexpr uint32_t kSiluLutBytes                    = 64u * 1024u * (uint32_t) sizeof(uint16_t);
constexpr uint32_t kQuantLutWorkspaceBytes          = kernel_q3_xxs_nolut::q3xxs_nolut_lut_workspace::total_bytes;
constexpr uint32_t kFusionLutWorkspaceBytes         = kQuantLutWorkspaceBytes + kSiluLutBytes;
constexpr uint32_t kMatmulBlockX                    = 32;
constexpr uint64_t kMatidFramBank0Base              = 0x1006000800ull;
constexpr uint64_t kMatidFramBank1Base              = 0x1006001800ull;
constexpr uint32_t kMatidExpertCountRegIndex        = 6u;
constexpr uint32_t kMatidExpertOffsetRegIndex       = 7u;

static void q2xs_super_scale_blocks_matid(uint32_t K,
                                          uint32_t super_group,
                                          uint32_t group,
                                          uint32_t N,
                                          dim3 &   threadsPerBlock,
                                          dim3 &   blocksPerGrid) {
    (void) group;
    const int nsg     = (int) (K / super_group);
    uint32_t  block_z = (uint32_t) nsg;
    uint32_t  block_x = N;
    uint32_t  cntX    = 1;
    while (block_x * block_z >= 8192) {
        block_x /= 2;
        cntX *= 2;
    }
    if (block_x * cntX * block_z != N * (uint32_t) nsg) {
        throw std::runtime_error("Matmul Q2XS TB Invalid");
    }

    threadsPerBlock.x = block_x;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = block_z;
    blocksPerGrid.x   = cntX;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;
}

static void q2xs_nolut_dequant_blocks_matid(uint32_t loop,
                                            uint32_t row,
                                            uint32_t column,
                                            dim3 &   threadsPerBlock,
                                            dim3 &   blocksPerGrid,
                                            int      weights_group) {
    (void) loop;
    const uint32_t elements_per_thread = 64;
    const uint32_t block_x             = kMatmulBlockX;
    const uint32_t block_y             = (uint32_t) weights_group / elements_per_thread;
    uint32_t       block_z             = 1;
    while (true) {
        if ((block_x * block_y * block_z) >= (128 * kMatmulBlockX)) {
            break;
        }
        if ((block_y * block_z * elements_per_thread) >= row) {
            break;
        }
        block_z *= 2;
    }

    const uint32_t base_rows_per_block_z1 = block_y * elements_per_thread;
    while (block_z > 1 && (row % (base_rows_per_block_z1 * block_z)) != 0) {
        block_z >>= 1;
    }

    const int sub_row_per_block = (int) (block_y * block_z * elements_per_thread);
    if (row < block_y * block_z) {
        throw -1;
    }
    if ((row % (uint32_t) sub_row_per_block) != 0) {
        throw std::runtime_error("q2xs_nolut_dequant_blocks row is not divisible by block tile");
    }

    const uint32_t grid_x = (column + kMatmulBlockX - 1) / kMatmulBlockX;
    const uint32_t grid_y = row / (uint32_t) sub_row_per_block;
    if (grid_y == 0) {
        throw std::runtime_error("q2xs_nolut_dequant_blocks grid_y became zero");
    }

    threadsPerBlock.x = block_x;
    threadsPerBlock.y = block_y;
    threadsPerBlock.z = block_z;
    blocksPerGrid.x   = grid_x;
    blocksPerGrid.y   = grid_y;
    blocksPerGrid.z   = 1;
}

static void q3xxs_nolut_super_scale_blocks_matid(uint32_t K,
                                                 uint32_t super_group,
                                                 uint32_t group,
                                                 uint32_t N,
                                                 dim3 &   threadsPerBlock,
                                                 dim3 &   blocksPerGrid) {
    (void) group;
    const int nsg     = (int) (K / super_group);
    uint32_t  block_z = (uint32_t) nsg;
    uint32_t  block_x = N;
    uint32_t  cntX    = 1;
    while (block_x * block_z >= 8192) {
        block_x /= 2;
        cntX *= 2;
    }
    if (block_x * cntX * block_z != N * (uint32_t) nsg) {
        throw std::runtime_error("Matmul Q3XXS NoLUT TB Invalid");
    }

    threadsPerBlock.x = block_x;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = block_z;
    blocksPerGrid.x   = cntX;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;
}

static void q3xxs_nolut_dequant_blocks_matid(uint32_t loop,
                                             uint32_t row,
                                             uint32_t column,
                                             dim3 &   threadsPerBlock,
                                             dim3 &   blocksPerGrid,
                                             int      weights_group) {
    (void) loop;
    const uint32_t elements_per_thread = 32;
    const uint32_t block_x             = kMatmulBlockX;
    const uint32_t block_y             = (uint32_t) weights_group / elements_per_thread;
    uint32_t       block_z             = 1;
    while (true) {
        if ((block_x * block_y * block_z) >= (128 * kMatmulBlockX)) {
            break;
        }
        if ((block_y * block_z * elements_per_thread) >= row) {
            break;
        }
        block_z *= 2;
    }

    const uint32_t base_rows_per_block_z1 = block_y * elements_per_thread;
    while (block_z > 1 && (row % (base_rows_per_block_z1 * block_z)) != 0) {
        block_z >>= 1;
    }

    const int sub_row_per_block = (int) (block_y * block_z * elements_per_thread);
    if (row < block_y * block_z) {
        throw -1;
    }
    if ((row % (uint32_t) sub_row_per_block) != 0) {
        throw std::runtime_error("q3xxs_nolut_dequant_blocks row is not divisible by block tile");
    }

    threadsPerBlock.x = block_x;
    threadsPerBlock.y = block_y;
    threadsPerBlock.z = block_z;
    blocksPerGrid.x   = (column + kMatmulBlockX - 1) / kMatmulBlockX;
    blocksPerGrid.y   = row / (uint32_t) sub_row_per_block;
    blocksPerGrid.z   = 1;
}

static void dequant_blocks_matid_exact(uint32_t     loop,
                                       uint32_t     row,
                                       uint32_t     column,
                                       int          weights_group,
                                       uint32_t     elements_per_thread,
                                       const char * tag,
                                       dim3 &       threadsPerBlock,
                                       dim3 &       blocksPerGrid) {
    (void) loop;
    const uint32_t block_x = kMatmulBlockX;
    const uint32_t block_y = (uint32_t) weights_group / elements_per_thread;
    if (block_y == 0) {
        throw std::runtime_error(std::string(tag) + " invalid weights_group");
    }

    const uint32_t base_rows_per_block_z1 = block_y * elements_per_thread;
    if ((row % base_rows_per_block_z1) != 0) {
        throw std::runtime_error(std::string(tag) + " row is not divisible by base tile");
    }

    const uint32_t max_block_z_by_threads = (128u * kMatmulBlockX) / (block_x * block_y);
    const uint32_t max_block_z_by_rows    = row / base_rows_per_block_z1;
    uint32_t       block_z                = std::min(max_block_z_by_threads, max_block_z_by_rows);
    while (block_z > 1 && (row % (base_rows_per_block_z1 * block_z)) != 0) {
        --block_z;
    }

    const uint32_t sub_row_per_block = base_rows_per_block_z1 * block_z;
    if (sub_row_per_block == 0 || (row % sub_row_per_block) != 0) {
        throw std::runtime_error(std::string(tag) + " row is not divisible by block tile");
    }

    threadsPerBlock.x = block_x;
    threadsPerBlock.y = block_y;
    threadsPerBlock.z = block_z;
    blocksPerGrid.x   = (column + kMatmulBlockX - 1) / kMatmulBlockX;
    blocksPerGrid.y   = row / sub_row_per_block;
    blocksPerGrid.z   = 1;
}

#if defined(RPP_SIM_RT) || defined(WIN_X86_SIMDRV_ISS)
constexpr uint32_t kSramAddrMask = 0x1ffffffu;

inline bool matid_dump_chw32_enabled() {
    const char * env = std::getenv("RPP_MATID_DUMP_CHW32");
    return env != nullptr && std::atoi(env) != 0;
}

inline int matid_dump_limit(const char * name, int default_value) {
    const char * env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return default_value;
    }
    const int value = std::atoi(env);
    return value > 0 ? value : default_value;
}

inline uint32_t matid_sram_offset(uint32_t addr) {
    uint32_t off = addr & kSramAddrMask;
    if (off >= (uint32_t) RPP_SRAM_SIZE) {
        off %= (uint32_t) RPP_SRAM_SIZE;
    }
    return off;
}

void dump_chw_bf16_from_sram(const char * tag, uint32_t addr, int rows, int cols) {
    if (dmem == nullptr || rows <= 0 || cols <= 0) {
        return;
    }

    const int        max_rows = std::min(rows, matid_dump_limit("RPP_MATID_DUMP_MAX_ROWS", 4));
    const int        max_cols = std::min(cols, matid_dump_limit("RPP_MATID_DUMP_MAX_COLS", 64));
    const uint32_t   off      = matid_sram_offset(addr);
    const uint16_t * ptr      = reinterpret_cast<const uint16_t *>(dmem + off);

    std::fprintf(stderr, "[MATID CHW->HW32] %s addr=0x%08x sram_off=0x%08x rows=%d cols=%d dump_rows=%d dump_cols=%d\n",
                 tag, addr, off, rows, cols, max_rows, max_cols);

    for (int r = 0; r < max_rows; ++r) {
        std::fprintf(stderr, "  %s row=%d:", tag, r);
        for (int c = 0; c < max_cols; ++c) {
            const uint16_t raw = ptr[(size_t) r * (size_t) cols + (size_t) c];
            std::fprintf(stderr, " %04x/%7.3f", raw, (double) bf16_to_float(raw));
        }
        if (max_cols < cols) {
            std::fprintf(stderr, " ...");
        }
        std::fprintf(stderr, "\n");
    }
}

void dump_hw32_bf16_from_sram(const char * tag, uint32_t addr, int rows, int cols) {
    if (dmem == nullptr || rows <= 0 || cols <= 0) {
        return;
    }

    const int        col_blocks = (cols + 31) / 32;
    const int        max_rows   = std::min(rows, matid_dump_limit("RPP_MATID_DUMP_MAX_ROWS", 4));
    const int        max_blocks = std::min(col_blocks, matid_dump_limit("RPP_MATID_DUMP_MAX_BLOCKS", 4));
    const uint32_t   off        = matid_sram_offset(addr);
    const uint16_t * ptr        = reinterpret_cast<const uint16_t *>(dmem + off);

    std::fprintf(
        stderr,
        "[MATID CHW->HW32] %s addr=0x%08x sram_off=0x%08x rows=%d cols=%d blocks=%d dump_rows=%d dump_blocks=%d\n", tag,
        addr, off, rows, cols, col_blocks, max_rows, max_blocks);

    for (int blk = 0; blk < max_blocks; ++blk) {
        for (int r = 0; r < max_rows; ++r) {
            std::fprintf(stderr, "  %s blk=%d row=%d:", tag, blk, r);
            for (int lane = 0; lane < 32; ++lane) {
                const int      col = blk * 32 + lane;
                const uint16_t raw = ptr[((size_t) blk * (size_t) rows + (size_t) r) * 32u + (size_t) lane];
                if (col < cols) {
                    std::fprintf(stderr, " [%03d]=%04x/%7.3f", col, raw, (double) bf16_to_float(raw));
                } else {
                    std::fprintf(stderr, " [pad]=%04x/%7.3f", raw, (double) bf16_to_float(raw));
                }
            }
            std::fprintf(stderr, "\n");
        }
    }
}

void dump_chw_to_chw32_dyn_buffers(const std::vector<uint32_t> & params, int rows, int cols) {
    if (!matid_dump_chw32_enabled() || params.size() < 15) {
        return;
    }

    std::fprintf(stderr,
                 "[MATID CHW->HW32] params in=0x%08x out=0x%08x inStrideY=%u outStrideY=%u "
                 "inBXStride=%u outBXStride=%u inBYStride=%u outBYStride=%u gridX=%u rows=%d cols=%d\n",
                 params[0], params[1], params[2], params[4], params[6], params[7], params[8], params[9], params[14],
                 rows, cols);

    dump_chw_bf16_from_sram("input_chw", params[0], rows, cols);
    dump_hw32_bf16_from_sram("output_hw32", params[1], rows, cols);
}

void dump_hw32_to_chw_dyn_buffers(const std::vector<uint32_t> & params, int rows, int cols) {
    if (!matid_dump_chw32_enabled() || params.size() < 15) {
        return;
    }

    std::fprintf(stderr,
                 "[MATID HW32->CHW] params in=0x%08x out=0x%08x inStrideY=%u outStrideY=%u "
                 "inBXStride=%u outBXStride=%u inBYStride=%u outBYStride=%u gridX=%u rows=%d cols=%d\n",
                 params[0], params[1], params[2], params[4], params[6], params[7], params[8], params[9], params[14],
                 rows, cols);

    dump_hw32_bf16_from_sram("input_hw32", params[0], rows, cols);
    dump_chw_bf16_from_sram("output_chw", params[1], rows, cols);
}
#endif

struct quant_expert_weights_ddr {
    RPPdeviceptr codebook    = 0;
    RPPdeviceptr scales      = 0;
    RPPdeviceptr sign        = 0;
    RPPdeviceptr super_scale = 0;
};

struct fusion_device_inputs {
    RPPdeviceptr             sparse_act      = 0;
    RPPdeviceptr             token_ids       = 0;
    RPPdeviceptr             expert_counts   = 0;
    RPPdeviceptr             expert_offsets  = 0;
    RPPdeviceptr             slot_ids        = 0;
    RPPdeviceptr             routing_weights = 0;
    quant_expert_weights_ddr gate;
    quant_expert_weights_ddr up;
    quant_expert_weights_ddr down;
};

struct quant_weight_layout {
    uint32_t codebook_bytes      = 0;
    uint32_t scales_bytes        = 0;
    uint32_t sign_bytes          = 0;
    uint32_t super_scale_bytes   = 0;
    uint32_t scale_scratch_bytes = 0;
    uint32_t dequant_bytes       = 0;
};

struct quant_weight_sram_window {
    RPPdeviceptr codebook    = 0;
    RPPdeviceptr scales      = 0;
    RPPdeviceptr sign        = 0;
    RPPdeviceptr super_scale = 0;
};

struct quant_lut_sram_window {
    RPPdeviceptr qscale = 0;
    RPPdeviceptr mag    = 0;
};

struct fusion_sram_layout {
    RPPdeviceptr             sparse_input = 0;
    RPPdeviceptr             sparse_bf16  = 0;
    RPPdeviceptr             dense_bf16   = 0;
    RPPdeviceptr             dense_hw32   = 0;
    quant_weight_sram_window gate_weight{};
    quant_weight_sram_window up_weight{};
    quant_weight_sram_window down_weight{};
    quant_lut_sram_window    gate_lut{};
    quant_lut_sram_window    up_lut{};
    quant_lut_sram_window    down_lut{};
    RPPdeviceptr             silu_lut        = 0;
    RPPdeviceptr             scale_scratch   = 0;
    RPPdeviceptr             weight_bf16     = 0;
    RPPdeviceptr             matmul_hw32     = 0;
    RPPdeviceptr             gate_chw_bf16   = 0;
    RPPdeviceptr             up_chw_bf16     = 0;
    RPPdeviceptr             glu_chw_bf16    = 0;
    RPPdeviceptr             final_chw_bf16  = 0;
    RPPdeviceptr             merged_chw_bf16 = 0;
    RPPdeviceptr             final_chw_out   = 0;
    RPPdeviceptr             token_ids       = 0;
    RPPdeviceptr             slot_ids        = 0;
    RPPdeviceptr             routing_weights = 0;
    uint32_t                 total_bytes     = 0;
};

struct fusion_graph_bundle {
    RPPgraph     experts_input_graph    = 0;
    RPPgraphExec experts_input_exec     = 0;
    RPPgraph     single_expert_graph    = 0;
    RPPgraphExec single_expert_exec     = 0;
    RPPgraph     update_expert_id_graph = 0;
    RPPgraphExec update_expert_id_exec  = 0;
    RPPgraph     experts_output_graph   = 0;
    RPPgraphExec experts_output_exec    = 0;
};

struct fusion_runtime_plan {
    fusion_device_inputs inputs{};
    fusion_sram_layout   io{};
    quant_weight_layout  gate_weights{};
    quant_weight_layout  up_weights{};
    quant_weight_layout  down_weights{};
    fusion_graph_bundle  graphs{};
    int                  B                     = 0;
    int                  K                     = 0;
    int                  N                     = 0;
    int                  in_bytes_per_element  = 0;
    int                  out_bytes_per_element = 0;
    int                  mat0_quant            = 0;
    int                  mat1_quant            = 0;
    int                  mat2_quant            = 0;
    bool                 has_down_stage        = false;
    bool                 has_topk_merge        = false;
};

inline int round_up_sram(int bytes) {
    return (bytes + 511) / 512 * 512 + 512;
}

inline int round_up_multiple(int value, int factor) {
    return ((value + factor - 1) / factor) * factor;
}

inline bool is_power_of_two_u32(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

inline void append_u64_param(std::vector<uint32_t> & params, uint64_t value) {
    params.emplace_back((uint32_t) value);
    params.emplace_back((uint32_t) (value >> 32));
}

inline uint64_t matid_fram_reg_addr(uint64_t bank_base, uint32_t reg_index) {
    return bank_base + (uint64_t) reg_index * sizeof(uint32_t);
}

inline void validate_reformat_io(const rpp_kernel_context & ctx) {
    if (ctx.dev_in.size() < 2) {
        throw std::runtime_error(
            "matmul_id_fusion reformat requires ctx.dev_in[0]=sparse_act and ctx.dev_in[1]=token_ids");
    }
    if (ctx.dev_out.empty()) {
        throw std::runtime_error("matmul_id_fusion reformat requires ctx.dev_out[0]=dense_act");
    }
    if (ctx.virtual_sram_base == 0) {
        throw std::runtime_error("matmul_id_fusion reformat requires valid virtual_sram_base");
    }
}

inline void launch_cvt_f32_to_bf16(RPPdeviceptr sram_in,
                                   RPPdeviceptr sram_out,
                                   int          elements,
                                   RPPmodule    mod,
                                   RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;
    calc_tbdim_flattern(1, elements, threadsPerBlock, blocksPerGrid);
    cvt_kernel_param_init(threadsPerBlock, (uint32_t) sram_in, (uint32_t) sram_out, kFLOAT, kBF16, params);
    launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_cvt_bf16_to_f32(RPPdeviceptr sram_in,
                                   RPPdeviceptr sram_out,
                                   int          elements,
                                   RPPmodule    mod,
                                   RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;
    calc_tbdim_flattern(1, elements * 2, threadsPerBlock, blocksPerGrid);
    cvt_kernel_param_init_opt(threadsPerBlock, (uint32_t) sram_in, (uint32_t) sram_out, kBF16, kFLOAT, params);
    launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_fill_zero_bf16(RPPdeviceptr addr, int elements, RPPmodule mod, RPPstream stream) {
    if (elements <= 0) {
        return;
    }

    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;
    calc_tbdim_flattern(1, elements, threadsPerBlock, blocksPerGrid);
    fill_16bits_align_params((int) addr, (int) threadsPerBlock.x, 0, (int) sizeof(uint16_t), params);
    launchWrapperAysnc("fill_16bits_align", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_init_expert_id(uint32_t expert_id, RPPmodule mod, RPPstream stream) {
    dim3                  threadsPerBlock(32, 1, 1);
    dim3                  blocksPerGrid(1, 1, 1);
    std::vector<uint32_t> params;
    params.emplace_back(expert_id);
    launchWrapperAysnc("init_expert_id", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_dma_expert_params_to_fram(RPPdeviceptr expert_counts,
                                             RPPdeviceptr expert_offsets,
                                             RPPmodule    mod,
                                             RPPstream    stream) {
    dim3                  threadsPerBlock(32, 1, 1);
    dim3                  blocksPerGrid(1, 1, 1);
    std::vector<uint32_t> params;
    const uint64_t        count_fram0  = matid_fram_reg_addr(kMatidFramBank0Base, kMatidExpertCountRegIndex);
    const uint64_t        count_fram1  = matid_fram_reg_addr(kMatidFramBank1Base, kMatidExpertCountRegIndex);
    const uint64_t        offset_fram0 = matid_fram_reg_addr(kMatidFramBank0Base, kMatidExpertOffsetRegIndex);
    const uint64_t        offset_fram1 = matid_fram_reg_addr(kMatidFramBank1Base, kMatidExpertOffsetRegIndex);

    append_u64_param(params, (uint64_t) expert_counts);
    append_u64_param(params, (uint64_t) expert_offsets);
    params.emplace_back((uint32_t) count_fram0);
    params.emplace_back((uint32_t) count_fram1);
    params.emplace_back((uint32_t) offset_fram0);
    params.emplace_back((uint32_t) offset_fram1);
    params.emplace_back((uint32_t) (count_fram0 >> 32));
    launchWrapperAysnc("dma_expert_params_to_fram", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_update_next_expert_id(uint32_t id_stride, RPPmodule mod, RPPstream stream) {
    dim3                  threadsPerBlock(32, 1, 1);
    dim3                  blocksPerGrid(1, 1, 1);
    std::vector<uint32_t> params;
    params.emplace_back(id_stride);
    launchWrapperAysnc("update_next_expert_id", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void validate_gate_up_io(const rpp_kernel_context & ctx) {
    if (ctx.dev_in.size() != 12 && ctx.dev_in.size() != 16 && ctx.dev_in.size() != 18) {
        throw std::runtime_error(
            "matmul_id_fusion runtime path requires either "
            "ctx.dev_in[0]=sparse_act, [1]=sorted_token_ids, [2]=expert_counts, [3]=expert_offsets, "
            "[4..7]=gate_packed_weights, [8..11]=up_packed_weights "
            "or the same plus [12..15]=down_packed_weights "
            "or the same plus [12..15]=down_packed_weights, [16]=sorted_topk_slots, [17]=routing_weights_bf16_topk8");
    }
    if (ctx.dev_out.empty()) {
        throw std::runtime_error("matmul_id_fusion runtime path requires ctx.dev_out[0]=final_output");
    }
    if (ctx.virtual_sram_base == 0) {
        throw std::runtime_error("matmul_id_fusion runtime path requires valid virtual_sram_base");
    }
}

inline fusion_device_inputs read_gate_up_inputs(const rpp_kernel_context & ctx) {
    fusion_device_inputs in{};
    in.sparse_act       = ctx.dev_in[0];
    in.token_ids        = ctx.dev_in[1];
    in.expert_counts    = ctx.dev_in[2];
    in.expert_offsets   = ctx.dev_in[3];
    // Packed expert weights are passed as expert-major DDR tensors:
    // [expert][packed_per_expert_bytes] for codebook/scales/sign/super_scale.
    in.gate.codebook    = ctx.dev_in[4];
    in.gate.scales      = ctx.dev_in[5];
    in.gate.sign        = ctx.dev_in[6];
    in.gate.super_scale = ctx.dev_in[7];
    in.up.codebook      = ctx.dev_in[8];
    in.up.scales        = ctx.dev_in[9];
    in.up.sign          = ctx.dev_in[10];
    in.up.super_scale   = ctx.dev_in[11];
    if (ctx.dev_in.size() == 16 || ctx.dev_in.size() == 18) {
        in.down.codebook    = ctx.dev_in[12];
        in.down.scales      = ctx.dev_in[13];
        in.down.sign        = ctx.dev_in[14];
        in.down.super_scale = ctx.dev_in[15];
    }
    if (ctx.dev_in.size() == 18) {
        in.slot_ids        = ctx.dev_in[16];
        in.routing_weights = ctx.dev_in[17];
    }
    return in;
}

inline void validate_quant_kind(int quant, const char * stage) {
    if (quant < 0 || quant > 2) {
        throw std::runtime_error(std::string("matmul_id_fusion ") + stage +
                                 " only supports mat_quant in {0=q2s, 1=q2xs, 2=q3xxs}");
    }
}

inline uint32_t quant_qscale_lut_bytes(int quant) {
    switch (quant) {
        case 0:
            return kernel_q2_s_nolut::q2s_nolut_lut_workspace::qscale_lut_bytes;
        case 1:
            return kernel_q2_xs_nolut::q2xs_nolut_lut_workspace::qscale_lut_bytes;
        case 2:
            return kernel_q3_xxs_nolut::q3xxs_nolut_lut_workspace::qscale_lut_bytes;
        default:
            std::abort();
    }
}

inline uint32_t quant_mag_lut_bytes(int quant) {
    switch (quant) {
        case 0:
            return kernel_q2_s_nolut::q2s_nolut_lut_workspace::mag_lut_bytes;
        case 1:
            return kernel_q2_xs_nolut::q2xs_nolut_lut_workspace::mag_lut_bytes;
        case 2:
            return kernel_q3_xxs_nolut::q3xxs_nolut_lut_workspace::mag_lut_bytes;
        default:
            std::abort();
    }
}

inline quant_weight_layout make_quant_weight_layout(int quant, int input_cols, int output_cols, const char * stage) {
    validate_quant_kind(quant, stage);
    if ((input_cols % kQ2WeightsGroup) != 0) {
        throw std::runtime_error(std::string("matmul_id_fusion ") + stage + " expects input_cols % 256 == 0");
    }
    if ((output_cols % 128) != 0) {
        throw std::runtime_error(std::string("matmul_id_fusion ") + stage +
                                 " currently only supports tn4 output widths (output_cols % 128 == 0)");
    }

    quant_weight_layout out{};
    switch (quant) {
        case 0:
        case 1:
            out.codebook_bytes = (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols / 4ull);
            out.scales_bytes   = (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols / 32ull);
            out.sign_bytes     = (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols / 8ull);
            out.super_scale_bytes =
                (uint32_t) ((uint64_t) (input_cols / kQ2WeightsGroup) * (uint64_t) output_cols * sizeof(uint16_t));
            out.scale_scratch_bytes =
                (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols / 16ull * sizeof(uint16_t));
            out.dequant_bytes = (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols * sizeof(uint16_t));
            break;
        case 2:
            out.codebook_bytes =
                (uint32_t) (((uint64_t) input_cols / 16ull) * 3ull * (uint64_t) output_cols * sizeof(uint16_t));
            out.scales_bytes = (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols / 64ull);
            out.sign_bytes   = (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols / 8ull);
            out.super_scale_bytes =
                (uint32_t) ((uint64_t) (input_cols / kQ2WeightsGroup) * (uint64_t) output_cols * sizeof(uint16_t));
            out.scale_scratch_bytes =
                (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols / 32ull * sizeof(uint16_t));
            out.dequant_bytes = (uint32_t) ((uint64_t) input_cols * (uint64_t) output_cols * sizeof(uint16_t));
            break;
        default:
            std::abort();
    }
    return out;
}

inline quant_weight_layout max_weight_layout(const quant_weight_layout & lhs, const quant_weight_layout & rhs) {
    quant_weight_layout out{};
    out.codebook_bytes      = std::max(lhs.codebook_bytes, rhs.codebook_bytes);
    out.scales_bytes        = std::max(lhs.scales_bytes, rhs.scales_bytes);
    out.sign_bytes          = std::max(lhs.sign_bytes, rhs.sign_bytes);
    out.super_scale_bytes   = std::max(lhs.super_scale_bytes, rhs.super_scale_bytes);
    out.scale_scratch_bytes = std::max(lhs.scale_scratch_bytes, rhs.scale_scratch_bytes);
    out.dequant_bytes       = std::max(lhs.dequant_bytes, rhs.dequant_bytes);
    return out;
}

inline void destroy_fusion_graph_bundle(fusion_graph_bundle & graphs) {
    if (graphs.experts_input_exec != 0) {
        rppGraphExecDestroy(graphs.experts_input_exec);
        graphs.experts_input_exec = 0;
    }
    if (graphs.experts_input_graph != 0) {
        rppGraphDestroy(graphs.experts_input_graph);
        graphs.experts_input_graph = 0;
    }
    if (graphs.single_expert_exec != 0) {
        rppGraphExecDestroy(graphs.single_expert_exec);
        graphs.single_expert_exec = 0;
    }
    if (graphs.single_expert_graph != 0) {
        rppGraphDestroy(graphs.single_expert_graph);
        graphs.single_expert_graph = 0;
    }
    if (graphs.update_expert_id_exec != 0) {
        rppGraphExecDestroy(graphs.update_expert_id_exec);
        graphs.update_expert_id_exec = 0;
    }
    if (graphs.update_expert_id_graph != 0) {
        rppGraphDestroy(graphs.update_expert_id_graph);
        graphs.update_expert_id_graph = 0;
    }
    if (graphs.experts_output_exec != 0) {
        rppGraphExecDestroy(graphs.experts_output_exec);
        graphs.experts_output_exec = 0;
    }
    if (graphs.experts_output_graph != 0) {
        rppGraphDestroy(graphs.experts_output_graph);
        graphs.experts_output_graph = 0;
    }
}

inline fusion_sram_layout make_gate_up_sram_layout(const rpp_kernel_context &  ctx,
                                                   int                         B,
                                                   int                         K,
                                                   int                         N,
                                                   int                         total_routes,
                                                   int                         max_expert_tokens,
                                                   int                         in_bytes_per_element,
                                                   int                         out_bytes_per_element,
                                                   const quant_weight_layout & gate_weights,
                                                   const quant_weight_layout & up_weights,
                                                   int                         mat0_quant,
                                                   int                         mat1_quant,
                                                   bool                        has_down_stage,
                                                   bool                        has_topk_merge,
                                                   const quant_weight_layout & down_weights,
                                                   int                         mat2_quant) {
    constexpr uint32_t kSramLimitBytes = 22u * 1024u * 1024u;

    fusion_sram_layout  out{};
    RPPdeviceptr        cursor                 = ctx.virtual_sram_base;
    quant_weight_layout shared_scratch_weights = max_weight_layout(gate_weights, up_weights);
    if (has_down_stage) {
        shared_scratch_weights = max_weight_layout(shared_scratch_weights, down_weights);
    }
    const int scratch_rows = round_up_multiple(std::max(max_expert_tokens, 1), kDynamicMatmulBlockY);
    const int max_cols     = std::max(K, N);
    const int final_cols   = has_down_stage ? K : N;

    const uint32_t sparse_f32_bytes  = (uint32_t) ((uint64_t) B * (uint64_t) K * sizeof(float));
    const uint32_t sparse_bf16_bytes = (uint32_t) ((uint64_t) B * (uint64_t) K * sizeof(uint16_t));
    const uint32_t dense_bf16_bytes  = (uint32_t) ((uint64_t) scratch_rows * (uint64_t) K * sizeof(uint16_t));
    const uint32_t dense_hw32_bytes  = (uint32_t) ((uint64_t) scratch_rows * (uint64_t) max_cols * sizeof(uint16_t));
    const uint32_t matmul_bf16_bytes = (uint32_t) ((uint64_t) scratch_rows * (uint64_t) N * sizeof(uint16_t));
    const uint32_t matmul_hw32_bytes = dense_hw32_bytes;
    const uint32_t merged_bf16_bytes = (uint32_t) ((uint64_t) B * (uint64_t) final_cols * sizeof(uint16_t));
    const int      final_out_rows    = has_topk_merge ? B : std::max(max_expert_tokens, 1);
    const uint32_t final_out_bytes =
        (uint32_t) ((uint64_t) final_out_rows * (uint64_t) final_cols * (uint64_t) out_bytes_per_element);
    const uint32_t token_id_bytes        = (uint32_t) ((uint64_t) B * (uint64_t) kDefaultTopK * sizeof(uint16_t));
    const uint32_t routing_weights_bytes = token_id_bytes;
    const bool     can_alias_sparse_input_and_final_out = in_bytes_per_element == kFloatBytes &&
                                                      out_bytes_per_element == kFloatBytes &&
                                                      sparse_f32_bytes >= final_out_bytes;
    (void) total_routes;

    if (in_bytes_per_element == kFloatBytes) {
        out.sparse_input = cursor;
        cursor += round_up_sram((int) sparse_f32_bytes);
    }

    out.sparse_bf16 = cursor;
    cursor += round_up_sram((int) sparse_bf16_bytes);

    out.dense_bf16 = cursor;
    cursor += round_up_sram((int) dense_bf16_bytes);

    out.dense_hw32 = cursor;
    cursor += round_up_sram((int) dense_hw32_bytes);

    out.gate_weight.codebook = cursor;
    cursor += round_up_sram((int) gate_weights.codebook_bytes);

    out.gate_weight.scales = cursor;
    cursor += round_up_sram((int) gate_weights.scales_bytes);

    out.gate_weight.sign = cursor;
    cursor += round_up_sram((int) gate_weights.sign_bytes);

    out.gate_weight.super_scale = cursor;
    cursor += round_up_sram((int) gate_weights.super_scale_bytes);

    out.up_weight.codebook = cursor;
    cursor += round_up_sram((int) up_weights.codebook_bytes);

    out.up_weight.scales = cursor;
    cursor += round_up_sram((int) up_weights.scales_bytes);

    out.up_weight.sign = cursor;
    cursor += round_up_sram((int) up_weights.sign_bytes);

    out.up_weight.super_scale = cursor;
    cursor += round_up_sram((int) up_weights.super_scale_bytes);

    if (has_down_stage) {
        out.down_weight.codebook = cursor;
        cursor += round_up_sram((int) down_weights.codebook_bytes);

        out.down_weight.scales = cursor;
        cursor += round_up_sram((int) down_weights.scales_bytes);

        out.down_weight.sign = cursor;
        cursor += round_up_sram((int) down_weights.sign_bytes);

        out.down_weight.super_scale = cursor;
        cursor += round_up_sram((int) down_weights.super_scale_bytes);
    }

    out.gate_lut.qscale = cursor;
    cursor += round_up_sram((int) quant_qscale_lut_bytes(mat0_quant));
    out.gate_lut.mag = cursor;
    cursor += round_up_sram((int) quant_mag_lut_bytes(mat0_quant));

    out.up_lut.qscale = cursor;
    cursor += round_up_sram((int) quant_qscale_lut_bytes(mat1_quant));
    out.up_lut.mag = cursor;
    cursor += round_up_sram((int) quant_mag_lut_bytes(mat1_quant));

    if (has_down_stage) {
        out.down_lut.qscale = cursor;
        cursor += round_up_sram((int) quant_qscale_lut_bytes(mat2_quant));
        out.down_lut.mag = cursor;
        cursor += round_up_sram((int) quant_mag_lut_bytes(mat2_quant));
    }

    out.silu_lut = cursor;
    cursor += round_up_sram((int) kSiluLutBytes);

    out.scale_scratch = cursor;
    cursor += round_up_sram((int) shared_scratch_weights.scale_scratch_bytes);

    out.weight_bf16 = cursor;
    cursor += round_up_sram((int) shared_scratch_weights.dequant_bytes);

    out.matmul_hw32 = cursor;
    cursor += round_up_sram((int) matmul_hw32_bytes);

    out.gate_chw_bf16 = cursor;
    cursor += round_up_sram((int) matmul_bf16_bytes);

    out.up_chw_bf16 = cursor;
    cursor += round_up_sram((int) matmul_bf16_bytes);

    out.glu_chw_bf16 = cursor;
    cursor += round_up_sram((int) matmul_bf16_bytes);

    if (has_down_stage) {
        out.final_chw_bf16 = out.dense_bf16;
    } else {
        out.final_chw_bf16 = out.glu_chw_bf16;
    }

    if (has_topk_merge) {
        out.merged_chw_bf16 = cursor;
        cursor += round_up_sram((int) merged_bf16_bytes);
    } else {
        out.merged_chw_bf16 = out.final_chw_bf16;
    }

    if (out_bytes_per_element == kFloatBytes) {
        if (can_alias_sparse_input_and_final_out) {
            out.final_chw_out = out.sparse_input;
        } else {
            out.final_chw_out = cursor;
            cursor += round_up_sram((int) final_out_bytes);
        }
    } else {
        out.final_chw_out = has_topk_merge ? out.merged_chw_bf16 : out.final_chw_bf16;
    }

    out.token_ids = cursor;
    cursor += round_up_sram((int) token_id_bytes);

    if (has_topk_merge) {
        out.slot_ids = cursor;
        cursor += round_up_sram((int) token_id_bytes);

        out.routing_weights = cursor;
        cursor += round_up_sram((int) routing_weights_bytes);
    }

    out.total_bytes = (uint32_t) (cursor - ctx.virtual_sram_base);
    if (out.total_bytes > kSramLimitBytes) {
        throw std::runtime_error("matmul_id_fusion SRAM overflow");
    }

    return out;
}

inline void ensure_fusion_lut_workspace(rpp_kernel_context & ctx) {
    if (ctx.dev_workspace == 0) {
        rtMalloc((void **) &ctx.dev_workspace, kFusionLutWorkspaceBytes);
    }
}

static void q2xs_super_scale_params_matid(uint32_t                in_scale,
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
    (void) K;
    (void) super_group;
    (void) q_group;
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

static void q2_family_nolut_dequant_params_matid_exact(dim3 &                  blocksPerGrid,
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
    const int32_t grid_x     = (int32_t) blocksPerGrid.x;
    const int32_t grid_y     = (int32_t) blocksPerGrid.y;
    const int32_t block_x    = (int32_t) threadsPerBlock.x;
    const int32_t block_y    = (int32_t) threadsPerBlock.y;
    const int32_t block_z    = (int32_t) threadsPerBlock.z;
    const int32_t loop_count = kQ2FamilyFusionElementsPerThread / 16;
    (void) in_type_of_bytes;
    (void) out_type_of_bytes;

    const int inStrideY0 = loop_count * 2 * column;
    const int inStrideZ0 = block_y * inStrideY0;
    const int inStrideY1 = loop_count * column;
    const int inStrideZ1 = block_y * inStrideY1;

    const int in_unroll_stride = column * (int) sizeof(short);
    int       inBlockXSize0    = block_x * (int) sizeof(short);
    inBlockXSize0 -= (loop_count * 2) * in_unroll_stride;
    const int inBlockXSize1 = block_x * (int) sizeof(short);

    int inBlockYSize0 = block_y * block_z * inStrideY0 * (int) sizeof(short);
    inBlockYSize0 -= grid_x * block_x * (int) sizeof(short);
    int inBlockYSize1 = block_y * block_z * inStrideY1 * (int) sizeof(short);
    inBlockYSize1 -= grid_x * block_x * (int) sizeof(short);

    const int outStrideY        = kQ2FamilyFusionElementsPerThread * block_x;
    const int outStrideZ        = block_y * outStrideY;
    const int out_unroll_stride = block_x * (int) sizeof(short);
    int       outBlockXSize     = block_x * row * (int) sizeof(short);
    outBlockXSize -= kQ2FamilyFusionElementsPerThread * out_unroll_stride;
    int outBlockYSize = kQ2FamilyFusionElementsPerThread * block_x * block_y * block_z * (int) sizeof(short);
    outBlockYSize -= grid_x * block_x * row * (int) sizeof(short);

    const int dequantStrideY    = loop_count * column;
    const int dequantStrideZ    = block_y * dequantStrideY;
    int       deQuantBlockSizeX = block_x * (int) sizeof(short);
    deQuantBlockSizeX -= loop_count * in_unroll_stride;
    int deQuantBlockSizeY = dequantStrideY * block_z * block_y * (int) sizeof(short);
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

static void q2xs_nolut_dequant_params_matid(dim3 &                  blocksPerGrid,
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
    const int32_t grid_x  = (int32_t) blocksPerGrid.x;
    const int32_t grid_y  = (int32_t) blocksPerGrid.y;
    const int32_t grid_z  = (int32_t) blocksPerGrid.z;
    const int32_t block_x = (int32_t) threadsPerBlock.x;
    const int32_t block_y = (int32_t) threadsPerBlock.y;
    const int32_t block_z = (int32_t) threadsPerBlock.z;
    (void) grid_z;
    (void) in_type_of_bytes;
    (void) out_type_of_bytes;

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

    int outStrideY        = 4 * 16 * block_x;
    int outStrideZ        = 4 * outStrideY;
    int out_unroll_stride = block_x * sizeof(short);
    int outBlockXSize     = block_x * row * sizeof(short);
    outBlockXSize -= 64 * out_unroll_stride;
    int outBlockYSize = 64 * block_x * block_y * block_z * sizeof(short);
    outBlockYSize -= grid_x * block_x * row * sizeof(short);

    int dequantStrideY    = 4 * column;
    int dequantStrideZ    = 4 * dequantStrideY;
    int deQuantBlockSizeX = block_x * sizeof(short);
    deQuantBlockSizeX -= (4) * in_unroll_stride;
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

static void q3xxs_nolut_super_scale_params_matid(uint32_t                in_scale,
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
    (void) K;
    (void) super_group;
    (void) q_group;
    uint32_t inStrideY          = N;
    uint32_t inStrideZ          = 2 * N;
    uint32_t outStrideY         = N;
    uint32_t outStrideZ         = 8 * N;
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

static void q3xxs_nolut_dequant_params_matid(dim3 &                  blocksPerGrid,
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
    const int32_t grid_x  = (int32_t) blocksPerGrid.x;
    const int32_t grid_y  = (int32_t) blocksPerGrid.y;
    const int32_t grid_z  = (int32_t) blocksPerGrid.z;
    const int32_t block_x = (int32_t) threadsPerBlock.x;
    const int32_t block_y = (int32_t) threadsPerBlock.y;
    const int32_t block_z = (int32_t) threadsPerBlock.z;
    (void) grid_z;
    (void) in_type_of_bytes;
    (void) out_type_of_bytes;

    int inStrideY0 = 2 * 3 * column;
    int inStrideZ0 = 8 * inStrideY0;
    int inStrideY1 = 2 * column;
    int inStrideZ1 = 8 * inStrideY1;

    int in_unroll_stride = column * sizeof(short);
    int inBlockXSize0    = block_x * sizeof(short);
    inBlockXSize0 -= (2 * 3) * in_unroll_stride;
    int inBlockXSize1 = block_x * sizeof(short);
    inBlockXSize1 -= 2 * in_unroll_stride;

    int inBlockYSize0 = block_y * block_z * inStrideY0 * sizeof(short);
    inBlockYSize0 -= grid_x * block_x * sizeof(short);
    int inBlockYSize1 = block_y * block_z * inStrideY1 * sizeof(short);
    inBlockYSize1 -= grid_x * block_x * sizeof(short);

    int outStrideY        = 2 * 16 * block_x;
    int outStrideZ        = 8 * outStrideY;
    int out_unroll_stride = block_x * sizeof(short);
    int outBlockXSize     = block_x * row * sizeof(short);
    outBlockXSize -= 32 * out_unroll_stride;
    int outBlockYSize = 32 * block_x * block_y * block_z * sizeof(short);
    outBlockYSize -= grid_x * block_x * row * sizeof(short);

    int dequantStrideY    = column;
    int dequantStrideZ    = 8 * dequantStrideY;
    int deQuantBlockSizeX = block_x * sizeof(short);
    int deQuantBlockSizeY = column * block_z * block_y * sizeof(short);
    deQuantBlockSizeY -= grid_x * deQuantBlockSizeX;
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

inline std::vector<uint32_t> copy_device_u32_to_host(RPPdeviceptr dev_ptr, size_t count) {
    std::vector<uint32_t> host(count, 0);
    rtMemcpy(host.data(), (const void *) dev_ptr, count * sizeof(uint32_t), rtMemcpyDeviceToHost);
    return host;
}

inline void validate_packed_id_metadata(const std::vector<uint32_t> & expert_counts,
                                        const std::vector<uint32_t> & expert_offsets,
                                        int                           B,
                                        int                           nr_of_experts) {
    if ((int) expert_counts.size() != nr_of_experts || (int) expert_offsets.size() != nr_of_experts + 1) {
        throw std::runtime_error("matmul_id_fusion gate packed-id metadata size mismatch");
    }
    if (expert_offsets.empty() || expert_offsets[0] != 0) {
        throw std::runtime_error("matmul_id_fusion gate expert_offsets must start from 0");
    }

    const uint32_t max_routes = (uint32_t) (B * kDefaultTopK);
    if (expert_offsets.back() > max_routes) {
        throw std::runtime_error("matmul_id_fusion gate expert_offsets exceed B * topk");
    }

    for (int expert_id = 0; expert_id < nr_of_experts; ++expert_id) {
        if (expert_offsets[expert_id + 1] < expert_offsets[expert_id]) {
            throw std::runtime_error("matmul_id_fusion gate expert_offsets must be monotonic");
        }
        const uint32_t token_count = expert_offsets[expert_id + 1] - expert_offsets[expert_id];
        if (expert_counts[expert_id] != token_count) {
            throw std::runtime_error("matmul_id_fusion gate expert_counts do not match expert_offsets");
        }
    }
}

inline void copy_sparse_input_to_sram(RPPdeviceptr               sparse_act,
                                      const fusion_sram_layout & io,
                                      int                        B,
                                      int                        K,
                                      int                        in_bytes_per_element,
                                      RPPmodule                  mod,
                                      RPPstream                  stream) {
    const uint32_t sparse_f32_bytes  = (uint32_t) ((uint64_t) B * (uint64_t) K * sizeof(float));
    const uint32_t sparse_bf16_bytes = (uint32_t) ((uint64_t) B * (uint64_t) K * sizeof(uint16_t));
    (void) mod;

    if (in_bytes_per_element == kFloatBytes) {
        rtMemcpyAsync((void *) io.sparse_input, (const void *) sparse_act, sparse_f32_bytes, rtMemcpyDeviceToSram,
                      stream);
    } else {
        rtMemcpyAsync((void *) io.sparse_bf16, (const void *) sparse_act, sparse_bf16_bytes, rtMemcpyDeviceToSram,
                      stream);
    }
}

inline void copy_packed_route_inputs_to_sram(const fusion_device_inputs & inputs,
                                             const fusion_sram_layout &   io,
                                             int                          B,
                                             bool                         has_topk_merge,
                                             RPPstream                    stream) {
    const uint32_t token_id_bytes        = (uint32_t) ((uint64_t) B * (uint64_t) kDefaultTopK * sizeof(uint16_t));
    const uint32_t routing_weights_bytes = token_id_bytes;

    rtMemcpyAsync((void *) io.token_ids, (const void *) inputs.token_ids, token_id_bytes, rtMemcpyDeviceToSram, stream);

    if (has_topk_merge) {
        rtMemcpyAsync((void *) io.slot_ids, (const void *) inputs.slot_ids, token_id_bytes, rtMemcpyDeviceToSram,
                      stream);
        rtMemcpyAsync((void *) io.routing_weights, (const void *) inputs.routing_weights, routing_weights_bytes,
                      rtMemcpyDeviceToSram, stream);
    }
}

inline void copy_gate_up_inputs_to_sram(const fusion_device_inputs & inputs,
                                        const fusion_sram_layout &   io,
                                        int                          B,
                                        int                          K,
                                        int                          in_bytes_per_element,
                                        bool                         has_topk_merge,
                                        RPPmodule                    mod,
                                        RPPstream                    stream) {
    copy_sparse_input_to_sram(inputs.sparse_act, io, B, K, in_bytes_per_element, mod, stream);
    if (in_bytes_per_element == kFloatBytes) {
        launch_cvt_f32_to_bf16(io.sparse_input, io.sparse_bf16, B * K, mod, stream);
    }
    copy_packed_route_inputs_to_sram(inputs, io, B, has_topk_merge, stream);
}

inline RPPdeviceptr prepare_quant_lut_workspace(rpp_kernel_context & ctx, int quant) {
    ensure_fusion_lut_workspace(ctx);
    switch (quant) {
        case 0:
            return kernel_q2_s_nolut::q2s_nolut_prepare_lut_workspace(ctx);
        case 1:
            return kernel_q2_xs_nolut::q2xs_nolut_prepare_lut_workspace(ctx);
        case 2:
            return kernel_q3_xxs_nolut::q3xxs_nolut_prepare_lut_workspace(ctx);
        default:
            std::abort();
    }
}

inline void copy_quant_lut_to_sram(rpp_kernel_context &          ctx,
                                   const quant_lut_sram_window & lut,
                                   int                           quant,
                                   RPPstream &                   stream) {
    const RPPdeviceptr dev_lut_workspace = prepare_quant_lut_workspace(ctx, quant);
    const uint32_t     qscale_bytes      = quant_qscale_lut_bytes(quant);
    const uint32_t     mag_bytes         = quant_mag_lut_bytes(quant);
    rtMemcpyAsync((void *) lut.qscale, (const void *) dev_lut_workspace, qscale_bytes, rtMemcpyDeviceToSram, stream);
    rtMemcpyAsync((void *) lut.mag, (const void *) (dev_lut_workspace + qscale_bytes), mag_bytes, rtMemcpyDeviceToSram,
                  stream);
}

inline void copy_quant_lut_to_sram(RPPdeviceptr                  dev_lut_workspace,
                                   const quant_lut_sram_window & lut,
                                   int                           quant,
                                   RPPstream &                   stream) {
    const uint32_t qscale_bytes = quant_qscale_lut_bytes(quant);
    const uint32_t mag_bytes    = quant_mag_lut_bytes(quant);
    rtMemcpyAsync((void *) lut.qscale, (const void *) dev_lut_workspace, qscale_bytes, rtMemcpyDeviceToSram, stream);
    rtMemcpyAsync((void *) lut.mag, (const void *) (dev_lut_workspace + qscale_bytes), mag_bytes, rtMemcpyDeviceToSram,
                  stream);
}

inline void copy_silu_lut_to_sram(rpp_kernel_context & ctx, const fusion_sram_layout & io, RPPstream & stream) {
    ensure_fusion_lut_workspace(ctx);

    std::vector<uint16_t> silu_lut(kSiluLutBytes / (uint32_t) sizeof(uint16_t), 0);
    for (uint32_t i = 0; i < (uint32_t) silu_lut.size(); ++i) {
        const float x = bf16_to_float((uint16_t) i);
        const float y = x / (1.0f + std::exp(-x));
        silu_lut[i]   = float_to_bf16(y);
    }

    const RPPdeviceptr dev_silu_lut_workspace = ctx.dev_workspace + kQuantLutWorkspaceBytes;
    rtMemcpy((void *) dev_silu_lut_workspace, silu_lut.data(), kSiluLutBytes, rtMemcpyHostToDevice);
    rtMemcpyAsync((void *) io.silu_lut, (const void *) dev_silu_lut_workspace, kSiluLutBytes, rtMemcpyDeviceToSram,
                  stream);
}

inline void copy_silu_lut_to_sram(RPPdeviceptr dev_lut_workspace, const fusion_sram_layout & io, RPPstream & stream) {
    rtMemcpyAsync((void *) io.silu_lut, (const void *) dev_lut_workspace, kSiluLutBytes, rtMemcpyDeviceToSram, stream);
}

inline void launch_gather_expert_tokens(RPPdeviceptr sparse_bf16,
                                        RPPdeviceptr dense_bf16,
                                        RPPdeviceptr token_ids,
                                        int          K,
                                        RPPmodule    mod,
                                        RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    threadsPerBlock.x = (uint32_t) K;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = 1;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;

    params.emplace_back((uint32_t) sparse_bf16);
    params.emplace_back((uint32_t) dense_bf16);
    params.emplace_back((uint32_t) token_ids);
    params.emplace_back((uint32_t) (K * kBf16Bytes));
    launchWrapperAysnc("gather_expert_tokens", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_chw_to_chw32_dyn(RPPdeviceptr src_bf16,
                                    RPPdeviceptr dst_hw32,
                                    int          rows,
                                    int          cols,
                                    RPPmodule    mod,
                                    RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  launchBlocks{};
    std::vector<uint32_t> params;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    launchBlocks.x    = 1;
    launchBlocks.y    = 1;
    launchBlocks.z    = 1;

    chw2chw32_align_params((int) src_bf16, (int) dst_hw32, rows, cols, 0, 32, 1, 1, kBf16Bytes, params, false);
    params.emplace_back((uint32_t) ((cols + 31) / 32));
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt_dyn_v1", launchBlocks, threadsPerBlock, params, mod,
                       stream);

#if defined(RPP_SIM_RT) || defined(WIN_X86_SIMDRV_ISS)
    dump_chw_to_chw32_dyn_buffers(params, rows, cols);
#endif
}

inline void launch_chw32_to_chw_dyn(RPPdeviceptr src_hw32,
                                    RPPdeviceptr dst_bf16,
                                    int          rows,
                                    int          cols,
                                    RPPmodule    mod,
                                    RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  launchBlocks{};
    std::vector<uint32_t> params;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    launchBlocks.x    = 1;
    launchBlocks.y    = 1;
    launchBlocks.z    = 1;

    chw322chw_align_params((int) src_hw32, (int) dst_bf16, rows, cols, 0, 32, 1, 1, kBf16Bytes, params);
    params.emplace_back((uint32_t) ((cols + 31) / 32));
    launchWrapperAysnc("mem_copy_2d_align_f16_f16_all_xy_opt_dyn_v1", launchBlocks, threadsPerBlock, params, mod,
                       stream);

#if defined(RPP_SIM_RT) || defined(WIN_X86_SIMDRV_ISS)
    dump_hw32_to_chw_dyn_buffers(params, rows, cols);
#endif
}

inline void launch_q2s_super_scale(RPPdeviceptr sram_scales,
                                   RPPdeviceptr sram_super_scale,
                                   RPPdeviceptr sram_qscale_lut,
                                   RPPdeviceptr sram_scale_scratch,
                                   int          K,
                                   int          N,
                                   RPPmodule    mod,
                                   RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    kernel_q2_s_nolut::q2s_nolut_super_scale_blocks(K, kQ2WeightsGroup, kQGroup, N, threadsPerBlock, blocksPerGrid);
    kernel_q2_s_nolut::q2s_nolut_super_scale_params((uint32_t) sram_scales, (uint32_t) sram_super_scale,
                                                    (uint32_t) sram_qscale_lut, (uint32_t) sram_scale_scratch, K, N,
                                                    kQ2WeightsGroup, kQGroup, blocksPerGrid, threadsPerBlock, params);
    launchWrapperAysnc("matrix_mul_q2s_nolut_super_scale", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_q2s_dequant(RPPdeviceptr sram_codebook,
                               RPPdeviceptr sram_sign,
                               RPPdeviceptr sram_scale_scratch,
                               RPPdeviceptr sram_weight_bf16,
                               RPPdeviceptr sram_mag_lut,
                               int          K,
                               int          N,
                               RPPmodule    mod,
                               RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    const uint32_t row_tiles = (uint32_t) K / kQ2WeightsGroup;
    if (is_power_of_two_u32(row_tiles)) {
        kernel_q2_s_nolut::q2s_nolut_dequant_blocks(1, K, N, threadsPerBlock, blocksPerGrid, kQ2WeightsGroup);
        kernel_q2_s_nolut::q2s_nolut_dequant_params(blocksPerGrid, threadsPerBlock, (uint32_t) sram_codebook,
                                                    (uint32_t) sram_sign, (uint32_t) sram_scale_scratch,
                                                    (uint32_t) sram_weight_bf16, (uint32_t) sram_mag_lut, N, K,
                                                    (int) sizeof(uint16_t), (int) sizeof(uint16_t), params);
        launchWrapperAysnc("matrix_mul_q2s_nolut_asym_opt", blocksPerGrid, threadsPerBlock, params, mod, stream);
        return;
    }

    dequant_blocks_matid_exact(1, K, N, kQ2WeightsGroup, (uint32_t) kQ2FamilyFusionElementsPerThread,
                               "q2s_nolut_dequant_blocks_matid_exact", threadsPerBlock, blocksPerGrid);
    q2_family_nolut_dequant_params_matid_exact(blocksPerGrid, threadsPerBlock, (uint32_t) sram_codebook,
                                               (uint32_t) sram_sign, (uint32_t) sram_scale_scratch,
                                               (uint32_t) sram_weight_bf16, (uint32_t) sram_mag_lut, N, K,
                                               (int) sizeof(uint16_t), (int) sizeof(uint16_t), params);
    launchWrapperAysnc("matrix_mul_q2s_nolut_asym_opt_ept32", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_q2xs_super_scale(RPPdeviceptr sram_scales,
                                    RPPdeviceptr sram_super_scale,
                                    RPPdeviceptr sram_qscale_lut,
                                    RPPdeviceptr sram_scale_scratch,
                                    int          K,
                                    int          N,
                                    RPPmodule    mod,
                                    RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    q2xs_super_scale_blocks_matid(K, kQ2WeightsGroup, kQGroup, N, threadsPerBlock, blocksPerGrid);
    q2xs_super_scale_params_matid((uint32_t) sram_scales, (uint32_t) sram_super_scale, (uint32_t) sram_qscale_lut,
                                  (uint32_t) sram_scale_scratch, K, N, kQ2WeightsGroup, kQGroup, blocksPerGrid,
                                  threadsPerBlock, params);
    launchWrapperAysnc("matrix_mul_q2xs_super_scale", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_q2xs_dequant(RPPdeviceptr sram_codebook,
                                RPPdeviceptr sram_sign,
                                RPPdeviceptr sram_scale_scratch,
                                RPPdeviceptr sram_weight_bf16,
                                RPPdeviceptr sram_mag_lut,
                                int          K,
                                int          N,
                                RPPmodule    mod,
                                RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    const uint32_t row_tiles = (uint32_t) K / kQ2WeightsGroup;
    if (is_power_of_two_u32(row_tiles)) {
        q2xs_nolut_dequant_blocks_matid(1, K, N, threadsPerBlock, blocksPerGrid, kQ2WeightsGroup);
        q2xs_nolut_dequant_params_matid(blocksPerGrid, threadsPerBlock, (uint32_t) sram_codebook, (uint32_t) sram_sign,
                                        (uint32_t) sram_scale_scratch, (uint32_t) sram_weight_bf16,
                                        (uint32_t) sram_mag_lut, N, K, (int) sizeof(uint16_t), (int) sizeof(uint16_t),
                                        params);
        launchWrapperAysnc("matrix_mul_q2xs_nolut_deqaunt_f16", blocksPerGrid, threadsPerBlock, params, mod, stream);
        return;
    }

    dequant_blocks_matid_exact(1, K, N, kQ2WeightsGroup, (uint32_t) kQ2FamilyFusionElementsPerThread,
                               "q2xs_nolut_dequant_blocks_matid_exact", threadsPerBlock, blocksPerGrid);
    q2_family_nolut_dequant_params_matid_exact(blocksPerGrid, threadsPerBlock, (uint32_t) sram_codebook,
                                               (uint32_t) sram_sign, (uint32_t) sram_scale_scratch,
                                               (uint32_t) sram_weight_bf16, (uint32_t) sram_mag_lut, N, K,
                                               (int) sizeof(uint16_t), (int) sizeof(uint16_t), params);
    launchWrapperAysnc("matrix_mul_q2xs_nolut_deqaunt_f16_ept32", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_q3xxs_super_scale(RPPdeviceptr sram_scales,
                                     RPPdeviceptr sram_super_scale,
                                     RPPdeviceptr sram_qscale_lut,
                                     RPPdeviceptr sram_scale_scratch,
                                     int          K,
                                     int          N,
                                     RPPmodule    mod,
                                     RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    q3xxs_nolut_super_scale_blocks_matid(K, kQ2WeightsGroup, kQGroup, N, threadsPerBlock, blocksPerGrid);
    q3xxs_nolut_super_scale_params_matid((uint32_t) sram_scales, (uint32_t) sram_super_scale,
                                         (uint32_t) sram_qscale_lut, (uint32_t) sram_scale_scratch, K, N,
                                         kQ2WeightsGroup, kQGroup, blocksPerGrid, threadsPerBlock, params);
    launchWrapperAysnc("matrix_mul_q3xxs_super_scale", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_q3xxs_dequant(RPPdeviceptr sram_codebook,
                                 RPPdeviceptr sram_sign,
                                 RPPdeviceptr sram_scale_scratch,
                                 RPPdeviceptr sram_weight_bf16,
                                 RPPdeviceptr sram_mag_lut,
                                 int          K,
                                 int          N,
                                 RPPmodule    mod,
                                 RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    const uint32_t row_tiles = (uint32_t) K / kQ2WeightsGroup;
    if (is_power_of_two_u32(row_tiles)) {
        q3xxs_nolut_dequant_blocks_matid(1, K, N, threadsPerBlock, blocksPerGrid, kQ2WeightsGroup);
        q3xxs_nolut_dequant_params_matid(blocksPerGrid, threadsPerBlock, (uint32_t) sram_codebook, (uint32_t) sram_sign,
                                         (uint32_t) sram_scale_scratch, (uint32_t) sram_weight_bf16,
                                         (uint32_t) sram_mag_lut, N, K, (int) sizeof(uint16_t), (int) sizeof(uint16_t),
                                         params);
        launchWrapperAysnc("matrix_mul_q3xxs_deqaunt_nolut_f16", blocksPerGrid, threadsPerBlock, params, mod, stream);
        return;
    }

    dequant_blocks_matid_exact(1, K, N, kQ2WeightsGroup, 32u, "q3xxs_nolut_dequant_blocks_matid_exact", threadsPerBlock,
                               blocksPerGrid);
    q3xxs_nolut_dequant_params_matid(blocksPerGrid, threadsPerBlock, (uint32_t) sram_codebook, (uint32_t) sram_sign,
                                     (uint32_t) sram_scale_scratch, (uint32_t) sram_weight_bf16,
                                     (uint32_t) sram_mag_lut, N, K, (int) sizeof(uint16_t), (int) sizeof(uint16_t),
                                     params);
    launchWrapperAysnc("matrix_mul_q3xxs_deqaunt_nolut_f16", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void matmul_opt_kernel_params_matid_dyn(const dim3 &            blocksPerGrid,
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
    (void) postScale;
    (void) in0_row;

    const uint32_t in0_col_round = (in0_col + 31) / 32 * 32;
    uint32_t       in1_row_round = (in1_row + 31) / 32 * 32;
    if (in1_col == 32 && (in1_row % 32) > 0) {
        in1_row_round = in1_row;
    }
    const uint32_t in1_col_round = (in1_col + 31) / 32 * 32;

    int inStrideY  = block_x * in_type_of_bytes;
    int outStrideY = block_x;
    if (use_hw_output) {
        outStrideY = (int) in1_col;
    }

    // This dynamic kernel computes the real M-dependent address delta at runtime
    // from FRAM register cfg_fft_w4_6, so the host must pass the base row-one form.
    const int in0_row_one = 1;

    const int inStrideZ  = 0;
    const int outStrideZ = 0;
    const int cn         = (int) ((in0_col_round + 31) / 32) - 1;
    const int inSwitchSize =
        ((in0_row_one - 1) * block_x + (in_type_of_bytes == (int) sizeof(short) ? 8 : 16)) * in_type_of_bytes;

    input_b += in1_row_round * block_x * in_type_of_bytes * (uint32_t) tn_offset;
    out += in0_row_one * block_x * out_type_of_bytes * (uint32_t) tn_offset;

    const int gridx_inb_stride = in1_row_round * block_x * in_type_of_bytes * tn;
    const int gridx_out_stride =
        use_hw_output ? block_x * out_type_of_bytes * tn : in0_row_one * block_x * out_type_of_bytes * tn;
    const int gridy_ina_stride = block_x * block_y * in_type_of_bytes;
    const int gridy_out_stride = block_x * block_y * out_type_of_bytes;

    int       gridz_ina_stride = (loop_in0 == 1) ? 0 : (int) (in0_row_one * (int) in0_col_round * in_type_of_bytes);
    int       gridz_inb_stride = (loop_in1 == 1) ? 0 : (int) (in1_row_round * (int) in1_col_round * in_type_of_bytes);
    const int gridz_out_stride = (loop_out == 1) ? 0 : (int) (in0_row_one * (int) in1_col_round * out_type_of_bytes);
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
        const int outTnStride   = use_hw_output ? (int) (32 * sizeof(short)) : (int) (in0_row_one * 32 * sizeof(short));
        const int filterOffset0 = (int) (in0_col_round * 32 * sizeof(short));
        const int filterOffset1 = (int) (in0_col_round * 32 * sizeof(short) * (tn - 1) - 512);
        params.emplace_back((uint32_t) outTnStride);
        params.emplace_back((uint32_t) filterOffset0);
        params.emplace_back((uint32_t) filterOffset1);
    }
}

inline void launch_matmul_tn4_dyn(RPPdeviceptr act_hw32,
                                  RPPdeviceptr weight_bf16,
                                  RPPdeviceptr out_hw32,
                                  int          M,
                                  int          K,
                                  int          N,
                                  RPPmodule    mod,
                                  RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = (uint32_t) (N / 128);
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;

    matmul_opt_kernel_params_matid_dyn(blocksPerGrid, threadsPerBlock, (uint32_t) act_hw32, (uint32_t) weight_bf16, 0,
                                       (uint32_t) out_hw32, (uint32_t) M, (uint32_t) K, (uint32_t) K, (uint32_t) N, 1,
                                       1, 1, 0, kGateTn, kBf16Bytes, kBf16Bytes, params, false, false);

    launchWrapperAysnc("matrix_mul_tn4_f16_f16_dyn", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_matmul_tn4_dyn_v1(RPPdeviceptr act_hw32,
                                     RPPdeviceptr weight_bf16,
                                     RPPdeviceptr out_hw32,
                                     int          M,
                                     int          K,
                                     int          N,
                                     RPPmodule    mod,
                                     RPPstream    stream) {
    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = (uint32_t) kDynamicMatmulBlockY;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = (uint32_t) (N / 128);
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;

    matmul_opt_kernel_params_matid_dyn(blocksPerGrid, threadsPerBlock, (uint32_t) act_hw32, (uint32_t) weight_bf16, 0,
                                       (uint32_t) out_hw32, (uint32_t) M, (uint32_t) K, (uint32_t) K, (uint32_t) N, 1,
                                       1, 1, 0, kGateTn, kBf16Bytes, kBf16Bytes, params, false, false);

    launchWrapperAysnc("matrix_mul_tn4_f16_f16_dyn_v1", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_silu_mode1_bf16(RPPdeviceptr gate_bf16,
                                   RPPdeviceptr up_bf16,
                                   RPPdeviceptr silu_lut,
                                   RPPdeviceptr out_bf16,
                                   int          rows,
                                   int          cols,
                                   RPPmodule    mod,
                                   RPPstream    stream) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    if (cols <= 32 || cols >= 8192) {
        throw std::runtime_error("matmul_id_fusion silu dynamic kernel requires 32 < N < 8192");
    }

    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    threadsPerBlock.x = (uint32_t) cols;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = 1;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;

    // The dynamic GELU kernel consumes one row-width launch over N and
    // advances across M rows internally via cfg_fft_w4_6.
    params.emplace_back((uint32_t) gate_bf16);
    params.emplace_back((uint32_t) up_bf16);
    params.emplace_back((uint32_t) silu_lut);
    params.emplace_back((uint32_t) out_bf16);
    params.emplace_back((uint32_t) (cols * kBf16Bytes));
    params.emplace_back((uint32_t) 0);
    params.emplace_back((uint32_t) 1);
    launchWrapperAysnc("llm_gelu_dyn", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

inline void launch_moe_topk_combine_dyn(RPPdeviceptr packed_input_bf16,
                                        RPPdeviceptr routing_weights_bf16,
                                        RPPdeviceptr slot_ids,
                                        RPPdeviceptr token_ids,
                                        RPPdeviceptr merged_out_bf16,
                                        int          cols,
                                        RPPmodule    mod,
                                        RPPstream    stream) {
    if (cols <= 0) {
        return;
    }

    dim3                  threadsPerBlock{};
    dim3                  blocksPerGrid{};
    std::vector<uint32_t> params;

    threadsPerBlock.x = (uint32_t) cols;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;
    blocksPerGrid.x   = 1;
    blocksPerGrid.y   = 1;
    blocksPerGrid.z   = 1;

    params.emplace_back((uint32_t) packed_input_bf16);
    params.emplace_back((uint32_t) routing_weights_bf16);
    params.emplace_back((uint32_t) slot_ids);
    params.emplace_back((uint32_t) token_ids);
    params.emplace_back((uint32_t) merged_out_bf16);
    params.emplace_back((uint32_t) (cols * kBf16Bytes));
    params.emplace_back((uint32_t) kDefaultTopK);
    params.emplace_back((uint32_t) (cols * kBf16Bytes));
    launchWrapperAysnc("moe_topk_combine_dyn", blocksPerGrid, threadsPerBlock, params, mod, stream);
}

template <typename EnqueueCopy>
inline void append_quant_weights_for_expert_copy(const quant_expert_weights_ddr & expert_weights,
                                                 const quant_weight_sram_window & sram_weights,
                                                 const quant_weight_layout &      weights,
                                                 int                              expert_id,
                                                 EnqueueCopy &&                   enqueue_d2s_copy);

inline void copy_quant_weights_for_expert(const quant_expert_weights_ddr & expert_weights,
                                          const quant_weight_sram_window & sram_weights,
                                          const quant_weight_layout &      weights,
                                          int                              expert_id,
                                          RPPstream                        stream) {
    std::vector<RPPdeviceptr> src_ddr;
    std::vector<RPPdeviceptr> dst_sram;
    std::vector<size_t>       byte_count;

    auto enqueue_d2s_copy = [&](RPPdeviceptr src, RPPdeviceptr dst, size_t bytes) {
        src_ddr.push_back(src);
        dst_sram.push_back(dst);
        byte_count.push_back(bytes);
    };

    append_quant_weights_for_expert_copy(expert_weights, sram_weights, weights, expert_id, enqueue_d2s_copy);
    if (!dst_sram.empty()) {
        rppMemcpyLinkDtoSAsync(dst_sram.data(), src_ddr.data(), byte_count.data(), dst_sram.size(), stream);
    }
}

template <typename EnqueueCopy>
inline void append_quant_weights_for_expert_copy(const quant_expert_weights_ddr & expert_weights,
                                                 const quant_weight_sram_window & sram_weights,
                                                 const quant_weight_layout &      weights,
                                                 int                              expert_id,
                                                 EnqueueCopy &&                   enqueue_d2s_copy) {
    const bool is_interleaved_layout =
        (expert_weights.scales == expert_weights.codebook + (RPPdeviceptr) weights.codebook_bytes) &&
        (expert_weights.sign == expert_weights.scales + (RPPdeviceptr) weights.scales_bytes) &&
        (expert_weights.super_scale == expert_weights.sign + (RPPdeviceptr) weights.sign_bytes);

    if (is_interleaved_layout) {
        // interleaved layout (mode 2): [codebook_e][scales_e][sign_e][super_e] per expert.
        const uint64_t expert_stride = (uint64_t) weights.codebook_bytes + (uint64_t) weights.scales_bytes +
                                       (uint64_t) weights.sign_bytes + (uint64_t) weights.super_scale_bytes;
        const RPPdeviceptr expert_base =
            expert_weights.codebook + (RPPdeviceptr) ((uint64_t) expert_id * expert_stride);

        enqueue_d2s_copy(expert_base, sram_weights.codebook, weights.codebook_bytes);
        enqueue_d2s_copy(expert_base + (RPPdeviceptr) weights.codebook_bytes, sram_weights.scales,
                         weights.scales_bytes);
        enqueue_d2s_copy(
            expert_base + (RPPdeviceptr) ((uint64_t) weights.codebook_bytes + (uint64_t) weights.scales_bytes),
            sram_weights.sign, weights.sign_bytes);
        enqueue_d2s_copy(expert_base + (RPPdeviceptr) ((uint64_t) weights.codebook_bytes +
                                                       (uint64_t) weights.scales_bytes + (uint64_t) weights.sign_bytes),
                         sram_weights.super_scale, weights.super_scale_bytes);
        return;
    }

    const uint64_t expert_codebook_offset = (uint64_t) expert_id * (uint64_t) weights.codebook_bytes;
    const uint64_t expert_scales_offset   = (uint64_t) expert_id * (uint64_t) weights.scales_bytes;
    const uint64_t expert_sign_offset     = (uint64_t) expert_id * (uint64_t) weights.sign_bytes;
    const uint64_t expert_super_offset    = (uint64_t) expert_id * (uint64_t) weights.super_scale_bytes;

    // grouped layout (mode 1): [all_codebook][all_scales][all_sign][all_super]
    enqueue_d2s_copy(expert_weights.codebook + (RPPdeviceptr) expert_codebook_offset, sram_weights.codebook,
                     weights.codebook_bytes);
    enqueue_d2s_copy(expert_weights.scales + (RPPdeviceptr) expert_scales_offset, sram_weights.scales,
                     weights.scales_bytes);
    enqueue_d2s_copy(expert_weights.sign + (RPPdeviceptr) expert_sign_offset, sram_weights.sign, weights.sign_bytes);
    enqueue_d2s_copy(expert_weights.super_scale + (RPPdeviceptr) expert_super_offset, sram_weights.super_scale,
                     weights.super_scale_bytes);
}

inline void copy_expert_output_to_ddr(RPPdeviceptr dev_out,
                                      RPPdeviceptr sram_out,
                                      int          token_count,
                                      int          token_begin_offset,
                                      int          cols,
                                      int          out_bytes_per_element,
                                      RPPstream    stream) {
    const uint64_t bytes      = (uint64_t) token_count * (uint64_t) cols * (uint64_t) out_bytes_per_element;
    const uint64_t dst_offset = (uint64_t) token_begin_offset * (uint64_t) cols * (uint64_t) out_bytes_per_element;
    rtMemcpyAsync((void *) (dev_out + (RPPdeviceptr) dst_offset), (const void *) sram_out, (size_t) bytes,
                  rtMemcpySramToDevice, stream);
}

inline void run_quant_matmul_from_preloaded_sram(rpp_kernel_context &             ctx,
                                                 const quant_weight_sram_window & sram_weights,
                                                 const quant_lut_sram_window &    lut,
                                                 const fusion_sram_layout &       io,
                                                 const quant_weight_layout &      weights,
                                                 int                              quant,
                                                 RPPdeviceptr                     sram_act_hw32,
                                                 int                              token_count,
                                                 int                              K,
                                                 int                              N,
                                                 RPPdeviceptr                     sram_out_bf16) {
    (void) weights;
    switch (quant) {
        case 0:
            launch_q2s_super_scale(sram_weights.scales, sram_weights.super_scale, lut.qscale, io.scale_scratch, K, N,
                                   ctx.rppBinMod, ctx.kernelStream);
            launch_q2s_dequant(sram_weights.codebook, sram_weights.sign, io.scale_scratch, io.weight_bf16, lut.mag, K,
                               N, ctx.rppBinMod, ctx.kernelStream);
            break;
        case 1:
            launch_q2xs_super_scale(sram_weights.scales, sram_weights.super_scale, lut.qscale, io.scale_scratch, K, N,
                                    ctx.rppBinMod, ctx.kernelStream);
            launch_q2xs_dequant(sram_weights.codebook, sram_weights.sign, io.scale_scratch, io.weight_bf16, lut.mag, K,
                                N, ctx.rppBinMod, ctx.kernelStream);
            break;
        case 2:
            launch_q3xxs_super_scale(sram_weights.scales, sram_weights.super_scale, lut.qscale, io.scale_scratch, K, N,
                                     ctx.rppBinMod, ctx.kernelStream);
            launch_q3xxs_dequant(sram_weights.codebook, sram_weights.sign, io.scale_scratch, io.weight_bf16, lut.mag, K,
                                 N, ctx.rppBinMod, ctx.kernelStream);
            break;
        default:
            std::abort();
    }
    launch_matmul_tn4_dyn_v1(sram_act_hw32, io.weight_bf16, io.matmul_hw32, token_count, K, N, ctx.rppBinMod,
                             ctx.kernelStream);
    launch_chw32_to_chw_dyn(io.matmul_hw32, sram_out_bf16, token_count, N, ctx.rppBinMod, ctx.kernelStream);
}

inline void run_quant_matmul_to_sram(rpp_kernel_context &             ctx,
                                     const quant_expert_weights_ddr & expert_weights,
                                     const quant_weight_sram_window & sram_weights,
                                     const quant_lut_sram_window &    lut,
                                     const fusion_sram_layout &       io,
                                     const quant_weight_layout &      weights,
                                     int                              quant,
                                     RPPdeviceptr                     sram_act_hw32,
                                     int                              expert_id,
                                     int                              token_count,
                                     int                              K,
                                     int                              N,
                                     RPPdeviceptr                     sram_out_bf16) {
    copy_quant_weights_for_expert(expert_weights, sram_weights, weights, expert_id, ctx.kernelStream);
    run_quant_matmul_from_preloaded_sram(ctx, sram_weights, lut, io, weights, quant, sram_act_hw32, token_count, K, N,
                                         sram_out_bf16);
}

inline void run_gate_up_silu_expert_loop(rpp_kernel_context &          ctx,
                                         const fusion_device_inputs &  inputs,
                                         const fusion_sram_layout &    io,
                                         const quant_weight_layout &   gate_weights,
                                         const quant_weight_layout &   up_weights,
                                         const quant_weight_layout &   down_weights,
                                         int                           mat0_quant,
                                         int                           mat1_quant,
                                         int                           mat2_quant,
                                         bool                          has_down_stage,
                                         bool                          has_topk_merge,
                                         const std::vector<uint32_t> & expert_counts,
                                         const std::vector<uint32_t> & expert_offsets,
                                         int                           B,
                                         int                           K,
                                         int                           N,
                                         int                           out_bytes_per_element) {
    const RPPdeviceptr dev_final_out = ctx.dev_out[0];
    const int          final_cols    = has_down_stage ? K : N;

    if (has_topk_merge) {
        launch_fill_zero_bf16(io.merged_chw_bf16, B * final_cols, ctx.rppBinMod, ctx.kernelStream);
    }

    for (size_t expert_id = 0; expert_id < expert_counts.size(); ++expert_id) {
        const uint32_t token_count        = expert_counts[expert_id];
        const uint32_t token_begin_offset = expert_offsets[expert_id];
        if (token_count == 0) {
            continue;
        }

        rpp_update_expert_tokens(token_count, token_begin_offset);

        launch_gather_expert_tokens(io.sparse_bf16, io.dense_bf16, io.token_ids, K, ctx.rppBinMod, ctx.kernelStream);
        launch_chw_to_chw32_dyn(io.dense_bf16, io.dense_hw32, (int) token_count, K, ctx.rppBinMod, ctx.kernelStream);

        run_quant_matmul_to_sram(ctx, inputs.gate, io.gate_weight, io.gate_lut, io, gate_weights, mat0_quant,
                                 io.dense_hw32, (int) expert_id, (int) token_count, K, N, io.gate_chw_bf16);
        run_quant_matmul_to_sram(ctx, inputs.up, io.up_weight, io.up_lut, io, up_weights, mat1_quant, io.dense_hw32,
                                 (int) expert_id, (int) token_count, K, N, io.up_chw_bf16);

        launch_silu_mode1_bf16(io.gate_chw_bf16, io.up_chw_bf16, io.silu_lut, io.glu_chw_bf16, (int) token_count, N,
                               ctx.rppBinMod, ctx.kernelStream);

        RPPdeviceptr sram_final_bf16 = io.glu_chw_bf16;
        if (has_down_stage) {
            launch_chw_to_chw32_dyn(io.glu_chw_bf16, io.dense_hw32, (int) token_count, N, ctx.rppBinMod,
                                    ctx.kernelStream);
            run_quant_matmul_to_sram(ctx, inputs.down, io.down_weight, io.down_lut, io, down_weights, mat2_quant,
                                     io.dense_hw32, (int) expert_id, (int) token_count, N, K, io.final_chw_bf16);
            sram_final_bf16 = io.final_chw_bf16;
        }

        if (!has_topk_merge && out_bytes_per_element == kFloatBytes) {
            launch_cvt_bf16_to_f32(sram_final_bf16, io.final_chw_out, (int) token_count * final_cols, ctx.rppBinMod,
                                   ctx.kernelStream);
        }

        if (has_topk_merge) {
            launch_moe_topk_combine_dyn(sram_final_bf16, io.routing_weights, io.slot_ids, io.token_ids,
                                        io.merged_chw_bf16, final_cols, ctx.rppBinMod, ctx.kernelStream);
        } else {
            copy_expert_output_to_ddr(
                dev_final_out, out_bytes_per_element == kFloatBytes ? io.final_chw_out : sram_final_bf16,
                (int) token_count, (int) token_begin_offset, final_cols, out_bytes_per_element, ctx.kernelStream);
        }
    }

    if (has_topk_merge) {
        if (out_bytes_per_element == kFloatBytes) {
            launch_cvt_bf16_to_f32(io.merged_chw_bf16, io.final_chw_out, B * final_cols, ctx.rppBinMod,
                                   ctx.kernelStream);
        }

        const uint64_t bytes = (uint64_t) B * (uint64_t) final_cols * (uint64_t) out_bytes_per_element;
        rtMemcpyAsync((void *) dev_final_out,
                      (const void *) (out_bytes_per_element == kFloatBytes ? io.final_chw_out : io.merged_chw_bf16),
                      (size_t) bytes, rtMemcpySramToDevice, ctx.kernelStream);
    }
}

template <typename CaptureFn>
inline void capture_graph_on_stream(RPPstream      stream,
                                    RPPgraph &     graph,
                                    RPPgraphExec & graphexec,
                                    CaptureFn &&   capture_fn) {
    rppGraphCreate(&graph, 0);
    rppStreamBeginCapture(stream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    capture_fn();
    rppStreamEndCapture(stream, &graph);
    rppGraphInstantiate(&graphexec, graph, NULL, NULL, 0);
}

inline void capture_experts_input_graph(rpp_kernel_context &       ctx,
                                        const fusion_sram_layout & io,
                                        int                        B,
                                        int                        K,
                                        int                        in_bytes_per_element,
                                        bool                       has_topk_merge,
                                        int                        final_cols,
                                        fusion_graph_bundle &      graphs) {
    capture_graph_on_stream(ctx.kernelStream, graphs.experts_input_graph, graphs.experts_input_exec, [&]() {
        if (has_topk_merge) {
            launch_fill_zero_bf16(io.merged_chw_bf16, B * final_cols, ctx.rppBinMod, ctx.kernelStream);
        }
        if (in_bytes_per_element == kFloatBytes) {
            launch_cvt_f32_to_bf16(io.sparse_input, io.sparse_bf16, B * K, ctx.rppBinMod, ctx.kernelStream);
        }
        launch_init_expert_id(0, ctx.rppBinMod, ctx.kernelStream);
        launch_init_expert_id(0, ctx.rppBinMod, ctx.kernelStream);
    });
}

inline void capture_single_expert_graph(rpp_kernel_context &        ctx,
                                        RPPdeviceptr                expert_counts_phys,
                                        RPPdeviceptr                expert_offsets_phys,
                                        const fusion_sram_layout &  io,
                                        const quant_weight_layout & gate_weights,
                                        const quant_weight_layout & up_weights,
                                        const quant_weight_layout & down_weights,
                                        int                         mat0_quant,
                                        int                         mat1_quant,
                                        int                         mat2_quant,
                                        bool                        has_down_stage,
                                        bool                        has_topk_merge,
                                        int                         K,
                                        int                         N,
                                        fusion_graph_bundle &       graphs) {
    const int final_cols = has_down_stage ? K : N;
    capture_graph_on_stream(ctx.kernelStream, graphs.single_expert_graph, graphs.single_expert_exec, [&]() {
        launch_dma_expert_params_to_fram(expert_counts_phys, expert_offsets_phys, ctx.rppBinMod, ctx.kernelStream);
        launch_gather_expert_tokens(io.sparse_bf16, io.dense_bf16, io.token_ids, K, ctx.rppBinMod, ctx.kernelStream);
        launch_chw_to_chw32_dyn(io.dense_bf16, io.dense_hw32, 1, K, ctx.rppBinMod, ctx.kernelStream);

        run_quant_matmul_from_preloaded_sram(ctx, io.gate_weight, io.gate_lut, io, gate_weights, mat0_quant,
                                             io.dense_hw32, 1, K, N, io.gate_chw_bf16);
        run_quant_matmul_from_preloaded_sram(ctx, io.up_weight, io.up_lut, io, up_weights, mat1_quant, io.dense_hw32, 1,
                                             K, N, io.up_chw_bf16);

        launch_silu_mode1_bf16(io.gate_chw_bf16, io.up_chw_bf16, io.silu_lut, io.glu_chw_bf16, 1, N, ctx.rppBinMod,
                               ctx.kernelStream);

        RPPdeviceptr sram_final_bf16 = io.glu_chw_bf16;
        if (has_down_stage) {
            launch_chw_to_chw32_dyn(io.glu_chw_bf16, io.dense_hw32, 1, N, ctx.rppBinMod, ctx.kernelStream);
            run_quant_matmul_from_preloaded_sram(ctx, io.down_weight, io.down_lut, io, down_weights, mat2_quant,
                                                 io.dense_hw32, 1, N, K, io.final_chw_bf16);
            sram_final_bf16 = io.final_chw_bf16;
        }

        if (has_topk_merge) {
            launch_moe_topk_combine_dyn(sram_final_bf16, io.routing_weights, io.slot_ids, io.token_ids,
                                        io.merged_chw_bf16, final_cols, ctx.rppBinMod, ctx.kernelStream);
        }
        launch_update_next_expert_id(1, ctx.rppBinMod, ctx.kernelStream);
        launch_update_next_expert_id(1, ctx.rppBinMod, ctx.kernelStream);
    });
}

inline void capture_update_expert_id_graph(rpp_kernel_context & ctx, fusion_graph_bundle & graphs) {
    capture_graph_on_stream(ctx.kernelStream, graphs.update_expert_id_graph, graphs.update_expert_id_exec, [&]() {
        launch_update_next_expert_id(1, ctx.rppBinMod, ctx.kernelStream);
        launch_update_next_expert_id(1, ctx.rppBinMod, ctx.kernelStream);
    });
}

inline void capture_experts_output_graph(rpp_kernel_context &       ctx,
                                         const fusion_sram_layout & io,
                                         int                        elements,
                                         int                        out_bytes_per_element,
                                         bool                       has_topk_merge,
                                         bool                       has_down_stage,
                                         fusion_graph_bundle &      graphs) {
    if (out_bytes_per_element != kFloatBytes) {
        return;
    }

    const RPPdeviceptr src_bf16 =
        has_topk_merge ? io.merged_chw_bf16 : (has_down_stage ? io.final_chw_bf16 : io.glu_chw_bf16);
    capture_graph_on_stream(ctx.kernelStream, graphs.experts_output_graph, graphs.experts_output_exec, [&]() {
        launch_cvt_bf16_to_f32(src_bf16, io.final_chw_out, elements, ctx.rppBinMod, ctx.kernelStream);
    });
}

inline void preload_single_expert_weights(rpp_kernel_context &        ctx,
                                          const fusion_runtime_plan & plan,
                                          int                         expert_id,
                                          RPPstream &                 stream) {
    (void) ctx;
    std::vector<RPPdeviceptr> src_ddr;
    std::vector<RPPdeviceptr> dst_sram;
    std::vector<size_t>       byte_count;

    auto enqueue_d2s_copy = [&](RPPdeviceptr src, RPPdeviceptr dst, size_t bytes) {
        src_ddr.push_back(src);
        dst_sram.push_back(dst);
        byte_count.push_back(bytes);
    };

    append_quant_weights_for_expert_copy(plan.inputs.gate, plan.io.gate_weight, plan.gate_weights, expert_id,
                                         enqueue_d2s_copy);
    append_quant_weights_for_expert_copy(plan.inputs.up, plan.io.up_weight, plan.up_weights, expert_id,
                                         enqueue_d2s_copy);
    if (plan.has_down_stage) {
        append_quant_weights_for_expert_copy(plan.inputs.down, plan.io.down_weight, plan.down_weights, expert_id,
                                             enqueue_d2s_copy);
    }
    if (!dst_sram.empty()) {
        rppMemcpyLinkDtoSAsync(dst_sram.data(), src_ddr.data(), byte_count.data(), dst_sram.size(), stream);
    }
}

static void rpp_matmul_id_fusion_build(rpp_kernel_context &  ctx,
                                       fusion_runtime_plan & plan,
                                       int                   B,
                                       int                   K,
                                       int                   N,
                                       int                   nr_of_experts,
                                       int                   in_bytes_per_element,
                                       int                   out_bytes_per_element,
                                       int                   mat0_quant,
                                       int                   mat1_quant,
                                       int                   mat2_quant) {
    validate_gate_up_io(ctx);
    const bool has_down_stage = ctx.dev_in.size() == 16 || ctx.dev_in.size() == 18;
    const bool has_topk_merge = ctx.dev_in.size() == 18;
    if (B <= 0 || K <= 0 || N <= 0 || nr_of_experts <= 0) {
        throw std::runtime_error("matmul_id_fusion runtime path requires B/K/N/nr_of_experts > 0");
    }
    validate_quant_kind(mat0_quant, "gate");
    validate_quant_kind(mat1_quant, "up");
    if (has_down_stage) {
        validate_quant_kind(mat2_quant, "down");
    }
    if (in_bytes_per_element != kBf16Bytes && in_bytes_per_element != kFloatBytes) {
        throw std::runtime_error("matmul_id_fusion runtime path only supports BF16/F32 sparse input");
    }
    if (out_bytes_per_element != kBf16Bytes && out_bytes_per_element != kFloatBytes) {
        throw std::runtime_error("matmul_id_fusion runtime path only supports BF16/F32 outputs");
    }

    const fusion_device_inputs inputs              = read_gate_up_inputs(ctx);
    RPPdeviceptr               expert_counts_phys  = 0;
    RPPdeviceptr               expert_offsets_phys = 0;
    if (rppMemGetPhyAddr(&expert_counts_phys, inputs.expert_counts) != RPP_SUCCESS) {
        throw std::runtime_error("matmul_id_fusion failed to get physical address for expert_counts");
    }
    if (rppMemGetPhyAddr(&expert_offsets_phys, inputs.expert_offsets) != RPP_SUCCESS) {
        throw std::runtime_error("matmul_id_fusion failed to get physical address for expert_offsets");
    }
    const int                 total_routes = (int) nr_of_experts;
    const quant_weight_layout gate_weights = make_quant_weight_layout(mat0_quant, K, N, "gate");
    const quant_weight_layout up_weights   = make_quant_weight_layout(mat1_quant, K, N, "up");
    const quant_weight_layout down_weights =
        has_down_stage ? make_quant_weight_layout(mat2_quant, N, K, "down") : quant_weight_layout{};
    const fusion_sram_layout io = make_gate_up_sram_layout(
        ctx, B, K, N, total_routes, B, in_bytes_per_element, out_bytes_per_element, gate_weights, up_weights,
        mat0_quant, mat1_quant, has_down_stage, has_topk_merge, down_weights, mat2_quant);

    if (rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/matmul_id_fusion.o") != RPP_SUCCESS) {
        throw std::runtime_error("Failed to load matmul_id_fusion.o");
    }

    if (!has_topk_merge) {
        throw std::runtime_error(
            "matmul_id_fusion graph mode currently requires slot_ids + routing_weights merged-output path");
    }

    plan.inputs                = inputs;
    plan.io                    = io;
    plan.gate_weights          = gate_weights;
    plan.up_weights            = up_weights;
    plan.down_weights          = down_weights;
    plan.B                     = B;
    plan.K                     = K;
    plan.N                     = N;
    plan.in_bytes_per_element  = in_bytes_per_element;
    plan.out_bytes_per_element = out_bytes_per_element;
    plan.mat0_quant            = mat0_quant;
    plan.mat1_quant            = mat1_quant;
    plan.mat2_quant            = mat2_quant;
    plan.has_down_stage        = has_down_stage;
    plan.has_topk_merge        = has_topk_merge;

    const int final_cols = has_down_stage ? K : N;
    capture_experts_input_graph(ctx, io, B, K, in_bytes_per_element, has_topk_merge, final_cols, plan.graphs);
    capture_single_expert_graph(ctx, expert_counts_phys, expert_offsets_phys, io, gate_weights, up_weights,
                                down_weights, mat0_quant, mat1_quant, mat2_quant, has_down_stage, has_topk_merge, K, N,
                                plan.graphs);
    capture_experts_output_graph(ctx, io, B * final_cols, out_bytes_per_element, has_topk_merge, has_down_stage,
                                 plan.graphs);
}
}  // namespace kernel_expert_forward
