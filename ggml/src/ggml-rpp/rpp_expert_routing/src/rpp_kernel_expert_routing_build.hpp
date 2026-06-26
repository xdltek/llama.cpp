#ifndef RPP_EXPERT_ROUTING_H
#define RPP_EXPERT_ROUTING_H

#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"

#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

inline int round_len(const int ori_len) {
    return (ori_len + 63) / 64 * 64;
}

int inline popcount_fallback(unsigned int x) {
    int count = 0;
    while (x) {
        count += (int) (x & 1);
        x >>= 1;
    }
    return count;
}

int inline clz_fallback(unsigned int x) {
    if (x == 0) {
        return 32;
    }
    int count = 0;
    while ((x & 0x80000000) == 0) {
        count++;
        x <<= 1;
    }
    return count;
}

uint32_t inline CalcTopKInputRoundLen(uint32_t input_len) {
    uint32_t round_len = input_len;

    if (round_len < 4096) {
        round_len     = (round_len + 63) / 64 * 64;
        int one_count = popcount_fallback(round_len);
        if (one_count > 1) {
            round_len = (1 << (31 - clz_fallback(round_len) + 1));
        }
    } else {
        round_len = (round_len + 4095) / 4096 * 4096;
    }

    return round_len;
}

unsigned int inline dim3_volume(const dim3 & dim) {
    return dim.x * dim.y * dim.z;
}

void inline rt_task_to_rpp_config(const RppTaskElement &  rt_task,
                                  dim3 &                  thread_block,
                                  dim3 &                  grid_block,
                                  std::vector<uint32_t> & params) {
    thread_block.x = rt_task.blockDim.x;
    thread_block.y = rt_task.blockDim.y;
    thread_block.z = rt_task.blockDim.z;
    grid_block.x   = rt_task.gridDim.x;
    grid_block.y   = rt_task.gridDim.y;
    grid_block.z   = rt_task.gridDim.z;
    params.clear();
    for (uint32_t p : rt_task.params.kernelList) {
        params.emplace_back(p);
    }
}

void inline config_bc_x_op_block(const int num_rows, const int row_len, std::vector<dim3> & blocks) {
    if (num_rows * row_len < 8192) {
        dim3 block{};
        block.x = row_len;
        block.y = num_rows;
        blocks.emplace_back(block);
    } else {
        int sub_block_y = 0;
        for (int i = num_rows >= 8192 ? 8191 : num_rows - 1; i > 0; i--) {
            if (i * row_len < 8192 && i % row_len == 0) {
                sub_block_y = i;
                break;
            }
        }
        assert(sub_block_y != 0);
        int remain_rows = num_rows;
        while (remain_rows > 0) {
            dim3 block{};
            block.x = row_len;
            block.y = sub_block_y > remain_rows ? remain_rows : sub_block_y;
            blocks.emplace_back(block);
            remain_rows -= sub_block_y;
        }
    }

    for (auto & block : blocks) {
        while (dim3_volume(block) <= 32) {
            block.y += 1;
        }
    }
}

void inline config_topk_2d_block(const uint32_t column_len,
                                 const uint32_t row_len,
                                 uint16_t &     row_loop_num,
                                 uint16_t &     block_x,
                                 uint16_t &     trunk_by,
                                 uint16_t &     grid_x,
                                 uint16_t &     tail_by) {
    // Match the row-wise launch policy used by rpprt TopK 2D:
    // - large rows are split along X in 4096-wide chunks
    // - small rows batch multiple rows through block.y / block.z
    if (row_len >= 4096) {
        block_x      = 4096;
        trunk_by     = 1;
        grid_x       = column_len;
        tail_by      = 0;
        row_loop_num = row_len / 4096;
    } else {
        block_x         = row_len;
        uint16_t max_by = 1;
        while (block_x * max_by < 4096) {
            max_by++;
        }
        trunk_by     = max_by > column_len ? column_len : max_by;
        grid_x       = column_len / trunk_by;
        tail_by      = column_len % trunk_by;
        row_loop_num = 1;
    }
}

void inline rpp_expert_routing_build(rpp_kernel_context & ctx,
                                     const int            num_tokens,
                                     const int            max_num_experts,
                                     const int            num_experts,
                                     int                  in_bytes_per_element,
                                     int                  out_bytes_per_element,
                                     int                  is_instantial = 1) {
    if (in_bytes_per_element != 4 && in_bytes_per_element != 2) {
        throw std::runtime_error("Not supported input type.");
    }
    if (out_bytes_per_element != 4 && out_bytes_per_element != 2) {
        throw std::runtime_error("Not supported output type.");
    }
    if (max_num_experts < num_experts) {
        throw std::runtime_error("Invalid max_num_experts or num_experts.");
    }

    uint64_t lut_table_size = 65536 * sizeof(uint16_t);

    void * exp_table = malloc(lut_table_size);
    for (uint32_t i = 0; i < 65536; i++) {
        uint32_t x = i;
        x <<= 16;
        float y                     = exp(*(float *) &x);
        ((uint16_t *) exp_table)[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }

    void * reciprocal_table = malloc(lut_table_size);
    for (uint32_t i = 0; i < 65536; i++) {
        uint32_t x = i;
        x <<= 16;
        float y                            = 1.0f / *(float *) &x;
        ((uint16_t *) reciprocal_table)[i] = rpp::bfloat16::round_to_bfloat16(y).value;
    }
    // 0, -0
    ((uint16_t *) reciprocal_table)[0]      = 0;
    ((uint16_t *) reciprocal_table)[0x8000] = 0;

    if (rtMemcpy((void *) ctx.dev_workspace, exp_table, lut_table_size, rtMemcpyHostToDevice) != rtSuccess) {
        throw std::runtime_error("Failed to copy exp table into DDR.");
    }
    if (rtMemcpy((void *) (ctx.dev_workspace + lut_table_size), reciprocal_table, lut_table_size,
                 rtMemcpyHostToDevice) != rtSuccess) {
        throw std::runtime_error("Failed to copy reciprocal table into DDR.");
    }

    free(reciprocal_table);
    free(exp_table);

    uint64_t input_size_round = in_bytes_per_element == 4 ? sizeof(float) : sizeof(uint16_t);
    input_size_round *= round_len(num_tokens * max_num_experts);
    uint64_t indices_output_size      = round_len(num_tokens * max_num_experts) * sizeof(uint32_t);
    uint64_t bf16_scores_output_size  = round_len(num_tokens * num_experts) * sizeof(uint16_t);
    uint64_t float_scores_output_size = round_len(num_tokens * num_experts) * sizeof(float);

    RPPdeviceptr sram_exp_table_addr           = ctx.virtual_sram_base;
    RPPdeviceptr sram_reciprocal_table_addr    = sram_exp_table_addr + lut_table_size;
    RPPdeviceptr sram_input_addr               = sram_reciprocal_table_addr + lut_table_size;
    RPPdeviceptr sram_indices_output_addr      = sram_input_addr + input_size_round;
    RPPdeviceptr sram_bf16_scores_output_addr  = sram_indices_output_addr + indices_output_size;
    RPPdeviceptr sram_float_scores_output_addr = sram_bf16_scores_output_addr + bf16_scores_output_size;
    RPPdeviceptr sram_workspace_base           = sram_float_scores_output_addr + float_scores_output_size;

    if (rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL) != RPP_SUCCESS) {
        throw std::runtime_error("rppStreamBeginCapture failed.");
    }

    if (rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/expert_routing.o") != RPP_SUCCESS) {
        throw std::runtime_error("rppModuleLoad failed.");
    }

    if (rtMemcpyAsync((void *) sram_exp_table_addr, (void *) ctx.dev_workspace, lut_table_size * 2,
                      rtMemcpyDeviceToSram, ctx.kernelStream) != rtSuccess) {
        throw std::runtime_error("Failed to copy lookup table to SRAM.");
    }

    if (in_bytes_per_element == 4) {
        if (rtMemcpyAsync((void *) sram_input_addr, (void *) ctx.dev_in[0],
                          num_tokens * max_num_experts * sizeof(float), rtMemcpyDeviceToSram,
                          ctx.kernelStream) != rtSuccess) {
            throw std::runtime_error("Failed to copy input to SRAM.");
        }
        dim3 threadsPerBlock{}, blocksPerGrid{};
        calc_tbdim_flattern(num_tokens, max_num_experts, threadsPerBlock, blocksPerGrid);
        std::vector<uint32_t> params;
        cvt_kernel_param_init(threadsPerBlock, sram_input_addr, sram_input_addr, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
    } else {
        if (rtMemcpyAsync((void *) sram_input_addr, (void *) ctx.dev_in[0],
                          num_tokens * max_num_experts * sizeof(uint16_t), rtMemcpyDeviceToSram,
                          ctx.kernelStream) != rtSuccess) {
            throw std::runtime_error("Failed to copy input to SRAM.");
        }
    }

    // TopK treats input as [num_tokens, max_num_experts] and reduces on the
    // last axis, so each token is one row and experts are the row width.
    uint32_t input_row_len_           = max_num_experts;
    uint32_t input_row_round_len_     = CalcTopKInputRoundLen(input_row_len_);
    uint32_t input_column_len_        = num_tokens;
    uint32_t input_round_len_         = input_row_round_len_ * input_column_len_;
    uint32_t input_copy_size_         = input_round_len_ * sizeof(uint16_t);
    uint32_t input_indices_low_size_  = input_round_len_ * sizeof(uint16_t);
    uint32_t input_indices_high_size_ = input_indices_low_size_;
    uint32_t ping_pong_row_round_len_ = input_row_round_len_ > 4096 ? 4096 : input_row_round_len_;
    uint32_t data_ping_size_          = ping_pong_row_round_len_ * input_column_len_ * sizeof(uint16_t);
    uint32_t data_pong_size_          = data_ping_size_;
    uint32_t indices_low_ping_size_   = data_ping_size_;
    uint32_t indices_low_pong_size_   = data_ping_size_;
    uint32_t indices_high_ping_size_  = data_ping_size_;
    uint32_t indices_high_pong_size_  = data_ping_size_;
    uint32_t final_indices_low_size_  = round_len((int) input_column_len_) * sizeof(uint16_t);
    uint32_t final_indices_high_size_ = final_indices_low_size_;

    RPPdeviceptr       topk_input_copy_addr         = sram_workspace_base;
    RPPdeviceptr       topk_indices_low_addr        = topk_input_copy_addr + input_copy_size_;
    RPPdeviceptr       topk_indices_high_addr       = topk_indices_low_addr + input_indices_low_size_;
    RPPdeviceptr       topk_data_ping_addr          = topk_indices_high_addr + input_indices_high_size_;
    RPPdeviceptr       topk_data_pong_addr          = topk_data_ping_addr + data_ping_size_;
    RPPdeviceptr       topk_indices_low_ping_addr   = topk_data_pong_addr + data_pong_size_;
    RPPdeviceptr       topk_indices_low_pong_addr   = topk_indices_low_ping_addr + indices_low_ping_size_;
    RPPdeviceptr       topk_indices_high_ping_addr  = topk_indices_low_pong_addr + indices_low_pong_size_;
    RPPdeviceptr       topk_indices_high_pong_addr  = topk_indices_high_ping_addr + indices_high_ping_size_;
    RPPdeviceptr       topk_final_indices_low_addr  = topk_indices_high_pong_addr + indices_high_pong_size_;
    RPPdeviceptr       topk_final_indices_high_addr = topk_final_indices_low_addr + final_indices_low_size_;
    RPPdeviceptr       topk_final_data_addr         = topk_final_indices_high_addr + final_indices_high_size_;
    uint32_t           topk_data_ping_pong_addr[2] = { (uint32_t) topk_data_ping_addr, (uint32_t) topk_data_pong_addr };
    uint32_t           topk_indices_low_ping_pong_addr[2]  = { (uint32_t) topk_indices_low_ping_addr,
                                                               (uint32_t) topk_indices_low_pong_addr };
    uint32_t           topk_indices_high_ping_pong_addr[2] = { (uint32_t) topk_indices_high_ping_addr,
                                                               (uint32_t) topk_indices_high_pong_addr };
    // Pad with negative infinity in bf16 so padded lanes never win TopK(max).
    constexpr uint32_t topk_max_padding_value_mask         = 0xff80;

    dim3 topkInitIndicesThreadsPerBlock{}, topkInitIndicesBlocksPerGrid{};
    calc_tbdim_flattern(1, 2 * num_tokens * max_num_experts, topkInitIndicesThreadsPerBlock,
                        topkInitIndicesBlocksPerGrid);
    std::vector<uint32_t> topkInitIndicesParams;
    topkInitIndicesParams.emplace_back(sram_indices_output_addr);
    topkInitIndicesParams.emplace_back(dim3_volume(topkInitIndicesThreadsPerBlock) * sizeof(uint16_t));
    topkInitIndicesParams.emplace_back(0xffff);
    launchWrapperAysnc("fill_16bits_align", topkInitIndicesBlocksPerGrid, topkInitIndicesThreadsPerBlock,
                       topkInitIndicesParams, ctx.rppBinMod, ctx.kernelStream);

    uint16_t input_padding_bx = 0, input_padding_trunk_by = 0, input_padding_gx = 0;
    uint16_t input_padding_tail_by = 0, input_padding_row_loop_num = 0;
    config_topk_2d_block(input_column_len_, input_row_round_len_, input_padding_row_loop_num, input_padding_bx,
                         input_padding_trunk_by, input_padding_gx, input_padding_tail_by);

    if (input_row_len_ == input_row_round_len_) {
        dim3 copy_input_thread_block{}, copy_input_grid_block{};
        copy_input_thread_block.x = input_padding_bx;
        copy_input_thread_block.y = input_padding_trunk_by;
        copy_input_grid_block.x   = input_padding_row_loop_num;
        copy_input_grid_block.y   = input_padding_gx;
        std::vector<uint32_t> copy_input_params;
        copy_input_params.emplace_back(sram_input_addr);
        copy_input_params.emplace_back(topk_input_copy_addr);
        copy_input_params.emplace_back(dim3_volume(copy_input_thread_block));
        launchWrapperAysnc("mem_copy_align_f16_f16_all", copy_input_grid_block, copy_input_thread_block,
                           copy_input_params, ctx.rppBinMod, ctx.kernelStream);
        if (input_padding_tail_by != 0) {
            uint32_t io_trunk_offset =
                input_row_round_len_ * input_padding_trunk_by * input_padding_gx * sizeof(uint16_t);
            dim3 copy_input_tail_thread_block{}, copy_input_tail_grid_block{};
            copy_input_tail_thread_block.x = input_padding_bx;
            copy_input_tail_thread_block.y = input_padding_tail_by;
            copy_input_tail_grid_block.x   = 1;
            std::vector<uint32_t> copy_input_tail_params;
            copy_input_tail_params.emplace_back(sram_input_addr + io_trunk_offset);
            copy_input_tail_params.emplace_back(topk_input_copy_addr + io_trunk_offset);
            copy_input_tail_params.emplace_back(0);
            launchWrapperAysnc("mem_copy_align_f16_f16_all", copy_input_tail_grid_block, copy_input_tail_thread_block,
                               copy_input_tail_params, ctx.rppBinMod, ctx.kernelStream);
        }
    } else {
        dim3 padding_thread_block{}, padding_grid_block{};
        padding_thread_block.x = input_padding_bx;
        padding_thread_block.y = input_padding_trunk_by;
        padding_grid_block.x   = input_padding_gx;
        std::vector<uint32_t> padding_params;
        padding_params.emplace_back(sram_input_addr);
        padding_params.emplace_back(topk_input_copy_addr);
        padding_params.emplace_back(input_row_len_ * input_padding_trunk_by * sizeof(uint16_t));
        padding_params.emplace_back(input_row_round_len_ * input_padding_trunk_by * sizeof(uint16_t));
        padding_params.emplace_back(input_row_len_);
        padding_params.emplace_back(input_padding_row_loop_num);
        padding_params.emplace_back(topk_max_padding_value_mask);
        launchWrapperAysnc("topk_2d_padding", padding_grid_block, padding_thread_block, padding_params, ctx.rppBinMod,
                           ctx.kernelStream);
        if (input_padding_tail_by != 0) {
            uint32_t input_trunk_offset = input_row_len_ * input_padding_trunk_by * input_padding_gx * sizeof(uint16_t);
            uint32_t output_trunk_offset =
                input_row_round_len_ * input_padding_trunk_by * input_padding_gx * sizeof(uint16_t);
            dim3 padding_tail_thread_block{}, padding_tail_grid_block{};
            padding_tail_thread_block.x = input_padding_bx;
            padding_tail_thread_block.y = input_padding_tail_by;
            padding_tail_grid_block.x   = 1;
            std::vector<uint32_t> padding_tail_params;
            padding_tail_params.emplace_back(sram_input_addr + input_trunk_offset);
            padding_tail_params.emplace_back(topk_input_copy_addr + output_trunk_offset);
            padding_tail_params.emplace_back(0);
            padding_tail_params.emplace_back(0);
            padding_tail_params.emplace_back(input_row_len_);
            padding_tail_params.emplace_back(input_padding_row_loop_num);
            padding_tail_params.emplace_back(topk_max_padding_value_mask);
            launchWrapperAysnc("topk_2d_padding", padding_tail_grid_block, padding_tail_thread_block,
                               padding_tail_params, ctx.rppBinMod, ctx.kernelStream);
        }
    }

    dim3 init_indices_thread_block{}, init_indices_grid_block{};
    init_indices_thread_block.x = input_padding_bx;
    init_indices_thread_block.y = input_padding_trunk_by;
    init_indices_grid_block.x   = input_padding_gx;
    std::vector<uint32_t> init_indices_params;
    init_indices_params.emplace_back(topk_indices_low_addr);
    init_indices_params.emplace_back(topk_indices_high_addr);
    init_indices_params.emplace_back(input_row_round_len_ * input_padding_trunk_by * sizeof(uint16_t));
    init_indices_params.emplace_back(input_padding_row_loop_num);
    launchWrapperAysnc("topk_2d_init_indices", init_indices_grid_block, init_indices_thread_block, init_indices_params,
                       ctx.rppBinMod, ctx.kernelStream);
    if (input_padding_tail_by != 0) {
        uint32_t output_trunk_offset =
            input_row_round_len_ * input_padding_trunk_by * input_padding_gx * sizeof(uint16_t);
        dim3 init_indices_tail_thread_block{}, init_indices_tail_grid_block{};
        init_indices_tail_thread_block.x = input_padding_bx;
        init_indices_tail_thread_block.y = input_padding_tail_by;
        init_indices_tail_grid_block.x   = 1;
        std::vector<uint32_t> init_indices_tail_params;
        init_indices_tail_params.emplace_back(topk_indices_low_addr + output_trunk_offset);
        init_indices_tail_params.emplace_back(topk_indices_high_addr + output_trunk_offset);
        init_indices_tail_params.emplace_back(0);
        init_indices_tail_params.emplace_back(input_padding_row_loop_num);
        launchWrapperAysnc("topk_2d_init_indices", init_indices_tail_grid_block, init_indices_tail_thread_block,
                           init_indices_tail_params, ctx.rppBinMod, ctx.kernelStream);
    }

    for (int i = 0; i < num_experts; i++) {
        uint32_t current_row_len = input_row_round_len_;
        uint32_t target_row_len  = 0;
        for (int j = 0; current_row_len != 1; j++) {
            if (current_row_len > 4096) {
                target_row_len    = 4096;
                uint16_t loop_num = current_row_len / target_row_len;
                dim3     reduce_thread_block{}, reduce_grid_block{};
                reduce_thread_block.x = target_row_len;
                reduce_grid_block.x   = input_column_len_;
                std::vector<uint32_t> reduce_params;
                reduce_params.emplace_back(topk_input_copy_addr);
                reduce_params.emplace_back(topk_data_ping_pong_addr[(j + 1) % 2]);
                reduce_params.emplace_back(topk_indices_low_addr);
                reduce_params.emplace_back(topk_indices_low_ping_pong_addr[(j + 1) % 2]);
                reduce_params.emplace_back(topk_indices_high_addr);
                reduce_params.emplace_back(topk_indices_high_ping_pong_addr[(j + 1) % 2]);
                reduce_params.emplace_back(current_row_len * sizeof(uint16_t));
                reduce_params.emplace_back(dim3_volume(reduce_thread_block) * sizeof(uint16_t));
                reduce_params.emplace_back(current_row_len);
                reduce_params.emplace_back(target_row_len * sizeof(uint16_t));
                reduce_params.emplace_back(loop_num);
                launchWrapperAysnc("topk_2d_opt_reduce_max", reduce_grid_block, reduce_thread_block, reduce_params,
                                   ctx.rppBinMod, ctx.kernelStream);
            } else {
                if (current_row_len == 32 || (current_row_len == 64 && input_column_len_ == 1)) {
                    target_row_len = 1;
                } else {
                    target_row_len = current_row_len / 2;
                }

                uint16_t reduce_bx = 0, reduce_trunk_by = 0, reduce_trunk_gx = 0;
                uint16_t reduce_tail_by = 0, reduce_row_loop_num = 0;
                config_topk_2d_block(input_column_len_, target_row_len, reduce_row_loop_num, reduce_bx, reduce_trunk_by,
                                     reduce_trunk_gx, reduce_tail_by);
                (void) reduce_row_loop_num;

                uint16_t loop_num = current_row_len / target_row_len;
                if (target_row_len == 1) {
                    dim3 reduce_thread_block{}, reduce_grid_block{};
                    reduce_thread_block.x = 1;
                    reduce_thread_block.y = reduce_trunk_by < 33 ? 33 : reduce_trunk_by;
                    reduce_grid_block.x   = reduce_trunk_gx;
                    std::vector<uint32_t> reduce_params;
                    reduce_params.emplace_back(j == 0 ? topk_input_copy_addr : topk_data_ping_pong_addr[j % 2]);
                    reduce_params.emplace_back(topk_final_data_addr);
                    reduce_params.emplace_back(j == 0 ? topk_indices_low_addr : topk_indices_low_ping_pong_addr[j % 2]);
                    reduce_params.emplace_back(topk_final_indices_low_addr);
                    reduce_params.emplace_back(j == 0 ? topk_indices_high_addr :
                                                        topk_indices_high_ping_pong_addr[j % 2]);
                    reduce_params.emplace_back(topk_final_indices_high_addr);
                    reduce_params.emplace_back(current_row_len * reduce_trunk_by * sizeof(uint16_t));
                    reduce_params.emplace_back(reduce_trunk_by * sizeof(uint16_t));
                    reduce_params.emplace_back(current_row_len);
                    reduce_params.emplace_back(loop_num);
                    reduce_params.emplace_back(sizeof(uint16_t));
                    launchWrapperAysnc("topk_2d_gen_reduce_max", reduce_grid_block, reduce_thread_block, reduce_params,
                                       ctx.rppBinMod, ctx.kernelStream);
                    if (reduce_tail_by != 0) {
                        uint32_t input_offset  = current_row_len * reduce_trunk_by * reduce_trunk_gx * sizeof(uint16_t);
                        uint32_t output_offset = reduce_trunk_by * dim3_volume(reduce_grid_block) * sizeof(uint16_t);
                        dim3     reduce_tail_thread_block{}, reduce_tail_grid_block{};
                        reduce_tail_thread_block.x = 1;
                        reduce_tail_thread_block.y = reduce_tail_by < 33 ? 33 : reduce_tail_by;
                        reduce_tail_grid_block.x   = 1;
                        std::vector<uint32_t> reduce_tail_params;
                        reduce_tail_params.emplace_back(
                            (j == 0 ? topk_input_copy_addr : topk_data_ping_pong_addr[j % 2]) + input_offset);
                        reduce_tail_params.emplace_back(topk_final_data_addr + output_offset);
                        reduce_tail_params.emplace_back(
                            (j == 0 ? topk_indices_low_addr : topk_indices_low_ping_pong_addr[j % 2]) + input_offset);
                        reduce_tail_params.emplace_back(topk_final_indices_low_addr + output_offset);
                        reduce_tail_params.emplace_back(
                            (j == 0 ? topk_indices_high_addr : topk_indices_high_ping_pong_addr[j % 2]) + input_offset);
                        reduce_tail_params.emplace_back(topk_final_indices_high_addr + output_offset);
                        reduce_tail_params.emplace_back(0);
                        reduce_tail_params.emplace_back(0);
                        reduce_tail_params.emplace_back(current_row_len);
                        reduce_tail_params.emplace_back(loop_num);
                        reduce_tail_params.emplace_back(sizeof(uint16_t));
                        launchWrapperAysnc("topk_2d_gen_reduce_max", reduce_tail_grid_block, reduce_tail_thread_block,
                                           reduce_tail_params, ctx.rppBinMod, ctx.kernelStream);
                    }
                } else {
                    dim3 reduce_thread_block{}, reduce_grid_block{};
                    reduce_thread_block.x = reduce_bx;
                    reduce_thread_block.z = reduce_trunk_by;
                    reduce_grid_block.x   = reduce_trunk_gx;
                    std::vector<uint32_t> reduce_params;
                    reduce_params.emplace_back(j == 0 ? topk_input_copy_addr : topk_data_ping_pong_addr[j % 2]);
                    reduce_params.emplace_back(topk_data_ping_pong_addr[(j + 1) % 2]);
                    reduce_params.emplace_back(j == 0 ? topk_indices_low_addr : topk_indices_low_ping_pong_addr[j % 2]);
                    reduce_params.emplace_back(topk_indices_low_ping_pong_addr[(j + 1) % 2]);
                    reduce_params.emplace_back(j == 0 ? topk_indices_high_addr :
                                                        topk_indices_high_ping_pong_addr[j % 2]);
                    reduce_params.emplace_back(topk_indices_high_ping_pong_addr[(j + 1) % 2]);
                    reduce_params.emplace_back(current_row_len * reduce_trunk_by * sizeof(uint16_t));
                    reduce_params.emplace_back(dim3_volume(reduce_thread_block) * sizeof(uint16_t));
                    reduce_params.emplace_back(current_row_len);
                    reduce_params.emplace_back(target_row_len * sizeof(uint16_t));
                    reduce_params.emplace_back(loop_num);
                    launchWrapperAysnc("topk_2d_opt_reduce_max", reduce_grid_block, reduce_thread_block, reduce_params,
                                       ctx.rppBinMod, ctx.kernelStream);
                    if (reduce_tail_by != 0) {
                        uint32_t input_offset = current_row_len * reduce_trunk_by * reduce_trunk_gx * sizeof(uint16_t);
                        uint32_t output_offset =
                            dim3_volume(reduce_thread_block) * dim3_volume(reduce_grid_block) * sizeof(uint16_t);
                        dim3 reduce_tail_thread_block{}, reduce_tail_grid_block{};
                        reduce_tail_thread_block.x = reduce_bx;
                        reduce_tail_thread_block.z = reduce_tail_by;
                        reduce_tail_grid_block.x   = 1;
                        std::vector<uint32_t> reduce_tail_params;
                        reduce_tail_params.emplace_back(
                            (j == 0 ? topk_input_copy_addr : topk_data_ping_pong_addr[j % 2]) + input_offset);
                        reduce_tail_params.emplace_back(topk_data_ping_pong_addr[(j + 1) % 2] + output_offset);
                        reduce_tail_params.emplace_back(
                            (j == 0 ? topk_indices_low_addr : topk_indices_low_ping_pong_addr[j % 2]) + input_offset);
                        reduce_tail_params.emplace_back(topk_indices_low_ping_pong_addr[(j + 1) % 2] + output_offset);
                        reduce_tail_params.emplace_back(
                            (j == 0 ? topk_indices_high_addr : topk_indices_high_ping_pong_addr[j % 2]) + input_offset);
                        reduce_tail_params.emplace_back(topk_indices_high_ping_pong_addr[(j + 1) % 2] + output_offset);
                        reduce_tail_params.emplace_back(0);
                        reduce_tail_params.emplace_back(0);
                        reduce_tail_params.emplace_back(current_row_len);
                        reduce_tail_params.emplace_back(target_row_len * sizeof(uint16_t));
                        reduce_tail_params.emplace_back(loop_num);
                        launchWrapperAysnc("topk_2d_opt_reduce_max", reduce_tail_grid_block, reduce_tail_thread_block,
                                           reduce_tail_params, ctx.rppBinMod, ctx.kernelStream);
                    }
                }
            }

            current_row_len = target_row_len;
        }

        uint16_t finalize_bx = 0, finalize_trunk_by = 0, finalize_trunk_gx = 0;
        uint16_t finalize_tail_by = 0, finalize_row_loop_num = 0;
        config_topk_2d_block(input_column_len_, 1, finalize_row_loop_num, finalize_bx, finalize_trunk_by,
                             finalize_trunk_gx, finalize_tail_by);
        (void) finalize_bx;
        (void) finalize_row_loop_num;

        dim3 finalize_thread_block{}, finalize_grid_block{};
        finalize_thread_block.x = 1;
        finalize_thread_block.y = finalize_trunk_by < 33 ? 33 : finalize_trunk_by;
        finalize_grid_block.x   = finalize_trunk_gx;
        std::vector<uint32_t> finalize_params;
        finalize_params.emplace_back(topk_input_copy_addr);
        finalize_params.emplace_back(topk_final_data_addr);
        finalize_params.emplace_back(topk_final_indices_low_addr);
        finalize_params.emplace_back(topk_final_indices_high_addr);
        // Values stay compact as [num_tokens, num_experts] for softmax, while
        // indices keep expert_routing's legacy sparse layout [num_tokens, max_num_experts].
        finalize_params.emplace_back(sram_input_addr + i * sizeof(uint16_t));
        finalize_params.emplace_back(sram_indices_output_addr + i * sizeof(uint32_t));
        finalize_params.emplace_back(finalize_trunk_by * input_row_round_len_ * sizeof(uint16_t));
        finalize_params.emplace_back(finalize_trunk_by * sizeof(uint16_t));
        finalize_params.emplace_back(finalize_trunk_by * num_experts * sizeof(uint16_t));
        finalize_params.emplace_back(finalize_trunk_by * max_num_experts * sizeof(uint32_t));
        finalize_params.emplace_back(input_row_round_len_ * sizeof(uint16_t));
        finalize_params.emplace_back(num_experts * sizeof(uint16_t));
        finalize_params.emplace_back(max_num_experts * sizeof(uint32_t));
        finalize_params.emplace_back(topk_max_padding_value_mask);
        finalize_params.emplace_back(finalize_trunk_by);
        launchWrapperAysnc("topk_2d_finalize", finalize_grid_block, finalize_thread_block, finalize_params,
                           ctx.rppBinMod, ctx.kernelStream);
        if (finalize_tail_by != 0) {
            uint32_t ori_input_offset = finalize_trunk_by * finalize_trunk_gx * input_row_round_len_ * sizeof(uint16_t);
            uint32_t final_data_offset  = finalize_trunk_by * finalize_trunk_gx * sizeof(uint16_t);
            uint32_t out_value_offset   = finalize_trunk_by * finalize_trunk_gx * num_experts * sizeof(uint16_t);
            uint32_t out_indices_offset = finalize_trunk_by * finalize_trunk_gx * max_num_experts * sizeof(uint32_t);
            dim3     finalize_tail_thread_block{}, finalize_tail_grid_block{};
            finalize_tail_thread_block.x = 1;
            finalize_tail_thread_block.y = finalize_tail_by < 33 ? 33 : finalize_tail_by;
            finalize_tail_grid_block.x   = 1;
            std::vector<uint32_t> finalize_tail_params;
            finalize_tail_params.emplace_back(topk_input_copy_addr + ori_input_offset);
            finalize_tail_params.emplace_back(topk_final_data_addr + final_data_offset);
            finalize_tail_params.emplace_back(topk_final_indices_low_addr + final_data_offset);
            finalize_tail_params.emplace_back(topk_final_indices_high_addr + final_data_offset);
            finalize_tail_params.emplace_back(sram_input_addr + out_value_offset + i * sizeof(uint16_t));
            finalize_tail_params.emplace_back(sram_indices_output_addr + out_indices_offset + i * sizeof(uint32_t));
            finalize_tail_params.emplace_back(0);
            finalize_tail_params.emplace_back(0);
            finalize_tail_params.emplace_back(0);
            finalize_tail_params.emplace_back(0);
            finalize_tail_params.emplace_back(input_row_round_len_ * sizeof(uint16_t));
            finalize_tail_params.emplace_back(num_experts * sizeof(uint16_t));
            finalize_tail_params.emplace_back(max_num_experts * sizeof(uint32_t));
            finalize_tail_params.emplace_back(topk_max_padding_value_mask);
            finalize_tail_params.emplace_back(finalize_tail_by);
            launchWrapperAysnc("topk_2d_finalize", finalize_tail_grid_block, finalize_tail_thread_block,
                               finalize_tail_params, ctx.rppBinMod, ctx.kernelStream);
        }
    }

    // Softmax consumes the compact TopK values buffer written by finalize.
    if (num_experts == 1) {
        dim3 write_scores_thread_block{}, write_scores_grid_block{};
        calc_tbdim_flattern(num_tokens, num_experts, write_scores_thread_block, write_scores_grid_block);
        std::vector<uint32_t> write_scores_params;
        write_scores_params.emplace_back(sram_bf16_scores_output_addr);
        write_scores_params.emplace_back(dim3_volume(write_scores_thread_block) * sizeof(uint16_t));
        write_scores_params.emplace_back(0x3f80);
        launchWrapperAysnc("fill_16bits_align", write_scores_grid_block, write_scores_thread_block, write_scores_params,
                           ctx.rppBinMod, ctx.kernelStream);
    } else {
        RppDims reduce_in_dims{}, reduce_out_dims{};
        reduce_in_dims.nbDims  = 2;
        reduce_in_dims.d[0]    = num_tokens;
        reduce_in_dims.d[1]    = num_experts;
        reduce_out_dims.nbDims = 2;
        reduce_out_dims.d[0]   = num_tokens;
        reduce_out_dims.d[1]   = 1;
        std::vector<RppDims> sub_reduce_io_dims;
        RppDims              middle_dims{};
        sub_reduce_io_dims.emplace_back(reduce_in_dims);
        sub_reduce_io_dims.emplace_back(reduce_out_dims);
        int io_cursor = 0;
        while (ReduceSpawnIO(1, sub_reduce_io_dims[io_cursor], sub_reduce_io_dims[sub_reduce_io_dims.size() - 1],
                             middle_dims, true, false)) {
            sub_reduce_io_dims.insert(sub_reduce_io_dims.begin() + io_cursor + 1, middle_dims);
            io_cursor++;
        }
        uint32_t reduce_workspace_size = 64 * sizeof(uint16_t) * 2;
        if (sub_reduce_io_dims.size() > 2) {
            reduce_workspace_size = round_len(sub_reduce_io_dims[1].Length()) * sizeof(uint16_t) * 2;
        }
        uint32_t reduce_ping_pong_addr[2] = { (uint32_t) sram_workspace_base,
                                              (uint32_t) sram_workspace_base + reduce_workspace_size / 2 };

        for (size_t i = 0; i < sub_reduce_io_dims.size() - 1; i++) {
            RppDims  in_dims    = sub_reduce_io_dims[i];
            RppDims  out_dims   = sub_reduce_io_dims[i + 1];
            uint32_t input_addr = i == 0 ? sram_input_addr : reduce_ping_pong_addr[i % 2];
            uint32_t output_addr =
                i == sub_reduce_io_dims.size() - 2 ? sram_bf16_scores_output_addr : reduce_ping_pong_addr[(i + 1) % 2];
            std::vector<RppTaskElement> tasks =
                create_reduce_kernel_task(1, kMAX, input_addr, output_addr, in_dims, out_dims, 0, true, false);
            for (const RppTaskElement & task : tasks) {
                dim3                  reduce_thread_block{}, reduce_grid_block{};
                std::vector<uint32_t> reduce_params;
                rt_task_to_rpp_config(task, reduce_thread_block, reduce_grid_block, reduce_params);
                launchWrapperAysnc(task.taskName, reduce_grid_block, reduce_thread_block, reduce_params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }

        std::vector<dim3> bc_x_op_blocks;
        config_bc_x_op_block(num_tokens, num_experts, bc_x_op_blocks);
        unsigned int io_block_offset  = dim3_volume(bc_x_op_blocks[0]) * sizeof(uint16_t);
        unsigned int in1_block_offset = bc_x_op_blocks[0].y * sizeof(uint16_t);
        dim3         bc_x_op_grid_block{};

        for (size_t i = 0; i < bc_x_op_blocks.size(); i++) {
            std::vector<uint32_t> params;
            params.emplace_back(sram_input_addr + i * io_block_offset);
            params.emplace_back(sram_bf16_scores_output_addr + i * in1_block_offset);
            params.emplace_back(sram_input_addr + i * io_block_offset);
            params.emplace_back(0);
            params.emplace_back(0);
            params.emplace_back(4);
            launchWrapperAysnc("opt_binary_bc_x_f16", bc_x_op_grid_block, bc_x_op_blocks[i], params, ctx.rppBinMod,
                               ctx.kernelStream);
        }

        dim3 exp_thread_block{}, exp_grid_block{};
        calc_tbdim_flattern(1, num_tokens * num_experts, exp_thread_block, exp_grid_block);
        std::vector<uint32_t> exp_params;
        exp_params.emplace_back(sram_input_addr);
        exp_params.emplace_back(sram_input_addr);
        exp_params.emplace_back(sram_exp_table_addr);
        exp_params.emplace_back(dim3_volume(exp_thread_block) * sizeof(uint16_t));
        launchWrapperAysnc("mish_f16", exp_grid_block, exp_thread_block, exp_params, ctx.rppBinMod, ctx.kernelStream);

        for (size_t i = 0; i < sub_reduce_io_dims.size() - 1; i++) {
            RppDims  in_dims    = sub_reduce_io_dims[i];
            RppDims  out_dims   = sub_reduce_io_dims[i + 1];
            uint32_t input_addr = i == 0 ? sram_input_addr : reduce_ping_pong_addr[i % 2];
            uint32_t output_addr =
                i == sub_reduce_io_dims.size() - 2 ? sram_float_scores_output_addr : reduce_ping_pong_addr[(i + 1) % 2];
            std::vector<RppTaskElement> tasks =
                create_reduce_kernel_task(1, kSUM, input_addr, output_addr, in_dims, out_dims, 0, true, false);
            for (RppTaskElement & task : tasks) {
                if (i == sub_reduce_io_dims.size() - 2) {
                    std::string last_reduce_task_kernel_name = task.taskName;
                    task.taskName = last_reduce_task_kernel_name.replace(last_reduce_task_kernel_name.size() - 4, 1,
                                                                         "_reciprocal_");
                    task.params.kernelList.emplace_back(sram_reciprocal_table_addr);
                }
                dim3                  reduce_thread_block{}, reduce_grid_block{};
                std::vector<uint32_t> reduce_params;
                rt_task_to_rpp_config(task, reduce_thread_block, reduce_grid_block, reduce_params);
                launchWrapperAysnc(task.taskName, reduce_grid_block, reduce_thread_block, reduce_params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }
        }

        for (size_t i = 0; i < bc_x_op_blocks.size(); i++) {
            std::vector<uint32_t> params;
            params.emplace_back(sram_input_addr + i * io_block_offset);
            params.emplace_back(sram_float_scores_output_addr + i * in1_block_offset);
            params.emplace_back(sram_bf16_scores_output_addr + i * io_block_offset);
            params.emplace_back(0);
            params.emplace_back(0);
            params.emplace_back(1);
            launchWrapperAysnc("opt_binary_bc_x_f16", bc_x_op_grid_block, bc_x_op_blocks[i], params, ctx.rppBinMod,
                               ctx.kernelStream);
        }
    }

    if (rtMemcpyAsync((void *) ctx.dev_out[0], (void *) sram_indices_output_addr,
                      num_tokens * max_num_experts * sizeof(uint32_t), rtMemcpySramToDevice,
                      ctx.kernelStream) != rtSuccess) {
        throw std::runtime_error("Failed to copy indices to DDR.");
    }

    if (out_bytes_per_element == 4) {
        dim3 threadsPerBlock{}, blocksPerGrid{};
        calc_tbdim_flattern(1, 2 * num_tokens * num_experts, threadsPerBlock, blocksPerGrid);
        std::vector<uint32_t> params;
        cvt_kernel_param_init(threadsPerBlock, sram_bf16_scores_output_addr, sram_float_scores_output_addr, kBF16,
                              kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_v2", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                           ctx.kernelStream);
        if (rtMemcpyAsync((void *) ctx.dev_out[1], (void *) sram_float_scores_output_addr,
                          num_tokens * num_experts * sizeof(float), rtMemcpySramToDevice,
                          ctx.kernelStream) != rtSuccess) {
            throw std::runtime_error("Failed to copy scores to DDR.");
        }
    } else {
        if (rtMemcpyAsync((void *) ctx.dev_out[1], (void *) sram_bf16_scores_output_addr,
                          num_tokens * num_experts * sizeof(uint16_t), rtMemcpySramToDevice,
                          ctx.kernelStream) != rtSuccess) {
            throw std::runtime_error("Failed to copy scores to DDR.");
        }
    }

    if (rppStreamEndCapture(ctx.kernelStream, &ctx.graph) != RPP_SUCCESS) {
        throw std::runtime_error("rppStreamEndCapture failed.");
    }
    if (is_instantial) {
        if (rppGraphInstantiate(&ctx.graphexec, ctx.graph, nullptr, nullptr, 0) != RPP_SUCCESS) {
            throw std::runtime_error("rppGraphInstantiate failed.");
        }
    }
}

#endif
