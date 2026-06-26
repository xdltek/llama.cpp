#include "ggml-rpp/rpp_kernel_utils.h"

#include "bfloat16.h"
#include "ggml-impl.h"
#include "ggml-rpp/rpp_dfs.h"

#include <assert.h>
#include <bfloat16.h>
#include <math.h>
#include <rpp_drv_api.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace {
struct rpp_dfs_state {
    int previous_frequency = -1;
    int current_frequency  = -1;
};

std::mutex                             g_rpp_dfs_mutex;
std::unordered_map<int, rpp_dfs_state> g_rpp_dfs_states;

void maybe_set_kernel_dfs_frequency(const std::string & kernName, dim3 threadsPerBlock, RPPstream kernelStream) {
    const int total_threads =
        static_cast<int>(threadsPerBlock.x) * static_cast<int>(threadsPerBlock.y) * static_cast<int>(threadsPerBlock.z);
    const int current_frequency =
        Kernel2Frequency(kernName, total_threads, RPP_FREQ_LEVEL2, RPP_FREQ_LEVEL2, RPP_FREQ_LEVEL1);
    int        device      = -1;
    const auto get_dev_res = rtGetDevice(&device);
    assert(get_dev_res == rtSuccess);

    bool need_set           = false;
    int  previous_frequency = -1;
    {
        std::lock_guard<std::mutex> lock(g_rpp_dfs_mutex);
        auto &                      state = g_rpp_dfs_states[device];
        previous_frequency                = state.current_frequency;
        state.previous_frequency          = previous_frequency;
        state.current_frequency           = current_frequency;
        need_set                          = (state.current_frequency != state.previous_frequency);
    }

    if (need_set & GGML_RPP_USE_DFS) {
        //printf("new DFS device=%d, name=%s, prev=%d, curr=%d\n", device, kernName.c_str(), previous_frequency,
        //       current_frequency);
        const auto dfs_res = rppDFSSetFreqAsync((RPPDFSFreqLevel) current_frequency, kernelStream);
        assert(dfs_res == rtSuccess);
    } else if (need_set & GGML_RPP_USE_DFS_FLEXIBLE) {
        //printf("new DFS device=%d, name=%s, prev=%d, curr=%d\n", device, kernName.c_str(), previous_frequency,
        //       current_frequency);
        struct RPPDFSFlexibleFreq_st freq;
        freq.type          = RPP_FLEXIBLE_FREQ_TYPE0;
        freq.levelA        = (RPPDFSFreqLevel) current_frequency;
        freq.levelB        = RPP_FREQ_LEVEL1;
        freq.pre_time      = 0;
        freq.hold_time     = 5;
        const auto dfs_res = rppDFSSetFlexibleFreqAsync(freq, kernelStream);
        assert(dfs_res == rtSuccess);
    }
}
}  // namespace

void rpp_reset_dfs_state(RPPstream kernel_stream) {
    // DFS is tracked per device, not per stream. Keep the cached device
    // frequency across per-op stream creation/destruction.
    (void) kernel_stream;
}

inline void logKernelLaunch(const std::string &     kernelName,
                            dim3 &                  gridDim,
                            dim3 &                  blockDim,
                            dim3 &                  tailBlockDim,
                            std::vector<uint32_t> & params) {
    std::ostringstream oss;

    oss << "Kernel function name: " << kernelName << "; Grid dimensions: x: " << gridDim.x << ", y: " << gridDim.y
        << ", z: " << gridDim.z << "; Block dimensions: x: " << blockDim.x << ", y: " << blockDim.y
        << ", z: " << blockDim.z << "; Tail block dimensions: x: " << tailBlockDim.x << ", y: " << tailBlockDim.y
        << ", z: " << tailBlockDim.z << ".\nParameters: ";

    for (size_t i = 0; i < params.size(); ++i) {
        oss << params[i];
        if (i + 1 < params.size()) {
            oss << ", ";
        }
    }
    oss << ".\n";
    // GGML_LOG_DEBUG("%s: %s\n", __func__, oss.str().c_str());
}

static void launchKernelRaw(RPPfunction             hfunc,
                            const std::string &     kernName,
                            dim3                    blocksPerGrid,
                            dim3                    threadsPerBlock,
                            dim3                    threadsPerBlockTail,
                            std::vector<uint32_t> & kparams,
                            RPPstream               kernelStream) {
    void *      params[64];
    tRppGridDim gridDim;
    tRppTbDim   tbDim;

    gridDim.X = blocksPerGrid.x;
    gridDim.Y = blocksPerGrid.y;
    gridDim.Z = blocksPerGrid.z;
    tbDim.X   = threadsPerBlock.x;
    tbDim.Y   = threadsPerBlock.y;
    tbDim.Z   = threadsPerBlock.z;
    for (int cnt = 0; cnt < kparams.size(); cnt++) {
        params[cnt] = &kparams[cnt];
    }
    void * extra_params[3] = { nullptr };
    extra_params[0]        = (void *) &threadsPerBlockTail.x;
    extra_params[1]        = (void *) &threadsPerBlockTail.y;
    extra_params[2]        = (void *) &threadsPerBlockTail.z;

    logKernelLaunch(kernName, blocksPerGrid, threadsPerBlock, threadsPerBlockTail, kparams);
    maybe_set_kernel_dfs_frequency(kernName, threadsPerBlock, kernelStream);

    rppLaunchKernel(hfunc, gridDim.X, gridDim.Y, gridDim.Z, tbDim.X, tbDim.Y, tbDim.Z, 0, kernelStream, params,
                    extra_params);
}

bool tryCustomKernelLaunch(const std::string &     kernName,
                           dim3                    blocksPerGrid,
                           dim3                    threadsPerBlock,
                           dim3                    threadsPerBlockTail,
                           std::vector<uint32_t> & kparams,
                           RPPmodule               cuMod,
                           RPPstream               kernelStream) {
    (void) threadsPerBlockTail;

    if (kernName != "opt_vector_cvt_f16_f32_opt" || blocksPerGrid.x <= 1 || (threadsPerBlock.x % 32) == 0) {
        return false;
    }

    assert(kparams.size() >= 4);

    const RppDataType inDataType   = (RppDataType) kparams[2];
    const RppDataType outDataType  = (RppDataType) kparams[3];
    const uint32_t    inAddr       = kparams[0];
    const uint32_t    outAddr      = kparams[1];
    const uint32_t    inElemSize   = (uint32_t) GetRppElementSize(inDataType);
    const uint32_t    outElemSize  = (uint32_t) GetRppElementSize(outDataType);
    const uint32_t    totalInBytes = threadsPerBlock.x * blocksPerGrid.x;
    const uint32_t    normalBlockX = (threadsPerBlock.x / 32) * 32;

    assert(normalBlockX > 0);

    RPPresult   res;
    RPPfunction hfunc;
    res = rppModuleGetFunction(&hfunc, cuMod, kernName.c_str());
    assert(res == RPP_SUCCESS);

    const uint32_t normalGridX   = totalInBytes / normalBlockX;
    const uint32_t tailInBytes   = totalInBytes % normalBlockX;
    const uint32_t normalInBytes = normalGridX * normalBlockX;

    if (normalGridX > 0) {
        dim3                  normalBlocks(normalGridX, blocksPerGrid.y, blocksPerGrid.z);
        dim3                  normalThreads(normalBlockX, threadsPerBlock.y, threadsPerBlock.z);
        std::vector<uint32_t> normalParams;
        cvt_kernel_param_init_opt(normalThreads, inAddr, outAddr, inDataType, outDataType, normalParams);
        launchKernelRaw(hfunc, kernName, normalBlocks, normalThreads, normalThreads, normalParams, kernelStream);
    }

    if (tailInBytes > 0) {
        dim3                  tailBlocks(1, blocksPerGrid.y, blocksPerGrid.z);
        dim3                  tailThreads(tailInBytes, threadsPerBlock.y, threadsPerBlock.z);
        std::vector<uint32_t> tailParams;
        cvt_kernel_param_init_opt(tailThreads, inAddr + normalInBytes,
                                  outAddr + (normalInBytes / inElemSize) * outElemSize, inDataType, outDataType,
                                  tailParams);
        launchKernelRaw(hfunc, kernName, tailBlocks, tailThreads, tailThreads, tailParams, kernelStream);
    }

    return true;
}

void adjustKernelLaunchConfig(const std::string &     kernName,
                              dim3 &                  blocksPerGrid,
                              dim3 &                  threadsPerBlock,
                              dim3 &                  threadsPerBlockTail,
                              std::vector<uint32_t> & kparams) {
    (void) kernName;
    (void) blocksPerGrid;
    (void) threadsPerBlock;
    (void) threadsPerBlockTail;
    (void) kparams;
}

void launchWrapperAysnc(string                  kernName,
                        dim3                    blocksPerGrid,
                        dim3                    threadsPerBlock,
                        std::vector<uint32_t> & kparams,
                        RPPmodule               cuMod,
                        RPPstream               kernelStream) {
    RPPresult   res;
    RPPfunction hfunc;
    dim3        threadsPerBlockTail = threadsPerBlock;
    assert(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z < 8192);
    if (tryCustomKernelLaunch(kernName, blocksPerGrid, threadsPerBlock, threadsPerBlockTail, kparams, cuMod,
                              kernelStream)) {
        return;
    }

    adjustKernelLaunchConfig(kernName, blocksPerGrid, threadsPerBlock, threadsPerBlockTail, kparams);

    res = rppModuleGetFunction(&hfunc, cuMod, kernName.c_str());
    assert(res == RPP_SUCCESS);
    launchKernelRaw(hfunc, kernName, blocksPerGrid, threadsPerBlock, threadsPerBlockTail, kparams, kernelStream);
}

void launchWrapperAysnc(string                  kernName,
                        dim3                    blocksPerGrid,
                        dim3                    threadsPerBlock,
                        dim3                    threadsPerBlockTail,
                        std::vector<uint32_t> & kparams,
                        RPPmodule               cuMod,
                        RPPstream               kernelStream) {
    RPPresult   res;
    RPPfunction hfunc;

    if (tryCustomKernelLaunch(kernName, blocksPerGrid, threadsPerBlock, threadsPerBlockTail, kparams, cuMod,
                              kernelStream)) {
        return;
    }

    adjustKernelLaunchConfig(kernName, blocksPerGrid, threadsPerBlock, threadsPerBlockTail, kparams);

    res = rppModuleGetFunction(&hfunc, cuMod, kernName.c_str());
    assert(res == RPP_SUCCESS);
    launchKernelRaw(hfunc, kernName, blocksPerGrid, threadsPerBlock, threadsPerBlockTail, kparams, kernelStream);
}

static int find_divisor(int dim, int cur_divisor = 1) {
    int divisor = 0;
    for (int i = cur_divisor + 1; i <= dim; i++) {
        if (dim % i == 0 && dim / i < 65535) {
            divisor = i;
            break;
        }
    }
    return divisor;
}

static bool reduce_block(std::vector<int> & grid_dims, int times, int axis) {
    bool found_place = false;
    if (grid_dims[axis] * times < 65535) {
        found_place = true;
    }

    return found_place;
}

static void combine_dims(RppDims & input_dims_ori,
                         RppDims & output_dims_ori,
                         RppDims & input_dims,
                         RppDims & output_dims,
                         size_t &  axis_index) {
    // combine dim after/before axis_index
    int nbDims                 = input_dims_ori.nbDims;
    int new_nbDims             = 2;
    int input_combined_before  = 1;
    int output_combined_before = 1;
    int new_axis_index         = 0;

    input_dims.d[0] = input_dims.d[1] = input_dims.d[2] = 1;
    output_dims.d[0] = output_dims.d[1] = output_dims.d[2] = 1;

    // combine dim before axis_index
    if (axis_index > 0) {
        for (size_t i = 0; i < axis_index; ++i) {
            input_combined_before *= input_dims_ori.d[i];
            output_combined_before *= output_dims_ori.d[i];
        }
        input_dims.d[0]  = input_combined_before;
        output_dims.d[0] = output_combined_before;
        new_axis_index   = 1;
    } else {
        input_dims.d[0]  = input_dims_ori.d[0];
        output_dims.d[0] = output_dims_ori.d[0];
    }

    // combine dim after axis_index
    input_dims.d[1]  = input_dims_ori.d[axis_index];
    output_dims.d[1] = output_dims_ori.d[axis_index];

    int input_combined_after  = 1;
    int output_combined_after = 1;
    if ((int) axis_index + 1 < nbDims) {
        for (int i = axis_index + 1; i < nbDims; ++i) {
            input_combined_after *= input_dims_ori.d[i];
            output_combined_after *= output_dims_ori.d[i];
        }

        if (axis_index == 0) {
            input_dims.d[1]  = input_combined_after;
            output_dims.d[1] = output_combined_after;
            new_nbDims       = 2;
        } else {
            input_dims.d[2]  = input_combined_after;
            output_dims.d[2] = output_combined_after;
            new_nbDims       = 3;
        }
    }

    input_dims.nbDims  = new_nbDims;
    output_dims.nbDims = new_nbDims;
    axis_index         = new_axis_index;
}

std::vector<RppTaskElement> create_reduce_kernel_task(size_t             axis_index,
                                                      RppReduceOperation operation,
                                                      uint32_t           input_addr,
                                                      uint32_t           output_addr,
                                                      RppDims &          input_dims_ori,
                                                      RppDims &          output_dims_ori,
                                                      uint32_t           mean_addr,
                                                      bool               for_reduce_task,
                                                      bool               is_input_chw32) {
    std::vector<RppTaskElement> multi_tasks;
    RppTaskElement              task;

    RppDims input_dims, output_dims;
    if (input_dims_ori.nbDims < 3 || is_input_chw32) {
        input_dims  = input_dims_ori;
        output_dims = output_dims_ori;
    } else {
        // while dims >= 3, need check if dims can combine
        combine_dims(input_dims_ori, output_dims_ori, input_dims, output_dims, axis_index);
    }

    RppDims opt_output_dims{};
    opt_output_dims.nbDims = output_dims.nbDims;
    for (int i = 0; i < opt_output_dims.nbDims; i++) {
        opt_output_dims.d[i] = output_dims.d[i];
    }

    int output_len             = 1;
    int ori_small_out_dim      = 0;
    int dim_one_count          = 0;
    int last_non_one_dim_index = output_dims.nbDims - 1;
    for (int i = 0; i < output_dims.nbDims; i++) {
        if (output_dims.d[i] == 1) {
            dim_one_count += 1;
        } else {
            last_non_one_dim_index = i;
        }
        output_len *= output_dims.d[i];
    }
    if (dim_one_count >= output_dims.nbDims - 1 && output_dims.d[last_non_one_dim_index] < 32) {
        ori_small_out_dim                         = output_dims.d[last_non_one_dim_index];
        opt_output_dims.d[last_non_one_dim_index] = 32;
    }

    int block_x = 1, block_y = 1, block_z = 1, grid_x = 1, grid_y = 1, grid_z = 1;
    for (int i = opt_output_dims.nbDims - 1; i >= 0; i--) {
        int dim = opt_output_dims[i];
        if (i == opt_output_dims.nbDims - 1) {
            block_x = dim;
        } else if (i == opt_output_dims.nbDims - 2) {
            if (mean_addr != 0) {
                grid_z = (ori_small_out_dim != 0 ? ori_small_out_dim : dim);
            } else {
                block_y = dim;
            }
        } else if (i == opt_output_dims.nbDims - 3) {
            block_z = dim;
        } else if (i == opt_output_dims.nbDims - 4) {
            grid_x = dim;
        } else if (i == opt_output_dims.nbDims - 5) {
            grid_y = dim;
        } else if (i == opt_output_dims.nbDims - 6) {
            grid_z = dim;
        } else {
            grid_z *= dim;
        }
    }
    std::vector<int> grid_dims;
    grid_dims.emplace_back(grid_x);
    grid_dims.emplace_back(grid_y);
    grid_dims.emplace_back(grid_z);

    int divisor_x = 1;
    int divisor_y = 1;
    int divisor_z = 1;
    while (block_x * block_y * block_z > 8191) {
        bool reduced_block = false;

        if (block_z != 1) {
            divisor_z = find_divisor(block_z, divisor_z);
            if (divisor_z != 0 && reduce_block(grid_dims, divisor_z, 2)) {
                if ((block_x * block_y * block_z / divisor_z < 8191) || divisor_z == block_z) {
                    //if grid_z != 1, we need to multiply and recalculate the grid_z
                    grid_z *= divisor_z;
                    block_z /= divisor_z;
                }

                reduced_block = true;
            }
        } else if (block_y != 1) {
            divisor_y = find_divisor(block_y, divisor_y);
            if (divisor_y != 0 && reduce_block(grid_dims, divisor_y, 1)) {
                if ((block_x * block_y * block_z / divisor_y < 8191) || divisor_y == block_y) {
                    grid_y *= divisor_y;
                    block_y /= divisor_y;
                }

                reduced_block = true;
            }
        } else {
            divisor_x = find_divisor(block_x, divisor_x);
            if (divisor_x != 0 && reduce_block(grid_dims, divisor_x, 0)) {
                if ((block_x * block_y * block_z / divisor_x) < 8191 || divisor_x == block_x) {
                    grid_x *= divisor_x;
                    block_x /= divisor_x;
                }

                reduced_block = true;
            }
        }

        if (!reduced_block) {
            throw std::runtime_error("reduce block is 0.");
        }
    }

    if (grid_x > 65535 || grid_y > 65535 || grid_z > 65535) {
        throw std::runtime_error("grid is larger than 65535.");
    }
    int config_len = block_x * block_y * block_z * grid_x * grid_y * grid_z;
    if (config_len != output_len && config_len != 32) {
        throw std::runtime_error("config len is illegal.");
    }

    uint32_t out_stride_y     = block_x * sizeof(uint16_t);
    uint32_t out_stride_z     = block_y * out_stride_y;
    uint32_t out_block_stride = 0;
    if (ori_small_out_dim != 0 && mean_addr == 0) {
        out_block_stride = ori_small_out_dim * sizeof(uint16_t);
    } else {
        out_block_stride = block_x * block_y * block_z * sizeof(uint16_t);
    }
    bool can_use_opt_kernel = (block_x * block_y * block_z % 32 == 0);

    if (block_x * block_y * block_z <= 32) {
        if (block_x != 1) {
            block_x = 33;
        } else if (block_y != 1) {
            block_y = 33;
        } else {
            block_z = 33;
        }
    }

    uint16_t loop_num    = input_dims.d[axis_index] / output_dims.d[axis_index];
    float    avg_scale   = loop_num;
    avg_scale            = 1.0f / avg_scale;
    uint32_t avg_scale_u = *(uint32_t *) &avg_scale;

    bool found_non_one_dim_ahead_axis = false;
    for (size_t i = 0; i < axis_index; i++) {
        if (input_dims.d[i] != 1) {
            found_non_one_dim_ahead_axis = true;
            break;
        }
    }

    task.blockDim.x = block_x;
    task.blockDim.y = block_y;
    task.blockDim.z = block_z;
    task.gridDim.x  = grid_x;
    task.gridDim.y  = grid_y;
    task.gridDim.z  = grid_z;

    bool found_non_one_after_axis = false;
    for (int i = (int) axis_index + 1; i < input_dims.nbDims; i++) {
        found_non_one_after_axis = (input_dims.d[i] != 1);
    }
    // before reduce_axis has 1 dim AND dims %32 == 0
    if (for_reduce_task && !found_non_one_dim_ahead_axis && can_use_opt_kernel) {
        task.taskName = "opt_reduce_f16";
        task.params.kernelList.emplace_back(input_addr);
        task.params.kernelList.emplace_back(output_addr);
        task.params.kernelList.emplace_back(out_block_stride);
        task.params.kernelList.emplace_back(out_block_stride);
        task.params.kernelList.emplace_back(output_len * sizeof(uint16_t));
        task.params.kernelList.emplace_back(avg_scale_u);
        task.params.kernelList.emplace_back(loop_num);
        task.params.kernelList.emplace_back((uint32_t) operation);
    }
    // axis_index is the last 2rd dim
    else if (for_reduce_task && (int) axis_index == input_dims.nbDims - 2 &&
             task.blockDim.x * task.blockDim.y % 32 == 0) {
        task.taskName = "opt_reduce_f16";

        if (grid_x > 1 && grid_z > 1) {
            for (int i = 0; i < grid_z; ++i) {
                task.params.kernelList.clear();
                task.gridDim.z = 1;
                task.params.kernelList.emplace_back(input_addr);
                task.params.kernelList.emplace_back(output_addr);
                task.params.kernelList.emplace_back(out_block_stride);
                task.params.kernelList.emplace_back(out_block_stride);
                task.params.kernelList.emplace_back(output_len * sizeof(uint16_t) / grid_z);
                task.params.kernelList.emplace_back(avg_scale_u);
                task.params.kernelList.emplace_back(loop_num);
                task.params.kernelList.emplace_back((uint32_t) operation);
                input_addr += output_len * loop_num * sizeof(uint16_t) / grid_z;
                output_addr += output_len * sizeof(uint16_t) / grid_z;
                multi_tasks.emplace_back(task);
            }

        } else {
            if (task.blockDim.z != 1) {
                grid_x *= block_z;
                if (grid_x > 65535) {
                    throw std::runtime_error("grid x is larger than 65535.");
                }
                task.gridDim.x  = grid_x;
                task.blockDim.z = 1;
            }

            task.params.kernelList.emplace_back(input_addr);
            task.params.kernelList.emplace_back(output_addr);
            task.params.kernelList.emplace_back(task.blockDim.x * task.blockDim.y * loop_num * sizeof(uint16_t));
            task.params.kernelList.emplace_back(task.blockDim.x * task.blockDim.y * sizeof(uint16_t));
            task.params.kernelList.emplace_back(task.blockDim.x * task.blockDim.y * sizeof(uint16_t));
            task.params.kernelList.emplace_back(avg_scale_u);
            task.params.kernelList.emplace_back(loop_num);
            task.params.kernelList.emplace_back((uint32_t) operation);
        }
    }
    // axis_index is the last axis
    else if (for_reduce_task && ((int) axis_index == input_dims.nbDims - 1 || !found_non_one_after_axis) &&
             output_dims.d[axis_index] % 32 == 0) {
        int out_block_y = 1;
        for (int i = 0; i < axis_index; i++) {
            out_block_y *= output_dims.d[i];
        }
        int              out_block_x = output_dims.d[axis_index];
        std::vector<int> divisor_list;
        for (int i = 2; i <= out_block_y; i++) {
            divisor_list.emplace_back(i);
        }
        int row_num         = 1;
        int ori_out_block_y = out_block_y;
        for (size_t i = 0;
             i < divisor_list.size() && (out_block_x * out_block_y > 8191 ||
                                         (is_input_chw32 && output_dims.d[output_dims.nbDims - 2] % out_block_y != 0));
             i++) {
            if (ori_out_block_y % divisor_list[i] == 0) {
                out_block_y = ori_out_block_y / divisor_list[i];
                row_num     = divisor_list[i];
            }
        }
        if (out_block_x * out_block_y > 8191) {
            throw std::runtime_error("Too large row size.");
        }
        loop_num    = input_dims.d[axis_index] / out_block_x;
        avg_scale   = loop_num;
        avg_scale   = 1.0f / avg_scale;
        avg_scale_u = *(uint32_t *) &avg_scale;

        if (is_input_chw32) {
            if (output_dims.nbDims < 3) {
                throw std::runtime_error("output dims is smaller than 3.");
            }
            if (output_dims.d[output_dims.nbDims - 2] % out_block_y != 0) {
                throw std::runtime_error("out_block_y issue for Reduce");
            }

            int hw_len          = output_dims.d[output_dims.nbDims - 1] * output_dims.d[output_dims.nbDims - 2];
            int kernel_loop_num = output_dims.d[output_dims.nbDims - 2] / out_block_y;
            if (row_num % kernel_loop_num != 0) {
                throw std::runtime_error("Unexpected case.");
            }

            for (int i = 0; i < kernel_loop_num; i++) {
                RppTaskElement t;
                t.taskName         = "opt_aln_reduce_f16";
                t.blockDim.x       = out_block_x;
                t.blockDim.y       = out_block_y;
                t.blockDim.z       = 1;
                t.gridDim.x        = row_num / kernel_loop_num;
                t.gridDim.y        = 1;
                t.gridDim.z        = 1;
                int out_block_size = out_block_x * out_block_y * (int) sizeof(uint16_t);
                t.params.kernelList.emplace_back(input_addr + i * out_block_size);
                t.params.kernelList.emplace_back(output_addr + i * out_block_size);
                t.params.kernelList.emplace_back(out_block_size * loop_num * kernel_loop_num);
                t.params.kernelList.emplace_back(out_block_size * kernel_loop_num);
                t.params.kernelList.emplace_back(hw_len * sizeof(uint16_t));
                t.params.kernelList.emplace_back(avg_scale_u);
                t.params.kernelList.emplace_back(loop_num);
                t.params.kernelList.emplace_back(out_block_x);
                t.params.kernelList.emplace_back((uint32_t) operation);
                multi_tasks.emplace_back(t);
            }
        } else {
            task.taskName   = "opt_aln_z32_reduce_f16";
            task.blockDim.x = out_block_x;
            task.blockDim.y = 1;
            task.blockDim.z = out_block_y;
            task.gridDim.x  = row_num;
            task.gridDim.y  = 1;
            task.gridDim.z  = 1;
            task.params.kernelList.emplace_back(input_addr);
            task.params.kernelList.emplace_back(output_addr);
            task.params.kernelList.emplace_back(out_block_x * out_block_y * sizeof(uint16_t) * loop_num);
            task.params.kernelList.emplace_back(out_block_x * out_block_y * sizeof(uint16_t));
            task.params.kernelList.emplace_back(out_block_x * sizeof(uint16_t));
            task.params.kernelList.emplace_back(avg_scale_u);
            task.params.kernelList.emplace_back(out_block_x * loop_num);
            task.params.kernelList.emplace_back(loop_num);
            task.params.kernelList.emplace_back((uint32_t) operation);
        }
    } else {
        uint32_t in_block_stride = found_non_one_dim_ahead_axis ? out_block_stride * loop_num : out_block_stride;
        uint32_t in_stride_y     = input_dims.d[input_dims.nbDims - 1] * sizeof(uint16_t);
        uint32_t in_stride_z     = input_dims.d[input_dims.nbDims - 2] * in_stride_y;

        uint32_t loop_stride = sizeof(uint16_t);
        if (ori_small_out_dim == 0) {
            for (int i = (int) axis_index; i < opt_output_dims.nbDims; i++) {
                loop_stride *= opt_output_dims.d[i];
            }
        }
        if (found_non_one_dim_ahead_axis && loop_stride > out_block_stride) {
            loop_stride = out_block_stride;
        }

        if (mean_addr == 0) {
            task.taskName = can_use_opt_kernel ? "opt_gen_reduce_f16" : "gen_reduce_f16";

            if (grid_x > 1 && grid_z > 1) {
                for (int i = 0; i < grid_z; ++i) {
                    task.params.kernelList.clear();
                    task.gridDim.z = 1;

                    task.params.kernelList.emplace_back(input_addr);
                    task.params.kernelList.emplace_back(output_addr);
                    task.params.kernelList.emplace_back(out_block_stride);
                    task.params.kernelList.emplace_back(out_block_stride);
                    task.params.kernelList.emplace_back(in_stride_y);
                    task.params.kernelList.emplace_back(in_stride_z);
                    task.params.kernelList.emplace_back(out_stride_y);
                    task.params.kernelList.emplace_back(out_stride_z);

                    loop_stride = sizeof(uint16_t);
                    if (ori_small_out_dim == 0) {
                        for (int i = (int) axis_index; i < opt_output_dims.nbDims; i++) {
                            loop_stride *= opt_output_dims.d[i];
                        }
                    }
                    task.params.kernelList.emplace_back(loop_stride);
                    task.params.kernelList.emplace_back(avg_scale_u);
                    task.params.kernelList.emplace_back(loop_num);
                    task.params.kernelList.emplace_back((uint32_t) operation);

                    multi_tasks.emplace_back(task);

                    input_addr += output_len * loop_num * sizeof(uint16_t) / grid_z;
                    output_addr += output_len * sizeof(uint16_t) / grid_z;
                }
            } else {
                task.params.kernelList.emplace_back(input_addr);
                task.params.kernelList.emplace_back(output_addr);
                task.params.kernelList.emplace_back(in_block_stride);
                task.params.kernelList.emplace_back(out_block_stride);
                task.params.kernelList.emplace_back(in_stride_y);
                task.params.kernelList.emplace_back(in_stride_z);
                task.params.kernelList.emplace_back(out_stride_y);
                task.params.kernelList.emplace_back(out_stride_z);
                task.params.kernelList.emplace_back(loop_stride);
                task.params.kernelList.emplace_back(avg_scale_u);
                task.params.kernelList.emplace_back(loop_num);
                task.params.kernelList.emplace_back((uint32_t) operation);
            }
        } else {
            task.taskName = "opt_variance_f16_f32_f16";
            task.params.kernelList.emplace_back(input_addr);
            task.params.kernelList.emplace_back(mean_addr);
            task.params.kernelList.emplace_back(output_addr);
            task.params.kernelList.emplace_back(in_block_stride);
            task.params.kernelList.emplace_back(out_block_stride);
            task.params.kernelList.emplace_back(in_stride_y);
            task.params.kernelList.emplace_back(in_stride_z);
            task.params.kernelList.emplace_back(out_stride_y);
            task.params.kernelList.emplace_back(out_stride_z);
            task.params.kernelList.emplace_back(loop_stride);
            task.params.kernelList.emplace_back(avg_scale_u);
            task.params.kernelList.emplace_back(loop_num);
        }
    }

    if (!multi_tasks.empty()) {
        return multi_tasks;
    } else {
        return { task };
    }
}

bool ReduceSpawnIO(int       axis,
                   RppDims & in_dims,
                   RppDims & out_dims,
                   RppDims & mid_dims,
                   bool      for_reduce_task,
                   bool      for_chw32) {
    bool found_non_one_after_axis = false;
    for (int i = axis + 1; i < in_dims.nbDims; i++) {
        found_non_one_after_axis = (in_dims.d[i] != 1);
    }

    if (for_reduce_task && for_chw32 && (axis == in_dims.nbDims - 1 || !found_non_one_after_axis) &&
        (in_dims.d[axis] <= 8191 * 2 && in_dims.d[axis] > 32 && in_dims.d[axis] % 32 == 0 &&
         (in_dims.d[axis] / 32) % 2 == 0)) {
        mid_dims.nbDims = in_dims.nbDims;
        for (int i = 0; i < in_dims.nbDims; i++) {
            mid_dims.d[i] = in_dims.d[i];
        }
        mid_dims.d[axis] = 32;
        return (mid_dims.Length() != out_dims.Length());
    } else {
        int ori_num_threads = out_dims.Length();
        int axis_dim        = in_dims.d[axis];
        if (ori_num_threads >= 4096 || axis_dim == 1) {
            return false;
        }

        std::vector<int> axis_dim_plans;
        for (int i = 1; i * ori_num_threads < 8192 && i < axis_dim; i++) {
            if (axis_dim % i == 0) {
                axis_dim_plans.emplace_back(i);
            }
        }
        // use opt_reduce in the beginning reduce step
        if (for_reduce_task && axis_dim_plans[axis_dim_plans.size() - 1] % 32 != 0 && in_dims.d[axis] > 32 &&
            in_dims.d[axis] % 32 == 0) {
            axis_dim_plans.emplace_back(32);
        }

        mid_dims.nbDims = out_dims.nbDims;
        for (int i = 0; i < out_dims.nbDims; i++) {
            mid_dims.d[i] = (i == axis ? axis_dim_plans[axis_dim_plans.size() - 1] : out_dims.d[i]);
        }

        // This will let all reduce kernels be opt_reduce_f16
        bool is_1d_input = false;
        for (int i = 0; i < in_dims.nbDims; i++) {
            if (in_dims.d[i] == in_dims.Length()) {
                is_1d_input = true;
                break;
            }
        }
        if (is_1d_input && mid_dims.d[axis] > 4096 && in_dims.Length() % 4096 == 0) {
            mid_dims.d[axis] = 4096;
        }

        return (mid_dims.Length() >= 64 && mid_dims.Length() != out_dims.Length());
    }
}

void cvt_kernel_param_init(const dim3 &            threadsPerBlock,
                           uint32_t                inAddr,
                           uint32_t                outAddr,
                           RppDataType             inDataType,
                           RppDataType             outDataType,
                           std::vector<uint32_t> & params) {
    uint32_t unRollInStride  = threadsPerBlock.x * GetRppElementSize(inDataType);
    uint32_t unRollOutStride = threadsPerBlock.x * GetRppElementSize(outDataType);
    params.emplace_back(inAddr);
    params.emplace_back(outAddr);
    params.emplace_back(inDataType);
    params.emplace_back(outDataType);
    params.emplace_back(1);
    params.emplace_back(unRollInStride);
    params.emplace_back(unRollOutStride);
}

void cvt_kernel_param_init_opt(const dim3 &            threadsPerBlock,
                               uint32_t                inAddr,
                               uint32_t                outAddr,
                               RppDataType             inDataType,
                               RppDataType             outDataType,
                               std::vector<uint32_t> & params) {
    uint32_t unRollInStride  = threadsPerBlock.x * GetRppElementSize(inDataType) / 2;
    uint32_t unRollOutStride = threadsPerBlock.x * GetRppElementSize(outDataType) / 2;
    params.emplace_back(inAddr);
    params.emplace_back(outAddr);
    params.emplace_back(inDataType);
    params.emplace_back(outDataType);
    params.emplace_back(1);
    params.emplace_back(unRollInStride);
    params.emplace_back(unRollOutStride);
}

void fill_16bits_align_params(int                     output,
                              int                     block_x,
                              int16_t                 value,
                              int                     type_of_bytes,
                              std::vector<uint32_t> & params) {
    int block_size = block_x * type_of_bytes;

    params.emplace_back(output);
    params.emplace_back(block_size);
    params.emplace_back(value);
}

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
                            bool                    is_row_round) {
    (void) block_z;

    int inStrideY, inStrideZ;
    int outStrideY, outStrideZ;
    int inBlockXStride, inBlockYStride, inBlockZStride;
    int outBlockXStride, outBlockYStride, outBlockZStride;
    int column_round, row_round;

    column_round = (column + 31) / 32 * 32;
    row_round    = is_row_round ? ((row + 31) / 32 * 32) : row;

    inStrideY = column;
    inStrideZ = 0;

    outStrideY = block_x;
    outStrideZ = 0;

    inBlockXStride = block_x * type_of_bytes;
    inBlockYStride = block_y * column * type_of_bytes;
    inBlockZStride = row * column * type_of_bytes;

    outBlockXStride = row_round * block_x * type_of_bytes;
    outBlockYStride = block_y * block_x * type_of_bytes;
    outBlockZStride = row_round * column_round * type_of_bytes;

    params.emplace_back(input);
    params.emplace_back(output);
    params.emplace_back(inStrideY);
    params.emplace_back(inStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(inBlockXStride);
    params.emplace_back(outBlockXStride);
    params.emplace_back(inBlockYStride);
    params.emplace_back(outBlockYStride);
    params.emplace_back(inBlockZStride);
    params.emplace_back(outBlockZStride);
    params.emplace_back(grid_y_tail);
    params.emplace_back(0);
}

void chw322chw_align_params(int                     input,
                            int                     output,
                            int                     row,
                            int                     column,
                            int                     grid_y_tail,
                            int                     block_x,
                            int                     block_y,
                            int                     block_z,
                            int                     type_of_bytes,
                            std::vector<uint32_t> & params) {
    //int in_block_size;
    //int out_block_size;
    int inStrideY, inStrideZ;
    int outStrideY, outStrideZ;
    int inBlockXStride, inBlockYStride, inBlockZStride;
    int outBlockXStride, outBlockYStride, outBlockZStride;

    int column_round = (column + 31) / 32 * 32;
    inStrideY        = block_x;
    inStrideZ        = 0;
    outStrideY       = column;
    outStrideZ       = 0;

    inBlockXStride = row * block_x * type_of_bytes;
    inBlockYStride = block_y * block_x * type_of_bytes;
    inBlockZStride = row * column_round * type_of_bytes;

    //output column will round to 32x
    outBlockXStride = block_x * type_of_bytes;
    outBlockYStride = block_y * column * type_of_bytes;
    outBlockZStride = row * column * type_of_bytes;

    params.emplace_back(input);
    params.emplace_back(output);
    params.emplace_back(inStrideY);
    params.emplace_back(inStrideZ);
    params.emplace_back(outStrideY);
    params.emplace_back(outStrideZ);
    params.emplace_back(inBlockXStride);
    params.emplace_back(outBlockXStride);
    params.emplace_back(inBlockYStride);
    params.emplace_back(outBlockYStride);
    params.emplace_back(inBlockZStride);
    params.emplace_back(outBlockZStride);
    params.emplace_back(grid_y_tail);
    params.emplace_back(0);
}

void calc_tbdim_flattern(uint32_t D0, uint32_t D1, dim3 & threadsPerBlock, dim3 & blocksPerGrid) {
    const uint32_t MAX_THREADS = 8192;
    threadsPerBlock.y          = 1;
    threadsPerBlock.z          = 1;
    blocksPerGrid.y            = 1;
    blocksPerGrid.z            = 1;

    if (D0 * D1 < MAX_THREADS) {
        threadsPerBlock.x = D0 * D1;
        blocksPerGrid.x   = 1;
    } else {
        threadsPerBlock.x = D1;
        blocksPerGrid.x   = D0;
    }
    while (threadsPerBlock.x >= MAX_THREADS) {
        threadsPerBlock.x = threadsPerBlock.x / 2;
        blocksPerGrid.x   = blocksPerGrid.x * 2;
    }

    if (threadsPerBlock.x <= 32) {
        if (threadsPerBlock.y * threadsPerBlock.z == 1) {
            threadsPerBlock.x = 33;
        } else {
            assert(false);
        }
    } else {
        assert(threadsPerBlock.x * blocksPerGrid.x == D0 * D1);
    }
}

uint32_t RppHW32Fix(uint32_t block_x, uint32_t block_y, uint32_t block_z) {
    uint32_t out_z;
    if (block_x <= 0 || block_y <= 0 || block_z <= 0) {
        throw std::runtime_error("Invalid thread block");
    }
    if (block_x * block_y * block_z > 32) {
        out_z = block_z;
    } else {
        auto old_block_z = block_z;
        while (1) {
            block_z += 1;
            if (block_x * block_y * block_z > 32) {
                break;
            }
        }
        out_z = block_z;
    }
    return out_z;
}
