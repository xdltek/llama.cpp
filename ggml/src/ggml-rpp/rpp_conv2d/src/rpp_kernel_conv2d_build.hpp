#include "rpp_drv_api.h"
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"

#include <rpp_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kSramLimitBytes = 22 * 1024 * 1024;

struct ConvTaskConfig {
    dim3                  block{};
    dim3                  grid{};
    dim3                  sub_block{};
    std::vector<uint32_t> params;
};

static inline int round_up_int(int x, int align) {
    return (x + align - 1) / align * align;
}

static inline size_t sram_round(size_t bytes) {
    return ((bytes + 511) / 512) * 512 + 512;
}

static inline int output_dim(int input, int kernel, int stride, int pad_before, int pad_after, int dilation) {
    return (input + pad_before + pad_after - dilation * (kernel - 1) - 1) / stride + 1;
}

static inline bool use_hwc_path(int in_channels) {
    return in_channels < 32;
}

static inline uint32_t hw32_workaround(uint32_t block_x, uint32_t block_y, uint32_t block_z) {
    if (block_x == 0 || block_y == 0 || block_z == 0) {
        throw std::runtime_error("Invalid zero block dimension");
    }
    if (block_x * block_y * block_z > 32) {
        return block_z;
    }
    while (block_x * block_y * block_z <= 32) {
        ++block_z;
    }
    return block_z;
}

static void calc_conv2d_dim(int        out_height,
                            int        out_width,
                            uint32_t & block_x,
                            uint32_t & block_y,
                            uint32_t & block_z,
                            uint32_t & sub_block_x,
                            uint32_t & sub_block_y,
                            uint32_t & sub_block_z,
                            uint32_t & grid_x,
                            uint32_t & grid_y,
                            uint32_t & grid_z) {
    int width  = out_width;
    int height = 1;

    while (true) {
        if ((width * height < KDC_H_MAX_EXECUT_NUM) && (height < out_height)) {
            ++height;
        } else {
            break;
        }
        if (height == out_height) {
            break;
        }
    }

    if (width * height >= KDC_H_MAX_EXECUT_NUM) {
        if (height > 1) {
            --height;
        } else {
            while (true) {
                width = (width % 2 == 0) ? width / 2 : width / 2 + 1;
                if (width * height < KDC_H_MAX_EXECUT_NUM) {
                    break;
                }
            }
        }
    }

    if (height != out_height) {
        if (height * 2 >= out_height) {
            height = (out_height + 1) / 2;
        } else if (height * 10 >= out_height * 3) {
            height = (out_height + 2) / 3;
        } else if (height * 5 >= out_height) {
            height = (out_height + 4) / 5;
        } else if (height * 10 >= out_height) {
            height = (out_height + 9) / 10;
        } else {
            if (height == 0) {
                height = 1;
            } else {
                int lastblock_height = out_height % height;
                if (lastblock_height > 0) {
                    while ((height - lastblock_height > 2) && (height > lastblock_height)) {
                        --height;
                        lastblock_height = out_height % height;
                    }
                }
            }
        }
    }

    if (width * height >= KDC_H_MAX_EXECUT_NUM) {
        while (height > 1) {
            --height;
            int lastblock_height = out_height % height;
            if (lastblock_height == 0 || (height - lastblock_height <= 2)) {
                break;
            }
        }
    }

    block_x     = KDC_H_DIM_X;
    block_y     = static_cast<uint32_t>(width);
    block_z     = hw32_workaround(block_x, block_y, static_cast<uint32_t>(height));
    grid_x      = 1;
    grid_y      = (out_width + width - 1) / width;
    grid_z      = (out_height + height - 1) / height;
    sub_block_x = block_x;
    sub_block_y = block_y;
    sub_block_z = block_z;
    if (height * static_cast<int>(grid_z) > out_height) {
        sub_block_z = out_height - (grid_z - 1) * height;
    }
}

static void calc_chw2hwc_dim(uint32_t   in_height,
                             uint32_t   in_width,
                             uint32_t & block_x,
                             uint32_t & block_y,
                             uint32_t & block_z,
                             uint32_t & grid_x,
                             uint32_t & grid_y,
                             uint32_t & grid_z) {
    uint32_t y = 1;
    grid_x     = 1;
    while (in_width > KDC_H_MAX_THREAD_NUM) {
        while (true) {
            ++grid_x;
            if (in_width % grid_x == 0) {
                in_width /= grid_x;
                break;
            }
        }
    }
    while (y * in_width < KDC_H_MAX_THREAD_NUM) {
        if (y >= in_height) {
            break;
        }
        ++y;
    }
    while (y * in_width > KDC_H_MAX_THREAD_NUM) {
        --y;
        if (y == 0) {
            throw std::runtime_error("Invalid CHW->HWC block shape");
        }
    }
    if (y < in_height) {
        y = 1;
    }
    block_x = in_width;
    block_y = y;
    block_z = hw32_workaround(block_x, block_y, 1);
    grid_y  = (in_height + y - 1) / y;
    grid_z  = 1;
}

static void calc_chw2hwc32_dim(uint32_t   in_channel,
                               uint32_t   in_height,
                               uint32_t   in_width,
                               uint32_t & block_x,
                               uint32_t & block_y,
                               uint32_t & block_z,
                               uint32_t & grid_x,
                               uint32_t & grid_y,
                               uint32_t & grid_z) {
    const uint32_t hw_size      = in_height * in_width;
    const uint32_t total_thread = in_channel * hw_size;
    uint32_t       bx           = in_channel;
    uint32_t       by           = 1;

    if (in_channel > KDC_H_MAX_THREAD_NUM) {
        while (bx > KDC_H_MAX_THREAD_NUM) {
            bx = (bx + 1) / 2;
        }
        by     = 1;
        grid_y = hw_size;
    } else if (total_thread > KDC_H_MAX_THREAD_NUM) {
        by     = KDC_H_MAX_THREAD_NUM / in_channel;
        grid_y = (hw_size + by - 1) / by;
    } else {
        by     = hw_size;
        grid_y = 1;
    }

    block_x = bx;
    block_y = by;
    block_z = hw32_workaround(block_x, block_y, 1);
    grid_x  = (in_channel + bx - 1) / bx;
    grid_z  = 1;
}

static void calc_hwc322chw_dim(uint32_t   out_channel,
                               uint32_t   out_height,
                               uint32_t   out_width,
                               uint32_t & block_x,
                               uint32_t & block_y,
                               uint32_t & block_z,
                               uint32_t & grid_x,
                               uint32_t & grid_y,
                               uint32_t & grid_z) {
    if (out_channel > KDC_H_MAX_THREAD_NUM) {
        throw std::runtime_error("HWC32->CHW output channel is too large");
    }
    const uint32_t hw_size      = out_height * out_width;
    const uint32_t total_thread = out_channel * hw_size;
    const uint32_t bx           = out_channel;
    uint32_t       by           = 1;

    if (total_thread > KDC_H_MAX_THREAD_NUM) {
        by     = KDC_H_MAX_THREAD_NUM / out_channel;
        grid_y = (hw_size + by - 1) / by;
    } else {
        by     = hw_size;
        grid_y = 1;
    }

    block_x = bx;
    block_y = by;
    block_z = hw32_workaround(block_x, block_y, 1);
    grid_x  = 1;
    grid_z  = 1;
}

static void calc_hwc322hwc33_dim(uint32_t   height,
                                 uint32_t   width,
                                 uint32_t   channel,
                                 uint32_t & block_x,
                                 uint32_t & block_y,
                                 uint32_t & block_z,
                                 uint32_t & grid_x,
                                 uint32_t & grid_y,
                                 uint32_t & grid_z) {
    block_x              = 32;
    const uint32_t group = (channel + 31) / 32;
    uint32_t       dim_y = group * height * width;
    while (dim_y >= 256) {
        dim_y = (dim_y + 1) / 2;
    }
    block_y = dim_y;
    block_z = hw32_workaround(block_x, block_y, 1);
    grid_x  = group * height * width / block_y;
    grid_y  = 1;
    grid_z  = 1;
}

static void launch_fill_16bits_zero(rpp_kernel_context &    ctx,
                                    RPPdeviceptr            addr,
                                    int                     bytes,
                                    dim3 &                  threads,
                                    dim3 &                  grid,
                                    std::vector<uint32_t> & params) {
    if (bytes <= 0) {
        return;
    }
    const int count16 = bytes / static_cast<int>(sizeof(uint16_t));
    if (count16 <= 0) {
        return;
    }
    params.clear();
    calc_tbdim_flattern(1, count16, threads, grid);
    fill_16bits_align_params(static_cast<int>(addr), static_cast<int>(threads.x), 0, static_cast<int>(sizeof(uint16_t)),
                             params);
    launchWrapperAysnc("fill_16bits_align", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);
}

static void append_conv_params(ConvTaskConfig & task,
                               uint32_t         in_addr,
                               uint32_t         filter_addr,
                               uint32_t         bias_addr,
                               uint32_t         out_addr,
                               int              filter_h,
                               int              filter_w,
                               int              in_channels,
                               int              stride_h,
                               int              stride_w,
                               int              in_height,
                               int              in_width,
                               int              out_height,
                               int              out_width,
                               bool             hwc32,
                               bool             has_bias) {
    const uint16_t in_left_shift  = 1;
    const uint16_t out_left_shift = 1;
    uint16_t       tail_acc       = in_channels & 31;
    if (tail_acc == 0) {
        tail_acc = 32;
    }

    const uint32_t max_acc    = std::min(in_channels, 32);
    const uint32_t cn         = (in_channels + 31) >> 5;
    const uint32_t tmp_tail   = max_acc * filter_w;
    const uint32_t rpt_col_m1 = ((tmp_tail + 7) >> 3) - 1;
    uint16_t       tail_len   = 8;
    const uint32_t tmp        = (filter_w * tail_acc) & 7;
    if (tmp) {
        tail_len = static_cast<uint16_t>(tmp);
    }

    const uint32_t wc_round      = round_up_int(filter_w * in_channels, 8);
    const uint32_t filter_offset = wc_round * filter_h << (in_left_shift + 5);
    const uint32_t tail_xyz      = task.block.x * task.block.y * task.block.z;

    uint32_t tail_n            = 0;
    uint32_t tail_stride_y     = 0;
    uint32_t tail_stride_z     = 0;
    uint32_t tail_rpt_col_m1   = 0;
    uint32_t in_stride_y       = 0;
    uint32_t in_stride_z       = 0;
    uint32_t in_switch_size    = 0;
    uint32_t in_combine_offset = 0;

    if (hwc32) {
        tail_n                  = 32;
        tail_stride_y           = tail_n * stride_w << 1;
        tail_stride_z           = in_width * tail_n * stride_h << 1;
        tail_rpt_col_m1         = ((filter_w * tail_n + 7) / 8) - 1;
        const uint32_t rollback = rpt_col_m1 << 4;
        in_stride_y             = stride_w << 6;
        in_stride_z             = in_width * stride_h << 6;
        in_switch_size          = (in_width << 6) - rollback;
        in_combine_offset       = in_width * in_height << 6;
    } else {
        tail_n = in_channels - ((in_channels >> 5) << 5);
        if (tail_n == 0) {
            tail_n = in_channels;
        }
        tail_rpt_col_m1         = ((filter_w * tail_n + 7) >> 3) - 1;
        tail_stride_y           = tail_n * stride_w << 1;
        tail_stride_z           = in_width * tail_n * stride_h << 1;
        const uint32_t rollback = rpt_col_m1 << 4;
        in_stride_y             = in_channels * stride_w << 1;
        in_stride_z             = in_channels * in_width * stride_h << 1;
        in_switch_size          = (in_width * in_channels << 1) - rollback;
        in_combine_offset       = in_width * in_height * in_channels * 2;
    }

    const uint32_t tail_rollback         = tail_rpt_col_m1 << 4;
    const uint32_t tail_switch_size      = (in_width * tail_n << 1) - tail_rollback;
    const uint32_t tail_block_stride     = (in_width * in_height << 6) * (cn - 1);
    const uint32_t out_stride_y          = 32;
    const uint32_t out_stride_z          = out_width << 5;
    const uint32_t out_block_size        = out_height * out_width << 6;
    const uint32_t in_un_stride          = in_stride_z * task.block.z;
    const uint32_t out_un_stride         = out_stride_z * task.block.z << out_left_shift;
    const uint32_t tail_in_un_stride     = tail_stride_z * task.block.z;
    const uint32_t in_grid_y_stride      = task.block.x * task.block.y * stride_w * 2;
    const uint32_t tail_in_grid_y_stride = tail_acc * task.block.y * stride_w * 2;
    const uint32_t out_grid_y_stride     = task.block.x * task.block.y * 2;
    const uint32_t bias_offset           = task.block.x << 1;

    task.params.clear();
    task.params.emplace_back(in_addr);
    task.params.emplace_back(filter_addr);
    task.params.emplace_back(bias_addr);
    task.params.emplace_back(out_addr);
    task.params.emplace_back(cn - 1);
    task.params.emplace_back(has_bias ? 1u : 0u);
    task.params.emplace_back(65536);
    task.params.emplace_back(0);
    task.params.emplace_back(0);
    task.params.emplace_back(0);
    task.params.emplace_back(in_stride_y);
    task.params.emplace_back(out_stride_y);
    task.params.emplace_back(in_stride_z);
    task.params.emplace_back(out_stride_z);
    task.params.emplace_back(in_switch_size);
    task.params.emplace_back(out_block_size);
    task.params.emplace_back(in_combine_offset);
    task.params.emplace_back(0);
    task.params.emplace_back(0);
    task.params.emplace_back(0);
    task.params.emplace_back(filter_h);
    task.params.emplace_back(rpt_col_m1);
    task.params.emplace_back(tail_len);
    task.params.emplace_back(1);
    task.params.emplace_back(in_un_stride);
    task.params.emplace_back(out_un_stride);
    task.params.emplace_back(0);
    task.params.emplace_back(tail_xyz);
    task.params.emplace_back(tail_rpt_col_m1);
    task.params.emplace_back(tail_switch_size);
    task.params.emplace_back(tail_block_stride);
    task.params.emplace_back(tail_in_un_stride);
    task.params.emplace_back(0);
    task.params.emplace_back(tail_stride_y);
    task.params.emplace_back(tail_stride_z);
    task.params.emplace_back(in_grid_y_stride);
    task.params.emplace_back(tail_in_grid_y_stride);
    task.params.emplace_back(out_grid_y_stride);
    task.params.emplace_back(bias_offset);
    task.params.emplace_back(filter_offset);
    task.params.emplace_back(0);
}

static void launch_chw2hwc(rpp_kernel_context &    ctx,
                           RPPdeviceptr            in_addr,
                           RPPdeviceptr            out_addr,
                           int                     channels,
                           int                     height,
                           int                     width,
                           dim3 &                  threads,
                           dim3 &                  grid,
                           std::vector<uint32_t> & params) {
    uint32_t bx = 1, by = 1, bz = 1, gx = 1, gy = 1, gz = 1;
    calc_chw2hwc_dim(height, width, bx, by, bz, gx, gy, gz);
    threads = { bx, by, bz };
    grid    = { gx, gy, gz };
    params.clear();
    params.emplace_back(static_cast<uint32_t>(in_addr));
    params.emplace_back(static_cast<uint32_t>(out_addr));
    params.emplace_back(height * width * static_cast<int>(sizeof(uint16_t)));
    params.emplace_back(1);
    params.emplace_back(0);
    params.emplace_back(channels);
    params.emplace_back(width);
    params.emplace_back(height);
    launchWrapperAysnc("chw2hwc_f16_f32_f16_all", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);
}

static void launch_chw2hwc32(rpp_kernel_context &    ctx,
                             RPPdeviceptr            in_addr,
                             RPPdeviceptr            out_addr,
                             int                     channels,
                             int                     height,
                             int                     width,
                             dim3 &                  threads,
                             dim3 &                  grid,
                             std::vector<uint32_t> & params) {
    uint32_t bx = 1, by = 1, bz = 1, gx = 1, gy = 1, gz = 1;
    calc_chw2hwc32_dim(channels, height, width, bx, by, bz, gx, gy, gz);
    threads = { bx, by, bz };
    grid    = { gx, gy, gz };
    params.clear();
    params.emplace_back(static_cast<uint32_t>(in_addr));
    params.emplace_back(static_cast<uint32_t>(out_addr));
    params.emplace_back(height * width);
    params.emplace_back(1);
    params.emplace_back(0);
    params.emplace_back(0);
    params.emplace_back(threads.x);
    params.emplace_back(0);
    launchWrapperAysnc("chw2hwc32_f16_f32_f16_all", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);
}

static void launch_hwc322chw(rpp_kernel_context &    ctx,
                             RPPdeviceptr            in_addr,
                             RPPdeviceptr            out_addr,
                             int                     channels,
                             int                     height,
                             int                     width,
                             dim3 &                  threads,
                             dim3 &                  grid,
                             std::vector<uint32_t> & params) {
    uint32_t bx = 1, by = 1, bz = 1, gx = 1, gy = 1, gz = 1;
    calc_hwc322chw_dim(channels, height, width, bx, by, bz, gx, gy, gz);
    threads              = { bx, by, bz };
    grid                 = { gx, gy, gz };
    const int hw_size    = height * width;
    const int loop       = channels / static_cast<int>(threads.x);
    const int block_size = static_cast<int>(threads.x) * hw_size * static_cast<int>(sizeof(uint16_t));
    params.clear();
    params.emplace_back(static_cast<uint32_t>(in_addr));
    params.emplace_back(static_cast<uint32_t>(out_addr));
    params.emplace_back(hw_size);
    params.emplace_back(loop);
    params.emplace_back(block_size);
    params.emplace_back(block_size);
    launchWrapperAysnc("hwc322chw_f16_f32_f16_all", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);
}

static void launch_hwc322hwc33(rpp_kernel_context &    ctx,
                               RPPdeviceptr            in_addr,
                               RPPdeviceptr            out_addr,
                               int                     channels,
                               int                     height,
                               int                     width,
                               dim3 &                  threads,
                               dim3 &                  grid,
                               std::vector<uint32_t> & params) {
    uint32_t bx = 1, by = 1, bz = 1, gx = 1, gy = 1, gz = 1;
    calc_hwc322hwc33_dim(height, width, channels, bx, by, bz, gx, gy, gz);
    const uint32_t total_blocks = ((channels + 31) / 32) * height * width;

    auto init_params = [&](RPPdeviceptr input, RPPdeviceptr output) {
        params.clear();
        params.emplace_back(static_cast<uint32_t>(input));
        params.emplace_back(static_cast<uint32_t>(output));
        params.emplace_back(bx * by * static_cast<uint32_t>(sizeof(uint16_t)));
        params.emplace_back(1);
        params.emplace_back((bx + 1) * by * static_cast<uint32_t>(sizeof(uint16_t)));
        params.emplace_back(bx);
        params.emplace_back(0);
        params.emplace_back(bx + 1);
        params.emplace_back(0);
    };

    threads = { bx, by, bz };
    grid    = { gx, gy, gz };
    init_params(in_addr, out_addr);
    launchWrapperAysnc("hwc322hwc128_f16_f32_f16_all", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);

    if (total_blocks > by * gx) {
        const uint32_t     handled_blocks = by * gx;
        const uint32_t     tail_blocks    = total_blocks - handled_blocks;
        const RPPdeviceptr tail_in        = in_addr + static_cast<RPPdeviceptr>(bx * handled_blocks * sizeof(uint16_t));
        const RPPdeviceptr tail_out =
            out_addr + static_cast<RPPdeviceptr>((bx + 1) * handled_blocks * sizeof(uint16_t));
        threads = { bx, tail_blocks, bz };
        grid    = { 1, gy, gz };
        by      = tail_blocks;
        init_params(tail_in, tail_out);
        launchWrapperAysnc("hwc322hwc128_f16_f32_f16_all", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);
    }
}

static void launch_hwc332chw_opt1(rpp_kernel_context &    ctx,
                                  RPPdeviceptr            in_addr,
                                  RPPdeviceptr            out_addr,
                                  int                     channels,
                                  int                     height,
                                  int                     width,
                                  dim3 &                  threads,
                                  dim3 &                  grid,
                                  std::vector<uint32_t> & params) {
    const uint32_t bx      = 32;
    const uint32_t by      = 32;
    const uint32_t bz      = 4;
    const uint32_t hw_size = height * width;
    const uint32_t loop    = hw_size / (bx * bz);
    if (channels % 32 != 0 || loop == 0) {
        throw std::runtime_error("HWC33->CHW opt1 requires channel multiple of 32 and HW >= 128");
    }

    auto init_params = [&](RPPdeviceptr input, RPPdeviceptr output, uint32_t loop_n) {
        params.clear();
        params.emplace_back(static_cast<uint32_t>(input));
        params.emplace_back(static_cast<uint32_t>(output));
        params.emplace_back((by + 1) * static_cast<uint32_t>(sizeof(uint16_t)));
        params.emplace_back(hw_size);
        params.emplace_back(loop_n);
        params.emplace_back((by + 1) * hw_size * static_cast<uint32_t>(sizeof(uint16_t)));
        params.emplace_back(by * hw_size * static_cast<uint32_t>(sizeof(uint16_t)));
    };

    threads = { bx, by, bz };
    grid    = { static_cast<uint32_t>(channels / 32), 1, 1 };
    init_params(in_addr, out_addr, loop);
    launchWrapperAysnc("hwc322chw_f16_f32_f16_all_opt1", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);

    if (hw_size % (bx * bz) != 0) {
        const RPPdeviceptr tail_offset = static_cast<RPPdeviceptr>((hw_size - bx * bz) * sizeof(uint16_t));
        init_params(in_addr + tail_offset * (by + 1), out_addr + tail_offset, 1);
        launchWrapperAysnc("hwc322chw_f16_f32_f16_all_opt1", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);
    }
}

static void launch_conv(rpp_kernel_context & ctx,
                        RPPdeviceptr         in_addr,
                        RPPdeviceptr         weight_addr,
                        RPPdeviceptr         bias_addr,
                        RPPdeviceptr         out_addr,
                        int                  in_channels,
                        int                  in_height,
                        int                  in_width,
                        int                  out_channels,
                        int                  out_height,
                        int                  out_width,
                        int                  kernel_h,
                        int                  kernel_w,
                        int                  stride_h,
                        int                  stride_w,
                        bool                 hwc32,
                        bool                 has_bias) {
    ConvTaskConfig task;
    uint32_t       bx = 1, by = 1, bz = 1, sbx = 1, sby = 1, sbz = 1, gx = 1, gy = 1, gz = 1;
    calc_conv2d_dim(out_height, out_width, bx, by, bz, sbx, sby, sbz, gx, gy, gz);
    task.block     = { bx, by, bz };
    task.grid      = { gx, gy, gz };
    task.sub_block = { sbx, sby, sbz };
    append_conv_params(task, static_cast<uint32_t>(in_addr), static_cast<uint32_t>(weight_addr),
                       static_cast<uint32_t>(bias_addr), static_cast<uint32_t>(out_addr), kernel_h, kernel_w,
                       in_channels, stride_h, stride_w, in_height, in_width, out_height, out_width, hwc32, has_bias);

    task.grid.x     = (out_channels + 31) / 32;
    bool width_tail = false;
    if (task.block.y * task.grid.y > static_cast<uint32_t>(out_width)) {
        --task.grid.y;
        width_tail = true;
        if (task.grid.y == 0) {
            throw std::runtime_error("Invalid Conv grid y");
        }
    }

    launchWrapperAysnc("conv_gen_tn1_mac8", task.grid, task.block, task.sub_block, task.params, ctx.rppBinMod,
                       ctx.kernelStream);

    if (width_tail) {
        ConvTaskConfig tail              = task;
        const uint32_t block_y           = out_width - task.block.y * task.grid.y;
        const uint32_t in_channel_stride = hwc32 ? 32 : in_channels;
        const uint32_t out_offset        = task.block.y * task.grid.y * task.block.x * sizeof(uint16_t);
        const uint32_t in_offset         = in_channel_stride * task.block.y * task.grid.y * stride_w * sizeof(uint16_t);
        tail.block.y                     = block_y;
        tail.grid.y                      = 1;
        tail.sub_block.y                 = block_y;
        tail.params[0] += in_offset;
        tail.params[3] += out_offset;
        launchWrapperAysnc("conv_gen_tn1_mac8", tail.grid, tail.block, tail.sub_block, tail.params, ctx.rppBinMod,
                           ctx.kernelStream);
    }
}

static void copy_input_to_sram(rpp_kernel_context &    ctx,
                               RPPdeviceptr            dev_in,
                               RPPdeviceptr            sram_in,
                               int                     channels,
                               int                     height,
                               int                     width,
                               int                     padded_height,
                               int                     padded_width,
                               int                     pad_top,
                               int                     pad_left,
                               int                     bytes_per_element,
                               dim3 &                  threads,
                               dim3 &                  grid,
                               std::vector<uint32_t> & params) {
    const size_t input_bytes  = static_cast<size_t>(channels) * height * width * bytes_per_element;
    const size_t padded_bytes = static_cast<size_t>(channels) * padded_height * padded_width * bytes_per_element;

    if (height == padded_height && width == padded_width && pad_top == 0 && pad_left == 0) {
        if (rtMemcpyAsync(reinterpret_cast<void *>(sram_in), reinterpret_cast<const void *>(dev_in), input_bytes,
                          rtMemcpyDeviceToSram, ctx.kernelStream) != rtSuccess) {
            throw std::runtime_error("Failed to copy Conv input to SRAM");
        }
        return;
    }

    launch_fill_16bits_zero(ctx, sram_in, static_cast<int>(padded_bytes), threads, grid, params);
    for (int c = 0; c < channels; ++c) {
        const RPPdeviceptr src = dev_in + static_cast<RPPdeviceptr>(c) * height * width * bytes_per_element;
        const RPPdeviceptr dst =
            sram_in +
            static_cast<RPPdeviceptr>((c * padded_height + pad_top) * padded_width + pad_left) * bytes_per_element;
        if (rtMemcpy2DAsync(reinterpret_cast<void *>(dst), padded_width * bytes_per_element,
                            reinterpret_cast<const void *>(src), width * bytes_per_element, width * bytes_per_element,
                            height, rtMemcpyDeviceToSram, ctx.kernelStream) != rtSuccess) {
            throw std::runtime_error("Failed to copy padded Conv input to SRAM");
        }
    }
}

static void validate_conv_args(int in_channels,
                               int in_height,
                               int in_width,
                               int out_channels,
                               int kernel_h,
                               int kernel_w,
                               int stride_h,
                               int stride_w,
                               int pad_top,
                               int pad_left,
                               int pad_bottom,
                               int pad_right,
                               int dilation_h,
                               int dilation_w,
                               int input_bytes_per_element,
                               int output_bytes_per_element) {
    if (in_channels <= 0 || in_height <= 0 || in_width <= 0 || out_channels <= 0 || kernel_h <= 0 || kernel_w <= 0) {
        throw std::runtime_error("Invalid Conv shape");
    }
    if (stride_h <= 0 || stride_w <= 0) {
        throw std::runtime_error("Invalid Conv stride");
    }
    if (pad_top < 0 || pad_left < 0 || pad_bottom < 0 || pad_right < 0) {
        throw std::runtime_error("Conv padding must be non-negative");
    }
    if (dilation_h != 1 || dilation_w != 1) {
        throw std::runtime_error("Conv BF16 demo path currently supports dilation == 1 only");
    }
    if (input_bytes_per_element != static_cast<int>(sizeof(uint16_t)) &&
        input_bytes_per_element != static_cast<int>(sizeof(float))) {
        throw std::runtime_error("Conv input element size must be 2 or 4 bytes");
    }
    if (output_bytes_per_element != static_cast<int>(sizeof(uint16_t)) &&
        output_bytes_per_element != static_cast<int>(sizeof(float))) {
        throw std::runtime_error("Conv output element size must be 2 or 4 bytes");
    }
}

}  // namespace

int rpp_conv_bf16_output_dim(int input, int kernel, int stride, int pad_before, int pad_after, int dilation) {
    return output_dim(input, kernel, stride, pad_before, pad_after, dilation);
}

size_t rpp_conv_bf16_preformatted_weight_size(int in_channels, int out_channels, int kernel_h, int kernel_w) {
    if (in_channels <= 0 || out_channels <= 0 || kernel_h <= 0 || kernel_w <= 0) {
        throw std::runtime_error("Invalid Conv weight shape");
    }
    const int out_channels_round = round_up_int(out_channels, 32);
    if (use_hwc_path(in_channels)) {
        const int wc_round = round_up_int(kernel_w * in_channels, 8);
        return static_cast<size_t>(kernel_h) * wc_round * out_channels_round * sizeof(uint16_t);
    }
    const int in_channels_round = round_up_int(in_channels, 32);
    return static_cast<size_t>(kernel_h) * kernel_w * in_channels_round * out_channels_round * sizeof(uint16_t);
}

size_t rpp_conv_bf16_preformatted_bias_size(int out_channels) {
    if (out_channels <= 0) {
        throw std::runtime_error("Invalid Conv bias shape");
    }
    return static_cast<size_t>(round_up_int(out_channels, 32)) * sizeof(uint16_t);
}

void rpp_conv_bf16_preformat_weights(const void * weights_oihw,
                                     void *       weights_preformatted,
                                     int          in_channels,
                                     int          out_channels,
                                     int          kernel_h,
                                     int          kernel_w,
                                     int          input_bytes_per_element) {
    if (weights_oihw == nullptr || weights_preformatted == nullptr) {
        throw std::runtime_error("Conv weight preformat received null pointer");
    }
    if (input_bytes_per_element != static_cast<int>(sizeof(uint16_t)) &&
        input_bytes_per_element != static_cast<int>(sizeof(float))) {
        throw std::runtime_error("Conv weight element size must be 2 or 4 bytes");
    }

    const int out_channels_round = round_up_int(out_channels, 32);
    std::memset(weights_preformatted, 0,
                rpp_conv_bf16_preformatted_weight_size(in_channels, out_channels, kernel_h, kernel_w));
    auto * dst = static_cast<uint16_t *>(weights_preformatted);

    auto load_weight = [&](int o, int c, int h, int w) -> uint16_t {
        const size_t idx = ((static_cast<size_t>(o) * in_channels + c) * kernel_h + h) * kernel_w + w;
        if (input_bytes_per_element == static_cast<int>(sizeof(float))) {
            return float_to_bf16_rne(static_cast<const float *>(weights_oihw)[idx]);
        }
        return static_cast<const uint16_t *>(weights_oihw)[idx];
    };

    if (use_hwc_path(in_channels)) {
        const int wc_round = round_up_int(kernel_w * in_channels, 8);
        int       tn       = 1;
        if (in_channels == 3) {
            if (kernel_w == 7 && kernel_h == 7 && out_channels == 64) {
                tn = 2;
            } else if (kernel_w == 3 && kernel_h == 3 && out_channels <= 64) {
                tn = out_channels_round / 32;
            }
        }
        const int segment        = (out_channels_round + tn * 32 - 1) / (tn * 32);
        const int out_stride_tn  = 32 * 8;
        const int segment_stride = wc_round * kernel_h * tn * 32;
        for (int seg = 0; seg < segment; ++seg) {
            const int tn_in_segment = std::min(tn, out_channels_round / 32 - seg * tn);
            const int out_stride_s0 = tn_in_segment * out_stride_tn;
            const int out_stride_h  = (wc_round / 8) * out_stride_s0;
            const int out_base      = seg * segment_stride;
            for (int h = 0; h < kernel_h; ++h) {
                for (int w = 0; w < kernel_w; ++w) {
                    for (int tn_idx = 0; tn_idx < tn_in_segment; ++tn_idx) {
                        const int out_channel_base   = (seg * tn + tn_idx) * 32;
                        const int out_segment_offset = out_base + h * out_stride_h + tn_idx * out_stride_tn;
                        for (int c = 0; c < in_channels; ++c) {
                            const int wc         = w * in_channels + c;
                            const int s0         = wc / 8;
                            const int s1         = wc % 8;
                            const int out_offset = out_segment_offset + s0 * out_stride_s0 + s1 * 32;
                            for (int lane = 0; lane < 32; ++lane) {
                                const int o            = out_channel_base + lane;
                                dst[out_offset + lane] = o < out_channels ? load_weight(o, c, h, w) : 0;
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    const int  in_channels_round = round_up_int(in_channels, 32);
    const int  cn                = in_channels_round / 32;
    const int  tn                = out_channels_round / 32;
    uint16_t * out               = dst;
    for (int cn_idx = 0; cn_idx < cn; ++cn_idx) {
        for (int h = 0; h < kernel_h; ++h) {
            for (int w = 0; w < kernel_w; ++w) {
                for (int s0 = 0; s0 < 4; ++s0) {
                    for (int tn_idx = 0; tn_idx < tn; ++tn_idx) {
                        for (int s1 = 0; s1 < 8; ++s1) {
                            const int c = cn_idx * 32 + s0 * 8 + s1;
                            for (int lane = 0; lane < 32; ++lane) {
                                const int o = tn_idx * 32 + lane;
                                *out++      = (c < in_channels && o < out_channels) ? load_weight(o, c, h, w) : 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

void rpp_conv_bf16_preformat_bias(const void * bias,
                                  void *       bias_preformatted,
                                  int          out_channels,
                                  int          input_bytes_per_element) {
    if (bias_preformatted == nullptr) {
        throw std::runtime_error("Conv bias preformat received null destination");
    }
    if (input_bytes_per_element != static_cast<int>(sizeof(uint16_t)) &&
        input_bytes_per_element != static_cast<int>(sizeof(float))) {
        throw std::runtime_error("Conv bias element size must be 2 or 4 bytes");
    }
    auto *    dst                = static_cast<uint16_t *>(bias_preformatted);
    const int out_channels_round = round_up_int(out_channels, 32);
    std::fill(dst, dst + out_channels_round, static_cast<uint16_t>(0));
    if (bias == nullptr) {
        return;
    }
    for (int o = 0; o < out_channels; ++o) {
        if (input_bytes_per_element == static_cast<int>(sizeof(float))) {
            dst[o] = float_to_bf16_rne(static_cast<const float *>(bias)[o]);
        } else {
            dst[o] = static_cast<const uint16_t *>(bias)[o];
        }
    }
}

void rpp_conv_bf16_build(rpp_kernel_context & ctx,
                         int                  in_channels,
                         int                  in_height,
                         int                  in_width,
                         int                  out_channels,
                         int                  kernel_h,
                         int                  kernel_w,
                         int                  stride_h,
                         int                  stride_w,
                         int                  pad_top,
                         int                  pad_left,
                         int                  pad_bottom,
                         int                  pad_right,
                         int                  dilation_h,
                         int                  dilation_w,
                         int                  input_bytes_per_element,
                         int                  output_bytes_per_element,
                         bool                 has_bias,
                         int                  is_instantial = 1) {
    validate_conv_args(in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_top,
                       pad_left, pad_bottom, pad_right, dilation_h, dilation_w, input_bytes_per_element,
                       output_bytes_per_element);
    if (ctx.dev_in.size() < (has_bias ? 3u : 2u) || ctx.dev_out.empty()) {
        throw std::runtime_error("Conv BF16 requires input, preformatted weights, optional bias, and one output");
    }

    const int out_height = output_dim(in_height, kernel_h, stride_h, pad_top, pad_bottom, dilation_h);
    const int out_width  = output_dim(in_width, kernel_w, stride_w, pad_left, pad_right, dilation_w);
    if (out_height <= 0 || out_width <= 0) {
        throw std::runtime_error("Invalid Conv output shape");
    }

    const bool hwc32                = !use_hwc_path(in_channels);
    const int  padded_height        = in_height + pad_top + pad_bottom;
    const int  padded_width         = in_width + pad_left + pad_right;
    const int  in_channels_internal = hwc32 ? round_up_int(in_channels, 32) : in_channels;
    const int  out_channels_round   = round_up_int(out_channels, 32);

    const size_t padded_input_elems   = static_cast<size_t>(in_channels) * padded_height * padded_width;
    const size_t input_external_bytes = padded_input_elems * input_bytes_per_element;
    const size_t input_bf16_bytes     = padded_input_elems * sizeof(uint16_t);
    const size_t input_internal_bytes =
        static_cast<size_t>(in_channels_internal) * padded_height * padded_width * sizeof(uint16_t);
    const size_t weight_bytes = rpp_conv_bf16_preformatted_weight_size(in_channels, out_channels, kernel_h, kernel_w);
    const size_t bias_bytes   = has_bias ? rpp_conv_bf16_preformatted_bias_size(out_channels) : 0;
    const size_t output_elems = static_cast<size_t>(out_channels) * out_height * out_width;
    const size_t output_chw_bf16_bytes = output_elems * sizeof(uint16_t);
    const size_t output_hwc32_bytes =
        static_cast<size_t>(out_channels_round) * out_height * out_width * sizeof(uint16_t);
    const size_t output_hwc33_bytes =
        static_cast<size_t>(out_channels_round / 32) * 33 * out_height * out_width * sizeof(uint16_t);
    const size_t output_external_bytes = output_elems * output_bytes_per_element;

    const size_t linear_workspace_bytes = std::max(input_external_bytes, output_chw_bf16_bytes);
    const size_t scratch_a_bytes =
        std::max({ input_bytes_per_element == static_cast<int>(sizeof(float)) ? input_bf16_bytes : (size_t) 0,
                   output_hwc32_bytes,
                   output_bytes_per_element == static_cast<int>(sizeof(float)) ? output_external_bytes : (size_t) 0 });
    const size_t scratch_b_bytes = std::max(input_internal_bytes, output_hwc33_bytes);

    RPPdeviceptr sram_linear = ctx.virtual_sram_base;
    RPPdeviceptr cursor      = sram_linear + static_cast<RPPdeviceptr>(sram_round(linear_workspace_bytes));
    RPPdeviceptr sram_scratch_a = cursor;
    cursor += static_cast<RPPdeviceptr>(sram_round(scratch_a_bytes));
    RPPdeviceptr sram_scratch_b = cursor;
    cursor += static_cast<RPPdeviceptr>(sram_round(scratch_b_bytes));
    RPPdeviceptr sram_weights = cursor;
    cursor += static_cast<RPPdeviceptr>(sram_round(weight_bytes));
    RPPdeviceptr sram_bias = cursor;
    cursor += static_cast<RPPdeviceptr>(sram_round(bias_bytes));

    RPPdeviceptr sram_input_bf16     = input_bytes_per_element == static_cast<int>(sizeof(float)) ? sram_scratch_a : sram_linear;
    RPPdeviceptr sram_input_internal = sram_scratch_b;
    RPPdeviceptr sram_output_hwc32   = sram_scratch_a;
    RPPdeviceptr sram_output_hwc33   = sram_scratch_b;
    RPPdeviceptr sram_output_io      = sram_scratch_a;

    const size_t conv_sram_bytes = static_cast<size_t>(cursor - ctx.virtual_sram_base);
    if (conv_sram_bytes > kSramLimitBytes) {
        throw std::runtime_error("Conv BF16 SRAM workspace overflow");
    }

    if (rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL) != RPP_SUCCESS) {
        throw std::runtime_error("rppStreamBeginCapture failed");
    }
    if (rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/conv_bf16.o") != RPP_SUCCESS) {
        throw std::runtime_error("rppModuleLoad failed for conv_bf16");
    }

    dim3                  threads{};
    dim3                  grid{};
    std::vector<uint32_t> params;

    copy_input_to_sram(ctx, ctx.dev_in[0], sram_linear, in_channels, in_height, in_width, padded_height, padded_width,
                       pad_top, pad_left, input_bytes_per_element, threads, grid, params);

    if (input_bytes_per_element == static_cast<int>(sizeof(float))) {
        params.clear();
        calc_tbdim_flattern(1, static_cast<uint32_t>(padded_input_elems), threads, grid);
        cvt_kernel_param_init(threads, sram_linear, sram_input_bf16, kFLOAT, kBF16, params);
        launchWrapperAysnc("opt_vector_cvt_32_16", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);
    }

    if (hwc32) {
        launch_chw2hwc32(ctx, sram_input_bf16, sram_input_internal, in_channels, padded_height, padded_width, threads,
                         grid, params);
    } else {
        launch_chw2hwc(ctx, sram_input_bf16, sram_input_internal, in_channels, padded_height, padded_width, threads,
                       grid, params);
    }

    if (rtMemcpyAsync(reinterpret_cast<void *>(sram_weights), reinterpret_cast<const void *>(ctx.dev_in[1]),
                      weight_bytes, rtMemcpyDeviceToSram, ctx.kernelStream) != rtSuccess) {
        throw std::runtime_error("Failed to copy Conv weights to SRAM");
    }
    if (has_bias) {
        if (rtMemcpyAsync(reinterpret_cast<void *>(sram_bias), reinterpret_cast<const void *>(ctx.dev_in[2]),
                          bias_bytes, rtMemcpyDeviceToSram, ctx.kernelStream) != rtSuccess) {
            throw std::runtime_error("Failed to copy Conv bias to SRAM");
        }
    }

    launch_conv(ctx, sram_input_internal, sram_weights, has_bias ? sram_bias : 0, sram_output_hwc32, in_channels,
                padded_height, padded_width, out_channels, out_height, out_width, kernel_h, kernel_w, stride_h,
                stride_w, hwc32, has_bias);
    if (out_height * out_width >= 128) {
        launch_hwc322hwc33(ctx, sram_output_hwc32, sram_output_hwc33, out_channels, out_height, out_width, threads,
                           grid, params);
        launch_hwc332chw_opt1(ctx, sram_output_hwc33, sram_linear, out_channels_round, out_height, out_width, threads,
                              grid, params);
    } else {
        launch_hwc322chw(ctx, sram_output_hwc32, sram_linear, out_channels, out_height, out_width, threads, grid,
                         params);
    }

    RPPdeviceptr output_copy_addr = sram_linear;
    if (output_bytes_per_element == static_cast<int>(sizeof(float))) {
        params.clear();
        calc_tbdim_flattern(1, static_cast<uint32_t>(output_elems * 2), threads, grid);
        cvt_kernel_param_init_opt(threads, sram_linear, sram_output_io, kBF16, kFLOAT, params);
        launchWrapperAysnc("opt_vector_cvt_f16_f32_v2", grid, threads, params, ctx.rppBinMod, ctx.kernelStream);
        output_copy_addr = sram_output_io;
    }

    if (rtMemcpyAsync(reinterpret_cast<void *>(ctx.dev_out[0]), reinterpret_cast<const void *>(output_copy_addr),
                      output_external_bytes, rtMemcpySramToDevice, ctx.kernelStream) != rtSuccess) {
        throw std::runtime_error("Failed to copy Conv output from SRAM");
    }

    if (rppStreamEndCapture(ctx.kernelStream, &ctx.graph) != RPP_SUCCESS) {
        throw std::runtime_error("rppStreamEndCapture failed");
    }
    // if (rppGraphInstantiate(&ctx.graphexec, ctx.graph, nullptr, nullptr, 0) != RPP_SUCCESS) {
    //     throw std::runtime_error("rppGraphInstantiate failed");
    // }
    const std::string graph_key = rpp_join_function_name_and_args(
        __func__, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, stride_h, stride_w, pad_top,
        pad_left, pad_bottom, pad_right, dilation_h, dilation_w, (int) input_bytes_per_element,
        (int) output_bytes_per_element, has_bias, is_instantial);
    if (rpp_graph_instantiate(ctx.graphexec, ctx.graph, graph_key.c_str(), is_instantial) != RPP_SUCCESS) {
        throw std::runtime_error("rpp_graph_instantiate failed.");
    }
}
