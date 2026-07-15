#ifndef RPP_CON2D_H
#define RPP_CON2D_H

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

bool ggml_rpp_can_fuse_conv2d(const ggml_cgraph * cgraph,
                              int &               node_idx,
                              ggml_tensor *&      im2col,
                              ggml_tensor *&      mul_mat,
                              ggml_tensor *&      cont);

bool ggml_rpp_op_kernel_fused_conv2d(ggml_backend_rpp_context & ctx,
                                     ggml_tensor *              im2col,
                                     ggml_tensor *              mul_mat,
                                     ggml_tensor *              cont,
                                     int                        is_instantial = 1,
                                     int                        is_launch     = 1);

size_t rpp_conv_bf16_preformatted_weight_size(int in_channels, int out_channels, int kernel_h, int kernel_w);

void rpp_conv_bf16_preformat_weights(const void * weights_oihw,
                                     void *       weights_preformatted,
                                     int          in_channels,
                                     int          out_channels,
                                     int          kernel_h,
                                     int          kernel_w,
                                     int          input_bytes_per_element);

struct rpp_kernel_fused_conv2d : public rpp_node_kernel {
    struct conv2d_params {
        int32_t stride_x   = 1;
        int32_t stride_y   = 1;
        int32_t padding_x  = 0;
        int32_t padding_y  = 0;
        int32_t dilation_x = 1;
        int32_t dilation_y = 1;
        bool    is_2d      = true;

        int64_t input_w  = 0;
        int64_t input_h  = 0;
        int64_t input_c  = 0;
        int64_t input_n  = 0;
        int64_t kernel_w = 0;
        int64_t kernel_h = 0;
        int64_t kernel_c = 0;
        int64_t kernel_n = 0;
        int64_t output_w = 0;
        int64_t output_h = 0;
        int64_t output_c = 0;
        int64_t output_n = 0;
    };

    explicit rpp_kernel_fused_conv2d(ggml_tensor * tensor) : rpp_node_kernel(tensor) { op = RPP_OP_FUSED_CON2D; }

    explicit rpp_kernel_fused_conv2d(ggml_tensor * tensor, ggml_rpp_node * rpp_node) :
        rpp_node_kernel(tensor, rpp_node) {
        op = RPP_OP_FUSED_CON2D;
    }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        GGML_UNUSED(dst);
        return ggml_rpp_op_kernel_fused_conv2d(ctx, im2col, mul_mat, output, is_instantial, is_launch);
    }

    ggml_tensor * im2col  = nullptr;
    ggml_tensor * mul_mat = nullptr;
    ggml_tensor * input   = nullptr;
    ggml_tensor * weight  = nullptr;
    ggml_tensor * output  = nullptr;

    conv2d_params params;
};

inline bool ggml_rpp_op_fused_conv2d(ggml_backend_rpp_context & ctx,
                                     ggml_tensor *              im2col,
                                     ggml_tensor *              mul_mat,
                                     ggml_tensor *              cont,
                                     int                        is_instantial = 1,
                                     int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_fused_conv2d(ctx, im2col, mul_mat, cont, is_instantial, is_launch);
}

#endif  // RPP_CON2D_H
