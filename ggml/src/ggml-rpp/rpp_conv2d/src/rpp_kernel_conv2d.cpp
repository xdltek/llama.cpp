#include "rpp_conv2d/rpp_conv2d.h"
#include "rpp_conv2d/src/rpp_kernel_conv2d_build.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

static constexpr const char * GGML_RPP_CONV2D_CPU_DATA_DIR =
    "/home/gaok/gaokao/Azurengine/test_framwork/test_llm_cpp/llama.cpp/build/bin/cpu_data";

static bool ggml_rpp_conv2d_load_cpu_reference(const char * basename, std::vector<float> & ref) {
    const std::string path = std::string(GGML_RPP_CONV2D_CPU_DATA_DIR) + "/" + basename + ".bin";
    std::ifstream     fin(path, std::ios::binary | std::ios::ate);
    if (!fin) {
        GGML_LOG_ERROR("%s: failed to open CPU reference: %s\n", __func__, path.c_str());
        return false;
    }

    const std::streamsize size = fin.tellg();
    if (size < 0 || size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        GGML_LOG_ERROR("%s: invalid CPU reference size: %s, bytes=%zu\n", __func__, path.c_str(), (size_t) size);
        return false;
    }
    fin.seekg(0, std::ios::beg);

    ref.resize((size_t) size / sizeof(float));
    if (!fin.read(reinterpret_cast<char *>(ref.data()), size)) {
        GGML_LOG_ERROR("%s: failed to read CPU reference: %s\n", __func__, path.c_str());
        return false;
    }

    return true;
}

static bool ggml_rpp_conv2d_copy_tensor_as_f32(const ggml_tensor * tensor,
                                               int                 bytes_per_element,
                                               rtStream_t          stream,
                                               std::vector<float> & data) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    const size_t         nelements = (size_t) ggml_nelements(tensor);
    const size_t         nbytes    = nelements * (size_t) bytes_per_element;
    std::vector<uint8_t> raw(nbytes);
    RPP_CHECK(rtMemcpyAsync(raw.data(), tensor->data, nbytes, rtMemcpyDeviceToHost, stream));
    RPP_CHECK(rtStreamSynchronize(stream));

    data.resize(nelements);
    switch (bytes_per_element) {
        case (int) sizeof(float):
            {
                const float * src = reinterpret_cast<const float *>(raw.data());
                std::copy(src, src + nelements, data.begin());
            }
            break;
        case (int) sizeof(uint16_t):
            {
                const ggml_bf16_t * src = reinterpret_cast<const ggml_bf16_t *>(raw.data());
                for (size_t i = 0; i < nelements; ++i) {
                    data[i] = ggml_bf16_to_fp32(src[i]);
                }
            }
            break;
        default:
            GGML_LOG_ERROR("%s: unsupported element size for MSE: %s bytes=%d type=%s\n", __func__, tensor->name,
                           bytes_per_element, ggml_type_name(tensor->type));
            return false;
    }

    return true;
}

static void ggml_rpp_conv2d_compare_tensor_with_cpu(const ggml_tensor * tensor,
                                                    const char *        basename,
                                                    int                 bytes_per_element,
                                                    rtStream_t          stream) {
    std::vector<float> ref;
    std::vector<float> got;
    if (!ggml_rpp_conv2d_load_cpu_reference(basename, ref) ||
        !ggml_rpp_conv2d_copy_tensor_as_f32(tensor, bytes_per_element, stream, got)) {
        return;
    }

    if (ref.size() != got.size()) {
        GGML_LOG_ERROR("%s: size mismatch for %s, cpu=%zu, rpp=%zu, rpp_tensor=%s\n", __func__, basename, ref.size(),
                       got.size(), tensor != nullptr ? tensor->name : "<null>");
        return;
    }

    double sum_sq      = 0.0;
    double ref_sq      = 0.0;
    float  max_abs     = 0.0f;
    size_t max_abs_idx = 0;
    float  max_abs_ref = 0.0f;
    float  max_abs_got = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = got[i] - ref[i];
        sum_sq += (double) diff * (double) diff;
        ref_sq += (double) ref[i] * (double) ref[i];
        const float abs_diff = std::fabs(diff);
        if (abs_diff > max_abs) {
            max_abs     = abs_diff;
            max_abs_idx = i;
            max_abs_ref = ref[i];
            max_abs_got = got[i];
        }
    }

    const double mse  = ref.empty() ? 0.0 : sum_sq / (double) ref.size();
    const double nmse = ref_sq > 0.0 ? sum_sq / ref_sq : (sum_sq == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    const int64_t x   = (int64_t) (max_abs_idx % (size_t) tensor->ne[0]);
    const int64_t y   = (int64_t) ((max_abs_idx / (size_t) tensor->ne[0]) % (size_t) tensor->ne[1]);
    const int64_t c   = (int64_t) ((max_abs_idx / ((size_t) tensor->ne[0] * (size_t) tensor->ne[1])) %
                                 (size_t) tensor->ne[2]);
    const int64_t n   = (int64_t) (max_abs_idx / ((size_t) tensor->ne[0] * (size_t) tensor->ne[1] *
                                                (size_t) tensor->ne[2]));
    GGML_LOG_INFO("%s: %s MSE=%.9g NMSE=%.9g max_abs=%.9g idx=%zu coord=[%lld,%lld,%lld,%lld] cpu=%.9g rpp=%.9g shape=[%lld,%lld,%lld,%lld] type=%s storage_bytes=%d\n",
                  __func__, basename, mse, nmse, (double) max_abs, max_abs_idx, (long long) x, (long long) y,
                  (long long) c, (long long) n, (double) max_abs_ref, (double) max_abs_got, (long long) tensor->ne[0],
                  (long long) tensor->ne[1], (long long) tensor->ne[2], (long long) tensor->ne[3],
                  ggml_type_name(tensor->type), bytes_per_element);
}

static void ggml_rpp_conv2d_compare_fused_io(ggml_backend_rpp_context & ctx, rpp_kernel_fused_conv2d * rpp_node) {
    if (rpp_node == nullptr) {
        return;
    }

    RPP_CHECK(rtStreamSynchronize(ctx.stream()));
    const int input_type_size  = ggml_rpp_get_io_type_size(ctx, rpp_node->input, 0);
    const int output_type_size = ggml_rpp_get_io_type_size(ctx, rpp_node->output, 1);
    ggml_rpp_conv2d_compare_tensor_with_cpu(rpp_node->input, "im2col_src1_inp_raw_scaled", input_type_size,
                                            ctx.stream());
    ggml_rpp_conv2d_compare_tensor_with_cpu(rpp_node->output, "conv_cont_dst", output_type_size, ctx.stream());
}

bool ggml_rpp_can_fuse_conv2d(const ggml_cgraph * cgraph,
                              int &               node_idx,
                              ggml_tensor *&      im2col,
                              ggml_tensor *&      mul_mat,
                              ggml_tensor *&      cont) {
    im2col  = nullptr;
    mul_mat = nullptr;
    cont    = nullptr;

    const int idx = node_idx;
    if (cgraph == nullptr || idx < 0 || idx + 6 >= cgraph->n_nodes) {
        return false;
    }

    ggml_tensor * const im2col_node = cgraph->nodes[idx + 0];
    if (im2col_node == nullptr || im2col_node->op != GGML_OP_IM2COL) {
        return false;
    }

    ggml_tensor * const im2col_reshape        = cgraph->nodes[idx + 1];
    ggml_tensor * const weight_reshape        = cgraph->nodes[idx + 2];
    ggml_tensor * const mul_mat_node          = cgraph->nodes[idx + 3];
    ggml_tensor * const mul_mat_reshape       = cgraph->nodes[idx + 4];
    ggml_tensor * const mul_mat_permute       = cgraph->nodes[idx + 5];
    ggml_tensor * const mul_mat_cont          = cgraph->nodes[idx + 6];
    // ggml_tensor * const output_reshape        = cgraph->nodes[idx + 7];
    // ggml_tensor * const output_transpose      = cgraph->nodes[idx + 8];
    // ggml_tensor * const output_transpose_cont = cgraph->nodes[idx + 9];

    // Gemma vision patch embedding is lowered as:
    // IM2COL -> RESHAPE, weight RESHAPE -> MUL_MAT -> RESHAPE -> PERMUTE -> CONT -> RESHAPE -> TRANSPOSE -> CONT.
    // The conv kernel produces the first CONT ([OW, OH, OC, N]); the trailing RESHAPE/TRANSPOSE/CONT keeps running.
    if (im2col_reshape == nullptr || im2col_reshape->op != GGML_OP_RESHAPE || weight_reshape == nullptr ||
        weight_reshape->op != GGML_OP_RESHAPE || mul_mat_node == nullptr || mul_mat_node->op != GGML_OP_MUL_MAT ||
        mul_mat_reshape == nullptr || mul_mat_reshape->op != GGML_OP_RESHAPE || mul_mat_permute == nullptr ||
        mul_mat_permute->op != GGML_OP_PERMUTE || mul_mat_cont == nullptr || mul_mat_cont->op != GGML_OP_CONT /*||
        output_reshape == nullptr || output_reshape->op != GGML_OP_RESHAPE || output_transpose == nullptr ||
        output_transpose->op != GGML_OP_TRANSPOSE || output_transpose_cont == nullptr ||
        output_transpose_cont->op != GGML_OP_CONT*/) {
        return false;
    }

    if (im2col_node->src[0] == nullptr || im2col_node->src[1] == nullptr) {
        return false;
    }

    if (im2col_reshape->src[0] != im2col_node || weight_reshape->src[0] == nullptr) {
        return false;
    }

    const bool mul_src0_is_im2col = mul_mat_node->src[0] == im2col_reshape;
    const bool mul_src1_is_im2col = mul_mat_node->src[1] == im2col_reshape;
    const bool mul_src0_is_weight = mul_mat_node->src[0] == weight_reshape;
    const bool mul_src1_is_weight = mul_mat_node->src[1] == weight_reshape;
    if (!((mul_src0_is_im2col && mul_src1_is_weight) || (mul_src1_is_im2col && mul_src0_is_weight))) {
        return false;
    }

    if (mul_mat_reshape->src[0] != mul_mat_node || mul_mat_permute->src[0] != mul_mat_reshape ||
        mul_mat_cont->src[0] != mul_mat_permute /*|| output_reshape->src[0] != mul_mat_cont ||
        output_transpose->src[0] != output_reshape || output_transpose_cont->src[0] != output_transpose*/) {
        return false;
    }

    for (int i = 0; i < 6; ++i) {
        const ggml_tensor * node            = cgraph->nodes[idx + i];
        const bool          is_compute_node = node->op == GGML_OP_IM2COL || node->op == GGML_OP_MUL_MAT;
        if (ggml_node_get_use_count(cgraph, idx + i) != 1 || (is_compute_node && node->view_src != nullptr) ||
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
    }

    im2col   = im2col_node;
    mul_mat  = mul_mat_node;
    cont     = mul_mat_cont;
    node_idx = idx + 6;
    return true;
}

static ggml_tensor * ggml_rpp_fused_conv2d_get_weight(const ggml_tensor * im2col, const ggml_tensor * mul_mat) {
    if (im2col == nullptr || mul_mat == nullptr || mul_mat->op != GGML_OP_MUL_MAT) {
        return nullptr;
    }

    for (int i = 0; i < 2; ++i) {
        const ggml_tensor * src = mul_mat->src[i];
        if (src == nullptr || src->op != GGML_OP_RESHAPE || src->src[0] == nullptr) {
            continue;
        }

        if (src->src[0] != im2col) {
            return src->src[0];
        }
    }

    return nullptr;
}

static bool ggml_rpp_fused_conv2d_properties_is_same(ggml_backend_rpp_context & ctx,
                                                     ggml_tensor *              dst,
                                                     ggml_rpp_node *            rpp_node) {
    GGML_ASSERT(rpp_node);
    if (dst != rpp_node->cur_ggml_tensor) {
        return false;
    }
    if (!rpp_node->ggml_node_properties.size()) {
        return false;
    }
    if (!rpp_node->ggml_node_properties.count(dst)) {
        return false;
    }

    auto & node                  = dst;
    auto & graph_node_properties = rpp_node->ggml_node_properties[node];
    if (node->data != graph_node_properties.node_address && node->op != GGML_OP_CPY && node->op != GGML_OP_VIEW) {
        return false;
    }
    if (node->op != graph_node_properties.node_op) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties.ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties.nb[i]) {
            return false;
        }
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] && node->src[i]->data != graph_node_properties.src_address[i] && node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW) {
            return false;
        }
    }
    if (node->op == GGML_OP_SCALE &&
        memcmp(graph_node_properties.op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

static bool ggml_rpp_create_kernel_fused_conv2d(ggml_backend_rpp_context & ctx,
                                                ggml_rpp_node *            rpp_base_node,
                                                ggml_tensor *              im2col,
                                                ggml_tensor *              mul_mat,
                                                ggml_tensor *              cont) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_fused_conv2d *>(rpp_base_node);
    if (im2col == nullptr || mul_mat == nullptr || cont == nullptr || im2col->op != GGML_OP_IM2COL ||
        mul_mat->op != GGML_OP_MUL_MAT || cont->op != GGML_OP_CONT) {
        GGML_LOG_ERROR("%s: invalid fused_conv2d tensors\n", __func__);
        return false;
    }
    GGML_ASSERT(ggml_is_contiguous(im2col->src[1]));
    GGML_ASSERT(ggml_is_contiguous(mul_mat->src[0]));
    GGML_ASSERT(ggml_is_contiguous(mul_mat->src[1]));
    GGML_ASSERT(ggml_is_contiguous(cont));

    ggml_tensor * input  = im2col->src[1];
    ggml_tensor * weight = ggml_rpp_fused_conv2d_get_weight(im2col, mul_mat);
    if (input == nullptr || weight == nullptr) {
        GGML_LOG_ERROR("%s: failed to parse fused_conv2d input/weight from %s and %s\n", __func__, im2col->name,
                       mul_mat->name);
        return false;
    }

    rpp_node->im2col  = im2col;
    rpp_node->mul_mat = mul_mat;
    rpp_node->input   = input;
    rpp_node->weight  = weight;
    rpp_node->output  = cont;

    rpp_node->params.stride_x   = ggml_get_op_params_i32(im2col, 0);
    rpp_node->params.stride_y   = ggml_get_op_params_i32(im2col, 1);
    rpp_node->params.padding_x  = ggml_get_op_params_i32(im2col, 2);
    rpp_node->params.padding_y  = ggml_get_op_params_i32(im2col, 3);
    rpp_node->params.dilation_x = ggml_get_op_params_i32(im2col, 4);
    rpp_node->params.dilation_y = ggml_get_op_params_i32(im2col, 5);
    rpp_node->params.is_2d      = ggml_get_op_params_i32(im2col, 6) != 0;
    rpp_node->params.input_w    = input->ne[0];
    rpp_node->params.input_h    = input->ne[1];
    rpp_node->params.input_c    = input->ne[2];
    rpp_node->params.input_n    = input->ne[3];
    rpp_node->params.kernel_w   = weight->ne[0];
    rpp_node->params.kernel_h   = weight->ne[1];
    rpp_node->params.kernel_c   = weight->ne[2];
    rpp_node->params.kernel_n   = weight->ne[3];
    rpp_node->params.output_w   = im2col->ne[1];
    rpp_node->params.output_h   = im2col->ne[2];
    rpp_node->params.output_c   = weight->ne[3];
    rpp_node->params.output_n   = input->ne[3];

    GGML_ASSERT(rpp_node->params.is_2d);
    GGML_ASSERT(input->data != nullptr);
    GGML_ASSERT(weight->data != nullptr);
    GGML_ASSERT(cont->data != nullptr);

    // kernel inputs: image tensor and patch embedding weight
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) input->data);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) weight->data);
    // kernel outputs: contiguous conv result after the fused reshape/permute chain
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) cont->data);
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(input, input->data);
    rpp_node->binding_i_buffers.emplace(weight, weight->data);
    rpp_node->binding_o_buffers.emplace(cont, cont->data);
    rpp_node->binding_io_buffers.emplace_back(input->data);
    rpp_node->binding_io_buffers.emplace_back(weight->data);
    rpp_node->binding_io_buffers.emplace_back(cont->data);

    const int input_type_size  = ggml_rpp_get_io_type_size(ctx, input, 0);
    const int output_type_size = ggml_rpp_get_io_type_size(ctx, cont, 1);
    GGML_LOG_INFO(
        "%s: fused_conv2d params in=[c=%lld,h=%lld,w=%lld,n=%lld] weight=[kw=%lld,kh=%lld,ic=%lld,oc=%lld] out=[w=%lld,h=%lld,c=%lld,n=%lld] stride=[x=%d,y=%d] pad=[x=%d,y=%d] dilation=[x=%d,y=%d] io_bytes=[in=%d,out=%d]\n",
        __func__, (long long) rpp_node->params.input_c, (long long) rpp_node->params.input_h,
        (long long) rpp_node->params.input_w, (long long) rpp_node->params.input_n,
        (long long) rpp_node->params.kernel_w, (long long) rpp_node->params.kernel_h,
        (long long) rpp_node->params.kernel_c, (long long) rpp_node->params.kernel_n,
        (long long) rpp_node->params.output_w, (long long) rpp_node->params.output_h,
        (long long) rpp_node->params.output_c, (long long) rpp_node->params.output_n,
        rpp_node->params.stride_x, rpp_node->params.stride_y, rpp_node->params.padding_x,
        rpp_node->params.padding_y, rpp_node->params.dilation_x, rpp_node->params.dilation_y, input_type_size,
        output_type_size);
    // build fused_conv2d kernel
    rpp_conv_bf16_build(*(rpp_node->kernel_ctx.get()), static_cast<int>(rpp_node->params.input_c),
                        static_cast<int>(rpp_node->params.input_h), static_cast<int>(rpp_node->params.input_w),
                        static_cast<int>(rpp_node->params.kernel_n), static_cast<int>(rpp_node->params.kernel_h),
                        static_cast<int>(rpp_node->params.kernel_w), rpp_node->params.stride_y,
                        rpp_node->params.stride_x, rpp_node->params.padding_y, rpp_node->params.padding_x,
                        rpp_node->params.padding_y, rpp_node->params.padding_x, rpp_node->params.dilation_y,
                        rpp_node->params.dilation_x, input_type_size, output_type_size, false, rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              im2col,
                                            ggml_tensor *              mul_mat,
                                            ggml_tensor *              cont) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_fused_conv2d *>(rpp_base_node);
    bool ret      = false;
    GGML_ASSERT(ctx.use_ubatch == false);
    ret = ggml_rpp_create_kernel_fused_conv2d(ctx, rpp_node, im2col, mul_mat, cont);
    GGML_ASSERT(ret);
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, im2col);
        ggml_rpp_node_set_properties(rpp_node, mul_mat);
        ggml_rpp_node_set_properties(rpp_node, cont);
        rpp_node->binding_io_tensors.emplace_back(im2col);
        rpp_node->binding_io_tensors.emplace_back(mul_mat);
        rpp_node->binding_io_tensors.emplace_back(cont);
    }
    return ret;
}

bool ggml_rpp_op_kernel_fused_conv2d(ggml_backend_rpp_context & ctx,
                                     ggml_tensor *              im2col,
                                     ggml_tensor *              mul_mat,
                                     ggml_tensor *              cont,
                                     int                        is_instantial,
                                     int                        is_launch) {
    if (!im2col || !mul_mat || !cont) {
        GGML_LOG_ERROR("%s: fused_conv2d tensor is nullptr\n", __func__);
        return false;
    }
    rpp_kernel_fused_conv2d * rpp_node = nullptr;
    auto                      iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(im2col);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(im2col);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_fused_conv2d");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[im2col];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    cur_node->op == ggml_rpp_node::RPP_OP_FUSED_CON2D &&
                    ggml_rpp_fused_conv2d_properties_is_same(ctx, cont, cur_node)) {
                    rpp_node = (rpp_kernel_fused_conv2d *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_fused_conv2d");
            auto new_node = std::make_unique<rpp_kernel_fused_conv2d>(im2col);
            ctx.cur_rpp_graph->rpp_nodes[im2col].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_fused_conv2d *) (ctx.cur_rpp_graph->rpp_nodes[im2col].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, im2col, mul_mat, cont))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[im2col] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_fused_conv2d *) (iter->second);
    }

    if (is_launch) {
        // compute fused_conv2d operator
        if (rpp_node->kernel_ctx->graphexec == nullptr) {
            return true;
        }
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_fused_conv2d");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
            ggml_rpp_conv2d_compare_fused_io(ctx, rpp_node);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, cont->name, ggml_op_name(cont->op),
                           e.what());
        }
    }
    return true;
}
