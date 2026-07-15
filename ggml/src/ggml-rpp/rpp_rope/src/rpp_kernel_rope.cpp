#include "rpp_rope/rpp_rope.h"
#include "rpp_rope/src/rpp_kernel_build.h"

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(float   theta_extrap,
                      float   freq_scale,
                      float   corr_dims[2],
                      int64_t i0,
                      float   ext_factor,
                      float   mscale,
                      float * cos_theta,
                      float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta        = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta          = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static void ggml_rope_cache_init(float         theta_base,
                                 float         freq_scale,
                                 const float * freq_factors,
                                 float         corr_dims[2],
                                 int64_t       ne0,
                                 float         ext_factor,
                                 float         mscale,
                                 float *       cache,
                                 float         sin_sign,
                                 float         theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

static void ggml_mrope_cache_init(float         theta_base_t,
                                  float         theta_base_h,
                                  float         theta_base_w,
                                  float         theta_base_e,
                                  int           sections[4],
                                  bool          is_imrope,
                                  bool          indep_sects,
                                  float         freq_scale,
                                  const float * freq_factors,
                                  float         corr_dims[2],
                                  int64_t       ne0,
                                  float         ext_factor,
                                  float         mscale,
                                  float *       cache,
                                  float         sin_sign,
                                  float         theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta_t   = theta_base_t;
    float theta_h   = theta_base_h;
    float theta_w   = theta_base_w;
    float theta_e   = theta_base_e;  // extra position id for vision encoder
    int   sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    int   sec_w     = sections[1] + sections[0];
    int   sec_e     = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff     = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        int         sector = (i0 / 2) % sect_dims;
        if (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            if (sector == 0) {
                theta_t = theta_base_t;
            } else if (sector == sections[0]) {
                theta_h = theta_base_h;
                ;
            } else if (sector == sec_w) {
                theta_w = theta_base_w;
            } else if (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        float theta = theta_t;
        if (is_imrope) {  // qwen3vl applies interleaved mrope
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } else {
                theta = theta_e;
            }
        } else {
            if (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            } else if (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            } else if (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]);
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}

static inline void push_u32_le(std::vector<int8_t> & out, uint32_t v) {
    out.push_back(static_cast<int8_t>(v & 0xFF));
    out.push_back(static_cast<int8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<int8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<int8_t>((v >> 24) & 0xFF));
}

static void ggml_cpu_rope_sincos_f32_calc(ggml_backend_rpp_context & ctx,
                                          const ggml_tensor *        dst,
                                          float *                    cos,
                                          float *                    sin) {
    bool                forward = true;
    const ggml_tensor * src0    = dst->src[0];
    const ggml_tensor * src1    = dst->src[1];
    const ggml_tensor * src2    = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int   sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections, (int32_t *) dst->op_params + 11, sizeof(int) * 4);

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;  // qwen3vl applies interleaved mrope
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;    // ggml_rope_multi, multimodal rotary position embedding
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne0 / 2);
    }

    std::vector<int8_t> freq_factors_data;
    const float *       freq_factors = NULL;
    if (src2 != NULL) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        size_t freq_factors_size = ggml_nbytes(src2);
        freq_factors_data.resize(freq_factors_size);
        RPP_CHECK(rtMemcpy(freq_factors_data.data(), src2->data, freq_factors_size, rtMemcpyDeviceToHost));
        freq_factors = (const float *) freq_factors_data.data();
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const float         sin_sign = forward ? 1.0f : -1.0f;
    std::vector<int8_t> pos_data;
    const size_t        pos_sections = is_mrope ? 4 : 1;
    pos_data.reserve(ctx.n_max_ctx * pos_sections * sizeof(uint32_t));
    for (size_t section = 0; section < pos_sections; ++section) {
        for (uint32_t i = 0; i < ctx.n_max_ctx; ++i) {
            push_u32_le(pos_data, i);
        }
    }

    const int32_t * pos = (const int32_t *) pos_data.data();

    size_t max_seq_len = ctx.n_max_ctx;
    for (int64_t i3 = 0; i3 < ne3; i3++) {              // batch
        for (int64_t i2 = 0; i2 < max_seq_len; i2++) {  // seq-len

            // float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            std::vector<float> cache(ne0);
            if (!is_mrope) {
                const int64_t p = pos[i2];
                ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache.data(),
                                     sin_sign, theta_scale);
            } else {
                const int64_t p_t = pos[i2];
                const int64_t p_h = pos[i2 + max_seq_len];
                const int64_t p_w = pos[i2 + max_seq_len * 2];
                const int64_t p_e = pos[i2 + max_seq_len * 3];
                ggml_mrope_cache_init(p_t, p_h, p_w, p_e, sections, is_imrope, is_vision, freq_scale, freq_factors,
                                      corr_dims, ne0, ext_factor, attn_factor, cache.data(), sin_sign, theta_scale);
            }
            const int64_t num = ne0 / 2;
            for (int64_t j = 0; j < num; j++) {
                cos[i2 * ne0 + j] = cos[i2 * ne0 + j + num] = cache[2 * j];
                sin[i2 * ne0 + j] = sin[i2 * ne0 + j + num] = cache[2 * j + 1];
            }
        }
    }
}

static void ggml_cpu_rope_sincos_f32(const ggml_tensor * dst, float * cos, float * sin) {
    bool                forward = true;
    const ggml_tensor * src0    = dst->src[0];
    const ggml_tensor * src1    = dst->src[1];
    const ggml_tensor * src2    = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    int   sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base, (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale, (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast, (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow, (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections, (int32_t *) dst->op_params + 11, sizeof(int) * 4);

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;  // qwen3vl applies interleaved mrope
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;    // ggml_rope_multi, multimodal rotary position embedding
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne0 / 2);
    }

    std::vector<int8_t> freq_factors_data;
    const float *       freq_factors = NULL;
    if (src2 != NULL) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        size_t freq_factors_size = ggml_nbytes(src2);
        freq_factors_data.resize(freq_factors_size);
        RPP_CHECK(rtMemcpy(freq_factors_data.data(), src2->data, freq_factors_size, rtMemcpyDeviceToHost));
        freq_factors = (const float *) freq_factors_data.data();
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const float         sin_sign = forward ? 1.0f : -1.0f;
    size_t              pos_size = ggml_nbytes(src1);
    std::vector<int8_t> pos_data(pos_size);
    RPP_CHECK(rtMemcpy(pos_data.data(), src1->data, pos_size, rtMemcpyDeviceToHost));
    const int32_t * pos = (const int32_t *) pos_data.data();

    for (int64_t i3 = 0; i3 < ne3; i3++) {      // batch
        for (int64_t i2 = 0; i2 < ne2; i2++) {  // seq-len

            // float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            std::vector<float> cache(ne0);
            if (!is_mrope) {
                const int64_t p = pos[i2];
                ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache.data(),
                                     sin_sign, theta_scale);
            } else {
                const int64_t p_t = pos[i2];
                const int64_t p_h = pos[i2 + ne2];
                const int64_t p_w = pos[i2 + ne2 * 2];
                const int64_t p_e = pos[i2 + ne2 * 3];
                ggml_mrope_cache_init(p_t, p_h, p_w, p_e, sections, is_imrope, is_vision, freq_scale, freq_factors,
                                      corr_dims, ne0, ext_factor, attn_factor, cache.data(), sin_sign, theta_scale);
            }
            const int64_t num = ne0 / 2;
            for (int64_t j = 0; j < num; j++) {
                cos[i2 * ne0 + j] = cos[i2 * ne0 + j + num] = cache[2 * j];
                sin[i2 * ne0 + j] = sin[i2 * ne0 + j + num] = cache[2 * j + 1];
            }
        }
    }
}

static void ggml_cpu_rope_sincos_f16(const ggml_tensor * dst, float * cos, float * sin) {}

static void ggml_cpu_rope_get_sincos(const ggml_tensor * dst, void * sin, void * cos) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F16:
            {
                ggml_cpu_rope_sincos_f16(dst, (float *) cos, (float *) sin);
            }
            break;
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
            {
                ggml_cpu_rope_sincos_f32(dst, (float *) cos, (float *) sin);
            }
            break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static int ggml_rpp_rope_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_should_compare_rope(const ggml_tensor * dst) {
    const char * name = getenv("GGML_RPP_COMPARE_ROPE_NAME");
    return name != nullptr && name[0] != '\0' && std::string(dst->name) == name;
}

static bool ggml_rpp_rope_uses_internal_bf16(const ggml_backend_rpp_context & ctx, const ggml_tensor * tensor) {
    return ctx.use_bf16 && tensor && tensor->type == GGML_TYPE_F32;
}

static size_t ggml_rpp_rope_device_type_size(const ggml_backend_rpp_context & ctx, const ggml_tensor * tensor) {
    if (ggml_rpp_rope_uses_internal_bf16(ctx, tensor)) {
        return sizeof(ggml_bf16_t);
    }
    return ggml_type_size(tensor->type);
}

static bool ggml_rpp_rope_is_k_shift(const ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_ROPE || !tensor->src[0] || !tensor->src[1]) {
        return false;
    }
    if (tensor->src[1]->type != GGML_TYPE_I32 || tensor->src[1]->ne[0] != tensor->ne[2]) {
        return false;
    }

    const ggml_tensor * src0_base = tensor->src[0]->view_src ? tensor->src[0]->view_src : tensor->src[0];
    const std::string   name      = ggml_get_name(src0_base);
    return name.find("cache_k") != std::string::npos;
}

static bool ggml_rpp_rope_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
    bool dims_is_same = true;
    if (rpp_node->ggml_node_properties.count(src)) {
        auto & property = rpp_node->ggml_node_properties[src];
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (dst->ne[i] != property.ne[i]) {
                dims_is_same = false;
                break;
            }
        }
    } else {
        dims_is_same = false;
    }
    return dims_is_same;
}

static ggml_rpp_node * ggml_rpp_find_rope_node(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    const bool dst_is_k_shift = ggml_rpp_rope_is_k_shift(dst);
    if (dst_is_k_shift) {
        return nullptr;
    }

    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_ROPE) {
                if (ggml_rpp_rope_is_k_shift(node_iter.first) != dst_is_k_shift) {
                    continue;
                }
                auto & node_vec = node_iter.second;
                for (size_t i = 0; i < node_vec.size(); i++) {
                    auto cur_node = node_vec[i].get();
                    if (ggml_rpp_rope_dims_is_same(dst, node_iter.first, cur_node)) {
                        return cur_node;
                    }
                }
            }
        }
    }
    return nullptr;
}

static bool ggml_rpp_rope_properties_is_same(ggml_backend_rpp_context & ctx,
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
    if (rpp_node->n_ubatch == 1) {
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (node->ne[i] != graph_node_properties.ne[i]) {
                return false;
            }
            if (node->nb[i] != graph_node_properties.nb[i]) {
                return false;
            }
        }
    } else {
        if (dst->ne[rpp_node->seq_len_index] == 1) {
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

static void init_sincos_cache(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node, ggml_tensor * dst) {
    GGML_ASSERT(base_rpp_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "init_kernel_rope_sin/cos");
    const size_t n_table_elements  = 8192 * dst->ne[0];
    const size_t device_type_size  = ggml_rpp_rope_device_type_size(ctx, dst->src[0]);
    const size_t device_cache_size = n_table_elements * device_type_size;
    const size_t host_cache_size   = n_table_elements * sizeof(float);
    ctx.cos_cache                  = ctx.pool().alloc(device_cache_size);
    ctx.sin_cache                  = ctx.pool().alloc(device_cache_size);
    ctx.rope_cache_D               = dst->ne[0];
    ctx.rope_cache_type_size       = device_type_size;
    if (!ctx.k_shift_sin_cache || !ctx.k_shift_cos_cache || ctx.k_shift_cache_size < device_cache_size) {
        ctx.k_shift_sin_cache  = ctx.pool().alloc(device_cache_size);
        ctx.k_shift_cos_cache  = ctx.pool().alloc(device_cache_size);
        ctx.k_shift_cache_size = device_cache_size;
    }

    void * sin_tmp = nullptr;
    void * cos_tmp = nullptr;
    rtMallocHost(&sin_tmp, host_cache_size);
    rtMallocHost(&cos_tmp, host_cache_size);
    ggml_cpu_rope_sincos_f32_calc(ctx, dst, (float *) cos_tmp, (float *) sin_tmp);
    if (device_type_size == sizeof(ggml_bf16_t)) {
        for (size_t i = 0; i < n_table_elements; i++) {
            ggml_bf16_t * bf16_sin = (ggml_bf16_t *) sin_tmp;
            ggml_bf16_t * bf16_cos = (ggml_bf16_t *) cos_tmp;
            float *       fp32_sin = (float *) sin_tmp;
            float *       fp32_cos = (float *) cos_tmp;
            bf16_sin[i]            = ggml_fp32_to_bf16(fp32_sin[i]);
            bf16_cos[i]            = ggml_fp32_to_bf16(fp32_cos[i]);
        }
        RPP_MEMCPY_DEV_AND_HOST(ctx.sin_cache, sin_tmp, device_cache_size, rtMemcpyHostToDevice, ctx.stream(), false);
        RPP_MEMCPY_DEV_AND_HOST(ctx.cos_cache, cos_tmp, device_cache_size, rtMemcpyHostToDevice, ctx.stream(), false);
    } else {
        RPP_MEMCPY_DEV_AND_HOST(ctx.sin_cache, sin_tmp, device_cache_size, rtMemcpyHostToDevice, ctx.stream(), false);
        RPP_MEMCPY_DEV_AND_HOST(ctx.cos_cache, cos_tmp, device_cache_size, rtMemcpyHostToDevice, ctx.stream(), false);
    }
    rtFreeHost(sin_tmp);
    rtFreeHost(cos_tmp);
}

static bool ggml_rpp_set_io_datas_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node) {
    GGML_ASSERT(base_rpp_node);
    rpp_kernel_rope * rpp_node = (rpp_kernel_rope *) base_rpp_node;
    // K-shift uses per-cell delta values, so its sin/cos table is generated from src[1] at launch time.
    if (!rpp_node->ori_rpp_node) {
        const size_t n_sin_elements    = ggml_nelements(rpp_node->ggml_sin.get());
        const size_t n_cos_elements    = ggml_nelements(rpp_node->ggml_cos.get());
        const size_t sin_device_size   = n_sin_elements * ggml_rpp_rope_device_type_size(ctx, rpp_node->cur_ggml_tensor->src[0]);
        const size_t cos_device_size   = n_cos_elements * ggml_rpp_rope_device_type_size(ctx, rpp_node->cur_ggml_tensor->src[0]);
        const size_t sin_host_f32_size = n_sin_elements * sizeof(float);
        const size_t cos_host_f32_size = n_cos_elements * sizeof(float);
        memset(rpp_node->sin_data, 0, sin_host_f32_size);
        memset(rpp_node->cos_data, 0, cos_host_f32_size);
        ggml_cpu_rope_get_sincos(rpp_node->cur_ggml_tensor, rpp_node->sin_data, rpp_node->cos_data);
        GGML_ASSERT(n_sin_elements == n_cos_elements);
        if (sin_device_size != sin_host_f32_size) {
            GGML_ASSERT(sin_device_size == n_sin_elements * sizeof(ggml_bf16_t));
            for (size_t i = 0; i < n_sin_elements; i++) {
                ggml_bf16_t * bf16_sin = (ggml_bf16_t *) rpp_node->sin_data;
                ggml_bf16_t * bf16_cos = (ggml_bf16_t *) rpp_node->cos_data;
                float *       fp32_sin = (float *) rpp_node->sin_data;
                float *       fp32_cos = (float *) rpp_node->cos_data;
                bf16_sin[i]            = ggml_fp32_to_bf16(fp32_sin[i]);
                bf16_cos[i]            = ggml_fp32_to_bf16(fp32_cos[i]);
            }
        }
        RPP_MEMCPY_DEV_AND_HOST(rpp_node->ggml_sin->data, rpp_node->sin_data, sin_device_size, rtMemcpyHostToDevice,
                                ctx.stream(), false);
        RPP_MEMCPY_DEV_AND_HOST(rpp_node->ggml_cos->data, rpp_node->cos_data, cos_device_size, rtMemcpyHostToDevice,
                                ctx.stream(), false);
    }
    // new function is support for not contiguous, so commented code
    // for (auto iter : rpp_node->binding_i_buffers) {
    //     if (!ggml_is_contiguous(iter.first)) {
    //         ggml_rpp_pack_tensor_to_contiguous(iter.first, iter.second, iter.first->view_offs);
    //     }
    // }
    return true;
}

static bool ggml_rpp_create_io_update_graph(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node) {
    GGML_ASSERT(base_rpp_node);
    auto *    rpp_node       = static_cast<rpp_kernel_rope *>(base_rpp_node);
    auto *    dst            = base_rpp_node->cur_ggml_tensor;
    const int start_pos_size = ggml_type_size(dst->src[1]->type);
    size_t    real_type_size = ggml_rpp_rope_device_type_size(ctx, dst->src[0]);
    size_t    cos_copy_size  = ggml_nelements(rpp_node->ggml_cos.get()) * real_type_size;
    size_t    sin_copy_size  = ggml_nelements(rpp_node->ggml_sin.get()) * real_type_size;
    size_t    block_size      = dst->ne[0] * real_type_size;

    // ggml_backend_get_device(ctx);
    rpp_node->io_update_kernel_ctx = std::make_unique<rpp_kernel_context>();
    rpp_init_kernel_ctx(*rpp_node->io_update_kernel_ctx.get());

    if (!rpp_node->start_pos_cdma_desc) {
        RPPdeviceptr phy_addr = 0;
        RPP_CHECK(rppGraphResourceAlloc(&phy_addr, start_pos_size, RPP_GRAPH_RESOURCE_CDMA_DESC));
        rppMemGetVirtAddr(&rpp_node->start_pos_cdma_desc, RPP_MEMORYTYPE_GRAPH_DESC, phy_addr);
    }
    GGML_ASSERT(rpp_node->start_pos_cdma_desc);

    // Use 64-bit offsets for the indirect update. Larger model-shaped RoPE caches
    // may span pooled-memory regions where querying the final byte's physical
    // address is not valid, while 64-bit offsets are safe for both small and
    // large cache slices.
    size_t element_size = sizeof(uint64_t);

    RPP_MEMCPY_INDIRECT_UPDATE_PARAMS updateSrcBaseOffsetParams_cos;
    memset(&updateSrcBaseOffsetParams_cos, 0, sizeof(updateSrcBaseOffsetParams_cos));
    updateSrcBaseOffsetParams_cos.inputType                    = RPP_MEMCPY_INDIRECT_INPUT_TYPE_BASE_OFFSET;
    updateSrcBaseOffsetParams_cos.input.baseOffset.indexAddr   = rpp_node->start_pos_cdma_desc;
    updateSrcBaseOffsetParams_cos.input.baseOffset.baseAddr    = (RPPdeviceptr) ctx.cos_cache;
    updateSrcBaseOffsetParams_cos.input.baseOffset.elementSize = element_size;
    updateSrcBaseOffsetParams_cos.input.baseOffset.blockSize   = block_size;
    updateSrcBaseOffsetParams_cos.input.baseOffset.offset      = 0;
    updateSrcBaseOffsetParams_cos.target                       = RPP_MEMCPY_INDIRECT_TARGET_SRC_ADDR;

    RPP_MEMCPY_INDIRECT_UPDATE_PARAMS updateSrcBaseOffsetParams_sin;
    memset(&updateSrcBaseOffsetParams_sin, 0, sizeof(updateSrcBaseOffsetParams_sin));
    updateSrcBaseOffsetParams_sin.inputType                    = RPP_MEMCPY_INDIRECT_INPUT_TYPE_BASE_OFFSET;
    updateSrcBaseOffsetParams_sin.input.baseOffset.indexAddr   = rpp_node->start_pos_cdma_desc;
    updateSrcBaseOffsetParams_sin.input.baseOffset.baseAddr    = (RPPdeviceptr) ctx.sin_cache;
    updateSrcBaseOffsetParams_sin.input.baseOffset.blockSize   = block_size;
    updateSrcBaseOffsetParams_sin.input.baseOffset.elementSize = element_size;
    updateSrcBaseOffsetParams_sin.input.baseOffset.offset      = 0;
    updateSrcBaseOffsetParams_sin.target                       = RPP_MEMCPY_INDIRECT_TARGET_SRC_ADDR;

    RPPcontext kernel_ctx = nullptr;
    // RPP_CHECK(rppCtxGetCurrent(&kernel_ctx));
    RPPstream  stream     = rpp_node->kernel_ctx->kernelStream;

    RPP_CHECK(rppStreamBeginCapture(stream, RPP_STREAM_CAPTURE_MODE_GLOBAL));
    RPP_CHECK(
        rppMemcpyDtoDAsync(rpp_node->start_pos_cdma_desc, (RPPdeviceptr) dst->src[1]->data, start_pos_size, stream));
    RPP_CHECK(rppGraphMemcpyNodeSetIndirectParamsAsync(rpp_node->io_update_kernel_ctx->graph, NULL,
                                                       &updateSrcBaseOffsetParams_cos, kernel_ctx, stream));
    RPP_CHECK(rppGraphMemcpyNodeSetIndirectParamsAsync(rpp_node->io_update_kernel_ctx->graph, NULL,
                                                       &updateSrcBaseOffsetParams_sin, kernel_ctx, stream));
    RPP_CHECK(rppMemcpyDtoDAsync((RPPdeviceptr) rpp_node->ggml_cos->data, (RPPdeviceptr) ctx.cos_cache, cos_copy_size,
                                 stream));
    RPP_CHECK(rppMemcpyDtoDAsync((RPPdeviceptr) rpp_node->ggml_sin->data, (RPPdeviceptr) ctx.sin_cache, sin_copy_size,
                                 stream));
    RPP_CHECK(rppStreamEndCapture(stream, &rpp_node->io_update_kernel_ctx->graph));

    size_t       num_nodes = 0;
    RPPgraphNode nodes[16];
    // RPPgraphNode cos_params = nullptr;
    // RPPgraphNode sin_params = nullptr;
    // RPPgraphNode cos_node   = nullptr;
    // RPPgraphNode sin_node   = nullptr;
    RPP_CHECK(rppGraphGetNodes(rpp_node->io_update_kernel_ctx->graph, nodes, &num_nodes));
    GGML_ASSERT(num_nodes == 5);
    RPPgraphNode set_cos_node    = nodes[1];
    RPPgraphNode set_sin_node    = nodes[2];
    RPPgraphNode target_cos_node = nodes[3];
    RPPgraphNode target_sin_node = nodes[4];

    // RPP_MEMCPY3D memcpy_params = {};
    // RPP_CHECK(rppGraphMemcpyNodeGetParams(set_cos_node, &memcpy_params));
    // for (size_t i = 0; i < num_nodes; ++i) {
    //     if (!nodes[i]) {
    //         continue;
    //     }
    //     RPPgraphNodeType type = RPP_GRAPH_NODE_TYPE_EMPTY;
    //     RPP_CHECK(rppGraphNodeGetType(nodes[i], &type));
    //     if (type != RPP_GRAPH_NODE_TYPE_MEMCPY) {
    //         continue;
    //     }

    //     // if (type == RPP_GRAPH_NODE_TYPE_MEMCPY_INDIRECT_UPDATE) {
    //     //     if (srcUpdateNode == NULL) {
    //     //         srcUpdateNode = nodes[i];
    //     //     } else if (dstUpdateNode == NULL) {
    //     //         dstUpdateNode = nodes[i];
    //     //     }
    //     //     continue;
    //     // }

    //     RPP_MEMCPY3D memcpy_params = {};
    //     RPP_CHECK(rppGraphMemcpyNodeGetParams(nodes[i], &memcpy_params));
    //     if (memcpy_params.dst == (RPPdeviceptr) rpp_node->ggml_cos->data) {
    //         cos_node = nodes[i];
    //     } else if (memcpy_params.dst == (RPPdeviceptr) rpp_node->ggml_sin->data) {
    //         sin_node = nodes[i];
    //     }

    //     if (cos_node && sin_node) {
    //         break;
    //     }
    // }
    RPP_MEMCPY_INDIRECT_UPDATE_NODE_PARAMS cosNodeSetParams;
    RPP_MEMCPY_INDIRECT_UPDATE_NODE_PARAMS sinNodeSetParams;
    memset(&cosNodeSetParams, 0, sizeof(cosNodeSetParams));
    cosNodeSetParams.targetNode = target_cos_node;
    memcpy(&cosNodeSetParams.updateParams, &updateSrcBaseOffsetParams_cos, sizeof(updateSrcBaseOffsetParams_cos));
    RPP_CHECK(rppGraphMemcpyIndirectUpdateNodeSetParams(rpp_node->io_update_kernel_ctx->graph, set_cos_node,
                                                        &cosNodeSetParams));

    memset(&sinNodeSetParams, 0, sizeof(sinNodeSetParams));
    sinNodeSetParams.targetNode = target_sin_node;
    memcpy(&sinNodeSetParams.updateParams, &updateSrcBaseOffsetParams_sin, sizeof(updateSrcBaseOffsetParams_sin));
    RPP_CHECK(rppGraphMemcpyIndirectUpdateNodeSetParams(rpp_node->io_update_kernel_ctx->graph, set_sin_node,
                                                        &sinNodeSetParams));

    const std::string graph_key =
        rpp_join_function_name_and_args(__func__, start_pos_size, real_type_size, cos_copy_size, sin_copy_size,
                                        block_size, element_size, ctx.use_bf16);
    RPP_CHECK(rpp_graph_instantiate(rpp_node->io_update_kernel_ctx->graphexec, rpp_node->io_update_kernel_ctx->graph,
                                    graph_key.c_str(), rpp_node->is_instantial));

    return true;
}

static bool ggml_rpp_update_io_datas_from_graph(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node) {
    GGML_ASSERT(base_rpp_node);
    auto * rpp_node = static_cast<rpp_kernel_rope *>(base_rpp_node);
    auto * dst      = base_rpp_node->cur_ggml_tensor;
    // only update io datas from graph if first rope node
    if (rpp_node->ori_rpp_node) {
        return true;
    }
    GGML_ASSERT(rpp_node->io_update_kernel_ctx);
    GGML_ASSERT(rpp_node->io_update_kernel_ctx->graphexec);
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_rope_io_update");
        RPP_LAUNCH_KERNEL(rpp_node->io_update_kernel_ctx->graphexec, ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }
    // rtStreamSynchronize(ctx.stream());
    return true;
}

struct ggml_rpp_rope_compare_inputs {
    std::vector<float> src0;
};

static std::vector<float> ggml_rpp_rope_copy_tensor_as_f32(ggml_backend_rpp_context & ctx,
                                                           ggml_tensor *              tensor,
                                                           int32_t                    io_type) {
    const size_t n    = (size_t) ggml_nelements(tensor);
    const int    size = ggml_rpp_get_io_type_size(ctx, tensor, io_type);
    std::vector<float> result(n);
    if (size == (int) sizeof(float)) {
        RPP_CHECK(rtMemcpy(result.data(), tensor->data, n * sizeof(float), rtMemcpyDeviceToHost));
    } else if (size == (int) sizeof(ggml_bf16_t)) {
        std::vector<ggml_bf16_t> tmp(n);
        RPP_CHECK(rtMemcpy(tmp.data(), tensor->data, n * sizeof(ggml_bf16_t), rtMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            result[i] = ggml_bf16_to_fp32(tmp[i]);
        }
    } else {
        throw std::runtime_error("unsupported RPP ROPE compare io type size");
    }
    return result;
}

static ggml_rpp_rope_compare_inputs ggml_rpp_prepare_rope_compare(ggml_backend_rpp_context & ctx,
                                                                  rtStream_t                 stream,
                                                                  ggml_tensor *              dst) {
    if (dst->type != GGML_TYPE_F32 || dst->src[0]->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[RPP ROPE COMPARE] %s: skip non-f32 src0=%s dst=%s\n", dst->name,
                     ggml_type_name(dst->src[0]->type), ggml_type_name(dst->type));
        return {};
    }
    RPP_CHECK(rtStreamSynchronize(stream));
    std::vector<float> src0 = ggml_rpp_rope_copy_tensor_as_f32(ctx, dst->src[0], 0);
    return { std::move(src0) };
}

static void ggml_rpp_compare_rope_with_cpu(ggml_backend_rpp_context &            ctx,
                                           rtStream_t                           stream,
                                           ggml_tensor *                        dst,
                                           const ggml_rpp_rope_compare_inputs & inputs) {
    if (inputs.src0.empty()) {
        return;
    }

    const int mode  = ((int32_t *) dst->op_params)[2];
    const int n_rot = dst->op_params[1];
    if (mode != GGML_ROPE_TYPE_NEOX || n_rot != dst->ne[0]) {
        std::fprintf(stderr, "[RPP ROPE COMPARE] %s: skip mode=%d n_rot=%d D=%lld\n", dst->name, mode, n_rot,
                     (long long) dst->ne[0]);
        return;
    }

    RPP_CHECK(rtStreamSynchronize(stream));
    const size_t nd = (size_t) ggml_nelements(dst);
    std::vector<float> got = ggml_rpp_rope_copy_tensor_as_f32(ctx, dst, 1);

    const int64_t D = dst->ne[0];
    const int64_t H = dst->ne[1];
    const int64_t T = dst->ne[2];
    std::vector<float> cos((size_t) T * (size_t) D);
    std::vector<float> sin((size_t) T * (size_t) D);
    ggml_cpu_rope_sincos_f32(dst, cos.data(), sin.data());

    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_sq_diff = 0.0;
    double sum_abs_diff = 0.0;
    double sum_sq_ref = 0.0;
    size_t max_idx = 0;
    size_t bad_cnt = 0;
    float  ref_at_max = 0.0f;
    for (int64_t i2 = 0; i2 < T; ++i2) {
        for (int64_t i1 = 0; i1 < H; ++i1) {
            const size_t row = (size_t) (i2 * H + i1) * (size_t) D;
            const size_t tbl = (size_t) i2 * (size_t) D;
            for (int64_t i0 = 0; i0 < D; ++i0) {
                const size_t idx = row + (size_t) i0;
                float ref;
                if (i0 < D / 2) {
                    ref = inputs.src0[idx] * cos[tbl + (size_t) i0] -
                          inputs.src0[idx + (size_t) D / 2] * sin[tbl + (size_t) i0];
                } else {
                    ref = inputs.src0[idx - (size_t) D / 2] * sin[tbl + (size_t) i0] +
                          inputs.src0[idx] * cos[tbl + (size_t) i0];
                }
                const double abs_diff = std::fabs((double) got[idx] - (double) ref);
                const double rel_diff = abs_diff / std::max(1.0, std::fabs((double) ref));
                sum_sq_diff += abs_diff * abs_diff;
                sum_abs_diff += abs_diff;
                sum_sq_ref += (double) ref * (double) ref;
                if (abs_diff > max_abs) {
                    max_abs = abs_diff;
                    max_rel = rel_diff;
                    max_idx = idx;
                    ref_at_max = ref;
                }
                if (abs_diff > 1e-3 && rel_diff > 1e-3) {
                    ++bad_cnt;
                }
            }
        }
    }
    const double mse  = nd == 0 ? 0.0 : sum_sq_diff / (double) nd;
    const double rmse = std::sqrt(mse);
    const double mae  = nd == 0 ? 0.0 : sum_abs_diff / (double) nd;
    const double nmse = sum_sq_ref == 0.0 ? 0.0 : sum_sq_diff / sum_sq_ref;

    std::fprintf(stderr,
                 "[RPP ROPE COMPARE] name=%s ne=[%lld,%lld,%lld,%lld] io=[%d,%d] mode=%d n_rot=%d mse=%g "
                 "rmse=%g mae=%g nmse=%g max_abs=%g max_rel=%g bad=%zu/%zu max_idx=%zu got=%g ref=%g\n",
                 dst->name, (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2],
                 (long long) dst->ne[3], ggml_rpp_get_io_type_size(ctx, dst->src[0], 0),
                 ggml_rpp_get_io_type_size(ctx, dst, 1), mode, n_rot, mse, rmse, mae, nmse, max_abs, max_rel,
                 bad_cnt, nd, max_idx, got[max_idx], ref_at_max);
}

static bool ggml_rpp_create_kernel_rope(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_rope *>(rpp_base_node);
    // in phi4, dst->src[0] is not contiguous
    // GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    rpp_node->is_k_shift = ggml_rpp_rope_is_k_shift(dst);

    // Reuse sin/cos workspace across compatible non-K-shift ROPE kernels.
    auto ori_rpp_node = ggml_rpp_find_rope_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rope_node             = static_cast<rpp_kernel_rope *>(ori_rpp_node);
        rpp_node->ori_rpp_node         = ori_rpp_node;
        rpp_node->ggml_cos             = ori_rope_node->ggml_cos;
        rpp_node->ggml_sin             = ori_rope_node->ggml_sin;
        rpp_node->cos_data             = ori_rope_node->cos_data;
        rpp_node->sin_data             = ori_rope_node->sin_data;
        rpp_node->io_update_kernel_ctx = ori_rope_node->io_update_kernel_ctx;
    } else {
        if (!rpp_node->sin_data || !rpp_node->cos_data) {
            rpp_node->init_sincos_tensors(rpp_node->is_k_shift ? ctx.k_shift_sin_cache : nullptr,
                                          rpp_node->is_k_shift ? ctx.k_shift_cos_cache : nullptr);
        }
        if (!rpp_node->is_k_shift && !rpp_node->io_update_kernel_ctx) {
            ggml_rpp_create_io_update_graph(ctx, rpp_node);
        }
    }

    const int seq_len = ggml_rpp_rope_seq_len(dst, rpp_node);
    int       T       = dst->ne[2];
    if (!rpp_node->is_k_shift && seq_len > 1 && ctx.use_ubatch) {
        T = rpp_node->n_ubatch;
    }
    const int H       = dst->ne[1];
    const int D       = dst->ne[0];
    const bool use_internal_bf16     = ggml_rpp_rope_uses_internal_bf16(ctx, dst->src[0]);
    const bool dst_use_internal_bf16 = ggml_rpp_rope_uses_internal_bf16(ctx, dst);
    const int  Tstride               = use_internal_bf16 ? dst->src[0]->nb[2] / 2 : dst->src[0]->nb[2];
    const int  Hstride               = use_internal_bf16 ? dst->src[0]->nb[1] / 2 : dst->src[0]->nb[1];
    const int  Dstride               = use_internal_bf16 ? dst->src[0]->nb[0] / 2 : dst->src[0]->nb[0];
    const int  out_Tstride           = dst_use_internal_bf16 ? dst->nb[2] / 2 : dst->nb[2];
    const int  out_Hstride           = dst_use_internal_bf16 ? dst->nb[1] / 2 : dst->nb[1];
    const int  out_Dstride           = dst_use_internal_bf16 ? dst->nb[0] / 2 : dst->nb[0];
    const int mode    = ((int32_t *) dst->op_params)[2];
    const int n_rot   = dst->op_params[1];
    if (getenv("GGML_RPP_DEBUG_ROPE") != nullptr) {
        std::vector<int32_t> pos((size_t) std::min<int64_t>(T, dst->src[1]->ne[0]));
        if (!pos.empty()) {
            RPP_CHECK(rtMemcpy(pos.data(), dst->src[1]->data, pos.size() * sizeof(int32_t), rtMemcpyDeviceToHost));
        }
        std::fprintf(stderr,
                     "[RPP ROPE] create name=%s T=%d H=%d D=%d mode=%d n_rot=%d in_type=%s out_type=%s "
                     "stride=[%d,%d,%d] out_stride=[%d,%d,%d] pos=",
                     dst->name, T, H, D, mode, n_rot, ggml_type_name(dst->src[0]->type), ggml_type_name(dst->type),
                     Tstride, Hstride, Dstride, out_Tstride, out_Hstride, out_Dstride);
        for (int32_t p : pos) {
            std::fprintf(stderr, "%d,", p);
        }
        std::fprintf(stderr, "\n");
    }

    void * i_buffer_0 = nullptr;
    if (use_internal_bf16) {
        if (dst->src[0]->view_offs == 0) {
            i_buffer_0 = dst->src[0]->data;
        } else {
            i_buffer_0 =
                reinterpret_cast<void *>(reinterpret_cast<char *>(dst->src[0]->data) - dst->src[0]->view_offs / 2);
        }
    } else {
        i_buffer_0 = dst->src[0]->data;
    }
    // in phi4 model, dst->src[0] is not contiguous,new function is support for not contiguous, so commented code
    // if (!ggml_is_contiguous(dst->src[0])) {
    //     size_t io_size = T * H * D * ggml_type_size(dst->src[0]->type);
    //     i_buffer_0     = ctx.pool().alloc(io_size);
    //     rpp_node->pool_buffers.emplace(i_buffer_0);
    // }

    // kernel inputs
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (i_buffer_0));
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (rpp_node->ggml_cos->data));
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (rpp_node->ggml_sin->data));

    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[0], i_buffer_0);
    rpp_node->binding_i_buffers.emplace(rpp_node->ggml_cos.get(), rpp_node->ggml_cos->data);
    rpp_node->binding_i_buffers.emplace(rpp_node->ggml_sin.get(), rpp_node->ggml_sin->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(i_buffer_0);
    rpp_node->binding_io_buffers.emplace_back(rpp_node->ggml_cos->data);
    rpp_node->binding_io_buffers.emplace_back(rpp_node->ggml_sin->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    const int i_type_size_0 = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int i_type_size_1 = ggml_rpp_get_io_type_size(ctx, rpp_node->ggml_cos.get(), 0);
    const int i_type_size_2 = ggml_rpp_get_io_type_size(ctx, rpp_node->ggml_sin.get(), 0);
    const int o_type_size   = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build rope kernel
    rpp_rope_build(*(rpp_node->kernel_ctx.get()), T, H, D, Tstride, Hstride, Dstride, out_Tstride, out_Hstride,
                   out_Dstride, mode, n_rot, i_type_size_0, i_type_size_1, i_type_size_2, o_type_size,
                   rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_rope *>(rpp_base_node);
    bool ret      = false;

    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n = ggml_n_dims(dst);
        GGML_ASSERT(n >= 1);
        rpp_node->seq_len_index = n <= 2 ? 2 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_rope_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    if (!ctx.cos_cache && !ctx.sin_cache) {
        init_sincos_cache(ctx, rpp_node, dst);
    }

    ret = ggml_rpp_create_kernel_rope(ctx, rpp_node, dst);
    GGML_ASSERT(ret);
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(rpp_node->ggml_cos.get());
        rpp_node->binding_io_tensors.emplace_back(rpp_node->ggml_sin.get());
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

bool ggml_rpp_op_kernel_rope(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_rope * rpp_node = nullptr;
    auto              iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_rope");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_rope_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_rope *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_rope");
            auto new_node = std::make_unique<rpp_kernel_rope>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_rope *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
        // ctx.cur_rpp_graph->add_launch_func(ggml_rpp_update_io_datas_from_graph, rpp_node);
    } else {
        rpp_node = (rpp_kernel_rope *) (iter->second);
    }

    if (is_launch) {
        {
            TRACE_SCOPE_GUARD(ctx.trace_id, "set_kernel_rope_datas");
            const size_t cur_type_size = ggml_rpp_rope_device_type_size(ctx, dst->src[0]);
            const bool can_use_graph_update =
                !rpp_node->is_k_shift && ctx.cos_cache && ctx.sin_cache && ctx.rope_cache_D == dst->ne[0] &&
                ctx.rope_cache_type_size == cur_type_size;
            if (can_use_graph_update) {
                ggml_rpp_update_io_datas_from_graph(ctx, rpp_node);
            } else {
                ggml_rpp_set_io_datas_device(ctx, rpp_node);
            }
        }
        // compute rope operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_rope");
            ggml_rpp_rope_compare_inputs compare_inputs;
            if (ggml_rpp_should_compare_rope(dst)) {
                compare_inputs = ggml_rpp_prepare_rope_compare(ctx, ctx.stream(), dst);
            }
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
            if (ggml_rpp_should_compare_rope(dst)) {
                ggml_rpp_compare_rope_with_cpu(ctx, ctx.stream(), dst, compare_inputs);
            }
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return true;
}
