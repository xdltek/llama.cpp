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
        const float ff_src = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        // is bug for freq factor, maybe 0, so need check
        const float ff     = ff_src ? ff_src : 1.0f;
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
        const float ff_src = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        // is bug for freq factor, maybe 0, so need check
        const float ff     = ff_src ? ff_src : 1.0f;
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
        if (sector >= sections[0] && sector < sec_w) {
            theta = theta_h;
        } else if (sector >= sec_w && sector < sec_w + sections[2]) {
            theta = theta_w;
        } else if (sector >= sec_w + sections[2]) {
            theta = theta_e;
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

    GGML_ASSERT(nb00 == sizeof(float));

    // const int ith = params->ith;
    // const int nth = params->nth;

    const int nr = dst->ne[1] * ctx.n_max_ctx * dst->ne[3];

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    // const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    // const int ir0 = dr*ith;
    // const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
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
    pos_data.reserve(ctx.n_max_ctx * 4);
    for (uint32_t i = 0; i < ctx.n_max_ctx; ++i) {
        push_u32_le(pos_data, i);
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
                ggml_mrope_cache_init(p_t, p_h, p_w, p_e, sections, is_vision, freq_scale, freq_factors, corr_dims, ne0,
                                      ext_factor, attn_factor, cache.data(), sin_sign, theta_scale);
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

    GGML_ASSERT(nb00 == sizeof(float));

    // const int ith = params->ith;
    // const int nth = params->nth;

    const int nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    // const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    // const int ir0 = dr*ith;
    // const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
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
                ggml_mrope_cache_init(p_t, p_h, p_w, p_e, sections, is_vision, freq_scale, freq_factors, corr_dims, ne0,
                                      ext_factor, attn_factor, cache.data(), sin_sign, theta_scale);
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
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_ROPE) {
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
    rpp_kernel_rope * rpp_node   = (rpp_kernel_rope *) base_rpp_node;
    size_t            cache_size = ctx.n_max_ctx * dst->ne[0] * ggml_type_size(dst->src[0]->type);
    ctx.cos_cache                = ctx.pool().alloc(cache_size);
    ctx.sin_cache                = ctx.pool().alloc(cache_size);

    void * sin_tmp = nullptr;
    void * cos_tmp = nullptr;
    rtMallocHost(&sin_tmp, cache_size);
    rtMallocHost(&cos_tmp, cache_size);
    ggml_cpu_rope_sincos_f32_calc(ctx, dst, (float *) cos_tmp, (float *) sin_tmp);
    if (ctx.use_bf16) {
        for (size_t i = 0; i < cache_size / 4; i++) {
            ggml_bf16_t * bf16_sin = (ggml_bf16_t *) sin_tmp;
            ggml_bf16_t * bf16_cos = (ggml_bf16_t *) cos_tmp;
            float *       fp32_sin = (float *) sin_tmp;
            float *       fp32_cos = (float *) cos_tmp;
            bf16_sin[i]            = ggml_fp32_to_bf16(fp32_sin[i]);
            bf16_cos[i]            = ggml_fp32_to_bf16(fp32_cos[i]);
        }
        RPP_MEMCPY_DEV_AND_HOST(ctx.sin_cache, sin_tmp, cache_size / 2, rtMemcpyHostToDevice, ctx.stream());
        RPP_MEMCPY_DEV_AND_HOST(ctx.cos_cache, cos_tmp, cache_size / 2, rtMemcpyHostToDevice, ctx.stream());
    } else {
        RPP_MEMCPY_DEV_AND_HOST(ctx.sin_cache, sin_tmp, cache_size, rtMemcpyHostToDevice, ctx.stream());
        RPP_MEMCPY_DEV_AND_HOST(ctx.cos_cache, cos_tmp, cache_size, rtMemcpyHostToDevice, ctx.stream());
    }
    rtFreeHost(sin_tmp);
    rtFreeHost(cos_tmp);
}

static bool ggml_rpp_set_io_datas_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node) {
    GGML_ASSERT(base_rpp_node);
    rpp_kernel_rope * rpp_node = (rpp_kernel_rope *) base_rpp_node;
    // get rope sin/cos data from cpu
    if (!rpp_node->ori_rpp_node) {
        size_t sin_size = ggml_nbytes(rpp_node->ggml_sin.get());
        size_t cos_size = ggml_nbytes(rpp_node->ggml_cos.get());
        memset(rpp_node->sin_data, 0, sin_size);
        memset(rpp_node->cos_data, 0, cos_size);
        ggml_cpu_rope_get_sincos(rpp_node->cur_ggml_tensor, rpp_node->sin_data, rpp_node->cos_data);
        GGML_ASSERT(sin_size == cos_size);
        if (ctx.use_bf16) {
            for (size_t i = 0; i < sin_size / 4; i++) {
                ggml_bf16_t * bf16_sin = (ggml_bf16_t *) rpp_node->sin_data;
                ggml_bf16_t * bf16_cos = (ggml_bf16_t *) rpp_node->cos_data;
                float *       fp32_sin = (float *) rpp_node->sin_data;
                float *       fp32_cos = (float *) rpp_node->cos_data;
                bf16_sin[i]            = ggml_fp32_to_bf16(fp32_sin[i]);
                bf16_cos[i]            = ggml_fp32_to_bf16(fp32_cos[i]);
            }
            RPP_MEMCPY_DEV_AND_HOST(rpp_node->ggml_sin->data, rpp_node->sin_data, sin_size / 2, rtMemcpyHostToDevice,
                                    ctx.stream());
            RPP_MEMCPY_DEV_AND_HOST(rpp_node->ggml_cos->data, rpp_node->cos_data, cos_size / 2, rtMemcpyHostToDevice,
                                    ctx.stream());
        } else {
            RPP_MEMCPY_DEV_AND_HOST(rpp_node->ggml_sin->data, rpp_node->sin_data, sin_size, rtMemcpyHostToDevice,
                                    ctx.stream());
            RPP_MEMCPY_DEV_AND_HOST(rpp_node->ggml_cos->data, rpp_node->cos_data, cos_size, rtMemcpyHostToDevice,
                                    ctx.stream());
        }
    }
    // new function is support for not contiguous, so commented code
    // for (auto iter : rpp_node->binding_i_buffers) {
    //     if (!ggml_is_contiguous(iter.first)) {
    //         ggml_rpp_pack_tensor_to_contiguous(iter.first, iter.second, iter.first->view_offs);
    //     }
    // }
    return true;
}

static bool ggml_rpp_get_io_datas_from_cahe(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node) {
    GGML_ASSERT(base_rpp_node);
    rpp_kernel_rope * rpp_node = (rpp_kernel_rope *) base_rpp_node;
    auto              dst      = base_rpp_node->cur_ggml_tensor;
    // get rope sin/cos data from cpu
    if (!rpp_node->ori_rpp_node) {
        const int start_pos_size  = ggml_type_size(dst->src[1]->type);
        const int copy_type_size  = ggml_type_size(dst->src[0]->type);
        size_t    real_type_size  = ctx.use_bf16 ? copy_type_size / 2 : copy_type_size;
        size_t    cache_used_size = dst->ne[rpp_node->seq_len_index] * dst->ne[0] * real_type_size;
        size_t    cache_total_size = ctx.n_max_ctx * dst->ne[0] * real_type_size;
        // NOTE: this value is consumed on host immediately
        RPP_MEMCPY_DEV_AND_HOST(rpp_node->start_pos_data, dst->src[1]->data, start_pos_size, rtMemcpyDeviceToHost,
                                ctx.stream(), 1);
        const int32_t start_pos = ((int32_t *) (rpp_node->start_pos_data))[0];
        GGML_ASSERT(start_pos >= 0);
        GGML_ASSERT((size_t) start_pos * dst->ne[0] * real_type_size + cache_used_size <= cache_total_size);

        RPPdeviceptr array_d[2];
        array_d[0] = (RPPdeviceptr) rpp_node->ggml_sin->data;
        array_d[1] = (RPPdeviceptr) rpp_node->ggml_cos->data;
        size_t size[2];
        size[0] = cache_used_size;
        size[1] = cache_used_size;
        RPPdeviceptr array_s[2];
        array_s[0] = (RPPdeviceptr) (char *) ctx.sin_cache + (size_t) start_pos * dst->ne[0] * real_type_size;
        array_s[1] = (RPPdeviceptr) (char *) ctx.cos_cache + (size_t) start_pos * dst->ne[0] * real_type_size;
        RPP_CHECK(rppMemcpyLinkDtoDAsync(array_d, array_s, size, 2, ctx.stream()));
    }
    // notice!!!, new function is support for not contiguous, so commented code
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
    const int copy_type_size = ggml_type_size(dst->src[0]->type);
    size_t    cos_size       = ggml_nbytes(rpp_node->ggml_cos.get());
    size_t    sin_size       = ggml_nbytes(rpp_node->ggml_sin.get());
    size_t    real_type_size = ctx.use_bf16 ? copy_type_size / 2 : copy_type_size;
    size_t    cos_copy_size   = ctx.use_bf16 ? cos_size / 2 : cos_size;
    size_t    sin_copy_size   = ctx.use_bf16 ? sin_size / 2 : sin_size;
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

    RPPdeviceptr min_phy_addr = 0;
    RPP_CHECK(rppMemGetPhyAddr(&min_phy_addr, (RPPdeviceptr) ctx.cos_cache));
    RPPdeviceptr max_phy_addr = 0;
    RPP_CHECK(rppMemGetPhyAddr(
        &max_phy_addr,
        (RPPdeviceptr) (reinterpret_cast<char *>(ctx.cos_cache) + ctx.n_max_ctx * block_size + sizeof(int64_t) - 1)));
    // Mpu only supports 32-bit load and store instructions, so it is necessary to modify the higher 32 bits, which are two instructions for Mpu
    auto isAddrRangeChanged = [&](RPPdeviceptr addr1, RPPdeviceptr addr2) -> bool {
        return (addr1 >> 32) != (addr2 >> 32);
    };
    size_t element_size = sizeof(uint32_t);
    if (isAddrRangeChanged(min_phy_addr, max_phy_addr)){
        element_size = sizeof(uint64_t);
    }

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

    if (rpp_node->is_instantial) {
        RPP_CHECK(rppGraphInstantiate(&rpp_node->io_update_kernel_ctx->graphexec, rpp_node->io_update_kernel_ctx->graph,
            NULL, NULL, 0));
    }

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

static bool ggml_rpp_create_kernel_rope(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_rope *>(rpp_base_node);
    // in phi4, dst->src[0] is not contiguous
    // GGML_ASSERT(ggml_is_contiguous(dst->src[0]));

    // find first rms_node, and all rope kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_rope_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node              = static_cast<rpp_kernel_rope *>(ori_rpp_node);
        rpp_node->ori_rpp_node         = ori_rpp_node;
        rpp_node->ggml_cos             = ori_rms_node->ggml_cos;
        rpp_node->ggml_sin             = ori_rms_node->ggml_sin;
        rpp_node->cos_data             = ori_rms_node->cos_data;
        rpp_node->sin_data             = ori_rms_node->sin_data;
        rpp_node->io_update_kernel_ctx = ori_rms_node->io_update_kernel_ctx;
    } else {
        if (!rpp_node->sin_data || !rpp_node->cos_data) {
            rpp_node->init_sincos_tensors();
        }
        if (!rpp_node->io_update_kernel_ctx) {
            ggml_rpp_create_io_update_graph(ctx, rpp_node);
        }
    }

    const int seq_len = ggml_rpp_rope_seq_len(dst, rpp_node);
    const int T       = (seq_len > 1 && ctx.use_ubatch) ? rpp_node->n_ubatch : dst->ne[2];
    const int H       = dst->ne[1];
    const int D       = dst->ne[0];
    const int Tstride = ctx.use_bf16 ? dst->src[0]->nb[2] / 2 : dst->src[0]->nb[2];
    const int Hstride = ctx.use_bf16 ? dst->src[0]->nb[1] / 2 : dst->src[0]->nb[1];
    const int Dstride = ctx.use_bf16 ? dst->src[0]->nb[0] / 2 : dst->src[0]->nb[0];
    const int mode    = ((int32_t *) dst->op_params)[2];
    const int n_rot   = dst->op_params[1];

    void * i_buffer_0 = nullptr;
    if (ctx.use_bf16) {
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
    rpp_rope_build(*(rpp_node->kernel_ctx.get()), T, H, D, Tstride, Hstride, Dstride, mode, n_rot, i_type_size_0,
                   i_type_size_1, i_type_size_2, o_type_size, rpp_node->is_instantial);
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
        GGML_ASSERT(n >= 2);
        rpp_node->seq_len_index = n == 2 ? n : n - 1;
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
            // ggml_rpp_get_io_datas_from_cahe(ctx, rpp_node);
            // this is for nup graph
            ggml_rpp_update_io_datas_from_graph(ctx, rpp_node);
        }
        // compute rope operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_rope");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return true;
}
