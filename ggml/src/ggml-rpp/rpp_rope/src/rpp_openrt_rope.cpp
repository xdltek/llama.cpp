
#include "rpp_rope/rpp_rope.h"

#include <numeric>

static int rope_num = 0;

#if GGML_RPP_USE_RT

static int ggml_rpp_rope_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

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

static bool ggml_rpp_create_engine_rope_ubatch(ggml_backend_rpp_context & ctx,
                                               ggml_rpp_node *            base_rpp_node,
                                               ggml_tensor *              dst) {
    GGML_ASSERT(base_rpp_node);
    auto rpp_node    = (rpp_openrt_rope *) base_rpp_node;
    auto rpp_network = rpp_node->network.get();

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    // notice!!! now only support batch = 1, ndims=3
    GGML_ASSERT(src0->ne[3] == 1);
    // GGML_ASSERT(ggml_is_contiguous(src0));

    // notice!!! now only support ndims = 3 in rope op
    infer1::ITensor * rt_inputs[3] = { nullptr, nullptr, nullptr };
    infer1::Dims      dims0;
    dims0.nbDims = 3;
    // dst->ne == dst->src[0]->ne
    // seq_len dimension needs to be expanded to the ubatch, if n_ubatch=128, so src[0] shape is 9,16,128->128,16,128,
    dims0.d[0]   = rpp_node->n_ubatch;
    dims0.d[1]   = src0->ne[1];
    dims0.d[2]   = src0->ne[0];
    rt_inputs[0] = rpp_node->network->addInput(src0->name, ggml_rpp_dtype_mapping(src0->type), dims0);

    infer1::Dims dims1;
    dims1.nbDims = 3;
    dims1.d[0]   = dst->ne[3];
    // seq_len dimension needs to be expanded to the ubatch,if n_ubatch=128, so src[1],src[2] shape is 1,9,128->1,128,128,
    dims1.d[1]   = rpp_node->n_ubatch;
    dims1.d[2]   = dst->ne[0];
    rt_inputs[1] = rpp_node->network->addInput("cos", ggml_rpp_dtype_mapping(src0->type), dims1);
    rt_inputs[2] = rpp_node->network->addInput("sin", ggml_rpp_dtype_mapping(src0->type), dims1);

    // create rope operator
    const int mode     = ((int32_t *) dst->op_params)[2];
    auto mul_mat_layer = rpp_network->addRoPE(*rt_inputs[0], *rt_inputs[1], *rt_inputs[2], mode, dst->op_params[1]);
    // auto mul_mat_layer = rpp_network->addRoPE(*rt_inputs[0], *rt_inputs[1], *rt_inputs[2]);
    if (!mul_mat_layer) {
        GGML_LOG_ERROR("%s: addRoPE faied  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = mul_mat_layer->getOutput(0);
    ort_tensor->setName(dst->name);
    rpp_network->markOutput(*ort_tensor);

    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_rope(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            base_rpp_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(base_rpp_node);
    auto rpp_node    = (rpp_openrt_rope *) base_rpp_node;
    auto rpp_graph   = ctx.cur_rpp_graph;
    auto rpp_network = rpp_node->network.get();

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    // now only support batch = 1, ndims=3
    GGML_ASSERT(src0->ne[3] == 1);
    // GGML_ASSERT(ggml_is_contiguous(src0));

    // now only support ndims = 3 in rope op
    infer1::ITensor * rt_inputs[3] = { nullptr, nullptr, nullptr };
    infer1::Dims      dims0;
    dims0.nbDims = 3;
    dims0.d[0]   = src0->ne[2];
    dims0.d[1]   = src0->ne[1];
    dims0.d[2]   = src0->ne[0];
    rt_inputs[0] = rpp_node->network->addInput(src0->name, ggml_rpp_dtype_mapping(src0->type), dims0);

    infer1::Dims dims1;
    dims1.nbDims = 3;
    dims1.d[0]   = dst->ne[3];
    dims1.d[1]   = dst->ne[2];
    dims1.d[2]   = dst->ne[0];
    rt_inputs[1] = rpp_node->network->addInput("cos", ggml_rpp_dtype_mapping(src0->type), dims1);
    rt_inputs[2] = rpp_node->network->addInput("sin", ggml_rpp_dtype_mapping(src0->type), dims1);

    // create rope operator
    const int mode     = ((int32_t *) dst->op_params)[2];
    auto mul_mat_layer = rpp_network->addRoPE(*rt_inputs[0], *rt_inputs[1], *rt_inputs[2], mode, dst->op_params[1]);
    // auto mul_mat_layer = rpp_network->addRoPE(*rt_inputs[0], *rt_inputs[1], *rt_inputs[2]);
    if (!mul_mat_layer) {
        GGML_LOG_ERROR("%s: addRoPE faied  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = mul_mat_layer->getOutput(0);
    ort_tensor->setName(dst->name);
    rpp_network->markOutput(*ort_tensor);

    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            base_rpp_node,
                                            ggml_tensor *              dst) {
    auto rpp_node = (rpp_openrt_rope *) base_rpp_node;
    GGML_ASSERT(rpp_node);
    bool ret = false;
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
    std::string save_path = STRINGIZE(GGML_RPP_SAVE_ENGINE);
    std::string load_path = STRINGIZE(GGML_RPP_LOAD_ENGINE);
    // load engine
    if (!load_path.empty()) {
        char engine_path[128];
        sprintf(engine_path, "%s/upatch%d_%s_%s.engine", load_path.c_str(), rpp_node->n_ubatch, ggml_op_name(dst->op),
                dst->name);
        ret = ggml_rpp_load_enigne(rpp_node, engine_path);
    }
    // create engine
    if (!ret) {
        if (ctx.use_ubatch && ggml_rpp_rope_seq_len(dst, rpp_node) > 1) {
            ret = ggml_rpp_create_engine_rope_ubatch(ctx, rpp_node, dst);
        } else {
            ret = ggml_rpp_create_engine_rope(ctx, rpp_node, dst);
        }
        if (ret && !save_path.empty()) {
            char engine_path[128];
            sprintf(engine_path, "%s/upatch%d_%s_%s.engine", save_path.c_str(), rpp_node->n_ubatch,
                    ggml_op_name(dst->op), dst->name);
            ggml_rpp_save_enigne(rpp_node, engine_path);
        }
    }
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        // get io tensor, only push src0, cos,sin has no original tensor and new tensor is placed in binding_io_tensors vector
        rpp_node->init_sincos_tensors();
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(rpp_node->ggml_cos.get());
        rpp_node->binding_io_tensors.emplace_back(rpp_node->ggml_sin.get());
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

static bool ggml_rpp_set_io_bingdings_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node) {
    GGML_ASSERT(base_rpp_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_rope_bingdings");
    // clear buffers
    rpp_openrt_rope * rpp_node = (rpp_openrt_rope *) base_rpp_node;
    // clear buffers
    rpp_node->binding_io_buffers.clear();
    for (auto iter : rpp_node->pool_buffers) {
        ctx.pool().free(iter);
    }
    rpp_node->pool_buffers.clear();

    if (!rpp_node->ori_rpp_node) {
        size_t sin_size = ggml_nbytes(rpp_node->ggml_sin.get());
        size_t cos_size = ggml_nbytes(rpp_node->ggml_cos.get());
        memset(rpp_node->sin_data, 0, sin_size);
        memset(rpp_node->cos_data, 0, cos_size);
        ggml_cpu_rope_get_sincos(rpp_node->cur_ggml_tensor, rpp_node->sin_data, rpp_node->cos_data);
    }

    for (int i = 0; i < rpp_node->engine->getNbBindings(); i++) {
        ggml_tensor * io_tensor = rpp_node->binding_io_tensors[i];
        void *        io_buffer = io_tensor->data;
        if (!ggml_is_contiguous(io_tensor)) {
            infer1::Dims dims    = rpp_node->engine->getBindingDimensions(i);
            size_t       io_size = ggml_rpp_nbytes(dims, ggml_type_size(io_tensor->type));
            io_buffer            = ctx.pool().alloc(io_size);
            rpp_node->pool_buffers.emplace(io_buffer);
            ggml_rpp_pack_tensor_to_contiguous(ctx, io_tensor, io_buffer);
        }
        if (rpp_node->engine->bindingIsInput(i)) {
            rpp_node->binding_i_buffers[io_tensor] = io_buffer;
        } else {
            rpp_node->binding_o_buffers[io_tensor] = io_buffer;
        }
        rpp_node->binding_io_buffers.emplace_back(io_buffer);
        ctx.rpp_io_buffers.emplace(io_tensor, io_buffer);
    }
    return true;
}

static bool ggml_rpp_set_io_datas_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * base_rpp_node) {
    GGML_ASSERT(base_rpp_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_rope_datas");
    rpp_openrt_rope * rpp_node = (rpp_openrt_rope *) base_rpp_node;
    // get rope sin/cos data from cpu
    if (!rpp_node->ori_rpp_node) {
        size_t sin_size = ggml_nbytes(rpp_node->ggml_sin.get());
        size_t cos_size = ggml_nbytes(rpp_node->ggml_cos.get());
        memset(rpp_node->sin_data, 0, sin_size);
        memset(rpp_node->cos_data, 0, cos_size);
        ggml_cpu_rope_get_sincos(rpp_node->cur_ggml_tensor, rpp_node->sin_data, rpp_node->cos_data);
    }
    for (auto iter : rpp_node->binding_i_buffers) {
        if (!ggml_is_contiguous(iter.first)) {
            ggml_rpp_pack_tensor_to_contiguous(ctx, iter.first, iter.second);
        }
    }
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

bool ggml_rpp_op_openrt_rope(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_openrt_rope * rpp_node = nullptr;
    auto              iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_rope");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                // use batch node, inputs need padding 0
                if (ctx.use_ubatch && ggml_rpp_rope_seq_len(dst, cur_node) > 1) {
                    if (cur_node->n_ubatch == ctx.n_ubatch) {
                        rpp_node = (rpp_openrt_rope *) cur_node;
                        break;
                    }
                } else {
                    if (ggml_rpp_node_has_matching_properties(iter_node->first, cur_node)) {
                        rpp_node = (rpp_openrt_rope *) cur_node;
                        break;
                    }
                }
            }
        }
        // find rpp_node from other graph, because of rms_norm can be shared if have same dims
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_rope");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                if (!rpp_graph_tmp) {
                    continue;
                }
                for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter.first == dst) {
                        continue;
                    }
                    if (node_iter.first->op == GGML_OP_ROPE && ggml_rpp_dims_is_same(node_iter.first, dst)) {
                        auto & node_vec = node_iter.second;
                        for (size_t i = 0; i < node_vec.size(); i++) {
                            auto cur_node = node_vec[i].get();
                            if (ggml_rpp_rope_dims_is_same(dst, node_iter.first, cur_node)) {
                                auto new_node = std::make_unique<rpp_openrt_rope>(dst, cur_node);
                                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                                rpp_node = (rpp_openrt_rope *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());

                                rpp_node->ggml_cos = ((rpp_openrt_rope *) cur_node)->ggml_cos;
                                rpp_node->ggml_sin = ((rpp_openrt_rope *) cur_node)->ggml_sin;
                                rpp_node->cos_data = ((rpp_openrt_rope *) cur_node)->cos_data;
                                rpp_node->sin_data = ((rpp_openrt_rope *) cur_node)->sin_data;

                                // set io tensor
                                rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
                                rpp_node->binding_io_tensors.emplace_back(rpp_node->ggml_cos.get());
                                rpp_node->binding_io_tensors.emplace_back(rpp_node->ggml_sin.get());
                                rpp_node->binding_io_tensors.emplace_back(dst);
                                // set node properties
                                ggml_rpp_node_set_properties(rpp_node, dst);
                                // set ubatch
                                rpp_node->n_ubatch      = cur_node->n_ubatch;
                                rpp_node->seq_len_index = cur_node->seq_len_index;
                                break;
                            }
                        }
                    }
                    if (rpp_node) {
                        break;
                    }
                }
                if (rpp_node) {
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_openrt_rope");
            auto new_node = std::make_unique<rpp_openrt_rope>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_openrt_rope *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            if (!(ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ggml_rpp_set_io_bingdings_device(ctx, rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_openrt_rope *) (iter->second);
        ggml_rpp_set_io_datas_device(ctx, rpp_node);
        if (!ctx.rpp_io_buffers.count(dst)) {
            ctx.rpp_io_buffers[dst] = rpp_node->binding_o_buffers[dst];
        }
    }

    // compute rope operator
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "execute_openrt_rope");
        RPP_EXECUTE_CONTEXT(rpp_node->context, 1, rpp_node->binding_io_buffers.data(), ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }

    return true;
}
#endif  // __linux__
