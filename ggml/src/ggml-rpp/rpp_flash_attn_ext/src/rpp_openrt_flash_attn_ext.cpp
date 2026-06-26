#include "ggml-cpu.h"
#include "ggml-cpu/ops.h"
#include "rpp_flash_attn_ext/rpp_flash_attn_ext.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#if GGML_RPP_USE_RT

static void calculate_permutation_robust(int * input_shape, int * output_shape, int * permutation, int ndims) {
    GGML_ASSERT(ndims <= GGML_MAX_DIMS);
    for (int out_idx = 0; out_idx < ndims; out_idx++) {
        for (int in_idx = 0; in_idx < ndims; in_idx++) {
            if (input_shape[in_idx] == output_shape[out_idx]) {
                bool already_used = false;
                for (int k = 0; k < out_idx; k++) {
                    if (permutation[k] == in_idx) {
                        already_used = true;
                        break;
                    }
                }
                if (!already_used) {
                    permutation[out_idx] = in_idx;
                    break;
                }
            }
        }
    }
}

static infer1::Dims calculate_input_dims(ggml_tensor * input, int ndims = GGML_MAX_DIMS) {
    GGML_ASSERT(ndims <= GGML_MAX_DIMS);

    struct Item {
        size_t value;
        int    idx;
    };

    int  params[ndims];
    Item items[ndims];
    for (int i = 0; i < ndims; ++i) {
        items[i] = { input->nb[i], i };
    }
    // according nb to calculate the actual index value
    std::sort(items, items + ndims, [](const Item & a, const Item & b) { return a.value <= b.value; });
    for (int i = 0; i < ndims; ++i) {
        params[i] = items[i].idx;
    }
    // get the dims of input
    infer1::Dims dims;
    dims.nbDims = ndims;
    for (int i = 0; i < ndims; i++) {
        dims.d[i] = input->ne[params[ndims - i - 1]];
    }
    return dims;
}

static int ggml_rpp_flash_attention_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    GGML_ASSERT(node);
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_flash_attention_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
    bool          dims_is_same              = true;
    ggml_tensor * dst_vec[GGML_MAX_SRC + 1] = { dst };
    auto          dst_iter =
        std::copy_if(dst->src, dst->src + GGML_MAX_SRC, dst_vec + 1, [](ggml_tensor * ptr) { return ptr != nullptr; });

    ggml_tensor * src_vec[GGML_MAX_SRC + 1] = { src };
    auto          src_iter =
        std::copy_if(src->src, src->src + GGML_MAX_SRC, src_vec + 1, [](ggml_tensor * ptr) { return ptr != nullptr; });
    int dst_size = dst_iter - dst_vec;
    int src_size = src_iter - src_vec;
    if (dst_size != src_size || rpp_node->ggml_node_properties.size() != src_size) {
        return false;
    }
    for (int i = 0; i < dst_size; i++) {
        auto & property = rpp_node->ggml_node_properties[src_vec[i]];
        auto & cur_dst  = dst_vec[i];
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (cur_dst->ne[i] != property.ne[i]) {
                dims_is_same = false;
                break;
            }
        }
        if (!dims_is_same) {
            break;
        }
    }
    return dims_is_same;
}

static bool ggml_rpp_flash_attention_kv_dims_is_same(ggml_tensor * dst, ggml_tensor * src, ggml_rpp_node * rpp_node) {
    bool dims_is_same = true;
    // 1 is k, and 2 is v
    for (int i = 1; i < 3; i++) {
        auto & property = rpp_node->ggml_node_properties[src->src[i]];
        auto & cur_dst  = dst->src[i];
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (cur_dst->ne[i] != property.ne[i]) {
                dims_is_same = false;
                break;
            }
        }
        if (!dims_is_same) {
            break;
        }
    }
    return dims_is_same;
}

static bool ggml_rpp_create_engine_flash_attention_ubatch(ggml_backend_rpp_context & ctx,
                                                          ggml_rpp_node *            rpp_base_node,
                                                          ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node    = static_cast<rpp_openrt_flash_attn_ext *>(rpp_base_node);
    auto rpp_graph   = ctx.cur_rpp_graph;
    auto rpp_network = rpp_node->network.get();

    // notice!!! must be 3 dims for flash attention
    int               n_dims           = GGML_MAX_DIMS - 1;
    infer1::ITensor * inputs[4]        = { nullptr, nullptr, nullptr, nullptr };
    infer1::ITensor * permut_tensor[3] = { nullptr, nullptr, nullptr };
    // create input tensors for q,k,v
    for (int i = 0; i < 3; i++) {
        if (dst->src[i]->op == GGML_OP_PERMUTE) {
            // is q, need to expand ubatch dimension
            if (i == 0) {
                infer1::Dims dims_q;
                dims_q.nbDims = n_dims;
                dims_q.d[0]   = rpp_node->n_ubatch;
                dims_q.d[1]   = dst->src[0]->src[0]->ne[1];
                dims_q.d[2]   = dst->src[0]->src[0]->ne[0];
                inputs[i]     = ggml_rpp_create_input_tensor(dst->src[i]->src[0], rpp_node, dims_q);
                if (!inputs[i]) {
                    GGML_LOG_ERROR("%s: creat  input-%d failed for flash_attention, %s (%s)\n", __func__, i, dst->name,
                                   ggml_op_name(dst->op));
                    return false;
                }
            } else {  // is k,v, need not expand ubatch dimension
                inputs[i] = ggml_rpp_create_input_tensor(dst->src[i]->src[0], rpp_node, n_dims);
                if (!inputs[i]) {
                    GGML_LOG_ERROR("%s: creat  input-%d failed for flash_attention, %s (%s)\n", __func__, i, dst->name,
                                   ggml_op_name(dst->op));
                    return false;
                }
            }
            infer1::Permutation permute;
            for (int j = 0; j < n_dims; j++) {
                permute.order[j] = n_dims - dst->src[i]->op_params[n_dims - 1 - j] - 1;
            }
            permute.order[n_dims] = -1;
            auto permute_layer    = rpp_network->addShuffle(*inputs[i]);
            if (!permute_layer) {
                GGML_LOG_ERROR("%s: addShuffle permute faied for flash_attn input-%d, %s (%s)\n", __func__, i,
                               dst->name, ggml_op_name(dst->op));
                return false;
            }
            permute_layer->setFirstTranspose(permute);
            permute_layer->setRealLayerType(infer1::LayerType::kTRANSPOSE);
            permut_tensor[i] = permute_layer->getOutput(0);
            permut_tensor[i]->setName(dst->src[i]->name);
        } else if (!ggml_is_contiguous(dst->src[i])) {
            // because not contiguous, so get permute info from src tensor, and create permute layer
            infer1::Dims        i_dims = calculate_input_dims(dst->src[i], n_dims);
            infer1::Dims        o_dims = ggml_rpp_dims_mapping(dst->src[i], n_dims);
            infer1::Permutation permute;
            calculate_permutation_robust(i_dims.d, o_dims.d, permute.order, n_dims);
            permute.order[n_dims] = -1;
            // is q, need to expand ubatch dimension
            if (i == 0) {
                i_dims.d[0] = rpp_node->n_ubatch;
            }
            inputs[i]          = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, i_dims);
            auto permute_layer = rpp_network->addShuffle(*inputs[i]);
            if (!permute_layer) {
                GGML_LOG_ERROR("%s: addShuffle permute faied for flash_attn input-%d, %s (%s)\n", __func__, i,
                               dst->name, ggml_op_name(dst->op));
                return false;
            }
            permute_layer->setFirstTranspose(permute);
            permute_layer->setRealLayerType(infer1::LayerType::kTRANSPOSE);
            permut_tensor[i] = permute_layer->getOutput(0);
            permut_tensor[i]->setName(dst->src[i]->name);
        } else {
            if (i == 0) {
                // is q, need to expand ubatch dimension
                infer1::Dims dims_q;
                dims_q.nbDims = n_dims;
                dims_q.d[0]   = dst->src[i]->src[0]->ne[2];
                dims_q.d[1]   = rpp_node->n_ubatch;
                dims_q.d[2]   = dst->src[i]->src[0]->ne[0];
                inputs[i]     = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, dims_q);
                if (!inputs[i]) {
                    GGML_LOG_ERROR("%s: creat  input-%d failed for flash_attention, %s (%s)\n", __func__, i, dst->name,
                                   ggml_op_name(dst->op));
                    return false;
                }
            } else {
                inputs[i] = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, n_dims);
                if (!inputs[i]) {
                    GGML_LOG_ERROR("%s: creat  input-%d failed for flash_attention, %s (%s)\n", __func__, i, dst->name,
                                   ggml_op_name(dst->op));
                    return false;
                }
            }
        }
    }
    // create mask input
    infer1::Dims dims_mask;
    dims_mask.nbDims = n_dims;
    dims_mask.d[0]   = 1;
    dims_mask.d[1]   = rpp_node->n_ubatch;
    dims_mask.d[2]   = dst->src[3]->ne[0];
    inputs[3]        = ggml_rpp_create_input_tensor(dst->src[3], rpp_node, dims_mask);
    if (!inputs[3]) {
        GGML_LOG_ERROR("%s: creat  input-3 failed for flash_attention, %s (%s)\n", __func__, dst->name,
                       ggml_op_name(dst->op));
        return false;
    }

    // create flash attention operator
    float scale = 1.0f;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));

    auto input_q    = (permut_tensor[0] != nullptr) ? permut_tensor[0] : inputs[0];
    auto input_k    = (permut_tensor[1] != nullptr) ? permut_tensor[1] : inputs[1];
    auto input_v    = (permut_tensor[2] != nullptr) ? permut_tensor[2] : inputs[2];
    auto input_mask = inputs[3];
    // create add operator
    auto opt_flash_attention =
        rpp_network->addFlashAttn(*input_q, *input_k, *input_v, *input_mask, scale, ggml_rpp_dtype_mapping(dst->type));
    if (!opt_flash_attention) {
        GGML_LOG_ERROR("%s: addFlashAttn faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = opt_flash_attention->getOutput(0);
    ort_tensor->setName(dst->name);
    rpp_network->markOutput(*ort_tensor);

    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_create_engine_flash_attention(ggml_backend_rpp_context & ctx,
                                                   ggml_rpp_node *            rpp_base_node,
                                                   ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node    = static_cast<rpp_openrt_flash_attn_ext *>(rpp_base_node);
    auto rpp_graph   = ctx.cur_rpp_graph;
    auto rpp_network = rpp_node->network.get();

    // notice!!! must be 3 dims for flash attention
    int               n_dims           = GGML_MAX_DIMS - 1;
    infer1::ITensor * inputs[4]        = { nullptr, nullptr, nullptr, nullptr };
    infer1::ITensor * permut_tensor[3] = { nullptr, nullptr, nullptr };
    // create input tensors for q,k,v
    for (int i = 0; i < 3; i++) {
        if (dst->src[i]->op == GGML_OP_PERMUTE) {
            // is q,k,v, need to permute
            inputs[i] = ggml_rpp_create_input_tensor(dst->src[i]->src[0], rpp_node, n_dims);
            if (!inputs[i]) {
                GGML_LOG_ERROR("%s: creat  input-%d failed for flash_attention, %s (%s)\n", __func__, i, dst->name,
                               ggml_op_name(dst->op));
                return false;
            }
            infer1::Permutation permute;
            for (int j = 0; j < n_dims; j++) {
                permute.order[j] = n_dims - dst->src[i]->op_params[n_dims - 1 - j] - 1;
            }
            permute.order[n_dims] = -1;
            auto permute_layer    = rpp_network->addShuffle(*inputs[i]);
            if (!permute_layer) {
                GGML_LOG_ERROR("%s: addShuffle permute faied for flash_attn input-%d, %s (%s)\n", __func__, i,
                               dst->name, ggml_op_name(dst->op));
                return false;
            }
            permute_layer->setFirstTranspose(permute);
            permute_layer->setRealLayerType(infer1::LayerType::kTRANSPOSE);
            permut_tensor[i] = permute_layer->getOutput(0);
            permut_tensor[i]->setName(dst->src[i]->name);
        } else if (!ggml_is_contiguous(dst->src[i])) {
            // because not contiguous, so get permute info from src tensor, and create permute layer
            infer1::Dims        i_dims = calculate_input_dims(dst->src[i], n_dims);
            infer1::Dims        o_dims = ggml_rpp_dims_mapping(dst->src[i], n_dims);
            infer1::Permutation permute;
            calculate_permutation_robust(i_dims.d, o_dims.d, permute.order, n_dims);
            permute.order[n_dims] = -1;
            inputs[i]             = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, i_dims);
            auto permute_layer    = rpp_network->addShuffle(*inputs[i]);
            if (!permute_layer) {
                GGML_LOG_ERROR("%s: addShuffle permute faied for flash_attn input-%d, %s (%s)\n", __func__, i,
                               dst->name, ggml_op_name(dst->op));
                return false;
            }
            permute_layer->setFirstTranspose(permute);
            permute_layer->setRealLayerType(infer1::LayerType::kTRANSPOSE);
            permut_tensor[i] = permute_layer->getOutput(0);
            permut_tensor[i]->setName(dst->src[i]->name);
        } else {
            inputs[i] = ggml_rpp_create_input_tensor(dst->src[i], rpp_node, n_dims);
            if (!inputs[i]) {
                GGML_LOG_ERROR("%s: creat  input-%d failed for flash_attention, %s (%s)\n", __func__, i, dst->name,
                               ggml_op_name(dst->op));
                return false;
            }
        }
    }
    // create mask input
    infer1::Dims dims_mask;
    dims_mask.nbDims = n_dims;
    dims_mask.d[0]   = 1;
    dims_mask.d[1]   = dst->ne[rpp_node->seq_len_index];
    dims_mask.d[2]   = dst->src[3]->ne[0];
    inputs[3]        = ggml_rpp_create_input_tensor(dst->src[3], rpp_node, dims_mask);
    if (!inputs[3]) {
        GGML_LOG_ERROR("%s: creat  input-3 failed for flash_attention, %s (%s)\n", __func__, dst->name,
                       ggml_op_name(dst->op));
        return false;
    }

    // create flash attention operator
    float scale = 1.0f;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));

    auto input_q    = (permut_tensor[0] != nullptr) ? permut_tensor[0] : inputs[0];
    auto input_k    = (permut_tensor[1] != nullptr) ? permut_tensor[1] : inputs[1];
    auto input_v    = (permut_tensor[2] != nullptr) ? permut_tensor[2] : inputs[2];
    auto input_mask = inputs[3];
    // create add operator
    auto opt_flash_attention =
        rpp_network->addFlashAttn(*input_q, *input_k, *input_v, *input_mask, scale, ggml_rpp_dtype_mapping(dst->type));
    if (!opt_flash_attention) {
        GGML_LOG_ERROR("%s: addFlashAttn faied, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    infer1::ITensor * ort_tensor = opt_flash_attention->getOutput(0);
    ort_tensor->setName(dst->name);
    rpp_network->markOutput(*ort_tensor);

    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine with config failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->engine);

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    GGML_ASSERT(rpp_node->context);
    return true;
}

static bool ggml_rpp_set_io_bingdings_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_flash_attn_ext_bingdings");
    auto rpp_node = static_cast<rpp_openrt_flash_attn_ext *>(rpp_base_node);
    // clear buffers
    rpp_node->binding_io_buffers.clear();
    for (auto iter : rpp_node->pool_buffers) {
        ctx.pool().free(iter);
    }
    rpp_node->pool_buffers.clear();
    for (int i = 0; i < rpp_node->engine->getNbBindings(); i++) {
        ggml_tensor * io_tensor = rpp_node->binding_io_tensors[i];
        void *        io_buffer = io_tensor->data;
        if (rpp_node->engine->bindingIsInput(i)) {
            if (i == 3) {  // mask input
                auto iter = ctx.rpp_io_buffers.find(io_tensor);
                if (iter != ctx.rpp_io_buffers.end()) {
                    io_buffer = ctx.rpp_io_buffers[io_tensor];
                } else {
                    if (ctx.use_ubatch && ggml_rpp_flash_attention_seq_len(rpp_node->cur_ggml_tensor, rpp_node) > 1) {
                        infer1::Dims dims    = rpp_node->engine->getBindingDimensions(i);
                        size_t       io_size = ggml_rpp_nbytes(dims, ggml_type_size(io_tensor->type));
                        io_buffer            = ctx.pool().alloc(io_size);
                        std::vector<int8_t> zeros(io_size, 0);
                        // std::vector<int16_t> zeros(io_size/2, 0xFC00);
                        RPP_MEMCPY_DEV_AND_HOST(io_buffer, zeros.data(), io_size, rtMemcpyHostToDevice, ctx.stream());
                        size_t actual_size = io_size < ggml_nbytes(io_tensor) ? io_size : ggml_nbytes(io_tensor);
                        RPP_MEMCPY_DEV_AND_HOST(io_buffer, io_tensor->data, actual_size, rtMemcpyDeviceToDevice,
                                                ctx.stream());
                        rpp_node->pool_buffers.emplace(io_buffer);
                    }
                }
            }
            rpp_node->binding_i_buffers[io_tensor] = io_buffer;
        } else {
            rpp_node->binding_o_buffers[io_tensor] = io_buffer;
        }
        rpp_node->binding_io_buffers.emplace_back(io_buffer);
        ctx.rpp_io_buffers.emplace(io_tensor, io_buffer);
    }
    return true;
}

static bool ggml_rpp_set_io_datas_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    auto rpp_node = static_cast<rpp_openrt_flash_attn_ext *>(rpp_base_node);
    for (int i = 0; i < rpp_node->engine->getNbBindings(); i++) {
        ggml_tensor * io_tensor = rpp_node->binding_io_tensors[i];
        void *        io_buffer = rpp_node->binding_i_buffers[io_tensor];
        if (rpp_node->engine->bindingIsInput(i) && i == 3 && !ctx.rpp_io_buffers.count(io_tensor) && ctx.use_ubatch &&
            ggml_rpp_flash_attention_seq_len(rpp_node->cur_ggml_tensor, rpp_node) > 1) {
            infer1::Dims        dims    = rpp_node->engine->getBindingDimensions(i);
            size_t              io_size = ggml_rpp_nbytes(dims, ggml_type_size(io_tensor->type));
            std::vector<int8_t> zeros(io_size, 0);
            // std::vector<int16_t> zeros(io_size/2, 0xFC00);
            RPP_MEMCPY_DEV_AND_HOST(io_buffer, zeros.data(), io_size, rtMemcpyHostToDevice, ctx.stream());
            size_t actual_size = io_size < ggml_nbytes(io_tensor) ? io_size : ggml_nbytes(io_tensor);
            RPP_MEMCPY_DEV_AND_HOST(io_buffer, io_tensor->data, actual_size, rtMemcpyDeviceToDevice, ctx.stream());
            ctx.rpp_io_buffers.emplace(io_tensor, io_buffer);
        }
    }
}

static bool ggml_rpp_create_engine_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_node);
    bool ret = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n = ggml_n_dims(dst);
        GGML_ASSERT(n >= 2);
        rpp_node->seq_len_index = n == 2 ? 2 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }

    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_flash_attention_seq_len(dst, rpp_node) > 1) {
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
        if (ctx.use_ubatch && ggml_rpp_flash_attention_seq_len(dst, rpp_node) > 1) {
            ret = ggml_rpp_create_engine_flash_attention_ubatch(ctx, rpp_node, dst);
        } else {
            ret = ggml_rpp_create_engine_flash_attention(ctx, rpp_node, dst);
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
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[2]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[3]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

bool ggml_rpp_op_openrt_flash_attn_ext(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    // notice!!! Because the dims in dst may be the same, so need further determine whether the input dims are consistent
    // if not same, erase the rpp_node, and rebuild a rpp_node
    // if (ctx.cur_rpp_graph->cur_rpp_nodes.count(dst)) {
    //     auto & cur_node = ctx.cur_rpp_graph->cur_rpp_nodes[dst];
    //     if (!ggml_rpp_node_has_matching_properties(dst, cur_node)) {
    //         ggml_rpp_reset_node(&ctx, ctx.cur_rpp_graph, dst);
    //         ctx.cur_rpp_graph->cur_rpp_nodes.erase(dst);
    //     }
    // }
    rpp_openrt_flash_attn_ext * rpp_node = nullptr;
    auto                        iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_flash_attn_ext");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (ctx.use_ubatch && cur_node && cur_node->n_ubatch != 1 &&
                    ggml_rpp_flash_attention_seq_len(dst, cur_node) > 1 &&
                    ggml_rpp_flash_attention_kv_dims_is_same(dst, dst, cur_node)) {
                    rpp_node = (rpp_openrt_flash_attn_ext *) cur_node;
                    break;
                } else {
                    if (ggml_rpp_node_has_matching_properties(iter_node->first, cur_node)) {
                        rpp_node = (rpp_openrt_flash_attn_ext *) cur_node;
                        break;
                    }
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_flash_attn_ext");
            for (auto & graph_iter : ctx.gglm_graphs) {
                ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
                if (!rpp_graph_tmp) {
                    continue;
                }
                for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
                    if (node_iter.first->op == GGML_OP_FLASH_ATTN_EXT && ggml_rpp_dims_is_same(node_iter.first, dst)) {
                        auto & node_vec = node_iter.second;
                        for (size_t i = 0; i < node_vec.size(); i++) {
                            auto cur_node = node_vec[i].get();
                            if (ggml_rpp_flash_attention_dims_is_same(dst, node_iter.first, cur_node)) {
                                auto new_node = std::make_unique<rpp_openrt_flash_attn_ext>(dst, cur_node);
                                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                                rpp_node =
                                    (rpp_openrt_flash_attn_ext *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());

                                // set io tensor
                                rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
                                rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
                                rpp_node->binding_io_tensors.emplace_back(dst->src[2]);
                                rpp_node->binding_io_tensors.emplace_back(dst->src[3]);
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
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_openrt_flash_attn_ext");
            auto new_node = std::make_unique<rpp_openrt_flash_attn_ext>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_openrt_flash_attn_ext *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            if (!(ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ggml_rpp_set_io_bingdings_device(ctx, rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_openrt_flash_attn_ext *) (iter->second);
        // ggml_rpp_set_io_datas_device(ctx, rpp_node);
        if (!ctx.rpp_io_buffers.count(dst)) {
            ctx.rpp_io_buffers[dst] = rpp_node->binding_o_buffers[dst];
        }
    }
    // compute falsh attention operator
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "execute_openrt_flash_attn_ext");
        RPP_EXECUTE_CONTEXT(rpp_node->context, 1, rpp_node->binding_io_buffers.data(), ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
    }
    return true;
}

#    if 0
struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};

bool ggml_rpp_op_openrt_flash_attn_ext(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    uint8_t *                  work_data = (uint8_t *) malloc(1000000);
    struct ggml_compute_params params    = {
        /*.ith       =*/0,
        /*.nth       =*/1,
        /*.wsize     =*/0,
        /*.wdata     =*/work_data,
        /*.threadpool=*/nullptr,
    };
    // ggml_compute_params *params = nullptr;
    ggml_compute_forward_flash_attn_ext(&params, dst);
    return false;
}
#    endif

#endif  // __linux__
