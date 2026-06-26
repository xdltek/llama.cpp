#include "ggml-cpu.h"
#include "ggml-cpu/ops.h"
#include "rpp_permute/rpp_permute.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#if GGML_RPP_USE_RT

static void ggml_rpp_calculate_permutation_robust(const int * input_shape,
                                                  const int * output_shape,
                                                  int *       permutation,
                                                  int         ndims) {
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

static infer1::Dims ggml_rpp_calculate_input_dims(ggml_tensor * input, int ndims) {
    GGML_ASSERT(ndims <= GGML_MAX_DIMS);

    struct Item {
        size_t value;
        int    idx;
    };

    Item items[GGML_MAX_DIMS];
    int  params[GGML_MAX_DIMS];

    for (int i = 0; i < ndims; ++i) {
        items[i] = { input->nb[i], i };
    }

    std::sort(items, items + ndims, [](const Item & a, const Item & b) { return a.value <= b.value; });

    for (int i = 0; i < ndims; ++i) {
        params[i] = items[i].idx;
    }

    infer1::Dims dims{};
    dims.nbDims = ndims;
    for (int i = 0; i < ndims; i++) {
        dims.d[i] = input->ne[params[ndims - i - 1]];
    }
    return dims;
}

static inline infer1::Permutation ggml_rpp_make_perm_from_ggml_params(const ggml_tensor * perm_dst, int n_dims) {
    infer1::Permutation perm{};
    for (int j = 0; j < n_dims; ++j) {
        perm.order[j] = n_dims - perm_dst->op_params[n_dims - 1 - j] - 1;
    }
    perm.order[n_dims] = -1;
    return perm;
}

static int ggml_rpp_permute_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    GGML_ASSERT(node);
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_set_io_bingdings_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_openrt_permute_bingdings");
    auto rpp_node = static_cast<rpp_openrt_permute *>(rpp_base_node);
    rpp_node->binding_io_buffers.clear();

    for (auto iter : rpp_node->pool_buffers) {
        ctx.pool().free(iter);
    }
    rpp_node->pool_buffers.clear();

    const int nb = rpp_node->engine->getNbBindings();
    for (int i = 0; i < nb; i++) {
        GGML_ASSERT(i < (int) rpp_node->binding_io_tensors.size());

        ggml_tensor * io_tensor = rpp_node->binding_io_tensors[i];
        void *        io_buffer = io_tensor->data;

        if (rpp_node->engine->bindingIsInput(i)) {
            // 地址复用（同 flash-attn）
            auto it = ctx.rpp_io_buffers.find(io_tensor);
            if (it != ctx.rpp_io_buffers.end()) {
                io_buffer = it->second;
            } else {
                // === ubatch 模式下：如果 engine 的输入 binding 维度要求的 bytes > 实际 ggml tensor bytes
                // 则分配临时 pool buffer，先清零，再拷贝实际数据（与 flash-attn mask 那段一致）
                if (ctx.use_ubatch && ggml_rpp_permute_seq_len(rpp_node->cur_ggml_tensor, rpp_node) > 1) {
                    infer1::Dims dims    = rpp_node->engine->getBindingDimensions(i);
                    size_t       io_size = ggml_rpp_nbytes(dims, ggml_type_size(io_tensor->type));

                    if (io_size > ggml_nbytes(io_tensor)) {
                        void *              tmp = ctx.pool().alloc(io_size);
                        std::vector<int8_t> zeros(io_size, 0);
                        RPP_MEMCPY_DEV_AND_HOST(tmp, zeros.data(), io_size, rtMemcpyHostToDevice, ctx.stream());

                        size_t actual_size = ggml_nbytes(io_tensor);
                        RPP_MEMCPY_DEV_AND_HOST(tmp, io_tensor->data, actual_size, rtMemcpyDeviceToDevice,
                                                ctx.stream());

                        io_buffer = tmp;
                        rpp_node->pool_buffers.emplace(tmp);
                    }
                }
            }
            rpp_node->binding_i_buffers[io_tensor] = io_buffer;
        } else {
            rpp_node->binding_o_buffers[io_tensor] = io_buffer;
        }

        rpp_node->binding_io_buffers.emplace_back(io_buffer);
        ctx.rpp_io_buffers[io_tensor] = io_buffer;
    }

    return true;
}

static bool ggml_rpp_create_engine_permute(ggml_backend_rpp_context & ctx,
                                           ggml_rpp_node *            rpp_base_node,
                                           ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->src[0]);

    auto rpp_node    = static_cast<rpp_openrt_permute *>(rpp_base_node);
    auto rpp_network = rpp_node->network.get();

    const int     n_dims = GGML_MAX_DIMS - 1;
    ggml_tensor * src    = dst->src[0];

    infer1::ITensor * input_tensor    = nullptr;
    infer1::ITensor * permuted_tensor = nullptr;

    if (!ggml_is_contiguous(src)) {
        infer1::Dims i_dims = ggml_rpp_calculate_input_dims(src, n_dims);
        infer1::Dims o_dims = ggml_rpp_dims_mapping(src, n_dims);

        infer1::Permutation perm{};
        for (int k = 0; k < n_dims; ++k) {
            perm.order[k] = 0;
        }
        ggml_rpp_calculate_permutation_robust(i_dims.d, o_dims.d, perm.order, n_dims);
        perm.order[n_dims] = -1;

        input_tensor = ggml_rpp_create_input_tensor(src, rpp_node, i_dims);
        if (!input_tensor) {
            GGML_LOG_ERROR("%s: create input failed (non-contig) for permute, %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
            return false;
        }

        auto shuf0 = rpp_network->addShuffle(*input_tensor);
        if (!shuf0) {
            GGML_LOG_ERROR("%s: addShuffle failed (non-contig) for permute, %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
            return false;
        }
        shuf0->setFirstTranspose(perm);
        shuf0->setRealLayerType(infer1::LayerType::kTRANSPOSE);
        permuted_tensor = shuf0->getOutput(0);
        permuted_tensor->setName(src->name);
    } else {
        input_tensor = ggml_rpp_create_input_tensor(src, rpp_node, n_dims);
        if (!input_tensor) {
            GGML_LOG_ERROR("%s: create input failed (contig) for permute, %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
            return false;
        }
        permuted_tensor = input_tensor;
    }

    infer1::Permutation perm1 = ggml_rpp_make_perm_from_ggml_params(dst, n_dims);

    auto shuf1 = rpp_network->addShuffle(*permuted_tensor);
    if (!shuf1) {
        GGML_LOG_ERROR("%s: addShuffle failed for permute(op_params), %s (%s)\n", __func__, dst->name,
                       ggml_op_name(dst->op));
        return false;
    }
    shuf1->setFirstTranspose(perm1);
    shuf1->setRealLayerType(infer1::LayerType::kTRANSPOSE);

    infer1::ITensor * out = shuf1->getOutput(0);
    out->setName(dst->name);
    rpp_network->markOutput(*out);

    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine failed for permute, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create context failed for permute, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }

    return true;
}

static bool ggml_rpp_create_engine_permute_ubatch(ggml_backend_rpp_context & ctx,
                                                  ggml_rpp_node *            rpp_base_node,
                                                  ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->src[0]);

    auto rpp_node    = static_cast<rpp_openrt_permute *>(rpp_base_node);
    auto rpp_network = rpp_node->network.get();

    const int     n_dims = GGML_MAX_DIMS - 1;
    ggml_tensor * src    = dst->src[0];

    // ubatch 维度索引（建议你在 rpp_openrt_permute 里加字段；没加就固定成 1）
    int ubatch_dim_index = 1;
    // 如果你已经在 rpp_openrt_permute 里加了该字段，就打开这行：
    // ubatch_dim_index = rpp_node->ubatch_dim_index;

    infer1::ITensor * input_tensor    = nullptr;
    infer1::ITensor * permuted_tensor = nullptr;

    // 非 contiguous：layout 修正 + ubatch 扩维
    if (!ggml_is_contiguous(src)) {
        infer1::Dims i_dims = ggml_rpp_calculate_input_dims(src, n_dims);
        infer1::Dims o_dims = ggml_rpp_dims_mapping(src, n_dims);

        // === ubatch 扩维：把某个维度改成 n_ubatch
        GGML_ASSERT(ubatch_dim_index >= 0 && ubatch_dim_index < n_dims);
        i_dims.d[ubatch_dim_index] = rpp_node->n_ubatch;

        infer1::Permutation perm{};
        for (int k = 0; k < n_dims; ++k) {
            perm.order[k] = 0;
        }
        ggml_rpp_calculate_permutation_robust(i_dims.d, o_dims.d, perm.order, n_dims);
        perm.order[n_dims] = -1;

        input_tensor = ggml_rpp_create_input_tensor(src, rpp_node, i_dims);
        if (!input_tensor) {
            GGML_LOG_ERROR("%s: create input failed (non-contig ubatch) for permute, %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
            return false;
        }

        auto shuf0 = rpp_network->addShuffle(*input_tensor);
        if (!shuf0) {
            GGML_LOG_ERROR("%s: addShuffle failed (non-contig ubatch) for permute, %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
            return false;
        }
        shuf0->setFirstTranspose(perm);
        shuf0->setRealLayerType(infer1::LayerType::kTRANSPOSE);
        permuted_tensor = shuf0->getOutput(0);
        permuted_tensor->setName(src->name);
    } else {
        // contiguous：直接 n_dims dims，但要先做 ubatch 扩维
        infer1::Dims i_dims = ggml_rpp_dims_mapping(src, n_dims);  // 这里用 mapping 得到“逻辑 dims”
        GGML_ASSERT(ubatch_dim_index >= 0 && ubatch_dim_index < n_dims);
        i_dims.d[ubatch_dim_index] = rpp_node->n_ubatch;

        input_tensor = ggml_rpp_create_input_tensor(src, rpp_node, i_dims);
        if (!input_tensor) {
            GGML_LOG_ERROR("%s: create input failed (contig ubatch) for permute, %s (%s)\n", __func__, dst->name,
                           ggml_op_name(dst->op));
            return false;
        }
        permuted_tensor = input_tensor;
    }

    // 真正的 permute（来自 dst->op_params）
    infer1::Permutation perm1 = ggml_rpp_make_perm_from_ggml_params(dst, n_dims);

    auto shuf1 = rpp_network->addShuffle(*permuted_tensor);
    if (!shuf1) {
        GGML_LOG_ERROR("%s: addShuffle failed for permute(op_params ubatch), %s (%s)\n", __func__, dst->name,
                       ggml_op_name(dst->op));
        return false;
    }
    shuf1->setFirstTranspose(perm1);
    shuf1->setRealLayerType(infer1::LayerType::kTRANSPOSE);

    infer1::ITensor * out = shuf1->getOutput(0);
    out->setName(dst->name);
    rpp_network->markOutput(*out);

    rpp_node->engine.reset(rpp_node->builder->buildEngineWithConfig(*(rpp_node->network), *(rpp_node->config)));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: build engine failed for permute(ubatch), %s (%s)\n", __func__, dst->name,
                       ggml_op_name(dst->op));
        return false;
    }

    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create context failed for permute(ubatch), %s (%s)\n", __func__, dst->name,
                       ggml_op_name(dst->op));
        return false;
    }

    return true;
}

static bool ggml_rpp_create_engine_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_node);
    GGML_ASSERT(dst);
    bool ret = false;
    // === ubatch 初始化（完全仿 flash-attn 结构）
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n = ggml_n_dims(dst);
        GGML_ASSERT(n >= 2);
        rpp_node->seq_len_index = n == 2 ? 2 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }

    // 设置 n_ubatch（seq_len > 1 才启用 ubatch）
    if (ctx.use_ubatch && ggml_rpp_permute_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    } else {
        rpp_node->n_ubatch = 1;
    }

    std::string save_path = STRINGIZE(GGML_RPP_SAVE_ENGINE);
    std::string load_path = STRINGIZE(GGML_RPP_LOAD_ENGINE);

    // === load engine：把 ubatch 信息纳入文件名（与 flash-attn 一致）
    if (!load_path.empty()) {
        char engine_path[128];
        sprintf(engine_path, "%s/ubatch%d_%s_%s.engine", load_path.c_str(), rpp_node->n_ubatch, ggml_op_name(dst->op),
                dst->name);
        ret = ggml_rpp_load_enigne(rpp_node, engine_path);
    }

    // === create engine
    if (!ret) {
        if (ctx.use_ubatch && ggml_rpp_permute_seq_len(dst, rpp_node) > 1) {
            ret = ggml_rpp_create_engine_permute_ubatch(ctx, rpp_node, dst);
        } else {
            ret = ggml_rpp_create_engine_permute(ctx, rpp_node, dst);
        }

        if (ret && !save_path.empty()) {
            char engine_path[128];
            sprintf(engine_path, "%s/ubatch%d_%s_%s.engine", save_path.c_str(), rpp_node->n_ubatch,
                    ggml_op_name(dst->op), dst->name);
            ggml_rpp_save_enigne(rpp_node, engine_path);
        }
    }

    // set properties + io tensors
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.clear();
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);  // input
        rpp_node->binding_io_tensors.emplace_back(dst);          // output
        // 供 binding 阶段读取 seq_len 的当前 ggml tensor
        rpp_node->cur_ggml_tensor = dst;
    }

    return ret;
}

bool ggml_rpp_op_openrt_permute(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    if (!dst || !dst->src[0]) {
        GGML_LOG_ERROR("%s: invalid tensor for permute\n", __func__);
        return false;
    }
    rpp_openrt_permute * rpp_node = nullptr;
    auto                 iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        // 1) current graph 里找可复用 node（properties matching）
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_openrt_permute");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node && ggml_rpp_node_has_matching_properties(dst, cur_node)) {
                    rpp_node = (rpp_openrt_permute *) cur_node;
                    break;
                }
            }
        }

        // 2) 跨图复用（可选，保持你上一版逻辑；这里略过“克隆构造”的细节依赖）
        // 若你已实现 rpp_openrt_permute(dst, other_node) 克隆，可把上一版跨图段落加回来

        // 3) 创建新 node + dispatch
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_openrt_permute");
            auto new_node = std::make_unique<rpp_openrt_permute>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node = (rpp_openrt_permute *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            if (!ggml_rpp_create_engine_dispatch(ctx, rpp_node, dst)) {
                return false;
            }
        }

        GGML_ASSERT(rpp_node);
        // set bindings
        ggml_rpp_set_io_bingdings_device(ctx, rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_openrt_permute *) (iter->second);
        if (!ctx.rpp_io_buffers.count(dst)) {
            ctx.rpp_io_buffers[dst] = rpp_node->binding_o_buffers[dst];
        }
    }

    // execute
    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "launch_openrt_permute");
        RPP_EXECUTE_CONTEXT(rpp_node->context, 1, rpp_node->binding_io_buffers.data(), ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
        return false;
    }

    return true;
}
#endif  // __linux__
