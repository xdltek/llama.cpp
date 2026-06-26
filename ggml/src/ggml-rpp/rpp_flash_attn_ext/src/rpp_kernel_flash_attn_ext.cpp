#include "rpp_flash_attn_ext/rpp_flash_attn_ext.h"
#include "rpp_flash_attn_ext/src/rpp_kernel_build.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

static int ggml_rpp_flash_attention_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    GGML_ASSERT(node);
    return dst->ne[node->seq_len_index];
}

static ggml_rpp_node * ggml_rpp_find_flash_attn_ext_node(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first != dst && node_iter.first->op == GGML_OP_FLASH_ATTN_EXT) {
                auto & node_vec = node_iter.second;
                for (size_t i = 0; i < node_vec.size(); i++) {
                    auto cur_node = node_vec[i].get();
                    return cur_node;
                }
            }
        }
    }
    return nullptr;
}

static bool ggml_rpp_create_engine_flash_attention(ggml_backend_rpp_context & ctx,
                                                   ggml_rpp_node *            rpp_base_node,
                                                   ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_flash_attn_ext *>(rpp_base_node);
    int  Tq = 0, Tk = 0, Hq = 0, Hkv = 0, D = 0;

    if (ctx.use_ubatch && ggml_rpp_flash_attention_seq_len(dst, rpp_node) > 1) {
        Tq = ctx.n_ubatch;
    } else {
        Tq = dst->ne[2];
    }

    Tk  = rpp_node->kv_length;
    Hq  = dst->src[0]->ne[2];
    Hkv = dst->src[1]->ne[2];
    D   = dst->ne[0];

    void * i_buffers[4] = {
        dst->src[0]->data,
        dst->src[1]->data,
        dst->src[2]->data,
        dst->src[3]->data,
    };

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) i_buffers[0]);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) i_buffers[1]);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) i_buffers[2]);
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) i_buffers[3]);

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[0], i_buffers[0]);
    rpp_node->binding_i_buffers.emplace(dst->src[1], i_buffers[1]);
    rpp_node->binding_i_buffers.emplace(dst->src[2], i_buffers[2]);
    rpp_node->binding_i_buffers.emplace(dst->src[3], i_buffers[3]);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(i_buffers[0]);
    rpp_node->binding_io_buffers.emplace_back(i_buffers[1]);
    rpp_node->binding_io_buffers.emplace_back(i_buffers[2]);
    rpp_node->binding_io_buffers.emplace_back(i_buffers[3]);
    rpp_node->binding_io_buffers.emplace_back(dst->data);
    // create flash attention operator
    float scale = 1.0f;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // find first rms_node, and all flash_attn_ext kernel will shared the same workspace
    auto ori_rpp_node = ggml_rpp_find_flash_attn_ext_node(ctx, dst);
    if (ori_rpp_node) {
        auto ori_rms_node = static_cast<rpp_kernel_flash_attn_ext *>(ori_rpp_node);
        rpp_node->init_workspace(ori_rms_node->kernel_ctx->dev_workspace);
    } else {
        rpp_node->init_workspace(ctx);
    }

    int kv_bytes_per_elem   = ggml_rpp_get_io_type_size(ctx, dst->src[1], 0);
    int io_bytes_per_elem   = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    int mask_bytes_per_elem = ggml_rpp_get_io_type_size(ctx, dst->src[3]);

    rpp_flash_atten_build(*(rpp_node->kernel_ctx.get()), Tq, Tk, Hq, Hkv, D, scale, kv_bytes_per_elem,
                          io_bytes_per_elem, mask_bytes_per_elem, rpp_node->is_instantial);

    return true;
}

static void ggml_rpp_node_flash_attn_ext_set_properties(ggml_rpp_node * rpp_node, ggml_tensor * dst) {
    ggml_tensor * cur_tensor[GGML_MAX_SRC + 1] = { dst };
    auto          end_iter                     = std::copy_if(dst->src, dst->src + GGML_MAX_SRC, cur_tensor + 1,
                                                              [](ggml_tensor * ptr) { return ptr != nullptr; });

    auto rpp_flash_attn_node = static_cast<rpp_kernel_flash_attn_ext *>(rpp_node);
    for (int i = 0; i < end_iter - cur_tensor; i++) {
        ggml_tensor *              node = cur_tensor[i];
        ggml_graph_node_properties graph_node_properties;
        graph_node_properties.node_address = node->data;
        graph_node_properties.node_op      = node->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            graph_node_properties.ne[i] = node->ne[i];
            // graph_node_properties.nb[i] = node->nb[i];
        }
        if (std::strstr(node->name, "cache_k") != nullptr || std::strstr(node->name, "cache_v") != nullptr) {
            graph_node_properties.ne[1] = rpp_flash_attn_node->kv_length;
        } else if (std::strstr(node->name, "copy") != nullptr) {
            graph_node_properties.ne[0] = rpp_flash_attn_node->kv_length;
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            graph_node_properties.src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
        }
        memcpy(graph_node_properties.op_params, node->op_params, GGML_MAX_OP_PARAMS);
        rpp_node->ggml_node_properties[cur_tensor[i]] = graph_node_properties;
    }
}

static bool ggml_rpp_flash_attention_create_engine_dispatch(ggml_backend_rpp_context & ctx,
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

    // create engine
    if (!ret) {
        ret = ggml_rpp_create_engine_flash_attention(ctx, rpp_node, dst);
    }
    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_flash_attn_ext_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[2]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[3]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

enum ggml_rpp_flash_attn_ext_stage {
    GGML_RPP_FLASH_ATTN_EXT_STAGE_PREFILL,
    GGML_RPP_FLASH_ATTN_EXT_STAGE_DECODE,
};

static ggml_rpp_flash_attn_ext_stage ggml_rpp_flash_attn_ext_get_stage(ggml_rpp_node * rpp_node) {
    // In RPP kernels, n_ubatch > 1 maps to prefill, n_ubatch == 1 maps to decode.
    return rpp_node->n_ubatch > 1 ? GGML_RPP_FLASH_ATTN_EXT_STAGE_PREFILL
                                  : GGML_RPP_FLASH_ATTN_EXT_STAGE_DECODE;
}

static rpp_kernel_flash_attn_ext * ggml_rpp_find_shared_kpara_node(ggml_backend_rpp_context &      ctx,
                                                                    uint32_t                         kv_length,
                                                                    ggml_rpp_flash_attn_ext_stage    stage) {
    for (auto & graph_iter : ctx.gglm_graphs) {
        ggml_rpp_cgraph * rpp_graph_tmp = ctx.rpp_graphs[graph_iter].get();
        if (!rpp_graph_tmp) {
            continue;
        }
        for (auto & node_iter : rpp_graph_tmp->rpp_nodes) {
            if (node_iter.first->op == GGML_OP_FLASH_ATTN_EXT) {
                auto & node_vec = node_iter.second;
                for (size_t i = 0; i < node_vec.size(); i++) {
                    auto cur_node = static_cast<rpp_kernel_flash_attn_ext *>(node_vec[i].get());
                    if (cur_node->rpp_type != ggml_rpp_node::RPP_NODE_TYPE_KERNEL) {
                        continue;
                    }
                    if (cur_node->kv_length != kv_length || cur_node->dev_shared_kparas == 0 ||
                        cur_node->shared_kpara_size == 0) {
                        continue;
                    }
                    const auto node_stage = ggml_rpp_flash_attn_ext_get_stage(cur_node);
                    if (node_stage == stage) {
                        return cur_node;
                    }
                }
            }
        }
    }
    return nullptr;
}

static bool ggml_rpp_flash_attn_ext_instantiate_child_exec(ggml_backend_rpp_context & ctx,
                                                            uint32_t kv_length,
                                                            ggml_rpp_node *             rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    auto                 rpp_node = static_cast<rpp_kernel_flash_attn_ext *>(rpp_base_node);
    rpp_kernel_context * kctx     = rpp_node->kernel_ctx.get();
    GGML_ASSERT(kctx && kctx->graph);

    if (kctx->graphexec != nullptr) {
        return false;
    }

    if (kctx->graphexec != nullptr) {
        RPP_CHECK(rppGraphExecDestroy(kctx->graphexec));
        kctx->graphexec = nullptr;
        return false;
    }

    const auto stage      = ggml_rpp_flash_attn_ext_get_stage(rpp_node);
    auto *     reuse_node = ggml_rpp_find_shared_kpara_node(ctx, kv_length, stage);
    RPP_GRAPH_INSTANTIATE_PARAMS instChild = {};
    instChild.flags                        = RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC;
    if (reuse_node != nullptr) {
        instChild.res_kpara.is_external = true;
        instChild.res_kpara.type        = RPP_GRAPH_RESOURCE_KPARA;
        instChild.res_kpara.daddr       = reuse_node->dev_shared_kparas;
        instChild.res_kpara.size        = reuse_node->shared_kpara_size;        
    }
    RPP_CHECK(rppGraphInstantiateWithParams(&kctx->graphexec, kctx->graph, &instChild));

    GGML_ASSERT(kctx->graphexec);
    if (reuse_node == nullptr) {
        RPP_GRAPH_INSTANTIATE_PARAMS probe_params = {};
        RPP_CHECK(rppGraphExecGetParams(kctx->graphexec, &probe_params));
        if (probe_params.res_kpara.size == 0) {
            GGML_LOG_ERROR("%s: invalid kpara size 0 for kv_length=%llu, %s (%s)\n", __func__,
                           (unsigned long long) kv_length, rpp_node->cur_ggml_tensor->name,
                           ggml_op_name(rpp_node->cur_ggml_tensor->op));
            return false;
        }

        RPPdeviceptr shared_kpara = 0;
        RPP_CHECK(rppGraphResourceAlloc(&shared_kpara, probe_params.res_kpara.size, RPP_GRAPH_RESOURCE_KPARA));
        rpp_node->dev_shared_kparas    = shared_kpara;
        rpp_node->shared_kpara_size = probe_params.res_kpara.size;        
    }

    return true;
}

static bool ggml_rpp_set_io_datas_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_flash_attn_ext *>(rpp_base_node);
    for (auto iter : rpp_node->binding_i_buffers) {
        auto i_tensor = iter.first;
        if (!ggml_is_contiguous(i_tensor)) {
            ggml_rpp_pack_tensor_to_contiguous(ctx, i_tensor, iter.second, iter.first->view_offs);
        }
    }
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

static bool ggml_rpp_flash_attn_ext_properties_is_same(ggml_backend_rpp_context & ctx,
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
            // if (node->nb[i] != graph_node_properties.nb[i]) {
            //     return false;
            // }
        }
    } else {
        if (dst->ne[rpp_node->seq_len_index] == 1) {
            return false;
        }
    }
    if (!ggml_rpp_flash_attention_kv_dims_is_same(dst, dst, rpp_node)) {
        return false;
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

bool ggml_rpp_op_kernel_flash_attn_ext(ggml_backend_rpp_context & ctx,
                                       ggml_tensor *              dst,
                                       int                        is_instantial,
                                       int                        is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_flash_attn_ext * rpp_node = nullptr;
    auto                        iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_flash_attn_ext");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_flash_attn_ext_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_flash_attn_ext *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            const int base_kv_len = 256;
            size_t    max_kv_step = std::min(ctx.n_max_ctx / base_kv_len, ctx.stub_kv_step);
            if (dst->src[1]->ne[1] < max_kv_step * base_kv_len) {
                for (size_t kv_step = 0; kv_step < max_kv_step; kv_step++) {
                    auto new_node = std::make_unique<rpp_kernel_flash_attn_ext>(dst);
                    ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                    auto rpp_node_tmp = (rpp_kernel_flash_attn_ext *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                    rpp_node_tmp->is_instantial = is_instantial;
                    rpp_node_tmp->kv_length     = (kv_step + 1) * base_kv_len;
                    const uint64_t kv_length_key = static_cast<uint64_t>(rpp_node_tmp->kv_length);
                    if (!(ggml_rpp_flash_attention_create_engine_dispatch(ctx, rpp_node_tmp, dst))) {
                        return false;
                    }
                    if (!is_instantial) {
                        if (!ggml_rpp_flash_attn_ext_instantiate_child_exec(ctx, rpp_node_tmp->kv_length, rpp_node_tmp)) {
                            return false;
                        }
                    }
                }
                auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
                for (size_t i = 0; i < node_vec.size(); i++) {
                    auto cur_node = node_vec[i].get();
                    if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                        ggml_rpp_flash_attn_ext_properties_is_same(ctx, dst, cur_node)) {
                        rpp_node = (rpp_kernel_flash_attn_ext *) cur_node;
                        break;
                    }
                }
            } else {
                auto new_node = std::make_unique<rpp_kernel_flash_attn_ext>(dst);
                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                rpp_node = (rpp_kernel_flash_attn_ext *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                rpp_node->is_instantial = is_instantial;
                rpp_node->kv_length     = dst->src[1]->ne[1];
                const uint64_t kv_length_key = static_cast<uint64_t>(rpp_node->kv_length);
                if (!(ggml_rpp_flash_attention_create_engine_dispatch(ctx, rpp_node, dst))) {
                    return false;
                }
                if (!is_instantial) {
                    ggml_rpp_flash_attn_ext_instantiate_child_exec(ctx, rpp_node->kv_length, rpp_node);
                }
            }
        }

        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_flash_attn_ext *) (iter->second);
    }

    if (is_launch) {
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_flash_attn_ext");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }

    return true;
}
