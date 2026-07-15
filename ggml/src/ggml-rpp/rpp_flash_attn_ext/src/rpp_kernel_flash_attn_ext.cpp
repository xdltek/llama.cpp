#include "rpp_flash_attn_ext/rpp_flash_attn_ext.h"
#include "rpp_flash_attn_ext/src/rpp_kernel_build.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

static int ggml_rpp_flash_attn_ext_parse_layer_suffix(const char * name, const char * prefix) {
    if (name == nullptr || prefix == nullptr) {
        return -1;
    }
    const char * pos = std::strstr(name, prefix);
    if (pos == nullptr) {
        return -1;
    }
    pos += std::strlen(prefix);
    if (*pos < '0' || *pos > '9') {
        return -1;
    }

    char *     end   = nullptr;
    const long layer = std::strtol(pos, &end, 10);
    if (end == pos || layer < 0 || layer > std::numeric_limits<int>::max()) {
        return -1;
    }
    return (int) layer;
}

static bool ggml_rpp_flash_attn_ext_uses_cross_layer_kv_cache(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT || dst->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * k_base = dst->src[1]->view_src ? dst->src[1]->view_src : dst->src[1];
    const int          attn_layer =
        ggml_rpp_flash_attn_ext_parse_layer_suffix(ggml_get_name(dst), "__fattn__-");
    const int kv_layer = ggml_rpp_flash_attn_ext_parse_layer_suffix(ggml_get_name(k_base), "cache_k_l");

    return attn_layer >= 0 && kv_layer >= 0 && attn_layer != kv_layer;
}

static int ggml_rpp_flash_attention_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    GGML_ASSERT(node);
    return dst->ne[node->seq_len_index];
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
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_flash_attn_ext");
            if (ggml_rpp_flash_attn_ext_uses_cross_layer_kv_cache(dst)) {
                auto new_node = std::make_unique<rpp_kernel_flash_attn_ext>(dst);
                ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                rpp_node = (rpp_kernel_flash_attn_ext *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                rpp_node->is_instantial = is_instantial;
                rpp_node->kv_length     = dst->src[1]->ne[1];
                if (!(ggml_rpp_flash_attention_create_engine_dispatch(ctx, rpp_node, dst))) {
                    return false;
                }
            } else {
                const int base_kv_len = 256;
                size_t    max_kv_step = std::min(ctx.n_max_ctx / base_kv_len, ctx.stub_kv_step);
                const int64_t max_prebuild_kv_len = (int64_t) max_kv_step * base_kv_len;
                if (dst->src[1]->ne[1] < max_prebuild_kv_len) {
                    for (size_t kv_step = 0; kv_step < max_kv_step; kv_step++) {
                        auto new_node = std::make_unique<rpp_kernel_flash_attn_ext>(dst);
                        ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
                        auto rpp_node_tmp =
                            (rpp_kernel_flash_attn_ext *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
                        rpp_node_tmp->is_instantial = is_instantial;
                        rpp_node_tmp->kv_length     = (kv_step + 1) * base_kv_len;
                        if (!(ggml_rpp_flash_attention_create_engine_dispatch(ctx, rpp_node_tmp, dst))) {
                            return false;
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
                    if (!(ggml_rpp_flash_attention_create_engine_dispatch(ctx, rpp_node, dst))) {
                        return false;
                    }
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
