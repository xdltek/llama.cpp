#include "rpp_get_rows/rpp_get_rows.h"
#include "rpp_get_rows/src/rpp_kernel_build.h"

static int ggml_rpp_get_rows_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_create_kernel_get_rows(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_get_rows *>(rpp_base_node);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    const int cols       = dst->ne[0];
    void *    i_buffer_0 = nullptr;
    void *    o_buffer   = nullptr;
    if (ctx.use_bf16) {
        if (dst->src[0]->view_offs == 0) {
            i_buffer_0 = dst->src[0]->data;
        } else {
            i_buffer_0 = (char *) dst->src[0]->data - dst->src[0]->view_offs / 2;
        }
        if (dst->view_offs == 0) {
            o_buffer = dst->data;
        } else {
            o_buffer = (char *) dst->data - dst->view_offs / 2;
        }
    } else {
        i_buffer_0 = dst->src[0]->data;
        o_buffer   = dst->data;
    }
    // kernel inputs
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (i_buffer_0));
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) (dst->src[1]->data));
    // kernel outputs
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (o_buffer));
    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[0], i_buffer_0);
    rpp_node->binding_i_buffers.emplace(dst->src[1], dst->src[1]->data);
    rpp_node->binding_o_buffers.emplace(dst, o_buffer);
    rpp_node->binding_io_buffers.emplace_back(i_buffer_0);
    rpp_node->binding_io_buffers.emplace_back(dst->src[1]->data);
    rpp_node->binding_io_buffers.emplace_back(o_buffer);

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    // build set_rows kernel
    rpp_get_rows_build(*(rpp_node->kernel_ctx.get()), cols, i_type_size, o_type_size, ggml_type_size(dst->src[1]->type),
                       rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_get_rows *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst->src[0]);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_get_rows_seq_len(dst->src[0], rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    ret = ggml_rpp_create_kernel_get_rows(ctx, rpp_node, dst);
    GGML_ASSERT(ret);

    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst->src[1]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

static bool ggml_rpp_get_rows_properties_is_same(ggml_backend_rpp_context & ctx,
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
    // because of,the sqe_len is control by rpp_update_rowids(warmup),
    // so no need to use u_batch, only build on kernel for dst, only check the data ptr
    // if (rpp_node->n_ubatch == 1) {
    //     for (int i = 0; i < GGML_MAX_DIMS; i++) {
    //         if (node->ne[i] != graph_node_properties.ne[i]) {
    //             return false;
    //         }
    //         if (node->nb[i] != graph_node_properties.nb[i]) {
    //             return false;
    //         }
    //     }
    // } else {
    //     if (dst->src[0]->ne[rpp_node->seq_len_index] == 1) {
    //         return false;
    //     }
    // }
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

static bool ggml_rpp_set_io_datas_device(ggml_backend_rpp_context & ctx, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_kernel_get_rows_datas");
    auto      rpp_node                = static_cast<rpp_kernel_get_rows *>(rpp_base_node);
    const int start_of_rows           = 0;
    const int contious_number_of_rows = rpp_node->cur_ggml_tensor->src[1]->ne[0];
    rpp_update_rowids(start_of_rows, contious_number_of_rows);
    return true;
}

bool ggml_rpp_op_kernel_get_rows(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_get_rows * rpp_node = nullptr;
    auto                  iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_get_rows");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                // because of,the sqe_len is control by rpp_update_rowids(warmup),
                // so no need to use u_batch, only build on kernel for dst
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_get_rows_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_get_rows *) cur_node;
                    break;
                }
            }
        }
        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_get_rows");
            auto new_node = std::make_unique<rpp_kernel_get_rows>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_get_rows *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
        ctx.cur_rpp_graph->add_launch_func(ggml_rpp_set_io_datas_device, rpp_node);
    } else {
        rpp_node = (rpp_kernel_get_rows *) (iter->second);
    }
    if (is_launch) {
        {
            TRACE_SCOPE_GUARD(ctx.trace_id, "set_kernel_get_rows_datas");
            ggml_rpp_set_io_datas_device(ctx, rpp_node);
        }
        // compute add operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_get_rows");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return true;
}
