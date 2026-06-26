#include "rpp_cont/rpp_cont.h"
#include "rpp_cont/src/rpp_kernel_build.h"
#include "rpp_cont/src/rpp_kernel_memcpy_align.h"

static int ggml_rpp_cont_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_cont_layout_matches(const ggml_tensor * node, const ggml_graph_node_properties & props) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (node->ne[i] != props.ne[i]) {
            return false;
        }
        if (node->nb[i] != props.nb[i]) {
            return false;
        }
    }
    return true;
}

static bool ggml_rpp_cont_get_memcpy_align_desc(const ggml_tensor * dst,
                                                uint32_t &          D0,
                                                uint32_t &          D1,
                                                uint32_t &          D2,
                                                uint32_t &          stride_x,
                                                uint32_t &          stride_y,
                                                uint32_t &          stride_z,
                                                size_t &            input_span_bytes,
                                                int &               i_type_size_0) {
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->src[0]);

    const ggml_tensor * src = dst->src[0];

    if (!ggml_is_contiguous(dst)) {
        return false;
    }

    i_type_size_0 = (int) ggml_type_size(src->type);
    if (i_type_size_0 <= 0 || src->type != dst->type) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] < 1 || dst->ne[i] < 1) {
            return false;
        }
        if (src->ne[i] != dst->ne[i]) {
            return false;
        }
    }
    if (src->nb[0] % (size_t) i_type_size_0 != 0 || src->nb[1] % (size_t) i_type_size_0 != 0 ||
        src->nb[2] % (size_t) i_type_size_0 != 0 || src->nb[3] % (size_t) i_type_size_0 != 0) {
        return false;
    }
    if (src->ne[3] > 1 && src->nb[3] != src->nb[2] * (size_t) src->ne[2]) {
        return false;
    }

    // memcpy_align is 3D, so collapse ggml dims 2 and 3 into its z dimension.
    const uint64_t flat_D0 = (uint64_t) src->ne[2] * (uint64_t) src->ne[3];
    const uint64_t flat_D1 = (uint64_t) src->ne[1];
    const uint64_t flat_D2 = (uint64_t) src->ne[0];
    const uint64_t sx      = (uint64_t) src->nb[0] / (uint64_t) i_type_size_0;
    const uint64_t sy      = (uint64_t) src->nb[1] / (uint64_t) i_type_size_0;
    const uint64_t sz      = (uint64_t) src->nb[2] / (uint64_t) i_type_size_0;

    if (flat_D0 > UINT32_MAX || flat_D1 > UINT32_MAX || flat_D2 > UINT32_MAX || sx > UINT32_MAX || sy > UINT32_MAX ||
        sz > UINT32_MAX) {
        return false;
    }

    const uint64_t max_offset = (flat_D2 - 1u) * sx + (flat_D1 - 1u) * sy + (flat_D0 - 1u) * sz;
    const uint64_t span_bytes = (max_offset + 1u) * (uint64_t) i_type_size_0;

    if (span_bytes > std::numeric_limits<size_t>::max()) {
        return false;
    }

    D0               = (uint32_t) flat_D0;
    D1               = (uint32_t) flat_D1;
    D2               = (uint32_t) flat_D2;
    stride_x         = (uint32_t) sx;
    stride_y         = (uint32_t) sy;
    stride_z         = (uint32_t) sz;
    input_span_bytes = (size_t) span_bytes;
    return true;
}

bool ggml_rpp_cont_supports_op(const ggml_tensor * dst) {
    if (dst == nullptr || dst->src[0] == nullptr) {
        return false;
    }

    const ggml_tensor * src = dst->src[0];
    if (src->type != dst->type) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] != dst->ne[i]) {
            return false;
        }
    }

    const int type_size = (int) ggml_type_size(src->type);
    if (type_size <= 0) {
        return false;
    }
    if (src->nb[0] % (size_t) type_size != 0 || src->nb[1] % (size_t) type_size != 0 ||
        src->nb[2] % (size_t) type_size != 0) {
        return false;
    }

    if (src->ne[3] == 1) {
        const int use_dyn_d0 = 1;
        const int D0         = src->ne[2];
        const int D1         = src->ne[1];
        const int D2         = src->ne[0];
        const int stride_x   = src->nb[0] / type_size;
        const int stride_y   = src->nb[1] / type_size;
        const int stride_z   = src->nb[2] / type_size;
        if (rpp_cont_supports_dma(use_dyn_d0, D0, D1, D2, stride_x, stride_y, stride_z)) {
            return true;
        }
    }

    uint32_t D0                = 0;
    uint32_t D1                = 0;
    uint32_t D2                = 0;
    uint32_t stride_x          = 0;
    uint32_t stride_y          = 0;
    uint32_t stride_z          = 0;
    size_t   input_span_bytes  = 0;
    int      bytes_per_element = 0;

    return ggml_rpp_cont_get_memcpy_align_desc(dst, D0, D1, D2, stride_x, stride_y, stride_z, input_span_bytes,
                                               bytes_per_element);
}

static bool ggml_rpp_create_kernel_cont(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);

    auto      rpp_node      = static_cast<rpp_kernel_cont *>(rpp_base_node);
    // prefill and decode supports dynamic D0
    const int use_dyn_d0    = 1;
    const int D0            = dst->src[0]->ne[2];
    const int D1            = dst->src[0]->ne[1];
    const int D2            = dst->src[0]->ne[0];
    const int stride_x      = dst->src[0]->nb[0] / ggml_type_size(dst->src[0]->type);
    const int stride_y      = dst->src[0]->nb[1] / ggml_type_size(dst->src[0]->type);
    const int stride_z      = dst->src[0]->nb[2] / ggml_type_size(dst->src[0]->type);
    const int i_type_size_0 = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);

    const bool   use_internal_bf16 = ctx.use_bf16 && dst->src[0]->type == GGML_TYPE_F32;
    RPPdeviceptr inputs_0_buf      = ctx.use_bf16 ?
                                         (RPPdeviceptr) dst->src[0]->data -
                                        (use_internal_bf16 ? dst->src[0]->view_offs / 2 : dst->src[0]->view_offs) :
                                         (RPPdeviceptr) dst->src[0]->data;

    // RPPdeviceptr inputs_0_buf = (RPPdeviceptr) dst->src[0]->data;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (dst->data));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[0], dst->src[0]->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(dst->src[0]->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    // build kernel
    if (dst->src[0]->ne[3] == 1 && rpp_cont_supports_dma(use_dyn_d0, D0, D1, D2, stride_x, stride_y, stride_z)) {
        rpp_node->exec_mode = RPP_CONT_EXEC_DMA;
        rpp_cont_build(*(rpp_node->kernel_ctx.get()), use_dyn_d0, D0, D1, D2, stride_x, stride_y, stride_z,
                       i_type_size_0, rpp_node->is_instantial);
        return true;
    }

    uint32_t memcpy_D0        = 0;
    uint32_t memcpy_D1        = 0;
    uint32_t memcpy_D2        = 0;
    uint32_t memcpy_stride_x  = 0;
    uint32_t memcpy_stride_y  = 0;
    uint32_t memcpy_stride_z  = 0;
    size_t   input_span_bytes = 0;
    int      memcpy_type_size = 0;

    if (!ggml_rpp_cont_get_memcpy_align_desc(dst, memcpy_D0, memcpy_D1, memcpy_D2, memcpy_stride_x, memcpy_stride_y,
                                             memcpy_stride_z, input_span_bytes, memcpy_type_size)) {
        GGML_LOG_ERROR(
            "%s: unsupported CONT layout for RPP backend: src ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
            __func__, (long long) dst->src[0]->ne[0], (long long) dst->src[0]->ne[1], (long long) dst->src[0]->ne[2],
            (long long) dst->src[0]->ne[3], dst->src[0]->nb[0], dst->src[0]->nb[1], dst->src[0]->nb[2],
            dst->src[0]->nb[3]);
        return false;
    }

    // Strides are derived from logical ggml layout, but actual buffer element size for
    // internal RPP flow must follow io type size (e.g. F32 tensors carried as BF16).
    memcpy_type_size    = i_type_size_0;
    rpp_node->exec_mode = RPP_CONT_EXEC_MEMCPY_ALIGN;
    rpp_cont_memcpy_align_build(*(rpp_node->kernel_ctx.get()), memcpy_D0, memcpy_D1, memcpy_D2, memcpy_stride_x,
                                memcpy_stride_y, memcpy_stride_z, input_span_bytes, (uint32_t) memcpy_type_size,
                                rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_cont *>(rpp_base_node);
    bool ret      = false;
    // first prefill stage can get sqe len
    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst->src[0]);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }
    // set ubacth for rpp_node
    if (ctx.use_ubatch && ggml_rpp_cont_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    ret = ggml_rpp_create_kernel_cont(ctx, rpp_node, dst);
    if (!ret) {
        return false;
    }

    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

static bool ggml_rpp_cont_properties_is_same(ggml_backend_rpp_context & ctx,
                                             ggml_tensor *              dst,
                                             ggml_rpp_node *            rpp_node) {
    GGML_ASSERT(rpp_node);
    GGML_UNUSED(ctx);
    auto * cont_node = static_cast<rpp_kernel_cont *>(rpp_node);
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
    if (cont_node->exec_mode == RPP_CONT_EXEC_MEMCPY_ALIGN &&
        !ggml_rpp_cont_layout_matches(node, graph_node_properties)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] && node->src[i]->data != graph_node_properties.src_address[i] && node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW) {
            return false;
        }
    }
    if (cont_node->exec_mode == RPP_CONT_EXEC_MEMCPY_ALIGN && node->src[0] != nullptr) {
        auto iter_src = rpp_node->ggml_node_properties.find(node->src[0]);
        if (iter_src == rpp_node->ggml_node_properties.end()) {
            return false;
        }
        if (!ggml_rpp_cont_layout_matches(node->src[0], iter_src->second)) {
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
    TRACE_SCOPE_GUARD(ctx.trace_id, "set_kernel_cont_datas");
    auto rpp_node = static_cast<rpp_kernel_cont *>(rpp_base_node);
    if (rpp_node->exec_mode != RPP_CONT_EXEC_DMA) {
        return true;
    }
    const int seq_len = ggml_rpp_cont_seq_len(rpp_node->cur_ggml_tensor->src[0], rpp_node);
    rpp_update_dyn_d0(seq_len);
    return true;
}

bool ggml_rpp_op_kernel_cont(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    rpp_kernel_cont * rpp_node = nullptr;
    auto              iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_cont");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_cont_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_cont *) cur_node;
                    break;
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_cont");
            auto new_node = std::make_unique<rpp_kernel_cont>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_cont *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
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
        rpp_node = (rpp_kernel_cont *) (iter->second);
    }
    if (is_launch) {
        {
            TRACE_SCOPE_GUARD(ctx.trace_id, "set_kernel_cont_datas");
            ggml_rpp_set_io_datas_device(ctx, rpp_node);
        }
        // compute add operator
        try {
            TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_cont");
            RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                           e.what());
        }
    }
    return true;
}
