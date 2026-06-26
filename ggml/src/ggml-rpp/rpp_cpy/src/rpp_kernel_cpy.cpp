#include "ggml-cpu/ops.h"
#include "rpp_cpy/rpp_cpy.h"
#include "rpp_cpy/src/rpp_kernel_build.h"

static int ggml_rpp_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_cpy_properties_is_same(ggml_backend_rpp_context & ctx,
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

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};

static bool ggml_compute_forward_cpy(ggml_tensor * dst) {
    GGML_ASSERT(dst);
    struct ggml_compute_params params = {
        /*.ith       =*/0,
        /*.nth       =*/1,
        /*.wsize     =*/0,
        /*.wdata     =*/nullptr,
        /*.threadpool=*/nullptr,
    };
    ggml_compute_forward_dup(&params, dst);
    return true;
}

bool ggml_rpp_op_cpu_cpy(ggml_backend_rpp_context & ctx, ggml_tensor * dst) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    TRACE_SCOPE_GUARD(ctx.trace_id, "execute_cpu_cpy");

    if (dst->type == dst->src[0]->type) {
        RPP_MEMCPY_DEV_AND_HOST(dst->data, dst->src[0]->data, ggml_nbytes(dst->src[0]), rtMemcpyDeviceToDevice,
                                ctx.stream(), 0);
    } else {
        size_t src_size = ggml_nbytes(dst->src[0]);
        size_t dst_size = ggml_nbytes(dst);
        void * src_buf  = ctx.pool_host().alloc(src_size);
        void * dst_buf  = ctx.pool_host().alloc(dst_size);
        RPP_MEMCPY_DEV_AND_HOST(src_buf, dst->src[0]->data, ggml_nbytes(dst->src[0]), rtMemcpyDeviceToHost,
                                ctx.stream(), 1);
        auto ori_src_buf  = dst->src[0]->data;
        dst->src[0]->data = src_buf;
        auto ori_dst_buf  = dst->data;
        dst->data         = dst_buf;
        ggml_compute_forward_cpy(dst);
        dst->src[0]->data = ori_src_buf;
        dst->data         = ori_dst_buf;
        RPP_MEMCPY_DEV_AND_HOST(dst->data, dst_buf, ggml_nbytes(dst), rtMemcpyHostToDevice, ctx.stream(), 0);

        ctx.pool_host().free(src_buf);
        ctx.pool_host().free(dst_buf);
    }
    return true;
}

static bool ggml_rpp_create_kernel_cpy(ggml_backend_rpp_context & ctx,
                                       ggml_rpp_node *            rpp_base_node,
                                       ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);

    auto   rpp_node = static_cast<rpp_kernel_cpy *>(rpp_base_node);
    size_t n        = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        n *= dst->src[0]->ne[i];
    }

    void * i_buffer_0 = nullptr;
    void * o_buffer   = nullptr;
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

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) i_buffer_0);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) (o_buffer));

    // set io buffer info to rpp_node
    rpp_node->binding_i_buffers.emplace(dst->src[0], i_buffer_0);
    rpp_node->binding_o_buffers.emplace(dst, o_buffer);
    rpp_node->binding_io_buffers.emplace_back(i_buffer_0);
    rpp_node->binding_io_buffers.emplace_back(o_buffer);
    const int   i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int   o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    RppDataType inType      = (i_type_size == sizeof(float)) ? kFLOAT : kBF16;
    RppDataType outType     = (o_type_size == sizeof(uint16_t)) ? kBF16 : kFLOAT;
    // build kernel
    rpp_copy_build(*(rpp_node->kernel_ctx.get()), n, inType, outType, rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto       rpp_node           = static_cast<rpp_kernel_cpy *>(rpp_base_node);
    bool       ret                = false;
    const int  i_type_size        = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int  o_type_size        = ggml_rpp_get_io_type_size(ctx, dst, 1);
    const bool io_sizes_supported = (i_type_size == (int) sizeof(float) || i_type_size == (int) sizeof(uint16_t)) &&
                                    (o_type_size == (int) sizeof(float) || o_type_size == (int) sizeof(uint16_t));
    const bool is_convert_path = (i_type_size != o_type_size);
    const bool is_float_family =
        (dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_BF16) &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_BF16 || dst->type == GGML_TYPE_F16);
    const bool can_use_kernel = io_sizes_supported && is_convert_path && is_float_family &&
                                ggml_is_contiguous(dst->src[0]) && ggml_is_contiguous(dst);
    if (!can_use_kernel) {
        rpp_node->use_cpu_fallback = true;
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
        return true;
    }
    rpp_node->use_cpu_fallback = false;
    // first prefill stage can get sqe len
    // if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
    //     int n                   = ggml_n_dims(dst);
    //     rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    // } else {
    //     rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    // }
    // // set ubacth for rpp_node
    // if (ctx.use_ubatch && ggml_rpp_seq_len(dst, rpp_node) > 1) {
    //     rpp_node->n_ubatch = ctx.n_ubatch;
    // }

    ret = ggml_rpp_create_kernel_cpy(ctx, rpp_node, dst);
    GGML_ASSERT(ret);

    // get io tensor and set properties
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

bool ggml_rpp_op_kernel_cpy(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    rpp_kernel_cpy * rpp_node = nullptr;
    auto             iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_cpy");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_cpy_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_cpy *) cur_node;
                    break;
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_cpy");
            auto new_node = std::make_unique<rpp_kernel_cpy>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_cpy *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
            if (!is_instantial) {
                rpp_kernel_context * kctx = rpp_node->kernel_ctx.get();
                GGML_ASSERT(kctx && kctx->graph);
                if (kctx->graphexec != nullptr) {
                    RPP_CHECK(rppGraphExecDestroy(kctx->graphexec));
                    kctx->graphexec = nullptr;
                }
                RPP_GRAPH_INSTANTIATE_PARAMS instChild = {};
                instChild.flags                        = RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC;
                RPP_CHECK(rppGraphInstantiateWithParams(&kctx->graphexec, kctx->graph, &instChild));
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_cpy *) (iter->second);
    }

    bool ret = true;
    if (is_launch) {
        if (rpp_node->use_cpu_fallback) {
            ret = ggml_rpp_op_cpu_cpy(ctx, dst);
        } else {
            try {
                TRACE_SCOPE_GUARD(ctx.trace_id, "execute_cpu_cpy");
                RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                               e.what());
            }
        }
    }
    return ret;
}
