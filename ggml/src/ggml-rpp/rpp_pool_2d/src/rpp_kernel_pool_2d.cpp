#include "rpp_pool_2d/rpp_pool_2d.h"
#include "rpp_pool_2d/src/rpp_kernel_build.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <vector>

static int ggml_rpp_pool_2d_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

bool ggml_rpp_pool_2d_supported(const ggml_tensor * dst) {
    const char * dump_cpu = std::getenv("GGML_DUMP_CPU_POOL_2D");
    if (dump_cpu != nullptr && dump_cpu[0] != '\0') {
        return false;
    }
    const char * disable = std::getenv("GGML_RPP_DISABLE_POOL_2D");
    if (disable != nullptr && disable[0] != '\0' && std::strcmp(disable, "0") != 0) {
        return false;
    }
    if (dst == nullptr || dst->src[0] == nullptr) {
        return false;
    }
    const ggml_tensor * src = dst->src[0];
    if (dst->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(dst) || !ggml_is_contiguous(src)) {
        return false;
    }

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int       op   = opts[0];
    const int       k0   = opts[1];
    const int       k1   = opts[2];
    const int       s0   = opts[3];
    const int       s1   = opts[4];
    const int       p0   = opts[5];
    const int       p1   = opts[6];

    const int64_t channels = src->ne[2] * src->ne[3];
    return op == GGML_OP_POOL_AVG && k0 == 2 && k1 == 2 && s0 == 2 && s1 == 2 && p0 == 0 && p1 == 0 &&
           src->ne[0] == 32 && src->ne[1] == 32 && channels == 1152 && dst->ne[0] == 16 && dst->ne[1] == 16 &&
           dst->ne[2] == src->ne[2] && dst->ne[3] == src->ne[3];
}

static bool ggml_rpp_pool_2d_properties_is_same(ggml_backend_rpp_context & ctx,
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

    const size_t seq_len           = static_cast<size_t>(ggml_rpp_pool_2d_seq_len(dst, rpp_node));
    const size_t expected_n_ubatch = ctx.use_ubatch && seq_len > 1 ? ctx.n_ubatch : 1;
    if (rpp_node->n_ubatch != expected_n_ubatch) {
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
    } else if (dst->ne[rpp_node->seq_len_index] == 1) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] && node->src[i]->data != graph_node_properties.src_address[i] && node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW) {
            return false;
        }
    }
    if (std::memcmp(graph_node_properties.op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }
    return true;
}

static bool ggml_rpp_create_kernel_pool_2d(ggml_backend_rpp_context & ctx,
                                           ggml_rpp_node *            rpp_base_node,
                                           ggml_tensor *              dst) {
    GGML_UNUSED(ctx);
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_pool_2d *>(rpp_base_node);

    if (!ggml_rpp_pool_2d_supported(dst)) {
        return false;
    }

    ggml_tensor * src = dst->src[0];

    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) src->data);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) dst->data);

    rpp_node->binding_i_buffers.emplace(src, src->data);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(src->data);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int       k0   = opts[1];
    const int       k1   = opts[2];
    const int       s0   = opts[3];
    const int       s1   = opts[4];

    const int in_w     = (int) src->ne[0];
    const int in_h     = (int) src->ne[1];
    const int out_w    = (int) dst->ne[0];
    const int out_h    = (int) dst->ne[1];
    const int channels = (int) (src->ne[2] * src->ne[3]);

    rpp_pool_2d_build(*(rpp_node->kernel_ctx.get()), in_w, in_h, channels, out_w, out_h, k0, k1, s0, s1,
                      (int) ggml_type_size(src->type), (int) ggml_type_size(dst->type), rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_pool_2d_dispatch(ggml_backend_rpp_context & ctx,
                                                    ggml_rpp_node *            rpp_base_node,
                                                    ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_pool_2d *>(rpp_base_node);

    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        rpp_node->seq_len_index = ggml_n_dims(dst) == 1 ? 1 : ggml_n_dims(dst) - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }

    const size_t seq_len = static_cast<size_t>(ggml_rpp_pool_2d_seq_len(dst, rpp_node));
    if (ctx.use_ubatch && seq_len > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    const bool ret = ggml_rpp_create_kernel_pool_2d(ctx, rpp_node, dst);
    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

bool ggml_rpp_op_kernel_pool_2d(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr\n", __func__);
        return false;
    }
    if (!ggml_rpp_pool_2d_supported(dst)) {
        GGML_LOG_ERROR("%s: unsupported POOL_2D shape/type/params for %s\n", __func__, dst->name);
        return false;
    }

    rpp_kernel_pool_2d * rpp_node = nullptr;
    auto iter = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_pool_2d");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_pool_2d_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_pool_2d *) cur_node;
                    break;
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_pool_2d");
            auto new_node = std::make_unique<rpp_kernel_pool_2d>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_pool_2d *) ctx.cur_rpp_graph->rpp_nodes[dst].back().get();
            rpp_node->is_instantial = is_instantial;
            if (!ggml_rpp_create_kernel_pool_2d_dispatch(ctx, rpp_node, dst)) {
                return false;
            }
        }

        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_pool_2d *) iter->second;
    }

    if (!is_launch) {
        return true;
    }

    ggml_tensor * src = dst->src[0];

    const int32_t * opts = (const int32_t *) dst->op_params;
    // std::fprintf(stderr,
    //              "%s: POOL_2D running on RPP device kernel (op=%d, src_ne=[%lld, %lld, %lld, %lld], "
    //              "dst_ne=[%lld, %lld, %lld, %lld], k=[%d, %d], s=[%d, %d], p=[%d, %d])\n",
    //              __func__, opts[0], (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2],
    //              (long long) src->ne[3], (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2],
    //              (long long) dst->ne[3], opts[1], opts[2], opts[3], opts[4], opts[5], opts[6]);

    try {
        TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_pool_2d");
        RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op),
                       e.what());
        return false;
    }

    const char * dump_path = std::getenv("GGML_DUMP_RPP_POOL_2D");
    if (dump_path != nullptr && dump_path[0] != '\0') {
        std::vector<float> dst_host((size_t) ggml_nelements(dst));
        const size_t       data_size = dst_host.size() * sizeof(float);
        RPP_CHECK(rtStreamSynchronize(ctx.stream()));
        RPP_CHECK(rtMemcpy(dst_host.data(), dst->data, data_size, rtMemcpyDeviceToHost));
        std::ofstream ofs_tensor_data(dump_path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!ofs_tensor_data) {
            std::fprintf(stderr, "%s: failed to open GGML_DUMP_RPP_POOL_2D path: %s\n", __func__, dump_path);
        } else {
            ofs_tensor_data.write(reinterpret_cast<const char *>(dst_host.data()), (std::streamsize) data_size);
            if (!ofs_tensor_data.good()) {
                std::fprintf(stderr, "%s: failed to write RPP POOL_2D dump to %s (bytes=%zu)\n", __func__, dump_path,
                             data_size);
            } else {
                std::fprintf(stderr, "%s: RPP POOL_2D dump saved to %s (ne=[%lld, %lld, %lld, %lld], bytes=%zu)\n",
                             __func__, dump_path, (long long) dst->ne[0], (long long) dst->ne[1],
                             (long long) dst->ne[2], (long long) dst->ne[3], data_size);
            }
            ofs_tensor_data.close();
        }
    }
    return true;
}
