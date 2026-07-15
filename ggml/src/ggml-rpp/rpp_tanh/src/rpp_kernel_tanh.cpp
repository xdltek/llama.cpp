#include "rpp_tanh/kernel/rpp_kernel_build.h"
#include "rpp_tanh/rpp_tanh.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>

struct rpp_tanh_dump_info {
    int         id;
    std::string data_path;
};

struct rpp_tanh_tensor_stats {
    int64_t elements = 0;
    int64_t finite_count = 0;
    int64_t nan_count = 0;
    int64_t inf_count = 0;
    int64_t first_nan_index = -1;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
};

static std::mutex                                      g_rpp_tanh_dump_mutex;
static std::atomic<int>                                g_rpp_tanh_dump_next_id{0};
static std::unordered_map<const ggml_tensor *, rpp_tanh_dump_info> g_rpp_tanh_dump_infos;

static std::string ggml_rpp_tanh_dump_sanitize_name(const char * name) {
    std::string result = name && name[0] ? name : "unnamed";
    for (char & c : result) {
        if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') &&
            c != '_' && c != '-' && c != '.') {
            c = '_';
        }
    }
    return result;
}

static bool ggml_rpp_tanh_dump_get_info(const ggml_tensor * dst, rpp_tanh_dump_info & info) {
    const char * dump_dir = std::getenv("GGML_RPP_TANH_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return false;
    }

    const char * filter = std::getenv("GGML_RPP_TANH_DUMP_FILTER");
    if (filter && filter[0] && std::strstr(dst->name, filter) == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_rpp_tanh_dump_mutex);

    auto it = g_rpp_tanh_dump_infos.find(dst);
    if (it != g_rpp_tanh_dump_infos.end()) {
        info = it->second;
        return true;
    }

    mkdir(dump_dir, 0777);

    const int id = g_rpp_tanh_dump_next_id.fetch_add(1);
    const std::string name = ggml_rpp_tanh_dump_sanitize_name(dst->name);
    const std::string prefix = std::string(dump_dir) + "/rpp_tanh_" + std::to_string(id) + "_" + name;
    const std::string data_path = prefix + ".f32.bin";
    const std::string meta_path = prefix + ".meta";

    {
        FILE * fp = std::fopen(meta_path.c_str(), "w");
        if (fp != nullptr) {
            fprintf(fp, "id=%d\n", id);
            fprintf(fp, "name=%s\n", dst->name);
            fprintf(fp, "op=TANH\n");
            fprintf(fp, "stored_type=f32\n");
            fprintf(fp, "original_type=%s\n", ggml_type_name(dst->type));
            fprintf(fp, "ne=%lld,%lld,%lld,%lld\n",
                    (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3]);
            fprintf(fp, "nb=%zu,%zu,%zu,%zu\n", dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            fprintf(fp, "elements=%lld\n", (long long) ggml_nelements(dst));
            fprintf(fp, "data=%s\n", data_path.c_str());
            std::fclose(fp);
        }
    }

    auto inserted = g_rpp_tanh_dump_infos.emplace(dst, rpp_tanh_dump_info{id, data_path});
    fprintf(stderr, "GGML_RPP_TANH_DUMP: #%d %s -> %s\n", id, dst->name, data_path.c_str());
    info = inserted.first->second;
    return true;
}

static bool ggml_rpp_tanh_log_enabled(const ggml_tensor * dst) {
    const char * enabled = std::getenv("GGML_RPP_TANH_LOG");
    if (enabled == nullptr || enabled[0] == '\0' || std::strcmp(enabled, "0") == 0) {
        return false;
    }

    const char * filter = std::getenv("GGML_RPP_TANH_LOG_FILTER");
    if (filter && filter[0] && std::strstr(dst->name, filter) == nullptr) {
        return false;
    }

    return true;
}

static void ggml_rpp_tanh_stats_update(rpp_tanh_tensor_stats & stats, float value, int64_t index) {
    if (std::isnan(value)) {
        stats.nan_count++;
        if (stats.first_nan_index < 0) {
            stats.first_nan_index = index;
        }
        return;
    }
    if (std::isinf(value)) {
        stats.inf_count++;
        return;
    }

    stats.finite_count++;
    stats.min_value = std::min(stats.min_value, value);
    stats.max_value = std::max(stats.max_value, value);
}

static bool ggml_rpp_tanh_collect_stats(
        ggml_backend_rpp_context & ctx,
        const ggml_tensor * tensor,
        rpp_tanh_tensor_stats & stats) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_BF16) {
        fprintf(stderr, "%s: unsupported tanh stats type: %s\n", __func__, ggml_type_name(tensor->type));
        return false;
    }

    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    RPP_CHECK(rtMemcpyAsync(raw.data(), (const char *) tensor->data, raw.size(), rtMemcpyDeviceToHost, ctx.stream()));
    RPP_CHECK(rtStreamSynchronize(ctx.stream()));

    GGML_TENSOR_LOCALS(int64_t, ne, tensor, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, tensor, nb);

    stats.elements = ggml_nelements(tensor);
    int64_t index = 0;
    for (int64_t i03 = 0; i03 < ne3; ++i03) {
        for (int64_t i02 = 0; i02 < ne2; ++i02) {
            for (int64_t i01 = 0; i01 < ne1; ++i01) {
                const char * row_ptr = (const char *) raw.data() + i03*nb3 + i02*nb2 + i01*nb1;
                if (tensor->type == GGML_TYPE_F32) {
                    const float * src = (const float *) row_ptr;
                    for (int64_t i00 = 0; i00 < ne0; ++i00) {
                        ggml_rpp_tanh_stats_update(stats, src[i00], index++);
                    }
                } else {
                    const ggml_bf16_t * src = (const ggml_bf16_t *) row_ptr;
                    for (int64_t i00 = 0; i00 < ne0; ++i00) {
                        ggml_rpp_tanh_stats_update(stats, ggml_bf16_to_fp32(src[i00]), index++);
                    }
                }
            }
        }
    }

    return true;
}

static bool ggml_rpp_tanh_collect_bf16_device_stats(
        ggml_backend_rpp_context & ctx,
        RPPdeviceptr device_ptr,
        int64_t elements,
        rpp_tanh_tensor_stats & stats) {
    std::vector<ggml_bf16_t> raw(elements);
    RPP_CHECK(rtMemcpyAsync(raw.data(), (const char *) device_ptr, raw.size() * sizeof(ggml_bf16_t),
                            rtMemcpyDeviceToHost, ctx.stream()));
    RPP_CHECK(rtStreamSynchronize(ctx.stream()));

    stats.elements = elements;
    for (int64_t i = 0; i < elements; ++i) {
        ggml_rpp_tanh_stats_update(stats, ggml_bf16_to_fp32(raw[i]), i);
    }

    return true;
}

static int ggml_rpp_tanh_seq_len(ggml_tensor * dst, ggml_rpp_node * node) {
    return dst->ne[node->seq_len_index];
}

static bool ggml_rpp_create_kernel_tanh(ggml_backend_rpp_context & ctx,
                                        ggml_rpp_node *            rpp_base_node,
                                        ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto                rpp_node = static_cast<rpp_kernel_tanh *>(rpp_base_node);
    const ggml_tensor * src0     = dst->src[0];
    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    GGML_ASSERT(dst->src[1] == nullptr);
    GGML_ASSERT(dst->op == GGML_OP_UNARY);
    GGML_ASSERT(ggml_get_unary_op(dst) == GGML_UNARY_OP_TANH);

    const int seq_len = ggml_rpp_tanh_seq_len(dst, rpp_node);
    const int C = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 2) ? rpp_node->n_ubatch : src0->ne[2];
    const int H = (ctx.use_ubatch && seq_len > 1 && rpp_node->seq_len_index == 1) ? rpp_node->n_ubatch : src0->ne[1];
    const int W = src0->ne[0];

    void * inputs_0_buf = src0->data;
    rpp_node->kernel_ctx->dev_in.emplace_back((RPPdeviceptr) inputs_0_buf);
    rpp_node->kernel_ctx->dev_out.emplace_back((RPPdeviceptr) dst->data);
    rpp_node->binding_i_buffers.emplace(dst->src[0], inputs_0_buf);
    rpp_node->binding_o_buffers.emplace(dst, dst->data);
    rpp_node->binding_io_buffers.emplace_back(inputs_0_buf);
    rpp_node->binding_io_buffers.emplace_back(dst->data);

    const int i_type_size = ggml_rpp_get_io_type_size(ctx, dst->src[0], 0);
    const int o_type_size = ggml_rpp_get_io_type_size(ctx, dst, 1);
    kernel_tanh::rpp_tanh_build(*(rpp_node->kernel_ctx.get()), C, H, W, i_type_size, o_type_size,
                                rpp_node->is_instantial);
    return true;
}

static bool ggml_rpp_create_kernel_dispatch(ggml_backend_rpp_context & ctx,
                                            ggml_rpp_node *            rpp_base_node,
                                            ggml_tensor *              dst) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_tanh *>(rpp_base_node);

    if (ctx.cur_rpp_graph->rpp_nodes[dst].size() == 1) {
        int n                   = ggml_n_dims(dst);
        rpp_node->seq_len_index = n == 1 ? 1 : n - 1;
    } else {
        rpp_node->seq_len_index = ctx.cur_rpp_graph->rpp_nodes[dst].front().get()->seq_len_index;
    }

    if (ctx.use_ubatch && ggml_rpp_tanh_seq_len(dst, rpp_node) > 1) {
        rpp_node->n_ubatch = ctx.n_ubatch;
    }

    const bool ret = ggml_rpp_create_kernel_tanh(ctx, rpp_node, dst);
    GGML_ASSERT(ret);

    if (ret) {
        ggml_rpp_node_set_properties(rpp_node, dst);
        rpp_node->binding_io_tensors.emplace_back(dst->src[0]);
        rpp_node->binding_io_tensors.emplace_back(dst);
    }
    return ret;
}

static bool ggml_rpp_tanh_properties_is_same(ggml_backend_rpp_context & ctx,
                                             ggml_tensor *              dst,
                                             ggml_rpp_node *            rpp_node) {
    (void) ctx;
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
    return true;
}

static bool ggml_rpp_launch_kernel(ggml_backend_rpp_context & ctx, ggml_tensor * dst, ggml_rpp_node * rpp_base_node) {
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_kernel_tanh *>(rpp_base_node);
    try {
        RPP_LAUNCH_KERNEL(rpp_node->kernel_ctx->graphexec, ctx.stream());
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: infer failed, %s (%s), error: %s\n", __func__, dst->name, ggml_op_name(dst->op), e.what());
        return false;
    }
    return true;
}

bool ggml_rpp_op_kernel_tanh(ggml_backend_rpp_context & ctx, ggml_tensor * dst, int is_instantial, int is_launch) {
    if (!dst) {
        GGML_LOG_ERROR("%s: ggml_tensor is nullptr\n", __func__);
        return false;
    }

    rpp_kernel_tanh * rpp_node = nullptr;
    auto              iter     = ctx.cur_rpp_graph->cur_rpp_nodes.find(dst);
    if (iter == ctx.cur_rpp_graph->cur_rpp_nodes.end()) {
        auto iter_node = ctx.cur_rpp_graph->rpp_nodes.find(dst);
        if (iter_node != ctx.cur_rpp_graph->rpp_nodes.end()) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "find_kernel_tanh");
            auto & node_vec = ctx.cur_rpp_graph->rpp_nodes[dst];
            for (size_t i = 0; i < node_vec.size(); i++) {
                auto cur_node = node_vec[i].get();
                if (cur_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL &&
                    ggml_rpp_tanh_properties_is_same(ctx, dst, cur_node)) {
                    rpp_node = (rpp_kernel_tanh *) cur_node;
                    break;
                }
            }
        }

        if (!rpp_node) {
            TRACE_SCOPE_GUARD(ctx.trace_id, "create_kernel_tanh");
            auto new_node = std::make_unique<rpp_kernel_tanh>(dst);
            ctx.cur_rpp_graph->rpp_nodes[dst].emplace_back(std::move(new_node));
            rpp_node                = (rpp_kernel_tanh *) (ctx.cur_rpp_graph->rpp_nodes[dst].back().get());
            rpp_node->is_instantial = is_instantial;
            if (!(ggml_rpp_create_kernel_dispatch(ctx, rpp_node, dst))) {
                return false;
            }
        }
        GGML_ASSERT(rpp_node);
        ctx.cur_rpp_graph->cur_rpp_nodes[dst] = rpp_node;
        ctx.cur_rpp_graph->rpp_in_use_nodes.emplace_back(rpp_node);
    } else {
        rpp_node = (rpp_kernel_tanh *) (iter->second);
    }

    bool ret = true;
    if (is_launch) {
        TRACE_SCOPE_GUARD(ctx.trace_id, "launch_kernel_tanh");
        ret = ggml_rpp_launch_kernel(ctx, dst, rpp_node);
    }
    return ret;
}
