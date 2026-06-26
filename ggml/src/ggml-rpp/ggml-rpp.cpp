
#include "ggml-rpp.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "ggml-rpp/rpp_ops_utils.h"
#include "ggml.h"
#include "llama.h"
#include "rpp_add/rpp_add.h"
#include "rpp_common.h"
#include "rpp_cont/rpp_cont.h"
#include "rpp_cpy/rpp_cpy.h"
#include "rpp_div/rpp_div.h"
#include "rpp_drv_api.h"
#include "rpp_expert_forward/rpp_expert_forward.h"
#include "rpp_expert_routing/rpp_expert_routing.h"
#include "rpp_flash_attn_ext/rpp_flash_attn_ext.h"
#include "rpp_get_rows/rpp_get_rows.h"
#include "rpp_glu/rpp_glu.h"
#include "rpp_l2_norm/rpp_l2_norm.h"
#include "rpp_mul/rpp_mul.h"
#include "rpp_mul_mat/rpp_mul_mat.h"
#include "rpp_mul_mat_id/rpp_mul_mat_id.h"
#include "rpp_norm/rpp_norm.h"
#include "rpp_pool_2d/rpp_pool_2d.h"
#include "rpp_reduce_sum/rpp_reduce_sum.h"
#include "rpp_rms_norm/rpp_rms_norm.h"
#include "rpp_rope/rpp_rope.h"
#include "rpp_runtime.h"
#include "rpp_scale/rpp_scale.h"
#include "rpp_set_rows/rpp_set_rows.h"
#include "rpp_unary/rpp_unary.h"

#include <float.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

void rpp_kernel_cgraph::add_child_graph(ggml_rpp_node * rpp_node) {
    GGML_ASSERT(rpp_node);
    rpp_nodes.emplace_back(rpp_node);
    auto kernel_node = static_cast<rpp_node_kernel *>(rpp_node);
    kernel_node->add_to_parent_node(graph);
    rpp_to_graph_nodes[rpp_node] = kernel_node->kernel_ctx->graph_node;
}

void rpp_kernel_cgraph::update_child_graph(std::unordered_map<ggml_tensor *, ggml_rpp_node *> & cur_rpp_nodes) {
    RPP_CHILD_GRAPHEXEC_REPLACE_PARAMS execReplaceParams;
    RPP_GRAPH_INSTANTIATE_PARAMS       instChild = {};
    if (!is_shared) {
        GGML_ASSERT(rpp_nodes.size() == 1);
        auto old_node  = rpp_nodes[0];
        auto ggml_node = old_node->cur_ggml_tensor;
        auto iter      = cur_rpp_nodes.find(ggml_node);
        GGML_ASSERT(iter != cur_rpp_nodes.end());
        if (iter->second != old_node) {
            auto new_node = static_cast<rpp_node_kernel *>(iter->second);
            if (new_node->kernel_ctx->graphexec == nullptr) {
                RPP_CHECK(rppGraphInstantiate(&(new_node->kernel_ctx->graphexec), new_node->kernel_ctx->graph, nullptr,
                                              nullptr, 0));
            }
            // Keep wrapper handles aligned with the real per-kernel context handles.
            graph        = new_node->kernel_ctx->graph;
            graphexec    = new_node->kernel_ctx->graphexec;
            rpp_nodes[0] = new_node;
            rpp_to_graph_nodes.erase(old_node);
            rpp_to_graph_nodes[new_node] = new_node->kernel_ctx->graph_node;
        }
        return;
    }

    for (auto & rpp_node : rpp_nodes) {
        auto ggml_tensor = rpp_node->cur_ggml_tensor;
        auto iter        = cur_rpp_nodes.find(ggml_tensor);
        GGML_ASSERT(iter != cur_rpp_nodes.end());
        if (iter->second != rpp_node) {
            // Shared parent graphexec: swap child binding via rppGraphInstantiateWithParams(CHILD_EXEC) +
            // rppGraphExecUpdateChildGraphExec only (no rppGraphUpdateChildGraph / rppGraphExecUpdate).
            ggml_rpp_node * new_rpp_node = iter->second;
            ggml_rpp_node * old_rpp_node = rpp_node;

            auto old_kernel_node = static_cast<rpp_node_kernel *>(old_rpp_node);
            auto new_kernel_node = static_cast<rpp_node_kernel *>(new_rpp_node);

            RPPgraphNode childNode = rpp_to_graph_nodes[old_rpp_node];

            rpp_kernel_context * old_kctx       = old_kernel_node->kernel_ctx.get();
            rpp_kernel_context * new_kctx       = new_kernel_node->kernel_ctx.get();
            RPPgraphExec         old_child_exec = old_kctx->graphexec;

            memset(&execReplaceParams, 0, sizeof(execReplaceParams));
            execReplaceParams.newChildGraphExec = new_kctx->graphexec;
            execReplaceParams.flags             = 0;
            RPP_CHECK(rppGraphExecUpdateChildGraphExec(graphexec, childNode, &execReplaceParams));

            new_kernel_node->kernel_ctx->graph_node = childNode;
            rpp_node                                = new_rpp_node;
            rpp_to_graph_nodes.erase(old_rpp_node);
            rpp_to_graph_nodes[new_rpp_node] = childNode;
        }
    }
}

void rpp_kernel_cgraph::graph_instantiate() {
    if (rpp_nodes.empty()) {
        return;
    }
    auto rpp_node = rpp_nodes.front();
    GGML_ASSERT(rpp_node && rpp_node->cur_ggml_tensor);

    if (is_shared) {
        if (graphexec == nullptr) {
            RPP_CHECK(rppGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));
        }
        return;
    }

    GGML_ASSERT(rpp_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL);
    auto kernel_node = static_cast<rpp_node_kernel *>(rpp_node);
    GGML_ASSERT(kernel_node->kernel_ctx);
    if (rpp_node->cur_ggml_tensor->op == GGML_OP_MUL_MAT_ID &&
        kernel_node->op == rpp_node_kernel::RPP_OP_EXPERT_FORWARD) {
        return;
    }
    if (kernel_node->kernel_ctx->graphexec == nullptr) {
        RPP_CHECK(rppGraphInstantiate(&(kernel_node->kernel_ctx->graphexec), kernel_node->kernel_ctx->graph, nullptr,
                                      nullptr, 0));
    }
    // Keep wrapper handles aligned with the real per-kernel context handles.
    graph     = kernel_node->kernel_ctx->graph;
    graphexec = kernel_node->kernel_ctx->graphexec;
}

void rpp_kernel_cgraph::graph_launch(ggml_backend_rpp_context & ctx, rtStream_t stream, int trace_id) {
    if (is_shared) {
        try {
            TRACE_SCOPE_GUARD(trace_id, "launch_kernel_global_graph");
            RPP_LAUNCH_KERNEL(graphexec, stream);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: infer failed, error: %s\n", __func__, e.what());
        }
    } else {
        TRACE_SCOPE_GUARD(trace_id, "launch_kernel_exclusive_graph");
        auto       rpp_node = rpp_nodes.front();
        const bool ok       = rpp_node->rpp_dispatch_func(ctx, rpp_node->cur_ggml_tensor, 1, 1);
        if (!ok) {
            GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, rpp_node->cur_ggml_tensor->name,
                           ggml_op_name(rpp_node->cur_ggml_tensor->op));
        }
        GGML_ASSERT(ok);
    }
}

[[noreturn]] void ggml_rpp_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1;  // in case rppGetDevice fails
    (void) rtGetDevice(&id);

    GGML_LOG_ERROR(GGML_RPP_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_RPP_NAME " error");
}

// this is faster on Windows
// probably because the Windows RPP libraries forget to make this check before invoking the drivers
void ggml_rpp_set_device(int device) {
    int current_device;
    RPP_CHECK(rtGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    RPP_CHECK(rtSetDevice(device));
}

int ggml_rpp_get_device() {
    int id;
    RPP_CHECK(rtGetDevice(&id));
    return id;
}

void ggml_rpp_reset_graph(ggml_backend_rpp_context * rpp_ctx, ggml_rpp_cgraph * rpp_graph) {
    // auto iter = rpp_graph->cur_rpp_nodes.begin();
    // while (iter != rpp_graph->cur_rpp_nodes.end()) {
    //     ggml_rpp_reset_node(rpp_ctx, rpp_graph, iter->first);
    //     iter++;
    // }
    TRACE_SCOPE_GUARD(rpp_ctx->trace_id, "reset_graph");
    rpp_graph->cur_rpp_nodes.clear();
    rpp_graph->nodes_all.clear();
    rpp_graph->nodes_i.clear();
    rpp_graph->nodes_o.clear();
    rpp_graph->nodes_cache_kv.clear();
    rpp_graph->nodes_mul_weight.clear();
    rpp_graph->nodes_matmul_weight.clear();
    rpp_graph->rpp_in_use_kernel_graphs.clear();
    rpp_graph->rpp_in_use_nodes.clear();
    rpp_graph->launch_funcs.clear();
}

void ggml_rpp_reset_node(ggml_backend_rpp_context * rpp_ctx, ggml_rpp_cgraph * rpp_graph, ggml_tensor * g_tensor) {
    auto iter = rpp_graph->cur_rpp_nodes.find(g_tensor);
    if (iter != rpp_graph->cur_rpp_nodes.end()) {
        auto & i_buffers = iter->second->binding_i_buffers;
        for (auto i_iter : i_buffers) {
            if (iter->second->pool_buffers.count(i_iter.second)) {
                rpp_ctx->rpp_io_buffers.erase(i_iter.first);
                rpp_ctx->pool().free(i_iter.second);
            }
            i_buffers[i_iter.first] = nullptr;
        }
        auto & o_buffers = iter->second->binding_o_buffers;
        for (auto o_iter : o_buffers) {
            if (iter->second->pool_buffers.count(o_iter.second)) {
                rpp_ctx->rpp_io_buffers.erase(o_iter.first);
                rpp_ctx->pool().free(o_iter.second);
            }
            o_buffers[o_iter.first] = nullptr;
        }
        iter->second->binding_io_buffers.clear();
        iter->second->pool_buffers.clear();
    }
}

void ggml_rpp_node_set_properties(ggml_rpp_node * rpp_node, ggml_tensor * dst) {
    ggml_tensor * cur_tensor[GGML_MAX_SRC + 1] = { dst };
    auto          end_iter                     = std::copy_if(dst->src, dst->src + GGML_MAX_SRC, cur_tensor + 1,
                                                              [](ggml_tensor * ptr) { return ptr != nullptr; });
    for (int i = 0; i < end_iter - cur_tensor; i++) {
        ggml_tensor *              node = cur_tensor[i];
        ggml_graph_node_properties graph_node_properties;
        graph_node_properties.node_address = node->data;
        graph_node_properties.node_op      = node->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            graph_node_properties.ne[i] = node->ne[i];
            graph_node_properties.nb[i] = node->nb[i];
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            graph_node_properties.src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
        }
        memcpy(graph_node_properties.op_params, node->op_params, GGML_MAX_OP_PARAMS);
        rpp_node->ggml_node_properties[cur_tensor[i]] = graph_node_properties;
    }
}

bool ggml_rpp_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    if (node->data != graph_node_properties->node_address && node->op != GGML_OP_CPY && node->op != GGML_OP_VIEW) {
        return false;
    }
    if (node->op != graph_node_properties->node_op) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties->ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties->nb[i]) {
            return false;
        }
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] && node->src[i]->data != graph_node_properties->src_address[i] && node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW) {
            return false;
        }
    }
    if (node->op == GGML_OP_SCALE &&
        memcmp(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }
    return true;
}

bool ggml_rpp_node_has_matching_properties(ggml_tensor * node, ggml_rpp_node * rpp_node) {
    bool has_matching_properties = true;
    if (!rpp_node->ggml_node_properties.size()) {
        return false;
    }
    for (auto iter : rpp_node->ggml_node_properties) {
        has_matching_properties = ggml_rpp_node_has_matching_properties(iter.first, &(iter.second));
        if (!has_matching_properties) {
            break;
        }
    }
    return has_matching_properties;
}

static rtError_t ggml_rpp_device_malloc(void ** ptr, size_t size, int device) {
    ggml_rpp_set_device(device);
    rtError_t err = rtMalloc(ptr, size);
    return err;
}

/**
 * @brief Initialize the RPP device information.
 *
 * This function initializes the RPP device information by obtaining the
 * device count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
static ggml_rpp_device_info ggml_rpp_init() {
    ggml_rpp_device_info info = {};

    rtError_t err = rtGetDeviceCount(&info.device_count);
    if (err != rtSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_RPP_NAME ": %s\n", __func__, rtGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_RPP_MAX_DEVICES);

    int64_t total_vram = 0;
    GGML_LOG_INFO("%s: found %d " GGML_RPP_NAME " devices:\n", __func__, info.device_count);

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

        info.devices[id].vmm = !!device_vmm;
        rtDeviceProp prop;
        RPP_CHECK(rtGetDeviceProperties(&prop, id));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;
        info.devices[id].integrated = prop.integrated;
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        // info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        // RPP_CHECK(rppDeviceGetAttribute(&info.devices[id].smpbo, RPP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, id));
        info.devices[id].cc = 100 * prop.major + 10 * prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s\n", id, prop.name, prop.major, prop.minor,
                      device_vmm ? "yes" : "no");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    return info;
}

const ggml_rpp_device_info & ggml_rpp_info() {
    static ggml_rpp_device_info info = ggml_rpp_init();
    return info;
}

// #define DEBUG_RPP_MALLOC

// buffer pool for rpp (legacy)
struct ggml_rpp_pool_leg : public ggml_rpp_pool {
    static const int MAX_BUFFERS = 256;
    int              device;
    int              type;  //0:in host, 1:in device

    struct ggml_rpp_buffer {
        void * ptr  = nullptr;
        size_t size = 0;
    };

    ggml_rpp_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t          pool_size                = 0;

    explicit ggml_rpp_pool_leg(int device, int type = 1) : device(device), type(type) {}

    ~ggml_rpp_pool_leg() {
        ggml_rpp_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_rpp_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                if (type) {
                    RPP_CHECK(rtFree(b.ptr));
                } else {
                    RPP_CHECK(rtFreeHost(b.ptr));
                }
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_RPP_MALLOC
        int    nnz      = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int    ibest     = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_rpp_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_RPP_MALLOC
                ++nnz;
                if (b.size > max_size) {
                    max_size = b.size;
                }
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest     = i;
                        if (!best_diff) {
                            void * ptr   = b.ptr;
                            *actual_size = b.size;
                            b.ptr        = nullptr;
                            b.size       = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_rpp_buffer & b   = buffer_pool[ibest];
            void *            ptr = b.ptr;
            *actual_size          = b.size;
            b.ptr                 = nullptr;
            b.size                = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size        = 256 * ((look_ahead_size + 255) / 256);
        ggml_rpp_set_device(device);
        if (type) {
            RPP_CHECK(ggml_rpp_device_malloc(&ptr, look_ahead_size, device));
        } else {
            RPP_CHECK(rtMallocHost(&ptr, look_ahead_size));
        }
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_RPP_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device,
                      nnz, (uint32_t) (max_size / 1024 / 1024), (uint32_t) (pool_size / 1024 / 1024),
                      (uint32_t) (size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_rpp_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_RPP_NAME " buffer pool full, increase MAX_RPP_BUFFERS\n");
        ggml_rpp_set_device(device);
        if (type) {
            RPP_CHECK(rtFree(ptr));
        } else {
            RPP_CHECK(rtFreeHost(ptr));
        }
        pool_size -= size;
    }
};

// buffer pool for rpp (mapping)
struct ggml_rpp_pool_map : public ggml_rpp_pool {
    int device;
    int type;  //0:in host, 1:in device

    std::unordered_set<void *>              buffer_set_ldle;
    std::unordered_multimap<size_t, void *> buffer_map_ldle;
    std::unordered_map<void *, size_t>      buffer_pool;
    size_t                                  pool_size = 0;

    explicit ggml_rpp_pool_map(int device, int type = 1) : device(device), type(type) {}

    ~ggml_rpp_pool_map() {
        ggml_rpp_set_device(device);
        for (auto iter : buffer_pool) {
            if (iter.first != nullptr) {
                if (type) {
                    RPP_CHECK(rtFree(iter.first));
                } else {
                    RPP_CHECK(rtFreeHost(iter.first));
                }
                pool_size -= iter.second;
            }
        }
        GGML_ASSERT(pool_size == 0);
        buffer_set_ldle.clear();
        buffer_map_ldle.clear();
        buffer_pool.clear();
    }

    void * alloc(size_t size, size_t * actual_size = nullptr) override {
        void * ptr  = nullptr;
        auto   iter = buffer_map_ldle.find(size);
        if (iter != buffer_map_ldle.end()) {
            ptr = iter->second;
            buffer_map_ldle.erase(iter);
            buffer_set_ldle.erase(ptr);
        }
        if (!ptr) {
            ggml_rpp_set_device(device);
            if (type) {
                RPP_CHECK(rtMalloc(&ptr, size));
            } else {
                RPP_CHECK(rtMallocHost(&ptr, size));
            }

            pool_size += size;
            buffer_pool.emplace(ptr, size);
        }
        return ptr;
    }

    void free(void * ptr, size_t size = 0) override {
        if (!buffer_pool.count(ptr)) {
            GGML_LOG_INFO("%s: the ptr:(%p) is not belonging to the pool \n", __func__, ptr);
            return;
        }
        if (size && size != buffer_pool[ptr]) {
            GGML_LOG_ERROR("%s: the size of ptr:(%p) is incorrect \n", __func__, ptr);
            return;
        }
        if (buffer_set_ldle.find(ptr) == buffer_set_ldle.end()) {
            buffer_set_ldle.emplace(ptr);
            buffer_map_ldle.emplace(buffer_pool[ptr], ptr);
        } else {
            GGML_LOG_INFO("%s: the ptr:(%p) is already in the pool \n", __func__, ptr);
        }
    }
};

struct ggml_rpp_pool_mem : public ggml_rpp_pool {
    int device;
    int type;  // 0: in host, 1: in device

    explicit ggml_rpp_pool_mem(int device, int type = 1, std::size_t capacity = 16 * 1024 * 1024) :
        device(device),
        type(type),
        capacity_(align_up(capacity)) {
        ggml_rpp_set_device(device);
        void * raw_ptr = nullptr;
        if (type) {
            RPP_CHECK(rtMalloc(&raw_ptr, capacity_));
        } else {
            RPP_CHECK(rtMallocHost(&raw_ptr, capacity_));
        }
        buffer_ = static_cast<uint8_t *>(raw_ptr);
        free_blocks_.emplace(0, capacity_);
    }

    ~ggml_rpp_pool_mem() override {
        if (buffer_ == nullptr) {
            return;
        }
        ggml_rpp_set_device(device);
        if (type) {
            RPP_CHECK(rtFree(buffer_));
        } else {
            RPP_CHECK(rtFreeHost(buffer_));
        }
        buffer_ = nullptr;
    }

    ggml_rpp_pool_mem(const ggml_rpp_pool_mem &)             = delete;
    ggml_rpp_pool_mem & operator=(const ggml_rpp_pool_mem &) = delete;

    void * alloc(size_t size, size_t * actual_size = nullptr) override {
        if (actual_size) {
            *actual_size = 0;
        }
        if (size == 0) {
            return nullptr;
        }

        const std::size_t           need = align_up(size);
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            std::size_t offset     = it->first;
            std::size_t block_size = it->second;

            if (block_size < need) {
                continue;
            }

            std::size_t granted = need;
            std::size_t remain  = block_size - need;
            // because the npu only can access the first 3G of ddr, so we need to avoid the small fragments
            if (remain > 0 && remain < kMinSplit) {
                granted = block_size;
                remain  = 0;
            }
            free_blocks_.erase(it);
            if (remain > 0) {
                free_blocks_.emplace(offset + granted, remain);
            }

            void * ptr             = buffer_ + offset;
            allocated_blocks_[ptr] = granted;

            if (actual_size) {
                *actual_size = granted;
            }
            return ptr;
        }

        return nullptr;
    }

    void free(void * ptr, size_t size = 0) override {
        if (ptr == nullptr) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        if (!owns(ptr)) {
            throw std::invalid_argument("pointer does not belong to this MemoryPool");
        }

        std::size_t block_size = 0;

        auto alloc_it = allocated_blocks_.find(ptr);
        if (alloc_it != allocated_blocks_.end()) {
            // the actual allocated size is used to avoid the mismatch between the size passed in and the actual allocated size
            block_size = alloc_it->second;
            allocated_blocks_.erase(alloc_it);
        } else {
            if (size == 0) {
                throw std::invalid_argument("unknown block size, please pass size or free a tracked pointer");
            }
            block_size = align_up(size);
        }

        const std::size_t offset = static_cast<uint8_t *>(ptr) - buffer_;
        insert_and_coalesce(offset, block_size);
    }

    std::size_t capacity() const noexcept { return capacity_; }

    std::size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t                 total = 0;
        for (const auto & kv : free_blocks_) {
            total += kv.second;
        }
        return total;
    }

  private:
    static constexpr std::size_t kAlign    = alignof(std::max_align_t);
    static constexpr std::size_t kMinSplit = kAlign;

    static std::size_t align_up(std::size_t n) noexcept { return ((n + kAlign - 1) / kAlign) * kAlign; }

    bool owns(void * ptr) const noexcept {
        auto * p = static_cast<uint8_t *>(ptr);
        return buffer_ != nullptr && p >= buffer_ && p < (buffer_ + capacity_);
    }

    void insert_and_coalesce(std::size_t offset, std::size_t size) {
        auto it = free_blocks_.lower_bound(offset);
        // check if the left block is overlapping or adjacent to the current block
        if (it != free_blocks_.begin()) {
            auto              prev     = std::prev(it);
            const std::size_t prev_end = prev->first + prev->second;

            if (prev_end > offset) {
                throw std::runtime_error("double free or overlapping free block");
            }

            if (prev_end == offset) {
                offset = prev->first;
                size += prev->second;
                free_blocks_.erase(prev);
            }
        }
        // check if the right block is overlapping or adjacent to the current block
        while (it != free_blocks_.end()) {
            if (offset + size < it->first) {
                break;
            }
            if (offset + size > it->first) {
                throw std::runtime_error("double free or overlapping free block");
            }
            // merge the current block with the right block
            size += it->second;
            it = free_blocks_.erase(it);
        }

        free_blocks_.emplace(offset, size);
    }

  private:
    std::size_t                             capacity_ = 0;
    uint8_t *                               buffer_   = nullptr;
    // free blocks: key=offset, value=size
    std::map<std::size_t, std::size_t>      free_blocks_;
    // allocated blocks: key=ptr, value=actual allocated size
    std::unordered_map<void *, std::size_t> allocated_blocks_;
    mutable std::mutex                      mutex_;
};

std::unique_ptr<ggml_rpp_pool> ggml_backend_rpp_context::new_pool_for_device(int device) {
    return std::unique_ptr<ggml_rpp_pool>(new ggml_rpp_pool_map(device, 1));
}

std::unique_ptr<ggml_rpp_pool> ggml_backend_rpp_context::new_pool_for_device_leg(int device) {
    return std::unique_ptr<ggml_rpp_pool>(new ggml_rpp_pool_leg(device, 1));
}

std::unique_ptr<ggml_rpp_pool> ggml_backend_rpp_context::new_pool_for_device_mem(int device) {
    return std::unique_ptr<ggml_rpp_pool>(new ggml_rpp_pool_mem(device, 1));
}

std::unique_ptr<ggml_rpp_pool> ggml_backend_rpp_context::new_pool_for_host(int device) {
    return std::unique_ptr<ggml_rpp_pool>(new ggml_rpp_pool_map(device, 0));
}

std::unique_ptr<ggml_rpp_pool> ggml_backend_rpp_context::new_pool_for_host_leg(int device) {
    return std::unique_ptr<ggml_rpp_pool>(new ggml_rpp_pool_leg(device, 0));
}

std::unique_ptr<ggml_rpp_pool> ggml_backend_rpp_context::new_pool_for_host_mem(int device) {
    return std::unique_ptr<ggml_rpp_pool>(new ggml_rpp_pool_mem(device, 0));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a RPP error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex              ggml_rpp_lock;
static std::condition_variable ggml_rpp_lock_cv;
static std::atomic<int>        ggml_rpp_lock_counter;

ggml_backend_rpp_context::~ggml_backend_rpp_context() {
    std::unique_lock<std::mutex> lock(ggml_rpp_lock);
    ggml_rpp_lock_cv.wait(lock, [] { return ggml_rpp_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        RPP_CHECK(rtEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_RPP_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_RPP_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                RPP_CHECK(rtStreamDestroy(streams[i][j]));
            }
        }
    }
#if GGML_RPP_PERF_TRACE
    TRACE_END_ALL();
#endif
}

// rpp buffer
struct ggml_backend_rpp_buffer_context {
    int         device;
    void *      dev_ptr = nullptr;
    std::string name;

    ggml_backend_rpp_buffer_context(int device, void * dev_ptr) :
        device(device),
        dev_ptr(dev_ptr),
        name(GGML_RPP_NAME + std::to_string(device)) {}

    ~ggml_backend_rpp_buffer_context() { RPP_CHECK(rtFree(dev_ptr)); }
};

/**
 * @brief Free resources associated with a RPP buffer.
 *
 * This function frees the resources associated with a RPP buffer, including
 * its context.
 *
 * @param buffer The RPP buffer to free.
 */
static void ggml_backend_rpp_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpp_buffer_context * ctx = (ggml_backend_rpp_buffer_context *) buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_rpp(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpp_buffer_free_buffer;
}

/**
 * @brief Retrieve the base pointer of a RPP buffer.
 *
 * This function returns the base pointer of a RPP buffer, which points to the
 * device memory allocated for the buffer.
 *
 * @param buffer The RPP buffer whose base pointer is to be retrieved.
 * @return A pointer to the base of the device memory allocated for the buffer.
 */
static void * ggml_backend_rpp_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpp_buffer_context * ctx = (ggml_backend_rpp_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

namespace {

static const char         GGML_RPP_WEIGHTS_CACHE_MAGIC[8] = { 'R', 'P', 'P', 'W', 'C', '0', '1', '\0' };
static constexpr uint32_t GGML_RPP_WEIGHTS_CACHE_VERSION  = 4;

struct ggml_rpp_weights_cache_file_header {
    char     magic[8];
    uint32_t version;
    uint32_t reserved;
};

struct ggml_rpp_weights_cache_entry_header {
    uint32_t key_size;
    uint32_t reserved;
    uint64_t input_size;
    uint64_t output_size;
    uint64_t input_fingerprint;
};

struct ggml_rpp_weights_cache_index_entry {
    uint64_t payload_offset;
    uint64_t input_size;
    uint64_t output_size;
    uint64_t input_fingerprint;
};

struct ggml_rpp_weights_cache_state {
    std::mutex                                                          mutex;
    bool                                                                initialized = false;
    bool                                                                enabled     = false;
    std::string                                                         file_path;
    std::unordered_map<std::string, ggml_rpp_weights_cache_index_entry> index;
};

static ggml_rpp_weights_cache_state & ggml_rpp_weights_cache_get_state() {
    static ggml_rpp_weights_cache_state cache_state;
    return cache_state;
}

static uint64_t ggml_rpp_weights_cache_hash_bytes(const void * data, size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    uint64_t        hash  = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis

    auto mix_byte = [&hash](uint8_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    auto mix_u64 = [&mix_byte](uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            mix_byte((uint8_t) (value & 0xFFu));
            value >>= 8;
        }
    };

    mix_u64((uint64_t) size);
    if (bytes == nullptr || size == 0) {
        return hash;
    }

    const size_t sample = std::min<size_t>(128, size);
    for (size_t i = 0; i < sample; ++i) {
        mix_byte(bytes[i]);
    }

    if (size > sample) {
        const size_t mid = (size - sample) / 2;
        for (size_t i = 0; i < sample; ++i) {
            mix_byte(bytes[mid + i]);
        }

        const size_t tail = size - sample;
        for (size_t i = 0; i < sample; ++i) {
            mix_byte(bytes[tail + i]);
        }
    }

    return hash;
}

static std::string ggml_rpp_weights_cache_make_key(const ggml_tensor * tensor) {
    std::string  key;
    const char * tensor_name = ggml_get_name(tensor);
    if (tensor_name != nullptr) {
        key = tensor_name;
    } else {
        key = "unnamed";
    }
    key.reserve(key.size() + 192);
    key += "|t=" + std::to_string((int) tensor->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        key += "|ne" + std::to_string(i) + "=" + std::to_string(tensor->ne[i]);
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        key += "|nb" + std::to_string(i) + "=" + std::to_string(tensor->nb[i]);
    }
    return key;
}

static bool ggml_rpp_weights_cache_create_empty_file(const std::string & cache_path) {
    std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        GGML_LOG_ERROR("%s: failed to create cache file: %s\n", __func__, cache_path.c_str());
        return false;
    }

    ggml_rpp_weights_cache_file_header header = {};
    memcpy(header.magic, GGML_RPP_WEIGHTS_CACHE_MAGIC, sizeof(header.magic));
    header.version = GGML_RPP_WEIGHTS_CACHE_VERSION;

    out.write((const char *) &header, sizeof(header));
    if (!out.good()) {
        GGML_LOG_ERROR("%s: failed to write cache header: %s\n", __func__, cache_path.c_str());
        return false;
    }

    return true;
}

static void ggml_rpp_weights_cache_init_locked(ggml_rpp_weights_cache_state & cache_state) {
    if (cache_state.initialized) {
        return;
    }
    cache_state.initialized = true;

    const char * cache_path = getenv("GGML_RPP_WEIGHTS_CACHE_FILE");
    if (cache_path == nullptr || cache_path[0] == '\0') {
        return;
    }

    cache_state.file_path = cache_path;

    std::ifstream in(cache_state.file_path, std::ios::binary);
    if (!in.is_open()) {
        if (!ggml_rpp_weights_cache_create_empty_file(cache_state.file_path)) {
            return;
        }
        cache_state.enabled = true;
        GGML_LOG_INFO("%s: enabled RPP weight cache file: %s (new file)\n", __func__, cache_state.file_path.c_str());
        return;
    }

    ggml_rpp_weights_cache_file_header header = {};
    in.read((char *) &header, sizeof(header));
    const bool valid_header = in.good() &&
                              memcmp(header.magic, GGML_RPP_WEIGHTS_CACHE_MAGIC, sizeof(header.magic)) == 0 &&
                              header.version == GGML_RPP_WEIGHTS_CACHE_VERSION;

    if (!valid_header) {
        in.close();
        if (!ggml_rpp_weights_cache_create_empty_file(cache_state.file_path)) {
            return;
        }
        cache_state.enabled = true;
        GGML_LOG_INFO("%s: reset invalid RPP weight cache file: %s\n", __func__, cache_state.file_path.c_str());
        return;
    }

    while (true) {
        ggml_rpp_weights_cache_entry_header entry_header = {};
        in.read((char *) &entry_header, sizeof(entry_header));
        if (in.eof()) {
            break;
        }
        if (!in.good()) {
            GGML_LOG_DEBUG("%s: stop parsing cache file due to read error: %s\n", __func__,
                           cache_state.file_path.c_str());
            break;
        }
        if (entry_header.key_size == 0 || entry_header.key_size > (1u << 20) || entry_header.output_size == 0 ||
            entry_header.output_size > (uint64_t) std::numeric_limits<std::streamsize>::max()) {
            GGML_LOG_DEBUG("%s: stop parsing cache file due to invalid entry header\n", __func__);
            break;
        }

        std::string key(entry_header.key_size, '\0');
        in.read((char *) key.data(), (std::streamsize) entry_header.key_size);
        if (!in.good()) {
            GGML_LOG_DEBUG("%s: stop parsing cache file due to key read error\n", __func__);
            break;
        }

        const std::streamoff payload_pos = in.tellg();
        if (payload_pos < 0) {
            GGML_LOG_DEBUG("%s: stop parsing cache file due to invalid payload position\n", __func__);
            break;
        }

        in.seekg((std::streamoff) entry_header.output_size, std::ios::cur);
        if (!in.good()) {
            GGML_LOG_DEBUG("%s: stop parsing cache file due to payload seek error\n", __func__);
            break;
        }

        cache_state.index[key] = ggml_rpp_weights_cache_index_entry{
            (uint64_t) payload_pos,
            entry_header.input_size,
            entry_header.output_size,
            entry_header.input_fingerprint,
        };
    }

    cache_state.enabled = true;
    GGML_LOG_INFO("%s: enabled RPP weight cache file: %s, entries: %zu\n", __func__, cache_state.file_path.c_str(),
                  cache_state.index.size());
}

static bool ggml_rpp_get_matmul_weight_converted_size(const ggml_tensor * tensor,
                                                      size_t              input_size,
                                                      size_t *            converted_size) {
    if (converted_size == nullptr || !is_matmul_weight(tensor)) {
        return false;
    }

    const int64_t n_experts = std::max<int64_t>(1, tensor->ne[2]);
    const int64_t K         = tensor->ne[0];
    const int64_t N         = tensor->ne[1];
    const size_t  Kp        = (size_t) (((uint64_t) K + 31u) & ~31u);
    const size_t  Np        = (size_t) (((uint64_t) N + 31u) & ~31u);
    if (K <= 0 || N <= 0 || n_experts <= 0) {
        return false;
    }
    const size_t n_experts_sz = (size_t) n_experts;

    switch (tensor->type) {
        case GGML_TYPE_BF16:
            {
                const size_t in_expert_sz  = (size_t) K * (size_t) N * sizeof(ggml_bf16_t);
                const size_t out_expert_sz = Kp * Np * sizeof(uint16_t);
                if (input_size != in_expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = out_expert_sz * n_experts_sz;
                return true;
            }
        case GGML_TYPE_F16:
            {
                const size_t in_expert_sz  = (size_t) K * (size_t) N * sizeof(ggml_fp16_t);
                const size_t out_expert_sz = Kp * Np * sizeof(uint16_t);
                if (input_size != in_expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = out_expert_sz * n_experts_sz;
                return true;
            }
        case GGML_TYPE_F32:
            {
                const size_t in_expert_sz  = (size_t) K * (size_t) N * sizeof(float);
                const size_t out_expert_sz = Kp * Np * sizeof(uint16_t);
                if (input_size != in_expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = out_expert_sz * n_experts_sz;
                return true;
            }
        case GGML_TYPE_IQ3_XXS:
            {
                const size_t codebook_sz  = (size_t) (K / 16) * 3u * (size_t) N * sizeof(uint16_t);
                const size_t scales_sz    = (size_t) (K / 128) * (size_t) N * sizeof(uint16_t);
                const size_t sign_sz      = (size_t) (K / 16) * (size_t) N * sizeof(uint16_t);
                const size_t super_sz     = (size_t) (K / QK_K) * (size_t) N * sizeof(bfloat16_u16);
                const size_t in_expert_sz = ggml_row_size(tensor->type, K) * (size_t) N;
                if (input_size != in_expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = (codebook_sz + scales_sz + sign_sz + super_sz) * n_experts_sz;
                return true;
            }
        case GGML_TYPE_Q4_1:
            {
                const size_t expert_sz = (size_t) (K / 4) * (size_t) N * sizeof(uint16_t) +
                                         (size_t) (K / 32) * (size_t) N * sizeof(bfloat16_u16) +
                                         (size_t) (K / 32) * (size_t) N * sizeof(bfloat16_u16);
                if (input_size != expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = expert_sz * n_experts_sz;
                return true;
            }
        case GGML_TYPE_Q8_0:
            {
                const size_t expert_sz = (size_t) (K / 2) * (size_t) N * sizeof(uint16_t) +
                                         (size_t) (K / 32) * (size_t) N * sizeof(bfloat16_u16);
                if (input_size != expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = expert_sz * n_experts_sz;
                return true;
            }
        case GGML_TYPE_Q4_K:
            {
                const int64_t ng = K / QK_K;
                const size_t  expert_sz =
                    (size_t) (K / 4) * (size_t) N * sizeof(uint16_t) +
                    (size_t) (2 * ng) * (size_t) N * sizeof(uint16_t) +
                    (size_t) (2 * ng) * (size_t) N * sizeof(uint16_t) + (size_t) ng * (size_t) N * sizeof(uint16_t) +
                    (size_t) ng * (size_t) N * sizeof(uint16_t) + (size_t) ng * (size_t) N * sizeof(bfloat16_u16) +
                    (size_t) ng * (size_t) N * sizeof(bfloat16_u16);
                if (input_size != expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = expert_sz * n_experts_sz;
                return true;
            }
        case GGML_TYPE_Q5_K:
            {
                const int64_t ng = K / QK_K;
                const size_t  expert_sz =
                    (size_t) (K / 4) * (size_t) N * sizeof(uint16_t) +
                    (size_t) (K / 16) * (size_t) N * sizeof(uint16_t) +
                    (size_t) (2 * ng) * (size_t) N * sizeof(uint16_t) +
                    (size_t) (2 * ng) * (size_t) N * sizeof(uint16_t) + (size_t) ng * (size_t) N * sizeof(uint16_t) +
                    (size_t) ng * (size_t) N * sizeof(uint16_t) + (size_t) ng * (size_t) N * sizeof(bfloat16_u16) +
                    (size_t) ng * (size_t) N * sizeof(bfloat16_u16);
                if (input_size != expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = expert_sz * n_experts_sz;
                return true;
            }
        case GGML_TYPE_Q6_K:
            {
                const size_t expert_sz = (size_t) (K / 4) * (size_t) N * sizeof(uint16_t) +
                                         (size_t) (K / 8) * (size_t) N * sizeof(uint16_t) +
                                         (size_t) (K / 32) * (size_t) N * sizeof(uint16_t) +
                                         (size_t) (K / QK_K) * (size_t) N * sizeof(bfloat16_u16);
                if (input_size != expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = expert_sz * n_experts_sz;
                return true;
            }
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_XS:
            {
                const size_t in_expert_sz  = ggml_row_size(tensor->type, K) * (size_t) N;
                const size_t out_expert_sz = (size_t) (K / 8) * (size_t) N * sizeof(uint16_t) +
                                             (size_t) (K / 64) * (size_t) N * sizeof(uint16_t) +
                                             (size_t) (K / 16) * (size_t) N * sizeof(uint16_t) +
                                             (size_t) (K / QK_K) * (size_t) N * sizeof(bfloat16_u16);
                if (input_size != in_expert_sz * n_experts_sz) {
                    return false;
                }
                *converted_size = out_expert_sz * n_experts_sz;
                return true;
            }
        default:
            GGML_LOG_ERROR("Unsupported tensor type: %d", tensor->type);
            return false;
    }
}

static bool ggml_rpp_weights_cache_try_load(const ggml_tensor * tensor,
                                            const void *        input_data,
                                            size_t              input_size,
                                            size_t              output_size,
                                            void **             out_data) {
    if (out_data == nullptr || output_size == 0 || output_size > (size_t) std::numeric_limits<std::streamsize>::max()) {
        return false;
    }

    ggml_rpp_weights_cache_state & cache_state = ggml_rpp_weights_cache_get_state();
    std::lock_guard<std::mutex>    lock(cache_state.mutex);
    ggml_rpp_weights_cache_init_locked(cache_state);

    if (!cache_state.enabled) {
        return false;
    }

    const std::string key         = ggml_rpp_weights_cache_make_key(tensor);
    const uint64_t    fingerprint = ggml_rpp_weights_cache_hash_bytes(input_data, input_size);

    auto it = cache_state.index.find(key);
    if (it == cache_state.index.end()) {
        return false;
    }

    const ggml_rpp_weights_cache_index_entry & entry = it->second;
    if (entry.input_size != input_size || entry.output_size != output_size || entry.input_fingerprint != fingerprint) {
        return false;
    }

    void * cached_data = nullptr;
    if (rtMallocHost(&cached_data, output_size) != rtSuccess || cached_data == nullptr) {
        return false;
    }

    std::ifstream in(cache_state.file_path, std::ios::binary);
    if (!in.is_open()) {
        rtFreeHost(cached_data);
        return false;
    }

    in.seekg((std::streamoff) entry.payload_offset, std::ios::beg);
    if (!in.good()) {
        rtFreeHost(cached_data);
        return false;
    }

    in.read((char *) cached_data, (std::streamsize) output_size);
    if (!in.good()) {
        rtFreeHost(cached_data);
        return false;
    }

    *out_data = cached_data;
    return true;
}

static void ggml_rpp_weights_cache_store(const ggml_tensor * tensor,
                                         const void *        input_data,
                                         size_t              input_size,
                                         const void *        output_data,
                                         size_t              output_size) {
    if (output_data == nullptr || output_size == 0 ||
        output_size > (size_t) std::numeric_limits<std::streamsize>::max()) {
        return;
    }

    ggml_rpp_weights_cache_state & cache_state = ggml_rpp_weights_cache_get_state();
    std::lock_guard<std::mutex>    lock(cache_state.mutex);
    ggml_rpp_weights_cache_init_locked(cache_state);

    if (!cache_state.enabled) {
        return;
    }

    const std::string key         = ggml_rpp_weights_cache_make_key(tensor);
    const uint64_t    fingerprint = ggml_rpp_weights_cache_hash_bytes(input_data, input_size);

    auto it = cache_state.index.find(key);
    if (it != cache_state.index.end()) {
        const ggml_rpp_weights_cache_index_entry & entry = it->second;
        if (entry.input_size == input_size && entry.output_size == output_size &&
            entry.input_fingerprint == fingerprint) {
            return;
        }
    }

    if (key.size() > (size_t) std::numeric_limits<uint32_t>::max()) {
        return;
    }

    std::ofstream out(cache_state.file_path, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        return;
    }

    const std::streamoff entry_pos = out.tellp();
    if (entry_pos < 0) {
        return;
    }

    ggml_rpp_weights_cache_entry_header entry_header = {};
    entry_header.key_size                            = (uint32_t) key.size();
    entry_header.input_size                          = input_size;
    entry_header.output_size                         = output_size;
    entry_header.input_fingerprint                   = fingerprint;

    out.write((const char *) &entry_header, sizeof(entry_header));
    out.write(key.data(), (std::streamsize) key.size());
    out.write((const char *) output_data, (std::streamsize) output_size);
    if (!out.good()) {
        return;
    }

    const uint64_t payload_offset = (uint64_t) entry_pos + sizeof(entry_header) + (uint64_t) key.size();
    cache_state.index[key]        = ggml_rpp_weights_cache_index_entry{
        payload_offset,
        input_size,
        output_size,
        fingerprint,
    };
}

}  // namespace

/**
 * @brief Initialize a tensor using data from a RPP buffer.
 *
 * This function initializes a tensor using data from a RPP buffer.
 * It handles special cases such as views and quantization.
 *
 * @param buffer The RPP buffer from which to initialize the tensor.
 * @param tensor Pointer to the tensor to be initialized.
 */
static enum ggml_status ggml_backend_rpp_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpp_buffer_context * ctx = (ggml_backend_rpp_buffer_context *) buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr &&
        ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size   = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_rpp_set_device(ctx->device);
            RPP_CHECK(rtMemset((char *) tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpp_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                  ggml_tensor *         tensor,
                                                  uint8_t               value,
                                                  size_t                offset,
                                                  size_t                size) {
    ggml_backend_rpp_buffer_context * ctx = (ggml_backend_rpp_buffer_context *) buffer->context;

    ggml_rpp_set_device(ctx->device);
    RPP_CHECK(rtMemsetAsync((char *) tensor->data + offset, value, size, rtStreamPerThread));
    RPP_CHECK(rtStreamSynchronize(rtStreamPerThread));
}

// TODO: need handle tensor which has paddings.
/**
 * @brief Set tensor data in a RPP buffer.
 *
 * This function sets tensor data in a RPP buffer, handling transformations
 * if needed based on the tensor's type.
 *
 * @param buffer The RPP buffer where the tensor data will be set.
 * @param tensor Pointer to the tensor whose data will be set.
 * @param data Pointer to the source data to be copied into the tensor.
 * @param offset Offset in the source data from where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
static void ggml_backend_rpp_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor *         tensor,
                                               const void *          data,
                                               size_t                offset,
                                               size_t                size) {
    ggml_backend_rpp_buffer_context * ctx = (ggml_backend_rpp_buffer_context *) buffer->context;
    ggml_rpp_set_device(ctx->device);
    void *        tmp              = nullptr;
    size_t        copy_size        = size;
    const int64_t n_experts        = std::max<int64_t>(1, tensor->ne[2]);
    size_t        cached_copy_size = 0;
    const bool    cache_eligible =
        offset == 0 && ggml_rpp_get_matmul_weight_converted_size(tensor, size, &cached_copy_size);
    bool cache_hit = false;

    if (cache_eligible) {
        cache_hit = ggml_rpp_weights_cache_try_load(tensor, data, size, cached_copy_size, &tmp);
        if (cache_hit) {
            copy_size = cached_copy_size;
        }
    }

    if (!cache_hit && is_matmul_weight(tensor)) {
        switch (tensor->type) {
            case GGML_TYPE_F32:
                {
                    const int    K             = tensor->ne[0];
                    const int    N             = tensor->ne[1];
                    const size_t Kp            = (size_t) (((uint64_t) K + 31u) & ~31u);
                    const size_t Np            = (size_t) (((uint64_t) N + 31u) & ~31u);
                    const size_t in_expert_sz  = (size_t) K * (size_t) N * sizeof(float);
                    const size_t out_expert_sz = Kp * Np * sizeof(uint16_t);
                    GGML_ASSERT(size == in_expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, out_expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * in_expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * out_expert_sz;
                        convert_fp32_to_rpp_bf16((const float *) src_base, K, N, dst_base);
                    }
                    copy_size = out_expert_sz * (size_t) n_experts;
                    break;
                }
            case GGML_TYPE_BF16:
                {
                    const int    K             = tensor->ne[0];
                    const int    N             = tensor->ne[1];
                    const size_t Kp            = (size_t) (((uint64_t) K + 31u) & ~31u);
                    const size_t Np            = (size_t) (((uint64_t) N + 31u) & ~31u);
                    const size_t in_expert_sz  = (size_t) K * (size_t) N * sizeof(ggml_bf16_t);
                    const size_t out_expert_sz = Kp * Np * sizeof(uint16_t);
                    GGML_ASSERT(size == in_expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, out_expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char *        src_base = (const char *) data + (size_t) ie * in_expert_sz;
                        char *              dst_base = (char *) tmp + (size_t) ie * out_expert_sz;
                        const ggml_bf16_t * src_bf16 = (const ggml_bf16_t *) src_base;
                        convert_bf16_to_rpp_bf16(src_bf16, K, N, dst_base);
                    }
                    copy_size = out_expert_sz * (size_t) n_experts;
                    break;
                }
            case GGML_TYPE_F16:
                {
                    const int    K             = tensor->ne[0];
                    const int    N             = tensor->ne[1];
                    const size_t Kp            = (size_t) (((uint64_t) K + 31u) & ~31u);
                    const size_t Np            = (size_t) (((uint64_t) N + 31u) & ~31u);
                    const size_t in_expert_sz  = (size_t) K * (size_t) N * sizeof(ggml_fp16_t);
                    const size_t out_expert_sz = Kp * Np * sizeof(uint16_t);
                    GGML_ASSERT(size == in_expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, out_expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char *        src_base = (const char *) data + (size_t) ie * in_expert_sz;
                        char *              dst_base = (char *) tmp + (size_t) ie * out_expert_sz;
                        const ggml_fp16_t * src_f16  = (const ggml_fp16_t *) src_base;
                        convert_f16_to_rpp_bf16(src_f16, K, N, dst_base);
                    }
                    copy_size = out_expert_sz * (size_t) n_experts;

                    break;
                }
            case GGML_TYPE_Q8_0:
                {
                    const int    K            = tensor->ne[0];
                    const int    N            = tensor->ne[1];
                    const size_t weights_size = (size_t) (K / 2) * (size_t) N * sizeof(uint16_t);
                    const size_t scales_size  = (size_t) (K / 32) * (size_t) N * sizeof(bfloat16_u16);
                    const size_t expert_sz    = weights_size + scales_size;
                    GGML_ASSERT(size == expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * expert_sz;
                        convert_q8_0_to_rpp((block_q8_0_dbg *) src_base, K, N, dst_base);
                    }
                    copy_size = expert_sz * (size_t) n_experts;
                    break;
                }
            case GGML_TYPE_Q6_K:
                {
                    const int    K                 = tensor->ne[0];
                    const int    N                 = tensor->ne[1];
                    size_t       weights_ql_size   = K / 4 * N * sizeof(uint16_t);
                    size_t       weights_qh_size   = K / 8 * N * sizeof(uint16_t);
                    size_t       scales_size       = K / 32 * N * sizeof(uint16_t);
                    size_t       super_scales_size = K / QK_K * N * sizeof(bfloat16_u16);
                    const size_t expert_sz = weights_ql_size + weights_qh_size + scales_size + super_scales_size;
                    GGML_ASSERT(size == expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * expert_sz;
                        convert_q6_k_to_rpp((block_q6_k_dbg *) src_base, K, N, dst_base);
                    }
                    copy_size = expert_sz * (size_t) n_experts;
                    break;
                }
            case GGML_TYPE_Q5_K:
                {
                    const int    K                = tensor->ne[0];
                    const int    N                = tensor->ne[1];
                    const int    ng               = K / QK_K;
                    const size_t weights_lsb_size = (size_t) (K / 4) * (size_t) N * sizeof(uint16_t);
                    const size_t weights_msb_size = (size_t) (K / 16) * (size_t) N * sizeof(uint16_t);
                    const size_t scale_lsb_size   = (size_t) (2 * ng) * (size_t) N * sizeof(uint16_t);
                    const size_t zero_lsb_size    = (size_t) (2 * ng) * (size_t) N * sizeof(uint16_t);
                    const size_t scale_msb_size   = (size_t) ng * (size_t) N * sizeof(uint16_t);
                    const size_t zero_msb_size    = (size_t) ng * (size_t) N * sizeof(uint16_t);
                    const size_t super_scale_size = (size_t) ng * (size_t) N * sizeof(bfloat16_u16);
                    const size_t super_zero_size  = (size_t) ng * (size_t) N * sizeof(bfloat16_u16);
                    const size_t expert_sz = weights_lsb_size + weights_msb_size + scale_lsb_size + zero_lsb_size +
                                             scale_msb_size + zero_msb_size + super_scale_size + super_zero_size;
                    GGML_ASSERT(size == expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * expert_sz;
                        convert_q5_k_to_rpp((const block_q5_k_dbg *) src_base, K, N, dst_base);
                    }
                    copy_size = expert_sz * (size_t) n_experts;
                    break;
                }

            case GGML_TYPE_Q4_1:
                {
                    const int    K            = tensor->ne[0];
                    const int    N            = tensor->ne[1];
                    const size_t weights_size = (size_t) (K / 4) * (size_t) N * sizeof(uint16_t);
                    const size_t scales_size  = (size_t) (K / 32) * (size_t) N * sizeof(bfloat16_u16);
                    const size_t zeros_size   = (size_t) (K / 32) * (size_t) N * sizeof(bfloat16_u16);
                    const size_t expert_sz    = weights_size + scales_size + zeros_size;
                    GGML_ASSERT(size == expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * expert_sz;
                        convert_q4_1_to_rpp((block_q4_1_dbg *) src_base, K, N, dst_base);
                    }
                    copy_size = expert_sz * (size_t) n_experts;
                    break;
                }
            case GGML_TYPE_Q4_K:
                {
                    const int    K                = tensor->ne[0];
                    const int    N                = tensor->ne[1];
                    const int    ng               = K / QK_K;
                    const size_t weights_size     = (size_t) (K / 4) * (size_t) N * sizeof(uint16_t);
                    const size_t scale_lsb_size   = (size_t) (2 * ng) * (size_t) N * sizeof(uint16_t);
                    const size_t zero_lsb_size    = (size_t) (2 * ng) * (size_t) N * sizeof(uint16_t);
                    const size_t scale_msb_size   = (size_t) ng * (size_t) N * sizeof(uint16_t);
                    const size_t zero_msb_size    = (size_t) ng * (size_t) N * sizeof(uint16_t);
                    const size_t super_scale_size = (size_t) ng * (size_t) N * sizeof(bfloat16_u16);
                    const size_t super_zero_size  = (size_t) ng * (size_t) N * sizeof(bfloat16_u16);
                    const size_t expert_sz        = weights_size + scale_lsb_size + zero_lsb_size + scale_msb_size +
                                             zero_msb_size + super_scale_size + super_zero_size;
                    GGML_ASSERT(size == expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * expert_sz;
                        convert_q4_k_to_rpp((const block_q4_k_dbg *) src_base, K, N, dst_base);
                    }
                    copy_size = expert_sz * (size_t) n_experts;
                    break;
                }
            case GGML_TYPE_IQ3_XXS:
                {
                    const int K = tensor->ne[0];
                    const int N = tensor->ne[1];
                    // const size_t codebook_sz   = (size_t) (K / 8) * (size_t) N * sizeof(uint16_t);
                    // const size_t scales_sz     = (size_t) (K / 128) * (size_t) N * sizeof(uint16_t);
                    // const size_t sign_sz       = (size_t) (K / 16) * (size_t) N * sizeof(uint16_t);
                    // const size_t super_sz      = (size_t) (K / QK_K) * (size_t) N * sizeof(bfloat16_u16);
                    // const size_t in_expert_sz  = ggml_row_size(tensor->type, K) * (size_t) N;
                    // const size_t out_expert_sz = codebook_sz + scales_sz + sign_sz + super_sz;
                    // GGML_ASSERT(size == in_expert_sz * (size_t) n_experts);
                    // rtMallocHost(&tmp, out_expert_sz * (size_t) n_experts);
                    // for (int64_t ie = 0; ie < n_experts; ++ie) {
                    //     const char * src_base = (const char *) data + (size_t) ie * in_expert_sz;
                    //     char *       dst_base = (char *) tmp + (size_t) ie * out_expert_sz;
                    //     convert_iq3_xxs_to_rpp((const block_iq3_xxs_dbg *) src_base, K, N, dst_base);
                    // }
                    // copy_size = out_expert_sz * (size_t) n_experts;

                    const size_t codebook_sz   = (size_t) (K / 16) * 3u * (size_t) N * sizeof(uint16_t);
                    const size_t scales_sz     = (size_t) (K / 128) * (size_t) N * sizeof(uint16_t);
                    const size_t sign_sz       = (size_t) (K / 16) * (size_t) N * sizeof(uint16_t);
                    const size_t super_sz      = (size_t) (K / QK_K) * (size_t) N * sizeof(bfloat16_u16);
                    const size_t in_expert_sz  = ggml_row_size(tensor->type, K) * (size_t) N;
                    const size_t out_expert_sz = codebook_sz + scales_sz + sign_sz + super_sz;
                    GGML_ASSERT(size == in_expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, out_expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * in_expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * out_expert_sz;
                        convert_iq3_xxs_to_nolut((const block_iq3_xxs_dbg *) src_base, K, N, dst_base);
                    }
                    copy_size = out_expert_sz * (size_t) n_experts;
                    break;
                }
            case GGML_TYPE_IQ2_S:
                {
                    const int K = tensor->ne[0];
                    const int N = tensor->ne[1];

                    // const size_t q2_codebook_lsb_size = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
                    // const size_t q2_codebook_msb_size = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
                    // const size_t q2_scales_size       = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
                    // const size_t q2_sign_size         = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
                    // const size_t q2_super_scale_size  = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
                    // const size_t q2_expert_size =
                    //     q2_codebook_lsb_size + q2_codebook_msb_size + q2_scales_size + q2_sign_size + q2_super_scale_size;
                    // GGML_ASSERT(size == q2_expert_size * (size_t) n_experts);
                    // rtMallocHost(&tmp, q2_expert_size * (size_t) n_experts);
                    // for (int64_t ie = 0; ie < n_experts; ++ie) {
                    //     const char * src_base = (const char *) data + (size_t) ie * q2_expert_size;
                    //     char *       dst_base = (char *) tmp + (size_t) ie * q2_expert_size;
                    //     convert_iq2_s_to_rpp((const block_iq2_s_dbg *) src_base, K, N, dst_base);
                    // }
                    // copy_size = q2_expert_size * (size_t) n_experts;

                    const size_t q2_codebook_nolut_size = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
                    const size_t q2_scales_size         = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
                    const size_t q2_sign_size           = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
                    const size_t q2_super_scale_size    = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
                    const size_t in_expert_sz           = ggml_row_size(tensor->type, K) * (size_t) N;
                    const size_t out_expert_sz =
                        q2_codebook_nolut_size + q2_scales_size + q2_sign_size + q2_super_scale_size;
                    GGML_ASSERT(size == in_expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, out_expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * in_expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * out_expert_sz;
                        convert_iq2_s_to_nolut((const block_iq2_s_dbg *) src_base, K, N, dst_base);
                    }
                    copy_size = out_expert_sz * (size_t) n_experts;
                    break;
                }
            case GGML_TYPE_IQ2_XS:
                {
                    const int K = tensor->ne[0];
                    const int N = tensor->ne[1];

                    // const int    ng                  = K / QK_K;
                    // const size_t q2_qs_size          = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
                    // const size_t q2_scales_size      = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
                    // const size_t q2_super_scale_size = (size_t) ng * (size_t) N * sizeof(bfloat16_u16);
                    // const size_t in_expert_sz        = ggml_row_size(tensor->type, K) * (size_t) N;
                    // const size_t out_expert_sz       = q2_qs_size + q2_scales_size + q2_super_scale_size;
                    // GGML_ASSERT(in_expert_sz == out_expert_sz);
                    // GGML_ASSERT(size == in_expert_sz * (size_t) n_experts);
                    // rtMallocHost(&tmp, out_expert_sz * (size_t) n_experts);
                    // for (int64_t ie = 0; ie < n_experts; ++ie) {
                    //     const char * src_base = (const char *) data + (size_t) ie * in_expert_sz;
                    //     char *       dst_base = (char *) tmp + (size_t) ie * out_expert_sz;
                    //     convert_iq2_xs_to_rpp((const block_iq2_xs_dbg *) src_base, K, N, dst_base);
                    // }
                    // copy_size = out_expert_sz * (size_t) n_experts;

                    const size_t q2_codebook_nolut_size = (size_t) K / 8 * (size_t) N * sizeof(uint16_t);
                    const size_t q2_scales_size         = (size_t) K / 64 * (size_t) N * sizeof(uint16_t);
                    const size_t q2_sign_size           = (size_t) K / 16 * (size_t) N * sizeof(uint16_t);
                    const size_t q2_super_scale_size    = (size_t) K / QK_K * (size_t) N * sizeof(bfloat16_u16);
                    const size_t in_expert_sz           = ggml_row_size(tensor->type, K) * (size_t) N;
                    const size_t out_expert_sz =
                        q2_codebook_nolut_size + q2_scales_size + q2_sign_size + q2_super_scale_size;
                    GGML_ASSERT(size == in_expert_sz * (size_t) n_experts);
                    rtMallocHost(&tmp, out_expert_sz * (size_t) n_experts);
                    for (int64_t ie = 0; ie < n_experts; ++ie) {
                        const char * src_base = (const char *) data + (size_t) ie * in_expert_sz;
                        char *       dst_base = (char *) tmp + (size_t) ie * out_expert_sz;
                        convert_iq2_xs_to_nolut((const block_iq2_xs_dbg *) src_base, K, N, dst_base);
                    }
                    copy_size = out_expert_sz * (size_t) n_experts;
                    break;
                }
            default:
                GGML_LOG_ERROR("ggml_backend_rpp_buffer_set_tensor for mul_mat: unsupported tensor type: %d\n",
                               int(tensor->type));
                GGML_ASSERT(false);
                return;
        }
    }
    const void * ddr_data = tmp == nullptr ? data : tmp;
    RPP_CHECK(
        rtMemcpyAsync((char *) tensor->data + offset, ddr_data, copy_size, rtMemcpyHostToDevice, rtStreamPerThread));
    RPP_CHECK(rtStreamSynchronize(rtStreamPerThread));
    if (!cache_hit && cache_eligible && tmp != nullptr && copy_size == cached_copy_size) {
        ggml_rpp_weights_cache_store(tensor, data, size, tmp, copy_size);
    }
    if (tmp) {
        rtFreeHost(tmp);
    }
}

/**
 * @brief Get tensor data from a RPP buffer.
 *
 * This function retrieves tensor data from a RPP buffer, handling
 * transformations if needed based on the tensor's type.
 *
 * @param buffer The RPP buffer from which to retrieve tensor data.
 * @param tensor Pointer to the tensor whose data will be retrieved.
 * @param data Pointer to the destination buffer where the tensor data will be
 * copied.
 * @param offset Offset in the destination buffer where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
static void ggml_backend_rpp_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor *   tensor,
                                               void *                data,
                                               size_t                offset,
                                               size_t                size) {
    ggml_backend_rpp_buffer_context * ctx = (ggml_backend_rpp_buffer_context *) buffer->context;

    ggml_rpp_set_device(ctx->device);
    RPP_CHECK(rtMemcpyAsync(data, (const char *) tensor->data + offset, size, rtMemcpyDeviceToHost, rtStreamPerThread));
    RPP_CHECK(rtStreamSynchronize(rtStreamPerThread));
}

/**
 * @brief Copy tensor data between RPP buffers if possible.
 *
 * This function copies tensor data between RPP buffers if the source and
 * destination buffers are RPP buffers and they meet the necessary conditions
 * (same device or devices can access each other).
 *
 * @param buffer The destination RPP buffer where the tensor data will be
 * copied.
 * @param src Pointer to the source tensor whose data will be copied.
 * @param dst Pointer to the destination tensor where the data will be copied.
 * @return true if the copy operation succeeded, false otherwise.
 */
static bool ggml_backend_rpp_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor *   src,
                                               ggml_tensor *         dst) {
    if (ggml_backend_buffer_is_rpp(src->buffer)) {
        ggml_backend_rpp_buffer_context * src_ctx = (ggml_backend_rpp_buffer_context *) src->buffer->context;
        ggml_backend_rpp_buffer_context * dst_ctx = (ggml_backend_rpp_buffer_context *) dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            RPP_CHECK(rtMemcpyAsync(dst->data, src->data, ggml_nbytes(src), rtMemcpyDeviceToDevice, rtStreamPerThread));
        } else {
            return false;
        }
        RPP_CHECK(rtStreamSynchronize(rtStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

/**
 * @brief Clear a RPP buffer by setting all its memory to a specified value.
 *
 * This function clears a RPP buffer by setting all its memory to a specified
 * value.
 *
 * @param buffer The RPP buffer to be cleared.
 * @param value The value to which each byte in the buffer will be set.
 */
static void ggml_backend_rpp_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpp_buffer_context * ctx = (ggml_backend_rpp_buffer_context *) buffer->context;

    ggml_rpp_set_device(ctx->device);
    RPP_CHECK(rtMemsetAsync(ctx->dev_ptr, value, buffer->size, rtStreamPerThread));
    RPP_CHECK(rtStreamSynchronize(rtStreamPerThread));
}

/**
 * @brief Interface for a RPP buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a RPP buffer within the backend.
 */
static const ggml_backend_buffer_i ggml_backend_rpp_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpp_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpp_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpp_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpp_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpp_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpp_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpp_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpp_buffer_clear,
    /* .reset           = */ NULL,
};

// rpp buffer type
struct ggml_backend_rpp_buffer_type_context {
    int         device;
    std::string name;
};

/**
 * @brief Retrieves the name associated with a RPP buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * RPP buffer type context.
 *
 * @param buft Pointer to the buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char * ggml_backend_rpp_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpp_buffer_type_context * ctx = (ggml_backend_rpp_buffer_type_context *) buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_rpp(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_rpp_buffer_type_get_name;
}

/**
 * @brief Allocates a new RPP buffer of the specified type and size.
 *
 * This function allocates a new RPP buffer on the specified device with the
 * given size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
static ggml_backend_buffer_t ggml_backend_rpp_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpp_buffer_type_context * buft_ctx = (ggml_backend_rpp_buffer_type_context *) buft->context;

    ggml_rpp_set_device(buft_ctx->device);

    void *    dev_ptr;
    rtError_t err = ggml_rpp_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != rtSuccess) {
        // clear the error
        (void) rtGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: rppMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0,
                       buft_ctx->device, rtGetErrorString(err));
        return nullptr;
    }

    ggml_backend_rpp_buffer_context * ctx = new ggml_backend_rpp_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_rpp_buffer_interface, ctx, size);
}

/**
 * @brief Retrieves the memory alignment requirement for RPP buffers of this
 * type.
 *
 * This function returns the alignment requirement in bytes for memory allocated
 * by the RPP buffer type.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @return The alignment requirement in bytes (fixed at 128 bytes for RPP
 * buffers).
 */
static size_t ggml_backend_rpp_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

/**
 * @brief Calculates the allocation size required for a tensor in a RPP buffer.
 *
 * Computes the total allocation size needed for storing the tensor's data in a
 * RPP buffer, considering any necessary padding or adjustments for quantized
 * types.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @param tensor Pointer to the tensor for which the allocation size is
 * calculated.
 * @return The total allocation size in bytes required for the tensor in the
 * RPP buffer.
 */
static size_t ggml_backend_rpp_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t  size           = ggml_nbytes(tensor);
    int64_t ne0            = tensor->ne[0];
    size_t  converted_size = 0;

    if (ggml_rpp_get_matmul_weight_converted_size(tensor, size, &converted_size)) {
        // ggml backends are required to report an allocation size that is at
        // least the logical tensor byte size. Some RPP weight conversions
        // reduce storage (for example F32/F16/BF16 weights converted to BF16
        // tiles), so returning the converted size directly violates the
        // backend contract and trips the generic assert in
        // ggml_backend_buft_get_alloc_size().
        return std::max(size, converted_size);
    }
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_rpp_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpp_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_rpp_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpp_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_rpp_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

/**
 * @brief Retrieves the RPP buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
ggml_backend_buffer_type_t ggml_backend_rpp_buffer_type(int device) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_rpp_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_rpp_buffer_types[GGML_RPP_MAX_DEVICES];

    static bool ggml_backend_rpp_buffer_type_initialized = false;

    if (!ggml_backend_rpp_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_rpp_get_device_count(); i++) {
            ggml_backend_rpp_buffer_types[i] = {
                /* .iface    = */ ggml_backend_rpp_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_rpp_reg(), i),
                /* .context  = */ new ggml_backend_rpp_buffer_type_context{ i, GGML_RPP_NAME + std::to_string(i) },
            };
        }
        ggml_backend_rpp_buffer_type_initialized = true;
    }

    return &ggml_backend_rpp_buffer_types[device];
}

// rpp split buffer

static int64_t get_row_rounding(const std::array<float, GGML_RPP_MAX_DEVICES> & tensor_split) {
    int64_t row_rounding = 0;
    for (int id = 0; id < ggml_backend_rpp_get_device_count(); ++id) {
        if (tensor_split[id] >= (id + 1 < ggml_backend_rpp_get_device_count() ? tensor_split[id + 1] : 1.0f)) {
            continue;
        }

        const int cc = ggml_rpp_info().devices[id].cc;

        ///< rpp not support, disable code
        // row_rounding = std::max(row_rounding, (int64_t)get_mmq_y_host(cc));
    }
    return row_rounding;
}

static void get_row_split(int64_t *                                       row_low,
                          int64_t *                                       row_high,
                          const ggml_tensor *                             tensor,
                          const std::array<float, GGML_RPP_MAX_DEVICES> & tensor_split,
                          int                                             id) {
    const int64_t nrows    = ggml_nrows(tensor);
    const int64_t rounding = get_row_rounding(tensor_split);

    *row_low = id == 0 ? 0 : nrows * tensor_split[id];
    *row_low -= *row_low % rounding;

    if (id == ggml_backend_rpp_get_device_count() - 1) {
        *row_high = nrows;
    } else {
        *row_high = nrows * tensor_split[id + 1];
        *row_high -= *row_high % rounding;
    }
}

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split * ggml_row_size(tensor->type, tensor->ne[0]);
}

struct ggml_backend_rpp_split_buffer_type_context {
    int                                     main_device;
    std::array<float, GGML_RPP_MAX_DEVICES> tensor_split;
    std::string                             name;
};

struct ggml_backend_rpp_split_buffer_context {
    ~ggml_backend_rpp_split_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            for (int id = 0; id < GGML_RPP_MAX_DEVICES; ++id) {
                for (int64_t is = 0; is < GGML_RPP_MAX_STREAMS; ++is) {
                    if (extra->events[id][is] != nullptr) {
                        RPP_CHECK(rtEventDestroy(extra->events[id][is]));
                    }
                }
                if (extra->data_device[id] != nullptr) {
                    RPP_CHECK(rtFree(extra->data_device[id]));
                }
            }
            delete extra;
        }
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};

static void ggml_backend_rpp_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpp_split_buffer_context * ctx = (ggml_backend_rpp_split_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_rpp_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *) 0x1000;

    GGML_UNUSED(buffer);
}

static enum ggml_status ggml_backend_rpp_split_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr);  // views of split tensors are not supported
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_rpp_split_buffer_context *      ctx = (ggml_backend_rpp_split_buffer_context *) buffer->context;
    ggml_backend_rpp_split_buffer_type_context * buft_ctx =
        (ggml_backend_rpp_split_buffer_type_context *) buffer->buft->context;

    const int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);

    for (int id = 0; id < ggml_backend_rpp_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        size_t       size          = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        // FIXME: do not crash if rtMalloc fails
        // currently, init_tensor cannot fail, it needs to be fixed in ggml-backend first
        ggml_rpp_set_device(id);
        char * buf;
        RPP_CHECK(ggml_rpp_device_malloc((void **) &buf, size, id));

        // set padding to 0 to avoid possible NaN values
        if (size > original_size) {
            RPP_CHECK(rtMemset(buf + original_size, 0, size - original_size));
        }

        extra->data_device[id] = buf;

        for (int64_t is = 0; is < GGML_RPP_MAX_STREAMS; ++is) {
            RPP_CHECK(rtEventCreateWithFlags(&extra->events[id][is], rtEventDisableTiming));
        }
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpp_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                     ggml_tensor *         tensor,
                                                     const void *          data,
                                                     size_t                offset,
                                                     size_t                size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_rpp_split_buffer_type_context * buft_ctx =
        (ggml_backend_rpp_split_buffer_type_context *) buffer->buft->context;

    const int64_t           ne0   = tensor->ne[0];
    const size_t            nb1   = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int id = 0; id < ggml_backend_rpp_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split  = row_low * nb1;
        size_t       size          = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        const char * buf_host = (const char *) data + offset_split;
        RPP_CHECK(
            rtMemcpyAsync(extra->data_device[id], buf_host, original_size, rtMemcpyHostToDevice, rtStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_rpp_get_device_count(); ++id) {
        RPP_CHECK(rtStreamSynchronize(rtStreamPerThread));
    }
}

static void ggml_backend_rpp_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                     const ggml_tensor *   tensor,
                                                     void *                data,
                                                     size_t                offset,
                                                     size_t                size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    ggml_backend_rpp_split_buffer_type_context * buft_ctx =
        (ggml_backend_rpp_split_buffer_type_context *) buffer->buft->context;

    const int64_t           ne0   = tensor->ne[0];
    const size_t            nb1   = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    for (int id = 0; id < ggml_backend_rpp_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, buft_ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        const size_t offset_split  = row_low * nb1;
        size_t       size          = ggml_nbytes_split(tensor, nrows_split);
        const size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        char * buf_host = (char *) data + offset_split;
        RPP_CHECK(
            rtMemcpyAsync(buf_host, extra->data_device[id], original_size, rtMemcpyDeviceToHost, rtStreamPerThread));
    }

    for (int id = 0; id < ggml_backend_rpp_get_device_count(); ++id) {
        RPP_CHECK(rtStreamSynchronize(rtStreamPerThread));
    }
}

static void ggml_backend_rpp_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i ggml_backend_rpp_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpp_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpp_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpp_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpp_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpp_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_rpp_split_buffer_clear,
    /* .reset           = */ NULL,
};

// rpp split buffer type

static const char * ggml_backend_rpp_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpp_split_buffer_type_context * ctx = (ggml_backend_rpp_split_buffer_type_context *) buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_rpp_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_rpp_split_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_rpp_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                             size_t                     size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_rpp_split_buffer_context * ctx = new ggml_backend_rpp_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_rpp_split_buffer_interface, ctx, size);
}

static size_t ggml_backend_rpp_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_rpp_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                                const ggml_tensor *        tensor) {
    ggml_backend_rpp_split_buffer_type_context * ctx = (ggml_backend_rpp_split_buffer_type_context *) buft->context;
    GGML_ASSERT(ggml_is_contiguous(tensor) && "split buffers only supported for contiguous tensors");

    size_t total_size = 0;

    const int64_t ne0 = tensor->ne[0];

    for (int id = 0; id < ggml_backend_rpp_get_device_count(); ++id) {
        int64_t row_low, row_high;
        get_row_split(&row_low, &row_high, tensor, ctx->tensor_split, id);

        int64_t nrows_split = row_high - row_low;
        if (nrows_split == 0) {
            continue;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return total_size;
}

static bool ggml_backend_rpp_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_rpp_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpp_split_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_rpp_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpp_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_rpp_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_rpp_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_rpp_split_buffer_type(int main_device, const float * tensor_split) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::map<std::pair<int, std::array<float, GGML_RPP_MAX_DEVICES>>, struct ggml_backend_buffer_type> buft_map;

    std::array<float, GGML_RPP_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr ||
                    std::all_of(tensor_split, tensor_split + GGML_RPP_MAX_DEVICES, [](float x) { return x == 0.0f; });
    if (all_zero) {
        tensor_split_arr = ggml_rpp_info().default_tensor_split;
    } else {
        float split_sum = 0.0f;
        for (int i = 0; i < ggml_backend_rpp_get_device_count(); ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        for (int i = 0; i < ggml_backend_rpp_get_device_count(); ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find({ main_device, tensor_split_arr });
    if (it != buft_map.end()) {
        return &it->second;
    }
    auto * ctx = new ggml_backend_rpp_split_buffer_type_context{
        main_device,
        tensor_split_arr,
        GGML_RPP_NAME + std::to_string(main_device) + "_Split",
    };

    struct ggml_backend_buffer_type buft{
        /* .iface   = */ ggml_backend_rpp_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_rpp_reg(), main_device),
        /* .context = */ ctx,
    };

    auto result = buft_map.emplace(std::make_pair(main_device, tensor_split_arr), buft);
    return &result.first->second;
}

// host buffer type

/**
 * @brief Retrieves the name associated with a RPP host buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * RPP host buffer type context.
 *
 * @param buft Pointer to the host buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char * ggml_backend_rpp_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_RPP_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_rpp_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_rpp_host_buffer_type_name;
}

static void ggml_backend_rpp_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    RPP_CHECK(rtFreeHost(buffer->context));
}

static void * ggml_rpp_host_malloc(size_t size) {
    if (getenv("GGML_RPP_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void *    ptr = nullptr;
    rtError_t err = rtMallocHost((void **) &ptr, size);
    if (err != rtSuccess) {
        // clear the error
        (void) rtGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                       rtGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

/**
 * @brief Allocates a new RPP host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
static ggml_backend_buffer_t ggml_backend_rpp_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t                     size) {
    void * ptr = ggml_rpp_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_rpp_host_buffer_free_buffer;

    return buffer;
}

/**
 * @brief Interface for managing RPP host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for RPP buffer types in the GGML backend.
 */
ggml_backend_buffer_type_t ggml_backend_rpp_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_rpp_buffer_type_host = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_rpp_host_buffer_type_name,
                           /* .alloc_buffer     = */ ggml_backend_rpp_host_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                           /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                           /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_rpp_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_rpp_buffer_type_host;
}

//static bool ggml_backend_buffer_is_rpp_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_rpp_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_rpp_op_mul_mat_t)(ggml_backend_rpp_context & ctx,
                                      const ggml_tensor *        src0,
                                      const ggml_tensor *        src1,
                                      ggml_tensor *              dst,
                                      const char *               src0_dd_i,
                                      const float *              src1_ddf_i,
                                      const char *               src1_ddq_i,
                                      float *                    dst_dd_i,
                                      const int64_t              row_low,
                                      const int64_t              row_high,
                                      const int64_t              src1_ncols,
                                      const int64_t              src1_padded_row_size,
                                      rtStream_t                 stream);

#define MUL_MAT_SRC1_COL_STRIDE 128

// this function is not use
static rtError_t ggml_rpp_cpy_tensor_2d(void *                     dst,
                                        const struct ggml_tensor * src,
                                        int64_t                    i3,
                                        int64_t                    i2,
                                        int64_t                    i1_low,
                                        int64_t                    i1_high,
                                        rtStream_t                 stream) {
    const char * src_ptr = (const char *) src->data;
    char *       dst_ptr = (char *) dst;

    const int64_t        ne0     = src->ne[0];
    const int64_t        nb0     = src->nb[0];
    const int64_t        nb1     = src->nb[1];
    const int64_t        nb2     = src->nb[2];
    const int64_t        nb3     = src->nb[3];
    const enum ggml_type type    = src->type;
    const int64_t        ts      = ggml_type_size(type);
    const int64_t        bs      = ggml_blck_size(type);
    const int64_t        i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low * nb1 + i2 * nb2 + i3 * nb3;
    if (nb0 == ts && nb1 == ts * ne0 / bs) {
        return rtMemcpyAsync(dst_ptr, x, i1_diff * nb1, rtMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return rtMemcpy2DAsync(dst_ptr, ts * ne0 / bs, x, nb1, ts * ne0 / bs, i1_diff, rtMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1 * nb1);
            void *       rd = (void *) (dst_ptr + i1 * ts * ne0 / bs);
            // pretend the row is a matrix with cols=1
            rtError_t    r  = rtMemcpy2DAsync(rd, ts / bs, rx, nb0, ts / bs, ne0, rtMemcpyDeviceToDevice, stream);
            if (r != rtSuccess) {
                return r;
            }
        }
        return rtSuccess;
    }
}

static bool ggml_rpp_compute_forward(ggml_backend_rpp_context & ctx,
                                     struct ggml_tensor *       dst,
                                     int                        is_instantial = 1,
                                     int                        is_launch     = 1) {
    // why is this here instead of mul_mat?
    // if (dst->src[0] != nullptr && ggml_backend_buft_is_rpp_split(dst->src[0]->buffer->buft)) {
    //     ggml_rpp_set_peer_access(dst->src[1]->ne[1], ctx.device);
    // }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            break;
        case GGML_OP_COUNT_EQUAL:
            break;
        case GGML_OP_REPEAT:
            break;
        case GGML_OP_REPEAT_BACK:
            break;
        case GGML_OP_GET_ROWS:
            ggml_rpp_op_get_rows(ctx, dst, is_instantial, is_launch);
            // {
            //     char info[256];
            //     double time = measure_time_mics(ggml_rpp_op_get_rows, ctx, dst);
            //     ggml_rpp_get_ops_info(ctx.cur_rpp_graph->ggml_graph, dst, info);
            //     printf("%s", info);
            //     printf("    xxxxxxxxxxxxxxxxxxxxxx take time: %.3f (ms)\n\n", time);
            // }
            break;
        case GGML_OP_GET_ROWS_BACK:
            break;
        case GGML_OP_SET_ROWS:
            ggml_rpp_op_set_rows(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_DUP:
            break;
        case GGML_OP_CPY:
            ggml_rpp_op_cpy(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_CONT:
            ggml_rpp_op_cont(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_ADD:
            ggml_rpp_op_add(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_ADD1:  // TODO: more efficient implementation
            break;
        case GGML_OP_ADD_ID:
            break;
        case GGML_OP_SUB:
            break;
        case GGML_OP_ACC:
            break;
        case GGML_OP_MUL:
            ggml_rpp_op_mul(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_DIV:
            ggml_rpp_op_div(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_ELU:
                    return false;
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_rpp_op_unary(ctx, dst, is_instantial, is_launch);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            ggml_rpp_op_glu(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_NORM:
            ggml_rpp_op_norm(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_GROUP_NORM:
            break;
        case GGML_OP_L2_NORM:
            ggml_rpp_op_l2_norm(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_CONCAT:
            break;
        case GGML_OP_UPSCALE:
            break;
        case GGML_OP_PAD:
            break;
        case GGML_OP_PAD_REFLECT_1D:
            break;
        case GGML_OP_ARANGE:
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            break;
        case GGML_OP_LEAKY_RELU:
            break;
        case GGML_OP_SILU_BACK:
            break;
        case GGML_OP_RMS_NORM:
            ggml_rpp_op_rms_norm(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_RMS_NORM_BACK:
            break;
        case GGML_OP_MUL_MAT:
            ggml_rpp_op_mul_mat(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_rpp_op_mul_mat_id(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_OUT_PROD:
            break;
        case GGML_OP_SCALE:
            ggml_rpp_op_scale(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_SQR:
            break;
        case GGML_OP_SQRT:
            break;
        case GGML_OP_SIN:
            break;
        case GGML_OP_COS:
            break;
        case GGML_OP_CLAMP:
            break;
        case GGML_OP_LOG:
            break;
        case GGML_OP_NONE:
            break;
        case GGML_OP_RESHAPE:
            break;
        case GGML_OP_VIEW:
            break;
        case GGML_OP_PERMUTE:
            //ggml_rpp_op_permute(ctx, dst);
            break;
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
            break;
        case GGML_OP_SOFT_MAX:
            //ggml_rpp_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            break;
        case GGML_OP_ROPE:
            ggml_rpp_op_rope(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_ROPE_BACK:
            break;
        case GGML_OP_ROLL:
            break;
        case GGML_OP_IM2COL:
            break;
        case GGML_OP_IM2COL_3D:
            break;
        case GGML_OP_CONV_2D:
            break;
        case GGML_OP_CONV_2D_DW:
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            break;
        case GGML_OP_POOL_2D:
            ggml_rpp_op_pool_2d(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_SUM:
            break;
        case GGML_OP_SUM_ROWS:
            break;
        case GGML_OP_MEAN:
            break;
        case GGML_OP_SSM_CONV:
            break;
        case GGML_OP_SSM_SCAN:
            break;
        case GGML_OP_ARGSORT:
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_rpp_op_flash_attn_ext(ctx, dst, is_instantial, is_launch);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            break;
        case GGML_OP_RWKV_WKV6:
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            break;
        case GGML_OP_RWKV_WKV7:
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            break;
        case GGML_OP_OPT_STEP_SGD:
            break;
        default:
            return false;
    }

    rtError_t err = rtGetLastError();
    if (err != rtSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        RPP_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend
/**
 * @brief Retrieves the name associated with the RPP backend.
 *
 * This function returns the name assigned to the RPP backend, which is stored
 * in the context of the provided backend structure.
 *
 * @param backend Pointer to the RPP backend structure.
 * @return A pointer to a constant string representing the backend name.
 */
static const char * ggml_backend_rpp_get_name(ggml_backend_t backend) {
    ggml_backend_rpp_context * rpp_ctx = (ggml_backend_rpp_context *) backend->context;

    return rpp_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the RPP backend.
 *
 * This function releases resources associated with the RPP backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the RPP backend structure to be freed.
 */
static void ggml_backend_rpp_free(ggml_backend_t backend) {
    ggml_backend_rpp_context * rpp_ctx = (ggml_backend_rpp_context *) backend->context;

    delete rpp_ctx;
    delete backend;
}

/**
 * @brief Sets tensor data asynchronously in the RPP backend.
 *
 * This function asynchronously sets tensor data in the RPP backend.
 *
 * @param backend Pointer to the RPP backend structure.
 * @param tensor Pointer to the tensor structure to set data for.
 * @param data Pointer to the host data to copy to the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
static void ggml_backend_rpp_set_tensor_async(ggml_backend_t backend,
                                              ggml_tensor *  tensor,
                                              const void *   data,
                                              size_t         offset,
                                              size_t         size) {
    if (size == 0) {
        return;
    }
    ggml_backend_rpp_context * rpp_ctx = (ggml_backend_rpp_context *) backend->context;
    ggml_backend_buffer_t      buf     = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_rpp_buffer_type(rpp_ctx->device) && "unsupported buffer type");
    RPP_CHECK(rtMemcpyAsync((char *) tensor->data + offset, data, size, rtMemcpyHostToDevice, rpp_ctx->stream()));
}

/**
 * @brief Gets tensor data asynchronously in the RPP backend.
 *
 * This function asynchronously gets tensor data in the RPP backend.
 *
 * @param backend Pointer to the RPP backend structure.
 * @param tensor Pointer to the tensor structure to get data from.
 * @param data Pointer to the host data to copy from the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
static void ggml_backend_rpp_get_tensor_async(ggml_backend_t      backend,
                                              const ggml_tensor * tensor,
                                              void *              data,
                                              size_t              offset,
                                              size_t              size) {
    ggml_backend_rpp_context * rpp_ctx = (ggml_backend_rpp_context *) backend->context;
    ggml_backend_buffer_t      buf     = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_rpp_buffer_type(rpp_ctx->device) && "unsupported buffer type");
    RPP_CHECK(rtMemcpyAsync(data, (const char *) tensor->data + offset, size, rtMemcpyDeviceToHost, rpp_ctx->stream()));
}

/**
 * @brief Asynchronously copies tensor data between RPP backends.
 *
 * This function copies tensor data asynchronously between two RPP backends. It
 * checks if both tensors reside in RPP buffers and whether the devices support
 * peer-to-peer access for direct copying. If not, it returns false.
 *
 * @param backend_src Pointer to the source RPP backend structure.
 * @param backend_dst Pointer to the destination RPP backend structure.
 * @param src Pointer to the source tensor to copy data from.
 * @param dst Pointer to the destination tensor to copy data to.
 * @return true if the copy operation succeeds, false otherwise.
 */
static bool ggml_backend_rpp_cpy_tensor_async(ggml_backend_t      backend_src,
                                              ggml_backend_t      backend_dst,
                                              const ggml_tensor * src,
                                              ggml_tensor *       dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_rpp(backend_src) || !ggml_backend_is_rpp(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_rpp(src->buffer) || !ggml_backend_buffer_is_rpp(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_rpp_context * rpp_ctx_src = (ggml_backend_rpp_context *) backend_src->context;
    ggml_backend_rpp_context * rpp_ctx_dst = (ggml_backend_rpp_context *) backend_dst->context;

    ggml_backend_rpp_buffer_context * buf_ctx_src = (ggml_backend_rpp_buffer_context *) buf_src->context;
    ggml_backend_rpp_buffer_context * buf_ctx_dst = (ggml_backend_rpp_buffer_context *) buf_dst->context;

    if (rpp_ctx_src->device != buf_ctx_src->device || rpp_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        if (rpp_ctx_src->device == rpp_ctx_dst->device) {
            RPP_CHECK(
                rtMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), rtMemcpyDeviceToDevice, rpp_ctx_src->stream()));
        } else {
#ifdef GGML_RPP_NO_PEER_COPY
            return false;
#else
            GGML_ASSERT(false && "peer copy is not supported");
            // RPP_CHECK(rppMemcpyPeerAsync(dst->data, rpp_ctx_dst->device, src->data, rpp_ctx_src->device, ggml_nbytes(dst), rpp_ctx_src->stream()));
#endif
        }

        // record event on src stream after the copy
        if (!rpp_ctx_src->copy_event) {
            ggml_rpp_set_device(rpp_ctx_src->device);
            RPP_CHECK(rtEventCreateWithFlags(&rpp_ctx_src->copy_event, rtEventDisableTiming));
        }

        RPP_CHECK(rtEventRecord(rpp_ctx_src->copy_event, rpp_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        RPP_CHECK(rtStreamWaitEvent(rpp_ctx_dst->stream(), rpp_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        RPP_CHECK(rtMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), rtMemcpyDeviceToDevice, rpp_ctx_src->stream()));
    }
    return true;
}

/**
 * @brief Synchronizes a RPP backend.
 *
 * This function synchronizes the specified RPP backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the RPP backend structure to synchronize.
 */
static void ggml_backend_rpp_synchronize(ggml_backend_t backend) {
    ggml_backend_rpp_context * rpp_ctx = (ggml_backend_rpp_context *) backend->context;
    // RPP_CHECK(rtStreamSynchronize(rpp_ctx->stream()));
    for (auto stream_ptr : rpp_ctx->streams[rpp_ctx->device]) {
        if (stream_ptr == nullptr) {
            break;
        }
        RPP_CHECK(rtStreamSynchronize(stream_ptr));
    }
    GGML_UNUSED(backend);
}

// #ifdef USE_RPP_GRAPH
#if 1

static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    graph_node_properties->node_address = node->data;
    graph_node_properties->node_op      = node->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        graph_node_properties->ne[i] = node->ne[i];
        graph_node_properties->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
    memcpy(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool ggml_graph_node_has_matching_properties(ggml_tensor *                node,
                                                    ggml_graph_node_properties * graph_node_properties) {
    if (node->data != graph_node_properties->node_address && node->op != GGML_OP_CPY && node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != graph_node_properties->node_op) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties->ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties->nb[i]) {
            return false;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] && node->src[i]->data != graph_node_properties->src_address[i] && node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW) {
            return false;
        }
    }

    if (node->op == GGML_OP_SCALE &&
        memcmp(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

static bool is_rpp_node_update_required(ggml_backend_rpp_context * rpp_ctx, ggml_tensor * gtensor) {
    bool rpp_engine_update_required = false;

    auto engine_iter = rpp_ctx->cur_rpp_graph->cur_rpp_nodes.find(gtensor);
    if (engine_iter != rpp_ctx->cur_rpp_graph->cur_rpp_nodes.end()) {
        ggml_tensor * cur_tensor[GGML_MAX_SRC + 1] = { gtensor };
        auto          end_iter = std::copy_if(gtensor->src, gtensor->src + GGML_MAX_SRC, cur_tensor + 1,
                                              [](ggml_tensor * ptr) { return ptr != nullptr; });
        auto          engine   = engine_iter->second;
        for (int i = 0; i < end_iter - cur_tensor; i++) {
            bool has_matching_properties = true;
            auto iter                    = engine->ggml_node_properties.find(cur_tensor[i]);
            if (iter == engine->ggml_node_properties.end()) {
                rpp_engine_update_required = true;
                ggml_graph_node_properties properties;
                set_ggml_graph_node_properties(cur_tensor[i], &properties);
                engine->ggml_node_properties[cur_tensor[i]] = properties;
            } else {
                has_matching_properties = ggml_graph_node_has_matching_properties(iter->first, &(iter->second));
                if (!has_matching_properties) {
                    set_ggml_graph_node_properties(iter->first, &(iter->second));
                    rpp_engine_update_required = true;
                }
            }
        }
    }
    return rpp_engine_update_required;
}

static bool is_rpp_graph_update_required(ggml_backend_rpp_context * rpp_ctx, ggml_cgraph * cgraph) {
    TRACE_SCOPE_GUARD(rpp_ctx->trace_id, "rpp_context_is_update_graph");
    bool rpp_graph_update_required = false;

    // Check if the graph size has changed
    if (rpp_ctx->cur_rpp_graph->ggml_graph_properties.size() != (size_t) cgraph->n_nodes) {
        rpp_graph_update_required = true;
        rpp_ctx->cur_rpp_graph->ggml_graph_properties.resize(cgraph->n_nodes);
    }

    // // Loop over nodes in GGML graph to determine if RPP graph update is required
    // // and store properties to allow this comparison for the next token
    for (int i = 0; i < cgraph->n_nodes; i++) {
        bool          has_matching_properties = true;
        ggml_tensor * node                    = cgraph->nodes[i];
        // if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
        //     node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
        //     set_ggml_graph_node_properties(cgraph->nodes[i], &rpp_ctx->cur_rpp_graph->ggml_graph_properties[i]);
        //     continue;
        // }
        if (!rpp_graph_update_required) {
            has_matching_properties = ggml_graph_node_has_matching_properties(
                cgraph->nodes[i], &rpp_ctx->cur_rpp_graph->ggml_graph_properties[i]);
        }
        if (!has_matching_properties) {
            rpp_graph_update_required = true;
        }
        set_ggml_graph_node_properties(cgraph->nodes[i], &rpp_ctx->cur_rpp_graph->ggml_graph_properties[i]);
    }
    return rpp_graph_update_required;
}

static bool update_ggml_rpp_infer_states(ggml_backend_rpp_context * rpp_ctx, ggml_rpp_cgraph * rpp_graph) {
    ggml_tensor * node = rpp_graph->ggml_graph->nodes[0];
    GGML_ASSERT(node);
    TRACE_SCOPE_GUARD(rpp_ctx->trace_id, "rpp_context_update_states");
    if (!rpp_ctx->ggml_first_properties.size()) {
        ggml_graph_node_properties properties;
        set_ggml_graph_node_properties(node, &properties);
        rpp_ctx->ggml_first_properties[node] = properties;
    }
    if (rpp_ctx->ggml_first_properties.begin()->first == node) {
        // prefill to decode or decode to prefill
        if (!ggml_graph_node_has_matching_properties(node, &rpp_ctx->ggml_first_properties[node])) {
            set_ggml_graph_node_properties(node, &rpp_ctx->ggml_first_properties[node]);
            for (auto & item : rpp_ctx->rpp_graphs) {
                item.second->cur_rpp_nodes.clear();
            }
#    if GGML_RPP_PERF_TRACE
            rpp_ctx->trace_num++;
#    endif
        }
        rpp_ctx->rpp_io_buffers.clear();
    }
    return true;
}

static void update_rpp_graph_executable(ggml_backend_rpp_context * rpp_ctx) {}
#endif

static bool ggml_rpp_can_fuse(const struct ggml_cgraph *                cgraph,
                              int                                       node_idx,
                              std::initializer_list<enum ggml_op>       ops,
                              std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx + 2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add &&
            (add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32)) {
            return false;
        }

        // if rms_norm is the B operand, then we don't handle broadcast.
        // It must be a plain elementwise multiply with equal-shaped lhs/rhs.
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contigous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_SCALE && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor * scale  = cgraph->nodes[node_idx];
        const ggml_tensor * tanh   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scale2 = cgraph->nodes[node_idx + 2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

static std::unordered_set<ggml_tensor *> get_ggml_graph_all_nodes(ggml_cgraph * cgraph) {
    std::unordered_set<ggml_tensor *> rppGgml_set;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        // if ((ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
        //      node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE)) {
        //     continue;
        // }
        rppGgml_set.emplace(node);
    }

    return rppGgml_set;
}

bool ggml_rpp_op_rms_norm_fused(ggml_backend_rpp_context & ctx,
                                ggml_tensor *              dst,
                                ggml_tensor *              mul_tensor,
                                int                        is_instantial = 1,
                                int                        is_launch     = 1) {
    return ggml_rpp_op_rms_norm_mul_fusion(ctx, dst, mul_tensor, is_instantial, is_launch);
}

static bool ggml_rpp_is_exclusive_graph_op(const ggml_tensor * node) {
    GGML_ASSERT(node && node->op != GGML_OP_NONE);
    switch (node->op) {
        // case GGML_OP_GET_ROWS:
        // case GGML_OP_SET_ROWS:
        // case GGML_OP_FLASH_ATTN_EXT:
        // case GGML_OP_ADD:
        // case GGML_OP_CONT:
        // case GGML_OP_ROPE:
        // case GGML_OP_CPY:
        case GGML_OP_MUL_MAT_ID:

            return true;
        default:
            return false;
    }
}

static void evaluate_and_capture_rpp_graph(ggml_backend_rpp_context * rpp_ctx,
                                           ggml_cgraph *              cgraph,
                                           bool &                     graph_evaluated_or_captured,
                                           bool &                     use_rpp_graph,
                                           bool &                     rpp_graph_update_required) {
    // flag used to determine whether it is an integrated_gpu
    const bool integrated = ggml_rpp_info().devices[rpp_ctx->device].integrated;
    GGML_UNUSED(integrated);
    GGML_UNUSED(use_rpp_graph);
    static bool disable_fusion = false;
    if (getenv("GGML_RPP_DISABLE_FUSION") != nullptr && atoi(getenv("GGML_RPP_DISABLE_FUSION")) != 0) {
        disable_fusion = true;
    }
    static bool disable_graph_capture = false;
    if (getenv("GGML_RPP_DISABLE_GRAPH_CAPTURE") != nullptr && atoi(getenv("GGML_RPP_DISABLE_GRAPH_CAPTURE")) != 0) {
        disable_graph_capture = true;
    }

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if RPP graphs are not enabled, or we are capturing the graph.
        // With the use of RPP graphs, the execution will be performed by the graph launch.
        if (rpp_graph_update_required) {
            TRACE_SCOPE_GUARD(rpp_ctx->trace_id, "rpp_context_update_graph_io");
            // to reset current graph
            ggml_rpp_reset_graph(rpp_ctx, rpp_ctx->cur_rpp_graph);
            rpp_ctx->cur_rpp_graph->nodes_all                         = get_ggml_graph_all_nodes(cgraph);
            std::unordered_set<ggml_tensor *> & rppGgml_set           = rpp_ctx->cur_rpp_graph->nodes_all;
            std::unordered_set<ggml_tensor *> & rppGgml_o             = rpp_ctx->cur_rpp_graph->nodes_o;
            std::unordered_set<ggml_tensor *> & rppGgml_i             = rpp_ctx->cur_rpp_graph->nodes_i;
            std::unordered_set<ggml_tensor *> & rppGgml_matmul_weight = rpp_ctx->cur_rpp_graph->nodes_matmul_weight;
            std::unordered_set<ggml_tensor *> & rppGgml_mul_weight    = rpp_ctx->cur_rpp_graph->nodes_mul_weight;
            std::unordered_set<ggml_tensor *> & rppGgml_cache_kv      = rpp_ctx->cur_rpp_graph->nodes_cache_kv;
            rppGgml_o                                                 = rppGgml_set;
            // get cgraph inputs, outputs, weights etc, and create rpprt tensor
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                // if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
                //     node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                //     continue;
                // }
                for (size_t j = 0; j < GGML_MAX_SRC; j++) {
                    auto src_node = node->src[j];
                    if (src_node == nullptr) {
                        break;
                    }
                    if (rppGgml_set.find(src_node) == rppGgml_set.end()) {
                        if (is_matmul_weight(src_node)) {
                            rppGgml_matmul_weight.emplace(src_node);
                        } else if (is_mul_weight(src_node)) {
                            rppGgml_mul_weight.emplace(src_node);
                        } else if (is_cache_kv(src_node)) {
                            rppGgml_cache_kv.emplace(src_node);
                        } else {
                            rppGgml_i.emplace(src_node);
                        }
                    } else {
                        rppGgml_o.erase(src_node);
                        if (node->op == GGML_OP_SET_ROWS && is_cache_kv(node)) {
                            rppGgml_o.erase(node);
                        }
                    }
                }
            }

            std::unordered_set<ggml_tensor *> expert_judgment_runtime_skips;
            // create rpp engines and compute forward
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (expert_judgment_runtime_skips.count(node)) {
                    continue;
                }
                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
                    node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }
                int is_instantial = disable_graph_capture ? 1 : 0;
                int is_launch     = disable_graph_capture ? 1 : 0;
                if (!disable_graph_capture && ggml_rpp_is_exclusive_graph_op(node)) {
                    is_instantial = 1;
                    is_launch     = 0;
                }

                if (!disable_fusion) {
                    if (node->op == GGML_OP_ADD) {
                        int     n_fuse = 0;
                        ggml_op ops[8];
                        std::fill(ops, ops + 8, GGML_OP_ADD);

                        for (; n_fuse <= 6; ++n_fuse) {
                            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                                break;
                            }
                            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                                break;
                            }
                            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1],
                                                      cgraph->nodes[i + n_fuse + 1]->src[1])) {
                                break;
                            }
                        }
                        n_fuse++;

                        if (n_fuse > 1) {
                            for (int j = 0; j < n_fuse - 1; ++j) {
                                node->src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
                            }
                            cgraph->nodes[i + n_fuse - 1]->data = node->data;
                            const bool ok_fuse = ggml_rpp_op_reduce_sum(*rpp_ctx, node, is_instantial, is_launch);
                            if (!ok_fuse) {
                                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name,
                                               ggml_op_name(node->op));
                            }
                            GGML_ASSERT(ok_fuse);
                            // because the reduce sum node is fused, so we need to set the node properties for the reduce sum node
                            set_ggml_graph_node_properties(cgraph->nodes[i],
                                                           &(rpp_ctx->cur_rpp_graph->ggml_graph_properties[i]));
                            i += n_fuse - 1;
                            continue;
                        }
                    }

                    if (node->op == GGML_OP_RMS_NORM) {
                        if (ggml_rpp_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
                            const bool ok_fuse = ggml_rpp_op_rms_norm_fused(*rpp_ctx, node, cgraph->nodes[i + 1],
                                                                            is_instantial, is_launch);
                            if (!ok_fuse) {
                                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name,
                                               ggml_op_name(node->op));
                            }
                            GGML_ASSERT(ok_fuse);
                            i++;
                            continue;
                        }
                    }

                    if (node->op == GGML_OP_SOFT_MAX) {
                        ggml_rpp_router_expert_judgment_fusion_desc routing_fusion;
                        if (ggml_rpp_can_fuse_expert_routing(cgraph, i, routing_fusion, expert_judgment_runtime_skips)) {
                            TRACE_SCOPE_GUARD(rpp_ctx->trace_id, "ggml_rpp_op_expert_routing_fusion");
                            const bool ok_fuse =
                                ggml_rpp_op_expert_routing_fusion(*rpp_ctx, routing_fusion, is_instantial, is_launch);
                            GGML_ASSERT(ok_fuse);
                            expert_judgment_runtime_skips.insert(routing_fusion.compute_nodes_to_skip.begin(),
                                                                 routing_fusion.compute_nodes_to_skip.end());
                            continue;
                        }
                    }

                    if (node->op == GGML_OP_MUL_MAT_ID) {
                        ggml_tensor * gate_tensor = nullptr;
                        ggml_tensor * up_tensor   = nullptr;
                        ggml_tensor * down_tensor = nullptr;
                        ggml_tensor * div_tensor  = nullptr;
                        ggml_tensor * add_tensor  = nullptr;
                        if (ggml_rpp_can_fuse_expert_forward(cgraph, i, gate_tensor, up_tensor, down_tensor, div_tensor,
                                                             add_tensor)) {
                            const bool ok_fuse =
                                ggml_rpp_op_export_forward(*rpp_ctx, gate_tensor, up_tensor, down_tensor, div_tensor,
                                                           add_tensor, is_instantial, is_launch);
                            GGML_ASSERT(ok_fuse);
                            continue;
                        }
                    }
                }

                const bool ok = ggml_rpp_compute_forward(*rpp_ctx, node, is_instantial, is_launch);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            }

            if (!disable_graph_capture) {
                int       pending_kernel_graph_idx = -1;
                const int u_ubatch                 = rpp_ctx->cur_rpp_graph->rpp_in_use_nodes[0]->n_ubatch;
                const int u_nodes                  = rpp_ctx->cur_rpp_graph->rpp_in_use_nodes.size();
                const int graph_index              = u_nodes + u_ubatch;
                auto &    rpp_kernel_graphs        = rpp_ctx->cur_rpp_graph->rpp_kernel_graphs;
                auto &    rpp_in_use_kernel_graphs = rpp_ctx->cur_rpp_graph->rpp_in_use_kernel_graphs;
                // if (rpp_kernel_graphs.count(graph_index)){
                //     rpp_kernel_graphs[graph_index].clear();
                //     rpp_kernel_graphs.erase(graph_index);
                // }
                if (rpp_kernel_graphs.count(graph_index)) {
                    TRACE_SCOPE_GUARD(rpp_ctx->trace_id, "rpp_update_child_graph");
                    auto & cur_kernel_graphs = rpp_kernel_graphs[graph_index];
                    auto & cur_rpp_nodes     = rpp_ctx->cur_rpp_graph->cur_rpp_nodes;
                    for (auto & cur_graph : cur_kernel_graphs) {
                        cur_graph->update_child_graph(cur_rpp_nodes);
                        rpp_in_use_kernel_graphs.emplace_back(cur_graph.get());
                    }
                } else {
                    rpp_kernel_graphs[graph_index] = std::vector<std::unique_ptr<rpp_kernel_cgraph>>();
                    auto & cur_kernel_graphs       = rpp_kernel_graphs[graph_index];

                    auto append_exclusive_kernel_graph = [&](ggml_tensor * node, int & pending_kernel_graph_idx) {
                        ggml_rpp_node * rpp_node = nullptr;
                        if (rpp_ctx->cur_rpp_graph->cur_rpp_nodes.count(node)) {
                            rpp_node = rpp_ctx->cur_rpp_graph->cur_rpp_nodes[node];
                        }
                        GGML_ASSERT(rpp_node && rpp_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL);
                        auto kernel_node = static_cast<rpp_node_kernel *>(rpp_node);
                        cur_kernel_graphs.emplace_back(std::make_unique<rpp_kernel_cgraph>(rpp_node));
                        pending_kernel_graph_idx = -1;
                    };

                    auto append_shared_kernel_graph = [&](ggml_tensor * node, int & pending_kernel_graph_idx) -> bool {
                        ggml_rpp_node * rpp_node = nullptr;
                        if (rpp_ctx->cur_rpp_graph->cur_rpp_nodes.count(node)) {
                            rpp_node = rpp_ctx->cur_rpp_graph->cur_rpp_nodes[node];
                        }
                        GGML_ASSERT(rpp_node && rpp_node->rpp_type == ggml_rpp_node::RPP_NODE_TYPE_KERNEL);
                        if (pending_kernel_graph_idx < 0) {
                            cur_kernel_graphs.emplace_back(std::make_unique<rpp_kernel_cgraph>());
                            pending_kernel_graph_idx = (int) cur_kernel_graphs.size() - 1;
                        }
                        cur_kernel_graphs[pending_kernel_graph_idx]->add_child_graph(rpp_node);
                        return true;
                    };

                    for (auto & rpp_node : rpp_ctx->cur_rpp_graph->rpp_in_use_nodes) {
                        auto &     ggml_node       = rpp_node->cur_ggml_tensor;
                        const bool is_exclusive_op = ggml_rpp_is_exclusive_graph_op(ggml_node);
                        if (is_exclusive_op) {
                            append_exclusive_kernel_graph(ggml_node, pending_kernel_graph_idx);
                        } else {
                            append_shared_kernel_graph(ggml_node, pending_kernel_graph_idx);
                        }
                    }
                    for (auto & cur_graph : cur_kernel_graphs) {
                        cur_graph->graph_instantiate();
                        rpp_in_use_kernel_graphs.emplace_back(cur_graph.get());
                    }
                }
                const bool ok_launch_funcs = rpp_ctx->cur_rpp_graph->run_launch_funcs(*rpp_ctx);
                if (!ok_launch_funcs) {
                    GGML_LOG_ERROR("%s: launch funcs failed before graph launch\n", __func__);
                }
                GGML_ASSERT(ok_launch_funcs);
                for (auto & cur_graph : rpp_in_use_kernel_graphs) {
                    cur_graph->graph_launch(*rpp_ctx, rpp_ctx->stream(), rpp_ctx->trace_id);
                }
            }
        } else {
            if (!disable_graph_capture) {
                auto &     kernel_graphs   = rpp_ctx->cur_rpp_graph->rpp_in_use_kernel_graphs;
                const bool ok_launch_funcs = rpp_ctx->cur_rpp_graph->run_launch_funcs(*rpp_ctx);
                if (!ok_launch_funcs) {
                    GGML_LOG_ERROR("%s: launch funcs failed before graph launch\n", __func__);
                }
                GGML_ASSERT(ok_launch_funcs);
                for (auto & cur_graph : kernel_graphs) {
                    cur_graph->graph_launch(*rpp_ctx, rpp_ctx->stream(), rpp_ctx->trace_id);
                }
            } else {
                auto & rpp_nodes = rpp_ctx->cur_rpp_graph->rpp_in_use_nodes;
                for (auto & rpp_node : rpp_nodes) {
                    const bool ok = rpp_node->rpp_dispatch_func(*rpp_ctx, rpp_node->cur_ggml_tensor, 1, 1);
                    if (!ok) {
                        GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, rpp_node->cur_ggml_tensor->name,
                                       ggml_op_name(rpp_node->cur_ggml_tensor->op));
                    }
                    GGML_ASSERT(ok);
                }
            }
        }
        {
            TRACE_SCOPE_GUARD(rpp_ctx->trace_id, "rpp_context_synchronize");
            for (auto stream_ptr : rpp_ctx->streams[rpp_ctx->device]) {
                if (stream_ptr == nullptr) {
                    break;
                }
                RPP_CHECK(rtStreamSynchronize(stream_ptr));
            }
        }
        graph_evaluated_or_captured = true;
    }
}

/**
 * @brief Computes a computational graph using a RPP backend.
 *
 * This function computes the operations defined in the computational graph
 * using the specified RPP backend.
 *
 * @param backend Pointer to the RPP backend structure to use for computation.
 * @param cgraph Pointer to the computational graph structure containing nodes
 *               representing operations to be computed.
 * @return enum ggml_status Returns GGML_STATUS_SUCCESS if computation
 *         completes successfully, otherwise an appropriate error status.
 */

static enum ggml_status ggml_backend_rpp_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpp_context * rpp_ctx = (ggml_backend_rpp_context *) backend->context;
    ggml_rpp_set_device(rpp_ctx->device);
    rpp_ctx->cur_rpp_graph = nullptr;
#if GGML_RPP_DUMP_OPS
    char str1_name[64];
    sprintf(str1_name, "./graph_%p_ops_dump.txt", (void *) cgraph);
    ggml_rpp_dump_cgraph_ops(cgraph, str1_name, "toy graph y = x + x");
    char str2_name[64];
    sprintf(str2_name, "./graph_%p_ops_dump.dot", (void *) cgraph);
    ggml_graph_dump_dot(cgraph, NULL, str2_name);
    printf("successfully dump graph txt file: %s\n", str1_name);
    printf("successfully dump graph dot file: %s\n", str2_name);
    exit(0);
#endif

    // get current rpp graph from rpp_graphs map, if not exit, new one
    auto iter = rpp_ctx->rpp_graphs.find(cgraph);
    if (iter != rpp_ctx->rpp_graphs.end()) {
        rpp_ctx->cur_rpp_graph = iter->second.get();
    } else {
        auto tmp_rpp_graph          = std::make_unique<ggml_rpp_cgraph>(cgraph);
        rpp_ctx->rpp_graphs[cgraph] = std::move(tmp_rpp_graph);
        rpp_ctx->cur_rpp_graph      = rpp_ctx->rpp_graphs[cgraph].get();
        rpp_ctx->gglm_graphs.emplace_back(cgraph);
    }
    update_ggml_rpp_infer_states(rpp_ctx, rpp_ctx->cur_rpp_graph);
#if GGML_RPP_PERF_TRACE
    if (rpp_ctx->trace_num >= 2) {
        rpp_ctx->trace_num++;
    }
    if (rpp_ctx->trace_num == 3) {
        TRACE_ENABLE(rpp_ctx->trace_id);
    }
    if (rpp_ctx->trace_num > 7) {
        TRACE_DISABLE(rpp_ctx->trace_id);
        exit(0);
    }
#endif
    bool rpp_graph_update_required   = is_rpp_graph_update_required(rpp_ctx, cgraph);
    bool use_rpp_graph               = true;
    bool graph_evaluated_or_captured = false;
    evaluate_and_capture_rpp_graph(rpp_ctx, cgraph, graph_evaluated_or_captured, use_rpp_graph,
                                   rpp_graph_update_required);
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Records an event on the RPP backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
static void ggml_backend_rpp_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpp_context * rpp_ctx = (ggml_backend_rpp_context *) backend->context;

    RPP_CHECK(rtEventRecord((rtEvent_t) event->context, rpp_ctx->stream()));
}

/**
 * @brief Waits for a recorded event to complete on the RPP backend stream.
 *
 * This function makes the given backend wait for the event to complete on its
 * ACL runtime stream.
 *
 * @param backend Pointer to the backend structure.
 * @param event Pointer to the event structure that the backend needs to wait
 * for.
 */
static void ggml_backend_rpp_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpp_context * rpp_ctx = (ggml_backend_rpp_context *) backend->context;

    if (ggml_backend_is_rpp(backend)) {
        RPP_CHECK(rtStreamWaitEvent(rpp_ctx->stream(), (rtEvent_t) event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        RPP_CHECK(rtLaunchHostFunc(rpp_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

/**
 * @brief Structure defining the interface for the RPP backend.
 *
 * This structure contains function pointers for various operations
 * supported by the RPP backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static const ggml_backend_i ggml_backend_rpp_interface = {
    /* .get_name                = */ ggml_backend_rpp_get_name,
    /* .free                    = */ ggml_backend_rpp_free,
    /* .set_tensor_async        = */ ggml_backend_rpp_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpp_get_tensor_async,
    /* .set_tensor_2d           = */ NULL,
    /* .get_tensor_2d           = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_rpp_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_rpp_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpp_graph_compute,
    /* .event_record            = */ ggml_backend_rpp_event_record,
    /* .event_wait              = */ ggml_backend_rpp_event_wait,
};

/**
 * @brief Return the hardcoded GUID for the RPP backend.
 *
 * This function returns a static GUID which uniquely identifies the RPP
 * backend.
 * notice：currently, rpp does not have a guid. So randomly created one.
 * @return A pointer to the static GUID.
 */
static ggml_guid_t ggml_backend_rpp_guid() {
    static ggml_guid guid = { 0x4a, 0x1f, 0x9b, 0x72, 0x8e, 0xaf, 0x34, 0x6d,
                              0xb1, 0x2c, 0x9a, 0x5e, 0x60, 0x7f, 0x11, 0x9c };
    return &guid;
}

bool ggml_backend_is_rpp(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpp_guid());
}

int ggml_backend_rpp_get_device_count() {
    return ggml_rpp_info().device_count;
}

void ggml_backend_rpp_get_device_description(int device, char * description, size_t description_size) {
    rtDeviceProp prop;
    RPP_CHECK(rtGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

void ggml_backend_rpp_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_rpp_set_device(device);
    rppMemGetInfo(free, total);
    *free  = (size_t) 6 * 1024 * 1024 * 1024;
    *total = (size_t) 8 * 1024 * 1024 * 1024;
}

bool ggml_backend_rpp_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_RPP_REGISTER_HOST") == nullptr) {
        return false;
    }
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
}

void ggml_backend_rpp_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_RPP_REGISTER_HOST") == nullptr) {
        return;
    }
    rtError_t err = rtSuccess;
    ///< rpp not support rtHostUnregister, disable code
    // err = rtHostUnregister(buffer);
    if (err != rtSuccess) {
        // clear the error
        (void) rtGetLastError();
    }
}

// backend device

struct ggml_backend_rpp_device_context {
    int         device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_rpp_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpp_device_context * ctx = (ggml_backend_rpp_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_rpp_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpp_device_context * ctx = (ggml_backend_rpp_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_rpp_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpp_device_context * ctx = (ggml_backend_rpp_device_context *) dev->context;
    ggml_rpp_set_device(ctx->device);
    rppMemGetInfo(free, total);
    *free  = (size_t) 6 * 1024 * 1024 * 1024;
    *total = (size_t) 8 * 1024 * 1024 * 1024;
}

static enum ggml_backend_dev_type ggml_backend_rpp_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_rpp_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpp_device_get_name(dev);
    props->description = ggml_backend_rpp_device_get_description(dev);
    props->type        = ggml_backend_rpp_device_get_type(dev);
    ggml_backend_rpp_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_RPP_NO_PINNED") == nullptr;
#ifdef GGML_RPP_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
    };
}

static ggml_backend_t ggml_backend_rpp_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    // GGML_UNUSED(params);
    ggml_backend_rpp_device_context * ctx = (ggml_backend_rpp_device_context *) dev->context;
    return ggml_backend_rpp_init(ctx->device, params);
}

static ggml_backend_buffer_type_t ggml_backend_rpp_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpp_device_context * ctx = (ggml_backend_rpp_device_context *) dev->context;
    return ggml_backend_rpp_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_rpp_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_rpp_host_buffer_type();
}

/**
 * @brief Checks if the RPP backend supports a specific operation.
 *
 * This function checks whether the specified operation is supported by the
 * RPP backend.
 *
 * @param backend Pointer to the RPP backend structure to check support for
 *                the operation.
 * @param op Pointer to the tensor representing the operation to check.
 * @return bool Returns true if the operation is supported by the backend,
 *              otherwise false.
 */
// TODO: move these functions here
static bool ggml_backend_rpp_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_rpp_device_context * dev_ctx = (ggml_backend_rpp_device_context *) dev->context;
    // split buffers can only be used with GGML_OP_MUL_MAT
    if (op->op != GGML_OP_MUL_MAT) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_rpp_split(op->src[i]->buffer->buft)) {
                return false;
            }
        }
    }

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_rpp(op->src[i]->buffer->buft)) {
            ggml_backend_rpp_buffer_type_context * buft_ctx =
                (ggml_backend_rpp_buffer_type_context *) op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }
    bool supported = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            supported = true;
            break;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            supported = true;
            break;
        case GGML_OP_DIV:
            supported = true;
            break;
        case GGML_OP_ARGSORT:
        case GGML_OP_SUM_ROWS:
            supported = true;
            break;
        case GGML_OP_RMS_NORM:
            supported = true;
            break;
        case GGML_OP_SET_ROWS:
            {
                supported = (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_BF16) &&
                            op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_I64;
            }
            break;
        case GGML_OP_GET_ROWS:
            supported = op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
            break;
        case GGML_OP_MUL_MAT:
            supported = op->src[0]->type == GGML_TYPE_Q4_1 || op->src[0]->type == GGML_TYPE_BF16 ||
                        op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                        op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q6_K ||
                        op->src[0]->type == GGML_TYPE_Q4_K || op->src[0]->type == GGML_TYPE_Q5_K ||
                        op->src[0]->type == GGML_TYPE_IQ3_XXS || op->src[0]->type == GGML_TYPE_IQ2_S ||
                        op->src[0]->type == GGML_TYPE_IQ2_XS;
            break;
        case GGML_OP_MUL_MAT_ID:
            supported = (op->src[0]->type == GGML_TYPE_Q4_1 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                         op->src[0]->type == GGML_TYPE_Q4_K || op->src[0]->type == GGML_TYPE_Q5_K ||
                         op->src[0]->type == GGML_TYPE_Q6_K || op->src[0]->type == GGML_TYPE_IQ3_XXS ||
                         op->src[0]->type == GGML_TYPE_IQ2_S || op->src[0]->type == GGML_TYPE_IQ2_XS) &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                        op->type == GGML_TYPE_F32;
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            supported = true;
            break;
        case GGML_OP_SOFT_MAX:
            supported = true;
            break;
        case GGML_OP_GLU:
            {
                switch (ggml_get_glu_op(op)) {
                    case GGML_GLU_OP_SWIGLU:
                    case GGML_GLU_OP_GEGLU_ERF:
                        supported = true;
                        break;
                    default:
                        supported = false;
                        break;
                }
            }
            break;
        case GGML_OP_UNARY:
            {
                switch (ggml_get_unary_op(op)) {
                    case GGML_UNARY_OP_GELU:
                    case GGML_UNARY_OP_GELU_ERF:
                        supported = true;
                        break;
                    default:
                        supported = false;
                        break;
                }
            }
            break;
        case GGML_OP_ROPE:
            supported = op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]) &&
                        op->src[0]->type == GGML_TYPE_F32;
            // supported = false;
            break;
        case GGML_OP_CONT:
            supported = true;
            break;
        case GGML_OP_CPY:
            supported = true;
            break;
        case GGML_OP_SCALE:
            supported = true;
            break;
        case GGML_OP_POOL_2D:
            supported = ggml_rpp_pool_2d_supported(op);
            break;
        case GGML_OP_NORM:
            supported = op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_CLAMP:
            supported = op->src[0]->type == GGML_TYPE_F32;
            break;
        default:
            supported = false;
            break;
    }
    if (!supported){
        // GGML_LOG_WARN("ggml_backend_rpp_device_supports_op: unsupported op: %s\n", ggml_op_name(op->op));
    }
    
    return supported;
}

/**
 * @brief Checks if the RPP backend supports a specific backend buffer type.
 *
 * This function determines whether the RPP backend supports the given backend
 * buffer type by comparing the device context of the backend and buffer type.
 * It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the RPP backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the RPP backend supports the buffer type,
 *              otherwise false.
 */
static bool ggml_backend_rpp_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_rpp_device_context * dev_ctx    = (ggml_backend_rpp_device_context *) dev->context;
    const bool                        integrated = ggml_rpp_info().devices[dev_ctx->device].integrated;
    return (((ggml_backend_buft_is_rpp(buft) || ggml_backend_buft_is_rpp_split(buft)) && buft->device == dev) ||
            (integrated && ggml_backend_buft_is_rpp_host(buft)));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

/**
 * @brief Determines if a tensor operation should be offloaded to the RPP
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * RPP backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the RPP backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
static bool ggml_backend_rpp_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return get_op_batch_size(op) >= min_batch_size;

    GGML_UNUSED(dev);
}

/**
 * @brief Creates a new event for the RPP backend device.
 *
 * This function initializes a new event for the RPP backend by setting the
 * device and creating an ACL runtime event. The created event is then wrapped
 * in a ggml_backend_event structure and returned.
 *
 * @param backend Pointer to the RPP backend.
 * @return ggml_backend_event_t Returns a pointer to the new event structure.
 */
static ggml_backend_event_t ggml_backend_rpp_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_RPP_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_rpp_device_context * dev_ctx = (ggml_backend_rpp_device_context *) dev->context;

    ggml_rpp_set_device(dev_ctx->device);

    rtEvent_t event;
    RPP_CHECK(rtEventCreateWithFlags(&event, rtEventDisableTiming));

    return new ggml_backend_event{
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

/**
 * @brief Frees a RPP backend event.
 *
 * This function destroys the runtime event associated with the given RPP
 * backend event and then deletes the event structure itself.
 *
 * @param event Pointer to the event structure to be freed.
 */
static void ggml_backend_rpp_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    RPP_CHECK(rtEventDestroy((rtEvent_t) event->context));
    delete event;
}

/**
 * @brief Synchronizes the given event on the RPP backend.
 *
 * This function waits for the specified event to complete on the rpp runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
static void ggml_backend_rpp_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    RPP_CHECK(rtEventSynchronize((rtEvent_t) event->context));
}

static const ggml_backend_device_i ggml_backend_rpp_device_interface = {
    /* .get_name                = */ ggml_backend_rpp_device_get_name,
    /* .get_description         = */ ggml_backend_rpp_device_get_description,
    /* .get_memory              = */ ggml_backend_rpp_device_get_memory,
    /* .get_type                = */ ggml_backend_rpp_device_get_type,
    /* .get_props               = */ ggml_backend_rpp_device_get_props,
    /* .init_backend            = */ ggml_backend_rpp_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_rpp_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_rpp_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,  //not support for rpp
    /* .supports_op             = */ ggml_backend_rpp_device_supports_op,
    /* .supports_buft           = */ ggml_backend_rpp_device_supports_buft,
    /* .offload_op              = */ ggml_backend_rpp_device_offload_op,
    /* .event_new               = */ ggml_backend_rpp_device_event_new,
    /* .event_free              = */ ggml_backend_rpp_device_event_free,
    /* .event_synchronize       = */ ggml_backend_rpp_device_event_synchronize,
};

// backend reg

struct ggml_backend_rpp_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpp_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_RPP_NAME;
}

static size_t ggml_backend_rpp_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpp_reg_context * ctx = (ggml_backend_rpp_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_rpp_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpp_reg_context * ctx = (ggml_backend_rpp_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_rpp_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
#define _STRINGIFY(...) #__VA_ARGS__
#define STRINGIFY(...)  _STRINGIFY(__VA_ARGS__)

#ifdef GGML_RPP_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", STRINGIFY(GGML_RPP_NO_PEER_COPY) });
#else
        features.push_back({ "NO_PEER_COPY", "1" });
#endif

#ifdef GGML_RPP_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", STRINGIFY(GGML_RPP_USE_GRAPHS) });
#else
        features.push_back({ "USE_GRAPHS", "1" });
#endif

#ifdef GGML_RPP_USE_UBATCH
        features.push_back({ "USE_UBATCH", STRINGIFY(GGML_RPP_USE_UBATCH) });
        if (getenv("GGML_RPP_BATCH_SIZE") != nullptr) {
            features.push_back({ "UBATCH_SIZE", getenv("GGML_RPP_BATCH_SIZE") });
        } else {
            features.push_back({ "UBATCH_SIZE", "128" });
        }
#else
        features.push_back({ "USE_UBATCH", "1" });
#endif

#ifdef GGML_RPP_USE_ASYNC
        features.push_back({ "USE_ASYNC", STRINGIFY(GGML_RPP_USE_ASYNC) });
#else
        features.push_back({ "USE_ASYNC", "1" });
#endif

#ifdef GGML_RPP_SAVE_ENGINE
        features.push_back({ "SAVE_ENGINE", STRINGIFY(GGML_RPP_SAVE_ENGINE) });
#else
        features.push_back({ "SAVE_ENGINE", "" });
#endif

#ifdef GGML_RPP_LOAD_ENGINE
        features.push_back({ "LOAD_ENGINE", STRINGIFY(GGML_RPP_LOAD_ENGINE) });
#else
        features.push_back({ "LOAD_ENGINE", "" });
#endif

#undef _STRINGIFY
#undef STRINGIFY
        features.push_back({ nullptr, nullptr });
        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void ggml_backend_rpp_set_params(ggml_backend_t backend, const int domain, const int u_batch, const int max_context) {
    if (!backend || !ggml_backend_is_rpp(backend)) {
        return;
    }
    auto * ctx = (ggml_backend_rpp_context *) backend->context;
    if (domain < 0 || domain > 3) {
        GGML_LOG_ERROR("%s: invalid domain %d\n", __func__, domain);
        GGML_ASSERT(false);
        return;
    }
    ctx->domain = (rpp_backend_domain) domain;
    GGML_LOG_INFO("%s: set rpp domain to %d\n", __func__, ctx->domain);
    if (ctx->domain == RPP_DOMAIN_VISION || ctx->domain == RPP_DOMAIN_AUDIO) {
        ctx->use_ubatch = false;
        GGML_LOG_INFO("%s: set rpp use ubatch to false\n", __func__);
        return;
    }
    ctx->n_ubatch = u_batch;
    GGML_LOG_INFO("%s: set rpp batch size to %d\n", __func__, ctx->n_ubatch);
    if (max_context <= ctx->n_max_ctx) {
        ctx->n_max_ctx = max_context;
        GGML_LOG_INFO("%s: set rpp max context to %d\n", __func__, ctx->n_max_ctx);
    } else {
        GGML_LOG_WARN("%s: max context %d is greater than rpp max context %d\n", __func__, max_context, ctx->n_max_ctx);
    }
    return;
}

static void * ggml_backend_rpp_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    // notice!!!, not support these functions for rpp, is cuda
    // if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
    //     return (void *)ggml_backend_rpp_split_buffer_type;
    // }
    // if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
    //     return (void *)ggml_backend_rpp_register_host_buffer;
    // }
    // if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
    //     return (void *)ggml_backend_rpp_unregister_host_buffer;
    // }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *) ggml_backend_rpp_get_features;
    }
    if (strcmp(name, "ggml_backend_set_params") == 0) {
        return (void *) ggml_backend_rpp_set_params;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_rpp_reg_interface = {
    /* .get_name          = */ ggml_backend_rpp_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpp_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpp_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpp_reg_get_proc_address,
};

bool ggml_rpp_backend_buft_is_rpp_split(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_rpp_split(buft);
}

// backend registry
ggml_backend_reg_t ggml_backend_rpp_reg() {
    static ggml_backend_reg reg;
    static bool             initialized = false;
    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_rpp_reg_context * ctx = new ggml_backend_rpp_reg_context;

            for (int i = 0; i < ggml_rpp_info().device_count; i++) {
                ggml_backend_rpp_device_context * dev_ctx = new ggml_backend_rpp_device_context;
                dev_ctx->device                           = i;
                dev_ctx->name                             = GGML_RPP_NAME + std::to_string(i);

                ggml_rpp_set_device(i);
                rtDeviceProp prop;
                RPP_CHECK(rtGetDeviceProperties(&prop, i));
                dev_ctx->description = prop.name;

                ggml_backend_dev_t dev = new ggml_backend_device{ /* .iface   = */ ggml_backend_rpp_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_rpp_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_rpp_init(int device, const char * params) {
    if (device < 0 || device >= ggml_backend_rpp_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }
    ggml_backend_rpp_context * ctx = new ggml_backend_rpp_context(device, params);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t rpp_backend = new ggml_backend{
        /* .guid    = */ ggml_backend_rpp_guid(),
        /* .iface   = */ ggml_backend_rpp_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_rpp_reg(), device),
        /* .context = */ ctx,
    };

    return rpp_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_rpp_reg)
