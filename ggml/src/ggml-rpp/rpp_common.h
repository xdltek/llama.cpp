#pragma once
#include "ggml-impl.h"
#include "ggml-rpp.h"
#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#define GGML_COMMON_DECL_RPP
#define GGML_COMMON_IMPL_RPP
#include "ggml-common.h"
#include "rpp_drv_api.h"
#include "rpp_kernel_ctx.h"
#include "rpp_runtime.h"

#include <array>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// #define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE_IMPL(...) (#__VA_ARGS__ == "\"\"" ? std::string() : #__VA_ARGS__)
#define STRINGIZE(...)      STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32

#define GGML_RPP_CC_AMPERE 100

#define GGML_RPP_CC_PASCAL          600
#define GGML_RPP_CC_DP4A            610  // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_RPP_CC_VOLTA           700
#define GGML_RPP_CC_TURING          750
#define GGML_RPP_CC_ADA_LOVELACE    890
#define GGML_RPP_CC_OFFSET_AMD      0x1000000
#define GGML_RPP_CC_OFFSET_MTHREADS 0x0100000
#define GGML_RPP_CC_IS_NVIDIA(cc)   (cc < GGML_RPP_CC_OFFSET_MTHREADS)

#ifndef GGML_RPP_USE_BF16
#    define GGML_RPP_USE_BF16 1
#endif

#ifndef GGML_RPP_USE_RT
#    define GGML_RPP_USE_RT 0
#endif
#if GGML_RPP_USE_RT
#    include "Infer.h"
#endif

#ifndef GGML_RPP_USE_UBATCH
#    define GGML_RPP_USE_UBATCH 1
#endif

#ifndef GGML_RPP_USE_GRAPHS
#    define GGML_RPP_USE_GRAPHS 1
#endif

#ifndef GGML_RPP_NO_PEER_COPY
#    define GGML_RPP_NO_PEER_COPY 1
#endif

#ifndef GGML_RPP_USE_ASYNC
#    define GGML_RPP_USE_ASYNC 1
#endif

#ifndef GGML_RPP_DUMP_OPS
#    define GGML_RPP_DUMP_OPS 0
#endif

#ifndef GGML_RPP_PERF_TRACE
#    define GGML_RPP_PERF_TRACE 0
#endif

#ifndef GGML_RPP_SAVE_ENGINE
#    define GGML_RPP_SAVE_ENGINE ""
#endif

#ifndef GGML_RPP_LOAD_ENGINE
#    define GGML_RPP_LOAD_ENGINE ""
#endif

// last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses
#define MATRIX_ROW_PADDING 32

#define GGML_RPP_MAX_STREAMS 8

//because of not support thread, so user 0
#define rtStreamPerThread (rtStream_t) 0

#if GGML_RPP_PERF_TRACE
#    include "rpp_perf.h"
//! use as:
//! {
//!     TRACE_SCOPE_GUARD(win, "name");
//!     // ...
//! }
//! It emits TRACE_SCOPE_GUARD at scope entry and TRACE_SCOPE_END at scope exit.
#    define _RPP_TRACE_CONCAT_INNER(a, b) a##b
#    define _RPP_TRACE_CONCAT(a, b)       _RPP_TRACE_CONCAT_INNER(a, b)
#    ifdef __cplusplus
struct _rpp_trace_scope_guard_t {
    uint32_t     win;
    const char * name;

    _rpp_trace_scope_guard_t(uint32_t w, const char * n) : win(w), name(n) { TRACE_SCOPE(win, name); }

    ~_rpp_trace_scope_guard_t() { TRACE_SCOPE_END(win, name); }

    _rpp_trace_scope_guard_t(const _rpp_trace_scope_guard_t &)             = delete;
    _rpp_trace_scope_guard_t & operator=(const _rpp_trace_scope_guard_t &) = delete;
};

#        define TRACE_SCOPE_GUARD(win, name) \
            _rpp_trace_scope_guard_t _RPP_TRACE_CONCAT(_rpp_trace_scope_guard_, __LINE__)((uint32_t) (win), (name))
#    else
#        define TRACE_SCOPE_GUARD(win, name) TRACE_SCOPE_GUARD((win), (name))
#    endif

#else
#    define TRACE_SCOPE_GUARD(win, name) 0
#endif

[[noreturn]] void ggml_rpp_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

inline bool ggml_rpp_is_success(rtError_t err) {
    return err == rtSuccess;
}

inline bool ggml_rpp_is_success(RPPresult err) {
    return err == RPP_SUCCESS;
}

inline const char * ggml_rpp_error_string(rtError_t err) {
    return rtGetErrorString(err);
}

inline const char * ggml_rpp_error_string(RPPresult err) {
    return getRppDrvErrorString(err);
}

template <typename T>
inline void ggml_rpp_check_impl(T err, const char * stmt, const char * func, const char * file, int line) {
    if (!ggml_rpp_is_success(err)) {
        ggml_rpp_error(stmt, func, file, line, ggml_rpp_error_string(err));
    }
}

#define RPP_CHECK(err) ggml_rpp_check_impl((err), #err, __func__, __FILE__, __LINE__)

#if GGML_RPP_USE_ASYNC
inline void ggml_rpp_memcpy(void *       dst,
                            const void * src,
                            size_t       size,
                            rtMemcpyKind kind,
                            rtStream_t   stream,
                            bool         sync = false) {
    RPP_CHECK(rtMemcpyAsync(dst, src, size, kind, stream));
    if (sync) {
        RPP_CHECK(rtStreamSynchronize(stream));
    }
}

#    define RPP_MEMCPY_DEV_AND_HOST(dst, src, size, kind, stream, ...) \
        ggml_rpp_memcpy((dst), (src), (size), (kind), (stream), ##__VA_ARGS__)

#    define RPP_EXECUTE_CONTEXT(context, batch, buffers, stream) \
        (context)->enqueue((batch), (buffers), (stream), nullptr)

#    define RPP_LAUNCH_KERNEL(graph, stream) rppGraphLaunch(graph, stream)
#else
#    define RPP_MEMCPY_DEV_AND_HOST(dst, src, size, kind, stream, ...) RPP_CHECK(rtMemcpy((dst), (src), (size), (kind)))
#    define RPP_EXECUTE_CONTEXT(context, batch, buffers, stream)       (context)->execute((batch), (buffers))
#    define RPP_LAUNCH_KERNEL(graph, stream)        \
        do {                                        \
            rppGraphLaunch(graph, stream);          \
            RPP_CHECK(rtStreamSynchronize(stream)); \
        } while (0)

#endif

//////////////////////
struct ggml_rpp_device_info {
    int device_count;

    struct rpp_device_info {
        int    cc;               // compute capability
        int    nsm;              // number of streaming multiprocessors
        size_t smpb;             // max. shared memory per block
        size_t smpbo;            // max. shared memory per block (with opt-in)
        bool   integrated;       // Device is integrated as opposed to discrete
        bool   vmm;              // virtual memory support
        size_t vmm_granularity;  // granularity of virtual memory
        size_t total_vram;
        int    warp_size;        // Number of threads in a dispatch
    };

    rpp_device_info devices[GGML_RPP_MAX_DEVICES] = {};

    std::array<float, GGML_RPP_MAX_DEVICES> default_tensor_split = {};
};

const ggml_rpp_device_info & ggml_rpp_info();
void                         ggml_rpp_set_device(int device);
int                          ggml_rpp_get_device();

struct ggml_rpp_buffer {
    void * ptr  = nullptr;
    size_t size = 0;
};

struct ggml_rpp_pool {
    virtual ~ggml_rpp_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size = nullptr) = 0;
    virtual void   free(void * ptr, size_t size = 0)                  = 0;
};

template <typename T> struct ggml_rpp_pool_alloc {
    ggml_rpp_pool * pool        = nullptr;
    T *             ptr         = nullptr;
    size_t          actual_size = 0;

    ggml_rpp_pool_alloc() = default;

    explicit ggml_rpp_pool_alloc(ggml_rpp_pool & pool) : pool(&pool) {}

    ggml_rpp_pool_alloc(ggml_rpp_pool & pool, size_t size) : pool(&pool) { alloc(size); }

    ~ggml_rpp_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_rpp_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() { return ptr; }

    ggml_rpp_pool_alloc(const ggml_rpp_pool_alloc &)             = delete;
    ggml_rpp_pool_alloc(ggml_rpp_pool_alloc &&)                  = delete;
    ggml_rpp_pool_alloc & operator=(const ggml_rpp_pool_alloc &) = delete;
    ggml_rpp_pool_alloc & operator=(ggml_rpp_pool_alloc &&)      = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
    void *    data_device[GGML_RPP_MAX_DEVICES];                   // 1 pointer for each device for split tensors
    rtEvent_t events[GGML_RPP_MAX_DEVICES][GGML_RPP_MAX_STREAMS];  // events for synchronizing multiple GPUs
};

#if (defined(GGML_RPP_USE_GRAPHS) || defined(GGML_HIP_GRAPHS)) || defined(GGML_MUSA_GRAPHS)
#    define USE_RPP_GRAPH
#endif

struct ggml_graph_node_properties {
    void *  node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    void *  src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

template <typename T> struct ORTDestroyer {
    void operator()(T * ptr) const {
        if (ptr) {
            ptr->destroy();
        }
    }
};

template <typename T> using ort_unique_ptr = std::unique_ptr<T, ORTDestroyer<T>>;

struct ggml_backend_rpp_context;
struct rpp_kernel_cgraph;
struct ggml_rpp_cgraph;

struct ggml_rpp_node {
    enum rpp_node_type {
        RPP_NODE_TYPE_OPENRT = 0,
        RPP_NODE_TYPE_KERNEL = 1,
    };

    enum rpp_node_op {
        RPP_OP_NONE           = 0,
        RPP_OP_RMS_MUL        = 1,
        RPP_OP_REDUCE_SUM     = 2,
        RPP_OP_EXPERT_ROUTING = 3,
        RPP_OP_EXPERT_FORWARD = 4,
        RPP_OP_FUSED_CON2D    = 5,
    };

    std::unordered_map<ggml_tensor *, void *> binding_i_buffers;
    std::unordered_map<ggml_tensor *, void *> binding_o_buffers;
    std::vector<void *>                       binding_io_buffers;
    std::vector<ggml_tensor *>                binding_io_tensors;
    std::unordered_set<void *>                pool_buffers;

    ggml_tensor *                                                 cur_ggml_tensor{ nullptr };
    std::unordered_set<void *>                                    release_buffers;
    std::unordered_map<ggml_tensor *, ggml_graph_node_properties> ggml_node_properties;
    ggml_rpp_node *                                               ori_rpp_node{ nullptr };

    size_t n_ubatch{ 1 };                       //ubatch number, is set by user, =ctx.n_ubatch, now default is 128
    size_t seq_len_index{ GGML_MAX_DIMS - 1 };  //seq len index, need set in prefill, used by decode

    rpp_node_type rpp_type{ RPP_NODE_TYPE_OPENRT };
    rpp_node_op   op{ RPP_OP_NONE };

    ggml_rpp_cgraph * rpp_graph{ nullptr };

    bool virtual rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                                   ggml_tensor *              dst,
                                   int                        is_instantial = 1,
                                   int                        is_launch     = 1) = 0;

    explicit ggml_rpp_node(ggml_tensor * tensor) : cur_ggml_tensor(tensor) {}

    virtual ~ggml_rpp_node() {}
};

#if GGML_RPP_USE_RT
struct RppLogger : public infer1::ILogger {
  public:
    RppLogger(Severity severity = Severity::kERROR) : cur_severity(severity) {}

    infer1::ILogger & get_rpp_logger() { return *this; }

    void log(Severity severity, const char * msg) override {
        if (int(severity) <= int(cur_severity)) {
            switch (severity) {
                case Severity::kVERBOSE:
                    GGML_LOG_DEBUG("%s\n", msg);
                    break;
                case Severity::kINFO:
                    GGML_LOG_INFO("%s\n", msg);
                    break;
                case Severity::kWARNING:
                    GGML_LOG_WARN("%s\n", msg);
                    break;
                case Severity::kERROR:
                case Severity::kINTERNAL_ERROR:
                    GGML_LOG_ERROR("%s\n", msg);
                    break;
                default:
                    GGML_LOG_INFO("%s\n", msg);
                    break;
            }
        }
    }
  private:
    Severity cur_severity;
};

struct rpp_node_openrt : public ggml_rpp_node {
    std::shared_ptr<infer1::IBuilder>           builder{ nullptr };
    std::shared_ptr<infer1::IEngine>            engine{ nullptr };
    std::shared_ptr<infer1::IExecutionContext>  context{ nullptr };
    std::shared_ptr<infer1::INetworkDefinition> network{ nullptr };
    std::shared_ptr<infer1::IBuilderConfig>     config{ nullptr };
    std::shared_ptr<RppLogger>                  logger{ nullptr };

    std::string engine_file;

    explicit rpp_node_openrt(ggml_tensor * tensor) : ggml_rpp_node(tensor) {
        logger = std::make_shared<RppLogger>(infer1::ILogger::Severity::kWARNING);
        builder.reset(infer1::createInferBuilder(logger->get_rpp_logger()));
        builder->setMaxBatchSize(1);
        GGML_ASSERT(builder);
        network.reset(builder->createNetwork());
        GGML_ASSERT(network);
        config.reset(builder->createBuilderConfig());
        GGML_ASSERT(config);
    }

    explicit rpp_node_openrt(ggml_tensor * tensor, std::string engine_file) :
        ggml_rpp_node(tensor),
        engine_file(engine_file) {
        logger = std::make_shared<RppLogger>(infer1::ILogger::Severity::kWARNING);
    }

    explicit rpp_node_openrt(ggml_tensor * tensor, ggml_rpp_node * rpp_base_node) : ggml_rpp_node(tensor) {
        auto rpp_node = static_cast<rpp_node_openrt *>(rpp_base_node);
        builder       = rpp_node->builder;
        engine        = rpp_node->engine;
        context       = rpp_node->context;
        network       = rpp_node->network;
        config        = rpp_node->config;
        logger        = rpp_node->logger;
        ori_rpp_node  = rpp_base_node;
    }
};
#endif
struct rpp_node_kernel : public ggml_rpp_node {
    std::shared_ptr<rpp_kernel_context> kernel_ctx{ nullptr };

    explicit rpp_node_kernel(ggml_tensor * tensor) : ggml_rpp_node(tensor) {
        kernel_ctx = std::make_shared<rpp_kernel_context>();
        rpp_init_kernel_ctx(*kernel_ctx.get());
        rpp_type = ggml_rpp_node::RPP_NODE_TYPE_KERNEL;
    }

    explicit rpp_node_kernel(ggml_tensor * tensor, ggml_rpp_node * rpp_base_node) : ggml_rpp_node(tensor) {
        auto rpp_node = static_cast<rpp_node_kernel *>(rpp_base_node);
        kernel_ctx    = rpp_node->kernel_ctx;
        ori_rpp_node  = rpp_base_node;
        rpp_type      = ggml_rpp_node::RPP_NODE_TYPE_KERNEL;
    }

    int is_instantial{ 1 };  // 1: instantial, 0: not instantial

    rpp_kernel_cgraph * kernel_graph{ nullptr };

    virtual void add_to_parent_node(RPPgraph             hGraph,
                                    const RPPgraphNode * dependencies    = nullptr,
                                    size_t               numDependencies = 0) {
        GGML_ASSERT(hGraph);
        GGML_ASSERT(kernel_ctx->graph);
        RPP_CHECK(rppGraphAddChildGraphNode(&(kernel_ctx->graph_node), hGraph, dependencies, numDependencies,
                                            kernel_ctx->graph));
    }

    ~rpp_node_kernel() {
        // kernel_ctx is shared when nodes are reused across graphs; destroy only once
        // on the last owner to avoid double-destroying graph/event handles.
        if (kernel_ctx && kernel_ctx.use_count() == 1) {
            rpp_destroy_kernel_ctx(*(kernel_ctx.get()));
        }
    }
};

struct rpp_kernel_cgraph {
    RPPgraph                     graph{ nullptr };      // RPP graph describing kernel + DMA ops
    RPPgraphExec                 graphexec{ nullptr };  // Executable graph (created after graph instantiation)
    std::vector<ggml_rpp_node *> rpp_nodes;             // RPP nodes
    std::unordered_map<ggml_rpp_node *, RPPgraphNode> rpp_to_graph_nodes;  // RPP node to graph node map

    bool is_shared{ true };

    // Wrap an already-instantiated per-op graph.
    explicit rpp_kernel_cgraph(ggml_rpp_node * rpp_node) : is_shared(false) {
        GGML_ASSERT(rpp_node);
        rpp_nodes.emplace_back(rpp_node);
        rpp_to_graph_nodes[rpp_node] = static_cast<rpp_node_kernel *>(rpp_node)->kernel_ctx->graph_node;
    }

    explicit rpp_kernel_cgraph() : is_shared(true) { RPP_CHECK(rppGraphCreate(&graph, RPP_GRAPH_NON_BLOCKING)); }

    ~rpp_kernel_cgraph() {
        if (!is_shared) {
            return;
        }
        if (graphexec != nullptr) {
            RPP_CHECK(rppGraphExecDestroy(graphexec));
            graphexec = nullptr;
        }
        if (graph != nullptr) {
            RPP_CHECK(rppGraphDestroy(graph));
            graph = nullptr;
        }
    }

    void add_child_graph(ggml_rpp_node * rpp_node);

    void update_child_graph(std::unordered_map<ggml_tensor *, ggml_rpp_node *> & cur_rpp_nodes);

    void graph_instantiate();

    void graph_launch(ggml_backend_rpp_context & ctx, rtStream_t stream, int trace_id);
};

struct ggml_rpp_cgraph {
    std::unordered_map<ggml_tensor *, std::vector<std::unique_ptr<ggml_rpp_node>>> rpp_nodes;
    std::unordered_map<int, std::vector<std::unique_ptr<rpp_kernel_cgraph>>>       rpp_kernel_graphs;
    std::unordered_set<ggml_tensor *>                  nodes_all;  //not include reshape，view etc.
    std::unordered_set<ggml_tensor *>                  nodes_i;
    std::unordered_set<ggml_tensor *>                  nodes_o;
    std::unordered_set<ggml_tensor *>                  nodes_matmul_weight;
    std::unordered_set<ggml_tensor *>                  nodes_mul_weight;
    std::unordered_set<ggml_tensor *>                  nodes_cache_kv;
    ggml_cgraph *                                      ggml_graph{ nullptr };
    std::unordered_map<ggml_tensor *, ggml_rpp_node *> cur_rpp_nodes;
    std::vector<ggml_graph_node_properties>            ggml_graph_properties;

    std::vector<rpp_kernel_cgraph *>                             rpp_in_use_kernel_graphs;
    std::vector<ggml_rpp_node *>                                 rpp_in_use_nodes;
    std::vector<std::function<bool(ggml_backend_rpp_context &)>> launch_funcs;  // for launch funcs before graph launch

    template <typename Fn, typename... Args> void add_launch_func(Fn && fn, Args &&... args) {
        auto fn_args = std::make_tuple(std::forward<Args>(args)...);
        launch_funcs.emplace_back([func    = std::forward<Fn>(fn),
                                   fn_args = std::move(fn_args)](ggml_backend_rpp_context & ctx) mutable -> bool {
            return std::apply(
                [&](auto &&... unpacked) -> bool {
                    return std::invoke(func, ctx, std::forward<decltype(unpacked)>(unpacked)...);
                },
                fn_args);
        });
    }

    bool run_launch_funcs(ggml_backend_rpp_context & ctx) {
        for (auto & launch_func : launch_funcs) {
            if (!launch_func(ctx)) {
                return false;
            }
        }
        return true;
    }

    explicit ggml_rpp_cgraph(ggml_cgraph * graph) : ggml_graph(graph) {}
};

enum rpp_backend_domain {
    RPP_DOMAIN_UNKNOWN = 0,
    RPP_DOMAIN_TEXT    = 1,
    RPP_DOMAIN_VISION  = 2,
    RPP_DOMAIN_AUDIO   = 3,
};

struct ggml_backend_rpp_context {
    rpp_backend_domain domain{ RPP_DOMAIN_TEXT };
    int                device;
    std::string        name;
    rtEvent_t          copy_event = nullptr;
    bool               use_ubatch{ GGML_RPP_USE_UBATCH };
    bool               use_bf16{ GGML_RPP_USE_BF16 };
    uint32_t           n_ubatch{ 512 };
    uint32_t           n_max_ctx{ 8192 };
    uint32_t           trace_id{ 0 };
    uint64_t           trace_num{ 0 };  // trace number, used for trace enable and disable
    uint32_t           stub_kv_step{ 8 };

    rtStream_t streams[GGML_RPP_MAX_DEVICES][GGML_RPP_MAX_STREAMS] = { { nullptr } };

    std::unordered_map<ggml_cgraph *, std::unique_ptr<ggml_rpp_cgraph>> rpp_graphs;
    std::vector<ggml_cgraph *>                                          gglm_graphs;

    std::unordered_map<ggml_tensor *, void *>                     rpp_io_buffers;
    ggml_rpp_cgraph *                                             cur_rpp_graph{ nullptr };
    std::unordered_map<ggml_tensor *, ggml_graph_node_properties> ggml_first_properties;

    void * sin_cache{ nullptr };
    void * cos_cache{ nullptr };

    explicit ggml_backend_rpp_context(int device, const char * params) :
        device(device),
        name(GGML_RPP_NAME + std::to_string(device)),
        trace_id(0) {
#if GGML_RPP_PERF_TRACE
        trace_id = TRACE_START("ggml_backend_rpp_context");
        rppLogsDumpToTraceWindows(trace_id, 0);
        TRACE_DISABLE(trace_id);
#endif
        if (getenv("GGML_RPP_BATCH_SIZE") != nullptr) {
            n_ubatch = atoi(getenv("GGML_RPP_BATCH_SIZE"));
        }
        if (getenv("GGML_RPP_MAX_CONTEXT") != nullptr) {
            n_max_ctx = atoi(getenv("GGML_RPP_MAX_CONTEXT"));
        }
        if (getenv("GGML_RPP_STUB_KV_STEP") != nullptr) {
            stub_kv_step = atoi(getenv("GGML_RPP_STUB_KV_STEP"));
        }
    }

    ~ggml_backend_rpp_context();

    rtStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_rpp_set_device(device);
            RPP_CHECK(rtStreamCreateWithFlags(&streams[device][stream], rtStreamNonBlocking));
        }
        return streams[device][stream];
    }

    rtStream_t stream() { return stream(device, 0); }

    // pool in device
    std::unique_ptr<ggml_rpp_pool>        pools[GGML_RPP_MAX_DEVICES];
    static std::unique_ptr<ggml_rpp_pool> new_pool_for_device(int device);

    ggml_rpp_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(device);
        }
        return *pools[device];
    }

    ggml_rpp_pool & pool() { return pool(device); }

    // pool in device legacy
    std::unique_ptr<ggml_rpp_pool>        pools_leg[GGML_RPP_MAX_DEVICES];
    static std::unique_ptr<ggml_rpp_pool> new_pool_for_device_leg(int device);

    ggml_rpp_pool & pool_leg(int device) {
        if (pools_leg[device] == nullptr) {
            pools_leg[device] = new_pool_for_device_leg(device);
        }
        return *pools_leg[device];
    }

    ggml_rpp_pool & pool_leg() { return pool_leg(device); }

    std::unique_ptr<ggml_rpp_pool>        pools_mem[GGML_RPP_MAX_DEVICES];
    static std::unique_ptr<ggml_rpp_pool> new_pool_for_device_mem(int device);

    ggml_rpp_pool & pool_mem(int device) {
        if (pools_mem[device] == nullptr) {
            pools_mem[device] = new_pool_for_device_mem(device);
        }
        return *pools_mem[device];
    }

    ggml_rpp_pool & pool_mem() { return pool_mem(device); }

    // pool in host
    std::unique_ptr<ggml_rpp_pool>        pools_host[GGML_RPP_MAX_DEVICES];
    static std::unique_ptr<ggml_rpp_pool> new_pool_for_host(int device);

    ggml_rpp_pool & pool_host(int device) {
        if (pools_host[device] == nullptr) {
            pools_host[device] = new_pool_for_host(device);
        }
        return *pools_host[device];
    }

    ggml_rpp_pool & pool_host() { return pool_host(device); }

    // pool in host legacy
    std::unique_ptr<ggml_rpp_pool>        pools_host_leg[GGML_RPP_MAX_DEVICES];
    static std::unique_ptr<ggml_rpp_pool> new_pool_for_host_leg(int device);

    ggml_rpp_pool & pool_host_leg(int device) {
        if (pools_host_leg[device] == nullptr) {
            pools_host_leg[device] = new_pool_for_host_leg(device);
        }
        return *pools_host_leg[device];
    }

    ggml_rpp_pool & pool_host_leg() { return pool_host_leg(device); }

    std::unique_ptr<ggml_rpp_pool>        pools_host_mem[GGML_RPP_MAX_DEVICES];
    static std::unique_ptr<ggml_rpp_pool> new_pool_for_host_mem(int device);

    ggml_rpp_pool & pool_host_mem(int device) {
        if (pools_host_mem[device] == nullptr) {
            pools_host_mem[device] = new_pool_for_host_mem(device);
        }
        return *pools_host_mem[device];
    }

    ggml_rpp_pool & pool_host_mem() { return pool_host_mem(device); }
};

void ggml_rpp_cpy_dest_ptrs_copy(ggml_rpp_cgraph * rpp_graph,
                                 char **           host_dest_ptrs,
                                 const int         host_dest_ptrs_size,
                                 rtStream_t        stream);

void ggml_rpp_reset_graph(ggml_backend_rpp_context * rpp_ctx, ggml_rpp_cgraph * rpp_graph);
void ggml_rpp_reset_node(ggml_backend_rpp_context * rpp_ctx, ggml_rpp_cgraph * rpp_graph, ggml_tensor * g_tensor);
void ggml_rpp_node_set_properties(ggml_rpp_node * rpp_node, ggml_tensor * dst);
bool ggml_rpp_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties);
bool ggml_rpp_node_has_matching_properties(ggml_tensor * node, ggml_rpp_node * rpp_node);
bool ggml_rpp_backend_buft_is_rpp_split(ggml_backend_buffer_type_t buft);
