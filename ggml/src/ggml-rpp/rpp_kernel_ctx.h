#pragma once
#include "ggml-rpp/rpp_dfs.h"
#include "rpp_drv_api.h"

#include <assert.h>
#include <math.h>
#include <rpp_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef GGML_RPP_PERF_TRACE
#    define GGML_RPP_PERF_TRACE 0
#endif

#ifndef TRACE_SCOPE_GUARD
#    if GGML_RPP_PERF_TRACE
#        include "rpp_perf.h"
#        define _RPP_TRACE_CONCAT_INNER(a, b) a##b
#        define _RPP_TRACE_CONCAT(a, b)       _RPP_TRACE_CONCAT_INNER(a, b)

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
#        define TRACE_SCOPE_GUARD(win, name) 0
#    endif
#endif

extern thread_local uint32_t ggml_rpp_trace_id_current;

// Kernel execution context holding all RPP runtime objects
typedef struct rpp_kernel_context {
    RPPmodule    rppBinMod{ nullptr };
    RPPgraph     graph{ nullptr };      // RPP graph describing kernel + DMA ops
    RPPgraphExec graphexec{ nullptr };  // Executable graph (created after graph instantiation)
    /** True when graphexec was created with RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC for rppGraphExecUpdateChildGraphExec;
     *  caller must destroy it when the kernel context is torn down (see ggml-rpp update_child_graph). */
    RPPgraphNode graph_node{ nullptr };

    RPPevent kernel_done_ping[2]{ nullptr, nullptr };        // Event: kernel done per ping buffer
    RPPevent dma_aux_done_ping[2]{ nullptr, nullptr };       // Event: aux DMA done per ping buffer
    RPPevent dma_done_ping[2]{ nullptr, nullptr };           // Event: q4/main DMA done per ping buffer
    RPPevent mpu_done_ping[2]{ nullptr, nullptr };           // Event: MPU update done per stage

    RPPstream kernelStream{ nullptr };                       // Stream used for kernel execution
    RPPstream dmaStream{ nullptr };                          // Stream used for DMA transfers
    RPPstream mpuStream{ nullptr };                          // Stream used for MPU / descriptor updates

    RPPdeviceptr              virtual_sram_base{ 0 };        // Working SRAM base used by kernel builders
    RPPdeviceptr              virtual_sram_alloc_base{ 0 };  // Owning pointer returned by rtMallocVirtSram
    RPPdeviceptr              dev_workspace{ 0 };
    RPPdeviceptr              dev_aux_workspace{ 0 };
    size_t                    dev_aux_workspace_bytes{ 0 };
    // Optional extra device buffers owned by this kernel context (debug / auxiliary paths).
    std::vector<RPPdeviceptr> dev_owned;
    std::vector<RPPdeviceptr> dev_in;
    std::vector<RPPdeviceptr> dev_out;
} rpp_kernel_context;

static inline uint32_t round_up_32(uint32_t x) {
    return (x + 31u) & ~31u;
}

inline void rpp_append_function_args(std::ostringstream &) {}

template <typename T, typename... Args>
inline void rpp_append_function_args(std::ostringstream & oss, const T & value, const Args &... args) {
    oss << ':' << value;
    rpp_append_function_args(oss, args...);
}

template <typename... Args>
inline std::string rpp_join_function_name_and_args(const char * func_name, const Args &... args) {
    std::ostringstream oss;
    oss << (func_name != nullptr ? func_name : "");
    rpp_append_function_args(oss, args...);
    return oss.str();
}

struct rpp_graph_module_cache {
    std::unordered_map<std::string, RPPmodule> graph_modules;

    ~rpp_graph_module_cache() {
        for (auto & item : graph_modules) {
            if (item.second) {
                (void) rppModuleUnload(item.second);
                item.second = nullptr;
            }
        }
    }
};

// Shared KPARA pools for identical graph shapes. All graphexecs that reuse a pool
// must instantiate with is_external=true; the cache owns the pool and frees it once
// at teardown (see rppGraphResourceAlloc / rppGraphResourceFree).
struct rpp_graph_kpara_cache {
    struct entry {
        RPPdeviceptr daddr{ 0 };
        size_t       size{ 0 };
    };

    std::unordered_map<std::string, entry> graph_kparas;

    ~rpp_graph_kpara_cache() {
        for (auto & item : graph_kparas) {
            if (item.second.daddr) {
                (void) rppGraphResourceFree(item.second.daddr, RPP_GRAPH_RESOURCE_KPARA);
                item.second.daddr = 0;
            }
        }
    }
};

inline rpp_graph_module_cache & rpp_graph_module_cache_instance() {
    static rpp_graph_module_cache cache;
    return cache;
}

inline rpp_graph_kpara_cache & rpp_graph_kpara_cache_instance() {
    static rpp_graph_kpara_cache cache;
    return cache;
}

inline RPPresult rpp_module_load_once(RPPmodule & module, const char * module_path) {
    auto & cache = rpp_graph_module_cache_instance().graph_modules;
    auto   iter  = cache.find(module_path);
    if (iter != cache.end() && iter->second) {
        module = iter->second;
        return RPP_SUCCESS;
    }

    RPPmodule loaded = nullptr;
    RPPresult result = rppModuleLoad(&loaded, module_path);
    if (result == RPP_SUCCESS && loaded) {
        cache[module_path] = loaded;
        module             = loaded;
    }
    return result;
}

inline RPPresult rpp_graph_instantiate(RPPgraphExec & graph_exec,
                                       RPPgraph &     hGraph,
                                       const char *   key,
                                       int            is_instantial    = 1,
                                       bool           use_shared_kpara = true) {
    TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rpp_graph_instantiate");
    RPPresult                    result = RPP_SUCCESS;
    RPP_GRAPH_INSTANTIATE_PARAMS params = {};

    // After instantiate the mutable graph is consumed; destroy it and clear the
    // caller's handle so destructors do not double-free the same RPPgraph.
    auto destroy_graph_after_instantiate = [&]() {
        if (hGraph == nullptr) {
            return RPP_SUCCESS;
        }
        TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphDestroy");
        result = rppGraphDestroy(hGraph);
        assert(result == RPP_SUCCESS);
        hGraph = nullptr;
        return result;
    };

    auto finalize_after_instantiate = [&]() {
        TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphExecFinalize");
        result = rppGraphExecFinalize(graph_exec, nullptr);
        assert(result == RPP_SUCCESS);
        return result;
    };

    params.flags = is_instantial == 0 ? RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC : 0;
    if (!use_shared_kpara) {
        // Some graphs encode runtime-specific addresses in kparams, so their params must not be reused.
        {
            TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphInstantiateWithParams");
            result = rppGraphInstantiateWithParams(&graph_exec, hGraph, &params);
            assert(result == RPP_SUCCESS);
        }
        if (destroy_graph_after_instantiate() != RPP_SUCCESS) {
            return result;
        }
        return finalize_after_instantiate();
    }

    auto & cache_kparas = rpp_graph_kpara_cache_instance().graph_kparas;
    auto   iter         = cache_kparas.find(key);
    if (iter == cache_kparas.end()) {
        // Probe once to learn KPARA size, then allocate a caller-owned shared pool.
        // The probe exec must be destroyed before the real instantiate so it does not
        // keep a driver-owned pool that would be double-freed when later shared.
        RPPgraphExec                 probe_exec   = nullptr;
        RPP_GRAPH_INSTANTIATE_PARAMS probe_params = {};
        probe_params.flags                        = params.flags;
        {
            TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphInstantiateWithParams_probe");
            result = rppGraphInstantiateWithParams(&probe_exec, hGraph, &probe_params);
            assert(result == RPP_SUCCESS);
        }
        {
            TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphExecGetParams_probe");
            result = rppGraphExecGetParams(probe_exec, &probe_params);
            assert(result == RPP_SUCCESS);
        }
        {
            TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphExecDestroy_probe");
            result = rppGraphExecDestroy(probe_exec);
            assert(result == RPP_SUCCESS);
            probe_exec = nullptr;
        }

        rpp_graph_kpara_cache::entry shared{};
        shared.size = probe_params.res_kpara.size;
        if (shared.size > 0) {
            TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphResourceAlloc_kpara");
            result = rppGraphResourceAlloc(&shared.daddr, shared.size, RPP_GRAPH_RESOURCE_KPARA);
            assert(result == RPP_SUCCESS);
        }
        cache_kparas.emplace(key, shared);
        iter = cache_kparas.find(key);
    }

    // Every consumer (including the first real exec) uses the shared external pool.
    params                       = {};
    params.flags                 = is_instantial == 0 ? RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC : 0;
    params.res_kpara.is_external = true;
    params.res_kpara.type        = RPP_GRAPH_RESOURCE_KPARA;
    params.res_kpara.daddr       = iter->second.daddr;
    params.res_kpara.size        = iter->second.size;
    {
        TRACE_SCOPE_GUARD(ggml_rpp_trace_id_current, "rppGraphInstantiateWithParams_shared_kpara");
        result = rppGraphInstantiateWithParams(&graph_exec, hGraph, &params);
        assert(result == RPP_SUCCESS);
    }
    if (destroy_graph_after_instantiate() != RPP_SUCCESS) {
        return result;
    }
    return finalize_after_instantiate();
}

// ------------------------------------------------------------
// Initialize kernel context
// Allocate resources required for RPP kernel execution
// ------------------------------------------------------------
inline void rpp_init_kernel_ctx(rpp_kernel_context & ctx) {
    //rppModuleLoad(&ctx.rppBinMod, "memcpy_align.o");
    // Allocate virtual SRAM for kernel / DMA usage
    // Size: 22MB (adjust based on kernel requirements)
    rtMallocVirtSram((void **) &ctx.virtual_sram_alloc_base, 22 * 1024 * 1024);
    ctx.virtual_sram_base = ctx.virtual_sram_alloc_base;

    // Create synchronization events
    rppEventCreate(&ctx.kernel_done_ping[0], 0);
    rppEventCreate(&ctx.kernel_done_ping[1], 0);
    rppEventCreate(&ctx.dma_aux_done_ping[0], 0);
    rppEventCreate(&ctx.dma_aux_done_ping[1], 0);
    rppEventCreate(&ctx.dma_done_ping[0], 0);
    rppEventCreate(&ctx.dma_done_ping[1], 0);
    rppEventCreate(&ctx.mpu_done_ping[0], 0);
    rppEventCreate(&ctx.mpu_done_ping[1], 0);
    // Create streams
    rppStreamCreate(&ctx.kernelStream, 0);  // Stream dedicated to kernel execution
    rppStreamCreate(&ctx.dmaStream, 0);     // Stream dedicated to DMA transfers
    rppStreamCreate(&ctx.mpuStream, 0);     // Stream dedicated to MPU / descriptor updates
    rpp_reset_dfs_state(ctx.kernelStream);
    rpp_reset_dfs_state(ctx.dmaStream);
    rpp_reset_dfs_state(ctx.mpuStream);
    // Create an empty RPP graph
    // Nodes (kernel / DMA) will be added later
    rppGraphCreate(&ctx.graph, RPP_GRAPH_NON_BLOCKING);
    ctx.dev_workspace           = 0;
    ctx.dev_aux_workspace       = 0;
    ctx.dev_aux_workspace_bytes = 0;
    ctx.dev_owned.clear();
    ctx.dev_in.clear();
    ctx.dev_out.clear();
}

// ------------------------------------------------------------
// Destroy kernel context
// Release all RPP runtime resources
// ------------------------------------------------------------
inline void rpp_destroy_kernel_ctx(rpp_kernel_context & ctx) {
    if (ctx.graphexec) {
        (void) rppGraphExecDestroy(ctx.graphexec);
        ctx.graphexec = nullptr;
    }
    // Destroy RPP graph
    // NOTE: other graphexec handles are destroyed elsewhere if created
    if (ctx.graph) {
        (void) rppGraphDestroy(ctx.graph);
        ctx.graph = nullptr;
    }

    // Destroy events
    if (ctx.kernel_done_ping[0]) {
        (void) rppEventDestroy(ctx.kernel_done_ping[0]);
        ctx.kernel_done_ping[0] = nullptr;
    }
    if (ctx.kernel_done_ping[1]) {
        (void) rppEventDestroy(ctx.kernel_done_ping[1]);
        ctx.kernel_done_ping[1] = nullptr;
    }
    if (ctx.dma_aux_done_ping[0]) {
        (void) rppEventDestroy(ctx.dma_aux_done_ping[0]);
        ctx.dma_aux_done_ping[0] = nullptr;
    }
    if (ctx.dma_aux_done_ping[1]) {
        (void) rppEventDestroy(ctx.dma_aux_done_ping[1]);
        ctx.dma_aux_done_ping[1] = nullptr;
    }
    if (ctx.dma_done_ping[0]) {
        (void) rppEventDestroy(ctx.dma_done_ping[0]);
        ctx.dma_done_ping[0] = nullptr;
    }
    if (ctx.dma_done_ping[1]) {
        (void) rppEventDestroy(ctx.dma_done_ping[1]);
        ctx.dma_done_ping[1] = nullptr;
    }
    if (ctx.mpu_done_ping[0]) {
        (void) rppEventDestroy(ctx.mpu_done_ping[0]);
        ctx.mpu_done_ping[0] = nullptr;
    }
    if (ctx.mpu_done_ping[1]) {
        (void) rppEventDestroy(ctx.mpu_done_ping[1]);
        ctx.mpu_done_ping[1] = nullptr;
    }
    // Destroy streams (reverse order is generally safe)
    if (ctx.dmaStream) {
        rpp_reset_dfs_state(ctx.dmaStream);
        (void) rppStreamDestroy(ctx.dmaStream);
        ctx.dmaStream = nullptr;
    }
    if (ctx.kernelStream) {
        rpp_reset_dfs_state(ctx.kernelStream);
        (void) rppStreamDestroy(ctx.kernelStream);
        ctx.kernelStream = nullptr;
    }
    if (ctx.mpuStream) {
        rpp_reset_dfs_state(ctx.mpuStream);
        (void) rppStreamDestroy(ctx.mpuStream);
        ctx.mpuStream = nullptr;
    }

    if (ctx.dev_aux_workspace) {
        rtFree((void *) ctx.dev_aux_workspace);
        ctx.dev_aux_workspace       = 0;
        ctx.dev_aux_workspace_bytes = 0;
    }

    if (ctx.dev_workspace) {
        ctx.dev_workspace = 0;
    }

    for (auto ptr : ctx.dev_owned) {
        if (ptr) {
            rtFree((void *) ptr);
            ptr = 0;
        }
    }
    ctx.dev_owned.clear();

    // Free virtual SRAM
    RPPdeviceptr sram_alloc_base =
        ctx.virtual_sram_alloc_base != 0 ? ctx.virtual_sram_alloc_base : ctx.virtual_sram_base;
    if (sram_alloc_base != 0) {
        rtFreeVirtSram((void *) sram_alloc_base);
    }
    ctx.virtual_sram_base       = 0;
    ctx.virtual_sram_alloc_base = 0;

    ctx.dev_in.clear();
    ctx.dev_out.clear();
}
