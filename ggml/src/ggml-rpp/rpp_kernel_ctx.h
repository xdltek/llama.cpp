#pragma once
#include "ggml-rpp/rpp_dfs.h"
#include "rpp_drv_api.h"

#include <assert.h>
#include <math.h>
#include <rpp_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Kernel execution context holding all RPP runtime objects
typedef struct rpp_kernel_context {
    RPPmodule    rppBinMod{ nullptr };
    RPPgraph     graph{ nullptr };      // RPP graph describing kernel + DMA ops
    RPPgraphExec graphexec{ nullptr };  // Executable graph (created after graph instantiation)
    /** True when graphexec was created with RPP_GRAPH_INSTANTIATE_FLAG_CHILD_EXEC for rppGraphExecUpdateChildGraphExec;
     *  caller must destroy it when the kernel context is torn down (see ggml-rpp update_child_graph). */
    RPPgraphNode graph_node{ nullptr };

    RPPevent kernel_done_ping[2]{ nullptr, nullptr };   // Event: kernel done per ping buffer
    RPPevent dma_aux_done_ping[2]{ nullptr, nullptr };  // Event: aux DMA done per ping buffer
    RPPevent dma_done_ping[2]{ nullptr, nullptr };      // Event: q4/main DMA done per ping buffer
    RPPevent mpu_done_ping[2]{ nullptr, nullptr };      // Event: MPU update done per stage

    RPPstream kernelStream{ nullptr };  // Stream used for kernel execution
    RPPstream dmaStream{ nullptr };     // Stream used for DMA transfers
    RPPstream mpuStream{ nullptr };     // Stream used for MPU / descriptor updates

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
    if (ctx.graphexec != nullptr) {
        (void) rppGraphExecDestroy(ctx.graphexec);
        ctx.graphexec = nullptr;
    }
    // Destroy RPP graph
    // NOTE: other graphexec handles are destroyed elsewhere if created
    if (ctx.graph != nullptr) {
        (void) rppGraphDestroy(ctx.graph);
        ctx.graph = nullptr;
    }

    // Destroy events
    if (ctx.kernel_done_ping[0] != nullptr) {
        (void) rppEventDestroy(ctx.kernel_done_ping[0]);
        ctx.kernel_done_ping[0] = nullptr;
    }
    if (ctx.kernel_done_ping[1] != nullptr) {
        (void) rppEventDestroy(ctx.kernel_done_ping[1]);
        ctx.kernel_done_ping[1] = nullptr;
    }
    if (ctx.dma_aux_done_ping[0] != nullptr) {
        (void) rppEventDestroy(ctx.dma_aux_done_ping[0]);
        ctx.dma_aux_done_ping[0] = nullptr;
    }
    if (ctx.dma_aux_done_ping[1] != nullptr) {
        (void) rppEventDestroy(ctx.dma_aux_done_ping[1]);
        ctx.dma_aux_done_ping[1] = nullptr;
    }
    if (ctx.dma_done_ping[0] != nullptr) {
        (void) rppEventDestroy(ctx.dma_done_ping[0]);
        ctx.dma_done_ping[0] = nullptr;
    }
    if (ctx.dma_done_ping[1] != nullptr) {
        (void) rppEventDestroy(ctx.dma_done_ping[1]);
        ctx.dma_done_ping[1] = nullptr;
    }
    if (ctx.mpu_done_ping[0] != nullptr) {
        (void) rppEventDestroy(ctx.mpu_done_ping[0]);
        ctx.mpu_done_ping[0] = nullptr;
    }
    if (ctx.mpu_done_ping[1] != nullptr) {
        (void) rppEventDestroy(ctx.mpu_done_ping[1]);
        ctx.mpu_done_ping[1] = nullptr;
    }
    // Destroy streams (reverse order is generally safe)
    if (ctx.dmaStream != nullptr) {
        rpp_reset_dfs_state(ctx.dmaStream);
        (void) rppStreamDestroy(ctx.dmaStream);
        ctx.dmaStream = nullptr;
    }
    if (ctx.kernelStream != nullptr) {
        rpp_reset_dfs_state(ctx.kernelStream);
        (void) rppStreamDestroy(ctx.kernelStream);
        ctx.kernelStream = nullptr;
    }
    if (ctx.mpuStream != nullptr) {
        rpp_reset_dfs_state(ctx.mpuStream);
        (void) rppStreamDestroy(ctx.mpuStream);
        ctx.mpuStream = nullptr;
    }

    if (ctx.dev_aux_workspace != 0) {
        rtFree((void *) ctx.dev_aux_workspace);
        ctx.dev_aux_workspace       = 0;
        ctx.dev_aux_workspace_bytes = 0;
    }

    for (auto ptr : ctx.dev_owned) {
        if (ptr != 0) {
            rtFree((void *) ptr);
        }
    }
    ctx.dev_owned.clear();

    // Free virtual SRAM
    RPPdeviceptr sram_alloc_base = ctx.virtual_sram_alloc_base != 0 ? ctx.virtual_sram_alloc_base : ctx.virtual_sram_base;
    if (sram_alloc_base != 0) {
        rtFreeVirtSram((void *) sram_alloc_base);
    }
    ctx.virtual_sram_base       = 0;
    ctx.virtual_sram_alloc_base = 0;

    ctx.dev_in.clear();
    ctx.dev_out.clear();
}
