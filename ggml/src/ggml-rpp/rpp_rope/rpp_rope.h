#ifndef RPP_ROPE
#define RPP_ROPE

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

#if GGML_RPP_USE_RT

/**
 * @brief   Rope operator in rpp 
 *
 * @details The rope operator runs on RPP. Due to the inconsistent input of the rope operator in RPPRT and GGML, 
 *          the CPU needs to generate the corresponding values of sin and cos. In one layer of the network, 
 *          two rope operators are constructed, which can be shared with other layers. Their sin and cos values 
 *          need to be updated each time a token is produced
 *
 * @param ctx The context of rpp
 * @param tensor Pointer to the target ggml_tensor object.
 */
bool ggml_rpp_op_openrt_rope(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

struct rpp_openrt_rope : public rpp_node_openrt {
    explicit rpp_openrt_rope(ggml_tensor * tensor) : rpp_node_openrt(tensor) {}

    explicit rpp_openrt_rope(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_openrt(tensor, rpp_node) {}

    ~rpp_openrt_rope() {
        if (!ori_rpp_node && ggml_sin && ggml_cos) {
            rtFree(ggml_sin->data);
            ggml_sin->data = nullptr;
            rtFree(ggml_cos->data);
            ggml_cos->data = nullptr;
            rtFreeHost(sin_data);
            ggml_sin->data = sin_data = nullptr;
            rtFreeHost(cos_data);
            ggml_cos->data = cos_data = nullptr;
        }
    }

    void init_sincos_tensors() {
        // create sin
        ggml_sin        = std::make_shared<ggml_tensor>();
        ggml_sin->ne[0] = cur_ggml_tensor->ne[0];
        // this dim is seq len,  for padding
        ggml_sin->ne[1] = n_ubatch == 1 ? cur_ggml_tensor->ne[2] : n_ubatch;
        ggml_sin->ne[2] = cur_ggml_tensor->ne[3];
        ggml_sin->ne[3] = 1;
        ggml_sin->type  = cur_ggml_tensor->src[0]->type;

        size_t type_size_sin = ggml_type_size(ggml_sin->type);
        size_t blck_size_sin = ggml_blck_size(ggml_sin->type);
        ggml_sin->nb[0]      = type_size_sin;
        ggml_sin->nb[1]      = ggml_sin->nb[0] * (ggml_sin->ne[0] / blck_size_sin);
        ggml_sin->nb[2]      = ggml_sin->nb[1] * ggml_sin->ne[1];
        ggml_sin->nb[3]      = ggml_sin->nb[2] * ggml_sin->ne[2];

        rtMallocHost(&sin_data, ggml_nelements(ggml_sin.get()) * ggml_type_size(ggml_sin->type));
        ggml_sin->data = nullptr;
        RPP_CHECK(rtMalloc(&(ggml_sin->data), ggml_nelements(ggml_sin.get()) * ggml_type_size(ggml_sin->type)));

        // create cos
        ggml_cos        = std::make_shared<ggml_tensor>();
        ggml_cos->ne[0] = cur_ggml_tensor->ne[0];
        // this dim is seq len,  for padding
        ggml_cos->ne[1] = n_ubatch == 1 ? cur_ggml_tensor->ne[2] : n_ubatch;
        ggml_cos->ne[2] = cur_ggml_tensor->ne[3];
        ggml_cos->ne[3] = 1;
        ggml_cos->type  = cur_ggml_tensor->src[0]->type;

        size_t type_size_cos = ggml_type_size(ggml_cos->type);
        size_t blck_size_cos = ggml_blck_size(ggml_cos->type);
        ggml_cos->nb[0]      = type_size_cos;
        ggml_cos->nb[1]      = ggml_cos->nb[0] * (ggml_cos->ne[0] / blck_size_cos);
        ggml_cos->nb[2]      = ggml_cos->nb[1] * ggml_cos->ne[1];
        ggml_cos->nb[3]      = ggml_cos->nb[2] * ggml_cos->ne[2];

        rtMallocHost(&cos_data, ggml_nelements(ggml_cos.get()) * ggml_type_size(ggml_cos->type));
        ggml_cos->data = nullptr;
        RPP_CHECK(rtMalloc(&(ggml_cos->data), ggml_nelements(ggml_cos.get()) * ggml_type_size(ggml_cos->type)));
    }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_openrt_rope(ctx, dst);
    }

    std::shared_ptr<ggml_tensor> ggml_sin{ nullptr };
    std::shared_ptr<ggml_tensor> ggml_cos{ nullptr };

    void * sin_data{ nullptr };
    void * cos_data{ nullptr };
};
#endif

/**
 * @brief   Rope operator in rpp 
 *
 * @details The rope operator runs on RPP. Due to the inconsistent input of the rope operator in RPPRT and GGML, 
 *          the CPU needs to generate the corresponding values of sin and cos. In one layer of the network, 
 *          two rope operators are constructed, which can be shared with other layers. Their sin and cos values 
 *          need to be updated each time a token is produced
 *
 * @param ctx The context of rpp
 * @param tensor Pointer to the target ggml_tensor object.
 */
bool ggml_rpp_op_kernel_rope(ggml_backend_rpp_context & ctx,
                             struct ggml_tensor *       dst,
                             int                        is_instantial = 1,
                             int                        is_launch     = 1);

struct rpp_kernel_rope : public rpp_node_kernel {
    explicit rpp_kernel_rope(ggml_tensor * tensor) : rpp_node_kernel(tensor) {}

    explicit rpp_kernel_rope(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_kernel(tensor, rpp_node) {}

    ~rpp_kernel_rope() {
        if (!ori_rpp_node && ggml_sin && ggml_cos) {
            rtFree(ggml_sin->data);
            ggml_sin->data = nullptr;
            rtFree(ggml_cos->data);
            ggml_cos->data = nullptr;
            rtFreeHost(sin_data);
            ggml_sin->data = sin_data = nullptr;
            rtFreeHost(cos_data);
            ggml_cos->data = cos_data = nullptr;
            rtFreeHost(start_pos_data);

            if (io_update_kernel_ctx) {
                rpp_destroy_kernel_ctx(*io_update_kernel_ctx.get());
                io_update_kernel_ctx.reset();
            }
            if (start_pos_cdma_desc) {
                RPP_CHECK(rppGraphResourceFree(start_pos_cdma_desc, RPP_GRAPH_RESOURCE_CDMA_DESC));
                start_pos_cdma_desc = 0;
            }
        }
    }

    void init_sincos_tensors() {
        // create sin
        ggml_sin        = std::make_shared<ggml_tensor>();
        ggml_sin->ne[0] = cur_ggml_tensor->ne[0];
        // this dim is seq len,  for padding
        ggml_sin->ne[1] = n_ubatch == 1 ? cur_ggml_tensor->ne[2] : n_ubatch;
        ggml_sin->ne[2] = cur_ggml_tensor->ne[3];
        ggml_sin->ne[3] = 1;
        ggml_sin->type  = cur_ggml_tensor->src[0]->type;

        size_t type_size_sin = ggml_type_size(ggml_sin->type);
        size_t blck_size_sin = ggml_blck_size(ggml_sin->type);
        ggml_sin->nb[0]      = type_size_sin;
        ggml_sin->nb[1]      = ggml_sin->nb[0] * (ggml_sin->ne[0] / blck_size_sin);
        ggml_sin->nb[2]      = ggml_sin->nb[1] * ggml_sin->ne[1];
        ggml_sin->nb[3]      = ggml_sin->nb[2] * ggml_sin->ne[2];

        rtMallocHost(&sin_data, ggml_nelements(ggml_sin.get()) * ggml_type_size(ggml_sin->type));
        ggml_sin->data = nullptr;
        RPP_CHECK(rtMalloc(&(ggml_sin->data), ggml_nelements(ggml_sin.get()) * ggml_type_size(ggml_sin->type)));

        // create cos
        ggml_cos        = std::make_shared<ggml_tensor>();
        ggml_cos->ne[0] = cur_ggml_tensor->ne[0];
        // this dim is seq len,  for padding
        ggml_cos->ne[1] = n_ubatch == 1 ? cur_ggml_tensor->ne[2] : n_ubatch;
        ggml_cos->ne[2] = cur_ggml_tensor->ne[3];
        ggml_cos->ne[3] = 1;
        ggml_cos->type  = cur_ggml_tensor->src[0]->type;

        size_t type_size_cos = ggml_type_size(ggml_cos->type);
        size_t blck_size_cos = ggml_blck_size(ggml_cos->type);
        ggml_cos->nb[0]      = type_size_cos;
        ggml_cos->nb[1]      = ggml_cos->nb[0] * (ggml_cos->ne[0] / blck_size_cos);
        ggml_cos->nb[2]      = ggml_cos->nb[1] * ggml_cos->ne[1];
        ggml_cos->nb[3]      = ggml_cos->nb[2] * ggml_cos->ne[2];

        rtMallocHost(&cos_data, ggml_nelements(ggml_cos.get()) * ggml_type_size(ggml_cos->type));
        ggml_cos->data = nullptr;
        RPP_CHECK(rtMalloc(&(ggml_cos->data), ggml_nelements(ggml_cos.get()) * ggml_type_size(ggml_cos->type)));

        rtMallocHost(&start_pos_data, ggml_type_size(cur_ggml_tensor->src[1]->type));
    }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_kernel_rope(ctx, dst, is_instantial, is_launch);
    }

    void add_to_parent_node(RPPgraph             hGraph,
                            const RPPgraphNode * dependencies    = nullptr,
                            size_t               numDependencies = 0) override {
        GGML_ASSERT(hGraph);
        GGML_ASSERT(kernel_ctx->graph);
        if (ori_rpp_node) {
            RPP_CHECK(rppGraphAddChildGraphNode(&(kernel_ctx->graph_node), hGraph, dependencies, numDependencies,
                                                kernel_ctx->graph));
            return;
        }
        RPP_CHECK(rppGraphAddChildGraphNode(&(io_update_kernel_ctx->graph_node), hGraph, dependencies, numDependencies,
                                            io_update_kernel_ctx->graph));
        RPP_CHECK(rppGraphAddChildGraphNode(&(kernel_ctx->graph_node), hGraph, dependencies, numDependencies,
                                            kernel_ctx->graph));
    }

    std::shared_ptr<ggml_tensor> ggml_sin{ nullptr };
    std::shared_ptr<ggml_tensor> ggml_cos{ nullptr };

    void * sin_data{ nullptr };
    void * cos_data{ nullptr };
    void * start_pos_data{ nullptr };

    std::shared_ptr<rpp_kernel_context> io_update_kernel_ctx{ nullptr };
    RPPdeviceptr                        start_pos_cdma_desc{ 0 };
};

inline bool ggml_rpp_op_rope(ggml_backend_rpp_context & ctx,
                             struct ggml_tensor *       dst,
                             int                        is_instantial = 1,
                             int                        is_launch     = 1) {
    return ggml_rpp_op_kernel_rope(ctx, dst, is_instantial, is_launch);
}

#endif
