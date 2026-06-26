#ifndef RPP_SOFT_MAX
#define RPP_SOFT_MAX

#include "ggml-rpp/rpp_common.h"
#include "ggml-rpp/rpp_ops_utils.h"

#include <functional>
#include <unordered_set>

#if GGML_RPP_USE_RT

bool ggml_rpp_op_openrt_soft_max(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst);

struct rpp_openrt_soft_max : public rpp_node_openrt {
    explicit rpp_openrt_soft_max(ggml_tensor * tensor) : rpp_node_openrt(tensor) {}

    explicit rpp_openrt_soft_max(ggml_tensor * tensor, ggml_rpp_node * rpp_node) : rpp_node_openrt(tensor, rpp_node) {}

    ~rpp_openrt_soft_max() {
        if (input1 && input1->data) {
            RPP_CHECK(rtFree(input1->data));
        }
    }

    void init_softmax_input_tensor(ggml_tensor * dst) {
        ggml_tensor * src0 = dst->src[0];
        ggml_tensor * src1 = dst->src[1];

        input1 = std::make_shared<ggml_tensor>();
        for (size_t i = 0; i < 4; i++) {
            input1->ne[i] = src0->ne[i];
            input1->nb[i] = src0->nb[i];
        }

        infer1::Dims dims    = this->engine->getBindingDimensions(0);
        size_t       io_size = ggml_rpp_nbytes(dims, ggml_type_size(src0->type));
        rtMallocHost(&input1_data, io_size);
        input1->data = nullptr;
        RPP_CHECK(rtMalloc(&(input1->data), io_size));
        void * src1_data = input1_data;
        RPP_CHECK(rtMemcpy(src1_data, src1->data, io_size, rtMemcpyDeviceToHost));

        int element_num = 1;
        for (int i = 0; i < dims.nbDims; i++) {
            element_num *= dims.d[i];
        }

        // att
        std::vector<float> att_vec(element_num, -INFINITY);
        float *            mp_f32 = nullptr;
        size_t             offset = 0;
        for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += 1) {
                mp_f32 = src1 ? (float *) ((char *) src1_data + i01 * src1->nb[1]) : NULL;
                // att_vec.insert(att_vec.end(), mp_f32, mp_f32 + 256);
                std::copy(mp_f32, mp_f32 + src0->ne[0], att_vec.begin() + offset);
                offset += src0->ne[0];
            }
        }
        RPP_CHECK(rtMemcpy(input1->data, att_vec.data(), io_size, rtMemcpyHostToDevice));
    }

    bool rpp_dispatch_func(ggml_backend_rpp_context & ctx,
                           ggml_tensor *              dst,
                           int                        is_instantial = 1,
                           int                        is_launch     = 1) override {
        return ggml_rpp_op_openrt_soft_max(ctx, dst);
    }

    std::shared_ptr<ggml_tensor> input1{ nullptr };
    void *                       input1_data{ nullptr };
    static uint32_t              trace_id;
};

#endif

inline bool ggml_rpp_op_soft_max(ggml_backend_rpp_context & ctx, struct ggml_tensor * dst) {
#if GGML_RPP_USE_RT
    return ggml_rpp_op_openrt_soft_max(ctx, dst);
#else
    GGML_LOG_ERROR("%s: RPP backend only supports Linux and Windows systems.", __func__);
    return false;
#endif
}

#endif
