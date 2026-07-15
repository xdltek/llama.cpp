#include "unary-ops.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>

static inline float op_abs(float x) {
    return fabsf(x);
}

static inline float op_sgn(float x) {
    return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

static inline float op_neg(float x) {
    return -x;
}

static inline float op_step(float x) {
    return (x > 0.f) ? 1.f : 0.f;
}

static inline float op_tanh(float x) {
    return tanhf(x);
}

static inline float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static inline float op_relu(float x) {
    return (x > 0.f) ? x : 0.f;
}

static inline float op_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

static inline float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_exp(float x) {
    return expf(x);
}

static inline float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_sqr(float x) {
    return x * x;
}

static inline float op_sqrt(float x) {
    return sqrtf(x);
}

static inline float op_xielu(float x, float alpha_n, float alpha_p, float beta, float eps) {
    if (x > 0.0f) {
        return alpha_p * x * x + beta * x;
    } else {
        const float min_x_eps = fminf(x, eps);
        return (expm1f(min_x_eps) - x) * alpha_n + beta * x;
    }
}

static inline float op_sin(float x) {
    return sinf(x);
}

static inline float op_cos(float x) {
    return cosf(x);
}

static inline float op_log(float x) {
    return logf(x);
}

static inline float op_expm1(float x) {
    return expf(x) - 1.0f;
}

static inline float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static inline float op_floor(float x) {
    return floorf(x);
}

static inline float op_ceil(float x) {
    return ceilf(x);
}

static inline float op_round(float x) {
    return roundf(x);
}

static inline float op_trunc(float x) {
    return truncf(x);
}

struct cpu_tanh_dump_info {
    int         id;
    std::string data_path;
};

static std::mutex                                      g_cpu_tanh_dump_mutex;
static std::atomic<int>                                g_cpu_tanh_dump_next_id{0};
static std::unordered_map<const ggml_tensor *, cpu_tanh_dump_info> g_cpu_tanh_dump_infos;

static std::string cpu_tanh_dump_sanitize_name(const char * name) {
    std::string result = name && name[0] ? name : "unnamed";
    for (char & c : result) {
        if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') &&
            c != '_' && c != '-' && c != '.') {
            c = '_';
        }
    }
    return result;
}

static bool cpu_tanh_dump_get_info(const ggml_tensor * dst, cpu_tanh_dump_info & info) {
    const char * dump_dir = std::getenv("GGML_CPU_TANH_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0') {
        return false;
    }

    const char * filter = std::getenv("GGML_CPU_TANH_DUMP_FILTER");
    if (filter && filter[0] && std::strstr(dst->name, filter) == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_cpu_tanh_dump_mutex);

    auto it = g_cpu_tanh_dump_infos.find(dst);
    if (it != g_cpu_tanh_dump_infos.end()) {
        info = it->second;
        return true;
    }

    mkdir(dump_dir, 0777);

    const int id = g_cpu_tanh_dump_next_id.fetch_add(1);
    const std::string name = cpu_tanh_dump_sanitize_name(dst->name);
    const std::string prefix = std::string(dump_dir) + "/cpu_tanh_" + std::to_string(id) + "_" + name;
    const std::string data_path = prefix + ".f32.bin";
    const std::string meta_path = prefix + ".meta";

    {
        FILE * fp = std::fopen(data_path.c_str(), "wb");
        if (fp == nullptr) {
            fprintf(stderr, "%s: failed to open %s\n", __func__, data_path.c_str());
            return false;
        }
        const int64_t nbytes = ggml_nelements(dst) * (int64_t) sizeof(float);
        if (nbytes > 0) {
            std::fseek(fp, nbytes - 1, SEEK_SET);
            std::fputc(0, fp);
        }
        std::fclose(fp);
    }

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

    auto inserted = g_cpu_tanh_dump_infos.emplace(dst, cpu_tanh_dump_info{id, data_path});
    fprintf(stderr, "GGML_CPU_TANH_DUMP: #%d %s -> %s\n", id, dst->name, data_path.c_str());
    info = inserted.first->second;
    return true;
}

template <typename dst_t>
static void cpu_tanh_dump_rows_as_f32_typed(
        const ggml_compute_params * params,
        const ggml_tensor * dst,
        const cpu_tanh_dump_info * info) {
    constexpr auto dst_to_f32 = type_conversion_table<dst_t>::to_f32;

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    GGML_ASSERT(nb0 == sizeof(dst_t));

    const auto [ir0, ir1] = get_thread_range(params, dst);
    if (ir0 >= ir1) {
        return;
    }

    FILE * fp = std::fopen(info->data_path.c_str(), "r+b");
    if (fp == nullptr) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, info->data_path.c_str());
        return;
    }

    std::vector<float> row(ne0);
    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne2*ne1);
        const int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        const int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        const dst_t * dst_ptr = (const dst_t *) ((const char *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);
        for (int64_t i00 = 0; i00 < ne0; ++i00) {
            row[i00] = dst_to_f32(dst_ptr[i00]);
        }

        const int64_t offset = ir * ne0 * (int64_t) sizeof(float);
        std::fseek(fp, offset, SEEK_SET);
        std::fwrite(row.data(), sizeof(float), ne0, fp);
    }

    std::fclose(fp);
}

static void cpu_tanh_dump_rows_as_f32(const ggml_compute_params * params, const ggml_tensor * dst) {
    cpu_tanh_dump_info info;
    if (!cpu_tanh_dump_get_info(dst, info)) {
        return;
    }

    switch (dst->type) {
        case GGML_TYPE_F32:
            cpu_tanh_dump_rows_as_f32_typed<float>(params, dst, &info);
            break;
        case GGML_TYPE_F16:
            cpu_tanh_dump_rows_as_f32_typed<ggml_fp16_t>(params, dst, &info);
            break;
        case GGML_TYPE_BF16:
            cpu_tanh_dump_rows_as_f32_typed<ggml_bf16_t>(params, dst, &info);
            break;
        default:
            fprintf(stderr, "%s: unsupported tanh dump type: %s\n", __func__, ggml_type_name(dst->type));
            break;
    }
}

static void cpu_tanh_log(const ggml_compute_params * params, const ggml_tensor * dst) {
    if (params->ith != 0) {
        return;
    }

    const char * enabled = std::getenv("GGML_CPU_TANH_LOG");
    if (enabled == nullptr || enabled[0] == '\0' || std::strcmp(enabled, "0") == 0) {
        return;
    }

    const ggml_tensor * src0 = dst->src[0];

    char log_buf[1024];
    std::sprintf(log_buf,
            "GGML_CPU_TANH_LOG: name=%s op=%s input_shape=[%lld,%lld,%lld,%lld] "
            "output_shape=[%lld,%lld,%lld,%lld] input_dtype=%s output_dtype=%s\n",
            dst->name,
            ggml_unary_op_name(ggml_get_unary_op(dst)),
            (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
            (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3],
            ggml_type_name(src0->type),
            ggml_type_name(dst->type));
    std::fputs(log_buf, stderr);
}

template <float (*op)(float), typename src0_t, typename dst_t>
static inline void vec_unary_op(int64_t n, dst_t * y, const src0_t * x) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float)>
static void unary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

template <float (*op)(float, ggml_tensor *)>
static void unary_op_params(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_bf16_t, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, ggml_fp16_t, float>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static inline void vec_unary_op_functor(int64_t n, dst_t * y, const src0_t * x, Op op) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static void apply_unary_op_functor(const ggml_compute_params * params, ggml_tensor * dst, Op op) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op_functor(ne0, dst_ptr, src0_ptr, op);
    }
}

// Generic dispatcher for functors
template <typename Op>
static void unary_op_functor(const ggml_compute_params * params, ggml_tensor * dst, Op op) {
    const ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op_functor<Op, float, float>(params, dst, op);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op_functor<Op, ggml_fp16_t, ggml_fp16_t>(params, dst, op);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op_functor<Op, ggml_bf16_t, ggml_bf16_t>(params, dst, op);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_bf16_t, float>(params, dst, op);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op_functor<Op, ggml_fp16_t, float>(params, dst, op);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_compute_forward_abs(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

void ggml_compute_forward_sgn(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

void ggml_compute_forward_neg(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

void ggml_compute_forward_step(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

void ggml_compute_forward_tanh(const ggml_compute_params * params, ggml_tensor * dst) {
    cpu_tanh_log(params, dst);
    unary_op<op_tanh>(params, dst);
    cpu_tanh_dump_rows_as_f32(params, dst);
}

void ggml_compute_forward_elu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

void ggml_compute_forward_relu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

void ggml_compute_forward_sigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

void ggml_compute_forward_hardsigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

void ggml_compute_forward_exp(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

void ggml_compute_forward_hardswish(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

void ggml_compute_forward_sqr(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

void ggml_compute_forward_sqrt(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

void ggml_compute_forward_sin(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

void ggml_compute_forward_cos(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

void ggml_compute_forward_log(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}

void ggml_compute_forward_expm1(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_expm1>(params, dst);
}

void ggml_compute_forward_softplus(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_softplus>(params, dst);
}

void ggml_compute_forward_floor(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_floor>(params, dst);
}

void ggml_compute_forward_ceil(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_ceil>(params, dst);
}

void ggml_compute_forward_round(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_round>(params, dst);
}

void ggml_compute_forward_trunc(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_trunc>(params, dst);
}

void ggml_compute_forward_xielu(const ggml_compute_params * params, ggml_tensor * dst) {
    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta = ggml_get_op_params_f32(dst, 3);
    const float eps = ggml_get_op_params_f32(dst, 4);

    const auto xielu_op_params = [alpha_n, alpha_p, beta, eps](float f) {
        return op_xielu(f, alpha_n, alpha_p, beta, eps);
    };

    unary_op_functor(params, dst, xielu_op_params);
}

