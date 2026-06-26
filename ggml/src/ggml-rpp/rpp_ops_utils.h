#ifndef RPP_TENSOR_H
#define RPP_TENSOR_H

#include "rpp_common.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>

#if GGML_RPP_USE_RT
size_t ggml_rpp_nbytes(infer1::Dims & dims, size_t type_size);

int ggml_rpp_dtype_size(infer1::DataType dtype);

infer1::DataType ggml_rpp_dtype_mapping(ggml_type g_type);

/**
 * @brief   Implement mapping the tensor dimension of ggml to the tensor dimension of rpprt.
 *
 * @details Mapping the tensor dimension of ggml to the tensor dimension of rpprt, ndims default is
 *          GGML_MAX_DIMS, and rpp_node param default is nullptr, if set rpp_node pointer, the seq len will 
 *          padding to the tensor dimension of rpprt
 *
 * @param tensor Pointer to the ggml_tensor object, its dimension will be map to dimension of rpprt.
 * @param ndims  the number of dimension will be mapping
 * @param rpp_node Pointer to the ggml_rpp_node object, its seq_len_index and n_ubatch will be used when mapping dimension
 */
infer1::Dims ggml_rpp_dims_mapping(const ggml_tensor * tensor,
                                   int                 ndims    = GGML_MAX_DIMS,
                                   ggml_rpp_node *     rpp_node = nullptr);

infer1::ITensor * ggml_rpp_create_constant_tensor(const ggml_tensor * tensor,
                                                  ggml_rpp_node *     rpp_node,
                                                  int                 ndims = GGML_MAX_DIMS);

infer1::ITensor * ggml_rpp_create_input_tensor(const ggml_tensor * tensor,
                                               ggml_rpp_node *     rpp_node,
                                               int                 ndims = GGML_MAX_DIMS);

infer1::ITensor * ggml_rpp_create_input_tensor(const ggml_tensor * tensor, ggml_rpp_node * rpp_node, infer1::Dims dims);

infer1::DataType ggml_rpp_get_io_type(ggml_backend_rpp_context & ctx, ggml_tensor * tensor, int32_t io_type = 0);

bool ggml_rpp_save_enigne(ggml_rpp_node * rpp_node, const std::string & file_name);

bool ggml_rpp_load_enigne(ggml_rpp_node * rpp_node, const std::string & file_name);

#endif

int ggml_rpp_n_dims(ggml_tensor ** tensor);

bool ggml_rpp_dims_is_same(const ggml_tensor * tensor0, const ggml_tensor * tensor1);

template <typename T> static int save_data_to_binary(const T * data, size_t count, const char * filename) {
    FILE * file = fopen(filename, "wb");
    if (file == NULL) {
        GGML_LOG_ERROR("%s: cannot open file: %s\n", __func__, filename);
        return -1;
    }

    size_t written = fwrite(data, sizeof(T), count, file);
    if (written != count) {
        GGML_LOG_ERROR("%s: error writing to file: %s\n", __func__, filename);
        fclose(file);
        return -1;
    }

    fclose(file);
    GGML_LOG_INFO("%s: successfully saved %zu floats to %s\n", __func__, count, filename);
    return 0;
}

template <typename T> static float calculate_mse(const std::string & file_name, const T * cur_data, int32_t number) {
    std::ifstream file(file_name, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        GGML_LOG_ERROR("%s: cannot open file: %s\n", __func__, file_name.c_str());
        return 100.0;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size % sizeof(T) != 0) {
        GGML_LOG_ERROR("%s: file size is not multiple of T size: %s\n", __func__, file_name.c_str());
        throw std::runtime_error("file size is not multiple of T size");
    }

    size_t         count = file_size / sizeof(T);
    std::vector<T> data(count);

    if (!file.read(reinterpret_cast<char *>(data.data()), file_size)) {
        GGML_LOG_ERROR("%s: failed to read file:  %s\n", __func__, file_name.c_str());
        throw std::runtime_error("Failed to read file: " + file_name);
    }
    file.close();

    if (data.size() != number) {
        GGML_LOG_ERROR("%s: data sizes must be equal,  file:   %s\n", __func__, file_name.c_str());
        return 100.0;
    }
    T sum   = 0.0f;
    T power = 0.0f;
    for (int32_t i = 0; i < number; ++i) {
        T diff = data[i] - cur_data[i];
        sum += diff * diff;
        power += data[i] * data[i] + cur_data[i] * cur_data[i];
    }
    return (float) (sum / power);
}

template <typename T> static float calculate_mse(const T * src_data, const T * dst_data, int32_t number) {
    float sum = 0.0f;
    for (int32_t i = 0; i < number; ++i) {
        float diff = float(src_data[i] - dst_data[i]);
        sum += diff * diff;
    }
    return sum / number;
}

template <typename Func, typename... Args> double measure_time_mics(Func && func, Args &&... args) {
    auto now     = std::chrono::system_clock::now();
    auto mis     = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto t_start = mis.count();

    std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);

    now             = std::chrono::system_clock::now();
    mis             = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto t_end      = mis.count();
    auto t_interval = ((double) t_end - (double) t_start) / 1000.0;
    return t_interval;
}

/**
 * @brief   Check whether a tensor is a weight tensor for matrix multiplication.
 *
 * @details Checks whether the given tensor serves as weight parameters in matrix multiplication operations,
 *          typically within neural network layers. The function maintains a static set of canonical weight
 *          naming suffixes from Transformer-based architectures. Uses substring matching to identify weight
 *          tensors even with hierarchical naming patterns.
 *
 * @param tensor Pointer to the target ggml_tensor object (const-qualified).
 */
static bool is_matmul_weight(const ggml_tensor * tensor) {
    std::string                                  name = ggml_get_name(tensor);
    static const std::unordered_set<std::string> matmul_weight_suffixes{
        "output.weight",      "attn_q.weight",      "attn_k.weight",        "attn_v.weight",
        "attn_output.weight", "ffn_gate.weight",    "ffn_up.weight",        "ffn_down.weight",
        "token_embd.weight",  "attn_qkv.weight",    "ffn_down_exps.weight", "ffn_gate_exps.weight",
        "ffn_up_exps.weight", "ffn_gate_inp.weight"
    };

    for (const auto & suffix : matmul_weight_suffixes) {
        if (name.find(suffix) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool is_mul_weight(const ggml_tensor * tensor) {
    std::string                                  name = ggml_get_name(tensor);
    static const std::unordered_set<std::string> mul_weight_suffixes{ "attn_q_norm.weight", "attn_k_norm.weight",
                                                                      "ffn_norm.weight", "output_norm.weight",
                                                                      // "token_embd.weight",
                                                                      "attn_norm.weight" };

    for (const auto & suffix : mul_weight_suffixes) {
        if (name.find(suffix) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool is_cache_kv(const ggml_tensor * tensor) {
    std::string                                  name = ggml_get_name(tensor);
    static const std::unordered_set<std::string> kv_suffixes{ "cache_k", "cache_v" };

    for (const auto & suffix : kv_suffixes) {
        if (name.find(suffix) != std::string::npos) {
            return true;
        }
    }
    return false;
}

/**
 * @brief   Dump all nodes in the graph.
 *
 * @details Dump all nodes in the graph. 
 *          Each node line contains: idx, op, type, shape, flags, name + list of src indices.
 *
 * @param tensor Pointer to the target ggml_cgraph object (const-qualified).
 * @param filename the filename which save
 * @param comment_prefix the commen prefix, as: toy graph y = x + x
 */
void ggml_rpp_dump_cgraph_ops(const ggml_cgraph * gf_const, const char * filename, const char * comment_prefix);

void ggml_rpp_get_ops_info(const ggml_cgraph * gf_const, const ggml_tensor * dst, char * info);

void ggml_rpp_pack_tensor_to_contiguous(ggml_backend_rpp_context & ctx,
                                        const ggml_tensor *        src,
                                        void *                     dst_buffer,
                                        size_t                     src_offset = 0);

/**
 * @brief   Get the I/O element byte size of a tensor in GGML RPP backend
 * 
 * @details This function determines the byte size of tensor elements in the RPP backend,
 *          with special handling for tensors in the input/output collections. For tensors
 *          not in these collections, it returns a default size based on BF16 support.
 * 
 * @param ctx GGML RPP backend context, containing current RPP graph and tensor collections
 * @param tensor Pointer to the tensor whose size is to be queried
 * @param io_type I/O type indicator: 0 for input tensor collection, non-0 for output tensor collection
 * @return int32_t Byte size of the tensor element
 */
int ggml_rpp_get_io_type_size(ggml_backend_rpp_context & ctx, ggml_tensor * tensor, int32_t io_type = 0);

/**
 * @brief   Convert BF16 to FP32
 * 
 * @details Convert BF16 data to FP32 data
 * 
 * @param bf16_data BF16 data
 * @param fp32_data FP32 data
 * @param nelements Number of elements
 * @return true Success
 * @return false Failed
 */
bool ggml_rpp_bf16_to_fp32(ggml_bf16_t * bf16_data, float * fp32_data, size_t nelements);

/**
 * @brief   Convert FP32 to BF16
 * 
 * @details Convert FP32 data to BF16 data
 * 
 * @param bf16_data BF16 data
 * @param fp32_data FP32 data
 * @param nelements Number of elements
 * @return true Success
 * @return false Failed
 */
bool ggml_rpp_fp32_to_bf16(ggml_bf16_t * bf16_data, float * fp32_data, size_t nelements);

#endif
