#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif


#define GGML_RPP_NAME "RPP"
#define GGML_RPP_MAX_DEVICES   16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpp_init(int device, const char* params);

GGML_BACKEND_API bool ggml_backend_is_rpp(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpp_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpp_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpp_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_rpp_get_device_count(void);
GGML_BACKEND_API void ggml_backend_rpp_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_rpp_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_rpp_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_rpp_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpp_reg(void);

GGML_BACKEND_API void ggml_backend_rpp_model_init(const void* params, const void* ctx, const void *vocab, const void *smpl);

#ifdef  __cplusplus
}
#endif
