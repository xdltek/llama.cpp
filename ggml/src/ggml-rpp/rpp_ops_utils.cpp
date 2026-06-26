#include "rpp_ops_utils.h"

#if GGML_RPP_USE_RT
int ggml_rpp_dtype_size(infer1::DataType dtype) {
    switch (dtype) {
        case infer1::DataType::kFLOAT:
        case infer1::DataType::kINT32:
            return 4;
        case infer1::DataType::kHALF:
        case infer1::DataType::kBF:
        case infer1::DataType::kINT16:
            return 2;
        case infer1::DataType::kINT8:
            return 1;
        default:
            return 4;
    }
    return 0;
}

size_t ggml_rpp_nbytes(infer1::Dims & dims, size_t type_size) {
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * type_size;
}

infer1::DataType ggml_rpp_dtype_mapping(ggml_type g_type) {
    static const std::unordered_map<ggml_type, infer1::DataType> dtype_map = {
        { GGML_TYPE_F32,  infer1::DataType::kFLOAT },
        // { GGML_TYPE_F16,  infer1::DataType::kHALF  },
        { GGML_TYPE_F16,  infer1::DataType::kBF    },
        { GGML_TYPE_BF16, infer1::DataType::kBF    },
        { GGML_TYPE_Q8_0, infer1::DataType::kINT8  },
        { GGML_TYPE_I32,  infer1::DataType::kINT32 },
        { GGML_TYPE_I16,  infer1::DataType::kINT16 },
        { GGML_TYPE_I64,  infer1::DataType::kINT32 }, //don't support KINT64
    };

    auto it = dtype_map.find(g_type);
    if (it != dtype_map.end()) {
        return it->second;
    }
    GGML_LOG_WARN("ggml_type: %d map to infer1::DataType failed, return default type kFLOAT\n", int(g_type));
    return infer1::DataType::kFLOAT;
}

infer1::Dims ggml_rpp_dims_mapping(const ggml_tensor * tensor, int ndims, ggml_rpp_node * rpp_node) {
    GGML_ASSERT(ndims <= GGML_MAX_DIMS);
    if (rpp_node) {
        GGML_ASSERT(rpp_node->seq_len_index < ndims);
    }
    infer1::Dims dims;
    dims.nbDims = ndims;
    for (int i = 0; i < ndims; i++) {
        if (rpp_node && rpp_node->seq_len_index == i) {
            dims.d[ndims - i - 1] = rpp_node->n_ubatch;
        } else {
            dims.d[ndims - i - 1] = tensor->ne[i];
        }
    }
    for (int i = 0; i < dims.nbDims; i++) {
        dims.type[i] = (infer1::DimensionType) 0;
    }

    return dims;
}

infer1::ITensor * ggml_rpp_create_constant_tensor(const ggml_tensor * tensor,
                                                  ggml_rpp_node *     rpp_base_node,
                                                  int                 ndims) {
    if (!tensor || !rpp_base_node) {
        GGML_LOG_ERROR("%s: ggml_tensor or rpp_graph is nullptr\n", __func__);
        return nullptr;
    }
    auto             rpp_node  = static_cast<rpp_node_openrt *>(rpp_base_node);
    infer1::DataType ort_dtype = ggml_rpp_dtype_mapping(tensor->type);
    infer1::Dims     dims      = ggml_rpp_dims_mapping(tensor, ndims);
    if (tensor->data) {
        infer1::Weights weights;
        weights.type                 = ort_dtype;
        weights.values               = tensor->data;
        weights.count                = ggml_nbytes(tensor) / ggml_rpp_dtype_size(ort_dtype);
        auto              constant   = rpp_node->network->addConstant(dims, weights);
        infer1::ITensor * ort_tensor = constant->getOutput(0);
        ort_tensor->setName(tensor->name);
        if (!ort_tensor) {
            GGML_LOG_ERROR("%s: ort tensor is nullptr, can not construct rpp constant tensor %s (%s)\n", __func__,
                           tensor->name, ggml_op_name(tensor->op));
        }
        return ort_tensor;
    } else {
        GGML_LOG_ERROR("%s: ggml_tensor data is nullptr, can not construct rpp constant tensor %s (%s)\n", __func__,
                       tensor->name, ggml_op_name(tensor->op));
        return nullptr;
    }
}

infer1::ITensor * ggml_rpp_create_input_tensor(const ggml_tensor * tensor, ggml_rpp_node * rpp_base_node, int ndims) {
    if (!tensor || !rpp_base_node) {
        GGML_LOG_ERROR("%s: ggml_tensor or network is nullptr\n", __func__);
        return nullptr;
    }
    auto              rpp_node   = static_cast<rpp_node_openrt *>(rpp_base_node);
    infer1::DataType  ort_dtype  = ggml_rpp_dtype_mapping(tensor->type);
    infer1::Dims      dims       = ggml_rpp_dims_mapping(tensor, ndims);
    infer1::ITensor * ort_tensor = rpp_node->network->addInput(tensor->name, ort_dtype, dims);
    if (!ort_tensor) {
        GGML_LOG_ERROR("%s: ort tensor is nullptr, can not construct rpp input tensor %s (%s)\n", __func__,
                       tensor->name, ggml_op_name(tensor->op));
    }
    return ort_tensor;
}

infer1::ITensor * ggml_rpp_create_input_tensor(const ggml_tensor * tensor,
                                               ggml_rpp_node *     rpp_base_node,
                                               infer1::Dims        dims) {
    if (!rpp_base_node) {
        GGML_LOG_ERROR("%s: ggml_tensor or network is nullptr\n", __func__);
        return nullptr;
    }
    auto              rpp_node   = static_cast<rpp_node_openrt *>(rpp_base_node);
    infer1::DataType  ort_dtype  = ggml_rpp_dtype_mapping(tensor->type);
    infer1::ITensor * ort_tensor = rpp_node->network->addInput(tensor->name, ort_dtype, dims);
    if (!ort_tensor) {
        GGML_LOG_ERROR("%s: ort tensor is nullptr, can not construct rpp input tensor %s (%s)\n", __func__,
                       tensor->name, ggml_op_name(tensor->op));
    }
    return ort_tensor;
}

infer1::DataType ggml_rpp_get_io_type(ggml_backend_rpp_context & ctx, ggml_tensor * tensor, int32_t io_type) {
    auto cur_grap = ctx.cur_rpp_graph;
    GGML_ASSERT(cur_grap);
    auto & io_tensor = io_type == 0 ? cur_grap->nodes_i : cur_grap->nodes_o;
    if (io_tensor.count(tensor)) {
        return ggml_rpp_dtype_mapping(tensor->type);
    } else {
        return ctx.use_bf16 ? infer1::DataType::kBF : ggml_rpp_dtype_mapping(tensor->type);
    }
}

bool ggml_rpp_save_enigne(ggml_rpp_node * rpp_base_node, const std::string & file_name) {
    std::ifstream file(file_name, std::ios::binary);
    if (file.good()) {
        // GGML_LOG_DEBUG("%s: file: %s  is exist, no need to save engine\n", __func__, file_name.c_str());
        file.close();
        return true;
    }
    GGML_ASSERT(rpp_base_node);
    auto rpp_node = static_cast<rpp_node_openrt *>(rpp_base_node);
    GGML_ASSERT(rpp_node->engine);
    infer1::IHostMemory * serializedModel = rpp_node->engine->serialize();
    std::ofstream         engineFile(file_name, std::ios::binary);
    if (!engineFile) {
        GGML_LOG_ERROR("%s: can not open file: %s\n", __func__, file_name.c_str());
        return false;
    }

    engineFile.write(static_cast<const char *>(serializedModel->data()), serializedModel->size());
    engineFile.close();
    serializedModel->destroy();
    return true;
}

bool ggml_rpp_load_enigne(ggml_rpp_node * rpp_base_node, const std::string & file_name) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.good()) {
        // GGML_LOG_DEBUG("%s: can not open file: %s\n", __func__, file_name.c_str());
        return false;
    }

    // read file
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    if (!file) {
        GGML_LOG_ERROR("%s: read engine file failed: %s\n", __func__, file_name.c_str());
        return false;
    }
    GGML_LOG_INFO("%s: load engine file : %s (% ld bytes)\n", __func__, file_name.c_str(), size);

    GGML_ASSERT(rpp_base_node);
    auto               rpp_node = static_cast<rpp_node_openrt *>(rpp_base_node);
    infer1::IRuntime * runtime  = infer1::createInferRuntime(*(rpp_node->logger.get()));
    if (!runtime) {
        GGML_LOG_ERROR("%s: create runtime failed\n", __func__);
        return false;
    }
    auto & dst = rpp_node->cur_ggml_tensor;
    rpp_node->engine.reset(runtime->deserializeCudaEngine(engineData.data(), size, nullptr));
    if (!rpp_node->engine) {
        GGML_LOG_ERROR("%s: deserialize rt engine failed,  %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        runtime->destroy();
        return false;
    }
    rpp_node->context.reset(rpp_node->engine->createExecutionContext());
    if (!rpp_node->context) {
        GGML_LOG_ERROR("%s: create execution context failed, %s (%s)\n", __func__, dst->name, ggml_op_name(dst->op));
        return false;
    }
    runtime->destroy();
    return true;
}

#endif

bool ggml_rpp_dims_is_same(const ggml_tensor * tensor0, const ggml_tensor * tensor1) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor0->ne[i] != tensor1->ne[i]) {
            return false;
        }
    }
    return true;
}

int ggml_rpp_n_dims(ggml_tensor ** tensor) {
    int ndims = 0;
    for (size_t i = 0; i < GGML_MAX_SRC; i++) {
        ggml_tensor * cur_tensor = tensor[i];
        if (!cur_tensor) {
            break;
        }
        int cur_ndims = (ggml_n_dims(cur_tensor) == GGML_MAX_DIMS) ? GGML_MAX_DIMS : (ggml_n_dims(cur_tensor) + 1);
        ndims         = cur_ndims > ndims ? cur_ndims : ndims;
    }
    return ndims;
}

static const char * ggml_tensor_safe_name(const ggml_tensor * t) {
    const char * n = ggml_get_name(t);
    if (n && n[0] != '\0') {
        return n;
    }
    return "<unnamed>";
}

// Find the index of a tensor inside a cgraph (0 .. n_nodes-1), or -1 if not found.
static int ggml_graph_find_node_index(ggml_cgraph * g, const ggml_tensor * t) {
    const int n = ggml_graph_n_nodes(g);
    for (int i = 0; i < n; ++i) {
        if (ggml_graph_node(g, i) == t) {
            return i;
        }
    }
    return -1;
}

void ggml_rpp_dump_cgraph_ops(const ggml_cgraph * gf_const, const char * filename, const char * comment_prefix) {
    if (gf_const == nullptr || filename == nullptr) {
        return;
    }

    FILE * f = std::fopen(filename, "w");
    if (!f) {
        std::perror("ggml_dump_cgraph_ops: fopen");
        return;
    }

    // ggml_graph_* APIs want a non-const pointer
    ggml_cgraph * gf = const_cast<ggml_cgraph *>(gf_const);

    const int n_nodes = ggml_graph_n_nodes(gf);

    if (comment_prefix && comment_prefix[0] != '\0') {
        std::fprintf(f, "# %s\n", comment_prefix);
    }
    std::fprintf(f, "# n_nodes = %d\n\n", n_nodes);

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        if (!t) {
            std::fprintf(f, "node %4d : <null>\n\n", i);
            continue;
        }

        const char * op_name   = ggml_op_desc(t);  // "unary/<name>" or op name
        const char * type_name = ggml_type_name(t->type);
        const char * name      = ggml_tensor_safe_name(t);

        std::fprintf(f, "node %4d : op=%-18s type=%-8s ne=[%lld,%lld,%lld,%lld] flags=0x%08x name=%s\n", i,
                     op_name ? op_name : "<none>", type_name ? type_name : "<unk>", (long long) t->ne[0],
                     (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], t->flags, name);

        bool any_src = false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = t->src[s];
            if (!src) {
                continue;
            }
            int src_idx = ggml_graph_find_node_index(gf, src);
            std::fprintf(f, "    src[%d] -> node=%d (%s)\n", s, src_idx, ggml_tensor_safe_name(src));
            any_src = true;
        }

        if (!any_src) {
            std::fprintf(f, "    (no src)\n");
        }

        std::fprintf(f, "\n");
    }

    std::fclose(f);
}

void ggml_rpp_get_ops_info(const ggml_cgraph * gf_const, const ggml_tensor * dst, char * info) {
    info[0] = '\0';
    char         buffer[256];
    const int    i         = ggml_graph_find_node_index((ggml_cgraph *) gf_const, dst);
    const char * op_name   = ggml_op_desc(dst);  // "unary/<name>" or op name
    const char * type_name = ggml_type_name(dst->type);
    const char * name      = ggml_tensor_safe_name(dst);

    std::sprintf(buffer, "node %4d : op=%-18s type=%-8s ne=[%lld,%lld,%lld,%lld] flags=0x%08x name=%s\n", i,
                 op_name ? op_name : "<none>", type_name ? type_name : "<unk>", (long long) dst->ne[0],
                 (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3], dst->flags, name);
    std::strcat(info, buffer);

    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = dst->src[s];
        if (!src) {
            continue;
        }
        int src_idx = ggml_graph_find_node_index((ggml_cgraph *) gf_const, src);
        std::sprintf(buffer, "    src[%d] -> node=%d (%s) type=%-8s ne=[%lld,%lld,%lld,%lld]\n", s, src_idx,
                     ggml_tensor_safe_name(src), type_name ? type_name : "<unk>", (long long) src->ne[0],
                     (long long) src->ne[1], (long long) src->ne[2], (long long) src->ne[3]);
        std::strcat(info, buffer);
    }
}

void ggml_rpp_pack_tensor_to_contiguous(ggml_backend_rpp_context & ctx,
                                        const ggml_tensor *        src,
                                        void *                     dst_buffer,
                                        size_t                     offset) {
    assert(src && src->data && dst_buffer);

    // Only F32 tensors are compacted to BF16 for internal RPP flow.
    const bool   use_internal_bf16 = ctx.use_bf16 && src->type == GGML_TYPE_F32;
    const size_t scale_divisor     = use_internal_bf16 ? 2 : 1;
    const size_t ts                = ggml_type_size(src->type) / scale_divisor;
    const size_t bs = ggml_blck_size(src->type);

    // ggml 的 ne0 维是按 block 存储的；拷贝的最小单位是 “一个 block row”
    assert(src->ne[0] % (int64_t) bs == 0);

    // 一行（固定 i1,i2,i3，沿 ne0）拷贝的字节数
    // 对 block=1：row_bytes = ne0 * ts
    // 对 block>1：row_bytes = (ne0/bs) * ts
    const size_t row_bytes = (size_t) (src->ne[0] / (int64_t) bs) * ts;

    // 目标连续布局 stride
    const size_t dst_nb0 = ts;
    const size_t dst_nb1 = dst_nb0 * (size_t) (src->ne[0] / (int64_t) bs);
    const size_t dst_nb2 = dst_nb1 * (size_t) src->ne[1];
    const size_t dst_nb3 = dst_nb2 * (size_t) src->ne[2];

    if (use_internal_bf16) {
        assert(offset % scale_divisor == 0);
        assert(src->nb[1] % scale_divisor == 0);
        assert(src->nb[2] % scale_divisor == 0);
        assert(src->nb[3] % scale_divisor == 0);
    }
    const char * sbase = ctx.use_bf16 ? (const char *) src->data - offset / scale_divisor : (const char *) src->data;
    char *       dbase = (char *) dst_buffer;

    // 按 ggml 的 4D 约定遍历：ne[0],ne[1],ne[2],ne[3]
    for (int64_t i3 = 0; i3 < src->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < src->ne[1]; ++i1) {
                const char * src_row =
                    ctx.use_bf16 ?
                        sbase + i1 * (ptrdiff_t) src->nb[1] / (ptrdiff_t) scale_divisor +
                            i2 * (ptrdiff_t) src->nb[2] / (ptrdiff_t) scale_divisor +
                            i3 * (ptrdiff_t) src->nb[3] / (ptrdiff_t) scale_divisor :
                        sbase + i1 * (ptrdiff_t) src->nb[1] + i2 * (ptrdiff_t) src->nb[2] +
                            i3 * (ptrdiff_t) src->nb[3];

                char * dst_row = dbase + i1 * (ptrdiff_t) dst_nb1 + i2 * (ptrdiff_t) dst_nb2 + i3 * (ptrdiff_t) dst_nb3;
                RPP_CHECK(rtMemcpy(dst_row, src_row, row_bytes, rtMemcpyDeviceToDevice));
            }
        }
    }
}

int ggml_rpp_get_io_type_size(ggml_backend_rpp_context & ctx, ggml_tensor * tensor, int32_t io_type) {
    if (!ctx.use_bf16) {
        return ggml_type_size(tensor->type);
    }
    auto cur_grap = ctx.cur_rpp_graph;
    GGML_ASSERT(cur_grap);
    auto & io_tensor = io_type == 0 ? cur_grap->nodes_i : cur_grap->nodes_o;
    if (io_tensor.count(tensor) || tensor->type != GGML_TYPE_F32 || is_matmul_weight(tensor) || is_mul_weight(tensor)) {
        return ggml_type_size(tensor->type);
    } else {
        return 2;
    }
}

bool ggml_rpp_bf16_to_fp32(ggml_bf16_t * bf16_data, float * fp32_data, size_t nelements) {
    for (size_t i = 0; i < nelements; ++i) {
        fp32_data[i] = ggml_bf16_to_fp32(bf16_data[i]);
    }
    return true;
}

bool ggml_rpp_fp32_to_bf16(ggml_bf16_t * bf16_data, float * fp32_data, size_t nelements) {
    for (size_t i = 0; i < nelements; ++i) {
        bf16_data[i] = ggml_fp32_to_bf16(fp32_data[i]);
    }
    return true;
}
