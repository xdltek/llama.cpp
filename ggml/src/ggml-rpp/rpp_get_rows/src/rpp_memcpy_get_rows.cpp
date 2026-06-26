#include "ggml-cpu.h"
#include "rpp_get_rows/rpp_get_rows.h"

bool ggml_rpp_op_cpy_get_rows(ggml_backend_rpp_context & ctx, ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0]; // F32 table
    const ggml_tensor* src1 = dst->src[1]; // int32 indices

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int64_t ne0  = dst->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];
    const size_t nb12 = src1->nb[2];
    const size_t nb13 = src1->nb[3];

    const size_t nb1  = dst->nb[1];
    const size_t nb2  = dst->nb[2];
    const size_t nb3  = dst->nb[3];

    const int64_t nc = ne00;                 // row width (elements)
    const int64_t nr = ggml_nelements(src1); // number of gathers

    // Sanity
    if (ne0 != nc) throw std::runtime_error("dst ne[0] != src0 ne[0]");
    if (nb00 != sizeof(float)) throw std::runtime_error("src0 nb[0] != sizeof(float)");
    if (nb10 != sizeof(int32_t)) throw std::runtime_error("src1 nb[0] != sizeof(int32)");

    // We D2H copy src1 in one shot => require contiguous indices tensor
    const bool src1_contig =
        nb11 == (size_t)ne10 * sizeof(int32_t) &&
        nb12 == (size_t)ne10 * (size_t)ne11 * sizeof(int32_t) &&
        nb13 == (size_t)ne10 * (size_t)ne11 * (size_t)ne12 * sizeof(int32_t);

    if (!src1_contig) {
        throw std::runtime_error("src1 is not contiguous; cannot D2H copy as one block in this implementation");
    }

    // 1) Copy all indices (device -> host)
    std::vector<int32_t> h_idx((size_t)nr);
    RPP_MEMCPY_DEV_AND_HOST(h_idx.data(),
                src1->data,        
                (size_t)nr * sizeof(int32_t),
                rtMemcpyDeviceToHost,
                ctx.stream());

    // 2) Copy rows (device -> device) according to indices
    const size_t row_bytes = (size_t)nc * sizeof(float);

    for (int64_t i = 0; i < nr; ++i) {
        // Same flattening as the ggml CPU reference
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
        const int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);

        const int32_t i01 = h_idx[(size_t)i];
        if (i01 < 0 || i01 >= ne01) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                          "GET_ROWS: index out of range i01=%d (valid [0,%lld)) at i=%lld",
                          i01, (long long)ne01, (long long)i);
            throw std::out_of_range(buf);
        }

        // dst row address: (i10,i11,i12)
        uint8_t* dst_ptr = (uint8_t*)dst->data
                         + (size_t)i10 * nb1
                         + (size_t)i11 * nb2
                         + (size_t)i12 * nb3;

        // src0 selected row address: (i01,i11,i12)
        const uint8_t* src_ptr = (const uint8_t*)src0->data
                               + (size_t)i01 * nb01
                               + (size_t)i11 * nb02
                               + (size_t)i12 * nb03;

        RPP_MEMCPY_DEV_AND_HOST(dst_ptr,
                                src_ptr,            
                                row_bytes,              // dstMax
                                rtMemcpyDeviceToDevice,
                                ctx.stream());

    }

    return true;
}