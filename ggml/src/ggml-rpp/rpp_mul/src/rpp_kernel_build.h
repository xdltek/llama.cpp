// memcpy_2d_rpp.cpp
#include "ggml-rpp/rpp_kernel_ctx.h"
#include "ggml-rpp/rpp_kernel_utils.h"
#include "rpp_drv_api.h"
#include "rpp_mul/src/rpp_kernel_block.h"
#include "rpp_mul/src/rpp_kernel_param.h"

#include <assert.h>
#include <rpp_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

inline void get_tile_info(int C, int H, int W, int & num_of_tiles, int & elements_per_tile) {
    num_of_tiles      = C;
    elements_per_tile = H * W;
    if (H * W > MAX_TILE_SIZE) {
        throw std::runtime_error("Elementwise Tensor Size Too Big");
    }
}

inline int rpp_elementwise_round_up(int a) {
    return (a + 511) / 512 * 512 + 512;
}

inline int rpp_elementwise_round_up_to_32(int a) {
    return (a + 31) / 32 * 32;
}

inline int rpp_elementwise_get_input1_bytes(int axis, int H, int W, int in1_bytes_per_element) {
    if (axis == -1 || axis == 0) {
        return H * W * in1_bytes_per_element;
    }
    if (axis == 1 || axis == 6 || axis == 7) {
        return W * in1_bytes_per_element;
    }
    if (axis == 2) {
        return H * in1_bytes_per_element;
    }
    throw std::runtime_error("Unsupported elementwise broadcast axis");
}

inline int rpp_elementwise_get_tile_sram_bytes(int tile_H,
                                               int W,
                                               int axis,
                                               int in0_bytes_per_element,
                                               int in1_bytes_per_element,
                                               int out_bytes_per_element,
                                               int reciprocal_table_bytes) {
    const int  padded_W             = rpp_elementwise_round_up_to_32(W);
    const bool use_row_tail_padding = padded_W != W;

    if (use_row_tail_padding) {
        assert(axis != 2);
        assert(padded_W < 8192);

        const int row_bytes_bf16   = padded_W * (int) sizeof(uint16_t);
        int       total_sram_bytes = reciprocal_table_bytes;

        if (in0_bytes_per_element == (int) sizeof(float)) {
            total_sram_bytes += rpp_elementwise_round_up(tile_H * W * (int) sizeof(float));
        }
        if (in1_bytes_per_element == (int) sizeof(float)) {
            total_sram_bytes +=
                rpp_elementwise_round_up(rpp_elementwise_get_input1_bytes(axis, tile_H, W, (int) sizeof(float)));
        }

        total_sram_bytes += rpp_elementwise_round_up(tile_H * row_bytes_bf16);
        total_sram_bytes +=
            rpp_elementwise_round_up((axis == -1 || axis == 0) ? tile_H * row_bytes_bf16 : row_bytes_bf16);
        total_sram_bytes += rpp_elementwise_round_up(tile_H * row_bytes_bf16);

        if (out_bytes_per_element == (int) sizeof(float) && in0_bytes_per_element != (int) sizeof(float) &&
            in1_bytes_per_element != (int) sizeof(float)) {
            total_sram_bytes += rpp_elementwise_round_up(tile_H * W * (int) sizeof(float));
        }

        return total_sram_bytes;
    }

    const int sizeA = tile_H * W * in0_bytes_per_element;
    const int sizeB = rpp_elementwise_get_input1_bytes(axis, tile_H, W, in1_bytes_per_element);
    const int sizeC = tile_H * W * out_bytes_per_element;
    return reciprocal_table_bytes + rpp_elementwise_round_up(sizeA) + rpp_elementwise_round_up(sizeB) +
           rpp_elementwise_round_up(sizeC);
}

inline int rpp_elementwise_get_max_tile_rows(int H,
                                             int W,
                                             int axis,
                                             int in0_bytes_per_element,
                                             int in1_bytes_per_element,
                                             int out_bytes_per_element,
                                             int reciprocal_table_bytes,
                                             int sram_limit) {
    int lo = 0;
    int hi = H;

    while (lo < hi) {
        const int mid              = (lo + hi + 1) / 2;
        const int total_sram_bytes = rpp_elementwise_get_tile_sram_bytes(
            mid, W, axis, in0_bytes_per_element, in1_bytes_per_element, out_bytes_per_element, reciprocal_table_bytes);
        if (total_sram_bytes <= sram_limit) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    if (lo <= 0) {
        std::cerr << "SRAM overflow: even one row does not fit in SRAM\n";
        std::abort();
    }

    return lo;
}

inline void rpp_elementwise_build_tiled(rpp_kernel_context & ctx,
                                        rppElementWiseType   type,
                                        int                  C,
                                        int                  H,
                                        int                  W,
                                        int                  axis,
                                        int                  in0_bytes_per_element,
                                        int                  in1_bytes_per_element,
                                        int                  out_bytes_per_element,
                                        int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA;
    RPPdeviceptr          devB;
    bool                  ab_switched = false;

    if (axis <= 2 || axis == 6) {
        devA = ctx.dev_in[0];
        devB = ctx.dev_in[1];
    } else {
        ab_switched           = true;
        devA                  = ctx.dev_in[1];
        devB                  = ctx.dev_in[0];
        int tmp               = in1_bytes_per_element;
        in1_bytes_per_element = in0_bytes_per_element;
        in0_bytes_per_element = tmp;
        if (axis != 7) {
            axis = axis - 3;
        }
    }

    RPPdeviceptr devC = ctx.dev_out[0];

    void * reciprocal_table       = nullptr;
    int    reciprocal_table_bytes = 0;
    if (type == RPP_ELEMWISE_DIV) {
        reciprocal_table_bytes = 65536 * (int) sizeof(uint16_t);
        reciprocal_table       = malloc(reciprocal_table_bytes);
        for (uint32_t i = 0; i < 65536; ++i) {
            uint32_t x = i;
            x <<= 16;
            float y                            = 1.0f / *(float *) &x;
            ((uint16_t *) reciprocal_table)[i] = rpp::bfloat16::round_to_bfloat16(y).value;
        }
        ((uint16_t *) reciprocal_table)[0]      = 0;
        ((uint16_t *) reciprocal_table)[0x8000] = 0;
        rtMemcpy((void *) ctx.dev_workspace, reciprocal_table, reciprocal_table_bytes, rtMemcpyHostToDevice);
    }

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/elementwise.o");

    RPPdeviceptr sram_base            = ctx.virtual_sram_base;
    RPPdeviceptr sramReciprocalTable  = sram_base;
    const int    padded_W             = rpp_elementwise_round_up_to_32(W);
    const bool   use_row_tail_padding = padded_W != W;
    const int    SRAM_LIMIT           = 22 * 1024 * 1024;
    const int max_tile_h = rpp_elementwise_get_max_tile_rows(H, W, axis, in0_bytes_per_element, in1_bytes_per_element,
                                                             out_bytes_per_element, reciprocal_table_bytes, SRAM_LIMIT);

    if (type == RPP_ELEMWISE_DIV) {
        rtMemcpyAsync((void *) sramReciprocalTable, (const void *) ctx.dev_workspace, reciprocal_table_bytes,
                      rtMemcpyDeviceToSram, ctx.kernelStream);
    }

    for (int c = 0; c < C; ++c) {
        const size_t devA_offset_base = (size_t) c * (size_t) H * (size_t) W * (size_t) in0_bytes_per_element;
        size_t       devB_offset_base = 0;
        if (axis == -1) {
            devB_offset_base = (size_t) c * (size_t) H * (size_t) W * (size_t) in1_bytes_per_element;
        } else if (axis == 1) {
            devB_offset_base = (size_t) c * (size_t) W * (size_t) in1_bytes_per_element;
        } else if (axis == 2) {
            devB_offset_base = (size_t) c * (size_t) H * (size_t) in1_bytes_per_element;
        }
        const size_t devC_offset_base = (size_t) c * (size_t) H * (size_t) W * (size_t) out_bytes_per_element;

        for (int h_begin = 0; h_begin < H; h_begin += max_tile_h) {
            const int    tile_H = std::min(max_tile_h, H - h_begin);
            const int    sizeA  = tile_H * W * in0_bytes_per_element;
            const int    sizeB  = rpp_elementwise_get_input1_bytes(axis, tile_H, W, in1_bytes_per_element);
            const int    sizeC  = tile_H * W * out_bytes_per_element;
            const size_t devA_offset =
                devA_offset_base + (size_t) h_begin * (size_t) W * (size_t) in0_bytes_per_element;
            size_t devB_offset = devB_offset_base;
            if (axis == -1 || axis == 0) {
                devB_offset += (size_t) h_begin * (size_t) W * (size_t) in1_bytes_per_element;
            } else if (axis == 2) {
                devB_offset += (size_t) h_begin * (size_t) in1_bytes_per_element;
            }
            const size_t devC_offset =
                devC_offset_base + (size_t) h_begin * (size_t) W * (size_t) out_bytes_per_element;

            const int total_sram_bytes =
                rpp_elementwise_get_tile_sram_bytes(tile_H, W, axis, in0_bytes_per_element, in1_bytes_per_element,
                                                    out_bytes_per_element, reciprocal_table_bytes);
            assert(total_sram_bytes <= SRAM_LIMIT);

            RPPdeviceptr sramA = sramReciprocalTable + reciprocal_table_bytes;
            RPPdeviceptr sramB = sramA + rpp_elementwise_round_up(sizeA);
            RPPdeviceptr sramC = sramB + rpp_elementwise_round_up(sizeB);

            if (use_row_tail_padding) {
                assert(axis != 2);
                assert(padded_W < 8192);

                const int row_bytes_bf16 = padded_W * (int) sizeof(uint16_t);
                const int sizeA_compact  = sizeA;
                const int sizeB_compact  = sizeB;
                const int sizeA_padded   = tile_H * row_bytes_bf16;
                const int sizeB_padded   = (axis == -1 || axis == 0) ? tile_H * row_bytes_bf16 : row_bytes_bf16;
                const int sizeOut_padded = tile_H * row_bytes_bf16;

                RPPdeviceptr sramCursor   = sramReciprocalTable + reciprocal_table_bytes;
                RPPdeviceptr sramACompact = 0;
                RPPdeviceptr sramBCompact = 0;
                if (in0_bytes_per_element == (int) sizeof(float)) {
                    sramACompact = sramCursor;
                    sramCursor += rpp_elementwise_round_up(sizeA_compact);
                }
                if (in1_bytes_per_element == (int) sizeof(float)) {
                    sramBCompact = sramCursor;
                    sramCursor += rpp_elementwise_round_up(sizeB_compact);
                }

                RPPdeviceptr sramAPadded = sramCursor;
                sramCursor += rpp_elementwise_round_up(sizeA_padded);
                RPPdeviceptr sramBPadded = sramCursor;
                sramCursor += rpp_elementwise_round_up(sizeB_padded);
                RPPdeviceptr sramOutPadded = sramCursor;
                sramCursor += rpp_elementwise_round_up(sizeOut_padded);

                RPPdeviceptr sramOutCompact = 0;
                if (out_bytes_per_element == (int) sizeof(float)) {
                    if (sramACompact != 0) {
                        sramOutCompact = sramACompact;
                    } else if (sramBCompact != 0) {
                        sramOutCompact = sramBCompact;
                    } else {
                        sramOutCompact = sramCursor;
                        sramCursor += rpp_elementwise_round_up(sizeC);
                    }
                }

                const int total_sram_bytes_padded = (int) (sramCursor - sram_base);
                if (total_sram_bytes_padded > SRAM_LIMIT) {
                    std::cerr << "SRAM overflow: need " << total_sram_bytes_padded << " bytes, but allocated "
                              << SRAM_LIMIT << " bytes\n";
                    std::abort();
                }

                auto copy_rows_from_dev = [&](RPPdeviceptr sramDst, RPPdeviceptr devSrc, int elem_bytes) {
                    for (int h = 0; h < tile_H; ++h) {
                        rtMemcpyAsync((void *) (sramDst + (RPPdeviceptr) h * row_bytes_bf16),
                                      (const void *) (devSrc + (RPPdeviceptr) h * W * elem_bytes), W * elem_bytes,
                                      rtMemcpyDeviceToSram, ctx.kernelStream);
                    }
                };

                auto convert_rows_f32_to_bf16 = [&](RPPdeviceptr sramSrcCompact, RPPdeviceptr sramDstPadded, int rows) {
                    for (int h = 0; h < rows; ++h) {
                        calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        cvt_kernel_param_init(
                            threadsPerBlock, (uint32_t) (sramSrcCompact + (RPPdeviceptr) h * W * (int) sizeof(float)),
                            (uint32_t) (sramDstPadded + (RPPdeviceptr) h * row_bytes_bf16), kFLOAT, kBF16, params);
                        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params,
                                           ctx.rppBinMod, ctx.kernelStream);
                    }
                };

                if (in0_bytes_per_element == (int) sizeof(float)) {
                    rtMemcpyAsync((void *) sramACompact, (const void *) (devA + (RPPdeviceptr) devA_offset),
                                  sizeA_compact, rtMemcpyDeviceToSram, ctx.kernelStream);
                    convert_rows_f32_to_bf16(sramACompact, sramAPadded, tile_H);
                } else {
                    copy_rows_from_dev(sramAPadded, devA + (RPPdeviceptr) devA_offset, (int) sizeof(uint16_t));
                }

                if (in1_bytes_per_element == (int) sizeof(float)) {
                    rtMemcpyAsync((void *) sramBCompact, (const void *) (devB + (RPPdeviceptr) devB_offset),
                                  sizeB_compact, rtMemcpyDeviceToSram, ctx.kernelStream);
                    if (axis == -1 || axis == 0) {
                        convert_rows_f32_to_bf16(sramBCompact, sramBPadded, tile_H);
                    } else {
                        calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        cvt_kernel_param_init(threadsPerBlock, (uint32_t) sramBCompact, (uint32_t) sramBPadded, kFLOAT,
                                              kBF16, params);
                        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params,
                                           ctx.rppBinMod, ctx.kernelStream);
                    }
                } else {
                    if (axis == -1 || axis == 0) {
                        copy_rows_from_dev(sramBPadded, devB + (RPPdeviceptr) devB_offset, (int) sizeof(uint16_t));
                    } else {
                        rtMemcpyAsync((void *) sramBPadded, (const void *) (devB + (RPPdeviceptr) devB_offset),
                                      W * (int) sizeof(uint16_t), rtMemcpyDeviceToSram, ctx.kernelStream);
                    }
                }

                if (type == RPP_ELEMWISE_DIV) {
                    RPPdeviceptr reciprocalInput = ab_switched ? sramAPadded : sramBPadded;
                    const int    reciprocalRows  = (ab_switched || axis == -1 || axis == 0) ? tile_H : 1;
                    for (int h = 0; h < reciprocalRows; ++h) {
                        calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        params.emplace_back((uint32_t) (reciprocalInput + (RPPdeviceptr) h * row_bytes_bf16));
                        params.emplace_back((uint32_t) (reciprocalInput + (RPPdeviceptr) h * row_bytes_bf16));
                        params.emplace_back((uint32_t) sramReciprocalTable);
                        params.emplace_back(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                                            sizeof(uint16_t));
                        launchWrapperAysnc("mish_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                           ctx.kernelStream);
                    }
                }

                threadsPerBlock.x = padded_W;
                threadsPerBlock.y = 1;
                threadsPerBlock.z = 1;
                blocksPerGrid.x   = 1;
                blocksPerGrid.y   = 1;
                blocksPerGrid.z   = 1;
                while (threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= 32) {
                    threadsPerBlock.y++;
                }

                params.clear();
                params.emplace_back((uint32_t) sramAPadded);
                params.emplace_back((uint32_t) sramBPadded);
                params.emplace_back((uint32_t) sramOutPadded);
                params.emplace_back(row_bytes_bf16);
                params.emplace_back((axis == 1 || axis == 6 || axis == 7) ? 0 : row_bytes_bf16);
                params.emplace_back(tile_H);
                params.emplace_back(padded_W);
                params.emplace_back((axis == 1 || axis == 6 || axis == 7) ? 0 : padded_W);
                if (type == RPP_ELEMWISE_ADD) {
                    launchWrapperAysnc("elementwise_add", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                } else if (type == RPP_ELEMWISE_MUL || type == RPP_ELEMWISE_DIV) {
                    launchWrapperAysnc("elementwise_mul", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                } else {
                    assert(0);
                }

                if (out_bytes_per_element == (int) sizeof(float)) {
                    for (int h = 0; h < tile_H; ++h) {
                        calc_tbdim_flattern(1, W * 2, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        cvt_kernel_param_init_opt(
                            threadsPerBlock, (uint32_t) (sramOutPadded + (RPPdeviceptr) h * row_bytes_bf16),
                            (uint32_t) (sramOutCompact + (RPPdeviceptr) h * W * (int) sizeof(float)), kBF16, kFLOAT,
                            params);
                        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params,
                                           ctx.rppBinMod, ctx.kernelStream);
                    }
                    rtMemcpyAsync((void *) (devC + (RPPdeviceptr) devC_offset), (const void *) sramOutCompact, sizeC,
                                  rtMemcpySramToDevice, ctx.kernelStream);
                } else {
                    for (int h = 0; h < tile_H; ++h) {
                        rtMemcpyAsync((void *) (devC + (RPPdeviceptr) devC_offset +
                                                (RPPdeviceptr) h * W * (int) sizeof(uint16_t)),
                                      (const void *) (sramOutPadded + (RPPdeviceptr) h * row_bytes_bf16),
                                      W * (int) sizeof(uint16_t), rtMemcpySramToDevice, ctx.kernelStream);
                    }
                }

                continue;
            }

            rtMemcpyAsync((void *) sramA, (const void *) (devA + (RPPdeviceptr) devA_offset), sizeA,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            if (in0_bytes_per_element == (int) sizeof(float)) {
                calc_tbdim_flattern(tile_H, W, threadsPerBlock, blocksPerGrid);
                params.clear();
                cvt_kernel_param_init(threadsPerBlock, (uint32_t) sramA, (uint32_t) sramA, kFLOAT, kBF16, params);
                launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }

            rtMemcpyAsync((void *) sramB, (const void *) (devB + (RPPdeviceptr) devB_offset), sizeB,
                          rtMemcpyDeviceToSram, ctx.kernelStream);
            if (in1_bytes_per_element == (int) sizeof(float)) {
                if (axis == -1 || axis == 0) {
                    calc_tbdim_flattern(tile_H, W, threadsPerBlock, blocksPerGrid);
                } else if (axis == 1 || axis == 6 || axis == 7) {
                    calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                } else if (axis == 2) {
                    calc_tbdim_flattern(1, tile_H, threadsPerBlock, blocksPerGrid);
                } else {
                    throw std::runtime_error("Unsupported elementwise broadcast axis");
                }
                params.clear();
                cvt_kernel_param_init(threadsPerBlock, (uint32_t) sramB, (uint32_t) sramB, kFLOAT, kBF16, params);
                launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            }

            if (type == RPP_ELEMWISE_DIV) {
                if (ab_switched) {
                    calc_tbdim_flattern(tile_H, W, threadsPerBlock, blocksPerGrid);
                    params.clear();
                    params.emplace_back((uint32_t) sramA);
                    params.emplace_back((uint32_t) sramA);
                    params.emplace_back((uint32_t) sramReciprocalTable);
                    params.emplace_back(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(uint16_t));
                    launchWrapperAysnc("mish_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                } else {
                    if (axis == -1 || axis == 0) {
                        calc_tbdim_flattern(tile_H, W, threadsPerBlock, blocksPerGrid);
                    } else if (axis == 1 || axis == 6 || axis == 7) {
                        calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                    } else if (axis == 2) {
                        calc_tbdim_flattern(1, tile_H, threadsPerBlock, blocksPerGrid);
                    } else {
                        throw std::runtime_error("Unsupported elementwise broadcast axis");
                    }
                    params.clear();
                    params.emplace_back((uint32_t) sramB);
                    params.emplace_back((uint32_t) sramB);
                    params.emplace_back((uint32_t) sramReciprocalTable);
                    params.emplace_back(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z * sizeof(uint16_t));
                    launchWrapperAysnc("mish_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                }
            }

            if (W * tile_H < 8192) {
                threadsPerBlock.x = W;
                threadsPerBlock.y = 1;
                threadsPerBlock.z = tile_H;
                blocksPerGrid.x   = 1;
                blocksPerGrid.y   = 1;
                blocksPerGrid.z   = 1;
            } else if (W < 8192) {
                threadsPerBlock.x = W;
                threadsPerBlock.y = 1;
                threadsPerBlock.z = 1;
                blocksPerGrid.x   = 1;
                blocksPerGrid.y   = 1;
                blocksPerGrid.z   = tile_H;
            } else {
                throw std::runtime_error("Elementwise Width is Too Big");
            }

            while (threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= 32) {
                threadsPerBlock.y++;
            }

            const uint32_t input0 = (uint32_t) sramA;
            const uint32_t input1 = (uint32_t) sramB;
            const uint32_t out_addr =
                out_bytes_per_element == (int) sizeof(float) ? (uint32_t) sramA : (uint32_t) sramC;
            uint32_t loop     = blocksPerGrid.z;
            uint32_t offset0  = threadsPerBlock.x * threadsPerBlock.z * sizeof(uint16_t);
            uint32_t stridez0 = threadsPerBlock.x;
            uint32_t offset1  = threadsPerBlock.x * threadsPerBlock.z * sizeof(uint16_t);
            uint32_t stridez1 = threadsPerBlock.x;
            blocksPerGrid.z   = 1;
            if (axis == 1 || axis == 6 || axis == 7) {
                offset1  = 0;
                stridez1 = 0;
            }

            params.clear();
            if (axis == 2) {
                if (threadsPerBlock.z > 1) {
                    unsigned int tmp  = threadsPerBlock.z;
                    threadsPerBlock.z = threadsPerBlock.y;
                    threadsPerBlock.y = tmp;
                }
                params.emplace_back(input0);
                params.emplace_back(input1);
                params.emplace_back(out_addr);
                params.emplace_back(threadsPerBlock.x * threadsPerBlock.y * sizeof(uint16_t));
                params.emplace_back(threadsPerBlock.y * sizeof(uint16_t));
                if (type == RPP_ELEMWISE_ADD) {
                    params.emplace_back(0);
                } else if (type == RPP_ELEMWISE_MUL || type == RPP_ELEMWISE_DIV) {
                    params.emplace_back(1);
                } else {
                    assert(0);
                }
                blocksPerGrid.z = loop;
                launchWrapperAysnc("opt_binary_bc_x", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);
            } else {
                params.emplace_back(input0);
                params.emplace_back(input1);
                params.emplace_back(out_addr);
                params.emplace_back(offset0);
                params.emplace_back(offset1);
                params.emplace_back(loop);
                params.emplace_back(stridez0);
                params.emplace_back(stridez1);
                if (type == RPP_ELEMWISE_ADD) {
                    launchWrapperAysnc("elementwise_add", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                } else if (type == RPP_ELEMWISE_MUL || type == RPP_ELEMWISE_DIV) {
                    launchWrapperAysnc("elementwise_mul", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                } else {
                    assert(0);
                }
            }

            if (out_bytes_per_element == (int) sizeof(float)) {
                params.clear();
                calc_tbdim_flattern(tile_H * 2, W, threadsPerBlock, blocksPerGrid);
                cvt_kernel_param_init_opt(threadsPerBlock, (uint32_t) sramA, (uint32_t) sramC, kBF16, kFLOAT, params);
                launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                   ctx.kernelStream);

                rtMemcpyAsync((void *) (devC + (RPPdeviceptr) devC_offset), (const void *) sramC, sizeC,
                              rtMemcpySramToDevice, ctx.kernelStream);
            } else {
                rtMemcpyAsync((void *) (devC + (RPPdeviceptr) devC_offset), (const void *) sramC, sizeC,
                              rtMemcpySramToDevice, ctx.kernelStream);
            }
        }
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }

    if (reciprocal_table != nullptr) {
        free(reciprocal_table);
    }
}

inline void rpp_elementwise_build_tiled_pipeline(rpp_kernel_context & ctx,
                                                 rppElementWiseType   type,
                                                 int                  C,
                                                 int                  H,
                                                 int                  W,
                                                 int                  axis,
                                                 int                  in0_bytes_per_element,
                                                 int                  in1_bytes_per_element,
                                                 int                  out_bytes_per_element,
                                                 int                  is_instantial = 1) {
    dim3                  threadsPerBlock;
    dim3                  blocksPerGrid;
    std::vector<uint32_t> params;
    RPPdeviceptr          devA;
    RPPdeviceptr          devB;
    bool                  ab_switched = false;

    if (axis <= 2 || axis == 6) {
        devA = ctx.dev_in[0];
        devB = ctx.dev_in[1];
    } else {
        ab_switched           = true;
        devA                  = ctx.dev_in[1];
        devB                  = ctx.dev_in[0];
        int tmp               = in1_bytes_per_element;
        in1_bytes_per_element = in0_bytes_per_element;
        in0_bytes_per_element = tmp;
        if (axis != 7) {
            axis = axis - 3;
        }
    }

    RPPdeviceptr devC = ctx.dev_out[0];

    void * reciprocal_table       = nullptr;
    int    reciprocal_table_bytes = 0;
    if (type == RPP_ELEMWISE_DIV) {
        reciprocal_table_bytes = 65536 * (int) sizeof(uint16_t);
        reciprocal_table       = malloc(reciprocal_table_bytes);
        for (uint32_t i = 0; i < 65536; ++i) {
            uint32_t x = i;
            x <<= 16;
            float y                            = 1.0f / *(float *) &x;
            ((uint16_t *) reciprocal_table)[i] = rpp::bfloat16::round_to_bfloat16(y).value;
        }
        ((uint16_t *) reciprocal_table)[0]      = 0;
        ((uint16_t *) reciprocal_table)[0x8000] = 0;
        rtMemcpy((void *) ctx.dev_workspace, reciprocal_table, reciprocal_table_bytes, rtMemcpyHostToDevice);
    }

    rppStreamBeginCapture(ctx.kernelStream, RPP_STREAM_CAPTURE_MODE_GLOBAL);
    rppModuleLoad(&ctx.rppBinMod, "rpp_kernel/elementwise.o");

    RPPdeviceptr sram_base                 = ctx.virtual_sram_base;
    RPPdeviceptr sramReciprocalTable       = sram_base;
    const int    padded_W                  = rpp_elementwise_round_up_to_32(W);
    const bool   use_row_tail_padding      = padded_W != W;
    const int    SRAM_LIMIT                = 22 * 1024 * 1024;
    // Pipeline path reserves SRAM for ping-pong staging, so each chunk only uses part of total SRAM.
    const int    PIPELINE_TILE_SRAM_BUDGET = SRAM_LIMIT / 2;
    const int    one_row_sram_bytes        = rpp_elementwise_get_tile_sram_bytes(
        1, W, axis, in0_bytes_per_element, in1_bytes_per_element, out_bytes_per_element, reciprocal_table_bytes);
    const int tile_sram_budget = std::max(PIPELINE_TILE_SRAM_BUDGET, one_row_sram_bytes);
    const int max_tile_h =
        rpp_elementwise_get_max_tile_rows(H, W, axis, in0_bytes_per_element, in1_bytes_per_element,
                                          out_bytes_per_element, reciprocal_table_bytes, tile_sram_budget);

    if (type == RPP_ELEMWISE_DIV) {
        rtMemcpyAsync((void *) sramReciprocalTable, (const void *) ctx.dev_workspace, reciprocal_table_bytes,
                      rtMemcpyDeviceToSram, ctx.kernelStream);
    }

    const int max_tile_total_bytes =
        rpp_elementwise_get_tile_sram_bytes(max_tile_h, W, axis, in0_bytes_per_element, in1_bytes_per_element,
                                            out_bytes_per_element, reciprocal_table_bytes);
    const int    workspace_bytes_per_ping = rpp_elementwise_round_up(max_tile_total_bytes - reciprocal_table_bytes);
    RPPdeviceptr sramTileBase0            = sramReciprocalTable + reciprocal_table_bytes;
    RPPdeviceptr sramTileBase1            = sramTileBase0 + workspace_bytes_per_ping;
    const int total_pipeline_sram_bytes   = (int) (sramTileBase1 + (RPPdeviceptr) workspace_bytes_per_ping - sram_base);
    if (total_pipeline_sram_bytes > SRAM_LIMIT) {
        std::cerr << "SRAM overflow: need " << total_pipeline_sram_bytes << " bytes, but allocated " << SRAM_LIMIT
                  << " bytes\n";
        std::abort();
    }
    auto sram_tile_base = [&](int ping) {
        return ping ? sramTileBase1 : sramTileBase0;
    };

    auto copy_rows_from_dev_async = [&](RPPdeviceptr sramDst, RPPdeviceptr devSrc, int rows, int row_bytes_dst,
                                        int elem_bytes, RPPstream stream) {
        for (int h = 0; h < rows; ++h) {
            rtMemcpyAsync((void *) (sramDst + (RPPdeviceptr) h * row_bytes_dst),
                          (const void *) (devSrc + (RPPdeviceptr) h * W * elem_bytes), W * elem_bytes,
                          rtMemcpyDeviceToSram, stream);
        }
    };

    struct tile_meta_t {
        int    tile_H;
        int    sizeA;
        int    sizeB;
        int    sizeC;
        int    row_bytes_bf16;
        int    sram_off_a_compact;
        int    sram_off_b_compact;
        int    sram_off_a_padded;
        int    sram_off_b_padded;
        int    sram_off_out_padded;
        int    sram_off_out_compact;
        size_t devA_offset;
        size_t devB_offset;
        size_t devC_offset;
    };

    auto schedule_dma = [&](int ping, const tile_meta_t & tile) {
        const int    tile_H      = tile.tile_H;
        const int    sizeA       = tile.sizeA;
        const int    sizeB       = tile.sizeB;
        const size_t devA_offset = tile.devA_offset;
        const size_t devB_offset = tile.devB_offset;

        const int total_sram_bytes =
            rpp_elementwise_get_tile_sram_bytes(tile_H, W, axis, in0_bytes_per_element, in1_bytes_per_element,
                                                out_bytes_per_element, reciprocal_table_bytes);
        assert(total_sram_bytes <= tile_sram_budget);
        assert(total_sram_bytes <= SRAM_LIMIT);

        rppStreamWaitEvent(ctx.dmaStream, ctx.kernel_done_ping[ping], 0);
        RPPdeviceptr sramTileBase = sram_tile_base(ping);
        if (use_row_tail_padding) {
            const int          row_bytes_bf16 = tile.row_bytes_bf16;
            const RPPdeviceptr sramACompact =
                tile.sram_off_a_compact >= 0 ? (sramTileBase + (RPPdeviceptr) tile.sram_off_a_compact) : 0;
            const RPPdeviceptr sramBCompact =
                tile.sram_off_b_compact >= 0 ? (sramTileBase + (RPPdeviceptr) tile.sram_off_b_compact) : 0;
            const RPPdeviceptr sramAPadded = sramTileBase + (RPPdeviceptr) tile.sram_off_a_padded;
            const RPPdeviceptr sramBPadded = sramTileBase + (RPPdeviceptr) tile.sram_off_b_padded;

            if (in0_bytes_per_element == (int) sizeof(float)) {
                rtMemcpyAsync((void *) sramACompact, (const void *) (devA + (RPPdeviceptr) devA_offset), sizeA,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
            } else {
                copy_rows_from_dev_async(sramAPadded, devA + (RPPdeviceptr) devA_offset, tile_H, row_bytes_bf16,
                                         (int) sizeof(uint16_t), ctx.dmaStream);
            }

            if (in1_bytes_per_element == (int) sizeof(float)) {
                rtMemcpyAsync((void *) sramBCompact, (const void *) (devB + (RPPdeviceptr) devB_offset), sizeB,
                              rtMemcpyDeviceToSram, ctx.dmaStream);
            } else {
                if (axis == -1 || axis == 0) {
                    copy_rows_from_dev_async(sramBPadded, devB + (RPPdeviceptr) devB_offset, tile_H, row_bytes_bf16,
                                             (int) sizeof(uint16_t), ctx.dmaStream);
                } else {
                    rtMemcpyAsync((void *) sramBPadded, (const void *) (devB + (RPPdeviceptr) devB_offset),
                                  W * (int) sizeof(uint16_t), rtMemcpyDeviceToSram, ctx.dmaStream);
                }
            }
        } else {
            RPPdeviceptr sramA = sramTileBase;
            RPPdeviceptr sramB = sramA + rpp_elementwise_round_up(sizeA);
            rtMemcpyAsync((void *) sramA, (const void *) (devA + (RPPdeviceptr) devA_offset), sizeA,
                          rtMemcpyDeviceToSram, ctx.dmaStream);
            rtMemcpyAsync((void *) sramB, (const void *) (devB + (RPPdeviceptr) devB_offset), sizeB,
                          rtMemcpyDeviceToSram, ctx.dmaStream);
        }
        rppEventRecord(ctx.dma_done_ping[ping], ctx.dmaStream);
    };

    // Mark ping buffers as available before first prefetch.
    rppEventRecord(ctx.kernel_done_ping[0], ctx.kernelStream);
    rppEventRecord(ctx.kernel_done_ping[1], ctx.kernelStream);

    for (int c = 0; c < C; ++c) {
        const size_t devA_offset_base = (size_t) c * (size_t) H * (size_t) W * (size_t) in0_bytes_per_element;
        size_t       devB_offset_base = 0;
        if (axis == -1) {
            devB_offset_base = (size_t) c * (size_t) H * (size_t) W * (size_t) in1_bytes_per_element;
        } else if (axis == 1) {
            devB_offset_base = (size_t) c * (size_t) W * (size_t) in1_bytes_per_element;
        } else if (axis == 2) {
            devB_offset_base = (size_t) c * (size_t) H * (size_t) in1_bytes_per_element;
        }
        const size_t devC_offset_base = (size_t) c * (size_t) H * (size_t) W * (size_t) out_bytes_per_element;

        const int nr_of_tiles_h = (H + max_tile_h - 1) / max_tile_h;
        if (nr_of_tiles_h <= 0) {
            continue;
        }

        std::vector<tile_meta_t> tiles;
        tiles.reserve(nr_of_tiles_h);
        for (int tile_idx = 0; tile_idx < nr_of_tiles_h; ++tile_idx) {
            const int    h_begin = tile_idx * max_tile_h;
            const int    tile_H  = std::min(max_tile_h, H - h_begin);
            const int    sizeA   = tile_H * W * in0_bytes_per_element;
            const int    sizeB   = rpp_elementwise_get_input1_bytes(axis, tile_H, W, in1_bytes_per_element);
            const int    sizeC   = tile_H * W * out_bytes_per_element;
            const size_t devA_offset =
                devA_offset_base + (size_t) h_begin * (size_t) W * (size_t) in0_bytes_per_element;
            size_t devB_offset = devB_offset_base;
            if (axis == -1 || axis == 0) {
                devB_offset += (size_t) h_begin * (size_t) W * (size_t) in1_bytes_per_element;
            } else if (axis == 2) {
                devB_offset += (size_t) h_begin * (size_t) in1_bytes_per_element;
            }
            const size_t devC_offset =
                devC_offset_base + (size_t) h_begin * (size_t) W * (size_t) out_bytes_per_element;
            tile_meta_t tile = {};
            tile.tile_H      = tile_H;
            tile.sizeA       = sizeA;
            tile.sizeB       = sizeB;
            tile.sizeC       = sizeC;
            tile.devA_offset = devA_offset;
            tile.devB_offset = devB_offset;
            tile.devC_offset = devC_offset;

            if (use_row_tail_padding) {
                const int row_bytes_bf16 = padded_W * (int) sizeof(uint16_t);
                const int sizeA_compact  = sizeA;
                const int sizeB_compact  = sizeB;
                const int sizeA_padded   = tile_H * row_bytes_bf16;
                const int sizeB_padded   = (axis == -1 || axis == 0) ? tile_H * row_bytes_bf16 : row_bytes_bf16;
                const int sizeOut_padded = tile_H * row_bytes_bf16;

                int sram_cursor        = 0;
                int sram_off_a_compact = -1;
                int sram_off_b_compact = -1;
                if (in0_bytes_per_element == (int) sizeof(float)) {
                    sram_off_a_compact = sram_cursor;
                    sram_cursor += rpp_elementwise_round_up(sizeA_compact);
                }
                if (in1_bytes_per_element == (int) sizeof(float)) {
                    sram_off_b_compact = sram_cursor;
                    sram_cursor += rpp_elementwise_round_up(sizeB_compact);
                }

                const int sram_off_a_padded = sram_cursor;
                sram_cursor += rpp_elementwise_round_up(sizeA_padded);
                const int sram_off_b_padded = sram_cursor;
                sram_cursor += rpp_elementwise_round_up(sizeB_padded);
                const int sram_off_out_padded = sram_cursor;
                sram_cursor += rpp_elementwise_round_up(sizeOut_padded);

                int sram_off_out_compact = -1;
                if (out_bytes_per_element == (int) sizeof(float)) {
                    if (sram_off_a_compact >= 0) {
                        sram_off_out_compact = sram_off_a_compact;
                    } else if (sram_off_b_compact >= 0) {
                        sram_off_out_compact = sram_off_b_compact;
                    } else {
                        sram_off_out_compact = sram_cursor;
                        sram_cursor += rpp_elementwise_round_up(sizeC);
                    }
                }

                const int total_sram_bytes_padded =
                    (int) ((sram_tile_base(1) + (RPPdeviceptr) sram_cursor) - sram_base);
                if (total_sram_bytes_padded > SRAM_LIMIT) {
                    std::cerr << "SRAM overflow: need " << total_sram_bytes_padded << " bytes, but allocated "
                              << SRAM_LIMIT << " bytes\n";
                    std::abort();
                }

                tile.row_bytes_bf16       = row_bytes_bf16;
                tile.sram_off_a_compact   = sram_off_a_compact;
                tile.sram_off_b_compact   = sram_off_b_compact;
                tile.sram_off_a_padded    = sram_off_a_padded;
                tile.sram_off_b_padded    = sram_off_b_padded;
                tile.sram_off_out_padded  = sram_off_out_padded;
                tile.sram_off_out_compact = sram_off_out_compact;
            }
            tiles.push_back(tile);
        }

        schedule_dma(0, tiles[0]);
        for (int tile_idx = 0; tile_idx < nr_of_tiles_h; ++tile_idx) {
            const int           ping        = tile_idx & 1;
            const tile_meta_t & tile        = tiles[tile_idx];
            const int           tile_H      = tile.tile_H;
            const int           sizeA       = tile.sizeA;
            const int           sizeB       = tile.sizeB;
            const int           sizeC       = tile.sizeC;
            const size_t        devA_offset = tile.devA_offset;
            const size_t        devB_offset = tile.devB_offset;
            const size_t        devC_offset = tile.devC_offset;

            rppStreamWaitEvent(ctx.kernelStream, ctx.dma_done_ping[ping], 0);
            if (tile_idx + 1 < nr_of_tiles_h) {
                schedule_dma((tile_idx + 1) & 1, tiles[tile_idx + 1]);
            }

            RPPdeviceptr sramTileBase = sram_tile_base(ping);
            if (use_row_tail_padding) {
                assert(axis != 2);
                assert(padded_W < 8192);

                const int          row_bytes_bf16 = tile.row_bytes_bf16;
                const RPPdeviceptr sramACompact =
                    tile.sram_off_a_compact >= 0 ? (sramTileBase + (RPPdeviceptr) tile.sram_off_a_compact) : 0;
                const RPPdeviceptr sramBCompact =
                    tile.sram_off_b_compact >= 0 ? (sramTileBase + (RPPdeviceptr) tile.sram_off_b_compact) : 0;
                const RPPdeviceptr sramAPadded   = sramTileBase + (RPPdeviceptr) tile.sram_off_a_padded;
                const RPPdeviceptr sramBPadded   = sramTileBase + (RPPdeviceptr) tile.sram_off_b_padded;
                const RPPdeviceptr sramOutPadded = sramTileBase + (RPPdeviceptr) tile.sram_off_out_padded;
                const RPPdeviceptr sramOutCompact =
                    tile.sram_off_out_compact >= 0 ? (sramTileBase + (RPPdeviceptr) tile.sram_off_out_compact) : 0;

                auto convert_rows_f32_to_bf16 = [&](RPPdeviceptr sramSrcCompact, RPPdeviceptr sramDstPadded, int rows) {
                    for (int h = 0; h < rows; ++h) {
                        calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        cvt_kernel_param_init(
                            threadsPerBlock, (uint32_t) (sramSrcCompact + (RPPdeviceptr) h * W * (int) sizeof(float)),
                            (uint32_t) (sramDstPadded + (RPPdeviceptr) h * row_bytes_bf16), kFLOAT, kBF16, params);
                        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params,
                                           ctx.rppBinMod, ctx.kernelStream);
                    }
                };

                if (in0_bytes_per_element == (int) sizeof(float)) {
                    convert_rows_f32_to_bf16(sramACompact, sramAPadded, tile_H);
                }

                if (in1_bytes_per_element == (int) sizeof(float)) {
                    if (axis == -1 || axis == 0) {
                        convert_rows_f32_to_bf16(sramBCompact, sramBPadded, tile_H);
                    } else {
                        calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        cvt_kernel_param_init(threadsPerBlock, (uint32_t) sramBCompact, (uint32_t) sramBPadded, kFLOAT,
                                              kBF16, params);
                        launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params,
                                           ctx.rppBinMod, ctx.kernelStream);
                    }
                }

                if (type == RPP_ELEMWISE_DIV) {
                    RPPdeviceptr reciprocalInput = ab_switched ? sramAPadded : sramBPadded;
                    const int    reciprocalRows  = (ab_switched || axis == -1 || axis == 0) ? tile_H : 1;
                    for (int h = 0; h < reciprocalRows; ++h) {
                        calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        params.emplace_back((uint32_t) (reciprocalInput + (RPPdeviceptr) h * row_bytes_bf16));
                        params.emplace_back((uint32_t) (reciprocalInput + (RPPdeviceptr) h * row_bytes_bf16));
                        params.emplace_back((uint32_t) sramReciprocalTable);
                        params.emplace_back(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                                            sizeof(uint16_t));
                        launchWrapperAysnc("mish_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                           ctx.kernelStream);
                    }
                }

                threadsPerBlock.x = padded_W;
                threadsPerBlock.y = 1;
                threadsPerBlock.z = 1;
                blocksPerGrid.x   = 1;
                blocksPerGrid.y   = 1;
                blocksPerGrid.z   = 1;
                while (threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= 32) {
                    threadsPerBlock.y++;
                }

                params.clear();
                params.emplace_back((uint32_t) sramAPadded);
                params.emplace_back((uint32_t) sramBPadded);
                params.emplace_back((uint32_t) sramOutPadded);
                params.emplace_back(row_bytes_bf16);
                params.emplace_back((axis == 1 || axis == 6 || axis == 7) ? 0 : row_bytes_bf16);
                params.emplace_back(tile_H);
                params.emplace_back(padded_W);
                params.emplace_back((axis == 1 || axis == 6 || axis == 7) ? 0 : padded_W);
                if (type == RPP_ELEMWISE_ADD) {
                    launchWrapperAysnc("elementwise_add", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                } else if (type == RPP_ELEMWISE_MUL || type == RPP_ELEMWISE_DIV) {
                    launchWrapperAysnc("elementwise_mul", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                } else {
                    assert(0);
                }

                if (out_bytes_per_element == (int) sizeof(float)) {
                    for (int h = 0; h < tile_H; ++h) {
                        calc_tbdim_flattern(1, W * 2, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        cvt_kernel_param_init_opt(
                            threadsPerBlock, (uint32_t) (sramOutPadded + (RPPdeviceptr) h * row_bytes_bf16),
                            (uint32_t) (sramOutCompact + (RPPdeviceptr) h * W * (int) sizeof(float)), kBF16, kFLOAT,
                            params);
                        launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params,
                                           ctx.rppBinMod, ctx.kernelStream);
                    }
                    rtMemcpyAsync((void *) (devC + (RPPdeviceptr) devC_offset), (const void *) sramOutCompact, sizeC,
                                  rtMemcpySramToDevice, ctx.kernelStream);
                } else {
                    for (int h = 0; h < tile_H; ++h) {
                        rtMemcpyAsync((void *) (devC + (RPPdeviceptr) devC_offset +
                                                (RPPdeviceptr) h * W * (int) sizeof(uint16_t)),
                                      (const void *) (sramOutPadded + (RPPdeviceptr) h * row_bytes_bf16),
                                      W * (int) sizeof(uint16_t), rtMemcpySramToDevice, ctx.kernelStream);
                    }
                }
            } else {
                RPPdeviceptr sramA = sramTileBase;
                RPPdeviceptr sramB = sramA + rpp_elementwise_round_up(sizeA);
                RPPdeviceptr sramC = sramB + rpp_elementwise_round_up(sizeB);

                if (in0_bytes_per_element == (int) sizeof(float)) {
                    calc_tbdim_flattern(tile_H, W, threadsPerBlock, blocksPerGrid);
                    params.clear();
                    cvt_kernel_param_init(threadsPerBlock, (uint32_t) sramA, (uint32_t) sramA, kFLOAT, kBF16, params);
                    launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                }

                if (in1_bytes_per_element == (int) sizeof(float)) {
                    if (axis == -1 || axis == 0) {
                        calc_tbdim_flattern(tile_H, W, threadsPerBlock, blocksPerGrid);
                    } else if (axis == 1 || axis == 6 || axis == 7) {
                        calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                    } else if (axis == 2) {
                        calc_tbdim_flattern(1, tile_H, threadsPerBlock, blocksPerGrid);
                    } else {
                        throw std::runtime_error("Unsupported elementwise broadcast axis");
                    }
                    params.clear();
                    cvt_kernel_param_init(threadsPerBlock, (uint32_t) sramB, (uint32_t) sramB, kFLOAT, kBF16, params);
                    launchWrapperAysnc("opt_vector_cvt_32_16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                }

                if (type == RPP_ELEMWISE_DIV) {
                    if (ab_switched) {
                        calc_tbdim_flattern(tile_H, W, threadsPerBlock, blocksPerGrid);
                        params.clear();
                        params.emplace_back((uint32_t) sramA);
                        params.emplace_back((uint32_t) sramA);
                        params.emplace_back((uint32_t) sramReciprocalTable);
                        params.emplace_back(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                                            sizeof(uint16_t));
                        launchWrapperAysnc("mish_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                           ctx.kernelStream);
                    } else {
                        if (axis == -1 || axis == 0) {
                            calc_tbdim_flattern(tile_H, W, threadsPerBlock, blocksPerGrid);
                        } else if (axis == 1 || axis == 6 || axis == 7) {
                            calc_tbdim_flattern(1, W, threadsPerBlock, blocksPerGrid);
                        } else if (axis == 2) {
                            calc_tbdim_flattern(1, tile_H, threadsPerBlock, blocksPerGrid);
                        } else {
                            throw std::runtime_error("Unsupported elementwise broadcast axis");
                        }
                        params.clear();
                        params.emplace_back((uint32_t) sramB);
                        params.emplace_back((uint32_t) sramB);
                        params.emplace_back((uint32_t) sramReciprocalTable);
                        params.emplace_back(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                                            sizeof(uint16_t));
                        launchWrapperAysnc("mish_f16", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                           ctx.kernelStream);
                    }
                }

                if (W * tile_H < 8192) {
                    threadsPerBlock.x = W;
                    threadsPerBlock.y = 1;
                    threadsPerBlock.z = tile_H;
                    blocksPerGrid.x   = 1;
                    blocksPerGrid.y   = 1;
                    blocksPerGrid.z   = 1;
                } else if (W < 8192) {
                    threadsPerBlock.x = W;
                    threadsPerBlock.y = 1;
                    threadsPerBlock.z = 1;
                    blocksPerGrid.x   = 1;
                    blocksPerGrid.y   = 1;
                    blocksPerGrid.z   = tile_H;
                } else {
                    throw std::runtime_error("Elementwise Width is Too Big");
                }

                while (threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= 32) {
                    threadsPerBlock.y++;
                }

                const uint32_t input0 = (uint32_t) sramA;
                const uint32_t input1 = (uint32_t) sramB;
                const uint32_t out_addr =
                    out_bytes_per_element == (int) sizeof(float) ? (uint32_t) sramA : (uint32_t) sramC;
                uint32_t loop     = blocksPerGrid.z;
                uint32_t offset0  = threadsPerBlock.x * threadsPerBlock.z * sizeof(uint16_t);
                uint32_t stridez0 = threadsPerBlock.x;
                uint32_t offset1  = threadsPerBlock.x * threadsPerBlock.z * sizeof(uint16_t);
                uint32_t stridez1 = threadsPerBlock.x;
                blocksPerGrid.z   = 1;
                if (axis == 1 || axis == 6 || axis == 7) {
                    offset1  = 0;
                    stridez1 = 0;
                }

                params.clear();
                if (axis == 2) {
                    if (threadsPerBlock.z > 1) {
                        unsigned int tmp  = threadsPerBlock.z;
                        threadsPerBlock.z = threadsPerBlock.y;
                        threadsPerBlock.y = tmp;
                    }
                    params.emplace_back(input0);
                    params.emplace_back(input1);
                    params.emplace_back(out_addr);
                    params.emplace_back(threadsPerBlock.x * threadsPerBlock.y * sizeof(uint16_t));
                    params.emplace_back(threadsPerBlock.y * sizeof(uint16_t));
                    if (type == RPP_ELEMWISE_ADD) {
                        params.emplace_back(0);
                    } else if (type == RPP_ELEMWISE_MUL || type == RPP_ELEMWISE_DIV) {
                        params.emplace_back(1);
                    } else {
                        assert(0);
                    }
                    blocksPerGrid.z = loop;
                    launchWrapperAysnc("opt_binary_bc_x", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                       ctx.kernelStream);
                } else {
                    params.emplace_back(input0);
                    params.emplace_back(input1);
                    params.emplace_back(out_addr);
                    params.emplace_back(offset0);
                    params.emplace_back(offset1);
                    params.emplace_back(loop);
                    params.emplace_back(stridez0);
                    params.emplace_back(stridez1);
                    if (type == RPP_ELEMWISE_ADD) {
                        launchWrapperAysnc("elementwise_add", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                           ctx.kernelStream);
                    } else if (type == RPP_ELEMWISE_MUL || type == RPP_ELEMWISE_DIV) {
                        launchWrapperAysnc("elementwise_mul", blocksPerGrid, threadsPerBlock, params, ctx.rppBinMod,
                                           ctx.kernelStream);
                    } else {
                        assert(0);
                    }
                }

                if (out_bytes_per_element == (int) sizeof(float)) {
                    params.clear();
                    calc_tbdim_flattern(tile_H * 2, W, threadsPerBlock, blocksPerGrid);
                    cvt_kernel_param_init_opt(threadsPerBlock, (uint32_t) sramA, (uint32_t) sramC, kBF16, kFLOAT,
                                              params);
                    launchWrapperAysnc("opt_vector_cvt_f16_f32_opt", blocksPerGrid, threadsPerBlock, params,
                                       ctx.rppBinMod, ctx.kernelStream);

                    rtMemcpyAsync((void *) (devC + (RPPdeviceptr) devC_offset), (const void *) sramC, sizeC,
                                  rtMemcpySramToDevice, ctx.kernelStream);
                } else {
                    rtMemcpyAsync((void *) (devC + (RPPdeviceptr) devC_offset), (const void *) sramC, sizeC,
                                  rtMemcpySramToDevice, ctx.kernelStream);
                }
            }

            rppEventRecord(ctx.kernel_done_ping[ping], ctx.kernelStream);
        }
    }

    rppStreamEndCapture(ctx.kernelStream, &ctx.graph);
    if (is_instantial) {
        rppGraphInstantiate(&ctx.graphexec, ctx.graph, NULL, NULL, 0);
    }

    if (reciprocal_table != nullptr) {
        free(reciprocal_table);
    }
}

// -----------------------------
// Build graph once
// -----------------------------
static void rpp_elementwise_build(rpp_kernel_context & ctx,
                                  rppElementWiseType   type,
                                  int                  C,
                                  int                  H,
                                  int                  W,
                                  int                  axis,
                                  int                  in0_bytes_per_element,
                                  int                  in1_bytes_per_element,
                                  int                  out_bytes_per_element,
                                  int                  use_pipeline  = 0,
                                  int                  is_instantial = 1) {
    // if (use_pipeline) {
    //     rpp_elementwise_build_tiled_pipeline(ctx, type, C, H, W, axis, in0_bytes_per_element, in1_bytes_per_element,
    //                                         out_bytes_per_element, is_instantial);
    // } else {
    //     rpp_elementwise_build_tiled(ctx, type, C, H, W, axis, in0_bytes_per_element, in1_bytes_per_element,
    //                             out_bytes_per_element, is_instantial);

    // }
    rpp_elementwise_build_tiled(ctx, type, C, H, W, axis, in0_bytes_per_element, in1_bytes_per_element,
                                out_bytes_per_element, is_instantial);
}
