// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef SODAP_REUSABLE_OPERATOR_ABI_H
#define SODAP_REUSABLE_OPERATOR_ABI_H
#include <stdint.h>
#include "sodap/ExecutionEngine/ReusableGemmABI.h"

// One instance of each engine shares an ESP socket and a DMA arena. Header
// fields always occupy the low 32 bits of eight-byte slots. Legacy v1/v2
// tensors use one FP32 per eight bytes and wire addresses count 32-bit words.
// Packed v3 tensors use beat-aligned packed rows and byte addresses; field 31
// records the DMA width. GEMM uses its matching v1/v2 descriptor format.
// Public APIs use byte offsets into the DATA area; the runtime adds the header.
enum {
  OP_PACKED_VERSION = 3, OP_DMA_WIDTH = 31, ADD_COLUMNS = 7,
  OP_VERSION = 1, OP_SLOT_BYTES = 8, OP_HEADER_FIELDS = 32,
  OP_HEADER_BYTES = OP_HEADER_FIELDS * OP_SLOT_BYTES,
  OP_CONV_MAGIC = 0x434f4e56, OP_ADD_MAGIC = 0x41444431,
  CONV_N = 2, CONV_IH, CONV_IW, CONV_IC, CONV_OC, CONV_KH, CONV_KW,
  CONV_OH, CONV_OW, CONV_SH, CONV_SW, CONV_PT, CONV_PL, CONV_DH, CONV_DW,
  CONV_RELU, CONV_A, CONV_W, CONV_BIAS, CONV_C, CONV_ALPHA,
  CONV_ACTIVATION = CONV_RELU, CONV_VERSION = 2,
  OP_ACT_NONE = 0, OP_ACT_RELU = 1, OP_ACT_LEAKY_RELU = 2,
  ADD_COUNT = 2, ADD_A, ADD_B, ADD_C, ADD_RELU
};

#ifdef __cplusplus
extern "C" {
#endif
int64_t esp_op_open(int64_t data_bytes, int64_t instance);
int64_t esp_op_open_packed(int64_t data_bytes, int64_t instance, int64_t dma_width);
// Bind a logical tensor to its physical packed row layout before copying or
// invoking it. Rebinding an offset is legal after the previous tensor dies.
void esp_op_tensor(int64_t context, int64_t offset, int64_t elements, int64_t columns);
void esp_op_repack(int64_t context, int64_t source, int64_t target, int64_t elements);
void esp_op_close(int64_t context);
void esp_op_copy_in(int64_t context, int64_t offset, int64_t rank, void *descriptor);
void esp_op_copy_out(int64_t context, int64_t offset, int64_t rank, void *descriptor);
void esp_op_conv(int64_t context, int64_t a, int64_t w, int64_t bias, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ic, int64_t oc, int64_t kh, int64_t kw,
    int64_t oh, int64_t ow, int64_t sh, int64_t sw, int64_t pt, int64_t pl,
    int64_t dh, int64_t dw, int64_t relu);
// v2 convolution: activation mode plus float32 slope bits in an integer slot.
// v1 offsets are unchanged; new engines also accept old v1 descriptors.
void esp_op_conv_activation(int64_t context, int64_t a, int64_t w, int64_t bias, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ic, int64_t oc, int64_t kh, int64_t kw,
    int64_t oh, int64_t ow, int64_t sh, int64_t sw, int64_t pt, int64_t pl,
    int64_t dh, int64_t dw, int64_t mode, int64_t alpha_bits);
void esp_op_add(int64_t context, int64_t a, int64_t b, int64_t c, int64_t count,
    int64_t relu);
void esp_op_gemm(int64_t context, int64_t a, int64_t b, int64_t c,
    int64_t m, int64_t n, int64_t k);
void esp_op_pool(int64_t context, int64_t a, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ch, int64_t oh, int64_t ow,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t pt, int64_t pl,
    int64_t average);
void esp_op_transpose(int64_t context, int64_t a, int64_t c, int64_t rank,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t p0, int64_t p1, int64_t p2, int64_t p3);

// Optional native co-simulation backend. When compiled with
// SODAP_OP_EXTERNAL_EXECUTOR, hardware calls go here instead of the C++ oracle.
// It must execute the complete descriptor and drain all DMA before returning.
void esp_op_execute_rtl(uint32_t *arena, uint64_t bytes);
#ifdef __cplusplus
}
#endif
#endif
