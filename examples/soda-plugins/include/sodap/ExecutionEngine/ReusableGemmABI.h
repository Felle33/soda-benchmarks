// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef SODAP_REUSABLE_GEMM_ABI_H
#define SODAP_REUSABLE_GEMM_ABI_H

#include <stdint.h>

// Shared by the MLIR generator and runtime. Descriptor fields always occupy
// the low 32 bits of an eight-byte slot, independently of the transport width.
// v1 matrix locations/strides count 32-bit words, with one FP32 per eight bytes.
// v2 locations/strides count bytes; FP32 rows are packed and beat aligned, and
// field 13 records the DMA width. Host APIs use bytes in both versions.
// No schedule choices occur in this interface.
enum {
  GEMM_MAGIC = 0x47454d4d,
  GEMM_VERSION = 1,
  GEMM_PACKED_VERSION = 2,
  GEMM_SLOT_BYTES = 8,
  GEMM_WORD_STRIDE = 2,
  GEMM_FIELD_MAGIC = 0,
  GEMM_FIELD_VERSION,
  GEMM_FIELD_M,
  GEMM_FIELD_N,
  GEMM_FIELD_K,
  GEMM_FIELD_A,
  GEMM_FIELD_B,
  GEMM_FIELD_C,
  GEMM_FIELD_LDA,
  GEMM_FIELD_LDB,
  GEMM_FIELD_LDC,
  GEMM_FIELD_ALPHA,
  GEMM_FIELD_BETA,
  GEMM_FIELD_DMA_WIDTH, // v2 only; byte addresses/strides, packed FP32 rows.
  GEMM_FIELD_COUNT = 16,
  GEMM_HEADER_BYTES = GEMM_FIELD_COUNT * GEMM_SLOT_BYTES
};

#ifdef __cplusplus
extern "C" {
#endif
int64_t esp_gemm_open(int64_t arena_bytes, int64_t instance);
int64_t esp_gemm_open_packed(int64_t arena_bytes, int64_t instance, int64_t dma_width);
void esp_gemm_close(int64_t context);
void esp_gemm_copy_in(int64_t context, int64_t offset_bytes,
                      int64_t rank, void *descriptor);
void esp_gemm_copy_out(int64_t context, int64_t offset_bytes,
                       int64_t rank, void *descriptor);
void esp_gemm_invoke(int64_t context, int64_t a_bytes, int64_t b_bytes,
                     int64_t c_bytes, int64_t m, int64_t n, int64_t k,
                     int64_t lda_bytes, int64_t ldb_bytes, int64_t ldc_bytes,
                     float alpha, float beta);
#ifdef __cplusplus
}
#endif
#endif
