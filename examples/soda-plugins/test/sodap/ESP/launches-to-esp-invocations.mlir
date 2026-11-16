// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-launches-to-esp{dma-data-width=64})" | FileCheck %s --check-prefix=PACKED
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-launches-to-esp)" | FileCheck %s

// 2mm's two launches of one GEMM tile (sodap-join-multiple-tiles): one
// arena, William's calls (ReusableGemmABI.h). The 128-byte header, then each
// buffer at 8 bytes an element: A 128, B 256, T 384, C 512, D 640. T stays in
// the arena; only the host's arguments are copied. The first launch has no
// C (an allocation only it touches: no copy, beta 0); the second's C, the
// host's %e, is copied into D's place with the other arguments, and the
// accelerator writes D over it. alpha and beta are the launches'.

// CHECK-LABEL: func.func @forward_kernel(
// CHECK-SAME:    %[[A:.*]]: memref<4x4xf32>, %[[B:.*]]: memref<4x4xf32>, %[[C:.*]]: memref<4x4xf32>, %[[D:.*]]: memref<4x4xf32>, %[[E:.*]]: memref<4x4xf32>, %[[ALPHA:.*]]: f32, %[[BETA:.*]]: f32)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[CTX:.*]] = call @esp_gemm_open(%c768_i64, %c0_i64)
// CHECK:         call @esp_gemm_copy_in(%[[CTX]], %c128_i64
// CHECK:         call @esp_gemm_copy_in(%[[CTX]], %c256_i64
// CHECK:         call @esp_gemm_copy_in(%[[CTX]], %c512_i64
// CHECK:         %[[EU:.*]] = memref.cast %[[E]]
// CHECK-NEXT:    call @esp_gemm_copy_in(%[[CTX]], %c640_i64{{.*}}, %[[EU]])
// CHECK-NOT:     esp_gemm_copy_in
// CHECK:         call @esp_gemm_invoke(%[[CTX]], %c128_i64{{.*}}, %c256_i64{{.*}}, %c384_i64{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %c32_i64{{.*}}, %c32_i64{{.*}}, %c32_i64{{.*}}, %[[ALPHA]], %cst
// CHECK-NOT:     esp_gemm_copy_in
// CHECK:         call @esp_gemm_invoke(%[[CTX]], %c384_i64{{.*}}, %c512_i64{{.*}}, %c640_i64{{.*}}, %[[ONE:.*]], %[[BETA]])
// CHECK:         call @esp_gemm_copy_out(%[[CTX]], %c640_i64
// CHECK:         call @esp_gemm_close(%[[CTX]])
// CHECK-NOT:     dataflow.launch
// CHECK-NOT:     module @forward_kernel
module {
  module @forward_kernel attributes {sodap.operator = "gemm", sodap.tile = 0 : i64} {
    func.func private @forward_kernel(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, index, index, index, f32, f32, memref<?x?xf32>)
  }
  func.func @forward(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>, %d: memref<4x4xf32>, %e: memref<4x4xf32>, %alpha: f32, %beta: f32) {
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %none = memref.alloc() : memref<4x4xf32>
    %t = memref.alloc() : memref<4x4xf32>
    dataflow.launch @forward_kernel::@forward_kernel(%a, %b, %none, %c4, %c4, %c4, %alpha, %zero) outs(%t : memref<4x4xf32>) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, index, index, index, f32, f32) -> ()
    %r = memref.alloc() : memref<4x4xf32>
    dataflow.launch @forward_kernel::@forward_kernel(%t, %c, %e, %c4, %c4, %c4, %one, %beta) outs(%r : memref<4x4xf32>) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, index, index, index, f32, f32) -> ()
    memref.copy %r, %d : memref<4x4xf32> to memref<4x4xf32>
    return
  }
}

// PACKED-LABEL: func.func @forward_kernel(
// PACKED: call @esp_gemm_open_packed(%c448_i64, %c0_i64, %c64_i64)
// PACKED: call @esp_gemm_copy_in(%{{.*}}, %c128_i64
// PACKED: call @esp_gemm_copy_in(%{{.*}}, %c192_i64
// PACKED: call @esp_gemm_copy_in(%{{.*}}, %c320_i64
// PACKED: call @esp_gemm_copy_in(%{{.*}}, %c384_i64
// PACKED: call @esp_gemm_invoke(%{{.*}}, %c128_i64{{.*}}, %c192_i64{{.*}}, %c256_i64{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %c16_i64
// PACKED: call @esp_gemm_invoke(%{{.*}}, %c256_i64{{.*}}, %c320_i64{{.*}}, %c384_i64
// PACKED: call @esp_gemm_copy_out(%{{.*}}, %c384_i64
