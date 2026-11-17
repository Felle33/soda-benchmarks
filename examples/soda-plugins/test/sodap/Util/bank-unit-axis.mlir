// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-bank-matmul-buffers{rows=8 cols=8})" | FileCheck %s
// CHECK: memref<16x20xf32>
// CHECK: memref.expand_shape
// CHECK-SAME: into memref<15x1x20xf32, strided<[20, 20, 1]>>
// CHECK: linalg.matmul
module {
  func.func @unit_axis() {
    %a = memref.alloc() : memref<15x20xf32>
    %b = memref.alloc() : memref<20x8xf32>
    %c = memref.alloc() : memref<15x8xf32>
    %expanded = memref.expand_shape %a [[0], [1, 2]] output_shape [15, 1, 20]
      : memref<15x20xf32> into memref<15x1x20xf32>
    %matrix = memref.subview %expanded[0, 0, 0] [15, 1, 20] [1, 1, 1]
      : memref<15x1x20xf32> to memref<15x20xf32, strided<[20, 1]>>
    linalg.matmul ins(%matrix, %b : memref<15x20xf32, strided<[20, 1]>>, memref<20x8xf32>)
      outs(%c : memref<15x8xf32>)
    return
  }
}
