// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline='builtin.module(sodap-bank-matmul-buffers{rows=8 cols=8 plm-bytes=6000})' 2>&1 | FileCheck %s
// The logical buffers use 5200 bytes, but 8-way banking requires 7040 bytes.
// A configuration that fit before padding must fail against the same budget.
// CHECK: padded local memory exceeds the configured storage budget
// CHECK-SAME: 7040 bytes, budget 6000
module {
  func.func @compute() {
    %a = memref.alloc() : memref<10x30xf32>
    %b = memref.alloc() : memref<30x25xf32>
    %c = memref.alloc() : memref<10x25xf32>
    linalg.matmul ins(%a, %b : memref<10x30xf32>, memref<30x25xf32>) outs(%c : memref<10x25xf32>)
    return
  }
}
