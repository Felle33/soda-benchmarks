// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-launches-to-esp{argument-order-file=%t.args})" | FileCheck %s
// RUN: cat %t.args | FileCheck %s --check-prefix=ORDER
// CHECK-NOT: sodap.kernel_argument_order
// CHECK: func.func @forward_kernel
// CHECK-NOT: sodap.kernel_argument_order
// ORDER: 0 1
module @forward_kernel attributes {sodap.tile = 0 : i64} {
  llvm.func @forward_kernel(!llvm.ptr)
}
func.func @main(%a: memref<4xf32>, %out: memref<2xf32>) attributes {sodap.kernel_argument_order = array<i64: 0, 1>} {
  %r = memref.alloc() : memref<2xf32>
  dataflow.launch @forward_kernel::@forward_kernel(%a) outs(%r : memref<2xf32>) : (memref<4xf32>) -> ()
  memref.copy %r, %out : memref<2xf32> to memref<2xf32>
  return
}

