// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-extract-hardware)" | FileCheck %s
// CHECK-NOT: @source
// CHECK: module @engine attributes {sodap.tile = 0 : i64}
// CHECK: func.func @kernel
// CHECK-NOT: @source
func.func @source(%a:tensor<2x3xf32>)->tensor<2x3xf32> { return %a:tensor<2x3xf32> }
module @engine attributes {sodap.tile=0:i64} {
  func.func @kernel(%a:tensor<?x?xf32>)->tensor<?x?xf32> { return %a:tensor<?x?xf32> }
}
