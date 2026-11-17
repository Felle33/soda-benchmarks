// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-outline-operators{source-semantics=true},sodap-join-multiple-tiles,func.func(canonicalize))" | FileCheck %s
// C presence is a separate descriptor scalar, not a comparison of beta to 0.
// CHECK: sodap.source_semantics
// CHECK: sodap.read_unless_zero = 8
// CHECK: sodap.descriptor_field = 14
// CHECK: sodap.descriptor_version = 3
// CHECK: tosa.equal
func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>, %c: tensor<2x4xf32>, %beta: tensor<f32>) -> tensor<2x4xf32> {
  %aa = tosa.reshape %a {new_shape = array<i64: 1,2,3>} : (tensor<2x3xf32>) -> tensor<1x2x3xf32>
  %bb = tosa.reshape %b {new_shape = array<i64: 1,3,4>} : (tensor<3x4xf32>) -> tensor<1x3x4xf32>
  %mm = tosa.matmul %aa,%bb : (tensor<1x2x3xf32>,tensor<1x3x4xf32>) -> tensor<1x2x4xf32>
  %m = tosa.reshape %mm {new_shape = array<i64: 2,4>} : (tensor<1x2x4xf32>) -> tensor<2x4xf32>
  %s = tosa.reshape %beta {new_shape = array<i64: 1,1>} : (tensor<f32>) -> tensor<1x1xf32>
  %bc = tosa.mul %s,%c {shift = 0 : i8} : (tensor<1x1xf32>,tensor<2x4xf32>) -> tensor<2x4xf32>
  %r = tosa.add %bc,%m : (tensor<2x4xf32>,tensor<2x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}
