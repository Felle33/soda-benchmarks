// RUN: mkdir -p %t.dir
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-network-host{directory=%t.dir})" | FileCheck %s
// CHECK: call @esp_op_gemm_source
// CHECK: call @esp_op_add
// CHECK: call @esp_op_add
// CHECK-NOT: call @esp_op_gemm_full_source
func.func @forward(%a:tensor<1x1x3xf32>,%b:tensor<1x3x2xf32>,%c:tensor<1x2xf32>)->tensor<1x2xf32> {
 %m=tosa.matmul %a,%b : (tensor<1x1x3xf32>,tensor<1x3x2xf32>)->tensor<1x1x2xf32>
 %r=tosa.reshape %m {new_shape=array<i64:1,2>} : (tensor<1x1x2xf32>)->tensor<1x2xf32>
 %z=tosa.add %r,%c : (tensor<1x2xf32>,tensor<1x2xf32>)->tensor<1x2xf32>
 %v=tosa.add %z,%r : (tensor<1x2xf32>,tensor<1x2xf32>)->tensor<1x2xf32>
 return %v:tensor<1x2xf32>
}
