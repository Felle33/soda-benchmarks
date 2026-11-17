// RUN: mkdir -p %t.dir
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-network-host{directory=%t.dir})" -o %t.dir/host.mlir
// RUN: FileCheck %s < %t.dir/host.mlir
// RUN: soda-opt %t.dir/host.mlir --lower-all-to-llvm=use-bare-ptr-memref-call-conv --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o %t.dir/host.ll
// RUN: /usr/bin/clang-19 -shared -fPIC -O2 -ffp-contract=off %t.dir/host.ll %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp -I %S/../../../include -o %t.dir/native.so
// RUN: python3 %S/Inputs/check_unified_host.py %t.dir expression
// CHECK-NOT: call @esp_op_repack
// CHECK: call @esp_op_scalar
// CHECK: call @esp_op_scalar
// CHECK: call @esp_op_gemm_full_source
func.func @forward(%a:tensor<1x2x3xf32>,%b:tensor<1x3x2xf32>,%c:tensor<2x2xf32>,%alpha:tensor<f32>,%beta:tensor<f32>)->tensor<2x2xf32> {
 %m=tosa.matmul %a,%b : (tensor<1x2x3xf32>,tensor<1x3x2xf32>)->tensor<1x2x2xf32>
 %r=tosa.reshape %m {new_shape=array<i64:2,2>} : (tensor<1x2x2xf32>)->tensor<2x2xf32>
 %ar=tosa.reshape %alpha {new_shape=array<i64:1,1>} : (tensor<f32>)->tensor<1x1xf32>
 %br=tosa.reshape %beta {new_shape=array<i64:1,1>} : (tensor<f32>)->tensor<1x1xf32>
 %x=tosa.mul %r,%ar {shift=0:i8} : (tensor<2x2xf32>,tensor<1x1xf32>)->tensor<2x2xf32>
 %y=tosa.mul %c,%br {shift=0:i8} : (tensor<2x2xf32>,tensor<1x1xf32>)->tensor<2x2xf32>
 %z=tosa.add %x,%y : (tensor<2x2xf32>,tensor<2x2xf32>)->tensor<2x2xf32>
 return %z:tensor<2x2xf32>
}
