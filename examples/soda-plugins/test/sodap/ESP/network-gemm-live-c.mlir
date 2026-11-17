// RUN: mkdir -p %t.dir
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-network-host{directory=%t.dir})" -o %t.dir/host.mlir
// RUN: FileCheck %s < %t.dir/host.mlir
// RUN: soda-opt %t.dir/host.mlir --lower-all-to-llvm=use-bare-ptr-memref-call-conv --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o %t.dir/host.ll
// RUN: /usr/bin/clang-19 -shared -fPIC -O2 -ffp-contract=off %t.dir/host.ll %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp -I %S/../../../include -o %t.dir/native.so
// RUN: python3 %S/Inputs/check_unified_host.py %t.dir live-c
// CHECK: call @esp_op_repack
// CHECK: call @esp_op_gemm_full_source
// CHECK: call @esp_op_add
func.func @forward(%a:tensor<1x1x3xf32>,%b:tensor<1x3x2xf32>,%c:tensor<1x2xf32>)->tensor<1x2xf32> {
 %m=tosa.matmul %a,%b : (tensor<1x1x3xf32>,tensor<1x3x2xf32>)->tensor<1x1x2xf32>
 %r=tosa.reshape %m {new_shape=array<i64:1,2>} : (tensor<1x1x2xf32>)->tensor<1x2xf32>
 %z=tosa.add %r,%c : (tensor<1x2xf32>,tensor<1x2xf32>)->tensor<1x2xf32>
 %v=tosa.add %z,%c : (tensor<1x2xf32>,tensor<1x2xf32>)->tensor<1x2xf32>
 return %v:tensor<1x2xf32>
}
