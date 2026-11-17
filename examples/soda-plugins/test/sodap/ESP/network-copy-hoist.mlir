// RUN: mkdir -p %t.dir
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-network-host{directory=%t.dir})" -o %t.dir/host.mlir
// RUN: soda-opt %t.dir/host.mlir --lower-all-to-llvm=use-bare-ptr-memref-call-conv --convert-func-to-llvm --reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o %t.dir/host.ll
// RUN: /usr/bin/clang-19 -shared -fPIC -O2 -ffp-contract=off -DSODAP_OP_PROFILE -fno-exceptions -fno-rtti %t.dir/host.ll %S/Inputs/copy_profile.cpp %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp %S/../../../lib/sodap/ExecutionEngine/esp_prof.c -I %S/../../../include -I %S/../../../include/sodap/ExecutionEngine -o %t.dir/native.so
// RUN: python3 %S/Inputs/check_copy_host.py %t.dir hoist
func.func @forward(%a:tensor<1x2x3xf32>,%b:tensor<1x3x2xf32>,%d:tensor<1x4xf32>)->tensor<1x2x2xf32> {
 %m=tosa.matmul %a,%b : (tensor<1x2x3xf32>,tensor<1x3x2xf32>)->tensor<1x2x2xf32>
 %r=tosa.reshape %d {new_shape=array<i64:1,2,2>} : (tensor<1x4xf32>)->tensor<1x2x2xf32>
 %z=tosa.add %m,%r : (tensor<1x2x2xf32>,tensor<1x2x2xf32>)->tensor<1x2x2xf32>
 return %z:tensor<1x2x2xf32>
}
