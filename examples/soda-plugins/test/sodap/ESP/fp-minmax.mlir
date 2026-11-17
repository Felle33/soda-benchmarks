// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-legalize-fp-minmax,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" | mlir-translate --mlir-to-llvmir > %t.ll
// RUN: clang -c %t.ll -o %t.o
// RUN: c++ -std=c++17 %S/Inputs/test_fp_minmax.cpp %t.o -o %t
// RUN: %t | FileCheck %s
// CHECK: FP32 minimum/maximum PASS
func.func @minimum(%a:f32,%b:f32)->f32 {
  %r=arith.minimumf %a,%b:f32
  return %r:f32
}
func.func @maximum(%a:f32,%b:f32)->f32 {
  %r=arith.maximumf %a,%b:f32
  return %r:f32
}
