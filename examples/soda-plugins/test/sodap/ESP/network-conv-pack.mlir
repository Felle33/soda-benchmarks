// RUN: mkdir -p %t.dir
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-network-host{directory=%t.dir})" -o %t.dir/host.mlir
// RUN: soda-opt %t.dir/host.mlir --lower-all-to-llvm=use-bare-ptr-memref-call-conv --convert-func-to-llvm --reconcile-unrealized-casts -o %t.dir/lowered.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.dir/lowered.mlir -o %t.dir/host.ll
// RUN: clang-19 -O2 -ffp-contract=off -shared -fPIC %t.dir/host.ll %S/../../../lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp -I%S/../../../include -o %t.dir/native.so
// RUN: python3 %S/Inputs/check_conv_pack.py %t.dir
// A persistent OHWI constant is also read in ordinary tensor order. Setup packs
// a separate matrix-view binding; repeated runs must preserve both meanings.
func.func @forward(%x:tensor<1x1x2x2xf32>)->tensor<1x1x2x3xf32> {
 %w= "tosa.const"() {value=dense<[[[[1.0,2.0]]],[[[3.0,4.0]]],[[[5.0,6.0]]]]>:tensor<3x1x1x2xf32>} : ()->tensor<3x1x1x2xf32>
 %b= "tosa.const"() {value=dense<0.0>:tensor<3xf32>} : ()->tensor<3xf32>
 %c=tosa.conv2d %x,%w,%b {pad=array<i64:0,0,0,0>,stride=array<i64:1,1>,dilation=array<i64:1,1>} : (tensor<1x1x2x2xf32>,tensor<3x1x1x2xf32>,tensor<3xf32>)->tensor<1x1x2x3xf32>
 %r=tosa.reshape %w {new_shape=array<i64:1,1,2,3>} : (tensor<3x1x1x2xf32>)->tensor<1x1x2x3xf32>
 %y=tosa.add %c,%r : (tensor<1x1x2x3xf32>,tensor<1x1x2x3xf32>)->tensor<1x1x2x3xf32>
 return %y:tensor<1x1x2x3xf32>
}
