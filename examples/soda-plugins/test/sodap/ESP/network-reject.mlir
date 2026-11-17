// RUN: mlir-opt %s --split-input-file --verify-diagnostics --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-network-host{directory=%t.dir})" -o /dev/null
func.func @forward(%a:tensor<2x3xf32>,%b:tensor<1x3xf32>)->tensor<2x3xf32> {
  // expected-error@+1 {{addition offload requires equal shapes}}
  %r=tosa.add %a,%b : (tensor<2x3xf32>,tensor<1x3xf32>)->tensor<2x3xf32>
  return %r:tensor<2x3xf32>
}
// -----
func.func @forward(%a:tensor<2x3xf32>,%b:tensor<2x3xf32>)->tensor<2x3xf32> {
  // expected-error@+1 {{invalid physical accelerator instance}}
  %r=tosa.add %a,%b {sodap.instance=-1:i64} : (tensor<2x3xf32>,tensor<2x3xf32>)->tensor<2x3xf32>
  return %r:tensor<2x3xf32>
}
// -----
func.func @forward(%a:tensor<2x3xf32>,%b:tensor<2x3xf32>)->tensor<2x3xf32> {
  %r=tosa.add %a,%b {sodap.implementation="wide",sodap.instance=0:i64} : (tensor<2x3xf32>,tensor<2x3xf32>)->tensor<2x3xf32>
  // expected-error@+1 {{different implementations assigned to the same physical device}}
  %s=tosa.add %r,%b {sodap.implementation="narrow",sodap.instance=0:i64} : (tensor<2x3xf32>,tensor<2x3xf32>)->tensor<2x3xf32>
  return %s:tensor<2x3xf32>
}
