// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-outline-kernel)" | FileCheck %s --check-prefix=OUTLINE
// OUTLINE: sodap.kernel_argument_order = array<i64: 0, 1, 2>
// The outline records the public ABI, independent of first use of operands.
module {
  func.func @forward(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %c = tosa.add %b, %a : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %c : tensor<2x2xf32>
  }
}
