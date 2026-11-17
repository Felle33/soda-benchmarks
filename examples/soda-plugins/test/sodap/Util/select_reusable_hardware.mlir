// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-select-reusable-hardware{lanes=64 plm-bytes=327680})" | FileCheck %s
// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-select-reusable-hardware{lanes=64 plm-bytes=1024})" 2>&1 | FileCheck %s --check-prefix=SMALL
// Invocation dimensions are irrelevant to hardware selection, including tails.
// CHECK: sodap.hardware_tile_sizes = {{\[\[}}128, 64, 64{{\]\]}}
// CHECK: sodap.hardware_tile_sizes = {{\[\[}}128, 64, 64{{\]\]}}
// SMALL: storage budget cannot service requested lanes
module {
  module @a attributes {sodap.operator = "gemm", sodap.invocations = [array<i64: 3, 5, 7>]} {}
  module @b attributes {sodap.operator = "gemm", sodap.invocations = [array<i64: 800, 1000, 900>]} {}
}
