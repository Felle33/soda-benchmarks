// RUN: mkdir -p %t.o %t.n %t.reject
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-select-kernel-tiles{schedule=overlap transform-file=%S/../../../../../scripts/lcs/transform.mlir report-dir=%t.o})" -o %t.o/tasks.mlir
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-select-kernel-tiles{schedule=nonoverlap transform-file=%S/../../../../../scripts/lcs/transform.mlir report-dir=%t.n})" -o %t.n/tasks.mlir
// RUN: python3 %S/Inputs/check_kernel_selection.py %t.o/tile_search.json %t.n/tile_search.json
// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-select-kernel-tiles{plm-bytes=1 transform-file=%S/../../../../../scripts/lcs/transform.mlir report-dir=%t.reject})" 2>&1 | FileCheck %s --check-prefix=REJECT
// REJECT: all whole-kernel tile candidates rejected
module attributes {torch.debug_module_name = "Gemm"} {
  module @forward_kernel attributes {sodap.tile = 0 : i64, sodap.tile_dir = "tiles/forward_kernel"} {
    func.func @forward_kernel(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<20x25xf32>, %arg3: tensor<20x30xf32>, %arg4: tensor<30x25xf32>) -> tensor<20x25xf32> {
      %0 = tosa.reshape %arg3 {new_shape = array<i64: 1, 20, 30>} : (tensor<20x30xf32>) -> tensor<1x20x30xf32>
      %1 = tosa.reshape %arg4 {new_shape = array<i64: 1, 30, 25>} : (tensor<30x25xf32>) -> tensor<1x30x25xf32>
      %2 = tosa.matmul %0, %1 : (tensor<1x20x30xf32>, tensor<1x30x25xf32>) -> tensor<1x20x25xf32>
      %3 = tosa.reshape %2 {new_shape = array<i64: 20, 25>} : (tensor<1x20x25xf32>) -> tensor<20x25xf32>
      %4 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
      %5 = tosa.mul %4, %3 {shift = 0 : i8} : (tensor<1x1xf32>, tensor<20x25xf32>) -> tensor<20x25xf32>
      %6 = tosa.reshape %arg1 {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
      %7 = tosa.mul %6, %arg2 {shift = 0 : i8} : (tensor<1x1xf32>, tensor<20x25xf32>) -> tensor<20x25xf32>
      %8 = tosa.add %7, %5 : (tensor<20x25xf32>, tensor<20x25xf32>) -> tensor<20x25xf32>
      return %8 : tensor<20x25xf32>
    }
  }
  func.func @forward(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<20x25xf32>, %arg3: tensor<20x30xf32>, %arg4: tensor<30x25xf32>) -> tensor<20x25xf32> {
    %0 = dataflow.launch @forward_kernel::@forward_kernel(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<f32>, tensor<f32>, tensor<20x25xf32>, tensor<20x30xf32>, tensor<30x25xf32>) -> tensor<20x25xf32>
    return %0 : tensor<20x25xf32>
  }
}

