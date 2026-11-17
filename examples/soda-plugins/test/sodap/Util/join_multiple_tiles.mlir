// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-outline-operators,sodap-join-multiple-tiles,func.func(canonicalize))" | FileCheck %s

// 2mm's shape, two GEMMs of different shapes: one tile with its shapes as
// arguments, launched twice. C and D share the descriptor's C, alpha and beta
// have their fields, and C is read only when beta isn't 0.

// CHECK:      module @forward_kernel attributes {sodap.invocations = [array<i64: 4, 7, 5>, array<i64: 4, 5, 8>], sodap.operator = "gemm", sodap.tile = 0 : i64, sodap.tile_dir = "tiles/forward_kernel"}
// CHECK-NEXT:   func.func @forward_kernel(
// CHECK-SAME:     %[[A:[^:]*]]: tensor<?x?xf32> {sodap.descriptor_field = 5 : i64, sodap.descriptor_ld = 8 : i64, sodap.dims = array<i64: 3, 4>},
// CHECK-SAME:     %[[B:[^:]*]]: tensor<?x?xf32> {sodap.descriptor_field = 6 : i64, sodap.descriptor_ld = 9 : i64, sodap.dims = array<i64: 4, 5>},
// CHECK-SAME:     %[[C:[^:]*]]: tensor<?x?xf32> {sodap.descriptor_field = 7 : i64, sodap.descriptor_ld = 10 : i64, sodap.dims = array<i64: 3, 5>, sodap.read_unless_zero = 7 : i64},
// CHECK-SAME:     %[[M:[^:]*]]: index {sodap.descriptor_field = 2 : i64},
// CHECK-SAME:     %[[K:[^:]*]]: index {sodap.descriptor_field = 4 : i64},
// CHECK-SAME:     %[[N:[^:]*]]: index {sodap.descriptor_field = 3 : i64},
// CHECK-SAME:     %[[ALPHA:[^:]*]]: f32 {sodap.descriptor_field = 11 : i64},
// CHECK-SAME:     %[[BETA:[^:]*]]: f32 {sodap.descriptor_field = 12 : i64})
// CHECK-SAME:     -> (tensor<?x?xf32> {sodap.descriptor_field = 7 : i64, sodap.descriptor_ld = 10 : i64, sodap.dims = array<i64: 3, 5>})
// CHECK-NEXT:     tensor.expand_shape %[[A]] {{\[}}[0, 1], [2]] output_shape [1, %[[M]], %[[K]]]
// CHECK-NEXT:     tensor.expand_shape %[[B]] {{\[}}[0, 1], [2]] output_shape [1, %[[K]], %[[N]]]
// CHECK-NEXT:     tosa.matmul
// CHECK-NEXT:     tensor.collapse_shape
// CHECK-NEXT:     tensor.from_elements %[[ALPHA]]
// CHECK-NEXT:     tensor.from_elements %[[BETA]]
// CHECK:          tosa.select
// CHECK-NOT:  module @gemm_
// CHECK:      func.func @forward(%[[X:.*]]: tensor<f32>, %[[Y:.*]]: tensor<4x7xf32>, %[[Z:.*]]: tensor<7x5xf32>, %[[W:.*]]: tensor<5x8xf32>, %[[V:.*]]: tensor<4x8xf32>)
// CHECK-DAG:    %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:    %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:    %[[C7:.*]] = arith.constant 7 : index
// CHECK-DAG:    %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:    %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[NONE:.*]] = tensor.empty() : tensor<4x5xf32>
// CHECK-NEXT:   %[[T:.*]] = dataflow.launch @forward_kernel::@forward_kernel(%[[Y]], %[[Z]], %[[NONE]], %[[C4]], %[[C7]], %[[C5]], %[[ONE]], %[[ZERO]]) : (tensor<4x7xf32>, tensor<7x5xf32>, tensor<4x5xf32>, index, index, index, f32, f32) -> tensor<4x5xf32>
// CHECK-NEXT:   %[[BE:.*]] = tensor.extract %[[X]][]
// CHECK-NEXT:   %[[D:.*]] = dataflow.launch @forward_kernel::@forward_kernel(%[[T]], %[[W]], %[[V]], %[[C4]], %[[C5]], %[[C8]], %[[ONE]], %[[BE]]) : (tensor<4x5xf32>, tensor<5x8xf32>, tensor<4x8xf32>, index, index, index, f32, f32) -> tensor<4x8xf32>
// CHECK-NEXT:   return %[[D]]
func.func @forward(%beta: tensor<f32>, %a: tensor<4x7xf32>, %b: tensor<7x5xf32>, %c: tensor<5x8xf32>, %d: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 7>} : (tensor<4x7xf32>) -> tensor<1x4x7xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 7, 5>} : (tensor<7x5xf32>) -> tensor<1x7x5xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x7xf32>, tensor<1x7x5xf32>) -> tensor<1x4x5xf32>
  %3 = tosa.reshape %c {new_shape = array<i64: 1, 5, 8>} : (tensor<5x8xf32>) -> tensor<1x5x8xf32>
  %4 = tosa.matmul %2, %3 : (tensor<1x4x5xf32>, tensor<1x5x8xf32>) -> tensor<1x4x8xf32>
  %5 = tosa.reshape %4 {new_shape = array<i64: 4, 8>} : (tensor<1x4x8xf32>) -> tensor<4x8xf32>
  %6 = tosa.reshape %beta {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
  %7 = tosa.mul %6, %d {shift = 0 : i8} : (tensor<1x1xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %8 = tosa.add %7, %5 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %8 : tensor<4x8xf32>
}
