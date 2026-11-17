// RUN: mlir-opt %s --split-input-file --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-outline-operators,func.func(canonicalize))" | FileCheck %s

// threemm's shape: one tile per matmul, 2-D interfaces; with no epilogue C
// is a tensor.empty, alpha 1 and beta 0. The model's reshapes cancel on the
// host and the third tile reads the first two's results.

// CHECK:      module @gemm_0 attributes {sodap.operator = "gemm", sodap.tile = 0 : i64, sodap.tile_dir = "tiles/gemm_0"}
// CHECK:        func.func @gemm_0(%[[TA:.*]]: tensor<4x6xf32>, %[[TB:.*]]: tensor<6x5xf32>, %[[TC:.*]]: tensor<4x5xf32>, %[[TALPHA:.*]]: f32, %[[TBETA:.*]]: f32) -> tensor<4x5xf32>
// CHECK:          %[[AB3:.*]] = tosa.matmul
// CHECK:          %[[AB:.*]] = tensor.collapse_shape %[[AB3]]
// CHECK:          %[[AT:.*]] = tensor.from_elements %[[TALPHA]] : tensor<1x1xf32>
// CHECK:          %[[BT:.*]] = tensor.from_elements %[[TBETA]] : tensor<1x1xf32>
// CHECK:          %[[SCALED:.*]] = tosa.mul %[[AT]], %[[AB]]
// CHECK:          %[[BC:.*]] = tosa.mul %[[BT]], %[[TC]]
// CHECK:          %[[SUM:.*]] = tosa.add %[[BC]], %[[SCALED]]
// CHECK:          %[[TZERO:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1xf32>}>
// CHECK:          %[[NOC:.*]] = tosa.equal %[[BT]], %[[TZERO]]
// CHECK:          %[[TD:.*]] = tosa.select %[[NOC]], %[[SCALED]], %[[SUM]]
// CHECK:          return %[[TD]]
// CHECK:      module @gemm_1 attributes {sodap.operator = "gemm", sodap.tile = 1 : i64
// CHECK:      module @gemm_2 attributes {sodap.operator = "gemm", sodap.tile = 2 : i64
// CHECK:      func.func @forward(%[[A:.*]]: tensor<4x6xf32>, %[[B:.*]]: tensor<6x5xf32>, %[[C:.*]]: tensor<5x8xf32>, %[[D:.*]]: tensor<8x7xf32>)
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:        %[[E0:.*]] = tensor.empty() : tensor<4x5xf32>
// CHECK-NEXT:   %[[E:.*]] = dataflow.launch @gemm_0::@gemm_0(%[[A]], %[[B]], %[[E0]], %[[ONE]], %[[ZERO]])
// CHECK-NEXT:   %[[F0:.*]] = tensor.empty() : tensor<5x7xf32>
// CHECK-NEXT:   %[[F:.*]] = dataflow.launch @gemm_1::@gemm_1(%[[C]], %[[D]], %[[F0]], %[[ONE]], %[[ZERO]])
// CHECK-NEXT:   %[[G0:.*]] = tensor.empty() : tensor<4x7xf32>
// CHECK-NEXT:   %[[G:.*]] = dataflow.launch @gemm_2::@gemm_2(%[[E]], %[[F]], %[[G0]], %[[ONE]], %[[ZERO]]) : (tensor<4x5xf32>, tensor<5x7xf32>, tensor<4x7xf32>, f32, f32) -> tensor<4x7xf32>
// CHECK-NEXT:   return %[[G]]
func.func @forward(%a: tensor<4x6xf32>, %b: tensor<6x5xf32>, %c: tensor<5x8xf32>, %d: tensor<8x7xf32>) -> tensor<4x7xf32> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 6>} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 6, 5>} : (tensor<6x5xf32>) -> tensor<1x6x5xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x6xf32>, tensor<1x6x5xf32>) -> tensor<1x4x5xf32>
  %3 = tosa.reshape %c {new_shape = array<i64: 1, 5, 8>} : (tensor<5x8xf32>) -> tensor<1x5x8xf32>
  %4 = tosa.reshape %d {new_shape = array<i64: 1, 8, 7>} : (tensor<8x7xf32>) -> tensor<1x8x7xf32>
  %5 = tosa.matmul %3, %4 : (tensor<1x5x8xf32>, tensor<1x8x7xf32>) -> tensor<1x5x7xf32>
  %6 = tosa.matmul %2, %5 : (tensor<1x4x5xf32>, tensor<1x5x7xf32>) -> tensor<1x4x7xf32>
  %7 = tosa.reshape %6 {new_shape = array<i64: 4, 7>} : (tensor<1x4x7xf32>) -> tensor<4x7xf32>
  return %7 : tensor<4x7xf32>
}

// -----

// Two matmuls with one signature: one tile, launched twice.

// CHECK:      module @gemm_0
// CHECK-NOT:  module @gemm_1
// CHECK:      func.func @forward
// CHECK:        %[[T1:.*]] = dataflow.launch @gemm_0::@gemm_0(%arg0, %arg1,
// CHECK:        dataflow.launch @gemm_0::@gemm_0(%[[T1]], %arg2,
func.func @forward(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 4>} : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 4, 4>} : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %3 = tosa.reshape %c {new_shape = array<i64: 1, 4, 4>} : (tensor<4x4xf32>) -> tensor<1x4x4xf32>
  %4 = tosa.matmul %2, %3 : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  %5 = tosa.reshape %4 {new_shape = array<i64: 4, 4>} : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
  return %5 : tensor<4x4xf32>
}

// -----

// gemm: alpha * (A.B) + beta * C, torch-mlir's ops, one launch with the
// model's alpha, beta and C.

// CHECK:       func.func @forward(
// CHECK-SAME:    %[[ALPHA2:.*]]: tensor<f32>, %[[BETA2:.*]]: tensor<f32>, %[[C2:.*]]: tensor<4x5xf32>, %[[A2:.*]]: tensor<4x6xf32>, %[[B2:.*]]: tensor<6x5xf32>)
// CHECK-NEXT:    %[[AL2:.*]] = tensor.extract %[[ALPHA2]][] : tensor<f32>
// CHECK-NEXT:    %[[BE2:.*]] = tensor.extract %[[BETA2]][] : tensor<f32>
// CHECK-NEXT:    %[[D2:.*]] = dataflow.launch @gemm_0::@gemm_0(%[[A2]], %[[B2]], %[[C2]], %[[AL2]], %[[BE2]]) : (tensor<4x6xf32>, tensor<6x5xf32>, tensor<4x5xf32>, f32, f32) -> tensor<4x5xf32>
// CHECK-NEXT:    return %[[D2]]
func.func @forward(%alpha: tensor<f32>, %beta: tensor<f32>, %c: tensor<4x5xf32>, %a: tensor<4x6xf32>, %b: tensor<6x5xf32>) -> tensor<4x5xf32> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 6>} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 6, 5>} : (tensor<6x5xf32>) -> tensor<1x6x5xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x6xf32>, tensor<1x6x5xf32>) -> tensor<1x4x5xf32>
  %3 = tosa.reshape %2 {new_shape = array<i64: 4, 5>} : (tensor<1x4x5xf32>) -> tensor<4x5xf32>
  %4 = tosa.reshape %alpha {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
  %5 = tosa.mul %4, %3 {shift = 0 : i8} : (tensor<1x1xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  %6 = tosa.reshape %beta {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
  %7 = tosa.mul %6, %c {shift = 0 : i8} : (tensor<1x1xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  %8 = tosa.add %7, %5 : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %8 : tensor<4x5xf32>
}

// -----

// 2mm: T = alpha * (A.B), beta = 0 and no C; D = T.C + beta * D, alpha = 1.

// CHECK:       func.func @forward(
// CHECK-SAME:    %[[ALPHA3:.*]]: tensor<f32>, %[[BETA3:.*]]: tensor<f32>, %[[A3:.*]]: tensor<4x6xf32>, %[[B3:.*]]: tensor<6x5xf32>, %[[C3:.*]]: tensor<5x7xf32>, %[[D3:.*]]: tensor<4x7xf32>)
// CHECK-DAG:     %[[ONE3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:     %[[ZERO3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[NONE3:.*]] = tensor.empty() : tensor<4x5xf32>
// CHECK-NEXT:    %[[AL3:.*]] = tensor.extract %[[ALPHA3]][]
// CHECK-NEXT:    %[[T3:.*]] = dataflow.launch @gemm_0::@gemm_0(%[[A3]], %[[B3]], %[[NONE3]], %[[AL3]], %[[ZERO3]])
// CHECK-NEXT:    %[[BE3:.*]] = tensor.extract %[[BETA3]][]
// CHECK-NEXT:    %[[R3:.*]] = dataflow.launch @gemm_1::@gemm_1(%[[T3]], %[[C3]], %[[D3]], %[[ONE3]], %[[BE3]])
// CHECK-NEXT:    return %[[R3]]
func.func @forward(%alpha: tensor<f32>, %beta: tensor<f32>, %a: tensor<4x6xf32>, %b: tensor<6x5xf32>, %c: tensor<5x7xf32>, %d: tensor<4x7xf32>) -> tensor<4x7xf32> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 6>} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 6, 5>} : (tensor<6x5xf32>) -> tensor<1x6x5xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x6xf32>, tensor<1x6x5xf32>) -> tensor<1x4x5xf32>
  %3 = tosa.reshape %2 {new_shape = array<i64: 4, 5>} : (tensor<1x4x5xf32>) -> tensor<4x5xf32>
  %4 = tosa.reshape %alpha {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
  %5 = tosa.mul %4, %3 {shift = 0 : i8} : (tensor<1x1xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  %6 = tosa.reshape %beta {new_shape = array<i64: 1, 1>} : (tensor<f32>) -> tensor<1x1xf32>
  %7 = tosa.mul %6, %d {shift = 0 : i8} : (tensor<1x1xf32>, tensor<4x7xf32>) -> tensor<4x7xf32>
  %8 = tosa.reshape %5 {new_shape = array<i64: 1, 4, 5>} : (tensor<4x5xf32>) -> tensor<1x4x5xf32>
  %9 = tosa.reshape %c {new_shape = array<i64: 1, 5, 7>} : (tensor<5x7xf32>) -> tensor<1x5x7xf32>
  %10 = tosa.matmul %8, %9 : (tensor<1x4x5xf32>, tensor<1x5x7xf32>) -> tensor<1x4x7xf32>
  %11 = tosa.reshape %10 {new_shape = array<i64: 4, 7>} : (tensor<1x4x7xf32>) -> tensor<4x7xf32>
  %12 = tosa.add %7, %11 : (tensor<4x7xf32>, tensor<4x7xf32>) -> tensor<4x7xf32>
  return %12 : tensor<4x7xf32>
}

// -----

// A shared intermediate: A.B is read by the add and returned, so the GEMM
// stops at A.B and the add stays on the host.

// CHECK:       func.func @forward(
// CHECK-SAME:    %[[A4:.*]]: tensor<4x6xf32>, %[[B4:.*]]: tensor<6x5xf32>, %[[C4:.*]]: tensor<4x5xf32>)
// CHECK:         %[[AB4:.*]] = dataflow.launch @gemm_0::@gemm_0(%[[A4]], %[[B4]], %{{.*}}, %{{.*}}, %{{.*}})
// CHECK-NEXT:    %[[SUM4:.*]] = tosa.add %[[C4]], %[[AB4]]
// CHECK-NEXT:    return %[[AB4]], %[[SUM4]]
func.func @forward(%a: tensor<4x6xf32>, %b: tensor<6x5xf32>, %c: tensor<4x5xf32>) -> (tensor<4x5xf32>, tensor<4x5xf32>) {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 6>} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 6, 5>} : (tensor<6x5xf32>) -> tensor<1x6x5xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x6xf32>, tensor<1x6x5xf32>) -> tensor<1x4x5xf32>
  %3 = tosa.reshape %2 {new_shape = array<i64: 4, 5>} : (tensor<1x4x5xf32>) -> tensor<4x5xf32>
  %4 = tosa.add %c, %3 : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %3, %4 : tensor<4x5xf32>, tensor<4x5xf32>
}

// -----

// The same add without a second use of A.B: C as it is, beta = 1.

// CHECK:       func.func @forward(
// CHECK-SAME:    %[[A5:.*]]: tensor<4x6xf32>, %[[B5:.*]]: tensor<6x5xf32>, %[[C5:.*]]: tensor<4x5xf32>)
// CHECK:         %[[ONE5:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %[[D5:.*]] = dataflow.launch @gemm_0::@gemm_0(%[[A5]], %[[B5]], %[[C5]], %[[ONE5]], %[[ONE5]])
// CHECK-NEXT:    return %[[D5]]
func.func @forward(%a: tensor<4x6xf32>, %b: tensor<6x5xf32>, %c: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %0 = tosa.reshape %a {new_shape = array<i64: 1, 4, 6>} : (tensor<4x6xf32>) -> tensor<1x4x6xf32>
  %1 = tosa.reshape %b {new_shape = array<i64: 1, 6, 5>} : (tensor<6x5xf32>) -> tensor<1x6x5xf32>
  %2 = tosa.matmul %0, %1 : (tensor<1x4x6xf32>, tensor<1x6x5xf32>) -> tensor<1x4x5xf32>
  %3 = tosa.reshape %2 {new_shape = array<i64: 4, 5>} : (tensor<1x4x5xf32>) -> tensor<4x5xf32>
  %4 = tosa.add %c, %3 : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  return %4 : tensor<4x5xf32>
}
