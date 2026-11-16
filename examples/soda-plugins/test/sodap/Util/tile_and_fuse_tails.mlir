// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter 2>&1 | FileCheck %s

// A tile from sodap-join-multiple-tiles (sodap.invocations): M=5, K=6, N=7
// with tile sizes 4, 4, 4, which divide none of them. The sizes are accepted
// and the tiles stay static: every min(4, range - offset) becomes 4, so the
// last tile reaches past the matrix (sodap-dataflow-tails cuts its transfers).

// CHECK:      remark: tile_sizes = {{\[\[}}4, 4, 4]]
// CHECK:      func.func @forward_kernel(
// CHECK-NOT:    affine.min
// CHECK:        %[[T:.*]] = arith.constant 4 : index
// CHECK:        scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %[[T]] iter_args
// CHECK:          scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK-NOT:    affine.min
// CHECK:            scf.for %[[KK:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args
// CHECK-NOT:    affine.min
// CHECK:              linalg.matmul
// CHECK-NOT:    affine.min
// CHECK:        return
module @forward_kernel attributes {sodap.invocations = [array<i64: 5, 6, 7>]} {
  func.func @forward_kernel(%A: tensor<?x?xf32> {sodap.dims = array<i64: 2, 3>},
                            %B: tensor<?x?xf32> {sodap.dims = array<i64: 3, 4>},
                            %m: index, %k: index, %n: index) -> tensor<?x?xf32> {
    %zero = arith.constant 0.0 : f32
    %0 = dataflow.dispatch : tensor<?x?xf32> {
      %edge = bufferization.alloc_tensor(%m, %n) : tensor<?x?xf32>
      %1 = dataflow.task attributes {compute} : tensor<?x?xf32> {
        %fill = linalg.fill ins(%zero : f32) outs(%edge : tensor<?x?xf32>) -> tensor<?x?xf32>
        %mm = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
        dataflow.yield %mm : tensor<?x?xf32>
      }
      dataflow.yield %1 : tensor<?x?xf32>
    }
    return %0 : tensor<?x?xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %tasks {tile_sizes = [[4, 4, 4]]} : !transform.any_op
    transform.yield
  }
}
