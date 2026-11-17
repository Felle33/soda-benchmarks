// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter 2>/dev/null | mlir-opt --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --canonicalize --cse | FileCheck %s

// GEMM's shape in a tile with sodap.invocations, sizes 4 that divide no
// range: the matmul and every tile stay static (4x4); the element-wise
// epilogue runs only on the part of its tile inside the matrix,
// min(4, M - i) x min(4, N - j), on [0, 0] slices of the static tiles, and is
// inserted back. The rest of the tile is never stored (sodap-dataflow-tails).

// CHECK-DAG:  #[[MIN:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
// CHECK-LABEL: func.func @forward_kernel(
// CHECK-SAME:    %[[A:[^:]*]]: tensor<?x?xf32> {{{[^}]*}}}, %[[B:[^:]*]]: tensor<?x?xf32> {{{[^}]*}}}, %[[C:[^:]*]]: tensor<?x?xf32> {{{[^}]*}}}, %[[M:[^:]*]]: index, %[[K:[^:]*]]: index, %[[N:[^:]*]]: index,
// CHECK:         scf.for %[[I:.*]] = %{{.*}} to %[[M]] step %[[C4:.*]] iter_args(%[[D:.*]] = %{{.*}})
// CHECK-NEXT:      scf.for %[[J:.*]] = %{{.*}} to %[[N]] step %[[C4]] iter_args(%[[D2:.*]] = %[[D]])
// CHECK-NEXT:        %[[ROWS:.*]] = affine.min #[[MIN]](%[[I]])[%[[M]]]
// CHECK-NEXT:        %[[COLS:.*]] = affine.min #[[MIN]](%[[J]])[%[[N]]]
// CHECK:             linalg.fill {{.*}} -> tensor<4x4xf32>
// CHECK:             %[[AB:.*]] = scf.for
// CHECK:               linalg.matmul {{.*}} -> tensor<4x4xf32>
// CHECK:             %[[CT:.*]] = tensor.extract_slice %[[C]][%[[I]], %[[J]]] [4, 4] [1, 1]
// CHECK-NEXT:        %[[DT:.*]] = tensor.extract_slice %[[D2]][%[[I]], %[[J]]] [4, 4] [1, 1]
// CHECK-NEXT:        %[[ABP:.*]] = tensor.extract_slice %[[AB]][0, 0] [%[[ROWS]], %[[COLS]]] [1, 1] : tensor<4x4xf32> to tensor<?x?xf32>
// CHECK-NEXT:        %[[CP:.*]] = tensor.extract_slice %[[CT]][0, 0] [%[[ROWS]], %[[COLS]]] [1, 1] : tensor<4x4xf32> to tensor<?x?xf32>
// CHECK-NEXT:        %[[DP:.*]] = tensor.extract_slice %[[DT]][0, 0] [%[[ROWS]], %[[COLS]]] [1, 1] : tensor<4x4xf32> to tensor<?x?xf32>
// CHECK-NEXT:        %[[E:.*]] = linalg.generic {{.*}} ins(%[[ABP]], %[[CP]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[DP]] : tensor<?x?xf32>)
// CHECK:             %[[W:.*]] = tensor.insert_slice %[[E]] into %[[DT]][0, 0] [%[[ROWS]], %[[COLS]]] [1, 1] : tensor<?x?xf32> into tensor<4x4xf32>
// CHECK-NEXT:        tensor.insert_slice %[[W]] into %[[D2]][%[[I]], %[[J]]] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
#id = affine_map<(d0, d1) -> (d0, d1)>
module @forward_kernel attributes {sodap.invocations = [array<i64: 5, 6, 7>]} {
  func.func @forward_kernel(%A: tensor<?x?xf32> {sodap.dims = array<i64: 3, 4>},
                            %B: tensor<?x?xf32> {sodap.dims = array<i64: 4, 5>},
                            %C: tensor<?x?xf32> {sodap.dims = array<i64: 3, 5>},
                            %m: index, %k: index, %n: index, %alpha: f32) -> tensor<?x?xf32> {
    %zero = arith.constant 0.0 : f32
    %0 = dataflow.dispatch : tensor<?x?xf32> {
      %edge = bufferization.alloc_tensor(%m, %n) : tensor<?x?xf32>
      %1 = dataflow.task attributes {compute} : tensor<?x?xf32> {
        %e = tensor.empty(%m, %n) : tensor<?x?xf32>
        %fill = linalg.fill ins(%zero : f32) outs(%e : tensor<?x?xf32>) -> tensor<?x?xf32>
        %mm = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
        %d = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%mm, %C : tensor<?x?xf32>, tensor<?x?xf32>) outs(%edge : tensor<?x?xf32>) {
        ^bb0(%x: f32, %c: f32, %o: f32):
          %s = arith.mulf %alpha, %x : f32
          %r = arith.addf %s, %c : f32
          linalg.yield %r : f32
        } -> tensor<?x?xf32>
        dataflow.yield %d : tensor<?x?xf32>
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
