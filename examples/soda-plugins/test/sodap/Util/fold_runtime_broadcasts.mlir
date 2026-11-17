// RUN: mlir-opt %s --split-input-file --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-linalg-fold-runtime-broadcasts))" | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (0, d1)>
#col = affine_map<(d0, d1) -> (d0, 0)>
#scalar = affine_map<(d0, d1) -> ()>

// tosa-to-linalg's broadcasts of %x (M x N) to M x N at run time: along d0
// when M is 1, then along d1 when dim 1 of the first result is 1. Both are
// copies; %x stays. The rank-0 tensor of %alpha becomes %alpha in the body.

// CHECK-LABEL: func.func @copies(
// CHECK-SAME:    %[[X:.*]]: tensor<?x?xf32>, %[[M:.*]]: index, %[[N:.*]]: index, %[[ALPHA:.*]]: f32)
// CHECK-NOT:     scf.if
// CHECK:         %[[COPY:.*]] = linalg.copy ins(%[[X]] : tensor<?x?xf32>)
// CHECK-NOT:     scf.if
// CHECK:         linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel"]} ins(%[[COPY]] : tensor<?x?xf32>)
// CHECK-NEXT:    ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:      %[[P:.*]] = arith.mulf %[[ALPHA]], %[[IN]] : f32
// CHECK-NEXT:      linalg.yield %[[P]]
func.func @copies(%x0: tensor<?x?xf32>, %m: index, %n: index, %alpha: f32) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %e0 = tensor.empty(%m, %n) : tensor<?x?xf32>
  %x = linalg.copy ins(%x0 : tensor<?x?xf32>) outs(%e0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %dim0 = tensor.dim %x, %c0 : tensor<?x?xf32>
  %max0 = arith.maxui %m, %dim0 : index
  %is1 = arith.cmpi eq, %dim0, %c1 : index
  %y = scf.if %is1 -> (tensor<?x?xf32>) {
    %e = tensor.empty(%max0, %n) : tensor<?x?xf32>
    %b = linalg.generic {indexing_maps = [#row, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<?x?xf32>) outs(%e : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    scf.yield %b : tensor<?x?xf32>
  } else {
    scf.yield %x : tensor<?x?xf32>
  }
  %dim1 = tensor.dim %y, %c1 : tensor<?x?xf32>
  %is1b = arith.cmpi eq, %dim1, %c1 : index
  %z = scf.if %is1b -> (tensor<?x?xf32>) {
    %d0 = tensor.dim %y, %c0 : tensor<?x?xf32>
    %e = tensor.empty(%d0, %n) : tensor<?x?xf32>
    %b = linalg.generic {indexing_maps = [#col, #id], iterator_types = ["parallel", "parallel"]} ins(%y : tensor<?x?xf32>) outs(%e : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    scf.yield %b : tensor<?x?xf32>
  } else {
    scf.yield %y : tensor<?x?xf32>
  }
  %a = tensor.from_elements %alpha : tensor<f32>
  %e1 = tensor.empty(%m, %n) : tensor<?x?xf32>
  %r = linalg.generic {indexing_maps = [#scalar, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%a, %z : tensor<f32>, tensor<?x?xf32>) outs(%e1 : tensor<?x?xf32>) {
  ^bb0(%s: f32, %in: f32, %out: f32):
    %p = arith.mulf %s, %in : f32
    linalg.yield %p : f32
  } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (0, d1)>

// A real broadcast: the result has K rows, not %x's M; it stays.

// CHECK-LABEL: func.func @broadcast(
// CHECK:         scf.if
// CHECK:           linalg.generic
func.func @broadcast(%x0: tensor<?x?xf32>, %m: index, %k: index, %n: index) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %e0 = tensor.empty(%m, %n) : tensor<?x?xf32>
  %x = linalg.copy ins(%x0 : tensor<?x?xf32>) outs(%e0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %is1 = arith.cmpi eq, %m, %c1 : index
  %y = scf.if %is1 -> (tensor<?x?xf32>) {
    %e = tensor.empty(%k, %n) : tensor<?x?xf32>
    %b = linalg.generic {indexing_maps = [#row, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<?x?xf32>) outs(%e : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    scf.yield %b : tensor<?x?xf32>
  } else {
    scf.yield %x : tensor<?x?xf32>
  }
  return %y : tensor<?x?xf32>
}
