// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-tails))" | FileCheck %s

// A tile whose shapes are arguments (sodap-join-multiple-tiles), with sizes
// that need not divide them: each copy of a 2x4 tile of %a or %c moves the
// part inside the matrix, min(2, M - i) x min(4, K - j). A load first fills
// the local tile with zeros, only when the part is smaller; a store doesn't.
// %s has a static shape: its tiles are whole.

// CHECK-LABEL: func.func @top(
// CHECK-SAME:    %[[A:.*]]: memref<?x?xf32> {{{.*}}}, %[[M:.*]]: index, %[[K:.*]]: index, %[[C:.*]]: memref<?x?xf32> {{{.*}}}, %[[S:.*]]: memref<8x8xf32>)
// CHECK:         %[[BUF:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:         dataflow.task attributes {load} {
// CHECK:           scf.for %[[I:.*]] = %{{.*}} to %[[M]] step
// CHECK-NEXT:        scf.for %[[J:.*]] = %{{.*}} to %[[K]] step
// CHECK-NEXT:          %[[T2:.*]] = arith.constant 2 : index
// CHECK-NEXT:          %[[LEFTM:.*]] = arith.subi %[[M]], %[[I]]
// CHECK-NEXT:          %[[ROWS:.*]] = arith.minui %[[LEFTM]], %[[T2]]
// CHECK-NEXT:          %[[CUTM:.*]] = arith.cmpi ult, %[[ROWS]], %[[T2]]
// CHECK-NEXT:          %[[T4:.*]] = arith.constant 4 : index
// CHECK-NEXT:          %[[LEFTK:.*]] = arith.subi %[[K]], %[[J]]
// CHECK-NEXT:          %[[COLS:.*]] = arith.minui %[[LEFTK]], %[[T4]]
// CHECK-NEXT:          %[[CUTK:.*]] = arith.cmpi ult, %[[COLS]], %[[T4]]
// CHECK-NEXT:          %[[PARTIAL:.*]] = arith.ori %[[CUTM]], %[[CUTK]]
// CHECK-NEXT:          scf.if %[[PARTIAL]] {
// CHECK-NEXT:            %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:            linalg.fill ins(%[[ZERO]] : f32) outs(%[[BUF]] : memref<2x4xf32>)
// CHECK-NEXT:          }
// CHECK-NEXT:          %[[OUTER:.*]] = memref.subview %[[A]][%[[I]], %[[J]]] [%[[ROWS]], %[[COLS]]] [1, 1]
// CHECK-NEXT:          %[[INNER:.*]] = memref.subview %[[BUF]][0, 0] [%[[ROWS]], %[[COLS]]] [1, 1]
// CHECK-NEXT:          memref.copy %[[OUTER]], %[[INNER]]
// CHECK-NEXT:          %[[WHOLE:.*]] = memref.subview %[[S]][%[[I]], %[[J]]] [2, 4] [1, 1]
// CHECK-NEXT:          memref.copy %[[WHOLE]], %[[BUF]]
// CHECK:         dataflow.task attributes {store} {
// CHECK-NOT:       linalg.fill
// CHECK:             %[[SROWS:.*]] = arith.minui
// CHECK:             %[[SCOLS:.*]] = arith.minui
// CHECK-NEXT:        %[[SOUTER:.*]] = memref.subview %[[C]][%{{.*}}, %{{.*}}] [%[[SROWS]], %[[SCOLS]]] [1, 1]
// CHECK-NEXT:        %[[SINNER:.*]] = memref.subview %[[BUF]][0, 0] [%[[SROWS]], %[[SCOLS]]] [1, 1]
// CHECK-NEXT:        memref.copy %[[SINNER]], %[[SOUTER]]
func.func @top(%a: memref<?x?xf32> {sodap.dims = array<i64: 1, 2>},
               %m: index, %k: index,
               %c: memref<?x?xf32> {sodap.dims = array<i64: 1, 2>},
               %s: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  dataflow.dispatch {
    %buf = memref.alloc() : memref<2x4xf32>
    dataflow.task attributes {load} {
      scf.for %i = %c0 to %m step %c2 {
        scf.for %j = %c0 to %k step %c4 {
          %t = memref.subview %a[%i, %j] [2, 4] [1, 1] : memref<?x?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
          memref.copy %t, %buf : memref<2x4xf32, strided<[?, 1], offset: ?>> to memref<2x4xf32>
          %u = memref.subview %s[%i, %j] [2, 4] [1, 1] : memref<8x8xf32> to memref<2x4xf32, strided<[8, 1], offset: ?>>
          memref.copy %u, %buf : memref<2x4xf32, strided<[8, 1], offset: ?>> to memref<2x4xf32>
        }
      }
    }
    dataflow.task attributes {store} {
      scf.for %i = %c0 to %m step %c2 {
        scf.for %j = %c0 to %k step %c4 {
          %t = memref.subview %c[%i, %j] [2, 4] [1, 1] : memref<?x?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
          memref.copy %buf, %t : memref<2x4xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
        }
      }
    }
  }
  return
}
