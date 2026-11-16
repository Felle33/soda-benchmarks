// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-ping-pong))" | FileCheck %s

// An element-wise op cut to the part of its tile inside the matrix
// (tile_and_fuse), as fold-memref-alias-ops leaves it: one subview
// [i, j][min(4, M - i), min(4, N - j)] of the shared buffer. Ping-pong splits
// it into the static 4x4 tile the store task reads and a [0, 0] part of it:
// the slots stay 4x4, the op writes the part of its slot, and the token is
// written after the op, not after the view.

// CHECK-LABEL: func.func @part(
// CHECK:         %[[S:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>
// CHECK:         %[[SLOTS:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK:         dataflow.task attributes {compute} {
// CHECK:             %[[ROWS:.*]] = affine.min
// CHECK-NEXT:        %[[COLS:.*]] = affine.min
// CHECK:             %[[SLOT:.*]] = memref.subview %[[SLOTS]][%{{.*}}, 0, 0] [1, 4, 4] [1, 1, 1]
// CHECK-NEXT:        %[[PART:.*]] = memref.subview %[[SLOT]][0, 0] [%[[ROWS]], %[[COLS]]] [1, 1]
// CHECK-NEXT:        %[[CAST:.*]] = memref.cast %[[PART]]
// CHECK-NEXT:        linalg.generic {{.*}} outs(%[[CAST]] :
// CHECK:             }
// CHECK-NEXT:        dataflow.stream_write %[[S]]
// CHECK:         dataflow.task attributes {store} {
// CHECK:             dataflow.stream_peek %[[S]]
// CHECK:             memref.subview %[[SLOTS]][%{{.*}}, 0, 0] [1, 4, 4] [1, 1, 1]
// CHECK-NEXT:        memref.copy
// CHECK-NEXT:        dataflow.stream_read %[[S]]
#min = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
#id = affine_map<(d0, d1) -> (d0, d1)>
func.func @part(%m: index, %n: index, %out: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %one = arith.constant 1.0 : f32
  dataflow.dispatch {
    %d = memref.alloc(%m, %n) : memref<?x?xf32>
    dataflow.task attributes {compute} {
      scf.for %i = %c0 to %m step %c4 {
        scf.for %j = %c0 to %n step %c4 {
          %rows = affine.min #min(%i)[%m]
          %cols = affine.min #min(%j)[%n]
          %part = memref.subview %d[%i, %j] [%rows, %cols] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          linalg.generic {indexing_maps = [#id], iterator_types = ["parallel", "parallel"]} outs(%part : memref<?x?xf32, strided<[?, 1], offset: ?>>) {
          ^bb0(%o: f32):
            linalg.yield %one : f32
          }
        }
      }
    }
    dataflow.task attributes {store} {
      scf.for %i = %c0 to %m step %c4 {
        scf.for %j = %c0 to %n step %c4 {
          %tile = memref.subview %d[%i, %j] [4, 4] [1, 1] : memref<?x?xf32> to memref<4x4xf32, strided<[?, 1], offset: ?>>
          %dst = memref.subview %out[%i, %j] [4, 4] [1, 1] : memref<?x?xf32> to memref<4x4xf32, strided<[?, 1], offset: ?>>
          memref.copy %tile, %dst : memref<4x4xf32, strided<[?, 1], offset: ?>> to memref<4x4xf32, strided<[?, 1], offset: ?>>
        }
      }
    }
  }
  return
}
