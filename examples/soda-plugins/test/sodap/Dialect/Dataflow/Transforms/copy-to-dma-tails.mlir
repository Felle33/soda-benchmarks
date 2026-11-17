// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-tails),sodap-dataflow-copy-to-dma{top-func=top dma-word-stride=2},canonicalize)" | FileCheck %s --check-prefix=LEGACY
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-tails),sodap-dataflow-copy-to-dma{top-func=top dma-data-width=64},canonicalize)" | FileCheck %s --check-prefix=PACKED


// Tiles cut at run time (sodap-dataflow-tails) of a matrix with a descriptor
// field: the row loop and the row's length are values, and each request asks
// for that length.

// LEGACY:      dataflow.task attributes {load} {
// LEGACY:        scf.for %[[I:.*]] = %c0 to %[[M:.*]] step %c2 {
// LEGACY-NEXT:     scf.for %[[J:.*]] = %c0 to %[[N:.*]] step %c4 {
// LEGACY:            %[[ROWS:.*]] = arith.minui %{{.*}}, %c2
// LEGACY:            %[[COLS:.*]] = arith.minui %{{.*}}, %c4
// LEGACY:            scf.if
// LEGACY-NEXT:         linalg.fill
// LEGACY:            scf.for %[[R:.*]] = %c0 to %[[ROWS]] step %c1 {
// LEGACY:              %[[WORDS:.*]] = arith.muli %[[COLS]], %c2
// LEGACY-NEXT:         %[[WORDS64:.*]] = arith.index_cast %[[WORDS]] : index to i64
// LEGACY-NEXT:         %[[LEN:.*]] = arith.shli %[[WORDS64]], %c32_i64
// LEGACY:              %[[REQ:.*]] = arith.ori %[[LEN]], %{{.*}}
// LEGACY-NEXT:         dataflow.stream_write %{{.*}}, %[[REQ]]
// LEGACY-NEXT:         scf.for %{{.*}} = %c0 to %[[COLS]] step %c1 {
// LEGACY:      dataflow.task attributes {store} {
// LEGACY-NOT:    linalg.fill
// LEGACY:            %[[SROWS:.*]] = arith.minui %{{.*}}, %c2
// LEGACY:            %[[SCOLS:.*]] = arith.minui %{{.*}}, %c4
// LEGACY:            scf.for %{{.*}} = %c0 to %[[SROWS]] step %c1 {
// LEGACY:              scf.for %{{.*}} = %c0 to %[[SCOLS]] step %c1 {

// Packed DMA: a cut tile ends at a whole size (4 elements, whole 64-bit
// beats) or at the row's end, which owns its padding: the store needs no
// neighbour, so no read-modify-write and no broker task.

// PACKED-NOT:  sodap.dma_broker
// PACKED:      dataflow.task attributes {load} {
// PACKED:            %[[ROWS:.*]] = arith.minui %{{.*}}, %c2
// PACKED:            %[[COLS:.*]] = arith.minui %{{.*}}, %c4
// PACKED:            scf.for %{{.*}} = %c0 to %[[ROWS]] step %c1 {
// PACKED:      dataflow.task attributes {store} {
// PACKED-NOT:    dataflow.stream_read %arg1
// PACKED:            %[[SROWS:.*]] = arith.minui %{{.*}}, %c2
// PACKED:            scf.for %{{.*}} = %c0 to %[[SROWS]] step %c1 {
// PACKED-NOT:    dataflow.stream_read %arg1
// PACKED:      return

func.func @top(%a: memref<?x?xf32> {sodap.descriptor_field = 5 : i64, sodap.descriptor_ld = 8 : i64, sodap.dims = array<i64: 1, 2>},
               %m: index {sodap.descriptor_field = 2 : i64},
               %n: index {sodap.descriptor_field = 3 : i64},
               %c: memref<?x?xf32> {sodap.descriptor_field = 7 : i64, sodap.descriptor_ld = 10 : i64, sodap.dims = array<i64: 1, 2>}) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  dataflow.dispatch {
    %buf = memref.alloc() : memref<2x4xf32>
    dataflow.task attributes {load} {
      scf.for %i = %c0 to %m step %c2 {
        scf.for %j = %c0 to %n step %c4 {
          %t = memref.subview %a[%i, %j] [2, 4] [1, 1] : memref<?x?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
          memref.copy %t, %buf : memref<2x4xf32, strided<[?, 1], offset: ?>> to memref<2x4xf32>
        }
      }
    }
    dataflow.task attributes {store} {
      scf.for %i = %c0 to %m step %c2 {
        scf.for %j = %c0 to %n step %c4 {
          %t = memref.subview %c[%i, %j] [2, 4] [1, 1] : memref<?x?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
          memref.copy %buf, %t : memref<2x4xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
        }
      }
    }
  }
  return
}
