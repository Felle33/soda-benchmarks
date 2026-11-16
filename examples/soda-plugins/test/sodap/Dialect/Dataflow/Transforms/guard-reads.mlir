// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-guard-reads))" | FileCheck %s
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-guard-reads),sodap-dataflow-copy-to-dma{top-func=top dma-data-width=64},canonicalize)" | FileCheck %s --check-prefix=DMA

// GEMM's C (sodap-join-multiple-tiles) is read only when beta isn't 0: the
// load of each C tile runs under beta une 0 (a NaN reads). The store of D,
// over C, is not a read and stays.

// CHECK-LABEL: func.func @top(
// CHECK-SAME:    %[[C:[^:]*]]: memref<?x?xf32> {{{.*}}sodap.read_unless_zero = 2 : i64}, %[[M:.*]]: index {{{.*}}}, %[[BETA:.*]]: f32 {{{.*}}}, %[[D:.*]]: memref<?x?xf32> {{{.*}}})
// CHECK:         dataflow.task attributes {load} {
// CHECK:           scf.for
// CHECK-NEXT:        %[[T:.*]] = memref.subview %[[C]]
// CHECK-NEXT:        %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:        %[[READ:.*]] = arith.cmpf une, %[[BETA]], %[[ZERO]] : f32
// CHECK-NEXT:        scf.if %[[READ]] {
// CHECK-NEXT:          memref.copy %[[T]], %{{.*}}
// CHECK-NEXT:        }
// CHECK:         dataflow.task attributes {store} {
// CHECK-NOT:       scf.if
// CHECK:           memref.copy

// The DMA lowering reads beta from its descriptor field, the bits of the
// slot's low 32 bits.

// DMA:         dataflow.task attributes {load} {
// DMA:           %[[W:.*]] = memref.load %{{.*}}[%c12] : memref<16xi32>
// DMA:           %[[BETA:.*]] = arith.bitcast %[[W]] : i32 to f32
// DMA:           %[[READ:.*]] = arith.cmpf une, %[[BETA]], %cst{{.*}} : f32
// DMA-NEXT:      scf.if %[[READ]] {
// DMA:             dataflow.stream_write
func.func @top(%c: memref<?x?xf32> {sodap.descriptor_field = 7 : i64, sodap.descriptor_ld = 10 : i64, sodap.read_unless_zero = 2 : i64},
               %m: index {sodap.descriptor_field = 2 : i64},
               %beta: f32 {sodap.descriptor_field = 12 : i64},
               %d: memref<?x?xf32> {sodap.descriptor_field = 7 : i64, sodap.descriptor_ld = 10 : i64}) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  dataflow.dispatch {
    %buf = memref.alloc() : memref<2x4xf32>
    dataflow.task attributes {load} {
      scf.for %i = %c0 to %m step %c2 {
        %t = memref.subview %c[%i, 0] [2, 4] [1, 1] : memref<?x?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
        memref.copy %t, %buf : memref<2x4xf32, strided<[?, 1], offset: ?>> to memref<2x4xf32>
      }
    }
    dataflow.task attributes {store} {
      scf.for %i = %c0 to %m step %c2 {
        %t = memref.subview %d[%i, 0] [2, 4] [1, 1] : memref<?x?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
        memref.copy %buf, %t : memref<2x4xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
      }
    }
  }
  return
}
