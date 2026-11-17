// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-copy-to-dma{top-func=top dma-data-width=64},canonicalize)" | FileCheck %s

// A non-GEMM schema has 32 slots and a rank-one packed tensor view. Its
// dimension comes from the descriptor, not a compile-time tensor extent.
// CHECK-LABEL: func.func @top(
// CHECK: dataflow.task attributes {load}
// CHECK: memref<32xi32>
// CHECK: dataflow.stream_write
// CHECK: dataflow.stream_read
// CHECK: memref.load {{.*}}[%c8]
func.func @top(%a:memref<?xf32> {sodap.descriptor_field=3:i64,sodap.dma_view="packed",sodap.dims=array<i64:1>},
               %n:index {sodap.descriptor_field=8:i64})
    attributes {sodap.descriptor_fields=32:i64,sodap.descriptor_magic=1094992945:i64,
                sodap.descriptor_version=3:i64,sodap.descriptor_width_field=31:i64} {
  %c0=arith.constant 0:index
  %c4=arith.constant 4:index
  dataflow.dispatch {
    %buf=memref.alloc():memref<4xf32>
    dataflow.task attributes {load} {
      scf.for %i=%c0 to %n step %c4 {
        %tile=memref.subview %a[%i] [4] [1]:memref<?xf32> to memref<4xf32,strided<[1],offset:?>>
        memref.copy %tile,%buf:memref<4xf32,strided<[1],offset:?>> to memref<4xf32>
      }
    }
  }
  return
}
