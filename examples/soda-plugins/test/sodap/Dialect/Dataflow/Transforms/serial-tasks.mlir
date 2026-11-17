// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-schedule{serial-tasks=true},fold-memref-alias-ops,sodap-dataflow-ping-pong{single-slot=true},canonicalize))" | FileCheck %s
// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-ping-pong{single-slot=true}))" 2>&1 | FileCheck %s --check-prefix=UNSAFE
// UNSAFE: single-slot storage requires the serial-tasks handoff

// Each worker acknowledges at the next work boundary, after publishing its tile.
// CHECK-LABEL: func.func @chain
// CHECK: memref.alloc() : memref<1x4xf32>
// CHECK: dataflow.task {
// CHECK: scf.for
// CHECK: dataflow.stream_write %[[LR:[0-9]+]], {{.*}} {sodap.dma_ordered}
// CHECK: dataflow.stream_read %[[LG:[0-9]+]]
// CHECK: memref.copy
// CHECK: dataflow.stream_write
// CHECK: dataflow.stream_write %[[LR]],
// CHECK: memref.alloc() : memref<1x4xf32>
// CHECK: dataflow.task {
// CHECK: scf.for
// CHECK: dataflow.stream_write %[[CR:[0-9]+]], {{.*}} {sodap.dma_ordered}
// CHECK: dataflow.stream_read %[[CG:[0-9]+]]
// CHECK: linalg.fill
// CHECK: dataflow.stream_write %[[CR]],
// CHECK: dataflow.stream_read %[[CG]]
// CHECK: linalg.add
// CHECK: linalg.copy
// CHECK: dataflow.stream_write %[[CR]],
// The store drains startup acknowledgements, then waits for completion of
// each grant before granting another worker or touching the output.
// CHECK: dataflow.task attributes {sodap.serial_coordinator}
// CHECK: dataflow.stream_read %[[LR]]
// CHECK: dataflow.stream_read %[[CR]]
// CHECK: scf.for
// CHECK: dataflow.stream_write %[[CG]],
// CHECK-NEXT: {{.*}} = dataflow.stream_peek %[[CR]] {sodap.dma_ordered}
// CHECK-NEXT: {{.*}} = dataflow.stream_read %[[CR]]
// CHECK-NEXT: dataflow.stream_write %[[LG]],
// CHECK-NEXT: {{.*}} = dataflow.stream_peek %[[LR]] {sodap.dma_ordered}
// CHECK-NEXT: {{.*}} = dataflow.stream_read %[[LR]]
// CHECK-NEXT: dataflow.stream_write %[[CG]],
// CHECK-NEXT: {{.*}} = dataflow.stream_peek %[[CR]] {sodap.dma_ordered}
// CHECK-NEXT: {{.*}} = dataflow.stream_read %[[CR]]
// CHECK: memref.copy

func.func @chain(%a: memref<8xf32>, %out: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  dataflow.dispatch {
    %in = memref.alloc() : memref<8xf32>
    dataflow.task {
      memref.copy %a, %in : memref<8xf32> to memref<8xf32>
      dataflow.yield
    }
    %acc = memref.alloc() : memref<8xf32>
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        %x = memref.subview %in[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        %y = memref.subview %acc[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        linalg.fill ins(%zero : f32) outs(%y : memref<4xf32, strided<[1], offset: ?>>)
        linalg.add ins(%x, %y : memref<4xf32, strided<[1], offset: ?>>, memref<4xf32, strided<[1], offset: ?>>) outs(%y : memref<4xf32, strided<[1], offset: ?>>)
      }
      dataflow.yield
    }
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        %y = memref.subview %acc[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        %o = memref.subview %out[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        memref.copy %y, %o : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
