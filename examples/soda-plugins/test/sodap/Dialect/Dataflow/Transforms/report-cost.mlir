// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-report-cost{file=%t.packed.json dma-data-width=64})" -o /dev/null
// RUN: python3 -c "import json; t=json.load(open('%t.packed.json'))['packed_transport']; assert t['useful_bytes']==96 and t['physical_bytes_min']==t['physical_bytes_max']==96 and t['beats_min']==12 and t['requests_max']==3 and t['rmw_reads_max']==0"
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-report-cost{file=%t.json dma-word-stride=2})" -o /dev/null
// RUN: FileCheck %s < %t.json
// CHECK: "dma_cycles": 240
// CHECK: "dma_requests": 3
// CHECK: "dma_word_stride": 2
// CHECK: "dma_words": 48
// CHECK: "plm_bytes": 64
func.func @repeated(%a: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  dataflow.dispatch {
    %tmp = memref.alloc() : memref<8xf32>
    %local = memref.alloc() : memref<8xf32>
    dataflow.task {
      scf.for %i = %c0 to %c3 step %c1 {
        memref.copy %a, %tmp : memref<8xf32> to memref<8xf32>
        memref.copy %tmp, %local : memref<8xf32> to memref<8xf32>
      }
      dataflow.yield
    }
  }
  return
}
