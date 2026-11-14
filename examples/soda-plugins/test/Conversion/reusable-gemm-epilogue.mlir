// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-reusable-gemm-to-esp{dma-data-width=64 report=%t.json})" | FileCheck %s
// RUN: python3 -c "import json; p=json.load(open('%t.json')); v=p['invocations']; assert len(v)==2 and v[0]['alpha_argument']==3 and v[1]['beta_argument']==4; assert v[1]['a']==v[0]['c']; assert len(p['initializations'])==4 and len(p['outputs'])==1"
#id = affine_map<(i,j)->(i,j)>
#scalar = affine_map<(i,j)->()>
module {
 func.func @forward_kernel(%a:memref<3x5xf32>,%b:memref<5x7xf32>,%d:memref<7x2xf32>,%alpha:memref<f32>,%beta:memref<f32>,%prior:memref<3x2xf32>,%out:memref<3x2xf32>) {
  %zero=arith.constant 0.0 : f32
  %t=memref.alloc():memref<3x7xf32>
  %scaled=memref.alloc():memref<3x7xf32>
  %u=memref.alloc():memref<3x2xf32>
  %old=memref.alloc():memref<3x2xf32>
  %result=memref.alloc():memref<3x2xf32>
  linalg.fill ins(%zero:f32) outs(%t:memref<3x7xf32>)
  linalg.matmul ins(%a,%b:memref<3x5xf32>,memref<5x7xf32>) outs(%t:memref<3x7xf32>)
  linalg.generic {indexing_maps=[#id,#scalar,#id],iterator_types=["parallel","parallel"]} ins(%t,%alpha:memref<3x7xf32>,memref<f32>) outs(%scaled:memref<3x7xf32>) {
   ^bb0(%x:f32,%y:f32,%z:f32):
    %r=arith.mulf %x,%y:f32
    linalg.yield %r:f32
  }
  linalg.fill ins(%zero:f32) outs(%u:memref<3x2xf32>)
  linalg.matmul ins(%scaled,%d:memref<3x7xf32>,memref<7x2xf32>) outs(%u:memref<3x2xf32>)
  linalg.generic {indexing_maps=[#scalar,#id,#id],iterator_types=["parallel","parallel"]} ins(%beta,%prior:memref<f32>,memref<3x2xf32>) outs(%old:memref<3x2xf32>) {
   ^bb0(%x:f32,%y:f32,%z:f32):
    %r=arith.mulf %x,%y:f32
    linalg.yield %r:f32
  }
  linalg.generic {indexing_maps=[#id,#id,#id],iterator_types=["parallel","parallel"]} ins(%u,%old:memref<3x2xf32>,memref<3x2xf32>) outs(%result:memref<3x2xf32>) {
   ^bb0(%x:f32,%y:f32,%z:f32):
    %r=arith.addf %x,%y:f32
    linalg.yield %r:f32
  }
  memref.copy %result,%out:memref<3x2xf32> to memref<3x2xf32>
  return
 }
}
// CHECK: sodap.gemm_invocations = 2
// CHECK-COUNT-4: call @esp_gemm_copy_in
// CHECK: memref.load %arg3[]
// CHECK: call @esp_gemm_invoke
// CHECK: memref.load %arg4[]
// CHECK: call @esp_gemm_invoke
// CHECK: call @esp_gemm_copy_out
// CHECK: call @esp_gemm_close
