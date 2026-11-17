// RUN: mkdir -p %t.dir
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-network-host{directory=%t.dir})" | FileCheck %s
// RUN: python3 -c "import json; p=json.load(open('%t.dir/network.json')); assert len(p['engines'])==2; assert p['engines'][0]['instances']==[0,1]; assert [i['operator'] for i in p['invocations']]==['gemm','add','add']; assert p['invocations'][1]['relu']; assert p['parameter_elements']==0"

// One implementation may serve multiple explicitly bound physical instances;
// a live matmul result survives the fused branch until the residual addition.
// CHECK: func.func @forward_init
// CHECK: call @esp_op_open_devices
// CHECK: func.func @forward_run
// CHECK: call @esp_op_gemm_source
// CHECK: call @esp_op_add
// CHECK-SAME: sodap.instance = 0
// CHECK: call @esp_op_add
// CHECK-SAME: sodap.instance = 1
// CHECK: func.func @forward_close
func.func @forward(%a:tensor<1x2x3xf32>,%b:tensor<1x3x2xf32>)->tensor<1x2x2xf32> {
  %m=tosa.matmul %a,%b : (tensor<1x2x3xf32>,tensor<1x3x2xf32>)->tensor<1x2x2xf32>
  %x=tosa.add %m,%m : (tensor<1x2x2xf32>,tensor<1x2x2xf32>)->tensor<1x2x2xf32>
  %r=tosa.clamp %x {min_fp=0.0:f32,max_fp=3.40282347E+38:f32,min_int=0:i64,max_int=2147483647:i64} : (tensor<1x2x2xf32>)->tensor<1x2x2xf32>
  %y=tosa.add %r,%m {sodap.instance=1:i64} : (tensor<1x2x2xf32>,tensor<1x2x2xf32>)->tensor<1x2x2xf32>
  return %y:tensor<1x2x2xf32>
}
