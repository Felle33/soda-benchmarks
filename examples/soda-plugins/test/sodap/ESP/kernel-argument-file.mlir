// RUN: mlir-opt %s --allow-unregistered-dialect --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-order-kernel-arguments{argument-order-file=%t.args})" -o /dev/null
// RUN: cat %t.args | FileCheck %s
// CHECK: 0 1 2
func.func @forward(%a: memref<f32>, %b: memref<2xf32>, %c: memref<4xf32>) {
  "soda.launch_func"(%b, %c, %a) {kernel = @forward_kernel::@forward_kernel, operandSegmentSizes = array<i32: 0, 3>} : (memref<2xf32>, memref<4xf32>, memref<f32>) -> ()
  return
}
"soda.module"() ({
  "soda.func"() ({
  ^bb0(%y: memref<2xf32>, %z: memref<4xf32>, %x: memref<f32>):
    "test.use"(%y, %z, %x) : (memref<2xf32>, memref<4xf32>, memref<f32>) -> ()
    "soda.return"() : () -> ()
  }) {function_type = (memref<2xf32>, memref<4xf32>, memref<f32>) -> (), sym_name = "forward_kernel"} : () -> ()
  "soda.module_end"() : () -> ()
}) {sym_name = "forward_kernel"} : () -> ()

