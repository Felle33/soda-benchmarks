// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODATosaPlugin%shlibext --pass-pipeline="builtin.module(sodap-tosa-outline-convolutions)" -o %t.mlir
// RUN: FileCheck %s --check-prefix=OUTLINE < %t.mlir
// RUN: mlir-opt %t.mlir --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-select-reusable-hardware{lanes=64 plm-bytes=327680 report=%t.json})" | FileCheck %s --check-prefix=SELECT
// RUN: python3 -c "import json; p=json.load(open('%t.json')); assert [p['tile_m'],p['tile_n'],p['tile_k']]==[128,64,64]; assert [p['register_rows'],p['register_columns']]==[8,8]"
// OUTLINE: sodap.invocations = [array<i64: 2, 9, 11, 3, 7, 7, 7, 5, 6, 2, 2, 3, 3, 1, 1, 30, 147>]
// OUTLINE: tensor<?x?x?xf32>
// OUTLINE-SAME: sodap.dma_view = "conv_window_flat"
// OUTLINE: sodap.descriptor_field = 23
// OUTLINE: sodap.descriptor_field = 24
// OUTLINE: sodap.descriptor_version = 4
// OUTLINE: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// OUTLINE: sodap.bound_axes = array<i64: 0, 1, 2, 3>
// SELECT: sodap.hardware_tile_sizes = {{\[\[}}1, 128, 64, 64{{\]\]}}
func.func @forward(%a:tensor<2x9x11x3xf32>,%w:tensor<7x7x7x3xf32>,%b:tensor<7xf32>)->tensor<2x5x6x7xf32> {
 %c=tosa.conv2d %a,%w,%b {pad=array<i64:3,3,3,3>,stride=array<i64:2,2>,dilation=array<i64:1,1>} : (tensor<2x9x11x3xf32>,tensor<7x7x7x3xf32>,tensor<7xf32>)->tensor<2x5x6x7xf32>
 return %c:tensor<2x5x6x7xf32>
}
