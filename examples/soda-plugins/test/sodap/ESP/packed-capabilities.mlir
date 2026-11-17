// RUN: not mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-generate-reusable-gemm{dma-data-width=32})" 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-generate-reusable-gemm{dma-data-width=128 tile-m=1 tile-n=1 tile-k=1},sodap-emit-bambu-architecture)" 2>&1 | FileCheck %s --check-prefix=BACKEND
// INVALID: dma-data-width must be 64, 128, 256 or 512
// BACKEND: Bambu
// BACKEND-SAME: 64
module {}
