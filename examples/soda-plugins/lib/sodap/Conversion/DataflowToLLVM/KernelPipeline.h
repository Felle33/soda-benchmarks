// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef SODAP_KERNEL_PIPELINE_H
#define SODAP_KERNEL_PIPELINE_H
#include <string>
namespace mlir::sodap::kernel {
inline std::string frontend(bool operators=false) {
  return "func.func(tosa-to-arith,tosa-to-tensor,tosa-to-linalg-named,tosa-to-linalg,"
         "sodap-linalg-rank-reduce-contractions),linalg-fold-unit-extent-dims," +
         std::string(operators ? "func.func(sodap-dataflow-bind-dims,sodap-linalg-fold-runtime-broadcasts)," : "") +
         "linalg-fuse-elementwise-ops,func.func(sodap-linalg-constants-to-indices,"
         "sodap-dataflow-bind-dims),canonicalize";
}
inline std::string bufferize() {
  return "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
         "canonicalize,cse,canonicalize,buffer-results-to-out-params";
}
inline std::string schedule(bool overlap, uint64_t reuse, bool serialTasks = false) {
  return "canonicalize,func.func(sodap-dataflow-bind-dims,sodap-dataflow-load-store,"
         "sodap-dataflow-bind-dims),canonicalize,func.func(sodap-dataflow-schedule{reuse-buffer-budget=" +
         std::to_string(reuse) + (serialTasks ? " serial-tasks=true" : overlap ? "" : " serial=true") +
         "}),fold-memref-alias-ops," +
         (serialTasks ? "func.func(sodap-dataflow-ping-pong{single-slot=true})," :
          overlap ? "func.func(sodap-dataflow-ping-pong)," : "") + "canonicalize";
}
inline std::string transport(bool overlap, unsigned width, unsigned stride=2) {
  return "sodap-dataflow-copy-to-dma{dma-word-stride=" + std::to_string(stride) +
         " dma-data-width=" + std::to_string(width) + "}," +
         (overlap ? "func.func(sodap-dataflow-outstanding-load),canonicalize," : "") +
         "func.func(sodap-dataflow-tasks-to-nodes),sodap-dataflow-nodes-to-func,";
}
inline std::string compute(unsigned lanes, uint64_t bytes,
                           const std::string &plan, const std::string &storage) {
  return "sodap-prepare-compute-budget{lanes=" + std::to_string(lanes) + " report=" + plan + "},"
         "sodap-bank-matmul-buffers{rows=1 cols=1 plm-bytes=" + std::to_string(bytes) +
         " report=" + storage + "}";
}
inline std::string llvmBackend() {
  return "sodap-dataflow-infer-top-interface,sodap-emit-bambu-architecture{file=architecture.xml},"
         "sodap-convert-dataflow-to-llvm,func.func(promote-buffers-to-stack{max-alloc-size-in-bytes=1048576}),"
         "func.func(fold-memref-alias-ops),lower-affine,convert-scf-to-cf,convert-math-to-llvm,"
         "expand-strided-metadata,lower-affine,sodap-finalize-banked-memrefs-to-llvm,"
         "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},convert-index-to-llvm,"
         "convert-cf-to-llvm,reconcile-unrealized-casts";
}
}
#endif
