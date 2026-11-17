//===- Pipelines.cpp - DataflowToLLVM pipeline ----------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/Conversion/DataflowToLLVM/Passes.h"
#include "KernelPipeline.h"
#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {
struct DataflowToLLVMPipelineOptions
    : public PassPipelineOptions<DataflowToLLVMPipelineOptions> {
  // The only option with no default: see Passes.td.
  Option<std::string> topFunc{
      *this, "top-func",
      llvm::cl::desc("Top function of the design (required)")};
  Option<std::string> archFile{
      *this, "arch-file", llvm::cl::init("architecture.xml"),
      llvm::cl::desc("Where to write the Bambu --architecture-xml file")};
  Option<bool> writeToTerminal{
      *this, "write-to-terminal", llvm::cl::init(false),
      llvm::cl::desc("Print the --architecture-xml file to stdout instead of "
                     "writing arch-file")};
  Option<bool> axisTopStreams{
      *this, "axis-top-streams", llvm::cl::init(false),
      llvm::cl::desc("Make the streams of the top that the design reads or "
                     "writes AXI4-Stream ports (axis) instead of FIFOs")};
  Option<std::string> clangxx{
      *this, "clangxx", llvm::cl::init(SODAP_BAMBU_CLANGXX),
      llvm::cl::desc("C++ compiler used to build the specialization unit")};
  Option<std::string> includePandaPath{
      *this, "include-panda-path", llvm::cl::init(SODAP_BAMBU_INCLUDE),
      llvm::cl::desc("Directory holding ac_channel.h")};
};
struct ReusableOperatorBackendOptions
    : public PassPipelineOptions<ReusableOperatorBackendOptions> {
  Option<std::string> family{*this,"family",llvm::cl::init("gemm"),llvm::cl::desc("Operator compute family: gemm, conv or add")};
  Option<unsigned> lanes{*this,"lanes",llvm::cl::init(64),llvm::cl::desc("FP32 core lane budget")};
  Option<unsigned> dmaWidth{*this,"dma-width",llvm::cl::init(64),llvm::cl::desc("Packed DMA width")};
  Option<uint64_t> plmBytes{*this,"plm-bytes",llvm::cl::init(327680),llvm::cl::desc("Local array storage budget")};
  Option<std::string> computeReport{*this,"compute-report",llvm::cl::init("compute_plan.json")};
  Option<std::string> storageReport{*this,"storage-report",llvm::cl::init("compute_storage.json")};
};
struct OperatorPipelineOptions : public PassPipelineOptions<OperatorPipelineOptions> {
  Option<std::string> family{*this,"family",llvm::cl::init("gemm")};
  Option<unsigned> lanes{*this,"lanes",llvm::cl::init(64)};
  Option<unsigned> dmaWidth{*this,"dma-width",llvm::cl::init(64)};
  Option<uint64_t> plmBytes{*this,"plm-bytes",llvm::cl::init(327680)};
  Option<uint64_t> reuseBytes{*this,"reuse-bytes",llvm::cl::init(262144)};
  Option<std::string> transformFile{*this,"transform-file",llvm::cl::init("")};
  Option<std::string> directory{*this,"directory",llvm::cl::init(".")};
};
struct KernelHostOptions : public PassPipelineOptions<KernelHostOptions> {
  Option<bool> profile{*this,"profile",llvm::cl::init(true)};
  Option<unsigned> dmaWidth{*this,"dma-width",llvm::cl::init(64)};
  Option<std::string> directory{*this,"directory",llvm::cl::init(".")};
};
struct WholeKernelBackendOptions : public PassPipelineOptions<WholeKernelBackendOptions> {
  Option<bool> singleCopyProposals{*this,"single-copy-proposals",llvm::cl::init(false)};
  Option<std::string> schedule{*this,"schedule",llvm::cl::init("overlap")};
  Option<unsigned> lanes{*this,"lanes",llvm::cl::init(64)};
  Option<unsigned> dmaWidth{*this,"dma-width",llvm::cl::init(64)};
  Option<unsigned> wordStride{*this,"word-stride",llvm::cl::init(2)};
  Option<uint64_t> plmBytes{*this,"plm-bytes",llvm::cl::init(327680)};
  Option<uint64_t> reuseBytes{*this,"reuse-bytes",llvm::cl::init(262144)};
  Option<unsigned> requestCycles{*this,"request-cycles",llvm::cl::init(64)};
  Option<unsigned> wordCycles{*this,"word-cycles",llvm::cl::init(1)};
  Option<std::string> transformFile{*this,"transform-file",llvm::cl::init("")};
  Option<std::string> reportDir{*this,"report-dir",llvm::cl::init(".")};
};
} // namespace

/// Takes the flat call-graph form -- func.func per node, dataflow streams
/// across the calls -- down to the LLVM dialect for Bambu. Kept separate from
/// the front half so iterating on the backend does not mean rerunning it.
void mlir::sodap::registerDataflowToLLVMPipeline() {
  PassPipelineRegistration<DataflowToLLVMPipelineOptions>(
      "sodap-dataflow-to-llvm-pipeline",
      "Lower the dataflow call-graph form to the LLVM dialect for Bambu, "
      "emitting the --architecture-xml file on the way. Anchor this on "
      "builtin.module.",
      [](OpPassManager &pm, const DataflowToLLVMPipelineOptions &opts) {
        // Ahead of the XML: which streams are ports is in their type.
        if (opts.axisTopStreams)
          pm.addPass(dataflow::createInferTopInterface({opts.topFunc}));
        // Must run before the lowering: it needs the stream types to tell a
        // FIFO from an array, and the next pass erases them.
        pm.addPass(createEmitBambuArchitecture({opts.archFile, opts.topFunc,
                                                opts.includePandaPath,
                                                opts.writeToTerminal}));
        pm.addPass(
            createConvertDataflowToLLVM({opts.clangxx, opts.includePandaPath}));

        // Node-local buffers become stack arrays instead of malloc calls.
        // A func.func pass: --pass-pipeline does not nest it on its own.
        pm.addNestedPass<func::FuncOp>(
            bufferization::createPromoteBuffersToStackPass(1024 * 1024));

        pm.addPass(createLowerAffinePass());
        pm.addPass(createConvertSCFToCFPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(createConvertMathToLLVMPass());
        pm.addPass(memref::createExpandStridedMetadataPass());
        // Strided metadata expansion may create affine expressions.
        pm.addPass(createLowerAffinePass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        // Bare pointers: Bambu's array protocol wants a plain pointer per
        // array, not an exploded memref descriptor.
        ConvertFuncToLLVMPassOptions funcToLLVMOptions;
        funcToLLVMOptions.useBarePtrCallConv = true;
        pm.addPass(createConvertFuncToLLVMPass(funcToLLVMOptions));
        pm.addPass(createConvertIndexToLLVMPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());
        // Only now can the canonicalizer fold extractvalue(insertvalue) and
        // drop the memref descriptors func-to-llvm rebuilt at every entry.
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });
  PassPipelineRegistration<ReusableOperatorBackendOptions>(
      "sodap-reusable-operator-backend",
      "Lower a scheduled model-derived operator tile through guarded/tail DMA, "
      "shared compute-budget/banking passes and LLVM conversion",
      [](OpPassManager &pm, const ReusableOperatorBackendOptions &o) {
        // Keep the registered sequence in the compiler. The shell driver only
        // passes target budgets and orchestrates frontends/tool invocations.
        std::string compute;
        if(o.family=="gemm" || o.family=="conv") compute=
          kernel::compute(o.lanes,o.plmBytes,o.computeReport,o.storageReport)+
          ",sodap-register-block-matmul{rows=1 cols=1},canonicalize,cse,";
        else if(o.family=="add") compute=
          "sodap-bank-matmul-buffers{pointwise=true rows=1 cols=" + std::to_string(o.lanes) +
          " plm-bytes=" + std::to_string(o.plmBytes) + " report=" + o.storageReport.getValue() + "},"
          "sodap-register-block-pointwise{lanes=" + std::to_string(o.lanes) + "},canonicalize,cse,";
        else llvm::report_fatal_error("unsupported reusable operator compute family");
        std::string pipeline =
          "func.func(sodap-dataflow-guard-reads,sodap-dataflow-tails),"
          "sodap-dataflow-copy-to-dma{dma-word-stride=2 dma-data-width=" + std::to_string(o.dmaWidth) + "},"
          "func.func(sodap-dataflow-tasks-to-nodes),sodap-dataflow-nodes-to-func,"
          + compute +
          "func.func(convert-linalg-to-loops),sodap-legalize-fp-minmax," + kernel::llvmBackend();
        if (failed(parsePassPipeline(pipeline, pm)))
          llvm::report_fatal_error("cannot construct reusable operator backend pipeline");
      });
  PassPipelineRegistration<WholeKernelBackendOptions>(
      "sodap-whole-kernel-backend",
      "Select and lower whole-kernel tiles with shared compute/storage heuristics",
      [](OpPassManager &pm, const WholeKernelBackendOptions &o) {
        std::string dir=o.reportDir;
        std::string common="dma-data-width="+std::to_string(o.dmaWidth)+
          " request-cycles="+std::to_string(o.requestCycles)+" word-cycles="+
          std::to_string(o.wordCycles)+" dma-word-stride="+std::to_string(o.wordStride)+
          " plm-bytes="+std::to_string(o.plmBytes);
        auto snapshot=[&](StringRef name) {
          return "sodap-snapshot-kernel-ir{file="+dir+"/"+name.str()+".mlir},";
        };
        std::string pipeline=
          "sodap-select-kernel-tiles{schedule="+o.schedule.getValue()+
          " single-copy-proposals="+(o.singleCopyProposals ? "true" : "false")+" lanes="+
          std::to_string(o.lanes)+" plm-bytes="+std::to_string(o.plmBytes)+
          " reuse-bytes="+std::to_string(o.reuseBytes)+" dma-width="+std::to_string(o.dmaWidth)+
          " word-stride="+std::to_string(o.wordStride)+" request-cycles="+
          std::to_string(o.requestCycles)+" word-cycles="+std::to_string(o.wordCycles)+
          " transform-file="+o.transformFile.getValue()+" report-dir="+dir+"},"+
          snapshot("tasks")+"builtin.module("+kernel::bufferize()+"),"+snapshot("bufferized")+
          "builtin.module("+kernel::schedule(o.schedule=="overlap",o.reuseBytes)+"),"+
          snapshot("dataflow")+"sodap-dataflow-report-cost{"+common+" file="+dir+"/schedule_cost.json},"+
          "builtin.module("+kernel::transport(o.schedule=="overlap",o.dmaWidth,o.wordStride)+
          kernel::compute(o.lanes,o.plmBytes,dir+"/compute_plan.json",dir+"/compute_storage.json")+
          ",sodap-register-block-matmul{rows=1 cols=1},canonicalize,cse,"
          "func.func(convert-linalg-to-loops),"+kernel::llvmBackend()+"),sodap-tiles-to-llvm,sodap-emit-hls-configuration{lanes="+
          std::to_string(o.lanes)+" directory="+dir+"}";
        if (failed(parsePassPipeline(pipeline,pm)))
          llvm::report_fatal_error("cannot construct whole-kernel backend pipeline");
      });

  PassPipelineRegistration<OperatorPipelineOptions>("sodap-operator-pipeline",
      "Model-derived operator frontend, transform scheduling and registered backend",
      [](OpPassManager &pm,const OperatorPipelineOptions &o) {
        std::string dir=o.directory;
        std::string select;
        if(o.family=="gemm" || o.family=="conv")
          select="sodap-select-reusable-hardware{lanes="+std::to_string(o.lanes)+
            " plm-bytes="+std::to_string(o.plmBytes)+" report="+dir+"/hardware_selection.json},"
            "sodap-snapshot-kernel-ir{file="+dir+"/hardware_selected.mlir},"
            "sodap-snapshot-kernel-ir{file="+dir+"/outlined.mlir},";
        std::string pipeline=select+"builtin.module("+kernel::frontend(true)+"),"
          "sodap-apply-lcs-transform{family="+o.family.getValue()+" lanes="+std::to_string(o.lanes)+
          " transform-file="+o.transformFile.getValue()+" report-dir="+dir+"},"
          "sodap-snapshot-kernel-ir{file="+dir+"/tasks.mlir},"
          "builtin.module("+kernel::bufferize()+"),sodap-snapshot-kernel-ir{file="+dir+"/bufferized.mlir},"
          "builtin.module("+kernel::schedule(true,o.reuseBytes)+"),sodap-snapshot-kernel-ir{file="+dir+"/dataflow.mlir},"
          "builtin.module(sodap-dataflow-report-cost{dma-data-width="+std::to_string(o.dmaWidth)+
          " file=schedule_cost.json request-cycles=64 word-cycles=1 dma-word-stride=2 plm-bytes="+std::to_string(o.plmBytes)+"}),"
          "builtin.module(sodap-reusable-operator-backend{family="+o.family.getValue()+" lanes="+std::to_string(o.lanes)+
          " plm-bytes="+std::to_string(o.plmBytes)+" dma-width="+std::to_string(o.dmaWidth)+
          " compute-report="+dir+"/compute_plan.json storage-report="+dir+"/compute_storage.json}),"
          "sodap-tiles-to-llvm,sodap-emit-hls-configuration{lanes="+std::to_string(o.lanes)+" directory="+dir+"}";
        if(failed(parsePassPipeline(pipeline,pm))) llvm::report_fatal_error("cannot construct operator pipeline");
      });
  PassPipelineRegistration<>("sodap-kernel-cpu-frontend","Whole-kernel CPU reference frontend",
      [](OpPassManager &pm) {
        if(failed(parsePassPipeline(kernel::frontend()+","+kernel::bufferize(),pm)))
          llvm::report_fatal_error("cannot construct CPU frontend");
      });
  PassPipelineRegistration<KernelHostOptions>("sodap-whole-kernel-host","Whole-kernel ESP host",
      [](OpPassManager &pm,const KernelHostOptions &o) {
        std::string pipeline="func.func(tosa-to-arith,tosa-to-tensor,tosa-to-linalg-named,tosa-to-linalg),"+
          kernel::bufferize()+",sodap-launches-to-esp{host=forward_kernel profile="+
          std::string(o.profile ? "true" : "false")+" dma-word-stride=2 dma-data-width="+std::to_string(o.dmaWidth)+
          " invocations-file="+o.directory.getValue()+"/invocations.json argument-order-file="+o.directory.getValue()+"/argument_order.txt}";
        if(failed(parsePassPipeline(pipeline,pm))) llvm::report_fatal_error("cannot construct kernel host pipeline");
      });

}
