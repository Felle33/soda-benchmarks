//===- Passes.td - DataflowToLLVM passes -------------------*- tablegen -*-===//

#ifndef SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_TD
#define SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertDataflowToLLVM :
    Pass<"sodap-convert-dataflow-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower dataflow streams to Bambu ac_channel calls";
  let description = [{
    Rewrites !dataflow.stream<T, D> to !llvm.ptr and the stream ops to the
    ac_channel ABI Bambu recognizes. The layout of ac_channel<T> and the
    mangled ctor/dtor/read/write symbols are not spelled out here: the pass
    generates a C++ specialization unit, compiles it with Clang, and reads
    them back out of the generated IR. That IR is dropped in the working
    directory as ac_channel_specializations.ll; llvm-link it into the
    translated kernel to get a single module.

    A stream_write marked sodap.dma_ordered is lowered to an explicit
    nb_write retry loop. Successful acceptance controls exit from the loop,
    preventing a later DMA request from being co-issued with a stalled tile
    publication on a different channel. Unmarked writes are unchanged.

    Both options default to the Bambu install -DSODAP_BAMBU_ROOT named at
    configure time, through the generated sodap/Config.h.
  }];
  let options = [
    Option<"clangxx", "clangxx", "std::string",
           /*default=*/"SODAP_BAMBU_CLANGXX",
           "C++ compiler used to build the specialization unit">,
    Option<"includePandaPath", "include-panda-path", "std::string",
           /*default=*/"SODAP_BAMBU_INCLUDE",
           "Directory holding ac_channel.h and friends">
  ];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::LLVM::LLVMDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def EmitBambuArchitecture :
    Pass<"sodap-emit-bambu-architecture", "::mlir::ModuleOp"> {
  let summary = "Emit the Bambu --architecture-xml description of the design";
  let description = [{
    Must run while the stream types are still present. Emits one <function>
    per non-external func.func: the top as dataflow_top, the rest as
    dataflow_module. Each parameter gets a positional bundle P<index>: fifo
    for a stream, array for a memref, default for a scalar. A stream typed
    `axis` (see sodap-dataflow-infer-top-interface) is an axis bundle on the
    top and stays a fifo on the node it is passed to.
  }];
  let options = [
    Option<"xmlFileName", "file", "std::string",
           /*default=*/"\"architecture.xml\"", "Output file name">,
    Option<"topFunc", "top-func", "std::string", /*default=*/"",
           "Top function of the design (default: the module's name)">,
    Option<"includePandaPath", "include-panda-path", "std::string",
           /*default=*/"SODAP_BAMBU_INCLUDE",
           "Directory holding ac_channel.h, named in the includes= of every "
           "stream parameter">,
    Option<"writeToTerminal", "write-to-terminal", "bool", /*default=*/"false",
           "Print the XML to stdout instead of writing `file`">
  ];
  let dependentDialects = ["::mlir::sodap::dataflow::DataflowDialect"];
}

def PrepareComputeBudget : Pass<"sodap-prepare-compute-budget", "::mlir::ModuleOp"> {
  let summary = "Normalize FP32 contractions and allocate one shared compute budget";
  let options = [
    Option<"lanes", "lanes", "unsigned", "64", "Whole-accelerator contraction lane budget">,
    Option<"report", "report", "std::string", "\"\"", "Optional allocation report JSON">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect", "::mlir::linalg::LinalgDialect"];
}

def BankMatmulBuffers : Pass<"sodap-bank-matmul-buffers", "::mlir::ModuleOp"> {
  let summary = "Mark matmul local memories for cyclic banking before register blocking";
  let options = [
    Option<"rows", "rows", "unsigned", "8", "A row banks">,
    Option<"cols", "cols", "unsigned", "8", "B and C column banks">,
    Option<"plmBytes", "plm-bytes", "uint64_t", "0", "Physical local array bytes after padding, zero disables check">,
    Option<"report", "report", "std::string", "\"\"", "Optional physical storage report JSON">
  ];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect",
    "::mlir::arith::ArithDialect", "::mlir::memref::MemRefDialect"];
}

def FinalizeBankedMemRefsToLLVM :
    Pass<"sodap-finalize-banked-memrefs-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lower static identity memrefs with array-shaped allocas and full GEPs";
  let description = [{
    Opt-in backend for banked memories. Run fold-memref-alias-ops first so
    subview accesses address the original array with all dimensions. This
    preserves the shape required by Bambu CSROA. Other memrefs use the standard
    MLIR conversion. Does not change the bare-pointer function ABI or DMA layout.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def RegisterBlockMatmul : Pass<"sodap-register-block-matmul", "::mlir::ModuleOp"> {
  let summary = "Lower matmul to a fully exposed register block with guarded tails";
  let options = [
    Option<"rows", "rows", "unsigned", "8", "Register block rows">,
    Option<"cols", "cols", "unsigned", "8", "Register block columns">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect"];
}

#endif // SODAP_CONVERSION_DATAFLOWTOLLVM_PASSES_TD
