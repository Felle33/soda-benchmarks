// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/Tosa/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/ErrorHandling.h"
using namespace mlir;
namespace {
struct OutlineOptions : PassPipelineOptions<OutlineOptions> {
  Option<std::string> family{*this,"family",llvm::cl::init("kernel")};
  Option<unsigned> lanes{*this,"lanes",llvm::cl::init(64)};
  Option<uint64_t> plmBytes{*this,"plm-bytes",llvm::cl::init(327680)};
};
struct NetworkOptions : PassPipelineOptions<NetworkOptions> {
  Option<std::string> directory{*this,"directory",llvm::cl::init(".")};
  Option<unsigned> dmaWidth{*this,"dma-width",llvm::cl::init(64)};
  Option<unsigned> convLanes{*this,"conv-lanes",llvm::cl::init(64)};
  Option<unsigned> gemmLanes{*this,"gemm-lanes",llvm::cl::init(0)};
  Option<unsigned> addLanes{*this,"add-lanes",llvm::cl::init(4)};
  Option<unsigned> convInstance{*this,"conv-instance",llvm::cl::init(0)};
  Option<unsigned> gemmInstance{*this,"gemm-instance",llvm::cl::init(0)};
  Option<unsigned> addInstance{*this,"add-instance",llvm::cl::init(0)};
  Option<uint64_t> plmBytes{*this,"plm-bytes",llvm::cl::init(327680)};
};
}
void mlir::sodap::registerSODAPTosaPipelines() {
  PassPipelineRegistration<OutlineOptions>("sodap-tosa-esp-outline","ESP kernel/operator hardware extraction",
    [](OpPassManager &pm,const OutlineOptions &o) {
      std::string outline;
      if(o.family=="kernel") outline="sodap-tosa-outline-kernel";
      else if(o.family=="gemm") outline="sodap-tosa-outline-operators{source-semantics=true},sodap-join-multiple-tiles,func.func(canonicalize)";
      else if(o.family=="conv" || o.family=="add") outline=
        std::string(o.family=="conv" ? "sodap-tosa-outline-convolutions" : "sodap-tosa-outline-additions")+
        "{lanes="+std::to_string(o.lanes)+" plm-bytes="+std::to_string(o.plmBytes)+"}";
      else llvm::report_fatal_error("unsupported ESP outline family");
      std::string pipeline="func.func(sodap-tosa-raise-transposes,sodap-tosa-fold-constant-transposes),"+outline;
      if(o.family!="kernel") pipeline+=",sodap-tosa-extract-hardware";
      if(failed(parsePassPipeline(pipeline,pm))) llvm::report_fatal_error("cannot construct ESP outlining");
    });
  PassPipelineRegistration<NetworkOptions>("sodap-tosa-network-pipeline","Unified operator host frontend",
    [](OpPassManager &pm,const NetworkOptions &o) {
      std::string pipeline="func.func(sodap-tosa-sink-transposes,tosa-layerwise-constant-fold,"
        "sodap-tosa-sink-transposes,tosa-layerwise-constant-fold,sodap-tosa-fold-conv-affine,"
        "sodap-tosa-sink-transposes,tosa-layerwise-constant-fold,sodap-tosa-fold-conv-affine),"
        "canonicalize,sodap-tosa-network-host{directory="+o.directory.getValue()+
        " dma-width="+std::to_string(o.dmaWidth)+" conv-instance="+std::to_string(o.convInstance)+
        " add-instance="+std::to_string(o.addInstance)+" gemm-instance="+std::to_string(o.gemmInstance)+
        " conv-lanes="+std::to_string(o.convLanes)+" gemm-lanes="+std::to_string(o.gemmLanes)+
        " add-lanes="+std::to_string(o.addLanes)+" plm-bytes="+std::to_string(o.plmBytes)+"}";
      if(failed(parsePassPipeline(pipeline,pm))) llvm::report_fatal_error("cannot construct network host pipeline");
    });
}
