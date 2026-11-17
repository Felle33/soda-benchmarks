// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/Conversion/DataflowToLLVM/Passes.h"
#include "KernelPipeline.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <set>
#include <tuple>

namespace mlir::sodap {
#define GEN_PASS_DEF_SELECTKERNELTILES
#define GEN_PASS_DEF_SNAPSHOTKERNELIR
#include "sodap/Conversion/DataflowToLLVM/Passes.h.inc"
}
using namespace mlir;
namespace {
LogicalResult writeText(StringRef path, StringRef text) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) { llvm::errs() << path << ": " << ec.message() << "\n"; return failure(); }
  out << text;
  return success();
}
std::string print(Operation *op) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  op->print(stream);
  stream << '\n';
  return text;
}
struct SnapshotKernelIR : sodap::impl::SnapshotKernelIRBase<SnapshotKernelIR> {
  using Base::Base;
  void runOnOperation() override {
    if (!file.empty() && failed(writeText(file, print(getOperation())))) signalPassFailure();
  }
};
struct SelectKernelTiles : sodap::impl::SelectKernelTilesBase<SelectKernelTiles> {
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
    OpPassManager pm;
    std::string pipeline="builtin.module("+sodap::kernel::frontend()+","+
      sodap::kernel::bufferize()+","+sodap::kernel::schedule(true,reuseBytes)+","+
      sodap::kernel::transport(true,dmaWidth,wordStride)+
      sodap::kernel::compute(lanes,plmBytes,"compute_plan.json","compute_storage.json")+")";
    if (succeeded(parsePassPipeline(pipeline,pm))) pm.getDependentDialects(registry);
  }
  using Score = std::tuple<int64_t,int64_t,int64_t>;
  struct Candidate {
    llvm::json::Object report;
    std::optional<Score> score;
    OwningOpRef<ModuleOp> tasks;
    std::string tiles;
  };
  LogicalResult run(ModuleOp module, const std::string &pipeline) {
    PassManager pm(&getContext());
    if (failed(parsePassPipeline(pipeline, pm))) return failure();
    return pm.run(module);
  }
  std::string costPipeline(StringRef path) {
    return "sodap-dataflow-report-cost{dma-data-width=" + std::to_string(dmaWidth) +
      " file=" + path.str() + " request-cycles=" + std::to_string(requestCycles) +
      " word-cycles=" + std::to_string(wordCycles) + " dma-word-stride=" +
      std::to_string(wordStride) + " plm-bytes=" + std::to_string(plmBytes) + "}";
  }
  Candidate evaluate(uint64_t cap, bool historical, bool overlap, bool doubleCopy,
                     StringRef path) {
    Candidate result;
    result.report["proposal"] = historical ? llvm::json::Object{{"max_tile",10}} :
                                             llvm::json::Object{{"plm",cap}};
    result.report["directory"] = path.str();
    result.report["proposal_set"] = doubleCopy ? "double-copy" : "single-copy";
    result.report["scored_schedule"] = overlap ? "overlap" : "nonoverlap";
    result.report["status"] = "REJECTED";
    result.report["returncode"] = 1;
    std::string diagnostics;
    llvm::raw_string_ostream diagnosticStream(diagnostics);
    ScopedDiagnosticHandler handler(&getContext(), [&](Diagnostic &d) {
      d.print(diagnosticStream); diagnosticStream << '\n'; return success();
    });
    auto evaluateImpl = [&]() -> LogicalResult {
      if (auto ec=llvm::sys::fs::create_directories(path))
        return getOperation().emitError(ec.message());
      OwningOpRef<ModuleOp> payload(cast<ModuleOp>(getOperation()->clone()));
      OwningOpRef<ModuleOp> library;
      if (failed(transform::detail::parseTransformModuleFromFile(
            &getContext(), transformFile, library))) return failure();
      OpBuilder b(&getContext());
      bool configured=false;
      library->walk([&](Operation *op) {
        if (op->getName().getStringRef() != "transform.sodap.tile_and_fuse") return;
        auto sequence=op->getParentOp();
        auto name=sequence->getAttrOfType<StringAttr>("sym_name");
        if (!name || name.getValue() != "tile_tasks") return;
        for (StringRef attr : {"tile_sizes","max_tile_size","plm_bytes"}) op->removeAttr(attr);
        op->setAttr(historical ? "max_tile_size" : "plm_bytes", b.getI64IntegerAttr(historical ? 10 : cap));
        op->setAttr("compute_lanes",b.getI64IntegerAttr(lanes));
        op->setAttr("double_buffer",b.getBoolAttr(doubleCopy));
        op->setAttr("dma_request_cycles",b.getI64IntegerAttr(requestCycles));
        op->setAttr("dma_word_cycles",b.getI64IntegerAttr(wordCycles));
        configured=true;
      });
      if (!configured) return getOperation().emitError("missing tile_tasks transform sequence");
      if (failed(run(*payload,"builtin.module("+sodap::kernel::frontend()+")"))) return failure();
      for (ModuleOp tile : payload->getOps<ModuleOp>()) {
        tile->setAttr("sodap.capture_tile_selection",b.getUnitAttr());
        auto entry=transform::detail::findTransformEntryPoint(tile,*library,"tasks");
        if (!entry || failed(transform::applyTransformNamedSequence(tile,entry,*library,
                                               transform::TransformOptions()))) return failure();
        auto sizes=tile->getAttrOfType<ArrayAttr>("sodap.selected_tile_sizes");
        if (!sizes) return tile.emitError("tiling did not produce selection metadata");
        llvm::raw_string_ostream out(result.tiles);
        out << "[";
        llvm::interleaveComma(sizes,out,[&](Attribute row) {
          out << "[";
          llvm::interleaveComma(cast<ArrayAttr>(row),out,[&](Attribute n) {
            out << cast<IntegerAttr>(n).getInt();
          });
          out << "]";
        });
        out << "]";
        tile->removeAttr("sodap.capture_tile_selection");
        tile->removeAttr("sodap.selected_tile_sizes");
      }
      result.tasks=OwningOpRef<ModuleOp>(cast<ModuleOp>((*payload)->clone()));
      if (failed(run(*payload,"builtin.module("+sodap::kernel::bufferize()+")")) ||
          failed(run(*payload,"builtin.module("+sodap::kernel::schedule(overlap,reuseBytes)+")"))) return failure();
      std::string costPath=path.str()+"/schedule_cost.json";
      if (failed(run(*payload,costPipeline(costPath)))) return failure();
      auto file=llvm::MemoryBuffer::getFile(costPath);
      if (!file) return getOperation().emitError("missing candidate schedule cost");
      auto json=llvm::json::parse((*file)->getBuffer());
      if (!json) { llvm::consumeError(json.takeError()); return failure(); }
      auto cost=json->getAsObject();
      if (!cost) return failure();
      Score score{cost->getInteger("dma_cycles").value(),
                  cost->getInteger("compute_iterations").value(),
                  cost->getInteger("plm_bytes").value()};
      result.report["cost"]=std::move(*json);
      auto tilesJSON=llvm::json::parse(result.tiles);
      if (!tilesJSON) { llvm::consumeError(tilesJSON.takeError()); return failure(); }
      result.report["tiles"]=std::move(*tilesJSON);
      // Physical banked storage is an admission check, not a replacement for
      // the historical lexicographic objective's logical storage term.
      if (failed(run(*payload,"builtin.module("+
          sodap::kernel::transport(overlap,dmaWidth,wordStride)+
          sodap::kernel::compute(lanes,plmBytes,path.str()+"/compute_plan.json",
                                 path.str()+"/compute_storage.json")+")"))) return failure();
      auto storageFile=llvm::MemoryBuffer::getFile(path.str()+"/compute_storage.json");
      if (!storageFile) return failure();
      auto storage=llvm::json::parse((*storageFile)->getBuffer());
      if (!storage) { llvm::consumeError(storage.takeError()); return failure(); }
      result.report["storage"]=std::move(*storage);
      result.score=score;
      result.report["returncode"]=0;
      result.report["status"]="VALID";
      return success();
    };
    if (failed(evaluateImpl())) {
      result.tasks={};
      result.report["diagnostic"]=diagnostics;
    }
    return result;
  }
  void runOnOperation() override {
    if ((schedule!="overlap" && schedule!="nonoverlap") || !lanes || lanes>256 ||
        !plmBytes || transformFile.empty()) {
      getOperation().emitError("expected schedule overlap/nonoverlap, 1..256 lanes, positive budget and transform-file");
      return signalPassFailure();
    }
    llvm::SmallString<256> absolute(reportDir.getValue());
    if (auto ec=llvm::sys::fs::make_absolute(absolute)) {
      getOperation().emitError(ec.message()); return signalPassFailure();
    }
    std::string root=std::string(absolute);
    std::set<uint64_t> caps;
    for (unsigned divisor : {32,16,8,4,1}) caps.insert(std::max<uint64_t>(64,plmBytes/divisor));
    SmallVector<std::pair<uint64_t,bool>> proposals;
    for (uint64_t cap:caps) proposals.emplace_back(cap,false);
    proposals.emplace_back(10,true);
    auto search=[&](bool overlap, bool doubleCopy, StringRef suffix) {
      std::vector<Candidate> candidates;
      for (auto [cap,historical] : proposals)
        candidates.push_back(evaluate(cap,historical,overlap,doubleCopy,root+"/.tile-search/"+
                                      suffix.str()+"/"+std::to_string(candidates.size())));
      return candidates;
    };
    auto winner=[](const std::vector<Candidate> &candidates) {
      int best=-1;
      for (unsigned i=0;i<candidates.size();++i)
        if (candidates[i].score && (best<0 || *candidates[i].score<*candidates[best].score)) best=i;
      return best;
    };
    // Stable ties retain the historical proposal. Both sets are admitted and
    // scored on the schedule actually generated, never on overlap for serial.
    auto selected=search(schedule=="overlap",true,"double-copy");
    if (schedule=="nonoverlap") {
      auto single=search(false,false,"single-copy");
      for (auto &candidate:single) selected.push_back(std::move(candidate));
    }
    int best=winner(selected);
    std::string winnerSet=best<0 ? "none" :
      selected[best].report.getString("proposal_set")->str();
    std::vector<Candidate> old;
    if (schedule=="nonoverlap") old=search(true,true,"old-overlap");
    int oldBest=schedule=="overlap" ? best : winner(old);
    llvm::json::Array reports,oldReports;
    for (auto &c:selected) reports.push_back(std::move(c.report));
    for (auto &c:old) oldReports.push_back(std::move(c.report));
    llvm::json::Object report{{"implementation","sodap-select-kernel-tiles"},
      {"plm_budget",uint64_t(plmBytes)}, {"reuse_budget",uint64_t(reuseBytes)},
      {"scope","scheduled DMA cost, compute iterations, logical storage; physical banked admission"},
      {"schedule",schedule.getValue()}, {"selected_set",winnerSet}, {"selected",best}, {"candidates",std::move(reports)},
      {"old_overlap_selected",oldBest}, {"old_overlap_candidates",std::move(oldReports)}};
    std::string reportText;
    llvm::raw_string_ostream out(reportText); out << llvm::formatv("{0:2}\n",llvm::json::Value(std::move(report)));
    if (failed(writeText(root+"/tile_search.json",reportText))) return signalPassFailure();
    if (best<0) {
      getOperation().emitError("all whole-kernel tile candidates rejected; see tile_search.json");
      return signalPassFailure();
    }
    if (failed(writeText(root+"/tile_sizes.txt",selected[best].tiles+"\n")) ||
        failed(writeText(root+"/selected_tile_sizes.txt",selected[best].tiles+"\n"))) return signalPassFailure();
    getOperation().getBodyRegion().takeBody(selected[best].tasks->getBodyRegion());
  }
};
}
