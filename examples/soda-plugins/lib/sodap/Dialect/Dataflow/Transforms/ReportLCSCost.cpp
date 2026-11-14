// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/Dialect/Dataflow/Transforms/Passes.h"
#include "Prefix.h"
#include "sodap/ExecutionEngine/DmaTransport.h"
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"
#include <limits>

namespace mlir::sodap::dataflow {
#define GEN_PASS_DEF_REPORTLCSCOST
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::sodap::dataflow;
namespace {
struct ReportLCSCost : mlir::sodap::dataflow::impl::ReportLCSCostBase<ReportLCSCost> {
  using ReportLCSCostBase::ReportLCSCostBase;
  bool invalid = false;
  int64_t multiply(int64_t a, int64_t b) {
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max()/b)) {
      invalid = true;
      return 0;
    }
    return a*b;
  }
  void add(int64_t &a, int64_t b) {
    if (b < 0 || a > std::numeric_limits<int64_t>::max()-b) invalid = true;
    else a += b;
  }
  /// The index arguments' values in the invocation analysed
  /// (sodap.invocations of a sodap-join-multiple-tiles tile), if any.
  SmallVector<int64_t> inv;
  std::optional<int64_t> valueOf(Value v) {
    if (std::optional<int64_t> c = getConstantIntValue(v))
      return c;
    auto arg = dyn_cast<BlockArgument>(v);
    auto func = arg ? dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp()) : nullptr;
    if (!func || !arg.getType().isIndex())
      return std::nullopt;
    size_t position = llvm::count_if(func.getArguments().take_front(arg.getArgNumber()),
                                     [](BlockArgument a) { return a.getType().isIndex(); });
    if (position >= inv.size())
      return std::nullopt;
    return inv[position];
  }
  int64_t repetitions(Operation *op) {
    int64_t n = 1;
    for (Operation *p=op->getParentOp(); p && !isa<func::FuncOp>(p); p=p->getParentOp()) {
      if (auto loop=dyn_cast<scf::ForOp>(p)) {
        auto lb=valueOf(loop.getLowerBound()), ub=valueOf(loop.getUpperBound()),
             step=valueOf(loop.getStep());
        if (!lb || !ub || !step || *lb<0 || *ub<*lb || *step<=0) { invalid=true; return 0; }
        n=multiply(n,llvm::divideCeil(*ub-*lb,*step));
      } else if (!isa<TaskOp,DispatchOp>(p)) {
        op->emitError("cost analysis requires static scf.for/task nesting");
        invalid=true;
        return 0;
      }
    }
    return n;
  }
  int64_t elements(ShapedType type) {
    int64_t n=1;
    if (!type.hasStaticShape()) { invalid=true; return 0; }
    for (int64_t d:type.getShape()) n=multiply(n,d);
    return n;
  }
  // CopyToDMA transfers only function arguments (or their subviews). Copies
  // between local buffers are not traffic on the external DMA interface.
  bool isExternal(Value value) {
    while (auto view = value.getDefiningOp<memref::SubViewOp>())
      value = view.getSource();
    auto arg = dyn_cast<BlockArgument>(value);
    return arg && isa<func::FuncOp>(arg.getOwner()->getParentOp());
  }
  // Possible column offsets modulo a beat. Enumeration is bounded by the
  // transport lane count, never by the (potentially LARGE) loop trip count.
  SmallVector<int64_t> residues(Value v, int64_t lanes, unsigned depth = 0) {
    if (auto c = valueOf(v); c && *c >= 0) return {*c % lanes};
    if (depth < 16) {
      if (auto arg = dyn_cast<BlockArgument>(v))
        if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
          if (arg == loop.getInductionVar()) {
            auto lo=valueOf(loop.getLowerBound()), hi=valueOf(loop.getUpperBound()),
                 step=valueOf(loop.getStep());
            if (lo && hi && step && *lo>=0 && *hi>=*lo && *step>0) {
              SmallVector<int64_t> result;
              int64_t count=std::min<int64_t>(lanes,llvm::divideCeil(*hi-*lo,*step));
              for (int64_t i=0;i<count;++i) result.push_back((*lo%lanes+i*(*step%lanes))%lanes);
              return result;
            }
          }
      if (isa_and_nonnull<arith::AddIOp,arith::MulIOp>(v.getDefiningOp())) {
        Operation *op=v.getDefiningOp();
        auto a=residues(op->getOperand(0),lanes,depth+1),
             b=residues(op->getOperand(1),lanes,depth+1);
        SmallVector<int64_t> result;
        for (auto x:a) for (auto y:b) {
          int64_t rem=(isa<arith::AddIOp>(op)?x+y:x*y)%lanes;
          if (!llvm::is_contained(result,rem)) result.push_back(rem);
        }
        return result;
      }
    }
    SmallVector<int64_t> result;
    for(int64_t i=0;i<lanes;++i)result.push_back(i);
    return result;
  }
  struct Traffic {
    int64_t useful=0, beatsMin=0, beatsMax=0, requestsMin=0, requestsMax=0,
            rmwMin=0, rmwMax=0;
  } traffic;
  llvm::json::Object packedTraffic(memref::CopyOp copy, int64_t rows,
                                  int64_t count, int64_t elementsPerCopy) {
    bool load=isExternal(copy.getSource());
    Value external=load?copy.getSource():copy.getTarget();
    auto type=cast<MemRefType>(external.getType());
    int64_t length=type.getRank()?type.getShape().back():1;
    int64_t lanes=dmaDataWidth/32;
    SmallVector<int64_t> first{0};
    bool ownsPadding=true;
    if(auto view=external.getDefiningOp<memref::SubViewOp>();
       view && view.getSourceType().getRank()) {
      auto offset=view.getMixedOffsets().back();
      if(auto c=getConstantIntValue(offset)) {
        if (*c < 0) invalid = true;
        first={*c < 0 ? 0 : *c%lanes};
      }
      else first=residues(cast<Value>(offset),lanes);
      auto col=getConstantIntValue(offset);
      ownsPadding=col && view.getSourceType().hasStaticShape() &&
          length<=view.getSourceType().getShape().back() &&
          *col==view.getSourceType().getShape().back()-length;
    }
    int64_t useful=multiply(multiply(count,elementsPerCopy),4);
    int64_t lo=INT64_MAX,hi=0,rmwLo=INT64_MAX,rmwHi=0;
    for(int64_t f:first) {
      int64_t covered=length;
      add(covered,f);
      int64_t beats=length?llvm::divideCeil(covered,lanes):0;
      int64_t head=length&&f!=0,tail=length&&covered%lanes!=0;
      // The broker may preserve even owned padding. Report that conservative
      // upper bound explicitly, rather than claiming a request-free tail.
      int64_t minRmw=load?0:head+(tail&&!ownsPadding);
      int64_t maxRmw=load?0:head+tail;
      if(beats==1) { minRmw=std::min(int64_t(1),minRmw); maxRmw=std::min(int64_t(1),maxRmw); }
      lo=std::min(lo,beats+minRmw);hi=std::max(hi,beats+maxRmw);
      rmwLo=std::min(rmwLo,minRmw);rmwHi=std::max(rmwHi,maxRmw);
    }
    if(first.empty())lo=rmwLo=0;
    lo=multiply(rows,lo);hi=multiply(rows,hi);
    rmwLo=multiply(rows,rmwLo);rmwHi=multiply(rows,rmwHi);
    int64_t baseRequests=length?rows:0;
    int64_t requestsLo=baseRequests,requestsHi=baseRequests;
    add(requestsLo,rmwLo);add(requestsHi,rmwHi);
    add(traffic.useful,useful);add(traffic.beatsMin,lo);add(traffic.beatsMax,hi);
    add(traffic.requestsMin,requestsLo);add(traffic.requestsMax,requestsHi);
    add(traffic.rmwMin,rmwLo);add(traffic.rmwMax,rmwHi);
    return llvm::json::Object{{"direction",load?"read":"write"},{"useful_bytes",useful},
        {"physical_bytes_min",multiply(lo,dmaDataWidth/8)},
        {"physical_bytes_max",multiply(hi,dmaDataWidth/8)},
        {"beats_min",lo},{"beats_max",hi},
        {"requests_min",requestsLo},{"requests_max",requestsHi},
        {"rmw_reads_min",rmwLo},{"rmw_reads_max",rmwHi}};
  }
  void runOnOperation() override {
    if (dmaDataWidth && !::sodap::DmaTransport{dmaDataWidth}.valid()) {
      getOperation().emitError("dma-data-width must be 64, 128, 256 or 512");
      return signalPassFailure();
    }
    if (requestCycles<0 || wordCycles<0 || (wordStride!=1 && wordStride!=2) || budget<0) {
      getOperation().emitError("expected nonnegative costs/budget and DMA stride 1 or 2");
      return signalPassFailure();
    }
    int64_t bytes=0,requests=0,words=0,iterations=0;
    llvm::json::Array copies,compute;
    // A tile with invocations: the work summed over them, the storage once.
    SmallVector<SmallVector<int64_t>> invocations{{}};
    if (auto list = getOperation()->getAttrOfType<ArrayAttr>("sodap.invocations")) {
      invocations.clear();
      for (Attribute each : list)
        invocations.push_back(llvm::to_vector(cast<DenseI64ArrayAttr>(each).asArrayRef()));
    }
    for (auto [i, each] : llvm::enumerate(invocations)) {
    inv = each;
    bool first = i == 0;
    getOperation().walk([&](Operation *op) {
      if (!op->getParentOfType<DispatchOp>()) return;
      if (isa<memref::AllocOp,memref::AllocaOp>(op)) {
        if (!first) return;
        auto t=cast<MemRefType>(op->getResult(0).getType());
        if (!t.getElementType().isIntOrFloat()) { invalid=true; return; }
        add(bytes,multiply(elements(t),llvm::divideCeil(t.getElementTypeBitWidth(),8)));
      } else if (auto copy=dyn_cast<memref::CopyOp>(op)) {
        if (!isExternal(copy.getSource()) && !isExternal(copy.getTarget()))
          return;
        auto t=copy.getSource().getType();
        if (!t.getElementType().isF32()) { invalid=true; return; }
        int64_t n=repetitions(op),rows=1;
        for (int64_t d:t.getShape().drop_back(t.getRank()?1:0)) rows=multiply(rows,d);
        int64_t r=multiply(n,rows),w=multiply(multiply(n,elements(t)),wordStride);
        add(requests,r); add(words,w);
        llvm::json::Object record{{"invocations",n},{"requests",r},{"words",w}};
        if(dmaDataWidth) record["packed_transport"]=packedTraffic(copy,r,n,elements(t));
        copies.push_back(std::move(record));
      } else if (auto linalg=dyn_cast<linalg::LinalgOp>(op)) {
        int64_t n=repetitions(op);
        for (int64_t d:linalg.getStaticLoopRanges()) n=multiply(n,d);
        add(iterations,n);
        compute.push_back(llvm::json::Object{{"op",op->getName().getStringRef()},
                                            {"iterations",n},{"reduction",linalg.getNumReductionLoops()>0}});
      }
    });
    }
    int64_t cycles=multiply(requests,requestCycles);
    add(cycles,multiply(words,wordCycles));
    if (invalid) {
      getOperation().emitError("unsupported dynamic/control-dependent schedule or overflow in static cost analysis");
      return signalPassFailure();
    }
    llvm::json::Object result{{"dma_cycles",cycles},{"dma_requests",requests},{"dma_words",words},
        {"plm_bytes",bytes},{"compute_iterations",iterations},{"dma_word_stride",int64_t(wordStride)},
        {"copies",std::move(copies)},{"compute",std::move(compute)},
        {"scope","Static scheduled work and logical allocation; not measured latency or FPGA resource use"}};
    if(dmaDataWidth) {
      result["legacy_dma_estimate_retained"] = true;
      result["packed_transport"] = llvm::json::Object{
          {"abi",2},{"width",int64_t(dmaDataWidth)},
          {"useful_bytes",traffic.useful},
          {"physical_bytes_min",multiply(traffic.beatsMin,dmaDataWidth/8)},
          {"physical_bytes_max",multiply(traffic.beatsMax,dmaDataWidth/8)},
          {"beats_min",traffic.beatsMin},{"beats_max",traffic.beatsMax},
          {"requests_min",traffic.requestsMin},{"requests_max",traffic.requestsMax},
          {"rmw_reads_min",traffic.rmwMin},{"rmw_reads_max",traffic.rmwMax},
          {"scope","Bounds on copy traffic including boundary RMW; excludes descriptors. Exact when min=max. Does not change tile policy or predict overlap/PLM service latency."}};
      if(invalid) {
        getOperation().emitError("overflow in packed traffic accounting");
        return signalPassFailure();
      }
    }
    std::error_code ec;
    // Nested in a tile module: its own report and budget, in its directory.
    llvm::raw_fd_ostream os(getTileFilePath(getOperation(),file),ec);
    if (ec) { getOperation().emitError("cannot write cost report: ")<<ec.message(); return signalPassFailure(); }
    os<<llvm::formatv("{0:2}\n",llvm::json::Value(std::move(result)));
    if (budget && bytes>budget) {
      getOperation().emitError("scheduled local storage ")<<bytes<<" exceeds budget "<<budget;
      signalPassFailure();
    }
  }
};
}
