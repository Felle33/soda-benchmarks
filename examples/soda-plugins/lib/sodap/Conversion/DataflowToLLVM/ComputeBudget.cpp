//===- ComputeBudget.cpp - Shared arithmetic budgets for contractions -------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/Conversion/DataflowToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/JSON.h"

namespace mlir::sodap {
#define GEN_PASS_DEF_PREPARECOMPUTEBUDGET
#include "sodap/Conversion/DataflowToLLVM/Passes.h.inc"
}
using namespace mlir;
namespace {

// Normalize contraction indexing, not benchmark names. Preserve the sequential
// reduction and scalar body; unsupported arithmetic must fail, never disappear.
static bool isMultiplyAdd(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      !op.hasPureBufferSemantics()) return false;
  Block &body = op->getRegion(0).front();
  if (body.getNumArguments() != 3 || body.getOperations().size() != 3) return false;
  auto mul = dyn_cast<arith::MulFOp>(body.front());
  auto add = dyn_cast<arith::AddFOp>(*std::next(body.begin()));
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!mul || !add || !yield || !mul.getType().isF32() ||
      mul.getFastmath() != arith::FastMathFlags::none || add.getFastmath() != arith::FastMathFlags::none) return false;
  auto samePair = [](Value a, Value b, Value x, Value y) {
    return (a == x && b == y) || (a == y && b == x);
  };
  return samePair(mul.getLhs(), mul.getRhs(), body.getArgument(0), body.getArgument(1)) &&
         samePair(add.getLhs(), add.getRhs(), body.getArgument(2), mul.getResult()) &&
         yield.getValues().size() == 1 && yield.getValues()[0] == add.getResult();
}

static LogicalResult normalize(linalg::LinalgOp op) {
  if (isa<linalg::MatmulOp>(op.getOperation())) return success();
  if (!isMultiplyAdd(op)) return op.emitError("compute budget requires a supported FP32 multiply/add contraction");
  auto maps = op.getIndexingMapsArray();
  auto types = op.getIteratorTypesArray();
  SmallVector<SmallVector<unsigned>> positions;
  for (AffineMap map : maps) {
    if (!map.isProjectedPermutation()) return op.emitError("contraction maps must be projected permutations");
    SmallVector<unsigned> dims;
    for (AffineExpr e : map.getResults()) dims.push_back(cast<AffineDimExpr>(e).getPosition());
    positions.push_back(dims);
  }
  auto contains = [&](unsigned operand, unsigned d) { return llvm::is_contained(positions[operand], d); };
  int m = -1, n = -1, k = -1;
  SmallVector<int64_t> bounds(types.size(), -1);
  for (unsigned operand = 0; operand != 3; ++operand) {
    auto mt = dyn_cast<MemRefType>(op->getOperand(operand).getType());
    if (!mt || !mt.hasStaticShape() || !mt.getElementType().isF32())
      return op.emitError("contraction budget needs static local FP32 buffers");
    for (auto [axis,d] : llvm::enumerate(positions[operand])) {
      if (bounds[d] != -1 && bounds[d] != mt.getDimSize(axis)) return op.emitError("inconsistent contraction extents");
      bounds[d] = mt.getDimSize(axis);
    }
  }
  for (unsigned d = 0; d != types.size(); ++d) {
    if (types[d] == utils::IteratorType::reduction) {
      if (k != -1 || !contains(0,d) || !contains(1,d) || contains(2,d))
        return op.emitError("expected one shared reduction dimension");
      k = d;
    } else {
      if (!contains(2,d)) return op.emitError("parallel contraction dimension absent from output");
      if (contains(0,d) && !contains(1,d) && (m < 0 || bounds[d] > bounds[m])) m = d;
      if (contains(1,d) && !contains(0,d) && (n < 0 || bounds[d] > bounds[n])) n = d;
    }
  }
  if (k == -1 || (m == -1 && n == -1)) return op.emitError("unsupported contraction dimensions");
  SmallVector<unsigned> outer;
  for (unsigned d = 0; d != types.size(); ++d)
    if (int(d) != m && int(d) != n && int(d) != k) outer.push_back(d);
  OpBuilder b(op); auto loc = op.getLoc();
  SmallVector<Value> indices(types.size());
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  auto one = b.create<arith::ConstantIndexOp>(loc, 1);
  bool failedMap = false;
  std::function<void(OpBuilder &, unsigned)> build = [&](OpBuilder &builder, unsigned depth) {
    if (depth < outer.size()) {
      unsigned d = outer[depth];
      Value upper = builder.create<arith::ConstantIndexOp>(loc, bounds[d]);
      builder.create<scf::ForOp>(loc, zero, upper, one, ValueRange{},
        [&](OpBuilder &nested, Location l, Value iv, ValueRange) {
          indices[d] = iv; build(nested, depth + 1); nested.create<scf::YieldOp>(l);
        });
      return;
    }
    SmallVector<Value> operands;
    for (unsigned operand = 0; operand != 3; ++operand) {
      Value source = op->getOperand(operand);
      SmallVector<int> wanted = operand == 0 ? SmallVector<int>{m,k} :
                                operand == 1 ? SmallVector<int>{k,n} : SmallVector<int>{m,n};
      SmallVector<int> actual;
      SmallVector<OpFoldResult> offsets, sizes, strides;
      SmallVector<int64_t> shape;
      for (unsigned d : positions[operand]) {
        bool keep = int(d) == wanted[0] || int(d) == wanted[1];
        offsets.push_back(keep ? OpFoldResult(builder.getIndexAttr(0)) : OpFoldResult(indices[d]));
        sizes.push_back(builder.getIndexAttr(keep ? bounds[d] : 1));
        strides.push_back(builder.getIndexAttr(1));
        if (keep) { actual.push_back(d); shape.push_back(bounds[d]); }
      }
      SmallVector<int> expected;
      for (int d : wanted) if (d >= 0) expected.push_back(d);
      if (actual != expected) { failedMap = true; return; }
      if (shape.size() != positions[operand].size()) {
        auto result = memref::SubViewOp::inferRankReducedResultType(shape,
            cast<MemRefType>(source.getType()), offsets, sizes, strides);
        source = builder.create<memref::SubViewOp>(loc, cast<MemRefType>(result), source, offsets, sizes, strides);
      }
      if (shape.size() == 1) {
        SmallVector<int64_t> expanded = wanted[0] < 0 ? SmallVector<int64_t>{1,shape[0]} : SmallVector<int64_t>{shape[0],1};
        source = builder.create<memref::ExpandShapeOp>(loc, expanded, source,
            ArrayRef<ReassociationIndices>{ReassociationIndices{0,1}});
      }
      operands.push_back(source);
    }
    if (!failedMap) builder.create<linalg::MatmulOp>(loc, ValueRange{operands[0],operands[1]}, ValueRange{operands[2]});
  };
  build(b, 0);
  if (failedMap) return op.emitError("contraction operand order requires unsupported transposition");
  op.erase(); return success();
}

// A DMA vector can be represented as 2x1xK at its producer and 2xK at
// its consumer. Bambu's shaped array interfaces need one physical rank.
static SmallVector<ReassociationIndices> unitGroups(MemRefType type) {
  SmallVector<ReassociationIndices> groups(1); bool seen = false;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.getDimSize(i) != 1) {
      if (seen) groups.push_back({});
      seen = true;
    }
    groups.back().push_back(i);
  }
  return groups;
}
static void compactUnitStorage(ModuleOp module) {
  SmallVector<Value> buffers;
  for (auto func : module.getOps<func::FuncOp>()) if (!func.isExternal())
    for (auto arg : func.getArguments()) buffers.push_back(arg);
  module.walk([&](Operation *op) {
    if (isa<memref::AllocOp,memref::AllocaOp>(op)) buffers.push_back(op->getResult(0));
  });
  for (Value buffer : buffers) {
    auto type = dyn_cast<MemRefType>(buffer.getType());
    if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity()) continue;
    SmallVector<int64_t> shape;
    for (int64_t d : type.getShape()) if (d != 1) shape.push_back(d);
    if (shape.size() < 2 || shape.size() == size_t(type.getRank())) continue;
    SmallVector<OpOperand *> uses;
    for (auto &use : buffer.getUses()) uses.push_back(&use);
    auto physical = MemRefType::get(shape,type.getElementType()); buffer.setType(physical);
    OpBuilder b(module.getContext());
    if (auto arg = dyn_cast<BlockArgument>(buffer)) b.setInsertionPointToStart(arg.getOwner());
    else b.setInsertionPointAfter(buffer.getDefiningOp());
    Value logical = b.create<memref::ExpandShapeOp>(buffer.getLoc(),type,buffer,unitGroups(type));
    for (auto *use : uses) if (!isa<func::CallOp,memref::DeallocOp>(use->getOwner())) use->set(logical);
  }
  for (auto func : module.getOps<func::FuncOp>()) if (!func.isExternal()) {
    SmallVector<Type> types; for (auto arg : func.getArguments()) types.push_back(arg.getType());
    func.setType(FunctionType::get(module.getContext(),types,func.getResultTypes()));
  }
  module.walk([&](func::CallOp call) {
    auto f = module.lookupSymbol<func::FuncOp>(call.getCallee()); if (!f || f.isExternal()) return;
    OpBuilder b(call);
    for (auto [i,v] : llvm::enumerate(call.getOperands())) {
      auto src = dyn_cast<MemRefType>(v.getType()); auto dst = dyn_cast<MemRefType>(f.getArgument(i).getType());
      if (src && dst && src != dst && src.getRank() > dst.getRank())
        call->setOperand(i,b.create<memref::CollapseShapeOp>(call.getLoc(),dst,v,unitGroups(src)));
    }
  });
}

struct PrepareComputeBudget : sodap::impl::PrepareComputeBudgetBase<PrepareComputeBudget> {
  using Base::Base;
  void runOnOperation() override {
    if (!lanes || lanes > 256) {
      getOperation().emitError("compute budget must be 1..256 lanes"); return signalPassFailure();
    }
    compactUnitStorage(getOperation());
    SmallVector<linalg::LinalgOp> contractions;
    getOperation().walk([&](linalg::LinalgOp op) { if (op.getNumReductionLoops()) contractions.push_back(op); });
    for (auto op : contractions) if (failed(normalize(op))) return signalPassFailure();
    SmallVector<linalg::MatmulOp> matmuls;
    getOperation().walk([&](linalg::MatmulOp op) { matmuls.push_back(op); });
    llvm::MapVector<Operation *, unsigned> groups;
    for (auto op : matmuls) {
      Operation *function = op->getParentOfType<func::FuncOp>();
      groups.try_emplace(function, groups.size());
    }
    if (matmuls.empty() || groups.size() > lanes) {
      getOperation().emitError("compute budget requires at least one lane per contraction"); return signalPassFailure();
    }
    // Equal initial power-of-two shares keep bank factors practical. Distribute
    // the remainder in source order. This fixed policy is separate from lowering.
    unsigned share = 1;
    while (2 * share <= unsigned(lanes) / groups.size()) share *= 2;
    SmallVector<unsigned> allocated(groups.size(), share);
    unsigned remaining = lanes - share * groups.size();
    for (unsigned i = 0; remaining; i = (i + 1) % allocated.size()) {
      unsigned extra = std::min(allocated[i], remaining); allocated[i] += extra; remaining -= extra;
    }
    llvm::json::Array stages, groupReport;
    for (auto [op,i] : groups) groupReport.push_back(llvm::json::Object{
        {"function",cast<func::FuncOp>(op).getSymName().str()}, {"lanes",allocated[i]}});
    OpBuilder b(&getContext());
    for (auto [index,mm] : llvm::enumerate(matmuls)) {
      unsigned count = allocated[groups[mm->getParentOfType<func::FuncOp>()]], r = 1, c = count;
      auto type = cast<MemRefType>(mm.getOutputs()[0].getType());
      if (type.getDimSize(1) == 1) { r = count; c = 1; }
      else if (type.getDimSize(0) != 1) {
        for (unsigned i = 1; i * i <= count; ++i) if (!(count % i)) { r = i; c = count/i; }
      }
      // Tiny validation tensors may be narrower than the hardware budget.
      // Never create a bank which has no logical element and hence no consumer.
      // LARGE tiles retain the full budget; record any small-shape reduction.
      while (r > unsigned(type.getDimSize(0)) && r > 1) r /= 2;
      while (c > unsigned(type.getDimSize(1)) && c > 1) c /= 2;
      mm->setAttr("sodap.register_rows", b.getI32IntegerAttr(r));
      mm->setAttr("sodap.register_columns", b.getI32IntegerAttr(c));
      auto f = mm->getParentOfType<func::FuncOp>();
      stages.push_back(llvm::json::Object{{"stage",int64_t(index)},
        {"function",f ? f.getSymName().str() : ""}, {"lanes",count},
        {"active_lanes",r*c}, {"rows",r}, {"columns",c}});
    }
    if (!report.empty()) {
      std::error_code error; llvm::raw_fd_ostream stream(report, error, llvm::sys::fs::OF_Text);
      if (error) { getOperation().emitError(error.message()); return signalPassFailure(); }
      stream << llvm::json::Value(llvm::json::Object{{"core_lane_budget",unsigned(lanes)},
        {"policy","balanced_power_of_two_concurrent_function_shares"}, {"groups",std::move(groupReport)}, {"stages",std::move(stages)},
        {"scope","Contraction lane allocation; actual HLS sharing and epilogue units require audit"}}) << "\n";
    }
  }
};
} // namespace
