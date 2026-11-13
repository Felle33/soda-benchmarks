//===- BankedMemrefs.cpp - Shape-preserving Bambu local memory lowering -----===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "sodap/Conversion/DataflowToLLVM/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"

namespace mlir::sodap {
#define GEN_PASS_DEF_BANKMATMULBUFFERS
#define GEN_PASS_DEF_FINALIZEBANKEDMEMREFSTOLLVM
#define GEN_PASS_DEF_REGISTERBLOCKMATMUL
#include "sodap/Conversion/DataflowToLLVM/Passes.h.inc"
}
using namespace mlir;

namespace {
// Same register-blocked i/j/k interchange as the compute-tiling study, but
// expose every lane explicitly. Cleanup loops with dynamic lane indices would
// multiplex bank pointers and hide parallelism from the HLS interface analysis.
struct RegisterBlockMatmul : sodap::impl::RegisterBlockMatmulBase<RegisterBlockMatmul> {
  using Base::Base;
  void runOnOperation() override {
    if (!rows || !cols || uint64_t(rows) * cols > 256) {
      getOperation().emitError("register block must have 1..256 lanes");
      return signalPassFailure();
    }
    SmallVector<linalg::MatmulOp> matmuls;
    getOperation().walk([&](linalg::MatmulOp mm) { matmuls.push_back(mm); });
    if (matmuls.size() > 1) {
      getOperation().emitError("resource-matched register blocking currently supports one matmul per accelerator; allocate a whole-accelerator budget before enabling multiple compute nodes");
      return signalPassFailure();
    }
    for (auto mm : matmuls) {
      auto at = dyn_cast<MemRefType>(mm.getInputs()[0].getType());
      auto bt = dyn_cast<MemRefType>(mm.getInputs()[1].getType());
      auto ct = dyn_cast<MemRefType>(mm.getOutputs()[0].getType());
      if (!at || !bt || !ct || !at.hasStaticShape() || !bt.hasStaticShape() ||
          !ct.hasStaticShape() || !ct.getElementType().isF32()) {
        mm.emitError("register blocking requires static local FP32 matmul buffers");
        return signalPassFailure();
      }
      OpBuilder b(mm); auto loc = mm.getLoc();
      Value a = mm.getInputs()[0], bb = mm.getInputs()[1], c = mm.getOutputs()[0];
      auto index = [&](int64_t n) -> Value { return b.create<arith::ConstantIndexOp>(loc, n); };
      Value zero = index(0), one = index(1), m = index(ct.getDimSize(0));
      Value n = index(ct.getDimSize(1)), k = index(at.getDimSize(1));
      Value mr = index(rows), nr = index(cols);
      Value fzero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0));
      auto maskedLoad = [&](OpBuilder &x, Value memory, ValueRange indices, Value valid) -> Value {
        if (!valid) return x.create<memref::LoadOp>(loc, memory, indices);
        auto conditional = x.create<scf::IfOp>(loc, valid,
          [&](OpBuilder &t, Location l) {
            Value v = t.create<memref::LoadOp>(l, memory, indices);
            t.create<scf::YieldOp>(l, v);
          }, [&](OpBuilder &e, Location l) { e.create<scf::YieldOp>(l, fzero); });
        return conditional.getResult(0);
      };
      b.create<scf::ForOp>(loc, zero, m, mr, ValueRange{},
        [&](OpBuilder &ib, Location l, Value i, ValueRange) {
        ib.create<scf::ForOp>(l, zero, n, nr, ValueRange{},
          [&](OpBuilder &jb, Location l, Value j, ValueRange) {
          SmallVector<Value> ri, cj, rv, cv, valid, initial;
          for (unsigned r = 0; r < rows; ++r) {
            Value offset = jb.create<arith::ConstantIndexOp>(l, r);
            ri.push_back(jb.create<arith::AddIOp>(l, i, offset));
            rv.push_back(ct.getDimSize(0) % rows ?
                jb.create<arith::CmpIOp>(l, arith::CmpIPredicate::ult, ri.back(), m).getResult() : Value());
          }
          for (unsigned s = 0; s < cols; ++s) {
            Value offset = jb.create<arith::ConstantIndexOp>(l, s);
            cj.push_back(jb.create<arith::AddIOp>(l, j, offset));
            cv.push_back(ct.getDimSize(1) % cols ?
                jb.create<arith::CmpIOp>(l, arith::CmpIPredicate::ult, cj.back(), n).getResult() : Value());
          }
          for (unsigned r = 0; r < rows; ++r) for (unsigned s = 0; s < cols; ++s) {
            Value cond = rv[r] ? rv[r] : cv[s];
            if (rv[r] && cv[s]) cond = jb.create<arith::AndIOp>(l, rv[r], cv[s]);
            valid.push_back(cond);
            initial.push_back(maskedLoad(jb, c, ValueRange{ri[r], cj[s]}, cond));
          }
          auto reduction = jb.create<scf::ForOp>(l, zero, k, one, initial,
            [&](OpBuilder &kb, Location l, Value kk, ValueRange accs) {
            SmallVector<Value> av, bv, next;
            for (unsigned r = 0; r < rows; ++r)
              av.push_back(maskedLoad(kb, a, ValueRange{ri[r], kk}, rv[r]));
            for (unsigned s = 0; s < cols; ++s)
              bv.push_back(maskedLoad(kb, bb, ValueRange{kk, cj[s]}, cv[s]));
            for (unsigned r = 0; r < rows; ++r) for (unsigned s = 0; s < cols; ++s) {
              Value product = kb.create<arith::MulFOp>(l, av[r], bv[s]);
              next.push_back(kb.create<arith::AddFOp>(l, accs[r * cols + s], product));
            }
            kb.create<scf::YieldOp>(l, next);
          });
          for (unsigned r = 0; r < rows; ++r) for (unsigned s = 0; s < cols; ++s) {
            auto store = [&](OpBuilder &sb, Location l) {
              sb.create<memref::StoreOp>(l, reduction.getResult(r * cols + s), c,
                                         ValueRange{ri[r], cj[s]});
            };
            if (valid[r * cols + s]) jb.create<scf::IfOp>(l, valid[r * cols + s],
                [&](OpBuilder &sb, Location l) { store(sb, l); sb.create<scf::YieldOp>(l); });
            else store(jb, l);
          }
          jb.create<scf::YieldOp>(l);
        });
        ib.create<scf::YieldOp>(l);
      });
      mm.erase();
    }
  }
};

// A partition marker is a compile-time directive, not a runtime memory access.
// Extracting the base pointer through the standard descriptor conversion avoids
// changing function signatures. LLVM folds ptrtoint/inttoptr back to the base.
struct BankMatmulBuffers : sodap::impl::BankMatmulBuffersBase<BankMatmulBuffers> {
  using Base::Base;
  void runOnOperation() override {
    if (!rows || !cols) {
      getOperation().emitError("bank factors must be positive");
      return signalPassFailure();
    }
    llvm::MapVector<Value, std::pair<unsigned, unsigned>> banks;
    bool invalid = false;
    auto mark = [&](Value value, unsigned dim, unsigned factor) {
      while (true) {
        if (auto sub = value.getDefiningOp<memref::SubViewOp>()) {
          auto dropped = sub.getDroppedDims();
          unsigned visible = 0;
          for (unsigned i = 0; i < dropped.size(); ++i)
            if (!dropped[i] && visible++ == dim) { dim = i; break; }
          value = sub.getSource();
        } else if (auto cast = value.getDefiningOp<memref::CastOp>()) {
          value = cast.getSource();
        } else break;
      }
      auto type = dyn_cast<MemRefType>(value.getType());
      if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity()) {
        getOperation().emitError("banking requires a static identity-layout base buffer");
        invalid = true;
        return;
      }
      if (factor < 2) return;
      auto [it, inserted] = banks.try_emplace(value, dim, factor);
      if (!inserted && it->second != std::make_pair(dim, factor)) {
        getOperation().emitError("conflicting bank mappings for a shared buffer");
        invalid = true;
      }
    };
    getOperation().walk([&](linalg::MatmulOp op) {
      if (!op.hasPureBufferSemantics()) {
        op.emitError("banking runs after bufferization"); invalid = true; return;
      }
      mark(op.getInputs()[0], 0, rows);
      mark(op.getInputs()[1], 1, cols);
      mark(op.getOutputs()[0], 1, cols);
    });
    if (invalid) return signalPassFailure();
    if (banks.empty()) return;
    // Every task sharing the same physical PLM needs the same padded layout.
    // Propagate through the explicit node call graph before changing any types.
    bool changed = true;
    while (changed && !invalid) {
      changed = false;
      getOperation().walk([&](func::CallOp call) {
        auto callee = getOperation().lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee || callee.isExternal()) return;
        for (auto [i, operand] : llvm::enumerate(call.getOperands())) {
          auto arg = callee.getArgument(i);
          auto a = banks.find(operand), b = banks.find(arg);
          if (a != banks.end() && b == banks.end()) {
            auto layout = a->second;
            mark(arg, layout.first, layout.second); changed = true;
          } else if (b != banks.end() && a == banks.end()) {
            // The node ABI passes whole local buffers, not interior pointers.
            if (!isa<MemRefType>(operand.getType()) ||
                operand.getDefiningOp<memref::SubViewOp>()) {
              call.emitError("banked node arguments must pass a whole local buffer");
              invalid = true; continue;
            }
            auto layout = b->second;
            mark(operand, layout.first, layout.second); changed = true;
          } else if (a != banks.end() && b != banks.end() && a->second != b->second) {
            call.emitError("inconsistent banking across a node call"); invalid = true;
          }
        }
      });
    }
    if (invalid) return signalPassFailure();
    // CSROA requires exact divisibility. Pad PHYSICAL storage, preserving the
    // original logical view, iteration bounds and DMA transfer lengths. This
    // handles partial register blocks without reducing the arithmetic budget.
    for (auto [buffer, layout] : banks) {
      auto original = cast<MemRefType>(buffer.getType());
      SmallVector<int64_t> shape(original.getShape());
      unsigned dim = layout.first, factor = layout.second;
      int64_t padded = llvm::divideCeil(shape[dim], int64_t(factor)) * factor;
      if (padded == shape[dim]) continue;
      shape[dim] = padded;
      auto physical = MemRefType::get(shape, original.getElementType());
      if (!isa<BlockArgument>(buffer) &&
          !isa<memref::AllocOp, memref::AllocaOp>(buffer.getDefiningOp())) {
        getOperation().emitError("bank padding requires an allocation or node argument");
        return signalPassFailure();
      }
      SmallVector<OpOperand *> uses;
      for (auto &use : buffer.getUses()) uses.push_back(&use);
      buffer.setType(physical);
      OpBuilder view(&getContext());
      if (auto arg = dyn_cast<BlockArgument>(buffer)) view.setInsertionPointToStart(arg.getOwner());
      else view.setInsertionPointAfter(buffer.getDefiningOp());
      SmallVector<OpFoldResult> offsets(original.getRank(), view.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(original.getRank(), view.getIndexAttr(1));
      SmallVector<OpFoldResult> sizes;
      for (int64_t d : original.getShape()) sizes.push_back(view.getIndexAttr(d));
      Value logical = view.create<memref::SubViewOp>(buffer.getLoc(), buffer, offsets, sizes, strides);
      for (OpOperand *use : uses)
        if (!isa<func::CallOp, memref::DeallocOp>(use->getOwner())) use->set(logical);
    }
    for (auto func : getOperation().getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;
      SmallVector<Type> types;
      for (auto arg : func.getArguments()) types.push_back(arg.getType());
      func.setType(FunctionType::get(&getContext(), types, func.getResultTypes()));
    }
    // Existing slices now inherit the padded row stride. Their logical shape
    // and rank reduction stay unchanged.
    getOperation().walk<WalkOrder::PreOrder>([&](memref::SubViewOp sub) {
      auto old = sub.getType();
      auto type = memref::SubViewOp::inferRankReducedResultType(old.getShape(),
          cast<MemRefType>(sub.getSource().getType()), sub.getMixedOffsets(),
          sub.getMixedSizes(), sub.getMixedStrides());
      sub.getResult().setType(cast<MemRefType>(type));
    });
    uint64_t physicalBytes = 0;
    getOperation().walk([&](Operation *op) {
      if (!isa<memref::AllocOp, memref::AllocaOp>(op)) return;
      auto type = cast<MemRefType>(op->getResult(0).getType());
      if (!type.hasStaticShape()) { invalid = true; return; }
      Type element = type.getElementType();
      unsigned bits = element.isIndex() ? 64 : element.getIntOrFloatBitWidth();
      uint64_t elements = type.getNumElements(), bytesPerElement = (bits + 7) / 8;
      if (elements > (UINT64_MAX - physicalBytes) / bytesPerElement) {
        invalid = true; return;
      }
      physicalBytes += elements * bytesPerElement;
    });
    if (invalid || (plmBytes && physicalBytes > plmBytes)) {
      getOperation().emitError("padded local memory exceeds the configured storage budget or has an unsupported allocation")
          << ": " << physicalBytes << " bytes, budget " << uint64_t(plmBytes);
      return signalPassFailure();
    }
    if (!report.empty()) {
      std::error_code error;
      llvm::raw_fd_ostream stream(report, error, llvm::sys::fs::OF_Text);
      if (error) {
        getOperation().emitError("cannot write physical storage report: ") << error.message();
        return signalPassFailure();
      }
      stream << llvm::json::Value(llvm::json::Object{
          {"physical_local_array_bytes", physicalBytes}, {"budget_bytes", uint64_t(plmBytes)},
          {"scope", "Local memref allocations after padding; excludes HLS replication and BRAM rounding"}}) << "\n";
    }
    OpBuilder b(&getContext());
    auto ptr = LLVM::LLVMPointerType::get(&getContext());
    auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&getContext()),
        {ptr, b.getI32Type(), b.getI32Type(), b.getI32Type()});
    constexpr StringLiteral name = "__bambu_csroa_partition__";
    auto fn = getOperation().lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!fn) {
      b.setInsertionPointToStart(getOperation().getBody());
      fn = b.create<LLVM::LLVMFuncOp>(getOperation().getLoc(), name, fnType);
    }
    for (auto [buffer, layout] : banks) {
      if (auto arg = dyn_cast<BlockArgument>(buffer)) b.setInsertionPointToStart(arg.getOwner());
      else b.setInsertionPointAfter(buffer.getDefiningOp());
      auto loc = buffer.getLoc();
      Value address = b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, buffer);
      Value integer = b.create<arith::IndexCastOp>(loc, b.getI64Type(), address);
      Value base = b.create<LLVM::IntToPtrOp>(loc, ptr, integer);
      auto constant = [&](unsigned v) -> Value {
        return b.create<LLVM::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(v));
      };
      b.create<LLVM::CallOp>(loc, fn, ValueRange{base, constant(2),
          constant(layout.second), constant(layout.first + 1)});
    }
  }
};

static bool shaped(MemRefType type) {
  return type.hasStaticShape() && type.getLayout().isIdentity() &&
         type.getMemorySpaceAsInt() == 0;
}
static Type arrayType(MemRefType type, const LLVMTypeConverter &converter) {
  Type result = converter.convertType(type.getElementType());
  for (auto d : llvm::reverse(type.getShape())) result = LLVM::LLVMArrayType::get(result, d);
  return result;
}

struct ArrayAlloca : ConvertOpToLLVMPattern<memref::AllocaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(memref::AllocaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &r) const override {
    auto type = op.getType();
    if (!shaped(type)) return failure();
    auto ptr = LLVM::LLVMPointerType::get(r.getContext());
    Value one = r.create<LLVM::ConstantOp>(op.getLoc(), r.getI64Type(), r.getI64IntegerAttr(1));
    Value memory = r.create<LLVM::AllocaOp>(op.getLoc(), ptr,
        arrayType(type, *getTypeConverter()), one, op.getAlignment().value_or(0));
    auto descriptor = MemRefDescriptor::fromStaticShape(r, op.getLoc(),
        *getTypeConverter(), type, memory);
    r.replaceOp(op, {descriptor});
    return success();
  }
};

static Value arrayAddress(Location loc, MemRefType type, Value memref,
                          ValueRange indices, bool argument, const LLVMTypeConverter &converter,
                          ConversionPatternRewriter &r) {
  Value base = MemRefDescriptor(memref).alignedPtr(r, loc);
  Type source = arrayType(type, converter);
  SmallVector<LLVM::GEPArg> gep;
  // Bambu follows the C ABI: an array argument points to its first row;
  // an alloca points to the whole array. The leading index must agree.
  if (argument && type.getRank()) source = cast<LLVM::LLVMArrayType>(source).getElementType();
  else gep.push_back(int32_t(0));
  for (Value i : indices) gep.push_back(i);
  return r.create<LLVM::GEPOp>(loc, base.getType(), source, base, gep);
}
struct ArrayLoad : ConvertOpToLLVMPattern<memref::LoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(memref::LoadOp op, OpAdaptor a,
                                ConversionPatternRewriter &r) const override {
    auto type = op.getMemRefType();
    if (!shaped(type)) return failure();
    Value ptr = arrayAddress(op.getLoc(), type, a.getMemref(), a.getIndices(),
        op->hasAttr("sodap.array_argument"), *getTypeConverter(), r);
    r.replaceOpWithNewOp<LLVM::LoadOp>(op, getTypeConverter()->convertType(type.getElementType()), ptr);
    return success();
  }
};
struct ArrayStore : ConvertOpToLLVMPattern<memref::StoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(memref::StoreOp op, OpAdaptor a,
                                ConversionPatternRewriter &r) const override {
    auto type = op.getMemRefType();
    if (!shaped(type)) return failure();
    Value ptr = arrayAddress(op.getLoc(), type, a.getMemref(), a.getIndices(),
        op->hasAttr("sodap.array_argument"), *getTypeConverter(), r);
    r.replaceOpWithNewOp<LLVM::StoreOp>(op, a.getValue(), ptr);
    return success();
  }
};

struct FinalizeBankedMemRefsToLLVM :
    sodap::impl::FinalizeBankedMemRefsToLLVMBase<FinalizeBankedMemRefsToLLVM> {
  void runOnOperation() override {
    // Record the original storage provenance before conversion materializations
    // replace memref operands with descriptor casts.
    getOperation().walk([&](Operation *op) {
      Value memory;
      if (auto load = dyn_cast<memref::LoadOp>(op)) memory = load.getMemref();
      if (auto store = dyn_cast<memref::StoreOp>(op)) memory = store.getMemref();
      if (memory && isa<BlockArgument>(memory))
        op->setAttr("sodap.array_argument", UnitAttr::get(&getContext()));
    });
    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    patterns.add<ArrayAlloca, ArrayLoad, ArrayStore>(converter, PatternBenefit(10));
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<memref::MemRefDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
