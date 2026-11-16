//===- GuardReads.cpp - Reads that a scalar makes useless -----------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GEMM's C is read only when beta isn't 0 (sodap-join-multiple-tiles marks C
// with sodap.read_unless_zero): each copy in of a tile of such an argument
// runs under scf.if (beta une 0), as the measured engine's hasC.
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_GUARDREADS
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;

namespace {
struct GuardReads
    : public mlir::sodap::dataflow::impl::GuardReadsBase<GuardReads> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;
    Block &entry = func.front();
    for (BlockArgument arg : entry.getArguments()) {
      auto guard = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                      "sodap.read_unless_zero");
      if (!guard)
        continue;
      int64_t s = guard.getInt();
      if (s < 0 || s >= int64_t(entry.getNumArguments()) ||
          !entry.getArgument(s).getType().isF32()) {
        func.emitError("sodap.read_unless_zero of argument ")
            << arg.getArgNumber() << " should be an f32 argument";
        return signalPassFailure();
      }
      Value scalar = entry.getArgument(s);
      // The copies out of the argument, or of a tile of it.
      SmallVector<memref::CopyOp> reads;
      SmallVector<Value> views = {arg};
      while (!views.empty()) {
        Value v = views.pop_back_val();
        for (Operation *user : v.getUsers()) {
          if (auto view = dyn_cast<memref::SubViewOp>(user))
            views.push_back(view);
          else if (auto copy = dyn_cast<memref::CopyOp>(user);
                   copy && copy.getSource() == v)
            reads.push_back(copy);
        }
      }
      for (memref::CopyOp copy : reads) {
        OpBuilder b(copy);
        Location loc = copy.getLoc();
        Value zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0));
        // Unordered: a NaN reads, as `hasC = beta UNE 0`.
        Value read = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                             scalar, zero);
        auto ifOp = b.create<scf::IfOp>(loc, read, /*withElseRegion=*/false);
        copy->moveBefore(ifOp.thenBlock()->getTerminator());
      }
    }
  }
};
} // namespace
