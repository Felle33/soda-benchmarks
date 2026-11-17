//===- Tails.cpp - Partial tiles of run-time shapes -----------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A tile from sodap-join-multiple-tiles has its shapes as arguments, and tile
// sizes that need not divide them: the tasks walk whole static tiles, and the
// last tile of a range reaches past it. Cut every copy of a tile of such an
// argument to the part inside the argument, min(size, dim - offset) along
// each dynamic dim; a read first fills the local tile with zeros when the
// part is smaller than the tile, and only then (see the pass description).
//
//===----------------------------------------------------------------------===//

#include "sodap/Dialect/Dataflow/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace sodap {
namespace dataflow {
#define GEN_PASS_DEF_TAILS
#include "sodap/Dialect/Dataflow/Transforms/Passes.h.inc"
} // namespace dataflow
} // namespace sodap
} // namespace mlir

using namespace mlir;

namespace {
struct Tails : public mlir::sodap::dataflow::impl::TailsBase<Tails> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;
    Block &entry = func.front();
    // The static tiles of the arguments with a dynamic shape, and their copies.
    SmallVector<std::pair<memref::SubViewOp, memref::CopyOp>> copies;
    for (BlockArgument arg : entry.getArguments()) {
      auto type = dyn_cast<MemRefType>(arg.getType());
      if (!type || type.hasStaticShape())
        continue;
      for (Operation *user : arg.getUsers()) {
        auto tile = dyn_cast<memref::SubViewOp>(user);
        if (!tile || !tile.hasUnitStride() ||
            tile.getType().getRank() != type.getRank() ||
            ShapedType::isDynamicShape(tile.getStaticSizes()))
          continue;
        for (Operation *tileUser : tile->getUsers())
          if (auto copy = dyn_cast<memref::CopyOp>(tileUser))
            copies.push_back({tile, copy});
      }
    }

    OpBuilder b(func.getContext());
    for (auto [tile, copy] : copies) {
      Location loc = copy.getLoc();
      auto arg = cast<BlockArgument>(tile.getSource());
      auto type = cast<MemRefType>(arg.getType());
      bool read = copy.getSource() == tile.getResult();
      Value local = read ? copy.getTarget() : copy.getSource();
      if (local == tile.getResult()) {
        copy.emitOpError("expected a copy between a tile of an argument and a "
                         "local buffer");
        return signalPassFailure();
      }
      auto dims = func.getArgAttrOfType<DenseI64ArrayAttr>(arg.getArgNumber(),
                                                           "sodap.dims");
      b.setInsertionPoint(copy);
      unsigned rank = type.getRank();
      SmallVector<OpFoldResult> offsets = tile.getMixedOffsets(), sizes;
      SmallVector<OpFoldResult> zeros(rank, b.getIndexAttr(0)),
          ones(rank, b.getIndexAttr(1));
      Value partial;
      for (unsigned d = 0; d < rank; ++d) {
        int64_t size = tile.getStaticSizes()[d];
        if (!type.isDynamicDim(d)) {
          sizes.push_back(b.getIndexAttr(size));
          continue;
        }
        Value dim = dims && d < dims.size()
                        ? Value(func.getArgument(dims[d]))
                        : Value(b.create<memref::DimOp>(loc, arg, d));
        Value whole = b.create<arith::ConstantIndexOp>(loc, size);
        Value left = b.create<arith::SubIOp>(
            loc, dim, getValueOrCreateConstantIndexOp(b, loc, offsets[d]));
        Value part = b.create<arith::MinUIOp>(loc, left, whole);
        sizes.push_back(part);
        if (!read)
          continue;
        Value cut = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                            part, whole);
        partial = partial ? Value(b.create<arith::OrIOp>(loc, partial, cut))
                          : cut;
      }
      if (read && !func->hasAttr("sodap.bounded_contractions")) {
        // Zeros outside the part, so the tasks compute on the whole tile; a
        // whole tile skips this.
        auto pad = b.create<scf::IfOp>(loc, partial, /*withElseRegion=*/false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(pad.thenBlock());
        Type element = type.getElementType();
        Value zero = b.create<arith::ConstantOp>(loc, element,
                                                 b.getZeroAttr(element));
        b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{local});
      }
      Value outer =
          b.create<memref::SubViewOp>(loc, arg, offsets, sizes, ones);
      Value inner =
          b.create<memref::SubViewOp>(loc, local, zeros, sizes, ones);
      b.create<memref::CopyOp>(loc, read ? outer : inner,
                               read ? inner : outer);
      copy.erase();
      if (tile.use_empty())
        tile.erase();
    }
  }
};
} // namespace
