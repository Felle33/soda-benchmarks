//===- Prefix.h - The loops a task walks a shared buffer with ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The prefix of an access to a buffer shared by the tasks of a dispatch: the
// outer loops of the task stepping through the buffer a block at a time (see
// "Ping-pong buffers and the load scheduler" in docs/LinalgToLCSPlan.md).
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_TRANSFORMS_PREFIX_H
#define SODAP_DIALECT_DATAFLOW_TRANSFORMS_PREFIX_H

#include "sodap/Dialect/Dataflow/Dataflow.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/MapVector.h"

#include <array>

namespace mlir {
namespace sodap {
namespace dataflow {

/// A loop of a prefix, and the dim of the buffer it steps through.
struct Step {
  scf::ForOp loop;
  unsigned dim;
};
using Prefix = SmallVector<Step>;

/// An op of a task using the buffer, through `view`. The op may be a part
/// of a static tile (partOfTile): it then reads or writes through its users.
struct Access {
  Operation *op;
  Value view;
  bool writes;
  bool reads;
  /// For a part of a tile, its last user (through views), in `op`'s block:
  /// where the access ends.
  Operation *end = nullptr;
  Operation *last() const { return end ? end : op; }
};

/// The constant `size` is at most: a constant, or a min with a constant
/// operand (tile_and_fuse's min(step, ub - iv)).
inline std::optional<int64_t> boundOf(OpFoldResult size) {
  if (std::optional<int64_t> c = getConstantIntValue(size))
    return c;
  Value v = cast<Value>(size);
  std::optional<int64_t> bound;
  auto take = [&](int64_t c) { bound = bound ? std::min(*bound, c) : c; };
  if (auto min = v.getDefiningOp<affine::AffineMinOp>()) {
    for (AffineExpr e : min.getMap().getResults())
      if (auto c = dyn_cast<AffineConstantExpr>(e))
        take(c.getValue());
  } else if (isa_and_present<arith::MinUIOp, arith::MinSIOp>(v.getDefiningOp())) {
    for (Value o : v.getDefiningOp()->getOperands())
      if (std::optional<int64_t> c = getConstantIntValue(o); c && *c >= 0)
        take(*c);
  }
  return bound;
}

inline bool atMost(OpFoldResult size, int64_t bound) {
  std::optional<int64_t> b = boundOf(size);
  return b && *b <= bound;
}

/// A part of a static tile: a unit-stride, rank-preserving subview at offset
/// 0 of a view with a static shape, no larger than it (an element-wise op
/// cut to the part of its tile inside the matrix, tile_and_fuse). The tasks'
/// schedule moves the whole tile; sodap-dataflow-tails cuts the transfer.
inline bool partOfTile(Operation *op) {
  auto subview = dyn_cast<memref::SubViewOp>(op);
  if (!subview)
    return false;
  auto source = subview.getSourceType();
  if (!source.hasStaticShape() ||
      subview.getType().getRank() != source.getRank() ||
      !llvm::all_of(subview.getMixedOffsets(),
                    [](OpFoldResult o) { return isConstantIntValue(o, 0); }) ||
      !llvm::all_of(subview.getStaticStrides(),
                    [](int64_t s) { return s == 1; }))
    return false;
  for (auto [size, bound] :
       llvm::zip_equal(subview.getMixedSizes(), source.getShape()))
    if (!atMost(size, bound))
      return false;
  return true;
}

/// The static tile a part of a tile is of, or `view` itself.
inline Value tileOfPart(Value view) {
  while (Operation *def = view.getDefiningOp()) {
    if (!partOfTile(def))
      break;
    view = cast<memref::SubViewOp>(def).getSource();
  }
  return view;
}

/// `op` reads `view` through `target` instead. A part of a tile stays the
/// same op (the schedule may hold it): its source becomes `target`, its type
/// follows, and a cast gives its users the type they had.
inline void replaceAccessedView(Operation *op, Value view, Value target) {
  auto part = dyn_cast<memref::SubViewOp>(op);
  if (!part || !partOfTile(op) || part.getSource() != view) {
    op->replaceUsesOfWith(view, target);
    return;
  }
  MemRefType old = part.getType();
  part.getSourceMutable().assign(target);
  auto type = cast<MemRefType>(memref::SubViewOp::inferResultType(
      cast<MemRefType>(target.getType()), part.getStaticOffsets(),
      part.getStaticSizes(), part.getStaticStrides()));
  if (type == old)
    return;
  part.getResult().setType(type);
  OpBuilder builder(op->getContext());
  builder.setInsertionPointAfter(op);
  auto cast = builder.create<memref::CastOp>(part.getLoc(), old, part.getResult());
  part.getResult().replaceAllUsesExcept(cast, cast);
}

/// Whether some use of `value`, through views, has the effect.
template <typename Effect>
inline bool hasEffectThroughViews(Value value) {
  for (OpOperand &use : value.getUses()) {
    Operation *op = use.getOwner();
    if (auto view = dyn_cast<ViewLikeOpInterface>(op);
        view && view.getViewSource() == value) {
      if (hasEffectThroughViews<Effect>(op->getResult(0)))
        return true;
      continue;
    }
    if (hasEffect<Effect>(op, value))
      return true;
  }
  return false;
}

/// Lower bound, upper bound and step, if constant.
inline std::optional<std::array<int64_t, 3>> boundsOf(scf::ForOp loop) {
  std::optional<int64_t> lb = getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(loop.getStep());
  if (!lb || !ub || !step)
    return std::nullopt;
  return std::array<int64_t, 3>{*lb, *ub, *step};
}

/// The step, if constant: a tile's size. The bounds may be values (a tile
/// from sodap-join-multiple-tiles has its ranges as arguments).
inline std::optional<int64_t> stepOf(scf::ForOp loop) {
  return getConstantIntValue(loop.getStep());
}

/// Lower bound, upper bound and step, constants or values.
inline std::array<OpFoldResult, 3> foldedBoundsOf(scf::ForOp loop) {
  return {getAsOpFoldResult(loop.getLowerBound()),
          getAsOpFoldResult(loop.getUpperBound()),
          getAsOpFoldResult(loop.getStep())};
}

/// The loops of `task`, from the outermost, whose induction variable is the
/// offset of `access` along a dim of `buffer`, with the step as size.
inline Prefix prefixOf(const Access &access, Value buffer, TaskOp task) {
  auto type = cast<MemRefType>(buffer.getType());
  unsigned rank = type.getRank();
  SmallVector<memref::SubViewOp> chain;
  for (Value view = access.view; view != buffer;) {
    auto subview = view.getDefiningOp<memref::SubViewOp>();
    if (!subview || subview.getType().getRank() != rank ||
        !llvm::all_of(subview.getStaticStrides(),
                      [](int64_t stride) { return stride == 1; }))
      return {};
    chain.push_back(subview);
    view = subview.getSource();
  }

  // Per dim: the one value the offset varies with (null if none, or if the
  // dim can't be a block dim), and the size of the access.
  SmallVector<Value> ivs(rank);
  SmallVector<int64_t> sizes(type.getShape());
  SmallVector<bool> fixed(rank, false);
  for (memref::SubViewOp subview : llvm::reverse(chain))
    for (unsigned d = 0; d < rank; ++d) {
      OpFoldResult offset = subview.getMixedOffsets()[d];
      if (std::optional<int64_t> constant = getConstantIntValue(offset))
        fixed[d] = fixed[d] || *constant != 0;
      else if (ivs[d])
        fixed[d] = true;
      else
        ivs[d] = offset.get<Value>();
      if (std::optional<int64_t> size =
              getConstantIntValue(subview.getMixedSizes()[d]))
        sizes[d] = *size;
      else
        fixed[d] = true;
    }

  SmallVector<scf::ForOp> loops;
  for (Operation *op = access.op->getParentOp(); op != task;
       op = op->getParentOp())
    if (auto loop = dyn_cast<scf::ForOp>(op))
      loops.push_back(loop);
  Prefix prefix;
  for (scf::ForOp loop : llvm::reverse(loops)) {
    std::optional<int64_t> step = stepOf(loop);
    auto *it = llvm::find(ivs, loop.getInductionVar());
    if (!step || it == ivs.end())
      break;
    unsigned d = it - ivs.begin();
    if (fixed[d] || sizes[d] != *step)
      break;
    prefix.push_back({loop, d});
  }
  return prefix;
}

/// The prefix all the accesses of a task share.
inline Prefix prefixOf(ArrayRef<Access> accesses, Value buffer, TaskOp task) {
  Prefix common = prefixOf(accesses.front(), buffer, task);
  for (const Access &access : accesses.drop_front()) {
    Prefix prefix = prefixOf(access, buffer, task);
    unsigned n = 0;
    while (n < common.size() && n < prefix.size() &&
           common[n].loop == prefix[n].loop && common[n].dim == prefix[n].dim)
      ++n;
    common.resize(n);
  }
  return common;
}

/// Cuts the prefixes of a writer and a reader where they stop agreeing (the
/// same dims, over loops with the same bounds): their shared prefix.
inline void sharePrefix(Prefix &writes, Prefix &reads) {
  unsigned n = 0;
  while (n < writes.size() && n < reads.size() &&
         writes[n].dim == reads[n].dim &&
         foldedBoundsOf(writes[n].loop) == foldedBoundsOf(reads[n].loop))
    ++n;
  writes.resize(n);
  reads.resize(n);
}

/// Replace a whole copy with a loop nest over `steps`, copying a block per
/// iteration. Returns the block copy.
inline memref::CopyOp splitCopy(OpBuilder &builder, memref::CopyOp copy,
                                ArrayRef<Step> steps) {
  if (steps.empty())
    return copy;
  Location loc = copy.getLoc();
  auto type = cast<MemRefType>(copy.getSource().getType());
  SmallVector<OpFoldResult> offsets(type.getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(type.getRank(), builder.getIndexAttr(1));
  builder.setInsertionPoint(copy);
  SmallVector<OpFoldResult> sizes;
  for (auto [d, size] : llvm::enumerate(type.getShape()))
    sizes.push_back(ShapedType::isDynamic(size)
                        ? OpFoldResult(builder.createOrFold<memref::DimOp>(
                              loc, copy.getSource(), d))
                        : builder.getIndexAttr(size));
  // A constant bound anew, a value (an argument) as it is.
  auto bound = [&](OpFoldResult ofr) -> Value {
    if (std::optional<int64_t> c = getConstantIntValue(ofr))
      return builder.create<arith::ConstantIndexOp>(loc, *c);
    return cast<Value>(ofr);
  };
  for (Step step : steps) {
    auto [lb, ub, stride] = foldedBoundsOf(step.loop);
    auto loop = builder.create<scf::ForOp>(loc, bound(lb), bound(ub),
                                           bound(stride));
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    offsets[step.dim] = loop.getInductionVar();
    sizes[step.dim] = builder.getIndexAttr(*getConstantIntValue(stride));
  }
  Value source = builder.create<memref::SubViewOp>(loc, copy.getSource(),
                                                   offsets, sizes, strides);
  Value target = builder.create<memref::SubViewOp>(loc, copy.getTarget(),
                                                   offsets, sizes, strides);
  auto block = builder.create<memref::CopyOp>(loc, source, target);
  copy.erase();
  return block;
}

/// The accesses of each task of `dispatch` to `buffer`, through views of it.
inline FailureOr<llvm::MapVector<TaskOp, SmallVector<Access>>>
accessesOf(Value buffer, DispatchOp dispatch) {
  Block &body = dispatch.getBody().front();
  llvm::MapVector<TaskOp, SmallVector<Access>> accesses;
  SmallVector<Value> views{buffer};
  while (!views.empty()) {
    Value view = views.pop_back_val();
    for (OpOperand &use : view.getUses()) {
      Operation *op = use.getOwner();
      if (auto viewOp = dyn_cast<ViewLikeOpInterface>(op);
          viewOp && viewOp.getViewSource() == view && !partOfTile(op)) {
        views.push_back(op->getResult(0));
        continue;
      }
      auto task = dyn_cast_or_null<TaskOp>(body.findAncestorOpInBlock(*op));
      if (!task) {
        if (isa<YieldOp>(op))
          continue;
        op->emitOpError("expected the buffers of a dispatch to be used in "
                        "its tasks");
        return failure();
      }
      if (partOfTile(op)) {
        Operation *end = op;
        SmallVector<Value> through{op->getResult(0)};
        while (!through.empty())
          for (Operation *user : through.pop_back_val().getUsers()) {
            if (isa<ViewLikeOpInterface, memref::CastOp>(user))
              through.push_back(user->getResult(0));
            if (Operation *at = op->getBlock()->findAncestorOpInBlock(*user);
                at && end->isBeforeInBlock(at))
              end = at;
          }
        accesses[task].push_back(
            {op, view, hasEffectThroughViews<MemoryEffects::Write>(op->getResult(0)),
             hasEffectThroughViews<MemoryEffects::Read>(op->getResult(0)), end});
      }
      else
        accesses[task].push_back({op, view,
                                  hasEffect<MemoryEffects::Write>(op, view),
                                  hasEffect<MemoryEffects::Read>(op, view)});
    }
  }
  return accesses;
}

} // namespace dataflow
} // namespace sodap
} // namespace mlir

#endif // SODAP_DIALECT_DATAFLOW_TRANSFORMS_PREFIX_H
