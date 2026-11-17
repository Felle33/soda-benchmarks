// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef SODAP_RUNTIME_TILE_BOUNDS_H
#define SODAP_RUNTIME_TILE_BOUNDS_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace mlir::sodap {
// Backing arrays and bank counts stay static. Logical iteration extents are
// bounded by the source iteration domain after tiling and task scheduling.
inline FailureOr<Value> runtimeTileBound(OpBuilder &b, Operation *op,
                                         int64_t axis, int64_t capacity) {
  auto loc=op->getLoc();
  Value whole=b.create<arith::ConstantIndexOp>(loc,capacity);
  if(!axis)return whole;
  for(Operation *parent=op->getParentOp();parent;parent=parent->getParentOp()) {
    auto loop=dyn_cast<scf::ForOp>(parent);
    auto role=parent->getAttrOfType<IntegerAttr>("sodap.bound_axis");
    if(loop && role && role.getInt()==axis) {
      Value left=b.create<arith::SubIOp>(loc,loop.getUpperBound(),loop.getInductionVar());
      return Value(b.create<arith::MinUIOp>(loc,whole,left));
    }
  }
  op->emitError("bounded contraction lost its enclosing semantic tile axis ")<<axis;
  return failure();
}
} // namespace mlir::sodap
#endif
