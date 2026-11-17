//===- Gemm.h - GEMM on tosa ops --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_TOSA_GEMM_H
#define SODAP_TOSA_GEMM_H

#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include <optional>

namespace mlir::sodap {
/// D = alpha * A.B + beta * C, as tosa ops; beta == 0 gives alpha * A.B
/// whatever C holds. `a` and `b` are tosa.matmul's operands, 1xMxK and 1xKxN;
/// `c`, MxN, has D's type; `alpha` and `beta` are f32.
Value buildGemmBody(OpBuilder &b, Location loc, Value a, Value rhs, Value c,
                    Value alpha, Value beta, Value hasC = {});
/// Shared source-expression matching. Hardware outlining and host planning must
/// agree on the operation, including explicit C presence and scalar operands.
inline Type squeezeGemmBatch(Type type) {
  auto t = cast<RankedTensorType>(type);
  if (t.getRank() < 2 || t.getDimSize(0) != 1) return type;
  return RankedTensorType::get(t.getShape().drop_front(), t.getElementType());
}
/// `v`'s only user, if it has one use.
inline Operation *onlyUser(Value v) {
  return v.hasOneUse() ? *v.user_begin() : nullptr;
}

/// The tensor of one f32 that `v`, of one element, reshapes (a model's
/// scalar argument), or null.
inline Value scalarOf(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || !type.hasStaticShape() || type.getNumElements() != 1 ||
      !type.getElementType().isF32())
    return {};
  while (auto r = v.getDefiningOp<tosa::ReshapeOp>())
    v = r.getInput1();
  return v;
}

/// `x` of `mul`, a tosa.mul of `x` by a scalar, and the scalar.
inline std::optional<std::pair<Value, Value>> scaled(tosa::MulOp mul) {
  if (!mul || mul.getShift() != 0)
    return std::nullopt;
  for (auto [s, x] : {std::pair(mul.getInput1(), mul.getInput2()),
                      std::pair(mul.getInput2(), mul.getInput1())})
    if (Value scalar = scalarOf(s); scalar && x.getType() == mul.getType())
      return std::pair(x, scalar);
  return std::nullopt;
}

/// A GEMM around a tosa.matmul: D = alpha * A.B + beta * C, each part
/// optional, as torch-mlir writes it (reshape, mul by a scalar, add of a mul
/// by a scalar). It grows only through values with one use: an intermediate
/// another op reads stays the GEMM's result.
struct Gemm {
  tosa::MatMulOp matmul;
  SmallVector<Operation *> ops; // matmul to the op giving D, in order
  Value c, alpha, beta;         // null when missing: none, 1, 0
  Operation *betaMul = nullptr; // beta * C, folded into the GEMM
};

inline Gemm matchGemm(tosa::MatMulOp op) {
  Gemm g{op, {op}};
  auto r = dyn_cast_or_null<tosa::ReshapeOp>(onlyUser(op.getResult()));
  if (!r || r.getType() != squeezeGemmBatch(op.getType()))
    return g;
  g.ops.push_back(r);
  Value d = r.getResult();
  if (auto s = scaled(dyn_cast_or_null<tosa::MulOp>(onlyUser(d)));
      s && s->first == d) {
    g.alpha = s->second;
    g.ops.push_back(*d.user_begin());
    d = g.ops.back()->getResult(0);
  }
  auto add = dyn_cast_or_null<tosa::AddOp>(onlyUser(d));
  if (!add || add.getType() != d.getType() ||
      add.getInput1() == add.getInput2())
    return g;
  Value x = add.getInput1() == d ? add.getInput2() : add.getInput1();
  g.ops.push_back(add);
  g.c = x;
  auto mul = x.getDefiningOp<tosa::MulOp>();
  if (auto s = scaled(mul); s && x.hasOneUse()) {
    g.c = s->first;
    g.beta = s->second;
    g.betaMul = mul;
  } else {
    g.beta = Value(); // C as it is: beta = 1
  }
  return g;
}

} // namespace mlir::sodap

#endif // SODAP_TOSA_GEMM_H
