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

namespace mlir::sodap {
/// D = alpha * A.B + beta * C, as tosa ops; beta == 0 gives alpha * A.B
/// whatever C holds. `a` and `b` are tosa.matmul's operands, 1xMxK and 1xKxN;
/// `c`, MxN, has D's type; `alpha` and `beta` are f32.
Value buildGemmBody(OpBuilder &b, Location loc, Value a, Value rhs, Value c,
                    Value alpha, Value beta, Value hasC = {});
} // namespace mlir::sodap

#endif // SODAP_TOSA_GEMM_H
