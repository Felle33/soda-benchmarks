//===- JoinMultipleTiles.cpp - One tile per operator ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// sodap-join-multiple-tiles: the tiles of one operator become one tile whose
// shapes are arguments, launched once per invocation.
//
//===----------------------------------------------------------------------===//

#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include "sodap/Tosa/Gemm.h"
#include "sodap/Tosa/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::sodap {
#define GEN_PASS_DEF_JOINMULTIPLETILES
#include "sodap/Tosa/Passes.h.inc"

Value buildGemmBody(OpBuilder &b, Location loc, Value a, Value rhs, Value c,
                    Value alpha, Value beta, Value hasC) {
  auto matrix = cast<RankedTensorType>(c.getType());
  auto at = cast<RankedTensorType>(a.getType()),
       bt = cast<RankedTensorType>(rhs.getType());
  auto batch = RankedTensorType::get({1, at.getDimSize(1), bt.getDimSize(2)},
                                     matrix.getElementType());
  auto scalar = RankedTensorType::get({1, 1}, b.getF32Type());
  auto flag = RankedTensorType::get({1, 1}, b.getI1Type());
  Value product = b.create<tosa::MatMulOp>(loc, batch, a, rhs);
  Value ab = b.create<tensor::CollapseShapeOp>(
      loc, matrix, product, ArrayRef<ReassociationIndices>{{0, 1}, {2}});
  auto mul = [&](Value x, Value y) {
    return b.create<tosa::MulOp>(loc, matrix, x, y, b.getI8IntegerAttr(0))
        .getResult();
  };
  Value alphaT = b.create<tensor::FromElementsOp>(loc, scalar, alpha);
  Value betaT = b.create<tensor::FromElementsOp>(loc, scalar, beta);
  Value scaled = mul(alphaT, ab);
  Value sum = b.create<tosa::AddOp>(loc, matrix, mul(betaT, c), scaled);
  Value zero = b.create<tosa::ConstOp>(
      loc, scalar, DenseElementsAttr::get(scalar, b.getF32FloatAttr(0)));
  // An ordered equality: a NaN beta reads C, as in the measured engine
  // (GenerateReusableGemm.cpp, `hasC = beta UNE 0`).
  Value guard = hasC ? b.create<tensor::FromElementsOp>(loc, scalar, hasC) : betaT;
  Value noC = b.create<tosa::EqualOp>(loc, flag, guard, zero);
  return b.create<tosa::SelectOp>(loc, matrix, noC, scaled, sum);
}

namespace {
constexpr llvm::StringLiteral kName = "forward_kernel";

/// GEMM's tile: (A: ?x?, B: ?x?, C: ?x?, M, K, N, alpha, beta) -> D: ?x?,
/// D = alpha * A.B + beta * C, as tosa ops, through tosa.matmul's batch of 1.
/// beta == 0 gives alpha * A.B whatever C holds (a NaN or nothing): the
/// accelerator then doesn't read C (sodap.read_unless_zero, D2). The shapes
/// reach the body only through M, K and N.
func::FuncOp buildGemm(OpBuilder &b, Location loc, bool sourceSemantics) {
  MLIRContext *ctx = b.getContext();
  Type f32 = b.getF32Type();
  int64_t dyn = ShapedType::kDynamic;
  auto matrix = RankedTensorType::get({dyn, dyn}, f32);
  auto batch = RankedTensorType::get({1, dyn, dyn}, f32);
  Type index = b.getIndexType();
  SmallVector<Type> inputs{matrix, matrix, matrix, index, index, index, f32, f32};
  if (sourceSemantics) inputs.push_back(f32);
  auto fn = b.create<func::FuncOp>(loc, kName, FunctionType::get(ctx, inputs, matrix));
  enum { A, B, C, M, K, N, Alpha, Beta, HasC };
  if (sourceSemantics) {
    fn->setAttr("sodap.descriptor_version", b.getI64IntegerAttr(GEMM_SOURCE_VERSION));
    fn.setArgAttr(HasC, "sodap.descriptor_field", b.getI64IntegerAttr(GEMM_FIELD_HAS_C));
  }
  // The descriptor's fields (ReusableGemmABI.h), argument by argument. C and
  // D are one matrix: the accelerator reads C and writes D in its place.
  int64_t fields[] = {GEMM_FIELD_A, GEMM_FIELD_B,     GEMM_FIELD_C,
                      GEMM_FIELD_M, GEMM_FIELD_K,     GEMM_FIELD_N,
                      GEMM_FIELD_ALPHA, GEMM_FIELD_BETA};
  for (auto [i, field] : llvm::enumerate(fields))
    fn.setArgAttr(i, "sodap.descriptor_field", b.getI64IntegerAttr(field));
  fn.setResultAttr(0, "sodap.descriptor_field",
                   b.getI64IntegerAttr(GEMM_FIELD_C));
  // Each matrix's row stride, in words (sodap-dataflow-copy-to-dma).
  fn.setArgAttr(A, "sodap.descriptor_ld", b.getI64IntegerAttr(GEMM_FIELD_LDA));
  fn.setArgAttr(B, "sodap.descriptor_ld", b.getI64IntegerAttr(GEMM_FIELD_LDB));
  fn.setArgAttr(C, "sodap.descriptor_ld", b.getI64IntegerAttr(GEMM_FIELD_LDC));
  fn.setResultAttr(0, "sodap.descriptor_ld",
                   b.getI64IntegerAttr(GEMM_FIELD_LDC));
  // The arguments holding each matrix's dimensions (sodap-dataflow-bind-dims).
  fn.setArgAttr(A, "sodap.dims", b.getDenseI64ArrayAttr({M, K}));
  fn.setArgAttr(B, "sodap.dims", b.getDenseI64ArrayAttr({K, N}));
  fn.setArgAttr(C, "sodap.dims", b.getDenseI64ArrayAttr({M, N}));
  fn.setResultAttr(0, "sodap.dims", b.getDenseI64ArrayAttr({M, N}));
  // C is read only when beta isn't 0 (sodap-dataflow-guard-reads).
  fn.setArgAttr(C, "sodap.read_unless_zero", b.getI64IntegerAttr(sourceSemantics ? HasC : Beta));

  Block *body = fn.addEntryBlock();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(body);
  auto arg = [&](int i) { return body->getArgument(i); };
  SmallVector<ReassociationIndices> batchOf = {{0, 1}, {2}};
  auto expand = [&](Value v, Value rows, Value cols) {
    return b.create<tensor::ExpandShapeOp>(
        loc, batch, v, batchOf,
        ArrayRef<OpFoldResult>{b.getIndexAttr(1), rows, cols});
  };
  Value d = buildGemmBody(b, loc, expand(arg(A), arg(M), arg(K)),
                          expand(arg(B), arg(K), arg(N)), arg(C), arg(Alpha),
                          arg(Beta), sourceSemantics ? arg(HasC) : Value());
  b.create<func::ReturnOp>(loc, d);
  return fn;
}

/// A GEMM launch's dimensions, M, K and N, from its static operand types.
FailureOr<SmallVector<int64_t>> gemmDims(dataflow::LaunchOp launch) {
  if ((launch.getInputs().size() != 5 && launch.getInputs().size() != 6) || launch.getNumResults() != 1)
    return launch.emitOpError("expected a gemm launch: A, B, C, alpha, beta "
                              "and one result");
  auto a = dyn_cast<RankedTensorType>(launch.getInputs()[0].getType());
  auto b = dyn_cast<RankedTensorType>(launch.getInputs()[1].getType());
  if (!a || !b || a.getRank() != 2 || b.getRank() != 2 ||
      !a.hasStaticShape() || !b.hasStaticShape() ||
      a.getDimSize(1) != b.getDimSize(0))
    return launch.emitOpError("expected static matrices MxK and KxN");
  auto c = dyn_cast<RankedTensorType>(launch.getInputs()[2].getType());
  if (!c || c != launch.getResult(0).getType() || c.getRank() != 2 ||
      c.getDimSize(0) != a.getDimSize(0) || c.getDimSize(1) != b.getDimSize(1))
    return launch.emitOpError("expected C and D MxN");
  for (Value s : launch.getInputs().drop_front(3))
    if (!s.getType().isF32())
      return launch.emitOpError("expected alpha and beta f32");
  return SmallVector<int64_t>{a.getDimSize(0), a.getDimSize(1),
                              b.getDimSize(1)};
}

struct JoinMultipleTiles
    : impl::JoinMultipleTilesBase<JoinMultipleTiles> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    llvm::MapVector<StringRef, SmallVector<ModuleOp>> groups;
    for (auto tile : module.getOps<ModuleOp>())
      if (auto op = tile->getAttrOfType<StringAttr>("sodap.operator"))
        groups[op.getValue()].push_back(tile);
    if (groups.empty())
      return;
    // ponytail: one operator, one tile named forward_kernel (Make, Bambu's
    // top, the ESP wrapper); name the tiles after their operator when a
    // second operator comes.
    if (groups.size() != 1 || groups.front().first != "gemm") {
      module.emitError("expected only gemm tiles");
      return signalPassFailure();
    }
    SmallVector<ModuleOp> &tiles = groups.front().second;
    bool sourceSemantics = tiles.front()->hasAttr("sodap.source_semantics");
    DenseSet<Attribute> names;
    for (ModuleOp tile : tiles) {
      if (tile->hasAttr("sodap.source_semantics") != sourceSemantics) {
        tile.emitError("cannot join operators with different FP semantics");
        return signalPassFailure();
      }
      names.insert(tile.getSymNameAttr());
      int matmuls = 0;
      tile.walk([&](Operation *op) {
        if (isa<tosa::MatMulOp>(op))
          ++matmuls;
      });
      if (matmuls != 1) {
        tile.emitError("expected one tosa.matmul in a gemm tile");
        return signalPassFailure();
      }
    }

    OpBuilder b(ctx);
    b.setInsertionPoint(tiles.front());
    Location loc = tiles.front().getLoc();
    auto joined = b.create<ModuleOp>(loc, kName);
    joined->setAttr("sodap.tile", b.getI64IntegerAttr(0));
    joined->setAttr("sodap.tile_dir", b.getStringAttr(("tiles/" + kName).str()));
    joined->setAttr("sodap.operator", b.getStringAttr("gemm"));
    if (sourceSemantics) joined->setAttr("sodap.source_semantics", b.getUnitAttr());
    b.setInsertionPointToStart(joined.getBody());
    buildGemm(b, loc, sourceSemantics);

    SmallVector<dataflow::LaunchOp> launches;
    module.walk([&](dataflow::LaunchOp l) {
      if (names.contains(l.getKernel().getRootReference()))
        launches.push_back(l);
    });
    SmallVector<Attribute> invocations;
    auto kernel =
        SymbolRefAttr::get(ctx, kName, {FlatSymbolRefAttr::get(ctx, kName)});
    for (dataflow::LaunchOp launch : launches) {
      FailureOr<SmallVector<int64_t>> dims = gemmDims(launch);
      if (failed(dims))
        return signalPassFailure();
      invocations.push_back(b.getDenseI64ArrayAttr(*dims));
      b.setInsertionPoint(launch);
      // (A, B, C, alpha, beta) -> (A, B, C, M, K, N, alpha, beta).
      SmallVector<Value> operands(launch.getInputs().take_front(3));
      for (int64_t d : *dims)
        operands.push_back(b.create<arith::ConstantIndexOp>(launch.getLoc(), d));
      llvm::append_range(operands, launch.getInputs().drop_front(3));
      auto joinedLaunch = b.create<dataflow::LaunchOp>(
          launch.getLoc(), launch.getResultTypes(), kernel, operands,
          ValueRange());
      launch.replaceAllUsesWith(joinedLaunch.getResults());
      launch.erase();
    }
    joined->setAttr("sodap.invocations", b.getArrayAttr(invocations));
    for (ModuleOp tile : tiles)
      tile.erase();
  }
};
} // namespace

} // namespace mlir::sodap
