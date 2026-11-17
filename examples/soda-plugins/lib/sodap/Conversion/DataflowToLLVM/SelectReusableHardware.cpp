// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/Conversion/DataflowToLLVM/Passes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <tuple>

namespace mlir::sodap {
#define GEN_PASS_DEF_SELECTREUSABLEHARDWARE
#include "sodap/Conversion/DataflowToLLVM/Passes.h.inc"
}
using namespace mlir;
namespace {
struct SelectReusableHardware
    : sodap::impl::SelectReusableHardwareBase<SelectReusableHardware> {
  using Base::Base;
  void runOnOperation() override {
    if (!lanes || lanes > 256 || !plmBytes) {
      getOperation().emitError("positive storage and 1..256 compute lanes required");
      return signalPassFailure();
    }
    unsigned mr = 1, nr = lanes;
    for (unsigned r = 1; r * r <= lanes; ++r)
      if (lanes % r == 0) { mr = r; nr = lanes / r; }
    // Port of the existing bounded policy. No invocation shapes, model names,
    // measured results or generated code enter selection. The final banking
    // report checks actual storage; this is a conservative proposal estimate.
    int64_t bestM = 0, bestN = 0, bestK = 0, bestBytes = 0;
    for (int64_t m : {8, 16, 32, 64, 128})
      for (int64_t n : {8, 16, 32, 64, 128})
        for (int64_t k : {8, 16, 32, 64, 128}) {
          int64_t bytes = 4 * (2*m*k + 2*k*n + 5*m*n);
          if (m % mr || n % nr || uint64_t(bytes + 4096) > plmBytes)
            continue;
          // Exact rational comparison of m*n/(m+n), then K, smaller storage,
          // then lexicographic dimensions, matching the previous policy.
          int64_t lhs = m*n*(bestM+bestN), rhs = bestM*bestN*(m+n);
          if (!bestM || lhs > rhs ||
              (lhs == rhs && std::make_tuple(k,-bytes,m,n) >
                             std::make_tuple(bestK,-bestBytes,bestM,bestN))) {
            bestM=m; bestN=n; bestK=k; bestBytes=bytes;
          }
        }
    if (!bestM) {
      getOperation().emitError("storage budget cannot service requested lanes");
      return signalPassFailure();
    }
    OpBuilder b(&getContext());
    unsigned count = 0;
    getOperation().walk([&](ModuleOp tile) {
      auto op = tile->getAttrOfType<StringAttr>("sodap.operator");
      if (!op || op.getValue() != "gemm") return;
      tile->setAttr("sodap.hardware_tile_sizes", b.getArrayAttr({b.getArrayAttr({
          b.getI64IntegerAttr(bestM), b.getI64IntegerAttr(bestN),
          b.getI64IntegerAttr(bestK)})}));
      tile->setAttr("sodap.compute_lanes", b.getI64IntegerAttr(lanes));
      ++count;
    });
    if (!count) {
      getOperation().emitError("expected a model-derived GEMM operator tile");
      return signalPassFailure();
    }
    if (!report.empty()) {
      std::error_code error;
      llvm::raw_fd_ostream out(report, error, llvm::sys::fs::OF_Text);
      if (error) { getOperation().emitError(error.message()); return signalPassFailure(); }
      out << llvm::json::Value(llvm::json::Object{
          {"tile_m",bestM}, {"tile_n",bestN}, {"tile_k",bestK},
          {"logical_bytes",bestBytes}, {"plm_bytes",uint64_t(plmBytes)},
          {"lanes",unsigned(lanes)}, {"register_rows",mr}, {"register_columns",nr},
          {"policy","balanced_operand_reuse_then_reduction_burst"},
          {"implementation","sodap-select-reusable-hardware"},
          {"domain","Runtime shapes and tails independent of these local extents"}})
          << "\n";
    }
  }
};
} // namespace
