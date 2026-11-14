// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Isolate invocations from an untiled, bufferized straight-line matmul graph.
// The pass has no tile/schedule options. Its output is shared by all hardware
// implementations of ReusableGemmABI.h.
#include "sodap/SODAPPasses.h"
#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include "sodap/ExecutionEngine/DmaTransport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace mlir::sodap {
#define GEN_PASS_DEF_REUSABLEGEMMTOESP
#include "sodap/SODAPPasses.h.inc"
namespace {
Value rootOf(Value value) {
  while (true) {
    if (auto op = value.getDefiningOp<memref::ExpandShapeOp>())
      value = op.getSrc();
    else if (auto op = value.getDefiningOp<memref::CollapseShapeOp>())
      value = op.getSrc();
    else if (auto op = value.getDefiningOp<memref::CastOp>())
      value = op.getSource();
    else
      return value;
  }
}

bool supportedBuffer(Type type) {
  auto mem = dyn_cast<MemRefType>(type);
  if (!mem || !mem.hasStaticShape() || !mem.getElementType().isF32() ||
      !mem.getLayout().isIdentity()) return false;
  uint64_t elements = 1;
  for (int64_t extent : mem.getShape()) {
    if (extent <= 0 || uint64_t(extent) > (1u << 27) / elements) return false;
    elements *= extent;
  }
  return true;
}

struct Invocation {
  Value a, b, c;
  int64_t m, n, k;
  Value alpha, beta, prior;
};

class ReusableGemmToESP : public impl::ReusableGemmToESPBase<ReusableGemmToESP> {
public:
  using ReusableGemmToESPBase::ReusableGemmToESPBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();
    if(dmaDataWidth && !::sodap::DmaTransport{dmaDataWidth}.valid()) {
      module.emitError("dma-data-width must be 64, 128, 256 or 512");
      return signalPassFailure();
    }
    auto function = module.lookupSymbol<func::FuncOp>(kernel);
    if (!function || function.isExternal() ||
        !llvm::hasSingleElement(function.getBody()) || function.getNumResults()) {
      module.emitError("reusable GEMM requires a single-block function with output arguments; canonicalize first");
      return signalPassFailure();
    }
    if (instance < 0) {
      function.emitError("instance must be nonnegative");
      return signalPassFailure();
    }
    for (Value arg : function.getArguments())
      if (!supportedBuffer(arg.getType())) {
        function.emitError("initial GEMM isolation requires static contiguous f32 arguments");
        return signalPassFailure();
      }

    SmallVector<Invocation> calls;
    SmallVector<std::pair<Value, Value>> outputs;
    llvm::SetVector<Value> buffers;
    llvm::SmallDenseSet<Value> filled, produced, readArguments, returned;
    bool sawCopy = false;
    struct ScaledInput { Value tensor, coefficient; };
    DenseMap<Value,ScaledInput> scaledInputs;
    llvm::SmallDenseSet<Value> consumed;
    auto producer = [&](Value v) -> Invocation * {
      for (auto &call : calls) if (call.c == v) return &call;
      return nullptr;
    };
    auto renameResult = [&](Invocation &call, Value result) {
      buffers.remove(call.c); produced.erase(call.c);
      call.c=result; buffers.insert(result); produced.insert(result);
    };
    // Preflight the complete graph before rewriting any operation.
    for (Operation &op : function.front()) {
      if (isa<arith::ConstantOp, func::ReturnOp>(op))
        continue;
      if (isa<memref::AllocOp, memref::ExpandShapeOp,
              memref::CollapseShapeOp, memref::CastOp>(op)) {
        if (op.getNumResults() != 1 || !supportedBuffer(op.getResult(0).getType())) {
          op.emitError("expected a static contiguous f32 buffer or full reshape");
          return signalPassFailure();
        }
        continue;
      }
      if (isa<memref::DeallocOp>(op))
        continue; // Arena lifetime covers the entire graph.
      if (auto fill = dyn_cast<linalg::FillOp>(op)) {
        Value root = rootOf(fill.getOutputs()[0]);
        auto scalar = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
        auto value = scalar ? dyn_cast<FloatAttr>(scalar.getValue()) : FloatAttr();
        if (!value || !value.getValue().isPosZero() ||
            produced.contains(root) || filled.contains(root)) {
          op.emitError("reusable GEMM supports one zero initialization per output");
          return signalPassFailure();
        }
        filled.insert(root);
        continue;
      }
      if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
        auto matmul = cast<linalg::LinalgOp>(&op);
        Value av = matmul.getDpsInputs()[0], bv = matmul.getDpsInputs()[1];
        Value cv = matmul.getDpsInits()[0];
        auto a = cast<MemRefType>(av.getType());
        auto b = cast<MemRefType>(bv.getType());
        auto c = cast<MemRefType>(cv.getType());
        unsigned rank = isa<linalg::BatchMatmulOp>(op) ? 3 : 2;
        if (sawCopy || a.getRank() != rank || b.getRank() != rank ||
            c.getRank() != rank || !supportedBuffer(a) || !supportedBuffer(b) ||
            !supportedBuffer(c) || (rank == 3 &&
                (a.getDimSize(0) != 1 || b.getDimSize(0) != 1 || c.getDimSize(0) != 1))) {
          op.emitError("expected matmul or batch-one matmul before terminal output copies");
          return signalPassFailure();
        }
        Invocation call{rootOf(av), rootOf(bv), rootOf(cv),
                        a.getDimSize(rank - 2), b.getDimSize(rank - 1),
                        a.getDimSize(rank - 1), {}, {}, {}};
        if (b.getDimSize(rank - 2) != call.k ||
            c.getDimSize(rank - 2) != call.m || c.getDimSize(rank - 1) != call.n ||
            call.a == call.c || call.b == call.c || !filled.erase(call.c) ||
            produced.contains(call.c)) {
          op.emitError("expected compatible GEMM shapes and a distinct zero-initialized output");
          return signalPassFailure();
        }
        if(dmaDataWidth) {
          for(auto pair : {std::make_pair(call.a,call.k),std::make_pair(call.b,call.n),std::make_pair(call.c,call.n)}) {
            auto rootType=cast<MemRefType>(pair.first.getType());
            if(!rootType.getRank() || rootType.getShape().back()!=pair.second) {
              op.emitError("packed GEMM reshape must preserve the innermost matrix dimension");
              return signalPassFailure();
            }
          }
        }
        for (Value input : {call.a, call.b}) {
          if (!produced.contains(input)) {
            if (!isa<BlockArgument>(input) || filled.contains(input)) {
              op.emitError("GEMM input must be a function argument or a completed GEMM result");
              return signalPassFailure();
            }
            readArguments.insert(input);
          }
          if (produced.contains(input)) consumed.insert(input);
          buffers.insert(input);
        }
        buffers.insert(call.c);
        produced.insert(call.c);
        calls.push_back(call);
        continue;
      }
      // Recognize canonical elementwise GEMM epilogues without changing
      // reduction order or moving a scale across a contraction. Coefficients
      // are runtime scalar arguments, never hardware tile parameters.
      if (auto generic=dyn_cast<linalg::GenericOp>(op)) {
        auto reject=[&]() { generic.emitError("unsupported reusable GEMM epilogue or shared intermediate"); signalPassFailure(); };
        if (sawCopy || generic.getNumDpsInputs()!=2 || generic.getNumDpsInits()!=1 ||
            generic.getNumReductionLoops()!=0 || generic.getBody()->getOperations().size()!=2) {
          reject(); return;
        }
        Block &body=*generic.getBody();
        auto yield=dyn_cast<linalg::YieldOp>(body.back());
        Operation &arithmetic=body.front();
        if (!yield || yield.getValues().size()!=1 || arithmetic.getNumResults()!=1 ||
            yield.getValues()[0]!=arithmetic.getResult(0) || arithmetic.getNumOperands()!=2 ||
            !((arithmetic.getOperand(0)==body.getArgument(0) && arithmetic.getOperand(1)==body.getArgument(1)) ||
              (arithmetic.getOperand(1)==body.getArgument(0) && arithmetic.getOperand(0)==body.getArgument(1)))) {
          reject();return;
        }
        Value x=rootOf(generic.getDpsInputs()[0]), y=rootOf(generic.getDpsInputs()[1]);
        Value result=rootOf(generic.getDpsInits()[0]);
        auto maps=generic.getIndexingMapsArray();
        if (!maps[2].isIdentity() || produced.contains(result) || filled.contains(result)) {reject();return;}
        if (auto mul=dyn_cast<arith::MulFOp>(arithmetic)) {
          unsigned scalar=cast<MemRefType>(x.getType()).getRank()==0 ? 0 : 1;
          Value coefficient=scalar==0 ? x:y, tensor=scalar==0 ? y:x;
          if (mul.getFastmath()!=arith::FastMathFlags::none || !isa<BlockArgument>(coefficient) ||
              cast<MemRefType>(coefficient.getType()).getRank()!=0 || maps[scalar].getNumResults()!=0 ||
              !maps[1-scalar].isIdentity() ||
              cast<MemRefType>(tensor.getType()).getNumElements()!=cast<MemRefType>(result.getType()).getNumElements()) {reject();return;}
          if (auto *call=producer(tensor)) {
            if (call->alpha || call->prior || consumed.contains(tensor) || isa<BlockArgument>(tensor)) {reject();return;}
            call->alpha=coefficient; renameResult(*call,result);
          } else {
            if (!isa<BlockArgument>(tensor) || scaledInputs.count(result)) {reject();return;}
            scaledInputs[result]={tensor,coefficient};
          }
          continue;
        }
        if (auto add=dyn_cast<arith::AddFOp>(arithmetic)) {
          if (add.getFastmath()!=arith::FastMathFlags::none || !maps[0].isIdentity() || !maps[1].isIdentity()) {reject();return;}
          Invocation *call=producer(x);Value other=y;
          if (!call) {call=producer(y);other=x;}
          if (!call || call->prior || consumed.contains(call->c) || isa<BlockArgument>(call->c)) {reject();return;}
          auto it=scaledInputs.find(other);
          Value prior=it==scaledInputs.end() ? other:it->second.tensor;
          Value beta=it==scaledInputs.end() ? Value():it->second.coefficient;
          if (!isa<BlockArgument>(prior) || producer(other) ||
              cast<MemRefType>(prior.getType()).getNumElements()!=call->m*call->n ||
              cast<MemRefType>(result.getType()).getNumElements()!=call->m*call->n) {reject();return;}
          call->prior=prior;call->beta=beta;
          readArguments.insert(prior);
          if (it!=scaledInputs.end()) scaledInputs.erase(it);
          renameResult(*call,result);
          continue;
        }
        reject();return;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        sawCopy = true;
        Value source = rootOf(copy.getSource()), dest = rootOf(copy.getTarget());
        if (!produced.contains(source) || !isa<BlockArgument>(dest) ||
            readArguments.contains(dest) || produced.contains(dest) ||
            !returned.insert(dest).second ||
            cast<MemRefType>(source.getType()).getNumElements() !=
                cast<MemRefType>(dest.getType()).getNumElements()) {
          op.emitError("only terminal copies of GEMM results to distinct output arguments are supported");
          return signalPassFailure();
        }
        outputs.push_back({source, dest});
        continue;
      }
      op.emitError("unsupported operation in reusable GEMM graph");
      return signalPassFailure();
    }
    if (calls.empty() || !filled.empty() || !scaledInputs.empty()) {
      function.emitError("expected GEMM invocations and no unused zero fills");
      return signalPassFailure();
    }
    for (Value arg : function.getArguments())
      if (produced.contains(arg))
        outputs.push_back({arg, arg});

    DenseMap<Value, int64_t> offset;
    int64_t bytes = GEMM_HEADER_BYTES;
    for (Value buffer : buffers) {
      auto type=cast<MemRefType>(buffer.getType());
      uint64_t words=type.getNumElements()*GEMM_WORD_STRIDE;
      if(dmaDataWidth && !::sodap::DmaTransport{dmaDataWidth}.tensorWords(type.getShape().data(),type.getRank(),words)) {
        function.emitError("invalid packed GEMM layout");return signalPassFailure();
      }
      if (words > uint64_t((1 << 30) - bytes) / 4) {
        function.emitError("GEMM arena exceeds the one-GiB interface limit");
        return signalPassFailure();
      }
      offset[buffer] = bytes;
      bytes += words * 4;
    }

    // Export the same preflighted invocation graph used by the generated host.
    // This also lets external backends execute an identical workload without
    // reconstructing a benchmark-specific sequence in their host application.
    if (!report.empty()) {
      DenseMap<Value, int64_t> ids;
      llvm::json::Array tensors, launches, inputs, results;
      for (auto [id, buffer] : llvm::enumerate(buffers)) {
        ids[buffer] = id;
        llvm::json::Array shape;
        for (int64_t dim : cast<MemRefType>(buffer.getType()).getShape())
          shape.push_back(dim);
        tensors.push_back(llvm::json::Object{{"id", int64_t(id)},
            {"shape", std::move(shape)}, {"offset_bytes", offset[buffer]}});
      }
      auto argIndex = [](Value arg) -> int64_t {
        return cast<BlockArgument>(arg).getArgNumber();
      };
      for (Value arg : function.getArguments())
        if (readArguments.contains(arg) && buffers.contains(arg))
          inputs.push_back(llvm::json::Object{{"buffer", ids[arg]}, {"argument", argIndex(arg)}});
      for (const auto &call : calls) {
        llvm::json::Object launch{{"a", ids[call.a]}, {"b", ids[call.b]},
            {"c", ids[call.c]}, {"m", call.m}, {"n", call.n}, {"k", call.k},
            {"alpha_constant", 1.0}, {"beta_constant", call.prior ? 1.0 : 0.0}};
        if (call.alpha) launch["alpha_argument"] = argIndex(call.alpha);
        if (call.beta) launch["beta_argument"] = argIndex(call.beta);
        launches.push_back(std::move(launch));
        if (call.prior)
          inputs.push_back(llvm::json::Object{{"buffer", ids[call.c]}, {"argument", argIndex(call.prior)}});
      }
      for (auto [src, dst] : outputs)
        results.push_back(llvm::json::Object{{"buffer", ids[src]}, {"argument", argIndex(dst)}});
      std::error_code ec;
      llvm::raw_fd_ostream stream(report, ec, llvm::sys::fs::OF_Text);
      if (ec) { function.emitError("cannot write invocation manifest: ") << ec.message(); return signalPassFailure(); }
      stream << llvm::json::Value(llvm::json::Object{{"schema_version", 1},
          {"kernel", kernel.getValue()}, {"instance", instance.getValue()},
          {"dma_data_width", int64_t(dmaDataWidth)}, {"arena_bytes", bytes},
          {"buffers", std::move(tensors)}, {"invocations", std::move(launches)},
          {"initializations", std::move(inputs)}, {"outputs", std::move(results)}}) << '\n';
    }

    OpBuilder builder(module.getContext());
    Location loc = function.getLoc();
    auto i64 = builder.getI64Type();
    auto f32 = builder.getF32Type();
    auto unranked = UnrankedMemRefType::get(f32, 0);
    auto declare = [&](StringRef name, TypeRange inputs, TypeRange results) {
      OpBuilder::InsertionGuard guard(builder);
      if (module.lookupSymbol(name))
        return;
      builder.setInsertionPointToStart(module.getBody());
      builder.create<func::FuncOp>(loc, name,
          builder.getFunctionType(inputs, results)).setPrivate();
    };
    StringRef openName=dmaDataWidth?"esp_gemm_open_packed":"esp_gemm_open";
    declare(openName,SmallVector<Type>(dmaDataWidth?3:2,i64),{i64});
    declare("esp_gemm_close", {i64}, {});
    declare("esp_gemm_copy_in", {i64, i64, unranked}, {});
    declare("esp_gemm_copy_out", {i64, i64, unranked}, {});
    SmallVector<Type> invokeTypes(10, i64);
    invokeTypes.append({f32, f32});
    declare("esp_gemm_invoke", invokeTypes, {});

    // Retain host arguments, but no host allocation for intermediate results.
    function.getBody().dropAllReferences();
    function.front().clear();
    builder.setInsertionPointToStart(&function.front());
    auto constant = [&](int64_t n) -> Value {
      return builder.create<arith::ConstantIntOp>(loc, n, 64);
    };
    SmallVector<Value> openArgs{constant(bytes),constant(instance)};
    if(dmaDataWidth)openArgs.push_back(constant(dmaDataWidth));
    Value context=builder.create<func::CallOp>(loc,openName,TypeRange{i64},openArgs).getResult(0);
    auto ld=[&](int64_t cols) {
      uint64_t words=cols*GEMM_WORD_STRIDE;
      if(dmaDataWidth)(void)::sodap::DmaTransport{dmaDataWidth}.rowWords(cols,words);
      return constant(words*4);
    };
    for (Value arg : function.getArguments()) {
      if (!readArguments.contains(arg) || !buffers.contains(arg)) continue;
      Value cast = builder.create<memref::CastOp>(loc, unranked, arg);
      builder.create<func::CallOp>(loc, "esp_gemm_copy_in", TypeRange{},
          ValueRange{context, constant(offset[arg]), cast});
    }
    Value one = builder.create<arith::ConstantFloatOp>(loc, APFloat(1.0f), f32);
    Value zero = builder.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), f32);
    // Initialize accumulating outputs from the original input once, before
    // the sequence. Intermediates produced by hardware never round-trip to CPU.
    for (auto &call : calls) if (call.prior) {
      Value cast=builder.create<memref::CastOp>(loc,unranked,call.prior);
      builder.create<func::CallOp>(loc,"esp_gemm_copy_in",TypeRange{},
          ValueRange{context,constant(offset[call.c]),cast});
    }
    auto coefficient = [&](Value arg, Value fallback) -> Value {
      return arg ? Value(builder.create<memref::LoadOp>(loc,arg,ValueRange{})) : fallback;
    };
    for (auto [index, call] : llvm::enumerate(calls)) {
      auto invoke = builder.create<func::CallOp>(loc, "esp_gemm_invoke", TypeRange{},
          ValueRange{context, constant(offset[call.a]), constant(offset[call.b]),
            constant(offset[call.c]), constant(call.m), constant(call.n), constant(call.k),
            ld(call.k),ld(call.n),ld(call.n), coefficient(call.alpha,one),
            call.prior ? coefficient(call.beta,one) : zero});
      invoke->setAttr("sodap.operator", builder.getStringAttr(dmaDataWidth?"gemm_f32_v2":"gemm_f32_v1"));
      invoke->setAttr("sodap.invocation", builder.getI64IntegerAttr(index));
      invoke->setAttr("sodap.instance", builder.getI64IntegerAttr(instance));
    }
    for (auto [source, destination] : outputs) {
      Value cast = builder.create<memref::CastOp>(loc, unranked, destination);
      builder.create<func::CallOp>(loc, "esp_gemm_copy_out", TypeRange{},
          ValueRange{context, constant(offset[source]), cast});
    }
    builder.create<func::CallOp>(loc, "esp_gemm_close", TypeRange{}, ValueRange{context});
    builder.create<func::ReturnOp>(loc);
    function->setAttr("sodap.gemm_abi", builder.getI32IntegerAttr(dmaDataWidth ? GEMM_PACKED_VERSION : GEMM_VERSION));
    function->setAttr("sodap.gemm_invocations", builder.getI64IntegerAttr(calls.size()));
  }
};
} // namespace
} // namespace mlir::sodap
