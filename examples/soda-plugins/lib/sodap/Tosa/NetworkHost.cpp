// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Host lowering from the actual TOSA SSA graph. No layer-name recognition or
// Python graph representation. Hardware lowering consumes source.mlir; runtime
// calls below implement the execution ABI, not accelerator microarchitecture.
#include "sodap/Tosa/Passes.h"
#include "sodap/Tosa/Gemm.h"
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cfloat>
#include <numeric>
#include <cctype>

namespace mlir::sodap {
#define GEN_PASS_DEF_TOSANETWORKHOST
#define GEN_PASS_DEF_TOSAEXTRACTHARDWARE
#include "sodap/Tosa/Passes.h.inc"
namespace {
struct ExtractHardware : impl::TosaExtractHardwareBase<ExtractHardware> {
  void runOnOperation() override {
    auto module=getOperation();
    auto hardware=[](Operation &op) {return isa<ModuleOp>(op) && op.hasAttr("sodap.tile");};
    if(!llvm::any_of(*module.getBody(),hardware)) {
      module.emitError("no source-outlined hardware modules to extract");
      return signalPassFailure();
    }
    for(auto &op:llvm::make_early_inc_range(*module.getBody()))
      if(!hardware(op))op.erase();
  }
};
struct Buffer {
  int64_t offset = 0, bytes = 0, elements = 0, columns = 0;
  int64_t parameter = -1;
  unsigned id = 0;
};
int64_t product(ArrayRef<int64_t> a) {
  return std::accumulate(a.begin(), a.end(), int64_t(1), std::multiplies<int64_t>());
}
SmallVector<int64_t> shape(Value v) {
  return llvm::to_vector(cast<RankedTensorType>(v.getType()).getShape());
}
SmallVector<int64_t> ints(Operation *op, StringRef key) {
  auto a = op->getAttrOfType<DenseI64ArrayAttr>(key);
  return a ? llvm::to_vector(a.asArrayRef()) : SmallVector<int64_t>{};
}
bool relu(Operation *op) {
  auto clamp = dyn_cast<tosa::ClampOp>(op);
  return clamp && clamp.getMinFp().convertToFloat() == 0.0f &&
         clamp.getMaxFp().convertToFloat() == FLT_MAX;
}
StringRef family(Operation *op) {
  if (isa<tosa::Conv2DOp>(op)) return "conv";
  if (isa<tosa::MatMulOp>(op)) return "gemm";
  if (isa<tosa::AddOp>(op)) return "add";
  return "";
}
llvm::json::Array jsonInts(ArrayRef<int64_t> a) {
  llvm::json::Array r; for (auto v : a) r.push_back(v); return r;
}

struct NetworkHost : impl::TosaNetworkHostBase<NetworkHost> {
  using Base::Base;
  DenseMap<Value, Buffer> buffers;
  DenseMap<Value, Value> aliases;
  DenseSet<Operation *> fused;
  DenseMap<Operation *, bool> activations;
  DenseMap<Operation *, Gemm> gemms;
  DenseSet<Operation *> consumeC, prepareC;
  DenseMap<Value, unsigned> lastUse;
  SmallVector<std::pair<int64_t, int64_t>> free;
  int64_t cursor = 0, parameters = 0;
  unsigned nextID = 0;
  ModuleOp module;
  OpBuilder *builder = nullptr;
  LocationAttr loc;
  llvm::json::Array records, storage;
  struct Engine {std::string family;unsigned lanes;SmallVector<unsigned> instances;};
  std::map<std::string,Engine> engines;

  StringRef operatorFamily(Operation *op) {
    return gemms.count(op) ? "gemm" : family(op);
  }
  Operation *binding(Operation *op) {
    auto it=gemms.find(op);
    return it==gemms.end()?op:it->second.matmul.getOperation();
  }
  SmallVector<Value> operands(Operation *op) {
    auto it=gemms.find(op);
    if(it==gemms.end()) {
      auto values=op->getOperands();
      return llvm::to_vector(isa<tosa::TransposeOp>(op)?values.take_front(1):values);
    }
    auto &g=it->second;
    SmallVector<Value> values{g.matmul.getA(),g.matmul.getB()};
    for(Value v:{g.c,g.alpha,g.beta})if(v)values.push_back(v);
    return values;
  }

  Value storageValue(Value v) {
    auto it = aliases.find(v);
    return it == aliases.end() ? v : storageValue(it->second);
  }
  bool sameLayout(Value a, Value b) {
    auto x=shape(a),y=shape(b);
    return product(x)==product(y) && (x.empty()?1:x.back())==(y.empty()?1:y.back());
  }
  Buffer &buffer(Value v) { return buffers.find(storageValue(v))->second; }
  LogicalResult type(Value v) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || !t.hasStaticShape() || !t.getElementType().isF32() ||
        t.getRank() > 4 ||
        llvm::any_of(t.getShape(), [](int64_t d) { return d <= 0; }))
      return emitError(v.getLoc(), "network tensors must be positive static FP32, rank 0..4");
    // Check before multiplication, including runtime's 32-bit address bound.
    int64_t count = 1;
    for (auto d : t.getShape()) {
      if (d > (1 << 27) / count)
        return emitError(v.getLoc(), "network tensor exceeds the runtime limit");
      count *= d;
    }
    return success();
  }
  Buffer allocate(Value v, bool persistent) {
    auto s = shape(v);
    Buffer b;
    b.id = nextID++; b.elements = product(s); b.columns = s.empty()?1:s.back();
    int64_t beat = dmaWidth / 8;
    int64_t row = ((b.columns * 4 + beat - 1) / beat) * beat;
    b.bytes = b.elements / b.columns * row;
    auto best = free.end();
    if (!persistent)
      for (auto it = free.begin(); it != free.end(); ++it)
        if (it->second >= b.bytes && (best == free.end() || it->second < best->second)) best = it;
    if (best != free.end()) {
      b.offset = best->first;
      best->first += b.bytes; best->second -= b.bytes;
      if (!best->second) free.erase(best);
    } else {
      b.offset = cursor; cursor += b.bytes;
    }
    buffers[v] = b;
    storage.push_back(llvm::json::Object{{"id", b.id}, {"offset", b.offset},
      {"bytes", b.bytes}, {"elements", b.elements}, {"columns", b.columns},
      {"shape", jsonInts(s)}, {"persistent", persistent}});
    return b;
  }
  void release(Buffer b) {
    free.push_back({b.offset, b.bytes}); llvm::sort(free);
    SmallVector<std::pair<int64_t, int64_t>> merged;
    for (auto item : free) {
      if (!merged.empty() && merged.back().first + merged.back().second == item.first)
        merged.back().second += item.second;
      else merged.push_back(item);
    }
    free = std::move(merged);
  }
  Value constant(int64_t n) { return builder->create<arith::ConstantIntOp>(loc, n, 64); }
  func::CallOp call(StringRef name, ValueRange args, TypeRange results = {}) {
    if (!module.lookupSymbol<func::FuncOp>(name)) {
      OpBuilder::InsertionGuard guard(*builder);
      builder->setInsertionPointToStart(module.getBody());
      auto fn = builder->create<func::FuncOp>(loc, name,
          builder->getFunctionType(args.getTypes(), results));
      fn.setPrivate();
    }
    return builder->create<func::CallOp>(loc, name, results, args);
  }
  func::CallOp invoke(StringRef name, Value context, ArrayRef<int64_t> args) {
    SmallVector<Value> values{context};
    for (auto n : args) values.push_back(constant(n));
    return call(name, values);
  }
  void reg(Value ctx, Buffer b) {
    invoke("esp_op_tensor", ctx, {b.offset, b.elements, b.columns});
  }
  void copy(Value ctx, Buffer b, Value memref, bool in) {
    auto view = builder->create<memref::CastOp>(loc,
        UnrankedMemRefType::get(builder->getF32Type(), 0), memref);
    call(in ? "esp_op_copy_in" : "esp_op_copy_out", {ctx, constant(b.offset), view});
  }
  // All arguments are derived from the verified operation and its SSA buffers.
  // Return a CPU/runtime operation or a typed accelerator invocation.
  FailureOr<std::pair<std::string, SmallVector<int64_t>>> arguments(Operation *op) {
    if(auto it=gemms.find(op);it!=gemms.end()) {
      auto &g=it->second;
      auto a=shape(g.matmul.getA()),b=shape(g.matmul.getB());
      auto result=shape(g.matmul.getResult());
      if(a.size()!=3 || b.size()!=3 || a[0]!=1 || b[0]!=1 || a[2]!=b[1] ||
          result!=SmallVector<int64_t>{1,a[1],b[2]})
        return op->emitError("reusable GEMM currently requires batch one");
      return std::make_pair(std::string(g.c || g.alpha ? "gemm_full_source":"gemm_source"),
          SmallVector<int64_t>{buffer(g.matmul.getA()).offset,buffer(g.matmul.getB()).offset,
            buffer(op->getResult(0)).offset,a[1],b[2],a[2]});
    }
    auto y = op->getResult(0); auto out = shape(y);
    auto x = op->getOperand(0); auto in = shape(x);
    int64_t a = buffer(x).offset, c = buffer(y).offset;
    SmallVector<int64_t> args;
    std::string kind;
    if (isa<tosa::Conv2DOp>(op)) {
      auto w = shape(op->getOperand(1)), bias = shape(op->getOperand(2));
      auto pad = ints(op,"pad"), stride = ints(op,"stride"), dilation = ints(op,"dilation");
      if (in.size()!=4 || w.size()!=4 || out.size()!=4 || pad.size()!=4 || stride.size()!=2 || dilation.size()!=2 ||
          llvm::any_of(pad, [](int64_t v){return v<0;}) ||
          llvm::any_of(stride, [](int64_t v){return v<=0;}) ||
          llvm::any_of(dilation, [](int64_t v){return v<=0;}) ||
          bias != SmallVector<int64_t>{w[0]} || in[0]!=out[0] || in[3]!=w[3] || out[3]!=w[0])
        return op->emitError("unsupported direct convolution contract");
      for (unsigned d=0;d<2;++d) {
        int64_t span = in[d+1]+pad[2*d]+pad[2*d+1]-(w[d+1]-1)*dilation[d]-1;
        if (span<0 || out[d+1]!=span/stride[d]+1)
          return op->emitError("convolution output shape disagrees with geometry");
      }
      kind="conv";
      args={a,buffer(op->getOperand(1)).offset,buffer(op->getOperand(2)).offset,c,
        in[0],in[1],in[2],in[3],w[0],w[1],w[2],out[1],out[2],
        stride[0],stride[1],pad[0],pad[2],dilation[0],dilation[1],activations.lookup(op)};
    } else if (isa<tosa::MatMulOp>(op)) {
      auto rhs=shape(op->getOperand(1));
      if (in.size()!=3 || rhs.size()!=3 || in[0]!=1 || rhs[0]!=1 || in[2]!=rhs[1] ||
          out!=SmallVector<int64_t>{1,in[1],rhs[2]})
        return op->emitError("reusable GEMM currently requires batch one");
      kind="gemm_source"; args={a,buffer(op->getOperand(1)).offset,c,in[1],rhs[2],in[2]};
    } else if (isa<tosa::AddOp>(op)) {
      if (in != out || shape(op->getOperand(1))!=out)
        return op->emitError("addition offload requires equal shapes; normalize broadcasting first");
      kind="add"; args={a,buffer(op->getOperand(1)).offset,c,product(out),activations.lookup(op)};
    } else if (isa<tosa::ReshapeOp>(op)) {
      kind="repack"; args={a,c,product(out)};
    } else if (auto transpose=dyn_cast<tosa::TransposeOp>(op)) {
      SmallVector<int64_t> perm;
      if (failed(transpose.getConstantPerms(perm))) return op->emitError("expected constant transpose permutation");
      kind="transpose"; args={a,c,int64_t(in.size())};
      args.append(in); args.append(4-in.size(),1);
      args.append(perm); for (unsigned i=perm.size();i<4;++i) args.push_back(i);
    } else if (isa<tosa::MaxPool2dOp,tosa::AvgPool2dOp>(op)) {
      auto k=ints(op,"kernel"), s=ints(op,"stride"), p=ints(op,"pad");
      bool avg=isa<tosa::AvgPool2dOp>(op);
      if (in.size()!=4 || out.size()!=4 || k.size()!=2 || s.size()!=2 || p.size()!=4 ||
          llvm::any_of(k,[](int64_t v){return v<=0;}) || llvm::any_of(s,[](int64_t v){return v<=0;}) ||
          llvm::any_of(p,[](int64_t v){return v<0;}) || (avg && llvm::any_of(p,[](int64_t v){return v!=0;})) ||
          in[0]!=out[0] || in[3]!=out[3]) return op->emitError("unsupported pooling geometry/divisor");
      for(unsigned d=0;d<2;++d)
        if(in[d+1]+p[2*d]+p[2*d+1]<k[d] || out[d+1]!=(in[d+1]+p[2*d]+p[2*d+1]-k[d])/s[d]+1)
          return op->emitError("pooling output shape disagrees with geometry");
      kind="pool"; args={a,c,in[0],in[1],in[2],in[3],out[1],out[2],k[0],k[1],s[0],s[1],p[0],p[2],avg};
    } else if (isa<tosa::ConcatOp>(op)) {
      if(op->getNumOperands()!=2) return op->emitError("concat must be normalized to binary concatenations");
      int64_t axis=op->getAttrOfType<IntegerAttr>("axis").getInt();
      auto rhs=shape(op->getOperand(1));
      if(axis<0 || axis>=int64_t(out.size()) || in.size()!=out.size() || rhs.size()!=out.size())
        return op->emitError("invalid concatenation ranks/axis");
      for(unsigned i=0;i<out.size();++i)
        if(out[i]!=(i==axis?in[i]+rhs[i]:in[i]) || (i!=axis && rhs[i]!=in[i]))
          return op->emitError("inconsistent concatenation shapes");
      kind="concat"; args={a,buffer(op->getOperand(1)).offset,c,
        product(ArrayRef<int64_t>(in).take_front(axis)),in[axis],rhs[axis],
        product(ArrayRef<int64_t>(in).drop_front(axis+1))};
    } else if(relu(op)) {
      kind="relu"; args={a,c,product(out)};
    } else return op->emitError("operation has no ESP network lowering");
    return std::make_pair(kind,args);
  }
  void runOnOperation() override {
    module=getOperation(); loc=module.getLoc();
    if (dmaWidth!=64) {
      module.emitError("network backend currently qualified for DMA width 64 only");
      return signalPassFailure();
    }
    auto functions=llvm::to_vector(module.getOps<func::FuncOp>());
    if(functions.size()!=1 || functions[0].isExternal() || !llvm::hasSingleElement(functions[0].getBody())) {
      module.emitError("expected one single-block network function"); return signalPassFailure();
    }
    auto source=functions[0]; auto ret=dyn_cast<func::ReturnOp>(source.front().getTerminator());
    if(!ret || ret.getNumOperands()!=1 || source.getNumArguments()==0) {
      source.emitError("expected tensor inputs and one tensor result"); return signalPassFailure();
    }
    SmallVector<Operation *> operations;
    SmallVector<tosa::ConstOp> weights;
    for (auto arg:source.getArguments()) if(failed(type(arg))) return signalPassFailure();
    for(auto &op:source.front()) {
      if(isa<func::ReturnOp>(op)) continue;
      if(op.getNumResults()!=1) {op.emitError("expected a single-result tensor operation"); return signalPassFailure();}
      if(auto c=dyn_cast<tosa::ConstOp>(op)) {
        if(c.getType().getElementType().isF32()) {
          if(failed(type(c.getResult()))) return signalPassFailure();
          weights.push_back(c);
        }
        continue;
      }
      if(failed(type(op.getResult(0)))) return signalPassFailure();
      if(auto instance=op.getAttrOfType<IntegerAttr>("sodap.instance"))
        if(instance.getInt()<0 || instance.getInt()>INT32_MAX) {
          op.emitError("invalid physical accelerator instance");return signalPassFailure();
        }
      operations.push_back(&op);
    }
    // Analyze the unchanged source graph. Hardware extraction consumes the
    // original module; only host scheduling absorbs these expression nodes.
    for(auto *op:operations)if(auto matmul=dyn_cast<tosa::MatMulOp>(op)) {
      Gemm g=matchGemm(matmul);
      if(g.c) {
        auto out=cast<RankedTensorType>(g.ops.back()->getResult(0).getType());
        auto c=cast<RankedTensorType>(g.c.getType());
        auto m=shape(matmul.getResult());
        bool exact=c.getShape()==out.getShape();
        bool vectorBias=m.size()==3 && m[0]==1 && m[1]==1 && c.getNumElements()==m[2];
        if(!exact && !vectorBias) {
          // Preserve the separate add and its existing broadcast diagnostic.
          g.ops.pop_back();g.c={};g.beta={};g.betaMul=nullptr;
        }
      }
      auto *last=g.ops.back();
      for(auto *part:g.ops)if(part!=last)fused.insert(part);
      if(g.betaMul)fused.insert(g.betaMul);
      gemms.try_emplace(last,std::move(g));
    }
    // Fusion is legal only with one consumer, so live residual values retain
    // their original semantics. Keep the actual clamp in the hardware source.
    for(auto *op:operations) if(relu(op)) {
      Value x=op->getOperand(0); auto *producer=x.getDefiningOp();
      if(producer && !gemms.count(producer) && x.hasOneUse() && (isa<tosa::Conv2DOp,tosa::AddOp>(producer))) {
        aliases[op->getResult(0)]=x; fused.insert(op); activations[producer]=true;
      }
    }
    // A view of the same packed rows owns no storage. Resolve transitively so
    // every consumer (including return) extends the underlying allocation.
    for(auto *op:operations)
      if(!fused.contains(op) && !gemms.count(op) && isa<tosa::ReshapeOp>(op) &&
          sameLayout(op->getOperand(0),op->getResult(0))) {
        aliases[op->getResult(0)]=op->getOperand(0);fused.insert(op);
      }
    // Stable topological partition: CPU work with no accelerator dependency
    // runs after input packing and before the first launch. Compute lifetimes
    // on this schedule, not on the original textual order.
    DenseSet<Value> inputOnly,immutable;
    for(auto arg:source.getArguments())inputOnly.insert(arg);
    for(auto c:weights) {inputOnly.insert(c.getResult());immutable.insert(c.getResult());}
    SmallVector<Operation *> early,dependent;
    for(auto *op:operations) {
      if(fused.contains(op))continue;
      bool ready=operatorFamily(op).empty() && llvm::all_of(operands(op),
          [&](Value v){return inputOnly.contains(storageValue(v));});
      if(ready) {early.push_back(op);inputOnly.insert(storageValue(op->getResult(0)));}
      else dependent.push_back(op);
    }
    operations=early;operations.append(dependent);
    // Last uses come from the MLIR graph, including return and residual branches.
    for(auto [i,op]:llvm::enumerate(operations))
      if(!fused.contains(op))
        for(Value v:operands(op)) lastUse[storageValue(v)]=i;
    lastUse[storageValue(ret.getOperand(0))]=operations.size();
    DenseSet<Operation *> inputOnlyC;
    for(auto *op:operations) {
      auto it=gemms.find(op);
      if(it!=gemms.end() && it->second.c && inputOnly.contains(storageValue(it->second.c)))inputOnlyC.insert(op);
    }
    // Transfer ownership only after C's last read. Model-input arena copies are
    // refreshed on every forward_run; parameter storage is never consumed.
    // A/B (or a scalar) sharing C is still read during the launch and cannot be
    // overwritten safely. Layout-changing C also needs its own packed result.
    for(auto [i,op]:llvm::enumerate(operations)) {
      auto it=gemms.find(op);if(it==gemms.end() || !it->second.c)continue;
      auto &g=it->second;Value c=storageValue(g.c),out=op->getResult(0);
      bool otherRead=false;
      for(Value v:SmallVector<Value>{g.matmul.getA(),g.matmul.getB(),g.alpha,g.beta})
        if(v && storageValue(v)==c)otherRead=true;
      if(!immutable.contains(c) && lastUse.lookup(c)==i && !otherRead && sameLayout(c,out)) {
        consumeC.insert(op);aliases[out]=c;
        lastUse[c]=std::max(lastUse.lookup(c),lastUse.lookup(out));
      } else if(inputOnlyC.contains(op))prepareC.insert(op);
    }
    DenseSet<Value> persistent;
    for(auto c:weights) {
      allocate(c.getResult(),true); auto &b=buffer(c.getResult());
      b.parameter=parameters; parameters+=b.elements; persistent.insert(c.getResult());
    }
    for(auto arg:source.getArguments()) {allocate(arg,true);persistent.insert(arg);}
    // Input-only C preparation can be hoisted too. Reserve its destination
    // before the schedule, so no intervening operation can reuse that storage.
    for(auto *op:operations)if(prepareC.contains(op))allocate(op->getResult(0),false);
    SmallVector<std::pair<std::string,SmallVector<int64_t>>> calls;
    SmallVector<Operation *> scheduled;
    for(auto [i,op]:llvm::enumerate(operations)) {
      if(fused.contains(op)) continue;
      // Reject data operands not allocated by the SSA analysis (integer
      // constants are allowed only as transpose permutations).
      for(Value v:operands(op))
        if(!buffers.count(storageValue(v))) {op->emitError("unsupported data operand");return signalPassFailure();}
      if(!consumeC.contains(op) && !prepareC.contains(op))allocate(op->getResult(0),false);
      auto a=arguments(op); if(failed(a)) return signalPassFailure();
      auto kind=operatorFamily(op);
      auto *bound=binding(op);
      if(!kind.empty()) {
        std::string name=kind.str()+"_default";
        if(auto attr=bound->getAttrOfType<StringAttr>("sodap.implementation"))name=attr.getValue().str();
        if(name.empty() || name.size()>64 || (!std::isalpha(name[0]) && name[0]!='_') ||
           llvm::any_of(name,[](unsigned char c){return !std::isalnum(c) && c!='_';})) {
          op->emitError("implementation must be a safe symbol name");return signalPassFailure();
        }
        unsigned lanes=kind=="conv"?convLanes:kind=="gemm"?gemmLanes:addLanes;
        if(auto attr=bound->getAttrOfType<IntegerAttr>("sodap.compute_lanes")) {
          if(attr.getInt()<1 || attr.getInt()>256) {op->emitError("invalid compute lane budget");return signalPassFailure();}
          lanes=attr.getInt();
        }
        if(!lanes || lanes>256) {op->emitError("compute lane budget must be 1..256");return signalPassFailure();}
        auto [it,inserted]=engines.try_emplace(name,Engine{kind.str(),lanes,{}});
        if(it->second.family!=kind || it->second.lanes!=lanes) {
          op->emitError("incompatible contracts/resources assigned to one implementation");return signalPassFailure();
        }
        unsigned instance=kind=="conv"?convInstance:kind=="gemm"?gemmInstance:addInstance;
        if(auto attr=bound->getAttrOfType<IntegerAttr>("sodap.instance"))instance=attr.getInt();
        for(auto &[other,e]:engines)
          if(other!=name && e.family==kind && llvm::is_contained(e.instances,instance)) {
            op->emitError("different implementations assigned to the same physical device");return signalPassFailure();
          }
        if(!llvm::is_contained(it->second.instances,instance))it->second.instances.push_back(instance);
      }
      calls.push_back(std::move(*a)); scheduled.push_back(op);
      SmallVector<Value> dead;
      for(auto &[v,b]:buffers)
        if(!persistent.contains(v) && lastUse.lookup(v)==i && v!=storageValue(op->getResult(0))) dead.push_back(v);
      for(Value v:dead) release(buffers[v]);
      // Entries remain for manifest provenance; prevent a second release.
      for(Value v:dead) lastUse[v]=UINT_MAX;
    }
    if(cursor>(1<<30)-OP_HEADER_BYTES) {source.emitError("network arena exceeds runtime address limit");return signalPassFailure();}
    std::error_code ec=llvm::sys::fs::create_directories(directory);
    if(ec) {module.emitError(ec.message());return signalPassFailure();}
    auto file=[&](StringRef name) {
      llvm::SmallString<256> path(directory); llvm::sys::path::append(path,name);
      return std::make_unique<llvm::ToolOutputFile>(path,ec,llvm::sys::fs::OF_None);
    };
    auto params=file("parameters.bin"); if(ec) {module.emitError(ec.message());return signalPassFailure();}
    for(auto c:weights) for(APFloat f:cast<DenseFPElementsAttr>(c.getValue()).getValues<APFloat>()) {
      uint32_t bits=f.bitcastToAPInt().getZExtValue();
      char bytes[]={char(bits),char(bits>>8),char(bits>>16),char(bits>>24)};
      params->os().write(bytes,4);
    }
    auto original=file("source.mlir"); if(ec) {module.emitError(ec.message());return signalPassFailure();}
    module.print(original->os()); original->os()<<"\n";
    OpBuilder b(&getContext()); builder=&b; b.setInsertionPointToEnd(module.getBody());
    auto mem=[&](int64_t n){return MemRefType::get({n},b.getF32Type());};
    auto make=[&](StringRef name,TypeRange inputs,TypeRange outputs) {
      b.setInsertionPointToEnd(module.getBody());
      auto fn=b.create<func::FuncOp>(loc,name,b.getFunctionType(inputs,outputs));
      b.setInsertionPointToStart(fn.addEntryBlock());return fn;
    };
    auto init=make("forward_init",TypeRange{mem(parameters)},TypeRange{b.getI64Type()});
    auto used=[&](StringRef family)->int64_t {
      for(auto &[name,e]:engines)if(e.family==family)return e.instances.front();
      return -1;
    };
    SmallVector<Value> open{constant(cursor),constant(dmaWidth),constant(used("conv")),constant(used("add")),constant(used("gemm"))};
    Value ctx=call("esp_op_open_devices",open,TypeRange{b.getI64Type()}).getResult(0);
    for(auto c:weights) {
      auto p=buffer(c.getResult()); reg(ctx,p);
      Value slice=b.create<memref::SubViewOp>(loc,init.getArgument(0),
          ArrayRef<int64_t>{p.parameter},ArrayRef<int64_t>{p.elements},ArrayRef<int64_t>{1});
      copy(ctx,p,slice,true);
    }
    b.create<func::ReturnOp>(loc,ctx);
    SmallVector<Type> inputs{b.getI64Type()};
    for(auto arg:source.getArguments()) inputs.push_back(mem(buffer(arg).elements));
    inputs.push_back(mem(buffer(ret.getOperand(0)).elements));
    auto run=make("forward_run",inputs,{}); ctx=run.getArgument(0);
    for(auto [i,arg]:llvm::enumerate(source.getArguments())) {reg(ctx,buffer(arg));copy(ctx,buffer(arg),run.getArgument(i+1),true);}
    auto copyC=[&](Operation *op) {
      auto input=buffer(gemms.find(op)->second.c),out=buffer(op->getResult(0));
      reg(ctx,out);
      invoke("esp_op_repack",ctx,{input.offset,out.offset,out.elements});
      records.push_back(llvm::json::Object{{"index",int64_t(records.size())},
        {"operator","repack"},{"execution","cpu"},{"runtime","repack"},
        {"arguments",jsonInts({input.offset,out.offset,out.elements})},
        {"inputs",llvm::json::Array{input.id}},{"output",out.id},
        {"reason","preserve GEMM C before in-place execution"}});
    };
    bool prepared=false;
    for(auto [i,op]:llvm::enumerate(scheduled)) {
      if(!prepared && !operatorFamily(op).empty()) {
        for(auto *candidate:scheduled)if(prepareC.contains(candidate))copyC(candidate);
        prepared=true;
      }
      auto &entry=calls[i]; Buffer out=buffer(op->getResult(0)); reg(ctx,out);
      auto f=operatorFamily(op); bool hw=!f.empty();
      auto *bound=binding(op);
      unsigned instance=f=="conv"?convInstance:f=="gemm"?gemmInstance:addInstance;
      if(auto attr=bound->getAttrOfType<IntegerAttr>("sodap.instance"))instance=attr.getInt();
      std::string implementation=f.str()+"_default";
      if(auto attr=bound->getAttrOfType<StringAttr>("sodap.implementation"))implementation=attr.getValue().str();
      if(hw)invoke("esp_op_select_device",ctx,{f=="conv"?0:f=="add"?1:2,instance});
      auto g=gemms.find(op);
      if(g!=gemms.end() && g->second.c && !consumeC.contains(op) && !prepareC.contains(op))copyC(op);
      func::CallOp callOp;
      if(entry.first=="gemm_full_source") {
        SmallVector<Value> values{ctx};
        for(auto n:entry.second)values.push_back(constant(n));
        auto scalar=[&](Value v,float fallback)->Value {
          if(v)return call("esp_op_scalar",{ctx,constant(buffer(v).offset)},TypeRange{b.getF32Type()}).getResult(0);
          return b.create<arith::ConstantOp>(loc,b.getF32FloatAttr(fallback));
        };
        values.push_back(scalar(g->second.alpha,1));
        values.push_back(scalar(g->second.beta,g->second.c?1:0));
        values.push_back(scalar({},g->second.c?1:0));
        callOp=call("esp_op_gemm_full_source",values);
      } else callOp=invoke("esp_op_"+entry.first,ctx,entry.second);
      callOp->setAttr("sodap.execution",b.getStringAttr(hw?"accelerator":"cpu"));
      callOp->setAttr("sodap.operator",b.getStringAttr(hw?f:entry.first));
      callOp->setAttr("sodap.invocation",b.getI64IntegerAttr(records.size()));
      if(hw) {
        callOp->setAttr("sodap.instance",b.getI64IntegerAttr(instance));
        callOp->setAttr("sodap.implementation",b.getStringAttr(implementation));
      }
      llvm::json::Array ids;
      for(auto v:operands(op)) ids.push_back(buffer(v).id);
      llvm::json::Object record{{"index",int64_t(records.size())},{"operator",hw?f.str():entry.first},
        {"execution",hw?"accelerator":"cpu"},{"runtime",entry.first},{"instance",hw?int64_t(instance):-1},
        {"descriptor_version",hw?3:0},{"implementation",hw?implementation:""},{"arguments",jsonInts(entry.second)},
        {"inputs",std::move(ids)},{"output",out.id},{"shape",jsonInts(shape(op->getResult(0)))},
        {"relu",activations.lookup(op)}};
      if(g!=gemms.end()) {
        auto coefficient=[&](Value v,float fallback) {
          return v?llvm::json::Object{{"buffer",buffer(v).id}}:llvm::json::Object{{"constant",fallback}};
        };
        record["alpha"]=coefficient(g->second.alpha,1);
        record["beta"]=coefficient(g->second.beta,g->second.c?1:0);
        record["has_c"]=bool(g->second.c);
      }
      records.push_back(std::move(record));
    }
    copy(ctx,buffer(ret.getOperand(0)),run.getArguments().back(),false);
    b.create<func::ReturnOp>(loc);
    auto close=make("forward_close",TypeRange{b.getI64Type()},{});
    call("esp_op_close",close.getArguments());b.create<func::ReturnOp>(loc);
    llvm::json::Array inputSizes;
    for(auto arg:source.getArguments()) inputSizes.push_back(buffer(arg).elements);
    llvm::json::Array implementations;
    for(auto &[name,e]:engines) {
      llvm::json::Array instances;for(auto i:e.instances)instances.push_back(i);
      implementations.push_back(llvm::json::Object{{"name",name},{"operator",e.family},
        {"precision","FP32"},{"lanes",e.lanes},{"plm_bytes",plmBytes.getValue()},{"instances",std::move(instances)}});
    }
    llvm::json::Object report{{"schema",1},{"compiler","sodap-tosa-network-host"},
      {"hardware_status","SOURCE_PRESERVED_NOT_LOWERED"},{"dma_width",dmaWidth.getValue()},
      {"arena_bytes",cursor},{"parameter_elements",parameters},{"input_elements",std::move(inputSizes)},
      {"output_elements",buffer(ret.getOperand(0)).elements},{"buffers",std::move(storage)},
      {"engines",std::move(implementations)},{"invocations",std::move(records)}};
    llvm::json::Array inputBuffers,parameterBindings;
    for(auto arg:source.getArguments())inputBuffers.push_back(buffer(arg).id);
    for(auto c:weights) {
      auto p=buffer(c.getResult());
      parameterBindings.push_back(llvm::json::Object{{"buffer",p.id},
        {"parameter_offset",p.parameter},{"elements",p.elements}});
    }
    report["input_buffers"]=std::move(inputBuffers);
    report["parameter_bindings"]=std::move(parameterBindings);
    report["output_buffer"]=buffer(ret.getOperand(0)).id;
    auto manifest=file("network.json"); if(ec) {module.emitError(ec.message());return signalPassFailure();}
    manifest->os()<<llvm::formatv("{0:2}\n",llvm::json::Value(std::move(report)));
    source.erase();
    params->os().flush();original->os().flush();manifest->os().flush();
    if(params->os().has_error() || original->os().has_error() || manifest->os().has_error()) {
      module.emitError("failed writing network artifacts");return signalPassFailure();
    }
    params->keep();original->keep();manifest->keep();
  }
};
} // namespace
} // namespace mlir::sodap
