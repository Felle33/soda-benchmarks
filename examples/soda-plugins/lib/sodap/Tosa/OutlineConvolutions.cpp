// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Source convolution normalization. The input is a LOGICAL window view;
// CopyToDMA maps each requested local tile directly to NHWC DRAM coordinates.
// There is no global im2col tensor, GEMM offload, handwritten LCS schedule or
// instantiated datapath. Common tiling, task extraction and banking follow.
#include "sodap/Tosa/Passes.h"
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include <cfloat>

namespace mlir::sodap {
#define GEN_PASS_DEF_TOSAOUTLINECONVOLUTIONS
#include "sodap/Tosa/Passes.h.inc"
namespace {
struct OutlineConvolutions : impl::TosaOutlineConvolutionsBase<OutlineConvolutions> {
  using Base::Base;
  void runOnOperation() override {
    auto module=getOperation();OpBuilder b(&getContext());auto loc=module.getLoc();
    SmallVector<tosa::Conv2DOp> ops;module.walk([&](tosa::Conv2DOp op){ops.push_back(op);});
    if(ops.empty() || !lanes || lanes>256) {
      module.emitError("expected source convolutions and a 1..256 MAC budget");return signalPassFailure();
    }
    // Semantic geometry fields, independent of implementation tiles.
    enum {A,W,Bias,N,IH,IW,IC,OC,KH,KW,OH,OW,SH,SW,PT,PL,DH,DW,P,R,Activation};
    int64_t fields[]={CONV_A,CONV_W,CONV_BIAS,CONV_N,CONV_IH,CONV_IW,CONV_IC,CONV_OC,
      CONV_KH,CONV_KW,CONV_OH,CONV_OW,CONV_SH,CONV_SW,CONV_PT,CONV_PL,CONV_DH,CONV_DW,
      CONV_PIXELS,CONV_REDUCTION,CONV_ACTIVATION_FP};
    SmallVector<Attribute> invocations;
    for(auto op:ops) {
      auto x=dyn_cast<RankedTensorType>(op.getInput().getType());
      auto w=dyn_cast<RankedTensorType>(op.getWeight().getType());
      auto y=dyn_cast<RankedTensorType>(op.getType());
      auto bias=dyn_cast<RankedTensorType>(op.getBias().getType());
      if(!x || !w || !y || x.getRank()!=4 || w.getRank()!=4 || y.getRank()!=4 ||
          !x.hasStaticShape() || !w.hasStaticShape() || !y.hasStaticShape() || !x.getElementType().isF32() ||
          !w.getElementType().isF32() || !y.getElementType().isF32() || !bias ||
          bias.getRank()!=1 || !bias.getElementType().isF32() || bias.getDimSize(0)!=w.getDimSize(0) ||
          llvm::any_of(x.getShape(),[](int64_t n){return n<=0;}) ||
          llvm::any_of(w.getShape(),[](int64_t n){return n<=0;}) ||
          llvm::any_of(y.getShape(),[](int64_t n){return n<=0;}) ||
          x.getDimSize(0)!=y.getDimSize(0) || x.getDimSize(3)!=w.getDimSize(3) || w.getDimSize(0)!=y.getDimSize(3)) {
        op.emitError("unsupported dense NHWC/OHWI FP32 convolution");return signalPassFailure();
      }
      auto s=op.getStride(),p=op.getPad(),d=op.getDilation();
      if(s.size()!=2 || d.size()!=2 || p.size()!=4 ||
          llvm::any_of(s,[](int64_t n){return n<=0;}) ||
          llvm::any_of(d,[](int64_t n){return n<=0;}) ||
          llvm::any_of(p,[](int64_t n){return n<0;})) {
        op.emitError("unsupported convolution geometry");return signalPassFailure();
      }
      for(unsigned i=0;i<2;++i) {
        int64_t span=x.getDimSize(i+1)+p[2*i]+p[2*i+1]-(w.getDimSize(i+1)-1)*d[i]-1;
        if(span<0 || y.getDimSize(i+1)!=span/s[i]+1) {
          op.emitError("convolution output shape disagrees with geometry");return signalPassFailure();
        }
      }
      invocations.push_back(b.getDenseI64ArrayAttr({x.getDimSize(0),x.getDimSize(1),x.getDimSize(2),x.getDimSize(3),
        w.getDimSize(0),w.getDimSize(1),w.getDimSize(2),y.getDimSize(1),y.getDimSize(2),s[0],s[1],p[0],p[2],d[0],d[1],
        y.getDimSize(1)*y.getDimSize(2),w.getDimSize(1)*w.getDimSize(2)*w.getDimSize(3)}));
    }
    // Hardware extents are selected later by the shared GEMM resource policy.
    // This pass specifies only the logical contraction and descriptor contract.
    b.setInsertionPointToStart(module.getBody());
    auto tile=b.create<ModuleOp>(loc,StringRef("forward_kernel"));
    tile->setAttr("sodap.tile",b.getI64IntegerAttr(0));tile->setAttr("sodap.tile_dir",b.getStringAttr("tiles/forward_kernel"));
    tile->setAttr("sodap.operator",b.getStringAttr("conv"));tile->setAttr("sodap.compute_lanes",b.getI64IntegerAttr(lanes));
    tile->setAttr("sodap.invocations",b.getArrayAttr(invocations));
    tile->setAttr("sodap.bounded_contractions",b.getUnitAttr());
    b.setInsertionPointToStart(tile.getBody());
    auto tensor=[&](unsigned rank){return RankedTensorType::get(SmallVector<int64_t>(rank,ShapedType::kDynamic),b.getF32Type());};
    SmallVector<Type> inputs{tensor(3),tensor(2),tensor(1)};
    inputs.append(17,b.getIndexType());inputs.push_back(b.getF32Type());
    auto fn=b.create<func::FuncOp>(loc,"forward_kernel",b.getFunctionType(inputs,{tensor(3)}));
    fn->setAttr("sodap.bounded_contractions",b.getUnitAttr());
    fn->setAttr("sodap.descriptor_fields",b.getI64IntegerAttr(OP_HEADER_FIELDS));
    fn->setAttr("sodap.descriptor_magic",b.getI64IntegerAttr(OP_CONV_MAGIC));
    fn->setAttr("sodap.descriptor_version",b.getI64IntegerAttr(CONV_MATRIX_VERSION));
    fn->setAttr("sodap.descriptor_width_field",b.getI64IntegerAttr(OP_DMA_WIDTH));
    for(auto [i,f]:llvm::enumerate(fields))fn.setArgAttr(i,"sodap.descriptor_field",b.getI64IntegerAttr(f));
    fn.setArgAttr(A,"sodap.dims",b.getDenseI64ArrayAttr({N,P,R}));
    fn.setArgAttr(W,"sodap.dims",b.getDenseI64ArrayAttr({R,OC}));
    fn.setArgAttr(Bias,"sodap.dims",b.getDenseI64ArrayAttr({OC}));
    fn.setArgAttr(A,"sodap.dma_view",b.getStringAttr("conv_window_flat"));
    fn.setArgAttr(W,"sodap.dma_view",b.getStringAttr("packed"));
    fn.setArgAttr(Bias,"sodap.dma_view",b.getStringAttr("packed"));
    fn.setResultAttr(0,"sodap.dims",b.getDenseI64ArrayAttr({N,P,OC}));
    fn.setResultAttr(0,"sodap.descriptor_field",b.getI64IntegerAttr(CONV_C));
    fn.setResultAttr(0,"sodap.dma_view",b.getStringAttr("packed"));
    b.setInsertionPointToStart(fn.addEntryBlock());
    auto arg=[&](unsigned n){return fn.getArgument(n);};
    Value empty=b.create<tensor::EmptyOp>(loc,SmallVector<int64_t>(3,ShapedType::kDynamic),b.getF32Type(),ValueRange{arg(N),arg(P),arg(OC)});
    Value zero=b.create<arith::ConstantOp>(loc,b.getF32FloatAttr(0));
    Value filled=b.create<linalg::FillOp>(loc,ValueRange{zero},ValueRange{empty}).getResult(0);
    filled.getDefiningOp()->setAttr("sodap.bound_axes",b.getDenseI64ArrayAttr({0,1,2}));
    auto map=[&](unsigned rank,ArrayRef<unsigned> positions){
      SmallVector<AffineExpr> dims;for(auto p:positions)dims.push_back(b.getAffineDimExpr(p));
      return AffineMap::get(rank,0,dims,&getContext());
    };
    SmallVector<utils::IteratorType> iters(4,utils::IteratorType::parallel);
    iters[3]=utils::IteratorType::reduction;
    auto conv=b.create<linalg::GenericOp>(loc,TypeRange{tensor(3)},ValueRange{arg(A),arg(W)},ValueRange{filled},
      ArrayRef<AffineMap>{map(4,{0,1,3}),map(4,{3,2}),map(4,{0,1,2})},iters,
      [&](OpBuilder &body,Location l,ValueRange v){
        Value product=body.create<arith::MulFOp>(l,v[0],v[1]);
        Value sum=body.create<arith::AddFOp>(l,v[2],product);body.create<linalg::YieldOp>(l,sum);
      });
    conv->setAttr("sodap.direct_convolution",b.getUnitAttr());
    conv->setAttr("sodap.bound_axes",b.getDenseI64ArrayAttr({0,1,2,3}));
    auto epilogue=b.create<linalg::GenericOp>(loc,TypeRange{tensor(3)},ValueRange{conv.getResult(0),arg(Bias)},ValueRange{empty},
      ArrayRef<AffineMap>{map(3,{0,1,2}),map(3,{2}),map(3,{0,1,2})},SmallVector<utils::IteratorType>(3,utils::IteratorType::parallel),
      [&](OpBuilder &body,Location l,ValueRange v){
        Value sum=body.create<arith::AddFOp>(l,v[0],v[1]);
        Value maximum=body.create<arith::ConstantOp>(l,body.getF32FloatAttr(FLT_MAX));
        Value clamped=body.create<arith::MinimumFOp>(l,sum,maximum);
        clamped=body.create<arith::MaximumFOp>(l,clamped,zero);
        Value disabled=body.create<arith::CmpFOp>(l,arith::CmpFPredicate::OEQ,arg(Activation),zero);
        Value result=body.create<arith::SelectOp>(l,disabled,sum,clamped);body.create<linalg::YieldOp>(l,result);
      });
    epilogue->setAttr("sodap.bound_axes",b.getDenseI64ArrayAttr({0,1,2}));
    b.create<func::ReturnOp>(loc,epilogue.getResult(0));
  }
};
} // namespace
} // namespace mlir::sodap
