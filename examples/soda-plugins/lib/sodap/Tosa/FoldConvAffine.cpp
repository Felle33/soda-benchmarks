// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/Tosa/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cmath>

namespace mlir::sodap {
#define GEN_PASS_DEF_TOSAFOLDCONVAFFINE
#include "sodap/Tosa/Passes.h.inc"
namespace {
// TOSA's general folder does not evaluate all inference constants (notably
// variance+epsilon and rsqrt). Evaluate only actual constant operands, with
// ordinary FP32 rounding and checked broadcast indices.
struct FoldConstants : RewritePattern {
  FoldConstants(MLIRContext *ctx):RewritePattern(MatchAnyOpTypeTag(),2,ctx){}
  LogicalResult matchAndRewrite(Operation *op,PatternRewriter &r) const override {
    bool rsqrt=isa<tosa::RsqrtOp>(op),reshape=isa<tosa::ReshapeOp>(op);
    bool mul=isa<tosa::MulOp>(op),sub=isa<tosa::SubOp>(op),add=isa<tosa::AddOp>(op);
    if(!rsqrt && !reshape && !mul && !sub && !add)return failure();
    if(mul && cast<tosa::MulOp>(op).getShift()!=0)return failure();
    auto out=dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if(!out || !out.hasStaticShape() || !out.getElementType().isF32())return failure();
    SmallVector<DenseFPElementsAttr> attrs;
    for(Value v:op->getOperands()) {
      DenseFPElementsAttr a;if(!matchPattern(v,m_Constant(&a)))return failure();attrs.push_back(a);
    }
    if(reshape) {
      r.replaceOpWithNewOp<tosa::ConstOp>(op,out,attrs[0].reshape(out));return success();
    }
    SmallVector<SmallVector<APFloat>> data;
    for(auto a:attrs) {
      auto shape=a.getType().getShape();
      if(shape.size()>size_t(out.getRank()))return failure();
      for(auto [i,d]:llvm::enumerate(shape))
        if(d!=1 && d!=out.getDimSize(out.getRank()-shape.size()+i))return failure();
      data.emplace_back(a.getValues<APFloat>());
    }
    SmallVector<APFloat> values;
    for(int64_t linear=0;linear<out.getNumElements();++linear) {
      SmallVector<APFloat> operands;
      for(auto [a,items]:llvm::zip(attrs,data)) {
        auto shape=a.getType().getShape();int64_t index=0,stride=1,remaining=linear;
        for(int d=out.getRank()-1;d>=0;--d) {
          auto coordinate=remaining%out.getDimSize(d);remaining/=out.getDimSize(d);
          int input=d-(out.getRank()-shape.size());
          if(input>=0) {if(shape[input]!=1)index+=coordinate*stride;stride*=shape[input];}
        }
        operands.push_back(items[index]);
      }
      APFloat value=operands[0];
      if(add)value.add(operands[1],APFloat::rmNearestTiesToEven);
      if(sub)value.subtract(operands[1],APFloat::rmNearestTiesToEven);
      if(mul)value.multiply(operands[1],APFloat::rmNearestTiesToEven);
      if(rsqrt)value=APFloat(1.0f/std::sqrt(value.convertToFloat()));
      values.push_back(value);
    }
    r.replaceOpWithNewOp<tosa::ConstOp>(op,out,DenseElementsAttr::get(out,values));
    return success();
  }
};
struct Fold : RewritePattern {
  Fold(MLIRContext *ctx):RewritePattern(MatchAnyOpTypeTag(),1,ctx){}
  LogicalResult matchAndRewrite(Operation *op,PatternRewriter &rewriter) const override {
    bool mul=isa<tosa::MulOp>(op),sub=isa<tosa::SubOp>(op),add=isa<tosa::AddOp>(op);
    if(!mul && !sub && !add)return failure();
    if(mul && cast<tosa::MulOp>(op).getShift()!=0)return failure();
    Value x=op->getOperand(0),channel=op->getOperand(1);
    auto conv=x.getDefiningOp<tosa::Conv2DOp>();
    if(!conv && !sub) {std::swap(x,channel);conv=x.getDefiningOp<tosa::Conv2DOp>();}
    if(!conv || !x.hasOneUse() || conv.getType()!=op->getResult(0).getType())return failure();
    DenseFPElementsAttr weight,bias,coeff;
    if(!matchPattern(conv.getWeight(),m_Constant(&weight)) ||
       !matchPattern(conv.getBias(),m_Constant(&bias)) || !matchPattern(channel,m_Constant(&coeff)))return failure();
    auto wt=weight.getType(),ct=coeff.getType();
    if(!wt.getElementType().isF32() || wt.getRank()!=4 || ct.getRank()!=4 ||
       ct.getDimSize(0)!=1 || ct.getDimSize(1)!=1 || ct.getDimSize(2)!=1 ||
       (ct.getDimSize(3)!=1 && ct.getDimSize(3)!=wt.getDimSize(0)) ||
       bias.getNumElements()!=wt.getDimSize(0))return failure();
    SmallVector<APFloat> ws(weight.getValues<APFloat>()),bs(bias.getValues<APFloat>()),cs(coeff.getValues<APFloat>());
    auto finite=[](const APFloat &f){return f.isFinite();};
    if(!llvm::all_of(ws,finite) || !llvm::all_of(bs,finite) || !llvm::all_of(cs,finite))return failure();
    int64_t perChannel=weight.getNumElements()/wt.getDimSize(0);
    for(int64_t c=0;c<wt.getDimSize(0);++c) {
      const APFloat &v=cs[cs.size()==1?0:c];
      if(mul) {
        for(int64_t j=0;j<perChannel;++j)ws[c*perChannel+j].multiply(v,APFloat::rmNearestTiesToEven);
        bs[c].multiply(v,APFloat::rmNearestTiesToEven);
      } else if(sub)bs[c].subtract(v,APFloat::rmNearestTiesToEven);
      else bs[c].add(v,APFloat::rmNearestTiesToEven);
    }
    if(!llvm::all_of(ws,finite) || !llvm::all_of(bs,finite))return failure();
    Value w=rewriter.create<tosa::ConstOp>(op->getLoc(),wt,DenseElementsAttr::get(wt,ws));
    Value biasValue=rewriter.create<tosa::ConstOp>(op->getLoc(),bias.getType(),DenseElementsAttr::get(bias.getType(),bs));
    auto *replacement=rewriter.clone(*conv);
    replacement->setOperand(1,w);replacement->setOperand(2,biasValue);
    int64_t count=1;
    if(auto old=conv->getAttrOfType<IntegerAttr>("sodap.folded_channel_affine"))count+=old.getInt();
    replacement->setAttr("sodap.folded_channel_affine",rewriter.getI64IntegerAttr(count));
    rewriter.replaceOp(op,replacement->getResults());
    return success();
  }
};
struct FoldConvAffine : impl::TosaFoldConvAffineBase<FoldConvAffine> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());patterns.add<FoldConstants,Fold>(&getContext());
    if(failed(applyPatternsAndFoldGreedily(getOperation(),std::move(patterns))))signalPassFailure();
  }
};
} // namespace
} // namespace mlir::sodap
