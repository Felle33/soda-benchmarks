// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/Tosa/Passes.h"
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include "mlir/IR/IRMapping.h"
#include <cfloat>

namespace mlir::sodap {
#define GEN_PASS_DEF_TOSAOUTLINEADDITIONS
#include "sodap/Tosa/Passes.h.inc"
namespace {
struct OutlineAdditions : impl::TosaOutlineAdditionsBase<OutlineAdditions> {
  using Base::Base;
  void runOnOperation() override {
    auto module=getOperation();OpBuilder b(&getContext());Location loc=module.getLoc();
    SmallVector<tosa::AddOp> adds;
    module.walk([&](tosa::AddOp op){adds.push_back(op);});
    if(adds.empty()) {module.emitError("no source addition to generalize");return signalPassFailure();}
    if(!lanes || lanes>256 || plmBytes<24*lanes) {
      module.emitError("invalid pointwise compute/storage budget");return signalPassFailure();
    }
    SmallVector<Attribute> invocations;
    for(auto op:adds) {
      auto t=dyn_cast<RankedTensorType>(op.getType());
      if(!t || !t.hasStaticShape() || t.getRank()<1 || !t.getElementType().isF32() ||
          op.getInput1().getType()!=t || op.getInput2().getType()!=t ||
          llvm::any_of(t.getShape(),[](int64_t d){return d<=0;})) {
        op.emitError("reusable addition requires equal positive FP32 tensor shapes");return signalPassFailure();
      }
      int64_t columns=t.getShape().back();
      invocations.push_back(b.getDenseI64ArrayAttr({t.getNumElements()/columns,columns}));
    }
    // Bounded policy: one row of a power-of-two burst tile, three ping-ponged
    // FP32 arrays. No model name or invocation extent enters hardware selection.
    int64_t columns=1;
    while(columns*2<=4096 && uint64_t(columns*2)*24<=plmBytes)columns*=2;
    b.setInsertionPointToStart(module.getBody());
    auto tile=b.create<ModuleOp>(loc,StringRef("forward_kernel"));
    tile->setAttr("sodap.tile",b.getI64IntegerAttr(0));
    tile->setAttr("sodap.tile_dir",b.getStringAttr("tiles/forward_kernel"));
    tile->setAttr("sodap.operator",b.getStringAttr("add"));
    tile->setAttr("sodap.compute_lanes",b.getI64IntegerAttr(lanes));
    tile->setAttr("sodap.invocations",b.getArrayAttr(invocations));
    tile->setAttr("sodap.hardware_tile_sizes",b.getArrayAttr({b.getArrayAttr({b.getI64IntegerAttr(1),b.getI64IntegerAttr(columns)})}));
    b.setInsertionPointToStart(tile.getBody());
    auto matrix=RankedTensorType::get({ShapedType::kDynamic,ShapedType::kDynamic},b.getF32Type());
    auto scalar=RankedTensorType::get({1,1},b.getF32Type());
    auto fn=b.create<func::FuncOp>(loc,"forward_kernel",b.getFunctionType(
      {matrix,matrix,b.getIndexType(),b.getIndexType(),b.getF32Type()},{matrix}));
    fn->setAttr("sodap.descriptor_fields",b.getI64IntegerAttr(OP_HEADER_FIELDS));
    fn->setAttr("sodap.descriptor_magic",b.getI64IntegerAttr(OP_ADD_MAGIC));
    fn->setAttr("sodap.descriptor_version",b.getI64IntegerAttr(OP_PACKED_VERSION));
    fn->setAttr("sodap.descriptor_width_field",b.getI64IntegerAttr(OP_DMA_WIDTH));
    int64_t fields[]={ADD_A,ADD_B,ADD_ROWS,ADD_COLUMNS,ADD_ACTIVATION_FP};
    for(auto [i,f]:llvm::enumerate(fields))fn.setArgAttr(i,"sodap.descriptor_field",b.getI64IntegerAttr(f));
    for(unsigned i=0;i<2;++i) {
      fn.setArgAttr(i,"sodap.dims",b.getDenseI64ArrayAttr({2,3}));
      fn.setArgAttr(i,"sodap.descriptor_ld",b.getI64IntegerAttr(ADD_ROW_BYTES));
    }
    fn.setResultAttr(0,"sodap.descriptor_field",b.getI64IntegerAttr(ADD_C));
    fn.setResultAttr(0,"sodap.descriptor_ld",b.getI64IntegerAttr(ADD_ROW_BYTES));
    fn.setResultAttr(0,"sodap.dims",b.getDenseI64ArrayAttr({2,3}));
    b.setInsertionPointToStart(fn.addEntryBlock());
    // Clone the source scalar operation; no handwritten load/compute/store
    // tasks or DMA schedule. Common tiling and scheduling produce those.
    auto *cloned=b.clone(*adds.front()); cloned->setOperands({fn.getArgument(0),fn.getArgument(1)});
    cloned->getResult(0).setType(matrix);
    Value sum=cloned->getResult(0);
    OperationState clamp(loc,"tosa.clamp");clamp.addOperands(sum);clamp.addTypes(matrix);
    clamp.addAttribute("min_fp",b.getF32FloatAttr(0));clamp.addAttribute("max_fp",b.getF32FloatAttr(FLT_MAX));
    clamp.addAttribute("min_int",b.getI64IntegerAttr(0));clamp.addAttribute("max_int",b.getI64IntegerAttr(INT32_MAX));
    Value activated=b.create(clamp)->getResult(0);
    Value flag=b.create<tensor::FromElementsOp>(loc,scalar,fn.getArgument(4));
    Value zero=b.create<tosa::ConstOp>(loc,scalar,DenseElementsAttr::get(scalar,b.getF32FloatAttr(0)));
    Value noActivation=b.create<tosa::EqualOp>(loc,RankedTensorType::get({1,1},b.getI1Type()),flag,zero);
    Value result=b.create<tosa::SelectOp>(loc,matrix,noActivation,sum,activated);
    b.create<func::ReturnOp>(loc,result);
  }
};
} // namespace
} // namespace mlir::sodap
