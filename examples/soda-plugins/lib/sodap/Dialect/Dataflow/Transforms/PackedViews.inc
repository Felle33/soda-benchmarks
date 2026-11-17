// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Descriptor-backed logical views. Only the requested local tile is copied.
// Window coordinates remain separate, avoiding flattened-index division by
// runtime spatial/kernel sizes. The shared broker retains burst ownership.
// Counter decoding is used only at tile entry. Inner traversal carries pixel,
// channel and kernel coordinates rather than dividing a flattened index.
static std::pair<Value,Value> splitWindowCounter(PackedEmitter &e,Value value,Value radix) {
  OpBuilder &b=e.b;auto loc=e.loc;OpBuilder::InsertionGuard guard(b);
  SmallVector<Type> types(2,b.getIndexType());
  auto loop=b.create<scf::WhileOp>(loc,types,ValueRange{e.idx(0),value});
  b.createBlock(&loop.getBefore(),{},types,SmallVector<Location>(2,loc));
  auto before=loop.getBefore().front().getArguments();
  b.create<scf::ConditionOp>(loc,b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::uge,before[1],radix),before);
  b.createBlock(&loop.getAfter(),{},types,SmallVector<Location>(2,loc));
  auto after=loop.getAfter().front().getArguments();
  b.create<scf::YieldOp>(loc,ValueRange{e.add(after[0],e.idx(1)),e.sub(after[1],radix)});
  return {loop.getResult(0),loop.getResult(1)};
}

static void lowerFlatWindowTransfer(OpBuilder &b,Transfer &t,const DMAStreams &dma,
    unsigned width,Value meta,Value notice,ArrayRef<Value> fields,bool mergeRuns) {
  b.setInsertionPoint(t.copy);auto loc=t.copy.getLoc();PackedEmitter e{b,loc,width};
  auto offsets=t.tile.getMixedOffsets(),sizes=t.tile.getMixedSizes();
  auto value=[&](OpFoldResult v){return getValueOrCreateConstantIndexOp(b,loc,v);};
  auto f=[&](unsigned field){return fields[field];};
  auto minimum=[&](Value a,Value c)->Value{return b.create<arith::MinUIOp>(loc,a,c);};
  Value zero=e.idx(0),one=e.idx(1),lanes=e.idx(width/32);
  Value inputRow=e.mul(e.div(e.add(f(CONV_IC),e.idx(width/32-1)),lanes),lanes);
  Value base=e.div(f(t.baseField),e.idx(4)),buffer=t.copy.getTarget();
  Value pixelOffset=value(offsets[1]),reductionOffset=value(offsets[2]);
  auto [row0,column0]=splitWindowCounter(e,pixelOffset,f(CONV_OW));
  auto [kh0,rowReduction]=splitWindowCounter(e,reductionOffset,e.mul(f(CONV_KW),f(CONV_IC)));
  auto [kw0,ic0]=splitWindowCounter(e,rowReduction,f(CONV_IC));
  Value pixels=value(sizes[1]),reduction=value(sizes[2]);
  e.loop(value(sizes[0]),[&](Value batch) {
    Value globalBatch=e.add(value(offsets[0]),batch);
    b.create<scf::ForOp>(loc,zero,pixels,one,ValueRange{row0,column0},
      [&](OpBuilder &nested,Location,Value pixel,ValueRange coords) {
        OpBuilder::InsertionGuard guard(b);b.setInsertionPoint(nested.getInsertionBlock(),nested.getInsertionPoint());
        SmallVector<Type> types(4,b.getIndexType());
        auto walk=b.create<scf::WhileOp>(loc,types,ValueRange{zero,kh0,kw0,ic0});
        b.createBlock(&walk.getBefore(),{},types,SmallVector<Location>(4,loc));
        auto before=walk.getBefore().front().getArguments();
        b.create<scf::ConditionOp>(loc,e.lt(before[0],reduction),before);
        b.createBlock(&walk.getAfter(),{},types,SmallVector<Location>(4,loc));
        auto state=walk.getAfter().front().getArguments();
        Value cursor=state[0],kh=state[1],kw=state[2],ic=state[3];
        Value y=e.sub(e.add(e.mul(coords[0],f(CONV_SH)),e.mul(kh,f(CONV_DH))),f(CONV_PT));
        Value x=e.sub(e.add(e.mul(coords[1],f(CONV_SW)),e.mul(kw,f(CONV_DW))),f(CONV_PL));
        Value valid=e.band(e.lt(y,f(CONV_IH)),e.lt(x,f(CONV_IW)));
        Value remaining=e.sub(reduction,cursor);
        Value length=minimum(remaining,e.sub(f(CONV_IC),ic));
        if(mergeRuns) {
          // Packed odd-channel rows contain a padding lane. Such a gap is
          // never merged into logical data, even when dilation is one.
          Value contiguous=e.band(e.eq(inputRow,f(CONV_IC)),e.band(e.eq(f(CONV_DH),one),e.eq(f(CONV_DW),one)));
          Value columns=minimum(e.sub(f(CONV_KW),kw),e.sub(f(CONV_IW),x));
          Value run=minimum(remaining,e.sub(e.mul(columns,f(CONV_IC)),ic));
          length=e.select(e.band(valid,contiguous),run,length);
        }
        auto at=[&](Value r){return SmallVector<Value>{batch,pixel,e.add(cursor,r)};};
        e.when(valid,[&] {
          Value position=e.add(e.mul(e.add(e.mul(globalBatch,f(CONV_IH)),y),f(CONV_IW)),x);
          Value address=e.add(base,e.add(e.mul(position,inputRow),ic));
          Value first=e.rem(address,lanes),end=e.add(first,length);
          Value count=e.div(e.add(end,e.idx(width/32-1)),lanes);
          if(meta)e.write(meta,e.integer(0));e.write(dma.readCtrl,e.request(e.div(address,lanes),count));
          if(notice)e.write(notice,e.integer(1,8));
          e.loop(count,[&](Value beat) {
            Value word=e.read(dma.readData);
            for(unsigned lane=0;lane<width/32;++lane) {
              Value offset=e.add(e.mul(beat,lanes),e.idx(lane));
              Value inside=e.band(b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::uge,offset,first),e.lt(offset,end));
              e.when(inside,[&] {
                Value bits=b.create<arith::TruncIOp>(loc,b.getI32Type(),b.create<arith::ShRUIOp>(loc,word,e.integer(lane*32,width)));
                Value data=b.create<arith::BitcastOp>(loc,b.getF32Type(),bits);
                b.create<memref::StoreOp>(loc,data,buffer,at(e.sub(offset,first)));
              });
            }
          });
        });
        e.when(b.create<arith::XOrIOp>(loc,valid,e.integer(1,1)),[&] {
          Value fzero=b.create<arith::ConstantOp>(loc,b.getF32FloatAttr(0));
          e.loop(length,[&](Value r){b.create<memref::StoreOp>(loc,fzero,buffer,at(r));});
        });
        auto [advance,nextIC]=splitWindowCounter(e,e.add(ic,length),f(CONV_IC));
        Value nextKW=e.add(kw,advance),wrap=e.eq(nextKW,f(CONV_KW));
        b.create<scf::YieldOp>(loc,ValueRange{e.add(cursor,length),e.select(wrap,e.add(kh,one),kh),e.select(wrap,zero,nextKW),nextIC});
        b.setInsertionPointAfter(walk);
        Value nextColumn=e.add(coords[1],one),wrapPixel=e.eq(nextColumn,f(CONV_OW));
        b.create<scf::YieldOp>(loc,ValueRange{e.select(wrapPixel,e.add(coords[0],one),coords[0]),e.select(wrapPixel,zero,nextColumn)});
      });
  });
}

static void lowerPackedViewTransfer(OpBuilder &b,Transfer &t,const DMAStreams &dma,
    unsigned width,Value meta,Value notice,ArrayRef<Value> fields,bool mergeRuns) {
  if(t.view=="conv_window_flat") {
    lowerFlatWindowTransfer(b,t,dma,width,meta,notice,fields,mergeRuns);return;
  }
  b.setInsertionPoint(t.copy);Location loc=t.copy.getLoc();PackedEmitter e{b,loc,width};
  auto offsets=t.tile.getMixedOffsets(),sizes=t.tile.getMixedSizes();
  unsigned rank=sizes.size(),axis=t.view=="conv_weights_ohwi"?2:rank-1;
  auto value=[&](OpFoldResult v){return getValueOrCreateConstantIndexOp(b,loc,v);};
  auto f=[&](unsigned field){return fields[field];};
  auto rowWords=[&](Value columns){return e.mul(e.div(e.add(columns,e.idx(width/32-1)),e.idx(width/32)),e.idx(width/32));};
  Value buffer=t.read?t.copy.getTarget():t.copy.getSource();
  Value base=e.div(f(t.baseField),e.idx(4)),length=value(sizes[axis]);
  SmallVector<Value> local(rank,e.idx(0)),global(rank);
  SmallVector<Value> strides(rank,e.idx(1));
  if(t.view=="packed" && rank>1) {
    strides[rank-2]=rowWords(f(t.shapeFields.back()));
    for(int i=int(rank)-3;i>=0;--i)strides[i]=e.mul(strides[i+1],f(t.shapeFields[i+1]));
  }
  Value inputRow;
  if(t.view!="packed")inputRow=rowWords(f(CONV_IC));
  auto at=[&](Value column) {
    SmallVector<Value> indices(local);indices[axis]=column;return indices;
  };
  auto transferRow=[&]() {
    for(unsigned d=0;d<rank;++d)global[d]=e.add(value(offsets[d]),local[d]);
    Value address=base,valid;
    if(t.view=="conv_window_nhwc") {
      Value y=e.sub(e.add(e.mul(global[1],f(CONV_SH)),e.mul(global[3],f(CONV_DH))),f(CONV_PT));
      Value x=e.sub(e.add(e.mul(global[2],f(CONV_SW)),e.mul(global[4],f(CONV_DW))),f(CONV_PL));
      // An unsigned comparison rejects both a negative coordinate and an
      // upper-bound overflow; invalid windows issue no external transaction.
      valid=e.band(e.lt(y,f(CONV_IH)),e.lt(x,f(CONV_IW)));
      Value pixel=e.add(e.mul(e.add(e.mul(global[0],f(CONV_IH)),y),f(CONV_IW)),x);
      address=e.add(base,e.add(e.mul(pixel,inputRow),global[5]));
    } else if(t.view=="conv_weights_ohwi") {
      Value row=e.add(e.mul(e.add(e.mul(global[3],f(CONV_KH)),global[0]),f(CONV_KW)),global[1]);
      address=e.add(base,e.add(e.mul(row,inputRow),global[2]));
    } else for(unsigned d=0;d<rank;++d)address=e.add(address,e.mul(global[d],strides[d]));
    auto emit=[&]() {
      Value lanes=e.idx(width/32),first=e.rem(address,lanes),start=e.div(address,lanes);
      Value end=e.add(first,length),count=e.div(e.add(end,e.idx(width/32-1)),lanes);
      Value lastEnd=e.add(e.rem(e.sub(end,e.idx(1)),lanes),e.idx(1));
      if(t.read) {
        if(meta)e.write(meta,e.integer(0));e.write(dma.readCtrl,e.request(start,count));
        if(notice)e.write(notice,e.integer(1,8));
        e.loop(count,[&](Value beat) {
          Value word=e.read(dma.readData);
          for(unsigned lane=0;lane<width/32;++lane) {
            Value offset=e.add(e.mul(beat,lanes),e.idx(lane));
            Value inside=e.band(b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::uge,offset,first),e.lt(offset,end));
            e.when(inside,[&]() {
              Value bits=b.create<arith::TruncIOp>(loc,b.getI32Type(),b.create<arith::ShRUIOp>(loc,word,e.integer(lane*32,width)));
              Value v=b.create<arith::BitcastOp>(loc,b.getF32Type(),bits);
              b.create<memref::StoreOp>(loc,v,buffer,at(e.sub(offset,first)));
            });
          }
        });
      } else {
        auto payload=[&](Value beat) {
          Value word=e.integer(0,width);
          for(unsigned lane=0;lane<width/32;++lane) {
            Value offset=e.add(e.mul(beat,lanes),e.idx(lane));
            Value inside=e.band(b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::uge,offset,first),e.lt(offset,end));
            Value bits=e.choose(inside,b.getI32Type(),[&]()->Value {
              Value v=b.create<memref::LoadOp>(loc,buffer,at(e.sub(offset,first)));
              return b.create<arith::BitcastOp>(loc,b.getI32Type(),v);
            },e.integer(0,32));
            word=e.bor(word,e.shift(b.create<arith::ExtUIOp>(loc,e.word(),bits),lane*32));
          }return word;
        };
        if(meta) {
          e.write(meta,e.bor(e.integer(1),e.bor(e.shift(e.i64(first),8),e.shift(e.i64(lastEnd),16))));
          e.write(dma.writeCtrl,e.request(start,count));if(notice)e.write(notice,e.integer(1,8));
          e.loop(count,[&](Value beat){e.write(dma.writeData,payload(beat));});
        } else e.store(dma,start,count,first,lastEnd,payload);
      }
    };
    if(valid) {
      e.when(valid,emit);
      Value invalid=b.create<arith::XOrIOp>(loc,valid,e.integer(1,1));
      e.when(invalid,[&] {
        Value zero=b.create<arith::ConstantOp>(loc,b.getF32FloatAttr(0));
        e.loop(length,[&](Value column){b.create<memref::StoreOp>(loc,zero,buffer,at(column));});
      });
    } else emit();
  };
  std::function<void(unsigned)> loops=[&](unsigned d) {
    if(d==rank) {transferRow();return;}
    if(d==axis) {loops(d+1);return;}
    e.loop(value(sizes[d]),[&](Value iv){local[d]=iv;loops(d+1);});
  };
  loops(0);
}
