// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef SODAP_PACKED_DMA_H
#define SODAP_PACKED_DMA_H
#include "sodap/Dialect/Dataflow/Dataflow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
namespace mlir::sodap::dataflow::packed {
struct DMAStreams { Value readCtrl, readData, writeCtrl, writeData; };
struct PackedEmitter {
  OpBuilder &b;
  Location loc;
  unsigned width;
  Type word() { return b.getIntegerType(width); }
  Value idx(int64_t n) { return b.create<arith::ConstantIndexOp>(loc, n); }
  Value integer(int64_t n, unsigned bits = 64) {
    Type ty = b.getIntegerType(bits);
    return b.create<arith::ConstantOp>(
        loc, ty, b.getIntegerAttr(ty, llvm::APInt(bits, uint64_t(n), true)));
  }
  Value add(Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value mul(Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value sub(Value a, Value c) { return b.create<arith::SubIOp>(loc, a, c); }
  Value div(Value a, Value c) { return b.create<arith::DivUIOp>(loc, a, c); }
  Value rem(Value a, Value c) { return b.create<arith::RemUIOp>(loc, a, c); }
  Value eq(Value a, Value c) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, a, c);
  }
  Value lt(Value a, Value c) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, a, c);
  }
  Value band(Value a, Value c) { return b.create<arith::AndIOp>(loc, a, c); }
  Value bor(Value a, Value c) { return b.create<arith::OrIOp>(loc, a, c); }
  Value i64(Value v) {
    return b.create<arith::IndexCastUIOp>(loc, b.getI64Type(), v);
  }
  Value index(Value v) {
    return b.create<arith::IndexCastUIOp>(loc, b.getIndexType(), v);
  }
  Value read(Value ch) {
    return b
        .create<StreamReadOp>(
            loc, cast<StreamType>(ch.getType()).getElementType(), ch)
        .getResult();
  }
  void write(Value ch, Value v) {
    auto op = b.create<StreamWriteOp>(loc, ch, v);
    op->setAttr("sodap.dma_ordered", b.getUnitAttr());
  }
  Value request(Value addr, Value count) {
    return bor(b.create<arith::ShLIOp>(loc, i64(count), integer(32)),
               i64(addr));
  }
  Value shift(Value v, unsigned n) {
    return b.create<arith::ShLIOp>(
        loc, v, integer(n, cast<IntegerType>(v.getType()).getWidth()));
  }
  Value select(Value cond, Value a, Value c) {
    return b.create<arith::SelectOp>(loc, cond, a, c);
  }
  template <class F>
  void loop(Value count, F body) {
    OpBuilder::InsertionGuard guard(b);
    auto op = b.create<scf::ForOp>(loc, idx(0), count, idx(1));
    b.setInsertionPointToStart(op.getBody());
    body(op.getInductionVar());
  }
  template <class F>
  void when(Value cond, F body) {
    OpBuilder::InsertionGuard guard(b);
    auto op = b.create<scf::IfOp>(loc, cond, false);
    b.setInsertionPointToStart(&op.getThenRegion().front());
    body();
  }
  template <class F>
  Value choose(Value cond, Type type, F yes, Value no) {
    OpBuilder::InsertionGuard guard(b);
    auto op = b.create<scf::IfOp>(loc, TypeRange{type}, cond, true);
    b.setInsertionPointToStart(&op.getThenRegion().front());
    b.create<scf::YieldOp>(loc, yes());
    b.setInsertionPointToStart(&op.getElseRegion().front());
    b.create<scf::YieldOp>(loc, no);
    return op.getResult(0);
  }
  Value mask(Value first, Value end) {
    // Avoid shifts by W (undefined in LLVM) at either full-beat endpoint.
    Value out = integer(0, width);
    for (unsigned l = 0; l < width / 32; ++l) {
      Value valid = band(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge,
                                                 idx(l), first),
                         lt(idx(l), end));
      llvm::APInt bits = llvm::APInt::getBitsSet(width, l * 32, l * 32 + 32);
      Value lane = b.create<arith::ConstantOp>(loc, word(),
                                               b.getIntegerAttr(word(), bits));
      out = bor(out, select(valid, lane, integer(0, width)));
    }
    return out;
  }
  // One atomic write transaction. Boundary reads precede publication of the
  // write request. The only caller in a concurrent design is the single DMA
  // owner below: no other transaction can interleave with this operation.
  template <class F>
  void store(const DMAStreams &dma, Value addr, Value count, Value first,
             Value end, F payload) {
    Value last = sub(count, idx(1)), lanes = idx(width / 32);
    Value head = mask(first, lanes), tail = mask(idx(0), end);
    auto fetch = [&](Value at) {
      write(dma.readCtrl, request(at, idx(1)));
      return read(dma.readData);
    };
    Value oldFirst = choose(
        lt(idx(0), first), word(), [&] { return fetch(addr); },
        integer(0, width));
    // If one beat is shared at both ends, read it just once.
    Value oldLast = choose(
        lt(end, lanes), word(),
        [&] {
          return choose(
              band(eq(last, idx(0)), lt(idx(0), first)), word(),
              [&] { return oldFirst; }, integer(0, width));
        },
        integer(0, width));
    Value needLast = band(
        lt(end, lanes),
        b.create<arith::XOrIOp>(loc, band(eq(last, idx(0)), lt(idx(0), first)),
                                integer(1, 1)));
    oldLast = choose(
        needLast, word(), [&] { return fetch(add(addr, last)); }, oldLast);
    write(dma.writeCtrl, request(addr, count));
    loop(count, [&](Value beat) {
      Value keep = band(select(eq(beat, idx(0)), head, integer(-1, width)),
                        select(eq(beat, last), tail, integer(-1, width)));
      Value old = bor(select(eq(beat, idx(0)), oldFirst, integer(0, width)),
                      select(eq(beat, last), oldLast, integer(0, width)));
      Value value = bor(
          band(payload(beat), keep),
          band(old, b.create<arith::XOrIOp>(loc, keep, integer(-1, width))));
      write(dma.writeData, value);
    });
  }
};

struct PackedClient {
  DMAStreams dma;
  Value meta, notice;
  TaskOp task;
};
// A finite round-robin broker exists only if stores can touch valid neighbors.
// Each client admits its receive buffer or acquires a ready output token before
// submitting a command. A zero descriptor terminates that client. Requests and
// all payload writes wait for acceptance. The broker owns the physical ports
// until the complete burst drains, including both RMW boundary reads.
inline DenseMap<Operation *, PackedClient>
packedBroker(OpBuilder &b, func::FuncOp top, const DMAStreams &physical,
             ArrayRef<TaskOp> tasks, unsigned width,
             DenseMap<Operation *, PackedClient> result = DenseMap<Operation *, PackedClient>()) {
  Location loc = top.getLoc();
  if (result.empty()) b.setInsertionPoint(tasks.front());
  else b.setInsertionPointAfter(tasks.back());
  auto channel = [&](unsigned bits) {
    return b
        .create<StreamOp>(
            loc, StreamType::get(b.getContext(), b.getIntegerType(bits), 1))
        .getResult();
  };
  for (TaskOp task : tasks) {
    if (result.count(task)) continue;
    Value ctrl = channel(64), rd = channel(width), wd = channel(width),
          meta = channel(64), notice = channel(8);
    result[task] = {{ctrl, rd, ctrl, wd}, meta, notice, task};
  }
  auto broker = b.create<TaskOp>(loc, TypeRange{});
  broker->setAttr("sodap.dma_broker", b.getUnitAttr());
  if (broker.getBody().empty())
    b.createBlock(&broker.getBody());
  b.setInsertionPointToStart(&broker.getBody().front());
  PackedEmitter e{b, loc, width};
  Value done = b.create<memref::AllocaOp>(
      loc, MemRefType::get({int64_t(tasks.size())}, b.getI1Type()));
  for (unsigned i = 0; i < tasks.size(); ++i)
    b.create<memref::StoreOp>(loc, e.integer(0, 1), done, e.idx(i));
  auto loop = b.create<scf::WhileOp>(loc, TypeRange{b.getIndexType()},
                                     ValueRange{e.idx(tasks.size())});
  b.createBlock(&loop.getBefore(), {}, TypeRange{b.getIndexType()},
                SmallVector<Location>{loc});
  b.create<scf::ConditionOp>(
      loc, e.lt(e.idx(0), loop.getBefore().front().getArgument(0)),
      loop.getBefore().front().getArguments());
  b.createBlock(&loop.getAfter(), {}, TypeRange{b.getIndexType()},
                SmallVector<Location>{loc});
  Value remaining = loop.getAfter().front().getArgument(0);
  for (unsigned i = 0; i < tasks.size(); ++i) {
    TaskOp task = tasks[i];
    PackedClient c = result[task];
    Value active = b.create<arith::XOrIOp>(
        loc, b.create<memref::LoadOp>(loc, done, e.idx(i)), e.integer(1, 1));
    Value ready = b.create<StreamAvailableOp>(loc, b.getI1Type(), c.notice);
    remaining = e.choose(
        e.band(active, ready), b.getIndexType(),
        [&] {
          e.read(c.notice);
          Value request = e.read(c.dma.readCtrl);
          return e.choose(
              e.eq(request, e.integer(0)), b.getIndexType(),
              [&] {
                b.create<memref::StoreOp>(loc, e.integer(1, 1), done, e.idx(i));
                return e.sub(remaining, e.idx(1));
              },
              e.choose(
                  b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                          request, e.integer(0)),
                  b.getIndexType(),
                  [&] {
                    Value meta = e.read(c.meta);
                    Value addr =
                              e.index(e.band(request, e.integer(UINT32_MAX))),
                          count = e.index(b.create<arith::ShRUIOp>(
                              loc, request, e.integer(32)));
                    Value writing =
                        e.eq(e.band(meta, e.integer(1)), e.integer(1));
                    e.when(writing, [&] {
                      Value first = e.index(e.band(
                          b.create<arith::ShRUIOp>(loc, meta, e.integer(8)),
                          e.integer(255)));
                      Value end = e.index(e.band(
                          b.create<arith::ShRUIOp>(loc, meta, e.integer(16)),
                          e.integer(255)));
                      e.store(physical, addr, count, first, end,
                              [&](Value) { return e.read(c.dma.writeData); });
                    });
                    e.when(
                        b.create<arith::XOrIOp>(loc, writing, e.integer(1, 1)),
                        [&] {
                          e.write(physical.readCtrl, request);
                          e.loop(count, [&](Value) {
                            e.write(c.dma.readData, e.read(physical.readData));
                          });
                        });
                    return remaining;
                  },
                  remaining));
        },
        remaining);
  }
  b.create<scf::YieldOp>(loc, remaining);
  b.setInsertionPointAfter(loop);
  b.create<YieldOp>(loc);
  for (auto task : tasks) {
    b.setInsertionPoint(task.getBody().front().getTerminator());
    Value eof = e.integer(0);
    eof.getDefiningOp()->setAttr("sodap.dma_eof", b.getUnitAttr());
    e.write(result[task].dma.readCtrl, eof);
    e.write(result[task].notice, e.integer(1, 8));
  }
  return result;
}


// Streaming scalar access inside generated operator tasks. The one-beat
// registers are explicit compiler-generated pack/extract state, not wrapper
// payload buffers. Arbitrary dynamic tails use the same single-owner broker.
struct PackedAccess {
  PackedEmitter e;
  PackedClient client;
  Value readPos, readStep, readCache, readLoaded;
  Value writePos, writeLeft, writeCache;
  PackedAccess(OpBuilder &b, Location loc, unsigned width, PackedClient client)
      : e{b,loc,width}, client(client) {
    auto scalar = [&](Type type) -> Value {
      return b.create<memref::AllocaOp>(loc,MemRefType::get({},type));
    };
    readPos=scalar(b.getIndexType());readStep=scalar(b.getIndexType());
    readCache=scalar(e.word());readLoaded=scalar(b.getI1Type());
    writePos=scalar(b.getIndexType());writeLeft=scalar(b.getIndexType());
    writeCache=scalar(e.word());
  }
  Value load(Value v) { return e.b.create<memref::LoadOp>(e.loc,v,ValueRange{}); }
  void save(Value value,Value v) { e.b.create<memref::StoreOp>(e.loc,value,v,ValueRange{}); }
  void request(bool writing,Value byteAddress,Value elements,bool descriptor=false) {
    Value lanes=e.idx(e.width/32),words=e.div(byteAddress,e.idx(4));
    Value first=e.rem(words,lanes),step=e.idx(descriptor?2:1);
    Value end=e.add(first,e.mul(elements,step));
    Value count=e.div(e.add(end,e.idx(e.width/32-1)),lanes);
    Value command=e.request(e.div(words,lanes),count);
    if(writing) {
      save(first,writePos);save(elements,writeLeft);save(e.integer(0,e.width),writeCache);
      Value last=e.add(e.rem(e.sub(end,e.idx(1)),lanes),e.idx(1));
      e.write(client.meta,e.bor(e.integer(1),e.bor(e.shift(e.i64(first),8),e.shift(e.i64(last),16))));
      e.write(client.dma.writeCtrl,command);
    } else {
      save(first,readPos);save(step,readStep);save(e.integer(0,1),readLoaded);
      save(e.integer(0,e.width),readCache);
      e.write(client.meta,e.integer(0));e.write(client.dma.readCtrl,command);
    }
    e.write(client.notice,e.integer(1,8));
  }
  Value read() {
    Value pos=load(readPos),lane=e.rem(pos,e.idx(e.width/32));
    Value fresh=e.bor(e.eq(lane,e.idx(0)),e.eq(load(readLoaded),e.integer(0,1)));
    Value beat=e.choose(fresh,e.word(),[&]{return e.read(client.dma.readData);},load(readCache));
    save(beat,readCache);save(e.integer(1,1),readLoaded);save(e.add(pos,load(readStep)),readPos);
    Value shift=e.b.create<arith::IndexCastUIOp>(e.loc,e.word(),e.mul(lane,e.idx(32)));
    return e.b.create<arith::TruncIOp>(e.loc,e.b.getI32Type(),
        e.b.create<arith::ShRUIOp>(e.loc,beat,shift));
  }
  void write(Value bits) {
    Value pos=load(writePos),lane=e.rem(pos,e.idx(e.width/32)),left=load(writeLeft);
    Value shift=e.b.create<arith::IndexCastUIOp>(e.loc,e.word(),e.mul(lane,e.idx(32)));
    Value beat=e.bor(load(writeCache),e.b.create<arith::ShLIOp>(e.loc,
        e.b.create<arith::ExtUIOp>(e.loc,e.word(),bits),shift));
    save(beat,writeCache);
    e.when(e.bor(e.eq(lane,e.idx(e.width/32-1)),e.eq(left,e.idx(1))),[&]{
      e.write(client.dma.writeData,beat);save(e.integer(0,e.width),writeCache);
    });
    save(e.add(pos,e.idx(1)),writePos);save(e.sub(left,e.idx(1)),writeLeft);
  }
};

inline PackedClient createClient(OpBuilder &b,Location loc,unsigned width) {
  auto channel=[&](unsigned bits) -> Value {return b.create<StreamOp>(loc,
      StreamType::get(b.getContext(),b.getIntegerType(bits),1));};
  Value ctrl=channel(64),rd=channel(width),wd=channel(width),meta=channel(64),notice=channel(8);
  return {{ctrl,rd,ctrl,wd},meta,notice,{}};
}

} // namespace mlir::sodap::dataflow::packed
#endif
