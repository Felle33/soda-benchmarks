// Sufficient, conservative proof of a multiple. Failure selects the boundary
// path; it does not change/reject the tile. Covers ordinary scheduled tile IVs.
static bool packedMultiple(Value value, int64_t factor) {
  if (factor == 1)
    return true;
  if (auto n = getConstantIntValue(value))
    return *n % factor == 0;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
      if (arg == loop.getInductionVar())
        return packedMultiple(loop.getLowerBound(), factor) &&
               packedMultiple(loop.getStep(), factor);
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>())
    return packedMultiple(mul.getLhs(), factor) ||
           packedMultiple(mul.getRhs(), factor);
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return packedMultiple(add.getLhs(), factor) &&
           packedMultiple(add.getRhs(), factor);
  return false;
}
static bool packedNeedsBoundary(Transfer t, unsigned width) {
  if (t.read)
    return false;
  if (!t.tile)
    return false; // whole argument owns its row padding
  auto offsets = t.tile.getMixedOffsets();
  if (offsets.empty())
    return false;
  auto last = offsets.back();
  int64_t lanes = width / 32, length = t.tile.getStaticSizes().back();
  auto c = getConstantIntValue(last);
  bool start = c ? (*c % lanes == 0) : packedMultiple(cast<Value>(last), lanes);
  if (!start)
    return true;
  if (length % lanes == 0)
    return false;
  // Only a proven tensor end owns its padding.
  return !c || *c + length != t.tile.getSourceType().getShape().back();
}

static void lowerPackedTransfer(OpBuilder &b, Transfer &t,
                                const DMAStreams &dma, unsigned width,
                                Value meta = {}, Value notice = {},
                                ArrayRef<Value> fields = {}) {
  b.setInsertionPoint(t.copy);
  Location loc = t.copy.getLoc();
  PackedEmitter e{b, loc, width};
  Value buffer = t.read ? t.copy.getTarget() : t.copy.getSource();
  auto arg =
      t.tile ? t.tile.getSourceType() : cast<MemRefType>(buffer.getType());
  auto sizes = t.tile ? llvm::to_vector(t.tile.getStaticSizes())
                      : llvm::to_vector(arg.getShape());
  unsigned rank = sizes.size();
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  if (t.tile)
    offsets = t.tile.getMixedOffsets();
  // No request for an empty logical copy (zero is reserved for broker EOF).
  if (llvm::is_contained(sizes, int64_t(0)))
    return;
  auto strides = t.baseField < 0 ? computeSuffixProduct(arg.getShape())
                                  : SmallVector<int64_t>(rank, 1);
  uint64_t rowWords;
  if (t.baseField < 0) ::sodap::DmaTransport{width}.rowWords(rank ? arg.getShape().back() : 1,
                                        rowWords);
  if (t.baseField < 0 && rank > 1)
    for (unsigned d = 0; d + 1 < rank; ++d)
      strides[d] = strides[d] / arg.getShape().back() * rowWords;
  SmallVector<Value> ivs;
  for (unsigned d = 0; d + 1 < rank; ++d) {
    auto loop = b.create<scf::ForOp>(loc, e.idx(0), e.idx(sizes[d]), e.idx(1));
    b.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
  }
  Value addr = t.baseField < 0 ? e.idx(t.base)
                               : e.div(fields[t.baseField], e.idx(4));
  for (unsigned d = 0; d < rank; ++d) {
    Value offset = getValueOrCreateConstantIndexOp(b, loc, offsets[d]);
    if (d + 1 < rank)
      offset = e.add(offset, ivs[d]);
    Value stride = t.baseField < 0 ? e.idx(strides[d])
        : (d == 0 ? e.div(fields[t.ldField], e.idx(4)) : e.idx(1));
    addr = e.add(addr, e.mul(offset, stride));
  }
  int64_t length = rank ? sizes.back() : 1;
  Value lanes = e.idx(width / 32), first = e.rem(addr, lanes),
        start = e.div(addr, lanes);
  Value end = e.add(first, e.idx(length)),
        count = e.div(e.add(end, e.idx(width / 32 - 1)), lanes);
  Value lastEnd = e.add(e.rem(e.sub(end, e.idx(1)), lanes), e.idx(1));
  Type element = cast<MemRefType>(buffer.getType()).getElementType();
  auto payload = [&](Value beat) -> Value {
    Value packed = e.integer(0, width);
    for (unsigned l = 0; l < width / 32; ++l) {
      Value at = e.add(e.mul(beat, lanes), e.idx(l));
      Value valid = e.band(
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, at, first),
          e.lt(at, end));
      Value lane = e.choose(
          valid, b.getI32Type(),
          [&] {
            SmallVector<Value> indices(ivs);
            if (rank)
              indices.push_back(e.sub(at, first));
            Value v = b.create<memref::LoadOp>(loc, buffer, indices);
            if (element != b.getI32Type())
              v = b.create<arith::BitcastOp>(loc, b.getI32Type(), v);
            return v;
          },
          e.integer(0, 32));
      packed =
          e.bor(packed,
                e.shift(b.create<arith::ExtUIOp>(loc, e.word(), lane), l * 32));
    }
    return packed;
  };
  if (t.read) {
    if (meta)
      e.write(meta, e.integer(0));
    e.write(dma.readCtrl, e.request(start, count));
    if (notice)
      e.write(notice, e.integer(1, 8));
    e.loop(count, [&](Value beat) {
      Value packed = e.read(dma.readData);
      for (unsigned l = 0; l < width / 32; ++l) {
        Value at = e.add(e.mul(beat, lanes), e.idx(l));
        Value valid = e.band(
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::uge, at, first),
            e.lt(at, end));
        e.when(valid, [&] {
          Value bits = b.create<arith::TruncIOp>(
              loc, b.getI32Type(),
              b.create<arith::ShRUIOp>(loc, packed, e.integer(l * 32, width)));
          if (element != b.getI32Type())
            bits = b.create<arith::BitcastOp>(loc, element, bits);
          SmallVector<Value> indices(ivs);
          if (rank)
            indices.push_back(e.sub(at, first));
          b.create<memref::StoreOp>(loc, bits, buffer, indices);
        });
      }
    });
  } else if (meta) {
    e.write(meta, e.bor(e.integer(1), e.bor(e.shift(e.i64(first), 8),
                                            e.shift(e.i64(lastEnd), 16))));
    e.write(dma.writeCtrl, e.request(start, count));
    if (notice)
      e.write(notice, e.integer(1, 8));
    e.loop(count, [&](Value beat) { e.write(dma.writeData, payload(beat)); });
  } else if (packedNeedsBoundary(t, width)) {
    e.store(dma, start, count, first, lastEnd, payload);
  } else {
    // Full rows own their padding. They need no RMW or broker.
    e.write(dma.writeCtrl, e.request(start, count));
    if (notice)
      e.write(notice, e.integer(1, 8));
    e.loop(count, [&](Value beat) { e.write(dma.writeData, payload(beat)); });
  }
}
