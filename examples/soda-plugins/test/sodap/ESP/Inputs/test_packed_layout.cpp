// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ESPRuntime.h"
#include <cassert>
#include <climits>
#include <cstdio>
#include <vector>
struct Matrix {
  sodap::MemRefDescriptorHeadF32 head;
  int64_t sizes[2], strides[2];
};
int main() {
  for (unsigned w : {64, 128, 256, 512}) {
    sodap::DmaTransport transport{w};
    uint64_t row, words, packet;
    int64_t shape[] = {3, 19};
    assert(transport.tensorWords(shape, 2, words));
    assert(transport.rowWords(19, row));
    std::vector<uint32_t> source(90), roundtrip(90, 0),
        arena(words + 32, 0xa5a5a5a5);
    for (unsigned i = 0; i < source.size(); ++i)
      source[i] = 0x80000000u + i;
    source[4] = 0x7fc01234;
    source[5] = 0x7f800000;
    source[6] = 1;
    Matrix a{{reinterpret_cast<float *>(source.data()),
              reinterpret_cast<float *>(source.data()), 4},
             {3, 19},
             {23, 1}};
    Matrix b{{reinterpret_cast<float *>(roundtrip.data()),
              reinterpret_cast<float *>(roundtrip.data()), 4},
             {3, 19},
             {23, 1}};
    assert(sodap::copyMemRefF32Packed(2, &a, arena.data(), arena.size() * 4, 64,
                                      true, w));
    assert(sodap::copyMemRefF32Packed(2, &b, arena.data(), arena.size() * 4, 64,
                                      false, w));
    for (unsigned r = 0; r < 3; ++r)
      for (unsigned c = 0; c < 19; ++c)
        assert(roundtrip[4 + 23 * r + c] == source[4 + 23 * r + c]);
    for (unsigned r = 0; r < 3; ++r)
      for (unsigned c = 19; c < row; ++c)
        assert(arena[16 + r * row + c] == 0);
    for (unsigned i = 0; i < 16; ++i)
      assert(arena[i] == 0xa5a5a5a5 && arena[16 + words + i] == 0xa5a5a5a5);
    auto before = arena;
    assert(!sodap::copyMemRefF32Packed(2, &a, arena.data(), 64 + words * 4 - 1,
                                       64, true, w));
    assert(!sodap::copyMemRefF32Packed(2, &a, arena.data(), arena.size() * 4,
                                       65, true, w));
    assert(arena == before);
    assert(transport.request(0, transport.beatBytes(), packet) &&
           packet == (1ull << 32));
    assert(!transport.request(0, 0, packet));
    assert(!transport.request(1, transport.beatBytes(), packet));
    assert(!transport.request(uint64_t(UINT32_MAX) * transport.beatBytes(),
                              transport.beatBytes() * 2, packet));
    int64_t huge[] = {INT64_MAX, 19};
    assert(!transport.tensorWords(huge, 2, words));
    int64_t empty[] = {0, 19};
    assert(transport.tensorWords(empty, 2, words) && words == 0);
    uint32_t scalar = 0x80000000;
    sodap::MemRefDescriptorHeadF32 zero{reinterpret_cast<float *>(&scalar),
                                        reinterpret_cast<float *>(&scalar), 0};
    assert(sodap::copyMemRefF32Packed(0, &zero, arena.data(), arena.size() * 4,
                                      0, true, w));
    assert(arena[0] == scalar);
  }
  assert(!sodap::DmaTransport{32}.valid());
  assert(!sodap::DmaTransport{96}.valid());
  puts("packed layout PASS");
}
