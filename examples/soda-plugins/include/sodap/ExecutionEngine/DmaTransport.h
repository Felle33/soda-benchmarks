// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef SODAP_DMA_TRANSPORT_H
#define SODAP_DMA_TRANSPORT_H
#include <climits>
#include <cstdint>
namespace sodap {
// ABI 2: packed FP32, low lane first; aligned bases and padded physical rows.
// Host API uses bytes. Native DMA requests use beat index/count (32 bits each).
struct DmaTransport {
  static constexpr unsigned version = 2;
  unsigned width;
  bool valid() const {
    return width == 64 || width == 128 || width == 256 || width == 512;
  }
  uint64_t lanes() const { return width / 32; }
  uint64_t beatBytes() const { return width / 8; }
  bool rowWords(uint64_t cols, uint64_t &words) const {
    if (!valid() || cols > UINT64_MAX - (lanes() - 1))
      return false;
    words = (cols + lanes() - 1) / lanes() * lanes();
    return true;
  }
  bool tensorWords(const int64_t *shape, unsigned rank, uint64_t &words) const {
    uint64_t cols = rank ? shape[rank - 1] : 1;
    if ((rank && shape[rank - 1] < 0) || !rowWords(cols, words))
      return false;
    for (unsigned i = 0; i + 1 < rank; ++i) {
      if (shape[i] < 0 ||
          __builtin_mul_overflow(words, uint64_t(shape[i]), &words))
        return false;
    }
    return words <= uint64_t(UINT32_MAX) * lanes() &&
           words <= uint64_t(INT64_MAX) / 4;
  }
  bool request(uint64_t byteOffset, uint64_t byteLength,
               uint64_t &encoded) const {
    if (!valid() || !byteLength || byteOffset % beatBytes() ||
        byteLength % beatBytes())
      return false;
    uint64_t index = byteOffset / beatBytes(), count = byteLength / beatBytes();
    if (index > UINT32_MAX || count > UINT32_MAX ||
        count > uint64_t(UINT32_MAX) + 1 - index)
      return false;
    encoded = (count << 32) | index;
    return true;
  }
};
} // namespace sodap
#endif
