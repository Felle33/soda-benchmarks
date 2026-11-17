// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include "sodap/ExecutionEngine/ESPRuntime.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#ifdef SODAP_OP_PROFILE
#include "sodap/ExecutionEngine/esp_prof.h"
#endif

#if defined(SODAP_OP_CPU_ONLY) && (defined(SODAP_OP_EXTERNAL_EXECUTOR) || defined(SODAP_OP_VERIFY_RTL))
#error CPU-only execution cannot be combined with an RTL executor
#endif

#if defined(__riscv) && !defined(SODAP_OP_CPU_ONLY)
#define SODAP_OP_ESP_HW
extern "C" {
#include <esp_accelerator.h>
#include <esp_probe.h>
}
#if !defined(SODAP_ESP_DEVICE_ID) && !defined(SODAP_OP_SEPARATE_DEVICES)
#error Define SODAP_ESP_DEVICE_ID and SODAP_ESP_DEVICE_NAME for the operator socket
#endif
#endif

namespace {
struct ProfileRegion {
#ifdef SODAP_OP_PROFILE
  unsigned id;
  explicit ProfileRegion(unsigned id):id(id) {esp_prof_begin(id);}
  ~ProfileRegion() {esp_prof_end(id);}
#else
  explicit ProfileRegion(unsigned) {}
#endif
};
#ifdef SODAP_OP_QUIET
#define OP_LOG(...) ((void)0)
#else
#define OP_LOG(...) std::printf(__VA_ARGS__)
#endif
#ifdef SODAP_OP_CPU_ONLY
constexpr const char *executionPrefix = "CPU_OP";
#else
constexpr const char *executionPrefix = "REUSABLE_OP";
#endif
// Baremetal software inference must use the cacheable lower DRAM half,
// rather than the uncached DMA allocator. The linker bounds this BSS arena.
#ifdef SODAP_OP_CPU_ARENA_BYTES
#ifndef SODAP_OP_CPU_ONLY
#error A CPU arena requires SODAP_OP_CPU_ONLY
#endif
alignas(64) unsigned char cpuArena[SODAP_OP_CPU_ARENA_BYTES];
size_t cpuArenaUsed = 0;
#endif
#if defined(__riscv) && defined(SODAP_OP_CPU_ONLY) && !defined(SODAP_OP_CPU_ARENA_BYTES)
#error Specify SODAP_OP_CPU_ARENA_BYTES for baremetal CPU execution
#endif
struct TensorLayout {
  int64_t offset;
  uint64_t elements,columns,rowWords,bytes;
  TensorLayout *next;
};
struct Context {
  uint32_t *arena;
  uint32_t bytes, instance, invocations, dmaWidth;
  TensorLayout *layouts, *recent[4];
  unsigned recentIndex;
  bool running;
  bool separateDevices;
  uint32_t operatorInstances[3]; // conv, add, gemm
  int64_t outputOffset;
  uint64_t outputElements;
#if defined(SODAP_OP_VERIFY_RTL) && !defined(__riscv)
  uint32_t *savedOutput;
#endif
  Context *next;
#ifdef SODAP_OP_ESP_HW
  esp_device *devices, *device;
  esp_device *bindings[3], *enumerations[3];
  unsigned enumerationCounts[3], pageChunks, pageShift;
  unsigned **pages;
  bool flush;
#endif
};
Context *live = nullptr;
[[noreturn]] void fail(const char *why) {
  std::printf("REUSABLE_OP TEST FAILED: %s\n", why);
  std::exit(1);
}
void *allocate(uint32_t size) {
#ifdef SODAP_OP_CPU_ARENA_BYTES
  size_t aligned = (size_t(size) + 63) & ~size_t(63);
  if (aligned > sizeof(cpuArena) - cpuArenaUsed) fail("CPU arena exhausted");
  void *p = cpuArena + cpuArenaUsed;
  cpuArenaUsed += aligned;
#elif defined(__riscv)
  void *p = aligned_malloc(size);
#else
  void *p = std::malloc(size);
#endif
  if (!p) fail("allocation failed");
  return p;
}
void release(void *p) {
#ifdef SODAP_OP_CPU_ARENA_BYTES
  (void)p; // Reset collectively when the last context is closed.
#elif defined(__riscv)
  aligned_free(p);
#else
  std::free(p);
#endif
}
Context &lookup(int64_t handle) {
  for (auto *p = live; p; p = p->next)
    if (reinterpret_cast<intptr_t>(p) == handle) return *p;
  fail("invalid context handle");
}
TensorLayout &layout(Context &ctx,int64_t offset) {
  for(auto *p:ctx.recent) if(p && p->offset==offset)return *p;
  for(auto *p=ctx.layouts;p;p=p->next)if(p->offset==offset) {
    ctx.recent[ctx.recentIndex++%4]=p;return *p;
  }
  fail("packed tensor layout is not registered");
}
void columns(Context &ctx,int64_t offset,uint64_t count) {
  if(ctx.dmaWidth && layout(ctx,offset).columns!=count)
    fail("operator shape disagrees with packed tensor row layout");
}
uint64_t product(std::initializer_list<int64_t> dimensions) {
  uint64_t count = 1;
  for (int64_t extent : dimensions) {
    if (extent <= 0 || uint64_t(extent) > (1u << 27) / count)
      fail("invalid or oversized tensor dimensions");
    count *= extent;
  }
  return count;
}
uint64_t check(Context &ctx, int64_t offset, uint64_t elements) {
  if(ctx.dmaWidth) {
    auto &t=layout(ctx,offset);
    if(ctx.running || elements>t.elements)fail("packed tensor exceeds registered layout");
    return offset+t.bytes;
  }
  if (ctx.running || offset < 0 || offset % OP_SLOT_BYTES ||
      uint64_t(offset) > ctx.bytes - OP_HEADER_BYTES ||
      elements > (ctx.bytes - OP_HEADER_BYTES - offset) / OP_SLOT_BYTES)
    fail("tensor exceeds arena or is misaligned");
  return offset + elements * OP_SLOT_BYTES;
}
void disjoint(int64_t a, uint64_t ae, int64_t c, uint64_t ce) {
  if (uint64_t(a) < ce && uint64_t(c) < ae) fail("output overlaps an input");
}
void flag(int64_t value) {
  if (value != 0 && value != 1) fail("invalid boolean parameter");
}
void spatial(std::initializer_list<int64_t> values, bool allowZero = false) {
  for (auto v : values)
    if (v < (allowZero ? 0 : 1) || v > (1 << 20))
      fail("invalid spatial parameter");
}
uint32_t wordAddress(int64_t offset) { return (offset + OP_HEADER_BYTES) / 4; }
uint32_t wireAddress(Context &ctx,int64_t offset) {
  return ctx.dmaWidth ? offset+OP_HEADER_BYTES : wordAddress(offset);
}
uint64_t elementAddress(Context &ctx,int64_t offset,uint64_t index) {
  if(!ctx.dmaWidth)return wordAddress(offset)+index*2;
  auto &t=layout(ctx,offset);
  if(index>=t.elements)fail("packed tensor index exceeds layout");
  return wordAddress(offset)+(index/t.columns)*t.rowWords+index%t.columns;
}
float read(Context &ctx, int64_t offset, uint64_t index) {
  float value;
  // Explicit aligned words also matter for CPU fallback on an FPGA: byte
  // accesses to the uncached DMA arena each incur a separate transaction.
  uint32_t bits = ctx.arena[elementAddress(ctx,offset,index)];
  std::memcpy(&value, &bits, 4);
  return value;
}
void write(Context &ctx, int64_t offset, uint64_t index, float value) {
  auto *p = ctx.arena + elementAddress(ctx,offset,index);
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  p[0] = bits;
  if(!ctx.dmaWidth)p[1] = 0;
}
// Construct only after check()/disjoint(): metadata and bounds are validated
// once per CPU operation, rather than rediscovered for every scalar access.
struct TensorView {
  uint32_t *base;
  uint64_t columns,rowWords,step;
  TensorView(Context &ctx,int64_t offset,uint64_t elements)
      : base(ctx.arena+wordAddress(offset)),columns(elements),
        rowWords(elements*2),step(ctx.dmaWidth?1:2) {
    if(ctx.dmaWidth) {
      auto &t=layout(ctx,offset);columns=t.columns;rowWords=t.rowWords;
    }
  }
  bool dense() const { return rowWords==columns*step; }
  uint64_t address(uint64_t index) const {
    return dense()?index*step:(index/columns)*rowWords+(index%columns)*step;
  }
  bool rowCompatible(uint64_t cols) const {return dense() || columns==cols;}
  uint64_t stride(uint64_t cols) const {return dense()?cols*step:rowWords;}
  float get(uint64_t at) const {
    uint32_t bits=base[at];float value;std::memcpy(&value,&bits,4);return value;
  }
  void put(uint64_t at,float value) const {
    uint32_t bits;std::memcpy(&bits,&value,4);base[at]=bits;
    if(step==2)base[at+1]=0;
  }
};
void copyRange(const TensorView &src,uint64_t si,const TensorView &dst,uint64_t di,uint64_t count) {
  while(count) {
    uint64_t chunk=count;
    if(!src.dense())chunk=std::min(chunk,src.columns-si%src.columns);
    if(!dst.dense())chunk=std::min(chunk,dst.columns-di%dst.columns);
    auto *s=src.base+src.address(si),*d=dst.base+dst.address(di);
    for(uint64_t i=0;i<chunk;++i,s+=src.step,d+=dst.step) {
      *d=*s;if(dst.step==2)d[1]=0;
    }
    si+=chunk;di+=chunk;count-=chunk;
  }
}
float activation(float value, int64_t relu) {
  if (!relu) return value;
  return value < 0 ? 0 : value > FLT_MAX ? FLT_MAX : value;
}
void header(Context &ctx, uint32_t magic) {
  std::memset(ctx.arena, 0, OP_HEADER_BYTES);
  ctx.arena[0] = magic;
  ctx.arena[2] = ctx.dmaWidth ? OP_PACKED_VERSION : OP_VERSION;
  if(ctx.dmaWidth)ctx.arena[2*OP_DMA_WIDTH]=ctx.dmaWidth;
}
void field(Context &ctx, unsigned index, uint32_t value) { ctx.arena[index * 2] = value; }
bool launch(Context &ctx, const char *kind, int64_t outputOffset, uint64_t outputElements) {
  ctx.running = true;
  ctx.outputOffset = outputOffset;
  ctx.outputElements = outputElements;
#if defined(SODAP_OP_VERIFY_RTL) && !defined(__riscv)
  if(ctx.arena[0]==GEMM_MAGIC && ctx.arena[2*GEMM_FIELD_HAS_C]==0x3f800000) {
    ctx.savedOutput=static_cast<uint32_t *>(allocate(outputElements*4));
    for(uint64_t i=0;i<outputElements;++i)
      ctx.savedOutput[i]=ctx.arena[elementAddress(ctx,outputOffset,i)];
  }
#endif
#ifdef SODAP_OP_TRACE
  std::printf("OP_CONFIG call=%u",ctx.invocations);
  for(unsigned i=0;i<24;++i) std::printf(" %x",ctx.arena[2*i]);
  std::printf("\n");
#endif
  unsigned family=ctx.arena[0]==OP_CONV_MAGIC?0:ctx.arena[0]==OP_ADD_MAGIC?1:2;
  OP_LOG("%s launch=%u kind=%s instance=%u socket=%u\n",
              executionPrefix, ctx.invocations, kind, ctx.operatorInstances[family], ctx.instance);
#ifndef __riscv
  std::fflush(stdout);
#endif
#ifdef SODAP_OP_ESP_HW
  if(ctx.separateDevices) {
    unsigned index=ctx.arena[0]==OP_CONV_MAGIC?0:ctx.arena[0]==OP_ADD_MAGIC?1:2;
    ctx.device=ctx.bindings[index];
  }
  if (ctx.flush) esp_flush(ACC_COH_NONE);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  iowrite32(ctx.device, CMD_REG, CMD_MASK_START);
  while (!(ioread32(ctx.device, STATUS_REG) & STATUS_MASK_DONE)) {}
  iowrite32(ctx.device, CMD_REG, 0);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  return true;
#elif defined(SODAP_OP_EXTERNAL_EXECUTOR) && !defined(SODAP_OP_VERIFY_RTL) && !defined(SODAP_OP_CPU_ONLY)
  esp_op_execute_rtl(ctx.arena, ctx.bytes);
  return true;
#else
  return false;
#endif
}
void complete(Context &ctx) {
#if defined(SODAP_OP_VERIFY_RTL) && !defined(__riscv)
  // In verification mode launch() requests the native oracle first. Preserve
  // its output, poison C, then run RTL. Every subsequent graph node consumes
  // the RTL result; the oracle is used only for the per-invocation assertion.
  auto *expected=static_cast<float *>(allocate(ctx.outputElements*sizeof(float)));
  for(uint64_t i=0;i<ctx.outputElements;++i) {
    expected[i]=read(ctx,ctx.outputOffset,i);
    ctx.arena[elementAddress(ctx,ctx.outputOffset,i)]=ctx.savedOutput?ctx.savedOutput[i]:0x7fc00000;
  }
  esp_op_execute_rtl(ctx.arena,ctx.bytes);
  float maxError=0;
  for(uint64_t i=0;i<ctx.outputElements;++i) {
    float actual=read(ctx,ctx.outputOffset,i), error=std::fabs(actual-expected[i]);
    bool sameNaN=std::isnan(actual) && std::isnan(expected[i]);
    if(!sameNaN && (!std::isfinite(actual) || !std::isfinite(expected[i]) || error>1e-4f+1e-4f*std::fabs(expected[i]))) {
      std::printf("RTL mismatch invocation=%u element=%lu actual=%g expected=%g\n",
                  ctx.invocations,(unsigned long)i,double(actual),double(expected[i]));
      fail("RTL output differs from operator oracle");
    }
    maxError=std::max(maxError,error);
  }
  release(expected);
  release(ctx.savedOutput);ctx.savedOutput=nullptr;
  std::printf("REUSABLE_OP RTL_CHECK elements=%lu max_error=%g\n",
              (unsigned long)ctx.outputElements,double(maxError));
#endif
#ifdef SODAP_OP_TRACE
  std::printf("OP_OUTPUT call=%u",ctx.invocations);
  for(uint64_t i=0;i<std::min(ctx.outputElements,uint64_t(16));++i)
    std::printf(" %x",ctx.arena[elementAddress(ctx,ctx.outputOffset,i)]);
  std::printf("\n");
#endif
  ctx.running = false;
  ++ctx.invocations;
  OP_LOG("%s complete=%u socket=%u\n", executionPrefix, ctx.invocations, ctx.instance);
#ifndef __riscv
  std::fflush(stdout);
#endif
}
void copy(int64_t handle, int64_t offset, int64_t rank, void *desc, bool in) {
  Context &ctx = lookup(handle);
  check(ctx, offset, 0);
  if(ctx.dmaWidth) {
    auto &t=layout(ctx,offset);
    if(!sodap::copyMemRefF32Packed(rank,desc,ctx.arena,
        OP_HEADER_BYTES+offset+t.bytes,OP_HEADER_BYTES+offset,in,ctx.dmaWidth,t.columns))
      fail("invalid packed tensor copy");
  } else if (!sodap::copyMemRefF32(rank, desc, ctx.arena, ctx.bytes / 4,
                           wordAddress(offset), in, 2)) fail("invalid tensor copy");
}
} // namespace

static int64_t openContext(int64_t bytes,int64_t instance,unsigned width,
    bool separate=false,int64_t conv=0,int64_t add=0,int64_t gemm=0) {
  ProfileRegion allocation(5); // ESP_PROF_ALLOC: context, arena and page table.
  if(width && !sodap::DmaTransport{width}.valid())fail("invalid DMA width");
#ifdef SODAP_OP_ESP_HW
  if(width && width!=64)fail("ESP backend supports packed DMA64 only");
#endif
  if (bytes <= 0 || bytes > (1 << 30) - OP_HEADER_BYTES || bytes % (width?width/8:8) ||
      instance < 0 || instance > UINT32_MAX) fail("invalid arena or instance");
  for (auto *p = live; p; p = p->next)
    if (p->instance == instance) fail("socket already owned");
  auto *ctx = static_cast<Context *>(allocate(sizeof(Context)));
  *ctx = {};
  ctx->bytes = bytes + OP_HEADER_BYTES;
  ctx->instance = instance;
  ctx->dmaWidth=width;
  ctx->separateDevices=separate;
  int64_t indices[]={conv,add,gemm};
  for(unsigned i=0;i<3;++i) {
    if(indices[i]<-1 || indices[i]>INT32_MAX)fail("invalid device instance");
    ctx->operatorInstances[i]=indices[i];
  }
#ifdef SODAP_OP_ESP_HW
  if(separate) {
#ifdef SODAP_OP_SEPARATE_DEVICES
    const unsigned ids[]={SODAP_CONV_DEVICE_ID,SODAP_ADD_DEVICE_ID,SODAP_GEMM_DEVICE_ID};
    const char *names[]={SODAP_CONV_DEVICE_NAME,SODAP_ADD_DEVICE_NAME,SODAP_GEMM_DEVICE_NAME};
    for(unsigned i=0;i<3;++i) {
      if(indices[i]<0)continue;
      int count=probe(&ctx->enumerations[i],VENDOR_SLD,ids[i],names[i]);
      if(count<=0 || indices[i]>=count)fail("typed operator socket not found");
      ctx->enumerationCounts[i]=count;
      ctx->bindings[i]=&ctx->enumerations[i][indices[i]];
    }
    for(auto *device:ctx->bindings)if(device) {ctx->device=device;break;}
    if(!ctx->device)fail("network has no accelerator binding");
#else
    fail("runtime was not built for separate operator devices");
#endif
  } else {
#ifdef SODAP_ESP_DEVICE_ID
    int count = probe(&ctx->devices, VENDOR_SLD, SODAP_ESP_DEVICE_ID, SODAP_ESP_DEVICE_NAME);
    if (instance >= count) fail("operator socket not found");
    ctx->device = &ctx->devices[instance];
#else
    fail("runtime was not built for a combined operator device");
#endif
  }
  unsigned shift = 20;
  unsigned capacity = ioread32(ctx->device, PT_NCHUNK_MAX_REG);
  if(separate)for(auto *device:ctx->bindings)
    if(device)capacity=std::min(capacity,ioread32(device,PT_NCHUNK_MAX_REG));
  if (!capacity) fail("missing DMA page table capacity");
  while ((uint64_t(ctx->bytes) + (1u << shift) - 1) >> shift > capacity) ++shift;
  unsigned chunk = 1u << shift, chunks = (uint64_t(ctx->bytes) + chunk - 1) >> shift;
  ctx->pageChunks=chunks;ctx->pageShift=shift;
  ctx->arena = static_cast<uint32_t *>(allocate(chunks * chunk));
  ctx->pages = static_cast<unsigned **>(allocate(chunks * sizeof(unsigned *)));
  for (unsigned i = 0; i < chunks; ++i) ctx->pages[i] = ctx->arena + i * (chunk / 4);
  esp_device *cache = nullptr;
  int llc = probe(&cache, VENDOR_CACHE, DEVID_LLC_CACHE, DEVNAME_LLC_CACHE);
  release(cache);
  int l2 = probe(&cache, VENDOR_CACHE, DEVID_L2_CACHE, DEVNAME_L2_CACHE);
  release(cache);
  ctx->flush = llc > 0 || l2 > 0;
  for(unsigned i=0;i<(separate?3u:1u);++i) {
    auto *device=separate?ctx->bindings[i]:ctx->device;
    if(!device)continue;
    iowrite32(device, COHERENCE_REG, ACC_COH_NONE);
    iowrite32(device, PT_ADDRESS_REG, reinterpret_cast<uintptr_t>(ctx->pages));
    iowrite32(device, PT_NCHUNK_REG, chunks);
    iowrite32(device, PT_SHIFT_REG, shift);
    iowrite32(device, SRC_OFFSET_REG, 0);
    iowrite32(device, DST_OFFSET_REG, 0);
  }
#else
  ctx->arena = static_cast<uint32_t *>(allocate(ctx->bytes));
#endif
  std::memset(ctx->arena, 0, ctx->bytes);
  ctx->next = live;
  live = ctx;
  OP_LOG("%s open socket=%u arena_bytes=%u engines=conv:1,add:1,gemm:1 separate=%u\n", executionPrefix, ctx->instance, ctx->bytes, separate);
  return reinterpret_cast<intptr_t>(ctx);
}
extern "C" int64_t esp_op_open(int64_t bytes,int64_t instance) {
  return openContext(bytes,instance,0);
}
extern "C" int64_t esp_op_open_packed(int64_t bytes,int64_t instance,int64_t width) {
  if(width<=0 || width>UINT32_MAX || !sodap::DmaTransport{unsigned(width)}.valid())fail("invalid DMA width");
  return openContext(bytes,instance,unsigned(width));
}
extern "C" int64_t esp_op_open_devices(int64_t bytes,int64_t width,
    int64_t conv,int64_t add,int64_t gemm) {
  if(width<=0 || width>UINT32_MAX || !sodap::DmaTransport{unsigned(width)}.valid())fail("invalid DMA width");
#if defined(SODAP_OP_ESP_HW) && !defined(SODAP_OP_SEPARATE_DEVICES)
  if((conv>=0)+(add>=0)+(gemm>=0)!=1)fail("single-device runtime requires one operator family");
  return openContext(bytes,std::max({conv,add,gemm}),unsigned(width),false,conv,add,gemm);
#else
  return openContext(bytes,0,unsigned(width),true,conv,add,gemm);
#endif
}
// Mapping is explicit and independent of the operator's arithmetic contract.
// The default experiment binds one instance per type; another graph may select
// additional compatible physical instances between synchronous invocations.
extern "C" void esp_op_select_device(int64_t h,int64_t family,int64_t instance) {
  Context &ctx=lookup(h);
  if(ctx.running || family<0 || family>=3 || instance<0 || uint64_t(instance)>UINT32_MAX)
    fail("invalid typed device binding");
  if(ctx.operatorInstances[family]==uint32_t(instance))return;
  if(!ctx.separateDevices)fail("single-device runtime cannot change operator binding");
#ifdef SODAP_OP_ESP_HW
  if(uint64_t(instance)>=ctx.enumerationCounts[family])fail("typed operator instance not found");
  auto *device=&ctx.enumerations[family][instance];
  if(ioread32(device,PT_NCHUNK_MAX_REG)<ctx.pageChunks)fail("selected socket page table too small");
  iowrite32(device,COHERENCE_REG,ACC_COH_NONE);
  iowrite32(device,PT_ADDRESS_REG,reinterpret_cast<uintptr_t>(ctx.pages));
  iowrite32(device,PT_NCHUNK_REG,ctx.pageChunks);
  iowrite32(device,PT_SHIFT_REG,ctx.pageShift);
  iowrite32(device,SRC_OFFSET_REG,0);iowrite32(device,DST_OFFSET_REG,0);
  ctx.bindings[family]=device;
#endif
  ctx.operatorInstances[family]=instance;
}

extern "C" void esp_op_tensor(int64_t handle,int64_t offset,int64_t elements,int64_t cols) {
  Context &ctx=lookup(handle);
  if(!ctx.dmaWidth || ctx.running || offset<0 || offset%(ctx.dmaWidth/8) ||
      elements<=0 || cols<=0 || elements%cols)fail("invalid packed tensor registration");
  uint64_t rowWords,bytes;
  if(!sodap::DmaTransport{ctx.dmaWidth}.rowWords(cols,rowWords) || rowWords>ctx.bytes/4 ||
      __builtin_mul_overflow(uint64_t(elements/cols),rowWords*4,&bytes) ||
      uint64_t(offset)>ctx.bytes-OP_HEADER_BYTES || bytes>ctx.bytes-OP_HEADER_BYTES-offset)
    fail("packed tensor registration exceeds arena");
  TensorLayout *entry=ctx.layouts;
  while(entry && entry->offset!=offset)entry=entry->next;
  if(!entry) {
    ProfileRegion allocation(5);
    entry=static_cast<TensorLayout *>(allocate(sizeof(TensorLayout)));
    entry->next=ctx.layouts;ctx.layouts=entry;
  }
  entry->offset=offset;entry->elements=elements;entry->columns=cols;
  entry->rowWords=rowWords;entry->bytes=bytes;
}
extern "C" void esp_op_repack(int64_t handle,int64_t a,int64_t c,int64_t elements) {
  ProfileRegion packing(1); // ESP_PROF_PACK: required packed-layout preparation.
  Context &ctx=lookup(handle);
  auto ae=check(ctx,a,product({elements})),ce=check(ctx,c,product({elements}));
  disjoint(a,ae,c,ce);
  copyRange(TensorView(ctx,a,elements),0,TensorView(ctx,c,elements),0,elements);
}

extern "C" void esp_op_concat(int64_t handle,int64_t a,int64_t b,int64_t c,
    int64_t outer,int64_t aAxis,int64_t bAxis,int64_t inner) {
  Context &ctx=lookup(handle);
  uint64_t aa=product({aAxis,inner}),bb=product({bAxis,inner});
  if(aAxis<=0 || bAxis<=0 || aAxis>INT64_MAX-bAxis)fail("invalid concatenation extent");
  auto ae=check(ctx,a,product({outer,aAxis,inner}));
  auto be=check(ctx,b,product({outer,bAxis,inner}));
  auto ce=check(ctx,c,product({outer,aAxis+bAxis,inner}));
  disjoint(a,ae,c,ce);disjoint(b,be,c,ce);
  TensorView av(ctx,a,outer*aa),bv(ctx,b,outer*bb),cv(ctx,c,outer*(aa+bb));
  for(int64_t o=0;o<outer;++o) {
    copyRange(av,o*aa,cv,o*(aa+bb),aa);
    copyRange(bv,o*bb,cv,o*(aa+bb)+aa,bb);
  }
}

extern "C" void esp_op_close(int64_t handle) {
  ProfileRegion allocation(5);
  Context &ctx = lookup(handle);
  if (ctx.running) fail("cannot close a running socket");
  OP_LOG("%s close socket=%u invocations=%u\n", executionPrefix, ctx.instance, ctx.invocations);
  Context **p = &live;
  while (*p != &ctx) p = &(*p)->next;
  *p = ctx.next;
#ifdef SODAP_OP_ESP_HW
  for(unsigned i=0;i<(ctx.separateDevices?3u:1u);++i)
    iowrite32(ctx.separateDevices?ctx.bindings[i]:ctx.device, PT_NCHUNK_REG, 0);
  release(ctx.pages);
  if(ctx.devices)release(ctx.devices);
  for(auto *list:ctx.enumerations)if(list)release(list);
#endif
  while(ctx.layouts) {auto *next=ctx.layouts->next;release(ctx.layouts);ctx.layouts=next;}
  release(ctx.arena);
  release(&ctx);
#ifdef SODAP_OP_CPU_ARENA_BYTES
  if (!live) cpuArenaUsed = 0;
#endif
}
extern "C" void esp_op_copy_in(int64_t h, int64_t o, int64_t r, void *d) { ProfileRegion packing(1);copy(h,o,r,d,true); }
extern "C" void esp_op_copy_out(int64_t h, int64_t o, int64_t r, void *d) { ProfileRegion unpacking(3);copy(h,o,r,d,false); }

extern "C" void esp_op_conv(int64_t handle, int64_t a, int64_t w, int64_t bias, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ic, int64_t oc, int64_t kh, int64_t kw,
    int64_t oh, int64_t ow, int64_t sh, int64_t sw, int64_t pt, int64_t pl,
    int64_t dh, int64_t dw, int64_t relu) {
  flag(relu);
  esp_op_conv_activation(handle,a,w,bias,c,n,ih,iw,ic,oc,kh,kw,oh,ow,sh,sw,pt,pl,dh,dw,relu,0);
}
extern "C" void esp_op_conv_activation(int64_t handle, int64_t a, int64_t w, int64_t bias, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ic, int64_t oc, int64_t kh, int64_t kw,
    int64_t oh, int64_t ow, int64_t sh, int64_t sw, int64_t pt, int64_t pl,
    int64_t dh, int64_t dw, int64_t mode, int64_t alpha_bits) {
  if (mode < OP_ACT_NONE || mode > OP_ACT_LEAKY_RELU || alpha_bits < 0 || alpha_bits > UINT32_MAX)
    fail("invalid convolution activation");
  uint32_t bits=uint32_t(alpha_bits);
  float alpha;
  std::memcpy(&alpha,&bits,4);
  if (!std::isfinite(alpha) || alpha < 0 || alpha > 1) fail("invalid LeakyReLU slope");
  Context &ctx = lookup(handle);
  columns(ctx,a,ic);columns(ctx,w,ic);columns(ctx,bias,oc);columns(ctx,c,oc);
  auto ae = check(ctx,a,product({n,ih,iw,ic}));
  auto we = check(ctx,w,product({oc,kh,kw,ic}));
  auto be = check(ctx,bias,product({oc}));
  auto ce = check(ctx,c,product({n,oh,ow,oc}));
  disjoint(a,ae,c,ce); disjoint(w,we,c,ce); disjoint(bias,be,c,ce);
  spatial({sh,sw,dh,dw}); spatial({pt,pl},true);
  header(ctx,OP_CONV_MAGIC);
  if (!ctx.dmaWidth && mode == OP_ACT_LEAKY_RELU) field(ctx,1,CONV_VERSION);
  const int64_t dims[] = {n,ih,iw,ic,oc,kh,kw,oh,ow,sh,sw,pt,pl,dh,dw,mode};
  for (unsigned i=0;i<16;++i) field(ctx,CONV_N+i,dims[i]);
  field(ctx,CONV_A,wireAddress(ctx,a)); field(ctx,CONV_W,wireAddress(ctx,w));
  field(ctx,CONV_BIAS,wireAddress(ctx,bias)); field(ctx,CONV_C,wireAddress(ctx,c));
  field(ctx,CONV_ALPHA,bits);
  field(ctx,CONV_ACTIVATION_FP,mode==OP_ACT_RELU?0x3f800000:0);
  if (!launch(ctx,"conv",c,product({n,oh,ow,oc}))) {
    for (int64_t batch=0;batch<n;++batch)
      for (int64_t y=0;y<oh;++y)
        for (int64_t x=0;x<ow;++x)
          for (int64_t channel=0;channel<oc;++channel) {
            float sum=0;
            for (int64_t ky=0;ky<kh;++ky) {
              int64_t sy=y*sh+ky*dh-pt;
              for (int64_t kx=0;kx<kw;++kx) {
                int64_t sx=x*sw+kx*dw-pl;
                if (sy<0 || sy>=ih || sx<0 || sx>=iw) continue;
                uint64_t ai=((batch*ih+sy)*iw+sx)*ic;
                uint64_t wi=((channel*kh+ky)*kw+kx)*ic;
                for (int64_t q=0;q<ic;++q) sum=sum+read(ctx,a,ai+q)*read(ctx,w,wi+q);
              }
            }
            float value=sum+read(ctx,bias,channel);
            if (mode == OP_ACT_LEAKY_RELU)
              value=(value < 0 ? 0.0f : value) + (value < 0 ? value : 0.0f)*alpha;
            else value=activation(value,mode);
            write(ctx,c,((batch*oh+y)*ow+x)*oc+channel,value);
          }
  }
  complete(ctx);
}
extern "C" void esp_op_add(int64_t handle, int64_t a, int64_t b, int64_t c,
    int64_t count, int64_t relu) {
  Context &ctx=lookup(handle);
  auto size=product({count}), ae=check(ctx,a,size), be=check(ctx,b,size), ce=check(ctx,c,size);
  disjoint(a,ae,c,ce); disjoint(b,be,c,ce); flag(relu);
  if(ctx.dmaWidth) {
    columns(ctx,b,layout(ctx,a).columns);columns(ctx,c,layout(ctx,a).columns);
    if(uint64_t(count)%layout(ctx,a).columns)fail("addition count must contain complete packed rows");
  }
  header(ctx,OP_ADD_MAGIC);
  if(ctx.dmaWidth) {
    auto &t=layout(ctx,a);
    field(ctx,ADD_COLUMNS,t.columns); field(ctx,ADD_ROWS,count/t.columns);
    field(ctx,ADD_ROW_BYTES,t.rowWords*4);
    field(ctx,ADD_ACTIVATION_FP,relu?0x3f800000:0);
  }
  field(ctx,ADD_COUNT,count); field(ctx,ADD_A,wireAddress(ctx,a)); field(ctx,ADD_B,wireAddress(ctx,b));
  field(ctx,ADD_C,wireAddress(ctx,c)); field(ctx,ADD_RELU,relu);
  if (!launch(ctx,"add",c,count))
    for (int64_t i=0;i<count;++i) write(ctx,c,i,activation(read(ctx,a,i)+read(ctx,b,i),relu));
  complete(ctx);
}
static void gemm(int64_t handle, int64_t a, int64_t b, int64_t c,
    int64_t m, int64_t n, int64_t k, bool source,
    float alpha=1,float beta=0,float hasC=0) {
  Context &ctx=lookup(handle);
#ifdef SODAP_OP_PROFILE
  esp_op_profile_event(1,ctx.invocations);
#endif
  if(source && (!ctx.dmaWidth || (hasC!=0 && hasC!=1)))
    fail("source GEMM requires packed DMA and an explicit C-presence flag");
  columns(ctx,a,k);columns(ctx,b,n);columns(ctx,c,n);
  auto ae=check(ctx,a,product({m,k})), be=check(ctx,b,product({k,n})), ce=check(ctx,c,product({m,n}));
  disjoint(a,ae,c,ce); disjoint(b,be,c,ce);
  header(ctx,GEMM_MAGIC);
  field(ctx,GEMM_FIELD_M,m); field(ctx,GEMM_FIELD_N,n); field(ctx,GEMM_FIELD_K,k);
  field(ctx,GEMM_FIELD_A,wireAddress(ctx,a)); field(ctx,GEMM_FIELD_B,wireAddress(ctx,b)); field(ctx,GEMM_FIELD_C,wireAddress(ctx,c));
  if(ctx.dmaWidth) {
    field(ctx,GEMM_FIELD_VERSION,source?GEMM_SOURCE_VERSION:GEMM_PACKED_VERSION);field(ctx,GEMM_FIELD_DMA_WIDTH,ctx.dmaWidth);
    uint32_t bits;std::memcpy(&bits,&hasC,4);
    if(source)field(ctx,GEMM_FIELD_HAS_C,bits);
    field(ctx,GEMM_FIELD_LDA,layout(ctx,a).rowWords*4);
    field(ctx,GEMM_FIELD_LDB,layout(ctx,b).rowWords*4);
    field(ctx,GEMM_FIELD_LDC,layout(ctx,c).rowWords*4);
  } else {
    field(ctx,GEMM_FIELD_LDA,k*2);field(ctx,GEMM_FIELD_LDB,n*2);field(ctx,GEMM_FIELD_LDC,n*2);
  }
  uint32_t alphaBits,betaBits;
  std::memcpy(&alphaBits,&alpha,4);std::memcpy(&betaBits,&beta,4);
  field(ctx,GEMM_FIELD_ALPHA,alphaBits); field(ctx,GEMM_FIELD_BETA,betaBits);
  if (!launch(ctx,"gemm",c,product({m,n}))) {
    TensorView av(ctx,a,m*k),bv(ctx,b,k*n),cv(ctx,c,m*n);
    auto as=av.stride(k),bs=bv.stride(n),cs=cv.stride(n);
    for (int64_t i=0;i<m;++i)
      for (int64_t j=0;j<n;++j) {
        float sum=0;
        for (int64_t q=0;q<k;++q) sum=sum+av.get(i*as+q*av.step)*bv.get(q*bs+j*bv.step);
        float value=alpha*sum;
        if(source?hasC!=0:beta!=0)value=beta*cv.get(i*cs+j*cv.step)+value;
        cv.put(i*cs+j*cv.step,value);
      }
  }
  complete(ctx);
#ifdef SODAP_OP_PROFILE
  esp_op_profile_event(0,ctx.invocations-1);
#endif
}

extern "C" void esp_op_gemm(int64_t h,int64_t a,int64_t b,int64_t c,int64_t m,int64_t n,int64_t k) {
  gemm(h,a,b,c,m,n,k,false);
}
extern "C" void esp_op_gemm_source(int64_t h,int64_t a,int64_t b,int64_t c,int64_t m,int64_t n,int64_t k) {
  if(!lookup(h).dmaWidth)fail("GEMM source semantics require packed DMA");
  gemm(h,a,b,c,m,n,k,true);
}
extern "C" void esp_op_gemm_full_source(int64_t h,int64_t a,int64_t b,int64_t c,
    int64_t m,int64_t n,int64_t k,float alpha,float beta,float hasC) {
  gemm(h,a,b,c,m,n,k,true,alpha,beta,hasC);
}
extern "C" float esp_op_scalar(int64_t h,int64_t offset) {
  Context &ctx=lookup(h);check(ctx,offset,1);
  if(layout(ctx,offset).elements!=1)fail("coefficient is not scalar");
  return read(ctx,offset,0);
}
extern "C" void esp_op_relu(int64_t h,int64_t a,int64_t c,int64_t count) {
  Context &ctx=lookup(h);
  auto ae=check(ctx,a,product({count})),ce=check(ctx,c,product({count}));
  disjoint(a,ae,c,ce);
  for(int64_t i=0;i<count;++i)write(ctx,c,i,activation(read(ctx,a,i),1));
}

extern "C" void esp_op_transpose(int64_t handle, int64_t a, int64_t c, int64_t rank,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t p0, int64_t p1, int64_t p2, int64_t p3) {
  Context &ctx=lookup(handle);
  int64_t dims[]={d0,d1,d2,d3}, perm[]={p0,p1,p2,p3}, sourceStrides[4], outputDims[4];
  if (rank<1 || rank>4) fail("unsupported transpose rank");
  uint64_t count=product({d0,d1,d2,d3});
  auto ae=check(ctx,a,count), ce=check(ctx,c,count); disjoint(a,ae,c,ce);
  unsigned used=0;
  for (unsigned i=0;i<4;++i) {
    if (perm[i]<0 || perm[i]>=4 || (used & (1u<<perm[i]))) fail("invalid permutation");
    used |= 1u<<perm[i]; outputDims[i]=dims[perm[i]];
  }
  sourceStrides[3]=1;
  for (int i=2;i>=0;--i) sourceStrides[i]=sourceStrides[i+1]*dims[i+1];
  TensorView av(ctx,a,count),cv(ctx,c,count);
  bool physical=av.rowCompatible(d3);
  if(physical) {
    sourceStrides[3]=av.step;
    sourceStrides[2]=av.stride(d3);
    for(int axis=1;axis>=0;--axis)sourceStrides[axis]=sourceStrides[axis+1]*dims[axis+1];
  }
  uint64_t coords[4]={},source=0,dest=0,column=0;
  for (uint64_t i=0;i<count;++i) {
    cv.base[dest]=av.base[physical?source:av.address(source)];
    if(cv.step==2)cv.base[dest+1]=0;
    dest+=cv.step;
    if(++column==cv.columns) {column=0;dest+=cv.rowWords-cv.columns*cv.step;}
    // Odometer traversal preserves arbitrary permutations without decoding
    // the flattened output index with division/remainder for every element.
    for(int axis=3;axis>=0;--axis) {
      source+=sourceStrides[perm[axis]];
      if(++coords[axis]<uint64_t(outputDims[axis]))break;
      coords[axis]=0;source-=outputDims[axis]*sourceStrides[perm[axis]];
    }
  }
  OP_LOG("REUSABLE_OP cpu=transpose\n");
}
extern "C" void esp_op_pool(int64_t handle, int64_t a, int64_t c,
    int64_t n, int64_t ih, int64_t iw, int64_t ch, int64_t oh, int64_t ow,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t pt, int64_t pl, int64_t average) {
  Context &ctx=lookup(handle);
  auto ae=check(ctx,a,product({n,ih,iw,ch})), ce=check(ctx,c,product({n,oh,ow,ch}));
  disjoint(a,ae,c,ce); spatial({kh,kw,sh,sw}); spatial({pt,pl},true); flag(average);
  if (average && (pt || pl)) fail("padded average pooling is unsupported");
  TensorView av(ctx,a,n*ih*iw*ch),cv(ctx,c,n*oh*ow*ch);
  bool inputRows=av.rowCompatible(ch),outputRows=cv.rowCompatible(ch);
  uint64_t inputStride=av.stride(ch),outputStride=cv.stride(ch);
  float divisor=float(kh*kw);
  for (int64_t batch=0;batch<n;++batch)
    for (int64_t y=0;y<oh;++y)
      for (int64_t x=0;x<ow;++x)
        for (int64_t channel=0;channel<ch;++channel) {
          float value=average ? 0 : -FLT_MAX;
          for (int64_t ky=0;ky<kh;++ky)
            for (int64_t kx=0;kx<kw;++kx) {
              int64_t sy=y*sh+ky-pt, sx=x*sw+kx-pl;
              if (sy<0 || sy>=ih || sx<0 || sx>=iw) continue;
              uint64_t pixel=(batch*ih+sy)*iw+sx;
              float item=av.get(inputRows?pixel*inputStride+channel*av.step:
                                av.address(pixel*ch+channel));
              value=average ? value+item : std::max(value,item);
            }
          if (average) value=value/divisor;
          uint64_t pixel=(batch*oh+y)*ow+x;
          cv.put(outputRows?pixel*outputStride+channel*cv.step:
                 cv.address(pixel*ch+channel),value);
        }
  OP_LOG("REUSABLE_OP cpu=%s_pool\n",average?"average":"max");
}
