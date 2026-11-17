// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
extern "C" float minimum(float,float);
extern "C" float maximum(float,float);
static float f(uint32_t bits){float v;std::memcpy(&v,&bits,4);return v;}
static uint32_t bits(float v){uint32_t n;std::memcpy(&n,&v,4);return n;}
int main() {
  uint32_t values[]={0,0x80000000,0x3f800000,0xbf800000,1,0x80000001,
    0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00002};
  for(auto x:values)for(auto y:values) {
    float a=f(x),b=f(y),lo=minimum(a,b),hi=maximum(a,b);
    if(std::isnan(a)||std::isnan(b)) {assert(std::isnan(lo)&&std::isnan(hi));continue;}
    if(a==0 && b==0) {assert(bits(lo)==((x|y)&0x80000000));assert(bits(hi)==((x&y)&0x80000000));}
    else {assert(lo==std::fmin(a,b));assert(hi==std::fmax(a,b));}
  }
  std::puts("FP32 minimum/maximum PASS (144 operand pairs)");
}
