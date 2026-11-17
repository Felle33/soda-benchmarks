// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
struct Vector {float *allocated,*aligned;int64_t offset,size,stride;};
int main() {
  for(unsigned width:{64u,128u,256u,512u}) {
    auto ctx=esp_op_open_packed(4096,0,width);
    esp_op_tensor(ctx,0,6,3);esp_op_tensor(ctx,512,9,3);esp_op_tensor(ctx,1024,3,3);
    esp_op_tensor(ctx,1536,6,3);esp_op_tensor(ctx,2048,6,3);esp_op_tensor(ctx,2560,6,6);
    esp_op_tensor(ctx,3072,6,1);esp_op_tensor(ctx,3584,1,1);
    float x[]={-1,2,3,4,-5,6},w[]={2,0,0,0,2,0,0,0,2},bias[]={1,0,-1},ones[]={1,1,1,1,1,1};
    Vector xv{x,x,0,6,1},wv{w,w,0,9,1},bv{bias,bias,0,3,1},gv{ones,ones,0,6,1};
    esp_op_copy_in(ctx,0,1,&xv);esp_op_copy_in(ctx,512,1,&wv);
    esp_op_copy_in(ctx,1024,1,&bv);esp_op_copy_in(ctx,3072,1,&gv);
    esp_op_conv_activation(ctx,0,512,1024,1536,1,1,2,3,3,1,1,1,2,1,1,0,0,1,1,2,0x3f000000);
    float c[6]={};Vector cv{c,c,0,6,1};esp_op_copy_out(ctx,1536,1,&cv);
    const float expected[]={-.5f,4,5,9,-5,11};assert(!std::memcmp(c,expected,sizeof(c)));
    esp_op_add(ctx,1536,1536,2048,6,0);
    // A second invocation reuses the addition engine for a shorter row range.
    esp_op_add(ctx,1536,1536,2048,3,1);
    esp_op_repack(ctx,2048,2560,6);
    esp_op_gemm(ctx,2560,3072,3584,1,1,6);
    float y=0;Vector yv{&y,&y,0,1,1};esp_op_copy_out(ctx,3584,1,&yv);assert(y==48);
    // CPU layout/pool operations must use the same padded channel layout.
    esp_op_tensor(ctx,2560,3,3);
    esp_op_pool(ctx,1536,2560,1,1,2,3,1,1,1,2,1,1,0,0,1);
    float pool[3]={};Vector pv{pool,pool,0,3,1};esp_op_copy_out(ctx,2560,1,&pv);
    assert(pool[0]==4.25f && pool[1]==-.5f && pool[2]==8);
    esp_op_close(ctx);
  }
  std::puts("Packed convolution/add/GEMM/layout runtime PASS");
}
