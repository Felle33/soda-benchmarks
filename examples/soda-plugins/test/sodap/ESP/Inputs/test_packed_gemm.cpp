// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "sodap/ExecutionEngine/ReusableGemmABI.h"
#include "sodap/ExecutionEngine/DmaTransport.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
struct Matrix { float *allocated, *aligned; int64_t offset, sizes[2], strides[2]; };
int main() {
  for (unsigned width : {64u,128u,256u,512u}) {
    sodap::DmaTransport layout{width};
    uint64_t ak, bn, wn; layout.rowWords(3,ak); layout.rowWords(2,bn); layout.rowWords(1,wn);
    int64_t a=128,b=a+2*ak*4,c=b+3*bn*4,w=c+2*bn*4,y=w+2*wn*4;
    int64_t ctx=esp_gemm_open_packed(y+2*wn*4,0,width);
    float A[]={1,2,3,99,4,5,6,99};
    float B[]={1,2,99,3,4,99,5,6,99};
    float C[]={2,2,99,2,2,99},W[]={1,2},Y[]={99,99};
    Matrix av{A,A,0,{2,3},{4,1}},bv{B,B,0,{3,2},{3,1}},cv{C,C,0,{2,2},{3,1}},
           wv{W,W,0,{2,1},{1,1}},yv{Y,Y,0,{2,1},{1,1}};
    esp_gemm_copy_in(ctx,a,2,&av); esp_gemm_copy_in(ctx,b,2,&bv);
    esp_gemm_copy_in(ctx,c,2,&cv); esp_gemm_copy_in(ctx,w,2,&wv);
    esp_gemm_copy_in(ctx,y,2,&yv);
    esp_gemm_invoke(ctx,a,b,c,2,2,3,ak*4,bn*4,bn*4,.5f,2);
    // Dependent invocation, different reduction/output shapes; data stays in DRAM.
    esp_gemm_invoke(ctx,c,w,y,2,1,2,bn*4,wn*4,wn*4,1,0);
    esp_gemm_invoke(ctx,c,w,y,2,1,2,bn*4,wn*4,wn*4,1,1);
    esp_gemm_copy_out(ctx,c,2,&cv); esp_gemm_copy_out(ctx,y,2,&yv);
    float expected[]={15,18,99,28.5f,36,99};
    assert(!std::memcmp(C,expected,sizeof(C)) && Y[0]==102 && Y[1]==201);
    esp_gemm_copy_out(ctx,a,2,&av); esp_gemm_copy_out(ctx,b,2,&bv);
    assert(A[3]==99 && A[7]==99 && B[2]==99 && B[8]==99);
    esp_gemm_close(ctx);
  }
  std::puts("Packed reusable GEMM runtime PASS");
}
