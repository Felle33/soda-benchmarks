#include "sodap/ExecutionEngine/ReusableOperatorABI.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
struct Matrix {float *allocated,*aligned;int64_t offset,sizes[2],strides[2];};
static uint32_t expected[GEMM_HEADER_BYTES/4];
extern "C" void esp_op_execute_rtl(uint32_t *arena,uint64_t bytes) {
  (void)bytes;
  assert(!std::memcmp(arena,expected,sizeof(expected)));
}
int main(int argc,char **argv) {
  assert(argc==2);setenv("SODAP_GEMM_TRACE_DIR",argv[1],1);
  auto old=esp_gemm_open_packed(512,0,64);auto current=esp_op_open_devices(256,64,-1,-1,0);
  esp_op_select_device(current,2,0);
  float A[]={1,2,3,4,5,6},B[]={1,2,3,4,5,6},C[]={2,3,4,5};
  Matrix av{A,A,0,{2,3},{3,1}},bv{B,B,0,{3,2},{2,1}},cv{C,C,0,{2,2},{2,1}};
  esp_op_tensor(current,0,6,3);esp_op_tensor(current,32,6,2);esp_op_tensor(current,56,4,2);
  esp_gemm_copy_in(old,256,2,&av);esp_gemm_copy_in(old,288,2,&bv);
  esp_op_copy_in(current,0,2,&av);esp_op_copy_in(current,32,2,&bv);
  for(int i=0;i<3;++i) {
    float alpha=i==0?1:.5f,beta=i==1?0:2,hasC=i==0?0:1;
    esp_gemm_copy_in(old,312,2,&cv);esp_op_copy_in(current,56,2,&cv);
    esp_gemm_invoke_source(old,256,288,312,2,2,3,16,8,8,alpha,beta,hasC);
    auto path=std::string(argv[1])+"/initial-"+std::to_string(i)+".hex";
    FILE *f=std::fopen(path.c_str(),"r");assert(f);
    for(auto &v:expected)assert(std::fscanf(f,"%x",&v)==1);
    std::fclose(f);
    esp_op_gemm_full_source(current,0,32,56,2,2,3,alpha,beta,hasC);
  }
  esp_gemm_close(old);esp_op_close(current);
  std::puts("Unified descriptor byte identity PASS");
}
