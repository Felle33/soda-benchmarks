// Generated MLIR host entrypoints; board timing excludes setup and validation.
#include <stdint.h>
#include <stdio.h>
#include "model.h"
extern const float nn_input[],nn_parameters[],nn_reference[];
extern int64_t forward_init(const float *);
extern void forward_run(int64_t,const float *,float *);
extern void forward_close(int64_t);
static float output[NN_OUTPUT_ELEMENTS];
#if NN_STREAM_INPUTS > 0
extern const float nn_input_second[],nn_reference_second[];
#endif
static uint64_t cycle(void){uint64_t n;__asm__ volatile("csrr %0,mcycle":"=r"(n));return n;}
static uint64_t stage_cycles[256],stage_begin;
static unsigned stage_count;
void esp_op_profile_begin(int64_t index){(void)index;stage_begin=cycle();}
void esp_op_profile_end(int64_t index){
  if(index>=0 && index<256){stage_cycles[index]=cycle()-stage_begin;if(stage_count<index+1)stage_count=index+1;}
}
extern void tohost_exit(uintptr_t) __attribute__((noreturn));
uintptr_t handle_trap(uintptr_t cause,uintptr_t epc,uintptr_t regs[32]) {
  (void)regs;printf("EXP3_GENERATED_TRAP cause=%lu epc=%lx\n",(unsigned long)cause,(unsigned long)epc);tohost_exit(1337);
}
int main(void) {
  printf("EXP3_GENERATED_BEGIN outputs=%u arena=%u\n",NN_OUTPUT_ELEMENTS,NN_ARENA_BYTES);
  uint64_t start=cycle();int64_t ctx=forward_init(nn_parameters);uint64_t setup=cycle()-start;
  unsigned errors=0;uint64_t elapsed[NN_STREAM_INPUTS+1];unsigned counts[NN_STREAM_INPUTS+1];
  for(unsigned run=0;run<=NN_STREAM_INPUTS;++run) {
    const float *input=nn_input,*reference=nn_reference;
#if NN_STREAM_INPUTS > 0
    if(run&1){input=nn_input_second;reference=nn_reference_second;}
#endif
    for(unsigned i=0;i<NN_OUTPUT_ELEMENTS;++i)output[i]=__builtin_nanf("");
    start=cycle();forward_run(ctx,input,output);elapsed[run]=cycle()-start;
    counts[run]=0;
    for(unsigned i=0;i<NN_OUTPUT_ELEMENTS;++i) {
      float delta=output[i]-reference[i];if(delta<0)delta=-delta;
      float mag=reference[i]<0?-reference[i]:reference[i];
      if(!(delta<=NN_ATOL+NN_RTOL*mag))counts[run]++;
    }
    errors+=counts[run];
  }
  forward_close(ctx);
  for(unsigned i=0;i<stage_count;++i)printf("EXP3_GENERATED_STAGE index=%u ticks=%lu\n",i,(unsigned long)stage_cycles[i]);
  uint64_t total=0;
  for(unsigned run=0;run<=NN_STREAM_INPUTS;++run) {
    printf("EXP3_GENERATED_INPUT index=%u ticks=%lu errors=%u\n",run,(unsigned long)elapsed[run],counts[run]);
    if(NN_STREAM_INPUTS==0 || run>0)total+=elapsed[run];
  }
  printf("EXP3_GENERATED_RESULT inputs=%u ticks=%lu setup_ticks=%lu errors=%u\n",NN_STREAM_INPUTS?NN_STREAM_INPUTS:1,(unsigned long)total,(unsigned long)setup,errors);
  printf(errors?"TEST FAILED\n":"TEST PASSED\n");return errors?1:0;
}
