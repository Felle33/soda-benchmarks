# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Execute compiler-emitted window addressing against an independent DMA oracle.
Only stream primitives are replaced, as in packed_native.py. No production code
or schedule is generated here. The external output observes one local tile.
"""
import ctypes,json,re,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);plugin=sys.argv[2];out.mkdir(parents=True,exist_ok=True)
def run(cmd):subprocess.run(list(map(str,cmd)),check=True,stdout=subprocess.DEVNULL)
source='''module {
func.func @top(%a: memref<?x?x?xf32> {sodap.descriptor_field=18:i64,sodap.dims=array<i64:2,3,4>,sodap.dma_view="conv_window_flat"},
 %o: memref<?x?x?xf32> {sodap.descriptor_field=21:i64,sodap.dims=array<i64:2,5,6>,sodap.dma_view="packed"},
 %n:index {sodap.descriptor_field=2:i64},%p:index {sodap.descriptor_field=23:i64},%r:index {sodap.descriptor_field=24:i64},
 %tp:index {sodap.descriptor_field=27:i64},%tr:index {sodap.descriptor_field=28:i64},
 %p0:index {sodap.descriptor_field=29:i64},%r0:index {sodap.descriptor_field=30:i64},%b0:index {sodap.descriptor_field=26:i64})
 attributes {sodap.descriptor_fields=32:i64,sodap.descriptor_magic=1129270870:i64,sodap.descriptor_version=4:i64,sodap.descriptor_width_field=31:i64} {
 dataflow.dispatch {
  dataflow.task attributes {load} {
   %t=memref.alloca() : memref<1x8x64xf32>
   %s=memref.subview %a[%b0,%p0,%r0][1,%tp,%tr][1,1,1] : memref<?x?x?xf32> to memref<1x?x?xf32,strided<[?,?,1],offset:?>>
   %d=memref.subview %o[0,0,0][1,%tp,%tr][1,1,1] : memref<?x?x?xf32> to memref<1x?x?xf32,strided<[?,?,1]>>
   %local=memref.subview %t[0,0,0][1,%tp,%tr][1,1,1] : memref<1x8x64xf32> to memref<1x?x?xf32,strided<[512,64,1]>>
   memref.copy %s,%local : memref<1x?x?xf32,strided<[?,?,1],offset:?>> to memref<1x?x?xf32,strided<[512,64,1]>>
   memref.copy %local,%d : memref<1x?x?xf32,strided<[512,64,1]>> to memref<1x?x?xf32,strided<[?,?,1]>>
  }
 }
 return
}
}'''
# N,IH,IW,IC,KH,KW,SH,SW,PT,PB,PL,PR,DH,DW,p0,r0,batch.
specs=[(1,4,7,4,3,3,1,1,1,1,1,1,1,1,5,0,0),
       (1,4,7,4,3,3,1,1,1,1,1,1,1,1,25,5,0),
       (1,224,224,3,7,7,2,2,3,3,3,3,1,1,109,128,0),
       (1,9,11,6,3,3,2,2,1,2,2,1,1,1,4,7,0),
       (1,7,9,2,3,3,1,1,2,2,2,2,2,2,7,3,0),
       (1,3,5,65,1,1,1,1,0,0,0,0,1,1,3,64,0),
       (2,4,6,4,3,3,1,1,1,1,1,1,1,1,4,0,1),
       (1,1,1,2,3,3,1,1,5,5,5,5,1,1,0,0,0)]
reports=[]
for merge in [False,True]:
 d=out/('merged' if merge else 'separate');d.mkdir(exist_ok=True);(d/'input.mlir').write_text(source)
 flags=['--load-dialect-plugin='+plugin,'--load-pass-plugin='+plugin]
 pipeline='builtin.module(sodap-dataflow-copy-to-dma{top-func=top dma-data-width=64 merge-window-runs='+str(merge).lower()+'},canonicalize,func.func(sodap-dataflow-tasks-to-nodes),sodap-dataflow-nodes-to-func,canonicalize)'
 run(['mlir-opt',d/'input.mlir',*flags,'--pass-pipeline='+pipeline,'-o',d/'packed.mlir'])
 ir=(d/'packed.mlir').read_text()
 assert not re.search(r'arith\.(?:divui|divsi|remui|remsi)\s+%\w+,\s+%(?!c\d)',ir),'dynamic geometry divider'
 ir=re.sub(r'(%\w+) = dataflow.stream_read (%\w+) : \(!dataflow.stream<i64, \d+>\) -> i64',r'\1 = func.call @test_read(\2) : (i64) -> i64',ir)
 ir=re.sub(r'dataflow.stream_write (%\w+), (%\w+)(?: \{[^}]*\})? : <i64, \d+>, i64',r'func.call @test_write(\1,\2) : (i64,i64) -> ()',ir)
 ir=re.sub(r'!dataflow.stream<i64, \d+>','i64',ir)
 ir=ir[:ir.rfind('}')]+'func.func private @test_read(i64) -> i64\nfunc.func private @test_write(i64,i64)\n}\n'
 assert 'dataflow.stream_' not in ir
 (d/'model.mlir').write_text(ir)
 run(['soda-opt',d/'model.mlir','--lower-all-to-llvm=use-bare-ptr-memref-call-conv','--convert-func-to-llvm','--reconcile-unrealized-casts','-o',d/'llvm.mlir'])
 run(['mlir-translate','--mlir-to-llvmir',d/'llvm.mlir','-o',d/'kernel.ll'])
 (d/'dma.c').write_text('''#include <stdint.h>
#include <assert.h>
uint32_t memory[1048576],source_end,input_requests;
static unsigned rpos,rleft,wpos,wleft;
uint64_t test_read(uint64_t ch){assert(ch==1&&rleft>=2&&!wleft);rleft-=2;uint64_t v=memory[rpos]|((uint64_t)memory[rpos+1]<<32);rpos+=2;return v;}
void test_write(uint64_t ch,uint64_t value){
 if(!(ch&1)){unsigned at=(uint32_t)value*2,len=(value>>32)*2;assert(!rleft&&!wleft&&len&&at+len<=1048576);
 if(ch==0){rpos=at;rleft=len;if(at>=64&&at<source_end){assert(at+len<=source_end);input_requests++;}}
 else {assert(ch==2);wpos=at;wleft=len;}}
 else{assert(ch==3&&wleft>=2&&!rleft);memory[wpos++]=value;memory[wpos++]=value>>32;wleft-=2;}}
void finish(void){assert(!rleft&&!wleft);}
''')
 run(['clang-19','-O2','-ffp-contract=off','-fPIC','-shared',d/'kernel.ll',d/'dma.c','-o',d/'test.so'])
 lib=ctypes.CDLL(str((d/'test.so').resolve()));mem=(ctypes.c_uint32*1048576).in_dll(lib,'memory')
 counts=[]
 for case,spec in enumerate(specs):
  n,ih,iw,ic,kh,kw,sh,sw,pt,pb,pl,pr,dh,dw,p0,r0,b0=spec
  oh=(ih+pt+pb-(kh-1)*dh-1)//sh+1;ow=(iw+pl+pr-(kw-1)*dw-1)//sw+1
  p=oh*ow;r=kh*kw*ic;tp=min(8,p-p0);tr=min(64,r-r0);assert tp>0 and tr>0
  stride=(ic+1)//2*2;end=64+n*ih*iw*stride;dest=end+16;row=(tr+1)//2*2;size=dest+n*tp*row+16
  for i in range(size):mem[i]=0xdeadbeef
  for i in range(n*ih*iw):
   for c in range(ic):mem[64+i*stride+c]=0x3e000000+((i*ic+c)*17)%0x1000000
  for i in range(64):mem[i]=0
  fields={0:0x434f4e56,1:4,2:n,3:ih,4:iw,5:ic,6:1,7:kh,8:kw,9:oh,10:ow,11:sh,12:sw,13:pt,14:pl,15:dh,16:dw,18:256,21:dest*4,23:p,24:r,26:b0,27:tp,28:tr,29:p0,30:r0,31:64}
  for f,v in fields.items():mem[2*f]=v
  expected=list(mem[:size]);input_addresses=[]
  for pp in range(tp):
   y,x=divmod(p0+pp,ow)
   for rr in range(tr):
    kr,c=divmod(r0+rr,ic);ky,kx=divmod(kr,kw);sy=y*sh+ky*dh-pt;sx=x*sw+kx*dw-pl
    address=64+((b0*ih+sy)*iw+sx)*stride+c
    valid=0<=sy<ih and 0<=sx<iw
    expected[dest+pp*row+rr]=mem[address] if valid else 0
    if valid:input_addresses.append(address)
  ctypes.c_uint32.in_dll(lib,'source_end').value=end;ctypes.c_uint32.in_dll(lib,'input_requests').value=0
  lib.top(ctypes.c_uint64(0),ctypes.c_uint64(1),ctypes.c_uint64(2),ctypes.c_uint64(3));lib.finish()
  assert list(mem[:size])==expected,(merge,case,[(i,hex(a),hex(b)) for i,(a,b) in enumerate(zip(mem,expected)) if a!=b][:5])
  count=ctypes.c_uint32.in_dll(lib,'input_requests').value;counts.append(count)
  if not input_addresses:assert count==0,'out-of-image tile issued input DMA'
 reports.append(dict(merge=merge,status='PASS',cases=len(specs),input_requests=counts))
assert all(a<=b for a,b in zip(reports[1]['input_requests'],reports[0]['input_requests']))
assert any(a<b for a,b in zip(reports[1]['input_requests'],reports[0]['input_requests']))
(out/'result.json').write_text(json.dumps(reports,indent=2)+'\n');print(reports)
