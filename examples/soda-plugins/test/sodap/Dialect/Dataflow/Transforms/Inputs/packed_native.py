# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Execute the actual packed-transfer MLIR at all widths with an exact DMA model.
Only stream primitives are replaced by test runtime calls; address arithmetic,
loops, lane masks, loads, stores and RMW are the compiler's emitted operations.
"""
import ctypes,json,os,re,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
plugin=sys.argv[2]
descriptor="--descriptor" in sys.argv[3:]
header=32 if descriptor else 0
def run(cmd):subprocess.run(list(map(str,cmd)),check=True,stdout=subprocess.DEVNULL)
report=[]
for width in (64,128,256,512):
 lanes=width//32;cols=19;rows=3;stride=(cols+lanes-1)//lanes*lanes;words=rows*stride
 # Copy consecutive, deliberately odd-sized tiles. Neighbor outputs start
 # with sentinels; later tiles must preserve already written neighboring lanes.
 src='module {\nfunc.func @top(%a: memref<3x19xf32>, %o: memref<3x19xf32>) {\n'
 for n,(col,size) in enumerate(((0,3),(3,5),(8,1),(9,10))):
  src+=f'''%a{n} = memref.subview %a[0,{col}][3,{size}][1,1] : memref<3x19xf32> to memref<3x{size}xf32, strided<[19,1], offset: {col}>>
%b{n} = memref.subview %o[0,{col}][3,{size}][1,1] : memref<3x19xf32> to memref<3x{size}xf32, strided<[19,1], offset: {col}>>
%t{n} = memref.alloca() : memref<3x{size}xf32>
memref.copy %a{n}, %t{n} : memref<3x{size}xf32, strided<[19,1], offset: {col}>> to memref<3x{size}xf32>
memref.copy %t{n}, %b{n} : memref<3x{size}xf32> to memref<3x{size}xf32, strided<[19,1], offset: {col}>>
'''
 src+='return\n}\n}\n'
 if descriptor:
  src=src.replace('func.func @top(%a: memref<3x19xf32>, %o: memref<3x19xf32>) {',
    'func.func @top(%a: memref<?x?xf32> {sodap.descriptor_field=5:i64, sodap.descriptor_ld=8:i64}, %o: memref<?x?xf32> {sodap.descriptor_field=7:i64, sodap.descriptor_ld=10:i64}) {\ndataflow.dispatch {\ndataflow.task attributes {load} {')
  src=src.replace('memref<3x19xf32>', 'memref<?x?xf32>')
  src=src.replace('strided<[19,1]', 'strided<[?,1]')
  src=src.replace('return\n', '}\n}\nreturn\n')
 d=out/str(width);d.mkdir(exist_ok=True);(d/'input.mlir').write_text(src)
 flags=['--load-dialect-plugin='+plugin,'--load-pass-plugin='+plugin]
 run(['mlir-opt',d/'input.mlir',*flags,f'--pass-pipeline=builtin.module(sodap-dataflow-copy-to-dma{{top-func=top dma-data-width={width}}},canonicalize'+(',func.func(sodap-dataflow-tasks-to-nodes),sodap-dataflow-nodes-to-func,canonicalize)' if descriptor else ')'), '-o',d/'packed.mlir'])
 ir=(d/'packed.mlir').read_text();types=set()
 def read(m):
  name,ch,w=m.groups();types.add(int(w));return f'{name} = func.call @test_read{w}({ch}) : (i64) -> i{w}'
 def write(m):
  ch,value,w=m.groups();types.add(int(w));return f'func.call @test_write{w}({ch}, {value}) : (i64, i{w}) -> ()'
 ir=re.sub(r'(%[\w]+) = dataflow.stream_read (%[\w]+) : \(!dataflow.stream<i(\d+), \d+>\) -> i\d+',read,ir)
 ir=re.sub(r'dataflow.stream_write (%[\w]+), (%[\w]+)(?: \{[^}]*\})? : <i(\d+), \d+>, i\d+',write,ir)
 ir=re.sub(r'!dataflow.stream<i\d+, \d+>','i64',ir)
 ir=ir[:ir.rfind('}')]+''.join(f'func.func private @test_read{w}(i64) -> i{w}\nfunc.func private @test_write{w}(i64,i{w})\n' for w in sorted(types))+'}\n'
 assert 'dataflow.stream_' not in ir
 (d/'model.mlir').write_text(ir)
 run(['soda-opt',d/'model.mlir','--lower-all-to-llvm=use-bare-ptr-memref-call-conv','--convert-func-to-llvm','--reconcile-unrealized-casts','-o',d/'llvm.mlir'])
 run(['mlir-translate','--mlir-to-llvmir',d/'llvm.mlir','-o',d/'kernel.ll'])
 # Wide integer return ABI is explicit LLVM, avoiding any host C wide-int ABI.
 shim='declare i32 @read32(i64)\ndeclare void @write32(i64,i32)\ndeclare void @request(i64,i64)\n'
 for w in sorted(types):
  shim+=f'define i{w} @test_read{w}(i64 %ch) {{\n'
  for lane in range(w//32):
   shim+=f'%r{lane}=call i32 @read32(i64 %ch)\n%e{lane}=zext i32 %r{lane} to i{w}\n%s{lane}=shl i{w} %e{lane}, {32*lane}\n'
   if lane:shim+=f'%v{lane}=or i{w} '+('%s0' if lane==1 else f'%v{lane-1}')+f', %s{lane}\n'
  shim+=f'ret i{w} %v{w//32-1}\n}}\n'
  shim+=f'define void @test_write{w}(i64 %ch,i{w} %value) {{\n'
  if w==64:
   shim+='%ctrl=and i64 %ch,1\n%c=icmp eq i64 %ctrl,0\nbr i1 %c,label %req,label %data\nreq:\ncall void @request(i64 %ch,i64 %value)\nret void\ndata:\n'
  for lane in range(w//32):shim+=f'%s{lane}=lshr i{w} %value,{lane*32}\n%t{lane}=trunc i{w} %s{lane} to i32\ncall void @write32(i64 %ch,i32 %t{lane})\n'
  shim+='ret void\n}\n'
 (d/'shim.ll').write_text(shim)
 (d/'model.c').write_text(f'''#include <stdint.h>
#include <assert.h>
uint32_t memory[{words*2+32+header}];
static unsigned rpos,rleft,wpos,wleft,nreads,nwrites;
void request(uint64_t ch,uint64_t req) {{
 unsigned index=(uint32_t)req,count=req>>32;
 assert(!rleft&&!wleft&&count&&index+count<={(words*2+header)//lanes});
 if(ch==0){{rpos=index*{lanes};rleft=count*{lanes};nreads++;}}
 else {{assert(ch==2);wpos=index*{lanes};wleft=count*{lanes};nwrites++;}}
}}
uint32_t read32(uint64_t ch){{assert(ch==1&&rleft&&!wleft);rleft--;return memory[16+rpos++];}}
void write32(uint64_t ch,uint32_t v){{assert(ch==3&&wleft&&!rleft);wleft--;memory[16+wpos++]=v;}}
void finish(void){{assert(!rleft&&!wleft&&nreads>nwrites);}}
''')
 run(['/usr/bin/clang-19','-shared','-fPIC','-O2',d/'kernel.ll',d/'shim.ll',d/'model.c','-o',d/'test.so'])
 lib=ctypes.CDLL(str((d/'test.so').resolve()));mem=(ctypes.c_uint32*(words*2+32+header)).in_dll(lib,'memory')
 patterns=[0x80000000,0x7fc01234,0x7f800000,0xff800000,0x00000001,0x3f800000,0xdeadbeef]
 for iteration in range(2):
  for i in range(len(mem)):mem[i]=0xa5a5a5a5
  for row in range(rows):
   for col in range(cols):mem[16+header+row*stride+col]=patterns[(row*cols+col+iteration)%len(patterns)]
  if descriptor:
   for f in range(header):mem[16+f]=0
   for f,v in {0:0x47454d4d,1:2,2:rows,3:cols,5:header*4,7:(header+words)*4,8:stride*4,10:stride*4,13:width}.items():mem[16+2*f]=v
  before=list(mem);lib.top(ctypes.c_uint64(0),ctypes.c_uint64(1),ctypes.c_uint64(2),ctypes.c_uint64(3));lib.finish()
  assert list(mem[:16+header+words])==before[:16+header+words]
  assert list(mem[16+header+2*words:])==before[16+header+2*words:]
  for row in range(rows):
   for col in range(cols):assert mem[16+header+words+row*stride+col]==mem[16+header+row*stride+col],(width,row,col)
 if descriptor:
  for field,bad in ((0,0),(1,1),(13,32)):
   saved=mem[16+2*field];mem[16+2*field]=bad;before=list(mem)
   lib.top(ctypes.c_uint64(0),ctypes.c_uint64(1),ctypes.c_uint64(2),ctypes.c_uint64(3));lib.finish()
   assert list(mem)==before,('invalid descriptor modified arena',width,field)
   mem[16+2*field]=saved
 report.append({'descriptor':descriptor,'width':width,'invocations':2,'logical_elements':rows*cols,'status':'PASS'})
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
