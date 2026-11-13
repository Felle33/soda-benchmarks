import ctypes,json,re,subprocess,sys
from pathlib import Path
import numpy as np
plugin,folder=sys.argv[1:];root=Path(folder);root.mkdir(parents=True,exist_ok=True)
cases={
'chain':'''module {
 func.func @compute(%a: memref<5x9xf32>, %b: memref<9x7xf32>, %d: memref<7x11xf32>, %c: memref<5x11xf32>) {
  %z = arith.constant 0.0 : f32
  %e = memref.alloc() : memref<5x7xf32>
  linalg.fill ins(%z : f32) outs(%e : memref<5x7xf32>)
  linalg.matmul ins(%a,%b : memref<5x9xf32>,memref<9x7xf32>) outs(%e : memref<5x7xf32>)
  linalg.matmul ins(%e,%d : memref<5x7xf32>,memref<7x11xf32>) outs(%c : memref<5x11xf32>)
  memref.dealloc %e : memref<5x7xf32>
  return
 }
}''',
'vectors':'''module {
 func.func @compute(%a: memref<9x11xf32>, %b: memref<11x9xf32>, %x: memref<11xf32>, %y: memref<11xf32>) {
  %z = arith.constant 0.0 : f32
  %t = memref.alloc() : memref<9xf32>
  linalg.fill ins(%z : f32) outs(%t : memref<9xf32>)
  linalg.matvec ins(%a,%x : memref<9x11xf32>,memref<11xf32>) outs(%t : memref<9xf32>)
  linalg.matvec ins(%b,%t : memref<11x9xf32>,memref<9xf32>) outs(%y : memref<11xf32>)
  memref.dealloc %t : memref<9xf32>
  return
 }
}''',
'tensor':'''module {
 func.func @compute(%a: memref<3x5x7xf32>, %b: memref<7x9xf32>, %c: memref<3x5x9xf32>) {
  linalg.generic {indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>,affine_map<(b,m,n,k)->(k,n)>,affine_map<(b,m,n,k)->(b,m,n)>],iterator_types=["parallel","parallel","parallel","reduction"]} ins(%a,%b : memref<3x5x7xf32>,memref<7x9xf32>) outs(%c : memref<3x5x9xf32>) {
   ^bb0(%x: f32,%y: f32,%v: f32):
    %p=arith.mulf %x,%y : f32
    %s=arith.addf %v,%p : f32
    linalg.yield %s : f32
  }
  return
 }
}'''}
lower=['func.func(convert-linalg-to-loops)','func.func(fold-memref-alias-ops)',
       'lower-affine','convert-scf-to-cf','expand-strided-metadata','lower-affine',
       'sodap-finalize-banked-memrefs-to-llvm',
       'convert-func-to-llvm{use-bare-ptr-memref-call-conv=true}',
       'convert-index-to-llvm','convert-cf-to-llvm','reconcile-unrealized-casts']
stub=root/'markers.c';stub.write_text('void __bambu_csroa_partition__(const void *p,int f,int b,int d) {}\n')
def run(args):subprocess.run(list(map(str,args)),check=True)
def shapes(text):
 signature=re.search(r'func.func @compute\((.*?)\)',text)[1]
 return [tuple(map(int,s.rstrip('x').split('x'))) for s in re.findall(r'memref<((?:\d+x)+)f32',signature)]
records=[]
for name,source in cases.items():
 path=root/name;path.mkdir(exist_ok=True);(path/'input.mlir').write_text(source)
 original=shapes(source)
 flags=['--load-dialect-plugin='+plugin,'--load-pass-plugin='+plugin]
 plan=path/'plan.json'
 run(['mlir-opt',path/'input.mlir',*flags,'--pass-pipeline=builtin.module(sodap-prepare-compute-budget{lanes=64 report='+str(plan)+'},sodap-bank-matmul-buffers)', '-o',path/'banked.mlir'])
 physical=shapes((path/'banked.mlir').read_text())
 alloc=json.loads(plan.read_text());assert sum(s['lanes'] for s in alloc['groups'])==64
 libs=[]
 for variant in ('reference','optimized'):
  inp=path/('input.mlir' if variant=='reference' else 'banked.mlir')
  passes=([] if variant=='reference' else ['sodap-register-block-matmul','canonicalize','cse'])+lower
  run(['mlir-opt',inp,*flags,'--pass-pipeline=builtin.module('+','.join(passes)+')','-o',path/(variant+'.mlir')])
  run(['mlir-translate','--mlir-to-llvmir',path/(variant+'.mlir'),'-o',path/(variant+'.ll')])
  run(['clang-19','-O2','-ffp-contract=off','-shared','-fPIC',path/(variant+'.ll'),stub,'-o',path/(variant+'.so')])
  lib=ctypes.CDLL(str((path/(variant+'.so')).resolve()));lib.compute.argtypes=[ctypes.c_void_p]*len(original);lib.compute.restype=None;libs.append(lib)
 dense=[];padded=[];masks=[]
 for i,(logical,storage) in enumerate(zip(original,physical)):
  data=(((np.arange(np.prod(logical))+3*i)*7)%17-8).astype(np.float32).reshape(logical)*np.float32(0.25)
  padded_array=np.full(storage,777,dtype=np.float32);mask=np.zeros(storage,dtype=bool)
  slices=tuple(slice(0,n) for n in logical);padded_array[slices]=data;mask[slices]=True
  dense.append(data.copy());padded.append(padded_array);masks.append(mask)
 for _ in range(2):
  libs[0].compute(*[ctypes.c_void_p(a.ctypes.data) for a in dense])
  libs[1].compute(*[ctypes.c_void_p(a.ctypes.data) for a in padded])
  for logical,expected,actual,mask in zip(original,dense,padded,masks):
   np.testing.assert_array_equal(actual[tuple(slice(0,n) for n in logical)],expected)
   assert np.all(actual[~mask]==777),'Padding overwritten'
 records.append({'case':name,'status':'PASS','invocations':2,'allocation':alloc})
 print(name,'PASS',flush=True)
(root/'results.json').write_text(json.dumps(records,indent=2)+'\n')
