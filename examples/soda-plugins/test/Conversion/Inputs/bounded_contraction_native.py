# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Execute bounded register-block compute with poisoned unused tile storage."""
import ctypes,json,subprocess,sys
from pathlib import Path
import numpy as np
plugin,folder=sys.argv[1:];root=Path(folder);root.mkdir(parents=True,exist_ok=True)
def run(cmd):subprocess.run(list(map(str,cmd)),check=True,stdout=subprocess.DEVNULL)
src='''module {
func.func @compute(%m:index,%n:index,%k:index,%a:memref<16x16xf32>,%b:memref<16x16xf32>,%bias:memref<16xf32>,%c:memref<16x16xf32>,%out:memref<16x16xf32>) {
 %z=arith.constant 0:index
 %step=arith.constant 16:index
 %f=arith.constant 0.0:f32
 scf.for %i=%z to %m step %step {
  scf.for %j=%z to %n step %step {
   linalg.fill {sodap.bound_axes=array<i64:1,2>} ins(%f:f32) outs(%c:memref<16x16xf32>)
   scf.for %kk=%z to %k step %step {
    linalg.generic {indexing_maps=[affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],iterator_types=["parallel","parallel","reduction"]}
    ins(%a,%b:memref<16x16xf32>,memref<16x16xf32>) outs(%c:memref<16x16xf32>) attrs={sodap.bound_axes=array<i64:1,2,3>} {
    ^bb0(%x:f32,%y:f32,%v:f32):
     %p=arith.mulf %x,%y:f32
     %s=arith.addf %v,%p:f32
     linalg.yield %s:f32
    }
   } {sodap.bound_axis=3:i64}
   linalg.generic {indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(n)>,affine_map<(m,n)->(m,n)>],iterator_types=["parallel","parallel"]}
   ins(%c,%bias:memref<16x16xf32>,memref<16xf32>) outs(%out:memref<16x16xf32>) attrs={sodap.bound_axes=array<i64:1,2>} {
   ^bb0(%x:f32,%y:f32,%v:f32):
    %s=arith.addf %x,%y:f32
    linalg.yield %s:f32
   }
  } {sodap.bound_axis=2:i64}
 } {sodap.bound_axis=1:i64}
 return
}
}'''
(root/'input.mlir').write_text(src)
flags=['--load-dialect-plugin='+plugin,'--load-pass-plugin='+plugin]
passes=['sodap-prepare-compute-budget{lanes=64}','sodap-bank-matmul-buffers','sodap-register-block-matmul','canonicalize','cse','func.func(convert-linalg-to-loops)','func.func(fold-memref-alias-ops)','lower-affine','convert-scf-to-cf','expand-strided-metadata','lower-affine','sodap-finalize-banked-memrefs-to-llvm','convert-func-to-llvm{use-bare-ptr-memref-call-conv=true}','convert-index-to-llvm','convert-cf-to-llvm','reconcile-unrealized-casts']
run(['mlir-opt',root/'input.mlir',*flags,'--pass-pipeline=builtin.module('+','.join(passes)+')','-o',root/'lowered.mlir'])
run(['mlir-translate','--mlir-to-llvmir',root/'lowered.mlir','-o',root/'kernel.ll'])
(root/'marker.c').write_text('void __bambu_csroa_partition__(const void *p,int f,int b,int d) {}\n')
run(['clang-19','-O2','-ffp-contract=off','-shared','-fPIC',root/'kernel.ll',root/'marker.c','-o',root/'test.so'])
lib=ctypes.CDLL(str((root/'test.so').resolve()));lib.compute.argtypes=[ctypes.c_int64]*3+[ctypes.c_void_p]*5
records=[]
for m,n,k in [(1,1,1),(13,9,7),(16,16,16),(9,3,15)]:
 a=np.full((16,16),np.nan,np.float32);b=a.copy();c=np.full((16,16),777,np.float32);out=c.copy();bias=np.arange(16,dtype=np.float32)/4
 a[:m,:k]=np.arange(m*k,dtype=np.float32).reshape(m,k)%7/4;b[:k,:n]=np.arange(k*n,dtype=np.float32).reshape(k,n)%5/4
 expected=c.copy();expected_out=out.copy();expected[:m,:n]=a[:m,:k]@b[:k,:n];expected_out[:m,:n]=expected[:m,:n]+bias[:n]
 for _ in range(2):
  lib.compute(m,n,k,*[ctypes.c_void_p(v.ctypes.data) for v in [a,b,bias,c,out]])
  np.testing.assert_array_equal(c,expected);np.testing.assert_array_equal(out,expected_out)
 records.append(dict(m=m,n=n,k=k,status='PASS',runs=2))
(root/'result.json').write_text(json.dumps(records,indent=2)+'\n');print(records)
