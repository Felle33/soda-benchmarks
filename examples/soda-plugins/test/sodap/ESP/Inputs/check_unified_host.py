import ctypes,json,sys
from pathlib import Path
import numpy as np
r=Path(sys.argv[1]);kind=sys.argv[2];p=json.loads((r/'network.json').read_text())
lib=ctypes.CDLL(str(r/'native.so'));ptr=ctypes.POINTER(ctypes.c_float);asptr=lambda a:a.ctypes.data_as(ptr)
lib.forward_init.argtypes=[ptr];lib.forward_init.restype=ctypes.c_int64
lib.forward_run.argtypes=[ctypes.c_int64]+[ptr]*(len(p['input_elements'])+1)
lib.forward_close.argtypes=[ctypes.c_int64]
params=np.fromfile(r/'parameters.bin',dtype='<f4');original=params.copy();ctx=lib.forward_init(asptr(params))
assert [e['operator'] for e in p['engines']]==(['add','gemm'] if kind=='live-c' else ['gemm'])
calls=[x for x in p['invocations'] if x['execution']=='accelerator']
assert len(calls)==(2 if kind=='live-c' else 1) and calls[0]['runtime']=='gemm_full_source'
a=np.arange(1,p['input_elements'][0]+1,dtype=np.float32);b=np.arange(1,7,dtype=np.float32);c=np.full(p['output_elements'],2,dtype=np.float32)
try:
 for alpha,beta in [(0.5,2.0),(2.0,0.0),(-1.0,0.25)]:
  if kind=='expression':
   inputs=[a,b,c,np.array([alpha],np.float32),np.array([beta],np.float32)]
   expected=alpha*(a.reshape(-1,3)@b.reshape(3,2))+beta*c.reshape(-1,2)
  else:
   inputs=[a,b]+([c] if kind=='live-c' else [])
   expected=a.reshape(-1,3)@b.reshape(3,2)+(2*c.reshape(-1,2) if kind=='live-c' else original.reshape(1,2))
  out=np.full(p['output_elements'],np.nan,np.float32)
  lib.forward_run(ctx,*map(asptr,inputs),asptr(out))
  np.testing.assert_allclose(out,expected.flatten(),rtol=1e-6,atol=1e-6)
  np.testing.assert_array_equal(c,2);np.testing.assert_array_equal(params,original)
  a+=1
finally:lib.forward_close(ctx)
print('Unified host numerical/persistence PASS',kind)
