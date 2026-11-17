"""Numerical/lifetime oracle for host aliasing and destructive GEMM ownership."""
import ctypes,json,sys
from pathlib import Path
import numpy as np
r=Path(sys.argv[1]);kind=sys.argv[2];p=json.loads((r/'network.json').read_text())
calls=p['invocations'];copies=[x for x in calls if x['runtime']=='repack']
assert len(copies)==int(kind in ['layout','bias','live-input','live-intermediate','hoist','consume-live-chain'])
if kind not in ['live-intermediate','consume-live-chain'] and copies:
    assert copies[0]['index']<next(x['index'] for x in calls if x['execution']=='accelerator')
if kind in ['consume','consume-chain']:
    for x in calls:
        assert x['output']==p['input_buffers'][2]
if kind in ['live-input','bias']:
    x=next(x for x in calls if x['execution']=='accelerator')
    protected=p['input_buffers'][2] if kind=='live-input' else p['parameter_bindings'][0]['buffer']
    assert x['output']!=protected
lib=ctypes.CDLL(str(r/'native.so'));ptr=ctypes.POINTER(ctypes.c_float);asptr=lambda a:a.ctypes.data_as(ptr)
lib.forward_init.argtypes=[ptr];lib.forward_init.restype=ctypes.c_int64
lib.forward_run.argtypes=[ctypes.c_int64]+[ptr]*(len(p['input_elements'])+1)
lib.forward_close.argtypes=[ctypes.c_int64]
lib.esp_prof_count.argtypes=[ctypes.c_uint32];lib.esp_prof_count.restype=ctypes.c_uint32
lib.esp_prof_reset()
params=np.fromfile(r/'parameters.bin',dtype='<f4');original=params.copy();ctx=lib.forward_init(asptr(params))
try:
    for repetition in range(2):
        inputs=[np.arange(1,n+1,dtype=np.float32)+repetition for n in p['input_elements']]
        snapshots=[x.copy() for x in inputs];a,b=inputs[:2]
        if kind=='layout':expected=a.reshape(3,2)@b.reshape(2,2)
        else:expected=a.reshape(-1,3)@b.reshape(3,2)
        if kind in ['consume','hoist']:expected+=inputs[2].reshape(expected.shape)
        if kind=='consume-chain':expected=2*expected+inputs[2].reshape(expected.shape)
        if kind=='live-input':expected+=2*inputs[2].reshape(expected.shape)
        if kind=='live-intermediate':expected*=3
        if kind=='alias-live':expected*=2
        if kind=='consume-live-chain':expected=3*expected+2*inputs[2].reshape(expected.shape)
        if kind=='bias':expected+=original.reshape(expected.shape)
        out=np.full(p['output_elements'],np.nan,np.float32)
        lib.forward_run(ctx,*map(asptr,inputs),asptr(out))
        np.testing.assert_allclose(out,expected.flatten(),rtol=1e-6,atol=1e-6)
        for value,before in zip(inputs,snapshots):np.testing.assert_array_equal(value,before)
        np.testing.assert_array_equal(params,original)
finally:lib.forward_close(ctx)
for region in [1,3,5]:assert lib.esp_prof_count(region)>0
print('Host copies, ownership, repeated execution and profiling PASS',kind)
