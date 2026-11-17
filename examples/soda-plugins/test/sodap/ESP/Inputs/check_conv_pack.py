# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import ctypes,json,sys
from pathlib import Path
import numpy as np
r=Path(sys.argv[1]);lib=ctypes.CDLL(str((r/'native.so').resolve()))
lib.forward_init.argtypes=[ctypes.c_void_p];lib.forward_init.restype=ctypes.c_int64
lib.forward_run.argtypes=[ctypes.c_int64,ctypes.c_void_p,ctypes.c_void_p];lib.forward_close.argtypes=[ctypes.c_int64]
p=np.fromfile(r/'parameters.bin',dtype=np.float32);before=p.copy();x=np.array([1,2,3,4],np.float32);saved=x.copy();y=np.empty(6,np.float32)
ctx=lib.forward_init(p.ctypes.data)
for _ in range(2):
 y[:]=np.nan;lib.forward_run(ctx,x.ctypes.data,y.ctypes.data)
 np.testing.assert_array_equal(y,np.array([6,13,20,15,30,45],np.float32));np.testing.assert_array_equal(p,before);np.testing.assert_array_equal(x,saved)
lib.forward_close(ctx)
plan=json.loads((r/'network.json').read_text());assert [i['descriptor_version'] for i in plan['invocations'] if i['execution']=='accelerator']==[4,3]
assert len(plan['parameter_bindings'])==3,'shared weight must retain its original non-convolution binding'
print('CONV_SETUP_PACK_SHARED_PARAMETER PASS twice')
