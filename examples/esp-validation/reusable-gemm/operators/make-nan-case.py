#!/usr/bin/env python3
"""In the container: MINI gemm with beta = 0 and C all NaN (G8, D2).
usage: make-nan-case.py <new source root>; writes <root>/MINI/gemm."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
src=Path('/exp2/artifacts/campaign/cases/MINI/gemm');out=Path(sys.argv[1])/'MINI'/'gemm';out.mkdir(parents=True,exist_ok=False)
spec=json.loads((src/'fixture.json').read_text())
args={a['name']:np.fromfile(src/a['file'],dtype='<f4').reshape(a['shape']) for a in spec['arguments']}
for f in ['01_tosa.mlir']:shutil.copy2(src/f,out/f)
args['beta']=np.zeros((),dtype='<f4');args['C']=np.full(args['C'].shape,np.nan,dtype='<f4')
for a in spec['arguments']:
    data=np.asarray(args[a['name']],dtype='<f4');(out/a['file']).write_bytes(data.tobytes());a['sha256']=hashlib.sha256(data.tobytes()).hexdigest()
# alpha * (A.B), summed in order in f32 as the hardware does
A,B=args['A'].astype('<f4'),args['B'].astype('<f4');m,k=A.shape;n=B.shape[1]
acc=np.zeros((m,n),dtype='<f4')
for q in range(k):acc=(acc+A[:,q:q+1]*B[q:q+1,:]).astype('<f4')
ref=(np.float32(args['alpha'])*acc).astype('<f4');assert np.isfinite(ref).all()
(out/'reference.bin').write_bytes(ref.tobytes());spec['variant']='beta=0, C all NaN';(out/'fixture.json').write_text(json.dumps(spec,indent=2)+'\n')
print('written',out,'alpha',float(args['alpha']),'shape',ref.shape)
