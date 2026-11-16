#!/usr/bin/env python3
"""In the container: packed RTL harness (as tools/exp2.py rtl) for one case.
usage: ops-rtl.py <hardware dir with bambu_syn> <capture dir> <new output dir> [stall ...]"""
import hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
hw,capture,out=map(Path,sys.argv[1:4]);stalls=list(map(int,sys.argv[4:])) or [0,50000]
h=hw/'bambu_syn';repo=Path('/workspace');assert (h/'forward_kernel.v').is_file()
out.mkdir(parents=True,exist_ok=False)
for f in [*capture.glob('*.hex'),capture/'parameters.svh',*h.glob('*.mem')]:shutil.copy2(f,out/f.name)
shutil.copy2(repo/'examples/esp-validation/reusable-gemm/dma64_tb.sv',out/'tb.sv')
source=[h/'forward_kernel.v',h/'panda_libtech.v',repo/'examples/esp-validation/native-dma/packed_dma64.sv',out/'tb.sv']
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
state={'status':'RUNNING','source_sha256':{str(f):sha(f) for f in source},'runs':[]}
try:
    with (out/'compile.log').open('w') as f:subprocess.run(['iverilog','-g2012','-DSODAP_PACKED_DMA','-I'+str(out),'-s','reusable_gemm_tb','-o',out/'test.vvp',*source],stdout=f,stderr=subprocess.STDOUT,check=True)
    for stall in stalls:
        log=out/f'simulation-{stall}.log'
        with log.open('w') as f:subprocess.run(['vvp','test.vvp',f'+gemm_stall={stall}'],cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=3000)
        text=log.read_text();ok='REUSABLE_GEMM_RTL PASS' in text
        state['runs'].append({'stall':stall,'pass':ok,'cycles':list(map(int,re.findall(r'errors=0 cycles=(\d+)',text)))})
        print('stall',stall,'PASS' if ok else 'FAIL',state['runs'][-1]['cycles'],flush=True)
        if not ok:print(text[-1500:]);raise SystemExit(1)
    state['status']='PASS'
finally:
    if state['status']!='PASS':state['status']='FAIL'
    (out/'result.json').write_text(json.dumps(state,indent=2)+'\n')
