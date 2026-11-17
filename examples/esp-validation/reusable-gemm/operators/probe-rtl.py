#!/usr/bin/env python3
"""In the container: run the packed RTL harness with the timeline probe.
usage: probe-rtl.py <bambu_syn dir> <capture dir> <new output dir> [stall]"""
import subprocess,sys,shutil
from pathlib import Path
h,capture,out=map(Path,sys.argv[1:4]);stall=int(sys.argv[4]) if len(sys.argv)>4 else 0
repo=Path('/workspace');out.mkdir(parents=True,exist_ok=False)
for f in [*capture.glob('*.hex'),capture/'parameters.svh',*h.glob('*.mem')]:shutil.copy2(f,out/f.name)
subprocess.run(['python3',str(Path(__file__).resolve().parent/'make-probe.py'),str(h/'forward_kernel.v'),str(repo/'examples/esp-validation/reusable-gemm/dma64_tb.sv'),str(out/'tb.sv'),str(out/'probe.json')],check=True,stdout=subprocess.DEVNULL)
src=[h/'forward_kernel.v',h/'panda_libtech.v',repo/'examples/esp-validation/native-dma/packed_dma64.sv',out/'tb.sv']
with (out/'compile.log').open('w') as f:subprocess.run(['iverilog','-g2012','-DSODAP_PACKED_DMA','-I'+str(out),'-s','reusable_gemm_tb','-o',out/'test.vvp',*map(str,src)],stdout=f,stderr=subprocess.STDOUT,check=True)
with (out/f'simulation-{stall}.log').open('w') as f:subprocess.run(['vvp','test.vvp',f'+gemm_stall={stall}'],cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3000)
t=(out/f'simulation-{stall}.log').read_text();assert 'REUSABLE_GEMM_RTL PASS' in t,t[-2000:]
print(out,'PASS',t.count('\nEV '),'events')
