#!/usr/bin/env python3
"""Run the common native-DMA RTL testbench using Verilator for larger engines."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('hardware',type=Path)
    p.add_argument('capture',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--stalls',type=int,nargs='+',default=[0,50000])
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[3];hw=a.hardware.resolve()/'bambu_syn'
    for f in [*a.capture.glob('*.hex'),a.capture/'parameters.svh',*hw.glob('*.mem')]:shutil.copy2(f,out/f.name)
    tb=repo/'examples/esp-validation/reusable-gemm/dma64_tb.sv'
    sources=[hw/'forward_kernel.v',hw/'panda_libtech.v',repo/'examples/esp-validation/native-dma/packed_dma64.sv',tb]
    state=dict(status='RUNNING',simulator='Verilator',runs=[],
               source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    try:
        cmd=['verilator','--binary','--timing','-Wno-fatal','-j','4','--Mdir',str(out/'obj'),
             '--top-module','reusable_gemm_tb','-DSODAP_PACKED_DMA','-I'+str(out),*map(str,sources)]
        with (out/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        for stall in a.stalls:
            log=out/f'simulation-{stall}.log'
            with log.open('w') as f:subprocess.run([out/'obj/Vreusable_gemm_tb',f'+gemm_stall={stall}'],cwd=out,
                stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
            text=log.read_text();passed='REUSABLE_GEMM_RTL PASS' in text
            state['runs'].append(dict(stall=stall,passed=passed,cycles=list(map(int,re.findall(r'errors=0 cycles=(\d+)',text)))))
            if not passed:raise RuntimeError('RTL success marker missing')
        state['status']='PASS'
    except BaseException as error:
        state.update(status='FAIL',error=repr(error));raise
    finally:
        (out/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state['runs'],indent=2))


if __name__=='__main__':main()
