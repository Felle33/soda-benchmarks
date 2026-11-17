#!/usr/bin/env python3
"""Convolution RTL qualification and passive throughput counters.

Uses the production adapter unchanged. All generated code is a testbench;
hierarchy discovery only observes HLS arithmetic handshakes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def arithmetic_paths(text):
    text=re.sub(r'//[^\n]*|/\*.*?\*/','',text,flags=re.S)
    modules=dict(re.findall(r'\bmodule\s+(\w+)\b(.*?)\bendmodule\b',text,re.S))
    graph={name:[(m[1],m[2]) for m in re.finditer(r'^\s*(\w+)\s+(?:#\s*\(.*?\)\s*)?(\w+)\s*\(',body,re.M|re.S)
                 if m[1] in modules] for name,body in modules.items()}
    found=[]
    def visit(name,path):
        if name in ('p_float_mule8m23b_127nih','p_float_adde8m23b_127nih'):
            found.append(path);return
        for child,instance in graph[name]:visit(child,path+'.'+instance)
    visit('forward_kernel','dut.u_core')
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['hardware','capture','output']:p.add_argument(name,type=Path)
    p.add_argument('--simulator',choices=['verilator','icarus'],default='verilator')
    p.add_argument('--stalls',type=int,nargs='+',default=[0,50000])
    p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[3];hw=a.hardware.resolve()/'bambu_syn'
    for f in [*a.capture.glob('*.hex'),a.capture/'parameters.svh',*hw.glob('*.mem')]:shutil.copy2(f,out/f.name)
    core=[hw/'forward_kernel.v',hw/'panda_libtech.v']
    paths=arithmetic_paths('\n'.join(f.read_text() for f in core));assert paths
    source=repo/'examples/esp-validation/reusable-gemm/dma64_tb.sv'
    tb=source.read_text().replace('parameter TIMEOUT_CYCLES=2000000','parameter TIMEOUT_CYCLES=100000000')
    tb=tb.replace('reg clk=0;', '''integer stall_mode=0;
  initial begin if(!$value$plusargs("gemm_stall=%d",stall_mode)) stall_mode=0; end
  reg clk=0;''')
    tb=tb.replace('(hold_read || cycles%5!=0)','(hold_read || stall_mode==0 || cycles%5!=0)')
    tb=tb.replace('&& cycles%7!=0','&& (stall_mode==0 || cycles%7!=0)').replace('&& cycles%11!=0','&& (stall_mode==0 || cycles%11!=0)')
    n=len(paths)
    counters=f'''\n  wire [{n-1}:0] fp_start={{'''+','.join(q+'.start_port' for q in paths)+'''};
  wire ['''+str(n-1)+''':0] fp_done={'''+','.join(q+'.done_port' for q in paths)+'''};
  reg ['''+str(n-1)+''':0] fp_pending=0;
  integer dma_only=0,fp_only=0,both_active=0,neither_active=0,first_cycle=0;
  wire dma_active=(rleft!=0 || wleft!=0 || rv || wv);
  wire fp_active=|(fp_pending|fp_start);
  always @(posedge clk) if(rst) begin
    fp_pending<=(fp_pending|fp_start)&~fp_done;
    if(start) begin
      dma_only<=0;fp_only<=0;both_active<=0;neither_active<=0;first_cycle<=cycles;
    end else if(!done) begin
      if(dma_active && fp_active)both_active<=both_active+1;
      else if(dma_active)dma_only<=dma_only+1;
      else if(fp_active)fp_only<=fp_only+1;
      else neither_active<=neither_active+1;
    end
  end
'''
    tb=tb.replace('  function real fp32',counters+'  function real fp32')
    tb=tb.replace('      if(errors) $fatal', '''      $display("CONV_METRICS invocation=%0d cycles=%0d dma_only=%0d fp_only=%0d both=%0d neither=%0d",invocation,cycles-first_cycle,dma_only,fp_only,both_active,neither_active);
      if(errors) $fatal''')
    (out/'tb.sv').write_text(tb)
    adapter=repo/'examples/esp-validation/native-dma/packed_dma64.sv'
    sources=[*core,adapter,out/'tb.sv'];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    result=dict(status='RUNNING',simulator=a.simulator,runs=[],sources={str(f):sha(f) for f in sources},
                arithmetic_instances=paths,cases=json.loads((a.capture/'cases.json').read_text()),
                counter_definition='Disjoint cycles with DMA request/in-flight transaction only, floating-point operation in flight only, both, or neither. Neither includes local-memory/address/control work; these are not additive task latencies.')
    started=time.monotonic()
    try:
        if a.simulator=='verilator':
            cmd=['verilator','--binary','--timing','-Wno-fatal','-j','4','--Mdir',str(out/'obj'),
                 '--top-module','reusable_gemm_tb','-DSODAP_PACKED_DMA','-I'+str(out),*map(str,sources)]
            execute=[str(out/'obj/Vreusable_gemm_tb')]
        else:
            cmd=['iverilog','-g2012','-DSODAP_PACKED_DMA','-I'+str(out),'-s','reusable_gemm_tb','-o',str(out/'test.vvp'),*map(str,sources)]
            execute=['vvp',str(out/'test.vvp')]
        with (out/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        for stall in a.stalls:
            log=out/f'simulation-{stall}.log'
            with log.open('w') as f:subprocess.run([*execute,f'+gemm_stall={stall}'],cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=a.timeout)
            text=log.read_text();assert 'REUSABLE_GEMM_RTL PASS' in text
            metrics=[dict(zip(['invocation','cycles','dma_only','fp_only','both','neither'],map(int,m))) for m in re.findall(r'CONV_METRICS invocation=(\d+) cycles=(\d+) dma_only=(\d+) fp_only=(\d+) both=(\d+) neither=(\d+)',text)]
            for row,case in zip(metrics,result['cases']['cases']):row['macs_per_cycle']=case['macs']/row['cycles']
            result['runs'].append(dict(stall=stall,status='PASS',metrics=metrics,log_sha256=sha(log)))
        result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL',error=repr(error));raise
    finally:
        result['wall_seconds']=time.monotonic()-started
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['runs'],indent=2))


if __name__=='__main__':main()
