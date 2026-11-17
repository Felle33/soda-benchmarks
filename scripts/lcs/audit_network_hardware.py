#!/usr/bin/env python3
"""Inventory reachable FP32 units in Bambu RTL, counting task multiplicity.

Verification only. No hardware selection or RTL modification is performed.
DSP counts, BRAM mapping, initiation intervals and routed timing are separate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for kind in ('conv','add','gemm'):p.add_argument('--'+kind,required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();report={}
    for kind in ('conv','add','gemm'):
        folder=getattr(a,kind);rtl=folder/'bambu_syn'
        sources=[rtl/'forward_kernel.v',rtl/'panda_libtech.v']
        text='\n'.join(f.read_text() for f in sources)
        text=re.sub(r'//[^\n]*|/\*.*?\*/','',text,flags=re.S)
        modules=dict(re.findall(r'\bmodule\s+(\w+)\b(.*?)\bendmodule\b',text,re.S))
        graph={name:[m[1] for m in re.finditer(r'^\s*(\w+)\s+(?:#\s*\(.*?\)\s*)?(\w+)\s*\(',body,re.M|re.S)
                     if m[1] in modules] for name,body in modules.items()}
        counts=Counter();dividers=Counter()
        def visit(name,path):
            if name in path:raise ValueError('Recursive RTL hierarchy: '+name)
            if name.startswith(('p_udiv','p_umod','p_sdiv','p_smod','p_div','p_mod')):
                dividers[name]+=1;return
            if name in ('p_float_mule8m23b_127nih','p_float_adde8m23b_127nih'):
                counts[name]+=1;return
            for child in graph[name]:visit(child,path+(name,))
        visit('forward_kernel',())
        units={'mul':counts['p_float_mule8m23b_127nih'],'add':counts['p_float_adde8m23b_127nih']}
        lanes=json.loads((folder/'compute_config.json').read_text())['fp_units_per_task']
        expected={'mul':0 if kind=='add' else lanes,'add':lanes}
        if units!=expected:raise ValueError(f'{kind}: installed arithmetic {units}, requested {expected}')
        report[kind]=dict(fp32=units,requested_lanes=lanes,
            dynamic_integer_dividers=dict(dividers),
            local_arrays=json.loads((folder/'compute_storage.json').read_text()),
            source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    result=dict(status='PASS',scope='Reachable HLS RTL arithmetic; not routed FPGA resources',
                operators=report,total_fp32={k:sum(v['fp32'][k] for v in report.values()) for k in ('mul','add')})
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['total_fp32']))


if __name__=='__main__':main()
