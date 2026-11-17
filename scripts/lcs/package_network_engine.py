#!/usr/bin/env python3
"""Namespace Bambu RTL and bind the unchanged native-DMA interface adapter.

No arithmetic, scheduling, banking or datapath transformations occur here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('hardware',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--kind',choices=['conv','add','gemm'],required=True)
    a=p.parse_args();source=a.hardware.resolve()/'bambu_syn';out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    # Reject a stale synthesis from another LLVM input.
    assert sha(source/'accelerator.ll')==sha(a.hardware/'05_hw_acc.ll')
    inputs=[source/'forward_kernel.v',source/'panda_libtech.v']
    code='\n'.join(p.read_text() for p in inputs)
    names=re.findall(r'^module\s+([\w$]+)\b',code,re.M)
    assert len(names)==len(set(names)) and 'forward_kernel' in names
    mapping={n:a.kind+'__'+n for n in names}
    mapping.update({n:a.kind+'__'+n for n in re.findall(r'`define\s+(\w+)',code)})
    tokens=re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|\b[A-Za-z_][\w$]*\b',re.S)
    code=tokens.sub(lambda m:mapping.get(m[0],m[0]),code)
    memories={}
    for f in source.glob('*.mem'):
        name=a.kind+'__'+f.name
        code=code.replace('"'+f.name+'"','"'+name+'"')
        shutil.copy2(f,out/name);memories[name]=sha(f)
    (out/'engine.v').write_text(code)
    repo=Path(__file__).resolve().parents[2]
    wrapper=repo/'examples/esp-validation/native-dma/packed_dma64.sv'
    top='cnn_'+a.kind+'_bambu_basic_dma64'
    text=wrapper.read_text().replace('module sqr_bambu_basic_dma64','module '+top)
    text=text.replace('forward_kernel u_core(',a.kind+'__forward_kernel u_core(')
    (out/(top+'.sv')).write_text(text)
    report=dict(kind=a.kind,top=top,hardware_llvm_sha256=sha(a.hardware/'05_hw_acc.ll'),
                inputs={f.name:sha(f) for f in inputs},module_renaming=mapping,
                memory_hashes=memories,adapter_source_sha256=sha(wrapper),
                files={f.name:sha(f) for f in out.iterdir() if f.is_file()},
                transformations=['namespace RTL symbols and initialization filenames',
                                 'rename adapter module and core binding; DMA logic unchanged'])
    (out/'packaging.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
