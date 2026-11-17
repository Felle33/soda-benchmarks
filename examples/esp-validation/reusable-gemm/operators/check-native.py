#!/usr/bin/env python3
"""Validate the compiler-generated host against the executed PyTorch model.

This creates only a test harness. The host and invocation graph are compiler
outputs; the harness does not choose offloads or hardware configurations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('case',type=Path)
p.add_argument('--capture',action='store_true')
a=p.parse_args();case=a.case.resolve();repo=Path(__file__).resolve().parents[4]
spec=json.loads((case/'fixture.json').read_text());gen=case/'generated'
order=list(map(int,(gen/'argument_order.txt').read_text().split()))
assert sorted(order)==list(range(len(spec['arguments'])+1))
out=case/'native';out.mkdir(exist_ok=False)
code=['#include <vector>','#include <cstdio>','#include <cmath>','#include <cassert>',
      'extern "C" void forward_kernel('+','.join('float *' for _ in order)+');',
      'std::vector<float> read(const char *path,unsigned n){std::vector<float> v(n);FILE *f=fopen(path,"rb");assert(f);assert(fread(v.data(),4,n,f)==n);fclose(f);return v;}',
      'int main(){']
for i,arg in enumerate(spec['arguments']):
    code.append(f'auto a{i}=read({json.dumps(str(case/arg["file"]))},{arg["elements"]});')
code.append(f'std::vector<float> output({spec["outputs"]});auto ref=read({json.dumps(str(case/"reference.bin"))},{spec["outputs"]});')
args=[f'a{i}.data()' for i in range(len(spec['arguments']))]+['output.data()']
code+=['forward_kernel('+','.join(args[i] for i in order)+');',
       'unsigned errors=0;float maxerr=0;',
       f'for(unsigned i=0;i<{spec["outputs"]};i++){{float x=output[i],r=ref[i],e=fabsf(x-r);bool ok=std::isnan(r)?std::isnan(x):(std::isinf(r)?x==r:std::isfinite(x)&&e<={spec["atol"]}+{spec["rtol"]}*fabsf(r));if(!ok)errors++;if(e>maxerr)maxerr=e;}}',
       'printf("SOURCE_NATIVE outputs=%zu errors=%u max_abs=%g\\n",output.size(),errors,maxerr);return errors?1:0;}']
(out/'main.cpp').write_text('\n'.join(code)+'\n')
engine=repo/'examples/soda-plugins/lib/sodap/ExecutionEngine'
cmd=['clang++-19','-O2','-ffp-contract=off']
capture=case/'capture'
if a.capture: capture.mkdir(exist_ok=False);cmd.append('-DSODAP_GEMM_TRACE')
cmd += [str(gen/'05_kernel_riscv.ll'),str(out/'main.cpp'),str(engine/'ReusableGemmRuntime.cpp'),str(engine/'esp_prof.c'),
        '-I'+str(repo/'examples/soda-plugins/include'),'-I'+str(repo/'examples/soda-plugins/include/sodap/ExecutionEngine'),'-o',str(out/'test')]
subprocess.run(cmd,check=True)
with (out/'run.log').open('w') as log:
    completed=subprocess.run([str(out/'test')],env={**os.environ,'SODAP_GEMM_TRACE_DIR':str(capture)},stdout=log,stderr=subprocess.STDOUT)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
result=dict(status='PASS' if completed.returncode==0 else 'FAIL',
            reference='executed PyTorch model',captured=a.capture,host_sha256=sha(gen/'05_kernel_riscv.ll'),
            fixture_sha256=sha(case/'fixture.json'),runtime_sha256=sha(engine/'ReusableGemmRuntime.cpp'),
            log_sha256=sha(out/'run.log'),command=cmd)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print((out/'run.log').read_text(),end='')
raise SystemExit(completed.returncode)
