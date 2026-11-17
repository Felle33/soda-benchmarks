#!/usr/bin/env python3
"""In the container: lower one frozen fixture in operators mode + native check.
usage: ops-lower.py <run-dir under /work/runs> <DATASET> <kernel> [ENV=VALUE ...]"""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
run,dataset,kernel=sys.argv[1:4];extra=dict(x.split('=',1) for x in sys.argv[4:])
repo=Path('/workspace');src=Path(os.environ.get('OPS_SRC','/exp2/artifacts/campaign/cases'))/dataset/kernel
out=Path('/work/runs')/run/'cases'/dataset/kernel;out.mkdir(parents=True,exist_ok=False)
spec=json.loads((src/'fixture.json').read_text())
for name in ['fixture.json','01_tosa.mlir','reference.bin',*[x['file'] for x in spec['arguments']]]:shutil.copy2(src/name,out/name)
generated=out/'generated'
env={**os.environ,'LCS_TILES':'operators','LCS_DMA_ABI':'packed','LCS_DMA_DATA_WIDTH':'64',**extra}
(out/'lowering-env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('LCS_')},indent=2,sort_keys=True)+'\n')
with (out/'compile.log').open('w') as log:subprocess.run(['bash',str(repo/'scripts/tosa_to_lcs.sh'),str(out/'01_tosa.mlir'),str(generated)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
assert 'tiles=operators' in (generated/'layout.txt').read_text().split('\n')
order=list(map(int,(generated/'argument_order.txt').read_text().split()));assert sorted(order)==list(range(len(spec['arguments'])+1)),order
code='''#include <vector>
#include <cstdio>
#include <cmath>
#include <cassert>
extern "C" void forward_kernel('''+','.join('float *' for _ in order)+''');
std::vector<float> read(const char *path, unsigned n) {std::vector<float> v(n);FILE *f=fopen(path,"rb");assert(f);assert(fread(v.data(),4,n,f)==n);fclose(f);return v;}
int main(){
'''
for i,arg in enumerate(spec['arguments']):code+=f'auto a{i}=read("{out/arg["file"]}",{arg["elements"]});\n'
code+=f'std::vector<float> output({spec["outputs"]});auto ref=read("{out/"reference.bin"}",{spec["outputs"]});\n'
args=[f'a{i}.data()' for i in range(len(spec['arguments']))]+['output.data()']
code+='forward_kernel('+','.join(args[i] for i in order)+');\nunsigned errors=0;float maxerr=0;\n'
code+=f'for(unsigned i=0;i<{spec["outputs"]};i++){{float e=fabsf(output[i]-ref[i]);if(!std::isfinite(output[i]) || e>{spec["atol"]}+{spec["rtol"]}*fabsf(ref[i]))errors++;if(e>maxerr)maxerr=e;}}\n'
code+='printf("OPS_NATIVE outputs=%zu errors=%u max_abs=%g\\n",output.size(),errors,maxerr);return errors?1:0;}\n'
(out/'native.cpp').write_text(code);capture=out/'capture';capture.mkdir()
E=repo/'examples/soda-plugins/lib/sodap/ExecutionEngine'
subprocess.run(['clang++-19','-O2','-ffp-contract=off','-DSODAP_GEMM_TRACE',str(generated/'05_kernel_riscv.ll'),str(out/'native.cpp'),str(E/'ReusableGemmRuntime.cpp'),str(E/'esp_prof.c'),'-I'+str(repo/'examples/soda-plugins/include'),'-I'+str(repo/'examples/soda-plugins/include/sodap/ExecutionEngine'),'-o',str(capture/'native')],check=True)
with (capture/'native.log').open('w') as log:r=subprocess.run([str(capture/'native')],env={**env,'SODAP_GEMM_TRACE_DIR':str(capture)},stdout=log,stderr=subprocess.STDOUT)
print((capture/'native.log').read_text().strip());assert r.returncode==0
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
(out/'result.json').write_text(json.dumps({'status':'PASS','mode':'operators','dataset':dataset,'kernel':kernel,'compiler_head':subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'tile_sizes':(generated/'tile_sizes.txt').read_text().strip(),'host_sha256':sha(generated/'05_kernel_riscv.ll'),'hardware_sha256':sha(generated/'05_hw_acc.ll')},indent=2)+'\n')
print(dataset,kernel,'PASS',(generated/'tile_sizes.txt').read_text().strip())
