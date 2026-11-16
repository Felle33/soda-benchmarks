#!/usr/bin/env python3
"""Run a command in the review container (tools/review.py's image, mounts and
environment), with the compiler at /workspace, exp2/ at /work and the
Experiment 2 release at /exp2. usage: incontainer.py <command> [args...]"""
import os,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[4].parent;exp='exp2'
IMAGE='sha256:192a841f6f5cf6d15103725d4cb4c2ed88b275b4ee334a53f158bc9c3aa33ec4'
cmd=['docker','run','--rm','--user',f'{os.getuid()}:{os.getgid()}','-w','/workspace']
for src,dst,ro in [(root/exp/'compiler','/workspace',False),(root/exp,'/work',False),(root/'tools','/review-tools',True),
 (root/'releases/exp2','/exp2',True),(root/'releases/exp3','/exp3',True),(root/'sdk','/home/tommaso/opt/bambu',True),
 (Path('/usr/bin/iverilog'),'/usr/bin/iverilog',True),(Path('/usr/bin/vvp'),'/usr/bin/vvp',True),
 (Path('/usr/lib/x86_64-linux-gnu/ivl'),'/usr/lib/x86_64-linux-gnu/ivl',True)]:
 cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
for name in ['opt_customSROA.so','opt_dumpBambuIrSSA.so']:
 cmd+=['--mount',f'type=bind,src={root}/releases/exp3/artifacts/toolchain/plugins/{name},dst=/home/tommaso/opt/bambu/lib/panda/clang-19/{name},readonly']
setup='''set -euo pipefail
source /opt/panda/settings.sh
source /home/tommaso/opt/bambu/settings.sh
export PATH=/workspace/scripts:/opt/soda-opt/bin:/opt/llvm-project/bin:/opt/torch-mlir/bin:$PATH
export PYTHONPATH=/exp3/artifacts/toolchain/python:/opt/llvm-project/python_packages/mlir_core:/opt/torch-mlir/python_packages/torch_mlir:/opt/soda-opt/python_packages/soda:/workspace
export MLIR_PLUGIN_PATH=/workspace/build/plugin/lib/SODAPlugin.so SODAP_LIB_DIR=/workspace/build/plugin/lib SODAP_DIR=/workspace/examples/soda-plugins
export BAMBU_BIN=/home/tommaso/opt/bambu/bin/bambu CLANG=/usr/bin/clang-19 CC=/usr/bin/clang-19 RISCV_CLANG=/usr/bin/clang-19
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export LCS_DMA_WORD_STRIDE=2 LCS_PROFILE=true LCS_REUSABLE_GEMM=false LCS_OUTSTANDING_LOADS=true
exec "$@"
'''
sys.exit(subprocess.run(cmd+['--entrypoint','/bin/bash',IMAGE,'-c',setup,'ops',*sys.argv[1:]]).returncode)
