#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Record a pipeline's DMA contract; select an adapter without mixing old IR.

This records provenance, not a proof for arbitrary manually modified hardware.
Only tosa_to_lcs.sh emits this contract after successful admitted DMA lowering.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

FILES = ('05_hw_acc.ll', '05_kernel_riscv.ll', 'architecture.xml', 'layout.txt')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_socket(directory, platform, require_generated=True):
    """Check the emitted hardware contract against generated ESP configuration."""
    contract = json.loads((directory/'native_dma_contract.json').read_text())
    if contract.get('version') != 2:
        return
    config = platform/'socgen/esp/.esp_config'
    match = re.search(r'^CONFIG_DMA_NOC_WIDTH\s*=\s*(\d+)\s*$', config.read_text(), re.M)
    if not match or int(match[1]) != contract['dma_data_width']:
        raise ValueError(f'Packed hardware DMA width does not match {config}')
    if not require_generated:
        return
    vhdl = platform/'socgen/esp/esp_global.vhd'
    match = re.search(r'constant\s+DMA_NOC_WIDTH\s*:\s*integer\s*:=\s*(\d+)', vhdl.read_text())
    if not match or int(match[1]) != contract['dma_data_width']:
        raise ValueError(f'Packed hardware DMA width does not match {vhdl}')

def record(directory):
    layout = dict(line.split('=', 1) for line in (directory/'layout.txt').read_text().splitlines())
    contract = {
        'version': 1,
        'producer': 'tosa_to_lcs: receiver admission and retained output-tile readiness',
        'dma_word_stride': int(layout['dma_word_stride']),
        'sha256': {name: digest(directory/name) for name in FILES},
    }
    if layout.get('dma_abi') == 'packed':
        width=int(layout['dma_data_width'])
        if width!=64: raise ValueError('Bambu backend supports packed DMA64 only')
        contract.update(version=2,dma_data_width=width,element_bits=32,
            layout='beat-aligned bases and row strides; dense lanes within rows',
            request_units='beats',host_address_units='bytes',lane_order='low-first',
            wrapper='packed_dma64',dma_word_stride=None)
    repo = Path(__file__).resolve().parents[2]
    if contract['version'] == 2:
        implementation = [p for folder in ('examples/soda-plugins/include',
            'examples/soda-plugins/lib', 'scripts/lcs', 'scripts/reusable_nn') for p in (repo/folder).rglob('*')
            if p.is_file() and p.suffix in ('.h','.cpp','.inc','.td','.py')]
        implementation += [repo/'scripts/tosa_to_lcs.sh',repo/'scripts/tosa_to_reusable_gemm.sh',
            repo/'examples/esp-validation/native-dma/packed_dma64.sv']
        contract['source_sha256'] = {str(p.relative_to(repo)): digest(p) for p in sorted(implementation)}
        contract['compiler_base_commit'] = subprocess.check_output(
            ['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        plugin = os.environ.get('MLIR_PLUGIN_PATH')
        if plugin: contract['compiler_plugin_sha256'] = digest(Path(plugin))
        contract['wrapper_sha256'] = digest(repo/'examples/esp-validation/native-dma/packed_dma64.sv')
    (directory/'native_dma_contract.json').write_text(json.dumps(contract, indent=2)+'\n')

def select(directory, mode):
    repo = Path(__file__).resolve().parents[2]
    if mode == 'buffered':
        if (directory/'native_dma_contract.json').exists() and json.loads((directory/'native_dma_contract.json').read_text()).get('version')==2:
            raise ValueError('Packed ABI requires packed adapter; legacy buffered wrapper is incompatible')
        return repo/'examples/esp-validation/generated-gemm/sqr_bambu_basic_dma64.sv'
    path = directory/'native_dma_contract.json'
    if not path.exists():
        raise ValueError('Thin wrapper requires fresh tosa_to_lcs output; use --wrapper buffered for legacy artifacts')
    contract = json.loads(path.read_text())
    packed=contract.get('version')==2 and contract.get('dma_data_width')==64
    if not packed and (contract['version'] != 1 or contract['dma_word_stride'] != 2):
        raise ValueError('Thin 64-bit wrapper requires native DMA contract v1 with dma_word_stride=2')
    for name in FILES:
        if digest(directory/name) != contract['sha256'][name]:
            raise ValueError(f'Stale native DMA contract: {name}')
    for generated, source in (('accelerator.ll','05_hw_acc.ll'), ('architecture.xml','architecture.xml')):
        if digest(directory/'bambu_syn'/generated) != contract['sha256'][source]:
            raise ValueError(f'Stale Bambu synthesis input: {generated}')
    wrapper=repo/('examples/esp-validation/native-dma/packed_dma64.sv' if packed else 'examples/esp-validation/native-dma/thin_dma64.sv')
    if packed and contract.get('wrapper_sha256') and digest(wrapper)!=contract['wrapper_sha256']:
        raise ValueError('Packed wrapper differs from the emitted contract; regenerate/requalify')
    return wrapper

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['record', 'select'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--wrapper', choices=['thin', 'buffered'], default='thin')
    args = parser.parse_args()
    try:
        if args.action == 'record':
            record(args.directory)
        else:
            print(select(args.directory, args.wrapper))
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f'{error}\n')

if __name__ == '__main__':
    main()
