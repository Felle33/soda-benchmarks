#!/usr/bin/env python3
"""G7 oracles of operators mode against the measured Experiment 2 flow.

usage: check_gemm_oracles.py <run cases dir> <release cases dir> [out.json]

Each case is <dataset>/<kernel>/ with generated/ (tosa_to_lcs.sh, operators
mode) and capture/ (the host run on ReusableGemmRuntime with
SODAP_GEMM_TRACE: native.log, initial-<i>.hex, ...). Checks:

  (a) the launches are the measured flow's (generated/invocations.json of the
      release): arena size, M, N, K, the A, B, C offsets and alpha and beta
      in each descriptor; where the release has a capture, every captured
      file is byte-identical;
  (b) one accelerator: a single 05_hw_acc.ll hash over every case;
  (c) storage within the budget: the scheduled logical storage
      (tiles/forward_kernel/schedule_cost.json) and the padded local arrays
      (compute_storage.json).
"""
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

GEMM_FIELD_ALPHA, GEMM_FIELD_BETA = 11, 12


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def scalar(case, spec, name):
    for arg in spec['arguments']:
        if arg['name'] == name:
            return struct.unpack('<f', (case / arg['file']).read_bytes()[:4])[0]
    raise KeyError(name)


def check_case(case, release):
    errors = []
    spec = json.loads((case / 'fixture.json').read_text())
    ref = json.loads((release / 'generated' / 'invocations.json').read_text())
    log = (case / 'capture' / 'native.log').read_text()
    opened = re.search(r'open instance=\d+ arena_bytes=(\d+)', log)
    if not opened or int(opened.group(1)) != ref['arena_bytes']:
        errors.append(f"arena bytes {opened and opened.group(1)} != {ref['arena_bytes']}")
    launches = [dict(zip('MNKABC', map(int, m))) for m in re.findall(
        r'launch=\d+ instance=\d+ M=(\d+) N=(\d+) K=(\d+) A=(\d+) B=(\d+) C=(\d+)', log)]
    if len(launches) != len(ref['invocations']):
        errors.append(f"{len(launches)} launches, expected {len(ref['invocations'])}")
    offset = {b['id']: b['offset_bytes'] for b in ref['buffers']}
    for i, (got, want) in enumerate(zip(launches, ref['invocations'])):
        expect = {'M': want['m'], 'N': want['n'], 'K': want['k'],
                  'A': offset[want['a']], 'B': offset[want['b']], 'C': offset[want['c']]}
        if got != expect:
            errors.append(f"launch {i}: {got} != {expect}")
        words = (case / 'capture' / f'initial-{i}.hex').read_text().split()
        # Descriptor fields in the low 32 bits of 8-byte slots.
        bits = {f: int(words[2 * f], 16) for f in (GEMM_FIELD_ALPHA, GEMM_FIELD_BETA)}
        for role, field in (('alpha', GEMM_FIELD_ALPHA), ('beta', GEMM_FIELD_BETA)):
            value = want[f'{role}_constant']
            if f'{role}_argument' in want:
                value *= scalar(case, spec, role)
            got_bits = bits[field]
            want_bits = struct.unpack('<I', struct.pack('<f', f32(value)))[0]
            if got_bits != want_bits:
                errors.append(f"launch {i}: {role} bits {got_bits:08x} != {want_bits:08x}")
    compared = []
    ref_capture = release / 'capture'
    if ref_capture.is_dir():
        for f in sorted(ref_capture.iterdir()):
            if f.suffix in ('.hex', '.svh'):
                compared.append(f.name)
                if (case / 'capture' / f.name).read_bytes() != f.read_bytes():
                    errors.append(f"capture {f.name} differs from the release")
    generated = case / 'generated'
    cost = json.loads((generated / 'tiles' / 'forward_kernel' / 'schedule_cost.json').read_text())
    storage = json.loads((generated / 'compute_storage.json').read_text())
    budget = storage['budget_bytes']
    if cost['plm_bytes'] > budget:
        errors.append(f"scheduled storage {cost['plm_bytes']} > {budget}")
    if storage['physical_local_array_bytes'] > budget:
        errors.append(f"local arrays {storage['physical_local_array_bytes']} > {budget}")
    return {
        'launches': len(launches),
        'captures_compared': compared,
        'scheduled_plm_bytes': cost['plm_bytes'],
        'physical_local_array_bytes': storage['physical_local_array_bytes'],
        'budget_bytes': budget,
        'tile_sizes': (generated / 'tile_sizes.txt').read_text().strip(),
        'hardware_sha256': hashlib.sha256((generated / '05_hw_acc.ll').read_bytes()).hexdigest(),
        'errors': errors,
    }


def main():
    run, release = Path(sys.argv[1]), Path(sys.argv[2])
    cases = {}
    for case in sorted(p.parent for p in run.glob('*/*/fixture.json')):
        name = f'{case.parent.name}/{case.name}'
        cases[name] = check_case(case, release / case.parent.name / case.name)
    hashes = sorted({c['hardware_sha256'] for c in cases.values()})
    errors = [f'{n}: {e}' for n, c in cases.items() for e in c['errors']]
    if len(hashes) != 1:
        errors.append(f'{len(hashes)} accelerators: {hashes}')
    result = {'status': 'PASS' if cases and not errors else 'FAIL',
              'hardware_sha256': hashes, 'cases': cases, 'errors': errors}
    text = json.dumps(result, indent=2) + '\n'
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text)
    print(text, end='')
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
