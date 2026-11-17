#!/usr/bin/env python3
"""Execute compiler-produced host LLVM against the independent PyTorch fixture.

This is a test harness, not a schedule generator. The native runtime emulates
operator execution, so a pass is host/numerical evidence, never RTL evidence.
"""
import argparse
import ctypes
import json
from pathlib import Path
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--repetitions', type=int, default=2)
    p.add_argument('--rtl-library', type=Path, help='Use the generated RTL executor instead of the native oracle')
    p.add_argument('--program', type=Path, help='Reuse another compilation with this input/reference fixture')
    a = p.parse_args()
    root = a.directory.resolve()
    fixture = json.loads((root/'fixture.json').read_text())
    generated = (a.program.resolve() if a.program else root)/'generated'
    if a.program:
        assert (root/'generated/parameters.bin').read_bytes()==(generated/'parameters.bin').read_bytes(), 'Checkpoint/normalized parameters changed'
    manifest = json.loads((generated/'network.json').read_text())
    inputs = [np.fromfile(root/x['file'], dtype='<f4') for x in fixture['arguments']]
    parameters = np.fromfile(generated/'parameters.bin', dtype='<f4')
    reference = np.fromfile(root/'reference.bin', dtype='<f4')
    output = np.full(reference.size, np.nan, dtype=np.float32)
    assert manifest['input_elements'] == [x.size for x in inputs]
    assert manifest['output_elements'] == reference.size
    assert manifest['parameter_elements'] == parameters.size
    pointer = ctypes.POINTER(ctypes.c_float)
    ptr = lambda x: x.ctypes.data_as(pointer)
    lib = ctypes.CDLL(str(a.rtl_library.resolve() if a.rtl_library else generated/'native.so'))
    lib.forward_init.argtypes = [pointer]
    lib.forward_init.restype = ctypes.c_int64
    lib.forward_run.argtypes = [ctypes.c_int64, *([pointer]*len(inputs)), pointer]
    lib.forward_run.restype = None
    lib.forward_close.argtypes = [ctypes.c_int64]
    lib.forward_close.restype = None
    start = time.monotonic()
    context = lib.forward_init(ptr(parameters))
    errors = []
    try:
        for _ in range(a.repetitions):
            output.fill(np.nan)
            lib.forward_run(context, *map(ptr, inputs), ptr(output))
            np.testing.assert_allclose(output, reference, rtol=fixture['rtol'],
                                       atol=fixture['atol'], equal_nan=False)
            errors.append(float(np.max(np.abs(output-reference))))
    finally:
        lib.forward_close(context)
    result = dict(status='PASS', scope=('compiler host + generated operator RTL + native-DMA adapter; simulated memory, no ESP CPU/socket'
                                       if a.rtl_library else 'compiler host + native operator oracle; no RTL'),
                  repetitions=a.repetitions, max_abs_errors=errors,
                  test_seconds=time.monotonic()-start,
                  accelerator_calls=sum(x['execution']=='accelerator' for x in manifest['invocations']),
                  cpu_calls=sum(x['execution']=='cpu' for x in manifest['invocations']))
    name='rtl-host-validation.json' if a.rtl_library else 'host-validation.json'
    (root/name).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
