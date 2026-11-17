#!/usr/bin/env python3
"""Import a PyTorch factory and invoke the common ESP compiler pipeline.

The factory returns (nn.Module, tuple[Tensor, ...]). This driver exports input
fixtures/reference data and invokes tools; all architecture selection and IR
transformations belong to the compiler passes. No archived TOSA is consumed.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--factory', default='create_model')
    p.add_argument('--model-options', default='{}', help='JSON passed to the factory')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--compute-lanes', type=int, default=None,
                   help='FP32 compute lanes (kernel default 64; network compiler default 16 with conv, otherwise 64)')
    p.add_argument('--plm-bytes', type=int, default=327680)
    p.add_argument('--dma-width', type=int, choices=[64], default=64)
    p.add_argument('--network-host-only', action='store_true',
                   help='Validate compiler-owned CNN host lowering; no hardware qualification')
    p.add_argument('--mode', choices=['kernel','network'], default=None)
    p.add_argument('--schedule', choices=['overlap','nonoverlap'], default=None,
                   help='Whole-kernel LCS schedule (default overlap)')
    p.add_argument('--network', action='store_true', help='Alias for --mode network')
    p.add_argument('--conv-lanes', type=int, default=64)
    p.add_argument('--add-lanes', type=int, default=4)
    p.add_argument('--atol', type=float, default=0.001)
    p.add_argument('--rtol', type=float, default=0.0001)
    a = p.parse_args()
    if a.network and a.mode == 'kernel':
        p.error('--network conflicts with --mode kernel')
    a.mode = a.mode or 'network'  # Preserve the existing operator-driver API.
    if a.mode == 'network' and a.schedule is not None:
        p.error('--schedule is only valid in --mode kernel')
    if a.mode == 'kernel' and a.network_host_only:
        p.error('--network-host-only is only valid in --mode network')
    import torch
    from torch_mlir import torchscript
    repo = Path(__file__).resolve().parents[1]
    model_path = a.model.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    options = json.loads(a.model_options)
    spec = importlib.util.spec_from_file_location('esp_input_model', model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model, inputs = getattr(module, a.factory)(**options)
    model.eval()
    inputs = tuple(inputs)
    if any(x.dtype != torch.float32 or not x.is_contiguous() for x in inputs):
        raise ValueError('This ESP target requires contiguous FP32 inputs')
    with torch.no_grad():
        reference = model(*inputs)
    if not isinstance(reference, torch.Tensor) or reference.dtype != torch.float32:
        raise ValueError('This target supports one FP32 tensor result')
    reference = reference.detach().cpu().contiguous()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    names = list(inspect.signature(model.forward).parameters)
    arguments = []
    for i, x in enumerate(inputs):
        path = out/f'arg{i}.bin'
        path.write_bytes(x.detach().cpu().numpy().astype('<f4').tobytes())
        arguments.append(dict(name=names[i], shape=list(x.shape), elements=x.numel(),
                              file=path.name, sha256=sha(path)))
    (out/'reference.bin').write_bytes(reference.numpy().astype('<f4').tobytes())
    ir = torchscript.compile(model, inputs, output_type='tosa', use_tracing=True)
    (out/'01_tosa.mlir').write_text(str(ir))
    sources = {str(model_path): sha(model_path)}
    implementation = Path(inspect.getfile(type(model))).resolve()
    sources[str(implementation)] = sha(implementation)
    fixture = dict(kernel=options.get('kernel', type(model).__name__),
                   dataset=options.get('dataset', 'CUSTOM'), arguments=arguments,
                   outputs=reference.numel(), output_shape=list(reference.shape),
                   atol=a.atol, rtol=a.rtol, reference_sha256=sha(out/'reference.bin'),
                   reference_method='PyTorch execution of the input model',
                   model_options=options, model_sources=sources,
                   frontend=dict(torch_version=torch.__version__,
                                 torch_mlir_module=torchscript.__file__),
                   tosa_sha256=sha(out/'01_tosa.mlir'))
    (out/'fixture.json').write_text(json.dumps(fixture, indent=2)+'\n')
    env = {**os.environ, 'LCS_TILES':'operators', 'LCS_REUSABLE_GEMM':'false',
           'LCS_DMA_ABI':'packed', 'LCS_DMA_DATA_WIDTH':str(a.dma_width),
           'LCS_COMPUTE_LANES':str(a.compute_lanes if a.compute_lanes is not None else 0), 'LCS_PLM_BYTES':str(a.plm_bytes),
           'LCS_GEMM_FP_SEMANTICS':'source', 'LCS_PROFILE':'true'}
    env.update(LCS_CONV_LANES=str(a.conv_lanes),LCS_ADD_LANES=str(a.add_lanes),
               LCS_GEMM_LANES=str(a.compute_lanes if a.compute_lanes is not None else 0),LCS_DUMP_IR='false')
    # Reproducible defaults: unrelated experiments must not inject schedules.
    for key in ('LCS_TILE_SIZES','LCS_REGISTER_BLOCK','LCS_TRANSFORM','LCS_ENTRY',
                'LCS_SEQUENTIAL','LCS_OVERLAP','LCS_FP_UNITS','LCS_BANK_LOCAL'):
        env.pop(key, None)
    if a.mode == 'kernel':
        env['LCS_TILES'] = 'kernel'
        env['LCS_OVERLAP'] = 'false' if a.schedule == 'nonoverlap' else 'true'
        if a.compute_lanes is None:
            env.pop('LCS_COMPUTE_LANES', None)  # Compiler pipeline default.
    (out/'lowering-env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith('LCS_')},indent=2)+'\n')
    driver = 'scripts/lcs/network_host.sh' if a.network_host_only else 'scripts/tosa_to_lcs.sh'
    with (out/'compile.log').open('w') as log:
        subprocess.run(['bash',str(repo/driver),str(out/'01_tosa.mlir'),str(out/'generated')],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if a.mode == 'kernel':
        result = dict(status='KERNEL_LOWERED_NOT_HARDWARE_VALIDATED',
                      schedule=a.schedule or 'overlap',
                      accelerator_sha256=sha(out/'generated/05_hw_acc.ll'),
                      architecture_sha256=sha(out/'generated/architecture.xml'),
                      host_sha256=sha(out/'generated/05_kernel_riscv.ll'),
                      fixture_sha256=sha(out/'fixture.json'))
    elif a.network_host_only:
        result = dict(status='NETWORK_HOST_LOWERED_NOT_HARDWARE_VALIDATED',
                      host_sha256=sha(out/'generated/host.ll'),
                      parameter_sha256=sha(out/'generated/parameters.bin'),
                      fixture_sha256=sha(out/'fixture.json'),
                      invocation_manifest_sha256=sha(out/'generated/network.json'))
    else:
        result = dict(status='OPERATOR_LOWERED_NOT_HARDWARE_VALIDATED',
                      compilation_sha256=sha(out/'generated/compilation.json'),
                      fixture_sha256=sha(out/'fixture.json'))
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
