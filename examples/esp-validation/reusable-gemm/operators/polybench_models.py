"""Input-model factory only: no tiling, banking, scheduling or IR rewriting."""
import importlib
import torch
from benches.PolyBenchPyTorch.linear_algebra.utils import get_dataset_dimensions


def create_model(kernel='gemm', dataset='MINI', dimensions=None, nan_c=False):
    modules = {'gemm': ('blas.gemm.gemm','Gemm'),
               'twomm': ('kernels.twomm.twomm','TwoMM'),
               'threemm': ('kernels.threemm.threemm','ThreeMM'),
               'syrk': ('blas.syrk.syrk','Syrk'),
               'syr2k': ('blas.syr2k.syr2k','Syr2k'),
               'symm': ('blas.symm.symm','Symm'),
               'trmm': ('blas.trmm.trmm','Trmm'),
               'atax': ('kernels.atax.atax','Atax'),
               'gesummv': ('blas.gesummv.gesummv','Gesummv'),
               'doitgen': ('kernels.doitgen.doitgen','Doitgen')}
    module, name = modules[kernel]
    source = importlib.import_module('benches.PolyBenchPyTorch.linear_algebra.'+module)
    dims = get_dataset_dimensions(kernel, dataset)
    if dimensions: dims.update(dimensions)
    model = getattr(source,name)(**dims)
    inputs = source.init_array(**dims,dtype=torch.float32)
    if nan_c:
        if kernel != 'gemm': raise ValueError('NaN-C semantic test requires GEMM')
        inputs[1].zero_()  # Explicit beta*C remains in the source expression.
        inputs[2].fill_(float('nan'))
    return model, inputs
