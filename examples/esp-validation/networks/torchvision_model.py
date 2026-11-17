"""Load a checkpoint/input; all graph normalization belongs to MLIR passes."""
import json
from pathlib import Path
import numpy as np
import torch
import torchvision


def create_model(name, frozen_models, input_file=None):
    root = Path(frozen_models)
    checkpoints = {
        'resnet18': 'resnet18-f37072fd.pth',
        'resnet34': 'resnet34-b627a593.pth',
        'squeezenet1_1': 'squeezenet1_1-b8a52dc0.pth',
    }
    model = getattr(torchvision.models, name)(weights=None).eval()
    model.load_state_dict(torch.load(root/'checkpoints'/checkpoints[name],
                                    map_location='cpu', weights_only=True))
    manifest = json.loads((root/name/'manifest.json').read_text())
    node = next(n for n in manifest['nodes'] if n['kind'] == 'placeholder')
    data = np.fromfile(Path(input_file) if input_file else root/name/node['reference'], dtype='<f4').reshape(node['shape'])
    return model, (torch.from_numpy(data),)
