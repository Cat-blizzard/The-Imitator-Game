"""Run the unchanged simulator evaluator with model and renderer on one GPU.

The physical GPU is selected before importing Torch/SAPIEN. Explicit PCI renderer
selection avoids SAPIEN choosing a different GPU from the policy. Shader settings,
action execution, seeds, human references and HyperNet holding are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def gpu_identity(physical_gpu: int) -> dict:
    if physical_gpu < 0:
        raise ValueError('The physical GPU index must be nonnegative')
    line = subprocess.check_output([
        'nvidia-smi', '-i', str(physical_gpu),
        '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader'], text=True).strip()
    index, uuid, bus = [x.strip() for x in line.split(',')]
    domain, bus_id, function = bus.lower().split(':')
    # NVIDIA reports eight domain digits; SAPIEN accepts the standard four.
    pci = f'pci:{int(domain, 16):04x}:{bus_id}:{function}'
    # Imitator 2.0: RESIMPAIR_RENDER_BACKEND overrides the renderer device, e.g. "pci:0" for a CPU Vulkan renderer
    # (lavapipe) on machines whose GPU has no graphics capability; physics and models stay on this GPU.
    render = os.environ.get('RESIMPAIR_RENDER_BACKEND') or pci
    return dict(physical_gpu=int(index), uuid=uuid, pci_bus_id=bus,
                render_backend=render, model_device='cuda:0')


def bind_renderer(factory, render_backend):
    """Change only the backend argument of the environment factory."""
    def make(base_env_name, level, num_envs, sim_backend, env_kwargs,
             other_kwargs, video_dir, wrappers):
        kwargs = dict(env_kwargs)
        if kwargs.get('render_backend', render_backend) != render_backend:
            raise ValueError('A conflicting renderer device was requested')
        kwargs['render_backend'] = render_backend
        return factory(base_env_name, level, num_envs, sim_backend, kwargs,
                       other_kwargs, video_dir, wrappers)
    return make


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    p.add_argument('--render-physical-gpu', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    own, rest = p.parse_known_args()
    if own.device != 'cuda:0':
        raise ValueError('Isolated evaluation uses local --device cuda:0')
    identity = gpu_identity(own.render_physical_gpu)
    if 'torch' in sys.modules or 'sapien' in sys.modules:
        raise RuntimeError('GPU isolation must happen before importing Torch/SAPIEN')
    os.environ['CUDA_VISIBLE_DEVICES'] = identity['uuid']
    # This evaluator and its DINO tokenizer use PyTorch only. Avoid loading the
    # unused multi-GB TensorFlow runtime during Hugging Face imports.
    os.environ['USE_TF'] = '0'
    # NVIDIA's Vulkan enumeration is independent of CUDA_VISIBLE_DEVICES. The
    # explicit PCI backend below is the authoritative renderer selection.
    own.output.mkdir(parents=True, exist_ok=True)
    record = dict(**identity, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                  use_tf=os.environ['USE_TF'],
                  wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  evaluation_source_sha256=hashlib.sha256(Path(__file__).with_name('evaluate.py').read_bytes()).hexdigest(),
                  intervention='Model and renderer co-located; shader and control protocol unchanged',
                  status='starting')
    record_path = own.output / 'renderer_device.json'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    from . import evaluate
    from examples.baselines.diffusion_policy import eval_dp_imitator
    original = eval_dp_imitator.make_eval_envs_with_level
    eval_dp_imitator.make_eval_envs_with_level = bind_renderer(
        original, identity['render_backend'])
    sys.argv = [sys.argv[0], '--output', str(own.output), '--device', own.device, *rest]
    try:
        evaluate.main()
    except BaseException as exc:
        record.update(status='failed', error=repr(exc))
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        raise
    else:
        record.update(status='complete')
        record_path.write_text(json.dumps(record, indent=2) + '\n')
    finally:
        eval_dp_imitator.make_eval_envs_with_level = original


if __name__ == '__main__':
    main()
