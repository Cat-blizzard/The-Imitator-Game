"""Bounded initial-frame rendering for a fixed physical scene and three appearances."""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from .appearance_v1 import DEFAULT_CATALOG, sample

VERSION = 'appearance_initial_render_v1'


def prepare(scene_path, output, parent_render_root=None):
    output = Path(output)
    if output.exists():
        raise ValueError('appearance_render_output_must_be_new')
    base = read(scene_path)
    specs = [
        {'table_ids': ['polyhaven:angli_veneer'], 'wall_ids': ['polyhaven:brick_wall_04'],
         'background_mode': 'wall', 'temperature_k': [4500]},
        {'table_ids': ['polyhaven:brown_leather'], 'environment_ids': ['polyhaven:poly_haven_studio'],
         'background_mode': 'environment', 'temperature_k': [6500]},
        {'table_ids': ['polyhaven:blue_metal_plate'], 'environment_ids': ['polyhaven:small_empty_room_4'],
         'background_mode': 'environment', 'temperature_k': [8000]},
    ]
    rows = []
    for i, spec in enumerate(specs):
        scene = sample(base, 2026102900 + i, config=spec)
        target = output / f'variant_{i:02d}' / 'scene.json'; write(target, scene)
        rows.append({'id': f'variant_{i:02d}', 'scene': binding(target)})
    prior = len(list(Path(parent_render_root).glob('*/reservation.json'))) if parent_render_root else 0
    write(output / 'protocol.json', {'schema': VERSION, 'source_scene': binding(scene_path),
        'catalog': binding(DEFAULT_CATALOG), 'cases': rows, 'maximum_attempts': 3,
        'timeout_seconds': 120, 'gpu': 2, 'minimum_free_gpu_mib': 12288,
        'physical_motion_executions': 0, 'api_calls': 0, 'new_trajectories': 0,
        'initializations_only': True, 'background_variants': 3,
        'prior_render_initializations': prior, 'lineage_render_initialization_limit': prior + 3,
        'parent_render_protocol': binding(Path(parent_render_root) / 'protocol.json') if parent_render_root else None,
        'code': [binding(__file__), binding(Path(__file__).with_name('appearance_v1.py')),
            binding(ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/simulation.py'),
            binding(ROOT / 'resimpair_pickplace/environment.py')]})
    return rows


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    from resimpair_pickplace import environment
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator
    root = Path(root).resolve(); protocol = read(root / 'protocol.json')
    for ref in protocol['code']: resolve(ref)
    case = next(c for c in protocol['cases'] if c['id'] == case_id)
    folder = root / case_id
    if os.environ.get('RESIMPAIR_APPEARANCE_RESERVATION') != str(folder / 'reservation.json'):
        raise ValueError('bounded_render_supervisor_required')
    scene = read(resolve(case['scene'])); torch.set_num_threads(2)
    environment.register(scene, Accumulator(), contact_policy='native_default')
    env = None
    try:
        env = gym.make(environment.ENV_ID, sim_backend='physx_cpu', render_backend='cuda:0',
            control_mode='pd_joint_pos', obs_mode='rgb', reward_mode='none', render_mode=None)
        obs, _ = env.reset(seed=int(scene['seed'] % (2**32)), options={'reconfigure': True})
        if env.unwrapped.gpu_sim_enabled:
            raise ValueError('CPU_physics_required')
        frames = {}
        for name, config in scene['cameras'].items():
            rgb = array(obs['sensor_data'][name]['rgb']).reshape(config['height'], config['width'], 3)
            Image.fromarray(rgb).save(folder / (name + '.png'))
            frames[name] = {'file': binding(folder / (name + '.png')),
                'std_rgb': float(rgb.std()), 'mean_rgb': rgb.mean(axis=(0, 1)).tolist()}
        state = array(env.unwrapped.get_state()).copy()
        np.save(folder / 'initial_physics_state.npy', state)
        write(folder / 'result.json', {'schema': VERSION, 'case': case, 'frames': frames,
            'physical_scene_digest': scene['appearance']['physical_scene_digest'],
            'initial_physics_state': binding(folder / 'initial_physics_state.npy'),
            'state_shape': list(state.shape), 'sim_backend': 'physx_cpu',
            'render_backend': 'Vulkan GPU rendering with CUDA readback; no CUDA physics or optimizer',
            'gpu_physical_index': protocol['gpu'], 'physical_motion_executions': 0,
            'appearance_runtime_evidence': env.unwrapped.appearance_runtime_evidence,
            'control_steps_executed': 0, 'api_calls': 0, 'new_trajectories': 0})
    finally:
        if env is not None: env.close()


def run(root):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.resources import lock_paths
    from .cudex_shared_gpu_v1 import shared_snapshot
    root = Path(root).resolve(); protocol = read(root / 'protocol.json')
    if len(protocol['cases']) > protocol['maximum_attempts']:
        raise ValueError('finite_render_limit_exceeded')
    for ref in protocol['code']: resolve(ref)
    for case in protocol['cases']:
        folder = root / case['id']
        if (folder / 'terminal.json').exists():
            continue
        if (folder / 'reservation.json').exists():
            raise RuntimeError('existing_render_submission_must_be_recovered_not_repeated')
        with ExitStack() as stack:
            inherited_locks = []
            for path in lock_paths(protocol['gpu']):
                lock = stack.enter_context(path.open('r+'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                inherited_locks.append(lock.fileno())
            snap = shared_snapshot(protocol['gpu'])
            if snap['free_memory_MiB'] < protocol['minimum_free_gpu_mib']:
                raise RuntimeError('insufficient_shared_gpu_headroom')
            write(folder / 'reservation.json', {'case': case, 'protocol': binding(root / 'protocol.json'),
                'gpu': snap, 'started_unix': time.time(), 'api_calls': 0,
                'physical_motion_executions': 0, 'existing_external_processes_preserved': True})
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=snap['uuid'], OMP_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', RESIMPAIR_APPEARANCE_RESERVATION=str(folder / 'reservation.json'))
            started = time.monotonic()
            with (folder / 'worker.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'resimpair_tasks.appearance_render_v1',
                    'worker', '--root', str(root), '--case', case['id']], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(inherited_locks))
                write(folder / 'process.json', {'pid': proc.pid, 'started_unix': time.time()})
                try:
                    code = proc.wait(timeout=protocol['timeout_seconds']); timeout = False
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try: code = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill(); code = proc.wait(timeout=10)
                    timeout = True
            write(folder / 'terminal.json', {'returncode': code, 'timeout': timeout,
                'elapsed_seconds': time.monotonic() - started, 'api_calls': 0})
            if code:
                raise RuntimeError('render_failed_preserved_no_automatic_retry:' + case['id'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--scene', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--parent-render-root', type=Path)
    for name in ('run', 'worker'):
        a = sub.add_parser(name); a.add_argument('--root', type=Path, required=True)
        if name == 'worker': a.add_argument('--case', required=True)
    args = p.parse_args()
    if args.command == 'prepare': print(prepare(args.scene, args.output, args.parent_render_root))
    elif args.command == 'run': run(args.root)
    else: worker(args.root, args.case)
