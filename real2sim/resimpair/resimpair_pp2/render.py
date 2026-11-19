"""Saved-state RGB replay of pp2 executions inside a sampled room.

Visual-only: the retained scene-domain sampler (room, lighting, furniture,
materials) changes no physics; every saved physics state is restored and
rendered once per camera, never stepped. One room seed per scene so all
tasks of a scene share the same look, which the human first frame needs.
v2 ray-traces by default (`lighting.SHADER`, interior emitters, fixed per-camera
exposure); `--shader minimal` reproduces the v1 rasterized look.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'pp2_saved_state_render_v2'
from .lighting import SHADER
CURATED_CATALOG = ROOT / 'assets/appearance_v2/curated/catalog.json'  # the material policy index is bound to it


VARIANT_SEED_STRIDE = 104729  # room seed of render variant k: room_seed(scene) + k * stride


def room_seed(scene_id):
    return int(digest(['pp2-room', scene_id])[:7], 16)


def room_scene(execution, seed, tries=8, camera_mode='fixed'):
    """Deterministic: seed, seed+1, ... until a room preset accepts the furniture overlay."""
    from resimpair_tasks import scene_domain_v2 as base
    from resimpair_tasks import scene_quality_v3 as quality
    scene = read(execution / 'effective_scene.json')
    envelope = base.build_motion_envelope(scene, execution / 'measurements.npz')
    errors = []
    for k in range(tries):
        try:
            domain = base.sample_domain(scene, seed + k, catalog_path=CURATED_CATALOG, mode='workspace',
                                        camera_mode=camera_mode, motion_envelope=envelope, asset_partition='development')
            return quality.apply(domain, 'combined', seed + k)
        except (ValueError, StopIteration) as error:
            errors.append(repr(error)[:120])
    raise ValueError('no_room_preset_accepted:' + ';'.join(errors))


def make_env(scene, obs_mode, shader, first_state):
    """Registered scene-quality env, reset once; ray-traced shaders also get emitters and exposure."""
    import gymnasium as gym
    from resimpair_tasks import scene_quality_v3 as quality
    from . import lighting
    env = gym.make(quality.ENV_ID, sim_backend='physx_cpu', render_backend='cuda:0', control_mode='pd_joint_pos',
                   obs_mode=obs_mode, reward_mode='none', render_mode=None, sensor_configs={'shader_pack': shader})
    env.reset(seed=int(scene['seed'] % (2**32)), options={'reconfigure': True}); b = env.unwrapped
    record = lighting.setup(b, scene, first_state) if shader.startswith('rt') else {'shader': shader}
    return env, b, record


def hide_robots(b):
    """Disable the render bodies of every robot link (visual only: physics and saved states are untouched)."""
    import sapien
    agents = getattr(b.agent, 'agents', None) or [b.agent]; hidden = 0
    for agent in agents:
        for link in agent.robot.links:
            for obj in link._objs:
                body = getattr(obj, 'entity', obj).find_component_by_type(sapien.render.RenderBodyComponent)
                if body is not None and body.is_enabled:
                    body.disable(); hidden += 1
    return hidden


def worker(execution, output, seed, stride, shader=SHADER):
    from resimpair_runtime import configure
    configure()
    import torch
    from PIL import Image
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator
    from resimpair_tasks import scene_quality_v3 as quality
    from resimpair_tasks.deferred_render_v1 import NativeClockVideo
    execution, output = Path(execution), Path(output); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    scene = room_scene(execution, seed); write(output / 'render_scene.json', scene)
    saved = np.load(execution / 'deferred_render_states.npz')
    states, times = saved['physics_states'], saved['timestamps_seconds']
    from fractions import Fraction
    fps = Fraction(1. / float(np.median(np.diff(times))) / stride).limit_denominator(1000)
    contact = read(execution / 'native_contact_policy.json')['policy']
    quality.register(scene, Accumulator(), contact_policy=contact)
    torch.set_num_threads(2)
    env, b, record = make_env(scene, 'rgb+segmentation', shader, states[0])
    writers = {}
    try:
        from resimpair_tasks.scene_quality_render_v3 import entity_mapping
        mapping = entity_mapping(b)['objects']; masks = {}
        if tuple(array(b.get_state()).shape) != tuple(states[0].shape):
            raise ValueError('render_environment_state_layout_differs')
        indices = list(range(0, len(states), stride))
        if indices[-1] != len(states) - 1:
            indices.append(len(states) - 1)
        for name, cfg in scene['cameras'].items():
            writers[name] = NativeClockVideo(output / f'{name}.mp4', cfg['width'], cfg['height'], fps)
        worst = 0.
        for k, index in enumerate(indices):
            b.set_state(torch.as_tensor(states[index], device='cpu'))
            worst = max(worst, float(np.abs(array(b.get_state()) - states[index]).max()))
            obs = b.get_obs()
            for name, cfg in scene['cameras'].items():
                rgb = array(obs['sensor_data'][name]['rgb']).reshape(cfg['height'], cfg['width'], 3)
                writers[name].append(rgb, k)
                if k == 0 or index == indices[-1]:
                    which = 'initial' if k == 0 else 'final'
                    Image.fromarray(rgb).save(output / f'{name}_{which}.png')
                    ids = array(obs['sensor_data'][name]['segmentation']).reshape(cfg['height'], cfg['width'])
                    masks.setdefault(name, {})[which] = np.stack([np.isin(ids, mapping[o]) for o in b.object_ids])
        for name, pair in masks.items():
            np.savez_compressed(output / f'{name}_object_masks.npz', object_ids=np.asarray(list(b.object_ids)), **pair)
    finally:
        for w in writers.values():
            w.close()
        env.close()
    result = {'schema': VERSION, 'execution': str(execution), 'room_seed': scene['scene_domain']['seed'],
              'room': scene['scene_domain']['preset_id'], 'frames': len(indices), 'stride': stride, 'fps': float(fps),
              'max_state_roundtrip_error': worst, 'render': record, 'videos': {n: binding(output / f'{n}.mp4') for n in writers},
              'physical_execution_count': 0, 'env_step_calls': 0, 'seconds': time.monotonic() - started}
    write(output / 'render_result.json', result)
    return result


def keyframes(execution, output, seed, scale=2, camera='opposite_high', camera_mode='fixed', shader=SHADER,
              hide_robot=False):
    """Initial and final states of one execution at `scale` x resolution (same room as its video).

    hide_robot renders the scene without the robots (lighting and exposure are still set with them present),
    which is what a human reference video starts and ends on."""
    from resimpair_runtime import configure
    configure()
    import torch
    from PIL import Image
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator
    from resimpair_tasks import scene_quality_v3 as quality
    execution, output = Path(execution), Path(output); output.mkdir(parents=True, exist_ok=True)
    scene = room_scene(execution, seed, camera_mode=camera_mode)
    if camera == 'robot_side' and scale != 1:
        raise ValueError('scene_domain_binds_robot_side_to_its_source_config')
    cameras = json.loads(json.dumps(scene['cameras']))
    cameras[camera] = {**cameras[camera], 'width': cameras[camera]['width'] * scale,
                       'height': cameras[camera]['height'] * scale}
    scene['cameras'] = cameras  # visual-only: the domain validator binds actual configs to the scene's cameras
    scene['scene_domain']['camera_sampling']['actual_configs'] = json.loads(json.dumps(cameras))
    write(output / 'keyframe_scene.json', scene)
    states = np.load(execution / 'deferred_render_states.npz')['physics_states']
    quality.register(scene, Accumulator(), contact_policy=read(execution / 'native_contact_policy.json')['policy'])
    env, b, record = make_env(scene, 'rgb+segmentation', shader, states[0])
    if hide_robot:
        record = {**record, 'hidden_robot_bodies': hide_robots(b)}
    try:
        from resimpair_tasks.scene_quality_render_v3 import entity_mapping
        cfg = scene['cameras'][camera]; files = {}; masks = {}
        mapping = entity_mapping(b)['objects']
        for name, index in (('initial', 0), ('final', len(states) - 1)):
            b.set_state(torch.as_tensor(states[index], device='cpu'))
            obs = b.get_obs()['sensor_data'][camera]
            rgb = array(obs['rgb']).reshape(cfg['height'], cfg['width'], 3)
            ids = array(obs['segmentation']).reshape(cfg['height'], cfg['width'])
            Image.fromarray(rgb).save(output / f'{camera}_{name}.png'); files[name] = binding(output / f'{camera}_{name}.png')
            masks[name] = np.stack([np.isin(ids, mapping[o]) for o in b.object_ids])
        np.savez_compressed(output / f'{camera}_object_masks.npz', object_ids=np.asarray(list(b.object_ids)), **masks)
        files['object_masks'] = binding(output / f'{camera}_object_masks.npz')
    finally:
        env.close()
    result = {'schema': VERSION + '_keyframes', 'execution': str(execution), 'camera': camera, 'scale': scale,
              'camera_mode': camera_mode, 'camera_config': scene['cameras'][camera],
              'lighting_temperature_k': scene['scene_domain']['lighting'].get('temperature_k'),
              'room': scene['scene_domain']['preset_id'], 'room_seed': scene['scene_domain']['seed'], 'render': record,
              'images': files}
    write(output / 'keyframes.json', result)
    return result


def gpu_free_mib(gpu):
    return int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True).strip())


def run(cohort, output, cases, gpus, stride=1, per_gpu=1, shader=SHADER, variants=None, min_free_mib=12 * 1024,
        timeout=900, settle_s=30, max_wait_s=6 * 3600):
    """Render selected executions; one subprocess per case (and room variant), bounded per GPU.

    variants=None keeps the v1 layout output/<case>; variants=n renders output/<case>/v0..v<n-1>, variant k in room
    seed room_seed(scene) + k * VARIANT_SEED_STRIDE, so all tasks of a scene share each variant's room.
    A render starts only while its GPU has min_free_mib free (the GPUs are shared). A job waits for that instead of
    being skipped (prod1 skipped 14k renders in seconds while another job held 15 GB per GPU); launches on one GPU
    are settle_s apart so each memory reading includes the renders started before it."""
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    slots = queue.Queue()  # a job holds one (gpu) slot while it runs, so no GPU exceeds per_gpu renders
    for g in gpus * per_gpu:
        slots.put(g)
    locks = {g: threading.Lock() for g in gpus}; launched = {g: 0. for g in gpus}
    jobs = [(case, k) for case in cases for k in (range(variants) if variants else [None])]

    def launch(item):
        case, k = item
        target = output / case if k is None else output / case / f'v{k}'
        if (target / 'render_result.json').exists():
            return read(target / 'render_result.json')
        started = time.monotonic()
        while True:
            gpu = slots.get()
            with locks[gpu]:
                time.sleep(max(0., launched[gpu] + settle_s - time.monotonic()))
                if gpu_free_mib(gpu) >= min_free_mib:
                    launched[gpu] = time.monotonic(); break
            slots.put(gpu)
            if time.monotonic() - started > max_wait_s:
                return {'case': case, 'variant': k, 'skipped': f'gpu_memory_below_{min_free_mib}MiB_for_{max_wait_s}s'}
            time.sleep(settle_s)
        try:
            return render_one(case, k, target, gpu)
        finally:
            slots.put(gpu)

    def render_one(case, k, target, gpu):
        execution = cohort / 'executions' / case
        seed = room_seed(case.rsplit('__', 1)[0]) + (k or 0) * VARIANT_SEED_STRIDE
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2')
        if target.exists():  # an unfinished earlier attempt (no render_result.json): visual-only, start over
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        with (target / 'render.log').open('a') as log:
            try:  # a ray-traced render rarely hangs on a shared GPU; it is visual-only, so it is killed and retried
                rc = subprocess.call([sys.executable, '-u', '-m', 'resimpair_pp2.render', 'worker', '--execution',
                                      str(execution), '--output', str(target), '--seed', str(seed), '--stride', str(stride),
                                      '--shader', shader],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                rc = 'timeout'
        return read(target / 'render_result.json') if rc == 0 else {'case': case, 'variant': k, 'failed': rc}
    with ThreadPoolExecutor(len(gpus) * per_gpu) as pool:
        rows = list(pool.map(launch, jobs))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    w = sub.add_parser('worker'); w.add_argument('--execution', type=Path, required=True)
    w.add_argument('--output', type=Path, required=True); w.add_argument('--seed', type=int, required=True)
    w.add_argument('--stride', type=int, default=1)
    w.add_argument('--shader', default=SHADER)
    k = sub.add_parser('keyframes'); k.add_argument('--execution', type=Path, required=True)
    k.add_argument('--output', type=Path, required=True); k.add_argument('--seed', type=int, required=True)
    k.add_argument('--scale', type=int, default=2); k.add_argument('--camera', default='opposite_high')
    k.add_argument('--camera-mode', default='fixed', choices=['fixed', 'bounded'])
    k.add_argument('--shader', default=SHADER); k.add_argument('--hide-robot', action='store_true')
    r = sub.add_parser('run'); r.add_argument('--cohort', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True); r.add_argument('--cases', nargs='+', required=True)
    r.add_argument('--gpus', type=int, nargs='+', required=True); r.add_argument('--stride', type=int, default=1)
    r.add_argument('--per-gpu', type=int, default=1); r.add_argument('--shader', default=SHADER)
    r.add_argument('--variants', type=int)
    a = p.parse_args()
    if a.command == 'worker':
        print(json.dumps(worker(a.execution, a.output, a.seed, a.stride, a.shader)))
    elif a.command == 'keyframes':
        print(json.dumps(keyframes(a.execution, a.output, a.seed, a.scale, a.camera, a.camera_mode, a.shader, a.hide_robot)))
    else:
        print(json.dumps(run(a.cohort, a.output, a.cases, a.gpus, a.stride, a.per_gpu, a.shader, a.variants), indent=1)[:3000])
