"""Render both cameras from each saved native state, without executing actions.

The output is a derivative of one identified physical execution. Measurements,
actions and contact evidence are bound to that execution; they are not measured
again during rendering. Full state/pose roundtrips and native video PTS are
checked. This does not change the existing R2H schema or its acceptance rules.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import fcntl
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'saved_native_state_multiview_render_v1'
REQUIRED_EVIDENCE = ('execution_bundle.json', 'effective_scene.json', 'effective_task.json',
    'measurements.npz', 'deferred_render_states.npz', 'deferred_render_manifest.json',
    'event_protocol.json', 'planning_declaration.json', 'planned_paths.json',
    'native_geometry.json', 'native_containers.json', 'native_contact_policy.json',
    'audit.json', 'container_audit.json', 'requested_goal_audit.json', 'terminal.json')
COPY_EVIDENCE = ('event_protocol.json', 'planning_declaration.json', 'planned_paths.json',
                 'native_geometry.json', 'native_containers.json', 'native_contact_policy.json')


def validate_timeline(states, timestamps, arrays):
    """No interpolation, resampling, missing frames, or implicit clock shift."""
    stamps = np.asarray(timestamps)
    if (states.ndim != 3 or states.shape[1] != 1 or len(states) < 2
            or stamps.shape != (len(states),) or not np.isfinite(states).all()
            or not np.isfinite(stamps).all() or stamps[0] != 0
            or not np.array_equal(stamps, arrays['timestamps_seconds'])
            or len(arrays['native_commands']) != len(states) - 1):
        raise ValueError('saved_states_measurements_or_action_timeline_mismatch')
    n = len(states)
    expected = {'states': (n, 18), 'tcp_world': (n, 2, 4, 4), 'native_commands': (n-1, 16)}
    if (any(np.asarray(arrays[k]).shape != shape or not np.isfinite(arrays[k]).all()
            for k, shape in expected.items()) or arrays['object_states'].shape[0] != n):
        raise ValueError('source_measurement_shapes_do_not_match_saved_states')
    differences = np.diff(stamps)
    if np.any(differences <= 0) or not np.allclose(differences, differences[0], rtol=0, atol=1e-10):
        raise ValueError('uniform_native_control_clock_required_no_resampling')
    rate = Fraction(float(1 / differences[0])).limit_denominator(100000)
    if not np.allclose(np.arange(len(stamps)) / float(rate), stamps, rtol=0, atol=1e-10):
        raise ValueError('native_clock_not_exactly_representable')
    return rate


def validate_source(source):
    source = Path(source).resolve()
    bindings = {name: binding(source / name) for name in REQUIRED_EVIDENCE}
    bundle = read(source / 'execution_bundle.json')
    scene = read(source / 'effective_scene.json'); task = read(source / 'effective_task.json')
    manifest = read(source / 'deferred_render_manifest.json')
    if (read(source / 'terminal.json')['returncode'] != 0 or bundle['planning_failure'] is not None
            or bundle['native_execution'] is not True or bundle['physical_execution_count'] != 1
            or bundle.get('saved_state_rerender') is not False):
        raise ValueError('completed_original_native_execution_required')
    if ({k: v for k, v in bundle['scene'].items() if k != 'cameras'} !=
            {k: v for k, v in scene.items() if k != 'cameras'} or task != bundle['task_intent']):
        raise ValueError('effective_scene_or_task_differs_from_executed_bundle')
    if (set(scene['cameras']) != {'robot_side', 'opposite_high'}
            or manifest['planned_cameras'] != scene['cameras']
            or resolve(manifest['physics_states']) != source / 'deferred_render_states.npz'
            or resolve(manifest['original_scene']) != source / 'effective_scene.json'):
        raise ValueError('saved_state_scene_or_camera_binding_mismatch')
    physics = read(source / 'audit.json'); containers = read(source / 'container_audit.json')
    requested = read(source / 'requested_goal_audit.json')
    if (physics['errors'] or any(not c['passed'] for k, c in physics['checks'].items() if k != 'synchronization')
            or containers['errors'] or any(not c['passed'] for c in containers['checks'].values())
            or not requested['goals'] or any(not c['passed'] for c in requested['goals'])):
        raise ValueError('source_physical_subchecks_did_not_pass')
    with np.load(source / 'deferred_render_states.npz', allow_pickle=False) as z:
        states = z['physics_states']; stamps = z['timestamps_seconds']
    with np.load(source / 'measurements.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if resolve(bundle['measurements']) != source / 'measurements.npz':
        raise ValueError('original_measurement_file_binding_mismatch')
    rate = validate_timeline(states, stamps, arrays)
    return {'source': source, 'bindings': bindings, 'bundle': bundle, 'scene': scene,
            'task': task, 'states': states, 'timestamps': stamps, 'arrays': arrays, 'fps': rate}


class NativeClockVideo:
    def __init__(self, path, width, height, fps):
        import av
        if Path(path).exists():
            raise ValueError('cannot_overwrite_rendered_video')
        self.container = av.open(str(path), 'w', options={'movflags': '+faststart'})
        self.stream = self.container.add_stream('libx264', rate=fps)
        self.stream.width = width; self.stream.height = height
        self.stream.pix_fmt = 'yuv420p'; self.stream.codec_context.thread_count = 2
        self.stream.options = {'crf': '18', 'preset': 'veryfast', 'bf': '0'}
        self.clock = 1 / fps; self.stream.time_base = self.clock
        self.stream.codec_context.time_base = self.clock
        self.count = 0

    def append(self, rgb, index):
        import av
        if index != self.count:
            raise ValueError('native_frames_must_be_encoded_in_order_without_gaps')
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        frame.pts = index; frame.time_base = self.clock
        for packet in self.stream.encode(frame): self.container.mux(packet)
        self.count += 1

    def close(self):
        for packet in self.stream.encode(): self.container.mux(packet)
        self.container.close()


def prepare(source, output):
    from .appearance_v1 import sample
    value = validate_source(source); output = Path(output).resolve()
    if output.exists():
        raise ValueError('deferred_render_output_must_be_new')
    appearance = sample(value['scene'], 20261029071, config={
        'background_mode': 'environment', 'table_ids': ['polyhaven:angli_veneer'],
        'environment_ids': ['polyhaven:poly_haven_studio'], 'temperature_k': [6500]})
    cases = []
    for case_id, scene in [('original', value['scene']), ('appearance', appearance)]:
        folder = output / case_id
        write(folder / 'render_scene.json', scene)
        write(folder / 'source_artifacts.json', value['bindings'])
        for name in COPY_EVIDENCE:
            shutil.copy2(value['source'] / name, folder / name)
        cases.append({'id': case_id, 'scene': binding(folder / 'render_scene.json')})
    write(output / 'protocol.json', {'schema': VERSION, 'source': str(value['source']),
        'source_execution': value['bindings']['execution_bundle.json'], 'source_artifacts': value['bindings'],
        'source_execution_id': value['bundle']['case'].get('execution_id', value['bundle']['case']['id']),
        'cases': cases, 'maximum_attempts': 2, 'timeout_seconds': 180,
        'gpu': 2, 'minimum_free_gpu_mib': 12288, 'resource_wait_seconds': 180,
        'physical_execution_count': 0, 'new_trajectory': False, 'api_calls': 0,
        'frame_count': len(value['states']), 'fps': float(value['fps']),
        'all_frames_required': True, 'state_roundtrip_tolerance': 1e-6, 'TCP_tolerance_m': 1e-5,
        'code': [binding(__file__), binding(Path(__file__).with_name('appearance_v1.py')),
            binding(ROOT / 'resimpair_pickplace/environment.py'),
            binding(ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/simulation.py')]})
    return {'output': str(output), 'cases': 2, 'frames_per_camera': len(value['states']), 'fps': float(value['fps'])}


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    import gymnasium as gym
    import torch
    from PIL import Image
    from resimpair_pickplace import environment
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator
    root = Path(root).resolve(); protocol = read(root / 'protocol.json')
    for ref in protocol['code']: resolve(ref)
    case = next(c for c in protocol['cases'] if c['id'] == case_id); folder = root / case_id
    if os.environ.get('RESIMPAIR_DEFERRED_RENDER_RESERVATION') != str(folder / 'reservation.json'):
        raise ValueError('bounded_render_supervisor_required')
    for ref in protocol['source_artifacts'].values(): resolve(ref)
    source = validate_source(protocol['source']); scene = read(resolve(case['scene']))
    if {k: v for k, v in scene.items() if k != 'appearance'} != {
            k: v for k, v in source['scene'].items() if k != 'appearance'}:
        raise ValueError('render_variants_may_only_change_appearance')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    contact = read(resolve(protocol['source_artifacts']['native_contact_policy.json']))['policy']
    environment.register(scene, Accumulator(), contact_policy=contact)
    env = None; writers = {}; rows = []; cameras = {}; last_frames = {}; frames_written = 0
    start = time.monotonic()
    try:
        env = gym.make(environment.ENV_ID, sim_backend='physx_cpu', render_backend='cuda:0',
            control_mode='pd_joint_pos', obs_mode='rgb', reward_mode='none', render_mode=None)
        env.reset(seed=int(scene['seed'] % (2**32)), options={'reconfigure': True})
        b = env.unwrapped
        if b.gpu_sim_enabled or list(b.object_ids) != source['bundle']['object_ids']:
            raise ValueError('saved_state_runtime_identity_mismatch')
        if tuple(array(b.get_state()).shape) != tuple(source['states'][0].shape):
            raise ValueError('saved_state_runtime_shape_mismatch')
        for name, config in scene['cameras'].items():
            params = b._sensors[name].get_params()
            cameras[name] = {'config': config, 'parameters': {
                k: array(v).reshape((-1,) + tuple(array(v).shape[-2:]))[0].tolist() for k, v in params.items()},
                'synchronous_with_measurements': True, 'frame_count': len(source['states']),
                'fps': float(source['fps']), 'capture_mode': 'saved_state_rerender',
                'source_state_index_equals_video_frame_index': True}
            writers[name] = NativeClockVideo(folder / (name + '.mp4'), config['width'], config['height'], source['fps'])
        for index, state in enumerate(source['states']):
            b.set_state(torch.as_tensor(state, device='cpu'))
            state_readback = array(b.get_state())
            robot_q = np.concatenate([array(a.robot.qpos).reshape(9) for a in b.agent.agents])
            object_states = np.stack([array(o.get_state()).reshape(13) for o in b.objects])
            tcp = np.stack([array(a.tcp.pose.to_transformation_matrix()).reshape(4, 4) for a in b.agent.agents])
            errors = {'state_max_abs': float(np.max(np.abs(state_readback - state))),
                'qpos_max_abs': float(np.max(np.abs(robot_q - source['arrays']['states'][index]))),
                'object_state_max_abs': float(np.max(np.abs(object_states - source['arrays']['object_states'][index]))),
                'TCP_max_abs': float(np.max(np.abs(tcp - source['arrays']['tcp_world'][index])))}
            if (max(errors[k] for k in ('state_max_abs', 'qpos_max_abs', 'object_state_max_abs')) > protocol['state_roundtrip_tolerance']
                    or errors['TCP_max_abs'] > protocol['TCP_tolerance_m']):
                write(folder / 'state_mismatch.json', {'frame': index, 'errors': errors})
                raise ValueError('restored_state_differs_from_original_measurements')
            obs = b.get_obs()  # one state, one render/readback call, both cameras
            if not np.array_equal(array(b.get_state()), state_readback):
                raise ValueError('rendering_changed_the_physics_state')
            for name, config in scene['cameras'].items():
                rgb = array(obs['sensor_data'][name]['rgb']).reshape(config['height'], config['width'], 3)
                writers[name].append(rgb, index); last_frames[name] = rgb
                if index == 0: Image.fromarray(rgb).save(folder / (name + '_initial.png'))
            rows.append({'frame': index, 'source_state_index': index,
                'source_timestamp_seconds': float(source['timestamps'][index]),
                'state_sha256': hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest(),
                'cameras': list(scene['cameras']), 'errors': errors})
            frames_written += 1
        for name, rgb in last_frames.items(): Image.fromarray(rgb).save(folder / (name + '_final.png'))
    finally:
        for writer in writers.values(): writer.close()
        if env is not None: env.close()
    if frames_written != len(source['states']):
        raise ValueError('incomplete_saved_state_render')
    write(folder / 'frame_mapping.json', {'schema': VERSION, 'source_execution': protocol['source_execution'],
        'source_states': protocol['source_artifacts']['deferred_render_states.npz'],
        'source_measurements': protocol['source_artifacts']['measurements.npz'], 'frames': rows,
        'all_states_rendered_once': True, 'env_step_calls': 0, 'contact_signals_recomputed': False})
    for name in cameras:
        cameras[name].update(video=binding(folder / (name + '.mp4')),
            initial_image=binding(folder / (name + '_initial.png')), final_image=binding(folder / (name + '_final.png')))
    bundle = deepcopy(source['bundle'])
    bundle.update(schema='resimpair_saved_state_render_bundle_v1', scene=scene, cameras=cameras,
        native_execution=False, physical_execution_count=0, saved_state_rerender=True, new_trajectory=False,
        source_execution=protocol['source_execution'], source_native_execution=True,
        source_execution_id=protocol['source_execution_id'],
        source_physical_execution_count=source['bundle']['physical_execution_count'],
        measurements=protocol['source_artifacts']['measurements.npz'],
        planned_paths=protocol['source_artifacts']['planned_paths.json'],
        event_protocol=protocol['source_artifacts']['event_protocol.json'],
        planning_declaration=protocol['source_artifacts']['planning_declaration.json'],
        code={'deferred_render_driver': binding(__file__)}, source_execution_code=source['bundle']['code'],
        frame_mapping=binding(folder / 'frame_mapping.json'),
        qualification_scope='saved-state media alignment plus inherited original native physics; no new physical execution',
        not_eligible_for_current_R2H_schema=True, training_admitted=False)
    write(folder / 'execution_bundle.json', bundle)
    write(folder / 'worker_result.json', {'schema': VERSION, 'bundle': binding(folder / 'execution_bundle.json'),
        'source_execution': protocol['source_execution'], 'physical_execution_count': 0,
        'new_trajectory': False, 'api_calls': 0, 'frames_per_camera': frames_written,
        'render_and_encoding_seconds': time.monotonic() - start,
        'maximum_errors': {k: max(r['errors'][k] for r in rows) for k in rows[0]['errors']}})


def run(root):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.resources import lock_paths
    from .cudex_shared_gpu_v1 import shared_snapshot
    root = Path(root).resolve(); protocol = read(root / 'protocol.json')
    for ref in protocol['code']: resolve(ref)
    if len(protocol['cases']) != protocol['maximum_attempts'] or protocol['maximum_attempts'] != 2:
        raise ValueError('finite_two_render_qualification_required')
    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in protocol['cases']:
            folder = root / case['id']
            if (folder / 'terminal.json').exists(): continue
            if (folder / 'reservation.json').exists():
                raise RuntimeError('existing_render_submission_requires_recovery_no_resubmit')
            deadline = time.monotonic() + protocol['resource_wait_seconds']
            acquired = None; snap = None; fds = []
            while time.monotonic() < deadline:
                candidate = ExitStack()
                try:
                    fds = []
                    for path in lock_paths(protocol['gpu']):
                        handle = candidate.enter_context(path.open('r+'))
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); fds.append(handle.fileno())
                    snap = shared_snapshot(protocol['gpu'])
                    if snap['free_memory_MiB'] < protocol['minimum_free_gpu_mib']:
                        raise RuntimeError('shared_gpu_headroom_unavailable')
                    acquired = candidate; break
                except (BlockingIOError, RuntimeError):
                    candidate.close(); time.sleep(min(3, max(0, deadline - time.monotonic())))
            if acquired is None:
                raise TimeoutError('bounded_GPU_resource_wait_expired_no_attempt_reserved')
            with acquired:
                write(folder / 'reservation.json', {'case': case, 'protocol': binding(root / 'protocol.json'),
                    'gpu': snap, 'started_unix': time.time(), 'physical_execution_count': 0,
                    'api_calls': 0, 'external_processes_preserved': True})
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=snap['uuid'], OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                    RESIMPAIR_DEFERRED_RENDER_RESERVATION=str(folder / 'reservation.json'))
                started = time.monotonic()
                with (folder / 'worker.log').open('x') as log:
                    proc = subprocess.Popen([sys.executable, '-m', 'resimpair_tasks.deferred_render_v1', 'worker',
                        '--root', str(root), '--case', case['id']], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(fds))
                    write(folder / 'process.json', {'pid': proc.pid, 'started_unix': time.time()})
                    try: code = proc.wait(timeout=protocol['timeout_seconds']); timeout = False
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try: code = proc.wait(timeout=10)
                        except subprocess.TimeoutExpired: proc.kill(); code = proc.wait(timeout=10)
                        timeout = True
                write(folder / 'terminal.json', {'returncode': code, 'timeout': timeout,
                    'elapsed_seconds': time.monotonic() - started, 'physical_execution_count': 0,
                    'new_trajectory': False, 'api_calls': 0})
                if code: raise RuntimeError('render_failure_preserved_no_automatic_retry:' + case['id'])


def audit(root):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import media, interface
    from resimpair_pickplace.requested_goal_audit import evaluate
    root = Path(root); protocol = read(root / 'protocol.json'); source = validate_source(protocol['source'])
    rows = []
    for case in protocol['cases']:
        folder = root / case['id']; bundle = read(folder / 'execution_bundle.json')
        camera_rows = []
        for name, camera in bundle['cameras'].items():
            packet = media.packet(resolve(camera['video']), folder / 'audit_frames' / name, frame_budget=9)
            delta = np.asarray(packet['all_native_pts_seconds']) - source['timestamps']
            camera_rows.append({'id': name, 'decoded_frames': packet['decoded_frames'],
                'native_frame_count_matches': packet['decoded_frames'] == len(source['states']),
                'maximum_PTS_error_seconds': float(np.abs(delta).max()),
                'PTS_match': bool(np.allclose(delta, 0., rtol=0, atol=1e-5)),
                'camera_config_unchanged': camera['config'] == source['scene']['cameras'][name]})
        result = evaluate(folder, source['bundle']['case'])
        try:
            interface.validate_bundle(bundle, source['arrays'])
            compatibility = {'accepted': True, 'reason': None}
        except Exception as exc:
            compatibility = {'accepted': False, 'reason': str(exc).split('\n')[0],
                'boundary': 'existing schema requires a fresh native execution; derivative schema and count 0 are explicit'}
        row = {'id': case['id'], 'bundle': binding(folder / 'execution_bundle.json'), 'cameras': camera_rows,
               'retained_independent_checks_passed': result['passed'], 'R2H_schema_compatibility': compatibility,
               'source_execution_id': bundle['source_execution_id'], 'physical_execution_count': 0,
               'new_trajectory': False, 'api_calls': 0}
        write(folder / 'render_audit.json', row); rows.append(row)
    write(root / 'summary.json', {'schema': VERSION, 'rows': rows,
        'original_physical_executions_reused': 1, 'new_physical_executions': 0,
        'render_jobs': len(rows), 'camera_videos': 2 * len(rows), 'api_calls': 0,
        'source_execution': protocol['source_execution'], 'no_R2H_schema_changes': True})
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--source', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    for command in ('worker', 'run', 'audit'):
        a = sub.add_parser(command); a.add_argument('--root', type=Path, required=True)
        if command == 'worker': a.add_argument('--case', required=True)
    args = p.parse_args()
    if args.command == 'prepare': print(prepare(args.source, args.output))
    elif args.command == 'run': run(args.root)
    elif args.command == 'worker': worker(args.root, args.case)
    else: print(audit(args.root))
