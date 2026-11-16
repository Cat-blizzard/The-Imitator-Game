"""Initial-state-only action replay of a complete paired TIG recording.

The original T paired actions [T,16] are submitted without replanning or state
correction. A reset wrapper restores source frame zero exactly once BEFORE
RecordEpisode captures its initial state; all later states arise from physics.
Existing replay error limits, physical contact policies, support release and
held-slip audits remain unchanged. Both task-specific placements, real release,
at least 8 cm retreat and the final common 20-tick stable window are independent
acceptance gates. Three TaskAdapters are exposed without assuming new results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import tempfile
import time
import traceback
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from .multitask_adapters import ALIASES, TASK_IDS
from .task_replay_alias import load_target_bindings, target_state, validate_live_bindings
from .replay_metrics import ACTION_KEYS, compare_states, pose_errors, summarize_errors
from .sync_replay_metrics import _array, _flag, _require, summarize_objects
from .validate_compact_replay import batch_state, replay_signal, unbatch_state
from .validate_replay import ACTION_LIMITS, STATE_LIMITS, replay_sequence, within_limits
from .validate_curobo import save_json
from .validate_grasp import json_value
from .runtime_policy import LEGACY_STRICT, RuntimePolicy, policy_from_report, resolved_policy_record
from .run_results import update_run_results_from_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[6]
CRITICAL_SOURCES = ('dual_arm_execution.py', 'dual_payload_collision.py',
                    'multitask_recording.py', 'multitask_adapters.py',
                    'validate_sync.py', 'validate_curobo_payload.py')
OPEN_PHASES = {'preparation', 'pregrasp', 'descend', 'release', 'retreat', 'verify_placement'}
REPLAY_MODES = ('strict', 'cross-version')


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    json.dumps(rows, allow_nan=False)
    return rows


def source_identity(directory: Path, report: dict[str, Any], replay_mode: str = 'strict',
                    *, current_directory: Path = HERE) -> dict[str, Any]:
    """Always authenticate the archive; only cross-version permits changed runtime files."""
    _require(replay_mode in REPLAY_MODES, 'Unknown replay mode')
    hashes = report.get('source_sha256', {})
    _require(isinstance(hashes, dict), 'Source identities must be a mapping')
    recorded: dict[str, str] = {}
    for name, digest in hashes.items():
        _require(isinstance(name, str) and isinstance(digest, str), 'Invalid source identity types')
        base = Path(name).name
        _require(base not in recorded and re.fullmatch(r'[0-9a-f]{64}', digest) is not None,
                 'Ambiguous source archive or invalid digest')
        recorded[base] = digest
        _require(hashlib.sha256((directory/'source'/base).read_bytes()).hexdigest() == digest,
                 f'Source archive changed: {base}')
    _require({path.name for path in (directory/'source').iterdir() if path.is_file()} == set(recorded),
             'Source archive file inventory differs from its manifest')
    critical = CRITICAL_SOURCES + (('runtime_policy.py',) if 'runtime_policy' in report else ())
    _require(set(critical) <= set(recorded), 'Source lacks critical execution identities')
    current_names = set(recorded) | {path.name for path in current_directory.glob('*.py') if path.is_file()}
    current = {name: hashlib.sha256((current_directory/name).read_bytes()).hexdigest()
               if (current_directory/name).is_file() else None for name in sorted(current_names)}
    differences = {name: {'source_sha256': recorded.get(name), 'current_sha256': current[name]}
                   for name in sorted(current_names) if recorded.get(name) != current[name]}
    critical_differences = [name for name in critical if name in differences]
    if replay_mode == 'strict':
        _require(not critical_differences,
                 'Current execution/contact dependency differs from source: '+', '.join(critical_differences))
    return {'replay_mode': replay_mode, 'source_archive_integrity': 'passed',
            'source_archive_sha256': recorded, 'current_source_sha256': current,
            'source_differences': differences, 'critical_sources': list(critical),
            'critical_execution_sources_match': not critical_differences,
            'critical_source_differences': critical_differences,
            'current_source_equality_required': replay_mode == 'strict'}


def load_geometry(directory: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Authenticate every saved task role, without choosing actors by position."""
    manifest = json.loads((directory/'task-geometry.json').read_text())
    _require(manifest == report.get('task_geometry'), 'Geometry JSON and report manifest differ')
    _require(hashlib.sha256((directory/'task-geometry.npz').read_bytes()).hexdigest()
             == manifest.get('geometry_npz_sha256'), 'Geometry NPZ file SHA differs')
    roles = manifest.get('actors')
    _require(isinstance(roles, dict) and bool(roles), 'Missing task geometry roles')
    values: dict[str, Any] = {}
    arrays_seen: set[str] = set()
    with np.load(directory/'task-geometry.npz', allow_pickle=False) as archive:
        for key, role in roles.items():
            _require(isinstance(key, str) and re.fullmatch(
                r'(target|destination|initial_support|destination_support|contents)[01]', key) is not None
                and isinstance(role, dict), 'Unknown task geometry role')
            vk, pk = role.get('local_vertices_array'), role.get('world_pose_array')
            _require(isinstance(vk, str) and isinstance(pk, str) and vk != pk
                     and not {vk, pk} & arrays_seen, 'Ambiguous geometry arrays')
            arrays_seen.update((vk, pk))
            vertices, pose = archive[vk], archive[pk]
            _require(vertices.ndim == 2 and vertices.shape[1:] == (3,) and vertices.dtype.kind == 'f'
                     and np.isfinite(vertices).all() and pose.shape == (7,)
                     and pose.dtype.kind == 'f' and np.isfinite(pose).all(), 'Invalid task geometry arrays')
            _require(abs(float(np.linalg.norm(pose[3:]))-1.) <= 1e-5, 'Nonunit geometry quaternion')
            _require(role.get('shape') == list(vertices.shape) and role.get('vertex_count') == len(vertices),
                     'Geometry role vertex count/shape differs')
            digest = hashlib.sha256(np.ascontiguousarray(vertices, dtype=np.float64).tobytes(order='C')).hexdigest()
            _require(digest == role.get('local_vertices_sha256'), f'Geometry role vertices SHA differs: {key}')
            _require(key.startswith('destination') and not key.startswith('destination_support')
                     or len(vertices) > 0, f'Missing physical role mesh: {key}')
            _require(all(isinstance(role.get(field), str) and bool(role[field])
                         for field in ('actor_name', 'state_name')), 'Missing task actor/state identity')
            values[key] = {'role': role, 'vertices': vertices.copy(), 'pose': pose.copy()}
        _require(set(archive.files) == arrays_seen, 'Geometry NPZ array inventory differs')
    return values


def validate_live_geometry(geometry: dict[str, Any], adapter: Any, scene_actors: Any) -> dict[str, Any]:
    """Check task asset identities and native collision vertices before the first action."""
    from .trajectory_execution import as_numpy
    live = {f'{role}{side}': actor for role, actors in (
        ('target', adapter.targets), ('destination', adapter.destination_actors),
        ('initial_support', adapter.initial_supports), ('destination_support', adapter.destination_supports),
        ('contents', getattr(adapter, 'destination_contents', ()))) for side, actor in enumerate(actors)}
    _require(set(live) == set(geometry), 'Live task geometry role inventory differs')
    receipts = {}
    for key, actor in live.items():
        recorded = geometry[key]
        role = recorded['role']
        _require(actor.name == role['actor_name']
                 and [name for name, value in scene_actors.items() if value is actor] == [role['state_name']],
                 f'Live task geometry actor/state identity differs: {key}')
        if key.startswith('contents'):
            side = int(key[-1])
            _require(role.get('canonical_target_index') == side
                     and role.get('corresponding_target') == adapter.target_names[side]
                     and role.get('corresponding_destination_actor') == adapter.destination_actors[side].name,
                     f'Live destination contents semantic pairing differs: {key}')
        mesh = actor.get_first_collision_mesh(to_world_frame=False)
        vertices = np.asarray(np.empty((0, 3)) if mesh is None else mesh.vertices, dtype=np.float64)
        _require(np.array_equal(vertices, recorded['vertices']), f'Live task collision mesh differs: {key}')
        pose = np.asarray(as_numpy(actor.pose.raw_pose), dtype=np.float64)
        _require(pose.shape == (1, 7) and np.array_equal(pose[0], recorded['pose']),
                 f'Live task initial role pose differs: {key}')
        receipts[key] = {'actor_name': actor.name, 'state_name': role['state_name'],
                         'local_vertices_sha256': role['local_vertices_sha256'],
                         'collision_vertices_exact': True, 'initial_world_pose_exact': True}
    return {'status': 'passed', 'role_inventory_exact': True, 'roles': receipts,
            'scope': 'task asset actor/state identities, actor-local native collision vertices and initial world poses'}


def load_recording(directory: Path, task: str | None = None, *,
                   replay_mode: str = 'strict') -> dict[str, Any]:
    """Validate source H5/JSON/tick/source identities without changing evidence."""
    directory = Path(directory).resolve()
    report = json.loads((directory/'report.json').read_text())
    _require(replay_mode in REPLAY_MODES, 'Unknown replay mode')
    policy = policy_from_report(report)
    metadata = json.loads((directory/'trajectory.json').read_text())
    json.dumps([report, metadata], allow_nan=False)
    ticks = _read_rows(directory/'ticks.jsonl')
    episodes, info = metadata.get('episodes', []), metadata.get('env_info', {})
    _require(len(episodes) == 1, 'Exactly one source episode is required')
    episode, kwargs = episodes[0], info.get('env_kwargs', {})
    canonical = ALIASES.get(task, task) if task is not None else info.get('env_id')
    _require(canonical in TASK_IDS and report.get('task') == info.get('env_id') == canonical,
             'Source and selected task identities disagree')
    _require(report.get('status') == 'passed' and report.get('mode') == 'positive'
             and report.get('project_complete_physical_closure') is True
             and report.get('official_200_step_complete') is True,
             'A passed complete official200 positive recording is required')
    _require(report.get('max_episode_steps') == info.get('max_episode_steps') == policy.max_episode_steps == 200
             and kwargs.get('max_episode_steps', 200) == 200, 'Official200 metadata disagrees')
    _require(kwargs.get('control_mode') == 'pd_joint_pos' and kwargs.get('sim_backend') == 'cpu'
             and kwargs.get('obs_mode') == 'none', 'Replay requires absolute control and CPU no-observation recording')
    _require(report.get('control_timestep_s') == .05 and report.get('physics_timestep_s') == .01,
             'Source control/physics timestamps must be20Hz/100Hz')
    _require(report.get('seed') == episode.get('episode_seed') == episode.get('reset_kwargs', {}).get('seed'),
             'Source seed receipts disagree')
    modes = episode.get('control_mode', {})
    _require(list(modes) == ACTION_KEYS and all(value == 'pd_joint_pos' for value in modes.values()),
             'Physical source action order or control mode changed')
    _require(report.get('recording', {}).get('action_key_order') == ACTION_KEYS,
             'RecordEpisode action mapping receipt missing')
    target_names = report.get('target_names')
    _require(isinstance(target_names, list) and len(target_names) == 2 and len(set(target_names)) == 2,
             'Two canonical target names are required')
    with h5py.File(directory/'trajectory.h5', 'r') as handle:
        key = f"traj_{episode['episode_id']}"
        _require(list(handle) == [key], 'Unexpected source H5 episodes')
        group = handle[key]
        actions = group['actions'][:]
        count = len(actions)
        _require(actions.shape == (count, 16) and actions.dtype.kind == 'f'
                 and np.isfinite(actions).all() and 0 < count <= policy.last_completion_step < 200,
                 'Actions must be finite [T,16] and complete before200')
        _require(count == len(ticks) == report.get('total_environment_steps')
                 == episode.get('elapsed_steps') == report.get('recording', {}).get('recorded_steps'),
                 'Source step receipts disagree')
        _require(set(group['env_states']) == {'actors', 'articulations'}, 'Source state categories differ')
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for category in ('actors', 'articulations'):
            arrays[category] = {}
            for name, dataset in group['env_states'][category].items():
                array = dataset[:]
                width = 13 if category == 'actors' else 31
                _require(array.shape == (count+1, width) and array.dtype.kind == 'f'
                         and np.isfinite(array).all(), 'Invalid source T+1 state shape/values')
                _require(np.all(np.abs(np.linalg.norm(array[:, 3:7], axis=1)-1.) <= 1e-3),
                         'Nonunit source state quaternion')
                arrays[category][name] = array
        _require(set(arrays['articulations']) == {'panda_wristcam-agent-0', 'panda_wristcam-agent-1'},
                 'Physical articulation identity differs')
        target_bindings, binding_receipt = load_target_bindings(directory, report, arrays['actors'])
        flags = {name: group[name][:] for name in ('success', 'terminated', 'truncated')}
    _require(all(value.shape == (count,) and value.dtype.kind == 'b' for value in flags.values()),
             'Native H5 flag type/shape mismatch')
    _require(not flags['truncated'].any() and not np.any(flags['terminated'] & ~flags['success']),
             'Source contains a timeout or failure termination')
    states = [{category: {name: value[index].copy() for name, value in values.items()}
               for category, values in arrays.items()} for index in range(count+1)]
    segments: list[str] = []
    counts: Counter[str] = Counter()
    for index, tick in enumerate(ticks):
        _require(tick.get('environment_step') == index+1 and not _flag(tick.get('fault_injected'), 'fault'),
                 'Source ticks must be sequential and unmodified positive actions')
        phase = tick.get('phase')
        _require(isinstance(phase, str) and phase, 'Missing source phase')
        counts[phase] += 1
        if not segments or segments[-1] != phase:
            segments.append(phase)
        for field in ('actions', 'requested_actions'):
            _require(list(tick[field]) == ACTION_KEYS, 'Tick action mapping changed')
            actual = np.concatenate([_array(tick[field][key], (8,), field) for key in ACTION_KEYS]).astype(actions.dtype)
            _require(np.array_equal(actual, actions[index]), 'Source H5 and requested/applied actions disagree')
        expected_grip = 1 if phase in OPEN_PHASES else -1
        _require(np.all(actions[index, [7, 15]] == expected_grip), 'Source phase/gripper mismatch')
        for flag, value in (('success', tick['official_evaluation']['success']),
                            ('terminated', tick['native_terminated']), ('truncated', tick['native_truncated'])):
            _require(_flag(value, flag) == bool(flags[flag][index]), 'Source native H5/tick flags disagree')
        _require(_flag(tick['success_only_termination_continued'], 'continued')
                 == bool(flags['success'][index] and flags['terminated'][index]),
                 'Source termination continuation receipt disagrees')
        for side, name in enumerate(target_names):
            obj = tick['objects'][side]
            _require(obj.get('actor') == name, 'Source target pairing changed')
            measured = np.r_[_array(obj['pose'], (7,), 'pose'),
                             _array(obj['linear_velocity_m_s'], (3,), 'velocity'),
                             _array(obj['angular_velocity_rad_s'], (3,), 'angular velocity')]
            actor = target_state(states[index+1], target_bindings[side])
            _require(np.array_equal(measured.astype(actor.dtype), actor), 'Target telemetry differs from H5 frame i+1')
            robot = states[index+1]['articulations'][f'panda_wristcam-agent-{side}']
            _require(np.array_equal(np.asarray(tick[('left_qpos', 'right_qpos')[side]], dtype=robot.dtype), robot[13:22]),
                     'Joint telemetry differs from H5 frame i+1')
            pose_errors(tick['tcp_poses'][side], tick['tcp_poses'][side])
            _flag(obj['is_grasping'], 'grasp')
    prefix = ['preparation', 'pregrasp', 'descend', 'close_and_confirm', 'lift_and_hold']
    release_index = segments.index('release') if 'release' in segments else -1
    retreat_options = [['retreat']] if policy.require_retreat else [[], ['retreat']]
    final_options = [['verify_placement']] if policy.require_final_stability else [[], ['verify_placement']]
    valid_tails = [['release']+retreat+final for retreat in retreat_options for final in final_options]
    _require(segments[:5] == prefix and release_index >= 5
             and segments[release_index:] in valid_tails
             and all(phase.startswith('carry_') for phase in segments[5:release_index]),
             'Source physical phase sequence is incomplete')
    _require(counts['close_and_confirm'] >= policy.grasp_stable_steps
             and counts['release'] >= policy.release_stable_steps
             and (not policy.require_final_stability or counts['verify_placement'] >= policy.final_stable_steps)
             and flags['success'][-(policy.final_stable_steps if policy.require_final_stability else 1):].all(),
             'Source lacks its resolved real grasp/release/final stability gates')
    stages = report.get('stages', [])
    _require([stage.get('name') for stage in stages] == segments[1:] and all(
        stage.get('status') == 'passed' and stage.get('environment_steps') == counts[stage['name']]
        for stage in stages), 'Source stage receipts disagree')
    identity = source_identity(directory, report, replay_mode)
    geometry = load_geometry(directory, report)
    return {'actions': actions, 'states': states, 'ticks': ticks, 'report': report, 'metadata': metadata,
            'native_flags': flags, 'action_keys': list(ACTION_KEYS), 'target_names': target_names,
            'task': canonical, 'source_sha256': report['source_sha256'], 'target_bindings': target_bindings,
            'runtime_policy': policy, 'geometry': geometry, 'source_identity': identity,
            'receipt': {'action_steps': count, 'state_frames': count+1, 'phase_counts': dict(counts),
                        'task': canonical, 'replay_mode': replay_mode,
                        'critical_execution_sources_match': identity['critical_execution_sources_match'],
                        'source_archive_integrity': 'passed', 'task_geometry_integrity': 'passed',
                        'runtime_policy_origin': 'source report' if 'runtime_policy' in report else 'legacy fallback',
                        'target_state_bindings': binding_receipt}}


def restore_initial_fixed_roots(raw: Any, state: dict[str, Any]) -> dict[str, Any]:
    """Use the existing exact-root policy; extend only its actor inventory for TaskAdapters."""
    from .fixed_root_restore import restore_fixed_roots
    from .sync_replay_metrics import LEAVES
    if set(state) == set(LEAVES) and all(set(state[category]) == names for category, names in LEAVES.items()):
        return restore_fixed_roots(raw, state)
    # Same preflight and non-pose setter sequence as fixed_root_restore.py.
    # The old helper intentionally hardcodes apple/banana's actor inventory.
    if raw.gpu_sim_enabled or raw.num_envs != 1:
        raise ValueError('Initial exact-root restore requires one CPU environment')
    current = unbatch_state(raw.get_state_dict())
    compare_states(state, current)
    _require(set(state['articulations']) == {'panda_wristcam-agent-0', 'panda_wristcam-agent-1'},
             'Exactly the canonical physical Panda pair is required')
    registry = raw.scene.state_dict_registry.articulations
    for name, requested in state['articulations'].items():
        articulation = registry[name]
        _require(articulation.max_dof == 9 and bool(np.asarray(articulation.fixed_root_link).all())
                 and current['articulations'][name].dtype == requested.dtype
                 and np.array_equal(current['articulations'][name][:7], requested[:7]),
                 'Fixed-root exact precondition failed before any setter')
    raw.set_state_dict({'actors': batch_state(state['actors'])})
    events = ['set_state_dict(actors)']
    for name, requested in state['articulations'].items():
        articulation = registry[name]
        articulation.set_root_linear_velocity(requested[7:10].copy())
        articulation.set_root_angular_velocity(requested[10:13].copy())
        articulation.set_qpos(requested[13:22].copy())
        articulation.set_qvel(requested[22:31].copy())
        events.append(f'{name}:root_linear_velocity,root_angular_velocity,qpos,qvel')
    actual = unbatch_state(raw.get_state_dict())
    _require(all(np.array_equal(actual['articulations'][name][:7], requested[:7])
                 for name, requested in state['articulations'].items()), 'Fixed root changed during restoration')
    return {'policy': 'already_exact_fixed_root_pose_task_actor_inventory',
            'root_pose_precondition_exact': True, 'root_pose_postcondition_exact': True,
            'skipped_root_pose_setters': list(state['articulations']), 'setter_events': events,
            'hidden_physics_state_restored': False}


def outcome(rows: list[dict[str, Any]], expected_steps: int, *,
            policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, Any]:
    """Apply the source's resolved limits; legacy callers retain the same gates."""
    from .sync_replay_metrics import _rows
    _rows(rows)
    _require(isinstance(policy, RuntimePolicy), 'A validated runtime policy is required')
    _require(type(expected_steps) is int and expected_steps > 0, 'Expected a positive source action count')
    ticks = rows[1:]
    release = [row for row in ticks if row['phase'] == 'release']
    retreat = [row for row in ticks if row['phase'] == 'retreat']
    final_count = policy.final_stable_steps if policy.require_final_stability else 1
    final = ticks[-final_count:]
    rises = [float(retreat[-1]['objects'][side]['tcp_pose'][2]-release[-1]['objects'][side]['tcp_pose'][2])
             if release and retreat else 0. for side in (0, 1)]
    checks = {'all_actions_observed': len(ticks) == expected_steps,
        'both_grasp_sequences_match': bool(ticks) and all(obj['grasp_matches_reference'] for row in ticks for obj in row['objects']),
        'both_physically_released_last_three': len(release) >= policy.release_stable_steps
            and all(obj['fingers_open'] and not obj['is_grasping']
                    for row in release[-policy.release_stable_steps:] for obj in row['objects']),
        'both_actual_retreats_at_least80mm': not policy.require_retreat
            or all(value >= policy.minimum_retreat_m for value in rises),
        'release_retreat_order': not policy.require_retreat
            or bool(release and retreat and release[-1]['frame'] < retreat[0]['frame']),
        'common_final20_window': not policy.require_final_stability or (len(final) == final_count
            and all(row['phase'] == 'verify_placement' for row in final)
            and final[-1]['frame'] == expected_steps
            and final[-1]['consecutive_stable_placement_steps'] >= policy.final_stable_steps),
        'both_final_placements_valid': not policy.require_final_stability or (len(final) == final_count
            and all(obj['placement_valid'] for row in final for obj in row['objects'])),
        'final_official_success': len(final) == final_count and all(row['native_success'] for row in final),
        'no_failure_or_truncation': all(not row['native_truncated'] and not row['native_fail']
            and (not row['native_terminated'] or row['native_success']) for row in ticks)}
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'actual_retreat_tcp_rise_m': rises, 'release_steps': len(release),
            'placement_steps': sum(row['phase'] == 'verify_placement' for row in ticks),
            'runtime_policy': resolved_policy_record(policy),
            'check_names': 'legacy names retained; resolved runtime_policy supplies counts and limits'}


def measure_replay_lift(rows: list[dict[str, Any]], target_names: Any) -> dict[str, Any]:
    """Observe height gains from the phase-entry state, independently of source results."""
    names = list(target_names)
    _require(len(names) == 2 and all(isinstance(name, str) and name for name in names)
             and len(set(names)) == 2, 'Two ordered semantic lift targets are required')
    result: dict[str, Any] = {'status': 'unknown', 'target_names': names,
        'reference_frame': None, 'first_lift_frame': None, 'end_frame': None,
        'gains_m': None, 'reference_object_z_m': None, 'end_object_z_m': None,
        'source': 'current replay observed rows',
        'scope': 'frame immediately before first lift_and_hold action to last observed lift_and_hold frame; not episode frame0'}
    indices = [index for index, row in enumerate(rows) if row.get('phase') == 'lift_and_hold']
    if not indices:
        result['reason'] = 'lift_phase_not_observed'
        return result
    first, last = indices[0], indices[-1]
    if first == 0:
        result['reason'] = 'pre_lift_reference_not_observed'
        return result
    reference, initial, end = rows[first-1], rows[first], rows[last]
    frames = [row.get('frame') for row in rows[first-1:last+1]]
    if (indices != list(range(first, last+1))
            or any(type(frame) is not int for frame in frames)
            or frames != list(range(frames[0], frames[0]+len(frames)))):
        result['reason'] = 'lift_frames_not_contiguous'
        return result
    try:
        heights = []
        for row in (reference, end):
            objects = row['objects']
            if len(objects) != 2 or [obj.get('actor') for obj in objects] != names:
                raise ValueError('Target pairing differs')
            poses = np.asarray([obj['pose'] for obj in objects], dtype=float)
            if poses.shape != (2, 7) or not np.isfinite(poses).all():
                raise ValueError('Target poses unavailable')
            heights.append(poses[:, 2])
    except (KeyError, TypeError, ValueError):
        result['reason'] = 'paired_measured_target_poses_unavailable'
        return result
    result.update(status='observed', reference_frame=reference['frame'],
        first_lift_frame=initial['frame'], end_frame=end['frame'],
        gains_m=(heights[1]-heights[0]).tolist(),
        reference_object_z_m=heights[0].tolist(), end_object_z_m=heights[1].tolist(),
        lift_phase_exited=last < len(rows)-1)
    return result


def video_timing(directory: Path, actions: int, report: dict[str, Any]) -> dict[str, Any]:
    """Count actual video frames against T+1 states plus declared presentation padding."""
    import imageio.v2 as imageio
    video = report.get('video', {})
    before, after = video.get('padding_frames_before'), video.get('padding_frames_after')
    _require(type(before) is int and type(after) is int and before >= 0 and after >= 0,
             'Explicit nonnegative video padding is required')
    reader = imageio.get_reader(directory/'execution.mp4')
    try:
        count = reader.count_frames()
        fps = float(reader.get_meta_data()['fps'])
    finally:
        reader.close()
    expected = actions+1+before+after
    return {'status': 'passed' if count == expected and abs(fps-20.) <= 1e-6 else 'failed',
            'actual_frames': count, 'expected_frames': expected, 'physical_state_frames': actions+1,
            'padding_frames_before': before, 'padding_frames_after': after, 'fps': fps}


def native_run(args: argparse.Namespace, recording: dict[str, Any], output: Path,
               report: dict[str, Any]) -> None:
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .multitask_adapters import get_task_adapter
    from .multitask_recording import make_physics_capture, finalize_physics_capture, rename_held_audit_targets
    from .trajectory_execution import as_numpy
    from .validate_curobo_payload import install_full_robot_contact_capture
    from .validate_curobo_complete import make_checker, measured_source_departure
    from .validate_sync import held_substep_audit
    env, raw, capture, checker, original_hook, adapter = None, None, None, None, None, None
    frames, ticks, substeps, contacts, measured, rows, errors, applied = [], [], [], [], [], [], [], []
    keys = recording['action_keys']
    restores: list[dict[str, Any]] = []
    references = None
    report.update(artifact_errors=[], fault_events=[], state_restore_frames=[], restoration_receipts=restores)
    prior_poses: list[np.ndarray] = []
    previous_fingers: list[np.ndarray] = []
    close_counts = [0, 0]
    placement_anchor: list[np.ndarray] | None = None
    stable_count = 0
    policy = recording.get('runtime_policy', policy_from_report(recording['report']))
    _require(isinstance(policy, RuntimePolicy), 'A validated source runtime policy is required')
    try:
        metadata = recording['metadata']
        kwargs = dict(metadata['env_info']['env_kwargs'])
        kwargs['max_episode_steps'] = policy.max_episode_steps
        base = gym.make(recording['task'], **kwargs)

        class RestoreAtReset(gym.Wrapper):
            def reset(self, *reset_args: Any, **reset_kwargs: Any) -> tuple[Any, Any]:
                if restores:
                    raise RuntimeError('Replay may reset and restore its initial state only once')
                obs, info = self.env.reset(*reset_args, **reset_kwargs)
                restores.append(restore_initial_fixed_roots(self.unwrapped, recording['states'][0]))
                return obs, info

        env = RecordEpisode(RestoreAtReset(base), output_dir=str(output), trajectory_name='trajectory',
            save_video=False, save_on_reset=False, clean_on_close=False, record_reward=False,
            record_env_state=True, source_type='action_replay',
            source_desc='original paired actions; frame0 restored once before standard recorder capture')

        def restore(index: int, state: dict[str, Any]) -> None:
            nonlocal raw, adapter, checker, capture, original_hook, prior_poses, previous_fingers
            _require(index == 0 and not restores, 'Only initial source restoration is allowed')
            env.reset(**metadata['episodes'][0]['reset_kwargs'])
            raw = env.unwrapped
            report['state_restore_frames'].append(0)
            report.update(control_timestep_s=float(raw.control_timestep),
                          physics_timestep_s=float(raw.sim_timestep))
            _require(not raw.gpu_sim_enabled and raw.num_envs == 1
                     and np.isclose(raw.control_timestep, .05, atol=1e-10, rtol=0)
                     and np.isclose(raw.sim_timestep, .01, atol=1e-10, rtol=0), 'Live control/physics differs')
            _require(list(raw.agent.agents_dict) == keys, 'Live canonical action mapping differs')
            adapter = get_task_adapter(recording['task'], raw)
            _require(list(adapter.target_names) == recording['target_names'], 'Live target identity differs')
            report['live_target_state_bindings'] = validate_live_bindings(
                recording['target_bindings'], adapter.target_names, adapter.targets,
                raw.scene.actors, unbatch_state(raw.get_state_dict()))
            if 'geometry' in recording:
                report['live_task_geometry_compatibility'] = validate_live_geometry(
                    recording['geometry'], adapter, raw.scene.actors)
            report['task_configuration'] = adapter.configuration()
            for agent in raw.agent.agents:
                _require(agent.control_mode == 'pd_joint_pos', 'Expected absolute joint position control')
                for controller in agent.controller.controllers.values():
                    _require(not any(bool(getattr(controller.config, name)) for name in
                        ('use_delta', 'use_target', 'interpolate')), 'Controller requires unrecorded internal state')
            solvers = [PandaArmMotionPlanningSolver(env, base_pose=agent.robot.pose, multi_robot_id=side,
                joint_vel_limits=.9, joint_acc_limits=.9, debug=False, vis=False,
                visualize_target_grasp_pose=False, print_env_info=False)
                for side, agent in enumerate(raw.agent.agents)]
            checker = make_checker(raw, solvers, adapter)
            capture = make_physics_capture(env, raw, adapter, SimpleNamespace(mode='positive'), report,
                frames, ticks, substeps, keys, checker_provider=lambda: checker,
                support_release_provider=lambda: measured_source_departure(checker),
                hard_step_budget=policy.max_episode_steps)
            original_hook = capture.original_hook
            install_full_robot_contact_capture(raw, capture, adapter, contacts)
            raw._after_simulation_step = capture.record_substep
            prior_poses = [as_numpy(actor.pose.raw_pose)[0].copy() for actor in adapter.targets]
            previous_fingers = [as_numpy(agent.robot.get_qpos())[0, -2:].copy() for agent in raw.agent.agents]

        def observe(index: int) -> None:
            nonlocal prior_poses, previous_fingers, placement_anchor, stable_count
            actual = unbatch_state(raw.get_state_dict())
            measured.append(actual)
            comparison = compare_states(recording['states'][index], actual)
            errors.append(comparison)
            reference = recording['ticks'][index-1] if index else None
            phase = reference['phase'] if reference else 'initial'
            source = ticks[-1] if index else None
            flag = replay_signal(source['native_terminated'], source['native_truncated'], source['official_evaluation']) if source else None
            evaluation = json_value(raw.evaluate())
            arm_speed = max(float(np.max(np.abs(as_numpy(agent.robot.get_qvel())[0, :7]))) for agent in raw.agent.agents)
            row = {'frame': index, 'phase': phase, 'environment_step': int(as_numpy(raw.elapsed_steps)[0]),
                'errors': comparison, 'objects': [], 'max_raw_arm_speed_rad_s': arm_speed,
                'native_success': flag['success'] if flag else False,
                'native_terminated': flag['terminated'] if flag else False,
                'native_truncated': flag['truncated'] if flag else False,
                'native_fail': flag['fail'] if flag else False, 'official_evaluation': evaluation,
                'consecutive_stable_placement_steps': stable_count}
            poses = [as_numpy(actor.pose.raw_pose)[0].copy() for actor in adapter.targets]
            fingers_now = []
            for side, (name, actor, agent) in enumerate(zip(adapter.target_names, adapter.targets, raw.agent.agents)):
                tcp = as_numpy(agent.tcp.pose.raw_pose)[0].copy()
                fingers = as_numpy(agent.robot.get_qpos())[0, -2:].copy()
                fingers_now.append(fingers)
                grasp = bool(as_numpy(agent.is_grasping(actor)).reshape(-1)[0])
                opened = bool(np.allclose(fingers, as_numpy(agent.robot.get_qlimits())[0, -2:, 1], atol=.002, rtol=0))
                motion = pose_errors(prior_poses[side], poses[side])
                obj = {'actor': name, 'pose': poses[side].tolist(), 'tcp_pose': tcp.tolist(),
                    'pose_error': pose_errors(target_state(recording['states'][index],
                                                          recording['target_bindings'][side])[:7], poses[side]),
                    'tcp_pose_error': pose_errors(reference['tcp_poses'][side], tcp) if reference else None,
                    'is_grasping': grasp, 'fingers_open': opened, 'finger_qpos_m': fingers.tolist(),
                    'grasp_matches_reference': grasp == reference['objects'][side]['is_grasping'] if reference else True,
                    'control_frame_linear_speed_m_s': motion['pose_translation_m']/.05,
                    'control_frame_angular_speed_rad_s': motion['pose_rotation_rad']/.05,
                    'placement_valid': False}
                if phase == 'close_and_confirm':
                    changed = float(np.max(np.abs(fingers-previous_fingers[side])))
                    close_counts[side] = close_counts[side]+1 if grasp and changed <= .0002 else 0
                if phase == 'verify_placement':
                    obj.update(physical_region=adapter.region_predicates()[side](actor),
                               official_region=_flag(evaluation[adapter.official_predicate_keys[side]], 'official region'))
                row['objects'].append(obj)
            if phase == 'verify_placement':
                basic = arm_speed <= .05 and all(obj['physical_region']['inside'] and obj['official_region']
                    and obj['fingers_open'] and not obj['is_grasping']
                    and obj['control_frame_linear_speed_m_s'] <= .03
                    and obj['control_frame_angular_speed_rad_s'] <= .2 for obj in row['objects'])
                if not basic:
                    stable_count, placement_anchor = 0, None
                else:
                    if placement_anchor is None:
                        placement_anchor = [pose.copy() for pose in poses]
                    drifted = False
                    for obj, pose, anchor in zip(row['objects'], poses, placement_anchor):
                        drift = pose_errors(anchor, pose)
                        obj.update(window_position_drift_m=drift['pose_translation_m'],
                                   window_rotation_drift_rad=drift['pose_rotation_rad'])
                        drifted |= drift['pose_translation_m'] > .003 or drift['pose_rotation_rad'] > .03
                    if drifted:
                        stable_count, placement_anchor = 1, [pose.copy() for pose in poses]
                    else:
                        stable_count += 1
                    for obj in row['objects']:
                        obj['placement_valid'] = not drifted
                row['consecutive_stable_placement_steps'] = stable_count
            prior_poses, previous_fingers = poses, fingers_now
            rows.append(row)
            if index == 0:
                capture.frame()

        def step(action: dict[str, np.ndarray]) -> None:
            nonlocal references
            index = len(applied)
            phase = recording['ticks'][index]['phase']
            capture.phase = phase
            if phase == 'lift_and_hold' and not checker.attached:
                _require(min(close_counts) >= policy.grasp_stable_steps
                    and all(bool(as_numpy(agent.is_grasping(actor)).reshape(-1)[0])
                    for agent, actor in zip(raw.agent.agents, adapter.targets)), 'Replay has not independently confirmed both real grasps')
                checker.attach()
                references = np.asarray(checker.attachments).copy()
                report['attachment_capture'] = {'T_tcp_objects': references.tolist(), 'physical_weld': False,
                    'grasp_stable_steps_observed': list(close_counts), 'environment_step': index}
            if phase == 'release' and checker.attached:
                checker.detach()
            allowed = phase in ('descend', 'close_and_confirm', 'lift_and_hold', 'release') or phase.startswith('carry_')
            checker.set_target_contacts_allowed((allowed, allowed))
            checker.refresh()
            before = int(as_numpy(raw.elapsed_steps)[0])
            try:
                capture.step(action)
                signal_row = ticks[-1]
                signal = replay_signal(signal_row['native_terminated'], signal_row['native_truncated'], signal_row['official_evaluation'])
                if signal['unexpected_end']:
                    raise RuntimeError('Replay reached failure or native truncation')
                checker.refresh()
                blocked = checker.measured_check()
                if blocked:
                    report.setdefault('measured_collision_failures', []).append({'step': before+1, 'contacts': blocked})
                    raise RuntimeError('Replay measured exact collision check failed')
            except BaseException:
                if int(as_numpy(raw.elapsed_steps)[0]) == before+1:
                    applied.append(np.concatenate([action[key] for key in keys]).copy())
                    observe(len(applied))
                raise
            _require(int(as_numpy(raw.elapsed_steps)[0]) == before+1, 'Each paired action must advance exactly one control tick')
            applied.append(np.concatenate([action[key] for key in keys]).copy())

        report['sequence'] = replay_sequence(recording, 'action', restore=restore, step=step, observe=observe)
        report['trajectory_errors'] = summarize_errors(errors)
        objects = summarize_objects(rows)
        for obj, name in zip(objects['objects'], adapter.target_names):
            obj['actor'] = name  # Existing numerical reducer has legacy apple/banana labels only.
        report['object_errors'] = objects
        report['outcome'] = outcome(rows, len(recording['actions']), policy=policy)
        report['held_physics_audit'] = rename_held_audit_targets(
            held_substep_audit(substeps, references, policy=policy), adapter.target_names)
        checks = {'all_state_errors_within_existing_action_limits': within_limits(report['trajectory_errors']['global'], ACTION_LIMITS),
            'all_frames_observed': len(rows) == len(recording['states']),
            'one_initial_restore_only': report['state_restore_frames'] == [0] and len(restores) == 1,
            'one_control_step_per_original_action': int(as_numpy(raw.elapsed_steps)[0]) == len(recording['actions']),
            'actions_applied_exactly': np.array_equal(np.asarray(applied), recording['actions']),
            'outcome_reproduced': report['outcome']['status'] == 'passed',
            'held_slip_within_existing_limits': report['held_physics_audit']['status'] == 'passed',
            'all100Hz_observations_recorded': len(substeps) == len(applied)*5 == len(contacts),
            'no_forbidden100Hz_contacts': not any(contact['physical_contact'] and not contact['allowed']
                for entry in contacts for contact in entry['contacts']) and not any(
                contact['physical_contact'] and contact['class'] == 'forbidden'
                for entry in substeps for contact in entry['object_contact_rows'])}
        for side, obj in enumerate(objects['objects']):
            for group in ('object_pose', 'tcp_pose'):
                checks[f'target{side}_{group}_within_existing_limits'] = within_limits(obj[group],
                    {key: ACTION_LIMITS[key] for key in obj[group]})
        checks['native_flags_match_source'] = all(np.array_equal(np.asarray([
            _flag(tick['official_evaluation']['success'], 'success') if name == 'success'
            else tick[f'native_{name}'] for tick in ticks]), recording['native_flags'][name])
            for name in ('success', 'terminated', 'truncated'))
        report['checks'] = checks
        report['status'] = 'passed' if all(checks.values()) else 'failed'
    finally:
        report['replay_lift_measurement'] = measure_replay_lift(rows, recording['target_names'])
        save_json(output/'all-robot-contacts.json', contacts)
        with (output/'frames.jsonl').open('w') as handle:
            for row in rows:
                handle.write(json.dumps(json_value(row), allow_nan=False)+'\n')
        if env is not None:
            finalize_physics_capture(env, raw, capture, output, report, frames, ticks, substeps,
                                     keys, original_hook, robot_contacts=contacts)
        if measured:
            with h5py.File(output/'trajectory.h5', 'r') as handle:
                group = handle[next(iter(handle))]
                written = group['env_states']
                state_equal = all(np.array_equal(written[category][name][:],
                    np.stack([state[category][name] for state in measured]))
                    for category in measured[0] for name in measured[0][category])
                action_equal = np.array_equal(group['actions'][:], np.asarray(applied))
            report['standard_recording_timing'] = {'status': 'passed' if state_equal and action_equal
                and len(measured) == len(applied)+1 else 'failed', 'state_frames': len(measured),
                'action_steps': len(applied), 'all_measured_states_match_H5': state_equal,
                'all_submitted_actions_match_H5': action_equal,
                'initial_restore_precedes_RecordEpisode_frame0': True}
            report['video_timing'] = video_timing(output, len(applied), report)
            if report['standard_recording_timing']['status'] != 'passed' or report['video_timing']['status'] != 'passed':
                report['status'] = 'failed'
        report['observed_state_frames'], report['applied_action_steps'] = len(measured), len(applied)


def self_test() -> dict[str, Any]:
    """CPU-only action/restore sequencing, final-window limits and malformed-source tests."""
    rejected = 0
    def reject(call: Any) -> None:
        nonlocal rejected
        try:
            call()
        except (ValueError, RuntimeError):
            rejected += 1
        else:
            raise AssertionError('Invalid replay input was accepted')
    actions = np.zeros((2, 16), dtype=np.float32)
    actions[:, [7, 15]] = 1.
    events = []
    sequence = replay_sequence({'states': [{}, {}, {}], 'actions': actions, 'action_keys': ACTION_KEYS},
        'action', restore=lambda i, state: events.append(('restore', i)),
        step=lambda action: events.append(('step', len(action))), observe=lambda i: events.append(('observe', i)))
    assert events == [('restore', 0), ('observe', 0), ('step', 2), ('observe', 1), ('step', 2), ('observe', 2)]
    assert sequence == {'state_restore_calls': 1, 'physical_environment_steps': 2, 'observed_frames': 3}
    reject(lambda: replay_sequence({'states': [{}], 'actions': actions, 'action_keys': ACTION_KEYS},
        'action', restore=lambda *a: None, step=lambda a: None, observe=lambda i: None))
    rows = []
    for frame in range(26):
        phase = 'initial' if frame == 0 else 'release' if frame <= 3 else 'retreat' if frame <= 5 else 'verify_placement'
        objects = [{'tcp_pose': [0., 0., .1 if frame <= 3 else .181, 1., 0., 0., 0.],
            'grasp_matches_reference': True, 'fingers_open': True, 'is_grasping': False,
            'placement_valid': phase == 'verify_placement'} for _ in (0, 1)]
        rows.append({'frame': frame, 'phase': phase, 'objects': objects, 'native_success': frame > 5,
            'native_terminated': frame > 5, 'native_truncated': False, 'native_fail': False,
            'consecutive_stable_placement_steps': max(0, frame-5)})
    assert outcome(rows, 25)['status'] == 'passed'
    changed = copy.deepcopy(rows)
    changed[5]['objects'][1]['tcp_pose'][2] = .179
    assert outcome(changed, 25)['status'] == 'failed'
    changed = copy.deepcopy(rows); changed[-1]['consecutive_stable_placement_steps'] = 19
    assert outcome(changed, 25)['status'] == 'failed'
    changed = copy.deepcopy(rows); changed[-1]['objects'][0]['placement_valid'] = False
    assert outcome(changed, 25)['status'] == 'failed'
    changed = copy.deepcopy(rows); changed[-1]['native_truncated'] = True
    assert outcome(changed, 25)['status'] == 'failed'
    assert within_limits({'pose_translation_m': .005, 'pose_rotation_rad': .05,
        'arm_joint_max_rad': .01, 'finger_max_m': .002}, ACTION_LIMITS)
    assert not within_limits({'pose_translation_m': .005001, 'pose_rotation_rad': .05,
        'arm_joint_max_rad': .01, 'finger_max_m': .002}, ACTION_LIMITS)
    assert ROOT == Path('/data/jxc/ld/ZeroWAM-for-TIG')
    assert len(TASK_IDS) == 3
    from .validate_curobo_complete import measured_source_departure
    def support() -> Any:
        return SimpleNamespace(support_released=False, _initial_support_exists=False,
            _target_names={'piece0'}, _support_state_receipt={
                'mode': 'measured_refresh', 'separation_numeric_band_m': 1e-6,
                'all_pieces_separated': True, 'fcl_touching_pieces': [],
                'pieces': [{'target': 'piece0', 'bottom_minus_table_top_m': 2e-6}]})
    refresh_calls = []
    checker = SimpleNamespace(attached=True, _supports=(support(), support()),
                              refresh=lambda: refresh_calls.append('measured_geometry_refresh'))
    assert measured_source_departure(checker) == (True, True)
    assert refresh_calls == ['measured_geometry_refresh']
    assert not any(item.support_released for item in checker._supports)
    departure_negatives = 0
    invalid_receipts = [None, {},
        {'mode': 'supplied_object_pose'}, {'mode': 'candidate'}]
    valid_receipt = copy.deepcopy(support()._support_state_receipt)
    invalid_receipts += [
        {**valid_receipt, 'mode': 'candidate'},
        {**valid_receipt, 'all_pieces_separated': False},
        {**valid_receipt, 'fcl_touching_pieces': ['piece0']},
        {**valid_receipt, 'pieces': []},
        {**valid_receipt, 'pieces': [{'target': 'other', 'bottom_minus_table_top_m': 2e-6}]},
        {**valid_receipt, 'pieces': [{'target': 'piece0', 'bottom_minus_table_top_m': 1e-6}]},
        {**valid_receipt, 'pieces': [{'target': 'piece0', 'bottom_minus_table_top_m': float('nan')}]},
        {**valid_receipt, 'separation_numeric_band_m': 0.}]
    for receipt in invalid_receipts:
        first, second = support(), support()
        first._support_state_receipt = receipt
        # A favorable candidate query cannot substitute for missing/invalid
        # actual measured evidence, even when its numbers look separated.
        first._last_support_validation = copy.deepcopy(valid_receipt)
        checker._supports = (first, second)
        assert measured_source_departure(checker) == (False, True)
        assert first.support_released is False
        departure_negatives += 1
    checker._supports = (support(), support())
    checker._supports[0]._initial_support_exists = True
    assert measured_source_departure(checker) == (False, True)
    checker._supports[0].support_released = True
    assert measured_source_departure(checker) == (True, True)
    checker.attached = False
    assert measured_source_departure(checker) == (False, False)
    from .multitask_recording import physical_support_transition
    assert not physical_support_transition(False, True, 0., .002999, False)
    assert not physical_support_transition(False, True, 0., .004, True)
    assert physical_support_transition(False, True, 0., .003, False)
    with tempfile.TemporaryDirectory(prefix='tig-curobo-complete-replay-cpu-') as temporary:
        directory = Path(temporary)
        (directory/'report.json').write_text(json.dumps({'status': 'failed'}))
        (directory/'trajectory.json').write_text(json.dumps({'episodes': []}))
        (directory/'ticks.jsonl').write_text('')
        reject(lambda: load_recording(directory))
        source, current = directory/'source', directory/'current'
        source.mkdir(); current.mkdir()
        hashes = {}
        for name in CRITICAL_SOURCES:
            payload = ('# '+name+'\n').encode()
            (source/name).write_bytes(payload); (current/name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        identity_report = {'source_sha256': hashes}
        assert source_identity(directory, identity_report, current_directory=current)['critical_execution_sources_match']
        changed_name = CRITICAL_SOURCES[0]
        (current/changed_name).write_text('# different current execution\n')
        reject(lambda: source_identity(directory, identity_report, current_directory=current))
        cross = source_identity(directory, identity_report, 'cross-version', current_directory=current)
        assert cross['source_archive_integrity'] == 'passed'
        assert cross['critical_source_differences'] == [changed_name]
        assert cross['source_differences'][changed_name]['current_sha256'] != hashes[changed_name]
        (source/changed_name).write_text('# damaged archive\n')
        for mode in REPLAY_MODES:
            reject(lambda mode=mode: source_identity(directory, identity_report, mode, current_directory=current))
        (source/changed_name).write_text('# '+changed_name+'\n')
        identity_report['runtime_policy'] = resolved_policy_record()
        for mode in REPLAY_MODES:
            reject(lambda mode=mode: source_identity(directory, identity_report, mode, current_directory=current))
        (current/changed_name).write_bytes((source/changed_name).read_bytes())
        payload = b'# archived runtime policy\n'
        (source/'runtime_policy.py').write_bytes(payload)
        (current/'runtime_policy.py').write_bytes(payload)
        hashes['runtime_policy.py'] = hashlib.sha256(payload).hexdigest()
        assert source_identity(directory, identity_report, current_directory=current)['critical_execution_sources_match']
        (current/'runtime_policy.py').write_text('# changed runtime policy implementation\n')
        reject(lambda: source_identity(directory, identity_report, current_directory=current))
        assert source_identity(directory, identity_report, 'cross-version', current_directory=current)[
            'critical_source_differences'] == ['runtime_policy.py']
        (source/'unlisted.py').write_text('# unauthenticated file\n')
        for mode in REPLAY_MODES:
            reject(lambda mode=mode: source_identity(directory, identity_report, mode, current_directory=current))
    assert policy_from_report({}) is LEGACY_STRICT
    assert policy_from_report({'runtime_policy': resolved_policy_record()}) == LEGACY_STRICT
    shorter = RuntimePolicy.from_dict(dict(LEGACY_STRICT.to_dict(), final_stable_steps=5))
    changed = copy.deepcopy(rows); changed[-1]['consecutive_stable_placement_steps'] = 5
    assert outcome(changed, 25, policy=shorter)['status'] == 'passed'
    assert outcome(changed, 25)['status'] == 'failed'
    farther = RuntimePolicy.from_dict(dict(LEGACY_STRICT.to_dict(), minimum_retreat_m=.09))
    assert outcome(rows, 25, policy=farther)['status'] == 'failed'
    longer_release = RuntimePolicy.from_dict(dict(LEGACY_STRICT.to_dict(), release_stable_steps=4))
    assert outcome(rows, 25, policy=longer_release)['status'] == 'failed'
    lift_rows = []
    for frame, (phase, heights) in enumerate((('initial', (0., 0.)),
            ('close_and_confirm', (.2, .3)), ('lift_and_hold', (.21, .31)),
            ('lift_and_hold', (.32, .42)), ('carry_lower', (.5, .7)))):
        lift_rows.append({'frame': frame, 'phase': phase, 'objects': [
            {'actor': name, 'pose': [0., 0., height, 1., 0., 0., 0.]}
            for name, height in zip(('a', 'b'), heights)]})
    lift = measure_replay_lift(lift_rows, ('a', 'b'))
    assert lift['status'] == 'observed' and lift['reference_frame'] == 1 and lift['end_frame'] == 3
    assert np.allclose(lift['gains_m'], [.12, .12], atol=1e-12, rtol=0)
    assert measure_replay_lift(lift_rows[:2], ('a', 'b'))['gains_m'] is None
    assert measure_replay_lift(lift_rows[2:], ('a', 'b'))['status'] == 'unknown'
    changed = copy.deepcopy(lift_rows); changed[3]['objects'][1]['actor'] = 'wrong'
    assert measure_replay_lift(changed, ('a', 'b'))['status'] == 'unknown'
    changed = copy.deepcopy(lift_rows); changed[1]['frame'] = 0
    assert measure_replay_lift(changed, ('a', 'b'))['status'] == 'unknown'
    return {'status': 'passed', 'action_shape': [2, 16], 'state_frames': 3,
        'single_initial_restore_verified': True, 'retreat_limit_m': .08,
        'final_stable_steps': 20, 'existing_action_limits': ACTION_LIMITS,
        'rejected_inputs': rejected, 'source_departure_negative_cases': departure_negatives,
        'strict_and_cross_version_archive_integrity_verified': True,
        'strict_runtime_policy_source_gate_verified': True,
        'source_resolved_release_retreat_final_limits_verified': True,
        'lift_measurement_uses_observed_phase_entry': True,
        'missing_lift_measurement_is_unknown': True,
        'measured_source_departure_provider_reused': True,
        'physical3mm_and_no_contact_gate_preserved': True, 'gpu_initialized': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--task', choices=tuple(ALIASES)+TASK_IDS)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--replay-mode', choices=REPLAY_MODES, default='strict',
                        help='strict requires source runtime identity; cross-version records source differences')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False)); return 0
    if args.input_dir is None or args.output_dir is None:
        parser.error('Provide an existing source --input-dir and unused --output-dir')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+', gpu) or any(os.environ.get(f'MANI_SKILL_L{i}') for i in (1, 2, 3)):
        parser.error('One explicit rendering GPU UUID and L0 are required')
    faulthandler.enable(); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from .validate_multitask import termination_handler
    signal.signal(signal.SIGTERM, termination_handler)
    recording = load_recording(args.input_dir, args.task, replay_mode=args.replay_mode)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    archive = output/'source'; archive.mkdir()
    sources = list(HERE.glob('*.py'))
    for source in sources:
        shutil.copy2(source, archive/source.name)
    input_files = ('trajectory.h5', 'trajectory.json', 'ticks.jsonl', 'physics-substeps.jsonl',
                   'all-robot-contacts.json', 'report.json', 'execution.mp4',
                   'task-geometry.json', 'task-geometry.npz')
    input_hashes = {name: hashlib.sha256((args.input_dir/name).read_bytes()).hexdigest() for name in input_files}
    input_hashes.update({'source/'+name: digest for name, digest in
                         recording['source_identity']['source_archive_sha256'].items()})
    report: dict[str, Any] = {'status': 'running', 'mode': 'action-replay', 'task': recording['task'],
        'seed': recording['report']['seed'], 'max_episode_steps': 200, 'gpu_uuid': gpu,
        'started_at': datetime.now(timezone.utc).isoformat(), 'input_dir': str(args.input_dir.resolve()),
        'input_sha256': input_hashes, 'input_receipt': recording['receipt'], 'limits': ACTION_LIMITS,
        'replay_mode': args.replay_mode, 'replay_source_identity': recording['source_identity'],
        'runtime_policy': resolved_policy_record(recording['runtime_policy']),
        'runtime_policy_origin': recording['receipt']['runtime_policy_origin'],
        'source_sha256': {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
        'replanning_performed': False, 'per_tick_state_correction': False,
        'hidden_contact_or_sleep_state_restored': False, 'controller_reset_after_restoration': False,
        'physical_attachments_created': False, 'continuous_collision_detection': False,
        'contact_policy': 'unchanged validate_curobo_payload.contact_allowed and multitask_recording',
        'source_video_timing': video_timing(args.input_dir, len(recording['actions']), recording['report'])}
    save_json(output/'report.json', report)
    started = time.monotonic()
    try:
        _require(report['source_video_timing']['status'] == 'passed', 'Source video T+1/padding timing disagrees')
        native_run(args, recording, output, report)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    report.update(wall_s=time.monotonic()-started, completed_at=datetime.now(timezone.utc).isoformat())
    report['input_evidence_unchanged'] = all(hashlib.sha256((args.input_dir/name).read_bytes()).hexdigest() == digest
                                             for name, digest in input_hashes.items())
    if not report['input_evidence_unchanged']:
        report['status'] = 'failed'
    report = json_value(report)
    update_run_results_from_file(report, output/'ticks.jsonl')
    save_json(output/'report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(output/'report.json')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
