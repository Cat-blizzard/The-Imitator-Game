"""Initial-state-only action replay of a complete paired TIG recording.

The original T paired actions [T,16] are submitted without replanning or state
correction. A reset wrapper restores source frame zero exactly once BEFORE
RecordEpisode captures its initial state; all later states arise from physics.
Existing physical contact policies, support release and held-slip audits remain
active. The serialized source policy selects quality stages. Diagnostic retains
the all-state reproducibility gate; normal keeps robot/target replay checks and
reports background errors independently. No historical report is re-scored.
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

from infrastructure.tig_runtime import resolve_runtime_paths

from .multitask_adapters import ALIASES, TASK_IDS
from .task_replay_alias import load_target_bindings, target_state, validate_live_bindings
from .replay_metrics import ACTION_KEYS, compare_states, pose_errors, summarize_errors
from .sync_replay_metrics import _array, _flag, _require, summarize_objects
from .replay_support import (ACTION_LIMITS, STATE_LIMITS, batch_state, replay_sequence,
                             replay_signal, unbatch_state, within_limits)
from .runtime_io import json_value, save_json, termination_handler
from .runtime_policy import LEGACY_STRICT, RuntimePolicy, policy_from_report, resolved_policy_record
from .run_results import update_run_results_from_file
from .recording_policy import (RecordingPolicy, archive_report, recording_policy_from_report,
                               write_robot_contacts)

HERE = Path(__file__).resolve().parent
RUNTIME_PATHS = resolve_runtime_paths()
CODE_ROOT = RUNTIME_PATHS.code_root
WORKSPACE_ROOT = RUNTIME_PATHS.workspace_root
CRITICAL_SOURCES = ('dual_arm_execution.py', 'dual_payload_collision.py',
                    'multitask_recording.py', 'multitask_adapters.py',
                    'validate_sync.py', 'validate_curobo_payload.py')
OPEN_PHASES = {'preparation', 'pregrasp', 'descend', 'release', 'retreat', 'verify_placement',
               'observe_task_outcome'}
SELECTED_COMPLETION_CONTRACT = 'selected-runtime-policy-v1'
STRATEGY_KEYS = ('runtime_strategy', 'runtime_profile')
REPLAY_MODES = ('strict', 'cross-version')
STAGE2_HELPERS = ('runtime_io.py', 'physical_observation.py', 'replay_support.py',
                  'phase_kinematics.py', 'task_targets.py', 'transport_recipes.py')
INFRASTRUCTURE_SOURCES = ('tig_runtime.py', 'curobo_tig_worker.py',
    'curobo_tig_payload_worker.py', 'curobo_tig_task_payload_worker.py',
    'curobo_tig_fast_payload_worker.py', 'curobo_tig_seed_policy_worker.py',
    'curobo_tig_session_worker.py')


def current_source_paths(current_directory: Path = HERE) -> dict[str, Path]:
    """Read this checkout's dual modules and infrastructure, never a saved archive.

    An explicit flat current_directory supports isolated CPU fixtures; its
    infrastructure lives underneath it. A real dual package uses its checkout's
    infrastructure, independently of the external dependency workspace.
    """
    current_directory = Path(current_directory).resolve()
    if current_directory == HERE:
        infrastructure = CODE_ROOT/'infrastructure'
    elif current_directory.parts[-4:] == ('mani_skill', 'examples', 'motionplanning', 'dual'):
        infrastructure = current_directory.parents[3]/'infrastructure'
    else:
        infrastructure = current_directory/'infrastructure'
    sources = {path.name: path for path in current_directory.glob('*.py') if path.is_file()}
    # Actual worker code belongs to infrastructure even if another same-named
    # file exists among the dual modules. A missing worker remains missing.
    for name in INFRASTRUCTURE_SOURCES:
        sources.pop(name, None)
        path = infrastructure/name
        if path.is_file():
            sources[name] = path
    return sources


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    json.dumps(rows, allow_nan=False)
    return rows


def source_completion_contract(report: dict[str, Any]) -> str:
    """Distinguish selected quality from the historical complete strict closure."""
    contract = report.get('completion_contract')
    _require(contract is None or contract == SELECTED_COMPLETION_CONTRACT, 'Unknown completion contract')
    _require(report.get('status') == 'passed' and report.get('mode') == 'positive'
             and report.get('official_200_step_complete') is True,
             'A passed official200 positive recording is required')
    if contract == SELECTED_COMPLETION_CONTRACT:
        _require(report.get('selected_policy_complete') is True,
                 'Source selected runtime policy is incomplete')
        _require(report.get('project_complete_physical_closure') is None,
                 'Selected contract cannot claim an unevaluated legacy closure')
        return contract
    _require(report.get('project_complete_physical_closure') is True,
             'Legacy source requires complete strict physical closure')
    return 'legacy-strict-closure'


def source_runtime_strategy(report: dict[str, Any], policy: RuntimePolicy
                            ) -> tuple[dict[str, Any] | None, str | None, tuple[str, ...]]:
    """Read canonical/old selection metadata without resolving current defaults.

    The serialized resolved policy remains authoritative. A present malformed
    key never falls back to its alias; old legacy sources may omit both keys.
    """
    _require(isinstance(policy, RuntimePolicy), 'A validated source runtime policy is required')
    present = tuple(key for key in STRATEGY_KEYS if key in report)
    if not present:
        _require(report.get('completion_contract') != SELECTED_COMPLETION_CONTRACT,
                 'Selected source lacks runtime strategy metadata')
        return None, None, present
    _require(policy.max_episode_steps == 200 and policy.last_completion_step == 199,
             'Strategy source must preserve the official200/last199 boundary')
    encoded = []
    for key in present:
        metadata = report[key]
        _require(isinstance(metadata, dict)
                 and metadata.get('schema') == 'tig-task-runtime-profile-v1'
                 and metadata.get('task_id') == report.get('task') in TASK_IDS
                 and metadata.get('profile_id') == policy.policy_id
                 and metadata.get('profile_id') in ('legacy-strict-v1','task-default-v1')
                 and metadata.get('policy_sha256') == resolved_policy_record(policy)['sha256'],
                 f'Source {key} metadata differs from serialized policy')
        overrides = metadata.get('overrides')
        _require(isinstance(overrides, dict) and all(name in policy.to_dict()
            and name not in ('policy_id','schema_version','max_episode_steps','last_completion_step')
            for name in overrides), f'Source {key} overrides are unknown or locked')
        _require(RuntimePolicy.from_dict(dict(policy.to_dict(),**overrides)) == policy,
                 f'Source {key} override values differ from serialized policy')
        if policy.policy_id == 'legacy-strict-v1':
            _require(policy == LEGACY_STRICT and not overrides,
                     'The frozen legacy strategy cannot contain changed values or overrides')
        encoded.append(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False))
    _require(len(set(encoded)) == 1, 'Canonical runtime_strategy and runtime_profile alias conflict')
    source_key = present[0]
    return copy.deepcopy(report[source_key]), source_key, present


def input_artifact_names(report: dict[str, Any]) -> tuple[str, ...]:
    """A missing storage marker retains the full legacy diagnostic contract."""
    policy = recording_policy_from_report(report)
    common = ('trajectory.h5', 'trajectory.json', 'ticks.jsonl', 'report.json',
              'task-geometry.json', 'task-geometry.npz')
    if policy.diagnostic:
        return common + ('physics-substeps.jsonl', 'all-robot-contacts.json', 'execution.mp4')
    return common + ('physics-events.jsonl', 'physics-summary.json')


def replay_state_validation(errors: list[dict[str, Any]], bindings: Any,
                            storage: RecordingPolicy) -> dict[str, Any]:
    """Separate background reproducibility from task replay, preserving both."""
    required = {'articulations/panda_wristcam-agent-0','articulations/panda_wristcam-agent-1'}
    _require(len(bindings) == 2, 'Replay state validation requires two explicit target bindings')
    required.update('actors/'+binding.state_name for binding in bindings)
    _require(len(required) == 4 and bool(errors), 'Missing paired robot/target replay measurements')
    task,background=[],[]
    for sample in errors:
        entities=sample.get('per_entity',{})
        _require(required <= set(entities), 'Replay error sample lacks an explicit robot or target')
        task.append({'per_entity':{key:value for key,value in entities.items() if key in required}})
        background.append({'per_entity':{key:value for key,value in entities.items() if key not in required}})
    full=summarize_errors(errors);selected=summarize_errors(task);other=summarize_errors(background)
    full_ok=within_limits(full['global'],ACTION_LIMITS)
    task_ok=within_limits(selected['global'],ACTION_LIMITS)
    return {'profile':'legacy-all-state-v1' if storage.diagnostic else 'task-outcome-v1',
        'required_checks':({'all_state_errors_within_existing_action_limits':full_ok} if storage.diagnostic
            else {'robot_and_target_state_errors_within_existing_action_limits':task_ok}),
        'task_errors':selected,'background_errors':other,
        'diagnostics':{'all_state_errors_within_existing_action_limits':full_ok,
            'background_errors_within_existing_action_limits':within_limits(other['global'],ACTION_LIMITS),
            'background_equality_required_for_acceptance':storage.diagnostic,
            'actual_background_collision_monitoring':'unchanged full100Hz robot/world and object/interarm channels'}}


def validate_compact_source(directory: Path, report: dict[str, Any], count: int,
                            segments: list[str]) -> dict[str, Any]:
    """Validate declared compact observations without inventing raw substeps."""
    summary = json.loads((directory/'physics-summary.json').read_text())
    events = _read_rows(directory/'physics-events.jsonl')
    _require(isinstance(summary, dict) and summary == report.get('compact_physical_summary'),
             'Compact physical summary and report differ')
    _require(summary.get('schema') == 'tig-compact-physical-observations-v1',
             'Unknown compact observation schema')
    monitor = report.get('physical_monitoring')
    _require(isinstance(monitor, dict) and summary.get('physical_monitoring') == monitor,
             'Compact physical monitoring identity differs')
    expected = count*5
    for key in ('expected_substeps', 'observed_substeps', 'robot_world_observed_substeps'):
        _require(type(monitor.get(key)) is int and monitor[key] == expected,
                 'Compact physical monitoring count differs: '+key)
    _require(monitor.get('coverage_complete') is True and monitor.get('robot_world_coverage_complete') is True,
             'Compact source lacks complete object/interarm and robot/world coverage')
    for key in ('forbidden_contact_count', 'interarm_forbidden_count', 'robot_world_forbidden_count'):
        _require(type(monitor.get(key)) is int and monitor[key] == 0,
                 'Positive compact source has forbidden contact evidence: '+key)
    _require(summary.get('actual_substeps') == summary.get('actual_robot_world_substeps') == expected
             and summary.get('expected_substeps_per_control') == 5
             and summary.get('control_substep_counts') == {str(i):5 for i in range(1,count+1)}
             and summary.get('partial_control_steps') == []
             and summary.get('observed_control_steps_without_completed_tick') == [],
             'Compact per-control physical observation coverage differs')
    _require(type(summary.get('event_count')) is int and summary['event_count'] == len(events)
             and all(isinstance(event, dict) and isinstance(event.get('kind'), str) for event in events),
             'Compact event inventory differs')
    _require([event.get('phase') for event in events if event['kind'] == 'phase_observed'] == segments,
             'Compact observed phases differ from action phases')
    _require(not any(event['kind'] in ('forbidden_contact', 'held_grasp_missing_observed',
                    'native_time_limit_observed', 'capture_exception', 'run_exception') for event in events),
             'Positive compact source contains physical/time/exception incidents')
    selection = report.get('physical_archive_selection', {})
    _require(selection.get('physics-events.jsonl') == selection.get('physics-summary.json') == 'written'
             and selection.get('physics-substeps.jsonl') == selection.get('all-robot-contacts.json') == 'not_selected',
             'Compact artifact selection receipt differs')
    _require(report.get('video', {}).get('status') == 'not_selected', 'Normal source video selection differs')
    return {'status':'passed', 'actual_substeps':expected, 'event_count':len(events),
            'raw_substeps_required':False, 'evidence':'declared compact observations, counters and events'}


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
    stage2 = bool(set(recorded) & set(STAGE2_HELPERS))
    if stage2:
        critical += STAGE2_HELPERS + INFRASTRUCTURE_SOURCES
    stage4 = ('runtime_profiles.py' in recorded or any(key in report for key in STRATEGY_KEYS) or 'recording_policy' in report
              or report.get('completion_contract') == SELECTED_COMPLETION_CONTRACT)
    if stage4:
        critical += ('runtime_profiles.py', 'recording_policy.py', 'task_outcome.py', 'run_results.py')
    _require(set(critical) <= set(recorded), 'Source lacks critical execution identities')
    paths = current_source_paths(current_directory)
    archive = (directory/'source').resolve()
    _require(not Path(current_directory).resolve().is_relative_to(archive)
             and all(not path.resolve().is_relative_to(archive) for path in paths.values()),
             'Current execution sources cannot be the recorded source archive')
    current_names = set(recorded) | set(paths)
    current = {name: hashlib.sha256(paths[name].read_bytes()).hexdigest() if name in paths else None
               for name in sorted(current_names)}
    differences = {name: {'source_sha256': recorded.get(name), 'current_sha256': current[name]}
                   for name in sorted(current_names) if recorded.get(name) != current[name]}
    critical_differences = [name for name in critical if name in differences]
    if replay_mode == 'strict':
        _require(not critical_differences,
                 'Current execution/contact dependency differs from source: '+', '.join(critical_differences))
    return {'replay_mode': replay_mode, 'source_archive_integrity': 'passed',
            'source_archive_sha256': recorded, 'current_source_sha256': current,
            'current_source_paths': {name: str(paths[name]) if name in paths else None for name in sorted(current_names)},
            'source_differences': differences, 'critical_sources': list(critical),
            'source_runtime_generation': 'stage4' if stage4 else 'stage2' if stage2 else 'legacy',
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
    contract = source_completion_contract(report)
    strategy, strategy_key, strategy_keys = source_runtime_strategy(report,policy)
    if contract == SELECTED_COMPLETION_CONTRACT or strategy is not None:
        _require(policy.max_episode_steps == 200 and policy.last_completion_step == 199,
                 'Profile source must preserve the official200/last199 boundary')
    storage = recording_policy_from_report(report)
    artifacts = input_artifact_names(report)
    _require(all((directory/name).is_file() for name in artifacts), 'Source lacks declared required artifacts')
    metadata = json.loads((directory/'trajectory.json').read_text())
    json.dumps([report, metadata], allow_nan=False)
    ticks = _read_rows(directory/'ticks.jsonl')
    episodes, info = metadata.get('episodes', []), metadata.get('env_info', {})
    _require(len(episodes) == 1, 'Exactly one source episode is required')
    episode, kwargs = episodes[0], info.get('env_kwargs', {})
    canonical = ALIASES.get(task, task) if task is not None else info.get('env_id')
    _require(canonical in TASK_IDS and report.get('task') == info.get('env_id') == canonical,
             'Source and selected task identities disagree')
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
    if contract == SELECTED_COMPLETION_CONTRACT:
        tail = ['release'] + (['retreat'] if policy.require_retreat else [])
        valid_tails = ([tail+['verify_placement']] if policy.require_final_stability
                       else [tail,tail+['observe_task_outcome']])
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
    if 'phase_step_counts' in report:
        _require(report['phase_step_counts'] == dict(counts), 'Source phase count receipt differs')
    _require([stage.get('name') for stage in stages] == segments[1:] and all(
        stage.get('status') == 'passed' and stage.get('environment_steps') == counts[stage['name']]
        for stage in stages), 'Source stage receipts disagree')
    identity = source_identity(directory, report, replay_mode)
    geometry = load_geometry(directory, report)
    compact = None if storage.diagnostic else validate_compact_source(directory, report, count, segments)
    return {'actions': actions, 'states': states, 'ticks': ticks, 'report': report, 'metadata': metadata,
            'native_flags': flags, 'action_keys': list(ACTION_KEYS), 'target_names': target_names,
            'task': canonical, 'source_sha256': report['source_sha256'], 'target_bindings': target_bindings,
            'runtime_policy': policy, 'geometry': geometry, 'source_identity': identity,
            'recording_policy': storage, 'completion_contract': contract,
            'runtime_strategy':strategy, 'runtime_strategy_source_key':strategy_key,
            'runtime_strategy_source_keys':strategy_keys,
            'receipt': {'action_steps': count, 'state_frames': count+1, 'phase_counts': dict(counts),
                        'task': canonical, 'replay_mode': replay_mode,
                        'critical_execution_sources_match': identity['critical_execution_sources_match'],
                        'source_archive_integrity': 'passed', 'task_geometry_integrity': 'passed',
                        'runtime_policy_origin': 'source report' if 'runtime_policy' in report else 'legacy fallback',
                        'completion_contract':contract, 'recording_mode':storage.mode,
                        'runtime_strategy_source_key':strategy_key,
                        'runtime_strategy_source_keys':list(strategy_keys),
                        'required_input_artifacts':list(artifacts), 'compact_source':compact,
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
            policy: RuntimePolicy = LEGACY_STRICT,
            completion_contract: str | None = None) -> dict[str, Any]:
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
    def consecutive(values: list[dict[str, Any]], predicate: Any) -> int:
        count = 0
        for row in values:
            count = count+1 if predicate(row) else 0
        return count
    release_count = consecutive(release, lambda row: all(
        obj['fingers_open'] and not obj['is_grasping'] for obj in row['objects']))
    final_observed = consecutive(ticks, lambda row: row['phase'] == 'verify_placement'
        and row['native_success'] and all(obj['placement_valid'] for obj in row['objects']))
    final_observed = min(final_observed, ticks[-1]['consecutive_stable_placement_steps']) if ticks else 0
    selected = completion_contract == SELECTED_COMPLETION_CONTRACT
    final_outcome = bool(ticks) and ticks[-1]['native_success'] and all(
        obj['fingers_open'] and not obj['is_grasping'] and obj.get('physical_region', {}).get('inside') is True
        and obj.get('official_region') is True for obj in ticks[-1]['objects'])
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
    if selected:
        checks.update(both_physically_released=release_count >= policy.release_stable_steps,
            selected_retreat_observed=None if not policy.require_retreat else (
                all(value >= policy.minimum_retreat_m for value in rises)
                and bool(release and retreat and release[-1]['frame'] < retreat[0]['frame'])),
            selected_final_stability_observed=None if not policy.require_final_stability else (
                final_observed >= policy.final_stable_steps
                and all(row['phase'] == 'verify_placement' for row in final)
                and all(obj['placement_valid'] for row in final for obj in row['objects'])),
            final_task_outcome_observed=(checks['final_official_success'] if policy.require_final_stability else final_outcome))
        # Historical names are only measurements of their literal old windows.
        # Unselected quality is not a passing strict closure observation.
        if policy.release_stable_steps != 3:
            checks['both_physically_released_last_three'] = None
        if not policy.require_retreat or policy.minimum_retreat_m != .08:
            checks['both_actual_retreats_at_least80mm'] = None
        if not policy.require_retreat:
            checks['release_retreat_order'] = None
        if not policy.require_final_stability or policy.final_stable_steps != 20:
            checks['common_final20_window'] = None
        if not policy.require_final_stability:
            checks['both_final_placements_valid'] = None
    measurements = {
        'release':{'consecutive_observed':release_count, 'required':policy.release_stable_steps,
                   'physically_released':release_count >= policy.release_stable_steps},
        'retreat':{'selected':policy.require_retreat,
                   'gains_m':rises if policy.require_retreat else None,
                   'minimum_m':policy.minimum_retreat_m},
        'final_stability':{'selected':policy.require_final_stability,
            'consecutive_observed':final_observed if policy.require_final_stability else None,
            'required':policy.final_stable_steps,
            'placement_valid':all(obj['placement_valid'] for row in final for obj in row['objects'])
                if policy.require_final_stability and len(final) == final_count else None}}
    return {'status': 'passed' if all(value for value in checks.values() if value is not None) else 'failed',
            'checks': checks, 'measurements':measurements,
            'completion_contract':completion_contract,
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
    if not recording_policy_from_report(report).diagnostic:
        _require(report.get('video', {}).get('status') == 'not_selected', 'Normal video omission is undeclared')
        return {'status':'not_selected', 'reason':'recording_mode_normal'}
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
    from .physical_observation import (held_substep_audit, install_full_robot_contact_capture,
                                       make_checker, measured_source_departure)
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
    storage = RecordingPolicy(getattr(args, 'recording_mode', 'diagnostic'))
    contract = recording.get('completion_contract', 'legacy-strict-closure')
    report['recording_policy'] = storage.record()
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
                hard_step_budget=policy.max_episode_steps, recording_policy=storage)
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
                if (phase == 'verify_placement' or phase == 'observe_task_outcome'
                        or not policy.require_final_stability and phase in ('release','retreat')):
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
                        # On drift the native verifier starts a new window at
                        # this basic-valid sample (count=1), with a new anchor.
                        obj['placement_valid'] = True
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
        state_validation = replay_state_validation(errors,recording['target_bindings'],storage)
        report['replay_validation_profile'] = state_validation['profile']
        report['task_trajectory_errors'] = state_validation['task_errors']
        report['background_trajectory_errors'] = state_validation['background_errors']
        report['replay_diagnostics'] = state_validation['diagnostics']
        objects = summarize_objects(rows)
        for obj, name in zip(objects['objects'], adapter.target_names):
            obj['actor'] = name  # Existing numerical reducer has legacy apple/banana labels only.
        report['object_errors'] = objects
        report['outcome'] = outcome(rows, len(recording['actions']), policy=policy,
            completion_contract=contract)
        report['held_physics_audit'] = rename_held_audit_targets(
            held_substep_audit(substeps, references, policy=policy), adapter.target_names)
        checks = {**state_validation['required_checks'],
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
        if contract == SELECTED_COMPLETION_CONTRACT:
            report['completion_contract'] = contract
            report['selected_policy_complete'] = all(checks.values())
            report['project_complete_physical_closure'] = None
    finally:
        report['replay_lift_measurement'] = measure_replay_lift(rows, recording['target_names'])
        write_robot_contacts(output, contacts, policy=storage)
        if storage.diagnostic:
            with (output/'frames.jsonl').open('w') as handle:
                for row in rows:
                    handle.write(json.dumps(json_value(row), allow_nan=False)+'\n')
        if env is not None:
            finalize_physics_capture(env, raw, capture, output, report, frames, ticks, substeps,
                                     keys, original_hook, robot_contacts=contacts, recording_policy=storage)
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
            if (report['standard_recording_timing']['status'] != 'passed'
                    or storage.diagnostic and report['video_timing']['status'] != 'passed'):
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
    assert CODE_ROOT.is_dir() and (HERE/Path(__file__).name).is_file()
    assert RUNTIME_PATHS.workspace_root == WORKSPACE_ROOT
    assert len(TASK_IDS) == 3
    from .physical_observation import measured_source_departure
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
        (source/'unlisted.py').unlink()
        (current/'runtime_policy.py').write_bytes((source/'runtime_policy.py').read_bytes())
        first_helper = STAGE2_HELPERS[0]
        payload = ('# '+first_helper+'\n').encode()
        (source/first_helper).write_bytes(payload); (current/first_helper).write_bytes(payload)
        hashes[first_helper] = hashlib.sha256(payload).hexdigest()
        for mode in REPLAY_MODES:
            reject(lambda mode=mode: source_identity(directory, identity_report, mode, current_directory=current))
        infrastructure = current/'infrastructure'; infrastructure.mkdir()
        for name in STAGE2_HELPERS+INFRASTRUCTURE_SOURCES:
            payload = ('# '+name+'\n').encode()
            (source/name).write_bytes(payload)
            location = infrastructure/name if name in INFRASTRUCTURE_SOURCES else current/name
            location.write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        modern = source_identity(directory, identity_report, current_directory=current)
        assert modern['source_runtime_generation'] == 'stage2'
        assert set(STAGE2_HELPERS+INFRASTRUCTURE_SOURCES) <= set(modern['critical_sources'])
        # A same-named dual file cannot stand in for actual infrastructure code.
        worker = 'curobo_tig_task_payload_worker.py'
        (current/worker).write_text('# not the worker being executed\n')
        assert source_identity(directory, identity_report, current_directory=current)['critical_execution_sources_match']
        for name in ('tig_runtime.py', 'physical_observation.py', worker):
            location = infrastructure/name if name in INFRASTRUCTURE_SOURCES else current/name
            original = location.read_bytes()
            location.write_text('# changed current runtime dependency\n')
            reject(lambda: source_identity(directory, identity_report, current_directory=current))
            changed = source_identity(directory, identity_report, 'cross-version', current_directory=current)
            assert changed['critical_source_differences'] == [name]
            assert changed['current_source_paths'][name] == str(location)
            location.write_bytes(original)
            original = (source/name).read_bytes()
            (source/name).write_text('# changed archived runtime dependency\n')
            for mode in REPLAY_MODES:
                reject(lambda mode=mode: source_identity(directory, identity_report, mode, current_directory=current))
            (source/name).write_bytes(original)
        reject(lambda: source_identity(directory, identity_report, current_directory=source))
        absent_worker = copy.deepcopy(identity_report)
        absent_worker['source_sha256'].pop(worker)
        original = (source/worker).read_bytes(); (source/worker).unlink()
        for mode in REPLAY_MODES:
            reject(lambda mode=mode: source_identity(directory, absent_worker, mode, current_directory=current))
        (source/worker).write_bytes(original)
        isolated = resolve_runtime_paths(code_root=current,
            environ={'TIG_WORKSPACE': str(directory/'dependencies')})
        assert isolated.code_root == current and isolated.workspace_root == directory/'dependencies'
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
        'complete_stage2_runtime_and_worker_sources_required': True,
        'runtime_helper_worker_current_changes_rejected_only_in_strict': True,
        'archived_runtime_helper_worker_changes_rejected_in_both_modes': True,
        'recorded_archive_cannot_substitute_for_current_sources': True,
        'source_resolved_release_retreat_final_limits_verified': True,
        'lift_measurement_uses_observed_phase_entry': True,
        'missing_lift_measurement_is_unknown': True,
        'measured_source_departure_provider_reused': True,
        'physical3mm_and_no_contact_gate_preserved': True, 'gpu_initialized': False,
        'stage4': stage4_self_test()}


def stage4_self_test() -> dict[str, Any]:
    """Synthetic contract/storage/selected-observation checks, without physics."""
    rejected = []
    def reject(name: str, call: Any) -> None:
        try: call()
        except (ValueError, RuntimeError, AssertionError, KeyError): rejected.append(name)
        else: raise AssertionError('Expected rejection: '+name)
    legacy = dict(status='passed',mode='positive',official_200_step_complete=True,
                  project_complete_physical_closure=True)
    assert source_completion_contract(legacy) == 'legacy-strict-closure'
    reject('legacy_incomplete', lambda:source_completion_contract(dict(legacy,project_complete_physical_closure=None)))
    selected = dict(legacy,completion_contract=SELECTED_COMPLETION_CONTRACT,
                    selected_policy_complete=True,project_complete_physical_closure=None)
    assert source_completion_contract(selected) == SELECTED_COMPLETION_CONTRACT
    for name, changes in [('selected_incomplete',{'selected_policy_complete':None}),
        ('invented_legacy_closure',{'project_complete_physical_closure':True}),
        ('unknown_contract',{'completion_contract':'unknown'})]:
        reject(name,lambda changes=changes:source_completion_contract(dict(selected,**changes)))
    policy = RuntimePolicy.from_dict(dict(LEGACY_STRICT.to_dict(),policy_id='task-default-v1',
        release_stable_steps=4, final_stable_steps=5, minimum_settle_budget=0,
        require_retreat=False,require_final_stability=False))
    strategy={'schema':'tig-task-runtime-profile-v1','task_id':TASK_IDS[1],
        'profile_id':policy.policy_id,'overrides':{'release_stable_steps':4,'final_stable_steps':5,
            'require_retreat':False,'require_final_stability':False},
        'policy_sha256':resolved_policy_record(policy)['sha256']}
    source=dict(selected,task=TASK_IDS[1],runtime_strategy=strategy)
    canonical,key,keys=source_runtime_strategy(source,policy)
    assert canonical==strategy and key=='runtime_strategy' and keys==('runtime_strategy',)
    canonical['overrides']['release_stable_steps']=99
    assert strategy['overrides']['release_stable_steps']==4
    alias=dict(source);alias['runtime_profile']=alias.pop('runtime_strategy')
    assert source_runtime_strategy(alias,policy)[1:]==('runtime_profile',('runtime_profile',))
    both=dict(source,runtime_profile=copy.deepcopy(strategy))
    assert source_runtime_strategy(both,policy)[1:]==('runtime_strategy',STRATEGY_KEYS)
    assert source_runtime_strategy(legacy,LEGACY_STRICT)==(None,None,())
    no_metadata=dict(source);no_metadata.pop('runtime_strategy')
    reject('selected_missing_strategy',lambda:source_runtime_strategy(no_metadata,policy))
    for target in STRATEGY_KEYS:
        for value in (None,{},[],False):
            damaged=dict(both,**{target:value})
            reject('explicit_'+target+'_'+repr(value),lambda damaged=damaged:source_runtime_strategy(damaged,policy))
    for field,value in (('schema','unknown'),('task_id',TASK_IDS[0]),('profile_id','unknown'),
                         ('policy_sha256','0'*64),('overrides',None),
                         ('overrides',{'max_episode_steps':200}),
                         ('overrides',{'release_stable_steps':True}),
                         ('overrides',{'release_stable_steps':5})):
        damaged=copy.deepcopy(source);damaged['runtime_strategy'][field]=value
        reject('invalid_strategy_'+field+'_'+repr(value),lambda damaged=damaged:source_runtime_strategy(damaged,policy))
    conflicted=copy.deepcopy(both);conflicted['runtime_profile']['selection_note']='different'
    reject('alias_metadata_conflict',lambda:source_runtime_strategy(conflicted,policy))
    unknown_policy=RuntimePolicy.from_dict(dict(policy.to_dict(),policy_id='unknown'))
    unknown=copy.deepcopy(source);unknown['runtime_strategy'].update(profile_id='unknown',
        policy_sha256=resolved_policy_record(unknown_policy)['sha256'])
    reject('unknown_strategy_id',lambda:source_runtime_strategy(unknown,unknown_policy))
    shortened=RuntimePolicy.from_dict(dict(policy.to_dict(),last_completion_step=198))
    shortened_source=copy.deepcopy(source);shortened_source['runtime_strategy']['policy_sha256']=resolved_policy_record(shortened)['sha256']
    reject('strategy_last199_boundary',lambda:source_runtime_strategy(shortened_source,shortened))
    def samples(phases: list[str]) -> list[dict[str, Any]]:
        rows=[]
        for frame,phase in enumerate(['initial']+phases):
            objects=[{'tcp_pose':[0.,0.,.1,1.,0.,0.,0.],'grasp_matches_reference':True,
                'fingers_open':True,'is_grasping':False,'placement_valid':phase=='verify_placement',
                'physical_region':{'inside':True},'official_region':True} for _ in (0,1)]
            success=frame==len(phases) or phase=='verify_placement'
            rows.append({'frame':frame,'phase':phase,'objects':objects,
                'native_success':success,'native_terminated':success,
                'native_truncated':False,'native_fail':False,
                'consecutive_stable_placement_steps':sum(p=='verify_placement' for p in phases[:frame])})
        return rows
    rows=samples(['release']*4+['observe_task_outcome']*2)
    value=outcome(rows,6,policy=policy,completion_contract=SELECTED_COMPLETION_CONTRACT)
    assert value['status']=='passed' and value['measurements']['release']['consecutive_observed']==4
    assert value['checks']['both_actual_retreats_at_least80mm'] is None
    assert value['checks']['common_final20_window'] is None
    assert value['measurements']['final_stability']['consecutive_observed'] is None
    assert outcome(samples(['release']*4),4,policy=policy,
                   completion_contract=SELECTED_COMPLETION_CONTRACT)['status']=='passed'
    for key,new_value in [('is_grasping',True),('fingers_open',False),
                          ('physical_region',{'inside':False}),('official_region',False)]:
        altered=copy.deepcopy(rows);altered[-1]['objects'][0][key]=new_value
        assert outcome(altered,6,policy=policy,completion_contract=SELECTED_COMPLETION_CONTRACT)['status']=='failed'
    with_final=RuntimePolicy.from_dict(dict(policy.to_dict(),require_final_stability=True))
    final=outcome(samples(['release']*4+['verify_placement']*5),9,policy=with_final,
        completion_contract=SELECTED_COMPLETION_CONTRACT)
    assert final['status']=='passed' and final['measurements']['final_stability']['consecutive_observed']==5
    assert final['checks']['common_final20_window'] is None
    assert outcome(samples(['release']*4+['verify_placement']*4),8,policy=with_final,
        completion_contract=SELECTED_COMPLETION_CONTRACT)['status']=='failed'
    normal=dict(recording_policy=RecordingPolicy('normal').record(),video={'status':'not_selected'})
    assert 'execution.mp4' not in input_artifact_names(normal)
    assert 'physics-summary.json' in input_artifact_names(normal)
    assert 'execution.mp4' in input_artifact_names({})
    assert video_timing(Path('/unopened-normal-video'),1,normal)['status']=='not_selected'
    bindings=[SimpleNamespace(state_name='target0'),SimpleNamespace(state_name='target1')]
    entities={key:{'pose_translation_m':0.,'pose_rotation_rad':0.} for key in
        ('actors/target0','actors/target1','articulations/panda_wristcam-agent-0',
         'articulations/panda_wristcam-agent-1','actors/background')}
    entities['actors/background']['pose_translation_m']=.1
    differences=[{'per_entity':entities}]
    task_only=replay_state_validation(differences,bindings,RecordingPolicy('normal'))
    assert all(task_only['required_checks'].values())
    assert not task_only['diagnostics']['all_state_errors_within_existing_action_limits']
    assert task_only['background_errors']['per_entity']['actors/background']['pose_translation_m']==.1
    assert not all(replay_state_validation(differences,bindings,RecordingPolicy('diagnostic'))['required_checks'].values())
    for key in ('actors/target0','articulations/panda_wristcam-agent-0'):
        changed=copy.deepcopy(differences);changed[0]['per_entity'][key]['pose_translation_m']=.006
        assert not all(replay_state_validation(changed,bindings,RecordingPolicy('normal'))['required_checks'].values())
    broken=copy.deepcopy(normal);broken['recording_policy']['sha256']='0'*64
    reject('recording_policy_hash',lambda:input_artifact_names(broken))
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary);source=directory/'source';source.mkdir()
        current=directory/'current';current.mkdir();infra=current/'infrastructure';infra.mkdir()
        names=set(CRITICAL_SOURCES+STAGE2_HELPERS+INFRASTRUCTURE_SOURCES+
                  ('runtime_profiles.py','recording_policy.py','task_outcome.py','run_results.py'))
        hashes={}
        for name in names:
            payload=('# synthetic identity fixture '+name+'\n').encode()
            (source/name).write_bytes(payload)
            path=(infra if name in INFRASTRUCTURE_SOURCES else current)/name;path.write_bytes(payload)
            hashes[name]=hashlib.sha256(payload).hexdigest()
        identity_report={'source_sha256':hashes,'completion_contract':SELECTED_COMPLETION_CONTRACT}
        assert source_identity(directory,identity_report,current_directory=current)['critical_execution_sources_match']
        for name in ('runtime_profiles.py','recording_policy.py','task_outcome.py','run_results.py'):
            path=current/name;original=path.read_bytes();path.write_text('# current changed\n')
            reject('strict_current_'+name,lambda:source_identity(directory,identity_report,current_directory=current))
            assert source_identity(directory,identity_report,'cross-version',current_directory=current)[
                'critical_source_differences']==[name]
            path.write_bytes(original);(source/name).write_text('# archive changed\n')
            for mode in REPLAY_MODES:
                reject('archive_'+mode+'_'+name,lambda mode=mode:source_identity(
                    directory,identity_report,mode,current_directory=current))
            (source/name).write_bytes(original)
        monitor={'observed_substeps':5,'expected_substeps':5,'robot_world_observed_substeps':5,
            'coverage_complete':True,'robot_world_coverage_complete':True,
            'forbidden_contact_count':0,'interarm_forbidden_count':0,'robot_world_forbidden_count':0}
        summary={'schema':'tig-compact-physical-observations-v1','physical_monitoring':monitor,
            'actual_substeps':5,'actual_robot_world_substeps':5,'expected_substeps_per_control':5,
            'control_substep_counts':{'1':5},'partial_control_steps':[],
            'observed_control_steps_without_completed_tick':[],'event_count':1}
        compact=dict(normal,compact_physical_summary=summary,physical_monitoring=monitor,
            physical_archive_selection={'physics-events.jsonl':'written','physics-summary.json':'written',
                'physics-substeps.jsonl':'not_selected','all-robot-contacts.json':'not_selected'})
        save_json(directory/'physics-summary.json',summary)
        (directory/'physics-events.jsonl').write_text(json.dumps({'kind':'phase_observed','phase':'preparation'})+'\n')
        assert validate_compact_source(directory,compact,1,['preparation'])['status']=='passed'
        changed=copy.deepcopy(compact);changed['physical_monitoring']['coverage_complete']=False
        reject('incomplete_compact_coverage',lambda:validate_compact_source(directory,changed,1,['preparation']))
        (directory/'physics-events.jsonl').write_text('')
        reject('compact_event_inventory',lambda:validate_compact_source(directory,compact,1,['preparation']))
    return {'status':'passed','rejected_inputs':rejected,'source_policy_re_resolution':False,
            'optional_zero_and_positive_outcome_ticks':True,'nonlegacy_release4_final5':True,
            'unselected_legacy_gates_remain_unknown':True,'normal_video_not_opened':True,
            'normal_robot_target_gate_and_full_background_diagnostics':True,
            'canonical_strategy_and_old_alias_with_provenance':True,
            'explicit_invalid_metadata_never_falls_back':True,
            'physical_environment_steps':0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--task', choices=tuple(ALIASES)+TASK_IDS)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--replay-mode', choices=REPLAY_MODES, default='strict',
                        help='strict requires source runtime identity; cross-version records source differences')
    parser.add_argument('--recording-mode', choices=('diagnostic','normal'), default='diagnostic',
                        help='output artifact selection; source mode is read from its recorded policy')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False)); return 0
    if args.input_dir is None or args.output_dir is None:
        parser.error('Provide an existing source --input-dir and unused --output-dir')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+', gpu) or any(os.environ.get(f'MANI_SKILL_L{i}') for i in (1, 2, 3)):
        parser.error('One explicit rendering GPU UUID and L0 are required')
    faulthandler.enable(); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGTERM, termination_handler)
    recording = load_recording(args.input_dir, args.task, replay_mode=args.replay_mode)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    archive = output/'source'; archive.mkdir()
    paths = current_source_paths()
    _require(set(STAGE2_HELPERS+INFRASTRUCTURE_SOURCES) <= set(paths),
             'Current checkout lacks complete runtime helpers or infrastructure workers')
    sources = list(paths.values())
    for source in sources:
        shutil.copy2(source, archive/source.name)
    input_files = input_artifact_names(recording['report'])
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
        'recording_policy': RecordingPolicy(args.recording_mode).record(),
        'runtime_paths': RUNTIME_PATHS.as_dict(),
        'source_sha256': {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
        'replanning_performed': False, 'per_tick_state_correction': False,
        'hidden_contact_or_sleep_state_restored': False, 'controller_reset_after_restoration': False,
        'physical_attachments_created': False, 'continuous_collision_detection': False,
        'contact_policy': 'unchanged physical_observation.contact_allowed and multitask_recording',
        'source_video_timing': video_timing(args.input_dir, len(recording['actions']), recording['report'])}
    if recording['completion_contract'] == SELECTED_COMPLETION_CONTRACT:
        report.update(completion_contract=SELECTED_COMPLETION_CONTRACT,
            selected_policy_complete=None, project_complete_physical_closure=None)
    report['runtime_strategy_source_key'] = recording['runtime_strategy_source_key']
    report['runtime_strategy_source_keys'] = list(recording['runtime_strategy_source_keys'])
    if recording['runtime_strategy'] is not None:
        report['runtime_strategy'] = copy.deepcopy(recording['runtime_strategy'])
    save_json(output/'report.json', archive_report(report, RecordingPolicy(args.recording_mode)))
    started = time.monotonic()
    try:
        _require(report['source_video_timing']['status'] in ('passed','not_selected'),
                 'Source video T+1/padding timing disagrees')
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
    save_json(output/'report.json', archive_report(report, RecordingPolicy(args.recording_mode)))
    print(json.dumps({'status': report['status'], 'report': str(output/'report.json')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
