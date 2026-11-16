"""Read-only paired backend accounting from frozen cases and actual recordings.

Counts every allocated case, including planning failures. Initial-state pairing
compares recorded actor/articulation arrays and exported task geometry arrays.
Native contact observations, complete physical closure and recording integrity
are separate from the task's official success predicate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


INITIAL_COMPARISON_SCOPE = {
    'included': 'H5 frame-zero actor/articulation arrays and every task-geometry.npz array',
    'task_geometry_export': 'task-role poses and get_first_collision_mesh vertices',
    'not_established': 'complete static scene geometry equality; ground, physics::wall and unexported collision shapes',
}
TIMING_FIELDS = ('initialization_s', 'warmup_s', 'state_update_s',
                 'native_solve_s', 'execution_stage_wall_s')
INITIAL_SCENE_SCHEMA = 'tig-complete-actual-initial-scene-v1'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays_equal(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> bool:
    """Exact shape, dtype and value equality of a named finite state collection."""
    if set(first) != set(second):
        return False
    for name, array in first.items():
        other = second[name]
        if array.dtype != other.dtype or array.shape != other.shape:
            return False
        if not np.isfinite(array).all() or not np.isfinite(other).all():
            raise ValueError('Nonfinite state cannot establish paired initial conditions')
        if not np.array_equal(array, other):
            return False
    return True


def recorded_initial(path: Path) -> dict[str, np.ndarray]:
    """Read only recorded frame zero and exported task-role geometry, not all static shapes."""
    import h5py
    result: dict[str, np.ndarray] = {}
    with h5py.File(path/'trajectory.h5', 'r') as handle:
        episodes = list(handle)
        if episodes != ['traj_0']:
            raise ValueError('Expected one actual recorded episode')

        def read(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                result['env_states/'+name] = np.asarray(value[0]).copy()

        handle['traj_0/env_states'].visititems(read)
    with np.load(path/'task-geometry.npz', allow_pickle=False) as archive:
        result.update({'task_geometry/'+name: archive[name].copy() for name in archive.files})
    if not result:
        raise ValueError('Initial state evidence is empty')
    return result


def contact_index_checks(ticks: list[dict[str, Any]], physics: list[dict[str, Any]],
                         robots: list[dict[str, Any]], steps: int) -> dict[str, bool]:
    """Require ordered 1..T ticks and the same five actual global substeps per tick.

    Both contact streams must enumerate (control_step, substep) as
    (1,1)..(1,5),(2,6)..(T,5T), with matching nonempty phase labels. Missing,
    duplicate, reordered or cross-stream mismatched observations fail closed.
    """
    positive = type(steps) is int and steps > 0
    expected = [((index-1)//5+1, index) for index in range(1, 5*steps+1)] if positive else []

    def exact_indices(rows: list[dict[str, Any]]) -> bool:
        return positive and len(rows) == len(expected) and all(
            type(row.get('control_step')) is int and type(row.get('substep')) is int
            and (row['control_step'], row['substep']) == pair
            for row, pair in zip(rows, expected))

    physics_exact, robots_exact = exact_indices(physics), exact_indices(robots)
    ticks_exact = positive and len(ticks) == steps and all(
        type(row.get('environment_step')) is int and row['environment_step'] == index
        for index, row in enumerate(ticks, 1))
    aligned = physics_exact and robots_exact and all(
        isinstance(a.get('phase'), str) and bool(a['phase']) and a['phase'] == b.get('phase')
        and a['phase'] == ticks[a['control_step']-1].get('phase')
        for a, b in zip(physics, robots)) if ticks_exact else False
    return {'actual_100Hz_recording': positive and len(ticks) == steps
                and len(physics) == len(robots) == steps*5,
            'control_tick_indices_exact': ticks_exact,
            'physics_tick_substep_indices_exact': physics_exact,
            'robot_tick_substep_indices_exact': robots_exact,
            'five_substeps_every_tick': physics_exact and robots_exact,
            'physics_robot_substep_alignment': aligned}


def timing_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Preserve measured seconds; absent/incomplete totals are null, never invented zero.

    Explicit empty receipt lists establish zero. Missing, nonfinite or negative
    components make that total unknown; available component subtotals remain
    visible in timing_receipts and do not enter known-total means.
    """
    result: dict[str, Any] = {}
    receipts: dict[str, Any] = {}

    def collect(name: str, rows: Any, key: str) -> None:
        if not isinstance(rows, list):
            result[name] = None
            receipts[name] = {'recorded_components': None, 'known_components': 0,
                              'known_subtotal_s': None, 'total_known': False}
            return
        values = [row.get(key) if isinstance(row, dict) else None for row in rows]
        known = [float(value) for value in values if type(value) in (int, float)
                 and math.isfinite(value) and value >= 0]
        complete = len(known) == len(rows)
        result[name] = sum(known) if complete else None
        receipts[name] = {'recorded_components': len(rows), 'known_components': len(known),
                          'known_subtotal_s': sum(known) if known or not rows else None,
                          'total_known': complete}

    initialization = report.get('initialization')
    initial_rows = [row.get('initialization') if isinstance(row, dict) else None
                    for row in initialization] if isinstance(initialization, list) else None
    collect('initialization_s', initial_rows, 'initialization_s')
    collect('warmup_s', initial_rows, 'warmup_s')
    updates: list[Any] | None = None
    for row in report.get('planning', []):
        if 'updates' in row:
            if updates is None:
                updates = []
            update_rows = row['updates'] if isinstance(row['updates'], list) else [None]
            updates.extend(item.get('native_update') if isinstance(item, dict) else None
                           for item in update_rows)
    if 'released_updates' in report:
        if updates is None:
            updates = []
        released = report['released_updates'] if isinstance(report['released_updates'], list) else [None]
        updates.extend(item.get('native_update') if isinstance(item, dict) else None for item in released)
    collect('state_update_s', updates, 'update_wall_s')
    collect('native_solve_s', report.get('native_planning_calls'), 'wall_seconds')
    collect('execution_stage_wall_s', report.get('stages'), 'wall_seconds')
    result['state_update_calls'] = len(updates) if updates is not None else None
    result['timing_receipts'] = receipts
    return result


def known_timing_mean(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """Mean over known finite run totals, with an explicit attempt denominator."""
    values = [row[field] for row in rows if row.get(field) is not None]
    return {'mean': sum(values)/len(values) if values else None,
            'known_attempts': len(values), 'unknown_attempts': len(rows)-len(values)}


def validate_frozen_expectations(value: Any) -> dict[str, Any]:
    """Validate optional protocol metadata before reading any attempts.

    source_sha256 uses exact original absolute keys from report.source_sha256.
    worker_sha256_by_backend selects the actual report.worker_script key in
    that same map. Archived bytes, rather than current workspace files, are
    checked against each required source hash.
    """
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value)-{
            'planner_seed', 'source_sha256', 'worker_sha256_by_backend', 'require_initial_scene_manifest'}:
        raise ValueError('Unsupported frozen_expectations keys')
    if 'planner_seed' in value and (type(value['planner_seed']) is not int or value['planner_seed'] < 0):
        raise ValueError('Frozen planner_seed must be a nonnegative integer')
    if 'require_initial_scene_manifest' in value and type(value['require_initial_scene_manifest']) is not bool:
        raise ValueError('require_initial_scene_manifest must be boolean')
    for key in ('source_sha256', 'worker_sha256_by_backend'):
        if key not in value:
            continue
        mapping = value[key]
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f'Frozen {key} must be a nonempty mapping')
        for name, expected in mapping.items():
            if (not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64
                    or any(character not in '0123456789abcdef' for character in expected)):
                raise ValueError('Frozen source identities require lowercase SHA256 hex strings')
            if key == 'source_sha256' and not Path(name).is_absolute():
                raise ValueError('Frozen source keys must be original absolute report paths')
            if key == 'worker_sha256_by_backend' and name not in ('mplib', 'curobo'):
                raise ValueError('Unknown frozen worker backend')
    return value


def frozen_checks(path: Path, report: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    """Requested freeze mismatches fail qualification while retaining the attempt."""
    checks: dict[str, bool] = {}
    if 'planner_seed' in expected:
        checks['frozen_planner_seed'] = type(report.get('planner_seed')) is int \
            and report['planner_seed'] == expected['planner_seed']
    source = report.get('source_sha256', {})

    def source_matches(name: Any, sha: str) -> bool:
        if not isinstance(name, str) or not isinstance(source, dict) or source.get(name) != sha:
            return False
        archived = path/'source'/Path(name).name
        try:
            return archived.is_file() and digest(archived) == sha
        except OSError:
            return False

    if 'source_sha256' in expected:
        checks['frozen_source_sha256'] = all(source_matches(name, sha)
            for name, sha in expected['source_sha256'].items())
    worker = expected.get('worker_sha256_by_backend', {}).get(report['backend'])
    if worker is not None:
        checks['frozen_worker_sha256'] = source_matches(report.get('worker_script'), worker)
    return checks


def initial_scene_evidence(path: Path, report: dict[str, Any], *, required: bool) -> dict[str, Any]:
    """Verify saved actual initial geometry independently of its planner model.

    This is the canonical JSON hash algorithm of initial_scene_manifest.py:
    sorted keys, compact separators, UTF-8, finite JSON, one trailing newline.
    Keeping the three-line hash local avoids importing native physics bindings
    into the read-only analysis process. File-byte hashes are checked per case;
    only canonical scene-content hashes are compared between backends.
    """
    receipt = report.get('initial_scene_manifest')
    result: dict[str, Any] = {'required':required,'checked':False,
                             'provided':isinstance(receipt, dict), 'verified':False}
    if not required:
        return result
    file = path/'initial-scene.json'
    result.update(file=str(file), checks={}, errors=[])
    try:
        if not isinstance(receipt, dict):
            raise ValueError('report.initial_scene_manifest is missing')
        encoded = file.read_bytes()
        data = json.loads(encoded)
        scene, coverage, proof = data['scene'], data['coverage'], data['read_only_proof']
        if not isinstance(scene, dict) or not isinstance(coverage, dict) or not isinstance(proof, dict):
            raise ValueError('Invalid initial scene, coverage or read-only proof object')
        canonical = (json.dumps(scene,sort_keys=True,separators=(',',':'),
                                ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
        scene_sha = hashlib.sha256(canonical).hexdigest()
        file_sha = hashlib.sha256(encoded).hexdigest()
        entities, robots = scene['entities'], scene['robots']
        if (not isinstance(entities, dict) or not isinstance(robots, list) or len(robots) != 2
                or any(not isinstance(row, dict) or not isinstance(row.get('links'), dict) for row in robots)):
            raise ValueError('Expected all scene entities and two physical robot link collections')
        robot_links = [len(row['links']) for row in robots]
        robot_shapes = [sum(len(link['shapes']) for link in row['links'].values()) for row in robots]
        nonrobot_shapes = sum(len(row['shapes']) for row in entities.values())
        coverage_counts = coverage.get('robot_count') == 2 \
            and coverage.get('nonrobot_entity_count') == len(entities) \
            and coverage.get('nonrobot_physical_body_count') == sum(row.get('body') is not None for row in entities.values()) \
            and coverage.get('robot_link_counts') == robot_links \
            and coverage.get('robot_shape_counts') == robot_shapes \
            and coverage.get('native_articulation_link_count') == sum(robot_links) \
            and coverage.get('nonrobot_shape_count') == nonrobot_shapes \
            and coverage.get('total_shape_count') == nonrobot_shapes+sum(robot_shapes) \
            and coverage.get('raw_physics_entity_names') == sorted(
                name for name, row in entities.items() if row.get('raw_physics_entity') is True)
        zero_steps = all(isinstance(proof.get(key),list) and len(proof[key]) == 1
                         and type(proof[key][0]) is int and proof[key][0] == 0
                         for key in ('elapsed_steps_before','elapsed_steps_after'))
        checks = {'schema_matches':data.get('schema') == receipt.get('schema') == INITIAL_SCENE_SCHEMA,
            'world_frame_declared':data.get('frame') == 'world',
            'file_sha256_matches':receipt.get('sha256') == file_sha,
            'canonical_scene_sha256_matches':receipt.get('scene_sha256') == data.get('scene_sha256') == scene_sha,
            'receipt_coverage_and_proof_match':receipt.get('coverage') == coverage and receipt.get('read_only_proof') == proof,
            'complete_actual_inventory_declared':all(coverage.get(key) is True for key in
                ('all_covered','native_body_and_link_inventories_exactly_matched','all_collision_shapes_included_without_mask_filtering')),
            'saved_geometry_inventory_counts_match':coverage_counts and [row.get('arm_index') for row in robots] == [0,1],
            'initial_zero_step_read_only_proof':zero_steps
                and type(proof.get('environment_steps_during_export')) is int and proof['environment_steps_during_export'] == 0
                and proof.get('state_and_elapsed_unchanged') is True and proof.get('full_scene_repeated_read_consistent') is True
                and proof.get('planner_or_sphere_model_created') is False
                and isinstance(proof.get('state_digest_before'), str) and bool(proof['state_digest_before'])
                and proof['state_digest_before'] == proof.get('state_digest_after')}
        result.update(checked=True,file_sha256=file_sha,scene_sha256=scene_sha,
                      checks=checks,verified=all(checks.values()))
    except (OSError,KeyError,TypeError,ValueError) as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def closure_checks(report: dict[str, Any]) -> dict[str, bool]:
    """Report-level checks supplement raw H5/contact evidence, never replace it."""
    steps = report.get('total_environment_steps', 0)
    stages = {row['name']: row for row in report.get('stages', [])}
    placement = stages.get('verify_placement', {})
    retreat = np.asarray(report.get('actual_retreat_displacement_m', []), dtype=float)
    recording = report.get('recording', {})
    return {
        'terminal_pass': report.get('status') == 'passed',
        'official_success_and_budget': report.get('official_200_step_complete') is True
            and report.get('max_episode_steps') == 200 and type(steps) is int and 0 < steps < 200,
        'project_complete_physical_closure': report.get('project_complete_physical_closure') is True,
        'all_executed_stages_passed': bool(stages) and all(row['status'] == 'passed' for row in stages.values()),
        'retreat_at_least_8cm': retreat.shape == (2, 3) and bool(np.isfinite(retreat).all())
            and bool(np.all(retreat[:, 2] >= .08)),
        'continuous_placement_20': placement.get('status') == 'passed'
            and placement.get('stable_steps_required') == 20 and placement.get('stable_steps_observed', 0) >= 20,
        'held_slip_audit': report.get('held_physics_audit', {}).get('status') == 'passed',
        'no_recording_artifact_errors': report.get('artifact_errors') == [],
        'recording_report_action_integrity': recording.get('submitted_actions_matched') is True
            and recording.get('actions_shape') == [steps, 16]
            and bool(recording.get('state_lengths'))
            and all(value == steps+1 for value in recording.get('state_lengths', {}).values()),
    }


def summarize_case(path: Path, expected: dict[str, Any], *,
                   frozen: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect one terminal native attempt and its actual commands/contact rows."""
    import h5py
    report_path = path/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status') not in ('passed', 'failed'):
        raise ValueError('Allocated case has no terminal result')
    for key in ('task', 'seed', 'backend'):
        if report[key] != expected[key]:
            raise ValueError(f'Allocated case identity differs in {key}')
    requirements = validate_frozen_expectations(frozen)
    freeze = frozen_checks(path, report, requirements)
    initial_scene = initial_scene_evidence(path, report,
        required=requirements.get('require_initial_scene_manifest',False))
    if initial_scene['required']:
        freeze['frozen_initial_scene_manifest'] = initial_scene['verified']
    timings = timing_fields(report)
    required = ['trajectory.h5', 'ticks.jsonl', 'physics-substeps.jsonl', 'all-robot-contacts.json']
    missing = [name for name in required if not (path/name).is_file()]
    if missing:
        wall = float(report['wall_s'])
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError('Finite measured failed-attempt wall time is required')
        physical_path,robot_path = path/'physics-substeps.jsonl',path/'all-robot-contacts.json'
        partial_physics = [json.loads(line) for line in physical_path.read_text(encoding='utf-8').splitlines()] if physical_path.exists() else []
        partial_robots = json.loads(robot_path.read_text(encoding='utf-8')) if robot_path.exists() else []
        target_count = sum(contact['physical_contact'] and contact['class']=='forbidden'
            for row in partial_physics for contact in row['object_contact_rows']) if physical_path.exists() else None
        robot_count = sum(contact['physical_contact'] and not contact['allowed']
            for row in partial_robots for contact in row['contacts']) if robot_path.exists() else None
        interarm_count = sum(contact['physical_contact'] and len(contact['sides'])==2
            and all(side.get('kind')=='robot' for side in contact['sides'])
            and contact['sides'][0]['robot_index']!=contact['sides'][1]['robot_index']
            for row in partial_robots for contact in row['contacts']) if robot_path.exists() else None
        return {'task':report['task'],'seed':report['seed'],'backend':report['backend'],
            'status':report['status'],'qualified':False,
            'checks':{'complete_physical_evidence_present':False, **freeze},
            'frozen_checks':freeze,
            'initial_scene_manifest':initial_scene,
            'report':str(report_path),'report_sha256':digest(report_path),
            'missing_evidence':missing,'steps':int(report.get('total_environment_steps',0)),
            'simulation_s':int(report.get('total_environment_steps',0))*float(report.get('control_timestep_s',.05)),
            'native_wall_s':wall, **timings,
            'forbidden_robot_contacts':robot_count,'forbidden_target_contacts':target_count,
            'actual_interarm_contacts':interarm_count,'contact_audit_complete':False,
            'physical_observation_count':len(partial_physics),'predicted_collision_rejections':0,
            'execution_configuration':report['execution_configuration'],'gpu_uuid':report['gpu_uuid'],
            'held_slip':report.get('held_physics_audit'),
            'failure':report.get('error',report.get('reason','missing_physical_evidence'))}
    ticks = [json.loads(line) for line in (path/'ticks.jsonl').read_text(encoding='utf-8').splitlines()]
    physics = [json.loads(line) for line in (path/'physics-substeps.jsonl').read_text(encoding='utf-8').splitlines()]
    robots = json.loads((path/'all-robot-contacts.json').read_text(encoding='utf-8'))
    steps = report.get('total_environment_steps', 0)
    checks = {**closure_checks(report), **freeze}
    keys = report.get('action_key_order', [])
    commands = np.asarray([np.concatenate([row['actions'][key] for key in keys]) for row in ticks],
                          dtype=np.float32) if ticks else np.empty((0, 16), dtype=np.float32)
    evidence_errors: list[str] = []
    checks['actual_H5_actions_match'] = checks['actual_H5_Tplus1'] = False
    try:
        with h5py.File(path/'trajectory.h5', 'r') as handle:
            actions = np.asarray(handle['traj_0/actions'])
            lengths: list[int] = []
            handle['traj_0/env_states'].visititems(lambda name, value:
                lengths.append(len(value)) if isinstance(value, h5py.Dataset) else None)
            checks['actual_H5_actions_match'] = actions.shape == (steps, 16) \
                and bool(np.isfinite(actions).all()) and bool(np.array_equal(actions, commands))
            checks['actual_H5_Tplus1'] = bool(lengths) and all(count == steps+1 for count in lengths)
    except (OSError, KeyError, ValueError) as error:
        evidence_errors.append(f'trajectory.h5: {type(error).__name__}: {error}')
    index_checks = contact_index_checks(ticks, physics, robots, steps)
    checks.update(index_checks)
    forbidden_robot = sum(contact['physical_contact'] and not contact['allowed']
                          for row in robots for contact in row['contacts'])
    forbidden_target = sum(contact['physical_contact'] and contact['class'] == 'forbidden'
                           for row in physics for contact in row['object_contact_rows'])
    interarm = sum(contact['physical_contact'] and len(contact['sides']) == 2
        and all(side.get('kind') == 'robot' for side in contact['sides'])
        and contact['sides'][0]['robot_index'] != contact['sides'][1]['robot_index']
        for row in robots for contact in row['contacts'])
    checks['zero_forbidden_physical_contacts'] = forbidden_robot == forbidden_target == interarm == 0
    wall = float(report['wall_s'])
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError('Finite measured native wall time is required')
    planning_rejections = sum(row.get('collision_validation', {}).get('status') == 'rejected'
                              for row in report.get('planning', []))
    return {'task':report['task'], 'seed':report['seed'], 'backend':report['backend'],
        'status':report['status'], 'qualified':all(checks.values()), 'checks':checks,
        'frozen_checks':freeze, 'evidence_errors':evidence_errors,
        'initial_scene_manifest':initial_scene,
        'report':str(report_path), 'report_sha256':digest(report_path), 'steps':steps,
        'simulation_s':steps*float(report.get('control_timestep_s', .05)), 'native_wall_s':wall,
        **timings,
        'forbidden_robot_contacts':int(forbidden_robot), 'forbidden_target_contacts':int(forbidden_target),
        'actual_interarm_contacts':int(interarm), 'physical_observation_count':len(physics),
        'contact_audit_complete':all(index_checks.values()),
        'predicted_collision_rejections':planning_rejections,
        'execution_configuration':report['execution_configuration'], 'gpu_uuid':report['gpu_uuid'],
        'held_slip':report.get('held_physics_audit'), 'failure':report.get('error', report.get('reason'))}


def summarize(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    frozen = validate_frozen_expectations(protocol.get('frozen_expectations'))
    root = protocol_path.parent
    cases = protocol['cases']
    identities = [(row['task'], row['seed'], row['backend']) for row in cases]
    if len(identities) != len(set(identities)) or not cases:
        raise ValueError('Frozen protocol needs unique nonempty allocated cases')
    rows, pending = [], []
    for case in cases:
        path = root/case['path']
        if not (path/'report.json').is_file():
            pending.append(case)
            continue
        if json.loads((path/'report.json').read_text(encoding='utf-8')).get('status') == 'running':
            pending.append(case)
            continue
        rows.append(summarize_case(path, case, frozen=frozen))
    pairs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row['task'], row['seed']][row['backend']] = row
    for (task, seed), backends in sorted(grouped.items()):
        if set(backends) != {'mplib', 'curobo'}:
            continue
        a, b = backends['mplib'], backends['curobo']
        scene_required = frozen.get('require_initial_scene_manifest',False)
        scene_equal = a['initial_scene_manifest']['verified'] and b['initial_scene_manifest']['verified'] \
            and a['initial_scene_manifest']['scene_sha256'] == b['initial_scene_manifest']['scene_sha256'] \
            if scene_required else None
        initial_error = None
        try:
            equal = arrays_equal(recorded_initial(Path(a['report']).parent), recorded_initial(Path(b['report']).parent))
        except (OSError,KeyError,ValueError) as error:
            equal = False
            initial_error = f'{type(error).__name__}: {error}'
        pairs.append({'task':task,'seed':seed,'recorded_initial_and_task_geometry_arrays_identical':equal,
            'initial_comparison_error':initial_error,
            'complete_initial_scene_manifest_required':scene_required,
            'complete_initial_scene_manifests_identical':scene_equal,
            'execution_configuration_identical':a['execution_configuration'] == b['execution_configuration'],
            'same_gpu':isinstance(a['gpu_uuid'], str) and a['gpu_uuid'].startswith('GPU-')
                and a['gpu_uuid'] == b['gpu_uuid'], 'mplib_qualified':a['qualified'],
            'curobo_qualified':b['qualified']})
    aggregates = []
    for task, backend in sorted({(row['task'], row['backend']) for row in rows}):
        subset = [row for row in rows if row['task'] == task and row['backend'] == backend]
        n, accepted = len(subset), sum(row['qualified'] for row in subset)
        wall = sum(row['native_wall_s'] for row in subset)
        observed = [row for row in subset if row['physical_observation_count'] and row['contact_audit_complete']]
        collision_cases = sum(bool(row['forbidden_robot_contacts'] or row['forbidden_target_contacts']
                                  or row['actual_interarm_contacts']) for row in observed)
        timing_means = {field:known_timing_mean(subset, field) for field in TIMING_FIELDS}
        allocated = sum(case['task'] == task and case['backend'] == backend for case in cases)
        aggregates.append({'task':task,'backend':backend,'attempts':n,'qualified':accepted,
            'allocated_attempts':allocated, 'terminal_attempts':n, 'pending_attempts':allocated-n,
            'success_rate':accepted/n,'native_wall_s':wall,'simulation_s':sum(row['simulation_s'] for row in subset),
            'effective_trajectories_per_native_wall_hour':3600*accepted/wall,
            'collision_case_count':collision_cases,'physically_observed_cases':len(observed),
            'observed_collision_case_rate':collision_cases/len(observed) if observed else None,
            'known_collision_cases':sum(any(row[key] is not None and row[key] > 0 for key in
                ('forbidden_robot_contacts', 'forbidden_target_contacts', 'actual_interarm_contacts')) for row in subset),
            'unobserved_cases':n-len(observed),
            'mean_steps':sum(row['steps'] for row in subset)/n,
            **{'mean_'+field:row['mean'] for field, row in timing_means.items()},
            'timing_mean_denominators':{field:{key:value for key, value in row.items() if key != 'mean'}
                                      for field, row in timing_means.items()}})
    integrity = all(pair['recorded_initial_and_task_geometry_arrays_identical']
                    and pair['execution_configuration_identical'] and pair['same_gpu']
                    and (not pair['complete_initial_scene_manifest_required']
                         or pair['complete_initial_scene_manifests_identical']) for pair in pairs)
    frozen_integrity = all(all(row['frozen_checks'].values()) for row in rows)
    complete = not pending and len(rows) == len(cases) and len(pairs)*2 == len(cases)
    return {'status':'complete' if complete and integrity and frozen_integrity else 'incomplete_or_incomparable',
        'protocol_sha256':digest(protocol_path),'allocated_cases':len(cases),'terminal_cases':len(rows),
        'pending':pending,'paired_recorded_initial_and_configuration_integrity':integrity,
        'initial_comparison_scope':INITIAL_COMPARISON_SCOPE,
        'complete_initial_scene_manifest_scope':{
            'required':frozen.get('require_initial_scene_manifest',False),
            'included_when_verified':'all saved actual nonrobot entities including raw ground/wall, both physical robot links/shapes, raw geometry, poses and collision groups at elapsed step zero',
            'comparison':'each file and canonical content hash independently verified; paired canonical scene hashes must match',
            'legacy_records_without_requirement':'missing manifest does not alter physical qualification or extend recorded-task-geometry scope'},
        'frozen_expectations':frozen, 'frozen_expectations_integrity':frozen_integrity,
        'pairs':pairs,'aggregates':aggregates,'runs':rows,
        'scope':'small fixed engineering cohort; native wall includes each run startup/warmup and failures; external orchestration/analysis wall excluded; internal planner collision models differ'}


def self_test() -> dict[str, Any]:
    assert arrays_equal({'x':np.zeros((1, 7), dtype=np.float32)}, {'x':np.zeros((1, 7), dtype=np.float32)})
    assert not arrays_equal({'x':np.zeros((1, 7))}, {'x':np.ones((1, 7))})
    assert not arrays_equal({'x':np.zeros((1, 7))}, {'x':np.zeros((7,))})
    assert not arrays_equal({'x':np.zeros((1, 7))}, {'y':np.zeros((1, 7))})
    assert not arrays_equal({'x':np.zeros((1, 7),dtype=np.float32)}, {'x':np.zeros((1, 7),dtype=np.float64)})
    report = {'status':'passed','official_200_step_complete':True,'max_episode_steps':200,
        'total_environment_steps':199,'project_complete_physical_closure':True,
        'stages':[{'name':'verify_placement','status':'passed','stable_steps_required':20,'stable_steps_observed':20}],
        'actual_retreat_displacement_m':[[0.,0.,.081],[0.,0.,.081]],
        'held_physics_audit':{'status':'passed'},'artifact_errors':[],
        'recording':{'submitted_actions_matched':True,'actions_shape':[199,16],'state_lengths':{'actor':200}}}
    assert all(closure_checks(report).values())
    for field, value in [('total_environment_steps',200),('project_complete_physical_closure',False),
                         ('artifact_errors',['lost recording']),('actual_retreat_displacement_m',[[0.,0.,.079]]*2)]:
        assert not all(closure_checks({**report,field:value}).values())
    try:
        arrays_equal({'x':np.array([np.nan])},{'x':np.array([np.nan])})
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite paired state accepted')
    import copy
    sample_ticks = [{'environment_step':index, 'phase':'synthetic'} for index in (1, 2)]
    sample_physics = [{'control_step':(index-1)//5+1, 'substep':index, 'phase':'synthetic'}
                      for index in range(1, 11)]
    assert all(contact_index_checks(sample_ticks, sample_physics, copy.deepcopy(sample_physics), 2).values())
    index_negative_cases = []
    missing = copy.deepcopy(sample_physics[:-1])
    index_negative_cases.append((sample_ticks, missing, sample_physics))
    duplicate = copy.deepcopy(sample_physics); duplicate[1] = copy.deepcopy(duplicate[0])
    index_negative_cases.append((sample_ticks, sample_physics, duplicate))
    wrong_tick = copy.deepcopy(sample_physics); wrong_tick[5]['control_step'] = 1
    index_negative_cases.append((sample_ticks, wrong_tick, sample_physics))
    reordered = copy.deepcopy(sample_physics); reordered[0], reordered[1] = reordered[1], reordered[0]
    index_negative_cases.append((sample_ticks, sample_physics, reordered))
    wrong_phase = copy.deepcopy(sample_physics); wrong_phase[0]['phase'] = 'other'
    index_negative_cases.append((sample_ticks, sample_physics, wrong_phase))
    bool_index = copy.deepcopy(sample_physics); bool_index[0]['substep'] = True
    index_negative_cases.append((sample_ticks, bool_index, sample_physics))
    tick_duplicate = copy.deepcopy(sample_ticks); tick_duplicate[1]['environment_step'] = 1
    index_negative_cases.append((tick_duplicate, sample_physics, sample_physics))
    for observed_ticks, observed_physics, observed_robots in index_negative_cases:
        assert not all(contact_index_checks(observed_ticks, observed_physics, observed_robots, 2).values())
    assert not any(contact_index_checks([], [], [], 0).values())
    unknown = timing_fields({})
    assert all(unknown[name] is None for name in TIMING_FIELDS)
    zero = timing_fields({'initialization':[], 'planning':[{'updates':[]}],
                          'native_planning_calls':[], 'stages':[]})
    assert all(zero[name] == 0 for name in TIMING_FIELDS)
    measured_report = {'initialization':[{'initialization':{'initialization_s':1.,'warmup_s':3.}},
                                         {'initialization':{'initialization_s':2.,'warmup_s':4.}}],
        'planning':[{'updates':[{'native_update':{'update_wall_s':.4}}]}],
        'released_updates':[{'native_update':{'update_wall_s':.6}}],
        'native_planning_calls':[{'wall_seconds':2.5}], 'stages':[{'wall_seconds':6.}]}
    measured = timing_fields(measured_report)
    assert [measured[name] for name in TIMING_FIELDS] == [3.,7.,1.,2.5,6.]
    incomplete = timing_fields({'initialization':[{'initialization':{'initialization_s':1.}}, {}]})
    assert incomplete['initialization_s'] is None
    assert incomplete['timing_receipts']['initialization_s']['known_subtotal_s'] == 1.
    invalid = timing_fields({'native_planning_calls':[{'wall_seconds':float('nan')}, {'wall_seconds':-1.}]})
    assert invalid['native_solve_s'] is None
    assert known_timing_mean([{'native_solve_s':None}, {'native_solve_s':0.}, {'native_solve_s':5.}],
                             'native_solve_s') == {'mean':2.5,'known_attempts':2,'unknown_attempts':1}
    assert known_timing_mean([{'native_solve_s':None}], 'native_solve_s')['mean'] is None
    for invalid_frozen in ({'planner_seed':True}, {'planner_seed':-1}, {'unknown':1},
            {'require_initial_scene_manifest':1},
            {'source_sha256':{'relative.py':'0'*64}},
            {'source_sha256':{'/absolute.py':'x'*64}},
            {'worker_sha256_by_backend':{'other':'0'*64}}):
        try:
            validate_frozen_expectations(invalid_frozen)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid frozen expectation accepted')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='tig-comparison-cpu-') as directory:
        path = Path(directory)
        failed = {'status':'failed','task':'synthetic','seed':1,'backend':'curobo','wall_s':2.,
                  'execution_configuration':{},'gpu_uuid':'GPU-test','error':'environment_creation_failed'}
        (path/'report.json').write_text(json.dumps(failed),encoding='utf-8')
        result = summarize_case(path,{'task':'synthetic','seed':1,'backend':'curobo'})
        assert not result['qualified'] and result['physical_observation_count']==0
        assert result['actual_interarm_contacts'] is None and len(result['missing_evidence'])==4
        assert result['initialization_s'] is None and result['native_solve_s'] is None
        (path/'physics-substeps.jsonl').write_text(json.dumps({'object_contact_rows':[
            {'physical_contact':True,'class':'forbidden'}]})+'\n',encoding='utf-8')
        partial = summarize_case(path,{'task':'synthetic','seed':1,'backend':'curobo'})
        assert partial['physical_observation_count']==1 and partial['forbidden_target_contacts']==1
        assert partial['actual_interarm_contacts'] is None and not partial['contact_audit_complete']
        with_timings = {**failed, **measured_report}
        (path/'report.json').write_text(json.dumps(with_timings),encoding='utf-8')
        retained = summarize_case(path,{'task':'synthetic','seed':1,'backend':'curobo'})
        assert [retained[name] for name in TIMING_FIELDS] == [3.,7.,1.,2.5,6.]
        archive = path/'source'; archive.mkdir()
        source_path = '/frozen/runtime/worker.py'
        (archive/'worker.py').write_bytes(b'synthetic CPU source identity\n')
        sha = digest(archive/'worker.py')
        freeze = validate_frozen_expectations({'planner_seed':0,
            'source_sha256':{source_path:sha}, 'worker_sha256_by_backend':{'curobo':sha}})
        identity_report = {**failed,'planner_seed':0,'worker_script':source_path,
                           'source_sha256':{source_path:sha}}
        assert all(frozen_checks(path,identity_report,freeze).values())
        assert not all(frozen_checks(path,{**identity_report,'planner_seed':1},freeze).values())
        assert not all(frozen_checks(path,{**identity_report,'planner_seed':True},freeze).values())
        assert not all(frozen_checks(path,{**identity_report,'source_sha256':{}},freeze).values())
        assert not all(frozen_checks(path,{**identity_report,'source_sha256':{source_path:'0'*64}},freeze).values())
        assert not all(frozen_checks(path,{**identity_report,'worker_script':None},freeze).values())
        (archive/'worker.py').write_bytes(b'changed archive bytes\n')
        assert not all(frozen_checks(path,identity_report,freeze).values())
        (archive/'worker.py').write_bytes(b'synthetic CPU source identity\n')

        # All filenames exist, but pre-action RecordEpisode H5 has no episode.
        import h5py
        with h5py.File(path/'trajectory.h5', 'w'):
            pass
        (path/'ticks.jsonl').write_text('',encoding='utf-8')
        (path/'physics-substeps.jsonl').write_text('',encoding='utf-8')
        (path/'all-robot-contacts.json').write_text('[]',encoding='utf-8')
        (path/'report.json').write_text(json.dumps(identity_report),encoding='utf-8')
        empty = summarize_case(path,{'task':'synthetic','seed':1,'backend':'curobo'},frozen=freeze)
        assert not empty['qualified'] and not empty['contact_audit_complete']
        assert empty['evidence_errors'] and not empty['checks']['actual_H5_actions_match']
        assert all(empty['frozen_checks'].values())

        good = path/'good'; good.mkdir(); (good/'source').mkdir()
        (good/'source/worker.py').write_bytes(b'synthetic CPU source identity\n')
        t = 20
        good_report = {**identity_report,**report,'task':'synthetic','seed':2,'backend':'curobo',
            'total_environment_steps':t,'wall_s':4.,'control_timestep_s':.05,
            'action_key_order':['left','right'], 'native_planning_calls':[{'wall_seconds':2.}],
            'recording':{'submitted_actions_matched':True,'actions_shape':[t,16],'state_lengths':{'actor':t+1}}}
        (good/'report.json').write_text(json.dumps(good_report),encoding='utf-8')
        with h5py.File(good/'trajectory.h5','w') as handle:
            episode = handle.create_group('traj_0')
            episode.create_dataset('actions',data=np.zeros((t,16),dtype=np.float32))
            episode.create_group('env_states').create_dataset('actor',data=np.zeros((t+1,13)))
        tick_rows = [{'environment_step':index,'phase':'synthetic',
                      'actions':{'left':[0.]*8,'right':[0.]*8}} for index in range(1,t+1)]
        physical_rows = [{'control_step':(index-1)//5+1,'substep':index,
                          'phase':'synthetic','object_contact_rows':[]} for index in range(1,5*t+1)]
        robot_rows = [{**row,'contacts':[]} for row in physical_rows]
        (good/'ticks.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in tick_rows),encoding='utf-8')
        (good/'physics-substeps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in physical_rows),encoding='utf-8')
        (good/'all-robot-contacts.json').write_text(json.dumps(robot_rows),encoding='utf-8')
        accepted = summarize_case(good,{'task':'synthetic','seed':2,'backend':'curobo'},frozen=freeze)
        assert accepted['qualified'] and accepted['contact_audit_complete']
        rejected = summarize_case(good,{'task':'synthetic','seed':2,'backend':'curobo'},
                                  frozen={**freeze,'planner_seed':1})
        assert not rejected['qualified'] and not rejected['frozen_checks']['frozen_planner_seed']
        protocol = {'frozen_expectations':freeze,'cases':[
            {'task':'synthetic','seed':1,'backend':'curobo','path':'.'},
            {'task':'synthetic','seed':2,'backend':'curobo','path':'good'}]}
        (path/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
        summary = summarize(path/'protocol.json')
        aggregate = summary['aggregates'][0]
        assert aggregate['attempts'] == 2 and aggregate['qualified'] == 1 and aggregate['success_rate'] == .5
        assert aggregate['native_wall_s'] == 6. and aggregate['mean_native_solve_s'] == 2.
        assert aggregate['timing_mean_denominators']['native_solve_s'] == {'known_attempts':1,'unknown_attempts':1}
        assert 'not_established' in summary['initial_comparison_scope']
        import shutil
        peer = path/'peer'
        shutil.copytree(good, peer)
        peer_report = {**good_report,'backend':'mplib','worker_script':None,'wall_s':3.}
        (peer/'report.json').write_text(json.dumps(peer_report),encoding='utf-8')
        for case_path in (good, peer):
            np.savez(case_path/'task-geometry.npz',vertices=np.zeros((1,3)),pose=np.array([0.,0.,0.,1.,0.,0.,0.]))
        pair_protocol = {'frozen_expectations':freeze,'cases':[
            {'task':'synthetic','seed':2,'backend':'curobo','path':'good'},
            {'task':'synthetic','seed':2,'backend':'mplib','path':'peer'}]}
        (path/'pair-protocol.json').write_text(json.dumps(pair_protocol),encoding='utf-8')
        paired = summarize(path/'pair-protocol.json')
        assert paired['status'] == 'complete' and paired['frozen_expectations_integrity']
        assert paired['pairs'][0]['recorded_initial_and_task_geometry_arrays_identical']
        (peer/'report.json').write_text(json.dumps({**peer_report,'planner_seed':1}),encoding='utf-8')
        mismatched = summarize(path/'pair-protocol.json')
        assert mismatched['terminal_cases'] == 2 and not mismatched['frozen_expectations_integrity']
        assert mismatched['status'] == 'incomplete_or_incomparable'
        assert sum(row['attempts'] for row in mismatched['aggregates']) == 2
        assert sum(row['qualified'] for row in mismatched['aggregates']) == 1
        assert sum(row['native_wall_s'] for row in mismatched['aggregates']) == 7.
        scene_freeze = {**freeze,'require_initial_scene_manifest':True}
        missing_scene = summarize_case(good,{'task':'synthetic','seed':2,'backend':'curobo'},frozen=scene_freeze)
        assert not missing_scene['qualified'] and not missing_scene['initial_scene_manifest']['verified']
        assert accepted['qualified'] and not accepted['initial_scene_manifest']['checked']
        (peer/'report.json').write_text(json.dumps(peer_report),encoding='utf-8')
        scene_content = {'entities':{
            'physics::ground':{'body':{},'raw_physics_entity':True,'shapes':[{'kind':'synthetic-plane'}]},
            'physics::wall':{'body':{},'raw_physics_entity':True,'shapes':[{'kind':'synthetic-box'}]}},
            'robots':[{'arm_index':index,'links':{'root':{'shapes':[{'kind':'synthetic-link'}]}}}
                      for index in range(2)],'physical_system':{'gravity':[0.,0.,-9.81]}}

        def save_synthetic_scene(case_path: Path, content: dict[str, Any], *,
                                 override: dict[str, Any] | None = None) -> dict[str, Any]:
            content_sha = hashlib.sha256((json.dumps(content,sort_keys=True,separators=(',',':'),
                ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')).hexdigest()
            data = {'schema':INITIAL_SCENE_SCHEMA,'frame':'world','scene':copy.deepcopy(content),
                'scene_sha256':content_sha,'coverage':{'all_covered':True,'robot_count':2,
                    'nonrobot_entity_count':2,'nonrobot_physical_body_count':2,'nonrobot_shape_count':2,
                    'robot_link_counts':[1,1],'robot_shape_counts':[1,1],'native_articulation_link_count':2,
                    'total_shape_count':4,'raw_physics_entity_names':['physics::ground','physics::wall'],
                    'native_body_and_link_inventories_exactly_matched':True,
                    'all_collision_shapes_included_without_mask_filtering':True},
                'read_only_proof':{'state_digest_before':'synthetic-cpu-state','state_digest_after':'synthetic-cpu-state',
                    'elapsed_steps_before':[0],'elapsed_steps_after':[0],'environment_steps_during_export':0,
                    'state_and_elapsed_unchanged':True,'full_scene_repeated_read_consistent':True,
                    'planner_or_sphere_model_created':False},
                'diagnostic_outside_canonical_scene':case_path.name}
            data.update(override or {})
            file = case_path/'initial-scene.json'
            file.write_text(json.dumps(data,allow_nan=False)+'\n',encoding='utf-8')
            receipt = {'schema':INITIAL_SCENE_SCHEMA,'path':str(file.resolve()),'sha256':digest(file),
                'scene_sha256':data['scene_sha256'],'coverage':data['coverage'],'read_only_proof':data['read_only_proof']}
            native_report = json.loads((case_path/'report.json').read_text(encoding='utf-8'))
            native_report['initial_scene_manifest'] = receipt
            (case_path/'report.json').write_text(json.dumps(native_report),encoding='utf-8')
            return data

        valid_scene = save_synthetic_scene(good,scene_content)
        save_synthetic_scene(peer,scene_content)
        for case_path in (good,peer):
            native_report = json.loads((case_path/'report.json').read_text(encoding='utf-8'))
            assert initial_scene_evidence(case_path,native_report,required=True)['verified']
        pair_protocol['frozen_expectations'] = scene_freeze
        (path/'pair-protocol.json').write_text(json.dumps(pair_protocol),encoding='utf-8')
        manifest_pair = summarize(path/'pair-protocol.json')
        assert manifest_pair['status'] == 'complete'
        assert manifest_pair['pairs'][0]['complete_initial_scene_manifests_identical'] is True
        assert digest(good/'initial-scene.json') != digest(peer/'initial-scene.json')
        native_report = json.loads((good/'report.json').read_text(encoding='utf-8'))
        native_report['initial_scene_manifest']['sha256'] = '0'*64
        assert not initial_scene_evidence(good,native_report,required=True)['verified']
        tampered_content = copy.deepcopy(scene_content)
        tampered_content['entities']['physics::wall']['changed_pose'] = [1.,0.,0.]
        save_synthetic_scene(good,tampered_content,override={'scene_sha256':valid_scene['scene_sha256']})
        native_report = json.loads((good/'report.json').read_text(encoding='utf-8'))
        rejected_scene = initial_scene_evidence(good,native_report,required=True)
        assert not rejected_scene['verified'] and not rejected_scene['checks']['canonical_scene_sha256_matches']
        for changed_key, changed_value in (
                ('coverage',{**valid_scene['coverage'],'all_covered':False}),
                ('coverage',{**valid_scene['coverage'],'robot_shape_counts':[0,1]}),
                ('read_only_proof',{**valid_scene['read_only_proof'],'elapsed_steps_after':[1]})):
            save_synthetic_scene(good,scene_content,override={changed_key:changed_value})
            native_report = json.loads((good/'report.json').read_text(encoding='utf-8'))
            assert not initial_scene_evidence(good,native_report,required=True)['verified']
        save_synthetic_scene(good,scene_content)
        save_synthetic_scene(peer,tampered_content)
        different_scene_pair = summarize(path/'pair-protocol.json')
        assert different_scene_pair['frozen_expectations_integrity']
        assert different_scene_pair['status'] == 'incomplete_or_incomparable'
        assert different_scene_pair['pairs'][0]['complete_initial_scene_manifests_identical'] is False
        assert sum(row['qualified'] for row in different_scene_pair['aggregates']) == 2
        (good/'report.json').write_text(json.dumps({**good_report,'status':'running'}),encoding='utf-8')
        pending = summarize(path/'protocol.json')
        assert pending['terminal_cases'] == 1 and len(pending['pending']) == 1
        assert pending['aggregates'][0]['attempts'] == 1 and pending['aggregates'][0]['pending_attempts'] == 1
        json.dumps(summary,allow_nan=False)
    return {'status':'passed','paired_state_shape':[1,7],'changed_state_shape_dtype_keys_rejected':True,
            'official_predicate_alone_cannot_qualify':True,'failed_without_artifacts_retained':True,
            'contact_index_positive_shapes':[2,10,10],'contact_index_negative_cases':len(index_negative_cases),
            'unknown_timing_is_null_and_known_denominators_explicit':True,
            'partial_failure_measured_timings_retained':True,'empty_H5_failure_retained':True,
            'optional_frozen_source_archive_worker_and_seed_checks':True,
            'paired_freeze_mismatch_retained_as_unqualified_and_incomparable':True,
            'frozen_invalid_protocol_cases':7,'failure_and_wall_denominators_retained':True,
            'complete_initial_scene_optional_legacy_compatible':True,
            'complete_initial_scene_file_content_hash_and_pair_mismatch_tests':True,
            'running_case_pending':True,'static_geometry_scope_explicit':True,
            'no_physics_or_cuda_import':True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    if args.protocol is None or args.output is None:
        parser.error('--protocol and --output are required')
    result = summarize(args.protocol.resolve())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','allocated_cases','terminal_cases')}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
