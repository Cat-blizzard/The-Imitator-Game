"""Versioned CPU continuation after the preserved two-job preview stop.

The factory and controller are separate. This driver never substitutes a task
after a failed attempt, regenerates a grasp, or changes native acceptance.
Candidates and the fixed paired baseline share the same input scene and task.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'bounded_multitask_composition_cpu_v2'
ALLOCATION_ID = 'multitask_10x8_composition_20261030_v2'
NEW_LIMIT = 88
CANDIDATE_LIMIT = 80
BASELINE_LIMIT = 8
TIMEOUT_SECONDS = 360
PARENT = ROOT / 'runs/multitask_10x8_20261030_v1'
REGISTRY = ROOT / 'runtime/multitask_campaign_registry_v2'
BASELINE_TASKS = ('collect_group', 'clear_then_place')
WORKER_MODULE = 'resimpair_tasks.multitask_worker_v2'


def path(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def prior_lineage(parent=PARENT):
    """Preserve earlier ceilings and actual usage as different quantities."""
    parent = path(parent); budget_path = parent / 'budget/limits.json'
    budget = read(budget_path)
    stop_path = parent / 'admission_stop.json'; stop = read(stop_path)
    if (stop.get('actual_attempts') != 2 or stop.get('unused_allocation_not_reassigned') != 86
            or stop.get('new_version_required') is not True or stop.get('repeat_failed_requests_in_this_root') is not False):
        raise ValueError('explicit_preserved_v1_stop_required')
    resolve(stop['diagnosis'])
    history = budget['prior_lineage']
    refs = {**history['budget_references'], 'multitask_v1_stopped_after_two': binding(budget_path)}
    ceilings = {name: read(resolve(ref))['physical_attempt_limit'] for name, ref in refs.items()}
    reservations = {name: len(list((resolve(ref).parent / 'reservations').glob('*.json')))
                    for name, ref in refs.items()}
    if (sum(ceilings.values()) != 122 or budget['cumulative_scope_limit'] != 122
            or budget['physical_attempt_limit'] != 88 or reservations['multitask_v1_stopped_after_two'] != 2):
        raise ValueError('expected_prior122_and_exactly_two_stopped_v1_attempts_required')
    if any(reservations[k] > ceilings[k] for k in ceilings):
        raise ValueError('historical_reservations_exceed_ceiling')
    stopped = []
    with (parent / '.supervisor.lock').open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for reservation in sorted((parent / 'budget/reservations').glob('*.json')):
            row = read(reservation)
            execution = Path(row['campaign']) / 'executions' / row['case']['id']
            terminal = read(execution / 'terminal.json')
            failure = read(execution / 'planning_failure.json')
            if terminal['physical_attempts'] != 1:
                raise ValueError('stopped_v1_attempt_has_no_physical_terminal')
            identity = read(execution / 'process.json'); proc = Path(f'/proc/{identity["pid"]}/stat')
            if proc.exists():
                ticks = int(proc.read_text().rsplit(')', 1)[1].split()[19])
                boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                if ticks == identity['start_ticks'] and boot == identity['boot_id']:
                    raise ValueError('predecessor_worker_still_alive')
            stopped.append({'reservation': binding(reservation), 'terminal': binding(execution / 'terminal.json'),
                'planning_failure': binding(execution / 'planning_failure.json'), 'failure_message': failure['message']})
    diagnosis = ROOT / 'validation/multitask_20261030/independent/INITIAL_PREVIEW_MUTATION_DIAGNOSIS.json'
    return {'budget_references': refs, 'allocation': ceilings, 'reservations_observed': reservations,
            'prior_physical_limit': sum(ceilings.values()),
            'prior_observed_attempts': sum(reservations.values()),
            'unspent_old_allowance_not_reassigned': True,
            'stopped_v1_campaign': str(parent), 'stopped_v1_actual_attempts': 2,
            'stopped_v1_admission_stop': binding(stop_path),
            'stopped_v1_unused_allowance_retained': 86, 'stopped_v1_evidence': stopped,
            'independent_stop_diagnosis': binding(diagnosis),
            'old_v1_continuation_forbidden': True,
            'current_repair_round_actual_attempt_ceiling_including_v1': 90}


def remaining_allocation(registrations):
    same = [r for r in registrations if r['allocation_id'] == ALLOCATION_ID]
    if any(r['allocation_limit'] != NEW_LIMIT for r in same):
        raise ValueError('allocation_must_not_reset_or_expand')
    return NEW_LIMIT - sum(r['physical_attempt_limit'] for r in same)


def validate_counts(candidate_count, baseline_count, scene_count):
    if (type(candidate_count) is not int or not 1 <= candidate_count <= CANDIDATE_LIMIT
            or type(baseline_count) is not int or not 0 <= baseline_count <= BASELINE_LIMIT
            or type(scene_count) is not int or not 1 <= scene_count <= 10
            or candidate_count + baseline_count > NEW_LIMIT):
        raise ValueError('bounded_scene_task_allocation_required')


def register_budget(root, parent, candidate_count, baseline_count, scene_count, registry=REGISTRY):
    validate_counts(candidate_count, baseline_count, scene_count)
    registry = Path(registry); registry.mkdir(parents=True, exist_ok=True)
    history = prior_lineage(parent)
    amount = candidate_count + baseline_count
    with (registry / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = [read(p) for p in sorted((registry / 'registrations').glob('*.json'))]
        for row in records:
            resolve(row['budget'])
        if amount > remaining_allocation(records):
            raise ValueError('allocation_reserved_do_not_change_directory_to_retry')
        budget = {'schema': VERSION, 'physical_attempt_limit': amount,
            'allocation_id': ALLOCATION_ID, 'allocation_limit': NEW_LIMIT,
            'candidate_attempt_limit': candidate_count, 'baseline_attempt_limit': baseline_count,
            'parallel_workers': 1, 'per_attempt_timeout_seconds': TIMEOUT_SECONDS,
            'api_limit': 0, 'gpu_jobs': 0, 'prior_lineage': history,
            'prior_scope_limit': history['prior_physical_limit'],
            'cumulative_scope_limit': history['prior_physical_limit'] + NEW_LIMIT,
            'retries_per_request': 0, 'unknown_attempts_must_be_recovered': True,
            'new_user_authorized_local_allocation': True, 'source_coverage': 'synthetic_only'}
        write(root / 'budget/limits.json', budget)
        receipt = {'schema': VERSION, 'allocation_id': ALLOCATION_ID, 'allocation_limit': NEW_LIMIT,
            'physical_attempt_limit': amount, 'root': str(root),
            'budget': binding(root / 'budget/limits.json'), 'registered_unix': time.time()}
        receipt_path = registry / 'registrations' / (digest([ALLOCATION_ID, str(root)]) + '.json')
        write(receipt_path, receipt)
    return binding(receipt_path), budget


def factory_inputs(factory):
    factory = path(factory); rows = read(factory / 'scenes.json')['scenes']
    if not 1 <= len(rows) <= 10 or len({r['scene_id'] for r in rows}) != len(rows):
        raise ValueError('finite_unique_scene_ids_required')
    cases = []
    for row in rows:
        ids = row['task_ids']
        if not 3 <= len(ids) <= 8 or len(set(ids)) != len(ids):
            raise ValueError('each_scene_requires_three_to_eight_unique_tasks')
        if row['requested_tasks'] != len(ids):
            raise ValueError('factory_request_denominator_mismatch')
        compiled = read(resolve(row['compiled_tasks']))['tasks'] if row['accepted_geometry'] else []
        tasks = {t['requested_task_id']: t for t in compiled}
        metadata = read(resolve(row['task_metadata']))['tasks'] if row.get('task_metadata') else {}
        request = read(resolve(row['request'])) if row.get('request') else None
        requested = {t['id']: t for t in request['tasks']} if request else {}
        if row['accepted_geometry'] and (set(tasks) != set(ids) or set(requested) != set(ids)):
            raise ValueError('accepted_factory_scene_must_bind_all_requested_tasks')
        for short in ids:
            meta = metadata.get(short, {})
            n = len(requested[short]['placements']) if short in requested else meta.get('operation_count')
            if n is not None and not 1 <= n <= 4:
                raise ValueError('one_to_four_distinct_object_operations_required')
            cases.append({'row': row, 'task': tasks.get(short), 'short': short,
                'operation_count': n, 'task_structure': meta.get('structure', short), 'metadata': meta})
    return rows, cases


def provisional_task(task, scene):
    """Give failure exports valid effectors; no reachability claim is made."""
    import numpy as np
    result = copy.deepcopy(task)
    objects = {o['id']: o for o in scene['objects']}
    roots = np.asarray(scene['robot_roots'])
    for op in result['operations']:
        if op.get('arm') is None:
            source = np.asarray(objects[op['object']]['position'])
            op['arm'] = int(np.argmin(np.linalg.norm(roots - source, axis=1)))
    return result


def prepare(factory, root, *, parent=PARENT, expected_scenes=10, expected_tasks=80):
    from resimpair_pickplace.runner import code, verify as verify_worker
    from resimpair_pickplace.whole_path_motion_v1 import VERSION as CONTROLLER
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    factory, root, parent = map(path, (factory, root, parent))
    if root.exists():
        raise ValueError('preserve_existing_campaign_use_status_or_run')
    if shutil.disk_usage(ROOT).free < 10 * 2**30:
        raise ValueError('insufficient_disk_preserve_history')
    rows, inputs = factory_inputs(factory)
    if len(rows) != expected_scenes or len(inputs) != expected_tasks:
        raise ValueError('frozen_factory_does_not_match_declared_experiment_size')
    baseline_scenes = {r['scene_id'] for r in rows[:4]}
    baseline_inputs = [i for i in inputs if i['row']['scene_id'] in baseline_scenes and i['short'] in BASELINE_TASKS]
    validate_counts(len(inputs), len(baseline_inputs), len(rows))
    parent_spec = verify_worker(parent / 'candidate')
    if parent_spec['candidate_motion_override'] != CONTROLLER:
        raise ValueError('retained_whole_path_parent_required')
    implementation = code()
    factory_implementation = read(factory / 'protocol.json')['implementation']
    factory_module = resolve(factory_implementation)
    if not factory_module.is_relative_to(ROOT / 'resimpair_tasks') or factory_module.suffix != '.py':
        raise ValueError('bound_repository_local_factory_implementation_required')
    implementation[str(factory_module.relative_to(ROOT))] = factory_implementation
    # A versioned repair may import its preserved predecessor. Freeze both,
    # while the actual entry point is always read from the factory protocol.
    for dependency in (ROOT / 'resimpair_tasks').glob('multitask_scene_factory_v*.py'):
        implementation[str(dependency.relative_to(ROOT))] = binding(dependency)
    for relative in ('resimpair_tasks/multitask_campaign_v2.py', 'resimpair_tasks/multitask_worker_v2.py',
                     'resimpair_tasks/multitask_campaign_v1.py', 'resimpair_tasks/multitask_worker_v1.py',
                     'resimpair_pickplace/rolling_pickplace_v1.py',
                     'resimpair_pickplace/native_mesh_access_v1.py',
                     'resimpair_tasks/headless_physics_v1.py',
                     'resimpair_tasks/asset_sampling_v3.py', 'resimpair_tasks/task_factory_v1.py',
                     'resimpair_tasks/spatial_factory_v2.py', 'resimpair_pickplace/rolling_pickplace_v2.py'):
        implementation[relative] = binding(ROOT / relative)
    receipt, budget = register_budget(root, parent, len(inputs), len(baseline_inputs), len(rows))
    manifest = []
    for arm, selected, policy in [('candidate', inputs, 'rolling_state_v2'),
                                  ('baseline', baseline_inputs, 'initial_state_v1')]:
        out = root / arm; cases = []
        for item in selected:
            row, short = item['row'], item['short']; case_id = row['scene_id'] + '_' + short
            directory = out / 'inputs' / case_id
            case = {'id': case_id, 'execution_id': case_id, 'scene_id': row['scene_id'],
                'eligible': bool(row['accepted_geometry']), 'partition': row['partition'],
                'experiment_role': 'paired_initial_state_baseline' if arm == 'baseline' else 'multitask_rolling_candidate',
                'source_ids': row['source_ids'], 'split_group': row['source_ids'],
                'source_request': row.get('request'), 'requested_task_id': short,
                'task_structure': item['task_structure'], 'operation_count': item['operation_count'],
                'task_metadata': row.get('task_metadata'), 'method': 'cudex',
                'source_coverage': 'synthetic_only', 'motion_planning_agent_participation': False,
                'initial_arm_binding': 'nearest_source_root_for_failure_export_only_not_qualification'}
            if case['eligible']:
                scene = read(resolve(row['compiled_scene'])); task = provisional_task(item['task'], scene)
                write(directory / 'task.json', task)
                case.update(scene=row['compiled_scene'], shared_scene=row['compiled_scene'],
                            task=binding(directory / 'task.json'), original_task_set=row['compiled_tasks'])
            else:
                case['preflight_reason'] = 'factory_geometry_rejected:' + str(row.get('failure'))
            write(directory / 'case.json', case); cases.append(case)
            manifest.append({'arm': arm, 'case_id': case_id, 'scene_id': row['scene_id'],
                'case': binding(directory / 'case.json'), 'directory': str(out / 'executions' / case_id)})
        write(out / 'experiment.json', {'schema': VERSION, 'budget': str(root / 'budget'),
            'code': implementation, 'legacy_code': core.implementation(), 'gpu': None, 'gpu_uuid': None,
            'timeout_seconds': TIMEOUT_SECONDS, 'api_limit': 0, 'render_backend': 'none',
            'candidate_motion_override': CONTROLLER, 'execution_selection_policy': policy,
            'motion_policy': parent_spec['motion_policy'], 'release_policy': parent_spec['release_policy'],
            'contact_policy': parent_spec['contact_policy'], 'parent_controller': binding(parent / 'candidate/experiment.json'),
            'default_promoted': False, 'motion_planning_agent_participation': False})
        write(out / 'rules.json', read(parent / 'candidate/rules.json'))
        write(out / 'cohort.json', {'cases': cases, 'selection_before_physics': True,
            'outcome_based_replacement': False, 'all_requested_tasks_retained': True})
        write(out / 'protocol.json', {'schema': VERSION, 'driver': binding(__file__),
            'worker': binding(ROOT / 'resimpair_tasks/multitask_worker_v2.py'), 'arm': arm,
            'factory_protocol': binding(factory / 'protocol.json'), 'factory_scenes': binding(factory / 'scenes.json'),
            'shared_budget': binding(root / 'budget/limits.json'), 'budget_registration': receipt,
            'original_rules': binding(parent / 'candidate/rules.json'),
            'execution_selection_policy': policy, 'api_calls': 0, 'gpu_jobs': 0,
            'independent_protocol_path': str(root / 'independent_protocol.json')})
        write(out / 'frozen_inputs.json', {name: binding(out / filename) for name, filename in
            [('experiment', 'experiment.json'), ('cohort', 'cohort.json'), ('protocol', 'protocol.json'), ('rules', 'rules.json')]})
    write(root / 'comparison_manifest.json', {'rows': manifest, 'paired_baseline_selection': {
        'scene_ids': [r['scene_id'] for r in rows[:4]], 'task_ids': list(BASELINE_TASKS),
        'selected_before_execution': True}})
    result = {'schema': VERSION, 'scenes': len(rows), 'candidate_requests': len(inputs),
        'baseline_requests': len(baseline_inputs), 'physical_attempt_limit': budget['physical_attempt_limit'],
        'cumulative_scope_limit': budget['cumulative_scope_limit'], 'factory': binding(factory / 'protocol.json'),
        'budget_registration': receipt, 'physical_attempts': 0, 'api_calls': 0, 'gpu_jobs': 0,
        'independent_protocol_required_before_run': str(root / 'independent_protocol.json')}
    write(root / 'prepared.json', result)
    verify(root, require_independent=False)
    return result


def verify(root, *, require_independent=True):
    from resimpair_pickplace.runner import verify as verify_worker
    root = path(root); budget = read(root / 'budget/limits.json')
    if (budget['allocation_id'] != ALLOCATION_ID or budget['allocation_limit'] != NEW_LIMIT
            or budget['prior_scope_limit'] != 122 or budget['cumulative_scope_limit'] != 210 or budget['parallel_workers'] != 1
            or budget['api_limit'] != 0 or budget['gpu_jobs'] != 0
            or budget['per_attempt_timeout_seconds'] != TIMEOUT_SECONDS
            or budget['physical_attempt_limit'] != budget['candidate_attempt_limit'] + budget['baseline_attempt_limit']):
        raise ValueError('frozen_finite_CPU_allocation_required')
    predecessor = prior_lineage(budget['prior_lineage']['stopped_v1_campaign'])
    if predecessor != budget['prior_lineage']:
        raise ValueError('stopped_predecessor_records_or_usage_changed')
    counts = {}
    for arm in ('candidate', 'baseline'):
        out = root / arm; spec = verify_worker(out); protocol = read(out / 'protocol.json')
        if spec['schema'] != VERSION or protocol['driver'] != binding(__file__):
            raise ValueError('multitask_driver_binding_changed')
        for ref in read(out / 'frozen_inputs.json').values(): resolve(ref)
        for key in ('factory_protocol', 'factory_scenes', 'shared_budget', 'budget_registration', 'original_rules', 'worker'):
            resolve(protocol[key])
        if read(out / 'rules.json') != read(resolve(protocol['original_rules'])):
            raise ValueError('native_acceptance_changed')
        rows = read(out / 'cohort.json')['cases']; counts[arm] = len(rows)
        for c in rows:
            if c.get('source_request'): resolve(c['source_request'])
            if c.get('task_metadata'): resolve(c['task_metadata'])
            if c['eligible']:
                resolve(c['scene']); resolve(c['task']); resolve(c['original_task_set'])
        if counts[arm] != budget[arm + '_attempt_limit']:
            raise ValueError('cohort_budget_denominator_changed')
    if require_independent:
        independent = root / 'independent_protocol.json'
        if not independent.exists(): raise ValueError('freeze_independent_protocol_before_execution')
        verify_independent_bindings(read(independent),
            [root / 'candidate/cohort.json', root / 'baseline/cohort.json', root / 'budget/limits.json'])
    return budget


def verify_independent_bindings(value, required):
    """Check actual structured evidence links, not hash text in commentary."""
    found = []
    def visit(item):
        if isinstance(item, dict):
            if isinstance(item.get('path'), str) and isinstance(item.get('sha256'), str):
                ref = {'path': item['path'], 'sha256': item['sha256']}
                found.append(binding(resolve(ref)))
            for child in item.values(): visit(child)
        elif isinstance(item, list):
            for child in item: visit(child)
    visit(value)
    if any(binding(p) not in found for p in required):
        raise ValueError('independent_protocol_must_bind_both_cohorts_and_budget')


def status(root):
    root = path(root); budget = verify(root, require_independent=False); rows = []
    for arm in ('candidate', 'baseline'):
        for case in read(root / arm / 'cohort.json')['cases']:
            directory = root / arm / 'executions' / case['id']
            terminal = read(directory / 'terminal.json') if (directory / 'terminal.json').exists() else None
            rows.append({'arm': arm, 'case_id': case['id'], 'scene_id': case['scene_id'],
                'state': 'terminal' if terminal else 'unknown_reserved' if (directory / 'reservation.json').exists() else 'pending',
                'terminal': terminal, 'eligible': case['eligible']})
    return {'schema': VERSION, 'rows': rows, 'requested': len(rows),
        'physical_reserved': len(list((root / 'budget/reservations').glob('*.json'))),
        'physical_attempt_limit': budget['physical_attempt_limit'],
        'terminal_requests': sum(r['terminal'] is not None for r in rows),
        'unknown_reserved': sum(r['state'] == 'unknown_reserved' for r in rows), 'api_calls': 0, 'gpu_jobs': 0}


def run(root, arm, limit):
    root = path(root); budget = verify(root)
    if arm not in ('candidate', 'baseline') or type(limit) is not int or not 1 <= limit <= budget[arm + '_attempt_limit']:
        raise ValueError('finite_registered_arm_launch_count_required')
    if shutil.disk_usage(root).free < 10 * 2**30: raise ValueError('insufficient_disk')
    launched = []; resolved = []; out = root / arm
    REGISTRY.mkdir(parents=True, exist_ok=True)
    with (REGISTRY / '.execution.lock').open('a+') as allocation_lock, (root / '.supervisor.lock').open('a+') as lock:
        # Partial allocations in distinct roots still share CPU concurrency 1.
        fcntl.flock(allocation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status(root)['unknown_reserved']:
            raise ValueError('unknown_attempt_recover_without_repeat')
        for case in read(out / 'cohort.json')['cases']:
            directory = out / 'executions' / case['id']
            if (directory / 'terminal.json').exists(): continue
            if len(resolved) >= limit: break
            if not case['eligible']:
                terminal = {'case': case['id'], 'physical_attempts': 0, 'returncode': 'preflight_rejected',
                    'reason': case['preflight_reason'], 'elapsed_seconds': 0, 'api_calls': 0, 'gpu_jobs': 0}
                write(directory / 'terminal.json', terminal); resolved.append(terminal); continue
            reservations = list((root / 'budget/reservations').glob('*.json'))
            used = len(reservations)
            used_arm = sum(read(p)['arm'] == arm for p in reservations)
            if used >= budget['physical_attempt_limit'] or used_arm >= budget[arm + '_attempt_limit']:
                raise ValueError('finite_shared_or_arm_budget_exhausted')
            reservation = {'case': case, 'campaign': str(out), 'started_unix': time.time(),
                'arm': arm, 'gpu_jobs': 0, 'api_calls': 0, 'protocol': binding(out / 'protocol.json'),
                'independent_protocol': binding(root / 'independent_protocol.json')}
            write(root / 'budget/reservations' / f'{used:04d}.json', reservation)
            write(directory / 'reservation.json', reservation)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT),
                RESIMPAIR_HEADLESS_RESERVATION=str(directory / 'reservation.json'))
            began = time.monotonic()
            with (directory / 'worker.log').open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', WORKER_MODULE,
                    '--root', str(out), '--case', case['id']], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                ticks = int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
                write(directory / 'process.json', {'pid': child.pid, 'started_unix': time.time(),
                    'start_ticks': ticks, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
                try: rc = child.wait(timeout=TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=8)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    rc = 'timeout'
            terminal = {'case': case['id'], 'physical_attempts': 1, 'returncode': rc,
                'elapsed_seconds': time.monotonic() - began, 'gpu_jobs': 0, 'api_calls': 0}
            write(directory / 'terminal.json', terminal); launched.append(terminal); resolved.append(terminal)
            print(json.dumps(terminal, ensure_ascii=False), flush=True)
            if rc != 0: break
    result = {'arm': arm, 'new_attempts': len(launched), 'resolved_requests': len(resolved),
              'elapsed_worker_seconds': sum(r['elapsed_seconds'] for r in launched)}
    write(root / 'run_receipts' / f'{time.time_ns()}.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--factory', type=Path, required=True)
    prep.add_argument('--root', type=Path, required=True); prep.add_argument('--parent', type=Path, default=PARENT)
    prep.add_argument('--expected-scenes', type=int, default=10); prep.add_argument('--expected-tasks', type=int, default=80)
    launch = sub.add_parser('run'); launch.add_argument('--root', type=Path, required=True)
    launch.add_argument('--arm', choices=('candidate', 'baseline'), required=True); launch.add_argument('--limit', type=int, required=True)
    stat = sub.add_parser('status'); stat.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.factory, args.root, parent=args.parent, expected_scenes=args.expected_scenes,
                     expected_tasks=args.expected_tasks) if args.command == 'prepare' else (
             run(args.root, args.arm, args.limit) if args.command == 'run' else status(args.root))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
