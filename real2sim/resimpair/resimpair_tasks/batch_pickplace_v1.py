"""One entry for cached assets -> random scenes -> CPU trajectories -> audit.

This integrates existing algorithms; it does not promote the whole-path
candidate or invent physical success from a pose cache. Preparation snapshots
live catalogs. Physics is finite, agent-free and API-free. Rendering and human
generation consume the separately audited source-execution manifest.
"""
import argparse
import copy
import fcntl
import importlib.util
from pathlib import Path
import shutil
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import asset_sampling_v3 as assets
from . import spatial_reachability_v6 as layout
from . import whole_path_campaign_v1 as execution

VERSION = 'unified_cached_pickplace_batch_v1'
DEFAULT_PROFILE = ROOT / 'configs/batch_pickplace/expanded_candidate_v1.json'
REGISTRY = ROOT / 'runtime/batch_pickplace_registry_v1'
REVIEWER = ROOT / 'validation/pickplace_automation_20261029/independent/review_executions.py'


def path(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def validate_profile(p, scenes, goals, physical_limit):
    if p.get('schema') != VERSION or p.get('controller_status') != 'opt_in_candidate':
        raise ValueError('versioned_candidate_profile_required')
    if (type(scenes) is not int or not 1 <= scenes <= 10000 or not goals
            or len(set(goals)) != len(goals) or 'all' in goals):
        raise ValueError('explicit_finite_scene_count_and_goals_required')
    if (type(physical_limit) is not int or not 1 <= physical_limit <= p['allocation_physical_limit']
            or physical_limit != scenes * len(goals)):
        raise ValueError('one_finite_attempt_per_requested_goal_required')
    if (p.get('api_limit') != 0 or p.get('gpu_jobs') != 0 or p.get('parallel_workers') != 1
            or not 1 <= p['maximum_layout_proposals'] <= 16
            or not 1 <= p['per_attempt_timeout_seconds'] <= 240):
        raise ValueError('bounded_CPU_zero_API_profile_required')
    return p


def historical_budget(parent):
    budget_path = parent / 'budget/limits.json'; budget = read(budget_path)
    refs = {**budget['existing_scope_budget_references'], 'v5_runtime_integration': binding(budget_path)}
    limits = {k: read(resolve(v))['physical_attempt_limit'] for k, v in refs.items()}
    used = {k: len(list((resolve(v).parent / 'reservations').glob('*.json'))) for k, v in refs.items()}
    if sum(limits.values()) != budget['scope_cumulative_attempt_limit']:
        raise ValueError('historical_allocation_does_not_reconcile')
    if any(read(resolve(v)).get('api_limit') != 0 or read(resolve(v)).get('gpu_jobs') != 0 for v in refs.values()):
        raise ValueError('historical_CPU_budget_required')
    return {'budget_references': refs, 'allocation': limits, 'reservations_observed': used,
            'prior_physical_limit': sum(limits.values()), 'prior_observed_attempts': sum(used.values())}


def remaining_allocation(profile, registered):
    same = [r for r in registered if r['allocation_id'] == profile['allocation_id']]
    if any(r['allocation_limit'] != profile['allocation_physical_limit'] for r in same):
        raise ValueError('same_allocation_must_not_reset_or_expand_limit')
    return profile['allocation_physical_limit'] - sum(r['physical_attempt_limit'] for r in same)


def register_budget(root, profile, parent, physical_limit, registry=REGISTRY):
    registry = Path(registry); registry.mkdir(parents=True, exist_ok=True)
    history = historical_budget(parent)
    with (registry / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registered = []
        for entry in sorted((registry / 'registrations').glob('*.json')):
            value = read(entry); resolve(value['budget']); registered.append(value)
        if physical_limit > remaining_allocation(profile, registered):
            raise ValueError('allocation_already_reserved_do_not_change_output_directory_to_retry')
        budget = {'schema': VERSION, 'physical_attempt_limit': physical_limit,
            'api_limit': 0, 'gpu_jobs': 0, 'parallel_workers': 1,
            'per_attempt_timeout_seconds': profile['per_attempt_timeout_seconds'],
            'allocation_id': profile['allocation_id'], 'allocation_limit': profile['allocation_physical_limit'],
            'historical_lineage': history,
            'cumulative_scope_limit': history['prior_physical_limit'] + profile['allocation_physical_limit'],
            'prior_batch_registration_count': len(registered),
            'new_local_validation_allowance': True, 'unspent_old_allowance_not_reassigned': True,
            'retries_per_request': 0, 'unknown_attempts_must_be_recovered': True,
            'real_source_coverage': 'synthetic_only'}
        write(root / 'budget/limits.json', budget)
        receipt = {'schema': VERSION, 'allocation_id': profile['allocation_id'],
            'allocation_limit': profile['allocation_physical_limit'], 'physical_attempt_limit': physical_limit,
            'root': str(root), 'budget': binding(root / 'budget/limits.json'),
            'profile': binding(root / 'profile.json'), 'registered_unix': time.time()}
        receipt_path = registry / 'registrations' / (digest({'root': str(root), 'allocation': profile['allocation_id']}) + '.json')
        write(receipt_path, receipt)
    return binding(receipt_path), budget


def prepare(root, profile=DEFAULT_PROFILE, *, scenes=None, goals=None, seed=None, physical_limit=None):
    from resimpair_pickplace.runner import code, verify as verify_worker
    from resimpair_pickplace.whole_path_motion_v1 import VERSION as CONTROLLER
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    from . import headless_physics_v1
    root, profile = path(root), path(profile)
    if root.exists(): raise ValueError('preserve_existing_batch_use_run_status_or_audit')
    p = read(profile)
    scenes = p['scenes'] if scenes is None else scenes
    goals = p['goals'] if goals is None else list(goals)
    seed = p['seed'] if seed is None else seed
    physical_limit = scenes * len(goals) if physical_limit is None else physical_limit
    validate_profile(p, scenes, goals, physical_limit)
    if type(seed) is not int or seed < 0: raise ValueError('nonnegative_integer_seed_required')
    parent = path(p['parent_execution']); parent_spec = verify_worker(parent / 'candidate')
    if parent_spec['candidate_motion_override'] != CONTROLLER:
        raise ValueError('qualified_parent_whole_path_controller_required')
    if shutil.disk_usage(root.parent if root.parent.exists() else ROOT).free < 10 * 2**30:
        raise ValueError('insufficient_disk_preserve_historical_data')
    write(root / 'profile.json', p)
    receipt, budget = register_budget(root, p, parent, physical_limit)
    started = time.monotonic(); timings = {}
    step = time.monotonic()
    inventory = assets.build(path(p['pose_index']), path(p['containers']), path(p['static_index']), root / 'factory',
        scenes=scenes, seed=seed, heldout_fraction=p['heldout_fraction'], containment_mode=p['containment_mode'])
    timings['asset_selection_geometry_compile_seconds'] = time.monotonic() - step
    step = time.monotonic()
    layout_summary = layout.build(root / 'factory', root / 'layouts', scenes=scenes, goals=goals,
        seed=seed + 100000, maximum_proposals=p['maximum_layout_proposals'], radius=p['sampling_radius_m'])
    timings['layout_proposals_and_endpoint_IK_seconds'] = time.monotonic() - step
    out = root / 'candidate'; cases = []; expected = []; selection = []
    for row in read(root / 'layouts/scenes.json')['scenes']:
        proof = read(resolve(row['accepted']['qualification'])) if row['accepted_geometry_and_IK'] else None
        source = root / 'layouts/scenes' / row['scene_id']
        for goal in row['required_goals']:
            case_id = row['scene_id'] + '_' + goal; directory = out / 'inputs' / case_id
            case = {'id': case_id, 'execution_id': case_id, 'eligible': row['accepted_geometry_and_IK'],
                'partition': row['partition'], 'experiment_role': 'new_batch_entry_integration',
                'source_ids': row['source_ids'], 'split_group': row['source_ids'],
                'source_request': row['request'] if row['accepted_geometry_and_IK'] else row['parent_request'],
                'requested_task_id': goal, 'method': 'cudex', 'source_coverage': 'synthetic_only',
                'layout_attempts': row['attempts'], 'motion_planning_agent_participation': False}
            if case['eligible']:
                endpoint = next(v for v in proof['checked'] if v['task'] == goal)
                if endpoint['passed'] is not True: raise ValueError('endpoint_receipt_missing')
                assigned = read(resolve(endpoint['report']).parent / 'task.json')
                original = next(t for t in read(source / 'compiled_tasks.json')['tasks'] if t['requested_task_id'] == goal)
                unassigned = copy.deepcopy(assigned)
                for operation in unassigned['operations']: operation['arm'] = None
                if unassigned != original: raise ValueError('saved_IK_assignment_changed_original_goal')
                write(directory / 'scene.json', read(source / 'compiled_scene.json'))
                write(directory / 'task.json', assigned)
                case.update(scene=binding(directory / 'scene.json'), task=binding(directory / 'task.json'),
                    shared_scene=binding(source / 'compiled_scene.json'), arm_selection=endpoint['report'],
                    qualification=row['accepted']['qualification'])
            else:
                case['preflight_reason'] = 'bounded_layout_endpoint_exhausted_or_source_rejected'
            write(directory / 'case.json', case); cases.append(case)
            expected.append({'case_id': case_id, 'source_request': case['source_request'], 'requested_task_id': goal,
                'dataset_partition': case['partition'], 'experiment_role': case['experiment_role']})
            selection.append({'case_id': case_id, 'case': binding(directory / 'case.json'),
                'directory': str(out / 'executions' / case_id), 'scene_id': row['scene_id']})
    write(root / 'selection.json', {'rows': selection, 'selected_before_physics': True, 'outcome_based_replacement': False})
    write(root / 'evaluation_cases.json', {'cases': expected, 'selected_before_execution': True})
    extra = {str(Path(m.__file__).relative_to(ROOT)): binding(m.__file__) for m in (assets, layout, execution, headless_physics_v1)}
    extra['resimpair_tasks/batch_pickplace_v1.py'] = binding(__file__)
    extra['resimpair_tasks/spatial_reachability_v5.py'] = binding(layout.retained.__file__)
    extra['resimpair_tasks/spatial_reachability_v4.py'] = binding(layout.geometry.__file__)
    write(out / 'experiment.json', {'schema': VERSION, 'budget': str(root / 'budget'), 'code': {**code(), **extra},
        'legacy_code': core.implementation(), 'gpu': None, 'gpu_uuid': None, 'api_limit': 0,
        'timeout_seconds': budget['per_attempt_timeout_seconds'], 'render_backend': 'none',
        'candidate_motion_override': CONTROLLER, 'motion_policy': parent_spec['motion_policy'],
        'release_policy': parent_spec['release_policy'], 'contact_policy': parent_spec['contact_policy'],
        'parent_controller': binding(parent / 'candidate/experiment.json'), 'default_promoted': False,
        'controller_status': 'opt_in_candidate', 'motion_planning_agent_participation': False})
    write(out / 'rules.json', read(parent / 'candidate/rules.json'))
    protocol = {'schema': VERSION, 'driver': binding(execution.__file__), 'batch_driver': binding(__file__),
        'shared_budget': binding(root / 'budget/limits.json'), 'budget_registration': receipt,
        'profile': binding(root / 'profile.json'), 'original_rules': binding(parent / 'candidate/rules.json'),
        'factory_protocol': binding(root / 'factory/protocol.json'), 'layout_protocol': binding(root / 'layouts/protocol.json'),
        'selection': binding(root / 'selection.json'), 'evaluation_cases': binding(root / 'evaluation_cases.json'),
        'independent_reviewer_implementation': binding(REVIEWER), 'arm': 'candidate',
        'api_calls': 0, 'gpu_jobs': 0, 'controller_is_unpromoted_candidate': True}
    write(out / 'protocol.json', protocol)
    write(out / 'cohort.json', {'cases': cases, 'selection': binding(root / 'selection.json'), 'outcome_based_replacement': False})
    write(out / 'frozen_inputs.json', {'experiment': binding(out / 'experiment.json'), 'cohort': binding(out / 'cohort.json'),
        'protocol': binding(out / 'protocol.json'), 'rules': binding(out / 'rules.json'),
        **{k: protocol[k] for k in ('shared_budget', 'budget_registration', 'profile', 'original_rules',
            'factory_protocol', 'layout_protocol', 'selection', 'evaluation_cases', 'independent_reviewer_implementation')}})
    timings['total_prepare_seconds'] = time.monotonic() - started
    result = {'schema': VERSION, 'registered_requests': len(cases), 'scenes': scenes,
        'endpoint_eligible': sum(c['eligible'] for c in cases), 'physical_attempt_limit': physical_limit,
        'physical_attempts': 0, 'cumulative_scope_limit': budget['cumulative_scope_limit'],
        'inventory': read(root / 'factory/inventory.json'), 'asset_factory': inventory,
        'layout_summary': layout_summary, 'timing': timings, 'api_calls': 0, 'gpu_jobs': 0,
        'render_status': 'not_run', 'human_generation_status': 'not_run', 'default_promoted': False}
    write(root / 'prepared.json', result)
    verify(root)
    return {k: result[k] for k in ('registered_requests', 'scenes', 'endpoint_eligible', 'physical_attempt_limit', 'timing')}


def verify(root):
    from resimpair_pickplace.runner import verify as verify_worker
    root = path(root); out = root / 'candidate'; spec = verify_worker(out)
    protocol = read(out / 'protocol.json')
    if spec['schema'] != VERSION or protocol['batch_driver'] != binding(__file__):
        raise ValueError('unified_batch_code_mismatch')
    for ref in read(out / 'frozen_inputs.json').values(): resolve(ref)
    if read(out / 'rules.json') != read(resolve(protocol['original_rules'])):
        raise ValueError('retained_rules_changed')
    budget = read(resolve(protocol['shared_budget'])); receipt = read(resolve(protocol['budget_registration']))
    if (receipt['budget'] != protocol['shared_budget'] or receipt['root'] != str(root)
            or budget['physical_attempt_limit'] != receipt['physical_attempt_limit']
            or budget['api_limit'] != 0 or budget['gpu_jobs'] != 0 or budget['parallel_workers'] != 1):
        raise ValueError('finite_registered_CPU_budget_required')
    return protocol, budget


def status(root):
    root = path(root); protocol, budget = verify(root); rows = []
    for case in read(root / 'candidate/cohort.json')['cases']:
        folder = root / 'candidate/executions' / case['id']; terminal = folder / 'terminal.json'
        state = read(terminal) if terminal.exists() else None
        rows.append({'case_id': case['id'], 'eligible': case['eligible'],
            'state': 'terminal' if state else 'unknown_reserved' if (folder / 'reservation.json').exists() else 'pending',
            'terminal': state, 'execution_directory': str(folder)})
    return {'schema': VERSION, 'requested': len(rows), 'rows': rows,
        'physical_reserved': len(list((root / 'budget/reservations').glob('*.json'))),
        'physical_attempt_limit': budget['physical_attempt_limit'],
        'physical_terminal': sum(r['terminal'].get('physical_attempts', 0) for r in rows if r['terminal']),
        'preflight_rejected': sum(r['terminal'].get('returncode') == 'preflight_rejected' for r in rows if r['terminal']),
        'unknown_reserved': sum(r['state'] == 'unknown_reserved' for r in rows),
        'api_calls': 0, 'gpu_jobs': 0, 'default_promoted': False}


def run(root, limit):
    root = path(root); _, budget = verify(root)
    if type(limit) is not int or not 1 <= limit <= budget['physical_attempt_limit']:
        raise ValueError('finite_launch_count_required')
    if status(root)['unknown_reserved']: raise ValueError('recover_unknown_attempt_without_resubmit')
    if shutil.disk_usage(root).free < 10 * 2**30: raise ValueError('insufficient_disk')
    started = time.monotonic(); stamp = time.time_ns()
    result = execution.run(root, 'candidate', limit)
    write(root / 'stages' / f'{stamp}_physics.json', {'requested_limit': limit, 'result': result,
        'seconds': time.monotonic() - started, 'api_calls': 0, 'gpu_jobs': 0})
    return result


def audit(root):
    root = path(root); protocol, _ = verify(root); started = time.monotonic()
    reviewer_path = resolve(protocol['independent_reviewer_implementation'])
    module_spec = importlib.util.spec_from_file_location('retained_batch_physics_review', reviewer_path)
    reviewer = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(reviewer)
    expected = {r['case_id']: r for r in read(resolve(protocol['evaluation_cases']))['cases']}
    rows = []; sources = []
    for case in read(root / 'candidate/cohort.json')['cases']:
        directory = root / 'candidate/executions' / case['id']
        if not (directory / 'terminal.json').exists(): continue
        row = reviewer.review(directory, case, expected[case['id']]); rows.append(row)
        if row['physical_subchecks_all_passed']:
            sources.append({'case_id': case['id'], 'execution_directory': str(directory),
                'case': binding(root / 'candidate/inputs' / case['id'] / 'case.json'),
                'execution_bundle': binding(directory / 'execution_bundle.json'),
                'deferred_render_manifest': binding(directory / 'deferred_render_manifest.json'),
                'source_request': case['source_request'], 'partition': case['partition'],
                'split_group': case['split_group'], 'source_coverage': 'synthetic_only',
                'physical_checks_passed': True, 'full_pair_qualified': row['full_pair_qualified']})
    report = {'schema': VERSION, 'protocol': binding(root / 'candidate/protocol.json'),
        'reviewer': protocol['independent_reviewer_implementation'], 'rows': rows,
        'physical_attempts': sum(r['physical_attempts'] for r in rows),
        'physical_passed': sum(r['physical_subchecks_all_passed'] for r in rows),
        'full_pairs_qualified': sum(r['full_pair_qualified'] for r in rows),
        'successful_source_executions': sources, 'render_status': 'not_run', 'human_status': 'not_run',
        'rules_unchanged': True, 'api_calls': 0, 'gpu_jobs': 0, 'default_promoted': False}
    destination = root / 'audits' / ('physical_' + digest(report)[:16] + '.json'); write(destination, report)
    write(root / 'stages' / f'{time.time_ns()}_audit.json', {'report': binding(destination),
        'seconds': time.monotonic() - started, 'api_calls': 0, 'gpu_jobs': 0})
    return {'report': binding(destination), 'physical_attempts': report['physical_attempts'],
            'physical_passed': report['physical_passed'], 'full_pairs_qualified': report['full_pairs_qualified']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    prep.add_argument('--scenes', type=int); prep.add_argument('--goals', nargs='+'); prep.add_argument('--seed', type=int)
    prep.add_argument('--physical-limit', type=int)
    for name in ('run', 'audit', 'status'):
        command = sub.add_parser(name); command.add_argument('--root', type=Path, required=True)
        if name == 'run': command.add_argument('--limit', type=int, required=True)
    a = p.parse_args(argv)
    if a.command == 'prepare': result = prepare(a.root, a.profile, scenes=a.scenes, goals=a.goals, seed=a.seed, physical_limit=a.physical_limit)
    elif a.command == 'run': result = run(a.root, a.limit)
    elif a.command == 'audit': result = audit(a.root)
    else: result = status(a.root)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
