"""Prepare-only adapter from the v5 IK-conditioned factory to the v1 worker.

The factory keeps its real schema and provenance. Saved, same-input endpoint
IK receipts are reused, not recomputed. This revision grants zero physical/API
allowance and cannot launch; the existing 26-attempt lineage remains intact.
"""
import argparse
import copy
from pathlib import Path

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from . import spatial_reachability_v5 as factory_v5
from . import random_pickplace_v1 as runtime_v1
from . import whole_path_campaign_v1 as execution

VERSION = 'IK_factory_runtime_adapter_v2_prepare_only'


def factory_chain(factory):
    factory = Path(factory).resolve(); p = read(factory / 'protocol.json')
    if p.get('schema') != factory_v5.VERSION:
        raise ValueError('actual_v5_factory_schema_required')
    refs = {'factory_protocol': binding(factory / 'protocol.json'),
            'factory_scenes': binding(factory / 'scenes.json'), 'factory_summary': binding(factory / 'summary.json')}
    for key in ('implementation', 'geometry_proposer', 'parent_factory', 'parent_scenes',
                'index', 'containers', 'compiler', 'endpoint_selector'):
        resolve(p[key]); refs['v5_' + key] = p[key]
    parent = read(resolve(p['parent_factory']))
    if parent.get('schema') != 'expanded_compatible_asset_sampling_v3':
        raise ValueError('v5_parent_asset_factory_required')
    for key in ('implementation', 'index', 'containers', 'catalog', 'source_role_index',
                'compiler', 'role_loader', 'layout_sampler'):
        resolve(parent[key]); refs['v3_' + key] = parent[key]
    for name, value in parent['snapshots'].items():
        resolve(value['snapshot']); refs['v3_snapshot_' + name] = value['snapshot']
    roles = read(resolve(parent['source_role_index']))
    if roles['catalog'] != parent['catalog'] or roles['pose_index'] != parent['index'] or roles['containers'] != parent['containers']:
        raise ValueError('parent_catalog_role_index_binding_mismatch')
    return refs


def zero_budget(lineage_budget, controller_parent):
    lineage_budget, controller_parent = Path(lineage_budget).resolve(), Path(controller_parent).resolve()
    latest = read(lineage_budget); parent_spec = read(controller_parent / 'experiment.json')
    integration = Path(parent_spec['budget']) / 'limits.json'
    refs = {'v1_comparison': latest['parent_comparison_budget'],
            'expanded_factory_integration': binding(integration), 'v2_regression': binding(lineage_budget)}
    allocations = {name: read(resolve(ref))['physical_attempt_limit'] for name, ref in refs.items()}
    if (allocations != latest['allocation_this_turn'] or sum(allocations.values()) != latest['this_turn_cumulative_scope_limit']
            or any(read(resolve(ref)).get('api_limit') != 0 or read(resolve(ref)).get('gpu_jobs') != 0 for ref in refs.values())):
        raise ValueError('existing_scope_budget_lineage_mismatch')
    return {'schema': VERSION, 'physical_attempt_limit': 0, 'launch_enabled': False,
        'api_limit': 0, 'gpu_jobs': 0, 'parallel_workers': 1,
        'per_attempt_timeout_seconds': latest['per_attempt_timeout_seconds'],
        'existing_scope_budget_references': refs, 'existing_scope_allocation': allocations,
        'scope_cumulative_attempt_limit': latest['this_turn_cumulative_scope_limit'],
        'existing_scope_reservations_observed': {
            name: len(list((resolve(ref).parent / 'reservations').glob('*.json'))) for name, ref in refs.items()},
        'prior_physics_lineage': binding(lineage_budget), 'new_physical_authorization': False,
        'unspent_old_allowance_not_reassigned': True}


def same_intent(assigned, intended):
    value = copy.deepcopy(assigned)
    for operation in value['operations']:
        operation['arm'] = None
    return value == intended


def prepare(factory, root, controller_parent, lineage_budget, *, scenes=3,
            goals=runtime_v1.DEFAULT_GOALS, appearance=True):
    from resimpair_pickplace.runner import code, verify
    from resimpair_pickplace.whole_path_motion_v1 import VERSION as CONTROLLER
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    from . import appearance_v1, headless_physics_v1
    factory, root, controller_parent = (Path(p).resolve() for p in (factory, root, controller_parent))
    if root.exists(): raise ValueError('existing_preparation_must_not_be_overwritten')
    dependencies = factory_chain(factory); rows = read(resolve(dependencies['factory_scenes']))['scenes']
    if type(scenes) is not int or not 1 <= scenes <= len(rows) or not goals or len(set(goals)) != len(goals):
        raise ValueError('bounded_scene_count_and_unique_goals_required')
    parent = verify(controller_parent)
    if parent['candidate_motion_override'] != CONTROLLER:
        raise ValueError('reuse_original_whole_path_v1_candidate_only')
    budget = zero_budget(lineage_budget, controller_parent); selected = []
    for row in rows[:scenes]:
        if not row['accepted_geometry_and_IK']:
            raise ValueError('requested_v5_scene_exhausted_or_not_endpoint_qualified')
        proof = read(resolve(row['accepted']['qualification']))
        if not proof['passed'] or set(goals) - set(row['required_goals']):
            raise ValueError('goal_not_in_accepted_v5_IK_set')
        for goal in goals:
            receipt = next(v for v in proof['checked'] if v['task'] == goal)
            if not receipt['passed']: raise ValueError('saved_endpoint_qualification_not_passed')
            report = resolve(receipt['report'])
            assigned_path = report.parent / 'task.json'
            scene_path = factory / 'scenes' / row['scene_id'] / 'compiled_scene.json'
            tasks_path = scene_path.with_name('compiled_tasks.json')
            intended = next(t for t in read(tasks_path)['tasks'] if t['requested_task_id'] == goal)
            assigned = read(assigned_path)
            if not same_intent(assigned, intended): raise ValueError('saved_endpoint_task_differs_from_original_intent')
            selected.append({'case_id': row['scene_id'] + '_' + goal, 'scene_id': row['scene_id'],
                'source_request': row['request'], 'compiled_scene': binding(scene_path),
                'assigned_task': binding(assigned_path), 'compiled_tasks': binding(tasks_path),
                'endpoint_report': receipt['report'], 'qualification': row['accepted']['qualification'],
                'goal': goal, 'partition': row['partition'], 'source_ids': row['source_ids']})
    write(root / 'selection.json', {'schema': VERSION, 'factory_schema': factory_v5.VERSION,
        'factory_protocol': dependencies['factory_protocol'], 'cases': selected,
        'selection': 'first N v5 scene rows and explicit goals; use existing same-input IK receipts',
        'IK_recomputed': False, 'physical_execution_requested': False})
    write(root / 'budget/limits.json', budget)
    out = root / 'candidate'
    extra = {str(Path(m.__file__).resolve().relative_to(ROOT)): binding(m.__file__) for m in (
        runtime_v1, execution, appearance_v1, headless_physics_v1, factory_v5)}
    extra['resimpair_tasks/random_pickplace_v2.py'] = binding(__file__)
    write(out / 'experiment.json', {'schema': VERSION, 'factory_schema': factory_v5.VERSION,
        'execution_schema': execution.VERSION, 'budget': str(root / 'budget'),
        'code': {**code(), **extra}, 'legacy_code': core.implementation(),
        'gpu': None, 'gpu_uuid': None, 'timeout_seconds': budget['per_attempt_timeout_seconds'],
        'api_limit': 0, 'render_backend': 'none', 'launch_enabled': False,
        'candidate_motion_override': CONTROLLER, 'motion_policy': parent['motion_policy'],
        'release_policy': parent['release_policy'], 'contact_policy': parent['contact_policy'],
        'parent_controller': binding(controller_parent / 'experiment.json'), 'default_promoted': False})
    write(out / 'rules.json', read(controller_parent / 'rules.json'))
    write(out / 'protocol.json', {'schema': VERSION, 'driver': binding(execution.__file__),
        'batch_driver': binding(__file__), 'factory_schema': factory_v5.VERSION,
        'factory_dependencies': dependencies, 'arm': 'candidate',
        'shared_budget': binding(root / 'budget/limits.json'),
        'original_rules': binding(controller_parent / 'rules.json'), 'launch_enabled': False})
    cases = []; scenes_by_id = {}
    for row in selected:
        folder = out / 'inputs' / row['case_id']
        if row['scene_id'] not in scenes_by_id:
            scene = read(resolve(row['compiled_scene']))
            if appearance: scene = appearance_v1.sample(scene, scene['seed'] + 20261029)
            scenes_by_id[row['scene_id']] = scene
        write(folder / 'scene.json', scenes_by_id[row['scene_id']])
        write(folder / 'task.json', read(resolve(row['assigned_task'])))
        write(folder / 'arm_selection.json', read(resolve(row['endpoint_report'])))
        case = {'id': row['case_id'], 'execution_id': row['case_id'], 'eligible': True,
            'partition': row['partition'], 'experiment_role': 'prepared_v5_endpoint_integration',
            'source_ids': row['source_ids'], 'split_group': row['source_ids'], 'source_request': row['source_request'],
            'requested_task_id': row['goal'], 'method': 'cudex', 'source_coverage': 'synthetic_only',
            'scene': binding(folder / 'scene.json'), 'task': binding(folder / 'task.json'),
            'shared_scene': row['compiled_scene'], 'arm_selection': binding(folder / 'arm_selection.json'),
            'qualification': row['qualification'], 'eligibility_basis': 'saved_same_input_endpoint_IK_only',
            'launch_enabled': False}
        write(folder / 'case.json', case); cases.append(case)
    write(out / 'cohort.json', {'cases': cases, 'selection': binding(root / 'selection.json'),
                              'outcome_based_replacement': False, 'launch_enabled': False})
    write(out / 'frozen_inputs.json', {'experiment': binding(out / 'experiment.json'),
        'cohort': binding(out / 'cohort.json'), 'protocol': binding(out / 'protocol.json'),
        'budget': binding(root / 'budget/limits.json'), 'selection': binding(root / 'selection.json'), **dependencies})
    result = {'schema': VERSION, 'factory_schema': factory_v5.VERSION, 'registered_requests': len(cases),
        'saved_endpoint_eligible': len(cases), 'new_endpoint_IK_calls': 0,
        'physical_attempt_limit': 0, 'physical_attempts': 0, 'launch_enabled': False,
        'scope_cumulative_attempt_limit': budget['scope_cumulative_attempt_limit'],
        'api_calls': 0, 'gpu_jobs': 0, 'default_promoted': False}
    write(root / 'prepared.json', result)
    return result


def run(root, limit):
    from resimpair_pickplace.runner import verify
    root = Path(root).resolve(); out = root / 'candidate'; verify(out)
    for bound in read(out / 'frozen_inputs.json').values(): resolve(bound)
    budget = read(root / 'budget/limits.json')
    if not budget.get('launch_enabled') or budget['physical_attempt_limit'] <= 0:
        raise ValueError('prepare_only_campaign_has_zero_physical_authorization')
    # Reuse the existing worker/supervisor if a separately approved future
    # revision supplies a new frozen campaign. This preparation cannot do so.
    return execution.run(root, 'candidate', limit)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('factory', 'root', 'controller-parent', 'lineage-budget'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--scenes', type=int, default=3); prep.add_argument('--goals', nargs='+', default=list(runtime_v1.DEFAULT_GOALS))
    prep.add_argument('--no-appearance', action='store_true')
    run_p = sub.add_parser('run'); run_p.add_argument('--root', type=Path, required=True); run_p.add_argument('--limit', type=int, required=True)
    a = p.parse_args()
    print(prepare(a.factory, a.root, a.controller_parent, a.lineage_budget, scenes=a.scenes,
        goals=a.goals, appearance=not a.no_appearance) if a.command == 'prepare' else run(a.root, a.limit))
