"""Prepare and run bounded, agent-free CPU pick/place batches from a factory.

Reuses asset_sampling_v3 outputs, appearance_v1, the existing whole-path
candidate worker and append-only supervisor. No new controller or acceptance
implementation is introduced. RGB/human-video export is a separate stage.
"""
import argparse
from pathlib import Path
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'agent_free_random_pickplace_candidate_v1'
DEFAULT_GOALS = ('relocate_selected', 'contain_selected')


def select_cases(factory, *, selection='first-scenes', scenes=3, goals=DEFAULT_GOALS):
    factory = Path(factory).resolve()
    rows = read(factory / 'scenes.json')['scenes']
    if selection not in {'first-scenes', 'coverage-pilot'}:
        raise ValueError('unknown_selection_mode')
    if type(scenes) is not int or not 1 <= scenes <= len(rows):
        raise ValueError('scene_count_exceeds_frozen_factory')
    if not goals or len(set(goals)) != len(goals):
        raise ValueError('unique_nonempty_goal_list_required')

    def available(row):
        if not row.get('request'):
            return set()
        return {t['id'] for t in read(resolve(row['request']))['tasks']}

    def first(predicate, reason):
        for row in rows:
            if predicate(row):
                return row
        raise ValueError(reason)

    if selection == 'coverage-pilot':
        development = first(lambda r: r['partition'] == 'development', 'development_scene_missing')
        heldout = first(lambda r: r['partition'] == 'heldout_asset', 'heldout_asset_scene_missing')
        collection = first(lambda r: 'collect_pair' in available(r), 'collection_scene_missing')
        specifications = [(development, DEFAULT_GOALS, 'new_asset_development'),
                          (heldout, DEFAULT_GOALS, 'new_asset_reserved'),
                          (collection, ('contain_selected', 'collect_pair'), 'new_asset_collection')]
    else:
        specifications = [(r, tuple(goals), 'new_asset_reserved' if r['partition'] == 'heldout_asset'
                           else 'new_asset_development') for r in rows[:scenes]]
    selected = []; identities = set()
    for row, requested, role in specifications:
        known = available(row)
        for goal in requested:
            if goal not in known:
                raise ValueError('requested_goal_absent_from_frozen_scene:' + row['scene_id'] + ':' + goal)
            case_id = row['scene_id'] + '_' + goal
            if case_id in identities:
                raise ValueError('pilot_scene_roles_overlap_or_duplicate_task:' + case_id)
            identities.add(case_id)
            directory = factory / 'scenes' / row['scene_id']
            selected.append({'case_id': case_id, 'scene_id': row['scene_id'], 'seed': row['seed'],
                'dataset_partition': row['partition'], 'experiment_role': role, 'source_ids': row['source_ids'],
                'source_request': row['request'], 'requested_task_id': goal,
                'accepted_geometry': row['accepted_geometry'], 'geometry_failure': row.get('failure'),
                'compiled_scene': binding(directory / 'compiled_scene.json') if row['accepted_geometry'] else None,
                'compiled_tasks': binding(directory / 'compiled_tasks.json') if row['accepted_geometry'] else None})
    return selected


def budget_spec(parent, physical_limit, selected_count):
    parent = Path(parent).resolve(); path = parent / 'budget/limits.json'; limits = read(path)
    if (type(physical_limit) is not int or not 1 <= physical_limit <= 96
            or selected_count > physical_limit or selected_count < 1):
        raise ValueError('finite_budget_must_cover_selected_requests_max96')
    if limits.get('api_limit') != 0 or limits.get('gpu_jobs') != 0:
        raise ValueError('CPU_zero_API_parent_required')
    parent_cap = limits['physical_attempt_limit']
    if type(parent_cap) is not int or parent_cap < 0:
        raise ValueError('invalid_parent_physical_limit')
    return {'schema': VERSION, 'physical_attempt_limit': physical_limit, 'parallel_workers': 1,
        'api_limit': 0, 'gpu_jobs': 0, 'per_attempt_timeout_seconds': limits['per_attempt_timeout_seconds'],
        'parent_budget': binding(path), 'prior_physics_lineage': binding(path),
        'parent_authorized_attempt_limit': parent_cap,
        'parent_reservations_observed_at_prepare': len(list((parent / 'budget/reservations').glob('*.json'))),
        'round_cumulative_attempt_limit_including_parent': parent_cap + physical_limit,
        'prior_history_before_parent': limits.get('prior_physics_lineage'),
        'no_budget_reset_or_retry': True, 'selection_precedes_physics': True}


def prepare(factory, root, parent, *, physical_limit, selection='first-scenes', scenes=3,
            goals=DEFAULT_GOALS, appearance=True, appearance_seed=2026102902):
    from resimpair_pickplace.runner import code, verify
    from resimpair_pickplace.cartesian_reachability import bind_selection
    from resimpair_pickplace.whole_path_motion_v1 import VERSION as CANDIDATE
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    from . import appearance_v1, asset_sampling_v3, whole_path_campaign_v1, headless_physics_v1
    factory, root, parent = (Path(p).resolve() for p in (factory, root, parent))
    if root.exists():
        raise ValueError('existing_campaign_resume_without_reprepare')
    if read(factory / 'protocol.json')['schema'] != asset_sampling_v3.VERSION:
        raise ValueError('expanded_frozen_factory_required')
    for field in ('implementation', 'index', 'containers', 'catalog', 'source_role_index',
                  'compiler', 'role_loader', 'layout_sampler'):
        resolve(read(factory / 'protocol.json')[field])
    parent_spec = verify(parent / 'candidate')
    if parent_spec['candidate_motion_override'] != CANDIDATE:
        raise ValueError('parent_uses_different_candidate_controller')
    selected = select_cases(factory, selection=selection, scenes=scenes, goals=goals)
    limits = budget_spec(parent, physical_limit, len(selected))
    if selection == 'coverage-pilot' and (len(selected) != 6 or physical_limit != 6):
        raise ValueError('coverage_pilot_requires_exactly_six_requests_and_budget')
    write(root / 'selection.json', {'schema': VERSION, 'factory': binding(factory / 'protocol.json'),
        'factory_scene_manifest': binding(factory / 'scenes.json'), 'selection': selection,
        'cases': selected, 'requested_goals': list(goals), 'selected_before_ik_and_physics': True,
        'outcome_based_replacement': False, 'source_partition_labels_unchanged': True})
    write(root / 'budget/limits.json', limits)
    extra = {str(Path(m.__file__).resolve().relative_to(ROOT)): binding(m.__file__)
             for m in (appearance_v1, asset_sampling_v3, whole_path_campaign_v1, headless_physics_v1)}
    extra['resimpair_tasks/random_pickplace_v1.py'] = binding(__file__)
    extra['resimpair.py'] = binding(ROOT / 'resimpair.py')
    out = root / 'candidate'
    write(out / 'experiment.json', {'schema': VERSION, 'budget': str(root / 'budget'),
        'code': {**code(), **extra}, 'legacy_code': core.implementation(), 'gpu': None, 'gpu_uuid': None,
        'timeout_seconds': limits['per_attempt_timeout_seconds'], 'api_limit': 0, 'render_backend': 'none',
        'candidate_motion_override': CANDIDATE, 'motion_policy': parent_spec['motion_policy'],
        'release_policy': parent_spec['release_policy'], 'contact_policy': parent_spec['contact_policy'],
        'parent_controller': binding(parent / 'candidate/experiment.json'), 'default_promoted': False,
        'appearance_sampled': appearance, 'motion_planning_agent_participation': False})
    write(out / 'rules.json', read(parent / 'candidate/rules.json'))
    write(out / 'protocol.json', {'schema': VERSION, 'driver': binding(whole_path_campaign_v1.__file__),
        'batch_driver': binding(__file__), 'arm': 'candidate', 'selection': binding(root / 'selection.json'),
        'shared_budget': binding(root / 'budget/limits.json'), 'original_rules': binding(parent / 'candidate/rules.json'),
        'appearance_catalog': binding(appearance_v1.DEFAULT_CATALOG) if appearance else None,
        'api_calls': 0, 'gpu_jobs': 0, 'controller_is_unpromoted_candidate': True})
    cases = []; appearances = {}; started = time.monotonic()
    for row in selected:
        folder = out / 'inputs' / row['case_id']
        case = {'id': row['case_id'], 'execution_id': row['case_id'], 'eligible': False,
            'partition': row['dataset_partition'], 'experiment_role': row['experiment_role'],
            'source_ids': row['source_ids'], 'split_group': row['source_ids'], 'source_request': row['source_request'],
            'requested_task_id': row['requested_task_id'], 'method': 'cudex', 'source_coverage': 'synthetic_only'}
        if not row['accepted_geometry']:
            case['preflight_reason'] = 'factory_geometry_rejected'
            case['geometry_failure'] = row['geometry_failure']
        else:
            scene = read(resolve(row['compiled_scene']))
            task = next(t for t in read(resolve(row['compiled_tasks']))['tasks']
                        if t['requested_task_id'] == row['requested_task_id'])
            write(folder / 'role_intent.json', task)
            endpoint_scene, assigned = bind_selection(scene, task, folder)
            case['arm_selection'] = binding(folder / 'arm_selection.json')
            if endpoint_scene is None:
                case['preflight_reason'] = 'no_single_arm_source_and_destination_grasp'
            else:
                # Keep the original cache for the candidate's complete-path
                # search. Appearance is sampled last and shared across goals.
                if appearance:
                    if row['scene_id'] not in appearances:
                        appearances[row['scene_id']] = appearance_v1.sample(scene, appearance_seed + row['seed'])
                    scene = appearances[row['scene_id']]
                write(folder / 'scene.json', scene); write(folder / 'task.json', assigned)
                case.update(eligible=True, scene=binding(folder / 'scene.json'), task=binding(folder / 'task.json'),
                            shared_scene=row['compiled_scene'])
        write(folder / 'case.json', case); cases.append(case)
        print({'case': case['id'], 'eligible': case['eligible']}, flush=True)
    write(out / 'cohort.json', {'cases': cases, 'selection': binding(root / 'selection.json'),
                              'outcome_based_replacement': False})
    write(out / 'frozen_inputs.json', {'experiment': binding(out / 'experiment.json'),
        'cohort': binding(out / 'cohort.json'), 'protocol': binding(out / 'protocol.json'),
        'selection': binding(root / 'selection.json'), 'budget': binding(root / 'budget/limits.json')})
    result = {'schema': VERSION, 'registered_requests': len(cases),
        'endpoint_eligible': sum(c['eligible'] for c in cases), 'physical_attempts': 0,
        'physical_attempt_limit': physical_limit, 'round_cumulative_attempt_limit_including_parent':
            limits['round_cumulative_attempt_limit_including_parent'],
        'seconds_prepare_cases': time.monotonic() - started, 'api_calls': 0, 'gpu_jobs': 0,
        'rendered_frames': 0, 'default_promoted': False}
    write(root / 'prepared.json', result)
    return result


def run(root, limit):
    from resimpair_pickplace.runner import verify
    from resimpair_pickplace.whole_path_motion_v1 import VERSION as CANDIDATE
    from . import whole_path_campaign_v1
    root = Path(root).resolve(); out = root / 'candidate'
    spec = verify(out); protocol = read(out / 'protocol.json')
    if spec['schema'] != VERSION or spec['candidate_motion_override'] != CANDIDATE:
        raise ValueError('random_pickplace_candidate_campaign_required')
    resolve(protocol['batch_driver']); resolve(protocol['original_rules'])
    limits = read(resolve(protocol['shared_budget'])); parent = read(resolve(limits['parent_budget']))
    if (limits['api_limit'] != 0 or limits['gpu_jobs'] != 0 or limits['parallel_workers'] != 1
            or limits['round_cumulative_attempt_limit_including_parent']
            != parent['physical_attempt_limit'] + limits['physical_attempt_limit']):
        raise ValueError('invalid_CPU_budget_lineage')
    if type(limit) is not int or not 1 <= limit <= limits['physical_attempt_limit']:
        raise ValueError('bounded_launch_count_required')
    return whole_path_campaign_v1.run(root, 'candidate', limit)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    for name in ('factory', 'root', 'parent'):
        prepare_parser.add_argument('--' + name, type=Path, required=True)
    prepare_parser.add_argument('--physical-limit', type=int, required=True)
    prepare_parser.add_argument('--selection', choices=('first-scenes', 'coverage-pilot'), default='first-scenes')
    prepare_parser.add_argument('--scenes', type=int, default=3)
    prepare_parser.add_argument('--goals', nargs='+', default=list(DEFAULT_GOALS))
    prepare_parser.add_argument('--no-appearance', action='store_true')
    prepare_parser.add_argument('--appearance-seed', type=int, default=2026102902)
    run_parser = sub.add_parser('run'); run_parser.add_argument('--root', type=Path, required=True)
    run_parser.add_argument('--limit', type=int, required=True)
    a = p.parse_args(argv)
    result = prepare(a.factory, a.root, a.parent, physical_limit=a.physical_limit,
        selection=a.selection, scenes=a.scenes, goals=a.goals, appearance=not a.no_appearance,
        appearance_seed=a.appearance_seed) if a.command == 'prepare' else run(a.root, a.limit)
    print(result)


if __name__ == '__main__':
    main()
