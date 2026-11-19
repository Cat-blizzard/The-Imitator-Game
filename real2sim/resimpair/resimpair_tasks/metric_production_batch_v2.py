"""Finite candidate batch: exact metric variants, capacity and same-arm layouts.

No agent, online pose optimizer, image model or per-asset controller is needed
for this preparation/execution interface. Qualified snapshots are inputs. The
independent physical auditor remains a separate command and rule owner.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from resimpair_grasp.common import binding, read, resolve, write

VERSION = 'metric_production_batch_v2'
GOALS = ('select_one', 'select_other_bin', 'collect_subset', 'distribute_groups', 'collect_all')


def prepare(campaign, name, bundle, scenes, seed, *, container_catalog=None,
            count_schedule=None, minimum=3, maximum=8, goals=('all',),
            corridor=True, release='supported_native_clearance'):
    from .metric_population_sampler_v3 import build
    from .metric_operation_corridor_v2 import qualify_scene
    from .production_integration_v4 import prepare_cohort as checked_prepare
    from .production_integration_v2 import prepare_cohort as baseline_prepare
    from resimpair_pickplace.requested_tasks import identity
    campaign = Path(campaign).resolve(); identity(name)
    if not goals or len(set(goals)) != len(goals) or (tuple(goals) != ('all',) and set(goals) - set(GOALS)):
        raise ValueError('explicit_unique_supported_goals_or_all_required')
    if release not in {'supported_native_clearance', 'legacy_supported'}:
        raise ValueError('explicit_supported_release_candidate_or_baseline_required')
    limits = read(campaign / 'budget/limits.json')
    allocated = sum(len(read(p)['cases']) for p in campaign.glob('*/cohort.json'))
    maximum_cases = scenes * (len(GOALS) if tuple(goals) == ('all',) else len(goals))
    if allocated + maximum_cases > limits['physical_attempt_limit']:
        raise ValueError('requested_goals_exceed_existing_cumulative_allocation')
    staging = campaign / 'prepared_inputs' / name
    if staging.exists() or (campaign / name).exists():
        raise ValueError('existing_batch_preserve_and_recover_do_not_resample')
    request = {'schema': VERSION, 'implementation': binding(__file__), 'bundle': binding(bundle),
        'container_catalog': binding(container_catalog) if container_catalog else None,
        'requested_scenes': scenes, 'seed': seed, 'count_schedule': count_schedule,
        'object_count_range': [minimum, maximum], 'requested_execution_goals': list(goals),
        'local_corridor_sampling': bool(corridor), 'release_policy': release,
        'maximum_physical_cases': maximum_cases, 'physical_attempts': 0, 'API_calls': 0,
        'cumulative_budget': binding(campaign / 'budget/limits.json'),
        'maximum_sampling_proposals_per_scene': 16, 'same_source_split_preserved': True}
    write(staging / 'preparation_request.json', request)
    sampled = build(bundle, staging / 'sampled', scenes=scenes, seed=seed,
        minimum=minimum, maximum=maximum, container_catalog=container_catalog,
        count_schedule=count_schedule, layout_qualifier=qualify_scene if corridor else None)
    factory = staging / 'sampled/factory'
    cases = []
    if factory.exists():
        for source in read(factory / 'cohort.json')['cases']:
            if tuple(goals) != ('all',) and source['requested_task_id'] not in goals:
                continue
            task = deepcopy(read(resolve(source['task'])))
            for op in task['operations']:
                op['release']['mode'] = 'supported_place'
                op['release']['allow_drop'] = False
            path = staging / 'release_contracts' / source['id'] / 'task.json'
            write(path, task)
            cases.append({**source, 'task': binding(path), 'parent_task': source['task'],
                'release_selection': release, 'batch_preparer': binding(__file__),
                'preparation_request': binding(staging / 'preparation_request.json'),
                'sampling_protocol': binding(staging / 'sampled/protocol.json')})
    if not cases:
        result = {'prepared': False, 'physical_cases': 0, 'reason': 'no_requested_goals_prepared'}
    else:
        driver = checked_prepare if release == 'supported_native_clearance' else baseline_prepare
        result = driver(campaign, name, cases, prefix_reuse=True, full_open=False)
    write(staging / 'preparation_result.json', {'schema': VERSION, 'request': request,
        'sampled': sampled, 'prepared': result, 'physical_cases': len(cases),
        'operator_interventions_per_scene': 0, 'physical_attempts': 0, 'API_calls': 0,
        'qualification_scope': 'geometry and optionally nominal local kinematics; full live paths and physical outcomes unmeasured'})
    return result


def run(cohort, limit=None):
    spec = read(Path(cohort) / 'experiment.json')
    if spec['schema'] == 'metric_asset_production_integration_v4':
        from .production_integration_v4 import run as execute
    elif spec['schema'] == 'metric_asset_production_integration_v2':
        from .production_integration_v2 import run as execute
    else:
        raise ValueError('unexpected_frozen_execution_driver')
    return execute(cohort, limit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--campaign', type=Path, required=True); q.add_argument('--name', required=True)
    q.add_argument('--bundle', type=Path, required=True); q.add_argument('--container-catalog', type=Path)
    q.add_argument('--scenes', type=int, required=True); q.add_argument('--seed', type=int, required=True)
    q.add_argument('--min-objects', type=int, default=3); q.add_argument('--max-objects', type=int, default=8)
    q.add_argument('--count-schedule', help='optional fixed comma-separated N per scene')
    q.add_argument('--goals', nargs='+', default=['all'])
    q.add_argument('--without-corridor', action='store_true', help='capacity-only sampling control')
    q.add_argument('--release-policy', choices=['supported_native_clearance', 'legacy_supported'],
                   default='supported_native_clearance')
    q = sub.add_parser('run'); q.add_argument('--cohort', type=Path, required=True); q.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.command == 'prepare':
        print(prepare(a.campaign, a.name, a.bundle, a.scenes, a.seed,
            container_catalog=a.container_catalog,
            count_schedule=[int(v) for v in a.count_schedule.split(',')] if a.count_schedule else None,
            minimum=a.min_objects, maximum=a.max_objects, goals=tuple(a.goals),
            corridor=not a.without_corridor, release=a.release_policy))
    else:
        print(run(a.cohort, a.limit))


if __name__ == '__main__':
    main()
