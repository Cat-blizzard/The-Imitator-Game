"""Agent-free finite metric scene programs, using supported placement by default.

Prepare takes a frozen admitted asset bundle and an existing cumulative CPU
campaign allocation. Execute and audit are separate commands: the generator
does not import, edit or manufacture the independent acceptance verdicts.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from resimpair_grasp.common import binding, read, resolve, write

VERSION = 'metric_production_batch_v1'


def prepare(campaign, name, bundle, scenes, seed, minimum=3, maximum=8, release='supported_place'):
    from .metric_population_sampler_v2 import build
    from .production_integration_v2 import prepare_cohort
    from resimpair_pickplace.requested_tasks import identity
    campaign = Path(campaign).resolve(); identity(name)
    if release not in {'supported_place', 'drop_from_above'}:
        raise ValueError('explicit_supported_or_low_drop_release_required')
    limits = read(campaign / 'budget/limits.json')
    used = sum(len(read(p)['cases']) for p in campaign.glob('*/cohort.json'))
    if used + scenes * 5 > limits['physical_attempt_limit']:
        raise ValueError('requested_scene_programs_exceed_registered_remaining_allocation')
    staging = campaign / 'prepared_inputs' / name
    if staging.exists() or (campaign / name).exists():
        raise ValueError('existing_batch_recover_without_resampling_or_reregistering')
    sampled = build(bundle, staging / 'sampled', scenes=scenes, seed=seed,
                    minimum=minimum, maximum=maximum)
    factory = staging / 'sampled/factory'
    if not factory.exists():
        write(staging / 'preparation_result.json', {'schema': VERSION, 'sampled': sampled,
            'physical_cases': 0, 'physical_attempts': 0, 'failure': 'all_scenes_geometry_rejected'})
        return {'prepared': False, 'sampled': sampled}
    cases = []
    for source in read(factory / 'cohort.json')['cases']:
        task = deepcopy(read(resolve(source['task'])))
        for op in task['operations']:
            is_drop = release == 'drop_from_above' and op['target']['support'] != 'table'
            op['release']['mode'] = 'drop_from_above' if is_drop else 'supported_place'
            op['release']['allow_drop'] = is_drop
        path = staging / 'release_contracts' / source['id'] / 'task.json'
        write(path, task)
        cases.append({**source, 'task': binding(path), 'parent_task': source['task'],
                      'release_selection': release, 'batch_preparer': binding(__file__)})
    result = prepare_cohort(campaign, name, cases, prefix_reuse=True, full_open=False)
    write(staging / 'preparation_result.json', {'schema': VERSION, 'sampled': sampled,
        'prepared': result, 'release_mode': release, 'operator_interventions_per_scene': 0,
        'release_selection_basis': 'supported retains 3/3 new-metric layouts and 4/4 three-source scene programs; local evidence only',
        'input_bundle': binding(bundle), 'physical_attempts': 0, 'API_calls': 0})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--campaign', type=Path, required=True); q.add_argument('--name', required=True)
    q.add_argument('--bundle', type=Path, required=True); q.add_argument('--scenes', type=int, required=True)
    q.add_argument('--seed', type=int, required=True)
    q.add_argument('--min-objects', type=int, default=3); q.add_argument('--max-objects', type=int, default=8)
    q.add_argument('--release-mode', choices=['supported_place', 'drop_from_above'], default='supported_place')
    q = sub.add_parser('run'); q.add_argument('--cohort', type=Path, required=True)
    q.add_argument('--limit', type=int)
    a = p.parse_args()
    if a.command == 'prepare':
        print(prepare(a.campaign, a.name, a.bundle, a.scenes, a.seed, a.min_objects, a.max_objects, a.release_mode))
    else:
        from .production_integration_v2 import run
        print(run(a.cohort, a.limit))


if __name__ == '__main__':
    main()
