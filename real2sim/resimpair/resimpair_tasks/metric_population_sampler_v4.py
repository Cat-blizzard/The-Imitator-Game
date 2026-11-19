"""Frozen v3 sampling semantics with complete generation-contract scene IDs.

This version changes names/provenance only. Sampling, geometry, source splits,
callback behavior and the exact-container factory are reused without mutation.
An ID identifies a declared generation contract; the actual compiled scene gets
its own fingerprint, so reproducibility is checked rather than assumed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected
from . import metric_population_sampler_v3 as retained

VERSION = 'metric_population_multitask_sampler_v4'
factory = retained.factory
previous = retained.previous
admitted_pool = retained.admitted_pool
_consumer_bins = retained._consumer_bins
capacity_select = retained.capacity_select
propose = retained.propose
callback_binding = retained.callback_binding
_callback = retained._callback


def generation_contract(bundle, *, scenes, seed, minimum, maximum, proposals,
                        container_catalog, count_schedule, hooks):
    """Canonical data/config bindings; output folders and clocks are excluded."""
    dependencies = [__file__, retained.__file__, previous.__file__, factory.__file__,
                    factory.previous.__file__, factory.retained.__file__]
    source_root = Path(__file__).resolve().parent.parent
    dependencies += [source_root / 'resimpair_grasp/scenes.py',
                     source_root / 'corl2026_workshop/experiments/autonomous_pipeline_v4/library.py']
    effective = list(count_schedule) if count_schedule is not None else [
        int(np.random.default_rng(seed + i).integers(minimum, maximum + 1)) for i in range(scenes)]
    return {'schema': 'metric_scene_generation_contract_v1', 'sampler': VERSION,
        'bundle': binding(bundle),
        'container_catalog': binding(container_catalog) if container_catalog is not None else None,
        'seed': int(seed), 'requested_scenes': int(scenes),
        'object_count_range': [minimum, maximum], 'declared_count_schedule': deepcopy(count_schedule),
        'effective_count_schedule': effective,
        'maximum_geometry_proposals_per_scene': int(proposals),
        'hooks': deepcopy(hooks), 'implementation_dependencies': [binding(x) for x in dependencies],
        'runtime_versions': {'python': list(sys.version_info[:3]), 'numpy': np.__version__},
        'clutter_seed_rule': 'seed + number of previously accepted scenes; entire schedule is bound',
        'source_partition_policy': 'retain original immutable bundle source partitions'}


def scene_identity(contract, scene_index, requested_count):
    if not 0 <= scene_index < contract['requested_scenes']:
        raise ValueError('scene_index_outside_generation_contract')
    if requested_count != contract['effective_count_schedule'][scene_index]:
        raise ValueError('requested_N_must_match_bound_count_schedule')
    return 'metric_group_' + digest({'contract_sha256': digest(contract),
        'scene_index': int(scene_index), 'requested_object_count': int(requested_count)})[:32]


def actual_fingerprints(output, contract):
    """Actual bytes and ID-independent contents remain separate from names."""
    output = Path(output)
    rows_path = output / 'factory/scenes.json'
    rows = read(rows_path)['scenes'] if rows_path.is_file() else []
    cohort_path = output / 'factory/cohort.json'
    cases = read(cohort_path)['cases'] if cohort_path.is_file() else []
    result = []
    for row in rows:
        if not row['compiled']:
            continue
        scene = read(resolve(row['scene']))
        content = deepcopy(scene)
        for key in ['id', 'scene_instance_id']:
            content.pop(key, None)
        tasks = []
        task_refs = []
        for case in cases:
            if case['scene'] != row['scene']:
                continue
            task_refs.append(case['task'])
            task = read(resolve(case['task']))
            task.pop('id', None)
            tasks.append(task)
        result.append({'scene_id': row['scene_id'], 'generation_contract_sha256': digest(contract),
            'actual_scene': row['scene'], 'actual_scene_sha256': row['scene']['sha256'],
            'scene_content_without_instance_names_sha256': digest(content),
            'actual_tasks': task_refs, 'task_content_without_instance_names_sha256': digest(tasks),
            'physical_success': 'unmeasured'})
    write(output / 'scene_fingerprints.json', {'schema': 'compiled_metric_scene_fingerprints_v1',
        'generation_contract_sha256': digest(contract), 'scenes': result,
        'scope': 'compiled scene and tasks, not executed state/action trajectories',
        'identity_claim': 'generation-contract identity; actual output fingerprint checked separately'})
    return binding(output / 'scene_fingerprints.json')


def build(bundle, output, *, scenes, seed, minimum=3, maximum=8, proposals=16,
          container_catalog=None, count_schedule=None, object_pose_proposal=None, layout_qualifier=None):
    if not (1 <= scenes <= 1000 and 3 <= minimum <= maximum <= 8 and 1 <= proposals <= 16):
        raise ValueError('bounded_scene_and_object_and_proposal_counts_required')
    if count_schedule is not None and (len(count_schedule) != scenes or any(
            not isinstance(n, int) or isinstance(n, bool) or not minimum <= n <= maximum for n in count_schedule)):
        raise ValueError('fixed_count_schedule_must_match_requested_scenes_and_range')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_sampling_failures')
    hooks = {'object_pose_proposal': callback_binding(object_pose_proposal),
             'layout_qualifier': callback_binding(layout_qualifier)}
    b, entries, source_pool, bins = admitted_pool(bundle)
    bins = _consumer_bins(bins, container_catalog)
    output.mkdir(parents=True)
    static = factory.static_source_catalog(resolve(b['static_catalog']), output / 'preflight_static')
    start = time.monotonic(); rows = []; accepted = []; bundle_ref = binding(bundle)
    contract = generation_contract(bundle, scenes=scenes, seed=seed, minimum=minimum, maximum=maximum,
        proposals=proposals, container_catalog=container_catalog, count_schedule=count_schedule, hooks=hooks)
    write(output / 'generation_contract.json', contract)
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'retained_sampler_and_slots': binding(previous.__file__), 'factory': binding(factory.__file__),
        'bundle': bundle_ref, 'container_catalog': binding(container_catalog) if container_catalog else None,
        'seed': seed, 'requested_scenes': scenes, 'object_count_range': [minimum, maximum],
        'count_schedule': count_schedule, 'maximum_geometry_proposals_per_scene': proposals,
        'hooks': hooks, 'physical_attempts': 0, 'API_calls': 0, 'runtime_rescaling': False,
        'source_selection_uses_physical_success': False,
        'scene_identity': 'sha256(full_generation_contract_sha256,scene_index,requested_N)[:32]',
        'generation_contract': binding(output / 'generation_contract.json'),
        'generation_contract_sha256': digest(contract),
        'capacity_selection': 'containers first; distinct-source capacity completion then source-uniform hash order'})
    partitions = sorted({d['partition'] for xs in source_pool.values() for _, d in xs})
    for i in range(scenes):
        directory = output / 'sampling' / f'{i:04d}'
        n = count_schedule[i] if count_schedule is not None else int(np.random.default_rng(seed + i).integers(minimum, maximum + 1))
        partition = partitions[i % len(partitions)] if partitions else 'unavailable'
        pool = {s: xs for s, xs in source_pool.items() if xs[0][1]['partition'] == partition}
        identity = scene_identity(contract, i, n)
        attempts = []; chosen = None
        for trial in range(proposals):
            trial_seed = seed + i * 1009 + trial * 31
            trial_dir = directory / f'proposal_{trial:02d}'
            began = time.monotonic()
            row = {'proposal': trial, 'seed': trial_seed, 'requested_object_count': n, 'passed': False}
            try:
                group, selected_bins, selection = capacity_select(pool, bins, n, partition, trial_seed)
                row.update(source_ids=[d['source_id'] for _, d in group], variant_ids=[e['variant_id'] for e, _ in group])
                write(trial_dir / 'selection.json', selection)
                spec, allocation = propose(identity, group, selected_bins, trial_seed, static,
                    scene_index=i, object_pose_proposal=object_pose_proposal)
                write(trial_dir / 'request.json', spec); write(trial_dir / 'capacity.json', allocation)
                # Match final factory clutter selection and exclusions exactly.
                spec['distractors']['exclude_source_ids'] = sorted({d['source_id'] for _, d in group})
                scene, tasks, checks = factory.compile_exact_scene(spec, [e for e, _ in group],
                    [ref for ref, _ in bins], seed + len(accepted))
                write(trial_dir / 'geometry_checks.json', checks)
                if layout_qualifier is not None:
                    write(trial_dir / 'preflight_scene.json', scene)
                    write(trial_dir / 'preflight_tasks.json', {'tasks': tasks})
                    report = layout_qualifier(deepcopy(scene), deepcopy(tasks))
                    if not isinstance(report, dict) or type(report.get('passed')) is not bool:
                        raise ValueError('layout_qualifier_must_return_explicit_passed_boolean')
                    write(trial_dir / 'layout_qualification.json', report)
                    row['layout_qualification'] = binding(trial_dir / 'layout_qualification.json')
                    if report['passed'] is not True:
                        raise Rejected('layout_qualifier_rejected', report=row['layout_qualification'])
                chosen = spec; row['passed'] = True
                row['request'] = binding(trial_dir / 'request.json')
            except (Rejected, ValueError, RuntimeError, KeyError) as error:
                row['reason'] = getattr(error, 'code', type(error).__name__)
                row['details'] = getattr(error, 'details', {'message': str(error)})
            row['seconds'] = time.monotonic() - began
            write(trial_dir / 'result.json', row); attempts.append(row)
            if chosen is not None:
                accepted.append(chosen); break
            if row.get('reason') == 'not_enough_distinct_admitted_sources_in_partition':
                break
        write(directory / 'attempts.json', {'attempts': attempts})
        rows.append({'scene_index': i, 'scene_id': identity, 'object_count': n, 'partition': partition,
                     'geometry_accepted': chosen is not None, 'attempts': binding(directory / 'attempts.json')})
        print({'scene': i, 'objects': n, 'geometry_accepted': chosen is not None}, flush=True)
    write(output / 'sampling.json', {'scenes': rows, 'all_requested_scenes_retained': True})
    write(output / 'requests.json', {'schema': factory.VERSION, 'scenes': accepted})
    result = factory.compile_requests(bundle, output / 'requests.json', output / 'factory', seed) if accepted else None
    summary = {'requested_scenes': scenes, 'accepted_geometry_scenes': len(accepted),
        'compiled_factory': result, 'scene_fingerprints': actual_fingerprints(output, contract),
        'seconds': time.monotonic() - start,
        'physical_attempts': 0, 'API_calls': 0}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--container-catalog', type=Path)
    parser.add_argument('--scenes', type=int, required=True); parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--min-objects', type=int, default=3); parser.add_argument('--max-objects', type=int, default=8)
    parser.add_argument('--proposals', type=int, default=16)
    parser.add_argument('--count-schedule', help='comma-separated fixed N per scene')
    parser.add_argument('--object-pose-proposal', help='module:function; same table/separation checks still apply')
    parser.add_argument('--layout-qualifier', help='module:function; receives scene,tasks, returns report with passed')
    args = parser.parse_args()
    print(build(args.bundle, args.output, scenes=args.scenes, seed=args.seed,
        minimum=args.min_objects, maximum=args.max_objects, proposals=args.proposals,
        container_catalog=args.container_catalog,
        count_schedule=[int(x) for x in args.count_schedule.split(',')] if args.count_schedule else None,
        object_pose_proposal=_callback(args.object_pose_proposal), layout_qualifier=_callback(args.layout_qualifier)))
