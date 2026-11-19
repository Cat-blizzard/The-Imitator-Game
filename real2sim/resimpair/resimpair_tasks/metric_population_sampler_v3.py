"""Container-conditioned, finite multi-object sampling from a metric snapshot.

No rescaling, asset-success whitelist, new grasp generation or physical execution.
The original v2 packing, table and separation rules remain the geometry gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import inspect
from itertools import combinations
import math
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected, check_separation, table_fit
from . import metric_population_sampler_v2 as previous
from . import metric_task_factory_v3 as factory

VERSION = 'metric_population_multitask_sampler_v3'
slots = previous.slots
admitted_pool = previous.admitted_pool


def scene_identity(bundle_hash, seed, scene_index):
    return 'metric_group_' + digest([VERSION, bundle_hash, int(seed), int(scene_index)])[:32]


def footprint(item):
    radius = float(np.linalg.norm(np.asarray(item[1]['half_size'], float)[:2]))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('finite_positive_native_footprint_required')
    return radius


def completion(selected, pool, sizes, selected_bins):
    """Find a geometry-only completion using the smallest remaining sources.

    There are only two groups and at most eight roles. For circular footprint
    packing, replacing a remaining radius by a smaller radius cannot invalidate
    the retained slots construction. Enumerating allocations of the smallest
    distinct-source footprints therefore proves a completion without planning
    or testing those assets in physics. At most C(8,4)=70 allocations are tried.
    The selected random variants themselves are never replaced by smaller ones.
    """
    used = {d['source_id'] for group in selected for _, d in group}
    excluded = used | {d['source_id'] for _, d in selected_bins}
    need = [sizes[i] - len(selected[i]) for i in range(2)]
    if min(need) < 0:
        return None
    smallest = [min(items, key=lambda x: (footprint(x), x[0]['variant_id']))
                for source, items in pool.items() if source not in excluded]
    smallest.sort(key=lambda x: (footprint(x), x[1]['source_id'], x[0]['variant_id']))
    if len(smallest) < sum(need):
        return None
    smallest = smallest[:sum(need)]
    for first in combinations(range(len(smallest)), need[0]):
        first = set(first)
        groups = [list(selected[0]) + [x for i, x in enumerate(smallest) if i in first],
                  list(selected[1]) + [x for i, x in enumerate(smallest) if i not in first]]
        if all(slots([footprint(x) for x in groups[c]],
                     selected_bins[c][1]['cavity']['flat_floor_radius_m'])['passed'] for c in range(2)):
            return groups
    return None


def capacity_select(pool, bins, n, partition, seed):
    """Select two container sources first, then condition N object choices."""
    if not 3 <= n <= 8:
        raise ValueError('three_to_eight_objects_required')
    if any(d['partition'] != partition or d['source_id'] != source
           for source, items in pool.items() for _, d in items):
        raise Rejected('source_partition_or_identity_conflict_in_pool')
    sizes = [(n + 1) // 2, n // 2]
    if len(pool) < n:
        raise Rejected('not_enough_distinct_admitted_sources_in_partition', requested=n, available=len(pool))
    candidates = []
    for c, size in enumerate(sizes):
        choices = []
        for ref, descriptor in bins:
            if descriptor['partition'] != partition:
                continue
            low = sorted((min(footprint(x) for x in xs) for source, xs in pool.items()
                          if source != descriptor['source_id']))[:size]
            if len(low) == size and slots(low, descriptor['cavity']['flat_floor_radius_m'])['passed']:
                choices.append((ref, descriptor))
        choices.sort(key=lambda x: digest([seed, c, x[1]['source_id'], x[0]['sha256']]))
        if not choices:
            raise Rejected('no_same_partition_container_for_requested_subgroup_size', group=c, size=size)
        candidates.append(choices)
    first = candidates[0][0]
    # A single original source cannot bind two different variants in one scene.
    second = next((x for x in candidates[1] if x[1]['source_id'] != first[1]['source_id']
                   or x[0]['sha256'] == first[0]['sha256']), None)
    if second is None:
        raise Rejected('no_consistent_second_container_variant')
    selected_bins = [first, second]
    chosen = [[], []]
    if completion(chosen, pool, sizes, selected_bins) is None:
        raise Rejected('container_pair_cannot_complete_distinct_sources', group_sizes=sizes,
                       container_variants=[ref for ref, _ in selected_bins])
    candidate_checks = []
    for step in range(n):
        c = step % 2
        used = {d['source_id'] for group in chosen for _, d in group}
        excluded = used | {d['source_id'] for _, d in selected_bins}
        selected = None; checked = 0
        for source in sorted(pool, key=lambda s: digest([seed, step, s])):
            if source in excluded:
                continue
            for item in sorted(pool[source], key=lambda x: digest([seed, step, source, x[0]['variant_id']])):
                checked += 1
                proposed = [list(x) for x in chosen]
                proposed[c].append(item)
                if completion(proposed, pool, sizes, selected_bins) is not None:
                    selected = item; break
            if selected is not None:
                break
        if selected is None:
            raise Rejected('capacity_conditioned_source_selection_exhausted', step=step,
                           candidate_checks=candidate_checks)
        chosen[c].append(selected)
        candidate_checks.append({'step': step, 'group': c, 'variants_examined': checked,
                                 'source_id': selected[1]['source_id'],
                                 'variant_id': selected[0]['variant_id'],
                                 'footprint_radius_m': footprint(selected)})
    interleaved = [chosen[i % 2][i // 2] for i in range(n)]
    return interleaved, selected_bins, {'group_sizes': sizes, 'candidate_checks': candidate_checks,
        'container_candidates_per_group': [len(xs) for xs in candidates],
        'selection_uses_physics_success': False,
        'policy': 'container hash order first; source hash order with exact distinct-source capacity completion'}


def default_object_pose_proposal(*, rng, **context):
    return {'xy': [float(rng.uniform(-.36, -.14)), float(rng.uniform(-.61, .61))],
            'yaw': float(rng.uniform(-math.pi, math.pi))}


def callback_binding(callback):
    if callback is None:
        return None
    source = inspect.getsourcefile(callback)
    if not source or not Path(source).is_file() or '<' in callback.__qualname__:
        raise ValueError('callback_must_be_a_named_file_backed_function')
    return {'implementation': binding(source), 'module': callback.__module__,
            'qualified_name': callback.__qualname__}


def propose(identity, group, selected, seed, static, *, scene_index=0,
            object_pose_proposal=None):
    rng = np.random.default_rng(seed)
    radii = [footprint(x) for x in group]
    packing = [slots(radii[c::2], d['cavity']['flat_floor_radius_m']) for c, (_, d) in enumerate(selected)]
    if not all(x['passed'] for x in packing):
        raise Rejected('subgroup_capacity_inconsistent')
    containers, occupied = [], []
    for i, (_, d) in enumerate(selected):
        radius = float(np.linalg.norm(np.asarray(d['dimensions_m'])[:2] / 2))
        xy = np.array([.10 + rng.uniform(-.015, .015), (-1 if i == 0 else 1) * (.40 + rng.uniform(-.02, .02))])
        table_fit(xy, radius); check_separation(xy, radius, occupied)
        occupied.append(('container_' + str(i), xy, radius))
        containers.append({'id': 'container_' + str(i), 'source_id': d['source_id'],
                           'xy': np.round(xy, 7).tolist(), 'yaw': float(rng.uniform(-.4, .4))})
    objects = []; pose_receipts = []
    for i, ((entry, descriptor), radius) in enumerate(zip(group, radii)):
        accepted = None; pose_attempts = []
        for proposal in range(128):
            context = {'rng': rng, 'scene_index': scene_index, 'object_index': i, 'proposal': proposal,
                       'descriptor': deepcopy(descriptor), 'entry': deepcopy(entry),
                       'container': deepcopy(containers[i % 2]),
                       'container_descriptor': deepcopy(selected[i % 2][1]),
                       'target_offset_xy_m': packing[i % 2]['offsets'][i // 2],
                       'occupied': [(key, np.asarray(xy).tolist(), r) for key, xy, r in occupied]}
            # Preserve the v2 default RNG order: yaw is drawn only after an XY
            # passes geometry. Custom hooks supply their own explicit yaw.
            value = object_pose_proposal(**context) if object_pose_proposal else {
                'xy': [rng.uniform(-.36, -.14), rng.uniform(-.61, .61)], 'yaw': None}
            if value is None:
                pose_attempts.append({'proposal': proposal, 'reason': 'pose_hook_no_candidate'}); continue
            xy = np.asarray(value['xy'], float)
            yaw = float(value['yaw']) if object_pose_proposal else 0.
            if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(yaw):
                raise ValueError('pose_hook_requires_finite_xy_and_yaw')
            try:
                table_fit(xy, radius); check_separation(xy, radius, occupied)
                if object_pose_proposal is None:
                    yaw = float(rng.uniform(-math.pi, math.pi))
                accepted = (xy, yaw); break
            except Rejected as error:
                pose_attempts.append({'proposal': proposal, 'reason': error.code, 'details': error.details})
        if accepted is None:
            raise Rejected('bounded_initial_object_layout_exhausted', object_index=i, attempts=128,
                           pose_attempts=pose_attempts)
        oid = 'object_' + str(i); xy, yaw = accepted
        occupied.append((oid, xy, radius))
        objects.append({'id': oid, 'source_id': descriptor['source_id'],
                        'xy': np.round(xy, 7).tolist(), 'yaw': yaw})
        pose_receipts.append({'object': oid, 'rejected_candidates': pose_attempts,
                              'accepted_proposal': proposal, 'pose': objects[-1]})
    def placements(indices, c, offsets):
        return [{'object': objects[i]['id'], 'target': {'kind': 'container',
                 'container': containers[c]['id'], 'offset_xy_m': offset}} for i, offset in zip(indices, offsets)]
    ids = list(range(len(objects))); subset = ids[0::2][:2]
    allpacking = slots(radii, selected[0][1]['cavity']['flat_floor_radius_m'])
    subpacking = slots([radii[i] for i in subset], selected[0][1]['cavity']['flat_floor_radius_m'])
    if not subpacking['passed']:
        raise Rejected('subset_capacity_inconsistent')
    tasks = [{'id': 'select_one', 'placements': placements([0], 0, [[0., 0.]])},
             {'id': 'select_other_bin', 'placements': placements([1], 1, [[0., 0.]])},
             {'id': 'collect_subset', 'placements': placements(subset, 0, subpacking['offsets'])},
             {'id': 'distribute_groups', 'placements': placements(ids[0::2], 0, packing[0]['offsets'])
                                                    + placements(ids[1::2], 1, packing[1]['offsets'])}]
    if allpacking['passed']:
        tasks.append({'id': 'collect_all', 'placements': placements(ids, 0, allpacking['offsets'])})
    spec = {'id': identity, 'objects': objects, 'containers': containers,
        'variant_choices': {d['source_id']: e['variant_id'] for e, d in group},
        'container_variant_choices': {d['source_id']: ref for ref, d in selected},
        'distractors': {'count': int(rng.integers(3, 6)), 'max_dimension_m': .18,
                       'catalog': static[0], 'source_role_index': static[1]}, 'tasks': tasks}
    return spec, {'collection': allpacking, 'subset': subpacking, 'distribution': packing,
        'compiled_goal_types': [t['id'] for t in tasks],
        'geometry_rejected_goal_types': [] if allpacking['passed'] else ['collect_all'],
        'all_in_one_container_is_not_a_scene_admission_requirement': True,
        'selected_container_variants': [ref for ref, _ in selected], 'pose_proposals': pose_receipts}


def _consumer_bins(bins, container_catalog):
    if container_catalog is None:
        return bins
    members = {(str(resolve(ref)), ref['sha256']) for ref, _ in bins}
    requested = read(container_catalog)['assets']
    keys = {(str(resolve(ref)), ref['sha256']) for ref in requested}
    if not keys <= members:
        raise ValueError('container_catalog_must_be_subset_of_immutable_bundle')
    return [(ref, d) for ref, d in bins if (str(resolve(ref)), ref['sha256']) in keys]


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
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'retained_sampler_and_slots': binding(previous.__file__), 'factory': binding(factory.__file__),
        'bundle': bundle_ref, 'container_catalog': binding(container_catalog) if container_catalog else None,
        'seed': seed, 'requested_scenes': scenes, 'object_count_range': [minimum, maximum],
        'count_schedule': count_schedule, 'maximum_geometry_proposals_per_scene': proposals,
        'hooks': hooks, 'physical_attempts': 0, 'API_calls': 0, 'runtime_rescaling': False,
        'source_selection_uses_physical_success': False,
        'scene_identity': 'sha256(version,bundle_sha256,seed,scene_index)[:32]',
        'capacity_selection': 'containers first; distinct-source capacity completion then source-uniform hash order'})
    partitions = sorted({d['partition'] for xs in source_pool.values() for _, d in xs})
    for i in range(scenes):
        directory = output / 'sampling' / f'{i:04d}'
        n = count_schedule[i] if count_schedule is not None else int(np.random.default_rng(seed + i).integers(minimum, maximum + 1))
        partition = partitions[i % len(partitions)] if partitions else 'unavailable'
        pool = {s: xs for s, xs in source_pool.items() if xs[0][1]['partition'] == partition}
        identity = scene_identity(bundle_ref['sha256'], seed, i)
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
        'compiled_factory': result, 'seconds': time.monotonic() - start,
        'physical_attempts': 0, 'API_calls': 0}
    write(output / 'summary.json', summary)
    return summary


def _callback(value):
    if value is None:
        return None
    module, name = value.split(':', 1)
    callback = getattr(importlib.import_module(module), name)
    callback_binding(callback)
    return callback


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
