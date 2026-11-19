"""Finite scene-conditioned goals for cached rigid pick/place.

Objects, cache scale, source partitions and the original geometric rules remain
fixed. A scene has 3..8 distinct goals rather than five hard-coded outputs.
Container placement uses one size-only rule for every measured circular cavity.
The one sequential extension is explicit: move B away, then put A at B's old
table position. No repeated object, handover, teleports or physical retry exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import math
from pathlib import Path
import re
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import (
    SCHEMA, Rejected, check_separation, compile_scene as legacy_compile_scene,
    request_index, table_fit,
)
from .asset_sampling_v3 import snapshot, make_role_catalog, partition_schedule
from .spatial_factory_v2 import sample_request
from .spatial_reachability_v4 import workspace_checks

VERSION = 'size_conditioned_multitask_scene_factory_v1'
PACK_GAP = .025
PACK_MARGIN = .012


def task_count(value, seed):
    if type(value) is int:
        limits = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2 and all(type(v) is int for v in value):
        limits = tuple(value)
    else:
        raise ValueError('tasks_per_scene_must_be_integer_or_two_integer_limits')
    if not 3 <= limits[0] <= limits[1] <= 8:
        raise ValueError('tasks_per_scene_must_be_between3and8')
    return int(np.random.default_rng(seed).integers(limits[0], limits[1] + 1))


def simple_slots(radii, cavity_radius, *, angle=0., gap=PACK_GAP, margin=PACK_MARGIN):
    """One conservative line/triangle/square policy, independent of asset ID."""
    r = np.asarray(radii, float)
    if r.ndim != 1 or not 1 <= len(r) <= 4 or not np.isfinite(r).all() or np.any(r <= 0):
        raise ValueError('one_to_four_positive_payload_radii_required')
    if not np.isfinite([cavity_radius, angle, gap, margin]).all() or cavity_radius <= 0 or gap < .015 or margin < .008:
        raise ValueError('invalid_cavity_or_clearance')
    if len(r) == 1:
        points = np.zeros((1, 2))
    elif len(r) == 2:
        points = np.array([[-(r[1] + gap / 2), 0.], [r[0] + gap / 2, 0.]])
    elif len(r) == 3:
        radius = (2 * r.max() + gap) / math.sqrt(3)
        points = np.array([[math.cos(a), math.sin(a)] for a in np.arange(3) * 2 * math.pi / 3]) * radius
    else:
        half = r.max() + gap / 2
        points = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * half
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = np.round(points @ rotation.T, 6)
    floor_remaining = cavity_radius - np.linalg.norm(points, axis=1) - r
    gaps = [float(np.linalg.norm(points[i] - points[j]) - r[i] - r[j])
            for i in range(len(r)) for j in range(i)]
    passed = bool(np.all(floor_remaining >= margin - 1e-6) and all(g >= gap - 1e-6 for g in gaps))
    return {'passed': passed, 'offsets_local_xy_m': points.tolist(), 'payload_radii_m': r.tolist(),
            'cavity_radius_m': float(cavity_radius), 'minimum_boundary_margin_m': float(floor_remaining.min()),
            'minimum_inter_payload_gap_m': min(gaps) if gaps else None,
            'required_boundary_margin_m': margin, 'required_inter_payload_gap_m': gap,
            'rule': {1: 'center', 2: 'balanced_line', 3: 'equal_triangle', 4: 'equal_square'}[len(r)],
            'gripper_entry_and_exit': 'must_be_checked_by_runtime_planner',
            'container_specific_settings': False, 'physical_success': 'unmeasured'}


def goal_signature(task):
    """Order alone does not define a distinct unordered destination goal."""
    placements = sorted(task['placements'], key=lambda p: p['object'])
    return digest(placements)


def select_tasks(tasks, count, selection_seed):
    if count == 8: return tasks
    fixed = ['relocate_selected', 'collect_group', 'clear_then_place']
    rest = [t for t in tasks if t['id'] not in fixed]
    order = np.random.default_rng(selection_seed).permutation(len(rest))
    allowed = set(fixed + [rest[i]['id'] for i in order[:count - 3]])
    return [t for t in tasks if t['id'] in allowed]


def sequential_vacancy(spec, task, radii, container_radii):
    """Validate the explicit B->safe, A->B.initial relation, with other obstacles."""
    placements = task['placements']
    if len(placements) != 2 or len({p['object'] for p in placements}) != 2:
        raise Rejected('vacancy_requires_two_distinct_objects_once')
    objects = {o['id']: o for o in spec['objects']}
    if any(p['object'] not in objects or p['target'].get('kind') != 'table' for p in placements):
        raise Rejected('vacancy_only_supports_existing_tabletop_objects')
    blocker, dependent = [p['object'] for p in placements]
    if not np.allclose(placements[1]['target']['xy'], objects[blocker]['xy'], atol=1e-9, rtol=0):
        raise Rejected('vacancy_destination_must_equal_predecessor_initial_position')
    occupied = [(o['id'], np.asarray(o['xy']), radii[o['id']]) for o in spec['objects']]
    occupied.extend((c['id'], np.asarray(c['xy']), container_radii[c['id']]) for c in spec.get('containers', []))
    checks = []
    for step, placement in enumerate(placements):
        oid = placement['object']; xy = np.asarray(placement['target']['xy'], float)
        table_fit(xy, radii[oid]); check_separation(xy, radii[oid], occupied, exclude=[oid])
        before = {name: point.tolist() for name, point, _ in occupied}
        occupied = [(name, xy.copy() if name == oid else point, radius) for name, point, radius in occupied]
        checks.append({'step': step, 'object': oid, 'destination_xy_m': xy.tolist(),
                       'occupancy_before': before, 'passed': True})
    return {'passed': True, 'required_object_precedence': [[blocker, dependent]],
            'initial_target_occupied_by': blocker, 'dependent_object': dependent,
            'initial_target_is_not_feasible': True, 'checks': checks,
            'scope': 'sequential footprint geometry; no assertion of grasp occlusion or successful motion'}


def compile_scene_sequential(spec, entries, container_entries, seed):
    """Keep the original compiler; extend only an explicit vacated table goal."""
    request_index({'schema': SCHEMA, 'scenes': [spec]})
    clear = [t for t in spec['tasks'] if t['id'] == 'clear_then_place']
    if len(clear) > 1:
        raise Rejected('duplicate_vacancy_goal')
    surrogate = copy.deepcopy(spec); vacancy = None
    if clear:
        asset_map = {read(resolve(e['asset']))['source_id']: read(resolve(e['asset'])) for e in entries}
        bin_map = {read(resolve(b))['source_id']: read(resolve(b)) for b in container_entries}
        radii = {o['id']: float(np.linalg.norm(asset_map[o['source_id']]['half_size'][:2])) for o in spec['objects']}
        bin_radii = {c['id']: float(np.linalg.norm(np.asarray(bin_map[c['source_id']]['dimensions_m'])[:2] / 2))
                     for c in spec.get('containers', [])}
        vacancy = sequential_vacancy(spec, clear[0], radii, bin_radii)
        next(t for t in surrogate['tasks'] if t['id'] == 'clear_then_place')['placements'] = clear[0]['placements'][:1]
    scene, tasks, evidence = legacy_compile_scene(surrogate, entries, container_entries, seed)
    objects = {o['id']: o for o in scene['objects']}
    for task in tasks:
        # A fixed controller loop order must not silently become a task demand.
        task['required_operation_precedence'] = []
        if task['requested_task_id'] != 'clear_then_place':
            continue
        ops = []
        for i, placement in enumerate(clear[0]['placements']):
            oid = placement['object']; goal = placement['target']
            yaw = float(goal.get('yaw', objects[oid]['yaw']))
            ops.append({'id': f'transfer_{i}', 'primitive': 'pick_place', 'object': oid, 'arm': None,
                        'target': {'support': 'table', 'offset_xy': list(goal['xy']), 'yaw': yaw}})
            if i:
                evidence['goals'].append({'task': task['requested_task_id'], 'object': oid,
                    'world_position_m': [*goal['xy'], scene['table_top_z'] + objects[oid]['half_size'][2]],
                    'yaw_radians': yaw, 'explicit_orientation_required': 'yaw' in goal,
                    'sequential_vacancy_required': True})
        task.update(operations=ops, object_ids=[o['object'] for o in ops],
                    required_operation_precedence=[['transfer_0', 'transfer_1']],
                    sequential_preconditions={'transfer_1': {'initial_occupant': vacancy['initial_target_occupied_by'],
                                                              'required_completed_operation': 'transfer_0'}},
                    scope='cached rigid pick/place with explicit vacated table destination; no repeated object')
    evidence['sequential_vacancy'] = vacancy
    evidence['compiler_extension'] = binding(__file__)
    evidence['old_compiler_request_difference'] = 'only vacancy second placement deferred to sequential occupancy check' if clear else None
    return scene, tasks, evidence


def load_inventory(index, containers):
    """Use descriptor extents for selection; exact meshes are checked when chosen."""
    from resimpair_grasp.grasps import load
    pool = []; rejected = []; seen = set()
    for entry in read(index)['entries']:
        source = entry.get('source_id')
        try:
            d = read(resolve(entry['asset'])); source = d['source_id']; half = np.asarray(d['half_size'], float)
            if source in seen: raise ValueError('duplicate_source')
            if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0): raise ValueError('invalid_metric_extent')
            if d['partition'] not in {'development', 'heldout_asset'}: raise ValueError('unknown_partition')
            if not d['roles'].get('rigid_manipulation_candidate'): raise ValueError('not_rigid_manipulation_candidate')
            cache = load(entry['cache'], entry['asset'])
            if cache.get('signature', {}).get('geometry_scope') != 'native' or not cache.get('candidates'):
                raise ValueError('nonempty_native_cache_required')
            seen.add(source); pool.append({'entry': entry, 'descriptor': d, 'source_id': source,
                'partition': d['partition'], 'radius': float(np.linalg.norm(half[:2]))})
        except (ValueError, KeyError, OSError, TypeError) as error:
            rejected.append({'source_id': source, 'reason': str(error)})
    bins = []; bin_rejected = []
    for b in read(containers)['assets']:
        d = read(resolve(b))
        if (not d.get('roles', {}).get('stationary_container') or re.search(r'\b(soup|filled|water|cereal|fruit|food)\b', d['name'], re.I)):
            bin_rejected.append({'source_id': d['source_id'], 'reason': 'unqualified_role_or_existing_contents_name'}); continue
        cavity = min(float(d['cavity']['flat_floor_radius_m']), float(d['cavity']['minimum_wall_radius_m']))
        if cavity <= PACK_MARGIN or not np.isfinite(cavity):
            bin_rejected.append({'source_id': d['source_id'], 'reason': 'invalid_cavity'}); continue
        bins.append({'binding': b, 'descriptor': d, 'source_id': d['source_id'], 'partition': d['partition'], 'capacity': cavity})
    return pool, bins, rejected, bin_rejected


def choose_group(pool, bins, partition, count, source_use, container_use, seed, number):
    choices = [c for c in bins if c['partition'] == partition]
    choices.sort(key=lambda c: (container_use[c['source_id']], -c['capacity'], digest([seed, number, c['source_id']])))
    for container in choices:
        assets = [a for a in pool if a['partition'] == partition
                  and simple_slots([a['radius']] * count, container['capacity'])['passed']]
        assets.sort(key=lambda a: (source_use[a['source_id']], digest([seed, number, a['source_id']])))
        if len(assets) < count:
            continue
        group = assets[:count]
        slots = simple_slots([a['radius'] for a in group], container['capacity'])
        if slots['passed']:
            second = [c for c in choices if c['source_id'] != container['source_id']
                      and all(simple_slots([a['radius']], c['capacity'])['passed'] for a in group)]
            use_two = count == 2 and number % 2 == 0 and bool(second)
            return group, [container, second[0]] if use_two else [container]
    raise Rejected('no_same_partition_group_with_simple_shared_container_capacity', partition=partition, objects=count)


def base_request(group, bins, number, seed, catalog, role_index):
    n = len(group); table = lambda i: {'object': f'object_{i}', 'target': {'kind': 'table', 'xy': [.0, -.6 + i * .4]}}
    inside = lambda i, c=0: {'object': f'object_{i}', 'target': {'kind': 'container', 'container': f'container_{c}', 'offset_xy_m': [0., 0.]}}
    subset = [1] if n == 2 else list(range(0, n, 2))
    if n == 3: subset = [0, 1]
    tasks = [
        {'id': 'relocate_selected', 'placements': [table(0)]},
        {'id': 'contain_selected', 'placements': [inside(0)]},
        {'id': 'orient_selected', 'placements': [{'object': 'object_0', 'target': {'kind': 'table', 'xy': [.0, -.6], 'yaw': math.pi / 2}}]},
        {'id': 'arrange_group', 'placements': [table(i) for i in range(n)]},
        {'id': 'collect_group', 'placements': [inside(i) for i in range(n)]},
        {'id': 'selected_subset', 'placements': [inside(i) for i in subset]},
        {'id': 'distribute_group' if len(bins) == 2 else 'split_group',
         'placements': [inside(i, i % 2) for i in range(n)] if len(bins) == 2 else
                       [table(i) if i % 2 == 0 else inside(i) for i in range(n)]},
    ]
    tasks.append({'id': 'clear_then_place', 'placements': [table(1),
        {'object': 'object_0', 'target': {'kind': 'table', 'xy': [-.3, -.2]}}]})
    return {'id': f'multitask_scene_{number:03d}_' + digest([seed, [a['source_id'] for a in group]])[:8],
        'objects': [{'id': f'object_{i}', 'source_id': a['source_id'], 'xy': [-.3, -.6 + i * .4], 'yaw': 0.} for i, a in enumerate(group)],
        'containers': [{'id': f'container_{i}', 'source_id': c['source_id'], 'xy': [.1, -.4 + i * .8], 'yaw': 0.} for i, c in enumerate(bins)],
        'distractors': {'count': 3 + number % 4, 'max_dimension_m': .10,
                       'catalog': catalog, 'source_role_index': role_index}, 'tasks': tasks}


def layout_request(original, group, bins, seed, count):
    assets = {a['source_id']: a['descriptor'] for a in group}; containers = {c['source_id']: c['descriptor'] for c in bins}
    sampling = copy.deepcopy(original)
    sampling['tasks'] = [t for t in sampling['tasks'] if t['id'] != 'clear_then_place']
    result, trace = sample_request(sampling, assets, containers, seed)
    result['id'] = original['id']; by_id = {o['id']: o for o in result['objects']}
    radii = {f'object_{i}': a['radius'] for i, a in enumerate(group)}
    capacities = {f'container_{i}': c['capacity'] for i, c in enumerate(bins)}
    packing = []
    for task in result['tasks']:
        for cid, capacity in capacities.items():
            placements = [p for p in task['placements'] if p['target'].get('container') == cid]
            if not placements: continue
            angle = float(np.random.default_rng(seed + 100 + len(packing)).uniform(-math.pi, math.pi))
            slots = simple_slots([radii[p['object']] for p in placements], capacity, angle=angle)
            if not slots['passed']:
                raise Rejected('simple_slots_failed_after_group_selection', task=task['id'], evidence=slots)
            for placement, xy in zip(placements, slots['offsets_local_xy_m']): placement['target']['offset_xy_m'] = xy
            packing.append({'task': task['id'], 'container': cid, **slots})
    group_goal = next(t for t in result['tasks'] if t['id'] == 'arrange_group')
    safe = copy.deepcopy(next(p for p in group_goal['placements'] if p['object'] == 'object_1'))
    result['tasks'].append({'id': 'clear_then_place', 'placements': [safe,
        {'object': 'object_0', 'target': {'kind': 'table', 'xy': list(by_id['object_1']['xy'])}}]})
    signatures = [goal_signature(t) for t in result['tasks']]
    if len(set(signatures)) != len(signatures): raise Rejected('duplicate_destination_goals')
    # Reserve a simple task, a collect task and the explicit dependency; sample
    # the rest by structure, not outcome. The eight-task qualification uses all.
    result['tasks'] = select_tasks(result['tasks'], count, int(digest(original['id'])[:8], 16))
    trace.update(simple_capacity_allocations=packing, tasks_requested=count,
                 repeated_object_allowed=False, unordered_sequence_permutations_count_as_new_tasks=False)
    return result, trace


def build(index, containers, static_index, output, *, scenes=10, tasks_per_scene=8,
          seed=202610300500, heldout_fraction=.2, maximum_layout_proposals=32, sampling_radius=.90):
    output = Path(output)
    if output.exists(): raise ValueError('preserve_existing_factory')
    if type(scenes) is not int or not 1 <= scenes <= 10000 or type(seed) is not int or seed < 0:
        raise ValueError('finite_scene_count_and_seed_required')
    if type(maximum_layout_proposals) is not int or not 1 <= maximum_layout_proposals <= 32:
        raise ValueError('at_most32_geometry_layout_proposals_required')
    task_count(tasks_per_scene, seed)
    if not 0 <= heldout_fraction <= 1 or not 0 < sampling_radius <= .90: raise ValueError('invalid_split_or_workspace')
    started = time.monotonic(); output.mkdir(parents=True)
    snaps = {name: snapshot(path, output / 'snapshots' / (name + '.json'))
             for name, path in [('index', index), ('containers', containers), ('static_index', static_index)]}
    index, containers, static_index = [resolve(snaps[k]['snapshot']) for k in ['index', 'containers', 'static_index']]
    catalog, role_index = make_role_catalog(static_index, index, containers, output / 'catalogs')
    pool, bins, rejects, bin_rejects = load_inventory(index, containers)
    conflicts = set(read(resolve(role_index))['partition_conflicts'])
    pool = [a for a in pool if a['source_id'] not in conflicts]; bins = [c for c in bins if c['source_id'] not in conflicts]
    schedule = partition_schedule(scenes, {a['partition'] for a in pool}, heldout_fraction, seed)
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'index': binding(index), 'containers': binding(containers), 'static_index': binding(static_index),
        'snapshots': snaps, 'scenes': scenes, 'tasks_per_scene': tasks_per_scene, 'seed': seed,
        'heldout_fraction': heldout_fraction, 'maximum_layout_proposals': maximum_layout_proposals,
        'sampling_radius_m': sampling_radius, 'source_counts': {'pose_assets': len(pool), 'containers': len(bins)},
        'compiler': binding(ROOT / 'resimpair_pickplace/requested_tasks.py'),
        'layout_sampler': binding(ROOT / 'resimpair_tasks/spatial_factory_v2.py'),
        'physical_executions': 0, 'API_calls': 0, 'optimizer_calls': 0, 'failed_sources_replaced': False})
    write(output / 'inventory.json', {'ready_unique_assets': len(pool), 'container_count': len(bins),
        'asset_rejections': rejects, 'container_rejections': bin_rejects, 'selection_uses_physical_results': False,
        'assets': [{'source_id': a['source_id'], 'radius_m': a['radius'], 'partition': a['partition'], **a['entry']} for a in pool]})
    rows = []; source_use = Counter(); container_use = Counter()
    for number, partition in enumerate(schedule):
        requested_count = task_count(tasks_per_scene, seed + number)
        directory = output / 'scenes' / f'unselected_{number:03d}'
        original = None; accepted = None; attempts = []; preflight = None
        try:
            group, selected_bins = choose_group(pool, bins, partition, 2 + number % 3, source_use, container_use, seed, number)
            for a in group: source_use[a['source_id']] += 1
            for c in selected_bins: container_use[c['source_id']] += 1
            original = base_request(group, selected_bins, number, seed, catalog, role_index)
            directory = output / 'scenes' / original['id']; write(directory / 'source_selection.json', original)
            raw_request = copy.deepcopy(original)
            raw_request['tasks'] = select_tasks(raw_request['tasks'], requested_count, int(digest(original['id'])[:8], 16))
            write(directory / 'request.json', raw_request)
            entries = [a['entry'] for a in group]; bin_entries = [c['binding'] for c in selected_bins]
            for trial in range(maximum_layout_proposals):
                trial_dir = directory / 'proposals' / f'{trial:02d}'; trial_seed = seed + number * 1009 + trial * 37
                try:
                    request, trace = layout_request(original, group, selected_bins, trial_seed, requested_count)
                    write(trial_dir / 'request.json', request); write(trial_dir / 'sampling.json', trace)
                    write(directory / 'request.json', request)
                    scene, tasks, checks = compile_scene_sequential(request, entries, bin_entries, trial_seed)
                    workspace = workspace_checks(scene, tasks, sampling_radius)
                    write(trial_dir / 'workspace.json', workspace)
                    if not workspace['passed']: raise Rejected('cached_TCP_workspace_failed', rejected_operations=workspace['rejected_operations'])
                    write(directory / 'request.json', request); write(directory / 'compiled_scene.json', scene)
                    write(directory / 'compiled_tasks.json', {'tasks': tasks}); write(directory / 'geometry_checks.json', checks)
                    metadata = {}
                    for task in tasks:
                        short = task['requested_task_id']
                        metadata[short] = {'structure': short, 'operation_count': len(task['operations']),
                            'required_object_precedence': checks['sequential_vacancy']['required_object_precedence'] if short == 'clear_then_place' else [],
                            'initial_target_occupied_by': checks['sequential_vacancy']['initial_target_occupied_by'] if short == 'clear_then_place' else None,
                            'goal_signature': goal_signature(next(t for t in request['tasks'] if t['id'] == short)),
                            'placement_policy': 'size_only_center_line_triangle_square', 'ordered_only_if_task_requires': True}
                    write(directory / 'task_metadata.json', {'tasks': metadata})
                    accepted = {'proposal': trial, 'seed': trial_seed, 'sampling': binding(trial_dir / 'sampling.json'),
                                'workspace': binding(trial_dir / 'workspace.json')}
                    attempts.append({'trial': trial, 'seed': trial_seed, 'accepted': True}); break
                except (Rejected, ValueError) as error:
                    failure = {'reason': getattr(error, 'code', str(error)), 'details': getattr(error, 'details', {})}
                    write(trial_dir / 'rejection.json', failure); attempts.append({'trial': trial, 'seed': trial_seed, 'accepted': False, **failure})
        except (Rejected, ValueError) as error:
            preflight = {'reason': getattr(error, 'code', str(error)), 'details': getattr(error, 'details', {})}
            write(directory / 'source_rejection.json', preflight)
        write(directory / 'attempts.json', {'attempts': attempts, 'accepted': accepted, 'preflight': preflight})
        row = {'scene_id': directory.name, 'partition': partition, 'source_ids': [a['source_id'] for a in group] if original else [],
            'requested_tasks': requested_count, 'task_ids': [t['id'] for t in read(directory / 'request.json')['tasks']] if original else [],
            'accepted_geometry': accepted is not None, 'request': binding(directory / 'request.json') if original else None,
            'directory': str(directory), 'failure': preflight or (attempts[-1] if not accepted and attempts else None),
            'compiled_scene': binding(directory / 'compiled_scene.json') if accepted else None,
            'compiled_tasks': binding(directory / 'compiled_tasks.json') if accepted else None,
            'task_metadata': binding(directory / 'task_metadata.json') if accepted else None,
            'accepted': accepted, 'attempts': binding(directory / 'attempts.json'), 'physical_status': 'not_run'}
        rows.append(row); print({'scene': number + 1, 'tasks': requested_count, 'accepted_geometry': bool(accepted), 'proposals': len(attempts)}, flush=True)
    write(output / 'scenes.json', {'scenes': rows})
    summary = {'schema': VERSION, 'requested_scenes': scenes, 'requested_tasks': sum(r['requested_tasks'] for r in rows),
        'geometry_passed_scenes': sum(r['accepted_geometry'] for r in rows),
        'geometry_passed_tasks': sum(r['requested_tasks'] for r in rows if r['accepted_geometry']),
        'selected_asset_sources': len(source_use), 'selected_container_sources': len(container_use),
        'seconds': time.monotonic() - started, 'physical_executions': 0, 'API_calls': 0, 'optimizer_calls': 0,
        'success_from_geometry_claimed': False}
    write(output / 'status.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['index', 'containers', 'static-index', 'output']: p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--scenes', type=int, default=10); p.add_argument('--tasks-per-scene', nargs='+', type=int, default=[8])
    p.add_argument('--seed', type=int, default=202610300500); p.add_argument('--maximum-layout-proposals', type=int, default=32)
    a = p.parse_args(); count = a.tasks_per_scene[0] if len(a.tasks_per_scene) == 1 else a.tasks_per_scene
    print(build(a.index, a.containers, a.static_index, a.output, scenes=a.scenes, tasks_per_scene=count,
                seed=a.seed, maximum_layout_proposals=a.maximum_layout_proposals))
