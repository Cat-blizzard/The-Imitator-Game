"""Finite source-aware task requests for the existing cached pick/place adapter.

This factory expands goal programs, assets and layouts, not contact mechanisms.
It never generates grasps, selects an arm, runs physics, or uses outcome-based
resampling. Every shared scene has five goals and a finite geometry-only layout
search. Rejected layout proposals and sources remain recorded. The independent
coverage auditor and physical acceptance rules are deliberately not imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import re
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import SCHEMA, Rejected, compile_scene, request_index

VERSION = 'cached_pickplace_task_factory_v1_r2'
TASKS_PER_SCENE = 5
CHUNK_SCENES = 8
# Enumerated before seeing either geometry checks or physical outcomes. Each
# proposal gets one fixed jitter draw; there is no recursive/retry-until-pass loop.
LAYOUTS = (
    ('forward', -.33, .12, 1.00, 1),
    ('mirror', -.33, .12, 1.00, -1),
    ('wide', -.34, .14, 1.08, 1),
    ('wide_mirror', -.34, .14, 1.08, -1),
    ('compact', -.30, .10, .94, 1),
    ('compact_mirror', -.30, .10, .94, -1),
)


def _failure(error):
    return {'reason': error.code, 'details': error.details} if isinstance(error, Rejected) else {
        'reason': type(error).__name__, 'details': {'message': str(error)}}


def load_pool(index, containers):
    """Hash-bound native descriptors/caches; no semantic or dynamics claim."""
    import trimesh
    from resimpair_grasp.grasps import load
    values = []; rejected = []; seen = {}; active_partitions = defaultdict(set)
    for entry in read(index)['entries']:
        source = None
        try:
            descriptor = read(resolve(entry['asset'])); source = descriptor['source_id']
            active_partitions[source].add(descriptor['partition'])
            cache = load(entry['cache'], entry['asset'])
            if cache.get('signature', {}).get('geometry_scope') != 'native':
                raise ValueError('native_collision_geometry_cache_required')
            if source in seen:
                if seen[source] != descriptor['partition']:
                    raise ValueError('source_assigned_to_multiple_partitions')
                rejected.append({'source_id': source, 'reason': 'duplicate_source_entry'}); continue
            seen[source] = descriptor['partition']
            half = np.asarray(descriptor['half_size'], dtype=float)
            if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
                raise ValueError('invalid_native_dimensions')
            if not descriptor.get('roles', {}).get('rigid_manipulation_candidate'):
                raise ValueError('not_rigid_manipulation_candidate')
            if not cache.get('candidates'):
                raise ValueError('empty_native_grasp_cache')
            mesh = trimesh.load(resolve(descriptor['visual']), force='scene', process=False).to_geometry()
            vertices = np.asarray(mesh.vertices)
            if not vertices.size or not np.isfinite(vertices).all():
                raise ValueError('invalid_native_visual_geometry')
            values.append({'entry': entry, 'descriptor': descriptor, 'vertices': vertices,
                           'source_id': source, 'partition': descriptor['partition'],
                           'radius': float(np.linalg.norm(half[:2])),
                           'floor_radial_extent_m': float(np.linalg.norm(vertices[:, :2], axis=1).max())})
        except (ValueError, KeyError, TypeError, OSError) as error:
            rejected.append({'source_id': source, **_failure(error)})
    container_values = []; container_rejected = []
    for bound in read(containers)['assets']:
        source = None
        try:
            descriptor = read(resolve(bound)); source = descriptor['source_id']
            if source in active_partitions and active_partitions[source] != {descriptor['partition']}:
                raise Rejected('container_active_source_partition_conflict',
                               active_partitions=sorted(active_partitions[source]),
                               container_partition=descriptor['partition'], descriptor=bound)
            if re.search(r'\b(soup|filled|water|cereal|fruit|food)\b', descriptor['name'], re.I):
                raise ValueError('container_name_implies_existing_contents')
            if not descriptor['roles']['stationary_container']:
                raise ValueError('stationary_container_role_missing')
            radius = float(descriptor['cavity']['flat_floor_radius_m'])
            if not np.isfinite(radius) or radius <= .008:
                raise ValueError('invalid_inner_floor_radius')
            container_values.append({'binding': bound, 'descriptor': descriptor, 'source_id': source,
                                     'partition': descriptor['partition']})
        except (ValueError, KeyError, TypeError, OSError) as error:
            container_rejected.append({'source_id': source, **_failure(error)})
    container_partitions = defaultdict(set)
    for item in container_values:
        container_partitions[item['source_id']].add(item['partition'])
    conflicts = {s for s, partitions in container_partitions.items() if len(partitions) != 1}
    for item in container_values:
        if item['source_id'] in conflicts:
            container_rejected.append({'source_id': item['source_id'], 'reason': 'container_source_partition_conflict',
                'details': {'partitions': sorted(container_partitions[item['source_id']]), 'descriptor': item['binding']}})
    container_values = [item for item in container_values if item['source_id'] not in conflicts]
    return values, container_values, rejected, container_rejected


def _fit(asset, container):
    # All factory container offsets are zero. The supported cavity is a disk,
    # so rotations leave this exact mesh-vertex radial extent unchanged. The
    # retained compiler subsequently repeats floor_fit at the requested yaw.
    return container['descriptor']['cavity']['flat_floor_radius_m'] - asset['floor_radial_extent_m'] >= .008


def choose_group(anchor, pool, containers, number, seed):
    """One unique anchor, same-split companions, explicit cavity-compatible role."""
    partition = anchor['partition']
    others = [a for a in pool if a['partition'] == partition and a['source_id'] != anchor['source_id']]
    others.sort(key=lambda a: digest([seed, number, a['source_id']]))
    choices = [c for c in containers if c['partition'] == partition]
    choices.sort(key=lambda c: digest([seed, number, c['source_id']]))
    fitting = [(a, c) for a in others for c in choices if _fit(a, c)]
    if not fitting:
        raise Rejected('no_same_partition_container_companion', anchor=anchor['source_id'], partition=partition)
    companion, first = fitting[0]
    desired = 2 + number % 3
    group = [anchor, companion]
    group.extend(a for a in others if a['source_id'] != companion['source_id'])
    group = group[:desired]
    selected_containers = [first]
    # Two explicit stations require the anchor to fit as well. The two stations
    # may share a source mesh: their instance identity and positions stay distinct.
    if len(group) == 2 and number % 2 == 0:
        second_choices = [c for c in choices if _fit(anchor, c)]
        if second_choices:
            selected_containers.append(second_choices[0])
    return group, selected_containers, {'requested_object_count': desired, 'actual_object_count': len(group),
        'selection': 'source/partition/cache/size/inner-floor geometry only',
        'anchor_source_id': anchor['source_id'], 'container_role_object': 'object_1'}


def _table(object_id, xy, yaw=None):
    target = {'kind': 'table', 'xy': [round(float(v), 6) for v in xy]}
    if yaw is not None: target['yaw'] = round(float(yaw), 6)
    return {'object': object_id, 'target': target}


def _container(object_id, name):
    return {'object': object_id, 'target': {'kind': 'container', 'container': name,
                                          'offset_xy_m': [0., 0.]}}


def make_request(group, containers, number, seed, layout_number, *, exclude_source_ids=()):
    """2–4 objects, 1–2 stationary containers, five actual destination programs."""
    name, source_x, container_x, spread, sign = LAYOUTS[layout_number]
    rng = np.random.default_rng(seed + number * 104729 + layout_number * 8191)
    n = len(group); two = len(containers) == 2
    # Target-source separation follows actual cached metric sizes; no scaling.
    gap = max(.225, 2 * max(a['radius'] for a in group) + .045)
    goal_x = source_x + gap
    if two:
        source_y = [-.35, .35]
        goals = [[source_x + .03, -.025], [goal_x + .05, .035]]
        station_xy = [[container_x, -.43], [container_x, .43]]
    elif n == 2:
        source_y = [-.42, .42]
        goals = [[goal_x, -.42], [goal_x, .42]]
        station_xy = [[container_x, 0.]]
    elif n == 3:
        source_y = [-.60, 0., .60]
        goals = [[goal_x, -.60], [goal_x - .015, .29], [goal_x, .60]]
        station_xy = [[container_x, 0.]]
    else:
        source_y = [-.62, -.22, .22, .62]
        goals = [[goal_x, -.62], [goal_x - .015, -.29], [goal_x - .015, .29], [goal_x, .62]]
        station_xy = [[container_x, 0.]]
    objects = []
    # Small metric randomization is bounded and recorded. Asset substitutions
    # and seed changes alone are not reported as different relation structures.
    for i, (asset, y) in enumerate(zip(group, source_y)):
        xy = [source_x + float(rng.uniform(-.008, .008)), sign * y * spread]
        objects.append({'id': f'object_{i}', 'source_id': asset['source_id'],
                        'xy': [round(v, 6) for v in xy], 'yaw': round(float(rng.uniform(-math.pi, math.pi)), 6)})
    goals = [[round(x, 6), round(sign * y * spread, 6)] for x, y in goals]
    station_requests = [{'id': f'container_{i}', 'source_id': c['source_id'],
        'xy': [round(x, 6), round(sign * y * spread, 6)],
        'yaw': round(float(rng.uniform(-math.pi, math.pi)), 6)}
        for i, (c, (x, y)) in enumerate(zip(containers, station_xy))]
    # First container accepts object_1; second, when present, accepts object_0.
    tasks = [
        {'id': 'relocate_selected', 'placements': [_table('object_0', goals[0])]},
        {'id': 'orient_selected', 'placements': [_table('object_0', goals[0], objects[0]['yaw'] + math.pi / 2)]},
        {'id': 'contain_selected', 'placements': [_container('object_1', 'container_0')]},
        {'id': 'arrange_group', 'placements': [_table(f'object_{i}', goals[i]) for i in range(n)]},
    ]
    if two:
        tasks.append({'id': 'distribute_containers', 'placements': [
            _container('object_0', 'container_1'), _container('object_1', 'container_0')]})
    else:
        tasks.append({'id': 'split_goals', 'placements': [
            _table('object_0', goals[0]), _container('object_1', 'container_0')]})
    sid = f'factory_scene_{number:03d}_' + digest([seed, [a['source_id'] for a in group]])[:8]
    request = {'id': sid, 'objects': objects, 'containers': station_requests,
               'distractors': {'count': 3 + number % 6, 'max_dimension_m': [.10, .13, .16][number % 3],
                               'exclude_source_ids': list(exclude_source_ids)},
               'tasks': tasks}
    request_index({'schema': SCHEMA, 'scenes': [request]})
    return request, {'layout': name, 'layout_index': layout_number, 'source_target_gap_m': gap,
                     'size_policy': 'unchanged native cached geometry', 'layout_seed': seed + number * 104729 + layout_number * 8191}


def build(index, containers, output, *, scenes=100, seed=20261027, min_unique_assets=None):
    """Write new, immutable requests; geometry rejection never means success.

    Returns a terminal summary. ``insufficient_assets`` writes inventory/status
    but fabricates no scenes. ``geometry_rejections`` retains all requested
    scenes, including failed preflights; no replacement source is sampled.
    Root subsequently registers each chunk with its recorded seed_start, which
    reproduces this compiler's distractor selection before separate arm IK.
    """
    started = time.monotonic(); index, containers, output = map(Path, (index, containers, output))
    if output.exists(): raise ValueError('Task factory output already exists; choose an independent version')
    if type(scenes) is not int or not 1 <= scenes <= 100:
        raise ValueError('scenes_must_be_between_1_and_100')
    minimum = scenes if min_unique_assets is None else min_unique_assets
    if type(minimum) is not int or not 1 <= minimum <= scenes * 4:
        raise ValueError('invalid_minimum_unique_asset_target')
    if type(seed) is not int or seed < 0: raise ValueError('seed_must_be_nonnegative_integer')
    output.mkdir(parents=True)
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'index': binding(index), 'containers': binding(containers), 'seed': seed,
        'scenes': scenes, 'tasks_per_scene': TASKS_PER_SCENE, 'minimum_unique_assets': minimum,
        'chunk_scenes': CHUNK_SCENES, 'layout_candidates': LAYOUTS,
        'compiler': binding(ROOT / 'resimpair_pickplace/requested_tasks.py'),
        'source_role_policy': 'exclude_complete_active_and_container_pools_from_clutter_v1',
        'selection_frozen_before_physics': True, 'mechanism': 'pick_place',
        'new_contact_mechanisms': 0, 'physical_executions': 0, 'api_calls': 0, 'optimizer_calls': 0,
        'acceptance_scope': 'request schema, geometry and source split only; IK and dynamics unmeasured'})
    pool, container_pool, rejected, container_rejected = load_pool(index, containers)
    # Reserve the entire available input pools, not just the sources selected
    # for this scene or output batch. This includes identifiable rejected pool
    # members, so a differently labelled static descriptor cannot reintroduce
    # their identity as clutter in another partition.
    exclusion_ids = sorted({row['source_id'] for rows in (pool, container_pool, rejected, container_rejected)
                            for row in rows if isinstance(row.get('source_id'), str) and row['source_id']})
    write(output / 'source_role_policy.json', {
        'policy': 'exclude_complete_active_and_container_pools_from_clutter_v1',
        'index': binding(index), 'containers': binding(containers),
        'exclude_source_ids': exclusion_ids, 'excluded_source_count': len(exclusion_ids),
        'container_partition_conflicts': [r for r in container_rejected if r.get('reason') in {
            'container_active_source_partition_conflict', 'container_source_partition_conflict'}],
        'labels_rewritten': False, 'scope': 'full known input pool identities across roles; not only selected scene'})
    pool.sort(key=lambda a: digest([seed, a['source_id']]))
    groups = {}; selection_rejections = []
    for i, anchor in enumerate(pool):
        try: groups[anchor['source_id']] = choose_group(anchor, pool, container_pool, i, seed)
        except Rejected as error: selection_rejections.append({'source_id': anchor['source_id'], **_failure(error)})
    ready = [a for a in pool if a['source_id'] in groups]
    write(output / 'inventory.json', {'ready_unique_assets': len(ready),
        'assets': [{'source_id': a['source_id'], 'partition': a['partition'],
                    'radius_m': a['radius'], **a['entry']} for a in ready],
        'partition_counts': dict(Counter(a['partition'] for a in ready)),
        'asset_rejections': rejected, 'container_rejections': container_rejected,
        'role_selection_rejections': selection_rejections})
    if len(ready) < max(minimum, scenes):
        status = {'status': 'insufficient_assets', 'requested_scenes': scenes,
                  'requested_tasks': scenes * TASKS_PER_SCENE, 'requested_unique_assets': max(minimum, scenes),
                  'ready_unique_assets': len(ready), 'generated_scenes': 0, 'generated_tasks': 0,
                  'reason': 'one distinct ready source anchor per scene; no seed-based asset substitution',
                  'physical_executions': 0, 'api_calls': 0}
        write(output / 'status.json', status); return status
    selected = ready[:scenes]
    # Recompute groups with final scene numbers; all choices precede geometry.
    assignments = [(a, choose_group(a, pool, container_pool, i, seed)) for i, a in enumerate(selected)]
    write(output / 'selection.json', {'rows': [{'scene_index': i, 'anchor': a['source_id'],
        'source_ids': [x['source_id'] for x in group], 'container_source_ids': [x['source_id'] for x in cs],
        'partition': a['partition'], **details} for i, (a, (group, cs, details)) in enumerate(assignments)],
        'source_replacement_after_geometry_or_physics': False})
    all_requests, rows, scene_rows, source_splits = [], [], [], defaultdict(set)
    entries = [a['entry'] for a in pool]; container_bindings = [c['binding'] for c in container_pool]
    for number, (anchor, (group, cs, selection)) in enumerate(assignments):
        attempts = []; compiled = tasks = checks = None
        scene_start = time.monotonic()
        for layout_number in range(len(LAYOUTS)):
            spec, layout = make_request(group, cs, number, seed, layout_number, exclude_source_ids=exclusion_ids)
            folder = output / 'scenes' / spec['id']; proposal = folder / 'layout_candidates' / f'{layout_number:02d}.json'
            write(proposal, spec)
            try:
                compiled, tasks, checks = compile_scene(spec, entries, container_bindings, seed + number)
                attempts.append({**layout, 'request': binding(proposal), 'accepted': True}); break
            except (Rejected, ValueError, KeyError, TypeError, OSError) as error:
                attempts.append({**layout, 'request': binding(proposal), 'accepted': False, **_failure(error)})
        accepted = bool(attempts[-1]['accepted'])
        write(folder / 'request.json', spec); source_request = binding(folder / 'request.json')
        write(folder / 'geometry_preflight.json', {'passed': accepted, 'attempts': attempts,
            'checks': checks, 'seed': seed + number, 'seconds': time.monotonic() - scene_start,
            'ik_status': 'not_evaluated', 'physical_status': 'not_evaluated'})
        if accepted:
            write(folder / 'compiled_scene.json', compiled)
            write(folder / 'compiled_tasks.json', {'tasks': tasks})
        source_ids = sorted(a['source_id'] for a in group)
        for sid in source_ids: source_splits[sid].add(anchor['partition'])
        for task in spec['tasks']:
            rows.append({'id': spec['id'] + '_' + task['id'], 'scene_id': spec['id'],
                'partition': anchor['partition'], 'mechanism': 'pick_place', 'goal_family': task['id'],
                'source_ids': source_ids, 'requested_task_id': task['id'], 'source_request': source_request,
                'split_group': source_ids, 'source_coverage': 'synthetic_only',
                'semantic_status': 'geometric_program; asset_category_unverified',
                'geometry_status': 'passed' if accepted else 'rejected',
                'physical_status': 'not_evaluated', 'human_video_status': 'not_evaluated'})
        all_requests.append(spec)
        scene_rows.append({'scene_id': spec['id'], 'partition': anchor['partition'],
            'request': source_request, 'geometry_preflight': binding(folder / 'geometry_preflight.json'),
            'accepted_geometry': accepted, 'seed': seed + number, 'selection': selection})
    chunks = []
    for start in range(0, scenes, CHUNK_SCENES):
        part = all_requests[start:start + CHUNK_SCENES]
        value = request_index({'schema': SCHEMA, 'scenes': part})
        path = output / 'chunks' / f'chunk_{start // CHUNK_SCENES:03d}.json'; write(path, value)
        chunks.append({'chunk': start // CHUNK_SCENES, 'requests': binding(path), 'seed_start': seed + start,
                       'scene_ids': [s['id'] for s in part], 'task_count': len(part) * TASKS_PER_SCENE})
    write(output / 'chunks.json', {'chunks': chunks,
        'registration': 'requested_tasks.register(..., seed=chunk.seed_start); arm selection remains independent'})
    write(output / 'coverage_manifest.json', {'schema': 'cached_pickplace_coverage_manifest_v1', 'rows': rows})
    write(output / 'source_splits.json', {'source_partitions': {sid: sorted(parts) for sid, parts in sorted(source_splits.items())},
        'scene_derivatives_share_partition': True, 'shared_static_props_excluded': True,
        'cross_role_source_policy': binding(output / 'source_role_policy.json')})
    write(output / 'scenes.json', {'scenes': scene_rows})
    status = {'status': 'ready_for_independent_ik' if all(s['accepted_geometry'] for s in scene_rows) else 'geometry_rejections',
        'generated_scenes': len(all_requests), 'generated_tasks': len(rows), 'unique_manipulable_sources': len(source_splits),
        'minimum_unique_assets': minimum, 'distinct_anchor_sources': len(selected),
        'geometry_passed_scenes': sum(s['accepted_geometry'] for s in scene_rows),
        'geometry_rejected_scenes': sum(not s['accepted_geometry'] for s in scene_rows),
        'goal_family_counts': dict(Counter(r['goal_family'] for r in rows)), 'contact_mechanisms': ['pick_place'],
        'new_contact_mechanisms': 0, 'chunks': len(chunks), 'seconds': time.monotonic() - started,
        'physical_executions': 0, 'api_calls': 0, 'optimizer_calls': 0,
        'coverage_status': 'requires_independent_coverage_audit_v1', 'physical_status': 'not_evaluated',
        'human_video_status': 'not_evaluated', 'failures_retained': True}
    if len(source_splits) < minimum:
        status['status'] = 'insufficient_assets'
    write(output / 'status.json', status)
    return status


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--containers', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenes', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--min-unique-assets', type=int)
    a = p.parse_args()
    result = build(a.index, a.containers, a.output, scenes=a.scenes, seed=a.seed, min_unique_assets=a.min_unique_assets)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(2 if result['status'] == 'insufficient_assets' else 0)
