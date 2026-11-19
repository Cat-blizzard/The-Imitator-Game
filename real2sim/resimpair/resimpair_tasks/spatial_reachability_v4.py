"""Bounded continuous layouts conditioned on a common source/goal arm.

This derives a new candidate factory; original requests and failed executions
are immutable. Sphere membership uses actual cached TCPs, robot roots and all
four endpoint stages. It is only a sampling condition; retained IK and physical
checks remain necessary. No physical or API attempt occurs in this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected, compile_scene
from resimpair_pickplace.cartesian_reachability import pose
from .spatial_factory_v2 import sample_request, SOURCE_XY, DESTINATION_XY

VERSION = 'same_arm_spatial_factory_v4'
MAX_PROPOSALS = 16
SAMPLING_RADIUS_M = .90
RUNTIME_RADIUS_M = .95
MAX_POSES_PER_OBJECT = 12


def endpoint_transforms(source, destination, candidate):
    local = np.asarray(candidate['tcp_in_asset'], dtype=float)
    grasp, release = source @ local, destination @ local
    pregrasp, above_goal = grasp.copy(), release.copy()
    pregrasp[2, 3] += .17; above_goal[2, 3] += .18
    return [('pregrasp', pregrasp), ('grasp', grasp), ('above_goal', above_goal), ('release', release)]


def common_arm(source, destination, candidates, robot_roots, radius=SAMPLING_RADIUS_M):
    roots = np.asarray(robot_roots, dtype=float)
    if roots.shape != (2, 3) or not np.isfinite(roots).all() or not 0 < radius <= RUNTIME_RADIUS_M:
        raise ValueError('two_pinned_robot_roots_and_bounded_radius_required')
    rows = []; selected = None
    for candidate in candidates[:MAX_POSES_PER_OBJECT]:
        stages = endpoint_transforms(source, destination, candidate)
        for arm, root in enumerate(roots):
            distances = [float(np.linalg.norm(T[:3, 3] - root)) for _, T in stages]
            passed = max(distances) <= radius
            row = {'candidate': candidate['id'], 'arm': arm, 'passed': passed,
                   'endpoint_distances_m': dict(zip([s[0] for s in stages], distances)),
                   'maximum_distance_m': max(distances)}
            rows.append(row)
            if passed and (selected is None or row['maximum_distance_m'] < selected['maximum_distance_m']):
                selected = {**row, 'world_tcp_poses': {stage: T.tolist() for stage, T in stages}}
    return {'passed': selected is not None, 'selected_geometric_proposal': selected,
            'candidates': rows, 'sampling_radius_m': radius, 'runtime_radius_unchanged_m': RUNTIME_RADIUS_M,
            'same_arm_for_all_stages': True, 'IK_status': 'not_run',
            'complete_path_and_physical_status': 'unmeasured'}


def workspace_checks(scene, tasks, radius=SAMPLING_RADIUS_M):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import targets
    objects = {o['id']: o for o in scene['objects']}
    caches = {oid: read(resolve(b)) for oid, b in scene['grasp_caches'].items()}
    reports = []
    for task in tasks:
        destination = targets(scene, task)
        for operation in task['operations']:
            obj = objects[operation['object']]
            start = pose(obj['position'], obj['yaw'])
            end = pose(np.asarray(destination[obj['id']]) + [0., 0., .002], operation['target']['yaw'])
            report = common_arm(start, end, caches[obj['id']]['candidates'], scene['robot_roots'], radius)
            reports.append({'task': task['requested_task_id'], 'operation': operation['id'],
                            'object': obj['id'], **report})
    return {'passed': all(r['passed'] for r in reports), 'operations': reports,
            'rejected_operations': sum(not r['passed'] for r in reports),
            'scope': 'sampling condition on cached TCP positions, not a collision/IK certificate'}


def proposal(original, assets, containers, seed):
    request, trace = sample_request(original, assets, containers, seed)
    request['id'] = original['id'].removesuffix('_spatialv2') + '_reachablev4'
    # Preserve distinct same-container packing from v3 when the layout sampler
    # independently changes offsets. Rotate the whole group, never individual
    # offsets into a collision. The circular footprint certificate is invariant.
    rng = np.random.default_rng(seed + 17)
    original_tasks = {t['id']: t for t in original['tasks']}
    packing = []
    for task in request['tasks']:
        groups = defaultdict(list)
        for i, placement in enumerate(task['placements']):
            target = placement['target']
            if target['kind'] == 'container': groups[target['container']].append(i)
        for container, indices in groups.items():
            if len(indices) < 2: continue
            angle = float(rng.uniform(-np.pi, np.pi))
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            offsets = []
            for i in indices:
                old = original_tasks[task['id']]['placements'][i]['target']['offset_xy_m']
                new = np.round(R @ np.asarray(old), 6).tolist()
                task['placements'][i]['target']['offset_xy_m'] = new; offsets.append(new)
            packing.append({'task': task['id'], 'container': container, 'rotation_rad': angle,
                            'offsets_local_xy_m': offsets, 'relative_packing_preserved': True})
    trace.update(schema=VERSION, simultaneous_container_placements=packing,
                 cache_scale_and_support_orientation_unchanged=True)
    return request, trace


def endpoint_qualification(output, rows, number):
    from resimpair_pickplace.cartesian_reachability import bind_selection
    directory = Path(output) / 'endpoint_qualification'
    selected = rows[:number]
    write(directory / 'selection.json', {'scene_ids': [r['scene_id'] for r in selected],
        'rule': 'first N derived scene rows, all original goal programs; no outcome-based replacement',
        'physical_attempts': 0, 'api_calls': 0})
    results = []; started = time.monotonic()
    for row in selected:
        folder = Path(output) / 'scenes' / row['scene_id']
        if not row['accepted_geometry']:
            results.append({'scene_id': row['scene_id'], 'requested_tasks': row['requested_tasks'],
                            'eligible_tasks': 0, 'reason': 'geometry_proposals_exhausted'}); continue
        scene = read(folder / 'compiled_scene.json'); tasks = read(folder / 'compiled_tasks.json')['tasks']
        task_results = []
        for task in tasks:
            task_dir = directory / row['scene_id'] / task['requested_task_id']
            selected_scene, assigned = bind_selection(scene, task, task_dir)
            passed = selected_scene is not None
            if passed:
                write(task_dir / 'scene.json', selected_scene); write(task_dir / 'task.json', assigned)
            task_results.append({'task': task['requested_task_id'], 'endpoint_eligible': passed,
                                 'report': binding(task_dir / 'arm_selection.json')})
        results.append({'scene_id': row['scene_id'], 'requested_tasks': len(tasks),
                        'eligible_tasks': sum(r['endpoint_eligible'] for r in task_results), 'tasks': task_results})
    summary = {'schema': VERSION, 'scenes': number, 'requested_tasks': sum(r['requested_tasks'] for r in results),
        'endpoint_eligible_tasks': sum(r['eligible_tasks'] for r in results),
        'all_tasks_eligible_scenes': sum(r['eligible_tasks'] == r['requested_tasks'] for r in results),
        'seconds': time.monotonic() - started, 'results': results, 'physical_attempts': 0,
        'api_calls': 0, 'complete_path_success': 'unmeasured', 'default_promoted': False}
    write(directory / 'summary.json', summary)
    return summary


def build(factory, output, *, seed=2026102903, maximum_proposals=MAX_PROPOSALS,
          radius=SAMPLING_RADIUS_M, ik_scenes=6):
    factory, output = Path(factory).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('existing_derived_factory_must_not_be_overwritten')
    if (type(maximum_proposals) is not int or not 1 <= maximum_proposals <= MAX_PROPOSALS
            or type(seed) is not int or seed < 0 or not 0 < radius <= RUNTIME_RADIUS_M):
        raise ValueError('bounded_geometry_proposal_configuration_required')
    parent = read(factory / 'protocol.json'); parent_rows = read(factory / 'scenes.json')['scenes']
    if type(ik_scenes) is not int or not 0 <= ik_scenes <= min(6, len(parent_rows)):
        raise ValueError('at_most_first_six_scenes_for_endpoint_qualification')
    entries = read(resolve(parent['index']))['entries']; container_entries = read(resolve(parent['containers']))['assets']
    by_source = {read(resolve(e['asset']))['source_id']: e for e in entries}
    container_by_source = {read(resolve(b))['source_id']: b for b in container_entries}
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'parent_factory': binding(factory / 'protocol.json'), 'parent_scenes': binding(factory / 'scenes.json'),
        'index': parent['index'], 'containers': parent['containers'],
        'compiler': binding(ROOT / 'resimpair_pickplace/requested_tasks.py'),
        'spatial_sampler': binding(ROOT / 'resimpair_tasks/spatial_factory_v2.py'),
        'endpoint_selector': binding(ROOT / 'resimpair_pickplace/cartesian_reachability.py'),
        'seed': seed, 'maximum_proposals_per_scene': maximum_proposals, 'sampling_radius_m': radius,
        'runtime_radius_unchanged_m': RUNTIME_RADIUS_M, 'cached_pose_limit': MAX_POSES_PER_OBJECT,
        'source_xy_bounds_m': SOURCE_XY.tolist(), 'destination_xy_bounds_m': DESTINATION_XY.tolist(),
        'endpoint_qualification_first_scenes': ik_scenes, 'physical_attempts': 0, 'api_calls': 0, 'gpu_jobs': 0,
        'old_requests_preserved': True, 'fixed_assets_containers_sizes_partitions_and_goal_types': True,
        'outcome_based_layout_replacement': False, 'default_promoted': False})
    started = time.monotonic(); rows = []; proposal_counts = Counter(); rejection_types = Counter()
    source_points = []; goal_points = []; container_points = []; source_ids = set(); container_ids = set()
    for index, parent_row in enumerate(parent_rows):
        original = read(resolve(parent_row['request']))
        identity = original['id'].removesuffix('_spatialv2') + '_reachablev4'
        folder = output / 'scenes' / identity
        selected_entries = [by_source[o['source_id']] for o in original['objects']]
        selected_containers = [container_by_source[c['source_id']] for c in original['containers']]
        assets = {read(resolve(e['asset']))['source_id']: read(resolve(e['asset'])) for e in selected_entries}
        containers = {read(resolve(b))['source_id']: read(resolve(b)) for b in selected_containers}
        attempts = []; chosen = None; last_failure = None
        for number in range(maximum_proposals):
            current = folder / 'proposals' / f'{number:02d}'; current_seed = seed + index * 101 + number
            request = None
            try:
                request, trace = proposal(original, assets, containers, current_seed)
                write(current / 'request.json', request); write(current / 'sampling.json', trace)
                scene, tasks, checks = compile_scene(request, selected_entries, selected_containers, current_seed)
                workspace = workspace_checks(scene, tasks, radius)
                write(current / 'workspace.json', workspace)
                if not workspace['passed']:
                    raise Rejected('no_same_arm_cached_TCP_workspace', rejected_operations=workspace['rejected_operations'])
                chosen = {'proposal_index': number, 'seed': current_seed,
                          'request': binding(current / 'request.json'), 'workspace': binding(current / 'workspace.json')}
                write(folder / 'request.json', request); write(folder / 'compiled_scene.json', scene)
                write(folder / 'compiled_tasks.json', {'tasks': tasks}); write(folder / 'geometry_checks.json', checks)
                source_points.extend(o['position'][:2] for o in scene['objects'] if o['kind'] == 'movable')
                container_points.extend(c['position'][:2] for c in scene['supports'])
                goal_points.extend(g['world_position_m'][:2] for g in checks['goals'])
                source_ids.update(o['source_id'] for o in request['objects'])
                container_ids.update(c['source_id'] for c in request['containers'])
                attempts.append({'proposal_index': number, 'seed': current_seed, 'accepted': True, **chosen}); break
            except (Rejected, ValueError) as error:
                last_failure = {'reason': error.code, 'details': error.details} if isinstance(error, Rejected) else {
                    'reason': type(error).__name__, 'details': {'message': str(error)}}
                write(current / 'rejection.json', last_failure); rejection_types[last_failure['reason']] += 1
                attempts.append({'proposal_index': number, 'seed': current_seed, 'accepted': False,
                    'failure': last_failure, 'request': binding(current / 'request.json') if request else None})
        proposal_counts[len(attempts)] += 1
        write(folder / 'attempts.json', {'attempts': attempts, 'chosen': chosen, 'physical_attempts': 0})
        row = {'scene_id': identity, 'parent_scene_id': parent_row['scene_id'], 'parent_request': parent_row['request'],
            'partition': parent_row['partition'], 'source_ids': parent_row['source_ids'],
            'requested_tasks': len(original['tasks']), 'accepted_geometry': chosen is not None,
            'seed': chosen['seed'] if chosen else None, 'request': binding(folder / 'request.json') if chosen else None,
            'attempts': binding(folder / 'attempts.json'), 'proposal_count': len(attempts),
            'failure': last_failure if chosen is None else None, 'physics_status': 'not_evaluated'}
        rows.append(row)
        if index % 10 == 0: print({'scene': index+1, 'accepted': chosen is not None, 'proposals': len(attempts)}, flush=True)
    write(output / 'scenes.json', {'scenes': rows})
    layout_seconds = time.monotonic() - started
    endpoints = endpoint_qualification(output, rows, ik_scenes) if ik_scenes else None
    span = lambda points: np.ptp(points, axis=0).tolist() if points else None
    summary = {'schema': VERSION, 'requested_scenes': len(rows), 'requested_tasks': sum(r['requested_tasks'] for r in rows),
        'geometry_and_workspace_passed_scenes': sum(r['accepted_geometry'] for r in rows),
        'exhausted_scenes': sum(not r['accepted_geometry'] for r in rows),
        'total_geometry_proposals': sum(r['proposal_count'] for r in rows),
        'proposal_count_distribution': {str(k): v for k, v in sorted(proposal_counts.items())},
        'rejection_types': dict(rejection_types), 'source_XY_span_m': span(source_points),
        'container_XY_span_m': span(container_points), 'goal_XY_span_m': span(goal_points),
        'selected_unique_source_assets': len(source_ids), 'selected_unique_containers': len(container_ids),
        'layout_seconds': layout_seconds, 'endpoint_qualification': binding(output / 'endpoint_qualification/summary.json') if endpoints else None,
        'physical_attempts': 0, 'api_calls': 0, 'gpu_jobs': 0, 'default_promoted': False}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factory', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2026102903); p.add_argument('--maximum-proposals', type=int, default=16)
    p.add_argument('--radius', type=float, default=.90); p.add_argument('--ik-scenes', type=int, default=6)
    a = p.parse_args()
    print(build(a.factory, a.output, seed=a.seed, maximum_proposals=a.maximum_proposals,
                radius=a.radius, ik_scenes=a.ik_scenes))
