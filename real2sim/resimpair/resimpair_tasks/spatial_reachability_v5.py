"""Six-scene candidate: admit random layouts only after required endpoint IK.

The required goal set is declared before proposals. Each original source scene
gets at most 16 deterministic layout proposals, independent of physical results.
Neither old failures nor acceptance tolerances are changed. This module never
runs physics, a video model or a pose optimizer.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected, compile_scene
from resimpair_pickplace.cartesian_reachability import bind_selection
from . import spatial_reachability_v4 as geometry

VERSION = 'IK_conditioned_spatial_factory_v5'


def required_goals(request, goals):
    known = [task['id'] for task in request['tasks']]
    if goals == ['all'] or goals == ('all',):
        return known
    if not goals or len(set(goals)) != len(goals) or set(goals) - set(known):
        raise ValueError('required_goal_set_must_exist_in_each_source_scene')
    return list(goals)


def qualify(scene, tasks, directory):
    """Stop at the first rejection; acceptance requires every requested goal."""
    directory = Path(directory); results = []; started = time.monotonic()
    for task in tasks:
        path = directory / task['requested_task_id']
        selected, assigned = bind_selection(scene, task, path)
        passed = selected is not None
        if passed:
            write(path / 'scene.json', selected); write(path / 'task.json', assigned)
        results.append({'task': task['requested_task_id'], 'passed': passed,
                        'report': binding(path / 'arm_selection.json')})
        if not passed: break
    report = {'passed': len(results) == len(tasks) and all(r['passed'] for r in results),
        'required_tasks': [t['requested_task_id'] for t in tasks], 'checked': results,
        'not_checked': [t['requested_task_id'] for t in tasks[len(results):]],
        'seconds': time.monotonic() - started, 'physical_success': 'unmeasured',
        'complete_path_success': 'unmeasured', 'nominal_initial_pose_only': True}
    write(directory / 'qualification.json', report)
    return report


def build(factory, output, *, scenes=6, goals=('all',), seed=2026102903,
          maximum_proposals=16, radius=.90):
    factory, output = Path(factory).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('preserve_existing_candidate_and_failures')
    all_rows = read(factory / 'scenes.json')['scenes']
    if (type(scenes) is not int or not 1 <= scenes <= min(6, len(all_rows))
            or type(maximum_proposals) is not int or not 1 <= maximum_proposals <= 16
            or type(seed) is not int or seed < 0 or not 0 < radius <= .95):
        raise ValueError('finite_six_scene_sixteen_proposal_configuration_required')
    selected = all_rows[:scenes]; parent = read(factory / 'protocol.json')
    entries = read(resolve(parent['index']))['entries']; container_entries = read(resolve(parent['containers']))['assets']
    by_source = {read(resolve(e['asset']))['source_id']: e for e in entries}
    bins = {read(resolve(b))['source_id']: b for b in container_entries}
    requests = [read(resolve(r['request'])) for r in selected]
    goal_sets = [required_goals(request, goals) for request in requests]
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'geometry_proposer': binding(geometry.__file__), 'parent_factory': binding(factory / 'protocol.json'),
        'parent_scenes': binding(factory / 'scenes.json'), 'selected_source_rows': selected,
        'index': parent['index'], 'containers': parent['containers'],
        'required_goal_sets': {r['scene_id']: g for r, g in zip(selected, goal_sets)},
        'selection': 'first six original source rows; goal sets fixed before IK; no physical feedback',
        'compiler': binding(ROOT / 'resimpair_pickplace/requested_tasks.py'),
        'endpoint_selector': binding(ROOT / 'resimpair_pickplace/cartesian_reachability.py'),
        'seed': seed, 'maximum_proposals_per_scene': maximum_proposals,
        'sampling_radius_m': radius, 'original_IK_tolerance_and_rules_unchanged': True,
        'assets_containers_sizes_partitions_goal_types_unchanged': True,
        'new_physical_attempt_limit': 0, 'api_limit': 0, 'gpu_jobs': 0, 'default_promoted': False})
    began = time.monotonic(); results = []; failures = Counter(); source_xy = []; container_xy = []
    ik_seconds = 0.; ik_checks = 0; proposals = 0
    for index, (row, original, goal_set) in enumerate(zip(selected, requests, goal_sets)):
        identity = original['id'].removesuffix('_spatialv2') + '_reachablev5'
        folder = output / 'scenes' / identity
        selected_entries = [by_source[o['source_id']] for o in original['objects']]
        selected_bins = [bins[c['source_id']] for c in original['containers']]
        assets = {read(resolve(e['asset']))['source_id']: read(resolve(e['asset'])) for e in selected_entries}
        containers = {read(resolve(b))['source_id']: read(resolve(b)) for b in selected_bins}
        attempts = []; accepted = None
        for number in range(maximum_proposals):
            current = folder / 'proposals' / f'{number:02d}'
            current_seed = seed + index * 101 + number; proposals += 1
            request = None
            try:
                request, trace = geometry.proposal(original, assets, containers, current_seed)
                request['id'] = identity
                write(current / 'request.json', request); write(current / 'sampling.json', trace)
                scene, tasks, checks = compile_scene(request, selected_entries, selected_bins, current_seed)
                required = [next(t for t in tasks if t['requested_task_id'] == goal) for goal in goal_set]
                workspace = geometry.workspace_checks(scene, required, radius)
                write(current / 'workspace.json', workspace)
                if not workspace['passed']:
                    raise Rejected('no_same_arm_cached_TCP_workspace', rejected_operations=workspace['rejected_operations'])
                proof = qualify(scene, required, current / 'endpoint_IK')
                ik_seconds += proof['seconds']; ik_checks += len(proof['checked'])
                if not proof['passed']:
                    raise Rejected('required_goal_endpoint_IK_failed', failed_goal=proof['checked'][-1]['task'],
                                   not_checked=proof['not_checked'])
                accepted = {'proposal_index': number, 'seed': current_seed,
                            'qualification': binding(current / 'endpoint_IK/qualification.json')}
                write(folder / 'request.json', request); write(folder / 'compiled_scene.json', scene)
                write(folder / 'compiled_tasks.json', {'tasks': tasks}); write(folder / 'geometry_checks.json', checks)
                source_xy.extend(o['position'][:2] for o in scene['objects'] if o['kind'] == 'movable')
                container_xy.extend(c['position'][:2] for c in scene['supports'])
                attempts.append({**accepted, 'accepted': True, 'request': binding(current / 'request.json')}); break
            except (Rejected, ValueError) as error:
                failure = {'reason': error.code, 'details': error.details} if isinstance(error, Rejected) else {
                    'reason': type(error).__name__, 'details': {'message': str(error)}}
                write(current / 'rejection.json', failure); failures[failure['reason']] += 1
                attempts.append({'proposal_index': number, 'seed': current_seed, 'accepted': False,
                    'failure': failure, 'request': binding(current / 'request.json') if request else None})
        write(folder / 'attempts.json', {'attempts': attempts, 'accepted': accepted})
        results.append({'scene_id': identity, 'parent_scene_id': row['scene_id'], 'parent_request': row['request'],
            'partition': row['partition'], 'source_ids': row['source_ids'], 'required_goals': goal_set,
            'requested_tasks': len(goal_set), 'accepted_geometry_and_IK': accepted is not None,
            'proposal_count': len(attempts), 'accepted': accepted, 'attempts': binding(folder / 'attempts.json'),
            'request': binding(folder / 'request.json') if accepted else None,
            'physical_status': 'not_evaluated', 'complete_path_status': 'not_evaluated'})
        print({'scene': index+1, 'accepted': accepted is not None, 'proposals': len(attempts)}, flush=True)
    write(output / 'scenes.json', {'scenes': results})
    summary = {'schema': VERSION, 'requested_scenes': len(results),
        'requested_goal_programs': sum(r['requested_tasks'] for r in results),
        'all_required_goals_endpoint_eligible_scenes': sum(r['accepted_geometry_and_IK'] for r in results),
        'endpoint_eligible_goal_programs': sum(r['requested_tasks'] for r in results if r['accepted_geometry_and_IK']),
        'exhausted_scenes': sum(not r['accepted_geometry_and_IK'] for r in results),
        'total_proposals': proposals, 'proposal_rejection_types': dict(failures),
        'endpoint_IK_task_checks_including_rejections': ik_checks,
        'endpoint_IK_seconds': ik_seconds, 'total_seconds': time.monotonic() - began,
        'source_XY_span_m': np.ptp(source_xy, axis=0).tolist() if source_xy else None,
        'container_XY_span_m': np.ptp(container_xy, axis=0).tolist() if container_xy else None,
        'physical_attempts': 0, 'api_calls': 0, 'gpu_jobs': 0, 'default_promoted': False,
        'acceptance_scope': 'required endpoint IK at nominal initial states; full paths and physics unmeasured'}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--factory', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenes', type=int, default=6); p.add_argument('--goals', nargs='+', required=True)
    p.add_argument('--seed', type=int, default=2026102903); p.add_argument('--maximum-proposals', type=int, default=16)
    p.add_argument('--radius', type=float, default=.90); a = p.parse_args()
    print(build(a.factory, a.output, scenes=a.scenes, goals=a.goals, seed=a.seed,
                maximum_proposals=a.maximum_proposals, radius=a.radius))
