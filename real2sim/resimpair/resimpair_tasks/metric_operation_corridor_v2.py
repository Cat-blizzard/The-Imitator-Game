"""Same-arm source/goal local kinematic sampling, before physical execution.

The earlier source-only diagnostic accepts a pose on an arm that cannot reach
the destination. This candidate preserves the existing endpoint/cavity selector
and checks descent, lift, placement and withdrawal on one arm and cached pose.
It is only a sampling condition; live full collision and physical checks remain.
"""
from __future__ import annotations

import time
import numpy as np

from resimpair_runtime import configure
configure()
from resimpair_grasp.common import binding, read, resolve, digest
from resimpair_pickplace.cartesian_reachability import ArmSelector, pose
from resimpair_pickplace.cartesian_transfer import local_cartesian_path
from resimpair_pickplace.shape_preserving_toppra_v2 import PlannerAdapter
from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk, validate_joint_path
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import targets

VERSION = 'metric_same_arm_operation_corridor_v2'


def qualify(scene, operation, selector=None, timeout_seconds=12.):
    if not 0 < timeout_seconds <= 30:
        raise ValueError('bounded_sampling_time_required')
    selector = selector or ArmSelector(scene['robot_roots'], scene['seed'])
    obj = next(o for o in scene['objects'] if o['id'] == operation['object'])
    support = next((s for s in scene['supports'] if s['id'] == operation['target']['support']), None)
    began = time.monotonic()
    endpoint = selector.select(scene, operation, support if support and 'container_descriptor' in support else None)
    rows = sorted((r for r in endpoint['decisions'] if r['accepted']), key=lambda r: r['joint_travel_proxy'])
    cache_ref = scene['grasp_caches'][obj['id']]
    cache = {c['id']: c for c in read(resolve(cache_ref))['candidates'][:12]}
    source = pose(obj['position'], obj['yaw'])
    target_xyz = targets(scene, {'operations': [operation]})[obj['id']]
    destination = pose(np.asarray(target_xyz) + [0., 0., .002], operation['target']['yaw'])
    decisions = []; selected = None; timed_out = False
    for r in rows:
        if time.monotonic() - began >= timeout_seconds:
            timed_out = True
            break
        arm = r['arm']; candidate = cache[r['candidate']]
        planner = PlannerAdapter(selector.planners[arm]); root = selector.roots[arm]
        q = {v['stage']: np.asarray(v['qpos'], float) for v in r['checks']}
        row = {'candidate': candidate['id'], 'arm': arm, 'passed': False, 'stages': []}
        decisions.append(row)
        grasp = source @ np.asarray(candidate['tcp_in_asset'])
        release = destination @ np.asarray(candidate['tcp_in_asset'])
        def move(current, goal, stage):
            path = local_cartesian_path(planner, root, current, goal, stage)
            check = validate_joint_path(path['position'], planner.joint_limits, current)
            row['stages'].append({'stage': stage, 'command_check': check})
            if not check['passed']:
                raise ValueError('retained_command_check_failed:' + stage)
            new = current.copy(); new[:7] = path['position'][-1]
            return new
        try:
            at_grasp = move(q['pregrasp'], grasp, 'grasp_reach')
            at_grasp[7:] = float(candidate['width_m']) / 2
            lift = world_fk(at_grasp[None, :7], root)[0]; lift[2, 3] += .20
            move(at_grasp, lift, 'lift')
            # For geometric kinematics a descending placement is the same
            # local continuation as grasp reach; no physical contact is assumed.
            at_release = move(q['above_goal'], release, 'grasp_reach')
            at_release[7:] = .04
            retreat = world_fk(at_release[None, :7], root)[0]; retreat[2, 3] += .18
            move(at_release, retreat, 'retreat')
            row['passed'] = True
            selected = {'arm': arm, 'candidate': candidate['id']}
            break
        except (RuntimeError, ValueError, TimeoutError) as error:
            row['reason'] = str(error)
    return {'schema': VERSION, 'operation': operation['id'], 'object': obj['id'],
            'passed': selected is not None, 'selected': selected, 'cache': cache_ref,
            'endpoint_report': endpoint, 'decisions': decisions, 'timed_out': timed_out,
            'seconds': time.monotonic() - began, 'timeout_seconds': timeout_seconds,
            'nominal_pose_only': True, 'same_arm_and_cached_grasp_for_source_and_goal': True,
            'collision_and_physical_status': 'unmeasured', 'physical_attempts': 0, 'GPU_jobs': 0, 'API_calls': 0}


def qualify_scene(scene, tasks, timeout_seconds=45.):
    """Deduplicate shared goals, retain all requested tasks and stop on failure."""
    if not 0 < timeout_seconds <= 60:
        raise ValueError('bounded_scene_sampling_time_required')
    began = time.monotonic(); selector = ArmSelector(scene['robot_roots'], scene['seed'])
    seen = {}; rows = []; failure = None; required = []
    for task in tasks:
        for op in task['operations']:
            required.append({'task': task['requested_task_id'], 'operation': op['id'],
                             'key': digest([op['object'], op['target']])})
            key = required[-1]['key']
            if key in seen:
                continue
            if time.monotonic() - began >= timeout_seconds:
                failure = 'bounded_scene_corridor_time_exhausted'
                break
            result = qualify(scene, op, selector, min(12., max(.001, timeout_seconds - (time.monotonic() - began))))
            seen[key] = len(rows)
            rows.append({'task': task['requested_task_id'], 'key': key, **result})
            if not result['passed']:
                failure = 'same_arm_local_corridor_unavailable'
                break
        if failure:
            break
    return {'schema': VERSION, 'passed': failure is None, 'failure': failure,
            'implementation': binding(__file__), 'operations': rows, 'references_checked': required,
            'required_tasks': [t['requested_task_id'] for t in tasks],
            'requested_operation_count': sum(len(t['operations']) for t in tasks),
            'checked_unique_operation_count': len(rows), 'seconds': time.monotonic() - began,
            'scope': 'sampling condition on nominal local paths; complete collision checks and physics remain mandatory',
            'physical_attempts': 0, 'GPU_jobs': 0, 'API_calls': 0}
