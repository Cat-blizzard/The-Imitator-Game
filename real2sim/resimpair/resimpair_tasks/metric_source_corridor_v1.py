"""CPU-only placement proposals conditioned on a cached source approach/lift.

Endpoint IK alone does not show that a fixed-orientation vertical path can be
followed on one joint branch. This optional scene sampler uses the *retained*
local path and command limits. It does not certify collisions, grasp forces,
transport, target capacity or physical execution. Those checks remain live.
"""
from __future__ import annotations

import time

import numpy as np

from resimpair_runtime import configure
configure()
from resimpair_grasp.common import binding, read, resolve
from resimpair_pickplace.cartesian_reachability import ArmSelector, INITIAL, pose
from resimpair_pickplace.cartesian_transfer import local_cartesian_path
from resimpair_pickplace.shape_preserving_toppra_v2 import PlannerAdapter
from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk, validate_joint_path

VERSION = 'metric_source_corridor_sampling_v1'


def qualify(scene, object_id, selector=None, max_candidates=12, timeout_seconds=12.):
    """Return a finite sampling diagnostic, never a physical pass verdict."""
    if not 1 <= max_candidates <= 12 or not 0 < timeout_seconds <= 30:
        raise ValueError('bounded_cached_pose_and_time_limits_required')
    item = next(o for o in scene['objects'] if o['id'] == object_id)
    cache_ref = scene['grasp_caches'][object_id]
    candidates = read(resolve(cache_ref))['candidates'][:max_candidates]
    selector = selector or ArmSelector(scene['robot_roots'], scene['seed'])
    start = pose(item['position'], item['yaw'])
    began = time.monotonic(); decisions = []; selected = None; timed_out = False
    # Alternate arms so a short budget is not spent entirely on one robot.
    for candidate in candidates:
        for arm in range(len(selector.planners)):
            if time.monotonic() - began >= timeout_seconds:
                timed_out = True
                break
            row = {'candidate': candidate['id'], 'arm': arm, 'passed': False}
            decisions.append(row)
            grasp = start @ np.asarray(candidate['tcp_in_asset'], dtype=float)
            approach = grasp.copy(); approach[2, 3] += .17
            q, status = selector.ik(arm, approach, INITIAL)
            row['pregrasp_IK_status'] = status
            if q is None:
                row['reason'] = 'pregrasp_endpoint_unavailable'
                continue
            planner = PlannerAdapter(selector.planners[arm]); root = selector.roots[arm]
            try:
                descent = local_cartesian_path(planner, root, q, grasp, 'grasp_reach')
                check = validate_joint_path(descent['position'], planner.joint_limits, q)
                row['descent_commands'] = check
                if not check['passed']:
                    raise ValueError('retained_descent_command_check_failed')
                at_grasp = q.copy(); at_grasp[:7] = descent['position'][-1]
                at_grasp[7:] = float(candidate['width_m']) / 2
                lift = world_fk(at_grasp[None, :7], root)[0]
                lift[2, 3] += .20
                ascent = local_cartesian_path(planner, root, at_grasp, lift, 'lift')
                check = validate_joint_path(ascent['position'], planner.joint_limits, at_grasp)
                row['lift_commands'] = check
                if not check['passed']:
                    raise ValueError('retained_lift_command_check_failed')
                row['passed'] = True
                selected = {'arm': arm, 'candidate': candidate['id']}
            except (RuntimeError, ValueError, TimeoutError) as error:
                row['reason'] = str(error)
            if selected is not None:
                break
        if selected is not None or timed_out:
            break
    return {'schema': VERSION, 'object': object_id, 'passed': selected is not None,
            'selected': selected, 'cache': cache_ref, 'implementation': binding(__file__),
            'decisions': decisions, 'timed_out': timed_out, 'seconds': time.monotonic() - began,
            'timeout_seconds': timeout_seconds, 'maximum_candidates': max_candidates,
            'nominal_initial_source_pose_only': True, 'runtime_kinematics_and_command_limits_unchanged': True,
            'complete_collision_path_status': 'unmeasured', 'physical_grasp_status': 'unmeasured',
            'target_reachability_status': 'unmeasured', 'physical_attempts': 0, 'API_calls': 0, 'GPU_jobs': 0}
