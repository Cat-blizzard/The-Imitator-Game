"""Separate geometric IK search bounds from final control-command limits.

The v2 proposal incorrectly used the 0.08 rad command-step guard to bound a
geometric waypoint search. Near a singular configuration a 2 mm TCP move can
need a larger joint displacement that a retimer must subdivide. This candidate
expands only that search region, deterministically and within native limits.
The final PCHIP/TOPPRA path still requires the unchanged full path guard.
"""
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk
from .shape_preserving_toppra_v2 import PlannerAdapter

VERSION = 'geometric_joint_continuation_v3'
SEARCH_RADII = (.079, .16, .32, .40)


def cartesian_path(planner, root, current, goal, timestep=.05, waypoint_m=.002):
    current, goal = np.asarray(current, float), np.asarray(goal, float)
    if (current.shape != (9,) or goal.shape != (4, 4)
            or not np.isfinite(current).all() or not np.isfinite(goal).all()):
        raise ValueError('finite_native_panda_state_required')
    if not .001 <= waypoint_m <= .004 or not .01 <= timestep <= .10:
        raise ValueError('bounded_resolution_required')
    start = world_fk(current[None, :7], root)[0]
    delta = goal[:3, 3] - start[:3, 3]
    correction = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).as_rotvec()
    if np.linalg.norm(delta[:2]) > .001 or not .002 <= delta[2] <= .30:
        raise ValueError('bounded_vertical_lift_only')
    if np.linalg.norm(correction) > .01:
        raise ValueError('lift_orientation_must_remain_fixed')
    count = int(np.ceil(delta[2] / waypoint_m))
    limits = np.asarray(planner.joint_limits, float)[:7]
    q = current[:7].copy()
    values, traces = [q.copy()], []
    began = time.monotonic()
    for i in range(1, count + 1):
        fraction = i / count
        target = start.copy()
        target[:3, 3] += delta * fraction
        target[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(correction * fraction).as_matrix()
        previous, attempts = q.copy(), []

        def residual(joints):
            pose = world_fk(joints[None, :], root)[0]
            translation = pose[:3, 3] - target[:3, 3]
            rotation = Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec()
            return np.r_[translation, .15 * rotation, .00001 * (joints - previous)]

        for radius in SEARCH_RADII:
            if time.monotonic() - began > 20:
                raise TimeoutError('finite_continuation_time_limit')
            lower = np.maximum(limits[:, 0] + 1e-7, previous - radius)
            upper = np.minimum(limits[:, 1] - 1e-7, previous + radius)
            if np.any(lower >= upper):
                raise ValueError('empty_joint_trust_region')
            solution = least_squares(residual, np.clip(previous, lower, upper),
                bounds=(lower, upper), method='trf', max_nfev=60,
                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            pose = world_fk(solution.x[None, :], root)[0]
            pe = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
            re = float(Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).magnitude())
            attempts.append({'radius_rad': radius, 'position_error_m': pe,
                'rotation_error_rad': re, 'nfev': solution.nfev,
                'maximum_geometric_joint_step_rad': float(np.max(np.abs(solution.x - previous)))})
            if pe <= .0002 and re <= .002:
                q = solution.x.copy()
                break
        else:
            raise RuntimeError(f'geometric_continuation_endpoint_failed:{i}:{pe}:{re}')
        traces.append({'waypoint': i, 'attempts': attempts,
            'position_error_m': pe, 'rotation_error_rad': re,
            'geometric_search_radius_rad': radius,
            'maximum_geometric_joint_step_rad': float(np.max(np.abs(q - previous)))})
        values.append(q.copy())
    times, positions, velocities, accelerations, duration = PlannerAdapter(planner).TOPP(
        np.asarray(values), timestep)
    return {'status': 'Success', 'position': positions, 'velocity': velocities,
        'acceleration': accelerations, 'time': times, 'duration': duration,
        'cartesian_waypoints': count, 'continuation_trace': traces,
        'proposal_seconds': time.monotonic() - began,
        'requires_retained_full_path_guard': True, 'policy': VERSION,
        'retiming_policy': 'shape_preserving_toppra_v2',
        'geometric_waypoints_are_not_control_commands': True}
