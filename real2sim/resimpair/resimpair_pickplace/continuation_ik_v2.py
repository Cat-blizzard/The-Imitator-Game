"""Local joint-continuation proposals, certified by the unchanged mesh guard.

The retained Cartesian planner asks a general IK solver to enumerate branches
at each waypoint. This implementation follows the measured branch directly in
a bounded joint neighbourhood with deterministic least squares. It does not
change the workspace, dense collision checks, lift corridor or physics rules.
"""
from pathlib import Path
import argparse
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk

VERSION = 'bounded_joint_continuation_v2'


def cartesian_path(planner, root, current, goal, timestep=.05, waypoint_m=.002):
    current = np.asarray(current, float)
    goal = np.asarray(goal, float)
    if current.shape != (9,) or goal.shape != (4, 4) or not np.isfinite(current).all() or not np.isfinite(goal).all():
        raise ValueError('finite_native_panda_state_required')
    if not .001 <= waypoint_m <= .004 or not .01 <= timestep <= .10:
        raise ValueError('bounded_resolution_required')
    start = world_fk(current[None, :7], root)[0]
    delta = goal[:3, 3] - start[:3, 3]
    correction = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).as_rotvec()
    # Same local-domain restrictions as the existing upward Cartesian solver.
    if np.linalg.norm(delta[:2]) > .001 or not .002 <= delta[2] <= .30:
        raise ValueError('bounded_vertical_lift_only')
    if np.linalg.norm(correction) > .01:
        raise ValueError('lift_orientation_must_remain_fixed')
    count = int(np.ceil(delta[2] / waypoint_m))
    limits = np.asarray(planner.joint_limits, float)[:7]
    q = current[:7].copy()
    values = [q.copy()]
    traces = []
    began = time.monotonic()
    for i in range(1, count + 1):
        if time.monotonic() - began > 20:
            raise TimeoutError('finite_continuation_time_limit')
        fraction = i / count
        target = start.copy()
        target[:3, 3] += delta * fraction
        target[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(correction * fraction).as_matrix()
        previous = q.copy()

        def residual(joints):
            pose = world_fk(joints[None, :], root)[0]
            translation = pose[:3, 3] - target[:3, 3]
            rotation = Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec()
            # The small null-space term chooses a nearby branch, not a change
            # in the endpoint tolerances; errors are independently checked.
            return np.r_[translation, .15 * rotation, .00001 * (joints - previous)]

        lower = np.maximum(limits[:, 0] + 1e-7, previous - .079)
        upper = np.minimum(limits[:, 1] - 1e-7, previous + .079)
        if np.any(lower >= upper):
            raise ValueError('empty_joint_trust_region')
        solution = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper),
                                 method='trf', max_nfev=60, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        pose = world_fk(solution.x[None, :], root)[0]
        position_error = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
        rotation_error = float(Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).magnitude())
        if position_error > .0002 or rotation_error > .002:
            raise RuntimeError(f'continuation_endpoint_failed:{i}:{position_error}:{rotation_error}')
        q = solution.x.copy()
        traces.append({'waypoint': i, 'position_error_m': position_error,
                       'rotation_error_rad': rotation_error, 'nfev': solution.nfev,
                       'maximum_joint_step_rad': float(np.max(np.abs(q - previous)))})
        values.append(q.copy())
    times, positions, velocities, accelerations, duration = planner.TOPP(np.asarray(values), timestep)
    return {'status': 'Success', 'position': positions, 'velocity': velocities,
            'acceleration': accelerations, 'time': times, 'duration': duration,
            'cartesian_waypoints': count, 'continuation_trace': traces,
            'proposal_seconds': time.monotonic() - began,
            'requires_retained_full_path_guard': True, 'policy': VERSION}


def replay(request_path, output):
    from .cartesian_lift import restore_guard
    from resimpair_tabletop.sample_report import native_scalars
    request_path, output = Path(request_path), Path(output)
    if output.exists():
        raise ValueError('Existing replay; retain it and recover instead')
    request = read(request_path)
    if request['stage'] != 'lift' or not request['attached']:
        raise ValueError('saved_attached_lift_required')
    motion = request_path.parent.parent
    execution = motion.parent
    guard = restore_guard(execution, motion, request)
    current = np.asarray(request['current'])
    goal = np.asarray(request['goal'])
    frozen = [binding(__file__), binding(ROOT / 'resimpair_grasp/motion.py')]
    output.mkdir(parents=True)
    write(output / 'protocol.json', {'schema': VERSION, 'request': binding(request_path),
          'code': frozen, 'old_acceptance_unchanged': True, 'physical_execution': False})
    started = time.monotonic()
    path = report = dense = poses = None
    error = None
    try:
        path = cartesian_path(guard.planner, guard.root, current, goal)
        report, dense, poses = guard.check(path, current, goal, 'lift')
    except Exception as exc:
        error = repr(exc)
    for item in frozen:
        resolve(item)
    if path is not None:
        np.savez_compressed(output / 'path.npz', positions=path['position'], velocity=path['velocity'],
                            acceleration=path['acceleration'], current=current, goal=goal)
        write(output / 'continuation_trace.json', path['continuation_trace'])
    if dense is not None:
        np.savez_compressed(output / 'dense.npz', joints=dense, tcp=poses)
    result = {'schema': VERSION, 'request': binding(request_path), 'candidate_guard': report,
              'error': error, 'seconds': time.monotonic() - started,
              'original_measured_state': read(execution / 'execution_bundle.json')['measurements'],
              'physical_success': 'unmeasured', 'physical_executions': 0, 'gpu_jobs': 0, 'api_calls': 0}
    write(output / 'report.json', native_scalars(result))
    return {'passed': bool(report and report['passed']), 'error': error, 'seconds': result['seconds']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(replay(args.request, args.output))
