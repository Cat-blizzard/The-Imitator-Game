"""One bounded SE(3) proposal after retained above-bowl paths are rejected.

Successful CartesianTransferMotion calls return unchanged, with no extra
solver calls. Only an attached above_bowl stage whose screw AND RRT paths
were rejected by the retained full-path guard can trigger this candidate.
No collision exemption, workspace/endpoint threshold, contact parameter or
physical retry is introduced. A straight SE(3) path can still be infeasible.
"""
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, read, write
from resimpair_grasp.motion import MeshMotion
from data_generation_methods.human_to_robot_pipeline.strict_motion import matrix_to_pose7, world_fk
from .cartesian_transfer import CartesianTransferMotion, restore_unattached_guard

VERSION = 'transport_fallback_v2'
WAYPOINT_M = .003
WAYPOINT_RAD = .015
MAX_WAYPOINTS = 512
MAX_TRANSLATION_M = 1.2
JOINT_JUMP_RAD = .08


def _pose(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0., 0., 0., 1.], atol=1e-8)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(value[:3, :3]) < .999):
        raise ValueError('invalid_rigid_transport_pose')
    return value


def se3_transport_path(planner, root, current, goal, timestep=.05):
    """Linear XYZ plus shortest SO(3) interpolation, continuation IK and TOPP.

    One path, at most 512 IK waypoints, <=3 mm translation and <=0.015 rad
    rotation per requested waypoint. Native IK/limits and the retained 0.08 rad
    continuation jump bound apply. The returned path requires MeshMotion.check.
    """
    current = np.asarray(current, dtype=float); root = _pose(root); goal = _pose(goal)
    if current.shape != (9,) or not np.isfinite(current).all():
        raise ValueError('invalid_transport_joint_state')
    if not .01 <= timestep <= .10:
        raise ValueError('invalid_transport_timestep')
    start = world_fk(current[None, :7], root)[0]
    delta = goal[:3, 3] - start[:3, 3]
    distance = float(np.linalg.norm(delta))
    rotation = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).as_rotvec()
    angle = float(np.linalg.norm(rotation))
    if distance > MAX_TRANSLATION_M or (distance < .0005 and angle < .002):
        raise ValueError('transport_displacement_outside_finite_scope')
    count = int(max(1, np.ceil(distance / WAYPOINT_M), np.ceil(angle / WAYPOINT_RAD)))
    if count > MAX_WAYPOINTS:
        raise ValueError('transport_waypoint_budget_exceeded')
    values = [current[:7].copy()]; q = current.copy(); inverse = np.linalg.inv(root)
    for i in range(1, count + 1):
        fraction = i / count
        pose = start.copy(); pose[:3, 3] = start[:3, 3] + delta * fraction
        pose[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(rotation * fraction).as_matrix()
        status, solutions = planner.IK(matrix_to_pose7(inverse @ pose), q.copy(),
            mask=[False] * 7 + [True, True], n_init_qpos=2, threshold=.0002)
        if status != 'Success' or not len(solutions):
            raise RuntimeError(f'transport_waypoint_ik_failed:{i}:{status}')
        candidate = min(solutions, key=lambda value: np.linalg.norm(value[:7] - q[:7]))
        if np.max(np.abs(candidate[:7] - q[:7])) > JOINT_JUMP_RAD:
            raise RuntimeError(f'transport_waypoint_branch_jump:{i}')
        q = np.asarray(candidate).copy(); values.append(q[:7].copy())
    times, positions, velocities, accelerations, duration = planner.TOPP(np.asarray(values), timestep)
    return {'status': 'Success', 'position': positions, 'velocity': velocities,
            'acceleration': accelerations, 'time': times, 'duration': duration,
            'cartesian_waypoints': count, 'translation_m': distance, 'rotation_rad': angle,
            'requires_retained_full_path_guard': True, 'transport_policy': VERSION}


def _retained_rejections(directory, first_counter):
    """Require actual guard rejections, not solver errors or missing evidence."""
    paths = [Path(directory) / f'{first_counter+i:03d}_above_bowl_{mode}' for i, mode in enumerate(('screw', 'rrt'))]
    requests = []
    try:
        for path in paths:
            decision = read(path / 'decision.json')
            if decision.get('accepted') is not False or decision.get('report', {}).get('passed') is not False:
                return None
            requests.append(read(path / 'request.json'))
        if requests[0] != requests[1] or requests[0].get('stage') != 'above_bowl' or requests[0].get('attached') is not True:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return requests[0], [{'request': binding(p / 'request.json'), 'decision': binding(p / 'decision.json')} for p in paths]


class TransportFallbackMotion(CartesianTransferMotion):
    """Opt-in drop-in guard class; successful retained branches are untouched."""

    def plan(self, pose, stage, current=None):
        if stage != 'above_bowl':
            return super().plan(pose, stage, current)
        first_counter = self.counter
        try:
            return super().plan(pose, stage, current)
        except RuntimeError as baseline_error:
            if (str(baseline_error) != 'independent_guard_rejected:above_bowl'
                    or not self.use_attach or self.counter != first_counter + 2):
                raise
            retained = _retained_rejections(self.directory, first_counter)
            if retained is None:
                raise
            request, prior = retained
            current, goal = np.array(request['current']), np.array(request['goal'])
            directory = self.directory / f'{self.counter:03d}_above_bowl_se3'
            self.counter += 1; directory.mkdir()
            write(directory / 'request.json', {**request, 'policy': VERSION, 'retained_rejections': prior})
            # super() refreshed the measured payload once. Planning changed no
            # physical state; do not refresh, move or retry the physical grasp.
            try:
                with self.rec.performance.section('motion_plan_transport_fallback'):
                    path = se3_transport_path(self.planner, self.root, current, goal, self.rec.control_timestep)
                with self.rec.performance.section('dense_physical_collision_guard'):
                    report, dense, poses = self.check(path, current, goal, stage)
                np.savez_compressed(directory / 'path.npz', positions=path['position'], current=current, goal=goal)
                if dense is not None:
                    np.savez_compressed(directory / 'dense.npz', joints=dense, tcp=poses)
                write(directory / 'decision.json', {'accepted': report['passed'], 'report': report, 'policy': VERSION})
                if report['passed']:
                    path['guard_evidence'] = str(directory / 'decision.json')
                    return path
            except Exception as error:
                if not (directory / 'decision.json').exists():
                    write(directory / 'decision.json', {'accepted': False, 'error': repr(error), 'policy': VERSION})
            raise baseline_error


def restore_transport_guard(execution, motion_directory, request):
    """Reuse exact native geometry restore, then bind THIS stage's attachment.

    The older lift replay helper only reads attachment_*_lift.json. Here the
    shared unattached geometry loader is used without modifying source files;
    its held state is then set to the recorded above_bowl frame transform.
    """
    from mplib.pymp.collision_detection import fcl
    if request.get('stage') != 'above_bowl' or request.get('attached') is not True:
        raise ValueError('attached_above_bowl_request_required')
    motion_directory = Path(motion_directory)
    attachments = [(p, read(p)) for p in motion_directory.glob('attachment_*_above_bowl.json')]
    matching = [(p, d) for p, d in attachments if d['frame'] == request['frame']]
    if len(matching) != 1:
        raise ValueError('missing_unambiguous_above_bowl_payload_transform')
    guard = restore_unattached_guard(execution, motion_directory, {**request, 'attached': False})
    guard.payload_to_tcp = _pose(matching[0][1]['measured_object_in_tcp']).copy()
    geometry = fcl.load_mesh_as_BVH(str(motion_directory / 'selected_physical.obj'), np.ones(3))
    guard.payload_collision = fcl.CollisionObject(geometry)
    guard.objects.append((geometry, guard.payload_collision)); guard.use_attach = True
    return guard, matching[0][0]


def replay(execution, request_path, output):
    """Recheck BOTH old paths and one new proposal from identical saved state."""
    from resimpair_tabletop.sample_report import native_scalars
    execution, request_path, output = map(Path, (execution, request_path, output))
    if output.exists(): raise ValueError('Preserve previous transport replay')
    request = read(request_path); motion_directory = request_path.parent.parent
    first = int(request_path.parent.name.split('_')[0]) - (1 if request_path.parent.name.endswith('_rrt') else 0)
    retained = _retained_rejections(motion_directory, first)
    if retained is None or retained[0] != request:
        raise ValueError('both_retained_full_guard_rejections_required')
    guard, attachment = restore_transport_guard(execution, motion_directory, request)
    current, goal = np.array(request['current']), np.array(request['goal'])
    original = []
    for i, mode in enumerate(('screw', 'rrt')):
        folder = motion_directory / f'{first+i:03d}_above_bowl_{mode}'
        with np.load(folder / 'path.npz', allow_pickle=False) as arrays:
            report, _, poses = guard.check({'position': arrays['positions']}, current, goal, 'above_bowl')
        original.append({'mode': mode, 'path': binding(folder / 'path.npz'), 'report': report,
                         'tcp_min_xyz': poses[:, :3, 3].min(axis=0).tolist() if poses is not None else None,
                         'tcp_max_xyz': poses[:, :3, 3].max(axis=0).tolist() if poses is not None else None})
    if any(r['report']['passed'] for r in original):
        raise ValueError('original_guard_rejection_not_reproduced')
    started = time.monotonic(); path = report = dense = poses = None; error = None
    try:
        path = se3_transport_path(guard.planner, guard.root, current, goal)
        report, dense, poses = guard.check(path, current, goal, 'above_bowl')
    except Exception as exc:
        error = repr(exc)
    result = {'schema': 'transport_fallback_offline_replay_v2', 'policy': VERSION,
        'request': binding(request_path), 'attachment': binding(attachment),
        'root_transform': guard.root.tolist(), 'payload_to_tcp': guard.payload_to_tcp.tolist(),
        'world': binding(motion_directory / 'world.json'),
        'source_meshes': [binding(p) for p in sorted(motion_directory.glob('*.obj'))],
        'original_rejections': original, 'candidate_guard': report, 'error': error,
        'planning_and_guard_seconds': time.monotonic() - started,
        'waypoints': path['cartesian_waypoints'] if path is not None else None,
        'implementation': binding(__file__), 'retained_guard': binding(ROOT / 'resimpair_grasp/motion.py'),
        'physical_executions': 0, 'gpu_executions': 0, 'api_calls': 0,
        'physical_success': 'unmeasured', 'live_controller_promotion': False}
    output.mkdir(parents=True)
    if path is not None:
        np.savez_compressed(output / 'candidate_path.npz', positions=path['position'], velocity=path['velocity'],
                            acceleration=path['acceleration'], current=current, goal=goal)
    if dense is not None:
        np.savez_compressed(output / 'dense.npz', joints=dense, tcp=poses)
    write(output / 'report.json', native_scalars(result))
    return {'passed': bool(report and report['passed']), 'error': error, 'seconds': result['planning_and_guard_seconds']}
