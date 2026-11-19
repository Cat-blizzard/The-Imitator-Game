"""Opt-in local Cartesian approach/lift/retreat with the retained path guard.

Only short, nearly vertical fixed-orientation transfers are proposed here.
Every path still passes MeshMotion.check. The shared controller may use this
guard class without copying its grasp/place loop. No physical retries, contact
parameter changes or new collision exemptions are introduced.
"""
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_grasp.motion import MeshMotion
from data_generation_methods.human_to_robot_pipeline.strict_motion import (
    matrix_to_pose7, sampled_mesh_points, world_fk)
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
from .cartesian_lift import cartesian_path as retained_lift_path, restore_guard
from .stability_motion import StabilityMotion

VERSION = 'cartesian_transfer_v1'
LOCAL_STAGES = frozenset({'grasp_reach', 'lift', 'retreat'})


def local_cartesian_path(planner, root, current, goal, stage, timestep=.05, waypoint_m=None):
    """Return a proposal, not a collision certificate or physical success.

    Lift/retreat reuse the existing upward IK chain. A grasp reach runs the
    same bounded continuation downward, with a <=0.01 rad endpoint correction.
    Descent uses 1 mm steps near the object; upward motion keeps the retained
    4 mm steps. Both retain the same joint-jump and complete-path checks.
    IK uses the preceding solution as seed and rejects >0.08 rad branch jumps.
    """
    if stage not in LOCAL_STAGES:
        raise ValueError('unsupported_local_cartesian_stage')
    if waypoint_m is None:
        waypoint_m = .001 if stage == 'grasp_reach' else .004
    current = np.asarray(current, dtype=float); goal = np.asarray(goal, dtype=float)
    if current.shape != (9,) or goal.shape != (4, 4) or not np.isfinite(current).all() or not np.isfinite(goal).all():
        raise ValueError('invalid_cartesian_input_shape_or_value')
    if stage in {'lift', 'retreat'}:
        result = retained_lift_path(planner, root, current, goal, timestep, waypoint_m)
        return {**result, 'cartesian_transfer_policy': VERSION, 'cartesian_stage': stage}
    start = world_fk(current[None, :7], root)[0]
    delta = goal[:3, 3] - start[:3, 3]
    if np.linalg.norm(delta[:2]) > .001 or not -.30 <= delta[2] <= -.002:
        raise ValueError('bounded_vertical_grasp_reach_only')
    correction = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).as_rotvec()
    if np.linalg.norm(correction) > .01:
        raise ValueError('grasp_reach_orientation_must_remain_fixed')
    if not .001 <= waypoint_m <= .01 or not .01 <= timestep <= .10:
        raise ValueError('invalid_cartesian_resolution')
    count = int(np.ceil(-delta[2] / waypoint_m))
    values = [current[:7].copy()]; q = current.copy(); inverse = np.linalg.inv(root)
    for i in range(1, count + 1):
        fraction = i / count
        pose = start.copy(); pose[:3, 3] = start[:3, 3] + delta * fraction
        pose[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(correction * fraction).as_matrix()
        status, solutions = planner.IK(matrix_to_pose7(inverse @ pose), q.copy(),
            mask=[False] * 7 + [True, True], n_init_qpos=2, threshold=.0002)
        if status != 'Success' or not len(solutions):
            raise RuntimeError(f'cartesian_waypoint_ik_failed:{i}:{status}')
        candidate = min(solutions, key=lambda value: np.linalg.norm(value[:7] - q[:7]))
        if np.max(np.abs(candidate[:7] - q[:7])) > .08:
            raise RuntimeError(f'cartesian_waypoint_branch_jump:{i}')
        q = np.asarray(candidate).copy(); values.append(q[:7].copy())
    times, positions, velocities, accelerations, duration = planner.TOPP(np.asarray(values), timestep)
    return {'status': 'Success', 'position': positions, 'velocity': velocities,
            'acceleration': accelerations, 'time': times, 'duration': duration,
            'cartesian_waypoints': count, 'requires_retained_full_path_guard': True,
            'cartesian_transfer_policy': VERSION, 'cartesian_stage': stage}


class CartesianTransferMotion(StabilityMotion):
    """Drop-in guard class for the one shared cached grasp/place controller."""

    def plan(self, pose, stage, current=None):
        if stage not in LOCAL_STAGES:
            return super().plan(pose, stage, current)
        current = array(self.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
        if self.use_attach:
            self.refresh_attachment(stage)
        goal = np.asarray(pose.to_transformation_matrix())
        self.selected_world_pose = array(self.selected_actor.pose.to_transformation_matrix()).reshape(4, 4)
        directory = self.directory / f'{self.counter:03d}_{stage}_cartesian'
        self.counter += 1; directory.mkdir()
        write(directory / 'request.json', {'stage': stage, 'current': current.tolist(), 'goal': goal.tolist(),
              'frame': len(self.rec.rows), 'attached': self.use_attach, 'policy': VERSION})
        try:
            with self.rec.performance.section('motion_plan_cartesian'):
                path = local_cartesian_path(self.planner, self.root, current, goal, stage,
                                            timestep=self.rec.control_timestep)
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
        # Only planning proposals are retried; fallback runs the retained guard
        # as well. Attachment was refreshed once above, not once per solver.
        return MeshMotion.plan(self, pose, stage, current)


def restore_unattached_guard(execution, motion_directory, request):
    """Restore recorded native geometry and the measured target pose on CPU."""
    import mplib
    import trimesh
    from mplib.pymp.collision_detection import fcl
    execution, motion_directory = Path(execution), Path(motion_directory)
    if request['attached']:
        raise ValueError('unattached_replay_requires_unattached_request')
    bundle = read(execution / 'execution_bundle.json'); scene = bundle['scene']
    operation = next(o for o in bundle['task_intent']['operations']
                     if o['id'] == motion_directory.name.removesuffix('_motion'))
    guard = MeshMotion.__new__(MeshMotion); guard.arm = operation['arm']
    guard.root = np.eye(4); guard.root[:3, 3] = scene['robot_roots'][guard.arm]
    if hasattr(mplib, 'set_global_seed'):
        mplib.set_global_seed(20261027)
    urdf = ROOT / 'PolicyGen/mani_skill/assets/robots/panda/panda_v3.urdf'
    guard.planner = mplib.Planner(urdf=str(urdf), srdf=str(urdf.with_suffix('.srdf')),
        move_group='panda_hand_tcp', joint_vel_limits=np.ones(7)*.8, joint_acc_limits=np.ones(7)*.8)
    guard.planner.set_base_pose(matrix_to_pose7(guard.root))
    guard.objects = []; guard.environment_collision_objects = []; guard.physical_robot_objects = []
    def collider(path):
        geometry = fcl.load_mesh_as_BVH(str(path), np.ones(3)); obj = fcl.CollisionObject(geometry)
        guard.objects.append((geometry, obj)); return obj
    for obj in read(motion_directory / 'world.json')['world']:
        collision = collider(motion_directory / (obj['id'] + '.obj'))
        guard.planner.set_normal_object(obj['id'], collision)
        guard.environment_collision_objects.append((obj['id'], collision))
    guard.table_collision = collider(motion_directory / 'complete_table.obj')
    for path in sorted(motion_directory.glob('robot_*.obj')):
        name = path.stem.removeprefix('robot_')
        guard.physical_robot_objects.append((name, guard.planner.link_name_2_idx[name], collider(path)))
    table = trimesh.load(motion_directory / 'complete_table.obj', force='mesh')
    base = trimesh.load(motion_directory / 'robot_panda_link0.obj', force='mesh'); base.apply_transform(guard.root)
    cloud = sampled_mesh_points(table, 18000, 401)
    inside = np.all((cloud[:, :2] >= base.bounds[0, :2]-.005) & (cloud[:, :2] <= base.bounds[1, :2]+.005), axis=1)
    guard.planner.update_point_cloud(cloud[~inside], radius=.001)
    guard.selected_collision = collider(motion_directory / 'selected_physical.obj')
    guard.payload_collision = None; guard.payload_to_tcp = None; guard.use_attach = False
    guard.support_floor_z = scene['table_top_z']
    with np.load(resolve(bundle['measurements']), allow_pickle=False) as data:
        state = data['object_states'][request['frame'], bundle['object_ids'].index(operation['object'])]
    guard.selected_world_pose = np.eye(4)
    guard.selected_world_pose[:3, :3] = Rotation.from_quat(state[3:7], scalar_first=True).as_matrix()
    guard.selected_world_pose[:3, 3] = state[:3]
    return guard


def replay(execution, request_path, output):
    """Recheck a saved path and the new local proposal; never run physics."""
    from resimpair_tabletop.sample_report import native_scalars
    execution, request_path, output = map(Path, [execution, request_path, output])
    if output.exists():
        raise ValueError('Preserve previous Cartesian replay')
    request = read(request_path); stage = request['stage']
    if stage not in LOCAL_STAGES:
        raise ValueError('unsupported_local_cartesian_stage')
    motion_directory = request_path.parent.parent
    guard = (restore_guard if request['attached'] else restore_unattached_guard)(execution, motion_directory, request)
    current, goal = np.array(request['current']), np.array(request['goal'])
    previous = None; prior = request_path.parent / 'path.npz'
    if prior.exists():
        with np.load(prior, allow_pickle=False) as data:
            previous, _, _ = guard.check({'position': data['positions']}, current, goal, stage)
    started = time.monotonic(); error = None; path = report = dense = poses = None
    try:
        path = local_cartesian_path(guard.planner, guard.root, current, goal, stage)
        report, dense, poses = guard.check(path, current, goal, stage)
    except Exception as exc:
        error = repr(exc)
    result = {'schema': 'cartesian_transfer_offline_replay_v1', 'policy': VERSION,
              'request': binding(request_path), 'measurements': read(execution / 'execution_bundle.json')['measurements'],
              'prior_path': binding(prior) if prior.exists() else None, 'original_guard_reproduced': previous,
              'candidate_guard': report, 'error': error, 'planning_and_guard_seconds': time.monotonic()-started,
              'retained_guard': binding(ROOT / 'resimpair_grasp/motion.py'),
              'retained_lift_solver': binding(ROOT / 'resimpair_pickplace/cartesian_lift.py'),
              'implementation': binding(__file__), 'source_meshes': [binding(p) for p in sorted(motion_directory.glob('*.obj'))],
              'physical_executions': 0, 'gpu_executions': 0, 'api_calls': 0,
              'physical_success': 'unmeasured', 'live_controller_promotion': False}
    output.mkdir(parents=True)
    if path is not None:
        np.savez_compressed(output / 'candidate_path.npz', positions=path['position'], velocity=path['velocity'],
                            acceleration=path['acceleration'], current=current, goal=goal)
    if dense is not None:
        np.savez_compressed(output / 'dense.npz', joints=dense, tcp=poses)
    write(output / 'report.json', native_scalars(result))
    return {'passed': bool(report and report['passed']), 'error': error,
            'seconds': result['planning_and_guard_seconds']}
