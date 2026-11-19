"""Candidate Cartesian placement with a payload-centre planning corridor.

Reuses the existing bounded SE(3) solver and exact native path guard. The
additional 12 mm planning margin is stricter than the unchanged 15 mm task
event requirement. It is not an evaluator, physical attachment or grasp retry.
"""
from contextlib import contextmanager
import numpy as np

from resimpair_grasp.common import binding, write
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
from data_generation_methods.human_to_robot_pipeline.strict_motion import densify_joints, world_fk
from . import whole_path_motion_v1 as retained

VERSION = 'payload_centre_cartesian_descent_v1'
MAX_PAYLOAD_XY_M = .012


def payload_corridor(poses, object_in_tcp, goal):
    poses, local, goal = (np.asarray(v, float) for v in (poses, object_in_tcp, goal))
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError('nonempty_dense_TCP_transforms_required')
    if local.shape != (4, 4) or goal.shape != (4, 4) or not all(np.isfinite(v).all() for v in (poses, local, goal)):
        raise ValueError('finite_bound_payload_and_goal_required')
    predicted = poses @ local
    target = (goal @ local)[:3, 3]
    errors = np.linalg.norm(predicted[:, :2, 3] - target[:2], axis=1)
    return {'passed': bool(np.max(errors) <= MAX_PAYLOAD_XY_M),
            'maximum_predicted_payload_XY_error_m': float(np.max(errors)),
            'planning_limit_m': MAX_PAYLOAD_XY_M, 'samples': len(poses),
            'target_payload_position_m': target.tolist(),
            'scope': 'rigid measured-attachment prediction, not actual future contact stability'}


@contextmanager
def install():
    original = retained.WholePathMotion.plan

    def plan(guard, pose, stage, current=None):
        if stage != 'release_above_rim' or not guard.use_attach:
            return original(guard, pose, stage, current)
        current = array(guard.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
        goal = np.asarray(pose.to_transformation_matrix(), float)
        if not getattr(guard, 'preview', False):
            guard.refresh_attachment(stage)
        local = np.asarray(guard.payload_to_tcp, float).copy()
        directory = guard.directory / f'{guard.counter:03d}_{stage}_payload_cartesian'
        guard.counter += 1
        directory.mkdir()
        write(directory / 'request.json', {'policy': VERSION, 'implementation': binding(__file__),
            'stage': stage, 'current': current.tolist(), 'goal': goal.tolist(),
            'payload_to_tcp': local.tolist(), 'frame': len(guard.rec.rows),
            'attached': True, 'preview': bool(getattr(guard, 'preview', False)),
            'retained_solver': binding(retained.__file__), 'physical_retries': 0})
        try:
            with guard.rec.performance.section('motion_plan_payload_cartesian'):
                path = retained.geometric_transport(guard.planner, guard.root, current, goal,
                                                     guard.rec.control_timestep)
            with guard.rec.performance.section('dense_physical_collision_guard'):
                report, dense, poses = guard.check(path, current, goal, stage)
            corridor = payload_corridor(poses, local, goal) if poses is not None else None
            accepted = report['passed'] and corridor is not None and corridor['passed']
            np.savez_compressed(directory / 'path.npz', positions=path['position'], current=current, goal=goal)
            if dense is not None:
                np.savez_compressed(directory / 'dense.npz', joints=dense, tcp=poses)
            write(directory / 'decision.json', {'accepted': bool(accepted), 'report': report,
                'payload_corridor': corridor, 'policy': VERSION})
            if accepted:
                path['guard_evidence'] = str(directory / 'decision.json')
                return path
        except Exception as error:
            if not (directory / 'decision.json').exists():
                write(directory / 'decision.json', {'accepted': False, 'error': repr(error), 'policy': VERSION})
        # Preserve the original geometry alternatives, but do not execute their
        # bowed payload paths merely because their much wider TCP check passed.
        fallback = original(guard, pose, stage, current)
        dense = densify_joints(np.asarray(fallback['position']))
        corridor = payload_corridor(world_fk(dense, guard.root), guard.payload_to_tcp, goal)
        write(directory / 'retained_fallback_corridor.json', {
            'retained_guard': binding(fallback['guard_evidence']), 'payload_corridor': corridor})
        if not corridor['passed']:
            raise RuntimeError('placement_payload_corridor_unavailable')
        return fallback

    retained.WholePathMotion.plan = plan
    try:
        yield
    finally:
        retained.WholePathMotion.plan = original
