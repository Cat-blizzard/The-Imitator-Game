"""Refine only a live placement path that violates the payload margin.

Frozen v1 changed every preview and descent, and regressed a later grasp.
This revision returns the original path unchanged when it satisfies the
planning corridor. Preview, grasp choice, closure and physical rules stay as
implemented by the retained worker; no case IDs or asset names are used.
"""
from contextlib import contextmanager
import numpy as np

from resimpair_grasp.common import binding, write
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
from data_generation_methods.human_to_robot_pipeline.strict_motion import densify_joints, world_fk
from . import whole_path_motion_v1 as retained
from .payload_descent_v1 import payload_corridor

VERSION = 'conditional_payload_centre_descent_v2'


@contextmanager
def install():
    original = retained.WholePathMotion.plan

    def plan(guard, pose, stage, current=None):
        # The unmodified planner still owns source selection, native collision
        # checks and all failures. This candidate cannot bypass a collision.
        path = original(guard, pose, stage, current)
        if stage != 'release_above_rim' or not guard.use_attach or getattr(guard, 'preview', False):
            return path
        current = array(guard.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
        goal = np.asarray(pose.to_transformation_matrix(), float)
        local = np.asarray(guard.payload_to_tcp, float).copy()
        dense = densify_joints(np.asarray(path['position']))
        prior = payload_corridor(world_fk(dense, guard.root), local, goal)
        directory = guard.directory / f'{guard.counter:03d}_{stage}_conditional_payload'
        guard.counter += 1
        directory.mkdir()
        write(directory / 'trigger.json', {'policy': VERSION, 'implementation': binding(__file__),
            'retained_guard': binding(path['guard_evidence']), 'payload_corridor': prior,
            'refinement_required': not prior['passed'], 'frame': len(guard.rec.rows),
            'preview_mutated': False, 'original_path_returned_unchanged': prior['passed']})
        if prior['passed']:
            return path
        write(directory / 'request.json', {'stage': stage, 'current': current.tolist(),
            'goal': goal.tolist(), 'payload_to_tcp': local.tolist(), 'attached': True,
            'frame': len(guard.rec.rows), 'preview': False})
        try:
            with guard.rec.performance.section('motion_plan_conditional_payload_descent'):
                candidate = retained.geometric_transport(guard.planner, guard.root, current, goal,
                                                         guard.rec.control_timestep)
            with guard.rec.performance.section('dense_physical_collision_guard'):
                report, dense, poses = guard.check(candidate, current, goal, stage)
            corridor = payload_corridor(poses, local, goal) if poses is not None else None
            accepted = report['passed'] and corridor is not None and corridor['passed']
            np.savez_compressed(directory / 'path.npz', positions=candidate['position'], current=current, goal=goal)
            if dense is not None:
                np.savez_compressed(directory / 'dense.npz', joints=dense, tcp=poses)
            write(directory / 'decision.json', {'accepted': bool(accepted), 'report': report,
                'payload_corridor': corridor, 'policy': VERSION})
            if accepted:
                candidate['guard_evidence'] = str(directory / 'decision.json')
                return candidate
            raise RuntimeError('conditional_payload_descent_not_qualified')
        except Exception as error:
            if not (directory / 'decision.json').exists():
                write(directory / 'decision.json', {'accepted': False, 'error': repr(error), 'policy': VERSION})
            raise

    retained.WholePathMotion.plan = plan
    try:
        yield
    finally:
        retained.WholePathMotion.plan = original
