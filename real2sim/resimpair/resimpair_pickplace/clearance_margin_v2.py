"""Generate paths with explicit inter-arm clearance; old acceptance is unchanged.

The v1 regression had a collision-free planned retreat but measured joint
tracking differed by 0.000579 rad and link6 touched the inactive arm's mesh.
Three millimetres is a proposal margin, not a relaxed collision tolerance or a
formal bound on physical tracking error. All physical failures remain failures.
"""
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from resimpair_grasp.common import binding
from data_generation_methods.human_to_robot_pipeline.strict_motion import matrix_to_pose7, pose7_to_matrix
from . import whole_path_motion_v1 as previous

VERSION = 'whole_path_interarm_margin_v2'
MINIMUM_INTERARM_DISTANCE_M = .003


class ClearanceMotion(previous.WholePathMotion):
    def check(self, path, current, goal, stage):
        from mplib.pymp.collision_detection import fcl
        report, dense, poses = super().check(path, current, goal, stage)
        if not report['passed']:
            return report, dense, poses
        obstacles = [o for name, o in self.environment_collision_objects if name == 'inactive_robot']
        if len(obstacles) != 1:
            raise ValueError('one_exact_inactive_arm_required_for_margin')
        model = self.planner.pinocchio_model
        minimum = float('inf'); closest = None; checked = 0; passed = True
        try:
            for sample, joints in enumerate(dense):
                full = current.copy(); full[:7] = joints
                model.compute_forward_kinematics(full)
                for name, index, collision in self.physical_robot_objects:
                    collision.set_transformation(matrix_to_pose7(self.root @ pose7_to_matrix(model.get_link_pose(index))))
                    distance = float(fcl.distance(collision, obstacles[0], fcl.DistanceRequest()).min_distance)
                    checked += 1
                    if not np.isfinite(distance):
                        raise ValueError('invalid_native_mesh_distance')
                    if distance < minimum:
                        minimum = distance; closest = {'dense_index': sample, 'link': name}
                    if distance < MINIMUM_INTERARM_DISTANCE_M:
                        passed = False; break
                if not passed: break
        finally:
            self.planner.robot.set_qpos(current.copy(), True)
        return {**report, 'passed': bool(passed), 'original_full_guard_passed': True,
                'proposal_clearance': {'passed': passed, 'minimum_required_m': MINIMUM_INTERARM_DISTANCE_M,
                    'minimum_checked_m': minimum, 'closest_checked': closest, 'distance_queries': checked,
                    'all_dense_distances_checked': passed, 'implementation': binding(__file__),
                    'scope': 'exact physical robot meshes versus inactive arm; a stricter generator proposal filter'}}, dense, poses


@contextmanager
def install(root):
    from .runner import verify
    spec = verify(Path(root))
    if (spec.get('candidate_motion_override') != VERSION
            or spec['code'].get('resimpair_pickplace/clearance_margin_v2.py') != binding(__file__)):
        raise ValueError('explicit_frozen_clearance_candidate_required')
    old_class, old_version = previous.WholePathMotion, previous.VERSION
    previous.WholePathMotion, previous.VERSION = ClearanceMotion, VERSION
    try:
        yield
    finally:
        previous.WholePathMotion, previous.VERSION = old_class, old_version
