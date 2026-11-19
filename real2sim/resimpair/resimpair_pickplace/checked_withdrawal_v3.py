"""Compose the existing geometry-driven release exit with whole-path planning.

The original checked-retreat skill was mutually exclusive with the whole-path
motion policy. This adapter reuses its aperture search before opening, and
adds its stricter released-target finger check to the current mesh guard.
No body is moved, no drive/contact parameters or acceptance limits change.
"""
from contextlib import contextmanager

import numpy as np

from .above_rim_release_v1 import policy
from .release_clearance import prepare
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array

VERSION = 'composable_native_checked_withdrawal_v3'


@contextmanager
def opening_phase(guard):
    """Opening before release retains contact; released withdrawal stays strict.

    The retained opening sampler labels its check stage ``retreat``. Do not
    confuse that geometric query name with physical release. Preserve all
    environment checks and restore the stricter guard even on an exception.
    """
    previous_stage = guard.check_stage
    previous_flag = getattr(guard, 'require_released_target_clearance', False)
    before_release = guard.rec.stage.endswith(':release_settle')
    try:
        if before_release:
            guard.require_released_target_clearance = False
        yield
    finally:
        guard.check_stage = previous_stage
        guard.require_released_target_clearance = previous_flag


@contextmanager
def install():
    from . import stability_motion as execution
    from resimpair_grasp.motion import MeshMotion
    from .stability_motion import StabilityMotion
    original_make = execution.make_solver
    original_width = execution.opening_width
    original_collisions = MeshMotion.independent_table_collisions
    original_sweep = StabilityMotion.opening_sweep
    active = {}

    def solver(*args, **kwargs):
        guard = original_make(*args, **kwargs)
        active['guard'] = guard
        return guard

    def opening_width(measured):
        legacy = original_width(measured)
        guard = active.get('guard')
        if guard is None or getattr(guard, 'preview', False):
            raise ValueError('checked_withdrawal_requires_live_guard')
        if not guard.rec.stage.endswith(':release_settle'):
            raise ValueError('checked_withdrawal_only_after_measured_release_readiness')
        op_id = guard.rec.stage.split(':')[0]
        operation = guard.rec.drop_operations[op_id]
        if policy(operation) is not None:
            raise ValueError('this_candidate_only_qualifies_supported_release')
        current = array(guard.agent.robot.qpos).reshape(9).copy()
        retreat = array(guard.agent.tcp.pose.to_transformation_matrix()).reshape(4, 4).copy()
        retreat[2, 3] += .18
        width, evidence = prepare(guard, current, legacy, retreat)
        return width

    def opening_sweep(guard, width):
        with opening_phase(guard):
            return original_sweep(guard, width)

    def collisions(guard, full, tcp):
        pairs = original_collisions(guard, full, tcp)
        if (guard.check_stage == 'retreat'
                and getattr(guard, 'require_released_target_clearance', False)):
            from mplib.pymp.collision_detection import fcl
            for name, _, collision in guard.physical_robot_objects:
                if name in {'panda_leftfinger', 'panda_rightfinger'}:
                    if fcl.collide(collision, guard.selected_collision, fcl.CollisionRequest()).is_collision():
                        pairs.append({'link1': name, 'link2': 'released_target'})
        return pairs

    execution.make_solver = solver
    execution.opening_width = opening_width
    MeshMotion.independent_table_collisions = collisions
    StabilityMotion.opening_sweep = opening_sweep
    try:
        yield
    finally:
        execution.make_solver = original_make
        execution.opening_width = original_width
        MeshMotion.independent_table_collisions = original_collisions
        StabilityMotion.opening_sweep = original_sweep
