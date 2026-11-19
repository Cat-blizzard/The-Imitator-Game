"""Scripted expert and designed counterexamples for DualLiftBox-v1 (E29b), any parallel-gripper dual configuration.

run(env, variant) with variant in
  expert             each arm grasps the end on its side (13 cm from the box centre along the long axis, closing axis =
                     the box's short horizontal axis), left first then right (RRT to a pre-grasp, straight descent), both
                     close together, both TCPs move up by the same displacement (synchronised Cartesian paths, 8 cm, or
                     7 / 6 cm when the higher lift is not IK-feasible for both arms), hold still
  single_arm_center  the LEFT arm alone grasps the box at its centre and lifts it by the same height; the right arm stays
                     home (the box is light enough to be lifted level by one gripper)
  one_end_lift       the left arm alone grasps its end and lifts it; the right arm stays home (the box pivots in the
                     grasp with its other end on the table, or, with a strong grasp, is held cantilevered)
  low_lift           as the expert, but the lift is 2 cm
  drop_at_end        as the expert; the policy's last action opens both grippers (the box falls after the policy ends)
  no_op              both arms stay at home
Phases: FREE for transit with an empty gripper and for an idle arm, CONTACT for the final descent, grasp, lift, hold.
"""
from __future__ import annotations

import numpy as np

from ..data.phase import CONTACT, FREE
from .dual_b1_common import Arms, actor_T, box_frames, grasp_z, result
from .scripted import UP, pose

BEHAVIOURS = ("expert", "single_arm_center", "one_end_lift", "low_lift", "drop_at_end", "no_op")
END_OFFSET = 0.13
LIFTS = (0.08, 0.07, 0.06)


def _plan_grasp(A: Arms, i: int, offset: float, lifts) -> tuple | None:
    """(R, g, lift) for arm i grasping the box at `offset` along its long axis (sign = side), first feasible lift"""
    u = A.u
    T = actor_T(u.box)
    c = T[:3, 3]
    a = T[:3, 1].copy(); a[2] = 0; a /= np.linalg.norm(a)
    if a[1] < 0:
        a = -a
    x = T[:3, 0].copy(); x[2] = 0; x /= np.linalg.norm(x)
    hz = float(u.box_half[2])
    p = c + offset * a
    inward = (c - p) if abs(offset) > 1e-6 else (A.bases[i].p - c)
    inward = np.asarray(inward, float); inward[2] = 0
    if np.linalg.norm(inward) < 1e-9:
        inward = np.array([1.0, 0, 0])
    inward /= np.linalg.norm(inward)
    if abs(offset) < 1e-6:          # centre grasp: lean towards the arm's own base
        inward = -inward
    A.obstacles(i)
    for lift in lifts:
        for tilt, R in box_frames(A, i, [x, -x], inward):
            g = np.array([p[0], p[1], grasp_z(A, i, tilt, hz)])
            for hpre in (0.10, 0.07, 0.05):
                if A.ik_chain(i, [g - R[:, 2] * hpre, g, g + UP * lift], R):
                    return R, g, lift, hpre, tilt
    return None


def _approach(A: Arms, i: int, plan, other_phase: str) -> bool:
    R, g, lift, hpre, tilt = plan
    A.set(1 - i, other_phase)
    A.obstacles(i)
    if not A.go(i, pose(g - R[:, 2] * hpre, R), "rrt", FREE, name=f"arm{i} pre"):
        return False
    return A.go(i, pose(g, R), "line", CONTACT, name=f"arm{i} grasp")


def run(env, variant: str = "expert") -> dict:
    if variant not in BEHAVIOURS:
        raise ValueError(variant)
    A = Arms(env)
    u = A.u
    log = {}
    A.wait(1, FREE, FREE)
    if variant == "no_op":
        A.wait(20, FREE, FREE)
        return result(A, True, variant, log)
    lifts = (0.02,) if variant == "low_lift" else LIFTS
    if variant in ("single_arm_center", "one_end_lift"):
        off = 0.0 if variant == "single_arm_center" else END_OFFSET
        plan = _plan_grasp(A, 0, off, lifts)
        if plan is None:
            A.failed.append("arm0: no IK-feasible grasp")
            return result(A, False, variant, log)
        if not _approach(A, 0, plan, FREE):
            return result(A, False, variant, log)
        A.grip(0, 1.0, n=25)
        A.set(1, FREE)
        paths = A.lift_paths(UP * plan[2], arms=(0,))
        if paths is None:
            return result(A, False, variant, log)
        A.phase(CONTACT, FREE)
        A.follow_both(paths, settle=5)
        A.wait(10, CONTACT, FREE)
        log.update(lift=plan[2], tilt0=plan[4])
        return result(A, True, variant, log)
    plans = []
    for lift in lifts:                       # the same lift height must be feasible for both arms
        p0 = _plan_grasp(A, 0, END_OFFSET, (lift,))
        p1 = _plan_grasp(A, 1, -END_OFFSET, (lift,)) if p0 is not None else None
        if p0 is not None and p1 is not None:
            plans = [p0, p1]
            break
    if not plans:
        A.failed.append("no lift height IK-feasible for both arms")
        return result(A, False, variant, log)
    if not _approach(A, 0, plans[0], FREE):
        return result(A, False, variant, log)
    if not _approach(A, 1, plans[1], CONTACT):
        return result(A, False, variant, log)
    A.grip_both(1.0, n=25)
    paths = A.lift_paths(UP * plans[0][2], arms=(0, 1))
    if paths is None:
        return result(A, False, variant, log)
    A.phase(CONTACT, CONTACT)
    A.follow_both(paths, settle=5)
    A.wait(10, CONTACT, CONTACT)
    if variant == "drop_at_end":
        A.phase(CONTACT, CONTACT)
        for i, ag in enumerate(A.ex.agents):
            A.ex.targets[i][ag.record.dof_arm:] = ag.gripper_action(0.0)
        A.ex.step(1)                         # the opening command is the policy's last action
    log.update(lift=plans[0][2], tilt=[plans[0][4], plans[1][4]])
    return result(A, True, variant, log)
