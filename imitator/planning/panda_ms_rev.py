"""A2 expert revisions (structural fixes, logged in research/results/a2_expert_tuning/STATUS.md). Written here while A2
runs were still using panda_ms.py; R1 was validated as stack_cube_v12 (research/scripts/a2_rev_eval.py) and then merged:
panda_ms.stack_cube (v1.2) calls pick_verified with the same arguments.

StackCube-v1 v1.2 (A2 revision R1): the frozen expert (+ A2 parameters) with a VERIFIED grasp. After closing on cube A
the gripper must hold A (agent.is_grasping); on a miss it opens, rises to the pre-grasp height, waits still (CONTACT),
re-measures A and grasps again, at most `retries` times; if every attempt misses the expert reports
'plan_failed: grasp' instead of continuing with an empty gripper (the frozen expert reports 'ok' after a missed grasp).
Diagnosis behind it (research/logs/a2/diag_stack_grasp.log): at learner-scale noise 5 of 10 episodes of the best tuned
StackCube candidate closed beside cube A and carried on with an empty gripper.
"""
from __future__ import annotations

import numpy as np
import sapien

from .panda_ms import (CONTACT, FINGER_LENGTH, FREE, PandaExec, cage_open, carry_offset, no_op, top_grasp,
                       yaw_angles)


def pick_verified(ex: PandaExec, actor, lift: float = 0.10, angles=None, grip: float = -1, reach_dz: float = 0.05,
                  depth: float = FINGER_LENGTH, pre_wait: int = 0, close_t: int = 6, retries: int = 2,
                  retry_wait: int = 10) -> tuple[sapien.Pose | None, str]:
    """panda_ms.pick with a grasp check and up to `retries` re-grasps; returns (lift pose or None, reason)"""
    for attempt in range(retries + 1):
        g = top_grasp(ex, actor, angles=angles or yaw_angles(), depth=depth)
        if not ex.move(g * sapien.Pose([0, 0, -reach_dz]), FREE if attempt == 0 else CONTACT,
                       name="reach" if attempt == 0 else f"regrasp_reach{attempt}"):
            return None, "reach"
        w = pre_wait if attempt == 0 else max(pre_wait, retry_wait)
        if w:
            ex.wait(w, CONTACT)
        if not ex.move(g, CONTACT, name="grasp" if attempt == 0 else f"regrasp{attempt}"):
            return None, "grasp_move"
        ex.grip(grip, t=close_t)
        if bool(ex.u.agent.is_grasping(actor)[0]):
            lp = sapien.Pose([0, 0, lift]) * g
            if not ex.move(lp, CONTACT, name="lift"):
                return None, "lift"
            return lp, "ok"
        ex.grip(1, t=close_t)                                    # missed: open and rise to the pre-grasp height
        t = ex.tcp()
        if not ex.move(sapien.Pose(t.p + np.array([0, 0, reach_dz]), t.q), CONTACT, name=f"regrasp_up{attempt + 1}"):
            return None, "regrasp_up"
    return None, "grasp"


def stack_cube_v12(env, variant: str = "expert", params: dict | None = None, retries: int = 2,
                   retry_wait: int = 10) -> dict:
    """panda_ms.stack_cube (frozen v1 + A2 parameters) with pick_verified (A2 revision R1); variants as stack_cube"""
    if variant == "no_op":
        return no_op(env)
    from . import params as PR
    p = PR.pv(params, "StackCube-v1", env, variant)
    ex = PandaExec(env, vel=p["vel"], acc=p["vel"])
    u = ex.u
    h = float(u.cube_half_size[2])
    lp, why = pick_verified(ex, u.cubeA, lift=p["lift_dz"], reach_dz=p["reach_dz"], depth=p["grasp_depth"],
                            pre_wait=p["pre_wait"], close_t=p["close_steps"], retries=retries, retry_wait=retry_wait)
    if lp is None:
        return dict(ex.log(False, variant), status=f"plan_failed: {why}")
    B = u.cubeB.pose.sp.p.copy()
    if variant == "beside":
        d = u.cubeA.pose.sp.p[:2] - B[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        goal = np.array([B[0] + d[0] * (2 * h + 0.004), B[1] + d[1] * (2 * h + 0.004), h + 0.003])
    else:
        goal = B + np.array([0, 0, 2 * h + p["place_dz"]])
    above = goal + np.array([0, 0, p["align_dz"]])
    ok = ex.move(carry_offset(ex, u.cubeA, above, lp), CONTACT, name="align")
    if ok and p["align_wait"]:
        ex.wait(p["align_wait"])
    ok = ok and ex.move(carry_offset(ex, u.cubeA, goal, ex.tcp()), CONTACT, name="lower")
    if not ok:
        return ex.log(False, variant)
    ex.wait(p["place_wait"])
    if PR.B(p["reaim"]) and variant != "beside":
        e = (u.cubeA.pose.sp.p - (u.cubeB.pose.sp.p + np.array([0, 0, 2 * h])))[:2]
        if 0.006 < np.linalg.norm(e) < 0.025:
            t = ex.tcp()
            ok = ex.move(sapien.Pose(t.p + np.array([0, 0, 0.015]), t.q), CONTACT, name="reaim_up")
            ok = ok and ex.move(sapien.Pose(t.p + np.array([-e[0], -e[1], 0.0]), t.q), CONTACT, name="reaim_down")
            ex.wait(p["place_wait"])
            if not ok:
                return ex.log(False, variant)
    if variant == "hold_on_top":
        ex.wait(5)
    elif variant == "cage":
        cage_open(ex, 2 * h)
    else:
        ex.grip(1, t=p["release_steps"])
        ok = ex.retreat_up(p["up_m"])
    return ex.log(ok, variant)
