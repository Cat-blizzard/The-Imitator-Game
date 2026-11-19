"""Card D2: scripted experts and designed counterexamples for RingOnPeg-v1 and MugOnHook-v1 (imitator/envs/topo_tasks.py),
with phase annotation (imitator.data.phase: FREE = transit with nothing grasped, CONTACT = final approach, grasp, carry,
place, release). Every behaviour is f(env, variant=...) -> dict(status=...), env = the (recorded) env, already reset.

Motion: imitator.planning.panda_ms.PandaExec (ManiSkill's mplib screw / RRT planner; it does not see the scene, so every
waypoint sequence below is collision-free by construction: lift high, move over, descend vertically or along the hook).

RingOnPeg   expert       grasp the ring's tube on the side facing the robot (top-down, closing radially), lift it 3 cm
                         above the peg top, centre it over the peg (offset 8 mm away from the gripper so the inner finger
                         clears the peg), lower until the peg is 2 cm inside the ring, open (the ring drops down the peg),
                         retreat 10 cm up
            on_top       the far side of the ring is put on the peg's top face, released (the near side drops to the
                         table: the ring ends leaning on the peg top or slides off beside it)
            beside       the ring is put flat on the table touching the peg (outer edge 2 mm from it), released
            held         as the expert, but lowered around the peg to 3 cm height and kept in the closed gripper
            no_op        the arm stays at its start pose
MugOnHook   expert       grasp the mug's rim opposite the handle (top-down, closing radially), lift, turn the handle
                         towards +y, bring the handle hole onto the hook axis 3 cm beyond the tip, slide it along the
                         hook to 5 cm from the post (the hook's middle), lower 8 mm, open (the mug swings and hangs),
                         retreat
            expert_shallow  positive control 2: the same, handle slid only to 8 cm from the post (2.5 cm of hook beyond
                         the handle)
            on_tip       "balanced on the hook tip": the handle slid only 6 mm past the hook's free end, released
            held_near    the handle brought in front of the hook tip (3 cm beyond it), gripper stays closed
            beside       the handle passed beside the hook (hole 3 cm off the hook axis, towards the mug body), released
            dropped_below  the handle released 6 cm below the hook's middle
            no_op        the arm stays at its start pose
"""
from __future__ import annotations

import numpy as np
import sapien

from ..data.phase import CONTACT, FREE
from .panda_ms import PandaExec, no_op

BASE_XY = np.array([-0.615, 0.0])


def _grasp_pose(ex: PandaExec, point, closing) -> list:
    """top-down grasp poses at `point` with the closing axis +-`closing` (both finger assignments)"""
    a = ex.u.agent
    out = []
    for s in (1.0, -1.0):
        out.append(a.build_grasp_pose(np.array([0, 0, -1.0]), s * np.asarray(closing, float), np.asarray(point, float)))
    return out


def _reachable(ex: PandaExec, poses) -> sapien.Pose | None:
    for p in poses:
        if ex.can_reach(p * sapien.Pose([0, 0, -0.06])) and ex.can_reach(p):
            return p
    return None


def _shift(ex: PandaExec, d, phase=CONTACT, name: str = "move") -> bool:
    """translate the TCP by d (world), keeping its orientation"""
    p = ex.tcp()
    return ex.move(sapien.Pose(p.p + np.asarray(d, float), p.q), phase, name=name)


# ================================================================================================ RingOnPeg
def ring_on_peg(env, variant: str = "expert") -> dict:
    if variant == "no_op":
        return no_op(env)
    ex = PandaExec(env)
    u = ex.u
    R, A, rp, H = u.RING_R, u.RING_A, u.PEG_R, u.PEG_H
    c = u.ring.pose.sp.p.copy()
    peg = u.peg.pose.sp.p.copy()
    # grasp point: the ring's core circle on the side facing the robot base
    to_base = BASE_XY - c[:2]
    phi = float(np.arctan2(to_base[1], to_base[0]))
    radial = np.array([np.cos(phi), np.sin(phi), 0.0])
    g = None
    for dphi in (0.0, 0.35, -0.35, 0.7, -0.7):
        rad = np.array([np.cos(phi + dphi), np.sin(phi + dphi), 0.0])
        pt = c + R * rad
        pt[2] = 0.012                                   # TCP height: fingertips 2.5 mm above the table
        g = _reachable(ex, _grasp_pose(ex, pt, rad))
        if g is not None:
            radial = rad
            break
    if g is None:
        return ex.log(False, variant, reason="no reachable grasp")
    if not ex.move(g * sapien.Pose([0, 0, -0.06]), FREE, name="reach"):
        return ex.log(False, variant)
    if not ex.move(g, CONTACT, name="grasp"):
        return ex.log(False, variant)
    ex.grip(-1)
    u_in = -radial                                      # from the gripper towards the ring centre
    # lift: ring bottom 3 cm above the peg top
    rz = lambda: float(u.ring.pose.sp.p[2])              # noqa: E731
    if not _shift(ex, [0, 0, (H + 0.03 + A) - rz()], name="lift"):
        return ex.log(False, variant)
    ring_c = lambda: u.ring.pose.sp.p.copy()              # noqa: E731
    if variant == "beside":
        target = peg[:2] - (R + A + rp + 0.002) * u_in[:2]
        z_goal = A + 0.003
    elif variant == "on_top":
        target = peg[:2] - R * u_in[:2]                 # far side of the ring over the peg axis
        z_goal = H + A + 0.002
    else:
        target = peg[:2] - 0.008 * u_in[:2]             # peg axis 8 mm towards the far side of the hole
        z_goal = H - 0.02
    cc = ring_c()
    if not _shift(ex, [target[0] - cc[0], target[1] - cc[1], 0.0], name="carry"):
        return ex.log(False, variant)
    cc = ring_c()                                       # re-aim once from the actual in-hand pose
    if not _shift(ex, [target[0] - cc[0], target[1] - cc[1], 0.0], name="aim"):
        return ex.log(False, variant)
    if not _shift(ex, [0, 0, z_goal - rz()], name="lower"):
        return ex.log(False, variant)
    if variant == "held":
        ok = _shift(ex, [0, 0, 0.03 - rz()], name="lower_more")
        ex.wait(5)
        return ex.log(ok, variant)
    ex.wait(2)
    ex.grip(1)
    ex.wait(4)
    ok = ex.retreat_up(0.10)
    return ex.log(ok, variant)


# ================================================================================================ MugOnHook
MUG_DEPTHS = {"expert": 0.05, "expert_shallow": 0.08}


def mug_on_hook(env, variant: str = "expert") -> dict:
    if variant == "no_op":
        return no_op(env)
    ex = PandaExec(env)
    u = ex.u
    Tm = u.mug.pose.sp.to_transformation_matrix()
    xm = Tm[:3, 0].copy(); xm[2] = 0.0; xm /= np.linalg.norm(xm)       # mug local +x (towards the handle), horizontal
    rim = Tm[:3, :3] @ np.array([-(u.MUG_R - u.MUG_WALL / 2), 0.0, u.MUG_H]) + Tm[:3, 3]
    pt = rim.copy(); pt[2] -= 0.015
    g = _reachable(ex, _grasp_pose(ex, pt, xm))
    if g is None:
        return ex.log(False, variant, reason="no reachable grasp")
    if not ex.move(g * sapien.Pose([0, 0, -0.06]), FREE, name="reach"):
        return ex.log(False, variant)
    if not ex.move(g, CONTACT, name="grasp"):
        return ex.log(False, variant)
    ex.grip(-1)
    if not _shift(ex, [0, 0, 0.06], name="lift"):
        return ex.log(False, variant)
    # turn the mug about the vertical through the grasp so that the handle points to world +y
    Tt = ex.tcp().to_transformation_matrix()
    s = 1.0 if float(Tt[:3, 1] @ xm) > 0 else -1.0                      # x_mug = s * y_tcp (fixed by the grasp)
    cand = u.agent.build_grasp_pose(np.array([0, 0, -1.0]), s * np.array([0.0, 1.0, 0.0]), ex.tcp().p)
    if not ex.move(cand, CONTACT, name="turn"):                          # handle -> world +y
        return ex.log(False, variant)
    Tr = u.rack_T()
    d = Tr[:3, :3] @ u.hook_dir()                                        # hook direction root -> tip (world)
    root = Tr[:3, :3] @ u.hook_root() + Tr[:3, 3]
    end = root + (u.HOOK_L + u.HOOK_R) * d                               # the hook's free end
    hw = lambda: u.handle_world()                                        # noqa: E731

    def to_handle(target, name):
        h = hw()
        return _shift(ex, target - h, name=name)

    side = np.array([0.0, -0.03, 0.0]) if variant == "beside" else np.zeros(3)
    pre = end + 0.03 * d + side
    if variant == "dropped_below":
        pre = root + 0.05 * d + np.array([0, 0, -0.06])
        ok = to_handle(pre + np.array([0, 0, -0.03]), "below") and to_handle(pre, "below_fine")
    else:
        # come from below the pre-thread point, then up to it (the hook stays above the mug until then)
        ok = to_handle(pre + np.array([0, 0, -0.06]), "pre_low") and to_handle(pre, "pre") and to_handle(pre, "pre_fine")
    if not ok:
        return ex.log(False, variant)
    if variant == "held_near":
        ex.wait(5)
        return ex.log(True, variant)
    if variant != "dropped_below":
        depth = {"on_tip": u.HOOK_L + u.HOOK_R - 0.006}.get(variant, MUG_DEPTHS.get(variant, 0.05))
        target = root + depth * d + side
        if not to_handle(target, "thread"):
            return ex.log(False, variant)
        if not _shift(ex, [0, 0, -0.008], name="lower"):
            return ex.log(False, variant)
    ex.wait(2)
    ex.grip(1)
    ex.wait(6)
    ok = ex.retreat_up(0.10)
    return ex.log(ok, variant)
