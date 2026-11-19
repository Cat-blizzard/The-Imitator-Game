"""Scripted expert and designed counterexamples for DualPegInHole-v1 (E29b), any parallel-gripper dual configuration.

run(env, variant) with variant in
  expert          left arm grasps the block's handle top-down (closing axis = block x, 11.5 cm from the hole axis;
                  8 cm for the Robotiq, see HANDLE_GRASP_Y_WEAK) and
                  lifts the block 3.5 cm, then holds it; right arm grasps the standing peg 2 cm
                  below its top (closing axis = a peg face normal), lifts it, carries its bottom centre over the hole
                  mouth (privileged poses re-read), inserts along the hole axis until the peg bottom is 4 mm above the
                  hole floor, opens, backs off along its approach axis, goes home; the left arm keeps holding the block
  hold_peg        as the expert, but the right arm never opens (the peg stays in its gripper, inside the hole)
  peg_at_entrance as the expert, but the right arm stops with the peg bottom 1 cm above the hole mouth and holds it there
  block_on_table  the left arm never moves (the block stays on the table); the right arm inserts the peg into the block
                  on the table, opens, backs off, goes home
  no_op           both arms stay at home
"""
from __future__ import annotations

import numpy as np

from ..data.phase import CONTACT, FREE
from ..envs.dual_peg_in_hole import FLOOR_TOP, HANDLE_C, HANDLE_HALF, HOLE_OUT, MOUTH, PEG_HALF, block_parts
from .dual_b2_common import UP, Ctx, T_of, attach_actor_box, candidates, horiz, search_pre, tip_clear_z

LIFT = 0.035                # block lift [m]
HANDLE_GRASP_Y = HOLE_OUT[1] + 0.085    # grasp point along the handle (block frame y): 11.5 cm from the hole axis
HANDLE_GRASP_Y_WEAK = HOLE_OUT[1] + 0.05  # ... 8 cm for the xArm6 + Robotiq 2F-85 (weak ManiSkill gripper drive, E23:
                                        # a smaller gravity torque on the grasp); the other grippers leave the inserting
                                        # arm more room with the farther grasp point
HOLD_CLOSE = {"xarm6_robotiq": 0.7}   # holder's closing command (fraction of fully closed): the Robotiq 2F-85's closed
                                      # linkage chatters when squeezed with the full 0.81 rad command (block and peg
                                      # vibrate ~1-2 cm/s in 3 of 20 final-run episodes); 0.7 * 0.81 rad stays ~0.13 rad
                                      # past the 4 cm handle's contact angle
PEG_GRASP_BELOW_TOP = 0.02
ABOVE = 0.025               # peg bottom above the mouth before inserting
ENTRANCE = 0.01             # peg_at_entrance: peg bottom this far above the mouth
INSERT_TO = 0.004           # peg bottom above the hole floor at the end of the insertion
PRE = 0.06
VARIANTS = ("expert", "hold_peg", "peg_at_entrance", "block_on_table", "no_op")


def block_boxes(u):
    T = T_of(u.block)
    out = []
    for c, h in block_parts():
        Tp = T.copy(); Tp[:3, 3] = T[:3, :3] @ np.asarray(c, float) + T[:3, 3]
        out.append((Tp, np.asarray(h, float)))
    return out


def peg_axis_up(u) -> np.ndarray:
    a = T_of(u.peg)[:3, 2]
    return a if a[2] >= 0 else -a


def peg_bottom(u) -> np.ndarray:
    return T_of(u.peg)[:3, 3] - peg_axis_up(u) * PEG_HALF[2]


def grasp_lift_block(ctx: Ctx) -> str:
    u = ctx.u
    Tb = T_of(u.block)
    gy = HANDLE_GRASP_Y_WEAK if ctx.ex.agents[0].record.name == "xarm6_robotiq" else HANDLE_GRASP_Y
    ctx.log["handle_grasp_y"] = gy
    g0 = Tb[:3, :3] @ np.array([0.0, gy, HANDLE_C[2]]) + Tb[:3, 3]
    xb = horiz(Tb[:3, 0])

    def gp(R, tilt):
        p = g0.copy(); p[2] = max(g0[2], tip_clear_z(ctx, 0, tilt, 0.0))
        return p
    ctx.obstacles(0)
    # tilted grasps first (approach leaning towards the hole, so the holder's wrist stays on the far side of the handle
    # and leaves the inserting arm room above the hole); upright only if no tilted frame is feasible
    def search(d, tilts):
        return candidates(ctx, 0, [xb], dict(pre=lambda R, t: gp(R, t) - R[:, 2] * d, grasp=gp,
                                             lift=lambda R, t: gp(R, t) + UP * LIFT), first_point=g0, tilts=tilts)
    cands, pre_d = search_pre(lambda d: search(d, (0.3, 0.55)) or search(d, (0.0,)))
    kp = dict(pre=lambda R, t: gp(R, t) - R[:, 2] * pre_d, grasp=gp)
    if not cands:
        return "L block: no_candidate"
    st = "?"
    for _, R, tilt in cands:
        ctx.obstacles(0, boxes=block_boxes(u))
        ctx.phase(FREE, None)
        st = ctx.go(0, kp["pre"](R, tilt), R, "rrt")
        if st != "ok":
            st = f"L pre: {st}"; continue
        ctx.phase(CONTACT, None)
        st = ctx.go(0, gp(R, tilt), R, "line")
        if st != "ok":
            st = f"L grasp: {st}"; continue
        ctx.log["L_center_err_m"] = ctx.center(0, gp(R, tilt), R)
        ctx.grip(0, HOLD_CLOSE.get(ctx.ex.agents[0].record.name, 1.0), n=25)
        st = ctx.line(0, gp(R, tilt) + UP * LIFT, R, speed=0.05)
        if st != "ok":
            return f"L lift: {st}"
        ctx.wait(10)
        ctx.log["L_tilt"] = float(tilt)
        return "ok"
    return st


def _mouth_frame(u):
    """(mouth centre, hole axis pointing out of the hole) in the world from the block's current pose"""
    Tb = T_of(u.block)
    n = Tb[:3, 2]
    return Tb[:3, :3] @ np.array([0.0, 0.0, MOUTH]) + Tb[:3, 3], n


def pick_insert_peg(ctx: Ctx, variant: str) -> str:
    u = ctx.u
    Tp = T_of(u.peg)
    top = Tp[:3, 3] + peg_axis_up(u) * PEG_HALF[2]
    g0 = top - UP * PEG_GRASP_BELOW_TOP
    faces = [horiz(Tp[:3, 0]), horiz(Tp[:3, 1])]
    mouth, n = _mouth_frame(u)
    off = g0 - (Tp[:3, 3] - peg_axis_up(u) * PEG_HALF[2])            # TCP - peg bottom (nominal grasp)
    above = mouth + n * ABOVE + off
    stop = mouth + n * (ENTRANCE if variant == "peg_at_entrance" else -(MOUTH - FLOOR_TOP - INSERT_TO)) + off
    lift_z = max(g0[2] + 0.03, above[2] + 0.02)

    def gp(R, tilt):
        return g0.copy()
    def kps(d):
        return dict(pre=lambda R, t: g0 - R[:, 2] * d, grasp=gp, lift=lambda R, t: np.array([g0[0], g0[1], lift_z]),
                    above=lambda R, t: above, stop=lambda R, t: stop)
    ctx.obstacles(1, radius=0.006)
    # closing axis = the peg face normal closest to the block's x axis first (fingers beside the hole, away from the
    # holder's gripper on the +y handle), the other face normal only if that has no feasible frame
    xb = horiz(T_of(u.block)[:3, 0])
    faces.sort(key=lambda f: -abs(float(f @ xb)))
    cands, pre_d = search_pre(lambda d: candidates(ctx, 1, faces[:1], kps(d), first_point=g0)
                              or candidates(ctx, 1, faces[1:], kps(d), first_point=g0))
    kp = kps(pre_d)
    if not cands:
        return "R peg: no_candidate"
    st = "?"
    for _, R, tilt in cands:
        ctx.obstacles(1, boxes=block_boxes(u))
        ctx.phase(None, FREE)
        st = ctx.go(1, kp["pre"](R, tilt), R, "rrt")
        if st != "ok":
            st = f"R pre: {st}"; continue
        ctx.phase(None, CONTACT)
        st = ctx.go(1, g0, R, "line")
        if st != "ok":
            st = f"R grasp: {st}"; continue
        ctx.log["R_center_err_m"] = ctx.center(1, g0, R)
        ctx.grip(1, 1.0, n=25)
        attach_actor_box(ctx, 1, u.peg, PEG_HALF)
        st = ctx.line(1, kp["lift"](R, tilt), R, speed=0.08)
        if st != "ok":
            return f"R lift: {st}"
        # carry: aim the peg's actual bottom centre (it may have shifted in the grasp) over the mouth
        mouth, n = _mouth_frame(u)
        d = ctx.tcp_T(1)[:3, 3] - peg_bottom(u)
        ctx.obstacles(1, boxes=block_boxes(u))
        st = ctx.go(1, mouth + n * ABOVE + d, R, "screw", attach=True)
        if st != "ok":
            return f"R carry: {st}"
        ctx.wait(5)
        mouth, n = _mouth_frame(u)
        d = ctx.tcp_T(1)[:3, 3] - peg_bottom(u)
        st = ctx.line(1, mouth + n * ABOVE + d, R, speed=0.05)          # re-aim from the actual pose
        if st != "ok":
            return f"R aim: {st}"
        tgt = mouth + n * (ENTRANCE if variant == "peg_at_entrance" else -(MOUTH - FLOOR_TOP - INSERT_TO)) + d
        st = ctx.line(1, tgt, R, speed=0.04)
        if st != "ok":
            return f"R insert: {st}"
        ctx.wait(8)
        ctx.log["R_tilt"] = float(tilt)
        return "ok"
    return st


def run(env, variant: str = "expert") -> dict:
    if variant not in VARIANTS:
        raise ValueError(variant)
    ctx = Ctx(env)
    ctx.log["variant"] = variant
    ctx.phase(FREE, FREE)
    ctx.wait(1)
    if variant == "no_op":
        ctx.wait(20)
        ctx.log["status"] = "ok"
        return _done(ctx)
    st = "ok"
    if variant != "block_on_table":
        st = grasp_lift_block(ctx)                      # right arm idle (FREE)
    if st == "ok":
        st = pick_insert_peg(ctx, variant)              # left arm holds (CONTACT) or idles (block_on_table: FREE)
    if st == "ok" and variant in ("expert", "block_on_table"):
        ctx.set_arm_phase(1, CONTACT)
        ctx.grip(1, 0.0, n=20)
        ctx.pl[1].detach()
        ctx.set_arm_phase(1, FREE)
        st = ctx.retreat(1)
    if st == "ok":
        ctx.wait(5)
    ctx.log["status"] = st
    return _done(ctx)


def _done(ctx: Ctx) -> dict:
    u = ctx.u
    r, h = u.peg_in_block()
    ctx.log.update(steps=ctx.ex.n, peg_radial=float(r[0]), peg_height=float(h[0]),
                   block_bottom_z=float(T_of(u.block)[2, 3] - HOLE_OUT[2]), final_targets=[t.tolist() for t in ctx.ex.targets])
    return ctx.log
