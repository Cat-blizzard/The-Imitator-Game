"""Scripted expert and designed counterexamples for DualDrawer-v1 (E29b), any parallel-gripper dual configuration.

run(env, variant) with variant in
  expert                    left arm grasps the knob on the cabinet's side wall (closing axis = cabinet x) and holds it;
                            then the right arm grasps the drawer handle (closing axis = cabinet y) and pulls it
                            10 cm along the drawer axis at 4 cm/s, opens, backs off, goes home; then the left arm opens,
                            backs off, goes home
  pull_without_stabilising  the left arm stays at home; the right arm does the expert's grasp, pull, release and retreat
                            (the unheld cabinet is dragged along: the drawer opens less, the cabinet moves)
  open_small                as the expert, but the pull is 3 cm
  hold_handle               as the expert, but the right arm never lets go of the handle (the left releases and goes home)
  no_op                     both arms stay at home
Grasp frames: closing-axis sign x approach tilt (0 / 0.3 rad leaning with the fingertips away from the plate / wall the knob
sticks out of, or 0.3 / 0.55 rad towards it when the fingertip shift ext * sin(tilt) stays <= 1.2 cm); every waypoint
IK-checked before execution (dual_b2_common.candidates).
"""
from __future__ import annotations

import numpy as np

from ..data.phase import CONTACT, FREE
from ..envs.dual_drawer import CAB, TRAY  # noqa: F401
from .dual_b2_common import UP, Ctx, T_of, candidates, horiz, search_pre, tip_clear_z

PULL = 0.10
PULL_SMALL = 0.03
PULL_SPEED = 0.04
PRE = 0.06
VARIANTS = ("expert", "pull_without_stabilising", "open_small", "hold_handle", "no_op")


def _part_world(link, c) -> np.ndarray:
    T = T_of(link)
    return T[:3, :3] @ np.asarray(c, float) + T[:3, 3]


def cabinet_boxes(u):
    """the cabinet body (outer box) and, at its current opening, the drawer tray: planning obstacles for the approaches"""
    Tc = T_of(u.cabinet_link)
    Td = T_of(u.drawer_link)
    return [(Tc, np.array([CAB["D"] / 2, CAB["W"] / 2, CAB["H"] / 2])),
            (Td, np.array([TRAY["D"] / 2, TRAY["W"] / 2, TRAY["H"] / 2]))]


def grasp_side_knob(ctx: Ctx) -> str:
    """left arm: top-down grasp of the knob on the cabinet's left side wall, closing axis = cabinet x; tilted approaches
    lean with the fingertips away from the wall (approach's horizontal part along +y_cab)"""
    u = ctx.u
    g = u.geo
    Tc = T_of(u.cabinet_link)
    k = _part_world(u.cabinet_link, g["knob_g"])
    xc, yc = horiz(Tc[:3, 0]), horiz(Tc[:3, 1])

    P = getattr(ctx, "p", None) or {}

    def gp(R, tilt):
        p = k.copy(); p[2] = max(k[2], tip_clear_z(ctx, 0, tilt, 0.0, **_tm(P)))
        return p
    ctx.obstacles(0)

    def search(pre_d):
        kp = dict(pre=lambda R, t: gp(R, t) - R[:, 2] * pre_d, grasp=gp)
        c = candidates(ctx, 0, [xc], kp, first_point=k, tilts=(0.0, 0.3), lean_sign=-1.0)
        c += [x for x in candidates(ctx, 0, [xc], kp, first_point=k, tilts=(0.3, 0.55), lean_sign=1.0)
              if ctx.ext[0] * np.sin(x[2]) <= 0.012]
        return sorted(c, key=lambda x: x[0])
    cands, pre_d = search_pre(search, _pres(P))
    kp = dict(pre=lambda R, t: gp(R, t) - R[:, 2] * pre_d, grasp=gp)
    if not cands:
        return "L knob: no_candidate"
    ctx.obstacles(0, boxes=cabinet_boxes(u))
    st = "?"
    for _, R, tilt in cands:
        ctx.phase(FREE, None)
        st = ctx.go(0, kp["pre"](R, tilt), R, "rrt")
        if st != "ok":
            st = f"L knob pre: {st}"; continue
        ctx.phase(CONTACT, None)
        st = ctx.go(0, gp(R, tilt), R, "line")
        if st != "ok":
            st = f"L knob grasp: {st}"; continue
        ctx.log["L_center_err_m"] = ctx.center(0, gp(R, tilt), R)
        ctx.grip(0, P.get("knob_close", 1.0), n=P.get("close_steps", 20))
        ctx.log["L_tilt"] = float(tilt)
        return "ok"
    return st


def grasp_handle(ctx: Ctx, pull: float) -> str:
    """right arm: top-down grasp of the drawer handle (closing axis = cabinet y); `pull` is only IK-checked here"""
    u = ctx.u
    g = u.geo
    Tc = T_of(u.cabinet_link)
    h = _part_world(u.drawer_link, g["handle_g"])
    yc, xc = horiz(Tc[:3, 1]), horiz(Tc[:3, 0])
    out = -xc                                            # drawer opening direction (towards the robots)

    P = getattr(ctx, "p", None) or {}

    def gp(R, tilt):
        p = h.copy(); p[2] = max(h[2], tip_clear_z(ctx, 1, tilt, 0.0, **_tm(P)))
        return p
    ctx.obstacles(1)

    # tilt 0 first; tilted approaches either lean with the fingertips away from the drawer front, or towards it only
    # as far as the 4.5 cm between the grasp point and the plate allows (tip shift ext * sin(tilt) <= 1.2 cm)
    def search(pre_d):
        kp = dict(pre=lambda R, t: gp(R, t) - R[:, 2] * pre_d, grasp=gp, pulled=lambda R, t: gp(R, t) + out * pull)
        c = candidates(ctx, 1, [yc], kp, first_point=h, tilts=(0.0, 0.3), lean_sign=-1.0)
        c += [x for x in candidates(ctx, 1, [yc], kp, first_point=h, tilts=(0.3, 0.55), lean_sign=1.0)
              if ctx.ext[1] * np.sin(x[2]) <= 0.012]
        return sorted(c, key=lambda x: x[0])
    cands, pre_d = search_pre(search, _pres(P))
    kp = dict(pre=lambda R, t: gp(R, t) - R[:, 2] * pre_d, grasp=gp)
    if not cands:
        return "R handle: no_candidate"
    ctx.obstacles(1, boxes=cabinet_boxes(u))
    st = "?"
    for _, R, tilt in cands:
        ctx.phase(None, FREE)
        st = ctx.go(1, kp["pre"](R, tilt), R, "rrt")
        if st != "ok":
            st = f"R handle pre: {st}"; continue
        ctx.phase(None, CONTACT)
        st = ctx.go(1, gp(R, tilt), R, "line")
        if st != "ok":
            st = f"R handle grasp: {st}"; continue
        ctx.log["R_center_err_m"] = ctx.center(1, gp(R, tilt), R)
        ctx.grip(1, P.get("handle_close", 1.0), n=P.get("close_steps", 20))
        ctx.log["R_tilt"] = float(tilt)
        ctx.R_handle = R
        return "ok"
    return st


def pull_handle(ctx: Ctx, pull: float) -> str:
    """right arm (holding the handle) pulls the TCP `pull` along the drawer axis at PULL_SPEED"""
    P = getattr(ctx, "p", None) or {}
    Tc = T_of(ctx.u.cabinet_link)
    out = -horiz(Tc[:3, 0])
    T = ctx.tcp_T(1)
    ctx.phase(None, CONTACT)
    st = ctx.line(1, T[:3, 3] + out * pull, T[:3, :3], speed=P.get("pull_speed", PULL_SPEED))
    if st != "ok":
        return f"R pull: {st}"
    ctx.wait(P.get("post_wait", 5))
    return "ok"


def release_retreat(ctx: Ctx, i: int) -> str:
    P = getattr(ctx, "p", None) or {}
    ctx.set_arm_phase(i, CONTACT)
    ctx.grip(i, 0.0, n=P.get("release_steps", 20))
    if P.get("release_wait", 0):
        ctx.wait(P["release_wait"])
    ctx.set_arm_phase(i, FREE)
    return ctx.retreat(i, back=P.get("back_m", 0.07))


def _tm(P: dict) -> dict:
    """tip_clear_z keyword (A2 'tip_margin'; frozen: the function's default 4 mm)"""
    return {} if "tip_margin" not in P or P["tip_margin"] == 0.004 else dict(margin=P["tip_margin"])


def _pres(P: dict):
    """pre-grasp distances tried in order (A2 'pre_scale' x the frozen 6 / 4 / 3 cm)"""
    from .dual_b2_common import PRES
    sc = P.get("pre_scale", 1.0)
    return PRES if sc == 1.0 else tuple(x * sc for x in PRES)


def run(env, variant: str = "expert", params: dict | None = None) -> dict:
    """params (A2, imitator.planning.params 'DualDrawer-v1'): pull length and speed, close commands / durations, grasp
    tip margin, pre-grasp distances, post-pull wait, release duration / wait, retreat distance; None = the frozen values
    (+ an adopted tuned set)"""
    if variant not in VARIANTS:
        raise ValueError(variant)
    from . import params as PR
    ctx = Ctx(env)
    ctx.p = PR.pv(params, "DualDrawer-v1", env, variant)
    ctx.log["variant"] = variant
    ctx.phase(FREE, FREE)
    ctx.wait(1)
    if variant == "no_op":
        ctx.wait(20)
        ctx.log["status"] = "ok"
        return _done(ctx)
    pull = PULL_SMALL if variant == "open_small" else ctx.p.get("pull_m", PULL)
    stabilise = variant != "pull_without_stabilising"
    # The holder takes the cabinet first (then the handle grasp cannot shove the free-standing cabinet); the right arm
    # plans its handle grasp around the left arm. (Kept as a switch: the earlier top-knob layout needed the reverse.)
    knob_first = stabilise
    ctx.log["order"] = "knob_first" if knob_first else "handle_first"
    st = grasp_side_knob(ctx) if knob_first else "ok"
    if st == "ok":
        st = grasp_handle(ctx, pull)
    if st == "ok" and stabilise and not knob_first:
        st = grasp_side_knob(ctx)                        # right arm holds the handle (CONTACT)
    if st == "ok":
        st = pull_handle(ctx, pull)
    if st == "ok" and variant != "hold_handle":
        st = release_retreat(ctx, 1)
    if st == "ok" and variant != "pull_without_stabilising":
        st = release_retreat(ctx, 0)
    if st == "ok":
        ctx.wait(5)
    ctx.log["status"] = st
    return _done(ctx)


def _done(ctx: Ctx) -> dict:
    u = ctx.u
    ctx.log.update(steps=ctx.ex.n, drawer_open=float(u.drawer_open()[0]), cabinet_disp=float(u.cabinet_disp()[0]),
                   final_targets=[t.tolist() for t in ctx.ex.targets])
    return ctx.log
