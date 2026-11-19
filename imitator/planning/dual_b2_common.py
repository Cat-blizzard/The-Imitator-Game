"""Shared helpers for the E29b fork-B2 dual-arm experts (DualPegInHole-v1, DualDrawer-v1): grasp-frame search over
closing-axis sign x approach tilt (every key waypoint IK-checked, as in planning/handover.py), phase-annotated motion
segments, release / retreat / home.

Phases (imitator.data.phase): every segment calls set_phase for BOTH arms before its env.step calls (FREE = free transit
with nothing grasped or an idle arm; CONTACT = final approach, grasp, hold, lift, carry, insert, pull, release).
"""
from __future__ import annotations

import numpy as np
import sapien

from ..data.phase import CONTACT, FREE, set_phase
from ..verify.v31 import box_surface_points
from .mplib_planner import ArmPlanner, arm_points
from .scripted import UP, DualExecutor, _go, frame, pose, rot_angle, tcp_extent

TILTS = (0.0, 0.3, 0.55)
PRES = (0.06, 0.04, 0.03)        # pre-grasp distances along the approach axis, tried in order (small arms cannot point
                                 # down 6 cm above a low grasp point)


def search_pre(fn, pres=PRES):
    """fn(pre_distance) -> candidate list; the first non-empty result and its pre distance"""
    for d in pres:
        c = fn(d)
        if c:
            return c, d
    return [], pres[-1]


class Ctx:
    def __init__(self, env):
        u = env.unwrapped
        self.env, self.u = env, u
        self.ex = DualExecutor(env)
        self.bases = u.dual.mount_poses()
        self.pl = [ArmPlanner(ag, b, control_dt=float(u.control_timestep)) for ag, b in zip(self.ex.agents, self.bases)]
        self.ext = [tcp_extent(ag) for ag in self.ex.agents]
        self.ph = [FREE, FREE]
        self.log = dict(status="?")

    # ---------------------------------------------------------------- phases
    def phase(self, p0=None, p1=None):
        if p0 is not None:
            self.ph[0] = p0
        if p1 is not None:
            self.ph[1] = p1
        set_phase(self.env, self.ph[0], arm=0)
        set_phase(self.env, self.ph[1], arm=1)

    def set_arm_phase(self, i: int, p: str):
        self.phase(*(([p, None]) if i == 0 else ([None, p])))

    # ---------------------------------------------------------------- obstacles
    def obstacles(self, i: int, boxes=(), radius: float = 0.01, other_arm: bool = True):
        """the other arm (point cloud) + extra boxes [(T_world (4x4), half3)] as obstacles of arm i's planner"""
        pts = [arm_points(self.ex.agents[1 - i])] if other_arm else []
        for T, h in boxes:
            pts.append(box_surface_points(h, 7) @ T[:3, :3].T + T[:3, 3])
        P = np.vstack(pts) if pts else None
        self.pl[i].set_obstacle_points(P, radius=radius)

    # ---------------------------------------------------------------- motion (phase set by the caller)
    def go(self, i: int, p, R, method: str, attach: bool = False, fallback: bool = True) -> str:
        return _go(self.ex, i, self.pl[i], pose(p, R), method, attach=attach, fallback=fallback)

    def line(self, i: int, p, R, speed: float = 0.1) -> str:
        res = self.pl[i].cartesian(pose(p, R), speed=speed)
        if res["status"] != "Success":
            res = self.pl[i].joint_interp(pose(p, R))
        if res["status"] != "Success":
            return res["status"]
        self.ex.follow(i, res["position"])
        return "ok"

    def center(self, i: int, p_des, R, iters: int = 3, tol: float = 0.001) -> float:
        """closed-loop correction of the TCP's horizontal steady-state error before closing the gripper (an off-centre
        close pushes a free-standing object by the error): command p_des + accumulated error until the measured TCP is
        within `tol` of p_des horizontally. Returns the final horizontal error [m]."""
        cmd = np.asarray(p_des, float).copy()
        e = np.zeros(3)
        for _ in range(iters):
            self.wait(6)
            e = np.asarray(p_des, float) - self.tcp_T(i)[:3, 3]
            e[2] = 0.0
            if np.linalg.norm(e) < tol:
                break
            cmd = cmd + e
            if self.line(i, cmd, R, speed=0.05) != "ok":
                break
        return float(np.linalg.norm(e))

    def tcp_T(self, i: int) -> np.ndarray:
        return self.ex.agents[i].tcp.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)

    def grip(self, i: int, closed: float, n: int = 20):
        self.ex.set_gripper(i, closed, n=n)

    def grip_ramp(self, i: int, closed: float, n: int = 25, hold: int = 10):
        """close / open with the gripper command ramped linearly over n control steps, then hold for `hold` steps (a
        step command makes the faster finger shove a free-standing object sideways before the other one arrives)"""
        ag = self.ex.agents[i]
        nd = ag.record.dof_arm
        g0 = np.asarray(self.ex.targets[i][nd:], np.float32).copy()
        g1 = np.asarray(ag.gripper_action(closed), np.float32)
        for k in range(1, n + 1):
            self.ex.targets[i][nd:] = g0 + (g1 - g0) * (k / n)
            self.ex.step()
        self.ex.step(hold)

    def wait(self, n: int):
        self.ex.step(n)

    def home(self, i: int) -> str:
        ag = self.ex.agents[i]
        self.obstacles(i)
        res = self.pl[i].plan_qpos(np.asarray(ag.record.home_arm_qpos, float))
        if res["status"] != "Success":
            return res["status"]
        self.ex.follow(i, res["position"])
        return "ok"

    def retreat(self, i: int, back: float = 0.07) -> str:
        """back the TCP off along its approach axis (-z_tcp), else straight up, then home (the gripper is open)"""
        T = self.tcp_T(i)
        st = "?"
        for d in (back, 0.05, 0.03):
            st = self.go(i, T[:3, 3] - T[:3, 2] * d, T[:3, :3], "line", fallback=False)
            if st == "ok":
                break
        if st != "ok":
            for d in (back, 0.05):
                st = self.go(i, T[:3, 3] + UP * d, T[:3, :3], "line", fallback=False)
                if st == "ok":
                    break
        h = self.home(i)
        return "ok" if (st == "ok" or h == "ok") else f"retreat {st}"


def candidates(ctx: Ctx, i: int, closing_axes, key_points: dict, tcp_z_of_tilt=None, lean_sign: float = 1.0,
               tilts=TILTS, max_n: int = 6, first_point=None):
    """IK-feasible grasp frames for arm i. closing_axes: horizontal unit vectors (both signs are tried); approach =
    -UP cos(tilt) + d sin(tilt), d horizontal, perpendicular to the closing axis, pointing away from the arm's base
    (lean_sign=-1: towards it). key_points: name -> f(R, tilt) -> TCP position; every one must be IK-feasible (seeded
    in order). Returns a list of (score, R, tilt) best first."""
    pl = ctx.pl[i]
    base = np.asarray(ctx.bases[i].p, float)
    R_now = pl.fk_tcp(pl.qpos())[:3, :3]
    p0 = first_point if first_point is not None else next(iter(key_points.values()))(np.eye(3), 0.0)
    radial = np.array([p0[0] - base[0], p0[1] - base[1], 0.0])
    radial /= np.linalg.norm(radial) + 1e-12
    out = []
    for a in closing_axes:
        for sgn in (1.0, -1.0):
            y = sgn * np.asarray(a, float)
            d = np.cross(y, UP)
            if lean_sign * (d @ radial) < 0:
                d = -d
            for tilt in tilts:
                R = frame(y, -UP * np.cos(tilt) + d * np.sin(tilt))
                out.append((tilt * 2.0 + rot_angle(R_now, R) * 0.3, R, tilt))
    out.sort(key=lambda c: c[0])
    good = []
    for score, R, tilt in out:
        ok, q_seed = True, None
        for name, fn in key_points.items():
            p = fn(R, tilt)
            g, sol = pl.ik(pose(p, R), q_start=q_seed, n_init=10)
            if not g and q_seed is not None:
                g, sol = pl.ik(pose(p, R), q_start=None, n_init=10)
            if not g:
                ok = False
                break
            q_seed = sol[0]
        if ok:
            good.append((score, R, tilt))
            if len(good) >= max_n:
                break
    return good


def tip_clear_z(ctx: Ctx, i: int, tilt: float, surface_z: float, margin: float = 0.004, half_w: float = 0.015) -> float:
    """lowest TCP height at which arm i's fingertips stay `margin` above a horizontal surface at surface_z"""
    return surface_z + margin + ctx.ext[i] * np.cos(tilt) + half_w * np.sin(tilt)


def T_of(actor) -> np.ndarray:
    return actor.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)


def horiz(v) -> np.ndarray:
    v = np.asarray(v, float).copy()
    v[2] = 0.0
    return v / (np.linalg.norm(v) + 1e-12)


def attach_actor_box(ctx: Ctx, i: int, actor, half, shrink: float = 0.012):
    """planning-only proxy of a grasped box-shaped actor, attached to arm i's TCP"""
    from transforms3d.quaternions import mat2quat
    T_rel = np.linalg.inv(ctx.tcp_T(i)) @ T_of(actor)
    ctx.pl[i].attach_box(np.maximum(2 * np.asarray(half, float) - shrink, 0.004),
                         sapien.Pose(T_rel[:3, 3], mat2quat(T_rel[:3, :3])))
