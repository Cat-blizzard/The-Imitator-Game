"""Shared helpers for the E29b B1 dual-arm experts (DualLiftBox-v1, DualStack-v1): mplib planning per arm, grasp candidate
search (closing axis along a box face normal, approach tilt 0 / 0.3 / 0.55 rad, waypoint heights 10 / 7 / 5 cm, every
waypoint IK-feasible with the other arm as a point cloud, as in planning/scripted.py and planning/handover.py), phase
annotation (imitator.data.phase) per arm before every segment.
"""
from __future__ import annotations

import numpy as np
import sapien

from ..data.phase import CONTACT, FREE, set_phase
from .mplib_planner import ArmPlanner, arm_points
from .scripted import UP, DualExecutor, _go, cage, frame, pose, rot_angle, tcp_extent

TILTS = (0.0, 0.3, 0.55)
HEIGHTS = (0.10, 0.07, 0.05)


def actor_T(a) -> np.ndarray:
    return a.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)


class Arms:
    def __init__(self, env):
        self.env = env
        self.u = env.unwrapped
        self.ex = DualExecutor(env)
        self.bases = self.u.dual.mount_poses()
        dt = float(self.u.control_timestep)
        self.p = [ArmPlanner(ag, b, control_dt=dt) for ag, b in zip(self.ex.agents, self.bases)]
        self.ext = [tcp_extent(ag) for ag in self.ex.agents]
        self.failed: list[str] = []

    # ---------------------------------------------------------------- phases
    def phase(self, p0: str | None = None, p1: str | None = None):
        if p0 is not None:
            set_phase(self.env, p0, arm=0)
        if p1 is not None:
            set_phase(self.env, p1, arm=1)

    def set(self, i: int, p: str):
        set_phase(self.env, p, arm=i)

    # ---------------------------------------------------------------- motion
    def obstacles(self, i: int, extra=None, radius: float = 0.01):
        pts = arm_points(self.ex.agents[1 - i])
        if extra is not None and len(extra):
            pts = np.vstack([pts, extra])
        self.p[i].set_obstacle_points(pts, radius=radius)

    def go(self, i: int, target: sapien.Pose, method: str, phase: str, attach: bool = False, name: str = "") -> bool:
        self.set(i, phase)
        st = _go(self.ex, i, self.p[i], target, method, attach=attach)
        if st != "ok":
            self.failed.append(f"{name or method}: {st}"[:80])
            return False
        return True

    def grip(self, i: int, closed: float, n: int = 20, phase: str = CONTACT):
        self.set(i, phase)
        self.ex.set_gripper(i, closed, n=n)

    def grip_both(self, closed: float, n: int = 25, phase: str = CONTACT):
        self.phase(phase, phase)
        for i, ag in enumerate(self.ex.agents):
            self.ex.targets[i][ag.record.dof_arm:] = ag.gripper_action(closed)
        self.ex.step(n)

    def wait(self, n: int, p0: str | None = None, p1: str | None = None):
        self.phase(p0, p1)
        self.ex.step(n)

    def home(self, i: int) -> bool:
        self.set(i, FREE)
        self.obstacles(i)
        res = self.p[i].plan_qpos(np.asarray(self.ex.agents[i].record.home_arm_qpos, float))
        if res["status"] == "Success":
            self.ex.follow(i, res["position"])
            return True
        return False

    def tcp_T(self, i: int) -> np.ndarray:
        return self.ex.agents[i].tcp.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)

    def follow_both(self, paths: dict, settle: int = 3):
        n = max(len(p) for p in paths.values())
        for k in range(n):
            for i, P in paths.items():
                self.ex.targets[i][:self.ex.agents[i].record.dof_arm] = P[min(k, len(P) - 1)]
            self.ex.step()
        self.ex.step(settle)

    def lift_paths(self, delta: np.ndarray, arms=(0, 1), speed: float = 0.06):
        """cartesian TCP paths moving every arm in `arms` by the same displacement (same number of steps)"""
        paths = {}
        for i in arms:
            T = self.tcp_T(i)
            tgt = pose(T[:3, 3] + delta, T[:3, :3])
            res = self.p[i].cartesian(tgt, speed=speed)
            if res["status"] != "Success":
                res = self.p[i].joint_interp(tgt)
            if res["status"] != "Success":
                self.failed.append(f"lift arm {i}: {res['status']}"[:80])
                return None
            paths[i] = res["position"]
        n = max(len(p) for p in paths.values())
        for i in paths:                         # resample to the same length so the TCPs move together
            P = paths[i]
            idx = np.linspace(0, len(P) - 1, n)
            paths[i] = np.array([P[int(round(j))] for j in idx])
        return paths

    def ik_chain(self, i: int, points: list, R: np.ndarray) -> bool:
        q_seed = None
        for p in points:
            ok, sol = self.p[i].ik(pose(p, R), q_start=q_seed, n_init=10)
            if not ok and q_seed is not None:
                ok, sol = self.p[i].ik(pose(p, R), q_start=None, n_init=10)
            if not ok:
                return False
            q_seed = sol[0]
        return True


def box_frames(A: Arms, i: int, closing_axes, inward: np.ndarray):
    """(tilt, R) grasp frames: closing axis among `closing_axes` (horizontal unit vectors), approach -z tilted towards
    `inward` (horizontal) by 0 / 0.3 / 0.55 rad"""
    out = []
    for y in closing_axes:
        d = inward - (inward @ y) * y
        d[2] = 0
        if np.linalg.norm(d) < 1e-6:
            d = np.cross(y, UP)
        d /= np.linalg.norm(d) + 1e-12
        for tilt in TILTS:
            out.append((tilt, frame(y, -UP * np.cos(tilt) + d * np.sin(tilt))))
    return out


def grasp_z(A: Arms, i: int, tilt: float, hz: float) -> float:
    """TCP height for a top-down table grasp: fingertips >= 4 mm above the table, never above the object centre"""
    return max(hz, A.ext[i] * np.cos(tilt) + 0.015 * np.sin(tilt) + 0.004)


def face_axes(T: np.ndarray) -> list:
    """the two horizontal face normals of a box (both signs)"""
    out = []
    for c in (0, 1):
        a = T[:3, c].copy(); a[2] = 0
        n = np.linalg.norm(a)
        if n > 0.5:
            a /= n
            out += [a, -a]
    return out


def cube_candidates(A: Arms, i: int, cube, half: float, place_xy, place_cz: float, max_n: int = 6):
    """IK-feasible pick-and-place candidates for a cube (arm i): list of dict(R, grasp, pre, lift, above, place, retreat,
    h, tilt), best first (low tilt, high waypoints, small rotation from the current TCP orientation)"""
    T = actor_T(cube)
    c = T[:3, 3]
    base = A.bases[i].p
    radial = np.array([c[0] - base[0], c[1] - base[1], 0.0]); radial /= np.linalg.norm(radial) + 1e-9
    R_now = A.tcp_T(i)[:3, :3]
    cands = []
    for y in face_axes(T):
        d = np.cross(y, UP)
        if d @ radial < 0:
            d = -d
        for tilt in TILTS:
            R = frame(y, -UP * np.cos(tilt) + d * np.sin(tilt))
            gz = grasp_z(A, i, tilt, half)
            g = np.array([c[0], c[1], gz])
            tcp_place_z = place_cz + (gz - c[2])
            for h in HEIGHTS:
                P = np.array([place_xy[0], place_xy[1], tcp_place_z])
                wp = dict(R=R, grasp=g, pre=g - R[:, 2] * h, lift=g + UP * h, above=P + UP * h, place=P,
                          retreat=P + UP * h, h=h, tilt=tilt)
                score = tilt * 2.0 + (0.10 - h) * 5.0 + rot_angle(R_now, R) * 0.3
                cands.append((score, wp))
    cands.sort(key=lambda s: s[0])
    out = []
    for _, wp in cands:
        if A.ik_chain(i, [wp[k] for k in ("pre", "grasp", "lift", "above", "place", "retreat")], wp["R"]):
            out.append(wp)
            if len(out) >= max_n:
                break
    return out


def pick_place_cube(A: Arms, i: int, cube, half: float, place_xy, place_cz: float, mode: str, log: dict, tag: str,
                    retreat_home: bool = True) -> bool:
    """arm i picks `cube` and puts its centre at (place_xy, place_cz); mode: 'release' (open, retreat, home), 'hold'
    (stays closed), 'cage' (opened to ~1.2 mm beyond the cube width). The other arm is FREE (idle) throughout."""
    ex = A.ex
    A.set(1 - i, FREE)
    A.obstacles(i)
    cands = cube_candidates(A, i, cube, half, place_xy, place_cz)
    log[f"{tag}_candidates"] = len(cands)
    if not cands:
        A.failed.append(f"{tag}: no IK-feasible candidate")
        return False
    for wp in cands:
        R = wp["R"]
        c0 = actor_T(cube)[:3, 3].copy()
        if not A.go(i, pose(wp["pre"], R), "rrt", FREE, name=f"{tag} pre"):
            continue
        if not A.go(i, pose(wp["grasp"], R), "line", CONTACT, name=f"{tag} grasp"):
            continue
        A.grip(i, 1.0, n=25)
        A.p[i].attach_box([2 * half - 0.012] * 3, sapien.Pose(np.linalg.inv(R) @ (c0 - wp["grasp"])))
        if not A.go(i, pose(wp["lift"], R), "line", CONTACT, attach=True, name=f"{tag} lift"):
            return False
        # aim the cube (not the TCP) at the place point: privileged in-hand offset after the grasp
        d = actor_T(cube)[:3, 3] - A.tcp_T(i)[:3, 3]
        nominal = c0 - wp["grasp"]
        corr = (d - nominal)[:2]
        if np.linalg.norm(corr) < 0.06:
            for k in ("above", "place", "retreat"):
                wp[k] = wp[k].copy(); wp[k][:2] -= corr
        if not A.go(i, pose(wp["above"], R), "screw", CONTACT, attach=True, name=f"{tag} above"):
            return False
        # re-aim from the actual cube position above the place point (removes steady-state tracking error / sag)
        err = actor_T(cube)[:3, 3][:2] - np.asarray(place_xy, float)
        if np.linalg.norm(err) < 0.04:
            for k in ("place", "retreat"):
                wp[k] = wp[k].copy(); wp[k][:2] -= err
            log[f"{tag}_reaim_m"] = float(np.linalg.norm(err))
        if not A.go(i, pose(wp["place"], R), "line", CONTACT, attach=True, name=f"{tag} place"):
            return False
        A.wait(5, *((CONTACT, FREE) if i == 0 else (FREE, CONTACT)))
        log[f"{tag}_tilt"], log[f"{tag}_h"] = wp["tilt"], wp["h"]
        if mode == "release":
            A.grip(i, 0.0, n=20)
            A.p[i].detach()
            return leave(A, i, R, wp, tag, log, retreat_home)
        elif mode == "cage":
            A.set(i, CONTACT)
            cage(ex, i, half)
        else:
            A.wait(10, *((CONTACT, FREE) if i == 0 else (FREE, CONTACT)))
        return True
    return False


def leave(A: Arms, i: int, R, wp: dict, tag: str, log: dict, home: bool = True) -> bool:
    """after a release: back off along the approach axis (-z of the TCP) by 8 / 5 / 3 cm, else straight up to the
    retreat waypoint; then (home=True) go home. If no straight retreat is feasible, plan home directly (RRT, the
    gripper open). Returns False only if the arm could neither retreat nor go home."""
    T = A.tcp_T(i)
    for d in (0.08, 0.05, 0.03):
        A.set(i, FREE)
        if _go(A.ex, i, A.p[i], pose(T[:3, 3] - T[:3, 2] * d, T[:3, :3]), "line", fallback=False) == "ok":
            log[f"{tag}_backoff_m"] = d
            break
    else:
        if not A.go(i, pose(wp["retreat"], R), "line", FREE, name=f"{tag} retreat"):
            log[f"{tag}_retreat"] = "failed, planning home directly"
            return A.home(i) or (A.failed.append(f"{tag} home: failed") or False)
    if home and not A.home(i):
        log[f"{tag}_home"] = "failed"
    return True


def result(A: Arms, ok: bool, variant: str, log: dict) -> dict:
    log.update(variant=variant, steps=A.ex.n, failed=list(A.failed))
    log["status"] = "ok" if ok else ("plan_failed: " + "; ".join(A.failed[-3:]) if A.failed else "failed")
    log["final_targets"] = [t.tolist() for t in A.ex.targets]
    return log
