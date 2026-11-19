"""Scripted dual-arm handover (DualHandover-v1) with mplib, any parallel-gripper dual configuration, plus the contract's
counterexamples.

run(env, variant) with variant in
  handover       left picks the block's left half from the table (top-down / tilted grasp found by an IK search), lifts and
                 carries it (block translated, orientation kept) to the handover pose (centre at x = 0.30, y = 0, bottom
                 10 cm above the table); the right arm plans to a pre-grasp above the block's right half (RRT, the left arm
                 and the block as point-cloud obstacles), descends, closes; the left opens, retreats along its approach axis
                 and goes home; the right carries the block over the target (privileged pose re-read: the block centre,
                 not the TCP, is aimed at the target), lowers it to 6 mm above the table, opens, retreats, goes home.
  no_handover    the right arm reaches over and picks the block's right half from the left zone itself, places it in the
                 target, opens, retreats, home; the left arm never moves.
  both_hold      handover grasp as above but the left never opens; both TCPs then follow the same Cartesian displacement
                 (block over the target, then down onto the table); neither gripper opens.
  drop_and_push  left lifts the block, carries it near the centre (E29b: to a point from which the right arm's push is
                 plannable) and opens 3-5 cm above the table (the block drops onto the table); the right
                 arm, gripper closed, pushes the block along the table into the target, opens and retreats.
  table_transfer (E29b) left puts the block down on the table near the centre, opens, leaves; the right arm picks it from
                 the table, places it in the target, opens, retreats (transfer through the table, no mid-air handover).
E29b changes: phase annotation (Planners.stage: FREE / CONTACT per arm before every stage); in-air regrasp search with
approach tilt 0.8 rad and a 3 cm pre-grasp; re-presentation loop (present_and_regrasp); both_hold sets the block down on
the table (2 mm press) and uses the largest reachable fraction of the carry when one arm cannot reach the target;
push starts at fingertip heights 8 / 15 / 22 mm with tilts up to 0.8 rad; new counterexample table_transfer.
Grasp candidates: closing axis = the block's short horizontal axis (both signs), approach tilt 0 / 0.3 / 0.55 rad towards
the block centre (the gripper leans away from the other grasp), pre-grasp / lift heights 10 / 7 / 5 cm; every waypoint
must be IK-feasible (mplib IK with the other arm as a point cloud) before the candidate is executed.
"""
from __future__ import annotations

import numpy as np
import sapien

from .mplib_planner import ArmPlanner, arm_points
from .scripted import UP, DualExecutor, _go, frame, pose, rot_angle, tcp_extent
from ..data.phase import CONTACT, FREE, set_phase
from ..verify.v31 import box_surface_points

HANDOVER_X = 0.30
HANDOVER_BOTTOMS = (0.10, 0.07, 0.05)   # block bottom above the table at the handover, first reachable [m]
GRASP_OFFSET = 0.05               # grasp point distance from the block centre along its long axis [m]
TILTS = (0.0, 0.3, 0.55)
AIR_TILTS = (0.0, 0.3, 0.55, 0.8)          # in-air regrasp (E29b: small arms need steeper approaches at the handover)
AIR_HS = (0.10, 0.07, 0.05, 0.03)          # in-air pre-grasp distances along the approach axis (E29b: + 3 cm)
# E29b re-presentation: if the right arm has no grasp at the handover pose, the left arm moves the block by
# (dx, dy, dz [m], dyaw [deg about the vertical through the block centre]) and the right arm tries again
PRESENT = ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, -0.03, 0.0), (0.0, -0.03, -0.03, 0.0), (0.0, 0.0, -0.05, 0.0),
           (-0.04, 0.0, -0.03, 0.0), (0.0, 0.0, -0.03, 15.0), (0.0, 0.0, -0.03, -15.0), (0.03, -0.03, -0.03, 0.0),
           (0.0, -0.05, -0.05, 0.0))
MIN_PRESENT_BOTTOM = 0.035                 # block bottom above the table during a re-presentation [m]
REGRASP_OFFSET = GRASP_OFFSET              # right arm's in-air grasp point from the block centre [m]
REGRASP_DZ = None                          # right arm's in-air TCP height above the block centre [m]; None (E29b
                                           # default): the table grasp's depth, fingertips 4 mm above the block bottom
                                           # (E23: 0 = TCP at the centre; the Robotiq then let the 72 g block pivot
                                           # 30-42 deg after the left released, probe regrasp_hold_probe_*.log)


def block_T(u) -> np.ndarray:
    return u.block.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)


def block_points(u, n: int = 7) -> np.ndarray:
    T = block_T(u)
    return box_surface_points(u.block_half, n) @ T[:3, :3].T + T[:3, 3]


def long_axis(T) -> np.ndarray:
    a = T[:3, 1].copy(); a[2] = 0
    return a / (np.linalg.norm(a) + 1e-12)


def short_axis(T) -> np.ndarray:
    a = T[:3, 0].copy(); a[2] = 0
    return a / (np.linalg.norm(a) + 1e-12)


def grasp_point(T, side: int, offset: float = GRASP_OFFSET) -> np.ndarray:
    """side +1 = the half with larger world y (left arm), -1 = smaller y (right arm)"""
    a = long_axis(T)
    if a[1] < 0:
        a = -a
    return T[:3, 3] + side * offset * a


class Planners:
    def __init__(self, env, ex: DualExecutor):
        u = env.unwrapped
        self.u, self.ex = u, ex
        self.bases = u.dual.mount_poses()
        self.p = [ArmPlanner(ag, b, control_dt=float(u.control_timestep)) for ag, b in zip(ex.agents, self.bases)]
        self.ext = [tcp_extent(ag) for ag in ex.agents]
        self.holding = [False, False]

    def stage(self, i: int, phase: str):
        """phase annotation (imitator.data.phase) before the env.step calls of a stage of arm i: arm i gets `phase`; the
        other arm is CONTACT while it holds the block, else FREE (idle, nothing grasped)"""
        set_phase(self.ex.env, phase, arm=i)
        set_phase(self.ex.env, CONTACT if self.holding[1 - i] else FREE, arm=1 - i)

    def obstacles(self, i: int, with_block: bool = False, radius: float = 0.01):
        pts = arm_points(self.ex.agents[1 - i])
        if with_block:
            pts = np.vstack([pts, block_points(self.u)])
        self.p[i].set_obstacle_points(pts, radius=radius)


def candidates(pl: Planners, i: int, g: np.ndarray, T_block: np.ndarray, inward: np.ndarray, key_points: dict,
               table_grasp: bool, tilts=TILTS, hs=(0.10, 0.07, 0.05), air_bottom: float | None = None):
    """yield (score, R, info) for IK-feasible grasp frames. key_points: name -> function(R, h, g_eff) -> TCP position"""
    planner, ag = pl.p[i], pl.ex.agents[i]
    R_now = planner.fk_tcp(planner.qpos())[:3, :3]
    hz = float(pl.u.block_half[2])
    out = []
    for sgn in (1.0, -1.0):
        y = sgn * short_axis(T_block)
        d = inward - (inward @ y) * y; d[2] = 0; d /= np.linalg.norm(d) + 1e-12
        for tilt in tilts:
            z = -UP * np.cos(tilt) + d * np.sin(tilt)
            R = frame(y, z)
            ge = g.copy()
            if table_grasp:
                ge[2] = max(hz, pl.ext[i] * np.cos(tilt) + 0.015 * np.sin(tilt) + 0.004)
            elif air_bottom is not None:     # in-air grasp at the table grasp's depth (fingertips 4 mm above the bottom)
                ge[2] = max(g[2], air_bottom + pl.ext[i] * np.cos(tilt) + 0.015 * np.sin(tilt) + 0.004)
            for h in hs:
                score = tilt * 2.0 + (0.10 - h) * 5.0 + rot_angle(R_now, R) * 0.3
                out.append((score, R, dict(tilt=tilt, h=h, g=ge)))
    out.sort(key=lambda c: c[0])
    for score, R, info in out:
        ok, q_seed = True, None
        for name, fn in key_points.items():
            p = fn(R, info["h"], info["g"])
            good, sol = planner.ik(pose(p, R), q_start=q_seed, n_init=10)
            if not good and q_seed is not None:
                good, sol = planner.ik(pose(p, R), q_start=None, n_init=10)
            if not good:
                ok = False
                break
            q_seed = sol[0]
        if ok:
            yield score, R, info


def _attach_block(pl: Planners, i: int, shrink: float = 0.012):
    ag = pl.ex.agents[i]
    T_tcp = ag.tcp.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)
    T_rel = np.linalg.inv(T_tcp) @ block_T(pl.u)
    from transforms3d.quaternions import mat2quat
    pl.p[i].attach_box(2 * np.asarray(pl.u.block_half) - shrink, sapien.Pose(T_rel[:3, 3], mat2quat(T_rel[:3, :3])))


def follow_both(ex: DualExecutor, paths: dict, settle: int = 3):
    n = max(len(p) for p in paths.values())
    for k in range(n):
        for i, P in paths.items():
            ex.targets[i][:ex.agents[i].record.dof_arm] = P[min(k, len(P) - 1)]
        ex.step()
    ex.step(settle)


def go_home(pl: Planners, i: int):
    ag = pl.ex.agents[i]
    pl.obstacles(i)
    res = pl.p[i].plan_qpos(np.asarray(ag.record.home_arm_qpos, float))
    if res["status"] == "Success":
        pl.stage(i, FREE)
        pl.ex.follow(i, res["position"])
        return "ok"
    return res["status"]


def tcp_T(ag) -> np.ndarray:
    return ag.tcp.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)


# ------------------------------------------------------------------------------------------------ phases
def pick(pl: Planners, i: int, side: int, dests, log: dict, tag: str, max_candidates: int = 6):
    """arm i grasps the block half `side` on the table and carries the block (translated) so that its centre reaches
    the first reachable of `dests` (list of centres, tried in order; e.g. handover heights 10 / 7 / 5 cm). Every
    waypoint (pre-grasp, grasp, lift, 3 cm above the destination, destination) is IK-checked before execution."""
    u, ex = pl.u, pl.ex
    T0 = block_T(u)
    c0 = T0[:3, 3].copy()
    g = grasp_point(T0, side)
    inward = c0 - g; inward[2] = 0
    pl.obstacles(i)
    dests = [np.asarray(d, float) for d in (dests if isinstance(dests, (list, tuple)) else [dests])]
    plan = None
    for dest in dests:
        kp = dict(pre=lambda R, h, ge: ge - R[:, 2] * h, grasp=lambda R, h, ge: ge, lift=lambda R, h, ge: ge + UP * h,
                  dest_above=lambda R, h, ge, dest=dest: ge + (dest - c0) + UP * 0.03,
                  dest=lambda R, h, ge, dest=dest: ge + (dest - c0))
        cands = []
        for c in candidates(pl, i, g, T0, inward, kp, table_grasp=True, tilts=AIR_TILTS):
            cands.append(c)
            if len(cands) >= max_candidates:
                break
        if cands:
            plan = (dest, cands)
            break
    if plan is None:
        return "no_candidate"
    dest_center, cands = plan
    log[f"{tag}_dest_z"] = float(dest_center[2])
    st = "no_candidate"
    for n, (score, R, info) in enumerate(cands):
        ge, h = info["g"], info["h"]
        pl.stage(i, FREE)
        st = _go(ex, i, pl.p[i], pose(ge - R[:, 2] * h, R), "rrt")
        if st != "ok":
            st = f"{tag} pre: {st}"; continue
        pl.stage(i, CONTACT)
        st = _go(ex, i, pl.p[i], pose(ge, R), "line")
        if st != "ok":
            st = f"{tag} grasp: {st}"; continue
        ex.set_gripper(i, 1.0, n=25)
        pl.holding[i] = True
        _attach_block(pl, i)
        st = _go(ex, i, pl.p[i], pose(ge + UP * h, R), "line", attach=True)
        if st != "ok":
            st = f"{tag} lift: {st}"; break
        # aim the block centre (privileged pose; the block may have shifted while closing); if the carry to the
        # chosen destination cannot be planned, try the remaining (lower) destinations from the lifted state
        d = block_T(u)[:3, 3] - tcp_T(ex.agents[i])[:3, 3]
        for dc in [dest_center] + [x for x in dests if x[2] < dest_center[2] - 1e-9]:
            tgt = dc - d
            st = _go(ex, i, pl.p[i], pose(tgt + UP * 0.03, R), "screw", attach=True)
            if st != "ok":
                st = f"{tag} carry: {st}"; continue
            st = _go(ex, i, pl.p[i], pose(tgt, R), "line", attach=True)
            if st != "ok":
                st = f"{tag} carry2: {st}"; continue
            log[f"{tag}_dest_z"] = float(dc[2])
            break
        if st != "ok":
            break
        ex.step(5)
        log[f"{tag}_tilt"], log[f"{tag}_h"] = info["tilt"], h
        st = "ok"
        break
    return st


def regrasp_in_air(pl: Planners, i: int, side: int, log: dict, tag: str, max_candidates: int = 6, place=None,
                   offset: float | None = None, dz: float | None = None):
    """arm i grasps the block half `side` while the block is held in the air by the other arm; with `place` (block
    centre at the target) the candidate must also reach the place pose and 5 cm above it with the same orientation.
    offset: grasp point distance from the block centre along its long axis (default REGRASP_OFFSET); dz: TCP height
    above the block centre (default REGRASP_DZ)."""
    u, ex = pl.u, pl.ex
    T0 = block_T(u)
    c0 = T0[:3, 3].copy()
    g = grasp_point(T0, side, REGRASP_OFFSET if offset is None else offset)
    dz = REGRASP_DZ if dz is None else dz
    air_bottom = None
    if dz is None:      # E29b default: the same grasp depth as on the table (fingertips 4 mm above the block bottom)
        air_bottom = float(c0[2] - np.abs(T0[2, :3]) @ np.asarray(u.block_half))
    else:
        g[2] += dz
    inward = c0 - g; inward[2] = 0
    pl.obstacles(i, with_block=False)            # IK of the grasp pose must not see the block it is grasping
    kp = dict(pre=lambda R, h, ge: ge - R[:, 2] * h, grasp=lambda R, h, ge: ge)
    if place is not None:
        P = np.asarray(place, float)
        kp["place_above"] = lambda R, h, ge: ge + (P - c0) + UP * 0.05
        kp["place"] = lambda R, h, ge: ge + (P - c0)
    cands = []
    for score, R, info in candidates(pl, i, g, T0, inward, kp, table_grasp=False, tilts=AIR_TILTS, hs=AIR_HS, air_bottom=air_bottom):
        cands.append((score, R, info))
        if len(cands) >= max_candidates:
            break
    if not cands:                                # grippers are close at the handover: 3 mm obstacle radius
        log[f"{tag}_fine_cloud"] = True
        pl.obstacles(i, with_block=False, radius=0.003)
        for score, R, info in candidates(pl, i, g, T0, inward, kp, table_grasp=False, tilts=AIR_TILTS, hs=AIR_HS, air_bottom=air_bottom):
            cands.append((score, R, info))
            if len(cands) >= max_candidates:
                break
    if not cands and place is not None:          # no grasp that also reaches the place pose: grasp anyway
        log[f"{tag}_place_unchecked"] = True
        kp = dict(pre=kp["pre"], grasp=kp["grasp"])
        for score, R, info in candidates(pl, i, g, T0, inward, kp, table_grasp=False, tilts=AIR_TILTS, hs=AIR_HS, air_bottom=air_bottom):
            cands.append((score, R, info))
            if len(cands) >= max_candidates:
                break
    st = "no_candidate"
    for score, R, info in cands:
        ge, h = info["g"], info["h"]
        pl.obstacles(i, with_block=True)         # ... but the path to the pre-grasp must avoid it
        pl.stage(i, FREE)
        st = _go(ex, i, pl.p[i], pose(ge - R[:, 2] * h, R), "rrt")
        if st != "ok":
            st = f"{tag} pre: {st}"; continue
        pl.stage(i, CONTACT)
        st = _go(ex, i, pl.p[i], pose(ge, R), "line")
        if st != "ok":
            st = f"{tag} grasp: {st}"; continue
        ex.set_gripper(i, 1.0, n=25)
        pl.holding[i] = True
        log[f"{tag}_tilt"], log[f"{tag}_h"] = info["tilt"], h
        return "ok"
    return st


def _rotz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def present_and_regrasp(pl: Planners, log: dict, place=None, tag: str = "R_grasp") -> str:
    """E29b: the right arm grasps the block's right half in the air (regrasp_in_air); if it finds no executable grasp,
    the left arm (still holding) re-presents the block (PRESENT offsets / yaw turns about the block's vertical axis, from
    the first handover pose) and the right arm tries again. A right arm that moved without grasping goes home first."""
    u, ex = pl.u, pl.ex
    hz = float(u.block_half[2])
    T_ref = tcp_T(ex.agents[0])
    c_ref = block_T(u)[:3, 3].copy()
    st = "no_candidate"
    tried = []
    for k, (dx, dy, dz, dyaw) in enumerate(PRESENT):
        if k > 0:
            c = c_ref + np.array([dx, dy, dz])
            if c[2] - hz < MIN_PRESENT_BOTTOM:
                continue
            Rz = _rotz(dyaw)
            p_tcp = c + Rz @ (T_ref[:3, 3] - c_ref)
            pl.stage(0, CONTACT)
            pl.obstacles(0)
            s2 = _go(ex, 0, pl.p[0], pose(p_tcp, Rz @ T_ref[:3, :3]), "screw", attach=True)
            if s2 != "ok":
                tried.append(f"{k}: present {s2}"[:60]); continue
            ex.step(5)
        n0 = ex.n
        st = regrasp_in_air(pl, 1, -1, log, tag, place=place)
        if st == "ok":
            log["present_k"] = k
            log["present_tried"] = tried
            return "ok"
        tried.append(f"{k}: {st}"[:60])
        if ex.n != n0:                                # the right arm moved without a grasp: clear the way
            go_home(pl, 1)
    log["present_tried"] = tried
    return st


def release_and_leave(pl: Planners, i: int, log: dict, tag: str, back: float = 0.08, home: bool = True):
    """open, back off along -z_tcp (8 / 5 / 3 cm; else straight up 8 / 5 cm), then home; if no straight retreat is
    feasible, plan to home directly (collision-aware RRT, gripper open)"""
    ex = pl.ex
    pl.stage(i, CONTACT)
    ex.set_gripper(i, 0.0, n=20)
    pl.holding[i] = False
    pl.stage(i, FREE)
    T = tcp_T(ex.agents[i])
    st = "?"
    for d in (back, 0.05, 0.03):
        st = _go(ex, i, pl.p[i], pose(T[:3, 3] - T[:3, 2] * d, T[:3, :3]), "line", fallback=False)
        if st == "ok":
            break
    if st != "ok":
        for d in (back, 0.05):
            st = _go(ex, i, pl.p[i], pose(T[:3, 3] + UP * d, T[:3, :3]), "line", fallback=False)
            if st == "ok":
                break
    if st != "ok":
        log[f"{tag}_retreat"] = st
        return "ok" if go_home(pl, i) == "ok" else f"{tag} retreat: {st}"
    if home:
        go_home(pl, i)
    return "ok"


def carry_to(pl: Planners, i: int, dest_center: np.ndarray, log: dict, tag: str):
    """arm i (holding the block alone) brings the block centre to dest_center via a point 5 cm above it. E29b: if the
    transport is not plannable with the grasp orientation, the hand is turned about the vertical (+-20 / +-40 deg; only
    the block centre has to be in the square) and the transport height lowered to 3 cm (logged)."""
    u, ex = pl.u, pl.ex
    pl.stage(i, CONTACT)
    pl.p[i].detach()
    _attach_block(pl, i)
    pl.obstacles(i)
    T = tcp_T(ex.agents[i])
    d0 = block_T(u)[:3, 3] - T[:3, 3]
    st = "?"
    R = T[:3, :3]
    for dyaw in (0.0, 20.0, -20.0, 40.0, -40.0):
        Rz = _rotz(dyaw)
        R = Rz @ T[:3, :3]
        tgt = dest_center - Rz @ d0
        for up in (0.05, 0.03):
            st = _go(ex, i, pl.p[i], pose(tgt + UP * up, R), "screw", attach=True)
            if st == "ok":
                break
        if st == "ok":
            if dyaw:
                log[f"{tag}_yaw_turn"] = dyaw
            break
    if st != "ok":
        return f"{tag} transport: {st}"
    # re-aim once more from the actual pose before lowering
    d = block_T(u)[:3, 3] - tcp_T(ex.agents[i])[:3, 3]
    tgt = dest_center - d
    st = _go(ex, i, pl.p[i], pose(tgt, R), "line", attach=True)
    if st != "ok":
        return f"{tag} lower: {st}"
    ex.step(5)
    return "ok"


def push_plan(pl: Planners, i: int, c: np.ndarray, Rb: np.ndarray, dest_center: np.ndarray, pre_check: bool = True):
    """push poses for arm i to push a block at centre c (rotation Rb) along the table to dest_center: (R, start, end,
    pre-push heights, info) or None. E29b: fingertip heights 8 / 15 / 22 / 30 mm (the 40 mm tall, 18 cm long block
    slides long before it can tip about its front edge when pushed on its end face), approach tilts up to 0.8 rad, and the pre-push pose above the start must be
    IK-feasible with the block as an obstacle (small arms such as ARX X5 have little low-height workspace)."""
    u = pl.u
    v = dest_center - c; v[2] = 0
    v /= float(np.linalg.norm(v)) + 1e-12
    ext_v = float(np.abs(Rb.T @ v) @ np.asarray(u.block_half))
    base = pl.bases[i].p
    pts = box_surface_points(u.block_half, 7) @ Rb.T + np.array([c[0], c[1], float(u.block_half[2])])
    pl.obstacles(i, with_block=False)            # IK of the push poses must not see the block it pushes
    try:
        for zf, back in ((0.008, 0.035), (0.015, 0.035), (0.022, 0.035), (0.03, 0.035), (0.022, 0.02), (0.03, 0.02)):
            start = c - v * (ext_v + back); start[2] = 0.0
            end = dest_center - v * (ext_v + 0.012); end[2] = 0.0
            radial = np.array([start[0] - base[0], start[1] - base[1], 0.0]); radial /= np.linalg.norm(radial) + 1e-12
            for y in (np.cross(UP, v), v):                            # closing axis across / along the push direction
                for dirv in (radial, -v, v):
                    dd = dirv - (dirv @ y) * y; dd[2] = 0
                    if np.linalg.norm(dd) < 1e-6:
                        continue
                    dd /= np.linalg.norm(dd)
                    for tilt in AIR_TILTS:
                        Rt = frame(y, -UP * np.cos(tilt) + dd * np.sin(tilt))
                        z_tcp = pl.ext[i] * np.cos(tilt) + 0.015 * np.sin(tilt) + zf
                        s_, e_ = start.copy(), end.copy(); s_[2] = z_tcp; e_[2] = z_tcp
                        if not (pl.p[i].ik(pose(s_, Rt))[0] and pl.p[i].ik(pose(e_, Rt))[0]):
                            continue
                        if not pre_check:
                            return Rt, s_, e_, (0.08, 0.05, 0.03), dict(fingertip_z=zf, tilt=tilt)
                        pl.p[i].set_obstacle_points(np.vstack([arm_points(pl.ex.agents[1 - i]), pts]), radius=0.01)
                        hp = next((hh for hh in (0.08, 0.05, 0.03, 0.02) if pl.p[i].ik(pose(s_ + UP * hh, Rt))[0]), None)
                        pl.obstacles(i, with_block=False)
                        if hp is not None:
                            return Rt, s_, e_, (hp,), dict(fingertip_z=zf, tilt=tilt, hpre=hp)
        return None
    finally:
        pl.obstacles(i, with_block=False)


def push_to(pl: Planners, i: int, dest_center: np.ndarray, log: dict, tag: str, speed: float = 0.05):
    """arm i, gripper closed, pushes the block along the table so that its centre ends at dest_center"""
    u, ex = pl.u, pl.ex
    pl.stage(i, FREE)
    ex.set_gripper(i, 1.0, n=15)
    T0 = block_T(u)
    c = T0[:3, 3].copy()
    plan = push_plan(pl, i, c, T0[:3, :3], dest_center)
    found = plan is not None
    if found:
        R, start, end, hpres, info = plan
        log.update({f"{tag}_{k}": v for k, v in info.items()})
    else:
        v = dest_center - c; v[2] = 0; v /= float(np.linalg.norm(v)) + 1e-12
        ext_v = float(np.abs(T0[:3, :3].T @ v) @ np.asarray(u.block_half))
        start = c - v * (ext_v + 0.035); end = dest_center - v * (ext_v + 0.012)
        start[2] = end[2] = pl.ext[i] + 0.008
        R, hpres = frame(np.cross(UP, v), -UP), (0.08, 0.05, 0.03)
    log[f"{tag}_orientation_found"] = found
    pl.obstacles(i, with_block=True)             # the approach to the pre-push pose must avoid the block
    st = "?"
    for hpre in hpres:
        st = _go(ex, i, pl.p[i], pose(start + UP * hpre, R), "rrt")
        if st == "ok":
            break
    if st != "ok":
        return f"{tag} pre: {st}"
    pl.stage(i, CONTACT)
    st = _go(ex, i, pl.p[i], pose(start, R), "line")
    if st != "ok":
        return f"{tag} down: {st}"
    res = pl.p[i].cartesian(pose(end, R), speed=speed)
    if res["status"] != "Success":
        res = pl.p[i].joint_interp(pose(end, R))
    if res["status"] != "Success":
        return f"{tag} push: {res['status']}"
    ex.follow(i, res["position"], settle=5)
    ex.set_gripper(i, 0.0, n=20)
    pl.stage(i, FREE)
    T = tcp_T(ex.agents[i])
    st = _go(ex, i, pl.p[i], pose(T[:3, 3] + UP * 0.08, T[:3, :3]), "line")
    if st != "ok":
        return f"{tag} retreat: {st}"
    go_home(pl, i)
    return "ok"


def both_carry(pl: Planners, dest_center: np.ndarray, log: dict, tag: str = "both"):
    """both arms (both holding) move their TCPs by the same displacement: block centre over dest + 4 cm, then down until
    the block's bottom presses 2 mm into the table (E29b: set it DOWN, E23 stopped it 6 mm above the table). If one arm
    cannot follow the full horizontal displacement (ARX X5's left arm cannot reach the right target), the largest
    reachable fraction (1.0, 0.85, ..., 0.1, 0) of it is used (logged as both_frac; 0 = set down where it is)."""
    u, ex = pl.u, pl.ex
    set_phase(ex.env, CONTACT, arm=0); set_phase(ex.env, CONTACT, arm=1)
    c = block_T(u)[:3, 3]
    hz = float(u.block_half[2])
    full = dest_center + UP * 0.04 - c
    for frac in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0):
        delta = full.copy(); delta[:2] *= frac
        paths, why = {}, None
        for i in (0, 1):
            T = tcp_T(ex.agents[i])
            res = pl.p[i].cartesian(pose(T[:3, 3] + delta, T[:3, :3]), speed=0.08)
            if res["status"] != "Success":
                res = pl.p[i].joint_interp(pose(T[:3, 3] + delta, T[:3, :3]))
            if res["status"] != "Success":
                why = f"{tag} arm {i}: {res['status']}"; break
            paths[i] = res["position"]
        if why is None:
            break
    if why is not None:
        return why
    log["both_frac"] = frac
    follow_both(ex, paths, settle=5)
    bottom = float(u.cube_bottom_z()[0])
    down = -UP * (bottom + 0.002)
    paths = {}
    for i in (0, 1):
        T = tcp_T(ex.agents[i])
        res = pl.p[i].cartesian(pose(T[:3, 3] + down, T[:3, :3]), speed=0.05)
        if res["status"] != "Success":
            res = pl.p[i].joint_interp(pose(T[:3, 3] + down, T[:3, :3]))
        if res["status"] != "Success":
            return f"{tag} down arm {i}: {res['status']}"
        paths[i] = res["position"]
    follow_both(ex, paths, settle=5)
    return "ok"


def feasible_handover(pl: Planners, dests, place=None) -> list:
    """handover centres (in order) at which the RIGHT arm has an IK-feasible grasp of the block's right half (and, with
    `place`, can also reach the place pose with that orientation); the block keeps its current orientation"""
    T0 = block_T(pl.u)
    out = []
    pl.obstacles(1)
    for dest in dests:
        Th = T0.copy(); Th[:3, 3] = dest
        g = grasp_point(Th, -1)
        inward = dest - g; inward[2] = 0
        kp = dict(pre=lambda R, h, ge: ge - R[:, 2] * h, grasp=lambda R, h, ge: ge)
        if place is not None:
            P = np.asarray(place, float)
            kp["place"] = lambda R, h, ge, dest=dest: ge + (P - dest)
        if next(candidates(pl, 1, g, Th, inward, kp, table_grasp=False, tilts=AIR_TILTS, hs=AIR_HS), None) is not None:
            out.append(dest)
    return out


# ------------------------------------------------------------------------------------------------ behaviours
def run(env, variant: str = "handover", verbose: bool = False) -> dict:
    u = env.unwrapped
    ex = DualExecutor(env)
    pl = Planners(env, ex)
    hz = float(u.block_half[2])
    Hs = [np.array([HANDOVER_X, 0.0, b + hz]) for b in HANDOVER_BOTTOMS]
    P = u.target_pos[0].cpu().numpy().astype(float).copy(); P[2] = hz + 0.006
    log = dict(variant=variant, status="?")
    set_phase(env, FREE, arm=0); set_phase(env, FREE, arm=1)
    ex.step(1)
    if variant == "no_op":
        ex.step(20)
        log["status"] = "ok"
    elif variant == "table_transfer":
        # E29b counterexample: the left arm puts the block down on the table near the centre, opens and leaves; the right
        # arm picks it up from the table and places it in the target (a transfer through the table, no mid-air handover)
        mids = [np.array([HANDOVER_X, y, hz + 0.006]) for y in (-0.02, 0.0, -0.04, 0.02)]
        st = pick(pl, 0, +1, mids, log, "L_pick")
        if st == "ok":
            st = release_and_leave(pl, 0, log, "L_leave")
        if st == "ok":
            st = pick(pl, 1, -1, P + UP * 0.05, log, "R_pick")
        if st == "ok":
            st = carry_to(pl, 1, P, log, "R_place")
        if st == "ok":
            st = release_and_leave(pl, 1, log, "R_leave")
        log["status"] = st
    elif variant == "no_handover":
        st = pick(pl, 1, -1, P + UP * 0.05, log, "R_pick")
        if st == "ok":
            st = carry_to(pl, 1, P, log, "R_place")
        if st == "ok":
            st = release_and_leave(pl, 1, log, "R_leave")
        log["status"] = st
    else:
        ok_h = feasible_handover(pl, Hs, place=P) or feasible_handover(pl, Hs)
        log["handover_heights_feasible_for_right"] = [float(h[2]) for h in ok_h]
        drop = [np.array([HANDOVER_X, y, b + hz]) for y in (-0.06, -0.03) for b in (0.05, 0.04, 0.03)]   # near the centre
        if variant == "drop_and_push":
            # E29b: drop where the right arm can push from (ARX X5 cannot reach the push start of a block dropped at
            # y = -0.06 / -0.03 on most seeds): drop points whose push is plannable first
            Rb = block_T(u)[:3, :3]
            ys = (-0.06, -0.03, -0.08, -0.10)
            ok_y = [y for y in ys if push_plan(pl, 1, np.array([HANDOVER_X, y, hz]), Rb, P, pre_check=True) is not None]
            log["drop_y_pushable"] = ok_y
            ys = ok_y + [y for y in ys if y not in ok_y]
            drop = [np.array([HANDOVER_X, y, b + hz]) for y in ys for b in (0.03, 0.04, 0.05)]
        st = pick(pl, 0, +1, drop if variant == "drop_and_push" else (ok_h or Hs), log, "L_pick")
        if st == "ok" and variant == "drop_and_push":
            pl.stage(0, CONTACT)
            ex.set_gripper(0, 0.0, n=20)
            pl.holding[0] = False
            pl.stage(0, FREE)
            ex.step(20)                                       # the block falls and comes to rest
            T = tcp_T(ex.agents[0])
            _go(ex, 0, pl.p[0], pose(T[:3, 3] + UP * 0.06, T[:3, :3]), "line")
            go_home(pl, 0)
            st = push_to(pl, 1, P, log, "R_push")
        elif st == "ok":
            st = present_and_regrasp(pl, log, place=P)
            if st == "ok" and variant == "both_hold":
                st = both_carry(pl, P, log)
                ex.step(10)
            elif st == "ok":
                st = release_and_leave(pl, 0, log, "L_leave")
                if st == "ok":
                    st = carry_to(pl, 1, P, log, "R_place")
                if st == "ok":
                    st = release_and_leave(pl, 1, log, "R_leave")
        log["status"] = st
    log["steps"] = ex.n
    log["final_targets"] = [t.tolist() for t in ex.targets]
    return log
