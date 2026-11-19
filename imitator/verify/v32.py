"""Protocol v3.2 for imsim task contracts (E29b) = protocol v3 / v3.1 (hold A where the robot stopped, quiescence, 1 s
judged window, release by contact + enclosure, retreat aftermath) plus the fixes proposed by the E24c red team
(research/results/e24c_adversarial/E24C_REPORT.md section 7):

  1. object speeds from POSITIONS (PosSpeed): the speed of the fastest of the object's bounding-box corners, from the
     corner positions of two consecutive control steps; never PhysX's reported velocity (which reads 0.4-6 cm/s for a
     loaded object that does not move, E24c row A);
  2. a retreat aftermath for every task where release matters (generalised here to several objects and to the single
     ManiSkill Panda): every arm whose gripper is within near_m of a judged object backs its TCP off by retreat_m along
     its approach axis with its last gripper command, holds, and the objects must not move;
  3. commanded vs actual gripper opening (GripCmdGap, signal grip_cmd_gap_m): for every arm whose gripper is within
     near_m of a judged object, measured opening minus the free-space opening its current gripper PD target would give;
     a gripper that is still trying to close on something (or is propped open) shows a positive gap;
  4. a 5 s settle cap (contract settle.max_seconds = 5.0; E24c fix 4).

Nothing in veritask is changed. The recorder is veritask's TraceRecorderV3; the verdict is veritask's
protocol_v3.evaluate_traces_v3 (through imitator.verify.v31.judge_traces), the aftermath value enters as an extra
measurement under the metric name 'aftermath.retreat_object_disp' (registered by v31).

Embodiments: imsim dual configurations (ArmRecord agents) and the stock ManiSkill Panda (single arm, PandaView below).

    rec = build_recorder(env, objects={"cube": Obj("cube", half=0.02)}, release=["cube"])
    log = expert(rec)
    j = judge(rec, contract)                 # settle (hold A) + aftermath + verdict
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import v31

NOT_ENCLOSED_M = v31.NOT_ENCLOSED_M
RETREAT_METRIC = v31.RETREAT_METRIC
AFT_PREFIX = v31.AFT_PREFIX
HERE = os.path.dirname(os.path.abspath(__file__))
GRIP_CALIB_PATH = os.path.join(HERE, "grip_calib.json")
SETTLE_V32 = dict(hold="A", max_seconds=5.0, window_seconds=1.0, quiet_steps=10)
NEAR_M = 0.03


# ================================================================================================ objects
@dataclass
class Obj:
    """a judged object: env attribute `attr` (Actor), bounding box in the actor frame (half extents + centre offset);
    half=None: computed from the collision meshes at the first reset (axis-aligned box in the actor frame)"""
    attr: str
    half: object = None             # float | (3,) | callable(u) -> (3,) | None
    center: tuple = (0.0, 0.0, 0.0)
    enclosure: bool = True          # record an enclosure signal (release objects)
    tilt_axis: int | None = None    # <name>_tilt_deg: angle of this local axis to vertical (None: min over the axes)

    def actor(self, u):
        return getattr(u, self.attr)

    def box(self, u):
        """(half3, centre offset) in the actor frame; `half` may be a callable u -> half extents (objects whose size is
        re-randomised at reconfiguration, e.g. ManiSkill's PegInsertionSide peg)"""
        if callable(self.half):
            return v31.half3(self.half(u)), np.asarray(self.center, float)
        if self.half is None:
            from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
            V = np.vstack([np.asarray(m.vertices, float) for m in get_component_meshes(self.actor(u)._bodies[0])])
            lo, hi = V.min(0), V.max(0)
            self.half = ((hi - lo) / 2).tolist()
            self.center = tuple(((hi + lo) / 2).tolist())
        return v31.half3(self.half), np.asarray(self.center, float)

    def T(self, u, env_idx: int = 0) -> np.ndarray:
        """world transform of the BOX (actor pose x centre offset)"""
        T = fast_T(self.actor(u), env_idx)
        _, c = self.box(u)
        if np.any(c):
            T = T.copy(); T[:3, 3] = T[:3, :3] @ c + T[:3, 3]
        return T

    def corners(self, u, env_idx: int = 0) -> np.ndarray:
        h, _ = self.box(u)
        s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * h
        T = self.T(u, env_idx)
        return s @ T[:3, :3].T + T[:3, 3]


def table_top_z(u) -> float:
    return 0.0          # imsim (dual.TABLE_TOP_Z) and ManiSkill's TableSceneBuilder both put the table top at z = 0


def table_actor(u):
    t = getattr(u, "table", None)
    for sb in ("table_scene", "scene_builder"):          # ManiSkill TableSceneBuilder under either attribute name
        if t is None and getattr(getattr(u, sb, None), "table", None) is not None:
            t = getattr(u, sb).table
    return t


# ================================================================================================ fast pose reads
def fast_T(struct, env_idx: int = 0) -> np.ndarray:
    """4x4 world transform of a ManiSkill Actor / Link for env `env_idx`. CPU physics (one SAPIEN object behind the
    struct): read the SAPIEN entity pose directly (identical values, ~300-500x faster than ManiSkill's torch
    Pose.to_transformation_matrix; added by E29b fork B1 for per-step probes). GPU physics: the torch path."""
    objs = getattr(struct, "_objs", None)
    sc = getattr(struct, "scene", None)
    if objs is not None and len(objs) == 1 and env_idx == 0 and sc is not None and not getattr(sc, "gpu_sim_enabled", True):
        o = objs[0]
        p = o.entity_pose if hasattr(o, "entity_pose") else o.pose
        return np.asarray(p.to_transformation_matrix(), float)
    return struct.pose.to_transformation_matrix()[env_idx].cpu().numpy().astype(float)


def fast_link_transforms(ag, names, env_idx: int = 0) -> dict:
    """v31.link_transforms with fast_T"""
    lm = ag.robot.links_map
    return {n: fast_T(lm[n], env_idx) for n in names if n in lm}


# ================================================================================================ arm views
def agents_of(u) -> list:
    return list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent]


class ImsimView:
    """an imsim ArmRecord agent (dual configurations)"""

    def __init__(self, ag):
        self.ag, self.r = ag, ag.record
        self.kind = self.r.kind

    def gripper_points(self, env_idx: int = 0) -> np.ndarray:
        names = v31.gripper_links(self.r)
        W = v31._world(self.r, fast_link_transforms(self.ag, names, env_idx), names)
        return np.vstack(list(W.values())) if W else np.zeros((0, 3))

    def tcp_T(self, env_idx: int = 0) -> np.ndarray:
        return fast_T(self.ag.tcp, env_idx)

    def reach(self) -> float:
        return v31.gripper_reach(self.r)

    def enclosure(self, Tb, half, env_idx: int = 0) -> float | None:
        Ts = fast_link_transforms(self.ag, v31.needed_links(self.ag), env_idx)
        v = v31.enclosure_all_value([self.ag], [Ts], Tb, half)
        return None if v >= NOT_ENCLOSED_M else v

    def opening(self, env_idx: int = 0) -> float:
        """registry.measured_opening; on CPU physics the same pad_geometry computation in numpy on fast_T poses"""
        sc = getattr(self.ag.robot, "scene", None)
        if self.r.pad_links and env_idx == 0 and sc is not None and not getattr(sc, "gpu_sim_enabled", True):
            names = sorted({n for side in self.r.sides for n in side})
            return float(v31.reg.pad_geometry(self.r, fast_link_transforms(self.ag, names, env_idx))["opening"])
        return float(self.ag.measured_opening()[env_idx])

    def grip_target(self, env_idx: int = 0) -> float | None:
        if self.kind != "arm_gripper":
            return None
        c = self.ag.controller.controllers.get("gripper")
        t = getattr(c, "_target_qpos", None)
        return None if t is None else float(t[env_idx, 0])

    def calib_key(self) -> str:
        return self.r.name


class PandaView:
    """the stock ManiSkill Panda (mani_skill.agents.robots.panda): hand + two prismatic fingers"""
    LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
    _PTS: dict = {}

    def __init__(self, ag):
        self.ag = ag
        self.kind = "arm_gripper"
        lm = ag.robot.links_map
        self.links = [lm[n] for n in self.LINKS if n in lm]

    def _local(self, link):
        k = link.name
        if k not in self._PTS:
            from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
            ms = get_component_meshes(link._bodies[0])
            P = np.vstack([np.asarray(m.vertices, float) for m in ms]) if ms else np.zeros((0, 3))
            if len(P) > 4:
                P = v31.reg._hull_samples(P, 400)
            self._PTS[k] = P
        return self._PTS[k]

    def link_points(self, link, env_idx: int = 0) -> np.ndarray:
        T = fast_T(link, env_idx)
        P = self._local(link)
        return P @ T[:3, :3].T + T[:3, 3]

    def gripper_points(self, env_idx: int = 0) -> np.ndarray:
        return np.vstack([self.link_points(l, env_idx) for l in self.links])

    def tcp_T(self, env_idx: int = 0) -> np.ndarray:
        return fast_T(self.ag.tcp, env_idx)

    def reach(self) -> float:
        return 0.13            # hand + fingers lie within 13 cm of panda_hand_tcp

    def enclosure(self, Tb, half, env_idx: int = 0, lateral_margin: float = 0.002) -> float | None:
        """v3.1 all-link rule with the Panda's hand + fingers: points split at the object centre along the closing
        axis (TCP y); only points inside the object's lateral shadow (+2 mm) count; both halves non-empty = enclosed;
        clearance = box distance of half 1 + of half 2"""
        T = self.tcp_T(env_idx)
        u_, v_, w_ = T[:3, 1], T[:3, 0], T[:3, 2]
        P = self.gripper_points(env_idx)
        h = v31.half3(half)
        c = Tb[:3, 3]
        ev = float(np.abs(Tb[:3, :3].T @ v_) @ h); ew = float(np.abs(Tb[:3, :3].T @ w_) @ h)
        D = P - c
        lat = (np.abs(D @ v_) <= ev + lateral_margin) & (np.abs(D @ w_) <= ew + lateral_margin)
        s = D @ u_
        P1, P2 = P[lat & (s < 0)], P[lat & (s > 0)]
        if not len(P1) or not len(P2):
            return None
        return max(float(v31.box_distance(P1, Tb, h).min()), 0.0) + max(float(v31.box_distance(P2, Tb, h).min()), 0.0)

    def opening(self, env_idx: int = 0) -> float:
        q = self.ag.robot.get_qpos()[env_idx, -2:].cpu().numpy()
        return float(q.sum())

    def grip_target(self, env_idx: int = 0) -> float | None:
        c = self.ag.controller.controllers.get("gripper")
        t = getattr(c, "_target_qpos", None)
        return None if t is None else float(t[env_idx, 0])

    def calib_key(self) -> str:
        return "panda"


def view_of(ag):
    return ImsimView(ag) if hasattr(ag, "record") else PandaView(ag)


def views(u) -> list:
    k = "_imsim_v32_views"
    vs = getattr(u, k, None)
    if vs is None or len(vs) != len(agents_of(u)) or any(v.ag is not a for v, a in zip(vs, agents_of(u))):
        vs = [view_of(a) for a in agents_of(u)]
        setattr(u, k, vs)
    return vs


def min_gripper_distance(view, obj: Obj, u, env_idx: int = 0) -> float:
    T = obj.T(u, env_idx)
    h, _ = obj.box(u)
    tcp = view.tcp_T(env_idx)[:3, 3]
    if float(np.linalg.norm(tcp - T[:3, 3])) > view.reach() + float(np.linalg.norm(h)) + 0.05:
        return np.inf
    P = view.gripper_points(env_idx)
    return float(v31.box_distance(P, T, h).min()) if len(P) else np.inf


# ================================================================================================ gripper calibration
_CALIB: dict | None = None


def grip_calib() -> dict:
    global _CALIB
    if _CALIB is None:
        _CALIB = json.load(open(GRIP_CALIB_PATH)) if os.path.exists(GRIP_CALIB_PATH) else {}
    return _CALIB


def free_opening(view, target: float) -> float | None:
    """free-space opening [m] reached at the gripper PD target `target` (first gripper joint): Panda analytic
    (2 fingers x clip(target, 0, 0.04)); imsim grippers from the calibration table verify/grip_calib.json (measured
    opening at steady state for a sweep of targets, nothing in the gripper), linear interpolation"""
    if target is None:
        return None
    if view.calib_key() == "panda":
        return 2.0 * float(np.clip(target, 0.0, 0.04))
    c = grip_calib().get(view.calib_key())
    if not c:
        return None
    t, o = np.asarray(c["target"], float), np.asarray(c["opening"], float)
    i = np.argsort(t)
    return float(np.interp(target, t[i], o[i]))


def calibrate_gripper(env, arm: int = 0, n: int = 21, steps: int = 40) -> dict:
    """sweep the gripper PD target of `arm` from closed to open in free space (arm at home) and record the steady-state
    measured opening; returns dict(target=[...], opening=[...])"""
    from ..planning.scripted import DualExecutor
    ex = DualExecutor(env)
    v = views(env.unwrapped)[arm]
    out_t, out_o = [], []
    for s in np.linspace(0.0, 1.0, n):
        ex.set_gripper(arm, float(s), n=steps)
        out_t.append(v.grip_target()); out_o.append(v.opening())
    return dict(target=out_t, opening=out_o)


# ================================================================================================ probes
class PosSpeed:
    """speed of the fastest bounding-box corner over the last control step, from corner POSITIONS (m/s), max over the
    given objects. The first step after a reset (or a jump of the env's step counter) has no previous positions; it
    reports PhysX's linear speed (as veritask's RobotSpeed does for joints)."""

    def __init__(self, objs: list[Obj]):
        self.objs = list(objs)
        self.prev = None

    def __call__(self, u) -> float:
        C = np.vstack([o.corners(u) for o in self.objs])
        k = int(np.asarray(u.elapsed_steps).reshape(-1)[0]) if hasattr(u, "elapsed_steps") else None
        p, self.prev = self.prev, (id(u), k, C)
        if p is not None and p[0] == id(u) and k is not None and p[1] == k - 1 and p[2].shape == C.shape:
            return float(np.linalg.norm(C - p[2], axis=1).max() / float(u.control_timestep))
        return float(max(np.linalg.norm(o.actor(u).linear_velocity[0].cpu().numpy()) for o in self.objs))


class EnclosureObj:
    """smallest enclosure clearance of `obj` over all arms (imsim: v3.1 all-link rule; Panda: PandaView.enclosure);
    NOT_ENCLOSED_M when no gripper encloses it"""

    def __init__(self, obj: Obj):
        self.obj = obj

    def __call__(self, u) -> float:
        T = self.obj.T(u)
        h, _ = self.obj.box(u)
        best = NOT_ENCLOSED_M
        for v in views(u):
            if float(np.linalg.norm(v.tcp_T()[:3, 3] - T[:3, 3])) > v.reach() + float(np.linalg.norm(h)) + 0.02:
                continue
            c = v.enclosure(T, h)
            if c is not None:
                best = min(best, c)
        return float(best)


class GripCmdGap:
    """max over arms whose gripper is within near_m of any of the objects of max(0, measured opening - free-space
    opening at the current gripper PD target) [m]; 0 when no arm is near (or for hands)"""

    def __init__(self, objs: list[Obj], near_m: float = NEAR_M):
        self.objs, self.near_m = list(objs), float(near_m)

    def __call__(self, u) -> float:
        g = 0.0
        for v in views(u):
            if v.kind != "arm_gripper":
                continue
            if min(min_gripper_distance(v, o, u) for o in self.objs) > self.near_m:
                continue
            f = free_opening(v, v.grip_target())
            if f is None:
                continue
            g = max(g, v.opening() - f)
        return float(g)


def _xyz(obj: Obj, i: int):
    return lambda u: float(obj.T(u)[i, 3])


def bottom_dz(obj: Obj):
    return lambda u: float(obj.corners(u)[:, 2].min() - table_top_z(u))


def tilt_deg(obj: Obj, axis: int | None = 2):
    """angle between the object's local axis `axis` and world z (0 = that axis points straight up or down);
    axis=None: the smallest such angle over the three local axes (a box lying flat on any face reads 0)"""
    def f(u):
        R = fast_T(obj.actor(u))[:3, :3]
        c = np.abs(R[2, :]).max() if axis is None else abs(R[2, axis])
        return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))
    return f


def robot_links_of(i: int):
    def f(u):
        ag = agents_of(u)[i]
        return [l for l in ag.robot.links if l._bodies[0].get_collision_shapes()]
    return f


# ================================================================================================ recorder
def build_recorder(env, objects: dict[str, Obj], release: list[str] | None = None, judged: list[str] | None = None,
                   extra_probes: dict | None = None, extra_pairs: dict | None = None, per_arm_pairs: list[str] = (),
                   table_pairs: bool = True, trackers=(), aftermath: list[str] | None = None):
    """TraceRecorderV3 with the v3.2 signals.

    objects       name -> Obj (signals prefixed with the name)
    release       objects that must be released (enclosure signal, robot contact pair, grip_cmd_gap over them)
    judged        objects whose speed enters 'objs_speed' (quiescence) and whose motion the aftermath measures
                  (default: all objects)
    aftermath     objects the retreat aftermath uses instead of `judged` (arms near them retreat; their largest
                  displacement is the measurement), e.g. only the released object when another arm must keep holding
                  something (E29b fork B2; default None = judged)
    Signals: <o>_x/_y/_z (box centre), <o>_speed (PosSpeed), <o>_bottom_dz_m, <o>_tilt_deg, objs_speed (max over
    judged), robot_<o>_impulse_Ns (+ _cum; every collision link of every robot), table_<o>_impulse_Ns,
    <o>_enclosure_m (release objects), grip_cmd_gap_m (release objects), robot_qvel_max, opening_<i>_m,
    grip_target_<i>; plus extra_probes / extra_pairs (PairV3)."""
    from veritask.adapters.maniskill_v2_tasks import _all_robot_links
    from veritask.adapters.maniskill_v3 import PairV3, TraceRecorderV3
    from veritask.adapters.maniskill_v3_tasks import RobotSpeed
    release = list(release or [])
    judged = list(judged if judged is not None else objects)
    probes: dict[str, Callable] = {}
    pairs: dict = {}
    for n, o in objects.items():
        for i, a in enumerate("xyz"):
            probes[f"{n}_{a}"] = _xyz(o, i)
        probes[f"{n}_speed"] = PosSpeed([o])
        probes[f"{n}_bottom_dz_m"] = bottom_dz(o)
        probes[f"{n}_tilt_deg"] = tilt_deg(o, o.tilt_axis)
        pairs[f"robot_{n}"] = PairV3(_all_robot_links, (lambda o: (lambda u: [o.actor(u)]))(o), cumulative=True)
        if table_pairs:
            pairs[f"table_{n}"] = PairV3(lambda u: [table_actor(u)], (lambda o: (lambda u: [o.actor(u)]))(o))
    for n in per_arm_pairs:
        o = objects[n]
        for i in range(len(agents_of(env.unwrapped))):
            pairs[f"arm{i}_{n}"] = PairV3(robot_links_of(i), (lambda o: (lambda u: [o.actor(u)]))(o))
    probes["objs_speed"] = PosSpeed([objects[n] for n in judged])
    for n in release:
        probes[f"{n}_enclosure_m"] = EnclosureObj(objects[n])
    if release:
        probes["grip_cmd_gap_m"] = GripCmdGap([objects[n] for n in release])
    probes["robot_qvel_max"] = RobotSpeed()
    for i, _ in enumerate(agents_of(env.unwrapped)):
        probes[f"opening_{i}_m"] = (lambda i: (lambda u: views(u)[i].opening()))(i)
        probes[f"grip_target_{i}"] = (lambda i: (lambda u: float(views(u)[i].grip_target() or 0.0)))(i)
    probes.update(extra_probes or {})
    pairs.update(extra_pairs or {})
    rec = TraceRecorderV3(env, probes, pairs=pairs, release=None, trackers=trackers)
    rec.v32_objects = dict(objects)
    rec.v32_judged = judged
    rec.v32_aftermath = list(aftermath) if aftermath is not None else None
    rec.v32_release = release
    rec.judge = lambda contract, derive=None: judge(rec, contract, derive=derive)
    return rec


# ================================================================================================ aftermath
def retreat_path_imsim(view: ImsimView, base_pose, dt: float, cfg: dict):
    return v31.plan_retreat(view.ag, base_pose, dt, cfg)


def retreat_path_panda(view: PandaView, dt: float, cfg: dict):
    """straight-line TCP retreat along -z_tcp for the stock Panda: mplib screw plan, else IK + joint interpolation"""
    import mplib
    ag = view.ag
    u_ag = ag
    key = "_imsim_v32_mplib"
    pl = getattr(u_ag, key, None)
    if pl is None:
        pl = mplib.Planner(urdf=ag.urdf_path, srdf=ag.urdf_path.replace(".urdf", ".srdf"),
                           user_link_names=[l.get_name() for l in ag.robot.get_links()],
                           user_joint_names=[j.get_name() for j in ag.robot.get_active_joints()],
                           move_group="panda_hand_tcp")
        setattr(u_ag, key, pl)
    bp = ag.robot.pose
    pl.set_base_pose(np.concatenate([bp.p[0].cpu().numpy(), bp.q[0].cpu().numpy()]).astype(np.float64))
    T = view.tcp_T()
    from transforms3d.quaternions import mat2quat
    goal = np.concatenate([T[:3, 3] - T[:3, 2] * cfg["retreat_m"], mat2quat(T[:3, :3])])
    q = ag.robot.get_qpos()[0].cpu().numpy().astype(np.float64)
    res = pl.plan_screw(goal, q, time_step=dt)
    if res["status"] != "Success":
        res = pl.plan_screw(goal, q, time_step=dt)
    if res["status"] != "Success":
        return None
    P = res["position"][:, :7]
    n_min = max(2, int(round(cfg["retreat_s"] / dt)))
    if len(P) < n_min:                                     # stretch to the contract's retreat duration
        idx = np.linspace(0, len(P) - 1, n_min)
        P = np.array([P[int(round(i))] for i in idx])
    return P


def run_aftermath(rec, cfg: dict, objs: list[Obj] | None = None) -> dict:
    """v31.run_aftermath generalised: several objects (largest displacement over them), imsim agents and the Panda.
    Runs after settle_v3 (hold A) on a CPU recorder; the judged traces are restored afterwards."""
    u = rec.env.unwrapped
    if objs is None:
        names = getattr(rec, "v32_aftermath", None) or rec.v32_judged
        objs = [rec.v32_objects[n] for n in names]
    vs = views(u)
    multi = hasattr(u.agent, "agents")
    keys = list(u.action_space.spaces.keys()) if multi else [None]
    dt = float(u.control_timestep)
    saved = {k: list(v) for k, v in rec.traces.items()}
    n0 = len(saved["phase"])
    if multi:
        act = OrderedDict((k, np.asarray(rec.hold_action[k], np.float32).copy()) for k in keys)
    else:
        act = np.asarray(rec.hold_action, np.float32).copy()
    p0 = [o.T(u)[:3, 3].copy() for o in objs]
    bases = u.dual.mount_poses() if hasattr(u, "dual") else [None] * len(vs)
    near, plans, failed = [], {}, []
    for i, v in enumerate(vs):
        if min(min_gripper_distance(v, o, u) for o in objs) <= cfg["near_m"]:
            near.append(i)
            q = retreat_path_imsim(v, bases[i], dt, cfg) if isinstance(v, ImsimView) else retreat_path_panda(v, dt, cfg)
            if q is None:
                failed.append(i)
            else:
                plans[i] = q
    n_ret = max([len(q) for q in plans.values()] + [int(round(cfg["retreat_s"] / dt))])
    n = n_ret + int(round(cfg["hold_s"] / dt))
    disp, path = 0.0, []
    rec._settling = True
    try:
        for k in range(n):
            for i, q in plans.items():
                row = q[min(k, len(q) - 1)]
                if multi:
                    act[keys[i]][:len(row)] = row
                else:
                    act[:len(row)] = row
            rec.step(act)
            ps = [o.T(u)[:3, 3] for o in objs]
            path.append(np.concatenate(ps))
            disp = max(disp, max(float(np.linalg.norm(p - q0)) for p, q0 in zip(ps, p0)))
    finally:
        rec._settling = False
    aft = {AFT_PREFIX + k: list(v[n0:]) for k, v in rec.traces.items()}
    rec.set_traces(saved)
    ok = not failed
    return dict(value=disp if ok else None, near=near, failed=failed, steps=n, aft=aft,
                obj_path=np.asarray(path).round(5).tolist())


def judge(rec, contract, derive=None) -> dict:
    """settle (hold A) + aftermath (if the contract has a retreat term) + verdict (same dict layout as v31.judge_rec)"""
    win = rec.settle_v3(contract, hold="A")
    extra, aft = {}, None
    cfg = v31.retreat_cfg(contract)
    if cfg is not None and win.get("status") == "ok":
        aft = run_aftermath(rec, cfg)
        if aft["value"] is not None:
            extra[RETREAT_METRIC] = aft["value"]
    u = rec.env.unwrapped
    j = v31.judge_traces(contract, rec.traces, float(u.control_freq), extra, derive=derive)
    j["window"] = win
    j["aftermath"] = None if aft is None else {k: v for k, v in aft.items() if k != "aft"}
    j["_aft_traces"] = None if aft is None else aft["aft"]
    j["extra"] = extra
    return j


# ================================================================================================ contract building
def term(metric: str, op: str, thr: float, units: str, real: str, **args) -> dict:
    return dict(metric=metric, op=op, threshold=thr, units=units, args=args, real_measurement=real)


def window_max(signal, thr, units, real, op="<="):
    return term("trace.window_max", op, thr, units, real, signal=signal, window=20)


def window_min(signal, thr, units, real, op=">="):
    return term("trace.window_min", op, thr, units, real, signal=signal, window=20)


def window_disp(signals, thr, real):
    return term("trace.window_disp", "<=", thr, "m", real, signal=signals[0], signals=list(signals), window=20)


def event_max(signal, thr, units, real, op=">=", **kw):
    return term("trace.max", op, thr, units, real, signal=signal, **kw)


def retreat_term(thr: float = 0.01, what: str = "the object") -> dict:
    return term(RETREAT_METRIC, "<=", thr, "m",
                f"Consequence test after the judged window (protocol v3.1/v3.2): every arm whose gripper is within 3 cm of "
                f"{what} retracts its TCP 3 cm along its approach axis in 0.5 s with its last gripper command unchanged, "
                f"then holds 0.5 s; {what} must move <= {thr * 100:.0f} cm. Real world: same motion, tracked pose.",
                retreat_m=0.03, retreat_s=0.5, hold_s=0.5, near_m=0.03)


def settle_block(objs: list[str], quiet_speed: float = 0.02, quiet_disp: float = 0.001, max_seconds: float = 5.0) -> dict:
    return dict(hold="A", max_seconds=max_seconds, window_seconds=1.0, quiet_steps=10,
                quiet_max={"objs_speed": quiet_speed},
                quiet_disp=[dict(signals=[f"{o}_x", f"{o}_y", f"{o}_z"], max=quiet_disp) for o in objs],
                robot_speed_signal="robot_qvel_max", robot_speed_max=0.01)


def release_block(objs: list[str]) -> dict:
    return dict(contact_signals=[f"robot_{o}_impulse_Ns" for o in objs], max_impulse_Ns=0.0,
                enclosure_signals=[f"{o}_enclosure_m" for o in objs], enclosure_margin_m=0.002)


def at_rest_terms(objs: list[str]) -> list[dict]:
    out = [window_max("objs_speed", 0.02, "m/s",
                      "Settled: the fastest bounding-box corner of every judged object moves <= 2 cm/s at every judged step, "
                      "speed from positions (v3.2, E24c fix 1). Real world: frame differencing of the tracked pose.")]
    for o in objs:
        out.append(window_disp([f"{o}_x", f"{o}_y", f"{o}_z"], 0.002,
                               f"Stays put: the {o} centre moves <= 2 mm over the judged window. Real world: tracked pose, 1 s."))
    return out


def grip_gap_term(thr: float = 0.005) -> dict:
    return window_max("grip_cmd_gap_m", thr, "m",
                      "Commanded vs actual opening (v3.2, E24c fix 3): for every arm whose gripper is within 3 cm of a judged "
                      f"object, measured opening minus the free-space opening of its current gripper command <= {thr * 1000:.0f} mm "
                      "at every judged step (a gripper still closing on, or propped open by, something fails). Real world: "
                      "gripper encoder vs command.")


def contract_dict(task_id: str, instruction: str, outcome: list, counterexamples: list, source: dict, notes: str,
                  require_released: bool = True, release: dict | None = None, settle: dict | None = None) -> dict:
    d = OrderedDict(schema_version="0.3", task_id=task_id, protocol="v3", instruction=instruction, outcome=outcome,
                    require_released=require_released)
    if require_released:
        d["release"] = release
    d["settle_seconds"] = 1.0
    d["settle"] = settle
    d["counterexamples"] = counterexamples
    d["source"] = source
    d["notes"] = notes
    return d


def cx(name: str, desc: str, expected: bool, generator: str) -> dict:
    return dict(name=name, description=desc, expected=expected, generator=generator)
