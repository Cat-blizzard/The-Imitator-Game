"""Protocol v3.1 additions for imsim contracts (E23): the two fixes proposed by E19 section 5.

Nothing in veritask is modified: v3.1 = protocol v3 (veritask/adapters/protocol_v3.py, unchanged) plus
  (a) an enclosure signal over ALL gripper / hand links (EnclosureAll), used as the contract's enclosure signal, and
  (b) an evaluator-run consequence test after the judged window ("robot moves away, object stays"): every arm whose
      gripper / hand is within `near_m` of the object retracts its TCP by `retreat_m` along the approach axis (-z of the
      TCP frame) over `retreat_s` with its last gripper command unchanged, then holds for `hold_s`; the outcome term
      `aftermath.retreat_object_disp` (largest displacement of the object centre during this aftermath) must stay
      below its threshold. The aftermath is NOT part of the judged traces (they are restored afterwards); it is saved
      under keys prefixed 'aft:' so a verdict can be recomputed from the trace file (judge_traces()).

EnclosureAll (per agent, smallest clearance over agents; NOT_ENCLOSED_M = 1 m when nothing encloses the object)
  parallel gripper  min of
                    - the v3 rule (smoke.EnclosureAny._gripper: object centre between the two finger SIDES' pads), and
                    - all-link rule: every collision-hull point of every gripper link (all links below the flange + the
                      record's grasp_links) is split by the closing axis u (pad-1 -> pad-2) at the OBJECT centre; only
                      points inside the object's lateral shadow (its support extent across u, grown by 2 mm) count. If
                      both halves are non-empty the object is enclosed; clearance = box distance of half 1 + of half 2
                      (each clipped at 0). This sees e.g. the Robotiq inner knuckles that trapped the cube in E19.
  hand              object centre inside the convex hull of all hand-link points (as v3); clearance = thumb distance +
                    min(distance of index / middle / ring finger, distance of the palm).
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np

from . import smoke as S
from ..embodiments import registry as reg
from ..embodiments.urdf_tools import load_urdf

NOT_ENCLOSED_M = S.NOT_ENCLOSED_M
AFT_PREFIX = "aft:"
RETREAT_METRIC = "aftermath.retreat_object_disp"
CONTRACT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "contracts")
DEFAULT_RETREAT = dict(retreat_m=0.03, retreat_s=0.5, hold_s=0.5, near_m=0.03)


def register_metrics() -> None:
    """register the aftermath metric name in veritask's outcome registry (runtime only, no file change) so contracts
    using it validate without an 'unknown metric' warning. Its value always comes from extra_measurements."""
    from veritask.outcomes.metrics import REGISTRY
    if RETREAT_METRIC not in REGISTRY:
        def _aft(**_):
            raise KeyError(f"{RETREAT_METRIC} is measured by the imsim evaluator's aftermath (extra_measurements)")
        REGISTRY[RETREAT_METRIC] = _aft


register_metrics()


def load_contract(name: str = "dual_cube_place_v3.1"):
    from veritask.spec.contract import TaskContract
    p = name if name.endswith(".json") else os.path.join(CONTRACT_DIR, name + ".json")
    return TaskContract.load(p)


def retreat_cfg(contract) -> dict | None:
    """aftermath parameters from the contract's retreat term (None: the contract has no retreat term)"""
    for t in contract.outcome:
        if t.metric == RETREAT_METRIC:
            return {**DEFAULT_RETREAT, **{k: float(v) for k, v in (t.args or {}).items() if k in DEFAULT_RETREAT}}
    return None


# ------------------------------------------------------------------------------------------------ box geometry
def half3(h) -> np.ndarray:
    a = np.asarray(h, float).reshape(-1)
    return np.repeat(a, 3) if a.size == 1 else a


def box_surface_points(half, n: int = 9) -> np.ndarray:
    h = half3(half)
    out = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        A, B = np.meshgrid(np.linspace(-h[o[0]], h[o[0]], n), np.linspace(-h[o[1]], h[o[1]], n))
        for s in (-h[ax], h[ax]):
            P = np.zeros((A.size, 3)); P[:, ax] = s; P[:, o[0]] = A.ravel(); P[:, o[1]] = B.ravel()
            out.append(P)
    return np.unique(np.vstack(out).round(9), axis=0)


def box_distance(P_world: np.ndarray, T_box: np.ndarray, half) -> np.ndarray:
    """signed distance of world points to a box (negative inside); half = scalar or 3 half extents"""
    h = half3(half)
    R, t = T_box[:3, :3], T_box[:3, 3]
    Q = (P_world - t) @ R
    q = np.abs(Q) - h
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


# ------------------------------------------------------------------------------------------------ link sets
_GL: dict = {}
_PTS: dict = {}


def gripper_links(rec: reg.ArmRecord) -> list[str]:
    """every link below the flange (gripper / hand incl. linkage, base, pads) + the record's grasp_links, with geometry"""
    if rec.name not in _GL:
        m = load_urdf(rec.urdf_path)
        out, stack = [], [rec.flange_link]
        while stack:
            l = stack.pop()
            for j in m.child_joints(l):
                out.append(j.child); stack.append(j.child)
        names = list(dict.fromkeys(out + list(rec.grasp_links)))
        _GL[rec.name] = [n for n in names if len(_link_pts(rec, n))]
    return _GL[rec.name]


def _link_pts(rec, link) -> np.ndarray:
    k = (rec.name, link)
    if k not in _PTS:
        V = load_urdf(rec.urdf_path).collision_points(link)
        _PTS[k] = reg._hull_samples(V) if len(V) >= 4 else V
    return _PTS[k]


def link_transforms(ag, names, env_idx: int = 0) -> dict:
    lm = ag.robot.links_map
    out = {}
    for n in names:
        if n in lm:
            out[n] = lm[n].pose.to_transformation_matrix()[env_idx].cpu().numpy().astype(float)
    return out


def _world(rec, Ts: dict, names) -> dict:
    W = {}
    for n in names:
        if n in Ts:
            V = _link_pts(rec, n)
            if len(V):
                T = Ts[n]
                W[n] = V @ T[:3, :3].T + T[:3, 3]
    return W


# ------------------------------------------------------------------------------------------------ clearances
def pad_clearance(rec, Ts, Tb, half) -> float | None:
    """v3 rule (smoke.EnclosureAny._gripper) for a box of half extents `half`"""
    g = reg.pad_geometry(rec, Ts)
    c = Tb[:3, 3]
    u, v, w = g["u"], g["v"], g["w"]
    if not ((c - g["c1"]) @ u > 0 and (g["c2"] - c) @ u > 0):
        return None
    h = half3(half)
    A = np.vstack([g["W1_all"], g["W2_all"]])
    for ax in (v, w):
        e = float(np.abs(Tb[:3, :3].T @ ax) @ h)
        if not (A @ ax).min() - e <= c @ ax <= (A @ ax).max() + e:
            return None
    d1 = max(float(box_distance(g["W1_all"], Tb, h).min()), 0.0)
    d2 = max(float(box_distance(g["W2_all"], Tb, h).min()), 0.0)
    return d1 + d2


def all_links_clearance(rec, Ts, Tb, half, lateral_margin: float = 0.002) -> float | None:
    g = reg.pad_geometry(rec, Ts)
    u, v, w = g["u"], g["v"], g["w"]
    W = _world(rec, Ts, gripper_links(rec))
    if not W:
        return None
    P = np.vstack(list(W.values()))
    h = half3(half)
    c = Tb[:3, 3]
    ev = float(np.abs(Tb[:3, :3].T @ v) @ h); ew = float(np.abs(Tb[:3, :3].T @ w) @ h)
    D = P - c
    lat = (np.abs(D @ v) <= ev + lateral_margin) & (np.abs(D @ w) <= ew + lateral_margin)
    s = D @ u
    P1, P2 = P[lat & (s < 0)], P[lat & (s > 0)]
    if not len(P1) or not len(P2):
        return None
    d1 = max(float(box_distance(P1, Tb, h).min()), 0.0)
    d2 = max(float(box_distance(P2, Tb, h).min()), 0.0)
    return d1 + d2


ALLEGRO_THUMB = ("link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip")
ALLEGRO_FINGERS = {f: (f"link_{a}.0", f"link_{a+1}.0", f"link_{a+2}.0", f"link_{a+3}.0", f"link_{a+3}.0_tip")
                   for f, a in (("index", 0), ("middle", 4), ("ring", 8))}


_RAW: dict = {}


def _link_raw(rec, link) -> np.ndarray:
    """raw collision vertices / primitive samples (hull containment test; the dense samples are for distances)"""
    k = (rec.name, link)
    if k not in _RAW:
        _RAW[k] = load_urdf(rec.urdf_path).collision_points(link)
    return _RAW[k]


def inside_hull(P: np.ndarray, x: np.ndarray) -> bool:
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(P)
    except Exception:
        return False
    return bool((h.equations[:, :3] @ x + h.equations[:, 3] <= 1e-9).all())


def hand_clearance(rec, Ts, Tb, half, with_palm: bool = True) -> float | None:
    names = gripper_links(rec)
    raw = [(Ts[n][:3, :3] @ _link_raw(rec, n).T).T + Ts[n][:3, 3] for n in names if n in Ts and len(_link_raw(rec, n))]
    if not raw or not inside_hull(np.vstack(raw), Tb[:3, 3]):
        return None
    W = _world(rec, Ts, names)
    h = half3(half)
    d = lambda names: min((float(box_distance(W[n], Tb, h).min()) for n in names if n in W), default=np.inf)  # noqa: E731
    d_th = d(ALLEGRO_THUMB)
    opp = [d(ns) for ns in ALLEGRO_FINGERS.values()]
    if with_palm:
        opp.append(d(["base_link"]))
    return max(d_th, 0.0) + max(min(opp), 0.0)


_REACH: dict = {}


def gripper_reach(rec) -> float:
    """largest distance from the TCP origin to any gripper-link collision point (home pose, gripper open)"""
    if rec.name not in _REACH:
        m = load_urdf(rec.urdf_path)
        Ts = m.fk(rec.home_qpos())
        Ti = np.linalg.inv(Ts[rec.tcp_link])
        d = 0.0
        for l in gripper_links(rec):
            V = _link_pts(rec, l)
            if len(V):
                T = Ti @ Ts[l]
                d = max(d, float(np.linalg.norm(V @ T[:3, :3].T + T[:3, 3], axis=1).max()))
        _REACH[rec.name] = d
    return _REACH[rec.name]


def too_far(rec, tcp_p, Tb, half, slack: float = 0.02) -> bool:
    """True when no gripper point can be within reach of the box (exact shortcut: every gripper point lies within
    gripper_reach of the TCP, every box point within its half diagonal of the centre)"""
    return float(np.linalg.norm(np.asarray(tcp_p) - Tb[:3, 3])) > gripper_reach(rec) + float(np.linalg.norm(half3(half))) + slack


def enclosure_all_value(agents, Ts_per_agent: list[dict], Tb: np.ndarray, half, tcp_ps=None) -> float:
    best = NOT_ENCLOSED_M
    for k, (ag, Ts) in enumerate(zip(agents, Ts_per_agent)):
        r = ag.record
        if tcp_ps is not None and too_far(r, tcp_ps[k], Tb, half):
            continue
        cands = []
        if r.pad_links:
            cands += [pad_clearance(r, Ts, Tb, half), all_links_clearance(r, Ts, Tb, half)]
        else:
            cands.append(hand_clearance(r, Ts, Tb, half))
        for c in cands:
            if c is not None:
                best = min(best, c)
    return float(best)


def needed_links(ag) -> list[str]:
    r = ag.record
    names = set(gripper_links(r))
    if r.pad_links:
        for side in r.sides:
            names |= set(side)
    return sorted(names)


class EnclosureAll:
    """probe (env 0): smallest all-link enclosure clearance of the object over all agents"""

    def __init__(self, obj_name: str = "cube", half_attr: str = "cube_half"):
        self.obj_name, self.half_attr = obj_name, half_attr

    def __call__(self, u) -> float:
        obj = getattr(u, self.obj_name)
        half = getattr(u, self.half_attr)
        Tb = obj.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)
        agents = list(u.agent.agents)
        tcp = [ag.tcp.pose.p[0].cpu().numpy().astype(float) for ag in agents]
        return enclosure_all_value(agents, [link_transforms(ag, needed_links(ag)) for ag in agents], Tb, half, tcp_ps=tcp)


def min_gripper_distance(ag, Tb, half, env_idx: int = 0) -> float:
    """smallest distance from any gripper / hand link point of `ag` to the box"""
    r = ag.record
    W = _world(r, link_transforms(ag, gripper_links(r), env_idx), gripper_links(r))
    if not W:
        return np.inf
    return float(box_distance(np.vstack(list(W.values())), Tb, half).min())


# ------------------------------------------------------------------------------------------------ recorder (cube place)
def probes_v31() -> dict:
    p = S.probes()
    p["cube_enclosure_all_m"] = EnclosureAll()
    return p


def make_recorder_v31(env):
    from veritask.adapters.maniskill_v2_tasks import _all_robot_links
    from veritask.adapters.maniskill_v3 import PairV3, TraceRecorderV3
    pairs = {"robot_cube": PairV3(_all_robot_links, lambda u: [u.cube], gap=False, cumulative=True)}
    return TraceRecorderV3(env, probes_v31(), pairs=pairs, release=None)


# ------------------------------------------------------------------------------------------------ aftermath (CPU, 1 env)
def _tcp_T(ag, env_idx: int = 0) -> np.ndarray:
    return ag.tcp.pose.to_transformation_matrix()[env_idx].cpu().numpy().astype(float)


def plan_retreat(ag, base_pose, control_dt: float, cfg: dict, env_idx: int = 0):
    """joint path (arm joints, one row per control step) that moves the TCP back by cfg['retreat_m'] along -z_tcp"""
    import sapien
    from transforms3d.quaternions import mat2quat
    from ..planning.mplib_planner import ArmPlanner
    planner = ArmPlanner(ag, base_pose, control_dt=control_dt, table=False, env_idx=env_idx)
    T = _tcp_T(ag, env_idx)
    goal = sapien.Pose(T[:3, 3] - T[:3, 2] * cfg["retreat_m"], mat2quat(T[:3, :3]))
    n_min = max(2, int(round(cfg["retreat_s"] / control_dt)))
    res = planner.cartesian(goal, speed=cfg["retreat_m"] / cfg["retreat_s"], min_steps=n_min)
    if res["status"] != "Success":
        res = planner.joint_interp(goal)
    return res.get("position") if res["status"] == "Success" else None


def run_aftermath(rec, obj, half, cfg: dict, base_poses, keys=None) -> dict:
    """Run the retreat aftermath on a CPU (1-env) recorder after settle_v3 (hold A): returns the measurement and the
    aftermath traces (the recorder's judged traces are restored)."""
    u = rec.env.unwrapped
    agents = list(u.agent.agents)
    keys = keys or list(u.action_space.spaces.keys())
    dt = float(u.control_timestep)
    saved = {k: list(v) for k, v in rec.traces.items()}
    n0 = len(saved["phase"])
    act = OrderedDict((k, np.asarray(rec.hold_action[k], np.float32).copy()) for k in keys)
    Tb0 = obj.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)
    p0 = Tb0[:3, 3].copy()
    near, plans, failed = [], {}, []
    for i, ag in enumerate(agents):
        if min_gripper_distance(ag, Tb0, half) <= cfg["near_m"]:
            near.append(i)
            q = plan_retreat(ag, base_poses[i], dt, cfg)
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
                nd = agents[i].record.dof_arm
                act[keys[i]][:nd] = q[min(k, len(q) - 1)]
            rec.step(act)
            p = obj.pose.p[0].cpu().numpy().astype(float)
            path.append(p)
            disp = max(disp, float(np.linalg.norm(p - p0)))
    finally:
        rec._settling = False
    aft = {AFT_PREFIX + k: list(v[n0:]) for k, v in rec.traces.items()}
    rec.set_traces(saved)
    ok = not failed
    return dict(value=disp if ok else None, near=near, failed=failed, steps=n, aft=aft,
                obj_path=np.asarray(path).round(5).tolist())


# ------------------------------------------------------------------------------------------------ judging
def strip_aft(traces: dict) -> tuple[dict, dict]:
    tr = {k: v for k, v in traces.items() if not k.startswith(AFT_PREFIX)}
    aft = {k[len(AFT_PREFIX):]: v for k, v in traces.items() if k.startswith(AFT_PREFIX)}
    return tr, aft


def judge_traces(contract, traces: dict, control_freq: float = 20.0, extra: dict | None = None,
                 derive=None) -> dict:
    """verdict from a trace dict (judged traces + optional 'aft:' keys + extra measurements). `derive(traces)` may add
    derived per-step signals (e.g. the handover event) before judging."""
    from veritask.adapters.protocol_v3 import evaluate_traces_v3
    tr, _ = strip_aft(traces)
    if derive is not None:
        tr = derive(dict(tr))
    v, bench, info = evaluate_traces_v3(contract, tr, control_freq=control_freq, extra_measurements=dict(extra or {}))
    return dict(success=v["success"], released=v.get("released"), binding_term=v.get("binding_term"),
                terms={t["metric"]: t["value"] for t in v["terms"]}, margins={t["metric"]: t["norm_margin"] for t in v["terms"]},
                bench=bench, info=info, protocol_reject=v.get("protocol_reject"),
                margin=min((t["norm_margin"] for t in v["terms"] if t.get("norm_margin") is not None), default=None))


def judge_rec(rec, contract, obj, half, base_poses, derive=None) -> dict:
    """settle (hold A) + aftermath (if the contract has a retreat term) + verdict for a CPU recorder. A recorder built by
    imitator.verify.v32.build_recorder carries its own judge (several objects, Panda, v3.2 signals): it is used instead."""
    if getattr(rec, "judge", None) is not None:
        return rec.judge(contract, derive=derive)
    win = rec.settle_v3(contract, hold="A")
    extra, aft = {}, None
    cfg = retreat_cfg(contract)
    if cfg is not None and win.get("status") == "ok":
        aft = run_aftermath(rec, obj, half, cfg, base_poses)
        if aft["value"] is not None:
            extra[RETREAT_METRIC] = aft["value"]
    u = rec.env.unwrapped
    j = judge_traces(contract, rec.traces, float(u.control_freq), extra, derive=derive)
    j["window"] = win
    j["aftermath"] = None if aft is None else {k: v for k, v in aft.items() if k != "aft"}
    j["_aft_traces"] = None if aft is None else aft["aft"]
    j["extra"] = extra
    return j
