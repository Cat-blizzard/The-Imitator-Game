"""Verifier hooks for the dual-arm smoke task (contract imsim/contracts/dual_cube_place_v3.json, protocol v3).

The judging is done by veritask (adapters/protocol_v3.evaluate_traces_v3 via TraceRecorderV3); this module only
supplies the signals, in an embodiment-agnostic way:

  cube_x / cube_y / cube_z         cube centre [m]
  cube_speed                       |linear velocity| [m/s]
  cube_target_margin_m             min(half - |dx|, half - |dy|) of the cube centre w.r.t. the target square [m]
  cube_bottom_dz_m                 lowest cube point minus table top [m]
  robot_qvel_max                   largest ARM joint speed of both robots from joint positions (veritask RobotSpeed;
                                   gripper / hand joints excluded through agent.gripper_joint_names)
  robot_cube_impulse_Ns (+ _cum)   contact impulse between the cube and every collision link of both robots,
                                   accumulated over all physics substeps (veritask PairV3 / SubstepContactsV3)
  cube_enclosure_clearance_m       EnclosureAny below (1 m = not enclosed)
  cube_grasped_any                 diagnostic: any agent's is_grasping(cube)
  opening_<i>_m                    diagnostic: measured opening of agent i (registry.measured_opening)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from veritask.adapters.maniskill_v2_tasks import _all_robot_links  # noqa: E402
from veritask.adapters.maniskill_v3 import PairV3, TraceRecorderV3  # noqa: E402
from veritask.adapters.maniskill_v3_tasks import RobotSpeed  # noqa: E402
from veritask.adapters.protocol_v3 import NOT_ENCLOSED_M  # noqa: E402
from veritask.spec.contract import TaskContract  # noqa: E402

from ..embodiments import registry as reg  # noqa: E402
from ..embodiments.urdf_tools import load_urdf  # noqa: E402

CONTRACT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "contracts",
                             "dual_cube_place_v3.json")


def load_contract() -> TaskContract:
    return TaskContract.load(CONTRACT_PATH)


# ------------------------------------------------------------------------------------------------ geometry
def box_surface_points(half: float, n: int = 9) -> np.ndarray:
    """points on the surface of a cube of half size `half` (box frame), n x n per face incl. edges"""
    g = np.linspace(-half, half, n)
    A, B = np.meshgrid(g, g)
    a, b = A.ravel(), B.ravel()
    faces = []
    for ax in range(3):
        for s in (-half, half):
            P = np.zeros((len(a), 3))
            P[:, ax] = s
            o = [i for i in range(3) if i != ax]
            P[:, o[0]] = a; P[:, o[1]] = b
            faces.append(P)
    return np.unique(np.vstack(faces).round(9), axis=0)


def box_distance(P_world: np.ndarray, T_box: np.ndarray, half: float) -> np.ndarray:
    """signed distance of world points to a cube (negative inside)"""
    Ti = np.linalg.inv(T_box)
    Q = (Ti[:3, :3] @ P_world.T).T + Ti[:3, 3]
    q = np.abs(Q) - half
    out = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(axis=1), 0)
    return out + inside


_LINK_PTS: dict = {}


def _link_pts(urdf: str, link: str) -> np.ndarray:
    k = (urdf, link)
    if k not in _LINK_PTS:
        _LINK_PTS[k] = load_urdf(urdf).collision_points(link)
    return _LINK_PTS[k]


def _T(x) -> np.ndarray:
    return x.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)


class EnclosureAny:
    """Embodiment-agnostic enclosure clearance of a box-shaped object (see module docstring of contracts).

    parallel gripper (record.pad_links): the cube centre lies between the two finger sides (between the pad centroids
      along the closing axis, inside the fingers' extent across it grown by the cube half size); clearance = distance
      of finger side 1's collision hull to the cube + the same for side 2 (each >= 0).
    hand (record.hand): the cube centre lies inside the convex hull of all hand-link collision vertices; clearance =
      distance(thumb links, cube) + min over the other fingers of distance(finger links, cube) (each >= 0).
    Returns the smallest clearance over all agents, NOT_ENCLOSED_M (1 m) when no agent encloses the cube."""

    def __init__(self, obj_name: str = "cube", half_attr: str = "cube_half", n: int = 9):
        self.obj_name, self.half_attr, self.n = obj_name, half_attr, n
        self._pts = None

    def __call__(self, u) -> float:
        obj = getattr(u, self.obj_name)
        half = float(getattr(u, self.half_attr))
        if self._pts is None or self._pts[0] != half:
            self._pts = (half, box_surface_points(half, self.n))
        Tb = _T(obj)
        P = (Tb[:3, :3] @ self._pts[1].T).T + Tb[:3, 3]
        best = NOT_ENCLOSED_M
        for ag in u.agent.agents:
            r = ag.record
            c = self._gripper(ag, r, Tb, half) if r.pad_links else self._hand(ag, r, Tb, half)
            if c is not None:
                best = min(best, c)
        return float(best)

    @staticmethod
    def _gripper(ag, r, Tb, half):
        """the cube lies between the two finger sides (its centre between the pad centroids along the closing axis and
        inside the fingers' extent across it, grown by the cube half size); clearance = distance from finger side 1's
        collision hull to the cube + the same for side 2 (box SDF, each clipped at 0). Using every hull point of each
        side (pad AND e.g. the Robotiq inner finger) means a touch anywhere on a finger reads as 0 clearance."""
        lm = ag.robot.links_map
        g = reg.pad_geometry(r, {l: _T(lm[l]) for side in r.sides for l in side})
        c = Tb[:3, 3]
        u, v, w = g["u"], g["v"], g["w"]
        if not ((c - g["c1"]) @ u > 0 and (g["c2"] - c) @ u > 0):
            return None
        A = np.vstack([g["W1_all"], g["W2_all"]])
        for ax in (v, w):
            if not (A @ ax).min() - half <= c @ ax <= (A @ ax).max() + half:
                return None
        d1 = max(float(box_distance(g["W1_all"], Tb, half).min()), 0.0)
        d2 = max(float(box_distance(g["W2_all"], Tb, half).min()), 0.0)
        return d1 + d2

    @staticmethod
    def _hand(ag, r, Tb, half):
        from scipy.spatial import Delaunay
        h = reg.hand_of(r)
        lm = ag.robot.links_map
        W = {}
        for name in r.grasp_links:
            if name not in lm:
                continue
            V = _link_pts(r.urdf_path, name)
            if len(V):
                T = _T(lm[name])
                W[name] = (T[:3, :3] @ V.T).T + T[:3, 3]
        allp = np.vstack(list(W.values()))
        try:        # point in the convex hull of the points (same set as Delaunay(allp).find_simplex >= 0, ~50x faster)
            from scipy.spatial import ConvexHull
            h = ConvexHull(allp)
            inside = bool((h.equations[:, :3] @ Tb[:3, 3] + h.equations[:, 3] <= 1e-9).all())
        except Exception:
            inside = False
        if not inside:
            return None
        thumb = [n for n in ("link_12.0", "link_13.0", "link_14.0", "link_15.0", "link_15.0_tip") if n in W]
        fingers = {f: [n for n in (f"link_{a}.0", f"link_{a+1}.0", f"link_{a+2}.0", f"link_{a+2}.0_tip") if n in W]
                   for f, a in (("index", 1), ("middle", 5), ("ring", 9))}
        d_th = min(float(box_distance(W[n], Tb, half).min()) for n in thumb)
        d_f = min(min(float(box_distance(W[n], Tb, half).min()) for n in ns) for ns in fingers.values() if ns)
        return max(d_th, 0.0) + max(d_f, 0.0)


# ------------------------------------------------------------------------------------------------ recorder
def probes() -> dict:
    def cube_xyz(i):
        return lambda u: float(u.cube.pose.p[0, i])

    def target_margin(u):
        d = (u.cube.pose.p[0, :2] - u.target_pos[0, :2]).abs().cpu().numpy()
        return float(u.dual.target_half - d.max())

    p = dict(cube_x=cube_xyz(0), cube_y=cube_xyz(1), cube_z=cube_xyz(2),
             cube_speed=lambda u: float(np.linalg.norm(u.cube.linear_velocity[0].cpu().numpy())),
             cube_target_margin_m=target_margin,
             cube_bottom_dz_m=lambda u: float(u.cube_bottom_z()[0]),
             robot_qvel_max=RobotSpeed(),
             cube_enclosure_clearance_m=EnclosureAny(),
             cube_grasped_any=lambda u: float(any(bool(ag.is_grasping(u.cube)[0]) for ag in u.agent.agents)))
    for i in range(2):
        p[f"opening_{i}_m"] = (lambda i: lambda u: float(u.agent.agents[i].measured_opening()[0]))(i)
    return p


def make_recorder(env) -> TraceRecorderV3:
    pairs = {"robot_cube": PairV3(_all_robot_links, lambda u: [u.cube], gap=False, cumulative=True)}
    return TraceRecorderV3(env, probes(), pairs=pairs, release=None)


def judge(rec: TraceRecorderV3, contract: TaskContract | None = None, hold: str = "A") -> dict:
    """run the evaluator's settle (hold A by default) and return the contract verdict + benchmark flags"""
    c = contract or load_contract()
    win = rec.settle_v3(c, hold=hold)
    v, bench, info = rec.evaluate_v3(c, hold=hold)
    return dict(success=v["success"], released=v["released"], binding_term=v.get("binding_term"),
                terms={t["metric"]: t["value"] for t in v["terms"]}, window=win, bench=bench, info=info,
                protocol_reject=v.get("protocol_reject"))
