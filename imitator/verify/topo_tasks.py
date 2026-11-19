"""v3.2 recorders for the card-D2 tasks (RingOnPeg-v1, MugOnHook-v1) and the winding-number variant of PlaceSphere-v1.

Generic v3.2 signals (imitator.verify.v32.build_recorder) plus
  * POSE signals <prefix>_px/_py/_pz/_qw/_qx/_qy/_qz (actor frame origin + wxyz quaternion, every control step): the
    topology trace metrics of veritask (outcomes/topology.py) compute crossings / insertion margin / winding number
    from them and from geometry declared in the contract, so a verdict is recomputable from the trace file;
  * a parts-based enclosure signal (<o>_enclosure_m): the Panda all-link enclosure rule (v32.PandaView.enclosure)
    applied to every convex part of the object (ring capsules, mug wall boxes, handle capsules) instead of its bounding
    box (a ring is held by its tube, never around its centre); smallest clearance over the parts near the hand;
  * the benchmark predicate's components (bench_*) for diagnostics; bench_success itself = env.evaluate()['success'].
Every recorder's judge is imitator.verify.proof_test.judge (v32 settle + aftermath + verdict, then the proof test when the
contract declares one).
"""
from __future__ import annotations

import numpy as np

from . import proof_test as PT
from .v32 import NOT_ENCLOSED_M, Obj, build_recorder, fast_T, views


def _pair(a: str, b: str):
    from veritask.adapters.maniskill_v3 import PairV3
    return PairV3(lambda u: [getattr(u, a)], lambda u: [getattr(u, b)])


class _PoseCache:
    def __init__(self, attr: str):
        self.attr, self.key, self.val = attr, None, None

    def __call__(self, u):
        k = (id(u), int(np.asarray(u.elapsed_steps).reshape(-1)[0]) if hasattr(u, "elapsed_steps") else None)
        if k != self.key or k[1] is None:
            p = PT.pose_of(getattr(u, self.attr))
            q = np.asarray(p.q, float)
            self.val, self.key = np.r_[np.asarray(p.p, float), q if q[0] >= 0 else -q], k
        return self.val


def pose_probes(prefix: str, attr: str) -> dict:
    """7 probes <prefix>_px .. _qz reading the actor `attr` once per control step"""
    c = _PoseCache(attr)
    names = ("px", "py", "pz", "qw", "qx", "qy", "qz")
    return {f"{prefix}_{n}": (lambda i: (lambda u: float(c(u)[i])))(i) for i, n in enumerate(names)}


class EnclosureParts:
    """smallest Panda enclosure clearance over the object's convex parts (each a box in the object frame);
    NOT_ENCLOSED_M when no part is enclosed. parts: list of (4x4 part-in-object transform, half extents (3,))."""

    def __init__(self, attr: str, parts):
        self.attr = attr
        self.parts = [(np.asarray(T, float), np.asarray(h, float)) for T, h in parts]

    def __call__(self, u) -> float:
        To = fast_T(getattr(u, self.attr))
        best = NOT_ENCLOSED_M
        for v in views(u):
            tcp = v.tcp_T()[:3, 3]
            for Tp, h in self.parts:
                Tb = To @ Tp
                if float(np.linalg.norm(Tb[:3, 3] - tcp)) > v.reach() + float(np.linalg.norm(h)) + 0.02:
                    continue
                c = v.enclosure(Tb, h)
                if c is not None:
                    best = min(best, c)
        return float(best)


def _T(pose) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), float)


def ring_parts(cls) -> list:
    from ..envs.topo_tasks import _ring_parts
    return [(_T(p), [hl + r, r, r]) for p, r, hl in _ring_parts(cls.RING_R, cls.RING_A, cls.RING_N)]


def mug_parts(cls) -> list:
    import sapien
    from transforms3d.euler import euler2quat
    from ..envs.topo_tasks import _ring_parts
    out = [(_T(sapien.Pose([0, 0, cls.MUG_BOTTOM / 2])), [cls.MUG_R, cls.MUG_R, cls.MUG_BOTTOM / 2])]
    hw = cls.MUG_R * np.tan(np.pi / cls.MUG_N) * 1.05
    for k in range(cls.MUG_N):
        th = 2 * np.pi * k / cls.MUG_N
        rr = cls.MUG_R - cls.MUG_WALL / 2
        out.append((_T(sapien.Pose([rr * np.cos(th), rr * np.sin(th), cls.MUG_H / 2], euler2quat(0, 0, th))),
                    [cls.MUG_WALL / 2, hw, cls.MUG_H / 2]))
    out += [(_T(p), [hl + r, r, r]) for p, r, hl in _ring_parts(cls.HANDLE_R, cls.HANDLE_A, cls.HANDLE_N, plane="xz",
                                                                  centre=cls.handle_centre())]
    return out


def _attach_judge(rec):
    rec.judge = lambda contract, derive=None: PT.judge(rec, contract, derive=derive)
    return rec


# ------------------------------------------------------------------------------------------------ RingOnPeg
def ring_on_peg(env):
    u = env.unwrapped
    R, A = u.RING_R, u.RING_A
    objs = {"ring": Obj("ring", half=(R + A, R + A, A))}
    probes = {}
    probes.update(pose_probes("ring", "ring"))
    probes.update(pose_probes("peg", "peg"))
    probes["ring_enclosure_m"] = EnclosureParts("ring", ring_parts(type(u)))
    probes["bench_on_peg"] = lambda u: float(u.bench_terms()["on_peg"])
    probes["bench_r_reach"] = lambda u: float(u.bench_terms()["r_reach"])
    rec = build_recorder(env, objs, release=["ring"], extra_probes=probes, extra_pairs={"ring_peg": _pair("ring", "peg")})
    return _attach_judge(rec)


# ------------------------------------------------------------------------------------------------ MugOnHook
def mug_on_hook(env):
    u = env.unwrapped
    objs = {"mug": Obj("mug", half=None)}
    probes = {}
    probes.update(pose_probes("mug", "mug"))
    probes.update(pose_probes("rack", "rack"))
    probes["mug_enclosure_m"] = EnclosureParts("mug", mug_parts(type(u)))
    for k in ("xy_ok", "z_ok", "grip_open"):
        probes[f"bench_{k}"] = (lambda k: (lambda u: float(u.bench_terms()[k])))(k)
    rec = build_recorder(env, objs, release=["mug"], extra_probes=probes, extra_pairs={"mug_rack": _pair("mug", "rack")})
    return _attach_judge(rec)


# ------------------------------------------------------------------------------------------------ PlaceSphere (D2 variant)
def place_sphere_wn(env):
    """the E29b PlaceSphere recorder (imitator.verify.panda_tasks.place_sphere, unchanged signals) + pose signals of the
    ball ('sphere_*') and the bin ('bin_*') for the winding-number contract, + the proof-test judge"""
    from .panda_tasks import place_sphere
    rec = place_sphere(env)
    rec.probes.update(pose_probes("sphere", "obj"))
    rec.probes.update(pose_probes("bin", "bin"))
    return _attach_judge(rec)
