"""Topological success primitives (card D2): quantities that say WHETHER an object is inside a container, threaded
through a loop or wrapped around an axis, instead of HOW FAR it is from a hand-picked goal point. They do not change
with the exact shape of the objects, so solver and parameter errors that move things by millimetres do not flip them.

Pure numpy. Three primitives (reference implementation and sanity check: research/scripts/e30_probe_topology.py):

  winding_number(V, F, points)          generalized winding number of a triangle mesh at query points (Jacobson,
                                        Kavan, Sorkine-Hornung 2013): sum of signed solid angles / 4 pi. Closed surface
                                        with outward normals: 1 inside, 0 outside. OPEN surfaces are allowed (a bin, a
                                        cup, a bag): the value is then a continuous "how enclosed is this point" in about
                                        [0, 1]; it is exactly 0.5 on the plane of a flat opening (at the opening's
                                        centre for a symmetric container). Margin for "inside": (w - 0.5) / 0.5.
  disk_crossings(polyline, centre, normal, radius)
                                        signed number of times a curve crosses the flat disk spanned by a loop (a
                                        ring's core circle, a mug handle): "threaded" <=> net crossings != 0. Also an
                                        insertion margin: the curve length beyond the crossing on the free side(s),
                                        i.e. how far the loop is from slipping off the free end.
  wrap_angle(polyline, axis_point, axis_dir)
                                        total signed angle (rad) swept by the curve's projection around an axis
                                        (rope wound around a post); turns = |angle| / 2 pi.

Registered metrics (outcomes.metrics.REGISTRY; contracts refer to them by name):
  geometry (end-state arrays from any adapter, e.g. a soft-body solver's vertices):
    topology.winding_number, topology.containment_margin, topology.disk_crossings, topology.insertion_margin,
    topology.wrap_angle (deg, signed), topology.wrap_turns
  trace metrics (per control step, from POSE signals recorded by an adapter + geometry declared in the contract's term
  args, so a verdict is recomputable from the trace file and the contract alone):
    trace.winding_window_min / trace.winding_min        min over the query points of the winding number
    trace.crossings_window_min / trace.crossings_min    net crossings (|net| unless args.absolute is false)
    trace.insertion_window_min / trace.insertion_min    insertion margin [m] (0 when not threaded)
    trace.wrap_turns_window_min / trace.wrap_turns_min  |wrap angle| / 2 pi
  '*_window_min' = worst value over the last `window` steps (end-state; protocol v3 sets window to the judged window);
  '*_min' = worst value over the whole trace given (event; e.g. "never unthreaded during a proof test").

Pose signals (trace metrics): an actor's world pose per step is recorded as <prefix>_px, _py, _pz (frame origin, m)
and <prefix>_qw, _qx, _qy, _qz (unit quaternion, SAPIEN's wxyz order). Wherever an arg expects a pose, it may be such a
prefix (a string), a static pose [px, py, pz, qw, qx, qy, qz], or omitted / null (identity: geometry given in world).

Trace-metric args (all lengths in metres; local = in the frame of the given pose):
  winding     signal (any recorded signal; the measurement key), mesh (mesh spec, see mesh_from_spec), container
              (pose of the mesh), object (pose of the queried object), points (local query points, default [[0,0,0]]),
              points_frame ('object': points rotate with the object; 'world': world-axis offsets from the object's
              origin, e.g. the lowest point of a ball), window
  crossings / insertion
              signal, ring (pose of the loop), ring_centre (local, default [0,0,0]), ring_normal (local unit axis,
              default [0,0,1]), ring_radius, curve (local polyline points), curve_pose (pose of the curve), absolute
              (crossings only, default true), free_end ('both' | 'start' | 'end'; insertion only, default 'both'), window
  wrap_turns  signal, curve, curve_pose, axis_point / axis_dir (local to axis_pose), axis_pose, window
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import metric

TWO_PI = 2.0 * np.pi
FREE_ENDS = ("both", "start", "end")


# ================================================================================================ winding number
def winding_number(V, F, points, max_pairs: int = 1 << 20):
    """Generalized winding number of the triangle mesh (V, F) at `points`.

    V (nv, 3) float, F (nf, 3) int (vertex indices; counter-clockwise seen from outside = outward normal), points (3,)
    or (n, 3). Returns a float for a single point, else an (n,) array. Open meshes are allowed. Each triangle adds its
    signed solid angle (Van Oosterom & Strackee 1983): tan(Omega / 2) = a.(b x c) / (|a||b||c| + (a.b)|c| + (b.c)|a| +
    (c.a)|b|), a, b, c = triangle corners relative to the point. Work is chunked so that at most `max_pairs`
    point-triangle pairs are held in memory. Points exactly on the surface get an arbitrary value between the two
    sides."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    P = np.asarray(points, float)
    single = P.ndim == 1
    P = P.reshape(-1, 3)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("winding_number: V must be (nv, 3) and F (nf, 3)")
    out = np.zeros(len(P))
    if len(F) == 0 or len(P) == 0:
        return float(out[0]) if single and len(out) else out
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    step = max(1, int(max_pairs) // max(1, len(F)))
    for i in range(0, len(P), step):
        p = P[i:i + step, None, :]
        a, b, c = A[None] - p, B[None] - p, C[None] - p
        la, lb, lc = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2), np.linalg.norm(c, axis=2)
        num = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc + np.einsum("ijk,ijk->ij", b, c) * la
               + np.einsum("ijk,ijk->ij", c, a) * lb)
        out[i:i + step] = np.sum(2.0 * np.arctan2(num, den), axis=1) / (4.0 * np.pi)
    return float(out[0]) if single else out


def containment_margin(w):
    """(w - 0.5) / 0.5: positive = inside (more than half of the directions blocked), 1 = fully enclosed"""
    return (np.asarray(w, float) - 0.5) / 0.5 if np.ndim(w) else (float(w) - 0.5) / 0.5


# ------------------------------------------------------------------------------------------------ mesh builders
def _grid(o, u, v, n: int):
    """(n+1)^2 vertices of the parallelogram o + s u + t v, s, t in [0, 1]; 2 n^2 triangles with normal u x v"""
    s = np.linspace(0.0, 1.0, n + 1)
    P = (o[None, None] + s[:, None, None] * u[None, None] + s[None, :, None] * v[None, None]).reshape(-1, 3)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    T = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return P, T


def _merge(parts):
    Vs, Fs, k = [], [], 0
    for P, T in parts:
        Vs.append(P); Fs.append(T + k); k += len(P)
    return np.vstack(Vs), np.vstack(Fs).astype(int)


def open_box_mesh(half_x: float, half_y: float, z0: float, z1: float, n: int = 8, closed: bool = False):
    """Open-top box (floor at z0, four walls up to z1, centred on the z axis), outward normals. closed=True adds the lid
    (a closed box, winding number 1 inside). This is the probe's open_box with the floor at z0 instead of 0."""
    X, Y, Z = np.array([2 * half_x, 0, 0.]), np.array([0, 2 * half_y, 0.]), np.array([0, 0, z1 - z0])
    o = np.array([-half_x, -half_y, z0])
    faces = [(o, Y, X), (o, X, Z), (o + Y, Z, X), (o, Z, Y), (o + X, Y, Z)]      # bottom, front, back, left, right
    if closed:
        faces.append((o + Z, X, Y))                                             # lid, normal +z
    return _merge([_grid(oo, u, v, n) for oo, u, v in faces])


def open_cylinder_mesh(radius: float, z0: float, z1: float, n: int = 32, n_z: int = 4, closed: bool = False):
    """Open-top cylinder (cup / bowl approximated by a straight wall): floor disk at z0 and side wall up to z1, outward
    normals. closed=True adds the lid."""
    th = np.linspace(0.0, TWO_PI, n, endpoint=False)
    ring = np.stack([radius * np.cos(th), radius * np.sin(th), np.zeros(n)], 1)
    zs = np.linspace(z0, z1, n_z + 1)
    W = np.vstack([ring + [0, 0, z] for z in zs])
    T = []
    for k in range(n_z):
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i
            T += [[a, b, c], [a, c, d]]                                       # counter-clockwise from outside
    parts = [(W, np.array(T))]
    for z, up in ((z0, False), (z1, True)) if closed else ((z0, False),):
        C = np.vstack([[0, 0, z], ring + [0, 0, z]])
        if up:
            D = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
        else:
            D = [[0, 1 + (i + 1) % n, 1 + i] for i in range(n)]
        parts.append((C, np.array(D)))
    return _merge(parts)


def mesh_from_spec(spec: dict):
    """(V, F) from a JSON mesh spec (contract args):
      {"kind": "open_box", "half_x", "half_y", "z0", "z1", "n"?, "closed"?}
      {"kind": "open_cylinder", "radius", "z0", "z1", "n"?, "n_z"?, "closed"?}
      {"kind": "vf", "V": [[x, y, z], ...], "F": [[i, j, k], ...]}"""
    k = spec.get("kind")
    if k == "open_box":
        return open_box_mesh(float(spec["half_x"]), float(spec["half_y"]), float(spec["z0"]), float(spec["z1"]),
                             int(spec.get("n", 8)), bool(spec.get("closed", False)))
    if k == "open_cylinder":
        return open_cylinder_mesh(float(spec["radius"]), float(spec["z0"]), float(spec["z1"]), int(spec.get("n", 32)),
                                  int(spec.get("n_z", 4)), bool(spec.get("closed", False)))
    if k == "vf":
        return np.asarray(spec["V"], float), np.asarray(spec["F"], int)
    raise ValueError(f"unknown mesh kind {k!r} (open_box | open_cylinder | vf)")


# ================================================================================================ crossings
@dataclass(frozen=True)
class Crossings:
    """Result of disk_crossings. Unpacks as (net, margin) like the probe's return value.

    net         signed crossings: +1 for each pass through the disk along +normal, -1 against it
    n           unsigned number of crossings (through-and-back: net 0, n 2)
    margin      insertion margin [m]: 0 when net == 0; else the curve length from the crossing to the free end
                (free_end='end': after the last crossing; 'start': before the first; 'both': the smaller of the two)
    arc_before  curve length from the start to the first crossing (nan without crossings)
    arc_after   curve length from the last crossing to the end (nan without crossings)
    length      total curve length
    arcs, signs arc position and sign of every crossing, in curve order"""
    net: int
    n: int
    margin: float
    arc_before: float
    arc_after: float
    length: float
    arcs: tuple = field(default=())
    signs: tuple = field(default=())

    def __iter__(self):
        yield self.net
        yield self.margin

    @property
    def threaded(self) -> bool:
        return self.net != 0


def disk_crossings(polyline, centre, normal, radius: float, free_end: str = "both") -> Crossings:
    """Signed crossings of the polyline (m, 3) through the flat disk {x: (x - centre).normal = 0, |x - centre| <= radius}.

    A segment crosses when its endpoints lie on different sides of the disk's plane (a vertex exactly on the plane
    counts as the + side, so a curve passing through a vertex on the plane is counted once) and the crossing point is
    within `radius` of the centre. For a rigid ring use the radius of its core circle (any value between the hole radius
    and the core radius gives the same count: a rigid curve cannot pass through the tube itself)."""
    if free_end not in FREE_ENDS:
        raise ValueError(f"free_end must be one of {FREE_ENDS}")
    C = np.asarray(polyline, float).reshape(-1, 3)
    n = np.asarray(normal, float).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        raise ValueError("disk_crossings: zero normal")
    n = n / nn
    c = np.asarray(centre, float).reshape(3)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(arc[-1]) if len(arc) else 0.0
    if len(C) < 2:
        return Crossings(0, 0, 0.0, float("nan"), float("nan"), L)
    s = (C - c) @ n
    side = s >= 0.0
    i = np.nonzero(side[:-1] != side[1:])[0]
    if len(i):
        t = s[i] / (s[i] - s[i + 1])
        X = C[i] + t[:, None] * (C[i + 1] - C[i])
        D = X - c
        inplane = np.linalg.norm(D - (D @ n)[:, None] * n[None], axis=1)
        keep = inplane <= float(radius)
        i, t = i[keep], t[keep]
    if not len(i):
        return Crossings(0, 0, 0.0, float("nan"), float("nan"), L)
    signs = np.where(side[i + 1], 1, -1)
    arcs = arc[i] + t * seg[i]
    net = int(signs.sum())
    before, after = float(arcs[0]), float(L - arcs[-1])
    if net == 0:
        margin = 0.0
    else:
        margin = {"both": min(before, after), "start": before, "end": after}[free_end]
    return Crossings(net, int(len(i)), float(margin), before, after, L, tuple(float(a) for a in arcs),
                     tuple(int(x) for x in signs))


# ================================================================================================ wrap angle
def _perp_basis(d: np.ndarray):
    a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(d, a); e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def wrap_angle(polyline, axis_point, axis_dir, eps: float = 1e-12) -> float:
    """Total signed angle [rad] swept around the axis (axis_point, axis_dir) by the projection of the polyline onto the
    plane perpendicular to the axis (right-hand rule about axis_dir). A curve wound twice around a post gives about
    +-4 pi; a curve passing beside it, less than pi in magnitude. Vertices within `eps` of the axis are skipped (their
    angle is undefined); consecutive vertices more than pi apart around the axis are ambiguous (sample the curve
    finer than half a turn per segment)."""
    C = np.asarray(polyline, float).reshape(-1, 3)
    d = np.asarray(axis_dir, float).reshape(3)
    d = d / np.linalg.norm(d)
    Q = C - np.asarray(axis_point, float).reshape(3)
    Q = Q - (Q @ d)[:, None] * d[None]
    e1, e2 = _perp_basis(d)
    x, y = Q @ e1, Q @ e2
    keep = np.hypot(x, y) > eps
    th = np.arctan2(y[keep], x[keep])
    if len(th) < 2:
        return 0.0
    dth = (np.diff(th) + np.pi) % TWO_PI - np.pi
    return float(np.sum(dth))


def wrap_turns(polyline, axis_point, axis_dir) -> float:
    return abs(wrap_angle(polyline, axis_point, axis_dir)) / TWO_PI


# ================================================================================================ poses
def quat_to_matrix(q) -> np.ndarray:
    """rotation matrix of a unit quaternion (w, x, y, z); also accepts (n, 4) -> (n, 3, 3)"""
    q = np.asarray(q, float)
    single = q.ndim == 1
    q = q.reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    return R[0] if single else R


POSE_SUFFIXES = ("px", "py", "pz", "qw", "qx", "qy", "qz")


def pose_signal_names(prefix: str) -> list[str]:
    return [f"{prefix}_{s}" for s in POSE_SUFFIXES]


def pose_series(traces: dict, pose, sl: slice = slice(None)):
    """(R (n, 3, 3), t (n, 3)) for the steps `sl` of the traces; pose = signal prefix | static [p, q] | None.
    For a static pose / None the arrays have length 1 (broadcast over steps)."""
    if pose is None:
        return np.eye(3)[None], np.zeros((1, 3))
    if isinstance(pose, str):
        names = pose_signal_names(pose)
        missing = [k for k in names if k not in traces]
        if missing:
            raise KeyError(f"pose signals missing from the traces: {missing}")
        X = np.stack([np.asarray(traces[k], float)[sl] for k in names], axis=1)
        return quat_to_matrix(X[:, 3:]).reshape(-1, 3, 3), X[:, :3]
    p = np.asarray(pose, float).reshape(-1)
    if p.size != 7:
        raise ValueError("a static pose is [px, py, pz, qw, qx, qy, qz]")
    return quat_to_matrix(p[3:])[None], p[None, :3]


def _n_steps(traces: dict, signal: str, sl: slice) -> int:
    return len(np.asarray(traces[signal], float)[sl])


def _at(R, t, k):
    return R[min(k, len(R) - 1)], t[min(k, len(t) - 1)]


# ================================================================================================ per-step series
def winding_series(traces: dict, signal: str, mesh: dict, object=None, container=None, points=None,
                   points_frame: str = "object", sl: slice = slice(None), **_) -> np.ndarray:
    """per step: min over the query points of the container mesh's winding number"""
    V, F = mesh_from_spec(mesh)
    pts = np.asarray(points if points is not None else [[0.0, 0.0, 0.0]], float).reshape(-1, 3)
    n = _n_steps(traces, signal, sl)
    Ro, to = pose_series(traces, object, sl)
    Rc, tc = pose_series(traces, container, sl)
    out = np.zeros(n)
    for k in range(n):
        R1, t1 = _at(Ro, to, k)
        R2, t2 = _at(Rc, tc, k)
        W = pts @ R1.T + t1 if points_frame == "object" else pts + t1
        local = (W - t2) @ R2                                  # world -> container frame
        out[k] = float(np.min(winding_number(V, F, local)))
    return out


def _crossing_series(traces: dict, signal: str, ring=None, ring_centre=(0.0, 0.0, 0.0), ring_normal=(0.0, 0.0, 1.0),
                     ring_radius: float = 0.0, curve=None, curve_pose=None, free_end: str = "both",
                     sl: slice = slice(None)):
    if curve is None:
        raise ValueError("crossing metrics need args.curve (local polyline points)")
    Cl = np.asarray(curve, float).reshape(-1, 3)
    n = _n_steps(traces, signal, sl)
    Rr, tr = pose_series(traces, ring, sl)
    Rk, tk = pose_series(traces, curve_pose, sl)
    cc, nc = np.asarray(ring_centre, float), np.asarray(ring_normal, float)
    out = []
    for k in range(n):
        R1, t1 = _at(Rr, tr, k)
        R2, t2 = _at(Rk, tk, k)
        out.append(disk_crossings(Cl @ R2.T + t2, R1 @ cc + t1, R1 @ nc, float(ring_radius), free_end=free_end))
    return out


def crossings_series(traces: dict, signal: str, absolute: bool = True, sl: slice = slice(None), **kw) -> np.ndarray:
    res = _crossing_series(traces, signal, sl=sl, **{k: v for k, v in kw.items() if k != "window"})
    net = np.array([r.net for r in res], float)
    return np.abs(net) if absolute else net


def insertion_series(traces: dict, signal: str, sl: slice = slice(None), **kw) -> np.ndarray:
    res = _crossing_series(traces, signal, sl=sl, **{k: v for k, v in kw.items() if k != "window"})
    return np.array([r.margin for r in res], float)


def wrap_turns_series(traces: dict, signal: str, curve=None, curve_pose=None, axis_point=(0.0, 0.0, 0.0),
                      axis_dir=(0.0, 0.0, 1.0), axis_pose=None, sl: slice = slice(None), **_) -> np.ndarray:
    if curve is None:
        raise ValueError("wrap metrics need args.curve (local polyline points)")
    Cl = np.asarray(curve, float).reshape(-1, 3)
    n = _n_steps(traces, signal, sl)
    Rk, tk = pose_series(traces, curve_pose, sl)
    Ra, ta = pose_series(traces, axis_pose, sl)
    ap, ad = np.asarray(axis_point, float), np.asarray(axis_dir, float)
    out = np.zeros(n)
    for k in range(n):
        R1, t1 = _at(Rk, tk, k)
        R2, t2 = _at(Ra, ta, k)
        out[k] = wrap_turns(Cl @ R1.T + t1, R2 @ ap + t2, R2 @ ad)
    return out


SERIES = {"winding": winding_series, "crossings": crossings_series, "insertion": insertion_series,
          "wrap_turns": wrap_turns_series}


def series(quantity: str, traces: dict, signal: str, window: int | None = None, **args) -> np.ndarray:
    """per-step values of a topological quantity over the whole trace (window=None) or its last `window` steps"""
    sl = slice(-int(window), None) if window else slice(None)
    return SERIES[quantity](traces, signal, sl=sl, **args)


# ================================================================================================ registered metrics
# ---- geometry (end-state arrays)
@metric("topology.winding_number")
def m_winding_number(V, F, points, reduce: str = "min", **_) -> float:
    """winding number of the mesh at the points, reduced over the points: min | max | mean | frac_inside (w > 0.5)"""
    w = np.atleast_1d(winding_number(V, F, points))
    if reduce == "frac_inside":
        return float(np.mean(w > 0.5))
    return float({"min": np.min, "max": np.max, "mean": np.mean}[reduce](w))


@metric("topology.containment_margin")
def m_containment_margin(V, F, points, reduce: str = "min", **_) -> float:
    """(w - 0.5) / 0.5 of the reduced winding number (positive = inside)"""
    return float(containment_margin(m_winding_number(V, F, points, reduce=reduce)))


@metric("topology.disk_crossings")
def m_disk_crossings(polyline, centre, normal, radius, absolute: bool = True, **_) -> float:
    net = disk_crossings(polyline, centre, normal, radius).net
    return float(abs(net) if absolute else net)


@metric("topology.insertion_margin")
def m_insertion_margin(polyline, centre, normal, radius, free_end: str = "both", **_) -> float:
    return float(disk_crossings(polyline, centre, normal, radius, free_end=free_end).margin)


@metric("topology.wrap_angle")
def m_wrap_angle(polyline, axis_point, axis_dir, **_) -> float:
    """signed wrap angle in DEGREES"""
    return float(np.degrees(wrap_angle(polyline, axis_point, axis_dir)))


@metric("topology.wrap_turns")
def m_wrap_turns(polyline, axis_point, axis_dir, **_) -> float:
    return float(wrap_turns(polyline, axis_point, axis_dir))


# ---- traces (per-step topology from recorded poses)
def _window_min(quantity):
    def f(traces: dict, signal: str, window: int = 10, **args) -> float:
        return float(np.min(series(quantity, traces, signal, window=window, **args)))
    f.__doc__ = f"worst (smallest) per-step {quantity} over the last `window` steps (end-state term)"
    return f


def _event_min(quantity):
    def f(traces: dict, signal: str, **args) -> float:
        args.pop("window", None)
        return float(np.min(series(quantity, traces, signal, window=None, **args)))
    f.__doc__ = f"worst (smallest) per-step {quantity} over the whole trace given (event term)"
    return f


TRACE_END_STATE = ("trace.winding_window_min", "trace.crossings_window_min", "trace.insertion_window_min",
                   "trace.wrap_turns_window_min")
TRACE_EVENT = ("trace.winding_min", "trace.crossings_min", "trace.insertion_min", "trace.wrap_turns_min")
for _q in ("winding", "crossings", "insertion", "wrap_turns"):
    metric(f"trace.{_q}_window_min")(_window_min(_q))
    metric(f"trace.{_q}_min")(_event_min(_q))
