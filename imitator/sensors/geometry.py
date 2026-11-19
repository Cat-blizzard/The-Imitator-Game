"""Gap queries for the tactile pads: ray casts from taxel points along the pad normal against the collision geometry of
the contacting bodies (Warp kernels; the same code runs on CUDA and on the CPU).

Geometry comes straight from the simulator's collision shapes (SAPIEN / PhysX), so the gap field is computed against
exactly what the solver collides with:

  box       half extents                               exact slab test
  sphere    radius                                     exact
  capsule   radius, half length (axis = local x)       exact (finite cylinder + two spheres)
  cylinder  radius, half length (axis = local x)       exact (flat caps)
  plane     PhysX plane: x = 0, solid half space x < 0 exact
  convex    convex hull (PhysX cooks dynamic meshes into hulls): exact half-space clipping over the hull's face planes,
            so a ray whose origin is inside the hull still reports its entry point (negative t)
  trimesh   triangle mesh (static / kinematic geometry): first hit of wp.mesh_query_ray

A ray returns the entry parameter t of the first shape of the body it enters (BIG = miss). Rays start `backoff` metres
behind the pad surface, so the gap = t - backoff and penetrations up to `backoff` are measured, not lost.

Tables (built once per scene by GeomBank, poses updated every step):
  shape table   type, local pose (in the body frame), params, convex plane range, mesh id
  body table    one row per (simulated body, env): shape range; its world pose is written every step
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

BIG = 1.0e9
BOX, SPHERE, CAPSULE, CYLINDER, PLANE, CONVEX, TRIMESH = 0, 1, 2, 3, 4, 5, 6
TYPE_NAMES = {BOX: "box", SPHERE: "sphere", CAPSULE: "capsule", CYLINDER: "cylinder", PLANE: "plane", CONVEX: "convex",
              TRIMESH: "trimesh"}

_wp = None
_KERNELS = {}


def _warp():
    global _wp
    if _wp is None:
        import warp as wp
        wp.config.quiet = True
        wp.init()
        _wp = wp
        globals()["wp"] = wp        # kernels are defined with string annotations (from __future__): resolve 'wp' here
    return _wp


def wp_device(torch_device) -> str:
    """'cuda:0' / 'cpu' from a torch device"""
    import torch
    d = torch.device(torch_device)
    if d.type == "cuda":
        return f"cuda:{d.index if d.index is not None else torch.cuda.current_device()}"
    return "cpu"


# ------------------------------------------------------------------------------------------------ kernels
def _build_kernels():
    if _KERNELS:
        return _KERNELS
    wp = _warp()
    big = float(BIG)

    @wp.func
    def ray_box(o: wp.vec3, d: wp.vec3, h: wp.vec3) -> float:
        tmin = float(-big)
        tmax = float(big)
        miss = int(0)
        for a in range(3):
            if wp.abs(d[a]) < 1.0e-12:
                if o[a] < -h[a] or o[a] > h[a]:
                    miss = 1
            else:
                t1 = (-h[a] - o[a]) / d[a]
                t2 = (h[a] - o[a]) / d[a]
                tmin = wp.max(tmin, wp.min(t1, t2))
                tmax = wp.min(tmax, wp.max(t1, t2))
        res = float(big)
        if miss == 0 and tmax >= tmin and tmax >= 0.0:
            res = tmin
        return res

    @wp.func
    def ray_sphere(o: wp.vec3, d: wp.vec3, c: wp.vec3, r: float) -> float:
        oc = o - c
        b = wp.dot(oc, d)
        cc = wp.dot(oc, oc) - r * r
        disc = b * b - cc
        res = float(big)
        if disc >= 0.0:
            s = wp.sqrt(disc)
            if -b + s >= 0.0:
                res = -b - s
        return res

    @wp.func
    def ray_cyl(o: wp.vec3, d: wp.vec3, r: float, hl: float) -> float:
        # finite cylinder along x with flat caps
        a = d[1] * d[1] + d[2] * d[2]
        b = o[1] * d[1] + o[2] * d[2]
        c = o[1] * o[1] + o[2] * o[2] - r * r
        tr0 = float(-big)
        tr1 = float(big)
        miss = int(0)
        if a < 1.0e-12:
            if c > 0.0:
                miss = 1
        else:
            disc = b * b - a * c
            if disc < 0.0:
                miss = 1
            else:
                s = wp.sqrt(disc)
                tr0 = (-b - s) / a
                tr1 = (-b + s) / a
        tx0 = float(-big)
        tx1 = float(big)
        if wp.abs(d[0]) < 1.0e-12:
            if o[0] < -hl or o[0] > hl:
                miss = 1
        else:
            t1 = (-hl - o[0]) / d[0]
            t2 = (hl - o[0]) / d[0]
            tx0 = wp.min(t1, t2)
            tx1 = wp.max(t1, t2)
        tn = wp.max(tr0, tx0)
        tf = wp.min(tr1, tx1)
        res = float(big)
        if miss == 0 and tf >= tn and tf >= 0.0:
            res = tn
        return res

    @wp.func
    def ray_capsule(o: wp.vec3, d: wp.vec3, r: float, hl: float) -> float:
        t = ray_cyl(o, d, r, hl)
        t = wp.min(t, ray_sphere(o, d, wp.vec3(hl, 0.0, 0.0), r))
        t = wp.min(t, ray_sphere(o, d, wp.vec3(-hl, 0.0, 0.0), r))
        return t

    @wp.func
    def ray_plane(o: wp.vec3, d: wp.vec3) -> float:
        # PhysX plane: solid half space x < 0
        res = float(big)
        if d[0] < -1.0e-12:
            res = -o[0] / d[0]      # entry parameter; negative when the origin is already inside the solid
        return res

    @wp.func
    def ray_convex(o: wp.vec3, d: wp.vec3, planes: wp.array(dtype=wp.vec4), start: int, count: int) -> float:
        tin = float(-big)
        tout = float(big)
        miss = int(0)
        for j in range(start, start + count):
            pl = planes[j]
            n = wp.vec3(pl[0], pl[1], pl[2])
            den = wp.dot(n, d)
            num = pl[3] - wp.dot(n, o)
            if wp.abs(den) < 1.0e-12:
                if num < 0.0:
                    miss = 1
            else:
                t = num / den
                if den < 0.0:
                    tin = wp.max(tin, t)
                else:
                    tout = wp.min(tout, t)
        res = float(big)
        if miss == 0 and tout >= tin and tout >= 0.0:
            res = tin
        return res

    @wp.kernel
    def pad_rays(pad_pos: wp.array(dtype=wp.vec3), pad_rot: wp.array(dtype=wp.quat),       # (N*P) world pose of the pad frame
                 sub_local: wp.array(dtype=wp.vec3),                                       # (P*S) ray origin in the pad frame
                 body_sel: wp.array(dtype=wp.int32),                                       # (N*P*K) body-table row or -1
                 body_pos: wp.array(dtype=wp.vec3), body_rot: wp.array(dtype=wp.quat),
                 body_shape_start: wp.array(dtype=wp.int32), body_shape_count: wp.array(dtype=wp.int32),
                 shape_type: wp.array(dtype=wp.int32), shape_pos: wp.array(dtype=wp.vec3), shape_rot: wp.array(dtype=wp.quat),
                 shape_param: wp.array(dtype=wp.vec4), shape_plane_start: wp.array(dtype=wp.int32),
                 shape_plane_count: wp.array(dtype=wp.int32), shape_mesh: wp.array(dtype=wp.uint64),
                 planes: wp.array(dtype=wp.vec4), P: int, K: int, S: int, max_t: float, out_t: wp.array(dtype=float)):
        i = wp.tid()
        s = i % S
        k = (i // S) % K
        np_ = i // (S * K)                      # = n * P + p
        p = np_ % P
        b = body_sel[np_ * K + k]
        best = float(big)
        if b >= 0:
            X_pad = wp.transform(pad_pos[np_], pad_rot[np_])
            o_w = wp.transform_point(X_pad, sub_local[p * S + s])
            d_w = wp.quat_rotate(pad_rot[np_], wp.vec3(0.0, 0.0, 1.0))
            X_b = wp.transform(body_pos[b], body_rot[b])
            st = body_shape_start[b]
            for j in range(st, st + body_shape_count[b]):
                X_s = wp.transform_multiply(X_b, wp.transform(shape_pos[j], shape_rot[j]))
                Xi = wp.transform_inverse(X_s)
                o = wp.transform_point(Xi, o_w)
                d = wp.transform_vector(Xi, d_w)
                ty = shape_type[j]
                prm = shape_param[j]
                t = float(big)
                if ty == 0:
                    t = ray_box(o, d, wp.vec3(prm[0], prm[1], prm[2]))
                elif ty == 1:
                    t = ray_sphere(o, d, wp.vec3(0.0, 0.0, 0.0), prm[0])
                elif ty == 2:
                    t = ray_capsule(o, d, prm[0], prm[1])
                elif ty == 3:
                    t = ray_cyl(o, d, prm[0], prm[1])
                elif ty == 4:
                    t = ray_plane(o, d)
                elif ty == 5:
                    t = ray_convex(o, d, planes, shape_plane_start[j], shape_plane_count[j])
                elif ty == 6:
                    q = wp.mesh_query_ray(shape_mesh[j], o, d, max_t)
                    if q.result:
                        t = q.t
                best = wp.min(best, t)
        out_t[i] = best

    _KERNELS["pad_rays"] = pad_rays
    return _KERNELS


# ------------------------------------------------------------------------------------------------ shapes
@dataclass
class Shape:
    type: int
    pos: np.ndarray                 # local pose in the body frame
    quat_wxyz: np.ndarray
    param: np.ndarray               # (4,)
    planes: np.ndarray | None = None        # convex: (F, 4) n . x <= d (outward unit normals)
    verts: np.ndarray | None = None         # trimesh (scaled)
    tris: np.ndarray | None = None

    def key(self) -> str:
        h = hashlib.sha1()
        h.update(np.int32(self.type).tobytes())
        for a in (self.pos, self.quat_wxyz, self.param):
            h.update(np.round(np.asarray(a, np.float64), 7).tobytes())
        for a in (self.planes, self.verts, self.tris):
            if a is not None:
                h.update(np.ascontiguousarray(np.round(np.asarray(a, np.float64), 7)).tobytes())
        return h.hexdigest()


def hull_planes(V: np.ndarray, T: np.ndarray | None = None) -> np.ndarray:
    """outward face planes (n, d) with n . x <= d of the convex hull of V (merged coplanar faces)"""
    import trimesh
    h = trimesh.Trimesh(np.asarray(V, float), process=True).convex_hull
    n = np.asarray(h.face_normals, float)
    d = np.einsum("ij,ij->i", n, np.asarray(h.vertices, float)[h.faces[:, 0]])
    P = np.c_[n, d]
    key = np.round(P / max(1e-9, float(np.abs(V).max())) * 1e6).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


def shapes_of_component(comp) -> list[Shape]:
    """collision shapes of a SAPIEN PhysX rigid component (actor body or articulation link), in the body frame"""
    import sapien.physx as px
    out = []
    for s in comp.get_collision_shapes():
        lp = s.get_local_pose()
        pos = np.asarray(lp.p, float)
        q = np.asarray(lp.q, float)
        prm = np.zeros(4)
        if isinstance(s, px.PhysxCollisionShapeBox):
            prm[:3] = np.asarray(s.get_half_size(), float)
            out.append(Shape(BOX, pos, q, prm))
        elif isinstance(s, px.PhysxCollisionShapeSphere):
            prm[0] = float(s.get_radius())
            out.append(Shape(SPHERE, pos, q, prm))
        elif isinstance(s, px.PhysxCollisionShapeCapsule):
            prm[0], prm[1] = float(s.get_radius()), float(s.get_half_length())
            out.append(Shape(CAPSULE, pos, q, prm))
        elif isinstance(s, px.PhysxCollisionShapeCylinder):
            prm[0], prm[1] = float(s.get_radius()), float(s.get_half_length())
            out.append(Shape(CYLINDER, pos, q, prm))
        elif isinstance(s, px.PhysxCollisionShapePlane):
            out.append(Shape(PLANE, pos, q, prm))
        elif isinstance(s, px.PhysxCollisionShapeConvexMesh):
            V = np.asarray(s.get_vertices(), float) * np.asarray(s.get_scale(), float)
            out.append(Shape(CONVEX, pos, q, prm, planes=hull_planes(V)))
        elif isinstance(s, px.PhysxCollisionShapeTriangleMesh):
            V = np.asarray(s.get_vertices(), float) * np.asarray(s.get_scale(), float)
            out.append(Shape(TRIMESH, pos, q, prm, verts=V, tris=np.asarray(s.get_triangles(), np.int64)))
    return out


# ------------------------------------------------------------------------------------------------ bank
class GeomBank:
    """Shape table + body table on one device. Bodies are registered once (shapes deduplicated); each body row gets a
    world pose per step (set_poses). cast(...) runs the pad-ray kernel."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._shape_rows: list[Shape] = []
        self._shape_index: dict[str, int] = {}
        self._body_ranges: list[tuple[int, int]] = []
        self._body_keys: dict = {}
        self._meshes: list = []
        self._dirty = True
        self.arrays = None

    # ---------------------------------------------------------------- registration
    def add_body(self, shapes: list[Shape], key=None) -> int:
        """register a body (its list of shapes); returns the body-table row. Same key -> same row."""
        if key is not None and key in self._body_keys:
            return self._body_keys[key]
        start = len(self._shape_rows)
        # a body's shapes are stored contiguously (shared shapes are duplicated rows; cheap)
        for s in shapes:
            self._shape_rows.append(s)
        row = len(self._body_ranges)
        self._body_ranges.append((start, len(shapes)))
        if key is not None:
            self._body_keys[key] = row
        self._dirty = True
        return row

    @property
    def n_bodies(self) -> int:
        return len(self._body_ranges)

    def finalize(self):
        if not self._dirty:
            return self.arrays
        wp = _warp()
        _build_kernels()
        dev = self.device
        rows = self._shape_rows
        n = max(1, len(rows))
        st = np.zeros(n, np.int32); sp = np.zeros((n, 3), np.float32); sq = np.zeros((n, 4), np.float32)
        sq[:, 3] = 1.0
        prm = np.zeros((n, 4), np.float32); pls = np.zeros(n, np.int32); plc = np.zeros(n, np.int32)
        mid = np.zeros(n, np.uint64)
        planes = []
        npl = 0
        self._meshes = []
        for i, s in enumerate(rows):
            st[i] = s.type
            sp[i] = s.pos
            w, x, y, z = [float(v) for v in s.quat_wxyz]
            sq[i] = (x, y, z, w)                                    # warp quaternions are (x, y, z, w)
            prm[i] = s.param
            if s.type == CONVEX:
                pls[i], plc[i] = npl, len(s.planes)
                planes.append(np.asarray(s.planes, np.float32)); npl += len(s.planes)
            elif s.type == TRIMESH:
                m = wp.Mesh(points=wp.array(np.asarray(s.verts, np.float32), dtype=wp.vec3, device=dev),
                            indices=wp.array(np.asarray(s.tris, np.int32).reshape(-1), dtype=wp.int32, device=dev))
                self._meshes.append(m)
                mid[i] = m.id
        planes = np.vstack(planes) if planes else np.zeros((1, 4), np.float32)
        nb = max(1, self.n_bodies)
        bs = np.zeros(nb, np.int32); bc = np.zeros(nb, np.int32)
        for i, (a, c) in enumerate(self._body_ranges):
            bs[i], bc[i] = a, c
        A = dict(shape_type=wp.array(st, dtype=wp.int32, device=dev), shape_pos=wp.array(sp, dtype=wp.vec3, device=dev),
                 shape_rot=wp.array(sq, dtype=wp.quat, device=dev), shape_param=wp.array(prm, dtype=wp.vec4, device=dev),
                 shape_plane_start=wp.array(pls, dtype=wp.int32, device=dev),
                 shape_plane_count=wp.array(plc, dtype=wp.int32, device=dev),
                 shape_mesh=wp.array(mid, dtype=wp.uint64, device=dev), planes=wp.array(planes, dtype=wp.vec4, device=dev),
                 body_shape_start=wp.array(bs, dtype=wp.int32, device=dev),
                 body_shape_count=wp.array(bc, dtype=wp.int32, device=dev))
        self.arrays = A
        self._dirty = False
        return A

    # ---------------------------------------------------------------- query
    def cast(self, pad_pos, pad_quat_wxyz, sub_local, body_sel, body_pos, body_quat_wxyz, max_t: float = 1.0):
        """Ray cast for every (env-pad, slot, sub-point).
        pad_pos (M, 3), pad_quat_wxyz (M, 4) torch, M = N*P (row m = n*P + p); sub_local (P, S, 3) ray origins in the pad
        frame (rays go along +z of the pad frame); body_sel (M, K) int body-table rows (-1 none); body_pos (B, 3),
        body_quat_wxyz (B, 4) world poses of every body-table row. Returns t (M, K, S) torch float32 (BIG = miss)."""
        import torch
        wp = _warp()
        A = self.finalize()
        K_ = _build_kernels()
        M = pad_pos.shape[0]
        P, S = sub_local.shape[0], sub_local.shape[1]
        K = body_sel.shape[1]
        dev = pad_pos.device
        pq = torch.cat([pad_quat_wxyz[:, 1:], pad_quat_wxyz[:, :1]], 1).contiguous().float()
        bq = torch.cat([body_quat_wxyz[:, 1:], body_quat_wxyz[:, :1]], 1).contiguous().float()
        out = torch.empty((M, K, S), dtype=torch.float32, device=dev)
        if body_pos.shape[0] == 0:
            out.fill_(BIG)
            return out
        wdev = wp_device(dev)
        # launch on torch's CURRENT stream: Warp otherwise uses its own stream and the torch ops that read `out` could race
        stream = wp.stream_from_torch(torch.cuda.current_stream(dev)) if dev.type == "cuda" else None
        wp.launch(K_["pad_rays"], dim=M * K * S, device=wdev, stream=stream, inputs=[
            wp.from_torch(pad_pos.contiguous().float(), dtype=wp.vec3), wp.from_torch(pq, dtype=wp.quat),
            wp.from_torch(sub_local.reshape(-1, 3).contiguous().float(), dtype=wp.vec3),
            wp.from_torch(body_sel.reshape(-1).to(torch.int32).contiguous(), dtype=wp.int32),
            wp.from_torch(body_pos.contiguous().float(), dtype=wp.vec3), wp.from_torch(bq, dtype=wp.quat),
            A["body_shape_start"], A["body_shape_count"], A["shape_type"], A["shape_pos"], A["shape_rot"], A["shape_param"],
            A["shape_plane_start"], A["shape_plane_count"], A["shape_mesh"], A["planes"], int(P), int(K), int(S),
            float(max_t), wp.from_torch(out.reshape(-1), dtype=wp.float32)])
        return out


def to_device_bank(bank: GeomBank, device: str) -> GeomBank:
    if bank.device == device:
        return bank
    nb = GeomBank(device)
    nb._shape_rows, nb._body_ranges, nb._body_keys = list(bank._shape_rows), list(bank._body_ranges), dict(bank._body_keys)
    nb._dirty = True
    return nb
