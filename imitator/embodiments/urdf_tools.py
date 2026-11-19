"""URDF utilities for the embodiment library (pure Python + numpy + trimesh; mplib only for SRDF generation).

Why this exists
  * Source URDFs (RoboTwin 2.0, ManiSkill assets, vendor packages) are read-only. Every embodiment record points at a
    DERIVED URDF written by `derive_urdf` into imitator/assets/embodiments/<name>/: mesh paths relative to the derived URDF (so neither
    SAPIEN nor mplib has to resolve or rewrite `package://`, and mplib never writes files next to a source URDF),
    joint limits corrected (RoboTwin's ARX-X5 and aloha-agilex carry +-10 rad placeholders), camera links dropped,
    an explicit TCP link added (ManiSkill convention: z = approach, y = finger closing axis).
  * SRDFs: SAPIEN turns every non-parent-child <disable_collisions> pair into ONE bit of collision group 2 (max 31 per
    articulation), and two arms loaded from SRDFs share the same bit numbers, which silently disables collisions between
    the arms (see dual.py). So we write two SRDFs per URDF:
        <name>.srdf        minimal (pairs that are in contact at the home pose or in >= 98 % of random samples);
                           SAPIEN picks it up automatically (same stem as the URDF)
        <name>_mplib.srdf  full (minimal + pairs never in contact in the samples); planning only.
  * A small forward-kinematics + collision-geometry reader used to place TCP links, compute finger pad geometry and
    the measured gripper opening (registry.py), independent of the simulator.
"""
from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------------------------ math
def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = [float(v) for v in rpy]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """inverse of rpy_to_matrix (URDF fixed-axis XYZ)."""
    sy = -R[2, 0]
    if abs(sy) < 1 - 1e-9:
        p = np.arcsin(sy)
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock
        p = np.pi / 2 * np.sign(sy)
        r = np.arctan2(-R[1, 2], R[1, 1])
        y = 0.0
    return float(r), float(p), float(y)


def make_T(R=None, p=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def axis_angle(axis, angle) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _origin(el) -> np.ndarray:
    o = el.find("origin") if el is not None else None
    if o is None:
        return np.eye(4)
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return make_T(rpy_to_matrix(rpy), xyz)


# ------------------------------------------------------------------------------------------------ model
@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    T: np.ndarray                     # parent link frame -> joint frame at q = 0
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None
    mimic: tuple | None = None        # (joint, multiplier, offset)


@dataclass
class Geom:
    kind: str                         # mesh | box | sphere | cylinder
    T: np.ndarray                     # link frame -> geometry frame
    params: dict = field(default_factory=dict)


@dataclass
class URDFModel:
    path: str
    name: str
    links: list[str]
    joints: dict[str, Joint]
    collisions: dict[str, list[Geom]]
    root: str

    def child_joints(self, link: str) -> list[Joint]:
        return [j for j in self.joints.values() if j.parent == link]

    def parent_joint(self, link: str) -> Joint | None:
        for j in self.joints.values():
            if j.child == link:
                return j
        return None

    @property
    def active_joints(self) -> list[str]:
        """movable joints in depth-first order from the root (SAPIEN's ordering for a tree)"""
        out = []

        def visit(link):
            for j in self.child_joints(link):
                if j.type not in ("fixed",):
                    out.append(j.name)
                visit(j.child)
        visit(self.root)
        return out

    def fk(self, q: dict[str, float] | None = None, base: np.ndarray | None = None) -> dict[str, np.ndarray]:
        q = dict(q or {})
        for j in self.joints.values():                     # mimic joints follow their leader unless given
            if j.mimic and j.name not in q and j.mimic[0] in q:
                q[j.name] = q[j.mimic[0]] * j.mimic[1] + j.mimic[2]
        Ts = {self.root: np.eye(4) if base is None else np.asarray(base, float)}
        stack = [self.root]
        while stack:
            link = stack.pop()
            for j in self.child_joints(link):
                v = float(q.get(j.name, 0.0))
                M = np.eye(4)
                if j.type in ("revolute", "continuous"):
                    M[:3, :3] = axis_angle(j.axis, v)
                elif j.type == "prismatic":
                    M[:3, 3] = np.asarray(j.axis, float) / (np.linalg.norm(j.axis) + 1e-12) * v
                Ts[j.child] = Ts[link] @ j.T @ M
                stack.append(j.child)
        return Ts

    def collision_points(self, link: str, n_prim: int = 12) -> np.ndarray:
        """collision geometry vertices in the LINK frame (mesh vertices / box corners / sampled primitives)"""
        pts = []
        for g in self.collisions.get(link, []):
            if g.kind == "mesh":
                V = load_mesh_vertices(g.params["filename"]) * np.asarray(g.params.get("scale", [1, 1, 1]), float)
            elif g.kind == "box":
                h = np.asarray(g.params["size"], float) / 2
                V = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            elif g.kind == "sphere":
                r = g.params["radius"]
                d = np.random.default_rng(0).normal(size=(200, 3)); V = r * d / np.linalg.norm(d, axis=1, keepdims=True)
            elif g.kind == "cylinder":
                r, L = g.params["radius"], g.params["length"]
                a = np.linspace(0, 2 * np.pi, n_prim, endpoint=False)
                V = np.vstack([np.c_[r * np.cos(a), r * np.sin(a), np.full_like(a, z)] for z in (-L / 2, L / 2)])
            else:
                continue
            pts.append((g.T[:3, :3] @ V.T).T + g.T[:3, 3])
        return np.vstack(pts) if pts else np.zeros((0, 3))


_MESH_CACHE: dict[str, np.ndarray] = {}


def load_mesh_vertices(filename: str) -> np.ndarray:
    if filename not in _MESH_CACHE:
        import trimesh
        src = filename
        if os.path.exists(filename + ".convex.stl"):        # ManiSkill ships hulls next to meshes (also for .dae)
            src = filename + ".convex.stl"
        m = trimesh.load(src, force="mesh", process=False)
        _MESH_CACHE[filename] = np.asarray(m.vertices, float)
    return _MESH_CACHE[filename]


def resolve_file(filename: str, urdf_dir: str, package_dirs=(), path_map: dict | None = None) -> str:
    """Absolute path of a URDF resource (relative path or package://pkg/...; same search as SAPIEN's loader).
    path_map {prefix: replacement directory} is applied first (e.g. ManiSkill's xarm6_allegro URDFs point at
    ../allegro/, which lives in the mani_skill package, not next to the downloaded xarm6 folder)."""
    for k, v in (path_map or {}).items():
        if filename.startswith(k):
            return str(Path(v) / filename[len(k):])
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        cands = [Path(d) / rest for d in package_dirs]
        p = Path(urdf_dir).absolute()
        while True:
            cands.append(p / rest)
            cands.append(p / "/".join(rest.split("/")[1:]))      # package name dropped
            if p == Path("/"):
                break
            p = p.parent
        for c in cands:
            if c.is_file():
                return str(c)
        return filename
    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    p = Path(filename) if os.path.isabs(filename) else Path(urdf_dir) / filename
    return str(p.resolve()) if p.exists() else str(p)


def load_urdf(path: str, package_dirs=(), path_map: dict | None = None) -> URDFModel:
    root = ET.parse(path).getroot()
    d = os.path.dirname(os.path.abspath(path))
    links = [l.get("name") for l in root.findall("link")]
    joints = {}
    for j in root.findall("joint"):
        ax = j.find("axis")
        lim = j.find("limit")
        mim = j.find("mimic")
        joints[j.get("name")] = Joint(
            name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            T=_origin(j), axis=np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]),
            lower=float(lim.get("lower")) if lim is not None and lim.get("lower") is not None else None,
            upper=float(lim.get("upper")) if lim is not None and lim.get("upper") is not None else None,
            mimic=(mim.get("joint"), float(mim.get("multiplier", 1.0)), float(mim.get("offset", 0.0))) if mim is not None else None)
    cols = {}
    for l in root.findall("link"):
        gs = []
        for c in l.findall("collision"):
            g = c.find("geometry")
            if g is None or len(g) == 0:
                continue
            e = g[0]
            T = _origin(c)
            if e.tag == "mesh":
                sc = [float(v) for v in e.get("scale", "1 1 1").split()]
                gs.append(Geom("mesh", T, dict(filename=resolve_file(e.get("filename"), d, package_dirs, path_map), scale=sc)))
            elif e.tag == "box":
                gs.append(Geom("box", T, dict(size=[float(v) for v in e.get("size").split()])))
            elif e.tag == "sphere":
                gs.append(Geom("sphere", T, dict(radius=float(e.get("radius")))))
            elif e.tag == "cylinder":
                gs.append(Geom("cylinder", T, dict(radius=float(e.get("radius")), length=float(e.get("length")))))
        cols[l.get("name")] = gs
    children = {j.child for j in joints.values()}
    roots = [l for l in links if l not in children]
    return URDFModel(path=os.path.abspath(path), name=root.get("name"), links=links, joints=joints, collisions=cols,
                     root=roots[0])


# ------------------------------------------------------------------------------------------------ derive
@dataclass
class ExtraLink:
    name: str
    parent: str
    xyz: tuple = (0.0, 0.0, 0.0)
    rpy: tuple = (0.0, 0.0, 0.0)


def _rel_to(path: str, out: str) -> str:
    """a resolved resource path, relative to the directory of the URDF being written (keeps the repo relocatable)"""
    if not os.path.isabs(path):
        return path
    return os.path.relpath(path, os.path.dirname(os.path.abspath(out)))


def derive_urdf(src: str, out: str, *, joint_limits: dict | None = None, drop_links=(), extra_links=(),
                robot_name: str | None = None, package_dirs=(), velocity_limits: dict | None = None,
                effort_limits: dict | None = None, mimic: dict | None = None, notes: str = "",
                path_map: dict | None = None) -> str:
    """Write a derived URDF (see module docstring). Returns `out`.

    joint_limits   {joint: (lower, upper)} overrides
    drop_links     links removed together with their whole subtree and the joints leading to them
    extra_links    ExtraLink(name, parent, xyz, rpy) appended as fixed children (empty links: TCP / reference frames)
    mimic          {joint: (leader, multiplier, offset)} mimic tags to add (kinematic info for planners; ManiSkill
                   controllers enforce the coupling)"""
    tree = ET.parse(src)
    root = tree.getroot()
    d = os.path.dirname(os.path.abspath(src))
    if robot_name:
        root.set("name", robot_name)
    # drop subtrees
    kids: dict[str, list] = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    to_drop = set()
    stack = list(drop_links)
    while stack:
        l = stack.pop()
        to_drop.add(l)
        for j in kids.get(l, []):
            stack.append(j.find("child").get("link"))
    for j in list(root.findall("joint")):
        if j.find("child").get("link") in to_drop or j.find("parent").get("link") in to_drop:
            root.remove(j)
    for l in list(root.findall("link")):
        if l.get("name") in to_drop:
            root.remove(l)
    # absolute resource paths
    for tag in ("mesh", "texture"):
        for e in root.iter(tag):
            if e.get("filename"):
                e.set("filename", _rel_to(resolve_file(e.get("filename"), d, package_dirs, path_map), out))
    # limits
    for j in root.findall("joint"):
        n = j.get("name")
        lim = j.find("limit")
        if joint_limits and n in joint_limits:
            if lim is None:
                lim = ET.SubElement(j, "limit", dict(effort="100", velocity="3"))
            lim.set("lower", repr(float(joint_limits[n][0]))); lim.set("upper", repr(float(joint_limits[n][1])))
        if velocity_limits and n in velocity_limits and lim is not None:
            lim.set("velocity", repr(float(velocity_limits[n])))
        if effort_limits and n in effort_limits and lim is not None:
            lim.set("effort", repr(float(effort_limits[n])))
        if mimic and n in mimic and j.find("mimic") is None:
            lead, mul, off = mimic[n]
            ET.SubElement(j, "mimic", dict(joint=lead, multiplier=repr(float(mul)), offset=repr(float(off))))
    # extra links
    for x in extra_links:
        ET.SubElement(root, "link", dict(name=x.name))
        j = ET.SubElement(root, "joint", dict(name=f"{x.name}_joint", type="fixed"))
        ET.SubElement(j, "parent", dict(link=x.parent))
        ET.SubElement(j, "child", dict(link=x.name))
        ET.SubElement(j, "origin", dict(xyz=" ".join(repr(float(v)) for v in x.xyz), rpy=" ".join(repr(float(v)) for v in x.rpy)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ET.indent(tree, space="  ")
    header = f"<!-- derived by imitator.embodiments.urdf_tools from {_rel_to(os.path.abspath(src), out)} {notes} -->\n"
    with open(out, "w") as f:
        f.write('<?xml version="1.0"?>\n' + header + ET.tostring(root, encoding="unicode"))
    return out


def merge_urdfs(arm_src: str, hand_src: str, out: str, *, mount_parent: str, hand_root: str, xyz=(0, 0, 0), rpy=(0, 0, 0),
                hand_prefix: str = "", robot_name: str = "arm_hand", package_dirs=()) -> str:
    """Attach `hand_src` (its root link `hand_root`) to `mount_parent` of `arm_src` with a fixed joint (arm + hand)."""
    a = ET.parse(arm_src).getroot(); h = ET.parse(hand_src).getroot()
    da, dh = os.path.dirname(os.path.abspath(arm_src)), os.path.dirname(os.path.abspath(hand_src))
    for root, d in ((a, da), (h, dh)):
        for tag in ("mesh", "texture"):
            for e in root.iter(tag):
                if e.get("filename"):
                    e.set("filename", _rel_to(resolve_file(e.get("filename"), d, package_dirs), out))
    out_root = ET.Element("robot", dict(name=robot_name))
    for el in list(a):
        out_root.append(copy.deepcopy(el))
    for el in list(h):
        el = copy.deepcopy(el)
        if hand_prefix and el.tag in ("link", "joint"):
            el.set("name", hand_prefix + el.get("name"))
            if el.tag == "joint":
                for t in ("parent", "child"):
                    el.find(t).set("link", hand_prefix + el.find(t).get("link"))
                m = el.find("mimic")
                if m is not None:
                    m.set("joint", hand_prefix + m.get("joint"))
        if el.tag in ("link", "joint", "material"):
            out_root.append(el)
    j = ET.SubElement(out_root, "joint", dict(name="hand_mount_joint", type="fixed"))
    ET.SubElement(j, "parent", dict(link=mount_parent)); ET.SubElement(j, "child", dict(link=hand_prefix + hand_root))
    ET.SubElement(j, "origin", dict(xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy))))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tree = ET.ElementTree(out_root); ET.indent(tree, space="  ")
    with open(out, "w") as f:
        f.write('<?xml version="1.0"?>\n<!-- merged by imitator.embodiments.urdf_tools -->\n' + ET.tostring(out_root, encoding="unicode"))
    return out


# ------------------------------------------------------------------------------------------------ SRDF
def _srdf_xml(name: str, pairs: list[tuple[str, str, str]]) -> str:
    lines = ['<?xml version="1.0" ?>', f'<robot name="{name}">']
    for a, b, why in pairs:
        lines.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="{why}"/>')
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def generate_srdfs(urdf: str, home: dict[str, float] | None = None, n_samples: int = 3000, seed: int = 0,
                   always_frac: float = 0.98, extra_disable=(), out_dir: str | None = None,
                   planning_urdf: str | None = None, sim_skip_groups=()) -> dict:
    """Sample random configurations with mplib (FCL, convex hulls as in PhysX) and write <stem>.srdf (minimal, for the
    simulator) and <stem>_mplib.srdf (full, for planning). Returns a summary dict (also written as <stem>_srdf.json)."""
    import mplib
    from mplib.pymp import ArticulatedModel, PlanningWorld
    stem = os.path.splitext(urdf)[0] if out_dir is None else os.path.join(out_dir, Path(urdf).stem)
    empty = stem + "_empty.srdf"
    with open(empty, "w") as f:
        f.write(_srdf_xml(Path(urdf).stem, []))
    model = ArticulatedModel(planning_urdf or urdf, empty, [0, 0, -9.81], [], [], convex=True, verbose=False)
    world = PlanningWorld([model], ["robot"], [], [])
    pin = model.get_pinocchio_model()
    links = pin.get_link_names()
    jn = pin.get_joint_names()
    um = load_urdf(urdf)
    adjacent = {frozenset((j.parent, j.child)) for j in um.joints.values()}
    # links connected through fixed joints only are rigidly attached: treat as adjacent too
    rng = np.random.default_rng(seed)
    lims = np.concatenate(pin.get_joint_limits())
    lo = np.where(np.isfinite(lims[:, 0]), lims[:, 0], -np.pi); hi = np.where(np.isfinite(lims[:, 1]), lims[:, 1], np.pi)
    lo = np.maximum(lo, -2 * np.pi); hi = np.minimum(hi, 2 * np.pi)
    from collections import Counter
    cnt = Counter()

    def pairs_now():
        return {frozenset((c.link_name1, c.link_name2)) for c in world.collide_full() if c.link_name1 != c.link_name2}
    for _ in range(n_samples):
        q = rng.uniform(lo, hi)
        model.set_qpos(q, True)
        for p in pairs_now():
            cnt[p] += 1
    qh = np.array([float((home or {}).get(n, 0.0)) for n in jn])
    qh = np.clip(qh, lo, hi)
    model.set_qpos(qh, True)
    at_home = pairs_now()
    always = {p for p, c in cnt.items() if c >= always_frac * n_samples}
    skip = [set(g) for g in sim_skip_groups]       # link groups the agent filters itself (e.g. ManiSkill's Robotiq bit 31)
    in_skip = lambda p: any(p <= g for g in skip)  # noqa: E731
    minimal = sorted({p for p in (always | at_home | {frozenset(x) for x in extra_disable})
                      if p not in adjacent and not in_skip(p)}, key=lambda s: tuple(sorted(s)))
    planning_extra = {p for p in (always | at_home) if in_skip(p)}
    all_pairs = {frozenset((a, b)) for i, a in enumerate(links) for b in links[i + 1:]}
    never = {p for p in all_pairs if cnt[p] == 0}
    full = sorted(set(minimal) | planning_extra | {frozenset(p) for g in skip for p in
                                                   [(a, b) for a in g for b in g if a < b]} | never | (adjacent & all_pairs),
                  key=lambda s: tuple(sorted(s)))
    why = lambda p: ("Adjacent" if p in adjacent else "Home" if p in at_home else "Always" if p in always else  # noqa: E731
                     "Never" if p in never else "User")
    sim_srdf, mp_srdf = stem + ".srdf", stem + "_mplib.srdf"
    with open(sim_srdf, "w") as f:
        f.write(_srdf_xml(Path(urdf).stem, [tuple(sorted(p)) + (why(p),) for p in minimal]))
    with open(mp_srdf, "w") as f:
        f.write(_srdf_xml(Path(urdf).stem, [tuple(sorted(p)) + (why(p),) for p in full]))
    os.remove(empty)
    summary = dict(urdf=urdf, sim_srdf=sim_srdf, mplib_srdf=mp_srdf, n_samples=n_samples, n_sim_pairs=len(minimal),
                   n_mplib_pairs=len(full), sim_pairs=[sorted(p) for p in minimal],
                   home_collisions=[sorted(p) for p in at_home if p not in adjacent],
                   always=[sorted(p) for p in always if p not in adjacent])
    with open(stem + "_srdf.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary

def make_planning_urdf(sim_urdf: str, out_urdf: str, clip_static_below: float | None = 0.005) -> str:
    """URDF for mplib 0.1.1: collision meshes baked into convex hulls under <out dir>/col/ with RELATIVE paths and the
    '<file>.convex.stl' sibling mplib's convex mode requires (it prepends the URDF directory even to absolute paths and
    never computes hulls itself); visuals removed. Link / joint names are unchanged.
    clip_static_below: hulls of links rigidly attached to the root are cut at this height (root frame), so a base that
    sits on (or slightly into) the table is not in permanent collision with the planner's table box."""
    import trimesh
    um = load_urdf(sim_urdf)
    Ts0 = um.fk({})

    def static(l):
        j = um.parent_joint(l)
        while j is not None:
            if j.type != "fixed":
                return False
            j = um.parent_joint(j.parent)
        return True
    tree = ET.parse(sim_urdf)
    root = tree.getroot()
    out_dir = os.path.dirname(os.path.abspath(out_urdf))
    os.makedirs(os.path.join(out_dir, "col"), exist_ok=True)
    for l in root.findall("link"):
        for v in list(l.findall("visual")):
            l.remove(v)
        for k, c in enumerate(l.findall("collision")):
            g = c.find("geometry")
            e = g[0] if g is not None and len(g) else None
            if e is None or e.tag != "mesh":
                continue
            sc = np.array([float(x) for x in e.get("scale", "1 1 1").split()])
            V = load_mesh_vertices(e.get("filename")) * sc
            hull = trimesh.Trimesh(V, process=True).convex_hull
            if clip_static_below is not None and static(l.get("name")):
                T = Ts0[l.get("name")] @ _origin(c)                       # mesh frame -> root frame
                W = (T[:3, :3] @ hull.vertices.T).T + T[:3, 3]
                if W[:, 2].min() < clip_static_below:
                    c0 = clip_static_below
                    keep = W[W[:, 2] >= c0]
                    E = hull.edges_unique
                    za, zb = W[E[:, 0], 2], W[E[:, 1], 2]
                    cross = (za - c0) * (zb - c0) < 0
                    A, B = W[E[cross, 0]], W[E[cross, 1]]
                    t = ((c0 - A[:, 2]) / (B[:, 2] - A[:, 2]))[:, None]
                    W2 = np.vstack([keep, A + t * (B - A)])
                    if len(W2) >= 4:
                        Ti = np.linalg.inv(T)
                        hull = trimesh.Trimesh((Ti[:3, :3] @ W2.T).T + Ti[:3, 3]).convex_hull
            rel = f"col/{l.get('name').replace('/', '_')}_{k}.stl"
            hull.export(os.path.join(out_dir, rel))
            hull.export(os.path.join(out_dir, rel + ".convex.stl"))
            e.set("filename", rel)
            if "scale" in e.attrib:
                del e.attrib["scale"]
    ET.indent(tree, space="  ")
    with open(out_urdf, "w") as f:
        f.write('<?xml version="1.0"?>\n<!-- planning URDF (convex hulls) derived from ' + os.path.abspath(sim_urdf) + ' -->\n'
                + ET.tostring(root, encoding="unicode"))
    return out_urdf
