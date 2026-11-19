"""Cloth colliders for any imsim embodiment (registry record), derived from the URDF collision geometry.

Every gripper link (all links below the flange + grasp_links, imitator.verify.v31.gripper_links) becomes one kinematic
collider whose shape is the box around the link's collision points, axis-aligned in the TCP frame (closing / approach
axes) at the home pose, expressed in the link frame. The two finger pads also define the PadPair (enclosure / release signal): pad box = that finger link's box, inner
axis = the box axis closest to the closing direction (pad 1 -> pad 2 at the home pose), sign towards the other finger.
Collider names are '<prefix><link>' so two arms can share one cloth model ('L:' / 'R:').

cloth_close_command(record, gap): normalised gripper action that leaves `gap` between the pads (prismatic grippers:
FK of the pad hulls over the finger joint range, bisection; the mimic relation of the record is applied). PhysX does not
know the cloth (one-way coupling), so the fingers are stopped where the pinched cloth layer would stop them (E20's
panda_cloth does the same with a 2 mm per finger target).
"""
from __future__ import annotations

import numpy as np

from .colliders import Box, Collider, PadPair
from ..embodiments import registry as reg
from ..embodiments.urdf_tools import load_urdf


def _quat_wxyz(R) -> tuple:
    from transforms3d.quaternions import mat2quat
    q = mat2quat(np.asarray(R, float))
    return tuple(float(x) for x in q)


def link_box(rec: reg.ArmRecord, link: str, pad: float = 0.0, aligned: bool = True) -> Box | None:
    """box around the link's collision points, in the link frame. aligned=True: axis-aligned in the TCP frame (closing
    axis, approach axis) at the home pose - exact pad faces for prismatic fingers, whose orientation relative to the TCP
    never changes; aligned=False: minimum-volume oriented bounding box (trimesh)."""
    import trimesh
    m = load_urdf(rec.urdf_path)
    P = m.collision_points(link)
    if len(P) < 4:
        return None
    if aligned:
        Ts = m.fk(rec.home_qpos())
        R = Ts[link][:3, :3].T @ Ts[rec.tcp_link][:3, :3]      # columns: TCP axes in the link frame
        Q = P @ R
        lo, hi = Q.min(0), Q.max(0)
        c = R @ ((lo + hi) / 2)
        half = (hi - lo) / 2 + pad
        return Box(tuple(float(x) for x in half), tuple(float(x) for x in c), _quat_wxyz(R))
    T, ext = trimesh.bounds.oriented_bounds(P)          # T: world(link) -> box frame
    Ti = np.linalg.inv(T)                                # box frame -> link frame
    half = np.asarray(ext, float) / 2 + pad
    return Box(tuple(float(x) for x in half), tuple(float(x) for x in Ti[:3, 3]), _quat_wxyz(Ti[:3, :3]))


def embodiment_colliders(rec: reg.ArmRecord, prefix: str = "") -> tuple[list[Collider], PadPair | None]:
    from ..verify.v31 import gripper_links
    cols = []
    for l in gripper_links(rec):
        b = link_box(rec, l)
        if b is not None:
            cols.append(Collider(prefix + l, [b]))
    pads = None
    if rec.pad_links:
        m = load_urdf(rec.urdf_path)
        Ts = m.fk(rec.home_qpos())
        g = reg.pad_geometry(rec, Ts)
        u = g["u"]
        axes = []
        for k, l in enumerate(rec.pad_links):
            b = link_box(rec, l)
            R_box_world = Ts[l][:3, :3] @ np.asarray(_quat_to_R(b.quat_wxyz))
            uu = R_box_world.T @ u
            ax = int(np.argmax(np.abs(uu)))
            sgn = float(np.sign(uu[ax])) if k == 0 else -float(np.sign(uu[ax]))
            axes.append((b, (ax, sgn)))
        pads = PadPair(prefix + rec.pad_links[0], prefix + rec.pad_links[1], axes[0][0], axes[1][0], axes[0][1], axes[1][1])
    return cols, pads


def _quat_to_R(q):
    from transforms3d.quaternions import quat2mat
    return quat2mat(np.asarray(q, float))


def opening_at(rec: reg.ArmRecord, q_first: float) -> float:
    m = load_urdf(rec.urdf_path)
    d = rec.home_qpos()
    first = rec.gripper_joints[0]
    d[first] = q_first
    for j, mm in (rec.gripper_mimic or {}).items():
        d[j] = mm.get("multiplier", 1.0) * d[mm["joint"]] + mm.get("offset", 0.0)
    return float(reg.pad_geometry(rec, m.fk(d))["opening"])


def cloth_close_command(rec: reg.ArmRecord, gap: float = 0.004) -> float:
    """normalised gripper command (the PD mimic controller maps [-1, 1] -> [gripper_lower, gripper_upper] of the first
    gripper joint) that leaves `gap` between the pads"""
    if rec.kind != "arm_gripper" or rec.passive_joints:
        raise ValueError(f"{rec.name}: cloth close command needs a prismatic parallel gripper")
    lo, hi = rec.gripper_lower, rec.gripper_upper
    o_lo, o_hi = opening_at(rec, lo), opening_at(rec, hi)
    a, b = (lo, hi) if o_lo < o_hi else (hi, lo)          # a = closed end
    for _ in range(50):
        mid = (a + b) / 2
        if opening_at(rec, mid) < gap:
            a = mid
        else:
            b = mid
    q = (a + b) / 2
    return float(2 * (q - lo) / (hi - lo) - 1)
