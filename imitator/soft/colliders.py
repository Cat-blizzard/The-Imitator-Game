"""Kinematic collider descriptions of robot links, as seen by the cloth solver.

A collider is one robot link (a rigid body whose pose is imposed every control step) carrying one or more boxes in the
link frame. Boxes are what Newton's VBD particle-shape contact handles most robustly; for the Panda fingers they are
exactly the URDF's own collision primitives (mani_skill/assets/robots/panda/panda_v2.urdf), for the hand a bounding box
of hand.stl. Quaternions are (w, x, y, z) here (SAPIEN convention); cloth.py converts to Warp's (x, y, z, w).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    half: tuple[float, float, float]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class Collider:
    link: str                       # robot link name (SAPIEN / URDF)
    boxes: list[Box] = field(default_factory=list)
    visual: list = field(default_factory=list)   # [(vertices, faces, T_local 4x4)] visual-only meshes (camera)


@dataclass
class PadPair:
    """Two finger pads facing each other; used for the enclosure (release) signal and the attachment rule.
    The pad box is given in each finger's link frame; inner faces are the faces pointing at the other finger."""
    left: str
    right: str
    left_pad: Box
    right_pad: Box
    left_inner_axis: tuple[int, float]    # (axis index, sign) of the inner-face normal in the left finger frame
    right_inner_axis: tuple[int, float]


def _rx(a):
    return (float(np.cos(a / 2)), float(np.sin(a / 2)), 0.0, 0.0)


# Panda (panda_v2.urdf, ManiSkill 3.0.1). Finger boxes copied from the URDF <collision> elements (sizes are full
# extents there; halved here). Hand: axis-aligned bounds of franka_description/meshes/collision/hand.stl
# x [-0.0316, 0.0316], y [-0.1040, 0.1004], z [-0.0259, 0.0660].
_LEFT = [
    Box((0.011, 0.0075, 0.010), (0.0, 0.0185, 0.011)),                 # screw mount
    Box((0.011, 0.0044, 0.0019), (0.0, 0.0068, 0.0022)),               # carriage sledge
    Box((0.00875, 0.0035, 0.01175), (0.0, 0.0159, 0.02835), _rx(0.5235987755982988)),  # diagonal finger
    Box((0.00875, 0.0076, 0.00925), (0.0, 0.00758, 0.04525)),          # rubber tip (grasp pad)
]
_RIGHT = [Box(b.half, (b.pos[0], -b.pos[1], b.pos[2]),
              b.quat_wxyz if b.quat_wxyz == (1.0, 0.0, 0.0, 0.0) else _rx(-0.5235987755982988)) for b in _LEFT]

PANDA_COLLIDERS = [
    Collider("panda_hand", [Box((0.0316, 0.1022, 0.0460), (0.0, -0.0018, 0.0201))]),
    Collider("panda_leftfinger", _LEFT),
    Collider("panda_rightfinger", _RIGHT),
]

PANDA_PADS = PadPair("panda_leftfinger", "panda_rightfinger", _LEFT[3], _RIGHT[3], (1, -1.0), (1, +1.0))


def collider_link_names(colliders: list[Collider]) -> list[str]:
    return [c.link for c in colliders]
