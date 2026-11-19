"""Default tactile pads (pad frame in the LINK frame) for the imsim / ManiSkill embodiments.

A pad is a rectangle on the inner surface of a finger: origin o (pad centre, on the surface) and rotation R = [x y n] in the
link frame, n = outward surface normal (toward the grasped object / the other finger), y = along the finger toward the
fingertip, x = y x n (across the finger). size = (W along x, L along y) in metres. Taxel (i, j) of an (nx, ny) layout
sits at x_i = -W/2 + (i + 1/2) W/nx, y_j = -L/2 + (j + 1/2) L/ny; arrays are indexed [..., i, j].

Derivation (research/scripts/e1_tactile_build_pads.py, output research/results/e1_tactile/pads_derived.json): PhysX
collides dynamic links with the CONVEX HULL of each collision mesh, so the pad lies on the hull facet that faces the other
finger (normal within 15 deg of the closing direction at a half-open gripper) and contains the projection of the TCP;
rectangle = that facet's bounding rectangle shrunk by 0.25 mm. Hull facets that are tilted with respect to the closing
direction are reported (ARX X5 7 deg, WSG-50 4.5 deg, PiPER 1.7 deg: the hulls of these finger meshes are wedges).
Allegro fingertips: the palmar side (direction the tip moves when the distal joint flexes), a 14 x 14 mm patch tangent to
the (near-spherical, r = 12 mm) tip hull at its palmar-most point; the curvature is handled by the per-taxel surface
height, ray-cast from the link's own collision shapes when the sensor starts.

FlexiTac (arXiv 2604.28156) is 12 x 32 taxels at 2 mm pitch (24 x 64 mm). The default layout keeps the 12 x 32 COUNT
and stretches it over each finger's pad (pitch below), so pitches differ from the real sensor on small fingers:
  Panda 17 x 18 mm (1.42 x 0.56 mm), Robotiq 2F-85 21.5 x 37 (1.79 x 1.16), WSG-50 29.5 x 37.7 (2.46 x 1.18),
  PiPER 24.9 x 66.1 (2.08 x 2.07, ~FlexiTac), ARX X5 31.9-34.5 x 71 (2.7-2.9 x 2.2), Allegro tip 14 x 14 (1.17 x 0.44).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PadSpec:
    name: str                       # unique within a sensor, e.g. "left:finger_left"
    link: str                       # link name in the articulation
    origin: tuple                   # pad centre in the link frame [m]
    R: tuple                        # 3x3 rotation, columns = pad x, y, n in the link frame
    size: tuple                     # (W, L) [m]
    agent: int = 0                  # index into env.agent.agents (0 for single-agent envs)
    stiffness: float | None = None  # Winkler foundation modulus k = E / h [Pa/m]; None = sensor default
    articulation: str | None = None  # resolve the link in env.scene.articulations[name] instead of an agent (rigs)
    note: str = ""

    def rot(self) -> np.ndarray:
        return np.asarray(self.R, float).reshape(3, 3)

    def pos(self) -> np.ndarray:
        return np.asarray(self.origin, float).reshape(3)


def _R(x, y, n):
    return tuple(map(tuple, np.c_[np.asarray(x, float), np.asarray(y, float), np.asarray(n, float)]))


# link name -> (origin [m], R columns x, y, n, size (W, L) [m])
_PANDA = {
    "panda_leftfinger": ((0.0, -0.00002, 0.04525), _R((-1, 0, 0), (0, 0, 1), (0, -1, 0)), (0.0170, 0.0180)),
    "panda_rightfinger": ((0.0, 0.00002, 0.04525), _R((1, 0, 0), (0, 0, 1), (0, 1, 0)), (0.0170, 0.0180)),
}
_WSG = {   # UR5 + WSG-50 (RoboTwin ur5-wsg); hull facet 4.5 deg off the closing axis
    "finger_left": ((0.0015, 0.0, 0.09152), _R((0, 1, 0), (0.0793, 0, 0.99685), (0.99685, 0, -0.0793)), (0.0295, 0.03765)),
    "finger_right": ((0.0015, 0.0, 0.09152), _R((0, 1, 0), (0.0793, 0, 0.99685), (0.99685, 0, -0.0793)), (0.0295, 0.03765)),
}
_ARX = {   # ARX X5 (RoboTwin ARX-X5); hull facet 6.4-7.3 deg off the closing axis
    "link7": ((0.0205, -0.02027, -0.00822), _R((0, 0, -1), (0.993, -0.118, 0), (-0.118, -0.993, 0)), (0.03186, 0.0710)),
    "link8": ((0.0205, 0.02027, 0.00712), _R((0, 0, 1), (0.993, 0.118, 0), (-0.118, 0.993, 0)), (0.03446, 0.0710)),
}
_PIPER = {  # AgileX PiPER (RoboTwin piper); hull facet 1.7 deg off the closing axis
    "link7": ((0.0, -0.03825, -0.0010), _R((1, 0, 0), (0, 1.0, 0.03), (0, -0.03, 1.0)), (0.0249, 0.06609)),
    "link8": ((0.0, 0.03825, -0.0010), _R((-1, 0, 0), (0, -1.0, 0.03), (0, 0.03, 1.0)), (0.0249, 0.06609)),
}
_ROBOTIQ = {  # Robotiq 2F-85 (ManiSkill-XArm6 xarm6_robotiq.urdf) inner finger pad boxes
    "left_inner_finger_pad": ((0.0, -0.00318, 0.0), _R((-1, 0, 0), (0, 0, 1), (0, -1, 0)), (0.0215, 0.0370)),
    "right_inner_finger_pad": ((0.0, -0.00318, 0.0), _R((-1, 0, 0), (0, 0, 1), (0, -1, 0)), (0.0215, 0.0370)),
}
_ALLEGRO = {  # Allegro fingertips (thumb, index, middle, ring); palmar side = +x of each tip link
    t: ((0.0120, 0.00001, 0.00611), _R((0, -1, 0), (0, 0, 1), (1, 0, 0)), (0.0140, 0.0140))
    for t in ("link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip")
}

# ManiSkill / imsim agent uid -> pad table
AGENT_PADS = {
    "panda": _PANDA, "panda_wristcam": _PANDA, "panda_cloth": _PANDA, "floating_panda_gripper": _PANDA,
    "imsim_ur5_wsg": _WSG, "imsim_arx_x5": _ARX, "imsim_piper": _PIPER, "imsim_xarm6_robotiq": _ROBOTIQ,
    "imsim_xarm6_allegro_left": _ALLEGRO, "imsim_xarm6_allegro_right": _ALLEGRO,
}
# embodiment record name -> pad table (rigs, tests)
RECORD_PADS = {"panda": _PANDA, "ur5_wsg": _WSG, "arx_x5": _ARX, "piper": _PIPER, "xarm6_robotiq": _ROBOTIQ,
               "xarm6_allegro_left": _ALLEGRO, "xarm6_allegro_right": _ALLEGRO}


def _orthonormal(R) -> tuple:
    """re-orthonormalise a rounded rotation (keeps n exact, then y, x = y x n)"""
    M = np.asarray(R, float).reshape(3, 3)
    n = M[:, 2] / np.linalg.norm(M[:, 2])
    y = M[:, 1] - n * (M[:, 1] @ n); y /= np.linalg.norm(y)
    x = np.cross(y, n)
    return _R(x, y, n)


def pads_for_table(table: dict, prefix: str = "", agent: int = 0, links=None) -> list[PadSpec]:
    out = []
    for link, (o, R, size) in table.items():
        if links is not None and link not in links:
            continue
        out.append(PadSpec(name=f"{prefix}{link}", link=link, origin=tuple(o), R=_orthonormal(R), size=tuple(size), agent=agent))
    return out


def default_pads(env) -> list[PadSpec]:
    """pads of every agent of a ManiSkill env (u.agent or u.agent.agents) that has a default table. Names are
    '<side>:<link>' for dual-arm envs (side = left / right from imsim's agent order) and '<link>' otherwise."""
    u = env.unwrapped
    agents = list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent]
    sides = ("left", "right") if len(agents) == 2 else tuple(f"a{i}" for i in range(len(agents)))
    out = []
    for i, ag in enumerate(agents):
        uid = getattr(ag, "uid", "")
        table = AGENT_PADS.get(uid)
        if table is None:
            continue
        lm = ag.robot.links_map
        prefix = f"{sides[i]}:" if len(agents) > 1 else ""
        out += pads_for_table({k: v for k, v in table.items() if k in lm}, prefix=prefix, agent=i)
    if not out:
        raise ValueError(f"no default tactile pads for agents {[getattr(a, 'uid', '?') for a in agents]}; pass pads=[PadSpec...]")
    return out
