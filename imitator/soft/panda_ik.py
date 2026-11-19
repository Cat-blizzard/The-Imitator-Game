"""Small numpy FK / IK for a serial URDF chain (used for the scripted Panda expert in both routes).

The chain is read from the same URDF the simulators load (ManiSkill's panda_v2.urdf), so the joint targets mean the same
in SAPIEN/PhysX (route A) and Newton/MuJoCo-Warp (route B). IK: damped least squares on position + orientation error,
joint limits by clamping, warm-started from the previous solution (smooth trajectories).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

def _panda_urdf() -> str:
    import mani_skill
    return os.path.join(os.path.dirname(os.path.abspath(mani_skill.__file__)), "assets", "robots", "panda", "panda_v2.urdf")


PANDA_URDF = _panda_urdf()
PANDA_REST = np.array([0.0, np.pi / 8, 0.0, -5 * np.pi / 8, 0.0, 3 * np.pi / 4, np.pi / 4])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(ax, q):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def _T(R, p):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


class UrdfChain:
    def __init__(self, urdf: str = PANDA_URDF, base_link: str = "panda_link0", tip_link: str = "panda_hand_tcp"):
        root = ET.parse(urdf).getroot()
        joints = {}
        for j in root.findall("joint"):
            child = j.find("child").get("link")
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])]
            ax = j.find("axis")
            lim = j.find("limit")
            joints[child] = dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                                 T=_T(_rpy(*rpy), xyz), axis=[float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1],
                                 lower=float(lim.get("lower", "-1e9")) if lim is not None else -1e9,
                                 upper=float(lim.get("upper", "1e9")) if lim is not None else 1e9)
        chain, link = [], tip_link
        while link != base_link:
            j = joints[link]; chain.append(j); link = j["parent"]
        self.chain = chain[::-1]
        self.active = [j for j in self.chain if j["type"] in ("revolute", "prismatic", "continuous")]
        self.lower = np.array([j["lower"] for j in self.active])
        self.upper = np.array([j["upper"] for j in self.active])

    def fk(self, q):
        """Pose of the tip in the base frame (4x4) and the world-frame (base-frame) joint axes / origins for the Jacobian."""
        T = np.eye(4); k = 0; axes, origins = [], []
        for j in self.chain:
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                axes.append(T[:3, :3] @ np.asarray(j["axis"])); origins.append(T[:3, 3].copy())
                T = T @ _T(_axis_angle(j["axis"], q[k]), np.zeros(3)); k += 1
        return T, np.array(axes), np.array(origins)

    def jacobian(self, q):
        T, axes, origins = self.fk(q)
        p = T[:3, 3]
        Jp = np.cross(axes, p - origins).T
        Jr = axes.T
        return T, np.vstack([Jp, Jr])

    def ik(self, R_target, p_target, q0, iters: int = 100, damping: float = 1e-3, tol: float = 1e-5):
        q = np.array(q0, float)
        for _ in range(iters):
            T, J = self.jacobian(q)
            ep = p_target - T[:3, 3]
            Re = R_target @ T[:3, :3].T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            e = np.r_[ep, er]
            if np.linalg.norm(e) < tol:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q = np.clip(q + dq, self.lower, self.upper)
        return q, float(np.linalg.norm(e))


TCP_DOWN_R = np.diag([1.0, -1.0, -1.0])     # rotation pi about x: hand z axis pointing down


def joint_trajectory(tcp_points_world: np.ndarray, base_xyz=(-0.615, 0.0, 0.0), q0=PANDA_REST, chain: UrdfChain | None = None):
    """Joint targets (T, 7) for a sequence of TCP positions (world frame), gripper pointing down; base frame has identity
    rotation. Returns (q, max position error)."""
    chain = chain or UrdfChain()
    q = np.array(q0, float); out = []; err = 0.0
    for p in tcp_points_world:
        q, e = chain.ik(TCP_DOWN_R, np.asarray(p, float) - np.asarray(base_xyz), q)
        out.append(q.copy()); err = max(err, e)
    return np.array(out), err
