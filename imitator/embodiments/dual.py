"""Dual-arm configuration layer.

A DualConfig names two arm records (left, right; hands come with the arm record) and the standard tabletop mount:

  World frame: origin on the table top, midway between the two arm bases; +x points from the bases into the
  workspace ("forward"), +y to the robot's left, +z up. The table top is the plane z = 0 (a kinematic box
  1.1 m deep x 1.6 m wide, x in [-0.35, 0.75]).
  Mount: both bases on the table top (z = base_z = 0), at x = base_x = 0, y = +separation/2 (left) and -separation/2
  (right), both facing +x (yaw = record.mount_yaw, 0 for every arm here). Separation: 0.60 m for the small arms
  (ARX X5, PiPER: reach ~0.62 m), 0.80 m for UR5 and xArm6 (reach 0.70-0.85 m).
  Shared workspace (smoke task): cube x in [0.25, 0.38], |y| in [0.13, 0.20] (side drawn per episode); target square
  (half size 0.05 m) centred at x in [0.25, 0.38], y in [-0.03, 0.03].

Collision filtering
  * SAPIEN encodes every SRDF <disable_collisions> pair, and ManiSkill's own per-robot filters (e.g. bit 31 on every
    Robotiq link), as bits of collision group 2; two shapes with ANY common group-2 bit never collide. Two copies of
    the same arm therefore share bits and silently stop colliding with each other (e.g. both Robotiq grippers).
    `make_group2_bits_disjoint` re-assigns the bits of each later arm to unused bit numbers, so intra-arm filtering
    is unchanged and inter-arm contact is real (verified in tests/test_embodiments.py).
  * Base-to-table contact is filtered with one dedicated bit (TABLE_BIT) on the table and on the links that sit on
    it at the home pose (root/base links); the robots' moving links still collide with the table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sapien

from . import registry as reg

TABLE_BIT = 26
TABLE_TOP_Z = 0.0
TABLE_X = (-0.35, 0.75)
TABLE_Y = (-0.80, 0.80)
TABLE_THICK = 0.75


@dataclass
class DualConfig:
    name: str
    left: str
    right: str
    separation: float
    base_x: float = 0.0
    base_z: float = 0.0
    cube_half: float = 0.025
    cube_x: tuple = (0.25, 0.38)
    cube_abs_y: tuple = (0.13, 0.20)
    target_x: tuple = (0.25, 0.38)
    target_y: tuple = (-0.03, 0.03)
    target_half: float = 0.05
    label: str = ""
    gpu_sim_freq: int = 100    # physics rate make_dual_env uses on GPU PhysX unless sim_config is given (E19 stress test:
                               # ARX X5 explodes at 100 Hz, stable at 200; PiPER needs 500); CPU always uses 100 Hz
    notes: str = ""

    @property
    def records(self) -> tuple[reg.ArmRecord, reg.ArmRecord]:
        return reg.ARMS[self.left], reg.ARMS[self.right]

    @property
    def robot_uids(self) -> tuple[str, str]:
        return reg.ARMS[self.left].agent_uid, reg.ARMS[self.right].agent_uid

    def mount_poses(self) -> list[sapien.Pose]:
        out = []
        for rec, sgn in ((reg.ARMS[self.left], 1.0), (reg.ARMS[self.right], -1.0)):
            yaw = rec.mount_yaw
            q = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            out.append(sapien.Pose([self.base_x, sgn * self.separation / 2, self.base_z], q))
        return out

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["mount_poses"] = [dict(p=list(map(float, p.p)), q=list(map(float, p.q))) for p in self.mount_poses()]
        d["robot_uids"] = list(self.robot_uids)
        return d


CONFIGS: dict[str, DualConfig] = {
    "dual_arx_x5": DualConfig("dual_arx_x5", "arx_x5", "arx_x5", 0.60, label="dual ARX X5", gpu_sim_freq=200),
    "dual_ur5_wsg": DualConfig("dual_ur5_wsg", "ur5_wsg", "ur5_wsg", 0.80, label="dual UR5 + WSG-50"),
    "dual_piper": DualConfig("dual_piper", "piper", "piper", 0.60, label="dual AgileX PiPER", gpu_sim_freq=500),
    "dual_xarm6_robotiq": DualConfig("dual_xarm6_robotiq", "xarm6_robotiq", "xarm6_robotiq", 0.80,
                                     label="dual xArm6 + Robotiq 2F-85"),
    "dual_xarm6_allegro": DualConfig("dual_xarm6_allegro", "xarm6_allegro_left", "xarm6_allegro_right", 0.80,
                                     cube_half=0.03, label="dual xArm6 + Allegro (L/R)"),
}


# ------------------------------------------------------------------------------------------------ collision filtering
def _shapes(link):
    for body in link._bodies:
        for cs in body.get_collision_shapes():
            yield cs


def group2_bits_of_link(link) -> set[int]:
    bits = set()
    for cs in _shapes(link):
        g = int(cs.get_collision_groups()[2])
        bits |= {i for i in range(32) if (g >> i) & 1}
    return bits


def group2_bits(agent) -> set[int]:
    bits = set()
    for link in agent.robot.get_links():
        bits |= group2_bits_of_link(link)
    return bits


def make_group2_bits_disjoint(agents, reserved=(TABLE_BIT,)) -> list[dict[int, int]]:
    """Re-assign collision-group-2 bits so that no two agents share one (see module docstring). Returns the remap
    applied to each agent ({old bit: new bit}; empty for the first)."""
    used = set(reserved)
    remaps = []
    for ag in agents:
        bits = group2_bits(ag)
        remap = {}
        taken = used | bits
        for b in sorted(bits):
            if b in used:
                free = [i for i in range(32) if i not in taken and i not in remap.values()]
                if not free:
                    raise RuntimeError("out of collision-group-2 bits")
                remap[b] = free[0]
        if remap:
            for link in ag.robot.get_links():
                for cs in _shapes(link):
                    cg = cs.get_collision_groups()
                    g = int(cg[2]); ng = g
                    for o, n in remap.items():
                        if (g >> o) & 1:
                            ng = (ng & ~(1 << o)) | (1 << n)
                    cg[2] = ng
                    cs.set_collision_groups(cg)
        used |= {remap.get(b, b) for b in bits}
        remaps.append(remap)
    return remaps


def base_links_on_table(rec: reg.ArmRecord, base_z: float = 0.0, tol: float = 0.01) -> list[str]:
    """links whose collision geometry reaches within `tol` of the table top at the home pose (FK on the derived URDF)"""
    from .urdf_tools import load_urdf
    m = load_urdf(rec.urdf_path)
    Ts = m.fk(rec.home_qpos())
    out = []

    def static(l):
        """height-invariant links: rigidly attached to the (fixed) root, or behind revolute joints whose axis is
        vertical (a base 'pan' joint): their height above the table never changes, so filtering them against the
        table cannot hide a real collision"""
        j = m.parent_joint(l)
        while j is not None:
            if j.type != "fixed":
                if j.type not in ("revolute", "continuous"):
                    return False
                ax = Ts[j.parent][:3, :3] @ j.T[:3, :3] @ (np.asarray(j.axis, float) / np.linalg.norm(j.axis))
                if abs(abs(ax[2]) - 1.0) > 1e-6:
                    return False
            j = m.parent_joint(j.parent)
        return True
    for l in m.links:
        if not static(l):
            continue
        P = m.collision_points(l)
        if not len(P):
            continue
        z = (Ts[l][:3, :3] @ P.T).T[:, 2] + Ts[l][2, 3] + base_z
        if z.min() < TABLE_TOP_Z + tol:
            out.append(l)
    return out


def filter_base_table(agent, table, links: list[str]) -> None:
    for n in links:
        if n in agent.robot.links_map:
            agent.robot.links_map[n].set_collision_group_bit(group=2, bit_idx=TABLE_BIT, bit=1)
    table.set_collision_group_bit(group=2, bit_idx=TABLE_BIT, bit=1)


# ------------------------------------------------------------------------------------------------ env helper
def make_dual_env(config: str = "dual_arx_x5", num_envs: int = 1, sim_backend: str = "cpu", render: bool = False,
                  control_mode: str = "pd_joint_pos", obs_mode: str = "state", **kw):
    """gym.make the smoke task for a dual configuration (robot_uids = (left, right) agent uids).
    render=True uses the CPU renderer (SAPIEN on lavapipe, render_backend 'pci:0')."""
    import gymnasium as gym
    from .. import compat
    from ..envs import dual_tabletop  # noqa: F401  (registers the env)
    compat.patch_render_backend()
    if sim_backend not in ("cpu", "physx_cpu"):
        compat.ensure_sapien_home()
        c = CONFIGS[config] if isinstance(config, str) else config
        if "sim_config" not in kw and c.gpu_sim_freq != 100:
            kw["sim_config"] = dict(sim_freq=c.gpu_sim_freq)
    return gym.make("ImsimDualTabletop-v1", dual_config=config, num_envs=num_envs, sim_backend=sim_backend,
                    render_backend="pci:0" if render else "none", render_mode="rgb_array" if render else None,
                    control_mode=control_mode, obs_mode=obs_mode, **kw)
