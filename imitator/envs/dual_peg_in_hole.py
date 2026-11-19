"""DualPegInHole-v1 (E29b): bimanual in-air assembly. The left arm lifts a holed block by its handle and holds it up; the
right arm picks a standing peg, inserts it into the block's (blind) hole and lets go; the left arm keeps holding the block.

Scene (table top z = 0, robots at x = 0 facing +x)
  block   one rigid body: a hole box 6 x 6 x 5 cm (density 100 kg/m^3, 13 g) with a 3.2 x 3.2 cm square blind hole
          (floor 6 mm thick, 4.4 cm deep) and a handle bar 4 (x) x 11 (y) x 4 (z) cm (density 500, 88 g) attached on its +y
          side (towards the left arm), so the centre of mass sits near the grasp point on the handle (a light part with a
          heavy handle: small gravity torque on the holder's grasp); friction 1.5 (block and peg); actor origin = hole box
          centre; on the table at reset: hole centre x in [0.28, 0.34], y in
          [-0.03, 0.01], yaw in [-0.15, 0.15] rad about the world-y-aligned pose
  peg     2.2 x 2.2 x 11 cm box (density 250, ~13 g), standing upright on the right side (x in [0.26, 0.34],
          y in [-0.21, -0.16]); clearance in the hole 5 mm per side
  success benchmark-style flag (written the way ManiSkill's PegInsertionSide / a RoboTwin insertion task would, for the
          card's bench column only; the contract judges episodes): peg bottom centre within 8 mm of the hole axis and
          within 12 mm of the hole floor, at the evaluated step. One step: no settle, no release check, and nothing about
          the block being held up.
Signals for the contract: imitator.verify.dual_tasks_b2.peg_in_hole.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from ..embodiments import dual as D
from .dual_tabletop import DualTabletopEnv

HOLE_OUT = (0.03, 0.03, 0.025)        # hole box half extents
HOLE_IN = 0.016                        # square hole half side
FLOOR_T = 0.006
HANDLE_HALF = (0.02, 0.055, 0.02)
HANDLE_C = (0.0, HOLE_OUT[1] + HANDLE_HALF[1], -HOLE_OUT[2] + HANDLE_HALF[2])
PEG_HALF = (0.011, 0.011, 0.055)
FLOOR_TOP = -HOLE_OUT[2] + FLOOR_T     # hole floor top, block frame z
GRIP_FRICTION = 1.5                    # static = dynamic friction of the block and the peg (PhysX averages it with the
                                       # gripper pads' 2.0); the table keeps ManiSkill's default 0.3
MOUTH = HOLE_OUT[2]                    # hole mouth (block top), block frame z


def block_parts():
    ox, oy, oz = HOLE_OUT
    wz = (oz - FLOOR_TOP) / 2                       # wall half height (floor top .. block top)
    zc = (oz + FLOOR_TOP) / 2
    w = (ox - HOLE_IN) / 2
    return [((0, 0, -oz + FLOOR_T / 2), (ox, oy, FLOOR_T / 2)),
            ((HOLE_IN + w, 0, zc), (w, oy, wz)), ((-HOLE_IN - w, 0, zc), (w, oy, wz)),
            ((0, HOLE_IN + w, zc), (HOLE_IN, w, wz)), ((0, -HOLE_IN - w, zc), (HOLE_IN, w, wz)),
            (HANDLE_C, HANDLE_HALF)]


@register_env("DualPegInHole-v1", max_episode_steps=3000)
class DualPegInHoleEnv(DualTabletopEnv):
    def __init__(self, *args, hole_x=(0.28, 0.34), hole_y=(-0.03, 0.01), peg_x=(0.26, 0.34), peg_y=(-0.21, -0.16),
                 yaw_range: float = 0.15, block_density: float = 100.0, handle_density: float = 500.0,
                 peg_density: float = 250.0, **kwargs):
        self.hole_x, self.hole_y, self.peg_x, self.peg_y = tuple(hole_x), tuple(hole_y), tuple(peg_x), tuple(peg_y)
        self.yaw_range, self.block_density, self.peg_density = float(yaw_range), float(block_density), float(peg_density)
        self.handle_density = float(handle_density)
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------- scene
    def _load_scene(self, options: dict):
        b = self.scene.create_actor_builder()
        hx, hy = (D.TABLE_X[1] - D.TABLE_X[0]) / 2, (D.TABLE_Y[1] - D.TABLE_Y[0]) / 2
        cx, cy = (D.TABLE_X[1] + D.TABLE_X[0]) / 2, (D.TABLE_Y[1] + D.TABLE_Y[0]) / 2
        half = [hx, hy, D.TABLE_THICK / 2]
        b.add_box_collision(half_size=half)
        b.add_box_visual(half_size=half, material=sapien.render.RenderMaterial(base_color=[0.78, 0.70, 0.60, 1]))
        b.initial_pose = sapien.Pose([cx, cy, D.TABLE_TOP_Z - D.TABLE_THICK / 2])
        self.table = b.build_kinematic(name="table")
        bb = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial(base_color=[0.15, 0.35, 0.85, 1])
        pm = sapien.pysapien.physx.PhysxMaterial(GRIP_FRICTION, GRIP_FRICTION, 0.0)
        for k, (c, h) in enumerate(block_parts()):
            dens = self.handle_density if k == len(block_parts()) - 1 else self.block_density
            bb.add_box_collision(sapien.Pose(c), h, material=pm, density=dens)
            bb.add_box_visual(sapien.Pose(c), h, material=mat)
        bb.initial_pose = sapien.Pose([0.31, -0.01, HOLE_OUT[2]])
        self.block = bb.build(name="block")
        pb = self.scene.create_actor_builder()
        pb.add_box_collision(half_size=PEG_HALF, material=pm, density=self.peg_density)
        pb.add_box_visual(half_size=PEG_HALF, material=sapien.render.RenderMaterial(base_color=[0.85, 0.15, 0.12, 1]))
        pb.initial_pose = sapien.Pose([0.30, -0.18, PEG_HALF[2]])
        self.peg = pb.build(name="peg")
        self.cube = self.peg                               # generic helpers read env.cube
        tb = self.scene.create_actor_builder()
        tb.add_box_visual(half_size=[0.0005, 0.0005, 0.0005])
        tb.initial_pose = sapien.Pose([0.3, 0.0, -0.01])
        self.target = tb.build_kinematic(name="target")    # unused marker (DualTabletopEnv keeps a target)
        ags = list(self.agent.agents)
        self.group2_remaps = D.make_group2_bits_disjoint(ags)
        for ag, links in zip(ags, self.base_table_links):
            D.filter_base_table(ag, self.table, links)

    @property
    def cube_half(self):
        return np.asarray(PEG_HALF, float)

    @cube_half.setter
    def cube_half(self, v):
        pass

    # ---------------------------------------------------------------- episode
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            for ag, pose in zip(self.agent.agents, self.dual.mount_poses()):
                q0 = torch.as_tensor(ag.home_qpos_vector(), dtype=torch.float32)
                ag.reset(q0.repeat(b, 1))
                ag.robot.set_pose(Pose.create_from_pq(torch.as_tensor(pose.p, dtype=torch.float32).repeat(b, 1),
                                                      torch.as_tensor(pose.q, dtype=torch.float32).repeat(b, 1)))
            u = lambda lo, hi: torch.rand(b) * (hi - lo) + lo  # noqa: E731
            p = torch.zeros((b, 3))
            p[:, 0] = u(*self.hole_x); p[:, 1] = u(*self.hole_y); p[:, 2] = HOLE_OUT[2] + 1e-4
            yaw = u(-self.yaw_range, self.yaw_range)
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], -1)
            self.block.set_pose(Pose.create_from_pq(p, q))
            pp = torch.zeros((b, 3))
            pp[:, 0] = u(*self.peg_x); pp[:, 1] = u(*self.peg_y); pp[:, 2] = PEG_HALF[2] + 1e-4
            pyaw = u(-0.3, 0.3)
            pq = torch.stack([torch.cos(pyaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(pyaw / 2)], -1)
            self.peg.set_pose(Pose.create_from_pq(pp, pq))
            if not hasattr(self, "target_pos") or self.target_pos.shape[0] != self.num_envs:
                self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
                self.cube_side = torch.ones(self.num_envs, device=self.device)

    # ---------------------------------------------------------------- task signals
    def peg_in_block(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(radial distance of the peg bottom centre from the hole axis, height of the peg bottom centre above the hole
        floor), both in the block frame; the peg bottom = the end of its long axis that is lower in the block frame"""
        Tb = self.block.pose.to_transformation_matrix()
        Tp = self.peg.pose.to_transformation_matrix()
        Rb = Tb[:, :3, :3]
        ax = Tp[:, :3, 2]
        c = Tp[:, :3, 3]
        ends = torch.stack([c + ax * PEG_HALF[2], c - ax * PEG_HALF[2]], 1)            # (n, 2, 3)
        loc = torch.einsum("nij,nkj->nki", Rb.transpose(1, 2), ends - Tb[:, None, :3, 3])  # block frame
        k = loc[..., 2].argmin(1)
        bot = loc[torch.arange(len(k)), k]
        return torch.linalg.norm(bot[:, :2], dim=-1), bot[:, 2] - FLOOR_TOP

    def evaluate(self) -> dict[str, Any]:
        r, h = self.peg_in_block()
        return dict(success=(r <= 0.008) & (h <= 0.012) & (h >= -0.003), peg_radial=r, peg_height=h)

    def _get_obs_extra(self, info: dict):
        obs = dict()
        for i, ag in enumerate(self.agent.agents):
            obs[f"tcp_pose_{i}"] = ag.tcp.pose.raw_pose
        if "state" in self.obs_mode:
            obs["block_pose"] = self.block.pose.raw_pose
            obs["peg_pose"] = self.peg.pose.raw_pose
        return obs
