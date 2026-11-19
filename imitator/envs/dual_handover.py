"""DualHandover-v1: the left arm picks a block from the left zone, hands it to the right arm in mid-air, the right arm
puts it in the right target square.

Object: a 4 x 18 x 4 cm block (half extents (0.02, 0.09, 0.02), long axis = its local y, density 250 kg/m^3, 72 g,
static / dynamic friction 1.0). E29b task change (E23: 500 kg/m^3, 144 g, scene-default friction 0.3): the stock ManiSkill
Robotiq 2F-85 drive (gripper_force_limit 0.1) cannot hold the 144 g block 5 cm off its centre; the block pivoted ~47 deg in
the grasp (E29b probe research/logs/e29b/handover/probe_robotiq_*.log; 72 g: < 1 deg) and touched the table during the
handover in 7/8 E23 expert episodes. Both values are constructor arguments (block_density, block_friction).
A 5 cm cube cannot be held by two parallel grippers at the same time (both palms would be above the same face); like
RoboTwin 2.0's handover_block we use an elongated block so that each arm can grasp one half.
  scene    table top z = 0 (DualTabletopEnv), block on the left zone (centre x in zone_x, y in +zone_abs_y, yaw in
           [-0.3, 0.3] about the world-y-aligned pose), green target square (half 0.05) on the right zone (centre x in
           zone_x, y in -zone_abs_y)
  success  (benchmark-style flag, RoboTwin handover_block style, for comparison only; the contract judges episodes):
           block centre inside the target square, block resting on the table (bottom within 5 mm), speed < 2 cm/s, and
           both grippers open (measured opening >= 80 % of the gripper's maximum; hands: thumb-finger aperture >= 8 cm).
           It does not check that a handover happened.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from ..embodiments import dual as D
from .dual_tabletop import DualTabletopEnv

BLOCK_HALF = (0.02, 0.09, 0.02)


@register_env("DualHandover-v1", max_episode_steps=3000)
class DualHandoverEnv(DualTabletopEnv):
    def __init__(self, *args, block_half=BLOCK_HALF, zone_x=(0.26, 0.36), zone_abs_y=(0.12, 0.18),
                 block_density: float = 250.0, yaw_range: float = 0.3, block_friction: float | None = 1.0,
                 block_patch_radius: float = 0.0, **kwargs):
        self.block_half = np.asarray(block_half, float)
        self.zone_x, self.zone_abs_y = tuple(zone_x), tuple(zone_abs_y)
        self.block_density = float(block_density)
        self.yaw_range = float(yaw_range)
        self.block_friction = block_friction            # None: the scene default (0.3 / 0.3)
        self.block_patch_radius = float(block_patch_radius)
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
        h = self.block_half.tolist()
        mat = None
        if self.block_friction is not None:
            mat = sapien.physx.PhysxMaterial(static_friction=float(self.block_friction),
                                             dynamic_friction=float(self.block_friction), restitution=0.0)
        bb.add_box_collision(half_size=h, density=self.block_density, material=mat,
                             patch_radius=self.block_patch_radius, min_patch_radius=self.block_patch_radius)
        bb.add_box_visual(half_size=h, material=sapien.render.RenderMaterial(base_color=[0.15, 0.35, 0.85, 1]))
        bb.initial_pose = sapien.Pose([0.3, 0.15, h[2]])
        self.block = bb.build(name="block")
        self.cube = self.block                       # generic helpers (lift probe, executors) read env.cube
        tb = self.scene.create_actor_builder()
        th = self.dual.target_half
        tb.add_box_visual(half_size=[th, th, 0.0008], material=sapien.render.RenderMaterial(base_color=[0.1, 0.75, 0.2, 1]))
        tb.initial_pose = sapien.Pose([0.3, -0.15, 0.0009])
        self.target = tb.build_kinematic(name="target")
        ags = list(self.agent.agents)
        self.group2_remaps = D.make_group2_bits_disjoint(ags)
        for ag, links in zip(ags, self.base_table_links):
            D.filter_base_table(ag, self.table, links)

    @property
    def cube_half(self):            # vector half extents (helpers accept scalar or 3-vector)
        return self.block_half

    @cube_half.setter
    def cube_half(self, v):         # DualTabletopEnv.__init__ assigns a scalar; the block keeps its own extents
        pass

    # ---------------------------------------------------------------- episode
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            for ag, pose in zip(self.agent.agents, self.dual.mount_poses()):
                q0 = torch.as_tensor(ag.home_qpos_vector(), dtype=torch.float32)
                q = q0.repeat(b, 1)
                if self.robot_init_qpos_noise > 0:
                    arm_idx = [i for i, j in enumerate(ag.robot.active_joints) if j.name in ag.record.arm_joints]
                    q[:, arm_idx] += torch.randn((b, len(arm_idx))) * self.robot_init_qpos_noise
                ag.reset(q)
                ag.robot.set_pose(Pose.create_from_pq(torch.as_tensor(pose.p, dtype=torch.float32).repeat(b, 1),
                                                      torch.as_tensor(pose.q, dtype=torch.float32).repeat(b, 1)))
            u = lambda lo, hi: torch.rand(b) * (hi - lo) + lo  # noqa: E731
            p = torch.zeros((b, 3))
            p[:, 0] = u(*self.zone_x); p[:, 1] = u(*self.zone_abs_y); p[:, 2] = float(self.block_half[2])
            yaw = u(-self.yaw_range, self.yaw_range)
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], -1)
            self.block.set_pose(Pose.create_from_pq(p, q))
            t = torch.zeros((b, 3))
            t[:, 0] = u(*self.zone_x); t[:, 1] = -u(*self.zone_abs_y); t[:, 2] = 0.0009
            self.target.set_pose(Pose.create_from_pq(t))
            if not hasattr(self, "target_pos") or self.target_pos.shape[0] != self.num_envs:
                self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
                self.cube_side = torch.ones(self.num_envs, device=self.device)
            self.target_pos[env_idx] = t

    # ---------------------------------------------------------------- task signals
    def cube_bottom_z(self) -> torch.Tensor:
        R = self.block.pose.to_transformation_matrix()[:, :3, :3]
        h = torch.as_tensor(self.block_half, dtype=R.dtype, device=R.device)
        return self.block.pose.p[:, 2] - (R[:, 2, :].abs() * h).sum(-1)

    def grippers_open(self) -> torch.Tensor:
        ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for ag in self.agent.agents:
            o = ag.measured_opening()
            lim = 0.8 * ag.record.gripper_max_opening_m if ag.record.pad_links else 0.08
            ok &= o >= lim
        return ok

    def evaluate(self) -> dict[str, Any]:
        speed = torch.linalg.norm(self.block.linear_velocity, dim=-1)
        inside = self.cube_in_target()
        on_table = self.cube_bottom_z() <= D.TABLE_TOP_Z + 0.005
        opened = self.grippers_open()
        return dict(success=inside & on_table & (speed < 0.02) & opened, cube_in_target=inside, cube_on_table=on_table,
                    grippers_open=opened, cube_speed=speed)

    def _get_obs_extra(self, info: dict):
        obs = dict(target_pos=self.target_pos)
        for i, ag in enumerate(self.agent.agents):
            obs[f"tcp_pose_{i}"] = ag.tcp.pose.raw_pose
        if "state" in self.obs_mode:
            obs["block_pose"] = self.block.pose.raw_pose
        return obs
