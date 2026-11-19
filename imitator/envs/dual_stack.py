"""DualStack-v1 (E29b): each arm places one cube, forming a two-cube stack.

Objects: cube A (red, on the left side: x in [0.25, 0.36], y in [0.14, 0.20]) and cube B (blue, right side: y in
[-0.20, -0.14]), both of half size DualConfig.cube_half (2.5 cm), yaw in [-0.3, 0.3]; a green target square (half
target_half = 5 cm) in the centre strip (x in [0.27, 0.35], y in [-0.02, 0.02]).
Instruction: the LEFT arm puts the red cube on the green square, the RIGHT arm puts the blue cube on top of the red
one; both let go and move away.
  success  benchmark-style flag (written the way RoboTwin 2.0's stack_blocks_two check is written: one step, for
           comparison only; the contract judges episodes): A's centre inside the target square, B within (2.5 cm, 2.5 cm,
           1.2 cm) of A + (0, 0, 2 h), and both grippers open (the opening implied by the gripper joint position >= 80 % of the
           maximum, as RoboTwin reads the gripper state) at the evaluated step. It does not check which arm carried which cube, contact / support, rest, or what happens after the step.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .dual_lift_box import _DualSceneMixin
from .dual_tabletop import DualTabletopEnv


@register_env("DualStack-v1", max_episode_steps=3000)
class DualStackEnv(_DualSceneMixin, DualTabletopEnv):
    def __init__(self, *args, zone_x=(0.25, 0.36), zone_abs_y=(0.14, 0.20), target_x=(0.27, 0.35),
                 target_y=(-0.02, 0.02), yaw_range: float = 0.3, **kwargs):
        self.zone_x, self.zone_abs_y = tuple(zone_x), tuple(zone_abs_y)
        self.tx, self.ty, self.yaw_range = tuple(target_x), tuple(target_y), float(yaw_range)
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self._build_table()
        h = self.cube_half
        self.cubeA = actors.build_cube(self.scene, half_size=h, color=[0.85, 0.15, 0.12, 1], name="cubeA",
                                       initial_pose=sapien.Pose([0.3, 0.17, h]))
        self.cubeB = actors.build_cube(self.scene, half_size=h, color=[0.15, 0.35, 0.85, 1], name="cubeB",
                                       initial_pose=sapien.Pose([0.3, -0.17, h]))
        self.cube = self.cubeA
        self.target = self._build_target(self.dual.target_half)
        self._filter_robots()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robots(b)
            u = lambda lo, hi: torch.rand(b) * (hi - lo) + lo  # noqa: E731
            for cube, sgn in ((self.cubeA, 1.0), (self.cubeB, -1.0)):
                p = torch.zeros((b, 3))
                p[:, 0] = u(*self.zone_x); p[:, 1] = sgn * u(*self.zone_abs_y); p[:, 2] = self.cube_half
                yaw = u(-self.yaw_range, self.yaw_range)
                q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], -1)
                cube.set_pose(Pose.create_from_pq(p, q))
            t = torch.zeros((b, 3))
            t[:, 0] = u(*self.tx); t[:, 1] = u(*self.ty); t[:, 2] = 0.0009
            self.target.set_pose(Pose.create_from_pq(t))
            if not hasattr(self, "target_pos") or self.target_pos.shape[0] != self.num_envs:
                self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
                self.cube_side = torch.ones(self.num_envs, device=self.device)
            self.target_pos[env_idx] = t

    def evaluate(self) -> dict[str, Any]:
        h = self.cube_half
        a, bp = self.cubeA.pose.p, self.cubeB.pose.p
        in_target = ((a[:, :2] - self.target_pos[:, :2]).abs() <= self.dual.target_half).all(-1)
        d = bp - a
        on_a = (d[:, 0].abs() <= 0.025) & (d[:, 1].abs() <= 0.025) & ((d[:, 2] - 2 * h).abs() <= 0.012)
        opened = self._grippers_open()
        return dict(success=in_target & on_a & opened, a_in_target=in_target, b_on_a=on_a, grippers_open=opened)

    def _get_obs_extra(self, info: dict):
        obs = dict(target_pos=self.target_pos)
        for i, ag in enumerate(self.agent.agents):
            obs[f"tcp_pose_{i}"] = ag.tcp.pose.raw_pose
        if "state" in self.obs_mode:
            obs["cubeA_pose"] = self.cubeA.pose.raw_pose
            obs["cubeB_pose"] = self.cubeB.pose.raw_pose
        return obs
