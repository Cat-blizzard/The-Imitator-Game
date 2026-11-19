"""DualLiftBox-v1 (E29b): cooperative two-handed lift of a long box, kept level.

Object: a 4 x 36 x 4 cm box (half extents (0.02, 0.18, 0.02), long axis = its local y, density 500 kg/m^3, 288 g),
lying on the table in the centre strip (centre x in [0.27, 0.33], y in [-0.02, 0.02], yaw in [-0.15, 0.15] rad about the
world-y-aligned pose). Each arm grasps the end on its side (13 cm from the centre) and both lift it together.
  scene    table top z = 0 (DualTabletopEnv), the box, a green marker square (visual only) under the box
  success  benchmark-style flag (written the way a RoboTwin 'lift_pot'-type check is written: one step, for comparison
           only; the contract judges episodes): box centre >= 5 cm above its reset height and the long axis within 10 deg of
           horizontal at the evaluated step. It does not check who holds the box (one arm, both arms, nothing), that the
           box is at rest, or that it stays up after the step.
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

BOX_HALF = (0.02, 0.18, 0.02)


class _DualSceneMixin:
    """table + robots reset shared by the E29b B1 dual tasks"""

    def _build_table(self):
        b = self.scene.create_actor_builder()
        hx, hy = (D.TABLE_X[1] - D.TABLE_X[0]) / 2, (D.TABLE_Y[1] - D.TABLE_Y[0]) / 2
        cx, cy = (D.TABLE_X[1] + D.TABLE_X[0]) / 2, (D.TABLE_Y[1] + D.TABLE_Y[0]) / 2
        half = [hx, hy, D.TABLE_THICK / 2]
        b.add_box_collision(half_size=half)
        b.add_box_visual(half_size=half, material=sapien.render.RenderMaterial(base_color=[0.78, 0.70, 0.60, 1]))
        b.initial_pose = sapien.Pose([cx, cy, D.TABLE_TOP_Z - D.TABLE_THICK / 2])
        self.table = b.build_kinematic(name="table")

    def _build_target(self, half: float, color=(0.1, 0.75, 0.2, 1), name: str = "target"):
        tb = self.scene.create_actor_builder()
        tb.add_box_visual(half_size=[half, half, 0.0008], material=sapien.render.RenderMaterial(base_color=list(color)))
        tb.initial_pose = sapien.Pose([0.3, 0.0, 0.0009])
        return tb.build_kinematic(name=name)

    def _filter_robots(self):
        ags = list(self.agent.agents)
        self.group2_remaps = D.make_group2_bits_disjoint(ags)
        for ag, links in zip(ags, self.base_table_links):
            D.filter_base_table(ag, self.table, links)

    def _reset_robots(self, b: int):
        for ag, pose in zip(self.agent.agents, self.dual.mount_poses()):
            q0 = torch.as_tensor(ag.home_qpos_vector(), dtype=torch.float32)
            q = q0.repeat(b, 1)
            if self.robot_init_qpos_noise > 0:
                arm_idx = [i for i, j in enumerate(ag.robot.active_joints) if j.name in ag.record.arm_joints]
                q[:, arm_idx] += torch.randn((b, len(arm_idx))) * self.robot_init_qpos_noise
            ag.reset(q)
            ag.robot.set_pose(Pose.create_from_pq(torch.as_tensor(pose.p, dtype=torch.float32).repeat(b, 1),
                                                  torch.as_tensor(pose.q, dtype=torch.float32).repeat(b, 1)))

    def _box_bottom_z(self, actor, half) -> torch.Tensor:
        R = actor.pose.to_transformation_matrix()[:, :3, :3]
        h = torch.as_tensor(np.asarray(half, float), dtype=R.dtype, device=R.device)
        return actor.pose.p[:, 2] - (R[:, 2, :].abs() * h).sum(-1)

    def _grippers_open(self) -> torch.Tensor:
        """benchmark-style 'gripper open' (RoboTwin reads the gripper's joint state): opening implied by the first gripper
        joint's position (free-space calibration imsim/verify/grip_calib.json) >= 80 % of the gripper's maximum"""
        from ..verify.v32 import grip_calib
        ok = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for ag in self.agent.agents:
            r = ag.record
            c = grip_calib().get(r.name)
            names = [j.name for j in ag.robot.active_joints]
            q = ag.robot.get_qpos()[:, names.index(r.gripper_joints[0])].cpu().numpy()
            t, o = np.asarray(c["target"], float), np.asarray(c["opening"], float)
            i = np.argsort(t)
            op = np.interp(q, t[i], o[i])
            ok &= torch.as_tensor(op >= 0.8 * r.gripper_max_opening_m, device=self.device)
        return ok


@register_env("DualLiftBox-v1", max_episode_steps=3000)
class DualLiftBoxEnv(_DualSceneMixin, DualTabletopEnv):
    def __init__(self, *args, box_half=BOX_HALF, box_density: float = 500.0, box_x=(0.27, 0.33), box_y=(-0.02, 0.02),
                 yaw_range: float = 0.15, **kwargs):
        self.box_half = np.asarray(box_half, float)
        self.box_density = float(box_density)
        self.box_x, self.box_y, self.yaw_range = tuple(box_x), tuple(box_y), float(yaw_range)
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self._build_table()
        bb = self.scene.create_actor_builder()
        h = self.box_half.tolist()
        bb.add_box_collision(half_size=h, density=self.box_density)
        bb.add_box_visual(half_size=h, material=sapien.render.RenderMaterial(base_color=[0.85, 0.55, 0.15, 1]))
        bb.initial_pose = sapien.Pose([0.3, 0.0, h[2]])
        self.box = bb.build(name="box")
        self.cube = self.box
        self.target = self._build_target(0.03, color=(0.1, 0.75, 0.2, 1))
        self._filter_robots()

    @property
    def cube_half(self):
        return self.box_half

    @cube_half.setter
    def cube_half(self, v):
        pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robots(b)
            u = lambda lo, hi: torch.rand(b) * (hi - lo) + lo  # noqa: E731
            p = torch.zeros((b, 3))
            p[:, 0] = u(*self.box_x); p[:, 1] = u(*self.box_y); p[:, 2] = float(self.box_half[2])
            yaw = u(-self.yaw_range, self.yaw_range)
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], -1)
            self.box.set_pose(Pose.create_from_pq(p, q))
            t = p.clone(); t[:, 2] = 0.0009
            self.target.set_pose(Pose.create_from_pq(t))
            if not hasattr(self, "target_pos") or self.target_pos.shape[0] != self.num_envs:
                self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
                self.box_z0 = torch.zeros(self.num_envs, device=self.device)
                self.cube_side = torch.ones(self.num_envs, device=self.device)
            self.target_pos[env_idx] = t
            self.box_z0[env_idx] = p[:, 2]

    def box_long_tilt(self) -> torch.Tensor:
        """angle of the box's long axis (local y) to the horizontal plane [rad]"""
        R = self.box.pose.to_transformation_matrix()[:, :3, :3]
        return torch.asin(R[:, 2, 1].abs().clamp(max=1.0))

    def evaluate(self) -> dict[str, Any]:
        lifted = self.box.pose.p[:, 2] >= self.box_z0 + 0.05
        level = self.box_long_tilt() < np.radians(10.0)
        return dict(success=lifted & level, box_lifted=lifted, box_level=level)

    def _get_obs_extra(self, info: dict):
        obs = dict(target_pos=self.target_pos)
        for i, ag in enumerate(self.agent.agents):
            obs[f"tcp_pose_{i}"] = ag.tcp.pose.raw_pose
        if "state" in self.obs_mode:
            obs["box_pose"] = self.box.pose.raw_pose
        return obs
