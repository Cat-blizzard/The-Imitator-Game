"""Generic dual-arm smoke task: one cube, one target square, two arms (any DualConfig).

ImsimDualTabletop-v1(dual_config="dual_arx_x5", ...)
  scene    table (kinematic box, top at z = 0), cube (dynamic, half size config.cube_half), target square (visual only,
           half size config.target_half), lighting; robots mounted by DualConfig.mount_poses()
  reset    both arms at their record's home pose (+ optional joint noise), gripper/hand open; cube on the left or right
           side (episode RNG), yaw random in [-0.3, 0.3] rad; target in the centre strip
  success  (benchmark flag, for comparison only; the contract in imsim/contracts/ is what judges an episode):
           cube centre inside the target square, cube resting on the table (bottom within 5 mm of the top), cube speed
           < 2 cm/s, and neither arm grasping it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from ..embodiments import agents as _agents  # noqa: F401  (registers the imsim agents)
from ..embodiments import dual as D
from ..embodiments import registry as reg


@register_env("ImsimDualTabletop-v1", max_episode_steps=1000)
class DualTabletopEnv(BaseEnv):
    SUPPORTED_ROBOTS = [c.robot_uids for c in D.CONFIGS.values()]

    def __init__(self, *args, dual_config: str = "dual_arx_x5", robot_init_qpos_noise: float = 0.0,
                 cube_half: float | None = None, side: str | None = None, gpu_memory_scale: float = 1.0,
                 render_size: tuple = (640, 480), **kwargs):
        self.dual = D.CONFIGS[dual_config] if isinstance(dual_config, str) else dual_config
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cube_half = float(cube_half or self.dual.cube_half)
        self.forced_side = side                     # "left" / "right" / None (random per episode)
        self.gpu_memory_scale = gpu_memory_scale
        self.render_size = render_size
        self.base_table_links = [D.base_links_on_table(r, self.dual.base_z) for r in self.dual.records]
        self.group2_remaps = None
        super().__init__(*args, robot_uids=self.dual.robot_uids, **kwargs)

    # ---------------------------------------------------------------- config
    @property
    def _default_sim_config(self):
        s = self.gpu_memory_scale
        return SimConfig(gpu_memory_config=GPUMemoryConfig(
            found_lost_pairs_capacity=int(2 ** 24 * s), max_rigid_patch_count=int(2 ** 18 * s),
            max_rigid_contact_count=int(2 ** 20 * s)))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.95, 0.0, 0.65], target=[0.25, 0.0, 0.05])
        return [CameraConfig("front_camera", pose, 128, 128, np.pi / 2.5, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.25, 0.95, 0.85], target=[0.18, 0.0, 0.12])
        w, h = self.render_size
        return CameraConfig("render_camera", pose, w, h, 0.95, 0.01, 100)

    # ---------------------------------------------------------------- scene
    def _load_agent(self, options: dict):
        super()._load_agent(options, self.dual.mount_poses())

    def _load_scene(self, options: dict):
        b = self.scene.create_actor_builder()
        hx, hy = (D.TABLE_X[1] - D.TABLE_X[0]) / 2, (D.TABLE_Y[1] - D.TABLE_Y[0]) / 2
        cx, cy = (D.TABLE_X[1] + D.TABLE_X[0]) / 2, (D.TABLE_Y[1] + D.TABLE_Y[0]) / 2
        half = [hx, hy, D.TABLE_THICK / 2]
        b.add_box_collision(half_size=half)
        b.add_box_visual(half_size=half, material=sapien.render.RenderMaterial(base_color=[0.78, 0.70, 0.60, 1]))
        b.initial_pose = sapien.Pose([cx, cy, D.TABLE_TOP_Z - D.TABLE_THICK / 2])
        self.table = b.build_kinematic(name="table")
        self.cube = actors.build_cube(self.scene, half_size=self.cube_half, color=[0.85, 0.15, 0.12, 1], name="cube",
                                      initial_pose=sapien.Pose([0.3, 0.2, self.cube_half]))
        tb = self.scene.create_actor_builder()
        th = self.dual.target_half
        tb.add_box_visual(half_size=[th, th, 0.0008], material=sapien.render.RenderMaterial(base_color=[0.1, 0.75, 0.2, 1]))
        tb.initial_pose = sapien.Pose([0.3, 0.0, 0.0009])
        self.target = tb.build_kinematic(name="target")
        # collision filtering (before GPU init): disjoint group-2 bits per arm, base links vs table
        ags = list(self.agent.agents)
        self.group2_remaps = D.make_group2_bits_disjoint(ags)
        for ag, links in zip(ags, self.base_table_links):
            D.filter_base_table(ag, self.table, links)

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
            d = self.dual
            u = lambda lo, hi: torch.rand(b) * (hi - lo) + lo  # noqa: E731
            forced = (options or {}).get("side", self.forced_side)     # reset(options={"side": ...}) overrides
            if forced is None:
                sgn = torch.where(torch.rand(b) < 0.5, torch.ones(b), -torch.ones(b))
            else:
                sgn = torch.full((b,), 1.0 if forced == "left" else -1.0)
            p = torch.zeros((b, 3))
            p[:, 0] = u(*d.cube_x); p[:, 1] = sgn * u(*d.cube_abs_y); p[:, 2] = self.cube_half
            yaw = u(-0.3, 0.3)
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], -1)
            self.cube.set_pose(Pose.create_from_pq(p, q))
            t = torch.zeros((b, 3))
            t[:, 0] = u(*d.target_x); t[:, 1] = u(*d.target_y); t[:, 2] = 0.0009
            self.target.set_pose(Pose.create_from_pq(t))
            if not hasattr(self, "target_pos") or self.target_pos.shape[0] != self.num_envs:
                self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
                self.cube_side = torch.zeros(self.num_envs, device=self.device)
            self.target_pos[env_idx] = t
            self.cube_side[env_idx] = sgn

    # ---------------------------------------------------------------- task signals
    def cube_in_target(self) -> torch.Tensor:
        d = self.cube.pose.p[:, :2] - self.target_pos[:, :2]
        return (d.abs() <= self.dual.target_half).all(-1)

    def cube_bottom_z(self) -> torch.Tensor:
        """lowest point of the cube (exact for a box)"""
        R = self.cube.pose.to_transformation_matrix()[:, :3, :3]
        return self.cube.pose.p[:, 2] - self.cube_half * R[:, 2, :].abs().sum(-1)

    def evaluate(self) -> dict[str, Any]:
        grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for ag in self.agent.agents:
            grasped |= ag.is_grasping(self.cube)
        speed = torch.linalg.norm(self.cube.linear_velocity, dim=-1)
        inside = self.cube_in_target()
        on_table = self.cube_bottom_z() <= D.TABLE_TOP_Z + 0.005
        return dict(success=inside & on_table & (speed < 0.02) & ~grasped, cube_in_target=inside, is_grasped=grasped,
                    cube_on_table=on_table, cube_speed=speed)

    def _get_obs_extra(self, info: dict):
        obs = dict(target_pos=self.target_pos)
        for i, ag in enumerate(self.agent.agents):
            obs[f"tcp_pose_{i}"] = ag.tcp.pose.raw_pose
        if "state" in self.obs_mode:
            obs["cube_pose"] = self.cube.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)
