"""One two-object pick/place task shared by fixed-base robot pairs.

Profiles describe reach and mounting. Objects, success predicates, observations
and recording share the same task. Units: metres, radians, WXYZ quaternions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig


@dataclass(frozen=True)
class PickPlaceProfile:
    base_x: float
    arm_separation: float
    source_x: float
    destination_x: float
    grasp_height_offset: float = 0.
    base_height: float = 0.
    grasp_quaternion: tuple[float, float, float, float] = (0., 1., 0., 0.)
    approach_height: float = .10
    carry_height: float = .16


PROFILES = {
    # The equivalent top-down grasp keeps joint6 away from its +/-pi limits.
    "piper": PickPlaceProfile(-.36, .48, -.04, .06,
                             grasp_height_offset=.012, base_height=.16,
                             grasp_quaternion=(0., 0., 1., 0.)),
    "panda": PickPlaceProfile(-.60, .60, -.08, .12),
}


@register_env("DualArmPickPlace-v1", max_episode_steps=400)
class DualArmPickPlaceEnv(BaseEnv):
    """Place each cube in its matching region and release it on the table.

    This integration task has a 400-step horizon. Existing TIG benchmark task
    horizons and success conditions are unaffected.
    """

    SUPPORTED_ROBOTS = [(name, name) for name in PROFILES]
    SUPPORTED_REWARD_MODES = ["none"]
    cube_half_size = .018
    region_half_size = .055

    def __init__(self, *args, robot_uids=("piper", "piper"), **kwargs):
        robot_uids = tuple(robot_uids)
        if robot_uids not in self.SUPPORTED_ROBOTS:
            raise ValueError(f"Choose a supported robot pair: {self.SUPPORTED_ROBOTS}")
        self.profile = PROFILES[robot_uids[0]]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([.65, -.8, .8], [-.05, 0, .12])
        return [CameraConfig("base_camera", pose, 256, 256, 1.15, .01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([.75, -1., .9], [-.05, 0, .15])
        return CameraConfig("render_camera", pose, 512, 512, 1.15, .01, 100)

    def _load_agent(self, options: dict):
        p = self.profile
        self.base_poses = [sapien.Pose([p.base_x, sign*p.arm_separation/2, p.base_height+.002])
                           for sign in (-1, 1)]
        super()._load_agent(options, self.base_poses)
        self.left_agent, self.right_agent = self.agent.agents

    def _load_scene(self, options: dict):
        self.table = actors.build_box(self.scene, half_sizes=[.85, .65, .025],
            color=[.65, .56, .45, 1], name="table", body_type="static",
            initial_pose=sapien.Pose([0, 0, -.025]))
        self.ground = build_ground(self.scene, altitude=-.8)
        if self.profile.base_height > 0:
            p = self.profile
            for i, sign in enumerate((-1, 1)):
                actors.build_box(self.scene, half_sizes=[.065, .065, p.base_height/2],
                    color=[.3, .3, .33, 1], name=f"mount_{i}", body_type="static",
                    initial_pose=sapien.Pose([p.base_x, sign*p.arm_separation/2, p.base_height/2]))
        self.targets = tuple(actors.build_cube(self.scene, half_size=self.cube_half_size,
            color=color, name=f"cube_{index}", initial_pose=sapien.Pose([0, 0, .1]))
            for index, color in enumerate(([.85, .15, .12, 1], [.12, .35, .85, 1])))
        self.regions = tuple(actors.build_box(self.scene,
            half_sizes=[self.region_half_size]*2+[.0005], color=color,
            name=f"region_{index}", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose([0, 0, .0005]))
            for index, color in enumerate(([.9, .4, .35, 1], [.35, .55, .9, 1])))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.num_envs != 1:
            raise ValueError("The integration example currently supports one CPU environment")
        for agent, pose in zip(self.agent.agents, self.base_poses):
            agent.reset(agent.keyframes["rest"].qpos)
            agent.robot.set_pose(pose)
        p = self.profile
        jitter = self._episode_rng.uniform(-.008, .008, (2, 2))
        self.destination_xy = []
        for i, sign in enumerate((-1, 1)):
            y = sign*p.arm_separation/2
            self.targets[i].set_pose(sapien.Pose(
                [p.source_x+jitter[i, 0], y+jitter[i, 1], self.cube_half_size+.001]))
            xy = [p.destination_x, y]
            self.destination_xy.append(xy)
            self.regions[i].set_pose(sapien.Pose([*xy, .0005]))

    def region_contains(self, index: int) -> torch.Tensor:
        actor = self.targets[index]
        matrix = actor.pose.to_transformation_matrix()
        half = self.cube_half_size * torch.abs(matrix[..., :3, :3]).sum(-1)
        centre = actor.pose.p
        goal = torch.as_tensor(self.destination_xy[index], device=self.device)
        inside_xy = (torch.abs(centre[..., :2]-goal)+half[..., :2]
                     <= self.region_half_size).all(-1)
        on_table = (centre[..., 2]-half[..., 2]).abs() <= .006
        return inside_xy & on_table

    def evaluate(self):
        inside = tuple(self.region_contains(i) for i in range(2))
        released = tuple(~agent.is_grasping(target) for agent, target
                         in zip(self.agent.agents, self.targets))
        static = tuple(agent.is_static(.05) for agent in self.agent.agents)
        return {"left_in_region": inside[0], "right_in_region": inside[1],
                "success": inside[0] & inside[1] & released[0] & released[1]
                           & static[0] & static[1]}

    def _get_obs_extra(self, info: dict):
        return {"left_tcp_pose": self.left_agent.tcp.pose.raw_pose,
                "right_tcp_pose": self.right_agent.tcp.pose.raw_pose,
                "left_object_pose": self.targets[0].pose.raw_pose,
                "right_object_pose": self.targets[1].pose.raw_pose}


if __name__ == "__main__":
    assert set(PROFILES) == {"piper", "panda"}
    for profile in PROFILES.values():
        assert profile.source_x > profile.base_x
        assert np.isclose(np.linalg.norm(profile.grasp_quaternion), 1)
    print("Pick/place profiles: valid units, mounting and quaternions")
