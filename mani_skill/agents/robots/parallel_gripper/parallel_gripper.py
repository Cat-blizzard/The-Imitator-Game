"""Fixed-base arms with a common absolute-joint / normalized-gripper interface.

URDFs are prepared outside the source tree by scripts/prepare_embodiment_assets.py.
The controller takes ``[arm joint positions (rad), gripper (-1 closed, +1 open)]``.
"""
from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.embodiment_profiles import get_robot_profile
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


class ParallelGripperArm(BaseAgent):
    """A two-finger arm whose imported gripper coordinates both increase outward."""

    ee_link_name = "grasp_tcp"
    gripper_open_action = 1.0
    gripper_closed_action = -1.0
    arm_stiffness = 1000
    arm_damping = 100
    arm_force_limit = 100
    gripper_stiffness = 1000
    gripper_damping = 50
    gripper_force_limit = 10
    rest_arm_qpos = (0.0,) * 6
    # Local fingertip axes pointing outwards (the direction of contact on the object).
    finger_open_directions = ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))

    @property
    def _controller_configs(self):
        arm = PDJointPosControllerConfig(
            self.arm_joint_names, lower=None, upper=None,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, normalize_action=False,
        )
        gripper = PDJointPosMimicControllerConfig(
            self.gripper_joint_names, lower=self.gripper_closed_qpos,
            upper=self.gripper_open_qpos, stiffness=self.gripper_stiffness,
            damping=self.gripper_damping, force_limit=self.gripper_force_limit,
            normalize_action=True,
        )
        delta = deepcopy(arm)
        delta.lower, delta.upper = -0.1, 0.1
        delta.use_delta, delta.normalize_action = True, True
        return {
            "pd_joint_pos": {"arm": arm, "gripper": gripper},
            "pd_joint_delta_pos": {"arm": delta, "gripper": deepcopy(gripper)},
        }

    def _after_loading_articulation(self) -> None:
        # Equal drive targets alone let a force-saturated parallel gripper's
        # centre translate. Represent the Piper's single opening coordinate as
        # q7=q8, using the same finite tendon stiffness as the URDF mimic loader.
        coupled = self.embodiment_profile.coupled_gripper_joints
        self.gripper_tendon_count = 0
        if coupled is not None:
            left, right = (self.robot.active_joints_map[name] for name in coupled)
            for root, first, second, articulation in zip(
                    left.get_parent_link()._objs, left.get_child_link()._objs,
                    right.get_child_link()._objs, self.robot._objs):
                articulation.create_fixed_tendon(
                    [root, first, second], [0., -1., 1.], [0., -1., 1.],
                    rest_length=0., stiffness=1e5)
                self.gripper_tendon_count += 1
        # Keyframe qpos follows the loader's active joint order, not a URDF guess.
        names = [joint.name for joint in self.robot.get_active_joints()]
        values = dict(zip(self.arm_joint_names, self.rest_arm_qpos))
        values.update({name: self.gripper_open_qpos for name in self.gripper_joint_names})
        if set(names) != set(values):
            raise ValueError(f"Unexpected active joints for {self.uid}: {names}")
        self.keyframes = {
            "rest": Keyframe(qpos=np.asarray([values[name] for name in names]), pose=sapien.Pose())
        }
        self.arm_joint_indices = torch.tensor(
            [names.index(name) for name in self.arm_joint_names], device=self.device
        )
        self.gripper_joint_indices = torch.tensor(
            [names.index(name) for name in self.gripper_joint_names], device=self.device
        )

    def _after_init(self) -> None:
        self.finger1_link, self.finger2_link = [
            sapien_utils.get_obj_by_name(self.robot.get_links(), name)
            for name in self.finger_link_names
        ]
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    def reset(self, init_qpos: torch.Tensor = None) -> None:
        if init_qpos is None:
            init_qpos = self.keyframes["rest"].qpos
        super().reset(init_qpos)

    def is_static(self, threshold: float = 0.2) -> torch.Tensor:
        return self.robot.get_qvel()[..., self.arm_joint_indices].abs().amax(dim=-1) <= threshold

    def is_grasping(self, object: Actor, min_force: float = 0.5, max_angle: float = 85) -> torch.Tensor:
        """Require actual contact forces on both fingers, directed into the object."""
        flags = []
        for link, local_axis in zip((self.finger1_link, self.finger2_link), self.finger_open_directions):
            forces = self.scene.get_pairwise_contact_forces(link, object)
            axis = torch.as_tensor(local_axis, dtype=forces.dtype, device=self.device)
            outward = link.pose.to_transformation_matrix()[..., :3, :3] @ axis
            angle = common.compute_angle_between(outward, forces)
            flags.append((forces.norm(dim=-1) >= min_force) & (torch.rad2deg(angle) <= max_angle))
        return flags[0] & flags[1]

    @staticmethod
    def build_grasp_pose(approaching, closing, center) -> sapien.Pose:
        # Same TCP convention as Panda: +z approaches, +y is the closing axis.
        approaching, closing = np.asarray(approaching), np.asarray(closing)
        assert np.isclose(np.linalg.norm(approaching), 1)
        assert np.isclose(np.linalg.norm(closing), 1)
        assert np.isclose(approaching @ closing, 0)
        transform = np.eye(4)
        transform[:3, :3] = np.stack((np.cross(closing, approaching), closing, approaching), axis=1)
        transform[:3, 3] = center
        return sapien.Pose(transform)


def _finger_material(links):
    return {
        "_materials": {"gripper": {"static_friction": 2.0, "dynamic_friction": 2.0, "restitution": 0.0}},
        "link": {name: {"material": "gripper", "patch_radius": 0.01, "min_patch_radius": 0.001} for name in links},
    }


@register_agent()
class Piper(ParallelGripperArm):
    uid = "piper"
    embodiment_profile = get_robot_profile(uid)
    urdf_path = "{ASSET_DIR}/robots/piper/robot.urdf"
    arm_joint_names = list(embodiment_profile.arm_joint_names)
    gripper_joint_names = list(embodiment_profile.gripper_joint_names)
    rest_arm_qpos = embodiment_profile.rest_arm_qpos
    ee_link_name = embodiment_profile.ee_link_name
    finger_link_names = embodiment_profile.finger_link_names
    gripper_closed_qpos = embodiment_profile.gripper_closed_qpos
    gripper_open_qpos = embodiment_profile.gripper_open_qpos
    gripper_open_action = embodiment_profile.gripper_open_action
    gripper_closed_action = embodiment_profile.gripper_closed_action
    gripper_coordinate_unit = embodiment_profile.gripper_coordinate_unit
    gripper_open_tolerance = embodiment_profile.gripper_open_tolerance
    finger_open_directions = embodiment_profile.finger_open_directions
    urdf_config = _finger_material(finger_link_names)


@register_agent()
class ARXX5(ParallelGripperArm):
    uid = "arx_x5"
    embodiment_profile = get_robot_profile(uid)
    urdf_path = "{ASSET_DIR}/robots/arx_x5/robot.urdf"
    arm_joint_names = list(embodiment_profile.arm_joint_names)
    gripper_joint_names = list(embodiment_profile.gripper_joint_names)
    rest_arm_qpos = embodiment_profile.rest_arm_qpos
    ee_link_name = embodiment_profile.ee_link_name
    finger_link_names = embodiment_profile.finger_link_names
    gripper_closed_qpos = embodiment_profile.gripper_closed_qpos
    gripper_open_qpos = embodiment_profile.gripper_open_qpos
    gripper_open_action = embodiment_profile.gripper_open_action
    gripper_closed_action = embodiment_profile.gripper_closed_action
    gripper_coordinate_unit = embodiment_profile.gripper_coordinate_unit
    gripper_open_tolerance = embodiment_profile.gripper_open_tolerance
    finger_open_directions = embodiment_profile.finger_open_directions
    urdf_config = _finger_material(finger_link_names)


@register_agent()
class UR5WSG(ParallelGripperArm):
    uid = "ur5_wsg"
    embodiment_profile = get_robot_profile(uid)
    urdf_path = "{ASSET_DIR}/robots/ur5_wsg/robot.urdf"
    arm_joint_names = list(embodiment_profile.arm_joint_names)
    gripper_joint_names = list(embodiment_profile.gripper_joint_names)
    rest_arm_qpos = embodiment_profile.rest_arm_qpos
    ee_link_name = embodiment_profile.ee_link_name
    finger_link_names = embodiment_profile.finger_link_names
    gripper_closed_qpos = embodiment_profile.gripper_closed_qpos
    gripper_open_qpos = embodiment_profile.gripper_open_qpos
    gripper_open_action = embodiment_profile.gripper_open_action
    gripper_closed_action = embodiment_profile.gripper_closed_action
    gripper_coordinate_unit = embodiment_profile.gripper_coordinate_unit
    gripper_open_tolerance = embodiment_profile.gripper_open_tolerance
    finger_open_directions = embodiment_profile.finger_open_directions
    # The URDF has a massless world root above its mounting body.
    robot_root_link_name = embodiment_profile.robot_root_link_name
    urdf_config = _finger_material(finger_link_names)

if __name__ == "__main__":
    for cls in (Piper, ARXX5, UR5WSG):
        assert len(cls.arm_joint_names) == 6
        assert len(set(cls.arm_joint_names + cls.gripper_joint_names)) == 8
        assert cls.gripper_closed_qpos < cls.gripper_open_qpos
        pose = cls.build_grasp_pose([0, 0, 1], [0, 1, 0], [0.1, 0.2, 0.3])
        assert np.allclose(pose.p, [0.1, 0.2, 0.3])
        print(cls.uid, "arm=6, physical fingers=2, action=7")
