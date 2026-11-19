"""Named fixed-base robot contracts, independent of simulation initialization.

Joint lists describe physical coordinates, including passive gripper joints.
The scalar gripper action is distinct from those coordinates. Arm coordinates
are radians; gripper coordinates carry their own unit. No asset or planner is
loaded here. Agent classes consume these values when constructing controllers.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from types import MappingProxyType


@dataclass(frozen=True)
class EmbodimentProfile:
    uid: str
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    ee_link_name: str
    finger_link_names: tuple[str, str]
    rest_arm_qpos: tuple[float, ...]
    gripper_open_qpos: float
    gripper_closed_qpos: float | None = None
    gripper_open_action: float = 1.
    gripper_closed_action: float = -1.
    gripper_coordinate_unit: str = "m"
    gripper_open_tolerance: float = .002
    actuated_gripper_joint_names: tuple[str, ...] = ()
    finger_open_directions: tuple[tuple[float, float, float], ...] = (
        (0., 0., -1.), (0., 0., -1.))
    robot_root_link_name: str | None = None
    prefer_nearby_revolute_goals: bool = False
    coupled_gripper_joints: tuple[str, str] | None = None
    # Native generalized-coordinate tendon values: N/m or Nm/rad and damping.
    gripper_coupling_stiffness: float = 1e5
    gripper_coupling_damping: float = 0.

    @property
    def action_size(self) -> int:
        """Absolute arm targets followed by one native gripper action."""
        return len(self.arm_joint_names) + 1

    @property
    def controlled_gripper_joints(self) -> tuple[str, ...]:
        return self.actuated_gripper_joint_names or self.gripper_joint_names

    @property
    def passive_gripper_joints(self) -> tuple[str, ...]:
        return tuple(name for name in self.gripper_joint_names
                     if name not in self.controlled_gripper_joints)


ROBOT_PROFILES = MappingProxyType({
    "panda": EmbodimentProfile(
        uid="panda", arm_joint_names=tuple(f"panda_joint{i}" for i in range(1, 8)),
        gripper_joint_names=("panda_finger_joint1", "panda_finger_joint2"),
        ee_link_name="panda_hand_tcp",
        finger_link_names=("panda_leftfinger", "panda_rightfinger"),
        rest_arm_qpos=(0., pi/8, 0., -pi*5/8, 0., pi*3/4, -pi/4),
        gripper_open_qpos=.04, gripper_closed_qpos=-.01,
        finger_open_directions=((0., 1., 0.), (0., -1., 0.))),
    # The unfolded rest pose avoids the source zero pose's link2/link4 contact.
    "piper": EmbodimentProfile(
        uid="piper", arm_joint_names=tuple(f"joint{i}" for i in range(1, 7)),
        gripper_joint_names=("joint7", "joint8"), ee_link_name="grasp_tcp",
        finger_link_names=("link7", "link8"), rest_arm_qpos=(0., .8, -.8, 0., .2, 0.),
        gripper_open_qpos=.04, gripper_closed_qpos=0.,
        coupled_gripper_joints=("joint7", "joint8")),
    "arx_x5": EmbodimentProfile(
        uid="arx_x5", arm_joint_names=tuple(f"joint{i}" for i in range(1, 7)),
        gripper_joint_names=("joint7", "joint8"), ee_link_name="grasp_tcp",
        finger_link_names=("link7", "link8"), rest_arm_qpos=(0.,)*6,
        gripper_open_qpos=.044, gripper_closed_qpos=0.,
        prefer_nearby_revolute_goals=True,
        coupled_gripper_joints=("joint7", "joint8"),
        finger_open_directions=((0., 1., 0.), (0., -1., 0.))),
    "ur5_wsg": EmbodimentProfile(
        uid="ur5_wsg", arm_joint_names=("shoulder_pan_joint", "shoulder_lift_joint",
            "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
        gripper_joint_names=("base_joint_gripper_left", "base_joint_gripper_right"),
        ee_link_name="grasp_tcp", finger_link_names=("finger_left", "finger_right"),
        rest_arm_qpos=(-1.5447, -1.5447, -1.5447, -1.5794, 1.5794, 0.),
        gripper_open_qpos=.055, gripper_closed_qpos=.0027,
        finger_open_directions=((-1., 0., 0.), (-1., 0., 0.)),
        robot_root_link_name="base_link",
        coupled_gripper_joints=("base_joint_gripper_left", "base_joint_gripper_right")),
    "xarm6_robotiq": EmbodimentProfile(
        uid="xarm6_robotiq", arm_joint_names=tuple(f"joint{i}" for i in range(1, 7)),
        gripper_joint_names=("left_outer_knuckle_joint", "left_inner_knuckle_joint",
            "right_outer_knuckle_joint", "right_inner_knuckle_joint",
            "left_inner_finger_joint", "right_inner_finger_joint"),
        ee_link_name="eef",
        finger_link_names=("left_inner_finger_pad", "right_inner_finger_pad"),
        rest_arm_qpos=(0., .22, -1.23, 0., 1.01, 0.), gripper_open_qpos=0.,
        gripper_open_action=0., gripper_closed_action=.81,
        gripper_coordinate_unit="rad", robot_root_link_name="link_base",
        actuated_gripper_joint_names=("left_outer_knuckle_joint", "right_outer_knuckle_joint"),
        coupled_gripper_joints=("left_outer_knuckle_joint", "right_outer_knuckle_joint"),
        gripper_coupling_stiffness=1e3, gripper_coupling_damping=10.,
        prefer_nearby_revolute_goals=True),
})


def get_robot_profile(uid: str) -> EmbodimentProfile:
    """Return the base contract; a wrist camera does not change Panda kinematics."""
    canonical = "panda" if uid == "panda_wristcam" else uid
    try:
        return ROBOT_PROFILES[canonical]
    except KeyError:
        raise ValueError(f"Unsupported fixed-base robot {uid!r}; choose {tuple(ROBOT_PROFILES)}") from None


if __name__ == "__main__":
    for profile in ROBOT_PROFILES.values():
        names = profile.arm_joint_names + profile.gripper_joint_names
        assert len(names) == len(set(names))
        assert len(profile.rest_arm_qpos) == len(profile.arm_joint_names)
        assert profile.gripper_open_action != profile.gripper_closed_action
        assert profile.gripper_coordinate_unit in ("m", "rad")
        print(profile.uid, "action", profile.action_size, "physical qpos", len(names))
