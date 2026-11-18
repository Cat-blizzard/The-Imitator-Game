"""Shared robot/task configuration preserves native controller contracts."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mani_skill.agents.robots.embodiment_profiles import ROBOT_PROFILES, get_robot_profile
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.robots.parallel_gripper import ARXX5, Piper, UR5WSG
from mani_skill.agents.robots.xarm6.xarm6_robotiq import XArm6Robotiq
from mani_skill.envs.tasks.tabletop.dual_arm_profiles import (
    MountPose, PICK_PLACE_PROFILES, PILL_PROFILES, PickPlaceProfile, PillTaskProfile,
    get_task_profile,
)
from mani_skill.envs.tasks.tabletop.dual_pick_place import PROFILES


@pytest.mark.parametrize("agent_cls", [Panda, PandaWristCam, Piper, ARXX5, UR5WSG, XArm6Robotiq])
def test_agents_consume_their_named_profile(agent_cls):
    profile = get_robot_profile(agent_cls.uid)
    assert agent_cls.embodiment_profile is profile
    assert tuple(agent_cls.arm_joint_names) == profile.arm_joint_names
    assert tuple(agent_cls.gripper_joint_names) == profile.gripper_joint_names
    assert agent_cls.ee_link_name == profile.ee_link_name
    assert agent_cls.finger_link_names == profile.finger_link_names
    assert agent_cls.gripper_open_action == profile.gripper_open_action
    assert agent_cls.gripper_closed_action == profile.gripper_closed_action
    assert agent_cls.gripper_open_qpos == profile.gripper_open_qpos
    assert agent_cls.gripper_coordinate_unit == profile.gripper_coordinate_unit
    assert agent_cls.gripper_open_tolerance == profile.gripper_open_tolerance
    agent = object.__new__(agent_cls)
    controllers = agent._controller_configs["pd_joint_pos"]
    assert tuple(controllers["arm"].joint_names) == profile.arm_joint_names
    assert not controllers["arm"].normalize_action
    gripper = controllers.get("gripper", controllers.get("gripper_active"))
    assert tuple(gripper.joint_names) == profile.controlled_gripper_joints
    assert gripper.normalize_action is (profile.uid != "xarm6_robotiq")
    if profile.uid == "xarm6_robotiq":
        assert tuple(controllers["gripper_passive"].joint_names) == profile.passive_gripper_joints
        assert len(profile.gripper_joint_names) == 6
        assert len(profile.passive_gripper_joints) == 4


def test_native_rest_and_gripper_values_preserve_verified_baseline():
    assert np.array_equal(Panda.keyframes["rest"].qpos,
        [0., np.pi/8, 0., -np.pi*5/8, 0., np.pi*3/4, -np.pi/4, .04, .04])
    assert np.array_equal(XArm6Robotiq.keyframes["rest"].qpos,
        [0., .22, -1.23, 0., 1.01, 0., 0., 0., 0., 0., 0., 0.])
    assert Piper.rest_arm_qpos == (0., .8, -.8, 0., .2, 0.)
    assert ARXX5.rest_arm_qpos == (0.,)*6
    assert UR5WSG.rest_arm_qpos == (-1.5447, -1.5447, -1.5447, -1.5794, 1.5794, 0.)
    expected = {"panda": (8, .04, -.01, 1., -1., "m"),
                "piper": (7, .04, 0., 1., -1., "m"),
                "arx_x5": (7, .044, 0., 1., -1., "m"),
                "ur5_wsg": (7, .055, .0027, 1., -1., "m"),
                "xarm6_robotiq": (7, 0., None, 0., .81, "rad")}
    for uid, profile in ROBOT_PROFILES.items():
        assert (profile.action_size, profile.gripper_open_qpos, profile.gripper_closed_qpos,
                profile.gripper_open_action, profile.gripper_closed_action,
                profile.gripper_coordinate_unit) == expected[uid]
    assert get_robot_profile("panda_wristcam") is ROBOT_PROFILES["panda"]
    with pytest.raises(FrozenInstanceError):
        ROBOT_PROFILES["piper"].ee_link_name = "changed"


@pytest.mark.parametrize("uid,expected", [
    ("piper", (-.36, .48, -.04, .06, .012, .16, (0., 0., 1., 0.))),
    ("panda", (-.60, .60, -.08, .12, 0., 0., (0., 1., 0., 0.))),
    ("arx_x5", (-.40, .60, -.04, .08, .022, .16, (0., 0., 1., 0.))),
    ("ur5_wsg", (-.55, .75, -.05, .10, .028, 0., (0., 1., 0., 0.))),
    ("xarm6_robotiq", (-.55, .70, -.05, .10, 0., 0., (0., 1., 0., 0.))),
])
def test_task_arrangements_preserve_verified_sampling_and_mounting(uid, expected):
    profile = get_task_profile("DualArmPickPlace-v1", uid)
    assert isinstance(profile, PickPlaceProfile)
    assert PROFILES is PICK_PLACE_PROFILES and PROFILES[uid] is profile
    assert (profile.base_x, profile.arm_separation, profile.source_x,
            profile.destination_x, profile.grasp_height_offset, profile.base_height,
            profile.grasp_quaternion) == expected
    assert profile.approach_height == .10 and profile.carry_height == .16
    assert profile.source_jitter_m == .008
    for sign, mount, workspace in zip((-1, 1), profile.mount_poses, profile.source_workspaces_xy):
        assert mount.position == (profile.base_x, sign*profile.arm_separation/2,
                                  profile.base_height+.002)
        assert mount.quaternion == (1., 0., 0., 0.)
        assert workspace == ((profile.source_x-.008, profile.source_x+.008),
                              (sign*profile.arm_separation/2-.008,
                               sign*profile.arm_separation/2+.008))


def test_task_lookup_reports_unsupported_combination_and_keeps_mount_rotations():
    with pytest.raises(ValueError, match="Unsupported fixed-base robot"):
        get_robot_profile("unknown")
    with pytest.raises(ValueError, match="no profile.*supported robots"):
        get_task_profile("unregistered-task", "piper")
    # A task may mount the arms differently and retain a distinct grasp per arm.
    mounts = (MountPose((-.6, -.35, 0.)), MountPose((-.6, .35, 0.), (0., 0., 0., 1.)))
    recipe = PillTaskProfile(mounts, (.01, .02), (.10, .12),
                            ((0., 1., 0., 0.), (0., 0., 1., 0.)))
    assert recipe.mount_poses == mounts
    assert recipe.grasp_height_offsets_m == (.01, .02)
    assert recipe.pregrasp_clearances_m == (.10, .12)


def test_original_pill_task_profile_preserves_panda_and_selects_supported_robots():
    original = get_task_profile("TwoRobotPickPillToRegions-v1", "panda_wristcam")
    assert original is PILL_PROFILES["panda"]
    assert tuple(mount.position for mount in original.mount_poses) == (
        (-.60, -.35, 0.), (-.60, .35, 0.))
    assert original.grasp_height_offsets_m == (.12, .13)
    assert original.pregrasp_clearances_m == (.09, .08)
    assert original.grasp_quaternions == (None, None)
    assert get_task_profile("TwoRobotPickPillToRegions-v1", "piper") is PILL_PROFILES["piper"]
    with pytest.raises(ValueError, match="no profile.*supported robots"):
        get_task_profile("TwoRobotPickPillToRegions-v1", "ur5_wsg")
