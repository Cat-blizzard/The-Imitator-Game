"""Task-specific mounting, sampling workspace and grasp configuration.

Positions are metres in task world coordinates; quaternions use WXYZ. These
profiles describe task arrangements, not a proof of an arm's full workspace.
Robot joints, TCP and native gripper mapping live in embodiment_profiles.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from mani_skill.agents.robots.embodiment_profiles import get_robot_profile


@dataclass(frozen=True)
class MountPose:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1., 0., 0., 0.)


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
    source_jitter_m: float = .008

    @property
    def mount_poses(self) -> tuple[MountPose, MountPose]:
        return tuple(MountPose((self.base_x, sign*self.arm_separation/2,
                                self.base_height+.002)) for sign in (-1, 1))

    @property
    def source_workspaces_xy(self) -> tuple:
        """The exact configured sampling rectangles for the left/right objects."""
        jitter = self.source_jitter_m
        return tuple(((self.source_x-jitter, self.source_x+jitter),
                      (sign*self.arm_separation/2-jitter,
                       sign*self.arm_separation/2+jitter)) for sign in (-1, 1))


@dataclass(frozen=True)
class PillTaskProfile:
    """A robot-specific mount/grasp recipe for the original pill scene.

    Offsets and clearance are ordered by the task's two selected objects/arms.
    They do not replace the original task's object sampling or success rules.
    """
    mount_poses: tuple[MountPose, MountPose]
    grasp_height_offsets_m: tuple[float, float]
    pregrasp_clearances_m: tuple[float, float]
    # None retains the task's original grasp built from the measured closing axis.
    grasp_quaternions: tuple[tuple[float, float, float, float] | None,
                             tuple[float, float, float, float] | None]


@dataclass(frozen=True)
class AppleBananaTaskProfile(PillTaskProfile):
    """Original basket task mounts and grasps, relative to object geometry centres."""


PICK_PLACE_PROFILES = {
    # The equivalent top-down grasp keeps joint6 away from its +/-pi limits.
    "piper": PickPlaceProfile(-.36, .48, -.04, .06,
        grasp_height_offset=.012, base_height=.16, grasp_quaternion=(0., 0., 1., 0.)),
    "panda": PickPlaceProfile(-.60, .60, -.08, .12),
    "arx_x5": PickPlaceProfile(-.40, .60, -.04, .08,
        grasp_height_offset=.022, base_height=.16, grasp_quaternion=(0., 0., 1., 0.)),
    "ur5_wsg": PickPlaceProfile(-.55, .75, -.05, .10, grasp_height_offset=.028),
    "xarm6_robotiq": PickPlaceProfile(-.55, .70, -.05, .10),
}

PILL_PROFILES = {
    "panda": PillTaskProfile(
        mount_poses=(MountPose((-.60, -.35, 0.)), MountPose((-.60, .35, 0.))),
        grasp_height_offsets_m=(.12, .13), pregrasp_clearances_m=(.09, .08),
        grasp_quaternions=(None, None)),
    "piper": PillTaskProfile(
        mount_poses=(MountPose((-.30, -.30, .202)), MountPose((-.30, .30, .202))),
        grasp_height_offsets_m=(.12, .12), pregrasp_clearances_m=(.09, .09),
        grasp_quaternions=((0., 0., 1., 0.), (0., 0., 1., 0.))),
    "ur5_wsg": PillTaskProfile(
        mount_poses=(MountPose((-.55, -.35, .002)),
                     MountPose((-.55, .35, .002),
                               (.25881904510252074, 0., 0., .9659258262890683))),
        grasp_height_offsets_m=(.12, .12), pregrasp_clearances_m=(.09, .09),
        grasp_quaternions=((0., 0., 1., 0.), (0., 0., 1., 0.))),
}

APPLE_BANANA_PROFILES = {
    "piper": AppleBananaTaskProfile(
        mount_poses=(MountPose((-.25, -.30, .202)), MountPose((-.25, .30, .202))),
        grasp_height_offsets_m=(.012, .012), pregrasp_clearances_m=(.08, .08),
        grasp_quaternions=((0., 0., 1., 0.), (0., 0., 1., 0.))),
}

TASK_PROFILES: dict[tuple[str, str], PickPlaceProfile | PillTaskProfile] = {
    ("DualArmPickPlace-v1", uid): profile for uid, profile in PICK_PLACE_PROFILES.items()
}
TASK_PROFILES.update({("TwoRobotPickPillToRegions-v1", uid): profile
                      for uid, profile in PILL_PROFILES.items()})
TASK_PROFILES.update({("TwoRobotPickAppleBananaToBaskets-v1", uid): profile
                      for uid, profile in APPLE_BANANA_PROFILES.items()})


def get_task_profile(task_id: str, robot_uid: str) -> PickPlaceProfile | PillTaskProfile:
    """Resolve only declared task/robot combinations with an actionable error."""
    uid = get_robot_profile(robot_uid).uid
    try:
        return TASK_PROFILES[task_id, uid]
    except KeyError:
        choices = tuple(robot for task, robot in TASK_PROFILES if task == task_id)
        raise ValueError(f"Task {task_id!r} has no profile for robot {robot_uid!r}; "
                         f"supported robots: {choices}") from None


if __name__ == "__main__":
    for uid, profile in PICK_PLACE_PROFILES.items():
        assert get_task_profile("DualArmPickPlace-v1", uid) is profile
        assert len(profile.mount_poses) == len(profile.source_workspaces_xy) == 2
        assert profile.source_x > profile.base_x
    print("Five task arrangements retain their configured mounts and source workspaces")
