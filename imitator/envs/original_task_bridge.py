"""Task-local mounts and resets for the original TIG scene classes.

Robot definitions remain in imitator.embodiments. Original default Panda
construction returns no adapter, preserving the serial benchmark path.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import sapien
import torch

from ..embodiments.native import resolve_robot, robot_pair
from .original_task_profiles import get_task_profile


@dataclass(frozen=True)
class OriginalTaskAdapter:
    """One lifecycle boundary between an original task and 2.0 embodiments.

    The task keeps ownership of scene objects, rewards, and success predicates;
    this adapter owns only embodiment-specific construction and reset details.
    """

    task_id: str
    robot_uids: tuple[str, str]
    embodiment: str | None
    profile: Any

    @classmethod
    def from_args(cls, task_id: str, robot_uids: tuple[str, str],
                  embodiment: str | None) -> "OriginalTaskAdapter":
        resolved_uids, profile = configure_task(task_id, robot_uids, embodiment)
        return cls(task_id, resolved_uids, embodiment, profile)

    def load_poses(self) -> list[sapien.Pose]:
        return load_poses(self.profile)

    def configure_agent(self, raw: Any) -> None:
        configure_gripper_effort(raw)

    def build_scene_mounts(self, raw: Any) -> None:
        build_mounts(raw)

    def initialize_agents(self, raw: Any, env_idx: torch.Tensor) -> None:
        initialize_agents(raw, env_idx)


def configure_task(env_id: str, robot_uids: tuple[str, str], embodiment: str | None) -> tuple[tuple[str, str], Any]:
    """Resolve an explicit native selection, leaving omitted Panda defaults alone."""
    if embodiment is None and tuple(robot_uids) == ("panda_wristcam",) * 2:
        return tuple(robot_uids), None
    robot = resolve_robot(embodiment or robot_uids[0])
    if len(set(robot_uids)) != 1:
        raise ValueError("The native original-task bridge supports homogeneous pairs")
    profile = get_task_profile(env_id, robot)
    if robot != "panda":
        from ..embodiments import agents  # noqa: F401
    return robot_pair(robot), profile


def load_poses(profile: Any) -> list[sapien.Pose]:
    if profile is None:
        return [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
    return [sapien.Pose(p=m.position, q=m.quaternion) for m in profile.mount_poses]


def configure_gripper_effort(raw: Any) -> None:
    """Apply a declared task effort override to the existing native controllers."""
    profile = raw.embodiment_profile
    effort = None if profile is None else profile.gripper_drive_force_limit
    if effort is None:
        return
    if not np.isfinite(effort) or effort <= 0:
        raise ValueError("Task gripper effort must be finite and positive")
    for agent in raw.agent.agents:
        for controller in agent.controllers.values():
            for name, part in controller.controllers.items():
                if name != "arm":
                    part.config.force_limit = float(effort)
                    part.set_drive_property()


def build_mounts(raw: Any) -> None:
    """Add real pedestals for task-local elevated mounts, before simulation."""
    profile = raw.embodiment_profile
    if profile is None:
        return
    from mani_skill.utils.building import actors
    from ..embodiments.dual import make_group2_bits_disjoint
    make_group2_bits_disjoint(raw.agent.agents)
    for i, mount in enumerate(profile.mount_poses):
        x, y, z = mount.position
        if z > .002:
            actors.build_box(raw.scene, half_sizes=[.065, .065, (z-.002)/2],
                             color=[.3, .3, .33, 1], name=f"mount_{i}", body_type="static",
                             initial_pose=sapien.Pose([x, y, (z-.002)/2]))


def initialize_agents(raw: Any, env_idx: torch.Tensor) -> None:
    """Reset explicit native agents using live 2.0 joint names and task mounts."""
    profile = raw.embodiment_profile
    if profile is None:
        return
    for agent, mount in zip(raw.agent.agents, profile.mount_poses):
        record = getattr(agent, "record", None)
        if record is not None:
            home = record.home_qpos()
            if profile.initial_arm_qpos is not None:
                start = np.asarray(profile.initial_arm_qpos, dtype=float)
                if start.shape != (record.dof_arm,) or not np.isfinite(start).all():
                    raise ValueError("Task arm start must match the existing record's joint names")
                home.update(zip(record.arm_joints, start))
            names = [j.name for j in agent.robot.active_joints]
            q = np.asarray([home[n] for n in names], np.float32)
            agent.reset(torch.as_tensor(q, device=raw.device).repeat(len(env_idx), 1))
        agent.robot.set_pose(sapien.Pose(p=mount.position, q=mount.quaternion))


@lru_cache(maxsize=8)
def legacy_to_native_tcp(robot: str) -> np.ndarray:
    """Fixed T_oldTCP,newTCP for migrating geometric goals, not old actions.

    The recipes were expressed at the previous adapter's TCP. We preserve the
    desired flange pose by applying the actual static frame transform. This
    does not imply that changed controllers or physical assets replay equally.
    """
    robot = resolve_robot(robot)
    if robot in ("panda", "piper"):
        return np.eye(4)
    from ..embodiments.registry import get_arm
    from ..embodiments.urdf_tools import load_urdf
    record = get_arm(robot)
    model = load_urdf(record.urdf_path)
    transforms = model.fk(record.home_qpos())
    if robot == "arx_x5":
        old = transforms["link6"] @ sapien.Pose(p=[.12, 0, 0],
                                                q=[2**-.5, 0, 2**-.5, 0]).to_transformation_matrix()
    elif robot == "ur5_wsg":
        old = transforms["wsg_50_base_link"] @ sapien.Pose(p=[0, 0, .11],
                                                          q=[2**-.5, 0, 0, -2**-.5]).to_transformation_matrix()
    else:
        old = transforms["eef"]
    matrix = np.linalg.inv(old) @ transforms[record.tcp_link]
    assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
    return matrix


def migrate_grasp(raw: Any, pose: sapien.Pose) -> sapien.Pose:
    matrix = legacy_to_native_tcp(raw.robot_uids[0])
    return sapien.Pose(pose.to_transformation_matrix() @ matrix)


if __name__ == "__main__":
    assert np.array_equal(legacy_to_native_tcp("piper"), np.eye(4))
    assert configure_task("TwoRobotPickPillToRegions-v1", ("panda_wristcam",) * 2, None)[1] is None
    print("Original Panda default and native task frame checks passed")
