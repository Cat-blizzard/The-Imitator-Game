"""Bind measured robot states to explicit controller joint/action order."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class ArmBinding:
    key: str
    agent: Any
    names: tuple[str, ...]
    indices: np.ndarray
    limits: np.ndarray
    all_names: tuple[str, ...]
    finger_indices: np.ndarray

    def full_qpos(self) -> np.ndarray:
        value = as_numpy(self.agent.robot.get_qpos())
        if value.shape != (1, len(self.all_names)) or not np.isfinite(value).all():
            raise ValueError("Expected one finite state matching the robot's active joint order")
        return value[0].copy()

    def arm_qpos(self) -> np.ndarray:
        return self.full_qpos()[self.indices]


def bind_arm(key: str, agent: Any) -> ArmBinding:
    if agent.control_mode != "pd_joint_pos":
        raise ValueError("Absolute pd_joint_pos control is required")
    arm = agent.controller.controllers["arm"]
    if arm.config.normalize_action or arm.config.use_delta:
        raise ValueError("Arm commands must be absolute, unnormalized radians")
    names = tuple(joint.get_name() for joint in arm.joints)
    all_names = tuple(joint.get_name() for joint in agent.robot.get_active_joints())
    if (not names or len(set(names)) != len(names)
            or len(set(all_names)) != len(all_names) or not set(names).issubset(all_names)):
        raise ValueError("Arm controller joints must uniquely name actual active joints")
    # Agents name the physical gripper joints; the complement preserves the
    # existing two-finger fixture/agent contract when no explicit list exists.
    finger_names = tuple(getattr(agent, "gripper_joint_names",
                                 tuple(name for name in all_names if name not in names)))
    if (not finger_names or len(set(finger_names)) != len(finger_names)
            or set(names) & set(finger_names)
            or set(names) | set(finger_names) != set(all_names)):
        raise ValueError("This binding requires one arm and its named physical gripper joints")
    indices = np.array([all_names.index(name) for name in names])
    fingers = np.array([all_names.index(name) for name in finger_names])
    limits = as_numpy(agent.robot.get_qlimits())[0, indices].copy()
    if limits.shape != (len(names), 2) or not np.isfinite(limits).all():
        raise ValueError("Invalid arm joint limits")
    return ArmBinding(key, agent, names, indices, limits, all_names, fingers)


def bindings(env: Any, keys: tuple[str, str]) -> tuple[ArmBinding, ArmBinding]:
    raw = env.unwrapped
    if raw.num_envs != 1 or set(raw.agent.agents_dict) != set(keys):
        raise ValueError("Schedule keys must cover exactly both robots in a single environment")
    return tuple(bind_arm(key, raw.agent.agents_dict[key]) for key in keys)


def snapshot(env: Any, arms: tuple[ArmBinding, ArmBinding]) -> list[np.ndarray]:
    result = [as_numpy(env.unwrapped.elapsed_steps).copy()]
    for arm in arms:
        result.extend([arm.full_qpos(), as_numpy(arm.agent.robot.pose.raw_pose).copy()])
    return result
