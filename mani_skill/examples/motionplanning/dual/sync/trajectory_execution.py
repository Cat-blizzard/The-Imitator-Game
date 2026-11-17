"""Bind measured Panda states to explicit controller joint/action order."""
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

    def full_qpos(self) -> np.ndarray:
        value = as_numpy(self.agent.robot.get_qpos())
        if value.shape != (1, 9) or not np.isfinite(value).all():
            raise ValueError("This executor requires one finite nine-joint Panda state")
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
    if len(names) != 7 or len(all_names) != 9 or len(set(names)) != 7:
        raise ValueError("Expected seven arm joints and two physical finger joints")
    indices = np.array([all_names.index(name) for name in names])
    limits = as_numpy(agent.robot.get_qlimits())[0, indices].copy()
    if limits.shape != (7, 2) or not np.isfinite(limits).all():
        raise ValueError("Invalid arm joint limits")
    return ArmBinding(key, agent, names, indices, limits)


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
