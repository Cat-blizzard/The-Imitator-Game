"""Bind measured robot states to explicit controller joint/action order."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _same_single_env_state(current: Any, recorded: Any) -> bool:
    if isinstance(recorded, dict):
        return (isinstance(current, dict) and current.keys() == recorded.keys()
                and all(_same_single_env_state(current[key], value)
                        for key, value in recorded.items()))
    actual, expected = as_numpy(current), as_numpy(recorded)
    # RecordEpisode removes the single-environment axis when saving H5 rows.
    if actual.shape == (1,) + expected.shape:
        actual = actual[0]
    elif expected.shape == (1,) + actual.shape:
        expected = expected[0]
    return np.array_equal(actual, expected)


def initialize_replay_state(raw: Any, initial_state: dict[str, Any]) -> dict[str, Any]:
    """Use an exact seeded reset, restoring S0 only when its values differ.

    State leaves are the recorded actor/articulation arrays for one environment.
    A read-then-write of an unchanged fixed articulation root is not numerically
    idempotent in the native simulator. Preserve the reset when every recorded
    value already matches, with no tolerance or physics step. Archives whose
    reset differs still use the original one-time recorded-state restoration.
    """
    if _same_single_env_state(raw.get_state_dict(), initial_state):
        return {"method": "seeded_reset", "state_write_count": 0}
    raw.set_state_dict(initial_state)
    return {"method": "recorded_state", "state_write_count": 1}


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


def native_arm_action(arm: ArmBinding, command: Any) -> np.ndarray:
    """Map canonical open/close to this robot's native scalar gripper action.

    Planning schedules use +1 open and -1 closed. The environment and recorder
    receive native controller values, so saved actions replay without remapping.
    """
    result = np.asarray(command, dtype=np.float32).copy()
    if (result.shape != (len(arm.names) + 1,) or not np.isfinite(result).all()
            or abs(float(result[-1])) > 1):
        raise ValueError("Expected named arm radians and one canonical gripper command")
    record = getattr(arm.agent, "record", None)
    opened = float(record.gripper_cmd_open[0] if record else getattr(arm.agent, "gripper_open_action", 1.))
    closed = float(record.gripper_cmd_closed[0] if record else getattr(arm.agent, "gripper_closed_action", -1.))
    if not np.isfinite([opened, closed]).all() or opened == closed:
        raise ValueError("Native gripper endpoints must be finite and different")
    fraction = (float(result[-1]) + 1.) / 2.
    result[-1] = closed + fraction * (opened - closed)
    return result


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
