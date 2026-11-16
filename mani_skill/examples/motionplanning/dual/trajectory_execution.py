"""Plan without stepping, validate a paired path, then advance both arms once/tick.

Initial scope: two Panda pd_joint_pos controllers, radians, stable grippers,
no grasped object. A scene checker supplied by the caller must cover the scene
used for execution. Sampled geometry checks do not prove continuous or dynamic
collision freedom. All telemetry below describes reaching, not task success.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from mani_skill.examples.motionplanning.dual.trajectory_schedule import (
    DualArmSchedule, plan_pose, build_hold_schedule,
)
from mani_skill.examples.motionplanning.dual.combined_collision import validate_schedule


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


def _geometry_guard(checker: Any) -> None:
    for name in ("assert_scene_unchanged", "assert_robot_geometry_unchanged"):
        if hasattr(checker, name):
            getattr(checker, name)()


def plan_pair(left_solver: Any, right_solver: Any, left_pose: Any,
              right_pose: Any) -> DualArmSchedule:
    """Use the native dry-run interface twice from exactly one simulator state."""
    raw = left_solver.base_env
    if raw is not right_solver.base_env:
        raise ValueError("Both planners must share one environment")
    keys = []
    for solver in (left_solver, right_solver):
        matches = [key for key, agent in raw.agent.agents_dict.items()
                   if agent is solver.env_agent]
        if len(matches) != 1:
            raise ValueError("Cannot bind solver to a unique robot action key")
        keys.append(matches[0])
    arms = bindings(raw, tuple(keys))
    before = snapshot(raw, arms)
    trajectories = [plan_pose(solver, pose, action_key=arm.key,
                              controller_joint_names=arm.names)
                    for solver, pose, arm in zip((left_solver, right_solver),
                                                (left_pose, right_pose), arms)]
    if not all(np.array_equal(a, b) for a, b in zip(before, snapshot(raw, arms))):
        raise RuntimeError("Simulator state changed while planning")
    for arm, trajectory in zip(arms, trajectories):
        if not np.allclose(arm.arm_qpos(), trajectory.positions_rad[0], rtol=0, atol=1e-6):
            raise ValueError("Planner trajectory does not start at the measured arm state")
    return build_hold_schedule(*trajectories, control_timestep_s=raw.control_timestep)


def execute_schedule(env: Any, schedule: DualArmSchedule, checker: Any, *,
                     max_joint_step_rad: float = 0.02,
                     tracking_tolerance_rad: float = 0.2,
                     endpoint_tolerance_rad: float = 0.03,
                     finger_tolerance_m: float = 0.002) -> dict[str, Any]:
    """Reject the whole phase before moving if preflight fails; stop on live failure.

    Both robots are passed to one env.step per interval. Early termination,
    truncation, excessive tracking error or a measured collision stops the
    phase. No extra settling ticks are silently appended to its time axis.
    """
    if not all(np.isfinite(x) and x > 0 for x in
               (tracking_tolerance_rad, endpoint_tolerance_rad, finger_tolerance_m)):
        raise ValueError("Tracking, endpoint and finger tolerances must be positive")
    raw = env.unwrapped
    if not np.isclose(raw.control_timestep, schedule.control_timestep_s, rtol=0, atol=1e-10):
        raise ValueError("Environment control clock differs from the scheduled clock")
    arms = bindings(env, tuple(t.action_key for t in schedule.trajectories))
    for arm, trajectory, commands in zip(arms, schedule.trajectories,
                                         (schedule.left_commands, schedule.right_commands)):
        if arm.names != trajectory.joint_names:
            raise ValueError("Scheduled joint order differs from the active controller")
        nodes = np.vstack([trajectory.positions_rad[0], commands[:, :7]])
        if np.any(nodes < arm.limits[:, 0] - 1e-6) or np.any(nodes > arm.limits[:, 1] + 1e-6):
            raise ValueError("Scheduled arm command exceeds joint limits")
        # This first executor permits only a gripper already at its requested
        # open/closed limit. Intermediate commands and grasping need new geometry.
        grip = trajectory.gripper_command
        gripper = arm.agent.controller.controllers["gripper"]
        if not gripper.config.normalize_action or gripper.config.use_delta or abs(grip) != 1:
            raise ValueError("Initial executor supports stable OPEN/CLOSED grippers only")
        finger_indices = [i for i in range(9) if i not in arm.indices]
        finger_limits = as_numpy(arm.agent.robot.get_qlimits())[0, finger_indices]
        expected = finger_limits[:, 1 if grip == 1 else 0]
        if not np.allclose(arm.full_qpos()[finger_indices], expected, rtol=0,
                           atol=finger_tolerance_m):
            raise ValueError("Gripper must be stable at its commanded endpoint before execution")
    initial_full = [arm.full_qpos() for arm in arms]
    _geometry_guard(checker)
    for action in schedule.iter_actions():
        if not raw.agent.single_action_space.contains({k: v.astype(np.float32) for k, v in action.items()}):
            raise ValueError("A scheduled action is outside the actual controller action space")
    validation = validate_schedule(schedule, *(arm.arm_qpos() for arm in arms),
                                   checker, max_joint_step_rad=max_joint_step_rad)
    report: dict[str, Any] = {"status": "rejected", "validation": validation,
                              "environment_steps": 0, "trace": [],
                              "scope": "unloaded reaching with stable grippers"}
    if validation["status"] != "passed":
        return report
    report["status"] = "completed"
    for index, action in enumerate(schedule.iter_actions()):
        try:
            _geometry_guard(checker)
        except RuntimeError as exc:
            report.update(status="aborted", reason="geometry_changed_before_step", error=str(exc))
            break
        _, _, terminated, truncated, _ = env.step({k: v.astype(np.float32) for k, v in action.items()})
        report["environment_steps"] += 1
        measured = [arm.arm_qpos() for arm in arms]
        error = max(float(np.max(np.abs(q - action[arm.key][:7])))
                    for q, arm in zip(measured, arms))
        geometry_error = None
        try:
            _geometry_guard(checker)
        except RuntimeError as exc:
            geometry_error = str(exc)
        collisions = checker.check(*measured) if geometry_error is None else []
        finger_error = max(float(np.max(np.abs(arm.full_qpos()[[i for i in range(9) if i not in arm.indices]] -
                                             full[[i for i in range(9) if i not in arm.indices]])))
                           for arm, full in zip(arms, initial_full))
        report["trace"].append({"step": index + 1, "time_s": float(schedule.target_times_s[index]),
                                 "left_qpos": measured[0].tolist(), "right_qpos": measured[1].tolist(),
                                 "left_target": action[arms[0].key].tolist(),
                                 "right_target": action[arms[1].key].tolist(),
                                 "tracking_error_rad": error, "finger_change_m": finger_error,
                                 "collisions": collisions, "geometry_error": geometry_error})
        if geometry_error is not None:
            report.update(status="aborted", reason="geometry_changed_after_step", error=geometry_error)
            break
        if collisions or error > tracking_tolerance_rad or finger_error > finger_tolerance_m:
            report["status"] = "aborted"
            report["reason"] = "collision_or_tracking_or_finger_change"
            break
        if np.any(as_numpy(terminated)) or np.any(as_numpy(truncated)):
            report["status"] = "terminated" if np.any(as_numpy(terminated)) else "truncated"
            break
    final_error = max(float(np.max(np.abs(arm.arm_qpos() - trajectory.positions_rad[-1])))
                      for arm, trajectory in zip(arms, schedule.trajectories))
    report["endpoint_error_rad"] = final_error
    if report["status"] == "completed" and final_error > endpoint_tolerance_rad:
        report.update(status="tracking_failed", reason="endpoint_tolerance")
    return report


def self_test() -> dict[str, Any]:
    """Exercise actual tick submission and fail-before-motion with a synthetic env."""
    from types import SimpleNamespace as NS
    from mani_skill.examples.motionplanning.dual.trajectory_schedule import ArmTrajectory
    names = tuple(f"joint{i}" for i in range(7))

    class Robot:
        def __init__(self) -> None:
            self.q = np.array([[0.] * 7 + [.04, .04]])
            self.pose = NS(raw_pose=np.array([[0, 0, 0, 1, 0, 0, 0]]))
        def get_qpos(self) -> np.ndarray:
            return self.q
        def get_qlimits(self) -> np.ndarray:
            return np.array([[[-2, 2]] * 7 + [[0, .04], [0, .04]]])
        def get_active_joints(self) -> list[Any]:
            return [NS(get_name=lambda name=name: name) for name in (*names, "f1", "f2")]

    class Env:
        def __init__(self) -> None:
            from gymnasium import spaces
            self.unwrapped = self
            self.num_envs, self.control_timestep, self.elapsed_steps = 1, .1, np.array([0])
            self.agent = NS(agents_dict={})
            for key in ("left", "right"):
                robot = Robot()
                controllers = {"arm": NS(config=NS(normalize_action=False, use_delta=False),
                                          joints=robot.get_active_joints()[:7]),
                               "gripper": NS(config=NS(normalize_action=True, use_delta=False))}
                self.agent.agents_dict[key] = NS(robot=robot, control_mode="pd_joint_pos",
                                                 controller=NS(controllers=controllers))
            self.agent.single_action_space = spaces.Dict({key: spaces.Box(-2., 2., (8,), dtype=np.float32)
                                                           for key in ("left", "right")})
            self.actions: list[dict[str, np.ndarray]] = []
        def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
            assert set(action) == {"left", "right"}
            self.actions.append(action)
            for key, value in action.items(): self.agent.agents_dict[key].robot.q[0, :7] = value[:7]
            self.elapsed_steps += 1
            return None, 0, False, len(self.actions) >= getattr(self, "truncate_at", 999), {}

    left = ArmTrajectory("left", names, [0, .2], np.array([[0.] * 7, [.1] * 7]), 1)
    right = ArmTrajectory("right", names, [0, .4], np.array([[0.] * 7, [.2] * 7]), 1)
    schedule = build_hold_schedule(left, right, control_timestep_s=.1)
    clear = NS(check=lambda left, right: [])
    env = Env()
    result = execute_schedule(env, schedule, clear)
    assert result["status"] == "completed" and len(env.actions) == schedule.num_steps == 4
    assert np.allclose(env.actions[2]["left"][:7], env.actions[3]["left"][:7])
    env = Env()
    rejected = execute_schedule(env, schedule, NS(check=lambda left, right: [{"pair": "fixture"}]))
    assert rejected["status"] == "rejected" and not env.actions
    env = Env(); env.truncate_at = 1
    assert execute_schedule(env, schedule, clear)["status"] == "truncated" and len(env.actions) == 1
    env = Env(); env.control_timestep = .2
    try: execute_schedule(env, schedule, clear)
    except ValueError: pass
    else: raise AssertionError("Wrong execution clock accepted")
    assert not env.actions
    env = Env()
    def scene_guard() -> None:
        if env.actions: raise RuntimeError("fixture moved on the last tick")
    assert execute_schedule(env, schedule, NS(check=clear.check, assert_scene_unchanged=scene_guard))["status"] == "aborted"
    assert len(env.actions) == 1
    env = Env()
    from gymnasium import spaces
    env.agent.single_action_space = spaces.Dict({key: spaces.Box(np.array([-2.] * 7 + [-1.], dtype=np.float32),
                                                                 np.array([.15] * 7 + [1.], dtype=np.float32))
                                                  for key in ("left", "right")})
    try: execute_schedule(env, schedule, clear)
    except ValueError: pass
    else: raise AssertionError("Invalid later action accepted")
    assert not env.actions
    return {"status": "passed", "checks": ["same-tick paired submission", "endpoint hold",
            "collision rejects before stepping", "truncation stops immediately", "clock mismatch rejects",
            "post-step geometry changes abort", "whole action-space preflight"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))
