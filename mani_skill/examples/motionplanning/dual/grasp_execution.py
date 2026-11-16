"""Bounded reaching and contact-confirmed grasping; no lifting or attachments.

Actions contain two Panda pd_joint_pos commands in one env.step. Finger qpos
are measured geometry, never inferred from the normalized gripper command.
Aborting stops advancing this phase; it does not brake, retreat, release, or
authorize continuation. A supplied scene checker owns contact permissions and
must refresh actual robot-finger and object geometry without stepping physics.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from typing import Any

import numpy as np

from mani_skill.examples.motionplanning.dual.combined_collision import validate_schedule
from mani_skill.examples.motionplanning.dual.trajectory_execution import (
    ArmBinding, as_numpy, bindings, snapshot,
)
from mani_skill.examples.motionplanning.dual.trajectory_schedule import (
    ArmTrajectory, DualArmSchedule, build_hold_schedule, plan_pose,
)


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _budget(value: Any, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _checker(checker: Any) -> None:
    if not all(callable(getattr(checker, name, None)) for name in
               ("refresh", "check", "set_target_contact_allowed")):
        raise ValueError("Checker must implement refresh/check/set_target_contact_allowed")


def _measure(arms: tuple[ArmBinding, ArmBinding]) -> dict[str, Any]:
    result = {}
    for arm in arms:
        full = arm.full_qpos()
        velocity = as_numpy(arm.agent.robot.get_qvel())
        if velocity.shape != (1, 9) or not np.isfinite(velocity).all():
            raise ValueError("Expected one finite nine-joint Panda velocity state")
        fingers = [i for i in range(9) if i not in arm.indices]
        result[arm.key] = {"qpos_rad": full[arm.indices].tolist(),
                           "qvel_rad_s": velocity[0, arm.indices].tolist(),
                           "finger_qpos_m": full[fingers].tolist(),
                           "finger_qvel_m_s": velocity[0, fingers].tolist()}
    return result


def _actions_valid(env: Any, arms: tuple[ArmBinding, ArmBinding], action: dict[str, Any]) -> None:
    if set(action) != {arm.key for arm in arms}:
        raise ValueError("Every control tick must contain exactly both robot actions")
    converted = {}
    for arm in arms:
        value = np.asarray(action[arm.key], dtype=np.float64)
        if value.shape != (8,) or not np.isfinite(value).all() or abs(value[7]) > 1:
            raise ValueError("Each action must contain seven finite radians and one normalized gripper command")
        if np.any(value[:7] < arm.limits[:, 0] - 1e-6) or np.any(value[:7] > arm.limits[:, 1] + 1e-6):
            raise ValueError("Arm target exceeds joint limits")
        converted[arm.key] = value.astype(np.float32)
    if not env.unwrapped.agent.single_action_space.contains(converted):
        raise ValueError("Action exceeds the actual controller action space")


def _report(env: Any, checker: Any, phase: str, initial: dict[str, Any]) -> dict[str, Any]:
    return {"status": "running", "phase": phase, "environment_steps": 0,
            "settling_steps": 0, "trace": [], "initial_measurements": initial,
            "control_timestep_s": float(env.unwrapped.control_timestep),
            "initial_elapsed_steps": as_numpy(env.unwrapped.elapsed_steps).tolist(),
            "collision_scope": str(checker.scope), "continuous_collision_detection": False,
            "abort_semantics": "stop advancing this phase; no braking, retreat, release or continuation"}


def _error(report: dict[str, Any], reason: str, exc: Exception) -> None:
    report.update(status="aborted", reason=reason, error_type=type(exc).__name__,
                  error=str(exc), traceback=traceback.format_exc())


def _preflight(checker: Any, initial: dict[str, Any], arms: tuple[ArmBinding, ArmBinding],
               report: dict[str, Any], allow_contact: bool) -> bool:
    try:
        checker.set_target_contact_allowed(allow_contact)
        checker.refresh()
        collisions = checker.check(*(np.asarray(initial[a.key]["qpos_rad"]) for a in arms))
    except Exception as exc:
        _error(report, "checker_preflight_failed", exc)
        report["status"] = "rejected"
        return False
    report["initial_collisions"] = collisions
    if collisions:
        report.update(status="rejected", reason="initial_collision")
        return False
    return True


def _advance(env: Any, arms: tuple[ArmBinding, ArmBinding], checker: Any,
             action: dict[str, Any], report: dict[str, Any], phase: str) -> dict[str, Any] | None:
    """Submit both arms once and record actual measurements even on a rejection."""
    actual = {k: np.asarray(v, dtype=np.float32).copy() for k, v in action.items()}
    row: dict[str, Any] = {"phase": phase, "action": {k: v.tolist() for k, v in actual.items()},
                           "step": report["environment_steps"] + 1, "is_grasping": None}
    before = as_numpy(env.unwrapped.elapsed_steps).copy()
    try:
        _, _, terminated, truncated, _ = env.step(actual)
    except Exception as exc:
        # A wrapper may fail after physics has already advanced. Preserve the
        # observed counter and attempted action, rather than claiming zero motion.
        report["observed_elapsed_steps_after_error"] = as_numpy(env.unwrapped.elapsed_steps).tolist()
        row["elapsed_step_delta_on_error"] = (as_numpy(env.unwrapped.elapsed_steps) - before).tolist()
        row["error"] = f"{type(exc).__name__}: {exc}"
        report["trace"].append(row)
        _error(report, "environment_step_failed", exc)
        return None
    report["environment_steps"] += 1
    report["settling_steps"] += int(phase == "settling")
    row["time_s"] = report["environment_steps"] * report["control_timestep_s"]
    try:
        row["measurements"] = _measure(arms)
        checker.refresh()
        row["collisions"] = checker.check(*(np.asarray(row["measurements"][a.key]["qpos_rad"]) for a in arms))
        row["terminated"] = bool(np.any(as_numpy(terminated)))
        row["truncated"] = bool(np.any(as_numpy(truncated)))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        report["trace"].append(row)
        _error(report, "measurement_or_collision_check_failed", exc)
        return None
    report["trace"].append(row)
    if row["collisions"]:
        report.update(status="aborted", reason="collision")
    elif row["terminated"] or row["truncated"]:
        report.update(status="terminated" if row["terminated"] else "truncated", reason="environment_done")
    return row


def plan_left_reach(left_solver: Any, right_solver: Any, pose: Any) -> DualArmSchedule:
    """Plan only the left TCP; the right arm holds its measured joint target."""
    raw = left_solver.base_env
    if raw is not right_solver.base_env:
        raise ValueError("Both solvers must share one environment")
    keys = []
    for solver in (left_solver, right_solver):
        matches = [key for key, agent in raw.agent.agents_dict.items() if agent is solver.env_agent]
        if len(matches) != 1:
            raise ValueError("Cannot bind a solver to a unique robot")
        keys.append(matches[0])
    arms = bindings(raw, tuple(keys))
    before = snapshot(raw, arms)
    left = plan_pose(left_solver, pose, action_key=arms[0].key, controller_joint_names=arms[0].names)
    right = ArmTrajectory(arms[1].key, arms[1].names, [0.0], [arms[1].arm_qpos()],
                          right_solver.gripper_state)
    if not all(np.array_equal(a, b) for a, b in zip(before, snapshot(raw, arms))):
        raise RuntimeError("Environment changed during dry-run planning")
    if not np.allclose(left.positions_rad[0], arms[0].arm_qpos(), rtol=0, atol=1e-6):
        raise ValueError("Left planner start differs from the measured state")
    return build_hold_schedule(left, right, control_timestep_s=raw.control_timestep)


def execute_reaching(env: Any, schedule: DualArmSchedule, checker: Any, *,
                     max_joint_step_rad: float = 0.02, tracking_tolerance_rad: float = 0.2,
                     endpoint_tolerance_rad: float = 0.03, velocity_tolerance_rad_s: float = 0.05,
                     settle_max_steps: int = 10, stable_steps: int = 3,
                     right_hold_tolerance_rad: float = 0.03,
                     allow_target_contact: bool = False) -> dict[str, Any]:
    """Run an open-hand left reach, then at most settle_max_steps recorded holds.

    Passing requires the final joint targets AND measured arm velocities to be
    stable for stable_steps consecutive observations. All settling is explicit.
    """
    tracking = _positive(tracking_tolerance_rad, "tracking_tolerance_rad")
    endpoint = _positive(endpoint_tolerance_rad, "endpoint_tolerance_rad")
    speed = _positive(velocity_tolerance_rad_s, "velocity_tolerance_rad_s")
    hold_tol = _positive(right_hold_tolerance_rad, "right_hold_tolerance_rad")
    resolution = _positive(max_joint_step_rad, "max_joint_step_rad")
    settling = _budget(settle_max_steps, "settle_max_steps", zero=True)
    stable_required = _budget(stable_steps, "stable_steps")
    if not isinstance(allow_target_contact, bool):
        raise ValueError("allow_target_contact must be explicit bool")
    _checker(checker)
    arms = bindings(env, tuple(t.action_key for t in schedule.trajectories))
    dt = _positive(env.unwrapped.control_timestep, "control_timestep")
    if not np.isclose(dt, schedule.control_timestep_s, rtol=0, atol=1e-10):
        raise ValueError("Environment and schedule control clocks differ")
    initial = _measure(arms)
    right_start = np.asarray(initial[arms[1].key]["qpos_rad"])
    for arm, trajectory in zip(arms, schedule.trajectories):
        if trajectory.joint_names != arm.names or trajectory.gripper_command != 1:
            raise ValueError("Reaching requires controller joint order and both grippers OPEN")
        finger_indices = [i for i in range(9) if i not in arm.indices]
        upper = as_numpy(arm.agent.robot.get_qlimits())[0, finger_indices, 1]
        if not np.allclose(initial[arm.key]["finger_qpos_m"], upper, rtol=0, atol=0.002):
            raise ValueError("Both fingers must already be open before reaching")
    if not np.allclose(schedule.trajectories[1].positions_rad, right_start, rtol=0, atol=1e-6):
        raise ValueError("This reach supports a stationary right arm only")
    actions = list(schedule.iter_actions())
    final = {arm.key: np.r_[trajectory.positions_rad[-1], 1.0]
             for arm, trajectory in zip(arms, schedule.trajectories)}
    for action in [*actions, final]:
        _actions_valid(env, arms, action)
    report = _report(env, checker, "reaching", initial)
    report.update(planned_control_steps=len(actions), settle_max_steps=settling,
                  stable_steps_required=stable_required, allow_target_contact=allow_target_contact)
    started = time.monotonic()
    if not _preflight(checker, initial, arms, report, allow_target_contact):
        return report
    report["validation"] = validate_schedule(schedule, *(np.asarray(initial[a.key]["qpos_rad"]) for a in arms),
                                             checker, max_joint_step_rad=resolution)
    if report["validation"]["status"] != "passed":
        report.update(status="rejected", reason="path_collision")
        return report
    streak = 0
    for index in range(len(actions) + settling):
        phase = "reaching" if index < len(actions) else "settling"
        action = actions[index] if phase == "reaching" else final
        row = _advance(env, arms, checker, action, report, phase)
        if row is None or report["status"] != "running":
            break
        measured = row["measurements"]
        tracking_error = max(float(np.max(np.abs(np.asarray(measured[a.key]["qpos_rad"]) - action[a.key][:7]))) for a in arms)
        right_error = float(np.max(np.abs(np.asarray(measured[arms[1].key]["qpos_rad"]) - right_start)))
        endpoint_error = max(float(np.max(np.abs(np.asarray(measured[a.key]["qpos_rad"]) - final[a.key][:7]))) for a in arms)
        velocity = max(float(np.max(np.abs(measured[a.key]["qvel_rad_s"]))) for a in arms)
        row.update(tracking_error_rad=tracking_error, right_hold_error_rad=right_error,
                   endpoint_error_rad=endpoint_error, max_arm_velocity_rad_s=velocity)
        if tracking_error > tracking or right_error > hold_tol:
            report.update(status="aborted", reason="tracking_or_right_hold_error")
            break
        streak = streak + 1 if endpoint_error <= endpoint and velocity <= speed else 0
        row["stable_steps"] = streak
        if index + 1 >= len(actions) and streak >= stable_required:
            report["status"] = "passed"
            break
    if report["status"] == "running":
        report.update(status="reach_timeout", reason="endpoint_or_velocity_not_stable")
    report.update(stable_steps_observed=streak, wall_seconds=time.monotonic() - started)
    return report


def close_and_confirm(env: Any, checker: Any, *, active_key: str, target_actor: Any,
                      max_steps: int = 40, stable_steps: int = 5,
                      arm_hold_tolerance_rad: float = 0.03,
                      finger_stability_tolerance_m: float = 0.0002) -> dict[str, Any]:
    """Hold both entry arm positions; close one hand and confirm real contact.

    A successful grasp can have nonzero finger qpos. Require consecutive true
    agent.is_grasping(target_actor) AND small measured finger-position changes.
    This does not establish that an object can be lifted or carried.
    """
    budget = _budget(max_steps, "max_steps")
    required = _budget(stable_steps, "stable_steps")
    hold_tol = _positive(arm_hold_tolerance_rad, "arm_hold_tolerance_rad")
    finger_tol = _positive(finger_stability_tolerance_m, "finger_stability_tolerance_m")
    if budget < required:
        raise ValueError("max_steps cannot be smaller than stable_steps")
    _checker(checker)
    raw = env.unwrapped
    _positive(raw.control_timestep, "control_timestep")
    keys = tuple(raw.agent.agents_dict)
    arms = bindings(env, keys)
    if active_key not in keys or target_actor is None:
        raise ValueError("An actual active robot key and target actor are required")
    active = next(arm for arm in arms if arm.key == active_key)
    if not callable(getattr(active.agent, "is_grasping", None)):
        raise ValueError("Active agent must expose the physical contact grasp predicate")
    initial = _measure(arms)
    hold = {a.key: np.asarray(initial[a.key]["qpos_rad"]) for a in arms}
    action = {a.key: np.r_[hold[a.key], -1.0 if a.key == active_key else 1.0] for a in arms}
    _actions_valid(env, arms, action)
    passive = next(a for a in arms if a.key != active_key)
    fingers = [i for i in range(9) if i not in passive.indices]
    if not np.allclose(initial[passive.key]["finger_qpos_m"],
                       as_numpy(passive.agent.robot.get_qlimits())[0, fingers, 1], rtol=0, atol=0.002):
        raise ValueError("The passive hand must already be open")
    report = _report(env, checker, "closing", initial)
    report.update(active_key=active_key, max_steps=budget, stable_steps_required=required,
                  lifted_or_attached=False)
    started = time.monotonic()
    if not _preflight(checker, initial, arms, report, True):
        return report
    previous = np.asarray(initial[active_key]["finger_qpos_m"])
    streak = 0
    for _ in range(budget):
        row = _advance(env, arms, checker, action, report, "closing")
        if row is None or report["status"] != "running":
            break
        measured = row["measurements"]
        drift = max(float(np.max(np.abs(np.asarray(measured[a.key]["qpos_rad"]) - hold[a.key]))) for a in arms)
        current = np.asarray(measured[active_key]["finger_qpos_m"])
        change = float(np.max(np.abs(current - previous)))
        previous = current
        try:
            predicate = as_numpy(active.agent.is_grasping(target_actor))
            if predicate.size != 1 or not np.isfinite(predicate).all() or not np.isin(predicate, [False, True]).all():
                raise ValueError("Expected a single finite contact-based grasp predicate")
            grasped = bool(predicate.reshape(-1)[0])
        except Exception as exc:
            _error(report, "grasp_measurement_failed", exc)
            row["error"] = f"{type(exc).__name__}: {exc}"
            break
        streak = streak + 1 if grasped and change <= finger_tol else 0
        row.update(is_grasping=grasped, arm_hold_error_rad=drift,
                   finger_change_m=change, stable_steps=streak)
        if drift > hold_tol:
            report.update(status="aborted", reason="arm_hold_error")
            break
        if streak >= required:
            report["status"] = "passed"
            break
    if report["status"] == "running":
        report.update(status="grasp_timeout", reason="contact_grasp_not_stable")
    report.update(stable_steps_observed=streak, wall_seconds=time.monotonic() - started)
    return report


def self_test() -> dict[str, Any]:
    """Synthetic CPU-only control and grasp tests; no SAPIEN scene or GPU."""
    from types import SimpleNamespace as NS
    from gymnasium import spaces

    names = tuple(f"joint{i}" for i in range(7))

    class Robot:
        def __init__(self) -> None:
            self.q = np.array([[0.0] * 7 + [0.04, 0.04]])
            self.v = np.zeros((1, 9))
        def get_qpos(self) -> np.ndarray: return self.q
        def get_qvel(self) -> np.ndarray: return self.v
        def get_qlimits(self) -> np.ndarray: return np.array([[[-2, 2]] * 7 + [[0, .04]] * 2])
        def get_active_joints(self) -> list[Any]:
            return [NS(get_name=lambda n=n: n) for n in (*names, "f1", "f2")]

    class Env:
        def __init__(self, grasp: bool = True, lag: bool = False) -> None:
            self.unwrapped = self
            self.num_envs, self.control_timestep, self.elapsed_steps = 1, .1, np.array([0])
            self.grasp, self.lag, self.actions = grasp, lag, []
            self.agent = NS(agents_dict={})
            for key in ("left", "right"):
                robot = Robot()
                arm = NS(joints=robot.get_active_joints()[:7], config=NS(normalize_action=False, use_delta=False))
                entry = NS(robot=robot, control_mode="pd_joint_pos", controller=NS(controllers={"arm": arm}))
                entry.is_grasping = lambda target, robot=robot: np.array([self.grasp and robot.q[0, 7] < .02])
                self.agent.agents_dict[key] = entry
            self.agent.single_action_space = spaces.Dict({key: spaces.Box(-2., 2., (8,), dtype=np.float32) for key in ("left", "right")})
        def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
            assert set(action) == {"left", "right"}
            self.actions.append({key: value.copy() for key, value in action.items()})
            for key, value in action.items():
                robot = self.agent.agents_dict[key].robot
                before = robot.q.copy()
                robot.q[0, :7] += (value[:7] - robot.q[0, :7]) * (.5 if self.lag else 1.)
                if value[7] == -1:
                    robot.q[0, 7:] = .012 if self.grasp else 0.
                robot.v = (robot.q - before) / self.control_timestep
            self.elapsed_steps += 1
            return None, 0, False, len(self.actions) >= getattr(self, "truncate_at", 999), {}

    class Checker:
        scope = "analytical_fixture"
        def __init__(self, env: Env, collide_at: int = 999) -> None:
            self.env, self.collide_at, self.allowed = env, collide_at, False
        def set_target_contact_allowed(self, allowed: bool) -> None: self.allowed = allowed
        def refresh(self) -> None: pass
        def check(self, left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
            return [{"collision_type": "fixture"}] if len(self.env.actions) >= self.collide_at else []

    def schedule() -> DualArmSchedule:
        left = ArmTrajectory("left", names, [0, .2], [[0.] * 7, [.1] * 7], 1)
        right = ArmTrajectory("right", names, [0], [[0.] * 7], 1)
        return build_hold_schedule(left, right, control_timestep_s=.1)

    env = Env(lag=True)
    reached = execute_reaching(env, schedule(), Checker(env), settle_max_steps=10, stable_steps=2)
    assert reached["status"] == "passed" and 0 < reached["settling_steps"] <= 10
    assert len(env.actions) == reached["environment_steps"] and all(np.array_equal(a["right"][:7], np.zeros(7)) for a in env.actions)
    env = Env(lag=True)
    timeout = execute_reaching(env, schedule(), Checker(env), settle_max_steps=0)
    assert timeout["status"] == "reach_timeout" and len(env.actions) == 2
    env = Env()
    grasped = close_and_confirm(env, Checker(env), active_key="left", target_actor=object(), max_steps=10)
    assert grasped["status"] == "passed" and grasped["stable_steps_observed"] >= 5
    assert env.agent.agents_dict["left"].robot.q[0, 7] == .012
    assert all(a["left"][7] == -1 and a["right"][7] == 1 for a in env.actions)
    env = Env(grasp=False)
    empty = close_and_confirm(env, Checker(env), active_key="left", target_actor=object(), max_steps=7)
    assert empty["status"] == "grasp_timeout" and len(env.actions) == 7
    env = Env()
    collision = close_and_confirm(env, Checker(env, 1), active_key="left", target_actor=object())
    assert collision["status"] == "aborted" and len(env.actions) == 1
    env = Env(); env.truncate_at = 1
    truncated = close_and_confirm(env, Checker(env), active_key="left", target_actor=object())
    assert truncated["status"] == "truncated" and len(env.actions) == 1
    env = Env()
    rejected = execute_reaching(env, schedule(), Checker(env, 0))
    assert rejected["status"] == "rejected" and not env.actions
    for kwargs in ({"max_steps": 0}, {"stable_steps": 0}, {"arm_hold_tolerance_rad": float("nan")}, {"active_key": "missing"}):
        env = Env()
        arguments = dict(active_key="left", target_actor=object(), **{})
        arguments.update(kwargs)
        try: close_and_confirm(env, Checker(env), **arguments)
        except ValueError: pass
        else: raise AssertionError("Invalid grasp input accepted")
        assert not env.actions
    return {"status": "passed", "environment": "synthetic CPU only", "gpu_used": False,
            "checks": ["same-step two-arm actions", "stationary right arm", "explicit bounded settling",
                       "reach timeout", "nonzero held finger positions", "five consecutive contact grasps",
                       "empty grasp timeout", "collision stop", "truncation stop", "preflight zero-step rejection",
                       "invalid input zero-step rejection"],
            "reaching_steps": reached["environment_steps"], "settling_steps": reached["settling_steps"],
            "grasp_confirmation_steps": grasped["environment_steps"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
