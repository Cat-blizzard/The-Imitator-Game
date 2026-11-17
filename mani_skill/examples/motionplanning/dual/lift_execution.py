"""Lift a contact-grasped apple and verify a bounded hover, without welding.

The caller supplies an attachment collision checker. Candidate checks predict
the object's pose; measured_check uses actual physics poses. No object pose,
constraint, finger state or environment state is ever assigned here. An abort
stops advancing this phase; it does not brake, retreat, or open either hand.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np

from mani_skill.examples.motionplanning.dual.combined_collision import validate_schedule
from mani_skill.examples.motionplanning.dual.grasp_execution import (
    _actions_valid, _budget, _error, _measure, _positive, plan_left_reach,
)
from mani_skill.examples.motionplanning.dual.trajectory_execution import as_numpy, bindings
from mani_skill.examples.motionplanning.dual.trajectory_schedule import DualArmSchedule


def _transform(value: Any) -> np.ndarray:
    matrix = np.array(as_numpy(value), dtype=np.float64, copy=True)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-5)):
        raise ValueError("Expected a single finite rigid 4x4 transform")
    return matrix


def _velocity(actor: Any, name: str) -> np.ndarray:
    value = np.array(as_numpy(getattr(actor, f"get_{name}_velocity")()), dtype=np.float64, copy=True)
    if value.shape == (1, 3):
        value = value[0]
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"Expected finite measured object {name} velocity")
    return value


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """SO(3) geodesic; project float32 pose rounding before taking the angle."""
    u, _, vh = np.linalg.svd(first.T @ second)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vh)
    relative = u @ correction @ vh
    sine = np.linalg.norm([relative[2, 1] - relative[1, 2],
                           relative[0, 2] - relative[2, 0],
                           relative[1, 0] - relative[0, 1]]) / 2
    cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    return float(np.arctan2(sine, cosine))


def _observe(arms: Any, actor: Any, reference: np.ndarray, initial_z: float, *,
             measurements: dict[str, Any] | None = None) -> dict[str, Any]:
    """Measure one object, optionally sharing joints from this same observation.

    Single-arm callers retain their own joint read. A caller measuring both
    targets may supply the joints it just read; it must not reuse them after a
    physics step. The supplied mapping is returned without modification.
    """
    agent = arms[0].agent
    tcp = _transform(agent.tcp.pose.to_transformation_matrix())
    obj = _transform(actor.pose.to_transformation_matrix())
    relative = np.eye(4)
    relative[:3, :3] = tcp[:3, :3].T @ obj[:3, :3]
    relative[:3, 3] = tcp[:3, :3].T @ (obj[:3, 3] - tcp[:3, 3])
    angle_cos = (np.trace(reference[:3, :3].T @ relative[:3, :3]) - 1) / 2
    predicate = as_numpy(agent.is_grasping(actor))
    if predicate.size != 1 or not np.isfinite(predicate).all() or not np.isin(predicate, [False, True]).all():
        raise ValueError("Expected one physical contact-based grasp predicate")
    forces = None
    if all(hasattr(agent, name) for name in ("scene", "finger1_link", "finger2_link")):
        forces = {}
        for name, link in (("finger1", agent.finger1_link), ("finger2", agent.finger2_link)):
            force = np.array(as_numpy(agent.scene.get_pairwise_contact_forces(link, actor)), dtype=np.float64)
            if force.shape != (1, 3) or not np.isfinite(force).all():
                raise ValueError("Expected a single finite finger-object contact force")
            forces[name] = force[0].tolist()
    linear, angular = _velocity(actor, "linear"), _velocity(actor, "angular")
    return {"measurements": _measure(arms) if measurements is None else measurements,
            "object_T_world": obj.tolist(),
            "tcp_T_world": tcp.tolist(), "T_tcp_object": relative.tolist(),
            "position_slip_m": float(np.linalg.norm(relative[:3, 3] - reference[:3, 3])),
            "rotation_slip_rad": float(np.arccos(np.clip(angle_cos, -1.0, 1.0))),
            "object_height_m": float(obj[2, 3]), "object_height_gain_m": float(obj[2, 3] - initial_z),
            "object_linear_velocity_m_s": linear.tolist(), "object_speed_m_s": float(np.linalg.norm(linear)),
            "object_angular_velocity_rad_s": angular.tolist(),
            "object_angular_speed_rad_s": float(np.linalg.norm(angular)),
            "is_grasping": bool(predicate.reshape(-1)[0]), "contact_forces_N": forces}


def plan_left_lift(left_solver: Any, right_solver: Any, target_pose: Any) -> DualArmSchedule:
    """Keep the grasp command CLOSED and plan only the left arm, without stepping."""
    left_solver.gripper_state, right_solver.gripper_state = -1, 1
    return plan_left_reach(left_solver, right_solver, target_pose)


def execute_lift(env: Any, schedule: DualArmSchedule, checker: Any, target_actor: Any, *,
                 lift_height_m: float = 0.12, hold_steps: int = 20,
                 max_position_slip_m: float = 0.005, max_rotation_slip_rad: float = 0.15,
                 grasp_loss_max_ticks: int = 2, min_lift_height_m: float = 0.10,
                 max_joint_step_rad: float = 0.02, tracking_tolerance_rad: float = 0.2,
                 endpoint_tolerance_rad: float = 0.03, velocity_tolerance_rad_s: float = 0.05,
                 settle_max_steps: int = 15, stable_steps: int = 3,
                 right_hold_tolerance_rad: float = 0.03, max_object_speed_m_s: float = 0.03,
                 max_object_angular_speed_rad_s: float = 0.2,
                 max_height_drop_m: float = 0.01, hover_position_drift_m: float = 0.003,
                 hover_rotation_drift_rad: float = 0.03) -> dict[str, Any]:
    """Validate a candidate lift, then measure actual attachment and 20-tick hover.

    min_lift_height_m is a physical object-height gain relative to this phase's
    initial measured pose. lift_height_m records the requested target offset;
    joint schedules alone do not encode a Cartesian endpoint. Nonzero physical
    finger positions are expected when holding an object. Settling and hover
    ticks are counted separately. Hover requires grasp on every tick. Stability
    uses net pose change on the control clock; raw physics velocity telemetry is
    retained. Hover also bounds displacement from its initial world pose. This
    does not establish that substep motion is absent or bounded.
    """
    nominal = _positive(lift_height_m, "lift_height_m")
    minimum = _positive(min_lift_height_m, "min_lift_height_m")
    if minimum > nominal:
        raise ValueError("min_lift_height_m cannot exceed the requested lift")
    hover_budget = _budget(hold_steps, "hold_steps")
    loss_limit = _budget(grasp_loss_max_ticks, "grasp_loss_max_ticks")
    settling = _budget(settle_max_steps, "settle_max_steps", zero=True)
    stable_required = _budget(stable_steps, "stable_steps")
    position_limit = _positive(max_position_slip_m, "max_position_slip_m")
    rotation_limit = _positive(max_rotation_slip_rad, "max_rotation_slip_rad")
    resolution = _positive(max_joint_step_rad, "max_joint_step_rad")
    tracking_limit = _positive(tracking_tolerance_rad, "tracking_tolerance_rad")
    endpoint_limit = _positive(endpoint_tolerance_rad, "endpoint_tolerance_rad")
    velocity_limit = _positive(velocity_tolerance_rad_s, "velocity_tolerance_rad_s")
    right_limit = _positive(right_hold_tolerance_rad, "right_hold_tolerance_rad")
    object_speed_limit = _positive(max_object_speed_m_s, "max_object_speed_m_s")
    angular_speed_limit = _positive(max_object_angular_speed_rad_s, "max_object_angular_speed_rad_s")
    drop_limit = _positive(max_height_drop_m, "max_height_drop_m")
    hover_position_limit = _positive(hover_position_drift_m, "hover_position_drift_m")
    hover_rotation_limit = _positive(hover_rotation_drift_rad, "hover_rotation_drift_rad")
    if not all(callable(getattr(checker, name, None)) for name in ("refresh", "check", "measured_check")):
        raise ValueError("Attachment checker requires refresh/check/measured_check")
    reference = _transform(checker.attachment_T_tcp_object)
    arms = bindings(env, tuple(t.action_key for t in schedule.trajectories))
    dt = _positive(env.unwrapped.control_timestep, "control_timestep")
    if not np.isclose(dt, schedule.control_timestep_s, rtol=0, atol=1e-10):
        raise ValueError("Environment and schedule clocks differ")
    if target_actor is None or not callable(getattr(arms[0].agent, "is_grasping", None)):
        raise ValueError("A real target and contact-based grasp predicate are required")
    for arm, source, command in zip(arms, schedule.trajectories, (-1, 1)):
        if source.joint_names != arm.names or source.gripper_command != command:
            raise ValueError("Require controller joint order, active hand CLOSED and passive hand OPEN")
    initial_z = float(_transform(target_actor.pose.to_transformation_matrix())[2, 3])
    initial = _observe(arms, target_actor, reference, initial_z)
    right_start = np.asarray(initial["measurements"][arms[1].key]["qpos_rad"])
    if not np.allclose(schedule.trajectories[1].positions_rad, right_start, rtol=0, atol=1e-6):
        raise ValueError("Right source trajectory must hold its measured entry position")
    actions = list(schedule.iter_actions())
    if any(not np.allclose(action[arms[1].key][:7], right_start, rtol=0, atol=1e-6) for action in actions):
        raise ValueError("Every scheduled right-arm target must hold the measured entry position")
    final = {a.key: np.r_[t.positions_rad[-1], t.gripper_command] for a, t in zip(arms, schedule.trajectories)}
    for action in [*actions, final]:
        _actions_valid(env, arms, action)
    report: dict[str, Any] = {
        "status": "running", "phase": "lift_and_hover", "environment_steps": 0, "trace": [],
        "planned_control_steps": len(actions), "settle_max_steps": settling, "settling_steps": 0,
        "hold_steps_required": hover_budget, "hold_steps_completed": 0,
        "stable_steps_required": stable_required, "control_timestep_s": dt,
        "requested_lift_height_m": nominal, "min_lift_height_m": minimum,
        "initial_object_height_m": initial_z, "attachment_T_tcp_object": reference.tolist(),
        "initial_observation": initial, "initial_elapsed_steps": as_numpy(env.unwrapped.elapsed_steps).tolist(),
        "collision_scope": str(getattr(checker, "scope", "provided_attachment_checker")),
        "continuous_collision_detection": False, "object_pose_assigned": False, "weld_added": False,
        "velocity_metric": "control_frame_pose_finite_difference",
        "velocity_thresholds": {"linear_m_s": object_speed_limit, "angular_rad_s": angular_speed_limit},
        "instant_physics_velocity_telemetry": "retained without alteration; not the control-frame stability metric",
        "hover_drift_limits": {"position_m": hover_position_limit, "rotation_rad": hover_rotation_limit},
        "abort_semantics": "stop advancing this phase; no braking, retreat, opening or continuation",
    }
    started = time.monotonic()
    if not initial["is_grasping"]:
        report.update(status="rejected", reason="initial_grasp_absent")
        return report
    if initial["position_slip_m"] > position_limit or initial["rotation_slip_rad"] > rotation_limit:
        report.update(status="rejected", reason="initial_attachment_slip")
        return report
    try:
        checker.refresh()
        report["initial_collisions"] = checker.measured_check()
        if report["initial_collisions"]:
            report.update(status="rejected", reason="initial_measured_collision")
            return report
        report["validation"] = validate_schedule(schedule, *(np.asarray(initial["measurements"][a.key]["qpos_rad"]) for a in arms),
                                                 checker, max_joint_step_rad=resolution)
    except ValueError:
        raise  # malformed schedules fail before the first environment step
    except Exception as exc:
        _error(report, "collision_preflight_failed", exc)
        report["status"] = "rejected"
        return report
    if report["validation"]["status"] != "passed":
        report.update(status="rejected", reason="candidate_attachment_collision")
        return report
    loss_ticks, peak_height, stable_count = 0, initial_z, 0
    previous_object = _transform(initial["object_T_world"])
    hover_reference: np.ndarray | None = None

    def advance(action: dict[str, Any], phase: str) -> dict[str, Any] | None:
        nonlocal loss_ticks, peak_height, previous_object
        actual = {key: np.asarray(value, dtype=np.float32).copy() for key, value in action.items()}
        row: dict[str, Any] = {"phase": phase, "step": report["environment_steps"] + 1,
                               "action": {k: v.tolist() for k, v in actual.items()}}
        before = as_numpy(env.unwrapped.elapsed_steps).copy()
        try:
            _, _, terminated, truncated, _ = env.step(actual)
            report["environment_steps"] += 1
            report["settling_steps"] += int(phase == "settling")
            row.update(time_s=report["environment_steps"] * dt,
                       terminated=bool(np.any(as_numpy(terminated))), truncated=bool(np.any(as_numpy(truncated))))
            row.update(_observe(arms, target_actor, reference, initial_z))
            current_object = _transform(row["object_T_world"])
            row["control_frame_dt_s"] = dt
            row["control_frame_linear_speed_m_s"] = float(np.linalg.norm(current_object[:3, 3] - previous_object[:3, 3]) / dt)
            row["control_frame_angular_speed_rad_s"] = _rotation_distance(previous_object[:3, :3], current_object[:3, :3]) / dt
            previous_object = current_object
            if phase == "hover":
                if hover_reference is None:
                    raise RuntimeError("Hover world-pose reference was not initialized")
                row["hover_position_drift_m"] = float(np.linalg.norm(current_object[:3, 3] - hover_reference[:3, 3]))
                row["hover_rotation_drift_rad"] = _rotation_distance(hover_reference[:3, :3], current_object[:3, :3])
            checker.refresh()
            row["collisions"] = checker.measured_check()
        except Exception as exc:
            observed = as_numpy(env.unwrapped.elapsed_steps)
            row["observed_elapsed_steps_after_error"] = observed.tolist()
            row["elapsed_step_delta_on_error"] = (observed - before).tolist()
            row["error"] = f"{type(exc).__name__}: {exc}"
            report["trace"].append(row)
            _error(report, "step_or_measurement_failed", exc)
            return None
        report["trace"].append(row)
        measurements = row["measurements"]
        row["tracking_error_rad"] = max(float(np.max(np.abs(np.asarray(measurements[a.key]["qpos_rad"]) - actual[a.key][:7]))) for a in arms)
        row["right_hold_error_rad"] = float(np.max(np.abs(np.asarray(measurements[arms[1].key]["qpos_rad"]) - right_start)))
        row["endpoint_error_rad"] = max(float(np.max(np.abs(np.asarray(measurements[a.key]["qpos_rad"]) - final[a.key][:7]))) for a in arms)
        row["max_arm_velocity_rad_s"] = max(float(np.max(np.abs(measurements[a.key]["qvel_rad_s"]))) for a in arms)
        peak_height = max(peak_height, row["object_height_m"])
        row["height_drop_from_peak_m"] = peak_height - row["object_height_m"]
        loss_ticks = 0 if row["is_grasping"] else loss_ticks + 1
        row["consecutive_grasp_loss_ticks"] = loss_ticks
        reason = None
        if row["collisions"]: reason = "measured_collision"
        elif (row["object_height_gain_m"] < -drop_limit
              or (row["height_drop_from_peak_m"] > drop_limit and not row["is_grasping"])): reason = "object_dropped"
        elif row["position_slip_m"] > position_limit: reason = "position_slip"
        elif row["rotation_slip_rad"] > rotation_limit: reason = "rotation_slip"
        elif loss_ticks >= loss_limit or (phase == "hover" and not row["is_grasping"]): reason = "grasp_lost"
        elif row["right_hold_error_rad"] > right_limit: reason = "right_hold_error"
        elif row["tracking_error_rad"] > tracking_limit: reason = "tracking_error"
        elif phase == "hover" and row["hover_position_drift_m"] > hover_position_limit: reason = "hover_position_drift"
        elif phase == "hover" and row["hover_rotation_drift_rad"] > hover_rotation_limit: reason = "hover_rotation_drift"
        if reason:
            report.update(status="aborted", reason=reason)
        elif row["terminated"] or row["truncated"]:
            report.update(status="terminated" if row["terminated"] else "truncated", reason="environment_done")
        return row

    def stable(row: dict[str, Any]) -> bool:
        return (row["is_grasping"] and row["object_height_gain_m"] >= minimum
                and row["endpoint_error_rad"] <= endpoint_limit and row["max_arm_velocity_rad_s"] <= velocity_limit
                and row["control_frame_linear_speed_m_s"] <= object_speed_limit
                and row["control_frame_angular_speed_rad_s"] <= angular_speed_limit)

    ready = False
    for index in range(len(actions) + settling):
        phase = "lifting" if index < len(actions) else "settling"
        row = advance(actions[index] if phase == "lifting" else final, phase)
        if row is None or report["status"] != "running": break
        stable_count = stable_count + 1 if stable(row) else 0
        row["stable_steps"] = stable_count
        if index + 1 >= len(actions) and stable_count >= stable_required:
            ready = True
            break
    if report["status"] == "running" and not ready:
        report.update(status="lift_timeout", reason="height_endpoint_or_velocity_not_stable")
    if ready and report["status"] == "running":
        hover_reference = _transform(report["trace"][-1]["object_T_world"])
        report["hover_reference_T_world"] = hover_reference.tolist()
        for _ in range(hover_budget):
            row = advance(final, "hover")
            if row is None or report["status"] != "running": break
            if not stable(row):
                report.update(status="aborted", reason="hover_not_stable")
                break
            report["hold_steps_completed"] += 1
        if report["status"] == "running": report["status"] = "passed"
    report.update(stable_steps_observed=stable_count, wall_seconds=time.monotonic() - started)
    if report["trace"] and "object_height_gain_m" in report["trace"][-1]:
        report["final_object_height_gain_m"] = report["trace"][-1]["object_height_gain_m"]
    return report


def self_test() -> dict[str, Any]:
    """Analytical fixture only; object-following below is simulated test input."""
    from types import SimpleNamespace as NS
    from gymnasium import spaces
    from mani_skill.examples.motionplanning.dual.trajectory_schedule import ArmTrajectory, build_hold_schedule
    names = tuple(f"joint{i}" for i in range(7))

    class Pose:
        def __init__(self, getter: Any): self.getter = getter
        def to_transformation_matrix(self) -> np.ndarray: return self.getter()[None].copy()

    class Robot:
        def __init__(self, fingers: float):
            self.q = np.array([[0.] * 7 + [fingers] * 2]); self.v = np.zeros((1, 9))
        def get_qpos(self) -> np.ndarray: return self.q
        def get_qvel(self) -> np.ndarray: return self.v
        def get_qlimits(self) -> np.ndarray: return np.array([[[-2, 2]] * 7 + [[0, .04]] * 2])
        def get_active_joints(self) -> list[Any]:
            return [NS(get_name=lambda n=n: n) for n in (*names, "f1", "f2")]

    class Env:
        def __init__(self, mode: str = "follow"):
            self.unwrapped = self; self.num_envs = 1; self.control_timestep = .1
            self.elapsed_steps = np.array([0]); self.mode = mode; self.actions = []
            self.object_matrix = np.eye(4); self.object_matrix[2, 3] = .02
            self.linear = np.zeros((1, 3)); self.angular = np.zeros((1, 3))
            self.agent = NS(agents_dict={})
            for key, fingers in (("left", .012), ("right", .04)):
                robot = Robot(fingers)
                arm = NS(config=NS(normalize_action=False, use_delta=False), joints=robot.get_active_joints()[:7])
                agent = NS(robot=robot, control_mode="pd_joint_pos", controller=NS(controllers={"arm": arm}))
                def tcp(robot: Any = robot) -> np.ndarray:
                    value = np.eye(4); value[2, 3] = .2 + robot.q[0, 0]
                    if self.mode == "hover_drift": value[0, 3] = .0002 * max(0, len(self.actions) - 5)
                    if self.mode == "hover_rotation":
                        angle = .005 * max(0, len(self.actions) - 5)
                        c, s = math.cos(angle), math.sin(angle)
                        value[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                    return value
                agent.tcp = NS(pose=Pose(tcp))
                agent.is_grasping = lambda target: np.array([self.mode != "grasp_loss" or not self.actions])
                self.agent.agents_dict[key] = agent
            self.agent.single_action_space = spaces.Dict({key: spaces.Box(-2., 2., (8,), dtype=np.float32) for key in ("left", "right")})
            self.actor = NS(pose=Pose(lambda: self.object_matrix), get_linear_velocity=lambda: self.linear,
                            get_angular_velocity=lambda: self.angular)
        def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
            assert set(action) == {"left", "right"} and action["left"][7] == -1 and action["right"][7] == 1
            self.actions.append(action)
            for key, value in action.items():
                robot = self.agent.agents_dict[key].robot
                before = robot.q.copy(); robot.q[0, :7] = value[:7]; robot.v = (robot.q - before) / .1
            if self.mode == "right_drift": self.agent.agents_dict["right"].robot.q[0, 0] += .05
            before_obj = self.object_matrix.copy()
            if self.mode != "not_follow": self.object_matrix[2, 3] = .02 + self.agent.agents_dict["left"].robot.q[0, 0]
            if self.mode == "position_slip": self.object_matrix[0, 3] = .006
            if self.mode == "rotation_slip":
                c, s = math.cos(.2), math.sin(.2); self.object_matrix[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            if self.mode == "dropped": self.object_matrix[2, 3] = 0.
            if self.mode == "hover_drift": self.object_matrix[0, 3] = .0002 * max(0, len(self.actions) - 5)
            if self.mode == "hover_rotation":
                angle = .005 * max(0, len(self.actions) - 5)
                c, s = math.cos(angle), math.sin(angle)
                self.object_matrix[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            self.linear = ((self.object_matrix[:3, 3] - before_obj[:3, 3]) / .1)[None]
            if self.mode == "instant_high":
                self.linear = np.array([[.1, 0., 0.]])
                self.angular = np.array([[1.6, 0., 0.]])
            self.elapsed_steps += 1
            return None, 0, False, len(self.actions) >= getattr(self, "truncate_at", 999), {}

    class Checker:
        scope = "analytical_attachment_fixture"
        attachment_T_tcp_object = np.eye(4); attachment_T_tcp_object[2, 3] = -.18
        def __init__(self, env: Env, candidate_collision: bool = False, measured_collision: bool = False):
            self.env, self.candidate_collision, self.measured_collision = env, candidate_collision, measured_collision
        def refresh(self) -> None: pass
        def check(self, left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
            return [{"collision_type": "candidate_fixture"}] if self.candidate_collision and left[0] > .03 else []
        def measured_check(self) -> list[dict[str, Any]]:
            return [{"collision_type": "measured_fixture"}] if self.measured_collision and self.env.actions else []

    def schedule() -> DualArmSchedule:
        q = np.zeros((2, 7)); q[1, 0] = .12
        left = ArmTrajectory("left", names, [0, .3], q, -1)
        right = ArmTrajectory("right", names, [0], [np.zeros(7)], 1)
        return build_hold_schedule(left, right, control_timestep_s=.1)

    env = Env(); passed = execute_lift(env, schedule(), Checker(env), env.actor, stable_steps=2)
    assert passed["status"] == "passed" and passed["hold_steps_completed"] == 20
    assert passed["settling_steps"] == 2 and len(env.actions) == passed["environment_steps"] == 25
    assert np.isclose(passed["final_object_height_gain_m"], .12) and env.agent.agents_dict["left"].robot.q[0, 7] == .012
    assert all(np.array_equal(action["right"][:7], np.zeros(7)) for action in env.actions)
    env = Env("instant_high")
    result = execute_lift(env, schedule(), Checker(env), env.actor, stable_steps=2)
    assert result["status"] == "passed" and result["hold_steps_completed"] == 20
    hover = [row for row in result["trace"] if row["phase"] == "hover"]
    assert all(row["object_speed_m_s"] == .1 and row["object_angular_speed_rad_s"] == 1.6 for row in hover)
    assert all(row["control_frame_linear_speed_m_s"] == 0 and row["control_frame_angular_speed_rad_s"] == 0 for row in hover)
    for mode, reason in (("hover_drift", "hover_position_drift"), ("hover_rotation", "hover_rotation_drift")):
        env = Env(mode)
        result = execute_lift(env, schedule(), Checker(env), env.actor, stable_steps=2)
        assert result["status"] == "aborted" and result["reason"] == reason
        assert result["hold_steps_completed"] < 20
        assert all(row["control_frame_linear_speed_m_s"] < .03 and row["control_frame_angular_speed_rad_s"] < .2
                   for row in result["trace"] if row["phase"] == "hover")
    for mode, reason in (("not_follow", "position_slip"), ("position_slip", "position_slip"),
                         ("rotation_slip", "rotation_slip"), ("dropped", "object_dropped"),
                         ("grasp_loss", "grasp_lost"), ("right_drift", "right_hold_error")):
        env = Env(mode); result = execute_lift(env, schedule(), Checker(env), env.actor)
        assert result["status"] == "aborted" and result["reason"] == reason, (mode, result)
        assert len(env.actions) <= 2 and result["hold_steps_completed"] == 0
    env = Env(); result = execute_lift(env, schedule(), Checker(env, candidate_collision=True), env.actor)
    assert result["status"] == "rejected" and not env.actions
    env = Env(); result = execute_lift(env, schedule(), Checker(env, measured_collision=True), env.actor)
    assert result["status"] == "aborted" and len(env.actions) == 1
    env = Env(); env.truncate_at = 1
    assert execute_lift(env, schedule(), Checker(env), env.actor)["status"] == "truncated" and len(env.actions) == 1
    for kwargs in ({"hold_steps": 0}, {"max_position_slip_m": float("nan")}, {"min_lift_height_m": .2}):
        env = Env()
        try: execute_lift(env, schedule(), Checker(env), env.actor, **kwargs)
        except ValueError: pass
        else: raise AssertionError("Invalid lift inputs accepted")
        assert not env.actions
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
            "checks": ["nonzero held fingers", "stationary right arm", "20 measured hover ticks",
                       "explicit settling", "object not following TCP", "translation slip", "rotation slip",
                       "object drop", "grasp loss", "right drift", "candidate collision before motion",
                       "measured collision", "truncation", "invalid inputs before motion",
                       "high instantaneous velocity retained with stable control-frame poses",
                       "slow cumulative hover position drift rejected", "slow cumulative hover rotation drift rejected"],
            "fixture_total_steps": passed["environment_steps"], "fixture_settling_steps": passed["settling_steps"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
