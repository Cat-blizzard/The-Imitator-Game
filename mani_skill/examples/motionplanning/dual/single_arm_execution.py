"""Single active Panda phases with canonical physical (left, right) collision I/O.

Only bindings used by the observation helper are active-first. Environment,
task predicates, schedules and checker calls always retain physical identities.
No physical pose, weld, reset, automatic release or recovery is applied here.
An abort stops advancing this phase. CPU analytical fixtures are not native
physics validation or evidence of a completed benchmark task.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np

from mani_skill.examples.motionplanning.dual.combined_collision import validate_schedule
from mani_skill.examples.motionplanning.dual.grasp_execution import (
    _actions_valid, _budget, _error, _measure, _positive, close_and_confirm as _native_close,
)
from mani_skill.examples.motionplanning.dual.lift_execution import _observe, _transform, _rotation_distance
from mani_skill.examples.motionplanning.dual.trajectory_execution import as_numpy, bindings, snapshot
from mani_skill.examples.motionplanning.dual.trajectory_schedule import ArmTrajectory, DualArmSchedule, build_hold_schedule, plan_pose

JOINT_STEP, TRACK, ENDPOINT, ARM_SPEED, OPEN_TOL = .02, .2, .03, .05, .002
POSITION_SLIP, ROTATION_SLIP, DROP = .005, .15, .01
OBJECT_SPEED, OBJECT_ANGULAR_SPEED, WINDOW_POSITION, WINDOW_ROTATION = .03, .2, .003, .03
WINDOW_JOINT = .003


def _bool(value: Any) -> bool:
    array = as_numpy(value)
    if array.size != 1 or array.dtype.kind not in "biuf" or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError("Expected one finite boolean predicate")
    return bool(array.reshape(-1)[0])


def _canonical(env: Any) -> tuple[Any, Any]:
    raw = env.unwrapped
    agents = raw.agent.agents
    if len(agents) != 2: raise ValueError("Exactly two physical robots are required")
    keys = []
    for agent in agents:  # MultiAgent physical left, right; never role order.
        matches = [key for key, entry in raw.agent.agents_dict.items() if entry is agent]
        if len(matches) != 1: raise ValueError("Each physical robot needs one unique action key")
        keys.append(matches[0])
    return bindings(env, tuple(keys))


def _identity(arms: Any, active_key: str, checker: Any) -> tuple[Any, Any]:
    if active_key not in {a.key for a in arms}: raise ValueError("Unknown active robot key")
    active = next(a for a in arms if a.key == active_key)
    passive = next(a for a in arms if a.key != active_key)
    expected = "left" if active is arms[0] else "right"
    if hasattr(checker, "active_robot") and checker.active_robot != expected:
        raise ValueError("Collision checker is bound to a different physical active robot")
    return active, passive


class _Phase:
    """One actual environment, one canonical checker adapter, one tick recorder."""

    def __init__(self, env: Any, checker: Any, active_key: str, name: str, *, target: Any = None, held: bool = False):
        self.env, self.checker, self.held = env, checker, held
        self.arms = _canonical(env)
        self.active, self.passive = _identity(self.arms, active_key, checker)
        required = ("refresh", "check", "measured_check") if held else ("refresh", "check", "set_target_contact_allowed")
        if not all(callable(getattr(checker, key, None)) for key in required): raise ValueError("Incomplete checker interface")
        self.dt = _positive(env.unwrapped.control_timestep, "control_timestep")
        self.target = target
        self.roles = (self.active, self.passive)  # observation only, never checker I/O.
        self.reference, self.initial_z, self.previous = None, 0., None
        if target is not None:
            obj = _transform(target.pose.to_transformation_matrix())
            tcp = _transform(self.active.agent.tcp.pose.to_transformation_matrix())
            self.reference = _transform(checker.attachment_T_tcp_object) if held else _transform(np.linalg.inv(tcp) @ obj)
            self.initial_z = float(obj[2, 3]); self.previous = obj.copy()
        elif held: raise ValueError("Held phases require a real target actor")
        initial = self.observe()
        self.previous_joints = {a.key: np.asarray(initial["measurements"][a.key]["qpos_rad"]).copy() for a in self.arms}
        self.passive_start = np.asarray(initial["measurements"][self.passive.key]["qpos_rad"])
        self.report: dict[str, Any] = {
            "status": "running", "phase": name, "active_key": self.active.key, "passive_key": self.passive.key,
            "canonical_action_keys": [a.key for a in self.arms], "environment_steps": 0, "settling_steps": 0,
            "hold_steps_completed": 0, "trace": [], "initial_observation": initial,
            "initial_elapsed_steps": as_numpy(env.unwrapped.elapsed_steps).tolist(), "control_timestep_s": self.dt,
            "collision_scope": str(getattr(checker, "scope", "provided_checker")),
            "velocity_metric": "control_frame_pose_finite_difference", "continuous_collision_detection": False,
            "arm_velocity_metric": "control_frame_joint_position_finite_difference" if held else "reported_joint_velocity",
            "thresholds": {"joint_sample_rad": JOINT_STEP, "joint_tracking_rad": TRACK, "endpoint_and_passive_rad": ENDPOINT,
                           "arm_speed_rad_s": ARM_SPEED, "open_finger_m": OPEN_TOL, "position_slip_m": POSITION_SLIP,
                           "rotation_slip_rad": ROTATION_SLIP, "drop_below_attachment_m": DROP,
                           "grasp_loss_ticks": 2, "object_speed_m_s": OBJECT_SPEED, "object_angular_speed_rad_s": OBJECT_ANGULAR_SPEED,
                           "window_position_m": WINDOW_POSITION, "window_rotation_rad": WINDOW_ROTATION,
                           "window_joint_rad": WINDOW_JOINT},
            "abort_semantics": "stop advancing this phase; no automatic release, reset, retreat or continuation",
        }
        self.started, self.loss = time.monotonic(), 0

    def observe(self) -> dict[str, Any]:
        if self.target is None:
            return {"measurements": _measure(self.arms), "is_grasping": None,
                    "tcp_T_world": _transform(self.active.agent.tcp.pose.to_transformation_matrix()).tolist()}
        value = _observe(self.roles, self.target, self.reference, self.initial_z)
        value["measurements"] = _measure(self.arms)
        return value

    def opened(self, arm: Any, row: dict[str, Any]) -> bool:
        fingers = [i for i in range(9) if i not in arm.indices]
        upper = as_numpy(arm.agent.robot.get_qlimits())[0, fingers, 1]
        return bool(np.allclose(row["measurements"][arm.key]["finger_qpos_m"], upper, rtol=0, atol=OPEN_TOL))

    def collisions(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        self.checker.refresh()
        if self.held: return self.checker.measured_check()
        return self.checker.check(*(np.asarray(row["measurements"][a.key]["qpos_rad"]) for a in self.arms))

    def preflight(self) -> bool:
        try: collisions = self.collisions(self.report["initial_observation"])
        except Exception as exc:
            _error(self.report, "collision_preflight_failed", exc); self.report["status"] = "rejected"; return False
        self.report["initial_collisions"] = collisions
        if collisions: self.report.update(status="rejected", reason="initial_collision"); return False
        return True

    def hold_action(self, active_grip: int) -> dict[str, np.ndarray]:
        return {a.key: np.r_[self.report["initial_observation"]["measurements"][a.key]["qpos_rad"], active_grip if a is self.active else 1]
                for a in self.arms}

    def tick(self, action: dict[str, Any], phase: str) -> dict[str, Any] | None:
        actual = {a.key: np.asarray(action[a.key], dtype=np.float32).copy() for a in self.arms}
        row: dict[str, Any] = {"phase": phase, "step": self.report["environment_steps"] + 1,
                               "action": {k: v.tolist() for k, v in actual.items()}}
        before = as_numpy(self.env.unwrapped.elapsed_steps).copy()
        try:
            _, _, terminated, truncated, _ = self.env.step(actual)
            self.report["environment_steps"] += 1
            self.report["settling_steps"] += int(phase == "settling")
            row.update(time_s=self.report["environment_steps"] * self.dt, terminated=_bool(terminated), truncated=_bool(truncated))
            row.update(self.observe())
            current_joints = {a.key: np.asarray(row["measurements"][a.key]["qpos_rad"]).copy() for a in self.arms}
            joint_velocity = {a.key: (current_joints[a.key] - self.previous_joints[a.key]) / self.dt for a in self.arms}
            row["control_frame_joint_velocity_rad_s"] = {a.key: joint_velocity[a.key].tolist() for a in self.arms}
            row["max_control_frame_joint_speed_rad_s"] = max(float(np.max(np.abs(joint_velocity[a.key]))) for a in self.arms)
            self.previous_joints = current_joints
            if self.target is not None:
                pose = _transform(row["object_T_world"])
                row["control_frame_linear_speed_m_s"] = float(np.linalg.norm(pose[:3, 3] - self.previous[:3, 3]) / self.dt)
                row["control_frame_angular_speed_rad_s"] = _rotation_distance(self.previous[:3, :3], pose[:3, :3]) / self.dt
                self.previous = pose
            row["collisions"] = self.collisions(row)
            row["tracking_error_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][a.key]["qpos_rad"]) - actual[a.key][:7]))) for a in self.arms)
            row["passive_hold_error_rad"] = float(np.max(np.abs(np.asarray(row["measurements"][self.passive.key]["qpos_rad"]) - self.passive_start)))
            row["max_arm_velocity_rad_s"] = max(float(np.max(np.abs(row["measurements"][a.key]["qvel_rad_s"]))) for a in self.arms)
        except Exception as exc:
            row["observed_elapsed_steps_after_error"] = as_numpy(self.env.unwrapped.elapsed_steps).tolist()
            row["elapsed_step_delta_on_error"] = (as_numpy(self.env.unwrapped.elapsed_steps) - before).tolist()
            row["error"] = f"{type(exc).__name__}: {exc}"; self.report["trace"].append(row)
            _error(self.report, "step_or_measurement_failed", exc); return None
        self.report["trace"].append(row)
        if row["collisions"]: self.report.update(status="aborted", reason="measured_collision")
        elif row["passive_hold_error_rad"] > ENDPOINT: self.report.update(status="aborted", reason="passive_hold_error")
        elif row["terminated"] or row["truncated"]:
            self.report.update(status="terminated" if row["terminated"] else "truncated", reason="environment_done")
        return row

    def monitor_held(self, row: dict[str, Any], *, hovering: bool = False) -> bool:
        expected = np.asarray(row["tcp_T_world"]) @ self.reference
        row["drop_below_predicted_attachment_m"] = max(0., float(expected[2, 3] - row["object_height_m"]))
        self.loss = 0 if row["is_grasping"] else self.loss + 1
        row["consecutive_grasp_loss_ticks"] = self.loss
        reason = None
        if row["drop_below_predicted_attachment_m"] > DROP: reason = "object_dropped"
        elif row["position_slip_m"] > POSITION_SLIP: reason = "position_slip"
        elif row["rotation_slip_rad"] > ROTATION_SLIP: reason = "rotation_slip"
        elif self.loss >= 2 or (hovering and not row["is_grasping"]): reason = "grasp_lost"
        if reason: self.report.update(status="aborted", reason=reason)
        return reason is None

    def finish(self) -> dict[str, Any]:
        self.report["wall_seconds"] = time.monotonic() - self.started
        return self.report


def plan_single_pose(left_solver: Any, right_solver: Any, target_pose: Any, *, active_key: str,
                     gripper_command: int = 1) -> DualArmSchedule:
    raw = left_solver.base_env
    if raw is not right_solver.base_env: raise ValueError("Both solvers must share the actual environment")
    arms = _canonical(raw)
    if any(s.env_agent is not a.agent for s, a in zip((left_solver, right_solver), arms)):
        raise ValueError("Solvers must be passed in physical left, right order")
    if active_key not in {a.key for a in arms} or gripper_command not in (-1, 1): raise ValueError("Invalid active key or phase gripper command")
    before = snapshot(raw, arms); trajectories = []
    for solver, arm in zip((left_solver, right_solver), arms):
        solver.gripper_state = gripper_command if arm.key == active_key else 1
        trajectories.append(plan_pose(solver, target_pose, action_key=arm.key, controller_joint_names=arm.names)
                            if arm.key == active_key else ArmTrajectory(arm.key, arm.names, [0.], [arm.arm_qpos()], 1))
    if not all(np.array_equal(a, b) for a, b in zip(before, snapshot(raw, arms))): raise RuntimeError("Environment changed during dry-run planning")
    return build_hold_schedule(*trajectories, control_timestep_s=raw.control_timestep)


def _scheduled(env: Any, schedule: DualArmSchedule, checker: Any, *, active_key: str,
               target: Any, held: bool, settle_max_steps: int, stable_steps: int,
               minimum: float | None, hold_steps: int) -> dict[str, Any]:
    settling = _budget(settle_max_steps, "settle_max_steps", zero=True)
    required = _budget(stable_steps, "stable_steps"); hover = _budget(hold_steps, "hold_steps", zero=True)
    if minimum is not None: minimum = _positive(minimum, "min_height_gain_m")
    phase = _Phase(env, checker, active_key, "held_move" if held else "reach", target=target, held=held)
    initial = phase.report["initial_observation"]
    if tuple(t.action_key for t in schedule.trajectories) != tuple(a.key for a in phase.arms): raise ValueError("Schedule must use canonical physical left, right order")
    if not np.isclose(schedule.control_timestep_s, phase.dt, rtol=0, atol=1e-10): raise ValueError("Control clocks differ")
    actions = list(schedule.iter_actions()); final = {}
    for arm, source in zip(phase.arms, schedule.trajectories):
        grip = -1 if held and arm is phase.active else 1
        if source.joint_names != arm.names or source.gripper_command != grip: raise ValueError("Invalid controller joint order or gripper command")
        if arm is phase.passive and not np.allclose(source.positions_rad, phase.passive_start, rtol=0, atol=1e-6): raise ValueError("Passive source must hold its measured entry joints")
        final[arm.key] = np.r_[source.positions_rad[-1], grip]
        if (arm is phase.passive or not held) and not phase.opened(arm, initial): raise ValueError("Required open hand is not physically open")
    if any(not np.allclose(a[phase.passive.key][:7], phase.passive_start, rtol=0, atol=1e-6) for a in actions): raise ValueError("Passive command must hold its entry joints")
    for action in [*actions, final]: _actions_valid(env, phase.arms, action)
    phase.report.update(planned_control_steps=len(actions), settle_max_steps=settling, stable_steps_required=required,
                        min_height_gain_m=minimum, hold_steps_required=hover)
    if held and (not initial["is_grasping"] or initial["position_slip_m"] > POSITION_SLIP or initial["rotation_slip_rad"] > ROTATION_SLIP):
        phase.report.update(status="rejected", reason="initial_grasp_or_attachment_invalid"); return phase.finish()
    if not phase.preflight(): return phase.finish()
    phase.report["validation"] = validate_schedule(schedule, *(np.asarray(initial["measurements"][a.key]["qpos_rad"]) for a in phase.arms), checker, max_joint_step_rad=JOINT_STEP)
    if phase.report["validation"]["status"] != "passed": phase.report.update(status="rejected", reason="candidate_collision"); return phase.finish()
    def stable(row: dict[str, Any]) -> bool:
        arm_speed = row["max_control_frame_joint_speed_rad_s"] if held else row["max_arm_velocity_rad_s"]
        result = row["endpoint_error_rad"] <= ENDPOINT and arm_speed <= ARM_SPEED
        return result and (not held or (row["is_grasping"] and row["control_frame_linear_speed_m_s"] <= OBJECT_SPEED
            and row["control_frame_angular_speed_rad_s"] <= OBJECT_ANGULAR_SPEED
            and (minimum is None or row["object_height_gain_m"] >= minimum)))
    count, ready = 0, False
    for index in range(len(actions) + settling):
        label = "moving" if index < len(actions) else "settling"
        row = phase.tick(actions[index] if label == "moving" else final, label)
        if row is None or phase.report["status"] != "running": break
        row["endpoint_error_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][a.key]["qpos_rad"]) - final[a.key][:7]))) for a in phase.arms)
        if row["tracking_error_rad"] > TRACK: phase.report.update(status="aborted", reason="tracking_error"); break
        if held and not phase.monitor_held(row): break
        count = count + 1 if stable(row) else 0; row["stable_steps"] = count
        if index + 1 >= len(actions) and count >= required: ready = True; break
    if phase.report["status"] == "running" and not ready:
        phase.report.update(status="carry_timeout" if held else "reach_timeout", reason="endpoint_height_or_velocity_not_stable")
    if ready and phase.report["status"] == "running":
        reference = _transform(phase.report["trace"][-1]["object_T_world"]) if held and hover else None
        if reference is not None:
            phase.report["hold_reference_T_world"] = reference.tolist()
            joint_reference = {a.key: np.asarray(phase.report["trace"][-1]["measurements"][a.key]["qpos_rad"]).copy() for a in phase.arms}
            phase.report["hold_reference_joint_qpos_rad"] = {a.key: joint_reference[a.key].tolist() for a in phase.arms}
        for _ in range(hover):
            row = phase.tick(final, "hover")
            if row is None or phase.report["status"] != "running": break
            row["endpoint_error_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][a.key]["qpos_rad"]) - final[a.key][:7]))) for a in phase.arms)
            if not phase.monitor_held(row, hovering=True): break
            pose = _transform(row["object_T_world"])
            row["hold_position_drift_m"] = float(np.linalg.norm(pose[:3, 3] - reference[:3, 3]))
            row["hold_rotation_drift_rad"] = _rotation_distance(reference[:3, :3], pose[:3, :3])
            row["hold_joint_drift_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][a.key]["qpos_rad"]) - joint_reference[a.key]))) for a in phase.arms)
            if not stable(row) or row["hold_position_drift_m"] > WINDOW_POSITION or row["hold_rotation_drift_rad"] > WINDOW_ROTATION or row["hold_joint_drift_rad"] > WINDOW_JOINT:
                phase.report.update(status="aborted", reason="hold_not_stable"); break
            phase.report["hold_steps_completed"] += 1
        if phase.report["status"] == "running": phase.report["status"] = "passed"
    phase.report["stable_steps_observed"] = count
    return phase.finish()


def execute_reach(env: Any, schedule: DualArmSchedule, checker: Any, *, active_key: str,
                  allow_target_contact: bool = False, settle_max_steps: int = 15, stable_steps: int = 3) -> dict[str, Any]:
    if not isinstance(allow_target_contact, bool): raise ValueError("Contact permission must be bool")
    target = getattr(checker, "target_actor", None)
    if target is None:
        actors = getattr(getattr(env.unwrapped, "scene", None), "actors", {})
        target = actors.get(getattr(checker, "target_actor_name", None))
    checker.set_target_contact_allowed(allow_target_contact)
    try: return _scheduled(env, schedule, checker, active_key=active_key, target=target, held=False,
                          settle_max_steps=settle_max_steps, stable_steps=stable_steps, minimum=None, hold_steps=0)
    finally: checker.set_target_contact_allowed(False)


def close_and_confirm(env: Any, checker: Any, *, active_key: str, target_actor: Any, **kwargs: Any) -> dict[str, Any]:
    arms = _canonical(env); active, passive = _identity(arms, active_key, checker)
    if tuple(env.unwrapped.agent.agents_dict) != tuple(a.key for a in arms): raise ValueError("Legacy contact confirmer requires canonical dictionary insertion order")
    report = _native_close(env, checker, active_key=active_key, target_actor=target_actor, **kwargs)
    report.update(active_key=active.key, passive_key=passive.key, canonical_action_keys=[a.key for a in arms])
    entry = np.asarray(report["initial_measurements"][passive.key]["qpos_rad"])
    for row in report["trace"]:
        if "measurements" in row:
            row["passive_hold_error_rad"] = float(np.max(np.abs(np.asarray(row["measurements"][passive.key]["qpos_rad"]) - entry)))
    return report


def execute_held_move(env: Any, schedule: DualArmSchedule, checker: Any, target_actor: Any, *,
                      active_key: str, min_height_gain_m: float | None = None, hold_steps: int = 0,
                      settle_max_steps: int = 15, stable_steps: int = 3) -> dict[str, Any]:
    return _scheduled(env, schedule, checker, active_key=active_key, target=target_actor, held=True,
                      settle_max_steps=settle_max_steps, stable_steps=stable_steps, minimum=min_height_gain_m, hold_steps=hold_steps)


def release_and_confirm(env: Any, live_checker: Any, target_actor: Any, *, active_key: str,
                        max_steps: int = 30, stable_steps: int = 3) -> dict[str, Any]:
    if target_actor is None: raise ValueError("Release confirmation requires a real target actor")
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    phase = _Phase(env, live_checker, active_key, "release", target=target_actor)
    if not phase.opened(phase.passive, phase.report["initial_observation"]): raise ValueError("Passive hand must already be open")
    action = phase.hold_action(1); _actions_valid(env, phase.arms, action)
    live_checker.set_target_contact_allowed(True); count = 0
    try:
        if not phase.preflight(): return phase.finish()
        for _ in range(budget):
            row = phase.tick(action, "releasing")
            if row is None or phase.report["status"] != "running": break
            row["fingers_open"] = all(phase.opened(a, row) for a in phase.arms)
            if row["tracking_error_rad"] > ENDPOINT: phase.report.update(status="aborted", reason="arm_hold_error"); break
            count = count + 1 if row["fingers_open"] and not row["is_grasping"] else 0; row["stable_steps"] = count
            if count >= required: phase.report["status"] = "passed"; break
        if phase.report["status"] == "running": phase.report.update(status="release_timeout", reason="fingers_not_open_or_grasp_persists")
        phase.report.update(max_steps=budget, stable_steps_required=required, stable_steps_observed=count)
        return phase.finish()
    finally: live_checker.set_target_contact_allowed(False)


def _region(env: Any, actor: Any, official_key: str, predicate: Callable[[Any], Any]) -> tuple[bool, dict[str, Any]]:
    value = env.unwrapped.evaluate()
    if official_key not in value: raise ValueError("Requested official region predicate is missing")
    official = _bool(value[official_key]); physical = predicate(actor)
    details = dict(physical) if isinstance(physical, dict) else {"inside": physical}
    if "inside" not in details: raise ValueError("Physical predicate requires an inside boolean")
    details["inside"] = _bool(details["inside"]); json.dumps(details, allow_nan=False)
    return official, details


def verify_placement(env: Any, live_checker: Any, target_actor: Any, *, active_key: str,
                     official_predicate_key: str, region_predicate: Callable[[Any], Any],
                     max_steps: int = 60, stable_steps: int = 20) -> dict[str, Any]:
    if target_actor is None: raise ValueError("Placement verification requires a real target actor")
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    if not isinstance(official_predicate_key, str) or not official_predicate_key or not callable(region_predicate): raise ValueError("Actual official and physical region predicates are required")
    phase = _Phase(env, live_checker, active_key, "placement_verification", target=target_actor)
    if not all(phase.opened(a, phase.report["initial_observation"]) for a in phase.arms): raise ValueError("Both hands must already be physically open")
    official, physical = _region(env, target_actor, official_predicate_key, region_predicate)
    phase.report.update(official_predicate_key=official_predicate_key, initial_official_region=official,
                        initial_physical_region=physical, max_steps=budget, stable_steps_required=required)
    action = phase.hold_action(1); _actions_valid(env, phase.arms, action)
    live_checker.set_target_contact_allowed(False)
    if not phase.preflight(): return phase.finish()
    count, window = 0, None
    for _ in range(budget):
        row = phase.tick(action, "verifying_placement")
        if row is None or phase.report["status"] != "running": break
        try: row["official_region"], row["physical_region"] = _region(env, target_actor, official_predicate_key, region_predicate)
        except Exception as exc: _error(phase.report, "region_measurement_failed", exc); row["error"] = str(exc); break
        row["fingers_open"] = all(phase.opened(a, row) for a in phase.arms)
        if row["tracking_error_rad"] > ENDPOINT: phase.report.update(status="aborted", reason="arm_hold_error"); break
        valid = (row["official_region"] and row["physical_region"]["inside"] and not row["is_grasping"] and row["fingers_open"]
                 and row["max_arm_velocity_rad_s"] <= ARM_SPEED and row["control_frame_linear_speed_m_s"] <= OBJECT_SPEED
                 and row["control_frame_angular_speed_rad_s"] <= OBJECT_ANGULAR_SPEED)
        pose = _transform(row["object_T_world"])
        if not valid: count, window = 0, None; row["stability_reset_reason"] = "region_release_or_velocity"
        else:
            if window is None: window = pose.copy()
            dp = float(np.linalg.norm(pose[:3, 3] - window[:3, 3])); dr = _rotation_distance(window[:3, :3], pose[:3, :3])
            row.update(window_position_drift_m=dp, window_rotation_drift_rad=dr)
            if dp > WINDOW_POSITION or dr > WINDOW_ROTATION: count, window = 1, pose.copy(); row["stability_reset_reason"] = "cumulative_window_drift"
            else: count += 1
        row["stable_steps"] = count
        if count >= required: phase.report["status"] = "passed"; break
    if phase.report["status"] == "running": phase.report.update(status="placement_timeout", reason="placement_not_continuously_stable")
    phase.report["stable_steps_observed"] = count
    return phase.finish()


def self_test() -> dict[str, Any]:
    from types import SimpleNamespace as NS
    from gymnasium import spaces
    names = tuple(f"joint{i}" for i in range(7))
    class Pose:
        def __init__(self, getter: Any): self.getter = getter
        def to_transformation_matrix(self) -> np.ndarray: return self.getter()[None].copy()
        @property
        def raw_pose(self) -> np.ndarray:
            m = self.getter(); return np.r_[m[:3, 3], [1., 0., 0., 0.]][None]
    class Robot:
        def __init__(self, left: bool, held: bool):
            self.q = np.array([[.25 if left else 0.] + [0.] * 6 + [.012 if held and not left else .04] * 2]); self.v = np.zeros((1, 9))
            self.pose = Pose(lambda: np.eye(4))
        def get_qpos(self) -> np.ndarray: return self.q
        def get_qvel(self) -> np.ndarray: return self.v
        def get_qlimits(self) -> np.ndarray: return np.array([[[-2, 2]] * 7 + [[0, .04]] * 2])
        def get_active_joints(self) -> list[Any]: return [NS(get_name=lambda n=n: n) for n in (*names, "f1", "f2")]
    class Env:
        def __init__(self, held: bool = True, mode: str = "normal"):
            self.unwrapped = self; self.num_envs = 1; self.control_timestep = .1; self.elapsed_steps = np.array([0])
            self.mode, self.actions = mode, []; self.matrix = np.eye(4); self.matrix[2, 3] = .12; self.linear = np.zeros((1, 3))
            self.agent = NS(agents=[], agents_dict={})
            for key in ("left", "right"):
                robot = Robot(key == "left", held); arm = NS(joints=robot.get_active_joints()[:7], config=NS(normalize_action=False, use_delta=False, joint_names=names))
                entry = NS(robot=robot, control_mode="pd_joint_pos", controller=NS(controllers={"arm": arm}))
                def tcp(robot: Any = robot) -> np.ndarray:
                    m = np.eye(4); m[0, 3] = robot.q[0, 0]; m[2, 3] = .3 + robot.q[0, 1]; return m
                entry.tcp = NS(pose=Pose(tcp)); entry.is_grasping = lambda target, robot=robot: np.array([robot.q[0, 7] < .03 and not (self.mode == "grasp_loss" and self.actions)])
                self.agent.agents.append(entry); self.agent.agents_dict[key] = entry
            self.agent.single_action_space = spaces.Dict({k: spaces.Box(-2., 2., (8,), dtype=np.float32) for k in ("left", "right")})
            self.actor = NS(pose=Pose(lambda: self.matrix), get_linear_velocity=lambda: self.linear, get_angular_velocity=lambda: np.zeros((1, 3)))
        def evaluate(self) -> dict[str, Any]: return {"apple_in_basket": self.mode == "outside", "banana_in_basket": self.mode != "outside" and not (self.mode == "bounce" and len(self.actions) == 11)}
        def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
            assert list(action) == ["left", "right"] and action["left"][7] == 1; self.actions.append(action)
            for key, value in action.items():
                robot = self.agent.agents_dict[key].robot; old = robot.q.copy(); robot.q[0, :7] = value[:7]
                if key == "right" and value[7] == -1: robot.q[0, 7:] = .012
                elif value[7] == 1 and not (key == "right" and self.mode == "retain"): robot.q[0, 7:] = .04
                robot.v = (robot.q - old) / .1
            if self.mode == "high_raw":
                for entry in self.agent.agents: entry.robot.v[0, :7] = .06
            if self.mode == "joint_fd_motion":
                self.agent.agents[1].robot.q[0, 2] += .006 * len(self.actions)
                for entry in self.agent.agents: entry.robot.v[0, :7] = 0.
            if self.mode == "joint_hover_drift":
                self.agent.agents[1].robot.q[0, 2] += .0002 * max(0, len(self.actions) - 5)
            old = self.matrix.copy(); right = self.agent.agents[1].robot
            if right.q[0, 7] < .03: self.matrix[:3, 3] = [right.q[0, 0], 0., .12 + right.q[0, 1]]
            if self.mode == "slide": self.matrix[0, 3] += .006
            if self.mode == "drift": self.matrix[0, 3] += .0002
            if self.mode == "passive_drift": self.agent.agents[0].robot.q[0, 0] += .05
            self.linear = ((self.matrix[:3, 3] - old[:3, 3]) / .1)[None]; self.elapsed_steps += 1
            return None, 0, False, len(self.actions) >= getattr(self, "truncate_at", 999), {}
    class Checker:
        scope = "analytical_canonical_checker"; active_robot = "right"
        attachment_T_tcp_object = np.eye(4); attachment_T_tcp_object[2, 3] = -.18
        def __init__(self, env: Env, collide: bool = False): self.env, self.target_actor, self.collide, self.calls, self.allowed = env, env.actor, collide, [], False
        def set_target_contact_allowed(self, value: bool) -> None: self.allowed = value
        def refresh(self) -> None: pass
        def check(self, left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
            assert left[0] >= .249 and right[0] < .2, "Collision arguments were role-swapped"
            self.calls.append((left.copy(), right.copy())); return [{"fixture": True}] if self.collide and right[0] > .03 else []
        def measured_check(self) -> list[dict[str, Any]]: return self.check(self.env.agent.agents[0].robot.q[0, :7], self.env.agent.agents[1].robot.q[0, :7])
    class Solver:
        def __init__(self, env: Env, index: int):
            self.base_env = env; self.env_agent = env.agent.agents[index]; self.gripper_state = 1; self.calls = 0
            self.planner = NS(user_joint_names=(*names, "f1", "f2"), move_group_joint_indices=list(range(7)))
        def move_to_pose_with_screw(self, pose: Any, *, dry_run: bool = False) -> dict[str, Any]:
            assert dry_run; self.calls += 1
            return {"status": "Success", "time": [0, .3], "position": [self.env_agent.robot.q[0, :7].copy(), pose], "duration": .3}
    def schedule(env: Env, held: bool, lift: bool = False) -> DualArmSchedule:
        left, right = Solver(env, 0), Solver(env, 1); end = np.zeros(7); end[:2] = [0, .12] if lift else [.08, -.02]
        result = plan_single_pose(left, right, end, active_key="right", gripper_command=-1 if held else 1)
        assert not env.actions and left.calls == 0 and right.calls == 1
        assert tuple(t.action_key for t in result.trajectories) == ("left", "right")
        return result
    env = Env(False); checker = Checker(env)
    reached = execute_reach(env, schedule(env, False), checker, active_key="right", stable_steps=2)
    assert reached["status"] == "passed" and checker.calls and all(a["left"][0] == .25 for a in env.actions)
    env = Env(False); checker = Checker(env); checker.target_actor = None
    assert execute_reach(env, schedule(env, False), checker, active_key="right", stable_steps=2)["status"] == "passed"
    env = Env(False); grasped = close_and_confirm(env, Checker(env), active_key="right", target_actor=env.actor)
    assert grasped["status"] == "passed" and env.agent.agents[1].robot.q[0, 7] == .012
    env = Env(); carried = execute_held_move(env, schedule(env, True), Checker(env), env.actor, active_key="right", stable_steps=2)
    assert carried["status"] == "passed" and carried["trace"][-1]["object_height_gain_m"] < 0
    env = Env(); lifted = execute_held_move(env, schedule(env, True, True), Checker(env), env.actor, active_key="right", min_height_gain_m=.10, hold_steps=20, stable_steps=2)
    assert lifted["status"] == "passed" and lifted["hold_steps_completed"] == 20
    env = Env(mode="high_raw")
    fixed = execute_held_move(env, schedule(env, True, True), Checker(env), env.actor, active_key="right", min_height_gain_m=.10, hold_steps=20, stable_steps=2)
    assert fixed["status"] == "passed" and fixed["hold_steps_completed"] == 20
    assert all(row["max_arm_velocity_rad_s"] == .06 for row in fixed["trace"])
    assert fixed["trace"][-1]["max_control_frame_joint_speed_rad_s"] == 0 and fixed["trace"][-1]["hold_joint_drift_rad"] == 0
    assert list(fixed["trace"][0]["control_frame_joint_velocity_rad_s"]) == ["left", "right"]
    env = Env(mode="joint_fd_motion")
    moving = execute_held_move(env, schedule(env, True), Checker(env), env.actor, active_key="right", stable_steps=1, settle_max_steps=1)
    assert moving["status"] == "carry_timeout" and moving["trace"][-1]["endpoint_error_rad"] < ENDPOINT
    assert moving["trace"][-1]["max_control_frame_joint_speed_rad_s"] > ARM_SPEED and moving["trace"][-1]["max_arm_velocity_rad_s"] == 0
    env = Env(mode="joint_hover_drift")
    drifting = execute_held_move(env, schedule(env, True, True), Checker(env), env.actor, active_key="right", min_height_gain_m=.10, hold_steps=20, stable_steps=2)
    assert drifting["status"] == "aborted" and drifting["reason"] == "hold_not_stable" and drifting["hold_steps_completed"] < 20
    assert drifting["trace"][-1]["hold_joint_drift_rad"] > WINDOW_JOINT and drifting["trace"][-1]["max_control_frame_joint_speed_rad_s"] < ARM_SPEED
    env = Env(False, "high_raw")
    assert execute_reach(env, schedule(env, False), Checker(env), active_key="right")["status"] == "reach_timeout"
    for mode, reason in (("slide", "position_slip"), ("grasp_loss", "grasp_lost"), ("passive_drift", "passive_hold_error")):
        env = Env(mode=mode); result = execute_held_move(env, schedule(env, True), Checker(env), env.actor, active_key="right")
        assert result["status"] == "aborted" and result["reason"] == reason and len(env.actions) <= 2
    env = Env(); result = execute_held_move(env, schedule(env, True), Checker(env, True), env.actor, active_key="right")
    assert result["status"] == "rejected" and not env.actions
    env = Env(mode="retain"); checker = Checker(env)
    result = release_and_confirm(env, checker, env.actor, active_key="right", max_steps=6)
    assert result["status"] == "release_timeout" and len(env.actions) == 6 and not checker.allowed and result["trace"][-1]["is_grasping"]
    env = Env(); result = release_and_confirm(env, Checker(env), env.actor, active_key="right")
    assert result["status"] == "passed" and len(env.actions) == 3
    predicate = lambda actor: {"inside": True, "height_bounds_m": [.01, .3]}
    for mode, status, steps in (("normal", "passed", 20), ("outside", "placement_timeout", 40), ("bounce", "passed", 31), ("drift", "placement_timeout", 40)):
        env = Env(False, mode); result = verify_placement(env, Checker(env), env.actor, active_key="right", official_predicate_key="banana_in_basket", region_predicate=predicate, max_steps=40)
        assert result["status"] == status and len(env.actions) == steps, (mode, result)
        assert result["active_key"] == "right" and result["passive_key"] == "left"
    env = Env(False); result = verify_placement(env, Checker(env), env.actor, active_key="right", official_predicate_key="banana_in_basket", region_predicate=lambda actor: False, max_steps=22)
    assert result["status"] == "placement_timeout"
    for call in (
        lambda env, checker: release_and_confirm(env, checker, None, active_key="right"),
        lambda env, checker: verify_placement(env, checker, None, active_key="right", official_predicate_key="banana_in_basket", region_predicate=predicate),
    ):
        env = Env(False); checker = Checker(env)
        try: call(env, checker)
        except ValueError: pass
        else: raise AssertionError("A missing target must be rejected before motion")
        assert not env.actions and not checker.allowed
    env = Env(False); env.truncate_at = 1
    assert verify_placement(env, Checker(env), env.actor, active_key="right", official_predicate_key="banana_in_basket", region_predicate=predicate)["status"] == "truncated" and len(env.actions) == 1
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
            "checks": ["right-only zero-step planning", "canonical physical collision order", "passive left retains nonzero entry target",
                       "open reach", "right nonzero-width contact grasp", "held carry permits descent", "20-tick lifted hover",
                       "slip and grasp loss reject", "passive drift stops", "combined candidate collision zero-step reject",
                       "retain-grip release timeout without continuation", "scoped release contacts", "actual banana predicate used",
                       "20 stable placement ticks", "outside and physical region reject", "bounce resets", "cumulative drift resets", "truncation stops",
                       "reach without a target remains supported", "release missing target rejects before motion",
                       "placement missing target rejects before motion", "held fixed position retains high raw velocity telemetry",
                       "held actual joint finite difference rejects motion", "held hover cumulative joint drift aborts",
                       "open reach retains raw joint velocity criterion"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
