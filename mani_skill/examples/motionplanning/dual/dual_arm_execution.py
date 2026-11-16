"""Synchronous dual-Panda phases with one paired environment step per tick.

Physical left/right identities are never reordered. Both arms cross each phase
barrier together; a shorter source trajectory holds its own endpoint. Carried
objects move through contact physics, never through pose assignment or a weld.
Held stability uses measured pose differences while raw velocities remain in
the trace. These sampled checks do not establish continuous collision safety.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np

from .combined_collision import validate_schedule
from .grasp_execution import _actions_valid, _budget, _error, _measure, _positive
from .lift_execution import _observe, _rotation_distance, _transform
from .trajectory_execution import as_numpy, bindings, snapshot
from .trajectory_schedule import ArmTrajectory, DualArmSchedule, build_hold_schedule, plan_pose

JOINT_STEP, TRACK, ENDPOINT, ARM_SPEED, OPEN_TOL = .02, .2, .03, .05, .002
POSITION_SLIP, ROTATION_SLIP, DROP = .005, .15, .01
OBJECT_SPEED, OBJECT_ANGULAR_SPEED = .03, .2
WINDOW_POSITION, WINDOW_ROTATION, WINDOW_JOINT = .003, .03, .003


def _pair(value: Any, name: str) -> tuple[Any, Any]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain physical left and right values")
    return tuple(value)


def _bool(value: Any) -> bool:
    array = as_numpy(value)
    if array.size != 1 or array.dtype.kind not in "biuf" or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError("Expected one finite physical boolean predicate")
    return bool(array.reshape(-1)[0])


def _canonical(env: Any) -> tuple[Any, Any]:
    raw = env.unwrapped
    agents = raw.agent.agents
    if len(agents) != 2:
        raise ValueError("Exactly two physical Panda agents are required")
    if hasattr(raw, "left_agent") and (agents[0] is not raw.left_agent or agents[1] is not raw.right_agent):
        raise ValueError("Agent sequence must retain physical left/right identity")
    keys = []
    for agent in agents:
        found = [key for key, value in raw.agent.agents_dict.items() if value is agent]
        if len(found) != 1:
            raise ValueError("Each physical arm requires one unique action key")
        keys.append(found[0])
    return bindings(env, tuple(keys))


class _Phase:
    """Canonical measurements and one joint action for a shared control tick."""

    def __init__(self, env: Any, checker: Any, name: str, *, targets: Any = None, held: bool = False):
        self.env, self.checker, self.held = env, checker, held
        self.arms = _canonical(env)
        required = ("refresh", "check", "set_target_contacts_allowed") + (("measured_check",) if held else ())
        if not all(callable(getattr(checker, name, None)) for name in required):
            raise ValueError("Incomplete dual collision-checker interface")
        self.dt = _positive(env.unwrapped.control_timestep, "control_timestep")
        self.targets = _pair(targets if targets is not None else getattr(checker, "targets", (None, None)), "targets")
        if hasattr(checker, "targets") and any(a is not b for a, b in zip(self.targets, _pair(checker.targets, "checker.targets"))):
            raise ValueError("Target actors must retain the collision checker's physical left/right pairing")
        if held and any(target is None for target in self.targets):
            raise ValueError("Both held targets must be real actors")
        references = _pair(checker.attachments, "attachments") if held else (None, None)
        self.references, self.initial_z, self.previous = [], [], []
        for arm, target, reference in zip(self.arms, self.targets, references):
            if target is None:
                self.references.append(None); self.initial_z.append(None); self.previous.append(None)
                continue
            obj = _transform(target.pose.to_transformation_matrix())
            tcp = _transform(arm.agent.tcp.pose.to_transformation_matrix())
            self.references.append(_transform(reference) if held else _transform(np.linalg.inv(tcp) @ obj))
            self.initial_z.append(float(obj[2, 3])); self.previous.append(obj.copy())
        initial = self.observe()
        self.previous_joints = {arm.key: np.asarray(initial["measurements"][arm.key]["qpos_rad"]).copy() for arm in self.arms}
        self.loss = [0, 0]
        self.report: dict[str, Any] = {
            "status": "running", "phase": name, "canonical_action_keys": [arm.key for arm in self.arms],
            "environment_steps": 0, "settling_steps": 0, "hold_steps_completed": 0,
            "trace": [], "initial_observation": initial, "initial_measurements": initial["measurements"],
            "initial_elapsed_steps": as_numpy(env.unwrapped.elapsed_steps).tolist(), "control_timestep_s": self.dt,
            "collision_scope": str(getattr(checker, "scope", "provided_dual_checker")),
            "velocity_metric": "control_frame_pose_finite_difference",
            "arm_velocity_metric": "control_frame_joint_position_finite_difference" if held else "reported_joint_velocity",
            "raw_velocity_telemetry": "retained without alteration",
            "continuous_collision_detection": False,
            "thresholds": {"joint_sample_rad": JOINT_STEP, "tracking_rad": TRACK, "endpoint_rad": ENDPOINT,
                           "arm_speed_rad_s": ARM_SPEED, "finger_open_m": OPEN_TOL,
                           "position_slip_m": POSITION_SLIP, "rotation_slip_rad": ROTATION_SLIP,
                           "drop_below_attachment_m": DROP, "grasp_loss_ticks": 2,
                           "object_speed_m_s": OBJECT_SPEED, "object_angular_speed_rad_s": OBJECT_ANGULAR_SPEED,
                           "window_position_m": WINDOW_POSITION, "window_rotation_rad": WINDOW_ROTATION,
                           "window_joint_rad": WINDOW_JOINT},
            "abort_semantics": "stop both arms; no automatic release, retreat, reset or next phase",
        }
        self.started = time.monotonic()

    def observe(self) -> dict[str, Any]:
        objects = []
        for index, (arm, target) in enumerate(zip(self.arms, self.targets)):
            if target is None:
                objects.append({"target_present": False, "is_grasping": None,
                                "tcp_T_world": _transform(arm.agent.tcp.pose.to_transformation_matrix()).tolist()})
                continue
            value = _observe((arm, self.arms[1 - index]), target, self.references[index], self.initial_z[index])
            value.pop("measurements")
            value["rotation_slip_rad"] = _rotation_distance(self.references[index][:3, :3], np.asarray(value["T_tcp_object"])[:3, :3])
            value.update(target_present=True, action_key=arm.key, object_index=index)
            objects.append(value)
        return {"measurements": _measure(self.arms), "objects": objects}

    def opened(self, arm: Any, row: dict[str, Any]) -> bool:
        fingers = [index for index in range(9) if index not in arm.indices]
        upper = as_numpy(arm.agent.robot.get_qlimits())[0, fingers, 1]
        return bool(np.allclose(row["measurements"][arm.key]["finger_qpos_m"], upper, rtol=0, atol=OPEN_TOL))

    def hold(self, grip: int) -> dict[str, np.ndarray]:
        return {arm.key: np.r_[self.report["initial_measurements"][arm.key]["qpos_rad"], grip] for arm in self.arms}

    def collisions(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        self.checker.refresh()
        if self.held:
            return self.checker.measured_check()
        return self.checker.check(*(np.asarray(row["measurements"][arm.key]["qpos_rad"]) for arm in self.arms))

    def preflight(self) -> bool:
        try:
            collisions = self.collisions(self.report["initial_observation"])
        except Exception as exc:
            _error(self.report, "collision_preflight_failed", exc); self.report["status"] = "rejected"
            return False
        self.report["initial_collisions"] = collisions
        if collisions:
            self.report.update(status="rejected", reason="initial_collision")
            return False
        return True

    def tick(self, action: dict[str, Any], label: str) -> dict[str, Any] | None:
        _actions_valid(self.env, self.arms, action)
        actual = {arm.key: np.asarray(action[arm.key], dtype=np.float32).copy() for arm in self.arms}
        before = as_numpy(self.env.unwrapped.elapsed_steps).copy()
        row: dict[str, Any] = {"phase": label, "step": self.report["environment_steps"] + 1,
                               "action": {key: value.tolist() for key, value in actual.items()}}
        try:
            _, _, terminated, truncated, _ = self.env.step(actual)
            self.report["environment_steps"] += 1
            self.report["settling_steps"] += int(label == "settling")
            after = as_numpy(self.env.unwrapped.elapsed_steps)
            if not np.array_equal(after, before + 1):
                raise RuntimeError("One paired action must advance exactly one environment control step")
            row.update(time_s=self.report["environment_steps"] * self.dt,
                       environment_step=int(after.reshape(-1)[0]), terminated=_bool(terminated), truncated=_bool(truncated))
            row.update(self.observe())
            joints = {arm.key: np.asarray(row["measurements"][arm.key]["qpos_rad"]).copy() for arm in self.arms}
            velocity = {arm.key: (joints[arm.key] - self.previous_joints[arm.key]) / self.dt for arm in self.arms}
            row["control_frame_joint_velocity_rad_s"] = {key: value.tolist() for key, value in velocity.items()}
            row["max_control_frame_joint_speed_rad_s"] = max(float(np.max(np.abs(value))) for value in velocity.values())
            self.previous_joints = joints
            for index, obj in enumerate(row["objects"]):
                if not obj["target_present"]:
                    continue
                pose = _transform(obj["object_T_world"])
                obj["control_frame_linear_speed_m_s"] = float(np.linalg.norm(pose[:3, 3] - self.previous[index][:3, 3]) / self.dt)
                obj["control_frame_angular_speed_rad_s"] = _rotation_distance(self.previous[index][:3, :3], pose[:3, :3]) / self.dt
                self.previous[index] = pose
            row["collisions"] = self.collisions(row)
            row["tracking_error_rad"] = max(float(np.max(np.abs(joints[arm.key] - actual[arm.key][:7]))) for arm in self.arms)
            row["max_arm_velocity_rad_s"] = max(float(np.max(np.abs(row["measurements"][arm.key]["qvel_rad_s"]))) for arm in self.arms)
        except Exception as exc:
            row["observed_elapsed_steps_after_error"] = as_numpy(self.env.unwrapped.elapsed_steps).tolist()
            row["elapsed_step_delta_on_error"] = (as_numpy(self.env.unwrapped.elapsed_steps) - before).tolist()
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.report["trace"].append(row); _error(self.report, "step_or_measurement_failed", exc)
            return None
        self.report["trace"].append(row)
        if row["collisions"]:
            self.report.update(status="aborted", reason="measured_collision")
        elif row["tracking_error_rad"] > TRACK:
            self.report.update(status="aborted", reason="tracking_error")
        elif row["terminated"] or row["truncated"]:
            self.report.update(status="terminated" if row["terminated"] else "truncated", reason="environment_done")
        return row

    def monitor(self, row: dict[str, Any], *, hovering: bool = False) -> bool:
        for index, obj in enumerate(row["objects"]):
            expected = np.asarray(obj["tcp_T_world"]) @ self.references[index]
            obj["drop_below_predicted_attachment_m"] = max(0., float(expected[2, 3] - obj["object_height_m"]))
            self.loss[index] = 0 if obj["is_grasping"] else self.loss[index] + 1
            obj["consecutive_grasp_loss_ticks"] = self.loss[index]
            reason = None
            if obj["drop_below_predicted_attachment_m"] > DROP: reason = "object_dropped"
            elif obj["position_slip_m"] > POSITION_SLIP: reason = "position_slip"
            elif obj["rotation_slip_rad"] > ROTATION_SLIP: reason = "rotation_slip"
            elif self.loss[index] >= 2 or (hovering and not obj["is_grasping"]): reason = "grasp_lost"
            if reason:
                self.report.update(status="aborted", reason=reason, object_index=index)
                return False
        return True

    def finish(self) -> dict[str, Any]:
        self.report["wall_seconds"] = time.monotonic() - self.started
        return self.report


def plan_dual_pose(left_solver: Any, right_solver: Any, target_poses: tuple[Any | None, Any | None], *,
                   gripper_commands: tuple[int, int] = (1, 1)) -> DualArmSchedule:
    """Dry-run both native plans at one state; None holds that measured arm."""
    raw = left_solver.base_env
    if raw is not right_solver.base_env:
        raise ValueError("Both solvers must share one physical environment")
    arms = _canonical(raw)
    poses, grips = _pair(target_poses, "target_poses"), _pair(gripper_commands, "gripper_commands")
    if any(solver.env_agent is not arm.agent for solver, arm in zip((left_solver, right_solver), arms)):
        raise ValueError("Solvers must retain physical left, right order")
    if any(grip not in (-1, 1) for grip in grips):
        raise ValueError("Each phase gripper command must be OPEN or CLOSED")
    before, trajectories = snapshot(raw, arms), []
    for solver, arm, pose, grip in zip((left_solver, right_solver), arms, poses, grips):
        solver.gripper_state = grip
        trajectory = (ArmTrajectory(arm.key, arm.names, [0.], [arm.arm_qpos()], grip) if pose is None else
                      plan_pose(solver, pose, action_key=arm.key, controller_joint_names=arm.names))
        if not all(np.array_equal(a, b) for a, b in zip(before, snapshot(raw, arms))):
            raise RuntimeError("Dry-run planning changed the physical environment")
        if not np.allclose(trajectory.positions_rad[0], arm.arm_qpos(), rtol=0, atol=1e-6):
            raise ValueError("Planner source does not start at the measured joints")
        trajectories.append(trajectory)
    return build_hold_schedule(*trajectories, control_timestep_s=raw.control_timestep)


def _scheduled(env: Any, schedule: DualArmSchedule, checker: Any, *, held: bool, targets: Any,
               minimum: Any, hold_steps: int, settle_max_steps: int, stable_steps: int) -> dict[str, Any]:
    settling, required = _budget(settle_max_steps, "settle_max_steps", zero=True), _budget(stable_steps, "stable_steps")
    hover = _budget(hold_steps, "hold_steps", zero=True)
    minima = tuple(None if value is None else _positive(value, "min_height_gain_m") for value in _pair(minimum, "min_height_gain_m"))
    phase = _Phase(env, checker, "dual_held_move" if held else "dual_reach", targets=targets, held=held)
    initial = phase.report["initial_observation"]
    if tuple(source.action_key for source in schedule.trajectories) != tuple(arm.key for arm in phase.arms):
        raise ValueError("Schedule must retain physical left/right order")
    if not np.isclose(schedule.control_timestep_s, phase.dt, rtol=0, atol=1e-10):
        raise ValueError("Schedule and environment control clocks differ")
    actions, final = list(schedule.iter_actions()), {}
    for arm, source in zip(phase.arms, schedule.trajectories):
        grip = -1 if held else 1
        if source.joint_names != arm.names or source.gripper_command != grip:
            raise ValueError("Unexpected controller joint order or phase gripper command")
        final[arm.key] = np.r_[source.positions_rad[-1], grip]
        if not held and not phase.opened(arm, initial):
            raise ValueError("Both hands must physically be open before reaching")
    for action in [*actions, final]:
        _actions_valid(env, phase.arms, action)
    phase.report.update(planned_control_steps=len(actions), settle_max_steps=settling,
                        stable_steps_required=required, min_height_gain_m=list(minima), hold_steps_required=hover)
    if held and any(not obj["is_grasping"] or obj["position_slip_m"] > POSITION_SLIP or obj["rotation_slip_rad"] > ROTATION_SLIP for obj in initial["objects"]):
        phase.report.update(status="rejected", reason="initial_grasp_or_attachment_invalid")
        return phase.finish()
    if not phase.preflight():
        return phase.finish()
    phase.report["validation"] = validate_schedule(schedule, *(np.asarray(initial["measurements"][arm.key]["qpos_rad"]) for arm in phase.arms), checker, max_joint_step_rad=JOINT_STEP)
    if phase.report["validation"]["status"] != "passed":
        phase.report.update(status="rejected", reason="candidate_collision")
        return phase.finish()

    def stable(row: dict[str, Any]) -> bool:
        speed = row["max_control_frame_joint_speed_rad_s"] if held else row["max_arm_velocity_rad_s"]
        if row["endpoint_error_rad"] > ENDPOINT or speed > ARM_SPEED:
            return False
        return not held or all(obj["is_grasping"] and obj["control_frame_linear_speed_m_s"] <= OBJECT_SPEED
                               and obj["control_frame_angular_speed_rad_s"] <= OBJECT_ANGULAR_SPEED
                               and (minimum is None or obj["object_height_gain_m"] >= minimum)
                               for obj, minimum in zip(row["objects"], minima))

    count, ready = 0, False
    for index in range(len(actions) + settling):
        label = "moving" if index < len(actions) else "settling"
        row = phase.tick(actions[index] if label == "moving" else final, label)
        if row is None or phase.report["status"] != "running": break
        row["endpoint_error_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][arm.key]["qpos_rad"]) - final[arm.key][:7]))) for arm in phase.arms)
        if held and not phase.monitor(row): break
        count = count + 1 if stable(row) else 0
        row["stable_steps"] = count
        if index + 1 >= len(actions) and count >= required:
            ready = True; break
    if phase.report["status"] == "running" and not ready:
        phase.report.update(status="held_move_timeout" if held else "reach_timeout", reason="both_endpoints_height_or_velocity_not_stable")
    if ready and phase.report["status"] == "running":
        if hover:
            references = [_transform(obj["object_T_world"]) for obj in phase.report["trace"][-1]["objects"]]
            joints = {arm.key: np.asarray(phase.report["trace"][-1]["measurements"][arm.key]["qpos_rad"]).copy() for arm in phase.arms}
            phase.report["hold_reference_T_world"] = [reference.tolist() for reference in references]
            phase.report["hold_reference_joint_qpos_rad"] = {key: value.tolist() for key, value in joints.items()}
        for _ in range(hover):
            row = phase.tick(final, "hover")
            if row is None or phase.report["status"] != "running": break
            row["endpoint_error_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][arm.key]["qpos_rad"]) - final[arm.key][:7]))) for arm in phase.arms)
            if not phase.monitor(row, hovering=True): break
            for obj, reference in zip(row["objects"], references):
                pose = _transform(obj["object_T_world"])
                obj["hold_position_drift_m"] = float(np.linalg.norm(pose[:3, 3] - reference[:3, 3]))
                obj["hold_rotation_drift_rad"] = _rotation_distance(reference[:3, :3], pose[:3, :3])
            row["hold_joint_drift_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][arm.key]["qpos_rad"]) - joints[arm.key]))) for arm in phase.arms)
            if (not stable(row) or row["hold_joint_drift_rad"] > WINDOW_JOINT or
                    any(obj["hold_position_drift_m"] > WINDOW_POSITION or obj["hold_rotation_drift_rad"] > WINDOW_ROTATION for obj in row["objects"])):
                phase.report.update(status="aborted", reason="hold_not_stable"); break
            phase.report["hold_steps_completed"] += 1
        if phase.report["status"] == "running": phase.report["status"] = "passed"
    phase.report["stable_steps_observed"] = count
    return phase.finish()


def execute_dual_reach(env: Any, schedule: DualArmSchedule, checker: Any, *,
                       allow_target_contacts: tuple[bool, bool] = (False, False),
                       settle_max_steps: int = 15, stable_steps: int = 3) -> dict[str, Any]:
    allowed = _pair(allow_target_contacts, "allow_target_contacts")
    if any(type(value) is not bool for value in allowed):
        raise ValueError("Target-contact permissions must be explicit booleans")
    checker.set_target_contacts_allowed(allowed)
    try:
        return _scheduled(env, schedule, checker, held=False, targets=None, minimum=(None, None), hold_steps=0,
                          settle_max_steps=settle_max_steps, stable_steps=stable_steps)
    finally:
        checker.set_target_contacts_allowed((False, False))


def close_dual_and_confirm(env: Any, checker: Any, targets: tuple[Any, Any], *,
                           max_steps: int = 40, stable_steps: int = 5) -> dict[str, Any]:
    targets = _pair(targets, "targets")
    if any(target is None for target in targets): raise ValueError("Both physical targets are required")
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    if budget < required: raise ValueError("Grasp budget cannot be smaller than the required stable interval")
    phase = _Phase(env, checker, "dual_close", targets=targets)
    action = phase.hold(-1); _actions_valid(env, phase.arms, action)
    previous = [np.asarray(phase.report["initial_measurements"][arm.key]["finger_qpos_m"]) for arm in phase.arms]
    counts = [0, 0]
    phase.report.update(max_steps=budget, stable_steps_required=required)
    checker.set_target_contacts_allowed((True, True))
    try:
        if not phase.preflight(): return phase.finish()
        for _ in range(budget):
            row = phase.tick(action, "closing")
            if row is None or phase.report["status"] != "running": break
            if row["tracking_error_rad"] > ENDPOINT:
                phase.report.update(status="aborted", reason="arm_hold_error"); break
            for index, arm in enumerate(phase.arms):
                fingers = np.asarray(row["measurements"][arm.key]["finger_qpos_m"])
                change = float(np.max(np.abs(fingers - previous[index])))
                previous[index] = fingers.copy()
                counts[index] = counts[index] + 1 if row["objects"][index]["is_grasping"] and change <= .0002 else 0
                row["objects"][index].update(finger_change_m=change, stable_steps=counts[index])
            row["stable_steps"] = min(counts)
            if min(counts) >= required:
                phase.report["status"] = "passed"; break
        if phase.report["status"] == "running":
            phase.report.update(status="grasp_timeout", reason="both_contact_grasps_not_stable")
        phase.report["stable_steps_observed"] = list(counts)
        return phase.finish()
    finally:
        checker.set_target_contacts_allowed((False, False))


def execute_dual_held_move(env: Any, schedule: DualArmSchedule, checker: Any, targets: tuple[Any, Any], *,
                           min_height_gain_m: tuple[float | None, float | None] = (None, None),
                           hold_steps: int = 0, settle_max_steps: int = 15, stable_steps: int = 3) -> dict[str, Any]:
    targets = _pair(targets, "targets")
    if any(target is None for target in targets): raise ValueError("Both physical held targets are required")
    return _scheduled(env, schedule, checker, held=True, targets=targets, minimum=min_height_gain_m,
                      hold_steps=hold_steps, settle_max_steps=settle_max_steps, stable_steps=stable_steps)


def release_dual_and_confirm(env: Any, checker: Any, targets: tuple[Any, Any], *,
                             max_steps: int = 30, stable_steps: int = 3) -> dict[str, Any]:
    targets = _pair(targets, "targets")
    if any(target is None for target in targets): raise ValueError("Both physical release targets are required")
    if getattr(checker, "attached", False): raise ValueError("Detach collision predictions before physical release")
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    phase = _Phase(env, checker, "dual_release", targets=targets)
    action = phase.hold(1); _actions_valid(env, phase.arms, action)
    count = 0
    phase.report.update(max_steps=budget, stable_steps_required=required)
    checker.set_target_contacts_allowed((True, True))
    try:
        if not phase.preflight(): return phase.finish()
        for _ in range(budget):
            row = phase.tick(action, "releasing")
            if row is None or phase.report["status"] != "running": break
            row["fingers_open"] = [phase.opened(arm, row) for arm in phase.arms]
            if row["tracking_error_rad"] > ENDPOINT:
                phase.report.update(status="aborted", reason="arm_hold_error"); break
            valid = all(row["fingers_open"]) and all(not obj["is_grasping"] for obj in row["objects"])
            count = count + 1 if valid else 0
            row["stable_steps"] = count
            if count >= required:
                phase.report["status"] = "passed"; break
        if phase.report["status"] == "running":
            phase.report.update(status="release_timeout", reason="both_hands_not_open_or_grasp_persists")
        phase.report["stable_steps_observed"] = count
        return phase.finish()
    finally:
        checker.set_target_contacts_allowed((False, False))


def _regions(env: Any, targets: Any, keys: Any, predicates: Any) -> list[dict[str, Any]]:
    evaluation = env.unwrapped.evaluate()
    regions = []
    for target, key, predicate in zip(targets, keys, predicates):
        if key not in evaluation: raise ValueError(f"Missing official placement predicate: {key}")
        value = predicate(target)
        details = dict(value) if isinstance(value, dict) else {"inside": value}
        if "inside" not in details: raise ValueError("Each physical region predicate requires an inside value")
        details["inside"] = _bool(details["inside"])
        json.dumps(details, allow_nan=False)
        regions.append({"official_region": _bool(evaluation[key]), "physical_region": details})
    return regions


def verify_dual_placement(env: Any, checker: Any, targets: tuple[Any, Any], *,
                          official_predicate_keys: tuple[str, str] = ("apple_in_basket", "banana_in_basket"),
                          region_predicates: tuple[Callable[[Any], Any], Callable[[Any], Any]],
                          max_steps: int = 60, stable_steps: int = 20) -> dict[str, Any]:
    targets, keys, predicates = _pair(targets, "targets"), _pair(official_predicate_keys, "official_predicate_keys"), _pair(region_predicates, "region_predicates")
    if any(target is None for target in targets) or any(not callable(predicate) for predicate in predicates):
        raise ValueError("Both actual targets and physical region callbacks are required")
    if any(not isinstance(key, str) or not key for key in keys): raise ValueError("Explicit official predicate keys are required")
    if getattr(checker, "attached", False): raise ValueError("Placement verification requires detached collision predictions")
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    phase = _Phase(env, checker, "dual_placement_verification", targets=targets)
    if not all(phase.opened(arm, phase.report["initial_observation"]) for arm in phase.arms):
        raise ValueError("Both hands must already physically be open")
    phase.report.update(initial_regions=_regions(env, targets, keys, predicates), max_steps=budget,
                        stable_steps_required=required, official_predicate_keys=list(keys))
    action = phase.hold(1); _actions_valid(env, phase.arms, action)
    checker.set_target_contacts_allowed((False, False))
    if not phase.preflight(): return phase.finish()
    count, window = 0, None
    for _ in range(budget):
        row = phase.tick(action, "verifying_placement")
        if row is None or phase.report["status"] != "running": break
        try:
            regions = _regions(env, targets, keys, predicates)
            for obj, region in zip(row["objects"], regions): obj.update(region)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"; _error(phase.report, "region_measurement_failed", exc); break
        row["fingers_open"] = [phase.opened(arm, row) for arm in phase.arms]
        if row["tracking_error_rad"] > ENDPOINT:
            phase.report.update(status="aborted", reason="arm_hold_error"); break
        valid = (all(row["fingers_open"]) and row["max_arm_velocity_rad_s"] <= ARM_SPEED
                 and all(obj["official_region"] and obj["physical_region"]["inside"] and not obj["is_grasping"]
                         and obj["control_frame_linear_speed_m_s"] <= OBJECT_SPEED
                         and obj["control_frame_angular_speed_rad_s"] <= OBJECT_ANGULAR_SPEED for obj in row["objects"]))
        poses = [_transform(obj["object_T_world"]) for obj in row["objects"]]
        if not valid:
            count, window = 0, None
            row["stability_reset_reason"] = "either_region_release_or_velocity"
        else:
            if window is None: window = [pose.copy() for pose in poses]
            drift = []
            for obj, pose, reference in zip(row["objects"], poses, window):
                dp = float(np.linalg.norm(pose[:3, 3] - reference[:3, 3])); dr = _rotation_distance(reference[:3, :3], pose[:3, :3])
                obj.update(window_position_drift_m=dp, window_rotation_drift_rad=dr)
                drift.append(dp > WINDOW_POSITION or dr > WINDOW_ROTATION)
            if any(drift):
                count, window = 1, [pose.copy() for pose in poses]
                row["stability_reset_reason"] = "either_cumulative_window_drift"
            else: count += 1
        row["stable_steps"] = count
        if count >= required:
            phase.report["status"] = "passed"; break
    if phase.report["status"] == "running":
        phase.report.update(status="placement_timeout", reason="both_placements_not_continuously_stable")
    phase.report["stable_steps_observed"] = count
    return phase.finish()


def self_test() -> dict[str, Any]:
    """Analytical CPU fixtures use actual schedule/controller contracts, no physics."""
    from types import SimpleNamespace as NS
    from gymnasium import spaces

    names = tuple(f"joint{index}" for index in range(7))

    class Pose:
        def __init__(self, matrix: Any = None): self.matrix = np.eye(4) if matrix is None else np.array(matrix, dtype=float, copy=True)
        def to_transformation_matrix(self) -> np.ndarray: return self.matrix[None].copy()
        @property
        def raw_pose(self) -> np.ndarray: return np.r_[self.matrix[:3, 3], 1., 0., 0., 0.][None]

    class Robot:
        def __init__(self, holding: bool):
            self.q = np.array([[0.] * 7 + [.024 if holding else .04] * 2]); self.v = np.zeros((1, 9)); self.pose = Pose()
        def get_qpos(self) -> np.ndarray: return self.q.copy()
        def get_qvel(self) -> np.ndarray: return self.v.copy()
        def get_qlimits(self) -> np.ndarray: return np.array([[[-3., 3.]] * 7 + [[0., .04]] * 2])
        def get_active_joints(self) -> list[Any]: return [NS(get_name=lambda name=name: name) for name in (*names, "f1", "f2")]

    class Actor:
        def __init__(self): self.pose = Pose(); self.linear = np.zeros((1, 3)); self.angular = np.zeros((1, 3))
        def get_linear_velocity(self) -> np.ndarray: return self.linear.copy()
        def get_angular_velocity(self) -> np.ndarray: return self.angular.copy()

    class Env:
        def __init__(self, holding: bool = True, mode: str = "normal"):
            self.unwrapped = self; self.num_envs = 1; self.control_timestep = .05; self.elapsed_steps = np.array([0])
            self.mode = mode; self.actions = []; self.targets = (Actor(), Actor()); self.grasp = [holding, holding]
            self.agent = NS(agents=[], agents_dict={})
            for index, key in enumerate(("left", "right")):
                robot = Robot(holding)
                controller = NS(joints=robot.get_active_joints()[:7], config=NS(normalize_action=False, use_delta=False, joint_names=names))
                agent = NS(robot=robot, control_mode="pd_joint_pos", controller=NS(controllers={"arm": controller}), tcp=NS(pose=Pose()))
                agent.is_grasping = lambda target, index=index: np.array([self.grasp[index]])
                self.agent.agents.append(agent); self.agent.agents_dict[key] = agent
            self.left_agent, self.right_agent = self.agent.agents
            box = spaces.Box(np.array([-3.] * 7 + [-1.], dtype=np.float32), np.array([3.] * 7 + [1.], dtype=np.float32))
            self.agent.single_action_space = spaces.Dict({"left": box, "right": box})
            self.update()

        def update(self) -> None:
            for index, agent in enumerate(self.agent.agents):
                matrix = np.eye(4); matrix[:3, 3] = agent.robot.q[0, :3] * [1., 1., 1.]
                matrix[0, 3] += -.3 if index == 0 else .3
                agent.tcp.pose.matrix = matrix
                if self.grasp[index]: self.targets[index].pose.matrix = matrix @ attachment(index)

        def step(self, action: dict[str, Any]) -> tuple[Any, ...]:
            assert list(action) == ["left", "right"] and set(action) == set(self.agent.agents_dict)
            self.actions.append({key: np.array(value, copy=True) for key, value in action.items()})
            self.elapsed_steps += 1; tick = len(self.actions)
            for index, (key, agent) in enumerate(self.agent.agents_dict.items()):
                agent.robot.q[0, :7] = action[key][:7]
                closed = action[key][7] == -1 or (self.mode == "retain_right" and index == 1)
                self.grasp[index] = closed and not (self.mode == "empty_right" and index == 1)
                agent.robot.q[0, 7:] = .024 if closed else .04
                if self.mode == "delayed_right" and index == 1 and tick <= 3: agent.robot.q[0, 7:] += .001 * (tick % 2)
                if self.mode == "joint_fd": agent.robot.q[0, 0] += .004 * tick
                if self.mode == "hover_drift": agent.robot.q[0, 0] += .0002 * tick
                agent.robot.v[:] = 1.6 if self.mode == "high_raw" else 0.
            self.update()
            if self.mode == "slip_right": self.targets[1].pose.matrix[0, 3] += .006
            if self.mode == "drop_right": self.targets[1].pose.matrix[2, 3] -= .012
            if self.mode == "rotation_right":
                angle = .2; self.targets[1].pose.matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            if self.mode == "placement_drift": self.targets[1].pose.matrix[0, 3] += .0002
            for target in self.targets:
                target.angular[:] = [0., 0., 1.6] if self.mode == "high_raw" else 0.
            return None, 0., np.array([self.mode == "terminated"]), np.array([self.mode == "truncated"]), {}

        def evaluate(self) -> dict[str, Any]:
            inside = not (self.mode == "outside_right" or (self.mode == "bounce_right" and len(self.actions) == 11))
            return {"apple_in_basket": np.array([True]), "banana_in_basket": np.array([inside])}

    def attachment(index: int) -> np.ndarray:
        value = np.eye(4); value[2, 3] = .02 + .01 * index
        return value

    class Checker:
        def __init__(self, env: Env, *, held: bool = True, mode: str = "normal"):
            self.env, self.targets, self.attached, self.mode = env, env.targets, held, mode
            self.scope = "analytical_fixture"; self.allowed = (False, False); self.candidates = []
        @property
        def attachments(self) -> tuple[np.ndarray, np.ndarray]:
            if not self.attached: raise ValueError("not attached")
            return attachment(0), attachment(1)
        def set_target_contacts_allowed(self, value: Any) -> None: self.allowed = value
        def refresh(self) -> None: pass
        def check(self, left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
            self.candidates.append((left.copy(), right.copy()))
            return [{"fixture": "candidate"}] if self.mode == "candidate" else []
        def measured_check(self) -> list[dict[str, Any]]:
            return [{"fixture": "live"}] if self.mode == "measured" and self.env.actions else []

    class Solver:
        """Analytical native-style planner; the real endpoint adapter still runs."""
        def __init__(self, env: Env, index: int):
            self.base_env = env; self.env_agent = env.agent.agents[index]; self.gripper_state = 1; self.index = index; self.calls = 0
            self.robot = self.env_agent.robot
            self.use_point_cloud = False
            self.last_native_result = None
            self.topp_calls = 0
            self.base_xyz = np.array([-.3 if index == 0 else .3, 0., 0.])
            self.planner = NS(user_joint_names=(*names, "f1", "f2"), move_group_joint_indices=list(range(7)),
                move_group="tcp", link_name_2_idx={"tcp": 0}, planning_world=NS(use_attach=False),
                transform_goal_to_wrt_base=self.goal_in_base, TOPP=self.topp, plan_screw=self.screw,
                pinocchio_model=NS(compute_forward_kinematics=self.fk, get_link_pose=lambda _: self.fk_pose.copy()))

        def goal_in_base(self, value: Any) -> np.ndarray:
            goal = np.array(value, dtype=float, copy=True)
            assert goal.shape == (7,)
            goal[:3] -= self.base_xyz
            return goal

        def fk(self, full_qpos: Any) -> None:
            full = np.asarray(full_qpos)
            assert full.shape == (9,) and np.array_equal(full[7:], self.robot.q[0, 7:])
            self.fk_pose = np.r_[full[:3], 1., 0., 0., 0.]

        def topp(self, path: Any, step: float, verbose: bool = False) -> Any:
            points = np.array(path, dtype=float, copy=True)
            assert points.shape == (2, 7) and step == self.base_env.control_timestep
            self.topp_calls += 1
            duration = .1 if self.index == 0 else .2
            velocity = np.repeat(((points[1] - points[0]) / duration)[None], 2, axis=0)
            return np.array([0., duration]), points, velocity, np.zeros_like(points), duration

        def screw(self, value: Any, full_qpos: Any, **kwargs: Any) -> dict[str, Any]:
            full = np.array(full_qpos, dtype=float, copy=True)
            assert full.shape == (9,) and np.array_equal(full[7:], self.robot.q[0, 7:])
            goal = self.goal_in_base(value)
            assert np.array_equal(goal[3:], [1., 0., 0., 0.])
            start = full[:7].copy(); end = start.copy(); end[:3] = goal[:3]
            raw = self.planner.TOPP(np.vstack((start, end)), self.base_env.control_timestep)
            return {"status": "Success", **dict(zip(("time", "position", "velocity", "acceleration", "duration"), raw))}

        def move_to_pose_with_screw(self, value: Any, *, dry_run: bool) -> dict[str, Any]:
            assert dry_run and not self.base_env.actions
            self.calls += 1
            result = self.screw(value, self.robot.get_qpos()[0], time_step=self.base_env.control_timestep)
            self.last_native_result = result
            return result

    def target_pose(env: Env, index: int, delta: Any) -> np.ndarray:
        # Preserve the old fixture's joint-translation increments as XYZ/WXYZ.
        xyz = env.agent.agents[index].robot.q[0, :3].copy()
        delta = np.asarray(delta)
        xyz[:len(delta)] += delta
        xyz[0] += -.3 if index == 0 else .3
        return np.r_[xyz, 1., 0., 0., 0.]

    def schedule(env: Env, *, held: bool = True, lift: bool = False) -> DualArmSchedule:
        deltas = ([0., 0., .12], [0., 0., .12]) if lift else ([.04], [.08])
        poses = tuple(target_pose(env, index, delta) for index, delta in enumerate(deltas))
        return plan_dual_pose(Solver(env, 0), Solver(env, 1), poses, gripper_commands=(-1, -1) if held else (1, 1))

    def reject(call: Callable[[], Any]) -> None:
        try: call()
        except (ValueError, TypeError): return
        raise AssertionError("Invalid input was accepted")

    checks = []
    env = Env(False); solvers = (Solver(env, 0), Solver(env, 1)); planned = plan_dual_pose(*solvers, (target_pose(env, 0, [.04]), target_pose(env, 1, [.08])))
    assert not env.actions and all(solver.calls == 1 for solver in solvers)
    assert all(solver.topp_calls == 1 and solver.last_endpoint_audit["status"] == "passed"
               and solver.last_endpoint_audit["corrections"] == 0 for solver in solvers)
    assert np.allclose(planned.left_commands[2:, 0], .04) and planned.num_steps == 4
    reject(lambda: plan_dual_pose(solvers[1], solvers[0], (None, None)))
    held = plan_dual_pose(*solvers, (None, target_pose(env, 1, [.08]))); assert solvers[0].calls == 1 and np.all(held.left_commands[:, :7] == 0)
    checks.append("dry-run both planners, canonical identity, unequal endpoint hold and None hold")
    reached = execute_dual_reach(env, planned, Checker(env, held=False), stable_steps=2)
    assert reached["status"] == "passed" and len(env.actions) == reached["environment_steps"]
    assert all(list(row["measurements"]) == ["left", "right"] and len(row["objects"]) == 2 for row in reached["trace"])
    checks.append("one paired step per control tick and canonical object/measurement arrays")
    env = Env(False, "delayed_right"); checker = Checker(env, held=False)
    closed = close_dual_and_confirm(env, checker, env.targets)
    assert closed["status"] == "passed" and len(env.actions) >= 8 and min(closed["stable_steps_observed"]) >= 5
    assert checker.allowed == (False, False)
    checks.append("both contact grasps and per-arm finger stability phase barrier")
    env = Env(False, "empty_right"); failed = close_dual_and_confirm(env, Checker(env, held=False), env.targets, max_steps=8)
    assert failed["status"] == "grasp_timeout" and len(env.actions) == 8
    checks.append("one empty grasp cannot pass and bounded timeout stops both")
    env = Env(); lifted = execute_dual_held_move(env, schedule(env, lift=True), Checker(env), env.targets, min_height_gain_m=(.10, .10), hold_steps=20)
    assert lifted["status"] == "passed" and lifted["hold_steps_completed"] == 20 and lifted["settling_steps"] >= 3
    assert all(obj["object_height_gain_m"] >= .10 for obj in lifted["trace"][-1]["objects"])
    assert len(env.actions) == lifted["environment_steps"] and lifted["trace"][-1]["step"] == len(env.actions)
    checks.append("both actual height gains, explicit settling, 20 shared hover ticks")
    env = Env(mode="high_raw"); result = execute_dual_held_move(env, schedule(env, lift=True), Checker(env), env.targets, min_height_gain_m=(.10, .10), hold_steps=20)
    assert result["status"] == "passed" and result["trace"][-1]["max_arm_velocity_rad_s"] == 1.6
    assert result["trace"][-1]["max_control_frame_joint_speed_rad_s"] == 0
    assert all(obj["object_angular_speed_rad_s"] == 1.6 for obj in result["trace"][-1]["objects"])
    checks.append("held pose FD preserves high raw joint/object velocity telemetry")
    env = Env(mode="joint_fd"); result = execute_dual_held_move(env, schedule(env), Checker(env), env.targets, settle_max_steps=3)
    assert result["status"] == "held_move_timeout" and result["trace"][-1]["max_arm_velocity_rad_s"] == 0
    assert result["trace"][-1]["max_control_frame_joint_speed_rad_s"] > ARM_SPEED
    env = Env(); asymmetric = plan_dual_pose(Solver(env, 0), Solver(env, 1), (target_pose(env, 0, [0., 0., .12]), target_pose(env, 1, [0., 0., .08])), gripper_commands=(-1, -1))
    result = execute_dual_held_move(env, asymmetric, Checker(env), env.targets, min_height_gain_m=(.10, .10), settle_max_steps=3)
    assert result["status"] == "held_move_timeout" and result["trace"][-1]["objects"][0]["object_height_gain_m"] >= .10
    assert result["trace"][-1]["objects"][1]["object_height_gain_m"] < .10
    checks.append("measured joint FD and the second object's height independently block readiness")
    for mode, reason in (("slip_right", "position_slip"), ("drop_right", "object_dropped"), ("rotation_right", "rotation_slip"), ("empty_right", "grasp_lost")):
        env = Env(mode=mode)
        candidate = (plan_dual_pose(Solver(env, 0), Solver(env, 1), (None, None), gripper_commands=(-1, -1))
                     if mode == "empty_right" else schedule(env))
        result = execute_dual_held_move(env, candidate, Checker(env), env.targets)
        assert result["status"] == "aborted" and result.get("reason") == reason and result.get("object_index") == 1, (mode, result["status"], result.get("reason"))
        assert len(env.actions) <= 2
    checks.append("either object slip, rotation, drop or two-tick grasp loss aborts both")
    env = Env(mode="hover_drift"); result = execute_dual_held_move(env, schedule(env, lift=True), Checker(env), env.targets, hold_steps=20)
    assert result["status"] == "aborted" and result["reason"] == "hold_not_stable"
    checks.append("cumulative hover joint/object drift cannot hide under low FD speed")
    env = Env(); result = execute_dual_held_move(env, schedule(env), Checker(env, mode="candidate"), env.targets)
    assert result["status"] == "rejected" and not env.actions
    env = Env(); result = execute_dual_held_move(env, schedule(env), Checker(env, mode="measured"), env.targets)
    assert result["reason"] == "measured_collision" and len(env.actions) == 1
    checks.append("candidate collision rejects before motion and live collision stops after one tick")
    for mode in ("terminated", "truncated"):
        env = Env(mode=mode); result = execute_dual_held_move(env, schedule(env), Checker(env), env.targets)
        assert result["status"] == mode and len(env.actions) == 1
    checks.append("received termination and truncation stop without masking")
    env = Env(); checker = Checker(env, held=False); released = release_dual_and_confirm(env, checker, env.targets)
    assert released["status"] == "passed" and len(env.actions) == 3 and all(released["trace"][-1]["fingers_open"])
    env = Env(mode="retain_right"); checker = Checker(env, held=False); retained = release_dual_and_confirm(env, checker, env.targets, max_steps=7)
    assert retained["status"] == "release_timeout" and len(env.actions) == 7
    assert not retained["trace"][-1]["objects"][0]["is_grasping"] and retained["trace"][-1]["objects"][1]["is_grasping"]
    checks.append("left releases while right retain-grip times out without continuation")
    predicates = (lambda target: {"inside": True, "height_bounds_m": [0., .3]},) * 2
    env = Env(False); placed = verify_dual_placement(env, Checker(env, held=False), env.targets, region_predicates=predicates)
    assert placed["status"] == "passed" and len(env.actions) == 20
    env = Env(False); result = verify_dual_placement(env, Checker(env, held=False), env.targets,
        region_predicates=(predicates[0], lambda actor: {"inside": False}), max_steps=21)
    assert result["status"] == "placement_timeout" and len(env.actions) == 21
    for mode in ("outside_right", "placement_drift"):
        env = Env(False, mode); result = verify_dual_placement(env, Checker(env, held=False), env.targets, region_predicates=predicates, max_steps=25)
        assert result["status"] == "placement_timeout" and len(env.actions) == 25
    env = Env(False, "bounce_right"); placed = verify_dual_placement(env, Checker(env, held=False), env.targets, region_predicates=predicates)
    assert placed["status"] == "passed" and len(env.actions) == 31
    checks.append("both official/physical regions, 20-tick common window, bounce and slow drift reset")
    env = Env(False); reject(lambda: release_dual_and_confirm(env, Checker(env), env.targets))
    reject(lambda: verify_dual_placement(env, Checker(env, held=False), (None, env.targets[1]), region_predicates=predicates))
    reject(lambda: close_dual_and_confirm(env, Checker(env, held=False), env.targets[::-1]))
    reject(lambda: execute_dual_held_move(env, schedule(env), Checker(env), env.targets, min_height_gain_m=(None, -1.)))
    assert not env.actions
    checks.append("attached release and invalid targets/thresholds reject before stepping")
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
            "checks": checks, "check_count": len(checks)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))
