"""Synchronous dual-Panda phases with one paired environment step per tick.

Physical left/right identities are never reordered. Both arms cross each phase
barrier together; a shorter source trajectory holds its own endpoint. Carried
objects move through contact physics, never through pose assignment or a weld.
Held stability uses measured pose differences while raw velocities remain in
the trace. These sampled checks do not establish continuous collision safety.
Policy supplies numeric defaults; this executor does not select or skip quality
phases. Explicit window arguments are recorded as per-phase overrides.
"""
from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable

import numpy as np

from .combined_collision import validate_schedule
from .measurements import _actions_valid, _budget, _error, _measure, _positive
from .measurements import _observe, _rotation_distance, _transform
from .runtime_policy import DEFAULT_POLICY, RuntimePolicy
from .trajectory_execution import as_numpy, bindings, snapshot
from .trajectory_schedule import ArmTrajectory, DualArmSchedule, build_hold_schedule, plan_pose

JOINT_STEP, TRACK, ENDPOINT, ARM_SPEED, OPEN_TOL = .02, .2, .03, .05, .002
DROP = .01
OBJECT_SPEED, OBJECT_ANGULAR_SPEED = .03, .2
WINDOW_POSITION, WINDOW_ROTATION, WINDOW_JOINT = .003, .03, .003


def _runtime_policy(value: RuntimePolicy) -> RuntimePolicy:
    if not isinstance(value, RuntimePolicy):
        raise ValueError("A resolved RuntimePolicy is required")
    return value


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

    def __init__(self, env: Any, checker: Any, name: str, *, targets: Any = None, held: bool = False,
                 policy: RuntimePolicy = DEFAULT_POLICY):
        self.policy = _runtime_policy(policy)
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
                           "position_slip_m": self.policy.position_slip_m, "rotation_slip_rad": self.policy.rotation_slip_rad,
                           "drop_below_attachment_m": DROP, "grasp_loss_ticks": 2,
                           "object_speed_m_s": OBJECT_SPEED, "object_angular_speed_rad_s": OBJECT_ANGULAR_SPEED,
                           "window_position_m": WINDOW_POSITION, "window_rotation_rad": WINDOW_ROTATION,
                           "window_joint_rad": WINDOW_JOINT},
            "abort_semantics": "stop both arms; no automatic release, retreat, reset or next phase",
        }
        self.started = time.monotonic()

    def observe(self) -> dict[str, Any]:
        # Share only within this call. Object reads do not advance physics;
        # the next observe() obtains fresh joints, including after every tick.
        measurements = _measure(self.arms)
        objects = []
        for index, (arm, target) in enumerate(zip(self.arms, self.targets)):
            if target is None:
                objects.append({"target_present": False, "is_grasping": None,
                                "tcp_T_world": _transform(arm.agent.tcp.pose.to_transformation_matrix()).tolist()})
                continue
            value = _observe((arm, self.arms[1 - index]), target, self.references[index], self.initial_z[index],
                             measurements=measurements)
            value.pop("measurements")
            value["rotation_slip_rad"] = _rotation_distance(self.references[index][:3, :3], np.asarray(value["T_tcp_object"])[:3, :3])
            value.update(target_present=True, action_key=arm.key, object_index=index)
            objects.append(value)
        return {"measurements": measurements, "objects": objects}

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
                               "action": {key: value.tolist() for key, value in actual.items()},
                               "control_command_submitted": False}
        try:
            if self.held and getattr(self.checker, "measured_attachment_prediction", False):
                current = self.observe()
                receipt = self.checker.validate_measured_command(
                    *(actual[arm.key][:7] for arm in self.arms),
                    grasping=tuple(bool(obj["is_grasping"]) for obj in current["objects"]),
                    max_joint_step_rad=JOINT_STEP)
                row["attachment_geometry"] = receipt
                if receipt["status"] != "passed":
                    row["control_command_submitted"] = False
                    self.report["trace"].append(row)
                    self.report.update(status="aborted", reason=receipt["reason"])
                    return None
            row["control_command_submitted"] = True
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
            elif obj["position_slip_m"] > self.policy.position_slip_m: reason = "position_slip"
            elif obj["rotation_slip_rad"] > self.policy.rotation_slip_rad: reason = "rotation_slip"
            elif self.loss[index] >= 2 or (hovering and not obj["is_grasping"]): reason = "grasp_lost"
            if reason:
                self.report.update(status="aborted", reason=reason, object_index=index)
                return False
        return True

    def finish(self) -> dict[str, Any]:
        self.report["wall_seconds"] = time.monotonic() - self.started
        return self.report


def plan_dual_pose(left_solver: Any, right_solver: Any, target_poses: tuple[Any | None, Any | None], *,
                   gripper_commands: tuple[int, int] = (1, 1),
                   policy: RuntimePolicy = DEFAULT_POLICY) -> DualArmSchedule:
    """Dry-run both native plans at one state; None holds that measured arm."""
    _runtime_policy(policy)
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
        if hasattr(solver, "last_native_result"):
            solver.last_native_result = None
        if hasattr(solver, "last_native_parameterization"):
            solver.last_native_parameterization = None
        trajectory = (ArmTrajectory(arm.key, arm.names, [0.], [arm.arm_qpos()], grip) if pose is None else
                      plan_pose(solver, pose, action_key=arm.key, controller_joint_names=arm.names))
        if not all(np.array_equal(a, b) for a, b in zip(before, snapshot(raw, arms))):
            raise RuntimeError("Dry-run planning changed the physical environment")
        if not np.allclose(trajectory.positions_rad[0], arm.arm_qpos(), rtol=0, atol=1e-6):
            raise ValueError("Planner source does not start at the measured joints")
        trajectories.append(trajectory)
    return build_hold_schedule(*trajectories, control_timestep_s=raw.control_timestep)


def _scheduled(env: Any, schedule: DualArmSchedule, checker: Any, *, held: bool, targets: Any,
               minimum: Any, hold_steps: int, settle_max_steps: int | None, stable_steps: int | None,
               policy: RuntimePolicy = DEFAULT_POLICY) -> dict[str, Any]:
    policy = _runtime_policy(policy)
    settle_max_steps = policy.max_settle_steps if settle_max_steps is None else settle_max_steps
    stable_steps = policy.reach_stable_steps if stable_steps is None else stable_steps
    settling, required = _budget(settle_max_steps, "settle_max_steps", zero=True), _budget(stable_steps, "stable_steps")
    hover = _budget(hold_steps, "hold_steps", zero=True)
    minima = tuple(None if value is None else _positive(value, "min_height_gain_m") for value in _pair(minimum, "min_height_gain_m"))
    phase = _Phase(env, checker, "dual_held_move" if held else "dual_reach", targets=targets, held=held, policy=policy)
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
    if held and any(not obj["is_grasping"] or obj["position_slip_m"] > policy.position_slip_m or obj["rotation_slip_rad"] > policy.rotation_slip_rad for obj in initial["objects"]):
        phase.report.update(status="rejected", reason="initial_grasp_or_attachment_invalid")
        return phase.finish()
    if not phase.preflight():
        return phase.finish()
    candidate = checker.candidate_checker() if callable(getattr(checker, "candidate_checker", None)) else checker
    phase.report["validation"] = validate_schedule(schedule, *(np.asarray(initial["measurements"][arm.key]["qpos_rad"]) for arm in phase.arms), candidate, max_joint_step_rad=JOINT_STEP)
    if phase.report["validation"]["status"] == "passed" and callable(getattr(candidate, "finish", None)):
        remaining = candidate.finish()
        if remaining:
            phase.report["validation"].update(status="collision", first_collision={"collisions": remaining})
    if phase.report["validation"]["status"] != "passed":
        phase.report.update(status="rejected", reason="candidate_collision")
        return phase.finish()

    def stable(row: dict[str, Any]) -> bool:
        speed = row["max_control_frame_joint_speed_rad_s"] if held else row["max_arm_velocity_rad_s"]
        if row["endpoint_error_rad"] > ENDPOINT or speed > ARM_SPEED:
            return False
        if getattr(checker, "release_separation", None) is not None and (
                not checker.release_separation.all_clear or not env.release_separation_complete):
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
    separation = getattr(checker, "release_separation", None)
    if phase.report["status"] == "passed" and separation is not None:
        remaining = separation.finish()
        if remaining:
            phase.report.update(status="aborted", reason="measured_collision", remaining_release_contacts=remaining)
    phase.report["stable_steps_observed"] = count
    return phase.finish()


def execute_dual_reach(env: Any, schedule: DualArmSchedule, checker: Any, *,
                       allow_target_contacts: tuple[bool, bool] = (False, False),
                       settle_max_steps: int | None = None, stable_steps: int | None = None,
                       policy: RuntimePolicy = DEFAULT_POLICY) -> dict[str, Any]:
    """Reach with policy defaults; explicit window overrides still apply."""
    allowed = _pair(allow_target_contacts, "allow_target_contacts")
    if any(type(value) is not bool for value in allowed):
        raise ValueError("Target-contact permissions must be explicit booleans")
    checker.set_target_contacts_allowed(allowed)
    try:
        return _scheduled(env, schedule, checker, held=False, targets=None, minimum=(None, None), hold_steps=0,
                          settle_max_steps=settle_max_steps, stable_steps=stable_steps, policy=policy)
    finally:
        checker.set_target_contacts_allowed((False, False))


def close_dual_and_confirm(env: Any, checker: Any, targets: tuple[Any, Any], *,
                           max_steps: int = 40, stable_steps: int | None = None,
                           policy: RuntimePolicy = DEFAULT_POLICY) -> dict[str, Any]:
    policy = _runtime_policy(policy)
    stable_steps = policy.grasp_stable_steps if stable_steps is None else stable_steps
    targets = _pair(targets, "targets")
    if any(target is None for target in targets): raise ValueError("Both physical targets are required")
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    if budget < required: raise ValueError("Grasp budget cannot be smaller than the required stable interval")
    phase = _Phase(env, checker, "dual_close", targets=targets, policy=policy)
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
                           hold_steps: int = 0, settle_max_steps: int | None = None, stable_steps: int | None = None,
                           policy: RuntimePolicy = DEFAULT_POLICY) -> dict[str, Any]:
    targets = _pair(targets, "targets")
    if any(target is None for target in targets): raise ValueError("Both physical held targets are required")
    return _scheduled(env, schedule, checker, held=True, targets=targets, minimum=min_height_gain_m,
                      hold_steps=hold_steps, settle_max_steps=settle_max_steps, stable_steps=stable_steps, policy=policy)


def release_dual_and_confirm(env: Any, checker: Any, targets: tuple[Any, Any], *,
                             max_steps: int = 30, stable_steps: int | None = None,
                             policy: RuntimePolicy = DEFAULT_POLICY) -> dict[str, Any]:
    policy = _runtime_policy(policy)
    stable_steps = policy.release_stable_steps if stable_steps is None else stable_steps
    targets = _pair(targets, "targets")
    if any(target is None for target in targets): raise ValueError("Both physical release targets are required")
    if getattr(checker, "attached", False): raise ValueError("Detach collision predictions before physical release")
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    phase = _Phase(env, checker, "dual_release", targets=targets, policy=policy)
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


def observe_dual_task_outcome(env: Any, checker: Any, targets: tuple[Any, Any], *,
                              official_predicate_keys: tuple[str, str] = ("apple_in_basket", "banana_in_basket"),
                              region_predicates: tuple[Callable[[Any], Any], Callable[[Any], Any]],
                              max_steps: int = 0,
                              policy: RuntimePolicy = DEFAULT_POLICY) -> dict[str, Any]:
    """Return on the first actual released/placed official success, or a failure.

    ``max_steps`` is the caller's remaining control-step budget and may be zero.
    The caller keeps zero-step results outside action stages/ticks. The returned
    final observation is a separate copy, including the measured control step;
    a later measurement failure never relabels an older observation as current.
    Native termination/truncation and actual collision rejection take priority
    over a task-success observation at the same tick.
    """
    policy = _runtime_policy(policy)
    targets = _pair(targets, "targets")
    keys = _pair(official_predicate_keys, "official_predicate_keys")
    predicates = _pair(region_predicates, "region_predicates")
    if any(target is None for target in targets) or any(not callable(predicate) for predicate in predicates):
        raise ValueError("Both actual targets and physical region callbacks are required")
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("Explicit official predicate keys are required")
    if getattr(checker, "attached", False):
        raise ValueError("Task outcome observation requires detached collision predictions")
    budget = _budget(max_steps, "max_steps", zero=True)
    phase = _Phase(env, checker, "observe_task_outcome", targets=targets, policy=policy)
    phase.report.update(max_steps=budget, official_predicate_keys=list(keys),
                        quality_window_evaluated=False, velocity_gate_evaluated=False,
                        task_outcome_observation_evaluated=False,
                        final_observation=None)

    def action(row: dict[str, Any]) -> dict[str, np.ndarray]:
        return {arm.key: np.r_[row["measurements"][arm.key]["qpos_rad"], 1.]
                for arm in phase.arms}

    def observe(row: dict[str, Any]) -> bool:
        regions = _regions(env, targets, keys, predicates)
        evaluation = env.unwrapped.evaluate()
        if "success" not in evaluation:
            raise ValueError("Missing official task success predicate")
        success = _bool(evaluation["success"])
        row.update(fingers_open=[phase.opened(arm, row) for arm in phase.arms],
                   official_success=success, official_evaluation={key: as_numpy(value).tolist() for key, value in evaluation.items()},
                   observed_elapsed_steps=as_numpy(env.unwrapped.elapsed_steps).tolist())
        for obj, region in zip(row["objects"], regions):
            obj.update(region)
        valid = bool(success and all(row["fingers_open"]) and all(
            obj["official_region"] and obj["physical_region"]["inside"] and not obj["is_grasping"]
            for obj in row["objects"]))
        row["task_outcome_satisfied"] = valid
        phase.report.update(task_outcome_observation_evaluated=True,
                            final_observation=copy.deepcopy(row))
        return valid

    checker.set_target_contacts_allowed((False, False))
    try:
        row = phase.report["initial_observation"]
        _actions_valid(env, phase.arms, action(row))
        if not phase.preflight():
            return phase.finish()
        try:
            satisfied = observe(row)
        except Exception as exc:
            _error(phase.report, "task_outcome_measurement_failed", exc)
            return phase.finish()
        if satisfied:
            phase.report["status"] = "passed"
            return phase.finish()
        for _ in range(budget):
            row = phase.tick(action(row), "observe_task_outcome")
            if row is None:
                break
            try:
                satisfied = observe(row)
            except Exception as exc:
                row["task_outcome_observation_error"] = f"{type(exc).__name__}: {exc}"
                if phase.report["status"] == "running":
                    _error(phase.report, "task_outcome_measurement_failed", exc)
                break
            if phase.report["status"] != "running":
                break
            if satisfied:
                phase.report["status"] = "passed"
                break
        if phase.report["status"] == "running":
            phase.report.update(status="task_outcome_timeout", reason="official_released_placements_not_observed")
        return phase.finish()
    finally:
        checker.set_target_contacts_allowed((False, False))


def verify_dual_placement(env: Any, checker: Any, targets: tuple[Any, Any], *,
                          official_predicate_keys: tuple[str, str] = ("apple_in_basket", "banana_in_basket"),
                          region_predicates: tuple[Callable[[Any], Any], Callable[[Any], Any]],
                          max_steps: int = 60, stable_steps: int | None = None,
                          policy: RuntimePolicy = DEFAULT_POLICY) -> dict[str, Any]:
    policy = _runtime_policy(policy)
    stable_steps = policy.final_stable_steps if stable_steps is None else stable_steps
    targets, keys, predicates = _pair(targets, "targets"), _pair(official_predicate_keys, "official_predicate_keys"), _pair(region_predicates, "region_predicates")
    if any(target is None for target in targets) or any(not callable(predicate) for predicate in predicates):
        raise ValueError("Both actual targets and physical region callbacks are required")
    if any(not isinstance(key, str) or not key for key in keys): raise ValueError("Explicit official predicate keys are required")
    if getattr(checker, "attached", False): raise ValueError("Placement verification requires detached collision predictions")
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    phase = _Phase(env, checker, "dual_placement_verification", targets=targets, policy=policy)
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
