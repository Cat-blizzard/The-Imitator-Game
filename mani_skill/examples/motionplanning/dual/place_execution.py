"""Carry, release and verify an apple placement; right arm holds throughout.

No physical pose, constraint or environment state is assigned. An abort means
stop advancing this phase, without opening, resetting, retreating or running
the next phase. Physics contacts between the released object and its supports
are normal; live scene checkers still check all robot collisions. Stability is
measured from control-frame poses, with raw velocities retained separately.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np

from mani_skill.examples.motionplanning.dual.combined_collision import validate_schedule
from mani_skill.examples.motionplanning.dual.grasp_execution import _actions_valid, _budget, _error, _positive
from mani_skill.examples.motionplanning.dual.lift_execution import _observe, _rotation_distance, _transform
from mani_skill.examples.motionplanning.dual.trajectory_execution import as_numpy, bindings
from mani_skill.examples.motionplanning.dual.trajectory_schedule import DualArmSchedule


def _boolean(value: Any) -> bool:
    array = as_numpy(value)
    if array.size != 1 or array.dtype.kind not in "biuf" or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError("Expected a single finite boolean predicate")
    return bool(array.reshape(-1)[0])


def _setup(env: Any, checker: Any, target: Any, keys: tuple[str, str], *, attached: bool) -> tuple[Any, np.ndarray, float, dict[str, Any]]:
    if keys != tuple(env.unwrapped.agent.agents_dict) or len(keys) != 2:
        raise ValueError("This left-active/right-held stage requires the environment's left/right key order")
    required = ("refresh", "check", "measured_check") if attached else ("refresh", "check", "set_target_contact_allowed")
    if not all(callable(getattr(checker, name, None)) for name in required):
        raise ValueError("Collision checker does not implement this phase's required interface")
    arms = bindings(env, keys)
    _positive(env.unwrapped.control_timestep, "control_timestep")
    if target is None or not callable(getattr(arms[0].agent, "is_grasping", None)):
        raise ValueError("A real target and physical grasp predicate are required")
    obj = _transform(target.pose.to_transformation_matrix())
    if attached:
        reference = _transform(checker.attachment_T_tcp_object)
    else:
        tcp = _transform(arms[0].agent.tcp.pose.to_transformation_matrix())
        reference = _transform(np.linalg.inv(tcp) @ obj)
    initial = _observe(arms, target, reference, float(obj[2, 3]))
    return arms, reference, float(obj[2, 3]), initial


def _report(env: Any, checker: Any, name: str, initial: dict[str, Any]) -> dict[str, Any]:
    return {"status": "running", "phase": name, "trace": [], "environment_steps": 0,
            "settling_steps": 0, "control_timestep_s": float(env.unwrapped.control_timestep),
            "initial_elapsed_steps": as_numpy(env.unwrapped.elapsed_steps).tolist(),
            "initial_observation": initial, "collision_scope": str(getattr(checker, "scope", "provided_checker")),
            "velocity_metric": "control_frame_pose_finite_difference", "continuous_collision_detection": False,
            "abort_semantics": "stop advancing this phase; no automatic release, reset, retreat or continuation"}


def _open(arm: Any, observation: dict[str, Any], tolerance: float) -> bool:
    fingers = [i for i in range(9) if i not in arm.indices]
    upper = as_numpy(arm.agent.robot.get_qlimits())[0, fingers, 1]
    return bool(np.allclose(observation["measurements"][arm.key]["finger_qpos_m"], upper, rtol=0, atol=tolerance))


def _hold(arms: Any, initial: dict[str, Any], closed: bool) -> dict[str, np.ndarray]:
    return {arm.key: np.r_[initial["measurements"][arm.key]["qpos_rad"], -1.0 if closed and i == 0 else 1.0]
            for i, arm in enumerate(arms)}


def _collisions(checker: Any, arms: Any, observation: dict[str, Any], attached: bool) -> list[dict[str, Any]]:
    checker.refresh()
    return checker.measured_check() if attached else checker.check(
        *(np.asarray(observation["measurements"][arm.key]["qpos_rad"]) for arm in arms))


def _preflight(checker: Any, arms: Any, initial: dict[str, Any], report: dict[str, Any], *, attached: bool) -> bool:
    try:
        collisions = _collisions(checker, arms, initial, attached)
    except Exception as exc:
        _error(report, "collision_preflight_failed", exc)
        report["status"] = "rejected"
        return False
    report["initial_collisions"] = collisions
    if collisions:
        report.update(status="rejected", reason="initial_collision")
        return False
    return True


def _tick(env: Any, arms: Any, checker: Any, target: Any, reference: np.ndarray, initial_z: float,
          previous: list[np.ndarray], action: dict[str, Any], report: dict[str, Any], *,
          phase: str, attached: bool) -> dict[str, Any] | None:
    actual = {k: np.asarray(v, dtype=np.float32).copy() for k, v in action.items()}
    row: dict[str, Any] = {"phase": phase, "step": report["environment_steps"] + 1,
                           "action": {k: v.tolist() for k, v in actual.items()}}
    before = as_numpy(env.unwrapped.elapsed_steps).copy()
    try:
        _, _, terminated, truncated, _ = env.step(actual)
        report["environment_steps"] += 1
        report["settling_steps"] += int(phase == "settling")
        row.update(time_s=report["environment_steps"] * report["control_timestep_s"],
                   terminated=_boolean(terminated), truncated=_boolean(truncated))
        row.update(_observe(arms, target, reference, initial_z))
        pose = _transform(row["object_T_world"])
        dt = report["control_timestep_s"]
        row["control_frame_linear_speed_m_s"] = float(np.linalg.norm(pose[:3, 3] - previous[0][:3, 3]) / dt)
        row["control_frame_angular_speed_rad_s"] = _rotation_distance(previous[0][:3, :3], pose[:3, :3]) / dt
        previous[0] = pose
        row["collisions"] = _collisions(checker, arms, row, attached)
        row["tracking_error_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][a.key]["qpos_rad"]) - actual[a.key][:7]))) for a in arms)
        row["right_hold_error_rad"] = float(np.max(np.abs(np.asarray(row["measurements"][arms[1].key]["qpos_rad"]) - report["initial_observation"]["measurements"][arms[1].key]["qpos_rad"])))
        row["max_arm_velocity_rad_s"] = max(float(np.max(np.abs(row["measurements"][a.key]["qvel_rad_s"]))) for a in arms)
    except Exception as exc:
        row["observed_elapsed_steps_after_error"] = as_numpy(env.unwrapped.elapsed_steps).tolist()
        row["elapsed_step_delta_on_error"] = (as_numpy(env.unwrapped.elapsed_steps) - before).tolist()
        row["error"] = f"{type(exc).__name__}: {exc}"
        report["trace"].append(row)
        _error(report, "step_or_measurement_failed", exc)
        return None
    report["trace"].append(row)
    if row["collisions"]:
        report.update(status="aborted", reason="measured_collision")
    elif row["terminated"] or row["truncated"]:
        report.update(status="terminated" if row["terminated"] else "truncated", reason="environment_done")
    return row


def execute_carry(env: Any, schedule: DualArmSchedule, checker: Any, target_actor: Any, *,
                  settle_max_steps: int = 15, stable_steps: int = 3, max_joint_step_rad: float = .02,
                  max_position_slip_m: float = .005, max_rotation_slip_rad: float = .15,
                  grasp_loss_max_ticks: int = 2, max_height_drop_m: float = .01,
                  tracking_tolerance_rad: float = .2, endpoint_tolerance_rad: float = .03,
                  velocity_tolerance_rad_s: float = .05, right_hold_tolerance_rad: float = .03,
                  max_object_speed_m_s: float = .03, max_object_angular_speed_rad_s: float = .2) -> dict[str, Any]:
    """Carry with a held-object checker; planned descent is permitted.

    Drop means the object falls below its current TCP-predicted attachment pose,
    not below a world-height maximum while the plan deliberately descends.
    """
    settling = _budget(settle_max_steps, "settle_max_steps", zero=True)
    required = _budget(stable_steps, "stable_steps")
    loss_limit = _budget(grasp_loss_max_ticks, "grasp_loss_max_ticks")
    values = {name: _positive(value, name) for name, value in {
        "resolution": max_joint_step_rad, "position_slip": max_position_slip_m, "rotation_slip": max_rotation_slip_rad,
        "drop": max_height_drop_m, "tracking": tracking_tolerance_rad, "endpoint": endpoint_tolerance_rad,
        "arm_speed": velocity_tolerance_rad_s, "right_hold": right_hold_tolerance_rad,
        "object_speed": max_object_speed_m_s, "angular_speed": max_object_angular_speed_rad_s}.items()}
    arms, reference, initial_z, initial = _setup(env, checker, target_actor,
        tuple(t.action_key for t in schedule.trajectories), attached=True)
    if not np.isclose(schedule.control_timestep_s, env.unwrapped.control_timestep, rtol=0, atol=1e-10):
        raise ValueError("Environment and schedule clocks differ")
    for arm, source, grip in zip(arms, schedule.trajectories, (-1, 1)):
        if source.joint_names != arm.names or source.gripper_command != grip:
            raise ValueError("Carry requires controller joint order, CLOSED active hand and OPEN passive hand")
    right_start = np.asarray(initial["measurements"][arms[1].key]["qpos_rad"])
    actions = list(schedule.iter_actions())
    if (not np.allclose(schedule.trajectories[1].positions_rad, right_start, rtol=0, atol=1e-6)
            or any(not np.allclose(a[arms[1].key][:7], right_start, rtol=0, atol=1e-6) for a in actions)):
        raise ValueError("Every right-arm source and command target must hold its measured entry position")
    final = {a.key: np.r_[t.positions_rad[-1], t.gripper_command] for a, t in zip(arms, schedule.trajectories)}
    for action in [*actions, final]: _actions_valid(env, arms, action)
    if not _open(arms[1], initial, .002): raise ValueError("Passive hand must already be open")
    report = _report(env, checker, "carry", initial)
    report.update(planned_control_steps=len(actions), settle_max_steps=settling, stable_steps_required=required,
                  attachment_T_tcp_object=reference.tolist(), velocity_thresholds={"linear_m_s": values["object_speed"], "angular_rad_s": values["angular_speed"]})
    started = time.monotonic()
    if not initial["is_grasping"]:
        report.update(status="rejected", reason="initial_grasp_absent"); return report
    if initial["position_slip_m"] > values["position_slip"] or initial["rotation_slip_rad"] > values["rotation_slip"]:
        report.update(status="rejected", reason="initial_attachment_slip"); return report
    if not _preflight(checker, arms, initial, report, attached=True): return report
    report["validation"] = validate_schedule(schedule, *(np.asarray(initial["measurements"][a.key]["qpos_rad"]) for a in arms),
                                             checker, max_joint_step_rad=values["resolution"])
    if report["validation"]["status"] != "passed":
        report.update(status="rejected", reason="candidate_attachment_collision"); return report
    previous, loss, stable = [_transform(initial["object_T_world"])], 0, 0
    for index in range(len(actions) + settling):
        phase = "carrying" if index < len(actions) else "settling"
        action = actions[index] if phase == "carrying" else final
        row = _tick(env, arms, checker, target_actor, reference, initial_z, previous, action, report, phase=phase, attached=True)
        if row is None or report["status"] != "running": break
        expected = np.asarray(row["tcp_T_world"]) @ reference
        row["drop_below_predicted_attachment_m"] = max(0.0, float(expected[2, 3] - row["object_height_m"]))
        row["endpoint_error_rad"] = max(float(np.max(np.abs(np.asarray(row["measurements"][a.key]["qpos_rad"]) - final[a.key][:7]))) for a in arms)
        loss = 0 if row["is_grasping"] else loss + 1
        row["consecutive_grasp_loss_ticks"] = loss
        reason = None
        if row["drop_below_predicted_attachment_m"] > values["drop"]: reason = "object_dropped"
        elif row["position_slip_m"] > values["position_slip"]: reason = "position_slip"
        elif row["rotation_slip_rad"] > values["rotation_slip"]: reason = "rotation_slip"
        elif loss >= loss_limit: reason = "grasp_lost"
        elif row["right_hold_error_rad"] > values["right_hold"]: reason = "right_hold_error"
        elif row["tracking_error_rad"] > values["tracking"]: reason = "tracking_error"
        if reason: report.update(status="aborted", reason=reason); break
        ready = (row["is_grasping"] and row["endpoint_error_rad"] <= values["endpoint"]
                 and row["max_arm_velocity_rad_s"] <= values["arm_speed"]
                 and row["control_frame_linear_speed_m_s"] <= values["object_speed"]
                 and row["control_frame_angular_speed_rad_s"] <= values["angular_speed"])
        stable = stable + 1 if ready else 0
        row["stable_steps"] = stable
        if index + 1 >= len(actions) and stable >= required: report["status"] = "passed"; break
    if report["status"] == "running": report.update(status="carry_timeout", reason="endpoint_or_velocity_not_stable")
    report.update(stable_steps_observed=stable, wall_seconds=time.monotonic() - started)
    return report


def release_and_confirm(env: Any, live_checker: Any, target_actor: Any, *, active_key: str,
                        max_steps: int = 30, stable_steps: int = 3,
                        arm_hold_tolerance_rad: float = .03, open_finger_tolerance_m: float = .002) -> dict[str, Any]:
    """Open both hands with entry joint targets held; require actual release."""
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    tolerance = _positive(arm_hold_tolerance_rad, "arm_hold_tolerance_rad")
    finger_tolerance = _positive(open_finger_tolerance_m, "open_finger_tolerance_m")
    keys = tuple(env.unwrapped.agent.agents_dict)
    if active_key not in keys or len(keys) != 2: raise ValueError("Expected a real active key and exactly two arms")
    keys = (active_key, next(k for k in keys if k != active_key))
    arms, reference, initial_z, initial = _setup(env, live_checker, target_actor, keys, attached=False)
    if not _open(arms[1], initial, finger_tolerance): raise ValueError("Passive hand must already be open")
    action = _hold(arms, initial, False); _actions_valid(env, arms, action)
    report = _report(env, live_checker, "release", initial)
    report.update(max_steps=budget, stable_steps_required=required, active_key=active_key)
    previous, stable, started = [_transform(initial["object_T_world"])], 0, time.monotonic()
    live_checker.set_target_contact_allowed(True)
    try:
        if not _preflight(live_checker, arms, initial, report, attached=False): return report
        for _ in range(budget):
            row = _tick(env, arms, live_checker, target_actor, reference, initial_z, previous, action, report, phase="releasing", attached=False)
            if row is None or report["status"] != "running": break
            row["fingers_open"] = all(_open(a, row, finger_tolerance) for a in arms)
            if row["right_hold_error_rad"] > tolerance: report.update(status="aborted", reason="right_hold_error"); break
            if row["tracking_error_rad"] > tolerance: report.update(status="aborted", reason="arm_hold_error"); break
            stable = stable + 1 if row["fingers_open"] and not row["is_grasping"] else 0
            row["stable_steps"] = stable
            if stable >= required: report["status"] = "passed"; break
        if report["status"] == "running": report.update(status="release_timeout", reason="fingers_not_open_or_grasp_persists")
        report.update(stable_steps_observed=stable, wall_seconds=time.monotonic() - started)
        return report
    finally:
        live_checker.set_target_contact_allowed(False)


def _region(env: Any, target: Any, predicate: Callable[[Any], Any]) -> tuple[bool, dict[str, Any]]:
    evaluation = env.unwrapped.evaluate()
    if "apple_in_basket" not in evaluation: raise ValueError("Official apple_in_basket predicate is required")
    official = _boolean(evaluation["apple_in_basket"])
    value = predicate(target)
    details = dict(value) if isinstance(value, dict) else {"inside": value}
    if "inside" not in details: raise ValueError("Physical region predicate must return bool or dict with inside")
    details["inside"] = _boolean(details["inside"])
    # Fail before motion for an unserializable or nonfinite geometry receipt.
    json.dumps(details, allow_nan=False)
    return official, details


def verify_placement(env: Any, live_checker: Any, target_actor: Any, *, max_steps: int = 60,
                     stable_steps: int = 20, arm_hold_tolerance_rad: float = .03,
                     velocity_tolerance_rad_s: float = .05, open_finger_tolerance_m: float = .002,
                     max_object_speed_m_s: float = .03, max_object_angular_speed_rad_s: float = .2,
                     max_window_position_drift_m: float = .003, max_window_rotation_drift_rad: float = .03,
                     region_predicate: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """Require 20 consecutive physically valid, released and stable basket ticks.

    The required predicate must additionally bound the object's actual physical
    placement region/height; the official flag alone has no lower-Z condition.
    A bounce/region failure resets the window. Slow accumulated drift also
    restarts it, so low speed alone cannot establish stable placement.
    """
    budget, required = _budget(max_steps, "max_steps"), _budget(stable_steps, "stable_steps")
    values = {name: _positive(value, name) for name, value in {
        "arm_hold": arm_hold_tolerance_rad, "arm_speed": velocity_tolerance_rad_s,
        "fingers": open_finger_tolerance_m, "object_speed": max_object_speed_m_s,
        "angular_speed": max_object_angular_speed_rad_s, "position_drift": max_window_position_drift_m,
        "rotation_drift": max_window_rotation_drift_rad}.items()}
    if not callable(region_predicate): raise ValueError("An explicit physical placement region/height predicate is required")
    arms, reference, initial_z, initial = _setup(env, live_checker, target_actor, tuple(env.unwrapped.agent.agents_dict), attached=False)
    if not all(_open(a, initial, values["fingers"]) for a in arms): raise ValueError("Both hands must already be open before placement verification")
    official, physical = _region(env, target_actor, region_predicate)
    action = _hold(arms, initial, False); _actions_valid(env, arms, action)
    report = _report(env, live_checker, "placement_verification", initial)
    report.update(max_steps=budget, stable_steps_required=required, initial_official_apple_in_basket=official,
                  initial_physical_region=physical, velocity_thresholds={"linear_m_s": values["object_speed"], "angular_rad_s": values["angular_speed"]},
                  window_drift_limits={"position_m": values["position_drift"], "rotation_rad": values["rotation_drift"]})
    live_checker.set_target_contact_allowed(False)
    if not _preflight(live_checker, arms, initial, report, attached=False): return report
    previous, stable, window, started = [_transform(initial["object_T_world"])], 0, None, time.monotonic()
    for _ in range(budget):
        row = _tick(env, arms, live_checker, target_actor, reference, initial_z, previous, action, report, phase="verifying_placement", attached=False)
        if row is None or report["status"] != "running": break
        try: row["official_apple_in_basket"], row["physical_region"] = _region(env, target_actor, region_predicate)
        except Exception as exc: _error(report, "region_measurement_failed", exc); row["error"] = str(exc); break
        row["fingers_open"] = all(_open(a, row, values["fingers"]) for a in arms)
        if row["right_hold_error_rad"] > values["arm_hold"]: report.update(status="aborted", reason="right_hold_error"); break
        if row["tracking_error_rad"] > values["arm_hold"]: report.update(status="aborted", reason="arm_hold_error"); break
        valid = (row["official_apple_in_basket"] and row["physical_region"]["inside"] and not row["is_grasping"]
                 and row["fingers_open"] and row["max_arm_velocity_rad_s"] <= values["arm_speed"]
                 and row["control_frame_linear_speed_m_s"] <= values["object_speed"]
                 and row["control_frame_angular_speed_rad_s"] <= values["angular_speed"])
        pose = _transform(row["object_T_world"])
        if not valid:
            stable, window = 0, None
            row["stability_reset_reason"] = "region_release_or_velocity"
        else:
            if window is None: window = pose.copy()
            position_drift = float(np.linalg.norm(pose[:3, 3] - window[:3, 3]))
            rotation_drift = _rotation_distance(window[:3, :3], pose[:3, :3])
            row.update(window_position_drift_m=position_drift, window_rotation_drift_rad=rotation_drift)
            if position_drift > values["position_drift"] or rotation_drift > values["rotation_drift"]:
                stable, window = 1, pose.copy()
                row["stability_reset_reason"] = "cumulative_window_drift"
            else: stable += 1
        row["stable_steps"] = stable
        if stable >= required: report["status"] = "passed"; break
    if report["status"] == "running": report.update(status="placement_timeout", reason="placement_not_continuously_stable")
    report.update(stable_steps_observed=stable, wall_seconds=time.monotonic() - started)
    return report


def self_test() -> dict[str, Any]:
    """Analytical CPU fixture; no real environment, GPU, pose reset or weld."""
    from types import SimpleNamespace as NS
    from gymnasium import spaces
    from mani_skill.examples.motionplanning.dual.trajectory_schedule import ArmTrajectory, build_hold_schedule
    names = tuple(f"joint{i}" for i in range(7))
    class Pose:
        def __init__(self, getter: Any): self.getter = getter
        def to_transformation_matrix(self) -> np.ndarray: return self.getter()[None].copy()
    class Robot:
        def __init__(self, left: bool, holding: bool):
            self.q = np.array([[0. if left else .25] + [0.] * 6 + [.012 if holding and left else .04] * 2]); self.v = np.zeros((1, 9))
        def get_qpos(self) -> np.ndarray: return self.q
        def get_qvel(self) -> np.ndarray: return self.v
        def get_qlimits(self) -> np.ndarray: return np.array([[[-2, 2]] * 7 + [[0, .04]] * 2])
        def get_active_joints(self) -> list[Any]: return [NS(get_name=lambda n=n: n) for n in (*names, "f1", "f2")]
    class Env:
        def __init__(self, holding: bool = True, mode: str = "normal"):
            self.unwrapped = self; self.num_envs = 1; self.control_timestep = .1; self.elapsed_steps = np.array([0])
            self.mode, self.actions = mode, []; self.matrix = np.eye(4); self.matrix[2, 3] = .12
            self.linear = np.zeros((1, 3)); self.agent = NS(agents_dict={})
            for key in ("left", "right"):
                robot = Robot(key == "left", holding)
                arm = NS(joints=robot.get_active_joints()[:7], config=NS(normalize_action=False, use_delta=False))
                agent = NS(robot=robot, control_mode="pd_joint_pos", controller=NS(controllers={"arm": arm}))
                def tcp(robot: Any = robot) -> np.ndarray:
                    matrix = np.eye(4); matrix[0, 3] = robot.q[0, 0]; matrix[2, 3] = .3 + robot.q[0, 1]; return matrix
                agent.tcp = NS(pose=Pose(tcp)); agent.is_grasping = lambda target, robot=robot: np.array([robot.q[0, 7] < .03])
                self.agent.agents_dict[key] = agent
            self.agent.single_action_space = spaces.Dict({k: spaces.Box(-2., 2., (8,), dtype=np.float32) for k in ("left", "right")})
            self.actor = NS(pose=Pose(lambda: self.matrix), get_linear_velocity=lambda: self.linear, get_angular_velocity=lambda: np.zeros((1, 3)))
        def evaluate(self) -> dict[str, Any]:
            return {"apple_in_basket": np.array([self.mode != "outside" and not (self.mode == "bounce" and len(self.actions) == 11)])}
        def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
            assert set(action) == {"left", "right"}; self.actions.append(action)
            for key, value in action.items():
                robot = self.agent.agents_dict[key].robot; before = robot.q.copy(); robot.q[0, :7] = value[:7]
                if value[7] == 1 and not (key == "left" and self.mode == "release_stuck"): robot.q[0, 7:] = .04
                robot.v = (robot.q - before) / .1
            before = self.matrix.copy(); left = self.agent.agents_dict["left"].robot
            if left.q[0, 7] < .03: self.matrix[:3, 3] = [left.q[0, 0], 0, .12 + left.q[0, 1]]
            if self.mode == "slide": self.matrix[0, 3] += .006
            if self.mode == "placement_drift": self.matrix[0, 3] += .0002
            if self.mode == "right_drift": self.agent.agents_dict["right"].robot.q[0, 0] += .05
            self.linear = ((self.matrix[:3, 3] - before[:3, 3]) / .1)[None]; self.elapsed_steps += 1
            return None, 0, False, len(self.actions) >= getattr(self, "truncate_at", 999), {}
    class Checker:
        scope = "analytical_fixture"; attachment_T_tcp_object = np.eye(4); attachment_T_tcp_object[2, 3] = -.18
        def __init__(self, env: Env, collision: bool = False): self.env, self.collision, self.allowed = env, collision, False
        def refresh(self) -> None: pass
        def set_target_contact_allowed(self, allowed: bool) -> None: self.allowed = allowed
        def check(self, left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]: return [{"fixture": True}] if self.collision and left[0] > .03 else []
        def measured_check(self) -> list[dict[str, Any]]: return []
    def schedule() -> DualArmSchedule:
        q = np.zeros((2, 7)); q[1, :2] = [.08, -.02]
        right = np.zeros(7); right[0] = .25
        return build_hold_schedule(ArmTrajectory("left", names, [0, .3], q, -1), ArmTrajectory("right", names, [0], [right], 1), control_timestep_s=.1)
    env = Env(); carried = execute_carry(env, schedule(), Checker(env), env.actor, stable_steps=2)
    assert carried["status"] == "passed" and carried["trace"][-1]["object_height_gain_m"] < 0
    assert env.agent.agents_dict["left"].robot.q[0, 7] == .012
    assert all(a["right"][0] == .25 and a["right"][7] == 1 for a in env.actions)
    env = Env(mode="slide"); assert execute_carry(env, schedule(), Checker(env), env.actor)["reason"] == "position_slip" and len(env.actions) == 1
    env = Env(); assert execute_carry(env, schedule(), Checker(env, True), env.actor)["status"] == "rejected" and not env.actions
    env = Env(); checker = Checker(env)
    released = release_and_confirm(env, checker, env.actor, active_key="left")
    assert released["status"] == "passed" and len(env.actions) == 3 and not checker.allowed
    env = Env(mode="release_stuck"); result = release_and_confirm(env, Checker(env), env.actor, active_key="left", max_steps=6)
    assert result["status"] == "release_timeout" and result["trace"][-1]["is_grasping"] and result["trace"][-1]["measurements"]["left"]["finger_qpos_m"][0] == .012
    predicate = lambda actor: {"inside": True, "height_bounds_m": [.01, .3]}
    env = Env(holding=False); result = verify_placement(env, Checker(env), env.actor, region_predicate=predicate)
    assert result["status"] == "passed" and result["environment_steps"] == 20
    env = Env(holding=False, mode="outside"); result = verify_placement(env, Checker(env), env.actor, max_steps=25, region_predicate=predicate)
    assert result["status"] == "placement_timeout" and len(env.actions) == 25
    env = Env(holding=False, mode="bounce"); result = verify_placement(env, Checker(env), env.actor, region_predicate=predicate)
    assert result["status"] == "passed" and len(env.actions) == 31 and result["trace"][10]["stable_steps"] == 0
    env = Env(holding=False); result = verify_placement(env, Checker(env), env.actor, max_steps=22, region_predicate=lambda actor: False)
    assert result["status"] == "placement_timeout"
    env = Env(holding=False, mode="placement_drift"); result = verify_placement(env, Checker(env), env.actor, max_steps=40, region_predicate=predicate)
    assert result["status"] == "placement_timeout" and any(r.get("stability_reset_reason") == "cumulative_window_drift" for r in result["trace"])
    env = Env(holding=False, mode="right_drift"); result = verify_placement(env, Checker(env), env.actor, region_predicate=predicate)
    assert result["reason"] == "right_hold_error" and len(env.actions) == 1
    env = Env(holding=False); env.truncate_at = 1
    assert verify_placement(env, Checker(env), env.actor, region_predicate=predicate)["status"] == "truncated" and len(env.actions) == 1
    env = Env()
    try: verify_placement(env, Checker(env), env.actor, region_predicate=predicate)
    except ValueError: pass
    else: raise AssertionError("Closed fingers were accepted before placement verification")
    assert not env.actions
    env = Env()
    try: release_and_confirm(env, Checker(env), env.actor, active_key="right")
    except ValueError: pass
    else: raise AssertionError("Unsupported right-active collision query order was accepted")
    assert not env.actions
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
            "checks": ["nonzero-finger carry with planned descent", "right holds its nonzero entry target", "carry slide rejection",
                       "candidate collision rejects before motion", "measured release", "retain-grip release timeout", "contact permission scoped to release",
                       "20 consecutive stable placement ticks", "outside basket times out", "bounce resets count", "physical height predicate required",
                       "slow cumulative drift resets window", "right drift stops", "truncation stops", "closed-hand zero-step rejection",
                       "unsupported right-active order rejects before motion"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
