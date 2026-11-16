"""Bounded start-delay coordination without changing either arm's source path.

Contract: seven joint positions in radians, timestamps in seconds, one constant
normalized gripper command per phase. A delay inserts a stationary prefix at
the measured source start; all original segment durations and positions stay
unchanged. Candidates share the existing controller clock and endpoint hold.
Every candidate within the control/sample budgets receives the existing full
combined collision check. This module never steps, resets or replans an env.
Passing sampled checks does not establish continuous collision safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from typing import Any

import numpy as np

try:
    from .combined_collision import CollisionChecker, validate_schedule
    from .trajectory_schedule import ArmTrajectory, DualArmSchedule, build_hold_schedule
except ImportError:
    from combined_collision import CollisionChecker, validate_schedule
    from trajectory_schedule import ArmTrajectory, DualArmSchedule, build_hold_schedule


def _integer(value: int, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def delay_trajectory(trajectory: ArmTrajectory, delay_ticks: int,
                     control_timestep_s: float) -> ArmTrajectory:
    """Hold the initial position for ``delay_ticks``, then follow source time.

    Zero delay returns the original immutable trajectory. Positive delay adds
    one initial point, followed by every original point translated in time.
    The original t=0 position therefore appears at both ends of the hold.
    """
    if not isinstance(trajectory, ArmTrajectory):
        raise TypeError("trajectory must be an ArmTrajectory")
    ticks = _integer(delay_ticks, "delay_ticks", minimum=0)
    dt = _positive(control_timestep_s, "control_timestep_s")
    if ticks == 0:
        return trajectory
    try:
        delay_s = ticks * dt
    except OverflowError as exc:
        raise ValueError("Delay exceeds finite timestamps") from exc
    if not math.isfinite(delay_s) or not math.isfinite(delay_s + trajectory.duration_s):
        raise ValueError("Delay exceeds finite timestamps")
    times = np.r_[0.0, trajectory.times_s + delay_s]
    positions = np.vstack([trajectory.positions_rad[0], trajectory.positions_rad])
    assert positions.shape == (len(times), 7)
    return ArmTrajectory(trajectory.action_key, trajectory.joint_names, times,
                         positions, trajectory.gripper_command)


def _summary(trajectory: ArmTrajectory) -> dict[str, Any]:
    return {
        "action_key": trajectory.action_key,
        "joint_names": list(trajectory.joint_names),
        "timestamp_count": len(trajectory.times_s),
        "duration_s": trajectory.duration_s,
        "gripper_command": trajectory.gripper_command,
        "times_s_sha256": hashlib.sha256(trajectory.times_s.tobytes()).hexdigest(),
        "positions_rad_sha256": hashlib.sha256(trajectory.positions_rad.tobytes()).hexdigest(),
        "start_qpos_rad": trajectory.positions_rad[0].tolist(),
        "end_qpos_rad": trajectory.positions_rad[-1].tolist(),
    }


def _required_samples(schedule: DualArmSchedule, starts: tuple[np.ndarray, np.ndarray],
                      step: float, max_samples: int) -> int | None:
    """Mirror only the validator's allocation budget, without calling checker.

    None means a single interpolation already exceeds the sample budget. The
    actual validator still owns all collision/input checks for eligible paths.
    """
    positions = np.vstack([np.r_[*starts], np.column_stack([
        schedule.left_commands[:, :7], schedule.right_commands[:, :7]])])
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.max(np.abs(np.diff(positions, axis=0)), axis=1) / step
    if not np.isfinite(distances).all() or np.any(distances > max_samples):
        return None
    return 1 + sum(max(1, math.ceil(float(distance))) for distance in distances)


def search_start_delays(
    left: ArmTrajectory, right: ArmTrajectory, checker: CollisionChecker, *,
    control_timestep_s: float, max_delay_ticks: int, delay_step_ticks: int = 1,
    max_steps: int = 200, max_joint_step_rad: float = 0.02,
    max_samples: int = 20_000,
) -> tuple[DualArmSchedule | None, dict[str, Any]]:
    """Try (0,0), then (0,n), (n,0), returning the first validated candidate.

    n follows range(delay_step_ticks, max_delay_ticks+1, delay_step_ticks).
    Budgets cover this phase, including all inserted waiting intervals. An
    eligible candidate is checked from the original measured source starts.
    Checker exceptions propagate; they are never classified as no solution.
    """
    if not isinstance(left, ArmTrajectory) or not isinstance(right, ArmTrajectory):
        raise TypeError("left and right must be ArmTrajectory values")
    if left.action_key == right.action_key:
        raise ValueError("Physical left and right require distinct action keys")
    if not callable(getattr(checker, "check", None)):
        raise TypeError("checker must provide check(left7, right7)")
    dt = _positive(control_timestep_s, "control_timestep_s")
    resolution = _positive(max_joint_step_rad, "max_joint_step_rad")
    maximum = _integer(max_delay_ticks, "max_delay_ticks", minimum=0)
    stride = _integer(delay_step_ticks, "delay_step_ticks", minimum=1)
    steps = _integer(max_steps, "max_steps", minimum=1)
    samples = _integer(max_samples, "max_samples", minimum=1)
    starts = (left.positions_rad[0], right.positions_rad[0])
    report: dict[str, Any] = {
        "status": "no_solution", "source_trajectories": [_summary(left), _summary(right)],
        "candidate_order": "(0,0), then (0,n),(n,0) for n=stride..maximum by stride",
        "control_timestep_s": dt, "max_delay_ticks": maximum,
        "delay_step_ticks": stride, "max_steps": steps,
        "max_joint_step_rad": resolution, "max_samples": samples,
        "selected_delay_ticks": None, "candidates": [], "candidate_count": 0,
        "checked_candidate_count": 0, "environment_steps": 0,
        "source_geometry_changed": False, "source_time_scale_changed": False,
        "continuous_collision_detection": False,
    }

    def candidates() -> Any:
        yield (0, 0)
        for delay in range(stride, maximum + 1, stride):
            yield (0, delay)
            yield (delay, 0)

    for delays in candidates():
        item: dict[str, Any] = {
            "delay_ticks": list(delays), "status": "rejected", "reason": None,
            "planned_control_steps": None, "validation": None,
        }
        report["candidates"].append(item)
        report["candidate_count"] += 1
        try:
            durations = [source.duration_s + delay * dt
                         for source, delay in zip((left, right), delays)]
        except OverflowError:
            durations = [math.inf, math.inf]
        horizon = max(durations)
        ratio = horizon / dt
        if not math.isfinite(ratio):
            item["reason"] = "nonfinite_duration_exceeds_control_step_budget"
            continue
        count = math.ceil(ratio)
        item["planned_control_steps"] = count
        if count > steps:
            item["reason"] = "control_step_budget_exceeded"
            continue
        delayed = tuple(delay_trajectory(source, delay, dt)
                        for source, delay in zip((left, right), delays))
        schedule = build_hold_schedule(*delayed, control_timestep_s=dt, max_steps=steps)
        assert schedule.num_steps == count
        required = _required_samples(schedule, starts, resolution, samples)
        item["planned_collision_samples"] = required
        if required is None or required > samples:
            item["reason"] = "collision_sample_budget_exceeded"
            continue
        # Do not catch exceptions here: a broken checker is not an infeasible path.
        validation = validate_schedule(schedule, *starts, checker,
                                       max_joint_step_rad=resolution, max_samples=samples)
        report["checked_candidate_count"] += 1
        item["validation"] = validation
        if validation["status"] == "passed":
            item.update(status="passed", reason=None)
            report.update(status="passed", selected_delay_ticks=list(delays))
            return schedule, report
        item["reason"] = "candidate_collision"
    return None, report


def self_test() -> dict[str, Any]:
    """CPU fixtures distinguish endpoint timing, swept conflicts and failures."""
    names = tuple(f"joint{index}" for index in range(7))
    checks: list[str] = []

    def trajectory(key: str, times: list[float], values: list[float], grip: float) -> ArmTrajectory:
        positions = np.zeros((len(times), 7))
        positions[:, 0] = values
        return ArmTrajectory(key, names, times, positions, grip)

    class AnalyticalChecker:
        scope = "analytical_crossing_fixture"

        def __init__(self, mode: str = "crossing") -> None:
            self.mode, self.calls = mode, 0

        def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]:
            assert left7.shape == right7.shape == (7,)
            self.calls += 1
            if self.mode == "raise":
                raise ValueError("checker_failure_sentinel")
            collision = (self.mode == "initial" or
                         self.mode == "blocked" and abs(left7[0] - .5) <= .06 or
                         self.mode == "crossing" and abs(left7[0] - .5) <= .06
                         and abs(right7[0] - .5) <= .06)
            return [{"collision_type": "analytical", "fixture_mode": self.mode}] if collision else []

    left = trajectory("left", [0., 1.], [0., 1.], -1.)
    right = trajectory("right", [0., 1.], [1., 0.], 1.)
    dt = .25
    before = [(source.times_s.tobytes(), source.positions_rad.tobytes()) for source in (left, right)]
    assert delay_trajectory(left, 0, dt) is left
    old = build_hold_schedule(left, right, control_timestep_s=dt)
    zero, zero_report = search_start_delays(left, right, AnalyticalChecker("clear"),
                                          control_timestep_s=dt, max_delay_ticks=2)
    assert zero is not None and zero_report["selected_delay_ticks"] == [0, 0]
    assert zero.trajectories == old.trajectories
    for field in ("target_times_s", "left_commands", "right_commands"):
        assert np.array_equal(getattr(zero, field), getattr(old, field))
    checks.append("zero delay exactly preserves original A schedule")

    irregular = trajectory("left", [0., .3, 1.1], [0., .7, 1.], -1.)
    delayed = delay_trajectory(irregular, 2, dt)
    assert np.array_equal(delayed.times_s, [0., .5, .8, 1.6])
    assert np.array_equal(delayed.positions_rad[0], irregular.positions_rad[0])
    assert np.array_equal(delayed.positions_rad[1:], irregular.positions_rad)
    assert delayed.gripper_command == -1. and delayed.joint_names == names
    other = trajectory("right", [0., .25], [2., 3.], 1.)
    mixed = build_hold_schedule(delayed, other, control_timestep_s=dt)
    assert mixed.num_steps == 7
    assert np.array_equal(mixed.left_commands[:2, :7], np.zeros((2, 7)))
    for index, clock in enumerate(mixed.target_times_s):
        expected = np.interp(max(0., clock - .5), irregular.times_s, irregular.positions_rad[:, 0])
        assert np.isclose(mixed.left_commands[index, 0], expected, rtol=0, atol=1e-15)
    assert mixed.left_commands[-1, 0] == 1. and np.all(mixed.left_commands[:, 7] == -1.)
    assert np.all(mixed.right_commands[:, 0] == 3.) and np.all(mixed.right_commands[:, 7] == 1.)
    stationary = trajectory("left", [0.], [4.], -1.)
    held = delay_trajectory(stationary, 1, dt)
    assert np.array_equal(held.times_s, [0., dt]) and np.all(held.positions_rad[:, 0] == 4.)
    checks.append("irregular source timestamps, one-tick boundaries, endpoint and gripper holds")

    resolved, resolution_report = search_start_delays(left, right, AnalyticalChecker(),
                                                     control_timestep_s=dt, max_delay_ticks=3)
    assert resolved is not None and resolution_report["selected_delay_ticks"] == [0, 1]
    assert resolved.num_steps == 5 and resolution_report["checked_candidate_count"] == 2
    rejected, accepted = resolution_report["candidates"]
    assert rejected["validation"]["first_collision"]["time_s"] > 0
    assert rejected["reason"] == "candidate_collision" and accepted["validation"]["status"] == "passed"
    assert resolved.right_commands[0, 0] == 1. and resolved.left_commands[0, 0] == .25
    checks.append("simultaneous crossing rejected and first one-tick wait resolves conflict")

    none, blocked = search_start_delays(left, right, AnalyticalChecker("blocked"),
                                       control_timestep_s=dt, max_delay_ticks=3, delay_step_ticks=2)
    assert none is None and blocked["candidate_count"] == blocked["checked_candidate_count"] == 3
    assert [row["delay_ticks"] for row in blocked["candidates"]] == [[0, 0], [0, 2], [2, 0]]
    assert all(row["validation"]["first_collision"] for row in blocked["candidates"])
    none, initial = search_start_delays(left, right, AnalyticalChecker("initial"),
                                       control_timestep_s=dt, max_delay_ticks=1)
    assert none is None and all(row["validation"]["first_collision"]["time_s"] == 0
                               for row in initial["candidates"])
    checks.append("exhausted search retains every first collision and cannot fix initial collision")

    clear = AnalyticalChecker("clear")
    none, budget = search_start_delays(left, right, clear, control_timestep_s=dt,
                                      max_delay_ticks=1, max_steps=3)
    assert none is None and clear.calls == 0 and budget["checked_candidate_count"] == 0
    assert all(row["reason"] == "control_step_budget_exceeded" for row in budget["candidates"])
    crossing = AnalyticalChecker()
    none, boundary = search_start_delays(left, right, crossing, control_timestep_s=dt,
                                        max_delay_ticks=1, max_steps=4)
    assert none is None and boundary["checked_candidate_count"] == 1
    assert [row["planned_control_steps"] for row in boundary["candidates"]] == [4, 5, 5]
    clear = AnalyticalChecker("clear")
    none, budget_samples = search_start_delays(left, right, clear, control_timestep_s=dt,
                                              max_delay_ticks=1, max_samples=1)
    assert none is None and clear.calls == 0
    assert all(row["reason"] == "collision_sample_budget_exceeded" for row in budget_samples["candidates"])
    one_point_right = trajectory("right", [0.], [2.], 1.)
    empty, empty_report = search_start_delays(stationary, one_point_right, clear,
                                             control_timestep_s=dt, max_delay_ticks=0,
                                             max_steps=1, max_samples=1)
    assert empty is not None and empty.num_steps == 0 and empty_report["checked_candidate_count"] == 1
    checks.append("control and interpolation budgets rejected before checker, zero-motion t0 checked")

    rejected_inputs = 0

    def reject(operation: Any) -> None:
        nonlocal rejected_inputs
        try:
            operation()
        except (ValueError, TypeError):
            rejected_inputs += 1
            return
        raise AssertionError("Malformed delay or search input was accepted")

    for bad in (-1, True, .5):
        reject(lambda bad=bad: delay_trajectory(left, bad, dt))
    for bad in (0., -1., float("inf"), float("nan"), True):
        reject(lambda bad=bad: delay_trajectory(left, 0, bad))
    for options in ({"max_delay_ticks": -1}, {"max_delay_ticks": True},
                    {"delay_step_ticks": 0}, {"max_steps": 0}, {"max_samples": 0},
                    {"max_joint_step_rad": 0.}, {"max_joint_step_rad": float("inf")}):
        values = {"control_timestep_s": dt, "max_delay_ticks": 1, **options}
        reject(lambda values=values: search_start_delays(left, right, clear, **values))
    try:
        search_start_delays(left, right, AnalyticalChecker("raise"),
                            control_timestep_s=dt, max_delay_ticks=1)
    except ValueError as exc:
        assert str(exc) == "checker_failure_sentinel"
    else:
        raise AssertionError("Checker exception was swallowed")
    assert before == [(source.times_s.tobytes(), source.positions_rad.tobytes()) for source in (left, right)]
    assert not any(array.flags.writeable for source in (left, right, delayed)
                   for array in (source.times_s, source.positions_rad))
    checks.append("invalid inputs rejected, checker error propagated and source arrays immutable")
    return {
        "status": "passed", "checks": checks, "invalid_input_checks": rejected_inputs,
        "zero_delay_steps": zero.num_steps, "resolved_steps": resolved.num_steps,
        "resolved_delay_ticks": resolution_report["selected_delay_ticks"],
        "resolved_search": resolution_report, "all_candidates_blocked": blocked,
        "environment_steps": 0, "native_mplib_tested": False, "gpu_used": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
