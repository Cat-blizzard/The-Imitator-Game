"""Option A: start both Panda trajectories together and hold each endpoint.

This module prepares joint-position commands; it never advances an environment.
It is not a collision checker or evidence of reaching/grasping in simulation.
Before live execution, a separate adapter must validate current joint states,
joint limits and the combined trajectory against the actual scene.

Contract: seconds, radians, seven arm joints and one normalized gripper command
per arm. Gripper commands are constant within one phase. Source timestamps are
preserved: only sampling onto the control clock changes. Targets are sampled at
the END of each control interval, with endpoint holding after an arm finishes.
Run this file directly with --self-test; only NumPy and the CPU are required.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _frozen_array(value: Any) -> FloatArray:
    """Own a finite float64 copy so callers cannot change queued commands."""
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("Trajectory arrays must contain finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class ArmTrajectory:
    """A phase trajectory in controller joint order; a single t=0 point holds."""

    action_key: str
    joint_names: tuple[str, ...]
    times_s: FloatArray
    positions_rad: FloatArray
    gripper_command: float

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        times = _frozen_array(self.times_s)
        positions = _frozen_array(self.positions_rad)
        grip = float(self.gripper_command)
        if not isinstance(self.action_key, str) or not self.action_key:
            raise ValueError("An explicit robot action key is required")
        if len(names) != 7 or len(set(names)) != 7 or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("Specify seven unique arm joint names in controller order")
        if times.ndim != 1 or len(times) == 0 or times[0] != 0.0:
            raise ValueError("Timestamps must be nonempty and start at zero seconds")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps must increase strictly")
        if positions.shape != (len(times), 7):
            raise ValueError("Joint positions must have shape [timestamps, 7]")
        if not math.isfinite(grip) or not -1.0 <= grip <= 1.0:
            raise ValueError("Gripper command must be finite and normalized to [-1, 1]")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "positions_rad", positions)
        object.__setattr__(self, "gripper_command", grip)

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])

    @classmethod
    def from_planner_result(
        cls, result: Any, *, action_key: str,
        planner_joint_names: Sequence[str], controller_joint_names: Sequence[str],
        gripper_command: float,
    ) -> ArmTrajectory:
        """Accept a successful dry-run result only after explicit order matching.

        Extract planner names in result.position column order; controller names
        are the seven arm joints, excluding the two physical finger joints.
        The official solver may return -1 on failure; that is rejected here.
        """
        if not isinstance(result, Mapping) or result.get("status") != "Success":
            raise ValueError("Planner did not return a successful trajectory")
        if tuple(planner_joint_names) != tuple(controller_joint_names):
            raise ValueError("Planner and controller joint order differ")
        if any(key not in result for key in ("time", "position", "duration")):
            raise ValueError("Planner result must include time, position and duration")
        trajectory = cls(action_key, tuple(controller_joint_names), result["time"],
                         result["position"], gripper_command)
        duration = float(result["duration"])
        if not math.isfinite(duration) or not math.isclose(
            duration, trajectory.duration_s, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("Planner duration and last timestamp disagree")
        return trajectory


def plan_pose(
    solver: Any, pose: Any, *, action_key: str,
    controller_joint_names: Sequence[str],
) -> ArmTrajectory:
    """Obtain a PandaArmMotionPlanningSolver trajectory without following it.

    The caller supplies the seven actual arm-controller joint names in action
    order. The native planner's output order is extracted from its move group;
    mismatches are rejected before asking the solver to plan. This adapter
    preserves the current phase gripper command rather than opening the hand.
    """
    if solver.env_agent.control_mode != "pd_joint_pos":
        raise ValueError("This adapter supports pd_joint_pos only")
    planner = solver.planner
    planner_names = tuple(planner.user_joint_names[index]
                          for index in planner.move_group_joint_indices)
    if planner_names != tuple(controller_joint_names):
        raise ValueError("Planner and controller joint order differ")
    # Native screw Success describes a linearized twist residual; verify its
    # actual TCP endpoint before passing the path to paired scene validation.
    if __package__:
        from .planning_endpoint import checked_screw_plan
    else:  # Preserve the direct CPU self-test command.
        from planning_endpoint import checked_screw_plan
    result = checked_screw_plan(solver, pose)
    return ArmTrajectory.from_planner_result(
        result, action_key=action_key, planner_joint_names=planner_names,
        controller_joint_names=controller_joint_names,
        gripper_command=solver.gripper_state,
    )


@dataclass(frozen=True, eq=False)
class DualArmSchedule:
    """Both arms' targets on one control clock, ready for scene validation."""

    trajectories: tuple[ArmTrajectory, ArmTrajectory]
    control_timestep_s: float
    target_times_s: FloatArray
    left_commands: FloatArray
    right_commands: FloatArray

    @property
    def num_steps(self) -> int:
        return len(self.target_times_s)

    def iter_actions(self) -> Iterator[dict[str, FloatArray]]:
        """Yield one fresh two-arm action dictionary per control interval."""
        left, right = self.trajectories
        for index in range(self.num_steps):
            yield {left.action_key: self.left_commands[index].copy(),
                   right.action_key: self.right_commands[index].copy()}


def build_hold_schedule(
    left: ArmTrajectory, right: ArmTrajectory, *, control_timestep_s: float,
    max_steps: int = 10_000,
) -> DualArmSchedule:
    """Resample by actual timestamps without slowing either arm to match the other.

    A command for interval [k*dt, (k+1)*dt] targets its endpoint. An arm that has
    finished holds its final joint target AND its phase gripper command. The
    phase ends after ceil(max_duration/dt) intervals, at most one tick later
    than the longer source trajectory. Two t=0-only holds need no intervals.
    A gripper-close/open dwell needs a positive-duration stationary trajectory;
    a zero-duration phase emits no commands, including gripper commands.
    Linear interpolation changes the sampled path; validate the resulting
    combined commands and interpolated segments before executing them.
    """
    dt = float(control_timestep_s)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Control timestep must be finite and positive")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if left.action_key == right.action_key:
        raise ValueError("The two arms must use distinct action keys")
    horizon = max(left.duration_s, right.duration_s)
    ratio = horizon / dt
    if not math.isfinite(ratio) or ratio > max_steps:
        raise ValueError("Phase exceeds the configured step budget")
    count = math.ceil(ratio)
    times = _frozen_array(dt * np.arange(1, count + 1, dtype=np.float64))

    def sample(trajectory: ArmTrajectory) -> FloatArray:
        joints = np.column_stack([
            np.interp(times, trajectory.times_s, trajectory.positions_rad[:, joint])
            for joint in range(7)
        ])
        return _frozen_array(np.column_stack([
            joints, np.full(count, trajectory.gripper_command)
        ]))

    return DualArmSchedule((left, right), dt, times, sample(left), sample(right))


def self_test() -> dict[str, Any]:
    """Check A against analytical targets, including irregular source timestamps."""
    names = tuple(f"joint_{index}" for index in range(1, 8))
    offset = np.arange(7, dtype=np.float64) / 10 + 0.2

    def trajectory(key: str, times: Sequence[float], values: Sequence[float],
                   grip: float) -> ArmTrajectory:
        positions = np.asarray(values)[:, None] + offset[None, :]
        return ArmTrajectory(key, names, np.asarray(times), positions, grip)

    def reject(operation: Any) -> None:
        try:
            operation()
        except (ValueError, TypeError):
            return
        raise AssertionError("Malformed trajectory/schedule was accepted")

    # Illustrative control dt=1 second; source durations remain 3 and 5 seconds.
    left = trajectory("left", [0, 3], [0, 3], -1)
    right = trajectory("right", [0, 2, 5], [0, 2, 5], 1)
    schedule = build_hold_schedule(left, right, control_timestep_s=1)
    assert schedule.num_steps == 5
    assert np.array_equal(schedule.target_times_s, [1, 2, 3, 4, 5])
    assert np.allclose(schedule.left_commands[:, :7],
                       np.array([1, 2, 3, 3, 3])[:, None] + offset)
    assert np.allclose(schedule.right_commands[:, :7],
                       np.arange(1, 6)[:, None] + offset)
    assert np.array_equal(schedule.left_commands[:, 7], [-1] * 5)
    frames = list(schedule.iter_actions())
    assert len(frames) == 5 and all(set(frame) == {"left", "right"} for frame in frames)
    assert all(value.shape == (8,) for frame in frames for value in frame.values())
    frames[0]["left"][:] = 0  # A caller cannot mutate queued or later commands.
    assert schedule.left_commands[0, 0] != 0 and frames[1]["left"][0] != 0

    irregular = trajectory("left", [0, 0.3, 1.3], [0, 0.6, 1.6], -1)
    stationary = trajectory("right", [0], [2], -1)
    uneven = build_hold_schedule(irregular, stationary, control_timestep_s=0.5)
    assert np.allclose(uneven.left_commands[:, 0], [1.0, 1.5, 1.8])
    assert np.allclose(uneven.right_commands[:, :7], 2 + offset)
    assert np.allclose(uneven.target_times_s, [0.5, 1.0, 1.5])
    assert uneven.trajectories[0].duration_s == 1.3
    zero = build_hold_schedule(trajectory("left", [0], [1], 1), stationary,
                               control_timestep_s=0.05)
    assert zero.num_steps == 0 and list(zero.iter_actions()) == []
    assert zero.left_commands.shape == (0, 8)
    swapped = build_hold_schedule(right, left, control_timestep_s=1)
    assert np.array_equal(swapped.left_commands, schedule.right_commands)
    assert np.array_equal(swapped.right_commands, schedule.left_commands)

    source = {"status": "Success", "time": [0, 0.3, 1.3],
              "position": irregular.positions_rad.copy(), "duration": 1.3}
    from types import SimpleNamespace

    class DryRunFixture:
        """Synthetic solver with the native fields used by the adapter."""

        def __init__(self) -> None:
            self.env_agent = SimpleNamespace(control_mode="pd_joint_pos")
            self.base_env = SimpleNamespace(control_timestep=.05)
            self.robot = SimpleNamespace(get_qpos=lambda: irregular.positions_rad[:1].copy())
            self.planner = SimpleNamespace(user_joint_names=list(names),
                move_group_joint_indices=list(range(7)), TOPP=lambda *args: None,
                transform_goal_to_wrt_base=lambda pose: pose.copy(), move_group="tcp",
                link_name_2_idx={"tcp": 0},
                pinocchio_model=SimpleNamespace(compute_forward_kinematics=lambda q: None,
                    get_link_pose=lambda _: np.array([0., 0., 0., 1., 0., 0., 0.])))
            self.gripper_state = -1
            self.calls = 0

        def move_to_pose_with_screw(self, pose: Any, *, dry_run: bool = False) -> Any:
            assert np.array_equal(pose, fixture_pose) and dry_run is True
            self.calls += 1
            self.planner.TOPP(irregular.positions_rad.copy(), .05)
            return source

    fixture_pose = np.array([0., 0., 0., 1., 0., 0., 0.])
    solver = DryRunFixture()
    planned = plan_pose(solver, fixture_pose, action_key="left",
                        controller_joint_names=names)
    assert solver.calls == 1 and planned.gripper_command == -1
    reject(lambda: plan_pose(solver, fixture_pose, action_key="left",
                             controller_joint_names=names[::-1]))
    assert solver.calls == 1
    solver.env_agent.control_mode = "pd_joint_delta_pos"
    reject(lambda: plan_pose(solver, fixture_pose, action_key="left",
                             controller_joint_names=names))
    assert solver.calls == 1
    imported = ArmTrajectory.from_planner_result(
        source, action_key="left", planner_joint_names=names,
        controller_joint_names=names, gripper_command=-1)
    source["position"][:] = 99
    assert np.array_equal(imported.positions_rad, irregular.positions_rad)
    for bad_result in (-1, {"status": "failed"},
                       {**source, "duration": 2}, {"status": "Success"}):
        reject(lambda: ArmTrajectory.from_planner_result(
            bad_result, action_key="left", planner_joint_names=names,
            controller_joint_names=names, gripper_command=-1))
    reject(lambda: ArmTrajectory.from_planner_result(
        source, action_key="left", planner_joint_names=names[::-1],
        controller_joint_names=names, gripper_command=-1))
    for times, values in [([], []), ([1, 2], [0, 1]), ([0, 0], [0, 1]),
                          ([0, 1], [0, float("nan")]),
                          ([0, float("inf")], [0, 1])]:
        reject(lambda: trajectory("left", times, values, 1))
    reject(lambda: ArmTrajectory("left", names, [0, 1], np.zeros((2, 6)), 1))
    reject(lambda: trajectory("left", [0, 1], [0, 1], 2))
    for dt in (0, -1, float("nan"), float("inf")):
        reject(lambda: build_hold_schedule(left, right, control_timestep_s=dt))
    reject(lambda: build_hold_schedule(left, right, control_timestep_s=1, max_steps=4))
    reject(lambda: build_hold_schedule(left, left, control_timestep_s=1))
    return {"status": "passed", "scope": "CPU command scheduling only",
            "policy": "simultaneous_start_independent_completion_endpoint_hold",
            "example": {"target_times_s": schedule.target_times_s.tolist(),
                        "left_first_joint_rad": schedule.left_commands[:, 0].tolist(),
                        "right_first_joint_rad": schedule.right_commands[:, 0].tolist(),
                        "left_gripper_command": schedule.left_commands[:, 7].tolist(),
                        "left_duration_s": left.duration_s, "right_duration_s": right.duration_s},
            "checks": ["endpoint and CLOSED gripper hold", "same-tick two-arm actions",
                       "irregular timestamps", "fractional final interval", "stationary arm",
                       "zero-duration phase", "left-right symmetry", "input/output isolation",
                       "dry-run solver adapter", "planner failure and duration", "joint order", "invalid arrays and clock",
                       "step budget"],
            "environment_steps": 0, "gpu_used": False,
            "collision_safety_validated": False, "physical_execution_validated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
