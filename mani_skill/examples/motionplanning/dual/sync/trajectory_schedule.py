"""Option A: start both arm trajectories together and hold each endpoint.

This module prepares joint-position commands; it never advances an environment.
It is not a collision checker or evidence of reaching/grasping in simulation.
Before live execution, a separate adapter must validate current joint states,
joint limits and the combined trajectory against the actual scene.

Contract: seconds, radians, named arm joints and one normalized gripper command
per arm. Gripper commands are constant within one phase. Source timestamps are
preserved: only sampling onto the control clock changes. Targets are sampled at
the END of each control interval, with endpoint holding after an arm finishes.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
        if not names or len(set(names)) != len(names) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("Specify unique arm joint names in controller order")
        if times.ndim != 1 or len(times) == 0 or times[0] != 0.0:
            raise ValueError("Timestamps must be nonempty and start at zero seconds")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps must increase strictly")
        if positions.shape != (len(times), len(names)):
            raise ValueError("Joint positions must have shape [timestamps, arm joints]")
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
        are the arm joints, excluding physical finger joints.
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
    """Obtain a native arm-planner trajectory without following it.

    The caller supplies the actual arm-controller joint names in action
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
    from .planning_endpoint import checked_screw_plan
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
            for joint in range(len(trajectory.joint_names))
        ])
        return _frozen_array(np.column_stack([
            joints, np.full(count, trajectory.gripper_command)
        ]))

    return DualArmSchedule((left, right), dt, times, sample(left), sample(right))
