from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any
import numpy as np
from mani_skill.examples.motionplanning.dual.sync.trajectory_schedule import (
    ArmTrajectory, plan_pose, build_hold_schedule
)


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


def test_runtime_contract():
    assert self_test()["status"] == "passed"
