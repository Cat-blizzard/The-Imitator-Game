"""Sample both Panda arms in one independent collision world, without stepping.

Finger geometry stays at the measured factory snapshot. This does not validate
opening/closing, carried objects, dynamics, or a complete scene. Only explicitly
supplied normal objects join the two robots. Finite sampling is not continuous
collision detection. MPLib is imported only by the native factory; --self-test
uses NumPy and an analytical checker on the CPU.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Joint states, poses and times must be finite")
    return result


def _vector(value: Any, size: int) -> np.ndarray:
    result = _array(value)
    if result.shape != (size,):
        raise ValueError(f"Expected shape ({size},), got {result.shape}")
    return result


def _single(value: Any, size: int) -> np.ndarray:
    result = _array(value)
    if result.shape == (1, size):
        result = result[0].copy()
    return _vector(result, size)


class CollisionChecker(Protocol):
    def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]: ...


class NativeDualCollisionChecker:
    """Private model clones; .models and .world remain available for native tests."""

    def __init__(self, models: Sequence[Any], world: Any,
                 full_qpos: Sequence[np.ndarray], arm_indices: Sequence[np.ndarray],
                 joint_limits: Sequence[np.ndarray], object_names: Sequence[str]) -> None:
        self.models = tuple(models)
        self.world = world
        self.full_qpos = tuple(q.copy() for q in full_qpos)
        self.arm_indices = tuple(i.copy() for i in arm_indices)
        self.joint_limits = tuple(q.copy() for q in joint_limits)
        self.normal_object_names = tuple(object_names)
        self.scope = "dual_arm_self_and_mutual_with_explicit_normal_objects"

    @classmethod
    def from_solvers(cls, left_solver: Any, right_solver: Any, *,
                     normal_objects: Sequence[Any] = (),
                     normal_object_names: Sequence[str] = ()) -> NativeDualCollisionChecker:
        """Clone actual Panda models, joint orders, base poses and finger positions.

        The two other active joints must be the Panda's physical finger joints.
        The normalized gripper command is deliberately never used as a qpos.
        """
        names = tuple(normal_object_names)
        objects = tuple(normal_objects)
        if (len(objects) != len(names) or len(set(names)) != len(names)
                or any(not isinstance(n, str) or not n or n in ("left", "right") for n in names)):
            raise ValueError("Normal objects need distinct, nonempty names")
        from mplib.pymp import ArticulatedModel, PlanningWorld
        from mplib.pymp.collision_detection import fcl
        if any(isinstance(obj.get_collision_geometry(), fcl.Plane) for obj in objects):
            raise ValueError("MPLib 0.1.1 Panda/Plane collision crashes; use an explicit halfspace guard")

        models, states, indices, limits = [], [], [], []
        for solver in (left_solver, right_solver):
            planner, agent = solver.planner, solver.env_agent
            joint_names = tuple(planner.user_joint_names)
            active_names = tuple(j.get_name() for j in solver.robot.get_active_joints())
            arm_names = tuple(agent.controller.controllers["arm"].config.joint_names)
            if (len(joint_names) != 9 or len(set(joint_names)) != 9
                    or len(arm_names) != 7 or len(set(arm_names)) != 7
                    or set(active_names) != set(joint_names)
                    or not set(arm_names).issubset(joint_names)
                    or set(joint_names) - set(arm_names) != set(agent.gripper_joint_names)):
                raise ValueError("Expected seven arm joints and two physical Panda finger joints")
            if not Path(planner.urdf).is_file() or not Path(planner.srdf).is_file():
                raise ValueError("Existing URDF and SRDF are required; no files are generated")
            measured = _single(solver.robot.get_qpos(), len(active_names))
            full = measured[[active_names.index(n) for n in joint_names]]
            pose = solver.robot.pose
            base = _single(pose.raw_pose, 7) if hasattr(pose, "raw_pose") else np.r_[
                _single(pose.p, 3), _single(pose.q, 4)]
            if not math.isclose(float(np.linalg.norm(base[3:])), 1.0, abs_tol=1e-5):
                raise ValueError("Base quaternion must be normalized (w,x,y,z)")
            model = ArticulatedModel(planner.urdf, planner.srdf, [0, 0, -9.81],
                                     list(planner.user_link_names), list(joint_names),
                                     convex=True, verbose=False)
            model.set_move_group(planner.move_group)
            model.set_base_pose(base)
            model.set_qpos(full, True)
            arm = np.array([joint_names.index(n) for n in arm_names], dtype=int)
            qlimits = _array(np.concatenate(model.get_pinocchio_model().get_joint_limits()))
            if qlimits.shape != (9, 2):
                raise ValueError("Unexpected native Panda joint-limit layout")
            models.append(model)
            states.append(full)
            indices.append(arm)
            limits.append(qlimits[arm])
        world = PlanningWorld(models, ["left", "right"], list(objects), list(names))
        return cls(models, world, states, indices, limits, names)

    def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]:
        """Set BOTH full states before querying either model; never touch an env."""
        joints = (_vector(left7, 7), _vector(right7, 7))
        for q, bounds in zip(joints, self.joint_limits):
            if np.any(q < bounds[:, 0] - 1e-6) or np.any(q > bounds[:, 1] + 1e-6):
                raise ValueError("Arm qpos violates native joint limits")
        for model, initial, arm, q in zip(self.models, self.full_qpos, self.arm_indices, joints):
            full = initial.copy()
            full[arm] = q
            model.set_qpos(full, True)
        records, seen = [], set()
        for index in (0, 1):
            for collision in self.world.collide_full(index):
                record = {key: str(getattr(collision, key)) for key in (
                    "collision_type", "object_name1", "link_name1", "object_name2", "link_name2")}
                pair = sorted(((record["object_name1"], record["link_name1"]),
                               (record["object_name2"], record["link_name2"])))
                identity = (record["collision_type"], *pair)
                if identity not in seen:
                    seen.add(identity)
                    records.append(record)
        return records


def validate_schedule(schedule: Any, start_left7: Any, start_right7: Any,
                      checker: CollisionChecker, *, max_joint_step_rad: float = 0.02,
                      max_samples: int = 20_000) -> dict[str, Any]:
    """Check the executed command polygon, including t=0, with a shared alpha.

    Starts use controller order and must match the source trajectories within
    1e-6 rad. Reject the entire sampling budget before invoking the checker.
    This does not establish physical tracking between controller targets.
    """
    step = float(max_joint_step_rad)
    if (not math.isfinite(step) or step <= 0 or isinstance(max_samples, bool)
            or not isinstance(max_samples, int) or max_samples < 1):
        raise ValueError("Positive finite resolution and positive integer budget required")
    starts = (_vector(start_left7, 7), _vector(start_right7, 7))
    times = _array(schedule.target_times_s)
    commands = (_array(schedule.left_commands), _array(schedule.right_commands))
    dt = float(schedule.control_timestep_s)
    if (times.ndim != 1 or not math.isfinite(dt) or dt <= 0
            or np.any(np.diff(times) <= 0) or np.any(times <= 0)
            or not np.allclose(times, dt * np.arange(1, len(times) + 1), rtol=0, atol=1e-9)
            or any(c.shape != (len(times), 8) for c in commands)
            or len(schedule.trajectories) != 2):
        raise ValueError("Expected two [control ticks,8] arrays on one positive control clock")
    for source, start, command in zip(schedule.trajectories, starts, commands):
        if not np.allclose(_vector(source.positions_rad[0], 7), start, rtol=0, atol=1e-6):
            raise ValueError("Measured start and source trajectory start differ")
        if (not math.isfinite(float(source.gripper_command))
                or abs(float(source.gripper_command)) > 1
                or np.any(command[:, 7] != source.gripper_command)):
            raise ValueError("Each phase must preserve a finite normalized gripper command")
    positions = np.vstack([np.r_[*starts], np.column_stack([c[:, :7] for c in commands])])
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.max(np.abs(np.diff(positions, axis=0)), axis=1) / step
    if not np.isfinite(distances).all() or np.any(distances > max_samples):
        raise ValueError("Interpolation exceeds the sampling budget")
    subdivisions = [max(1, math.ceil(float(d))) for d in distances]
    required = 1 + sum(subdivisions)
    if required > max_samples:
        raise ValueError(f"Interpolation needs {required} samples, budget is {max_samples}")
    report: dict[str, Any] = {
        "status": "passed", "sample_count": 0, "planned_sample_count": required,
        "max_joint_step_rad": step, "max_observed_joint_step_rad": 0.0,
        "first_collision": None, "scope": getattr(checker, "scope", "provided_checker"),
        "continuous_collision_detection": False, "physical_tracking_validated": False,
        "finger_geometry": "fixed; gripper motion and carried objects are not validated",
    }

    def inspect(q: np.ndarray, time_s: float, interval: int, alpha: float) -> bool:
        collisions = checker.check(q[:7].copy(), q[7:].copy())
        report["sample_count"] += 1
        if collisions:
            report.update(status="collision", first_collision={
                "time_s": time_s, "control_interval": interval, "alpha": alpha,
                "left_qpos_rad": q[:7].tolist(), "right_qpos_rad": q[7:].tolist(),
                "collisions": collisions})
        return bool(collisions)

    if inspect(positions[0], 0.0, 0, 0.0):
        return report
    previous = positions[0]
    for i, count in enumerate(subdivisions):
        initial_time = 0.0 if i == 0 else float(times[i - 1])
        for j in range(1, count + 1):
            alpha = j / count
            q = (1.0 - alpha) * positions[i] + alpha * positions[i + 1]
            report["max_observed_joint_step_rad"] = max(report["max_observed_joint_step_rad"],
                                                       float(np.max(np.abs(q - previous))))
            time_s = initial_time + alpha * (float(times[i]) - initial_time)
            if inspect(q, time_s, i + 1, alpha):
                return report
            previous = q
    return report


def self_test() -> dict[str, Any]:
    try:
        from .trajectory_schedule import ArmTrajectory, build_hold_schedule
    except ImportError:
        from trajectory_schedule import ArmTrajectory, build_hold_schedule
    from dataclasses import replace

    names = tuple(f"joint{i}" for i in range(7))

    def trajectory(key: str, start: float, end: float, duration: float) -> Any:
        q = np.zeros((2, 7))
        q[:, 0] = [start, end]
        return ArmTrajectory(key, names, [0, duration], q, -1)

    class AnalyticalChecker:
        def __init__(self, detect: bool) -> None:
            self.detect, self.samples = detect, []

        def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]:
            self.samples.append(np.r_[left7, right7])
            return [{"collision_type": "analytical_mutual", "link_name1": "left",
                     "link_name2": "right"}] if self.detect and abs(left7[0] - right7[0]) < 0.01 else []

    left, right = trajectory("left", 0, 1, 1), trajectory("right", 1, 0, 1)
    schedule = build_hold_schedule(left, right, control_timestep_s=1)
    checker = AnalyticalChecker(True)
    assert checker.check(left.positions_rad[0], right.positions_rad[0]) == []
    assert checker.check(left.positions_rad[-1], right.positions_rad[-1]) == []
    checker.samples.clear()
    collision = validate_schedule(schedule, left.positions_rad[0], right.positions_rad[0],
                                  checker, max_joint_step_rad=0.125)
    assert collision["status"] == "collision" and collision["first_collision"]["time_s"] == 0.5
    assert collision["sample_count"] == 5
    assert all(abs(q[0] + q[7] - 1) < 1e-12 for q in checker.samples)
    right = trajectory("right", 2, 0, 2)
    schedule = build_hold_schedule(left, right, control_timestep_s=1)
    clear = AnalyticalChecker(False)
    passed = validate_schedule(schedule, left.positions_rad[0], right.positions_rad[0],
                               clear, max_joint_step_rad=0.125)
    assert passed["status"] == "passed" and passed["sample_count"] == 17
    assert passed["max_observed_joint_step_rad"] <= 0.125
    for time_s, q in zip(np.arange(17) / 8, clear.samples):
        assert np.isclose(q[0], min(time_s, 1)) and np.isclose(q[7], 2 - time_s)
    hold = ArmTrajectory("left", names, [0], [np.zeros(7)], -1)
    zero = build_hold_schedule(hold, replace(hold, action_key="right"), control_timestep_s=0.1)
    assert validate_schedule(zero, np.zeros(7), np.zeros(7), clear)["sample_count"] == 1

    def reject(**overrides: Any) -> None:
        args = dict(schedule=schedule, start_left7=left.positions_rad[0],
                    start_right7=right.positions_rad[0], checker=clear)
        args.update(overrides)
        before = len(clear.samples)
        try:
            validate_schedule(**args)
        except ValueError:
            assert len(clear.samples) == before
            return
        raise AssertionError("Invalid schedule/state/budget was accepted")

    reject(start_left7=np.zeros(6))
    reject(start_left7=np.full(7, np.nan))
    reject(start_left7=np.ones(7))
    reject(max_samples=3)
    reject(max_samples=True)
    reject(max_joint_step_rad=0)
    reject(max_joint_step_rad=float("inf"))
    reject(schedule=replace(schedule, left_commands=np.zeros((2, 7))))
    reject(schedule=replace(schedule, right_commands=np.full((2, 8), np.inf)))
    reject(schedule=replace(schedule, target_times_s=[1, 1]))
    reject(schedule=replace(schedule, target_times_s=[1, 3]))
    return {"status": "passed", "endpoint_clear_midpoint_collision": collision,
            "independent_completion": passed, "invalid_input_checks": 11,
            "environment_steps": 0, "native_mplib_tested": False, "gpu_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
