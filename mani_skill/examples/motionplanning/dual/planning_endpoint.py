"""Check MPLib screw-plan TCP endpoints and refine them without physics.

The TIG screw planner stops on a linearized twist residual. A ``Success`` is
therefore checked against actual Pinocchio FK. Short native screw corrections
are appended to the original geometric path and parameterized together by the
same native TOPP implementation. Scene/payload checks remain the caller's duty.
Poses are XYZ/WXYZ, distances metres, angles radians, and joint order is explicit.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


POSITION_TOLERANCE_M = 5e-4
ROTATION_TOLERANCE_RAD = 1e-3
MAX_CORRECTIONS = 3


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Endpoint inputs must be finite")
    return result


def _pose(value: Any) -> np.ndarray:
    if hasattr(value, "p") and hasattr(value, "q"):
        result = np.r_[_array(value.p).reshape(-1), _array(value.q).reshape(-1)]
    else:
        result = _array(value)
    if result.shape != (7,) or not np.isclose(
            np.linalg.norm(result[3:]), 1., atol=1e-5, rtol=0):
        raise ValueError("Expected one normalized XYZ/WXYZ TCP pose")
    result[3:] /= np.linalg.norm(result[3:])
    return result


def pose_error(actual: Any, expected: Any) -> dict[str, float]:
    """Quaternion-sign invariant TCP error in a shared coordinate frame."""
    actual, expected = _pose(actual), _pose(expected)
    dot = abs(float(np.dot(actual[3:], expected[3:])))
    return {"position_m": float(np.linalg.norm(actual[:3] - expected[:3])),
            "rotation_rad": float(2 * np.arccos(np.clip(dot, 0., 1.)))}


def checked_screw_plan(solver: Any, pose: Any, *,
                       position_tolerance_m: float = POSITION_TOLERANCE_M,
                       rotation_tolerance_rad: float = ROTATION_TOLERANCE_RAD,
                       max_corrections: int = MAX_CORRECTIONS) -> dict[str, Any]:
    """Return a native dry-run result whose *sampled* endpoint meets FK bounds.

    Only this planner instance's TOPP method is temporarily wrapped, forwarding
    every call to any existing instrumentation. Corrections use the current
    full measured joint configuration, replacing only move-group columns; the
    two finger joints are never inferred from a gripper command. No environment
    method is called. Existing audit caches are updated to the final result and
    the last TOPP invocation describes its complete geometric source path.
    """
    if (not np.isfinite([position_tolerance_m, rotation_tolerance_rad]).all()
            or position_tolerance_m <= 0 or rotation_tolerance_rad <= 0
            or isinstance(max_corrections, bool) or not isinstance(max_corrections, int)
            or max_corrections < 0):
        raise ValueError("Endpoint bounds must be positive and correction count nonnegative")
    planner = solver.planner
    indices = np.asarray(planner.move_group_joint_indices, dtype=np.int64)
    start = _array(solver.robot.get_qpos())
    if start.ndim == 2 and start.shape[0] == 1:
        start = start[0]
    if (start.ndim != 1 or len(start) != len(planner.user_joint_names)
            or indices.shape != (7,) or len(set(indices.tolist())) != 7
            or np.any(indices < 0) or np.any(indices >= len(start))):
        raise ValueError("Expected full measured joints and seven unique move-group columns")
    goal_world = _pose(pose)
    goal_base = _pose(planner.transform_goal_to_wrt_base(goal_world))
    timestep = float(solver.base_env.control_timestep)
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("Control timestep must be positive seconds")
    native_topp = planner.TOPP
    captured_paths: list[np.ndarray] = []

    def capture_topp(path: Any, step: float = .1, verbose: bool = False) -> Any:
        path = _array(path)
        if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
            raise ValueError("Native TOPP source must have shape [N>=2, 7]")
        captured_paths.append(path)
        return native_topp(path, step, verbose)

    def fk(arm_qpos: Any) -> np.ndarray:
        qpos = start.copy()
        arm_qpos = _array(arm_qpos)
        if arm_qpos.shape != (7,):
            raise ValueError("Expected seven arm joints at the endpoint")
        qpos[indices] = arm_qpos
        planner.pinocchio_model.compute_forward_kinematics(qpos)
        return _pose(planner.pinocchio_model.get_link_pose(
            planner.link_name_2_idx[planner.move_group]))

    def within(error: dict[str, float]) -> bool:
        return (error["position_m"] <= position_tolerance_m
                and error["rotation_rad"] <= rotation_tolerance_rad)

    audit: dict[str, Any] = {"policy": "mplib-actual-fk-endpoint-v1",
        "position_tolerance_m": position_tolerance_m,
        "rotation_tolerance_rad": rotation_tolerance_rad,
        "max_corrections": max_corrections,
        "goal_world_xyz_wxyz": goal_world.tolist(),
        "goal_base_xyz_wxyz": goal_base.tolist(),
        "fk_link": planner.move_group,
        "full_joint_names": list(planner.user_joint_names),
        "move_group_joint_indices": indices.tolist(),
        "preserved_non_arm_joint_positions": {
            name: float(start[i]) for i, name in enumerate(planner.user_joint_names)
            if i not in indices},
        "attempts": [], "environment_steps": 0,
        "collision_and_payload_checks_replaced": False}
    solver.last_endpoint_audit = audit
    planner.TOPP = capture_topp
    try:
        result = solver.move_to_pose_with_screw(pose, dry_run=True)
        if not isinstance(result, Mapping) or result.get("status") != "Success":
            raise ValueError("Planner did not return a successful trajectory")
        positions = _array(result["position"])
        if positions.ndim != 2 or positions.shape[1] != 7 or not len(positions):
            raise ValueError("Successful native plan has no sampled joint endpoint")
        if not captured_paths:
            raise ValueError("Native screw plan did not expose its TOPP geometric path")
        combined_path = captured_paths[-1].copy()
        if not np.allclose(combined_path[0], start[indices], atol=1e-6, rtol=0):
            raise ValueError("Native geometric path does not start at measured joints")
        actual = fk(positions[-1])
        error = pose_error(actual, goal_base)
        audit["attempts"].append({"kind": "original_native_screw", **error})
        corrections = 0
        while not within(error) and corrections < max_corrections:
            full_qpos = start.copy()
            full_qpos[indices] = combined_path[-1]
            count = len(captured_paths)
            correction = planner.plan_screw(goal_world.copy(), full_qpos,
                time_step=timestep, use_point_cloud=bool(solver.use_point_cloud),
                use_attach=bool(getattr(planner.planning_world, "use_attach", False)))
            if (not isinstance(correction, Mapping) or correction.get("status") != "Success"
                    or len(captured_paths) == count):
                raise ValueError("Native endpoint correction failed")
            path = captured_paths[-1]
            if not np.allclose(path[0], combined_path[-1], atol=1e-9, rtol=0):
                raise ValueError("Native correction does not continue the original path")
            combined_path = np.vstack((combined_path, path[1:]))
            actual = fk(combined_path[-1])
            error = pose_error(actual, goal_base)
            corrections += 1
            audit["attempts"].append({"kind": "native_screw_correction",
                "correction_index": corrections, **error})
        if not within(error):
            raise ValueError(f"Actual FK endpoint exceeds contract after {corrections} corrections: {error}")
        if corrections:
            raw = planner.TOPP(combined_path, timestep)
            if len(raw) != 5 or any(value is None for value in raw):
                raise ValueError("Native TOPP failed for the corrected geometric path")
            result = dict(zip(("time", "position", "velocity", "acceleration", "duration"), raw))
            result["status"] = "Success"
        else:
            result = dict(result)
        positions = _array(result["position"])
        if positions.ndim != 2 or positions.shape[1] != 7 or not len(positions):
            raise ValueError("Native corrected TOPP has no sampled joint endpoint")
        actual = fk(positions[-1])
        final_error = pose_error(actual, goal_base)
        if not within(final_error):
            raise ValueError(f"Time-parameterized FK endpoint exceeds contract: {final_error}")
        if not np.allclose(positions[0], start[indices], atol=1e-6, rtol=0):
            raise ValueError("Corrected TOPP changed the measured start")
        audit.update(status="passed", corrections=corrections,
            endpoint_base_xyz_wxyz=actual.tolist(), final_error=final_error,
            geometric_path_samples=len(combined_path),
            native_topp_calls=len(captured_paths),
            parameterization="native TOPP over complete corrected geometric path" if corrections
                else "original native TOPP unchanged")
        result["endpoint_audit"] = audit
        if hasattr(solver, "last_native_result"):
            solver.last_native_result = result
        return result
    except BaseException as exc:
        audit.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        planner.TOPP = native_topp


def self_test() -> dict[str, Any]:
    """Exercise correction, frame/order handling and failure without native libs."""
    from types import SimpleNamespace

    class Fixture:
        def __init__(self, bias: float) -> None:
            self.indices = np.array([2, 0, 1, 4, 3, 5, 6])
            self.state = np.array([0., 0., 0., 0., 0., 0., 0., .031, .027])
            self.bias, self.calls, self.topp_calls = bias, 0, []
            self.env_agent = SimpleNamespace(control_mode="pd_joint_pos")
            self.base_env = SimpleNamespace(control_timestep=.05)
            self.robot = SimpleNamespace(get_qpos=lambda: self.state[None].copy())
            self.gripper_state, self.use_point_cloud = -1, True
            self.last_native_result = None
            self.planner = SimpleNamespace(user_joint_names=[f"j{i}" for i in range(9)],
                move_group_joint_indices=self.indices.tolist(), move_group="tcp",
                link_name_2_idx={"tcp": 0}, planning_world=SimpleNamespace(use_attach=True),
                transform_goal_to_wrt_base=lambda pose: np.r_[pose[:3] - [1., 2., 3.], pose[3:]],
                TOPP=self.topp, plan_screw=self.screw)
            self.planner.pinocchio_model = SimpleNamespace(
                compute_forward_kinematics=self.fk, get_link_pose=lambda _: self.actual)

        def fk(self, full: np.ndarray) -> None:
            assert np.array_equal(full[7:], self.state[7:])
            self.actual = np.r_[full[self.indices][:3], 1., 0., 0., 0.]

        def topp(self, path: np.ndarray, step: float, verbose: bool = False) -> Any:
            self.topp_calls.append(path.copy())
            self.last_native_parameterization = path.copy()
            return np.linspace(0., .1, len(path)), path.copy(), path * 0, path * 0, .1

        def screw(self, goal: np.ndarray, full: np.ndarray, **kwargs: Any) -> Any:
            self.calls += 1
            assert np.array_equal(full[7:], self.state[7:])
            if self.calls > 1:
                assert kwargs["use_point_cloud"] and kwargs["use_attach"]
            end = full[self.indices].copy()
            end[:3] = goal[:3] - [1., 2., 3.]
            if self.calls == 1:
                end[0] -= self.bias
            raw = self.planner.TOPP(np.vstack((full[self.indices], end)), .05)
            result = dict(zip(("time", "position", "velocity", "acceleration", "duration"), raw))
            return {"status": "Success", **result}

        def move_to_pose_with_screw(self, pose: Any, *, dry_run: bool) -> Any:
            assert dry_run
            return self.screw(_pose(pose), self.state.copy())

    goal = np.array([1.1, 2.2, 3.3, 1., 0., 0., 0.])
    solver = Fixture(.014)
    state = solver.state.copy()
    original_topp = solver.planner.TOPP
    result = checked_screw_plan(solver, goal)
    assert solver.planner.TOPP == original_topp and solver.calls == 2
    assert solver.last_native_result is result
    assert np.array_equal(solver.last_native_parameterization, solver.topp_calls[-1])
    assert len(solver.topp_calls[-1]) == 3 and np.array_equal(solver.state, state)
    assert result["endpoint_audit"]["final_error"]["position_m"] < 1e-12
    assert result["endpoint_audit"]["attempts"][0]["position_m"] > .013
    assert result["endpoint_audit"]["corrections"] == 1
    clean = Fixture(0.)
    checked_screw_plan(clean, goal)
    assert clean.calls == 1 and len(clean.topp_calls) == 1
    bad = Fixture(.014)
    try:
        checked_screw_plan(bad, goal, max_corrections=0)
    except ValueError:
        assert bad.last_endpoint_audit["status"] == "failed"
        assert bad.planner.TOPP == bad.topp
    else:
        raise AssertionError("Uncorrected endpoint must be rejected")
    assert pose_error(goal, np.r_[goal[:3], -goal[3:]])["rotation_rad"] == 0.
    return {"status": "passed", "environment_steps": 0, "gpu_used": False,
        "checks": ["actual FK rejects 14mm native Success", "same-backend correction",
            "full corrected native TOPP and audit cache", "nonsequential arm joint order",
            "nonzero base translation", "finger positions preserved", "original start preserved",
            "already accurate native result unchanged", "bounded failure and wrapper restoration",
            "quaternion sign invariance"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
