"""Check MPLib screw-plan TCP endpoints and refine them without physics.

The TIG screw planner stops on a linearized twist residual. A ``Success`` is
therefore checked against actual Pinocchio FK. Native corrections obtain a
verified goal configuration; native joint planning then generates a fresh path
from the measured start. The old approximate path is diagnostic only, and the
fresh path uses the same native TOPP. Scene/payload checks remain the caller's duty.
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


class _CorrectionGeometryCaptured(Exception):
    """Stop a native correction at its TOPP boundary; no trajectory is made."""


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
    capture_geometry_only = False
    native_topp_calls = 0

    def capture_topp(path: Any, step: float = .1, verbose: bool = False) -> Any:
        nonlocal native_topp_calls
        path = _array(path)
        if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
            raise ValueError("Native TOPP source must have shape [N>=2, 7]")
        captured_paths.append(path)
        if capture_geometry_only:
            raise _CorrectionGeometryCaptured
        native_topp_calls += 1
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

    # Planner.IK accepts on ||dp|| + min(||q1-q2||, ||q1+q2||), whereas
    # its native CLIK stops on a different six-dimensional residual. Reusing
    # CLIK's 1e-5 stopping epsilon here can reject a converged candidate. For
    # normalized quaternions, the sign-invariant chord is 2*sin(angle/4).
    # Taking the smaller component bound makes this mixed-distance test a
    # sufficient (conservative) condition for our unchanged separate FK bounds.
    ik_threshold = float(min(position_tolerance_m,
        2 * np.sin(min(rotation_tolerance_rad, np.pi) / 4)))
    non_arm_indices = np.setdiff1d(np.arange(len(start)), indices)
    ik_mask = np.ones(len(start), dtype=bool)
    ik_mask[indices] = False

    audit: dict[str, Any] = {"policy": "mplib-actual-fk-endpoint-v5",
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
        "ik_distance_threshold": ik_threshold,
        "ik_distance_definition": "position norm + sign-invariant unit-quaternion chord",
        "ik_threshold_derivation": "min(position_tolerance_m, 2*sin(min(rotation_tolerance_rad, pi)/4))",
        "ik_initial_configurations": 1,
        "ik_locked_joint_indices": non_arm_indices.tolist(),
        "ik_attempts": [],
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
            # Native screw's rot2so3 treats np.isclose(trace(R), 3) as zero
            # rotation. This can also leave position outside the FK bound while
            # rotation already satisfies it: repeating screw then stagnates.
            # Try at most one screw correction before using native IK within
            # the existing correction budget. Both routes expose a real native
            # geometric path; no short-segment TOPP or fake trajectory is made.
            use_ik = (corrections > 0 or (error["rotation_rad"] > rotation_tolerance_rad
                and np.isclose(1 + 2 * np.cos(error["rotation_rad"]), 3)))
            kind = "native_ik_joint_path_correction" if use_ik else "native_screw_correction"
            capture_geometry_only = True
            try:
                options = dict(time_step=timestep, use_point_cloud=bool(solver.use_point_cloud),
                    use_attach=bool(getattr(planner.planning_world, "use_attach", False)))
                if use_ik:
                    ik_audit: dict[str, Any] = {
                        "correction_index": corrections + 1,
                        "trigger": "previous_native_correction_outside_fk_contract" if corrections
                            else "native_screw_small_angle_dead_zone",
                        "candidates": []}
                    audit["ik_attempts"].append(ik_audit)
                    status, goals = planner.IK(goal_base.copy(), full_qpos.copy(),
                        mask=ik_mask.tolist(), n_init_qpos=1, threshold=ik_threshold)
                    ik_audit.update(native_status=str(status), returned_candidates=len(goals))
                    if status != "Success" or not goals:
                        raise ValueError(f"Native endpoint IK failed: {status}")
                    for goal in goals:
                        goal = _array(goal)
                        if goal.shape != start.shape:
                            raise ValueError("Native endpoint IK returned an invalid full joint configuration")
                        if not np.array_equal(goal[non_arm_indices], start[non_arm_indices]):
                            raise ValueError("Native endpoint IK changed non-arm joints")
                        candidate_error = pose_error(fk(goal[indices]), goal_base)
                        ik_audit["candidates"].append({**candidate_error,
                            "within_final_fk_contract": within(candidate_error)})
                        if not within(candidate_error):
                            raise ValueError(f"Native endpoint IK candidate exceeds FK contract: {candidate_error}")
                    planner.plan_qpos_to_qpos(goals, full_qpos.copy(),
                        fix_joint_limits=False, **options)
                else:
                    planner.plan_screw(goal_world.copy(), full_qpos.copy(), **options)
            except _CorrectionGeometryCaptured:
                pass
            else:
                raise ValueError("Native correction failed before producing a geometric path")
            finally:
                capture_geometry_only = False
            if len(captured_paths) != count + 1:
                raise ValueError("Native correction did not produce exactly one geometric path")
            path = captured_paths[-1]
            if not np.allclose(path[0], combined_path[-1], atol=1e-9, rtol=0):
                raise ValueError("Native correction does not continue the original path")
            combined_path = np.vstack((combined_path, path[1:]))
            actual = fk(combined_path[-1])
            error = pose_error(actual, goal_base)
            corrections += 1
            audit["attempts"].append({"kind": kind,
                "correction_index": corrections, **error})
        if not within(error):
            raise ValueError(f"Actual FK endpoint exceeds contract after {corrections} corrections: {error}")
        if corrections:
            # The old approximate endpoint is diagnostic only. Replan the whole
            # path from the measured start to the FK-verified goal with native
            # joint planning, rather than executing the old path plus a tail.
            goal_qpos = start.copy()
            goal_qpos[indices] = combined_path[-1]
            count = len(captured_paths)
            capture_geometry_only = True
            try:
                planner.plan_qpos_to_qpos([goal_qpos], start.copy(),
                    time_step=timestep, fix_joint_limits=False,
                    use_point_cloud=bool(solver.use_point_cloud),
                    use_attach=bool(getattr(planner.planning_world, "use_attach", False)))
            except _CorrectionGeometryCaptured:
                pass
            else:
                raise ValueError("Native full replan failed before producing a geometric path")
            finally:
                capture_geometry_only = False
            if len(captured_paths) != count + 1:
                raise ValueError("Native full replan did not produce exactly one geometric path")
            combined_path = captured_paths[-1].copy()
            if not np.allclose(combined_path[0], start[indices], atol=1e-9, rtol=0):
                raise ValueError("Native full replan changed the measured start")
            if not np.allclose(combined_path[-1], goal_qpos[indices], atol=1e-9, rtol=0):
                raise ValueError("Native full replan changed the FK-verified goal")
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
            geometric_path_captures=len(captured_paths),
            correction_geometry_only=True, native_topp_calls=native_topp_calls,
            original_bad_endpoint_executed=False,
            fresh_native_joint_replan=bool(corrections),
            parameterization="native TOPP over fresh measured-start-to-verified-goal joint path" if corrections
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
                TOPP=self.topp, plan_screw=self.screw, plan_qpos_to_qpos=self.joint_path)
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

        def joint_path(self, goals: Any, full: Any, **kwargs: Any) -> Any:
            assert np.array_equal(full, self.state)
            return self.planner.TOPP(np.vstack((full[self.indices], goals[0][self.indices])), .05)

    goal = np.array([1.1, 2.2, 3.3, 1., 0., 0., 0.])
    solver = Fixture(.014)
    state = solver.state.copy()
    original_topp = solver.planner.TOPP
    result = checked_screw_plan(solver, goal)
    assert solver.planner.TOPP == original_topp and solver.calls == 2
    assert solver.last_native_result is result
    assert np.array_equal(solver.last_native_parameterization, solver.topp_calls[-1])
    assert len(solver.topp_calls) == 2 and len(solver.topp_calls[-1]) == 2
    assert not np.any(np.all(np.isclose(solver.topp_calls[-1],
        solver.topp_calls[0][-1], atol=1e-12, rtol=0), axis=1))
    assert np.array_equal(solver.state, state)
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

    class SmallAngleFixture(Fixture):
        """Model native IK's mixed-distance acceptance, not the CLIK epsilon."""
        def __init__(self, *, position_residual: float = 8e-6,
                     rotation_residual: float = 16e-6,
                     force_candidate: bool = False, change_finger: bool = False,
                     native_failure: bool = False) -> None:
            super().__init__(.008)
            self.position_residual = position_residual
            self.rotation_residual = rotation_residual
            self.force_candidate, self.change_finger = force_candidate, change_finger
            self.native_failure = native_failure
            self.ik_calls: list[dict[str, Any]] = []
            self.joint_calls: list[np.ndarray] = []
            self.planner.IK = self.ik

        def fk(self, full: np.ndarray) -> None:
            assert np.array_equal(full[7:], self.state[7:])
            arm = full[self.indices]
            self.actual = np.r_[arm[:3], np.cos(arm[3] / 2), 0., 0., np.sin(arm[3] / 2)]

        def screw(self, goal: np.ndarray, full: np.ndarray, **kwargs: Any) -> Any:
            self.calls += 1
            end = full[self.indices].copy()
            end[:3] = goal[:3] - [1., 2., 3.]
            end[0] -= self.bias
            end[3] = .002
            raw = self.planner.TOPP(np.vstack((full[self.indices], end)), .05)
            return {"status": "Success", **dict(zip(
                ("time", "position", "velocity", "acceleration", "duration"), raw))}

        def ik(self, target: np.ndarray, full: np.ndarray, *, mask: list[bool],
               n_init_qpos: int, threshold: float) -> tuple[str, list[np.ndarray]]:
            assert n_init_qpos == 1
            assert mask == [False] * 7 + [True, True]
            assert np.array_equal(full[7:], self.state[7:])
            candidate = self.state.copy()
            arm = candidate[self.indices].copy()
            arm[:3] = target[:3]
            arm[0] += self.position_residual
            arm[3] = self.rotation_residual
            candidate[self.indices] = arm
            distance = self.position_residual + 2 * np.sin(self.rotation_residual / 4)
            self.ik_calls.append({"threshold": threshold, "mixed_distance": float(distance),
                                  "mask": mask.copy()})
            if self.native_failure:
                return "IK Failed! Cannot find valid solution.", []
            if distance >= threshold and not self.force_candidate:
                return "IK Failed! Distance is greater than threshold.", []
            if self.change_finger:
                candidate[7] += .001
            return "Success", [candidate]

        def joint_path(self, goals: Any, full: Any, **kwargs: Any) -> Any:
            assert kwargs["use_point_cloud"] and kwargs["use_attach"]
            assert kwargs["fix_joint_limits"] is False
            assert np.array_equal(full[7:], self.state[7:])
            assert np.array_equal(goals[0][7:], self.state[7:])
            self.joint_calls.append(_array(full))
            return self.planner.TOPP(np.vstack((full[self.indices], goals[0][self.indices])), .05)

    # A converged small-angle candidate can have mixed residual 16e-6 even
    # though both final FK errors are far inside 0.5mm / 0.001rad. The old
    # 1e-5 outer acceptance threshold rejected it before joint planning.
    small = SmallAngleFixture()
    old_status, old_goals = small.ik(np.r_[goal[:3] - [1., 2., 3.], goal[3:]],
        small.state.copy(), mask=[False] * 7 + [True, True], n_init_qpos=1, threshold=1e-5)
    assert old_status != "Success" and not old_goals
    small.ik_calls.clear()
    small_state, small_topp = small.state.copy(), small.planner.TOPP
    small_result = checked_screw_plan(small, goal)
    small_audit = small_result["endpoint_audit"]
    threshold = min(POSITION_TOLERANCE_M, 2 * np.sin(ROTATION_TOLERANCE_RAD / 4))
    assert small_audit["ik_distance_threshold"] == threshold
    assert len(small.ik_calls) == 1 and len(small.joint_calls) == 2
    assert small_audit["attempts"][1]["kind"] == "native_ik_joint_path_correction"
    assert small_audit["ik_attempts"][0]["candidates"][0]["within_final_fk_contract"]
    assert small_audit["final_error"]["position_m"] < POSITION_TOLERANCE_M
    assert small_audit["final_error"]["rotation_rad"] < ROTATION_TOLERANCE_RAD
    assert np.array_equal(small.state, small_state) and small.planner.TOPP == small_topp
    assert small_audit["environment_steps"] == 0 and small_audit["ik_locked_joint_indices"] == [7, 8]
    boundary = SmallAngleFixture(position_residual=threshold * (1 - 1e-6), rotation_residual=0.)
    checked_screw_plan(boundary, goal)

    rejected = [
        (SmallAngleFixture(position_residual=threshold * (1 + 1e-6), rotation_residual=0.),
         "Native endpoint IK failed"),
        (SmallAngleFixture(position_residual=.0006, force_candidate=True),
         "Native endpoint IK candidate exceeds FK contract"),
        (SmallAngleFixture(rotation_residual=.0011, force_candidate=True),
         "Native endpoint IK candidate exceeds FK contract"),
        (SmallAngleFixture(change_finger=True), "Native endpoint IK changed non-arm joints"),
        (SmallAngleFixture(native_failure=True), "Native endpoint IK failed"),
    ]
    for fixture, message in rejected:
        initial, topp = fixture.state.copy(), fixture.planner.TOPP
        try:
            checked_screw_plan(fixture, goal)
        except ValueError as exc:
            assert message in str(exc), str(exc)
            assert fixture.last_endpoint_audit["status"] == "failed"
            assert fixture.last_endpoint_audit["ik_attempts"][0]["native_status"]
            assert not fixture.joint_calls and fixture.planner.TOPP == topp
            assert np.array_equal(fixture.state, initial)
        else:
            raise AssertionError(f"Expected rejection: {message}")

    class StalledScrewFixture(SmallAngleFixture):
        """Retreat failure: screw leaves 0.523mm with rotation already valid."""
        def screw(self, goal: np.ndarray, full: np.ndarray, **kwargs: Any) -> Any:
            self.calls += 1
            end = full[self.indices].copy()
            end[:3] = goal[:3] - [1., 2., 3.]
            end[0] -= .0026 if self.calls == 1 else .000523
            end[3] = .000716
            raw = self.planner.TOPP(np.vstack((full[self.indices], end)), .05)
            return {"status": "Success", **dict(zip(
                ("time", "position", "velocity", "acceleration", "duration"), raw))}

    stalled = StalledScrewFixture()
    stalled_state = stalled.state.copy()
    stalled_result = checked_screw_plan(stalled, goal)
    stalled_audit = stalled_result["endpoint_audit"]
    assert stalled.calls == 2 and len(stalled.ik_calls) == 1
    assert stalled_audit["corrections"] == 2
    assert [entry["kind"] for entry in stalled_audit["attempts"]] == [
        "original_native_screw", "native_screw_correction", "native_ik_joint_path_correction"]
    assert stalled_audit["attempts"][1]["position_m"] > POSITION_TOLERANCE_M
    assert stalled_audit["attempts"][1]["rotation_rad"] < ROTATION_TOLERANCE_RAD
    assert stalled_audit["ik_attempts"][0]["trigger"] == "previous_native_correction_outside_fk_contract"
    assert stalled_audit["final_error"]["position_m"] < POSITION_TOLERANCE_M
    assert stalled_audit["final_error"]["rotation_rad"] < ROTATION_TOLERANCE_RAD
    assert np.array_equal(stalled.state, stalled_state) and stalled.planner.TOPP == stalled.topp
    assert len(stalled.topp_calls) == 2 and stalled_audit["environment_steps"] == 0
    # The fallback consumes an existing correction slot, never adds retries or
    # bypasses native IK's refusal. A caller allowing only one correction fails.
    for fixture, limit, expected_ik_calls in [
            (StalledScrewFixture(), 1, 0), (StalledScrewFixture(native_failure=True), 3, 1)]:
        try:
            checked_screw_plan(fixture, goal, max_corrections=limit)
        except ValueError:
            assert fixture.last_endpoint_audit["status"] == "failed"
            assert fixture.calls == 2 and len(fixture.ik_calls) == expected_ik_calls
            assert not fixture.joint_calls and fixture.planner.TOPP == fixture.topp
            assert np.array_equal(fixture.state, stalled_state)
        else:
            raise AssertionError("Bounded fallback must preserve insufficient-budget/native-refusal failure")
    assert pose_error(goal, np.r_[goal[:3], -goal[3:]])["rotation_rad"] == 0.
    return {"status": "passed", "environment_steps": 0, "gpu_used": False,
        "checks": ["actual FK rejects 14mm native Success", "same-backend correction",
            "fresh measured-start native joint path discards the old bad endpoint",
            "full corrected native TOPP and audit cache", "nonsequential arm joint order",
            "nonzero base translation", "finger positions preserved", "original start preserved",
            "already accurate native result unchanged", "bounded failure and wrapper restoration",
            "small-angle native IK branch accepts independently valid 16e-6 mixed residual",
            "IK mixed-distance threshold derives from unchanged FK bounds",
            "IK threshold boundary acceptance and rejection", "IK explicitly locks non-arm joints",
            "inaccurate IK candidates rejected before joint planning",
            "native IK refusal and changed-finger candidate remain rejected",
            "stalled screw switches to native IK while rotation already satisfies FK bounds",
            "IK fallback consumes existing correction budget and preserves native refusal",
            "quaternion sign invariance"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
