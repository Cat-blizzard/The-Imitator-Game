"""Check MPLib screw-plan TCP endpoints and refine them without physics.

The TIG screw planner stops on a linearized twist residual. A ``Success`` is
therefore checked against actual Pinocchio FK. Native corrections obtain a
verified goal configuration; native joint planning then generates a fresh path
from the measured start. The old approximate path is diagnostic only, and the
fresh path uses the same native TOPP. Scene/payload checks remain the caller's duty.
Poses are XYZ/WXYZ, distances metres, angles radians, and joint order is explicit.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import xml.etree.ElementTree as ET

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


def nearest_revolute_goal(goal: Any, start: Any, joint_types: tuple[str, ...],
                          limits: Any) -> np.ndarray:
    """Choose only limit-valid 2*pi equivalents for actual revolute joints.

    This selects a new goal, never rewrites a trajectory. A native full replan
    and FK validation are required before any such goal can be executed.
    """
    goal, start, limits = _array(goal), _array(start), _array(limits)
    if (goal.ndim != 1 or start.shape != goal.shape or len(joint_types) != len(goal)
            or limits.shape != (len(goal), 2) or np.any(limits[:, 0] >= limits[:, 1])):
        raise ValueError("Revolute goal selection requires named joints and matching finite limits")
    result = goal.copy()
    period = 2 * np.pi
    for i, kind in enumerate(joint_types):
        lower, upper = limits[i]
        if kind == "revolute":
            first = int(np.ceil((lower - goal[i]) / period))
            last = int(np.floor((upper - goal[i]) / period))
            if first > last:
                raise ValueError("No equivalent revolute goal lies inside joint limits")
            shift = int(np.clip(np.rint((start[i] - goal[i]) / period), first, last))
            result[i] += period * shift
        if result[i] < lower or result[i] > upper:
            raise ValueError("Selected goal exceeds actual joint limits")
    return result


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
    if hasattr(solver.robot, "get_active_joints"):
        active_names = tuple(joint.get_name() for joint in solver.robot.get_active_joints())
        planner_names = tuple(planner.user_joint_names)
        if (start.shape != (len(active_names),) or len(set(active_names)) != len(active_names)
                or len(set(planner_names)) != len(planner_names)
                or set(active_names) != set(planner_names)):
            raise ValueError("Measured and planner joint names must describe the same robot")
        start = start[[active_names.index(name) for name in planner_names]]
    if (start.ndim != 1 or len(start) != len(planner.user_joint_names)
            or indices.ndim != 1 or not len(indices) or len(set(indices.tolist())) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(start))):
        raise ValueError("Expected full measured joints and unique move-group columns")
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
        if path.ndim != 2 or path.shape[1] != len(indices) or len(path) < 2:
            raise ValueError("Native TOPP source must match the move-group columns")
        captured_paths.append(path)
        if capture_geometry_only:
            raise _CorrectionGeometryCaptured
        native_topp_calls += 1
        return native_topp(path, step, verbose)

    def fk(arm_qpos: Any) -> np.ndarray:
        qpos = start.copy()
        arm_qpos = _array(arm_qpos)
        if arm_qpos.shape != (len(indices),):
            raise ValueError("Endpoint joints must match the move-group columns")
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
        if positions.ndim != 2 or positions.shape[1] != len(indices) or not len(positions):
            raise ValueError("Successful native plan has no sampled joint endpoint")
        if not captured_paths:
            raise ValueError("Native screw plan did not expose its TOPP geometric path")
        combined_path = captured_paths[-1].copy()
        if not np.allclose(combined_path[0], start[indices], atol=1e-6, rtol=0):
            raise ValueError("Native geometric path does not start at measured joints")
        actual = fk(positions[-1])
        error = pose_error(actual, goal_base)
        initial_method = getattr(solver, "last_plan_method", {}).get("method", "screw")
        audit["attempts"].append({"kind": f"original_native_{initial_method}", **error})
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
        goal_qpos = start.copy()
        goal_qpos[indices] = combined_path[-1]
        equivalent_goal_changed = False
        if getattr(getattr(solver, "env_agent", None), "prefer_nearby_revolute_goals", False):
            # Some imported arms allow multiple revolutions. Native IK can
            # return -2*pi instead of an equivalent nearby zero. Verify the
            # actual URDF type and measured limits, then replan the WHOLE path.
            descriptions = {joint.get("name"): joint.get("type")
                            for joint in ET.parse(planner.urdf).getroot().findall("joint")}
            names = tuple(planner.user_joint_names[i] for i in indices)
            if not all(name in descriptions for name in names):
                raise ValueError("Robot description lacks named arm joints")
            active_names = tuple(joint.get_name() for joint in solver.robot.get_active_joints())
            measured_limits = _array(solver.robot.get_qlimits())[0]
            limits = measured_limits[[active_names.index(name) for name in names]]
            original_goal = goal_qpos[indices].copy()
            selected = nearest_revolute_goal(original_goal, start[indices],
                tuple(descriptions[name] for name in names), limits)
            selected_error = pose_error(fk(selected), goal_base)
            if not within(selected_error):
                raise ValueError(f"Equivalent revolute goal exceeds FK contract: {selected_error}")
            equivalent_goal_changed = not np.array_equal(selected, original_goal)
            audit["periodic_goal_selection"] = {"enabled": True,
                "changes": [{"joint": name, "original_rad": float(old), "selected_rad": float(new)}
                            for name, old, new in zip(names, original_goal, selected) if old != new],
                "selected_goal_fk_error": selected_error,
                "actual_urdf_types_and_limits_checked": True}
            goal_qpos[indices] = selected
        if corrections or equivalent_goal_changed:
            # The old approximate endpoint is diagnostic only. Replan the whole
            # path from the measured start to the FK-verified goal with native
            # joint planning, rather than executing the old path plus a tail.
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
        if positions.ndim != 2 or positions.shape[1] != len(indices) or not len(positions):
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
            fresh_native_joint_replan=bool(corrections or equivalent_goal_changed),
            parameterization="native TOPP over fresh measured-start-to-verified-goal joint path" if corrections or equivalent_goal_changed
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
