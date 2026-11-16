"""Evaluate compact paired manipulation under the unchanged official clock.

Contract: physical left/right actions are [7 joint radians, normalized gripper]
per 0.05 s tick; physics remains 0.01 s. Geometry merges lift/raise and turn/carry.
Version analytic-phase-retiming-v1 preserves raw native output, bounds its
piecewise source curve, and slows both arms by one common chain-rule factor.
The default TimeLimit is 200;
its truncation still stops immediately, so completion should precede tick 200.
All original collision, grasp, height, release and final-placement gates remain.
The execution profile omits the separate 20-tick airborne diagnostic hover;
--execution-profile diagnostic-hover retains that independent diagnostic gate.
No physical state/pose assignment, weld or collision-mask mutation is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import time
import traceback
from typing import Any

import numpy as np

from .validate_grasp import TASK, grasp_poses, json_value, recording_summary, save_report
from .validate_banana import banana_grasp_poses

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
ACTOR_NAMES = ("apple_out", "banana_out")


# Reuse frozen acceptance logic; only this new entry changes route/configuration.
from .validate_sync import (continue_success_only, applied_action, retain_right_valid,
                            physical_region, held_substep_audit, hold_joint_substep_audit,
                            self_test as original_self_test)


ROUTES = ("merge-transport", "merge-lift")
PROFILES = ("execution", "diagnostic-hover")
# Native TOPP interpolation can marginally exceed its own requested bound.
# Solve inside the nominal envelope, while validating unchanged nominal gates.
SOLVER_LIMIT_FRACTION = .99
RETIMING_POLICY = "analytic-phase-retiming-v1"
RELEASE_POLICY = "geometry-informed-v1"
RELEASE_TCP_OFFSETS_BASKET_M = ((.03, .08, .10), (.03, 0., .12))
TRANSPORT_CLEARANCE_ABOVE_RELEASE_M = .06
ROUTE_GEOMETRY_POLICY = "compact-clearance-v2"


def compact_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Validate explicit experiment choices before any native environment exists."""
    if args.route not in ROUTES or args.execution_profile not in PROFILES:
        raise ValueError("Unknown compact route or diagnostic profile")
    if (isinstance(args.max_episode_steps, bool) or not isinstance(args.max_episode_steps, int)
            or not 200 <= args.max_episode_steps <= 400):
        raise ValueError("Budget must be an integer in [200,400]; only 200 is official")
    speed, acceleration = float(args.joint_vel_limit), float(args.joint_acc_limit)
    if not np.isfinite([speed, acceleration]).all() or min(speed, acceleration) <= 0:
        raise ValueError("Finite positive joint velocity/acceleration limits are required")
    approach_speed_override = getattr(args, "approach_vel_limit", None)
    approach_acc_override = getattr(args, "approach_acc_limit", None)
    approach_speed = speed if approach_speed_override is None else float(approach_speed_override)
    approach_acceleration = acceleration if approach_acc_override is None else float(approach_acc_override)
    if (not np.isfinite([approach_speed, approach_acceleration]).all()
            or min(approach_speed, approach_acceleration) <= 0):
        raise ValueError("Approach overrides must be finite positive limits")
    release_offsets = np.asarray(RELEASE_TCP_OFFSETS_BASKET_M, dtype=float)
    if release_offsets.shape != (2, 3) or not np.isfinite(release_offsets).all():
        raise ValueError("Release TCP basket offsets must be finite paired XYZ values")
    return {"route": args.route, "execution_profile": args.execution_profile,
            # Previously hardcoded route values are now exposed for the ledger.
            "transport_clearance_above_release_m": TRANSPORT_CLEARANCE_ABOVE_RELEASE_M,
            "minimum_commanded_lift_m": .12,
            "route_geometry_policy": ROUTE_GEOMETRY_POLICY,
            "airborne_hover_steps": 20 if args.execution_profile == "diagnostic-hover" else 0,
            "preparation_steps": 10, "min_actual_height_gain_m": [.10, .10],
            "release_stable_steps": 3, "retreat_command_m": .08,
            "release_tcp_offsets_basket_m": release_offsets.tolist(),
            "release_pose_policy": RELEASE_POLICY,
            "release_pose_evidence_scope": "fixed common recipe selected from cooked geometry endpoint checks; full candidate and measured motion collision gates remain required",
            "final_common_placement_steps": 20,
            "max_episode_steps": args.max_episode_steps,
            "official_budget_candidate": args.max_episode_steps == 200,
            "joint_vel_limit_rad_s": speed, "joint_acc_limit_rad_s2": acceleration,
            "approach_vel_limit_override_rad_s": approach_speed_override,
            "approach_acc_limit_override_rad_s2": approach_acc_override,
            "approach_joint_vel_limit_rad_s": approach_speed,
            "approach_joint_acc_limit_rad_s2": approach_acceleration,
            "approach_limit_phases": ["pregrasp", "descend"],
            "solver_limit_fraction": SOLVER_LIMIT_FRACTION,
            "solver_joint_vel_limit_rad_s": speed * SOLVER_LIMIT_FRACTION,
            "solver_joint_acc_limit_rad_s2": acceleration * SOLVER_LIMIT_FRACTION,
            "solver_margin_reason": "solve 1 percent inside nominal derivative limits; nominal output gates and tolerance remain unchanged",
            "joint_limit_interpretation": "planner outputs checked; velocity bounded by URDF; no hardware acceleration certification",
            "source_timestamp_scaling": True, "time_parameterization_policy": RETIMING_POLICY,
            "native_source_timestamp_scaling": False,
            "retiming_scope": "one common phase factor from numerically evaluated analytic source-curve extrema; not physical actuator or controller continuous-time certification",
            "control_hz": 20, "physics_hz": 100,
            "acceptance_difference_from_original": (
                "separate airborne 20-tick hover omitted; held 100Hz slip, grasp/height, release, retreat and final20 unchanged"
                if args.execution_profile == "execution" else "airborne 20-tick hover and its original audit retained")}


def compact_phase_limits(configuration: dict[str, Any], phase: str) -> dict[str, Any]:
    """Resolve one phase's nominal and99pct solver limits for shared retiming."""
    if not isinstance(phase, str) or not phase:
        raise ValueError("An explicit planned phase is required")
    approach = phase in ("pregrasp", "descend")
    prefix = "approach_" if approach else ""
    velocity = float(configuration[f"{prefix}joint_vel_limit_rad_s"])
    acceleration = float(configuration[f"{prefix}joint_acc_limit_rad_s2"])
    fraction = float(configuration["solver_limit_fraction"])
    if (not np.isfinite([velocity, acceleration, fraction]).all()
            or min(velocity, acceleration) <= 0 or not 0 < fraction <= 1):
        raise ValueError("Invalid phase limits or solver margin")
    return {"phase": phase, "limit_group": "approach" if approach else "global",
            "nominal_velocity_limit_rad_s": velocity,
            "nominal_acceleration_limit_rad_s2": acceleration,
            "solver_velocity_limit_rad_s": velocity * fraction,
            "solver_acceleration_limit_rad_s2": acceleration * fraction,
            "solver_limit_fraction": fraction, "source_timestamp_scaling": True,
            "time_parameterization_policy": RETIMING_POLICY}


def compact_set_phase_limits(solvers: Any, limits: dict[str, Any]) -> None:
    """Update each private MPLib planner before TOPP reads its current arrays.

    Local Planner.TOPP obtains self.joint_vel_limits[:dof] and
    self.joint_acc_limits[:dof] on every invocation. Setting the wrapper's scalar
    attributes alone would leave constructor-time MPLib arrays unchanged.
    """
    if not isinstance(solvers, (tuple, list)) or len(solvers) != 2:
        raise ValueError("Two canonical solver objects are required")
    speed = float(limits["solver_velocity_limit_rad_s"])
    acceleration = float(limits["solver_acceleration_limit_rad_s2"])
    if not np.isfinite([speed, acceleration]).all() or min(speed, acceleration) <= 0:
        raise ValueError("Finite positive solver limits are required")
    for solver in solvers:
        if len(solver.planner.move_group_joint_indices) != 7:
            raise ValueError("Phase limits require a seven-joint move group")
        solver.joint_vel_limits, solver.joint_acc_limits = speed, acceleration
        solver.planner.joint_vel_limits = np.full(7, speed, dtype=float)
        solver.planner.joint_acc_limits = np.full(7, acceleration, dtype=float)
        assert solver.planner.joint_vel_limits.shape == solver.planner.joint_acc_limits.shape == (7,)


def compact_route_positions(current_xyz: Any, release_xyz: Any, *, route: str) -> dict[str, Any]:
    """Return world XYZ[2,3] endpoints; yaw is merged only after safe lift.

    Each arm first physically gains >=0.10m at its current XY. For merge-lift,
    that same vertical move reaches basket-clearance height. The intended route
    keeps a separate vertical basket descent. Returned paths require planning and
    the original combined payload/robot/world collision and execution checks.
    """
    current, release = np.asarray(current_xyz, dtype=float), np.asarray(release_xyz, dtype=float)
    if route not in ROUTES or current.shape != (2, 3) or release.shape != (2, 3):
        raise ValueError("Expected one compact route and paired world XYZ arrays [2,3]")
    if not np.isfinite(current).all() or not np.isfinite(release).all():
        raise ValueError("Route endpoints must be finite")
    clearance = np.maximum(current[:, 2] + .12, release[:, 2] + TRANSPORT_CLEARANCE_ABOVE_RELEASE_M)
    raised = current.copy(); raised[:, 2] = clearance
    lift = raised.copy() if route == "merge-lift" else current + [0, 0, .12]
    above = release.copy(); above[:, 2] = clearance
    assert lift.shape == raised.shape == above.shape == (2, 3)
    assert np.all(lift[:, 2] - current[:, 2] >= .12 - 1e-12)
    return {"lift": lift, "raise": None if route == "merge-lift" else raised,
            "above_baskets": above, "release": release.copy(), "clearance_z": clearance}


def compact_urdf_velocity_limits(urdf_path: Any, joint_names: Any,
                                 requested_limit: float) -> dict[str, Any]:
    """Require all seven named arm URDF velocity limits; never infer acceleration."""
    import xml.etree.ElementTree as ET
    names = tuple(joint_names)
    value = float(requested_limit)
    if len(names) != 7 or len(set(names)) != 7 or not np.isfinite(value) or value <= 0:
        raise ValueError("Need seven unique controller joints and a positive speed")
    contents = Path(urdf_path).read_bytes()
    nodes = {node.attrib["name"]: node for node in ET.fromstring(contents).findall("joint")}
    limits = []
    for name in names:
        limit = None if name not in nodes else nodes[name].find("limit")
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(f"URDF joint {name} lacks an explicit velocity limit")
        bound = float(limit.attrib["velocity"])
        if not np.isfinite(bound) or bound <= 0 or value > bound + 1e-12:
            raise ValueError(f"Requested velocity {value} exceeds/invalidates URDF joint {name} limit {bound}")
        limits.append(bound)
    return {"urdf_path": str(urdf_path), "urdf_sha256": hashlib.sha256(contents).hexdigest(),
            "joint_names": list(names), "velocity_limits_rad_s": limits,
            "requested_velocity_limit_rad_s": value,
            "acceleration_limit_source": "explicit planner configuration; URDF has no claimed hardware certification"}


def compact_kinematic_audit(result: Any, *, velocity_limit: float,
                             acceleration_limit: float) -> dict[str, Any]:
    """Check native TOPP velocity/acceleration arrays[N,7] without retiming."""
    if not isinstance(result, dict) or result.get("status") != "Success":
        raise ValueError("Native planner did not return a successful trajectory")
    times = np.asarray(result.get("time"), dtype=float)
    positions = np.asarray(result.get("position"), dtype=float)
    velocities = np.asarray(result.get("velocity"), dtype=float)
    accelerations = np.asarray(result.get("acceleration"), dtype=float)
    limits = np.asarray([velocity_limit, acceleration_limit], dtype=float)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0)
            or any(a.shape != (len(times), 7) for a in (positions, velocities, accelerations))
            or not all(np.isfinite(a).all() for a in (times, positions, velocities, accelerations, limits))
            or np.any(limits <= 0)):
        raise ValueError("Need finite time[N], position/velocity/acceleration[N,7] and positive limits")
    max_v, max_a = float(np.max(np.abs(velocities))), float(np.max(np.abs(accelerations)))
    passed = max_v <= velocity_limit + 1e-6 and max_a <= acceleration_limit + 1e-6
    return {"status": "passed" if passed else "failed", "samples": len(times),
            "max_source_velocity_rad_s": max_v, "max_source_acceleration_rad_s2": max_a,
            "max_source_position_difference_speed_rad_s": float(np.max(np.abs(np.diff(positions, axis=0) / np.diff(times)[:, None]))),
            "configured_velocity_limit_rad_s": velocity_limit,
            "configured_acceleration_limit_rad_s2": acceleration_limit,
            "numeric_tolerance": 1e-6,
            "scope": "native planner derivative outputs, not physical actuator acceleration or a continuous-safety certificate"}


def compact_topp_reconstruction(path: Any, velocity_limits: Any, acceleration_limits: Any,
                                step: float, raw_output: Any) -> dict[str, Any]:
    """Rebuild the untouched native TOPP call and require exact returned samples.

    The second TOPPRA solve exposes the path spline and constant-s-acceleration
    segments; it must reproduce all four raw arrays and duration exactly. Any
    incompatible backend, failed solve or mismatch rejects planning.
    """
    import toppra as ta
    import toppra.algorithm as algo
    import toppra.constraint as constraint
    points = np.asarray(path, dtype=np.float64)
    velocity, acceleration = np.asarray(velocity_limits, dtype=float), np.asarray(acceleration_limits, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (7,) or len(points) < 2
            or velocity.shape != (7,) or acceleration.shape != (7,)
            or not all(np.isfinite(a).all() for a in (points, velocity, acceleration))
            or min(float(velocity.min()), float(acceleration.min()), float(step)) <= 0
            or not np.isfinite(step) or not isinstance(raw_output, (tuple, list)) or len(raw_output) != 5):
        raise ValueError("Need finite native path[N,7], positive seven-joint limits, step and raw TOPP tuple")
    spline = ta.SplineInterpolator(np.linspace(0, 1, len(points)), points)
    instance = algo.TOPPRA([constraint.JointVelocityConstraint(velocity),
                           constraint.JointAccelerationConstraint(acceleration)], spline,
                          parametrizer="ParametrizeConstAccel")
    trajectory = instance.compute_trajectory()
    if trajectory is None:
        raise RuntimeError("Native TOPP reconstruction failed")
    times = np.linspace(0, trajectory.duration, int(trajectory.duration / step))
    rebuilt = (times, trajectory(times), trajectory(times, 1), trajectory(times, 2), trajectory.duration)
    if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(rebuilt, raw_output)):
        raise RuntimeError("Native TOPP reconstruction differs from original raw result")
    return {"native_path": points.copy(), "path_knots_s": spline.cspl.x.copy(),
            "path_coefficients": spline.cspl.c.copy(), "grid_s": trajectory._ss.copy(),
            "grid_speed_squared": trajectory._xs.copy(), "grid_acceleration": trajectory._us.copy(),
            "grid_time": trajectory._ts.copy(), "sample_time": np.asarray(raw_output[0]).copy(),
            "sample_position": np.asarray(raw_output[1]).copy(), "sample_velocity": np.asarray(raw_output[2]).copy(),
            "sample_acceleration": np.asarray(raw_output[3]).copy(), "duration": float(raw_output[4]),
            "solver_velocity_limits": velocity.copy(), "solver_acceleration_limits": acceleration.copy()}


def _compact_real_roots(a: float, b: float, c: float) -> list[float]:
    """Finite real roots of a quadratic, including linear and constant cases."""
    if a == 0.:
        return [] if b == 0. else [-c / b]
    discriminant = b * b - 4 * a * c
    if discriminant < 0.:
        return []
    square = math.sqrt(discriminant)
    q = -.5 * (b + math.copysign(square, b))
    return [-b / (2 * a)] if q == 0. else [q / a, c / q]


def compact_continuous_derivative_bounds(trace: dict[str, Any]) -> dict[str, Any]:
    """Numerical analytic extrema of the reconstructed TOPPRA source curve.

    On each common path-cubic/parameterization interval, x(s)=s_dot^2 is
    linear and u=s_ddot is constant. Joint a(s)=q''(s)x(s)+q'(s)u is quadratic.
    Its stationary points bound acceleration; roots a(s)=0 bound velocity.
    Both sides of all parameterization boundaries are evaluated. This bounds
    this source curve in floating point, not physical or 20Hz actuator motion.
    """
    knots = np.asarray(trace["path_knots_s"], dtype=float)
    coefficients = np.asarray(trace["path_coefficients"], dtype=float)
    grid = np.asarray(trace["grid_s"], dtype=float)
    speeds = np.asarray(trace["grid_speed_squared"], dtype=float)
    controls = np.asarray(trace["grid_acceleration"], dtype=float)
    if (knots.ndim != 1 or len(knots) < 2 or grid.ndim != 1 or len(grid) < 2
            or coefficients.shape != (4, len(knots) - 1, 7) or speeds.shape != grid.shape
            or controls.shape != (len(grid) - 1,)
            or not all(np.isfinite(a).all() for a in (knots, coefficients, grid, speeds, controls))
            or np.any(np.diff(knots) <= 0) or np.any(np.diff(grid) <= 0) or np.any(speeds < 0)
            or knots[0] != grid[0] or knots[-1] != grid[-1]
            or not np.allclose(np.diff(speeds), 2 * controls * np.diff(grid), rtol=1e-10, atol=1e-10)):
        raise ValueError("Malformed cubic path or constant-acceleration parameterization")
    cuts = np.unique(np.r_[knots, grid])
    max_velocity, max_acceleration = np.zeros(7), np.zeros(7)
    velocity_witness, acceleration_witness = {}, {}
    candidates_checked = 0
    negative_speed_roundoff = 0.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for left, right in zip(cuts[:-1], cuts[1:]):
            middle = (left + right) / 2
            k = min(int(np.searchsorted(knots, middle, side="right") - 1), len(knots) - 2)
            j = min(int(np.searchsorted(grid, middle, side="right") - 1), len(grid) - 2)
            if k < 0 or j < 0:
                raise ValueError("Common source interval is outside its path")
            cubic, quadratic, linear, _ = coefficients[:, k]
            u = controls[j]
            x0 = speeds[j] + 2 * u * (knots[k] - grid[j])
            # Coordinates below are y=s-path_knots[k].
            aa = 15 * u * cubic
            ab = 6 * cubic * x0 + 6 * u * quadratic
            ac = 2 * quadratic * x0 + u * linear
            low, high = left - knots[k], right - knots[k]
            candidates = [float(low), float(high)]
            for axis in range(7):
                if aa[axis] != 0.:
                    root = -ab[axis] / (2 * aa[axis])
                    if low <= root <= high:
                        candidates.append(float(root))
                for root in _compact_real_roots(float(aa[axis]), float(ab[axis]), float(ac[axis])):
                    if not np.isfinite(root):
                        raise ValueError("Nonfinite source-curve stationary point")
                    if low <= root <= high:
                        candidates.append(float(root))
            y = np.asarray(candidates)
            x = x0 + 2 * u * y
            negative_speed_roundoff = min(negative_speed_roundoff, float(x.min()))
            if np.any(x < -1e-10):
                raise ValueError("Source squared speed is negative within an interval")
            velocity = (3 * cubic * y[:, None] ** 2 + 2 * quadratic * y[:, None] + linear) * np.sqrt(np.maximum(x, 0))[:, None]
            acceleration = aa * y[:, None] ** 2 + ab * y[:, None] + ac
            if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
                raise ValueError("Nonfinite analytic source derivative")
            candidates_checked += len(candidates)
            for values, maxima, witness in ((velocity, max_velocity, velocity_witness),
                                            (acceleration, max_acceleration, acceleration_witness)):
                indices = np.abs(values).argmax(axis=0)
                peaks = np.abs(values[indices, np.arange(7)])
                if float(peaks.max()) > float(maxima.max()) or not witness:
                    axis = int(peaks.argmax()); row = int(indices[axis])
                    witness.update(joint_index=axis, path_s=float(y[row] + knots[k]),
                                   path_segment=k, parameterization_segment=j,
                                   signed_value=float(values[row, axis]))
                maxima[:] = np.maximum(maxima, peaks)
    return {"policy": RETIMING_POLICY, "scope": "numerical analytic extrema of reconstructed source TOPPRA curve; not physical/controller acceleration",
            "per_joint_max_velocity_rad_s": max_velocity.tolist(),
            "per_joint_max_acceleration_rad_s2": max_acceleration.tolist(),
            "max_velocity_rad_s": float(max_velocity.max()), "max_acceleration_rad_s2": float(max_acceleration.max()),
            "velocity_witness": velocity_witness, "acceleration_witness": acceleration_witness,
            "common_intervals": len(cuts) - 1, "candidates_checked": candidates_checked,
            "squared_speed_roundoff_min": negative_speed_roundoff,
            "squared_speed_roundoff_rejection_threshold": -1e-10}


def compact_retime_pair(results: tuple[Any, Any], traces: tuple[Any, Any], *,
                        velocity_limit: float, acceleration_limit: float) -> tuple[tuple[Any, Any], dict[str, Any]]:
    """Apply one common phase time dilation with q fixed and exact chain rule."""
    if (not isinstance(results, tuple) or len(results) != 2 or not isinstance(traces, tuple) or len(traces) != 2
            or not np.isfinite([velocity_limit, acceleration_limit]).all() or min(velocity_limit, acceleration_limit) <= 0):
        raise ValueError("Need two canonical results/traces and positive nominal limits")
    bounds, required = [], 1.
    for raw, trace in zip(results, traces):
        if raw is None:
            if trace is not None:
                raise ValueError("A stationary arm must have no native trajectory trace")
            bounds.append(None); continue
        if not isinstance(raw, dict) or not isinstance(trace, dict) or raw.get("status") != "Success":
            raise ValueError("A moving arm needs a successful native result and full trace")
        compact_kinematic_audit(raw, velocity_limit=velocity_limit, acceleration_limit=acceleration_limit)
        if float(raw["duration"]) != float(np.asarray(raw["time"])[-1]):
            raise ValueError("Raw native duration differs from its final timestamp")
        for field in ("time", "position", "velocity", "acceleration"):
            if not np.array_equal(np.asarray(raw[field]), np.asarray(trace["sample_" + field])):
                raise ValueError("Derivative certificate does not describe the preserved raw native result")
        bound = compact_continuous_derivative_bounds(trace)
        required = max(required, bound["max_velocity_rad_s"] / velocity_limit,
                       math.sqrt(bound["max_acceleration_rad_s2"] / acceleration_limit))
        bounds.append(bound)
    scale = float(np.nextafter(required, np.inf)) if required > 1. else 1.
    retimed, certificates = [], []
    for raw, bound in zip(results, bounds):
        if raw is None:
            retimed.append(None); certificates.append({"stationary_hold": True}); continue
        value = {**raw, "time": np.asarray(raw["time"]).copy() * scale,
                 "position": np.asarray(raw["position"]).copy(),
                 "velocity": np.asarray(raw["velocity"]).copy() / scale,
                 "acceleration": np.asarray(raw["acceleration"]).copy() / scale ** 2,
                 "duration": float(raw["duration"]) * scale}
        if compact_kinematic_audit(value, velocity_limit=velocity_limit, acceleration_limit=acceleration_limit)["status"] != "passed":
            raise RuntimeError("Retimed source samples still violate unchanged derivative gates")
        assert bound is not None
        velocity = bound["max_velocity_rad_s"] / scale
        acceleration = bound["max_acceleration_rad_s2"] / scale ** 2
        if velocity > velocity_limit + 1e-6 or acceleration > acceleration_limit + 1e-6:
            raise RuntimeError("Retimed analytic source curve violates unchanged nominal gates")
        retimed.append(value); certificates.append({"raw_source_curve": bound,
            "retimed_max_velocity_rad_s": velocity, "retimed_max_acceleration_rad_s2": acceleration,
            "raw_duration_s": float(raw["duration"]), "retimed_duration_s": value["duration"]})
    return tuple(retimed), {"policy": RETIMING_POLICY, "shared_phase_scale": scale,
        "scale_formula_before_rounding": required, "rounding_margin": "one upward float64 ULP only when scale exceeds1",
        "nominal_velocity_limit_rad_s": velocity_limit, "nominal_acceleration_limit_rad_s2": acceleration_limit,
        "positions_unchanged": True, "chain_rule": "t*=s; duration*=s; q unchanged; v/=s; a/=s^2",
        "raw_native_arrays_preserved": True, "source_timestamp_scaling_applied": scale > 1., "arms": certificates}


def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any]) -> None:
    import gymnasium as gym
    import imageio.v2 as imageio
    import sapien
    import torch
    import mani_skill.envs
    from mani_skill.utils.wrappers.record import RecordEpisode
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from .trajectory_execution import as_numpy
    from .dual_payload_collision import DualPayloadCollisionChecker
    from .dual_arm_execution import (plan_dual_pose, execute_dual_reach, close_dual_and_confirm,
                                    execute_dual_held_move, release_dual_and_confirm, verify_dual_placement)
    from .trajectory_schedule import ArmTrajectory, build_hold_schedule
    from mani_skill.envs.tasks.tabletop.dual_tasks._008_pick_apple_banana_baskets import world_aabb_from_local_mesh

    env, raw, original_hook = None, None, None
    frames, ticks, substeps, keys = [], [], [], []
    report["fault_events"] = []
    try:
        base = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos", sim_backend="cpu",
                        render_backend="gpu", render_mode="rgb_array", max_episode_steps=args.max_episode_steps)
        env = RecordEpisode(base, output_dir=str(output), trajectory_name="trajectory", save_video=False,
                            save_on_reset=False, clean_on_close=False, record_reward=False, record_env_state=True,
                            source_type="motionplanning", source_desc=f"compact synchronized apple and banana; route={args.route}; profile={args.execution_profile}; mode={args.mode}")
        env.reset(seed=args.seed)
        raw = env.unwrapped
        agents, targets = (raw.left_agent, raw.right_agent), (raw.apple_out, raw.banana_out)
        keys = list(raw.agent.agents_dict)
        if len(keys) != 2 or any(raw.agent.agents_dict[k] is not a for k, a in zip(keys, agents)):
            raise ValueError("Physical left/right action identity mismatch")
        report.update(backend={"physics": raw.backend.sim_backend, "render": raw.backend.render_backend},
                      control_timestep_s=float(raw.control_timestep), stages=[], action_key_order=keys)

        def measurement() -> dict[str, Any]:
            objects = []
            for name, agent, actor in zip(ACTOR_NAMES, agents, targets):
                forces = [as_numpy(raw.scene.get_pairwise_contact_forces(link, actor))[0]
                          for link in (agent.finger1_link, agent.finger2_link)]
                objects.append({"actor": name, "pose": as_numpy(actor.pose.raw_pose)[0].tolist(),
                                "linear_velocity_m_s": as_numpy(actor.linear_velocity)[0].tolist(),
                                "angular_velocity_rad_s": as_numpy(actor.angular_velocity)[0].tolist(),
                                "is_grasping": bool(as_numpy(agent.is_grasping(actor))[0]),
                                "finger_contact_force_norms_N": [float(np.linalg.norm(f)) for f in forces]})
            return {"left_qpos": as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                    "right_qpos": as_numpy(agents[1].robot.get_qpos())[0].tolist(),
                    "tcp_poses": [as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents], "objects": objects}

        class Capture:
            unwrapped = raw
            phase = "preparation"
            observe_substeps = False
            pending_control_step = 0

            def frame(self) -> None:
                frame = as_numpy(env.render())
                if frame.ndim == 4 and frame.shape[0] == 1: frame = frame[0]
                if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                    raise ValueError("Expected HWC uint8 rendering")
                frames.append(frame.copy())

            def step(self, action: dict[str, Any]) -> tuple[Any, ...]:
                actual, inject = applied_action(action, mode=args.mode, phase=self.phase, keys=keys)
                self.pending_control_step = int(as_numpy(raw.elapsed_steps)[0]) + 1
                if inject:
                    report["fault_events"].append({"control_step": self.pending_control_step, "type": "retain_right_gripper_during_release"})
                obs, reward, terminated, truncated, info = env.step(actual)
                done, continued = continue_success_only(terminated, truncated, info)
                ticks.append({"phase": self.phase, "environment_step": int(as_numpy(raw.elapsed_steps)[0]),
                              "requested_actions": json_value(action), "actions": json_value(actual), "fault_injected": inject,
                              "native_terminated": bool(as_numpy(terminated).reshape(-1)[0]),
                              "native_truncated": bool(as_numpy(truncated).reshape(-1)[0]),
                              "success_only_termination_continued": continued,
                              "official_evaluation": json_value(info), **measurement()})
                self.frame()
                return obs, reward, done, truncated, info

        captured = Capture()
        original_hook = raw._after_simulation_step

        def record_substep() -> None:
            original_hook()
            if captured.observe_substeps:
                substeps.append({"substep": len(substeps) + 1, "phase": captured.phase,
                                 "control_step": captured.pending_control_step, "sim_timestep_s": float(raw.sim_timestep),
                                 "object_poses": [as_numpy(a.pose.raw_pose)[0].tolist() for a in targets],
                                 "tcp_poses": [as_numpy(a.tcp.pose.raw_pose)[0].tolist() for a in agents],
                                 "left_qpos": as_numpy(agents[0].robot.get_qpos())[0].tolist(),
                                 "right_qpos": as_numpy(agents[1].robot.get_qpos())[0].tolist(),
                                 "left_qvel": as_numpy(agents[0].robot.get_qvel())[0].tolist(),
                                 "right_qvel": as_numpy(agents[1].robot.get_qvel())[0].tolist(),
                                 "object_linear_velocities": [as_numpy(a.linear_velocity)[0].tolist() for a in targets],
                                 "object_angular_velocities": [as_numpy(a.angular_velocity)[0].tolist() for a in targets]})

        from .dual_arm_execution import _canonical
        arms = _canonical(captured)
        report["urdf_joint_velocity_checks"] = [compact_urdf_velocity_limits(a.urdf_path, arm.names,
            max(args.joint_vel_limit, report["execution_configuration"]["approach_joint_vel_limit_rad_s"]))
            for a, arm in zip(agents, arms)]
        if (not np.isclose(raw.control_timestep, .05, rtol=0, atol=1e-12)
                or not np.isclose(raw.sim_timestep, .01, rtol=0, atol=1e-12)):
            raise ValueError("Compact runs require the unchanged 20Hz control/100Hz physics clock")
        report["physics_timestep_s"] = float(raw.sim_timestep)
        raw._after_simulation_step = record_substep
        captured.frame()
        initial_hold = {k: np.r_[as_numpy(a.robot.get_qpos())[0, :7], 1].astype(np.float32) for k, a in zip(keys, agents)}
        for _ in range(10):
            _, _, done, truncated, _ = captured.step(initial_hold)
            if done or bool(as_numpy(truncated).reshape(-1)[0]): raise RuntimeError("Environment ended during preparation")
        report["preparation_steps"] = 10
        class AuditedSolver(PandaArmMotionPlanningSolver):
            """Retain original native TOPP derivative arrays before conversion."""
            def setup_planner(self) -> Any:
                planner = super().setup_planner()
                native_topp = planner.TOPP

                def traced_topp(path: Any, step: float = .1, verbose: bool = False) -> Any:
                    raw_output = native_topp(path, step, verbose)
                    payload = {"native_path": np.asarray(path).copy(),
                               "solver_velocity_limits": np.asarray(planner.joint_vel_limits).copy(),
                               "solver_acceleration_limits": np.asarray(planner.joint_acc_limits).copy()}
                    for field, value in zip(("time", "position", "velocity", "acceleration", "duration"), raw_output):
                        if value is not None:
                            payload["raw_" + field] = np.asarray(value).copy()
                    # Persist the original output even if reconstruction rejects.
                    np.savez(output / f"{captured.phase}-arm{self.multi_robot_id}-native-topp-raw.npz", **payload)
                    self.last_native_parameterization = compact_topp_reconstruction(
                        path, planner.joint_vel_limits, planner.joint_acc_limits, step, raw_output)
                    return raw_output

                planner.TOPP = traced_topp
                return planner

            def move_to_pose_with_screw(self, pose: Any, dry_run: bool = False,
                                        refine_steps: int = 0, other_gripper_state: Any = None) -> Any:
                if not dry_run:
                    raise ValueError("Compact execution requires plan-only solver calls")
                self.last_native_parameterization = None
                result = super().move_to_pose_with_screw(pose, dry_run=True,
                    refine_steps=refine_steps, other_gripper_state=other_gripper_state)
                self.last_native_result = result
                return result
        solvers = [AuditedSolver(env, base_pose=a.robot.pose, multi_robot_id=i, debug=False,
                    vis=False, visualize_target_grasp_pose=False, print_env_info=False,
                    joint_vel_limits=report["execution_configuration"]["solver_joint_vel_limit_rad_s"],
                    joint_acc_limits=report["execution_configuration"]["solver_joint_acc_limit_rad_s2"])
                   for i, a in enumerate(agents)]
        checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=targets)
        report["initial_collision_coverage"] = json_value(checker.coverage)

        def begin(name: str) -> None:
            report["stage"] = captured.phase = name
            save_report(output / "report.json", report)

        def finish(name: str, result: dict[str, Any]) -> bool:
            report["stages"].append({"name": name, **result})
            if result["status"] != "passed": report.update(status="failed", reason=f"{name}:{result['status']}")
            save_report(output / "report.json", report)
            return result["status"] == "passed"

        def plan(name: str, poses: tuple[Any, Any], *, closed: bool) -> Any:
            phase_limits = compact_phase_limits(report["execution_configuration"], name)
            compact_set_phase_limits(solvers, phase_limits)
            report.setdefault("phase_limit_configurations", []).append(phase_limits)
            before = int(as_numpy(raw.elapsed_steps)[0])
            raw_schedule = plan_dual_pose(*solvers, poses, gripper_commands=(-1, -1) if closed else (1, 1))
            if int(as_numpy(raw.elapsed_steps)[0]) != before: raise RuntimeError("Planning advanced environment")
            raw_results = tuple(solver.last_native_result if target is not None else None for solver, target in zip(solvers, poses))
            traces = tuple(solver.last_native_parameterization if target is not None else None for solver, target in zip(solvers, poses))
            retimed, retiming = compact_retime_pair(raw_results, traces,
                velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"])
            trajectories = tuple(source if result is None else ArmTrajectory(source.action_key, source.joint_names,
                                  result["time"], result["position"], source.gripper_command)
                                 for source, result in zip(raw_schedule.trajectories, retimed))
            schedule = build_hold_schedule(*trajectories, control_timestep_s=raw_schedule.control_timestep_s)
            report.setdefault("phase_time_parameterization", []).append({"phase": name, **retiming,
                "raw_scheduled_steps": raw_schedule.num_steps, "retimed_scheduled_steps": schedule.num_steps})
            derivatives = []
            derivative_arrays = {}
            for index, (solver, target) in enumerate(zip(solvers, poses)):
                if target is None:
                    derivatives.append({"status": "passed", "stationary_hold": True})
                    continue
                result = retimed[index]
                audit = compact_kinematic_audit(result,
                    velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                    acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"])
                audit.update(solver_velocity_limit_rad_s=solver.joint_vel_limits,
                    solver_acceleration_limit_rad_s2=solver.joint_acc_limits,
                    solver_limit_fraction=SOLVER_LIMIT_FRACTION,
                    source_timestamp_scaling_applied=retiming["source_timestamp_scaling_applied"],
                    raw_native_derivative_audit=compact_kinematic_audit(raw_results[index],
                        velocity_limit=phase_limits["nominal_velocity_limit_rad_s"],
                        acceleration_limit=phase_limits["nominal_acceleration_limit_rad_s2"]))
                derivatives.append(audit)
                arm_name = "left" if index == 0 else "right"
                derivative_arrays[f"{arm_name}_velocity"] = np.asarray(result["velocity"])
                derivative_arrays[f"{arm_name}_acceleration"] = np.asarray(result["acceleration"])
                for field in ("time", "position", "velocity", "acceleration", "duration"):
                    derivative_arrays[f"{arm_name}_raw_{field}"] = np.asarray(raw_results[index][field])
                    derivative_arrays[f"{arm_name}_retimed_{field}"] = np.asarray(result[field])
                for field in ("native_path", "path_knots_s", "path_coefficients", "grid_s", "grid_speed_squared",
                              "grid_acceleration", "grid_time", "solver_velocity_limits", "solver_acceleration_limits"):
                    derivative_arrays[f"{arm_name}_{field}"] = traces[index][field]
            report.setdefault("kinematic_audits", []).append({"phase": name, "phase_limits": phase_limits, "arms": derivatives})
            np.savez(output / f"{name}-plan.npz", **derivative_arrays, left_time=schedule.trajectories[0].times_s,
                     right_time=schedule.trajectories[1].times_s, left_position=schedule.trajectories[0].positions_rad,
                     right_position=schedule.trajectories[1].positions_rad, target_times=schedule.target_times_s,
                     left_commands=schedule.left_commands, right_commands=schedule.right_commands)
            if any(audit["status"] != "passed" for audit in derivatives):
                raise RuntimeError(f"Native planner velocity/acceleration limit violation in {name}")
            report.setdefault("planning", []).append({"phase": name, "phase_limits": phase_limits,
                "environment_steps": 0, "scheduled_steps": schedule.num_steps,
                "source_durations_s": [t.duration_s for t in schedule.trajectories],
                "target_tcp_poses": [None if p is None else np.r_[p.p, p.q].tolist() for p in poses]})
            return schedule

        grasp_pairs = (grasp_poses(raw), banana_grasp_poses(raw))
        for index, name, contact in ((0, "pregrasp", False), (1, "descend", True)):
            begin(name)
            schedule = plan(name, (grasp_pairs[0][index], grasp_pairs[1][index]), closed=False)
            if not finish(name, execute_dual_reach(captured, schedule, checker, allow_target_contacts=(contact, contact))): return
        begin("close_and_confirm")
        if not finish("close_and_confirm", close_dual_and_confirm(captured, checker, targets)): return
        begin("attach_collision_prediction")
        checker.attach()
        references = np.asarray(checker.attachments).copy()
        report["attachment_T_tcp_objects"] = references.tolist()
        report["attachment_initial_coverage"] = json_value(checker.coverage)
        baskets = (raw.basket_apple, raw.basket_banana)
        release_offsets = report["execution_configuration"]["release_tcp_offsets_basket_m"]
        release_xyz = [as_numpy(b.pose.p)[0] + offset for b, offset in zip(baskets, release_offsets)]
        route = compact_route_positions([as_numpy(a.tcp.pose.p)[0] for a in agents], release_xyz, route=args.route)
        original_q = [as_numpy(a.tcp.pose.q)[0].copy() for a in agents]
        carry_q = [original_q[0], (sapien.Pose(q=[2**-.5, 0, 0, 2**-.5]) * sapien.Pose(q=original_q[1])).q]
        report["compact_route_world_positions_m"] = json_value(route)
        report["selected_release_tcp_xyz_m"] = [p.tolist() for p in release_xyz]
        lift_poses = tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["lift"], original_q))
        begin("lift_and_hold")
        schedule = plan("lift_and_hold", lift_poses, closed=True)
        captured.observe_substeps = True
        hover = report["execution_configuration"]["airborne_hover_steps"]
        lifted = execute_dual_held_move(captured, schedule, checker, targets,
            min_height_gain_m=(.10, .10), hold_steps=hover)
        if not finish("lift_and_hold", lifted): return
        if hover:
            hover_steps = [int(lifted["initial_elapsed_steps"][0]) + row["step"] for row in lifted["trace"] if row["phase"] == "hover"]
            report["physics_hold_joint_audit"] = hold_joint_substep_audit(substeps, hover_steps)
            if report["physics_hold_joint_audit"]["status"] != "passed":
                report.update(status="failed", reason="physics_hold_joint_stability"); return
        else:
            report["physics_hold_joint_audit"] = {"status": "not_applicable", "observed_hover_steps": 0,
                "reason": "execution profile omits the separate airborne diagnostic hover; old hover regression preserved"}
        waypoints = []
        if route["raise"] is not None:
            waypoints.append(("carry_raise", tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["raise"], original_q))))
        waypoints += [
            ("carry_orient_and_transport", tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["above_baskets"], carry_q))),
            ("carry_lower", tuple(sapien.Pose(xyz, q) for xyz, q in zip(route["release"], carry_q)))]
        for name, poses in waypoints:
            begin(name)
            schedule = plan(name, poses, closed=True)
            if not finish(name, execute_dual_held_move(captured, schedule, checker, targets)): return
        report["attachment_final_coverage"] = json_value(checker.coverage)
        report["held_physics_substep_audit"] = held_substep_audit(substeps, references)
        if report["held_physics_substep_audit"]["status"] != "passed":
            report.update(status="failed", reason="held_physics_substep_envelope"); return
        checker.detach()
        begin("release")
        released = release_dual_and_confirm(captured, checker, targets)
        finish("release", released)
        if args.mode == "retain-right":
            report["retain_right_control_verified"] = retain_right_valid(released, ticks, keys)
            report["status"] = "passed" if report["retain_right_control_verified"] else "failed"
            report["expected_outcome"] = "left releases; right stays grasped; release_timeout; no retreat/verification"
            return
        if released["status"] != "passed": return
        retreat_initial = np.asarray([as_numpy(a.tcp.pose.p)[0].copy() for a in agents])
        retreat_poses = tuple(sapien.Pose(xyz + [0, 0, .08], as_numpy(a.tcp.pose.q)[0]) for xyz, a in zip(retreat_initial, agents))
        begin("retreat")
        schedule = plan("retreat", retreat_poses, closed=False)
        if not finish("retreat", execute_dual_reach(captured, schedule, checker)): return
        actual_retreat = np.asarray([as_numpy(a.tcp.pose.p)[0] for a in agents]) - retreat_initial
        report["actual_retreat_displacement_m"] = actual_retreat.tolist()
        report["actual_retreat_vertical_gain_m"] = actual_retreat[:, 2].tolist()
        if np.any(actual_retreat[:, 2] < .08):
            report.update(status="failed", reason="actual_retreat_below_8cm"); return

        def predicate(index: int) -> Any:
            def measure(actor: Any) -> dict[str, Any]:
                return physical_region(as_numpy(raw._get_actor_geom_center(actor))[0],
                    as_numpy(world_aabb_from_local_mesh(baskets[index], raw.device))[0],
                    as_numpy(world_aabb_from_local_mesh(actor, raw.device))[0])
            return measure
        begin("verify_placement")
        result = verify_dual_placement(captured, checker, targets, region_predicates=(predicate(0), predicate(1)))
        if finish("verify_placement", result):
            evaluation = json_value(raw.evaluate())
            report["physical_task_outcome"] = "passed" if bool(np.asarray(evaluation["success"]).all()) and not report["fault_events"] else "failed"
            report["official_200_step_complete"] = bool(args.max_episode_steps == 200 and report["physical_task_outcome"] == "passed" and int(as_numpy(raw.elapsed_steps)[0]) < 200)
            report["status"] = report["physical_task_outcome"]
        report["stage"] = "finished"
    finally:
        errors = report.setdefault("artifact_errors", [])
        if raw is not None:
            for name, read in (("total_environment_steps", lambda: int(as_numpy(raw.elapsed_steps)[0])),
                               ("torch_cuda_initialized", torch.cuda.is_initialized),
                               ("official_task_evaluation", lambda: json_value(raw.evaluate()))):
                try: report[name] = read()
                except Exception as exc: errors.append(f"final {name}: {type(exc).__name__}: {exc}")
            if original_hook is not None: raw._after_simulation_step = original_hook
        for filename, rows in (("ticks.jsonl", ticks), ("physics-substeps.jsonl", substeps)):
            try:
                with (output / filename).open("w") as f:
                    for row in rows: f.write(json.dumps(json_value(row), allow_nan=False) + "\n")
            except Exception as exc: errors.append(f"{filename}: {type(exc).__name__}: {exc}")
        if env is not None:
            try:
                if getattr(env, "_trajectory_buffer", None) is not None: env.flush_trajectory()
            except Exception as exc: errors.append(f"trajectory: {type(exc).__name__}: {exc}")
            finally:
                try: env.close()
                except Exception as exc: errors.append(f"close: {type(exc).__name__}: {exc}")
            try:
                actions = np.asarray([np.concatenate([t["actions"][k] for k in keys]) for t in ticks]) if ticks else None
                report["recording"] = recording_summary(output / "trajectory.h5", len(ticks), actions)
                report["recording"]["action_key_order"] = keys
            except Exception as exc: errors.append(f"record check: {type(exc).__name__}: {exc}")
        if frames:
            try:
                imageio.mimsave(output / "execution.mp4", [frames[0]] * 10 + frames + [frames[-1]] * 20, fps=20, macro_block_size=1)
                imageio.imwrite(output / "preview.png", frames[-1])
                report["video"] = {"path": str(output / "execution.mp4"), "fps": 20, "padding_frames_before": 10, "padding_frames_after": 20}
            except Exception as exc: errors.append(f"video: {type(exc).__name__}: {exc}")
        report["success_only_termination_continuation_ticks"] = sum(t["success_only_termination_continued"] for t in ticks)
        report["phase_step_counts"] = {phase: sum(t["phase"] == phase for t in ticks) for phase in dict.fromkeys(t["phase"] for t in ticks)}
        report["first_official_success_step"] = next((t["environment_step"] for t in ticks
            if bool(np.asarray(t["official_evaluation"].get("success", False)).all())), None)
        report["official_200_step_complete"] = bool(report.get("official_200_step_complete", False))
        if errors: report.update(status="failed", official_200_step_complete=False)


def retiming_self_test() -> dict[str, Any]:
    """Extra CPU suite: analytic curves, native reconstruction and fail-closed edges."""
    from types import SimpleNamespace
    from mani_skill.examples.motionplanning.planner import Planner
    tests = []

    def toy(*, cubic: float = 0., quadratic: float = 0., linear: float = 0.,
            x0: float = 1., x1: float = 1.) -> tuple[dict[str, Any], dict[str, Any]]:
        u = (x1 - x0) / 2
        duration = 2 / (math.sqrt(x0) + math.sqrt(x1))
        coefficients = np.zeros((4, 1, 7)); coefficients[:, 0, 0] = [cubic, quadratic, linear, 0]
        raw = {"status": "Success", "time": np.array([0., duration]),
               "position": np.zeros((2, 7)), "velocity": np.zeros((2, 7)),
               "acceleration": np.zeros((2, 7)), "duration": duration}
        raw["position"][:, 0] = [0., cubic + quadratic + linear]
        raw["velocity"][:, 0] = [linear * math.sqrt(x0), (3 * cubic + 2 * quadratic + linear) * math.sqrt(x1)]
        raw["acceleration"][:, 0] = [2 * quadratic * x0 + linear * u,
            (6 * cubic + 2 * quadratic) * x1 + (3 * cubic + 2 * quadratic + linear) * u]
        trace = {"path_knots_s": np.array([0., 1.]), "path_coefficients": coefficients,
                 "grid_s": np.array([0., 1.]), "grid_speed_squared": np.array([x0, x1]),
                 "grid_acceleration": np.array([u])}
        trace.update({"sample_" + k: raw[k].copy() for k in ("time", "position", "velocity", "acceleration")})
        return raw, trace

    def reject(call: Any) -> None:
        try: call()
        except (ValueError, RuntimeError, FloatingPointError): return
        raise AssertionError("Invalid or mismatched retiming input accepted")

    raw, trace = toy(cubic=-1., quadratic=4., x0=1., x1=2.)
    bounds = compact_continuous_derivative_bounds(trace)
    assert np.max(np.abs(raw["acceleration"])) == 8.
    assert np.isclose(bounds["max_acceleration_rad_s2"], 9.2, rtol=0, atol=1e-12)
    assert np.isclose(bounds["acceleration_witness"]["path_s"], .4, rtol=0, atol=1e-12)
    corrected, receipt = compact_retime_pair((raw, None), (trace, None), velocity_limit=100., acceleration_limit=8.5)
    scale = receipt["shared_phase_scale"]
    assert scale > 1. and receipt["arms"][0]["retimed_max_acceleration_rad_s2"] <= 8.5 + 1e-12
    assert np.array_equal(corrected[0]["position"], raw["position"])
    assert np.array_equal(trace["sample_time"], raw["time"])
    assert np.allclose(corrected[0]["velocity"] * scale, raw["velocity"], rtol=0, atol=1e-14)
    assert np.allclose(corrected[0]["acceleration"] * scale ** 2, raw["acceleration"], rtol=0, atol=1e-14)
    tests.append("analytic_internal_acceleration_peak_missed_by_raw_samples_is_bounded")
    # Independent finite differences of q(s(t/s_scale)) check the chain rule.
    at = .4 * raw["duration"] * scale; h = 1e-4
    def position(t: float) -> float:
        old_t = t / scale; path_s = old_t + .25 * old_t ** 2
        return -path_s ** 3 + 4 * path_s ** 2
    old_t = at / scale; path_s = old_t + .25 * old_t ** 2
    expected_v = (-3 * path_s ** 2 + 8 * path_s) * (1 + .5 * old_t) / scale
    expected_a = (8 + 6 * path_s - 7.5 * path_s ** 2) / scale ** 2
    assert abs((position(at + h) - position(at - h)) / (2 * h) - expected_v) < 1e-6
    assert abs((position(at + h) - 2 * position(at) + position(at - h)) / h ** 2 - expected_a) < 1e-5
    tests.append("independent_curve_finite_differences_confirm_time_dilation_chain_rule")
    left, lt = toy(cubic=-.05, quadratic=.2, x0=1., x1=2.)
    right, rt = toy(linear=3.)
    _, v_dominates = compact_retime_pair((left, right), (lt, rt), velocity_limit=1., acceleration_limit=.1)
    _, a_dominates = compact_retime_pair((left, right), (lt, rt), velocity_limit=1., acceleration_limit=.02)
    assert np.isclose(v_dominates["shared_phase_scale"], 3.)
    assert np.isclose(a_dominates["shared_phase_scale"], math.sqrt(.46 / .02))
    tests.append("same_shared_factor_covers_velocity_and_acceleration_from_different_arms")
    small, st = toy(linear=.1)
    unchanged, same = compact_retime_pair((small, None), (st, None), velocity_limit=1., acceleration_limit=1.)
    assert same["shared_phase_scale"] == 1. and unchanged[1] is None
    assert np.array_equal(unchanged[0]["time"], small["time"])
    assert compact_retime_pair((None, None), (None, None), velocity_limit=1., acceleration_limit=1.)[0] == (None, None)
    zero_start, zt = toy(linear=1., x0=0., x1=1.)
    zb = compact_continuous_derivative_bounds(zt)
    assert zb["max_velocity_rad_s"] == 1. and zb["max_acceleration_rad_s2"] == .5
    tests.append("scale_one_stationary_none_zero_acceleration_and_zero_initial_speed")
    boundary = dict(st)
    boundary["path_knots_s"] = np.array([0., .5, 1.])
    boundary["path_coefficients"] = np.zeros((4, 2, 7))
    boundary["path_coefficients"][2, :, 0] = .1
    boundary["path_coefficients"][3, 1, 0] = .05
    boundary["grid_s"] = np.array([0., .25, .75, 1.])
    boundary["grid_speed_squared"] = np.ones(4); boundary["grid_acceleration"] = np.zeros(3)
    assert compact_continuous_derivative_bounds(boundary)["common_intervals"] == 4
    tests.append("different_spline_and_parameterization_boundaries_are_all_partitioned")
    reject(lambda: compact_continuous_derivative_bounds({**trace, "grid_s": [0, 0]}))
    reject(lambda: compact_continuous_derivative_bounds({**trace, "grid_speed_squared": [-1, 2]}))
    reject(lambda: compact_continuous_derivative_bounds({**trace, "grid_acceleration": [float("nan")]}))
    reject(lambda: compact_continuous_derivative_bounds({**trace, "path_coefficients": np.zeros((4, 1, 6))}))
    reject(lambda: compact_retime_pair((raw, None), (trace, None), velocity_limit=0., acceleration_limit=1.))
    reject(lambda: compact_retime_pair((raw, None), (None, None), velocity_limit=1., acceleration_limit=1.))
    reject(lambda: compact_retime_pair((raw, None), ({**trace, "sample_position": raw["position"] + 1e-6}, None), velocity_limit=1., acceleration_limit=1.))
    tests.append("nonfinite_invalid_shapes_clock_speed_limits_and_certificate_mismatch_rejected")
    s = np.linspace(0, 1, 21)
    points = np.column_stack((.2 * s, .1 * s ** 2, .08 * np.sin(1.2 * s), np.zeros((len(s), 4))))
    planner = SimpleNamespace(joint_vel_limits=np.full(7, .5), joint_acc_limits=np.full(7, 1.))
    output = Planner.TOPP(planner, points, .05)
    rebuilt = compact_topp_reconstruction(points, planner.joint_vel_limits, planner.joint_acc_limits, .05, output)
    assert np.array_equal(rebuilt["sample_position"], output[1])
    reject(lambda: compact_topp_reconstruction(points, planner.joint_vel_limits, planner.joint_acc_limits, .05,
                                               (output[0], output[1] + 1e-8, *output[2:])))
    tests.append("untouched_native_TOPP_rebuild_is_exact_and_mismatch_fails_closed")
    return {"status": "passed", "tests": tests, "physics_steps": 0, "environment_created": False,
            "toy_raw_sample_acceleration_peak": 8., "toy_analytic_acceleration_peak": bounds["max_acceleration_rad_s2"],
            "policy": RETIMING_POLICY}


def self_test() -> dict[str, Any]:
    """Synthetic geometry, parser config and native derivative/URDF checks only."""
    import tempfile
    original = original_self_test()
    args = argparse.Namespace(route="merge-lift", execution_profile="execution", max_episode_steps=200,
        joint_vel_limit=.9, joint_acc_limit=.9)
    config = compact_configuration(args)
    assert config["airborne_hover_steps"] == 0 and config["max_episode_steps"] == 200
    assert config["final_common_placement_steps"] == 20 and config["source_timestamp_scaling"]
    assert config["time_parameterization_policy"] == RETIMING_POLICY and not config["native_source_timestamp_scaling"]
    assert np.asarray(config["release_tcp_offsets_basket_m"]).shape == (2, 3)
    assert np.isfinite(config["release_tcp_offsets_basket_m"]).all()
    assert config["release_pose_policy"] == RELEASE_POLICY
    assert config["joint_vel_limit_rad_s"] == config["joint_acc_limit_rad_s2"] == .9
    assert config["solver_limit_fraction"] == .99
    assert config["solver_joint_vel_limit_rad_s"] == config["solver_joint_acc_limit_rad_s2"] == .891
    assert compact_phase_limits(config, "pregrasp")["nominal_velocity_limit_rad_s"] == .9
    args.joint_vel_limit, args.joint_acc_limit = 1.5, 2.4
    args.approach_vel_limit, args.approach_acc_limit = .9, .9
    separated = compact_configuration(args)
    approach = compact_phase_limits(separated, "pregrasp")
    assert compact_phase_limits(separated, "descend")["nominal_acceleration_limit_rad_s2"] == .9
    assert approach["nominal_velocity_limit_rad_s"] == approach["nominal_acceleration_limit_rad_s2"] == .9
    assert approach["solver_velocity_limit_rad_s"] == approach["solver_acceleration_limit_rad_s2"] == .891
    for phase in ("lift_and_hold", "carry_raise", "carry_orient_and_transport", "carry_lower", "retreat"):
        limits = compact_phase_limits(separated, phase)
        assert limits["nominal_velocity_limit_rad_s"] == 1.5
        assert limits["nominal_acceleration_limit_rad_s2"] == 2.4
    from types import SimpleNamespace
    fake_solvers = [SimpleNamespace(planner=SimpleNamespace(move_group_joint_indices=list(range(7)))) for _ in range(2)]
    compact_set_phase_limits(fake_solvers, approach)
    assert all(np.array_equal(s.planner.joint_vel_limits, np.full(7, .891)) for s in fake_solvers)
    compact_set_phase_limits(fake_solvers, compact_phase_limits(separated, "retreat"))
    assert all(np.array_equal(s.planner.joint_acc_limits, np.full(7, 2.4 * .99)) for s in fake_solvers)
    args.joint_vel_limit = args.joint_acc_limit = .9
    args.approach_vel_limit = args.approach_acc_limit = None
    args.execution_profile = "diagnostic-hover"
    assert compact_configuration(args)["airborne_hover_steps"] == 20
    args.max_episode_steps = 400
    assert not compact_configuration(args)["official_budget_candidate"]
    def reject(call: Any) -> None:
        try: call()
        except (ValueError, TypeError): return
        raise AssertionError("Invalid compact configuration or input was accepted")
    args.approach_vel_limit = -1.; reject(lambda: compact_configuration(args))
    args.approach_vel_limit = None
    args.max_episode_steps = 199; reject(lambda: compact_configuration(args))
    args.max_episode_steps = 200; args.joint_vel_limit = float("nan")
    reject(lambda: compact_configuration(args))
    current = [[-.2, -.1, .1], [.2, -.1, .11]]
    release = [[-.3, .2, .17], [.3, .2, .17]]
    merged = compact_route_positions(current, release, route="merge-lift")
    separate = compact_route_positions(current, release, route="merge-transport")
    assert merged["raise"] is None and separate["raise"].shape == (2, 3)
    assert np.array_equal(merged["lift"][:, :2], np.asarray(current)[:, :2])
    assert np.allclose(merged["above_baskets"][:, 2], [.23, .23])
    assert np.allclose(separate["lift"][:, 2], [.22, .23])
    assert np.array_equal(merged["release"], release)
    assert np.all(merged["lift"][:, 2] - np.asarray(current)[:, 2] >= .12 - 1e-12)
    # Isolate the unchanged lift floor from the changed transport clearance.
    mixed_current = np.array([[-.2, -.1, .5], [.2, -.1, .1]])
    mixed_release = np.array([[-.3, .2, .02], [.3, .2, .35]])
    initial_current, initial_release = mixed_current.copy(), mixed_release.copy()
    for selected_route in ROUTES:
        mixed = compact_route_positions(mixed_current, mixed_release, route=selected_route)
        assert np.allclose(mixed["clearance_z"], [.62, .41], rtol=0, atol=1e-12)
        assert mixed["clearance_z"][0] == mixed_current[0, 2] + .12
        assert mixed["clearance_z"][1] == mixed_release[1, 2] + .06
        assert np.array_equal(mixed["release"], initial_release)
        assert np.array_equal(mixed["above_baskets"][:, :2], initial_release[:, :2])
        assert np.array_equal(mixed_current, initial_current) and np.array_equal(mixed_release, initial_release)
        assert np.all(mixed["lift"][:, 2] - mixed_current[:, 2] >= .12 - 1e-12)
    assert config["transport_clearance_above_release_m"] == .06
    assert config["minimum_commanded_lift_m"] == .12
    assert config["route_geometry_policy"] == "compact-clearance-v2"
    reject(lambda: compact_route_positions(current[:1], release, route="merge-lift"))
    reject(lambda: compact_route_positions(current, [[0, 0, np.inf]] * 2, route="merge-lift"))
    native = {"status": "Success", "time": [0., .1, .2], "position": np.zeros((3, 7)),
              "velocity": np.full((3, 7), .9), "acceleration": np.full((3, 7), -.9)}
    assert compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)["status"] == "passed"
    native["velocity"][1, 0] = .91
    assert compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)["status"] == "failed"
    native["velocity"][1, 0] = .9; native["acceleration"][2, 6] = -.91
    assert compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)["status"] == "failed"
    native["acceleration"][2, 6] = .9000788313717154
    original_overshoot = compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)
    assert original_overshoot["status"] == "failed" and original_overshoot["numeric_tolerance"] == 1e-6
    native["acceleration"][2, 6] = .8910788313717154
    assert compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9)["status"] == "passed"
    native["acceleration"] = np.zeros((2, 7))
    reject(lambda: compact_kinematic_audit(native, velocity_limit=.9, acceleration_limit=.9))
    names = [f"joint{i}" for i in range(7)]
    with tempfile.TemporaryDirectory(prefix="tig-compact-urdf-") as td:
        path = Path(td) / "robot.urdf"
        path.write_text("<robot>" + "".join(f'<joint name="{n}"><limit velocity="2.0"/></joint>' for n in names) + "</robot>")
        audit = compact_urdf_velocity_limits(path, names, 1.5)
        assert audit["velocity_limits_rad_s"] == [2.] * 7
        reject(lambda: compact_urdf_velocity_limits(path, names, 2.01))
        reject(lambda: compact_urdf_velocity_limits(path, names[:-1] + ["unknown"], .9))
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
        "array_shapes": {"paired_world_positions": [2, 3], "native_derivatives": [3, 7]},
        "checks": ["original frozen acceptance helper regression", "official200 and explicit extended diagnostic budget",
            "execution vs retained20 hover profile", "vertical clearance and >=.12 commanded lift geometry",
            "separate-vs-merged lift endpoints", "mixed lift-floor/clearance branch preserves release endpoints and inputs",
            "native derivative limits and malformed array rejection",
            "seven URDF velocity limits and over-limit rejection",
            "explicit99pct solver envelope; nominal .9 gate rejects observed original overshoot",
            "approach overrides apply only to pregrasp/descend; runtime MPLib arrays update for held/retreat"], "original_checks": original["checks"],
        "retiming_suite": retiming_self_test()}


def main() -> int:
    faulthandler.enable()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("positive", "retain-right"), default="positive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--route", choices=ROUTES, default="merge-lift")
    parser.add_argument("--execution-profile", choices=PROFILES, default="execution")
    parser.add_argument("--max-episode-steps", type=int, default=200,
                        help="200 is official; 201..400 are explicitly extended timing diagnostics")
    parser.add_argument("--joint-vel-limit", type=float, default=.9)
    parser.add_argument("--joint-acc-limit", type=float, default=.9)
    parser.add_argument("--approach-vel-limit", type=float,
                        help="Nominal limit only for pregrasp/descend; defaults to global joint velocity limit")
    parser.add_argument("--approach-acc-limit", type=float,
                        help="Nominal limit only for pregrasp/descend; defaults to global joint acceleration limit")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    try: configuration = compact_configuration(args)
    except ValueError as exc: parser.error(str(exc))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu): parser.error("Select one available GPU UUID for rendering")
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime(f"sync-compact-{args.route}-{args.execution_profile}-{args.mode}-%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    names = ["trajectory_schedule.py", "trajectory_execution.py", "combined_collision.py", "scene_collision.py", "grasp_collision.py",
             "grasp_execution.py", "validate_grasp.py", "held_collision.py", "lift_execution.py", "validate_lift.py", "place_execution.py",
             "validate_place.py", "single_arm_execution.py", "table_support_collision.py", "validate_banana.py",
             "dual_payload_collision.py", "dual_arm_execution.py", "validate_sync.py", "validate_sync_compact.py"]
    sources = {n: (HERE / n).read_bytes() for n in names}
    snapshot = output / "source"; snapshot.mkdir()
    for name, contents in sources.items(): (snapshot / name).write_bytes(contents)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "task": TASK, "seed": args.seed,
        "mode": args.mode, "gpu_uuid": gpu, "scope": "synchronized left apple and right banana on one physical control timeline",
        "synchronization": "phase barriers; simultaneous start; each arm holds final joint and gripper targets until both ready",
        "max_episode_steps": args.max_episode_steps, "official_default_max_episode_steps": 200,
        "execution_configuration": configuration,
        "source_reference": str(ROOT / "infrastructure/validation/dual-arm/sync-positive-20261115T134421956872Z/report.json"),
        "official_time_limit_policy": "stop at native truncation including tick200; full physical closure should finish by199",
        "termination_policy": "retain native success termination in H5/ticks; continue success-only signals to verify actual release and settling; failure and truncation stop",
        "physical_attachment_created": False, "continuous_collision_detection": False,
        "source_snapshot": str(snapshot), "source_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()}}
    save_report(output / "report.json", report)
    started = time.monotonic()
    try: native_run(args, output, report)
    except Exception as exc: report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
    save_report(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
