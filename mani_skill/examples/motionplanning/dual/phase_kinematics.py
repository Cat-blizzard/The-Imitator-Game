"""CPU phase limits, compact route endpoints and native TOPP parameterization.

Extracted from the complete runner's existing helper dependency without changing
its numerical policy: seconds, seven joint radians, world XYZ metres, one
common phase time scale, and immutable source samples. These helpers do not own
an environment or certify physical/controller motion or collision safety.
Runtime imports contain no validation entrypoints. TOPPRA is loaded only when
native curve reconstruction is requested; the native Planner is used only by
the independent CPU self-test. Run this file directly with --self-test.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

# Extraction source SHA-256: ecc2088e12947921d829056d02cf11041f048a6da682497ed2a5871ee6758409


SOLVER_LIMIT_FRACTION = .99

RETIMING_POLICY = "analytic-phase-retiming-v1"

ROUTES = ("merge-transport", "merge-lift")

TRANSPORT_CLEARANCE_ABOVE_RELEASE_M = .06

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
    """Existing numerical boundary suite plus phase/route contracts, CPU only."""
    from types import SimpleNamespace

    def reject(call: Any) -> None:
        try:
            call()
        except (ValueError, TypeError, RuntimeError, FloatingPointError):
            return
        raise AssertionError("Invalid phase kinematics input was accepted")

    configuration = {"joint_vel_limit_rad_s": 1.5,
        "joint_acc_limit_rad_s2": 2.4, "approach_joint_vel_limit_rad_s": .9,
        "approach_joint_acc_limit_rad_s2": .9, "solver_limit_fraction": SOLVER_LIMIT_FRACTION}
    approach = compact_phase_limits(configuration, "pregrasp")
    assert compact_phase_limits(configuration, "descend")["nominal_acceleration_limit_rad_s2"] == .9
    assert approach["solver_velocity_limit_rad_s"] == approach["solver_acceleration_limit_rad_s2"] == .891
    for phase in ("lift_and_hold", "carry_raise", "carry_orient_and_transport", "carry_lower", "retreat"):
        limits = compact_phase_limits(configuration, phase)
        assert limits["nominal_velocity_limit_rad_s"] == 1.5
        assert limits["nominal_acceleration_limit_rad_s2"] == 2.4
    solvers = [SimpleNamespace(planner=SimpleNamespace(move_group_joint_indices=list(range(7)))) for _ in range(2)]
    compact_set_phase_limits(solvers, approach)
    assert all(np.array_equal(s.planner.joint_vel_limits, np.full(7, .891)) for s in solvers)
    compact_set_phase_limits(solvers, compact_phase_limits(configuration, "retreat"))
    assert all(np.array_equal(s.planner.joint_acc_limits, np.full(7, 2.4 * .99)) for s in solvers)
    reject(lambda: compact_phase_limits(configuration, ""))
    reject(lambda: compact_phase_limits({**configuration, "solver_limit_fraction": 1.1}, "retreat"))
    reject(lambda: compact_set_phase_limits(solvers[:1], approach))

    current = np.array([[-.2, -.1, .5], [.2, -.1, .1]])
    release = np.array([[-.3, .2, .02], [.3, .2, .35]])
    original_current, original_release = current.copy(), release.copy()
    for route in ROUTES:
        result = compact_route_positions(current, release, route=route)
        assert np.allclose(result["clearance_z"], [.62, .41])
        assert np.array_equal(result["release"], release)
        assert np.array_equal(result["lift"][:, :2], current[:, :2])
        assert np.array_equal(current, original_current) and np.array_equal(release, original_release)
        if route == "merge-lift":
            assert result["raise"] is None and np.allclose(result["lift"][:,2], [.62, .41])
        else:
            assert np.allclose(result["lift"][:,2], [.62, .22])
    reject(lambda: compact_route_positions(current[:1], release, route="merge-lift"))
    reject(lambda: compact_route_positions(current, [[0.,0.,np.inf]]*2, route="merge-lift"))

    raw = {"status":"Success", "time":[0.,.1,.2], "position":np.zeros((3,7)),
        "velocity":np.full((3,7),.9), "acceleration":np.full((3,7),.9)}
    assert compact_kinematic_audit(raw, velocity_limit=.9, acceleration_limit=.9)["status"] == "passed"
    raw["acceleration"][2,6] = .9000788313717154
    failed = compact_kinematic_audit(raw, velocity_limit=.9, acceleration_limit=.9)
    assert failed["status"] == "failed" and failed["numeric_tolerance"] == 1e-6
    raw["acceleration"][2,6] = .8910788313717154
    assert compact_kinematic_audit(raw, velocity_limit=.9, acceleration_limit=.9)["status"] == "passed"
    reject(lambda: compact_kinematic_audit({**raw,"time":[0.,0.,.2]}, velocity_limit=.9, acceleration_limit=.9))
    return {"status":"passed", "environment_steps":0, "environment_created":False, "gpu_used":False,
        "checks":["unchanged approach/global phase profiles and native solver arrays",
            "both compact routes, lift floor, paired world XYZ and input preservation",
            "unchanged observed derivative overshoot and numeric tolerance",
            "malformed phase/route/clock inputs reject"],
        "retiming_suite":retiming_self_test()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
