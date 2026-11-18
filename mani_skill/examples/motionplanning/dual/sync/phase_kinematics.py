"""CPU phase limits, compact route endpoints and native TOPP parameterization.

Numerical policy: seconds, named joint radians, world XYZ metres, one
common phase time scale, and immutable source samples. These helpers do not own
an environment or certify physical/controller motion or collision safety.
TOPPRA is loaded only when native curve reconstruction is requested.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np



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
        joints = len(solver.planner.move_group_joint_indices)
        if joints < 1:
            raise ValueError("Phase limits require a nonempty named move group")
        solver.joint_vel_limits, solver.joint_acc_limits = speed, acceleration
        solver.planner.joint_vel_limits = np.full(joints, speed, dtype=float)
        solver.planner.joint_acc_limits = np.full(joints, acceleration, dtype=float)
        assert solver.planner.joint_vel_limits.shape == solver.planner.joint_acc_limits.shape == (joints,)

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
    joints = positions.shape[1] if positions.ndim == 2 else 0
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0)
            or joints < 1 or any(a.shape != (len(times), joints) for a in (positions, velocities, accelerations))
            or not all(np.isfinite(a).all() for a in (times, positions, velocities, accelerations, limits))
            or np.any(limits <= 0)):
        raise ValueError("Need finite time[N], position/velocity/acceleration[N,J] and positive limits")
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
    joints = points.shape[1] if points.ndim == 2 else 0
    if (points.ndim != 2 or joints < 1 or len(points) < 2
            or velocity.shape != (joints,) or acceleration.shape != (joints,)
            or not all(np.isfinite(a).all() for a in (points, velocity, acceleration))
            or min(float(velocity.min()), float(acceleration.min()), float(step)) <= 0
            or not np.isfinite(step) or not isinstance(raw_output, (tuple, list)) or len(raw_output) != 5):
        raise ValueError("Need finite native path[N,J], positive matching joint limits, step and raw TOPP tuple")
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
    joints = coefficients.shape[2] if coefficients.ndim == 3 else 0
    sampled = np.asarray(trace.get("sample_position", np.empty((0, joints))))
    if (knots.ndim != 1 or len(knots) < 2 or grid.ndim != 1 or len(grid) < 2
            or joints < 1 or coefficients.shape != (4, len(knots) - 1, joints) or speeds.shape != grid.shape
            or sampled.ndim != 2 or sampled.shape[1] != joints
            or controls.shape != (len(grid) - 1,)
            or not all(np.isfinite(a).all() for a in (knots, coefficients, grid, speeds, controls))
            or np.any(np.diff(knots) <= 0) or np.any(np.diff(grid) <= 0) or np.any(speeds < 0)
            or knots[0] != grid[0] or knots[-1] != grid[-1]
            or not np.allclose(np.diff(speeds), 2 * controls * np.diff(grid), rtol=1e-10, atol=1e-10)):
        raise ValueError("Malformed cubic path or constant-acceleration parameterization")
    cuts = np.unique(np.r_[knots, grid])
    max_velocity, max_acceleration = np.zeros(joints), np.zeros(joints)
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
            for axis in range(joints):
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
                peaks = np.abs(values[indices, np.arange(joints)])
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



def attach_topp_capture(solver: Any) -> None:
    """Retain the latest native result and its continuous path parameterization.

    Install once after constructing an ordinary Panda planner. The planner's
    actual TOPP result is returned unchanged; reconstruction supplies the cubic
    path/velocity data needed for analytic phase retiming. No files are written.
    """
    planner = solver.planner
    if getattr(planner, "_sync_topp_capture", False):
        raise ValueError("TOPP capture is already installed on this planner")
    native_topp = planner.TOPP
    solver.last_native_result = None
    solver.last_native_parameterization = None

    def traced_topp(path: Any, step: float = .1, verbose: bool = False) -> Any:
        raw = native_topp(path, step, verbose)
        solver.last_native_parameterization = compact_topp_reconstruction(
            path, planner.joint_vel_limits, planner.joint_acc_limits, step, raw)
        return raw

    planner.TOPP = traced_topp
    planner._sync_topp_capture = True
