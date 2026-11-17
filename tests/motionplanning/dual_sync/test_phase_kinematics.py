from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any
import numpy as np
from mani_skill.examples.motionplanning.dual.sync.phase_kinematics import (
    SOLVER_LIMIT_FRACTION, RETIMING_POLICY, ROUTES, compact_phase_limits, compact_set_phase_limits, compact_route_positions, compact_kinematic_audit, compact_topp_reconstruction, compact_continuous_derivative_bounds, compact_retime_pair
)


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


def test_runtime_contract():
    assert self_test()["status"] == "passed"


def test_topp_capture_preserves_native_output():
    from types import SimpleNamespace

    from mani_skill.examples.motionplanning.planner import Planner
    from mani_skill.examples.motionplanning.dual.sync.phase_kinematics import attach_topp_capture

    planner = SimpleNamespace(joint_vel_limits=np.full(7, .5), joint_acc_limits=np.full(7, 1.))
    planner.TOPP = lambda path, step, verbose=False: Planner.TOPP(planner, path, step, verbose)
    solver = SimpleNamespace(planner=planner)
    x = np.linspace(0, 1, 21)
    points = np.column_stack((.2 * x, .1 * x ** 2, .08 * np.sin(1.2 * x), np.zeros((len(x), 4))))
    expected = planner.TOPP(points, .05)
    attach_topp_capture(solver)
    actual = planner.TOPP(points, .05)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    assert np.array_equal(solver.last_native_parameterization["sample_position"], actual[1])
    try:
        attach_topp_capture(solver)
    except ValueError:
        pass
    else:
        raise AssertionError("A planner may only be instrumented once")
