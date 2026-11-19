"""Action-limit measurements must not affect the native action pipeline."""
import json

import numpy as np
import pytest

from examples.baselines.act.action_limit_diagnostics import ActionLimitDiagnostics


def make_diagnostics():
    coordinates = [
        {"arm": "left", "key": "arx_x5-0", "joint": "joint1", "kind": "arm", "unit": "rad"},
        {"arm": "left", "key": "arx_x5-0", "joint": "gripper", "kind": "gripper", "unit": "normalized"},
    ]
    low = np.array([-2, -1], dtype=np.float32)
    high = np.array([2, 1], dtype=np.float32)
    return ActionLimitDiagnostics(coordinates, low, high), coordinates, low, high


def test_mixed_units_and_directional_counts_are_not_converted():
    diagnostic, _, low, high = make_diagnostics()
    for requested in [np.array([2.25, -1.5], dtype=np.float32),
                      np.array([-2.75, 1.25], dtype=np.float32),
                      np.array([0.1, 0.5], dtype=np.float32)]:
        diagnostic.record(requested / 2, requested, np.clip(requested, low, high))
    summary = diagnostic.summary()
    assert (summary["total_steps"], summary["any_clip_steps"],
            summary["total_clipped_coordinates"]) == (3, 2, 4)
    joint, gripper = summary["by_coordinate"]
    assert joint["unit"] == "rad" and gripper["unit"] == "normalized"
    for entry in (joint, gripper):
        assert (entry["lower_clip_count"], entry["upper_clip_count"],
                entry["total_clip_count"]) == (1, 1, 2)
    assert joint["requested_min"] == -2.75
    assert joint["requested_max"] == 2.25
    assert joint["bound_span_fraction_max"] == 0.75 / 4
    assert gripper["bound_span_fraction_max"] == 0.5 / 2
    json.dumps(summary, allow_nan=False)


def test_single_float32_increment_is_counted_without_tolerance():
    diagnostic, _, low, high = make_diagnostics()
    requested = np.array([np.nextafter(high[0], np.float32(np.inf)),
                          np.nextafter(low[1], np.float32(-np.inf))])
    diagnostic.record(np.zeros(2, dtype=np.float32), requested, np.clip(requested, low, high))
    assert diagnostic.summary()["any_clip_steps"] == 1
    joint, gripper = diagnostic.summary()["by_coordinate"]
    assert joint["upper_clip_count"] == gripper["lower_clip_count"] == 1
    assert joint["max_abs_excess_on_clipped"] == float(requested[0]) - float(high[0])
    assert gripper["max_abs_excess_on_clipped"] == float(low[1]) - float(requested[1])


def test_legal_actions_include_exact_endpoints_and_have_no_conditional_distribution():
    diagnostic, _, low, high = make_diagnostics()
    for requested in [low, high, np.zeros(2, dtype=np.float32)]:
        diagnostic.record(requested, requested, requested)
    summary = diagnostic.summary()
    assert summary["total_steps"] == 3
    assert summary["any_clip_steps"] == summary["total_clipped_coordinates"] == 0
    for entry in summary["by_coordinate"]:
        for name in ("max", "p50", "p95", "mean"):
            assert entry[f"{name}_abs_excess_on_clipped"] is None
        assert entry["bound_span_fraction_max"] is None


def test_statistics_are_conditional_on_actual_excess_only():
    diagnostic, _, low, high = make_diagnostics()
    for value in [2.25, 0, 0, 0, 0, 2.75]:
        requested = np.array([value, 0], dtype=np.float32)
        diagnostic.record(requested, requested, np.clip(requested, low, high))
    joint = diagnostic.summary()["by_coordinate"][0]
    assert joint["p50_abs_excess_on_clipped"] == 0.5
    assert joint["p95_abs_excess_on_clipped"] == pytest.approx(0.725)
    assert joint["mean_abs_excess_on_clipped"] == 0.5


def test_inputs_snapshots_metadata_and_returned_arrays_are_independent():
    diagnostic, coordinates, low, high = make_diagnostics()
    normalized = np.array([0.5, 1.5], dtype=np.float32)
    requested = np.array([1.5, 1.2], dtype=np.float32)
    executed = np.clip(requested, low, high)
    originals = [value.copy() for value in (normalized, requested, executed, low, high)]
    diagnostic.record(normalized, requested, executed)
    for original, value in zip(originals, (normalized, requested, executed, low, high)):
        np.testing.assert_array_equal(value, original)
    normalized[:] = requested[:] = executed[:] = low[:] = high[:] = 99
    coordinates[0]["joint"] = "changed"
    saved = diagnostic.arrays()
    for index, (key, original) in enumerate(zip(
            ("normalized", "requested", "executed", "low", "high"), originals)):
        np.testing.assert_array_equal(saved[key], original[None] if index < 3 else original)
    assert diagnostic.summary()["by_coordinate"][0]["joint"] == "joint1"
    saved["requested"][:] = 99
    saved["high"][:] = 99
    assert diagnostic.summary()["by_coordinate"][1]["upper_clip_count"] == 1
    assert diagnostic.arrays()["high"][1] == 1


def test_arrays_roundtrip_npz_without_pickle_and_preserve_float32(tmp_path):
    diagnostic, _, low, high = make_diagnostics()
    requested = np.array([2.25, 0], dtype=np.float32)
    diagnostic.record(requested / 2, requested, np.clip(requested, low, high))
    path = tmp_path / "actions.npz"
    np.savez_compressed(path, **diagnostic.arrays())
    with np.load(path, allow_pickle=False) as restored:
        for key, array in diagnostic.arrays().items():
            np.testing.assert_array_equal(restored[key], array)
            assert restored[key].dtype == np.float32


def test_no_records_is_well_defined():
    diagnostic, _, _, _ = make_diagnostics()
    assert diagnostic.summary()["total_steps"] == 0
    assert diagnostic.summary()["any_clip_steps"] == 0
    assert diagnostic.arrays()["requested"].shape == (0, 2)
    assert diagnostic.summary()["by_coordinate"][0]["requested_min"] is None


def test_zero_width_and_unbounded_coordinates_have_no_relative_excess():
    diagnostic = ActionLimitDiagnostics([{}, {}], np.array([0, -np.inf]), np.array([0, 1]))
    requested = np.array([0.25, 1.25])
    diagnostic.record(requested, requested, np.array([0, 1]))
    assert all(entry["bound_span_fraction_max"] is None
               for entry in diagnostic.summary()["by_coordinate"])


def test_wrong_shapes_are_reported_as_diagnostic_api_errors():
    with pytest.raises(ValueError, match="bounds"):
        ActionLimitDiagnostics([{}], np.zeros((1, 1)), np.ones(1))
    with pytest.raises(ValueError, match="metadata"):
        ActionLimitDiagnostics([], np.zeros(1), np.ones(1))
    diagnostic, _, _, _ = make_diagnostics()
    with pytest.raises(ValueError, match="snapshots"):
        diagnostic.record(np.zeros((1, 2)), np.zeros(2), np.zeros(2))
    assert diagnostic.summary()["total_steps"] == 0
