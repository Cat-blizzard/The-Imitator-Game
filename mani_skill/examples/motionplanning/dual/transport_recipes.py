"""Legacy pill-only direct transport selector, without diagnostic dependencies.

The route's release positions are finite world-frame XYZ metres, shape (2, 3).
Both effective measured source-departure flags must be literal Python True.
The caller retains physical grasp/lift/departure validation, planning, collision
checks, execution and release handling. This pure selector returns one owned
copy of the unchanged release target and never creates or steps an environment.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import numpy as np

PILL = "TwoRobotPickPillToRegions-v1"


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite {label}{shape}")
    return result


def _true_pair(value: Any, label: str) -> None:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(v is not True for v in value):
        raise ValueError(f"Both measured {label} must be boolean True")


def pill_direct_transport_recipe(task_id: str, route: dict[str, Any],
                                 source_departure: dict[str, Any]) -> tuple[tuple[str, np.ndarray], ...]:
    """Select one finite world-frame [2,3] release target, without side effects.

    The shared caller has already completed its physical grasp, >=10cm lift,
    geometric source departure and100Hz physical departure barriers. This
    selector additionally rejects absent/nonboolean measured departure. It
    cannot alter quaternions, limits, collision permissions or the physical tail.
    """
    if task_id != PILL:
        raise ValueError("Direct-release transport is confined to the pill task")
    if not isinstance(route, dict) or not isinstance(source_departure, dict):
        raise ValueError("Actual route and measured source-departure mappings required")
    _true_pair(source_departure.get("effective_measured_departure"), "geometric source departure")
    release = _finite(route.get("release"), (2, 3), "original release XYZ")
    return (("carry_lower", release.copy()),)

def self_test() -> dict[str, Any]:
    """Exercise preserved selector inputs and alias isolation on the CPU."""
    def reject(task: str, route: Any, departure: Any) -> None:
        try:
            pill_direct_transport_recipe(task, route, departure)
        except ValueError:
            return
        raise AssertionError("Malformed legacy pill route was accepted")

    release = np.array([[.1, -.2, .3], [.4, .5, -.6]], dtype=np.float32)
    route = {"release": release, "above_baskets": release + [0., 0., .06]}
    departure = {"effective_measured_departure": [True, True]}
    snapshot = release.copy()
    selected = pill_direct_transport_recipe(PILL, route, departure)
    assert len(selected) == 1 and selected[0][0] == "carry_lower"
    assert selected[0][1].shape == (2, 3) and selected[0][1].dtype == np.float64
    assert np.array_equal(selected[0][1], release)
    selected[0][1][0, 0] += 1.
    assert np.array_equal(release, snapshot)
    assert departure == {"effective_measured_departure": [True, True]}
    zeros = {"release": [[0., 0., 0.], [0., 0., 0.]]}
    zeros_snapshot = deepcopy(zeros)
    selected = pill_direct_transport_recipe(PILL, zeros, {"effective_measured_departure": (True, True)})
    assert not selected[0][1].any() and zeros == zeros_snapshot

    for task in ("TwoRobotPickAppleBananaToBaskets-v1", "TwoRobotPickFruitsToPlate-v1", "unknown"):
        reject(task, route, departure)
    for invalid in (None, [], {}, {"effective_measured_departure": [True, False]},
                    {"effective_measured_departure": [1, True]},
                    {"effective_measured_departure": [np.bool_(True), True]},
                    {"effective_measured_departure": np.array([True, True])},
                    {"effective_measured_departure": [True]},
                    {"effective_measured_departure": [True, True, True]}):
        reject(PILL, route, invalid)
    for invalid in (None, [], {}, {"release": [[0., 0., 0.]]},
                    {"release": np.zeros((2, 3, 1))},
                    {"release": np.full((2, 3), np.nan)},
                    {"release": np.full((2, 3), np.inf)},
                    {"release": [["bad", 0., 0.], [0., 0., 0.]]}):
        reject(PILL, invalid, departure)
    return {"status": "passed", "task_id": PILL, "selected_phases": ["carry_lower"],
            "release_shape": [2, 3], "release_dtype": "float64", "release_target_preserved": True,
            "literal_true_pair_required": True, "input_alias_isolation": True,
            "physical_environment_steps": 0}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, allow_nan=False))
