"""CPU evidence-source classification; historical failure reasons stay intact."""
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from mani_skill.examples.motionplanning.dual.sync.solve import _failure_category


@pytest.mark.parametrize("status,reason", [
    ("aborted", "measured_collision"),
    ("rejected", "initial_collision"),
    ("rejected", "candidate_collision"),
    ("aborted", "updated_candidate_collision"),
])
def test_fcl_geometry_rejection_is_distinct_from_native_physics(status, reason):
    event = {"status": status, "reason": reason}
    assert _failure_category("lift_and_hold", event, physical_events=False) == "geometry"
    assert event["reason"] == reason
    assert _failure_category("lift_and_hold", event, physical_events=True) == "physics"


@pytest.mark.parametrize("reason", ["grasp_lost", "object_dropped", "position_slip", "rotation_slip"])
def test_actual_grasp_failure_retains_physics_category(reason):
    assert _failure_category("carry_lower", {"status": "aborted", "reason": reason},
                             physical_events=False) == "physics"


@pytest.mark.parametrize("name,status,expected", [
    ("retreat", "reach_timeout", "quality"),
    ("verify_placement", "placement_timeout", "quality"),
    ("pregrasp", "reach_timeout", "execution"),
    ("pregrasp", "rejected", "planning"),
])
def test_quality_timeout_does_not_override_official_or_execution_facts(name, status, expected):
    assert _failure_category(name, {"status": status}, physical_events=False) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
