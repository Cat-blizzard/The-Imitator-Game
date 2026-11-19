import json
import math
from pathlib import Path

import pytest

from data_generation_methods.r2h_pipeline.common import file_sha256
from data_generation_methods.r2h_pipeline.quality_cascade import (
    aggregate_decision, dimension_decisions, load_evidence, temporal_integrity,
    validate_background_roi,
)


def test_pts_repeated_rejected_but_variable_rate_allowed():
    assert temporal_integrity([0., .1, .1], 3)["status"] == "fail"
    assert temporal_integrity([0., .1, .3], 3)["status"] == "pass"


@pytest.mark.parametrize("timestamps,count", [([0., None], 2), ([0., math.nan], 2),
                                             ([0., .1], 3), ([], 0)])
def test_corrupt_time_or_count_cannot_pass(timestamps, count):
    assert temporal_integrity(timestamps, count)["status"] == "fail"


def test_clean_pixels_do_not_cancel_confirmed_extra_hand_failure():
    checks = [{"dimension": "embodiment", "result": "fail"}]
    dimensions = dimension_decisions(checks, "synthetic_human")
    result = aggregate_decision({"status": "pass"}, dimensions, [])
    assert result["status"] == "reject"
    assert result["hard_failure_dimensions"] == ["embodiment"]
    assert result["training_admitted"] is False


def test_missing_semantics_is_unknown_not_approval():
    decision = aggregate_decision({"status": "pass"},
                                  dimension_decisions([], "robot_sim"), [])
    assert decision["status"] == "needs_review"
    assert "sim_physics" in decision["unknown_dimensions"]


def test_positive_scope_still_does_not_admit_training():
    checks = [{"dimension": key, "result": "pass_in_scope"}
              for key in ("task_semantics", "embodiment", "interaction")]
    decision = aggregate_decision({"status": "pass"},
                                  dimension_decisions(checks, "synthetic_human"), [])
    assert decision["status"] == "candidate_in_reviewed_scope"
    assert not decision["quality_release_approved"]


def test_camera_cannot_treat_ego_or_unspecified_roi_as_fixed():
    with pytest.raises(ValueError):
        validate_background_roi([0, 0, .5, .5], "proposal", "egocentric")
    with pytest.raises(ValueError):
        validate_background_roi([0, 0, .5, .5], "", "fixed")
    assert validate_background_roi(None, None, "egocentric") is None


def ledger(tmp_path, **updates):
    evidence = tmp_path/"evidence.json"
    evidence.write_text('{"observed": "no"}')
    data = {
        "schema": "r2h_quality_evidence_v1", "media_sha256": "abc",
        "checks": [{"dimension": "task_semantics", "result": "fail",
                    "scope": "observed frame", "description": "teleport",
                    "sources": [{"path": str(evidence), "sha256": file_sha256(evidence)}]}],
    }
    data.update(updates)
    path = tmp_path/"ledger.json"
    path.write_text(json.dumps(data))
    return path, evidence


def test_binding_rejects_another_video_and_changed_evidence(tmp_path):
    path, evidence = ledger(tmp_path)
    with pytest.raises(ValueError, match="another video"):
        load_evidence([path], "different")
    evidence.write_text('{"observed": "yes"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        load_evidence([path], "abc")


def test_scope_and_evidence_are_preserved(tmp_path):
    path, evidence = ledger(tmp_path)
    checks, manifests = load_evidence([path], "abc")
    assert checks[0]["result"] == "fail"
    assert checks[0]["sources"][0]["sha256"] == file_sha256(evidence)
    assert manifests[0]["sha256"] == file_sha256(path)


def test_unknown_cannot_be_washed_out_by_another_positive():
    checks = [{"dimension": "interaction", "result": "unknown"},
              {"dimension": "interaction", "result": "pass_in_scope"}]
    assert dimension_decisions(checks, "real_human")["interaction"]["status"] == "unknown"
