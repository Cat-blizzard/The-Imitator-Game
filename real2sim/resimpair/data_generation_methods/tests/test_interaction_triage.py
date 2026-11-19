import csv
import hashlib
import json

import pytest

from data_generation_methods.r2h_pipeline.interaction_triage import triage_interaction
from data_generation_methods.r2h_pipeline.refinement import plan_refinement


def bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path, count=20):
    video, model = tmp_path / "media.mp4", tmp_path / "hand.task"
    video.write_bytes(b"actual bound media bytes")
    model.write_bytes(b"actual bound model bytes")
    rows = [{
        "frame_index": i, "timestamp_seconds": i / 24,
        "status": "tracked_2d_proxy", "detected_hand_count": 1,
        "nearest_hand_keypoint_to_handle_track_px": 100.0,
        "handle_step_px": None if i == 0 else 0.1,
    } for i in range(count)]
    data = {
        "schema": "fixed_camera_interaction_diagnostic_v1",
        "video": {**bind(video), "frames": count}, "model": {**bind(model), "num_hands": 6},
        "handle_roi_normalized_xyxy": [0.4, 0.4, 0.6, 0.6],
        "background_roi_normalized_xyxy": [0.1, 0.1, 0.2, 0.2],
        "roi_provenance": "manual fixture ROI; not projected 3D geometry",
        "parameters": {"hand_near_threshold_px": 20.0, "motion_threshold_px_per_frame": 1.0},
        "rows": rows,
    }
    return data


def save(tmp_path, data):
    path = tmp_path / "diagnostic.json"
    path.write_text(json.dumps(data))
    return path


def test_169_extra_hand_frames_are_grouped_not_169_independent_errors(tmp_path):
    data = fixture(tmp_path, 193)
    for row in data["rows"][:169]:
        row["detected_hand_count"] = 3
    result = triage_interaction(save(tmp_path, data), maximum_visible_hands=2)
    assert result["summary"]["extra_hand_signal_frames"] == 169
    assert result["summary"]["extra_hand_candidate_groups"] == 1
    assert result["candidate_groups"][0]["signal_frame_interval"] == [0, 168]
    assert result["refinement_requests"][0]["frame_interval"] == [0, 11]
    assert result["candidate_groups"][0]["window_selection"] == "onset_only"
    assert not result["human_actor_mismatch_confirmed"]


def test_missing_detection_does_not_mean_no_hand_or_contact(tmp_path):
    data = fixture(tmp_path)
    for row in data["rows"][1:5]:
        row.update(detected_hand_count=0, nearest_hand_keypoint_to_handle_track_px=None,
                   handle_step_px=4.0)
    result = triage_interaction(save(tmp_path, data), maximum_visible_hands=1)
    assert result["summary"]["selected_intervals"] == 0
    assert result["unknown_coverage"]["hand_nearness_unknown_frames"] == 4
    assert result["unknown_coverage"]["physical_contact_unknown_frames"] == 20
    assert not result["quality_release_approved"]


def test_lost_track_stays_unknown_instead_of_stationary_or_contact_failure(tmp_path):
    data = fixture(tmp_path)
    for row in data["rows"][8:]:
        row.update(status="unknown", nearest_hand_keypoint_to_handle_track_px=None,
                   handle_step_px=None)
    result = triage_interaction(save(tmp_path, data), maximum_visible_hands=1)
    assert result["unknown_coverage"]["tracking_unknown_frames"] == 12
    assert result["refinement_requests"][0]["reason"] == "lost_track_unknown"
    assert not result["physical_contact_ground_truth"]


def test_motion_only_before_first_observed_nearness_is_a_candidate(tmp_path):
    data = fixture(tmp_path)
    for i in (2, 3, 10, 11):
        data["rows"][i]["handle_step_px"] = 4.0
    data["rows"][5]["nearest_hand_keypoint_to_handle_track_px"] = 5.0
    result = triage_interaction(save(tmp_path, data), maximum_visible_hands=1)
    groups = [x for x in result["candidate_groups"]
              if x["kind"] == "motion_before_hand_near_candidate"]
    assert len(groups) == 1
    assert groups[0]["signal_frame_interval"] == [2, 3]
    assert result["rules"]["first_observed_hand_near_frame"] == 5
    assert groups[0]["physical_contact"] == "unknown"


def test_two_request_limit_and_refinement_compatibility(tmp_path):
    data = fixture(tmp_path, 30)
    for i in (2, 3):
        data["rows"][i]["handle_step_px"] = 4.0
    for i in range(8, 15):
        data["rows"][i]["detected_hand_count"] = 2
    for row in data["rows"][25:]:
        row.update(status="unknown", nearest_hand_keypoint_to_handle_track_px=None,
                   handle_step_px=None)
    result = triage_interaction(save(tmp_path, data), maximum_visible_hands=1)
    assert len(result["refinement_requests"]) == 2
    assert result["summary"]["unselected_candidate_groups"] == 1
    refinement = plan_refinement(30, [0, 29], result["refinement_requests"], max_frames=30)
    assert all(x["selection"] == "continuous_interval_selected" for x in refinement["requests"])
    assert result["selection_is_not_verification"]


def test_no_candidates_is_not_success(tmp_path):
    result = triage_interaction(save(tmp_path, fixture(tmp_path)), maximum_visible_hands=1)
    assert result["refinement_requests"] == []
    assert result["unknown_coverage"]["zero_candidates_is_not_quality_pass"]
    assert not result["training_admitted"]
    assert result["rules"]["provisional"]
    assert not result["rules"]["calibrated_acceptance_thresholds"]


def test_binding_model_roi_and_actual_timestamps(tmp_path):
    data = fixture(tmp_path)
    for i in (3, 4):
        data["rows"][i]["detected_hand_count"] = 2
    data["rows"][1]["timestamp_seconds"] = 0.031
    path = save(tmp_path, data)
    result = triage_interaction(path, maximum_visible_hands=1)
    assert result["evidence_bindings"]["diagnostic"] == bind(path)
    assert result["evidence_bindings"]["model"] == {
        k: data["model"][k] for k in ("path", "sha256")}
    assert result["roi"]["provenance"] == data["roi_provenance"]
    assert result["refinement_requests"][0]["timestamp_interval_seconds"][0] == 0.031


@pytest.mark.parametrize("target", ["video", "model"])
def test_changed_source_or_model_is_rejected(tmp_path, target):
    data = fixture(tmp_path)
    path = save(tmp_path, data)
    with open(data[target]["path"], "wb") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        triage_interaction(path, maximum_visible_hands=1)


def test_missing_frames_are_not_silently_interpolated(tmp_path):
    data = fixture(tmp_path)
    data["rows"][5]["frame_index"] = 6
    with pytest.raises(ValueError, match="complete ordered timeline"):
        triage_interaction(save(tmp_path, data), maximum_visible_hands=1)


def test_csv_is_checked_against_raw_json_not_counted_as_independent_evidence(tmp_path):
    data = fixture(tmp_path)
    path = save(tmp_path, data)
    csv_path = tmp_path / "timeline.csv"
    with csv_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["rows"][0]))
        writer.writeheader()
        writer.writerows(data["rows"])
    result = triage_interaction(path, maximum_visible_hands=1, timeline_path=csv_path)
    assert result["evidence_bindings"]["timeline"] == bind(csv_path)
    csv_path.write_text(csv_path.read_text().replace("100.0", "999.0", 1))
    with pytest.raises(ValueError, match="Timeline disagrees"):
        triage_interaction(path, maximum_visible_hands=1, timeline_path=csv_path)


def test_one_frame_budget_still_contains_signal_and_zero_requests_is_allowed(tmp_path):
    data = fixture(tmp_path)
    for i in (5, 6):
        data["rows"][i]["detected_hand_count"] = 2
    path = save(tmp_path, data)
    result = triage_interaction(path, maximum_visible_hands=1, max_interval_frames=1)
    assert result["refinement_requests"][0]["frame_interval"] == [5, 5]
    result = triage_interaction(path, maximum_visible_hands=1, max_requests=0)
    assert result["refinement_requests"] == []


@pytest.mark.parametrize("kwargs", [
    {"max_requests": 3}, {"max_requests": True}, {"max_interval_frames": 0},
    {"minimum_candidate_run_frames": -1},
])
def test_invalid_rules_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        triage_interaction(save(tmp_path, fixture(tmp_path)), maximum_visible_hands=1, **kwargs)
