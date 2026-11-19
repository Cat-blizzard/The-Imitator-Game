"""Temporary fixtures test evidence logic, never approve production video."""
from copy import deepcopy
import json

import pytest

from data_generation_methods.r2h_pipeline.task_evidence import (
    canonical_sha256, evaluate_task_evidence, file_sha256, validate_contract, validate_policy,
)


@pytest.fixture
def case(tmp_path):
    def artifact(name):
        path = tmp_path / name
        path.write_text(json.dumps({"test_fixture_only": name}))
        return {"path": str(path), "sha256": file_sha256(path)}
    contract = {
        "schema": "r2h_task_evidence_contract_v1", "task_id": "test:apple_transfer",
        "training_admitted": False, "entities": ["actor", "apple", "source", "target"],
        "source_spec": artifact("source_spec.json"),
        "source_window": {"video": artifact("source_media_fixture.json"), "origin": "real_human", "split": "train",
                          "window_frames": [0, 240], "fps": 30., "interval": "start_inclusive_end_exclusive"},
        "initial_relations": [{"id": "i_inside", "predicate": "inside", "args": ["apple", "source"], "expected": True}],
        "final_relations": [{"id": "f_inside", "predicate": "inside", "args": ["apple", "target"], "expected": True},
                            {"id": "f_released", "predicate": "released", "args": ["apple"], "expected": True}],
        "required_events": [{"id": "grasp", "verb": "grasp", "args": ["actor", "apple"]},
                            {"id": "transport", "verb": "transport", "args": ["apple", "target"]},
                            {"id": "release", "verb": "release", "args": ["apple", "target"]}],
        "precedence": [["grasp", "transport"], ["transport", "release"]],
        "forbidden_events": [{"id": "move_bowl", "verb": "move", "args": ["source"]}],
        "allowed_additional_events": [], "allowed_variations": ["appearance", "layout", "embodiment"],
    }
    media = artifact("generated_media_fixture.json")
    report = artifact("observations_fixture.json")
    common = {"source_type": "vlm_observed", "artifact": report, "media_sha256": media["sha256"],
              "method": "fixture_vlm", "description": "Synthetic unit-test observation, not actual pixels.",
              "state": "observed", "confidence": .99}
    relations = []
    for phase in ("initial", "final"):
        for item in contract[phase + "_relations"]:
            relations.append({**deepcopy(common), "phase": phase, "requirement_id": item["id"],
                              "predicate": item["predicate"], "args": item["args"], "value": True,
                              "t_start": 0. if phase == "initial" else 7.,
                              "t_end": .5 if phase == "initial" else 8.})
    events = [{**deepcopy(common), "verb": item["verb"], "args": item["args"],
               "t_start": 1. + 2 * i, "t_end": 2. + 2 * i}
              for i, item in enumerate(contract["required_events"])]
    events.append({**deepcopy(common), "verb": "move", "args": ["source"], "state": "not_observed",
                   "t_start": 0., "t_end": 8.})
    evidence = {"schema": "r2h_task_observations_v1", "contract_sha256": canonical_sha256(contract),
                "subject": {"media": media, "origin": "synthetic_human", "duration_seconds": 8.},
                "relation_observations": relations, "event_observations": events,
                "observed_goal": "correct_answer_is_not_evidence"}
    policy = {"time_tolerance_seconds": 0., "initial_window_seconds": 1., "final_window_seconds": 1.,
              "motion_requirements": []}
    return contract, evidence, policy


def evaluate(case, **kwargs):
    return evaluate_task_evidence(case[0], case[1], policy=case[2], **kwargs)


def codes(result):
    return {x["code"] for x in result["issues"]}


def test_correct_vlm_records_only_become_task_candidate(case):
    before = deepcopy(case)
    result = evaluate(case)
    assert result["decision"] == "task_equivalent_candidate"
    assert result["task_equivalent_candidate"] is True
    assert result["motion_verified"] is False
    assert result["training_admitted"] is False
    assert result["confidence_used_as_probability"] is False
    assert case == before


@pytest.mark.parametrize("phase", ["initial", "final"])
def test_correct_goal_cannot_replace_missing_boundary_evidence(case, phase):
    case[1]["relation_observations"] = [x for x in case[1]["relation_observations"] if x["phase"] != phase]
    result = evaluate(case)
    assert result["decision"] == "needs_review"
    assert "RELATION_EVIDENCE_MISSING" in codes(result)


def test_wrong_terminal_relation_rejected(case):
    case[1]["relation_observations"][1]["value"] = False
    result = evaluate(case)
    assert result["decision"] == "rejected"
    assert "RELATION_CONTRADICTION" in codes(result)


@pytest.mark.parametrize("state", ["unknown", "not_observed"])
def test_unknown_or_not_observed_is_not_true(case, state):
    case[1]["relation_observations"][-1].update(state=state, value=None)
    result = evaluate(case)
    assert result["decision"] == "needs_review"
    assert "RELATION_NOT_ESTABLISHED" in codes(result)


def test_missing_release_blocks(case):
    case[1]["event_observations"].pop(2)
    result = evaluate(case)
    assert result["decision"] == "needs_review"
    assert "REQUIRED_EVENT_NOT_ESTABLISHED" in codes(result)


def test_wrong_order_rejected(case):
    case[1]["event_observations"][2].update(t_start=.1, t_end=.5)
    result = evaluate(case)
    assert result["decision"] == "rejected"
    assert "EVENT_PRECEDENCE_VIOLATION" in codes(result)


def test_forbidden_event_and_missing_coverage(case):
    case[1]["event_observations"][-1].update(state="observed", t_start=6., t_end=7.)
    assert "FORBIDDEN_EVENT_OBSERVED" in codes(evaluate(case))
    case[1]["event_observations"].pop()
    assert "FORBIDDEN_EVENT_COVERAGE_MISSING" in codes(evaluate(case))


def test_late_evidence_cannot_prove_initial_state(case):
    case[1]["relation_observations"][0].update(t_start=7., t_end=7.5)
    assert "INITIAL_EVIDENCE_OUTSIDE_BOUNDARY" in codes(evaluate(case))


def test_early_evidence_cannot_prove_terminal_state(case):
    case[1]["relation_observations"][1].update(t_start=.1, t_end=.5)
    assert "FINAL_EVIDENCE_OUTSIDE_BOUNDARY" in codes(evaluate(case))


def test_confidence_does_not_define_correctness(case):
    for observation in case[1]["relation_observations"] + case[1]["event_observations"]:
        observation["confidence"] = .01
    assert evaluate(case)["decision"] == "task_equivalent_candidate"
    # This means the gate has no calibrated confidence claim, not that .01 is reliable.
    case[1]["relation_observations"][1]["value"] = False
    case[1]["relation_observations"][1]["confidence"] = 1.
    assert evaluate(case)["decision"] == "rejected"


def add_motion(case, source="tracker_measured"):
    from pathlib import Path
    path = Path(case[1]["subject"]["media"]["path"]).parent / "raw_tracks.json"
    raw = {
        "schema": "r2h_paired_2d_tracks_v1", "coordinate_system": "shared_pixel_coordinates",
        "alignment_method": "fixture_explicit_time_pairs_only",
        "candidate_media": case[1]["subject"]["media"], "reference_media": case[0]["source_window"]["video"],
        "tracker": {"name": "fixture_tracker", "version": "unit_test"},
        "candidate_timestamps_seconds": [0., 4., 8.], "reference_timestamps_seconds": [0., 4., 8.],
        "candidate_xy": [[[1.2, 0.]], [[1.2, 0.]], [[1.2, 0.]]],
        "reference_xy": [[[0., 0.]], [[0., 0.]], [[0., 0.]]], "valid_pairs": [[True], [True], [True]],
    }
    path.write_text(json.dumps(raw))
    case[2]["motion_requirements"] = [{
        "id": "tracks", "metric": "visible_track_rmse", "unit": "pixels", "scope": "visible_aligned_2d_tracks_only",
        "comparison": "<=", "threshold": 2., "allowed_sources": ["tracker_measured"],
        "allowed_methods": ["paired_2d_tracks_v1"],
    }]
    case[1]["motion_measurements"] = [{
        "requirement_id": "tracks", "metric": "visible_track_rmse", "unit": "pixels",
        "scope": "visible_aligned_2d_tracks_only", "state": "observed", "source_type": source,
        "method": "paired_2d_tracks_v1", "artifact": {"path": str(path), "sha256": file_sha256(path)},
        "media_sha256": case[1]["subject"]["media"]["sha256"],
    }]


@pytest.mark.parametrize("source", ["vlm_observed", "manual_verified", "sim_measured"])
def test_nontracker_or_sim_truth_cannot_certify_generated_human_motion(case, source):
    add_motion(case, source)
    result = evaluate(case)
    assert result["decision"] == "task_equivalent_candidate"
    assert result["motion_verified"] is False
    assert result["motion_issues"]


def test_scoped_numeric_measurement_gate(case):
    from pathlib import Path
    add_motion(case)
    result = evaluate(case)
    assert result["decision"] == "motion_verified"
    assert result["motion_verification_scope"] == ["visible_aligned_2d_tracks_only"]
    assert result["training_admitted"] is False
    assert result["human_3d_ground_truth"] is False
    assert result["motion_measured_values"][0]["value"] == pytest.approx(1.2)
    assert result["motion_measured_values"][0]["recomputed_from_raw_tracks"] is True
    artifact = case[1]["motion_measurements"][0]["artifact"]
    path = Path(artifact["path"])
    raw = json.loads(path.read_text())
    raw["candidate_xy"] = [[[3., 0.]], [[3., 0.]], [[3., 0.]]]
    path.write_text(json.dumps(raw))
    artifact["sha256"] = file_sha256(path)
    assert evaluate(case)["motion_verified"] is False


@pytest.mark.parametrize("field,value", [("unit", "meters"), ("value", float("nan")),
                                       ("media_sha256", "a" * 64), ("state", "unknown")])
def test_invalid_numeric_measurement_fails_closed(case, field, value):
    add_motion(case)
    case[1]["motion_measurements"][0][field] = value
    assert evaluate(case)["motion_verified"] is False


def test_artifact_hash_and_media_binding(case):
    case[1]["relation_observations"][0]["artifact"]["sha256"] = "a" * 64
    assert evaluate(case)["decision"] == "rejected"


def test_inspection_only_never_approves(case):
    assert evaluate(case, verify_files=False)["decision"] == "needs_review"


def test_explicit_thresholds_required(case):
    del case[2]["time_tolerance_seconds"]
    with pytest.raises(ValueError, match="explicit threshold"):
        evaluate(case)


def test_vlm_not_a_measured_motion_source_even_if_config_requests_it(case):
    add_motion(case)
    case[2]["motion_requirements"][0]["allowed_sources"] = ["vlm_observed"]
    with pytest.raises(ValueError, match="sim/tracker"):
        validate_policy(case[2])


def test_cycle_and_repeated_event_cannot_reuse_one_observation(case):
    case[0]["precedence"].append(["release", "grasp"])
    with pytest.raises(ValueError, match="Cyclic"):
        validate_contract(case[0])
    case[0]["precedence"].pop()
    case[0]["required_events"].append({"id": "release_again", "verb": "release", "args": ["apple", "target"]})
    case[1]["contract_sha256"] = canonical_sha256(case[0])
    assert evaluate(case)["decision"] == "rejected"


def test_duplicate_reviewer_reports_are_not_two_physical_occurrences(case):
    case[0]["required_events"].append({"id": "release_again", "verb": "release", "args": ["apple", "target"]})
    case[1]["contract_sha256"] = canonical_sha256(case[0])
    copy = deepcopy(case[1]["event_observations"][2])
    copy["source_type"] = "manual_verified"
    case[1]["event_observations"].append(copy)
    assert evaluate(case)["decision"] == "rejected"


def test_unknown_motion_and_false_self_report_do_not_override_missing_raw_tracks(case):
    add_motion(case)
    case[1]["motion_measurements"][0]["value"] = 0.
    assert evaluate(case)["motion_verified"] is False


def test_mask_coverage_computed_from_raw_pairs(case):
    from pathlib import Path
    add_motion(case)
    requirement = case[2]["motion_requirements"][0]
    measurement = case[1]["motion_measurements"][0]
    requirement.update(metric="visible_pair_coverage", unit="fraction", comparison=">=", threshold=.9)
    measurement.update(metric="visible_pair_coverage", unit="fraction")
    artifact = measurement["artifact"]
    path = Path(artifact["path"])
    raw = json.loads(path.read_text())
    raw["valid_pairs"] = [[True], [False], [False]]
    path.write_text(json.dumps(raw))
    artifact["sha256"] = file_sha256(path)
    result = evaluate(case)
    assert result["motion_verified"] is False
    assert result["motion_measured_values"][0]["value"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("field", ["relation_observations", "event_observations", "motion_measurements"])
def test_malformed_provider_list_entries_reject_without_crashing(case, field):
    case[1][field] = [None]
    assert evaluate(case)["decision"] == "rejected"
