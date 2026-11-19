import json
import pytest
from data_generation_methods.r2h_pipeline.automated_review import (
    choose_rois, condition_handoff, check_actor_budget, boundary_handoff)
from data_generation_methods.r2h_pipeline.scene_proposals import binding


def test_unknown_or_multiple_targets_do_not_pick_first():
    p = {"target": {"roi_candidates": [
        {"id": "x", "role": "interaction_target", "status": "unknown"},
        {"id": "b", "role": "background_reference", "status": "candidate",
         "box_xyxy_normalized": [0, 0, .1, .1]}]}}
    rois, pending = choose_rois(p)
    assert rois is None
    assert "interaction_target_requires_unambiguous_localization" in pending
    p["target"]["roi_candidates"] = [
        {"id": str(i), "role": "interaction_target", "status": "candidate",
         "box_xyxy_normalized": [.2, .2, .4, .4]} for i in range(2)]
    assert choose_rois(p)[0] is None


def test_reveal_frames_not_silently_assigned_to_human_timebase():
    p = {
        "source_keyframes": [{"frame_index_0based": 330, "timestamp_seconds": 16.5}],
        "hidden_state_proposals": [{"hidden_then_revealed": True,
                                   "object_index": 0, "maximum_visibility_frame": 330}],
        "actor_plan_proposal": {"number_of_moving_arms": 1, "maximum_visible_hands": None},
    }
    handoff = condition_handoff(p)
    assert handoff["revealed_state_conditions"][0]["generated_human_frame_index"] is None
    assert handoff["actor_plan"]["maximum_visible_hands"] is None
    p["source_keyframes"] = []
    assert condition_handoff(p)["revealed_state_conditions"][0]["status"] == "needs_source_frame_extraction"


def test_budget_is_bound_to_media_and_declared_count(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Predeclared one visible hand.")
    plan_path = tmp_path / "actor.json"
    plan_path.write_text(json.dumps({
        "schema": "r2h_actor_plan_v1", "media_sha256": "a" * 64,
        "maximum_visible_hands": 1, "maximum_visible_people": None,
        "declaration_scope": "single active hand",
        "sources": [binding(source)],
    }))
    assert check_actor_budget(binding(plan_path), "a" * 64, 1) == plan_path
    with pytest.raises(ValueError, match="media binding"):
        check_actor_budget(binding(plan_path), "b" * 64, 1)
    with pytest.raises(ValueError, match="differs"):
        check_actor_budget(binding(plan_path), "a" * 64, 2)


def test_boundary_evidence_routes_native_frames_without_passing_or_rejecting():
    report = {"source": {"sha256": "a"*64, "frames": 53},
              "candidates": [{"previous_frame": 51, "frame_index": 52,
                              "peak_robust_z": 12., "regions": [{"normalized_xyxy": [0,0,.2,.2]}]}]}
    requests = boundary_handoff(report, "a"*64)
    assert requests[0]["frame_range_inclusive"] == [49, 52]
    assert requests[0]["automatic_task_verdict"] is None
    with pytest.raises(ValueError, match="different media"):
        boundary_handoff(report, "b"*64)
    report["candidates"][0]["frame_index"] = 53
    with pytest.raises(ValueError, match="interval"):
        boundary_handoff(report, "a"*64)


def test_no_boundary_alert_does_not_make_quality_pass():
    assert boundary_handoff({"source": {"sha256": "a"*64, "frames": 53},
                             "candidates": []}, "a"*64) == []
