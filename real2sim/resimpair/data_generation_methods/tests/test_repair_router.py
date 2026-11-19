from copy import deepcopy
import hashlib
import json

import pytest

from data_generation_methods.r2h_pipeline.repair_router import plan_repairs


def artifact(tmp_path, name, content=b"fixture bytes"):
    if not isinstance(content, bytes):
        content = json.dumps(content).encode()
    path = tmp_path / name
    path.write_bytes(content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def inventory(n=2, exact=True):
    return {"minimum_visible_hand_count": n, "exact_count_known": exact,
            "hands": [{"location": f"region {i}", "description": "visible palm and fingers"}
                      for i in range(n)], "limitations": []}


@pytest.fixture
def facts(tmp_path):
    review = inventory()
    anchor = artifact(tmp_path, "anchor.png")
    return {
        "schema": "r2h_repair_facts_v1",
        "reference": artifact(tmp_path, "source.mp4", b"original real-human video"),
        "anchor": anchor,
        "protocol": {"state": "ok"},
        "hidden_state": {"required": False, "reveal_condition_present": False,
                         "condition_transport_verified": False},
        "actor_inventory": {
            "maximum_visible_hands": 2, "review_scope": "global",
            "review": review, "media_sha256": anchor["sha256"],
            "evidence_artifact": artifact(tmp_path, "body.json", {"review": review})},
        "pair_validation": {"state": "not_run"},
        "attempts": {"body_reviews": 1, "anchor_repairs": 0, "contact_reviews": 0},
        "budget": {"paid_calls_remaining": 2, "max_body_reviews": 2,
                   "max_anchor_repairs": 1, "max_contact_reviews": 1,
                   "max_review_frames": 12},
    }


def set_body(facts, tmp_path, review, scope="global"):
    facts["actor_inventory"].update(
        review=review, review_scope=scope,
        evidence_artifact=artifact(tmp_path, "new_body.json", {"review": review}))


def add_contact(facts, tmp_path, interval=(2, 5), base=(0, 2, 5, 19)):
    video = artifact(tmp_path, "candidate.mp4", b"candidate video")
    facts["video"] = video
    facts["contact"] = {
        "media_sha256": video["sha256"],
        "evidence_artifact": artifact(tmp_path, "contact.json", {"unresolved": "grasp"}),
        "unknown_event_ids": ["grasp"], "frame_count": 20,
        "base_frame_indices": list(base),
        "requests": [{"query_id": "grasp", "frame_interval": list(interval),
                      "interval": "start_inclusive_end_inclusive"}],
    }


def test_known_inventory_enables_submission_but_never_training(facts):
    plan = plan_repairs(facts)
    assert plan["state"] == "ready_for_submission"
    assert plan["allowed"]["video_submission"] is True
    assert plan["allowed"]["first_frame_only_submission"] is True
    assert plan["allowed"]["training_handoff"] is False
    assert plan["training_admitted"] is False
    assert plan["actions_executed"] is False
    assert plan["paid_calls_reserved"] == 0


def test_missing_hidden_state_blocks_single_image_submission(facts):
    facts["hidden_state"]["required"] = True
    plan = plan_repairs(facts)
    assert plan["state"] == "condition_preparation"
    assert plan["actions"][0]["kind"] == "extract_reveal_or_terminal_anchor"
    assert not plan["allowed"]["video_submission"]
    assert not plan["allowed"]["first_frame_only_submission"]
    assert plan["paid_calls_reserved"] == 0


def test_reveal_image_without_verified_payload_transport_still_blocks(facts, tmp_path):
    facts["hidden_state"].update(
        required=True, reveal_condition_present=True,
        condition_artifact=artifact(tmp_path, "reveal.png"))
    plan = plan_repairs(facts)
    assert plan["actions"][0]["kind"] == "verify_multi_anchor_condition_transport"
    assert not plan["allowed"]["video_submission"]
    assert "condition_transport_not_verified" in plan["blockers"]


def test_verified_reveal_transport_never_enables_first_image_only(facts, tmp_path):
    facts["hidden_state"].update(
        required=True, reveal_condition_present=True, condition_transport_verified=True,
        condition_artifact=artifact(tmp_path, "reveal.png"),
        transport_evidence=artifact(tmp_path, "payload_probe.json", {"transport": "two_images"}))
    plan = plan_repairs(facts)
    assert plan["allowed"]["video_submission"]
    assert not plan["allowed"]["first_frame_only_submission"]


@pytest.mark.parametrize("review", [inventory(3), inventory(2, False)])
def test_global_mismatch_or_ambiguity_gets_local_review_before_edit(facts, tmp_path, review):
    set_body(facts, tmp_path, review)
    plan = plan_repairs(facts)
    assert plan["state"] == "anchor_review"
    assert [a["kind"] for a in plan["actions"]] == ["review_anchor_body_localized"]
    assert not plan["allowed"]["video_submission"]
    assert plan["paid_calls_reserved"] == 1


def test_confirmed_local_mismatch_plans_bounded_edit_and_reinspection(facts, tmp_path):
    set_body(facts, tmp_path, inventory(3), "localized")
    plan = plan_repairs(facts)
    assert plan["actions"][0]["kind"] == "repair_anchor_body"
    assert plan["actions"][0]["parameters"]["new_anchor_requires_new_body_and_task_review"]
    assert not plan["allowed"]["video_submission"]
    facts["attempts"]["anchor_repairs"] = 1
    limited = plan_repairs(facts)
    assert limited["actions"] == []
    assert "anchor_repair_attempt_limit" in limited["blockers"]


def test_local_ambiguity_does_not_repeat_paid_request_without_new_evidence(facts, tmp_path):
    set_body(facts, tmp_path, inventory(2, False), "localized")
    plan = plan_repairs(facts)
    assert plan["paid_calls_reserved"] == 0
    assert plan["actions"][0]["kind"] == "request_manual_body_review"
    assert not plan["allowed"]["video_submission"]


def test_body_review_budget_and_attempt_limit(facts, tmp_path):
    set_body(facts, tmp_path, inventory(2, False))
    facts["budget"]["paid_calls_remaining"] = 0
    plan = plan_repairs(facts)
    assert plan["actions"] == []
    assert "paid_call_budget_exhausted" in plan["blockers"]
    facts["budget"]["paid_calls_remaining"] = 2
    facts["attempts"]["body_reviews"] = 2
    plan = plan_repairs(facts)
    assert plan["actions"] == []
    assert "body_review_attempt_limit" in plan["blockers"]


def test_small_unknown_contact_interval_gets_continuous_frames(facts, tmp_path):
    add_contact(facts, tmp_path)
    plan = plan_repairs(facts)
    assert plan["state"] == "contact_review"
    action = plan["actions"][0]
    assert action["kind"] == "review_dense_contact_frames"
    assert action["parameters"]["refinement_plan"]["selected_frame_indices"] == [0, 2, 3, 4, 5, 19]
    assert plan["paid_calls_reserved"] == 1
    assert not plan["allowed"]["training_handoff"]


def test_contact_without_extra_frames_cannot_spend_again(facts, tmp_path):
    add_contact(facts, tmp_path, base=(0, 2, 3, 4, 5, 19))
    plan = plan_repairs(facts)
    assert plan["actions"] == []
    assert plan["paid_calls_reserved"] == 0
    assert "no_additional_contact_frames" in plan["blockers"]


def test_contact_frame_or_paid_budget_exhaustion_does_not_pass(facts, tmp_path):
    add_contact(facts, tmp_path)
    facts["budget"]["max_review_frames"] = 4
    plan = plan_repairs(facts)
    assert plan["actions"] == []
    assert plan["contact_refinement_plan"]["requests"][0]["selection"] == "deferred_budget"
    facts["budget"]["max_review_frames"] = 12
    facts["budget"]["paid_calls_remaining"] = 0
    plan = plan_repairs(facts)
    assert plan["actions"] == []
    assert "paid_call_budget_exhausted" in plan["blockers"]
    assert "contact_events_unknown" in plan["blockers"]


def test_contact_attempt_limit_is_separate_from_frame_budget(facts, tmp_path):
    add_contact(facts, tmp_path)
    facts["attempts"]["contact_reviews"] = 1
    plan = plan_repairs(facts)
    assert "contact_review_attempt_limit" in plan["blockers"]
    assert plan["paid_calls_reserved"] == 0


def test_unresolved_paid_submission_is_never_resubmitted(facts):
    facts["protocol"] = {"state": "submission_unknown"}
    plan = plan_repairs(facts)
    assert [a["kind"] for a in plan["actions"]] == ["reconcile_existing_submission"]
    assert plan["paid_calls_reserved"] == 0
    assert not plan["actions"][0]["parameters"]["allow_paid_resubmit"]


def test_existing_pending_id_only_gets_read_only_poll(facts):
    facts["protocol"] = {"state": "poll_pending", "task_id": "task_existing"}
    plan = plan_repairs(facts)
    assert plan["actions"][0]["kind"] == "poll_existing_task"
    assert plan["actions"][0]["parameters"]["task_id"] == "task_existing"
    assert plan["paid_calls_reserved"] == 0


def test_schema_error_only_allows_lossless_adapter(facts, tmp_path):
    facts["protocol"] = {
        "state": "schema_error", "lossless_shape_adapter_available": True,
        "raw_response": artifact(tmp_path, "response.json", {"events": "fixture"}),
    }
    plan = plan_repairs(facts)
    assert plan["actions"][0]["kind"] == "apply_lossless_shape_adapter"
    assert not plan["actions"][0]["parameters"]["allow_semantic_changes"]
    assert not plan["allowed"]["video_submission"]
    facts["protocol"]["lossless_shape_adapter_available"] = False
    plan = plan_repairs(facts)
    assert plan["actions"][0]["kind"] == "inspect_schema_failure"


def test_failed_pair_cannot_be_delivered_for_training(facts, tmp_path):
    facts["video"] = artifact(tmp_path, "candidate.mp4")
    facts["pair_validation"] = {
        "state": "failed", "evidence_artifact": artifact(tmp_path, "pair.json", {"valid": False})}
    plan = plan_repairs(facts)
    assert plan["state"] == "pair_repair"
    assert plan["actions"][0]["kind"] == "repair_explicit_pair_manifest"
    assert not plan["allowed"]["training_handoff"]
    assert not plan["training_admitted"]


def test_even_passed_pairs_require_independent_quality_release(facts, tmp_path):
    facts["video"] = artifact(tmp_path, "candidate.mp4")
    facts["pair_validation"] = {
        "state": "passed", "evidence_artifact": artifact(tmp_path, "pair.json", {"valid": True})}
    plan = plan_repairs(facts)
    assert plan["state"] == "candidate_inspection"
    assert plan["actions"][0]["parameters"]["requires_independent_quality_release"]
    assert not plan["allowed"]["training_handoff"]


def test_stale_inventory_and_changed_source_hash_are_rejected(facts):
    facts["actor_inventory"]["media_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different anchor"):
        plan_repairs(facts)
    facts["reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        plan_repairs(facts)


def test_claimed_inventory_must_equal_bound_observation(facts):
    facts["actor_inventory"]["review"] = inventory(1)
    with pytest.raises(ValueError, match="differs from bound review"):
        plan_repairs(facts)


def test_stale_contact_hash_and_unbound_event_are_rejected(facts, tmp_path):
    add_contact(facts, tmp_path)
    facts["contact"]["media_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different video"):
        plan_repairs(facts)
    facts["contact"]["media_sha256"] = facts["video"]["sha256"]
    facts["contact"]["requests"][0]["query_id"] = "not_unknown"
    with pytest.raises(ValueError, match="unknown event"):
        plan_repairs(facts)


def test_dry_run_does_not_enable_submission_and_does_not_mutate_facts(facts):
    original = deepcopy(facts)
    plan = plan_repairs(facts, verify_files=False)
    assert plan["state"] == "inspection_only"
    assert not plan["allowed"]["video_submission"]
    assert facts == original


@pytest.mark.parametrize("field,value", [("paid_calls_remaining", -1), ("max_body_reviews", True)])
def test_budget_is_explicit_nonnegative_integer(facts, field, value):
    facts["budget"][field] = value
    with pytest.raises(ValueError, match="explicit count"):
        plan_repairs(facts)
