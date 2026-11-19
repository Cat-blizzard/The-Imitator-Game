"""Interface/adversarial fixtures; none constitutes a real visual acceptance."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from data_generation_methods.r2h_pipeline.batch_pipeline import binding
from data_generation_methods.r2h_pipeline.common import file_sha256, read_json, write_json
from data_generation_methods.r2h_pipeline.providers import GeminiClient, SeedanceClient
from data_generation_methods.r2h_pipeline.single_actor_acceptance import RULE_VERSION
from data_generation_methods.r2h_pipeline.single_actor_candidate import (
    _budgeted_call, create_plan, execute_anchor, execute_video, prepare_anchor, prepare_video,
    validate_plan,
)


@pytest.fixture
def candidate(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "gray").save(source)
    anchor = tmp_path / "anchor.png"
    Image.new("RGB", (32, 32), "blue").save(anchor)
    raw = tmp_path / "source.json"
    write_json(raw, {"fixture_only": True})
    contract = {
        "schema": "r2h_task_evidence_contract_v1", "task_id": "fixture:transfer",
        "training_admitted": False, "entities": ["actor", "apple", "source", "target"],
        "entity_roles": {"actor": "operator", "apple": "fruit", "source": "left bowl",
                         "target": "right bowl"},
        "source_spec": binding(raw),
        "source_window": {"video": binding(raw), "origin": "real_human", "split": "train",
                          "window_frames": [0, 240], "fps": 30.,
                          "interval": "start_inclusive_end_exclusive"},
        "initial_relations": [{"id": "initial", "predicate": "inside",
                               "args": ["apple", "source"], "expected": True}],
        "final_relations": [{"id": "final", "predicate": "inside",
                             "args": ["apple", "target"], "expected": True}],
        "required_events": [{"id": "grasp", "verb": "grasp", "args": ["actor", "apple"]},
                            {"id": "release", "verb": "release", "args": ["apple", "target"]}],
        "precedence": [["grasp", "release"]], "forbidden_events": [],
        "allowed_additional_events": [], "allowed_variations": ["embodiment"],
    }
    contract_path = tmp_path / "contract.json"
    write_json(contract_path, contract)
    placement = {
        "actor": {"id": "operator_1",
                  "body_position": {"table_side": "far", "image_region": "upper center",
                                    "facing": "towards table and camera"},
                  "hands": {
                      side: {"owner_id": "operator_1", "anatomical_side": side,
                             "initial_entry_region": f"connected to {side} shoulder",
                             "initial_role": "rest clear of task objects"}
                      for side in ("left", "right")}},
        "event_assignments": [
            {"event_id": event["id"], "actor_id": "operator_1", "hands": ["right"],
             "role": "manipulate target fruit"} for event in contract["required_events"]],
    }
    plan = create_plan(source, contract_path, placement, {
        "human_source_id": "fixture", "physical_execution_id": "fixture", "split": "train"})
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    cfg = {"anchor_model": "image-fixture",
           "gemini": {"model": "review-fixture", "api_key_env": "FIXTURE_KEY",
                      "base_url": "https://example.test/v1beta"},
           "seedance": {"model": "video-fixture", "api_key_env": "FIXTURE_KEY",
                        "base_url": "https://example.test/api/v3"}}
    output = tmp_path / "candidate"
    prepare_anchor(plan_path, output, cfg)
    review = {
        "schema": "r2h_single_actor_review_v2", "rule_version": RULE_VERSION,
        "artifact": binding(anchor), "plan_sha256": file_sha256(plan_path), "scope": "anchor",
        "reviewer": {"id": "fixture_reviewer", "kind": "human"},
        "evidence": [{"id": "observed", "kind": "human_annotation",
                      "artifact_sha256": file_sha256(anchor), "frame_indices": [0],
                      "observation": "TEST FIXTURE ONLY, not an actual visual observation."}],
        "checks": {key: {"status": "pass", "evidence_ids": ["observed"],
                        "note": "Synthetic fixture exercises binding logic only."}
                   for key in ("person_count", "left_hand_ownership", "right_hand_ownership",
                               "body_structure", "body_position", "task_roles")},
        "limitations": ["Synthetic test fixtures do not establish image quality."],
    }
    review_path = tmp_path / "review.json"
    write_json(review_path, review)
    budget_path = tmp_path / "budget.json"
    write_json(budget_path, {"schema": "r2h_single_actor_budget_v1",
                            "cumulative_call_limits": {"image": 0, "video": 0, "review": 0},
                            "max_concurrent": 1, "timeout_seconds": 30})
    return dict(plan=plan, plan_path=plan_path, anchor=anchor, output=output, cfg=cfg,
                review=review, review_path=review_path, budget_path=budget_path)


def test_plan_is_required_before_anchor_request(candidate, tmp_path):
    bad = tmp_path / "legacy.json"
    write_json(bad, {"schema": "r2h_actor_plan_v1", "maximum_visible_hands": 2})
    with pytest.raises(ValueError, match="pre-generation"):
        prepare_anchor(bad, tmp_path / "legacy_output", candidate["cfg"])
    assert not (tmp_path / "legacy_output").exists()


@pytest.mark.parametrize("mutate", [
    lambda p: p["actor"]["hands"]["left"].update(owner_id="operator_2"),
    lambda p: p["actor"]["hands"]["right"].update(owner_id=None),
    lambda p: p["event_assignments"].pop(),
    lambda p: p["event_assignments"][0].update(actor_id="operator_2"),
    lambda p: p["actor"]["body_position"].update(image_region=""),
    lambda p: p["lineage"].update(split="validation"),
])
def test_inconsistent_or_unknown_actor_assignments_are_rejected(candidate, mutate):
    plan = deepcopy(candidate["plan"])
    mutate(plan)
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_two_people_with_two_total_hands_cannot_prepare_video(candidate):
    report = deepcopy(candidate["review"])
    report["checks"]["person_count"].update(
        status="fail", note="Two distinct operators, despite only two visible hands.")
    write_json(candidate["review_path"], report, overwrite=True)
    result = prepare_video(candidate["output"], candidate["anchor"],
                           candidate["review_path"], candidate["cfg"])
    assert result["prepared"] is False
    assert result["decision"]["status"] == "reject"
    assert not (candidate["output"] / "video_request.json").exists()


def test_unknown_arm_ownership_blocks_video(candidate):
    report = deepcopy(candidate["review"])
    report["checks"]["left_hand_ownership"].update(status="unknown", evidence_ids=[])
    write_json(candidate["review_path"], report, overwrite=True)
    result = prepare_video(candidate["output"], candidate["anchor"],
                           candidate["review_path"], candidate["cfg"])
    assert result["prepared"] is False
    assert result["decision"]["status"] == "needs_review"


def test_vlm_only_review_cannot_supply_independent_acceptance(candidate):
    report = deepcopy(candidate["review"])
    report["reviewer"]["kind"] = "vlm"
    report["evidence"][0]["kind"] = "vlm"
    write_json(candidate["review_path"], report, overwrite=True)
    result = prepare_video(candidate["output"], candidate["anchor"],
                           candidate["review_path"], candidate["cfg"])
    assert result["prepared"] is False


def test_legacy_hand_count_report_is_not_upgraded(candidate):
    write_json(candidate["review_path"], {
        "minimum_visible_hand_count": 2, "exact_count_known": True,
        "hands": [{"location": "left"}, {"location": "right"}],
    }, overwrite=True)
    with pytest.raises(ValueError):
        prepare_video(candidate["output"], candidate["anchor"],
                      candidate["review_path"], candidate["cfg"])
    assert not (candidate["output"] / "video_request.json").exists()


@pytest.mark.parametrize("field", ["artifact", "plan", "unrelated_evidence"])
def test_changed_or_unrelated_hashes_cannot_pass(candidate, field):
    report = deepcopy(candidate["review"])
    if field == "artifact":
        report["artifact"]["sha256"] = "0" * 64
    elif field == "plan":
        report["plan_sha256"] = "0" * 64
    else:
        report["evidence"][0]["artifact_sha256"] = "0" * 64
    write_json(candidate["review_path"], report, overwrite=True)
    try:
        result = prepare_video(candidate["output"], candidate["anchor"],
                               candidate["review_path"], candidate["cfg"])
    except ValueError:
        pass
    else:
        assert result["prepared"] is False
    assert not (candidate["output"] / "video_request.json").exists()


def test_preparation_and_submit_recheck_exact_anchor_review(candidate):
    result = prepare_video(candidate["output"], candidate["anchor"],
                           candidate["review_path"], candidate["cfg"])
    assert result["prepared"] is True
    # A valid prepared payload is still blocked if its independently reviewed image changes.
    Image.new("RGB", (32, 32), "red").save(candidate["anchor"])
    with patch.object(SeedanceClient, "submit", side_effect=AssertionError("No API call")):
        with pytest.raises(ValueError, match="hash mismatch"):
            execute_video(candidate["output"], candidate["cfg"], candidate["budget_path"])


def test_zero_budget_blocks_real_provider_entry_before_credentials(candidate):
    with patch.object(GeminiClient, "generate_image", side_effect=AssertionError("No API call")):
        with pytest.raises(RuntimeError, match="budget exhausted"):
            execute_anchor(candidate["output"], candidate["cfg"], candidate["budget_path"])
    assert not list(candidate["budget_path"].parent.glob("*_ledger/stages/*/call_started.json"))


def test_shared_campaign_budget_counts_failures_and_cannot_reset_per_candidate(candidate):
    budget = read_json(candidate["budget_path"])
    budget["cumulative_call_limits"]["image"] = 1
    write_json(candidate["budget_path"], budget, overwrite=True)
    def fail(_):
        raise RuntimeError("Synthetic unknown submission")
    with pytest.raises(RuntimeError, match="Synthetic unknown"):
        _budgeted_call(candidate["budget_path"], "image", {"candidate": "one"}, fail)
    with pytest.raises(RuntimeError, match="automatic retry prohibited"):
        _budgeted_call(candidate["budget_path"], "image", {"candidate": "one"}, fail)
    with pytest.raises(RuntimeError, match="unresolved submission"):
        _budgeted_call(candidate["budget_path"], "image", {"candidate": "different_directory"}, fail)


def test_async_video_keeps_campaign_slot_occupied_after_post(candidate):
    budget = read_json(candidate["budget_path"])
    budget["cumulative_call_limits"].update(image=2, video=2)
    write_json(candidate["budget_path"], budget, overwrite=True)
    _budgeted_call(candidate["budget_path"], "video", {"candidate": "one"},
                   lambda _: {"task_id": "test_task_only"})
    with pytest.raises(RuntimeError, match="in-flight video"):
        _budgeted_call(candidate["budget_path"], "video", {"candidate": "two"},
                       lambda _: pytest.fail("No second POST while the first task runs"))
    with pytest.raises(RuntimeError, match="in-flight video"):
        _budgeted_call(candidate["budget_path"], "image", {"candidate": "two"},
                       lambda _: pytest.fail("Concurrency is shared across candidate kinds"))


def test_successful_call_consumes_budget_across_directories(candidate):
    budget = read_json(candidate["budget_path"])
    budget["cumulative_call_limits"]["image"] = 1
    write_json(candidate["budget_path"], budget, overwrite=True)
    _budgeted_call(candidate["budget_path"], "image", {"candidate": "one"},
                   lambda _: {"result": "fixture"})
    with pytest.raises(RuntimeError, match="budget exhausted"):
        _budgeted_call(candidate["budget_path"], "image", {"candidate": "two"},
                       lambda _: pytest.fail("A new candidate directory does not reset the budget"))
