"""Portable orchestration fixtures, separate from actual g00/g01 evidence audits.

Only verify_sources is substituted here; no synthetic pixel is a quality result.
Provider methods return marked test fixtures and never contact a service.
"""
import base64
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
import pytest

from data_generation_methods.r2h_pipeline import unified_anchor_candidate as module
from data_generation_methods.r2h_pipeline.batch_pipeline import binding
from data_generation_methods.r2h_pipeline.common import file_sha256, json_sha256, read_json, write_json
from data_generation_methods.r2h_pipeline.single_actor_acceptance import CHECKS as ACTOR_CHECKS, RULE_VERSION as ACTOR_VERSION
from data_generation_methods.r2h_pipeline.static_state_acceptance import CHECKS as STATE_CHECKS, RULE_VERSION as STATE_VERSION


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CAMPAIGN_REGISTRY", tmp_path / "test_campaign_registry")
    image = tmp_path / "initial.png"
    Image.new("RGB", (48, 48), "blue").save(image)
    source_file = tmp_path / "source_fixture.json"
    write_json(source_file, {"fixture_only": True})
    source = {
        "id": "source_fixture", "session": "session_fixture", "split": "train",
        "video": binding(source_file), "review": binding(source_file),
        "actual_review_sheet": binding(source_file), "window_frames": [0, 10], "fps": 30.,
        "interval": "start_inclusive_end_exclusive", "initial": "apple inside occupied bowl",
        "steps": ["grasp", "lift", "transport", "release", "withdraw"],
        "final": "apple inside target bowl",
    }
    spec = tmp_path / "human_spec.json"
    write_json(spec, {"tasks": [{"id": "apple_bowl_transfer", "canonical_goal": "fixture_goal",
                                 "human_windows": [source], "allowed_differences": ["embodiment"]}]})
    execution = "execution_fixture"
    ids = {"selected_object": execution + "/object/0", "source_container": execution + "/container/0",
           "destination_container": execution + "/container/1"}
    roles = {k: {"id": v} for k, v in ids.items()}
    initial = {"labels": ["initial"], "frame": {**binding(image), "frame_index_0based": 0},
               "relative_size_reference": {"fixture_geometry_only": True}}
    packet = {"roles": roles, "selected_camera": "opposite",
              "camera_calibration": {"fixture_only": True},
              "camera_role_projections": {"opposite": {"relative_order": {
                  "source_relative_to_destination": "screen_left", "destination_relative_to_source": "screen_right"}}},
              "state_delta": {"edit_instruction": "TERMINAL_ACTION_MUST_NOT_LEAK"},
              "conditions": [initial, {"labels": ["terminal"], "fixture": "LATE_STATE"}]}
    condition_path, input_path = tmp_path / "conditions.json", tmp_path / "input.json"
    execution_input = {"contract": {"human_spec": binding(spec)}}
    write_json(condition_path, packet)
    write_json(input_path, execution_input)
    context = {"condition_role": "initial", "condition_packet_sha256": file_sha256(condition_path),
               "reference_frame_sha256": file_sha256(image), "execution_id": execution,
               "camera_id": "opposite", "human_source_id": source["id"], "human_session_id": source["session"],
               "human_video_sha256": source["video"]["sha256"], "human_window_frames": [0, 10],
               "human_fps": 30., "human_split": "train",
               "human_window_interval": "start_inclusive_end_exclusive", "role_ids": ids}
    data = {"packet": packet, "receipt": {"execution_id": execution}, "input": execution_input,
            "human_source": source, "initial": initial, "context": context}
    verify = Mock(side_effect=lambda *_: deepcopy(data))
    monkeypatch.setattr(module, "verify_sources", verify)
    placement = {"actor": {"id": "operator_1", "body_position": {
        "table_side": "far", "image_region": "upper center", "facing": "towards table"},
        "hands": {side: {"owner_id": "operator_1", "anatomical_side": side,
                         "initial_entry_region": side + " shoulder", "initial_role": "rest_clear_of_task_objects"}
                  for side in ("left", "right")}},
        "event_assignments": [{"event_id": eid, "actor_id": "operator_1", "hands": ["right"], "role": eid}
                              for eid in ("grasp_apple", "lift_from_source", "transport_to_target",
                                          "release_in_target", "withdraw_hand")]}
    placement_path = tmp_path / "placement.json"
    write_json(placement_path, placement)
    cfg = {"anchor_model": "fixture_image", "gemini": {"model": "fixture_review", "api_key_env": "FIXTURE_KEY",
           "base_url": "https://example.test/v1beta"}, "seedance": {"model": "fixture_video", "api_key_env": "FIXTURE_KEY",
           "base_url": "https://example.test/api/v3"}}
    out = tmp_path / "candidate"
    module.prepare_unified(condition_path, input_path, placement_path, out, cfg)
    budget = tmp_path / "budget.json"
    write_json(budget, {"schema": "r2h_unified_budget_v1", "campaign_id": "test_fixture_campaign",
                       "ledger_path": str(tmp_path / "fixed_ledger"), "cumulative_call_limits": {
        "image": 2, "video": 1, "review": 0}, "max_concurrent": 1, "timeout_seconds": 30})
    provider = Mock(return_value={"images": [{"mime_type": "image/png",
                                            "data": base64.b64encode(image.read_bytes()).decode()}],
                                 "fixture_only": True})
    monkeypatch.setattr(module.GeminiClient, "generate_image", provider)
    return {"output": out, "package": out / "package.json", "config": cfg, "budget": budget,
            "context": context, "provider": provider, "verify": verify,
            "image": out / "human_initial.png", "placement": placement_path,
            "condition_path": condition_path, "input_path": input_path}


def generate(case):
    return module.execute_initial(case["package"], case["config"], case["budget"])


def reviews(case, *, actor_status="pass", state_status="pass"):
    actor = {"schema": "r2h_single_actor_review_v2", "rule_version": ACTOR_VERSION,
             "artifact": binding(case["image"]), "scope": "anchor",
             "plan_sha256": file_sha256(case["output"] / "actor_plan.json"),
             "reviewer": {"id": "unit_test_not_visual_evidence", "kind": "human"},
             "evidence": [{"id": "fixture", "kind": "human_annotation",
                           "artifact_sha256": file_sha256(case["image"]), "frame_indices": [0],
                           "observation": "Synthetic test annotation, not a real image judgment."}],
             "checks": {k: {"status": actor_status, "evidence_ids": ["fixture"],
                            "note": "Synthetic test annotation."} for k in ACTOR_CHECKS},
             "limitations": ["Synthetic test annotation does not establish quality."]}
    state = {"schema": "r2h_static_state_review_v1", "rule_version": STATE_VERSION,
             "artifact": binding(case["image"]), "package_sha256": file_sha256(case["package"]),
             "context": case["context"], "reviewer": actor["reviewer"],
             "evidence": [{k: v for k, v in actor["evidence"][0].items() if k != "frame_indices"}],
             "checks": {k: {"status": state_status, "evidence_ids": ["fixture"],
                            "note": "Synthetic test annotation."} for k in STATE_CHECKS},
             "limitations": actor["limitations"]}
    ap, sp = case["output"] / "actor_review.json", case["output"] / "state_review.json"
    write_json(ap, actor)
    write_json(sp, state)
    return ap, sp


def test_initial_payload_excludes_future_plan_and_terminal_conditions(case):
    request = read_json(case["output"] / "initial_request.json")
    assert "TERMINAL_ACTION_MUST_NOT_LEAK" not in request["prompt"]
    assert "LATE_STATE" not in request["prompt"]
    assert "event_assignments" not in request["prompt"]
    assert "transport_to_target" not in request["prompt"]
    assert "final_relations" not in request["prompt"]
    assert request["initial_conditions"]["camera_id"] == "opposite"


def test_active_initial_hand_role_rejected(case, tmp_path):
    placement = read_json(case["placement"])
    placement["actor"]["hands"]["right"]["initial_role"] = "release apple in target"
    write_json(case["placement"], placement, overwrite=True)
    with pytest.raises(ValueError, match="inactive"):
        module.prepare_unified(case["condition_path"], case["input_path"], case["placement"],
                               tmp_path / "bad", case["config"])


@pytest.mark.parametrize("rehash", [False, True])
def test_package_camera_change_is_rejected_even_with_updated_self_hash(case, rehash):
    package = read_json(case["package"])
    package["initial_context"]["camera_id"] = "native"
    if rehash:
        package["package_sha256"] = json_sha256({k: v for k, v in package.items() if k != "package_sha256"})
    write_json(case["package"], package, overwrite=True)
    with pytest.raises(ValueError, match="hash changed|contents changed"):
        module.validate_package(case["package"])


def test_rebound_actor_lineage_cannot_override_execution_source(case):
    plan_path = case["output"] / "actor_plan.json"
    plan = read_json(plan_path)
    plan["lineage"]["human_source_id"] = "unrelated_source"
    write_json(plan_path, plan, overwrite=True)
    package = read_json(case["package"])
    package["actor_plan"] = binding(plan_path)
    package["package_sha256"] = json_sha256({k: v for k, v in package.items() if k != "package_sha256"})
    write_json(case["package"], package, overwrite=True)
    with pytest.raises(ValueError, match="Actor plan source"):
        module.validate_package(case["package"])


def test_execute_rechecks_source_before_real_provider_adapter(case):
    result = generate(case)
    assert case["verify"].call_count == 2
    args, kwargs = case["provider"].call_args
    request = read_json(case["output"] / "initial_request.json")
    assert str(args[0]) == request["source_image"]["path"]
    assert args[1] == request["prompt"] and kwargs["model"] == request["model"]
    assert result["status"] == "needs_independent_actor_and_initial_state_reviews"
    assert not (case["output"] / "video_request.json").exists()


def test_tampered_initial_prompt_blocks_before_provider(case):
    path = case["output"] / "initial_request.json"
    request = read_json(path)
    request["prompt"] += "Move the apple into the destination now."
    write_json(path, request, overwrite=True)
    with pytest.raises(ValueError, match="Prepared initial request changed"):
        generate(case)
    case["provider"].assert_not_called()


def test_zero_budget_blocks_provider(case):
    budget = read_json(case["budget"])
    budget["cumulative_call_limits"]["image"] = 0
    write_json(case["budget"], budget, overwrite=True)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        generate(case)
    case["provider"].assert_not_called()


def test_actor_pass_does_not_replace_missing_state_review(case):
    generate(case)
    actor, _ = reviews(case)
    gate = module.video_gate(case["package"], case["image"], actor, None)
    assert gate["can_generate_video"] is False
    assert gate["actor_gate"]["can_generate_video"] is True
    assert "independent_initial_state_gate_not_passed" in gate["blocking_reasons"]


@pytest.mark.parametrize("dimension,status", [("actor", "unknown"), ("state", "unknown"), ("state", "fail")])
def test_either_failed_or_unknown_gate_blocks_video(case, dimension, status):
    generate(case)
    actor, state = reviews(case, **{dimension + "_status": status})
    result = module.prepare_video(case["package"], case["image"], actor, state, case["config"])
    assert result["status"] == "blocked"
    assert not (case["output"] / "video_request.json").exists()


def test_both_fixture_gates_prepare_only_and_submit_to_existing_adapter(case, monkeypatch):
    generate(case)
    actor, state = reviews(case)
    result = module.prepare_video(case["package"], case["image"], actor, state, case["config"])
    assert result["status"] == "prepared_video_candidate"
    assert result["quality_release_approved"] is False
    submit = Mock(return_value={"task_id": "fixture_video_only"})
    monkeypatch.setattr(module.SeedanceClient, "submit", submit)
    result = module.execute_video(case["package"], case["config"], case["budget"])
    assert result["task_id"] == "fixture_video_only"
    submit.assert_called_once()


def test_pass_annotations_without_generation_provenance_cannot_authorize_video(case):
    Image.new("RGB", (48, 48), "red").save(case["image"])
    actor, state = reviews(case)
    gate = module.video_gate(case["package"], case["image"], actor, state)
    assert gate["can_generate_video"] is False
    assert "missing_generation_provenance_for_planned_anchor" in gate["blocking_reasons"]


def test_changed_review_after_preparation_blocks_submission(case):
    generate(case)
    actor, state = reviews(case)
    module.prepare_video(case["package"], case["image"], actor, state, case["config"])
    review = read_json(state)
    review["checks"]["role_mapping"]["status"] = "fail"
    write_json(state, review, overwrite=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        module.execute_video(case["package"], case["config"], case["budget"])


def test_rebound_image_provenance_cannot_substitute_for_provider_pixels(case):
    generate(case)
    Image.new("RGB", (48, 48), "red").save(case["image"])
    provenance = read_json(case["output"] / "image_generation_provenance.json")
    provenance["image"] = binding(case["image"])
    write_json(case["output"] / "image_generation_provenance.json", provenance, overwrite=True)
    actor, state = reviews(case)
    with pytest.raises(ValueError, match="pixels differ"):
        module.video_gate(case["package"], case["image"], actor, state)


def test_copied_config_resumes_same_stage_and_does_not_reset_allowance(case, tmp_path):
    budget = read_json(case["budget"])
    budget["cumulative_call_limits"]["image"] = 1
    write_json(case["budget"], budget, overwrite=True)
    other = tmp_path / "copied_budget.json"
    write_json(other, budget)
    operation = Mock(return_value={"fixture_only": True})
    first = module._campaign_budgeted_call(case["budget"], "image", {"request": "one"}, operation)
    resumed = module._campaign_budgeted_call(other, "image", {"request": "one"}, operation)
    assert first == resumed
    operation.assert_called_once()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        module._campaign_budgeted_call(other, "image", {"request": "two"}, operation)


@pytest.mark.parametrize("field", ["ledger_path", "cumulative_call_limits"])
def test_registered_campaign_cannot_relocate_ledger_or_increase_limits(case, tmp_path, field):
    module._campaign_budgeted_call(case["budget"], "image", {"request": "one"}, lambda _: {})
    budget = read_json(case["budget"])
    if field == "ledger_path":
        budget[field] = str(tmp_path / "new_ledger")
    else:
        budget[field]["image"] = 100
    write_json(tmp_path / "changed_budget.json", budget)
    with pytest.raises(ValueError, match="Campaign ID already binds"):
        module._campaign_budgeted_call(tmp_path / "changed_budget.json", "image",
                                      {"request": "two"}, lambda _: pytest.fail("No extra call"))


def test_unknown_submission_remains_blocked_after_budget_copy(case, tmp_path):
    operation = Mock(side_effect=RuntimeError("fixture_unknown_submission"))
    with pytest.raises(RuntimeError, match="fixture_unknown"):
        module._campaign_budgeted_call(case["budget"], "image", {"request": "one"}, operation)
    copied = tmp_path / "copied_budget.json"
    write_json(copied, read_json(case["budget"]))
    with pytest.raises(RuntimeError, match="unresolved submission"):
        module._campaign_budgeted_call(copied, "image", {"request": "two"}, operation)
    operation.assert_called_once()
