import copy
import json
from pathlib import Path
import pytest

from data_generation_methods.r2h_pipeline import controlled_experiment as mod
from data_generation_methods.r2h_pipeline.common import write_json


@pytest.fixture
def prepared(tmp_path):
    asset = tmp_path/"source.png"
    from PIL import Image
    Image.new("RGB",(32,32),"white").save(asset)
    binding = mod.binding(asset)
    case = {k:copy.deepcopy(binding) for k in (
        "human_source","robot_video","robot_trajectory","sim_initial","sim_terminal",
        "human_initial","source_record")}
    case.update(id="apple", maximum_visible_hands=2,
                task={"initial":"Apple in source bowl", "necessary_steps":["grasp","lift","place","release"],
                      "goal":"Apple in target bowl", "invariants":["Do not move bowls"]})
    config = {
        "seedance":{"model":"doubao-seedance-2-5-260628","base_url":"https://example.invalid/api/v3"},
        "gemini":{"model":"gemini-3.1-pro-preview","base_url":"https://example.invalid/v1beta"},
        "anchor_model":"gemini-nano-banana-2.1",
    }
    root = tmp_path/"experiment"
    manifest = mod.create_experiment(root,[case],config)
    return root, manifest, case


def test_same_first_prompt_settings_and_model_only_roles_differ(prepared):
    root, manifest, case = prepared
    a = mod.video_payload(manifest,case,"A",root/"apple/A",{"attempt":1})
    bdir = root/"apple/B"
    (bdir/"human_terminal_01.png").write_bytes(Path(case["sim_terminal"]["path"]).read_bytes())
    b = mod.video_payload(manifest,case,"B",bdir,{"attempt":1,"terminal_version":1})
    assert a["content"][:2] == b["content"][:2]
    assert a["model"] == b["model"]
    assert a["duration"] == b["duration"] == 8
    assert a["resolution"] == b["resolution"] == "720p"
    assert [p.get("role") for p in b["content"][1:]] == ["first_frame","last_frame"]
    assert mod.SeedanceClient(manifest["config"]["seedance"]).prepare_submit(a)["json"] == a
    assert mod.SeedanceClient(manifest["config"]["seedance"]).prepare_submit(b)["json"] == b


def test_prepare_never_calls_provider_and_exposes_missing_terminal(prepared):
    root, manifest, _ = prepared
    assert mod.call_counts(root/"apple/A") == {"image":0,"video":0,"review":0}
    assert mod.plan_next(root,"apple","A")["operation"] == "submit_video"
    assert mod.plan_next(root,"apple","B")["operation"] == "edit_terminal"
    assert manifest["initial_comparison"]["candidate_count"] == 2


def test_unresolved_post_blocks_any_new_paid_stage(prepared):
    root, _, _ = prepared
    stage = root/"apple/A/stages/video_01"
    write_json(stage/"request_metadata.json",{"experiment_call_kind":"video"})
    write_json(stage/"call_started.json",{"started":True})
    plan = mod.plan_next(root,"apple","A")
    assert plan["operation"] == "blocked_unresolved_paid_stage"
    assert not plan["automatic_retry"]
    assert mod.call_counts(root/"apple/A")["video"] == 1


def test_known_task_only_polls_does_not_submit(prepared):
    root, _, _ = prepared
    stage = root/"apple/A/stages/video_01"
    write_json(stage/"result.json",{"task_id":"task_abc"})
    plan = mod.plan_next(root,"apple","A")
    assert plan["operation"] == "poll_video" and not plan["paid"]
    assert plan["task_id"] == "task_abc"


def test_retry_requires_completed_review_and_retains_initial(prepared):
    root, _, _ = prepared
    with pytest.raises(ValueError, match="completed prior review"):
        mod.plan_next(root,"apple","A",attempt=2)
    review = {"checks":{key:{"value":"yes"} for key in mod.CHECKS}, "review_windows":[]}
    write_json(root/"apple/A/attempt_01/review_summary.json",review)
    assert mod.plan_next(root,"apple","A",attempt=2)["operation"].startswith("stop_")
    assert mod.summarize(root)["records"][0]["initial"] == review
    assert mod.summarize(root)["records"][0]["recovery"] is None


def test_budget_exhaustion_prevents_call_even_on_new_stage(prepared, monkeypatch):
    root, _, _ = prepared
    arm = root/"apple/A"
    stage = arm/"stages/video_01"
    write_json(stage/"request_metadata.json",{"experiment_call_kind":"video"})
    write_json(stage/"call_started.json",{})
    def never(*args,**kwargs):
        raise AssertionError("Provider must not be called")
    monkeypatch.setattr(mod,"paid_stage",never)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        mod.budgeted_paid(arm,"video_02",{"experiment_call_kind":"video"},never,
                          {"image":2,"video":1,"review":2})


def test_b_retry_targets_observed_entity_failure_only_after_review(prepared):
    root, _, _ = prepared
    b = root/"apple/B"
    review = {"checks":{key:{"value":"yes"} for key in mod.CHECKS}, "review_windows":[]}
    review["checks"]["source_required_entities_preserved"] = {"value":"no","observation":"Target disappeared"}
    write_json(b/"attempt_01/review_summary.json",review)
    plan = mod.plan_next(root,"apple","B",attempt=2)
    assert plan["operation"] == "edit_terminal" and plan["terminal_version"] == 2
    assert "source_required_entities_preserved" in plan["repair_issues"]


def test_uniform_and_targeted_reviews_share_frame_budget_and_boundaries():
    baseline = mod.select_review_frames(193)
    targeted = mod.select_review_frames(193,[{"start_frame":70,"end_frame":74}])
    assert len(baseline) == len(targeted) == 24
    assert targeted[:4] == baseline[:4] == [0,1,2,3]
    assert targeted[-4:] == baseline[-4:] == [189,190,191,192]
    assert 72 in targeted


def test_vlm_cannot_invent_frames_or_pts():
    observation = {"value":"yes","frame_indices":[0],"timestamp_seconds":[0.],"observation":"Visible"}
    result = {"review":{"checks":{key:copy.deepcopy(observation) for key in mod.CHECKS},
                        "review_windows":[], "limitations":["Only sampled evidence."]}}
    frames = [{"frame_index":0,"timestamp_seconds":0.}]
    assert mod.validate_task_review(result,frames,193)
    result["review"]["checks"]["actor_plan"]["timestamp_seconds"] = [1.]
    with pytest.raises(ValueError,match="invented"):
        mod.validate_task_review(result,frames,193)


def test_manifest_does_not_claim_hidden_content_is_in_a_first_frame(prepared):
    root, manifest, case = prepared
    assert "source_required_entities_preserved" in mod.CHECKS
    assert manifest["cost_scope"]["new_B_terminal_edit"] == "charged to B image budget"
    assert manifest["initial_comparison"]["initial_and_recovery_results_must_be_reported_separately"]
    assert not manifest["training_admitted"]


@pytest.mark.parametrize("status",["failed","expired","cancelled","unknown"])
def test_terminal_or_unknown_get_stops_without_new_post(prepared, status):
    root, _, _ = prepared
    arm = root/"apple/A"
    write_json(arm/"stages/video_01/result.json",{"task_id":"task_abc","status":"submitted"})
    write_json(arm/"attempt_01/polls/100/normalized_response.json",
               {"task_id":"task_abc","status":"running"})
    write_json(arm/"attempt_01/polls/200/normalized_response.json",
               {"task_id":"task_abc","status":status})
    plan = mod.plan_next(root,"apple","A")
    assert plan["operation"].startswith("blocked_" if status == "unknown" else "stop_")
    assert not plan["paid"] and not plan["automatic_retry"]
    assert plan["status"] == status


def test_nominal_720p_square_is_960_square_not_height720(prepared):
    _, manifest, case = prepared
    evidence = mod.nominal_resolution_evidence(
        manifest["generation_settings"],case["human_initial"]["path"],
        {"width":960,"height":960})
    assert evidence["matches_documented_preset"] is True
    assert evidence["expected_dimensions_if_documented_standard_ratio"] == [960,960]
    assert mod.seed_policy(manifest)["policy"] == "not_specified"
    assert not mod.seed_policy(manifest)["paired_generator_randomness_claimed"]
