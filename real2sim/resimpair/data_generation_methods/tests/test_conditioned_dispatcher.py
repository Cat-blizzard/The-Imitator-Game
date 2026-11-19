import json

import pytest

from data_generation_methods.scripts import execute_conditioned_pipeline_20261020c as dispatch
from data_generation_methods.r2h_pipeline.common import write_json
from data_generation_methods.r2h_pipeline.repair_router import plan_repairs


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch.runner, "CASE", tmp_path)
    for name in ("source.mp4", "sim_initial.png", "human_initial_v1.png"):
        (tmp_path / name).write_bytes(("fixture " + name).encode())
    source = {
        "original_human_window": {"video": dispatch._bind(tmp_path / "source.mp4")},
        "sim_initial": dispatch._bind(tmp_path / "sim_initial.png"),
    }
    return tmp_path, source


def finish_stage(case_path, stage, result=None):
    write_json(case_path / stage / "call_started.json", {"started": True})
    write_json(case_path / stage / "result.json", result or {"status": "succeeded"})


def make_review(case_path, *, exact=True, hands=1):
    review = {"first_hands": {"minimum_visible_count": hands, "exact_count_known": exact,
                             "locations": [f"image-right hand {i}" for i in range(hands)]}}
    finish_stage(case_path, "condition_review_v1", {"review": review})
    write_json(case_path / "condition_review_v1/request_metadata.json", {
        "images": [{"label": "CANDIDATE HUMAN CLOSED",
                    **dispatch._bind(case_path / "human_initial_v1.png")}]
    })
    return review


def test_no_inventory_routes_to_global_review_not_localized(case):
    path, source = case
    (path / "human_terminal_v1.png").write_bytes(b"terminal")
    write_json(path / "dispatcher_v1/condition_transport_preparation.json", {
        "initial": dispatch._bind(path / "human_initial_v1.png"),
        "terminal": dispatch._bind(path / "human_terminal_v1.png")})
    plan = plan_repairs(dispatch.build_facts(source))
    assert plan["actions"][0]["kind"] == "review_anchor_body_global"
    assert not plan["actions"][0]["parameters"]["include_full_image_and_original_resolution_crops"]


def test_missing_reveal_handler_reuses_first_and_executes_terminal(case, monkeypatch):
    path, source = case
    finish_stage(path, "edit_initial_v1")
    calls = []
    monkeypatch.setattr(dispatch.runner, "edit_first", lambda cfg, src: calls.append("cached first"))
    monkeypatch.setattr(dispatch.runner, "edit_last", lambda cfg, src: calls.append("new last"))
    result = dispatch.execute_action({"kind": "extract_reveal_or_terminal_anchor"}, {}, source)
    assert result["executed"]
    assert calls == ["cached first", "new last"]
    assert result["new_paid_stages_expected"] == ["edit_terminal_v1"]


def test_real_terminal_edit_cost_is_bounded_even_though_extraction_plan_is_local(case, monkeypatch):
    path, source = case
    finish_stage(path, "edit_initial_v1")
    monkeypatch.setattr(dispatch.runner, "edit_last",
                        lambda *a: pytest.fail("Must not issue paid stage beyond budget"))
    result = dispatch.execute_action({"kind": "extract_reveal_or_terminal_anchor"},
                                     {}, source, max_total_paid_stages=1)
    assert not result["executed"]
    assert result["reason"] == "actual_paid_stage_budget_exhausted"


def test_unresolved_started_stage_cannot_be_reissued(case):
    path, source = case
    write_json(path / "edit_terminal_v1/call_started.json", {"started": True})
    with pytest.raises(RuntimeError, match="reconcile"):
        dispatch.execute_action({"kind": "extract_reveal_or_terminal_anchor"}, {}, source)


def test_derived_inventory_copies_original_uncertainty_and_locations(case):
    path, _ = case
    source = make_review(path, exact=False)
    mapped = dispatch.map_actor_evidence()
    assert mapped["review_scope"] == "global"
    assert mapped["review"]["exact_count_known"] is False
    assert mapped["review"]["hands"][0]["description"] == source["first_hands"]["locations"][0]
    sidecar = json.loads((path / "dispatcher_v1/initial_body_mapping.json").read_text())
    assert sidecar["source_observation"]["sha256"]
    assert not sidecar["observation_values_changed"]


def test_changed_anchor_cannot_reuse_old_observation(case):
    path, _ = case
    make_review(path)
    (path / "human_initial_v1.png").write_bytes(b"changed after review")
    with pytest.raises(ValueError, match="different initial image"):
        dispatch.map_actor_evidence()


def test_exact_legacy_mapping_gets_new_sidecar_without_changing_original(case):
    path, _ = case
    make_review(path)
    dispatch.map_actor_evidence()
    old = path / "dispatcher_v1/initial_body_mapping.json"
    legacy = json.loads(old.read_text())
    legacy.pop("source_request")
    write_json(old, legacy, overwrite=True)
    before = old.read_bytes()
    mapped = dispatch.map_actor_evidence()
    new = path / "dispatcher_v1/initial_body_mapping_v2.json"
    assert mapped["evidence_artifact"]["path"] == str(new)
    assert old.read_bytes() == before
    migrated = json.loads(new.read_text())
    assert migrated["legacy_mapping"] == dispatch._bind(old)
    assert migrated["review"] == legacy["review"]
    assert migrated["source_request"]["sha256"]
    assert dispatch.map_actor_evidence() == mapped


def test_legacy_mapping_with_changed_observations_cannot_be_migrated(case):
    path, _ = case
    make_review(path)
    dispatch.map_actor_evidence()
    old = path / "dispatcher_v1/initial_body_mapping.json"
    legacy = json.loads(old.read_text())
    legacy.pop("source_request")
    legacy["review"]["exact_count_known"] = False
    write_json(old, legacy, overwrite=True)
    with pytest.raises(ValueError, match="differs"):
        dispatch.map_actor_evidence()
    assert not (path / "dispatcher_v1/initial_body_mapping_v2.json").exists()


def test_unresolved_stage_is_visible_to_protocol_router(case):
    path, source = case
    write_json(path / "edit_terminal_v1/call_started.json", {"started": True})
    facts = dispatch.build_facts(source)
    plan = plan_repairs(facts)
    assert plan["state"] == "submission_reconciliation"
    assert plan["paid_calls_reserved"] == 0


def test_unimplemented_localized_review_stops_without_api(case):
    _, source = case
    result = dispatch.execute_action({"kind": "review_anchor_body_localized"}, {}, source)
    assert not result["executed"]
    assert result["reason"] == "handler_not_implemented"


def test_condition_gate_stops_video_despite_known_hand_count(case, monkeypatch):
    path, source = case
    write_json(path / "condition_review_v1/decision.json", {
        "ready_for_conditioned_generation": False, "issues": ["hidden_object_unknown"]})
    monkeypatch.setattr(dispatch.runner, "video", lambda *a, **k: pytest.fail("No video"))
    result = dispatch.execute_action({"kind": "submit_verified_conditioned_video"}, {}, source)
    assert not result["executed"]
    assert result["reason"] == "condition_gate_unresolved"


def test_plan_only_never_calls_handler(case, monkeypatch):
    _, source = case
    monkeypatch.setattr(dispatch, "execute_action", lambda *a, **k: pytest.fail("No handler"))
    result = dispatch.run({}, source, execute=False)
    assert result["mode"] == "plan_only"
    assert result["next_action"]["kind"] == "extract_reveal_or_terminal_anchor"


def test_transport_preparation_is_explicitly_local_not_provider_verified(case, monkeypatch):
    path, _ = case
    (path / "human_terminal_v1.png").write_bytes(b"terminal")
    payload = {"model": "fixture", "content": [
        {"type": "image_url", "role": "first_frame", "image_url": {"url": "data:test"}},
        {"type": "image_url", "role": "last_frame", "image_url": {"url": "data:test"}}]}
    monkeypatch.setattr(dispatch.runner, "build_video_payload", lambda cfg: payload)
    class Client:
        def __init__(self, cfg):
            pass
        def prepare_submit(self, value):
            assert value == payload
            return {"request": "local"}
    monkeypatch.setattr(dispatch, "SeedanceClient", Client)
    record = dispatch.prepare_transport({"seedance": {"model": "fixture"}})
    assert record["structured_input_roles"] == ["first_frame", "last_frame"]
    assert not record["provider_request_performed"]
    assert not record["provider_condition_use_verified"]


def test_actual_submission_reuses_the_exact_prepared_payload(case, monkeypatch):
    path, source = case
    payload = {"model": "fixture", "content": []}
    write_json(path / "condition_review_v1/decision.json", {
        "ready_for_conditioned_generation": True, "issues": []})
    write_json(path / "dispatcher_v1/condition_transport_preparation.json", {
        "payload_sha256": dispatch.json_sha256(payload)})
    monkeypatch.setattr(dispatch.runner, "build_video_payload", lambda cfg: payload)
    calls = []
    monkeypatch.setattr(dispatch.runner, "video", lambda cfg, **kw: calls.append(kw))
    result = dispatch.execute_action({"kind": "submit_verified_conditioned_video"}, {}, source)
    assert result["executed"]
    assert calls == [{"poll_minutes": 25}]


def test_execution_journal_records_only_actual_implemented_handler(case, monkeypatch):
    path, source = case
    def fake_handler(action, cfg, src, **kwargs):
        return {"executed": False, "reason": "intentional fixture stop", "kind": action["kind"]}
    monkeypatch.setattr(dispatch, "execute_action", fake_handler)
    result = dispatch.run({}, source, execute=True)
    journal = [json.loads(line) for line in
               (path / "dispatcher_v1/execution_journal.jsonl").read_text().splitlines()]
    assert journal[0]["phase"] == "action_started"
    assert journal[-1]["action_executed"] is False
    assert not result["outcome"]["executed"]


def test_plan_only_then_successful_execute_updates_report_status_and_preserves_history(case, monkeypatch):
    path, source = case
    first = dispatch.run({}, source, execute=False)
    latest = path / "dispatcher_v1/STATUS.json"
    assert json.loads(latest.read_text())["mode"] == "plan_only"
    monkeypatch.setattr(dispatch, "planned_action", lambda plan: {
        "kind": "submit_verified_conditioned_video", "execution_status": "planned"})
    monkeypatch.setattr(dispatch, "execute_action", lambda action, cfg, source, **kwargs: {
        "executed": True, "kind": action["kind"], "handler": "fixture_cached_video"})
    final = dispatch.run({}, source, execute=True)
    assert final["outcome"]["executed"]
    assert json.loads(latest.read_text()) == final
    snapshots = [json.loads(p.read_text()) for p in
                 (path / "dispatcher_v1/status_snapshots").glob("*.json")]
    assert first in snapshots and final in snapshots
    journal = [json.loads(line) for line in
               (path / "dispatcher_v1/execution_journal.jsonl").read_text().splitlines()]
    assert journal[-1]["action_executed"] is True


def test_postprocess_handler_requires_config_and_stops_after_one_completed_call(case, monkeypatch):
    from data_generation_methods.scripts import postprocess_conditioned_case_20261020c as post
    path, source = case
    action = {"kind": "inspect_task_evidence_and_training_pair", "execution_status": "planned"}
    missing = dispatch.execute_action(action, {}, source)
    assert not missing["executed"]
    assert missing["reason"] == "explicit_postprocess_configuration_required"
    video = path / "state_conditioned_human_v1.mp4"
    video.write_bytes(b"bound generated fixture")
    write_json(path / "postprocess_config_v1.json", {"video": dispatch._bind(video)})
    calls = []
    def completed(config_path):
        calls.append(config_path)
        return {
            "state": "awaiting_independent_pair_and_quality_release",
            "execution_record": {"path": "fixture report", "sha256": "0" * 64},
            "new_paid_review_calls": 0,
        }
    monkeypatch.setattr(post, "run_postprocess", completed)
    monkeypatch.setattr(dispatch, "plan_repairs", lambda facts: {
        "state": "candidate_inspection", "actions": [action],
        "allowed": {"video_submission": False}})
    result = dispatch.run({}, source, execute=True, max_actions=3)
    assert len(calls) == 1
    assert result["router_state"] == "awaiting_independent_pair_and_quality_release"
    assert result["next_action"] is None
    assert result["outcome"]["pair_validation"] == "not_run"
    assert not result["outcome"]["quality_release_approved"]
    assert not result["outcome"]["training_admitted"]
