"""Offline fixtures exercise the directed-review bridge; no paid calls or pixels."""
from copy import deepcopy
import json

import pytest

from data_generation_methods.r2h_pipeline.providers import ProviderError, parse_review_json
from data_generation_methods.r2h_pipeline.task_evidence import evaluate_task_evidence, file_sha256
from data_generation_methods.r2h_pipeline.task_review import (
    build_review_prompt, build_review_schema, normalize_review,
)


@pytest.fixture
def case(tmp_path):
    def artifact(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": str(path), "sha256": file_sha256(path)}

    contract = {
        "schema": "r2h_task_evidence_contract_v1", "task_id": "private-answer:put_apple_target",
        "training_admitted": False, "entities": ["actor", "apple", "source", "target"],
        "entity_roles": {"actor": "the acting hand", "source": "the initially occupied bowl",
                         "target": "the initially empty bowl"},
        "source_spec": artifact("private-source-spec.json", {"answer": "hidden_correct_answer"}),
        "source_window": {
            "video": artifact("private-source-video.json", {"fixture": True}),
            "origin": "real_human", "split": "train", "window_frames": [0, 80],
            "fps": 10, "interval": "start_inclusive_end_exclusive"},
        "initial_relations": [
            {"id": "private-initial-true", "predicate": "inside",
             "args": ["apple", "source"], "expected": True}],
        "final_relations": [
            {"id": "private-final-false", "predicate": "inside",
             "args": ["apple", "source"], "expected": False},
            {"id": "private-final-true", "predicate": "inside",
             "args": ["apple", "target"], "expected": True}],
        "required_events": [
            {"id": "private-grasp", "verb": "grasp", "args": ["actor", "apple"]},
            {"id": "private-lift", "verb": "lift_clear_of", "args": ["apple", "source"]},
            {"id": "private-release", "verb": "release_in", "args": ["apple", "target"]},
            {"id": "private-withdraw", "verb": "withdraw_from", "args": ["actor", "target"]}],
        "forbidden_events": [],
        "allowed_additional_events": [],
        "precedence": [["private-grasp", "private-lift"], ["private-lift", "private-release"],
                       ["private-release", "private-withdraw"]],
        "allowed_variations": ["appearance"],
        "source_final_description": "hidden_correct_answer",
    }
    frames = []
    for index, time in [(0, 0.), (10, 1.), (11, 1.1), (30, 3.), (50, 5.), (60, 6.), (79, 7.9)]:
        frame = artifact(f"frame-{index}.png", {"fixture_only": index})
        frames.append({"path": frame["path"], "frame_index": index, "timestamp_seconds": time})
    media = {**artifact("candidate.mp4", {"fixture_only": True}), "duration_seconds": 8.}
    public = json.loads(build_review_prompt(contract).split("\n")[-1])

    def point(index):
        frame = next(frame for frame in frames if frame["frame_index"] == index)
        return {"frame_index": index, "timestamp_seconds": frame["timestamp_seconds"],
                "description": "Unit fixture describes a visible state, not actual pixels."}

    review = {"relations": {}, "events": {}, "additional_events": [], "limitations": []}
    for key, relation in public["relation_queries"].items():
        initial = relation["phase"] == "initial"
        truth = initial or relation["args"][-1] == "target"
        review["relations"][key] = {
            "observed": truth, "evidence": [point(0 if initial else 79)],
            "description": "Unit fixture boundary relation."}
    for key, event in public["event_queries"].items():
        index = {"grasp": 10, "lift_clear_of": 30, "release_in": 50, "withdraw_from": 60}[event["verb"]]
        evidence = point(index)
        review["events"][key] = [{
            "observed": True, "temporal_semantics": "visible_milestone",
            "t_start": evidence["timestamp_seconds"], "t_end": evidence["timestamp_seconds"],
            "evidence": [evidence], "description": "Unit fixture visible milestone."}]
    return {
        "contract": contract, "media": media, "frames": frames, "review": review, "point": point,
        "artifact": artifact, "public": public,
        "policy": {"time_tolerance_seconds": 0, "initial_window_seconds": 1,
                   "final_window_seconds": 1, "motion_requirements": []},
    }


def normalized(case, result=None, **kwargs):
    result = case["review"] if result is None else result
    args = {"media": case["media"], "supplied_frames": case["frames"],
            "review_artifact": case["artifact"]("review.json", result)}
    args.update(kwargs)
    return normalize_review(case["contract"], result, **args)


def decision(case, evidence=None):
    return evaluate_task_evidence(case["contract"], normalized(case) if evidence is None else evidence,
                                  policy=case["policy"])


def event(case, verb):
    key = next(key for key, value in case["public"]["event_queries"].items() if value["verb"] == verb)
    return key, case["review"]["events"][key][0]


def relation(case, phase):
    key = next(key for key, value in case["public"]["relation_queries"].items()
               if value["phase"] == phase)
    return key, case["review"]["relations"][key]


def test_schema_and_prompt_hide_private_answers_and_remain_directed(case):
    contract = case["contract"]
    before = deepcopy(contract)
    schema = build_review_schema(contract)
    prompt = build_review_prompt(contract)
    public = prompt + json.dumps(schema)
    for secret in ("private-answer", "private-initial-true", "private-final-false",
                   "private-grasp", "hidden_correct_answer", "private-source", "precedence",
                   '"expected"', '"required_events"', '"forbidden_events"'):
        assert secret not in public
    swapped = deepcopy(contract)
    swapped["task_id"] = "different-answer"
    for phase in ("initial", "final"):
        for item in swapped[phase + "_relations"]:
            item["expected"] = not item["expected"]
    assert build_review_prompt(swapped) == prompt
    assert build_review_schema(swapped) == schema
    assert "visible_milestone" in prompt
    assert "contact force" in prompt
    assert "Sparse images" in prompt
    assert contract == before


def test_required_and_forbidden_labels_and_order_are_not_disclosed(case):
    contract = case["contract"]
    first = build_review_prompt(contract)
    contract["forbidden_events"].append(contract["required_events"].pop())
    contract["precedence"] = contract["precedence"][:-1]
    assert build_review_prompt(contract) == first


def test_strict_schema_checks_every_nested_object(case):
    schema = build_review_schema(case["contract"])
    assert parse_review_json(json.dumps(case["review"]), schema)["schema_validated"]

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["properties"]) == set(node["required"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)
    check(schema)
    key, item = relation(case, "initial")
    item["observed"] = 1
    with pytest.raises(ProviderError, match="review_schema_violation"):
        parse_review_json(json.dumps(case["review"]), schema)


def test_prompt_embeds_exact_schema_and_complete_neutral_unknown_examples(case):
    prompt = build_review_prompt(case["contract"])
    public = json.loads(prompt.split("\n")[-1])
    schema = build_review_schema(case["contract"])
    assert public["response_json_schema"] == schema
    assert "QUERY DEFINITIONS, NOT the output template" in prompt
    assert "never quoted strings" in prompt
    examples = public["unknown_observation_examples"]
    assert examples["relation"] == {
        "observed": "unknown", "evidence": [],
        "description": "The supplied frames do not establish this relation."}
    assert examples["event"] == {
        "observed": "unknown", "temporal_semantics": "unknown",
        "t_start": None, "t_end": None, "evidence": [],
        "description": "The supplied frames do not establish this event."}
    unknown = {
        "relations": {key: deepcopy(examples["relation"]) for key in public["relation_queries"]},
        "events": {key: [deepcopy(examples["event"])] for key in public["event_queries"]},
        "additional_events": [], "limitations": [],
    }
    assert parse_review_json(json.dumps(unknown), schema)["schema_validated"]
    assert decision(case, normalized(case, unknown))["decision"] == "needs_review"


def test_query_named_provider_output_with_missing_fields_stays_rejected(case):
    invalid = {
        "relation_queries": {
            key: {"observed": value["observed"], "evidence": value["evidence"]}
            for key, value in case["review"]["relations"].items()},
        "event_queries": {
            key: [{"start": str(items[0]["t_start"]), "end": str(items[0]["t_end"]),
                   "evidence": items[0]["evidence"]}]
            for key, items in case["review"]["events"].items()},
        "additional_events": [],
    }
    with pytest.raises(ValueError, match="schema"):
        normalized(case, invalid)
    assert "relations" not in invalid
    assert "events" not in invalid


@pytest.mark.parametrize("mutation", [
    lambda review: review.update(expected_goal="private"),
    lambda review: review.pop("additional_events"),
    lambda review: review["events"]["event_000"][0].update(confidence=1),
    lambda review: review["events"]["event_000"][0]["evidence"][0].update(unknown_field=True),
    lambda review: review["events"]["event_000"][0]["evidence"][0].update(frame_index=True),
])
def test_normalizer_revalidates_schema_instead_of_trusting_envelope(case, mutation):
    mutation(case["review"])
    envelope = {"review": case["review"], "schema_validated": True, "parse_status": "valid"}
    with pytest.raises(ValueError, match="schema"):
        normalized(case, envelope)


def test_sparse_milestones_can_be_task_candidates_without_motion_or_release(case):
    before = deepcopy((case["review"], case["frames"], case["contract"]))
    evidence = normalized(case)
    result = decision(case, evidence)
    assert result["decision"] == "task_equivalent_candidate"
    assert all(item["t_start"] == item["t_end"] for item in evidence["event_observations"])
    assert result["motion_verified"] is False
    assert result["training_admitted"] is False
    assert result["quality_release_approved"] is False
    assert evidence["review_metadata"]["schema_validated"] is True
    assert evidence["review_metadata"]["task_correctness"] == "not_evaluated"
    assert evidence["review_metadata"]["blind_review"] is False
    assert not evidence["refinement_requests"]
    assert before == (case["review"], case["frames"], case["contract"])


def test_actual_false_relation_is_preserved_even_when_expected_true(case):
    _, item = relation(case, "initial")
    item["observed"] = False
    evidence = normalized(case)
    observed = next(item for item in evidence["relation_observations"] if item["phase"] == "initial")
    assert observed["state"] == "observed"
    assert observed["value"] is False
    assert decision(case, evidence)["decision"] == "rejected"


@pytest.mark.parametrize("reported", ["unknown", True, False])
def test_missing_relation_frame_evidence_remains_unknown(case, reported):
    _, item = relation(case, "initial")
    item.update(observed=reported, evidence=[])
    evidence = normalized(case)
    observed = next(item for item in evidence["relation_observations"] if item["phase"] == "initial")
    assert observed["state"] == "unknown" and observed["value"] is None
    assert decision(case, evidence)["decision"] == "needs_review"
    assert evidence["refinement_requests"][0]["frame_interval"] == [0, 79]


@pytest.mark.parametrize("field,value", [
    ("frame_index", 5), ("timestamp_seconds", .5), ("timestamp_seconds", 99),
    ("timestamp_seconds", -1), ("timestamp_seconds", float("nan")),
])
def test_citations_must_be_exact_supplied_frames_with_valid_times(case, field, value):
    _, item = relation(case, "initial")
    item["evidence"][0][field] = value
    with pytest.raises(ValueError):
        normalized(case)


@pytest.mark.parametrize("start,end", [(1., 9.), (5., 1.), (1.05, 1.1), (None, 1.), (True, True)])
def test_event_bounds_cannot_invent_times_or_leave_clip(case, start, end):
    _, item = event(case, "grasp")
    item.update(t_start=start, t_end=end)
    with pytest.raises(ValueError):
        normalized(case)


def test_sparse_transition_is_unknown_and_requests_dense_bracket(case):
    key, item = event(case, "lift_clear_of")
    item.update(temporal_semantics="visible_transition", t_start=1.1, t_end=3.,
                evidence=[case["point"](11), case["point"](30)])
    evidence = normalized(case)
    actual = next(item for item in evidence["event_observations"] if item["verb"] == "lift_clear_of")
    assert actual["state"] == "unknown"
    refinement = next(item for item in evidence["refinement_requests"] if item["query_id"] == key)
    assert refinement["frame_interval"] == [11, 30]
    assert refinement["time_interval_seconds"] == [1.1, 3.]
    assert decision(case, evidence)["decision"] == "needs_review"


def test_dense_transition_uses_short_visible_transition_not_held_state(case):
    _, item = event(case, "grasp")
    item.update(temporal_semantics="visible_transition", t_start=1., t_end=1.1,
                evidence=[case["point"](10), case["point"](11)])
    evidence = normalized(case)
    actual = next(item for item in evidence["event_observations"] if item["verb"] == "grasp")
    assert (actual["t_start"], actual["t_end"], actual["state"]) == (1., 1.1, "observed")
    assert decision(case, evidence)["decision"] == "task_equivalent_candidate"


def test_milestone_does_not_accept_long_hold_interval(case):
    _, item = event(case, "grasp")
    item.update(t_end=6.)
    with pytest.raises(ValueError, match="milestone"):
        normalized(case)


def test_event_point_requires_explicit_milestone_semantics_at_gate(case):
    evidence = normalized(case)
    evidence["event_observations"][0]["temporal_semantics"] = "visible_transition"
    assert decision(case, evidence)["decision"] == "rejected"
    evidence["event_observations"][0].update(state="unknown", temporal_semantics="visible_milestone")
    assert decision(case, evidence)["decision"] == "needs_review"


def test_temporal_unknown_does_not_become_milestone_from_true_boolean(case):
    _, item = event(case, "grasp")
    item["temporal_semantics"] = "unknown"
    evidence = normalized(case)
    assert next(item for item in evidence["event_observations"] if item["verb"] == "grasp")["state"] == "unknown"
    assert decision(case, evidence)["decision"] == "needs_review"


def test_forbidden_nondetection_never_expands_sparse_frames_to_full_clip_coverage(case):
    case["contract"]["forbidden_events"] = [{"id": "move_bowl", "verb": "move", "args": ["source"]}]
    case["public"] = json.loads(build_review_prompt(case["contract"]).split("\n")[-1])
    old = list(case["review"]["events"].values())
    case["review"]["events"] = {}
    for key, signature in case["public"]["event_queries"].items():
        if signature["verb"] == "move":
            item = {"observed": False, "temporal_semantics": "unknown", "t_start": 0., "t_end": 7.9,
                    "evidence": [case["point"](0), case["point"](79)],
                    "description": "No movement seen in the two endpoint frames."}
        else:
            item = old.pop(0)[0]
        case["review"]["events"][key] = [item]
    evidence = normalized(case)
    actual = next(item for item in evidence["event_observations"] if item["verb"] == "move")
    assert actual["state"] == "unknown"
    result = decision(case, evidence)
    assert result["decision"] == "needs_review"
    assert "FORBIDDEN_EVENT_COVERAGE_MISSING" in {issue["code"] for issue in result["issues"]}


def test_extra_visible_action_reaches_existing_gate(case):
    item = deepcopy(event(case, "grasp")[1])
    item.update(verb="push", args=["source"])
    case["review"]["additional_events"].append(item)
    result = decision(case)
    assert result["decision"] == "needs_review"
    assert "UNEXPLAINED_OBSERVED_EVENT" in {issue["code"] for issue in result["issues"]}


def test_one_observation_cannot_satisfy_two_required_occurrences(case):
    case["contract"]["required_events"].append(
        {"id": "another-grasp", "verb": "grasp", "args": ["actor", "apple"]})
    assert decision(case)["decision"] == "rejected"


def test_schema_valid_wrong_order_is_rejected_by_gate_only(case):
    _, item = event(case, "release_in")
    item.update(t_start=0., t_end=0., evidence=[case["point"](0)])
    evidence = normalized(case)
    assert evidence["review_metadata"]["schema_validated"] is True
    assert decision(case, evidence)["decision"] == "rejected"


def test_review_envelope_and_artifact_content_must_match(case):
    envelope = {"review": case["review"], "raw_text": json.dumps(case["review"]),
                "parse_status": "valid", "schema_validated": True, "normalization": "none"}
    assert normalized(case, envelope)["review_metadata"]["schema_validated"]
    other = deepcopy(case["review"])
    next(iter(other["relations"].values()))["observed"] = "unknown"
    wrong = case["artifact"]("different-review.json", other)
    with pytest.raises(ValueError, match="different review"):
        normalized(case, review_artifact=wrong)


@pytest.mark.parametrize("target", ["media", "review_artifact"])
def test_hash_bound_files_cannot_be_changed(case, target):
    value = (deepcopy(case["media"]) if target == "media"
             else case["artifact"]("wrong-report.json", case["review"]))
    value["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="SHA256"):
        normalized(case, **{target: value})


@pytest.mark.parametrize("mutation", [
    lambda frames: frames.reverse(),
    lambda frames: frames[1].update(frame_index=0),
    lambda frames: frames[1].update(frame_index=True),
    lambda frames: frames[1].update(timestamp_seconds=0.),
    lambda frames: frames[1].update(timestamp_seconds=9.),
    lambda frames: frames[1].update(timestamp_seconds=float("inf")),
    lambda frames: frames[1].update(sha256="0" * 64),
])
def test_supplied_frame_manifest_is_verified(case, mutation):
    mutation(case["frames"])
    with pytest.raises(ValueError):
        normalized(case)


def test_stable_support_requires_more_than_single_frame(case):
    case["contract"]["final_relations"][0]["predicate"] = "stably_supported_by"
    case["public"] = json.loads(build_review_prompt(case["contract"]).split("\n")[-1])
    case["review"]["relations"] = {
        key: {"observed": True, "evidence": [case["point"](0 if item["phase"] == "initial" else 79)],
              "description": "One fixture still image."}
        for key, item in case["public"]["relation_queries"].items()}
    evidence = normalized(case)
    relation = next(item for item in evidence["relation_observations"] if item["predicate"] == "stably_supported_by")
    assert relation["state"] == "unknown"
    assert relation["value"] is None


def test_comparison_relation_binds_final_claim_and_keeps_initial_reference(case):
    case["contract"]["final_relations"] = [
        {"id": "same_source", "predicate": "unchanged_from_initial", "args": ["source"], "expected": True}]
    public = json.loads(build_review_prompt(case["contract"]).split("\n")[-1])
    case["review"]["relations"] = {}
    for key, item in public["relation_queries"].items():
        indices = [0] if item["phase"] == "initial" else [0, 79]
        case["review"]["relations"][key] = {
            "observed": True, "evidence": [case["point"](index) for index in indices],
            "description": "Unit fixture compares initial and final appearance."}
    evidence = normalized(case)
    final = next(item for item in evidence["relation_observations"] if item["phase"] == "final")
    assert (final["t_start"], final["t_end"], final["state"]) == (7.9, 7.9, "observed")
    assert final["comparison_frame_evidence"][0]["frame_index"] == 0
    assert decision(case, evidence)["decision"] == "task_equivalent_candidate"
    final_key = next(key for key, item in public["relation_queries"].items() if item["phase"] == "final")
    case["review"]["relations"][final_key]["evidence"] = [case["point"](79)]
    assert decision(case)["decision"] == "needs_review"


def test_book_and_cabinet_predicate_and_event_definitions_are_explicit(case):
    predicates = ["closed", "open", "supported_by", "unchanged_from_initial"]
    verbs = ["place_inside", "open", "remove_from_cabinet"]
    contract = case["contract"]
    for i, predicate in enumerate(predicates):
        contract["final_relations"].append({
            "id": f"extra_relation_{i}", "predicate": predicate, "args": ["source"], "expected": True})
    for i, verb in enumerate(verbs):
        contract["allowed_additional_events"].append({
            "id": f"extra_event_{i}", "verb": verb, "args": ["source"]})
    public = json.loads(build_review_prompt(contract).split("\n")[-1])
    assert all(not item["definition"].startswith("Use the literal")
               for item in public["relation_queries"].values() if item["predicate"] in predicates)
    assert all(not item["definition"].startswith("Use a literal")
               for item in public["event_queries"].values() if item["verb"] in verbs)
