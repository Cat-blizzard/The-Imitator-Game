"""Offline checks for an explicitly logged singleton-event container adapter."""
from copy import deepcopy
import hashlib
import json

import pytest

from data_generation_methods.r2h_pipeline.providers import ProviderError, parse_review_json
from data_generation_methods.r2h_pipeline.review_protocol import (
    ReviewProtocolError, adapt_provider_review_shape,
)
from data_generation_methods.r2h_pipeline.task_evidence import canonical_sha256
from data_generation_methods.r2h_pipeline.task_review import build_review_schema


@pytest.fixture
def case(tmp_path):
    binding = {"path": "/offline-fixture", "sha256": "a" * 64}
    contract = {
        "schema": "r2h_task_evidence_contract_v1", "task_id": "offline-protocol-fixture",
        "training_admitted": False, "entities": ["actor", "object", "target"],
        "source_spec": binding,
        "source_window": {
            "video": binding, "origin": "real_human", "split": "train",
            "window_frames": [0, 80], "fps": 10, "interval": "start_inclusive_end_exclusive"},
        "initial_relations": [
            {"id": "initial", "predicate": "inside", "args": ["object", "target"], "expected": False}],
        "final_relations": [
            {"id": "final", "predicate": "inside", "args": ["object", "target"], "expected": True}],
        "required_events": [
            {"id": "grasp", "verb": "grasp", "args": ["actor", "object"]},
            {"id": "release", "verb": "release_in", "args": ["object", "target"]}],
        "forbidden_events": [], "allowed_additional_events": [],
        "precedence": [["grasp", "release"]], "allowed_variations": [],
    }
    schema = build_review_schema(contract)
    event = {
        "observed": True, "temporal_semantics": "visible_milestone",
        "t_start": 1.25, "t_end": 1.25,
        "evidence": [{"frame_index": 30, "timestamp_seconds": 1.25, "description": "Visible fixture state."}],
        "description": "Fixture visible milestone.",
    }
    review = {
        "relations": {key: {"observed": "unknown", "evidence": [], "description": "Fixture unknown."}
                      for key in schema["properties"]["relations"]["properties"]},
        "events": {key: deepcopy(event) for key in schema["properties"]["events"]["properties"]},
        "additional_events": [], "limitations": ["Fixture only, not actual pixels."],
    }
    path = tmp_path / "provider_response.json"

    def save(value=None, *, raw_text=None, wrapper="{}", finish="STOP"):
        text = wrapper.format(json.dumps(review if value is None else value)) if raw_text is None else raw_text
        envelope = {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}],
                    "usageMetadata": {"promptTokenCount": 10}}
        path.write_text(json.dumps(envelope))
        return path
    return {"contract": contract, "schema": schema, "review": review, "path": path, "save": save}


@pytest.mark.parametrize("wrapper,normalization", [("{}", "none"), ("```json\n{}\n```", "outer_json_fence")])
def test_exact_wrapping_is_bound_logged_and_strictly_revalidated(case, wrapper, normalization):
    path = case["save"](wrapper=wrapper)
    before_bytes, original = path.read_bytes(), deepcopy(case["review"])
    sidecar = adapt_provider_review_shape(path, case["contract"])
    assert sidecar["provider_schema_compliant"] is False
    assert sidecar["schema_validated"] is True
    assert sidecar["raw_text_normalization"] == normalization
    assert sidecar["raw_artifact"] == {"path": str(path), "sha256": hashlib.sha256(before_bytes).hexdigest()}
    assert sidecar["contract_sha256"] == canonical_sha256(case["contract"])
    assert sidecar["response_schema_sha256"] == canonical_sha256(case["schema"])
    assert sidecar["task_correctness"] == "not_evaluated"
    assert not sidecar["training_admitted"] and not sidecar["quality_release_approved"]
    assert sidecar["provider_calls"] == 0
    assert not sidecar["values_fields_timestamps_evidence_modified"]
    assert len(sidecar["changes"]) == len(original["events"])
    for key, item in original["events"].items():
        assert sidecar["review"]["events"][key] == [item]
        change = next(value for value in sidecar["changes"] if value["path"] == "/events/" + key)
        assert change["original_item_sha256"] == change["wrapped_item_sha256"] == canonical_sha256(item)
        assert change["item_contents_changed"] is False
    assert parse_review_json(json.dumps(sidecar["review"]), case["schema"])["schema_validated"]
    with pytest.raises(ProviderError, match="review_schema_violation"):
        parse_review_json(json.dumps(original), case["schema"])
    assert path.read_bytes() == before_bytes
    assert case["review"] == original
    assert list(path.parent.iterdir()) == [path]  # No sidecar or rewritten response created implicitly.


def test_mixed_valid_arrays_and_singletons_preserve_array_occurrences(case):
    key = next(iter(case["review"]["events"]))
    event = case["review"]["events"][key]
    case["review"]["events"][key] = [deepcopy(event), deepcopy(event)]
    path = case["save"]()
    sidecar = adapt_provider_review_shape(path, case["contract"])
    assert sidecar["review"]["events"][key] == case["review"]["events"][key]
    assert len(sidecar["changes"]) == 1


def test_already_compliant_review_uses_normal_parser_not_false_adapter_label(case):
    for key, item in case["review"]["events"].items():
        case["review"]["events"][key] = [item]
    with pytest.raises(ReviewProtocolError, match="already satisfies"):
        adapt_provider_review_shape(case["save"](), case["contract"])


@pytest.mark.parametrize("mutation", [
    lambda review: review.pop("limitations"),
    lambda review: review.update(task_success=True),
    lambda review: review.update(event_queries=review.pop("events")),
    lambda review: review.update(relation_queries=review.pop("relations")),
    lambda review: review["events"].pop("event_000"),
    lambda review: review["events"].update(event_extra=review["events"]["event_000"]),
    lambda review: review["relations"]["relation_000"].pop("description"),
    lambda review: review["relations"]["relation_000"].update(observed=1),
    lambda review: review["additional_events"].append({"observed": True}),
    lambda review: review["events"].update(event_001=[{"observed": False}]),
])
def test_other_top_level_or_nested_differences_are_not_repaired(case, mutation):
    mutation(case["review"])
    path = case["save"]()
    original = path.read_bytes()
    with pytest.raises(ReviewProtocolError):
        adapt_provider_review_shape(path, case["contract"])
    assert path.read_bytes() == original


@pytest.mark.parametrize("mutation", [
    lambda event: event.pop("observed"),
    lambda event: event.pop("temporal_semantics"),
    lambda event: event.pop("description"),
    lambda event: event.pop("evidence"),
    lambda event: event.update(observed="true"),
    lambda event: event.update(t_start="1.25"),
    lambda event: event.update(start=event.pop("t_start")),
    lambda event: event["evidence"][0].update(frame_index=True),
    lambda event: event.update(confidence=1.),
])
def test_singleton_contents_must_already_satisfy_complete_item_schema(case, mutation):
    mutation(case["review"]["events"]["event_000"])
    with pytest.raises(ReviewProtocolError, match="incomplete or invalid"):
        adapt_provider_review_shape(case["save"](), case["contract"])


@pytest.mark.parametrize("observed", [False, "unknown"])
def test_false_and_unknown_are_preserved_without_filling_fields(case, observed):
    item = case["review"]["events"]["event_000"]
    item.update(observed=observed, temporal_semantics="unknown", t_start=None, t_end=None, evidence=[])
    sidecar = adapt_provider_review_shape(case["save"](), case["contract"])
    assert sidecar["review"]["events"]["event_000"] == [item]
    assert sidecar["review"]["events"]["event_000"][0]["observed"] == observed


@pytest.mark.parametrize("raw_text", [
    '{"relations":{},"relations":{}}',
    '{"value":NaN}',
    '{} trailing prose',
    '```json\n{}\n```\nMore text.',
])
def test_ambiguous_or_nonfinite_provider_text_remains_invalid(case, raw_text):
    with pytest.raises(ReviewProtocolError):
        adapt_provider_review_shape(case["save"](raw_text=raw_text), case["contract"])


@pytest.mark.parametrize("mutation", [
    lambda envelope: envelope["candidates"][0].update(finishReason="MAX_TOKENS"),
    lambda envelope: envelope["candidates"].append(deepcopy(envelope["candidates"][0])),
    lambda envelope: envelope.update(promptFeedback={"blockReason": "SAFETY"}),
])
def test_incomplete_blocked_or_multiple_candidate_envelopes_remain_invalid(case, mutation):
    path = case["save"]()
    envelope = json.loads(path.read_text())
    mutation(envelope)
    path.write_text(json.dumps(envelope))
    with pytest.raises(ReviewProtocolError):
        adapt_provider_review_shape(path, case["contract"])
