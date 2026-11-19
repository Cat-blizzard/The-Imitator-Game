import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from resimpair_tasks.review_wire_v2 import omit_additional_properties
from resimpair_tasks.human_pilot_recovery_v2 import OLD_IMPORT, NEW_IMPORT
from corl2026_workshop.experiments.autonomous_pipeline_v4 import budget as retained_budget
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind, read, write


def test_nested_wire_dialect_does_not_weaken_local_schema():
    schema = {"type": "object", "additionalProperties": False,
        "properties": {"checks": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"value": {"type": "string", "enum": ["yes", "no", "unknown"]}}, "required": ["value"]}}},
        "required": ["checks"]}
    original = copy.deepcopy(schema)
    wire, edits = omit_additional_properties(schema)
    assert schema == original and len(edits) == 2
    good = {"checks": [{"value": "unknown"}]}
    Draft202012Validator(schema).validate(good)
    Draft202012Validator(wire).validate(good)
    for bad in ({"checks": [{"value": "yes", "invented": True}]}, {"checks": [], "extra": True}):
        Draft202012Validator(wire).validate(bad)
        with pytest.raises(ValidationError): Draft202012Validator(schema).validate(bad)
    for bad in ({"checks": [{"value": "maybe"}]}, {"checks": [{}]}, {}):
        with pytest.raises(ValidationError): Draft202012Validator(schema).validate(bad)


def test_schema_property_name_is_not_treated_as_keyword():
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "additionalProperties": {"type": "string"}, "x": {"anyOf": [{"type": "object", "additionalProperties": False}]}}}
    wire, changes = omit_additional_properties(schema)
    assert wire["properties"]["additionalProperties"] == {"type": "string"}
    assert len(changes) == 2


def test_copy_changes_model_dispatch_only():
    source = Path("corl2026_workshop/experiments/autonomous_pipeline_v4/models.py").read_text()
    assert source.count(OLD_IMPORT) == 1
    changed = source.replace(OLD_IMPORT, NEW_IMPORT)
    assert changed.replace(NEW_IMPORT, OLD_IMPORT) == source
    assert "c._review_result(response, request[\"response_schema\"], True)" in changed


@pytest.fixture
def budget(tmp_path, monkeypatch):
    monkeypatch.setattr(retained_budget, "REGISTRY", tmp_path / "registry")
    path = tmp_path / "budget.json"
    write(path, {"schema": "resimpair_cumulative_budget_v4", "campaign_id": "review_wire_test",
        "ledger_path": str((tmp_path / "ledger").resolve()), "max_concurrent": 1,
        "new_call_limits": {"image": 0, "video": 0, "review": 2},
        "ancestor_started_calls": {"image": 30, "video": 23, "review": 52},
        "lineage_cumulative_limits": {"image": 30, "video": 23, "review": 54},
        "ancestor_evidence": [], "timeout_seconds": 1})
    return retained_budget.Budget(path)


def request(path, code):
    write(path, {"implementation": {"models.py": {"path": str(path.parent / "test_dispatch.py"), "sha256": code}},
                 "response_schema": {"type": "object", "additionalProperties": False}})
    return path


def test_http400_new_protocol_keeps_failed_call_and_shared_ledger(budget, tmp_path):
    old = request(tmp_path / "old.json", "old")
    class Definite(Exception):
        http_status = 400
        code = "http_error"
    with pytest.raises(Definite):
        budget.call("review", old, lambda stage: (_ for _ in ()).throw(Definite()))
    failed = budget.directory("review", old) / "failure.json"
    saved = failed.read_bytes()
    with pytest.raises(ValueError, match="new_protocol_revision"):
        budget.call("review", request(tmp_path / "same_protocol.json", "old"), lambda stage: {})
    alias = tmp_path / "repair/budget.json"; alias.parent.mkdir(); alias.symlink_to(budget.path)
    shared = retained_budget.Budget(alias)
    result, stage = shared.call("review", request(tmp_path / "new_protocol.json", "new"), lambda stage: {"test_fixture": True})
    assert shared.ledger == budget.ledger and result == {"test_fixture": True}
    assert shared.usage()["new_started"]["review"] == 2
    assert shared.usage()["lineage_started"]["review"] == 54
    assert failed.read_bytes() == saved
    with pytest.raises(RuntimeError, match="no_resubmission"):
        shared.call("review", old, lambda stage: {})


def test_unknown_old_submission_blocks_even_new_protocol(budget, tmp_path):
    old = request(tmp_path / "unknown.json", "old")
    with pytest.raises(TimeoutError):
        budget.call("review", old, lambda stage: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(ValueError, match="unresolved_provider_submission"):
        budget.call("review", request(tmp_path / "new_after_unknown.json", "new"), lambda stage: {})
    assert budget.usage()["new_started"]["review"] == 1
