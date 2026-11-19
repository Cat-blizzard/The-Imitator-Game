import json

import pytest

from data_generation_methods.r2h_pipeline.batch_pipeline import paid_stage
from data_generation_methods.r2h_pipeline.providers import ProviderError


def test_completed_stage_is_reused_without_another_post(tmp_path):
    calls = []
    def operation(transport):
        calls.append(1)
        return {"status": "succeeded", "review": {"observed": "unknown"}}
    first = paid_stage(tmp_path / "stage", {"input_hash": "a"}, operation)
    second = paid_stage(tmp_path / "stage", {"input_hash": "a"}, operation)
    assert first == second
    assert len(calls) == 1
    assert second["training_admitted"] is False


def test_changed_source_cannot_reuse_a_result(tmp_path):
    stage = tmp_path / "stage"
    paid_stage(stage, {"input_hash": "a"}, lambda _: {"status": "succeeded"})
    with pytest.raises(ValueError, match="different inputs"):
        paid_stage(stage, {"input_hash": "b"}, lambda _: pytest.fail("No second request"))


def test_ambiguous_submission_never_automatically_retries(tmp_path):
    stage = tmp_path / "stage"
    def fail(_):
        raise ProviderError("submission_unknown", submission_unknown=True)
    with pytest.raises(ProviderError):
        paid_stage(stage, {"input_hash": "a"}, fail)
    with pytest.raises(RuntimeError, match="automatic retry prohibited"):
        paid_stage(stage, {"input_hash": "a"}, lambda _: pytest.fail("Must not issue a request"))
    assert json.loads((stage / "failure.json").read_text())["submission_unknown"] is True


def test_unbound_result_is_rejected(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "result.json").write_text('{"status":"succeeded"}')
    with pytest.raises(ValueError, match="lacks input binding"):
        paid_stage(stage, {"input_hash": "a"}, lambda _: pytest.fail("Must not issue a request"))
