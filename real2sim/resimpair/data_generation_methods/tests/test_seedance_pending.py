import pytest

from data_generation_methods.r2h_pipeline.providers import ProviderError, SeedanceClient


def test_pending_is_nonterminal_and_original_status_is_retained(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test")
    client = SeedanceClient(
        {"model": "doubao-seedance-2-5-260628"},
        transport=lambda *args, **kwargs: {"id": "task_test", "status": "pending"},
    )
    result = client.poll("task_test")
    assert result["status"] == "queued"
    assert result["provider_status"] == "pending"
    assert result["output_url"] is None


def test_unrecognized_status_is_not_assumed_pending(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test")
    client = SeedanceClient(
        {"model": "doubao-seedance-2-5-260628"},
        transport=lambda *args, **kwargs: {"id": "task_test", "status": "mystery"},
    )
    with pytest.raises(ProviderError, match="invalid_task_status"):
        client.poll("task_test")
