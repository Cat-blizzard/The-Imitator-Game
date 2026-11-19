"""Accounting and no-resubmit properties; these are not model-quality tests."""
import json
from pathlib import Path

import pytest
from resimpair_tasks import human_examples_v2 as candidate


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return candidate.bind(path)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, "ROOT", tmp_path)
    monkeypatch.setattr(candidate, "REGISTRY", tmp_path / "registry")
    put(tmp_path / "provenance/lineage_snapshot.json", {"lineage_started_calls": {"image":30,"video":23,"review":52}})
    ledger=tmp_path / "old_ledger"; spec={"campaign_id":"old","ledger_path":str(ledger)}
    put(tmp_path / "registry/old.json",spec);put(ledger / "budget.json",spec)
    task=put(tmp_path / "old_contract.json",{"task_id":"old_task"})
    request=put(tmp_path / "old_request.json",{"contract":task})
    stage=ledger / "stages/video_old"
    marker=put(stage / "call_started.json",{"kind":"video","request":request})
    failure=put(stage / "failure.json",{"provider_outcome":"unknown","http_status":503})
    return {"pending_call":marker,"original_failure":failure,"original_request":request},stage


def test_contained_unknown_still_counted(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch)
    h=candidate.historical(c)
    assert h["started_calls"] == {"image":30,"video":24,"review":52}
    assert h["attempted_video_tasks"] == ["old_task"]
    assert h["markers"] == [c["pending_call"]]


def test_second_unknown_stops_lineage(tmp_path,monkeypatch):
    c,stage=fixture(tmp_path,monkeypatch)
    put(stage.parent / "video_other/call_started.json",{"kind":"video","request":c["original_request"]})
    put(stage.parent / "video_other/failure.json",{"provider_outcome":"unknown","http_status":503})
    with pytest.raises((ValueError,RuntimeError)):candidate.historical(c)


def test_changed_unknown_not_silently_exempted(tmp_path,monkeypatch):
    c,stage=fixture(tmp_path,monkeypatch)
    put(stage / "failure.json",{"provider_outcome":"unknown","http_status":500})
    with pytest.raises((ValueError,RuntimeError)):candidate.historical(c)


def test_new_definite_failure_also_counted(tmp_path,monkeypatch):
    c,stage=fixture(tmp_path,monkeypatch)
    put(stage.parent / "review_failed/call_started.json",{"kind":"review"})
    put(stage.parent / "review_failed/failure.json",{"provider_outcome":"definite_failure","http_status":400})
    assert candidate.historical(c)["started_calls"]["review"]==53
