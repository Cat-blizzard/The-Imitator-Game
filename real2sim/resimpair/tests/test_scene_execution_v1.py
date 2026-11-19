"""Actual JSON/ledger and retained-prepare integration; never launch physics."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_scene import execution


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root, registry, parent = tmp_path / "campaign", tmp_path / "registry", tmp_path / "parent"
    monkeypatch.setattr(execution, "ALLOCATION_REGISTRY", registry)
    limits = {"physical_attempt_limit": 11, "parallel_workers": 1, "timeout_seconds": 37,
        "retries_per_request": 0, "API_calls": 0, "GPU_jobs": 0,
        "allocation_id": "unit_allocation", "prior_scope_limit": 14, "cumulative_scope_limit": 25}
    write(root / "budget/limits.json", limits)
    receipt = registry / "unit_allocation.json"
    write(receipt, {"root": str(root.resolve()), "budget": binding(root / "budget/limits.json"),
        "allocation_id": limits["allocation_id"], "limit": limits["physical_attempt_limit"]})
    write(root / "allocation.json", {"registration": binding(receipt)})
    # Actual old prepare_cohort implementations read this isolated parent; the
    # old driver source/dependency files and all immutable JSON writes are real.
    write(parent / "experiment.json", {"code": {}, "legacy_code": {}})
    write(parent / "rules.json", {"files": {}, "test_fixture_not_physical_evidence": True})
    for name in execution.PROFILE_REGISTRY:
        driver, _ = execution._profile(name)
        monkeypatch.setattr(driver, "PARENT", parent)
    return root


def cases(campaign, count, prefix="task"):
    rows = []
    for i in range(count):
        case_id = f"{prefix}_{i}"
        folder = campaign.parent / "inputs" / case_id
        write(folder / "scene.json", {"id": case_id, "seed": i})
        write(folder / "task.json", {"id": case_id, "operation": "place_on"})
        write(folder / "source_request.json", {"id": case_id})
        rows.append({"id": case_id, "execution_id": case_id, "eligible": True,
            "scene": binding(folder / "scene.json"), "task": binding(folder / "task.json"),
            "source_request": binding(folder / "source_request.json"), "requested_task_id": "place_on"})
    return rows


def record_submission(campaign, name, case, *, terminal=True):
    """Ledger fixture only: never represents or claims a simulated trajectory."""
    folder = campaign / name
    index = len(list((campaign / "budget/reservations").glob("*.json")))
    record = {"case": case, "cohort": name, "campaign": str(folder),
              "protocol": binding(folder / "protocol.json"), "test_fixture": True}
    write(campaign / "budget/reservations" / f"{index:04d}.json", record)
    write(folder / "executions" / case["id"] / "reservation.json", record)
    if terminal:
        write(folder / "executions" / case["id"] / "terminal.json",
              {"case": case["id"], "physical_attempts": 1, "test_fixture_not_physical_evidence": True})


def test_config_registry_keeps_default_and_candidate_boundaries():
    description = execution.describe_profiles()
    assert description["default_profile"] == "checked_withdrawal"
    assert {k: v["status"] for k, v in description["profiles"].items()} == {
        "checked_withdrawal": "default", "contact_latch": "candidate", "conditional_descent": "candidate", "task_release": "candidate"}
    assert [row["module"].rsplit("_", 1)[1] for row in description["profiles"].values()] == ["v4", "v6", "v8", "v2"]
    description["profiles"]["contact_latch"]["status"] = "default"
    assert execution.describe_profiles()["profiles"]["contact_latch"]["status"] == "candidate"


@pytest.mark.parametrize("profile", list(execution.PROFILE_REGISTRY))
def test_real_retained_prepare_freezes_exact_profile_without_execution(campaign, profile):
    source = cases(campaign, 2)
    before = {p: p.read_bytes() for p in [campaign / "budget/limits.json", campaign / "allocation.json"]}
    result = execution.prepare(campaign, "batch", source, profile)
    assert result["requests"] == 2 and not result["reused"]
    assert read(campaign / "batch/cohort.json")["cases"] == source
    assert read(campaign / "batch/protocol.json")["driver"] == execution._profile(profile)[1]["driver"]
    frozen = read(campaign / "batch/scene_execution.json")
    assert frozen["frozen_inputs"] == binding(campaign / "batch/frozen_inputs.json")
    status = execution.campaign_status(campaign)
    assert status["physical_attempt_limit"] == 11
    assert status["budget"]["timeout_seconds"] == 37
    assert status["allocated_requests"] == status["prepared_unsubmitted"] == 2
    assert status["remaining_unallocated"] == 9 and status["remaining_submission_allowance"] == 11
    assert status["submitted_attempts"] == 0
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not list(campaign.rglob("reservation.json"))


def test_idempotent_name_and_reject_rename_profile_or_input_conflicts(campaign):
    source = cases(campaign, 2)
    execution.prepare(campaign, "batch", source)
    frozen_before = {p: p.read_bytes() for p in (campaign / "batch").glob("*.json")}
    assert execution.prepare(campaign, "batch", source)["reused"]
    assert all(p.read_bytes() == contents for p, contents in frozen_before.items())
    with pytest.raises(ValueError, match="name_input_or_profile_conflict"):
        execution.prepare(campaign, "batch", source, "contact_latch")
    changed = deepcopy(source)
    changed[0]["partition"] = "changed"
    with pytest.raises(ValueError, match="name_input_or_profile_conflict"):
        execution.prepare(campaign, "batch", changed)
    with pytest.raises(ValueError, match="request_already_allocated"):
        execution.prepare(campaign, "different_folder", source)
    renamed = deepcopy(source)
    for case in renamed:
        case["id"] += "_renamed"
        case["execution_id"] = case["id"]
    with pytest.raises(ValueError, match="request_already_allocated"):
        execution.prepare(campaign, "different_ids", renamed)
    assert execution.campaign_status(campaign)["allocated_requests"] == 2


def test_legacy_prepared_eight_are_counted_before_submission(campaign):
    driver, _ = execution._profile(execution.DEFAULT_PROFILE)
    driver.prepare_cohort(campaign, "old_prepared", cases(campaign, 8, "old"))
    execution.prepare(campaign, "new_prepared", cases(campaign, 3, "new"))
    status = execution.campaign_status(campaign)
    assert status["allocated_requests"] == 11 and status["submitted_attempts"] == 0
    assert status["prepared_unsubmitted"] == 11 and status["remaining_unallocated"] == 0
    with pytest.raises(ValueError, match="exceed_allocation"):
        execution.prepare(campaign, "over", cases(campaign, 1, "over"))
    assert not (campaign / "over").exists()


def test_explicit_cross_profile_comparison_is_allocated_and_bound(campaign):
    source = cases(campaign, 2)
    execution.prepare(campaign, "baseline", source, "checked_withdrawal")
    execution.prepare(campaign, "candidate", source, "conditional_descent")
    request = read(campaign / ".scene_execution/requests/candidate.json")
    assert request["cross_profile_predecessors"] == [{"cohort": "baseline",
        "case_ids": [c["id"] for c in source],
        "protocol": binding(campaign / "baseline/protocol.json"),
        "driver": execution._profile("checked_withdrawal")[1]["driver"]}]
    assert execution.campaign_status(campaign)["allocated_requests"] == 4
    with pytest.raises(ValueError, match="request_already_allocated"):
        execution.prepare(campaign, "candidate_renamed", source, "conditional_descent")


def test_legacy_same_inputs_resume_read_only(campaign):
    source = cases(campaign, 2)
    driver, _ = execution._profile(execution.DEFAULT_PROFILE)
    driver.prepare_cohort(campaign, "old", source)
    before = {p: p.read_bytes() for p in (campaign / "old").glob("*.json")}
    assert execution.prepare(campaign, "old", source)["reused"]
    assert not (campaign / "old/scene_execution.json").exists()
    assert all(p.read_bytes() == data for p, data in before.items())


def test_original_input_hash_change_is_rejected_before_intent(campaign):
    source = cases(campaign, 1)
    Path(source[0]["task"]["path"]).write_text('{"changed": true}')
    with pytest.raises(ValueError, match="hash changed"):
        execution.prepare(campaign, "changed", source)
    assert not (campaign / ".scene_execution").exists()


def test_copied_campaign_cannot_reset_allocation(campaign):
    copied = campaign.parent / "renamed_campaign"
    shutil.copytree(campaign, copied)
    with pytest.raises(ValueError, match="allocation_root_budget_or_limit_mismatch"):
        execution.campaign_status(copied)


def test_receipt_limit_must_equal_bound_budget(campaign):
    registry = execution.ALLOCATION_REGISTRY
    receipt = read(registry / "unit_allocation.json")
    receipt["limit"] += 1
    (registry / "unit_allocation.json").write_text(json.dumps(receipt))
    (campaign / "allocation.json").write_text(json.dumps({"registration": binding(registry / "unit_allocation.json")}))
    with pytest.raises(ValueError, match="allocation_root_budget_or_limit_mismatch"):
        execution.campaign_status(campaign)


def test_interrupted_preparation_keeps_allocation_and_never_retries(campaign, monkeypatch):
    source = cases(campaign, 3)
    driver, _ = execution._profile(execution.DEFAULT_PROFILE)
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("interrupt_before_cohort_freeze")
    monkeypatch.setattr(driver, "prepare_cohort", interrupted)
    with pytest.raises(RuntimeError, match="interrupt_before"):
        execution.prepare(campaign, "interrupted", source)
    status = execution.campaign_status(campaign)
    assert status["allocated_requests"] == status["prepared_unsubmitted"] == 3
    assert status["remaining_unallocated"] == 8
    assert status["incomplete_preparations"] == ["interrupted"]
    for name in ["interrupted", "another_folder"]:
        with pytest.raises(ValueError, match="reconcile_without_retry"):
            execution.prepare(campaign, name, source)
    assert calls == [1]


def test_unknown_reservation_stops_all_dispatch_and_new_preparation(campaign, monkeypatch):
    source = cases(campaign, 2)
    execution.prepare(campaign, "batch", source)
    record_submission(campaign, "batch", source[0], terminal=False)
    status = execution.campaign_status(campaign)
    assert status["submitted_attempts"] == status["unknown_or_active_attempts"] == 1
    assert status["prepared_unsubmitted"] == 1
    driver, _ = execution._profile(execution.DEFAULT_PROFILE)
    monkeypatch.setattr(driver, "run", lambda *args, **kwargs: pytest.fail("must not dispatch unknown state"))
    with pytest.raises(ValueError, match="unknown_attempt_recover_no_retry"):
        execution.run(campaign / "batch", 1)
    with pytest.raises(ValueError, match="unknown_attempt_recover_no_retry"):
        execution.prepare(campaign, "new", cases(campaign, 1, "new"))


def test_run_dispatches_frozen_driver_and_retains_serial_limit(campaign, monkeypatch):
    source = cases(campaign, 3)
    execution.prepare(campaign, "batch", source, "contact_latch")
    record_submission(campaign, "batch", source[0])
    calls = []
    driver, _ = execution._profile("contact_latch")
    monkeypatch.setattr(driver, "run", lambda path, limit: calls.append((path, limit)) or {"new_attempts": 0})
    for invalid in [None, True, 0, -1, 4, 3]:
        with pytest.raises(ValueError, match="limit"):
            execution.run(campaign / "batch", invalid)
    assert calls == []
    result = execution.run(campaign / "batch", 2)
    assert calls == [(campaign / "batch", 2)]
    assert result["profile"] == "contact_latch"
    assert execution.DEFAULT_PROFILE == "checked_withdrawal"
    assert len(list((campaign / "budget/reservations").glob("*.json"))) == 1


def test_all_terminal_repeat_does_not_dispatch(campaign, monkeypatch):
    source = cases(campaign, 2)
    execution.prepare(campaign, "batch", source)
    for case in source:
        record_submission(campaign, "batch", case)
    driver, _ = execution._profile(execution.DEFAULT_PROFILE)
    monkeypatch.setattr(driver, "run", lambda *args, **kwargs: pytest.fail("already terminal"))
    assert execution.run(campaign / "batch", 2)["all_attempts_terminal"]
    assert execution.campaign_status(campaign)["terminal_attempts"] == 2


def test_prepared_profile_change_cannot_redirect_run(campaign, monkeypatch):
    execution.prepare(campaign, "batch", cases(campaign, 1))
    replacement = dict(execution.PROFILE_REGISTRY)
    replacement["checked_withdrawal"] = replacement["contact_latch"]
    monkeypatch.setattr(execution, "PROFILE_REGISTRY", replacement)
    with pytest.raises(ValueError, match="frozen_execution_profile"):
        execution.run(campaign / "batch", 1)


def test_ledger_gap_and_missing_local_mirror_are_rejected(campaign):
    source = cases(campaign, 1)
    execution.prepare(campaign, "batch", source)
    record_submission(campaign, "batch", source[0])
    path = campaign / "budget/reservations/0000.json"
    path.rename(path.with_name("0002.json"))
    with pytest.raises(ValueError, match="sequence_gap"):
        execution.campaign_status(campaign)
    path.with_name("0002.json").rename(path)
    (campaign / "batch/executions" / source[0]["id"] / "reservation.json").unlink()
    with pytest.raises(ValueError, match="mirror_mismatch"):
        execution.campaign_status(campaign)


@pytest.mark.parametrize("name", ["../escape", "x/y", ".", "..", "budget", "prepared_inputs"])
def test_names_cannot_escape_or_replace_campaign_metadata(campaign, name):
    with pytest.raises(ValueError):
        execution.prepare(campaign, name, cases(campaign, 1))


def test_unknown_campaign_is_not_registered_as_side_effect(tmp_path):
    root = tmp_path / "new"
    with pytest.raises(ValueError, match="existing_registered_campaign_required"):
        execution.prepare(root, "batch", [])
    assert not root.exists()
