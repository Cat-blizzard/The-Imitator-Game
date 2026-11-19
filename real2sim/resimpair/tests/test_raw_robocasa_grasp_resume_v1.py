"""Deadline admission tests; all ledgers are temporary and no GPU is queried."""
import fcntl
from pathlib import Path
import time

import pytest

from resimpair_assets import raw_robocasa_grasp_resume_v1 as resume
from resimpair_grasp.common import binding, read, write


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "original"; root.mkdir()
    registry = tmp_path / "registry"
    specs = []
    for index in range(4):
        folder = root / "gpu" / str(index)
        write(folder / "job.json", {"source_id": str(index)})
        write(registry / "claims" / (str(index) + ".json"), {"original_job": binding(folder / "job.json")})
        specs.append({"source_id": str(index), "output": str(folder), "allocation_key": str(index)})
    protocol = {"maximum_wall_seconds": 3600, "gpu_index": 3, "registry": str(registry), "job_specs": specs,
        "minimum_free_GPU_GiB": 12, "GPU_timeout_seconds": 120, "export_timeout_seconds": 120}
    write(root / "protocol.json", protocol)
    now = time.time()
    write(root / "START.json", {"unix": now - 7200, "deadline_unix": now - 3600})
    write(tmp_path / "original_release.json", {"fixture": "parent original four-job release"})
    monkeypatch.setattr(resume.bridge, "validate", lambda _: protocol)
    monkeypatch.setattr(resume.bridge, "require_release", lambda _: binding(tmp_path / "original_release.json"))
    monkeypatch.setattr(resume, "sufficient_disk", lambda _: 100 * 2**30)
    monkeypatch.setattr(resume.bridge.retained_queue, "gpu_refresh", lambda _, state: {"state": state})
    def forbid_gpu(*args, **kwargs):
        pytest.fail("test must not run or query GPU")
    monkeypatch.setattr(resume.bridge, "_run_process", forbid_gpu)
    monkeypatch.setattr(resume.subprocess, "check_output", forbid_gpu)
    audit = tmp_path / "audit.json"; resume.audit(root, audit)
    issued = time.time()
    gate = {"schema": resume.RELEASE_SCHEMA, "reviewer_role": "parent", "passed": True,
        "scope": resume.frozen_scope(root, protocol), "implementation": binding(resume.__file__),
        "authorized_existing_GPU_jobs": 4, "authorized_new_source_claims": 0, "retries": 0,
        "maximum_additional_wall_seconds": 3600, "interruption_audit": binding(audit),
        "issued_unix": issued, "deadline_unix": issued + 3600}
    return root, protocol, gate


def test_original_expired_START_does_not_authorize_extension(prepared):
    root, _, _ = prepared; old_start = binding(root / "START.json")
    with pytest.raises(ValueError, match="explicit_parent_deadline_extension"):
        resume.run(root)
    assert binding(root / "START.json") == old_start
    assert not (root / "DEADLINE_RESUME_ADMISSION.json").exists()
    assert not list(root.glob("gpu/*/reservation.json"))


@pytest.mark.parametrize("change", ["unpassed", "new_claim", "retry", "overlong", "future", "wrong_job"])
def test_gate_cannot_expand_scope_or_renew_unbounded_time(prepared, change, tmp_path):
    root, _, gate = prepared
    if change == "unpassed": gate["passed"] = False
    elif change == "new_claim": gate["authorized_new_source_claims"] = 1
    elif change == "retry": gate["retries"] = 1
    elif change == "overlong": gate["deadline_unix"] += 1
    elif change == "future":
        gate["issued_unix"] += 86400; gate["deadline_unix"] += 86400
    else: gate["scope"]["jobs"][0]["source_id"] = "replacement"
    release = tmp_path / "bad_gate.json"; write(release, gate)
    with pytest.raises(ValueError): resume.run(root, release)
    assert not (root / "DEADLINE_RESUME_ADMISSION.json").exists()
    assert not list(root.glob("gpu/*/reservation.json"))


def test_first_admission_rechecks_zero_execution_after_audit(prepared, tmp_path):
    root, _, gate = prepared
    write(root / "gpu/0/reservation.json", {"unknown": True})
    release = tmp_path / "gate.json"; write(release, gate)
    with pytest.raises(ValueError, match="zero_submission_state"):
        resume.run(root, release)
    assert not (root / "DEADLINE_RESUME_ADMISSION.json").exists()


def test_contention_reuses_gate_without_creating_attempt_or_changing_START(prepared, tmp_path, monkeypatch):
    root, _, gate = prepared; old_start = binding(root / "START.json")
    release = tmp_path / "gate.json"; write(release, gate)
    resource_lock = tmp_path / "shared_gpu.lock"
    monkeypatch.setattr(resume.bridge.resources, "lock_paths", lambda _: [resource_lock])
    with resource_lock.open("a+") as busy:
        fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert resume.run(root, release)["state"] == "shared_GPU_lock_deferred_no_submission"
        admission = binding(root / "DEADLINE_RESUME_ADMISSION.json")
        assert resume.run(root)["state"] == "shared_GPU_lock_deferred_no_submission"
    assert binding(root / "DEADLINE_RESUME_ADMISSION.json") == admission
    assert binding(root / "START.json") == old_start
    assert not list(root.glob("gpu/*/reservation.json"))


def test_expired_extension_does_not_submit_and_cannot_change_gate(prepared, tmp_path, monkeypatch):
    root, protocol, gate = prepared; old_start = binding(root / "START.json")
    release = tmp_path / "gate.json"; write(release, gate)
    monkeypatch.setattr(resume.time, "time", lambda: gate["deadline_unix"] + 1)
    assert resume.run(root, release)["state"] == "extension_deadline_no_new_submission"
    replacement = {**gate, "issued_unix": gate["deadline_unix"], "deadline_unix": gate["deadline_unix"] + 3600}
    changed = tmp_path / "second_gate.json"; write(changed, replacement)
    with pytest.raises(ValueError, match="gate_cannot_change"):
        resume.run(root, changed)
    assert binding(root / "START.json") == old_start
    assert not list(root.glob("gpu/*/reservation.json"))


def test_interrupted_reservation_never_reruns_optimizer(prepared, tmp_path):
    root, protocol, gate = prepared
    release = tmp_path / "gate.json"; write(release, gate)
    with resume.campaign_lock(root): resume.require_extension(root, protocol, release)
    write(root / "gpu/0/reservation.json", {"job": binding(root / "gpu/0/job.json"),
        "protocol": binding(root / "protocol.json"), "attempt": 1, "deadline_extension": binding(release)})
    with pytest.raises(ValueError, match="unknown_GPU_reservation_never_resubmit"):
        resume.run(root)
    assert len(list(root.glob("gpu/*/reservation.json"))) == 1
    assert not (root / "gpu/0/optimizer_result.json").exists()
