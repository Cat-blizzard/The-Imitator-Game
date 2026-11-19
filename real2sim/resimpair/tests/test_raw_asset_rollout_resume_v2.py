"""Temporary ledgers only; native/GPU source processes are never executed."""
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil
import time

import pytest

from resimpair_grasp.common import binding, digest, read, write
from resimpair_tasks import raw_asset_rollout_resume_v2 as resume


@pytest.fixture
def factory(tmp_path, monkeypatch):
    clock = [100000.]
    real_sleep = time.sleep
    monkeypatch.setattr(resume.time, "time", lambda: clock[0])
    monkeypatch.setattr(resume.time, "sleep", lambda _: real_sleep(.001))
    monkeypatch.setitem(resume.CAPS, "maximum_CPU_sources", 2)
    monkeypatch.setitem(resume.CAPS, "maximum_optimizer_jobs", 2)
    counter = [0]
    def create(*, expired=True):
        counter[0] += 1
        base = tmp_path / str(counter[0]); root = base / "original"; root.mkdir(parents=True)
        (root / "gpu").mkdir()
        for name in (".cpu.lock", ".gpu.lock"): (root / name).touch()
        jobs = []; rows = []
        for sid in ("fixture:first", "fixture:second"):
            path = root / "cpu" / digest(sid)[:24] / "job.json"
            write(path, {"raw_row": {"id": sid}, "output": str(path.parent)})
            jobs.append(str(path)); rows.append({"source_id": sid, "job": binding(path)})
        write(base / "manip.json", {"fixture": True})
        write(root / "population.json", {"sources": rows})
        protocol = {**deepcopy(resume.CAPS), "schema": resume.retained.VERSION,
            "population": binding(root / "population.json"), "job_paths": jobs,
            "qualification_job_paths": jobs, "implementation_bindings": [binding(resume.retained.__file__)],
            "solver_configuration": [], "manip": binding(base / "manip.json")}
        write(root / "protocol.json", protocol)
        original_end = clock[0] - 10 if expired else clock[0] + 10
        write(root / "START.json", {"unix": original_end - 43200, "deadline_unix": original_end})
        write(base / "independent/FINAL_REVIEW.json", {"fixture": "independent original review"})
        write(base / "independent/PROTOCOL.json", {"fixture": "independent original protocol"})
        old_gate = {"continue_population": True, "protocol": binding(root / "protocol.json"),
            "independent_review": binding(base / "independent/FINAL_REVIEW.json"),
            "independent_protocol": binding(base / "independent/PROTOCOL.json"), "frozen_transitive_dependencies": []}
        write(root / "continuation_gate.json", old_gate); write(base / "independent/CONTINUATION_GATE.json", old_gate)
        qualification = base / "qualification.json"; resume.audit(root, qualification)
        gate = resume.release_template(qualification); gate.update(passed=True, issued_unix=clock[0])
        release = base / "parent_release.json"; write(release, gate)
        return {"root": root, "protocol": protocol, "scope": resume.frozen_scope(root)[2],
            "gate": gate, "release": release, "clock": clock, "jobs": [Path(path) for path in jobs], "base": base}
    return create


def fake_source_processes(monkeypatch, state, *, failure=None):
    calls = []
    def process(args, folder, stage, timeout, env, pass_fds=()):
        assert stage == "compile", "tests must never query or execute a real GPU job"
        assert timeout <= 240
        assert state["gate"]["not_before_unix"] <= state["clock"][0] < state["gate"]["deadline_unix"]
        folder = Path(folder)
        reservation = read(folder / "cpu_reservation.json")
        assert reservation["deadline_extension"] == binding(state["release"])
        calls.append(str(folder / "job.json"))
        value = {"source_id": read(folder / "job.json")["raw_row"]["id"],
            "status": failure or "geometry_or_policy_failed", "recorded_settle_steps": 0}
        write(folder / "cpu_result.json", value)
        return 0
    monkeypatch.setattr(resume.retained, "_run_process", process)
    monkeypatch.setattr(resume.retained, "disk_ok", lambda *_: None)
    return calls


def test_missing_parent_gate_cannot_start_or_register_extension(factory):
    state = factory(); root = state["root"]; original = binding(root / "START.json")
    with pytest.raises(ValueError, match="parent_time_extension_gate"):
        resume.run(root)
    assert binding(root / "START.json") == original
    assert not (root / "TIME_EXTENSION_GATE.json").exists()
    assert not list(root.glob("cpu/*/cpu_reservation.json"))


@pytest.mark.parametrize("change", ["unpassed", "overlong", "new_sources", "retry", "API", "wrong_code", "wrong_independent_gate", "not_before", "future_issue"])
def test_release_cannot_change_scope_or_absolute_ceiling(factory, change):
    state = factory(); gate = deepcopy(state["gate"])
    if change == "unpassed": gate["passed"] = False
    elif change == "overlong": gate["deadline_unix"] += .001
    elif change == "new_sources": gate["new_source_allocations"] = 1
    elif change == "retry": gate["retries"] = 1
    elif change == "API": gate["API_calls"] = 1
    elif change == "wrong_code": gate["implementation"]["sha256"] = "wrong"
    elif change == "wrong_independent_gate": gate["scope"]["original_independent_gate"]["sha256"] = "wrong"
    elif change == "not_before": gate["not_before_unix"] -= 1
    else: gate["issued_unix"] += 1
    path = state["base"] / "bad_gate.json"; write(path, gate)
    with pytest.raises(ValueError): resume.run(state["root"], path)
    assert not (state["root"] / "TIME_EXTENSION_GATE.json").exists()


def test_copying_root_cannot_move_source_population_or_deadline(factory):
    state = factory(); copied = state["base"] / "replacement"
    shutil.copytree(state["root"], copied)
    with pytest.raises(ValueError, match="original_root_CPU_job_paths"):
        resume.frozen_scope(copied)


def test_expired_extension_never_registers_a_new_source_or_accepts_another_gate(factory):
    state = factory(); root = state["root"]; start = binding(root / "START.json")
    state["clock"][0] = state["gate"]["deadline_unix"] + 1
    result = resume.run(root, state["release"])
    assert result["state"] == "extension_expired_without_admission"
    assert not (root / "TIME_EXTENSION_ADMISSION.json").exists()
    second = deepcopy(state["gate"]); second["deadline_unix"] -= 100
    path = state["base"] / "second_release.json"; write(path, second)
    with pytest.raises(ValueError, match="gate_cannot_change"): resume.run(root, path)
    assert binding(root / "START.json") == start
    assert not list(root.glob("cpu/*/cpu_reservation.json"))


def test_actual_v1_loops_are_called_once_CPU_first_with_fresh_status(factory, monkeypatch):
    state = factory(); calls = fake_source_processes(monkeypatch, state)
    cpu_impl, gpu_impl = resume.retained.cpu_run, resume.retained.gpu_run
    order = []
    def cpu(root, phase):
        order.append("cpu")
        return cpu_impl(root, phase)
    def gpu(root, phase):
        status = read(root / "cpu_status.json")
        assert status["deadline_extension"] == binding(state["release"])
        assert status["extension_session"]
        assert status["state"] in {"running", "completed_scope"}
        order.append("gpu")
        return gpu_impl(root, phase)
    monkeypatch.setattr(resume.retained, "cpu_run", cpu); monkeypatch.setattr(resume.retained, "gpu_run", gpu)
    old_start = binding(state["root"] / "START.json")
    result = resume.run(state["root"], state["release"])
    assert result["state"] == "completed_original_scope"
    assert order == ["cpu", "gpu"] and sorted(calls) == sorted(map(str, state["jobs"]))
    assert binding(state["root"] / "START.json") == old_start
    assert resume.retained.deadline is not None
    assert resume.retained.fcntl is fcntl
    handles = resume.acquire_original_locks(state["root"])
    assert handles is not None
    for handle in handles.values(): handle.close()


def test_original_window_must_expire_before_any_delegation(factory, monkeypatch):
    state = factory(expired=False); calls = fake_source_processes(monkeypatch, state)
    probes = []; actual = resume.acquire_original_locks
    def probe(root):
        probes.append(state["clock"][0]); assert state["clock"][0] >= state["gate"]["not_before_unix"]
        return actual(root)
    def advance(seconds):
        assert not calls
        state["clock"][0] += seconds
        if state["clock"][0] >= state["gate"]["not_before_unix"]:
            monkeypatch.setattr(resume.time, "sleep", lambda _: None)
    monkeypatch.setattr(resume, "acquire_original_locks", probe)
    monkeypatch.setattr(resume.time, "sleep", advance)
    result = resume.run(state["root"], state["release"])
    assert result["state"] == "completed_original_scope" and len(calls) == 2 and probes


def test_busy_original_supervisor_is_observed_then_released_to_v1(factory, monkeypatch):
    state = factory(); calls = fake_source_processes(monkeypatch, state)
    busy = (state["root"] / ".cpu.lock").open("r")
    fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
    real_sleep = time.sleep
    released = []
    def release_on_wait(_):
        assert not calls
        busy.close(); released.append(True)
        monkeypatch.setattr(resume.time, "sleep", real_sleep)
    monkeypatch.setattr(resume.time, "sleep", release_on_wait)
    result = resume.run(state["root"], state["release"])
    assert released == [True] and result["state"] == "completed_original_scope" and len(calls) == 2


@pytest.mark.parametrize("terminal", [False, True])
def test_unknown_CPU_state_never_reexecutes_or_creates_admission(factory, monkeypatch, terminal):
    state = factory(); calls = fake_source_processes(monkeypatch, state); path = state["jobs"][0]
    write(path.parent / "cpu_reservation.json", {"job": binding(path), "protocol": state["scope"]["protocol"], "attempt": 1, "unix": 99980.})
    if terminal: write(path.parent / "cpu_terminal.json", {"status": "unknown_system_failure"})
    with pytest.raises(RuntimeError, match="unknown_or_live_original_state"):
        resume.run(state["root"], state["release"])
    assert calls == [] and not (state["root"] / "TIME_EXTENSION_ADMISSION.json").exists()
    assert not (state["jobs"][1].parent / "cpu_reservation.json").exists()


def test_saved_CPU_result_terminalizes_without_second_attempt(factory, monkeypatch):
    state = factory(); calls = fake_source_processes(monkeypatch, state); path = state["jobs"][0]
    write(path.parent / "cpu_reservation.json", {"job": binding(path), "protocol": state["scope"]["protocol"], "attempt": 1, "unix": 99980.})
    write(path.parent / "cpu_result.json", {"source_id": "fixture:first", "status": "geometry_or_policy_failed"})
    old = binding(path.parent / "cpu_reservation.json"); old_result = binding(path.parent / "cpu_result.json")
    result = resume.run(state["root"], state["release"])
    assert result["state"] == "completed_original_scope" and calls == [str(state["jobs"][1])]
    assert binding(path.parent / "cpu_reservation.json") == old and binding(path.parent / "cpu_result.json") == old_result
    assert (path.parent / "cpu_terminal.json").exists()


def context_fixture(state):
    launch = state["base"] / "test_launch.json"; write(launch, {"fixture": "no execution"})
    return resume.delegated_context(state["root"], state["protocol"], state["scope"],
        binding(state["release"]), state["gate"], binding(launch))


def test_reservation_hook_binds_one_gate_and_never_rewrites_existing_attempt(factory):
    state = factory(); folder = state["jobs"][0].parent
    old_write, old_deadline = resume.retained.write, resume.retained.deadline
    value = {"job": binding(folder / "job.json"), "protocol": state["scope"]["protocol"], "attempt": 1, "unix": state["clock"][0]}
    with context_fixture(state):
        resume.retained.write(folder / "cpu_reservation.json", value)
        with pytest.raises(RuntimeError, match="never_be_rewritten"):
            resume.retained.write(folder / "cpu_reservation.json", value)
    recorded = read(folder / "cpu_reservation.json")
    assert recorded["deadline_extension"] == binding(state["release"]) and "deadline_extension" not in value
    assert resume.retained.write is old_write and resume.retained.deadline is old_deadline


@pytest.mark.parametrize("outside", ["before", "after"])
def test_reservation_hook_enforces_absolute_window(factory, outside):
    state = factory(); folder = state["jobs"][0].parent
    state["clock"][0] = state["gate"]["not_before_unix"] - 1 if outside == "before" else state["gate"]["deadline_unix"]
    value = {"job": binding(folder / "job.json"), "protocol": state["scope"]["protocol"], "attempt": 1, "unix": state["clock"][0]}
    with context_fixture(state), pytest.raises(RuntimeError, match="absolute_extension_admission_window"):
        resume.retained.write(folder / "cpu_reservation.json", value)
    assert not (folder / "cpu_reservation.json").exists()


def test_gpu_reservation_is_bound_and_original_optimizer_never_reruns(factory, monkeypatch):
    state = factory(); calls = fake_source_processes(monkeypatch, state)
    folder = state["root"] / "gpu/test"
    write(folder / "job.json", {"fixture": "GPU binding test only"})
    value = {"job": binding(folder / "job.json"), "protocol": state["scope"]["protocol"], "attempt": 1, "unix": 99980.}
    write(folder / "reservation.json", value)
    with context_fixture(state), pytest.raises(RuntimeError, match="optimizer_attempt_never_reexecuted"):
        resume.retained._run_process([], folder, "worker", 120, {})
    assert calls == []


def test_failed_or_unknown_export_never_repeats(factory, monkeypatch):
    state = factory(); calls = fake_source_processes(monkeypatch, state); folder = state["root"] / "gpu/test"
    write(folder / "export_process.json", {"fixture": "previous unknown export"})
    with context_fixture(state), pytest.raises(RuntimeError, match="previous_export_requires_recovery"):
        resume.retained._run_process([], folder, "export", 120, {})
    assert calls == []


def test_stale_cpu_status_cannot_release_GPU(factory):
    state = factory(); launch = state["base"] / "launch.json"; write(launch, {"fixture": True})
    value = {"state": "running", "phase_source_count": 2, "unix": 100001., "counts": {}}
    resume.retained.atomic(state["root"] / "cpu_status.json", value)
    assert resume.fresh_cpu_status(state["root"], binding(state["release"]), binding(launch), 100000.) is None
    value.update(deadline_extension=binding(state["release"]), extension_session=binding(launch), unix=99999.)
    resume.retained.atomic(state["root"] / "cpu_status.json", value)
    assert resume.fresh_cpu_status(state["root"], binding(state["release"]), binding(launch), 100000.) is None


def test_CPU_exception_is_not_hidden_by_fresh_status_or_other_stage(factory, monkeypatch):
    state = factory(); gpu_calls = []
    def failed_cpu(root, phase):
        resume.retained.atomic(root / "cpu_status.json", {"state": "running", "phase_source_count": 2, "unix": state["clock"][0], "counts": {}})
        raise RuntimeError("fixture_CPU_failure")
    monkeypatch.setattr(resume.retained, "cpu_run", failed_cpu)
    monkeypatch.setattr(resume.retained, "gpu_run", lambda *_: gpu_calls.append(True))
    with pytest.raises(ExceptionGroup, match="delegated_original_queue_failure") as error:
        resume.run(state["root"], state["release"])
    assert any("fixture_CPU_failure" in str(item) for item in error.value.exceptions)
    assert (state["root"] / "STOP").exists()
    result = read(next((state["root"] / "time_extension_v2/sessions").glob("*/RESULT.json")))
    assert result["state"] == "stopped_or_deadline_or_recovery"
    assert not list(state["root"].glob("cpu/*/cpu_terminal.json"))


def test_GPU_exception_propagates_after_real_CPU_loop_drains(factory, monkeypatch):
    state = factory(); calls = fake_source_processes(monkeypatch, state)
    def failed_gpu(*_): raise RuntimeError("fixture_GPU_failure")
    monkeypatch.setattr(resume.retained, "gpu_run", failed_gpu)
    with pytest.raises(ExceptionGroup) as error: resume.run(state["root"], state["release"])
    assert any("fixture_GPU_failure" in str(item) for item in error.value.exceptions)
    assert len(calls) == 2 and len(list(state["root"].glob("cpu/*/cpu_terminal.json"))) == 2
    assert not list(state["root"].glob("gpu/*/terminal.json"))


def test_unknown_source_is_preserved_and_reported_as_failure(factory, monkeypatch):
    state = factory(); fake_source_processes(monkeypatch, state, failure="unknown_system_failure")
    with pytest.raises(ExceptionGroup): resume.run(state["root"], state["release"])
    assert (state["root"] / "STOP").exists()
    result = read(next((state["root"] / "time_extension_v2/sessions").glob("*/RESULT.json")))
    final = read(Path(resume.resolve(result["final_ledger"])))
    assert final["unknown"] and result["state"] == "stopped_or_deadline_or_recovery"


def test_deadline_stop_is_not_reported_complete_even_with_zero_prepared_GPU_jobs(factory, monkeypatch):
    state = factory(); fake_source_processes(monkeypatch, state)
    monkeypatch.setattr(resume.retained, "gpu_run", lambda *_: {"state": "stopped_needs_recovery_or_deadline"})
    result = resume.run(state["root"], state["release"])
    assert result["state"] == "stopped_or_deadline_or_recovery"


def test_v1_CPU_worker_future_exception_receipt_stops_and_propagates(factory, monkeypatch):
    state = factory(); calls = fake_source_processes(monkeypatch, state)
    def failed_disk(*_): raise RuntimeError("fixture_disk_reserve_failure")
    monkeypatch.setattr(resume.retained, "disk_ok", failed_disk)
    with pytest.raises(ExceptionGroup) as error: resume.run(state["root"], state["release"])
    assert any("fixture_disk_reserve_failure" in str(item) for item in error.value.exceptions)
    assert not calls and (state["root"] / "STOP").exists()
    assert "fixture_disk_reserve_failure" in read(state["root"] / "cpu_supervisor_error.json")["error"]
    assert not list(state["root"].glob("cpu/*/cpu_terminal.json"))


def test_v1_GPU_worker_future_exception_receipt_cannot_hide_behind_CPU_success(factory, monkeypatch):
    state = factory(); fake_source_processes(monkeypatch, state)
    def failed_gpu(root, _):
        resume.retained.write(root / "gpu_supervisor_error.json", {"error": "fixture_GPU_worker_exception"})
        return {"state": "stopped_needs_recovery_or_deadline"}
    monkeypatch.setattr(resume.retained, "gpu_run", failed_gpu)
    with pytest.raises(ExceptionGroup) as error: resume.run(state["root"], state["release"])
    assert any("fixture_GPU_worker_exception" in str(item) for item in error.value.exceptions)
    assert (state["root"] / "STOP").exists()
    result = read(next((state["root"] / "time_extension_v2/sessions").glob("*/RESULT.json")))
    assert result["stages"]["cpu"]["exception"] is None
    assert result["state"] == "stopped_or_deadline_or_recovery"
