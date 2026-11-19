"""One explicit bounded time amendment for the original raw-source queue.

Delegates directly to the unchanged v1 CPU/GPU supervisors. No new source
population, queue implementation, source attempt, retry policy or START reset.
The parent gate fixes one absolute deadline at most eight hours after the
original deadline. Original supervisors acquire their own locks after a
read-only handoff check. GPU starts only after delegated CPU publishes a fresh
status. Queue exceptions propagate after any already started work drains.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback
from unittest.mock import patch

from resimpair_grasp.common import binding, read, resolve, write
from resimpair_tasks import raw_asset_rollout_v1 as retained

SCHEMA = "raw_asset_rollout_deadline_extension_v2"
QUALIFICATION_SCHEMA = "raw_asset_rollout_deadline_extension_readonly_qualification_v2"
MAX_EXTENSION_SECONDS = 8 * 3600
CAPS = {"maximum_CPU_sources": 22344, "maximum_optimizer_jobs": 22344,
    "CPU_parallelism": 8, "GPU_parallelism": 3, "GPU_indices": [1, 2, 3],
    "CPU_timeout_seconds": 240, "GPU_timeout_seconds": 120, "export_timeout_seconds": 120,
    "minimum_free_GPU_GiB": 12, "minimum_free_disk_GiB": 200,
    "maximum_attempts_per_source_and_stage": 1, "retries": 0,
    "maximum_wall_seconds": 43200, "API_calls": 0, "robot_physical_executions": 0}


def frozen_scope(root):
    root, protocol = retained.validate(root)
    if any(protocol.get(key) != value for key, value in CAPS.items()):
        raise ValueError("original_22344_source_resource_attempt_and_API_caps_required")
    paths = protocol["job_paths"]
    if len(paths) != CAPS["maximum_CPU_sources"] or len(set(paths)) != len(paths):
        raise ValueError("original_unique_source_population_required")
    if any(Path(path).name != "job.json" or Path(path).parent.parent != root / "cpu" for path in paths):
        raise ValueError("original_root_CPU_job_paths_cannot_move")
    start = read(root / "START.json")
    if not math.isclose(start["deadline_unix"] - start["unix"], protocol["maximum_wall_seconds"], abs_tol=.001):
        raise ValueError("original_persistent_twelve_hour_START_required")
    retained.selected_paths(root, protocol, "population")
    continuation = read(root / "continuation_gate.json")
    independent_gate = resolve(continuation["independent_review"]).parent / "CONTINUATION_GATE.json"
    if read(independent_gate) != continuation:
        raise ValueError("original_independent_continuation_gate_must_remain_exact")
    for ref in continuation.get("frozen_transitive_dependencies", []) + [continuation["independent_protocol"]]:
        resolve(ref)
    population = read(resolve(protocol["population"]))["sources"]
    if (len(population) != len(paths) or len({row["source_id"] for row in population}) != len(paths)
            or {str(resolve_path(row["job"]["path"])) for row in population} != set(paths)):
        raise ValueError("original_registered_population_paths_required")
    scope = {"root": str(root), "protocol": binding(root / "protocol.json"),
        "original_start": binding(root / "START.json"), "population": protocol["population"],
        "original_continuation_gate": binding(root / "continuation_gate.json"),
        "original_independent_gate": binding(independent_gate),
        "original_independent_review": continuation["independent_review"],
        "retained_queue": binding(retained.__file__), "unchanged_caps": {key: protocol[key] for key in CAPS}}
    return root, protocol, scope


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else retained.ROOT / path


def acquire_original_locks(root):
    """Read-open existing lock files; retain both descriptors or neither."""
    handles = {}
    try:
        for stage in ("cpu", "gpu"):
            handle = (Path(root) / ("." + stage + ".lock")).open("r")
            handles[stage] = handle
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except BlockingIOError:
        for handle in handles.values(): handle.close()
        return None
    except BaseException:
        for handle in handles.values(): handle.close()
        raise


def ledger_snapshot(root, protocol):
    """Observe receipts only; never terminalize, export, or submit a source."""
    root = Path(root); counts = {"cpu": Counter(), "gpu": Counter()}
    reservations = Counter(); recoverable = Counter(); live = []; unknown = []
    paths_by_stage = {"cpu": [Path(path) for path in protocol["job_paths"]],
        "gpu": sorted((root / "gpu").glob("*/job.json"))}
    protocol_ref = binding(root / "protocol.json")
    frozen_jobs = {str(resolve_path(row["job"]["path"])): row["job"]
        for row in read(resolve(protocol["population"]))["sources"]}
    for stage, paths in paths_by_stage.items():
        for path in paths:
            folder = path.parent
            terminal = folder / ("cpu_terminal.json" if stage == "cpu" else "terminal.json")
            reservation = folder / ("cpu_reservation.json" if stage == "cpu" else "reservation.json")
            reserved = reservation.exists(); reservations[stage] += int(reserved)
            if reserved:
                reserved_value = read(reservation)
                expected_job = frozen_jobs[str(path)] if stage == "cpu" else binding(path)
                if (reserved_value.get("attempt") != 1 or reserved_value.get("protocol") != protocol_ref
                        or reserved_value.get("job") != expected_job):
                    unknown.append({"stage": stage, "job": str(path), "reason": "original_reservation_binding_or_attempt_changed"})
            if terminal.exists():
                status = read(terminal)["status"]; counts[stage][status] += 1
                if not reserved:
                    unknown.append({"stage": stage, "job": str(path), "reason": "terminal_without_reservation"})
                if status in {"unknown_system_failure", "needs_recovery"}:
                    unknown.append({"stage": stage, "job": str(path), "reason": status})
                continue
            if not reserved:
                evidence_names = (["compile_process.json", "compile.log", "cpu_result.json"] if stage == "cpu"
                    else ["worker_process.json", "worker.log", "optimizer_result.json", "export_process.json", "export.log", "result.json"])
                if any((folder / name).exists() for name in evidence_names):
                    unknown.append({"stage": stage, "job": str(path), "reason": "execution_evidence_without_reservation"})
                counts[stage]["unsubmitted"] += 1
                continue
            receipt_names = ["compile_process.json"] if stage == "cpu" else ["worker_process.json", "export_process.json"]
            processes = [read(folder / name).get("identity") for name in receipt_names if (folder / name).exists()]
            active = [identity for identity in processes if retained.process_alive(identity)]
            if active:
                live.append({"stage": stage, "job": str(path), "identities": active})
                counts[stage]["live_reserved"] += 1
                continue
            try:
                if stage == "cpu" and (folder / "cpu_result.json").exists():
                    result = read(folder / "cpu_result.json")
                    if result["status"] == "unknown_system_failure": raise ValueError("saved_CPU_result_is_unknown")
                    recoverable["CPU_saved_result"] += 1; counts[stage]["saved_result_needs_terminal"] += 1
                elif stage == "gpu" and (folder / "result.json").exists():
                    retained.validate_result(path)
                    recoverable["GPU_saved_result"] += 1; counts[stage]["saved_result_needs_terminal"] += 1
                elif (stage == "gpu" and (folder / "optimizer_result.json").exists()
                        and not (folder / "export_process.json").exists() and not (folder / "export.log").exists()):
                    retained.validate_optimizer(path)
                    recoverable["GPU_saved_optimizer_first_export"] += 1; counts[stage]["saved_optimizer_first_export_pending"] += 1
                else:
                    raise ValueError("unknown_reserved_attempt_never_resubmit")
            except Exception as error:
                counts[stage]["unknown_reserved"] += 1
                unknown.append({"stage": stage, "job": str(path), "reason": str(error)})
    errors = [binding(root / name) for name in ("cpu_supervisor_error.json", "gpu_supervisor_error.json") if (root / name).exists()]
    return {"unix": time.time(), "registered_CPU_jobs": len(paths_by_stage["cpu"]),
        "registered_GPU_jobs": len(paths_by_stage["gpu"]), "counts": {key: dict(value) for key, value in counts.items()},
        "reservations": dict(reservations), "recoverable_without_optimizer_or_CPU_retry": dict(recoverable),
        "live_reserved_processes": live, "unknown": unknown, "supervisor_errors": errors,
        "STOP_exists": (root / "STOP").exists(), "source_attempts_started_by_observer": 0}


def audit(root, output):
    root, protocol, scope = frozen_scope(root)
    handles = acquire_original_locks(root)
    try:
        snapshot = ledger_snapshot(root, protocol)
    finally:
        if handles:
            for handle in handles.values(): handle.close()
    original_end = read(resolve(scope["original_start"]))["deadline_unix"]
    value = {"schema": QUALIFICATION_SCHEMA, "scope": scope, "implementation": binding(__file__),
        "frozen_scope_valid": True, "original_deadline_unix": original_end,
        "maximum_extension_deadline_unix": original_end + MAX_EXTENSION_SECONDS,
        "original_window_expired_at_observation": time.time() >= original_end,
        "original_supervisor_locks_free_at_observation": handles is not None,
        "ledger_observation": snapshot,
        "observation_is_stable_handoff": handles is not None,
        "admission_requires_fresh_original_lock_and_ledger_recheck": True, "new_source_attempts": 0,
        "original_independent_scope_text_preserved": True}
    write(output, value)
    return value


def release_template(qualification):
    evidence = read(qualification)
    if evidence["schema"] != QUALIFICATION_SCHEMA or evidence["implementation"] != binding(__file__):
        raise ValueError("current_readonly_qualification_required")
    return {"schema": SCHEMA, "reviewer_role": "parent", "passed": False,
        "scope": evidence["scope"], "implementation": binding(__file__),
        "qualification": binding(qualification), "issued_unix": None,
        "not_before_unix": evidence["original_deadline_unix"],
        "deadline_unix": evidence["maximum_extension_deadline_unix"],
        "maximum_additional_wall_seconds": MAX_EXTENSION_SECONDS,
        "new_source_allocations": 0, "retries": 0, "API_calls": 0, "robot_physical_executions": 0,
        "amendment": "Only the absolute first-attempt scheduling deadline changes; the original independent twelve-hour scope remains recorded and is not rewritten."}


def require_release(root, scope, release_path=None):
    root = Path(root); registered = root / "TIME_EXTENSION_GATE.json"
    admission = root / "TIME_EXTENSION_ADMISSION.json"
    previous = registered if registered.exists() else admission if admission.exists() else None
    reference = read(previous)["gate"] if previous else binding(release_path) if release_path else None
    if reference is None: raise ValueError("explicit_parent_time_extension_gate_required")
    if release_path and reference != binding(release_path): raise ValueError("one_time_extension_gate_cannot_change")
    gate = read(resolve(reference)); original_end = read(resolve(scope["original_start"]))["deadline_unix"]
    if (gate.get("schema") != SCHEMA or gate.get("reviewer_role") != "parent" or gate.get("passed") is not True
            or gate.get("scope") != scope or gate.get("implementation") != binding(__file__)
            or gate.get("maximum_additional_wall_seconds") != MAX_EXTENSION_SECONDS
            or any(gate.get(key) != 0 for key in ("new_source_allocations", "retries", "API_calls", "robot_physical_executions"))):
        raise ValueError("exact_original_scope_and_parent_time_amendment_required")
    evidence = read(resolve(gate["qualification"]))
    if (evidence.get("schema") != QUALIFICATION_SCHEMA or evidence.get("scope") != scope
            or evidence.get("implementation") != binding(__file__) or evidence.get("frozen_scope_valid") is not True):
        raise ValueError("bound_readonly_qualification_required")
    issued, deadline = gate.get("issued_unix"), gate.get("deadline_unix")
    if (type(issued) not in (int, float) or type(deadline) not in (int, float)
            or not math.isfinite(issued) or not math.isfinite(deadline)
            or issued > time.time() or issued < evidence["ledger_observation"]["unix"]
            or gate.get("not_before_unix") != original_end or not original_end < deadline <= original_end + MAX_EXTENSION_SECONDS):
        raise ValueError("finite_absolute_deadline_at_most_original_plus_eight_hours_required")
    if registered.exists() and read(registered) != {"gate": reference, "scope": scope}:
        raise ValueError("original_extension_registration_changed")
    if admission.exists() and read(admission) != {"gate": reference, "scope": scope,
            "original_START_preserved": True, "original_window_expired": True}:
        raise ValueError("original_extension_admission_changed")
    return reference, gate


@contextmanager
def delegated_context(root, protocol, scope, gate_ref, gate, launch_ref):
    """Process-local hooks; all queue and worker algorithms remain v1."""
    root = Path(root); original_write = retained.write; original_atomic = retained.atomic
    original_process = retained._run_process
    stop_mutex = threading.Lock(); end = gate["deadline_unix"]
    def stop(reason):
        with stop_mutex:
            path = root / "STOP"
            if not path.exists():
                write(path, {"schema": SCHEMA, "reason": reason, "unix": time.time(),
                    "deadline_extension": gate_ref, "extension_session": launch_ref,
                    "automatic_removal_allowed": False})
    def effective_deadline(request_root, request_protocol):
        if Path(request_root).resolve() != root or request_protocol != protocol:
            raise ValueError("deadline_hook_only_for_original_frozen_queue")
        resolve(scope["original_start"])
        return end
    def accountable_write(path, value):
        path = Path(path)
        is_cpu = path.name == "cpu_reservation.json" and path.parent.parent == root / "cpu"
        is_gpu = path.name == "reservation.json" and path.parent.parent == root / "gpu"
        if is_cpu or is_gpu:
            if path.exists(): raise RuntimeError("existing_reservation_must_never_be_rewritten")
            if (root / "STOP").exists(): raise RuntimeError("extension_admission_stopped")
            now = time.time()
            if not gate["not_before_unix"] <= now < end: raise RuntimeError("outside_absolute_extension_admission_window")
            if value.get("protocol") != scope["protocol"] or value.get("attempt") != 1:
                raise ValueError("original_protocol_first_attempt_reservation_required")
            if value.get("job") != binding(path.parent / "job.json"):
                raise ValueError("reservation_job_binding_required")
            value = {**value, "deadline_extension": gate_ref, "extension_session": launch_ref}
        result = original_write(path, value)
        # v1 deliberately records some worker-future exceptions instead of
        # re-raising them. Stop both admission loops and preserve that receipt.
        if path in {root / "cpu_supervisor_error.json", root / "gpu_supervisor_error.json"}:
            stop("delegated_worker_exception:" + str(path))
        return result
    def accountable_atomic(path, value):
        path = Path(path)
        source_terminal = ((path.name == "cpu_terminal.json" and path.parent.parent == root / "cpu")
            or (path.name == "terminal.json" and path.parent.parent == root / "gpu"))
        if source_terminal and path.exists(): raise RuntimeError("preserve_existing_source_terminal")
        if path in {root / "cpu_status.json", root / "gpu_status.json"}:
            value = {**value, "deadline_extension": gate_ref, "extension_session": launch_ref,
                "extension_supervisor": retained.process_identity(os.getpid())}
        result = original_atomic(path, value)
        if source_terminal and value.get("status") in {"unknown_system_failure", "needs_recovery"}:
            stop("observed_unknown_source_terminal:" + str(path))
        return result
    def bounded_process(args, folder, stage, timeout, env, pass_fds=()):
        folder = Path(folder)
        if stage in {"compile", "worker"}:
            reservation = read(folder / ("cpu_reservation.json" if stage == "compile" else "reservation.json"))
            if reservation.get("deadline_extension") != gate_ref:
                raise RuntimeError("original_reserved_CPU_or_optimizer_attempt_never_reexecuted")
        if stage == "export":
            if (root / "STOP").exists(): raise RuntimeError("extension_admission_stopped_before_export")
            if (folder / "export_process.json").exists() or (folder / "export.log").exists():
                raise RuntimeError("previous_export_requires_recovery_not_retry")
            if time.time() >= end: raise RuntimeError("extension_deadline_before_first_saved_optimizer_export")
        # An already reserved worker drains under the retained one-second floor.
        return original_process(args, folder, stage, min(timeout, max(1, end - time.time())), env, pass_fds)
    with patch.object(retained, "deadline", effective_deadline), patch.object(retained, "write", accountable_write), \
            patch.object(retained, "atomic", accountable_atomic), \
            patch.object(retained, "_run_process", bounded_process):
        yield stop


def fresh_cpu_status(root, gate_ref, launch_ref, launched_unix):
    path = Path(root) / "cpu_status.json"
    if not path.exists(): return None
    value = read(path)
    if (value.get("deadline_extension") != gate_ref or value.get("extension_session") != launch_ref
            or value.get("unix", 0) < launched_unix or value.get("phase_source_count") != CAPS["maximum_CPU_sources"]):
        return None
    if value.get("state") not in {"running", "completed_scope"} or value.get("counts", {}).get("unknown_system_failure", 0):
        return None
    return value


def run(root, release_path=None):
    root, protocol, scope = frozen_scope(root); gate_ref, gate = require_release(root, scope, release_path)
    with (root / ".time_extension_v2.lock").open("a+") as supervisor:
        fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "TIME_EXTENSION_GATE.json", {"gate": gate_ref, "scope": scope})
        session = root / "time_extension_v2" / "sessions" / str(time.time_ns())
        launch = {"schema": SCHEMA, "gate": gate_ref, "scope": scope,
            "identity": retained.process_identity(os.getpid()), "unix": time.time(), "implementation": binding(__file__)}
        write(session / "LAUNCH.json", launch); launch_ref = binding(session / "LAUNCH.json")
        handles = None; last_message = 0
        while time.time() < gate["deadline_unix"]:
            if (root / "STOP").exists(): raise RuntimeError("existing_STOP_requires_explicit_recovery")
            if time.time() >= gate["not_before_unix"]:
                handles = acquire_original_locks(root)
                if handles is not None: break
            if time.monotonic() - last_message >= 30:
                print(json.dumps({"state": "waiting_for_original_deadline_and_supervisors", "unix": time.time(),
                    "original_deadline_unix": gate["not_before_unix"], "extension_deadline_unix": gate["deadline_unix"]}), flush=True)
                last_message = time.monotonic()
            time.sleep(min(5, max(.01, gate["deadline_unix"] - time.time())))
        if handles is None:
            value = {"state": "extension_expired_without_admission", "new_attempts": 0, "session": launch_ref}
            write(session / "RESULT.json", value); return value
        original_locks_free_unix = time.time()
        for handle in handles.values(): handle.close()
        handles = None
        try:
            _, current, current_scope = frozen_scope(root)
            if current != protocol or current_scope != scope: raise ValueError("frozen_scope_changed_before_handoff")
            snapshot = ledger_snapshot(root, protocol)
            snapshot["original_supervisor_locks_confirmed_free_unix"] = original_locks_free_unix
            snapshot["original_locks_released_for_v1_to_acquire"] = True
            write(session / "HANDOFF_LEDGER.json", snapshot)
            if snapshot["unknown"] or snapshot["live_reserved_processes"] or snapshot["supervisor_errors"] or snapshot["STOP_exists"]:
                raise RuntimeError("unknown_or_live_original_state_requires_recovery_before_extension")
            if time.time() >= gate["deadline_unix"]: raise RuntimeError("extension_expired_during_handoff_no_new_submission")
            write(root / "TIME_EXTENSION_ADMISSION.json", {"gate": gate_ref, "scope": scope,
                "original_START_preserved": True, "original_window_expired": True})
            with delegated_context(root, protocol, scope, gate_ref, gate, launch_ref) as stop:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    cpu_started = time.time(); cpu = pool.submit(retained.cpu_run, root, "population")
                    gpu = None; ready = None
                    while time.time() < gate["deadline_unix"]:
                        if (root / "STOP").exists() or (cpu.done() and cpu.exception() is not None): break
                        ready = fresh_cpu_status(root, gate_ref, launch_ref, cpu_started)
                        if ready is not None:
                            write(session / "FRESH_CPU_STATUS.json", ready)
                            gpu = pool.submit(retained.gpu_run, root, "population"); break
                        if cpu.done(): break
                        time.sleep(2)
                    if gpu is None:
                        stop("no_fresh_delegated_CPU_status_before_GPU_admission")
                    stages = {}; exceptions = []; futures = {"cpu": cpu, **({"gpu": gpu} if gpu is not None else {})}
                    while len(stages) < len(futures):
                        for stage, future in futures.items():
                            if stage in stages or not future.done(): continue
                            try:
                                stages[stage] = {"returned": future.result(), "exception": None}
                            except Exception as error:
                                exceptions.append(error)
                                stages[stage] = {"returned": None, "exception": repr(error), "traceback": traceback.format_exc()}
                                stop("delegated_" + stage + "_supervisor_exception")
                            write(session / (stage.upper() + "_SUPERVISOR_EXIT.json"), stages[stage])
                        if len(stages) < len(futures): time.sleep(2)
            final = ledger_snapshot(root, protocol); write(session / "FINAL_LEDGER.json", final)
            cpu_done = sum(value for key, value in final["counts"]["cpu"].items()
                if key not in {"unsubmitted", "live_reserved", "saved_result_needs_terminal", "unknown_reserved"})
            gpu_pending = sum(final["counts"]["gpu"].get(key, 0) for key in
                {"unsubmitted", "live_reserved", "saved_result_needs_terminal", "saved_optimizer_first_export_pending", "unknown_reserved"})
            complete = (cpu_done == CAPS["maximum_CPU_sources"] and not gpu_pending and not final["unknown"]
                and not final["supervisor_errors"] and not final["STOP_exists"]
                and not final["live_reserved_processes"] and gpu is not None
                and final["registered_GPU_jobs"] == final["counts"]["cpu"].get("native_ready", 0)
                and all(value["exception"] is None for value in stages.values())
                and stages["cpu"]["returned"].get("state") == "completed_scope"
                and stages["gpu"]["returned"].get("state") == "completed_CPU_scope")
            value = {"state": "completed_original_scope" if complete else "stopped_or_deadline_or_recovery",
                "session": launch_ref, "gate": gate_ref, "stages": stages, "final_ledger": binding(session / "FINAL_LEDGER.json"),
                "original_START": scope["original_start"], "original_START_unchanged": binding(root / "START.json") == scope["original_start"],
                "registered_source_scope": CAPS["maximum_CPU_sources"], "unsubmitted_or_unknown_preserved": True,
                "source_retries": 0, "API_calls": 0, "robot_physical_executions": 0}
            write(session / "RESULT.json", value)
            if final["unknown"]:
                exceptions.append(RuntimeError("unknown_source_outcomes_preserved_see_FINAL_LEDGER"))
            for ref in final["supervisor_errors"]:
                exceptions.append(RuntimeError("retained_worker_exception_receipt:" + str(resolve(ref))
                    + ":" + str(read(resolve(ref)).get("error"))))
            if exceptions:
                raise ExceptionGroup("delegated_original_queue_failure_after_drain", exceptions)
            return value
        except BaseException as error:
            write(session / "SUPERVISOR_EXCEPTION.json", {"error": repr(error), "traceback": traceback.format_exc(),
                "unix": time.time(), "no_source_outcome_synthesized": True})
            raise
        finally:
            if handles:
                for handle in handles.values(): handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deadline-release", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.root, args.output)
        print(json.dumps({"qualification": binding(args.output), "frozen_scope_valid": result["frozen_scope_valid"],
            "original_window_expired": result["original_window_expired_at_observation"],
            "original_supervisor_locks_free": result["original_supervisor_locks_free_at_observation"]}))
    else:
        result = run(args.root, args.deadline_release)
        print(json.dumps({key: value for key, value in result.items() if key != "stages"}))
