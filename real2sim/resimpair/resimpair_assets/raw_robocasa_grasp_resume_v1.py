"""One explicitly approved deadline extension for the original four XML jobs.

The original bridge, START, protocol, jobs and registry claims stay byte-bound.
Only an absolute scheduling deadline changes. The retained worker/export and
original bridge's publisher remain the execution and consumer implementations.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from resimpair_assets import raw_robocasa_grasp_v1 as bridge
from resimpair_assets.raw_metric_containers_v1 import campaign_lock, identity, sufficient_disk
from resimpair_grasp.common import binding, read, resolve, write

RELEASE_SCHEMA = "raw_robocasa_grasp_deadline_resume_release_v1"
AUDIT_SCHEMA = "raw_robocasa_grasp_zero_submission_interruption_v1"


def frozen_scope(root, protocol):
    root = Path(root).resolve()
    if protocol["maximum_wall_seconds"] != 3600 or protocol["gpu_index"] != 3:
        raise ValueError("original_one_hour_GPU3_scope_required")
    return {"protocol": binding(root / "protocol.json"),
        "original_start": binding(root / "START.json"),
        "parent_gpu_release": bridge.require_release(root),
        "jobs": [{"source_id": spec["source_id"],
            "job": binding(Path(spec["output"]) / "job.json"),
            "claim": binding(Path(protocol["registry"]) / "claims" / (spec["allocation_key"] + ".json"))}
            for spec in protocol["job_specs"]]}


def live_processes(root):
    """Match actual argv tokens, excluding this read-only audit process."""
    root = Path(root).resolve(); rows = []
    for folder in Path("/proc").glob("[0-9]*"):
        if int(folder.name) == os.getpid():
            continue
        try:
            args = [part.decode(errors="replace") for part in (folder / "cmdline").read_bytes().split(b"\0") if part]
            relevant = str(root) in args or any(arg.startswith(str(root / "gpu") + "/") for arg in args)
            process = identity(int(folder.name))
            if relevant and process:
                rows.append({"identity": process, "argv": args})
        except OSError:
            continue
    return rows


def require_unsubmitted(protocol):
    for spec in protocol["job_specs"]:
        folder = Path(spec["output"])
        if {path.name for path in folder.iterdir()} != {"job.json"}:
            raise ValueError("deadline_extension_requires_original_zero_submission_state")


def audit(root, output):
    root = Path(root).resolve(); protocol = bridge.validate(root)
    with campaign_lock(root):
        scope = frozen_scope(root, protocol)
        require_unsubmitted(protocol)
        processes = live_processes(root)
        if processes:
            raise ValueError("live_original_pilot_process_blocks_interruption_audit")
        now = time.time(); start = read(resolve(scope["original_start"]))
        if now < start["deadline_unix"]:
            raise ValueError("original_deadline_has_not_expired")
        value = {"schema": AUDIT_SCHEMA, "unix": now, "scope": scope,
            "original_deadline_unix": start["deadline_unix"], "original_window_expired": True,
            "registered_jobs": 4, "actual_GPU_reservations": 0,
            "actual_optimizer_results": 0, "actual_exports": 0, "matching_live_processes": processes,
            "interruption_reported_by_parent": "first shared-GPU lock deferral followed by interruption",
            "interruption_observed_now": "original window expired; four original job-only directories; no actual submission",
            "original_START_protocol_jobs_claims_modified": False, "new_source_claims": 0,
            "new_attempts_created_by_audit": 0, "implementation": binding(__file__)}
        write(output, value)
    return value


def require_extension(root, protocol, release_path=None):
    """Called under the original campaign lock; never edits original START."""
    root = Path(root).resolve(); admission = root / "DEADLINE_RESUME_ADMISSION.json"
    reference = read(admission)["gate"] if admission.exists() else binding(release_path) if release_path else None
    if reference is None:
        raise ValueError("explicit_parent_deadline_extension_required")
    if release_path and reference != binding(release_path):
        raise ValueError("deadline_extension_gate_cannot_change_on_resume")
    gate = read(resolve(reference)); scope = frozen_scope(root, protocol)
    if (gate.get("schema") != RELEASE_SCHEMA or gate.get("reviewer_role") != "parent"
            or gate.get("passed") is not True or gate.get("scope") != scope
            or gate.get("implementation") != binding(__file__)
            or gate.get("authorized_existing_GPU_jobs") != 4
            or gate.get("authorized_new_source_claims") != 0 or gate.get("retries") != 0
            or gate.get("maximum_additional_wall_seconds") != 3600):
        raise ValueError("exact_original_four_job_parent_deadline_extension_required")
    evidence = read(resolve(gate["interruption_audit"]))
    if (evidence.get("schema") != AUDIT_SCHEMA or evidence.get("scope") != scope
            or evidence.get("implementation") != gate["implementation"]
            or evidence.get("original_window_expired") is not True
            or evidence.get("actual_GPU_reservations") != 0
            or evidence.get("actual_optimizer_results") != 0 or evidence.get("actual_exports") != 0
            or evidence.get("matching_live_processes") != []):
        raise ValueError("original_zero_submission_interruption_evidence_required")
    issued, deadline = gate.get("issued_unix"), gate.get("deadline_unix")
    if (type(issued) not in (int, float) or type(deadline) not in (int, float)
            or not math.isfinite(issued) or not math.isfinite(deadline)
            or not 0 < deadline - issued <= 3600 or issued > time.time()
            or issued < evidence["unix"] or issued < read(resolve(scope["original_start"]))["deadline_unix"]):
        raise ValueError("finite_absolute_one_hour_extension_deadline_required")
    record = {"gate": reference, "scope": scope, "original_START_preserved": True}
    if admission.exists():
        if read(admission) != record:
            raise ValueError("original_deadline_extension_admission_changed")
    else:
        require_unsubmitted(protocol)
        if live_processes(root):
            raise ValueError("live_original_pilot_process_blocks_deadline_extension")
        write(admission, record)
    return deadline, reference


def run(root, release_path=None):
    """Original serial scheduler with the separately admitted absolute deadline."""
    root = Path(root).resolve(); protocol = bridge.validate(root)
    with campaign_lock(root):
        deadline, extension = require_extension(root, protocol, release_path)
        sufficient_disk(root)
        for spec in protocol["job_specs"]:
            folder = Path(spec["output"]); path = folder / "job.json"
            if (folder / "terminal.json").exists():
                if read(folder / "terminal.json")["status"] not in {"completed", "asset_timeout", "asset_numeric_failure"}:
                    raise ValueError("existing_unknown_optimizer_state_requires_recovery")
                continue
            for stage in ("worker", "export"):
                receipt = folder / (stage + "_process.json")
                if receipt.exists() and bridge.retained.process_alive(read(receipt).get("identity")):
                    raise ValueError("live_previous_GPU_or_export_stage_no_resubmission")
            if (folder / "reservation.json").exists():
                reservation = read(folder / "reservation.json")
                if (reservation["job"] != binding(path) or reservation["protocol"] != binding(root / "protocol.json")
                        or reservation["attempt"] != 1 or reservation.get("deadline_extension") != extension):
                    raise ValueError("original_first_attempt_reservation_required")
            elif any((folder / name).exists() for name in ("worker_process.json", "export_process.json", "optimizer_result.json", "result.json")):
                raise ValueError("execution_evidence_without_reservation_requires_recovery")
            if (folder / "result.json").exists():
                result = bridge.retained.validate_result(path)
            else:
                if not (folder / "reservation.json").exists():
                    if time.time() >= deadline:
                        return bridge.retained_queue.gpu_refresh(root, "extension_deadline_no_new_submission")
                    with ExitStack() as stack:
                        fds = []
                        try:
                            for lock_path in bridge.resources.lock_paths(protocol["gpu_index"]):
                                handle = stack.enter_context(lock_path.open("a+"))
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); fds.append(handle.fileno())
                        except BlockingIOError:
                            return bridge.retained_queue.gpu_refresh(root, "shared_GPU_lock_deferred_no_submission")
                        free = int(subprocess.check_output(["nvidia-smi", "-i", str(protocol["gpu_index"]),
                            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=20).strip())
                        if free < protocol["minimum_free_GPU_GiB"] * 1024:
                            return bridge.retained_queue.gpu_refresh(root, "shared_GPU_memory_deferred_no_submission")
                        snapshot = bridge.resources.gpu_snapshot(protocol["gpu_index"])
                        if time.time() >= deadline:
                            return bridge.retained_queue.gpu_refresh(root, "extension_deadline_no_new_submission")
                        write(folder / "reservation.json", {"job": binding(path), "protocol": binding(root / "protocol.json"),
                            "gpu": snapshot, "free_memory_MiB": free, "attempt": 1, "unix": time.time(),
                            "deadline_extension": extension})
                        code = bridge._run_process([sys.executable, "-u", "-m", bridge.WORKER, "worker", "--job", str(path)],
                            folder, "worker", min(protocol["GPU_timeout_seconds"], max(1, deadline - time.time())),
                            bridge.retained._environment(path, snapshot["uuid"]), fds)
                    if code:
                        value = bridge._asset_error(folder, code); write(folder / "terminal.json", value)
                        if value["status"] == "needs_recovery":
                            raise ValueError("unknown_optimizer_failure_preserved")
                        continue
                elif not (folder / "optimizer_result.json").exists():
                    raise ValueError("unknown_GPU_reservation_never_resubmit")
                bridge.retained.validate_optimizer(path)
                if (folder / "export_process.json").exists():
                    raise ValueError("previous_export_requires_recovery_no_automatic_retry")
                if time.time() >= deadline:
                    return bridge.retained_queue.gpu_refresh(root, "extension_deadline_before_export_saved_optimizer")
                code = bridge._run_process([sys.executable, "-u", "-m", bridge.WORKER, "export", "--job", str(path)],
                    folder, "export", min(protocol["export_timeout_seconds"], max(1, deadline - time.time())),
                    bridge.retained._environment(path))
                if code:
                    write(folder / "terminal.json", {"status": "needs_recovery", "stage": "export", "returncode": code})
                    raise ValueError("retained_export_failed_preserve_for_recovery")
                result = bridge.retained.validate_result(path)
            write(folder / "terminal.json", {"status": "completed", "retained": result["retained"],
                "result": binding(folder / "result.json"), "optimizer_seconds": result["worker_seconds"],
                "export_seconds": result["export_seconds"], "native_conversion_reexecuted": False})
        return bridge.retained_queue.gpu_refresh(root, "completed_fixed_four_source_scope")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deadline-release", type=Path)
    args = parser.parse_args()
    print(audit(args.root, args.output) if args.command == "audit" else run(args.root, args.deadline_release))
