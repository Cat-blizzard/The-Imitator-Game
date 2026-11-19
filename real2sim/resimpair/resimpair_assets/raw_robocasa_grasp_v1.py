"""Four-source XML-native -> retained cuDex/export -> candidate consumer bridge.

Only input provenance differs from the existing raw pipeline. Reuses its GPU
worker, process runner, locks, result validators, rolling-index writer and
candidate publisher. Original XML native producers are verified directly;
metadata mirrors for the publisher never impersonate CPU execution receipts.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_grasp.cudex_native_com_v4 import condition
from resimpair_assets.metric_admission_v1 import inspect
from resimpair_assets import raw_candidate_bundle_v1 as publisher
from resimpair_assets import raw_candidate_bundle_v2 as producer_checks
from resimpair_assets import raw_candidate_bundle_v3 as final_checks
from resimpair_tasks import raw_asset_rollout_v1 as retained_queue
from resimpair_tasks import cudex_bulk_queue_v2 as retained
from resimpair_tasks.metric_cudex_stream_v8 import _run_process, _asset_error
from resimpair_assets.raw_metric_containers_v1 import campaign_lock, sufficient_disk
from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources

VERSION = "four_source_XML_native_cuDex_bridge_v1"
WORKER = "resimpair_grasp.cudex_native_com_v4"
REGISTRY = ROOT / "runtime/raw_robocasa_grasp_registry_v1"


def native_producer(entry, population_protocol):
    """Check the actual XML CPU job/result chain, without cooking or dynamics."""
    source_id = entry["source_id"]
    matches = [r for r in read(resolve(population_protocol))["jobs"] if r["source_id"] == source_id]
    if len(matches) != 1: raise ValueError("one_registered_XML_native_source_required")
    job_ref = matches[0]["job"]; path = resolve(job_ref); job = read(path)
    protocol = binding(Path(job["campaign"]) / "PROTOCOL.json")
    if not any(r["job"] == job_ref and r["owned"] for r in read(resolve(protocol))["jobs"]):
        raise ValueError("XML_native_job_not_owned_by_original_protocol")
    reservation = read(path.parent / "reservation.json")
    terminal_ref = binding(path.parent / "terminal.json"); terminal = read(resolve(terminal_ref))
    result = read(resolve(terminal["result"]))
    if (reservation["job"] != job_ref or reservation["protocol"] != protocol or reservation["attempt"] != 1
            or terminal["status"] != "native_ready" or result["status"] != "native_ready"
            or read(resolve(result["entry"])) != entry or result["state"] != entry["source_state"]):
        raise ValueError("XML_native_entry_not_actual_qualified_producer_output")
    descriptor = read(resolve(entry["asset"])); state = read(resolve(entry["source_state"]))
    raw = job["raw_row"]
    if (raw["source"] != "robocasa" or raw["id"] != source_id or descriptor["source_id"] != source_id
            or descriptor["provenance"]["source"] != "robocasa"
            or raw["partition"] != descriptor["partition"] or state["partition"] != raw["partition"]
            or entry.get("source_aliases", []) != job.get("source_aliases", [])):
        raise ValueError("XML_original_source_partition_or_alias_changed")
    for ref in job["frozen_inputs"]["files"]: resolve(ref)
    producer_checks.check_native_properties(entry, descriptor)
    gate = inspect(entry, state)
    if not gate["passed"] or gate != entry["semantic_scale_admission"]:
        raise ValueError("XML_native_current_scale_admission_changed")
    native_com = condition(entry["asset"])
    return {"job": job_ref, "protocol": protocol, "terminal": terminal_ref, "result": terminal["result"],
        "entry": result["entry"], "source_state": entry["source_state"],
        "raw_row": raw, "native_COM_condition": native_com,
        "recorded_prior_CPU_settle_steps": result["recorded_settle_steps"],
        "new_native_compilations": 0, "new_CPU_dynamic_steps": 0}


def code_bindings():
    paths = [Path(__file__), ROOT / "resimpair_grasp/cudex_native_com_v4.py",
        ROOT / "resimpair_grasp/cudex_seed_initialization_v2.py", Path(retained_queue.__file__),
        ROOT / "resimpair_tasks/metric_cudex_stream_v8.py", Path(publisher.__file__),
        Path(producer_checks.__file__), Path(final_checks.__file__),
        ROOT / "resimpair_assets/metric_admission_v1.py", ROOT / "resimpair_assets/raw_metric_containers_v1.py"]
    paths += list(retained.implementation_files())
    return [binding(path) for path in dict.fromkeys(paths)]


def prepare(root, native_index, xml_release, solver_protocol, *, gpu_index=3, registry=REGISTRY):
    """Register exactly the four existing qualified sources; submit zero jobs."""
    root, registry = Path(root).resolve(), Path(registry).resolve()
    if root.exists(): raise ValueError("existing_XML_grasp_pilot_must_resume_not_reprepare")
    value = read(native_index); entries = value["entries"]
    if len(entries) != 4 or len({e["source_id"] for e in entries}) != 4:
        raise ValueError("fixed_four_existing_native_sources_required")
    xml_protocol = value["population_protocol"]; release = read(xml_release)
    if (release.get("schema") != "raw_robocasa_native_v2_independent_release"
            or release.get("protocol") != xml_protocol or release.get("passed") is not True
            or release.get("reviewer_role") != "independent"):
        raise ValueError("independent_XML_native_release_binding_required")
    solver = read(solver_protocol)
    if type(gpu_index) is not int or gpu_index not in solver["GPU_indices"]:
        raise ValueError("one_existing_shared_GPU_index_required")
    for ref in solver["solver_configuration"] + [solver["manip"]]: resolve(ref)
    proofs = [native_producer(entry, xml_protocol) for entry in entries]
    code = code_bindings(); registry.mkdir(parents=True, exist_ok=True)
    with campaign_lock(registry):
        keys = [digest({"source_id": entry["source_id"], "asset": entry["asset"],
            "manip": solver["manip"], "solver_configuration": solver["solver_configuration"],
            "worker": binding(ROOT / "resimpair_grasp/cudex_native_com_v4.py"), "seed": 20261031}) for entry in entries]
        for key in keys:
            claim = registry / "claims" / (key + ".json")
            if claim.exists(): raise ValueError("existing_exact_geometry_grasp_allocation_resume_at:" + read(claim)["root"])
        root.mkdir(parents=True); specs = []
        for entry, proof, key in zip(entries, proofs, keys):
            folder = root / "gpu" / digest(entry["source_id"])[:24]
            metadata = root / "cpu" / digest(entry["source_id"])[:24] / "job.json"
            write(metadata, {"raw_row": proof["raw_row"], "source_aliases": entry.get("source_aliases", []),
                "original_XML_native_job": proof["job"], "metadata_mirror_only": True,
                "CPU_execution_claimed_here": False})
            specs.append({"source_id": entry["source_id"], "variant_id": entry["variant_id"],
                "asset": entry["asset"], "entry": entry, "output": str(folder), "content_key": entry["variant_id"],
                "manip": solver["manip"], "seed": 20261031, "source_state": entry["source_state"],
                "semantic_scale_admission": entry["semantic_scale_admission"], "source_aliases": entry.get("source_aliases", []),
                "XML_native_producer": proof, "metadata_mirror": binding(metadata), "allocation_key": key})
        protocol = {"schema": VERSION, "root": str(root), "native_index": binding(native_index),
            "XML_population_protocol": xml_protocol, "independent_XML_release": binding(xml_release),
            "solver_parent_protocol": binding(solver_protocol), "job_specs": specs,
            "registry": str(registry), "code": code, "configuration": solver["solver_configuration"],
            "maximum_new_GPU_jobs": 4, "seed_count_per_job": 256, "maximum_parallel_GPU_jobs": 1,
            "gpu_index": gpu_index, "minimum_free_GPU_GiB": 12, "GPU_timeout_seconds": 120,
            "export_timeout_seconds": 120, "maximum_wall_seconds": 3600, "retries": 0,
            "new_native_compilation_jobs": 0, "new_robot_executions": 0, "API_calls": 0,
            "GPU_release_required": True, "default_promoted": False,
            "native_conversion_failure_denominator": "separate original XML population, never included as optimizer failure"}
        write(root / "protocol.json", protocol)
        for spec in specs:
            path = Path(spec["output"]) / "job.json"
            write(path, {**spec, "protocol": binding(root / "protocol.json")})
            write(registry / "claims" / (spec["allocation_key"] + ".json"), {
                "root": str(root), "job": binding(path), "protocol": binding(root / "protocol.json"),
                "source_id": spec["source_id"], "asset": spec["asset"], "GPU_submitted_at_prepare": False})
    retained_queue.gpu_refresh(root, "prepared_not_submitted")
    return {"protocol": binding(root / "protocol.json"), "registered_jobs": 4, "GPU_submitted": 0,
            "total_optimizer_candidates_if_all_jobs_run": 1024, "new_CPU_native_jobs": 0}


def validate(root):
    root = Path(root).resolve(); p = read(root / "protocol.json")
    if (p["schema"] != VERSION or p["root"] != str(root) or len(p["job_specs"]) != 4
            or p["maximum_new_GPU_jobs"] != 4 or p["seed_count_per_job"] != 256
            or p["maximum_parallel_GPU_jobs"] != 1 or p["minimum_free_GPU_GiB"] != 12
            or p["GPU_timeout_seconds"] != 120 or p["export_timeout_seconds"] != 120 or p["retries"] != 0):
        raise ValueError("four_source_GPU_budget_and_original_limits_required")
    for ref in p["code"] + p["configuration"] + [p["native_index"], p["XML_population_protocol"],
            p["independent_XML_release"], p["solver_parent_protocol"]]: resolve(ref)
    expected_paths = {Path(spec["output"]) / "job.json" for spec in p["job_specs"]}
    if set((root / "gpu").glob("*/job.json")) != expected_paths:
        raise ValueError("GPU_jobs_outside_exact_four_source_allocation")
    for spec in p["job_specs"]:
        path = Path(spec["output"]) / "job.json"
        if read(path) != {**spec, "protocol": binding(root / "protocol.json")}:
            raise ValueError("prepared_GPU_job_or_geometry_changed")
        claim = read(Path(p["registry"]) / "claims" / (spec["allocation_key"] + ".json"))
        if claim["job"] != binding(path) or claim["protocol"] != binding(root / "protocol.json"):
            raise ValueError("original_GPU_claim_must_not_be_reallocated")
        resolve(spec["metadata_mirror"])
        if native_producer(spec["entry"], p["XML_population_protocol"]) != spec["XML_native_producer"]:
            raise ValueError("original_XML_native_producer_changed")
    return p


def require_release(root, release_path=None):
    root = Path(root); existing = root / "GPU_RELEASE.json"
    reference = read(existing)["gate"] if existing.exists() else binding(release_path) if release_path else None
    if reference is None: raise ValueError("parent_GPU_ledger_review_required_before_submission")
    if release_path and reference != binding(release_path): raise ValueError("GPU_release_cannot_change_on_resume")
    gate = read(resolve(reference))
    if (gate.get("schema") != "raw_robocasa_grasp_pilot_release_v1" or gate.get("passed") is not True
            or gate.get("protocol") != binding(root / "protocol.json") or gate.get("authorized_GPU_jobs") != 4
            or not gate.get("ledger_review")):
        raise ValueError("exact_four_job_parent_ledger_review_gate_required")
    resolve(gate["ledger_review"])
    return reference


def run(root, release_path=None):
    """Serial four-job bridge. Resource deferral returns without a submission."""
    root = Path(root).resolve(); p = validate(root); release = require_release(root, release_path)
    with campaign_lock(root):
        write(root / "GPU_RELEASE.json", {"gate": release, "protocol": binding(root / "protocol.json")})
        sufficient_disk(root)
        if not (root / "START.json").exists():
            now = time.time(); write(root / "START.json", {"unix": now, "deadline_unix": now + p["maximum_wall_seconds"]})
        deadline = read(root / "START.json")["deadline_unix"]
        for spec in p["job_specs"]:
            folder = Path(spec["output"]); path = folder / "job.json"
            if (folder / "terminal.json").exists():
                if read(folder / "terminal.json")["status"] == "needs_recovery":
                    raise ValueError("existing_unknown_optimizer_state_requires_recovery")
                continue
            for stage in ("worker", "export"):
                receipt = folder / (stage + "_process.json")
                if receipt.exists() and retained.process_alive(read(receipt).get("identity")):
                    raise ValueError("live_previous_GPU_or_export_stage_no_resubmission")
            if (folder / "result.json").exists():
                result = retained.validate_result(path)
            else:
                if not (folder / "reservation.json").exists():
                    if time.time() >= deadline: return retained_queue.gpu_refresh(root, "deadline_no_new_submission")
                    with ExitStack() as stack:
                        fds = []
                        try:
                            for lock_path in resources.lock_paths(p["gpu_index"]):
                                handle = stack.enter_context(lock_path.open("a+"))
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); fds.append(handle.fileno())
                        except BlockingIOError:
                            return retained_queue.gpu_refresh(root, "shared_GPU_lock_deferred_no_submission")
                        free = int(subprocess.check_output(["nvidia-smi", "-i", str(p["gpu_index"]),
                            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=20).strip())
                        if free < p["minimum_free_GPU_GiB"] * 1024:
                            return retained_queue.gpu_refresh(root, "shared_GPU_memory_deferred_no_submission")
                        snapshot = resources.gpu_snapshot(p["gpu_index"])
                        write(folder / "reservation.json", {"job": binding(path), "protocol": binding(root / "protocol.json"),
                            "gpu": snapshot, "free_memory_MiB": free, "attempt": 1, "unix": time.time()})
                        code = _run_process([sys.executable, "-u", "-m", WORKER, "worker", "--job", str(path)],
                            folder, "worker", min(p["GPU_timeout_seconds"], max(1, deadline - time.time())),
                            retained._environment(path, snapshot["uuid"]), fds)
                    if code:
                        value = _asset_error(folder, code); write(folder / "terminal.json", value)
                        if value["status"] == "needs_recovery": raise ValueError("unknown_optimizer_failure_preserved")
                        continue
                elif not (folder / "optimizer_result.json").exists():
                    raise ValueError("unknown_GPU_reservation_never_resubmit")
                retained.validate_optimizer(path)
                if (folder / "export_process.json").exists():
                    raise ValueError("previous_export_requires_recovery_no_automatic_retry")
                if time.time() >= deadline:
                    return retained_queue.gpu_refresh(root, "deadline_before_export_saved_optimizer_no_resubmission")
                code = _run_process([sys.executable, "-u", "-m", WORKER, "export", "--job", str(path)],
                    folder, "export", min(p["export_timeout_seconds"], max(1, deadline - time.time())), retained._environment(path))
                if code:
                    write(folder / "terminal.json", {"status": "needs_recovery", "stage": "export", "returncode": code})
                    raise ValueError("retained_export_failed_preserve_for_recovery")
                result = retained.validate_result(path)
            write(folder / "terminal.json", {"status": "completed", "retained": result["retained"],
                "result": binding(folder / "result.json"), "optimizer_seconds": result["worker_seconds"],
                "export_seconds": result["export_seconds"], "native_conversion_reexecuted": False})
        return retained_queue.gpu_refresh(root, "completed_fixed_four_source_scope")


def publish(root, baseline, container_catalog, output):
    """Validate true XML/GPU lineage, then call the existing candidate consumer."""
    root, output = Path(root).resolve(), Path(output).resolve(); p = validate(root)
    if output.exists(): raise ValueError("preserve_existing_XML_candidate_publication")
    rows = []
    for spec in p["job_specs"]:
        folder = Path(spec["output"]); path = folder / "job.json"
        if not (folder / "terminal.json").exists(): raise ValueError("four_GPU_outcomes_required_before_candidate_publication")
        terminal = read(folder / "terminal.json")
        if terminal["status"] == "needs_recovery": raise ValueError("unknown_GPU_outcome_cannot_be_published")
        reservation = read(folder / "reservation.json")
        if reservation["job"] != binding(path) or reservation["protocol"] != binding(root / "protocol.json"):
            raise ValueError("actual_GPU_reservation_binding_required")
        row = {"source_id": spec["source_id"], "XML_native_producer": spec["XML_native_producer"],
            "GPU_job": binding(path), "GPU_terminal": binding(folder / "terminal.json"), "status": terminal["status"]}
        if terminal["status"] == "completed":
            actual = retained.validate_result(path)
            if terminal["result"] != binding(folder / "result.json") or terminal["retained"] != actual["retained"]:
                raise ValueError("GPU_final_export_not_actual_completed_result")
            row["export"] = binding(folder / "result.json")
        rows.append(row)
    retained_queue.gpu_refresh(root, "capturing_actual_fixed_four_source_outputs")
    live = read(root / "native_cache_index.json"); final_checks.rolling_contract(live)
    receivers = producer_checks.validate_receiver_chains(baseline, container_catalog)
    capture = output / "captured_raw_root"; capture.mkdir(parents=True)
    for name in ("protocol.json", "cpu", "gpu"):
        (capture / name).symlink_to(root / name, target_is_directory=name != "protocol.json")
    write(capture / "native_cache_index.json", live)
    write(output / "XML_GPU_CHAIN_REVIEW.json", {"rows": rows, "receiver_review": receivers,
        "protocol": binding(root / "protocol.json"), "rolling_input": binding(capture / "native_cache_index.json"),
        "original_XML_producers_verified_directly": True, "CPU_execution_receipts_fabricated": False,
        "native_rules_and_export_filters_changed": False, "implementation": binding(__file__)})
    try:
        result = publisher.publish(capture, baseline, container_catalog, output / "consumer")
        checks = final_checks.check_published(root, resolve(result["bundle"]))
    except Exception as error:
        write(output / "REJECTED.json", {"error": repr(error), "published_for_task_use": False}); raise
    write(output / "CONSUMER_FIELD_CHECKS.json", {"rows": checks, "all_native_fields_equal_original_producer": True})
    result = {**result, "XML_GPU_producer_review": binding(output / "XML_GPU_CHAIN_REVIEW.json"),
        "consumer_field_checks": binding(output / "CONSUMER_FIELD_CHECKS.json"),
        "publication_integrity_passed": True, "physical_task_success": "unmeasured"}
    write(output / "RESULT.json", result); return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "status", "publish"])
    parser.add_argument("--root", type=Path, required=True)
    for name in ("native-index", "xml-release", "solver-protocol", "gpu-release", "baseline", "container-catalog", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--gpu-index", type=int, default=3)
    args = parser.parse_args()
    if args.command == "prepare": print(prepare(args.root, args.native_index, args.xml_release, args.solver_protocol, gpu_index=args.gpu_index))
    elif args.command == "run": print(run(args.root, args.gpu_release))
    elif args.command == "publish": print(publish(args.root, args.baseline, args.container_catalog, args.output))
    else:
        validate(args.root); print(retained_queue.gpu_refresh(args.root, "status_only_no_submission"))
