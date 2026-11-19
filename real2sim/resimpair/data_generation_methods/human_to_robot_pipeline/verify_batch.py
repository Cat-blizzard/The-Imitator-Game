"""Finite CPU-only verifier for the frozen human-first scale192 producer.

This never calls the legacy review main, starts a simulator, edits receipts, or
publishes training rows. All outputs are immutable and remain pending RGB review.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / "PolicyGen"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read_result_inputs(queues=(), results_json=()):
    """Read explicitly supplied queues once; no hardcoded queue names or watcher."""
    result, inputs = [], []
    for supplied in queues:
        path = Path(supplied)
        if path.is_dir():
            path = next((path / name for name in ("summary.json", "STOPPED.json", "state.json")
                         if (path / name).exists()), None)
            if path is None:
                raise ValueError("Queue has no readable result snapshot")
        payload = load(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("completed"), list):
            raise ValueError("Queue snapshot must contain a completed list")
        result.extend(payload["completed"])
        inputs.append({**binding(path), "kind": "queue_snapshot", "terminal_queue": path.name != "state.json"})
    for path in results_json:
        payload = load(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Results JSON must contain a results list")
        result.extend(payload["results"])
        inputs.append({**binding(path), "kind": "explicit_results"})
    return result, inputs


def validate_job_and_windows(result, jobs, tasks):
    """Pure semantic binding checks; no same-key/cartesian pool expansion."""
    if not isinstance(result, dict) or result.get("job_id") not in jobs:
        raise ValueError("Result job is not present in the frozen plan")
    job = jobs[result["job_id"]]
    if result.get("job") != job:
        raise ValueError("Native result job differs from the frozen plan")
    task = tasks.get(job["task"])
    if task is None:
        raise ValueError("Job task has no frozen human task specification")
    allowed = {window["id"]: window for window in task["human_windows"]
               if window["split"] == job["split"]}
    ids = job.get("human_window_ids")
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or
            not set(ids) <= set(allowed) or job.get("human_window_id") not in ids):
        raise ValueError("Missing, duplicate, or cross-split human source window")
    windows = [allowed[name] for name in ids]
    for window in windows:
        interval = window.get("window_frames")
        if (not isinstance(interval, list) or len(interval) != 2 or
                any(type(value) is not int for value in interval) or not 0 <= interval[0] < interval[1] or
                not isinstance(window.get("fps"), (int, float)) or
                not math.isfinite(window["fps"]) or window["fps"] <= 0 or
                window.get("interval") != "start_inclusive_end_exclusive"):
            raise ValueError("Invalid human native interval or clock")
        if not all(window.get(key) for key in ("initial", "steps", "final", "review", "video")):
            raise ValueError("Incomplete human whole-atomic-task evidence")
    return job, task, windows


def verify_batch(plan_path, results, output, *, prior_records=(), input_bindings=()):
    """Run bounded raw checks. Return the persisted report; no training admission."""
    # This must precede importing legacy verifier dependencies.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    frozen, global_errors, records, excluded, pairs, human_checks = {}, [], [], [], [], []

    def pin(path, expected=None):
        path = str(Path(path).resolve(strict=True))
        if path not in frozen:
            frozen[path] = sha(path)
        if expected is not None and frozen[path] != expected:
            raise ValueError("Bound file hash differs: " + path)
        return {"path": path, "sha256": frozen[path]}

    def pin_binding(value):
        if not isinstance(value, dict) or not value.get("path") or not value.get("sha256"):
            raise ValueError("Missing explicit file hash binding")
        return pin(value["path"], value["sha256"])

    try:
        legacy = importlib.import_module("examples.baselines.hypericl_task_library.generation.scale192_v1.review_CPU_v1")
        plan_binding = pin(plan_path)
        plan = load(plan_path)
        pin_binding(plan["spec"])
        spec = load(plan["spec"]["path"])
        for path, digest in plan["source_sha256"].items():
            pin(path, digest)
        pin(__file__)
        pin(legacy.__file__)
        tasks = {task["id"]: task for task in spec["tasks"]}
        jobs = {job["id"]: job for job in plan["jobs"]}
        if len(jobs) != len(plan["jobs"]) or len(tasks) != len(spec["tasks"]):
            raise ValueError("Frozen plan/spec has duplicate identities")
        if not results:
            raise ValueError("No terminal attempts supplied")
        for source in input_bindings:
            pin_binding(source)
        prior_fingerprints = defaultdict(list)
        for path in prior_records:
            pin(path)
            prior = load(path)
            prior = prior.get("records") if isinstance(prior, dict) else prior
            if not isinstance(prior, list):
                raise ValueError("Prior records must be a list or contain records")
            for row in prior:
                fingerprint = row.get("physical_rollout_sha256", row.get("physical_fingerprint"))
                if not fingerprint:
                    raise ValueError("Prior record lacks physical fingerprint")
                prior_fingerprints[fingerprint].append(row.get("id", row.get("robot_id")))
        seen_jobs, human_cache, fingerprints = set(), {}, defaultdict(list)
        for result in results:
            name = result.get("job_id") if isinstance(result, dict) else None
            try:
                job, task, windows = validate_job_and_windows(result, jobs, tasks)
                if name in seen_jobs:
                    raise ValueError("Duplicate attempted job identity in batch")
                seen_jobs.add(name)
                process = result.get("process")
                if not isinstance(process, dict) or not isinstance(process.get("pid"), int) or not isinstance(process.get("start_ticks"), int):
                    raise ValueError("Terminal result lacks original native process identity")
                if legacy.same_live(process):
                    raise ValueError("Native process is still live; verification requires terminal output")
                if (result.get("outcome") != "native_success_pending_CPU_and_visual_review" or
                        result.get("exit_code") != 0 or result.get("deadline_exceeded", False)):
                    excluded.append({"job_id": name, "original_native_outcome": result.get("outcome"),
                                     "original_result": result, "native_outcome_preserved": True,
                                     "CPU_success_claimed": False, "training_admitted": False})
                    continue
                sidecar_root = legacy.ROOT / "scale192_bindings_v1" / name
                human_binding_path = sidecar_root / "human_task_binding.json"
                terminal_path = sidecar_root / "native_terminal_binding.json"
                pin(human_binding_path)
                pin(terminal_path)
                human_binding, terminal = load(human_binding_path), load(terminal_path)
                if human_binding.get("job") != job or terminal.get("job_id") != name:
                    raise ValueError("Native source binding differs from plan job")
                pin_binding(human_binding["human_spec"])
                pin_binding(human_binding["pilot_plan"])
                if (human_binding["human_spec"] != plan["spec"] or
                        Path(human_binding["pilot_plan"]["path"]).resolve() != Path(plan_path).resolve() or
                        human_binding["pilot_plan"]["sha256"] != plan_binding["sha256"]):
                    raise ValueError("Native binding does not refer to this exact plan/spec")
                pin_binding(human_binding["authorization"])
                authorization = load(human_binding["authorization"]["path"])
                if (name not in authorization.get("case_ids", []) or
                        not authorization.get("native_launch_authorized") or
                        authorization.get("pilot_plan_sha256") != plan_binding["sha256"]):
                    raise ValueError("Original launch authorization does not bind the case/plan")
                pin_binding(terminal["human_task_binding"])
                if Path(terminal["human_task_binding"]["path"]).resolve() != human_binding_path.resolve():
                    raise ValueError("Terminal record points to another human binding")
                summary_binding = pin_binding(terminal["native_summary"])
                expected_summary = Path(job["output"]) / "summary.json"
                if Path(summary_binding["path"]) != expected_summary.resolve():
                    raise ValueError("Terminal summary is outside the planned output")
                if result.get("native_summary") is not None:
                    pin_binding(result["native_summary"])
                    if result["native_summary"]["sha256"] != summary_binding["sha256"]:
                        raise ValueError("Queue summary and native terminal binding differ")
                summary = load(expected_summary)
                if (summary.get("attempted") != 1 or len(summary.get("results", [])) != 1 or
                        summary["results"][0] != result.get("native_result") or
                        summary["results"][0].get("passed") is not True):
                    raise ValueError("Original one-attempt summary/result differs")
                receipt_path = Path(result["native_result"]["receipt"])
                if not receipt_path.resolve().is_relative_to(Path(job["output"]).resolve()):
                    raise ValueError("Receipt is outside the planned native output")
                receipt_binding = pin(receipt_path)
                receipt = load(receipt_path)
                archive_path = Path(job["output"]) / "source_archive.json"
                pin(archive_path)
                archive = load(archive_path)
                for path, digest in receipt["source_sha256"].items():
                    archived = archive.get(path)
                    if not isinstance(archived, dict) or archived.get("sha256") != digest:
                        raise ValueError("Native source archive mapping differs from receipt")
                    pin(archived["path"], digest)
                commit_path = receipt_path.parent / "COMMITTED.json"
                commit_evidence = {"present": commit_path.exists(), "original_file_created_or_modified": False}
                if commit_path.exists():
                    commit_evidence["binding"] = pin(commit_path)
                    commit = load(commit_path)
                    if commit.get("verified") is not True or commit.get("receipt_sha256") != receipt_binding["sha256"]:
                        raise ValueError("Original COMMITTED receipt hash differs")
                elif job["asset"]["kind"] == "cabinet":
                    # Frozen v7 deliberately exports candidates without COMMITTED.
                    commit_evidence["scope"] = "native_v7_pending_independent_commit_no_original_commit_claimed"
                else:
                    raise ValueError("Original bowl/book COMMITTED record missing")
                for window in windows:
                    cache_key = (window["id"], window["video"]["sha256"], tuple(window["window_frames"]))
                    if cache_key in human_cache:
                        continue
                    pin_binding(window["video"])
                    pin_binding(window["review"])
                    actual_sheet = window.get("actual_review_sheet")
                    if actual_sheet:
                        pin_binding(actual_sheet)
                    start, end = window["window_frames"]
                    import numpy as np
                    preview = legacy._sheet(window["video"]["path"],
                                            np.linspace(start, end - 1, min(24, end - start), dtype=int),
                                            output / "human" / task["id"] / (window["id"] + ".jpg"),
                                            window["id"] + " complete human atomic window",
                                            expected_fps=float(window["fps"]))
                    checked_window = {"human_window": window, "preview": preview,
                                      "native_window_in_bounds": True, "source_hash_and_PTS_checked": True,
                                      "new_actual_visual_review_completed": False}
                    human_cache[cache_key] = checked_window
                    human_checks.append(checked_window)
                record = legacy.inspect_native(result, plan, output / "robot")
                fingerprint = record["physical_rollout_sha256"]
                if fingerprint in prior_fingerprints or fingerprint in fingerprints:
                    raise ValueError("Duplicate physical rollout: prior=" + str(prior_fingerprints.get(fingerprint, [])) +
                                     "; batch=" + str(fingerprints.get(fingerprint, [])))
                fingerprints[fingerprint].append(name)
                record.update(training_admitted=False, whole_task_pair_admitted=False,
                              original_COMMITTED=commit_evidence,
                              source_archive=pin(archive_path),
                              status="CPU_verified_pending_actual_visual_and_pair_review")
                records.append(record)
                for window in windows:
                    pairs.append({"id": name + "__" + window["id"], "robot_case_id": name,
                                  "human_window_id": window["id"], "task": task["id"],
                                  "canonical_goal": task["canonical_goal"], "split": job["split"],
                                  "source_session": window.get("session"),
                                  "human_video": window["video"], "human_window_frames": window["window_frames"],
                                  "human_native_fps": window["fps"], "interval": "start_inclusive_end_exclusive",
                                  "robot_receipt": record["receipt"], "robot_native_fps": record["native_FPS"],
                                  "robot_frame_window": [0, record["native_frame_count"]],
                                  "initial_state": task["initial"], "ordered_actions": task["required_steps"],
                                  "final_state": task["final"], "allowed_differences": task.get("allowed_differences", []),
                                  "forbidden_differences": task.get("forbidden_differences", []),
                                  "human_task_binding": record["human_task_binding"], "source_spec": plan["spec"],
                                  "same_split": True, "whole_atomic_goal_candidate": True,
                                  "status": "source_bound_candidate_pending_actual_visual_review",
                                  "time_or_motor_alignment": False, "human_action_retargeting": False,
                                  "actual_visual_review_completed": False, "training_admitted": False,
                                  "loader_pool_claimed": False})
            except Exception as error:
                excluded.append({"job_id": name, "original_native_outcome": result.get("outcome") if isinstance(result, dict) else None,
                                 "CPU_rejected": True, "error_type": type(error).__name__, "error": str(error),
                                 "original_result": result, "native_outcome_preserved": True,
                                 "not_retried": True, "training_admitted": False})
        # Guard against files changing while CPU verification was in progress.
        for path, digest in frozen.items():
            if sha(path) != digest:
                raise ValueError("Bound source changed during verification: " + path)
    except Exception as error:
        global_errors.append({"error_type": type(error).__name__, "error": str(error)})
    write(output / "verified_robot_records.json", {"records": records, "training_admitted": False,
          "actual_visual_and_semantic_pair_review_pending": True, "source_or_native_files_modified": False})
    write(output / "excluded_outcomes.json", {"records": excluded, "native_outcomes_preserved": True, "not_retried": True})
    write(output / "human_media_checks.json", {"checks": human_checks, "new_visual_review_completed": False})
    with (output / "paired_candidates.jsonl").open("x") as destination:
        for pair in pairs:
            destination.write(json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n")
    write(output / "frozen_inputs.json", {"source_sha256": frozen, "input_result_snapshots": list(input_bindings),
          "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "legacy_main_called": False})
    initial_groups = defaultdict(list)
    for record in records:
        initial_groups[record["initial_physics_sha256"]].append(record["id"])
    report = {"passed": bool(results) and not excluded and not global_errors and len(records) == len(results),
              "input_attempt_records": len(results), "CPU_verified_robot_trajectories": len(records),
              "excluded_attempts": len(excluded), "global_errors": global_errors,
              "original_outcome_counts": dict(Counter(r.get("outcome", "missing") if isinstance(r, dict) else "invalid" for r in results)),
              "by_task": dict(Counter(row["task"] for row in records)), "paired_source_candidates": len(pairs),
              "human_windows_checked": len(human_checks),
              "all_query_EEF_roundtrips": sum(row["all_query_EEF_roundtrips"] for row in records),
              "duplicate_initial_states_for_review": {key: value for key, value in initial_groups.items() if len(value) > 1},
              "training_admitted": False, "accepted_training_trajectories": 0,
              "actual_RGB_and_semantic_pair_review_pending": True,
              "simulation_or_model_training_started": False, "source_or_native_files_modified": False,
              "legacy_inspect_native_used": True, "legacy_main_called": False,
              "elapsed_seconds": time.time() - started}
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--queue", type=Path, action="append", default=[])
    parser.add_argument("--results-json", type=Path, action="append", default=[])
    parser.add_argument("--prior-records", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.queue and not args.results_json:
        parser.error("Provide --queue or --results-json")
    results, inputs = read_result_inputs(args.queue, args.results_json)
    report = verify_batch(args.plan, results, args.output, prior_records=args.prior_records, input_bindings=inputs)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
