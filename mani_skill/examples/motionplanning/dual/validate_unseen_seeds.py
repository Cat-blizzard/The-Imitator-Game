"""Single-attempt seed11..20 test of the unchanged clearance-v2 executor.

Inputs are one frozen protocol and ten independent native subprocesses. Outputs
retain every attempted case, raw actions[T,16], seven state leaves[T+1,...],
source-curve arrays and physical audit traces. This runner never creates an
environment during preparation/self-test or modifies the execution modules.
"""
from __future__ import annotations

import argparse
from collections import Counter
from importlib import metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

from .validate_clearance_regression import (
    DEV, ENVIRONMENT, GPU, HERE, LEVEL_KEYS, PYTHON, ROOT, TIMEOUT,
    configuration, now, outcome, protected_history, sha,
)
from .validate_compact_seeds import atomic_json, evidence, read_ticks

BASELINE = ROOT / "infrastructure/validation/dual-arm/clearance-regression-20261115T171457407702Z"
PRECHANGE = ROOT / "infrastructure/validation/dual-arm/unseen-seeds-prechange-20261116.json"
SEEDS = tuple(range(11, 21))


def native_command(seed: int, output: Path) -> list[str]:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only the preregistered unseen seeds11..20 are allowed")
    return [str(PYTHON), "-m", "mani_skill.examples.motionplanning.dual.validate_sync_compact",
        "--mode", "positive", "--seed", str(seed), "--route", "merge-lift",
        "--execution-profile", "execution", "--joint-vel-limit", "1.5", "--joint-acc-limit", "2.6",
        "--approach-vel-limit", "0.9", "--approach-acc-limit", "0.9",
        "--max-episode-steps", "200", "--output-dir", str(output)]


def cases(batch: Path) -> list[dict[str, Any]]:
    return [{"id": f"positive-seed-{seed:02d}", "kind": "native", "mode": "positive",
        "seed": seed, "role": "unseen_evaluation", "output": str(batch / f"positive-seed-{seed:02d}"),
        "command": native_command(seed, batch / f"positive-seed-{seed:02d}")} for seed in SEEDS]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [r for r in rows if r.get("attempted") and r.get("role") == "unseen_evaluation"]
    seeds = [r["seed"] for r in attempted]
    if len(set(seeds)) != len(seeds) or any(seed not in SEEDS for seed in seeds):
        raise ValueError("Duplicate or ineligible seed in evaluation results")
    full = [r for r in attempted if r["outcome"]["complete_physical_closure"]]
    quality = [r for r in full if r.get("raw_recording_check", {}).get("status") == "passed"]
    official = sum(r["outcome"]["observed_official_success"] for r in attempted)
    steps = [r["outcome"]["total_environment_steps"] for r in full]
    wall = sum(r.get("wall_seconds", 0.) for r in attempted)
    return {"planned_denominator": 10, "attempted_count": len(attempted),
        "all_ten_attempted": len(attempted) == 10,
        "observed_official_success_count": official, "observed_official_success_rate": official / 10,
        "complete_physical_closure_count": len(full), "complete_physical_closure_rate": len(full) / 10,
        "classifications": dict(Counter(r["outcome"]["classification"] for r in attempted)),
        "successful_completion_steps_mean": statistics.mean(steps) if steps else None,
        "successful_completion_steps_sample_std": statistics.stdev(steps) if len(steps) > 1 else None,
        "successful_completion_steps_min": min(steps) if steps else None,
        "successful_completion_steps_max": max(steps) if steps else None,
        "all_attempted_subprocess_wall_seconds": wall,
        "quality_accepted_trajectory_count": len(quality),
        "accepted_trajectories_per_minute": len(quality) * 60 / wall if wall > 0 else None,
        "planning_wall_seconds": None,
        "rate_interpretation": "fixed ten-case denominator; interim counts are progress, not a completed evaluation"}


def versions(names: Any) -> dict[str, str]:
    result = {}
    for name in names:
        try: result[name] = metadata.version(name)
        except metadata.PackageNotFoundError: result[name] = "distribution metadata unavailable"
    return result


def prepare() -> Path:
    old_path = BASELINE / "protocol.json"
    baseline = json.loads(old_path.read_text())
    guard = json.loads(PRECHANGE.read_text())
    eligibility_path = Path(guard["eligibility_audit"])
    eligibility = json.loads(eligibility_path.read_text())
    if (guard["candidate_unseen_seeds"] != list(SEEDS)
            or not guard["eligibility_audit_qualified_all_ten"]
            or sha(eligibility_path) != guard["eligibility_audit_sha256"]
            or eligibility["status"] != "passed" or eligibility["candidate_unseen_seeds"] != list(SEEDS)):
        raise RuntimeError("Unseen qualification evidence must cover exactly seeds11..20")
    expected = baseline["source19_new_sha256"]
    if guard["baseline19_sha256"] != expected or guard["baseline_protocol_sha256"] != sha(old_path):
        raise RuntimeError("Unseen eligibility guard refers to a different baseline")
    if len(expected) != 19 or {name: sha(HERE / name) for name in expected} != expected:
        raise RuntimeError("All19 execution files must exactly match the tested clearance version")
    config = configuration()
    if config != baseline["execution_configuration"]:
        raise RuntimeError("Unseen evaluation must not change any execution parameter")
    if versions(baseline["package_versions"]) != baseline["package_versions"]:
        raise RuntimeError("Runtime package versions differ from the tested baseline")
    core, assets = baseline["core11_sha256"], baseline["asset_builder_sha256"]
    aux = dict(baseline["auxiliary_frozen_source_sha256"])
    aux[str(Path(baseline["runner_path"]).relative_to(DEV))] = baseline["runner_sha256"]
    if {name: sha(DEV / name) for name in core} != core or {name: sha(DEV / name) for name in aux} != aux:
        raise RuntimeError("Core or immutable helper identity changed")
    if {name: sha(Path(name)) for name in assets} != assets:
        raise RuntimeError("Asset/builder identity changed")
    levels = {key: os.environ.get(key) for key in LEVEL_KEYS}
    if levels != baseline["task_level_environment"]:
        raise RuntimeError("Task-level environment must retain L0 configuration")
    history = protected_history(guard) | {str(PRECHANGE): sha(PRECHANGE)}
    if {name: sha(Path(name)) for name in history} != history:
        raise RuntimeError("Protected historical evidence changed")
    batch = ROOT / "infrastructure/validation/dual-arm" / time.strftime("unseen-seeds-%Y%m%dT%H%M%SZ", time.gmtime())
    batch.mkdir(exist_ok=False)
    snapshot = batch / "source"; snapshot.mkdir()
    for name in set(expected) | {Path(name).name for name in aux} | {Path(__file__).name}:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    atomic_json(batch / "historical-manifest.json", history)
    protocol = {"schema_version": 1, "prepared_at": now(),
        "purpose": "new random initializations of the same L0 task after all method parameters were frozen",
        "seeds": list(SEEDS), "planned_denominator": 10, "attempts_per_case": 1,
        "retry_allowed": False, "per_seed_parameter_adjustment_allowed": False,
        "parameter_search_allowed": False, "post_failure_method_changes_allowed": False,
        "execution_order": "ascending, independent sequential subprocesses",
        "failure_retention": "all attempted failures retained in ten-case denominator; no replacement seeds",
        "ordered_cases": cases(batch), "case_timeout_seconds": TIMEOUT,
        "task": baseline["task"], "execution_configuration": config,
        "source19_new_sha256": expected, "allowed_changed_files": [], "actually_changed_files": [],
        "core11_sha256": core, "asset_builder_sha256": assets, "auxiliary_frozen_source_sha256": aux,
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha(Path(__file__)),
        "new_source_snapshot": str(snapshot),
        "new_source_snapshot_sha256": {item.name: sha(item) for item in snapshot.iterdir()},
        "historical_manifest": str(batch / "historical-manifest.json"),
        "historical_manifest_sha256": sha(batch / "historical-manifest.json"), "historical_file_count": len(history),
        "prechange_guard": str(PRECHANGE), "prechange_guard_sha256": sha(PRECHANGE),
        "eligibility_audit": str(eligibility_path), "eligibility_audit_sha256": sha(eligibility_path),
        "baseline_batch": str(BASELINE), "baseline_protocol_sha256": sha(old_path),
        "seen_seeds_excluded": list(range(11)), "unseen_scope": "not previously used in development; same task and initial-state distribution, not unseen tasks or object classes",
        "environment": ENVIRONMENT, "gpu_uuid": GPU, "task_level_environment": levels,
        "python_executable": str(PYTHON), "package_versions": baseline["package_versions"],
        "physics_backend": "cpu", "render_backend": "gpu", "control_hz": 20, "physics_hz": 100,
        "max_episode_steps": 200, "acceptance_contract": guard["contract_to_preserve"],
        "source_violation_policy": "stop subsequent launches; retain attempted evidence and unattempted slots",
        "native_time_limit_policy": baseline["native_time_limit_policy"],
        "native_success_policy": baseline["native_success_policy"],
        "prior_controls_and_replays": {"source_batch": str(BASELINE), "audit_sha256": sha(BASELINE / "audit.json"),
            "scope": "existing seed0 regression only; no claim of replaying the new episodes", "rerun": False},
        "planning_wall_seconds": None, "wall_time_scope": "startup, planning, collision checks, physics, rendering and recording"}
    atomic_json(batch / "protocol.json", protocol)
    return batch


def check_frozen(protocol: dict[str, Any], path: Path, digest: str) -> dict[str, Any]:
    if sha(path) != digest or sha(Path(__file__)) != protocol["runner_sha256"]:
        raise RuntimeError("Protocol or runner changed after freezing")
    for names, base in (("source19_new_sha256", HERE), ("core11_sha256", DEV),
                        ("auxiliary_frozen_source_sha256", DEV), ("asset_builder_sha256", Path("/"))):
        if {name: sha(base / name) for name in protocol[names]} != protocol[names]:
            raise RuntimeError(f"Frozen identity changed: {names}")
    if configuration() != protocol["execution_configuration"] or versions(protocol["package_versions"]) != protocol["package_versions"]:
        raise RuntimeError("Frozen configuration or runtime packages changed")
    snapshot = Path(protocol["new_source_snapshot"])
    if {item.name: sha(item) for item in snapshot.iterdir()} != protocol["new_source_snapshot_sha256"]:
        raise RuntimeError("Source snapshot changed")
    manifest = Path(protocol["historical_manifest"])
    if sha(manifest) != protocol["historical_manifest_sha256"]:
        raise RuntimeError("Historical manifest changed")
    history = json.loads(manifest.read_text())
    if {name: sha(Path(name)) for name in history} != history:
        raise RuntimeError("Protected historical evidence changed")
    return {"source19_sha256": protocol["source19_new_sha256"], "historical_files_verified": len(history),
        "core_assets_auxiliary_and_packages_match": True, "source_snapshot_match": True}


def run(batch: Path) -> int:
    path = batch / "protocol.json"; protocol = json.loads(path.read_text()); digest = sha(path)
    if (protocol["ordered_cases"] != cases(batch) or protocol["seeds"] != list(SEEDS)
            or protocol["attempts_per_case"] != 1 or protocol["retry_allowed"]
            or protocol["case_timeout_seconds"] != TIMEOUT):
        raise RuntimeError("Preregistered seeds, commands or allowance differ")
    with (batch / "launch-once.json").open("x") as stream:
        json.dump({"started_at": now(), "protocol_sha256": digest}, stream)
    check_frozen(protocol, path, digest)
    env = os.environ.copy(); env.update(ENVIRONMENT)
    for key, value in protocol["task_level_environment"].items():
        if value is None: env.pop(key, None)
        else: env[key] = value
    summary: dict[str, Any] = {"status": "running", "started_at": now(),
        "protocol_sha256": digest, "cases": [], "aggregate": aggregate([])}
    atomic_json(batch / "summary.json", summary)
    try:
        for case in protocol["ordered_cases"]:
            before = check_frozen(protocol, path, digest)
            output = Path(case["output"])
            if output.exists(): raise RuntimeError("Case output already exists; retry is prohibited")
            row = {**case, "attempted": True, "attempt": 1, "started_at": now(), "source_before": before}
            receipt = batch / f"{case['id']}-receipt.json"; log = batch / f"{case['id']}.log"
            row.update(log=str(log), cwd=str(DEV), environment=ENVIRONMENT)
            atomic_json(receipt, row)
            start = time.monotonic(); timed_out = False; spawn_error = None
            with log.open("wb") as stream:
                try:
                    process = subprocess.Popen(case["command"], cwd=DEV, env=env,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as exc:
                    spawn_error = f"{type(exc).__name__}: {exc}"; exit_code = 127
                    stream.write(spawn_error.encode("utf-8"))
                else:
                    row["pid"] = process.pid; atomic_json(receipt, row)
                    try: exit_code = process.wait(timeout=TIMEOUT)
                    except subprocess.TimeoutExpired:
                        timed_out = True; os.killpg(process.pid, signal.SIGTERM)
                        try: exit_code = process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL); exit_code = process.wait()
            row.update(exit_code=exit_code, timed_out=timed_out, spawn_error=spawn_error,
                wall_seconds=time.monotonic() - start, completed_at=now())
            atomic_json(receipt, row)
            report_path = output / "report.json"; violations = []
            try:
                report = json.loads(report_path.read_text()) if report_path.exists() else {}
                ticks = read_ticks(output / "ticks.jsonl")
            except (OSError, ValueError) as exc:
                report, ticks = {}, []; row["artifact_parse_error"] = f"{type(exc).__name__}: {exc}"
            row["outcome"] = outcome(case, report, ticks, timed_out, exit_code)
            if spawn_error: row["outcome"]["classification"] = "subprocess_launch_error"
            try: row["raw_recording_check"] = evidence(output, report, ticks)
            except Exception as exc:
                row["raw_recording_check"] = {"status": "invalid_or_unavailable", "error": f"{type(exc).__name__}: {exc}"}
            if report and (report.get("source_sha256") != protocol["source19_new_sha256"]
                    or report.get("execution_configuration") != protocol["execution_configuration"]):
                violations.append("native source or configuration differs from frozen protocol")
            try: row["source_after"] = check_frozen(protocol, path, digest)
            except Exception as exc: violations.append(f"{type(exc).__name__}: {exc}")
            row.update(report=str(report_path), report_sha256=sha(report_path) if report_path.exists() else None,
                artifact_errors=report.get("artifact_errors"), protocol_violations=violations)
            atomic_json(receipt, row); summary["cases"].append(row)
            summary["aggregate"] = aggregate(summary["cases"]); atomic_json(batch / "summary.json", summary)
            print(json.dumps({"case": case["id"], "exit_code": exit_code,
                "outcome": row["outcome"], "raw_recording_check": row["raw_recording_check"]["status"]}), flush=True)
            if violations: raise RuntimeError("; ".join(violations))
        summary.update(status="completed", completed_at=now(), all_preregistered_cases_attempted_once=True)
        atomic_json(batch / "summary.json", summary); return 0
    except Exception as exc:
        summary.update(status="stopped_protocol_or_runner_error", error=f"{type(exc).__name__}: {exc}", stopped_at=now())
        atomic_json(batch / "summary.json", summary); raise


def self_test() -> dict[str, Any]:
    records = cases(Path("/tmp/unseen-seed-contract"))
    assert len(records) == 10 and [r["seed"] for r in records] == list(SEEDS)
    for row in records:
        cmd = row["command"]
        assert row["kind"] == "native" and cmd[cmd.index("--mode") + 1] == "positive"
        assert cmd[cmd.index("--seed") + 1] == str(row["seed"])
        assert cmd[cmd.index("--max-episode-steps") + 1] == "200"
    for seed in (True, 0, 10, 21, "11"):
        try: native_command(seed, Path("/tmp/unused"))
        except ValueError: pass
        else: raise AssertionError("Ineligible seed accepted")
    success = records[0] | {"attempted": True, "wall_seconds": 30., "raw_recording_check": {"status": "passed"}, "outcome": {
        "complete_physical_closure": True, "observed_official_success": True,
        "total_environment_steps": 190, "classification": "complete"}}
    failure = records[1] | {"attempted": True, "wall_seconds": 30., "outcome": {
        "complete_physical_closure": False, "observed_official_success": True,
        "total_environment_steps": 200, "classification": "native_time_limit"}}
    result = aggregate([success, failure])
    assert result["attempted_count"] == 2 and result["complete_physical_closure_rate"] == .1
    assert result["observed_official_success_rate"] == .2 and not result["all_ten_attempted"]
    assert result["successful_completion_steps_mean"] == 190 and result["accepted_trajectories_per_minute"] == 1.
    invalid_data = success | {"raw_recording_check": {"status": "invalid_or_unavailable"}}
    assert aggregate([invalid_data])["quality_accepted_trajectory_count"] == 0
    assert aggregate([invalid_data])["complete_physical_closure_count"] == 1
    try: aggregate([success, success])
    except ValueError: pass
    else: raise AssertionError("Duplicate attempt accepted")
    timed = outcome(records[0], {"status": "passed", "official_200_step_complete": True}, [], True, 0)
    assert not timed["complete_physical_closure"]
    with tempfile.TemporaryDirectory(prefix="tig-unseen-contract-") as td:
        marker = Path(td) / "launch-once.json"
        marker.open("x").close()
        try: marker.open("x")
        except FileExistsError: pass
        else: raise AssertionError("A second launch was accepted")
    return {"status": "passed", "environment_created": False, "physics_steps": 0,
        "checks": ["ten new seeds only; no development/control/replay cases", "exact frozen argument construction",
            "fixed denominator and failures retained", "success-only step statistics and all-attempt wall cost",
            "duplicate/type/range rejection", "timeout cannot count as closure", "exclusive launch marker"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true"); group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--batch-dir", type=Path); args = parser.parse_args()
    if args.self_test: print(json.dumps(self_test(), indent=2)); return 0
    if Path(sys.executable).resolve() != PYTHON.resolve(): parser.error("Use the frozen official Python")
    if args.prepare_only:
        batch = prepare(); print(json.dumps({"batch": str(batch), "protocol": str(batch / "protocol.json")})); return 0
    return run(args.batch_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
