"""Single-attempt regression of bounded geometric initial support on known seeds.

Inputs: a frozen19-source protocol, recorded prechange history, positive seeds
0..20, one retain-right0 control and state/action replay of the new seed0.
Outputs retain all attempted cases and failures; fixed positive denominator20
excludes development0 and controls. No per-seed tuning or retries. Every native
case uses the original200-step physical acceptance contract and configuration.
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
    configuration, now, outcome, protected_history, replay_evidence, sha,
)
from .validate_compact_seeds import atomic_json, evidence, read_ticks

BASELINE = ROOT / "infrastructure/validation/dual-arm/unseen-seeds-20261115T173800Z"
PRECHANGE = ROOT / "infrastructure/validation/dual-arm/near-support-prechange-20261116.json"
SEEDS = tuple(range(1, 21))
PREPARATION = ROOT / "infrastructure/validation/dual-arm/near-support-fix-20261116"
ALLOWED_CHANGES = frozenset(("table_support_collision.py",))


def native_command(seed: int, mode: str, output: Path) -> list[str]:
    if type(seed) is not int or seed not in range(21) or mode not in ("positive", "retain-right"):
        raise ValueError("Only positive0..20 and retain-right0 are allowed")
    if mode == "retain-right" and seed != 0:
        raise ValueError("Only development0 is the preregistered fault control")
    return [str(PYTHON), "-m", "mani_skill.examples.motionplanning.dual.validate_sync_compact",
        "--mode", mode, "--seed", str(seed), "--route", "merge-lift",
        "--execution-profile", "execution", "--joint-vel-limit", "1.5", "--joint-acc-limit", "2.6",
        "--approach-vel-limit", "0.9", "--approach-acc-limit", "0.9",
        "--max-episode-steps", "200", "--output-dir", str(output)]


def cases(batch: Path) -> list[dict[str, Any]]:
    records = [{"id": f"positive-seed-{seed:02d}", "kind": "native", "mode": "positive", "seed": seed,
        "role": "development_regression" if seed == 0 else "matched_regression",
        "output": str(batch / f"positive-seed-{seed:02d}"),
        "command": native_command(seed, "positive", batch / f"positive-seed-{seed:02d}")} for seed in range(21)]
    output = batch / "negative-seed-00"
    records.append({"id": "negative-seed-00", "kind": "native", "mode": "retain-right", "seed": 0,
        "role": "fault_detection_control", "output": str(output), "command": native_command(0, "retain-right", output)})
    for mode, policy in (("state", "full"), ("action", "fixed-root-exact")):
        output = batch / f"replay-{mode}"
        command = [str(PYTHON), "-m", "mani_skill.examples.motionplanning.dual.validate_compact_replay",
            "--mode", mode, "--recording-kind", "compact", "--restore-policy", policy,
            "--input-dir", str(batch / "positive-seed-00"), "--output-dir", str(output)]
        records.append({"id": f"replay-{mode}", "kind": "replay", "mode": mode, "restore_policy": policy,
            "seed": 0, "role": "seed0_recording_integrity", "output": str(output), "command": command})
    return records


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [r for r in rows if r.get("attempted") and r.get("role") == "matched_regression"]
    seeds = [r["seed"] for r in attempted]
    if len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed not in SEEDS for seed in seeds):
        raise ValueError("Duplicate or ineligible regression seed")
    full = [r for r in attempted if r["outcome"]["complete_physical_closure"]]
    quality = [r for r in full if r.get("raw_recording_check", {}).get("status") == "passed"]
    official = sum(r["outcome"]["observed_official_success"] for r in attempted)
    steps = [r["outcome"]["total_environment_steps"] for r in full]
    wall = sum(r.get("wall_seconds", 0.) for r in attempted)
    groups = {}
    for label, selected in (("known1_to10", range(1, 11)), ("known11_to20", range(11, 21))):
        group = [r for r in attempted if r["seed"] in selected]
        groups[label] = {"fixed_denominator": 10, "attempted": len(group),
            "observed_official_success_count": sum(r["outcome"]["observed_official_success"] for r in group),
            "complete_physical_closure_count": sum(r["outcome"]["complete_physical_closure"] for r in group)}
    return {"planned_denominator": 20, "attempted_count": len(attempted), "all_twenty_attempted": len(attempted) == 20,
        "observed_official_success_count": official, "observed_official_success_rate": official / 20,
        "complete_physical_closure_count": len(full), "complete_physical_closure_rate": len(full) / 20,
        "classifications": dict(Counter(r["outcome"]["classification"] for r in attempted)), "cohorts": groups,
        "successful_completion_steps_mean": statistics.mean(steps) if steps else None,
        "successful_completion_steps_sample_std": statistics.stdev(steps) if len(steps) > 1 else None,
        "successful_completion_steps_min": min(steps) if steps else None,
        "successful_completion_steps_max": max(steps) if steps else None,
        "all_attempted_subprocess_wall_seconds": wall, "quality_accepted_trajectory_count": len(quality),
        "accepted_trajectories_per_minute": len(quality) * 60 / wall if wall > 0 else None,
        "planning_wall_seconds": None, "development_control_and_replay_excluded": True,
        "rate_interpretation": "fixed known-seed20 regression denominator, all failures retained; no fresh generalization claim"}


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
    old19 = baseline["source19_new_sha256"]
    if guard["baseline19_sha256"] != old19 or guard["baseline_protocol_sha256"] != sha(old_path):
        raise RuntimeError("Prechange guard refers to a different baseline")
    expected = {name: sha(HERE / name) for name in old19}
    changed = sorted(name for name in old19 if expected[name] != old19[name])
    if len(expected) != 19 or set(changed) != ALLOWED_CHANGES:
        raise RuntimeError("Exactly table_support_collision.py may change within the19 execution files")
    revision_path = PREPARATION / "revision-evidence.json"
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    if revision["status"] != "passed" or revision["source19_new_sha256"] != expected:
        raise RuntimeError("Zero-step repair evidence does not identify this exact method")
    if any(sha(Path(name)) != digest for name, digest in revision["references_sha256"].items()):
        raise RuntimeError("Repair self-test or zero-step evidence changed")
    config = configuration()
    if config != baseline["execution_configuration"]:
        raise RuntimeError("Support repair must not change any execution parameter")
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
    batch = ROOT / "infrastructure/validation/dual-arm" / time.strftime("near-support-regression-%Y%m%dT%H%M%SZ", time.gmtime())
    batch.mkdir(exist_ok=False)
    snapshot = batch / "source"; snapshot.mkdir()
    for name in set(expected) | {Path(name).name for name in aux} | {Path(__file__).name}:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    previous = batch / "previous-source"; previous.mkdir()
    source_before = PREPARATION / "source-before/table_support_collision.py"
    if sha(source_before) != old19["table_support_collision.py"]:
        raise RuntimeError("Reviewed previous support snapshot changed")
    (previous / source_before.name).write_bytes(source_before.read_bytes())
    atomic_json(batch / "historical-manifest.json", history)
    protocol = {"schema_version": 1, "prepared_at": now(),
        "purpose": "matched known-seed regression of geometric-only initial support; all execution parameters and acceptance gates retained",
        "seeds": list(SEEDS), "positive_seeds": list(range(21)), "development_seed": 0, "planned_denominator": 20, "attempts_per_case": 1,
        "retry_allowed": False, "per_seed_parameter_adjustment_allowed": False,
        "parameter_search_allowed": False, "post_failure_method_changes_allowed": False,
        "execution_order": "ascending, independent sequential subprocesses",
        "failure_retention": "all attempted failures retained in twenty-case regression denominator; no replacement seeds",
        "ordered_cases": cases(batch), "case_timeout_seconds": TIMEOUT,
        "task": baseline["task"], "execution_configuration": config,
        "source19_old_sha256": old19, "source19_new_sha256": expected, "allowed_changed_files": sorted(ALLOWED_CHANGES), "actually_changed_files": changed,
        "core11_sha256": core, "asset_builder_sha256": assets, "auxiliary_frozen_source_sha256": aux,
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha(Path(__file__)),
        "new_source_snapshot": str(snapshot),
        "new_source_snapshot_sha256": {item.name: sha(item) for item in snapshot.iterdir()},
        "historical_manifest": str(batch / "historical-manifest.json"),
        "historical_manifest_sha256": sha(batch / "historical-manifest.json"), "historical_file_count": len(history),
        "prechange_guard": str(PRECHANGE), "prechange_guard_sha256": sha(PRECHANGE),
        "revision_evidence": str(revision_path), "revision_evidence_sha256": sha(revision_path),
        "old_source_snapshot": str(previous), "old_source_snapshot_sha256": {source_before.name: sha(source_before)},
        "baseline_batch": str(BASELINE), "baseline_protocol_sha256": sha(old_path),
        "interpretation": "all1..20 are now observed regression seeds; original unseen9/10 remains unchanged",
        "environment": ENVIRONMENT, "gpu_uuid": GPU, "task_level_environment": levels,
        "python_executable": str(PYTHON), "package_versions": baseline["package_versions"],
        "physics_backend": "cpu", "render_backend": "gpu", "control_hz": 20, "physics_hz": 100,
        "max_episode_steps": 200, "acceptance_contract": guard["contract_to_preserve"],
        "source_violation_policy": "stop subsequent launches; retain attempted evidence and unattempted slots",
        "native_time_limit_policy": baseline["native_time_limit_policy"],
        "native_success_policy": baseline["native_success_policy"],
        "controls_and_replays": "one new retain-right0, state replay and fixed-root-exact action replay on new development0 recording",
        "support_repair": "initial geometric-only near-top evidence without fabricated FCL contacts; original candidate/motion/release gates unchanged",
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
    revision_path = Path(protocol["revision_evidence"])
    if sha(revision_path) != protocol["revision_evidence_sha256"]:
        raise RuntimeError("Repair evidence manifest changed")
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    if any(sha(Path(name)) != digest for name, digest in revision["references_sha256"].items()):
        raise RuntimeError("Repair evidence reference changed")
    previous = Path(protocol["old_source_snapshot"])
    if {item.name: sha(item) for item in previous.iterdir()} != protocol["old_source_snapshot_sha256"]:
        raise RuntimeError("Previous support source snapshot changed")
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
            if case["kind"] == "replay" and not summary["cases"][0]["outcome"]["complete_physical_closure"]:
                row = {**case, "attempted": False, "status": "skipped_seed0_not_passed", "completed_at": now()}
                atomic_json(batch / f"{case['id']}-receipt.json", row)
                summary["cases"].append(row); atomic_json(batch / "summary.json", summary); continue
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
            try:
                row["raw_recording_check"] = (evidence(output, report, ticks) if case["kind"] == "native"
                    else replay_evidence(output, batch / "positive-seed-00", case["mode"]))
            except Exception as exc:
                row["raw_recording_check"] = {"status": "invalid_or_unavailable", "error": f"{type(exc).__name__}: {exc}"}
            if report and case["kind"] == "native" and (report.get("source_sha256") != protocol["source19_new_sha256"]
                    or report.get("execution_configuration") != protocol["execution_configuration"]):
                violations.append("native source or configuration differs from frozen protocol")
            if report and case["kind"] == "replay" and report.get("input_execution_source_sha256") != protocol["source19_new_sha256"]:
                violations.append("replay input source differs from frozen protocol")
            try: row["source_after"] = check_frozen(protocol, path, digest)
            except Exception as exc: violations.append(f"{type(exc).__name__}: {exc}")
            row.update(report=str(report_path), report_sha256=sha(report_path) if report_path.exists() else None,
                artifact_errors=report.get("artifact_errors"), protocol_violations=violations)
            atomic_json(receipt, row); summary["cases"].append(row)
            summary["aggregate"] = aggregate(summary["cases"]); atomic_json(batch / "summary.json", summary)
            print(json.dumps({"case": case["id"], "exit_code": exit_code,
                "outcome": row["outcome"], "raw_recording_check": row["raw_recording_check"]["status"]}), flush=True)
            if violations: raise RuntimeError("; ".join(violations))
        summary.update(status="completed", completed_at=now(),
            all_preregistered_cases_attempted_once=all(row["attempted"] for row in summary["cases"]),
            all_preregistered_native_cases_attempted_once=all(row["attempted"] for row in summary["cases"] if row["kind"] == "native"))
        atomic_json(batch / "summary.json", summary); return 0
    except Exception as exc:
        summary.update(status="stopped_protocol_or_runner_error", error=f"{type(exc).__name__}: {exc}", stopped_at=now())
        atomic_json(batch / "summary.json", summary); raise


def self_test() -> dict[str, Any]:
    records = cases(Path("/tmp/near-support-contract"))
    assert len(records) == 24 and [r["seed"] for r in records[:21]] == list(range(21))
    assert records[21]["mode"] == "retain-right" and records[22]["restore_policy"] == "full"
    assert records[23]["restore_policy"] == "fixed-root-exact"
    for row in records[:22]:
        cmd = row["command"]
        assert cmd[cmd.index("--seed") + 1] == str(row["seed"])
        assert cmd[cmd.index("--max-episode-steps") + 1] == "200"
        assert cmd[cmd.index("--joint-acc-limit") + 1] == "2.6"
    for seed in (True, -1, 21, "12"):
        try: native_command(seed, "positive", Path("/tmp/unused"))
        except ValueError: pass
        else: raise AssertionError("Ineligible seed accepted")
    success = records[1] | {"attempted": True, "wall_seconds": 30., "raw_recording_check": {"status": "passed"}, "outcome": {
        "complete_physical_closure": True, "observed_official_success": True,
        "total_environment_steps": 190, "classification": "complete"}}
    failure = records[12] | {"attempted": True, "wall_seconds": 30., "outcome": {
        "complete_physical_closure": False, "observed_official_success": True,
        "total_environment_steps": 200, "classification": "native_time_limit"}}
    development = success | {"seed": 0, "role": "development_regression"}
    result = aggregate([success, failure, development])
    assert result["attempted_count"] == 2 and result["complete_physical_closure_rate"] == .05
    assert result["observed_official_success_rate"] == .1 and not result["all_twenty_attempted"]
    assert result["cohorts"]["known1_to10"]["complete_physical_closure_count"] == 1
    assert result["cohorts"]["known11_to20"]["complete_physical_closure_count"] == 0
    assert result["accepted_trajectories_per_minute"] == 1.
    invalid_data = success | {"raw_recording_check": {"status": "failed"}}
    assert aggregate([invalid_data])["quality_accepted_trajectory_count"] == 0
    try: aggregate([success, success])
    except ValueError: pass
    else: raise AssertionError("Duplicate attempt accepted")
    for case, report in ((records[0], {"status": "passed", "official_200_step_complete": True}),
                         (records[21], {"status": "passed", "retain_right_control_verified": True}),
                         (records[22], {"status": "passed"})):
        result = outcome(case, report, [], True, 0)
        assert not result.get("complete_physical_closure") and not result.get("negative_control_verified") and not result.get("validation_passed")
    with tempfile.TemporaryDirectory(prefix="tig-near-support-contract-") as td:
        marker = Path(td) / "launch-once.json"; marker.open("x").close()
        try: marker.open("x")
        except FileExistsError: pass
        else: raise AssertionError("A second launch was accepted")
    return {"status": "passed", "environment_created": False, "physics_steps": 0,
        "checks": ["24 fixed single-attempt cases and exact original native arguments", "positive1..20 fixed denominator and two known cohorts",
            "development/control/replay excluded", "all-attempt wall cost and raw-quality yield", "duplicate/type/range rejection",
            "timeout cannot count as closure/control/replay success", "exclusive launch marker"]}


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
