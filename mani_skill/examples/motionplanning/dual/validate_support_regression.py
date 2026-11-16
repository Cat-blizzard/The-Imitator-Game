"""Frozen single-attempt regression for the independently reviewed support fix.

Only held_collision.py and table_support_collision.py may differ from the old
19-file execution snapshot. Positives 0..10, one retain-right negative, and
state/action replay of this batch's seed0 positive are separate observations.
All old data, the other 17 execution files, core11, limits and clocks are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from .validate_compact_seeds import atomic_json, classify, evidence, read_ticks, truth

ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent
DEV = ROOT / "infrastructure/development/tig-dual-arm"
PYTHON = ROOT / "infrastructure/envs/tig-official/bin/python"
OLD = ROOT / "infrastructure/validation/dual-arm/sync-compact-seeds-20261115T155130470922Z"
DIAGNOSTICS = (ROOT / "infrastructure/validation/dual-arm/support-geometry-20261115T160722994022Z",
               ROOT / "infrastructure/validation/dual-arm/support-geometry-20261116T000715Z")
GEOMETRY_AUDIT = ROOT / "infrastructure/validation/dual-arm/support-diagnostic-input-audit-20261116.json"
SUPERSEDED_BATCH = ROOT / "infrastructure/validation/dual-arm/support-regression-20261115T163412631092Z"
ALLOWED_CHANGES = frozenset(("held_collision.py", "table_support_collision.py"))
REPLAY_AUX = ("validate_compact_replay.py", "compact_replay_metrics.py", "fixed_root_restore.py",
              "validate_sync_replay.py", "sync_replay_metrics.py", "replay_metrics.py", "validate_replay.py")
GPU = "GPU-1f8a72b4-0821-9275-e36f-49ab74f13572"
ENVIRONMENT = {"PYTHONPATH": str(DEV), "MS_ASSET_DIR": str(ROOT / "infrastructure/maniskill"), "CUDA_VISIBLE_DEVICES": GPU}
LEVEL_KEYS = ("MANI_SKILL_L1", "MANI_SKILL_L2", "MANI_SKILL_L3")
TIMEOUT = 300


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def tree_manifest(roots: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): sha(path) for root in roots for path in sorted(root.rglob("*")) if path.is_file()}


def validate_changes(old: dict[str, str], new: dict[str, str]) -> list[str]:
    if set(old) != set(new) or len(old) != 19:
        raise RuntimeError("Expected the exact old19 execution file set")
    changed = sorted(name for name in old if old[name] != new[name])
    if set(changed) != ALLOWED_CHANGES:
        raise RuntimeError(f"Expected only the two reviewed support files to change, got {changed}")
    return changed


def native_command(seed: int, mode: str, output: Path) -> list[str]:
    if type(seed) is not int or seed not in range(11) or mode not in ("positive", "retain-right"):
        raise ValueError("Unknown preregistered native case")
    if mode == "retain-right" and seed != 0: raise ValueError("Only seed0 negative is preregistered")
    return [str(PYTHON), "-m", "mani_skill.examples.motionplanning.dual.validate_sync_compact",
            "--mode", mode, "--seed", str(seed), "--route", "merge-lift", "--execution-profile", "execution",
            "--joint-vel-limit", "1.5", "--joint-acc-limit", "2.6", "--approach-vel-limit", "0.9",
            "--approach-acc-limit", "0.9", "--max-episode-steps", "200", "--output-dir", str(output)]


def cases(batch: Path) -> list[dict[str, Any]]:
    records = [{"id": f"positive-seed-{seed:02d}", "kind": "native", "mode": "positive", "seed": seed,
                "role": "development_regression" if seed == 0 else "matched_old_denominator",
                "output": str(batch / f"positive-seed-{seed:02d}"),
                "command": native_command(seed, "positive", batch / f"positive-seed-{seed:02d}")} for seed in range(11)]
    records.append({"id": "negative-seed-00", "kind": "native", "mode": "retain-right", "seed": 0,
                    "role": "fault_detection_control", "output": str(batch / "negative-seed-00"),
                    "command": native_command(0, "retain-right", batch / "negative-seed-00")})
    for mode, policy in (("state", "full"), ("action", "fixed-root-exact")):
        output = batch / f"replay-{mode}"
        command = [str(PYTHON), "-m", "mani_skill.examples.motionplanning.dual.validate_compact_replay",
                   "--mode", mode, "--recording-kind", "compact", "--restore-policy", policy,
                   "--input-dir", str(batch / "positive-seed-00"), "--output-dir", str(output)]
        records.append({"id": f"replay-{mode}", "kind": "replay", "mode": mode, "restore_policy": policy,
                        "seed": 0, "role": "seed0_recording_integrity", "output": str(output), "command": command})
    return records


def prepare() -> Path:
    old_protocol = json.loads((OLD / "protocol.json").read_text())
    old19 = old_protocol["source19_sha256"]
    new19 = {name: sha(HERE / name) for name in old19}
    changed = validate_changes(old19, new19)
    core = old_protocol["core_source_sha256"]
    if {name: sha(DEV / name) for name in core} != core: raise RuntimeError("Frozen core11 changed")
    if {key: os.environ.get(key) for key in LEVEL_KEYS} != old_protocol["task_level_environment"]:
        raise RuntimeError("Task level environment differs from original L0 batch")
    stage = json.loads(Path(old_protocol["reference_stage"]).read_text())
    aux = {f"mani_skill/examples/motionplanning/dual/{name}": sha(HERE / name) for name in REPLAY_AUX}
    previous_replay = stage["replay"]["state"]["source_sha256"]
    if any(previous_replay[name] != value for name, value in aux.items()): raise RuntimeError("Frozen replay helpers changed")
    helper = HERE / "validate_compact_seeds.py"
    if sha(helper) != old_protocol["runner_sha256"]: raise RuntimeError("Old batch runner helper changed")
    aux["mani_skill/examples/motionplanning/dual/validate_compact_seeds.py"] = sha(helper)
    geometry_identities = json.loads(GEOMETRY_AUDIT.read_text())["geometry_asset_and_builder_identity_first_frozen_for_this_diagnosis"]
    assets = {path: value["sha256"] if isinstance(value, dict) else value for path, value in geometry_identities.items()}
    if not assets or {path: sha(Path(path)) for path in assets} != assets:
        raise RuntimeError("Geometry assets/builders differ from diagnostic identity")
    superseded = json.loads((SUPERSEDED_BATCH / "status.json").read_text())
    if superseded["status"] != "not_run_superseded_before_launch" or superseded["native_attempts"] != 0:
        raise RuntimeError("Unexpected superseded preparation status")
    history = tree_manifest((OLD, *DIAGNOSTICS))
    batch = ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime("support-regression-%Y%m%dT%H%M%S%fZ")
    batch.mkdir(exist_ok=False)
    snapshot = batch / "source"; snapshot.mkdir()
    for name in (*new19, *REPLAY_AUX, "validate_compact_seeds.py", Path(__file__).name):
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    previous = batch / "old-two-source"; previous.mkdir()
    for name in sorted(ALLOWED_CHANGES):
        original = OLD / "source" / name
        if sha(original) != old19[name]: raise RuntimeError("Old support snapshot hash mismatch")
        (previous / name).write_bytes(original.read_bytes())
    atomic_json(batch / "historical-manifest.json", history)
    versions = {}
    for name in ("mani_skill", "gymnasium", "sapien", "mplib", "numpy", "torch", "h5py", "imageio", "trimesh"):
        try: versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError: versions[name] = "distribution metadata unavailable"
    protocol = {"schema_version": 1, "prepared_at": now(), "status_interpretation": "batch completion is administrative, outcomes remain separate",
        "purpose": "single-attempt matched-seed regression of independently reviewed support implementation",
        "positive_seeds": list(range(11)), "development_seed": 0, "matched_evaluation_seeds": list(range(1, 11)),
        "matched_evaluation_denominator": 10, "attempts_per_case": 1, "retry_allowed": False,
        "per_seed_parameter_adjustment_allowed": False, "new_parameter_search_allowed": False,
        "failure_retention": "all attempted native/replay failures remain; engineering failures stay in positive denominator",
        "ordered_cases": cases(batch), "case_timeout_seconds": TIMEOUT,
        "task": old_protocol["task"], "execution_configuration": old_protocol["execution_configuration"],
        "source19_old_sha256": old19, "source19_new_sha256": new19, "allowed_changed_files": sorted(ALLOWED_CHANGES),
        "actually_changed_files": changed, "core11_sha256": core, "auxiliary_frozen_source_sha256": aux,
        "asset_builder_sha256": assets, "asset_builder_identity_source": str(GEOMETRY_AUDIT),
        "asset_builder_identity_source_sha256": sha(GEOMETRY_AUDIT),
        "old_batch_asset_identity_limitation": "original ten-seed batch did not freeze assets; present identities were first frozen during support diagnosis and are now checked before and after every regression case",
        "superseded_before_launch": {"batch": str(SUPERSEDED_BATCH), "status": superseded["status"],
            "status_file": str(SUPERSEDED_BATCH / "status.json"), "status_file_sha256": sha(SUPERSEDED_BATCH / "status.json"),
            "protocol": str(SUPERSEDED_BATCH / "protocol.json"), "protocol_sha256": sha(SUPERSEDED_BATCH / "protocol.json"),
            "native_attempts": 0, "counted_as_native_failure": False},
        "support_semantics_revision": {
            "authorized_change": "independent vertex/tabletop geometry fallback and whole-object support across all convex pieces",
            "top_geometry_fallback_tolerance_m": 1e-6, "fallback_requires_zero_contact_normal": False,
            "support_identity": "whole held object and all its convex pieces, rather than only initially contacting shape pairs",
            "permanent_release_rule": "once the entire object leaves support, later contacts are blocked",
            "retained_support_limits": {"upward_band_m": .003, "downward_translation_floor_m": -1e-6,
                "xy_displacement_m": .001, "rotation_rad": .01, "penetration_limit": "initial whole-object deepest penetration + 0.0002 m"},
            "candidate_queries_mutate_support_state": False, "physical_collision_masks_changed": False,
            "nonconvex_primitive_compatibility": "explicit legacy exact-pair registration, refresh and allowance methods retained; no exception-swallowing fallback",
            "interpretation": "planner configuration and clocks are unchanged; support collision semantics are intentionally revised and require this new regression"},
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha(Path(__file__)),
        "historical_roots": [str(path) for path in (OLD, *DIAGNOSTICS)], "historical_manifest": str(batch / "historical-manifest.json"),
        "historical_manifest_sha256": sha(batch / "historical-manifest.json"), "historical_file_count": len(history),
        "old_protocol_sha256": sha(OLD / "protocol.json"), "old_protocol": str(OLD / "protocol.json"),
        "old_source_snapshot": str(previous), "new_source_snapshot": str(snapshot),
        "gpu_uuid": GPU, "environment": ENVIRONMENT, "task_level_environment": old_protocol["task_level_environment"],
        "python_executable": str(PYTHON), "python_version": platform.python_version(), "package_versions": versions,
        "physics_backend": "cpu", "render_backend": "gpu", "control_hz": 20, "physics_hz": 100, "max_episode_steps": 200,
        "native_time_limit_policy": "unchanged: stop at truncation, full closure before tick200",
        "native_success_policy": "unchanged: original geometric success signal alone does not establish release/closure; safety-stopped runs retained",
        "replay_policy": "new seed0 positive metadata/source19 generated by unchanged native entry; old loader/input/index not rewritten; replay only if new seed0 is passed",
        "source_violation_policy": "stop subsequent launches; preserve attempted artifacts and receipt",
        "planning_wall_seconds": None, "planning_wall_time_status": "not collected by original native entry; total subprocess wall is not planning time"}
    atomic_json(batch / "protocol.json", protocol)
    return batch


def check_frozen(protocol: dict[str, Any], protocol_path: Path, protocol_sha: str) -> dict[str, Any]:
    if sha(protocol_path) != protocol_sha or sha(Path(__file__)) != protocol["runner_sha256"]:
        raise RuntimeError("Protocol or runner changed after freezing")
    identity = {name: sha(HERE / name) for name in protocol["source19_new_sha256"]}
    if identity != protocol["source19_new_sha256"]: raise RuntimeError("Reviewed execution source19 changed")
    core = {name: sha(DEV / name) for name in protocol["core11_sha256"]}
    aux = {name: sha(DEV / name) for name in protocol["auxiliary_frozen_source_sha256"]}
    if core != protocol["core11_sha256"] or aux != protocol["auxiliary_frozen_source_sha256"]:
        raise RuntimeError("Core11 or frozen replay/helper source changed")
    assets = {path: sha(Path(path)) for path in protocol["asset_builder_sha256"]}
    if assets != protocol["asset_builder_sha256"]:
        raise RuntimeError("Frozen geometry asset or builder changed")
    manifest = Path(protocol["historical_manifest"])
    if sha(manifest) != protocol["historical_manifest_sha256"]: raise RuntimeError("Historical manifest changed")
    history = tree_manifest(tuple(Path(path) for path in protocol["historical_roots"]))
    if history != json.loads(manifest.read_text()): raise RuntimeError("Original history changed")
    return {"source19_sha256": identity, "core11_sha256": core, "auxiliary_source_sha256": aux,
            "asset_builder_sha256": assets,
            "historical_file_count": len(history), "historical_manifest_sha256": sha(manifest), "original_history_unchanged": True}


def replay_evidence(output: Path, source: Path, mode: str) -> dict[str, Any]:
    import h5py
    import numpy as np
    try:
        with h5py.File(source / "trajectory.h5") as original, h5py.File(output / "observed-states.h5") as observed:
            group = original["traj_0"]
            actions = group["actions"][:]
            expected_actions = actions if mode == "action" else np.empty((0, 16), dtype=np.float32)
            action_match = np.array_equal(observed["applied_actions"][:], expected_actions)
            state_checks = {}
            def visit(name: str, item: Any) -> None:
                if not isinstance(item, h5py.Dataset): return
                actual = observed["env_states/" + name][:]
                state_checks[name] = {"expected_shape": list(item.shape), "actual_shape": list(actual.shape),
                                     "all_values_finite": bool(np.isfinite(actual).all()), "all_values_exactly_equal": bool(np.array_equal(actual, item[:]))}
            group["env_states"].visititems(visit)
            flags = {name: bool(np.array_equal(observed[f"native_flags/{name}"][:], group[name][:] if mode == "action" else np.empty(0, dtype=bool)))
                     for name in ("success", "terminated", "truncated")}
            complete = action_match and len(state_checks) == 7 and all(s["expected_shape"] == s["actual_shape"] and s["all_values_finite"] for s in state_checks.values()) and all(flags.values())
            return {"status": "passed" if complete else "failed", "applied_actions_exactly_match": action_match,
                    "state_leaves": state_checks, "all7_states_exactly_equal": all(x["all_values_exactly_equal"] for x in state_checks.values()),
                    "native_flags_match": flags, "interpretation": "state exactness is reported independently of the unchanged native replay tolerance gates"}
    except Exception as exc: return {"status": "unavailable_or_invalid", "error": f"{type(exc).__name__}: {exc}"}


def outcome(case: dict[str, Any], report: dict[str, Any], ticks: list[dict[str, Any]], timed_out: bool, exit_code: int) -> dict[str, Any]:
    if case["kind"] == "replay":
        return {"reported_status": report.get("status"), "validation_passed": bool(report.get("status") == "passed" and exit_code == 0 and not timed_out),
                "checks": report.get("checks"), "outcome": report.get("outcome"), "sequence": report.get("sequence"),
                "classification": "replay_passed" if report.get("status") == "passed" and exit_code == 0 else "replay_failed"}
    result = {"reported_status": report.get("status"), "total_environment_steps": report.get("total_environment_steps"),
        "first_official_success_step": report.get("first_official_success_step"),
        "observed_official_success": any(truth(t.get("official_evaluation", {}).get("success")) for t in ticks),
        "complete_physical_closure": bool(case["mode"] == "positive" and report.get("official_200_step_complete") is True and report.get("status") == "passed" and exit_code == 0 and not timed_out),
        "native_truncation_observed": any(t.get("native_truncated") for t in ticks),
        "classification": classify(report, ticks, timed_out, exit_code), "error_type": report.get("error_type"), "error": report.get("error"),
        "reason": report.get("reason"), "phase_step_counts": report.get("phase_step_counts"),
        "failed_phases": [{k: stage.get(k) for k in ("name", "status", "reason")} for stage in report.get("stages", []) if stage.get("status") != "passed"]}
    if case["mode"] == "retain-right":
        result["negative_control_verified"] = bool(report.get("retain_right_control_verified") is True and report.get("status") == "passed" and exit_code == 0)
        result["classification"] = "expected_fault_detected" if result["negative_control_verified"] else "negative_control_failed"
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [r for r in rows if r["role"] == "matched_old_denominator" and r.get("attempted")]
    official = sum(r["outcome"]["observed_official_success"] for r in matched)
    full = sum(r["outcome"]["complete_physical_closure"] for r in matched)
    return {"matched_old_denominator": 10, "matched_cases_attempted": len(matched), "observed_official_success_count": official,
        "observed_official_success_rate": official / 10, "complete_physical_closure_count": full, "complete_physical_closure_rate": full / 10,
        "matched_classifications": dict(Counter(r["outcome"]["classification"] for r in matched)),
        "development_seed0_excluded_from_rates": True, "planning_wall_seconds": None}


def run(batch: Path) -> int:
    protocol_path = batch / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha(protocol_path)
    if protocol["ordered_cases"] != cases(batch) or protocol["case_timeout_seconds"] != TIMEOUT or protocol["attempts_per_case"] != 1 or protocol["retry_allowed"]:
        raise RuntimeError("Preregistered case order/budget mismatch")
    if (batch / "summary.json").exists(): raise RuntimeError("Batch already started; resume/retry prohibited")
    check_frozen(protocol, protocol_path, protocol_sha)
    env = os.environ.copy(); env.update(ENVIRONMENT)
    for key, value in protocol["task_level_environment"].items():
        if value is None: env.pop(key, None)
        else: env[key] = value
    summary: dict[str, Any] = {"status": "running", "status_interpretation": "administrative completion only; outcome evidence remains separate",
                              "started_at": now(), "protocol": str(protocol_path), "protocol_sha256": protocol_sha, "cases": [], "aggregate": aggregate([])}
    atomic_json(batch / "summary.json", summary)
    try:
        for case in protocol["ordered_cases"]:
            before = check_frozen(protocol, protocol_path, protocol_sha)
            output = Path(case["output"])
            if output.exists(): raise RuntimeError("Output exists before its sole allowed attempt")
            if case["kind"] == "replay":
                first = summary["cases"][0]
                if not first["outcome"]["complete_physical_closure"]:
                    row = {**case, "attempted": False, "status": "skipped_seed0_positive_not_passed", "completed_at": now()}
                    summary["cases"].append(row); atomic_json(batch / "summary.json", summary); continue
            row: dict[str, Any] = {**case, "attempted": True, "attempt": 1, "started_at": now(), "source_before": before}
            receipt = batch / f"{case['id']}-receipt.json"
            log = batch / f"{case['id']}.log"
            row.update(log=str(log), cwd=str(DEV), environment=ENVIRONMENT)
            atomic_json(receipt, row)
            start = time.monotonic(); timed_out = False
            with log.open("wb") as stream:
                process = subprocess.Popen(case["command"], cwd=DEV, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                row["pid"] = process.pid
                try: exit_code = process.wait(timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    timed_out = True; os.killpg(process.pid, signal.SIGTERM)
                    try: exit_code = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); exit_code = process.wait()
            row.update(exit_code=exit_code, timed_out=timed_out, wall_seconds=time.monotonic() - start, completed_at=now())
            atomic_json(receipt, row)
            row["source_after"] = check_frozen(protocol, protocol_path, protocol_sha)
            report_path = output / "report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            ticks = read_ticks(output / "ticks.jsonl")
            violations = []
            if report:
                if case["kind"] == "native":
                    if report.get("source_sha256") != protocol["source19_new_sha256"]: violations.append("native source identity differs")
                    if report.get("execution_configuration") != protocol["execution_configuration"]: violations.append("native config differs")
                elif report.get("input_execution_source_sha256") != protocol["source19_new_sha256"]:
                    violations.append("replay input source identity differs")
            row.update(report=str(report_path), report_sha256=sha(report_path) if report_path.exists() else None,
                       outcome=outcome(case, report, ticks, timed_out, exit_code), artifact_errors=report.get("artifact_errors"), protocol_violations=violations)
            row["raw_recording_check"] = evidence(output, report, ticks) if case["kind"] == "native" else replay_evidence(output, batch / "positive-seed-00", case["mode"])
            atomic_json(receipt, row)
            summary["cases"].append(row); summary["aggregate"] = aggregate(summary["cases"])
            atomic_json(batch / "summary.json", summary)
            print(json.dumps({"case": case["id"], "exit_code": exit_code, "outcome": row["outcome"], "raw_recording_check": row["raw_recording_check"]["status"]}), flush=True)
            if violations: raise RuntimeError("Protocol violation: " + "; ".join(violations))
        summary.update(status="completed", completed_at=now(), all_preregistered_native_cases_attempted_once=True)
        atomic_json(batch / "summary.json", summary)
        return 0
    except Exception as exc:
        summary.update(status="stopped_protocol_or_runner_error", stopped_at=now(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(batch / "summary.json", summary)
        raise


def self_test() -> dict[str, Any]:
    original = {f"file{i}.py": str(i) for i in range(17)} | {name: "old" for name in ALLOWED_CHANGES}
    changed = original | {name: "new" for name in ALLOWED_CHANGES}
    assert set(validate_changes(original, changed)) == ALLOWED_CHANGES
    for invalid in (original, changed | {"file0.py": "tamper"}):
        try: validate_changes(original, invalid)
        except RuntimeError: pass
        else: raise AssertionError("Invalid source drift accepted")
    records = cases(Path("/tmp/support-regression-selftest"))
    assert len(records) == 14 and [c["seed"] for c in records[:11]] == list(range(11))
    assert records[11]["mode"] == "retain-right" and records[-1]["restore_policy"] == "fixed-root-exact"
    assert records[-2]["restore_policy"] == "full"
    for c in records[:12]:
        cmd = c["command"]
        assert cmd[cmd.index("--max-episode-steps") + 1] == "200"
        assert cmd[cmd.index("--joint-acc-limit") + 1] == "2.6"
    positive = records[1] | {"attempted": True, "outcome": {"observed_official_success": True, "complete_physical_closure": True, "classification": "complete"}}
    dev = records[0] | {"attempted": True, "outcome": positive["outcome"]}
    assert aggregate([positive, dev])["complete_physical_closure_rate"] == .1
    negative = outcome(records[11], {"status": "passed", "retain_right_control_verified": True}, [], False, 0)
    assert negative["negative_control_verified"] and not negative["complete_physical_closure"]
    with tempfile.TemporaryDirectory(prefix="tig-support-regression-selftest-") as td:
        p = Path(td); (p / "evidence").write_text("preserve")
        first = tree_manifest((p,)); (p / "evidence").write_text("changed")
        assert tree_manifest((p,)) != first
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
        "checks": ["exact two-file change whitelist", "14 ordered single-attempt cases and unchanged native arguments",
                   "development seed excluded from fixed denominator", "expected negative separated from physical closure", "historical mutation detection"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--self-test", action="store_true")
    choice.add_argument("--prepare-only", action="store_true")
    choice.add_argument("--batch-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    if Path(sys.executable).resolve() != PYTHON.resolve(): parser.error("Use the frozen official Python")
    if args.prepare_only:
        batch = prepare(); print(json.dumps({"batch": str(batch), "protocol": str(batch / "protocol.json")}), flush=True); return 0
    return run(args.batch_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
