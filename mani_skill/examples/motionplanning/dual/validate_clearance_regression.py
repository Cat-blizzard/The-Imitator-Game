"""Frozen regression of shorter clearance under the same physical acceptance gates.

The previous timing-fix batch is retained, including its seed6 native timeout.
Only the compact validation entry may change; all18 shared execution sources,
core11, assets, clocks and physical acceptance gates remain fixed. Each positive
seed0..10, the retain-right control, and seed0 state/action replay run once.
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
OLD = ROOT / "infrastructure/validation/dual-arm/timing-regression-20261115T170213786951Z"
PRECHANGE = ROOT / "infrastructure/validation/dual-arm/clearance-fix-prechange-20261116.json"
ALLOWED_CHANGES = frozenset(("validate_sync_compact.py",))
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
        raise RuntimeError(f"Expected only the reviewed compact entry to change, got {changed}")
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


def configuration() -> dict[str, Any]:
    from .validate_sync_compact import compact_configuration
    args = argparse.Namespace(route="merge-lift", execution_profile="execution", max_episode_steps=200,
        joint_vel_limit=1.5, joint_acc_limit=2.6, approach_vel_limit=.9, approach_acc_limit=.9)
    config = compact_configuration(args)
    required = {"preparation_steps": 10, "min_actual_height_gain_m": [.10, .10],
        "release_stable_steps": 3, "retreat_command_m": .08, "final_common_placement_steps": 20,
        "max_episode_steps": 200, "official_budget_candidate": True,
        "joint_vel_limit_rad_s": 1.5, "joint_acc_limit_rad_s2": 2.6,
        "approach_joint_vel_limit_rad_s": .9, "approach_joint_acc_limit_rad_s2": .9,
        "solver_limit_fraction": .99, "control_hz": 20, "physics_hz": 100,
        "source_timestamp_scaling": True, "airborne_hover_steps": 0,
        "transport_clearance_above_release_m": .06,
        "minimum_commanded_lift_m": .12,
        "route_geometry_policy": "compact-clearance-v2"}
    if any(config.get(key) != value for key, value in required.items()):
        raise RuntimeError("Clearance revision changed an acceptance gate, limit or clock")
    return config


def protected_history(guard: dict[str, Any]) -> dict[str, str]:
    """Normalize the read-only guard's per-file digest/size records."""
    entries = guard["all_existing_raw_history_manifest"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("A nonempty prechange file manifest is required")
    result = {}
    for name, item in entries.items():
        digest = item.get("sha256") if isinstance(item, dict) else item
        if (not isinstance(name, str) or not Path(name).is_absolute()
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Expected absolute filenames and SHA256 hex digests")
        result[name] = digest
    return result


def prepare() -> Path:
    old_protocol = json.loads((OLD / "protocol.json").read_text())
    guard = json.loads(PRECHANGE.read_text())
    old19 = old_protocol["source19_new_sha256"]
    if old19 != guard["baseline19_sha256"] or sha(OLD / "protocol.json") != guard["baseline_protocol_sha256"]:
        raise RuntimeError("Reviewed timing baseline identity mismatch")
    new19 = {name: sha(HERE / name) for name in old19}
    changed = validate_changes(old19, new19)
    core, assets = old_protocol["core11_sha256"], old_protocol["asset_builder_sha256"]
    aux = dict(old_protocol["auxiliary_frozen_source_sha256"])
    aux[str(Path(old_protocol["runner_path"]).relative_to(DEV))] = old_protocol["runner_sha256"]
    if {name: sha(DEV / name) for name in core} != core:
        raise RuntimeError("Frozen core11 changed")
    if {name: sha(DEV / name) for name in aux} != aux:
        raise RuntimeError("Frozen replay/helper or previous runner changed")
    if {name: sha(Path(name)) for name in assets} != assets:
        raise RuntimeError("Frozen geometry assets/builders changed")
    if {key: os.environ.get(key) for key in LEVEL_KEYS} != old_protocol["task_level_environment"]:
        raise RuntimeError("Task level environment differs from original L0 batch")
    history = protected_history(guard) | {str(PRECHANGE): sha(PRECHANGE)}
    if {name: sha(Path(name)) for name in history} != history:
        raise RuntimeError("Prechange protected history changed")
    config = configuration()
    old_config = old_protocol["execution_configuration"]
    config_changes = {key: {"old": old_config.get(key), "new": config.get(key)}
        for key in sorted(old_config.keys() | config.keys()) if old_config.get(key) != config.get(key)}
    if set(config_changes) != {"transport_clearance_above_release_m", "minimum_commanded_lift_m", "route_geometry_policy"}:
        raise RuntimeError("Clearance revision changed parameters outside its three explicit route metadata fields")
    batch = ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime("clearance-regression-%Y%m%dT%H%M%S%fZ")
    batch.mkdir(exist_ok=False)
    snapshot = batch / "source"; snapshot.mkdir()
    for name in (*new19, *REPLAY_AUX, "validate_compact_seeds.py", "validate_support_regression.py", "validate_timing_regression.py", Path(__file__).name):
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    previous = batch / "previous-entry-source"; previous.mkdir()
    for name in ALLOWED_CHANGES:
        source = OLD / "source" / name
        if sha(source) != old19[name]: raise RuntimeError("Baseline entry snapshot mismatch")
        (previous / name).write_bytes(source.read_bytes())
    atomic_json(batch / "historical-manifest.json", history)
    versions = {}
    for name in ("mani_skill", "gymnasium", "sapien", "mplib", "toppra", "scipy", "numpy", "torch", "h5py", "imageio", "trimesh"):
        try: versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError: versions[name] = "distribution metadata unavailable"
    protocol = {"schema_version": 3, "prepared_at": now(),
        "purpose": "matched known-seed regression of one uniform 60mm transport clearance; original 120mm lift floor, placement recipe, retiming, clocks and physical gates retained",
        "interpretation": "Known seeds were available during diagnosis; this is regression evidence, not unseen-seed generalization or BPP performance",
        "positive_seeds": list(range(11)), "development_seed": 0, "matched_evaluation_seeds": list(range(1, 11)),
        "matched_evaluation_denominator": 10, "attempts_per_case": 1, "retry_allowed": False,
        "per_seed_parameter_adjustment_allowed": False, "new_parameter_search_allowed": False,
        "failure_retention": "all attempted failures remain, including engineering failures in the positive denominator",
        "ordered_cases": cases(batch), "case_timeout_seconds": TIMEOUT,
        "task": old_protocol["task"], "execution_configuration": config,
        "previous_execution_configuration": old_config, "configuration_changes": config_changes,
        "route_change": {"previous_above_release_m_hardcoded_in_source": .08,
            "new_above_release_m": .06, "minimum_commanded_lift_m_unchanged": .12,
            "final_release_offsets_unchanged": True,
            "interpretation": "new fields expose the formerly hardcoded clearance/floor; only above-release clearance changes behavior"},
        "source19_old_sha256": old19, "source19_new_sha256": new19,
        "allowed_changed_files": sorted(ALLOWED_CHANGES), "actually_changed_files": changed,
        "core11_sha256": core, "auxiliary_frozen_source_sha256": aux,
        "asset_builder_sha256": assets, "asset_builder_identity_source": old_protocol["asset_builder_identity_source"],
        "old_batch_asset_identity_limitation": old_protocol["old_batch_asset_identity_limitation"],
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha(Path(__file__)),
        "historical_manifest": str(batch / "historical-manifest.json"),
        "historical_manifest_sha256": sha(batch / "historical-manifest.json"),
        "historical_file_count": len(history), "history_protection": "all listed prechange files must remain present with identical bytes; new artifact files are allowed",
        "baseline_batch": str(OLD), "old_protocol": str(OLD / "protocol.json"), "old_protocol_sha256": sha(OLD / "protocol.json"),
        "prechange_guard": str(PRECHANGE), "prechange_guard_sha256": sha(PRECHANGE),
        "old_source_snapshot": str(previous), "new_source_snapshot": str(snapshot),
        "old_source_snapshot_sha256": {name: sha(previous / name) for name in ALLOWED_CHANGES},
        "new_source_snapshot_sha256": {item.name: sha(item) for item in snapshot.iterdir()},
        "gpu_uuid": GPU, "environment": ENVIRONMENT, "task_level_environment": old_protocol["task_level_environment"],
        "python_executable": str(PYTHON), "python_version": platform.python_version(), "package_versions": versions,
        "physics_backend": "cpu", "render_backend": "gpu", "control_hz": 20, "physics_hz": 100, "max_episode_steps": 200,
        "native_time_limit_policy": "unchanged: stop at native truncation; complete all physical gates by tick199",
        "native_success_policy": old_protocol["native_success_policy"],
        "acceptance_contract": guard["contract_to_preserve"],
        "source_curve_retiming": "common phase scale, t*=s, q unchanged, v/=s, a/=s^2; retain all raw native arrays and numeric analytic extrema; rebuild schedules and recheck collisions",
        "derivative_scope": "source TOPPRA curves only; neither physical actuator acceleration certification nor continuous collision proof",
        "replay_policy": "unchanged loaders and gates; new seed0 input only if full closure passed; old recordings and indexes are not overwritten",
        "source_violation_policy": "stop subsequent launches, preserve all attempted artifacts",
        "planning_wall_seconds": None, "planning_wall_time_status": "not independently instrumented; subprocess wall is not pure planning time"}
    atomic_json(batch / "protocol.json", protocol)
    return batch


def check_frozen(protocol: dict[str, Any], protocol_path: Path, protocol_sha: str) -> dict[str, Any]:
    if sha(protocol_path) != protocol_sha or sha(Path(__file__)) != protocol["runner_sha256"]:
        raise RuntimeError("Protocol or runner changed after freezing")
    identity = {name: sha(HERE / name) for name in protocol["source19_new_sha256"]}
    if identity != protocol["source19_new_sha256"]: raise RuntimeError("Reviewed execution source19 changed")
    core = {name: sha(DEV / name) for name in protocol["core11_sha256"]}
    aux = {name: sha(DEV / name) for name in protocol["auxiliary_frozen_source_sha256"]}
    assets = {name: sha(Path(name)) for name in protocol["asset_builder_sha256"]}
    if core != protocol["core11_sha256"] or aux != protocol["auxiliary_frozen_source_sha256"] or assets != protocol["asset_builder_sha256"]:
        raise RuntimeError("Core, replay helpers, previous runner or geometry assets changed")
    if configuration() != protocol["execution_configuration"]: raise RuntimeError("Frozen experiment configuration changed")
    for path_key, digest_key in (("new_source_snapshot", "new_source_snapshot_sha256"),
                                 ("old_source_snapshot", "old_source_snapshot_sha256")):
        directory = Path(protocol[path_key])
        if {item.name: sha(item) for item in directory.iterdir()} != protocol[digest_key]:
            raise RuntimeError("Frozen source snapshot changed")
    manifest = Path(protocol["historical_manifest"])
    if sha(manifest) != protocol["historical_manifest_sha256"]: raise RuntimeError("Historical manifest changed")
    history = json.loads(manifest.read_text())
    if {name: sha(Path(name)) for name in history} != history: raise RuntimeError("Protected original history changed")
    return {"source19_sha256": identity, "core11_sha256": core, "auxiliary_source_sha256": aux,
        "asset_builder_sha256": assets, "historical_file_count": len(history),
        "historical_manifest_sha256": sha(manifest), "original_history_unchanged": True}


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
                "classification": "replay_passed" if report.get("status") == "passed" and exit_code == 0 and not timed_out else "replay_failed"}
    result = {"reported_status": report.get("status"), "total_environment_steps": report.get("total_environment_steps"),
        "first_official_success_step": report.get("first_official_success_step"),
        "observed_official_success": any(truth(t.get("official_evaluation", {}).get("success")) for t in ticks),
        "complete_physical_closure": bool(case["mode"] == "positive" and report.get("official_200_step_complete") is True and report.get("status") == "passed" and exit_code == 0 and not timed_out),
        "native_truncation_observed": any(t.get("native_truncated") for t in ticks),
        "classification": classify(report, ticks, timed_out, exit_code), "error_type": report.get("error_type"), "error": report.get("error"),
        "reason": report.get("reason"), "phase_step_counts": report.get("phase_step_counts"),
        "failed_phases": [{k: stage.get(k) for k in ("name", "status", "reason")} for stage in report.get("stages", []) if stage.get("status") != "passed"]}
    if case["mode"] == "retain-right":
        result["negative_control_verified"] = bool(report.get("retain_right_control_verified") is True and report.get("status") == "passed" and exit_code == 0 and not timed_out)
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
    original = {f"file{i}.py": str(i) for i in range(18)} | {name: "old" for name in ALLOWED_CHANGES}
    changed = original | {name: "new" for name in ALLOWED_CHANGES}
    assert set(validate_changes(original, changed)) == ALLOWED_CHANGES
    for invalid in (original, changed | {"file0.py": "tamper"}):
        try: validate_changes(original, invalid)
        except RuntimeError: pass
        else: raise AssertionError("Invalid source drift accepted")
    records = cases(Path("/tmp/clearance-regression-selftest"))
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
    timeout_negative = outcome(records[11], {"status": "passed", "retain_right_control_verified": True}, [], True, 0)
    assert not timeout_negative["negative_control_verified"] and timeout_negative["classification"] == "negative_control_failed"
    timeout_replay = outcome(records[12], {"status": "passed"}, [], True, 0)
    assert not timeout_replay["validation_passed"] and timeout_replay["classification"] == "replay_failed"
    with tempfile.TemporaryDirectory(prefix="tig-clearance-regression-selftest-") as td:
        p = Path(td); (p / "evidence").write_text("preserve")
        first = tree_manifest((p,)); (p / "evidence").write_text("changed")
        assert tree_manifest((p,)) != first
        guard = {"all_existing_raw_history_manifest": {str(p / "evidence"): {"sha256": first[str(p / "evidence")], "size_bytes": 8}}}
        assert protected_history(guard) == first
        assert {name: sha(Path(name)) for name in protected_history(guard)} != first
    for manifest in ({}, {"relative": "0" * 64}, {"/absolute": {"sha256": "invalid"}}):
        try: protected_history({"all_existing_raw_history_manifest": manifest})
        except ValueError: pass
        else: raise AssertionError("Malformed prechange protection accepted")
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
        "checks": ["exact one-entry change whitelist", "14 ordered single-attempt cases and unchanged native arguments",
                   "development seed excluded from fixed denominator", "expected negative separated from physical closure", "historical mutation detection",
                   "digest/size guard normalization and malformed manifest rejection", "timeout cannot pass replay or fault detection"]}


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
