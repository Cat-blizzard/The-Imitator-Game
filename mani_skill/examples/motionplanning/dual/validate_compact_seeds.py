"""One preregistered, fixed-parameter compact dual-arm attempt for seeds 1..10.

The runner never changes the frozen 19-file planner/executor or retries a seed.
Native observations are T paired 16-D actions and T+1 seven-entity state arrays.
Batch completion is administrative; physical and official outcomes remain separate.
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

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[7]
CHECKOUT = ROOT / "infrastructure/development/tig-dual-arm"
PYTHON = ROOT / "infrastructure/envs/tig-official/bin/python"
REFERENCE = ROOT / "infrastructure/validation/dual-arm/sync-compact-stage.json"
GPU = "GPU-1f8a72b4-0821-9275-e36f-49ab74f13572"
SEEDS = tuple(range(1, 11))
TIMEOUT = 300
MODULE = "mani_skill.examples.motionplanning.dual.validate_sync_compact"
ENVIRONMENT = {"PYTHONPATH": str(CHECKOUT), "MS_ASSET_DIR": str(ROOT / "infrastructure/maniskill"),
               "CUDA_VISIBLE_DEVICES": GPU}
LEVEL_KEYS = ("MANI_SKILL_L1", "MANI_SKILL_L2", "MANI_SKILL_L3")
CORE_FILES = ("mani_skill/envs/tasks/tabletop/dual_tasks/_008_pick_apple_banana_baskets.py",
    "mani_skill/envs/tasks/tabletop/utils/L0_L3_utils.py", "mani_skill/envs/sapien_env.py",
    "mani_skill/envs/scene.py", "mani_skill/utils/registration.py", "mani_skill/utils/structs/actor.py",
    "mani_skill/utils/structs/articulation.py", "mani_skill/agents/controllers/base_controller.py",
    "mani_skill/agents/controllers/pd_joint_pos.py", "mani_skill/agents/robots/panda/panda_wristcam.py",
    "mani_skill/assets/robots/panda/panda_v3.urdf")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace only the runner's own current progress artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def command(seed: int, output: Path) -> list[str]:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only preregistered non-development seeds 1..10 are allowed")
    return [str(PYTHON), "-m", MODULE, "--mode", "positive", "--seed", str(seed),
            "--route", "merge-lift", "--execution-profile", "execution",
            "--joint-vel-limit", "1.5", "--joint-acc-limit", "2.6",
            "--approach-vel-limit", "0.9", "--approach-acc-limit", "0.9",
            "--max-episode-steps", "200", "--output-dir", str(output)]


def source_identity(expected: dict[str, str]) -> dict[str, str]:
    actual = {name: sha(HERE / name) for name in expected}
    if len(actual) != 19 or actual != expected:
        raise RuntimeError("Protocol violation: frozen 19-file native source identity changed")
    return actual


def prepare() -> Path:
    reference = json.loads(REFERENCE.read_text())
    expected = reference["positive"]["source19_sha256"]
    source_identity(expected)
    configuration = reference["execution_configuration"]
    assert configuration["route"] == "merge-lift" and configuration["execution_profile"] == "execution"
    assert configuration["joint_vel_limit_rad_s"] == 1.5 and configuration["joint_acc_limit_rad_s2"] == 2.6
    assert configuration["approach_joint_vel_limit_rad_s"] == configuration["approach_joint_acc_limit_rad_s2"] == .9
    assert configuration["max_episode_steps"] == 200
    gpu_observation = subprocess.run(["nvidia-smi", "--id=" + GPU,
        "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        check=True, text=True, stdout=subprocess.PIPE, timeout=20).stdout.strip()
    versions = {}
    for name in ("mani_skill", "gymnasium", "sapien", "mplib", "numpy", "torch", "h5py", "imageio", "trimesh"):
        try: versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError: versions[name] = "distribution metadata unavailable"
    batch = ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime("sync-compact-seeds-%Y%m%dT%H%M%S%fZ")
    batch.mkdir(exist_ok=False)
    snapshot = batch / "source"; snapshot.mkdir()
    for name in expected: (snapshot / name).write_bytes((HERE / name).read_bytes())
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    protocol = {"schema_version": 1, "prepared_at": now(), "scope": "fixed-parameter out-of-development-seed engineering validation",
        "seeds": list(SEEDS), "development_seed_excluded": 0, "attempts_per_seed": 1, "execution_order": "ascending, sequential independent subprocesses",
        "retry_allowed": False, "per_seed_adjustment_allowed": False, "parameter_search_allowed": False,
        "planned_denominator": 10, "engineering_failures_remain_in_denominator": True,
        "mode": "positive", "task": reference["task"], "execution_configuration": configuration,
        "max_episode_steps": 200, "control_hz": 20, "physics_hz": 100,
        "native_truncation_policy": "original native entry stops at truncation, including tick200; complete closure must finish before200",
        "native_success_policy": "original success-only continuation retained to check release, retreat, common20 settling",
        "per_seed_subprocess_timeout_seconds": TIMEOUT, "physics_backend": "cpu", "render_backend": "gpu",
        "environment": ENVIRONMENT, "gpu_uuid": GPU, "gpu_preflight": gpu_observation,
        "task_level_environment": {key: os.environ.get(key) for key in LEVEL_KEYS},
        "core_source_sha256": {name: sha(CHECKOUT / name) for name in CORE_FILES},
        "python_executable": str(PYTHON), "python_version": platform.python_version(), "package_versions": versions,
        "reference_stage": str(REFERENCE), "reference_stage_sha256": sha(REFERENCE),
        "reference_seed0_report": reference["positive"]["report"], "source19_sha256": expected,
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha(Path(__file__)),
        "source_snapshot": str(snapshot), "commands": {str(s): command(s, batch / f"seed-{s:02d}") for s in SEEDS},
        "protocol_violation_policy": "stop subsequent launches and retain all attempted results if source/configuration identity changes",
        "metrics": ["observed official success signal /10", "complete physical closure under original gates /10",
                    "native truncation, subprocess timeout, candidate and measured collision, grasp/support initialization, planner and engineering error categories"],
        "planning_wall_seconds": None, "planning_wall_time_status": "not_collected_by_frozen_native_entry; total subprocess wall is not planning wall",
        "collision_limitations": "finite candidate joint samples and measured control-step checks; not CCD or continuous collision safety",
        "acceptance_configuration": "all frozen entry gates unchanged; execution profile already omits the separate airborne diagnostic hover"}
    atomic_json(batch / "protocol.json", protocol)
    return batch


def truth(value: Any) -> bool:
    if isinstance(value, list): return bool(value) and all(truth(x) for x in value)
    return value is True


def read_ticks(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify(report: dict[str, Any], ticks: list[dict[str, Any]], timed_out: bool, exit_code: int | None) -> str:
    if timed_out: return "subprocess_timeout"
    stages = report.get("stages", [])
    failed = next((s for s in stages if s.get("status") != "passed"), {})
    reason = str(failed.get("reason", report.get("reason", "")))
    text = " ".join(str(report.get(k, "")) for k in ("stage", "error", "error_type", "reason")) + " " + reason
    if any(t.get("native_truncated") for t in ticks): return "native_time_limit"
    if reason == "candidate_collision": return "candidate_collision_rejected"
    if reason == "measured_collision": return "measured_collision_aborted"
    if reason == "initial_collision": return "initial_collision_rejected"
    if "support" in text.lower(): return "support_initialization_error"
    if failed.get("name") == "close_and_confirm" or "initial_grasp_or_attachment" in text: return "grasp_failure"
    if any(x in text.lower() for x in ("planning failed", "planner", "topp", "ik failed", "trajectory", "screw")) and report.get("error"):
        return "planning_error"
    if report.get("official_200_step_complete") is True and report.get("status") == "passed" and exit_code == 0: return "complete"
    if report.get("error") or report.get("artifact_errors") or not report or exit_code not in (0, 1): return "engineering_error"
    return "physical_gate_failure"


def evidence(output: Path, report: dict[str, Any], ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Inspect raw T-actions/T+1-state arrays without modifying native outputs."""
    import h5py
    import numpy as np
    result: dict[str, Any] = {"status": "unavailable", "tick_count": len(ticks), "state_leaf_count": 0}
    path = output / "trajectory.h5"
    if not path.exists(): return result
    try:
        with h5py.File(path, "r") as h5:
            group = h5["traj_0"]
            actions = group["actions"][()]
            leaves: dict[str, int] = {}
            group["env_states"].visititems(lambda name, obj: leaves.update({name: len(obj)}) if isinstance(obj, h5py.Dataset) else None)
            keys = report.get("action_key_order", [])
            submitted = np.asarray([np.concatenate([row["actions"][key] for key in keys]) for row in ticks])
            matches = actions.shape == (len(ticks), 16) and np.isfinite(actions).all() and np.array_equal(actions, submitted)
            flags = {}
            for name, field in (("terminated", "native_terminated"), ("truncated", "native_truncated"), ("success", None)):
                actual = group[name][()].reshape(-1).astype(bool)
                expected = np.asarray([truth(row["official_evaluation"].get("success")) if field is None else row[field] for row in ticks], dtype=bool)
                flags[name] = bool(np.array_equal(actual, expected))
            result.update(status="passed" if matches and len(leaves) == 7 and all(n == len(ticks) + 1 for n in leaves.values()) and all(flags.values()) else "failed",
                          actions_shape=list(actions.shape), submitted_actions_exactly_matched=bool(matches),
                          state_leaf_count=len(leaves), state_lengths=leaves, native_flags_match_ticks=flags)
    except Exception as exc: result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    official = sum(row["observed_official_success"] for row in rows)
    complete = sum(row["complete_physical_closure"] for row in rows)
    return {"planned_denominator": 10, "attempted": len(rows), "remaining": 10 - len(rows),
            "observed_official_success_count": official, "observed_official_success_rate": official / 10,
            "complete_physical_closure_count": complete, "complete_physical_closure_rate": complete / 10,
            "classification_counts": dict(counts),
            "native_truncation_count": sum(row["native_truncation_observed"] for row in rows),
            "subprocess_timeout_count": sum(row["subprocess_timed_out"] for row in rows),
            "candidate_collision_rejection_count": sum(row["candidate_collision_rejected"] for row in rows),
            "measured_collision_abort_count": sum(row["measured_collision_aborted"] for row in rows),
            "initial_collision_rejection_count": sum(row["initial_collision_rejected"] for row in rows),
            "planning_wall_seconds": None, "planning_wall_time_status": "not_collected; subprocess wall includes initialization, rendering, execution and serialization"}


def run(batch: Path) -> int:
    protocol_path = batch / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (protocol["seeds"] != list(SEEDS) or protocol["attempts_per_seed"] != 1 or protocol["retry_allowed"]
            or protocol["runner_sha256"] != sha(Path(__file__)) or protocol["per_seed_subprocess_timeout_seconds"] != TIMEOUT):
        raise RuntimeError("Protocol identity mismatch before launch")
    if (batch / "summary.json").exists() or any((batch / f"seed-{s:02d}").exists() for s in SEEDS):
        raise RuntimeError("This batch already started; resume or retries are prohibited")
    protocol_sha = sha(protocol_path)
    expected = protocol["source19_sha256"]
    source_identity(expected)
    env = os.environ.copy(); env.update(ENVIRONMENT)
    for key, value in protocol["task_level_environment"].items():
        if value is None: env.pop(key, None)
        else: env[key] = value
    summary: dict[str, Any] = {"status": "running", "status_interpretation": "batch administration only; success rates are reported separately",
        "started_at": now(), "protocol": str(protocol_path), "protocol_sha256": protocol_sha, "rows": [],
        "aggregate": summarize([])}
    atomic_json(batch / "summary.json", summary)
    try:
        for seed in SEEDS:
            before = source_identity(expected)
            if {name: sha(CHECKOUT / name) for name in CORE_FILES} != protocol["core_source_sha256"]:
                raise RuntimeError("Protocol violation: task, physics or controller core source changed")
            if sha(protocol_path) != protocol_sha or sha(Path(__file__)) != protocol["runner_sha256"]:
                raise RuntimeError("Protocol or runner mutated between attempts")
            output = batch / f"seed-{seed:02d}"
            cmd = command(seed, output)
            if cmd != protocol["commands"][str(seed)]: raise RuntimeError("Command differs from preregistration")
            log = batch / f"seed-{seed:02d}.log"
            row: dict[str, Any] = {"seed": seed, "attempt": 1, "command": cmd, "cwd": str(CHECKOUT), "environment": ENVIRONMENT,
                "output_directory": str(output), "log": str(log), "started_at": now(), "source19_sha256_before": before}
            atomic_json(batch / f"seed-{seed:02d}-invocation.json", row)
            started = time.monotonic(); timed_out = False
            with log.open("wb") as stream:
                process = subprocess.Popen(cmd, cwd=CHECKOUT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                row["pid"] = process.pid
                try: exit_code = process.wait(timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try: exit_code = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); exit_code = process.wait()
            row.update(exit_code=exit_code, subprocess_timed_out=timed_out, wall_seconds=time.monotonic() - started, completed_at=now())
            after = source_identity(expected)
            row["source19_sha256_after"] = after
            report_path = output / "report.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            ticks = read_ticks(output / "ticks.jsonl")
            protocol_violations = []
            if {name: sha(CHECKOUT / name) for name in CORE_FILES} != protocol["core_source_sha256"]:
                protocol_violations.append("task, physics or controller core source changed")
            if report and report.get("source_sha256") != expected: protocol_violations.append("native source snapshot differs")
            if report and report.get("execution_configuration") != protocol["execution_configuration"]: protocol_violations.append("native configuration differs")
            if sha(protocol_path) != protocol_sha or sha(Path(__file__)) != protocol["runner_sha256"]: protocol_violations.append("protocol or runner mutated")
            stages = report.get("stages", [])
            failure = next((s for s in stages if s.get("status") != "passed"), {})
            reasons = [str(s.get("reason", "")) for s in stages]
            classification = classify(report, ticks, timed_out, exit_code)
            observed = any(truth(t.get("official_evaluation", {}).get("success")) for t in ticks)
            row.update(report=str(report_path), native_report_status=report.get("status"), total_environment_steps=report.get("total_environment_steps"),
                first_official_success_step=report.get("first_official_success_step"), observed_official_success=observed,
                complete_physical_closure=bool(report.get("official_200_step_complete") is True and report.get("status") == "passed" and exit_code == 0 and not timed_out),
                native_truncation_observed=any(t.get("native_truncated") for t in ticks), classification=classification,
                candidate_collision_rejected="candidate_collision" in reasons, measured_collision_aborted="measured_collision" in reasons,
                initial_collision_rejected="initial_collision" in reasons, failed_phase=failure.get("name", report.get("stage")),
                failed_phase_status=failure.get("status"), failure_reason=failure.get("reason", report.get("reason")),
                error_type=report.get("error_type"), error=report.get("error"), traceback=report.get("traceback"),
                initial_collision_coverage_complete=report.get("initial_collision_coverage", {}).get("all_covered"),
                attachment_initialized="attachment_T_tcp_objects" in report,
                phase_step_counts=report.get("phase_step_counts", {}), first_failed_phase=failure.get("name"),
                artifact_errors=report.get("artifact_errors", []), raw_recording_check=evidence(output, report, ticks),
                protocol_violations=protocol_violations,
                planning_wall_seconds=None, planning_wall_time_status="not_collected_by_frozen_native_entry")
            atomic_json(batch / f"seed-{seed:02d}-invocation.json", row)
            summary["rows"].append(row); summary["aggregate"] = summarize(summary["rows"])
            atomic_json(batch / "summary.json", summary)
            print(json.dumps({"seed": seed, "classification": classification, "steps": row["total_environment_steps"],
                              "observed_official_success": observed, "complete_physical_closure": row["complete_physical_closure"],
                              "wall_seconds": row["wall_seconds"]}), flush=True)
            if protocol_violations: raise RuntimeError("Protocol violation: " + "; ".join(protocol_violations))
        summary.update(status="completed", completed_at=now(), all_attempts_executed_once=True)
        atomic_json(batch / "summary.json", summary)
        return 0
    except Exception as exc:
        summary.update(status="stopped_protocol_or_runner_error", stopped_at=now(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(batch / "summary.json", summary)
        raise


def self_test() -> dict[str, Any]:
    """Synthetic command, false/unknown outcome, denominator and atomicity tests."""
    cmd = command(1, Path("/tmp/seed-01"))
    assert cmd[cmd.index("--seed") + 1] == "1" and cmd[cmd.index("--max-episode-steps") + 1] == "200"
    assert cmd[cmd.index("--joint-acc-limit") + 1] == "2.6" and cmd[cmd.index("--approach-acc-limit") + 1] == "0.9"
    for bad in (0, 11, True):
        try: command(bad, Path("/tmp/invalid"))
        except ValueError: pass
        else: raise AssertionError("Unauthorized seed accepted")
    assert truth([True]) and not truth([False]) and not truth(None) and not truth([])
    assert classify({"status": "passed", "official_200_step_complete": True}, [], False, 0) == "complete"
    assert classify({"status": "failed", "stages": [{"name": "pregrasp", "status": "rejected", "reason": "candidate_collision"}]}, [], False, 1) == "candidate_collision_rejected"
    assert classify({}, [{"native_truncated": True}], False, 1) == "native_time_limit"
    assert classify({}, [], True, -15) == "subprocess_timeout"
    assert classify({"error": "Initial support not valid", "stage": "attach_collision_prediction"}, [], False, 1) == "support_initialization_error"
    assert classify({}, [], False, -11) == "engineering_error"
    rows = [{"classification": "complete", "observed_official_success": True, "complete_physical_closure": True,
             "native_truncation_observed": False, "subprocess_timed_out": False, "candidate_collision_rejected": False,
             "measured_collision_aborted": False, "initial_collision_rejected": False}]
    result = summarize(rows)
    assert result["observed_official_success_rate"] == .1 and result["remaining"] == 9 and result["planned_denominator"] == 10
    with tempfile.TemporaryDirectory(prefix="tig-compact-seeds-selftest-") as td:
        p = Path(td) / "summary.json"
        atomic_json(p, {"values": [False, 0, None]}); assert json.loads(p.read_text()) == {"values": [False, 0, None]}
        atomic_json(p, {"status": "completed"}); assert json.loads(p.read_text())["status"] == "completed"
        assert not p.with_suffix(".json.tmp").exists()
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
            "checks": ["frozen command and forbidden seed rejection", "unknown/false are not success",
                       "physical/candidate/time-limit/support/engineering categories", "fixed denominator includes pending/failures", "atomic progress writes"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--self-test", action="store_true")
    choice.add_argument("--prepare-only", action="store_true")
    choice.add_argument("--batch-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    if Path(sys.executable).resolve() != PYTHON.resolve(): parser.error("Use the frozen official environment Python")
    if args.prepare_only:
        batch = prepare(); print(json.dumps({"batch": str(batch), "protocol": str(batch / "protocol.json")}), flush=True); return 0
    return run(args.batch_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
