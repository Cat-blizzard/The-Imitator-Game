"""Bounded local qualification only; never imports or submits a paid provider."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
RUN = Path("/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview")
PROPOSALS = Path("/mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e/household_layouts_v3/proposals.json")
CONTRACT = WORKSPACE / "data_generation_methods/human_to_robot_pipeline/evidence_20261020d/apple_contract.json"
CASES = {
    "development": "household_e_g00_baseline_2026302001_nativeclear_v3",
    "validation": "household_e_g01_clutter4_2026302003_nativeclear_v3",
}
GPU_UUID = "GPU-807bd44d-36ea-31d0-427c-50e264428c67"


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_new(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def run(case):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / ".qualification.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Path("/tmp/real_sim_human_gpu2.lock").open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol_path = RUN / "protocol.json"
            if not protocol_path.exists():
                module_root = WORKSPACE / "data_generation_methods/human_to_robot_pipeline"
                save_new(protocol_path, {
                    "schema": "local_multiview_qualification_v1", "frozen_at": now(),
                    "case_order": list(CASES), "cases": CASES, "max_attempts": 2,
                    "max_concurrency": 1, "timeout_seconds": 1200, "new_api_calls_allowed": 0,
                    "gpu_uuid": GPU_UUID, "minimum_free_bytes": 10 * 1024**3,
                    "inputs": [binding(PROPOSALS), binding(CONTRACT)],
                    "implementation": [binding(module_root / name) for name in (
                        "strict_apple_adapter.py", "strict_motion.py",
                        "experimental_apple_adapter.py", "multiview_recording.py")],
                    "acceptance": {
                        "physical": "unchanged native verifier and strict replay",
                        "synchronization": "same execution, exact frame-state-command mapping, all decoded PTS",
                        "preservation": "original camera calibration and dynamic arrays compared to old baseline",
                        "visibility": "per-event segmentation pixels; 16 pixel diagnostic threshold frozen from old audit",
                        "promotion": "candidate only; two apple layouts cannot establish other-task generalization",
                    },
                    "validation_scope": "g01 layout withheld from multiview implementation tuning; historically used for strict planning, same human source as g00; not unseen-source validation",
                })
            protocol = json.loads(protocol_path.read_text())
            for bound in protocol["implementation"] + protocol["inputs"]:
                if binding(bound["path"]) != bound:
                    raise ValueError("Frozen implementation/input changed; preserve this protocol and record a new experiment decision")
            if case == "validation":
                prior = RUN / "development" / "result.json"
                if not prior.exists() or json.loads(prior.read_text()).get("passed") is not True:
                    raise ValueError("Development physical attempt must pass before validation")
            out = RUN / case
            started_path = RUN / f"{case}.started.json"
            if started_path.exists() or out.exists():
                raise FileExistsError("Existing attempt: inspect or recover it; never launch it again")
            if len(list(RUN.glob("*.started.json"))) >= protocol["max_attempts"]:
                raise ValueError("Local cumulative attempt limit reached")
            free = shutil.disk_usage(RUN).free
            if free < protocol["minimum_free_bytes"]:
                raise ValueError("Insufficient disk space")
            gpu_line = subprocess.check_output([
                "nvidia-smi", "--id=" + GPU_UUID, "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip().split(", ")
            if gpu_line[0] != GPU_UUID or int(gpu_line[1]) > 256 or int(gpu_line[2]) > 10:
                raise RuntimeError("GPU 2 is not idle; no work launched")
            apps = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                text=True)
            if GPU_UUID in apps:
                raise RuntimeError("GPU 2 has an existing compute process")
            command = [str(WORKSPACE / "PolicyGen/.venv/bin/python"), "-m",
                       "data_generation_methods.human_to_robot_pipeline.strict_apple_adapter",
                       "--proposals", str(PROPOSALS), "--contract", str(CONTRACT),
                       "--case-id", CASES[case], "--output", str(out), "--gpu", "2", "--multiview"]
            t0 = time.monotonic()
            save_new(started_path, {"started_at": now(), "case": case, "command": command,
                                   "disk_free_bytes": free, "gpu": gpu_line,
                                   "protocol": binding(protocol_path), "new_api_calls": 0})
            with (RUN / f"{case}.log").open("x") as log:
                proc = subprocess.Popen(command, cwd=WORKSPACE, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                save_new(RUN / f"{case}.process.json", {
                    "pid": proc.pid, "started_at": now(), "output": str(out),
                    "note": "PID alone can be reused; verify command and result before treating as live"})
                timed_out = False
                try:
                    code = proc.wait(timeout=protocol["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        code = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        code = proc.wait()
            result = {"finished_at": now(), "exit_code": code, "timed_out": timed_out,
                      "wall_seconds": time.monotonic() - t0, "result_exists": (out / "result.json").exists(),
                      "new_api_calls": 0}
            save_new(RUN / f"{case}.finished.json", result)
            print(json.dumps(result))
            return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    args = parser.parse_args()
    sys.exit(run(args.case))
