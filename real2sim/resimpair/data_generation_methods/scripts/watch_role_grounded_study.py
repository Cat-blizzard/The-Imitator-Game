"""Read-only bounded completion watcher for an already-submitted repair study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from data_generation_methods.r2h_pipeline.common import file_sha256, write_json
from data_generation_methods.scripts.echoflow_repair_study import ROOT, BASELINE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--minutes", type=float, default=20)
    args = p.parse_args()
    root = args.root
    submission = json.loads((root / "submit-role-grounded/result.json").read_text())
    if not submission.get("task_id"):
        raise ValueError("Requires a task ID from the actual POST response")
    deadline = time.monotonic() + args.minutes * 60
    state = {
        "task_id": submission["task_id"], "only_GET_polling_no_POST": True,
        "task_binding": "confirmed_by_POST_response",
        "training_admitted": False, "training_started": False,
    }
    while time.monotonic() < deadline:
        result = subprocess.run(
            [sys.executable, "-B", str(WORKSPACE / "data_generation_methods/scripts/echoflow_repair_study.py"),
             "poll-role-grounded", "--root", str(root)],
            capture_output=True, text=True, timeout=210,
        )
        # Child prints only sanitized summaries, never an output URL.
        if result.returncode == 0:
            record = json.loads(result.stdout.strip().splitlines()[-1])
            state.update(record)
            state["updated_unix"] = time.time()
            state["quality_status"] = "not_evaluated"
            write_json(root / "WATCH_STATUS.json", state, overwrite=True)
            print(json.dumps(state), flush=True)
            if record["status"] == "succeeded":
                video = root / "role_grounded.mp4"
                state["video_sha256"] = file_sha256(video)
                evidence = root / "repair_boundary_probe"
                if not evidence.exists():
                    diagnostic = subprocess.run(
                        [sys.executable, "-B", "-m", "data_generation_methods.r2h_pipeline.boundary_probe",
                         "--video", str(video),
                         "--anchor", str(BASELINE / "edit-anchor/human_initial.png"),
                         "--anchor-roi", "0.59", "0.50", "0.68", "0.60",
                         "--search-roi", "0", "0.30", "1", "0.80",
                         "--output-dir", str(evidence)],
                        cwd=WORKSPACE, capture_output=True, text=True, timeout=300,
                    )
                    (root / "repair_boundary_probe.log").write_text(
                        diagnostic.stdout + "\n" + diagnostic.stderr)
                    state["boundary_probe_exit_code"] = diagnostic.returncode
                if (evidence / "report.json").exists():
                    state["boundary_probe"] = str(evidence / "report.json")
                    state["quality_status"] = "diagnostics_ready_requires_visual_review"
                state["watcher_status"] = "finished"
                state["updated_unix"] = time.time()
                write_json(root / "WATCH_STATUS.json", state, overwrite=True)
                print(json.dumps(state), flush=True)
                return
            if record["status"] in {"failed", "expired", "cancelled"}:
                return
        else:
            state.update(status="read_query_failed", updated_unix=time.time(),
                         child_exit_code=result.returncode)
            write_json(root / "WATCH_STATUS.json", state, overwrite=True)
            # Raw traceback is kept local; do not turn an API issue into a resubmission.
            (root / "watch_last_query_error.log").write_text(result.stderr)
        time.sleep(min(45, max(0, deadline-time.monotonic())))
    state.update(watcher_status="bounded_window_ended", updated_unix=time.time())
    write_json(root / "WATCH_STATUS.json", state, overwrite=True)


if __name__ == "__main__":
    main()
