"""Bounded, resumable synchronous collection behind collect_data.py --robot.

Each seed launches the existing expert/recorder in its own immutable directory.
The predeclared evaluation pool is never collected by this entrypoint.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


SUPPORTED_TASKS = ("DualArmPickPlace-v1", "TwoRobotPickPillToRegions-v1",
                   "TwoRobotPickAppleBananaToBaskets-v1")
ACCEPTANCE_RULE_VERSION = 2  # Honor failed quality checks selected in the saved policy.


def selected_task(tasks: list[str] | None) -> str:
    """An explicit single task; retain the historical shared-task default."""
    if tasks is None:
        return SUPPORTED_TASKS[0]
    if len(tasks) != 1:
        raise ValueError("--robot collection requires exactly one task")
    task = tasks[0] if tasks[0].endswith("-v1") else tasks[0]+"-v1"
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported synchronous collection task: {task}")
    return task


def worker_spec(task: str, robot: str, seed: int, output: Path) -> tuple[list[str], Path]:
    """Return one-seed official-worker command and its native recording path."""
    if task == "DualArmPickPlace-v1":
        return ([sys.executable, "-m", "mani_skill.examples.motionplanning.dual.sync.pick_place",
                 "--robot", robot, "--seed", str(seed), "--output-dir", str(output)],
                output / "trajectory.h5")
    return ([sys.executable, "-m", "mani_skill.examples.motionplanning.dual.two_robot_run",
             "--env-id", task, "--execution-mode", "sync", "--robot", robot,
             "--seed", str(seed), "--num-traj", "1", "--obs-mode", "rgb",
             "--sim-backend", "cpu", "--shader", "default", "--save-video",
             "--record-dir", str(output), "--traj-name", "trajectory"],
            output / task / "motionplanning" / "trajectory.h5")


def worker_result(task: str, output: Path, source: Path) -> dict:
    """Read task/execution/quality/recording facts without conflating them."""
    if task == "DualArmPickPlace-v1":
        return json.loads((output / "result.json").read_text())
    metadata = json.loads(source.with_suffix(".json").read_text())
    episodes = metadata.get("episodes", [])
    if len(episodes) != 1 or episodes[0].get("episode_id") != 0:
        raise ValueError("A bounded worker must preserve exactly its requested episode")
    result = episodes[0].get("dual_sync")
    if not isinstance(result, dict):
        raise ValueError("Official episode is missing its synchronous execution report")
    return {**result, "task_success": bool(result.get("official", {}).get("success", False)),
            "steps": result.get("official", {}).get("elapsed_steps")}


def training_usable(result: dict, returncode: int) -> bool:
    """Native recorder success labels differ; preserve their original reports."""
    return (returncode == 0 and bool(result.get("task_success", False))
            and result.get("execution", {}).get("status") == "passed"
            and result.get("recording", {}).get("status") in ("passed", "saved")
            and result.get("quality", {}).get("status") != "failed")


def failure_reason(result: dict, returncode: int, error: str | None = None) -> str | None:
    """Prefer the worker's specific cause over its independent official result."""
    if training_usable(result, returncode):
        return None
    if error:
        return error
    for category in ("execution", "quality", "recording"):
        for key in ("error", "reason"):
            if result.get(category, {}).get(key):
                return str(result[category][key])
    if result.get("quality", {}).get("status") == "failed":
        return "selected_quality_failed"
    return ("official_task_unsuccessful" if returncode == 0 and not result.get("task_success")
            else f"worker_exit_{returncode}")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_splits(path: str) -> dict[str, list[int]]:
    spec = json.loads(Path(path).read_text())
    pools = spec.get("seed_splits", {})
    if spec.get("schema_version") != 1 or set(pools) != {"train", "development", "evaluation"}:
        raise ValueError("Expected schema_version 1 and train/development/evaluation seed_splits")
    seen: set[int] = set()
    for split, seeds in pools.items():
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"Seed pool must be a nonempty list: {split}")
        for seed in seeds:
            if type(seed) is not int or seed < 0 or seed in seen:
                raise ValueError(f"Seed pools must contain unique nonnegative integers: {seed}")
            seen.add(seed)
    return pools


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def append_event(path: Path, event: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(event) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def collection_state(root: Path, pools: dict, events: list[dict]) -> tuple[dict, dict, set[int]]:
    """Use completed facts for selection, launches for the finite attempt budget."""
    launched = [row for row in events if row["event"] == "launch"]
    finished = [row for row in events if row["event"] == "finish"]
    seed_split = {seed: split for split, seeds in pools.items() for seed in seeds}
    if any(row["seed"] not in seed_split or row["split"] != seed_split[row["seed"]]
           for row in launched + finished):
        raise ValueError("Saved attempt differs from the frozen seed split")
    selected = []
    for row in finished:
        if row["training_usable"]:
            source = root / row["h5_path"]
            if not source.is_file() or not source.with_suffix(".json").is_file():
                raise FileNotFoundError(f"Accepted source is missing: {source}")
            selected.append({key: row[key] for key in ("h5_path", "episode_id", "seed", "split")})
    summary = {"schema_version": 1, "attempts": len(launched),
        "completed_attempts": len(finished),
        "elapsed_seconds": sum(row.get("wall_seconds", 0) for row in finished),
        "failure_reasons": dict(Counter(row["failure_reason"] for row in finished
                                        if not row["training_usable"])), "splits": {}}
    for split in pools:
        done = [row for row in finished if row["split"] == split]
        summary["splits"][split] = {
            "attempts": sum(row["split"] == split for row in launched),
            "completed_attempts": len(done),
            "task_successes": sum(row["task_success"] for row in done),
            "training_usable": sum(row["split"] == split for row in selected)}
    manifest = {"schema_version": 1, "seed_splits": pools, "episodes": selected}
    return summary, manifest, {row["seed"] for row in launched}


def _stop(child: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    if child.poll() is None:
        os.killpg(child.pid, sig)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def collect(args: Any) -> int:
    """Run finite seed pools; retain failures and resume without replacing attempts."""
    if not args.seed_splits or args.target_episodes < 1 or args.max_attempts < 1:
        raise ValueError("Collection requires --seed-splits and positive target/attempt limits")
    task = selected_task(args.tasks)
    pools = read_splits(args.seed_splits)
    root = Path(args.demos_dir).resolve()
    if args.dry_run:
        summary, _, _ = collection_state(root, pools, read_events(root / "attempts.jsonl"))
        print(json.dumps(summary, indent=2))
        return 0
    # Import the live profile only for an actual collection, not CPU summaries.
    from mani_skill.agents.robots.embodiment_profiles import get_robot_profile
    from mani_skill.envs.tasks.tabletop.dual_arm_profiles import get_task_profile
    from mani_skill.examples.motionplanning.dual.sync.pick_place import POLICY
    from mani_skill.examples.motionplanning.dual.sync.runtime_policy import DEFAULT_POLICY
    policy = POLICY if task == "DualArmPickPlace-v1" else DEFAULT_POLICY
    settings = {"task": task, "robot": args.robot, "fps": 20,
                "acceptance_rule_version": ACCEPTANCE_RULE_VERSION,
                "profile": asdict(get_task_profile(task, args.robot)),
                "robot_profile": asdict(get_robot_profile(args.robot)),
                "policy": policy.to_dict()}
    settings = json.loads(json.dumps(settings))
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for filename, value in (("seed_splits.json", {"schema_version": 1, "seed_splits": pools}),
                                ("collection_config.json", settings)):
            path = root / filename
            if path.exists() and json.loads(path.read_text()) != value:
                raise ValueError(f"Collection configuration changed: {filename}; use a new data directory")
            if not path.exists():
                write_json(path, value)
        events_path = root / "attempts.jsonl"

        def publish() -> tuple[dict, set[int]]:
            summary, manifest, attempted = collection_state(root, pools, read_events(events_path))
            write_json(root / "generation_stats.json", summary)
            write_json(root / "episode_manifest.json", manifest)
            return summary, attempted

        child = None
        previous = {}

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        try:
            while True:
                summary, attempted = publish()
                state = summary["splits"][args.split]
                if state["training_usable"] >= args.target_episodes:
                    return 0
                seeds = [seed for seed in pools[args.split] if seed not in attempted]
                if state["attempts"] >= args.max_attempts or not seeds:
                    return 2
                seed = seeds[0]
                output = root / f"{args.split}-{seed}"
                command, source = worker_spec(task, args.robot, seed, output)
                append_event(events_path, {"event": "launch", "seed": seed, "split": args.split,
                                          "output_dir": str(output), "time": time.time()})
                started = time.monotonic()
                print(f"{args.robot} {args.split} seed={seed}, attempt={state['attempts']+1}", flush=True)
                code, error = 127, None
                with (root / f"{args.split}-{seed}.log").open("x") as log:
                    try:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            cwd=Path(__file__).resolve().parents[1], start_new_session=True)
                        code = child.wait(timeout=args.task_timeout)
                    except subprocess.TimeoutExpired:
                        _stop(child)
                        code, error = 124, "worker_timeout"
                    except OSError as exc:
                        error = f"worker_startup: {type(exc).__name__}: {exc}"
                child = None
                result_read_error = None
                try:
                    result = worker_result(task, output, source)
                    if not isinstance(result, dict):
                        raise ValueError("Expected a result object")
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    result = {}
                    result_read_error = f"worker_result_invalid: {type(exc).__name__}: {exc}"
                success = bool(result.get("task_success", False))
                usable = training_usable(result, code)
                reason = failure_reason(result, code, error or result_read_error)
                append_event(events_path, {"event": "finish", "seed": seed, "split": args.split,
                    "returncode": code, "task_success": success, "training_usable": usable,
                    "failure_reason": reason, "steps": result.get("steps"),
                    "worker_error": error, "result_read_error": result_read_error,
                    "quality_status": result.get("quality", {}).get("status"),
                    "quality": {key: result.get("quality", {})[key]
                                for key in ("status", "reason", "error", "category")
                                if key in result.get("quality", {})},
                    "official": result.get("official", {}),
                    "execution": result.get("execution", {}),
                    "recording": result.get("recording", {}),
                    "h5_path": str(source.relative_to(root)), "episode_id": 0,
                    "wall_seconds": time.monotonic()-started})
                publish()
        except KeyboardInterrupt:
            if child is not None:
                _stop(child, signal.SIGINT)
            publish()
            return 130
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pools = {"train": [1], "development": [2], "evaluation": [3]}
        summary, manifest, attempted = collection_state(root, pools,
            [{"event": "launch", "seed": 1, "split": "train"}])
        assert summary["attempts"] == 1 and attempted == {1} and not manifest["episodes"]
        print("Interrupted attempts consume their budget and cannot leak into training: passed")
