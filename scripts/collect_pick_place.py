"""Bounded, resumable collection behind scripts/collect_data.py --robot.

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
    if args.tasks and args.tasks != ["DualArmPickPlace"]:
        raise ValueError("--robot currently collects the shared DualArmPickPlace task")
    pools = read_splits(args.seed_splits)
    root = Path(args.demos_dir).resolve()
    if args.dry_run:
        summary, _, _ = collection_state(root, pools, read_events(root / "attempts.jsonl"))
        print(json.dumps(summary, indent=2))
        return 0
    # Import the live profile only for an actual collection, not CPU summaries.
    from mani_skill.envs.tasks.tabletop.dual_pick_place import PROFILES
    from mani_skill.examples.motionplanning.dual.sync.pick_place import POLICY
    settings = {"task": "DualArmPickPlace-v1", "robot": args.robot, "fps": 20,
                "profile": asdict(PROFILES[args.robot]), "policy": POLICY.to_dict()}
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
                command = [sys.executable, "-m", "mani_skill.examples.motionplanning.dual.sync.pick_place",
                           "--robot", args.robot, "--seed", str(seed), "--output-dir", str(output)]
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
                result_path = output / "result.json"
                try:
                    result = json.loads(result_path.read_text()) if result_path.exists() else {}
                    if not isinstance(result, dict):
                        raise ValueError("Expected a result object")
                except (ValueError, OSError) as exc:
                    result = {}
                    error = f"worker_result_invalid: {type(exc).__name__}: {exc}"
                success = bool(result.get("task_success", False))
                usable = (code == 0 and success and result.get("execution", {}).get("status") == "passed"
                          and result.get("recording", {}).get("status") == "passed")
                reason = None if usable else (error or result.get("execution", {}).get("error")
                    or result.get("recording", {}).get("error")
                    or ("official_task_unsuccessful" if code == 0 and not success else f"worker_exit_{code}"))
                append_event(events_path, {"event": "finish", "seed": seed, "split": args.split,
                    "returncode": code, "task_success": success, "training_usable": usable,
                    "failure_reason": reason, "steps": result.get("steps"),
                    "h5_path": str((output / "trajectory.h5").relative_to(root)), "episode_id": 0,
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
