"""CLI integration for synchronous collection, without importing a planner.

The caller owns environment creation and the episode loop. This module selects
the solver, configures optional session lifetime, and records synchronous facts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
import os
import sys
from typing import Any

from mani_skill.utils.io_utils import dump_json


SYNC_TASKS = frozenset({"TwoRobotPickPillToRegions-v1"})
CUROBO_DEFAULTS = {"curobo_python": None, "curobo_device": "cuda:0", "curobo_cache_dir": None}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution-mode", choices=("serial", "sync"), default="serial",
                        help="Use the original serial solver or the optional synchronous dual-arm solver.")
    parser.add_argument("--planner-backend", choices=("mplib", "curobo"), default="mplib",
                        help="Planning backend for synchronous execution (default: mplib).")
    parser.add_argument("--curobo-python", default=argparse.SUPPRESS, metavar="PYTHON",
                        help="Interpreter with cuRobo 0.9.0 installed (default: TIG_CUROBO_PYTHON, then the current interpreter).")
    parser.add_argument("--curobo-device", default=argparse.SUPPRESS, metavar="DEVICE",
                        help="CUDA device in the cuRobo worker (default: cuda:0).")
    parser.add_argument("--curobo-cache-dir", default=argparse.SUPPRESS, metavar="DIR",
                        help="Optional writable directory for prepared cuRobo robot assets.")
    parser.add_argument("--sync-config", type=str,
                        help="Optional JSON policy overrides for synchronous execution.")


def complete_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args._explicit_curobo_options = tuple(name for name in CUROBO_DEFAULTS if hasattr(args, name))
    for name, default in CUROBO_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def _curobo_options(args: argparse.Namespace) -> dict[str, str | None]:
    return {"python": args.curobo_python or os.environ.get("TIG_CUROBO_PYTHON") or sys.executable,
            "device": args.curobo_device, "cache_dir": args.curobo_cache_dir}


def resolve_solver(args: argparse.Namespace,
                   serial_solutions: Mapping[str, Callable[..., Any]]) -> Callable[..., Any]:
    """Validate the selection before the caller constructs an environment."""
    if args.planner_backend != "curobo" and args._explicit_curobo_options:
        raise ValueError("--curobo-* options require --planner-backend curobo")
    if args.planner_backend == "curobo" and args.execution_mode != "sync":
        raise ValueError("--planner-backend curobo requires --execution-mode sync")
    if args.execution_mode == "sync":
        if args.env_id not in SYNC_TASKS:
            raise ValueError(f"Synchronous execution is not supported for {args.env_id}. "
                             f"Supported tasks: {sorted(SYNC_TASKS)}")
        from .solve import load_policy, solve

        options = {}
        if args.sync_config is not None:
            options["policy"] = load_policy(args.sync_config)
        if args.planner_backend == "curobo":
            options.update(planner_backend="curobo", curobo_options=_curobo_options(args))
        return partial(solve, **options) if options else solve
    if args.sync_config is not None:
        raise ValueError("--sync-config requires --execution-mode sync")
    if args.env_id not in serial_solutions:
        raise RuntimeError(f"No motion planning solution for {args.env_id}.")
    return serial_solutions[args.env_id]


def recording_options(args: argparse.Namespace, control_freq: int) -> dict[str, Any]:
    if args.execution_mode == "serial":
        return {"source_desc": "official motion planning solution from ManiSkill contributors", "video_fps": 30}
    label = {"mplib": "MPLib", "curobo": "cuRobo"}[args.planner_backend]
    return {"source_desc": f"synchronous {label} dual-Panda solution", "video_fps": control_freq}


def prepare_collection(args: argparse.Namespace, env: Any, solver: Callable[..., Any],
                       resources: ExitStack) -> Callable[..., Any]:
    """Attach synchronous metadata and register owned session cleanup."""
    if args.execution_mode == "serial":
        return solver
    if args.planner_backend == "curobo" and (args.num_traj > 1 or args.only_count_success):
        from .curobo_backend import CuroboSession

        session = CuroboSession(options=_curobo_options(args))
        resources.callback(session.close)
        solver = partial(solver, curobo_session=session)
    env._json_data["motion_planning"] = {
        "execution_mode": "sync", "planner_backend": args.planner_backend,
        "control_freq": env.unwrapped.control_freq,
        "action_order": ["panda_wristcam-0", "panda_wristcam-1"],
    }
    return solver


def flush_trajectory(env: Any, execution_mode: str, save: bool = True) -> None:
    """Keep recording facts independent of actual task and quality outcomes."""
    previous_count = len(env._json_data["episodes"])
    try:
        env.flush_trajectory(save=save)
        if execution_mode == "sync" and len(env._json_data["episodes"]) > previous_count:
            env.unwrapped.dual_sync_result["recording"] = {
                "status": "saved", "format": "RecordEpisode-h5-json"}
            env._json_data["episodes"][-1]["dual_sync"] = deepcopy(env.unwrapped.dual_sync_result)
            dump_json(env._json_path, env._json_data, indent=2)
    except Exception as exc:
        if execution_mode == "sync":
            failure = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            env.unwrapped.dual_sync_result["recording"] = failure
            if len(env._json_data["episodes"]) > previous_count:
                env._json_data["episodes"][-1].setdefault("dual_sync", {})["recording"] = failure
        raise


@dataclass
class OutcomeSummary:
    """Independent synchronous result counters for the caller's progress bar."""

    official_successes: int = 0
    failure_categories: Counter[str] = field(default_factory=Counter)
    quality_failures: int = 0

    def record(self, facts: Mapping[str, Any], official_success: bool) -> None:
        self.official_successes += bool(official_success)
        if facts["execution"]["status"] != "passed":
            self.failure_categories[facts["execution"].get("category", "execution")] += 1
        self.quality_failures += facts["quality"]["status"] == "failed"

    def postfix(self, attempts: int) -> dict[str, Any]:
        return {"execution_failure_rate": sum(self.failure_categories.values()) / attempts,
                "failure_categories": dict(self.failure_categories),
                "official_success_rate": self.official_successes / attempts,
                "quality_failures": self.quality_failures}


def self_test() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = complete_arguments(parser.parse_args([]))
    args.env_id = "synthetic"
    solver = lambda env: env
    assert resolve_solver(args, {args.env_id: solver}) is solver
    assert recording_options(args, 20)["video_fps"] == 30
    summary = OutcomeSummary()
    summary.record({"execution": {"status": "failed", "category": "geometry"},
                    "quality": {"status": "not_evaluated"}}, False)
    summary.record({"execution": {"status": "passed"}, "quality": {"status": "failed"}}, True)
    assert summary.postfix(2) == {"execution_failure_rate": .5,
        "failure_categories": {"geometry": 1}, "official_success_rate": .5, "quality_failures": 1}
    print("sync entry self-test passed: serial dispatch, recording frequency, independent outcomes")


if __name__ == "__main__":
    self_test()
