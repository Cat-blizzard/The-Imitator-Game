"""Run an explicit generation manifest once, with bounded worker ownership.

Manifest keys are the existing single-episode parser destinations. The batch
owns output directories and worker lifetime. Failed episodes are retained and
never retried; official success and accepted data throughput stay separate.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import io
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable

SCHEMA = "tig-generation-batch-v1"
SUMMARY_SCHEMA = "tig-generation-batch-results-v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
_OWNED = frozenset(("help", "self_test", "output_dir", "worker_lifetime", "episode_id"))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value: {value}")))
    if not isinstance(value, dict):
        raise ValueError("Generation manifest must be a JSON object")
    return value


def prepare_episodes(document: dict[str, Any], manifest_dir: Path, output_dir: Path, *,
                     parser: argparse.ArgumentParser | None = None,
                     validator: Callable[..., Any] | None = None) -> list[argparse.Namespace]:
    """Resolve and validate every episode before opening a worker or output."""
    if parser is None or validator is None:
        from .validate_curobo_complete import build_parser, validate_episode_args
        parser = build_parser() if parser is None else parser
        validator = validate_episode_args if validator is None else validator
    if (document.get("schema") != SCHEMA or set(document) - {"schema", "defaults", "episodes"}
            or not isinstance(document.get("defaults", {}), dict)
            or not isinstance(document.get("episodes"), list) or not document["episodes"]):
        raise ValueError("Expected tig-generation-batch-v1 with defaults object and nonempty episodes list")
    actions = {action.dest: action for action in parser._actions if action.option_strings}
    defaults = {"quality_profile": "task-default-v1", "recording_mode": "normal", **document.get("defaults", {})}
    plans = []
    identifiers: set[str] = set()
    for row in document["episodes"]:
        if not isinstance(row, dict):
            raise ValueError("Each episode must be an object with a safe unique id")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier in identifiers:
            raise ValueError(f"Unsafe or duplicate episode id: {identifier!r}")
        identifiers.add(identifier)
        values = {**defaults, **{key: value for key, value in row.items() if key != "id"}}
        unknown = set(values) - (set(actions) - _OWNED)
        if unknown:
            raise ValueError(f"Unknown or batch-owned episode parameters: {sorted(unknown)}")
        arguments = []
        for key, value in values.items():
            action = actions[key]
            if action.type is int:
                if type(value) is not int:
                    raise ValueError(f"{key} must be a JSON integer")
            elif not isinstance(value, str) or not value:
                raise ValueError(f"{key} must be a nonempty JSON string")
            if action.type is Path:
                path = Path(value).expanduser()
                value = str(path if path.is_absolute() else (manifest_dir/path).resolve())
            # '=' keeps values beginning with '-' inside their own option.
            arguments.append(f"{action.option_strings[0]}={value}")
        arguments.append(f"--output-dir={output_dir/'episodes'/identifier}")
        args = parser.parse_args(arguments)
        args.episode_id = identifier
        validator(args, parser)
        plans.append(args)
    return plans


def episode_result(args: argparse.Namespace, exit_code: int, wall_s: float,
                   error: str | None = None) -> dict[str, Any]:
    """Read the existing recorded acceptance; do not recalculate quality gates."""
    report_path = args.output_dir/'report.json'
    report: dict[str, Any] = {}
    try:
        report = json.loads(report_path.read_text())
        if not isinstance(report, dict):
            raise ValueError("Episode report must be an object")
    except (OSError, ValueError) as exc:
        report = {}
        error = error or f"Report unavailable: {exc}"
    results = report.get("results", {})
    if not isinstance(results, dict):
        results = {}
        error = error or "Report results must be an object"
    recording_result = results.get("recording", {})
    recording = recording_result.get("status") if isinstance(recording_result, dict) else None
    complete = (report.get("selected_policy_complete") is True
        if report.get("completion_contract") == "selected-runtime-policy-v1"
        else report.get("project_complete_physical_closure") is True)
    accepted = exit_code == 0 and report.get("status") == "passed" and error is None
    qualified = accepted and complete and recording == "passed"
    official_result = results.get("official", {})
    official = official_result.get("final_success_before_horizon") if isinstance(official_result, dict) else None
    if type(official) is not bool:
        official = None
    return {"id": args.episode_id, "task": args.task, "seed": args.seed,
        "planner_seed": args.planner_seed, "backend": args.backend,
        "status": "passed" if accepted else "failed", "attempts": 1,
        "exit_code": exit_code, "wall_s": wall_s, "report": str(report_path),
        "report_status": report.get("status"), "selected_completion": complete,
        "recording_status": recording, "qualified_trajectory": qualified,
        "official_success_before_horizon": official,
        "error": error or report.get("error")}


def aggregate(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    qualified = sum(row.get("qualified_trajectory") is True for row in rows)
    official = [row.get("official_success_before_horizon") for row in rows]
    successes = sum(value is True for value in official)
    observed = sum(type(value) is bool for value in official)
    return {"allocated_episodes": len(rows), "attempted_episodes": sum(row.get("attempts", 0) for row in rows),
        "passed_episodes": sum(row.get("status") == "passed" for row in rows),
        "failed_episodes": sum(row.get("status") == "failed" for row in rows),
        "qualified_trajectories": qualified,
        "qualified_trajectories_per_hour": qualified*3600./wall_s if wall_s > 0 else None,
        "official_successes": successes, "official_observed_episodes": observed,
        "official_unknown_episodes": len(rows)-observed,
        "official_success_rate_all_allocated": successes/len(rows) if rows else None,
        "official_success_rate_observed": successes/observed if observed else None,
        "wall_s": wall_s,
        "throughput_denominator": "whole batch including cold initialization, episode resets, failures and final worker shutdown"}


def _save_summary(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run_batch(manifest: Path, output_dir: Path, *, worker_lifetime: str = "batch",
              runner: Callable[..., int] | None = None, pool_factory: Callable[[], Any] | None = None,
              parser: argparse.ArgumentParser | None = None,
              validator: Callable[..., Any] | None = None) -> int:
    if worker_lifetime not in ("episode", "batch"):
        raise ValueError("worker_lifetime must be episode or batch")
    manifest, output_dir = manifest.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Batch output directory must be unused")
    document = read_manifest(manifest)
    plans = prepare_episodes(document, manifest.parent, output_dir, parser=parser, validator=validator)
    if runner is None:
        from .validate_curobo_complete import run_episode
        runner = run_episode
    needs_curobo = any(args.backend == "curobo" for args in plans)
    if pool_factory is None and needs_curobo:
        from .curobo_episode_pool import CuroboEpisodePool
        pool_factory = CuroboEpisodePool
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/'episodes').mkdir()
    (output_dir/'manifest.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
    rows = [{"id": args.episode_id, "status": "not_run", "attempts": 0} for args in plans]
    summary = {"schema": SUMMARY_SCHEMA, "status": "running", "worker_lifetime": worker_lifetime,
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(), "retry_policy": "no retries",
        "qualified_definition": "existing accepted completion plus passed recording; no additional physics or quality threshold",
        "episodes": rows, "metrics": aggregate(rows, 0.)}
    summary_path = output_dir/'summary.json'
    _save_summary(summary_path, summary)
    batch_pool = None
    begin = time.monotonic()
    try:
        if worker_lifetime == "batch" and needs_curobo and pool_factory is not None:
            batch_pool = pool_factory()
        for index, args in enumerate(plans):
            episode_begin = time.monotonic()
            episode_pool = None
            error = None
            exit_code = 1
            try:
                if args.backend == "curobo":
                    assert pool_factory is not None
                    episode_pool = batch_pool if worker_lifetime == "batch" else pool_factory()
                exit_code = runner(args, session_pool=episode_pool)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if episode_pool is not None and worker_lifetime == "episode":
                    try:
                        episode_pool.close()
                    except Exception as exc:
                        error = error or f"Worker shutdown failed: {type(exc).__name__}: {exc}"
                rows[index] = episode_result(args, exit_code, time.monotonic()-episode_begin, error)
                summary["metrics"] = aggregate(rows, time.monotonic()-begin)
                _save_summary(summary_path, summary)
    finally:
        try:
            if batch_pool is not None:
                batch_pool.close()
        except Exception as exc:
            summary["batch_error"] = f"Worker shutdown failed: {type(exc).__name__}: {exc}"
        finally:
            finished = all(row["status"] != "not_run" for row in rows)
            summary.update(status=("passed" if all(row["status"] == "passed" for row in rows) and "batch_error" not in summary else "failed")
                if finished else "interrupted", completed_at=datetime.now(timezone.utc).isoformat(),
                metrics=aggregate(rows, time.monotonic()-begin))
            _save_summary(summary_path, summary)
    print(json.dumps({"status": summary["status"], "summary": str(summary_path),
                      "metrics": summary["metrics"]}), flush=True)
    return 0 if summary["status"] == "passed" else 1


def self_test() -> dict[str, Any]:
    """CPU manifest rejection, worker ownership, and failure-accounting checks."""
    parser = argparse.ArgumentParser()
    for name, values in (("backend", ("curobo", "mplib")), ("task", ("task-a",)),
                         ("quality-profile", ("legacy-strict-v1", "task-default-v1")),
                         ("recording-mode", ("normal", "diagnostic"))):
        parser.add_argument('--'+name, choices=values, default=values[0])
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--planner-seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path)
    validate = lambda args, parser: None
    document = {"schema": SCHEMA, "defaults": {"backend": "curobo"},
                "episodes": [{"id": "A", "seed": 401}, {"id": "B", "seed": 402},
                             {"id": "C", "seed": 403, "backend": "mplib"}]}
    rejected = 0
    with tempfile.TemporaryDirectory(prefix='tig-generation-batch-cpu-') as directory:
        root = Path(directory)
        for mutate in (lambda value: value['episodes'].append({"id": "A"}),
                       lambda value: value['defaults'].update(unknown=1),
                       lambda value: value['defaults'].update(output_dir='elsewhere'),
                       lambda value: value['defaults'].update(worker_lifetime='batch'),
                       lambda value: value['episodes'][0].update(id='../escape'),
                       lambda value: value['episodes'][0].update(seed=True)):
            bad = copy.deepcopy(document)
            mutate(bad)
            try:
                prepare_episodes(bad, root, root/'out', parser=parser, validator=validate)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("Invalid manifest accepted")
        manifest = root/'input.json'
        manifest.write_text(json.dumps(document))
        created = []
        class FakePool:
            def __init__(self) -> None:
                self.closed = False
                created.append(self)
            def close(self) -> None:
                self.closed = True
        calls = []
        def run(args: argparse.Namespace, *, session_pool: Any) -> int:
            assert args.quality_profile == 'task-default-v1' and args.recording_mode == 'normal'
            calls.append((args.episode_id, id(session_pool)))
            if args.backend == 'mplib':
                assert session_pool is None
                raise RuntimeError('synthetic runner failure without a report')
            args.output_dir.mkdir()
            passed = args.episode_id == 'A'
            report = {"status": "passed" if passed else "failed", "completion_contract": "selected-runtime-policy-v1",
                "selected_policy_complete": passed, "results": {"recording": {"status": "passed"},
                    "official": {"final_success_before_horizon": True}}}
            (args.output_dir/'report.json').write_text(json.dumps(report))
            return 0 if passed else 1
        for lifetime, count in (("batch", 1), ("episode", 2)):
            created.clear()
            calls.clear()
            destination = root/lifetime
            with contextlib.redirect_stdout(io.StringIO()):
                assert run_batch(manifest, destination, worker_lifetime=lifetime, runner=run,
                    pool_factory=FakePool, parser=parser, validator=validate) == 1
            result = json.loads((destination/'summary.json').read_text())
            assert len(created) == count and all(pool.closed for pool in created)
            assert len(calls) == 3 and result['metrics']['attempted_episodes'] == 3
            assert result['metrics']['qualified_trajectories'] == 1
            assert result['metrics']['failed_episodes'] == 2 and result['metrics']['official_successes'] == 2
            assert result['metrics']['official_success_rate_all_allocated'] == 2/3
            assert result['metrics']['official_success_rate_observed'] == 1.
            assert result['metrics']['official_unknown_episodes'] == 1
            assert 'synthetic runner failure' in result['episodes'][2]['error']
    assert rejected == 6
    return {"status": "passed", "rejected_manifests": rejected, "worker_lifetimes_tested": ["episode", "batch"],
            "failed_episode_retained": True, "gpu_initialized": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--worker-lifetime', choices=('episode', 'batch'), default='batch')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        print(json.dumps(self_test(), allow_nan=False))
        return 0
    if args.manifest is None or args.output_dir is None:
        parser.error('--manifest and --output-dir are required')
    try:
        return run_batch(args.manifest, args.output_dir, worker_lifetime=args.worker_lifetime)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
