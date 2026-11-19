from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from .core import DEFAULT_ROOT, activate, read


def main():
    parser = argparse.ArgumentParser(description="Unified ReSimPair pipeline with an explicit planning boundary")
    parser.add_argument("command", choices=["prepare", "prepare-workbench", "prepare-assets", "prepare-novel", "novel-phase", "register-novel-phase", "register-novel-repairs", "prepare-scene-pilot", "scene-batch", "scene-package", "benchmark-prepare", "benchmark-preflight", "freeze", "interface", "planning-input", "register-planning", "replay-protocols", "replay-hand-bindings", "profile", "audit-reviews", "intake-source", "intake-case", "extract", "simulate", "worker", "compile", "run", "suite", "report"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--profile-roots", type=Path, nargs="+")
    parser.add_argument("--ancestor-root", type=Path)
    parser.add_argument("--release")
    parser.add_argument("--phase", choices=["discovery","validation","repair"], default="discovery")
    parser.add_argument("--source")
    parser.add_argument("--case")
    parser.add_argument("--attempt")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--scene-file", type=Path)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--make-reference", action="store_true")
    parser.add_argument("--scene-id", default="mechanism_workbench_C")
    parser.add_argument("--seed", type=int, default=2026102317)
    args = parser.parse_args()
    if args.command == "prepare":
        from .experiment import prepare
        result = prepare(args.root)
    elif args.command == "prepare-novel":
        from .novel_campaign import prepare
        if not args.ancestor_root: parser.error("prepare-novel requires --ancestor-root")
        result = prepare(args.root,args.ancestor_root)
    elif args.command == "register-novel-phase":
        from .novel_campaign import register_phase
        result = register_phase(args.root,args.phase)
    elif args.command == "novel-phase":
        from .novel_campaign import run_phase
        activate()
        result = run_phase(args.root,args.phase)
    elif args.command == "register-novel-repairs":
        from .novel_campaign import register_repairs
        if not args.task_file:parser.error("register-novel-repairs requires --task-file with a bound repair request")
        result=register_repairs(args.root,args.task_file)
    elif args.command == "prepare-scene-pilot":
        from .scene_pilot import prepare
        if not args.ancestor_root:
            parser.error("prepare-scene-pilot requires --ancestor-root")
        result = prepare(args.root, args.ancestor_root, args.seed)
    elif args.command in {"scene-batch", "scene-package"}:
        from .scene_pilot import run_batch, package
        activate()
        result = run_batch(args.root) if args.command == "scene-batch" else package(args.root)
    elif args.command == "profile":
        from .profiling import audit
        if not args.output:
            parser.error("profile requires --output outside the retained campaign roots")
        result = audit(args.profile_roots or [args.root], args.output)
    elif args.command == "benchmark-prepare":
        from .benchmark import prepare
        if not args.ancestor_root:
            parser.error("benchmark-prepare requires --ancestor-root")
        result = prepare(args.root, args.ancestor_root, args.seed)
    elif args.command == "prepare-assets":
        from .asset_experiment import prepare
        if not args.ancestor_root:
            parser.error("prepare-assets requires --ancestor-root")
        result = prepare(args.root, args.ancestor_root)
    elif args.command == "benchmark-preflight":
        from .benchmark import preflight
        result = preflight(args.root)
    elif args.command == "prepare-workbench":
        from .workbench import prepare
        result = prepare(args.root, args.scene_id, args.seed)
    elif args.command == "freeze":
        from .experiment import freeze
        if not args.release:
            parser.error("--release required")
        result = freeze(args.root, args.release)
    elif args.command == "interface":
        from .interface import publish
        result = {"interfaces": publish(args.root)}
    elif args.command == "replay-protocols":
        from .compatibility import replay
        if not args.output:
            parser.error("replay-protocols requires --output")
        result = replay(args.root, args.output)
    elif args.command == "replay-hand-bindings":
        from .hand_regression import replay
        if not args.output:
            parser.error("replay-hand-bindings requires --output")
        result = replay(args.root, args.output)
    elif args.command == "register-planning":
        from .intake import register_planning
        if not all([args.case, args.bundle]):
            parser.error("register-planning requires --case --bundle")
        result = register_planning(args.root, args.case, args.bundle, args.make_reference)
    elif args.command == "intake-source":
        from .intake import register_source
        if not all([args.source, args.video, args.session, args.split]):
            parser.error("intake-source requires --source --video --session --split")
        result = register_source(args.root, args.source, args.video, args.session, args.split, args.start, args.end)
    elif args.command == "intake-case":
        from .intake import register_case
        if not all([args.case, args.scene_file, args.task_file]):
            parser.error("intake-case requires --case --scene-file --task-file")
        result = register_case(args.root, args.case, args.scene_file, args.task_file, args.make_reference)
    else:
        activate()
        if args.command == "planning-input":
            from .source import planning_input
            result = planning_input(args.root, args.source, execute=args.execute)
        elif args.command == "audit-reviews":
            from .review_audit import run
            result = run(args.root, execute=args.execute)
        elif args.command == "extract":
            from .source import extract
            from .budget import Budget
            from .intake import sources
            if not args.execute:
                result = {"status": "ready", "sources": [s["id"] for s in sources(args.root)]}
            else:
                results = []
                for s in sources(args.root):
                    if args.source and args.source != s["id"]:
                        continue
                    record = extract(args.root, s["id"], Budget(args.root / "budget.json"))
                    results.append({"source": s["id"], "skills": record["validated_skills"], "status": record["status"]})
                result = {"sources": results}
        elif args.command in {"simulate", "worker"}:
            from .simulation import launch, worker
            result = (worker if args.command == "worker" else launch)(args.root, args.case, args.attempt or "attempt_01")
        elif args.command == "compile":
            from .compiler import compile_execution
            result = compile_execution(args.root, args.case, args.attempt)
        elif args.command in {"run", "suite"}:
            from .pipeline import run, suite
            result = run(args.root, args.case, args.attempt, execute=args.execute) if args.command == "run" else suite(args.root, execute=args.execute)
        else:
            from .report import report
            result = report(args.root)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Do not print raw exception payloads from network providers.
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__,
                          "code": getattr(exc, "code", "local_validation_or_execution_error"),
                          "detail": str(exc) if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError, AttributeError, KeyError, TypeError)) and not hasattr(exc, "code") else None,
                          "local_trace": [{"file": f.filename, "line": f.lineno, "function": f.name}
                                          for f in traceback.extract_tb(exc.__traceback__)] if not hasattr(exc, "code") else []}), flush=True)
        raise SystemExit(2)
