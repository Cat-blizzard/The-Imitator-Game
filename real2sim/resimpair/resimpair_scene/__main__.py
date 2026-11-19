"""Local scene tools; all new physical attempts use the existing ledger."""
import argparse
from contextlib import contextmanager, redirect_stdout
import ctypes
import json
import os
from pathlib import Path
import sys

MAX_REQUEST_BYTES = 2**20


@contextmanager
def diagnostic_stdout():
    """Keep native C libraries and Python progress out of JSON stdout."""
    sys.stdout.flush()
    original_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield
    finally:
        # Linux simulator workers may print through libc, outside sys.stdout.
        ctypes.CDLL(None).fflush(None)
        sys.stderr.flush()
        os.dup2(original_fd, 1)
        os.close(original_fd)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


def request_file(path):
    if path.stat().st_size > MAX_REQUEST_BYTES:
        raise ValueError("request_exceeds_one_MiB")
    return json.loads(path.read_text())


def invoke(method, params):
    from .api import call
    # Native workers may print progress; reserve stdout for the JSON response.
    with diagnostic_stdout():
        return call(method, params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tools")
    sub.add_parser("serve", help="local JSON lines; not an MCP server")
    p = sub.add_parser("call")
    p.add_argument("method")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--arguments", type=Path, help="JSON object file")
    group.add_argument("--json", help="JSON object")
    for name in ("measure", "status", "observe", "diagnose"):
        p = sub.add_parser(name); p.add_argument("path")
        if name in ("measure", "observe"):
            p.add_argument("--frame", type=int)
        if name in ("observe", "diagnose"):
            p.add_argument("--operation")
        if name in ("status", "observe", "diagnose"):
            p.add_argument("--independent-audit")
    p = sub.add_parser("repair")
    p.add_argument("execution"); p.add_argument("--output", required=True)
    p.add_argument("--operation"); p.add_argument("--maximum-candidates", type=int, default=24)
    p.add_argument("--maximum-seconds", type=float, default=90)
    p = sub.add_parser("apply")
    p.add_argument("--proposals", required=True); p.add_argument("--proposal-id", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("campaign"); p.add_argument("path")
    p = sub.add_parser("prepare")
    p.add_argument("--campaign", required=True); p.add_argument("--name", required=True)
    p.add_argument("--cases", type=Path, required=True); p.add_argument("--profile", default="checked_withdrawal")
    p = sub.add_parser("run")
    p.add_argument("--cohort", required=True); p.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    if args.command == "tools":
        from .api import capabilities
        emit(capabilities()); return
    if args.command == "serve":
        while True:
            line = sys.stdin.readline(MAX_REQUEST_BYTES + 1)
            if not line:
                break
            request = None
            try:
                if len(line.encode()) > MAX_REQUEST_BYTES:
                    while line and not line.endswith("\n"):
                        line = sys.stdin.readline(MAX_REQUEST_BYTES + 1)
                    raise ValueError("request_exceeds_one_MiB")
                request = json.loads(line)
                if not isinstance(request, dict) or set(request) - {"id", "method", "params"}:
                    raise ValueError("expected_id_method_params_object")
                result = invoke(request["method"], request.get("params", {}))
                emit({"id": request.get("id"), "result": result})
            except Exception as error:
                emit({"id": request.get("id") if isinstance(request, dict) else None,
                      "error": {"type": type(error).__name__, "message": str(error)}})
            sys.stdout.flush()
        return
    if args.command == "call":
        if args.json is not None and len(args.json.encode()) > MAX_REQUEST_BYTES:
            raise ValueError("request_exceeds_one_MiB")
        params = request_file(args.arguments) if args.arguments else json.loads(args.json)
        result = invoke(args.method, params)
    elif args.command in ("measure", "status", "observe", "diagnose"):
        method = {"measure": "scene_measure", "status": "run_status",
                  "observe": "run_observe", "diagnose": "run_diagnose"}[args.command]
        params = {"source" if args.command == "measure" else "execution": args.path}
        params.update({name: getattr(args, name) for name in ("frame", "operation", "independent_audit")
                       if getattr(args, name, None) is not None})
        result = invoke(method, params)
    elif args.command == "repair":
        result = invoke("repair_propose", {k: v for k, v in vars(args).items() if k != "command"})
    elif args.command == "apply":
        result = invoke("scene_apply_revision", {k: v for k, v in vars(args).items() if k != "command"})
    elif args.command == "campaign":
        result = invoke("campaign_status", {"campaign": args.path})
    elif args.command == "prepare":
        cases = request_file(args.cases)
        if isinstance(cases, dict):
            cases = cases.get("cases", [cases])
        result = invoke("execution_prepare", {"campaign": args.campaign, "name": args.name,
                                             "cases": cases, "profile": args.profile})
    else:
        result = invoke("execution_run", {"cohort": args.cohort, "limit": args.limit})
    emit(result)


if __name__ == "__main__":
    main()
