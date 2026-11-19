"""Finite, zero-API asset preparation and generic grasp experiments."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, binding, read, write


def prepare(a):
    from .assets import index_raw, key
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    request = {"index": binding(a.index), "selection": binding(a.selection),
               "asset_timeout_seconds": a.timeout, "api_budget_usd": 0, "cpu_concurrency": 1}
    write(root / "preparation_request.json", request)
    if not (root / "raw_index.sqlite").exists():
        print(index_raw(a.index, root / "raw_index.sqlite"), flush=True)
    selections = read(a.selection)["assets"]
    records, failures = [], []
    for i, row in enumerate(selections):
        directory = root / key(row[0]); result = directory / "descriptor.json"
        if result.exists():
            records.append(binding(result)); continue
        if directory.exists():
            failures.append({"source_id": row[0], "failure": "prior_incomplete_import_preserved"}); continue
        start = time.monotonic()
        with (root / f"import_{i:02d}.log").open("x") as f:
            command = [sys.executable, "-m", "resimpair_grasp", "prepare-one", "--index", str(a.index),
                       "--selection", str(a.selection), "--number", str(i), "--output", str(root)]
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGKILL); process.wait(); code = "timeout"
        if code == 0 and result.exists():
            records.append(binding(result)); status = "prepared"
        else:
            failures.append({"source_id": row[0], "returncode": code, "log": binding(root / f"import_{i:02d}.log")}); status = "failed"
        print(row[0], status, round(time.monotonic() - start, 2), flush=True)
    catalog = root / "catalog.json"
    write(catalog, {"schema": "resimpair_mesh_catalog_v1", "assets": records, "failures": failures,
        "attempted_assets": len(selections), "physics_qualified_count": 0,
        "all_failures_retained": True, "api_calls": 0})
    return {"catalog": str(catalog), "prepared": len(records), "failed": len(failures)}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    for name in ["prepare", "prepare-one"]:
        q = sub.add_parser(name); q.add_argument("--index", type=Path, required=True)
        q.add_argument("--selection", type=Path, default=ROOT / "configs/grasp_asset_selection.json")
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--timeout", type=int, default=120)
        q.add_argument("--number", type=int)
    c = sub.add_parser("cache"); c.add_argument("--catalog", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True); c.add_argument("--partition", default="development")
    c.add_argument("--asset", action="append", default=[])
    f = sub.add_parser("refine"); f.add_argument("--catalog", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("select"); s.add_argument("--catalog", type=Path, required=True)
    s.add_argument("--role", default="rigid_manipulation_candidate")
    s.add_argument("--category", action="append"); s.add_argument("--max-dimension", type=float)
    for name in ["init", "run", "audit", "compile"]:
        q = sub.add_parser(name); q.add_argument("--root", type=Path, required=True)
        if name == "init":
            q.add_argument("--catalog", type=Path, required=True); q.add_argument("--cache-index", type=Path, required=True)
            q.add_argument("--gpu", type=int, default=2); q.add_argument("--layouts", type=int, default=2)
            q.add_argument("--asset", action="append", default=[]); q.add_argument("--methods", nargs="+", default=["baseline_obb", "cached_antipodal"])
            q.add_argument("--physical-limit", type=int, default=24); q.add_argument("--seed", type=int, default=20261026)
        if name == "run":
            q.add_argument("--case", action="append", default=[]); q.add_argument("--limit", type=int, default=1)
        if name == "compile":
            q.add_argument("--case", required=True)
    w = sub.add_parser("worker"); w.add_argument("--root", type=Path, required=True); w.add_argument("--case", required=True)
    a = p.parse_args()
    if a.command == "prepare": result = prepare(a)
    elif a.command == "prepare-one":
        from .assets import prepare_one
        result = prepare_one(a.index, read(a.selection)["assets"][a.number], a.output)
    elif a.command == "refine":
        from .assets import refine_catalog
        result = refine_catalog(a.catalog, a.output)
    elif a.command == "select":
        from .assets import select
        result = [{"asset": b, "id": d["id"], "name": d["name"], "category": d["category"],
                   "size_m": d["dimensions_m"], "qualification": d["qualification"]}
                  for b, d in select(a.catalog, role=a.role, category=a.category, max_dimension=a.max_dimension)]
    elif a.command == "cache":
        from .grasps import generate
        from .assets import select
        results = []; failures = []
        for b, d in select(a.catalog, role="rigid_manipulation_candidate"):
            if d["partition"] != a.partition or (a.asset and d["id"] not in a.asset and d["source_id"] not in a.asset): continue
            try:
                cache = generate(b, a.output); results.append({"asset": b, "cache": cache})
                print(d["source_id"], len(read(__import__("resimpair_grasp.common", fromlist=["resolve"]).resolve(cache))["candidates"]), flush=True)
            except Exception as e:
                failures.append({"asset": b, "error": repr(e)})
        result = {"schema": "resimpair_grasp_cache_index_v1", "entries": results, "failures": failures}
        write(a.output / (a.partition + "_index.json"), result)
    else:
        from . import runner
        result = getattr(runner, a.command)(a)
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
