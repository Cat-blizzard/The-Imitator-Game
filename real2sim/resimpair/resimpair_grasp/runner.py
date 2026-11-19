"""Finite CPU/GPU campaign supervisor; reservations survive failure and timeout."""
from pathlib import Path
from collections import Counter
import contextlib
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

from .common import ROOT, binding, code_bindings, digest, read, resolve, write
from corl2026_workshop.experiments.autonomous_pipeline_v4 import core


def init(a):
    from . import evaluator
    from .scenes import sample
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
    if a.root.exists(): raise ValueError("Campaign exists; resume it or use a new version")
    if not (1 <= a.layouts <= 5 and 1 <= a.physical_limit <= 300): raise ValueError("Finite campaign limits required")
    if not a.methods or set(a.methods) - {"baseline_obb", "cached_antipodal"}: raise ValueError("Unknown method")
    snapshot = resources.gpu_snapshot(a.gpu)
    if shutil.disk_usage(ROOT).free < 10 * 2**30: raise ValueError("Need at least 10 GiB free")
    entries = read(a.cache_index)["entries"]
    if a.asset:
        entries = [x for x in entries if read(resolve(x["asset"]))["source_id"] in a.asset or read(resolve(x["asset"]))["id"] in a.asset]
    if not entries: raise ValueError("No matching assets")
    a.root.mkdir(parents=True)
    cbind = code_bindings()
    for name, b in cbind.items():
        target = a.root / "release" / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(resolve(b), target)
    write(a.root / "rules.json", {"version": evaluator.VERSION, "files": evaluator.bindings(), "frozen_before_attempts": True})
    write(a.root / "experiment.json", {"schema": "resimpair_generic_grasp_campaign_v1", "gpu": a.gpu, "gpu_uuid": snapshot["uuid"],
        "physical_limit": a.physical_limit, "timeout_seconds": 300, "max_concurrent": 1,
        "api_limit": 0, "api_cost_usd": 0, "automatic_retry": False, "source_coverage": "synthetic_only",
        "new_training_runs": 0, "code": cbind, "legacy_code": core.implementation(),
        "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "initial_snapshot": snapshot, "seed": a.seed, "catalog": binding(a.catalog), "cache_index": binding(a.cache_index)})
    cases = []
    for ai, entry in enumerate(entries):
        for li in range(a.layouts):
            scene, tasks = sample(a.catalog, entry["asset"], entry["cache"], a.seed + ai * 100 + li, li)
            scene_path = a.root / "inputs" / (scene["id"] + ".json"); write(scene_path, scene)
            # Both task goals are registered before execution. Default batch
            # selects tray placement; second goal exercises shared scene reuse.
            for task in tasks:
                task_path = a.root / "inputs" / (task["id"] + ".json"); write(task_path, task)
                for method in a.methods:
                    case = {"id": task["id"], "execution_id": task["id"] + "__" + method,
                        "scene": core.bind(scene_path), "task": core.bind(task_path), "method": method,
                        "semantic_key": digest(task), "split_group": scene["split_group"], "partition": scene["partition"],
                        "source_coverage": "synthetic_only", "asset": entry["asset"], "layout": li}
                    cases.append(case)
    write(a.root / "cohort.json", {"cases": cases, "attempted_assets": len(entries), "failures_remain_in_denominator": True,
        "same_scene_across_methods_and_tasks": True, "physical_budget_counts_failed_and_unknown": True})
    return {"root": str(a.root), "registered_cases": len(cases), "physical_limit": a.physical_limit, "api_limit": 0}


def verify(root):
    spec = read(root / "experiment.json")
    for b in spec["code"].values(): resolve(b)
    for b in spec["legacy_code"].values(): core.checked(b)
    for b in read(root / "rules.json")["files"].values(): resolve(b)
    return spec


def run(a):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
    root = a.root.resolve(); spec = verify(root)
    cases = read(root / "cohort.json")["cases"]
    if a.case: cases = [c for c in cases if c["execution_id"] in a.case]
    else: cases = [c for c in cases if c["id"].endswith("tray_place")]
    if not 1 <= a.limit <= spec["physical_limit"]: raise ValueError("Bounded launch count required")
    results = []
    for case in cases:
        directory = root / "executions" / case["execution_id"]
        if (directory / "terminal.json").exists(): continue
        if len(results) >= a.limit: break
        if case["method"] == "cached_antipodal":
            scene = core.read(core.checked(case["scene"]))
            if any(not read(resolve(scene["grasp_caches"][o]))["candidates"] for o in core.read(core.checked(case["task"]))["object_ids"]):
                terminal = {"returncode": "preflight_rejected", "reason": "no_geometric_grasp_candidates",
                    "elapsed_seconds": 0., "physical_attempts": 0, "api_calls": 0, "case": case["execution_id"]}
                write(directory / "eligibility_rejection.json", {"case": case, "reason": terminal["reason"], "retained_in_attempt_denominator": True})
                write(directory / "terminal.json", terminal); results.append(terminal); print(terminal, flush=True)
                continue
        with contextlib.ExitStack() as stack:
            for path in resources.lock_paths(spec["gpu"]):
                f = stack.enter_context(path.open("r")); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify(root)
            if (directory / "reservation.json").exists(): raise ValueError("Unknown prior attempt; recover without restarting")
            if len(list((root / "executions").glob("*/reservation.json"))) >= spec["physical_limit"]: raise ValueError("Physical budget exhausted")
            snapshot = resources.gpu_snapshot(spec["gpu"]); resources.check_idle(snapshot)
            if snapshot["uuid"] != spec["gpu_uuid"]: raise ValueError("GPU identity changed")
            if shutil.disk_usage(root).free < 10 * 2**30: raise ValueError("Insufficient disk space")
            write(directory / "reservation.json", {"case": case, "experiment": binding(root / "experiment.json"), "gpu": snapshot, "started_unix": time.time()})
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1", RESIMPAIR_GRASP_RESERVATION=str(directory / "reservation.json"))
            env.pop("CUDA_VISIBLE_DEVICES", None)
            start = time.monotonic()
            with (directory / "worker.log").open("x") as log:
                p = subprocess.Popen([sys.executable, "-u", "-m", "resimpair_grasp", "worker", "--root", str(root), "--case", case["execution_id"]],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(directory / "process.json", {"pid": p.pid, "pgid": p.pid, "started_unix": time.time()})
                try:
                    code = p.wait(timeout=spec["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=8)
                    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
                    code = "timeout"
            terminal = {"returncode": code, "elapsed_seconds": time.monotonic() - start,
                "physical_attempts": 1, "api_calls": 0, "case": case["execution_id"]}
            write(directory / "terminal.json", terminal)
            results.append(terminal); print(terminal, flush=True)
    return {"executions": results}


def native_geometry(rec, directory):
    records = []
    for actor, item in zip(rec.objects, rec.scene_spec["objects"]):
        body = actor._bodies[0]
        rows = [{"type": type(s).__name__, "vertices": np.asarray(s.vertices).tolist(), "scale": np.asarray(s.scale).tolist(),
                 "pose": np.asarray(s.local_pose.to_transformation_matrix()).tolist(), "density": float(s.density)} for s in body.get_collision_shapes()]
        records.append({"id": item["id"], "descriptor": item["asset_descriptor"], "shapes": rows,
                        "mass": float(body.mass) if item["kind"] == "movable" else None})
    write(directory / "native_geometry.json", {"objects": records})


def worker(a):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import simulation, planning_protocols
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator, span
    root = a.root.resolve(); spec = verify(root)
    case = next(c for c in read(root / "cohort.json")["cases"] if c["execution_id"] == a.case)
    directory = root / "executions" / a.case
    if os.environ.get("RESIMPAIR_GRASP_RESERVATION") != str(directory / "reservation.json"): raise ValueError("Supervisor reservation required")
    from examples.baselines.hypericl_task_expansion.common import bind_gpu
    gpu = bind_gpu(spec["gpu"])
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    import gymnasium as gym
    scene, task = core.read(core.checked(case["scene"])), core.read(core.checked(case["task"]))
    object_ids = [o["id"] for o in scene["objects"]]
    support_ids = object_ids + [s["id"] for s in scene["supports"]] + ["table"]
    write(directory / "event_protocol.json", planning_protocols.pick_place(scene, task, object_ids, support_ids, simulation.targets(scene, task)))
    write(directory / "planning_declaration.json", {"event_protocol": core.bind(directory / "event_protocol.json"), "case": case,
        "scope": "before_physical_execution", "declared_unix": time.time(), "implementation": core.implementation()})
    performance = Accumulator(); env = rec = None; failure = bundle = None
    try:
        with span(directory, "simulation_environment_setup"):
            simulation.register(scene, performance)
            env = gym.make("ReSimPairSharedTable-v4", sim_backend="physx_cpu", render_backend=gpu["render_backend"],
                control_mode="pd_joint_pos", obs_mode="rgb", reward_mode="none", render_mode="rgb_array", sensor_configs={"shader_pack": "rt-fast"})
        with span(directory, "simulation_reset_and_recorders"):
            performance.phase = "reset"; rec = simulation.Recorder(env, directory); rec.reset(scene["seed"])
            native_geometry(rec, directory)
        with span(directory, "planning_execution_and_rendering"):
            from .motion import solve
            performance.phase = "execution"; solve(rec, task, case["method"])
    except Exception as exc:
        import traceback
        failure = {"type": type(exc).__name__, "message": str(exc), "trace": traceback.format_exc()}
        write(directory / "planning_failure.json", failure)
    finally:
        if rec is not None:
            with span(directory, "simulation_finalize"):
                performance.phase = "finalize"; rec.finish()
                bundle = simulation.export(rec, root, case, task, scene, directory, failure)
        if env is not None: env.close()
        performance.persist(directory / "timing_breakdown.json")
    result = {"bundle": bundle, "failure": failure, "method": case["method"], "api_calls": 0}
    write(directory / "worker_result.json", result)
    return result


def audit(a):
    from . import evaluator
    root = a.root.resolve(); verify(root)
    rows = []
    for case in read(root / "cohort.json")["cases"]:
        directory = root / "executions" / case["execution_id"]
        if not (directory / "reservation.json").exists() and not (directory / "terminal.json").exists(): continue
        report = evaluator.evaluate(directory)
        row = {"case": case["execution_id"], "method": case["method"], "partition": case["partition"],
               "source_id": read(resolve(case["asset"]))["source_id"], "layout": case["layout"],
               "passed": report["passed"], "audit": binding(directory / "audit.json"), "errors": report["errors"]}
        if (directory / "terminal.json").exists():
            terminal = read(directory / "terminal.json")
            row["elapsed_seconds"] = terminal["elapsed_seconds"]
            row["physical_attempts"] = terminal["physical_attempts"]
            row["preflight_reason"] = terminal.get("reason")
        else:
            row["physical_attempts"] = 1
        rows.append(row)
    counts = {m: {"attempted": sum(r["method"] == m for r in rows), "passed": sum(r["method"] == m and r["passed"] for r in rows)} for m in {r["method"] for r in rows}}
    result = {"schema": "generic_grasp_audit_summary_v1", "counts": counts, "rows": rows,
        "physical_attempts": sum(r["physical_attempts"] for r in rows), "attempted_cases": len(rows), "api_calls": 0, "real_source_qualified_pairs": 0,
        "formal_training_admission": False, "rule_version": evaluator.VERSION}
    output = root / "audits" / (digest(result)[:16] + ".json"); write(output, result)
    return {"summary": str(output), **result}


def compile(a):
    from . import evaluator
    from corl2026_workshop.experiments.autonomous_pipeline_v4.generic_compiler import compile_bundle
    root = a.root.resolve(); verify(root)
    directory = root / "executions" / a.case
    if not evaluator.evaluate(directory)["passed"]: raise ValueError("Native qualification required before human conditioning")
    mapping = {"correspondence": "synthetic_only", "source_status": "missing", "formal_pair_qualified": False,
               "scope": "generic mechanism coverage; no real-source equivalence inferred"}
    result = compile_bundle(directory / "execution_bundle.json", mapping)
    write(directory / "compiled" / "contract.json", result)
    return {"contract": str(directory / "compiled" / "contract.json"), "api_calls": 0, "generation_started": False}
