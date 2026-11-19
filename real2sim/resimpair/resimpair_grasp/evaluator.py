"""Versioned acceptance, frozen before attempts; never imported by controller.

Retains v4 event thresholds and independent FK. Adds asset identity, native
collider and both-camera full-timeline checks for the new geometry interface.
"""
from pathlib import Path
import numpy as np
import trimesh

from .common import binding, read, resolve, write

VERSION = "generic_mesh_acceptance_v1"


def bindings():
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import event_protocol, compiler, planning_protocols, media
    return {Path(p).name: binding(p) for p in [__file__, event_protocol.__file__, compiler.__file__, planning_protocols.__file__, media.__file__]}


def geometry(evidence, scene):
    rows = read(evidence)["objects"]; result = []
    for row, obj in zip(rows, scene["objects"]):
        d = read(resolve(obj["asset_descriptor"]))
        if row["id"] != obj["id"] or row["descriptor"] != obj["asset_descriptor"]:
            raise ValueError("Native geometry identity mismatch")
        if len(row["shapes"]) != len(d["collisions"]): raise ValueError("Native collision part count mismatch")
        errors = []
        for actual, expected in zip(row["shapes"], d["collisions"]):
            vertices = trimesh.load(resolve(expected), force="mesh", process=False).vertices
            cooked = np.asarray(actual["vertices"])
            distance = np.linalg.norm(vertices[:, None] - cooked[None, :], axis=-1)
            error = float(max(distance.min(0).max(), distance.min(1).max()))
            if error >= 1e-5 or not np.allclose(actual["scale"], [1, 1, 1]) or not np.allclose(actual["pose"], np.eye(4), atol=1e-6):
                raise ValueError("Native collider differs from declared cache geometry")
            if abs(actual["density"] - d["density_kg_m3"]) >= 1e-3: raise ValueError("Native density mismatch")
            errors.append(error)
        if obj["kind"] == "movable" and abs(row["mass"] - d["mass_kg"]) >= 1e-4:
            raise ValueError("Native mass mismatch")
        result.append({"id": obj["id"], "max_vertex_error_m": max(errors), "passed": True})
    if len(rows) != len(scene["objects"]): raise ValueError("Missing native object")
    return {"passed": True, "objects": result}


def evaluate(directory):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import event_protocol, compiler, media
    directory = Path(directory); output = directory / "audit.json"
    if output.exists(): return read(output)
    result = {"rule_version": VERSION, "rule_bindings": bindings(), "passed": False, "checks": {}, "errors": [], "api_calls": 0}
    bundle_path = directory / "execution_bundle.json"
    if not bundle_path.exists():
        result["errors"].append("no_native_execution_bundle"); write(output, result); return result
    b = read(bundle_path)
    result["bundle"] = binding(bundle_path)
    if b["planning_failure"] is not None:
        result["errors"].append(b["planning_failure"]["message"])
    try:
        from corl2026_workshop.experiments.autonomous_pipeline_v4.core import checked
        with np.load(checked(b["measurements"]), allow_pickle=False) as f:
            arrays = {k: f[k] for k in f.files}
        measured = event_protocol.compile_events(read(checked(b["event_protocol"])), arrays)
        result["checks"]["task_events"] = measured["qualification"]
        result["checks"]["kinematics"] = compiler.verify_native_kinematics(arrays)
        result["checks"]["geometry"] = geometry(directory / "native_geometry.json", b["scene"])
        cameras = []
        for name, c in b["cameras"].items():
            packet = media.packet(checked(c["video"]), directory / "audit_frames" / name, frame_budget=9)
            valid = (packet["decoded_frames"] == len(arrays["timestamps_seconds"]) and
                np.allclose(packet["all_native_pts_seconds"], arrays["timestamps_seconds"], atol=1e-5))
            cameras.append({"id": name, "frames": packet["decoded_frames"], "passed": bool(valid)})
        result["checks"]["synchronization"] = {"passed": len(cameras) == 2 and all(x["passed"] for x in cameras), "cameras": cameras}
        planned = read(directory / "planned_paths.json")["paths"]
        decisions = [read(x["guard_evidence"]) for x in planned]
        result["checks"]["executed_paths_guarded"] = {"passed": bool(planned) and all(x["accepted"] for x in decisions),
            "path_count": len(planned), "dense_samples": sum(x.get("report", {}).get("dense_samples", 0) for x in decisions),
            "scope": "live checked dense commands with immutable decisions; not a formal continuous-time proof"}
        result["passed"] = not result["errors"] and all(c["passed"] for c in result["checks"].values())
        result["event_frames"] = [{k: x[k] for k in ["id", "frame"]} for x in measured["events"]]
    except Exception as exc:
        result["errors"].append(repr(exc))
    write(output, result)
    return result
