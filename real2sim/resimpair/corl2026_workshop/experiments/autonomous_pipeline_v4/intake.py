"""Register new media and declarative scene/tasks without editing pipeline code."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess

from .core import bind, identifier, read, require, write
from .library import TASK_TEMPLATES, task_semantic_key


def sources(root):
    root = Path(root)
    rows = {s["id"]: s for s in read(root / "experiment.json")["sources"]}
    for p in sorted((root / "registered_sources").glob("*.json")):
        row = read(p)
        require(row["id"] not in rows or rows[row["id"]] == row, "Source ID already binds another video/window")
        rows[row["id"]] = row
    return list(rows.values())


def cases(root):
    root = Path(root)
    rows = {c["id"]: c for c in read(root / "experiment.json")["cases"]}
    for p in sorted((root / "registered_cases").glob("*.json")):
        row = read(p)
        require(row["id"] not in rows or rows[row["id"]] == row, "Case ID already binds another scene/task")
        rows[row["id"]] = row
    return list(rows.values())


def register_source(root, source_id, video, session, split, start=0, end=None):
    root = Path(root)
    identifier(source_id)
    require(source_id not in {s["id"] for s in sources(root)}, "Use a distinct new source ID")
    require(session and split in {"train", "validation", "test"}, "Source session and inherited data split required")
    require(all(s["split"] == split for s in sources(root) if s["session"] == session), "One source session cannot cross data splits")
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "json", str(video)], capture_output=True, text=True, check=True, timeout=120)
    total = int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"])
    end = total if end is None else end
    require(0 <= start < end <= total, "Selected source window exceeds media")
    item = {"id": source_id, "video": bind(video), "window_frames": [start, end], "session": session, "split": split,
            "selection_origin": "user_or_dataset_selected_media", "semantic_annotation_supplied": False}
    write(root / "registered_sources" / (source_id + ".json"), item)
    return item


def register_case(root, case_id, scene_path, task_path, make_reference=False):
    root = Path(root)
    identifier(case_id)
    require(case_id not in {c["id"] for c in cases(root)}, "Use a distinct new case ID")
    scene, task = read(scene_path), read(task_path)
    require(scene["schema"] == "resimpair_tabletop_scene_v4" and task["schema"] == "resimpair_task_intent_v4",
            "Declarative scene and task schemas required")
    require(task["family"] in TASK_TEMPLATES and task["id"] == case_id, "Task family or ID mismatch")
    require(all(o["shape"] in {"box", "convex_mesh"} for o in scene["objects"]), "Unsupported scene collision objects")
    for obj in scene["objects"]:
        if obj["shape"] == "convex_mesh":
            from .active_assets import validate_object
            validate_object(obj)
    require(all(o["shape"] in {"tray", "platform"} for o in scene["supports"]), "Unsupported support geometry plugin")
    require(set(scene["cameras"]) == {"robot_side", "opposite_high"}, "Both standard cameras required")
    require(all(o["primitive"] == "pick_place" for o in task["operations"]), "Unsupported planning primitive")
    row = {"id": case_id, "scene": bind(scene_path), "task": bind(task_path), "semantic_key": task_semantic_key(task),
           "partition": "new_registered_input", "request_human_reference": bool(make_reference),
           "reference_policy": "one_or_two_per_task_semantics_not_per_nuisance_variant"}
    write(root / "registered_cases" / (case_id + ".json"), row)
    return row


def register_planning(root, case_id, bundle_path, make_reference=False):
    """Accept any protocol-capable planner without extending the scene CLI plugin."""
    from .interface import validate_bundle
    from .core import checked
    from .event_protocol import validate
    root = Path(root)
    identifier(case_id)
    bundle = read(bundle_path)
    validate_bundle(bundle)
    require(bundle["schema"] == "resimpair_execution_bundle_v5", "External planners must use the complete event protocol")
    require(bundle["case"]["id"] == case_id and bundle["planning_failure"] is None, "Invalid external planning handoff")
    validate(read(checked(bundle["event_protocol"])))
    declaration = read(checked(bundle["planning_declaration"]))
    require(declaration["scope"] == "before_physical_execution", "Replay evidence cannot be registered as a new autonomous task")
    row = {**bundle["case"], "request_human_reference": bool(make_reference),
           "planner_plugin": "external_protocol_export", "reference_policy": "one_or_two_per_task_semantics_not_per_nuisance_variant"}
    known = {c["id"]: c for c in cases(root)}
    require(case_id not in known or known[case_id] == row, "Case ID already has a different definition")
    write(root / "registered_cases" / (case_id + ".json"), row)
    handoff = {"schema": "resimpair_planner_handoff_v5", "case": case_id, "attempt": "external_export",
               "execution_bundle": bind(bundle_path), "manual_downstream_annotations": False}
    write(root / "planner_handoffs" / (case_id + ".json"), handoff)
    return {"status": "planner_export_registered", "handoff": handoff}
