"""One compiler for all mechanisms expressed by the planning protocol."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .core import bind, checked, digest, implementation, read, require, verify_implementation
from .event_protocol import compile_events, load_channels, public_condition, validate
from .prompts import ACTOR


def generation_information(protocol, measured, arrays, camera):
    from .compiler import project
    from .hand_correspondence import attach, compile_correspondence
    correspondence = compile_correspondence(protocol, arrays, camera)
    channels = load_channels(protocol, arrays)
    inventory = []
    for entity in protocol["entities"]:
        item = {k: entity[k] for k in ["id", "description", "kind", "role", "geometry"]}
        if "pose_channel" in entity:
            pose = channels[entity["pose_channel"]]
            positions = pose[..., :3, 3] if pose.ndim == 3 else pose[:, :3]
            item["initial_image_xy_normalized"] = project(positions[0], camera)
            item["final_image_xy_normalized"] = project(positions[-1], camera)
        elif "fixed_position_world" in entity:
            item["initial_image_xy_normalized"] = project(entity["fixed_position_world"], camera)
        inventory.append(item)
    goals, location_goals = [], []
    for goal in protocol["goals"]:
        item = public_condition(goal)
        item["parameters"] = deepcopy(item["parameters"])
        position = item["parameters"].get("target_position_world")
        if position is not None:
            projected = project(position, camera)
            item["parameters"]["target_image_xy_normalized"] = projected
            location_goals.append({"entity": goal["entities"][0], "image_xy": projected})
        goals.append(item)
    spatial = []
    # Relational wording is derived for every scene, never a sample repair note.
    if len(location_goals) >= 2:
        xy = np.asarray([g["image_xy"] for g in location_goals])
        if np.ptp(xy[:, 1]) < .05 and np.min(np.diff(np.sort(xy[:, 0]))) > .03:
            spatial.append({"relation": "left_to_right_in_image", "entities":
                            [g["entity"] for g in sorted(location_goals, key=lambda g: g["image_xy"][0])]})
    duration = protocol["duration_seconds"]
    native_duration = float(arrays["timestamps_seconds"][-1] - arrays["timestamps_seconds"][0])
    required_events = [{"id": e["id"], "action": e["phase"], "entities": e["entities"], "effector": e["effector"],
                        "suggested_seconds": round((e["seconds"] - float(arrays["timestamps_seconds"][0])) /
                                                   native_duration * (duration - 1), 2)} for e in measured["events"]]
    actor = {k: v for k, v in ACTOR.items() if k != "initial_hands"}
    shared = {"actor": actor, "effectors": protocol["effectors"], "initial_hands": protocol["initial_hands"],
              "inventory": [{k: v for k, v in item.items() if k not in {"role", "final_image_xy_normalized"}} for item in inventory],
              "initial_relations": [public_condition(c) for c in protocol["initial_conditions"]]}
    generation = {"protocol": protocol["schema"], "task_family": protocol["family"],
                  "duration_seconds": duration, "inventory": inventory, "actor": actor,
                  "effectors": protocol["effectors"], "initial_hands": protocol["initial_hands"],
                  "terminal_hands": protocol["terminal_hands"], "initial_relations": shared["initial_relations"],
                  "final_relations": goals, "spatial_goal_relations": spatial,
                  "required_events": required_events, "required_event_precedence": measured["required_event_precedence"],
                  "persistent_constraints": [public_condition(c) for c in protocol["constraints"] if c["human_visible_check"]],
                  "non_target_entities": [e["id"] for e in protocol["entities"] if e["role"] == "unchanged"],
                  "equivalence_groups": protocol["equivalence_groups"],
                  "frame_coordinates": "normalized image x from left and y from top; geometry of the specified camera, not anatomical left/right",
                  "physical_thresholds_scope": "simulation qualification; human review assesses visible relations, never infers forces or exact 3D from RGB"}
    # Only human roles enter prompts; measurement channel names stay in evidence.
    generation["effectors"] = {k: {"human_hand": v["human_hand"]} for k, v in protocol["effectors"].items()}
    from .grounding import compile_grounding
    generation["spatial_grounding"] = compile_grounding(protocol, channels, inventory, camera)
    attach(generation, shared, correspondence)
    return generation, shared


def compile_bundle(bundle_path, source_mapping):
    from .interface import validate_bundle
    from .compiler import verify_native_kinematics
    from .hand_correspondence import compile_correspondence
    bundle = read(bundle_path)
    validate_bundle(bundle)
    require(bundle["schema"] == "resimpair_execution_bundle_v5" and bundle["planning_failure"] is None,
            "successful_v5_planning_export_required")
    verify_implementation(bundle["code"])
    protocol = read(checked(bundle["event_protocol"]))
    validate(protocol)
    declaration = read(checked(bundle["planning_declaration"]))
    require(declaration["event_protocol"] == bundle["event_protocol"] and declaration["case"] == bundle["case"],
            "protocol_mismatch_with_planning_declaration")
    require(declaration["scope"] == "before_physical_execution", "retrospective_protocol_is_not_a_new_autonomous_run")
    for name, key in [("scene", "scene"), ("task", "task_intent")]:
        require(read(checked(bundle["case"][name])) == bundle[key], "planner_changed_declared_" + name)
    require(protocol["task_id"] == bundle["task_intent"]["id"] == bundle["case"]["id"], "protocol_task_identity_mismatch")
    require(set(protocol["required_skills"]) == set(bundle["task_intent"]["required_skills"]), "protocol_source_skill_mismatch")
    with np.load(checked(bundle["measurements"]), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    validate_bundle(bundle, arrays)
    measured = compile_events(protocol, arrays)
    measured["qualification"]["native_kinematics"] = verify_native_kinematics(arrays)
    require(measured["qualification"]["passed"], "declared_native_predicate_failed:" + ",".join(
        c["id"] for c in measured["qualification"]["checks"] if not c["passed"]))
    for camera in bundle["cameras"].values():
        for key in ["video", "initial_image", "final_image"]:
            checked(camera[key])
        require(camera["synchronous_with_measurements"] and camera["frame_count"] == len(arrays["timestamps_seconds"]),
                "native_camera_measurement_timeline_mismatch")
    camera = bundle["cameras"]["opposite_high"]
    generation, initial = generation_information(protocol, measured, arrays, camera)
    return {"schema": "resimpair_compiled_task_v5", "execution": bind(bundle_path),
            "event_protocol": bundle["event_protocol"], "planning_declaration": bundle["planning_declaration"],
            "source_mapping": source_mapping, "task_id": protocol["task_id"], "semantic_key": bundle["case"]["semantic_key"],
            "measured": measured, "generation_information": generation, "initial_scene_information": initial,
            "hand_correspondence_evidence": compile_correspondence(protocol, arrays, camera),
            "selected_camera": "opposite_high", "initial_image": camera["initial_image"], "final_image": camera["final_image"],
            "simulation_video": camera["video"], "other_simulation_video": bundle["cameras"]["robot_side"]["video"],
            "condition_consumption": {"initial_image": "image_model_primary_input", "initial_scene_information": "image_model_text",
                                      "generation_information": "video_and_review_model_text", "source_mapping": "skill_compatibility_and_lineage",
                                      "native_predicates": "simulation_evidence_not_human_3d_ground_truth"},
            "manual_downstream_annotation_count": 0, "code": implementation(), "training_admitted": False}
