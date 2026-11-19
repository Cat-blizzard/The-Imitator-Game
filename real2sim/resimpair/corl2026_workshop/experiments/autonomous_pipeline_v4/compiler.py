"""Task-agnostic measured-state compiler for the declared pick/place primitives."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .core import WORKSPACE, bind, checked, implementation, read, require, write, verify_implementation
from .library import task_semantic_key
from .prompts import ACTOR
from .source import select_sources


def first(mask, reason, start=0):
    indices = np.flatnonzero(mask & (np.arange(len(mask)) >= start))
    require(len(indices) > 0, reason)
    return int(indices[0])


def measured_events(bundle, arrays):
    """Only numeric state/contact evidence defines milestones; stage labels are unused."""
    task, scene = bundle["task_intent"], bundle["scene"]
    states, grasp, impulses = arrays["object_states"], arrays["grasp"], arrays["support_impulses"]
    ids, support_ids = bundle["object_ids"], bundle["support_ids"]
    require(set(ids) == {x["id"] for x in scene["objects"]}, "Exported object identity set differs from scene")
    require(len(ids) == len(set(ids)) and len(support_ids) == len(set(support_ids)), "Duplicate physical identities")
    n = len(states)
    require(20 <= n <= 4001, "Unsupported native timeline length")
    require(states.shape == (n, len(ids), 13) and grasp.shape == (n, 2, len(ids)) and
            impulses.shape == (n, len(ids), len(support_ids)), "Measurement dimensions differ from declared identities")
    for key in ["states", "tcp_world", "object_states", "support_impulses", "timestamps_seconds", "native_commands",
                "robot_object_force_N", "cross_robot_force_N"]:
        require(np.isfinite(arrays[key]).all(), "Nonfinite measured data: " + key)
    require(arrays["native_commands"].shape == (n - 1, 16), "Actions and pre/post-step frames do not align")
    require(np.all(np.diff(arrays["timestamps_seconds"]) > 0), "Nonmonotone measured time")
    require(grasp.dtype == np.bool_, "Grasp evidence must be measured booleans")
    require(not grasp[0].any(), "Initial human policy requires empty initial robot grippers")
    specs = {o["id"]: o for o in scene["objects"]}
    events, edges, outcomes = [], [], []
    op_events = {}
    selected = {o["object"] for o in task["operations"]}
    require(len(selected) == len(task["operations"]), "This primitive release does not yet support repeated transfers of the same object")
    for op in task["operations"]:
        require(op["primitive"] == "pick_place", "Unsupported measured primitive")
        oid, arm, sid = op["object"], op["arm"], op["target"]["support"]
        require(oid in ids and sid in support_ids and arm in (0, 1), "Invalid operation binding")
        i, s = ids.index(oid), support_ids.index(sid)
        held = grasp[:, arm, i]
        g = first(held, "no_measured_grasp:" + oid)
        lifted = states[:, i, 2] > states[0, i, 2] + .045
        lift = first(held & lifted, "no_measured_lift:" + oid, g)
        target = np.asarray(bundle["target_positions_world"][oid])
        distance = np.linalg.norm(states[:, i, :3] - target, axis=1)
        arrive = first(distance < .015, "target_never_reached:" + oid, lift + 1)
        release = first(~grasp[:, :, i].any(axis=1), "no_measured_release:" + oid, arrive)
        contact = impulses[:, i, s] > 1e-6
        support = first(contact & ~grasp[:, :, i].any(axis=1) & (distance < .015),
                        "no_released_target_support:" + oid, release)
        tcp_distance = np.linalg.norm(arrays["tcp_world"][:, :, :3, 3] - states[:, None, i, :3], axis=2)
        withdraw = first((tcp_distance > .075).all(axis=1) & ~grasp[:, :, i].any(axis=1),
                         "no_withdrawal_clearance:" + oid, release)
        marks = {"grasp": g, "lift": lift, "arrive": arrive, "release": release, "supported": support, "withdraw": withdraw}
        for phase, frame in marks.items():
            events.append({"id": op["id"] + "." + phase, "operation": op["id"], "phase": phase,
                           "object": oid, "support": sid, "arm_index": arm,
                           "frame": frame, "seconds": float(arrays["timestamps_seconds"][frame]),
                           "evidence_kind": "measured_state_and_contact",
                           "measurement_fields": ["grasp", "object_states", "support_impulses", "tcp_world"]})
        for before, after in [("grasp", "lift"), ("lift", "arrive"), ("arrive", "release"),
                              ("release", "supported"), ("release", "withdraw")]:
            edges.append([op["id"] + "." + before, op["id"] + "." + after])
        op_events[op["id"]] = marks
        tail = states[-20:, i]
        upright = Rotation.from_quat(tail[:, [4, 5, 6, 3]]).as_matrix()[:, 2, 2] > .95
        checks = {"goal_position": bool((distance[-20:] < .015).all()),
                  "stable_position": bool(np.linalg.norm(np.ptp(tail[:, :3], axis=0)) < .008),
                  "stable_velocity": bool((np.linalg.norm(tail[:, 7:10], axis=1) < .025).all()),
                  "stable_rotation": bool((np.linalg.norm(tail[:, 10:13], axis=1) < .4).all() and upright.all()),
                  "released": bool(not grasp[-20:, :, i].any()), "measured_support": bool(contact[-20:].all()),
                  "grasp_before_lift": g <= lift, "lift_before_arrival": lift < arrive,
                  "held_during_transport": bool(held[lift:arrive].all()),
                  "end_clearance": bool((tcp_distance[-10:] > .075).all())}
        outcomes.append({"object": oid, "checks": checks, "passed": all(checks.values()),
                         "final_position_error_m": float(distance[-1]),
                         "final_support_impulse_sum": float(impulses[-20:, i, s].sum())})
    for before, after in task["required_operation_precedence"]:
        require(before in op_events and after in op_events, "Unknown necessary operation order")
        # Support must exist before the next block can be placed on it. The
        # next grasp may overlap with settling without changing task semantics.
        edges.append([before + ".supported", after + ".arrive"])
    event_by_id = {e["id"]: e for e in events}
    require(all(event_by_id[a]["frame"] <= event_by_id[b]["frame"] for a, b in edges), "Required task precedence violated")
    non_targets = []
    for i, oid in enumerate(ids):
        if oid not in selected:
            displacement = np.linalg.norm(states[:, i, :3] - states[0, i, :3], axis=1)
            rotations = Rotation.from_quat(states[:, i, [4, 5, 6, 3]])
            angular_change = (rotations[0].inv() * rotations).magnitude()
            contact_force = arrays["robot_object_force_N"][:, :, i]
            non_targets.append({"object": oid, "max_displacement_m": float(displacement.max()),
                                "max_orientation_change_rad": float(angular_change.max()),
                                "max_robot_contact_force_N": float(contact_force.max()),
                                "passed": bool(displacement.max() < .008 and angular_change.max() < .05 and contact_force.max() < .5)})
    cross = float(arrays["cross_robot_force_N"].max())
    passed = all(x["passed"] for x in outcomes + non_targets) and cross < .5
    return {"events": events, "required_event_precedence": edges,
            "qualification": {"passed": passed, "targets": outcomes, "non_targets": non_targets,
                              "max_cross_robot_contact_force_N": cross,
                              "scope": "native sampled dynamics and measured contacts; not an all-path collision proof"}}


def project(point, camera):
    params = camera["parameters"]
    extrinsic = np.asarray(params["extrinsic_cv"])
    xyz = extrinsic[:3, :3] @ np.asarray(point) + extrinsic[:3, 3]
    require(xyz[2] > 0, "Object behind selected fixed camera")
    pixel = np.asarray(params["intrinsic_cv"]) @ xyz
    config = camera["config"]
    return [round(float(pixel[0] / pixel[2] / config["width"]), 4),
            round(float(pixel[1] / pixel[2] / config["height"]), 4)]


def verify_native_kinematics(arrays):
    import importlib.util
    path = WORKSPACE / "PolicyGen/examples/baselines/hypericlzq/eef.py"
    spec = importlib.util.spec_from_file_location("resimpair_panda_fk_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    measured = arrays["states"]
    require(measured.shape == (len(arrays["timestamps_seconds"]), 18), "Measured joint dimensions mismatch")
    qpos = np.stack([measured[:, :7], measured[:, 9:16]], axis=1)
    poses = arrays["robot_roots_world"][None] @ module.panda_fk(qpos)
    error = float(np.max(np.abs(poses - arrays["tcp_world"])))
    require(error < 3e-6, "Native joint/TCP frame or identity mismatch")
    commands = arrays["native_commands"]
    qcommands = np.stack([commands[:, :7], commands[:, 8:15]], axis=1)
    limits = np.array([row[2] for row in module.panda_chain() if row[1]])
    require(bool((qcommands >= limits[:, 0] - 1e-4).all() and (qcommands <= limits[:, 1] + 1e-4).all()),
            "Native commanded joint limit exceeded")
    require(bool((np.abs(commands[:, [7, 15]]) <= 1).all()), "Invalid native gripper command")
    return {"passed": True, "all_measured_frames": len(measured), "all_command_frames": len(commands),
            "maximum_fk_matrix_error": error, "fk_implementation": bind(path), "urdf": bind(module.URDF)}


def compile_bundle(bundle_path, source_mapping):
    from .interface import validate_bundle
    bundle = read(bundle_path)
    if bundle.get("schema") == "resimpair_execution_bundle_v5":
        from .generic_compiler import compile_bundle as generic
        return generic(bundle_path, source_mapping)
    validate_bundle(bundle)
    require(bundle["schema"] == "resimpair_execution_bundle_v4" and bundle["native_execution"] is True,
            "A native planning execution bundle is required")
    require(bundle["planning_failure"] is None, "Planning execution failed")
    require(bundle["manual_generation_semantics_supplied"] is False and bundle["manual_review_observations_supplied"] is False,
            "Manual downstream annotations violate this release's interface")
    checked(bundle["case"]["scene"])
    checked(bundle["case"]["task"])
    require(read(bundle["case"]["scene"]["path"]) == bundle["scene"] and read(bundle["case"]["task"]["path"]) == bundle["task_intent"],
            "Planner changed the declared scene or task")
    verify_implementation(bundle["code"])
    with np.load(checked(bundle["measurements"]), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    validate_bundle(bundle, arrays)
    native_kinematics = verify_native_kinematics(arrays)
    measured = measured_events(bundle, arrays)
    measured["qualification"]["native_kinematics"] = native_kinematics
    require(measured["qualification"]["passed"], "native_physical_qualification_failed")
    require(set(bundle["cameras"]) == {"robot_side", "opposite_high"}, "Both fixed synchronized cameras required")
    for camera in bundle["cameras"].values():
        for name in ["video", "initial_image", "final_image"]:
            checked(camera[name])
        require(camera["synchronous_with_measurements"] and camera["frame_count"] == len(arrays["states"]),
                "Camera and measurement timelines differ")
    scene, task = bundle["scene"], bundle["task_intent"]
    camera = bundle["cameras"]["opposite_high"]
    inventory = []
    for i, obj in enumerate(scene["objects"]):
        index = bundle["object_ids"].index(obj["id"])
        inventory.append({"id": obj["id"], "description": obj["semantic_description"], "shape": obj["shape"],
                          "size_m": [2 * h for h in obj["half_size"]], "role": "target" if obj["id"] in task["object_ids"] else "unchanged",
                          "initial_image_xy_normalized": project(arrays["object_states"][0, index, :3], camera),
                          "final_image_xy_normalized": project(arrays["object_states"][-1, index, :3], camera)})
    for support in scene["supports"]:
        inventory.append({"id": support["id"], "description": support["semantic_description"], "shape": support["shape"],
                          "role": "fixed_support", "size_m": [2 * h for h in support["half_size"]],
                          "initial_image_xy_normalized": project(np.asarray(support["position"]) + [0, 0, support["top_z"]], camera)})
    inventory.append({"id": "table", "description": "the tabletop", "role": "fixed_support"})
    duration = 8 if len(task["operations"]) <= 2 else 12
    native_duration = float(arrays["timestamps_seconds"][-1])
    events_for_generation = [{k: e[k] for k in ["id", "phase", "object", "support"]} |
                             {"effector": f"arm{e['arm_index']}",
                              "suggested_seconds": round(e["seconds"] / native_duration * (duration - 1), 2)}
                             for e in measured["events"]]
    goals = [{"object": op["object"], "relation": "supported_by", "support": op["target"]["support"],
              "target_image_xy_normalized": project(bundle["target_positions_world"][op["object"]], camera),
              "released": True} for op in task["operations"]]
    generation = {"task_family": task["family"], "goal": task["goal"], "duration_seconds": duration,
                  "inventory": inventory, "final_relations": goals,
                  "required_events": events_for_generation, "required_event_precedence": measured["required_event_precedence"],
                  "independent_transfers_may_reorder": task["family"] != "stack", "actor": ACTOR,
                  "initial": "all loose target blocks on table, both hands empty, supports remain where shown",
                  "non_target_policy": task["non_target_policy"], "equivalence_groups": task["equivalence_groups"],
                  "frame_coordinates": "normalized x from image left, y from top; descriptive grounding, no exact human pose claim"}
    initial_inventory = [{k: v for k, v in row.items() if k not in {"final_image_xy_normalized", "role"}} |
                         {"role": "movable" if row["id"] in {o["id"] for o in scene["objects"] if o["kind"] == "movable"}
                          else "fixed_scene_object"} for row in inventory]
    from .hand_correspondence import attach, compile_correspondence
    from .planning_protocols import pick_place
    protocol = pick_place(scene, task, bundle["object_ids"], bundle["support_ids"], bundle["target_positions_world"])
    correspondence = compile_correspondence(protocol, arrays, camera)
    initial = {"inventory": initial_inventory, "actor": ACTOR, "initial_hands": protocol["initial_hands"]}
    generation["initial_hands"] = protocol["initial_hands"]
    generation["terminal_hands"] = protocol["terminal_hands"]
    attach(generation, initial, correspondence)
    return {"schema": "resimpair_compiled_task_v4", "execution": bind(bundle_path), "source_mapping": source_mapping,
            "task_id": task["id"], "semantic_key": task_semantic_key(task), "measured": measured,
            "generation_information": generation, "initial_scene_information": initial,
            "hand_correspondence_evidence": correspondence,
            "selected_camera": "opposite_high", "initial_image": camera["initial_image"],
            "final_image": camera["final_image"], "simulation_video": camera["video"],
            "other_simulation_video": bundle["cameras"]["robot_side"]["video"],
            "condition_consumption": {"initial_image": "image_model_primary_input", "initial_scene_information": "image_model_text",
                                      "generation_information": "video_model_text", "source_mapping": "skill_compatibility_and_lineage",
                                      "native_joint_paths": "audit_only_not_claimed_human_condition"},
            "manual_downstream_annotation_count": 0, "code": implementation(), "training_admitted": False}


def resolve_attempt(root, case_id, attempt=None):
    handoff = Path(root) / "planner_handoffs" / (case_id + ".json")
    if attempt is None and handoff.exists():
        record = read(handoff)
        checked(record["execution_bundle"])
        require(record["case"] == case_id and record["manual_downstream_annotations"] is False, "Invalid planning handoff")
        return record["attempt"]
    return attempt or "attempt_01"


def compile_execution(root, case_id, attempt=None):
    root = Path(root)
    attempt = resolve_attempt(root, case_id, attempt)
    bundle_path = resolve_bundle(root, case_id, attempt)
    bundle = read(bundle_path)
    mapping = select_sources(root, bundle["task_intent"]["required_skills"])
    compiled = compile_bundle(bundle_path, mapping)
    if any(o["shape"] == "convex_mesh" for o in bundle["scene"]["objects"]):
        from .asset_qualification import qualify
        evidence = qualify(bundle)
        require(evidence["passed"], "active_asset_native_geometry_qualification_failed")
        compiled["active_asset_qualification"] = evidence
    output = compiled_path(root, case_id, attempt).parent
    write(output / "contract.json", compiled)
    from .media import packet
    media = {}
    with np.load(checked(bundle["measurements"]), allow_pickle=False) as archive:
        timestamps = archive["timestamps_seconds"].copy()
    for name, video in [("opposite_high", compiled["simulation_video"]), ("robot_side", compiled["other_simulation_video"])]:
        info = packet(checked(video), output / (name + "_frames"), frame_budget=9)
        require(info["decoded_frames"] == bundle["cameras"][name]["frame_count"], "Actual video/measurement length differs")
        require(np.allclose(info["all_native_pts_seconds"], timestamps, atol=1e-5, rtol=0), "Actual video/measurement PTS differ")
        require(abs(info["fps"] - bundle["cameras"][name]["fps"]) < 1e-5, "Actual camera FPS differs")
        require([info["width"], info["height"]] == [bundle["cameras"][name]["config"]["width"],
                                                  bundle["cameras"][name]["config"]["height"]], "Actual camera dimensions differ")
        media[name] = bind(output / (name + "_frames") / "packet.json")
    write(output / "media_validation.json", {"contract": bind(output / "contract.json"), "cameras": media,
                                             "both_fully_decoded": True})
    return {"status": "compiled", "contract": bind(output / "contract.json"), "events": len(compiled["measured"]["events"]),
            "physical_qualification": compiled["measured"]["qualification"]["passed"]}


def resolve_bundle(root, case_id, attempt):
    handoff = Path(root) / "planner_handoffs" / (case_id + ".json")
    if handoff.exists():
        record = read(handoff)
        require(record["case"] == case_id and record["attempt"] == attempt, "planning_handoff_attempt_mismatch")
        return checked(record["execution_bundle"])
    return Path(root) / "executions" / case_id / attempt / "execution_bundle.json"


def completed_contract(path):
    """A written contract alone is not a complete, media-qualified compilation."""
    path = Path(path)
    marker = path.parent / "media_validation.json"
    require(marker.exists(), "incomplete_contract_media_validation")
    contract, validation = read(path), read(marker)
    require(validation["contract"] == bind(path) and validation["both_fully_decoded"] is True,
            "media_validation_contract_mismatch")
    require(set(validation["cameras"]) == {"robot_side", "opposite_high"}, "media_validation_requires_both_cameras")
    bundle = read(checked(contract["execution"]))
    with np.load(checked(bundle["measurements"]), allow_pickle=False) as archive:
        timestamps = archive["timestamps_seconds"].copy()
    for name, key in [("opposite_high", "simulation_video"), ("robot_side", "other_simulation_video")]:
        camera = bundle["cameras"][name]
        info = read(checked(validation["cameras"][name]))
        checked(contract[key])
        require(info["video"] == contract[key] == camera["video"], "media_validation_video_mismatch")
        require(info["decode_complete"] is True and info["decoded_frames"] == camera["frame_count"] == len(timestamps),
                "incomplete_decoded_camera")
        require(len(info["all_native_pts_seconds"]) == len(timestamps) and
                np.allclose(info["all_native_pts_seconds"], timestamps, atol=1e-5, rtol=0), "media_validation_clock_mismatch")
        require(abs(info["fps"] - camera["fps"]) < 1e-5 and
                [info["width"], info["height"]] == [camera["config"]["width"], camera["config"]["height"]],
                "media_validation_camera_mismatch")
        for frame in info["frames"]:
            checked(frame["image"])
    return contract


def compiled_path(root, case_id, attempt):
    from .pipeline import release_id
    return Path(root) / "compiled_by_release" / release_id() / case_id / attempt / "contract.json"
