"""Opt-in, evidence-bound stage compiler; no new acceptance rule or model claim.

The executable integration is the existing unified apple request constructor.
Historical book/cabinet imports are non-dispatchable generalization probes.
Structured evidence remains a sidecar. Only bounded, stage-specific directions
go into model text; simulator keyframes are NOT native video pose controls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .batch_pipeline import binding, checked_binding
from .common import read_json, write_json

VERSION = "candidate_v1"
SCHEMA = "r2h_stage_condition_compiler_candidate_v1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def prompt_stats(text):
    return {"characters": len(text), "english_whitespace_words": len(text.split()),
            "utf8_bytes": len(text.encode()), "provider_token_count": None,
            "video_target_words": [150, 250], "target_is_not_provider_limit": True}


def actor_direction(plan, inactive_policy="retain_plan"):
    require(inactive_policy in {"retain_plan", "may_remain_out_of_frame"},
            "Unknown inactive-hand visibility policy")
    actor = plan["actor"]
    require(actor["id"] == "operator_1" and set(actor["hands"]) == {"left", "right"},
            "One declared operator with anatomical left/right ownership is required")
    for side, hand in actor["hands"].items():
        require(hand["owner_id"] == actor["id"] and hand["anatomical_side"] == side,
                "All anatomical hands must belong to the same operator")
    active = set()
    for event in plan["event_assignments"]:
        require(event["actor_id"] == actor["id"] and event["hands"]
                and set(event["hands"]) <= {"left", "right"}, "Invalid event hand ownership")
        active.update(event["hands"])
    require(active, "An explicit working hand assignment is required")
    body = actor["body_position"]
    direction = (f"Exactly one operator is {body['image_region']}, facing {body['facing']}. "
                 "All visible arms originate from this same person; do not turn the two robot bases "
                 "into separate people. Human arm entry may change while the object layout stays fixed. ")
    if len(active) == 1:
        direction += f"The {next(iter(active))} hand is the working hand. "
        if inactive_policy == "may_remain_out_of_frame":
            direction += "The other hand may remain outside the image and must not perform a second action. "
        else:
            direction += "The other hand stays inactive as specified by the body plan. "
    else:
        direction += "Both working hands belong to this one operator; follow their assigned event roles. "
    return direction.strip()


def _compile(*, initial, terminal, sequence, event_ids, invariants, actor_plan, initial_image,
             terminal_image, event_evidence, source_bindings, state_delta=None,
             hidden=None, inactive_policy="retain_plan", geometry_summary=None, scope):
    require(all(isinstance(x, str) and x.strip() for x in [initial, terminal, *sequence, *invariants]),
            "Stage descriptions must be explicit nonempty strings")
    require(1 <= len(sequence) <= 8, "Split tasks with more than eight necessary events")
    require(len(sequence) == len(event_ids) == len(set(event_ids))
            and [x["event_id"] for x in actor_plan["event_assignments"]] == event_ids,
            "Every necessary event ID and its order must match the operator/hand assignments")
    for item in source_bindings.values():
        checked_binding(item)
    for item in (initial_image, terminal_image):
        checked_binding(item)
    require(initial_image["sha256"] != terminal_image["sha256"],
            "Initial and changed terminal state cannot share identical image bytes")
    direction = actor_direction(actor_plan, inactive_policy)
    if geometry_summary:
        direction += " " + geometry_summary["prompt_fragment"]
    assigned_sequence = ["with " + ("both hands" if len(x["hands"]) == 2 else
                         "the " + x["hands"][0] + " hand") + ", " + step
                         for step, x in zip(sequence, actor_plan["event_assignments"])]
    initial_text = ("Edit only the supplied initial simulation frame into a realistic human reference image. "
                    + direction + " Both hands are initially inactive and clear of task objects. "
                    "Initial state: " + initial + " Preserve every task object, its size and current location, "
                    "camera and background. Remove robot parts. Concealed contents remain concealed. "
                    "Do not enact an action or show a later state. No captions or montage.")
    terminal_text = ("Edit the first input, the simulation terminal frame, into a realistic human terminal image. "
                     "Supporting image 1 is the approved human initial image: use it for appearance and "
                     "operator identity, never to copy its object state. " + direction +
                     " Required change from initial to terminal: " + (state_delta or terminal) +
                     " Terminal state: " + terminal + " Hands have released and withdrawn. Preserve: " +
                     "; ".join(invariants) + ". Keep the selected camera, object shapes and relative sizes. "
                     "No text, montage, duplicate objects or extra people.")
    video_text = ("Create one continuous fixed-camera human manipulation video from the supplied initial image. "
                  + direction + " Begin with: " + initial + " Perform in order: " +
                  "; then ".join(assigned_sequence) + ". End with: " + terminal + " Preserve: " +
                  "; ".join(invariants) + ". Show approach, continuous manipulation contact, release and withdrawal. "
                  "Hold the outcome. Preserve identity and scale: no cuts, duplication, "
                  "teleportation, extra people or final-frame reset. Simulator event times establish "
                  "order, not target-video timing.")
    refs = []
    if hidden:
        require(hidden["identity_status"] == "source_bound_not_generated_verified",
                "Hidden identity cannot be promoted into generated-image acceptance")
        if hidden.get("reference"):
            checked_binding(hidden["reference"])
            refs.append(hidden["reference"])
        description = hidden["description"]
        terminal_text += " Hidden-content reference: " + description
        video_text += " When revealed, retain " + description
    require(len(video_text.split()) <= 250, "Video text too long; compact the task instead of truncating events")
    base = {"schema": SCHEMA, "scope": scope, "source_bindings": source_bindings,
            "operator_plan": deepcopy(actor_plan), "inactive_hand_visibility_policy": inactive_policy,
            "existing_single_actor_acceptance_unchanged": True,
            "single_hand_plan_may_still_need_review_under_existing_rule": inactive_policy != "retain_plan",
            "measured_event_sidecar": event_evidence, "hidden_identity_sidecar": hidden,
            "static_actor_geometry": geometry_summary,
            "model_receives_sim_keyframes_as_pose_control": False,
            "quality_release_approved": False, "training_admitted": False, "visual_improvement_verified": False}
    base["stages"] = {
        "initial": {"prompt": initial_text, "target": initial_image, "reference_images": [],
                    "state_fields_consumed": ["initial", "operator_initial_placement"],
                    "future_task_state_or_manipulation_instructions_consumed": False},
        "terminal": {"prompt": terminal_text, "target": terminal_image,
                     "reference_images_after_approved_initial": refs,
                     "requires_approved_generated_initial": True, "state_delta": state_delta or terminal},
        "video": {"prompt": video_text, "required_events": sequence,
                  "event_ids": event_ids, "hand_assigned_sequence": assigned_sequence,
                  "requires_independent_actor_and_static_state_gates": True,
                  "source_simulator_event_timing_is_not_video_frame_timing": True}}
    for value in base["stages"].values():
        value["static_actor_geometry"] = geometry_summary
        value["prompt_stats"] = prompt_stats(value["prompt"])
    return base


def compile_unified(data, *, inactive_policy="retain_plan"):
    """Input is freshly revalidated by unified.validate_package/verify_sources."""
    p = data["packet"]
    require(p["status"] == "prepared_for_condition_image_review", "Unqualified source conditions")
    order = p["camera_role_projections"][p["selected_camera"]]["relative_order"]
    require(order["status"] == "projected_geometry", "Unknown camera-relative role mapping")
    source = order["source_relative_to_destination"].replace("screen_", "image-")
    target = order["destination_relative_to_source"].replace("screen_", "image-")
    initial = next(c for c in p["conditions"] if "initial" in c["labels"])
    terminal = next(c for c in p["conditions"] if "terminal" in c["labels"])
    events = p["measured_events"]
    require(events["initial"] == initial["frame"]["frame_index_0based"] == 0
            and events["terminal"] == terminal["frame"]["frame_index_0based"],
            "Stage frame labels disagree with measured events")
    require(events["grasp_established"] <= events["lift_clear"]
            < events["released"] <= events["terminal"], "Invalid measured manipulation order")
    packet_bound = data["package"]["inputs"]["conditions"]
    geometry_summary = None
    declared = data["plan"]["actor"].get("synthetic_geometry")
    if declared:
        from .actor_geometry_candidate import validate_geometry
        checked_binding(declared["geometry_preflight_report"])
        report = validate_geometry(declared["geometry_preflight_report"]["path"])
        require(report["context"] == data["context"], "Actor rig belongs to another source/camera context")
        checked_binding(report["bindings"]["source_package"])
        geometry_source = read_json(report["bindings"]["source_package"]["path"])
        checked_binding(geometry_source["actor_plan"])
        geometry_plan = read_json(geometry_source["actor_plan"]["path"])
        actor_without_rig = {k: v for k, v in data["plan"]["actor"].items() if k != "synthetic_geometry"}
        require(actor_without_rig == geometry_plan["actor"]
                and data["plan"]["event_assignments"] == geometry_plan["event_assignments"],
                "Actor position or event hand assignment differs from the checked geometry plan")
        checked_binding(report["bindings"]["rig"])
        rig = read_json(report["bindings"]["rig"]["path"])
        require(declared["rig"] == rig, "Static actor rig differs from the checked geometry report")
        shoulders = report["projection"]["shoulders"]
        coordinates = []
        for hand in ["left", "right"]:
            uv = shoulders[hand]["normalized_uv"]
            coordinates.append(hand + (f" ({uv[0]:.3f},{uv[1]:.3f})" if uv else " unknown"))
        geometry_summary = {"report": declared["geometry_preflight_report"],
            "rig": report["bindings"]["rig"], "clears_geometry_preflight": report["clears_geometry_preflight"],
            "waist_world_m": rig["torso"]["waist_world_m"], "shoulders_world_m": report["shoulders"],
            "shoulders_normalized_image_xy": {s: shoulders[s]["normalized_uv"] for s in shoulders},
            "torso_lean_degrees": rig["torso"]["lean_degrees"],
            "prompt_fragment": f"Lean torso {rig['torso']['lean_degrees']} degrees forward. "
                + "Anatomical shoulders, normalized from image top-left: " + ", ".join(coordinates) + ".",
            "native_pose_control": False, "generated_structure_verified": False}
    return _compile(initial=f"The same apple is inside the {source} source bowl; the {target} destination bowl is empty.",
        terminal=f"The apple is released inside the {target} destination bowl; the {source} source bowl no longer contains it.",
        sequence=["grasp the apple in the source bowl", "lift it above the source rim",
                  "carry it over the destination bowl", "lower and release it inside", "withdraw the hand"],
        event_ids=["grasp_apple", "lift_from_source", "transport_to_target", "release_in_target", "withdraw_hand"],
        invariants=["both bowls stay in place", "all nonselected objects stay unchanged"],
        actor_plan=data["plan"], initial_image=initial["frame"], terminal_image=terminal["frame"],
        event_evidence={"binding": packet_bound, "measured_events": events,
                        "conditions": [{"labels": c["labels"], "frame": c["frame"],
                                        "grasped_by_arm_indices": c["simulator_grasped_by_arm_indices"]}
                                       for c in p["conditions"]]},
        source_bindings={"conditions": packet_bound, "task_contract": data["package"]["task_contract"],
                         "operator_plan": data["package"]["actor_plan"]},
        state_delta=p["state_delta"]["edit_instruction"], inactive_policy=inactive_policy,
        geometry_summary=geometry_summary,
        scope="qualified_strict_apple_source_only")


def prepare_terminal(package_path, image_path, actor_review_path, state_review_path, config, output):
    """Actual Gemini request construction, blocked until existing initial gates pass."""
    from . import unified_anchor_candidate as unified
    data = unified.validate_package(package_path)
    compiled = compile_unified(data, inactive_policy=config.get("inactive_hand_visibility", "retain_plan"))
    gate = unified._video_gate(data, image_path, actor_review_path, state_review_path)
    stage = compiled["stages"]["terminal"]
    request = {"schema": "r2h_compiled_terminal_request_candidate_v1", "package": binding(package_path),
               "compiled": compiled, "model": config["anchor_model"], "prompt": stage["prompt"],
               "target": stage["target"], "reference_images": [binding(image_path)],
               "initial_actor_review": binding(actor_review_path) if actor_review_path else None,
               "initial_state_review": binding(state_review_path) if state_review_path else None,
               "gate": gate, "provider_called": False, "new_paid_calls": 0,
               "status": "ready_for_terminal_image" if gate["can_generate_video"] else "blocked_initial_reviews"}
    output = Path(output)
    require(not output.exists(), "Terminal request output is immutable")
    write_json(output, request)
    return request


def execute_terminal(request_path, config, budget_path, credentials_path=None):
    """Use existing cumulative budget/unknown-submit protection; no auto retries."""
    from . import unified_anchor_candidate as unified
    from .common import load_env_file
    from .providers import GeminiClient
    request = read_json(request_path)
    checked_binding(request["package"])
    data = unified.validate_package(request["package"]["path"])
    compiled = compile_unified(data, inactive_policy=config.get("inactive_hand_visibility", "retain_plan"))
    require(request["compiled"] == compiled and request["model"] == config["anchor_model"]
            and request["target"] == compiled["stages"]["terminal"]["target"]
            and request["prompt"] == compiled["stages"]["terminal"]["prompt"],
            "Terminal request differs from freshly compiled conditions")
    require(len(request["reference_images"]) == 1, "Exactly one approved initial reference required")
    for value in request["reference_images"]:
        checked_binding(value)
    reviews = []
    for key in ["initial_actor_review", "initial_state_review"]:
        value = request[key]
        if value:
            checked_binding(value)
        reviews.append(value["path"] if value else None)
    gate = unified._video_gate(data, request["reference_images"][0]["path"], *reviews)
    require(gate["can_generate_video"], "Initial image lacks independent actor/state approval")
    timeout = read_json(budget_path)["timeout_seconds"]
    def operation(transport):
        load_env_file(credentials_path)
        return GeminiClient(dict(config["gemini"], timeout=timeout), transport=transport).generate_image_with_references(
            Path(request["target"]["path"]), [Path(x["path"]) for x in request["reference_images"]],
            request["prompt"], model=request["model"])
    return unified._campaign_budgeted_call(budget_path, "image", {
        "method": SCHEMA, "stage": "terminal", "request": binding(request_path)}, operation)


def compile_legacy_probe(manifest_path, case_id, *, hidden_path=None):
    """Frozen non-dispatchable probes; no historical image is accepted as single actor."""
    import numpy as np
    manifest = read_json(manifest_path)
    matches = [x for x in manifest["cases"] if x["id"] == case_id]
    require(len(matches) == 1, "Unknown or duplicate historical case")
    case = matches[0]
    keys = ["source_record", "human_source", "robot_video", "robot_trajectory", "sim_initial", "sim_terminal"]
    binds = {k: case[k] for k in keys}
    for value in binds.values():
        checked_binding(value)
    record = read_json(case["source_record"]["path"])
    require(record["source_window"]["video"] == case["human_source"]
            and record["source_window"]["window_frames"] == case["human_window_frames"],
            "Historical case source window mismatch")
    arrays = dict(np.load(case["robot_trajectory"]["path"], allow_pickle=False))
    held = arrays["grasp"]
    require(held.dtype == np.bool_ and held.ndim == 3, "Expected measured time/arm/object grasp array")
    measured = {"source": case["robot_trajectory"], "scope": "raw grasp transitions; object-role mapping not requalified",
                "frames": len(held), "transitions": [], "strict_apple_qualification_inherited": False}
    for arm in range(held.shape[1]):
        for obj in range(held.shape[2]):
            track = held[:, arm, obj]
            for frame in np.flatnonzero(track[1:] != track[:-1]) + 1:
                measured["transitions"].append({"frame": int(frame), "arm_index": arm,
                    "object_index": obj, "grasped": bool(track[frame])})
    hidden = None
    if hidden_path:
        h = read_json(hidden_path)
        require(h["bindings"]["arrays"] == case["robot_trajectory"]
                and h["bindings"]["video"] == case["robot_video"], "Hidden object packet belongs to another execution")
        for value in h["bindings"].values():
            checked_binding(value)
        binds["hidden_conditions"] = binding(hidden_path)
        reference = next((x["frame"] for x in h["conditions"] if x["role"] == "reviewed_revealed_reference"), None)
        hidden = {"identity_status": "source_bound_not_generated_verified", "asset_uid": h["object"]["asset_uid"],
                  "reference": reference, "dimensions_m": h["object"]["collision_local_extent_m"],
                  "description": "the same small cuboid in its original shelf position and proportion; never replace it with a large flat grid panel.",
                  "RGB_reference_status": "reviewed_source_only" if reference else "unknown",
                  "physical_slot": h["slot"]["id"]}
    task = case["task"]
    plan = {"actor": {"id": "operator_1", "body_position": {
        "image_region": "behind the table, at one common body position", "facing": "the task objects"},
        "hands": {s: {"owner_id": "operator_1", "anatomical_side": s} for s in ["left", "right"]}},
        "event_assignments": [{"event_id": f"event_{i}", "actor_id": "operator_1", "hands": ["right"]}
                              for i in range(len(task["necessary_steps"]))],
        "status": "proposed_not_geometry_or_image_qualified"}
    compiled = _compile(initial=task["initial"], terminal=task["goal"], sequence=task["necessary_steps"],
        event_ids=[x["event_id"] for x in plan["event_assignments"]],
        invariants=task["invariants"], actor_plan=plan, initial_image=case["sim_initial"],
        terminal_image=case["sim_terminal"], event_evidence=measured,
        source_bindings={"historical_manifest": binding(manifest_path), **binds}, hidden=hidden,
        inactive_policy="may_remain_out_of_frame", scope="historical_source_probe_not_dispatchable")
    compiled.update(dispatch_allowed=False, blocking_reasons=["no_new_bound_and_qualified_operator_plan",
        "no_new_independent_generated_image_reviews", "no_new_campaign_budget"],
        physical_semantics="historical scope only; raw transitions are not a new independent physical acceptance")
    return compiled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package"); parser.add_argument("--manifest"); parser.add_argument("--case")
    parser.add_argument("--hidden"); parser.add_argument("--output", required=True)
    parser.add_argument("--inactive-hand", choices=["retain_plan", "may_remain_out_of_frame"], default="retain_plan")
    args = parser.parse_args()
    require(bool(args.package) != bool(args.manifest), "Choose a unified package or historical probe manifest")
    if args.package:
        from .unified_anchor_candidate import validate_package
        result = compile_unified(validate_package(args.package), inactive_policy=args.inactive_hand)
    else:
        result = compile_legacy_probe(args.manifest, args.case, hidden_path=args.hidden)
    require(not Path(args.output).exists(), "Use a new output; evidence is immutable")
    write_json(args.output, result)
    print(json.dumps({"schema": SCHEMA, "scope": result["scope"], "new_paid_calls": 0}))
