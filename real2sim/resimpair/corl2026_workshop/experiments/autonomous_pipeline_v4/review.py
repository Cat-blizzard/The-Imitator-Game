"""Automatic visual observation with strict citations and explicit unknowns."""
from __future__ import annotations

from pathlib import Path

from .core import bind, checked, implementation, read, require, write
from .models import reviewed_json
from .prompts import review_prompt
from .hand_correspondence import review_requirements
from . import hand_review
from .timing import traced

ANCHOR_CHECKS = {
    "one_operator": "Exactly one connected visible upper body behind the robot side of the table, including both shoulders.",
    "left_arm_connected": "The operator's left shoulder, elbow, wrist and hand form one plausible unbroken limb.",
    "right_arm_connected": "The operator's right shoulder, elbow, wrist and hand form one plausible unbroken limb.",
    "two_hands_empty": "Exactly two hands, both empty and clear of the initial task objects; no extra or floating hands.",
    "robot_removed": "Both robots and all robot/gripper fragments are absent.",
    "initial_inventory": "All listed objects/supports remain present with the specified initial identities, shape, color, scale and placement.",
    "camera_and_visibility": "The same elevated opposite-side view retains the task objects and supports without new severe occlusion.",
}

VIDEO_CHECKS = {
    "single_operator": "Only one operator; visible hands/arms belong to the same coherent torso; no extra limbs.",
    "initial_state": "Begins at the declared initial object relations and per-hand states, before the required new events.",
    "final_goal": "Every declared final relation and per-hand state holds in the ending samples; release only where required.",
    "object_identity": "The exact initial colored/shaped objects persist; no substitution, merging, resizing or disappearing geometry.",
    "non_targets_unchanged": "All objects identified as unchanged retain the declared identity, geometry, role and pose.",
    "natural_interaction": "Declared interactions are visibly plausible; no observed jump/cut or floating object; do not invent lifting or release requirements.",
    "camera_fixed": "A fixed camera throughout, with task events sufficiently visible to evaluate.",
}


def anchor_checks(expected):
    checks = dict(ANCHOR_CHECKS)
    if "initial_hands" in expected:
        del checks["two_hands_empty"]
        checks["declared_hand_states"] = "Exactly two connected hands match their individually declared initial states, including any held object."
    if "hand_correspondence" in expected:
        checks["body_frame_orientation"] = (
            "The one operator faces the declared workspace, with anatomical left/right shoulders in the "
            "initial image order specified by hand_correspondence; trace shoulders through elbows to wrists. "
            "Do not infer anatomy from wrist position or accept a mirrored body. Return unknown if not visible.")
    return checks


def video_checks(contract):
    checks = dict(VIDEO_CHECKS)
    info = contract["generation_information"]
    for event in info["required_events"]:
        if "action" in event:
            action, entities = event["action"], event["entities"]
        else:
            action, entities = event["phase"], [event["object"], event["support"]]
        checks["event:" + event["id"]] = (
            f"The declared {action} milestone for {', '.join(entities)} is visibly supported. "
            "Cite its earliest clear occurrence; do not copy simulator frame numbers or impose undeclared hand actions.")
    for goal in info.get("final_relations", []):
        if "id" in goal:
            checks["goal:" + goal["id"]] = (
                f"The final {goal['relation']} relation for {', '.join(goal['entities'])} agrees with its declared parameters "
                "in the ending candidate samples. Numeric physics thresholds are not measurable from RGB; report visible evidence and limits.")
    for constraint in info.get("persistent_constraints", []):
        checks["constraint:" + constraint["id"]] = (
            f"The {constraint['relation']} constraint for {', '.join(constraint['entities'])} holds across the visible "
            f"interval from {constraint['from']} through {constraint['until']}, with the declared effector. "
            "Check all supplied samples in the interval, including occlusions; endpoint agreement alone is insufficient. "
            "If the relevant interaction cannot be seen, return unknown.")
    if "terminal_hands" in info:
        checks["terminal_hand_states"] = "Each hand matches its declared terminal state; a support hand must keep holding if specified."
    if info.get("spatial_goal_relations"):
        checks["spatial_goal_relations"] = "Final object identities obey the explicit image-relative spatial relations; do not substitute color order or confuse it with event order."
    for rid, expected in review_requirements(info).items():
        checks["hand:" + rid] = (
            f"The {expected['action']} of {', '.join(expected['entities'])} is performed by the operator's "
            f"anatomical {expected['human_hand'].upper()} hand, originating at the {expected['initial_shoulder_image_side']} "
            "shoulder in the initial candidate frame. Follow shoulder, elbow and wrist through cross-body reaches; "
            "task success alone does not satisfy this requirement. Unknown ownership must remain unknown.")
    return checks


def schema(checks, census_frames=None, census_targets=(), hand_requirements=None):
    item = {"type": "object", "additionalProperties": False,
            "properties": {"value": {"type": "string", "enum": ["yes", "no", "unknown"]},
                           "frame_indices": {"type": "array", "uniqueItems": True, "items": {"type": "integer", "minimum": 0}},
                           "observation": {"type": "string", "minLength": 1}},
            "required": ["value", "frame_indices", "observation"]}
    result = {"type": "object", "additionalProperties": False,
            "properties": {"checks": {"type": "object", "additionalProperties": False,
                                       "properties": {key: item for key in checks}, "required": list(checks)},
                           "limitations": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}},
            "required": ["checks", "limitations"]}
    if census_frames is not None:
        count = {"type": "integer", "minimum": -1, "maximum": 32}
        frame = {"type": "object", "additionalProperties": False, "properties": {
            "frame_index": {"type": "integer", "minimum": 0},
            "visible_hands": count,
            "visible_target_instances": {"type": "object", "additionalProperties": False,
                                         "properties": {key: count for key in census_targets}, "required": list(census_targets)},
            "observation": {"type": "string", "minLength": 1}},
            "required": ["frame_index", "visible_hands", "visible_target_instances", "observation"]}
        result["properties"]["frame_census"] = {"type": "array", "items": frame,
                                                "minItems": len(census_frames), "maxItems": len(census_frames)}
        result["required"].append("frame_census")
    if hand_requirements:
        result["properties"]["hand_traces"] = hand_review.rows_schema()
        result["required"].append("hand_traces")
    return result


def wire_schema(census=False, hands=False):
    """Small provider schema; exact identities/cardinalities remain local gates.

    Repeating dozens of check properties or a fixed 49-row array can expand a
    provider's constrained decoder. This represents the SAME observations as
    arrays; it does not remove any check or any requested candidate frame.
    """
    def obj(properties):
        return {"type": "object", "additionalProperties": False,
                "properties": properties, "required": list(properties)}
    string={"type":"string"}
    integer={"type":"integer"}
    result=obj({"checks":{"type":"array","items":obj({"check_id":string,
        "value":{"type":"string","enum":["yes","no","unknown"]},
        "frame_indices":{"type":"array","items":integer},"observation":string})},
        "limitations":{"type":"array","items":string}})
    if census:
        result["properties"]["frame_census"]={"type":"array","items":obj({
            "frame_index":integer,"visible_hands":integer,
            "target_instances":{"type":"array","items":obj({"target_id":string,"count":integer})},
            "observation":string})}
        result["required"].append("frame_census")
    if hands:
        result["properties"]["hand_traces"] = hand_review.rows_schema()
        result["required"].append("hand_traces")
    return result


def normalize_wire(observations, checks, census_targets=None, hand_requirements=None):
    """Reject duplicate/missing/unknown IDs before converting to the local form."""
    from jsonschema import validate as json_validate
    json_validate(observations, wire_schema(census_targets is not None, bool(hand_requirements)))
    rows=observations["checks"]
    ids=[r["check_id"] for r in rows]
    require(len(ids)==len(set(ids)) and set(ids)==set(checks), "review_must_cover_exact_check_ids_once")
    result={"checks":{r["check_id"]:{k:v for k,v in r.items() if k!="check_id"} for r in rows},
            "limitations":observations["limitations"]}
    if census_targets is not None:
        result["frame_census"]=[]
        for row in observations["frame_census"]:
            targets=[v["target_id"] for v in row["target_instances"]]
            require(len(targets)==len(set(targets)) and set(targets)==set(census_targets),
                    "census_must_cover_exact_target_ids_once")
            result["frame_census"].append({k:v for k,v in row.items() if k!="target_instances"} |
                {"visible_target_instances":{v["target_id"]:v["count"] for v in row["target_instances"]}})
    if hand_requirements:
        result["hand_traces"] = observations["hand_traces"]
    return result


WIRE_INSTRUCTION = ("\nResponse encoding: checks is an ARRAY, with exactly one row per named check using check_id. "
                    "For each frame_census row, target_instances is an ARRAY with exactly one row per listed target, "
                    "using target_id and count. All required checks, identities and supplied frames must be covered "
                    "exactly once. Omission, duplication, unexpected IDs or unknown counts fail local validation.")


def validate(observations, checks, supplied_frames, edges=(), census_targets=None, hand_requirements=None):
    from jsonschema import validate as json_validate
    json_validate(observations, schema(checks, supplied_frames if census_targets is not None else None, census_targets or (), hand_requirements))
    supplied = set(supplied_frames)
    for key, item in observations["checks"].items():
        require(set(item["frame_indices"]) <= supplied, "Review cited an unobserved candidate frame: " + key)
        require(item["value"] == "unknown" or bool(item["frame_indices"]), "yes/no review needs frame evidence")
    # An order claim must also agree with the returned event observations.
    contradictions = []
    for a, b in edges:
        a, b = "event:" + a, "event:" + b
        if a in observations["checks"] and b in observations["checks"]:
            left, right = observations["checks"][a], observations["checks"][b]
            if left["value"] == right["value"] == "yes" and min(left["frame_indices"]) > min(right["frame_indices"]):
                contradictions.append([a, b])
    failed = [k for k, v in observations["checks"].items() if v["value"] == "no"]
    unknown = [k for k, v in observations["checks"].items() if v["value"] == "unknown"]
    census = None
    if census_targets is not None:
        extra_hands, duplicate_objects, uncertain = [], [], []
        frames = [row["frame_index"] for row in observations["frame_census"]]
        require(len(frames) == len(set(frames)) and set(frames) == supplied, "frame_census_must_cover_exact_supplied_frames_once")
        for counts in observations["frame_census"]:
            frame = counts["frame_index"]
            if counts["visible_hands"] < 0:
                uncertain.append({"frame": frame, "quantity": "visible_hands"})
            elif counts["visible_hands"] > 2:
                extra_hands.append({"frame": frame, "visible_hands": counts["visible_hands"]})
            for entity, count in counts["visible_target_instances"].items():
                if count < 0:
                    uncertain.append({"frame": frame, "quantity": entity})
                elif count > 1:
                    duplicate_objects.append({"frame": frame, "entity": entity, "visible_instances": count})
        if extra_hands:
            failed.append("frame_census.extra_hands")
        if duplicate_objects:
            failed.append("frame_census.duplicate_target")
        if uncertain:
            unknown.append("frame_census.uncertain_count")
        census = {"all_supplied_frames_accounted_for": True, "extra_hands": extra_hands,
                  "duplicate_targets": duplicate_objects, "uncertain_counts": uncertain,
                  "scope": "visible instances in sampled candidate frames; zero visible copies may reflect occlusion"}
    hand_gate = None
    if hand_requirements:
        hand_gate = hand_review.validate(observations["hand_traces"], hand_requirements, supplied_frames, observations["checks"])
        failed.extend(hand_gate["failed"])
        unknown.extend(hand_gate["unknown"])
    result = {"passed_in_sampled_visual_scope": not failed and not unknown and not contradictions,
            "failed_checks": failed, "unknown_checks": unknown, "order_contradictions": contradictions,
            "all_frame_visual_guarantee": False, "human_3d_or_force_labels_created": False,
            "training_admitted": False}
    if census is not None:
        result["frame_census"] = census
    if hand_gate is not None:
        result["hand_correspondence"] = hand_gate
    return result


def census_instruction(targets):
    from .prompts import compact
    return (
        "\nAdditionally fill the frame_census ARRAY with exactly one entry for EVERY supplied candidate frame, "
        "using its exact native frame_index. Never omit or duplicate a supplied frame. "
        "Count all visible hands, including resting hands on the background torso and hands manipulating foreground objects. "
        "For EACH listed target identity, count visible instances anywhere in the image. A hand carrying a copy while the "
        "original remains at its starting location means TWO instances, even if later frames return to one. "
        "Do not infer the count from the desired inventory or assume a successful pickup. Trace initial color, shape and "
        "location; distinguish two simultaneous objects from ordinary motion blur. Zero means no visible instance, which "
        "may be occlusion; describe the occlusion. Use -1 if the visible count cannot be determined. "
        "Give a short concrete observation for each frame. Aggregate identity or operator checks cannot override an extra "
        "visible object or hand in the census. Targets to count: " + compact(list(targets))
    )


def anchor(root, budget, directory, receipt, expected, simulation_image):
    directory = Path(directory)
    output = directory / "review.json"
    if output.exists():
        result = read(output)
        require(result["candidate"] == receipt["image"], "Review candidate changed")
        checked(result["model_evidence"]["request"])
        checked(result["model_evidence"]["result"])
        return result
    supplied = [{"frame_index": 0, "pts_seconds": 0.}]
    checks = anchor_checks(expected)
    request = {"schema": "resimpair_anchor_review_request_v4", "implementation": implementation(),
               "model_config": bind(Path(root) / "config.json"),
               "prompt": review_prompt(checks, expected, supplied, "human_initial_image") + WIRE_INSTRUCTION,
               "response_schema": wire_schema(), "acceptance_schema": schema(checks),
               "images": [{"label": "REFERENCE ONLY: simulation initial scene; not a candidate observation", "image": simulation_image},
                          {"label": "CANDIDATE initial human image, frame_index=0", "image": receipt["image"]}]}
    write(directory / "review_request.json", request)
    observations, evidence = reviewed_json(root, budget, directory / "review_request.json")
    observations = normalize_wire(observations, checks)
    gate = validate(observations, checks, [0])
    result = {"candidate": receipt["image"], "model_evidence": evidence, "observations": observations,
              "gate": gate, "automatic_visual_inference": True, "manual_observation_count": 0}
    write(output, result)
    return result


@traced("video_review", lambda root, budget, directory, contract, execute=True: directory,
        cache_file="review.json")
def video(root, budget, directory, contract, *, execute=True):
    from .media import packet
    directory = Path(directory)
    path = directory / "human.mp4"
    output = directory / "review.json"
    hands = review_requirements(contract["generation_information"])
    if output.exists():
        result = read(output)
        if hands:
            require(result.get("hand_correspondence_rules") == hand_review.VERSION,
                    "prior_review_has_different_hand_rules_preserve_it_and_use_a_new_audit_directory")
        require(result["candidate"] == bind(path), "Review candidate changed")
        checked(result["model_evidence"]["request"])
        checked(result["model_evidence"]["result"])
        return result
    frame_packet = packet(path, directory / "review_frames", read(Path(root) / "experiment.json")["review_frame_budget"])
    checks = video_checks(contract)
    frames = [{k: f[k] for k in ["frame_index", "pts_seconds"]} for f in frame_packet["frames"]]
    targets = [e["id"] for e in contract["generation_information"]["inventory"] if e["role"] == "target"]
    request = {"schema": "resimpair_video_review_request_v4", "implementation": implementation(),
               "model_config": bind(Path(root) / "config.json"),
               "candidate": bind(path), "frames": bind(directory / "review_frames/packet.json"),
               "prompt": review_prompt(checks, contract["generation_information"], frames, "generated_human_video") + census_instruction(targets) + hand_review.instruction(hands) + WIRE_INSTRUCTION,
               "response_schema": wire_schema(census=True, hands=bool(hands)),
               "acceptance_schema": schema(checks, [f["frame_index"] for f in frames], targets, hands),
               "hand_correspondence_rules": hand_review.VERSION if hands else None,
               "hand_requirements": hands,
               "frame_census_targets": targets,
               "images": [{"label": f"CANDIDATE native frame {f['frame_index']} PTS {f['pts_seconds']:.9f} seconds", "image": f["image"]}
                          for f in frame_packet["frames"]]}
    write(directory / "review_request.json", request)
    if not execute:
        return {"status": "prepared_review_only", "request": bind(directory / "review_request.json"),
                "candidate": bind(path), "sample_frames": len(frames), "census_targets": targets}
    observations, evidence = reviewed_json(root, budget, directory / "review_request.json")
    observations = normalize_wire(observations, checks, targets, hands)
    gate = validate(observations, checks, [f["frame_index"] for f in frames], contract["measured"]["required_event_precedence"], targets, hands)
    result = {"candidate": bind(path), "packet": bind(directory / "review_frames/packet.json"),
              "observations": observations, "model_evidence": evidence, "gate": gate,
              "automatic_visual_inference": True, "manual_observation_count": 0,
              "hand_correspondence_rules": hand_review.VERSION if hands else None}
    write(output, result)
    return result
