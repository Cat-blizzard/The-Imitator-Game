"""Versioned visible hand-ownership gate, separate from physics qualification.

These checks only test the consistency and completeness of supplied visual
observations. They cannot make an inaccurate VLM observation true.
"""
from .core import require
from .prompts import compact

VERSION = "resimpair_anatomical_hand_review_v1"


def rows_schema():
    properties = {
        "requirement_id": {"type": "string", "minLength": 1},
        "observed_hand": {"enum": ["left", "right", "unknown"]},
        "initial_shoulder_image_side": {"enum": ["image_left", "image_right", "unknown"]},
        "limb_trace": {"enum": ["continuous", "occluded", "unknown"]},
        "frame_indices": {"type": "array", "uniqueItems": True, "items": {"type": "integer", "minimum": 0}},
        "observation": {"type": "string", "minLength": 1},
    }
    return {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}}


def instruction(requirements):
    if not requirements:
        return ""
    return (
        "\nAlso return hand_traces, exactly one row for each requirement_id below. Independently identify "
        "the hand actually doing the action or holding the object; do not copy the expected side. Report "
        "observed_hand anatomically and the image side of THAT arm's shoulder in candidate frame 0. "
        "Trace from that initial shoulder through elbow and wrist to the action, even when the wrist "
        "crosses the body. A wrist's screen position is not its anatomical side. Cite frame 0 AND the "
        "event frames (for terminal requirements also the last supplied frame). Describe the visible "
        "shoulder, elbow, wrist and contacted object. Set limb_trace=occluded or unknown when ownership "
        "cannot be established in the sampled evidence; never infer ownership from the desired role. "
        "continuous means a coherent visible arm identity in the cited samples, not proof of every "
        "unsampled frame. A positive goal or event check cannot override a wrong or unknown hand. "
        "Hand requirements: " + compact(requirements)
    )


def validate(rows, requirements, supplied_frames, checks):
    from jsonschema import validate as json_validate
    json_validate(rows, rows_schema())
    ids = [row["requirement_id"] for row in rows]
    require(len(ids) == len(set(ids)) and set(ids) == set(requirements), "hand_traces_must_cover_exact_requirements_once")
    supplied = set(supplied_frames)
    require(0 in supplied, "hand_review_requires_initial_candidate_frame")
    failed, unknown = [], []
    for row in rows:
        rid = row["requirement_id"]
        expected = requirements[rid]
        frames = set(row["frame_indices"])
        require(frames <= supplied, "hand_trace_cites_unobserved_frame:" + rid)
        key = "hand_trace:" + rid
        if row["observed_hand"] not in {"unknown", expected["human_hand"]} or row["initial_shoulder_image_side"] not in {
                "unknown", expected["initial_shoulder_image_side"]}:
            failed.append(key + ":wrong_hand_or_shoulder_origin")
        if row["observed_hand"] == "unknown" or row["initial_shoulder_image_side"] == "unknown" or row["limb_trace"] != "continuous":
            unknown.append(key + ":unresolved_ownership")
        if 0 not in frames or len(frames) < 2:
            unknown.append(key + ":missing_initial_to_action_evidence")
        if rid.startswith("terminal:"):
            if max(supplied) not in frames:
                unknown.append(key + ":missing_ending_evidence")
        elif rid in checks and checks[rid]["value"] == "yes":
            if not (frames - {0}) & set(checks[rid]["frame_indices"]):
                unknown.append(key + ":not_bound_to_event_evidence")
    return {"rules": VERSION, "failed": failed, "unknown": unknown,
            "scope": "structured visual observations in sampled candidate frames; not an independent pixel detector or all-frame guarantee"}
