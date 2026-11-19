"""Independent, versioned single-actor evidence gate.

This module validates observations and their media binding. It performs no
visual inference. A successful replay of annotations is not detector accuracy.
Legacy hand-count reviews cannot be converted to single-person acceptance.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from jsonschema import Draft202012Validator

from .common import file_sha256, read_json

RULE_VERSION = "single_actor_v2.1"
SCHEMA_PATH = Path(__file__).with_name("schemas") / "single_actor_review_v2.schema.json"
SCHEMA = read_json(SCHEMA_PATH)
Draft202012Validator.check_schema(SCHEMA)
if SCHEMA["properties"]["rule_version"]["const"] != RULE_VERSION:
    raise RuntimeError("Actor schema and evaluator versions differ")
_VALIDATOR = Draft202012Validator(SCHEMA)
CHECKS = (
    "person_count", "left_hand_ownership", "right_hand_ownership",
    "body_structure", "body_position", "task_roles",
)
DIRECT_KINDS = {"direct_visual", "human_annotation"}

# The frozen semantics are deliberately separate from the generator's plan.
RULES = {
    "person_count": "Exactly one operator; count bodies/arm origins, never hands/2.",
    "left_hand_ownership": "The anatomical left hand belongs to that one operator; "
                           "image-left is not anatomical left.",
    "right_hand_ownership": "The anatomical right hand belongs to that same operator; "
                            "unseen or ambiguous ownership is unknown.",
    "body_structure": "Visible arms, hands and torso form a plausible continuous "
                      "single body without extra, disconnected or swapped limbs.",
    "body_position": "Body/arm origins agree with the declared single operator "
                     "position and reach; independent opposite-side torsos fail.",
    "task_roles": "Visible hand roles agree with the task/actor plan, including "
                  "inactive hands; no robot-to-person one-for-one substitution.",
}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _media_frame_count(path, scope):
    if scope == "anchor":
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise ValueError("Bound anchor is not a decodable image")
        return 1
    # Count decoded frames rather than trusting a caller-supplied count or
    # container nb_frames tag. No timestamps/frames are synthesized.
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or result.stderr.strip():
        raise ValueError("Bound video did not decode cleanly")
    try:
        streams = json.loads(result.stdout)["streams"]
        count = int(streams[0]["nb_read_frames"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("Cannot establish actual video frame count") from exc
    if count < 1:
        raise ValueError("Empty video cannot support visual evidence")
    return count


def evaluate(report, *, artifact_path=None, plan_sha256=None):
    """Validate a report and reduce it to a conservative gate decision.

    Invalid schemas and altered media raise ValueError. Valid negative or
    incomplete evidence returns reject or needs_review. Historical reviews may
    have plan_sha256=null but can never authorize new generation. An accepted
    anchor only clears this actor gate, not task/object/overall quality gates.
    """
    if not isinstance(report, dict):
        raise ValueError("Review must be an object")
    errors = list(_VALIDATOR.iter_errors(report))
    if errors:
        first = errors[0]
        location = ".".join(str(x) for x in first.absolute_path) or "<root>"
        raise ValueError(f"Actor review schema violation at {location}: {first.message}")
    if (report.get("schema") != "r2h_single_actor_review_v2"
            or report.get("rule_version") != RULE_VERSION):
        raise ValueError("Explicit single-actor v2 review required; legacy hand counts are insufficient")
    artifact = report.get("artifact")
    if not isinstance(artifact, dict) or not _sha(artifact.get("sha256")):
        raise ValueError("Review needs a media path and SHA256")
    try:
        bound_path = Path(artifact["path"]).resolve(strict=True)
        actual_path = Path(artifact_path).resolve(strict=True) if artifact_path else bound_path
        if actual_path != bound_path or file_sha256(actual_path) != artifact["sha256"]:
            raise ValueError("Review media binding differs from the actual artifact")
    except (KeyError, OSError, TypeError) as exc:
        raise ValueError("Missing or unreadable bound media") from exc
    declared_plan = report.get("plan_sha256")
    if declared_plan is not None and not _sha(declared_plan):
        raise ValueError("plan_sha256 must be null or SHA256")
    if plan_sha256 is not None and (not _sha(plan_sha256) or declared_plan != plan_sha256):
        raise ValueError("Review does not belong to the current actor plan")
    if report.get("scope") not in {"anchor", "sampled_video"}:
        raise ValueError("Review scope must be anchor or sampled_video")
    media_frame_count = _media_frame_count(actual_path, report["scope"])
    reviewer = report.get("reviewer", {})
    if not isinstance(reviewer, dict) or not reviewer.get("id") or reviewer.get("kind") not in {
        "human", "assistant_visual", "vlm"
    }:
        raise ValueError("An identified reviewer and observation kind are required")
    limitations = report.get("limitations")
    if (not isinstance(limitations, list) or not limitations
            or not all(isinstance(x, str) and x.strip() for x in limitations)):
        raise ValueError("Nonempty evidence limitations are required")
    evidence = report.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("Evidence must be an explicit list")
    evidence_map = {}
    for item in evidence:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not item["id"] or item["id"] in evidence_map):
            raise ValueError("Evidence IDs must be unique nonempty strings")
        if item.get("kind") not in DIRECT_KINDS | {"vlm", "detector"}:
            raise ValueError("Unknown evidence kind")
        if item.get("artifact_sha256") != artifact["sha256"]:
            raise ValueError("Evidence belongs to different media")
        frames = item.get("frame_indices")
        if (not isinstance(frames, list) or not frames
                or not all(type(x) is int and x >= 0 for x in frames)
                or len(frames) != len(set(frames))):
            raise ValueError("Evidence needs distinct nonnegative frame indices")
        if report["scope"] == "anchor" and frames != [0]:
            raise ValueError("A static anchor only has frame 0")
        if any(frame >= media_frame_count for frame in frames):
            raise ValueError("Evidence references a frame beyond the actual decoded media")
        if not isinstance(item.get("observation"), str) or not item["observation"].strip():
            raise ValueError("Evidence needs a concrete observation")
        evidence_map[item["id"]] = item
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise ValueError("All six independent actor checks are mandatory")
    effective, reasons = {}, []
    for name in CHECKS:
        check = checks[name]
        if not isinstance(check, dict) or check.get("status") not in {"pass", "fail", "unknown"}:
            raise ValueError(f"Invalid status for {name}")
        ids = check.get("evidence_ids")
        if (not isinstance(ids, list) or not all(isinstance(x, str) and x in evidence_map for x in ids)
                or len(ids) != len(set(ids))):
            raise ValueError(f"Invalid evidence references for {name}")
        if not isinstance(check.get("note"), str) or not check["note"].strip():
            raise ValueError(f"Missing observation limit/reason for {name}")
        status = check["status"]
        directly_supported = (
            reviewer["kind"] in {"human", "assistant_visual"}
            and any(evidence_map[x]["kind"] in DIRECT_KINDS for x in ids)
        )
        if status in {"pass", "fail"} and not directly_supported:
            status = "unknown"
            reasons.append(f"{name}:unsupported_by_direct_observation")
        effective[name] = {**check, "effective_status": status}
    if any(x["effective_status"] == "fail" for x in effective.values()):
        status = "reject"
    elif any(x["effective_status"] == "unknown" for x in effective.values()):
        status = "needs_review"
    elif declared_plan is None:
        status = "needs_review"
        reasons.append("no_predeclared_actor_plan_binding")
    else:
        status = "accepted"
    if status != "accepted":
        reasons.extend(f"{name}:{item['effective_status']}" for name, item in effective.items()
                       if item["effective_status"] != "pass")
    return {
        "rule_version": RULE_VERSION, "status": status,
        "can_generate_video": status == "accepted" and report["scope"] == "anchor",
        "artifact": artifact, "plan_sha256": declared_plan,
        "actual_media_frame_count": media_frame_count,
        "scope": report["scope"], "checks": effective, "reasons": reasons,
        "visual_inference_performed_by_gate": False,
        "full_video_actor_consistency_verified": False,
        "quality_release_approved": False, "training_admitted": False,
        "limitations": limitations,
    }


def gate_anchor(report_path, anchor_path, plan_path):
    """Bind a pre-video anchor review to exact current image and plan bytes."""
    report = read_json(report_path)
    if report.get("scope") != "anchor":
        raise ValueError("Video generation needs an anchor-scope review")
    return evaluate(report, artifact_path=anchor_path, plan_sha256=file_sha256(plan_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--anchor")
    parser.add_argument("--plan")
    args = parser.parse_args()
    if bool(args.anchor) != bool(args.plan):
        parser.error("--anchor and --plan must be used together")
    result = (gate_anchor(args.report, args.anchor, args.plan) if args.anchor
              else evaluate(read_json(args.report)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
