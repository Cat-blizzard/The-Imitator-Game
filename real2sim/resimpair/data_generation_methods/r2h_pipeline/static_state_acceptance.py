"""Independent static-image task/object/state gate, separate from actor identity.

This is an annotation and binding validator, not an automatic visual detector.
The caller must derive expected_context from the independently verified package
chain, never from this review. Passing cannot authorize video or physical labels.
"""
from __future__ import annotations

from pathlib import Path
import math

from jsonschema import Draft202012Validator

from .common import file_sha256, read_json

RULE_VERSION = "static_state_v1.0"
SCHEMA_PATH = Path(__file__).with_name("schemas") / "static_state_review_v1.schema.json"
SCHEMA = read_json(SCHEMA_PATH)
Draft202012Validator.check_schema(SCHEMA)
if SCHEMA["properties"]["rule_version"]["const"] != RULE_VERSION:
    raise RuntimeError("Static-state schema and evaluator versions differ")
_VALIDATOR = Draft202012Validator(SCHEMA)
_CONTEXT_VALIDATOR = Draft202012Validator(SCHEMA["properties"]["context"])
CHECKS = (
    "task_correspondence", "object_identity", "relative_scale", "role_mapping",
    "state_relations", "non_target_preservation", "camera_correspondence",
)
RULES = {
    "task_correspondence": "Visible static scene matches the bound task and exact condition role; no claim about actions that occurred.",
    "object_identity": "Each visible task object retains its category, shape and defining identity, not color alone; hidden identity is unknown.",
    "relative_scale": "Corresponding visible objects have plausible bound-reference relative dimensions; missing correspondence is unknown.",
    "role_mapping": "Selected object, source and destination retain their execution-bound entity roles; screen-left is not a semantic role.",
    "state_relations": "Visible containment and positions match initial or terminal relation as bound, without importing later states; RGB does not prove physical contact forces.",
    "non_target_preservation": "Visible non-target task objects retain identity and state; robot removal does not license object removal.",
    "camera_correspondence": "Scene layout and visible role mapping match the selected physical camera; mirrored camera or camera substitution fails.",
}


def _validate(validator, value, label):
    errors = list(validator.iter_errors(value))
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.absolute_path)) or "<root>"
        raise ValueError(f"{label} schema violation at {location}: {first.message}")


def evaluate(report, *, image_path, package_path, expected_context):
    """Require exact source/execution/camera/condition bindings and direct evidence.

    expected_context is a required integration boundary: the verified generator
    package supplies it. This function does not accept a report-derived default.
    It validates strict schemas and immutable bytes, then replays observations.
    """
    _validate(_VALIDATOR, report, "Static-state review")
    _validate(_CONTEXT_VALIDATOR, expected_context, "Expected context")
    if report["context"] != expected_context:
        raise ValueError("Review context differs from independently verified package context")
    for context in (report["context"], expected_context):
        if isinstance(context["human_fps"], bool) or not math.isfinite(context["human_fps"]):
            raise ValueError("Human fps must be finite")
        start, end = context["human_window_frames"]
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError("Human window must be a nonempty integer interval")
        roles = context["role_ids"]
        if len(set(roles.values())) != len(roles):
            raise ValueError("Selected object/source/destination role IDs must be distinct")
    try:
        package = Path(package_path).resolve(strict=True)
        image = Path(image_path).resolve(strict=True)
        reported_image = Path(report["artifact"]["path"]).resolve(strict=True)
        if report["package_sha256"] != file_sha256(package):
            raise ValueError("Review belongs to a different package")
        if image != reported_image or file_sha256(image) != report["artifact"]["sha256"]:
            raise ValueError("Review belongs to a different image")
    except (OSError, TypeError) as exc:
        raise ValueError("Missing or unreadable image/package") from exc
    import cv2
    decoded = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise ValueError("Bound artifact is not a decodable image")
    evidence = {}
    for item in report["evidence"]:
        if item["id"] in evidence:
            raise ValueError("Evidence IDs must be unique")
        if item["artifact_sha256"] != report["artifact"]["sha256"]:
            raise ValueError("Evidence belongs to different media")
        evidence[item["id"]] = item
    effective, reasons = {}, []
    for name in CHECKS:
        check = report["checks"][name]
        ids = check["evidence_ids"]
        if any(i not in evidence for i in ids):
            raise ValueError(f"Invalid evidence reference for {name}")
        status = check["status"]
        direct = (report["reviewer"]["kind"] in {"human", "assistant_visual"}
                  and any(evidence[i]["kind"] in {"direct_visual", "human_annotation"}
                          for i in ids))
        if status in {"pass", "fail"} and not direct:
            status = "unknown"
            reasons.append(f"{name}:unsupported_by_direct_observation")
        effective[name] = {**check, "effective_status": status}
    values = {v["effective_status"] for v in effective.values()}
    status = "reject" if "fail" in values else "needs_review" if "unknown" in values else "accepted"
    reasons.extend(f"{name}:{check['effective_status']}" for name, check in effective.items()
                   if check["effective_status"] != "pass")
    return {
        "rule_version": RULE_VERSION, "status": status,
        "clears_static_state": status == "accepted", "can_generate_video": False,
        "artifact": report["artifact"], "package_sha256": report["package_sha256"],
        "context": report["context"], "checks": effective, "reasons": reasons,
        "visual_inference_performed_by_gate": False,
        "actor_identity_verified": False, "action_order_verified": False,
        "physical_contact_verified": False, "quality_release_approved": False,
        "training_admitted": False, "limitations": report["limitations"],
    }


def gate_state(report_path, image_path, package_path, expected_context):
    return evaluate(read_json(report_path), image_path=image_path,
                    package_path=package_path, expected_context=expected_context)
