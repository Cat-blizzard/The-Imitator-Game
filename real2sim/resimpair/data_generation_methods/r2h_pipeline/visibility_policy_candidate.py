"""Opt-in routing of EXTRA visibility review; never an acceptance decision.

Known clear scenes default to the opposite camera. Only declared risks or
measured event gaps request an extra VLM review. No model call is made here.
Unknown qualification requires local review, not a fabricated 'clear' label.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from copy import deepcopy

RISK_FIELDS = (
    "hidden_content_reveal", "dense_or_large_clutter", "fine_contact_required",
    "articulation_occludes_target", "cheap_anomaly_detected",
)


def plan_review(scene, event_visibility=None):
    if not isinstance(scene, dict):
        raise ValueError("Explicit scene risk assessment required")
    for key in ("adapter_qualified", "layout_reviewed", *RISK_FIELDS):
        if key in scene and type(scene[key]) is not bool:
            raise ValueError("Risk/qualification values must be booleans: " + key)
    reasons = [key for key in RISK_FIELDS if scene.get(key) is True]
    missing = [key for key in ("adapter_qualified", "layout_reviewed", *RISK_FIELDS)
               if key not in scene]
    events = []
    view = "opposite"
    selection = scene.get("reference_view")
    if selection is not None:
        if (not isinstance(selection, dict) or selection.get("camera_id") not in ("native", "opposite")
                or not selection.get("reason") or not isinstance(selection.get("evidence"), dict)):
            raise ValueError("Reference view override requires a camera, reason and bound evidence")
        evidence = selection["evidence"]
        path = Path(evidence.get("path", ""))
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != evidence.get("sha256"):
            raise ValueError("Reference view evidence missing or changed")
        view = selection["camera_id"]
    if event_visibility is not None:
        if not isinstance(event_visibility, dict):
            raise ValueError("Event diagnostics must be a mapping")
        for event, report in event_visibility.items():
            cameras = report.get("cameras", {})
            selected = cameras.get(view)
            if selected is None:
                missing.append(view + "_event_evidence:" + event)
                continue
            count, visible = selected.get("frames"), selected.get("frames_ge16")
            if (type(count) is not int or type(visible) is not int or
                    count < 1 or not 0 <= visible <= count):
                raise ValueError("Malformed event pixel counts")
            if visible < count:
                reasons.append("measured_event_visibility_gap")
                events.append(event)
    needs_local = bool(missing or not scene.get("adapter_qualified") or not scene.get("layout_reviewed"))
    return {
        "schema": "extra_visibility_review_policy_candidate_v2",
        "scene_assessment": deepcopy(scene),
        "event_diagnostics": deepcopy(event_visibility),
        "policy_source": {"path": str(Path(__file__).resolve()),
                          "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "assessment_scope": "Caller-supplied risk/view assessment; hash binding is not an automatic semantic review",
        "preferred_reference_camera": view,
        "reference_view_selection": selection,
        "visibility_vlm_check": int(bool(reasons)),
        "reason": list(dict.fromkeys(reasons)), "event_ids": events,
        "needs_local_risk_review": needs_local,
        "missing_risk_evidence": missing,
        "routing_status": "needs_local_review" if needs_local else "ready",
        "extra_visibility_vlm_requested": bool(reasons),
        "paid_call_authorized": False, "calls_executed": 0,
        "mandatory_checks_unchanged": ["task_correspondence", "independent_physics",
            "actor_structure", "object_identity", "key_event_evidence", "continuity", "provenance"],
        "quality_release_approved": False,
        "limitations": ["A zero label skips only extra visibility VLM work, not other acceptance checks.",
            "16-pixel diagnostics do not prove visible contact; explicit fine-contact risks remain separate.",
            "Risk flags are source assessments, not an automatic semantic detector."],
    }
