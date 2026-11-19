"""Compare declared actor plans with independent observations; never average them away.

This is a decision/evidence-packet layer. It runs no new detectors or paid models.
Detected hand counts are candidate lower bounds, not verified people counts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .common import file_sha256, read_json, write_json
from .quality_model_review import validate_result


def _binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def _verify(binding):
    if (not isinstance(binding, dict) or not binding.get("path")
            or file_sha256(binding["path"]) != binding.get("sha256")):
        raise ValueError("Bound evidence is missing or changed")
    return Path(binding["path"])


def validate_plan(plan, media_hash):
    if plan.get("schema") != "r2h_actor_plan_v1" or plan.get("media_sha256") != media_hash:
        raise ValueError("Actor plan schema/media binding mismatch")
    budget = plan.get("maximum_visible_hands")
    if type(budget) is not int or not 0 <= budget <= 20:
        raise ValueError("maximum_visible_hands must be an explicit integer 0..20")
    people = plan.get("maximum_visible_people")
    if people is not None and (type(people) is not int or not 0 <= people <= 20):
        raise ValueError("maximum_visible_people must be an integer or null")
    if not plan.get("declaration_scope") or not plan.get("sources"):
        raise ValueError("Plan needs declaration scope and actual source artifacts")
    for source in plan["sources"]:
        _verify(source)
    return budget


def decide_counts(plan, inventory, rows, scores, *, max_requests=2, radius_frames=3):
    """Pure decision logic; disagreeing detectors cannot establish a clean pass."""
    if type(max_requests) is not int or max_requests < 0:
        raise ValueError("Nonnegative request budget required")
    if type(radius_frames) is not int or not 1 <= radius_frames <= 30:
        raise ValueError("radius_frames must be 1..30")
    budget = plan["maximum_visible_hands"]
    row_map = {r["frame_index"]: r for r in rows}
    if len(row_map) != len(rows):
        raise ValueError("Duplicate diagnostic frames")
    excess_vlm, count_disagreements, supported_same_frame = [], [], []
    for observation in inventory:
        frame = observation["frame_index"]
        if frame not in row_map:
            raise ValueError("VLM hand observation has no corresponding diagnostic frame")
        row = row_map[frame]
        if abs(observation["timestamp_seconds"]-row["timestamp_seconds"]) > 1e-5:
            raise ValueError("VLM and local observations disagree on source timestamp")
        vlm = observation["minimum_distinct_visible_hands"]
        detector = row["detected_hand_count"]
        if type(vlm) is not int or type(detector) is not int or min(vlm, detector) < 0:
            raise ValueError("Hand counts must be nonnegative integers")
        item = {"frame_index": frame, "timestamp_seconds": row["timestamp_seconds"],
                "vlm_minimum_distinct_visible_hands": vlm,
                "vlm_exact_count_claimed": observation["exact_count_known"],
                "mediapipe_detected_hand_candidates": detector,
                "actor_plan_maximum_visible_hands": budget}
        if vlm > budget:
            excess_vlm.append(item)
            if detector > budget:
                supported_same_frame.append(item)
        if vlm != detector:
            count_disagreements.append(item)
    excess_detector = [r["frame_index"] for r in rows if r["detected_hand_count"] > budget]
    requests, selected = [], []
    # A disagreement is useful for local inspection even when both counts may
    # reflect incomplete detection; it is not a logical disproof of either.
    seeds = [(x["frame_index"], "count_disagreement") for x in count_disagreements]
    seeds += [(x["frame_index"], "vlm_observed_actor_budget_excess") for x in excess_vlm]
    seeds += [(i, "local_detector_actor_budget_candidate") for i in excess_detector]
    maximum = max(row_map) if row_map else 0
    for center, reason in seeds:
        if any(abs(center-old) <= radius_frames*2 for old in selected):
            continue
        if len(requests) >= max_requests:
            break
        first, last = max(0, center-radius_frames), min(maximum, center+radius_frames)
        selected.append(center)
        requests.append({"request_id": f"actor_local_{len(requests)+1}",
                         "trigger": reason, "center_frame": center,
                         "frame_interval_inclusive": [first, last],
                         "selection": "continuous_native_frames_and_each_detected_hand_crop",
                         "no_new_review_performed": True})
    violation = bool(excess_vlm)
    if violation:
        status = "blocked_actor_plan_violation_observed"
    elif excess_detector or count_disagreements:
        status = "needs_local_actor_review"
    else:
        status = "no_actor_budget_violation_detected_in_available_evidence"
    return {
        "status": status, "vlm_budget_excess_observations": excess_vlm,
        "detector_budget_excess_frames": excess_detector,
        "cross_source_same_frame_excess": supported_same_frame,
        "count_disagreements": count_disagreements, "review_requests": requests,
        "quality_score_did_not_override_plan": True, "ordinal_scores_retained": scores,
        "plan_deviation_is_not_anatomical_impossibility": True,
        "person_count_status": "unknown_no_direct_person_count_evidence",
        "hand_count_not_converted_to_person_count": True,
        "local_detector_counts_are_candidates_not_ground_truth": True,
        "full_clip_actor_absence_verified": False,
        "quality_release_approved": False, "training_admitted": False,
    }


def _export_packets(video_path, output, requests, points, rows, *, max_hands=6):
    import cv2
    import numpy as np
    point_map = {p["frame_index"]: p for p in points}
    row_map = {r["frame_index"]: r for r in rows}
    needed = sorted({i for r in requests
                     for i in range(r["frame_interval_inclusive"][0],
                                    r["frame_interval_inclusive"][1]+1)})
    cap = cv2.VideoCapture(str(video_path))
    packets = []
    for index in needed:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if not ok:
            raise ValueError("Could not decode requested review frame")
        target = output/"local_review_packets"/f"frame_{index:06d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Refusing to replace an existing review packet")
        if not cv2.imwrite(str(target), image):
            raise ValueError("Could not write review frame")
        packet = {"frame_index": index, "timestamp_seconds": row_map[index]["timestamp_seconds"],
                  "full_frame": _binding(target), "hand_candidate_crops": []}
        if index in point_map:
            h, w = image.shape[:2]
            for hand_index, landmarks in enumerate(point_map[index]["hand_landmarks_pixels"][:max_hands]):
                coords = np.asarray(landmarks, dtype=float)
                if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
                    raise ValueError("Malformed hand landmarks")
                low, high = coords.min(0), coords.max(0)
                margin = np.maximum(16, (high-low)*.35)
                x0, y0 = np.maximum(0, np.floor(low-margin)).astype(int)
                x1, y1 = np.minimum([w, h], np.ceil(high+margin)).astype(int)
                if x1 <= x0 or y1 <= y0:
                    continue
                crop_path = target.with_name(f"frame_{index:06d}_hand{hand_index}.png")
                if not cv2.imwrite(str(crop_path), image[y0:y1, x0:x1]):
                    raise ValueError("Could not write hand review crop")
                packet["hand_candidate_crops"].append({
                    "artifact": _binding(crop_path), "bbox_native_xyxy": [int(v) for v in (x0,y0,x1,y1)],
                    "basis": "Existing detector keypoint extent plus margin; not hand identity ground truth"})
        packets.append(packet)
    cap.release()
    return packets


def run_guard(plan_path, quality_review_path, diagnostic_path, output, *,
              point_evidence_path=None, max_requests=2, radius_frames=3):
    plan_path, quality_review_path, diagnostic_path = map(Path, (
        plan_path, quality_review_path, diagnostic_path))
    output = Path(output)
    if (output/"actor_plan_guard.json").exists():
        raise ValueError("Use a new output directory; guard evidence is immutable")
    plan, quality, diagnostic = map(read_json, (plan_path, quality_review_path, diagnostic_path))
    _verify(quality["media"])
    media_hash = quality["media"]["sha256"]
    if diagnostic["video"]["sha256"] != media_hash:
        raise ValueError("Local diagnostic belongs to another video")
    _verify(diagnostic["video"])
    validate_plan(plan, media_hash)
    packet_path = _verify(quality["frame_packet"])
    raw_path = _verify(quality["raw_response"])
    frames = read_json(packet_path)["frames"]
    if read_json(packet_path)["source_sha256"] != media_hash:
        raise ValueError("VLM frame packet media binding mismatch")
    for frame in frames:
        _verify(frame)
    raw_validated = validate_result(read_json(raw_path), frames)
    for key in ("dimensions", "constraints", "visible_hand_inventory", "limitations"):
        if raw_validated[key] != quality[key]:
            raise ValueError("VLM summary differs from the bound original response")
    decision = decide_counts(plan, quality["visible_hand_inventory"], diagnostic["rows"],
                             quality["ordinal_vector"], max_requests=max_requests,
                             radius_frames=radius_frames)
    points = []
    point_binding = None
    if point_evidence_path:
        point_evidence_path = Path(point_evidence_path)
        points = read_json(point_evidence_path)
        point_binding = _binding(point_evidence_path)
        row_map = {r["frame_index"]: r for r in diagnostic["rows"]}
        if len(points) != len(row_map):
            raise ValueError("Point evidence and detector timeline differ")
        for p in points:
            row = row_map.get(p["frame_index"])
            if (row is None or abs(row["timestamp_seconds"]-p["timestamp_seconds"]) > 1e-5
                    or len(p["hand_landmarks_pixels"]) != row["detected_hand_count"]):
                raise ValueError("Point evidence count/time differs from bound diagnostic")
    output.mkdir(parents=True, exist_ok=True)
    packets = _export_packets(Path(quality["media"]["path"]), output,
                              decision["review_requests"], points, diagnostic["rows"])
    result = {
        "schema": "r2h_actor_plan_guard_v1", "media": quality["media"],
        "bindings": {"plan": _binding(plan_path), "quality_review": _binding(quality_review_path),
                     "local_diagnostic": _binding(diagnostic_path), "point_evidence": point_binding},
        "decision": decision, "packets": packets, "new_paid_calls": 0,
        "new_detector_inference": False, "generated_review_frames": len(packets),
        "local_review_executed": False,
        "limitations": [
            "Count discrepancy is a review trigger, not proof that either detector is wrong.",
            "VLM may infer people from cropped hands; this layer never turns that into a person count.",
            "A declared plan violation is distinct from anatomical impossibility.",
            "Local packets are prepared evidence, not completed VLM or human review.",
        ],
    }
    write_json(output/"actor_plan_guard.json", result)
    return result
