"""Bounded review intervals from existing 2D diagnostics, never contact truth.

This module does not run detectors, trackers, providers or training. It groups
signals into candidate spans and selects at most two short evidence windows for
independent VLM/manual inspection. Missed hands and lost tracks stay unknown.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .contact_diagnostics import validate_roi
from .task_evidence import canonical_sha256, file_sha256


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def _binding(value, verify_files=True):
    _require(isinstance(value, dict) and isinstance(value.get("path"), str),
             "Missing bound artifact")
    digest = value.get("sha256")
    _require(isinstance(digest, str) and len(digest) == 64
             and all(c in "0123456789abcdef" for c in digest), "Invalid SHA256")
    if verify_files:
        _require(file_sha256(value["path"]) == digest, "Bound artifact SHA256 mismatch")
    return {"path": value["path"], "sha256": digest}


def _runs(indices):
    """Contiguous frames form one span, not one claimed error per frame."""
    spans = []
    for frame in indices:
        if spans and frame == spans[-1][1] + 1:
            spans[-1][1] = frame
        else:
            spans.append([frame, frame])
    return spans


def _verify_csv(path, rows):
    """CSV is a redundant rendering of JSON rows, not independent evidence."""
    with Path(path).open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    _require(len(actual) == len(rows), "Timeline frame count differs from diagnostic")
    for saved, row in zip(actual, rows):
        _require(set(saved) == set(row), "Timeline fields differ from diagnostic")
        for key, value in row.items():
            expected = "" if value is None else str(value)
            _require(saved[key] == expected, "Timeline disagrees with diagnostic: " + key)


def triage_interaction(diagnostic_path, *, maximum_visible_hands,
                       max_requests=2, max_interval_frames=12, context_frames=2,
                       minimum_candidate_run_frames=2, timeline_path=None,
                       verify_files=True):
    """Return refinement-compatible, provisional review requests.

    ``maximum_visible_hands`` is a declared rendering budget, not anatomy GT.
    Motion-before-nearness means a visual timing candidate before the first
    measured 2D-near frame. It is not evidence of impossible physical motion.
    """
    for key, value, minimum in (
        ("maximum_visible_hands", maximum_visible_hands, 1),
        ("max_requests", max_requests, 0),
        ("max_interval_frames", max_interval_frames, 1),
        ("context_frames", context_frames, 0),
        ("minimum_candidate_run_frames", minimum_candidate_run_frames, 1),
    ):
        _require(type(value) is int and value >= minimum, "Invalid " + key)
    _require(max_requests <= 2, "At most two review intervals are allowed")
    path = Path(diagnostic_path).resolve(strict=True)
    data = json.loads(path.read_text())
    _require(data.get("schema") == "fixed_camera_interaction_diagnostic_v1",
             "Unsupported diagnostic schema")
    media = _binding(data.get("video"), verify_files)
    model = _binding(data.get("model"), verify_files)
    rows, params = data.get("rows"), data.get("parameters", {})
    _require(isinstance(rows, list) and rows, "Missing diagnostic rows")
    _require(data["video"].get("frames") == len(rows), "Declared frame count mismatch")
    _require([r.get("frame_index") for r in rows] == list(range(len(rows))),
             "Require a complete ordered timeline; missing frames remain unreviewable")
    stamps = [r.get("timestamp_seconds") for r in rows]
    _require(all(_number(t) and t >= 0 for t in stamps)
             and all(a < b for a, b in zip(stamps, stamps[1:])), "Invalid actual timestamps")
    for key in ("hand_near_threshold_px", "motion_threshold_px_per_frame"):
        _require(_number(params.get(key)) and params[key] > 0, "Invalid diagnostic threshold")
    handle_roi = validate_roi(data.get("handle_roi_normalized_xyxy"))
    background_roi = validate_roi(data.get("background_roi_normalized_xyxy"))
    _require(isinstance(data.get("roi_provenance"), str) and data["roi_provenance"].strip(),
             "Explicit ROI provenance is required")
    bindings = {
        "diagnostic": {"path": str(path), "sha256": file_sha256(path)},
        "media": media, "model": model,
    }
    if timeline_path is not None:
        timeline_path = Path(timeline_path).resolve(strict=True)
        _verify_csv(timeline_path, rows)
        bindings["timeline"] = {"path": str(timeline_path), "sha256": file_sha256(timeline_path)}
    unknown_tracking, no_hand, unknown_near, unknown_motion = [], [], [], []
    extra, movement_before_near = [], []
    first_near = None
    for row in rows:
        i, count = row["frame_index"], row.get("detected_hand_count")
        _require(type(count) is int and count >= 0, "Invalid detected-hand count")
        _require(row.get("status") in {"tracked_2d_proxy", "unknown"}, "Unknown track status")
        tracked = row["status"] == "tracked_2d_proxy"
        distance = row.get("nearest_hand_keypoint_to_handle_track_px")
        step = row.get("handle_step_px")
        _require(distance is None or (_number(distance) and distance >= 0),
                 "Invalid visual distance")
        _require(step is None or (_number(step) and step >= 0), "Invalid track displacement")
        near = None if not tracked or count == 0 or distance is None else (
            distance <= params["hand_near_threshold_px"])
        moving = None if not tracked or step is None else step > params["motion_threshold_px_per_frame"]
        if not tracked:
            unknown_tracking.append(i)
        if count == 0:
            no_hand.append(i)
        if near is None:
            unknown_near.append(i)
        if moving is None:
            unknown_motion.append(i)
        if count > maximum_visible_hands:
            extra.append(i)
        if near is True and first_near is None:
            first_near = i
        if moving is True and near is False and first_near is None:
            movement_before_near.append(i)

    signals = {
        "extra_hand_candidate": extra, "lost_track_unknown": unknown_tracking,
        "motion_before_hand_near_candidate": movement_before_near,
    }
    candidates, omitted_short = [], {}
    priorities = {"motion_before_hand_near_candidate": 0,
                  "extra_hand_candidate": 1, "lost_track_unknown": 2}
    for kind, indices in signals.items():
        omitted_short[kind] = 0
        for start, end in _runs(indices):
            length = end - start + 1
            # A lost-track interval is uncertainty even if it lasts one frame.
            if kind != "lost_track_unknown" and length < minimum_candidate_run_frames:
                omitted_short[kind] += length
                continue
            left = max(0, start - min(context_frames, max_interval_frames - 1))
            right = min(len(rows) - 1, end + context_frames, left + max_interval_frames - 1)
            # Long spans get an onset evidence window; their unsampled remainder
            # is explicitly retained below rather than claimed as reviewed.
            candidates.append({
                "candidate_id": f"{kind}_{len(candidates):03d}", "kind": kind,
                "signal_frame_interval": [start, end], "signal_frame_count": length,
                "signal_timestamp_interval_seconds": [stamps[start], stamps[end]],
                "proposed_review_interval": [left, right],
                "window_selection": "full_span_with_context" if right >= end else "onset_only",
                "confirmation": "unknown",
                "requires_vlm_or_manual_visual_confirmation": True,
                "physical_contact": "unknown",
            })
    # Prefer short complete/representative windows and avoid spending both
    # requests on the same diagnostic family while another family is available.
    ranked = sorted(candidates, key=lambda item: (
        item["proposed_review_interval"][1] - item["proposed_review_interval"][0] + 1,
        priorities[item["kind"]], item["proposed_review_interval"][0]))
    selected, kinds = [], set()
    for candidate in ranked:
        if len(selected) == max_requests:
            break
        if candidate["kind"] not in kinds:
            selected.append(candidate)
            kinds.add(candidate["kind"])
    requests, selected_frames = [], set()
    for candidate in selected:
        start, end = candidate["proposed_review_interval"]
        selected_frames.update(range(start, end + 1))
        requests.append({
            "query_id": candidate["candidate_id"], "frame_interval": [start, end],
            "interval": "start_inclusive_end_inclusive", "reason": candidate["kind"],
            "timestamp_interval_seconds": [stamps[start], stamps[end]],
            "media_sha256": media["sha256"], "evidence_artifact": bindings["diagnostic"],
            "candidate_signal_interval": candidate["signal_frame_interval"],
            "semantic_uncertainty_resolved": False,
        })
    return {
        "schema": "r2h_interaction_triage_v1", "evidence_bindings": bindings,
        "source_model_metadata": data["model"],
        "roi": {"handle_normalized_xyxy": handle_roi,
                "background_normalized_xyxy": background_roi,
                "provenance": data["roi_provenance"]},
        "roi_configuration_sha256": canonical_sha256({
            "handle": handle_roi, "background": background_roi,
            "provenance": data["roi_provenance"]}),
        "rules": {
            "provisional": True, "calibrated_acceptance_thresholds": False,
            "maximum_visible_hands": maximum_visible_hands,
            "source_diagnostic_parameters": params,
            "minimum_candidate_run_frames": minimum_candidate_run_frames,
            "max_interval_frames": max_interval_frames, "context_frames": context_frames,
            "max_requests": max_requests,
            "selection": "shortest_windows_with_one_per_signal_family",
            "first_observed_hand_near_frame": first_near,
        },
        "candidate_groups": candidates, "refinement_requests": requests,
        "summary": {
            "frames": len(rows), "extra_hand_signal_frames": len(extra),
            "extra_hand_candidate_groups": sum(x["kind"] == "extra_hand_candidate" for x in candidates),
            "selected_intervals": len(requests), "selected_unique_frames": len(selected_frames),
            "unselected_candidate_groups": len(candidates) - len(selected),
            "short_signal_frames_not_selected": omitted_short,
        },
        "unknown_coverage": {
            "tracking_unknown_frames": len(unknown_tracking),
            "tracking_unknown_intervals": _runs(unknown_tracking),
            "hand_detection_absent_frames": len(no_hand),
            "hand_nearness_unknown_frames": len(unknown_near),
            "motion_unknown_frames": len(unknown_motion),
            "physical_contact_unknown_frames": len(rows),
            "outside_selected_review_windows_frames": len(rows) - len(selected_frames),
            "selected_windows_actually_reviewed": False,
            "zero_candidates_is_not_quality_pass": True,
        },
        "claim_scope": "provisional_2d_visual_anomaly_and_observability_triage",
        "selection_is_not_verification": True,
        "human_actor_mismatch_confirmed": False, "physical_contact_ground_truth": False,
        "quality_release_approved": False, "training_admitted": False,
        "limitations": [
            "Detected extra hands are candidates; independent VLM or visual review must confirm a rendering-plan mismatch.",
            "A missing hand detection does not establish absence of a human hand.",
            "KLT tracks image appearance; occluding fingers can capture the handle track.",
            "2D motion before measured hand nearness does not establish impossible motion or absence of 3D contact.",
            "ROIs and thresholds are provisional and have not been calibrated as acceptance criteria.",
            "An onset-only evidence window cannot verify the remainder of its candidate span.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--maximum-visible-hands", type=int, required=True)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=2)
    args = parser.parse_args(argv)
    result = triage_interaction(args.diagnostic, maximum_visible_hands=args.maximum_visible_hands,
                                max_requests=args.max_requests, timeline_path=args.timeline)
    if args.output.exists():
        raise ValueError("Refusing to overwrite triage evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
