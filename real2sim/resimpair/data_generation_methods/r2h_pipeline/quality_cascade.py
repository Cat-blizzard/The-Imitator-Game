"""Evidence-first video screening. Perceptual scores cannot cancel task failures.

CPU metrics are uncalibrated triage/ranking signals, not embodied-task ground truth.
This module performs no downloads, model/API calls, training, or automatic release.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import time

from .common import file_sha256, read_json, write_json


DOMAINS = ("task_semantics", "embodiment", "interaction", "sim_physics")
RESULTS = {"pass_in_scope", "fail", "unknown", "not_applicable"}
DEFAULT_THRESHOLDS = {
    "low_laplacian_variance": 30.0,
    "low_global_change_mean_abs_difference": 0.5,
    "dark_pixel_value": 8,
    "bright_pixel_value": 247,
    "extreme_exposure_fraction": 0.5,
    "fixed_background_motion_diagonal_fraction": 0.004,
}


def _stats(values):
    import numpy as np
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {"count": 0, "median": None, "p05": None, "p95": None, "max": None}
    return {"count": len(values), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)), "max": float(max(values))}


def validate_background_roi(roi, provenance, camera_mode):
    if roi is None:
        return None
    if camera_mode != "fixed":
        raise ValueError("Background stability ROI is only supported for declared fixed cameras")
    if not provenance or not provenance.strip():
        raise ValueError("An explicit static-background ROI provenance is required")
    if (len(roi) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                             or not math.isfinite(v) for v in roi)):
        raise ValueError("ROI requires four finite normalized coordinates")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("Invalid normalized background ROI")
    return list(map(float, roi))


def temporal_integrity(timestamps, decoded_frames, declared_frames=None):
    finite = bool(timestamps) and all(isinstance(t, (int, float)) and math.isfinite(t)
                                      for t in timestamps)
    increasing = finite and all(b > a for a, b in zip(timestamps, timestamps[1:]))
    failures = []
    if not finite:
        failures.append("missing_or_nonfinite_timestamps")
    elif not increasing:
        failures.append("non_increasing_timestamps")
    if len(timestamps) != decoded_frames:
        failures.append("pts_decoded_frame_count_mismatch")
    if declared_frames is not None and declared_frames != decoded_frames:
        failures.append("declared_decoded_frame_count_mismatch")
    if decoded_frames < 2:
        failures.append("fewer_than_two_decoded_frames")
    intervals = [b-a for a, b in zip(timestamps, timestamps[1:])] if increasing else []
    cadence = _stats(intervals)
    # Variable-rate media is not invalid merely because its cadence is uneven.
    return {"status": "fail" if failures else "pass", "failures": failures,
            "strictly_increasing_pts": bool(increasing), "frame_count": decoded_frames,
            "pts_count": len(timestamps), "declared_frames": declared_frames,
            "cadence_seconds": cadence, "variable_rate_is_not_rejected": True}


def _verify_bound_file(binding):
    if not isinstance(binding, dict) or set(("path", "sha256")) - binding.keys():
        raise ValueError("Evidence requires an explicit path and sha256")
    path = Path(binding["path"]).resolve(strict=True)
    if file_sha256(path) != binding["sha256"]:
        raise ValueError("Evidence file hash mismatch: " + str(path))
    return path


def load_evidence(paths, media_sha256):
    """Read explicit small evidence ledgers; never infer a positive from no issue.

    Ledger creation is an adapter operation, not an additional review. Every check
    carries its actual source file and scope; unverified/missing bindings fail.
    """
    checks, manifests = [], []
    for source in paths:
        source = Path(source).resolve(strict=True)
        data = read_json(source)
        if data.get("schema") != "r2h_quality_evidence_v1":
            raise ValueError("Unsupported evidence ledger schema")
        if data.get("media_sha256") != media_sha256:
            raise ValueError("Evidence is bound to another video")
        if not isinstance(data.get("checks"), list):
            raise ValueError("Evidence checks must be a list")
        manifests.append({"path": str(source), "sha256": file_sha256(source)})
        for check in data["checks"]:
            if check.get("dimension") not in DOMAINS or check.get("result") not in RESULTS:
                raise ValueError("Unknown evidence dimension/result")
            if not check.get("scope") or not check.get("description"):
                raise ValueError("Evidence scope and description are required")
            if check.get("result") == "not_applicable" and not check.get("rationale"):
                raise ValueError("Not-applicable evidence requires a rationale")
            sources = check.get("sources")
            if not isinstance(sources, list) or not sources:
                raise ValueError("Every evidence check needs source artifacts")
            for binding in sources:
                _verify_bound_file(binding)
            checks.append(check)
    return checks, manifests


def dimension_decisions(checks, origin):
    dimensions = {}
    for dimension in DOMAINS:
        values = [x for x in checks if x["dimension"] == dimension]
        if dimension == "sim_physics" and origin != "robot_sim" and not values:
            dimensions[dimension] = {
                "status": "not_applicable",
                "reason": "This media is not a simulator execution; human physical correctness is not certified.",
                "checks": [],
            }
        elif any(x["result"] == "fail" for x in values):
            dimensions[dimension] = {"status": "fail", "checks": values}
        elif not values or any(x["result"] == "unknown" for x in values):
            dimensions[dimension] = {"status": "unknown", "checks": values,
                                     "reason": "Missing or unresolved evidence is not a pass."}
        elif all(x["result"] == "not_applicable" for x in values):
            dimensions[dimension] = {"status": "not_applicable", "checks": values}
        else:
            dimensions[dimension] = {"status": "pass_in_scope", "checks": values,
                                     "reason": "Only the recorded evidence scope is supported."}
    return dimensions


def aggregate_decision(integrity, dimensions, flags):
    failures = [key for key, value in dimensions.items() if value["status"] == "fail"]
    if integrity["status"] == "fail":
        failures.insert(0, "media_integrity")
    unknown = [key for key, value in dimensions.items() if value["status"] == "unknown"]
    return {
        "status": "reject" if failures else ("needs_review" if unknown or flags
                                            else "candidate_in_reviewed_scope"),
        "hard_failure_dimensions": failures, "unknown_dimensions": unknown,
        "triage_flags": flags,
        "perceptual_average_used": False,
        "quality_release_approved": False, "training_admitted": False,
        "release_requires_independent_task_pair_action_and_split_validation": True,
    }


def _background_motion(previous, current, roi):
    import cv2
    import numpy as np
    h, w = current.shape
    mask = np.zeros_like(current)
    x0, y0, x1, y1 = roi
    mask[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = 255
    points = cv2.goodFeaturesToTrack(previous, 120, .01, 4, mask=mask)
    if points is None or len(points) < 8:
        return None
    nxt, good, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if nxt is None:
        return None
    back, back_good, _ = cv2.calcOpticalFlowPyrLK(current, previous, nxt, None)
    if back is None:
        return None
    p, q, b = [x.reshape(-1, 2) for x in (points, nxt, back)]
    valid = (good.reshape(-1).astype(bool) & back_good.reshape(-1).astype(bool)
             & np.isfinite(q).all(axis=1) & np.isfinite(b).all(axis=1)
             & (np.linalg.norm(p-b, axis=1) <= 1.5))
    if valid.sum() < 8:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        p[valid], q[valid], method=cv2.RANSAC, ransacReprojThreshold=2.)
    if matrix is None or inliers is None or inliers.sum() < 6:
        return None
    return {"translation_diagonal_fraction": float(np.linalg.norm(matrix[:, 2])/np.hypot(w, h)),
            "rotation_degrees": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
            "scale": float(np.hypot(matrix[0, 0], matrix[1, 0])),
            "inlier_fraction": float(inliers.mean())}


def inspect_video(video, output, *, origin, camera_mode="unknown",
                  background_roi=None, roi_provenance=None, evidence_paths=(),
                  resize_max=384):
    import cv2
    import numpy as np
    if origin not in {"real_human", "robot_sim", "synthetic_human"}:
        raise ValueError("origin must be real_human, robot_sim, or synthetic_human")
    if camera_mode not in {"fixed", "egocentric", "unknown"}:
        raise ValueError("Unsupported camera mode")
    if not isinstance(resize_max, int) or not 64 <= resize_max <= 2048:
        raise ValueError("resize_max must be an integer within 64..2048")
    roi = validate_background_roi(background_roi, roi_provenance, camera_mode)
    video, output = Path(video).resolve(strict=True), Path(output).resolve()
    if (output/"quality.json").exists():
        raise ValueError("Refusing to overwrite quality evidence; select a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    media_hash = file_sha256(video)
    evidence, evidence_manifests = load_evidence(evidence_paths, media_hash)
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:frame=best_effort_timestamp_time",
        "-of", "json", str(video)], capture_output=True, text=True, timeout=300)
    metadata = json.loads(result.stdout) if result.returncode == 0 else {}
    streams = metadata.get("streams", [])
    stream = streams[0] if streams else {}
    timestamps = []
    for frame in metadata.get("frames", []):
        try:
            timestamps.append(float(frame["best_effort_timestamp_time"]))
        except (KeyError, ValueError):
            timestamps.append(None)
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(video),
                             "-map", "0:v:0", "-f", "null", "-"],
                            capture_output=True, text=True, timeout=300)
    cap = cv2.VideoCapture(str(video))
    rows, previous, duplicate_run, maximum_duplicate_run, expected_shape = [], None, 0, 0, None
    shape_changed = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if expected_shape is None:
            expected_shape = frame.shape
        elif expected_shape != frame.shape:
            shape_changed = True
        h, w = frame.shape[:2]
        scale = min(1., resize_max/max(h, w))
        gray = cv2.cvtColor(cv2.resize(frame, (round(w*scale), round(h*scale)),
                                      interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        index = len(rows)
        row = {
            "frame_index": index, "pts_seconds": timestamps[index] if index < len(timestamps) else None,
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "dark_fraction": float((gray <= DEFAULT_THRESHOLDS["dark_pixel_value"]).mean()),
            "bright_fraction": float((gray >= DEFAULT_THRESHOLDS["bright_pixel_value"]).mean()),
            "adjacent_mean_abs_difference": None, "exact_duplicate_previous": False,
            "background_motion_diagonal_fraction": None,
            "background_rotation_degrees": None, "background_track_inlier_fraction": None,
        }
        if previous is not None and previous.shape == gray.shape:
            row["adjacent_mean_abs_difference"] = float(cv2.absdiff(previous, gray).mean())
            row["exact_duplicate_previous"] = bool(np.array_equal(previous, gray))
            duplicate_run = duplicate_run + 1 if row["exact_duplicate_previous"] else 0
            maximum_duplicate_run = max(maximum_duplicate_run, duplicate_run)
            if roi:
                motion = _background_motion(previous, gray, roi)
                if motion:
                    row["background_motion_diagonal_fraction"] = motion["translation_diagonal_fraction"]
                    row["background_rotation_degrees"] = motion["rotation_degrees"]
                    row["background_track_inlier_fraction"] = motion["inlier_fraction"]
        rows.append(row)
        previous = gray
    cap.release()
    declared = stream.get("nb_frames")
    integrity = temporal_integrity(timestamps, len(rows),
                                   int(declared) if str(declared).isdigit() else None)
    if result.returncode or not streams:
        integrity["failures"].append("ffprobe_failure_or_no_video_stream")
    if decode.returncode:
        integrity["failures"].append("strict_full_decode_failed")
    if shape_changed:
        integrity["failures"].append("dimensions_changed")
    integrity.update(status="fail" if integrity["failures"] else "pass",
                     ffmpeg_strict_full_decode_passed=decode.returncode == 0)
    if decode.returncode:
        (output/"decode_error.txt").write_text(decode.stderr[-8000:])
    denominator = max(1, len(rows))
    transition_count = max(1, len(rows)-1)
    stats = {key: _stats([row[key] for row in rows]) for key in (
        "laplacian_variance", "dark_fraction", "bright_fraction",
        "adjacent_mean_abs_difference", "background_motion_diagonal_fraction",
        "background_rotation_degrees", "background_track_inlier_fraction")}
    blur_fraction = sum(row["laplacian_variance"] < DEFAULT_THRESHOLDS["low_laplacian_variance"]
                        for row in rows)/denominator
    dark_fraction = sum(row["dark_fraction"] > DEFAULT_THRESHOLDS["extreme_exposure_fraction"]
                        for row in rows)/denominator
    bright_fraction = sum(row["bright_fraction"] > DEFAULT_THRESHOLDS["extreme_exposure_fraction"]
                          for row in rows)/denominator
    low_global_change_fraction = sum(
        row["adjacent_mean_abs_difference"] is not None
        and row["adjacent_mean_abs_difference"] < DEFAULT_THRESHOLDS["low_global_change_mean_abs_difference"]
        for row in rows)/transition_count
    exact_duplicate_fraction = sum(row["exact_duplicate_previous"] for row in rows)/transition_count
    flags = []
    if blur_fraction > .5:
        flags.append("low_spatial_detail_or_blur_candidate_not_calibrated")
    if dark_fraction > .1 or bright_fraction > .1:
        flags.append("exposure_candidate_not_calibrated")
    if exact_duplicate_fraction > .5:
        flags.append("many_exactly_repeated_downscaled_frames_review_task_stage")
    motion = stats["background_motion_diagonal_fraction"]
    if roi and motion["p95"] is not None and (
            motion["p95"] > DEFAULT_THRESHOLDS["fixed_background_motion_diagonal_fraction"]):
        flags.append("declared_fixed_background_motion_candidate")
    dimensions = dimension_decisions(evidence, origin)
    vector = {
        "low_detail_frame_fraction_lower_better": blur_fraction,
        "extreme_dark_frame_fraction_lower_better": dark_fraction,
        "extreme_bright_frame_fraction_lower_better": bright_fraction,
        "low_global_change_transition_fraction_context_dependent": low_global_change_fraction,
        "exact_duplicate_transition_fraction_context_dependent": exact_duplicate_fraction,
        "laplacian_variance_median_content_dependent": stats["laplacian_variance"]["median"],
        "fixed_background_translation_p95_lower_better": motion["p95"],
    }
    report = {
        "schema": "r2h_quality_cascade_v1",
        "implementation": {"path": str(Path(__file__).resolve()),
                           "sha256": file_sha256(__file__)},
        "media": {"path": str(video), "sha256": media_hash, "origin": origin,
                  "width": stream.get("width"), "height": stream.get("height")},
        "media_integrity": integrity,
        "perceptual": {"status": "uncalibrated_ranking_only", "quality_vector": vector,
                       "metrics": stats, "maximum_exact_duplicate_transition_run": maximum_duplicate_run,
                       "metric_resolution_max_side": resize_max, "thresholds": DEFAULT_THRESHOLDS,
                       "thresholds_calibrated": False, "learned_quality_model": "not_run"},
        "camera": {"declared_mode": camera_mode, "background_roi": roi,
                   "roi_provenance": roi_provenance,
                   "measurement_status": "background_2d_proxy" if roi else "unknown",
                   "tracked_transition_fraction": motion["count"]/transition_count if roi else None,
                   "limitations": [
                       "No full-frame flow is interpreted as camera motion.",
                       "ROI must contain static background; this assumption is not automatically certified.",
                       "Ego camera movement is not rejected by the fixed-camera heuristic.",
                       "Background tracking loss remains unknown, not zero motion."]},
        "dimensions": dimensions, "evidence_manifests": evidence_manifests,
        "decision": aggregate_decision(integrity, dimensions, flags),
        "runtime_seconds": time.perf_counter()-started,
        "training_admitted": False,
        "limitations": [
            "No weighted overall score: semantic/physics failures cannot be compensated by appearance.",
            "Pixel sharpness, detail, exposure and repeated frames depend on scene content and task stage.",
            "Low global pixel change can be caused by a large static background; it is not treated as duplicate frames or a failure.",
            "Known failures are retained from source-bound evidence; this CPU module does not itself recognize teleportation.",
            "The quality of action labels, contact, scene reachability and task equivalence requires separate evidence.",
        ],
    }
    write_json(output/"quality.json", report)
    with (output/"frame_metrics.csv").open("w", newline="") as stream_out:
        writer = csv.DictWriter(stream_out, fieldnames=list(rows[0]) if rows else ["frame_index"])
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--origin", choices=("real_human", "robot_sim", "synthetic_human"), required=True)
    parser.add_argument("--camera-mode", choices=("fixed", "egocentric", "unknown"), default="unknown")
    parser.add_argument("--background-roi", nargs=4, type=float)
    parser.add_argument("--roi-provenance")
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = inspect_video(args.video, args.output, origin=args.origin,
                           camera_mode=args.camera_mode, background_roi=args.background_roi,
                           roi_provenance=args.roi_provenance, evidence_paths=args.evidence)
    print(json.dumps({"output": str(args.output), "decision": report["decision"],
                      "quality_vector": report["perceptual"]["quality_vector"]}, indent=2))


if __name__ == "__main__":
    main()
