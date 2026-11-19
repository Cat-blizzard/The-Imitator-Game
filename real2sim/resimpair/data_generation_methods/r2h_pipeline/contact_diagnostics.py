"""Optional fixed-camera 2D interaction diagnostic, not contact ground truth.

Uses manually declared handle/background ROIs, forward-backward KLT and an
independent MediaPipe hand inventory. Lost tracks remain unknown; no point
reinitialization silently transfers the handle identity to an occluding hand.
Run with the isolated MediaPipe environment. No downloads or paid calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time


def validate_roi(roi: list[float]) -> list[float]:
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in roi)):
        raise ValueError("ROI must be four normalized numbers")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI must have ordered corners within [0, 1]")
    return list(map(float, roi))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def forward_backward_valid(previous, forward, backward, forward_status,
                           backward_status, *, width, height, max_error_px=1.5):
    """Geometric consistency mask, not semantic object-identity confidence."""
    import numpy as np
    p, q, back = [np.asarray(x).reshape(-1, 2) for x in (previous, forward, backward)]
    error = np.linalg.norm(back-p, axis=1)
    valid = (
        np.asarray(forward_status).reshape(-1).astype(bool)
        & np.asarray(backward_status).reshape(-1).astype(bool)
        & np.isfinite(q).all(axis=1) & np.isfinite(back).all(axis=1)
        & (error <= max_error_px)
        & (q[:, 0] >= 0) & (q[:, 0] < width)
        & (q[:, 1] >= 0) & (q[:, 1] < height)
    )
    return valid, error


def corrected_displacement(handle_current, handle_initial,
                           background_current, background_initial, *,
                           min_handle_points=3, min_background_points=6):
    import numpy as np
    if (len(handle_current) < min_handle_points
            or len(background_current) < min_background_points):
        return None
    handle = np.median(np.asarray(handle_current)-np.asarray(handle_initial), axis=0)
    background = np.median(np.asarray(background_current)-np.asarray(background_initial), axis=0)
    return (handle-background).tolist()


def _intervals(rows, field):
    intervals, active = [], None
    for row in rows:
        if row[field] is True:
            if active is None:
                active = {"first_frame": row["frame_index"],
                          "start_seconds": row["timestamp_seconds"]}
            active.update(last_frame=row["frame_index"],
                          end_seconds=row["timestamp_seconds"])
        elif active is not None:
            intervals.append(active)
            active = None
    if active is not None:
        intervals.append(active)
    return intervals


def diagnose(video_path: Path, model_path: Path, output: Path, *,
             handle_roi, background_roi, roi_provenance: str,
             near_diagonal_fraction=0.025, min_handle_points=3,
             min_background_points=6, max_fb_error_px=1.5):
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    handle_roi, background_roi = validate_roi(handle_roi), validate_roi(background_roi)
    if not roi_provenance.strip():
        raise ValueError("explicit ROI provenance required")
    video_path, model_path, output = [Path(p).resolve() for p in (video_path, model_path, output)]
    if (output/"diagnostic.json").exists():
        raise ValueError("refusing to overwrite evidence")
    output.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "json", str(video_path)
    ], check=True, capture_output=True, text=True)
    timestamps = [float(f["best_effort_timestamp_time"])
                  for f in json.loads(raw.stdout)["frames"]]
    if not timestamps or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("missing or non-increasing source timestamps")
    timestamps_ms = [round(t*1000) for t in timestamps]
    if any(b <= a for a, b in zip(timestamps_ms, timestamps_ms[1:])):
        raise ValueError("timestamps too close for MediaPipe VIDEO milliseconds")
    cap = cv2.VideoCapture(str(video_path))
    ok, first = cap.read()
    if not ok:
        raise ValueError("video cannot be decoded")
    height, width = first.shape[:2]
    diagonal = float(np.hypot(width, height))
    near_px = diagonal*near_diagonal_fraction
    motion_px_per_frame = diagonal*0.001
    gray_prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    def corners(roi):
        mask = np.zeros_like(gray_prev)
        x0, y0, x1, y1 = roi
        mask[round(y0*height):round(y1*height), round(x0*width):round(x1*width)] = 255
        points = cv2.goodFeaturesToTrack(gray_prev, maxCorners=40, qualityLevel=0.01,
                                        minDistance=3, mask=mask, blockSize=5)
        return (points.reshape(-1, 2) if points is not None
                else np.empty((0, 2), dtype=np.float32))

    handle, background = corners(handle_roi), corners(background_roi)
    handle_initial, background_initial = handle.copy(), background.copy()
    initial_counts = {"handle": len(handle), "background": len(background)}
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_hands=6,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.5)
    rows, hand_evidence, snapshots = [], [], []
    selected = {round(i*(len(timestamps)-1)/5) for i in range(6)}
    previous_delta = None
    started = time.perf_counter()
    hand_seconds = 0.0

    def track(current_gray, points, initial_points):
        if not len(points):
            return points, initial_points, None
        p = points.astype(np.float32).reshape(-1, 1, 2)
        q, st, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, current_gray, p, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if q is None:
            return points[:0], initial_points[:0], None
        back, bst, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, gray_prev, q, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if back is None:
            return points[:0], initial_points[:0], None
        valid, errors = forward_backward_valid(
            p, q, back, st, bst, width=width, height=height,
            max_error_px=max_fb_error_px)
        return q.reshape(-1, 2)[valid], initial_points[valid], (
            float(np.median(errors[valid])) if valid.any() else None)

    with vision.HandLandmarker.create_from_options(options) as hand_model:
        frame, index = first, 0
        while True:
            if index >= len(timestamps):
                raise ValueError("decoder emitted more frames than ffprobe")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hfb = bfb = 0.0
            if index:
                handle, handle_initial, hfb = track(gray, handle, handle_initial)
                background, background_initial, bfb = track(gray, background, background_initial)
            delta = corrected_displacement(
                handle, handle_initial, background, background_initial,
                min_handle_points=min_handle_points,
                min_background_points=min_background_points)
            status = "tracked_2d_proxy" if delta is not None else "unknown"
            hand_started = time.perf_counter()
            hand_result = hand_model.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                timestamps_ms[index])
            hand_seconds += time.perf_counter()-hand_started
            hands = [
                [[float(p.x*width), float(p.y*height)] for p in landmarks]
                for landmarks in hand_result.hand_landmarks
            ]
            nearest = None
            if delta is not None and hands:
                hp = np.asarray(hands).reshape(-1, 2)
                nearest = float(np.linalg.norm(hp[:, None, :]-handle[None, :, :], axis=2).min())
            step = None
            if delta is not None and previous_delta is not None:
                step = float(np.linalg.norm(np.asarray(delta)-previous_delta))
            previous_delta = None if delta is None else np.asarray(delta)
            row = {
                "frame_index": index, "timestamp_seconds": timestamps[index],
                "status": status, "handle_valid_points": len(handle),
                "background_valid_points": len(background),
                "handle_fb_error_median_px": hfb, "background_fb_error_median_px": bfb,
                "handle_corrected_dx_px": None if delta is None else delta[0],
                "handle_corrected_dy_px": None if delta is None else delta[1],
                "handle_corrected_displacement_px": None if delta is None else float(np.linalg.norm(delta)),
                "handle_step_px": step,
                "handle_moving_visual_proxy": None if step is None else step > motion_px_per_frame,
                "detected_hand_count": len(hands),
                "nearest_hand_keypoint_to_handle_track_px": nearest,
                "nearest_distance_normalized_diagonal": None if nearest is None else nearest/diagonal,
                "hand_near_visual_proxy": None if nearest is None else nearest <= near_px,
                "physical_contact": "unknown",
            }
            rows.append(row)
            hand_evidence.append({"frame_index": index, "timestamp_seconds": timestamps[index],
                                  "hand_landmarks_pixels": hands,
                                  "handle_track_points_pixels": handle.tolist(),
                                  "background_track_points_pixels": background.tolist()})
            if index in selected:
                diagnostic = frame.copy()
                for roi, label, color in [(handle_roi, "initial handle ROI", (0, 160, 255)),
                                          (background_roi, "fixed cabinet ROI", (255, 180, 0))]:
                    x0, y0, x1, y1 = roi
                    cv2.rectangle(diagnostic, (round(x0*width), round(y0*height)),
                                  (round(x1*width), round(y1*height)), color, 2)
                for points, color in [(handle, (0, 180, 255)), (background, (255, 200, 0))]:
                    for x, y in points:
                        cv2.circle(diagnostic, (round(x), round(y)), 3, color, -1)
                for hand in hands:
                    for x, y in hand:
                        cv2.circle(diagnostic, (round(x), round(y)), 2, (0, 255, 0), -1)
                label = f"t={timestamps[index]:.2f}s  H={len(handle)} BG={len(background)} {status}"
                cv2.putText(diagnostic, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                path = output/f"diagnostic_frame{index:04d}.jpg"
                cv2.imwrite(str(path), diagnostic)
                snapshots.append({"frame_index": index, "path": str(path)})
            gray_prev = gray
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
    cap.release()
    if len(rows) != len(timestamps):
        raise ValueError("decoded frame count differs from ffprobe")
    result = {
        "schema": "fixed_camera_interaction_diagnostic_v1",
        "video": {"path": str(video_path), "sha256": _sha256(video_path),
                  "width": width, "height": height, "frames": len(rows)},
        "model": {"path": str(model_path), "sha256": _sha256(model_path),
                  "mediapipe_version": mp.__version__, "num_hands": 6},
        "roi_provenance": roi_provenance,
        "handle_roi_normalized_xyxy": handle_roi,
        "background_roi_normalized_xyxy": background_roi,
        "parameters": {"max_fb_error_px": max_fb_error_px,
                       "min_handle_points": min_handle_points,
                       "min_background_points": min_background_points,
                       "hand_near_threshold_px": near_px,
                       "motion_threshold_px_per_frame": motion_px_per_frame,
                       "background_model": "translation_only_median_displacement",
                       "point_reinitialization": False},
        "initial_feature_counts": initial_counts,
        "summary": {
            "frames_with_track_support": sum(r["status"] == "tracked_2d_proxy" for r in rows),
            "frames_with_unknown_tracking": sum(r["status"] == "unknown" for r in rows),
            "frames_with_no_hand_detection": sum(r["detected_hand_count"] == 0 for r in rows),
            "frames_with_unknown_nearness": sum(r["hand_near_visual_proxy"] is None for r in rows),
            "hand_near_intervals_visual_only": _intervals(rows, "hand_near_visual_proxy"),
            "handle_motion_intervals_visual_only": _intervals(rows, "handle_moving_visual_proxy"),
        },
        "timing_seconds": {"loop_including_hand_model_initialization": time.perf_counter()-started,
                           "hand_inference_total": hand_seconds},
        "physical_contact_ground_truth": False,
        "quality_release": False, "training_admitted": False,
        "limitations": [
            "ROIs are manually declared visual regions, not simulator-projected geometry.",
            "KLT tracks local appearance; forward-backward consistency cannot prove persistent object identity.",
            "Occluding hands may capture tracks even when the forward-backward test passes.",
            "Translation-only background compensation is unsuitable for large camera rotation or parallax.",
            "Lost points are not reinitialized; insufficient support becomes unknown, not stationary.",
            "MediaPipe misses overlapped or generated hands; absent detections are unknown.",
            "2D nearness and motion timing are visual evidence only, not 3D contact, force or successful grasp.",
            "Thresholds are diagnostic settings for this pilot, not calibrated dataset acceptance criteria.",
        ],
        "snapshots": snapshots,
        "rows": rows,
    }
    (output/"diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/"point_evidence.json").write_text(json.dumps(hand_evidence, allow_nan=False)+"\n")
    with (output/"timeline.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_figures(result, output)
    return result


def render_figures(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = result["rows"]
    times = [r["timestamp_seconds"] for r in rows]
    values = lambda key: [np.nan if r[key] is None else r[key] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(times, values("handle_corrected_displacement_px"), color="black")
    axes[0].set_ylabel("Handle displacement (px)")
    axes[0].set_title("2D interaction diagnostic; missing segments = unknown; not contact GT")
    axes[1].plot(times, values("nearest_hand_keypoint_to_handle_track_px"), color="black")
    axes[1].axhline(result["parameters"]["hand_near_threshold_px"], color="gray", linestyle="--",
                    label="pilot nearness threshold")
    axes[1].set_ylabel("Nearest hand point (px)")
    axes[1].legend()
    for key, label in [("handle_valid_points", "handle points"),
                       ("background_valid_points", "background points"),
                       ("detected_hand_count", "detected hands")]:
        axes[2].plot(times, values(key), label=label)
    axes[2].legend()
    axes[2].set_ylabel("Count")
    axes[2].set_xlabel("Actual video timestamp (seconds)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output/"timeline.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, snapshot in zip(axes.flat, result["snapshots"]):
        ax.imshow(plt.imread(snapshot["path"]))
        row = rows[snapshot["frame_index"]]
        ax.set_title(f"{row['timestamp_seconds']:.2f} s; {row['status']}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output/"diagnostic_contact_sheet.jpg", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--handle-roi", type=float, nargs=4, required=True)
    parser.add_argument("--background-roi", type=float, nargs=4, required=True)
    parser.add_argument("--roi-provenance", required=True)
    args = parser.parse_args()
    result = diagnose(args.video, args.model, args.output,
                      handle_roi=args.handle_roi, background_roi=args.background_roi,
                      roi_provenance=args.roi_provenance)
    print(json.dumps({"summary": result["summary"], "timing_seconds": result["timing_seconds"]}))


if __name__ == "__main__":
    main()
