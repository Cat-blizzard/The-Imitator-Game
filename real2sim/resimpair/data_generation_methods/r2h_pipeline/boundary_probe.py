"""Dense, reference-conditioned red-object *review proposals*, not physics QA.

This deliberately small probe supports a single, mostly visible red target in a
fixed camera view. A user-supplied anchor ROI establishes appearance and scale.
Red components are proposals only: skin, merged hand/object regions, competing
matches and missing detections can all produce ``unknown``. Nothing here detects
contact, establishes object identity, calibrates probability or admits training.

Run with ``python -m data_generation_methods.r2h_pipeline.boundary_probe --help``.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class ProbeConfig:
    hue_low: int = 11
    hue_high: int = 170
    saturation_min: int = 130
    value_min: int = 45
    min_component_pixels: int = 25
    min_area_ratio: float = .45
    max_area_ratio: float = 2.5
    min_fill_fraction: float = .42
    max_dimension_ratio: float = 1.85
    max_hue_delta: float = 4.
    min_value_ratio: float = .42
    max_chromaticity_distance: float = .24
    ambiguity_margin: float = .12
    step_review_diameters: float = 1.5
    boundary_seconds: float = .5
    context_seconds: float = .25
    max_gap_seconds_for_displacement: float = .5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_roi(roi) -> tuple[float, float, float, float]:
    if len(roi) != 4 or not all(math.isfinite(float(x)) for x in roi):
        raise ValueError("ROI must contain four finite normalized coordinates")
    x0, y0, x1, y1 = map(float, roi)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("ROI must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def red_candidates(frame: np.ndarray, config: ProbeConfig, search_roi=(0, 0, 1, 1)):
    """Keep each broad red component, including rejected/skin-like proposals."""
    x0, y0, x1, y1 = validate_roi(search_roi)
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = (((hue < config.hue_low) | (hue > config.hue_high))
            & (saturation > config.saturation_min) & (value > config.value_min))
    region = np.zeros(mask.shape, dtype=bool)
    region[int(y0 * height):math.ceil(y1 * height),
           int(x0 * width):math.ceil(x1 * width)] = True
    mask = (mask & region).astype(np.uint8)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for label in range(1, len(stats)):
        x, y, w, h, area = map(int, stats[label])
        if area < config.min_component_pixels:
            continue
        local = labels[y:y + h, x:x + w] == label
        pixels = frame[y:y + h, x:x + w][local][:, ::-1].astype(float)
        hsv_pixels = hsv[y:y + h, x:x + w][local].astype(float)
        # Wrap red hue around zero before taking a median.
        red_hue = ((hsv_pixels[:, 0] + 90) % 180) - 90
        median_rgb = np.median(pixels, axis=0)
        candidates.append({
            "candidate_id": len(candidates),
            "centroid_px": [float(v) for v in centroids[label]],
            "centroid_normalized": [float(centroids[label, 0] / width),
                                    float(centroids[label, 1] / height)],
            "bbox_xywh_px": [x, y, w, h],
            "red_area_px": area,
            "red_fill_fraction": area / (w * h),
            "median_red_hue": float(np.median(red_hue)),
            "median_saturation": float(np.median(hsv_pixels[:, 1])),
            "median_value": float(np.median(hsv_pixels[:, 2])),
            "median_rgb": median_rgb.tolist(),
            "chromaticity_rgb": (median_rgb / max(float(median_rgb.sum()), 1.)).tolist(),
        })
    return candidates


def build_reference(anchor, anchor_roi, config: ProbeConfig):
    """The supplied ROI must isolate exactly one sufficiently large red region."""
    candidates = red_candidates(anchor, config, anchor_roi)
    if len(candidates) != 1:
        raise ValueError("Anchor ROI must isolate exactly one red component; "
                         f"found {len(candidates)}. Supply a tighter, visually checked ROI.")
    reference = candidates[0]
    if reference["red_fill_fraction"] < config.min_fill_fraction:
        raise ValueError("Anchor component is too fragmented for this limited probe")
    reference["equivalent_diameter_px"] = 2 * math.sqrt(reference["red_area_px"] / math.pi)
    reference["selection"] = "explicit_user_or_operator_ROI; identity_not_automatically_verified"
    return reference


def score_candidates(candidates, reference, config: ProbeConfig):
    """Appearance/scale gates cannot certify target identity or exclude all skin."""
    result = []
    ref_w, ref_h = reference["bbox_xywh_px"][2:]
    for original in candidates:
        candidate = dict(original)
        area_ratio = candidate["red_area_px"] / reference["red_area_px"]
        width, height = candidate["bbox_xywh_px"][2:]
        dimension_ratio = max(width / ref_w, height / ref_h)
        hue_delta = abs(candidate["median_red_hue"] - reference["median_red_hue"])
        hue_delta = min(hue_delta, 180 - hue_delta)
        chroma_distance = float(np.linalg.norm(
            np.array(candidate["chromaticity_rgb"]) - reference["chromaticity_rgb"]))
        value_ratio = candidate["median_value"] / max(reference["median_value"], 1)
        reasons = []
        if not config.min_area_ratio <= area_ratio <= config.max_area_ratio:
            reasons.append("red_area_inconsistent_with_anchor")
        if candidate["red_fill_fraction"] < config.min_fill_fraction:
            reasons.append("fragmented_or_merged_component")
        if dimension_ratio > config.max_dimension_ratio:
            reasons.append("component_extent_inconsistent_with_anchor")
        if hue_delta > config.max_hue_delta:
            reasons.append("hue_inconsistent_with_anchor")
        if value_ratio < config.min_value_ratio:
            reasons.append("too_dark_relative_to_anchor")
        if chroma_distance > config.max_chromaticity_distance:
            reasons.append("chromaticity_inconsistent_with_anchor")
        # Dimensionless ranking, explicitly not a calibrated probability.
        cost = (.30 * abs(math.log(max(area_ratio, 1e-8)))
                + .25 * hue_delta / max(config.max_hue_delta, 1e-8)
                + .30 * chroma_distance / max(config.max_chromaticity_distance, 1e-8)
                + .15 * abs(candidate["red_fill_fraction"] - reference["red_fill_fraction"]))
        candidate.update({
            "area_ratio": area_ratio,
            "dimension_ratio": dimension_ratio,
            "hue_delta": hue_delta,
            "chromaticity_distance": chroma_distance,
            "value_ratio": value_ratio,
            "appearance_cost": cost,
            "appearance_eligible": not reasons,
            "rejection_reasons": reasons,
        })
        result.append(candidate)
    return result


def select_candidate(candidates, config: ProbeConfig):
    eligible = sorted((c for c in candidates if c["appearance_eligible"]),
                      key=lambda c: c["appearance_cost"])
    if not eligible:
        return None, "no_appearance_consistent_candidate", None
    margin = eligible[1]["appearance_cost"] - eligible[0]["appearance_cost"] if len(eligible) > 1 else None
    if margin is not None and margin < config.ambiguity_margin:
        return None, "multiple_similar_appearance_candidates", margin
    return eligible[0], "appearance_candidate", margin


def annotate_temporal(rows, reference, config: ProbeConfig):
    """Do not interpolate unknown frames or connect long missing spans."""
    last = None
    events = []
    diameter = reference["equivalent_diameter_px"]
    for row in rows:
        selected = next((c for c in row["candidates"]
                         if c["candidate_id"] == row["selected_candidate_id"]), None)
        row["step_from_last_observed"] = None
        row["contact_state"] = "unknown_not_measured"
        row["physical_validity"] = "unknown_not_measured"
        row["identity_status"] = "unverified_appearance_match" if selected else "unknown"
        if selected is None:
            row["tracking_state"] = "unknown"
            row["tracking_uncertainty"] = "high"
            row["trajectory_px"] = None
            continue
        row["trajectory_px"] = selected["centroid_px"]
        row["tracking_state"] = "appearance_candidate"
        row["tracking_uncertainty"] = "medium"
        if last is not None:
            gap_frames = row["frame_index"] - last["frame_index"] - 1
            elapsed = row["timestamp_seconds"] - last["timestamp_seconds"]
            distance = math.dist(row["trajectory_px"], last["trajectory_px"])
            row["step_from_last_observed"] = {
                "from_frame": last["frame_index"], "missing_frames": gap_frames,
                "elapsed_seconds": elapsed, "distance_px": distance,
                "distance_anchor_diameters": distance / diameter,
                "image_speed_px_per_second": distance / elapsed if elapsed > 0 else None,
                "physical_speed": "not_available_without_geometry",
            }
            if gap_frames:
                row["tracking_state"] = "reacquired_appearance_candidate"
                row["tracking_uncertainty"] = "high"
            if elapsed <= config.max_gap_seconds_for_displacement and distance / diameter > config.step_review_diameters:
                row["tracking_state"] = "discontinuity_review_candidate"
                row["tracking_uncertainty"] = "high"
                events.append({
                    "kind": "appearance_candidate_displacement",
                    "priority": 1, "start_frame": last["frame_index"],
                    "end_frame": row["frame_index"],
                    "distance_px": distance, "distance_anchor_diameters": distance / diameter,
                    "missing_frames": gap_frames,
                    "interpretation": "Review relocation, occlusion, identity switch, edit/camera change; "
                                      "this is not a teleportation or no-contact verdict.",
                })
        last = row
    start = None
    for index in range(len(rows) + 1):
        unknown = index < len(rows) and rows[index]["trajectory_px"] is None
        if unknown and start is None:
            start = index
        if not unknown and start is not None:
            events.append({
                "kind": "tracking_unknown_interval", "priority": 2,
                "start_frame": start, "end_frame": index - 1,
                "interpretation": "Missing, fragmented, merged or ambiguous color evidence; "
                                  "object absence/motion/contact is unknown.",
            })
            start = None
    return events


def review_windows(rows, events, config: ProbeConfig):
    times = np.array([r["timestamp_seconds"] for r in rows])
    proposals = list(events)
    for kind, low, high in (
        ("start_boundary_dense_review", 0., config.boundary_seconds),
        ("end_boundary_dense_review", max(0., times[-1] - config.boundary_seconds), times[-1]),
    ):
        indices = np.flatnonzero((times >= low) & (times <= high))
        proposals.append({
            "kind": kind, "priority": 3,
            "start_frame": int(indices[0]), "end_frame": int(indices[-1]),
            "interpretation": "Always review the sampled start/end boundary, even without a detector alert.",
        })
    windows = []
    for order, event in enumerate(sorted(proposals, key=lambda e: (e["priority"], e["start_frame"]))):
        start, end = event["start_frame"], event["end_frame"]
        indices = np.flatnonzero(
            (times >= times[start] - config.context_seconds - 1e-9)
            & (times <= times[end] + config.context_seconds + 1e-9))
        windows.append({
            "window_id": f"review_{order + 1:03d}", **event,
            "evidence_start_seconds": float(times[start]),
            "evidence_end_seconds": float(times[end]),
            "review_start_seconds": float(times[indices[0]]),
            "review_end_seconds": float(times[indices[-1]]),
            "review_frame_indices": indices.tolist(),
            "reviewer_decision": "pending", "contact_state": "unknown_not_measured",
        })
    return windows


def _draw_tile(frame, row, width=320):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail((width, width))
    tile = Image.new("RGB", (width, image.height + 48), "white")
    tile.paste(image, (0, 48))
    draw = ImageDraw.Draw(tile)
    draw.text((5, 3), f"Frame {row['frame_index']} | {row['timestamp_seconds']:.4f}s", fill="black")
    draw.text((5, 19), row["tracking_state"], fill="black")
    draw.text((5, 33), "Appearance only; contact unknown", fill="black")
    sx, sy = image.width / frame.shape[1], image.height / frame.shape[0]
    for candidate in row["candidates"]:
        x, y, w, h = candidate["bbox_xywh_px"]
        selected = candidate["candidate_id"] == row["selected_candidate_id"]
        color = "#008800" if selected else "#cc6600"
        draw.rectangle((x * sx, y * sy + 48, (x + w) * sx, (y + h) * sy + 48),
                       outline=color, width=2 if selected else 1)
    return tile


def _sheet(tiles, path, columns=4):
    width, height = tiles[0].size
    canvas = Image.new("RGB", (columns * width, math.ceil(len(tiles) / columns) * height), "#dddddd")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % columns) * width, (i // columns) * height))
    canvas.save(path)


def export_review_pack(video, output, rows, windows):
    selected = {i for w in windows for i in w["review_frame_indices"]}
    selected.update([0, len(rows) - 1])
    frames_dir, sheets_dir = output / "keyframes", output / "contact_sheets"
    frames_dir.mkdir(exist_ok=True)
    sheets_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    tiles = {}
    try:
        for index in range(len(rows)):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video ended unexpectedly during evidence export at frame {index}")
            if index in selected:
                path = frames_dir / f"frame_{index:05d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"Could not write {path}")
                rows[index]["evidence_frame"] = str(path.relative_to(output))
                tiles[index] = _draw_tile(frame, rows[index])
    finally:
        cap.release()
    for window in windows:
        window["contact_sheets"] = []
        indices = window["review_frame_indices"]
        for page, offset in enumerate(range(0, len(indices), 16), start=1):
            path = sheets_dir / f"{window['window_id']}_page_{page:02d}.jpg"
            _sheet([tiles[i] for i in indices[offset:offset + 16]], path)
            window["contact_sheets"].append(str(path.relative_to(output)))
    summary_indices = sorted({0, len(rows) - 1, *(w["start_frame"] for w in windows),
                              *(w["end_frame"] for w in windows)})
    _sheet([tiles[i] for i in summary_indices], output / "overview_contact_sheet.jpg")
    return len(selected)


def export_trajectory_plot(rows, output, frame_width):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = [r["timestamp_seconds"] for r in rows]
    x = [r["trajectory_px"][0] / frame_width if r["trajectory_px"] else np.nan for r in rows]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    # Scatter avoids implying observed travel across either gaps or jump proposals.
    ax.scatter(time, x, s=10, color="#236a42", label="Appearance-consistent candidate")
    unknown = [r["timestamp_seconds"] for r in rows if r["trajectory_px"] is None]
    ax.scatter(unknown, [-.04] * len(unknown), s=14, marker="x", color="#b45a16",
               label="Unknown (no target trajectory)")
    ax.set(xlabel="Video timestamp (seconds)", ylabel="Image x / width", ylim=(-.08, 1),
           title="Dense red-object review probe — unverified identity; contact unknown")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "candidate_trajectory.png", dpi=170)
    plt.close(fig)


def run_probe(video, anchor_path, anchor_roi, output, *, search_roi=(0, 0, 1, 1),
              config=None, prior_diagnostic=None):
    config = config or ProbeConfig()
    validate_roi(anchor_roi)
    validate_roi(search_roi)
    video, anchor_path = Path(video).resolve(strict=True), Path(anchor_path).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Use a new output directory; existing completed evidence is immutable")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Could not open input video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Input video has no usable frame rate")
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Input video contains no decodable frames")
    height, width = first.shape[:2]
    anchor = cv2.imread(str(anchor_path))
    if anchor is None:
        cap.release()
        raise ValueError("Could not decode anchor image")
    anchor_original_dimensions = [int(anchor.shape[1]), int(anchor.shape[0])]
    if abs(anchor.shape[1] / anchor.shape[0] - width / height) > .01:
        cap.release()
        raise ValueError("Anchor/video aspect ratios differ; align them explicitly first")
    anchor = cv2.resize(anchor, (width, height), interpolation=cv2.INTER_AREA)
    try:
        reference = build_reference(anchor, anchor_roi, config)
    except Exception:
        cap.release()
        raise
    rows, frame, index, previous_time = [], first, 0, -1.
    try:
        while True:
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not math.isfinite(timestamp) or timestamp <= previous_time:
                if index == 0 and timestamp == 0:
                    pass
                else:
                    raise ValueError("Decoder timestamps are non-monotonic; refusing synthetic timing")
            candidates = score_candidates(red_candidates(frame, config, search_roi), reference, config)
            selected, reason, margin = select_candidate(candidates, config)
            rows.append({
                "frame_index": index, "timestamp_seconds": timestamp,
                "selected_candidate_id": selected["candidate_id"] if selected else None,
                "selection_reason": reason, "appearance_rank_margin": margin,
                "candidates": candidates,
            })
            previous_time = timestamp
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
    finally:
        cap.release()
    events = annotate_temporal(rows, reference, config)
    windows = review_windows(rows, events, config)
    exported_count = export_review_pack(video, output, rows, windows)
    export_trajectory_plot(rows, output, width)
    prior = None
    if prior_diagnostic:
        prior_path = Path(prior_diagnostic).resolve(strict=True)
        raw = json.loads(prior_path.read_text())
        legacy_rows = {r["frame"]: r for r in raw["rows"]}
        disagreements = []
        for row in rows:
            legacy = legacy_rows.get(row["frame_index"])
            if legacy and legacy.get("x_px") is not None and row["trajectory_px"] is None:
                disagreements.append({
                    "frame_index": row["frame_index"],
                    "legacy_centroid_px": [legacy["x_px"], legacy["y_px"]],
                    "new_state": "unknown", "reason": row["selection_reason"],
                })
        prior = {"path": str(prior_path), "sha256": sha256(prior_path),
                 "method": raw.get("method"), "role": "comparison_only_not_ground_truth",
                 "legacy_detection_now_unknown": disagreements}
    report = {
        "schema": "r2h_dense_boundary_probe_v1",
        "purpose": "Prioritize dense video evidence for human/VLM review",
        "scope": "One red target, explicit anchor ROI, comparable scale, approximately fixed camera; "
                 "visibility loss, merged hand/object regions and competing matches remain unknown.",
        "source": {"path": str(video), "sha256": sha256(video), "fps_reported": fps,
                   "decoded_frames": len(rows), "width": width, "height": height,
                   "first_timestamp_seconds": rows[0]["timestamp_seconds"],
                   "last_timestamp_seconds": rows[-1]["timestamp_seconds"],
                   "timing": "OpenCV decoder presentation timestamps; no subsampling/interpolation"},
        "anchor": {"path": str(anchor_path), "sha256": sha256(anchor_path),
                   "original_dimensions": anchor_original_dimensions,
                   "resized_to_video_dimensions": [width, height],
                   "operator_selected_roi_normalized": list(anchor_roi), "reference": reference},
        "search_roi_normalized": list(search_roi), "config": asdict(config),
        "method": [
            "Decode every frame; generate broad HSV connected-component proposals.",
            "Gate area, occupied bounding-box fraction, component extent, wrapped hue, "
            "relative brightness and RGB chromaticity against the explicit anchor.",
            "Rank eligible proposals by appearance. Close competitors and failed gates produce unknown.",
            "Keep locations independent of temporal association so discontinuities are not smoothed away.",
            "Propose review of large image displacements only across short gaps; never interpolate unknowns.",
            "Export all frames around start/end boundaries, displacement proposals and unknown intervals.",
        ],
        "score_semantics": "appearance_cost is a heuristic ranking, not calibrated confidence; "
                           "ordinal uncertainty is conservative and not an error probability",
        "limitations": [
            "Color/appearance consistency does not establish semantic identity; skin or other red regions can still match.",
            "Hard thresholds are exploratory choices for this clip family; no held-out calibration, recall or precision claim.",
            "No hand segmentation, contact detector, depth, 3D trajectory or physical plausibility verification.",
            "Unknown is not evidence of absence, teleportation, correct motion or invalid motion.",
            "Camera changes, perspective/lighting changes, object rotations and occlusion can cause false alerts or missed targets.",
            "A clear-looking final frame and no alert cannot establish correct event order or task completion.",
        ],
        "summary": {
            "frames_with_appearance_candidate": sum(r["trajectory_px"] is not None for r in rows),
            "unknown_frames": sum(r["trajectory_px"] is None for r in rows),
            "displacement_review_candidates": sum(e["kind"] == "appearance_candidate_displacement" for e in events),
            "unknown_intervals": sum(e["kind"] == "tracking_unknown_interval" for e in events),
            "review_windows": len(windows), "exported_lossless_keyframes": exported_count,
        },
        "review_questions": [
            "Is the marked region the intended object, a hand/skin fragment, or unresolved?",
            "Across the full dense boundary window, does the object remain visible, disappear, reappear or change identity?",
            "Is physical contact actually visible before motion? Mark unknown if contact is hidden.",
            "Are initial layout, ordered manipulations and final layout consistent with the separate task contract?",
        ],
        "events": events, "prior_color_diagnostic": prior,
        "artifacts": {"trajectory": "trajectory.csv", "candidates": "frame_candidates.jsonl",
                      "windows": "review_windows.json", "overview": "overview_contact_sheet.jpg",
                      "trajectory_plot": "candidate_trajectory.png"},
        "contact_state": "unknown_not_measured", "physical_validity": "unknown_not_measured",
        "review_status": "pending_human_or_vlm", "training_admitted": False,
    }
    with (output / "frame_candidates.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (output / "trajectory.csv").open("w", newline="") as stream:
        fieldnames = ["frame_index", "timestamp_seconds", "tracking_state", "tracking_uncertainty",
                      "x_px", "y_px", "candidate_id", "selection_reason", "contact_state"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            xy = row["trajectory_px"] or [None, None]
            writer.writerow({
                **{k: row[k] for k in ("frame_index", "timestamp_seconds", "tracking_state",
                                      "tracking_uncertainty", "selection_reason", "contact_state")},
                "x_px": xy[0], "y_px": xy[1], "candidate_id": row["selected_candidate_id"],
            })
    write_json(output / "review_windows.json", windows)
    write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--anchor-roi", nargs=4, required=True, type=float, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--search-roi", nargs=4, default=(0, 0, 1, 1), type=float)
    parser.add_argument("--prior-diagnostic", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_probe(args.video, args.anchor, args.anchor_roi, args.output_dir,
                       search_roi=args.search_roi, prior_diagnostic=args.prior_diagnostic)
    print(json.dumps({"report": str(args.output_dir / "report.json"), **report["summary"],
                      "training_admitted": False}, indent=2))


if __name__ == "__main__":
    main()
