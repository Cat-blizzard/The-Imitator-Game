"""Select a bounded, auditable frame packet from a completed boundary probe.

Only metadata is read/written. No VLM call, frame extraction, confidence
calibration, review verdict or training admission is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _unknown(row):
    return row.get("trajectory_px") is None or row.get("tracking_state") == "unknown"


def _representatives(event, rows):
    start, end = event["start_frame"], event["end_frame"]
    midpoint = (rows[start]["timestamp_seconds"] + rows[end]["timestamp_seconds"]) / 2
    representatives = [start] if start == end else [start, end]
    if event["kind"] == "appearance_candidate_displacement":
        interior = [i for i in range(start + 1, end) if _unknown(rows[i])]
    elif event["kind"] == "tracking_unknown_interval":
        interior = list(range(start + 1, end))
    else:
        interior = []
    if interior:
        representatives.append(min(interior, key=lambda i: (
            abs(rows[i]["timestamp_seconds"] - midpoint), i)))
    return representatives


def select_review_frames(report, rows, budget):
    """First/last + primary displacement evidence, then farthest-time coverage.

    A packet can represent an event without densely covering it. Those are
    separately reported, and neither implies that any event has been validated.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if not rows:
        raise ValueError("No frame metadata")
    if len(rows) != report["source"]["decoded_frames"]:
        raise ValueError("Report/frame count mismatch")
    if [r["frame_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Expected contiguous, ordered frame indices starting at zero")
    times = [r["timestamp_seconds"] for r in rows]
    if (not all(math.isfinite(t) for t in times)
            or any(right <= left for left, right in zip(times, times[1:]))):
        raise ValueError("Expected finite, strictly increasing frame timestamps")
    events = []
    for index, original in enumerate(report.get("events", [])):
        event = dict(original)
        start, end = event["start_frame"], event["end_frame"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(rows)):
            raise ValueError("Event range lies outside frame metadata")
        event["event_id"] = f"event_{index + 1:03d}"
        events.append(event)

    displacement_events = sorted(
        (e for e in events if e["kind"] == "appearance_candidate_displacement"),
        key=lambda e: (e["priority"], e["start_frame"], e["end_frame"], e["event_id"]))
    primary = displacement_events[0] if displacement_events else None
    selected, reasons = set(), {}
    mandatory = {}

    def require(index, reason):
        mandatory.setdefault(index, []).append(reason)

    require(0, "video_first_frame")
    require(len(rows) - 1, "video_last_frame")
    if primary:
        reps = _representatives(primary, rows)
        require(primary["start_frame"], f"{primary['event_id']}:primary_displacement_start")
        require(primary["end_frame"], f"{primary['event_id']}:primary_displacement_end")
        for index in reps:
            if index not in (primary["start_frame"], primary["end_frame"]):
                require(index, f"{primary['event_id']}:unknown_midpoint_inside_displacement")
    # Dict insertion order guarantees the two video boundaries before event
    # endpoints, then an unknown midpoint, when a budget is too small for all.
    for index, frame_reasons in mandatory.items():
        if len(selected) < budget:
            selected.add(index)
            reasons[index] = frame_reasons.copy()
    global_steps = []
    while len(selected) < min(budget, len(rows)):
        candidates = [i for i in range(len(rows)) if i not in selected]
        distances = {i: min(abs(times[i] - times[j]) for j in selected)
                     for i in candidates}
        best_distance = max(distances.values())
        # Numerical noise in decoder timestamps must not choose random tie sides.
        index = min(i for i in candidates if abs(distances[i] - best_distance) <= 1e-9)
        selected.add(index)
        reasons[index] = ["global_temporal_coverage:maximize_distance_to_selected_times"]
        global_steps.append({"frame_index": index,
                             "distance_to_previous_packet_seconds": distances[index]})

    coverage = []
    for event in events:
        interval = set(range(event["start_frame"], event["end_frame"] + 1))
        representatives = set(_representatives(event, rows))
        observed = sorted(interval & selected)
        missing = sorted(representatives - selected)
        if not observed:
            status = "unrepresented"
        elif missing:
            status = "partial_representatives"
        else:
            status = "representatives_selected"
        coverage.append({
            "event_id": event["event_id"], "kind": event["kind"], "priority": event["priority"],
            "start_frame": event["start_frame"], "end_frame": event["end_frame"],
            "start_seconds": times[event["start_frame"]], "end_seconds": times[event["end_frame"]],
            "representative_frame_indices": sorted(representatives),
            "selected_frame_indices_in_interval": observed,
            "missing_representative_frame_indices": missing,
            "sampling_status": status,
            "dense_interval_fully_sampled": interval <= selected,
            "unselected_frame_count_in_interval": len(interval - selected),
            "verification_status": "not_reviewed_unknown",
        })
    mandatory_missing = sorted(set(mandatory) - selected)
    missing_event_representatives = [c["event_id"] for c in coverage
                                     if c["missing_representative_frame_indices"]]
    insufficient_primary = bool(mandatory_missing)
    if insufficient_primary:
        status = "insufficient_for_required_boundaries_and_primary_event"
    elif missing_event_representatives:
        status = "required_frames_selected_other_event_representatives_uncovered"
    else:
        status = "required_and_event_representatives_selected_review_pending"
    selected_rows = []
    for index in sorted(selected):
        row = rows[index]
        selected_rows.append({
            "frame_index": index, "timestamp_seconds": times[index],
            "selection_reasons": reasons[index],
            "event_ids_containing_frame": [c["event_id"] for c in coverage
                                           if index in c["selected_frame_indices_in_interval"]],
            "tracking_state": row.get("tracking_state", "unknown"),
            "tracking_uncertainty": row.get("tracking_uncertainty", "unknown"),
            "trajectory_px": row.get("trajectory_px"),
            "evidence_frame_from_probe": row.get("evidence_frame"),
            "requires_source_frame_extraction": not bool(row.get("evidence_frame")),
            "contact_state": "unknown_not_measured", "verification_status": "not_reviewed_unknown",
        })
    return {
        "schema": "r2h_budgeted_review_packet_v1",
        "source": report["source"],
        "requested_frame_budget": budget, "selected_frame_count": len(selected),
        "unused_frame_budget": max(0, budget - len(selected)),
        "selected_frame_indices": sorted(selected), "selected_frames": selected_rows,
        "selection_policy": [
            "Always request the first and last decoded frames.",
            "Choose one primary displacement event by recorded priority, then earliest start/end.",
            "Request its two endpoints and one unknown interior frame nearest the temporal midpoint, if present.",
            "Spend remaining slots on maximal temporal distance from already selected frames; ties use the earlier frame.",
            "Do not silently increase the budget or substitute synthetic/interpolated frames.",
        ],
        "primary_displacement_event_id": primary["event_id"] if primary else None,
        "required_frame_indices": sorted(mandatory),
        "required_frame_count": len(mandatory),
        "missing_required_frame_indices": mandatory_missing,
        "budget_status": status,
        "budget_insufficient_for_required_frames": insufficient_primary,
        "event_representative_coverage_incomplete": bool(missing_event_representatives),
        "minimum_union_size_for_all_event_representatives": len(
            set(mandatory) | {i for c in coverage for i in c["representative_frame_indices"]}),
        "global_coverage_selection_steps": global_steps,
        "event_coverage": coverage,
        "unrepresented_event_ids": [c["event_id"] for c in coverage
                                   if c["sampling_status"] == "unrepresented"],
        "events_with_missing_representatives": missing_event_representatives,
        "unknown_intervals_with_missing_representatives": [
            c for c in coverage if c["kind"] == "tracking_unknown_interval"
            and c["missing_representative_frame_indices"]],
        "intervals_not_densely_sampled": [
            c["event_id"] for c in coverage if not c["dense_interval_fully_sampled"]],
        "frames_needing_source_extraction": [
            r["frame_index"] for r in selected_rows if r["requires_source_frame_extraction"]],
        "media_note": "Metadata packet only. Existing evidence paths are copied from the probe; "
                      "missing images must be decoded from the original source at the recorded frame index.",
        "uncertainty_note": "Recorded probe priorities are rules, not calibrated probabilities. "
                            "Representatives selected is a sampling statement, never a verification result.",
        "dense_video_fully_sampled": len(selected) == len(rows),
        "full_video_verified": False, "vlm_called": False,
        "contact_state": "unknown_not_measured", "physical_validity": "unknown_not_measured",
        "review_status": "pending_human_or_vlm", "training_admitted": False,
    }


def create_packet(report_path, rows_path, budget, output_path):
    report_path, rows_path, output_path = map(Path, (report_path, rows_path, output_path))
    report_bytes, rows_bytes = report_path.read_bytes(), rows_path.read_bytes()
    packet = select_review_frames(
        json.loads(report_bytes),
        [json.loads(line) for line in rows_bytes.decode().splitlines() if line.strip()],
        budget)
    packet["inputs"] = {
        "report_path": str(report_path.resolve()),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "frame_candidates_path": str(rows_path.resolve()),
        "frame_candidates_sha256": hashlib.sha256(rows_bytes).hexdigest(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(packet, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--frames", type=Path, help="Defaults to frame_candidates.jsonl next to report")
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    packet = create_packet(args.report, args.frames or args.report.with_name("frame_candidates.jsonl"),
                           args.budget, args.output)
    print(json.dumps({k: packet[k] for k in (
        "selected_frame_indices", "budget_status", "missing_required_frame_indices",
        "events_with_missing_representatives", "training_admitted")}, indent=2))


if __name__ == "__main__":
    main()
