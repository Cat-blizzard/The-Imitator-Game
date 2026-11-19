"""Deterministic, bounded dense-frame selection from unresolved observations."""
from __future__ import annotations


def plan_refinement(frame_count, base_indices, requests, *, max_frames=48):
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("Invalid frame count")
    if type(max_frames) is not int or max_frames < 1:
        raise ValueError("Invalid frame budget")
    if any(type(i) is not int or not 0 <= i < frame_count for i in base_indices):
        raise ValueError("Base frame outside media")
    selected = set(base_indices)
    if len(selected) > max_frames:
        raise ValueError("Base coverage exceeds budget")
    parsed = []
    for number, request in enumerate(requests):
        bounds = request.get("frame_interval")
        if (request.get("interval") != "start_inclusive_end_inclusive"
                or not isinstance(bounds, list) or len(bounds) != 2
                or any(type(i) is not int for i in bounds)
                or not 0 <= bounds[0] <= bounds[1] < frame_count):
            raise ValueError("Invalid refinement frame interval")
        parsed.append((bounds[1] - bounds[0] + 1, number, bounds, request))
    output = []
    # Complete short intervals first. Never quietly replace a requested
    # continuous interval with sparse samples and call it fully covered.
    for size, number, bounds, request in sorted(parsed):
        requested = set(range(bounds[0], bounds[1] + 1))
        needed = requested - selected
        can_fit = len(selected | requested) <= max_frames
        if can_fit:
            selected |= requested
        output.append({
            "request_index": number, "query_id": request["query_id"],
            "frame_interval": bounds, "requested_frame_count": size,
            "additional_frames_needed": len(needed),
            "selection": "continuous_interval_selected" if can_fit else "deferred_budget",
            "semantic_uncertainty_resolved": False,
        })
    return {
        "schema": "r2h_dense_refinement_plan_v1",
        "base_frame_indices": sorted(set(base_indices)),
        "selected_frame_indices": sorted(selected), "maximum_frames": max_frames,
        "added_frame_count": len(selected - set(base_indices)),
        "requests": sorted(output, key=lambda value: value["request_index"]),
        "selection_is_not_verification": True, "training_admitted": False,
    }
