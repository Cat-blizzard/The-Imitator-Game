"""Safety-relevant regression cases for the deliberately limited color probe."""
import json

import cv2
import numpy as np
import pytest

from data_generation_methods.r2h_pipeline.boundary_probe import (
    ProbeConfig, annotate_temporal, build_reference, red_candidates, run_probe,
    score_candidates, select_candidate, validate_roi,
)


def image_with_target(x=150, y=125, skin=False):
    frame = np.full((200, 200, 3), 220, dtype=np.uint8)
    if x is not None:
        cv2.circle(frame, (x, y), 10, (35, 55, 195), -1)
    if skin:
        # A dark finger-colored red component: old largest-HSV logic may select it.
        cv2.rectangle(frame, (20, 20), (27, 43), (13, 32, 78), -1)
    return frame


def reference():
    return build_reference(image_with_target(), (.65, .5, .85, .75), ProbeConfig())


def make_row(index, frame, ref):
    config = ProbeConfig()
    candidates = score_candidates(red_candidates(frame, config), ref, config)
    selected, reason, margin = select_candidate(candidates, config)
    return {"frame_index": index, "timestamp_seconds": index / 24,
            "selected_candidate_id": selected["candidate_id"] if selected else None,
            "selection_reason": reason, "appearance_rank_margin": margin,
            "candidates": candidates}


def test_skin_only_missing_frame_is_unknown_not_target_motion():
    ref = reference()
    frame = image_with_target(x=None, skin=True)
    candidates = score_candidates(red_candidates(frame, ProbeConfig()), ref, ProbeConfig())
    assert candidates, "The confusing red skin proposal must actually exist"
    selected, reason, _ = select_candidate(candidates, ProbeConfig())
    assert selected is None and reason == "no_appearance_consistent_candidate"
    assert "too_dark_relative_to_anchor" in candidates[0]["rejection_reasons"]


def test_short_disappearance_and_relocation_proposes_review_without_fabricating_path():
    ref = reference()
    rows = [make_row(0, image_with_target(), ref),
            make_row(1, image_with_target(x=None, skin=True), ref),
            make_row(2, image_with_target(x=60), ref)]
    events = annotate_temporal(rows, ref, ProbeConfig())
    assert rows[1]["trajectory_px"] is None
    assert rows[1]["tracking_state"] == "unknown"
    displacement = next(e for e in events if e["kind"] == "appearance_candidate_displacement")
    assert displacement["missing_frames"] == 1
    assert displacement["distance_px"] == pytest.approx(90)
    assert all(r["contact_state"] == "unknown_not_measured" for r in rows)
    assert all(r["physical_validity"] == "unknown_not_measured" for r in rows)


def test_two_indistinguishable_red_objects_are_ambiguous():
    frame = image_with_target()
    cv2.circle(frame, (50, 125), 10, (35, 55, 195), -1)
    row = make_row(0, frame, reference())
    assert row["selected_candidate_id"] is None
    assert row["selection_reason"] == "multiple_similar_appearance_candidates"


def test_smooth_motion_does_not_trigger_displacement_review():
    ref = reference()
    rows = [make_row(i, image_with_target(x=150 - 2 * i), ref) for i in range(8)]
    assert annotate_temporal(rows, ref, ProbeConfig()) == []
    assert all(row["trajectory_px"] is not None for row in rows)


def test_long_occlusion_does_not_claim_jump_across_unobserved_motion():
    ref = reference()
    rows = [make_row(i, image_with_target(x=150 if i == 0 else 60 if i == 25 else None), ref)
            for i in range(26)]
    events = annotate_temporal(rows, ref, ProbeConfig())
    assert [event["kind"] for event in events] == ["tracking_unknown_interval"]
    assert rows[-1]["tracking_state"] == "reacquired_appearance_candidate"


def test_merged_component_extent_is_unknown_even_with_similar_color():
    frame = image_with_target()
    cv2.rectangle(frame, (148, 80), (153, 124), (35, 55, 195), -1)
    row = make_row(0, frame, reference())
    assert row["selected_candidate_id"] is None
    assert "component_extent_inconsistent_with_anchor" in row["candidates"][0]["rejection_reasons"]


def test_reference_requires_explicit_unambiguous_roi():
    with pytest.raises(ValueError, match="exactly one"):
        build_reference(image_with_target(), (0, 0, .2, .2), ProbeConfig())
    for bad_roi in ((.5, 0, .4, 1), (0, 0, 2, 1), (0, 0, float("nan"), 1)):
        with pytest.raises(ValueError):
            validate_roi(bad_roi)


def test_dense_video_export_keeps_unknown_and_never_admits_training(tmp_path):
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 24, (200, 200))
    assert writer.isOpened()
    frames = [image_with_target(), image_with_target(x=None, skin=True),
              image_with_target(x=60), image_with_target(x=60)]
    for frame in frames:
        writer.write(frame)
    writer.release()
    anchor = tmp_path / "anchor.png"
    assert cv2.imwrite(str(anchor), frames[0])
    output = tmp_path / "evidence"
    report = run_probe(video, anchor, (.65, .5, .85, .75), output)
    rows = [json.loads(line) for line in (output / "frame_candidates.jsonl").read_text().splitlines()]
    assert report["source"]["decoded_frames"] == len(frames)
    assert report["training_admitted"] is False
    assert report["physical_validity"] == "unknown_not_measured"
    assert report["summary"]["displacement_review_candidates"] == 1
    assert rows[1]["trajectory_px"] is None
    assert len(list((output / "keyframes").glob("*.png"))) == len(frames)
    assert list((output / "contact_sheets").glob("*.jpg"))
    assert (output / "candidate_trajectory.png").is_file()
    with pytest.raises(FileExistsError):
        run_probe(video, anchor, (.65, .5, .85, .75), output)
