import json

import pytest

from data_generation_methods.scripts import review_conditioned_video_20261020c as review_module
from data_generation_methods.r2h_pipeline.interaction_triage import triage_interaction
from data_generation_methods.r2h_pipeline.common import write_json


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    import cv2
    import numpy as np
    folder = tmp_path_factory.mktemp("actual_review_media")
    video = folder / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24, (32, 32))
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((32, 32, 3), 20 + index * 5, dtype=np.uint8))
    writer.release()
    anchor = folder / "terminal.png"
    cv2.imwrite(str(anchor), np.full((32, 32, 3), 200, dtype=np.uint8))
    return video, anchor


@pytest.fixture
def case(tmp_path, monkeypatch, media):
    video, anchor = media
    monkeypatch.setattr(review_module, "CASE", tmp_path)
    write_json(tmp_path / "source_manifest.json", {"sim_terminal": review_module.binding(anchor)})
    return video, tmp_path / "review", tmp_path


def fixture_triage(case):
    video, output, folder = case
    model = folder / "hand.task"
    model.write_bytes(b"diagnostic model fixture")
    stamps = review_module._timestamps(video)
    rows = [{
        "frame_index": i, "timestamp_seconds": stamp, "status": "tracked_2d_proxy",
        "detected_hand_count": 2 if 5 <= i <= 12 else 1,
        "nearest_hand_keypoint_to_handle_track_px": 10.0,
        "handle_step_px": None if i == 0 else 0.1,
    } for i, stamp in enumerate(stamps)]
    diagnostic = {
        "schema": "fixed_camera_interaction_diagnostic_v1",
        "video": {**review_module.binding(video), "frames": len(rows)},
        "model": {**review_module.binding(model), "num_hands": 6},
        "handle_roi_normalized_xyxy": [0.4, 0.4, 0.6, 0.6],
        "background_roi_normalized_xyxy": [0.1, 0.1, 0.2, 0.2],
        "roi_provenance": "manually declared fixture",
        "parameters": {"hand_near_threshold_px": 20, "motion_threshold_px_per_frame": 1},
        "rows": rows,
    }
    diagnostic_path = folder / "diagnostic.json"
    write_json(diagnostic_path, diagnostic)
    triage_path = folder / "triage.json"
    write_json(triage_path, triage_interaction(diagnostic_path, maximum_visible_hands=1))
    return triage_path


def unknown_result():
    return {"review": {
        "observations": {key: {"value": "unknown", "frame_indices": [],
                               "description": "Not established in supplied observations"}
                         for key in review_module.QUERIES},
        "limitations": ["RGB observations are not physical contact ground truth"],
    }}


def test_actual_frame_packet_matches_pts_dimensions_and_no_provider_is_called(case, monkeypatch):
    video, output, _ = case
    monkeypatch.setattr(review_module, "paid_stage", lambda *a: pytest.fail("No paid call"))
    prepared = review_module.prepare_review_inputs(video, output)
    assert prepared["provider_called"] is False
    packet = json.loads((output / "frames/packet.json").read_text())
    assert packet["timestamp_source"] == "ffprobe_best_effort_timestamp_time"
    assert prepared["frames"] == packet["frames"]
    stamps = review_module._timestamps(video)
    for frame in prepared["frames"]:
        assert frame["timestamp_seconds"] == stamps[frame["frame_index"]]
        assert frame["sha256"] == review_module.file_sha256(frame["path"])
        assert (frame["width"], frame["height"]) == (32, 32)
    assert review_module.prepare_review_inputs(video, output) == prepared


def test_tampered_frame_jpeg_is_rejected_before_a_review(case):
    video, output, _ = case
    prepared = review_module.prepare_review_inputs(video, output)
    from pathlib import Path
    Path(prepared["frames"][0]["path"]).write_bytes(b"replaced image")
    with pytest.raises(ValueError, match="content binding"):
        review_module.prepare_review_inputs(video, output)


def test_cached_packet_cannot_invent_timestamps(case):
    video, output, _ = case
    review_module.prepare_review_inputs(video, output)
    path = output / "frames/packet.json"
    packet = json.loads(path.read_text())
    packet["frames"][1]["timestamp_seconds"] += 0.02
    write_json(path, packet, overwrite=True)
    with pytest.raises(ValueError, match="actual media PTS"):
        review_module.prepare_review_inputs(video, output)


def test_budget_is_enforced_even_without_triage(case):
    video, output, _ = case
    with pytest.raises(ValueError, match="Base coverage exceeds budget"):
        review_module.prepare_review_inputs(video, output, frame_budget=5)


def test_real_triage_to_refinement_to_frame_packet_interface(case):
    video, output, _ = case
    triage_path = fixture_triage(case)
    prepared = review_module.prepare_review_inputs(video, output, triage_path=triage_path)
    selection = prepared["selection"]
    assert selection["source_triage"] == review_module.binding(triage_path)
    assert selection["requests"][0]["selection"] == "continuous_interval_selected"
    interval = selection["requests"][0]["frame_interval"]
    assert set(range(interval[0], interval[1] + 1)) <= set(selection["selected_frame_indices"])


def test_triage_request_or_roi_cannot_be_changed_while_preserving_top_media_hash(case):
    video, output, _ = case
    triage_path = fixture_triage(case)
    triage = json.loads(triage_path.read_text())
    triage["refinement_requests"][0]["media_sha256"] = "0" * 64
    write_json(triage_path, triage, overwrite=True)
    with pytest.raises(ValueError, match="bound diagnostic and rules"):
        review_module.prepare_review_inputs(video, output, triage_path=triage_path)


def test_static_source_reference_is_checked_against_manifest(case):
    video, output, folder = case
    path = folder / "source_manifest.json"
    source = json.loads(path.read_text())
    source["sim_terminal"]["sha256"] = "0" * 64
    write_json(path, source, overwrite=True)
    with pytest.raises(ValueError, match="content binding"):
        review_module.prepare_review_inputs(video, output)


def test_short_video_selection_has_no_negative_indices(case, monkeypatch):
    video, output, _ = case
    # Local selection validation is reached before packet extraction.
    monkeypatch.setattr(review_module, "_timestamps", lambda path: [0.0, 0.1])
    info = review_module.probe_video(video)
    info["frames"] = 2
    monkeypatch.setattr(review_module, "probe_video", lambda path: info)
    def capture_packet(path, folder, chooser):
        assert chooser(2) == [0, 1]
        raise RuntimeError("selection verified; do not extract an incompatible fixture")
    monkeypatch.setattr(review_module, "frame_packet", capture_packet)
    with pytest.raises(RuntimeError, match="selection verified"):
        review_module.prepare_review_inputs(video, output, frame_budget=2)


def test_unknown_observations_do_not_require_invented_citations():
    result = unknown_result()
    assert review_module.validate_review_result(result, [0, 1, 2, 3, 26, 27, 28, 29]) == result["review"]


@pytest.mark.parametrize("indices,message", [
    ([99], "not submitted"), ([], "no frame evidence"), ([0, 0], "Duplicate"),
])
def test_invalid_definite_frame_citations_are_rejected(indices, message):
    result = unknown_result()
    result["review"]["observations"]["one_hand_visible"].update(value="yes", frame_indices=indices)
    with pytest.raises(ValueError, match=message):
        review_module.validate_review_result(result, [0, 1, 2, 3, 26, 27, 28, 29])


def test_final_state_cannot_be_supported_only_by_initial_frame():
    result = unknown_result()
    result["review"]["observations"]["final_right_door_open"].update(value="yes", frame_indices=[0])
    with pytest.raises(ValueError, match="ending-frame evidence"):
        review_module.validate_review_result(result, [0, 1, 2, 3, 26, 27, 28, 29])


def test_cached_result_is_rechecked_against_full_schema():
    result = unknown_result()
    result["review"]["observations"].pop("one_hand_visible")
    with pytest.raises(Exception):
        review_module.validate_review_result(result, [0, 1])
