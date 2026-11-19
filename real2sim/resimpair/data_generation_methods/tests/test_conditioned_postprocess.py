import csv
import json
import math
from pathlib import Path
import shutil
import sys

import pytest

from data_generation_methods.scripts import postprocess_conditioned_case_20261020c as post
from data_generation_methods.r2h_pipeline.batch_pipeline import paid_stage
from data_generation_methods.r2h_pipeline.common import write_json


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(post.visible_review, "CASE", tmp_path)
    video, model = tmp_path / "video.mp4", tmp_path / "hand.task"
    video.write_bytes(b"bound media fixture")
    model.write_bytes(b"bound hand model fixture")
    manifest = tmp_path / "source_manifest.json"
    write_json(manifest, {"source": "fixture"})
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.symlink_to(sys.executable)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    provider_config = tmp_path / "provider_config.json"
    write_json(provider_config, {"gemini": {"model": "fixture-review-model"}})
    monkeypatch.setattr(post.visible_review, "DEFAULT_CONFIG", provider_config)
    width, height, frames = 960, 960, 2
    parameters = {
        "max_fb_error_px": 1.5, "min_handle_points": 3, "min_background_points": 6,
        "hand_near_threshold_px": math.hypot(width, height) * 0.025,
        "motion_threshold_px_per_frame": math.hypot(width, height) * 0.001,
        "background_model": "translation_only_median_displacement", "point_reinitialization": False,
    }
    output = tmp_path / "diagnostic"
    output.mkdir()
    diagnostic = {
        "schema": "fixed_camera_interaction_diagnostic_v1",
        "video": {**post.binding(video), "width": width, "height": height, "frames": frames},
        "model": {**post.binding(model), "mediapipe_version": "fixture-version", "num_hands": 6},
        "parameters": parameters,
        "handle_roi_normalized_xyxy": [0.5, 0.4, 0.6, 0.5],
        "background_roi_normalized_xyxy": [0.4, 0.2, 0.5, 0.3],
        "roi_provenance": "manual visible fixture selection",
        "rows": [{"frame_index": i, "timestamp_seconds": i / 24, "status": "tracked_2d_proxy",
                  "detected_hand_count": 1, "nearest_hand_keypoint_to_handle_track_px": 1.0,
                  "handle_step_px": None if i == 0 else 0.0} for i in range(frames)],
    }
    write_json(output / "diagnostic.json", diagnostic)
    with (output / "timeline.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostic["rows"][0]))
        writer.writeheader()
        writer.writerows(diagnostic["rows"])
    monkeypatch.setattr(post, "probe_video", lambda path: {
        **post.binding(video), "width": width, "height": height, "frames": frames})
    config = {
        "schema": "r2h_cabinet_postprocess_config_v1",
        "video": post.binding(video), "hand_model": post.binding(model),
        "source_manifest": post.binding(manifest),
        "diagnostic_environment": {
            "python_path": str(python), "python_sha256": post.file_sha256(python),
            "pyvenv_config": post.binding(venv / "pyvenv.cfg"), "mediapipe_version": "fixture-version"},
        "diagnostic": {
            "output_directory": str(output), "expected_parameters": parameters,
            **{key: diagnostic[key] for key in (
                "handle_roi_normalized_xyxy", "background_roi_normalized_xyxy", "roi_provenance")}},
        "triage": {"path": str(output / "triage.json"), "maximum_visible_hands": 1,
                   "max_requests": 2, "max_interval_frames": 12, "context_frames": 2,
                   "minimum_candidate_run_frames": 2},
        "review": {"output_directory": str(tmp_path / "review"), "frame_budget": 48,
                   "model": "fixture-review-model"},
        "records_directory": str(tmp_path / "execution"),
        "max_new_paid_reviews": 0, "prior_tools_and_review_launched_independently": True,
        "pair_validation": "not_run", "training_admitted": False,
    }
    config_path = tmp_path / "postprocess.json"
    write_json(config_path, config)
    return config, config_path, diagnostic


def test_cached_diagnostic_checks_video_model_roi_and_environment(configured):
    config, _, _ = configured
    post.validate_config(config)
    path, reused = post.ensure_diagnostic(config)
    assert reused
    config["diagnostic"]["handle_roi_normalized_xyxy"] = [0.1, 0.1, 0.2, 0.2]
    with pytest.raises(ValueError, match="ROI configuration differs"):
        post.validate_diagnostic(path, config)


def test_config_content_binding_mismatch_rejected(configured):
    config, _, _ = configured
    config["video"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        post.validate_config(config)


def test_missing_diagnostic_dispatches_declared_isolated_cli(configured, monkeypatch):
    config, _, diagnostic = configured
    output = Path(config["diagnostic"]["output_directory"])
    timeline = (output / "timeline.csv").read_bytes()
    shutil.rmtree(output)
    calls = []
    def local_cli(command, **kwargs):
        calls.append((command, kwargs))
        write_json(output / "diagnostic.json", diagnostic)
        (output / "timeline.csv").write_bytes(timeline)
    monkeypatch.setattr(post.subprocess, "run", local_cli)
    _, reused = post.ensure_diagnostic(config)
    assert not reused
    command, options = calls[0]
    assert command[0] == config["diagnostic_environment"]["python_path"]
    assert command[1:3] == ["-m", "data_generation_methods.r2h_pipeline.contact_diagnostics"]
    assert "--handle-roi" in command and "--background-roi" in command
    assert options["cwd"] == post.WORKSPACE


def test_zero_paid_budget_reuses_exact_stage_but_refuses_missing_or_changed_cache(tmp_path):
    stage = tmp_path / "review"
    metadata = {"kind": "fixture", "model": "fixture-review-model"}
    expected = paid_stage(stage, metadata, lambda transport: {"review": {"value": "unknown"}})
    with post.limit_paid_review_calls(0) as count:
        actual = post.visible_review.paid_stage(
            stage, metadata, lambda transport: pytest.fail("Must not issue API"))
        assert actual == expected
        assert count["new_paid_review_calls"] == 0
        with pytest.raises(ValueError, match="different inputs"):
            post.visible_review.paid_stage(stage, {**metadata, "kind": "changed"},
                                           lambda transport: pytest.fail("Must not issue API"))
        with pytest.raises(RuntimeError, match="budget exhausted"):
            post.visible_review.paid_stage(tmp_path / "missing", metadata,
                                           lambda transport: pytest.fail("Must not issue API"))
    assert not (tmp_path / "missing/call_started.json").exists()


def test_unknown_paid_result_never_resubmits_even_with_one_call_budget(tmp_path):
    stage = tmp_path / "unknown"
    write_json(stage / "call_started.json", {"started": True})
    with post.limit_paid_review_calls(1):
        with pytest.raises(RuntimeError, match="automatic resubmission prohibited"):
            post.visible_review.paid_stage(stage, {"kind": "fixture"},
                                           lambda transport: pytest.fail("Must not issue API"))


def test_at_most_one_new_review_operation_is_allowed(tmp_path):
    with post.limit_paid_review_calls(1) as count:
        post.visible_review.paid_stage(tmp_path / "first", {"kind": "fixture"},
                                      lambda transport: {"review": {"value": "unknown"}})
        assert count["new_paid_review_calls"] == 1
        with pytest.raises(RuntimeError, match="budget exhausted"):
            post.visible_review.paid_stage(tmp_path / "second", {"kind": "fixture"},
                                           lambda transport: pytest.fail("Second operation forbidden"))


def test_cached_postprocess_stops_unpaired_and_unapproved(configured, monkeypatch):
    config, config_path, _ = configured
    diagnostic_path, _ = post.ensure_diagnostic(config)
    post.ensure_triage(config, diagnostic_path)
    review_dir = Path(config["review"]["output_directory"])
    metadata = {"kind": "fixture", "model": "fixture-review-model"}
    observation = {"observations": {"interaction": {"value": "unknown"}}}
    paid_stage(review_dir, metadata, lambda transport: {"review": observation})
    def prepare(*args, **kwargs):
        prepared = {"frames": [{"frame_index": 0}], "provider_called": False}
        write_json(review_dir / "prepared_inputs.json", prepared)
        return prepared
    def cached_review(*args, **kwargs):
        result = post.visible_review.paid_stage(
            review_dir, metadata, lambda transport: pytest.fail("Cache-only execution made new call"))
        summary = result["review"]
        write_json(review_dir / "visible_evidence_summary.json", summary)
        return summary
    monkeypatch.setattr(post.visible_review, "prepare_review_inputs", prepare)
    monkeypatch.setattr(post.visible_review, "review", cached_review)
    result = post.run_postprocess(config_path)
    assert result["state"] == "needs_additional_visible_evidence"
    assert result["unresolved_queries"] == ["interaction"]
    assert result["new_paid_review_calls"] == 0
    assert result["diagnostic_reused"] and result["review_reused"]
    assert result["this_execution_claim"] == "cache_validation_and_reuse"
    assert result["prior_tools_and_review_launched_independently"]
    assert result["pair_validation"] == "not_run"
    assert not result["quality_release_approved"] and not result["training_admitted"]
    # A second inspection must preserve evidence and remain cache-only.
    again = post.run_postprocess(config_path)
    assert again["new_paid_review_calls"] == 0


@pytest.mark.parametrize("value,state", [
    ("yes", "awaiting_independent_pair_and_quality_release"),
    ("unknown", "needs_additional_visible_evidence"),
    ("no", "needs_generation_repair"),
])
def test_visible_failure_and_uncertainty_do_not_look_like_waiting_release(value, state):
    result = post.visible_outcome({"handle": {"value": value}})
    assert result["state"] == state
    assert result["visible_problem_queries"] == (["handle"] if value == "no" else [])
    assert result["unresolved_queries"] == (["handle"] if value == "unknown" else [])


@pytest.mark.parametrize("field,value", [
    ("max_new_paid_reviews", 2), ("max_new_paid_reviews", True),
    ("pair_validation", "passed"), ("training_admitted", True),
])
def test_configuration_cannot_expand_budget_or_claim_unperformed_release(configured, field, value):
    config, _, _ = configured
    config[field] = value
    with pytest.raises(ValueError):
        post.validate_config(config)
