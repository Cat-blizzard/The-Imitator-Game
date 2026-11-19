"""One bounded cabinet postprocess: diagnostic -> triage -> visible review.

The complete local configuration is required. Cached evidence is verified, not
overwritten. This entry never performs pair validation or quality release.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from data_generation_methods.scripts import review_conditioned_video_20261020c as visible_review
from data_generation_methods.r2h_pipeline.common import file_sha256, json_sha256, probe_video, write_json
from data_generation_methods.r2h_pipeline.batch_pipeline import binding
from data_generation_methods.r2h_pipeline.contact_diagnostics import validate_roi
from data_generation_methods.r2h_pipeline.interaction_triage import triage_interaction


def read(path):
    return json.loads(Path(path).read_text())


def checked(value):
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str)
            or file_sha256(value["path"]) != value.get("sha256")):
        raise ValueError("Postprocess artifact binding mismatch")
    return value


def validate_config(config):
    if config.get("schema") != "r2h_cabinet_postprocess_config_v1":
        raise ValueError("Unknown fixed-case postprocess configuration")
    for key in ("video", "hand_model", "source_manifest"):
        checked(config[key])
    if Path(config["source_manifest"]["path"]).resolve() != (
            visible_review.CASE / "source_manifest.json").resolve():
        raise ValueError("Postprocess configuration refers to a different fixed case")
    environment = config["diagnostic_environment"]
    python = Path(environment["python_path"])
    if (not python.is_absolute() or not python.is_file()
            or file_sha256(python) != environment["python_sha256"]):
        raise ValueError("Isolated diagnostic Python binding mismatch")
    checked(environment["pyvenv_config"])
    if (python.parent.name != "bin"
            or Path(environment["pyvenv_config"]["path"]).resolve()
            != (python.parent.parent / "pyvenv.cfg").resolve()):
        raise ValueError("Diagnostic Python must belong to the declared isolated venv")
    if not isinstance(environment.get("mediapipe_version"), str):
        raise ValueError("Explicit MediaPipe version required")
    diagnostic = config["diagnostic"]
    validate_roi(diagnostic["handle_roi_normalized_xyxy"])
    validate_roi(diagnostic["background_roi_normalized_xyxy"])
    if not isinstance(diagnostic.get("roi_provenance"), str) or not diagnostic["roi_provenance"].strip():
        raise ValueError("Manual ROI provenance is required")
    budget = config.get("max_new_paid_reviews")
    if type(budget) is not int or budget not in (0, 1):
        raise ValueError("At most one new paid review may be configured")
    if config.get("pair_validation") != "not_run" or config.get("training_admitted") is not False:
        raise ValueError("This entry cannot claim pair validation or training admission")
    if type(config.get("prior_tools_and_review_launched_independently")) is not bool:
        raise ValueError("Explicit prior execution provenance is required")
    if type(config["review"].get("frame_budget")) is not int or config["review"]["frame_budget"] < 1:
        raise ValueError("Explicit positive review frame budget required")
    return config


def validate_diagnostic(path, config):
    data = read(path)
    if data.get("schema") != "fixed_camera_interaction_diagnostic_v1":
        raise ValueError("Unexpected diagnostic schema")
    for field, expected in (("video", config["video"]), ("model", config["hand_model"])):
        checked(data[field])
        if any(data[field].get(key) != expected[key] for key in ("path", "sha256")):
            raise ValueError("Diagnostic binds different " + field)
    if data["model"].get("mediapipe_version") != config["diagnostic_environment"]["mediapipe_version"]:
        raise ValueError("Diagnostic model environment version mismatch")
    diagnostic = config["diagnostic"]
    for field in ("handle_roi_normalized_xyxy", "background_roi_normalized_xyxy", "roi_provenance"):
        if data.get(field) != diagnostic[field]:
            raise ValueError("Diagnostic ROI configuration differs: " + field)
    if data.get("parameters") != diagnostic["expected_parameters"]:
        raise ValueError("Diagnostic parameters differ from declared configuration")
    media = probe_video(config["video"]["path"])
    if any(data["video"].get(key) != media[key] for key in ("frames", "width", "height")):
        raise ValueError("Diagnostic media dimensions or frame count differ")
    return data


def ensure_diagnostic(config):
    settings = config["diagnostic"]
    output = Path(settings["output_directory"])
    path = output / "diagnostic.json"
    reused = path.exists()
    if not reused:
        # CLI currently exposes ROI controls and fixed default thresholds.
        media = probe_video(config["video"]["path"])
        diagonal = math.hypot(media["width"], media["height"])
        supported = {
            "max_fb_error_px": 1.5, "min_handle_points": 3, "min_background_points": 6,
            "hand_near_threshold_px": diagonal * 0.025,
            "motion_threshold_px_per_frame": diagonal * 0.001,
            "background_model": "translation_only_median_displacement",
            "point_reinitialization": False,
        }
        if settings["expected_parameters"] != supported:
            raise ValueError("Requested thresholds are not exposed by the current diagnostic CLI")
        if output.exists() and any(output.iterdir()):
            raise ValueError("Incomplete diagnostic directory exists; choose a new evidence version")
        command = [
            config["diagnostic_environment"]["python_path"], "-m",
            "data_generation_methods.r2h_pipeline.contact_diagnostics",
            "--video", config["video"]["path"], "--model", config["hand_model"]["path"],
            "--output", str(output), "--handle-roi",
            *map(str, settings["handle_roi_normalized_xyxy"]), "--background-roi",
            *map(str, settings["background_roi_normalized_xyxy"]),
            "--roi-provenance", settings["roi_provenance"],
        ]
        subprocess.run(command, check=True, cwd=WORKSPACE, timeout=600,
                       env={**os.environ, "PYTHONPATH": str(WORKSPACE)},
                       capture_output=True, text=True)
    validate_diagnostic(path, config)
    if not (output / "timeline.csv").exists():
        raise ValueError("Diagnostic timeline is missing")
    return path, reused


def ensure_triage(config, diagnostic_path):
    settings = config["triage"]
    result = triage_interaction(
        diagnostic_path, maximum_visible_hands=settings["maximum_visible_hands"],
        max_requests=settings["max_requests"], max_interval_frames=settings["max_interval_frames"],
        context_frames=settings["context_frames"],
        minimum_candidate_run_frames=settings["minimum_candidate_run_frames"],
        timeline_path=Path(diagnostic_path).parent / "timeline.csv")
    output = Path(settings["path"])
    reused = output.exists()
    if reused and read(output) != result:
        raise ValueError("Cached triage differs from bound diagnostics and configured rules")
    if not reused:
        write_json(output, result)
    return output, reused


@contextmanager
def limit_paid_review_calls(maximum_new_calls):
    """Bound the existing paid_stage API without changing review artifacts."""
    original = visible_review.paid_stage
    counter = {"new_paid_review_calls": 0}

    def limited(directory, metadata, operation):
        directory = Path(directory)
        result = directory / "result.json"
        started = directory / "call_started.json"
        if started.exists() and not result.exists():
            raise RuntimeError("Unresolved paid review; automatic resubmission prohibited")
        if not result.exists() and counter["new_paid_review_calls"] >= maximum_new_calls:
            raise RuntimeError("New paid review budget exhausted; cached result required")
        def counted(transport):
            if counter["new_paid_review_calls"] >= maximum_new_calls:
                raise RuntimeError("New paid review budget exhausted before operation")
            counter["new_paid_review_calls"] += 1
            return operation(transport)
        # Existing paid_stage revalidates input fingerprint before cache reuse.
        return original(directory, metadata, counted)

    visible_review.paid_stage = limited
    try:
        yield counter
    finally:
        visible_review.paid_stage = original


def visible_outcome(observations):
    values = {key: item["value"] for key, item in observations.items()}
    problems = sorted(key for key, value in values.items() if value == "no")
    unknowns = sorted(key for key, value in values.items() if value == "unknown")
    state = ("needs_generation_repair" if problems else
             "needs_additional_visible_evidence" if unknowns else
             "awaiting_independent_pair_and_quality_release")
    return {"state": state, "observation_values": values,
            "visible_problem_queries": problems, "unresolved_queries": unknowns}


def run_postprocess(config_path):
    config_path = Path(config_path).resolve(strict=True)
    config = validate_config(read(config_path))
    records = Path(config["records_directory"])
    records.mkdir(parents=True, exist_ok=True)
    with (records / "execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        diagnostic_path, diagnostic_reused = ensure_diagnostic(config)
        triage_path, triage_reused = ensure_triage(config, diagnostic_path)
        review_dir = Path(config["review"]["output_directory"])
        review_reused = (review_dir / "result.json").exists()
        # Fail before a paid stage if its saved model identity differs.
        if review_reused:
            metadata = read(review_dir / "request_metadata.json")
            if metadata.get("model") != config["review"]["model"]:
                raise ValueError("Cached review model differs from declared model")
        actual_model = read(visible_review.DEFAULT_CONFIG)["gemini"]["model"]
        if actual_model != config["review"]["model"]:
            raise ValueError("Active review model differs from configured model")
        prepared = visible_review.prepare_review_inputs(
            Path(config["video"]["path"]), review_dir, triage_path=triage_path,
            frame_budget=config["review"]["frame_budget"])
        with limit_paid_review_calls(config["max_new_paid_reviews"]) as calls:
            summary = visible_review.review(
                Path(config["video"]["path"]), review_dir, triage_path=triage_path,
                frame_budget=config["review"]["frame_budget"])
        # Visible evidence alone never supplies the independent robot pair.
        outcome = {
            "schema": "r2h_cabinet_postprocess_execution_v1",
            **visible_outcome(summary["observations"]),
            "config": binding(config_path), "video": config["video"],
            "diagnostic": binding(diagnostic_path), "triage": binding(triage_path),
            "prepared_inputs": binding(review_dir / "prepared_inputs.json"),
            "visible_evidence_summary": binding(review_dir / "visible_evidence_summary.json"),
            "diagnostic_reused": diagnostic_reused, "triage_reused": triage_reused,
            "review_reused": review_reused, **calls,
            "maximum_new_paid_reviews": config["max_new_paid_reviews"],
            "actual_review_frame_count": len(prepared["frames"]),
            "prior_tools_and_review_launched_independently":
                config["prior_tools_and_review_launched_independently"],
            "this_execution_claim": (
                "cache_validation_and_reuse" if diagnostic_reused and review_reused
                else "bounded_postprocess_execution"),
            "pair_validation": "not_run", "framewise_action_alignment": False,
            "physical_contact_ground_truth": False,
            "quality_release_approved": False, "training_admitted": False,
        }
        snapshot = records / "runs" / f"{time.time_ns()}_{json_sha256(outcome)[:16]}.json"
        write_json(snapshot, outcome)
        write_json(records / "LATEST.json", {
            "run": binding(snapshot), "state": outcome["state"],
            "training_admitted": False}, overwrite=True)
        return {**outcome, "execution_record": binding(snapshot)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_postprocess(args.config)
    print(json.dumps({key: result[key] for key in (
        "state", "diagnostic_reused", "review_reused", "new_paid_review_calls",
        "pair_validation", "training_admitted")}))


if __name__ == "__main__":
    main()
