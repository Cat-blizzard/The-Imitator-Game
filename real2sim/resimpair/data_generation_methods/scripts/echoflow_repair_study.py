"""Small, immutable pipeline repair experiments. No training or automatic POST retry."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from data_generation_methods.r2h_pipeline.common import (
    file_sha256, json_sha256, load_env_file, write_json,
)
from data_generation_methods.r2h_pipeline.providers import GeminiClient, SeedanceClient, ProviderError
from data_generation_methods.r2h_pipeline.cli import download_video
from data_generation_methods.scripts.echoflow_pilot import diagnostic_transport, VIDEO_PROMPT

BASELINE = Path("/mnt/sdg/datasets/real_sim_human_pipeline/pilot_20261020")
ROOT = Path("/mnt/sdg/datasets/real_sim_human_pipeline/repair_study_20261020_v1")
CONFIG = WORKSPACE / "data_generation_methods/config.echoflow.example.json"
ENV = Path("/home/xzzhou/.config/policygen/credentials/echoflow.env")

# The sole generation intervention is explicit role/direction grounding.
# Image, model, resolution, duration and other generation settings match the pilot.
ROLE_SUFFIX = """
Resolve object roles once from the first image and keep them fixed for the entire shot.
The SOURCE is the bowl on the RIGHT SIDE OF THE IMAGE: it contains the red apple.
The DESTINATION is the bowl on the LEFT SIDE OF THE IMAGE: it starts empty.
Image left/right are the viewer's left/right, never the person's left/right.
The red apple stays motionless in the image-right source bowl until the hand
visibly grasps it. Then carry this same red apple from IMAGE RIGHT to IMAGE LEFT.
At the end the image-right bowl is empty and the image-left bowl contains the apple.
The small green object outside the bowls is a distractor, not the object to move.
"""

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "frame_observations": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "frame_index": {"type": "integer"},
                    "red_apple_location": {"type": "string", "enum": [
                        "image_left_bowl", "image_right_bowl", "held",
                        "elsewhere", "unknown"]},
                    "hand_contact": {"type": "string", "enum": [
                        "visible_contact", "no_visible_contact", "unknown"]},
                    "visible_evidence": {"type": "string"},
                },
                "required": ["frame_index", "red_apple_location", "hand_contact", "visible_evidence"],
            },
        },
        "suspected_discontinuities": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "before_frame_index": {"type": "integer"},
                    "after_frame_index": {"type": "integer"},
                    "visible_evidence": {"type": "string"},
                },
                "required": ["before_frame_index", "after_frame_index", "visible_evidence"],
            },
        },
        "unresolved_questions": {"type": "array", "items": {"type": "string"}},
        "observed_summary": {"type": "string"},
    },
    "required": ["frame_observations", "suspected_discontinuities",
                 "unresolved_questions", "observed_summary"],
}
REVIEW_PROMPT = """Observe the supplied time-stamped frames, which are from one video.
Do not infer an intended task or use a supplied success criterion: none is given.
Use image-left/image-right, relative to the viewer. Track the same red apple,
distinguishing it from the small green object. Describe EVERY supplied frame in order.
An absent view of contact is not proof that contact never occurred between distant
frames. Use timestamps to distinguish a large unobserved interval from a very short
state jump. Report suspect discontinuities only with explicit visible evidence and
the supplied frame indices bracketing the change. Do not certify physical realism.
Unknown or occluded evidence must remain unknown. Do not invent intermediate frames.
Return one JSON object exactly matching this schema, with no other text:
""" + json.dumps(REVIEW_SCHEMA, ensure_ascii=False)


def save(path, value):
    write_json(path, value, overwrite=False)


def prepare_call(directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "call_started.json").exists():
        raise RuntimeError(f"Already attempted; inspect existing evidence, do not resubmit: {directory}")
    metadata.update(
        schema="r2h_repair_request_v1", training_admitted=False,
        script_sha256=file_sha256(Path(__file__)),
        providers_sha256=file_sha256(WORKSPACE / "data_generation_methods/r2h_pipeline/providers.py"),
        created_unix=time.time(),
    )
    save(directory / "request_metadata.json", metadata)
    # Exclusive, durable marker before any potentially billable request.
    import os
    with (directory / "call_started.json").open("x") as stream:
        json.dump({"started_unix": time.time(), "automatic_retry": False}, stream)
        stream.flush()
        os.fsync(stream.fileno())


def frame_packet(video, directory, indices):
    import cv2
    raw = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video),
    ], check=True, capture_output=True, text=True)
    pts = [float(f["best_effort_timestamp_time"]) for f in json.loads(raw.stdout)["frames"]]
    indices = sorted(set(indices(len(pts)) if callable(indices) else indices))
    if not indices or min(indices) < 0 or max(indices) >= len(pts):
        raise ValueError("Frame indices outside the actual decoded sequence")
    directory.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    result = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Could not decode frame {idx}")
            path = directory / f"frame_{idx:06d}.jpg"
            if path.exists():
                raise ValueError("Frame packet already exists")
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError("Could not save review frame")
            result.append({
                "path": str(path), "timestamp_seconds": pts[idx], "frame_index": idx,
                "sha256": file_sha256(path), "width": frame.shape[1], "height": frame.shape[0],
            })
    finally:
        cap.release()
    save(directory / "packet.json", {
        "source_path": str(video), "source_sha256": file_sha256(video),
        "timestamp_source": "ffprobe_best_effort_timestamp_time", "frames": result,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["submit-role-grounded", "poll-role-grounded",
                                          "review-uniform", "review-boundary", "review-repair"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    load_env_file(ENV)
    cfg = json.loads(CONFIG.read_text())
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    directory = root / args.command
    started = time.monotonic()
    if args.command == "poll-role-grounded":
        submission = json.loads((root / "submit-role-grounded/result.json").read_text())
        seed_cfg = dict(cfg["seedance"], timeout=30)
        result = SeedanceClient(seed_cfg).poll(submission["task_id"])
        save(directory / f"{time.time_ns()}.json", result)
        if result["status"] == "succeeded":
            target = root / "role_grounded.mp4"
            if not target.exists():
                result["media"] = download_video(
                    result["output_url"], target, task_id=result["task_id"],
                    request_sha256=file_sha256(root / "submit-role-grounded/request_metadata.json"))
        print(json.dumps({k: v for k, v in result.items() if k != "output_url"}))
        return
    try:
        if args.command == "submit-role-grounded":
            anchor = BASELINE / "edit-anchor/human_initial.png"
            prompt = VIDEO_PROMPT + ROLE_SUFFIX
            payload = {
                "model": cfg["seedance"]["model"],
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(anchor.read_bytes()).decode()},
                     "role": "first_frame"},
                ],
                "ratio": "adaptive", "duration": 8, "resolution": "720p", "generate_audio": False,
            }
            client = SeedanceClient(dict(cfg["seedance"], timeout=600),
                                    transport=diagnostic_transport(directory))
            client.prepare_submit(payload)
            # Preserve all generation settings and the input hash; no credentials.
            metadata = {
                "experiment": "explicit_role_direction_prompt_only",
                "baseline_request": str(BASELINE / "submit-video/request_metadata.json"),
                "anchor_path": str(anchor), "anchor_sha256": file_sha256(anchor),
                "prompt": prompt, "payload_sha256": json_sha256(payload),
                "generation_config": {k: v for k, v in payload.items() if k != "content"},
                "limitation": "One stochastic case per prompt; baseline has no fixed seed; not a statistical ablation.",
            }
            prepare_call(directory, metadata)
            result = client.submit(payload)
        else:
            video = BASELINE / "seedance_recovery/candidate_generated_human.mp4"
            if args.command == "review-uniform":
                selection = lambda n: [round(i * (n - 1) / 7) for i in range(8)]
                selector = "uniform_8_including_first_last"
            elif args.command == "review-boundary":
                # Case-study schedule fixed before either frame review, informed by
                # the already-observed boundary issue; not a blind benchmark.
                selection = [0, 1, 3, 6, 24, 72, 144, 192]
                selector = "known_failure_boundary_8_not_blind_benchmark"
            else:
                video = root / "role_grounded.mp4"
                selection = lambda n: sorted(set([0, 1, 3, 6, n // 8, n // 3, 2 * n // 3, n - 1]))
                selector = "same_boundary_heavy_schedule_new_video"
            if (directory / "call_started.json").exists():
                raise RuntimeError("Review already attempted; inspect stored response")
            frames = frame_packet(video, directory / "frames", selection)
            metadata = {
                "experiment": args.command, "model": cfg["gemini"]["model"],
                "source_path": str(video), "source_sha256": file_sha256(video),
                "prompt": REVIEW_PROMPT, "response_schema": REVIEW_SCHEMA,
                "schema_sha256": json_sha256(REVIEW_SCHEMA), "frame_selector": selector,
                "frames": frames, "allow_outer_json_fence_only": True,
                "generation_config": {"responseMimeType": "application/json",
                                      "responseJsonSchema": REVIEW_SCHEMA},
            }
            prepare_call(directory, metadata)
            result = GeminiClient(cfg["gemini"], transport=diagnostic_transport(directory)).review_frames(
                [{k: f[k] for k in ("path", "timestamp_seconds", "frame_index")} for f in frames],
                REVIEW_PROMPT, REVIEW_SCHEMA, allow_json_fence=True)
            actual = [x["frame_index"] for x in result["review"]["frame_observations"]]
            expected = [x["frame_index"] for x in frames]
            if actual != expected:
                save(directory / "parsed_but_invalid_frame_coverage.json", result)
                raise ProviderError("review_frame_coverage_mismatch")
            result["frame_coverage_verified"] = True
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["training_admitted"] = False
        save(directory / "result.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"output_url", "raw_text"}}))
    except ProviderError as exc:
        failure = {
            "status": "failed_or_unresolved", "code": exc.code, "http_status": exc.http_status,
            "submission_unknown": exc.submission_unknown,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "automatic_retry": False,
        }
        save(directory / "failure.json", failure)
        print(json.dumps(failure))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
