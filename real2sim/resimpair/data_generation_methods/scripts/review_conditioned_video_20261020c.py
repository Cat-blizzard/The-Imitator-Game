"""Bounded visible-state review after a real generated media artifact exists."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.scripts.run_state_conditioned_case_20261020c import CASE
from data_generation_methods.scripts.echoflow_repair_study import frame_packet
from data_generation_methods.r2h_pipeline.batch_pipeline import paid_stage, binding, ENV, DEFAULT_CONFIG
from data_generation_methods.r2h_pipeline.common import load_env_file, probe_video, write_json, json_sha256, file_sha256
from data_generation_methods.r2h_pipeline.providers import GeminiClient, parse_review_json
from data_generation_methods.r2h_pipeline.refinement import plan_refinement
from data_generation_methods.r2h_pipeline.interaction_triage import triage_interaction

QUERIES = {
    "initial_right_door_closed": "At the beginning, is the image-right door closed?",
    "final_right_door_open": "At the end, is that same image-right door open?",
    "left_door_unchanged": "Does the image-left door retain its initial state in the visible evidence?",
    "one_hand_visible": "In inspected video frames, is there at most one distinct human hand?",
    "cube_visible_after_opening": "Is the small multicolored object visibly present in the upper compartment after opening?",
    "cube_appearance_matches_source": "Does that object's visible appearance and shelf location agree with the source simulation OPEN image?",
    "hand_handle_interaction_visible": "Is a hand visibly manipulating the handle rather than the door moving independently? Do not infer hidden contact.",
    "door_motion_after_hand_arrival": "Is the door visibly still before hand arrival and does opening occur after arrival? Missing timing evidence means unknown.",
    "hand_released_and_withdrawn": "Is the hand clearly away from the handle at the end, after manipulating it?",
    "no_visible_object_discontinuity": "Do inspected observations avoid a visible object jump, duplication or appearance change? Sparse non-detection cannot prove full-video absence.",
}
ITEM = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "value": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "frame_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "description": {"type": "string", "minLength": 1},
    }, "required": ["value", "frame_indices", "description"],
}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "observations": {"type": "object", "additionalProperties": False,
                         "properties": {key: ITEM for key in QUERIES}, "required": list(QUERIES)},
        "limitations": {"type": "array", "items": {"type": "string"}},
    }, "required": ["observations", "limitations"],
}


def _checked_binding(value):
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str)
            or file_sha256(value["path"]) != value.get("sha256")):
        raise ValueError("Artifact differs from its declared content binding")
    return value


def _save_consistent(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Existing evidence differs; use a new output directory")
        return
    write_json(path, value)


def _timestamps(video):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=best_effort_timestamp_time", "-of", "json", str(video),
    ], check=True, capture_output=True, text=True, timeout=60)
    values = [float(frame["best_effort_timestamp_time"])
              for frame in json.loads(result.stdout)["frames"]]
    if (not values or any(not math.isfinite(t) or t < 0 for t in values)
            or any(b <= a for a, b in zip(values, values[1:]))):
        raise ValueError("Require actual, finite, increasing frame timestamps")
    return values


def _validated_triage(path, media):
    triage = json.loads(path.read_text())
    if triage.get("schema") != "r2h_interaction_triage_v1":
        raise ValueError("Unknown interaction triage schema")
    evidence = triage.get("evidence_bindings", {})
    if evidence.get("media", {}).get("sha256") != media["sha256"]:
        raise ValueError("Triage is bound to different media")
    for key in ("media", "model", "diagnostic"):
        _checked_binding(evidence.get(key))
    if "timeline" in evidence:
        _checked_binding(evidence["timeline"])
    rules = triage["rules"]
    # Recompute cheap deterministic planning from the same bound observations.
    # This verifies ROI/model provenance and every emitted request; no inference.
    recomputed = triage_interaction(
        evidence["diagnostic"]["path"],
        maximum_visible_hands=rules["maximum_visible_hands"],
        max_requests=rules["max_requests"], max_interval_frames=rules["max_interval_frames"],
        context_frames=rules["context_frames"],
        minimum_candidate_run_frames=rules["minimum_candidate_run_frames"],
        timeline_path=evidence.get("timeline", {}).get("path"))
    if recomputed != triage:
        raise ValueError("Triage content does not match its bound diagnostic and rules")
    if triage["summary"]["frames"] != media["frames"]:
        raise ValueError("Triage and actual video have different frame counts")
    return triage


def _validated_packet(packet, media, indices, stamps):
    if (packet.get("source_sha256") != media["sha256"]
            or packet.get("timestamp_source") != "ffprobe_best_effort_timestamp_time"):
        raise ValueError("Review packet media or timestamp source changed")
    _checked_binding({"path": packet["source_path"], "sha256": packet["source_sha256"]})
    frames = packet.get("frames", [])
    if [f.get("frame_index") for f in frames] != indices:
        raise ValueError("Saved review frame indices changed")
    for frame in frames:
        _checked_binding(frame)
        index = frame["frame_index"]
        stamp = frame.get("timestamp_seconds")
        if (type(index) is not int or not 0 <= index < len(stamps)
                or type(stamp) not in (float, int) or not math.isfinite(stamp)
                or abs(stamp - stamps[index]) > 1e-6):
            raise ValueError("Saved review timestamp does not match actual media PTS")
        if (frame.get("width") != media["width"] or frame.get("height") != media["height"]):
            raise ValueError("Saved review frame dimensions changed")
    return frames


def prepare_review_inputs(video, output, *, triage_path=None, frame_budget=48):
    """Only local decoding, hash checks and deterministic frame selection."""
    video, output = Path(video).resolve(strict=True), Path(output)
    if type(frame_budget) is not int or frame_budget < 1:
        raise ValueError("A positive integer frame budget is required")
    media = probe_video(video)
    output.mkdir(parents=True, exist_ok=True)
    stamps = _timestamps(video)
    count = len(stamps)
    if media["frames"] not in (None, count):
        raise ValueError("Container frame count differs from decoded timestamps")
    media["frames"] = count
    base = sorted(set([0, 1, 2, 3, count-4, count-3, count-2, count-1] +
                      [round(i*(count-1)/17) for i in range(18)]))
    base = [index for index in base if 0 <= index < count]
    requests = []
    if triage_path:
        triage_path = Path(triage_path).resolve(strict=True)
        triage = _validated_triage(triage_path, media)
        requests = triage["refinement_requests"]
    selection = plan_refinement(count, base, requests, max_frames=frame_budget)
    if triage_path:
        selection["source_triage"] = binding(triage_path)
    _save_consistent(output / "frame_selection.json", selection)
    indices = selection["selected_frame_indices"]
    packet_path = output / "frames/packet.json"
    if not packet_path.exists():
        frame_packet(video, output / "frames", lambda _: indices)
    frames = _validated_packet(json.loads(packet_path.read_text()), media, indices, stamps)
    manifest_path = CASE / "source_manifest.json"
    source = json.loads(manifest_path.read_text())
    _checked_binding(source["sim_terminal"])
    anchor = Path(source["sim_terminal"]["path"])
    prepared = {
        "media": media, "frames": frames, "selection": selection,
        "packet": binding(packet_path), "static_identity_reference": binding(anchor),
        "source_manifest": binding(manifest_path),
        "preparation_only": True, "provider_called": False,
    }
    _save_consistent(output / "prepared_inputs.json", prepared)
    return prepared


def validate_review_result(result, indices):
    parsed = parse_review_json(json.dumps(result["review"]), SCHEMA)["review"]
    available = set(indices)
    for key, observation in parsed["observations"].items():
        if not set(observation["frame_indices"]) <= available:
            raise ValueError("Model cited a frame that was not submitted")
        if len(observation["frame_indices"]) != len(set(observation["frame_indices"])):
            raise ValueError("Duplicate frame evidence citation")
        if observation["value"] != "unknown" and not observation["frame_indices"]:
            raise ValueError("Definite visual statement has no frame evidence")
        if observation["value"] != "unknown" and key.startswith("initial_"):
            if not set(observation["frame_indices"]) & set(indices[:4]):
                raise ValueError("Initial-state statement lacks beginning-frame evidence")
        if observation["value"] != "unknown" and key in {
                "final_right_door_open", "hand_released_and_withdrawn"}:
            if not set(observation["frame_indices"]) & set(indices[-4:]):
                raise ValueError("Final-state statement lacks ending-frame evidence")
    return parsed


def review(video, output, *, triage_path=None, frame_budget=48):
    prepared = prepare_review_inputs(video, output, triage_path=triage_path, frame_budget=frame_budget)
    video, output = Path(video), Path(output)
    frames, selection = prepared["frames"], prepared["selection"]
    indices = selection["selected_frame_indices"]
    anchor = Path(prepared["static_identity_reference"]["path"])
    prompt = (
        "First there is ONE labelled STATIC SOURCE SIMULATION OPEN IMAGE. It is NOT a video "
        "frame and provides only a visual identity/location reference for the object inside "
        "the upper compartment. Then there are actual timestamped frames from ONE candidate "
        "human video. Inspect what is visible. Do not assume its intended action succeeds. "
        "Keep image-left/right and object identities fixed. Answer every observation with "
        "yes/no/unknown plus exact supplied VIDEO frame indices and a literal explanation. "
        "Do not invent citations; cite no index for the static source. Do not infer hidden "
        "contact, force, 3D physical validity or full-clip absence from sampled images. If "
        "occluded, temporally missing or ambiguous, answer unknown. The one-hand question "
        "counts distinct visible hands, including partial overlapping ones. Review the "
        "video independently of source robot pose. Return ONLY JSON matching the full schema.\n"
        + json.dumps({"queries": QUERIES, "schema": SCHEMA}, ensure_ascii=False)
    )
    load_env_file(ENV)
    cfg = json.loads(DEFAULT_CONFIG.read_text())["gemini"]
    def call(transport):
        client = GeminiClient(cfg, transport=transport)
        parts = [{"text": prompt}, {"text": "STATIC SOURCE SIMULATION OPEN IMAGE:"},
                 client._inline(anchor, "image"), {"text": "CANDIDATE VIDEO FRAME SEQUENCE:"}]
        for frame in frames:
            parts += [{"text": json.dumps({k: frame[k] for k in ("frame_index", "timestamp_seconds")})},
                      client._inline(Path(frame["path"]), "image")]
        return client._review_result(client._generate(
            client.model, parts, client._review_config(prompt, SCHEMA, True)), SCHEMA, True)
    result = paid_stage(output, {
        "kind": "visible_state_and_interaction_review",
        "media": binding(video), "static_identity_reference": binding(anchor),
        "frames": frames, "model": cfg["model"], "prompt": prompt,
        "response_schema": SCHEMA,
        "selection": selection, "frame_packet": prepared["packet"],
        "source_manifest": prepared["source_manifest"],
    }, call)
    parsed = validate_review_result(result, indices)
    summary = {
        "schema": "r2h_visible_case_review_v1", "video": binding(video),
        "raw_review": binding(output / "result.json"),
        "observations": parsed["observations"],
        "limitations": parsed["limitations"],
        "submitted_video_frames": len(frames), "actual_frame_indices": indices,
        "structured_review_validated": True,
        "frame_packet": prepared["packet"], "frame_selection": binding(output / "frame_selection.json"),
        "full_clip_absence_verified": False,
        "physical_contact_gt": False, "framewise_action_alignment": False,
        "quality_release_approved": False, "training_admitted": False,
    }
    write_json(output / "visible_evidence_summary.json", summary)
    print(json.dumps({key: value["value"] for key, value in summary["observations"].items()}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triage", type=Path)
    parser.add_argument("--frame-budget", type=int, default=48)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Decode, validate bindings and select frames; no provider call")
    args = parser.parse_args()
    if args.prepare_only:
        prepared = prepare_review_inputs(args.video, args.output, triage_path=args.triage,
                                         frame_budget=args.frame_budget)
        print(json.dumps({"provider_called": False, "frames": len(prepared["frames"]),
                          "indices": prepared["selection"]["selected_frame_indices"]}))
    else:
        review(args.video, args.output, triage_path=args.triage, frame_budget=args.frame_budget)


if __name__ == "__main__":
    main()
