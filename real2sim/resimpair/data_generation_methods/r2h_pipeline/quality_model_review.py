"""Blind, timestamp-grounded ordinal VLM review; no automatic training approval."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

from .batch_pipeline import DEFAULT_CONFIG, ENV, binding, paid_stage
from .common import file_sha256, load_env_file, probe_video, write_json
from .providers import GeminiClient, parse_review_json

DIMENSIONS = {
    "task_readability": "Can an observer identify the intended manipulation, object roles, and outcome?",
    "temporal_consistency": "Are object and actor identities, shapes and visible movement temporally coherent?",
    "human_structure": "Do visible people, limbs, hands and fingers form plausible and consistent bodies? Count partial or overlapping hands; do not assume every visible hand belongs to one person.",
    "interaction_visibility": "Are approach, manipulation, release and withdrawal visually interpretable? Score visible evidence, never real forces or true 3D contact.",
}
EVIDENCE = {
    "type": "object", "additionalProperties": False,
    "properties": {"frame_index": {"type": "integer", "minimum": 0},
                   "timestamp_seconds": {"type": "number", "minimum": 0},
                   "observation": {"type": "string", "minLength": 1}},
    "required": ["frame_index", "timestamp_seconds", "observation"],
}
SCORE = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "score": {"enum": [0, 1, 2, 3, 4, "unknown"]},
        "evidence": {"type": "array", "items": EVIDENCE},
        "explanation": {"type": "string", "minLength": 1},
        "unobserved_or_ambiguous": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "evidence", "explanation", "unobserved_or_ambiguous"],
}
CONSTRAINT = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["violation_observed", "supported_in_observed_frames", "unknown"]},
        "evidence": {"type": "array", "items": EVIDENCE},
        "explanation": {"type": "string", "minLength": 1},
    }, "required": ["status", "evidence", "explanation"],
}
CONSTRAINTS = ("task_goal_and_roles", "necessary_event_order", "visible_object_continuity",
               "visible_human_structural_consistency")
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "dimensions": {"type": "object", "additionalProperties": False,
                       "properties": {name: SCORE for name in DIMENSIONS}, "required": list(DIMENSIONS)},
        "constraints": {"type": "object", "additionalProperties": False,
                        "properties": {name: CONSTRAINT for name in CONSTRAINTS}, "required": list(CONSTRAINTS)},
        "visible_hand_inventory": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "frame_index": {"type": "integer", "minimum": 0},
                    "timestamp_seconds": {"type": "number", "minimum": 0},
                    "minimum_distinct_visible_hands": {"type": "integer", "minimum": 0},
                    "exact_count_known": {"type": "boolean"},
                    "description": {"type": "string", "minLength": 1}},
                "required": ["frame_index", "timestamp_seconds", "minimum_distinct_visible_hands",
                             "exact_count_known", "description"]},
            "minItems": 1},
        "limitations": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
    }, "required": ["dimensions", "constraints", "visible_hand_inventory", "limitations"],
}


def select_uniform_boundary_frames(count):
    if type(count) is not int or count < 2:
        raise ValueError("At least two actual frames required")
    return sorted(set(i for i in (
        [0, 1, 2, 3, count-4, count-3, count-2, count-1] +
        [round(i*(count-1)/17) for i in range(18)])
                      if 0 <= i < count))


def build_prompt(task_contract):
    if (set(task_contract) != {"initial", "necessary_steps", "goal", "invariants"}
            or not all(isinstance(task_contract[k], str) and task_contract[k].strip()
                       for k in ("initial", "goal"))
            or not all(isinstance(task_contract[k], list)
                       and all(isinstance(v, str) and v.strip() for v in task_contract[k])
                       for k in ("necessary_steps", "invariants"))):
        raise ValueError("Supply only initial/necessary_steps/goal/invariants; no previous review labels")
    return (
        "Independently inspect ONE anonymous candidate video from the supplied timestamped frames. "
        "No baseline, repair history, prior grade or human verdict is provided. The task description "
        "is intended behavior, not evidence that it occurred. Rate each dimension on an ORDINAL scale: "
        "0=clear severe defect or unusable; 1=major visible problems; 2=mixed/partly readable; "
        "3=good within observed evidence with minor issues; 4=very clear and consistent in the "
        "observed evidence. Use the literal string unknown when evidence is insufficient; unknown "
        "is not 0 or 4. These are not probabilities. Every numeric score and every definite constraint "
        "must cite actual provided frame_index AND exact timestamp_seconds with literal observations. "
        "A high visual score cannot cancel a visible task or continuity violation. Check the first "
        "and last states and necessary event order. Sparse samples cannot establish absence of "
        "short errors; report missing intervals and occlusion explicitly. Never claim measured "
        "physical validity, forces, true contact or 3D ground truth from pixels. Inventory visible "
        "hands at several informative frames, including overlaps and partial hands. More than two "
        "hands need not be anatomically impossible if multiple people are visible; explain evidence, "
        "not assumptions. Do not infer hidden bodies. Return ONLY schema-conforming JSON.\n"
        + json.dumps({"task": task_contract, "dimensions": DIMENSIONS, "schema": SCHEMA},
                     ensure_ascii=False)
    )


def _timestamps(video):
    raw = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True, timeout=120)
    values = [float(f["best_effort_timestamp_time"]) for f in json.loads(raw.stdout)["frames"]]
    if (not values or any(not math.isfinite(v) or v < 0 for v in values)
            or any(b <= a for a, b in zip(values, values[1:]))):
        raise ValueError("Invalid media timestamps")
    return values


def prepare_review(video, output, task_contract):
    from data_generation_methods.scripts.echoflow_repair_study import frame_packet
    video, output = Path(video).resolve(strict=True), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(task_contract)
    media = probe_video(video)
    timestamps = _timestamps(video)
    if media["frames"] not in (None, len(timestamps)):
        raise ValueError("Video frame count mismatch")
    indices = select_uniform_boundary_frames(len(timestamps))
    packet_path = output/"frames/packet.json"
    if not packet_path.exists():
        frame_packet(video, output/"frames", lambda _: indices)
    packet = json.loads(packet_path.read_text())
    if packet["source_sha256"] != media["sha256"]:
        raise ValueError("Frame packet belongs to another video")
    frames = packet["frames"]
    if [f["frame_index"] for f in frames] != indices:
        raise ValueError("Frame sampler changed")
    for frame in frames:
        if file_sha256(frame["path"]) != frame["sha256"]:
            raise ValueError("Prepared frame changed")
        if abs(frame["timestamp_seconds"]-timestamps[frame["frame_index"]]) > 1e-6:
            raise ValueError("Frame PTS changed")
    prepared = {
        "schema": "r2h_blind_quality_inputs_v1", "media": binding(video),
        "frame_packet": binding(packet_path), "frames": frames,
        "task_contract": task_contract, "prompt": prompt, "response_schema": SCHEMA,
        "sampler": "same_boundary4_plus_uniform18_union",
        "frame_count": len(frames), "previous_grade_or_history_sent": False,
        "prepared_only": True,
    }
    write_json(output/"prepared_inputs.json", prepared)
    return prepared


def validate_result(result, frames):
    parsed = parse_review_json(json.dumps(result["review"]), SCHEMA)["review"]
    available = {f["frame_index"]: f["timestamp_seconds"] for f in frames}
    def citations(items, definite):
        if definite and not items:
            raise ValueError("Definite score/constraint lacks actual frame evidence")
        used = set()
        for item in items:
            index, timestamp = item["frame_index"], item["timestamp_seconds"]
            if index not in available or type(index) is not int:
                raise ValueError("Model cited a frame not supplied")
            if (type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                    or abs(timestamp-available[index]) > 1e-5):
                raise ValueError("Model evidence timestamp differs from actual PTS")
            if index in used:
                raise ValueError("Duplicate evidence index")
            used.add(index)
    for item in parsed["dimensions"].values():
        if isinstance(item["score"], bool):
            raise ValueError("Boolean is not an ordinal score")
        citations(item["evidence"], item["score"] != "unknown")
    for item in parsed["constraints"].values():
        citations(item["evidence"], item["status"] != "unknown")
    citations(parsed["visible_hand_inventory"], True)
    return parsed


def review_video(video, output, task_contract, *, config_path=DEFAULT_CONFIG):
    output = Path(output)
    prepared = prepare_review(video, output, task_contract)
    load_env_file(ENV)
    config = json.loads(Path(config_path).read_text())["gemini"]
    def call(transport):
        client = GeminiClient(config, transport=transport)
        parts = [{"text": prepared["prompt"]}]
        for frame in prepared["frames"]:
            parts += [{"text": json.dumps({k: frame[k] for k in
                                           ("frame_index", "timestamp_seconds")})},
                      client._inline(Path(frame["path"]), "image")]
        return client._review_result(
            client._generate(client.model, parts, client._review_config(prepared["prompt"], SCHEMA, True)),
            SCHEMA, True)
    result = paid_stage(output, {
        "kind": "blind_ordinal_video_quality_v1", "model": config["model"],
        "media": prepared["media"], "frames": prepared["frames"],
        "task_contract": task_contract, "prompt": prepared["prompt"],
        "response_schema": SCHEMA, "frame_packet": prepared["frame_packet"],
        "implementation_sha256": file_sha256(__file__),
        "independent_request_without_previous_grades": True,
    }, call)
    parsed = validate_result(result, prepared["frames"])
    hard = [k for k, v in parsed["constraints"].items() if v["status"] == "violation_observed"]
    unknown = [k for k, v in parsed["dimensions"].items() if v["score"] == "unknown"]
    summary = {
        "schema": "r2h_blind_ordinal_quality_review_v1", "media": prepared["media"],
        "raw_response": binding(output/"result.json"), "frame_packet": prepared["frame_packet"],
        "dimensions": parsed["dimensions"], "constraints": parsed["constraints"],
        "visible_hand_inventory": parsed["visible_hand_inventory"],
        "limitations": parsed["limitations"],
        "ordinal_vector": {k: v["score"] for k, v in parsed["dimensions"].items()},
        "hard_visible_violations": hard, "unknown_score_dimensions": unknown,
        "status": "reject_visible_violation" if hard else "needs_independent_evidence_review",
        "overall_average": None, "scores_are_probabilities": False,
        "unobserved_intervals_verified": False, "physical_validity_certified": False,
        "training_admitted": False, "quality_release_approved": False,
        "previous_grade_or_history_sent": False, "new_provider_calls_this_stage_maximum": 1,
    }
    write_json(output/"quality_review.json", summary)
    return summary
