"""Budget-matched A/B preparation and one-step execution. No training.

A: common first image + complete task prompt.
B: same first image/prompt + terminal state image, then targeted retry if needed.
Initial generation and adaptive recovery are separate cohorts. One command
executes at most one paid stage; ambiguous submissions never silently retry.
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import time

from .batch_pipeline import ENV, binding, checked_binding, paid_stage
from .cli import download_video
from .common import file_sha256, json_sha256, load_env_file, probe_video, read_json, write_json
from .providers import GeminiClient, SeedanceClient, parse_review_json
from .quality_model_review import select_uniform_boundary_frames

DEFAULT_BUDGET = {"image": 2, "video": 2, "review": 2}
TERMINAL_VIDEO_FAILURES = {"failed", "expired", "cancelled"}
CHECKS = {
    "initial_task_state": "Does the candidate begin with the required initial relations?",
    "final_task_goal": "Does the intended goal visibly hold after release/withdrawal?",
    "necessary_event_order": "Are the required visible events observed in their required order?",
    "source_required_entities_preserved": "Are required source entities and revealed contents visually retained rather than replaced/removed?",
    "actor_plan": "Does observed hand count respect the declared rendering plan? Count overlapping/partial hands; do not excuse plan deviation by inventing another participant.",
    "interaction_visibility": "Are approach, manipulation and release visibly interpretable rather than missing/ambiguous?",
    "visible_object_continuity": "Are inspected object identities and movements consistent, with no observed jump/duplication?",
}
ITEM = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "value": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "frame_indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "timestamp_seconds": {"type": "array", "items": {"type": "number", "minimum": 0}},
        "observation": {"type": "string", "minLength": 1},
    }, "required": ["value", "frame_indices", "timestamp_seconds", "observation"],
}
REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "checks": {"type": "object", "additionalProperties": False,
                   "properties": {key: ITEM for key in CHECKS}, "required": list(CHECKS)},
        "review_windows": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"start_frame": {"type": "integer", "minimum": 0},
                           "end_frame": {"type": "integer", "minimum": 0},
                           "reason": {"type": "string", "minLength": 1}},
            "required": ["start_frame", "end_frame", "reason"]}},
        "limitations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    }, "required": ["checks", "review_windows", "limitations"],
}


def common_video_prompt(case):
    task = case["task"]
    return (
        "Create one continuous fixed-camera human manipulation video of this intended task. "
        "The provided first image is the initial state. Preserve object identities, camera, "
        "background and all task-relevant geometry. Use natural, clearly visible approach, "
        "manipulation, release and withdrawal. Do not switch source/target roles, add people "
        "or hands, teleport objects, insert cuts, or omit necessary steps. If a terminal "
        "state image is supplied, treat it as the final-state constraint, not as evidence "
        "that the intermediate events may be skipped. Finish with a settled observable outcome. "
        f"Maximum distinct visible hands: {case['maximum_visible_hands']}. "
        "Visible hands must form a plausible, consistent actor arrangement. "
        + json.dumps(task, ensure_ascii=False)
    )


def terminal_image_prompt(case, repair=None):
    prompt = (
        "Edit the FIRST input (a simulation terminal frame) into a human terminal frame. "
        "Supporting reference image is the common human initial frame: match its camera, "
        "scene appearance and actor identity, while preserving the simulation TERMINAL "
        "task relations, object identities, articulated geometry and revealed contents. "
        "Remove robot bodies/grippers, retain human hands already released and withdrawn. "
        f"At most {case['maximum_visible_hands']} distinct visible hands. No montage, text, "
        "new objects, missing required contents, extra actors, or copied initial-state goal error. "
        + json.dumps(case["task"], ensure_ascii=False)
    )
    if repair:
        prompt += "\nPrevious candidate observed problems to correct, without changing the task:\n" + json.dumps(repair, ensure_ascii=False)
    return prompt


def create_experiment(directory, cases, config, *, budget=None, duration=8, resolution="720p"):
    directory = Path(directory)
    budget = dict(DEFAULT_BUDGET if budget is None else budget)
    if (set(budget) != set(DEFAULT_BUDGET)
            or any(type(v) is not int or v < 1 for v in budget.values())):
        raise ValueError("Positive image/video/review budget required")
    if duration != 8 or resolution != "720p":
        raise ValueError("This preregistration currently fixes all candidates to 8s/720p")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate source case ID")
    for case in cases:
        if not case["id"].replace("_", "").isalnum():
            raise ValueError("Simple alphanumeric source IDs required")
        for key in ("human_source", "robot_video", "robot_trajectory", "sim_initial",
                    "sim_terminal", "human_initial", "source_record"):
            checked_binding(case[key])
        if type(case.get("maximum_visible_hands")) is not int or case["maximum_visible_hands"] < 1:
            raise ValueError("An explicit common actor plan is required")
        if set(case["task"]) != {"initial", "necessary_steps", "goal", "invariants"}:
            raise ValueError("Task contract fields are incomplete")
    manifest = {
        "schema": "r2h_controlled_experiment_v1",
        "cases": cases, "groups": {
            "A": "common first frame and complete task prompt; equal-budget redraw on failure",
            "B": "same first frame and task prompt plus terminal condition; evidence-targeted retry"},
        "config": config, "per_case_per_group_call_budget": budget,
        "generation_settings": {"duration": duration, "resolution": resolution,
                                "ratio": "adaptive", "generate_audio": False},
        "initial_comparison": {
            "candidate_count": len(cases)*2, "common_prompt": True,
            "common_initial_image": True, "common_video_model": True,
            "common_review_model_and_frame_rule": True,
            "terminal_condition_is_the_initial_treatment": True,
            "contrast_scope": "whole condition-construction pipeline, not isolated prompt or backbone causality",
            "initial_and_recovery_results_must_be_reported_separately": True},
        "recovery_comparison": {
            "attempts_maximum": min(budget["video"], 2),
            "A": "same original request, new paid generation only after its review has no/unknown",
            "B": "append observed failure constraints; optionally revise terminal image when entity/goal evidence failed",
            "initial_outcomes_retained": True, "actual_calls_and_usage_reported": True},
        "cost_scope": {
            "common_existing_human_initial": "reused sunk cost, provenance recorded; not claimed free to reproduce",
            "new_B_terminal_edit": "charged to B image budget",
            "budget_means_equal_caps_not_forced_equal_spend": True,
            "monetary_cost": "unknown unless actual provider billing is supplied",
        },
        "limitations": [
            "Three source tasks are an engineering pilot, not statistical superiority.",
            "Provider stochastic seeds are not assumed supported or paired.",
            "No separate paid static-anchor review is included under review=2; only local image decoding precedes generation.",
            "Post-generation event evidence is not physical truth or training release.",
            "B retries may change prompt, terminal image and review allocation; do not attribute recovery to one factor.",
        ],
        "training_admitted": False,
    }
    write_json(directory/"experiment.json", manifest)
    for case in cases:
        for arm in ("A", "B"):
            write_json(directory/case["id"]/arm/"initial_request_plan.json", {
                "case_id": case["id"], "group": arm,
                "common_first_frame": case["human_initial"],
                "prompt": common_video_prompt(case),
                "model": config["seedance"]["model"],
                "roles": ["first_frame"] + (["last_frame"] if arm == "B" else []),
                "settings": manifest["generation_settings"],
                "terminal_image_request": None if arm == "A" else {
                    "model": config["anchor_model"], "target": case["sim_terminal"],
                    "reference": case["human_initial"], "prompt": terminal_image_prompt(case)},
                "budget": budget, "provider_called": False,
            })
    return manifest


def call_counts(arm_directory):
    counts = {key: 0 for key in DEFAULT_BUDGET}
    for marker in Path(arm_directory).glob("stages/*/call_started.json"):
        metadata = read_json(marker.parent/"request_metadata.json")
        kind = metadata.get("experiment_call_kind")
        if kind not in counts:
            raise ValueError("Unknown paid stage in experiment budget")
        counts[kind] += 1
    return counts


def budgeted_paid(arm_directory, stage_name, metadata, operation, budget):
    directory = Path(arm_directory)/"stages"/stage_name
    kind = metadata["experiment_call_kind"]
    counts = call_counts(arm_directory)
    if not (directory/"call_started.json").exists() and counts[kind] >= budget[kind]:
        raise RuntimeError(f"{kind} budget exhausted")
    return paid_stage(directory, metadata, operation)


def _case(manifest, case_id):
    cases = [c for c in manifest["cases"] if c["id"] == case_id]
    if len(cases) != 1:
        raise ValueError("Unknown case")
    return cases[0]


def _existing_review(arm_dir, attempt):
    path = arm_dir/f"attempt_{attempt:02d}"/"review_summary.json"
    return read_json(path) if path.exists() else None


def retry_issues(review):
    if not review:
        raise ValueError("Adaptive retry requires a completed prior review")
    return {k: v for k, v in review["checks"].items() if v["value"] != "yes"}


def seed_policy(manifest):
    requested = manifest["generation_settings"].get("seed")
    return {
        "requested_seed": requested,
        "policy": "explicit_request" if requested is not None else "not_specified",
        "paired_generator_randomness_claimed": False,
        "reason": ("A common seed does not guarantee matched noise across different condition inputs."
                   if requested is not None else
                   "Seedance 2.5 relay seed control has not been verified; omit instead of assuming support."),
    }


def nominal_resolution_evidence(settings, first_image, actual):
    """Check a documented preset without equating the string '720p' with height."""
    from PIL import Image
    with Image.open(first_image) as image:
        first_width, first_height = image.size
    first_ratio = first_width/first_height
    # Official 2.5 guide, checked 2026-10-20. Arbitrary adaptive ratios are left unknown.
    preset_720p = {
        (16,9): (1280,720), (4,3): (1112,834), (1,1): (960,960),
        (3,4): (834,1112), (9,16): (720,1280), (21,9): (1470,630),
    }
    matches = [size for ratio,size in preset_720p.items()
               if abs(first_ratio-ratio[0]/ratio[1]) < 1e-6]
    expected = list(matches[0]) if settings["resolution"] == "720p" and len(matches) == 1 else None
    observed = [actual["width"], actual["height"]]
    return {
        "source": "https://docs.volcengine.com/docs/ark/seedance-2-5?lang=zh",
        "source_checked_on": "2026-10-20",
        "first_frame_dimensions": [first_width, first_height],
        "actual_dimensions": observed,
        "expected_dimensions_if_documented_standard_ratio": expected,
        "matches_documented_preset": observed == expected if expected else None,
        "first_frame_aspect_ratio_preserved": abs(actual["width"]/actual["height"]-first_ratio) < .002,
        "uncertainty": "A matching raster size does not independently verify the backend model identity.",
    }


def latest_video_response(arm_dir, attempt):
    """Retain terminal/unknown states instead of displaying stale 'running'."""
    attempt_dir = Path(arm_dir)/f"attempt_{attempt:02d}"
    polls = sorted((attempt_dir/"polls").glob("*/normalized_response.json"))
    if polls:
        return read_json(polls[-1]), polls[-1]
    stage = Path(arm_dir)/"stages"/f"video_{attempt:02d}"/"result.json"
    return (read_json(stage), stage) if stage.exists() else (None, None)


def plan_next(directory, case_id, arm, *, attempt=1):
    root = Path(directory)
    manifest = read_json(root/"experiment.json")
    case = _case(manifest, case_id)
    if arm not in ("A", "B") or attempt not in (1, 2):
        raise ValueError("Only groups A/B and attempts 1/2 are preregistered")
    arm_dir = root/case_id/arm
    previous = _existing_review(arm_dir, 1) if attempt == 2 else None
    issues = retry_issues(previous) if attempt == 2 else {}
    if attempt == 2 and not issues:
        return {"operation": "stop_initial_visible_checks_supported", "paid": False}
    # An unresolved paid operation is never escaped by creating another version.
    for marker in (arm_dir/"stages").glob("*/call_started.json"):
        if not (marker.parent/"result.json").exists():
            return {"operation": "blocked_unresolved_paid_stage", "stage": str(marker.parent),
                    "paid": False, "automatic_retry": False}
    terminal_version = 1
    if arm == "B" and attempt == 2 and any(k in issues for k in (
            "source_required_entities_preserved", "final_task_goal")):
        terminal_version = 2
    terminal = arm_dir/f"human_terminal_{terminal_version:02d}.png"
    if arm == "B" and not terminal.exists():
        return {"operation": "edit_terminal", "paid": True, "terminal_version": terminal_version,
                "attempt": attempt, "repair_issues": issues if terminal_version == 2 else {}}
    stage = arm_dir/"stages"/f"video_{attempt:02d}"
    video = arm_dir/f"attempt_{attempt:02d}"/"candidate.mp4"
    if not (stage/"result.json").exists():
        return {"operation": "submit_video", "paid": True, "attempt": attempt,
                "terminal_version": terminal_version, "repair_issues": issues if arm == "B" else {}}
    if not video.exists():
        latest, evidence_path = latest_video_response(arm_dir, attempt)
        status = latest.get("status", "submitted")
        if status in TERMINAL_VIDEO_FAILURES or status == "unknown":
            return {"operation": ("stop_provider_terminal_failure" if status in TERMINAL_VIDEO_FAILURES
                                  else "blocked_provider_status_unknown"),
                    "paid": False, "attempt": attempt, "status": status,
                    "task_id": latest["task_id"], "evidence": binding(evidence_path),
                    "automatic_retry": False,
                    "next_action": "Inspect the existing task and raw receipt; do not create another POST."}
        return {"operation": "poll_video", "paid": False, "attempt": attempt,
                "task_id": read_json(stage/"result.json")["task_id"]}
    if not _existing_review(arm_dir, attempt):
        return {"operation": "review_video", "paid": True, "attempt": attempt}
    return {"operation": "complete_attempt", "paid": False, "attempt": attempt,
            "result": _existing_review(arm_dir, attempt)}


def video_payload(manifest, case, arm, arm_dir, plan):
    prompt = common_video_prompt(case)
    if arm == "B" and plan.get("repair_issues"):
        prompt += ("\nCorrect these independently observed problems without changing the task, "
                   "object roles or the actor plan:\n" + json.dumps(plan["repair_issues"], ensure_ascii=False))
    content = [{"type": "text", "text": prompt}]
    images = [("first_frame", Path(case["human_initial"]["path"]))]
    if arm == "B":
        images.append(("last_frame", arm_dir/f"human_terminal_{plan['terminal_version']:02d}.png"))
    for role, path in images:
        content.append({"type": "image_url", "role": role,
                        "image_url": {"url": "data:image/png;base64," +
                                      base64.b64encode(path.read_bytes()).decode()}})
    return {"model": manifest["config"]["seedance"]["model"], "content": content,
            **manifest["generation_settings"]}


def select_review_frames(count, previous_windows=(), *, budget=24):
    base = select_uniform_boundary_frames(count)
    if not previous_windows:
        return base[:budget]
    # Old candidate intervals only guide attention; they are not new-video event GT.
    required = {i for i in (0,1,2,3,count-4,count-3,count-2,count-1) if 0 <= i < count}
    focus = []
    for window in previous_windows[:2]:
        start, end = window["start_frame"], window["end_frame"]
        if not (type(start) is int and type(end) is int and 0 <= start <= end):
            raise ValueError("Bad review attention window")
        center = min(count-1, (start+end)//2)
        focus += [i for i in range(max(0, center-2), min(count, center+3))]
    selected = set(required)
    for index in focus:
        if len(selected) < budget:
            selected.add(index)
    for index in base:
        if len(selected) < budget:
            selected.add(index)
    return sorted(selected)


def validate_task_review(result, frames, count):
    parsed = parse_review_json(json.dumps(result["review"]), REVIEW_SCHEMA)["review"]
    stamps = {f["frame_index"]: f["timestamp_seconds"] for f in frames}
    for item in parsed["checks"].values():
        indices, timestamps = item["frame_indices"], item["timestamp_seconds"]
        if len(indices) != len(timestamps) or len(indices) != len(set(indices)):
            raise ValueError("Invalid cited frame/time pairs")
        if item["value"] != "unknown" and not indices:
            raise ValueError("Definite judgement lacks evidence")
        if any(i not in stamps or type(t) not in (int,float) or not math.isfinite(t)
               or abs(t-stamps[i]) > 1e-5 for i,t in zip(indices,timestamps)):
            raise ValueError("Review invented a frame or timestamp")
    for window in parsed["review_windows"]:
        if not 0 <= window["start_frame"] <= window["end_frame"] < count:
            raise ValueError("Review window lies outside media")
    return parsed


def _execute_next_unlocked(directory, case_id, arm, *, attempt=1):
    from PIL import Image
    from data_generation_methods.scripts.echoflow_repair_study import frame_packet
    root = Path(directory)
    manifest = read_json(root/"experiment.json")
    case = _case(manifest, case_id)
    for key in ("human_source","robot_video","robot_trajectory","sim_initial",
                "sim_terminal","human_initial","source_record"):
        checked_binding(case[key])
    arm_dir = root/case_id/arm
    plan = plan_next(root, case_id, arm, attempt=attempt)
    operation = plan["operation"]
    if operation.startswith(("stop_", "blocked_", "complete_")):
        return plan
    load_env_file(ENV)
    config, budget = manifest["config"], manifest["per_case_per_group_call_budget"]
    base_meta = {"case_id": case_id, "group": arm, "experiment": binding(root/"experiment.json")}
    if operation == "edit_terminal":
        version = plan["terminal_version"]
        prompt = terminal_image_prompt(case, plan["repair_issues"])
        result = budgeted_paid(arm_dir, f"image_{version:02d}", {
            **base_meta, "experiment_call_kind": "image", "model": config["anchor_model"],
            "target": case["sim_terminal"], "reference": case["human_initial"], "prompt": prompt,
        }, lambda transport: GeminiClient(config["gemini"], transport=transport).generate_image_with_references(
            Path(case["sim_terminal"]["path"]), [Path(case["human_initial"]["path"])],
            prompt, model=config["anchor_model"]), budget)
        image = Image.open(io.BytesIO(base64.b64decode(result["images"][0]["data"], validate=True)))
        image.load()
        image.save(arm_dir/f"human_terminal_{version:02d}.png")
        return {"operation": operation, "completed": True, "semantic_anchor_precheck": "not_run"}
    if operation == "submit_video":
        payload = video_payload(manifest, case, arm, arm_dir, plan)
        SeedanceClient(config["seedance"]).prepare_submit(payload)
        result = budgeted_paid(arm_dir, f"video_{attempt:02d}", {
            **base_meta, "experiment_call_kind": "video", "model": config["seedance"]["model"],
            "attempt": attempt, "payload_sha256": json_sha256(payload),
            "prompt": payload["content"][0]["text"], "settings": manifest["generation_settings"],
            "seed_policy": seed_policy(manifest),
            "first": case["human_initial"],
            "terminal": binding(arm_dir/f"human_terminal_{plan['terminal_version']:02d}.png") if arm == "B" else None,
        }, lambda transport: SeedanceClient(dict(config["seedance"], timeout=600),
                                            transport=transport).submit(payload), budget)
        return {"operation": operation, "task_id": result["task_id"], "completed": True}
    target = arm_dir/f"attempt_{attempt:02d}"/"candidate.mp4"
    if operation == "poll_video":
        from data_generation_methods.scripts.echoflow_pilot import diagnostic_transport
        poll_dir = arm_dir/f"attempt_{attempt:02d}"/"polls"/str(time.time_ns())
        poll_dir.mkdir(parents=True, exist_ok=False)
        response = SeedanceClient(dict(config["seedance"], timeout=30),
                                  transport=diagnostic_transport(poll_dir)).poll(plan["task_id"])
        write_json(poll_dir/"normalized_response.json", response)
        if response["status"] == "succeeded":
            stage = arm_dir/"stages"/f"video_{attempt:02d}"
            download_video(response["output_url"], target, task_id=plan["task_id"],
                           request_sha256=file_sha256(stage/"request_metadata.json"))
            actual = probe_video(target)
            raw_response = read_json(poll_dir/"provider_response.json")
            write_json(target.parent/"media.json", actual)
            write_json(target.parent/"api_effect_validation.json", {
                "schema": "r2h_requested_vs_actual_media_v1",
                "requested_model_alias": config["seedance"]["model"],
                "requested_settings": manifest["generation_settings"],
                "actual_media": actual, "accepted_task_id": plan["task_id"],
                "provider_raw_response": binding(poll_dir/"provider_response.json"),
                "provider_echoed_parameters": {key: raw_response[key] for key in
                                                ("model","resolution","ratio","duration","framespersecond",
                                                 "frames","seed","generate_audio")
                                                if key in raw_response},
                "provider_usage": raw_response.get("usage", {}),
                "seed_policy": seed_policy(manifest),
                "resolution_evidence": nominal_resolution_evidence(
                    manifest["generation_settings"], case["human_initial"]["path"], actual),
                "duration_compatible_with_one_frame_rounding": (
                    abs(actual["duration_seconds"]-manifest["generation_settings"]["duration"])
                    <= 1/actual["fps"]+.01),
                "model_backend_identity": "not_independently_verified_by_normalized_client_alias",
                "resolution_preset_semantics": "requested nominal preset; actual width/height recorded separately",
                "actual_use_of_condition_content": "requires output evidence, not proven by request acceptance",
            })
        return {"operation": operation, "status": response["status"], "task_id": plan["task_id"],
                "automatic_retry": False,
                "terminal_failure": response["status"] in TERMINAL_VIDEO_FAILURES,
                "status_unknown": response["status"] == "unknown"}
    media = probe_video(target)
    review_dir = target.parent/"review"
    previous = _existing_review(arm_dir, 1) if arm == "B" and attempt == 2 else None
    windows = previous["review_windows"] if previous else []
    indices = select_review_frames(media["frames"], windows)
    packet_path = review_dir/"packet.json"
    if packet_path.exists():
        packet = read_json(packet_path)
        if packet["source_sha256"] != media["sha256"]:
            raise ValueError("Review packet media changed")
        frames = packet["frames"]
    else:
        frames = frame_packet(target, review_dir, lambda _: indices)
    if [f["frame_index"] for f in frames] != indices:
        raise ValueError("Review packet selection changed")
    for frame in frames:
        checked_binding(frame)
    prompt = (
        "Independently inspect this ONE anonymous candidate. No group, prior success/failure "
        "grade or method identity is supplied. Two labelled STATIC SIMULATION images show "
        "source initial and terminal task states; they are not candidate video frames. Then "
        "inspect the timestamped candidate frames. Intended requirements are not evidence "
        "they occurred. Judge each query yes/no/unknown using only cited supplied video frame "
        "indices and their exact PTS. Hidden/occluded or temporally missing evidence means "
        "unknown. Do not claim full-clip absence, real contact forces, true 3D or physical "
        "certification. Preserve object roles and compare revealed contents to the source "
        "terminal image. Extra plausible people do not excuse exceeding the declared hand "
        "budget. For actor_plan, state the minimum visually supported hand count in the "
        "observation, including partially visible or overlapping hands; distinguish inferred "
        "participant identity from visible hand evidence. A value of yes refers only to the "
        "supplied frames, not every frame in the clip. Report bounded intervals needing closer "
        "examination. No overall average.\n"
        + json.dumps({"task": case["task"], "maximum_visible_hands": case["maximum_visible_hands"],
                      "queries": CHECKS, "schema": REVIEW_SCHEMA}, ensure_ascii=False))
    def call(transport):
        client = GeminiClient(config["gemini"], transport=transport)
        parts = [{"text": prompt}]
        for key in ("sim_initial","sim_terminal"):
            parts += [{"text": "STATIC SOURCE "+key.upper()},
                      client._inline(Path(case[key]["path"]), "image")]
        for frame in frames:
            parts += [{"text": json.dumps({k:frame[k] for k in ("frame_index","timestamp_seconds")})},
                      client._inline(Path(frame["path"]), "image")]
        return client._review_result(client._generate(
            client.model, parts, client._review_config(prompt, REVIEW_SCHEMA, True)), REVIEW_SCHEMA, True)
    result = budgeted_paid(arm_dir, f"review_{attempt:02d}", {
        **base_meta, "experiment_call_kind": "review", "model": config["gemini"]["model"],
        "video": binding(target), "source_initial": case["sim_initial"],
        "source_terminal": case["sim_terminal"], "prompt": prompt,
        "frames": frames, "response_schema": REVIEW_SCHEMA,
    }, call, budget)
    parsed = validate_task_review(result, frames, media["frames"])
    summary = {
        "schema": "r2h_controlled_review_summary_v1", "video": binding(target),
        "checks": parsed["checks"], "review_windows": parsed["review_windows"],
        "limitations": parsed["limitations"],
        "status": ("visible_violation" if any(v["value"]=="no" for v in parsed["checks"].values())
                   else "needs_evidence" if any(v["value"]=="unknown" for v in parsed["checks"].values())
                   else "visible_contract_supported_in_sampled_scope"),
        "review_result": binding(arm_dir/"stages"/f"review_{attempt:02d}"/"result.json"),
        "frame_packet": binding(packet_path), "frame_indices": indices,
        "quality_release_approved": False, "training_admitted": False,
    }
    write_json(target.parent/"review_summary.json", summary)
    return {"operation": operation, "status": summary["status"], "completed": True}


def execute_next(directory, case_id, arm, *, attempt=1):
    """Serialize each arm's budget/read-modify-write operations across processes."""
    import fcntl
    root = Path(directory)
    manifest = read_json(root/"experiment.json")
    _case(manifest, case_id)
    if arm not in ("A", "B"):
        raise ValueError("Unknown group")
    arm_dir = root/case_id/arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    with (arm_dir/".execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _execute_next_unlocked(directory, case_id, arm, attempt=attempt)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def summarize(directory):
    directory = Path(directory)
    manifest = read_json(directory/"experiment.json")
    rows = []
    for case in manifest["cases"]:
        for arm in ("A","B"):
            root = directory/case["id"]/arm
            stages = []
            for result_path in sorted((root/"stages").glob("*/result.json")):
                result, meta = read_json(result_path), read_json(result_path.parent/"request_metadata.json")
                stages.append({"stage": result_path.parent.name, "kind": meta["experiment_call_kind"],
                               "usage": result.get("usage", {}), "elapsed_seconds": result.get("elapsed_seconds"),
                               "provider_reported_model_version": result.get("provider_reported_model_version"),
                               "result": binding(result_path)})
            actual_media = {}
            for index in (1,2):
                media_path = root/f"attempt_{index:02d}"/"api_effect_validation.json"
                if media_path.exists():
                    actual_media[str(index)] = read_json(media_path)
            rows.append({"case_id": case["id"], "arm": arm, "call_counts": call_counts(root),
                         "initial": _existing_review(root,1), "recovery": _existing_review(root,2),
                         "actual_media": actual_media, "stages": stages,
                         "video_task_status": {
                             str(i): (latest_video_response(root,i)[0] or {}).get("status")
                             for i in (1,2)}})
    return {"schema": "r2h_controlled_summary_v1", "records": rows,
            "initial_and_recovery_separate": True, "statistics_not_claimed": True,
            "seed_policy": seed_policy(manifest),
            "quality_release_approved": False, "training_admitted": False}
