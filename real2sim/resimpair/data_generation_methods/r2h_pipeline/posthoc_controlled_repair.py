"""Explicit new-content repairs using the initial experiment's remaining budget.

This is NOT recovery/reissue of an unknown POST. Original stages stay unchanged.
No operation runs automatically; every paid operation requires an explicit CLI
command. Parent-arm and all descendant call markers share the same caps.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import time

from .batch_pipeline import ENV, binding, checked_binding, paid_stage
from .cli import download_video
from .common import file_sha256, json_sha256, load_env_file, probe_video, read_json, write_json
from .controlled_experiment import (
    CHECKS, REVIEW_SCHEMA, nominal_resolution_evidence, seed_policy,
    select_review_frames, validate_task_review,
)
from .providers import GeminiClient, SeedanceClient

BRANCH_NAME = "posthoc_remaining_budget_01"
ALLOWED = {("apple", "B"): "apple_B_boundary_failure.json",
           ("cabinet", "A"): "cabinet_A_hidden_identity_failure.json"}


def aggregate_counts(parent_arm):
    counts = {"image": 0, "video": 0, "review": 0}
    for marker in Path(parent_arm).rglob("call_started.json"):
        metadata = read_json(marker.parent/"request_metadata.json")
        kind = metadata.get("experiment_call_kind")
        if kind not in counts:
            raise ValueError("An unclassified paid marker blocks budget accounting")
        counts[kind] += 1
    return counts


def apple_state_delta_prompt():
    return (
        "Edit the supplied HUMAN INITIAL image into the completed HUMAN FINAL state of "
        "one apple transfer. Use this image itself as the edit canvas, keeping its camera, "
        "table, background, lighting, both people/hands and all non-target objects fixed. "
        "The small red apple is currently INSIDE THE IMAGE-LEFT DARK BOWL. The "
        "IMAGE-RIGHT DARK BOWL is currently empty. Make this exact state change: remove "
        "the same red apple from the LEFT bowl and place it, released and resting inside "
        "the RIGHT bowl. The LEFT bowl must be EMPTY; the RIGHT bowl must contain exactly "
        "that one red apple. Keep both bowls in exactly their original positions and "
        "orientations. Keep the small green pear stationary. Keep the same TWO resting "
        "hands withdrawn on the table, not touching the apple or either bowl. No new hands, "
        "objects, duplicate apple, table movement, camera movement or robot parts. "
        "Return one image only. Do not return the unchanged initial state."
    )


def prepare_branch(experiment, case_id, arm):
    experiment = Path(experiment).resolve()
    if (case_id, arm) not in ALLOWED:
        raise ValueError("Only the explicitly requested apple B and cabinet A repairs are supported")
    parent = read_json(experiment/"experiment.json")
    case = next(c for c in parent["cases"] if c["id"] == case_id)
    parent_arm = experiment/case_id/arm
    parent_video = parent_arm/"attempt_01/candidate.mp4"
    local_failure = experiment/"audit_initial"/ALLOWED[(case_id, arm)]
    failure = read_json(local_failure)
    if failure["video"]["sha256"] != file_sha256(parent_video):
        raise ValueError("Local repair trigger is not bound to the initial video")
    prior_review_meta = parent_arm/"stages/review_01/request_metadata.json"
    prior_review_failure = parent_arm/"stages/review_01/failure.json"
    status = read_json(prior_review_failure)
    expected_code = "submission_unknown" if case_id == "apple" else "review_schema_violation"
    if status.get("code") != expected_code:
        raise ValueError("Original review failure differs from the authorized repair context")
    prior_video_meta = read_json(parent_arm/"stages/video_01/request_metadata.json")
    branch = parent_arm/BRANCH_NAME
    caps = parent["per_case_per_group_call_budget"]
    counts = aggregate_counts(parent_arm)
    fresh = {"image": int(arm == "B"), "video": 1, "review": 1}
    if not (branch/"branch.json").exists() and any(counts[k]+fresh[k] > caps[k] for k in counts):
        raise ValueError("Insufficient parent budget for the proposed new-content repair")
    record = {
        "schema": "r2h_posthoc_remaining_budget_v1", "case_id": case_id, "arm": arm,
        "parent_experiment": binding(experiment/"experiment.json"),
        "parent_arm": str(parent_arm), "parent_initial_video": binding(parent_video),
        "local_visual_trigger": binding(local_failure),
        "original_review_request": binding(prior_review_meta),
        "original_review_failure": binding(prior_review_failure),
        "original_review_response_not_reissued": True,
        "original_review_stage_not_modified": True,
        "posthoc_not_preregistered_initial_result": True,
        "parent_and_all_descendants_combined_caps": caps,
        "this_branch_new_call_caps": fresh,
        "case": case, "config": parent["config"],
        "generation_settings": parent["generation_settings"],
        "seed_policy": seed_policy(parent),
        "video_prompt": prior_video_meta["prompt"],
        "review_prompt": read_json(prior_review_meta)["prompt"],
        "review_schema": REVIEW_SCHEMA,
        "image_prompt": apple_state_delta_prompt() if arm == "B" else None,
        "image_condition": ("One common human initial edit target; explicit visually grounded "
                            "left-source/right-target delta; sim terminal is bound as provenance "
                            "but not uploaded again." if arm == "B" else None),
        "change": ("Replace the incorrect terminal anchor; keep the original video prompt."
                   if arm == "B" else
                   "Redraw with the exact original first frame, video prompt and settings."),
        "requires_explicit_terminal_acceptance_before_video": arm == "B",
        "training_admitted": False,
    }
    write_json(branch/"branch.json", record)
    write_json(branch/"image_request_plan.json", {
        "provider_called": False, "new_call_required": arm == "B",
        "model": parent["config"]["anchor_model"] if arm == "B" else None,
        "edit_target": case["human_initial"] if arm == "B" else None,
        "supporting_images": [], "prompt": record["image_prompt"],
        "parent_budget_remaining_before_branch": {k:caps[k]-counts[k] for k in counts},
    })
    write_json(branch/"video_request_plan.json", {
        "provider_called": False, "model": parent["config"]["seedance"]["model"],
        "settings": parent["generation_settings"], "seed_policy": record["seed_policy"],
        "prompt": record["video_prompt"], "common_first": case["human_initial"],
        "last_frame": "new human_terminal.png, after explicit local acceptance" if arm == "B" else None,
        "unchanged_prompt_from_original_request": True,
    })
    return {"branch": str(branch), "provider_called": False, "new_call_caps": fresh}


def _load(branch):
    branch = Path(branch).resolve()
    manifest = read_json(branch/"branch.json")
    for key in ("parent_experiment", "parent_initial_video", "local_visual_trigger",
                "original_review_request", "original_review_failure"):
        checked_binding(manifest[key])
    for key in ("human_initial", "sim_initial", "sim_terminal"):
        checked_binding(manifest["case"][key])
    if branch.parent != Path(manifest["parent_arm"]):
        raise ValueError("The repair branch must remain inside its bound parent arm")
    return branch, manifest


def _new_paid(branch, manifest, kind, name, metadata, operation):
    stage = branch/"stages"/name
    # Only pre-existing unknown parent review is permitted by this explicit new-content
    # protocol. Any unresolved operation within the new branch blocks further work.
    original_failed_review = Path(manifest["original_review_failure"]["path"]).parent
    for marker in Path(manifest["parent_arm"]).rglob("call_started.json"):
        if marker.parent == original_failed_review:
            # This exact old request is accounted for, retained, and explicitly not
            # reissued. The authorized branch acts on new image/video content.
            continue
        if not (marker.parent/"result.json").exists():
            raise RuntimeError("Unresolved repair stage; no automatic reissue or further paid call")
    total = aggregate_counts(manifest["parent_arm"])
    local = aggregate_counts(branch)
    if not (stage/"call_started.json").exists():
        if total[kind] >= manifest["parent_and_all_descendants_combined_caps"][kind]:
            raise RuntimeError("Combined parent/repair budget exhausted")
        if local[kind] >= manifest["this_branch_new_call_caps"][kind]:
            raise RuntimeError("Repair branch budget exhausted")
    return paid_stage(stage, {
        "experiment_call_kind": kind, "branch": binding(branch/"branch.json"),
        "new_content_not_retry_of_original_review": True, **metadata,
    }, operation)


def accept_terminal(branch, *, reviewer, reason):
    branch, manifest = _load(branch)
    if manifest["arm"] != "B" or not reviewer.strip() or not reason.strip():
        raise ValueError("Explicit reviewer and visual evidence are required for B terminal acceptance")
    terminal = branch/"human_terminal.png"
    write_json(branch/"terminal_acceptance.json", {
        "terminal": binding(terminal), "reviewer": reviewer, "reason": reason,
        "scope": "Static endpoint state and actor plan only; not a video/physics approval.",
        "source": manifest["case"]["human_initial"],
        "expected_delta": "left bowl empty, same apple in right bowl, other scene/actors retained",
    })
    return {"accepted_terminal": binding(terminal), "provider_called": False}


def _video_payload(branch, manifest):
    images = [("first_frame", Path(manifest["case"]["human_initial"]["path"]))]
    if manifest["arm"] == "B":
        acceptance = read_json(branch/"terminal_acceptance.json")
        checked_binding(acceptance["terminal"])
        if Path(acceptance["terminal"]["path"]) != branch/"human_terminal.png":
            raise ValueError("Terminal acceptance belongs to another image")
        images.append(("last_frame", branch/"human_terminal.png"))
    content = [{"type":"text","text":manifest["video_prompt"]}]
    for role, path in images:
        content.append({"type":"image_url", "role":role,
                        "image_url":{"url":"data:image/png;base64,"+base64.b64encode(path.read_bytes()).decode()}})
    return {"model":manifest["config"]["seedance"]["model"], "content":content,
            **manifest["generation_settings"]}


def _execute(branch, operation):
    from PIL import Image
    branch, manifest = _load(branch)
    config = manifest["config"]
    load_env_file(ENV)
    if operation == "image":
        if manifest["arm"] != "B":
            raise ValueError("A redraw has no image call")
        result = _new_paid(branch,manifest,"image","image_02",{
            "model":config["anchor_model"], "target":manifest["case"]["human_initial"],
            "supporting_images":[], "prompt":manifest["image_prompt"],
        },lambda transport:GeminiClient(config["gemini"],transport=transport).generate_image(
            Path(manifest["case"]["human_initial"]["path"]),manifest["image_prompt"],
            model=config["anchor_model"]))
        image = Image.open(io.BytesIO(base64.b64decode(result["images"][0]["data"],validate=True)))
        image.load()
        target = branch/"human_terminal.png"
        if not target.exists():
            image.save(target)
        return {"operation":"image","image":binding(target),
                "next_action":"Stop for explicit terminal inspection; video is not submitted."}
    if operation == "video":
        payload = _video_payload(branch,manifest)
        SeedanceClient(config["seedance"]).prepare_submit(payload)
        result = _new_paid(branch,manifest,"video","video_02",{
            "model":config["seedance"]["model"],"payload_sha256":json_sha256(payload),
            "settings":manifest["generation_settings"],"prompt":manifest["video_prompt"],
            "first":manifest["case"]["human_initial"],
            "terminal":binding(branch/"human_terminal.png") if manifest["arm"]=="B" else None,
        },lambda transport:SeedanceClient(dict(config["seedance"],timeout=600),
                                         transport=transport).submit(payload))
        return {"operation":"video","task_id":result["task_id"]}
    video = branch/"candidate.mp4"
    if operation == "poll":
        from data_generation_methods.scripts.echoflow_pilot import diagnostic_transport
        submitted = read_json(branch/"stages/video_02/result.json")
        polls = sorted((branch/"polls").glob("*/normalized_response.json"))
        if polls:
            last = read_json(polls[-1])
            if last["status"] in {"failed","cancelled","expired","unknown"}:
                return {"operation":"stop_existing_task_status", "status":last["status"],
                        "automatic_retry":False}
        if video.exists():
            return {"operation":"video_already_downloaded", "video":binding(video)}
        poll_dir = branch/"polls"/str(time.time_ns())
        poll_dir.mkdir(parents=True,exist_ok=False)
        result = SeedanceClient(dict(config["seedance"],timeout=30),
                                transport=diagnostic_transport(poll_dir)).poll(submitted["task_id"])
        write_json(poll_dir/"normalized_response.json",result)
        if result["status"] == "succeeded":
            download_video(result["output_url"],video,task_id=submitted["task_id"],
                           request_sha256=file_sha256(branch/"stages/video_02/request_metadata.json"))
            actual=probe_video(video);raw=read_json(poll_dir/"provider_response.json")
            write_json(branch/"media.json",actual)
            write_json(branch/"api_effect_validation.json",{
                "requested_settings":manifest["generation_settings"],
                "requested_model":config["seedance"]["model"], "actual_media":actual,
                "seed_policy":manifest["seed_policy"], "provider_raw":binding(poll_dir/"provider_response.json"),
                "provider_echoed_parameters":{k:raw[k] for k in
                    ("model","resolution","duration","ratio","frames","framespersecond","seed") if k in raw},
                "provider_usage":raw.get("usage",{}),
                "resolution_evidence":nominal_resolution_evidence(
                    manifest["generation_settings"],manifest["case"]["human_initial"]["path"],actual),
            })
        return {"operation":"poll","status":result["status"],"task_id":submitted["task_id"]}
    if operation != "review":
        raise ValueError("Unknown operation")
    from data_generation_methods.scripts.echoflow_repair_study import frame_packet
    media=probe_video(video)
    if media["sha256"] == manifest["parent_initial_video"]["sha256"]:
        raise ValueError("New-content review cannot silently re-review the old candidate")
    folder=branch/"review";packet=folder/"packet.json";indices=select_review_frames(media["frames"])
    if packet.exists():
        cached=read_json(packet)
        if cached["source_sha256"] != media["sha256"]:
            raise ValueError("Review packet belongs to different media")
        frames=cached["frames"]
    else:
        frames=frame_packet(video,folder,lambda _:indices)
    if [f["frame_index"] for f in frames] != indices:
        raise ValueError("Uniform 24-frame review protocol changed")
    for frame in frames:
        checked_binding(frame)
    def review_call(transport):
        client=GeminiClient(config["gemini"],transport=transport)
        parts=[{"text":manifest["review_prompt"]}]
        for key in ("sim_initial","sim_terminal"):
            parts += [{"text":"STATIC SOURCE "+key.upper()},
                      client._inline(Path(manifest["case"][key]["path"]),"image")]
        for frame in frames:
            parts += [{"text":json.dumps({k:frame[k] for k in ("frame_index","timestamp_seconds")})},
                      client._inline(Path(frame["path"]),"image")]
        return client._review_result(client._generate(
            client.model,parts,client._review_config(manifest["review_prompt"],REVIEW_SCHEMA,True)),
            REVIEW_SCHEMA,True)
    result=_new_paid(branch,manifest,"review","review_02",{
        "model":config["gemini"]["model"],"video":binding(video),"frames":frames,
        "prompt":manifest["review_prompt"],"schema":REVIEW_SCHEMA,
        "source_initial":manifest["case"]["sim_initial"],"source_terminal":manifest["case"]["sim_terminal"],
    },review_call)
    parsed=validate_task_review(result,frames,media["frames"])
    summary={
        "video":binding(video),"checks":parsed["checks"],"limitations":parsed["limitations"],
        "review_windows":parsed["review_windows"],"frame_packet":binding(packet),
        "status":("visible_violation" if any(v["value"]=="no" for v in parsed["checks"].values())
                  else "needs_evidence" if any(v["value"]=="unknown" for v in parsed["checks"].values())
                  else "visible_contract_supported_in_sampled_scope"),
        "posthoc_not_initial_result":True,"quality_release_approved":False,"training_admitted":False,
    }
    write_json(branch/"review_summary.json",summary)
    return {"operation":"review","status":summary["status"]}


def execute(branch, operation):
    """Serialize against the original arm and every repair branch's paid operations."""
    import fcntl
    branch,manifest=_load(branch)
    with (Path(manifest["parent_arm"])/".execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _execute(branch,operation)
        finally:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
