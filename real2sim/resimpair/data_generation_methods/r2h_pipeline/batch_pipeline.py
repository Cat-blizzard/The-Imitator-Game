"""Human-grounded, bounded synthetic-reference pilot; never starts training.

Keeps generation, observations, evidence checks and training release separate.
All paid stages have durable exclusive markers and immutable request bindings.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import file_sha256, json_sha256, load_env_file, probe_video, write_json
from .providers import GeminiClient, SeedanceClient, ProviderError
from .cli import download_video
from .task_evidence import build_apple_contract, validate_contract, evaluate_task_evidence

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("/mnt/sdg/datasets/real_sim_human_pipeline/batch_20261020b")
DEFAULT_CONFIG = WORKSPACE / "data_generation_methods/config.echoflow.example.json"
ENV = Path("/home/xzzhou/.config/policygen/credentials/echoflow.env")


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def checked_binding(value):
    if file_sha256(value["path"]) != value["sha256"]:
        raise ValueError("Source artifact hash mismatch")
    return value


def contract_for(job):
    """Only three reviewed task families; unknown tasks require an explicit compiler."""
    if job["task_family"] == "bowl":
        contract = build_apple_contract(job["source_spec"]["path"], job["source_window"]["id"])
    else:
        source = deepcopy(job["source_window"])
        source.update(origin="real_human", review_evidence=[
            deepcopy(source[k]) for k in ("review", "actual_review_sheet") if k in source])
        def relation(name, predicate, *args, expected=True):
            return dict(id=name, predicate=predicate, args=list(args), expected=expected)
        def event(name, verb, *args):
            return dict(id=name, verb=verb, args=list(args))
        common = dict(
            schema="r2h_task_evidence_contract_v1", task_id=job.get("canonical_goal", job["task_id"]),
            source_spec=deepcopy(job["source_spec"]), source_window=source,
            allowed_additional_events=[], allowed_variations=deepcopy(job["allowed_variations"]),
            training_admitted=False, human_wrist_3d_ground_truth_created=False,
            contract_status="prepared_from_reviewed_source_not_quality_approval",
        )
        if job["task_family"] == "book":
            events = [event("grasp_book", "grasp", "actor", "book"),
                      event("lift_book", "lift_clear_of", "book", "table"),
                      event("place_book", "place_inside", "book", "box"),
                      event("release_book", "release_in", "book", "box"),
                      event("withdraw", "withdraw_from", "actor", "box")]
            contract = dict(common, entities=["actor", "book", "box", "table"],
                entity_roles={"book": "the unique closed book initially on the table",
                              "box": "the open box; it need not be empty",
                              "actor": "acting hand or robot gripper", "table": "table surface"},
                initial_relations=[relation("book_closed", "closed", "book"),
                                   relation("book_on_table", "supported_by", "book", "table"),
                                   relation("box_open", "open", "box")],
                final_relations=[relation("book_in_box", "inside", "book", "box"),
                                 relation("book_released", "released_from", "book", "actor"),
                                 relation("hand_withdrawn", "withdrawn_from", "actor", "box")],
                required_events=events,
                forbidden_events=[event("open_book", "open", "book"),
                                  event("move_box", "transport", "box")])
        elif job["task_family"] in {"cabinet", "cab"}:
            events = [event("handle_contact", "grasp", "actor", "right_handle"),
                      event("open_right", "open", "right_door"),
                      event("release_handle", "release", "actor", "right_handle"),
                      event("withdraw", "withdraw_from", "actor", "right_handle")]
            contract = dict(common,
                entities=["actor", "right_door", "left_door", "right_handle", "cabinet_contents"],
                entity_roles={"right_door": "the cabinet door on IMAGE RIGHT in the initial view",
                              "left_door": "the cabinet door on IMAGE LEFT in the initial view",
                              "right_handle": "the handle attached to that image-right door",
                              "actor": "acting human hand or robot gripper",
                              "cabinet_contents": "objects originally inside the cabinet"},
                initial_relations=[relation("right_closed", "closed", "right_door")],
                final_relations=[relation("right_open", "open", "right_door"),
                                 relation("left_unchanged", "unchanged_from_initial", "left_door"),
                                 relation("handle_released", "released_from", "right_handle", "actor"),
                                 relation("hand_withdrawn", "withdrawn_from", "actor", "right_handle")],
                required_events=events,
                forbidden_events=[event("open_left", "open", "left_door"),
                                  event("remove_contents", "remove_from_cabinet", "cabinet_contents")])
        else:
            raise ValueError("Unsupported human-grounded task family")
        contract["precedence"] = [[a["id"], b["id"]] for a, b in zip(events, events[1:])]
    contract["source_initial_description"] = job["initial"]
    contract["source_final_description"] = job["final"]
    contract["source_necessary_steps"] = deepcopy(job["necessary_steps"])
    validate_contract(contract, verify_files=True)
    return contract


def paid_stage(directory, metadata, operation):
    """Resume persisted results, never reissue an ambiguous POST."""
    from data_generation_methods.scripts.echoflow_pilot import diagnostic_transport
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = json_sha256(metadata)
    result_path = directory / "result.json"
    old_meta = directory / "request_metadata.json"
    if old_meta.exists():
        old = json.loads(old_meta.read_text())
        if old["request_fingerprint"] != fingerprint:
            raise ValueError("Cannot reuse a stage with different inputs")
    if result_path.exists():
        if not old_meta.exists():
            raise ValueError("Result lacks input binding")
        return json.loads(result_path.read_text())
    if (directory / "call_started.json").exists():
        raise RuntimeError("Unresolved or failed paid stage: inspect it; automatic retry prohibited")
    write_json(old_meta, {
        **metadata, "request_fingerprint": fingerprint,
        "implementation_sha256": file_sha256(Path(__file__)), "created_unix": time.time(),
    })
    with (directory / "call_started.json").open("x") as stream:
        json.dump({"started_unix": time.time(), "automatic_retry": False}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    started = time.monotonic()
    try:
        result = operation(diagnostic_transport(directory))
        result.update(elapsed_seconds=round(time.monotonic()-started, 3), training_admitted=False)
        write_json(result_path, result)
        return result
    except ProviderError as error:
        write_json(directory / "failure.json", {
            "code": error.code, "submission_unknown": error.submission_unknown,
            "http_status": error.http_status, "elapsed_seconds": time.monotonic()-started,
            "automatic_retry": False,
        })
        raise


ANCHOR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        **{k: {"type": "string", "enum": ["yes", "no", "unknown"]} for k in (
            "initial_state_preserved", "task_objects_preserved",
            "required_geometry_visible", "robot_parts_removed")},
        "role_grounding": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"entity": {"type": "string"}, "visual_identity": {"type": "string"},
                           "image_location": {"type": "string"}},
            "required": ["entity", "visual_identity", "image_location"]}},
        "issues": {"type": "array", "items": {"type": "string"}},
        "visible_description": {"type": "string"},
    },
    "required": ["initial_state_preserved", "task_objects_preserved", "required_geometry_visible",
                 "robot_parts_removed", "role_grounding", "issues", "visible_description"],
}


def image_review(client, source, candidate, prompt):
    config = client._review_config(prompt, ANCHOR_SCHEMA, True)
    raw = client._generate(client.model, [
        {"text": prompt}, {"text": "SOURCE SIMULATION IMAGE:"},
        client._inline(Path(source), "image"), {"text": "EDITED CANDIDATE IMAGE:"},
        client._inline(Path(candidate), "image")], config)
    return client._review_result(raw, ANCHOR_SCHEMA, True)


def anchor_prompt(job):
    return (
        "Edit this simulation image into a realistic human manipulation scene. Replace robot "
        "arms and grippers ONLY with plausible human forearms/hands entering from the same "
        "image edges. Do not add a full person blocking the objects. Hands rest away from "
        "task objects and do not act yet. Preserve camera, all task objects and distractors, "
        "their positions, states, count, openings, handles, table and background. No text, "
        "arrows, objects already transferred, or completed task. Preserve container interiors "
        "and door hinge/handle geometry. Initial state to preserve: " + str(job["initial"]) +
        "\nVisible source role descriptions: " + json.dumps(job["role_grounding"], ensure_ascii=False))


def video_prompt(job, grounded_roles):
    return (
        "One continuous fixed-camera shot starting from the supplied image. Natural human "
        "manipulation with visible reach, contact, movement, release and a settled final "
        "state. Resolve entity roles ONCE from the first frame; do not redefine source as "
        "'whichever container currently has the object'. Image left/right is viewer-relative. "
        "Until the hand visibly touches the task object, that object stays in its original "
        "place. No cuts, crossfades, teleportation, duplication, disappearance or extra actors. "
        "Keep all non-task objects stationary. No robot parts or text. Finish the entire "
        "atomic task and withdraw the hand. Do not invent a second task.\n"
        "Initial state: " + str(job["initial"]) +
        "\nNecessary actions in semantic order: " + json.dumps(job["necessary_steps"], ensure_ascii=False) +
        "\nRequired final state: " + str(job["final"]) +
        "\nFixed roles grounded in THIS human initial image: " + json.dumps(grounded_roles, ensure_ascii=False))


def contact_sheet(video, path, count=24):
    import cv2
    from PIL import Image, ImageDraw
    cap = cv2.VideoCapture(str(video))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    sheet = Image.new("RGB", (1280, ((count+3)//4)*345), "white")
    draw = ImageDraw.Draw(sheet)
    for j, idx in enumerate([round(i*(n-1)/(count-1)) for i in range(count)]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            raise ValueError("Cannot decode contact-sheet frame")
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        im.thumbnail((320, 320))
        x, y = (j % 4)*320, (j//4)*345
        sheet.paste(im, (x, y+25))
        draw.text((x+4, y+4), f"frame {idx}  t={idx/fps:.3f}s", fill="black")
    cap.release()
    sheet.save(path)


def audit_video(job, directory, contract, cfg, video, *, review_name="task_review",
                refine_from=None, frame_budget=48):
    from .task_review import build_review_schema, build_review_prompt, normalize_review
    from data_generation_methods.scripts.echoflow_repair_study import frame_packet
    media = probe_video(video)
    if not review_name.replace("_", "").isalnum():
        raise ValueError("Review name must be a simple versioned stage name")
    audit_dir = directory / review_name
    refinement = None
    if refine_from:
        from .refinement import plan_refinement
        if not refine_from.replace("_", "").isalnum() or refine_from == review_name:
            raise ValueError("Refinement requires a separate prior stage")
        previous = directory / refine_from
        previous_packet = json.loads((previous / "frames/packet.json").read_text())
        if previous_packet["source_sha256"] != media["sha256"]:
            raise ValueError("Refinement belongs to a different video")
        prior = json.loads((previous / "normalized_evidence.json").read_text())
        refinement = plan_refinement(
            media["frames"], [f["frame_index"] for f in previous_packet["frames"]],
            prior["refinement_requests"], max_frames=frame_budget)
        refinement.update(
            source_evidence=binding(previous / "normalized_evidence.json"),
            source_packet=binding(previous / "frames/packet.json"),
            video=binding(video))
        write_json(audit_dir / "refinement_plan.json", refinement)
        if refinement["added_frame_count"] == 0:
            raise ValueError("Refinement budget adds no evidence; no paid call issued")
    packet_path = audit_dir / "frames/packet.json"
    if packet_path.exists():
        packet = json.loads(packet_path.read_text())
        if packet["source_sha256"] != media["sha256"]:
            raise ValueError("Review packet source changed")
        frames = packet["frames"]
    else:
        # Mandatory actual boundaries plus broad coverage. This is not dense
        # verification of every interaction; missing evidence remains explicit.
        def indices(n):
            if refinement:
                return refinement["selected_frame_indices"]
            return sorted(set([0, 1, 2, 3, n-4, n-3, n-2, n-1] +
                              [round(i*(n-1)/17) for i in range(18)]))
        frames = frame_packet(video, audit_dir / "frames", indices)
    if refinement and [f["frame_index"] for f in frames] != refinement["selected_frame_indices"]:
        raise ValueError("Existing frame packet disagrees with refinement plan")
    if not (directory / "contact_sheet.jpg").exists():
        contact_sheet(video, directory / "contact_sheet.jpg")
    schema, prompt = build_review_schema(contract), build_review_prompt(contract)
    supplied = [{k: f[k] for k in ("path", "frame_index", "timestamp_seconds")} for f in frames]
    result = paid_stage(audit_dir, {
        "kind": "directed_visible_task_evidence", "model": cfg["gemini"]["model"],
        "video": binding(video), "contract_sha256": json_sha256(contract),
        "frames": frames, "prompt": prompt, "response_schema": schema,
        "scope": "visible_evidence_only_not_full_video_physics",
    }, lambda transport: GeminiClient(cfg["gemini"], transport=transport).review_frames(
        supplied, prompt, schema, allow_json_fence=True))
    evidence = normalize_review(contract, result, media={
        "path": str(video), "sha256": media["sha256"], "duration_seconds": media["duration_seconds"]},
        review_artifact=binding(audit_dir / "result.json"), supplied_frames=supplied)
    write_json(audit_dir / "normalized_evidence.json", evidence)
    policy = dict(time_tolerance_seconds=0.001, initial_window_seconds=0.25,
                  final_window_seconds=0.6, motion_requirements=[])
    decision = evaluate_task_evidence(contract, evidence, policy=policy, verify_files=True)
    write_json(audit_dir / "gate_decision.json", decision)
    if not (directory / "contact_sheet.jpg").exists():
        contact_sheet(video, directory / "contact_sheet.jpg")
    return {"media": media, "evidence_gate": decision["decision"],
            "issues": decision["issues"], "training_admitted": False}


def run_job(job, root, cfg, *, reuse=False, poll_minutes=20):
    from PIL import Image
    directory = root / "jobs" / job["job_id"]
    directory.mkdir(parents=True, exist_ok=True)
    state = {"job_id": job["job_id"], "training_started": False, "training_admitted": False}
    def update(**kwargs):
        state.update(kwargs, updated_unix=time.time())
        write_json(directory / "STATUS.json", state, overwrite=True)
        print(json.dumps({k: v for k, v in state.items() if k != "issues"}), flush=True)
    try:
        checked_binding(job["source_spec"])
        checked_binding(job["anchor"])
        checked_binding(job["robot"]["video"])
        if file_sha256(job["robot"]["trajectory_path"]) != job["robot"]["trajectory_sha256"]:
            raise ValueError("Robot action artifact changed")
        contract = contract_for(job)
        write_json(directory / "input.json", job)
        write_json(directory / "contract.json", contract)
        anchor = directory / "human_initial.png"
        video = directory / "generated_human.mp4"
        if reuse:
            previous = Path("/mnt/sdg/datasets/real_sim_human_pipeline/repair_study_20261020_v1")
            source_anchor = Path("/mnt/sdg/datasets/real_sim_human_pipeline/pilot_20261020/edit-anchor/human_initial.png")
            if "891911000" not in job["robot"]["episode_id"]:
                raise ValueError("Reuse is restricted to the exact original robot trajectory")
            old_input = json.loads((WORKSPACE / "data_generation_methods/pilot_inputs" /
                                    job["robot"]["episode_id"] / "input.json").read_text())
            if old_input["source_files"]["arrays"]["sha256"] != job["robot"]["trajectory_sha256"]:
                raise ValueError("Reuse source trajectory mismatch")
            for dst, src in [(anchor, source_anchor), (video, previous / "role_grounded.mp4")]:
                if not dst.exists():
                    dst.symlink_to(src)
                if file_sha256(dst) != file_sha256(src):
                    raise ValueError("Reused artifact changed")
            write_json(directory / "reuse.json", {
                "generation_request": binding(previous / "submit-role-grounded/request_metadata.json"),
                "generation_result": binding(previous / "submit-role-grounded/result.json"),
                "media": binding(video), "anchor": binding(anchor),
                "new_video_generation_calls": 0, "training_admitted": False})
            update(status="reused_exact_source_video", video=str(video))
        else:
            prompt = anchor_prompt(job)
            update(status="editing_anchor")
            result = paid_stage(directory / "edit_anchor", {
                "kind": "anchor_edit", "model": cfg["anchor_model"],
                "input": job["anchor"], "prompt": prompt,
            }, lambda transport: GeminiClient(cfg["gemini"], transport=transport).generate_image(
                Path(job["anchor"]["path"]), prompt, model=cfg["anchor_model"]))
            if not anchor.exists():
                image = Image.open(io.BytesIO(base64.b64decode(result["images"][0]["data"], validate=True)))
                image.load()
                image.save(anchor)
            prompt = (
                "Compare two separate static images: SOURCE SIMULATION and EDITED CANDIDATE. "
                "These are not consecutive video frames. Only robot arms should have changed "
                "to human forearms/hands. Inspect object identity/count, initial relationships, "
                "openings/hinges/handles and visibility. Report no or unknown when preservation "
                "is not visibly supported. Do not assume a successful edit. Ground each task "
                "entity in the EDITED image with its visible identity and IMAGE-relative location; "
                "do not prescribe a future action. Robot parts must be absent.\nEntities: " +
                json.dumps(contract["entity_roles"], ensure_ascii=False) +
                "\nOutput exactly this JSON schema:\n" + json.dumps(ANCHOR_SCHEMA))
            update(status="checking_anchor")
            checked = paid_stage(directory / "anchor_review", {
                "kind": "source_candidate_static_comparison", "model": cfg["gemini"]["model"],
                "source": job["anchor"], "candidate": binding(anchor),
                "prompt": prompt, "response_schema": ANCHOR_SCHEMA,
            }, lambda transport: image_review(GeminiClient(cfg["gemini"], transport=transport),
                                              job["anchor"]["path"], anchor, prompt))
            observations = checked["review"]
            required = ["initial_state_preserved", "task_objects_preserved",
                        "required_geometry_visible", "robot_parts_removed"]
            if any(observations[k] != "yes" for k in required) or observations["issues"]:
                update(status="anchor_needs_review", issues=observations["issues"],
                       anchor_checks={k: observations[k] for k in required})
                return state
            prompt = video_prompt(job, observations["role_grounding"])
            payload = dict(model=cfg["seedance"]["model"], content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "role": "first_frame", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(anchor.read_bytes()).decode()}}],
                duration=8, ratio="adaptive", resolution="720p", generate_audio=False)
            update(status="submitting_video")
            submitted = paid_stage(directory / "submit_video", {
                "kind": "first_frame_task_semantic_generation", "model": cfg["seedance"]["model"],
                "anchor": binding(anchor), "prompt": prompt, "payload_sha256": json_sha256(payload),
                "generation_settings": {k: v for k, v in payload.items() if k != "content"},
            }, lambda transport: SeedanceClient(dict(cfg["seedance"], timeout=600),
                                                 transport=transport).submit(payload))
            task_id = submitted["task_id"]
            update(status="video_running", task_id=task_id)
            client = SeedanceClient(dict(cfg["seedance"], timeout=30))
            deadline = time.monotonic() + poll_minutes*60
            while not video.exists() and time.monotonic() < deadline:
                result = client.poll(task_id)
                write_json(directory / "polls" / f"{time.time_ns()}.json", result)
                if result["status"] == "succeeded":
                    download_video(result["output_url"], video, task_id=task_id,
                                   request_sha256=file_sha256(directory / "submit_video/request_metadata.json"))
                    break
                if result["status"] in {"failed", "expired", "cancelled"}:
                    update(status="video_generation_failed", provider_status=result["status"])
                    return state
                time.sleep(45)
            if not video.exists():
                update(status="poll_window_ended", resumable_GET_only=True)
                return state
        update(status="auditing_video", video=str(video))
        audited = audit_video(job, directory, contract, cfg, video)
        update(status="completed_candidate_audit", **audited)
    except (ProviderError, OSError, RuntimeError, ValueError, KeyError) as error:
        update(status="needs_inspection", error_type=type(error).__name__,
               error_code=error.code if isinstance(error, ProviderError) else str(error)[:400])
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--job", action="append")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--review-name", default="task_review")
    p.add_argument("--refine-from")
    p.add_argument("--frame-budget", type=int, default=48)
    args = p.parse_args()
    manifest = json.loads((args.root / "input_manifest.json").read_text())
    jobs = manifest["jobs"]
    if args.job:
        jobs = [job for job in jobs if job["job_id"] in args.job]
    if not jobs or len(jobs) > 4 or not 1 <= args.workers <= 2:
        raise ValueError("Bounded pilot supports one to four jobs and one or two workers")
    if args.prepare_only:
        for job in jobs:
            directory = args.root / "jobs" / job["job_id"]
            write_json(directory / "input.json", job)
            write_json(directory / "contract.json", contract_for(job))
        print(json.dumps({"prepared_jobs": len(jobs), "paid_calls": 0}))
        return
    load_env_file(ENV)
    cfg = json.loads(args.config.read_text())
    if args.audit_only:
        for job in jobs:
            directory = args.root / "jobs" / job["job_id"]
            state = json.loads((directory / "STATUS.json").read_text())
            try:
                result = audit_video(job, directory, contract_for(job), cfg,
                                     directory / "generated_human.mp4",
                                     review_name=args.review_name, refine_from=args.refine_from,
                                     frame_budget=args.frame_budget)
                state.pop("error_type", None)
                state.pop("error_code", None)
                state.update(result, status="completed_candidate_audit")
            except (ProviderError, OSError, RuntimeError, ValueError, KeyError) as error:
                state.update(status="needs_inspection", error_type=type(error).__name__,
                             error_code=error.code if isinstance(error, ProviderError) else str(error)[:400])
            state.update(updated_unix=time.time(), review_stage=args.review_name,
                         training_started=False)
            write_json(directory / "STATUS.json", state, overwrite=True)
            print(json.dumps(state), flush=True)
        return
    # The shared in-account budget is explicit: at most 3 new images, 3 new
    # videos, 3 anchor reviews and 4 video reviews in this four-job pilot.
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job, args.root, cfg,
                               reuse="891911000" in job["robot"]["episode_id"]) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            snapshots = [json.loads(path.read_text()) for path in
                         sorted((args.root / "jobs").glob("*/STATUS.json"))]
            write_json(args.root / "BATCH_STATUS.json", {
                "updated_unix": time.time(), "job_status_snapshots": snapshots,
                "planned_jobs": len(manifest["jobs"]),
                "requested_jobs_this_process": [job["job_id"] for job in jobs],
                "training_started": False,
                "quality_release_approved": False}, overwrite=True)


if __name__ == "__main__":
    main()
