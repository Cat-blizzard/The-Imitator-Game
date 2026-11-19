"""Bounded cabinet experiment with actual first/last image conditioning.

Each paid stage is cached and immutable. No training or automatic retry of
ambiguous submissions. This is an explicit experiment runner, not a scheduler.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.r2h_pipeline.batch_pipeline import (
    paid_stage, binding, DEFAULT_CONFIG, ENV, contact_sheet)
from data_generation_methods.r2h_pipeline.common import (
    file_sha256, json_sha256, load_env_file, probe_video, write_json)
from data_generation_methods.r2h_pipeline.providers import GeminiClient, SeedanceClient
from data_generation_methods.r2h_pipeline.cli import download_video

ROOT = Path("/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c")
CASE = ROOT / "cabinet_case"
OLD = Path("/mnt/sdg/datasets/real_sim_human_pipeline/batch_20261020b")
JOB = "right_cabinet_open_only_891914000"
CHECKS = [
    "first_right_door_closed", "last_right_door_open",
    "left_door_unchanged", "hidden_object_visible_in_last",
    "hidden_object_matches_sim_terminal", "camera_and_scene_consistent",
    "robot_parts_absent",
]
CONDITION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        **{key: {"type": "string", "enum": ["yes", "no", "unknown"]} for key in CHECKS},
        **{phase + "_hands": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "minimum_visible_count": {"type": "integer", "minimum": 0},
                "exact_count_known": {"type": "boolean"},
                "locations": {"type": "array", "items": {"type": "string"}},
            }, "required": ["minimum_visible_count", "exact_count_known", "locations"],
        } for phase in ("first", "last")},
        "observations": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": {"type": "string"}},
    }, "required": CHECKS + ["first_hands", "last_hands", "observations", "issues"],
}


def prepare():
    CASE.mkdir(parents=True, exist_ok=True)
    job = json.loads((OLD / "jobs" / JOB / "input.json").read_text())
    write_json(CASE / "input.json", job)
    source = {
        "schema": "r2h_state_conditioned_case_v1",
        "original_human_window": job["source_window"],
        "sim_video": job["robot"]["video"],
        "sim_initial": binding(job["robot"]["initial_frame"]["path"]),
        "sim_terminal": binding(job["robot"]["final_frame"]["path"]),
        "baseline_human_video": binding(OLD / "jobs" / JOB / "generated_human.mp4"),
        "baseline_human_initial": binding(OLD / "jobs" / JOB / "human_initial.png"),
        "source_stage_runs": job["robot"]["controller_stage_runs"],
        "rendering_plan": {
            "maximum_visible_hands": 1,
            "active_actor": "one human hand/forearm entering from image right",
            "inactive_sim_robot": "removed from rendered human scene",
            "closed_to_open_right_door": True,
            "revealed_content": "the small multicolored cube visible in the sim upper compartment",
        },
        "hidden_state_condition_required": True,
        "original_reference_contains_human_3d_gt": False,
        "robot_stage_timestamps_are_not_synthetic_contact_labels": True,
        "training_admitted": False,
    }
    write_json(CASE / "source_manifest.json", source)
    return job, source


def edit_first(cfg, source):
    prompt = (
        "Edit the FIRST input simulation image. This is the CLOSED initial state of a "
        "small two-door cabinet task, not the finished action. Remove BOTH robot arms, "
        "grippers and their robot-shaped shadows. Use exactly ONE natural human hand and "
        "forearm, from the IMAGE RIGHT edge, resting well away from the cabinet. A plain "
        "dark sleeve is fine. No person/hand/arm on the left, no second hand, no torso. "
        "Keep the original camera viewpoint, cabinet size and placement, both closed doors, "
        "handles, markings, green pear, tabletop and green background unchanged. Do not "
        "open the cabinet, reveal its interior, add new objects or act yet. Preserve the "
        "two-door cabinet mechanism. Produce one coherent edited photograph, no montage "
        "or text. The single active arm is a deliberate rendering plan for this one-arm task."
    )
    result = paid_stage(CASE / "edit_initial_v1", {
        "kind": "actor_budget_anchor_repair", "model": cfg["anchor_model"],
        "target": source["sim_initial"], "prompt": prompt,
        "maximum_visible_hands": 1,
    }, lambda transport: GeminiClient(cfg["gemini"], transport=transport).generate_image(
        Path(source["sim_initial"]["path"]), prompt, model=cfg["anchor_model"]))
    save_image(result, CASE / "human_initial_v1.png")


def edit_last(cfg, source):
    first = CASE / "human_initial_v1.png"
    prompt = (
        "The FIRST input is the edit target: a simulation TERMINAL frame with the "
        "IMAGE-RIGHT cabinet door open. Supporting reference image 1 is the humanized "
        "CLOSED initial frame of exactly this scene, for camera, materials, lighting, "
        "single human actor and sleeve identity. Output the corresponding OPEN terminal "
        "frame. Keep the right door at the same hinge/angle as the simulation target and "
        "the left door unchanged and closed. The small multicolored cube visible inside "
        "the upper compartment in the simulation target MUST remain visibly the same "
        "object at the same shelf location; do not leave that compartment dark or empty. "
        "Do not add any object not visible in the target. Remove all robot arms/grippers "
        "and their shadows. Render at most the SAME ONE human hand and forearm entering "
        "from IMAGE RIGHT as the reference, already released and withdrawn away from "
        "the handle. No left person, extra arms or torso. Preserve the target scene, "
        "cabinet/handles/markings, pear, table and background. Match the supporting "
        "reference appearance without copying its closed-door state. One image only."
    )
    result = paid_stage(CASE / "edit_terminal_v1", {
        "kind": "revealed_state_anchor_edit", "model": cfg["anchor_model"],
        "target": source["sim_terminal"], "supporting_reference": binding(first),
        "prompt": prompt,
    }, lambda transport: GeminiClient(cfg["gemini"], transport=transport).generate_image_with_references(
        Path(source["sim_terminal"]["path"]), [first], prompt, model=cfg["anchor_model"]))
    save_image(result, CASE / "human_terminal_v1.png")


def save_image(result, path):
    from PIL import Image
    if path.exists():
        return
    image = Image.open(io.BytesIO(base64.b64decode(result["images"][0]["data"], validate=True)))
    image.load()
    image.save(path)


def review_conditions(cfg, source):
    images = [
        ("SOURCE SIM CLOSED", source["sim_initial"]["path"]),
        ("SOURCE SIM OPEN", source["sim_terminal"]["path"]),
        ("CANDIDATE HUMAN CLOSED", CASE / "human_initial_v1.png"),
        ("CANDIDATE HUMAN OPEN", CASE / "human_terminal_v1.png"),
    ]
    prompt = (
        "Inspect FOUR labelled STATIC images, not sequential video frames. Compare the "
        "source simulation closed/open states to the two edited human candidates. Report "
        "what is visibly established, not what the intended task says. Keep image-relative "
        "left/right fixed. Check the small multicolored object in the source open cabinet "
        "against the human open candidate; dark/occluded/unclear means unknown. Inventory "
        "all visible hands in EACH candidate, including overlapping partial hands. Count "
        "only visible distinct hands, not inferred people. Describe disagreements. For "
        "camera_and_scene_consistent compare the TWO human candidates as well as source "
        "layout; door angle is supposed to change. These observations are not contact "
        "or physical-validity verification. Return ONLY JSON matching:\n" + json.dumps(CONDITION_SCHEMA)
    )
    def call(transport):
        client = GeminiClient(cfg["gemini"], transport=transport)
        parts = [{"text": prompt}]
        for label, path in images:
            parts += [{"text": label}, client._inline(Path(path), "image")]
        return client._review_result(client._generate(
            client.model, parts, client._review_config(prompt, CONDITION_SCHEMA, True)),
            CONDITION_SCHEMA, True)
    result = paid_stage(CASE / "condition_review_v1", {
        "kind": "first_last_condition_verification", "model": cfg["gemini"]["model"],
        "images": [{"label": label, **binding(path)} for label, path in images],
        "prompt": prompt, "response_schema": CONDITION_SCHEMA,
    }, call)
    obs = result["review"]
    issues = [key for key in CHECKS if obs[key] != "yes"]
    for phase in ("first", "last"):
        inventory = obs[phase + "_hands"]
        if not inventory["exact_count_known"] or inventory["minimum_visible_count"] > 1:
            issues.append(phase + "_actor_budget_unresolved")
        if inventory["minimum_visible_count"] != len(inventory["locations"]):
            issues.append(phase + "_inventory_inconsistent")
    issues += obs["issues"]
    decision = {
        "schema": "r2h_condition_gate_v1",
        "ready_for_conditioned_generation": not issues,
        "review": binding(CASE / "condition_review_v1/result.json"),
        "initial": binding(CASE / "human_initial_v1.png"),
        "terminal": binding(CASE / "human_terminal_v1.png"),
        "issues": issues, "training_admitted": False,
        "motion_verified": False,
    }
    write_json(CASE / "condition_review_v1/decision.json", decision)
    return decision


def build_video_payload(cfg):
    prompt = (
        "One continuous fixed-camera human manipulation shot transitioning between the "
        "provided CLOSED first frame and OPEN last frame. Exactly one human hand/forearm "
        "acts from IMAGE RIGHT; no left-side actor or extra hands. The hand approaches "
        "the IMAGE-RIGHT door handle, visibly grasps it, then gently pulls that door "
        "outward around its right-side hinge. The left door and cabinet body remain still. "
        "The little multicolored cube already inside the upper compartment is revealed "
        "by the opening door; it is NEVER created, moved or removed. Match its identity "
        "and location to the supplied open terminal image. Release the handle and "
        "withdraw the hand, leaving the right door open. The door stays closed until "
        "the hand reaches the handle. No cuts, crossfades, teleportation, detached handles "
        "or popping objects. The green pear stays still. Natural steady pace, complete "
        "the action before the end, with an observable final settled state."
    )
    content = [{"type": "text", "text": prompt}]
    for role, name in (("first_frame", "human_initial_v1.png"), ("last_frame", "human_terminal_v1.png")):
        content.append({"type": "image_url", "role": role, "image_url": {
            "url": "data:image/png;base64," + base64.b64encode((CASE / name).read_bytes()).decode()}})
    return dict(model=cfg["seedance"]["model"], content=content, duration=8,
                ratio="adaptive", resolution="720p", generate_audio=False)


def video(cfg, *, poll_minutes=25):
    gate_path = CASE / "condition_review_v1/decision.json"
    gate = json.loads(gate_path.read_text())
    if not gate["ready_for_conditioned_generation"]:
        raise ValueError("Conditions unresolved; no video POST issued")
    for key in ("initial", "terminal", "review"):
        value = gate[key]
        if file_sha256(value["path"]) != value["sha256"]:
            raise ValueError("Condition changed after review")
    payload = build_video_payload(cfg)
    client = SeedanceClient(dict(cfg["seedance"], timeout=30))
    client.prepare_submit(payload)  # actual role compatibility, no network
    stage = CASE / "submit_first_last_v1"
    submitted = paid_stage(stage, {
        "kind": "verified_first_last_video_generation", "model": cfg["seedance"]["model"],
        "condition_gate": binding(gate_path), "first_frame": gate["initial"],
        "last_frame": gate["terminal"], "prompt": payload["content"][0]["text"],
        "payload_sha256": json_sha256(payload),
        "structured_input_roles": ["first_frame", "last_frame"],
        "settings": {k: v for k, v in payload.items() if k != "content"},
    }, lambda transport: SeedanceClient(dict(cfg["seedance"], timeout=600),
                                         transport=transport).submit(payload))
    target = CASE / "state_conditioned_human_v1.mp4"
    deadline = time.monotonic() + poll_minutes * 60
    while not target.exists() and time.monotonic() < deadline:
        result = client.poll(submitted["task_id"])
        write_json(CASE / "polls" / f"{time.time_ns()}.json", result)
        print(json.dumps({"stage": "video_poll", "status": result["status"]}), flush=True)
        if result["status"] == "succeeded":
            download_video(result["output_url"], target, task_id=submitted["task_id"],
                           request_sha256=file_sha256(stage / "request_metadata.json"))
            break
        if result["status"] in {"failed", "cancelled", "expired"}:
            raise RuntimeError("Provider video task terminated: " + result["status"])
        time.sleep(45)
    if target.exists():
        write_json(CASE / "generated_media.json", probe_video(target))
        contact_sheet(target, CASE / "new_contact_sheet.jpg")
        print(json.dumps({"stage": "downloaded", "path": str(target)}), flush=True)
    else:
        print(json.dumps({"stage": "poll_timeout", "resumable_GET_only": True}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "edit-first", "edit-last", "review", "video"])
    args = p.parse_args()
    job, source = prepare()
    if args.stage == "prepare":
        print(json.dumps({"case": str(CASE)}))
        return
    load_env_file(ENV)
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    if args.stage == "edit-first":
        edit_first(cfg, source)
    elif args.stage == "edit-last":
        edit_last(cfg, source)
    elif args.stage == "review":
        print(json.dumps(review_conditions(cfg, source)), flush=True)
    else:
        video(cfg)


if __name__ == "__main__":
    main()
