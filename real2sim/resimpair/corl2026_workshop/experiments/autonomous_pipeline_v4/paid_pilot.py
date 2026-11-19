"""A finite, separately labelled Sim-to-human experiment on qualified native data.

Reuse the existing model transports, immutable ledger, image gate and video
review. No new native execution, source invention, per-sample repair or retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .core import activate, bind, checked, digest, implementation, read, replace_state, require, write

SCENES = ["spring_console", "hook_retrieval", "bead_stirring"]


def shared_anchor_request(path, request):
    """Two immutable episodes may contain identical initial pixels at different paths."""
    path = Path(path)
    if path.exists():
        prior = read(path)
        for key in ["prompt", "model_config", "initial_scene", "implementation"]:
            require(prior[key] == request[key], "shared_anchor_condition_changed:" + key)
        require(len(prior["images"]) == len(request["images"]) == 1
                and prior["images"][0]["sha256"] == request["images"][0]["sha256"], "shared_anchor_pixels_changed")
        checked(prior["images"][0])
        checked(request["images"][0])
    else:
        write(path, request)


def prepare(root, ancestor):
    from .budget import Budget, counts
    root, ancestor = Path(root).resolve(), Path(ancestor).resolve()
    require(not root.exists(), "new_paid_campaign_root_required")
    require(shutil.disk_usage(root.parent).free > 20 * 2**30, "insufficient_disk_for_paid_pilot")
    library_path = ancestor / "qualified_task_library_v3.json"
    library = read(library_path)
    for bound in library["audits"]:
        checked(bound)
    selections = []
    for name in SCENES:
        group = next(g for g in library["groups"] if g["scene"] == name)
        require(group["same_initial_physics"] and group["same_initial_objects"] and group["same_initial_images"],
                "shared_anchor_requires_same_native_initial_state")
        for branch in [0, 1]:
            selected = next(t for t in group["tasks"] if t["task_branch"] == branch)
            for key in ["physical_contract", "execution", "result", "independent_audit"]:
                checked(selected[key])
            selections.append(selected)
    require(len(selections) == 6, "fixed_six_task_denominator")
    old = read(ancestor / "budget.json")
    prior = {k: old["ancestor_started_calls"][k] + counts(old["ledger_path"])[k] for k in old["new_call_limits"]}
    root.mkdir()
    write(root / "ancestor_usage.json", {"budget": bind(ancestor / "budget.json"), "lineage_started": prior,
                                         "old_zero_budget_and_native_evidence_unchanged": True})
    limits = {"image": 3, "video": 6, "review": 9}
    write(root / "budget.json", {"schema": "resimpair_cumulative_budget_v4", "campaign_id": "novel-human-20261024-v1",
        "ledger_path": str(root / "campaign_ledger"), "new_call_limits": limits, "ancestor_started_calls": prior,
        "ancestor_evidence": [bind(root / "ancestor_usage.json")],
        "lineage_cumulative_limits": {k: prior[k] + limits[k] for k in prior},
        "max_concurrent": 1, "timeout_seconds": 600, "currency_budget": None, "automatic_retry": False,
        "authorization": "2026-10-24 standing user authorization: API calls for task data need no repeated approval; record paid quota consumption.",
        "allocation_by_agent": "3 shared first images, 6 task videos, 3 image reviews and 6 video reviews; failures count; no semantic retry",
        "monetary_accounting": "provider credit before/after each stage plus public denomination; exact RMB/request invoice remains unknown if unexposed"})
    write(root / "config.json", read(ancestor / "config.json"))
    write(root / "experiment.json", {"schema": "qualified_native_to_human_pilot_v1", "native_library": bind(library_path),
        "selections": selections, "sources": [], "source_coverage": "missing_for_these_new_mechanisms",
        "review_frame_budget": 49, "native_simulation_calls": 0, "new_api_call_limits": limits,
        "max_concurrent": 1, "per_video_poll_deadline_seconds": 900,
        "shared_first_image_per_scene": True, "semantic_retry": False, "manual_downstream_annotation_count": 0,
        "candidate_only": True, "formal_real_sim_human_pair": False, "training_admitted": False,
        "selection_policy": "Before all API outcomes: first two distinct branches of spring, hook and stir; no replacement of failures",
        "changes": ["Expose frozen native color/compound geometry and required spatial predicates", "Capture provider paid-credit intervals"],
        "hypotheses": ["One shared image can support two distinct physically accepted task branches",
                       "A single connected actor and preserved anatomical roles can be observed",
                       "Explicit intermediate spatial geometry preserves spring rebound, selected cargo and opposite stir directions"],
        "validation_scope": "New mechanisms for visual generation; native cases already used for physics development; no unseen real-source claim"})
    Budget(root / "budget.json")
    return {"root": str(root), "tasks": 6, "scenes": 3, "new_call_limits": limits, "lineage_before": prior}


def compact_video_prompt(contract):
    """Preserve every semantic check while removing measurement-only bulk."""
    from .prompts import compact
    info = contract["generation_information"]
    inventory = []
    for entity in info["inventory"]:
        item = {k: v for k, v in entity.items() if k in {"id", "description", "kind", "role", "initial_image_xy_normalized", "final_image_xy_normalized"}}
        appearance = entity.get("native_appearance", {})
        item["color"] = appearance.get("color_name")
        inventory.append(item)
    events = []
    for event in info["required_events"]:
        item = {k: v for k, v in event.items() if k in {"id", "action", "entities", "human_hand", "suggested_seconds"}}
        if event.get("required_spatial_targets"):
            item["required_contact_points"] = [{"entities": t["entities"], "image_xy": t["image_xy_normalized"]}
                                                for t in event["required_spatial_targets"]]
        if event.get("joint_state_evidence"):
            item["joint_motion"] = event["joint_state_evidence"]
        events.append(item)
    payload = {"task": info["task_description_from_frozen_plan"], "duration_seconds": info["duration_seconds"],
               "inventory": inventory, "events": events,
               **{k: info[k] for k in ["effectors", "initial_hands", "terminal_hands", "initial_relations", "final_relations",
                                       "required_event_precedence", "persistent_constraints", "non_target_entities"]}}
    if info.get("required_cyclic_motion"):
        payload["required_cyclic_motion"] = info["required_cyclic_motion"]
    return (
        "One continuous fixed elevated camera human demonstration, starting from the supplied image. "
        "Exactly ONE operator and TWO arms belonging to the same visible torso throughout. Both hands "
        "must originate at that person's shoulders through elbows and wrists; never add a new reaching arm "
        "while leaving both original hands resting. Preserve the initial body, camera, table and ALL objects. "
        "Anatomical RIGHT is initially image-left, LEFT image-right; track the same limb even across the body. "
        "The event's human_hand is mandatory. Do not choose a hand merely because it is nearest. "
        "Follow required order and intermediate motion, not only the final state. Image_xy is normalized "
        "from image left/top, distinct from anatomy. Reach naturally; copying robot joint paths is unnecessary. "
        "For spring controls, visibly depress the chosen cap DOWN into its fixed housing and release it "
        "to spring back; merely touching an unmoving block is not success. For tool tasks, grasp the handle "
        "and keep holding the SAME rigid tool with the prescribed hand. Move cargo using the hook head "
        "without direct hand-cargo contact. Do not move unselected cargo. For cyclic paths, keep the tool "
        "tip inside the container and visibly traverse the listed points in their exact order, direction "
        "and full turn count. Let contacted solid beads move plausibly and remain inside; there is NO liquid. "
        "Preserve native functional shapes, small scale, compound wall pieces, gaps and component counts. "
        "Do not replace geometry with a more familiar bowl, spoon or switch. No cuts, teleports, duplicated "
        "objects, unsupported floating, mirroring, hand swaps or camera moves. Finish with a settled one-second "
        "view of declared object AND hand states; if a hand must hold a tool, do not put it down. "
        "Text and coordinates are instructions, never draw overlays.\n" + compact(payload)
    )


def compile_all(root):
    from .mechanism_conditions import compile_candidate
    from .media import packet
    from .pipeline import release_id
    from .prompts import anchor_prompt
    from . import review
    root = Path(root)
    release = release_id()
    records = []
    for selection in read(root / "experiment.json")["selections"]:
        case = selection["case"]["id"]
        folder = root / "compiled" / release / case
        contract = compile_candidate(selection)
        write(folder / "contract.json", contract)
        require(contract["human_evidence_plan"]["declaration_coverage_complete"],
                "unhandled_evidence_requirement_before_generation")
        write(folder / "evidence_plan.json", contract["human_evidence_plan"])
        bundle = read(checked(selection["execution"]))
        with np.load(checked(bundle["measurements"]), allow_pickle=False) as archive:
            times = archive["timestamps_seconds"].copy()
        media = {}
        for name, camera in bundle["cameras"].items():
            dest = folder / (name + "_frames")
            decoded = packet(checked(camera["video"]), dest, frame_budget=9)
            require(decoded["decoded_frames"] == camera["frame_count"] == len(times)
                    and np.allclose(decoded["all_native_pts_seconds"], times, atol=1e-5, rtol=0), "native_media_clock_mismatch")
            media[name] = bind(dest / "packet.json")
        write(folder / "media_validation.json", {"contract": bind(folder / "contract.json"), "cameras": media, "both_fully_decoded": True})
        prompt = anchor_prompt(contract["initial_scene_information"])
        anchor_id = digest({"image": contract["initial_image"]["sha256"], "prompt": prompt,
                            "config": bind(root / "config.json"), "review": implementation()["review.py"]})[:24]
        anchor = root / "anchors" / anchor_id
        shared_anchor_request(anchor / "image_request.json", {"schema": "resimpair_initial_request_v4", "prompt": prompt,
            "model_config": bind(root / "config.json"), "images": [contract["initial_image"]],
            "initial_scene": contract["initial_scene_information"], "implementation": implementation(),
            "task_specific_prompt_override": None})
        video_text = compact_video_prompt(contract)
        write(folder / "prepared_video.json", {"prompt": video_text, "characters": len(video_text), "submitted": False})
        records.append({"case": case, "scene": selection["scene_id"], "branch": selection["task_branch"],
                        "contract": bind(folder / "contract.json"), "anchor_id": anchor_id,
                        "evidence_plan": bind(folder / "evidence_plan.json"),
                        "video_prompt": bind(folder / "prepared_video.json"), "video_prompt_characters": len(video_text),
                        "anchor_checks": review.anchor_checks(contract["initial_scene_information"])})
    for scene in SCENES:
        require(len({r["anchor_id"] for r in records if r["scene"] == scene}) == 1, "scene_tasks_do_not_share_anchor")
    write(root / "compiled" / release / "index.json", {"release": release, "records": records})
    return {"prepared_tasks": len(records), "shared_anchors": len({r["anchor_id"] for r in records}),
            "prompt_characters": [r["video_prompt_characters"] for r in records]}


def run_case(root, record, budget):
    from .compiler import completed_contract
    from .pipeline import release_id
    from .models import generate_image, submit_video, poll_video
    from . import review, mechanism_review
    root = Path(root)
    directory = root / "runs" / record["case"] / release_id()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "outcome.json"
    if output.exists():
        return read(output)
    previous = sorted((root / "runs" / record["case"]).glob("*/outcome.json"))
    if previous:
        require(len(previous) == 1, "multiple_terminal_runs_require_audit")
        result = read(previous[0])
        checked(result["contract"])
        return result
    contract = completed_contract(checked(record["contract"]))
    require(contract["generation_scope"]["allowed_for_synthetic_quality_probe"] is True
            and not contract["generation_scope"]["formal_real_sim_human_pair"], "candidate_scope_not_explicit")
    anchor = root / "anchors" / record["anchor_id"]
    with (directory / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.time()
        receipt = generate_image(root, budget, anchor / "image_request.json", anchor / "human.png")
        initial = review.anchor(root, budget, anchor, receipt, contract["initial_scene_information"], contract["initial_image"])
        result = {"case": record["case"], "scene": record["scene"], "branch": record["branch"],
                  "contract": record["contract"], "shared_anchor": bind(anchor / "human.png"),
                  "anchor_review": bind(anchor / "review.json"), "source_coverage": "unverified",
                  "formal_real_sim_human_pair": False, "training_admitted": False,
                  "automatic_retry": False, "manual_downstream_annotation_count": 0}
        if not initial["gate"]["passed_in_sampled_visual_scope"]:
            result.update(status="initial_image_rejected", gate=initial["gate"], video=None)
            write(output, result)
            return result
        prepared = read(checked(record["video_prompt"]))
        request = directory / "video_request.json"
        write(request, {"schema": "resimpair_video_request_v4", "contract": record["contract"],
            "model_config": bind(root / "config.json"), "prompt": prepared["prompt"], "first_frame": receipt["image"],
            "first_frame_review": bind(anchor / "review.json"), "implementation": implementation(),
            "settings": {"duration": contract["generation_information"]["duration_seconds"], "resolution": "720p", "ratio": "adaptive", "generate_audio": False},
            "generation_scope": contract["generation_scope"], "automatic_fallback": False})
        submission = submit_video(root, budget, request)
        write(directory / "submission.json", submission)
        deadline = time.monotonic() + read(root / "experiment.json")["per_video_poll_deadline_seconds"]
        while True:
            state = poll_video(root, submission, directory / "human.mp4")
            replace_state(directory / "progress.json", {"status": state["status"], "task_id": submission["task_id"],
                                                       "updated_unix": time.time(), "resubmit": False})
            if state["status"] in {"downloaded", "failed", "expired", "cancelled"}:
                budget.settle_video(submission["stage"])
                break
            if time.monotonic() >= deadline:
                return {**result, "status": "waiting_existing_video", "task_id": submission["task_id"], "resubmit": False}
            time.sleep(8)
        if state["status"] != "downloaded":
            result.update(status="video_provider_failed", provider_status=state["status"], video=None)
        else:
            result["video"] = state["video"]
            final = mechanism_review.video(root, budget, directory, contract)
            result.update(status="sampled_visual_pass" if final["gate"]["passed_in_sampled_visual_scope"] else "generated_video_rejected",
                          gate=final["gate"], video_review=bind(directory / "review.json"))
        result["invocation_elapsed_seconds"] = time.time() - start
        write(output, result)
        return result


def run(root):
    from .billing import MeteredBudget, snapshot, report
    from .pipeline import release_id
    root = Path(root)
    release = release_id()
    with (root / ".paid_batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = MeteredBudget(root)
        snapshot(root, root / "billing/campaign_before.json")
        rows = []
        for record in read(root / "compiled" / release / "index.json")["records"]:
            replace_state(root / "progress.json", {"status": "running", "pid": os.getpid(), "case": record["case"],
                                                   "completed": len(rows), "total": 6, "release": release, "updated_unix": time.time()})
            try:
                result = run_case(root, record, budget)
            except Exception as exc:
                # Runtime state must not remain "running" after a worker exits.
                # The immutable stage ledger carries the original failure.
                replace_state(root / "progress.json", {"status": "stopped_on_exception", "pid": None,
                    "case": record["case"], "completed": len(rows), "total": 6, "release": release,
                    "exception_type": type(exc).__name__, "code": getattr(exc, "code", "operation_failed"),
                    "automatic_retry": False, "updated_unix": time.time()})
                raise
            rows.append(result)
            print(json.dumps({"case": record["case"], "status": result["status"], "usage": budget.usage()}, ensure_ascii=False), flush=True)
            if result["status"] == "waiting_existing_video":
                return {"status": "waiting_existing_video", "cases": rows}
        after = snapshot(root, root / "billing/campaign_after.json")
        report(root, root / "billing/usage_report.json")
        write(root / "batch_results.json", {"release": release, "cases": rows, "complete": len(rows) == 6,
                                           "manual_downstream_annotation_count": 0, "budget_usage": budget.usage()})
        replace_state(root / "progress.json", {"status": "complete", "pid": None, "completed": len(rows), "total": 6,
                                               "release": release, "updated_unix": time.time()})
        return {"status": "complete", "case_statuses": {r["case"]: r["status"] for r in rows}, "budget": budget.usage()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "compile", "run"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--ancestor", type=Path)
    args = p.parse_args()
    activate()
    if args.command == "prepare":
        require(args.ancestor is not None, "ancestor_required")
        result = prepare(args.root, args.ancestor)
    elif args.command == "compile":
        result = compile_all(args.root)
    else:
        result = run(args.root)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
