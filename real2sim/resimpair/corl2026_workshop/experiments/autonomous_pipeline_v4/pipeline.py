"""Source contracts + planner export -> generation -> independent model observations.

No repair-loop branches, hand-filled review JSON, or per-task prompt overrides.
A semantic failure is terminal. Changes belong to a new shared software release.
"""
from __future__ import annotations

import fcntl
from pathlib import Path
import time

from .core import bind, checked, digest, implementation, read, replace_state, require, write, verify_implementation
from .budget import Budget
from .models import generate_image, submit_video, poll_video
from .prompts import anchor_prompt, video_prompt
from . import review
from .timing import span, traced


def release_id():
    path = Path(__file__).parent.parent / "release.json"
    require(path.exists(), "Execute an immutable release snapshot, not the mutable development tree")
    manifest = read(path)
    for bound in manifest["files"]:
        checked(bound)
    return manifest["release"]


def terminal(directory, status, stage, contract, evidence=None, diagnosis=None):
    result = {"schema": "resimpair_autonomous_outcome_v4", "status": status, "terminal_stage": stage,
              "contract": contract, "evidence": evidence, "diagnosis": diagnosis,
              "execution_completed_without_downstream_human_annotation": True,
              "manual_downstream_annotation_count": 0, "automatic_semantic_retry_count": 0,
              "model_quality_success": status == "sampled_visual_pass", "training_admitted": False,
              "all_frame_or_physical_human_guarantee": False}
    write(directory / "outcome.json", result)
    return result


def one_step(root, case_id, attempt, *, execute):
    root = Path(root)
    release = release_id()
    directory = root / "runs" / case_id / release
    directory.mkdir(parents=True, exist_ok=True)
    outcome = directory / "outcome.json"
    if outcome.exists():
        record = read(outcome)
        if record["contract"]:
            checked(record["contract"])
        return record
    previous = sorted((root / "runs" / case_id).glob("*/outcome.json"))
    if previous:
        prior = previous[-1]
        record = read(prior)
        if record.get("contract"):
            checked(record["contract"])
        return {"status": "existing_terminal_run_preserved", "original_status": record["status"],
                "original_outcome": bind(prior), "automatic_semantic_retry": False,
                "next_generation_requires": "a_new_explicitly_registered_experiment_case"}
    from .compiler import resolve_attempt, resolve_bundle, compiled_path, completed_contract
    attempt = resolve_attempt(root, case_id, attempt)
    contract_path = compiled_path(root, case_id, attempt)
    if not contract_path.exists() or not (contract_path.parent / "media_validation.json").exists():
        from .compiler import compile_execution
        bundle_path = resolve_bundle(root, case_id, attempt)
        if not bundle_path.exists():
            return {"status": "awaiting_planner_bundle", "case": case_id,
                    "only_permitted_manual_boundary": "motion_planning"}
        try:
            with span(directory, "contract_compile_and_media_validation"):
                compile_execution(root, case_id, attempt)
        except (ValueError, FileNotFoundError) as exc:
            return terminal(directory, "compiler_rejected", "execution_interface", None,
                            evidence=bind(bundle_path), diagnosis={"code": str(exc),
                            "next_development_action": "change_shared_export_or_compiler_in_a_new_release; do_not_fill_task_annotation"})
    contract = completed_contract(contract_path)
    verify_implementation(contract["code"])
    for key in ["execution", "initial_image", "final_image", "simulation_video", "other_simulation_video"]:
        checked(contract[key])
    for source in contract["source_mapping"]["sources"]:
        checked(source["contract"])
    budget = Budget(root / "budget.json")
    initial_prompt = anchor_prompt(contract["initial_scene_information"])
    # Different goals sharing identical initial pixels and scene information can
    # use the same automatically reviewed human initial image.
    model_config = bind(root / "config.json")
    anchor_id = digest({"initial_sha": contract["initial_image"]["sha256"], "prompt": initial_prompt,
                        "model_config": model_config, "review_implementation": implementation()["review.py"],
                        "review_checks": review.anchor_checks(contract["initial_scene_information"])})[:24]
    anchor_dir = root / "anchors" / anchor_id
    if not execute:
        return {"status": "prepared", "contract": bind(contract_path), "anchor_id": anchor_id,
                "generation_mode": "first_frame_plus_compiled_task", "budget": budget.usage()}
    if not (directory / "video_request.json").exists():
        needed = {"image": int(not (anchor_dir / "human.provenance.json").exists()), "video": 1,
                  "review": 1 + int(not (anchor_dir / "review.json").exists())}
        remaining = budget.usage()["new_remaining"]
        if any(needed[k] > remaining[k] for k in needed):
            return {"status": "awaiting_budget_capacity", "required_new_calls": needed,
                    "remaining_calls": remaining, "calls_started_by_this_invocation": 0,
                    "contract": bind(contract_path), "semantic_failure": False}
    request_path = anchor_dir / "image_request.json"
    if not request_path.exists():
        write(request_path, {"schema": "resimpair_initial_request_v4", "prompt": initial_prompt,
                            "model_config": model_config,
                            "images": [contract["initial_image"]], "initial_scene": contract["initial_scene_information"],
                            "implementation": implementation(), "task_specific_prompt_override": None})
    else:
        request = read(request_path)
        require(request["prompt"] == initial_prompt and request["images"][0]["sha256"] == contract["initial_image"]["sha256"],
                "Shared initial condition mismatch")
    with span(anchor_dir, "anchor_generate", cache_hit=(anchor_dir / "human.provenance.json").exists()):
        image = generate_image(root, budget, request_path, anchor_dir / "human.png")
    with span(anchor_dir, "anchor_review", cache_hit=(anchor_dir / "review.json").exists()):
        initial_review = review.anchor(root, budget, anchor_dir, image, contract["initial_scene_information"],
                                      read(request_path)["images"][0])
    if not initial_review["gate"]["passed_in_sampled_visual_scope"]:
        return terminal(directory, "initial_image_rejected", "initial_visual_review", bind(contract_path),
                        evidence=bind(anchor_dir / "review.json"),
                        diagnosis={"gate": initial_review["gate"],
                                   "next_development_action": "revise_shared_actor_or_scene_condition_or_model; this_run_has_ended"})
    video_request = directory / "video_request.json"
    write(video_request, {"schema": "resimpair_video_request_v4", "contract": bind(contract_path),
                          "model_config": model_config,
                          "prompt": video_prompt(contract), "first_frame": image["image"],
                          "first_frame_review": bind(anchor_dir / "review.json"), "implementation": implementation(),
                          "settings": {"duration": contract["generation_information"]["duration_seconds"], "resolution": "720p",
                                       "ratio": "adaptive", "generate_audio": False},
                          "mode": "predeclared_first_frame_and_task_contract", "automatic_fallback": False,
                          "condition_consumption": {"compiled_events_states_roles": "actual_text_payload",
                                                    "approved_initial_image": "actual_first_frame_image_bytes"}})
    submission = submit_video(root, budget, video_request)
    write(directory / "submission.json", submission)
    poll = poll_video(root, submission, directory / "human.mp4")
    replace_state(directory / "progress.json", {"case": case_id, "release": release, "status": poll["status"],
                                               "task_id": submission["task_id"], "budget": budget.usage()})
    if poll["status"] in {"failed", "expired", "cancelled"}:
        return terminal(directory, "video_provider_failed", "video_generation", bind(contract_path),
                        evidence=bind(directory / "submission.json"), diagnosis={"provider_status": poll["status"]})
    if poll["status"] != "downloaded":
        return {"status": "waiting_existing_video", "case": case_id, "task_id": submission["task_id"],
                "observed_provider_status": poll["status"], "resubmit": False}
    final = review.video(root, budget, directory, contract)
    return terminal(directory, "sampled_visual_pass" if final["gate"]["passed_in_sampled_visual_scope"] else "generated_video_rejected",
                    "generated_visual_review", bind(contract_path), evidence=bind(directory / "review.json"),
                    diagnosis=None if final["gate"]["passed_in_sampled_visual_scope"] else {
                        "gate": final["gate"], "next_development_action": "aggregate_failure_across_tasks_then_version_shared_conditions_or_model; no_sample_specific_repair"})


@traced("pipeline_invocation", lambda root, case_id, attempt=None, execute=False:
        Path(root) / "runs" / case_id / release_id(), cache_file="outcome.json")
def run(root, case_id, attempt=None, *, execute=False):
    root = Path(root)
    release = release_id()
    directory = root / "runs" / case_id / release
    directory.mkdir(parents=True, exist_ok=True)
    # A second CLI invocation can inspect/restore results, never concurrently
    # choose another generation path or spend on the same task.
    with (directory / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 900
        while True:
            try:
                result = one_step(root, case_id, attempt, execute=execute)
            except Exception as exc:
                if isinstance(exc, BlockingIOError):
                    raise
                stable = {"exception_type": type(exc).__name__, "code": getattr(exc, "code", "local_or_transport_error"),
                          "detail": str(exc) if isinstance(exc, (ValueError, FileNotFoundError, RuntimeError)) and not hasattr(exc, "code") else None}
                # A known video task is resumed even when its poll/download
                # temporarily fails. Do not convert an observation failure into
                # a terminal provider result or erase its task ID.
                submission = directory / "submission.json"
                if submission.exists() and not (directory / "human.mp4").exists():
                    result = {"status": "waiting_existing_video", "task_id": read(submission)["task_id"],
                              "observation_failure": stable, "resubmit": False}
                else:
                    result = terminal(directory, "runtime_stopped", "automatic_execution", None,
                                      diagnosis={**stable, "counted_calls_retained": True, "automatic_retry": False})
            replace_state(directory / "runtime.json", {"last_result": result, "updated_unix": time.time(),
                                                       "release": release, "manual_decisions": 0})
            if result["status"] != "waiting_existing_video" or not execute or time.monotonic() >= deadline:
                return result
            time.sleep(8)


def suite(root, *, execute=False):
    from .intake import cases
    outcomes = []
    for case in cases(root):
        if case["request_human_reference"]:
            result = run(root, case["id"], execute=execute)
            outcomes.append({"case": case["id"], "result": result})
            if result["status"] == "waiting_existing_video":
                break
    return {"outcomes": outcomes, "manual_stage_selection": False}
