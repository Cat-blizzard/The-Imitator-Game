"""Bounded multi-task scene campaign; exact episode bindings, no paid calls."""
from collections import Counter
import fcntl
import os
from pathlib import Path
import shutil
import time

from .core import activate, bind, checked, digest, read, require, replace_state, write


def prepare(root, ancestor, seed=2026102461):
    from .active_assets import object_spec, validate_descriptor
    from .budget import counts, Budget
    from .intake import register_case, sources
    from .library import make_scene
    from .scene_tasks import enumerate_tasks, select
    activate()
    root, ancestor = Path(root).resolve(), Path(ancestor).resolve()
    require(not root.exists(), "scene_pilot_output_must_be_new")
    require(shutil.disk_usage(root.parent).free > 100 * 2**30, "insufficient_scene_pilot_disk")
    descriptor = bind(ancestor / "assets/milk_carton_cooked_r2/descriptor.json")
    validate_descriptor(descriptor)
    old = read(ancestor / "budget.json")
    current = counts(old["ledger_path"])
    lineage = {k: old["ancestor_started_calls"][k] + current[k] for k in current}
    source_rows = sources(ancestor)
    root.mkdir()
    for source in source_rows:
        original = ancestor / "sources" / source["id"] / "source_contract.json"
        checked(read(original)["input"]["video"])
        folder = root / "sources" / source["id"]
        folder.mkdir(parents=True)
        (folder / "source_contract.json").symlink_to(original.resolve())
    write(root / "ancestor_usage.json", {"budget": bind(ancestor / "budget.json"), "started": current,
          "lineage_started": lineage, "old_physical_attempts_not_reopened": True})
    write(root / "budget.json", {"schema": "resimpair_cumulative_budget_v4",
        "campaign_id": "scene-task-local-20261024-v1", "ledger_path": str(root / "campaign_ledger"),
        "new_call_limits": {k: 0 for k in lineage}, "ancestor_started_calls": lineage,
        "ancestor_evidence": [bind(root / "ancestor_usage.json")], "lineage_cumulative_limits": lineage,
        "max_concurrent": 1, "timeout_seconds": 600, "currency_budget": 0, "automatic_retry": False,
        "authorization": "User approved starting the proposed 3 scenes x 3 tasks local pilot; no new paid generation allocation."})
    write(root / "config.json", read(ancestor / "config.json"))
    write(root / "experiment.json", {"schema": "resimpair_scene_task_pilot_v1", "sources": source_rows,
        "cases": [], "simulation_limit": 9, "planned_physical_executions": 9, "simulation_timeout_seconds": 180,
        "source_frame_budget": 24, "review_frame_budget": 49, "gpu": 2,
        "gpu_uuid": "GPU-807bd44d-36ea-31d0-427c-50e264428c67", "protected_gpus": [0, 1, 3],
        "no_training": True, "automatic_retry": False, "new_api_call_limits": {k: 0 for k in lineage},
        "source_split": "train", "source_independent_holdout_claimed": False,
        "validation_plan": "Freeze all nine selections before running; reset the SAME scene for every distinct task; no replacement/retry.",
        "scope": "three known tabletop layouts, one or two pick-place operations per task; not arbitrary contact mechanisms",
        "candidate_asset_descriptor": descriptor, "ancestor": bind(ancestor / "experiment.json")})
    groups = []
    for index in range(3):
        sid = "scene_" + "ABC"[index]
        scene = make_scene(sid, seed + index, nuisance=True, layout=True)
        primary = "red_block"
        if index:
            primary = "milk_carton"
            xy = scene["objects"][0]["position"][:2]
            scene["objects"][0] = object_spec(descriptor, primary, xy, .12 if index == 1 else -.15)
        scene_path = root / "scenes" / (sid + ".json")
        write(scene_path, scene)
        pool = enumerate_tasks(scene, required_object=primary)
        draw = select(pool, seed + 100 + index)
        pool_path = root / "selection" / (sid + ".json")
        write(pool_path, {"scene": bind(scene_path), "required_object": primary,
                         "pool": pool, "draw": draw, "outcomes_available_when_selected": False})
        members = []
        for selected in draw["selection"]:
            task = selected["task"]
            task_path = root / "tasks" / (task["id"] + ".json")
            write(task_path, task)
            row = register_case(root, task["id"], scene_path, task_path, make_reference=True)
            members.append({"case": row, "candidate_id": selected["candidate_id"], "goal_key": selected["goal_key"],
                            "attempt": "attempt_01", "max_attempts": 1})
        groups.append({"scene": bind(scene_path), "scene_id": sid, "selection": bind(pool_path),
                       "cases": members, "validation_scope": "new_layout_after_code_freeze; source_split_train"})
    write(root / "cohort.json", {"groups": groups, "physical_limit": 9, "replacement": False,
          "tasks_per_scene": 3, "reset_each_task": True, "source_split": "train"})
    Budget(root / "budget.json")
    return {"status": "frozen_inputs", "root": str(root), "cohort": bind(root / "cohort.json"),
            "cases": [c["case"]["id"] for g in groups for c in g["cases"]], "paid_call_limit": 0}


def validate_episode(member, bundle, contract=None):
    """Scene/task identity cannot be replaced by a layout-independent semantic key."""
    case = member["case"]
    require(bundle["case"] == case, "episode_case_binding_mismatch")
    require(bundle["scene"] == read(checked(case["scene"])), "episode_scene_binding_mismatch")
    require(bundle["task_intent"] == read(checked(case["task"])), "episode_task_binding_mismatch")
    if contract is not None:
        require(contract["task_id"] == case["id"], "contract_task_identity_mismatch")
        require(read(checked(contract["execution"])) == bundle, "contract_execution_binding_mismatch")
        require(contract["initial_image"] == bundle["cameras"]["opposite_high"]["initial_image"],
                "contract_initial_scene_mismatch")
        require(contract["simulation_video"] == bundle["cameras"]["opposite_high"]["video"],
                "contract_simulation_binding_mismatch")


def prepare_human(root, member):
    from .pipeline import run, release_id
    from .compiler import completed_contract
    from .prompts import anchor_prompt, video_prompt
    from .review import anchor_checks, video_checks
    case_id, attempt = member["case"]["id"], member["attempt"]
    prepared = run(root, case_id, attempt, execute=False)
    if prepared["status"] != "prepared":
        return prepared
    path = checked(prepared["contract"])
    contract = completed_contract(path)
    bundle = read(checked(contract["execution"]))
    validate_episode(member, bundle, contract)
    anchor = root / "prepared" / "anchors" / (prepared["anchor_id"] + ".json")
    write(anchor, {"schema": "resimpair_unsubmitted_anchor_v1", "anchor_id": prepared["anchor_id"],
                  "initial_image_sha256": contract["initial_image"]["sha256"],
                  "prompt": anchor_prompt(contract["initial_scene_information"]),
                  "review_checks": anchor_checks(contract["initial_scene_information"]),
                  "model_config": bind(root / "config.json"), "submitted": False})
    proposal = root / "prepared" / "tasks" / (case_id + ".json")
    write(proposal, {"schema": "resimpair_unsubmitted_task_video_v1", "contract": bind(path),
                    "shared_anchor_plan": bind(anchor), "initial_image": contract["initial_image"],
                    "prompt": video_prompt(contract), "review_checks": video_checks(contract),
                    "release": release_id(), "submitted": False,
                    "submission_requires": "new cumulative budget AND a generated, independently approved human anchor"})
    # Exercise the real admission path at the immutable zero budget. No model request runs.
    blocked = run(root, case_id, attempt, execute=True)
    require(blocked["status"] == "awaiting_budget_capacity" and blocked["calls_started_by_this_invocation"] == 0,
            "zero_budget_admission_did_not_block")
    return {"status": "simulation_qualified_human_prepared", "contract": bind(path),
            "media_validation": bind(path.parent / "media_validation.json"),
            "anchor_id": prepared["anchor_id"], "task_proposal": bind(proposal),
            "budget_gate": blocked, "human_video": None, "training_admitted": False}


def run_batch(root):
    from .budget import Budget
    from .pipeline import release_id
    from .simulation import launch
    root = Path(root).resolve()
    release = release_id()
    require(all(v == 0 for v in Budget(root / "budget.json").spec["new_call_limits"].values()), "local_pilot_requires_zero_api_budget")
    cohort = read(root / "cohort.json")
    all_members = [c for g in cohort["groups"] for c in g["cases"]]
    require(len(all_members) == cohort["physical_limit"] == 9, "frozen_nine_case_denominator_required")
    with (root / ".scene_batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = []
        for member in all_members:
            case_id, attempt = member["case"]["id"], member["attempt"]
            out = root / "batch_results" / release / (case_id + ".json")
            if out.exists():
                rows.append(read(out))
                continue
            replace_state(root / "batch_progress.json", {"status": "running", "pid": os.getpid(), "release": release,
                          "case": case_id, "completed": len(rows), "total": 9, "updated_unix": time.time()})
            start = time.monotonic()
            launch(root, case_id, attempt)
            directory = root / "executions" / case_id / attempt
            terminal = read(directory / "terminal.json")
            result = {"case": case_id, "attempt": attempt, "terminal": bind(directory / "terminal.json"),
                      "simulation_seconds": terminal["elapsed_seconds"], "pipeline": None}
            if terminal["status"] == "executed":
                result["pipeline"] = prepare_human(root, member)
            else:
                result["pipeline"] = {"status": "simulation_failed", "original_failure_retained": True}
            result["local_case_elapsed_seconds"] = time.monotonic() - start
            write(out, result)
            rows.append(result)
            print({"case": case_id, "status": result["pipeline"]["status"], "completed": len(rows), "total": 9}, flush=True)
        replace_state(root / "batch_progress.json", {"status": "complete", "release": release, "completed": len(rows),
                      "total": 9, "updated_unix": time.time(), "active_worker": None})
    return package(root)


def package(root):
    from .pipeline import release_id
    from .compiler import completed_contract
    from .budget import Budget
    root = Path(root).resolve()
    cohort = read(root / "cohort.json")
    release = release_id()
    groups, statuses = [], Counter()
    for group in cohort["groups"]:
        members = []
        for selected in group["cases"]:
            case_id, attempt = selected["case"]["id"], selected["attempt"]
            result_path = root / "batch_results" / release / (case_id + ".json")
            result = read(result_path)
            bundle_path = root / "executions" / case_id / attempt / "execution_bundle.json"
            bundle = read(bundle_path) if bundle_path.exists() else None
            if bundle:
                validate_episode(selected, bundle)
            status = result["pipeline"]["status"]
            statuses[status] += 1
            row = {"case": selected["case"], "attempt": attempt, "status": status, "result": bind(result_path),
                   "execution": bind(bundle_path) if bundle else None, "human": {"status": "not_generated", "video": None},
                   "source_split": "train", "training_admitted": False}
            if bundle:
                row["initial_state"] = {"physics": bundle["initial_physics_sha256"], "objects": bundle["initial_object_state_sha256"]}
                row["cameras"] = bundle["cameras"]
            if status == "simulation_qualified_human_prepared":
                contract = completed_contract(checked(result["pipeline"]["contract"]))
                validate_episode(selected, bundle, contract)
                row.update({k: result["pipeline"][k] for k in ["contract", "media_validation", "anchor_id", "task_proposal"]})
                row["source_mapping"] = contract["source_mapping"]
            members.append(row)
        recorded = [m for m in members if m["execution"]]
        same_state = bool(recorded) and len(recorded) == len(members) and len({digest(m["initial_state"]) for m in recorded}) == 1
        same_rgb = {name: bool(recorded) and len(recorded) == len(members) and
                    len({m["cameras"][name]["initial_image"]["sha256"] for m in recorded}) == 1
                    for name in ["robot_side", "opposite_high"]}
        anchors = sorted({m["anchor_id"] for m in members if "anchor_id" in m})
        groups.append({"scene": group["scene"], "scene_id": group["scene_id"], "members": members,
                       "same_initial_native_state": same_state, "same_initial_rgb": same_rgb,
                       "prepared_anchor_ids": anchors, "fully_simulation_qualified": all("contract" in m for m in members),
                       "complete_human_robot_pairs": 0})
    result = {"schema": "resimpair_exact_scene_task_package_v1", "cohort": bind(root / "cohort.json"),
              "release": bind(root / "releases" / release / "release.json"), "groups": groups,
              "counts": {"planned_cases": 9, "statuses": dict(statuses), "scene_groups": len(groups),
                         "fully_simulation_qualified_groups": sum(g["fully_simulation_qualified"] for g in groups),
                         "complete_human_robot_pairs": 0}, "budget": Budget(root / "budget.json").usage(),
              "semantic_key_cross_layout_video_reuse": False, "training_admitted": False,
              "scope": "bounded known adapters and frozen layouts; no random-task population success-rate claim"}
    write(root / "packages" / release / "scene_tasks.json", result)
    return {"package": bind(root / "packages" / release / "scene_tasks.json"), "counts": result["counts"]}
