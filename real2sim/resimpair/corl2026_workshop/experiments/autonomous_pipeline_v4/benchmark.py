"""Frozen stratified capability trial, retaining unsupported inputs in its denominator.

Layout cases use existing automatically observed source skills. Asset cases
probe actual local asset records. New-mechanism cases are dataset-instruction
probes, NOT newly interpreted human videos or automatically verified semantics.
No case is replaced, repaired, or retried after the draw.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import re
import shutil

from .core import bind, checked, digest, implementation, read, require, write
from .library import TASK_TEMPLATES, instantiate, make_scene
from .timing import span

VERSION = "resimpair_stratified_capability_v1"
STRATA = ("known_skill_new_layout", "known_mechanism_new_asset", "new_mechanism_instruction_probe")
from resimpair_runtime import ROOT
ASSETS = ROOT / 'runtime/household_catalog.json'
TASK_INDEX = ROOT / 'assets/task_instructions.json'
MECHANISMS = {
    "fluid_pour": r"\bpour\b", "stir_tool_contact": r"\bstir\b",
    "deformable_fold": r"\bfold the\b", "cut_material": r"\bcut the\b",
    "fluid_filter": r"\bfilter\b", "grinding": r"\bgrind\b",
    "box_sealing": r"\bseal\b|\bsealing\b", "instrument_weighing": r"\bweigh\b",
    "barcode_sensing": r"\bbarcode\b", "washing_wiping": r"\bwash\b|\bclean\b",
}


def candidate_pool(asset_catalog, task_index):
    pool = []
    # Explicit finite population: this is not a random sample of every possible
    # real-world task, and the layout range remains the existing library's range.
    for family in sorted(TASK_TEMPLATES):
        for variant in range(12):
            pool.append({"id": f"layout_{family}_{variant:02d}", "stratum": STRATA[0],
                         "family": family, "variant": variant, "scene_seed": 2026102400 + variant,
                         "required_skills": TASK_TEMPLATES[family]["required_skills"]})
    for asset in sorted(asset_catalog["assets"], key=lambda a: a["uid"]):
        pool.append({"id": "asset_relocate_" + asset["category"], "stratum": STRATA[1],
                     "family": "relocate", "required_skills": ["pick_place"],
                     "desired_role": "active_movable_task_object", "asset": deepcopy(asset)})
    for task_id, description in sorted(task_index.items()):
        goal = description.split(";")[0].strip()
        required = sorted(key for key, pattern in MECHANISMS.items() if re.search(pattern, goal, re.I))
        if required:
            pool.append({"id": "mechanism_" + task_id, "stratum": STRATA[2],
                         "dataset_task_id": task_id, "dataset_goal": goal, "mechanism_hints": required,
                         "source_scope": "dataset instruction only; not supplied to a source model or downstream generator",
                         "human_video_binding": None, "source_semantics_independently_verified": False})
    require(len({r["id"] for r in pool}) == len(pool), "duplicate_benchmark_candidate")
    return pool


def draw(pool, seed, per_stratum=3):
    require(type(seed) is int and type(per_stratum) is int and per_stratum > 0, "invalid_draw_configuration")
    selected, counts = [], {}
    for stratum in STRATA:
        candidates = [r for r in pool if r["stratum"] == stratum]
        require(len(candidates) >= per_stratum, "insufficient_stratum_population")
        ranked = sorted(candidates, key=lambda r: (digest({"seed": seed, "stratum": stratum, "id": r["id"]}), r["id"]))
        counts[stratum] = len(candidates)
        for order, candidate in enumerate(ranked[:per_stratum]):
            selected.append({**deepcopy(candidate), "selection_rank": order,
                             "selection_hash": digest({"seed": seed, "stratum": stratum, "id": candidate["id"]}),
                             "case_id": "strata_20261023_" + candidate["id"]})
    return {"seed": seed, "per_stratum": per_stratum, "population_counts": counts,
            "selection": selected, "replacement_after_failure": False,
            "rule": "lowest SHA256(seed,stratum,id), without replacement; canonical IDs fixed before checking outcomes"}


def layout_input(row):
    scene = make_scene("strata_scene_" + str(row["variant"]), row["scene_seed"], nuisance=True, layout=True)
    family = row["family"]
    object_ids = ["red_block"] if family == "relocate" else ["red_block", "blue_block", "yellow_block"] if family in {"arrange", "stack"} else ["red_block", "blue_block"]
    supports = (["left_tray"] if family in {"relocate", "pack"} else ["left_tray", "right_tray"]
                if family == "sort" else ["platform"] if family == "stack" else [])
    return scene, instantiate(family, row["case_id"], scene, object_ids, supports)


def screen(row, root):
    """Static eligibility never claims physical success or human-video quality."""
    if row["stratum"] == STRATA[0]:
        from .source import select_sources
        from .planning_protocols import pick_place
        from .simulation import targets
        scene, task = layout_input(row)
        source = select_sources(root, task["required_skills"])
        ids = [o["id"] for o in scene["objects"]]
        supports = ids + [s["id"] for s in scene["supports"]] + ["table"]
        plan = pick_place(scene, task, ids, supports, targets(scene, task))
        return {"status": "ready_for_one_physical_trial", "blockers": [],
                "scene": scene, "task": task, "protocol": plan, "source_mapping": source,
                "physical_qualification": "unknown_until_execution", "exact_source_episode_equivalence": False}
    if row["stratum"] == STRATA[1]:
        asset = row["asset"]
        files = [{"path": path, "sha256": value} for path, value in asset["files"].items()]
        for item in files:
            checked(item)
        # Keep a static visual asset's evidence scope; do not invent a mass,
        # grasp frame, mechanism, or active qualification to raise coverage.
        require("static untouched distractor" in asset["intended_use"], "asset_scope_changed_requires_new_benchmark_version")
        return {"status": "needs_active_asset_adapter", "blockers": [
                    "current_tabletop_intake_and_solver_require_declared_box_geometry",
                    "selected_asset_has_static_distractor_evidence_only",
                    "no_bound_active_grasp_mass_and_contact_adapter_for_this_asset"],
                "asset_files": files, "asset_uid": asset["uid"], "category": asset["category"],
                "collision_policy": asset["collision_policy"],
                "no_visual_or_collision_substitution": True, "physical_qualification": "not_attempted"}
    require(row["stratum"] == STRATA[2], "unknown_benchmark_stratum")
    return {"status": "needs_source_interpretation_before_mechanism_decision", "blockers": [
                "source_video_window_and_automatic_source_contract_not_yet_available_for_this_probe",
                "instruction_keyword_hints_cannot_establish_required_mechanism",
                "hinted_mechanisms_have_no_registered_planner_and_native_acceptance"],
            "mechanism_hints": row["mechanism_hints"], "physical_qualification": "not_attempted",
            "actual_required_mechanism": "unknown", "stratum_assignment_visually_verified": False,
            "scope": "instruction-level capability probe only; neither proven unsupported task nor a visual source-extraction failure"}


def prepare(root, ancestor, seed=2026102321):
    from .budget import counts
    from .intake import sources
    root, ancestor = Path(root), Path(ancestor)
    require(not (root / "experiment.json").exists(), "benchmark_already_registered_no_reset")
    require(shutil.disk_usage(root).free > 100 * 2**30, "insufficient_output_disk")
    with span(root, "benchmark_preparation"):
        catalog, instructions = read(ASSETS), read(TASK_INDEX)
        write(root / "input_snapshots/assets.json", catalog)
        write(root / "input_snapshots/dataset_task_instructions.json", instructions)
        pool = candidate_pool(catalog, instructions)
        write(root / "population.json", {"schema": VERSION, "candidates": pool,
            "asset_catalog": bind(ASSETS), "task_instruction_catalog": bind(TASK_INDEX),
            "new_mechanism_labels_are_partition_hints_only": True})
        chosen = draw(pool, seed)
        write(root / "cohort.json", {"schema": VERSION, "population": bind(root / "population.json"), **chosen,
            "denominator": len(chosen["selection"]), "automatic_redraw": False,
            "evaluation_scope": "layout trials plus asset/instruction capability probes; not an unbiased all-source-video success rate",
            "split_policy": "all reused source sessions remain train; new layouts held out from generator tuning, not a separate training-validation split"})
        old = read(ancestor / "budget.json")
        started = counts(old["ledger_path"])
        lineage = {k: old["ancestor_started_calls"][k] + started[k] for k in started}
        source_rows = sources(ancestor)
        source_bindings = []
        for item in source_rows:
            path = ancestor / "sources" / item["id"] / "source_contract.json"
            record = read(path)
            require(record["source_split"] == "train", "unexpected_source_split")
            checked(record["input"]["video"])
            dest = root / "sources" / item["id"]
            dest.mkdir(parents=True)
            (dest / "source_contract.json").symlink_to(path.resolve())
            source_bindings.append(bind(path))
        write(root / "ancestor_usage.json", {"budget": bind(ancestor / "budget.json"), "started": started,
            "lineage_started": lineage, "source_contracts_reused": source_bindings,
            "no_old_terminal_case_reopened": True})
        write(root / "budget.json", {"schema": "resimpair_cumulative_budget_v4", "campaign_id": "stratified-task-local-20261023-v1",
            "ledger_path": str((root / "campaign_ledger").resolve()), "new_call_limits": {k: 0 for k in lineage},
            "ancestor_started_calls": lineage, "lineage_cumulative_limits": lineage,
            "ancestor_evidence": [bind(root / "ancestor_usage.json")], "max_concurrent": 1, "timeout_seconds": 600,
            "automatic_retry": False, "currency_budget": 0,
            "authorization": "User requested starting stratified trials; no new paid allocation supplied, all API limits remain zero."})
        write(root / "config.json", read(ancestor / "config.json"))
        write(root / "experiment.json", {"schema": "resimpair_stratified_experiment_v1", "cohort": bind(root / "cohort.json"),
            "sources": source_rows, "cases": [], "source_frame_budget": 24, "review_frame_budget": 49,
            "simulation_limit": 3, "planned_physical_executions": 3, "simulation_timeout_seconds": 180,
            "max_video_poll_seconds_per_invocation": 45, "gpu": 2,
            "gpu_uuid": "GPU-807bd44d-36ea-31d0-427c-50e264428c67", "protected_gpus": [0, 1, 3],
            "no_training": True, "automatic_retry": False, "new_api_call_limits": {k: 0 for k in lineage},
            "scope": "new registered cases only; one physical attempt per eligible sampled layout; no asset or mechanism substitution",
            "validation_rules_frozen_before_execution": True,
            "manual_downstream_semantic_annotation_allowed": False,
            "ancestor": bind(ancestor / "experiment.json")})
    return {"cohort": str(root / "cohort.json"), "population_counts": chosen["population_counts"],
            "selected": [{k: r[k] for k in ("case_id", "stratum")} for r in chosen["selection"]],
            "new_api_limit": 0, "physical_attempt_limit": 3}


def preflight(root):
    from .pipeline import release_id
    from .intake import register_case
    root = Path(root)
    release = release_id()
    directory = root / "preflight" / release
    if (directory / "summary.json").exists():
        return read(directory / "summary.json")
    cohort = read(checked(read(root / "experiment.json")["cohort"]))
    records = []
    for row in cohort["selection"]:
        path = directory / (row["case_id"] + ".json")
        if path.exists():
            result = read(path)
        else:
            start = perf_counter()
            with span(directory / row["case_id"], "capability_preflight"):
                try:
                    result = screen(row, root)
                except (ValueError, FileNotFoundError) as exc:
                    result = {"status": "preflight_failed", "blockers": [type(exc).__name__],
                              "physical_qualification": "not_attempted"}
                result = {**result, "case_id": row["case_id"], "stratum": row["stratum"],
                          "elapsed_seconds": perf_counter() - start, "cohort": bind(root / "cohort.json"),
                          "implementation": implementation(), "training_admitted": False}
            write(path, result)
        if result["status"] == "ready_for_one_physical_trial":
            scene, task = root / "scenes" / (row["case_id"] + ".json"), root / "tasks" / (row["case_id"] + ".json")
            write(scene, result["scene"])
            write(task, result["task"])
            registry = root / "registered_cases" / (row["case_id"] + ".json")
            if not registry.exists():
                register_case(root, row["case_id"], scene, task, make_reference=True)
        records.append({"case_id": row["case_id"], "stratum": row["stratum"], "status": result["status"], "evidence": bind(path)})
    result = {"schema": VERSION, "release": release, "cohort": bind(root / "cohort.json"), "denominator": len(records),
              "cases": records, "physical_trials_eligible": sum(r["status"] == "ready_for_one_physical_trial" for r in records),
              "physical_success_claimed": False, "new_paid_calls": 0}
    write(directory / "summary.json", result)
    return result
