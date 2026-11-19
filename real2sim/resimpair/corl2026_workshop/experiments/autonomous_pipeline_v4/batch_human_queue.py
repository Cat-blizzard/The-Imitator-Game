"""Finite synthetic-human queue from independently qualified native batch attempts.

`select` and `prepare` are entirely local. Prepare executes from a frozen human
release. `run --phase qualification` spends at most one shared image and two
task videos. Expansion requires both complete visual reviews to pass. No failed
case is replaced, retried, or relabelled as a real-source pair.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from .core import activate, bind, checked, digest, identifier, implementation, read, replace_state, require, write
from .mechanism_conditions import BATCH_AUDIT_VERSION, validate_batch_selection, validated_batch_audit
from .motion_comparison import prepared_contract, review_or_unknown, run_video
from .motion_reference import ComparisonBudget
from .pipeline import release_id

VERSION = "batch_qualified_native_human_queue_20261025_v1"
LIMITS = {"image": 4, "video": 8, "review": 16}
ANCESTORS = {"image": 28, "video": 20, "review": 44}
KIND_ORDER = ("spring_console", "hook_retrieval", "linear_faders", "rotary_indexer",
              "rocker_console", "granule_sweeping", "lever_lift", "gravity_gate")
POLL_DEADLINE_SECONDS = 900
MIGRATION_VERSION = "batch_human_pre_video_migration_v1"
PROTECTED_CONDITION_MODULES = (
    "models.py", "image_response.py", "prompts.py", "paid_pilot.py", "review.py", "mechanism_review.py",
    "review_transport.py", "hand_review.py", "hand_correspondence.py", "compiler.py", "generic_compiler.py",
    "mechanism_conditions.py", "evidence_plan.py", "event_protocol.py", "interface.py", "source.py",
    "motion_comparison.py", "budget.py", "billing.py", "media.py")
PROTECTED_REFERENCE_SYMBOLS = ("REFERENCE_INSTRUCTION", "ComparisonBudget", "derivative",
                               "reference_payload", "submit_reference")


def validate_budget(root):
    """Local admission only; do not construct a metered/network client here."""
    root = Path(root).resolve()
    spec = read(root / "budget.json")
    if spec.get('portable_campaign'):
        require(spec.get('schema') == 'resimpair_cumulative_budget_v4',
                'portable_budget_schema_invalid')
        registration = read(checked(spec['portable_registration']))
        limits = spec['new_call_limits']; ancestors = spec['ancestor_started_calls']
        require(set(limits) == set(ancestors) == {'image', 'video', 'review'}
                and all(type(v) is int and v >= 0 for v in [*limits.values(), *ancestors.values()]),
                'portable_nonnegative_call_counts_required')
        require(registration['new_call_limits'] == limits
                and spec['lineage_cumulative_limits'] == {k: limits[k]+ancestors[k] for k in limits}
                and spec['max_concurrent'] == 1 and spec['automatic_retry'] is False
                and Path(spec['ledger_path']).resolve() == root / 'campaign_ledger',
                'portable_budget_or_ledger_changed')
        require(spec['observed_quota_stop_limit'] == registration['observed_quota_stop_limit']
                and spec['nominal_usd_admission_limit'] == registration['nominal_usd_admission_limit']
                and 0 < spec['nominal_usd_admission_limit'] <= 60
                and spec['observed_quota_stop_limit'] == int(spec['nominal_usd_admission_limit']*500000)
                and all(type(spec['reserve_quota_before_new_call'][k]) is int
                        and spec['reserve_quota_before_new_call'][k] > 0 for k in limits),
                'portable_observed_credit_stop_required')
        for bound in spec['ancestor_evidence']: checked(bound)
        return spec
    require(spec["schema"] == "resimpair_cumulative_budget_v4"
            and spec["new_call_limits"] == LIMITS and spec["ancestor_started_calls"] == ANCESTORS
            and spec["max_concurrent"] == 1 and spec["automatic_retry"] is False,
            "batch_human_budget_or_lineage_changed")
    require(spec["lineage_cumulative_limits"] == {k: LIMITS[k]+ANCESTORS[k] for k in LIMITS}
            and Path(spec["ledger_path"]).resolve() == root / "campaign_ledger",
            "batch_human_budget_ledger_or_cumulative_mismatch")
    require(spec["observed_quota_stop_limit"] == 30_000_000
            and spec["nominal_usd_admission_limit"] == 60
            and all(type(spec["reserve_quota_before_new_call"][k]) is int
                    and spec["reserve_quota_before_new_call"][k] > 0 for k in LIMITS),
            "batch_human_observed_credit_stop_required")
    for bound in spec["ancestor_evidence"]:
        checked(bound)
    return spec


def _selection_for_row(root, row, audit_binding):
    """The native worker result locates a contract; only the auditor admits it."""
    case, attempt = row["case_record"], row["attempt"]
    identifier(case["id"]); identifier(attempt)
    candidates = []
    for path in sorted((root / "qualification").glob("*/" + case["id"] + "/result.json")):
        value = read(path)
        if value.get("attempt") == attempt and value.get("execution") == row["execution"]:
            candidates.append((path, value))
    require(len(candidates) == 1, "native_contract_result_missing_or_ambiguous")
    path, result = candidates[0]
    selection = {"case": case, "attempt": attempt, "execution": row["execution"],
                 "physical_contract": result["physics_contract"], "result": bind(path),
                 "rule": BATCH_AUDIT_VERSION, "independent_audit": audit_binding}
    bundle = read(checked(selection["execution"]))
    original = read(checked(selection["physical_contract"]))
    validate_batch_selection(selection, original, bundle)
    require(original["measured"]["qualification"]["passed"] is True, "native_contract_unqualified")
    mapping = original["source_mapping"]
    require(mapping["sources"] == [] and mapping["source_coverage"] == "missing"
            and mapping["allowed_for_human_generation"] is False,
            "batch_queue_scope_requires_explicit_missing_real_source")
    require(not list((root / "runs" / case["id"]).glob("**/video_request.json"))
            and not list((root / "runs" / case["id"]).glob("**/outcome.json")),
            "prior_human_case_must_not_be_reopened")
    return selection


def select_candidates(root, audit_path, max_scenes=4, scene_kinds=()):
    """Freeze up to four distinct mechanisms, two semantic tasks per exact scene."""
    root = Path(root).resolve()
    require(type(max_scenes) is int and 1 <= max_scenes <= 4, "batch_human_max_scenes_out_of_range")
    require(len(scene_kinds) == len(set(scene_kinds)), "duplicate_scene_kind_filter")
    audit_binding = bind(audit_path)
    report = validated_batch_audit(audit_binding, root)
    groups = defaultdict(list)
    excluded = []
    for row in sorted(report["cases"], key=lambda r: (r["case"], r.get("attempt", ""))):
        if not row.get("independent_qualified"):
            continue
        case = row["case_record"]
        scene = read(checked(case["scene"]))
        kind = scene["id"]
        if scene_kinds and kind not in scene_kinds:
            continue
        try:
            selection = _selection_for_row(root, row, audit_binding)
        except (ValueError, FileNotFoundError) as exc:
            excluded.append({"case": row["case"], "attempt": row["attempt"],
                             "reason": str(exc), "human_calls_started": 0})
            continue
        task = read(checked(case["task"]))
        selection.update(scene_kind=kind, scene_instance_id=case["scene_instance_id"],
                         split_group=case["split_group"], partition=case["partition"],
                         task_program_sha256=row["declared_program_sha256"],
                         initial_evidence=row["initial"],
                         selected_target_count=len(task["planning_parameters"].get("selected", task["object_ids"])),
                         trajectory_variation=task.get("trajectory_variation", {}))
        groups[(kind, case["scene_instance_id"], case["scene"]["sha256"])].append(selection)
    ranking = {kind: i for i, kind in enumerate(scene_kinds or KIND_ORDER)}
    selected = []; seen_kinds = set()
    for (kind, scene_id, _), members in sorted(groups.items(),
            key=lambda item: (ranking.get(item[0][0], len(ranking)), item[0][0], item[0][1])):
        if kind in seen_kinds or len(selected) == max_scenes:
            continue
        programs = {}
        for member in sorted(members, key=lambda x: (
                x["case"].get("phase") != "base", x["selected_target_count"],
                x["trajectory_variation"].get("index", 0), x["case"]["id"])):
            programs.setdefault(member["task_program_sha256"], member)
        if len(programs) < 2:
            excluded.append({"scene_instance_id": scene_id, "reason": "fewer_than_two_qualified_distinct_task_programs"})
            continue
        tasks = list(programs.values())[:2]
        require(tasks[0]["initial_evidence"] == tasks[1]["initial_evidence"], "shared_scene_initial_evidence_differs")
        require(tasks[0]["split_group"] == tasks[1]["split_group"]
                and tasks[0]["partition"] == tasks[1]["partition"], "shared_scene_partition_differs")
        selected.append({"scene_kind": kind, "scene_instance_id": scene_id,
                         "split_group": tasks[0]["split_group"], "partition": tasks[0]["partition"],
                         "selections": tasks})
        seen_kinds.add(kind)
    require(selected, "no_scene_with_two_independently_qualified_tasks")
    return {"schema": VERSION, "root": str(root), "independent_native_audit": audit_binding,
            "max_scenes": max_scenes, "scene_kind_filter": list(scene_kinds), "scenes": selected,
            "excluded_before_paid_calls": excluded, "native_audit_counts": report["counts"],
            "selection_policy": "fixed mechanism priority, first exact scene with two distinct qualified semantic programs; base trajectory then fewer selected targets then lexical case order; no outcome-driven replacement",
            "source_coverage": "missing", "pairing_scope": "synthetic_only", "sources": [],
            "formal_real_sim_human_pair": False, "training_admitted": False,
            "method": "motion_reference", "candidate_only": True,
            "poll_deadline_seconds": POLL_DEADLINE_SECONDS,
            "qualification": "first selected scene, one anchor and two task videos; expansion requires complete passing anchor and both complete passing video reviews",
            "automatic_semantic_retry": False, "manual_downstream_annotation_count": 0}


def _anchor_id(selection, contract, config, code):
    from .prompts import anchor_prompt
    return digest({"scene_instance_id": selection["scene_instance_id"],
        "initial_evidence": selection["initial_evidence"], "image_sha256": contract["initial_image"]["sha256"],
        "prompt": anchor_prompt(contract["initial_scene_information"]), "model_config": config,
        "implementation": code})[:24]


def prepare(root, audit_path, max_scenes=4, scene_kinds=()):
    """Compile all frozen selections and shared anchors, without API or credentials."""
    from .budget import counts
    from .paid_pilot import compact_video_prompt, shared_anchor_request
    from .prompts import anchor_prompt
    root = Path(root).resolve(); release = release_id()
    require(shutil.disk_usage(root).free > 20 * 2**30, "insufficient_disk_for_human_queue")
    spec = validate_budget(root)
    directory = root / "human_queue" / release
    index_path = directory / "index.json"
    if index_path.exists():
        prior = read(index_path)
        require(prior["selection"]["independent_native_audit"] == bind(audit_path)
                and prior["selection"]["max_scenes"] == max_scenes
                and prior["selection"]["scene_kind_filter"] == list(scene_kinds), "frozen_human_queue_selection_changed")
        validate_index(root, prior)
        return {"status": "already_prepared", "index": bind(index_path), "scenes": len(prior["groups"]),
                "tasks": len(prior["records"]), "paid_calls_started": 0}
    require(not any(counts(spec["ledger_path"]).values()), "freeze_human_selection_before_any_paid_call")
    require(not list((root / "human_queue").glob("*/index.json")), "another_human_queue_release_already_prepared")
    selection = select_candidates(root, audit_path, max_scenes, scene_kinds)
    write(directory / "selection.json", selection)
    records = []; groups = []
    for group in selection["scenes"]:
        group_records = []
        for selected in group["selections"]:
            case = selected["case"]["id"]
            contract_binding = prepared_contract(root, selected, case)
            contract = read(checked(contract_binding))
            prompt = anchor_prompt(contract["initial_scene_information"])
            anchor_id = _anchor_id(selected, contract, bind(root / "config.json"), implementation())
            anchor = root / "anchors" / anchor_id
            shared_anchor_request(anchor / "image_request.json", {"schema": "resimpair_initial_request_v4",
                "prompt": prompt, "model_config": bind(root / "config.json"),
                "images": [contract["initial_image"]], "initial_scene": contract["initial_scene_information"],
                "implementation": implementation(), "task_specific_prompt_override": None})
            record = {"key": case, "case": case, "attempt": selected["attempt"],
                "scene_kind": group["scene_kind"], "scene_instance_id": group["scene_instance_id"],
                "split_group": group["split_group"], "partition": group["partition"],
                "contract": contract_binding, "execution": selected["execution"],
                "independent_native_audit": selected["independent_audit"],
                "anchor_id": anchor_id, "anchor_directory": str(anchor),
                "image_request": bind(anchor / "image_request.json"), "base_prompt": compact_video_prompt(contract),
                "method": "motion_reference", "native_qualified": True,
                "source_coverage": "missing", "pairing_scope": "synthetic_only",
                "formal_real_sim_human_pair": False, "training_admitted": False}
            records.append(record); group_records.append(record)
        require(len(group_records) == 2 and len({r["anchor_id"] for r in group_records}) == 1,
                "same_scene_tasks_cannot_share_identical_anchor_conditions")
        groups.append({k: group[k] for k in ["scene_kind", "scene_instance_id", "split_group", "partition"]} |
                      {"anchor_id": group_records[0]["anchor_id"], "keys": [r["key"] for r in group_records]})
    index = {"schema": VERSION, "release": release, "selection": selection,
             "selection_binding": bind(directory / "selection.json"), "groups": groups, "records": records,
             "budget": bind(root / "budget.json"), "model_config": bind(root / "config.json"),
             "implementation": implementation(), "condition_implementation": implementation(),
             "condition_release": release, "new_call_ceiling": spec['new_call_limits'],
             "planned_calls": {"image": len(groups), "video": len(records), "review": len(groups)+len(records)},
             "full_review_interface_qualified": False, "candidate_only": True, "formal_pairs": 0}
    validate_index(root, index)
    write(index_path, index)
    return {"status": "prepared", "index": bind(index_path), "scenes": len(groups),
            "tasks": len(records), "planned_calls": index["planned_calls"], "paid_calls_started": 0}


def validate_index(root, index, *, expected_implementation=None):
    """Local immutable bindings and finite denominator; no transport construction."""
    from .paid_pilot import compact_video_prompt
    from .prompts import anchor_prompt
    root = Path(root).resolve(); validate_budget(root)
    require(index["schema"] == VERSION and index["selection"]["root"] == str(root), "human_queue_root_or_version_changed")
    require(index["budget"] == bind(root / "budget.json") and index["model_config"] == bind(root / "config.json"),
            "human_queue_budget_or_config_changed")
    expected_implementation = implementation() if expected_implementation is None else expected_implementation
    require(index["implementation"] == expected_implementation, "human_queue_executor_implementation_changed")
    condition_code = index.get("condition_implementation", index["implementation"])
    if "migration" in index:
        validate_migration_binding(root, index)
    else:
        require(condition_code == index["implementation"], "separate_conditions_require_explicit_migration")
    require(read(checked(index["selection_binding"])) == index["selection"], "human_queue_selection_changed")
    require(1 <= len(index["groups"]) <= 4 and len(index["records"]) == 2*len(index["groups"]),
            "human_queue_denominator_invalid")
    require(index["planned_calls"] == {"image": len(index["groups"]), "video": len(index["records"]),
                                     "review": len(index["groups"])+len(index["records"])}, "human_queue_call_plan_mismatch")
    require(index["selection"]["method"] == "motion_reference"
            and index["selection"]["poll_deadline_seconds"] == POLL_DEADLINE_SECONDS,
            "human_queue_conditioning_or_deadline_changed")
    keys = [r["key"] for r in index["records"]]
    require(len(keys) == len(set(keys)) and set(keys) == {k for g in index["groups"] for k in g["keys"]},
            "human_queue_task_group_binding_mismatch")
    frozen = [s for g in index["selection"]["scenes"] for s in g["selections"]]
    require([(r["case"], r["attempt"]) for r in index["records"]]
            == [(s["case"]["id"], s["attempt"]) for s in frozen], "human_queue_frozen_selection_identity_mismatch")
    for group in index["groups"]:
        members = [r for r in index["records"] if r["key"] in group["keys"]]
        require(len(group["keys"]) == len(members) == 2
                and all(r["anchor_id"] == group["anchor_id"]
                        and r["scene_instance_id"] == group["scene_instance_id"]
                        and r["split_group"] == group["split_group"] and r["partition"] == group["partition"]
                        for r in members), "human_queue_shared_scene_binding_mismatch")
    for record, selected in zip(index["records"], frozen):
        for name in ["contract", "execution", "independent_native_audit", "image_request"]:
            checked(record[name])
        require(record["key"] == record["case"] and record["method"] == "motion_reference"
                and record["execution"] == selected["execution"]
                and record["independent_native_audit"] == selected["independent_audit"],
                "human_queue_record_selection_binding_mismatch")
        require(all(record[k] == selected[k] for k in ["scene_kind", "scene_instance_id", "split_group", "partition"]),
                "human_queue_record_scene_binding_mismatch")
        contract = read(checked(record["contract"]))
        original = read(checked(selected["physical_contract"]))
        bundle = read(checked(selected["execution"]))
        validate_batch_selection(selected, original, bundle)
        require(contract["native_baseline_contract"] == selected["physical_contract"]
                and contract["independent_native_audit"] == selected["independent_audit"]
                and contract["execution"] == selected["execution"]
                and contract["task_id"] == bundle["task_intent"]["id"],
                "human_queue_compiled_contract_selection_mismatch")
        scope = contract["generation_scope"]
        require(contract["source_mapping"] == original["source_mapping"]
                and scope["source_coverage"] == "missing" and scope["pairing_scope"] == "synthetic_only"
                and scope["formal_real_sim_human_pair"] is False and scope["training_admitted"] is False,
                "human_queue_source_scope_changed")
        require(record["base_prompt"] == compact_video_prompt(contract), "human_queue_video_prompt_changed")
        anchor_id = _anchor_id(selected, contract, index["model_config"], condition_code)
        anchor = root / "anchors" / anchor_id
        request = read(checked(record["image_request"]))
        require(record["anchor_id"] == anchor_id and Path(record["anchor_directory"]).resolve() == anchor
                and record["image_request"] == bind(anchor / "image_request.json"), "human_queue_anchor_location_changed")
        require(request["prompt"] == anchor_prompt(contract["initial_scene_information"])
                and request["initial_scene"] == contract["initial_scene_information"]
                and request["model_config"] == index["model_config"]
                and request["implementation"] == condition_code
                and request["task_specific_prompt_override"] is None
                and len(request["images"]) == 1
                and request["images"][0]["sha256"] == contract["initial_image"]["sha256"],
                "human_queue_anchor_condition_changed")
        checked(request["images"][0]); checked(contract["initial_image"])
    for bound in index["implementation"].values():
        checked(bound)
    for bound in condition_code.values():
        checked(bound)
    validated_batch_audit(index["selection"]["independent_native_audit"], root)


def frozen_release_implementation(root, release):
    identifier(release)
    folder = Path(root) / "releases" / release
    manifest = read(folder / "release.json")
    require(manifest["release"] == release, "human_migration_release_manifest_mismatch")
    for bound in manifest["files"]:
        checked(bound)
    package = (folder / "autonomous_pipeline_v4").resolve()
    return {Path(b["path"]).name: b for b in manifest["files"]
            if Path(b["path"]).parent.resolve() == package and Path(b["path"]).suffix == ".py"}


def _protected_symbols(binding):
    tree = ast.parse(Path(checked(binding)).read_text())
    selected = {}
    for node in tree.body:
        names = ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                 [n.id for n in node.targets if isinstance(n, ast.Name)] if isinstance(node, ast.Assign) else [])
        for name in names:
            if name in PROTECTED_REFERENCE_SYMBOLS:
                require(name not in selected, "duplicate_reference_condition_symbol")
                selected[name] = digest(ast.dump(node, include_attributes=False))
    require(set(selected) == set(PROTECTED_REFERENCE_SYMBOLS), "reference_condition_symbol_missing")
    return selected


def validate_condition_compatibility(condition_code, executor_code):
    unchanged = {}
    for name in PROTECTED_CONDITION_MODULES:
        require(name in condition_code and name in executor_code, "human_migration_critical_module_missing:" + name)
        checked(condition_code[name]); checked(executor_code[name])
        require(condition_code[name]["sha256"] == executor_code[name]["sha256"],
                "human_migration_condition_or_review_changed:" + name)
        unchanged[name] = condition_code[name]["sha256"]
    before = _protected_symbols(condition_code["motion_reference.py"])
    after = _protected_symbols(executor_code["motion_reference.py"])
    require(before == after, "human_migration_reference_condition_changed")
    return {"unchanged_critical_module_sha256": unchanged, "unchanged_reference_symbol_sha256": before}


def _runtime_policy(root):
    root = Path(root)
    path = root / "human_runtime.json"; policy = read(path)
    require(policy["schema"] == "resimpair_human_runtime_v1" and policy["budget"] == bind(root / "budget.json"),
            "human_migration_runtime_budget_mismatch")
    limit = policy["media_upload_limit"]
    require(type(limit) is int and 0 < limit <= read(root / "budget.json")["new_call_limits"]["video"],
            "human_migration_runtime_upload_limit_invalid")
    public = checked(policy["provider_application"])
    require("https://imageproxy.zhongzhuan.chat/api/upload" in public.read_text(), "human_migration_public_upload_route_missing")
    return bind(path)


def migration_preconditions(root, index):
    """No model/upload request may be repeated or silently switched to a new code version."""
    root = Path(root)
    require(not list((root / "uploads").glob("**/upload_started.json")), "human_migration_upload_already_started")
    require(not list((root / "runs").glob("**/video_request.json"))
            and not list((root / "runs").glob("**/submission.json")), "human_migration_video_request_already_prepared")
    terminals = (list((root / "runs").glob("**/outcome.json"))
                 + list((root / "human_queue").glob("*/groups/*/outcome.json"))
                 + list((root / "human_queue").glob("*/*_results.json")))
    require(not terminals, "human_migration_semantic_terminal_must_not_reopen")
    anchors = {r["anchor_id"]: Path(r["anchor_directory"]) for r in index["records"]}
    allowed = {str(p / "image_request.json"): "image" for p in anchors.values()}
    allowed.update({str(p / "review_request.json"): "review" for p in anchors.values()})
    completed = []
    ledger = Path(read(root / "budget.json")["ledger_path"])
    for marker in sorted(ledger.glob("stages/*/call_started.json")):
        started = read(marker); stage = marker.parent
        require(started["kind"] != "video", "human_migration_video_already_started")
        require(started["kind"] in {"image", "review"} and (stage / "result.json").exists(),
                "human_migration_unresolved_submission")
        require(not (stage / "failure.json").exists() and not (stage / "local_validation_failure.json").exists(),
                "human_migration_failed_review_or_image_must_not_reopen")
        request = checked(started["request"])
        require(allowed.get(str(request)) == started["kind"], "human_migration_unrelated_paid_request")
        metadata = read(stage / "request_metadata.json")
        require(metadata["kind"] == started["kind"] and metadata["request"] == started["request"],
                "human_migration_call_request_binding_mismatch")
        result = read(stage / "result.json")
        require(result.get("status") == "succeeded", "human_migration_provider_result_not_complete")
        completed.append({"kind": started["kind"], "started": bind(marker), "request": started["request"],
                          "result": bind(stage / "result.json")})
    cached = []
    for anchor_id, folder in anchors.items():
        require(not (folder / "unusable_review.json").exists() and not list(folder.glob("*rejection*.json")),
                "human_migration_anchor_semantic_terminal_must_not_reopen")
        provenance = folder / "human.provenance.json"
        if not provenance.exists():
            require(not (folder / "human.png").exists() and not (folder / "review.json").exists(),
                    "human_migration_partial_anchor_requires_recovery")
            continue
        require((folder / "review.json").exists(), "human_migration_anchor_review_not_complete")
        image = read(provenance); reviewed = read(folder / "review.json")
        require(image["request"] == bind(folder / "image_request.json")
                and reviewed["candidate"] == image["image"]
                and reviewed["gate"]["passed_in_sampled_visual_scope"] is True,
                "human_migration_anchor_not_qualified")
        checked(image["image"]); checked(image["result"])
        for key in ["request", "result"]:
            checked(reviewed["model_evidence"][key])
        require(any(c["kind"] == "image" and c["request"] == image["request"] and c["result"] == image["result"] for c in completed)
                and any(c["kind"] == "review" and c["request"] == reviewed["model_evidence"]["request"]
                        and c["result"] == reviewed["model_evidence"]["result"] for c in completed),
                "human_migration_anchor_cache_not_bound_to_completed_calls")
        cached.append({"anchor_id": anchor_id, "image_provenance": bind(provenance), "image": image["image"],
                       "review": bind(folder / "review.json")})
    require(cached and len(completed) == 2*len(cached), "human_migration_requires_complete_anchor_and_review_pairs")
    return {"completed_prior_calls": completed, "cached_qualified_anchors": cached,
            "prior_video_submissions": 0, "prior_upload_submissions": 0, "prior_semantic_terminals": 0,
            "unknown_submissions": 0}


def validate_migration_binding(root, index):
    receipt = read(checked(index["migration"]))
    require(receipt["schema"] == MIGRATION_VERSION and receipt["to_release"] == index["release"],
            "human_migration_receipt_version_mismatch")
    previous = read(checked(receipt["previous_index"]))
    require(previous["release"] == receipt["from_release"], "human_migration_previous_release_mismatch")
    require(receipt["new_release"] == bind(Path(root) / "releases" / index["release"] / "release.json")
            and receipt["runtime_policy"] == _runtime_policy(root), "human_migration_runtime_or_release_changed")
    for key in ["selection", "groups", "records", "budget", "model_config", "planned_calls", "new_call_ceiling"]:
        require(index[key] == previous[key], "human_migration_frozen_input_changed:" + key)
    condition = previous.get("condition_implementation", previous["implementation"])
    require(index["condition_implementation"] == condition
            and index["condition_release"] == previous.get("condition_release", previous["release"]),
            "human_migration_original_conditions_changed")
    compatibility = validate_condition_compatibility(condition, index["implementation"])
    require(receipt["compatibility"] == compatibility, "human_migration_compatibility_receipt_mismatch")
    for call in receipt["preconditions"]["completed_prior_calls"]:
        for key in ["started", "request", "result"]:
            checked(call[key])
    for anchor in receipt["preconditions"]["cached_qualified_anchors"]:
        for key in ["image_provenance", "image", "review"]:
            checked(anchor[key])


def migrate(root, from_release):
    """Explicit local recovery before the first upload/video; retain all old artifacts."""
    root = Path(root).resolve(); target = release_id(); identifier(from_release)
    require(target != from_release, "human_migration_requires_distinct_frozen_release")
    prior_path = root / "human_queue" / from_release / "index.json"
    previous = read(prior_path)
    prior_code = frozen_release_implementation(root, from_release)
    target_code = frozen_release_implementation(root, target)
    require(previous["release"] == from_release and target_code == implementation(), "human_migration_executor_not_frozen")
    validate_index(root, previous, expected_implementation=prior_code)
    condition = previous.get("condition_implementation", previous["implementation"])
    compatibility = validate_condition_compatibility(condition, target_code)
    runtime = _runtime_policy(root)
    ledger = Path(read(root / "budget.json")["ledger_path"])
    with (root / ".paid_batch.lock").open("a+") as lock, (ledger / ".execution.lock").open("a+") as execution:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(execution, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preconditions = migration_preconditions(root, previous)
        folder = root / "human_queue" / target
        changed = {name: {"condition_sha256": condition.get(name, {}).get("sha256"),
                          "executor_sha256": target_code.get(name, {}).get("sha256")}
                   for name in sorted(set(condition) | set(target_code))
                   if condition.get(name, {}).get("sha256") != target_code.get(name, {}).get("sha256")}
        receipt = {"schema": MIGRATION_VERSION, "from_release": from_release, "to_release": target,
                   "previous_index": bind(prior_path), "new_release": bind(root / "releases" / target / "release.json"),
                   "runtime_policy": runtime, "compatibility": compatibility, "preconditions": preconditions,
                   "implementation_differences": changed, "condition_or_acceptance_change": False,
                   "recompiled_contracts": 0, "new_image_calls": 0, "new_review_calls": 0, "new_video_calls": 0,
                   "automatic_semantic_retry": False}
        write(folder / "migration.json", receipt)
        write(folder / "selection.json", previous["selection"])
        candidate = deepcopy(previous)
        candidate.update(release=target, implementation=target_code, condition_implementation=condition,
                         condition_release=previous.get("condition_release", previous["release"]),
                         selection_binding=bind(folder / "selection.json"), migration=bind(folder / "migration.json"))
        validate_index(root, candidate)
        write(folder / "index.json", candidate)
        return {"status": "migrated_without_generation", "index": bind(folder / "index.json"),
                "migration": bind(folder / "migration.json"), "cached_qualified_anchors": len(preconditions["cached_qualified_anchors"]),
                "paid_calls_started": 0, "recompiled_contracts": 0, "from_release": from_release, "to_release": target}


def _result_record(record, status, *, anchor=None, anchor_review=None, video=None, outcome=None, unknown=None):
    return {k: record[k] for k in ["case", "attempt", "scene_kind", "scene_instance_id", "split_group", "partition",
            "contract", "execution", "independent_native_audit"]} | {
        "status": status, "anchor": anchor, "anchor_review": anchor_review, "video": video, "outcome": outcome,
        "review_unknown": unknown, "native_qualified": True,
        "human_sampled_visual_qualified": status == "sampled_visual_pass",
        "source_coverage": "missing", "pairing_scope": "synthetic_only", "formal_real_sim_human_pair": False,
        "training_admitted": False, "automatic_semantic_retry_count": 0, "manual_downstream_annotation_count": 0}


def run_group(root, budget, index, group):
    """One shared anchor; an unknown/rejected anchor starts zero task videos."""
    from . import review
    from .models import generate_image
    root = Path(root)
    records = [next(r for r in index["records"] if r["key"] == k) for k in group["keys"]]
    directory = root / "human_queue" / index["release"] / "groups" / group["scene_instance_id"]
    if (directory / "outcome.json").exists():
        previous = read(directory / "outcome.json")
        require(previous["group"] == group, "terminal_human_queue_group_changed")
        return previous
    anchor = Path(records[0]["anchor_directory"])
    contract = read(checked(records[0]["contract"]))
    receipt = generate_image(root, budget, anchor / "image_request.json", anchor / "human.png")
    initial, unavailable = review_or_unknown(budget, anchor / "review_request.json",
        lambda: review.anchor(root, budget, anchor, receipt,
                             contract["initial_scene_information"], contract["initial_image"]))
    if unavailable:
        write(anchor / "unusable_review.json", unavailable)
        rows = [_result_record(r, "initial_image_review_unknown_video_unattempted", anchor=receipt["image"],
                unknown=bind(anchor / "unusable_review.json")) for r in records]
    elif not initial["gate"]["passed_in_sampled_visual_scope"]:
        rows = [_result_record(r, "initial_image_rejected_video_unattempted", anchor=receipt["image"],
                               anchor_review=bind(anchor / "review.json")) for r in records]
    else:
        rows = []
        for record in records:
            result = run_video(root, budget, record, "motion_reference", receipt["image"],
                               bind(anchor / "review.json"), poll_deadline_seconds=POLL_DEADLINE_SECONDS)
            if result["status"] == "waiting_existing_video":
                return {"status": "waiting_existing_video", "group": group, "rows": rows,
                        "waiting_case": record["case"], "submission": result["submission"], "resubmit": False}
            outcome_path = root / "runs" / record["key"] / "motion_reference" / index["release"] / "outcome.json"
            rows.append(_result_record(record, result["status"], anchor=receipt["image"],
                anchor_review=bind(anchor / "review.json"), video=result.get("video"),
                outcome=bind(outcome_path), unknown=result.get("review_failure")))
            print(json.dumps({"case": record["case"], "attempt": record["attempt"], "status": result["status"],
                              "usage": budget.usage()}, ensure_ascii=False), flush=True)
    value = {"schema": VERSION, "status": "terminal", "group": group, "rows": rows,
             "full_anchor_review_passed": bool(initial and initial["gate"]["passed_in_sampled_visual_scope"]),
             "two_complete_passing_video_reviews": len(rows) == 2 and all(r["human_sampled_visual_qualified"] for r in rows),
             "formal_pairs": 0, "training_admitted": False}
    write(directory / "outcome.json", value)
    return value


def qualification_decision(outcome):
    permitted = (outcome.get("status") == "terminal" and outcome.get("full_anchor_review_passed") is True
                 and outcome.get("two_complete_passing_video_reviews") is True
                 and len(outcome.get("rows", [])) == 2
                 and all(r.get("status") == "sampled_visual_pass" and r.get("outcome")
                         and r.get("video") and r.get("anchor_review") for r in outcome["rows"]))
    return {"proceed_to_frozen_expansion": bool(permitted),
            "reason": "complete_anchor_and_both_video_reviews_passed" if permitted else "full_small_batch_qualification_not_passed",
            "no_retry_or_replacement": True, "failure_does_not_establish_model_incapability": True,
            "formal_pairs": 0, "all_frame_human_physics_guarantee": False}


def validate_phase_result(root, index, index_path, phase, value):
    """A cached phase cannot substitute another task or an unbound video result."""
    root = Path(root); directory = Path(index_path).parent
    require(value["schema"] == VERSION and value["phase"] == phase and value["status"] == "phase_complete"
            and value["index"] == bind(index_path), "human_queue_cached_phase_binding_mismatch")
    expected = index["groups"][:1] if phase == "qualification" else index["groups"][1:]
    require(len(value["groups"]) == len(expected), "human_queue_cached_phase_denominator_mismatch")
    records = {r["key"]: r for r in index["records"]}
    for group, outcome in zip(expected, value["groups"]):
        path = directory / "groups" / group["scene_instance_id"] / "outcome.json"
        require(outcome == read(path) and outcome["group"] == group and outcome["status"] == "terminal"
                and [r["case"] for r in outcome["rows"]] == group["keys"],
                "human_queue_cached_group_binding_mismatch")
        for row in outcome["rows"]:
            record = records[row["case"]]
            require(all(row[k] == record[k] for k in ["case", "attempt", "contract", "execution", "independent_native_audit"]),
                    "human_queue_cached_row_task_mismatch")
            require(row["human_sampled_visual_qualified"] == (row["status"] == "sampled_visual_pass")
                    and row["source_coverage"] == "missing" and row["pairing_scope"] == "synthetic_only"
                    and row["formal_real_sim_human_pair"] is False, "human_queue_cached_quality_or_scope_changed")
            for key in ["anchor", "anchor_review", "video", "outcome", "review_unknown"]:
                if row[key] is not None:
                    checked(row[key])
            if row["outcome"] is not None:
                generated = read(checked(row["outcome"]))
                require(generated["key"] == record["key"] and generated["method"] == "motion_reference"
                        and generated["contract"] == record["contract"]
                        and generated["status"] == row["status"] and generated.get("video") == row["video"]
                        and generated["human_image"] == row["anchor"]
                        and generated["anchor_review"] == row["anchor_review"],
                        "human_queue_cached_video_binding_mismatch")
    require(value["rows"] == [r for outcome in value["groups"] for r in outcome["rows"]],
            "human_queue_cached_rows_differ_from_groups")


def finalize_phase(root, index, index_path, phase, value):
    """Recover only immutable derived records and read-only credit accounting.

    This function has no generation operation or metered-client construction.
    A crash after results, decision, snapshot, usage or completion marker resumes
    here and never re-enters `run_group`.
    """
    from .billing import report, snapshot
    root = Path(root); directory = Path(index_path).parent
    result_path = directory / (phase + "_results.json")
    require(read(result_path) == value, "human_queue_phase_results_changed")
    validate_phase_result(root, index, index_path, phase, value)
    decision_binding = None
    if phase == "qualification":
        decision = qualification_decision(value["groups"][0])
        path = directory / "qualification_decision.json"
        write(path, {**decision, "qualification_outcome": bind(result_path)})
        decision_binding = bind(path)
    credit_path = directory / (phase + "_credit_after.json")
    if not credit_path.exists():
        snapshot(root, credit_path)
    usage_path = directory / (phase + "_usage.json")
    if not usage_path.exists():
        report(root, usage_path)
    usage = read(usage_path)
    require(usage["budget"] == index["budget"]
            and usage["new_started"] == value["usage"]["new_started"]
            and usage["lineage_started"] == value["usage"]["lineage_started"],
            "human_queue_phase_usage_binding_mismatch")
    write(directory / (phase + "_completed.json"), {"schema": VERSION, "phase": phase,
        "phase_results": bind(result_path), "qualification_decision": decision_binding,
        "credit_after": bind(credit_path), "usage": bind(usage_path),
        "generation_calls_started_during_finalization": 0})
    replace_state(directory / "progress.json", {"status": "phase_complete", "phase": phase,
        "release": index["release"], "pid": None, "updated_unix": time.time()})
    return value


def run(root, phase):
    """Paid entry point. Separate single-process lock and existing quota accounting."""
    from .billing import snapshot
    root = Path(root).resolve(); release = release_id()
    require(phase in {"qualification", "expansion"}, "invalid_human_queue_phase")
    index_path = root / "human_queue" / release / "index.json"
    index = read(index_path); validate_index(root, index)
    require(index["release"] == release, "resume_human_queue_using_original_frozen_release")
    require(shutil.disk_usage(root).free > 20 * 2**30, "insufficient_disk_for_human_queue")
    directory = index_path.parent
    with (root / ".paid_batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            cached = directory / (phase + "_results.json")
            if cached.exists():
                return finalize_phase(root, index, index_path, phase, read(cached))
            if phase == "expansion":
                final_qualification = read(directory / "qualification_results.json")
                finalize_phase(root, index, index_path, "qualification", final_qualification)
                decision = read(directory / "qualification_decision.json")
                require(decision["qualification_outcome"] == bind(directory / "qualification_results.json")
                        and decision["proceed_to_frozen_expansion"] is True,
                        "full_human_review_qualification_required_before_expansion")
            budget = ComparisonBudget(root)
            snapshot(root, root / "billing/campaign_before.json")
            replace_state(directory / "progress.json", {"status": "running", "phase": phase, "pid": os.getpid(),
                "release": release, "updated_unix": time.time(), "index": bind(index_path)})
            groups = index["groups"][:1] if phase == "qualification" else index["groups"][1:]
            outcomes = []
            for group in groups:
                outcome = run_group(root, budget, index, group); outcomes.append(outcome)
                if outcome["status"] == "waiting_existing_video":
                    replace_state(directory / "progress.json", {"status": "waiting_existing_video", "phase": phase,
                        "release": release, "pid": None, "updated_unix": time.time(), "resubmit": False})
                    return outcome
            value = {"schema": VERSION, "phase": phase, "status": "phase_complete", "index": bind(index_path),
                     "groups": outcomes, "rows": [r for o in outcomes for r in o["rows"]],
                     "source_coverage": "missing", "pairing_scope": "synthetic_only", "formal_pairs": 0,
                     "usage": budget.usage(), "automatic_semantic_retry_count": 0, "training_admitted": False}
            write(directory / (phase + "_results.json"), value)
            return finalize_phase(root, index, index_path, phase, value)
        except Exception as exc:
            replace_state(directory / "progress.json", {"status": "stopped_on_exception", "phase": phase,
                "release": release, "pid": None, "exception_type": type(exc).__name__,
                "code": getattr(exc, "code", "operation_failed"), "updated_unix": time.time(), "automatic_retry": False})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["select", "prepare", "migrate", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--max-scenes", type=int, default=4)
    parser.add_argument("--scene-kind", action="append", default=[])
    parser.add_argument("--phase", choices=["qualification", "expansion"], default="qualification")
    parser.add_argument("--from-release", help="Explicit pre-video recovery; keep old conditions, image and review cache")
    args = parser.parse_args()
    if args.command == "run":
        activate()
        result = run(args.root, args.phase)
    elif args.command == "migrate":
        require(args.from_release is not None, "human_migration_requires_from_release")
        result = migrate(args.root, args.from_release)
    else:
        require(args.audit is not None, "local_selection_requires_independent_audit")
        action = select_candidates if args.command == "select" else prepare
        result = action(args.root, args.audit, args.max_scenes, args.scene_kind)
    print(json.dumps(result, ensure_ascii=False, indent=2))
