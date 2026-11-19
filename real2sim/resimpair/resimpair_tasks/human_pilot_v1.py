"""Finite generic pick/place -> single-person motion-reference pilot.

``prepare-local`` uses actual accepted executions and performs full CPU media
validation without registering a budget or contacting a provider. ``register``
reserves a cumulative campaign locally; only ``run`` can make provider calls.
The retained generator, visual rules, accounting and recovery clients are copied
into an immutable release and executed unchanged. No grasp or robot run occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

from resimpair_runtime import ROOT, configure
configure()
from resimpair_grasp.common import resolve
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import (
    bind, checked, digest, identifier, read, require, write,
)

VERSION = "qualified_pickplace_human_pilot_v1"
RELEASE = "human_pilot_r1"
CALL_CEILINGS = {"image": 3, "video": 6, "review": 18}
SPOT_CHECKS = ("one_operator", "hand_correspondence", "object_identity_and_relative_size",
               "declared_task_goal", "critical_events")


def same_artifact(left, right):
    return resolve(left).resolve() == resolve(right).resolve() and left["sha256"] == right["sha256"]


def semantic_task_signature(info):
    """Names, suggested pacing and incidental independent-event order add no task."""
    events = {event["id"]: {key: value for key, value in event.items()
              if key not in {"id", "suggested_seconds", "effector"}} for event in info["required_events"]}
    def relation(row):
        return {key: events.get(value, value) if key in {"from", "until"} else value
                for key, value in row.items() if key not in {"id", "effector"}}
    ordered = lambda values: sorted(values, key=digest)
    return digest({"events": ordered(events.values()),
        "precedence": ordered([[events[left], events[right]] for left, right in info["required_event_precedence"]]),
        "goals": ordered([relation(row) for row in info["final_relations"]]),
        "constraints": ordered([relation(row) for row in info.get("persistent_constraints", [])]),
        "terminal_hands": info["terminal_hands"]})


def check_bindings(value):
    """Verify both portable relative bindings and retained absolute bindings."""
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            resolve(value)
        else:
            for child in value.values():
                check_bindings(child)
    elif isinstance(value, list):
        for child in value:
            check_bindings(child)


def qualify_contract(path):
    """Read independent reports; never manufacture or overwrite their verdicts."""
    path = Path(path).resolve()
    contract = read(path)
    provenance_path = path.parent / "general_pickplace_provenance.json"
    provenance = read(provenance_path)
    check_bindings(provenance)
    require(same_artifact(provenance["contract"], bind(path))
            and same_artifact(provenance["bridge"], contract["execution"]), "export_provenance_mismatch")
    bridge = read(resolve(contract["execution"]))
    original_ref = bridge["original_execution_bundle"]
    original = read(resolve(original_ref))
    require(original["planning_failure"] is None and original["native_execution"] is True,
            "successful_native_execution_required")
    for key in ("measurements", "event_protocol", "task_intent", "scene", "cameras"):
        require(original[key] == bridge[key], "metadata_bridge_changed_native_content:" + key)
    container = read(resolve(provenance["container_acceptance"]))
    check_bindings(container)
    native = read(resolve(container["retained_audit"]))
    check_bindings(native)
    require(container["passed"] is True and native["passed"] is True, "independent_native_acceptance_required")
    require(same_artifact(native["bundle"], original_ref), "native_audit_execution_mismatch")
    requested_ref = provenance.get("requested_goal_acceptance")
    if "source_request" in original["case"]:
        require(requested_ref is not None, "requested_goal_acceptance_missing")
        requested = read(resolve(requested_ref))
        check_bindings(requested)
        require(requested["passed"] is True
                and same_artifact(requested["native_audit"], provenance["container_acceptance"])
                and same_artifact(requested["source_request"], original["case"]["source_request"]),
                "original_requested_goal_not_qualified")
    require(contract["source_mapping"].get("formal_pair_qualified") is False
            and contract["source_mapping"].get("source_status") == "missing",
            "pilot_only_supports_explicit_synthetic_scope")
    return {"source_contract": bind(path), "export_provenance": bind(provenance_path),
            "original_execution": bind(resolve(original_ref)),
            "native_audit": bind(resolve(container["retained_audit"])),
            "container_audit": bind(resolve(provenance["container_acceptance"])),
            "requested_goal_audit": bind(resolve(requested_ref)) if requested_ref else None}


def _freeze(root):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.experiment import freeze
    freeze(root, RELEASE)
    dest = root / "releases" / RELEASE / "human_pilot_v1.py"
    shutil.copyfile(__file__, dest)
    write(root / "human_pilot_release.json", {"schema": VERSION, "release": RELEASE,
          "retained_release": bind(dest.parent / "release.json"), "bridge": bind(dest)})


def _modules(root):
    """Import the frozen package, rather than a development module claiming its hash."""
    receipt = read(Path(root) / "human_pilot_release.json")
    require(receipt["schema"] == VERSION and receipt["bridge"]["sha256"] == bind(__file__)["sha256"],
            "bridge_changed_use_original_frozen_script")
    check_bindings(receipt)
    manifest = read(checked(receipt["retained_release"]))
    check_bindings(manifest)
    package = "_human_pilot_" + digest(receipt)[:20]
    directory = checked(receipt["retained_release"]).parent / "autonomous_pipeline_v4"
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, directory / "__init__.py",
                    submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return lambda name: importlib.import_module(package + "." + name)


def _experiment(local_only):
    return {"schema": VERSION, "local_only": local_only, "sources": [], "cases": [],
            "review_frame_budget": 49, "per_video_poll_deadline_seconds": 900,
            "no_training": True, "automatic_retry": False,
            "native_physics_and_visual_acceptance_unchanged": True,
            "source_coverage": "missing", "pairing_scope": "synthetic_only",
            "formal_pair_count": 0, "training_admitted": False}


def register(root, limits, usd_limit):
    """Local cumulative registration only; never read credentials or query billing."""
    from resimpair import historical_lineage
    from corl2026_workshop.experiments.autonomous_pipeline_v4.budget import Budget
    root = Path(root).resolve()
    identifier(root.name)
    require(not root.exists(), "campaign_exists_resume_do_not_reset")
    require(set(limits) == set(CALL_CEILINGS) and all(type(limits[k]) is int
            and 0 <= limits[k] <= CALL_CEILINGS[k] for k in limits), "finite_pilot_call_limits_required")
    require(type(usd_limit) in (int, float) and 0 < usd_limit <= 60, "finite_nominal_credit_limit_required")
    require(shutil.disk_usage(root.parent).free > 10 * 2**30, "need_10GiB_free")
    (ROOT / "runtime").mkdir(exist_ok=True)
    with (ROOT / "runtime/.new_campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ancestors, evidence = historical_lineage()
        registration = {"schema": "resimpair_portable_campaign_v1", "name": root.name,
            "purpose": VERSION, "new_call_limits": limits, "nominal_usd_admission_limit": usd_limit,
            "observed_quota_stop_limit": int(usd_limit * 500000),
            "api_concurrency": 1, "automatic_retry": False, "gpu_used": False}
        write(root / "portable_registration.json", registration)
        write(root / "budget.json", {"schema": "resimpair_cumulative_budget_v4", "campaign_id": root.name,
            "ledger_path": str(root / "campaign_ledger"), "portable_campaign": True,
            "portable_registration": bind(root / "portable_registration.json"), "new_call_limits": limits,
            "ancestor_started_calls": ancestors, "ancestor_evidence": evidence,
            "lineage_cumulative_limits": {k: ancestors[k] + limits[k] for k in limits},
            "max_concurrent": 1, "timeout_seconds": 600, "automatic_retry": False,
            "currency_budget": None, "nominal_usd_admission_limit": usd_limit,
            "observed_quota_stop_limit": registration["observed_quota_stop_limit"],
            "reserve_quota_before_new_call": {"image": 500000, "video": 8000000, "review": 500000},
            "authorization": "Standing user task-data-generation authorization; finite pilot; preserve all prior calls."})
        write(root / "experiment.json", _experiment(False))
        write(root / "config.json", read(ROOT / "configs/providers.json"))
        write(root / "human_runtime.json", {"schema": "resimpair_human_runtime_v1",
            "budget": bind(root / "budget.json"), "media_upload_limit": limits["video"],
            "provider_application": bind(ROOT / "provenance/provider_app.js"),
            "per_video_poll_deadline_seconds": 900, "automatic_retry": False})
        Budget(root / "budget.json")
        _freeze(root)
    return {"root": str(root), "new_call_limits": limits, "ancestor_started_calls": ancestors,
            "nominal_usd_admission_limit": usd_limit, "api_calls": 0, "gpu_used": False}


def _prepare_one(root, module, path):
    import numpy as np
    from resimpair_tabletop.sample_report import native_scalars
    qualified = qualify_contract(path)
    source = read(path)
    # Re-run the retained compiler on actual measured state. Do not infer any
    # new goal, synthesize event evidence, or mark human checks as passed.
    compiled = native_scalars(module("generic_compiler").compile_bundle(resolve(source["execution"]), source["source_mapping"]))
    for key in ("generation_information", "initial_scene_information", "measured", "hand_correspondence_evidence"):
        require(compiled[key] == source[key], "frozen_compiler_changed_existing_conditions:" + key)
    key = identifier(source["task_id"])
    destination = root / "prepared" / key
    contract = deepcopy(compiled)
    contract["generation_scope"] = {"schema": "unpaired_sim_to_human_quality_probe_v1",
        "source_coverage": "missing", "pairing_scope": "synthetic_only", "formal_pair_qualified": False,
        "task_conditions": "unchanged_predeclared_native_contract_and_measured_events"}
    contract["human_evidence_plan"] = module("evidence_plan").compile_plan(contract)
    require(contract["human_evidence_plan"]["declaration_coverage_complete"], "unsupported_declared_evidence")
    contract["qualified_native_handoff"] = qualified
    contract["human_pilot_bridge"] = bind(root / "releases" / RELEASE / "human_pilot_v1.py")
    write(destination / "contract.json", contract)
    write(destination / "evidence_plan.json", contract["human_evidence_plan"])
    bundle = read(resolve(contract["execution"]))
    with np.load(resolve(bundle["measurements"]), allow_pickle=False) as archive:
        times = archive["timestamps_seconds"]
    cameras = {}
    for name in ("robot_side", "opposite_high"):
        camera = bundle["cameras"][name]
        frames = module("media").packet(resolve(camera["video"]), destination / (name + "_frames"), frame_budget=9)
        require(frames["decoded_frames"] == camera["frame_count"] == len(times)
                and np.allclose(frames["all_native_pts_seconds"], times, atol=1e-5, rtol=0),
                "native_timeline_mismatch")
        cameras[name] = bind(destination / (name + "_frames") / "packet.json")
    write(destination / "media_validation.json", {"contract": bind(destination / "contract.json"),
        "cameras": cameras, "both_fully_decoded": True})
    module("compiler").completed_contract(destination / "contract.json")
    reference = module("motion_reference").derivative(contract, root / "motion_inputs" / key / "reference.mp4")
    prompt = module("prompts").anchor_prompt(contract["initial_scene_information"])
    anchor_id = digest({"native_initial_sha256": contract["initial_image"]["sha256"],
                        "prompt": prompt, "model_config": bind(root / "config.json")})[:24]
    anchor = root / "anchors" / anchor_id
    module("paid_pilot").shared_anchor_request(anchor / "image_request.json", {"schema": "resimpair_initial_request_v4", "prompt": prompt,
        "model_config": bind(root / "config.json"), "images": [contract["initial_image"]],
        "initial_scene": contract["initial_scene_information"], "implementation": module("core").implementation(),
        "task_specific_prompt_override": None})
    info = contract["generation_information"]
    projected = [{"entity": e["id"], "xy": e["initial_image_xy_normalized"]} for e in info["inventory"]
        if "initial_image_xy_normalized" in e and not all(0 <= n <= 1 for n in e["initial_image_xy_normalized"])]
    original = read(resolve(qualified["original_execution"]))
    case = original["case"]
    return {"key": key, "case": key, "contract": bind(destination / "contract.json"),
        "source_contract": qualified["source_contract"], "native_qualification": qualified,
        "anchor_id": anchor_id, "anchor_directory": str(anchor), "image_request": bind(anchor / "image_request.json"),
        "base_prompt": module("prompts").video_prompt(contract),
        "motion_reference": bind(root / "motion_inputs" / key / "reference.json"),
        "reference_video": reference["video"], "split_group": case.get("split_group"),
        "initial_physics_sha256": original["initial_physics_sha256"],
        "initial_object_state_sha256": original["initial_object_state_sha256"],
        "task_condition_signature": semantic_task_signature(info),
        "partition": case.get("partition", "unknown"), "outside_initial_frame_centers": projected,
        "projected_center_is_not_visibility_proof": True, "independent_pixel_detector_status": "unknown",
        "pixel_detector_limitation": "Existing color/rail diagnostics are not calibrated for these textured meshes.",
        "human_visual_quality": "unattempted", "source_coverage": "missing", "formal_pair_qualified": False}


def prepare(root, contracts, *, local_only=False):
    root = Path(root).resolve()
    contracts = [Path(p).resolve() for p in contracts]
    require(contracts and len(set(contracts)) == len(contracts), "duplicate_or_empty_contracts")
    require(shutil.disk_usage(root.parent).free > 10 * 2**30, "need_10GiB_free")
    if local_only and not root.exists():
        write(root / "experiment.json", _experiment(True))
        write(root / "config.json", read(ROOT / "configs/providers.json"))
        _freeze(root)
    module = _modules(root)
    require(read(root / "experiment.json")["local_only"] is local_only, "local_and_paid_campaign_must_be_separate")
    spec = None if local_only else module("batch_human_queue").validate_budget(root)
    records = [_prepare_one(root, module, path) for path in contracts]
    require(len({r["key"] for r in records}) == len(records), "duplicate_task_id")
    groups = []
    for row in records:
        found = next((g for g in groups if g["anchor_id"] == row["anchor_id"]), None)
        if found is None:
            found = {"anchor_id": row["anchor_id"], "keys": [], "initial_state_sha256":
                     read(resolve(row["contract"]))["initial_image"]["sha256"],
                     "initial_physics_sha256": row["initial_physics_sha256"],
                     "initial_object_state_sha256": row["initial_object_state_sha256"]}
            groups.append(found)
        require(all(found[k] == row[k] for k in ("initial_physics_sha256", "initial_object_state_sha256")),
                "same_anchor_requires_same_initial_native_state")
        found["keys"].append(row["key"])
    planned = {"image": len(groups), "video": len(records), "review": len(groups) + len(records)}
    if not local_only:
        require(all(len(g["keys"]) == 2 for g in groups) and 1 <= len(groups) <= 3,
                "paid_pilot_requires_one_to_three_exact_shared_initial_scenes_each_with_two_tasks")
        require(all(len({r["task_condition_signature"] for r in records if r["key"] in g["keys"]}) == 2
                    for g in groups), "shared_scene_requires_distinct_declared_task_conditions")
        require(all(planned[k] <= spec["new_call_limits"][k] for k in planned), "prepared_queue_exceeds_budget")
    index = {"schema": VERSION, "release": RELEASE, "release_receipt": bind(root / "human_pilot_release.json"),
        "local_only": local_only, "records": records, "groups": groups, "planned_calls": planned,
        "budget": None if local_only else bind(root / "budget.json"), "model_config": bind(root / "config.json"),
        "inputs_order_frozen_before_generation": True, "no_failed_case_replacement": True,
        "source_coverage": "missing", "formal_pairs": 0, "training_admitted": False,
        "qualification_requires": "first shared scene, both complete video reviews, independent candidate-pixel spot check"}
    write(root / "human_pilot_index.json", index)
    return {"status": "local_inputs_prepared", "index": bind(root / "human_pilot_index.json"),
            "tasks": len(records), "exact_shared_initial_scenes": len(groups), "planned_calls": planned,
            "api_calls": 0, "gpu_used": False, "human_videos_generated": 0}


def validate_index(root, module):
    index = read(root / "human_pilot_index.json")
    require(index["schema"] == VERSION and index["local_only"] is False, "paid_run_requires_registered_preparation")
    require(index["release_receipt"] == bind(root / "human_pilot_release.json")
            and index["model_config"] == bind(root / "config.json")
            and index["budget"] == bind(root / "budget.json"), "frozen_pilot_bindings_changed")
    module("batch_human_queue").validate_budget(root)
    for row in index["records"]:
        require(qualify_contract(resolve(row["source_contract"])) == row["native_qualification"],
                "native_qualification_changed")
        check_bindings(row["native_qualification"])
        for name in ("contract", "image_request", "motion_reference", "reference_video"):
            checked(row[name])
        contract = module("compiler").completed_contract(checked(row["contract"]))
        require(row["base_prompt"] == module("prompts").video_prompt(contract), "compiled_video_prompt_changed")
        require(contract["human_evidence_plan"] == module("evidence_plan").compile_plan(contract),
                "declared_evidence_plan_changed")
    return index


def expansion_qualified(result, spotcheck):
    """Independent observations cannot be replaced by the generator's VLM score."""
    require(result.get("phase") == "qualification" and result.get("status") == "terminal"
            and len(result.get("rows", [])) == 2
            and all(r.get("status") == "sampled_visual_pass" for r in result["rows"]),
            "qualification_requires_both_complete_passing_video_reviews")
    require(spotcheck.get("schema") == "resimpair_human_independent_spotcheck_v1"
            and spotcheck.get("status") == "pass" and spotcheck.get("major_conflicts_found") is False
            and isinstance(spotcheck.get("observer"), str) and spotcheck["observer"]
            and spotcheck.get("independent_of_generation") is True, "independent_spotcheck_required")
    rows = spotcheck.get("records", [])
    require(len(rows) == 2 and {r["key"] for r in rows} == {r["key"] for r in result["rows"]},
            "independent_spotcheck_denominator_changed")
    for expected in result["rows"]:
        row = next(r for r in rows if r["key"] == expected["key"])
        require(row.get("video") == expected["video"], "independent_spotcheck_video_mismatch")
        packet = read(checked(row["frame_packet"]))
        require(packet["video"] == expected["video"] and packet["decode_complete"] is True,
                "independent_spotcheck_frame_packet_mismatch")
        supplied = [frame["image"] for frame in packet["frames"]]
        for name in SPOT_CHECKS:
            finding = row.get("checks", {}).get(name, {})
            require(finding.get("status") == "pass" and finding.get("candidate_frames")
                    and isinstance(finding.get("observation"), str) and finding["observation"],
                    "independent_spotcheck_missing_evidence_or_unknown:" + name)
            for frame in finding["candidate_frames"]:
                checked(frame)
                require(frame in supplied, "independent_spotcheck_frame_not_from_candidate")
    return True


def spotcheck_template(root, output):
    """Populate references, never verdicts; an independent observer supplies evidence."""
    root = Path(root).resolve()
    result_path = root / "qualification_results.json"
    result = read(result_path)
    require(result["phase"] == "qualification" and result["status"] == "terminal", "qualification_not_terminal")
    rows = []
    for row in result["rows"]:
        require(row.get("video") and row.get("outcome"), "both_generated_videos_required_for_spotcheck")
        outcome = read(checked(row["outcome"]))
        require(outcome["video"] == row["video"] and outcome.get("video_review"), "completed_review_required_for_spotcheck")
        reviewed = read(checked(outcome["video_review"]))
        packet = read(checked(reviewed["packet"]))
        require(packet["video"] == row["video"], "review_candidate_binding_changed")
        rows.append({"key": row["key"], "video": row["video"], "anchor": row["anchor"],
            "frame_packet": reviewed["packet"], "checks": {name: {"status": "unknown",
                "candidate_frames": [], "observation": ""} for name in SPOT_CHECKS}})
    write(output, {"schema": "resimpair_human_independent_spotcheck_v1", "qualification_result": bind(result_path),
        "observer": "", "independent_of_generation": False, "status": "unknown", "major_conflicts_found": None,
        "records": rows, "scope": "candidate RGB evidence only; no inferred human 3D or all-frame guarantee"})
    return {"status": "unknown_template_prepared", "template": bind(output), "api_calls": 0}


def run(root, phase, independent_spotcheck=None):
    root = Path(root).resolve(); module = _modules(root)
    index = validate_index(root, module)
    module("core").activate()
    require(phase in {"qualification", "expansion"}, "invalid_phase")
    require(shutil.disk_usage(root).free > 10 * 2**30, "need_10GiB_free")
    result_path = root / (phase + "_results.json")
    with (root / ".paid_batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result_path.exists():
            previous = read(result_path)
            require(previous["index"] == bind(root / "human_pilot_index.json"), "terminal_inputs_changed")
            check_bindings(previous)
            return previous
        if phase == "expansion":
            require(independent_spotcheck is not None, "independent_spotcheck_required_before_expansion")
            spot = read(independent_spotcheck)
            require(spot.get("qualification_result") == bind(root / "qualification_results.json"),
                    "independent_spotcheck_result_binding_mismatch")
            expansion_qualified(read(root / "qualification_results.json"), spot)
            write(root / "expansion_admission.json", {"independent_spotcheck": bind(independent_spotcheck),
                    "qualification": bind(root / "qualification_results.json"), "passed": True})
        budget = module("motion_reference").ComparisonBudget(root)
        module("billing").snapshot(root, root / "billing/campaign_before.json")
        selected = index["groups"][:1] if phase == "qualification" else index["groups"][1:]
        rows = []
        for group in selected:
            records = [next(r for r in index["records"] if r["key"] == key) for key in group["keys"]]
            anchor = Path(records[0]["anchor_directory"])
            contract = read(checked(records[0]["contract"]))
            receipt = module("models").generate_image(root, budget, anchor / "image_request.json", anchor / "human.png")
            reviewed, unavailable = module("motion_comparison").review_or_unknown(budget, anchor / "review_request.json",
                lambda: module("review").anchor(root, budget, anchor, receipt,
                    contract["initial_scene_information"], contract["initial_image"]))
            if unavailable or not reviewed["gate"]["passed_in_sampled_visual_scope"]:
                if unavailable:
                    write(anchor / "unusable_review.json", unavailable)
                status = "initial_image_review_unknown_video_unattempted" if unavailable else "initial_image_rejected_video_unattempted"
                rows += [{"key": r["key"], "status": status, "contract": r["contract"], "video": None,
                          "anchor": receipt["image"]} for r in records]
                continue
            for row in records:
                outcome = module("motion_comparison").run_video(root, budget, row, "motion_reference", receipt["image"],
                    bind(anchor / "review.json"), poll_deadline_seconds=900,
                    wire_schema_adapter=module("review_transport").WIRE_SCHEMA_ADAPTER)
                if outcome["status"] == "waiting_existing_video":
                    return outcome
                rows.append({"key": row["key"], "status": outcome["status"], "contract": row["contract"],
                    "video": outcome.get("video"), "anchor": receipt["image"],
                    "outcome": bind(root / "runs" / row["key"] / "motion_reference" / RELEASE / "outcome.json")})
                print(json.dumps({"key": row["key"], "status": outcome["status"], "usage": budget.usage()}), flush=True)
        usage_path = root / (phase + "_usage.json")
        module("billing").report(root, usage_path)
        result = {"schema": VERSION, "phase": phase, "status": "terminal", "rows": rows,
            "index": bind(root / "human_pilot_index.json"), "usage": bind(usage_path), "formal_pairs": 0,
            "training_admitted": False, "independent_quality": "unknown_until_separate_candidate_pixel_inspection",
            "automatic_retry": False}
        write(result_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reg = commands.add_parser("register")
    reg.add_argument("--root", type=Path, required=True)
    for kind, ceiling in CALL_CEILINGS.items():
        reg.add_argument("--" + kind + "-limit", type=int, default=ceiling)
    reg.add_argument("--usd-limit", type=float, default=60)
    for name in ("prepare", "prepare-local"):
        p = commands.add_parser(name)
        p.add_argument("--root" if name == "prepare" else "--output", dest="root", type=Path, required=True)
        p.add_argument("--contract", type=Path, action="append", required=True)
    paid = commands.add_parser("run")
    paid.add_argument("--root", type=Path, required=True)
    paid.add_argument("--phase", choices=("qualification", "expansion"), required=True)
    paid.add_argument("--independent-spotcheck", type=Path)
    spot = commands.add_parser("spotcheck-template")
    spot.add_argument("--root", type=Path, required=True)
    spot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        result = register(args.root, {k: getattr(args, k + "_limit") for k in CALL_CEILINGS}, args.usd_limit)
    elif args.command.startswith("prepare"):
        result = prepare(args.root, args.contract, local_only=args.command == "prepare-local")
    elif args.command == "spotcheck-template":
        result = spotcheck_template(args.root, args.output)
    else:
        result = run(args.root, args.phase, args.independent_spotcheck)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
