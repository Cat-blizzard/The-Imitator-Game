"""One explicit schema repair, same image/index/budget, no automatic retry.

prepare freezes an additive release under H/repairs. run makes at most one new
anchor review; only a strictly passing review admits the two original videos.
All calls, including the old failure, use H's original cumulative ledger.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from . import review_wire_v2
from .human_pilot_v1 import _modules, validate_index, check_bindings
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind, checked, digest, read, require, write, write_bytes

VERSION = "schema_wire_v2"
OLD_IMPORT = "from .review_transport import generation_config, wire_schema"
NEW_IMPORT = "from .review_wire_v2 import generation_config, wire_schema"


def repair_root(root):
    return Path(root).resolve() / "repairs" / VERSION


def _link(link, source):
    link, source = Path(link), Path(source).resolve(strict=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(source, target_is_directory=source.is_dir())
    require(link.is_symlink() and link.resolve(strict=True) == source, "repair_alias_changed:" + str(link))


def _load(repair):
    release = read(repair / "release/release.json")
    for item in release["files"]:
        checked(item)
    package = "_human_recovery_" + digest(release)[:20]
    directory = repair / "release/autonomous_pipeline_v4"
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, directory / "__init__.py", submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return lambda name: importlib.import_module(package + "." + name)


def prepare(root):
    root = Path(root).resolve(); repair = repair_root(root)
    original = _modules(root); index = validate_index(root, original)
    require(shutil.disk_usage(root).free > 10 * 2**30, "need_10GiB_free")
    if (repair / "protocol.json").exists():
        validate(root)
        return {"status": "already_prepared", "protocol": bind(repair / "protocol.json"), "api_calls": 0}
    selected = index["groups"][0]
    records = [next(r for r in index["records"] if r["key"] == k) for k in selected["keys"]]
    require(len(records) == 2, "two_original_qualification_tasks_required")
    anchor = Path(records[0]["anchor_directory"])
    image_receipt = read(anchor / "human.provenance.json")
    check_bindings(image_receipt)
    old_request = anchor / "review_request.json"
    budget = original("budget").Budget(root / "budget.json")
    failed = budget.directory("review", old_request)
    failure = read(failed / "failure.json")
    require(failure["http_status"] == 400 and failure["provider_outcome"] == "definite_failure"
            and not (failed / "result.json").exists(), "explicit_prior_400_required")
    require(budget.usage()["new_started"] == {"image": 1, "review": 1, "video": 0},
            "repair_requires_original_one_image_one_failed_review_no_other_calls")
    original_release = read(root / "human_pilot_release.json")
    source = checked(original_release["retained_release"]).parent / "autonomous_pipeline_v4"
    copied = repair / "release/autonomous_pipeline_v4"
    comparisons = []
    for path in sorted(source.glob("*.py")):
        raw = path.read_bytes()
        if path.name == "models.py":
            require(raw.count(OLD_IMPORT.encode()) == 1, "unexpected_retained_model_import")
            raw = raw.replace(OLD_IMPORT.encode(), NEW_IMPORT.encode())
        write_bytes(copied / path.name, raw)
        comparisons.append({"file": path.name, "original": bind(path), "candidate": bind(copied / path.name),
                            "same_bytes": path.read_bytes() == raw})
    require([c["file"] for c in comparisons if not c["same_bytes"]] == ["models.py"], "unexpected_protocol_diff")
    write_bytes(copied / "review_wire_v2.py", Path(review_wire_v2.__file__).read_bytes())
    write_bytes(repair / "release/human_pilot_recovery_v2.py", Path(__file__).read_bytes())
    write(repair / "release/release.json", {"release": VERSION, "source_release": original_release["retained_release"],
          "files": [bind(p) for p in sorted(copied.glob("*.py"))] + [bind(repair / "release/human_pilot_recovery_v2.py")]})
    candidate = _load(repair)
    old = read(old_request)
    wire, encoding = candidate("review_wire_v2").wire_schema(old["response_schema"], read(root / "config.json")["gemini"].get("wire_schema_adapter"))
    write(repair / "anchor_local_response_schema.json", old["response_schema"])
    write(repair / "anchor_local_acceptance_schema.json", old["acceptance_schema"])
    write(repair / "anchor_wire_schema.json", wire)
    work = repair / "work"
    for name in ("budget.json", "config.json", "experiment.json", "human_runtime.json"):
        _link(work / name, root / name)
    for name in ("provider_metadata.json", "campaign_before.json"):
        _link(work / "billing" / name, root / "billing" / name)
    _link(work / "motion_inputs", root / "motion_inputs")
    aliased = candidate("budget").Budget(work / "budget.json")
    require(aliased.ledger == budget.ledger and aliased.usage() == budget.usage()
            and bind(work / "budget.json") == bind(root / "budget.json"), "cumulative_budget_not_shared")
    revised_request = deepcopy(old)
    revised_request["implementation"] = candidate("core").implementation()
    write(work / "anchor_review/review_request.json", revised_request)
    require(aliased.directory("review", work / "anchor_review/review_request.json") != failed,
            "repaired_stage_must_be_distinct")
    require(revised_request["implementation"]["models.py"]["sha256"]
            != old["implementation"]["models.py"]["sha256"], "new_protocol_scope_required")
    # New work path only holds candidate artifacts. It does not register a new
    # campaign and is incapable of resetting the absolute parent ledger.
    write(repair / "protocol.json", {"schema": "human_pilot_explicit_schema_repair_v2", "parent_root": str(root),
        "parent_index": bind(root / "human_pilot_index.json"), "budget": bind(root / "budget.json"),
        "shared_ledger": str(budget.ledger), "usage_at_freeze": budget.usage(),
        "old_failure": bind(failed / "failure.json"), "old_request": bind(old_request),
        "new_anchor_request": bind(work / "anchor_review/review_request.json"),
        "candidate_release": bind(repair / "release/release.json"),
        "driver": bind(repair / "release/human_pilot_recovery_v2.py"),
        "wire_encoder": bind(copied / "review_wire_v2.py"), "file_comparison": comparisons,
        "schema_encoding": encoding, "local_response_schema": bind(repair / "anchor_local_response_schema.json"),
        "local_acceptance_schema": bind(repair / "anchor_local_acceptance_schema.json"),
        "wire_schema": bind(repair / "anchor_wire_schema.json"),
        "retained_image_receipt": bind(anchor / "human.provenance.json"), "image": image_receipt["image"],
        "selected_keys": selected["keys"], "max_new_anchor_reviews": 1, "max_new_videos": 2,
        "total_review_ceiling_including_video_reviews": 3,
        "new_image_calls": 0, "automatic_retry": False, "prompt_changes": False,
        "local_json_schema_and_visual_acceptance_changed": False,
        "stage_identity": "new models protocol digest and immutable request path; original failure remains counted",
        "authorization": "Parent explicitly authorized one wire repair review then the two original videos if strict gate passes."})
    validate(root)
    return {"status": "repair_prepared", "protocol": bind(repair / "protocol.json"), "api_calls": 0,
            "same_cumulative_usage": budget.usage()}


def validate(root):
    root = Path(root).resolve(); repair = repair_root(root); p = read(repair / "protocol.json")
    check_bindings(p)
    require(p["driver"]["sha256"] == bind(__file__)["sha256"]
            and p["wire_encoder"]["sha256"] == bind(review_wire_v2.__file__)["sha256"], "repair_code_changed")
    require(p["parent_index"] == bind(root / "human_pilot_index.json")
            and p["budget"] == bind(root / "budget.json"), "parent_inputs_or_budget_changed")
    original = _modules(root); index = validate_index(root, original); candidate = _load(repair)
    retained = read(p["old_request"]["path"])
    require(retained["response_schema"] == read(p["local_response_schema"]["path"])
            and retained["acceptance_schema"] == read(p["local_acceptance_schema"]["path"]), "local_rules_changed")
    new = read(checked(p["new_anchor_request"]))
    require({k:v for k,v in retained.items() if k != "implementation"}
            == {k:v for k,v in new.items() if k != "implementation"}, "repair_changed_anchor_conditions")
    work = repair / "work"; b = candidate("budget").Budget(work / "budget.json")
    require(str(b.ledger) == p["shared_ledger"] and bind(work / "budget.json") == p["budget"], "budget_alias_changed")
    return candidate, index, p


def run(root):
    root = Path(root).resolve(); repair = repair_root(root); module, index, protocol = validate(root)
    work = repair / "work"; module("core").activate()
    with (root / ".paid_batch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = repair / "qualification_results.json"
        if output.exists():
            result = read(output); check_bindings(result); return result
        budget = module("motion_reference").ComparisonBudget(work)
        records = [next(r for r in index["records"] if r["key"] == k) for k in protocol["selected_keys"]]
        receipt = read(checked(protocol["retained_image_receipt"]))
        directory = work / "anchor_review"
        contract = read(checked(records[0]["contract"]))
        reviewed, unavailable = module("motion_comparison").review_or_unknown(budget, directory / "review_request.json",
            lambda: module("review").anchor(work, budget, directory, receipt,
                contract["initial_scene_information"], contract["initial_image"]))
        new_request = read(directory / "review_request.json"); old = read(checked(protocol["old_request"]))
        require({k:v for k,v in new_request.items() if k != "implementation"}
                == {k:v for k,v in old.items() if k != "implementation"}, "repair_changed_anchor_prompt_or_rule")
        write(repair / "anchor_request_equivalence.json", {"old": protocol["old_request"],
            "new": bind(directory / "review_request.json"), "identical_except_implementation": True,
            "same_images_prompt_original_response_and_acceptance_schema": True})
        rows = []
        if unavailable or not reviewed["gate"]["passed_in_sampled_visual_scope"]:
            if unavailable: write(directory / "unusable_review.json", unavailable)
            status = "initial_image_review_unknown_video_unattempted" if unavailable else "initial_image_rejected_video_unattempted"
            rows = [{"key": r["key"], "status": status, "contract": r["contract"], "video": None,
                     "anchor": receipt["image"]} for r in records]
        else:
            print(json.dumps({"stage": "repaired_anchor_review", "passed_original_strict_gate": True,
                              "review": str(directory / "review.json"), "usage": budget.usage()}), flush=True)
            for row in records:
                outcome = module("motion_comparison").run_video(work, budget, row, "motion_reference", receipt["image"],
                    bind(directory / "review.json"), poll_deadline_seconds=900,
                    wire_schema_adapter=module("review_transport").WIRE_SCHEMA_ADAPTER)
                if outcome["status"] == "waiting_existing_video": return outcome
                rows.append({"key": row["key"], "status": outcome["status"], "contract": row["contract"],
                    "video": outcome.get("video"), "anchor": receipt["image"],
                    "outcome": bind(work / "runs" / row["key"] / "motion_reference" / VERSION / "outcome.json")})
                print(json.dumps({"key": row["key"], "status": outcome["status"], "usage": budget.usage()}), flush=True)
        module("billing").snapshot(work, work / "billing/campaign_after.json")
        usage = repair / "qualification_usage.json"; module("billing").report(work, usage)
        result = {"schema": "human_pilot_repaired_qualification_v2", "phase": "qualification", "status": "terminal",
            "rows": rows, "index": protocol["parent_index"], "repair_protocol": bind(repair / "protocol.json"),
            "usage": bind(usage), "formal_pairs": 0, "training_admitted": False,
            "independent_quality": "unknown_until_separate_candidate_pixel_inspection", "automatic_retry": False}
        write(output, result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("command", choices=("prepare", "validate", "run"));p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare": result = prepare(a.root)
    elif a.command == "validate":
        _, _, protocol = validate(a.root); result = {"status": "local_validation_passed", "protocol": bind(repair_root(a.root) / "protocol.json"), "api_calls": 0}
    else: result = run(a.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
