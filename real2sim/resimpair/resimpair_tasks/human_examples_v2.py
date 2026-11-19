"""Three fixed native-qualified examples with explicit old-unknown containment.

This candidate never retires an unknown call. It carries all historical counts,
freezes its exact request denial and reserve, and serializes against old ledgers.
No source task, acceptance schema, provider model or generation prompt is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import fcntl
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

from resimpair_runtime import ROOT, REGISTRY, configure
configure()
from resimpair_grasp.common import resolve
from resimpair_tasks import human_pilot_v1 as pilot
from resimpair_tasks import review_wire_v2
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import (
    bind, checked, digest, read, require, write, write_bytes, replace_state,
)

VERSION = "human_examples_v2"
LIMITS = {"image": 3, "video": 3, "review": 6}
RECOVERY = ROOT / "runs/simulator_alignment_20261028/api_recovery_readonly"
OLD = ROOT / "runs/batch500_human_pilot_20261027"


@contextmanager
def exclusive(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def historical(containment, exclude_campaign=None):
    """Same historical denominator, one explicitly bound unknown stays unknown."""
    base = ROOT / "provenance/lineage_snapshot.json"
    total = dict(read(base)["lineage_started_calls"]); evidence = [bind(base)]
    carried = {}; portable = ROOT / "provenance/portable_validation_lineage.json"
    if portable.exists():
        saved = read(portable)
        require(saved["base_snapshot_sha256"] == bind(base)["sha256"]
                and saved["unresolved_submissions"] == 0, "portable_lineage_requires_recovery")
        observed = Counter()
        for campaign in saved["campaigns"]:
            for marker in campaign["markers"]:
                key = (campaign["campaign_id"], marker["stage"])
                require(key not in carried and marker["resolved"] is True and marker["kind"] in total,
                        "portable_lineage_ambiguous")
                carried[key] = marker; observed[marker["kind"]] += 1
        require(set(saved["additional_started_calls"]) == set(total)
                and all(observed[k] == saved["additional_started_calls"][k] for k in total),
                "portable_lineage_count_mismatch")
        total = {k: total[k]+observed[k] for k in total}; evidence.append(bind(portable))
    markers = []; ledgers = []; contained = []; attempted_video_tasks = []
    from corl2026_workshop.experiments.autonomous_pipeline_v4.billing import validate_retired_billed_review
    for path in sorted(REGISTRY.glob("*.json")):
        spec = read(path)
        if spec["campaign_id"] == exclude_campaign: continue
        ledger = Path(spec["ledger_path"])
        require(read(ledger / "budget.json") == spec, "registered_budget_changed")
        ledgers.append(str(ledger))
        for marker in sorted(ledger.glob("stages/*/call_started.json")):
            started = read(marker); stage = marker.parent; b = bind(marker)
            prior = carried.get((spec["campaign_id"], stage.name))
            if prior:
                require(b["sha256"] == prior["sha256"] and started["kind"] == prior["kind"], "carried_marker_changed")
            else: total[started["kind"]] += 1
            failure = read(stage / "failure.json") if (stage / "failure.json").exists() else {}
            if started["kind"] == "video":
                request = read(checked(started["request"]))
                if request.get("contract"):
                    attempted_video_tasks.append(read(checked(request["contract"]))["task_id"])
            if (stage / "result.json").exists():
                if started["kind"] == "video":
                    terminal = read(stage / "video_terminal.json") if (stage / "video_terminal.json").exists() else {}
                    require(terminal.get("status") in {"succeeded", "failed", "expired", "cancelled"},
                            "different_video_pending_recover_before_new_campaign")
            elif failure.get("provider_outcome") == "definite_failure" or validate_retired_billed_review(stage):
                pass
            else:
                require(b == containment["pending_call"] and bind(stage / "failure.json") == containment["original_failure"]
                        and started["kind"] == "video" and failure.get("http_status") == 503
                        and failure.get("provider_outcome") == "unknown", "uncontained_unknown_blocks_new_call")
                contained.append(b)
            evidence.append(b); markers.append(b)
    require(contained == [containment["pending_call"]], "exactly_one_predeclared_unknown_required")
    return {"started_calls": total, "evidence": evidence, "markers": markers,
            "ledgers": ledgers, "attempted_video_tasks": sorted(set(attempted_video_tasks))}


def containment_record():
    observation = read(RECOVERY / "task_list_observation.json")
    require(observation["original_outcome"] == "unknown" and observation["resubmit_allowed"] is False
            and observation["new_posts"] == 0 and observation["http_status"] == 200,
            "readonly_recovery_required")
    logs = read(RECOVERY / "provider_logs.json")
    require(logs["success"] is True, "latest_provider_logs_required")
    stage = checked(observation["pending_call"]).parent
    started = read(stage / "call_started.json"); request = read(checked(started["request"]))
    return {"schema": "preserved_unknown_submission_containment_v1",
        "pending_call": observation["pending_call"], "original_failure": observation["original_failure"],
        "original_request": started["request"], "forbidden_task_id": read(checked(request["contract"]))["task_id"],
        "old_ledger": str(stage.parent.parent), "recovery_evidence": [bind(RECOVERY / n) for n in
            ("task_list_observation.json", "provider_logs.json", "credit_snapshot.json")],
        "state": "unknown_contained_not_terminal", "resubmit_allowed": False,
        "reserved_nominal_usd": 16.0, "new_nominal_usd_limit": 40.0,
        "prior_known_nominal_usd": 4.2502,
        "prior_known_cost_evidence": bind(OLD / "repairs/schema_wire_v2/partial_usage_after_503.json"),
        "scope_reserved_total_nominal_usd": 60.2502,
        "scope_note": "Prior pilot known debit + conservative old unknown reserve + new allowance; not full history spend or cash invoice.",
        "resolution_claimed": False, "automatic_retry": False}


def freeze(root):
    source = ROOT / "corl2026_workshop/experiments/autonomous_pipeline_v4"
    dest = root / "releases" / VERSION / "autonomous_pipeline_v4"
    comparisons = []
    for path in sorted(source.glob("*.py")):
        raw = path.read_bytes()
        if path.name == "models.py":
            old = b"from .review_transport import generation_config, wire_schema"
            require(raw.count(old) == 1, "unexpected_models_import")
            raw = raw.replace(old, b"from .review_wire_v2 import generation_config, wire_schema")
        write_bytes(dest / path.name, raw)
        comparisons.append({"file": path.name, "source": bind(path), "candidate": bind(dest / path.name),
                            "same_bytes": path.read_bytes() == raw})
    require([r["file"] for r in comparisons if not r["same_bytes"]] == ["models.py"], "unexpected_release_change")
    write_bytes(dest / "review_wire_v2.py", Path(review_wire_v2.__file__).read_bytes())
    bridge = root / "releases" / pilot.RELEASE / "human_pilot_v1.py"
    write_bytes(bridge, Path(pilot.__file__).read_bytes())
    driver = dest.parent / "human_examples_v2.py"; write_bytes(driver, Path(__file__).read_bytes())
    manifest = {"schema": "resimpair_release_v4", "release": VERSION,
        "files": [bind(p) for p in sorted(dest.glob("*.py"))]+[bind(driver), bind(bridge)],
        "experiment": bind(root / "experiment.json"), "release_changes": comparisons,
        "only_protocol_change": "Previously qualified Gemini responseSchema wire encoder; original local schemas unchanged."}
    write(dest.parent / "release.json", manifest)
    return bind(dest.parent / "release.json")


def modules(root):
    manifest = read(root / "releases" / VERSION / "release.json")
    for item in manifest["files"]: checked(item)
    require(bind(root / "releases" / VERSION / "human_examples_v2.py")["sha256"] == bind(__file__)["sha256"],
            "execute_frozen_driver")
    require(bind(root / "releases" / pilot.RELEASE / "human_pilot_v1.py")["sha256"] == bind(pilot.__file__)["sha256"],
            "retained_bridge_changed")
    package = "_human_examples_" + digest(manifest)[:20]
    directory = root / "releases" / VERSION / "autonomous_pipeline_v4"
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, directory / "__init__.py", submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec); sys.modules[package] = module; spec.loader.exec_module(module)
    return lambda name: importlib.import_module(package + "." + name)


def prepare(root, contracts):
    root = Path(root).resolve()
    require(not root.exists(), "candidate_exists_resume_never_reset")
    require(len(contracts) == 3 and len(set(contracts)) == 3, "three_fixed_examples_required")
    require(shutil.disk_usage(root.parent).free > 10*2**30, "need_10GiB_free")
    with exclusive(ROOT / "runtime/.new_campaign.lock"):
        containment = containment_record(); h = historical(containment)
        qualified = [pilot.qualify_contract(Path(p)) for p in contracts]
        keys = [read(p)["task_id"] for p in contracts]
        require(not set(keys) & set(h["attempted_video_tasks"]), "historical_task_video_attempt_must_not_repeat")
        write(root / "containment.json", containment)
        write(root / "historical_lineage.json", h)
        registration = {"schema": "resimpair_portable_campaign_v1", "name": root.name, "purpose": VERSION,
            "new_call_limits": LIMITS, "nominal_usd_admission_limit": 40,
            "observed_quota_stop_limit": 20_000_000, "api_concurrency": 1, "automatic_retry": False,
            "containment": bind(root / "containment.json")}
        write(root / "portable_registration.json", registration)
        write(root / "budget.json", {"schema": "resimpair_cumulative_budget_v4", "campaign_id": root.name,
            "ledger_path": str(root / "campaign_ledger"), "portable_campaign": True,
            "portable_registration": bind(root / "portable_registration.json"), "new_call_limits": LIMITS,
            "ancestor_started_calls": h["started_calls"], "ancestor_evidence": h["evidence"]+[bind(root / "containment.json")],
            "lineage_cumulative_limits": {k:h["started_calls"][k]+LIMITS[k] for k in LIMITS},
            "max_concurrent": 1, "timeout_seconds": 600, "automatic_retry": False, "currency_budget": None,
            "nominal_usd_admission_limit": 40, "observed_quota_stop_limit": 20_000_000,
            "reserve_quota_before_new_call": {"image":500000, "video":8000000, "review":500000},
            "authorization": "User standing generation authorization; parent approved bounded3 examples and exact unknown containment.",
            "containment": bind(root / "containment.json")})
        write(root / "experiment.json", {**pilot._experiment(False), "schema": VERSION,
            "simulation_executions_this_campaign": 0, "reused_historical_native_executions": len(contracts)})
        write(root / "config.json", read(ROOT / "configs/providers.json"))
        write(root / "human_runtime.json", {"schema": "resimpair_human_runtime_v1", "budget": bind(root / "budget.json"),
            "media_upload_limit": 3, "provider_application": bind(ROOT / "provenance/provider_app.js"),
            "per_video_poll_deadline_seconds": 900, "automatic_retry": False})
        release = freeze(root); module = modules(root)
        records = [pilot._prepare_one(root, module, Path(p)) for p in contracts]
        for row in records: row["simulation_scope"] = "reused_historical_native_execution_not_new_physics"
        index = {"schema": VERSION, "records": records, "release": release,
            "planned_calls": {"image": len({r["anchor_id"] for r in records}), "video": 3,
                              "review": 3+len({r["anchor_id"] for r in records})},
            "budget": bind(root / "budget.json"), "model_config": bind(root / "config.json"),
            "inputs_frozen_before_generation": True, "no_failed_case_replacement": True}
        write(root / "index.json", index)
        module("budget").Budget(root / "budget.json")
        write(root / "protocol.json", {"schema": VERSION, "index": bind(root / "index.json"), "release": release,
            "containment": bind(root / "containment.json"), "historical_lineage": bind(root / "historical_lineage.json"),
            "driver": bind(root / "releases" / VERSION / "human_examples_v2.py"),
            "original_native_qualification": qualified, "calls": LIMITS, "incremental_nominal_usd": 40,
            "unknown_old_nominal_usd_reserved": 16, "old_known_nominal_usd": 4.2502,
            "api_concurrency":1, "new_unknown_stops_campaign":True, "automatic_retry":False,
            "independent_of_old_request":True, "new_physical_executions":0, "reused_physical_executions":3,
            "native_and_human_acceptance_unchanged":True, "source_coverage":"missing", "formal_pairs":0})
        validate(root)
    return {"status":"prepared", "protocol":bind(root / "protocol.json"), "planned_calls":index["planned_calls"], "api_calls":0}


def validate(root):
    root = Path(root).resolve(); p=read(root / "protocol.json"); pilot.check_bindings(p)
    c=read(root / "containment.json"); pilot.check_bindings(c)
    h=historical(c, root.name); original=read(root / "historical_lineage.json")
    require(h == original, "historical_lineage_changed_reaudit_required")
    require(c["resubmit_allowed"] is False and c["state"] == "unknown_contained_not_terminal"
            and c["reserved_nominal_usd"] == 16, "containment_changed")
    module=modules(root); module("batch_human_queue").validate_budget(root)
    index=read(root / "index.json")
    require(index["budget"]==bind(root / "budget.json") and index["model_config"]==bind(root / "config.json"), "frozen_inputs_changed")
    for row in index["records"]:
        require(pilot.qualify_contract(resolve(row["source_contract"])) == row["native_qualification"], "native_acceptance_changed")
        require(row["key"] not in h["attempted_video_tasks"], "old_task_resubmit_blocked")
        for name in ("contract","image_request","motion_reference","reference_video"): checked(row[name])
        contract=module("compiler").completed_contract(checked(row["contract"]))
        require(row["base_prompt"]==module("prompts").video_prompt(contract)
                and contract["human_evidence_plan"]==module("evidence_plan").compile_plan(contract), "conditions_or_evidence_changed")
    return module,index,c,h


def guarded_budget(module, root, containment):
    class ContainedBudget(module("motion_reference").ComparisonBudget):
        def call(self, kind, request, operation):
            # The normal Budget retains its exact no-resubmit and new-unknown
            # guard. Only registration recognizes the separately locked ancestor.
            pilot.check_bindings(containment)
            require(bind(request)["sha256"] != containment["original_request"]["sha256"], "old_request_resubmit_forbidden")
            value=read(request)
            if value.get("contract"):
                require(read(checked(value["contract"]))["task_id"] != containment["forbidden_task_id"], "old_task_resubmit_forbidden")
            return super().call(kind, request, operation)
    return ContainedBudget(root)


def run(root):
    root=Path(root).resolve(); module,index,c,h=validate(root); module("core").activate()
    with ExitStack() as stack:
        for lock in [ROOT / "runtime/.paid_containment.lock", root / ".paid_batch.lock"]+[Path(p)/".execution.lock" for p in h["ledgers"]]:
            stack.enter_context(exclusive(lock))
        validate(root)
        if (root / "results.json").exists(): return read(root / "results.json")
        budget=guarded_budget(module,root,c)
        module("billing").snapshot(root,root / "billing/campaign_before.json")
        rows=[]; started=time.time()
        replace_state(root / "progress.json", {"status":"running","started_unix":started,"automatic_retry":False})
        try:
            for row in index["records"]:
                dest=root / "example_results" / (row["key"]+".json")
                if dest.exists(): rows.append(read(dest)); continue
                begin=time.time(); anchor=Path(row["anchor_directory"]); contract=read(checked(row["contract"]))
                receipt=module("models").generate_image(root,budget,anchor / "image_request.json",anchor / "human.png")
                reviewed,unavailable=module("motion_comparison").review_or_unknown(budget,anchor / "review_request.json",
                    lambda:module("review").anchor(root,budget,anchor,receipt,contract["initial_scene_information"],contract["initial_image"]))
                if unavailable:
                    write(anchor / "unusable_review.json",unavailable)
                    result={"key":row["key"],"status":"initial_image_review_unknown_video_unattempted","video":None}
                elif not reviewed["gate"]["passed_in_sampled_visual_scope"]:
                    result={"key":row["key"],"status":"initial_image_rejected_video_unattempted","video":None}
                else:
                    result=module("motion_comparison").run_video(root,budget,row,"motion_reference",receipt["image"],
                        bind(anchor / "review.json"),poll_deadline_seconds=900,
                        wire_schema_adapter=module("review_transport").WIRE_SCHEMA_ADAPTER)
                result={**result,"contract":row["contract"],"anchor":receipt["image"],"source_contract":row["source_contract"],
                    "simulation_scope":row["simulation_scope"],"elapsed_seconds":time.time()-begin,
                    "source_coverage":"missing","formal_pair_qualified":False}
                write(dest,result); rows.append(result)
                print(json.dumps({"key":row["key"],"status":result["status"],"usage":budget.usage()},ensure_ascii=False),flush=True)
                if "unknown" in result["status"] or result["status"]=="waiting_existing_video": break
            status="stopped_on_unknown" if len(rows)<len(index["records"]) or any("unknown" in r["status"] or r["status"]=="waiting_existing_video" for r in rows) else "terminal"
        except Exception as exc:
            replace_state(root / "progress.json",{"status":"stopped_on_exception","exception_type":type(exc).__name__,
                "code":getattr(exc,"code","operation_failed"),"automatic_retry":False,"updated_unix":time.time()})
            module("billing").snapshot(root,root / "billing/campaign_after.json")
            module("billing").report(root,root / "usage.json")
            raise
        module("billing").snapshot(root,root / "billing/campaign_after.json")
        module("billing").report(root,root / "usage.json")
        result={"schema":VERSION,"status":status,"rows":rows,"unattempted_keys":[r["key"] for r in index["records"] if r["key"] not in {x["key"] for x in rows}],
            "protocol":bind(root / "protocol.json"),"usage":bind(root / "usage.json"),"elapsed_seconds":time.time()-started,
            "training_admitted":False,"independent_quality":"unknown_until_separate_pixel_inspection","formal_pairs":0}
        write(root / "results.json",result); replace_state(root / "progress.json",{"status":status,"updated_unix":time.time(),"automatic_retry":False})
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=["prepare","validate","run"])
    p.add_argument("--root",type=Path,required=True);p.add_argument("--contract",type=Path,action="append",default=[])
    a=p.parse_args()
    if a.command=="prepare": r=prepare(a.root,a.contract)
    elif a.command=="run": r=run(a.root)
    else:
        validate(a.root); r={"status":"local_validation_passed","api_calls":0}
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
