"""Bounded endpoint-conditioning comparisons over retained native executions.

Every arm shares a task prompt, physical execution and initial human anchor.
The two endpoint arms additionally share the exact terminal anchor. Preparation
freezes cumulative budgets and code; unknown submissions never get reposted.
This candidate does not change default generation or independent acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

from resimpair_runtime import ROOT, configure
configure()
from resimpair_tasks import human_examples_v2 as ancestor
from corl2026_workshop.experiments.autonomous_pipeline_v4 import billing, media
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import (
    bind, checked, digest, read, replace_state, require, write, write_bytes,
)
from corl2026_workshop.experiments.autonomous_pipeline_v4.motion_reference import (
    ComparisonBudget, derivative, upload_reference,
)

VERSION = "endpoint_campaign_v1"
LIMITS = {"image": 4, "video": 6, "review": 10}
METHODS = ("motion_reference", "endpoint_reference", "endpoint_frames")
RETAINED = ROOT / "runs/alignment_human_examples_20261028"


def retained_modules():
    """Reuse already exercised immutable transport/schema adapter, no rewrite."""
    manifest = read(RETAINED / "releases/human_examples_v2/release.json")
    for item in manifest["files"]:
        checked(item)
    directory = RETAINED / "releases/human_examples_v2/autonomous_pipeline_v4"
    package = "_endpoint_retained_" + digest(manifest)[:16]
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, directory / "__init__.py",
                                                      submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return lambda name: importlib.import_module(package + "." + name)


def _unknown_observation(root, containment):
    """Read only; failure to identify the old 503 does not retire its reserve."""
    dest = root / "recovery/old_unknown_observation.json"
    if dest.exists():
        return read(dest)
    _, origin, key = billing.connection(root)
    response = billing._get(origin, "/api/v3/contents/generations/tasks?page_num=1&page_size=500", key)
    body = response.get("data", {})
    rows = body.get("items", body.get("data", [])) if isinstance(body, dict) else []
    if isinstance(rows, dict):
        rows = rows.get("items", [])
    safe = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        safe.append({"id_sha256": hashlib.sha256(str(row.get("id", "")).encode()).hexdigest(),
                     "status": row.get("status"), "created_at": row.get("created_at")})
    value = {"schema": "endpoint_old_unknown_readonly_v1", "observed_unix": time.time(),
             "pending_call": containment["pending_call"], "http_status": response.get("http_status"),
             "returned_safe_rows": safe, "old_outcome": "unknown", "resubmit_allowed": False,
             "new_posts": 0, "matching_limit": "No task ID was returned by the old503; absence is not proof of nonexecution."}
    write(dest, value)
    old_start = read(checked(containment["pending_call"]))["started_unix"]
    billing.token_logs(root, root / "recovery/old_window_logs.json", old_start - 60, old_start + 1800)
    return value


def prepare(root, cases, rubric, rubric_map=None):
    root = Path(root).resolve()
    require(not (root / "protocol.json").exists(), "prepared_campaign_resume_do_not_reset")
    require(1 <= len(cases) <= 2, "one_development_and_at_most_one_layout_validation_case")
    require(shutil.disk_usage(root.parent).free > 10 * 2**30, "need_10GiB_free")
    from resimpair_tasks import endpoint_conditioning_v1 as candidate
    with ancestor.exclusive(ROOT / "runtime/.new_campaign.lock"):
        containment = ancestor.containment_record()
        history = ancestor.historical(containment)
        write(root / "historical_lineage.json", history)
        write(root / "config.json", read(ROOT / "configs/providers.json"))
        recovery = _unknown_observation(root, containment)
        containment = {**containment, "new_readonly_observation": bind(root / "recovery/old_unknown_observation.json"),
                       "preceding_completed_campaign": bind(RETAINED / "protocol.json"),
                       "preceding_completed_campaign_nominal_usd": 12.723492,
                       "scope_reserved_total_nominal_usd": 72.973692,
                       "scope_note": "Two latest campaigns known16.973692 + old unknown reserve16 + new limit40; earlier total cash spend unknown."}
        write(root / "containment.json", containment)
        registration = {"schema": "resimpair_portable_campaign_v1", "name": root.name, "purpose": VERSION,
                        "new_call_limits": LIMITS, "nominal_usd_admission_limit": 40,
                        "observed_quota_stop_limit": 20_000_000, "api_concurrency": 1,
                        "automatic_retry": False, "containment": bind(root / "containment.json")}
        write(root / "portable_registration.json", registration)
        spec = {"schema": "resimpair_cumulative_budget_v4", "campaign_id": root.name,
                "ledger_path": str(root / "campaign_ledger"), "portable_campaign": True,
                "portable_registration": bind(root / "portable_registration.json"),
                "new_call_limits": LIMITS, "ancestor_started_calls": history["started_calls"],
                "ancestor_evidence": history["evidence"] + [bind(root / "containment.json")],
                "lineage_cumulative_limits": {k: history["started_calls"][k] + LIMITS[k] for k in LIMITS},
                "max_concurrent": 1, "timeout_seconds": 600, "automatic_retry": False,
                "currency_budget": None, "nominal_usd_admission_limit": 40,
                "observed_quota_stop_limit": 20_000_000,
                "reserve_quota_before_new_call": {"image": 500000, "video": 8000000, "review": 500000},
                "authorization": "User approved implementing and testing endpoint conditioning; standing bounded API authorization.",
                "containment": bind(root / "containment.json")}
        write(root / "budget.json", spec)
        write(root / "human_runtime.json", {"schema": "resimpair_human_runtime_v1",
              "budget": bind(root / "budget.json"), "media_upload_limit": len(cases),
              "provider_application": bind(ROOT / "provenance/provider_app.js"),
              "per_video_poll_deadline_seconds": 1200, "automatic_retry": False})
        case_records = []
        for file in cases:
            case = read(file)
            for field in ("task_contract", "simulation_video", "simulation_initial", "simulation_terminal", "human_initial", "execution"):
                checked(case[field])
            if case.get("human_terminal"):
                checked(case["human_terminal"])
            require(case["case_id"] not in [v["case_id"] for v in case_records], "duplicate_case")
            require(read(checked(case["task_contract"])).get("task_id") != containment["forbidden_task_id"],
                    "old_unknown_task_is_forbidden")
            case_records.append({"case_id": case["case_id"], "manifest": bind(file)})
        case_rubrics = read(rubric_map) if rubric_map else {x["case_id"]: bind(rubric) for x in case_records}
        require(set(case_rubrics) == {x["case_id"] for x in case_records}, "exact_case_rubrics_required")
        for item in case_rubrics.values():
            checked(item)
        protocol = {"schema": VERSION, "created_unix": time.time(), "cases": case_records,
                    "methods": list(METHODS), "calls": LIMITS, "new_nominal_usd_limit": 40,
                    "old_unknown_nominal_usd_reserve": 16, "api_concurrency": 1,
                    "budget": bind(root / "budget.json"), "rubric": bind(rubric), "case_rubrics": case_rubrics,
                    "code": {"driver": bind(__file__), "conditioning": bind(candidate.__file__),
                             "retained_release": bind(RETAINED / "releases/human_examples_v2/release.json")},
                    "history": bind(root / "historical_lineage.json"),
                    "hypothesis": "Shared terminal state may improve correspondence without losing same-task motion evidence.",
                    "primary_comparison": "A vs B adds only terminal reference and its role instruction; B vs C trades motion-reference input for dedicated boundary roles.",
                    "shared_conditions": "same native execution, initial/terminal images, task prompt, duration720p8s; all arms explicit anatomical hand rule",
                    "baseline_scope": "fresh single-reference control with same cleaned task prompt; not the earlier offscreen-inventory prompt",
                    "replications_per_arm_per_case": 1,
                    "heldout_gate": "Second exposed task/layout only after independent development review identifies useful endpoint gain without a new confirmed required-dimension failure; same real source, not a blind holdout.",
                    "semantic_retry": False, "new_unknown_stops_campaign": True,
                    "promotion": "none; one draw per arm cannot establish general yield", "new_physics": 0}
        write(root / "protocol.json", protocol)
        ComparisonBudget(root)
        before = billing.snapshot(root, root / "billing/campaign_before.json")
        write(root / "prepared.json", {"protocol": bind(root / "protocol.json"),
                                      "starting_credit": bind(root / "billing/campaign_before.json"),
                                      "api_generation_calls": 0})
        return {"status": "prepared", "calls": LIMITS, "ancestor_started": history["started_calls"],
                "new_nominal_limit": 40, "old_unknown_preserved": True}


def validate(root):
    root = Path(root).resolve()
    p = read(root / "protocol.json")
    for b in p["code"].values():
        checked(b)
    checked(p["rubric"]); checked(p["budget"]); checked(p["history"])
    for item in p["case_rubrics"].values():
        checked(item)
    require(p["code"]["driver"] == bind(__file__), "execute_frozen_driver")
    c = read(root / "containment.json")
    h = ancestor.historical(c, root.name)
    require(h == read(root / "historical_lineage.json"), "ancestor_activity_changed_reaudit")
    for row in p["cases"]:
        checked(row["manifest"])
    return p, c, h


def locked(root):
    root = Path(root).resolve(); p, c, h = validate(root)
    stack = ExitStack()
    try:
        for lock in [ROOT / "runtime/.paid_containment.lock", root / ".paid_batch.lock"] + [Path(x) / ".execution.lock" for x in h["ledgers"]]:
            stack.enter_context(ancestor.exclusive(lock))
        validate(root)
    except Exception:
        stack.close(); raise
    return stack


class GuardedBudget(ComparisonBudget):
    def call(self, kind, request, operation):
        c = read(self.root / "containment.json")
        require(bind(request)["sha256"] != c["original_request"]["sha256"], "old_unknown_resubmit_forbidden")
        value = read(request)
        if value.get("contract"):
            require(read(checked(value["contract"])).get("task_id") != c["forbidden_task_id"], "old_unknown_task_forbidden")
        return super().call(kind, request, operation)


def case_at(root, case_id):
    p = read(Path(root) / "protocol.json")
    rows = [x for x in p["cases"] if x["case_id"] == case_id]
    require(len(rows) == 1, "undeclared_case")
    return read(checked(rows[0]["manifest"]))


def _case_gate(root, case):
    p = read(root / "protocol.json")
    if case["case_id"] != p["cases"][0]["case_id"]:
        gate = read(root / "development_decision.json")
        require(gate["allow_next_layout"] is True and gate["rubric"] == p["rubric"], "heldout_not_yet_qualified")
        checked(gate["independent_evidence"])


def run_arm(root, case_id, method, admission):
    from resimpair_tasks import endpoint_conditioning_v1 as candidate
    root = Path(root).resolve(); case = case_at(root, case_id)
    require(method in METHODS, "undeclared_arm")
    module = retained_modules(); module("core").activate()
    with locked(root):
        _case_gate(root, case)
        approval = read(admission)
        require(approval.get("anchors_approved_for_controlled_generation") is True,
                "independent_anchor_admission_required")
        case_rubric = read(root / "protocol.json")["case_rubrics"][case_id]
        require(approval["rubric"] == case_rubric, "anchor_rubric_mismatch")
        budget = GuardedBudget(root); directory = root / "cases" / case_id / method
        if (directory / "result.json").exists():
            return read(directory / "result.json")
        terminal = case.get("human_terminal")
        if not terminal:
            terminal = bind(root / "cases" / case_id / "terminal/human.png")
        human_initial, human_terminal = bind(checked(case["human_initial"])), bind(checked(terminal))
        for field, actual in [("human_initial", human_initial), ("human_terminal", human_terminal)]:
            reviewed = approval["bindings"][field]
            checked(reviewed)
            require(reviewed["sha256"] == actual["sha256"], "anchor_admission_bound_to_other_bytes")
        kwargs = {"contract": case["task_contract"], "task_prompt": case["task_prompt"],
                  "initial_image": human_initial, "condition_evidence": {"anchor_verdict": bind(admission),
                    "rubric": case_rubric, "case_manifest": bind(checked(next(x["manifest"] for x in read(root / "protocol.json")["cases"] if x["case_id"]==case_id)))},
                  "settings": {"duration": 8, "resolution": "720p", "ratio": "adaptive", "generate_audio": False, "watermark": False}}
        if method != "motion_reference":
            kwargs["terminal_image"] = human_terminal
        if method != "endpoint_frames":
            native = {"simulation_video": case["simulation_video"], "execution": case["execution"],
                      "generation_information": {"duration_seconds": 8}}
            derived = derivative(native, root / "cases" / case_id / "motion/reference.mp4")
            upload_reference(root, derived)
            kwargs.update(reference_video=derived["video"],
                          remote_reference=bind(root / "uploads" / derived["video"]["sha256"] / "receipt.json"))
        request_path = directory / "request.json"
        write(request_path, candidate.build_request(root, method, **kwargs))
        started_path = directory / "started.json"
        if not started_path.exists():
            write(started_path, {"started_unix": time.time(), "request": bind(request_path), "method": method})
        submission = candidate.submit(root, budget, request_path)
        write(directory / "submission.json", submission)
        replace_state(root / "status.json", {"status": "video_inflight", "case": case_id, "method": method,
                      "stage": submission["stage"], "updated_unix": time.time()})
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            result = module("models").poll_video(root, submission, directory / "human.mp4")
            if result["status"] in {"downloaded", "failed", "expired", "cancelled"}:
                break
            time.sleep(12)
        else:
            replace_state(root / "status.json", {"status": "poll_timeout_resume_same_id", "stage": submission["stage"],
                          "case": case_id, "method": method, "updated_unix": time.time()})
            return {"status": "poll_timeout_resume_same_id", "submission": submission}
        budget.settle_video(submission["stage"])
        if result["status"] == "downloaded":
            packet = media.packet(directory / "human.mp4", directory / "frames", frame_budget=49)
            result["frame_packet"] = bind(directory / "frames/packet.json")
        result.update(case_id=case_id, method=method, request=bind(request_path), submission=submission,
                      elapsed_seconds=time.time()-read(started_path)["started_unix"], formal_pair=False)
        write(directory / "result.json", result)
        replace_state(root / "status.json", {"status": "arm_terminal", "case": case_id, "method": method,
                      "updated_unix": time.time(), "usage": budget.usage()})
        return result


def generate_terminal(root, case_id, evidence):
    from resimpair_tasks import endpoint_conditioning_v1 as candidate
    root = Path(root).resolve(); case = case_at(root, case_id)
    require(not case.get("human_terminal"), "reuse_existing_terminal_do_not_regenerate")
    module = retained_modules(); module("core").activate()
    with locked(root):
        _case_gate(root, case)
        approval = read(evidence)
        require(approval.get("initial_approved_for_terminal_image_generation") is True,
                "independent_initial_admission_required_for_new_terminal")
        require(approval["rubric"] == read(root / "protocol.json")["case_rubrics"][case_id],
                "terminal_generation_rubric_mismatch")
        for field in ["human_initial", "simulation_initial", "simulation_terminal"]:
            reviewed = approval["bindings"][field]
            checked(reviewed)
            require(reviewed["sha256"] == case[field]["sha256"], "terminal_input_not_independently_reviewed")
        directory = root / "cases" / case_id / "terminal"
        req = candidate.build_terminal_image_request(root, human_initial=case["human_initial"],
              simulation_initial=case["simulation_initial"], simulation_terminal=case["simulation_terminal"],
              task_delta_prompt=case["task_delta_prompt"], condition_evidence={"anchor_verdict": bind(evidence)})
        write(directory / "request.json", req)
        return module("models").generate_image(root, GuardedBudget(root), directory / "request.json", directory / "human.png")


def run_review(root, request_path, output):
    root = Path(root).resolve(); module = retained_modules(); module("core").activate()
    with locked(root):
        result, receipt = module("models").reviewed_json(root, GuardedBudget(root), request_path)
        value = {"observation": result, "receipt": receipt, "scope": "VLM observations only; independent verdict required"}
        write(output, value)
        return value


def close(root):
    root = Path(root).resolve()
    with locked(root):
        billing.snapshot(root, root / "billing/campaign_after.json")
        value = billing.report(root, root / "usage.json")
        replace_state(root / "status.json", {"status": "closed", "updated_unix": time.time(),
                      "usage": value["new_started"], "nominal_usd": value.get("provider_logged_nominal_usd"),
                      "old_unknown_preserved": True, "default_promoted": False})
        return {"status": "closed", "calls": value["new_started"],
                "nominal_usd": value.get("provider_logged_nominal_usd"), "formal_pairs": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "run-arm", "terminal", "review", "close"])
    p.add_argument("--root", type=Path, required=True);p.add_argument("--case", action="append", default=[])
    p.add_argument("--rubric", type=Path);p.add_argument("--rubric-map", type=Path);p.add_argument("--case-id");p.add_argument("--method", choices=METHODS)
    p.add_argument("--admission", type=Path);p.add_argument("--request", type=Path);p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.command == "prepare":result = prepare(a.root, a.case, a.rubric, a.rubric_map)
    elif a.command == "run-arm":result = run_arm(a.root, a.case_id, a.method, a.admission)
    elif a.command == "terminal":result = generate_terminal(a.root, a.case_id, a.admission)
    elif a.command == "review":result = run_review(a.root, a.request, a.output)
    else:result = close(a.root)
    # Contains local bindings and opaque IDs, never provider credentials or URLs.
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
