"""Finite dispatcher for the bound cabinet case; plan-only by default.

Only --execute can invoke paid cached stages. It does not approve training.
Unimplemented routes or unknown conditions stop, rather than silently bypassing
the planner. The first anchor may have been produced manually; provenance keeps
that fact and reuses its immutable paid-stage cache.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation_methods.scripts import run_state_conditioned_case_20261020c as runner
from data_generation_methods.r2h_pipeline.common import file_sha256, json_sha256, load_env_file, write_json
from data_generation_methods.r2h_pipeline.providers import SeedanceClient
from data_generation_methods.r2h_pipeline.repair_router import plan_repairs

ROOT, CASE = runner.ROOT, runner.CASE
PAID_STAGES = ("edit_initial_v1", "edit_terminal_v1", "condition_review_v1", "submit_first_last_v1")


def _case():
    # Runtime runner.CASE lookup supports isolated fixture cases.
    return runner.CASE


def _dispatch():
    return _case() / "dispatcher_v1"


def _bind(path):
    return runner.binding(path)


def _read(path):
    return json.loads(Path(path).read_text())


def started_stages():
    return [name for name in PAID_STAGES if (_case() / name / "call_started.json").exists()]


def stage_needs_new_call(name):
    folder = _case() / name
    # A started-but-unresolved call cannot be resubmitted by this dispatcher.
    if (folder / "call_started.json").exists() and not (folder / "result.json").exists():
        raise RuntimeError("Unresolved paid stage; reconcile first: " + name)
    return not (folder / "result.json").exists()


def map_actor_evidence():
    """Copy visible counts/locations from the actual static review, adding no GT."""
    source_path = _case() / "condition_review_v1/result.json"
    if not source_path.exists():
        return None
    source = _read(source_path)
    request_path = _case() / "condition_review_v1/request_metadata.json"
    request = _read(request_path)
    observed_images = [item for item in request["images"]
                       if item.get("label") == "CANDIDATE HUMAN CLOSED"]
    if len(observed_images) != 1:
        raise ValueError("Review request must bind exactly one candidate initial image")
    current_initial = _bind(_case() / "human_initial_v1.png")
    if any(observed_images[0].get(key) != current_initial[key] for key in ("path", "sha256")):
        raise ValueError("Body observation was made on a different initial image")
    observed = source["review"]["first_hands"]
    locations = observed["locations"]
    if any(not isinstance(location, str) or not location.strip() for location in locations):
        raise ValueError("Cannot map an empty or non-string observed hand location")
    mapped = {
        "minimum_visible_hand_count": observed["minimum_visible_count"],
        "exact_count_known": observed["exact_count_known"],
        "hands": [{"location": location, "description": location} for location in locations],
        "limitations": [
            "Global inventory from four labelled static images, not a localized crop review.",
            "Descriptions repeat original location text; no extra anatomical evidence was inferred.",
        ],
    }
    artifact = {
        "schema": "r2h_global_body_inventory_mapping_v1", "review": mapped,
        "source_observation": _bind(source_path), "source_field": "review.first_hands",
        "source_request": _bind(request_path), "media": current_initial,
        "mapping": {
            "minimum_visible_count": "minimum_visible_hand_count",
            "exact_count_known": "exact_count_known",
            "locations": "hands[].location and identical hands[].description",
        },
        "observation_values_changed": False, "review_scope": "global",
        "training_admitted": False,
    }
    path = _dispatch() / "initial_body_mapping.json"
    if path.exists() and _read(path) != artifact:
        # The already-executed first dispatcher version omitted source_request.
        # Migrate only that exact shape after checking the original request image
        # above. Original observations and the legacy file remain untouched.
        legacy = dict(artifact)
        legacy.pop("source_request")
        if _read(path) != legacy:
            raise ValueError("Existing derived inventory differs; preserve old evidence and version it")
        artifact = {
            **artifact, "schema": "r2h_global_body_inventory_mapping_v2",
            "legacy_mapping": _bind(path),
            "migration": "source_request_binding_added_without_observation_changes",
        }
        path = _dispatch() / "initial_body_mapping_v2.json"
        if path.exists() and _read(path) != artifact:
            raise ValueError("Existing versioned inventory differs from verified migration")
    write_json(path, artifact)
    return {"maximum_visible_hands": 1, "review_scope": "global",
            "review": mapped, "media_sha256": artifact["media"]["sha256"],
            "evidence_artifact": _bind(path)}


def prepare_transport(cfg):
    """Validate the exact shared payload locally; no provider use is claimed."""
    payload = runner.build_video_payload(cfg)
    prepared = SeedanceClient(cfg["seedance"]).prepare_submit(payload)
    images = [item for item in payload["content"] if item.get("type") == "image_url"]
    roles = [item.get("role") for item in images]
    if roles != ["first_frame", "last_frame"]:
        raise ValueError("Require exactly the actual first_frame and last_frame payload roles")
    record = {
        "schema": "r2h_condition_transport_preparation_v1",
        "payload_sha256": json_sha256(payload), "structured_input_roles": roles,
        "initial": _bind(_case() / "human_initial_v1.png"),
        "terminal": _bind(_case() / "human_terminal_v1.png"),
        "model": cfg["seedance"]["model"],
        "validation_scope": "local_payload_structure_and_adapter_compatibility",
        "actual_video_function_uses_same_payload_builder": True,
        "provider_request_performed": False, "provider_condition_use_verified": False,
        "training_admitted": False,
    }
    # Never persist inline image bytes or credentials from prepared transport.
    del prepared
    path = _dispatch() / "condition_transport_preparation.json"
    if path.exists() and _read(path) != record:
        raise ValueError("Prepared payload changed; version condition evidence")
    write_json(path, record)
    return record


def build_facts(source, *, max_total_paid_stages=4):
    initial = _case() / "human_initial_v1.png"
    if not initial.exists():
        # The hidden-state handler first creates this human anchor. This binding
        # names the supplied sim initial and never calls it a verified human image.
        initial_binding = source["sim_initial"]
    else:
        initial_binding = _bind(initial)
    terminal = _case() / "human_terminal_v1.png"
    transport = _dispatch() / "condition_transport_preparation.json"
    hidden = {
        "required": True, "reveal_condition_present": terminal.exists(),
        "condition_transport_verified": transport.exists(),
    }
    if terminal.exists():
        hidden["condition_artifact"] = _bind(terminal)
    if transport.exists():
        recorded = _read(transport)
        for key, path in (("initial", initial), ("terminal", terminal)):
            if recorded[key] != _bind(path):
                raise ValueError("Condition image changed after transport preparation")
        hidden["transport_evidence"] = _bind(transport)
    body = map_actor_evidence() if initial.exists() else None
    facts = {
        "schema": "r2h_repair_facts_v1",
        "reference": source["original_human_window"]["video"],
        "anchor": initial_binding, "hidden_state": hidden,
        "actor_inventory": body or {
            "maximum_visible_hands": 1, "review_scope": "none", "review": None},
        "protocol": {"state": "ok"}, "pair_validation": {"state": "not_run"},
        "attempts": {
            "body_reviews": int((_case() / "condition_review_v1/call_started.json").exists()),
            "anchor_repairs": int((_case() / "edit_initial_v1/call_started.json").exists()),
            "contact_reviews": 0,
        },
        "budget": {
            "paid_calls_remaining": max(0, max_total_paid_stages - len(started_stages())),
            "max_body_reviews": 2, "max_anchor_repairs": 1,
            "max_contact_reviews": 1, "max_review_frames": 48,
        },
    }
    unresolved = [name for name in PAID_STAGES
                  if (_case() / name / "call_started.json").exists()
                  and not (_case() / name / "result.json").exists()]
    if unresolved:
        facts["protocol"] = {
            "state": "submission_unknown", "unresolved_stages": unresolved}
    target = _case() / "state_conditioned_human_v1.mp4"
    if target.exists():
        facts["video"] = _bind(target)
    return facts


def planned_action(plan):
    if plan["actions"]:
        return plan["actions"][0]
    if plan["allowed"]["video_submission"]:
        return {"kind": "submit_verified_conditioned_video",
                "execution_status": "planned", "parameters": {}}
    return None


def actual_stages_for_action(kind):
    return {
        "extract_reveal_or_terminal_anchor": ["edit_initial_v1", "edit_terminal_v1"],
        "verify_multi_anchor_condition_transport": [],
        "review_anchor_body_global": ["condition_review_v1"],
        "submit_verified_conditioned_video": ["submit_first_last_v1"],
        "inspect_task_evidence_and_training_pair": [],
    }.get(kind)


def execute_action(action, cfg, source, *, max_total_paid_stages=4, poll_minutes=25):
    """Execute only implemented handlers after local budget checks."""
    kind = action["kind"]
    stages = actual_stages_for_action(kind)
    if stages is None:
        return {"executed": False, "reason": "handler_not_implemented", "kind": kind}
    needs_new = [stage for stage in stages if stage_needs_new_call(stage)]
    if len(started_stages()) + len(needs_new) > max_total_paid_stages:
        return {"executed": False, "reason": "actual_paid_stage_budget_exhausted", "kind": kind}
    if kind == "extract_reveal_or_terminal_anchor":
        runner.edit_first(cfg, source)  # Reuses the manually produced stage when present.
        runner.edit_last(cfg, source)
    elif kind == "verify_multi_anchor_condition_transport":
        prepare_transport(cfg)
    elif kind == "review_anchor_body_global":
        runner.review_conditions(cfg, source)
        map_actor_evidence()
    elif kind == "inspect_task_evidence_and_training_pair":
        from data_generation_methods.scripts.postprocess_conditioned_case_20261020c import run_postprocess
        config_path = _case() / "postprocess_config_v1.json"
        if not config_path.exists():
            return {"executed": False, "reason": "explicit_postprocess_configuration_required",
                    "kind": kind}
        post_config = _read(config_path)
        if post_config.get("video") != _bind(_case() / "state_conditioned_human_v1.mp4"):
            raise ValueError("Postprocess configuration binds a different generated video")
        result = run_postprocess(config_path)
        return {
            "executed": True, "kind": kind,
            "handler": "postprocess_conditioned_case_20261020c.run_postprocess",
            "state": result["state"], "execution_record": result["execution_record"],
            "new_paid_review_calls": result["new_paid_review_calls"],
            "visible_problem_queries": result.get("visible_problem_queries", []),
            "unresolved_queries": result.get("unresolved_queries", []),
            "pair_validation": "not_run", "quality_release_approved": False,
            "training_admitted": False,
        }
    else:
        gate = _read(_case() / "condition_review_v1/decision.json")
        if not gate["ready_for_conditioned_generation"]:
            return {"executed": False, "reason": "condition_gate_unresolved", "kind": kind,
                    "issues": gate["issues"]}
        record = _read(_dispatch() / "condition_transport_preparation.json")
        if record["payload_sha256"] != json_sha256(runner.build_video_payload(cfg)):
            raise ValueError("Actual submission payload differs from prepared payload")
        runner.video(cfg, poll_minutes=poll_minutes)
    return {"executed": True, "kind": kind, "handler": {
        "extract_reveal_or_terminal_anchor": "runner.edit_first(cache) + runner.edit_last",
        "verify_multi_anchor_condition_transport": "prepare_transport(local_only)",
        "review_anchor_body_global": "runner.review_conditions + map_actor_evidence",
        "submit_verified_conditioned_video": "runner.video",
    }[kind], "new_paid_stages_expected": needs_new,
        "actor_review_scope": "global" if kind == "review_anchor_body_global" else None,
        "training_admitted": False}


def append_journal(record):
    _dispatch().mkdir(parents=True, exist_ok=True)
    with (_dispatch() / "execution_journal.jsonl").open("a") as stream:
        stream.write(json.dumps({"time_ns": time.time_ns(), **record}, ensure_ascii=False) + "\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())


def record_status(status):
    """Immutable snapshots plus one explicitly mutable atomic latest pointer."""
    latest = _dispatch() / "STATUS.json"
    snapshots = _dispatch() / "status_snapshots"
    if latest.exists():
        previous = _read(latest)
        write_json(snapshots / f"previous_{json_sha256(previous)}.json", previous)
    write_json(snapshots / f"{time.time_ns()}_{json_sha256(status)[:16]}.json", status)
    write_json(latest, status, overwrite=True)


@contextmanager
def execution_lock():
    _dispatch().mkdir(parents=True, exist_ok=True)
    with (_dispatch() / "execution.lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(cfg, source, *, execute=False, max_actions=6, max_total_paid_stages=4, poll_minutes=25):
    for index in range(max_actions):
        facts = build_facts(source, max_total_paid_stages=max_total_paid_stages)
        plan = plan_repairs(facts)
        stamp = str(time.time_ns())
        write_json(_dispatch() / "plans" / f"{stamp}_facts.json", facts)
        write_json(_dispatch() / "plans" / f"{stamp}_plan.json", plan)
        action = planned_action(plan)
        status = {"router_state": plan["state"], "next_action": action,
                  "mode": "execute" if execute else "plan_only", "iteration": index,
                  "training_admitted": False, "quality_release_approved": False}
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if not execute or action is None:
            record_status(status)
            return status
        before = started_stages()
        append_journal({"phase": "action_started", "action": action,
                        "plan": _bind(_dispatch() / "plans" / f"{stamp}_plan.json"),
                        "paid_stages_before": before, "action_executed": False})
        try:
            outcome = execute_action(action, cfg, source,
                                     max_total_paid_stages=max_total_paid_stages,
                                     poll_minutes=poll_minutes)
        except Exception as error:
            append_journal({"phase": "action_failed", "kind": action["kind"],
                            "error_type": type(error).__name__, "message": str(error),
                            "paid_stages_before": before, "paid_stages_after": started_stages(),
                            "action_executed": False})
            raise
        append_journal({"phase": "action_completed", "outcome": outcome,
                        "paid_stages_before": before, "paid_stages_after": started_stages(),
                        "action_executed": outcome["executed"]})
        if (not outcome["executed"] or action["kind"] in {
                "submit_verified_conditioned_video", "inspect_task_evidence_and_training_pair"}):
            status["outcome"] = outcome
            if action["kind"] == "inspect_task_evidence_and_training_pair" and outcome["executed"]:
                status.update(router_state=outcome["state"], next_action=None,
                              completed_action=action["kind"])
            record_status(status)
            return status
    status = {"mode": "execute", "stopped": "max_actions_reached", "training_admitted": False}
    record_status(status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--max-total-paid-stages", type=int, default=4)
    parser.add_argument("--poll-minutes", type=float, default=25)
    args = parser.parse_args(argv)
    if args.max_actions < 1 or args.max_total_paid_stages < 0 or args.poll_minutes <= 0:
        parser.error("Invalid bounded execution limits")
    cfg = _read(runner.DEFAULT_CONFIG)
    source = _read(_case() / "source_manifest.json")
    if args.execute:
        load_env_file(runner.ENV)
        with execution_lock():
            run(cfg, source, execute=True, max_actions=args.max_actions,
                max_total_paid_stages=args.max_total_paid_stages, poll_minutes=args.poll_minutes)
    else:
        run(cfg, source, execute=False, max_actions=1,
            max_total_paid_stages=args.max_total_paid_stages, poll_minutes=args.poll_minutes)


if __name__ == "__main__":
    main()
