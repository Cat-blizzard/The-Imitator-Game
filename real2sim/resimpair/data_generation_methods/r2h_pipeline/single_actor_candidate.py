"""Opt-in single-operator candidate; no generation or training on import.

The plan is a rendering constraint, not visual evidence. An independent v2
anchor review is mandatory before preparing AND submitting a video. This
runner reuses the existing provider clients and durable paid-stage receipts.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import io
import json
from pathlib import Path

from .batch_pipeline import binding, checked_binding
from .common import file_sha256, json_sha256, load_env_file, read_json, write_json
from .controlled_experiment import budgeted_paid, call_counts
from .providers import GeminiClient, SeedanceClient
from .task_evidence import validate_contract

METHOD = "r2h_single_actor_candidate_v2"
PLAN_SCHEMA = "r2h_single_actor_plan_v2"


def validate_plan(plan):
    """Reject internally inconsistent ownership and incomplete task assignment."""
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("A pre-generation single-actor v2 plan is required")
    if (plan.get("maximum_visible_people") != 1
            or type(plan.get("maximum_visible_people")) is not int
            or plan.get("maximum_visible_hands") != 2
            or type(plan.get("maximum_visible_hands")) is not int):
        raise ValueError("This candidate requires exactly one operator and at most two hands")
    checked_binding(plan["source_image"])
    checked_binding(plan["task_contract"])
    contract = read_json(plan["task_contract"]["path"])
    validate_contract(contract, verify_files=True)
    actor = plan["actor"]
    if not isinstance(actor.get("id"), str) or not actor["id"].strip():
        raise ValueError("An explicit operator identity is required")
    position = actor.get("body_position", {})
    if (position.get("table_side") not in {"near", "far", "left", "right"}
            or any(not isinstance(position.get(k), str) or not position[k].strip()
                   for k in ("image_region", "facing"))):
        raise ValueError("An explicit body position and facing direction are required")
    hands = actor.get("hands", {})
    if set(hands) != {"left", "right"}:
        raise ValueError("Both anatomical hands must be assigned, including a resting hand")
    for side, hand in hands.items():
        if hand.get("owner_id") != actor["id"] or hand.get("anatomical_side") != side:
            raise ValueError("Both hands must have explicit ownership by the same operator")
        if any(not isinstance(hand.get(k), str) or not hand[k].strip()
               for k in ("initial_entry_region", "initial_role")):
            raise ValueError("Each hand needs its planned entry region and initial role")
    required = [event["id"] for event in contract["required_events"]]
    assignments = plan.get("event_assignments", [])
    if len(assignments) != len(required) or {a["event_id"] for a in assignments} != set(required):
        raise ValueError("Every required task event must have exactly one actor assignment")
    for assignment in assignments:
        if (assignment.get("actor_id") != actor["id"]
                or not isinstance(assignment.get("hands"), list)
                or not assignment["hands"]
                or len(set(assignment["hands"])) != len(assignment["hands"])
                or not set(assignment["hands"]) <= {"left", "right"}
                or not isinstance(assignment.get("role"), str)
                or not assignment["role"].strip()):
            raise ValueError("Event ownership, anatomical hands and operation role are required")
    if plan.get("allow_hand_entry_relocation") is not True:
        raise ValueError("Robot image-edge positions must not constrain human hand ownership")
    lineage = plan.get("lineage", {})
    if any(not isinstance(lineage.get(k), str) or not lineage[k].strip()
           for k in ("human_source_id", "physical_execution_id", "split")):
        raise ValueError("Source, execution and split lineage are required")
    if lineage["split"] != contract["source_window"]["split"]:
        raise ValueError("Derivatives must preserve their real-human source split")
    return contract


def create_plan(source_image, contract_path, placement, lineage):
    """Placement is an explicit design choice; this function does not infer a pose."""
    contract = read_json(contract_path)
    plan = {
        "schema": PLAN_SCHEMA, "source_image": binding(source_image),
        "task_contract": binding(contract_path), "actor": placement["actor"],
        "event_assignments": placement["event_assignments"],
        "lineage": lineage, "maximum_visible_people": 1, "maximum_visible_hands": 2,
        "allow_hand_entry_relocation": True,
        "status": "declared_structure_not_visually_verified",
        "source_task_id": contract["task_id"],
        "quality_release_approved": False, "training_admitted": False,
    }
    validate_plan(plan)
    return plan


def _structure_prompt(plan):
    return (
        "Use ONE human operator whose body, left arm and right arm form one coherent "
        "person. Recompose human arm entry positions according to the supplied body "
        "placement; the two robot bases do not represent two people. Do not retain "
        "robot entry edges when they would imply separate people. Keep enough visible "
        "arm-to-body connection to establish ownership without blocking task objects. "
        "Anatomical left and right refer to the operator, not the viewer. "
        + json.dumps({"actor": plan["actor"], "event_assignments": plan["event_assignments"]},
                     ensure_ascii=False)
    )


def prepare_anchor(plan_path, output, config):
    plan = read_json(plan_path)
    contract = validate_plan(plan)
    prompt = (
        "Edit this simulation into the INITIAL human task frame. " + _structure_prompt(plan)
        + " Hands begin resting away from task objects; do not perform the task yet. "
        "Preserve task objects, identities, relative sizes, initial relations, openings, "
        "handles, hidden contents, camera and background. Body position and human hand "
        "entry regions may change as explicitly planned. No text or annotations.\n"
        + json.dumps({"entity_roles": contract["entity_roles"],
                      "initial_relations": contract["initial_relations"]}, ensure_ascii=False)
    )
    request = {
        "schema": METHOD, "stage": "anchor", "plan": binding(plan_path),
        "source_image": plan["source_image"], "prompt": prompt,
        "model": config["anchor_model"], "new_paid_calls": 0,
        "quality_release_approved": False, "training_admitted": False,
    }
    write_json(Path(output) / "anchor_request.json", request)
    return request


def _check_anchor_request(directory, config):
    request = read_json(Path(directory) / "anchor_request.json")
    if request.get("schema") != METHOD or request.get("stage") != "anchor":
        raise ValueError("Not a single-actor candidate anchor request")
    checked_binding(request["plan"])
    plan = read_json(request["plan"]["path"])
    validate_plan(plan)
    if request["source_image"] != plan["source_image"]:
        raise ValueError("Anchor source differs from its pre-generation plan")
    # Recompute all expected request content so a changed prompt cannot silently run.
    import tempfile
    with tempfile.TemporaryDirectory() as check_dir:
        expected = prepare_anchor(request["plan"]["path"], check_dir, config)
    if request != expected:
        raise ValueError("Prepared anchor request changed")
    return request, plan


def _video_request(directory, anchor_path, review_path, config):
    from .single_actor_acceptance import gate_anchor
    request, plan = _check_anchor_request(directory, config)
    decision = gate_anchor(review_path, anchor_path, request["plan"]["path"])
    if not decision["can_generate_video"]:
        return None, decision
    contract = validate_plan(plan)
    prompt = (
        "One continuous fixed-camera shot starting from the supplied INITIAL frame. "
        + _structure_prompt(plan)
        + " Preserve the source task, object roles and necessary event order below. "
        "Use visible reach, contact, manipulation, release and withdrawal. Do not add "
        "an operator, cuts, teleportation, duplication or an extra task. Human grasp "
        "and hand entry choices may differ from the robot while preserving the task.\n"
        + json.dumps({k: contract[k] for k in (
            "entity_roles", "initial_relations", "final_relations", "required_events",
            "precedence", "forbidden_events")}, ensure_ascii=False)
    )
    # Existing provider validation runs locally and does not issue a request.
    from .cli import image_uri
    payload = {
        "model": config["seedance"]["model"], "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "role": "first_frame",
             "image_url": {"url": image_uri(anchor_path)}}],
        "duration": 8, "ratio": "adaptive", "resolution": "720p", "generate_audio": False,
    }
    SeedanceClient(config["seedance"]).prepare_submit(payload)
    return {
        "schema": METHOD, "stage": "video", "plan": request["plan"],
        "anchor_request": binding(Path(directory) / "anchor_request.json"),
        "anchor": binding(anchor_path), "independent_review": binding(review_path),
        "acceptance_rule_version": decision["rule_version"],
        "payload": payload, "payload_sha256": json_sha256(payload),
        "gate_scope": "static_single_actor_only_not_video_or_task_quality_release",
        "quality_release_approved": False, "training_admitted": False,
    }, decision


def prepare_video(directory, anchor_path, review_path, config):
    request, decision = _video_request(directory, anchor_path, review_path, config)
    if request is None:
        return {"prepared": False, "decision": decision, "new_paid_calls": 0}
    write_json(Path(directory) / "video_request.json", request)
    return {"prepared": True, "request": binding(Path(directory) / "video_request.json"),
            "decision": decision, "new_paid_calls": 0}


def _budgeted_call(budget_path, kind, metadata, operation):
    """One campaign ledger across all candidate directories; conservatively count POSTs."""
    budget_path = Path(budget_path).resolve(strict=True)
    budget = read_json(budget_path)
    limits = budget.get("cumulative_call_limits", {})
    if (budget.get("schema") != "r2h_single_actor_budget_v1"
            or set(limits) != {"image", "video", "review"}
            or any(type(v) is not int or v < 0 for v in limits.values())
            or type(budget.get("max_concurrent")) is not int
            or budget.get("max_concurrent") != 1
            or type(budget.get("timeout_seconds")) is not int
            or not 1 <= budget["timeout_seconds"] <= 600):
        raise ValueError("Explicit cumulative call limits, concurrency=1 and timeout are required")
    ledger = budget_path.parent / (budget_path.stem + "_ledger")
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Frozen per campaign, so a new output directory never resets its allowance.
        write_json(ledger / "budget.json", budget)
        stage = kind + "_" + json_sha256(metadata)[:32]
        stage_directory = ledger / "stages" / stage
        # An async accepted video is still in flight after its POST returns.
        # Unknown requests remain occupied until explicitly reconciled.
        if not (stage_directory / "call_started.json").exists():
            for started in (ledger / "stages").glob("*/call_started.json"):
                old_result = started.parent / "result.json"
                if not old_result.exists():
                    raise RuntimeError("Campaign has an unresolved submission; recover it before new paid calls")
                old_meta = read_json(started.parent / "request_metadata.json")
                if (old_meta["experiment_call_kind"] == "video"
                        and not (started.parent / "video_terminal.json").exists()):
                    raise RuntimeError("Campaign has an in-flight video; poll it before new paid calls")
        if (not (ledger / "stages" / stage / "call_started.json").exists()
                and call_counts(ledger)[kind] >= limits[kind]):
            raise RuntimeError(f"{kind} cumulative budget exhausted; no paid call")
        return budgeted_paid(ledger, stage, {
            **metadata, "experiment_call_kind": kind, "budget": binding(budget_path),
        }, operation, limits)


def execute_anchor(directory, config, budget_path, *, credentials_path=None):
    from PIL import Image
    request, _ = _check_anchor_request(directory, config)
    timeout = read_json(budget_path)["timeout_seconds"]
    def operation(transport):
        load_env_file(credentials_path)
        return GeminiClient(dict(config["gemini"], timeout=timeout), transport=transport).generate_image(
            Path(request["source_image"]["path"]), request["prompt"], model=request["model"])
    result = _budgeted_call(budget_path, "image", {
        "method": METHOD, "request": binding(Path(directory) / "anchor_request.json"),
    }, operation)
    image = Image.open(io.BytesIO(base64.b64decode(result["images"][0]["data"], validate=True)))
    image.load()
    image_path = Path(directory) / "human_initial.png"
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    if image_path.exists():
        if image_path.read_bytes() != encoded.getvalue():
            raise ValueError("Existing candidate anchor differs from saved provider result")
    else:
        with image_path.open("xb") as stream:
            stream.write(encoded.getvalue())
    return {"anchor": binding(image_path), "status": "needs_independent_v2_anchor_review",
            "quality_release_approved": False, "training_admitted": False}


def execute_video(directory, config, budget_path, *, credentials_path=None):
    request_path = Path(directory) / "video_request.json"
    request = read_json(request_path)
    for key in ("anchor", "plan", "anchor_request", "independent_review"):
        checked_binding(request[key])
    expected, decision = _video_request(directory, request["anchor"]["path"],
                                       request["independent_review"]["path"], config)
    if expected is None or request != expected:
        raise ValueError("Independent anchor gate no longer supports this exact video request")
    timeout = read_json(budget_path)["timeout_seconds"]
    def operation(transport):
        load_env_file(credentials_path)
        return SeedanceClient(dict(config["seedance"], timeout=timeout),
                              transport=transport).submit(request["payload"])
    metadata = {
        "method": METHOD, "request": binding(request_path),
        "anchor_gate_rule": decision["rule_version"],
    }
    result = _budgeted_call(budget_path, "video", metadata, operation)
    write_json(Path(directory) / "video_submission.json", result)
    budget_path = Path(budget_path).resolve(strict=True)
    ledger = budget_path.parent / (budget_path.stem + "_ledger")
    stage = ledger / "stages" / ("video_" + json_sha256(metadata)[:32])
    write_json(Path(directory) / "video_submission_origin.json", {
        "budget": binding(budget_path), "request": binding(request_path),
        "stage_metadata": binding(stage / "request_metadata.json"),
        "stage_result": binding(stage / "result.json"),
    })
    return result


def poll_video(directory, config, *, credentials_path=None):
    """Only GET an accepted task; persist terminal state so campaign slots can reopen."""
    import time
    from .cli import download_video
    directory = Path(directory)
    origin = read_json(directory / "video_submission_origin.json")
    for value in origin.values():
        checked_binding(value)
    submitted = read_json(origin["stage_result"]["path"])
    budget = read_json(origin["budget"]["path"])
    stage = Path(origin["stage_result"]["path"]).parent
    if (stage / "video_terminal.json").exists():
        result = read_json(stage / "video_terminal.json")
    else:
        load_env_file(credentials_path)
        result = SeedanceClient(dict(config["seedance"], timeout=min(
            30, budget["timeout_seconds"]))).poll(submitted["task_id"])
        write_json(directory / "polls" / f"{time.time_ns()}.json", result)
        if result["status"] in {"succeeded", "failed", "expired", "cancelled"}:
            write_json(stage / "video_terminal.json", result)
    if result["status"] == "succeeded":
        download_video(result["output_url"], directory / "candidate.mp4",
                       task_id=submitted["task_id"],
                       request_sha256=origin["request"]["sha256"])
    return {"status": result["status"], "task_id": submitted["task_id"],
            "new_paid_calls": 0, "quality_release_approved": False, "training_admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-anchor", "generate-anchor",
                                            "prepare-video", "submit-video", "poll-video"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--credentials", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.command == "prepare-anchor":
        if not args.plan:
            parser.error("--plan is required before anchor preparation")
        result = prepare_anchor(args.plan, args.output, config)
    elif args.command == "prepare-video":
        if not args.anchor or not args.review:
            parser.error("--anchor and independent --review are required")
        result = prepare_video(args.output, args.anchor, args.review, config)
    elif args.command == "poll-video":
        result = poll_video(args.output, config, credentials_path=args.credentials)
    else:
        if not args.budget:
            parser.error("An explicit cumulative --budget is required for paid stages")
        fn = execute_anchor if args.command == "generate-anchor" else execute_video
        result = fn(args.output, config, args.budget, credentials_path=args.credentials)
    # Never emit inline images, provider payloads or provider raw results.
    status = result.get("status")
    if status is None:
        status = (result["decision"]["status"] if result.get("prepared") is False
                  else "submitted" if args.command == "submit-video" else "prepared")
    print(json.dumps({"command": args.command, "output": str(args.output),
                      "status": status,
                      "prepared": result.get("prepared"),
                      "new_paid_calls": 0 if args.command.startswith(("prepare-", "poll-"))
                      else "see_campaign_ledger",
                      "training_admitted": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
