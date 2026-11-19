"""Opt-in apple anchor generation with verified source, state, camera and actor lineage.

This candidate leaves the previous actor runner and independent acceptance rules
unchanged. Preparation and execution revalidate the same immutable package.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import fcntl
import io
import json
from pathlib import Path
import re
import tempfile

from .batch_pipeline import binding, checked_binding
from .common import file_sha256, json_sha256, load_env_file, probe_video, read_json, write_json
from .providers import GeminiClient, SeedanceClient
from .single_actor_candidate import create_plan
from .controlled_experiment import budgeted_paid, call_counts
from .task_evidence import build_apple_contract

SCHEMA = "r2h_unified_anchor_package_v1"
CAMPAIGN_REGISTRY = Path("/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v2/campaign_registry")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(bound):
    checked_binding(bound)
    return read_json(bound["path"])


def _budget_spec(budget_path):
    budget = read_json(budget_path)
    _require(set(budget) == {"schema", "campaign_id", "ledger_path", "cumulative_call_limits",
                             "max_concurrent", "timeout_seconds"}
             and budget["schema"] == "r2h_unified_budget_v1", "Explicit unified campaign budget is required")
    _require(isinstance(budget["campaign_id"], str)
             and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", budget["campaign_id"]) is not None,
             "A stable simple campaign_id is required")
    ledger = Path(budget["ledger_path"])
    _require(ledger.is_absolute() and str(ledger.resolve()) == budget["ledger_path"],
             "Campaign ledger_path must be a canonical absolute path")
    limits = budget["cumulative_call_limits"]
    _require(set(limits) == {"image", "video", "review"}
             and all(type(v) is int and v >= 0 for v in limits.values())
             and type(budget["max_concurrent"]) is int and budget["max_concurrent"] == 1
             and type(budget["timeout_seconds"]) is int and 1 <= budget["timeout_seconds"] <= 600,
             "Explicit nonnegative cumulative limits, concurrency=1 and timeout are required")
    return budget, ledger


def _campaign_budgeted_call(budget_path, kind, metadata, operation):
    """Fixed registry prevents resetting one allowance by copying or relocating config."""
    budget, ledger = _budget_spec(budget_path)
    _require(kind in budget["cumulative_call_limits"], "Unknown campaign call kind")
    CAMPAIGN_REGISTRY.mkdir(parents=True, exist_ok=True)
    with (CAMPAIGN_REGISTRY / ".registration.lock").open("a+") as registration_lock:
        fcntl.flock(registration_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = CAMPAIGN_REGISTRY / (budget["campaign_id"] + ".json")
        if registration.exists():
            _require(read_json(registration) == budget,
                     "Campaign ID already binds a different ledger or allowance")
        else:
            write_json(registration, budget)
        ledger.mkdir(parents=True, exist_ok=True)
        # The canonical budget path is independent of the submitted config path.
        write_json(ledger / "budget.json", budget)
    with (ledger / ".execution.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage = kind + "_" + json_sha256(metadata)[:32]
        stage_dir = ledger / "stages" / stage
        if not (stage_dir / "call_started.json").exists():
            for marker in (ledger / "stages").glob("*/call_started.json"):
                if not (marker.parent / "result.json").exists():
                    raise RuntimeError("Campaign has an unresolved submission; recover before new paid calls")
                old_meta = read_json(marker.parent / "request_metadata.json")
                if (old_meta["experiment_call_kind"] == "video"
                        and not (marker.parent / "video_terminal.json").exists()):
                    raise RuntimeError("Campaign has an in-flight video; query existing task before a new call")
            if call_counts(ledger)[kind] >= budget["cumulative_call_limits"][kind]:
                raise RuntimeError(f"{kind} cumulative campaign budget exhausted; no paid call")
        return budgeted_paid(ledger, stage, {**metadata, "experiment_call_kind": kind,
                              "budget": binding(ledger / "budget.json"),
                              "campaign_id": budget["campaign_id"]}, operation,
                             budget["cumulative_call_limits"])


def _normalize_packet(packet):
    result = deepcopy(packet)
    for condition in result["conditions"]:
        # Recompilation has a different output directory, but exact frame bytes,
        # decoded RGB, selected video and frame/time indices must still agree.
        condition["frame"].pop("path")
    return result


def verify_sources(condition_path, input_path):
    """Recompute sim conditions and trace the execution's exact reviewed human window."""
    from .sim_state_conditions import SCHEMA as CONDITION_SCHEMA, compile_conditions
    from ..human_to_robot_pipeline.planning_contract import validate_contract
    packet, execution_input = read_json(condition_path), read_json(input_path)
    _require(packet.get("schema") == CONDITION_SCHEMA
             and packet.get("status") == "prepared_for_condition_image_review",
             "A qualified simulation condition packet is required")
    for bound in packet["bindings"].values():
        checked_binding(bound)
    receipt = _read(packet["bindings"]["receipt"])
    proposal, motion_contract = execution_input["proposal"], execution_input["contract"]
    _require(execution_input.get("execution_id") == receipt.get("execution_id")
             == packet.get("execution_id"), "Execution input, receipt and conditions disagree")
    _require(proposal == receipt["randomization"]["structured_proposal"],
             "Execution proposal differs from the proposal embedded in the receipt")
    _require(proposal["seed"] == receipt["layout_seed"]
             and proposal["split"] == receipt["split"] == packet["source_data_split"],
             "Execution seed or source split differs")
    validate_contract(motion_contract, verify_files=True)
    source = proposal["human_source_window"]
    matches = [w for w in motion_contract["human_windows"] if w["id"] == source["id"]]
    _require(len(matches) == 1 and matches[0] == source,
             "Execution human window is not the exact reviewed contract member")
    _require(proposal["human_window_id"] == source["id"]
             and proposal["human_video"] == source["video"]
             and source["split"] == receipt["split"], "Proposal human identity/video/split mismatch")
    for key in ("video", "review", "actual_review_sheet"):
        checked_binding(source[key])
    semantic = _read(source["review"])
    observations = [x for x in semantic["episodes"] if x["id"] == source["id"]]
    _require(len(observations) == 1, "Human semantic review does not identify this window")
    observation = observations[0]
    _require(all(observation.get(k) == source[k] for k in ("id", "session", "split"))
             and observation.get("source_sha256") == source["video"]["sha256"]
             and Path(observation["native_video"]).resolve() == Path(source["video"]["path"]).resolve()
             and observation.get("review_sheet_sha256") == source["actual_review_sheet"]["sha256"]
             and Path(observation["review_sheet"]).resolve() == Path(source["actual_review_sheet"]["path"]).resolve()
             and observation.get("reviewed") is True
             and observation.get("reference_operation_candidate_approved") is True,
             "Human visual review is absent or belongs to another source/session")
    human_media = probe_video(source["video"]["path"])
    start, end = source["window_frames"]
    _require(type(start) is int and type(end) is int and 0 <= start < end <= human_media["frames"]
             and abs(source["fps"] - human_media["fps"]) < 1e-8
             and source["interval"] == "start_inclusive_end_exclusive",
             "Human window is outside the actual decoded media or uses another clock")
    _require(semantic["goal_key"] == receipt["goal_key"], "Human review and robot goal differ")
    for condition in packet["conditions"]:
        checked_binding(condition["frame"])
        checked_binding(condition["frame"]["video"])
    with tempfile.TemporaryDirectory(prefix="r2h-unified-recheck-") as tmp:
        recomputed = compile_conditions(
            packet["bindings"]["receipt"]["path"], packet["bindings"]["verification"]["path"],
            packet["selected_camera"], Path(tmp) / "conditions",
            partition=packet["experiment_partition"])
        _require(recomputed["status"] == "prepared_for_condition_image_review",
                 "Fresh simulation evidence verification failed: " + recomputed.get("reason", ""))
        _require(_normalize_packet(recomputed) == _normalize_packet(packet),
                 "Condition roles, selected camera, frames or states differ from fresh evidence")
    initial = [c for c in packet["conditions"] if "initial" in c["labels"]]
    _require(len(initial) == 1 and initial[0]["frame"]["frame_index_0based"] == 0,
             "Exactly one actual initial frame is required")
    roles = packet["roles"]
    expected_roles = {
        "selected_object": f"{receipt['execution_id']}/object/{proposal['active_arm']}",
        "source_container": f"{receipt['execution_id']}/container/{receipt['randomization']['source_bowl_index']}",
        "destination_container": f"{receipt['execution_id']}/container/{receipt['randomization']['destination_bowl_index']}",
    }
    _require({k: v["id"] for k, v in roles.items()} == expected_roles,
             "Stable task roles differ from the executed object/container indices")
    context = {
        "condition_role": "initial", "condition_packet_sha256": file_sha256(condition_path),
        "reference_frame_sha256": initial[0]["frame"]["sha256"],
        "execution_id": receipt["execution_id"], "camera_id": packet["selected_camera"],
        "human_source_id": source["id"], "human_session_id": source["session"],
        "human_video_sha256": source["video"]["sha256"],
        "human_window_frames": source["window_frames"], "human_fps": source["fps"],
        "human_split": source["split"], "human_window_interval": source["interval"],
        "role_ids": expected_roles,
    }
    return {"packet": packet, "receipt": receipt, "input": execution_input,
            "human_source": source, "context": context, "initial": initial[0]}


def _derived_contract(data):
    source = data["human_source"]
    contract = build_apple_contract(data["input"]["contract"]["human_spec"]["path"], source["id"])
    roles, camera_id = data["packet"]["roles"], data["packet"]["selected_camera"]
    order = data["packet"]["camera_role_projections"][camera_id]["relative_order"]
    contract["entity_roles"] = {
        "actor": "the sole planned human operator; both anatomical hands belong to this operator",
        "apple": "the same selected apple " + roles["selected_object"]["id"],
        "bowl_source": "the initially occupied bowl " + roles["source_container"]["id"]
                       + "; " + order["source_relative_to_destination"] + " in the selected camera",
        "bowl_target": "the unique initially empty bowl " + roles["destination_container"]["id"]
                       + "; " + order["destination_relative_to_source"] + " in the selected camera",
    }
    contract["simulation_role_ids"] = data["context"]["role_ids"]
    contract["selected_camera"] = camera_id
    contract["execution_id"] = data["receipt"]["execution_id"]
    return contract


def _validate_placement(placement):
    _require(set(placement) == {"actor", "event_assignments"}, "Placement must only declare actor and event assignments")
    for hand in placement["actor"]["hands"].values():
        _require(hand["initial_role"] in {"rest_clear_of_task_objects", "rest_visible_at_body"},
                 "Initial hand roles must be inactive; future manipulation belongs only to event assignments")


def _package_core(condition_path, input_path, placement_path, directory, data):
    return {
        "schema": SCHEMA,
        "inputs": {"conditions": binding(condition_path), "execution_input": binding(input_path),
                   "placement": binding(placement_path)},
        "task_contract": binding(directory / "task_contract.json"),
        "actor_plan": binding(directory / "actor_plan.json"),
        "initial_context": data["context"], "human_source": data["human_source"],
        "selected_camera_calibration_sha256": json_sha256(data["packet"]["camera_calibration"]),
        "lineage": {"human_source_id": data["human_source"]["id"],
                    "human_session_id": data["human_source"]["session"],
                    "physical_execution_id": data["receipt"]["execution_id"],
                    "split": data["human_source"]["split"],
                    "basis": "receipt structured proposal and reviewed human specification; not free text"},
        "initial_source_image": {k: data["initial"]["frame"][k] for k in ("path", "sha256")},
        "stage": "initial_anchor_only", "video_requires_independent_actor_and_state_gates": True,
        "quality_release_approved": False, "training_admitted": False,
    }


def validate_package(package_path):
    package_path = Path(package_path)
    package = read_json(package_path)
    _require(package.get("schema") == SCHEMA, "Not a unified anchor candidate package")
    core = {k: v for k, v in package.items() if k != "package_sha256"}
    _require(package.get("package_sha256") == json_sha256(core), "Unified package content hash changed")
    for value in package["inputs"].values():
        checked_binding(value)
    for key in ("actor_plan", "task_contract", "initial_source_image"):
        checked_binding(package[key])
    inputs = package["inputs"]
    data = verify_sources(inputs["conditions"]["path"], inputs["execution_input"]["path"])
    placement = _read(inputs["placement"])
    _validate_placement(placement)
    _require(_read(package["task_contract"]) == _derived_contract(data),
             "Task contract changed source, roles or required events")
    expected_plan = create_plan(
        data["initial"]["frame"]["path"], package["task_contract"]["path"], placement,
        {"human_source_id": data["human_source"]["id"],
         "physical_execution_id": data["receipt"]["execution_id"],
         "split": data["human_source"]["split"]})
    _require(_read(package["actor_plan"]) == expected_plan,
             "Actor plan source image, lineage or event assignment differs from this execution")
    expected = _package_core(inputs["conditions"]["path"], inputs["execution_input"]["path"],
                             inputs["placement"]["path"], package_path.parent, data)
    _require(core == expected, "Unified package source/camera/role contents changed")
    return {**data, "package": package, "package_path": package_path, "plan": expected_plan}


def _initial_request(package_path, data, config):
    packet, initial = data["packet"], data["initial"]
    roles = packet["roles"]
    order = packet["camera_role_projections"][packet["selected_camera"]]["relative_order"]
    # Do not serialize the full plan: its future event assignments are not
    # instructions for an initial image. No state_delta or terminal condition enters.
    conditions = {
        "stage": "initial", "camera_id": packet["selected_camera"],
        "operator": data["plan"]["actor"],
        "role_ids": data["context"]["role_ids"],
        "initial_relations": [
            {"relation": "inside", "object": roles["selected_object"]["id"],
             "container": roles["source_container"]["id"]},
            {"relation": "empty", "container": roles["destination_container"]["id"]}],
        "camera_relative_roles": {k: order[k] for k in (
            "source_relative_to_destination", "destination_relative_to_source")},
        "same_camera_relative_size_reference": initial["relative_size_reference"],
        "source_geometry_is_not_generated_pixel_acceptance": True,
    }
    prompt = (
        "Edit the supplied simulation INITIAL image into the initial frame of one human "
        "operator. Use the specified single body position; left and right hands belong to "
        "that same body. Human hand entry positions may differ from the two robot bases. "
        "Keep both hands inactive and clear of objects. Preserve this selected camera, "
        "all object instances, their relative size, their initial positions, and the "
        "occupied and empty bowl relations. Remove robot parts. Do not perform any "
        "manipulation or depict a completed task. Preserve nonselected objects and "
        "background. No text or annotations.\n" + json.dumps(conditions, ensure_ascii=False)
    )
    candidate = {}
    if config.get("condition_compiler") is not None:
        from . import condition_compiler_candidate as compiler
        _require(config["condition_compiler"] == compiler.VERSION, "Unknown condition compiler candidate")
        compiled = compiler.compile_unified(data, inactive_policy=config.get("inactive_hand_visibility", "retain_plan"))
        prompt = compiled["stages"]["initial"]["prompt"]
        candidate = {"condition_compiler": compiler.VERSION, "compiler_source": binding(compiler.__file__),
                     "inactive_hand_visibility": compiled["inactive_hand_visibility_policy"],
                     "compiled_stage": compiled["stages"]["initial"],
                     "prompt_stats": compiled["stages"]["initial"]["prompt_stats"]}
    return {"schema": "r2h_unified_initial_request_v1", "package": binding(package_path),
            "source_image": data["package"]["initial_source_image"], "model": config["anchor_model"],
            "initial_conditions": conditions, "prompt": prompt,
            **candidate,
            "training_admitted": False, "quality_release_approved": False}


def prepare_unified(condition_path, input_path, placement_path, output, config):
    output = Path(output).resolve()
    _require(not output.exists(), "Use a new package directory; previous packages are immutable")
    data = verify_sources(condition_path, input_path)
    placement = read_json(placement_path)
    _validate_placement(placement)
    output.mkdir(parents=True, exist_ok=False)
    contract = _derived_contract(data)
    write_json(output / "task_contract.json", contract)
    plan = create_plan(data["initial"]["frame"]["path"], output / "task_contract.json", placement,
                       {"human_source_id": data["human_source"]["id"],
                        "physical_execution_id": data["receipt"]["execution_id"],
                        "split": data["human_source"]["split"]})
    write_json(output / "actor_plan.json", plan)
    package = _package_core(condition_path, input_path, placement_path, output, data)
    package["package_sha256"] = json_sha256(package)
    write_json(output / "package.json", package)
    data.update(package=package, plan=plan)
    if config.get("condition_compiler") is not None:
        from .condition_compiler_candidate import compile_unified
        write_json(output / "compiled_conditions.json", compile_unified(
            data, inactive_policy=config.get("inactive_hand_visibility", "retain_plan")))
    write_json(output / "initial_request.json", _initial_request(output / "package.json", data, config))
    return {"package": binding(output / "package.json"), "request": binding(output / "initial_request.json"),
            "status": "prepared_initial_anchor_candidate", "new_paid_calls": 0}


def execute_initial(package_path, config, budget_path, *, credentials_path=None):
    """Real image provider adapter; gate and budget run before credentials or network."""
    from PIL import Image
    data = validate_package(package_path)
    directory = Path(package_path).parent
    request_path = directory / "initial_request.json"
    request = read_json(request_path)
    _require(request == _initial_request(package_path, data, config),
             "Prepared initial request changed source, camera, roles or conditions")
    static_geometry = request.get("compiled_stage", {}).get("static_actor_geometry")
    _require(not static_geometry or static_geometry["clears_geometry_preflight"],
             "Compiled actor geometry preflight blocked; no provider call")
    timeout = read_json(budget_path)["timeout_seconds"]
    def operation(transport):
        load_env_file(credentials_path)
        return GeminiClient(dict(config["gemini"], timeout=timeout), transport=transport).generate_image(
            Path(request["source_image"]["path"]), request["prompt"], model=request["model"])
    metadata = {"method": SCHEMA, "package": binding(package_path), "request": binding(request_path)}
    result = _campaign_budgeted_call(budget_path, "image", metadata, operation)
    generated = Image.open(io.BytesIO(base64.b64decode(result["images"][0]["data"], validate=True)))
    generated.load()
    raw = io.BytesIO()
    generated.save(raw, format="PNG")
    image_path = directory / "human_initial.png"
    if image_path.exists():
        _require(image_path.read_bytes() == raw.getvalue(), "Existing generated image differs from saved response")
    else:
        with image_path.open("xb") as stream:
            stream.write(raw.getvalue())
    write_json(directory / "image_generation_provenance.json", {
        "schema": "r2h_unified_image_generation_v1", "package": binding(package_path),
        "request": binding(request_path), "image": binding(image_path),
        "provider_result_sha256": json_sha256(result), "budget": binding(_budget_spec(budget_path)[1] / "budget.json"),
        **_stage_bindings(budget_path, "image", metadata),
        "status": "needs_independent_actor_and_initial_state_reviews",
        "quality_release_approved": False, "training_admitted": False})
    return {"image": binding(image_path), "status": "needs_independent_actor_and_initial_state_reviews"}


def _stage_bindings(budget_path, kind, metadata):
    budget, ledger = _budget_spec(budget_path)
    registration = CAMPAIGN_REGISTRY / (budget["campaign_id"] + ".json")
    _require(read_json(registration) == budget and read_json(ledger / "budget.json") == budget,
             "Campaign registry/ledger budget binding changed")
    stage = ledger / "stages" / (kind + "_" + json_sha256(metadata)[:32])
    return {"stage_metadata": binding(stage / "request_metadata.json"),
            "stage_result": binding(stage / "result.json")}


def _video_gate(data, image_path, actor_review_path=None, state_review_path=None):
    from .single_actor_acceptance import gate_anchor
    from .static_state_acceptance import gate_state
    package_path = data["package_path"]
    missing = []
    provenance_path = Path(package_path).parent / "image_generation_provenance.json"
    if not provenance_path.exists():
        missing.append("missing_generation_provenance_for_planned_anchor")
    else:
        provenance = read_json(provenance_path)
        for key in ("package", "request", "image", "budget", "stage_metadata", "stage_result"):
            checked_binding(provenance[key])
        _require(provenance["package"] == binding(package_path)
                 and provenance["image"] == binding(image_path)
                 and provenance["request"] == binding(Path(package_path).parent / "initial_request.json"),
                 "Generated anchor provenance belongs to another image or package")
        request = _read(provenance["request"])
        _require(request == _initial_request(package_path, data, {"anchor_model": request["model"],
                 **{key: request[key] for key in ("condition_compiler", "inactive_hand_visibility") if key in request}}),
                 "Generated anchor request is not the exact initial-only package request")
        metadata = {"method": SCHEMA, "package": binding(package_path), "request": provenance["request"]}
        expected_stage = _stage_bindings(provenance["budget"]["path"], "image", metadata)
        _require(all(provenance[k] == v for k, v in expected_stage.items()),
                 "Generated anchor does not bind its actual campaign stage")
        stage_meta, result = _read(provenance["stage_metadata"]), _read(provenance["stage_result"])
        _require(all(stage_meta.get(k) == v for k, v in metadata.items())
                 and provenance["provider_result_sha256"] == json_sha256(result),
                 "Generated anchor stage or provider response changed")
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(result["images"][0]["data"], validate=True))) as image:
            image.load()
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
        _require(Path(image_path).read_bytes() == encoded.getvalue(),
                 "Reviewed anchor pixels differ from the bound provider result")
    actor = gate_anchor(actor_review_path, image_path, data["package"]["actor_plan"]["path"]) \
        if actor_review_path else None
    state = gate_state(state_review_path, image_path, package_path, data["context"]) \
        if state_review_path else None
    if not actor or not actor["can_generate_video"]:
        missing.append("independent_single_actor_gate_not_passed")
    if not state or not state["clears_static_state"]:
        missing.append("independent_initial_state_gate_not_passed")
    return {"status": "ready_for_video_preparation" if not missing else "blocked",
            "can_generate_video": not missing, "blocking_reasons": missing,
            "actor_gate": actor, "state_gate": state, "new_paid_calls": 0,
            "quality_release_approved": False, "training_admitted": False}


def video_gate(package_path, image_path, actor_review_path=None, state_review_path=None):
    return _video_gate(validate_package(package_path), image_path, actor_review_path, state_review_path)


def _video_request(package_path, image_path, actor_review_path, state_review_path, config):
    from .cli import image_uri
    data = validate_package(package_path)
    gate = _video_gate(data, image_path, actor_review_path, state_review_path)
    if not gate["can_generate_video"]:
        return None, gate
    task = _read(data["package"]["task_contract"])
    prompt = (
        "Create one continuous fixed-camera human task video from this approved initial image. "
        "Maintain the same single operator and anatomical hand ownership. Keep exact object "
        "roles and camera-relative mapping. Show approach, contact, manipulation, release "
        "and withdrawal in the required order; keep non-task objects stationary. No added "
        "people, cuts, teleportation, duplication or second task.\n"
        + json.dumps({"actor_plan": data["plan"]["actor"],
                      "event_assignments": data["plan"]["event_assignments"],
                      "task": {k: task[k] for k in (
                          "entity_roles", "initial_relations", "final_relations",
                          "required_events", "precedence", "forbidden_events")}}, ensure_ascii=False)
    )
    candidate = {}
    if config.get("condition_compiler") is not None:
        from . import condition_compiler_candidate as compiler
        _require(config["condition_compiler"] == compiler.VERSION, "Unknown condition compiler candidate")
        compiled = compiler.compile_unified(data, inactive_policy=config.get("inactive_hand_visibility", "retain_plan"))
        _require(not compiled["static_actor_geometry"] or compiled["static_actor_geometry"]["clears_geometry_preflight"],
                 "Compiled actor geometry preflight blocked; no video request")
        prompt = compiled["stages"]["video"]["prompt"]
        candidate = {"condition_compiler": compiler.VERSION, "compiler_source": binding(compiler.__file__),
                     "compiled_stage": compiled["stages"]["video"],
                     "event_evidence": compiled["measured_event_sidecar"],
                     "inactive_hand_visibility": compiled["inactive_hand_visibility_policy"]}
    payload = {"model": config["seedance"]["model"], "content": [
        {"type": "text", "text": prompt}, {"type": "image_url", "role": "first_frame",
                                               "image_url": {"url": image_uri(image_path)}}],
        "duration": 8, "resolution": "720p", "ratio": "adaptive", "generate_audio": False}
    SeedanceClient(config["seedance"]).prepare_submit(payload)
    request = {"schema": "r2h_unified_video_request_v1", "package": binding(package_path),
               "image": binding(image_path), "actor_review": binding(actor_review_path),
               "state_review": binding(state_review_path),
               "generation_provenance": binding(Path(package_path).parent / "image_generation_provenance.json"),
               "actor_rule_version": gate["actor_gate"]["rule_version"],
               "state_rule_version": gate["state_gate"]["rule_version"],
               "payload": payload, **candidate, "quality_release_approved": False, "training_admitted": False}
    return request, gate


def prepare_video(package_path, image_path, actor_review_path, state_review_path, config):
    request, gate = _video_request(package_path, image_path, actor_review_path, state_review_path, config)
    if request is None:
        return gate
    target = Path(package_path).parent / "video_request.json"
    write_json(target, request)
    return {"status": "prepared_video_candidate", "request": binding(target), "new_paid_calls": 0,
            "quality_release_approved": False, "training_admitted": False}


def execute_video(package_path, config, budget_path, *, credentials_path=None):
    directory = Path(package_path).parent
    request_path = directory / "video_request.json"
    request = read_json(request_path)
    for key in ("package", "image", "actor_review", "state_review", "generation_provenance"):
        checked_binding(request[key])
    expected, gate = _video_request(package_path, request["image"]["path"],
                                    request["actor_review"]["path"], request["state_review"]["path"], config)
    _require(expected is not None and expected == request,
             "Prepared video no longer passes both independent gates with the exact package")
    metadata = {"method": SCHEMA, "package": binding(package_path), "request": binding(request_path),
                "actor_rule_version": gate["actor_gate"]["rule_version"],
                "state_rule_version": gate["state_gate"]["rule_version"]}
    timeout = read_json(budget_path)["timeout_seconds"]
    def operation(transport):
        load_env_file(credentials_path)
        return SeedanceClient(dict(config["seedance"], timeout=timeout), transport=transport).submit(request["payload"])
    result = _campaign_budgeted_call(budget_path, "video", metadata, operation)
    write_json(directory / "video_submission.json", result)
    write_json(directory / "video_submission_origin.json", {
        "budget": binding(_budget_spec(budget_path)[1] / "budget.json"), "request": binding(request_path),
        **_stage_bindings(budget_path, "video", metadata)})
    return {"status": "video_submitted", "task_id": result["task_id"], "training_admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute-initial", "video-gate", "validate",
                                            "prepare-video", "submit-video", "poll-video"))
    for key in ("conditions", "input", "placement", "output", "package", "config", "budget",
                "credentials", "image", "actor-review", "state-review"):
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_unified(args.conditions, args.input, args.placement, args.output, read_json(args.config))
    elif args.command == "execute-initial":
        result = execute_initial(args.package, read_json(args.config), args.budget,
                                 credentials_path=args.credentials)
    elif args.command == "video-gate":
        result = video_gate(args.package, args.image, args.actor_review, args.state_review)
    elif args.command == "prepare-video":
        result = prepare_video(args.package, args.image, args.actor_review, args.state_review, read_json(args.config))
    elif args.command == "submit-video":
        result = execute_video(args.package, read_json(args.config), args.budget, credentials_path=args.credentials)
    elif args.command == "poll-video":
        from .single_actor_candidate import poll_video
        result = poll_video(args.package.parent, read_json(args.config), credentials_path=args.credentials)
    else:
        data = validate_package(args.package)
        result = {"status": "package_bindings_verified", "context": data["context"], "new_paid_calls": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
