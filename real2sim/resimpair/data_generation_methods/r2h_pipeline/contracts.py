"""Explicit task contracts for the existing three-block access recordings.

This module translates recorded phase metadata, not inferred human ground truth.
Canonical goals and expected events are private verifier inputs.  Blind review
gets the same three alternatives for every job and no source locators.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


PROGRAMS = (
    "retrieve_middle_blue_restore",
    "retrieve_bottom_brown_restore",
    "reverse_stack",
)
GOAL_PREFIX = "factory:long_block_access_restore:"
INITIAL_HASH_KEYS = (
    "initial_physics_sha256", "initial_rgb_sha256",
    "reference_raw_initial_rgb_sha256", "initial_decoded_rgb_sha256",
)
BLIND_GOAL_ALIASES = {f"choice_{c}": GOAL_PREFIX + p for c, p in zip("abc", PROGRAMS)}
ROLE_IDS = ("block0", "block1", "block2", "platform", "brown_pad", "blue_pad")
PHASES = {
    "clear_top_to_brown": (2, "brown_pad", "Move the initially top block to the brown parking area."),
    "retrieve_middle_to_blue": (1, "blue_pad", "Take the initially middle block to the blue parking area."),
    "restore_top_on_bottom": (2, "block0", "Place the initially top block on the initially bottom block."),
    "clear_middle_to_blue": (1, "blue_pad", "Move the initially middle block to the blue parking area to clear access."),
    "retrieve_bottom_to_brown": (0, "brown_pad", "Take the initially bottom block to the brown parking area."),
    "restore_middle_as_base": (1, "platform", "Return the initially middle block to the assembly platform as the base."),
    "restore_top_on_middle": (2, "block1", "Place the initially top block on the initially middle block."),
    "rebuild_top_as_base": (2, "platform", "Use the initially top block as the new base on the platform."),
    "rebuild_middle": (1, "block2", "Place the initially middle block on the initially top block."),
    "rebuild_bottom_as_top": (0, "block1", "Place the initially bottom block on top to complete the reversed stack."),
}
PROGRAM_PHASES = {
    PROGRAMS[0]: ("clear_top_to_brown", "retrieve_middle_to_blue", "restore_top_on_bottom"),
    PROGRAMS[1]: ("clear_top_to_brown", "clear_middle_to_blue", "retrieve_bottom_to_brown",
                  "restore_middle_as_base", "restore_top_on_middle"),
    PROGRAMS[2]: ("clear_top_to_brown", "clear_middle_to_blue", "retrieve_bottom_to_brown",
                  "rebuild_top_as_base", "rebuild_middle", "rebuild_bottom_as_top"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return isinstance(value, list) and all(_finite(x) for x in value)


def build_contract(row: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict:
    """Translate exact known phase records; reject unsupported or missing steps."""
    program = receipt.get("program")
    _require(program in PROGRAMS, "Unsupported program; no invented task contract")
    _require(row.get("program") == program, "Row/receipt program conflict")
    goal = GOAL_PREFIX + program
    _require(row.get("goal_key") == receipt.get("goal_key") == goal, "Row/receipt goal conflict")
    _require(row.get("group") == receipt.get("group"), "Row/receipt group conflict")
    _require(row.get("split") == receipt.get("split"), "Row/receipt split conflict")
    n, fps = receipt.get("length"), receipt.get("fps")
    _require(isinstance(n, int) and n > 0, "Invalid recording length")
    _require(isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0, "Invalid FPS")
    phases, observed = receipt.get("phase_targets", []), receipt.get("events", [])
    expected = PROGRAM_PHASES[program]
    _require(tuple(p.get("name") for p in phases) == expected, "Missing/reordered task phase")
    _require(len(observed) == len(phases), "Missing physical milestone evidence")
    events, previous_end = [], 0
    for i, (phase, witness) in enumerate(zip(phases, observed), 1):
        name = phase["name"]
        object_index, destination, description = PHASES[name]
        _require(phase.get("object_index") == witness.get("object_index") == object_index,
                 "Phase object identity mismatch")
        _require(witness.get("phase") == name and witness.get("passed") is True,
                 "Missing/pending physical milestone")
        start, end = witness.get("first_frame"), witness.get("end_frame")
        _require(isinstance(start, int) and isinstance(end, int) and previous_end <= start < end <= n,
                 "Invalid or overlapping event time range")
        _require(_finite(phase.get("position")) and len(phase["position"]) == 3,
                 "Invalid metric phase target")
        rotation = phase.get("rotation")
        _require(_finite(rotation) and len(rotation) == 3 and all(len(r) == 3 for r in rotation),
                 "Invalid metric rotation")
        _require(isinstance(phase.get("required_clear_objects"), list)
                 and all(x in (0, 1, 2) for x in phase["required_clear_objects"]), "Invalid access dependencies")
        previous_end = end
        events.append({
            "id": f"e{i:02d}", "phase_name": name, "role": f"block{object_index}",
            "verb": "pick_and_place", "destination_role": destination,
            "description": description,
            "required_clear_roles": [f"block{x}" for x in phase["required_clear_objects"]],
            "first_frame": start, "end_frame_exclusive": end,
            "start_seconds": start / fps, "end_seconds": end / fps,
            "recorded_metric_target": phase,
            "source_milestone": witness,
        })
    required = [{k: e[k] for k in ("id", "role", "verb", "destination_role")} for e in events]
    contract = {
        "schema": "r2h_task_contract_v1", "task_id": "long_block_access_restore",
        "goal_id": goal, "program": program, "group_id": row["group"], "split": row["split"],
        "training_admitted": False, "window_scope": "full_task",
        "duration_policy": "full_task_no_prefix_crop",
        "alignment_scope": "task_goal_and_required_event_order_not_framewise",
        "equivalence_scope": "specified_source_program_and_role_destinations_not_all_valid_human_solutions",
        "roles": [
            {"id": "block0", "identity": "initially_bottom_block"},
            {"id": "block1", "identity": "initially_middle_block"},
            {"id": "block2", "identity": "initially_top_block"},
            {"id": "platform", "identity": "raised_assembly_platform"},
            {"id": "brown_pad", "identity": "brown_parking_area"},
            {"id": "blue_pad", "identity": "blue_parking_area"}],
        "initial_relations": ["supported_by:block2:block1", "supported_by:block1:block0",
                              "supported_by:block0:platform"],
        "initial_measured_object_states": receipt.get("initial_measured_object_states"),
        "initial_hashes": {k: receipt.get(k) for k in INITIAL_HASH_KEYS},
        "required_events": events,
        "event_order": [[a["id"], b["id"]] for a, b in zip(events, events[1:])],
        "non_target_invariants": (["block0_remains_on_platform"] if program == PROGRAMS[0] else []),
        "allowed_changes": ["robot_body_to_human_body", "natural_human_arm_kinematics"],
        "forbidden_changes": ["object_identity", "selected_object", "destination_role",
                              "required_event_order", "omit_required_phase", "prefix_only_output"],
        "human_wrist_3d_valid": False,
        "ground_truth_scope": "Recorded simulator phases/milestones; no measured human action or force labels.",
        "review_spec": {
            "schema": "r2h_review_spec_v1", "blind_goal_aliases": BLIND_GOAL_ALIASES,
            "required_events": required,
            "forbidden_events": ([{"role": "block0", "verb": "pick_and_place", "destination_role": "*"}]
                                 if program == PROGRAMS[0] else []),
            "event_order": [[a["id"], b["id"]] for a, b in zip(events, events[1:])],
            "allowed_roles": list(ROLE_IDS), "allowed_verbs": ["pick_and_place"],
            "expected_window_scope": "full_task"},
    }
    validate_contract(contract)
    return contract


def validate_contract(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema") == "r2h_task_contract_v1", "Unknown contract schema")
    _require(contract.get("training_admitted") is False, "Preparation cannot admit training data")
    _require(contract.get("program") in PROGRAMS, "Unknown program")
    _require(contract.get("goal_id") == GOAL_PREFIX + contract["program"], "Goal/program mismatch")
    _require(contract.get("window_scope") == "full_task"
             and contract.get("duration_policy") == "full_task_no_prefix_crop", "Incomplete task scope")
    _require(contract.get("human_wrist_3d_valid") is False, "Synthetic video has no measured human wrist GT")
    hashes = contract.get("initial_hashes", {})
    _require(all(isinstance(hashes.get(k), str) and len(hashes[k]) == 64
                 and all(c in "0123456789abcdef" for c in hashes[k]) for k in INITIAL_HASH_KEYS),
             "Missing/invalid initial state hashes")
    events = contract.get("required_events", [])
    _require(tuple(e.get("phase_name") for e in events) == PROGRAM_PHASES[contract["program"]],
             "Missing/reordered contract events")
    _require(contract.get("event_order") == [[a["id"], b["id"]] for a, b in zip(events, events[1:])],
             "Invalid event-order edges")
    spec = contract.get("review_spec", {})
    _require(spec.get("blind_goal_aliases") == BLIND_GOAL_ALIASES, "Blind alternatives must be identical across goals")
    _require(spec.get("required_events") == [{k: e[k] for k in ("id", "role", "verb", "destination_role")} for e in events],
             "Private review expectations disagree with contract")
    forbidden = ([{"role": "block0", "verb": "pick_and_place", "destination_role": "*"}]
                 if contract["program"] == PROGRAMS[0] else [])
    _require(spec.get("forbidden_events") == forbidden, "Private non-target invariant disagrees with contract")


def generation_prompt(contract: Mapping[str, Any]) -> str:
    validate_contract(contract)
    steps = "\n".join(f"{i}. {e['description']}" for i, e in enumerate(contract["required_events"], 1))
    return (
        "Create a complete human demonstration of the supplied robot manipulation recording.\n"
        "Initially three blocks are stacked: block0 at the bottom, block1 in the middle, block2 at the top. "
        "Track those initial identities throughout; never rename a block when its height changes.\n"
        "Required complete sequence:\n" + steps + "\n"
        "Preserve the selected objects, destinations, access dependencies, releases and final support relations. "
        "Replace the robot with plausible human hands/arms, while preserving task objects and their identities. "
        "Do not add arrows, text, answer labels or extra goal hints to the scene.\n"
        "Show the initial state and the complete final state; do not truncate to an early prefix to fit a duration limit. "
        "If the backend cannot represent the complete task, report unsupported instead of silently dropping phases. "
        "Do not claim human 3D wrist, contact force or framewise action ground truth.\n"
    )


def blind_review_prompt() -> str:
    """Delegates to the independent review module when available."""
    from .review import blind_review_prompt as independent_prompt
    return independent_prompt()
