"""Select reusable evidence requirements from an existing task declaration.

No task-ID, scene-name, color, or benchmark-specific dispatch. This is an input
compiler, not a new visual acceptance gate. Native metric/force predicates stay
with native physics; candidate RGB needs independently localized observations.
"""
import math

from .core import require
from .event_protocol import ACTIONS, RELATIONS

VERSION = "declared_task_evidence_plan_v1"


def compile_plan(contract):
    info = contract["generation_information"]
    profiles, covered, unsupported = {}, {}, []
    inventory_ids = [x["id"] for x in info["inventory"]]
    require(len(inventory_ids) == len(set(inventory_ids)), "duplicate_inventory_identity")

    def add(name, declaration, entities=()):
        require(set(entities) <= set(inventory_ids), "evidence_entity_not_bound_to_inventory")
        row = profiles.setdefault(name, {"triggered_by": set(), "entities": set()})
        row["triggered_by"].add(declaration)
        row["entities"].update(entities)
        covered.setdefault(declaration, set()).add(name)

    add("actor_and_hand_identity", "common:actor")
    add("object_identity", "common:inventory", [x["id"] for x in info["inventory"]])
    events = info["required_events"]
    ids = [e["id"] for e in events]
    require(len(ids) == len(set(ids)), "duplicate_declared_event")
    for event in events:
        key, entities = "event:" + event["id"], event["entities"]
        action = event.get("action")
        if action not in ACTIONS:
            unsupported.append({"declaration": key, "reason": "unknown_event_action", "value": action})
            continue
        add("event_occurrence", key, entities)
        if event.get("human_hand") in {"left", "right"}:
            add("actor_and_hand_identity", key, entities)
        if action in {"grasp", "lift", "release", "supported", "withdraw"}:
            add("hand_object_state_transition", key, entities)
        if action in {"articulate", "trigger"} and event.get("joint_state_evidence"):
            add("relative_joint_motion", key, entities)
        if action == "passive_contact":
            add("contact_and_motion_order", key, entities)
        if action == "insert":
            add("functional_part_alignment", key, entities)
        if event.get("required_spatial_targets"):
            add("ordered_functional_path", key, entities)

    for section, prefix in [("initial_relations", "initial:"), ("final_relations", "goal:"), ("persistent_constraints", "constraint:")]:
        for declaration in info.get(section, []):
            key, relation = prefix + declaration["id"], declaration["relation"]
            if relation not in RELATIONS:
                unsupported.append({"declaration": key, "reason": "unknown_relation", "value": relation})
                continue
            entities = declaration["entities"]
            add("interval_constraints" if section == "persistent_constraints" else "endpoint_relations", key, entities)
            if relation in {"continuous_grasp", "held", "no_grasp"}:
                add("persistent_hand_roles", key, entities)
            if relation in {"contact", "no_contact"}:
                add("contact_and_motion_order", key, entities)
            if relation in {"inserted", "clearance"}:
                add("functional_part_alignment", key, entities)
            if relation == "joint_range":
                add("relative_joint_motion", key, entities)

    edges = info.get("required_event_precedence", [])
    require(all(len(e) == 2 and set(e) <= set(ids) for e in edges), "invalid_declared_order_edge")
    if edges:
        add("event_order", "common:declared_partial_order")
    cycle = info.get("required_cyclic_motion")
    if cycle:
        require(cycle.get("direction_in_image") in {"clockwise", "counterclockwise"}
                and type(cycle.get("turns")) in {int, float} and math.isfinite(cycle["turns"]) and cycle["turns"] > 0,
                "invalid_explicit_cycle_requirement")
        require(set(cycle.get("ordered_event_ids", [])) <= set(ids), "cycle_references_unknown_event")
        add("angular_motion_and_extra_cycles", "explicit:required_cyclic_motion",
            {entity for event in events if event["id"] in cycle["ordered_event_ids"] for entity in event["entities"]})

    # Exact cycles are added ONLY by a declared task requirement, not by a
    # closed robot path, a task name such as 'stir', or identical endpoints.
    rows = []
    for name, row in sorted(profiles.items()):
        local = name in {"relative_joint_motion", "ordered_functional_path", "angular_motion_and_extra_cycles", "functional_part_alignment"}
        rows.append({"type": name, "triggered_by": sorted(row["triggered_by"]), "entities": sorted(row["entities"]),
            "candidate_relocalization_required": local,
            "current_capability": "measurement_or_local_crop_candidate_not_calibrated"
                if local else "existing_vlm_observation_with_known_accuracy_limits",
            "failure_behavior": "unknown_if_required_candidate_evidence_is_missing"})
    process_types = {"relative_joint_motion", "ordered_functional_path", "angular_motion_and_extra_cycles",
                     "contact_and_motion_order", "persistent_hand_roles", "functional_part_alignment"}
    declared = {"event:" + e["id"] for e in events}
    declared |= {prefix + x["id"] for field, prefix in [("initial_relations", "initial:"), ("final_relations", "goal:"), ("persistent_constraints", "constraint:")] for x in info.get(field, [])}
    return {"schema": VERSION, "profiles": rows,
        "declared_requirement_coverage": {k: sorted(v) for k, v in sorted(covered.items()) if k in declared},
        "declaration_coverage_complete": not unsupported and declared <= set(covered),
        "unsupported_declarations": unsupported,
        "conditioning_study": {"baseline": "first_image_and_text",
            "motion_reference_hypothesis_relevant": bool(set(profiles) & process_types),
            "native_video_is_optional_process_evidence": True, "runtime_route_changed": False},
        "semantic_boundaries": {"source_correspondence": contract.get("source_mapping", {}),
            "constraints_come_from": "declared_task_contract_not_inferred_robot_motion",
            "robot_joint_paths_are_human_constraints": False, "suggested_timing_is_hard_constraint": False,
            "native_force_or_metric_thresholds_apply_to_human_rgb": False,
            "same_endpoint_proves_cycle_or_press": False,
            "native_projected_pixels_are_exact_human_pixels": False},
        "native_evidence_may_seed_relocalization": True,
        "visual_checks_executed": False, "visual_accuracy_established": False,
        "acceptance_rules_replaced": False, "training_admitted": False}
