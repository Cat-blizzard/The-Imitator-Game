"""Immutable task edits with original scene and source-request provenance."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from .repair import source_case, target_geometry


def apply_proposal(proposals, proposal_id, output):
    proposals, output = Path(proposals).resolve(), Path(output).resolve()
    report = read(proposals)
    protocol = read(resolve(report["protocol"]))
    source_execution = resolve(protocol["source_execution"])
    parent, _, _ = source_case(source_execution.parent)
    if (source_execution.name != "execution_bundle.json"
            or report["source_case"] != parent
            or protocol["source_scene"] != parent["scene"]
            or protocol["source_task"] != parent["task"]):
        raise ValueError("repair_protocol_source_case_mismatch")
    matches = [p for p in report["proposals"] if p["id"] == proposal_id]
    if len(matches) != 1:
        raise ValueError("one_existing_proposal_id_required")
    proposal = matches[0]
    prefix = {"reorder": "order_", "target_offset": "layout_"}.get(proposal["edit"]["type"])
    if prefix is None or proposal_id != prefix + digest(proposal["edit"])[:12]:
        raise ValueError("repair_proposal_id_does_not_bind_edit")
    request = {"schema": "scene_revision_request_v1", "proposals": binding(proposals),
               "proposal_id": proposal_id, "implementation": binding(__file__)}
    if output.exists():
        if not (output / "revision.json").exists() or read(output / "request.json") != request:
            raise ValueError("existing_revision_conflict_or_incomplete")
        result = read(output / "revision.json")
        for ref in (result["case"]["scene"], result["case"]["task"], result["case"]["source_request"]):
            resolve(ref)
        return result
    scene, task = deepcopy(read(resolve(parent["scene"]))), deepcopy(read(resolve(parent["task"])))
    source = deepcopy(read(resolve(parent["source_request"])))
    edit = proposal["edit"]
    layout = None
    if edit["type"] == "reorder":
        if task.get("incidental_controller_order_is_required", False):
            raise ValueError("ordered_task_cannot_be_reordered")
        ids = edit["operation_ids"]
        by_id = {o["id"]: o for o in task["operations"]}
        if len(ids) != len(by_id) or set(ids) != set(by_id):
            raise ValueError("all_original_operations_required_once")
        if any(ids.index(a) >= ids.index(b) for a, b in task.get("required_operation_precedence", [])):
            raise ValueError("required_precedence_changed")
        task["operations"] = [by_id[i] for i in ids]
        scope = "same_exact_goal_task"
    elif edit["type"] == "target_offset":
        if proposal.get("native_path_qualified") is not True:
            raise ValueError("native_path_evidence_required_for_layout_proposal")
        evidence = read(resolve(proposal["evidence"]))
        if evidence.get("native_path_qualified") is not True or evidence["offset_world_xy"] != edit["offset_world_xy_m"]:
            raise ValueError("proposal_does_not_match_qualified_geometry")
        if not resolve(proposal["evidence"]).is_relative_to(resolve(report["protocol"]).parent / "native_candidates"):
            raise ValueError("native_candidate_must_belong_to_bound_repair_run")
        attachment_path = resolve(evidence["attachment"])
        if not attachment_path.is_relative_to(source_execution.parent):
            raise ValueError("attachment_belongs_to_another_execution")
        attachment = read(attachment_path)
        if attachment["frame"] != protocol["failure"]["frame"] or edit["operation"] != protocol["failure"]["operation"]:
            raise ValueError("native_candidate_operation_or_attachment_frame_mismatch")
        layout = target_geometry(scene, task, edit["operation"], edit["offset_world_xy_m"])
        if not layout["passed"]:
            raise ValueError("new_target_outside_original_support_or_overlaps_other_goals")
        op = next(o for o in task["operations"] if o["id"] == edit["operation"])
        op["target"]["offset_xy"] = list(edit["offset_world_xy_m"])
        selected = next(t for t in source["tasks"] if t["id"] == parent["requested_task_id"])
        placement = next(p for p in selected["placements"] if p["object"] == op["object"])
        if placement["target"].get("container") != op["target"]["support"]:
            raise ValueError("source_support_instance_mismatch")
        placement["target"]["offset_xy_m"] = layout["source_local_offset_xy_m"]
        scope = "new_related_layout_task"
    else:
        raise ValueError("unsupported_scene_revision")
    identity = digest(request)[:10]
    scene_id = "revision_" + identity
    scene["id"] = scene_id
    scene["scene_instance_id"] = scene_id
    source["id"] = scene_id
    task["id"] = scene_id + "_" + parent["requested_task_id"]
    case = deepcopy(parent)
    case.update(id=task["id"], execution_id=task["id"], physical_status="not_executed")
    output.mkdir(parents=True)
    write(output / "request.json", request)
    write(output / "scene.json", scene)
    write(output / "task.json", task)
    write(output / "source_request.json", source)
    case.update(scene=binding(output / "scene.json"), task=binding(output / "task.json"),
                source_request=binding(output / "source_request.json"),
                revision_parent={"scene": parent["scene"], "task": parent["task"],
                    "source_request": parent["source_request"], "execution_id": parent["execution_id"]},
                revision_scope=scope)
    result = {"schema": "immutable_scene_task_revision_v1", "request": binding(output / "request.json"),
        "proposal": proposal, "case": case, "scope": scope, "layout_rechecked": layout,
        "original_physics_objects_and_supports_unchanged": True,
        "original_goal_tolerances_unchanged": True, "physical_success": "unmeasured",
        "additional_physical_steps": 0, "API_calls": 0, "GPU_jobs": 0}
    write(output / "case.json", case)
    write(output / "revision.json", result)
    return result
