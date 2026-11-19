"""Candidate: expose existing native geometry and spatial event predicates.

No new per-example annotation. The old physics contract and missing real-source
coverage are retained. This compiler may prepare a synthetic quality probe only.
"""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from .core import checked, implementation, read, require

BATCH_AUDIT_VERSION = "batch_native_independent_acceptance_20261025_v1"


def validated_batch_audit(binding, root=None):
    """Read the independent frozen report, never infer admission from worker success."""
    report = read(checked(binding))
    require(report.get("schema") == BATCH_AUDIT_VERSION, "candidate_batch_audit_version_mismatch")
    if root is not None:
        require(Path(report["root"]).resolve() == Path(root).resolve(), "candidate_batch_audit_root_mismatch")
    require(report.get("batch_ready_for_sampling_within_verified_scope") is True
            and not report.get("split_leakages"), "candidate_batch_audit_not_ready")
    require(all(n.get("rejected") is True for n in report["negative_controls"])
            and all(g.get("same_initial_physics_objects_joints_and_views") is True for g in report["groups"]),
            "candidate_batch_negative_or_initial_control_failed")
    pairs = [(r["case"], r["attempt"]) for r in report["qualified_attempts"]]
    rows = [r for r in report["cases"] if r.get("independent_qualified") is True]
    require(len(pairs) == len(set(pairs)) == report["counts"]["independent_qualified"] == len(rows)
            and set(pairs) == {(r["case"], r["attempt"]) for r in rows},
            "candidate_batch_qualified_denominator_mismatch")
    require(report["counts"]["split_leakages"] == 0
            and report["counts"]["negative_controls_rejected"] == report["counts"]["negative_controls"],
            "candidate_batch_control_counts_failed")
    checked(report["auditor"])
    manifest = read(checked(report["evaluator_manifest"]))
    declared = manifest.get("files", manifest.get("evaluator_bindings", manifest))
    names = {"audit_batch_generation_20261025.py", "trace_geometry_acceptance_v3.py",
             "event_protocol.py", "interface.py", "compiler.py"}
    require(names <= set(report["evaluators"]), "candidate_batch_evaluators_missing")
    for name, bound in report["evaluators"].items():
        checked(bound)
        require(name in declared and bound["sha256"] == declared[name]["sha256"],
                "candidate_batch_evaluator_not_frozen:" + name)
        checked(declared[name])
    return report


def validate_batch_selection(selection, original, bundle):
    """Bind admission to one audited case AND attempt and its native contract."""
    require(selection.get("rule") == BATCH_AUDIT_VERSION, "candidate_batch_rule_mismatch")
    report = validated_batch_audit(selection["independent_audit"])
    case, attempt = selection["case"], selection["attempt"]
    updates_path = Path(report['root']) / 'admission_holdbacks.json'
    if updates_path.exists():
        updates = read(updates_path); checked(updates['evidence'])
        require(not any(r['case'] == case['id'] and r['attempt'] == attempt
                        for r in updates['case_attempts']), 'candidate_native_admission_superseded')
    require({"case": case["id"], "attempt": attempt} in report["qualified_attempts"],
            "candidate_case_attempt_not_independently_qualified")
    matches = [r for r in report["cases"] if r.get("case") == case["id"] and r.get("attempt") == attempt]
    require(len(matches) == 1, "candidate_batch_attempt_ambiguous")
    row = matches[0]
    require(row.get("independent_qualified") is True and row.get("native_measurements_qualified") is True
            and row.get("status") == "qualified" and not row.get("errors")
            and all(n.get("rejected") is True for n in row["negative_controls"]),
            "candidate_batch_attempt_rejected")
    require(row["case_record"] == case == bundle["case"]
            and row["execution"] == selection["execution"] == original["execution"],
            "candidate_batch_execution_binding_mismatch")
    expected = Path(report["root"]) / "executions" / case["id"] / attempt / "execution_bundle.json"
    require(Path(checked(selection["execution"])).resolve() == expected.resolve(),
            "candidate_batch_execution_path_mismatch")
    require(row["measurements"] == bundle["measurements"]
            and row["event_protocol"] == bundle["event_protocol"] == original["event_protocol"],
            "candidate_batch_native_evidence_mismatch")
    result = read(checked(selection["result"]))
    require(result["case"] == case and result["attempt"] == attempt and result["native_qualified"] is True
            and result["execution"] == selection["execution"]
            and result["physics_contract"] == selection["physical_contract"]
            and result["terminal"] == row["terminal"], "candidate_batch_contract_result_mismatch")
    checked(row["terminal"])
    return row


def spatial_targets(expression):
    """Extract position-near predicates; do not mistake forces or scalars for XYZ."""
    found = []
    if not isinstance(expression, dict):
        return found
    if expression.get("op") == "lt":
        args = expression.get("args", [])
        if len(args) == 2 and args[0].get("op") == "norm" and args[1].get("op") == "constant":
            sub = args[0].get("args", [{}])[0]
            operands = sub.get("args", [])
            if sub.get("op") == "sub" and len(operands) == 2:
                signal, target = operands
                point = target.get("value") if target.get("op") == "constant" else None
                if signal.get("op") == "signal" and isinstance(point, list) and len(point) == 3:
                    found.append({"channel": signal["name"], "world_xyz": point, "simulation_tolerance_m": args[1]["value"]})
    for arg in expression.get("args", []):
        found.extend(spatial_targets(arg))
    return found


def image_turn(points):
    points = np.asarray(points, float)
    require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 5, "closed_path_points_missing")
    require(np.linalg.norm(points[-1] - points[0]) < .001, "circular_path_is_not_closed")
    center = points[:-1].mean(axis=0)
    theta = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    turns = float((theta[-1] - theta[0]) / (2 * math.pi))
    return {"direction_in_image": "clockwise" if turns > 0 else "counterclockwise",
            "turns": round(abs(turns), 6), "normalized_image_center": center.round(6).tolist()}


def compile_candidate(selection):
    from .compiler import project
    from .event_protocol import load_channels
    from .novel_catalog import COLORS
    original = read(checked(selection["physical_contract"]))
    bundle = read(checked(selection["execution"]))
    require(original["execution"] == selection["execution"], "candidate_native_binding_mismatch")
    require(original["measured"]["qualification"]["passed"], "candidate_native_not_qualified")
    if selection.get("rule") == BATCH_AUDIT_VERSION:
        validate_batch_selection(selection, original, bundle)
    else:
        require(selection["rule"] == "novel_mechanism_acceptance_v3", "candidate_requires_independent_v3_audit")
        checked(selection["independent_audit"])
    result = deepcopy(original)
    result["native_baseline_contract"] = selection["physical_contract"]
    result["independent_native_audit"] = selection["independent_audit"]
    result["code"] = implementation()
    result["generation_scope"] = {"schema": "unpaired_sim_to_human_quality_probe_v1",
        "allowed_for_synthetic_quality_probe": True, "real_source_correspondence": "unverified",
        "formal_real_sim_human_pair": False, "training_admitted": False,
        "native_source_mapping_retained": True}
    if selection.get("rule") == BATCH_AUDIT_VERSION:
        result["generation_scope"].update(source_coverage="missing", pairing_scope="synthetic_only")
    require(result["source_mapping"]["sources"] == [] and not result["source_mapping"]["allowed_for_human_generation"],
            "candidate_scope_must_not_relabel_source_pairing")
    protocol = read(checked(original["event_protocol"]))
    scene, task = bundle["scene"], bundle["task_intent"]
    camera = bundle["cameras"][original["selected_camera"]]
    native = {x["id"]: x for x in scene["objects"] + scene.get("joints", [])}
    appearance = {}
    for name, entity in native.items():
        color = next((key for key, rgba in COLORS.items() if np.allclose(entity["rgba"], rgba)), None)
        appearance[name] = {"color_name": color, "rgba_from_native_scene": entity["rgba"],
                            "native_primitive_parts": entity.get("parts", []),
                            "preserve_shape_and_scale": True}
    for info in [result["initial_scene_information"], result["generation_information"]]:
        for entity in info["inventory"]:
            if entity["id"] in appearance:
                entity["native_appearance"] = appearance[entity["id"]]
    generation = result["generation_information"]
    generation["task_description_from_frozen_plan"] = task["goal_description"]
    generation["functional_shape_rule"] = (
        "Keep the actual supplied geometry, including compound components, gaps and proportions. "
        "Semantic names such as bowl or paddle do not authorize replacement with a different familiar object. "
        "Solid beads stay solid beads; do not invent water, powder, electricity, or material deformation.")
    with np.load(checked(bundle["measurements"]), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    channels = load_channels(protocol, arrays)
    measured = {e["id"]: e for e in original["measured"]["events"]}
    declared = {e["id"]: e for e in protocol["events"]}
    path_points = []
    for event in generation["required_events"]:
        original_event = declared[event["id"]]
        targets = spatial_targets(original_event["when"])
        for target in targets:
            channel = protocol["channels"][target["channel"]]
            require(channel["kind"] == "position", "spatial_event_requires_native_position_channel")
            target["entities"] = channel["entities"]
            target["image_xy_normalized"] = project(target["world_xyz"], camera)
            target["meaning"] = "required functional contact point; visible path geometry, not robot joint motion"
            del target["channel"]
            path_points.append({"event_id": event["id"], **target})
        if targets:
            event["required_spatial_targets"] = targets
        frame = measured[event["id"]]["frame"]
        joints = []
        for name, channel in protocol["channels"].items():
            if channel["kind"] == "joint_position" and set(channel["entities"]) & set(event["entities"]):
                joints.append({"entities": channel["entities"], "native_joint_position_at_event": float(channels[name][frame]),
                               "initial_position": float(channels[name][0]),
                               "visible_meaning": "demonstrate the declared joint displacement, then any rebound event; exact metric thresholds are simulation-only"})
        if joints:
            event["joint_state_evidence"] = joints
    generation["required_path_geometry"] = path_points
    params = task["planning_parameters"]
    if "turns" in params:
        turn = image_turn([p["image_xy_normalized"] for p in path_points])
        require(abs(turn["turns"] - params["turns"]) < .01, "task_turn_count_not_in_event_geometry")
        generation["required_cyclic_motion"] = {**turn,
            "ordered_event_ids": [p["event_id"] for p in path_points],
            "visible_rule": "complete the full ordered closed path in this direction and this number of turns; endpoints alone cannot show success"}
    result["condition_revision"] = "native_geometry_and_spatial_events_v1"
    result["manual_downstream_annotation_count"] = 0
    from .evidence_plan import compile_plan
    result["human_evidence_plan"] = compile_plan(result)
    return result
