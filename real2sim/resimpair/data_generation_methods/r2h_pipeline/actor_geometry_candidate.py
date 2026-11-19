"""CPU-only synthetic actor geometry preflight; no human or image acceptance.

All dimensions below are frozen engineering assumptions. Point reach is a
necessary condition for this proxy rig, not a human grasp/pose feasibility test.
The old unified candidate and its independent image gates remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import trimesh

from .batch_pipeline import binding, checked_binding
from .common import json_sha256, read_json, write_json
from .sim_state_conditions import world_vertices
from . import unified_anchor_candidate as unified

SCHEMA = "r2h_actor_geometry_candidate_v1"
RIG_SCHEMA = "r2h_actor_geometry_rig_v1"
EPS = 1e-8
PARAMETERS = {"nominal_upper_arm_m": .30, "nominal_forearm_m": .26,
              "nominal_shoulder_width_m": .36, "numerical_distance_tolerance_m": EPS,
              "anthropometric_population_claim": False,
              "parameter_changes_require_new_candidate_version": True}
TORSO_DIMS = [.24, .36, .60]
LEAN_ANGLES = [0, 15, 30, 45]
WAIST_HEIGHTS = [-.05, 0, .05]
EVENT_LABELS = {"grasp_apple": "grasp_established", "lift_from_source": "lift_clear",
                "transport_to_target": "last_grasp_before_release", "release_in_target": "released"}
UNKNOWN = ["human_wrist_to_object_surface_correspondence",
           "withdraw_hand_trajectory", "arm_hand_body_collision",
           "human_joint_limits_balance_and_anatomy", "rendered_occlusion_and_image_structure"]


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _vector(value, size, name):
    v = np.asarray(value, dtype=float)
    _require(v.shape == (size,) and np.isfinite(v).all(), "Invalid finite vector: " + name)
    return v


def _protocol(path):
    p = read_json(path)
    _require(p.get("schema") == "independent_actor_geometry_protocol_v1"
             and p.get("fixed_engineering_parameters") == PARAMETERS,
             "Geometry protocol/parameters differ from this frozen candidate version")
    directory = Path(path).parent
    lean_path = directory / "actor_geometry_lean_protocol.json"
    lean = read_json(lean_path)
    _require(lean.get("schema") == "independent_actor_geometry_lean_ablation_protocol_v1"
             and lean.get("parameters") == {
                 "torso_dimensions_depth_width_height_m": TORSO_DIMS,
                 "shoulder_axis_offset_from_waist_m": .54, "shoulder_width_m": .36,
                 "upper_arm_m": .30, "forearm_m": .26, "distance_tolerance_m": EPS,
                 "far_waist_clearance_from_projected_table_extent_m": .15,
                 "waist_height_relative_to_tabletop_m": WAIST_HEIGHTS,
                 "lean_degrees": LEAN_ANGLES,
                 "nominal_baseline": {"waist_height_relative_to_tabletop_m": 0., "lean_degrees": 0}},
             "Lean protocol parameters differ from frozen grid")
    return {"protocol": binding(path), "lean_protocol": binding(lean_path),
            "stage_gate_protocol": binding(directory / "actor_geometry_stage_gate_addendum.json"),
            "framing_protocol": binding(directory / "actor_geometry_framing_addendum.json")}


def camera_axes(camera):
    e = np.asarray(camera["camera_from_world"], dtype=float)
    _require(e.shape == (3, 4), "Expected calibrated 3x4 camera_from_world")
    return -np.linalg.solve(e[:, :3], e[:, 3])


def project(point, camera):
    e = np.asarray(camera["camera_from_world"])
    q = e[:, :3] @ np.asarray(point) + e[:, 3]
    result = {"camera_depth_m": float(q[2]), "normalized_uv": None, "in_frame": False}
    if q[2] <= camera["near"] or q[2] >= camera["far"]:
        result["reason"] = "depth_clipped"
        return result
    uv = (np.asarray(camera["intrinsic_matrix"]) @ q)[:2] / q[2]
    uv /= [camera["width"], camera["height"]]
    result.update(normalized_uv=uv.tolist(), in_frame=bool(np.all((uv >= 0) & (uv <= 1))))
    result["reason"] = "projected_point_only" if result["in_frame"] else "outside_image"
    return result


def rig_geometry(rig):
    """Derive BOTH shoulders and torso from one waist; no independent arm bases."""
    _require(rig.get("schema") == RIG_SCHEMA and rig.get("frame") == "world_m_z_up",
             "An explicit world-frame synthetic rig is required")
    _require(rig.get("arms") == {"upper_arm_m": .30, "forearm_m": .26, "shoulder_width_m": .36},
             "Arm parameters must equal frozen engineering assumptions")
    torso = rig["torso"]
    _require(set(torso) == {"waist_world_m", "facing_world_xy", "lean_degrees",
                           "dimensions_depth_width_height_m", "shoulder_height_from_waist_m"},
             "Only one waist-derived torso/shoulder system is allowed")
    _require(torso["dimensions_depth_width_height_m"] == TORSO_DIMS
             and torso["shoulder_height_from_waist_m"] == .54
             and torso["lean_degrees"] in LEAN_ANGLES,
             "Torso parameters differ from frozen upright/lean ablation")
    waist = _vector(torso["waist_world_m"], 3, "waist")
    fxy = _vector(torso["facing_world_xy"], 2, "facing")
    _require(abs(np.linalg.norm(fxy) - 1) < EPS, "Facing must be a unit horizontal direction")
    forward = np.r_[fxy, 0.]
    left = np.cross([0., 0., 1.], forward)
    theta = np.deg2rad(torso["lean_degrees"])
    axis = np.sin(theta) * forward + np.cos(theta) * np.array([0., 0., 1.])
    body_forward = np.cos(theta) * forward - np.sin(theta) * np.array([0., 0., 1.])
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((body_forward, left, axis))
    transform[:3, 3] = waist + .30 * axis
    mesh = trimesh.creation.box(extents=TORSO_DIMS, transform=transform)
    center = waist + .54 * axis
    shoulders = {"left": center + .18 * left, "right": center - .18 * left}
    return {"waist": waist, "forward": forward, "left": left, "axis": axis,
            "shoulders": shoulders, "torso_mesh": mesh}


def reach_chain(shoulder, target):
    """Exact two-segment point reach and one geometric elbow; not joint feasibility."""
    shoulder, target = np.asarray(shoulder), np.asarray(target)
    delta = target - shoulder
    d = float(np.linalg.norm(delta))
    possible = abs(.30 - .26) - EPS <= d <= .30 + .26 + EPS
    result = {"distance_m": d, "minimum_m": .04, "maximum_m": .56,
              "status": "point_reachable" if possible else "point_unreachable",
              "elbow_world_m": None, "chain_world_m": None,
              "joint_or_collision_feasibility_verified": False}
    if possible:
        unit = delta / d
        bend = np.array([0., 0., 1.]) - unit * unit[2]
        if np.linalg.norm(bend) < EPS:
            bend = np.array([1., 0., 0.]) - unit * unit[0]
        bend /= np.linalg.norm(bend)
        along = (.30**2 - .26**2 + d*d) / (2*d)
        elbow = shoulder + along * unit + np.sqrt(max(0., .30**2 - along**2)) * bend
        result.update(elbow_world_m=elbow.tolist(),
                      chain_world_m=[shoulder.tolist(), elbow.tolist(), target.tolist()])
    return result


def _contains_points(mesh, points):
    """Closed mesh solid-angle test avoids optional rtree; surface tolerance separate."""
    inside = np.zeros(len(points), dtype=bool)
    for component in mesh.split(only_watertight=False):
        if not component.is_watertight:
            continue
        for i, point in enumerate(points):
            a, b, c = (component.triangles - point).transpose(1, 0, 2)
            na, nb, nc = [np.linalg.norm(x, axis=1) for x in (a, b, c)]
            numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
            denominator = na*nb*nc + np.einsum("ij,ij->i", a, b)*nc \
                + np.einsum("ij,ij->i", b, c)*na + np.einsum("ij,ij->i", c, a)*nb
            inside[i] |= abs(np.sum(2*np.arctan2(numerator, denominator))) > 2*np.pi
    return inside


def torso_table_check(torso_mesh, table_mesh, table_path):
    """AABB broad phase then actual FCL surface intersection/closed-solid containment."""
    a, b = torso_mesh.bounds, table_mesh.bounds
    overlap = bool(np.all(a[0] <= b[1] + EPS) and np.all(b[0] <= a[1] + EPS))
    result = {"torso_bounds_world_m": a.tolist(), "table_bounds_world_m": b.tolist(),
              "aabb_overlap": overlap, "mesh_collision": False,
              "scope": "declared torso box versus complete saved physical table mesh only",
              "arm_hand_collision_status": "unknown"}
    if not overlap:
        result["status"] = "separated_by_aabb"
        return result
    from mplib.pymp.collision_detection import fcl
    with tempfile.TemporaryDirectory(prefix="actor-torso-") as tmp:
        torso_path = Path(tmp) / "torso.obj"
        torso_mesh.export(torso_path)
        obj1 = fcl.CollisionObject(fcl.load_mesh_as_BVH(str(torso_path), np.ones(3)))
        obj2 = fcl.CollisionObject(fcl.load_mesh_as_BVH(str(table_path), np.ones(3)))
        surface = bool(fcl.collide(obj1, obj2, fcl.CollisionRequest()).is_collision())
    closed = table_mesh.is_watertight and torso_mesh.is_watertight
    containment = bool(_contains_points(table_mesh, torso_mesh.vertices).any()
                       or _contains_points(torso_mesh, table_mesh.vertices).any()) if closed else False
    result.update(mesh_collision=surface or containment, surface_intersection=surface,
                  closed_mesh_containment=containment, closed_meshes=bool(closed),
                  status="modeled_mesh_collision" if surface or containment else
                  "no_modeled_collision_detected" if closed else "unknown_open_mesh_containment")
    return result


def _load_world(data, collision_world_path):
    receipt_path = Path(data["packet"]["bindings"]["receipt"]["path"]).resolve()
    expected_dir = receipt_path.parent.parent / "plans"
    _require(Path(collision_world_path).resolve() == expected_dir / "collision_world.json",
             "Collision world must belong to this receipt execution's plans directory")
    world = read_json(collision_world_path)
    for key in ("table_mesh", "selected_object_physical_mesh"):
        checked_binding(world[key])
        _require(Path(world[key]["path"]).resolve().parent == expected_dir,
                 "Geometry mesh belongs to another execution")
    table = trimesh.load(world["table_mesh"]["path"], force="mesh", process=False)
    selected = trimesh.load(world["selected_object_physical_mesh"]["path"], force="mesh", process=False)
    _require(len(table.faces) > 0 and len(selected.faces) > 0, "Empty physical mesh")
    return world, table, selected


def nominal_rig(data, table, *, lean=0, waist_height=0):
    _require(lean in LEAN_ANGLES and waist_height in WAIST_HEIGHTS, "Outside frozen ablation grid")
    eye = camera_axes(data["packet"]["camera_calibration"])
    center = table.bounds.mean(axis=0)
    toward = eye[:2] - center[:2]
    toward /= np.linalg.norm(toward)
    # The camera-facing extent is maximal; actual far extent is minimal.
    projections = (table.vertices[:, :2] - center[:2]) @ toward
    xy = center[:2] + (float(projections.min()) - .15) * toward
    return {"schema": RIG_SCHEMA, "actor_id": data["plan"]["actor"]["id"],
            "frame": "world_m_z_up", "torso": {
                "waist_world_m": [*xy.tolist(), float(table.bounds[1, 2] + waist_height)],
                "facing_world_xy": toward.tolist(), "lean_degrees": lean,
                "dimensions_depth_width_height_m": TORSO_DIMS,
                "shoulder_height_from_waist_m": .54},
            "arms": {"upper_arm_m": .30, "forearm_m": .26, "shoulder_width_m": .36},
            "camera_binding": {"camera_id": data["packet"]["selected_camera"],
                               "calibration_sha256": json_sha256(data["packet"]["camera_calibration"]),
                               "declared_table_side": "far"},
            "target_assumption": {"kind": "selected_object_surface_vertex_proxy",
                                  "selector": "highest_world_z_vertex",
                                  "wrist_offset_world_m": [0, 0, 0]},
            "anthropometric_population_claim": False}


def _compute(data, package_path, rig_path, collision_world_path, protocol_path):
    protocol_bindings = _protocol(protocol_path)
    rig = read_json(rig_path)
    _require(set(rig) == {"schema", "actor_id", "frame", "torso", "arms", "camera_binding",
                         "target_assumption", "anthropometric_population_claim"}
             and rig["anthropometric_population_claim"] is False,
             "Rig must have exactly the declared candidate fields and no human validation claim")
    g = rig_geometry(rig)
    world, table, local_object = _load_world(data, collision_world_path)
    camera = data["packet"]["camera_calibration"]
    _require(rig["actor_id"] == data["plan"]["actor"]["id"], "Rig belongs to another actor")
    _require(rig["camera_binding"] == {"camera_id": data["packet"]["selected_camera"],
             "calibration_sha256": json_sha256(camera), "declared_table_side": "far"},
             "Rig camera calibration/declared far side mismatch")
    _require(data["plan"]["actor"]["body_position"]["table_side"] == "far",
             "This version implements the declared far-side hypothesis only")
    _require(rig["target_assumption"] == {"kind": "selected_object_surface_vertex_proxy",
             "selector": "highest_world_z_vertex", "wrist_offset_world_m": [0, 0, 0]},
             "Unsupported human target mapping: robot TCP is not a human wrist")
    height = float(g["waist"][2] - table.bounds[1, 2])
    matches = [h for h in WAIST_HEIGHTS if abs(height-h) < EPS]
    _require(len(matches) == 1, "Waist height outside frozen declared ablation grid")
    expected_rig = nominal_rig(data, table, lean=rig["torso"]["lean_degrees"], waist_height=matches[0])
    _require(np.allclose(g["waist"], expected_rig["torso"]["waist_world_m"], atol=EPS, rtol=0)
             and np.allclose(g["forward"][:2], expected_rig["torso"]["facing_world_xy"], atol=EPS, rtol=0),
             "Waist/facing is not the selected camera's actual table far edge")
    assignments = {a["event_id"]: a for a in data["plan"]["event_assignments"]}
    known = []
    unknown_events = []
    a = np.load(data["packet"]["bindings"]["arrays"]["path"])
    selected = int(data["input"]["proposal"]["active_arm"])
    labels = {label: c for c in data["packet"]["conditions"] for label in c["labels"]}
    def target_record(index, hand, event_id, condition_labels):
        state = a["object_states"][index, selected]
        vertices = world_vertices(local_object.vertices, state)
        vertex = int(np.argmax(vertices[:, 2]))
        target = vertices[vertex]
        shoulder = g["shoulders"][hand]
        posed = trimesh.Trimesh(vertices=vertices, faces=local_object.faces, process=False)
        nearest, distances, _ = trimesh.proximity.closest_point_naive(posed, shoulder[None, :])
        return {"event_id": event_id, "condition_labels": condition_labels,
                "frame_index_0based": int(index), "timestamp_seconds": float(a["timestamps_seconds"][index]),
                "acting_hand": hand, "actor_id": rig["actor_id"],
                "object_id": data["packet"]["roles"]["selected_object"]["id"],
                "object_pose_wxyz_world": state[:7].tolist(), "surface_vertex_index": vertex,
                "target_world_m": target.tolist(), "target_kind": "assumed_object_surface_vertex_not_human_wrist",
                "human_contact_correspondence": "unknown", "robot_tcp_used_as_human_target": False,
                "shoulder_world_m": shoulder.tolist(), "reach": reach_chain(shoulder, target),
                "closest_surface_point_world_m": nearest[0].tolist(),
                "closest_surface_distance_m": float(distances[0]),
                "all_surface_points_beyond_maximum_reach": bool(distances[0] > .56 + EPS),
                "projection": project(target, camera)}
    for event_id, assignment in assignments.items():
        if event_id not in EVENT_LABELS:
            unknown_events.append({"event_id": event_id, "status": "unknown",
                                   "reason": "No human hand trajectory or surface mapping; object pose is not withdrawal"})
            continue
        label = EVENT_LABELS[event_id]
        c = labels[label]
        for hand in assignment["hands"]:
            known.append(target_record(c["frame"]["frame_index_0based"], hand, event_id, c["labels"]))
    held = np.flatnonzero(a["grasp"][:, :, selected].any(axis=1))
    grasp_hands = assignments["grasp_apple"]["hands"]
    _require(all(assignments[event]["hands"] == grasp_hands for event in EVENT_LABELS),
             "This candidate has no evidence for hand switching during held trajectory")
    trajectory = [target_record(int(i), hand, "measured_held_object_trajectory", [])
                  for i in held for hand in grasp_hands]
    collision = torso_table_check(g["torso_mesh"], table, world["table_mesh"]["path"])
    shoulder_projection = {k: project(v, camera) for k, v in g["shoulders"].items()}
    torso_points = [project(v, camera) for v in g["torso_mesh"].vertices]
    visible_torso_uv = [p["normalized_uv"] for p in torso_points if p["normalized_uv"] is not None]
    torso_box = np.array(visible_torso_uv)
    overlaps = []
    if len(torso_box):
        lo, hi = torso_box.min(0), torso_box.max(0)
        for r in known:
            uv = r["projection"]["normalized_uv"]
            if uv is not None and np.all(np.asarray(uv) >= lo) and np.all(np.asarray(uv) <= hi):
                overlaps.append({"event_id": r["event_id"], "frame_index_0based": r["frame_index_0based"]})
    rejected = []
    if any(r["reach"]["status"] == "point_unreachable" for r in known + trajectory):
        rejected.append("declared_surface_proxy_outside_fixed_arm_reach")
    if collision["mesh_collision"]:
        rejected.append("declared_torso_intersects_saved_table_mesh")
    risks = []
    if any(not p["in_frame"] for p in shoulder_projection.values()):
        risks.append("one_or_both_shoulders_outside_selected_camera")
    if any(not p["in_frame"] for p in torso_points):
        risks.append("torso_box_partially_clipped")
    if overlaps:
        risks.append("projected_torso_box_overlaps_key_object_points")
    if any(not r["projection"]["in_frame"] for r in known + trajectory):
        risks.append("task_target_projection_clipped")
    # Scope unknowns (human balance/withdrawal/real contact) are NOT prerequisites
    # for trying an initial image. Missing bindings fail earlier. Point clipping
    # blocks this fixed-camera hypothesis without claiming true rendered occlusion.
    projection_blocks = [x for x in risks if x in {
        "one_or_both_shoulders_outside_selected_camera", "task_target_projection_clipped"}]
    clears = not rejected and not projection_blocks
    return {"schema": SCHEMA, "status": "reject" if rejected else
            "blocked_projection" if projection_blocks else "not_falsified_by_declared_model",
            "bindings": {"source_package": binding(package_path), "rig": binding(rig_path),
                         **protocol_bindings, "collision_world": binding(collision_world_path),
                         "table_mesh": world["table_mesh"], "selected_object_mesh": world["selected_object_physical_mesh"],
                         "implementation": binding(__file__)},
            "context": data["context"], "synthetic_parameters": PARAMETERS,
            "waist_world_m": g["waist"].tolist(), "torso_axis_world": g["axis"].tolist(),
            "anatomical_left_world": g["left"].tolist(),
            "shoulders": {k: v.tolist() for k, v in g["shoulders"].items()},
            "event_targets": known, "unknown_events": unknown_events,
            "held_trajectory_targets": trajectory, "held_frame_count": int(len(held)),
            "torso_table": collision, "projection": {"shoulders": shoulder_projection,
                "torso_vertices": torso_points, "bbox_overlap_events": overlaps,
                "status": "risk" if risks else "projected_points_only", "risks": risks,
                "constraint_kind": "declared_framing_constraint_not_rendered_visibility",
                "rendered_visibility_verified": False},
            "reject_reasons": rejected, "projection_blocks": projection_blocks, "unknowns": UNKNOWN,
            "scope_unknowns_do_not_require_unavailable_human_3d_ground_truth": True,
            "clears_geometry_preflight": clears, "human_anatomy_verified": False,
            "generated_image_verified": False, "physical_execution_verified": False,
            "quality_release_approved": False, "training_admitted": False, "can_generate_video": False,
            "new_paid_calls": 0}


def compile_geometry(package_path, rig_path, collision_world_path, protocol_path, output):
    output = Path(output).resolve()
    _require(not output.exists(), "Geometry reports are immutable; use a new output file")
    started = time.monotonic()
    data = unified.validate_package(package_path)
    report = _compute(data, package_path, rig_path, collision_world_path, protocol_path)
    write_json(output, report)
    return {"report": binding(output), "status": report["status"],
            "wall_seconds": time.monotonic()-started, "new_paid_calls": 0}


def validate_geometry(report_path):
    report = read_json(report_path)
    _require(report.get("schema") == SCHEMA, "Unknown geometry report schema")
    for bound in report["bindings"].values():
        checked_binding(bound)
    b = report["bindings"]
    data = unified.validate_package(b["source_package"]["path"])
    expected = _compute(data, b["source_package"]["path"], b["rig"]["path"],
                        b["collision_world"]["path"], b["protocol"]["path"])
    _require(report == expected, "Geometry report differs from recomputed bound source/rig evidence")
    return report


def prepare_with_geometry(report_path, output, config):
    """Build the real provider request with static rig; retain a hard preflight block."""
    report = validate_geometry(report_path)
    output = Path(output).resolve()
    _require(not output.exists(), "Use a new geometry wrapper output directory")
    output.mkdir(parents=True)
    source = read_json(report["bindings"]["source_package"]["path"])
    placement = deepcopy(read_json(source["inputs"]["placement"]["path"]))
    rig = read_json(report["bindings"]["rig"]["path"])
    # Only static body geometry enters the initial prompt: no target path or final state.
    placement["actor"]["synthetic_geometry"] = {
        "rig": rig, "geometry_preflight_report": binding(report_path),
        "is_anthropometric_or_visual_evidence": False,
        "geometry_preflight_required_before_provider": True}
    write_json(output / "placement.json", placement)
    prepared = unified.prepare_unified(source["inputs"]["conditions"]["path"],
        source["inputs"]["execution_input"]["path"], output / "placement.json", output / "unified", config)
    wrapper = {"schema": "r2h_geometry_unified_wrapper_v1", "geometry": binding(report_path),
               "package": prepared["package"], "request": prepared["request"],
               "status": "prepared_unaccepted_anchor_candidate" if report["clears_geometry_preflight"]
                         else "blocked_geometry_preflight", "preflight_status": report["status"],
               "provider_adapter": "existing unified Gemini image adapter after successful wrapper validation",
               "implementation": binding(__file__), "new_paid_calls": 0,
               "can_generate_video": False, "quality_release_approved": False}
    write_json(output / "wrapper.json", wrapper)
    return wrapper


def execute_with_geometry(wrapper_path, config, budget_path, *, credentials_path=None):
    wrapper = read_json(wrapper_path)
    _require(wrapper.get("schema") == "r2h_geometry_unified_wrapper_v1", "Unknown wrapper schema")
    for key in ("geometry", "package", "request", "implementation"):
        checked_binding(wrapper[key])
    _require(wrapper["implementation"] == binding(__file__), "Wrapper implementation changed")
    report = validate_geometry(wrapper["geometry"]["path"])
    data = unified.validate_package(wrapper["package"]["path"])
    _require(data["context"] == report["context"], "Wrapper source/camera/role context differs")
    expected = {"rig": read_json(report["bindings"]["rig"]["path"]),
                "geometry_preflight_report": wrapper["geometry"],
                "is_anthropometric_or_visual_evidence": False,
                "geometry_preflight_required_before_provider": True}
    _require(data["plan"]["actor"].get("synthetic_geometry") == expected,
             "Prepared operator did not preserve bound geometry")
    source_package = read_json(report["bindings"]["source_package"]["path"])
    source_plan = read_json(source_package["actor_plan"]["path"])
    expected_actor = deepcopy(source_plan["actor"])
    expected_actor["synthetic_geometry"] = expected
    _require(data["plan"]["actor"] == expected_actor
             and data["plan"]["event_assignments"] == source_plan["event_assignments"],
             "Wrapper changed geometry's body/hand ownership or event assignment")
    _require(wrapper["request"] == binding(Path(wrapper["package"]["path"]).parent / "initial_request.json"),
             "Wrapper request belongs to another package")
    _require(report["clears_geometry_preflight"] is True,
             "Geometry preflight blocked: " + ", ".join(report["reject_reasons"] + report["projection_blocks"]))
    # A candidate image can be tried with declared scope unknowns. This does not
    # clear either independent image gate or authorize video generation.
    return unified.execute_initial(wrapper["package"]["path"], config, budget_path,
                                   credentials_path=credentials_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile")
    for name in ("package", "rig", "collision-world", "protocol", "output"):
        compile_parser.add_argument("--"+name, required=True)
    for command in ("prepare", "execute"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "prepare":
            p.add_argument("--report", required=True); p.add_argument("--output", required=True)
        else:
            p.add_argument("--wrapper", required=True); p.add_argument("--budget", required=True)
            p.add_argument("--credentials")
    p = sub.add_parser("validate"); p.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.command == "compile":
        result = compile_geometry(args.package, args.rig, args.collision_world, args.protocol, args.output)
    elif args.command == "prepare":
        result = prepare_with_geometry(args.report, args.output, read_json(args.config))
    elif args.command == "execute":
        result = execute_with_geometry(args.wrapper, read_json(args.config), args.budget,
                                       credentials_path=args.credentials)
    else:
        report = validate_geometry(args.report)
        result = {"status": report["status"], "recomputed": True, "new_paid_calls": 0}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
