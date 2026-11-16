"""Actual TIG grasp/payload snapshots and isolated cuRobo session transport.

This module only reads physical state. Reserved spheres belong to a fixed
planner-only TCP link; they never weld or move a simulator actor. Payload
spheres conservatively cover every actual collision shape in actor coordinates.
World shapes remain present, including the peer's carried object. The native
worker disables only this arm's exact held-target shapes and permits only its
two finger links to contact that target. A temporary, explicit source-table
exception is bounded by the caller's unchanged exact support checker.

Poses are XYZ/WXYZ in metres, arm arrays contain seven radians, sphere arrays
are [N,4] XYZ/radius, and inertia matrices are about the COM in actor axes.
Physical payload inertials are exported for audit. They do not assert payload
torque enforcement by cuRobo's independent URDF-based TimeOpt model.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from .curobo_scene import (_cover_convex, _finite, _kind, _matrix, _pose,
                           _shape_mesh, _shape_spheres, _transform_vertices,
                           export_curobo_snapshot)
from .curobo_sessions import (PROFILE_LIMITS, CuroboSessions, SessionTransport,
                             _SessionProcess, _digest, profile_config)

from .runtime_policy import RuntimePolicy, LEGACY_STRICT

PAYLOAD_LINK = "tig_payload"
FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")
SOURCE_TABLE = "scene::table::shape_0"
SCHEMA = "tig-actual-confirmed-grasp-payload-v1"
TIGHT_BOUND_PADDING_M = 1e-8


@lru_cache(maxsize=64)
def _tight_convex_cover_cached(vertices_bytes: bytes, count: int, pitch_m: float,
                                positive_z_padding_m: float | None,
                                max_depth: int) -> tuple[tuple[tuple[float, ...], ...], str]:
    """Enclose every nonempty convex-solid/cell intersection by a sphere.

    The coarse grid partitions the full solid's bounding box. The existing
    expanded-halfspace prefilter cannot remove an intersecting cell. Six LPs
    give each intersection's coordinate extrema, whose enclosing AABB and
    circumsphere contain every intersection point. Only proven LP-infeasible
    cells are omitted. Subdivision partitions the ORIGINAL cell in X/Y; using
    its newly fitted sphere or tight AABB for subdivision would lose this
    partition proof. The fixed positive-Z criterion depends on geometry alone.
    LP bounds and radius receive 10 nm outward numerical padding; infeasible
    optimization or an unmet refinement bound fails closed.
    """
    vertices = np.frombuffer(vertices_bytes, dtype=np.float64).reshape(count, 3)
    hull = ConvexHull(vertices)
    equations = hull.equations
    matrix, bound = equations[:, :3], -equations[:, 3]
    centers, _ = _cover_convex(vertices, pitch_m)
    max_z = float(vertices[:, 2].max())
    spheres: list[tuple[float, ...]] = []
    stats = {"initial_cells": len(centers), "empty_cells": 0, "refined_cells": 0,
             "maximum_depth": 0, "linear_programs": 0}
    options = {"primal_feasibility_tolerance": 1e-9,
               "dual_feasibility_tolerance": 1e-9}

    def cell(lower: np.ndarray, upper: np.ndarray, depth: int) -> None:
        stats["maximum_depth"] = max(stats["maximum_depth"], depth)
        low, high = np.empty(3), np.empty(3)
        bounds = list(zip(lower.tolist(), upper.tolist(), strict=True))
        for axis in range(3):
            objective = np.eye(3)[axis]
            extrema = []
            for sign in (1., -1.):
                result = linprog(sign * objective, A_ub=matrix, b_ub=bound,
                                 bounds=bounds, method="highs", options=options)
                stats["linear_programs"] += 1
                if result.status == 2:
                    stats["empty_cells"] += 1
                    return
                if not result.success or not math.isfinite(float(result.fun)):
                    raise ValueError(f"Convex cell LP failed: status={result.status}")
                extrema.append(float(result.fun))
            low[axis], high[axis] = extrema[0], -extrema[1]
        if np.any(low > high + 2e-9):
            raise ValueError("Inconsistent convex-cell coordinate extrema")
        low -= TIGHT_BOUND_PADDING_M
        high += TIGHT_BOUND_PADDING_M
        center = (low + high) / 2.
        radius = float(np.linalg.norm((high - low) / 2.)) + TIGHT_BOUND_PADDING_M
        assert center.shape == (3,) and np.isfinite(center).all() and radius > 0
        if (positive_z_padding_m is not None
                and center[2] + radius > max_z + positive_z_padding_m + 1e-12):
            if depth >= max_depth:
                raise ValueError("Positive-Z conservative cover needs more than the fixed refinement depth")
            stats["refined_cells"] += 1
            midpoint = (lower + upper) / 2.
            for x_half in (0, 1):
                for y_half in (0, 1):
                    child_low, child_high = lower.copy(), upper.copy()
                    for axis, half in ((0, x_half), (1, y_half)):
                        if half == 0:
                            child_high[axis] = midpoint[axis]
                        else:
                            child_low[axis] = midpoint[axis]
                    cell(child_low, child_high, depth + 1)
        else:
            spheres.append((*center.tolist(), radius))

    for center in centers:
        cell(center - pitch_m / 2., center + pitch_m / 2., 0)
    array = _spheres(spheres)
    source_gap = float(np.max(np.min(
        np.linalg.norm(vertices[:, None] - array[None, :, :3], axis=2) - array[None, :, 3], axis=1)))
    if source_gap > 1e-8:
        raise ValueError("Tight sphere model failed the necessary source-vertex coverage check")
    proof = {"coverage_policy": "full_convex_solid_cell_intersection_six_LP_tight_AABB_circumspheres_v1",
             "sphere_count": len(spheres), "sphere_pitch_m": pitch_m,
             "linear_program_bound_padding_m": TIGHT_BOUND_PADDING_M,
             "radius_outward_padding_m": TIGHT_BOUND_PADDING_M,
             "positive_z_padding_m": positive_z_padding_m,
             "maximum_allowed_refinement_depth": max_depth, **stats,
             "maximum_source_vertex_uncovered_gap_m": source_gap,
             "source_solid_max_positive_z_m": max_z,
             "maximum_sphere_positive_z_extent_m": float(np.max(array[:, 2] + array[:, 3])),
             "complete_solid_coverage": True,
             "proof": "Grid covers the solid bounding box; expanded-halfspace prefilter preserves every intersecting cell; six LP extrema bound the entire convex-solid/cell intersection; circumsphere encloses its AABB; recursive children partition the original parent cell in XY; only LP-infeasible cells are removed.",
             "numerical_scope": "HiGHS primal/dual feasibility tolerance 1nm; LP extrema and sphere radius padded outward by10nm; numerical failure or unmet fixed envelope rejects export."}
    return tuple(spheres), json.dumps(proof, sort_keys=True, allow_nan=False)


def tight_convex_spheres(vertices: Any, pitch_m: float = .025, *,
                         positive_z_padding_m: float | None = None,
                         max_depth: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """Return conservative [N,4] spheres for finite full-dimensional vertices[M,3]."""
    array = np.ascontiguousarray(vertices, dtype=np.float64)
    pitch = float(pitch_m)
    if (array.ndim != 2 or array.shape[1] != 3 or len(array) < 4
            or not np.isfinite(array).all() or np.any(np.ptp(array, axis=0) <= 0)
            or not math.isfinite(pitch) or pitch <= 0):
        raise ValueError("A finite nondegenerate solid and positive grid pitch are required")
    if (type(max_depth) is not int or not 0 <= max_depth <= 8
            or (positive_z_padding_m is not None
                and (not math.isfinite(positive_z_padding_m) or positive_z_padding_m <= 0))):
        raise ValueError("Invalid fixed refinement envelope or maximum depth")
    ConvexHull(array)  # Reject coplanar/degenerate inputs before caching.
    values, proof = _tight_convex_cover_cached(array.tobytes(), len(array), pitch,
        None if positive_z_padding_m is None else float(positive_z_padding_m), max_depth)
    return _spheres(values).copy(), json.loads(proof)


def _tight_shape_spheres(shape: Any, pitch_m: float, *,
                          hand_envelope: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cover a physical shape in its owning link/actor coordinates without state writes."""
    if _kind(shape) == "PhysxCollisionShapeSphere":
        return _shape_spheres(shape, pitch_m)
    vertices, _, geometry_policy = _shape_mesh(shape)
    local = _matrix(shape.get_local_pose())
    if hand_envelope:
        # The fixed +Z rule refers to physical hand/TCP axes, not mesh axes.
        vertices = _transform_vertices(vertices, local)
    values, proof = tight_convex_spheres(vertices, pitch_m,
        positive_z_padding_m=.002 if hand_envelope else None, max_depth=2)
    if not hand_envelope:
        values[:, :3] = _transform_vertices(values[:, :3], local)
    spheres = [{"center": row[:3].tolist(), "radius": float(row[3])} for row in values]
    proof.update(geometry_policy=geometry_policy, shape_local_pose_wxyz=_pose(local),
                 adaptive_rule_task_independent=hand_envelope)
    return spheres, proof


def _spheres(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1] != 4 or not len(result)
            or not np.isfinite(result).all() or np.any(result[:, 3] <= 0)):
        raise ValueError("Payload spheres must be finite [N>=1,4] with positive radii")
    return result


def reserve_payload_config(robot_cfg: dict[str, Any], capacity: int) -> dict[str, Any]:
    """Copy the real robot config and append a disabled, fixed TCP payload link.

    Existing robot geometry, limits, fingers, SRDF pairs and source URDF remain
    unchanged. Only the payload's two legitimate own-finger pairs are ignored.
    Zero nominal mass/inertia avoids silently adding LinkParams' default mass.
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 100_000:
        raise ValueError("Payload capacity must be an integer in [1,100000]")
    result = copy.deepcopy(robot_cfg)
    kin = result["kinematics"]
    links = kin["collision_link_names"]
    if (not isinstance(links, list) or len(set(links)) != len(links)
            or any(link not in links for link in FINGER_LINKS)
            or PAYLOAD_LINK in links or PAYLOAD_LINK in kin.get("collision_spheres", {})):
        raise ValueError("Expected unextended physical robot with both finger collision links")
    if kin.get("extra_links") or kin.get("extra_collision_spheres"):
        raise ValueError("Unexpected existing extra links/spheres cannot be overwritten")
    if kin.get("tool_frames") != ["panda_hand_tcp"]:
        raise ValueError("Payload must be fixed to the actual panda_hand_tcp tool frame")
    links.append(PAYLOAD_LINK)
    kin["collision_spheres"][PAYLOAD_LINK] = [
        {"center": [0., 0., 0.], "radius": -100.} for _ in range(capacity)]
    kin["extra_collision_spheres"] = {PAYLOAD_LINK: capacity}
    kin["extra_links"] = {PAYLOAD_LINK: {
        "parent_link_name": "panda_hand_tcp", "link_name": PAYLOAD_LINK,
        "joint_name": "tig_payload_fixed_joint", "joint_type": "FIXED",
        "fixed_transform": [0., 0., 0., 1., 0., 0., 0.],
        # Omit link_com: LinkParams' native default is a zero numpy vector.
        # Its v0.9 JSON constructor does not convert an explicit list to numpy.
        "link_mass": 0., "link_inertia": [0.] * 6}}
    ignores = kin.setdefault("self_collision_ignore", {})
    if PAYLOAD_LINK in ignores or any(PAYLOAD_LINK in values for values in ignores.values()):
        raise ValueError("Reserved payload name already occurs in physical SRDF policy")
    ignores[PAYLOAD_LINK] = list(FINGER_LINKS)
    for link in FINGER_LINKS:
        ignores.setdefault(link, []).append(PAYLOAD_LINK)
    kin.setdefault("self_collision_buffer", {})[PAYLOAD_LINK] = 0.
    json.dumps(result, allow_nan=False)
    return result


def physical_inertial(body: Any) -> dict[str, Any]:
    """Read PhysX mass, COM frame and principal inertia without using cached wrappers."""
    mass = float(body.mass)
    principal = _finite(body.inertia, (3,), "PhysX principal inertia")
    transform = _matrix(body.cmass_local_pose)
    if not math.isfinite(mass) or mass <= 0 or np.any(principal <= 0):
        raise ValueError("A dynamic grasp target needs positive mass/principal inertia")
    rotation = transform[:3, :3]
    inertia = rotation @ np.diag(principal) @ rotation.T
    assert inertia.shape == (3, 3) and np.isfinite(inertia).all()
    assert np.allclose(inertia, inertia.T, atol=1e-12, rtol=0)
    return {"mass_kg": mass, "com_actor_m": transform[:3, 3].tolist(),
            "inertia_actor_kgm2": inertia.tolist(),
            "principal_inertia_kgm2": principal.tolist(),
            "cmass_local_pose_wxyz": _pose(transform),
            "inertia_about": "physical_center_of_mass_in_actor_axes",
            "source": "actual_PhysxRigidDynamicComponent",
            "payload_inertia_applied_to_native_timeopt": False}


def _confirmation(evidence: Any, arm_index: int, current_tcp_object: np.ndarray, *,
                  policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, Any]:
    if not isinstance(policy, RuntimePolicy):
        raise ValueError("A validated RuntimePolicy is required")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("evidence_id"), str) or not evidence["evidence_id"].strip():
        raise ValueError("Attachment requires the ID of an actual grasp confirmation event")
    report = evidence.get("confirmation")
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise ValueError("Attachment requires the passed shared-executor grasp report")
    required = report.get("stable_steps_required")
    observed = report.get("stable_steps_observed")
    if (isinstance(required, bool) or not isinstance(required, int) or required < policy.grasp_stable_steps
            or not isinstance(observed, (list, tuple)) or len(observed) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < required for value in observed)):
        raise ValueError("Both real grasps must pass the selected stability barrier")
    anchor = evidence.get("tcp_to_object_wxyz")
    position_error, angle_error = 0., 0.
    if anchor is not None:
        fixed = _matrix(anchor)
        position_error = float(np.linalg.norm(fixed[:3, 3] - current_tcp_object[:3, 3]))
        relative = fixed[:3, :3].T @ current_tcp_object[:3, :3]
        angle_error = math.acos(float(np.clip((np.trace(relative) - 1.) / 2., -1., 1.)))
        if position_error > policy.position_slip_m or angle_error > policy.rotation_slip_rad:
            raise ValueError("Measured object slipped beyond the existing attachment tolerances")
    return {"evidence_id": evidence["evidence_id"], "physical_arm_index": arm_index,
            "stable_steps_required": required, "stable_steps_observed": list(observed),
            "measured_anchor_position_error_m": position_error,
            "measured_anchor_rotation_error_rad": angle_error,
            "confirmation_report_sha256": _digest(report)}


def _shape_names(snapshot: dict[str, Any], actor_name: str, count: int) -> list[str]:
    """Resolve exact canonical actor keys; similarly prefixed actors cannot match."""
    if not isinstance(actor_name, str) or not actor_name or count < 1:
        raise ValueError("Target must have an unambiguous actor key and collision shapes")
    names = [f"scene::{actor_name}::shape_{index}" for index in range(count)]
    world = snapshot["world"]
    all_names = [name for group in world.values() for name in group]
    coverage_names = [row["name"] for row in snapshot["coverage"]["world_shapes"]]
    if len(all_names) != len(set(all_names)):
        raise ValueError("World shape identifiers must be unique")
    prefix = f"scene::{actor_name}::shape_"
    actual = sorted(name for name in all_names if name.startswith(prefix))
    if actual != sorted(names) or any(coverage_names.count(name) != 1 for name in names):
        raise ValueError("Every actual target shape must occur exactly once in world coverage")
    return names


def _refine_finger_spheres(snapshot: dict[str, Any], robot: Any,
                            pitch_m: float) -> dict[str, Any]:
    """Replace only both actual finger solids by a finer conservative cover.

    The default exporter's 25 mm cells can protrude below an otherwise clear
    finger. Five-millimetre cells reduce that approximation error while still
    covering the complete solid, including each rotated local collision shape.
    No geometric subtraction, collision mask or table permission is involved.
    Input snapshot is owned by this exporter and is updated in place.
    """
    pitch = float(pitch_m)
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("Finger sphere pitch must be finite and positive")
    kin, coverage = snapshot["robot_cfg"]["kinematics"], snapshot["coverage"]
    physical_links = list(robot.get_links())
    changes: dict[str, Any] = {}
    new_rows: list[dict[str, Any]] = []
    for name in FINGER_LINKS:
        matches = [link for link in physical_links if link.get_name() == name]
        if len(matches) != 1 or len(matches[0]._objs) != 1:
            raise ValueError("Each physical finger needs exactly one unambiguous body")
        shapes = list(matches[0]._objs[0].get_collision_shapes())
        previous = [row for row in coverage["active_robot_shapes"] if row["link"] == name]
        if not shapes or len(previous) != len(shapes) or name not in kin["collision_spheres"]:
            raise ValueError("Actual finger shape coverage differs from the complete original export")
        values: list[dict[str, Any]] = []
        for index, shape in enumerate(shapes):
            spheres, proof = _shape_spheres(shape, pitch)
            values.extend(spheres)
            new_rows.append({"link": name, "shape_index": index, "type": _kind(shape),
                             "sphere_pitch_m": pitch, **proof})
        changes[name] = {"actual_shape_count": len(shapes),
                        "original_sphere_count": len(kin["collision_spheres"][name]),
                        "refined_sphere_count": len(values)}
        kin["collision_spheres"][name] = values
    coverage["active_robot_shapes"] = [
        row for row in coverage["active_robot_shapes"] if row["link"] not in FINGER_LINKS] + new_rows
    coverage["robot_collision_sphere_count"] = sum(len(values) for values in kin["collision_spheres"].values())
    coverage["finger_sphere_refinement"] = {
        "policy": "same_actual_solids_finer_conservative_grid_cell_circumspheres",
        "original_robot_sphere_pitch_m": float(coverage["sphere_pitch_m"]),
        "finger_sphere_pitch_m": pitch, "links": changes,
        "actual_physical_geometry_unchanged": True,
        "all_actual_finger_solids_covered": True,
        "collision_masks_unchanged": True, "table_permissions_unchanged": True}
    json.dumps(snapshot, allow_nan=False)
    return coverage["finger_sphere_refinement"]


def _refine_hand_spheres(snapshot: dict[str, Any], robot: Any,
                          pitch_m: float) -> dict[str, Any]:
    """Tight full-solid hand coverage, with a fixed two-level TCP-side envelope."""
    kin, coverage = snapshot["robot_cfg"]["kinematics"], snapshot["coverage"]
    matches = [link for link in robot.get_links() if link.get_name() == "panda_hand"]
    if len(matches) != 1 or len(matches[0]._objs) != 1:
        raise ValueError("Physical hand needs one unambiguous body")
    shapes = list(matches[0]._objs[0].get_collision_shapes())
    previous = [row for row in coverage["active_robot_shapes"] if row["link"] == "panda_hand"]
    if not shapes or len(previous) != len(shapes):
        raise ValueError("Physical hand coverage changed")
    original_count = len(kin["collision_spheres"]["panda_hand"])
    values, rows = [], []
    for index, shape in enumerate(shapes):
        spheres, proof = _tight_shape_spheres(shape, pitch_m, hand_envelope=True)
        values.extend(spheres)
        rows.append({"link": "panda_hand", "shape_index": index, "type": _kind(shape), **proof})
    kin["collision_spheres"]["panda_hand"] = values
    coverage["active_robot_shapes"] = [
        row for row in coverage["active_robot_shapes"] if row["link"] != "panda_hand"] + rows
    coverage["robot_collision_sphere_count"] = sum(len(values) for values in kin["collision_spheres"].values())
    refinement = {"original_sphere_count": original_count, "refined_sphere_count": len(values),
        "base_grid_pitch_m": float(pitch_m), "positive_z_padding_m": .002,
        "maximum_refinement_depth": 2, "subdivision_axes": ["X", "Y"],
        "actual_physical_geometry_unchanged": True, "all_actual_hand_solids_covered": True,
        "task_or_seed_dependent": False, "collision_masks_unchanged": True,
        "table_permissions_unchanged": True, "shape_proofs": rows}
    coverage["hand_sphere_refinement"] = refinement
    return refinement


def export_payload_snapshot(env: Any, arm_index: int, target_actor: Any, *,
                            contact_allowed: bool = False,
                            grasp_evidence: dict[str, Any] | None = None,
                            source_support_allowed: bool = False,
                            sphere_pitch_m: float = .025,
                            finger_sphere_pitch_m: float = .005,
                            payload_sphere_pitch_m: float = .025,
                            capacity: int | None = None,
                            policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, Any]:
    """Read actual target geometry/state and extend the old complete snapshot.

    Attach only after ``close_dual_and_confirm`` passes and this physical arm's
    ``is_grasping`` is still true. ``grasp_evidence`` contains ``evidence_id``
    and the exact ``confirmation`` report; it may additionally contain a fixed
    ``tcp_to_object_wxyz`` captured at the initial grasp for a slip check.
    ``source_support_allowed`` is only valid for a confirmed held object during
    the separately checked initial table departure, never ordinary transport.
    """
    if not isinstance(policy, RuntimePolicy):
        raise ValueError("A validated RuntimePolicy is required")
    if type(contact_allowed) is not bool or type(source_support_allowed) is not bool:
        raise ValueError("Contact and source-support permissions require actual booleans")
    if source_support_allowed and grasp_evidence is None:
        raise ValueError("Source support permission requires a confirmed held payload")
    snapshot = export_curobo_snapshot(env, arm_index, sphere_pitch_m=sphere_pitch_m)
    raw = getattr(env, "unwrapped", env)
    _refine_finger_spheres(snapshot, raw.agent.agents[arm_index].robot, finger_sphere_pitch_m)
    _refine_hand_spheres(snapshot, raw.agent.agents[arm_index].robot, sphere_pitch_m)
    from .scene_collision import scene_entities
    import sapien.physx as physx
    entities = scene_entities(raw)
    if isinstance(target_actor, str):
        if target_actor not in entities:
            raise ValueError("Target actor is absent from the actual scene")
        entity = entities[target_actor]
    elif hasattr(target_actor, "_objs"):
        if len(target_actor._objs) != 1:
            raise ValueError("Target must represent one physical actor")
        entity = target_actor._objs[0]
    else:
        entity = target_actor
    keys = [key for key, candidate in entities.items() if candidate == entity]
    if len(keys) != 1:
        raise ValueError("Target must resolve to exactly one canonical scene actor")
    actor_name = keys[0]
    bodies = [component for component in entity.components
              if isinstance(component, physx.PhysxRigidBaseComponent)]
    if len(bodies) != 1 or not isinstance(bodies[0], physx.PhysxRigidDynamicComponent):
        raise ValueError("Only an actual single dynamic rigid-body target can be grasped")
    body = bodies[0]
    shapes = list(body.get_collision_shapes())
    names = _shape_names(snapshot, actor_name, len(shapes))
    values, shape_coverage = [], []
    for name, shape in zip(names, shapes, strict=True):
        spheres, coverage = _tight_shape_spheres(shape, payload_sphere_pitch_m)
        values.extend([*sphere["center"], sphere["radius"]] for sphere in spheres)
        shape_coverage.append({"world_shape_name": name, **coverage})
    spheres = _spheres(values)
    if capacity is None:
        capacity = len(spheres)
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < len(spheres):
        raise ValueError("Reserved capacity must cover every conservative physical payload sphere")
    snapshot["robot_cfg"] = reserve_payload_config(snapshot["robot_cfg"], capacity)
    inertial = physical_inertial(body)
    model = {"schema": SCHEMA, "link_name": PAYLOAD_LINK, "capacity": capacity,
             "target_actor_name": actor_name, "target_obstacle_names": names,
             "finger_links": list(FINGER_LINKS), "sphere_count": len(spheres),
             "spheres_actor_local": spheres.tolist(),
             "geometry_sha256": _digest({"spheres": spheres.tolist(), "coverage": shape_coverage}),
             "shape_coverage": shape_coverage, "sphere_pitch_m": float(payload_sphere_pitch_m),
             "physical_inertial": inertial}
    # Keep the legacy wire representation byte-for-byte compatible. The worker
    # needs only the init-frozen grasp window; the report owns the full policy.
    if policy.grasp_stable_steps != LEGACY_STRICT.grasp_stable_steps:
        model["grasp_stable_steps"] = policy.grasp_stable_steps
    state: dict[str, Any] = {"contact_allowed": contact_allowed, "attachment": None,
                              "source_support_allowed": source_support_allowed,
                              "source_support_obstacle_names": [SOURCE_TABLE] if source_support_allowed else []}
    if grasp_evidence is not None:
        agent = raw.agent.agents[arm_index]
        observed = agent.is_grasping(raw.scene.actors[actor_name])
        if hasattr(observed, "detach"):
            observed = observed.detach().cpu().numpy()
        signal = np.asarray(observed)
        if signal.shape != (1,) or signal.dtype.kind != "b" or not bool(signal[0]):
            raise ValueError("The actual selected arm does not currently report grasping its target")
        world_base = np.linalg.inv(_matrix(snapshot["base_pose_wxyz"]))
        object_base = world_base @ _matrix(entity.pose)
        tcp_object = np.linalg.inv(_matrix(agent.tcp.pose)) @ _matrix(entity.pose)
        confirmation = _confirmation(grasp_evidence, arm_index, tcp_object, policy=policy)
        if source_support_allowed and SOURCE_TABLE not in snapshot["world"].get("cuboid", {}):
            raise ValueError("Source support must be the exact original table cuboid")
        state["attachment"] = {"spheres": spheres.tolist(),
            "object_pose_base_wxyz": _pose(object_base), "positions": snapshot["q_start"],
            "obstacle_names": names, "evidence_id": confirmation["evidence_id"],
            "tcp_to_object_wxyz": _pose(tcp_object), "physical_inertial": inertial,
            "grasp_confirmation": confirmation}
    snapshot.update(payload_model=model, payload_state=state)
    snapshot["coverage"]["payload"] = {
        "all_actual_target_shapes_covered": True, "physical_attachment": False,
        "physical_state_changed": False, "target_actor_name": actor_name,
        "target_obstacle_names": names, "reserved_sphere_capacity": capacity,
        "sphere_count": len(spheres), "payload_self_ignore_links": list(FINGER_LINKS),
        "payload_sphere_pitch_m": float(payload_sphere_pitch_m),
        "payload_sphere_policy": "full_convex_solid_cell_intersection_six_LP_tight_AABB_circumspheres_v1",
        "peer_payload_retained_in_world": True, "world_target_geometry_removed": False,
        "source_support_allowed": source_support_allowed,
        "source_support_external_exact_checker_required": source_support_allowed,
        "payload_inertia_applied_to_native_timeopt": False}
    json.dumps(snapshot, allow_nan=False)
    return snapshot


class _PayloadProcess(_SessionProcess):
    """Old supervised NDJSON transport, launching only the new payload worker."""
    def __init__(self, gpu_uuid: str, log_path: Path, *, timeout_s: float = 180.,
                 python: Path | None = None) -> None:
        super().__init__(gpu_uuid, log_path, timeout_s=timeout_s, python=python,
            _worker="curobo_tig_payload_worker.py")


class CuroboPayloadSessions(CuroboSessions):
    """Same four fixed planner/profile interface with versioned payload states."""
    def __init__(self, gpu_uuid: str, log_path: Path, *, timeout_s: float = 180.,
                 python: Path | None = None, transport: SessionTransport | None = None) -> None:
        super().__init__(gpu_uuid, log_path, timeout_s=timeout_s, python=python,
            transport=transport if transport is not None else _PayloadProcess(
                gpu_uuid, log_path, timeout_s=timeout_s, python=python))
        self._models: tuple[dict[str, Any], dict[str, Any]] | None = None

    @staticmethod
    def _payload_pair(snapshots: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        pair = CuroboSessions._pair(snapshots)
        names: list[set[str]] = []
        for snapshot in pair:
            model, state = snapshot.get("payload_model"), snapshot.get("payload_state")
            if not isinstance(model, dict) or not isinstance(state, dict):
                raise ValueError("Each complete payload snapshot needs model and state")
            if model.get("link_name") != PAYLOAD_LINK or model.get("finger_links") != list(FINGER_LINKS):
                raise ValueError("Only the actual TCP payload and two own fingers are supported")
            grasp_steps = model.get("grasp_stable_steps", LEGACY_STRICT.grasp_stable_steps)
            if type(grasp_steps) is not int or grasp_steps < 1:
                raise ValueError("Payload grasp_stable_steps must be a positive integer")
            count = len(_spheres(model["spheres_actor_local"]))
            if (type(model.get("capacity")) is not int or model["capacity"] < count
                    or model.get("sphere_count") != count):
                raise ValueError("Payload model capacity/sphere count is inconsistent")
            obstacles = model.get("target_obstacle_names")
            if not isinstance(obstacles, list) or not obstacles or len(set(obstacles)) != len(obstacles):
                raise ValueError("Payload model needs exact unique target obstacles")
            available = {name for group in snapshot["world"].values() for name in group}
            if not set(obstacles) <= available:
                raise ValueError("Full world must retain the target geometry")
            names.append(set(obstacles))
            if type(state.get("contact_allowed")) is not bool or type(state.get("source_support_allowed")) is not bool:
                raise ValueError("Payload collision permissions must be actual booleans")
            attachment = state.get("attachment")
            if state["source_support_allowed"]:
                if attachment is None or state.get("source_support_obstacle_names") != [SOURCE_TABLE] or SOURCE_TABLE not in available:
                    raise ValueError("Source exception is restricted to a held object and exact table")
            elif state.get("source_support_obstacle_names") != []:
                raise ValueError("Disabled source exception cannot retain obstacle permissions")
            if attachment is not None:
                if (not isinstance(attachment, dict) or attachment.get("obstacle_names") != obstacles
                        or not isinstance(attachment.get("evidence_id"), str) or not attachment["evidence_id"].strip()
                        or not np.array_equal(_spheres(attachment["spheres"]), _spheres(model["spheres_actor_local"]))):
                    raise ValueError("Attachment must use exactly its confirmed target model")
                _finite(attachment["positions"], (7,), "actual attachment arm state")
                _matrix(attachment["object_pose_base_wxyz"])
                _matrix(attachment["tcp_to_object_wxyz"])
        if names[0] & names[1]:
            raise ValueError("Physical arms need distinct canonical target actors")
        return pair

    def initialize_pair(self, snapshots: tuple[dict[str, Any], dict[str, Any]], *,
                        seed: int = 401, interpolation_dt: float = .02,
                        warmup: bool = True) -> list[dict[str, Any]]:
        if self._initialization_attempted:
            raise RuntimeError("Initialize each process only once")
        pair = self._payload_pair(snapshots)
        if any(snapshot["payload_state"]["attachment"] is not None for snapshot in pair):
            raise ValueError("Initialize reserved payload planners before grasping")
        if (type(seed) is not int or seed < 0 or not math.isfinite(interpolation_dt)
                or interpolation_dt <= 0 or type(warmup) is not bool):
            raise ValueError("Seed, interpolation timestep and warmup are invalid")
        configs = {(arm, profile): profile_config(snapshot["robot_cfg"], profile)
                   for profile in PROFILE_LIMITS for arm, snapshot in enumerate(pair)}
        self._initialization_attempted = True
        self._initial_seed, self._interpolation_dt = seed, float(interpolation_dt)
        self._models = tuple(copy.deepcopy(snapshot["payload_model"]) for snapshot in pair)
        results = []
        for profile in PROFILE_LIMITS:
            for arm, snapshot in enumerate(pair):
                session = self.session_id(arm, profile)
                self._tokens[session] = None
                results.append(self._rpc.request("init_session", session_id=session,
                    robot_cfg=configs[arm, profile], world=snapshot["world"], seed=seed,
                    interpolation_dt=float(interpolation_dt), warmup=warmup,
                    payload_model=snapshot["payload_model"]))
        self._initialized = True
        self._episode_reset_failed = False
        return results

    def update_pair(self, snapshots: tuple[dict[str, Any], dict[str, Any]], *,
                    profile: str, snapshot_id: str) -> list[dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Initialize all fixed payload sessions before updating")
        self._require_episode_ready()
        sessions = [self.session_id(arm, profile) for arm in (0, 1)]
        self._current_snapshot_id = None
        for session in sessions:
            self._tokens[session] = None
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("A nonempty physical snapshot version is required")
        pair = self._payload_pair(snapshots)
        if self._models is None or any(snapshot["payload_model"] != self._models[arm]
                                       for arm, snapshot in enumerate(pair)):
            raise ValueError("Persistent payload target geometry/inertial model cannot silently change")
        digest = _digest(pair)
        if snapshot_id in self._snapshot_digests and self._snapshot_digests[snapshot_id] != digest:
            raise ValueError("A physical snapshot token cannot identify different payload state")
        configs = [profile_config(snapshot["robot_cfg"], profile) for snapshot in pair]
        results = []
        for arm, snapshot in enumerate(pair):
            results.append(self._rpc.request("update_session", session_id=sessions[arm],
                robot_cfg=configs[arm], world=snapshot["world"], snapshot_id=self._wire_snapshot_id(snapshot_id),
                payload_state=snapshot["payload_state"]))
        self._snapshot_digests[snapshot_id] = digest
        for session in sessions:
            self._tokens[session] = snapshot_id
        self._current_snapshot_id = snapshot_id
        return results


def self_test() -> dict[str, Any]:
    """CPU geometry reservation, inertia/frame, confirmation and token checks."""
    from types import SimpleNamespace as NS
    import itertools
    rejected = 0
    def reject(function: Any) -> None:
        nonlocal rejected
        try:
            function()
        except (ValueError, RuntimeError, TypeError):
            rejected += 1
        else:
            raise AssertionError("Malformed payload input was accepted")
    identity = [0., 0., 0., 1., 0., 0., 0.]
    angle = math.pi / 2
    body = NS(mass=.15, inertia=np.array([1., 2., 3.]) * 1e-4,
              cmass_local_pose=NS(p=np.array([.01, .02, .03]),
                                   q=np.array([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])))
    inertia = physical_inertial(body)
    assert np.allclose(inertia["inertia_actor_kgm2"], np.diag([2., 1., 3.]) * 1e-4)
    reject(lambda: physical_inertial(NS(mass=-1., inertia=body.inertia, cmass_local_pose=body.cmass_local_pose)))
    report = {"status": "passed", "stable_steps_required": 5, "stable_steps_observed": [5, 6]}
    evidence = {"evidence_id": "synthetic-real-barrier-contract", "confirmation": report,
                "tcp_to_object_wxyz": identity}
    assert _confirmation(evidence, 0, np.eye(4))["stable_steps_observed"] == [5, 6]
    shifted = np.eye(4); shifted[0, 3] = .00501
    reject(lambda: _confirmation(evidence, 0, shifted))
    reject(lambda: _confirmation({**evidence, "confirmation": {**report, "stable_steps_observed": [5, 4]}}, 1, np.eye(4)))
    box_type = type("PhysxCollisionShapeBox", (), {})
    box = box_type()
    box.half_size = np.array([.00875, .0035, .01175])
    shape_pose = NS(p=np.array([.001, .0159, .02835]),
                    q=np.array([math.cos(math.pi / 12), math.sin(math.pi / 12), 0., 0.]))
    box.get_local_pose = lambda: shape_pose
    coarse, coarse_proof = _shape_spheres(box, .025)
    finger_cfg = {"kinematics": {"collision_spheres": {
        "panda_hand": [{"center": [0., 0., 0.], "radius": .01}],
        **{name: copy.deepcopy(coarse) for name in FINGER_LINKS}}}}
    finger_snapshot = {"robot_cfg": finger_cfg, "coverage": {
        "sphere_pitch_m": .025, "active_robot_shapes": [
            {"link": "panda_hand", "shape_index": 0, "type": "PhysxCollisionShapeSphere"},
            *[{"link": name, "shape_index": 0, "type": "PhysxCollisionShapeBox", **coarse_proof}
              for name in FINGER_LINKS]]}}
    fake_robot = NS(get_links=lambda: [NS(get_name=lambda name=name: name,
        _objs=[NS(get_collision_shapes=lambda: [box])]) for name in FINGER_LINKS])
    coarse_hand = copy.deepcopy(finger_cfg["kinematics"]["collision_spheres"]["panda_hand"])
    refined = _refine_finger_spheres(finger_snapshot, fake_robot, .005)
    dense = np.asarray(list(itertools.product(*(np.linspace(-half, half, 11) for half in box.half_size))))
    transform = _matrix(shape_pose)
    dense = dense @ transform[:3, :3].T + transform[:3, 3]
    fine = finger_cfg["kinematics"]["collision_spheres"][FINGER_LINKS[0]]
    centers = np.asarray([sphere["center"] for sphere in fine])
    radii = np.asarray([sphere["radius"] for sphere in fine])
    assert len(fine) > len(coarse) and np.allclose(radii, math.sqrt(3) * .005 / 2.)
    assert np.all(np.min(np.linalg.norm(dense[:, None] - centers[None], axis=2) - radii[None], axis=1) <= 1e-12)
    assert finger_cfg["kinematics"]["collision_spheres"]["panda_hand"] == coarse_hand
    assert finger_snapshot["coverage"]["robot_collision_sphere_count"] == 2 * len(fine) + 1
    assert refined["all_actual_finger_solids_covered"] and refined["collision_masks_unchanged"]
    json.dumps(finger_snapshot, allow_nan=False)
    reject(lambda: _refine_finger_spheres(finger_snapshot, fake_robot, 0.))
    cube = np.asarray(list(itertools.product([-.01, .01], [-.01, .01], [0., .02])))
    tight, tight_proof = tight_convex_spheres(cube, .025, positive_z_padding_m=.002)
    cube_points = np.asarray(list(itertools.product(np.linspace(-.01, .01, 13),
        np.linspace(-.01, .01, 13), np.linspace(0., .02, 13))))
    gap = np.min(np.linalg.norm(cube_points[:, None] - tight[None, :, :3], axis=2)
                 - tight[None, :, 3], axis=1)
    assert np.max(gap) <= 1e-12 and tight_proof["complete_solid_coverage"]
    assert np.max(tight[:, 2] + tight[:, 3]) <= .022 + 1e-12
    cache_before = _tight_convex_cover_cached.cache_info()
    tight_again, _ = tight_convex_spheres(cube, .025, positive_z_padding_m=.002)
    assert np.array_equal(tight, tight_again)
    assert _tight_convex_cover_cached.cache_info().hits == cache_before.hits + 1
    tetra = np.array([[0., 0., 0.], [.022, 0., 0.], [0., .017, 0.], [0., 0., .019]])
    mesh_type = type("PhysxCollisionShapeConvexMesh", (), {})
    mesh = mesh_type()
    mesh.vertices, mesh.scale = tetra, np.ones(3)
    mesh.get_triangles = lambda: ConvexHull(tetra).simplices
    mesh.get_local_pose = lambda: shape_pose
    tetra_spheres, tetra_proof = _tight_shape_spheres(mesh, .025)
    transformed = _spheres([[*sphere["center"], sphere["radius"]] for sphere in tetra_spheres])
    tetra_points = np.random.default_rng(406).dirichlet(np.ones(4), 1500) @ tetra
    tetra_points = _transform_vertices(tetra_points, _matrix(shape_pose))
    assert np.max(np.min(np.linalg.norm(tetra_points[:, None] - transformed[None, :, :3], axis=2)
                         - transformed[None, :, 3], axis=1)) <= 1e-12
    json.dumps({"cube": tight.tolist(), "proof": tight_proof,
                "rigid_tetrahedron": tetra_spheres, "tetrahedron_proof": tetra_proof}, allow_nan=False)
    reject(lambda: tight_convex_spheres(cube, 0.))
    reject(lambda: tight_convex_spheres(cube, .025, positive_z_padding_m=-.001))
    reject(lambda: tight_convex_spheres(cube, .025, positive_z_padding_m=.002, max_depth=0))
    reject(lambda: tight_convex_spheres(np.full((4, 3), float("nan"))))
    with tempfile.TemporaryDirectory(prefix="tig-curobo-payload-cpu-") as directory:
        from .curobo_scene import ARM_NAMES, FINGER_NAMES
        path = Path(directory) / "nine.urdf"
        names = list(ARM_NAMES + FINGER_NAMES)
        path.write_text("<robot name='test'>" + "".join(
            f"<joint name='{name}'><limit velocity='2.5'/></joint>" for name in names) + "</robot>")
        cfg = {"kinematics": {"urdf_path": str(path), "tool_frames": ["panda_hand_tcp"],
            "collision_link_names": list(FINGER_LINKS),
            "collision_spheres": {link: [{"center": [0., 0., 0.], "radius": .01}] for link in FINGER_LINKS},
            "self_collision_ignore": {"panda_leftfinger": ["panda_rightfinger"]},
            "self_collision_buffer": {link: 0. for link in FINGER_LINKS},
            "lock_joints": dict(zip(FINGER_NAMES, [.03, .031])),
            "cspace": {"joint_names": names, "default_joint_position": [0.] * 7 + [.03, .031],
                       "velocity_scale": [.2] * 9, "max_acceleration": .9}}, "load_dynamics": False}
        original = copy.deepcopy(cfg)
        reserved = reserve_payload_config(cfg, 2)
        assert cfg == original
        assert len(reserved["kinematics"]["collision_spheres"][PAYLOAD_LINK]) == 2
        assert reserved["kinematics"]["self_collision_ignore"][PAYLOAD_LINK] == list(FINGER_LINKS)
        assert "panda_hand" not in reserved["kinematics"]["self_collision_ignore"][PAYLOAD_LINK]
        reject(lambda: reserve_payload_config(cfg, True))
        reject(lambda: reserve_payload_config(reserved, 2))
        pair = []
        for arm in (0, 1):
            actor = f"target_{arm}"
            obstacle = f"scene::{actor}::shape_0"
            world = {"cuboid": {obstacle: {"pose": identity, "dims": [.01] * 3},
                                SOURCE_TABLE: {"pose": identity, "dims": [1., 1., .1]},
                                f"scene::{actor}_extra::shape_0": {"pose": identity, "dims": [.01] * 3}}, "mesh": {}}
            snapshot = {"robot_cfg": reserved, "world": world,
                "coverage": {"world_shapes": [{"name": name} for name in world["cuboid"]]},
                "payload_model": {"link_name": PAYLOAD_LINK, "capacity": 2,
                    "target_actor_name": actor, "target_obstacle_names": [obstacle],
                    "finger_links": list(FINGER_LINKS), "sphere_count": 1,
                    "spheres_actor_local": [[0., 0., 0., .005]], "physical_inertial": inertia},
                "payload_state": {"contact_allowed": False, "attachment": None,
                    "source_support_allowed": False, "source_support_obstacle_names": []}}
            assert _shape_names(snapshot, actor, 1) == [obstacle]
            pair.append(snapshot)
        class FakeTransport:
            def __init__(self) -> None:
                self.receipts: list[dict[str, Any]] = []
                self.fail_right = False
            def request(self, op: str, **payload: Any) -> dict[str, Any]:
                self.receipts.append({"op": op, **copy.deepcopy(payload)})
                if op == "update_session" and self.fail_right and payload["session_id"].startswith("right:"):
                    raise RuntimeError("synthetic partial pair failure")
                return {"op": op, **payload}
            def close(self) -> None:
                return None
        fake = FakeTransport()
        client = CuroboPayloadSessions("GPU-cpu-contract-only", Path(directory) / "unused", transport=fake)
        client.initialize_pair(tuple(pair), warmup=False)
        assert len(fake.receipts) == 4
        assert all("payload_model" in receipt for receipt in fake.receipts)
        client.update_pair(tuple(pair), profile="approach", snapshot_id="free")
        client.fk(arm=0, profile="approach", snapshot_id="free", positions=[0.] * 7)
        held = copy.deepcopy(pair)
        for snapshot in held:
            model = snapshot["payload_model"]
            snapshot["payload_state"].update(contact_allowed=True, source_support_allowed=True,
                source_support_obstacle_names=[SOURCE_TABLE], attachment={
                    "spheres": model["spheres_actor_local"], "obstacle_names": model["target_obstacle_names"],
                    "evidence_id": "barrier-passed", "positions": [0.] * 7,
                    "object_pose_base_wxyz": identity, "tcp_to_object_wxyz": identity,
                    "physical_inertial": inertia})
        client.update_pair(tuple(held), profile="transport", snapshot_id="liftoff")
        reject(lambda: client.fk(arm=0, profile="approach", snapshot_id="free", positions=[0.] * 7))
        client.plan_pose(arm=0, profile="transport", snapshot_id="liftoff", start=[0.] * 7, goal_pose=identity)
        bad = copy.deepcopy(held)
        bad[0]["payload_state"]["source_support_obstacle_names"] = ["arbitrary_table"]
        reject(lambda: client.update_pair(tuple(bad), profile="transport", snapshot_id="bad"))
        reject(lambda: client.plan_pose(arm=0, profile="transport", snapshot_id="liftoff", start=[0.] * 7, goal_pose=identity))
        bad = copy.deepcopy(held)
        bad[0]["payload_state"]["attachment"]["spheres"] = [[0., 0., 0., -.1]]
        reject(lambda: client.update_pair(tuple(bad), profile="transport", snapshot_id="bad2"))
        bad = copy.deepcopy(held)
        bad[0]["payload_model"]["capacity"] = 3
        reject(lambda: client.update_pair(tuple(bad), profile="transport", snapshot_id="bad3"))
        client.update_pair(tuple(held), profile="transport", snapshot_id="second")
        fake.fail_right = True
        reject(lambda: client.update_pair(tuple(pair), profile="approach", snapshot_id="partial"))
        reject(lambda: client.fk(arm=0, profile="transport", snapshot_id="second", positions=[0.] * 7))
        client.close()
    assert rejected == 17
    return {"status": "passed", "sphere_shape": [1, 4], "inertia_shape": [3, 3],
            "fixed_sessions": 4, "rejected_cases": rejected, "gpu_initialized": False,
            "finger_refinement": {"coarse_spheres_per_test_shape": len(coarse),
                "fine_spheres_per_test_shape": len(fine), "dense_solid_points_covered": len(dense),
                "finger_sphere_pitch_m": .005},
            "tight_solid_coverage": {"adaptive_cube_spheres": len(tight),
                "dense_cube_points_covered": len(cube_points),
                "rigid_tetrahedron_points_covered": len(tetra_points),
                "fixed_positive_z_padding_m": .002, "geometry_cache_verified": True}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), allow_nan=False))
