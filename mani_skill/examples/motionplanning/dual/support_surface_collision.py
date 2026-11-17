"""Bounded initial source-plate support for actual convex collision geometry.

This adapter leaves the public dual executor and all ordinary collision checks
unchanged. Only a named source actor may support its own payload while the
payload separates upward. Geometry is in metres in the world frame; an object
transform is (4, 4), cooked convex vertices are (N, 3), and contacts retain their
original FCL fields. A candidate query never advances measured release state.
No function constructs an environment, advances physics, changes a mask, welds
an actor, or recaptures an existing attachment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import sapien
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl

from .dual_payload_collision import DualPayloadCollisionChecker
from .held_collision import AttachedObjectCollisionChecker, _object_at, _object_transform, _pair_result
from .table_support_collision import (
    NUMERICAL_TOP_TOLERANCE_M, _contact_values, _rigid_transform, _scaled_vertices, _vertices_at,
)

MIN_SUPPORT_NORMAL_Z = .25
SUPPORT_VERSION = "captured_source_contact_interior_support_v08"


def _one_entity(actor: Any) -> Any:
    objects = getattr(actor, "_objs", None)
    if objects is not None:
        if len(objects) != 1:
            raise ValueError("Source support must denote exactly one physical entity")
        return objects[0]
    if actor is None:
        raise ValueError("Source support actor is required")
    return actor


def convex_planes(vertices: Any, triangles: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return exact convex-vertex halfspaces A*x <= b, without changing a shape.

    PhysX's reported polygon fan triangles can be slightly nonplanar. They are
    retained as raw geometry, but the convex solid's support function is defined
    by its vertices. Qhull reconstructs that same vertex hull for this independent
    upper-surface proof; no reconstructed mesh is installed into either backend.
    """
    from scipy.spatial import ConvexHull, QhullError

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4
            or triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) < 4
            or not np.isfinite(vertices).all() or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise RuntimeError("Finite cooked convex vertices and indexed triangles are required")
    try:
        hull = ConvexHull(vertices)
    except QhullError as exc:
        raise RuntimeError("Degenerate convex support geometry") from exc
    normals, offsets = hull.equations[:, :3].copy(), -hull.equations[:, 3].copy()
    if (not np.isfinite(hull.equations).all()
            or not np.allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-10)
            or np.max(vertices @ normals.T - offsets[None]) > 1e-9):
        raise RuntimeError("Invalid reconstructed convex-vertex halfspaces")
    if not np.any(normals[:, 2] > MIN_SUPPORT_NORMAL_Z):
        raise RuntimeError("No upward source support surface")
    return normals, offsets


def upper_surface_at_xy(normals: np.ndarray, offsets: np.ndarray,
                        xy: Any) -> dict[str, Any] | None:
    """Intersect a world vertical line with an actual convex support piece."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise RuntimeError("Finite world XY required")
    normals, offsets = np.asarray(normals), np.asarray(offsets)
    residual = offsets - normals[:, :2] @ xy
    nz = normals[:, 2]
    horizontal = np.abs(nz) <= 1e-10
    if np.any(residual[horizontal] < -NUMERICAL_TOP_TOLERANCE_M):
        return None
    upward, downward = nz > 1e-10, nz < -1e-10
    if not np.any(upward) or not np.any(downward):
        raise RuntimeError("Support hull must be bounded vertically")
    upper_values = residual[upward] / nz[upward]
    upper = float(upper_values.min())
    lower = float((residual[downward] / nz[downward]).max())
    if lower > upper + NUMERICAL_TOP_TOLERANCE_M:
        return None
    # At a shared edge all limiting faces must remain upward support faces;
    # a vertical or underside side-touch cannot gain an allowance.
    candidate = np.flatnonzero(upward)[np.abs(upper_values - upper) <= 1e-7]
    return {"top_z_m": upper, "bottom_z_m": lower,
            "upper_face_normal_z": float(normals[candidate, 2].min()),
            "upper_face_indices": candidate.tolist()}


def _xy_projection(vertices: np.ndarray) -> np.ndarray:
    """CCW projection of actual convex vertices; no proxy or filled union."""
    from scipy.spatial import ConvexHull, QhullError

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise RuntimeError("Finite actual world convex vertices required")
    try:
        return vertices[ConvexHull(vertices[:, :2]).vertices, :2].copy()
    except QhullError as exc:
        raise RuntimeError("Degenerate actual convex XY projection") from exc


def _positive_xy_overlap(target: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    """Clip actual convex XY polygons; a line/corner is not top support.

    Both inputs are CCW (N,2) metre coordinates. The area uncertainty uses
    floating-point roundoff and the existing 1um geometry resolution. A cooked
    nanometre sliver at a nominal outer corner is therefore not positive proof.
    Clipping never expands a polygon or joins source pieces.
    Other callers still prove actual top/bottom heights and contact depths.
    """
    target, source = np.asarray(target, float), np.asarray(source, float)
    if any(p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 3 or not np.isfinite(p).all()
           for p in (target, source)):
        raise RuntimeError("Finite convex CCW projected polygons required")
    # Translate before cross products to avoid an origin-dependent area test.
    origin = source[0].copy()
    polygon = [point.copy() for point in target - origin]
    clip = source - origin
    cross = lambda a, b: float(a[0] * b[1] - a[1] * b[0])
    for index, a in enumerate(clip):
        b = clip[(index + 1) % len(clip)]
        edge = b - a
        previous = polygon[-1] if polygon else None
        output = []
        for current in polygon:
            before, after = cross(edge, previous - a), cross(edge, current - a)
            if (before >= 0.) != (after >= 0.):
                fraction = before / (before - after)
                output.append(previous + fraction * (current - previous))
            if after >= 0.:
                output.append(current)
            previous = current
        polygon = output
        if not polygon:
            break
    result = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    area = 0.
    if len(result) >= 3:
        centered = result - result[0]
        area = abs(sum(cross(a, b) for a, b in zip(centered, np.roll(centered, -1, axis=0)))) / 2
    extent = float(np.ptp(np.vstack((target, source)), axis=0).max())
    roundoff = 32 * np.finfo(np.float64).eps * (len(target) + len(source) + len(result)) * extent**2
    perimeter = (float(np.linalg.norm(result - np.roll(result, -1, axis=0), axis=1).sum())
                 if len(result) else 0.)
    # Convex parallel-body area bound at the already-declared numerical
    # resolution. This only rejects ambiguous overlap; it adds no geometry.
    numerical_area_bound = perimeter * NUMERICAL_TOP_TOLERANCE_M + np.pi * NUMERICAL_TOP_TOLERANCE_M**2
    receipt = {"proof": "positive_area_intersection_of_actual_target_and_source_convex_XY_projections",
               "area_m2": float(area), "floating_point_area_bound_m2": float(roundoff),
               "existing_geometry_resolution_m": NUMERICAL_TOP_TOLERANCE_M,
               "numerical_area_bound_m2": float(numerical_area_bound),
               "positive_area": bool(area > max(roundoff, numerical_area_bound)),
               "intersection_xy_world_m": (result + origin).tolist(),
               "source_union_filled_or_geometry_replaced": False}
    if not receipt["positive_area"]:
        raise RuntimeError("No positive-area actual target/source top-support projection")
    return receipt


def _source_contact_interior(point: Any, polygon: np.ndarray) -> dict[str, Any]:
    """Prove this contact lies inside its actual source-piece projection.

    A body's inward overhang does not turn a contact on the outside rim into
    upper support. Unlike fixed-offset probes, this tests the supplied contact
    against the actual projected halfspaces without requiring a physical
    clearance margin. A boundary indistinguishable from floating-point
    roundoff is conservatively unresolved. Other source pieces or their hull
    never fill a gap or establish contact here.
    """
    point, polygon = np.asarray(point, float), np.asarray(polygon, float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise RuntimeError("Finite original source-contact XY required")
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    delta = point - polygon
    clearance = (edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0]) / lengths
    scale = max(float(np.abs(polygon).max()), float(np.abs(point).max()),
                float(np.ptp(polygon, axis=0).max()))
    roundoff = 32 * np.finfo(np.float64).eps * len(polygon) * scale
    receipt = {"proof": "original_contact_strictly_inside_actual_source_convex_XY_projection",
               "contact_xy_world_m": point.tolist(), "minimum_interior_distance_m": float(clearance.min()),
               "floating_point_distance_bound_m": float(roundoff),
               "physical_clearance_margin_added": False, "source_union_filled": False}
    if not np.isfinite(clearance).all() or np.min(clearance) <= roundoff:
        raise RuntimeError("Source contact lacks proven actual projected interior")
    return receipt


class SourceSurfaceSupportCollisionChecker(AttachedObjectCollisionChecker):
    """One captured source actor, with a permanent measured-only release gate."""

    @classmethod
    def from_live(cls, live_checker: Any, target_actor: Any, active_robot: str = "left",
                  tcp_link_name: str = "panda_hand_tcp", *, support_actor: Any,
                  support_band_m: float = .003, support_penetration_m: float = .003,
                  support_depth_slop_m: float = .0002) -> SourceSurfaceSupportCollisionChecker:
        entity = _one_entity(support_actor)

        # The base factory captures the attachment exactly once and calls the
        # virtual registration hook before returning. A construction-local
        # subclass provides that hook's immutable source identity without any
        # global factory state or changes to the live checker.
        class ConfiguredSourceSupport(cls):
            def __init__(self) -> None:
                self._source_entity = entity

        return AttachedObjectCollisionChecker.from_live.__func__(
            ConfiguredSourceSupport, live_checker, target_actor, active_robot, tcp_link_name,
            support_band_m=support_band_m, support_penetration_m=support_penetration_m,
            support_depth_slop_m=support_depth_slop_m)

    def _control_step(self) -> int | None:
        value = getattr(getattr(self, "env", None), "elapsed_steps", None)
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.size != 1 or not np.isfinite(value).all():
            raise RuntimeError("Invalid source-support step telemetry")
        return int(value.reshape(-1)[0])

    def _register_initial_support(self, objects: dict[str, Any]) -> None:
        keys = [key for key, entity in self.live._entities.items() if entity == self._source_entity]
        if len(keys) != 1 or self._source_entity == self._entity:
            raise RuntimeError("Initial source support requires one distinct registered actor")
        self.source_actor_name = keys[0]
        self._source_names = frozenset(name for name, (entity, _) in self.live._shape_refs.items()
                                       if entity == self._source_entity)
        if not self._source_names or not self._source_names <= set(objects):
            raise RuntimeError("Source actor must have finite registered collision shapes")
        body = self._source_entity.find_component_by_type(physx.PhysxRigidStaticComponent)
        if body is None:
            raise RuntimeError("This source-plate adapter requires a physically static source actor")
        self._local_vertices = {name: _scaled_vertices(self.live, name).copy()
                                for name in self._target_names | self._source_names}
        self._source_transforms = {name: _rigid_transform(_object_transform(objects[name])).copy()
                                   for name in self._source_names}
        self._source_planes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._target_local_planes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._source_xy_projections: dict[str, np.ndarray] = {}
        self._geometry_export: dict[str, Any] = {}
        for name, local in self._local_vertices.items():
            _, shape = self.live._shape_refs[name]
            triangles = np.asarray(shape.get_triangles(), dtype=np.int64)
            self._geometry_export[name] = {"scaled_shape_local_vertices": local.tolist(),
                                           "triangles": triangles.tolist()}
            if name in self._source_names:
                world = _vertices_at(local, self._source_transforms[name])
                self._source_planes[name] = convex_planes(world, triangles)
                self._source_xy_projections[name] = _xy_projection(world)
                self._geometry_export[name]["captured_world_T"] = self._source_transforms[name].tolist()
                normals, offsets = self._source_planes[name]
                self._geometry_export[name].update(
                    upper_surface_proof="exact_convex_hull_of_captured_scaled_vertices_no_collision_geometry_replacement",
                    world_convex_halfspace_normals=normals.tolist(), world_convex_halfspace_offsets=offsets.tolist())
            else:
                self._geometry_export[name]["shape_local_T"] = self._shape_local[name].tolist()
                self._target_local_planes[name] = convex_planes(local, triangles)
        self._initial_receipts: list[dict[str, Any]] = []
        self._initial_region_receipts: list[dict[str, Any]] = []
        self._initial_region_queries: list[dict[str, Any]] = []
        self._validation_history: list[dict[str, Any]] = []
        self._state_history: list[dict[str, Any]] = []
        self._initial_object_depth_m = 0.
        for target in sorted(self._target_names):
            for source in sorted(self._source_names):
                collided, contacts = _pair_result(objects[target], objects[source])
                if not collided:
                    continue
                receipt = self._contact_geometry(target, source, contacts, self._initial_object,
                                                  self.support_penetration_m)
                self._support_pairs[(target, source)] = {"initial_depth": receipt["max_contact_depth_m"],
                                                         "contacts": copy.deepcopy(contacts)}
                self._initial_object_depth_m = max(self._initial_object_depth_m, receipt["max_contact_depth_m"])
                self._initial_receipts.append(receipt)
        # Registration describes the original 3 mm physical support region;
        # it is not an assertion of an FCL or native physical contact. Cooked
        # geometry can have a tiny positive gap at capture, then settle into
        # a shallow real contact. Keep all original contact receipts separate.
        for target in sorted(self._target_names):
            for source in sorted(self._source_names):
                distance = float(fcl.distance(objects[target], objects[source], fcl.DistanceRequest()).min_distance)
                if not np.isfinite(distance):
                    raise RuntimeError("Nonfinite initial source-region distance")
                if distance > self.support_band_m:
                    continue
                region = self._upper_region_geometry(target, source, self._initial_object)
                collided, contacts = _pair_result(objects[target], objects[source])
                region.update(fcl_distance_m=distance, fcl_collision=bool(collided),
                    actual_fcl_contacts_present=bool(collided and contacts),
                    raw_contacts=copy.deepcopy(contacts), registration_band_m=self.support_band_m,
                    registration_does_not_assert_contact=True,
                    native_physx_contact_proven_here=False, support_version=SUPPORT_VERSION)
                self._initial_region_queries.append(copy.deepcopy(region))
                if region["status"] == "bounded_upper_region":
                    self._initial_region_receipts.append(copy.deepcopy(region))
        self._initial_support_exists = bool(self._support_pairs or self._initial_region_receipts)
        # The capture pose, attachment and shape-local transforms stay fixed.
        # Only an independently validated REAL measurement may lower this
        # scalar origin reference. Candidates can read it, never update it.
        self._support_z_reference_m = float(self._initial_object[2, 3])
        self._support_z_reference_history: list[dict[str, Any]] = []
        self._state_receipt = {"mode": "initial_registration", "reason": "source_actor_captured",
                               "source_actor": self.source_actor_name, "released_after": False,
                               "registered_region_is_contact_assertion": False}

    def _upper_region_geometry(self, target: str, source: str, pose: np.ndarray) -> dict[str, Any]:
        """Prove a near upper region from the actual convex halfspaces.

        The LP variables are world (x, y, target_z, source_z). Minimizing
        target_z-source_z over both convex solids gives the smallest vertical
        bottom-to-top gap on their shared XY projection. This is a geometric
        witness, never a synthesized collision contact. LP coordinates are
        independently re-evaluated against the actual upper halfspaces and
        the unchanged union-interior probes used by real contact validation.
        """
        from scipy.optimize import linprog

        pose = _rigid_transform(pose)
        shape_pose = pose @ self._shape_local[target]
        local_normals, local_offsets = self._target_local_planes[target]
        target_normals = local_normals @ shape_pose[:3, :3].T
        target_offsets = local_offsets + target_normals @ shape_pose[:3, 3]
        source_normals, source_offsets = self._source_planes[source]
        target_a = np.column_stack((target_normals[:, :2], target_normals[:, 2], np.zeros(len(target_normals))))
        source_a = np.column_stack((source_normals[:, :2], np.zeros(len(source_normals)), source_normals[:, 2]))
        solution = linprog([0., 0., 1., -1.], A_ub=np.vstack((target_a, source_a)),
            b_ub=np.concatenate((target_offsets, source_offsets)), bounds=[(None, None)] * 4,
            method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                     "dual_feasibility_tolerance": 1e-9})
        receipt: dict[str, Any] = {"target": target, "source": source, "source_actor": self.source_actor_name,
            "mode": "actual_convex_upper_region_without_contact_assertion", "status": "outside_upper_region",
            "linear_program_status": int(solution.status), "support_band_m": self.support_band_m,
            "support_penetration_m": self.support_penetration_m}
        if solution.status == 2:
            receipt["reason"] = "no_common_vertical_profile"
            return receipt
        if not solution.success or not np.isfinite(solution.x).all():
            raise RuntimeError("Initial source-region convex proof failed")
        xy = solution.x[:2]
        surface = upper_surface_at_xy(source_normals, source_offsets, xy)
        target_surface = upper_surface_at_xy(target_normals, target_offsets, xy)
        if surface is None or target_surface is None:
            receipt["reason"] = "LP_witness_not_in_actual_convex_profiles"
            return receipt
        gap = float(target_surface["bottom_z_m"] - surface["top_z_m"])
        try:
            overlap = _positive_xy_overlap(
                _xy_projection(_vertices_at(self._local_vertices[target], shape_pose)),
                self._source_xy_projections[source])
        except RuntimeError as exc:
            receipt["reason"] = str(exc)
            return receipt
        receipt.update(witness_xy_world_m=xy.tolist(), minimum_vertical_gap_m=gap,
                       source_surface=surface, target_vertical_profile=target_surface,
                       actual_xy_support_overlap=overlap)
        if (gap < -self.support_penetration_m - NUMERICAL_TOP_TOLERANCE_M
                or gap > self.support_band_m or surface["upper_face_normal_z"] < MIN_SUPPORT_NORMAL_Z):
            receipt["reason"] = "deep_side_rim_or_outside_original_support_band"
            return receipt
        receipt.update(status="bounded_upper_region", reason="independent_actual_convex_upper_region")
        return receipt

    def _contact_geometry(self, target: str, source: str, contacts: list[dict[str, Any]],
                          pose: np.ndarray, depth_limit: float) -> dict[str, Any]:
        if not contacts:
            raise RuntimeError("Source support requires original finite FCL contacts")
        normals, offsets = self._source_planes[source]
        shape_pose = pose @ self._shape_local[target]
        world = _vertices_at(self._local_vertices[target], shape_pose)
        local_target_normals, local_target_offsets = self._target_local_planes[target]
        target_normals = local_target_normals @ shape_pose[:3, :3].T
        target_offsets = local_target_offsets + target_normals @ shape_pose[:3, 3]
        overlap = _positive_xy_overlap(_xy_projection(world), self._source_xy_projections[source])
        profiles = []
        for vertex in world:
            surface = upper_surface_at_xy(normals, offsets, vertex[:2])
            if surface is not None:
                gap = float(vertex[2] - surface["top_z_m"])
                if gap < -depth_limit - NUMERICAL_TOP_TOLERANCE_M:
                    raise RuntimeError("Target convex vertex penetrates the source upper surface too deeply")
                profiles.append({"vertex_world_m": vertex.tolist(), "gap_m": gap, **surface})
        evidence = []
        for contact in contacts:
            depth, normal, position = _contact_values(contact)
            interior = _source_contact_interior(position[:2], self._source_xy_projections[source])
            surface = upper_surface_at_xy(normals, offsets, position[:2])
            target_surface = upper_surface_at_xy(target_normals, target_offsets, position[:2])
            if (depth > depth_limit or surface is None
                    or target_surface is None
                    or surface["upper_face_normal_z"] < MIN_SUPPORT_NORMAL_Z
                    or abs(position[2] - surface["top_z_m"]) > depth_limit + NUMERICAL_TOP_TOLERANCE_M
                    or abs(position[2] - target_surface["bottom_z_m"]) > depth_limit + NUMERICAL_TOP_TOLERANCE_M
                    or abs(target_surface["bottom_z_m"] - surface["top_z_m"]) > depth_limit + NUMERICAL_TOP_TOLERANCE_M):
                raise RuntimeError("Contact is deep, sideways, outside, or below source upper support")
            # A fixed 2um neighbourhood confuses internal cooked seams with
            # an outer side touch. Require positive-area ACTUAL pair overlap
            # instead; pure outer corner/line contact still has zero area.
            # An unstable FCL normal is diagnostic only; the actual cooked
            # upper halfspaces above independently bound contact and vertices.
            evidence.append({"depth": depth, "normal": normal.tolist(), "position": position.tolist(),
                             "surface": surface, "target_vertical_profile": target_surface,
                             "actual_xy_support_overlap": overlap, "actual_contact_source_interior": interior})
        return {"target": target, "source": source, "source_actor": self.source_actor_name,
                "max_contact_depth_m": max(c["depth"] for c in evidence), "depth_limit_m": depth_limit,
                "raw_contacts": copy.deepcopy(contacts), "contact_geometry": evidence,
                "target_world_vertex_bounds_m": [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
                "overlapping_vertex_profiles": profiles}

    def _support_allowed(self, target_name: str, other_name: str, object_pose: np.ndarray,
                         contacts: list[dict[str, Any]]) -> bool:
        receipt = {"mode": "supplied_object_pose", "target": target_name, "source": other_name,
                   "source_actor": self.source_actor_name, "support_released": self.support_released,
                   "control_step": self._control_step(), "raw_contacts": copy.deepcopy(contacts),
                   "candidate_query_changes_release_state": False}
        try:
            if (not self._initial_support_exists or self.support_released or target_name not in self._target_names
                    or other_name not in self._source_names):
                raise RuntimeError("No active captured initial source-actor support")
            pose = _rigid_transform(object_pose)
            displacement = pose[:3, 3] - self._initial_object[:3, 3]
            reference_dz = float(pose[2, 3] - self._support_z_reference_m)
            rotation = pose[:3, :3] @ self._initial_object[:3, :3].T
            angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
            receipt.update(object_pose=pose.tolist(), displacement_m=displacement.tolist(), rotation_rad=angle,
                support_z_reference_m=self._support_z_reference_m,
                reference_vertical_displacement_m=reference_dz,
                captured_origin_z_m=float(self._initial_object[2, 3]),
                origin_vertical_reference_semantics="lowest_independently_validated_measured_origin_within_original_capture_band",
                candidate_negative_z_bound_m=-1e-6)
            if (reference_dz < -1e-6 or displacement[2] > self.support_band_m
                    or np.linalg.norm(displacement[:2]) > .001 or angle > .01):
                raise RuntimeError("Initial source-support motion bounds exceeded")
            receipt["geometry"] = self._contact_geometry(
                target_name, other_name, contacts, pose, self._initial_object_depth_m + self.support_depth_slop_m)
            receipt.update(status="allowed", reason="bounded_initial_source_upper_surface_separation")
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            receipt.update(status="rejected", reason=str(exc))
        self._validation_history.append(copy.deepcopy(receipt))
        for contact in contacts:
            contact["support_validation"] = copy.deepcopy(receipt)
        if hasattr(self, "coverage"):
            self.coverage["last_support_validation"] = copy.deepcopy(receipt)
        return receipt["status"] == "allowed"

    def _update_support_z_reference(self, objects: dict[str, Any], actual_object: np.ndarray) -> None:
        """Lower the origin reference only at the parent's real-measurement hook.

        This explicitly changes the old monotone capture-origin Z rule. All
        cooked top/contact proofs, depth-growth, XY/rotation and permanent
        release bounds still apply. The absolute original-capture 3 mm bound
        prevents a sequence of measurements from accumulating unlimited drift.
        A collision candidate never calls this method or mutates its reference.
        """
        actual_object = _rigid_transform(actual_object)
        old_reference = float(self._support_z_reference_m)
        if actual_object[2, 3] >= old_reference:
            return
        receipt: dict[str, Any] = {
            "mode": "real_measured_refresh_support_z_reference", "support_version": SUPPORT_VERSION,
            "source_actor": self.source_actor_name, "control_step": self._control_step(),
            "capture_object_pose_immutable": self._initial_object.tolist(),
            "capture_attachment_T_tcp_object_immutable": self._attachment.tolist() if hasattr(self, "_attachment") else None,
            "actual_object_pose": actual_object.tolist(), "reference_before_m": old_reference,
            "reference_after_m": old_reference, "support_region_released": bool(self.support_released),
            "maximum_total_reference_drop_from_capture_m": self.support_band_m,
            "candidate_query_can_update_reference": False,
            "origin_reference_rule_changed_from_v05": True}
        try:
            # Parent Held.refresh sets both members immediately after reading
            # the actual actor and before invoking the measured-only hook.
            # No candidate object dictionary is accepted as a measurement.
            if (objects is not getattr(self, "_measured_objects", None)
                    or actual_object is not getattr(self, "_measured_object_pose", None)):
                raise RuntimeError("Support Z reference requires the parent's actual measured-refresh context")
            if not self._initial_support_exists or self.support_released:
                raise RuntimeError("Support Z reference cannot open an absent or released initial region")
            displacement = actual_object[:3, 3] - self._initial_object[:3, 3]
            rotation = actual_object[:3, :3] @ self._initial_object[:3, :3].T
            angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
            limit = min(self.support_penetration_m,
                        self._initial_object_depth_m + self.support_depth_slop_m)
            receipt.update(displacement_from_capture_m=displacement.tolist(), rotation_from_capture_rad=angle,
                           depth_limit_m=limit, total_reference_drop_from_capture_m=float(-displacement[2]))
            if (not -self.support_band_m <= displacement[2] <= self.support_band_m
                    or np.linalg.norm(displacement[:2]) > .001 or angle > .01):
                raise RuntimeError("Measured support origin exceeds original capture band/XY/rotation bounds")
            contact_proofs, region_proofs, pair_queries = [], [], []
            receipt.update(actual_contact_geometry_proofs=contact_proofs,
                           independent_upper_region_proofs=region_proofs,
                           actual_source_pair_queries=pair_queries)
            for target in sorted(self._target_names):
                measured_shape = _object_transform(objects[target])
                if not np.allclose(actual_object @ self._shape_local[target], measured_shape, rtol=0, atol=2e-6):
                    raise RuntimeError("Measured source-support geometry differs from original shape-local capture")
                for source in sorted(self._source_names):
                    collided, contacts = _pair_result(objects[target], objects[source])
                    distance = float(fcl.distance(objects[target], objects[source], fcl.DistanceRequest()).min_distance)
                    pair_queries.append({"target": target, "source": source, "fcl_collision": bool(collided),
                                         "distance_m": distance, "raw_contacts": copy.deepcopy(contacts)})
                    if not np.isfinite(distance):
                        raise RuntimeError("Nonfinite measured source-support reference distance")
                    if collided:
                        contact_proofs.append(self._contact_geometry(target, source, contacts, actual_object, limit))
                    if distance <= self.support_band_m:
                        region = self._upper_region_geometry(target, source, actual_object)
                        region.update(fcl_distance_m=distance, fcl_collision=bool(collided), raw_contacts=copy.deepcopy(contacts))
                        if (region["status"] == "bounded_upper_region"
                                and region["minimum_vertical_gap_m"] >= -limit - NUMERICAL_TOP_TOLERANCE_M):
                            region_proofs.append(region)
            if not region_proofs:
                raise RuntimeError("Measured origin has no independently bounded current upper-support region")
            # Assign an absolute world Z, never add an incremental allowance.
            self._support_z_reference_m = float(actual_object[2, 3])
            receipt.update(status="updated", reason="actual_bounded_top_support_measurement",
                           reference_after_m=self._support_z_reference_m,
                           actual_contact_geometry_proofs=contact_proofs,
                           independent_upper_region_proofs=region_proofs)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            receipt.update(status="rejected", reason=str(exc))
        # Planner refresh can read one unchanged physical state repeatedly.
        # Preserve each distinct proof/state, without duplicate query growth.
        history = self._support_z_reference_history
        if not history or any(history[-1].get(k) != receipt.get(k) for k in (
                "control_step", "actual_object_pose", "reference_before_m", "reference_after_m", "status", "reason")):
            receipt["sequence"] = len(history) + 1
            history.append(copy.deepcopy(receipt))

    def _update_support_release(self, objects: dict[str, Any], actual_object: np.ndarray) -> None:
        before = self.support_released
        for source in self._source_names:
            if not np.allclose(_object_transform(objects[source]), self._source_transforms[source], rtol=0, atol=1e-6):
                raise RuntimeError("Captured static source support moved")
        distances, touching = [], []
        for target in sorted(self._target_names):
            for source in sorted(self._source_names):
                collided, contacts = _pair_result(objects[target], objects[source])
                distance = float(fcl.distance(objects[target], objects[source], fcl.DistanceRequest()).min_distance)
                if not np.isfinite(distance):
                    raise RuntimeError("Nonfinite source support separation distance")
                distances.append({"target": target, "source": source, "distance_m": distance,
                                  "fcl_collision": collided, "raw_contacts": contacts})
                if collided:
                    touching.append([target, source])
        all_clear = bool(not touching and distances
                         and all(d["distance_m"] > NUMERICAL_TOP_TOLERANCE_M for d in distances))
        # The numerical clearance fact is retained unchanged.  A cached native
        # contact manifold can outlive that tiny gap; origin displacement is
        # also insufficient for an offset, rotating fruit.  The original
        # support region closes only once EVERY actual convex target/source
        # distance exceeds the original physical support band.  This does not
        # change contact, penetration, motion, side or rim acceptance rules.
        physical_band_clear = bool(not touching and distances
                                   and all(d["distance_m"] > self.support_band_m for d in distances))
        if not before and (not self._initial_support_exists or physical_band_clear):
            self.support_released = True
        if not self.support_released:
            self._update_support_z_reference(objects, actual_object)
        self._state_receipt = {"mode": "measured_refresh", "source_actor": self.source_actor_name,
            "support_version": SUPPORT_VERSION,
            "reason": "already_released" if before else "no_initial_support" if not self._initial_support_exists
                      else "all_target_source_convex_pairs_exceed_initial_support_band" if physical_band_clear else "initial_source_support_active",
            "released_before": before, "released_after": self.support_released,
            "all_pieces_separated": all_clear, "separation_numeric_band_m": NUMERICAL_TOP_TOLERANCE_M,
            "geometry_all_pieces_clear": all_clear,
            "physical_support_band_clear": physical_band_clear, "physical_support_band_m": self.support_band_m,
            "physical_support_region_released": bool(self.support_released),
            "support_z_reference_m": self._support_z_reference_m,
            "native_physical_contact_lifecycle_measured_here": False,
            "release_rule": "measured_all_target_source_convex_distances_gt_original_support_band_AND_no_FCL_intersections",
            "control_step": self._control_step(), "actor_vertical_displacement_m": float(actual_object[2, 3] - self._initial_object[2, 3]),
            "pair_distances": distances, "fcl_touching_pairs": touching}
        # The latest refresh retains every pair distance. This transition
        # history stores only state/contact-set changes, like the table adapter,
        # so repeated planner snapshots do not grow quadratically with ticks.
        previous = self._state_history[-1] if self._state_history else None
        identity = (self._state_receipt["reason"], self.support_released, touching, all_clear, physical_band_clear)
        if previous is None or identity != (previous["reason"], previous["released_after"], previous["fcl_touching_pairs"],
                                            previous.get("geometry_all_pieces_clear"), previous.get("physical_support_band_clear")):
            self._state_history.append(copy.deepcopy(self._state_receipt))

    def refresh(self) -> dict[str, Any]:
        coverage = super().refresh()
        coverage.update(support_semantics="captured_source_actor_bounded_convex_upper_surface",
            support_adapter_version=SUPPORT_VERSION,
            geometry_all_pieces_clear=self._state_receipt.get("geometry_all_pieces_clear"),
            physical_support_band_clear=self._state_receipt.get("physical_support_band_clear"),
            physical_support_region_released=bool(self.support_released),
            source_actor=self.source_actor_name, source_shape_names=sorted(self._source_names),
            initial_object_support_exists=self._initial_support_exists,
            initial_object_support_max_depth_m=self._initial_object_depth_m,
            initial_object_support_receipts=copy.deepcopy(self._initial_receipts),
            initial_source_support_region_receipts=copy.deepcopy(self._initial_region_receipts),
            initial_source_support_region_queries=copy.deepcopy(self._initial_region_queries),
            registered_region_is_contact_assertion=False,
            support_z_reference_m=self._support_z_reference_m,
            support_z_reference_rule="lowest_independently_validated_real_measured_origin_within_original_capture_3mm_band",
            origin_vertical_reference_rule_changed_from_v05=True,
            support_z_reference_history=copy.deepcopy(self._support_z_reference_history),
            initial_support_geometry=copy.deepcopy(self._geometry_export),
            support_state=copy.deepcopy(self._state_receipt),
            support_state_transition_history=copy.deepcopy(self._state_history),
            support_validation_history=copy.deepcopy(self._validation_history),
            support_geometry_fingerprints={name: hashlib.sha256(value.tobytes()).hexdigest()
                                           for name, value in sorted(self._local_vertices.items())})
        return coverage


class SourceSupportDualPayloadCollisionChecker(DualPayloadCollisionChecker):
    """Original dual checker with source-plate adapters for its two attachments."""

    @classmethod
    def from_solvers(cls, env: Any, left_solver: Any, right_solver: Any, *, targets: tuple[Any, Any],
                     source_supports: tuple[Any, Any]) -> SourceSupportDualPayloadCollisionChecker:
        if (not isinstance(source_supports, (tuple, list)) or len(source_supports) != 2
                or _one_entity(source_supports[0]) == _one_entity(source_supports[1])):
            raise ValueError("Two distinct source actors in physical payload order are required")
        self = super().from_solvers(env, left_solver, right_solver, targets=targets)
        self.source_supports = tuple(source_supports)
        return self

    def attach(self) -> None:
        if self.attached:
            raise ValueError("Payload relations are already captured")
        supports = tuple(SourceSurfaceSupportCollisionChecker.from_live(
            live, target, active_robot=robot, support_actor=source)
            for live, target, robot, source in zip(self.lives, self.targets, ("left", "right"), self.source_supports))
        self._supports, self.attached = supports, True
        try:
            self.refresh()
        except Exception:
            self.attached, self._fresh = False, False
            raise


def _self_test() -> dict[str, Any]:
    """Actual SAPIEN cooked convex/FCL queries; no scene, env or physics steps."""
    from scipy.spatial import ConvexHull
    from transforms3d.euler import euler2mat

    tests = []
    material = physx.PhysxMaterial(.5, .5, 0)
    source_vertices = np.array([[x, y, z] for x in (-.1, .1) for y in (-.1, .1) for z in (-.02, 0)], dtype=np.float32)
    target_vertices = np.array([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (0, .02)], dtype=np.float32)

    def fixture(z: float = -2e-5, *, x: float = 0.) -> tuple[Any, dict[str, Any]]:
        source_entity, target_entity = sapien.Entity(), sapien.Entity()
        source_entity.name, target_entity.name = "plate_left", "fruit_a"
        source_shape = physx.PhysxCollisionShapeConvexMesh(source_vertices, [1., 1., 1.], material)
        target_shape = physx.PhysxCollisionShapeConvexMesh(target_vertices, [1., 1., 1.], material)
        body = physx.PhysxRigidStaticComponent(); body.attach(source_shape); source_entity.add_component(body)
        checker = SourceSurfaceSupportCollisionChecker()
        checker._source_entity, checker._entity = source_entity, target_entity
        checker._target_names = frozenset(("fruit",)); checker._shape_local = {"fruit": np.eye(4)}
        checker._initial_object = np.eye(4); checker._initial_object[:3, 3] = [x, 0, z]
        checker.support_band_m = checker.support_penetration_m = .003; checker.support_depth_slop_m = .0002
        checker._support_pairs = {}; checker.support_released = False
        checker.live = NS(_entities={"plate_left": source_entity, "fruit_a": target_entity},
                          _shape_refs={"fruit": (target_entity, target_shape), "source": (source_entity, source_shape)})
        source_geo = fcl.Convex(np.asarray(source_shape.vertices, float), np.asarray(source_shape.get_triangles(), np.int32), True)
        target_geo = fcl.Convex(np.asarray(target_shape.vertices, float), np.asarray(target_shape.get_triangles(), np.int32), True)
        objects = {"fruit": _object_at(target_geo, checker._initial_object), "source": _object_at(source_geo, np.eye(4))}
        checker._register_initial_support(objects); checker.coverage = {}
        return checker, objects

    checker, objects = fixture()
    assert checker._initial_support_exists and len(checker._support_pairs) == 1
    collided, contacts = _pair_result(objects["fruit"], objects["source"])
    assert collided and contacts and checker._support_allowed("fruit", "source", checker._initial_object, contacts)
    tests.append("actual_cooked_convex_initial_support_accepted_with_original_contacts")
    zero_normal = [{"depth": contacts[0]["depth"], "normal": [0., 0, 0], "position": contacts[0]["position"]}]
    assert checker._support_allowed("fruit", "source", checker._initial_object, zero_normal)
    assert zero_normal[0]["normal"] == [0., 0, 0]
    tests.append("numerical_fcl_normal_preserved_with_independent_upper_surface_proof")
    attachment = np.eye(4); checker._attachment = attachment.copy()
    initial_state = checker.support_released
    pose = checker._initial_object.copy(); pose[2, 3] -= .004
    collided, contacts = _pair_result(_object_at(objects["fruit"].get_collision_geometry(), pose), objects["source"])
    assert collided and not checker._support_allowed("fruit", "source", pose, contacts)
    assert checker.support_released == initial_state and np.array_equal(checker._attachment, attachment)
    tests.append("candidate_deep_penetration_rejected_without_release_or_attachment_mutation")
    for offset, label in (([.0011, 0, 0], "lateral"), ([0, 0, -.0001], "downward")):
        pose = checker._initial_object.copy(); pose[:3, 3] += offset
        _, contacts = _pair_result(_object_at(objects["fruit"].get_collision_geometry(), pose), objects["source"])
        assert not checker._support_allowed("fruit", "source", pose, contacts)
        tests.append(label + "_bound_rejected")
    pose = checker._initial_object.copy(); pose[:3, :3] = euler2mat(.02, 0, 0)
    _, contacts = _pair_result(_object_at(objects["fruit"].get_collision_geometry(), pose), objects["source"])
    assert not checker._support_allowed("fruit", "source", pose, contacts)
    tests.append("rotation_bound_rejected")
    assert not checker._support_allowed("fruit", "unrelated_plate", checker._initial_object, contacts)
    tests.append("unrelated_actor_not_allowlisted")
    checker._update_support_release(objects, checker._initial_object)
    assert not checker.support_released
    # A rotating offset object can lift its origin by >3mm while a low piece
    # remains within 3mm of its source. Numeric separation must be a distinct
    # reported fact, never the irreversible region-release trigger.
    near = checker._initial_object.copy(); near[2, 3] += .004
    near[:3, :3] = euler2mat(0., .2, 0.)
    objects["fruit"] = _object_at(objects["fruit"].get_collision_geometry(), near)
    checker._update_support_release(objects, near)
    assert checker._state_receipt["actor_vertical_displacement_m"] > checker.support_band_m
    assert checker._state_receipt["geometry_all_pieces_clear"]
    assert not checker._state_receipt["physical_support_band_clear"] and not checker.support_released
    tests.append("origin_lift_over_3mm_numeric_clear_but_actual_piece_within_3mm_region_stays_active")
    # Distinct convex pieces must ALL exceed the band; another elevated piece
    # cannot hide the near-source piece. These are private FCL test objects.
    checker._target_names = frozenset(("fruit", "fruit_second"))
    higher = near.copy(); higher[2, 3] += .01
    objects["fruit_second"] = _object_at(objects["fruit"].get_collision_geometry(), higher)
    checker._update_support_release(objects, near)
    assert checker._state_receipt["geometry_all_pieces_clear"] and not checker.support_released
    assert min(p["distance_m"] for p in checker._state_receipt["pair_distances"]) < checker.support_band_m
    tests.append("all_target_source_convex_pairs_must_exceed_original_physical_band")
    checker._target_names = frozenset(("fruit",)); del objects["fruit_second"]
    lifted = checker._initial_object.copy(); lifted[2, 3] += .01
    objects["fruit"] = _object_at(objects["fruit"].get_collision_geometry(), lifted)
    checker._update_support_release(objects, lifted)
    assert checker.support_released and checker._state_receipt["all_pieces_separated"]
    objects["fruit"] = _object_at(objects["fruit"].get_collision_geometry(), checker._initial_object)
    checker._update_support_release(objects, checker._initial_object)
    _, contacts = _pair_result(objects["fruit"], objects["source"])
    assert checker.support_released and not checker._support_allowed("fruit", "source", checker._initial_object, contacts)
    tests.append("measured_all_pair_separation_permanently_closes_support")
    try:
        fixture(-.004)
    except RuntimeError:
        tests.append("initial_deep_penetration_rejected")
    else:
        raise AssertionError("Initial deep penetration was accepted")
    checker, objects = fixture(.002)
    assert checker._initial_support_exists and not checker._support_pairs
    assert checker._initial_region_receipts and not checker._initial_receipts
    checker._update_support_release(objects, checker._initial_object)
    assert not checker.support_released and checker._state_receipt["geometry_all_pieces_clear"]
    assert not checker._state_receipt["physical_support_band_clear"]
    assert all(r["registration_does_not_assert_contact"] and not r["actual_fcl_contacts_present"]
               for r in checker._initial_region_receipts)
    tests.append("positive_gap_within_original_3mm_upper_region_registered_without_inventing_contact")
    checker, objects = fixture(5e-7)
    assert checker._initial_support_exists and not checker._support_pairs
    checker._update_support_release(objects, checker._initial_object)
    assert not checker.support_released and not checker._state_receipt["geometry_all_pieces_clear"]
    micro_settle = checker._initial_object.copy(); micro_settle[2, 3] -= 7.5e-7
    collided, contacts = _pair_result(_object_at(objects["fruit"].get_collision_geometry(), micro_settle), objects["source"])
    assert collided and checker._support_allowed("fruit", "source", micro_settle, contacts)
    assert not checker.support_released
    tests.append("submicrometre_capture_gap_keeps_region_for_real_shallow_contact_within_original_motion_bound")
    checker, objects = fixture(.00301)
    assert not checker._initial_support_exists
    checker._update_support_release(objects, checker._initial_object)
    assert checker.support_released
    tests.append("outside_original_3mm_region_cannot_create_future_allowance")
    checker, objects = fixture(-.0405)
    assert not checker._initial_support_exists and not checker._support_pairs
    tests.append("near_source_underside_does_not_register_upper_support")
    checker, objects = fixture(5e-7, x=.111)
    assert not checker._initial_support_exists and not checker._support_pairs
    tests.append("near_lateral_source_gap_does_not_register_upper_support")
    normals, offsets = convex_planes(source_vertices, ConvexHull(source_vertices).simplices)
    assert upper_surface_at_xy(normals, offsets, [0, 0])["top_z_m"] == 0
    assert upper_surface_at_xy(normals, offsets, [.2, 0]) is None
    side_contact = [{"depth": 1e-5, "normal": [1., 0, 0], "position": [.1, 0, -.01]}]
    checker, objects = fixture()
    assert not checker._support_allowed("fruit", "source", checker._initial_object, side_contact)
    tests.append("side_contact_below_actual_upper_surface_rejected")
    corner_contact = [{"depth": 0., "normal": [1., 0, 0], "position": [.1, 0, 0]}]
    corner_pose = checker._initial_object.copy(); corner_pose[0, 3] = .11
    try:
        checker._contact_geometry("fruit", "source", corner_contact, corner_pose, .0002)
    except RuntimeError:
        tests.append("horizontal_outer_rim_corner_touch_not_misclassified_as_upper_support")
    else:
        raise AssertionError("Outer source rim corner was accepted as support")
    # The contact can sit within 1um of a cooked decomposition seam while the
    # two actual projected solids overlap over a substantial top-support area.
    seam_contact = [{"depth": 1e-7, "normal": [0., 0, 0.], "position": [0., .1 - .9e-6, 0.]}]
    seam_pose = checker._initial_object.copy(); seam_pose[1, 3] = .095
    proof = checker._contact_geometry("fruit", "source", seam_contact, seam_pose, .0002)
    overlap = proof["contact_geometry"][0]["actual_xy_support_overlap"]
    assert overlap["positive_area"] and overlap["area_m2"] > 1e-4
    assert not overlap["source_union_filled_or_geometry_replaced"]
    tests.append("near_seam_contact_with_actual_positive_area_top_support_needs_no_fixed_2um_neighbourhood")
    # A real FCL contact at an outer rim remains invalid even when an inward
    # overhang gives the WHOLE body's XY projection a large positive area.
    # This is a contact-location condition, not a contact-patch-area demand:
    # ordinary fruit point contact strictly inside the source still qualifies.
    rim_source = np.array([[x, y, z] for x in (-.1, .1) for y in (-.1, .1)
                           for z in (-.02, 0.)], dtype=np.float64)
    overhang = np.array([[.1, 0., 0.], [.05, -.03, .02],
                        [.05, .03, .02], [.03, 0., .03]], dtype=np.float64)
    source_faces = ConvexHull(rim_source).simplices.astype(np.int32)
    target_faces = ConvexHull(overhang).simplices.astype(np.int32)
    rim_source_object = _object_at(fcl.Convex(rim_source, source_faces, True), np.eye(4))
    overhang_object = _object_at(fcl.Convex(overhang, target_faces, True), np.eye(4))
    rim_hit, rim_contacts = _pair_result(overhang_object, rim_source_object)
    assert rim_hit and rim_contacts
    rim_check = SourceSurfaceSupportCollisionChecker()
    rim_check.source_actor_name = "outer_rim_fixture"
    rim_check._source_planes = {"source": convex_planes(rim_source, source_faces)}
    rim_check._target_local_planes = {"overhang": convex_planes(overhang, target_faces)}
    rim_check._source_xy_projections = {"source": _xy_projection(rim_source)}
    rim_check._local_vertices = {"overhang": overhang}
    rim_check._shape_local = {"overhang": np.eye(4)}
    projected = _positive_xy_overlap(_xy_projection(overhang), _xy_projection(rim_source))
    assert np.isclose(projected["area_m2"], .0021, rtol=0, atol=1e-15)
    raw_rim_contacts = copy.deepcopy(rim_contacts)
    try:
        rim_check._contact_geometry("overhang", "source", rim_contacts, np.eye(4), .0002)
    except RuntimeError as exc:
        assert "actual projected interior" in str(exc)
    else:
        raise AssertionError("Outer rim point with inward body overhang was accepted")
    assert rim_contacts == raw_rim_contacts
    tests.append("native_FCL_outer_rim_tip_rejected_despite_positive_inward_overhang_projection")
    polygon = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    for shift, label in (([1., 0.], "line"), ([1., 1.], "corner"), ([2., 0.], "disjoint")):
        try:
            _positive_xy_overlap(polygon + shift, polygon)
        except RuntimeError:
            tests.append("actual_projection_" + label + "_has_no_positive_support_area")
        else:
            raise AssertionError("Degenerate projected support overlap was accepted")
    # A sloped support surface is evaluated in world coordinates, with actual
    # mesh scale and pose already applied to the captured vertices.
    rotation = euler2mat(0, .15, 0)
    tilted_vertices = source_vertices.astype(float) @ rotation.T + [.3, -.1, .2]
    tilted_a, tilted_b = convex_planes(tilted_vertices, ConvexHull(source_vertices).simplices)
    center = upper_surface_at_xy(tilted_a, tilted_b, [.3, -.1])
    assert center is not None and abs(center["top_z_m"] - .2) < 1e-8
    assert abs(center["upper_face_normal_z"] - np.cos(.15)) < 1e-8
    tests.append("scaled_transformed_sloped_convex_surface_uses_world_halfspaces")
    moved = dict(objects); moved_pose = np.eye(4); moved_pose[0, 3] = .001
    moved["source"] = _object_at(objects["source"].get_collision_geometry(), moved_pose)
    try:
        checker._update_support_release(moved, checker._initial_object)
    except RuntimeError:
        tests.append("captured_static_source_motion_rejected")
    else:
        raise AssertionError("Moved source actor was accepted")
    # More than one convex payload piece must clear before release.
    checker._target_names = frozenset(("fruit", "fruit_second"))
    objects["fruit_second"] = objects["fruit"]
    objects["fruit"] = _object_at(objects["fruit"].get_collision_geometry(), lifted)
    checker._update_support_release(objects, lifted)
    assert not checker.support_released and checker._state_receipt["fcl_touching_pairs"]
    objects["fruit_second"] = objects["fruit"]
    checker._update_support_release(objects, lifted)
    assert checker.support_released
    tests.append("all_payload_convex_pieces_required_for_release")

    def measured_fixture_refresh(check: Any, previous_objects: dict[str, Any],
                                 measured_pose: np.ndarray) -> dict[str, Any]:
        """CPU replay of the parent's measured-hook contract, not physics."""
        current = {**previous_objects, **{name: _object_at(previous_objects[name].get_collision_geometry(),
                    measured_pose @ check._shape_local[name]) for name in check._target_names}}
        check._measured_objects = current
        check._measured_object_pose = measured_pose
        check._update_support_release(current, measured_pose)
        return current

    checker, objects = fixture(5e-7)
    checker._attachment = np.eye(4); checker.env = NS(elapsed_steps=np.array([68]))
    immutable_capture = checker._initial_object.copy()
    immutable_attachment = checker._attachment.copy()
    immutable_local = checker._shape_local["fruit"].copy()
    reference0 = checker._support_z_reference_m
    settled = checker._initial_object.copy(); settled[2, 3] -= 7.5e-5
    collided, contacts = _pair_result(_object_at(objects["fruit"].get_collision_geometry(), settled), objects["source"])
    assert collided and contacts
    assert not checker._support_allowed("fruit", "source", settled, copy.deepcopy(contacts))
    assert checker._support_z_reference_m == reference0 and not checker._support_z_reference_history
    tests.append("candidate_cannot_lower_reference_or_accept_75um_downward_query")
    checker._update_support_z_reference(objects, settled)
    assert checker._support_z_reference_m == reference0
    assert checker._support_z_reference_history[-1]["status"] == "rejected"
    assert "measured-refresh context" in checker._support_z_reference_history[-1]["reason"]
    tests.append("unbound_fake_measurement_cannot_update_reference")
    current = measured_fixture_refresh(checker, objects, settled)
    assert checker._support_z_reference_m == settled[2, 3] and not checker.support_released
    allowed = checker._support_allowed("fruit", "source", settled, copy.deepcopy(contacts))
    assert allowed
    update = checker._support_z_reference_history[-1]
    assert update["status"] == "updated" and update["control_step"] == 68
    assert update["actual_contact_geometry_proofs"] and update["independent_upper_region_proofs"]
    assert update["reference_before_m"] == reference0 and update["reference_after_m"] == settled[2, 3]
    assert np.array_equal(checker._initial_object, immutable_capture)
    assert np.array_equal(checker._attachment, immutable_attachment)
    assert np.array_equal(checker._shape_local["fruit"], immutable_local)
    assert [r["sequence"] for r in checker._support_z_reference_history] == list(range(1, len(checker._support_z_reference_history) + 1))
    json.dumps(checker._support_z_reference_history, allow_nan=False)
    tests.append("bounded_real_measured_settling_updates_only_scalar_reference_with_complete_serializable_receipt")
    old_count = len(checker._support_z_reference_history)
    for _ in range(4):
        current = measured_fixture_refresh(checker, current, settled)
        assert checker._support_allowed("fruit", "source", settled, copy.deepcopy(contacts))
    assert checker._support_z_reference_m == settled[2, 3] and len(checker._support_z_reference_history) == old_count
    tests.append("repeated_identical_measurements_and_candidate_queries_cannot_accumulate_reference_drift")
    lower_candidate = settled.copy(); lower_candidate[2, 3] -= 2e-6
    _, candidate_contacts = _pair_result(_object_at(objects["fruit"].get_collision_geometry(), lower_candidate), objects["source"])
    assert not checker._support_allowed("fruit", "source", lower_candidate, candidate_contacts)
    assert checker._support_z_reference_m == settled[2, 3] and len(checker._support_z_reference_history) == old_count
    tests.append("negative_candidate_dz_bound_stays_1um_relative_to_validated_measured_reference")
    # An actual origin argument inconsistent with the measured object geometry
    # cannot gain a new reference even if both context attributes are supplied.
    inconsistent_measurement = settled.copy(); inconsistent_measurement[2, 3] -= 2e-5
    checker._measured_object_pose = inconsistent_measurement
    checker._update_support_z_reference(current, inconsistent_measurement)
    assert checker._support_z_reference_m == settled[2, 3]
    assert "shape-local capture" in checker._support_z_reference_history[-1]["reason"]
    tests.append("measured_pose_inconsistent_with_actual_cooked_shape_rejected")
    # Total movement always uses immutable original capture, never the last
    # small step. The original +0.2 mm depth-growth guard also remains active.
    for dz, label in ((-.00025, "depth_growth"), (-.00301, "absolute_original_3mm_band")):
        invalid = checker._initial_object.copy(); invalid[2, 3] += dz
        before_ref = checker._support_z_reference_m
        measured_fixture_refresh(checker, current, invalid)
        assert checker._support_z_reference_m == before_ref
        assert checker._support_z_reference_history[-1]["status"] == "rejected"
        tests.append("measured_" + label + "_rejected_without_reference_reset")
    for kind in ("xy_drag", "rotation"):
        invalid = settled.copy(); invalid[2, 3] -= 5e-7
        if kind == "xy_drag":
            invalid[0, 3] += .00101
        else:
            invalid[:3, :3] = euler2mat(.0101, 0., 0.)
        measured_fixture_refresh(checker, current, invalid)
        assert checker._support_z_reference_m == settled[2, 3]
        assert checker._support_z_reference_history[-1]["status"] == "rejected"
        tests.append("measured_" + kind + "_cannot_reset_reference")
    released_pose = checker._initial_object.copy(); released_pose[2, 3] += .01
    current = measured_fixture_refresh(checker, current, released_pose)
    assert checker.support_released
    current = measured_fixture_refresh(checker, current, settled)
    assert checker.support_released and checker._support_z_reference_m == settled[2, 3]
    assert not checker._support_allowed("fruit", "source", settled, copy.deepcopy(contacts))
    tests.append("return_to_released_source_never_reopens_region_or_reference")
    return {"status": "passed", "support_version": SUPPORT_VERSION, "tests": tests, "test_count": len(tests),
            "environment_creations": 0, "physics_steps": 0, "standalone_fcl_queries": True,
            "native_validation": "NOT_RUN", "numerical_top_tolerance_m": NUMERICAL_TOP_TOLERANCE_M,
            "physical_support_band_m": checker.support_band_m}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("use --self-test")
    print(json.dumps(_self_test(), indent=2))
