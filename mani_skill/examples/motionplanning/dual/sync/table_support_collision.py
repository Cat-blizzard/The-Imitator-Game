"""Bounded whole-object tabletop support, without changing contact physics.

A held convex object may transfer initial tabletop support between its convex
pieces while it separates upward. Every exception is independently bounded by
scaled convex vertices, the horizontal table Box, the original motion/depth
limits, and a permanent measured-only release state. Near-coplanar contacts
within the existing 1 micrometre bound do not rely on an unstable FCL normal.
Candidate vertices come from the supplied object pose, never the live actor.
Version 2 distinguishes the original 3 mm initial support region from a
numerical contact. Region registration never invents an FCL contact; measured
separation permanently closes it only after every piece clears that same band.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl

from .held_collision import AttachedObjectCollisionChecker, _object_transform, _pair_result

NUMERICAL_TOP_TOLERANCE_M = 1e-6
ZERO_NORMAL_TOLERANCE = 1e-8  # Diagnostic compatibility; no longer a support prerequisite.
TABLE_NAME = "scene::table::shape_0"
TABLE_SUPPORT_VERSION = "captured_table_physical_support_region_v02"


def _contact_values(contact: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        depth = float(contact["depth"])
        normal = np.asarray(contact["normal"], dtype=np.float64)
        position = np.asarray(contact["position"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Incomplete tabletop contact data") from exc
    if (normal.shape != (3,) or position.shape != (3,)
            or not np.isfinite(np.r_[depth, normal, position]).all() or depth < 0):
        raise RuntimeError("Invalid tabletop contact data")
    return depth, normal, position


def _scaled_vertices(live: Any, name: str) -> np.ndarray:
    try:
        _, shape = live._shape_refs[name]
        if not isinstance(shape, physx.PhysxCollisionShapeConvexMesh):
            raise RuntimeError("Tabletop support requires actual SAPIEN convex shapes")
        vertices = np.asarray(shape.vertices, dtype=np.float64)
        scale = np.asarray(shape.scale, dtype=np.float64)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Actual target convex geometry unavailable") from exc
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4
            or scale.shape != (3,) or not np.isfinite(vertices).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise RuntimeError("Invalid actual target convex geometry")
    return vertices * scale


def _rigid_transform(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, rtol=0, atol=1e-5)):
        raise RuntimeError("Invalid rigid tabletop/object transform")
    return transform


def _vertices_at(vertices: np.ndarray, transform: Any) -> np.ndarray:
    pose = _rigid_transform(transform)
    vertices = np.asarray(vertices, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4
            or not np.isfinite(vertices).all()):
        raise RuntimeError("Invalid convex vertices")
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def _target_world_vertices(live: Any, name: str) -> np.ndarray:
    """Read-only diagnostic helper; candidate support never calls this helper."""
    vertices = _scaled_vertices(live, name)
    entity, shape = live._shape_refs[name]
    return _vertices_at(vertices, (entity.pose * shape.get_local_pose()).to_transformation_matrix())


def _vertex_receipt(name: str, vertices: Any, table_pose: Any, table_side: Any) -> dict[str, Any]:
    """Summarize actual convex vertices against a finite horizontal tabletop."""
    vertices = np.asarray(vertices, dtype=np.float64)
    table_pose = _rigid_transform(table_pose)
    side = np.asarray(table_side, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4
            or side.shape != (3,) or not np.isfinite(vertices).all()
            or not np.isfinite(side).all() or np.any(side <= 0)
            or not np.allclose(table_pose[:3, 2], [0, 0, 1], rtol=0, atol=1e-6)):
        raise RuntimeError("Support requires finite convex vertices and a horizontal tabletop")
    local = np.linalg.solve(table_pose[:3, :3], (vertices - table_pose[:3, 3]).T).T
    top = float(table_pose[2, 3] + side[2] / 2)
    gap = float(vertices[:, 2].min() - top)
    return {"target": name, "table": TABLE_NAME, "vertex_count": len(vertices),
            "world_vertex_bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            "table_local_vertex_xy_bounds_m": [local[:, :2].min(axis=0).tolist(), local[:, :2].max(axis=0).tolist()],
            "table_local_half_size_xy_m": (side[:2] / 2).tolist(), "table_top_z_m": top,
            "bottom_minus_table_top_m": gap, "vertex_penetration_m": max(0., -gap),
            "vertices_above_top": bool(vertices[:, 2].max() > top),
            "entire_footprint_inside": bool(np.all(local[:, :2] >= -side[:2] / 2)
                                            and np.all(local[:, :2] <= side[:2] / 2))}


def _numerical_top_receipt(name: str, contact: dict[str, Any], vertices: np.ndarray,
                           table_pose: np.ndarray, table_side: np.ndarray) -> dict[str, Any]:
    """Prove a near-plane top touch independently of the raw FCL normal."""
    depth, normal, position = _contact_values(contact)
    receipt = _vertex_receipt(name, vertices, table_pose, table_side)
    if (depth > NUMERICAL_TOP_TOLERANCE_M
            or abs(position[2] - receipt["table_top_z_m"]) > NUMERICAL_TOP_TOLERANCE_M
            or abs(receipt["bottom_minus_table_top_m"]) > NUMERICAL_TOP_TOLERANCE_M
            or not receipt["entire_footprint_inside"] or not receipt["vertices_above_top"]):
        raise RuntimeError(f"Not an independently verified numerical tabletop touch: {name}")
    return {**receipt, "classification": "numerical_top_touch",
            "raw_contact": {"depth": depth, "normal": normal.tolist(), "position": position.tolist()},
            "normal_norm": float(np.linalg.norm(normal)), "tolerance_m": NUMERICAL_TOP_TOLERANCE_M}


class NumericalTableSupportCollisionChecker(AttachedObjectCollisionChecker):
    """One initial support region for all convex pieces of a single held actor."""

    def _table(self, objects: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if TABLE_NAME not in objects or not isinstance(objects[TABLE_NAME].get_collision_geometry(), fcl.Box):
            raise RuntimeError("Tabletop support requires the known table Box")
        pose = _object_transform(objects[TABLE_NAME])
        side = np.asarray(objects[TABLE_NAME].get_collision_geometry().side, dtype=float).reshape(3)
        # Validate even when no current target contact exists.
        _vertex_receipt("table-validation", np.array([[0., 0, 0]] * 4), pose, side)
        return pose, side

    def _world_vertices(self, name: str, object_pose: Any) -> np.ndarray:
        """Both candidate and measured callers supply their own object transform."""
        if name not in self._local_vertices or name not in self._shape_local:
            raise RuntimeError("Unknown target convex piece")
        return _vertices_at(self._local_vertices[name], _rigid_transform(object_pose) @ self._shape_local[name])

    def _pieces(self, object_pose: Any) -> list[dict[str, Any]]:
        return [_vertex_receipt(name, self._world_vertices(name, object_pose), self._table_pose, self._table_side)
                for name in sorted(self._target_names)]

    def _validate_contact_geometry(self, name: str, contacts: list[dict[str, Any]],
                                   object_pose: Any, depth_limit: float) -> dict[str, Any]:
        if not contacts:
            raise RuntimeError("Missing support contacts")
        receipt = _vertex_receipt(name, self._world_vertices(name, object_pose), self._table_pose, self._table_side)
        if (not receipt["entire_footprint_inside"] or not receipt["vertices_above_top"]
                or receipt["bottom_minus_table_top_m"] > NUMERICAL_TOP_TOLERANCE_M
                or receipt["vertex_penetration_m"] > depth_limit):
            raise RuntimeError("Convex geometry is not bounded top support")
        values = [_contact_values(contact) for contact in contacts]
        top = receipt["table_top_z_m"]
        for depth, _, position in values:
            local = np.linalg.solve(self._table_pose[:3, :3], position - self._table_pose[:3, 3])
            if (depth > depth_limit or abs(position[2] - top) > self.support_penetration_m
                    or np.any(local[:2] < -self._table_side[:2] / 2)
                    or np.any(local[:2] > self._table_side[:2] / 2)):
                raise RuntimeError("Contact is not bounded tabletop support")
        return {**receipt, "max_contact_depth_m": max(v[0] for v in values), "depth_limit_m": depth_limit,
                "raw_contacts": [{"depth": d, "normal": n.tolist(), "position": p.tolist()} for d, n, p in values]}

    def _capture_initial_geometry(self) -> None:
        """Build fixed registration evidence; live guards still validate geometry.

        The captured vertices and shape-local transforms belong to this support
        registration. Refresh measures poses and support state, while the live
        checker continues checking actual shape identity and mesh/scale changes.
        """
        self._initial_geometry_export = {
            name: {"scaled_shape_local_vertices": vertices.tolist(), "shape_local_T": self._shape_local[name].tolist()}
            for name, vertices in sorted(self._local_vertices.items())}
        self._initial_geometry_fingerprints = {
            name: hashlib.sha256(vertices.tobytes()).hexdigest()
            for name, vertices in sorted(self._local_vertices.items())}

    def _register_initial_support(self, objects: dict[str, Any]) -> None:
        self.numerical_top_support_receipts: list[dict[str, Any]] = []
        self._local_vertices: dict[str, np.ndarray] = {}
        self._initial_geometry_export: dict[str, Any] = {}
        self._initial_geometry_fingerprints: dict[str, str] = {}
        self._legacy_support_mode: str | None = None
        self._initial_support_exists = False
        self._initial_object_depth_m = 0.
        self._initial_support_receipts: list[dict[str, Any]] = []
        self._initial_near_top_support_receipts: list[dict[str, Any]] = []
        self._support_state_receipt: dict[str, Any] = {"mode": "initial_registration", "reason": "no_initial_support"}
        self._last_support_validation: dict[str, Any] | None = None
        self._support_validation_history: list[dict[str, Any]] = []
        self._support_state_history: list[dict[str, Any]] = []
        self._table_pose = self._table_side = None
        if TABLE_NAME not in objects:
            self._legacy_support_mode = "legacy_missing_table"
        elif not all(isinstance(self.live._shape_refs[name][1], physx.PhysxCollisionShapeConvexMesh)
                     for name in self._target_names):
            self._legacy_support_mode = "legacy_nonconvex_support"
        if self._legacy_support_mode is not None:
            # Preserve ordinary sphere/primitive support and no-table worlds.
            # They receive exactly the base's original pair-specific exception,
            # never the new whole-convex-object transfer allowance.
            AttachedObjectCollisionChecker._register_initial_support(self, objects)
            self._initial_support_exists = bool(self._support_pairs)
            self._initial_object_depth_m = max((value["initial_depth"] for value in self._support_pairs.values()), default=0.)
            self._support_state_receipt.update(reason=self._legacy_support_mode)
            return
        self._local_vertices = {name: _scaled_vertices(self.live, name).copy() for name in self._target_names}
        self._capture_initial_geometry()
        self._table_pose, self._table_side = self._table(objects)
        initial_pieces = self._pieces(self._initial_object)
        pieces_by_name = {piece["target"]: piece for piece in initial_pieces}
        self._initial_object_depth_m = max(p["vertex_penetration_m"] for p in initial_pieces)
        for name in sorted(self._target_names):
            collision, contacts = _pair_result(objects[name], objects[TABLE_NAME])
            if not collision:
                piece = pieces_by_name[name]
                if (0. <= piece["bottom_minus_table_top_m"] <= self.support_band_m
                        and piece["entire_footprint_inside"] and piece["vertices_above_top"]):
                    # Capture the existing physical support region, not a
                    # claim of touching. This is deliberately distinct from
                    # the 1um test for a numerical contact below.
                    self._initial_near_top_support_receipts.append({**piece,
                        "classification": "geometry_only_initial_near_top_support",
                        "fcl_collision": False, "registration_band_m": self.support_band_m,
                        "registration_does_not_assert_contact": True,
                        "numerical_contact_tolerance_m": NUMERICAL_TOP_TOLERANCE_M,
                        "support_version": TABLE_SUPPORT_VERSION})
                continue
            receipt = self._validate_contact_geometry(name, contacts, self._initial_object, self.support_penetration_m)
            for contact in contacts:
                _, normal, _ = _contact_values(contact)
                if abs(normal[2]) < .99:
                    self.numerical_top_support_receipts.append(_numerical_top_receipt(
                        name, contact, self._world_vertices(name, self._initial_object), self._table_pose, self._table_side))
            depth = receipt["max_contact_depth_m"]
            self._support_pairs[(name, TABLE_NAME)] = {"initial_depth": depth, "contacts": copy.deepcopy(contacts)}
            self._initial_object_depth_m = max(self._initial_object_depth_m, depth)
            self._initial_support_receipts.append(receipt)
        self._initial_support_exists = bool(self._support_pairs or self._initial_near_top_support_receipts)
        if self._initial_support_exists and any(not p["entire_footprint_inside"] or not p["vertices_above_top"]
                                                or p["vertex_penetration_m"] > self.support_penetration_m
                                                for p in initial_pieces):
            raise RuntimeError("Initial whole object is not contained in the tabletop support region")
        self._support_state_receipt.update(reason="initial_object_support" if self._initial_support_exists else "no_initial_support",
                                          initial_pieces=initial_pieces)

    def _control_step(self) -> int | None:
        """Read optional telemetry without stepping or inferring a query mode."""
        raw = getattr(self, "env", None)
        raw = getattr(raw, "unwrapped", raw)
        value = getattr(raw, "elapsed_steps", None)
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.size != 1 or not np.isfinite(array).all():
            raise RuntimeError("Invalid support control-step telemetry")
        return int(array.reshape(-1)[0])

    def _update_support_release(self, objects: dict[str, Any], actual_object: np.ndarray) -> None:
        """Measured-only monotone transition, considering all actual convex pieces.

        Numerical clearance is telemetry. Region release requires every
        actual piece to clear the original physical support band with no FCL
        intersections. Origin motion alone is insufficient for a rotating,
        offset convex piece. Candidates never advance this permanent state.
        """
        before = self.support_released
        if self._legacy_support_mode is not None:
            AttachedObjectCollisionChecker._update_support_release(self, objects, actual_object)
            self._support_state_receipt = {"mode": "measured_refresh", "semantics": self._legacy_support_mode,
                "reason": "legacy_exact_pair_released" if self.support_released else "legacy_exact_pair_state",
                "released_before": before, "released_after": self.support_released,
                "control_step": self._control_step(), "fcl_touching_pieces": []}
            if not self._support_state_history or self._support_state_history[-1]["released_after"] != self.support_released:
                self._support_state_history.append(copy.deepcopy(self._support_state_receipt))
            return
        if self._table_pose is None:
            self._support_state_receipt = {"mode": "measured_refresh", "reason": "no_table", "released_before": before,
                                           "released_after": self.support_released}
            return
        self._table_pose, self._table_side = self._table(objects)
        pieces = self._pieces(actual_object)
        touching = [name for name in sorted(self._target_names) if _pair_result(objects[name], objects[TABLE_NAME])[0]]
        displacement_z = float(actual_object[2, 3] - self._initial_object[2, 3])
        all_clear = bool(all(p["bottom_minus_table_top_m"] > NUMERICAL_TOP_TOLERANCE_M for p in pieces) and not touching)
        physical_band_clear = bool(all(p["bottom_minus_table_top_m"] > self.support_band_m for p in pieces)
                                   and not touching)
        reason = "already_released" if before else "object_support_active"
        if not self._initial_support_exists:
            reason = "no_initial_support"
        elif not before and physical_band_clear:
            self.support_released = True
            reason = "all_convex_pieces_exceed_initial_support_band"
        self._support_state_receipt = {"mode": "measured_refresh", "reason": reason, "released_before": before,
            "released_after": self.support_released, "actor_vertical_displacement_m": displacement_z,
            "control_step": self._control_step(),
            "separation_numeric_band_m": NUMERICAL_TOP_TOLERANCE_M, "all_pieces_separated": all_clear,
            "geometry_all_pieces_clear": all_clear, "physical_support_band_clear": physical_band_clear,
            "physical_support_band_m": self.support_band_m, "support_version": TABLE_SUPPORT_VERSION,
            "fcl_touching_pieces": touching, "pieces": pieces}
        identity = (reason, self.support_released, touching)
        previous = self._support_state_history[-1] if self._support_state_history else None
        if previous is None or identity != (previous["reason"], previous["released_after"], previous["fcl_touching_pieces"]):
            self._support_state_history.append(copy.deepcopy(self._support_state_receipt))

    def _support_allowed(self, target_name: str, other_name: str, object_pose: np.ndarray,
                         contacts: list[dict[str, Any]]) -> bool:
        receipt: dict[str, Any] = {"mode": "supplied_object_pose", "target": target_name, "table": other_name,
            "vertex_source": "supplied_object_pose_times_captured_shape_local", "support_released": self.support_released,
            "control_step": self._control_step(),
            "mode_interpretation": "Caller supplies candidate or measured pose; this query does not advance measured state"}
        try:
            if self._legacy_support_mode is not None:
                pose = _rigid_transform(object_pose)
                permitted = AttachedObjectCollisionChecker._support_allowed(self, target_name, other_name, pose, contacts)
                receipt.update(object_pose=pose.tolist(), semantics=self._legacy_support_mode,
                               status="allowed" if permitted else "rejected", reason="legacy_exact_pair_support_rule")
                return self._store_support_validation(receipt, contacts)
            if (not self._initial_support_exists or self.support_released or other_name != TABLE_NAME
                    or target_name not in self._target_names or not contacts):
                raise RuntimeError("No active initial whole-object tabletop support")
            pose = _rigid_transform(object_pose)
            receipt["object_pose"] = pose.tolist()
            displacement = pose[:3, 3] - self._initial_object[:3, 3]
            rotation = pose[:3, :3] @ self._initial_object[:3, :3].T
            angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
            limit = self._initial_object_depth_m + self.support_depth_slop_m
            receipt.update(displacement_m=displacement.tolist(), rotation_rad=angle,
                           initial_object_max_depth_m=self._initial_object_depth_m, depth_limit_m=limit)
            if (not -1e-6 <= displacement[2] <= self.support_band_m
                    or np.linalg.norm(displacement[:2]) > .001 or angle > .01):
                raise RuntimeError("Whole-object support motion bound exceeded")
            receipt["geometry"] = self._validate_contact_geometry(target_name, contacts, pose, limit)
            receipt.update(status="allowed", reason="bounded_initial_object_support_transition")
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            receipt.update(status="rejected", reason=str(exc))
        return self._store_support_validation(receipt, contacts)

    def _store_support_validation(self, receipt: dict[str, Any], contacts: list[dict[str, Any]]) -> bool:
        """Keep bounded receipts through the parent checker's refresh snapshots."""
        self._last_support_validation = receipt
        self._support_validation_history.append(copy.deepcopy(receipt))
        if hasattr(self, "coverage"):
            self.coverage["last_support_validation"] = copy.deepcopy(receipt)
            self.coverage["support_validation_history"] = copy.deepcopy(self._support_validation_history)
        # Preserve all original raw fields. DualPayloadCollisionChecker includes
        # this same list in its allowed/blocked receipt without another adapter.
        for contact in contacts:
            if isinstance(contact, dict):
                contact["support_validation"] = copy.deepcopy(receipt)
        return receipt["status"] == "allowed"

    def support_initial_geometry(self) -> dict[str, Any]:
        """Export cooked geometry once per coverage snapshot, never per contact."""
        return copy.deepcopy(self._initial_geometry_export)

    def refresh(self) -> dict[str, Any]:
        coverage = super().refresh()
        coverage.update(numerical_top_support_receipts=copy.deepcopy(self.numerical_top_support_receipts),
            support_semantics=self._legacy_support_mode or "whole_initial_object_with_bounded_convex_piece_transfer",
            support_adapter_version=TABLE_SUPPORT_VERSION,
            initial_object_support_exists=self._initial_support_exists,
            initial_object_support_max_depth_m=self._initial_object_depth_m,
            initial_object_support_receipts=copy.deepcopy(self._initial_support_receipts),
            initial_near_top_support_receipts=copy.deepcopy(self._initial_near_top_support_receipts),
            initial_support_geometry=self.support_initial_geometry(),
            support_state=copy.deepcopy(self._support_state_receipt),
            support_state_transition_history=copy.deepcopy(self._support_state_history),
            support_validation_history=copy.deepcopy(self._support_validation_history),
            last_support_validation=copy.deepcopy(self._last_support_validation),
            support_geometry_fingerprints=self._initial_geometry_fingerprints.copy())
        return coverage
