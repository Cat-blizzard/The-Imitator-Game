"""Bounded whole-object tabletop support, without changing contact physics.

A held convex object may transfer initial tabletop support between its convex
pieces while it separates upward. Every exception is independently bounded by
scaled convex vertices, the horizontal table Box, the original motion/depth
limits, and a permanent measured-only release state. Near-coplanar contacts
within the existing 1 micrometre bound do not rely on an unstable FCL normal.
Candidate vertices come from the supplied object pose, never the live actor.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import sapien
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl
from transforms3d.quaternions import mat2quat

from .held_collision import AttachedObjectCollisionChecker, _object_transform, _pair_result

NUMERICAL_TOP_TOLERANCE_M = 1e-6
ZERO_NORMAL_TOLERANCE = 1e-8  # Diagnostic compatibility; no longer a support prerequisite.
TABLE_NAME = "scene::table::shape_0"


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

    def _register_initial_support(self, objects: dict[str, Any]) -> None:
        self.numerical_top_support_receipts: list[dict[str, Any]] = []
        self._local_vertices: dict[str, np.ndarray] = {}
        self._initial_geometry_export: dict[str, Any] = {}
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
        self._initial_geometry_export = {
            name: {"scaled_shape_local_vertices": vertices.tolist(), "shape_local_T": self._shape_local[name].tolist()}
            for name, vertices in sorted(self._local_vertices.items())}
        self._table_pose, self._table_side = self._table(objects)
        initial_pieces = self._pieces(self._initial_object)
        pieces_by_name = {piece["target"]: piece for piece in initial_pieces}
        self._initial_object_depth_m = max(p["vertex_penetration_m"] for p in initial_pieces)
        for name in sorted(self._target_names):
            collision, contacts = _pair_result(objects[name], objects[TABLE_NAME])
            if not collision:
                piece = pieces_by_name[name]
                if (abs(piece["bottom_minus_table_top_m"]) <= NUMERICAL_TOP_TOLERANCE_M
                        and piece["entire_footprint_inside"] and piece["vertices_above_top"]):
                    # Initial geometry in the existing numerical band can be
                    # valid support without a reported FCL intersection. Keep
                    # this evidence distinct from real collision pairs/contacts.
                    self._initial_near_top_support_receipts.append({**piece,
                        "classification": "geometry_only_initial_near_top_support",
                        "fcl_collision": False, "tolerance_m": NUMERICAL_TOP_TOLERANCE_M})
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

        Release requires all vertex gaps > 1 micrometre AND no FCL piece/table
        intersections, or the original 3mm actor-height band to be exceeded.
        A sub-micrometre separation remains numerically ambiguous support; it
        cannot undo a previous release or bypass any contact/motion gate.
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
        reason = "already_released" if before else "object_support_active"
        if not self._initial_support_exists:
            reason = "no_initial_support"
        elif not before and (displacement_z > self.support_band_m or all_clear):
            self.support_released = True
            reason = "vertical_support_band_exceeded" if displacement_z > self.support_band_m else "all_convex_pieces_separated"
        self._support_state_receipt = {"mode": "measured_refresh", "reason": reason, "released_before": before,
            "released_after": self.support_released, "actor_vertical_displacement_m": displacement_z,
            "control_step": self._control_step(),
            "separation_numeric_band_m": NUMERICAL_TOP_TOLERANCE_M, "all_pieces_separated": all_clear,
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
            initial_object_support_exists=self._initial_support_exists,
            initial_object_support_max_depth_m=self._initial_object_depth_m,
            initial_object_support_receipts=copy.deepcopy(self._initial_support_receipts),
            initial_near_top_support_receipts=copy.deepcopy(self._initial_near_top_support_receipts),
            initial_support_geometry=self.support_initial_geometry(),
            support_state=copy.deepcopy(self._support_state_receipt),
            support_state_transition_history=copy.deepcopy(self._support_state_history),
            support_validation_history=copy.deepcopy(self._support_validation_history),
            last_support_validation=copy.deepcopy(self._last_support_validation),
            support_geometry_fingerprints={name: hashlib.sha256(v.tobytes()).hexdigest()
                                           for name, v in sorted(self._local_vertices.items())})
        return coverage


def _self_test() -> dict[str, Any]:
    """Real convex/Box geometry tests; no scene, environment or integration steps."""
    from transforms3d.euler import euler2mat
    tests: list[str] = []
    side = np.array([2., 1.2, .1]); table_pose = np.eye(4); table_pose[2, 3] = -.05
    table = fcl.CollisionObject(fcl.Box(side), table_pose[:3, 3])
    material = physx.PhysxMaterial(.5, .5, 0.)
    cube = np.array([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (0., .02)], dtype=np.float32)

    def fixture(*, two_pieces: bool = False, height: float = -2e-5,
                x: float = 0.) -> tuple[Any, dict[str, Any]]:
        checker = NumericalTableSupportCollisionChecker()
        checker._initial_object = np.eye(4); checker._initial_object[:3, 3] = [x, 0, height]
        checker._target_names = frozenset(("piece0", "piece1") if two_pieces else ("piece0",))
        checker._shape_local = {}; refs = {}; objects = {TABLE_NAME: table}
        entity = sapien.Entity(); entity.pose = sapien.Pose([x, 0, height])
        for index, name in enumerate(sorted(checker._target_names)):
            shape = physx.PhysxCollisionShapeConvexMesh(cube, [1., 1., 1.], material)
            local = np.eye(4); local[0, 3] = .04 * (2 * index - 1) if two_pieces else 0.
            if two_pieces: local[2, 3] = 4e-5 * (1 - index)
            shape.set_local_pose(sapien.Pose(local[:3, 3]))
            refs[name] = (entity, shape); checker._shape_local[name] = local
            geometry = fcl.Convex(np.asarray(shape.vertices, dtype=float), np.asarray(shape.get_triangles(), dtype=np.int32), True)
            pose = checker._initial_object @ local
            objects[name] = fcl.CollisionObject(geometry, pose[:3, 3], mat2quat(pose[:3, :3]))
        checker.live = SimpleNamespace(_shape_refs=refs)
        checker._support_pairs = {}; checker.support_released = False
        checker.support_band_m = checker.support_penetration_m = .003
        checker.support_depth_slop_m = .0002
        checker._register_initial_support(objects); checker.coverage = {}
        return checker, objects

    def check_contact(checker: Any, objects: dict[str, Any], pose: np.ndarray, name: str = "piece0") -> tuple[bool, list[dict[str, Any]]]:
        geometry = objects[name].get_collision_geometry(); transform = pose @ checker._shape_local[name]
        obj = fcl.CollisionObject(geometry, transform[:3, 3], mat2quat(transform[:3, :3]))
        collided, contacts = _pair_result(obj, table)
        assert collided and contacts, "Test must contain actual FCL contact geometry"
        return checker._support_allowed(name, TABLE_NAME, pose, contacts), contacts

    def reject(call: Any, label: str) -> None:
        try: call()
        except RuntimeError: tests.append(label)
        else: raise AssertionError("Invalid support accepted: " + label)

    vertices = cube.astype(float) + [0, 0, 2e-7]
    raw = {"depth": 1e-7, "normal": [.15, -.948, -.281], "position": [0, 0, 1e-7]}
    for normal in ([0., 0, 0], raw["normal"]):
        contact = {**raw, "normal": normal}
        assert _numerical_top_receipt("near-plane", contact, vertices, table_pose, side)["raw_contact"] == contact
    tests.append("near_coplanar_geometry_accepts_zero_and_tilted_raw_normal_without_rewriting")
    reject(lambda: _numerical_top_receipt("deep", raw, vertices - [0, 0, .0001], table_pose, side), "numerical_real_penetration_rejected")
    reject(lambda: _numerical_top_receipt("edge", raw, vertices + [1., 0, 0], table_pose, side), "numerical_edge_projection_rejected")
    reject(lambda: _numerical_top_receipt("gap", raw, vertices + [0, 0, 2e-6], table_pose, side), "numerical_above_plane_rejected")
    reject(lambda: _numerical_top_receipt("depth", {**raw, "depth": 2e-6}, vertices, table_pose, side), "numerical_contact_depth_rejected")
    reject(lambda: _numerical_top_receipt("position", {**raw, "position": [0, 0, 2e-6]}, vertices, table_pose, side), "numerical_contact_off_plane_rejected")
    reject(lambda: _numerical_top_receipt("bad", {**raw, "normal": [float("nan"), 0, 0]}, vertices, table_pose, side), "nonfinite_contact_rejected")
    scaled = physx.PhysxCollisionShapeConvexMesh(cube, [2., 3., 4.], material)
    scaled.set_local_pose(sapien.Pose([.1, -.1, .04]))
    entity = sapien.Entity(); entity.pose = sapien.Pose([-.1, -.1, 2e-7], [2**-.5, 0, 0, 2**-.5])
    local = np.asarray(scaled.vertices, dtype=float) * scaled.scale + [.1, -.1, .04]
    expected = np.column_stack((-local[:, 1], local[:, 0], local[:, 2])) + [-.1, -.1, 2e-7]
    assert np.allclose(_target_world_vertices(SimpleNamespace(_shape_refs={"scaled": (entity, scaled)}), "scaled"), expected, atol=2e-8, rtol=0)
    tests.append("actual_scaled_convex_vertices_actor_rotation_and_nonzero_local_pose")
    reject(lambda: fixture(height=-.004), "initial_real_deep_penetration_rejected")
    reject(lambda: fixture(height=-.115), "initial_real_bottom_contact_rejected")
    reject(lambda: fixture(x=.999), "initial_real_table_edge_crossing_rejected")
    checker, objects = fixture()
    allowed, contacts = check_contact(checker, objects, checker._initial_object)
    assert allowed and contacts[0]["support_validation"]["status"] == "allowed"
    tests.append("independently_bounded_actual_initial_support")
    for delta, label in (([0, 0, -.0001], "downward_motion"), ([.0011, 0, 0], "lateral_motion")):
        pose = checker._initial_object.copy(); pose[:3, 3] += delta
        assert not check_contact(checker, objects, pose)[0]
        tests.append(label + "_rejected")
    pose = checker._initial_object.copy(); pose[:3, :3] = euler2mat(0, .011, 0)
    assert not check_contact(checker, objects, pose)[0]
    tests.append("actual_rotation_exceeds_bound_rejected")
    # Candidate vertices must follow the supplied pose, even if the live actor
    # still has valid initial support. This pose is a genuine deep collision.
    deep = checker._initial_object.copy(); deep[2, 3] -= .004
    assert not check_contact(checker, objects, deep)[0] and not checker.support_released
    assert checker._world_vertices("piece0", deep)[:, 2].min() < -.004
    tests.append("candidate_deep_geometry_not_live_actor_geometry_and_no_release_mutation")
    for offset, label in (([.999, 0, -.002], "side_edge"), ([0, 0, -.115], "table_bottom")):
        pose = checker._initial_object.copy(); pose[:3, 3] += offset
        assert not check_contact(checker, objects, pose)[0]
        tests.append("real_" + label + "_contact_rejected")
    tilted_table = table_pose.copy(); tilted_table[:3, :3] = euler2mat(.1, 0, 0)
    reject(lambda: _vertex_receipt("tilted", vertices, tilted_table, side), "nonhorizontal_table_rejected")
    empty, empty_objects = fixture(height=.002)
    assert not empty._initial_support_exists
    pose = empty._initial_object.copy(); pose[2, 3] = -1e-5
    assert not check_contact(empty, empty_objects, pose)[0]
    tests.append("no_initial_support_cannot_be_created_by_candidate")
    # Near-top registration is based on initial geometry alone, never invented
    # collision pairs or contacts, and remains within the existing 1um band.
    near, near_objects = fixture(height=5e-7)
    assert not _pair_result(near_objects["piece0"], table)[0]
    assert near._initial_support_exists and not near._support_pairs and not near._initial_support_receipts
    assert len(near._initial_near_top_support_receipts) == 1
    receipt = near._initial_near_top_support_receipts[0]
    assert receipt["classification"] == "geometry_only_initial_near_top_support" and receipt["fcl_collision"] is False
    assert receipt["bottom_minus_table_top_m"] == 5e-7 and receipt["tolerance_m"] == 1e-6
    near_pose = near._initial_object.copy(); near_pose[2, 3] = -2.5e-7
    assert check_contact(near, near_objects, near_pose)[0] and not near.support_released
    assert not near._support_pairs and not near._initial_support_receipts
    tests.append("geometry_only_near_top_registration_allows_bounded_candidate_without_fabricated_contacts")
    at_limit, _ = fixture(height=1e-6)
    beyond, beyond_objects = fixture(height=np.nextafter(1e-6, np.inf))
    assert at_limit._initial_support_exists and not beyond._initial_support_exists
    assert len(at_limit._initial_near_top_support_receipts) == 1 and not beyond._initial_near_top_support_receipts
    tests.append("geometry_only_registration_uses_original_inclusive_one_micrometer_limit")
    outside, _ = fixture(height=5e-7, x=.999)
    assert not outside._initial_support_exists and not outside._initial_near_top_support_receipts
    reject(lambda: fixture(two_pieces=True, height=5e-7, x=-.951),
           "geometry_only_support_requires_entire_object_footprint_inside")
    tests.append("near_top_piece_outside_table_footprint_does_not_register")
    beyond_pose = beyond._initial_object.copy(); beyond_pose[:3, :3] = euler2mat(0, .0002, 0)
    assert not check_contact(beyond, beyond_objects, beyond_pose)[0]
    assert not beyond._initial_support_exists and not beyond._initial_near_top_support_receipts and not beyond.support_released
    tests.append("candidate_contact_within_rotation_limit_cannot_create_initial_near_top_support")
    up = near._initial_object.copy(); up[2, 3] += .001
    clear = {TABLE_NAME: table, "piece0": fcl.CollisionObject(near_objects["piece0"].get_collision_geometry(), up[:3, 3])}
    near._update_support_release(clear, up)
    assert near.support_released and near._support_state_receipt["reason"] == "all_convex_pieces_separated"
    assert not check_contact(near, near_objects, near_pose)[0]
    near._update_support_release(near_objects, near._initial_object)
    assert near.support_released
    tests.append("geometry_only_initial_support_is_permanently_released_from_measured_separation")
    # Transfer support from the right convex piece to the left by a 0.002 rad
    # whole-object tilt: true shallow contact, inside every original motion gate.
    whole, whole_objects = fixture(two_pieces=True)
    assert ("piece1", TABLE_NAME) in whole._support_pairs and ("piece0", TABLE_NAME) not in whole._support_pairs
    pose = whole._initial_object.copy(); pose[:3, :3] = euler2mat(0, -.002, 0); pose[2, 3] += 1e-5
    allowed, contacts = check_contact(whole, whole_objects, pose, "piece0")
    assert allowed and not whole.support_released
    deep_rotation = whole._initial_object.copy(); deep_rotation[:3, :3] = euler2mat(0, -.009, 0)
    assert not check_contact(whole, whole_objects, deep_rotation, "piece0")[0]
    rejected = whole._last_support_validation
    assert rejected["rotation_rad"] < .01 and rejected["displacement_m"] == [0., 0., 0.]
    assert rejected["reason"] == "Convex geometry is not bounded top support"
    tests.append("real_depth_gate_rejects_with_rotation_and_translation_inside_motion_limits")
    edge, edge_objects = fixture(x=.9895)
    edge_pose = edge._initial_object.copy(); edge_pose[0, 3] += .0008
    assert not check_contact(edge, edge_objects, edge_pose)[0]
    rejected = edge._last_support_validation
    assert abs(rejected["displacement_m"][0]) < .001 and rejected["rotation_rad"] == 0.
    assert rejected["reason"] == "Convex geometry is not bounded top support"
    tests.append("real_footprint_gate_rejects_edge_crossing_inside_lateral_motion_limit")
    measured_objects = {TABLE_NAME: table}
    for name in sorted(whole._target_names):
        transform = pose @ whole._shape_local[name]
        measured_objects[name] = fcl.CollisionObject(whole_objects[name].get_collision_geometry(), transform[:3, 3], mat2quat(transform[:3, :3]))
    whole._update_support_release(measured_objects, pose)
    assert not whole.support_released and whole._support_state_receipt["fcl_touching_pieces"] == ["piece0"]
    tests.append("real_convex_piece_transfer_keeps_whole_object_support")
    up = pose.copy(); up[2, 3] += .001
    clear = {TABLE_NAME: table}
    for name in sorted(whole._target_names):
        transform = up @ whole._shape_local[name]
        clear[name] = fcl.CollisionObject(whole_objects[name].get_collision_geometry(), transform[:3, 3], mat2quat(transform[:3, :3]))
    whole._update_support_release(clear, up)
    assert whole.support_released and whole._support_state_receipt["reason"] == "all_convex_pieces_separated"
    assert not check_contact(whole, whole_objects, pose)[0]
    whole._update_support_release(measured_objects, pose)
    assert whole.support_released
    tests.append("fully_separated_state_permanent_against_candidate_and_measured_recontact")
    assert NumericalTableSupportCollisionChecker.check is AttachedObjectCollisionChecker.check
    assert NumericalTableSupportCollisionChecker.measured_check is AttachedObjectCollisionChecker.measured_check
    assert len(whole.support_initial_geometry()) == 2
    # Ordinary primitive payloads retain the old exact-pair exception. No
    # convex vertices are invented or required for a sphere/no-table world.
    for height, touching in ((.05, False), (.00998, True)):
        primitive = NumericalTableSupportCollisionChecker()
        primitive._initial_object = np.eye(4); primitive._initial_object[2, 3] = height
        primitive._target_names = frozenset(("sphere",)); primitive._shape_local = {"sphere": np.eye(4)}
        primitive._support_pairs = {}; primitive.support_released = False
        primitive.support_band_m = primitive.support_penetration_m = .003
        primitive.support_depth_slop_m = .0002
        primitive.live = SimpleNamespace(_shape_refs={"sphere": (sapien.Entity(), physx.PhysxCollisionShapeSphere(.01, material))})
        sphere = fcl.CollisionObject(fcl.Sphere(.01), np.array([0., 0, height]))
        primitive_objects = {TABLE_NAME: table, "sphere": sphere}
        primitive._register_initial_support(primitive_objects)
        assert primitive._legacy_support_mode == "legacy_nonconvex_support"
        assert primitive._initial_support_exists == touching
        collided, contacts = _pair_result(sphere, table)
        assert collided == touching
        if touching:
            assert primitive._support_allowed("sphere", TABLE_NAME, primitive._initial_object, contacts)
            up = primitive._initial_object.copy(); up[2, 3] += .01
            primitive._update_support_release({TABLE_NAME: table, "sphere": fcl.CollisionObject(fcl.Sphere(.01), up[:3, 3])}, up)
            assert primitive.support_released
            assert not primitive._support_allowed("sphere", TABLE_NAME, primitive._initial_object, contacts)
        tests.append("legacy_sphere_" + ("top_support_and_permanent_release" if touching else "without_table_contact"))
    missing = NumericalTableSupportCollisionChecker()
    missing._target_names = frozenset(("unread_geometry",)); missing._shape_local = {}
    missing._support_pairs = {}; missing.support_released = False
    missing._register_initial_support({})  # No live object or geometry getter.
    assert missing._legacy_support_mode == "legacy_missing_table" and not missing._initial_support_exists
    missing._update_support_release({}, np.eye(4))
    assert not missing.support_released and missing.support_initial_geometry() == {}
    tests.append("missing_table_does_not_read_or_require_convex_geometry")
    return {"status": "passed", "tests": tests, "physics_steps": 0,
            "geometry_shapes": {"convex_piece_vertices": list(cube.shape), "whole_object_pieces": 2},
            "whole_object_support_receipt": whole._support_state_receipt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test; this module never advances physics")
    print(json.dumps(_self_test(), indent=2, allow_nan=False))
