"""Pill-only bounded initial tabletop regions, without changing contact physics.

An initial region is registered from every actual cooked convex piece and the
finite horizontal table Box.  Its registration need not assert an FCL contact:
native contact offsets and quaternion roundoff can leave a micrometre gap at
capture.  Raw FCL contacts, when present, are retained unchanged; their normals
are diagnostics, while vertices and contact positions prove top support.

The actor-origin motion window is diagnostic. Every actual convex piece,
finite-table containment, contact geometry, penetration and depth growth own
support acceptance. The original 1 micrometre numerical clearance is reported
separately from leaving the original 3 mm initial-support region. Only measured
refresh can close the region permanently, after full clearance and departure.
This module neither steps an environment nor writes physical poses or masks.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .dual_payload_collision import DualPayloadCollisionChecker
from .held_collision import _pair_result
from .table_support_collision import (
    NUMERICAL_TOP_TOLERANCE_M, TABLE_NAME,
    NumericalTableSupportCollisionChecker, _rigid_transform, _scaled_vertices,
)

SUPPORT_VERSION = "pill_bounded_initial_tabletop_region_v2"


class PillTableSupportCollisionChecker(NumericalTableSupportCollisionChecker):
    """Task-specific support proof; all other held collision checks are inherited."""

    def _register_initial_support(self, objects: dict[str, Any]) -> None:
        # Keep the parent receipt/export contract for the existing full checker.
        # A pill adapter requires the real convex pieces and finite table; it
        # deliberately has no primitive or missing-table fallback.
        self.numerical_top_support_receipts = []
        self._local_vertices = {name: _scaled_vertices(self.live, name).copy()
                                for name in self._target_names}
        self._initial_geometry_export = {
            name: {"scaled_shape_local_vertices": vertices.tolist(),
                   "shape_local_T": self._shape_local[name].tolist()}
            for name, vertices in sorted(self._local_vertices.items())}
        self._legacy_support_mode = None
        self._initial_support_exists = False
        self._initial_object_depth_m = 0.
        self._initial_support_receipts = []
        self._initial_near_top_support_receipts = []
        self._initial_region_receipts: list[dict[str, Any]] = []
        self._last_support_validation = None
        self._support_validation_history = []
        self._support_state_history = []
        self._table_pose, self._table_side = self._table(objects)
        pieces = self._pieces(self._initial_object)
        if not pieces:
            raise RuntimeError("Pill support requires at least one actual convex piece")
        self._initial_object_depth_m = max(p["vertex_penetration_m"] for p in pieces)
        closest_gap = min(p["bottom_minus_table_top_m"] for p in pieces)
        region_near = closest_gap <= self.support_band_m
        if region_near and any(not p["entire_footprint_inside"] or not p["vertices_above_top"]
                               or p["vertex_penetration_m"] > self.support_penetration_m for p in pieces):
            raise RuntimeError("Initial pill is not bounded finite-table top support")
        for piece in pieces:
            name = piece["target"]
            collision, contacts = _pair_result(objects[name], objects[TABLE_NAME])
            if collision:
                # This independent geometric proof retains the original 3 mm
                # penetration bound.  No decision depends on a convenient FCL
                # normal, and no contact field is rewritten or synthesized.
                receipt = self._validate_contact_geometry(
                    name, contacts, self._initial_object, self.support_penetration_m)
                receipt.update(classification="independently_bounded_initial_top_contact",
                               support_version=SUPPORT_VERSION, fcl_collision=True,
                               actual_fcl_contacts_present=True,
                               native_physx_contact_proven_here=False,
                               normal_used_for_acceptance=False)
                self._support_pairs[(name, TABLE_NAME)] = {
                    "initial_depth": receipt["max_contact_depth_m"],
                    "contacts": copy.deepcopy(contacts)}
                self._initial_object_depth_m = max(self._initial_object_depth_m,
                                                   receipt["max_contact_depth_m"])
                self._initial_support_receipts.append(receipt)
            if region_near and piece["bottom_minus_table_top_m"] <= self.support_band_m:
                region = {**piece, "classification": "initial_bounded_top_support_region",
                          "support_version": SUPPORT_VERSION, "fcl_collision": bool(collision),
                          "actual_fcl_contacts_present": bool(collision and contacts),
                          "native_physx_contact_proven_here": False,
                          "registration_band_m": self.support_band_m,
                          "registration_does_not_assert_contact": not bool(collision)}
                self._initial_region_receipts.append(region)
                if not collision:
                    self._initial_near_top_support_receipts.append(copy.deepcopy(region))
        self._initial_support_exists = bool(region_near)
        self._support_state_receipt = {
            "mode": "initial_registration", "support_version": SUPPORT_VERSION,
            "reason": "initial_bounded_region_registered" if region_near else "no_initial_support_region",
            "initial_pieces": pieces, "initial_closest_gap_m": closest_gap,
            "registered_region_is_contact_assertion": False,
            "initial_FCL_collision_piece_count": len(self._support_pairs)}

    def _update_support_release(self, objects: dict[str, Any], actual_object: np.ndarray) -> None:
        """Close only after measured origin leaves band AND all pieces are clear.

        Geometric clearance alone is a separate fact.  Native physical contact
        lifecycle remains owned by the shared recorder, including cached native
        manifolds; this checker does not claim to measure those contacts.
        """
        actual_object = _rigid_transform(actual_object)
        before = bool(self.support_released)
        self._table_pose, self._table_side = self._table(objects)
        pieces = self._pieces(actual_object)
        touching = [name for name in sorted(self._target_names)
                    if _pair_result(objects[name], objects[TABLE_NAME])[0]]
        displacement_z = float(actual_object[2, 3] - self._initial_object[2, 3])
        vertex_clear = bool(all(p["bottom_minus_table_top_m"] > NUMERICAL_TOP_TOLERANCE_M for p in pieces))
        all_clear = bool(vertex_clear and not touching)
        band_left = displacement_z > self.support_band_m
        if self._initial_support_exists and not before and band_left and all_clear:
            self.support_released = True
        reason = ("already_released" if before else
                  "measured_band_departure_and_all_convex_pieces_clear" if self.support_released else
                  "initial_support_region_active" if self._initial_support_exists else
                  "no_initial_support_region")
        self._support_state_receipt = {
            "mode": "measured_refresh", "support_version": SUPPORT_VERSION, "reason": reason,
            "released_before": before, "released_after": self.support_released,
            "control_step": self._control_step(), "actor_vertical_displacement_m": displacement_z,
            "support_band_m": self.support_band_m,
            "actor_exceeds_initial_support_band": bool(band_left),
            "separation_numeric_band_m": NUMERICAL_TOP_TOLERANCE_M,
            "geometry_all_vertex_gaps_clear": vertex_clear,
            "geometry_all_pieces_clear": all_clear, "all_pieces_separated": all_clear,
            "physical_support_region_released": bool(self.support_released),
            "native_physical_contact_lifecycle_measured_here": False,
            "release_rule": "measured_actor_dz_gt_original_band_AND_all_vertex_gaps_gt_original_numeric_band_AND_no_FCL_intersections",
            "fcl_touching_pieces": touching, "pieces": pieces}
        previous = self._support_state_history[-1] if self._support_state_history else None
        keys = ("reason", "released_after", "geometry_all_pieces_clear", "actor_exceeds_initial_support_band", "fcl_touching_pieces")
        if previous is None or any(previous.get(key) != self._support_state_receipt[key] for key in keys):
            self._support_state_history.append(copy.deepcopy(self._support_state_receipt))

    def _support_allowed(self, target_name: str, other_name: str, object_pose: np.ndarray,
                         contacts: list[dict[str, Any]]) -> bool:
        receipt: dict[str, Any] = {
            "mode": "supplied_object_pose", "support_version": SUPPORT_VERSION,
            "target": target_name, "table": other_name,
            "vertex_source": "supplied_object_pose_times_captured_shape_local",
            "support_released": self.support_released, "control_step": self._control_step(),
            "normal_used_for_acceptance": False,
            "mode_interpretation": "Same candidate/measured geometric rule; query never advances measured region state"}
        try:
            if (not self._initial_support_exists or self.support_released or other_name != TABLE_NAME
                    or target_name not in self._target_names or not contacts):
                raise RuntimeError("No active initial bounded pill/tabletop region")
            pose = _rigid_transform(object_pose)
            displacement = pose[:3, 3] - self._initial_object[:3, 3]
            rotation = pose[:3, :3] @ self._initial_object[:3, :3].T
            angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
            # Existing depth growth, additionally capped by the original hard
            # penetration bound; region registration expands no collision bound.
            limit = min(self.support_penetration_m,
                        self._initial_object_depth_m + self.support_depth_slop_m)
            receipt.update(object_pose=pose.tolist(), displacement_m=displacement.tolist(), rotation_rad=angle,
                           initial_object_max_depth_m=self._initial_object_depth_m, depth_limit_m=limit)
            # Origin motion is not penetration: a rotating supported object can
            # settle several micrometres while its actual contact stays shallow.
            # Retain the former window for comparison; geometry below decides.
            receipt.update(original_motion_window_within_bounds=bool(
                -1e-6 <= displacement[2] <= self.support_band_m
                and np.linalg.norm(displacement[:2]) <= .001 and angle <= .01),
                original_motion_window_used_for_acceptance=False)
            pieces = self._pieces(pose)
            receipt["whole_object_pieces"] = pieces
            if any(not p["entire_footprint_inside"] or not p["vertices_above_top"]
                   or p["vertex_penetration_m"] > limit for p in pieces):
                raise RuntimeError("Supplied whole-object geometry exceeds bounded top support")
            receipt["geometry"] = self._validate_contact_geometry(target_name, contacts, pose, limit)
            receipt.update(status="allowed", reason="independently_bounded_initial_pill_top_support")
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            receipt.update(status="rejected", reason=str(exc))
        return self._store_support_validation(receipt, contacts)

    def refresh(self) -> dict[str, Any]:
        coverage = super().refresh()
        coverage.update(support_semantics=SUPPORT_VERSION, support_adapter_version=SUPPORT_VERSION,
                        initial_support_region_receipts=copy.deepcopy(self._initial_region_receipts),
                        registered_region_is_contact_assertion=False,
                        geometry_all_pieces_clear=self._support_state_receipt.get("geometry_all_pieces_clear"),
                        physical_support_region_released=bool(self.support_released),
                        native_physical_contact_lifecycle_measured_here=False)
        return coverage


class PillDualPayloadCollisionChecker(DualPayloadCollisionChecker):
    """Original dual checker, replacing only its pill tabletop support factory."""

    def attach(self) -> None:
        if self.attached:
            raise ValueError("Payload relations are already captured")
        supports = tuple(PillTableSupportCollisionChecker.from_live(live, target, active_robot=robot)
                         for live, target, robot in zip(self.lives, self.targets, ("left", "right")))
        self._supports, self.attached = supports, True
        try:
            self.refresh()
        except Exception:
            self.attached, self._fresh = False, False
            raise


def self_test() -> dict[str, Any]:
    """Exercise geometric support bounds on synthetic convex vertices, on CPU."""
    checker = PillTableSupportCollisionChecker()
    checker._initial_object = np.eye(4)
    checker._target_names = frozenset(("piece",))
    vertices = np.array([[x, y, z] for x in (-.01, .01) for y in (-.01, .01)
                         for z in (0., .02)])
    checker._local_vertices = {"piece": vertices}
    checker._shape_local = {"piece": np.eye(4)}
    checker._table_pose = np.eye(4)
    checker._table_pose[2, 3] = -.05
    checker._table_side = np.array([2., 1.2, .1])
    checker._initial_support_exists = True
    checker._initial_object_depth_m = 0.
    checker.support_released = False
    checker.support_band_m = checker.support_penetration_m = .003
    checker.support_depth_slop_m = .0002
    checker._support_validation_history = []
    checker.coverage = {}
    pose = np.eye(4)
    pose[2, 3] = -8e-6

    def allowed(value: np.ndarray) -> bool:
        return checker._support_allowed("piece", TABLE_NAME, value,
            [{"depth": max(0., -float(value[2, 3])),
              "normal": [0., 0., 1.], "position": [0., 0., 0.]}])

    assert allowed(pose)
    assert not checker._last_support_validation["original_motion_window_within_bounds"]
    deep = pose.copy(); deep[2, 3] = -.0003
    assert not allowed(deep)
    edge = pose.copy(); edge[0, 3] = 1.
    assert not allowed(edge)
    checker.support_released = True
    assert not allowed(pose)
    return {"status": "passed", "vertices_shape": list(vertices.shape),
            "support_cases": 4, "environment_steps": 0}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), allow_nan=False))
