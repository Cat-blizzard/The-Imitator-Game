"""Pill-only bounded initial tabletop regions, without changing contact physics.

An initial region is registered from every actual cooked convex piece and the
finite horizontal table Box.  Its registration need not assert an FCL contact:
native contact offsets and quaternion roundoff can leave a micrometre gap at
capture.  Raw FCL contacts, when present, are retained unchanged; their normals
are diagnostics, while vertices and contact positions prove top support.

All original motion, penetration and depth-growth bounds still apply.  The
original 1 micrometre numerical clearance is reported separately from leaving
the original 3 mm initial-support region.  Only measured refresh can close the
region, permanently, after both full geometric clearance and band departure.
This module neither steps an environment nor writes physical poses or masks.
"""
from __future__ import annotations

import argparse
import copy
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import sapien
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl
from transforms3d.quaternions import mat2quat

from .dual_payload_collision import DualPayloadCollisionChecker
from .held_collision import _object_at, _pair_result
from .table_support_collision import (
    NUMERICAL_TOP_TOLERANCE_M, TABLE_NAME,
    NumericalTableSupportCollisionChecker, _rigid_transform, _scaled_vertices,
)

SUPPORT_VERSION = "pill_bounded_initial_tabletop_region_v1"


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
            if (not -1e-6 <= displacement[2] <= self.support_band_m
                    or np.linalg.norm(displacement[:2]) > .001 or angle > .01):
                raise RuntimeError("Original whole-object support motion bound exceeded")
            pieces = self._pieces(pose)
            if any(not p["entire_footprint_inside"] or not p["vertices_above_top"]
                   or p["vertex_penetration_m"] > limit for p in pieces):
                raise RuntimeError("Supplied whole-object geometry exceeds bounded top support")
            receipt["whole_object_pieces"] = pieces
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


def _self_test() -> dict[str, Any]:
    """Standalone cooked convex/FCL boundaries, no scene/env/physics steps."""
    from transforms3d.euler import euler2mat
    tests: list[str] = []
    side = np.array([2., 1.2, .1])
    table_pose = np.eye(4); table_pose[2, 3] = -.05
    table = _object_at(fcl.Box(side), table_pose)
    material = physx.PhysxMaterial(.5, .5, 0.)
    cube = np.array([[x, y, z] for x in (-.01, .01) for y in (-.01, .01)
                     for z in (0., .02)], dtype=np.float32)

    def fixture(height: float = -1.14e-6, *, x: float = 0., two: bool = False,
                table_object: Any = table) -> tuple[Any, dict[str, Any]]:
        checker = PillTableSupportCollisionChecker()
        checker._initial_object = np.eye(4); checker._initial_object[:3, 3] = [x, 0, height]
        checker._target_names = frozenset(("piece0", "piece1") if two else ("piece0",))
        checker._shape_local = {}; refs = {}; objects = {TABLE_NAME: table_object}
        entity = sapien.Entity(); entity.pose = sapien.Pose([x, 0, height])
        for index, name in enumerate(sorted(checker._target_names)):
            shape = physx.PhysxCollisionShapeConvexMesh(cube, [1., 1., 1.], material)
            local = np.eye(4)
            if two:
                local[0, 3] = .04 * (2 * index - 1)
                local[2, 3] = .004 * index
            shape.set_local_pose(sapien.Pose(local[:3, 3])); refs[name] = (entity, shape)
            checker._shape_local[name] = local
            geometry = fcl.Convex(np.asarray(shape.vertices, dtype=float),
                                  np.asarray(shape.get_triangles(), dtype=np.int32), True)
            objects[name] = _object_at(geometry, checker._initial_object @ local)
        checker.live = SimpleNamespace(_shape_refs=refs)
        checker._support_pairs = {}; checker.support_released = False
        checker.support_band_m = checker.support_penetration_m = .003
        checker.support_depth_slop_m = .0002
        checker._register_initial_support(objects); checker.coverage = {}
        return checker, objects

    def objects_at(checker: Any, objects: dict[str, Any], pose: np.ndarray) -> dict[str, Any]:
        return {**objects, **{name: _object_at(objects[name].get_collision_geometry(),
                                             pose @ checker._shape_local[name]) for name in checker._target_names}}

    def query(checker: Any, objects: dict[str, Any], pose: np.ndarray,
              name: str = "piece0") -> tuple[bool, list[dict[str, Any]]]:
        collided, contacts = _pair_result(objects_at(checker, objects, pose)[name], objects[TABLE_NAME])
        assert collided and contacts, "Boundary test requires real standalone FCL intersection"
        return checker._support_allowed(name, TABLE_NAME, pose, contacts), contacts

    def reject(call: Any, label: str) -> None:
        try:
            call()
        except (RuntimeError, ValueError):
            tests.append(label)
        else:
            raise AssertionError("Invalid support accepted: " + label)

    checker, objects = fixture()
    assert checker._initial_support_exists and checker._support_pairs
    allowed, contacts = query(checker, objects, checker._initial_object)
    assert allowed and contacts[0]["depth"] > NUMERICAL_TOP_TOLERANCE_M
    raw = {key: copy.deepcopy(contacts[0][key]) for key in ("depth", "position", "normal")}
    for normal in ([0., 0., 0.], [-.131, .105, -.986], [1., 0., 0.]):
        contact = {**raw, "normal": normal}
        assert checker._support_allowed("piece0", TABLE_NAME, checker._initial_object, [contact])
        assert contact["normal"] == normal and contact["depth"] == raw["depth"] and contact["position"] == raw["position"]
    tests.append("bounded_top_contact_above_1um_arbitrary_raw_normal_preserved")
    near, near_objects = fixture(2.01e-6)
    assert near._initial_support_exists and not near._support_pairs and not near._initial_support_receipts
    assert near._initial_near_top_support_receipts[0]["registration_does_not_assert_contact"]
    assert not near._initial_near_top_support_receipts[0]["actual_fcl_contacts_present"]
    tests.append("2um_initial_gap_registers_region_without_fabricating_FCL_contact")
    touch = near._initial_object.copy(); touch[:3, :3] = euler2mat(0., .0004, 0.)
    touch[2, 3] += .5e-6
    assert query(near, near_objects, touch)[0]
    assert not near.support_released
    tests.append("region_without_initial_FCL_intersection_allows_bounded_rotated_liftoff_contact")
    near._update_support_release(near_objects, near._initial_object)
    assert near._support_state_receipt["geometry_all_pieces_clear"] and not near.support_released
    assert not near._support_state_receipt["actor_exceeds_initial_support_band"]
    tests.append("numeric_all_clear_alone_does_not_close_original_3mm_region")
    at_band = near._initial_object.copy(); at_band[2, 3] += near.support_band_m
    near._update_support_release(objects_at(near, near_objects, at_band), at_band)
    assert not near.support_released
    clear = at_band.copy(); clear[2, 3] += 1e-6
    near._update_support_release(objects_at(near, near_objects, clear), clear)
    assert near.support_released and near._support_state_receipt["geometry_all_pieces_clear"]
    near._update_support_release(objects_at(near, near_objects, touch), touch)
    assert near.support_released and not query(near, near_objects, touch)[0]
    tests.append("strict_measured_band_and_allclear_release_is_permanent_recontact_rejected")
    # One piece is geometrically clear; another stays in the contact band.
    whole, whole_objects = fixture(two=True)
    blocked = whole._initial_object.copy(); blocked[2, 3] += .0031
    blocked[:3, :3] = euler2mat(0., -.1, 0.)
    whole._update_support_release(objects_at(whole, whole_objects, blocked), blocked)
    assert whole._support_state_receipt["actor_exceeds_initial_support_band"]
    assert not whole._support_state_receipt["geometry_all_pieces_clear"] and not whole.support_released
    tests.append("origin_band_departure_insufficient_when_one_actual_convex_piece_still_touches")
    for delta, label in (([0, 0, -2e-6], "downward_origin"), ([.0011, 0, 0], "XY_drag")):
        pose = checker._initial_object.copy(); pose[:3, 3] += delta
        assert not query(checker, objects, pose)[0]
        tests.append(label + "_original_bound_rejected")
    pose = checker._initial_object.copy(); pose[:3, :3] = euler2mat(0., .011, 0.)
    assert not query(checker, objects, pose)[0]
    tests.append("rotation_original_bound_rejected")
    pose = checker._initial_object.copy(); pose[:3, :3] = euler2mat(0., .009, 0.)
    # An independent contact-depth violation cannot be masked by a normal.
    false_depth = {**raw, "depth": checker._initial_object_depth_m + .000201}
    assert not checker._support_allowed("piece0", TABLE_NAME, checker._initial_object, [false_depth])
    tests.append("initial_depth_plus_original_slop_not_enlarged")
    for kwargs, label in (({"height": -.00301}, "deep_initial"), ({"height": -.115}, "underside_initial"),
                          ({"x": .999}, "edge_crossing_initial")):
        reject(lambda kw=kwargs: fixture(**kw), label + "_rejected")
    tilted = table_pose.copy(); tilted[:3, :3] = euler2mat(.1, 0, 0)
    reject(lambda: fixture(table_object=_object_at(fcl.Box(side), tilted)), "nonhorizontal_table_rejected")
    empty, empty_objects = fixture(.00301)
    assert not empty._initial_support_exists
    down = empty._initial_object.copy(); down[2, 3] = -1e-6
    assert not query(empty, empty_objects, down)[0]
    tests.append("no_initial_region_never_creates_later_support")
    for other, name in (("scene::other::shape_0", "piece0"), (TABLE_NAME, "unknown")):
        assert not checker._support_allowed(name, other, checker._initial_object, [copy.deepcopy(raw)])
    tests.append("unrelated_actor_and_unknown_piece_rejected")
    bad = {**raw, "normal": [float("nan"), 0, 0]}
    assert not checker._support_allowed("piece0", TABLE_NAME, checker._initial_object, [bad])
    for position, label in (([1.01, 0, 0], "contact_XY_outside"), ([0, 0, -.01], "contact_off_top")):
        assert not checker._support_allowed("piece0", TABLE_NAME, checker._initial_object, [{**raw, "position": position}])
        tests.append(label + "_rejected")
    tests.append("nonfinite_contact_rejected")
    initial = checker._initial_object.copy(); local = {n: t.copy() for n, t in checker._shape_local.items()}
    state = copy.deepcopy(checker._support_state_receipt); before = checker.support_released
    deep = initial.copy(); deep[2, 3] -= .004
    assert not query(checker, objects, deep)[0]
    assert np.array_equal(checker._initial_object, initial) and checker.support_released == before
    assert checker._support_state_receipt == state
    assert all(np.array_equal(t, local[n]) for n, t in checker._shape_local.items())
    assert checker._world_vertices("piece0", deep)[:, 2].min() < -.004
    tests.append("supplied_candidate_geometry_used_no_query_release_or_attachment_mutation")
    assert NUMERICAL_TOP_TOLERANCE_M == 1e-6
    assert checker.support_band_m == checker.support_penetration_m == .003 and checker.support_depth_slop_m == .0002
    assert PillDualPayloadCollisionChecker.check is DualPayloadCollisionChecker.check
    assert PillDualPayloadCollisionChecker._evaluate is DualPayloadCollisionChecker._evaluate
    assert PillDualPayloadCollisionChecker.measured_check is DualPayloadCollisionChecker.measured_check
    tests.append("original_constants_full_collision_engine_and_executor_unchanged")
    return {"status": "passed", "support_version": SUPPORT_VERSION, "tests": tests,
            "test_count": len(tests), "physics_steps": 0, "environments_created": 0, "gpu_used": False,
            "geometry_backend": "standalone_actual_PhysX_cooked_convex_and_MPLib_FCL",
            "native_validation": "NOT_RUN", "numerical_top_tolerance_m": NUMERICAL_TOP_TOLERANCE_M}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(_self_test(), indent=2, allow_nan=False))
