"""Geometry adapter for the original TIG apple/banana basket task.

Only robot goals are computed here. Original objects, reset sampling, basket
predicates and the 200-step task horizon remain owned by the environment.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sapien

from .pill_task import PillTask, _single, quaternion_matrix_wxyz, transform_local_vertices
from .trajectory_execution import as_numpy

TASK_ID = "TwoRobotPickAppleBananaToBaskets-v1"
CONTENTS_PHASES = ("release", "retreat", "verify_placement", "observe_task_outcome")


def basket_contents_geometry_proof(target_vertices: Any, contents_vertices: Any,
                                   basket_vertices: Any, contact_positions: Any, *,
                                   target_center: Any = None, contents_center: Any = None,
                                   height_threshold: float = .10) -> dict:
    """Bound normal landing contact to the corresponding original basket.

    The environment's geometric centers and height threshold define placement.
    Full collision bounds remain diagnostics, without an extra containment or
    orientation gate. Normals, impulses and separations are never changed.
    """
    meshes = [np.asarray(value, dtype=float) for value in
              (target_vertices, contents_vertices, basket_vertices)]
    if any(v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 4
           or not np.isfinite(v).all() for v in meshes):
        raise ValueError("Complete finite collision vertices[N,3] required")
    target, contents, basket = [np.stack((v.min(axis=0), v.max(axis=0))) for v in meshes]
    if any(np.any(b[1] <= b[0]) for b in (target, contents, basket)):
        raise ValueError("Nondegenerate collision geometry required")
    positions = np.asarray(contact_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("Original finite contact positions[N,3] required")
    centers = [np.asarray(value, dtype=float) if value is not None else bounds.mean(axis=0)
               for value, bounds in ((target_center, target), (contents_center, contents))]
    if (any(value.shape != (3,) or not np.isfinite(value).all() for value in centers)
            or not np.isfinite(height_threshold) or height_threshold <= 0):
        raise ValueError("Original finite geometric centers and positive height threshold required")
    def inside(points):
        return (np.all((points[..., :2] >= basket[0, :2]) &
                       (points[..., :2] <= basket[1, :2]), axis=-1)
                & (points[..., 2] <= basket[1, 2]+height_threshold))
    regions = []
    for center in centers:
        regions.append(bool(inside(center)))
    points_inside = bool(np.all(inside(positions)))
    return {"allowed_geometry": bool(all(regions) and points_inside),
        "target_and_contents_inside": regions, "contact_points_inside": points_inside,
        "basket_bounds_m": basket.tolist(), "object_collision_bounds_m": [target.tolist(), contents.tolist()],
        "height_threshold_m": height_threshold, "full_bounds_are_acceptance_gate": False,
        "raw_contact_fields_changed": False}


def basket_release_center(bounds: Any, payload_half_extent: Any,
                          contents_bounds: Any, source_xy: Any) -> np.ndarray:
    """Choose a geometric landing proposal, without assigning physical poses.

    All inputs are world metres: basket and contents bounds[2,3], payload
    horizontal half extents[2], current grasp center[2]. This finite search
    prefers central support with clearance from the existing contents.
    Bounding boxes only propose goals; native collision and physics checks
    retain responsibility for accepting or rejecting the actual execution.
    """
    bounds = np.asarray(bounds, dtype=float)
    contents = np.asarray(contents_bounds, dtype=float)
    half = np.asarray(payload_half_extent, dtype=float)
    source = np.asarray(source_xy, dtype=float)
    if (bounds.shape != (2, 3) or contents.shape != (2, 3)
            or half.shape != (2,) or source.shape != (2,)
            or not all(np.isfinite(x).all() for x in (bounds, contents, half, source))
            or np.any(bounds[1] <= bounds[0]) or np.any(half <= 0)):
        raise ValueError("Finite basket/contents bounds and positive payload extents required")
    free = (bounds[1, :2]-bounds[0, :2])/2-half
    margin = np.minimum(.012, np.maximum(0., free/2))
    low = bounds[0, :2] + half + margin
    high = bounds[1, :2] - half - margin
    if np.any(low > high):
        raise ValueError("Payload has no contained landing proposal inside measured basket bounds")
    candidates = [np.array([x, y]) for x in np.linspace(low[0], high[0], 5)
                  for y in np.linspace(low[1], high[1], 5)]

    def score(center: np.ndarray) -> tuple[float, float, float]:
        separation = np.maximum(contents[0, :2]-(center+half),
                                (center-half)-contents[1, :2])
        # First avoid placing the payload on the original fruit when the
        # measured basket has a free proposal. The old weighted sum always
        # preferred the center: moving away gains at most the same distance
        # in separation but paid twice that distance in centrality. Clip free
        # separation at zero so excess clearance cannot pull a goal into a
        # distant corner. These bounds propose goals; they never accept or
        # reject physical contacts.
        centrality = float(np.linalg.norm(center-bounds.mean(axis=0)[:2]))
        return (min(float(np.max(separation)), 0.), -centrality,
                -float(np.linalg.norm(center-source)))

    proposal = max(candidates, key=score).copy()
    assert proposal.shape == (2,) and np.isfinite(proposal).all()
    return proposal


class AppleBananaTask(PillTask):
    """Original object identities and measured goals for a Piper pair."""

    target_names = ("apple_out", "banana_out")
    official_predicate_keys = ("apple_in_basket", "banana_in_basket")
    destination_contents_state_names = ("apple_in", "banana_in")

    def __init__(self, raw: Any):
        self.raw = raw
        self.targets = raw.apple_out, raw.banana_out
        self.destination_actors = raw.basket_apple, raw.basket_banana
        self.destination_contents = raw.apple_in, raw.banana_in
        self.table = raw.table_scene.table
        self.initial_supports = (self.table, self.table)
        self.destination_supports = self.destination_actors
        self.local_vertices = {}
        self.initial_quaternions = tuple(_single(a.pose.q, 4, "initial WXYZ") for a in self.targets)
        self.release_goal_measurements = []

    def destination_contents_contact_proof(self, index: int, other_state_name: str,
                                          phase: str, points: list[dict]) -> dict:
        """Allow only measured same-basket landing against its original fruit."""
        if type(index) is not int or index not in (0, 1):
            raise ValueError("Canonical left/right target index required")
        receipt = {"allowed": False, "phase": phase, "canonical_target_index": index,
            "expected_contents": self.destination_contents_state_names[index],
            "raw_contact_fields_changed": False}
        if phase not in CONTENTS_PHASES or other_state_name != self.destination_contents_state_names[index]:
            return {**receipt, "reason": "phase_or_corresponding_contents_identity_rejected"}
        try:
            geometry = []
            for actor in (self.targets[index], self.destination_contents[index], self.destination_actors[index]):
                mesh = actor.get_first_collision_mesh(to_world_frame=False)
                if mesh is None:
                    raise ValueError("Current native collision mesh required")
                geometry.append(transform_local_vertices(mesh.vertices,
                    _single(actor.pose.p, 3, "actor XYZ"), _single(actor.pose.q, 4, "actor WXYZ")))
            proof = basket_contents_geometry_proof(*geometry, [p["position_m"] for p in points],
                target_center=_single(self.raw._get_actor_geom_center(self.targets[index]), 3, "target geometry center"),
                contents_center=_single(self.raw._get_actor_geom_center(self.destination_contents[index]), 3, "contents geometry center"),
                height_threshold=self.raw.basket_height_threshold)
            return {**receipt, "allowed": proof["allowed_geometry"], "geometry": proof,
                "reason": "corresponding_contents_and_target_inside_basket" if proof["allowed_geometry"]
                          else "current_geometry_or_contact_points_outside_basket"}
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return {**receipt, "reason": "invalid_current_native_geometry", "error": str(exc)}

    def grasp_poses(self) -> tuple:
        profile = self.raw.embodiment_profile
        pairs = []
        for i, actor in enumerate(self.targets):
            # Use the mesh bounds center, independent of mesh tessellation.
            vertices = self.vertices(actor)
            center = (vertices.min(axis=0)+vertices.max(axis=0))/2
            grasp = sapien.Pose(center+[0., 0., profile.grasp_height_offsets_m[i]],
                                profile.grasp_quaternions[i])
            from ...envs.original_task_bridge import migrate_grasp
            grasp = migrate_grasp(self.raw, grasp)
            pregrasp = sapien.Pose(grasp.p+[0., 0., profile.pregrasp_clearances_m[i]], grasp.q)
            pairs.append((pregrasp, grasp))
        return tuple(pairs)

    def release_poses(self) -> tuple:
        poses = []
        self.release_goal_measurements = []
        for agent, actor, basket, contents in zip(self.raw.agent.agents, self.targets,
                                                self.destination_actors, self.destination_contents):
            if not bool(as_numpy(agent.is_grasping(actor)).item()):
                raise ValueError("Release target requires an actual confirmed grasp")
            tcp = as_numpy(agent.tcp.pose.to_transformation_matrix())[0]
            obj = as_numpy(actor.pose.to_transformation_matrix())[0]
            self.vertices(actor)
            relation = np.linalg.inv(tcp) @ obj
            vertices = self.local_vertices[id(actor)] @ relation[:3, :3].T + relation[:3, 3]
            basket_vertices, contents_vertices = self.vertices(basket), self.vertices(contents)
            bounds = np.stack((basket_vertices.min(axis=0), basket_vertices.max(axis=0)))
            contents_bounds = np.stack((contents_vertices.min(axis=0), contents_vertices.max(axis=0)))
            quaternion = _single(agent.tcp.pose.q, 4, "carry WXYZ")
            current_rotation = quaternion_matrix_wxyz(quaternion)
            source_center = (self.vertices(actor).min(axis=0)+self.vertices(actor).max(axis=0))[:2]/2
            relative = vertices @ current_rotation.T
            relative_low, relative_high = relative.min(axis=0), relative.max(axis=0)
            center_xy = basket_release_center(bounds, (relative_high-relative_low)[:2]/2,
                                               contents_bounds, source_center)
            position = np.empty(3)
            position[:2] = center_xy-(relative_low+relative_high)[:2]/2
            position[2] = max(bounds[1, 2], contents_bounds[1, 2])+.015-relative_low[2]
            poses.append(sapien.Pose(position, quaternion))
            self.release_goal_measurements.append({"basket_bounds_m": bounds.tolist(),
                "contents_bounds_m": contents_bounds.tolist(), "landing_center_xy_m": center_xy.tolist(),
                "tcp_goal_m": position.tolist(), "yaw_change_rad": 0., "minimum_air_gap_m": .015})
        return tuple(poses)

    def region_predicates(self) -> tuple:
        def make(index: int):
            def region(actor: Any) -> dict:
                if actor is not self.targets[index]:
                    raise ValueError("Wrong canonical target")
                evaluation = self.raw.evaluate()
                return {"inside": bool(as_numpy(evaluation[self.official_predicate_keys[index]]).item()),
                    "source": "original environment geometric-center basket predicate",
                    "object_collision_bounds_m": np.stack((self.vertices(actor).min(axis=0),
                                                            self.vertices(actor).max(axis=0))).tolist()}
            return region
        return make(0), make(1)


if __name__ == "__main__":
    point = basket_release_center([[-.15, -.15, 0], [.15, .15, .08]], [.025, .025],
        [[-.03, -.03, 0], [.03, .03, .07]], [-.2, -.2])
    assert point.shape == (2,) and np.all(np.abs(point) <= .113+1e-12)
    assert np.max(np.abs(point)) >= .055
    print("Basket landing proposal has two finite metre coordinates:", point)
