"""Task geometry and predicates for the three fixed dual-Panda M1 tasks.

Contract: canonical pair index 0 is left, 1 is right; world XYZ is in metres,
quaternions are WXYZ, and angles are radians. Native pose arguments below are
single-environment SAPIEN poses. TaskAdapter never steps, resets, assigns poses,
changes collision masks, or changes a common executor. Target actors have live
pose arrays (1,3)/(1,4); local collision geometry has shape (N,3). Pair geometry
has shape (2,3), pair quaternions (2,4), and AABB endpoints (2,3).

The official success predicates are deliberately separate from conservative
physical containment. Pill's original source does not require an upright bottle
or static robots in its success conjunction; orientation is reported, not added
to official success. Fruits starts on two source plates, not on the table.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import dataclass, field
import itertools
import json
from typing import Any, Callable

import numpy as np

TASK_IDS = (
    "TwoRobotPickAppleBananaToBaskets-v1",
    "TwoRobotPickPillToRegions-v1",
    "TwoRobotPickFruitsToPlate-v1",
)
ALIASES = dict(zip(("apple-banana", "pill", "fruits"), TASK_IDS))
ADAPTER_VERSION = "m1-task-adapters-v12"
DESTINATION_CONTENTS_POLICY_VERSION = "paired-applebanana-basket-contents-v1"
DESTINATION_CONTENTS_PHASES = ("release", "retreat", "verify_placement")
PILL_RELEASE_CLEARANCE_M = .015
PILL_RELEASE_POLICY = "measured-attachment-full-collision-geometry-clearance-v2"
FRUIT_RELEASE_CLEARANCE_M = .015
FRUIT_RELEASE_POLICY = "fruit-measured-attachment-full-collision-geometry-clearance-v12"
PILL_GRASP_HEIGHTS_M = (.12, .13)
PILL_PREGRASP_HEIGHTS_M = (.21, .21)
PILL_SIDE_CONTACT_METADATA_HEIGHT_M = .9059799909591675 * .08
PILL_GRASP_POLICY = "fixed-model-specific-grasp-heights-v7"


def task_phase_limits(configuration: dict[str, Any], task_id: str, phase: str) -> dict[str, Any]:
    """Resolve task-only phase limits through the existing common limit format."""
    from .validate_sync_compact import compact_phase_limits
    canonical = ALIASES.get(task_id, task_id)
    if canonical not in TASK_IDS: raise ValueError(f'Unsupported task {task_id}')
    limits = compact_phase_limits(configuration, phase)
    return limits


def task_execution_parameters(task_id: str) -> dict[str, Any]:
    """Pure pre-environment task phase limits; development choices, not results.

    These are input to the same common planner/retimer/derivative acceptance
    gates. Changing a task's approach limits does not relax grasp, collision,
    release, retreat, stability, or official 200-tick acceptance. The paired
    serial/synchronous protocols must receive the same task-specific values.
    """
    if not isinstance(task_id, str):
        raise ValueError("Task ID must be a supported string")
    canonical = ALIASES.get(task_id, task_id)
    if canonical not in TASK_IDS:
        raise ValueError(f"Unsupported task {task_id}")
    approach_velocity, approach_acceleration = {TASK_IDS[0]:(.9,.9), TASK_IDS[1]:(1.,1.3), TASK_IDS[2]:(1.5,2.6)}[canonical]
    return {"task_id": canonical, "parameter_version": "development-v12",
            "joint_vel_limit": 1.5, "joint_acc_limit": 2.6,
            "approach_vel_limit": approach_velocity, "approach_acc_limit": approach_acceleration,
            "preparation_steps": 10, "max_episode_steps": 200,
            "approach_phases": ["pregrasp", "descend"],
            "task_phase_overrides": {},
            "units": {"joint_velocity": "radians/second", "joint_acceleration": "radians/second^2"},
            "physical_acceptance_thresholds_changed": False,
            "evidence_status": "development configuration; native acceptance must be established separately"}


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = _array(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}, got {array.shape}")
    return array


def _single(value: Any, width: int, label: str) -> np.ndarray:
    array = _array(value)
    if array.shape == (1, width):
        array = array[0]
    return _finite(array, (width,), label).copy()


def quaternion_matrix_wxyz(quaternion: Any) -> np.ndarray:
    """Return SO(3)[3,3] for one finite nonzero WXYZ[4] quaternion."""
    quaternion = _finite(quaternion, (4,), "WXYZ quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("Quaternion must be nonzero")
    w, x, y, z = quaternion / norm
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y-z*w), 2 * (x*z+y*w)],
                     [2 * (x*y+z*w), 1 - 2 * (x*x+z*z), 2 * (y*z-x*w)],
                     [2 * (x*z-y*w), 2 * (y*z+x*w), 1 - 2 * (x*x+y*y)]])


def quaternion_multiply_wxyz(left: Any, right: Any) -> np.ndarray:
    a, b, c, d = _finite(left, (4,), "left quaternion")
    e, f, g, h = _finite(right, (4,), "right quaternion")
    value = np.array([a*e-b*f-c*g-d*h, a*f+b*e+c*h-d*g,
                      a*g-b*h+c*e+d*f, a*h+b*g-c*f+d*e])
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError("Quaternion product is zero")
    return value / norm


def transform_local_vertices(local_vertices: Any, position: Any, quaternion: Any) -> np.ndarray:
    """Transform finite local collision vertices (N,3) into world metres."""
    vertices = _array(local_vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Need nonempty finite local collision vertices[N,3]")
    position = _finite(position, (3,), "world position")
    result = vertices @ quaternion_matrix_wxyz(quaternion).T + position
    assert result.shape == vertices.shape and np.isfinite(result).all()
    return result


def _bounds(vertices: Any) -> np.ndarray:
    vertices = _array(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Need nonempty finite world vertices[N,3]")
    return np.stack([vertices.min(axis=0), vertices.max(axis=0)])


def _ordered_bounds(value: Any, label: str) -> np.ndarray:
    bounds = _finite(value, (2, 3), label)
    if np.any(bounds[0] > bounds[1]):
        raise ValueError(f"{label} must be ordered")
    return bounds


def rotation_error_rad(reference_wxyz: Any, observed_wxyz: Any) -> float:
    relative = quaternion_matrix_wxyz(reference_wxyz).T @ quaternion_matrix_wxyz(observed_wxyz)
    return float(np.arccos(np.clip((np.trace(relative)-1) / 2, -1, 1)))


def pill_region_measurement(origin: Any, world_vertices: Any, region_center: Any, *,
                            half_size_xy: Any = (.08, .08), height_threshold: float = .20,
                            table_top_z: float = 0., initial_quaternion: Any | None = None,
                            current_quaternion: Any | None = None) -> dict[str, Any]:
    """Official origin test plus conservative full collision-geometry XY test.

    The marker is noncolliding; its z is not the support height. The real table
    provides support. No orientation gate is invented: native success ignores it.
    """
    origin = _finite(origin, (3,), "object origin")
    center = _finite(region_center, (3,), "region center")
    half = _finite(half_size_xy, (2,), "region half sizes")
    if np.any(half <= 0) or not np.isfinite([height_threshold, table_top_z]).all():
        raise ValueError("Need positive region size and finite heights")
    bounds = _bounds(world_vertices)
    lower, upper = center[:2]-half, center[:2]+half
    native = bool(np.all(np.abs(origin[:2]-center[:2]) <= half) and origin[2] <= height_threshold)
    contained = bool(np.all(bounds[0, :2] >= lower) and np.all(bounds[1, :2] <= upper))
    no_below_support = bool(bounds[0, 2] >= table_top_z-.005)
    inside = bool(native and contained and no_below_support and bounds[1, 2] <= height_threshold)
    result = {"inside": inside, "official_origin_inside": native, "object_xy_contained": contained,
              "object_origin_m": origin.tolist(), "object_geom_center_m": bounds.mean(axis=0).tolist(),
              "object_aabb_m": bounds.tolist(), "region_xy_bounds_m": [lower.tolist(), upper.tolist()],
              "region_center_m": center.tolist(), "height_threshold_m": float(height_threshold),
              "table_top_z_m": float(table_top_z), "no_below_support": no_below_support,
              "orientation_is_acceptance_gate": False,
              "pose_requirement": "official source imposes no upright condition; stable physical release is checked by executor"}
    if (initial_quaternion is None) != (current_quaternion is None):
        raise ValueError("Supply both initial and current quaternion, or neither")
    if initial_quaternion is not None:
        result["rotation_from_initial_rad"] = rotation_error_rad(initial_quaternion, current_quaternion)
    return result


def plate_region_measurement(origin: Any, world_vertices: Any, plate_center: Any, *,
                             radius: float = .13, height_threshold: float = .10,
                             plate_world_bounds: Any | None = None) -> dict[str, Any]:
    """Official origin disk test and full carried collision-vertex disk test.

    A disk is convex: checking all convex-part vertices also contains each part's
    faces and interior. Plate bounds only constrain the lower physical floor;
    the official task disk, not the mesh's outer rim, defines allowed XY.
    """
    origin = _finite(origin, (3,), "object origin")
    center = _finite(plate_center, (3,), "plate center")
    vertices = _array(world_vertices)
    bounds = _bounds(vertices)
    if not np.isfinite([radius, height_threshold]).all() or radius <= 0:
        raise ValueError("Need finite positive plate radius and finite height threshold")
    origin_radius = float(np.linalg.norm(origin[:2]-center[:2]))
    max_radius = float(np.linalg.norm(vertices[:, :2]-center[:2], axis=1).max())
    native = bool(origin_radius <= radius and origin[2] <= height_threshold)
    below_floor = False
    floor = 0.
    if plate_world_bounds is not None:
        plate_bounds = _ordered_bounds(plate_world_bounds, "plate AABB")
        floor = float(plate_bounds[0, 2])
        below_floor = bool(bounds[0, 2] < floor-.005)
    inside = bool(native and max_radius <= radius and not below_floor)
    return {"inside": inside, "official_origin_inside": native,
            "object_xy_contained": bool(max_radius <= radius),
            "object_origin_m": origin.tolist(), "object_geom_center_m": bounds.mean(axis=0).tolist(),
            "object_aabb_m": bounds.tolist(), "plate_center_m": center.tolist(),
            "plate_radius_m": float(radius), "max_collision_vertex_radius_m": max_radius,
            "origin_radius_m": origin_radius, "height_threshold_m": float(height_threshold),
            "support_floor_z_m": floor, "no_below_support": not below_floor,
            "height_predicate_applies_to": "actor_origin",
            "object_top_z_m": float(bounds[1, 2]),
            "object_top_within_origin_height_threshold": bool(bounds[1, 2] <= height_threshold),
            "object_top_height_is_acceptance_gate": False}


def basket_contents_geometry_proof(target_vertices: Any, contents_vertices: Any,
                                   basket_vertices: Any, contact_positions: Any) -> dict[str, Any]:
    """Pure current-world full-geometry proof for a corresponding basket pair.

    The region uses the existing physical_region bounds and world AABB centre.
    Contact points additionally require strict basket XY and the same existing
    [-5mm floor,+10mm rim] centre-height band. No raw separation, normal, impulse
    or official success predicate is changed. Identity/phase gates are separate.
    """
    from .validate_sync import physical_region
    arrays = [_array(v) for v in (target_vertices, contents_vertices, basket_vertices)]
    for vertices in arrays:
        bounds = _bounds(vertices)
        if len(vertices) < 4 or np.any(bounds[1] <= bounds[0]):
            raise ValueError("Contents proof requires nondegenerate complete native collision vertices")
    target, contents, basket = [_bounds(v) for v in arrays]
    positions = _array(contact_positions)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("Contents proof requires original finite contact points[N,3]")
    target_region = physical_region(target.mean(axis=0), basket, target)
    contents_region = physical_region(contents.mean(axis=0), basket, contents)
    lower, upper = basket[0].copy(), basket[1].copy()
    lower[2] -= .005; upper[2] += .01
    each_point = np.all((positions >= lower) & (positions <= upper), axis=1)
    points_inside = bool(each_point.all())
    return {"allowed_geometry": bool(target_region["inside"] and contents_region["inside"] and points_inside),
            "target_region": target_region, "contents_region": contents_region,
            "basket_world_aabb_m": basket.tolist(), "contact_point_bounds_m": [lower.tolist(), upper.tolist()],
            "contact_points_inside": points_inside, "each_contact_point_inside": each_point.tolist(),
            "center_definition": "midpoint_of_current_world_collision_AABB",
            "contact_position_source": "unmodified_native_contact_points",
            "geometry_vertex_counts": [len(v) for v in arrays]}



def support_clearance_tcp_height(local_vertices: Any, T_tcp_object: Any, carry_quaternion: Any, *,
                                 support_top_z: float, clearance: float) -> dict[str, Any]:
    """Compute TCP target z without assigning any physical state.

    For local collision vertices V[N,3], measured rigid transform T_tcp_object
    [4,4], and commanded world R_carry[3,3], each predicted point relative to
    TCP is R_carry @ (R_tcp_object @ v + t_tcp_object). Thus target TCP z is
    support_top_z + clearance - min(relative_vertex_z). Clearance is a positive
    physical air gap, independent of collision-query tolerances.
    """
    vertices = _array(local_vertices)
    _bounds(vertices)
    transform = _finite(T_tcp_object, (4, 4), "measured T_tcp_object")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-9):
        raise ValueError("Attachment requires homogeneous bottom row")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-6)):
        raise ValueError("Attachment rotation must belong to SO(3)")
    if not np.isfinite([support_top_z, clearance]).all() or clearance <= 0:
        raise ValueError("Need finite support top and strictly positive physical release gap")
    in_tcp = vertices @ rotation.T + transform[:3, 3]
    relative_world = in_tcp @ quaternion_matrix_wxyz(carry_quaternion).T
    minimum = float(relative_world[:, 2].min())
    tcp_z = float(support_top_z + clearance - minimum)
    predicted_minimum = tcp_z + minimum
    assert np.isclose(predicted_minimum, support_top_z+clearance, rtol=0, atol=1e-12)
    return {"target_tcp_z_m": tcp_z, "support_top_z_m": float(support_top_z),
            "physical_clearance_m": float(clearance), "minimum_collision_vertex_relative_to_tcp_z_m": minimum,
            "predicted_minimum_collision_vertex_world_z_m": predicted_minimum,
            "local_collision_vertex_count": len(vertices), "T_tcp_object": transform.tolist(),
            "carry_quaternion_wxyz": _finite(carry_quaternion, (4,), "carry WXYZ").tolist(),
            "formula": "tcp_z=support_top_z+clearance-min_z(R_carry*(R_tcp_object*v+t_tcp_object))"}


def _table_actor(raw: Any) -> Any:
    if hasattr(raw, "table_scene") and hasattr(raw.table_scene, "table"):
        return raw.table_scene.table
    if hasattr(raw, "table"):
        return raw.table
    if hasattr(raw, "scene") and "table" in raw.scene.actors:
        return raw.scene.actors["table"]
    raise ValueError("Cannot identify native table actor")


def _actor_name(actor: Any) -> str:
    return str(getattr(actor, "name", type(actor).__name__))


@dataclass
class TaskAdapter:
    """Live task-only adapter; methods perform geometry queries, never physics."""
    task_id: str
    raw: Any
    _local_geometry: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _initial_quaternions: tuple[np.ndarray, np.ndarray] = field(init=False, repr=False)
    _release_captures: list[dict[str, Any] | None] = field(default_factory=lambda: [None, None], init=False, repr=False)

    def __post_init__(self) -> None:
        self.task_id = ALIASES.get(self.task_id, self.task_id)
        if self.task_id not in TASK_IDS:
            raise ValueError(f"Unsupported task {self.task_id}")
        self._initial_quaternions = tuple(_single(a.pose.q, 4, "initial object WXYZ") for a in self.targets)
        if len(self.targets) != 2 or len(self.raw.agent.agents) != 2:
            raise ValueError("Adapter requires one pair of physical Pandas")

    @property
    def target_names(self) -> tuple[str, str]:
        return {TASK_IDS[0]: ("apple_out", "banana_out"), TASK_IDS[1]: ("pill_a", "pill_b"),
                TASK_IDS[2]: ("fruit_a", "fruit_b")}[self.task_id]

    @property
    def targets(self) -> tuple[Any, Any]:
        return tuple(getattr(self.raw, name) for name in self.target_names)

    @property
    def official_predicate_keys(self) -> tuple[str, str]:
        return {TASK_IDS[0]: ("apple_in_basket", "banana_in_basket"),
                TASK_IDS[1]: ("pill_a_in_region", "pill_b_in_region"),
                TASK_IDS[2]: ("fruit_a_in_plate", "fruit_b_in_plate")}[self.task_id]

    @property
    def initial_supports(self) -> tuple[Any, Any]:
        if self.task_id == TASK_IDS[2]:
            return self.raw.plate_left, self.raw.plate_right
        table = _table_actor(self.raw)
        return table, table

    @property
    def destination_actors(self) -> tuple[Any, Any]:
        if self.task_id == TASK_IDS[0]:
            return self.raw.basket_apple, self.raw.basket_banana
        if self.task_id == TASK_IDS[1]:
            return self.raw.region_a, self.raw.region_b
        return self.raw.plate, self.raw.plate

    @property
    def destination_supports(self) -> tuple[Any, Any]:
        if self.task_id == TASK_IDS[1]:
            table = _table_actor(self.raw)
            return table, table
        return self.destination_actors

    @property
    def initial_support_names(self) -> tuple[str, str]:
        return tuple(_actor_name(actor) for actor in self.initial_supports)

    @property
    def destination_support_names(self) -> tuple[str, str]:
        return tuple(_actor_name(actor) for actor in self.destination_supports)

    @property
    def destination_contents(self) -> tuple[Any, ...]:
        """Only the two explicitly declared pre-existing apple/banana actors."""
        if self.task_id != TASK_IDS[0]:
            return ()
        values = tuple(getattr(self.raw, name) for name in ("apple_in", "banana_in"))
        if tuple(_actor_name(a) for a in values) != ("apple_in", "banana_in") or values[0] is values[1]:
            raise ValueError("Expected exactly the official apple_in/banana_in contents actors")
        if any(a is target for a in values for target in self.targets):
            raise ValueError("Basket contents cannot alias transported targets")
        return values

    def _scene_state_name(self, actor: Any) -> str:
        matches = [name for name, value in self.raw.scene.actors.items() if value is actor]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous scene state identity for {_actor_name(actor)}")
        return matches[0]

    @property
    def destination_contents_state_names(self) -> tuple[str, ...]:
        return tuple(self._scene_state_name(a) for a in self.destination_contents)

    def destination_contents_policy_configuration(self) -> dict[str, Any]:
        contents = self.destination_contents
        pairs = [{"canonical_target_index": i, "target_name": self.target_names[i],
                  "target_actor_name": _actor_name(self.targets[i]),
                  "target_state_name": self._scene_state_name(self.targets[i]),
                  "contents_actor_name": _actor_name(content), "contents_state_name": self._scene_state_name(content),
                  "basket_actor_name": _actor_name(self.destination_actors[i]),
                  "basket_state_name": self._scene_state_name(self.destination_actors[i])}
                 for i, content in enumerate(contents)]
        return {"policy_version": DESTINATION_CONTENTS_POLICY_VERSION, "enabled": bool(contents),
                "allowed_phases": list(DESTINATION_CONTENTS_PHASES) if contents else [],
                "contact_class": "destination_contents", "pairs": pairs,
                "proof": "current full native collision geometry and pose for target, corresponding contents, corresponding basket; both physical_region and all original contact points inside basket region",
                "geometry_center": "midpoint_of_current_world_collision_AABB",
                "geometry_xy_margin_m": .002, "geometry_floor_margin_m": .005, "geometry_center_above_rim_m": .01,
                "contact_point_xy_margin_m": 0., "contact_point_floor_margin_m": .005, "contact_point_above_rim_m": .01,
                "local_geometry_hash": "sha256(np.ascontiguousarray(vertices,dtype=float64).tobytes(order='C'))",
                "other_contact_rules_changed": False, "physical_contact_definition_changed": False}

    def destination_contents_pose_snapshot(self) -> dict[str, Any]:
        contents = self.destination_contents
        def pose(actor: Any) -> list[float]:
            return np.r_[_single(actor.pose.p, 3, "actor XYZ"), _single(actor.pose.q, 4, "actor WXYZ")].tolist()
        return {"destination_contents_poses": [pose(a) for a in contents],
                "destination_basket_poses": [pose(a) for a in self.destination_actors] if contents else []}

    def _current_native_geometry(self, actor: Any) -> tuple[np.ndarray, dict[str, Any]]:
        # Read actual complete native geometry for EACH proof. No stale world
        # bounds, actor-origin proxy or arbitrary actor-name exception is used.
        mesh = actor.get_first_collision_mesh(to_world_frame=False)
        if mesh is None:
            raise ValueError("Contents proof requires current native collision mesh")
        local = np.ascontiguousarray(_array(mesh.vertices), dtype=np.float64)
        _bounds(local)
        pose = np.r_[_single(actor.pose.p, 3, "actor XYZ"), _single(actor.pose.q, 4, "actor WXYZ")]
        if not np.isclose(np.linalg.norm(pose[3:]), 1., rtol=0, atol=1e-5):
            raise ValueError("Contents proof requires native normalized quaternion")
        world = transform_local_vertices(local, pose[:3], pose[3:])
        return world, {"actor_name": _actor_name(actor), "state_name": self._scene_state_name(actor),
                       "pose_world_xyz_wxyz": pose.tolist(), "vertex_count": len(local),
                       "local_vertices_sha256": hashlib.sha256(local.tobytes(order="C")).hexdigest()}

    def destination_contents_contact_proof(self, index: int, other_state_name: str,
                                          phase: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        index = self._arm_index(index)
        policy = self.destination_contents_policy_configuration()
        pair = policy["pairs"][index] if policy["enabled"] else None
        phase_allowed = bool(pair is not None and phase in DESTINATION_CONTENTS_PHASES)
        identity_allowed = bool(pair is not None and other_state_name == pair["contents_state_name"])
        receipt = {"policy_version": DESTINATION_CONTENTS_POLICY_VERSION, "canonical_target_index": index,
                   "phase": phase, "expected_pair": copy.deepcopy(pair),
                   "observed_contents_state_name": other_state_name,
                   "phase_allowed": phase_allowed, "identity_allowed": identity_allowed,
                   "geometry_evaluated": False, "allowed": False,
                   "geometry_source": "current_native_full_collision_mesh_and_current_pose",
                   "raw_contact_fields_changed": False}
        if not phase_allowed or not identity_allowed:
            receipt["reason"] = "phase_or_corresponding_contents_identity_rejected"
            return receipt
        try:
            actors = (self.targets[index], self.destination_contents[index], self.destination_actors[index])
            geometry = [self._current_native_geometry(actor) for actor in actors]
            receipt["native_geometry"] = {role: value[1] for role, value in zip(("target", "contents", "basket"), geometry)}
            receipt["geometry_evaluated"] = True
            receipt["geometry"] = basket_contents_geometry_proof(*(value[0] for value in geometry),
                                                                 [point["position_m"] for point in points])
            receipt["allowed"] = receipt["geometry"]["allowed_geometry"]
            receipt["reason"] = "corresponding_contents_and_target_inside_basket" if receipt["allowed"] else "current_geometry_or_contact_points_outside_basket"
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            receipt.update(reason="invalid_current_native_geometry", geometry_error=f"{type(exc).__name__}: {exc}")
        return receipt

    def _vertices(self, actor: Any) -> np.ndarray:
        key = id(actor)
        if key not in self._local_geometry:
            mesh = actor.get_first_collision_mesh(to_world_frame=False)
            if mesh is None:
                raise ValueError(f"No native collision geometry for {_actor_name(actor)}")
            vertices = _array(mesh.vertices).copy()
            _bounds(vertices)
            self._local_geometry[key] = vertices
        return transform_local_vertices(self._local_geometry[key], _single(actor.pose.p, 3, "actor XYZ"),
                                        _single(actor.pose.q, 4, "actor WXYZ"))

    def grasp_poses(self) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
        """Return ((left_pre,left_grasp),(right_pre,right_grasp)), world poses."""
        import sapien
        if self.task_id == TASK_IDS[0]:
            from .validate_grasp import grasp_poses
            from .validate_banana import banana_grasp_poses
            return grasp_poses(self.raw), banana_grasp_poses(self.raw)
        approaching = np.array([0., 0., -1.])
        agents = tuple(self.raw.agent.agents)
        result = []
        # This shared closing direction matches original pill solution exactly.
        pill_closing = None
        if self.task_id == TASK_IDS[1]:
            transform = _array(agents[1].tcp.pose.to_transformation_matrix())
            if transform.shape != (1, 4, 4):
                raise ValueError("Need one native right TCP transform[1,4,4]")
            pill_closing = transform[0, :3, 1]
            pill_closing -= np.dot(pill_closing, approaching)*approaching
            norm = np.linalg.norm(pill_closing)
            if norm < 1e-8:
                raise ValueError("Official pill closing direction degenerates under horizontal projection")
            pill_closing /= norm
        for index, (agent, actor) in enumerate(zip(agents, self.targets)):
            if self.task_id == TASK_IDS[1]:
                grasp = agent.build_grasp_pose(approaching, pill_closing,
                        _single(actor.pose.p, 3, "pill origin")) * sapien.Pose(p=[0., 0., -PILL_GRASP_HEIGHTS_M[index]])
            else:
                from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
                transform = _array(agent.tcp.pose.to_transformation_matrix())
                if transform.shape != (1, 4, 4):
                    raise ValueError("Need one native TCP transform[1,4,4]")
                info = compute_grasp_info_by_obb(get_actor_obb(actor), approaching=approaching,
                                                target_closing=transform[0, :3, 1], depth=.025)
                grasp = agent.build_grasp_pose(approaching, info["closing"], info["center"])
            pregrasp_distance = PILL_PREGRASP_HEIGHTS_M[index] - PILL_GRASP_HEIGHTS_M[index] if self.task_id == TASK_IDS[1] else .08
            result.append((grasp * sapien.Pose([0., 0., -pregrasp_distance]), grasp))
        return tuple(result)

    @staticmethod
    def _arm_index(index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index not in (0, 1):
            raise ValueError("Canonical arm index must be integer 0=left or 1=right")
        return int(index)

    def carry_quaternion_for_arm(self, index: int) -> np.ndarray:
        """Snapshot only that side after its grasp; WXYZ[4], no peer dependency."""
        index = self._arm_index(index)
        result = _single(self.raw.agent.agents[index].tcp.pose.q, 4, "measured grasp TCP WXYZ")
        if self.task_id == TASK_IDS[0] and index == 1:
            result = quaternion_multiply_wxyz([2**-.5, 0, 0, 2**-.5], result)
        if self.task_id == TASK_IDS[2]:
            result = quaternion_multiply_wxyz([np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)], result)
        assert result.shape == (4,) and np.isfinite(result).all()
        return result

    def carry_quaternions(self) -> np.ndarray:
        """Snapshot after paired grasp confirmation; apple banana keeps world Z90."""
        result = np.stack([self.carry_quaternion_for_arm(index) for index in range(2)])
        assert result.shape == (2, 4) and np.isfinite(result).all()
        return result

    def release_positions(self) -> np.ndarray:
        """World TCP XYZ[2,3]; pill/fruits require two measured side captures."""
        centers = np.stack([_single(actor.pose.p, 3, "destination origin") for actor in self.destination_actors])
        if self.task_id == TASK_IDS[0]:
            from .validate_sync_compact import RELEASE_TCP_OFFSETS_BASKET_M
            offsets = np.asarray(RELEASE_TCP_OFFSETS_BASKET_M)
        else:
            if any(row is None for row in self._release_captures):
                raise ValueError("Paired release positions require each side's confirmed-grasp geometry capture")
            return _finite([row["selected_release_tcp_xyz_m"] for row in self._release_captures],
                           (2, 3), "captured release XYZ").copy()
        result = centers + offsets
        assert result.shape == (2, 3) and np.isfinite(result).all()
        return result

    def _actual_grasp(self, index: int) -> bool:
        flag = _array(self.raw.agent.agents[index].is_grasping(self.targets[index]))
        return bool(flag.size == 1 and np.isfinite(flag).all() and np.isin(flag, [0, 1]).all()
                    and bool(flag.reshape(-1)[0]))

    def release_pose_for_arm(self, index: int, carry_quaternion: Any | None = None) -> Any:
        """World release pose for one arm; pill/fruits capture that side once.

        Serial left may finish while right has not grasped. Existing captures
        are immutable: later calls reuse the same target even after release.
        The optional quaternion must match the captured orientation up to sign.
        """
        import sapien
        index = self._arm_index(index)
        quaternion = self.carry_quaternion_for_arm(index) if carry_quaternion is None else _finite(carry_quaternion, (4,), "carry WXYZ")
        if self.task_id == TASK_IDS[0]:
            return sapien.Pose(p=self.release_positions()[index], q=quaternion)
        if self._release_captures[index] is None:
            if not self._actual_grasp(index):
                raise ValueError("Release geometry requires that side's actual confirmed grasp")
            agent, actor = self.raw.agent.agents[index], self.targets[index]
            world_tcp = _array(agent.tcp.pose.to_transformation_matrix())
            world_object = _array(actor.pose.to_transformation_matrix())
            if world_tcp.shape != (1, 4, 4) or world_object.shape != (1, 4, 4):
                raise ValueError("Need single-environment TCP/object transforms[1,4,4]")
            attachment = np.linalg.inv(world_tcp[0]) @ world_object[0]
            self._vertices(actor)
            support_top = float(_bounds(self._vertices(self.destination_supports[index]))[1, 2])
            clearance = PILL_RELEASE_CLEARANCE_M if self.task_id == TASK_IDS[1] else FRUIT_RELEASE_CLEARANCE_M
            policy = PILL_RELEASE_POLICY if self.task_id == TASK_IDS[1] else FRUIT_RELEASE_POLICY
            value = support_clearance_tcp_height(self._local_geometry[id(actor)], attachment, quaternion,
                                                 support_top_z=support_top, clearance=clearance)
            position = _single(self.destination_actors[index].pose.p, 3, "destination origin")
            if self.task_id == TASK_IDS[2]:
                # Keep the already selected side-preserving slots. Only TCP z
                # changes; original serial solution waypoints are not invoked or
                # rewritten by this task adapter API.
                position[1] += (-.075, .075)[index]
            position[2] = value["target_tcp_z_m"]
            self._release_captures[index] = {"policy": policy,
                "canonical_arm_index": index, "capture_count": 1, "actual_grasp_confirmed": True,
                "support_geometry": value, "selected_release_tcp_xyz_m": position.tolist()}
        captured = self._release_captures[index]
        expected = captured["support_geometry"]["carry_quaternion_wxyz"]
        if rotation_error_rad(expected, quaternion) > 1e-7:
            raise ValueError("Target orientation changed after one-time release geometry capture")
        return sapien.Pose(p=captured["selected_release_tcp_xyz_m"], q=quaternion)

    def release_poses(self, carry_quaternions: Any | None = None) -> tuple[Any, Any]:
        """Paired compatibility API, implemented by the same per-arm captures."""
        quaternions = self.carry_quaternions() if carry_quaternions is None else _finite(carry_quaternions, (2, 4), "carry WXYZ")
        if self.task_id in (TASK_IDS[1], TASK_IDS[2]):
            # Validate both before creating either capture: a failed paired grasp
            # must not silently create a one-sided capture as a side effect.
            if any(row is None and not self._actual_grasp(index)
                   for index, row in enumerate(self._release_captures)):
                raise ValueError("Paired release geometry requires both actual confirmed grasps")
        return tuple(self.release_pose_for_arm(index, quaternions[index]) for index in range(2))

    def _paired_release_capture(self) -> dict[str, Any] | None:
        """Compatibility summary; two independent side captures are not recaptures."""
        if self.task_id == TASK_IDS[0] or any(row is None for row in self._release_captures):
            return None
        return {"policy": self._release_captures[0]["policy"], "capture_count": 1,
                "capture_count_scope": "one complete paired snapshot composed of one capture per arm",
                "per_arm_capture_counts": [1, 1], "total_arm_captures": 2,
                "actual_grasp_confirmed": [True, True],
                "support_geometry": [row["support_geometry"] for row in self._release_captures],
                "selected_release_tcp_xyz_m": self.release_positions().tolist()}

    def execution_parameters(self) -> dict[str, Any]:
        return task_execution_parameters(self.task_id)

    def region_predicates(self) -> tuple[Callable[[Any], dict[str, Any]], Callable[[Any], dict[str, Any]]]:
        def make(index: int) -> Callable[[Any], dict[str, Any]]:
            def measure(actor: Any) -> dict[str, Any]:
                if actor is not self.targets[index]:
                    raise ValueError("Region callback received the wrong canonical target actor")
                if self.task_id == TASK_IDS[0]:
                    from .validate_sync import physical_region
                    from mani_skill.envs.tasks.tabletop.dual_tasks._008_pick_apple_banana_baskets import world_aabb_from_local_mesh
                    return physical_region(_array(self.raw._get_actor_geom_center(actor))[0],
                        _array(world_aabb_from_local_mesh(self.destination_actors[index], self.raw.device))[0],
                        _array(world_aabb_from_local_mesh(actor, self.raw.device))[0])
                origin = _single(actor.pose.p, 3, "object origin")
                if self.task_id == TASK_IDS[1]:
                    table_bounds = _bounds(self._vertices(self.destination_supports[index]))
                    return pill_region_measurement(origin, self._vertices(actor),
                        _single(self.destination_actors[index].pose.p, 3, "region origin"),
                        half_size_xy=self.raw.region_half_size[:2], height_threshold=self.raw.region_height_threshold,
                        table_top_z=float(table_bounds[1, 2]), initial_quaternion=self._initial_quaternions[index],
                        current_quaternion=_single(actor.pose.q, 4, "object WXYZ"))
                return plate_region_measurement(origin, self._vertices(actor),
                    _single(self.raw.plate.pose.p, 3, "plate origin"), radius=self.raw.plate_radius,
                    height_threshold=self.raw.plate_height_threshold, plate_world_bounds=_bounds(self._vertices(self.raw.plate)))
            return measure
        return make(0), make(1)

    def configuration(self) -> dict[str, Any]:
        """JSON snapshot; call again after grasp to record measured release poses."""
        captured_positions = [None if row is None else row["selected_release_tcp_xyz_m"] for row in self._release_captures]
        selected = self.release_positions().tolist() if self.task_id == TASK_IDS[0] else (None if all(row is None for row in self._release_captures) else captured_positions)
        return copy.deepcopy({"adapter_version": ADAPTER_VERSION, "task_id": self.task_id,
            "canonical_arm_order": ["left", "right"], "units": {"position": "metres", "quaternion": "WXYZ", "angle": "radians"},
            "target_names": list(self.target_names), "official_predicate_keys": list(self.official_predicate_keys),
            "initial_support_names": list(self.initial_support_names),
            "destination_names": [_actor_name(a) for a in self.destination_actors],
            "destination_support_names": list(self.destination_support_names),
            "destination_contents_contact_policy": self.destination_contents_policy_configuration(),
            "initial_object_quaternions_wxyz": [q.tolist() for q in self._initial_quaternions],
            "grasp_policy": PILL_GRASP_POLICY if self.task_id == TASK_IDS[1] else "official-OBB-depth25mm",
            "pregrasp_local_tcp_offset_m": [[0., 0., -(pre-grasp)] for pre, grasp in zip(PILL_PREGRASP_HEIGHTS_M, PILL_GRASP_HEIGHTS_M)] if self.task_id == TASK_IDS[1] else [0., 0., -.08],
            "pill_grasp_height_above_actor_origin_m": list(PILL_GRASP_HEIGHTS_M) if self.task_id == TASK_IDS[1] else None,
            "pill_pregrasp_height_above_actor_origin_m": list(PILL_PREGRASP_HEIGHTS_M) if self.task_id == TASK_IDS[1] else None,
            "pill_grasp_object_order": ["pill_a", "pill_b"] if self.task_id == TASK_IDS[1] else None,
            "pill_object_model_configuration": [{"target": "pill_a", "canonical_arm_index": 0, "model_name": "080_pillbottle", "model_id": 1,
                "cooked_collision_extents_local_m": [.07406770810484886, .15478295693174005, .07304728776216507], "grasp_height_m": .12},
                {"target": "pill_b", "canonical_arm_index": 1, "model_name": "080_pillbottle", "model_id": 5,
                "cooked_collision_extents_local_m": [.0609708484262228, .1547949966043234, .06093970872461796], "grasp_height_m": .13}] if self.task_id == TASK_IDS[1] else None,
            "pill_grasp_geometry_evidence": {"asset_metadata_reference": "robotwin/objects/080_pillbottle/model_data1.json",
                "native_side_contact_local_y": .9059799909591675, "native_asset_scale": .08,
                "native_side_contact_height_above_actor_origin_m": PILL_SIDE_CONTACT_METADATA_HEIGHT_M,
                "source_orientation": "object local +Y becomes world +Z under native Rx(pi/2)",
                "observed_development_failures": "v3 pill-dev-004-seed-1 left cap slide; v4 pill-dev-008-seed-1 palm collision at .08m; v5 pill-dev-012-seed-0 right lift slide and pill-dev-013-seed-2 right support collision at .12m; v6 pill-dev-014-seed-1 left lower slide despite slower lowering",
                "selection_scope": "fixed per-object candidate: wider model1 .12m, narrower model5 original .13m; no seed branch and no proven stability claim; original physical gates retained",
                "geometry_evidence_source": "development/pill-dev-014-seed-1/task-geometry.npz exact cooked vertices; original collision/base1.glb and base5.glb match size ordering",
                "physical_contact_gate": "unchanged native checker decides hand/body feasibility"} if self.task_id == TASK_IDS[1] else None,
            "selected_release_tcp_xyz_m": selected,
            "release_height_policy": {TASK_IDS[0]: "retained-fixed-offset", TASK_IDS[1]: PILL_RELEASE_POLICY,
                                      TASK_IDS[2]: FRUIT_RELEASE_POLICY}[self.task_id],
            "release_geometry_capture": self._paired_release_capture(),
            "release_geometry_captures": self._release_captures,
            "release_geometry_capture_counts": [0 if row is None else row["capture_count"] for row in self._release_captures],
            "task_execution_parameters": self.execution_parameters(),
            "planned_pill_release_clearance_m": PILL_RELEASE_CLEARANCE_M if self.task_id == TASK_IDS[1] else None,
            "planned_fruit_release_clearance_m": FRUIT_RELEASE_CLEARANCE_M if self.task_id == TASK_IDS[2] else None,
            "release_height_formula": "tcp_z=support_top_z+clearance-min_z(R_carry*(R_tcp_object*v+t_tcp_object))" if self.task_id != TASK_IDS[0] else None,
            "fruit_release_height_evidence": {"policy": FRUIT_RELEASE_POLICY,
                "capture_scope": "one measured actual-grasp TCP-to-object transform per arm; no peer-grasp dependency and no recapture",
                "support_top": "maximum world Z of the current complete native goal-plate collision vertices, including its rim",
                "clearance_is_physical_air_gap_not_collision_tolerance": True,
                "observed_development_failures": ["development/fruits-dev-004-seed-0:200ticks final stable7/20", "development/fruits-dev-005-seed-0:200ticks final stable18/20"],
                "problem": "fixed +12cm TCP release permits different fruit-to-TCP geometries to create unequal long free-falls and rolling",
                "selection_scope": "shared full-geometry release formula; fixed15mm gap, no seed branch or changed grasp/collision/stability threshold",
                "evidence_status": "development-v12 candidate; CPU geometry tests do not establish native feasibility or budget success",
                "serial_official_fixed_release_waypoints_unchanged": True} if self.task_id == TASK_IDS[2] else None,
            "source_plate_support_required": self.task_id == TASK_IDS[2],
            "fruit_goal_slot_y_offsets_m": [-.075,.075] if self.task_id == TASK_IDS[2] else None,
            "fruit_carry_world_yaw_rad": np.pi/4 if self.task_id == TASK_IDS[2] else None,
            "official_max_episode_steps": 200,
            "official_success_conjunction": " AND ".join(self.official_predicate_keys),
            "official_static_or_upright_required": False,
            "physical_region_policy": {TASK_IDS[0]: "unchanged-applebanana-physical-region",
                TASK_IDS[1]: "full-collision-geometry-XY-and-native-origin-height-no-penetration",
                TASK_IDS[2]: "full-collision-vertices-in-native-goal-disk-native-origin-height-no-penetration-v5"}[self.task_id],
            "route_difference_from_official": "same applebanana selected release recipe" if self.task_id == TASK_IDS[0]
                else "pill_a-model1 grasp .12m and pill_b-model5 grasp .13m versus official both .13m; both pregrasp .21m; original common held limits1.5rad/s,2.6rad/s2 restored after v6 failed; release z derives from measured attachment plus15mm clearance instead of official fixed +.12m" if self.task_id == TASK_IDS[1]
                else "sync keeps Y=-.075m/+.075m slots and dual world-yaw45deg; release TCP z derives from each actual measured grasp and full native fruit geometry to leave15mm above the highest native goal-plate collision point; official serial keeps its fixed +.12m release and crossing diagonal slots unchanged"})


def get_task_adapter(task_id: str, raw: Any) -> TaskAdapter:
    return TaskAdapter(task_id=task_id, raw=raw)


def self_test() -> dict[str, Any]:
    """CPU-only boundary and shape checks; no environment, renderer, or physics."""
    tests = []
    vertices = np.asarray(list(itertools.product([-.01, .01], [-.02, .02], [0., .04])))
    transformed = transform_local_vertices(vertices, [.1, .2, 0], [2**-.5, 0, 0, 2**-.5])
    assert transformed.shape == (8, 3)
    assert np.allclose(_bounds(transformed), [[.08, .19, 0], [.12, .21, .04]])
    assert np.isclose(rotation_error_rad([1, 0, 0, 0], [0, 0, 0, 1]), np.pi)
    assert np.allclose(quaternion_multiply_wxyz([1, 0, 0, 0], [0, 0, 0, 1]), [0, 0, 0, 1])
    tests.append("XYZ collision geometry and WXYZ convention")
    origin = [0., 0., .02]
    measure = pill_region_measurement(origin, vertices, [0, 0, .005],
                                     initial_quaternion=[1, 0, 0, 0], current_quaternion=[0, 0, 0, 1])
    assert measure["inside"] and measure["rotation_from_initial_rad"] == np.pi
    boundary = np.asarray(list(itertools.product([-.08, .08], [-.08, .08], [0., .2])))
    assert pill_region_measurement(origin, boundary, [0, 0, .005])["inside"]
    protruding = boundary.copy(); protruding[-1, 0] += 1e-6
    failure = pill_region_measurement(origin, protruding, [0, 0, .005])
    assert failure["official_origin_inside"] and not failure["inside"]
    sinking = vertices.copy(); sinking[:, 2] -= .006
    assert not pill_region_measurement(origin, sinking, [0, 0, .005])["inside"]
    tests.append("pill full geometry, inclusive XY/height boundary, diagnostic orientation")
    plate_bounds = [[-.15, -.15, 0], [.15, .15, .02]]
    assert plate_region_measurement(origin, vertices, [0, 0, .01], plate_world_bounds=plate_bounds)["inside"]
    circle = np.array([[.13, 0, 0], [0, .13, .04], [-.13, 0, .02]])
    assert plate_region_measurement(origin, circle, [0, 0, .01])["inside"]
    circle[0, 0] += 1e-6
    assert not plate_region_measurement(origin, circle, [0, 0, .01])["inside"]
    tests.append("plate full disk and source-independent destination support")
    rejected = 0
    for operation in (
        lambda: transform_local_vertices([], [0]*3, [1, 0, 0, 0]),
        lambda: quaternion_matrix_wxyz([0]*4),
        lambda: pill_region_measurement(origin, vertices, [0]*3, half_size_xy=[0, .08]),
        lambda: pill_region_measurement(origin, vertices, [0]*3, current_quaternion=[1, 0, 0, 0]),
        lambda: plate_region_measurement(origin, vertices, [0]*3, radius=float("nan")),
        lambda: plate_region_measurement(origin, vertices, [0]*3, plate_world_bounds=[[0, 0, 1], [0, 0, 0]]),
        lambda: transform_local_vertices([[0, 0, float("nan")]], [0]*3, [1, 0, 0, 0]),
    ):
        try:
            operation()
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Malformed geometry accepted")
    assert rejected == 7
    tests.append("seven malformed geometry and quaternion inputs rejected")
    from types import SimpleNamespace
    def actor(name: str, position: list[float]) -> Any:
        matrix = np.eye(4); matrix[:3, 3] = position
        return SimpleNamespace(name=name, pose=SimpleNamespace(p=np.array([position]), q=np.array([[1, 0, 0, 0]]),
            to_transformation_matrix=lambda: matrix[None].copy()),
            get_first_collision_mesh=lambda to_world_frame=False: SimpleNamespace(vertices=vertices.copy()))
    table = actor("table", [0, 0, 0])
    table.get_first_collision_mesh = lambda to_world_frame=False: SimpleNamespace(vertices=np.array([[-1., -1., -.1], [1., 1., 0.]]))
    grasp_flags = [False, False]
    agents = [SimpleNamespace(tcp=actor("tcp", [0, 0, .2]), is_grasping=lambda target, i=i: grasp_flags[i]) for i in range(2)]
    raw = SimpleNamespace(agent=SimpleNamespace(agents=agents), table_scene=SimpleNamespace(table=table),
        pill_a=actor("pill_a", [-.25, -.15, .03]), pill_b=actor("pill_b", [-.25, .15, .03]),
        region_a=actor("region_a", [.1, -.2, .005]), region_b=actor("region_b", [.1, .2, .005]))
    adapter = get_task_adapter("pill", raw)
    assert adapter.initial_supports == (table, table) and adapter.destination_supports == (table, table)
    assert adapter.configuration()["selected_release_tcp_xyz_m"] is None
    assert adapter.configuration()["release_geometry_capture"] is None
    try:
        adapter.release_positions()
    except ValueError:
        pass
    else:
        raise AssertionError("Reset-period pill release geometry was estimated without a grasp")
    assert adapter.carry_quaternions().shape == (2, 4)
    assert not adapter.configuration()["official_static_or_upright_required"]
    try:
        adapter.release_poses(carry_quaternions=adapter.carry_quaternions())
    except ValueError:
        pass
    else:
        raise AssertionError("Unconfirmed physical grasp accepted for release capture")
    grasp_flags[:] = [True, True]
    release = adapter.release_poses(carry_quaternions=adapter.carry_quaternions())
    assert np.allclose([pose.p for pose in release], [[.1, -.2, .185], [.1, .2, .185]])
    assert adapter.configuration()["release_geometry_capture"]["capture_count"] == 1
    repeat = adapter.release_poses(carry_quaternions=adapter.carry_quaternions())
    assert np.array_equal([pose.p for pose in release], [pose.p for pose in repeat])
    assert adapter.configuration()["release_geometry_capture"]["capture_count"] == 1
    fruit_raw = copy.copy(raw)
    fruit_raw.fruit_a, fruit_raw.fruit_b = actor("fruit_a", [-.25, -.15, .03]), actor("fruit_b", [-.25, .15, .03])
    fruit_raw.plate_left, fruit_raw.plate_right, fruit_raw.plate = actor("plate_left", [-.25, -.2, 0]), actor("plate_right", [-.25, .2, 0]), actor("plate", [.02, 0, .01])
    goal_plate_vertices = np.array(list(itertools.product([-.14, .14], [-.14, .14], [0., .03195])))
    fruit_raw.plate.get_first_collision_mesh = lambda to_world_frame=False: SimpleNamespace(vertices=goal_plate_vertices.copy())
    fruit_grasp_flags = [False, False]
    fruit_agents = [SimpleNamespace(tcp=actor("fruit_tcp", [-.25, (-.15,.15)[i], (.08,.09)[i]]),
        is_grasping=lambda target, i=i: fruit_grasp_flags[i]) for i in range(2)]
    fruit_raw.agent = SimpleNamespace(agents=fruit_agents)
    fruit = get_task_adapter("fruits", fruit_raw)
    assert fruit.initial_support_names == ("plate_left", "plate_right")
    assert fruit.destination_support_names == ("plate", "plate")
    fruit_reset_snapshot = fruit.configuration()
    assert fruit_reset_snapshot["selected_release_tcp_xyz_m"] is None
    assert fruit_reset_snapshot["release_geometry_capture"] is None
    assert fruit_reset_snapshot["release_geometry_capture_counts"] == [0, 0]
    assert fruit_reset_snapshot["planned_fruit_release_clearance_m"] == .015
    for operation in (lambda: fruit.release_positions(), lambda: fruit.release_poses(fruit.carry_quaternions())):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("Reset-period fruit release geometry fabricated without actual grasps")
    fruit_grasp_flags[:] = [True, False]
    try:
        fruit.release_poses(fruit.carry_quaternions())
    except ValueError:
        pass
    else:
        raise AssertionError("Fruit pair capture accepted while peer not grasping")
    assert fruit.configuration()["release_geometry_capture_counts"] == [0, 0]
    fruit_left = fruit.release_pose_for_arm(0, fruit.carry_quaternion_for_arm(0))
    fruit_left_snapshot = fruit.configuration()
    assert fruit_left_snapshot["release_geometry_capture_counts"] == [1, 0]
    assert fruit_left_snapshot["selected_release_tcp_xyz_m"][1] is None
    assert fruit_left_snapshot["release_geometry_capture"] is None
    fruit_grasp_flags[:] = [False, True]
    fruit_right = fruit.release_pose_for_arm(1, fruit.carry_quaternion_for_arm(1))
    assert np.allclose(fruit.release_positions(), [[.02, -.075, .10695], [.02, .075, .11695]])
    fruit_snapshot = fruit.configuration()
    assert fruit_snapshot["release_geometry_capture_counts"] == [1, 1]
    assert fruit_snapshot["release_geometry_capture"]["policy"] == FRUIT_RELEASE_POLICY
    for capture in fruit_snapshot["release_geometry_captures"]:
        assert capture["capture_count"] == 1 and capture["actual_grasp_confirmed"]
        assert np.isclose(capture["support_geometry"]["support_top_z_m"], .04195)
        assert np.isclose(capture["support_geometry"]["predicted_minimum_collision_vertex_world_z_m"], .05695)
        assert capture["support_geometry"]["physical_clearance_m"] == .015
    fruit_agents[0].tcp = actor("fruit_tcp_moved_after_capture", [9., 9., 9.])
    assert np.array_equal(fruit.release_pose_for_arm(0, fruit.carry_quaternion_for_arm(0)).p, fruit_left.p)
    assert np.array_equal(fruit.release_pose_for_arm(1, fruit.carry_quaternion_for_arm(1)).p, fruit_right.p)
    assert fruit.configuration()["release_geometry_capture_counts"] == [1, 1]
    fruit_left_snapshot["release_geometry_captures"][0]["capture_count"] = 999
    assert fruit.configuration()["release_geometry_capture_counts"] == [1, 1]
    assert fruit_reset_snapshot["selected_release_tcp_xyz_m"] is None
    assert fruit.configuration()["source_plate_support_required"]
    expected_yaw=np.array([np.cos(np.pi/8),0,0,np.sin(np.pi/8)])
    for side in range(2):
        assert np.allclose(fruit.carry_quaternion_for_arm(side), quaternion_multiply_wxyz(expected_yaw, _single(fruit_agents[side].tcp.pose.q,4,'test TCP')))
    try:
        fruit.release_pose_for_arm(0, [0,1,0,0])
    except ValueError:
        pass
    else:
        raise AssertionError("Fruit release quaternion changed after one-time capture")
    assert fruit.configuration()["fruit_release_height_evidence"]["serial_official_fixed_release_waypoints_unchanged"]
    json.dumps([adapter.configuration(), fruit.configuration()])
    tests.append("native-free task config, canonical left/right, real support identities")
    tests.append("fruit once-per-side actual attachment and full geometry gives15mm gap above highest goal-plate vertex; no peer dependency or recapture")
    attachment = np.eye(4); attachment[:3, 3] = [0, 0, -.13]
    height = support_clearance_tcp_height(vertices, attachment, [1, 0, 0, 0], support_top_z=0., clearance=.015)
    assert np.isclose(height["target_tcp_z_m"], .145)
    assert np.isclose(height["predicted_minimum_collision_vertex_world_z_m"], .015)
    # A world half-turn in X reverses the object and its measured TCP offset.
    flipped = support_clearance_tcp_height(vertices, attachment, [0, 1, 0, 0], support_top_z=.02, clearance=.015)
    assert np.isclose(flipped["target_tcp_z_m"], -.055)
    assert np.isclose(flipped["predicted_minimum_collision_vertex_world_z_m"], .035)
    for transform, gap in ((np.zeros((4, 4)), .015), (np.diag([2., 1, 1, 1]), .015), (np.eye(4), 0.)):
        try:
            support_clearance_tcp_height(vertices, transform, [1, 0, 0, 0], support_top_z=0., clearance=gap)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid attachment or nonpositive clearance accepted")
    tests.append("one-time post-grasp full-geometry release height and positive physical gap")
    serial = get_task_adapter("pill", raw)
    grasp_flags[:] = [True, False]
    reset_snapshot = serial.configuration()
    try:
        serial.release_poses(serial.carry_quaternions())
    except ValueError:
        pass
    else:
        raise AssertionError("Paired capture accepted before right grasp")
    assert serial.configuration()["release_geometry_capture_counts"] == [0, 0]
    left_pose = serial.release_pose_for_arm(0, serial.carry_quaternion_for_arm(0))
    left_snapshot = serial.configuration()
    assert left_snapshot["release_geometry_capture_counts"] == [1, 0]
    assert left_snapshot["selected_release_tcp_xyz_m"][1] is None
    assert left_snapshot["release_geometry_capture"] is None
    grasp_flags[:] = [False, True]
    right_pose = serial.release_pose_for_arm(1, serial.carry_quaternion_for_arm(1))
    assert serial.configuration()["release_geometry_capture_counts"] == [1, 1]
    assert serial.configuration()["release_geometry_capture"]["total_arm_captures"] == 2
    assert np.allclose([left_pose.p, right_pose.p], [[.1, -.2, .185], [.1, .2, .185]])
    assert left_snapshot["release_geometry_capture_counts"] == [1, 0]
    assert reset_snapshot["release_geometry_capture_counts"] == [0, 0]
    assert left_snapshot["release_geometry_captures"][1] is None
    left_snapshot["release_geometry_captures"][0]["capture_count"] = 999
    assert serial.configuration()["release_geometry_capture_counts"] == [1, 1]
    assert np.array_equal(serial.release_pose_for_arm(0, [1, 0, 0, 0]).p, left_pose.p)
    for operation in (lambda: serial.carry_quaternion_for_arm(-1),
                      lambda: serial.release_pose_for_arm(True, [1, 0, 0, 0]),
                      lambda: serial.carry_quaternion_for_arm(1.),
                      lambda: serial.release_pose_for_arm(0, [0, 1, 0, 0])):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid side index or changed captured orientation accepted")
    tests.append("serial side order, paired atomic preflight, per-arm capture once, deep-copy evidence")
    for task_id, approach in zip(TASK_IDS, ((.9, .9), (1., 1.3), (1.5, 2.6))):
        parameters = task_execution_parameters(task_id)
        assert (parameters["approach_vel_limit"], parameters["approach_acc_limit"]) == approach
        assert parameters["joint_vel_limit"] == 1.5 and parameters["joint_acc_limit"] == 2.6
        assert parameters["preparation_steps"] == 10 and parameters["max_episode_steps"] == 200
        assert not parameters["physical_acceptance_thresholds_changed"]
        parameters["approach_phases"].append("bad")
        assert task_execution_parameters(task_id)["approach_phases"] == ["pregrasp", "descend"]
    for invalid in ("unknown-task", None, []):
        try:
            task_execution_parameters(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported task parameters accepted")
    tests.append("pure pre-environment task phase limits and malformed task rejection")
    assert np.isclose(PILL_SIDE_CONTACT_METADATA_HEIGHT_M, .0724783992767334)
    assert PILL_GRASP_HEIGHTS_M == (.12, .13) and PILL_PREGRASP_HEIGHTS_M == (.21, .21)
    assert np.allclose(np.array(PILL_PREGRASP_HEIGHTS_M)-PILL_GRASP_HEIGHTS_M, [.09, .08])
    import sapien
    for agent in agents:
        agent.build_grasp_pose = lambda approaching, closing, center: sapien.Pose(p=center, q=[0, 1, 0, 0])
    grasp_pairs = serial.grasp_poses()
    for index, pair in enumerate(grasp_pairs):
        assert np.allclose(pair[0].p, np.array(raw.pill_a.pose.p[0] if index == 0 else raw.pill_b.pose.p[0])+[0, 0, PILL_PREGRASP_HEIGHTS_M[index]])
        assert np.allclose(pair[1].p, np.array(raw.pill_a.pose.p[0] if index == 0 else raw.pill_b.pose.p[0])+[0, 0, PILL_GRASP_HEIGHTS_M[index]])
    fake_limits = {'joint_vel_limit_rad_s':1.5,'joint_acc_limit_rad_s2':2.6,
                   'approach_joint_vel_limit_rad_s':1.,'approach_joint_acc_limit_rad_s2':1.3,'solver_limit_fraction':.99}
    assert task_phase_limits(fake_limits,'pill','carry_lower')['nominal_acceleration_limit_rad_s2'] == 2.6
    assert task_phase_limits(fake_limits,'pill','lift_and_hold')['nominal_acceleration_limit_rad_s2'] == 2.6
    assert task_phase_limits(fake_limits,'fruits','carry_lower')['nominal_acceleration_limit_rad_s2'] == 2.6
    assert serial.configuration()["grasp_policy"] == PILL_GRASP_POLICY
    assert np.allclose(serial.configuration()["pregrasp_local_tcp_offset_m"], [[0,0,-.09],[0,0,-.08]])
    from .validate_sync_compact import compact_phase_limits
    for task_id in TASK_IDS:
        for phase in ("pregrasp", "descend", "lift_and_hold", "carry_orient_and_transport", "carry_lower", "retreat"):
            assert task_phase_limits(fake_limits, task_id, phase) == compact_phase_limits(fake_limits, phase)
    assert serial.configuration()["pill_grasp_height_above_actor_origin_m"] == [.12,.13]
    assert serial.configuration()["task_execution_parameters"]["task_phase_overrides"] == {}
    tests.append("fixed pill_a-model1 height12cm and pill_b-model5 height13cm with both pregrasp21cm")
    tests.append("all task phase limits exactly reuse original common resolver; no v6 lowering override")
    tall = vertices.copy(); tall[:, 2] *= 3
    positive = plate_region_measurement([0, 0, .04], tall, [0, 0, .01], plate_world_bounds=plate_bounds)
    assert positive["inside"] and positive["object_top_z_m"] > .10
    assert not positive["object_top_within_origin_height_threshold"]
    assert not positive["object_top_height_is_acceptance_gate"]
    assert not plate_region_measurement([0, 0, .100001], tall, [0, 0, .01], plate_world_bounds=plate_bounds)["inside"]
    outside = tall.copy(); outside[-1, 0] = .130001
    assert not plate_region_measurement([0, 0, .04], outside, [0, 0, .01], plate_world_bounds=plate_bounds)["inside"]
    below = tall.copy(); below[0, 2] = -.005001
    assert not plate_region_measurement([0, 0, .04], below, [0, 0, .01], plate_world_bounds=plate_bounds)["inside"]
    tests.append("fruit height applies to native actor origin; tall valid fruit positive and height/disk/floor negatives")
    # Contents contact legality is limited by canonical actor identity, phase
    # and current complete geometry, independently of raw contact forces.
    apple = actor("apple_out", [0, 0, .03]); apple_in = actor("apple_in", [0, 0, .04])
    banana = actor("banana_out", [.5, 0, .03]); banana_in = actor("banana_in", [.5, 0, .04])
    basket_a = actor("basket_a", [0., 0., 0.]); basket_b = actor("basket_b", [.5, 0., 0.])
    basket_vertices = np.array(list(itertools.product([-.15, .15], [-.15, .15], [0., .12])))
    for basket in (basket_a, basket_b):
        basket.get_first_collision_mesh = lambda to_world_frame=False: SimpleNamespace(vertices=basket_vertices.copy())
    apple_raw = SimpleNamespace(agent=SimpleNamespace(agents=agents), table_scene=SimpleNamespace(table=table),
        apple_out=apple, banana_out=banana, apple_in=apple_in, banana_in=banana_in,
        basket_apple=basket_a, basket_banana=basket_b,
        scene=SimpleNamespace(actors={a.name: a for a in (table, apple, banana, apple_in, banana_in, basket_a, basket_b)}))
    apple_adapter = get_task_adapter("apple-banana", apple_raw)
    point = {"position_m": [0., 0., .05], "normal": [0., 0., 1.], "impulse_Ns": [0., 0., .01], "separation_m": -.001}
    original_point = copy.deepcopy(point)
    for phase in DESTINATION_CONTENTS_PHASES:
        proof = apple_adapter.destination_contents_contact_proof(0, "apple_in", phase, [point])
        assert proof["allowed"] and proof["geometry_evaluated"]
        assert proof["geometry"]["geometry_vertex_counts"] == [8, 8, 8]
        assert proof["native_geometry"]["contents"]["state_name"] == "apple_in"
        assert point == original_point
    assert apple_adapter.destination_contents_contact_proof(1, "banana_in", "release", [{**point, "position_m": [.5, 0, .05]}])["allowed"]
    assert not adapter.destination_contents and not fruit.destination_contents
    assert not adapter.configuration()["destination_contents_contact_policy"]["enabled"]
    assert not fruit.configuration()["destination_contents_contact_policy"]["enabled"]
    assert apple_adapter.configuration()["destination_contents_contact_policy"]["pairs"][0]["contents_actor_name"] == "apple_in"
    tests.append("contents paired identities placement phases full native geometry hashes and raw fields preserved; other tasks disabled")
    for phase in ("preparation", "close_and_confirm", "lift_and_hold", "carry_orient_and_transport", "carry_lower"):
        proof = apple_adapter.destination_contents_contact_proof(0, "apple_in", phase, [point])
        assert not proof["allowed"] and not proof["phase_allowed"] and not proof["geometry_evaluated"]
    for wrong in ("banana_in", "table", "basket_a", "panda_hand", "apple_out"):
        proof = apple_adapter.destination_contents_contact_proof(0, wrong, "release", [point])
        assert not proof["allowed"] and not proof["identity_allowed"]
    tests.append("contents carry wrong phase cross side table basket palm and target identities rejected")
    apple_in.pose.p[0, 0] = .3
    assert not apple_adapter.destination_contents_contact_proof(0, "apple_in", "release", [point])["allowed"]
    apple_in.pose.p[0, 0] = 0
    apple.pose.p[0, 0] = .2
    assert not apple_adapter.destination_contents_contact_proof(0, "apple_in", "release", [point])["allowed"]
    apple.pose.p[0, 0] = 0
    basket_a.pose.p[0, 0] = .4
    assert not apple_adapter.destination_contents_contact_proof(0, "apple_in", "release", [point])["allowed"]
    basket_a.pose.p[0, 0] = 0
    for position in ([.150001, 0, .05], [0, 0, -.005001], [0, 0, .130001]):
        proof = apple_adapter.destination_contents_contact_proof(0, "apple_in", "release", [{**point, "position_m": position}])
        assert not proof["allowed"] and not proof["geometry"]["contact_points_inside"]
    assert apple_adapter.destination_contents_contact_proof(0, "apple_in", "release", [{**point, "position_m": [.15, 0, .05]}])["allowed"]
    assert not apple_adapter.destination_contents_contact_proof(0, "apple_in", "release", [])["allowed"]
    tests.append("contents current target contents basket poses and strict contact point XY height boundaries rechecked")
    original_mesh = apple_in.get_first_collision_mesh
    for invalid in (None, np.empty((0, 3)), np.array([[float('nan'), 0, 0]]), np.zeros((4, 3))):
        apple_in.get_first_collision_mesh = lambda to_world_frame=False, v=invalid: None if v is None else SimpleNamespace(vertices=v)
        proof = apple_adapter.destination_contents_contact_proof(0, "apple_in", "release", [point])
        assert not proof["allowed"] and proof["reason"] == "invalid_current_native_geometry"
    apple_in.get_first_collision_mesh = original_mesh
    assert apple_adapter.destination_contents_contact_proof(0, "apple_in", "release", [point])["allowed"]
    snapshot = apple_adapter.destination_contents_pose_snapshot()
    assert np.asarray(snapshot['destination_contents_poses']).shape == (2, 7)
    assert np.asarray(snapshot['destination_basket_poses']).shape == (2, 7)
    assert fruit.destination_contents_pose_snapshot() == {'destination_contents_poses': [], 'destination_basket_poses': []}
    tests.append("contents absent malformed empty degenerate geometry rejected without stale cached acceptance")
    return {"status": "passed", "adapter_version": ADAPTER_VERSION, "tests": tests,
            "checks": len(tests), "rejected_malformed_inputs": rejected,
            "array_shapes": {"vertices": list(vertices.shape), "world_vertices": list(transformed.shape),
                             "paired_release": list(fruit.release_positions().shape), "paired_quaternions": [2, 4]},
            "native_environments_created": 0, "physics_steps": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
