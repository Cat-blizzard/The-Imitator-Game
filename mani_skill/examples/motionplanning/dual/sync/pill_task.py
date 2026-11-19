"""Geometry and phase goals for the optional synchronous pill task."""
from __future__ import annotations

from typing import Any
import numpy as np
import sapien

from ._arrays import as_float_array as _array

TASK_ID = "TwoRobotPickPillToRegions-v1"

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

class PillTask:
    """Task-specific targets; all state changes belong to the executor."""

    target_names = ("pill_a", "pill_b")
    official_predicate_keys = ("pill_a_in_region", "pill_b_in_region")

    def __init__(self, raw: Any):
        self.raw = raw
        self.targets = raw.pill_a, raw.pill_b
        self.destination_actors = raw.region_a, raw.region_b
        self.table = raw.table_scene.table
        self.initial_supports = self.destination_supports = (self.table, self.table)
        self.local_vertices = {}
        self.initial_quaternions = tuple(_single(a.pose.q, 4, "initial WXYZ") for a in self.targets)

    def vertices(self, actor: Any) -> np.ndarray:
        if id(actor) not in self.local_vertices:
            mesh = actor.get_first_collision_mesh(to_world_frame=False)
            if mesh is None:
                raise ValueError(f"Missing collision mesh: {actor.name}")
            self.local_vertices[id(actor)] = _array(mesh.vertices).copy()
        return transform_local_vertices(self.local_vertices[id(actor)],
            _single(actor.pose.p, 3, "XYZ"), _single(actor.pose.q, 4, "WXYZ"))

    def grasp_poses(self) -> tuple:
        agents = self.raw.agent.agents
        profile = self.raw.embodiment_profile
        approaching = np.array([0., 0., -1.])
        closing = None
        if any(quaternion is None for quaternion in profile.grasp_quaternions):
            closing = _array(agents[1].tcp.pose.to_transformation_matrix())[0, :3, 1].copy()
            closing -= np.dot(closing, approaching) * approaching
            if np.linalg.norm(closing) < 1e-8:
                raise ValueError("Degenerate horizontal closing direction")
            closing /= np.linalg.norm(closing)
        pairs = []
        for i, (agent, actor) in enumerate(zip(agents, self.targets)):
            height = profile.grasp_height_offsets_m[i]
            quaternion = profile.grasp_quaternions[i]
            origin = _single(actor.pose.p, 3, "pill origin")
            offsets = profile.grasp_position_offsets_m
            offset = (_finite(offsets[i], (3,), "world grasp offset") if offsets is not None
                      else np.array([0., 0., height]))
            if quaternion is None:
                grasp = agent.build_grasp_pose(approaching, closing,
                    origin if offsets is None else origin+offset)
                if offsets is None:
                    grasp *= sapien.Pose(p=[0., 0., -height])
            else:
                grasp = sapien.Pose(p=origin+offset, q=quaternion)
            pregrasp_offsets = profile.pregrasp_position_offsets_m
            pregrasp = (grasp * sapien.Pose(p=[0., 0., -profile.pregrasp_clearances_m[i]])
                        if pregrasp_offsets is None else sapien.Pose(
                            p=grasp.p+_finite(pregrasp_offsets[i], (3,), "world pregrasp offset"),
                            q=grasp.q))
            pairs.append((pregrasp, grasp))
        return tuple(pairs)

    def release_poses(self) -> tuple:
        top = float(self.vertices(self.table)[:, 2].max())
        poses = []
        for agent, actor, destination in zip(self.raw.agent.agents, self.targets, self.destination_actors):
            if not bool(_array(agent.is_grasping(actor)).item()):
                raise ValueError("Release target requires an actual confirmed grasp")
            tcp = _array(agent.tcp.pose.to_transformation_matrix())[0]
            obj = _array(actor.pose.to_transformation_matrix())[0]
            self.vertices(actor)
            quaternion = _single(agent.tcp.pose.q, 4, "carry WXYZ")
            value = support_clearance_tcp_height(self.local_vertices[id(actor)],
                np.linalg.inv(tcp) @ obj, quaternion, support_top_z=top,
                clearance=self.raw.embodiment_profile.release_clearance_m)
            position = _single(destination.pose.p, 3, "destination XYZ").copy()
            position[2] = value["target_tcp_z_m"]
            poses.append(sapien.Pose(p=position, q=quaternion))
        return tuple(poses)

    def retreat_poses(self, vertical_distance_m: float) -> tuple:
        """Withdraw in configured world XY and raise by the shared policy distance."""
        distance = float(vertical_distance_m)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Retreat vertical distance must be finite and positive")
        offsets = self.raw.embodiment_profile.retreat_horizontal_offsets_m
        poses = []
        for i, agent in enumerate(self.raw.agent.agents):
            xy = np.zeros(2) if offsets is None else _finite(offsets[i], (2,), "world retreat XY")
            position = _single(agent.tcp.pose.p, 3, "retreat XYZ") + np.r_[xy, distance]
            poses.append(sapien.Pose(p=position,
                q=_single(agent.tcp.pose.q, 4, "retreat WXYZ")))
        return tuple(poses)

    def region_predicates(self) -> tuple:
        def make(index):
            def region(actor):
                if actor is not self.targets[index]:
                    raise ValueError("Wrong canonical target")
                return pill_region_measurement(_single(actor.pose.p, 3, "XYZ"), self.vertices(actor),
                    _single(self.destination_actors[index].pose.p, 3, "region XYZ"),
                    half_size_xy=self.raw.region_half_size[:2],
                    height_threshold=self.raw.region_height_threshold,
                    table_top_z=float(self.vertices(self.table)[:, 2].max()),
                    initial_quaternion=self.initial_quaternions[index],
                    current_quaternion=_single(actor.pose.q, 4, "WXYZ"))
            return region
        return make(0), make(1)


if __name__ == "__main__":
    local = np.array([[x, y, z] for x in (-.01, .01)
                      for y in (-.01, .01) for z in (0., .02)])
    quaternion = np.array([1. + 1e-7, 0., 0., 0.])
    before = quaternion.copy()
    vertices = transform_local_vertices(local, [0., 0., .015], quaternion)
    measurement = pill_region_measurement([0., 0., .015], vertices, [0., 0., 0.])
    assert vertices.shape == (8, 3) and measurement["inside"]
    assert np.array_equal(quaternion, before)
    clearance = support_clearance_tcp_height(local, np.eye(4), quaternion,
                                             support_top_z=0., clearance=.015)
    assert np.isclose(clearance["predicted_minimum_collision_vertex_world_z_m"], .015)
    print("pill geometry CPU check passed: vertices (8,3), quaternion (4,)")
