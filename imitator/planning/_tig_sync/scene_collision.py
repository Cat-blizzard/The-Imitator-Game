"""Export a single CPU scene's collision geometry without advancing physics.

Each result is an MPLib 0.1.1 ``fcl.CollisionObject`` at its world pose.
Robot articulations are checked separately by the combined robot model. Dynamic
actors are a snapshot: call this exporter again after their poses change.
PhysX planes bound a solid negative half-space; FCL's Plane is a two-sided
surface. ``coverage['planes']`` supplies the half-space equations for the caller
to additionally reject shapes wholly below the surface.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import sapien
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl


class SceneCollisionExportError(ValueError):
    """An incomplete export, with the partial coverage report attached."""

    def __init__(self, message: str, coverage: dict[str, Any]) -> None:
        super().__init__(message)
        self.coverage = coverage


def _reject(coverage: dict[str, Any], reason: str, **details: Any) -> None:
    coverage["unsupported"].append({"reason": reason, **details})
    raise SceneCollisionExportError(reason, coverage)


def _positive(values: Any, name: str, *, allow_zero: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all() or not (
        (result >= 0).all() if allow_zero else (result > 0).all()
    ):
        raise ValueError(f"invalid {name}: {result}")
    return result


def _mesh(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(shape.vertices, dtype=np.float64)
    faces = np.asarray(shape.get_triangles(), dtype=np.int32).reshape(-1, 3)
    scale = _positive(shape.scale, "mesh scale")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or scale.shape != (3,):
        raise ValueError("invalid mesh vertices or scale dimensions")
    if not np.isfinite(vertices).all() or len(vertices) < 3 or len(faces) == 0:
        raise ValueError("empty or nonfinite collision mesh")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("collision mesh index is out of bounds")
    return np.ascontiguousarray(vertices * scale), np.ascontiguousarray(faces)


def _geometry(shape: Any) -> tuple[Any, sapien.Pose]:
    """Return shape-local FCL geometry and the FCL-to-PhysX frame rotation."""
    identity = sapien.Pose()
    if isinstance(shape, physx.PhysxCollisionShapeBox):
        sides = 2 * _positive(shape.half_size, "box half size")
        return fcl.Box(sides), identity
    if isinstance(shape, physx.PhysxCollisionShapeSphere):
        return fcl.Sphere(float(_positive(shape.radius, "sphere radius"))), identity
    if isinstance(shape, physx.PhysxCollisionShapePlane):
        # PhysX uses the local +X axis for a plane's outward normal.
        return fcl.Plane(np.array([1.0, 0.0, 0.0]), 0.0), identity
    if isinstance(shape, (physx.PhysxCollisionShapeCapsule, physx.PhysxCollisionShapeCylinder)):
        radius = float(_positive(shape.radius, "radius"))
        length = 2 * float(_positive(shape.half_length, "half length", allow_zero=True))
        cls = fcl.Capsule if isinstance(shape, physx.PhysxCollisionShapeCapsule) else fcl.Cylinder
        # FCL's longitudinal Z axis becomes PhysX's longitudinal X axis.
        rotation = sapien.Pose(q=[2**-0.5, 0.0, 2**-0.5, 0.0])
        return cls(radius, length), rotation
    if isinstance(shape, physx.PhysxCollisionShapeConvexMesh):
        vertices, faces = _mesh(shape)
        return fcl.Convex(vertices, faces, True), identity
    if isinstance(shape, physx.PhysxCollisionShapeTriangleMesh):
        vertices, faces = _mesh(shape)
        model = fcl.BVHModel()
        status = model.beginModel(len(faces), len(vertices))
        if status != 0:
            raise ValueError(f"FCL beginModel failed: {status}")
        status = model.addSubModel(vertices, faces)
        if status not in (None, 0):
            raise ValueError(f"FCL addSubModel failed: {status}")
        if model.endModel() != 0:
            raise ValueError("FCL endModel failed")
        return model, identity
    raise TypeError(f"unsupported collision shape {type(shape).__name__}")


def _robot_articulations(env: Any) -> set[Any]:
    agent = getattr(env, "agent", None)
    agents = getattr(agent, "agents", [agent]) if agent is not None else []
    result: set[Any] = set()
    for entry in agents:
        robot = getattr(entry, "robot", None)
        if robot is not None:
            result.update(getattr(robot, "_objs", []))
    return result


def scene_entities(env: Any) -> dict[str, Any]:
    """Return registered actors plus raw nonarticulation physics entities.

    Stable explicit keys such as ``physics::wall`` identify additional entities
    so an executor can monitor their poses as well as registered actor poses.
    Robot articulation links are intentionally handled by the robot model.
    """
    base = getattr(env, "unwrapped", env)
    scene = base.scene
    if getattr(base, "num_envs", None) != 1 or len(scene.sub_scenes) != 1:
        raise ValueError("only a single environment is supported")
    if getattr(scene, "gpu_sim_enabled", False) or not isinstance(scene.px, physx.PhysxCpuSystem):
        raise ValueError("only CPU physics is supported")
    entities: dict[str, Any] = {}
    for name, actor in scene.actors.items():
        instances = list(getattr(actor, "_objs", []))
        if len(instances) != 1:
            raise ValueError(f"actor {name} does not represent one entity")
        entities[str(name)] = instances[0]
    registered = set(entities.values())
    bodies = (*scene.px.get_rigid_static_components(), *scene.px.get_rigid_dynamic_components())
    for body in bodies:
        if isinstance(body, physx.PhysxArticulationLinkComponent) or body.entity in registered:
            continue
        entity = body.entity
        stem = f"physics::{entity.name or 'unnamed'}"
        name, suffix = stem, 1
        while name in entities:
            name = f"{stem}#{suffix}"
            suffix += 1
        entities[name] = entity
        registered.add(entity)
    return entities


def export_scene_collision(env: Any) -> tuple[list[Any], list[str], dict[str, Any]]:
    """Export registered and raw nonrobot collision shapes, or raise on omissions.

    Input: a single-environment ManiSkill CPU environment after reset. Output:
    parallel lists of collision objects/names and a JSON-compatible report.
    ``all_covered`` means geometry export coverage, not half-space enforcement,
    continuous collision checking, or prediction of moving obstacles.
    """
    base = getattr(env, "unwrapped", env)
    scene = base.scene
    coverage: dict[str, Any] = {
        "all_covered": False,
        "snapshot_only": True,
        "shape_types": {},
        "shapes": [],
        "planes": [],
        "skipped_visual_actors": [],
        "robot_articulations": [],
        "dynamic_actors": [],
        "additional_physics_entities": [],
        "unsupported": [],
        "plane_halfspace_check_required": False,
    }
    if getattr(base, "num_envs", None) != 1 or len(scene.sub_scenes) != 1:
        _reject(coverage, "only a single environment is supported")
    if getattr(scene, "gpu_sim_enabled", False) or not isinstance(scene.px, physx.PhysxCpuSystem):
        _reject(coverage, "only CPU physics is supported")
    accepted_robots = _robot_articulations(base)
    for name, articulation in scene.articulations.items():
        instances = list(getattr(articulation, "_objs", []))
        if len(instances) != 1 or instances[0] not in accepted_robots:
            _reject(coverage, "nonrobot or batched articulation", name=str(name))
        coverage["robot_articulations"].append(str(name))
    for link in scene.px.get_articulation_link_components():
        if link.articulation not in accepted_robots:
            _reject(coverage, "unregistered nonrobot articulation", name=link.entity.name)

    objects: list[Any] = []
    names: list[str] = []
    shape_types: Counter[str] = Counter()
    try:
        entities = scene_entities(base)
    except Exception as exc:
        _reject(coverage, f"failed to enumerate scene entities: {exc}")
    for actor_name, entity in entities.items():
        bodies = [c for c in entity.components if isinstance(c, physx.PhysxRigidBaseComponent)]
        if len(bodies) > 1:
            _reject(coverage, "multiple rigid bodies on one entity", actor=str(actor_name))
        if bodies and isinstance(bodies[0], physx.PhysxArticulationLinkComponent):
            _reject(coverage, "articulation link registered as actor", actor=str(actor_name))
        shapes = list(bodies[0].get_collision_shapes()) if bodies else []
        if actor_name not in scene.actors:
            coverage["additional_physics_entities"].append({
                "name": actor_name, "entity_name": entity.name, "shape_count": len(shapes),
            })
        if not shapes:
            coverage["skipped_visual_actors"].append(str(actor_name))
            continue
        if isinstance(bodies[0], physx.PhysxRigidDynamicComponent) and not bodies[0].kinematic:
            coverage["dynamic_actors"].append(str(actor_name))
        for index, shape in enumerate(shapes):
            name = f"scene::{actor_name}::shape_{index}"
            shape_type = type(shape).__name__
            try:
                geometry, correction = _geometry(shape)
                shape_pose = entity.pose * shape.get_local_pose()
                pose = shape_pose * correction
                p, q = np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)
                if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
                    raise ValueError("invalid world pose")
                obj = fcl.CollisionObject(geometry, p, q)
            except Exception as exc:
                _reject(coverage, f"{type(exc).__name__}: {exc}", name=name, shape_type=shape_type)
            objects.append(obj)
            names.append(name)
            shape_types[shape_type] += 1
            coverage["shapes"].append({"name": name, "type": shape_type, "world_pose_wxyz": np.r_[p, q].tolist()})
            if isinstance(shape, physx.PhysxCollisionShapePlane):
                normal = shape_pose.to_transformation_matrix()[:3, 0].astype(np.float64)
                coverage["planes"].append({
                    "name": name, "normal": normal.tolist(),
                    "offset": float(normal @ shape_pose.p), "solid_side": "normal_dot_x_le_offset",
                })
    if len(names) != len(set(names)):
        _reject(coverage, "duplicate exported object names")
    coverage["shape_types"] = dict(shape_types)
    coverage["shape_count"] = len(objects)
    coverage["actor_count"] = len(scene.actors)
    coverage["entity_count"] = len(entities)
    coverage["plane_halfspace_check_required"] = bool(coverage["planes"])
    coverage["all_covered"] = True
    return objects, names, coverage


def check_plane_halfspaces(models: list[Any], planes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conservatively reject robot bounding spheres touching a solid half-space.

    The permitted region is ``normal @ x > offset`` for every plane. Each FCL
    geometry's local AABB bounding sphere is transformed by its collision
    object's current pose. A rejection may be a false positive; a wholly
    submerged object is rejected too. Invalid geometry/plane data also returns
    a rejection, never an empty successful result.
    """
    violations: list[dict[str, Any]] = []
    normalized: list[tuple[str, np.ndarray, float]] = []
    for index, plane in enumerate(planes):
        try:
            normal = np.asarray(plane["normal"], dtype=np.float64).reshape(3)
            offset = float(plane["offset"])
            norm = float(np.linalg.norm(normal))
            if not np.isfinite(normal).all() or not np.isfinite(offset) or not np.isfinite(norm) or norm <= 0:
                raise ValueError("plane normal/offset is nonfinite or normal is zero")
            normalized.append((str(plane.get("name", index)), normal / norm, offset / norm))
        except Exception as exc:
            violations.append({"collision_type": "conservative_plane_halfspace", "plane_index": index,
                               "reason": f"invalid_plane: {exc}", "conservative": True})
    if not normalized:
        return violations
    if not models:
        return [*violations, {"collision_type": "conservative_plane_halfspace",
                              "reason": "missing_collision_models", "conservative": True}]
    for model_index, model in enumerate(models):
        try:
            collision_model = model.get_fcl_model()
            objects = collision_model.get_collision_objects()
            links = collision_model.get_collision_link_names()
            if not objects or len(objects) != len(links):
                raise ValueError("empty collision model or object/link name counts differ")
        except Exception as exc:
            violations.append({"collision_type": "conservative_plane_halfspace", "model_index": model_index,
                               "reason": f"invalid_collision_model: {exc}", "conservative": True})
            continue
        for obj, link in zip(objects, links):
            record = {"collision_type": "conservative_plane_halfspace", "model_index": model_index,
                      "link_name": str(link), "conservative": True}
            try:
                geometry = obj.get_collision_geometry()
                geometry.computeLocalAABB()
                center = np.asarray(geometry.aabb_center, dtype=np.float64).reshape(3)
                radius = float(geometry.aabb_radius)
                rotation = np.asarray(obj.get_rotation(), dtype=np.float64).reshape(3, 3)
                translation = np.asarray(obj.get_translation(), dtype=np.float64).reshape(3)
                if not np.isfinite(np.r_[center, rotation.ravel(), translation, radius]).all() or radius <= 0:
                    raise ValueError("invalid AABB bounding sphere or transform")
                if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
                    raise ValueError("collision rotation is not a rigid rotation")
                world_center = rotation @ center + translation
                if not np.isfinite(world_center).all():
                    raise ValueError("nonfinite world bounding sphere center")
            except Exception as exc:
                violations.append({**record, "reason": f"invalid_geometry_aabb: {exc}"})
                continue
            for plane_name, normal, offset in normalized:
                bound = float(normal @ world_center - radius - offset)
                if not np.isfinite(bound) or bound <= 0:
                    violations.append({**record, "plane_name": plane_name, "geometry_type": type(geometry).__name__,
                                       "bound": bound if np.isfinite(bound) else None,
                                       "world_center": world_center.tolist(), "radius": radius,
                                       "reason": "bounding_sphere_touches_or_enters_solid_halfspace"})
    return violations
