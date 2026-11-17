"""Live measured scene/finger snapshots for one dual-Panda grasp phase.

Refresh before each control tick. Each check tests a static snapshot with the
supplied arm targets and the last measured physical finger positions. It does
not predict moving obstacles, certify continuous collision freedom, or model
an attached object. No method advances physics or modifies physical collisions.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import sapien
import sapien.physx as physx
from mplib.pymp import PlanningWorld
from mplib.pymp.collision_detection import fcl

from .combined_collision import NativeDualCollisionChecker, _single
from .scene_collision import check_plane_halfspaces, export_scene_collision, scene_entities


FINGER_LINKS = frozenset(("panda_leftfinger", "panda_rightfinger"))


def _pose(robot: Any) -> np.ndarray:
    pose = robot.pose
    result = _single(pose.raw_pose, 7) if hasattr(pose, "raw_pose") else np.r_[
        _single(pose.p, 3), _single(pose.q, 4)]
    if not np.isclose(np.linalg.norm(result[3:]), 1, atol=1e-6):
        raise ValueError("Robot base quaternion must be normalized")
    return result


def _mapping(solver: Any) -> tuple[Any, ...]:
    return (tuple(solver.planner.user_joint_names), tuple(solver.planner.user_link_names),
            tuple(j.get_name() for j in solver.robot.get_active_joints()),
            tuple(solver.env_agent.controller.controllers["arm"].config.joint_names),
            tuple(solver.env_agent.gripper_joint_names), solver.planner.move_group)


def _signature(shape: Any) -> tuple[str, str]:
    """Fingerprint geometry, while permitting updates to the shape's local pose."""
    digest = hashlib.sha256()
    if isinstance(shape, physx.PhysxCollisionShapeBox):
        arrays = (shape.half_size,)
    elif isinstance(shape, physx.PhysxCollisionShapeSphere):
        arrays = ([shape.radius],)
    elif isinstance(shape, (physx.PhysxCollisionShapeCapsule, physx.PhysxCollisionShapeCylinder)):
        arrays = ([shape.radius, shape.half_length],)
    elif isinstance(shape, (physx.PhysxCollisionShapeConvexMesh, physx.PhysxCollisionShapeTriangleMesh)):
        arrays = (shape.vertices, shape.get_triangles(), shape.scale)
    elif isinstance(shape, physx.PhysxCollisionShapePlane):
        arrays = ()
    else:
        raise RuntimeError(f"Unsupported live collision shape: {type(shape).__name__}")
    for value in arrays:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise RuntimeError("Nonfinite collision geometry")
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return type(shape).__name__, digest.hexdigest()


class LiveGraspCollisionChecker:
    """Exact pair allowlists over a refreshed dual-arm collision world."""

    @classmethod
    def from_solvers(cls, env: Any, left_solver: Any, right_solver: Any,
                     target_actor: Any, active_robot: str = "left") -> LiveGraspCollisionChecker:
        if active_robot not in ("left", "right"):
            raise ValueError("active_robot must be left or right")
        self = cls()
        self.env = getattr(env, "unwrapped", env)
        self.solvers = (left_solver, right_solver)
        if any(s.base_env is not self.env for s in self.solvers):
            raise ValueError("Both solvers must share the given environment")
        self.active_robot = active_robot
        self.target_contact_allowed = False
        self.scope = "live_measured_scene_and_fingers_sampled_dual_arm_grasp_no_attachment_or_prediction"
        self._mapping = tuple(_mapping(s) for s in self.solvers)
        self._base_poses = tuple(_pose(s.robot).copy() for s in self.solvers)
        self._robot_instances = {r for s in self.solvers for r in getattr(s.robot, "_objs", [])}
        objects, names, self.coverage = export_scene_collision(self.env)
        entities = scene_entities(self.env)
        if isinstance(target_actor, str):
            if target_actor not in entities:
                raise ValueError("Target actor is absent from the scene")
            target_entity = entities[target_actor]
        elif hasattr(target_actor, "_objs"):
            if len(target_actor._objs) != 1:
                raise ValueError("Target must represent exactly one entity")
            target_entity = target_actor._objs[0]
        else:
            target_entity = target_actor
        target_keys = [key for key, entity in entities.items() if entity == target_entity]
        if len(target_keys) != 1:
            raise ValueError("Target needs one unambiguous scene entity")
        self.target_actor_name = target_keys[0]
        self._entities, self._shape_refs, poses = self._capture_scene()
        if set(self._shape_refs) != set(names):
            raise RuntimeError("Exported and measured shape sets differ")
        self._geometry = {name: obj.get_collision_geometry() for name, obj in zip(names, objects)}
        self._shape_signatures = {name: _signature(shape) for name, (_, shape) in self._shape_refs.items()}
        self._target_names = frozenset(name for name in names
                                       if self._shape_refs[name][0] == target_entity)
        if not self._target_names or any(isinstance(self._geometry[n], fcl.Plane) for n in self._target_names):
            raise ValueError("Grasp target must have finite collision shapes")
        self.allowed_mount_contacts = frozenset((robot, "panda_link0", "scene::table::shape_0")
                                                for robot in ("left", "right"))
        self._mount_pose = poses.get("scene::table::shape_0")
        self._normal_names = tuple(name for name in names if not isinstance(self._geometry[name], fcl.Plane))
        normal_objects = [obj for name, obj in zip(names, objects) if name in self._normal_names]
        self.native = NativeDualCollisionChecker.from_solvers(
            left_solver, right_solver, normal_objects=normal_objects, normal_object_names=self._normal_names)
        self.refresh_count = 0
        self.allowed_contact_observations = 0
        self.last_allowed_contacts: list[dict[str, Any]] = []
        self._fresh = False
        self.coverage.update(scope=self.scope, target_actor=self.target_actor_name,
                             target_shape_names=sorted(self._target_names),
                             allowed_target_links=sorted(FINGER_LINKS), active_robot=active_robot,
                             plane_check_backend="conservative_bounding_sphere_halfspace",
                             plane_objects_excluded_from_native_world=True,
                             continuous_collision_detection=False, dynamic_prediction=False,
                             attached_object=False, native_world_object_count=len(normal_objects))
        self.refresh()
        return self

    @property
    def models(self) -> tuple[Any, ...]:
        return self.native.models

    @property
    def world(self) -> Any:
        return self.native.world

    def _capture_scene(self) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]], dict[str, np.ndarray]]:
        scene = self.env.scene
        for articulation in scene.articulations.values():
            objects = list(getattr(articulation, "_objs", []))
            if len(objects) != 1 or objects[0] not in self._robot_instances:
                raise RuntimeError("Unexpected articulation in live grasp scene")
        for link in scene.px.get_articulation_link_components():
            if link.articulation not in self._robot_instances:
                raise RuntimeError("Unregistered external articulation in live grasp scene")
        entities = scene_entities(self.env)
        shapes: dict[str, tuple[Any, Any]] = {}
        poses: dict[str, np.ndarray] = {}
        for key, entity in entities.items():
            bodies = [c for c in entity.components if isinstance(c, physx.PhysxRigidBaseComponent)]
            if len(bodies) > 1 or (bodies and isinstance(bodies[0], physx.PhysxArticulationLinkComponent)):
                raise RuntimeError("Invalid actor rigid-body registration")
            for index, shape in enumerate(bodies[0].get_collision_shapes() if bodies else []):
                name = f"scene::{key}::shape_{index}"
                shapes[name] = (entity, shape)
                pose = (entity.pose * shape.get_local_pose()).to_transformation_matrix().astype(np.float64)
                if not np.isfinite(pose).all():
                    raise RuntimeError("Nonfinite scene collision pose")
                poses[name] = pose
        return entities, shapes, poses

    def _measure_robots(self) -> tuple[np.ndarray, np.ndarray]:
        measured = []
        for index, solver in enumerate(self.solvers):
            if _mapping(solver) != self._mapping[index]:
                raise RuntimeError("Robot joint/link mapping changed")
            if not np.allclose(_pose(solver.robot), self._base_poses[index], rtol=0, atol=1e-6):
                raise RuntimeError("Fixed robot base moved")
            planner_names, _, active_names, *_ = self._mapping[index]
            q = _single(solver.robot.get_qpos(), len(active_names))
            measured.append(q[[active_names.index(name) for name in planner_names]])
        return tuple(measured)

    def _assert_scene_structure(self, entities: dict[str, Any], shapes: dict[str, tuple[Any, Any]],
                                poses: dict[str, np.ndarray]) -> None:
        if set(entities) != set(self._entities) or any(entities[n] != self._entities[n] for n in entities):
            raise RuntimeError("Scene entity set or identities changed")
        if set(shapes) != set(self._shape_refs):
            raise RuntimeError("Scene collision shape set changed")
        for name, (entity, shape) in shapes.items():
            if (entity, shape) != self._shape_refs[name] or _signature(shape) != self._shape_signatures[name]:
                raise RuntimeError(f"Scene collision geometry changed: {name}")
        if self._mount_pose is not None and not np.allclose(
                poses["scene::table::shape_0"], self._mount_pose, rtol=0, atol=1e-6):
            raise RuntimeError("Fixed mounting table moved")

    def refresh(self) -> dict[str, Any]:
        """Read current scene poses/fingers; reuse geometry and rebuild broad phase."""
        self._fresh = False
        entities, shapes, poses = self._capture_scene()
        self._assert_scene_structure(entities, shapes, poses)
        measured = self._measure_robots()
        objects: list[Any] = []
        planes: list[dict[str, Any]] = []
        world_poses: dict[str, list[float]] = {}
        for name, (entity, shape) in shapes.items():
            shape_pose = entity.pose * shape.get_local_pose()
            correction = sapien.Pose(q=[2**-0.5, 0, 2**-0.5, 0]) if isinstance(
                shape, (physx.PhysxCollisionShapeCapsule, physx.PhysxCollisionShapeCylinder)) else sapien.Pose()
            pose = shape_pose * correction
            p, q = np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)
            world_poses[name] = np.r_[p, q].tolist()
            if isinstance(shape, physx.PhysxCollisionShapePlane):
                normal = poses[name][:3, 0]
                planes.append({"name": name, "normal": normal.tolist(),
                               "offset": float(normal @ shape_pose.p), "solid_side": "normal_dot_x_le_offset"})
            else:
                objects.append((name, fcl.CollisionObject(self._geometry[name], p, q)))
        by_name = dict(objects)
        if set(by_name) != set(self._normal_names):
            raise RuntimeError("Finite native collision object set changed")
        for model, q in zip(self.models, measured):
            model.set_qpos(q, True)
        world = PlanningWorld(list(self.models), ["left", "right"],
                              [by_name[name] for name in self._normal_names], list(self._normal_names))
        self.native.world = world
        self.native.full_qpos = tuple(q.copy() for q in measured)
        self._snapshot_poses = {name: pose.copy() for name, pose in poses.items()}
        self._measured_qpos = tuple(q.copy() for q in measured)
        self.refresh_count += 1
        self.coverage.update(planes=planes, refresh_count=self.refresh_count,
                             target_contact_allowed=self.target_contact_allowed,
                             last_measured_fingers=[q[[j for j in range(9) if j not in arm]].tolist()
                                                    for q, arm in zip(measured, self.native.arm_indices)])
        for entry in self.coverage["shapes"]:
            entry["world_pose_wxyz"] = world_poses[entry["name"]]
        self._fresh = True
        return self.coverage

    def assert_scene_unchanged(self) -> None:
        if not self._fresh:
            raise RuntimeError("A successful refresh is required")
        entities, shapes, poses = self._capture_scene()
        self._assert_scene_structure(entities, shapes, poses)
        if any(not np.allclose(poses[n], self._snapshot_poses[n], rtol=0, atol=1e-6) for n in poses):
            raise RuntimeError("Scene moved since refresh; refresh before checking")

    def assert_robot_geometry_unchanged(self) -> None:
        measured = self._measure_robots()
        for q, previous, arm in zip(measured, self._measured_qpos, self.native.arm_indices):
            fingers = [j for j in range(9) if j not in arm]
            if not np.allclose(q[fingers], previous[fingers], rtol=0, atol=1e-6):
                raise RuntimeError("Physical fingers moved since refresh")

    def set_target_contact_allowed(self, allowed: bool) -> None:
        if type(allowed) is not bool:
            raise ValueError("Target contact permission must be bool")
        self.target_contact_allowed = allowed
        self.coverage["target_contact_allowed"] = allowed

    def _allowed(self, record: dict[str, Any]) -> bool:
        pairs = ((record.get("object_name1"), record.get("link_name1"), record.get("object_name2")),
                 (record.get("object_name2"), record.get("link_name2"), record.get("object_name1")))
        for robot, link, other in pairs:
            if (robot, link, other) in self.allowed_mount_contacts:
                return True
            if self.target_contact_allowed and robot == self.active_robot and link in FINGER_LINKS and other in self._target_names:
                return True
        return False

    def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]:
        """Return unallowed collisions in the refreshed static scene snapshot."""
        self.assert_scene_unchanged()
        self.assert_robot_geometry_unchanged()
        blocked: list[dict[str, Any]] = []
        self.last_allowed_contacts = []
        for record in self.native.check(left7, right7):
            if self._allowed(record):
                self.last_allowed_contacts.append(record)
                self.allowed_contact_observations += 1
            else:
                blocked.append(record)
        # These violations are never passed through any contact allowlist.
        blocked.extend(check_plane_halfspaces(list(self.models), self.coverage["planes"]))
        return blocked
