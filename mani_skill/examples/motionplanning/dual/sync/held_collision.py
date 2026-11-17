"""Predict rigidly held geometry, while checking measured poses separately.

This module never attaches/welds a physical actor, changes collision masks, or
steps an environment. A candidate check replaces the target's measured shapes
in a private collision world with their TCP-relative prediction. A measured
check uses the real actor instead. Refresh before each physical control tick.
Neither mode predicts other actors or provides continuous collision checking.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
from mplib.pymp import PlanningWorld
from mplib.pymp.collision_detection import fcl
from transforms3d.quaternions import mat2quat, quat2mat

from .combined_collision import NativeDualCollisionChecker, _vector
from .grasp_collision import FINGER_LINKS, LiveGraspCollisionChecker, _pose
from .scene_collision import check_plane_halfspaces


def _transform(pose7: Any) -> np.ndarray:
    pose = np.asarray(pose7, dtype=np.float64).reshape(7)
    if not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-6):
        raise ValueError("Invalid position/wxyz quaternion")
    result = np.eye(4)
    result[:3, :3] = quat2mat(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def _entity_transform(entity: Any) -> np.ndarray:
    return _transform(np.r_[entity.pose.p, entity.pose.q])


def _object_transform(obj: Any) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(obj.get_rotation()).reshape(3, 3)
    result[:3, 3] = np.asarray(obj.get_translation()).reshape(3)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite collision object transform")
    return result


def _object_at(geometry: Any, transform: np.ndarray) -> Any:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Invalid collision transform")
    return fcl.CollisionObject(geometry, transform[:3, 3], mat2quat(transform[:3, :3]))


def _world_tcp(model: Any, qpos: np.ndarray, tcp_link_name: str) -> np.ndarray:
    """Pinocchio FK is base-local even after ArticulatedModel.set_base_pose."""
    pinocchio = model.get_pinocchio_model()
    names = pinocchio.get_link_names()
    if tcp_link_name not in names:
        raise ValueError(f"Missing TCP link: {tcp_link_name}")
    pinocchio.compute_forward_kinematics(qpos)
    return _transform(model.get_base_pose()) @ _transform(pinocchio.get_link_pose(names.index(tcp_link_name)))


def _pair_result(first: Any, second: Any) -> tuple[bool, list[dict[str, Any]]]:
    # Plane/convex queries crash MPLib 0.1.1; all planes use half-space guards.
    if any(isinstance(o.get_collision_geometry(), fcl.Plane) for o in (first, second)):
        raise RuntimeError("Plane must use the half-space backend")
    result = fcl.collide(first, second, fcl.CollisionRequest(num_max_contacts=32, enable_contact=True))
    contacts = []
    for contact in result.get_contacts():
        depth = float(contact.penetration_depth)
        normal = np.asarray(contact.normal, dtype=float).reshape(3)
        position = np.asarray(contact.pos, dtype=float).reshape(3)
        if not np.isfinite(np.r_[depth, normal, position]).all() or depth < -1e-6:
            raise RuntimeError("Invalid FCL contact data")
        contacts.append({"depth": max(0.0, depth), "normal": normal.tolist(), "position": position.tolist()})
    return result.is_collision(), contacts


class AttachedObjectCollisionChecker:
    """Fixed TCP/object relation for prediction; no physical attachment."""

    @classmethod
    def from_live(cls, live_checker: LiveGraspCollisionChecker, target_actor: Any,
                  active_robot: str = "left", tcp_link_name: str = "panda_hand_tcp", *,
                  support_band_m: float = 0.003, support_penetration_m: float = 0.003,
                  support_depth_slop_m: float = 0.0002) -> AttachedObjectCollisionChecker:
        if active_robot not in ("left", "right") or active_robot != live_checker.active_robot:
            raise ValueError("Held object must use the live checker's active robot")
        if not all(np.isfinite(v) and 0 < v <= .005 for v in
                   (support_band_m, support_penetration_m, support_depth_slop_m)):
            raise ValueError("Support tolerances must be positive and at most 5mm")
        live_checker.refresh()
        entity = live_checker._entities[live_checker.target_actor_name]
        supplied = target_actor._objs[0] if hasattr(target_actor, "_objs") and len(target_actor._objs) == 1 else target_actor
        if isinstance(supplied, str):
            supplied = live_checker._entities.get(supplied)
        if supplied != entity:
            raise ValueError("Held actor differs from the live grasp target")
        self = cls()
        self.live = live_checker
        self.env = live_checker.env
        self.active_robot = active_robot
        self._active_index = ("left", "right").index(active_robot)
        self.tcp_link_name = tcp_link_name
        self._entity = entity
        self._target_names = frozenset(live_checker._target_names)
        self._normal_names = tuple(live_checker.native.normal_object_names)
        objects = dict(zip(live_checker.world.get_normal_object_names(), live_checker.world.get_normal_objects()))
        if set(objects) != set(self._normal_names):
            raise RuntimeError("Live collision world has inconsistent object names")
        self.native = NativeDualCollisionChecker.from_solvers(
            *live_checker.solvers, normal_objects=[objects[n] for n in self._normal_names],
            normal_object_names=self._normal_names)
        self._initial_object = _entity_transform(entity)
        measured = live_checker._measured_qpos
        tcp = _world_tcp(self.models[self._active_index], measured[self._active_index], tcp_link_name)
        measured_tcp = self._assert_tcp_matches_measurement(tcp)
        self._attachment = np.linalg.inv(measured_tcp) @ self._initial_object
        self._shape_local = {n: np.linalg.inv(self._initial_object) @ _object_transform(objects[n])
                             for n in self._target_names}
        self.support_band_m = float(support_band_m)
        self.support_penetration_m = float(support_penetration_m)
        self.support_depth_slop_m = float(support_depth_slop_m)
        self._support_pairs: dict[tuple[str, str], dict[str, Any]] = {}
        self.support_released = False
        self._register_initial_support(objects)
        self.scope = "rigid_held_object_prediction_and_separate_measured_geometry_no_physical_attachment"
        self.refresh_count = 0
        self.coverage: dict[str, Any] = {}
        self.last_allowed_contacts: list[dict[str, Any]] = []
        self.attachment_info = {"target_actor": live_checker.target_actor_name, "tcp_link": tcp_link_name,
                                "active_robot": active_robot, "base_pose": self.models[self._active_index].get_base_pose().tolist()}
        self.refresh()
        return self

    @property
    def models(self) -> tuple[Any, ...]:
        return self.native.models

    @property
    def world(self) -> Any:
        return self.native.world

    @property
    def attachment_T_tcp_object(self) -> np.ndarray:
        return self._attachment.copy()

    def _assert_tcp_matches_measurement(self, tcp: np.ndarray) -> np.ndarray:
        agent = self.live.solvers[self._active_index].env_agent
        if not hasattr(agent, "tcp"):
            raise ValueError("Measured TCP link is required to validate FK/base composition")
        actual = _transform(_pose(agent.tcp))
        if not np.allclose(tcp, actual, rtol=0, atol=2e-5):
            raise RuntimeError("Native world TCP differs from measured SAPIEN TCP")
        return actual

    def _register_initial_support(self, objects: dict[str, Any]) -> None:
        table_name = "scene::table::shape_0"
        if table_name not in objects:
            return
        table = objects[table_name]
        table_geometry = table.get_collision_geometry()
        table_pose = _object_transform(table)
        horizontal_box = isinstance(table_geometry, fcl.Box) and np.allclose(
            table_pose[:3, 2], [0, 0, 1], atol=1e-6, rtol=0)
        for name in self._target_names:
            collision, contacts = _pair_result(objects[name], table)
            if not collision:
                continue
            if not horizontal_box or not contacts:
                raise RuntimeError("Initial held/table collision is not a known horizontal top support")
            top_z = float(table_pose[2, 3] + np.asarray(table_geometry.side).reshape(3)[2] / 2)
            depth = max(c["depth"] for c in contacts)
            if depth > self.support_penetration_m or any(
                    abs(c["normal"][2]) < .99 or abs(c["position"][2] - top_z) > self.support_penetration_m
                    for c in contacts):
                raise RuntimeError(f"Initial support penetration/top-face contact exceeds allowance: {name}, {depth}")
            self._support_pairs[(name, table_name)] = {"initial_depth": depth, "contacts": contacts}

    def refresh(self) -> dict[str, Any]:
        """Refresh real actors/fingers, preserving the original attachment relation."""
        self._fresh = False
        self.live.refresh()
        objects = dict(zip(self.live.world.get_normal_object_names(), self.live.world.get_normal_objects()))
        if set(objects) != set(self._normal_names):
            raise RuntimeError("Held collision world object set changed")
        actual_object = _entity_transform(self._entity)
        for name in self._target_names:
            if not np.allclose(actual_object @ self._shape_local[name], _object_transform(objects[name]), rtol=0, atol=2e-6):
                raise RuntimeError("Rigid held object's shape-local transform changed")
        measured = self.live._measured_qpos
        for model, qpos in zip(self.models, measured):
            model.set_qpos(qpos, True)
        tcp = _world_tcp(self.models[self._active_index], measured[self._active_index], self.tcp_link_name)
        self._assert_tcp_matches_measurement(tcp)
        self.native.full_qpos = tuple(q.copy() for q in measured)
        self.native.world = PlanningWorld(list(self.models), ["left", "right"],
                                          [objects[n] for n in self._normal_names], list(self._normal_names))
        self._measured_objects = objects
        self._measured_object_pose = actual_object
        self._measured_qpos = tuple(q.copy() for q in measured)
        self._update_support_release(objects, actual_object)
        self.refresh_count += 1
        self.coverage = copy.deepcopy(self.live.coverage)
        self.coverage.update(scope=self.scope, attached_object=True, physical_attachment=False,
                             rigid_prediction=True, measured_check_uses_real_object=True,
                             refresh_count=self.refresh_count, continuous_collision_detection=False,
                             attachment_T_tcp_object=self._attachment.tolist(),
                             support_released=self.support_released, support_band_m=self.support_band_m,
                             support_penetration_m=self.support_penetration_m,
                             initial_support_pairs=[{"target": a, "table": b, **info}
                                                    for (a, b), info in self._support_pairs.items()])
        self._fresh = True
        return self.coverage

    def _update_support_release(self, objects: dict[str, Any], actual_object: np.ndarray) -> None:
        """Measured-only support transition; subclasses may recognize whole objects.

        The base keeps its original exact-pair semantics. Candidate collision
        queries never call this hook or advance the permanent release state.
        """
        if self._support_pairs and not self.support_released:
            displacement_z = actual_object[2, 3] - self._initial_object[2, 3]
            still_touching = any(_pair_result(objects[a], objects[b])[0] for a, b in self._support_pairs)
            if displacement_z > self.support_band_m or not still_touching:
                self.support_released = True

    def assert_scene_unchanged(self) -> None:
        if not self._fresh:
            raise RuntimeError("A successful held-object refresh is required")
        self.live.assert_scene_unchanged()

    def assert_robot_geometry_unchanged(self) -> None:
        self.live.assert_robot_geometry_unchanged()

    def _allowed_robot_contact(self, record: dict[str, Any]) -> bool:
        for robot_key, link_key, other_key in (("object_name1", "link_name1", "object_name2"),
                                                ("object_name2", "link_name2", "object_name1")):
            robot, link, other = record.get(robot_key), record.get(link_key), record.get(other_key)
            if (robot, link, other) in self.live.allowed_mount_contacts:
                return True
            if robot == self.active_robot and link in FINGER_LINKS and other in self._target_names:
                return True
        return False

    def _support_allowed(self, target_name: str, other_name: str, object_pose: np.ndarray,
                         contacts: list[dict[str, Any]]) -> bool:
        baseline = self._support_pairs.get((target_name, other_name))
        if baseline is None or self.support_released or not contacts:
            return False
        displacement = object_pose[:3, 3] - self._initial_object[:3, 3]
        rotation_delta = object_pose[:3, :3] @ self._initial_object[:3, :3].T
        angle = float(np.arccos(np.clip((np.trace(rotation_delta) - 1) / 2, -1, 1)))
        return bool(-1e-6 <= displacement[2] <= self.support_band_m
                    and np.linalg.norm(displacement[:2]) <= .001 and angle <= .01
                    and max(c["depth"] for c in contacts) <= baseline["initial_depth"] + self.support_depth_slop_m)

    def _evaluate(self, joints: tuple[np.ndarray, np.ndarray], object_pose: np.ndarray, mode: str) -> list[dict[str, Any]]:
        self.assert_scene_unchanged()
        self.assert_robot_geometry_unchanged()
        objects = dict(self._measured_objects)
        for name in self._target_names:
            # Exact replacement: never retain a second target at its old pose.
            objects[name] = _object_at(self.live._geometry[name], object_pose @ self._shape_local[name])
        self.native.world = PlanningWorld(list(self.models), ["left", "right"],
                                          [objects[n] for n in self._normal_names], list(self._normal_names))
        blocked: list[dict[str, Any]] = []
        self.last_allowed_contacts = []
        for record in self.native.check(*joints):
            if self._allowed_robot_contact(record):
                self.last_allowed_contacts.append(record)
            else:
                blocked.append(record)
        # PlanningWorld does not check normal objects against normal objects.
        for name in self._target_names:
            held = objects[name]
            for other_name in self._normal_names:
                if other_name in self._target_names:
                    continue
                collision, contacts = _pair_result(held, objects[other_name])
                if not collision:
                    continue
                record = {"collision_type": "held_object_environment", "object_name1": name, "link_name1": name,
                          "object_name2": other_name, "link_name2": other_name, "contacts": contacts}
                if self._support_allowed(name, other_name, object_pose, contacts):
                    self.last_allowed_contacts.append({**record, "allowance": "initial_upward_support_separation_only"})
                else:
                    blocked.append(record)
            # Explicit payload/robot checks retain all palm and other-arm pairs.
            for robot, model in zip(("left", "right"), self.models):
                collision_model = model.get_fcl_model()
                links, link_objects = collision_model.get_collision_link_names(), collision_model.get_collision_objects()
                if len(links) != len(link_objects) or not links:
                    raise RuntimeError("Invalid robot collision geometry")
                for link, robot_object in zip(links, link_objects):
                    collision, _ = _pair_result(held, robot_object)
                    if collision:
                        record = {"collision_type": "held_object_robot", "object_name1": robot, "link_name1": str(link),
                                  "object_name2": name, "link_name2": name}
                        if self._allowed_robot_contact(record):
                            self.last_allowed_contacts.append(record)
                        else:
                            blocked.append(record)
        blocked.extend(check_plane_halfspaces(list(self.models), self.live.coverage["planes"]))
        held_model = SimpleNamespace(get_fcl_model=lambda: SimpleNamespace(
            get_collision_objects=lambda: [objects[n] for n in sorted(self._target_names)],
            get_collision_link_names=lambda: sorted(self._target_names)))
        for record in check_plane_halfspaces([held_model], self.live.coverage["planes"]):
            blocked.append({**record, "held_object": self.live.target_actor_name})
        self.coverage.update(last_check_mode=mode, last_object_T_world=object_pose.tolist(),
                             last_allowed_contact_count=len(self.last_allowed_contacts))
        return blocked

    def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]:
        """Check candidate arms with rigid TCP-relative payload prediction."""
        joints = (_vector(left7, 7), _vector(right7, 7))
        full_states = []
        for q, initial, arm, limits in zip(joints, self.native.full_qpos, self.native.arm_indices, self.native.joint_limits):
            if np.any(q < limits[:, 0] - 1e-6) or np.any(q > limits[:, 1] + 1e-6):
                raise ValueError("Candidate arm state exceeds joint limits")
            full = initial.copy()
            full[arm] = q
            full_states.append(full)
        tcp = _world_tcp(self.models[self._active_index], full_states[self._active_index], self.tcp_link_name)
        return self._evaluate(joints, tcp @ self._attachment, "candidate_rigid_prediction")

    def measured_check(self) -> list[dict[str, Any]]:
        """Check measured arms and actual payload pose; no rigid pose substitution."""
        measured = self.live._measure_robots()
        if any(not np.allclose(q, previous, rtol=0, atol=1e-6) for q, previous in zip(measured, self._measured_qpos)):
            raise RuntimeError("Measured arms changed since held refresh")
        joints = tuple(q[arm] for q, arm in zip(measured, self.native.arm_indices))
        return self._evaluate(joints, self._measured_object_pose, "measured_actual_geometry")
