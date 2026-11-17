"""Two measured or rigidly predicted payloads in one private dual-Panda world.

Robot identities are always physical (left, right). ``attach`` captures two
TCP/object relations for collision prediction only: it never welds, teleports,
steps, or changes a physical collision mask. The caller must confirm both real
grasps before attach. Refresh after every physical tick. Other actors are a
measured snapshot; finite joint sampling is not continuous collision checking.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
from mplib.pymp import PlanningWorld

from .combined_collision import NativeDualCollisionChecker, _array, _vector
from .grasp_collision import FINGER_LINKS, LiveGraspCollisionChecker
from .held_collision import _entity_transform, _object_at, _pair_result, _world_tcp
from .scene_collision import check_plane_halfspaces
from .table_support_collision import NumericalTableSupportCollisionChecker


class DualPayloadCollisionChecker:
    """Canonical two-robot/two-payload collision checks, without physics writes."""

    @classmethod
    def from_solvers(cls, env: Any, left_solver: Any, right_solver: Any, *,
                     targets: tuple[Any, Any]) -> DualPayloadCollisionChecker:
        raw = getattr(env, "unwrapped", env)
        if (not isinstance(targets, (tuple, list)) or len(targets) != 2
                or any(target is None for target in targets)):
            raise ValueError("Two actual target actors in physical left/right order are required")
        if tuple(raw.agent.agents) != (left_solver.env_agent, right_solver.env_agent):
            raise ValueError("Solvers must follow the environment's physical left/right identity")
        self = cls()
        self.env, self.solvers, self.targets = raw, (left_solver, right_solver), tuple(targets)
        self.lives = tuple(LiveGraspCollisionChecker.from_solvers(
            raw, left_solver, right_solver, target_actor=target, active_robot=robot)
            for target, robot in zip(self.targets, ("left", "right")))
        self.target_actor_names = tuple(live.target_actor_name for live in self.lives)
        self._target_names = tuple(frozenset(live._target_names) for live in self.lives)
        if self._target_names[0] & self._target_names[1]:
            raise ValueError("Payloads must be two distinct actors with disjoint shapes")
        self._payload_names = self._target_names[0] | self._target_names[1]
        self._normal_names = tuple(self.lives[0]._normal_names)
        objects = dict(zip(self.lives[0].world.get_normal_object_names(), self.lives[0].world.get_normal_objects()))
        self.native = NativeDualCollisionChecker.from_solvers(
            left_solver, right_solver, normal_objects=[objects[n] for n in self._normal_names],
            normal_object_names=self._normal_names)
        self.scope = "two_measured_or_rigidly_predicted_payloads_one_private_dual_robot_scene_world"
        self.attached, self._supports = False, ()
        self.target_contacts_allowed = (False, False)
        self.refresh_count, self.last_allowed_contacts = 0, []
        self.coverage: dict[str, Any] = {}
        self._fresh = False
        self.measured_attachment_prediction = False
        self.attachment_geometry_epoch = 0
        self._prediction_relations = None
        self._prediction_control_step = None
        self.refresh()
        return self

    @property
    def models(self) -> tuple[Any, ...]:
        return self.native.models

    @property
    def world(self) -> Any:
        return self.native.world

    @property
    def attachments(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.attached:
            raise ValueError("Both payload relations must be captured by attach first")
        return tuple(support.attachment_T_tcp_object for support in self._supports)

    def set_measured_attachment_prediction(self, enabled: bool) -> None:
        """Select geometry refresh before grasp; never reset a grasp reference."""
        if type(enabled) is not bool or self.attached:
            raise ValueError("Select the prediction strategy before attaching")
        self.measured_attachment_prediction = enabled
        self._prediction_relations = None
        self._prediction_control_step = None
        self.coverage.update(self._prediction_metadata())

    def _control_step(self) -> int:
        # The actual environment counter may be a CUDA tensor. Reuse the
        # existing finite tensor-to-CPU conversion, without initializing CUDA.
        value = _array(self.env.elapsed_steps).reshape(-1)
        if value.size != 1 or value[0] < 0 or value[0] != int(value[0]):
            raise RuntimeError("One physical environment step counter is required")
        return int(value[0])

    def _prediction_metadata(self) -> dict[str, Any]:
        return {
            "attachment_prediction_mode": ("measured_control_step_snapshot"
                if self.measured_attachment_prediction else "fixed_attachment"),
            "attachment_geometry_epoch": self.attachment_geometry_epoch,
            "prediction_control_step": self._prediction_control_step,
            "prediction_T_tcp_objects": ([relation.tolist() for relation in self._prediction_relations]
                if self._prediction_relations is not None else None),
            "prediction_snapshot_valid": bool(self._fresh and self.attached
                and self.measured_attachment_prediction and self._prediction_relations is not None),
            "quality_reference_reanchored": False,
        }

    def _assert_prediction_current(self) -> None:
        if (not self._fresh or not self.attached or not self.measured_attachment_prediction
                or self._prediction_relations is None):
            raise RuntimeError("A current measured attached geometry snapshot is required")
        if self._control_step() != self._prediction_control_step:
            raise RuntimeError("Physical control step changed after attachment geometry refresh")
        measured = self.lives[0]._measure_robots()
        if any(not np.allclose(q, previous, rtol=0, atol=1e-6)
               for q, previous in zip(measured, self._measured_qpos)):
            raise RuntimeError("Robot state changed since dual-payload refresh")

    def validate_measured_command(self, left7: np.ndarray, right7: np.ndarray, *,
                                  grasping: tuple[bool, bool],
                                  max_joint_step_rad: float) -> dict[str, Any]:
        """Recheck this command under the latest measured attachment snapshot.

        The executor already refreshes after each physical control tick. Reuse
        only that snapshot, while every check retains the existing actual pose,
        finger, base and shape guards. Never reuse a phase-path certificate as
        proof for a changed relation. Sampling is not continuous collision proof.
        """
        self._assert_prediction_current()
        if (not isinstance(grasping, (tuple, list)) or len(grasping) != 2
                or any(type(value) is not bool for value in grasping)):
            raise ValueError("Both actual grasp observations are required")
        receipt = {"geometry_epoch": self.attachment_geometry_epoch,
                   "control_step": self._prediction_control_step,
                   "target_actor_names": list(self.target_actor_names),
                   "actual_grasping": list(grasping), "sampled_configurations": 0,
                   "prediction_mode": self.coverage["attachment_prediction_mode"],
                   "prediction_T_tcp_objects": [relation.tolist() for relation in self._prediction_relations],
                   "quality_reference_reanchored": False,
                   "support_release_reopened": False}
        if not all(grasping):
            return {**receipt, "status": "rejected", "reason": "attachment_grasp_unconfirmed"}
        if (type(max_joint_step_rad) not in (int, float)
                or not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0):
            raise ValueError("Use the existing positive joint sampling step")
        goals = (_vector(left7, 7), _vector(right7, 7))
        starts = tuple(q[index] for q, index in zip(self._measured_qpos, self.native.arm_indices))
        intervals = max(1, int(np.ceil(max(float(np.max(np.abs(b-a)))
            for a, b in zip(starts, goals)) / max_joint_step_rad)))
        for fraction in np.linspace(0., 1., intervals+1):
            collisions = self.check(*(a + fraction*(b-a) for a, b in zip(starts, goals)))
            receipt["sampled_configurations"] += 1
            if collisions:
                return {**receipt, "status": "rejected", "reason": "updated_candidate_collision",
                        "fraction": float(fraction), "collisions": collisions}
        return {**receipt, "status": "passed"}

    def set_target_contacts_allowed(self, allowed: tuple[bool, bool]) -> None:
        if (not isinstance(allowed, (tuple, list)) or len(allowed) != 2
                or any(type(value) is not bool for value in allowed)):
            raise ValueError("Exactly two actual bool contact permissions are required")
        self.target_contacts_allowed = tuple(allowed)
        for live, value in zip(self.lives, allowed):
            live.set_target_contact_allowed(value)
        self.coverage["target_contacts_allowed"] = list(self.target_contacts_allowed)

    def attach(self) -> None:
        """Capture both real TCP/object relations; real grasp confirmation is external."""
        if self.attached:
            raise ValueError("Payload relations are already captured")
        supports = tuple(NumericalTableSupportCollisionChecker.from_live(live, target, active_robot=robot)
                         for live, target, robot in zip(self.lives, self.targets, ("left", "right")))
        self._supports, self.attached = supports, True
        try:
            self.refresh()
        except Exception:
            self.attached, self._fresh = False, False
            raise

    def detach(self) -> None:
        """Return to measured live geometry; no physical attachment ever existed."""
        self.attached = False
        self.refresh()

    def refresh(self) -> dict[str, Any]:
        self._fresh = False
        self._prediction_relations = None
        self._prediction_control_step = None
        self.coverage.update(self._prediction_metadata())
        for checker in self._supports if self.attached else self.lives:
            checker.refresh()  # Support release is updated ONLY from real measured poses.
        primary = self.lives[0]
        objects = dict(zip(primary.world.get_normal_object_names(), primary.world.get_normal_objects()))
        if set(objects) != set(self._normal_names):
            raise RuntimeError("The finite scene object set changed")
        for live in self.lives:
            if tuple(live._normal_names) != self._normal_names:
                raise RuntimeError("The two target views have different scene registrations")
            live.assert_scene_unchanged()
            live.assert_robot_geometry_unchanged()
            if any(not np.allclose(q, previous, rtol=0, atol=1e-6)
                   for q, previous in zip(live._measured_qpos, primary._measured_qpos)):
                raise RuntimeError("The two target views measured different robot states")
        self._measured_qpos = tuple(q.copy() for q in primary._measured_qpos)
        self._measured_objects = objects
        self._measured_payload_poses = tuple(_entity_transform(live._entities[name])
                                             for live, name in zip(self.lives, self.target_actor_names))
        for model, full in zip(self.models, self._measured_qpos):
            model.set_qpos(full, True)
        self.native.full_qpos = tuple(q.copy() for q in self._measured_qpos)
        self.native.world = PlanningWorld(list(self.models), ["left", "right"],
                                          [objects[n] for n in self._normal_names], list(self._normal_names))
        self.refresh_count += 1
        if self.attached and self.measured_attachment_prediction:
            current_step = self._control_step()
            self._prediction_relations = tuple(
                np.linalg.inv(support._assert_tcp_matches_measurement(
                    _world_tcp(model, full, support.tcp_link_name))) @ pose
                for model, full, support, pose in zip(self.models, self._measured_qpos,
                    self._supports, self._measured_payload_poses))
            self._prediction_control_step = current_step
            self.attachment_geometry_epoch += 1
        self.coverage = copy.deepcopy(primary.coverage)
        for key in ("active_robot", "target_actor", "target_contact_allowed", "attached_object"):
            self.coverage.pop(key, None)  # A single-target view is not the dual contract.
        self.coverage.update(
            scope=self.scope, attached_objects=2 if self.attached else 0, physical_attachment=False,
            target_actor_names=list(self.target_actor_names), target_shape_names=[sorted(n) for n in self._target_names],
            target_contacts_allowed=list(self.target_contacts_allowed), refresh_count=self.refresh_count,
            canonical_robot_order=["left", "right"], payloads_share_one_world=True,
            own_fingers_only=True, measured_check_uses_real_objects=True,
            rigid_prediction=self.attached, continuous_collision_detection=False,
            dynamic_prediction=False, normal_object_names=list(self._normal_names),
            attachment_T_tcp_objects=[a.tolist() for a in self.attachments] if self.attached else None,
            support_checks=[copy.deepcopy(support.coverage) for support in self._supports] if self.attached else [],
            last_check_mode=None,
        )
        self._fresh = True
        self.coverage.update(self._prediction_metadata())
        return self.coverage

    def assert_scene_unchanged(self) -> None:
        if not self._fresh:
            raise RuntimeError("A successful dual-payload refresh is required")
        for live in self.lives:
            live.assert_scene_unchanged()

    def assert_robot_geometry_unchanged(self) -> None:
        for live in self.lives:
            live.assert_robot_geometry_unchanged()

    def _allowed(self, record: dict[str, Any]) -> bool:
        mounts = self.lives[0].allowed_mount_contacts
        for robot_key, link_key, other_key in (("object_name1", "link_name1", "object_name2"),
                                                ("object_name2", "link_name2", "object_name1")):
            robot, link, other = record.get(robot_key), record.get(link_key), record.get(other_key)
            if (robot, link, other) in mounts:
                return True
            for index, physical in enumerate(("left", "right")):
                if (robot == physical and link in FINGER_LINKS and other in self._target_names[index]
                        and (self.attached or self.target_contacts_allowed[index])):
                    return True
        return False

    @staticmethod
    def _identity(record: dict[str, Any]) -> tuple[Any, ...]:
        if record.get("collision_type") == "conservative_plane_halfspace":
            return ("plane", record.get("model_index"), record.get("link_name"), record.get("plane_name"),
                    record.get("plane_index"), record.get("payload_index"), record.get("reason"))
        return tuple(sorted(((record.get("object_name1"), record.get("link_name1")),
                             (record.get("object_name2"), record.get("link_name2")))))

    def _evaluate(self, joints: tuple[np.ndarray, np.ndarray], poses: tuple[np.ndarray, np.ndarray] | None,
                  mode: str) -> list[dict[str, Any]]:
        self.assert_scene_unchanged()
        self.assert_robot_geometry_unchanged()
        objects = dict(self._measured_objects)
        if poses is not None:
            if not self.attached:
                raise RuntimeError("Rigid payload replacement needs both captured relations")
            for support, pose in zip(self._supports, poses):
                for name in support._target_names:
                    # Replace BOTH target groups before constructing the ONE world.
                    objects[name] = _object_at(support.live._geometry[name], pose @ support._shape_local[name])
        self.native.world = PlanningWorld(list(self.models), ["left", "right"],
                                          [objects[n] for n in self._normal_names], list(self._normal_names))
        blocked, seen, allowed_seen = [], set(), set()
        self.last_allowed_contacts = []

        def record(value: dict[str, Any], *, allowed: bool | None = None) -> None:
            permitted = self._allowed(value) if allowed is None else allowed
            identity = self._identity(value)
            if permitted:
                if identity not in allowed_seen:
                    allowed_seen.add(identity); self.last_allowed_contacts.append(value)
            elif identity not in seen:
                seen.add(identity); blocked.append(value)

        # This query first sets BOTH robot states and checks self/interarm/world.
        for value in self.native.check(*joints):
            record(value)
        if poses is not None:
            # Normal-object/normal-object pairs are absent from PlanningWorld.
            for index, names in enumerate(self._target_names):
                support = self._supports[index]
                for name in sorted(names):
                    payload = objects[name]
                    for other in self._normal_names:
                        if other in self._payload_names:
                            continue
                        collided, contacts = _pair_result(payload, objects[other])
                        if collided:
                            permitted = support._support_allowed(name, other, poses[index], contacts)
                            value = {"collision_type": "payload_environment", "payload_index": index,
                                     "object_name1": name, "link_name1": name, "object_name2": other,
                                     "link_name2": other, "contacts": contacts}
                            if permitted: value["allowance"] = "initial_upward_support_separation_only"
                            record(value, allowed=permitted)
                    for robot, model in zip(("left", "right"), self.models):
                        collision_model = model.get_fcl_model()
                        links, link_objects = collision_model.get_collision_link_names(), collision_model.get_collision_objects()
                        if not links or len(links) != len(link_objects):
                            raise RuntimeError("Incomplete private robot collision geometry")
                        for link, robot_object in zip(links, link_objects):
                            collided, contacts = _pair_result(payload, robot_object)
                            if collided:
                                record({"collision_type": "payload_robot", "payload_index": index,
                                        "object_name1": robot, "link_name1": str(link),
                                        "object_name2": name, "link_name2": name, "contacts": contacts})
            for left_name in sorted(self._target_names[0]):
                for right_name in sorted(self._target_names[1]):
                    collided, contacts = _pair_result(objects[left_name], objects[right_name])
                    if collided:
                        record({"collision_type": "payload_payload", "payload_indices": [0, 1],
                                "object_name1": left_name, "link_name1": left_name,
                                "object_name2": right_name, "link_name2": right_name,
                                "contacts": contacts}, allowed=False)
        for value in check_plane_halfspaces(list(self.models), self.lives[0].coverage["planes"]):
            record(value, allowed=False)
        if poses is not None:
            for index, names in enumerate(self._target_names):
                ordered = sorted(names)
                payload_model = NS(get_fcl_model=lambda ordered=ordered: NS(
                    get_collision_objects=lambda: [objects[n] for n in ordered],
                    get_collision_link_names=lambda: ordered))
                for value in check_plane_halfspaces([payload_model], self.lives[0].coverage["planes"]):
                    record({**value, "payload_index": index, "payload_actor": self.target_actor_names[index]}, allowed=False)
        self.coverage.update(last_check_mode=mode, last_allowed_contact_count=len(self.last_allowed_contacts),
                             last_payload_T_world=[pose.tolist() for pose in poses] if poses is not None else None)
        return blocked

    def check(self, left7: np.ndarray, right7: np.ndarray) -> list[dict[str, Any]]:
        if self.attached and self.measured_attachment_prediction:
            self._assert_prediction_current()
        joints = (_vector(left7, 7), _vector(right7, 7))
        full_states = []
        for q, full, arm, limits in zip(joints, self._measured_qpos, self.native.arm_indices, self.native.joint_limits):
            if np.any(q < limits[:, 0] - 1e-6) or np.any(q > limits[:, 1] + 1e-6):
                raise ValueError("Candidate arm exceeds joint limits")
            state = full.copy(); state[arm] = q; full_states.append(state)
        relations = self._prediction_relations if self.measured_attachment_prediction else (
            self.attachments if self.attached else None)
        if self.attached and relations is None:
            raise RuntimeError("Attachment prediction snapshot is invalid")
        poses = tuple(_world_tcp(model, state, support.tcp_link_name) @ relation
                      for model, state, support, relation in zip(self.models, full_states,
                          self._supports, relations)) if self.attached else None
        return self._evaluate(joints, poses, "candidate_two_rigid_predictions" if self.attached else "candidate_live_measured_objects")

    def measured_check(self) -> list[dict[str, Any]]:
        measured = self.lives[0]._measure_robots()
        if any(not np.allclose(q, previous, rtol=0, atol=1e-6)
               for q, previous in zip(measured, self._measured_qpos)):
            raise RuntimeError("Robot state changed since dual-payload refresh")
        joints = tuple(q[arm] for q, arm in zip(measured, self.native.arm_indices))
        return self._evaluate(joints, self._measured_payload_poses if self.attached else None,
                              "measured_actual_two_payloads" if self.attached else "measured_live_objects")
