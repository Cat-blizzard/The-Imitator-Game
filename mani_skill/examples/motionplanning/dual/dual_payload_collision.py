"""Two measured or rigidly predicted payloads in one private dual-Panda world.

Robot identities are always physical (left, right). ``attach`` captures two
TCP/object relations for collision prediction only: it never welds, teleports,
steps, or changes a physical collision mask. The caller must confirm both real
grasps before attach. Refresh after every physical tick. Other actors are a
measured snapshot; finite joint sampling is not continuous collision checking.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
from mplib.pymp import PlanningWorld
from mplib.pymp.collision_detection import fcl

from .combined_collision import NativeDualCollisionChecker, _array, _vector
from .grasp_collision import FINGER_LINKS, LiveGraspCollisionChecker, _pose
from .held_collision import _entity_transform, _object_at, _object_transform, _pair_result, _world_tcp
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


def _self_test() -> dict[str, Any]:
    """Real CPU SAPIEN geometry/FCL queries, without any simulation step."""
    import xml.etree.ElementTree as ET
    import sapien
    import sapien.physx as physx
    from transforms3d.quaternions import mat2quat

    asset = Path(__file__).resolve().parents[3] / "assets/robots/panda/panda_v3"
    urdf = ET.parse(str(asset) + ".urdf").getroot()
    joints = [j.attrib["name"] for j in urdf.findall("joint") if j.attrib["type"] != "fixed"]
    links = [link.attrib["name"] for link in urdf.findall("link")]
    scene = sapien.Scene([physx.PhysxCpuSystem()]); actors: dict[str, Any] = {}
    raw = NS(num_envs=1, elapsed_steps=np.array([0]), scene=NS(actors=actors, articulations={}, sub_scenes=[scene],
                                px=scene.physx_system, gpu_sim_enabled=False))
    solvers = []
    for x in (0., 1.3):
        robot = NS(q=np.array([0, .39, 0, -1.96, 0, 2.36, .78, .02, .02]), _objs=[],
                   pose=NS(raw_pose=np.array([[x, 0, 0, 1, 0, 0, 0.]])))
        robot.get_qpos = lambda r=robot: r.q[None].copy()
        robot.get_active_joints = lambda: [NS(get_name=lambda n=n: n) for n in joints]
        agent = NS(robot=robot, gripper_joint_names=joints[-2:],
                   controller=NS(controllers={"arm": NS(config=NS(joint_names=joints[:7]))}))
        planner = NS(urdf=str(asset) + ".urdf", srdf=str(asset) + ".srdf", move_group="panda_hand",
                     user_joint_names=joints, user_link_names=links)
        solvers.append(NS(base_env=raw, robot=robot, env_agent=agent, planner=planner))
    raw.agent = NS(agents=[s.env_agent for s in solvers])
    fk = NativeDualCollisionChecker.from_solvers(*solvers)
    class TCP:
        def __init__(self, index: int): self.index = index
        @property
        def pose(self) -> Any:
            matrix = _world_tcp(fk.models[self.index], solvers[self.index].robot.q, "panda_hand_tcp")
            return NS(raw_pose=np.r_[matrix[:3, 3], mat2quat(matrix[:3, :3])][None])
    for index, solver in enumerate(solvers): solver.env_agent.tcp = TCP(index)
    material = physx.PhysxMaterial(.5, .5, 0)
    def add(name: str, shapes: list[Any], position: Any) -> Any:
        entity = sapien.Entity(); entity.name = name; body = physx.PhysxRigidStaticComponent()
        for shape in shapes: body.attach(shape)
        entity.add_component(body); entity.pose = sapien.Pose(position); scene.add_entity(entity)
        actors[name] = NS(_objs=[entity]); return actors[name]
    targets = []
    for index, name in enumerate(("apple", "banana")):
        shapes = [physx.PhysxCollisionShapeSphere(.01, material) for _ in range(2)]
        for shape, x in zip(shapes, (-.004, .004)): shape.set_local_pose(sapien.Pose([x, 0, 0]))
        tcp = _world_tcp(fk.models[index], solvers[index].robot.q, "panda_hand_tcp")
        targets.append(add(name, shapes, tcp[:3, 3] + [0, 0, -.03]))
    add("third", [physx.PhysxCollisionShapeSphere(.02, material)], [8, 8, 8])
    add("table", [physx.PhysxCollisionShapeBox([.2, .2, .03], material)], [8, 7, 8])
    ground = add("ground", [physx.PhysxCollisionShapePlane(material)], [0, 0, -3])
    ground._objs[0].pose = sapien.Pose([0, 0, -3], [2**-.5, 0, -2**-.5, 0])
    physical_before = {name: entity._objs[0].pose.to_transformation_matrix().copy() for name, entity in actors.items()}
    q_before = tuple(s.robot.q.copy() for s in solvers)
    checker = DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=tuple(targets))
    q = tuple(s.robot.q[:7].copy() for s in solvers)
    tests = []
    try: checker.attachments
    except ValueError: tests.append("unattached_relations_rejected")
    else: raise AssertionError("Unattached relations were exposed")
    checker.set_target_contacts_allowed((True, True))
    assert not checker.check(*q), checker.check(*q)
    checker.attach()
    assert checker.attached and len(checker.attachments) == 2
    assert not checker.measured_check(), checker.measured_check()
    assert not checker.check(*q), checker.check(*q)
    tests.append("two_real_Panda_models_and_two_payload_relations")
    initial = tuple(p.copy() for p in checker._measured_payload_poses)
    # Same path as public check: two independent replacements and real FCL.
    clear = tuple(np.eye(4) for _ in range(2)); clear[0][:3, 3] = [4, 0, 1]; clear[1][:3, 3] = [5, 0, 1]
    assert not checker._evaluate(q, clear, "fixture_clear"), checker._evaluate(q, clear, "fixture_clear")
    overlap = tuple(p.copy() for p in clear); overlap[1][:3, 3] = overlap[0][:3, 3]
    collisions = checker._evaluate(q, overlap, "fixture_overlap")
    pairs = [c for c in collisions if c["collision_type"] == "payload_payload"]
    assert pairs and all(c["object_name1"] in checker._target_names[0] and c["object_name2"] in checker._target_names[1] for c in pairs)
    assert len({checker._identity(c) for c in pairs}) == len(pairs)
    assert not checker._evaluate(q, clear, "fixture_clear_again")
    world_objects = dict(zip(checker.world.get_normal_object_names(), checker.world.get_normal_objects()))
    assert len(world_objects) == len(checker._normal_names)
    for support, pose in zip(checker._supports, clear):
        for name in support._target_names:
            assert np.allclose(_object_transform(world_objects[name]), pose @ support._shape_local[name])
    tests.extend(["actual_FCL_payload_payload_clear_overlap_clear", "both_payloads_replaced_no_stale_ghosts", "payload_pairs_evaluated_once"])
    candidate = (q[0] + [ .1, 0, 0, 0, 0, 0, 0], q[1] + [-.1, 0, 0, 0, 0, 0, 0])
    assert not checker.check(*candidate), checker.check(*candidate)
    predicted = checker.coverage["last_payload_T_world"]
    for index in (0, 1):
        full = checker._measured_qpos[index].copy(); full[checker.native.arm_indices[index]] = candidate[index]
        expected = _world_tcp(checker.models[index], full, "panda_hand_tcp") @ checker.attachments[index]
        assert np.allclose(predicted[index], expected) and not np.allclose(predicted[index], initial[index])
    assert np.linalg.norm(np.asarray(predicted[0])[:3, 3] - np.asarray(predicted[1])[:3, 3]) > 1.
    tests.append("canonical_both_distinct_candidate_FK_predictions")
    saved_base = checker.models[1].get_base_pose().copy()
    checker.models[1].set_base_pose(checker.models[0].get_base_pose())
    interarm = checker.check(*q)
    assert any({c.get("object_name1"), c.get("object_name2")} == {"left", "right"} for c in interarm)
    checker.models[1].set_base_pose(saved_base)
    assert not checker.check(*q)
    tests.append("actual_native_full_joint_interarm_overlap_rejects")
    world_hit = tuple(p.copy() for p in clear); world_hit[0][:3, 3] = [8, 8, 8]
    assert any(c["collision_type"] == "payload_environment" and c["object_name2"] == "scene::third::shape_0"
               for c in checker._evaluate(q, world_hit, "fixture_world"))
    robot_hit = tuple(p.copy() for p in clear); robot_hit[0][:3, 3] = [1.3, 0, .1]
    assert any(c["object_name1"] == "right" and c["object_name2"] in checker._target_names[0]
               for c in checker._evaluate(q, robot_hit, "fixture_other_robot"))
    collision_model = checker.models[0].get_fcl_model()
    palm = next(obj for link, obj in zip(collision_model.get_collision_link_names(), collision_model.get_collision_objects()) if link == "panda_hand")
    palm.get_collision_geometry().computeLocalAABB()
    palm_center = np.asarray(palm.get_rotation()) @ np.asarray(palm.get_collision_geometry().aabb_center) + np.asarray(palm.get_translation())
    palm_hit = tuple(p.copy() for p in clear); palm_hit[0][:3, 3] = palm_center
    assert any(c.get("object_name1") == "left" and c.get("link_name1") == "panda_hand" for c in checker._evaluate(q, palm_hit, "fixture_palm"))
    below = tuple(p.copy() for p in clear); below[1][2, 3] = -4
    assert any(c["collision_type"] == "conservative_plane_halfspace" and c.get("payload_index") == 1
               for c in checker._evaluate(q, below, "fixture_plane"))
    tests.extend(["payload_environment", "payload_other_robot", "actual_payload_palm_collision_not_allowed", "payload_plane_halfspace"])
    record = lambda robot, link, name: dict(object_name1=robot, link_name1=link, object_name2=name, link_name2=name)
    own = [next(iter(names)) for names in checker._target_names]
    for index, robot in enumerate(("left", "right")):
        assert checker._allowed(record(robot, "panda_leftfinger", own[index]))
        assert not checker._allowed(record(robot, "panda_hand", own[index]))
        assert not checker._allowed(record(robot, "panda_leftfinger", own[1-index]))
        assert not checker._allowed(record(robot, "panda_rightfinger", "scene::third::shape_0"))
    checker.detach(); assert not checker.attached
    # Place private FCL fixture objects at real finger-geometry centers. This
    # changes no physical entity; it exercises the live one-world query/filter.
    saved_objects = dict(checker._measured_objects)
    for index, model in enumerate(checker.models):
        cm = model.get_fcl_model()
        finger = next(obj for link, obj in zip(cm.get_collision_link_names(), cm.get_collision_objects()) if link == "panda_leftfinger")
        finger.get_collision_geometry().computeLocalAABB()
        center = np.asarray(finger.get_rotation()) @ np.asarray(finger.get_collision_geometry().aabb_center) + np.asarray(finger.get_translation())
        pose = np.eye(4); pose[:3, 3] = center
        for name in checker._target_names[index]:
            checker._measured_objects[name] = _object_at(checker.lives[index]._geometry[name], pose)
        if index == 0:
            name = "scene::third::shape_0"
            checker._measured_objects[name] = _object_at(checker.lives[0]._geometry[name], pose)
    def own_pair(records: Any, robot: str, target_names: Any) -> bool:
        return any(((r.get("object_name1") == robot and r.get("link_name1") in FINGER_LINKS and r.get("object_name2") in target_names)
                    or (r.get("object_name2") == robot and r.get("link_name2") in FINGER_LINKS and r.get("object_name1") in target_names)) for r in records)
    checker.set_target_contacts_allowed((False, False))
    both_blocked = checker.check(*q)
    assert own_pair(both_blocked, "left", checker._target_names[0]) and own_pair(both_blocked, "right", checker._target_names[1])
    checker.set_target_contacts_allowed((True, False))
    one_allowed = checker.check(*q)
    assert not own_pair(one_allowed, "left", checker._target_names[0]) and own_pair(one_allowed, "right", checker._target_names[1])
    checker.set_target_contacts_allowed((True, True))
    own_allowed = checker.check(*q)
    assert not own_pair(own_allowed, "left", checker._target_names[0]) and not own_pair(own_allowed, "right", checker._target_names[1])
    assert own_pair(checker.last_allowed_contacts, "left", checker._target_names[0]) and own_pair(checker.last_allowed_contacts, "right", checker._target_names[1])
    assert any("scene::third::shape_0" in (r.get("object_name1"), r.get("object_name2")) for r in own_allowed)
    checker._measured_objects = saved_objects; checker.refresh()
    tests.append("actual_live_FCL_both_own_fingers_union_but_third_object_blocked")
    checker.set_target_contacts_allowed((True, False))
    assert checker._allowed(record("left", "panda_leftfinger", own[0]))
    assert not checker._allowed(record("right", "panda_rightfinger", own[1]))
    checker.set_target_contacts_allowed((True, True))
    assert checker._allowed(record("right", "panda_rightfinger", own[1]))
    assert not checker._allowed(record("left", "panda_hand", own[0]))
    checker.set_target_contacts_allowed((False, False))
    assert not checker._allowed(record("left", "panda_leftfinger", own[0]))
    tests.extend(["live_union_of_own_fingers_only", "palm_cross_payload_and_third_object_not_allowed", "detach_returns_to_measured_live_world"])
    for name, actor in actors.items(): assert np.array_equal(actor._objs[0].pose.to_transformation_matrix(), physical_before[name])
    for solver, before in zip(solvers, q_before): assert np.array_equal(solver.robot.q, before)
    tests.append("no_physical_pose_joint_setters_or_environment_steps")
    # The bare CPU fixture now changes measured inputs explicitly; no scene or
    # task is stepped. Quality references remain the original attach capture.
    assert checker.measured_attachment_prediction is False
    checker.attach(); checker.check(*candidate)
    legacy_predictions = copy.deepcopy(checker.coverage["last_payload_T_world"])
    checker.detach(); checker.set_measured_attachment_prediction(True); checker.attach()
    checker.check(*candidate)
    assert np.array_equal(checker.coverage["last_payload_T_world"], legacy_predictions)
    references = tuple(value.copy() for value in checker.attachments)
    initial_relations = tuple(value.copy() for value in checker._prediction_relations)
    count, epoch = checker.refresh_count, checker.attachment_geometry_epoch
    current = checker.validate_measured_command(*q, grasping=(True, True), max_joint_step_rad=.02)
    assert current["status"] == "passed" and current["sampled_configurations"] == 2
    assert checker.refresh_count == count and checker.attachment_geometry_epoch == epoch
    assert checker.coverage["attachment_prediction_mode"] == "measured_control_step_snapshot"
    assert checker.coverage["prediction_snapshot_valid"] and current["prediction_T_tcp_objects"]
    tests.append("opt_in_unchanged_geometry_matches_legacy_and_command_reuses_one_refresh")
    sampled, original_check = [], checker.check
    def sample(left: np.ndarray, right: np.ndarray) -> list[dict[str, Any]]:
        sampled.append((left.copy(), right.copy())); return original_check(left, right)
    checker.check = sample
    sampled_command = checker.validate_measured_command(*candidate, grasping=(True, True), max_joint_step_rad=.02)
    del checker.check
    assert sampled_command["status"] == "passed" and len(sampled) == sampled_command["sampled_configurations"] >= 6
    assert all(np.array_equal(a, b) for a, b in zip(sampled[0], q))
    assert all(np.array_equal(a, b) for a, b in zip(sampled[-1], candidate))
    assert np.max(np.abs(np.diff(np.asarray(sampled), axis=0))) <= .02 + 1e-12
    assert checker.refresh_count == count
    class TensorCounter:
        def detach(self) -> Any: return self
        def cpu(self) -> Any: return self
        def numpy(self) -> np.ndarray: return np.array([0])
        def __array__(self, *args: Any, **kwargs: Any) -> Any: raise AssertionError("Direct tensor conversion is forbidden")
    raw.elapsed_steps = TensorCounter()
    assert checker.validate_measured_command(*q, grasping=(True, True), max_joint_step_rad=.02)["status"] == "passed"
    raw.elapsed_steps = np.array([0])
    tests.extend(["actual_FCL_every_measured_to_command_interval_at_existing_point02_joint_step",
                  "tensor_counter_uses_detach_cpu_numpy_without_direct_array_conversion"])

    def stale(call: Any) -> str:
        try: call()
        except (RuntimeError, ValueError) as exc: return str(exc)
        raise AssertionError("Stale measured attachment geometry was accepted")
    validate = lambda: checker.validate_measured_command(*q, grasping=(True, True), max_joint_step_rad=.02)
    raw.elapsed_steps += 1
    assert "control step" in stale(validate)
    assert "control step" in stale(lambda: checker.check(*q))
    raw.elapsed_steps -= 1
    solvers[0].robot.q[0] += .001
    assert "Robot state" in stale(validate)
    solvers[0].robot.q[0] -= .001
    solvers[1].robot.q[7] += .001
    assert "Robot state" in stale(validate)
    solvers[1].robot.q[7] -= .001
    solvers[0].robot.pose.raw_pose[0, 0] += .001
    assert "base" in stale(validate).lower()
    solvers[0].robot.pose.raw_pose[0, 0] -= .001
    tests.append("physical_counter_arm_finger_and_base_changes_reject_old_snapshot")

    entity = targets[0]._objs[0]; saved_pose = entity.pose
    entity.pose = sapien.Pose(np.asarray(saved_pose.p) + [.007, 0., .02], saved_pose.q)
    stale(validate)  # Actual object-pose guard, before a fresh capture.
    raw.elapsed_steps += 1; checker.refresh()
    assert checker.attachment_geometry_epoch == epoch + 1
    assert not np.allclose(checker._prediction_relations[0], initial_relations[0])
    assert all(np.array_equal(before, after) for before, after in zip(references, checker.attachments))
    assert np.array_equal(checker.coverage["attachment_T_tcp_objects"], references)
    assert np.array_equal(checker.coverage["prediction_T_tcp_objects"], checker._prediction_relations)
    assert validate()["status"] == "passed"
    tests.append("measured_relation_moves_without_reanchoring_fixed_quality_reference")
    obstacle = actors["third"]._objs[0]
    body = next(value for value in obstacle.components
                if isinstance(value, physx.PhysxRigidStaticComponent))
    # SAPIEN freezes a shape's local pose once attached. Replace the actual
    # body with the changed geometry, preserving the registered entity itself.
    moved_body = physx.PhysxRigidStaticComponent(); moved_shape = physx.PhysxCollisionShapeSphere(.02, material)
    moved_shape.set_local_pose(sapien.Pose([.001, 0., 0.])); moved_body.attach(moved_shape)
    obstacle.remove_component(body); obstacle.add_component(moved_body)
    stale(validate)
    obstacle.remove_component(moved_body); obstacle.add_component(body)
    changed_body = physx.PhysxRigidStaticComponent(); changed_body.attach(physx.PhysxCollisionShapeSphere(.021, material))
    obstacle.remove_component(body); obstacle.add_component(changed_body)
    stale(validate)
    # An unsuccessful refresh must revoke the previous prediction receipt.
    stale(checker.refresh)
    assert not checker.coverage["prediction_snapshot_valid"] and checker._prediction_relations is None
    stale(validate)
    obstacle.remove_component(changed_body); obstacle.add_component(body); entity.pose = saved_pose
    checker.refresh(); assert validate()["status"] == "passed"
    unconfirmed = checker.validate_measured_command(*q, grasping=(True, False), max_joint_step_rad=.02)
    assert unconfirmed["status"] == "rejected" and unconfirmed["reason"] == "attachment_grasp_unconfirmed"
    assert unconfirmed["sampled_configurations"] == 0
    tests.extend(["shape_local_pose_and_actual_shape_replacement_reject_old_snapshot",
                  "failed_refresh_invalidates_prediction_receipt", "unconfirmed_grasp_rejects_without_FCL_or_refresh"])
    # Exact locally installed YCB convex cooking, with task scales. This is a
    # separate bare CPU geometry scene; it does not instantiate/step the task.
    asset_root = Path(__file__).resolve().parents[6] / "maniskill/data/assets/mani_skill2_ycb/models"
    asset_scene = sapien.Scene([physx.PhysxCpuSystem()]); asset_actors = {}
    asset_raw = NS(num_envs=1, scene=NS(actors=asset_actors, articulations={}, sub_scenes=[asset_scene],
                                      px=asset_scene.physx_system, gpu_sim_enabled=False),
                   agent=NS(agents=[s.env_agent for s in solvers]))
    asset_solvers = [NS(**{**vars(s), "base_env": asset_raw}) for s in solvers]
    asset_targets, asset_centers, asset_info = [], [], []
    for index, (model_id, scale) in enumerate((("013_apple", .7), ("011_banana", 1.))):
        filename = asset_root / model_id / "collision.ply"
        if not filename.is_file(): raise ValueError(f"Required installed official asset is absent: {filename}")
        shapes = physx.PhysxCollisionShapeConvexMesh.load_multiple(str(filename), np.full(3, scale, dtype=np.float32), material)
        assert shapes
        entity = sapien.Entity(); entity.name = model_id; body = physx.PhysxRigidStaticComponent()
        for shape in shapes: body.attach(shape)
        entity.add_component(body); entity.pose = sapien.Pose([4. + index, 0, 1]); asset_scene.add_entity(entity)
        actor = NS(_objs=[entity]); asset_actors[model_id] = actor; asset_targets.append(actor)
        # Mean of the first cooked convex component's vertices is interior.
        local = np.asarray(shapes[0].get_local_pose().to_transformation_matrix())
        center = np.mean(np.asarray(shapes[0].vertices) * np.asarray(shapes[0].scale), axis=0)
        asset_centers.append(local[:3, :3] @ center + local[:3, 3])
        asset_info.append({"path": str(filename), "scale": scale, "cooked_convex_shapes": len(shapes)})
    asset_checker = DualPayloadCollisionChecker.from_solvers(asset_raw, *asset_solvers, targets=tuple(asset_targets))
    asset_checker.attach()
    asset_clear = tuple(p.copy() for p in asset_checker._measured_payload_poses)
    assert not asset_checker._evaluate(q, asset_clear, "official_asset_clear")
    asset_overlap = tuple(p.copy() for p in asset_clear)
    for index in (0, 1): asset_overlap[index][:3, 3] = np.array([4., 0, 1]) - asset_centers[index]
    asset_collisions = asset_checker._evaluate(q, asset_overlap, "official_asset_overlap")
    asset_pairs = [c for c in asset_collisions if c["collision_type"] == "payload_payload"]
    assert asset_pairs and all(c["object_name1"] in asset_checker._target_names[0] and c["object_name2"] in asset_checker._target_names[1] for c in asset_pairs)
    assert not asset_checker._evaluate(q, asset_clear, "official_asset_clear_again")
    asset_world = dict(zip(asset_checker.world.get_normal_object_names(), asset_checker.world.get_normal_objects()))
    assert len(asset_world) == len(asset_checker._normal_names)
    for support, pose in zip(asset_checker._supports, asset_clear):
        for name in support._target_names: assert np.allclose(_object_transform(asset_world[name]), pose @ support._shape_local[name])
    tests.append("official_apple_banana_scaled_cooked_convex_FCL_clear_overlap_clear_without_ghosts")
    # Actual cooked convex payloads initially touch one finite table. Measured
    # separation releases support permanently, even if the object returns.
    tabletop = .98
    source_poses = []
    for actor in asset_targets:
        body = next(value for value in actor._objs[0].components if isinstance(value, physx.PhysxRigidStaticComponent))
        vertices = []
        for shape in body.get_collision_shapes():
            local = shape.get_local_pose().to_transformation_matrix()
            scaled = np.asarray(shape.vertices) * np.asarray(shape.scale)
            vertices.append(scaled @ local[:3, :3].T + local[:3, 3])
        pose = actor._objs[0].pose
        position = np.asarray(pose.p).copy(); position[2] = tabletop - np.concatenate(vertices)[:, 2].min() - 5e-7
        actor._objs[0].pose = sapien.Pose(position, pose.q)
        source_poses.append(actor._objs[0].pose)
    table = sapien.Entity(); table.name = "table"; table_body = physx.PhysxRigidStaticComponent()
    table_body.attach(physx.PhysxCollisionShapeBox([1., .5, .03], material)); table.add_component(table_body)
    table.pose = sapien.Pose([4.5, 0., tabletop - .03]); asset_scene.add_entity(table); asset_actors["table"] = NS(_objs=[table])
    asset_raw.elapsed_steps = np.array([0])
    support_checker = DualPayloadCollisionChecker.from_solvers(asset_raw, *asset_solvers, targets=tuple(asset_targets))
    support_checker.set_measured_attachment_prediction(True); support_checker.attach()
    assert all(support._initial_support_exists and not support.support_released for support in support_checker._supports)
    support_references = tuple(relation.copy() for relation in support_checker.attachments)
    assert support_checker.validate_measured_command(*q, grasping=(True, True), max_joint_step_rad=.02)["status"] == "passed"
    for actor, pose in zip(asset_targets, source_poses):
        actor._objs[0].pose = sapien.Pose(np.asarray(pose.p) + [0., 0., .02], pose.q)
    asset_raw.elapsed_steps += 1; support_checker.refresh()
    assert all(support.support_released for support in support_checker._supports)
    assert support_checker.validate_measured_command(*q, grasping=(True, True), max_joint_step_rad=.02)["status"] == "passed"
    for actor, pose in zip(asset_targets, source_poses): actor._objs[0].pose = pose
    asset_raw.elapsed_steps += 1; support_checker.refresh()
    assert all(support.support_released for support in support_checker._supports)
    assert all(np.array_equal(a, b) for a, b in zip(support_references, support_checker.attachments))
    rejected_support = support_checker.validate_measured_command(*q, grasping=(True, True), max_joint_step_rad=.02)
    assert rejected_support["status"] == "rejected" and any(c["collision_type"] == "payload_environment"
        and c["object_name2"] == "scene::table::shape_0" for c in rejected_support["collisions"])
    tests.append("actual_convex_table_support_release_is_permanent_across_measured_relation_refresh")
    return {"status": "passed", "tests": tests, "physics_steps": 0, "gpu_used": False,
            "native_cpu_models": 2, "actual_FCL_payload_overlap_records": pairs,
            "official_asset_fixtures": asset_info, "official_asset_payload_overlap_records": asset_pairs,
            "measured_prediction_receipt": current, "source_support_return_receipt": rejected_support,
            "test_input_mutations": "Bare CPU geometry fixture only; no task or scene step was invoked"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(_self_test(), indent=2, allow_nan=False))
