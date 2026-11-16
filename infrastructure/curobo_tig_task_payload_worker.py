#!/usr/bin/env python3
"""Three fixed TIG tasks with cached payload updates and exact static sources.

The process task is immutable. Source permissions are confined to the official
adapter's actual table or left/right source plate and all its exact registered
collision shapes. Complete geometry/poses are frozen at initialization. Only
the carried payload can receive a temporary source exception; robot links and
the peer source remain checked. Legacy protocol, grasp evidence, topology,
fixed-limit, stale-token and failed-update checks are retained.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, TextIO
import xml.etree.ElementTree as ET

if __package__:
    from .curobo_tig_fast_payload_worker import FastPayloadBackend
    from .curobo_tig_payload_worker import (
        PAYLOAD_LINK, FINGER_LINKS, SOURCE_TABLE, PayloadSessionManager, SessionManager,
        _finite_json, canonical_hash, robot_topology, world_topology,
        self_test as legacy_self_test, validate_model, validate_request as legacy_validate_request,
        validate_state,
    )
    from .curobo_tig_worker import ARM_JOINT_NAMES, FINGER_JOINT_NAMES, _vector, _tensor_json
else:
    from curobo_tig_fast_payload_worker import FastPayloadBackend
    from curobo_tig_payload_worker import (
        PAYLOAD_LINK, FINGER_LINKS, SOURCE_TABLE, PayloadSessionManager, SessionManager,
        _finite_json, canonical_hash, robot_topology, world_topology,
        self_test as legacy_self_test, validate_model, validate_request as legacy_validate_request,
        validate_state,
    )
    from curobo_tig_worker import ARM_JOINT_NAMES, FINGER_JOINT_NAMES, _vector, _tensor_json

TASK_IDS = ("TwoRobotPickAppleBananaToBaskets-v1", "TwoRobotPickPillToRegions-v1", "TwoRobotPickFruitsToPlate-v1")
TASK_ALIASES = dict(zip(("apple-banana", "pill", "fruits"), TASK_IDS))
APPROACH_LIMITS = dict(zip(TASK_IDS, ((.9, .9), (1., 1.3), (1.5, 2.6))))
SOURCES = {TASK_IDS[0]: ("table", "table"), TASK_IDS[1]: ("table", "table"),
           TASK_IDS[2]: ("003_plate_1", "003_plate_2")}
SOURCE_FIELDS = {"task_id", "physical_arm_index", "source_actor_name", "source_obstacle_names",
                 "source_all_obstacle_names", "source_body_type", "source_is_kinematic",
                 "source_linear_velocity_m_s", "source_angular_velocity_rad_s",
                 "source_world_shapes", "source_geometry_sha256"}


def canonical_task(value: str) -> str:
    result = TASK_ALIASES.get(value, value)
    if result not in TASK_IDS:
        raise ValueError("Only the three fixed TIG M1 task IDs are supported")
    return result


def legacy_model(model: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in model.items() if key not in SOURCE_FIELDS}


def legacy_state(state: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(state)
    result["source_support_obstacle_names"] = [SOURCE_TABLE] if result["source_support_allowed"] else []
    return result


def validate_source(model: dict[str, Any], world: dict[str, Any], task_id: str, arm: int) -> None:
    """Reject arbitrary actors, peer sources, missing shapes and changed solids."""
    task = canonical_task(task_id)
    if type(arm) is not int or arm not in (0, 1) or not SOURCE_FIELDS <= set(model):
        raise ValueError("source model needs the immutable canonical task and physical arm")
    if model["task_id"] != task or model["physical_arm_index"] != arm:
        raise ValueError("source manifest belongs to another task or physical arm")
    actor = model["source_actor_name"]
    if actor != SOURCES[task][arm] or actor == model["target_actor_name"]:
        raise ValueError("source actor differs from the fixed official task adapter")
    body_type, kinematic = model["source_body_type"], model["source_is_kinematic"]
    if type(kinematic) is not bool or not (
            (body_type == "PhysxRigidStaticComponent" and kinematic is False)
            or (body_type == "PhysxRigidDynamicComponent" and kinematic is True)):
        raise ValueError("source must be actual static or explicitly kinematic dynamic rigid body")
    for key in ("source_linear_velocity_m_s", "source_angular_velocity_rad_s"):
        values = _vector(model[key], 3, key)
        if any(abs(value) > 1e-12 for value in values):
            raise ValueError("fixed source must have zero measured linear and angular velocity")
    available = {name: {"kind": kind, **shape} for kind, group in world.items() for name, shape in group.items()}
    exact = sorted(name for name in available if name.startswith(f"scene::{actor}::shape_"))
    names = model["source_obstacle_names"]
    if not exact or model["source_all_obstacle_names"] != exact:
        raise ValueError("source manifest must freeze every exact own source actor shape")
    expected_permissions = exact if task == TASK_IDS[2] else [SOURCE_TABLE]
    if not isinstance(names, list) or names != expected_permissions:
        raise ValueError("table tasks retain exactly the already validated tabletop shape")
    if set(names) & set(model["target_obstacle_names"]):
        raise ValueError("source and target collision geometry must be disjoint")
    observed = {name: available[name] for name in exact}
    if model["source_world_shapes"] != observed or model["source_geometry_sha256"] != canonical_hash(observed):
        raise ValueError("static source geometry or pose changed after its frozen manifest")


def validate_task_profile(session_id: str, robot_cfg: dict[str, Any], task_id: str) -> None:
    velocity, acceleration = APPROACH_LIMITS[canonical_task(task_id)] if session_id.endswith(":approach") else (1.5, 2.6)
    kin = robot_cfg["kinematics"]
    names, cfg = kin["cspace"]["joint_names"], kin["cspace"]
    scales = _vector(cfg["velocity_scale"], 9, "task velocity scales")
    values = cfg["max_acceleration"]
    accel = _vector(values, 9, "task acceleration") if isinstance(values, list) else [float(values)]*9
    if any(not math.isfinite(a) or a <= 0 for a in accel):
        raise ValueError("task acceleration must be finite and positive")
    joints = {node.attrib["name"]: node for node in ET.parse(kin["urdf_path"]).getroot().findall("joint")}
    for name in ARM_JOINT_NAMES:
        index = names.index(name)
        physical = float(joints[name].find("limit").attrib["velocity"])
        if abs(scales[index]*physical-velocity) > 1e-9 or abs(accel[index]-acceleration) > 1e-10:
            raise ValueError("request does not match this task's immutable phase limit profile")


def validation_view(request: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Normalize only a strictly checked CPU view for legacy validations.

    The native request remains untouched. The old validator knows .9/.9 for
    approach and one tabletop identifier; actual task limits and exact source
    names are checked separately before constructing this view.
    """
    result = deepcopy(request)
    if request.get("op") == "init_session":
        result["payload_model"] = legacy_model(result["payload_model"])
        if request["session_id"].endswith(":approach"):
            kin = result["robot_cfg"]["kinematics"]
            names = kin["cspace"]["joint_names"]
            joints = {node.attrib["name"]: node for node in ET.parse(kin["urdf_path"]).getroot().findall("joint")}
            for name in ARM_JOINT_NAMES:
                index = names.index(name)
                kin["cspace"]["velocity_scale"][index] = .9/float(joints[name].find("limit").attrib["velocity"])
            values = kin["cspace"]["max_acceleration"]
            if not isinstance(values, list):
                values = [float(values)]*9
            for name in ARM_JOINT_NAMES:
                values[names.index(name)] = .9
            kin["cspace"]["max_acceleration"] = values
    elif request.get("op") == "update_session":
        result["payload_state"] = legacy_state(result["payload_state"])
    return result


def validate_task_request(request: Any, task_id: str) -> dict[str, Any]:
    _finite_json(request)
    if not isinstance(request, dict):
        raise ValueError("request must be a mapping")
    operation = request.get("op")
    if operation in {"init_session", "update_session"}:
        validate_task_profile(request["session_id"], request["robot_cfg"], task_id)
    if operation == "init_session":
        arm = 0 if request["session_id"].startswith("left:") else 1
        validate_source(request["payload_model"], request["world"], task_id, arm)
    legacy_validate_request(validation_view(request, task_id))
    return request


class TaskPayloadBackend(FastPayloadBackend):
    """Apply full inherited state, then exact task-source layers atomically."""
    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = canonical_task(task_id)

    def update(self, robot_cfg: dict[str, Any], world: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        begin = time.perf_counter()
        restricted = deepcopy(state)
        restricted.update(source_support_allowed=False, source_support_obstacle_names=[])
        result = super().update(robot_cfg, world, restricted)
        extra = time.perf_counter()
        sources = set(self.payload_model["source_obstacle_names"])
        targets = set(self.payload_model["target_obstacle_names"])
        for group in world.values():
            for name in group:
                self.planner.scene_collision_checker.data.set_obstacle_layers(
                    name, 2 if name in targets else (4 if name in sources else 1))
        for model in self._models().values():
            model.config.kinematics_config.set_link_collision_mask(
                PAYLOAD_LINK, 3 if state["source_support_allowed"] else 7)
        self.torch.cuda.synchronize()
        self.payload_state = deepcopy(state)
        result["native_readback"] = self._readback()
        result["payload"] = self._payload_readback()
        link_masks: dict[str, Any] = {}
        for model_name, model in self._models().items():
            params = model.config.kinematics_config
            link_masks[model_name] = {}
            for link in robot_cfg["kinematics"]["collision_link_names"]:
                indices = params.get_sphere_index_from_link_name(link)
                masks = sorted(set(_tensor_json(params.sphere_collision_mask.view(self.torch.int32)[0, indices])))
                expected = (3 if state["source_support_allowed"] else 7) if link == PAYLOAD_LINK else (
                    5 if link in FINGER_LINKS and state["contact_allowed"] else 7)
                if masks != [expected]:
                    raise RuntimeError("task-source update weakened another robot-link collision mask")
                link_masks[model_name][link] = masks
        result["payload"]["source_policy"] = {
            "task_id": self.task_id, "source_actor_name": self.payload_model["source_actor_name"],
            "source_obstacle_names": sorted(sources),
            "source_all_obstacle_names": self.payload_model["source_all_obstacle_names"],
            "source_geometry_sha256": self.payload_model["source_geometry_sha256"],
            "robot_link_collision_masks": link_masks,
            "all_robot_links_check_source": True, "peer_source_remains_obstacle": True,
            "external_exact_source_checker_required": state["source_support_allowed"]}
        identity = self._storage_identity()
        if identity != self.initial_storage_identity:
            raise RuntimeError("task-source permissions changed resident storage")
        result["storage_identity"] = identity
        result["update_timing_s"]["task_source_permissions_and_full_readback"] = time.perf_counter()-extra
        result["update_wall_s"] = time.perf_counter()-begin
        return result


class TaskSessionManager(PayloadSessionManager):
    def __init__(self, task_id: str, backend_factory: Callable[[], Any] | None = None) -> None:
        self.task_id = canonical_task(task_id)
        super().__init__(backend_factory or (lambda: TaskPayloadBackend(self.task_id)))

    def init_session(self, request: dict[str, Any]) -> dict[str, Any]:
        name = request["session_id"]
        if name in self.models:
            raise RuntimeError("resident task session cannot be reinitialized")
        arm = 0 if name.startswith("left:") else 1
        validate_source(request["payload_model"], request["world"], self.task_id, arm)
        self.models[name] = deepcopy(request["payload_model"])
        self.states[name] = {"contact_allowed": False, "source_support_allowed": False,
                             "source_support_obstacle_names": [], "attachment": None}
        return SessionManager.init_session(self, request)

    def update_session(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session(request, require_valid=False)
        name, side = request["session_id"], request["session_id"].split(":")[0]
        arm = 0 if side == "left" else 1
        self.invalidate(name, "task payload update in progress")
        robot, world, state = request["robot_cfg"], request["world"], request["payload_state"]
        try:
            validate_source(self.models[name], world, self.task_id, arm)
            validate_model(legacy_model(self.models[name]), robot, world)
            validate_state(legacy_state(state), legacy_model(self.models[name]), robot)
            expected = self.models[name]["source_obstacle_names"] if state["source_support_allowed"] else []
            if state["source_support_obstacle_names"] != expected:
                raise ValueError("payload exception differs from the frozen exact own-source shapes")
            if (state["attachment"] is not None
                    and state["attachment"]["grasp_confirmation"].get("physical_arm_index") != arm):
                raise ValueError("grasp confirmation belongs to another physical arm")
            if robot_topology(robot) != robot_topology(session.robot_cfg):
                raise ValueError("reserved robot topology/limits/spheres/permissions changed")
            if world_topology(world) != world_topology(session.world):
                raise ValueError("complete world topology/geometry changed")
            if state["source_support_allowed"] and side in self.support_revoked:
                raise ValueError("initial source support was already revoked for this grasp lifecycle")
            content = canonical_hash({"robot_cfg": robot, "world": world, "payload_state": state})
            token = request["snapshot_id"]
            if token in session.used_snapshots and session.used_snapshots[token] != content:
                raise ValueError("snapshot_id cannot identify different physical/source/payload content")
            native = session.backend.update(deepcopy(robot), deepcopy(world), deepcopy(state))
            if state["attachment"] is not None and not state["source_support_allowed"]:
                self.support_revoked.add(side)
            if state["attachment"] is None and self.states[name]["attachment"] is not None:
                self.support_revoked.discard(side)
            self.states[name] = deepcopy(state)
            session.robot_cfg, session.world = deepcopy(robot), deepcopy(world)
            session.snapshot_id, session.version = token, session.version+1
            session.used_snapshots[token] = content
            session.update_count += 1
            session.valid, session.invalid_reason = True, None
        except Exception as error:
            self.invalidate(name, f"{type(error).__name__}: {error}")
            raise
        return {**self._receipt(name), "native_update": native}


def dispatch_task(request: Any, manager: TaskSessionManager) -> dict[str, Any]:
    try:
        request = validate_task_request(request, manager.task_id)
        with contextlib.redirect_stdout(sys.stderr):
            result = getattr(manager, request["op"])(request)
        response = {"id": request["id"], "ok": True, "result": result}
        _finite_json(response)
        return response
    except Exception as error:
        if isinstance(request, dict) and request.get("op") == "update_session":
            manager.invalidate(request.get("session_id"), f"{type(error).__name__}: {error}")
        raise


def serve_task(source: TextIO, sink: TextIO, manager: TaskSessionManager) -> None:
    for line in source:
        request: Any = None
        try:
            request = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"nonfinite {v}")))
            response = dispatch_task(request, manager)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            ident = request.get("id") if isinstance(request, dict) else None
            if type(ident) is not int:
                ident = None
            response = {"id": ident, "ok": False, "error": {"type": type(error).__name__, "message": str(error)}}
        sink.write(json.dumps(response, allow_nan=False, separators=(",", ":"))+"\n")
        sink.flush()
        if response.get("ok") and request["op"] == "shutdown":
            return
    if not manager.closed:
        manager.shutdown({})


def self_test() -> None:
    legacy_self_test()
    assert "torch" not in sys.modules and "curobo" not in sys.modules
    rejected = 0
    with tempfile.TemporaryDirectory(prefix="tig-task-payload-cpu-") as directory:
        path = Path(directory)/"panda.urdf"
        path.write_text('<robot name="test">'+''.join(f'<joint name="{name}"><limit velocity="2"/></joint>' for name in ARM_JOINT_NAMES)+'</robot>')
        for task in TASK_IDS:
            for arm in (0, 1):
                source = SOURCES[task][arm]
                target = ("apple_out", "banana_out") [arm] if task == TASK_IDS[0] else (("pill_a", "pill_b")[arm] if task == TASK_IDS[1] else ("fruit_a", "fruit_b")[arm])
                names = [f"scene::{source}::shape_0"]
                world = {"cuboid": {name: {"pose": [0., 0., 0., 1., 0., 0., 0.], "dims": [.1]*3}
                    for name in set(names+[SOURCE_TABLE, f"scene::{target}::shape_0", "scene::foreign::shape_0"])}, "mesh": {}}
                shapes = {name: {"kind": "cuboid", **world["cuboid"][name]} for name in names}
                model = {"task_id": task, "physical_arm_index": arm, "target_actor_name": target,
                         "target_obstacle_names": [f"scene::{target}::shape_0"], "source_actor_name": source,
                         "source_obstacle_names": names, "source_body_type": "PhysxRigidStaticComponent",
                         "source_all_obstacle_names": names, "source_is_kinematic": False,
                         "source_linear_velocity_m_s": [0.]*3, "source_angular_velocity_rad_s": [0.]*3,
                         "source_world_shapes": shapes, "source_geometry_sha256": canonical_hash(shapes)}
                validate_source(model, world, task, arm)
                kinematic_model = {**model, "source_body_type": "PhysxRigidDynamicComponent", "source_is_kinematic": True}
                validate_source(kinematic_model,world,task,arm)
                for key, value in [("source_actor_name", "foreign"), ("source_body_type", "PhysxRigidDynamicComponent"),
                                   ("physical_arm_index", 1-arm), ("source_geometry_sha256", "bad"),
                                   ("source_linear_velocity_m_s", [0.,0.,.001]),
                                   ("source_angular_velocity_rad_s", [.001,0.,0.])]:
                    invalid = {**model, key: value}
                    try:
                        validate_source(invalid, world, task, arm)
                    except ValueError:
                        rejected += 1
                    else:
                        raise AssertionError("invalid source manifest accepted")
                changed = deepcopy(world)
                changed["cuboid"][names[0]]["pose"][0] = .1
                try:
                    validate_source(model, changed, task, arm)
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError("moved static source accepted")
                velocity, acceleration = APPROACH_LIMITS[task]
                robot = {"kinematics": {"urdf_path": str(path), "cspace": {"joint_names": ARM_JOINT_NAMES+FINGER_JOINT_NAMES,
                    "velocity_scale": [velocity/2.]*7+[1., 1.], "max_acceleration": [acceleration]*7+[.9,.9]}}}
                session = ("left" if arm == 0 else "right")+":approach"
                validate_task_profile(session, robot, task)
                view = validation_view({"op": "init_session", "session_id": session, "robot_cfg": robot, "payload_model": model}, task)
                assert view["robot_cfg"]["kinematics"]["cspace"]["velocity_scale"][:7] == [.45]*7
                assert robot["kinematics"]["cspace"]["velocity_scale"][:7] == [velocity/2.]*7
        assert rejected == 42
    print(json.dumps({"task_self_test": "PASS", "task_arm_shape": [3, 2], "negative_checks": rejected,
                      "checks": ["fixed_official_source_enum", "exact_shape_membership", "static_full_geometry_and_pose",
                                 "no_peer_or_foreign_permission", "task_specific_fixed_profiles", "legacy_validation_view_only",
                                 "inherited_full_payload_protocol_and_lifecycle", "no_native_or_simulator_import"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--task-id")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.task_id is None:
        parser.error("--task-id must fix one canonical supported process task")
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with protocol, contextlib.redirect_stdout(sys.stderr):
        serve_task(sys.stdin, protocol, TaskSessionManager(args.task_id))


if __name__ == "__main__":
    main()
