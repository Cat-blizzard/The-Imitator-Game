#!/usr/bin/env python3
"""Resident, versioned cuRobo V2 sessions for TIG, using JSON lines only.

Four independent MotionPlanner instances are owned by one process: physical
left/right arm times approach/transport limits.  Initialization and warmup are
one-time operations. Updates change only measured fingers, default joint state
and existing obstacle poses; the planner, sphere model and world storage remain
resident.  No TIG import, environment step, grasp attachment or payload is made.

Contract: positions are seven arm radians in explicit Panda joint order; poses
are selected-arm-base XYZ+WXYZ, and each plan names the exact successful update
snapshot. A failed update invalidates its session until a later complete update.
Run --self-test on CPU without importing CUDA, torch or cuRobo.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
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
    from .curobo_tig_worker import (
        ARM_JOINT_NAMES, FINGER_JOINT_NAMES, CuroboBackend, _finite_json,
        _matrix, _positive_float, _tensor_json, _vector,
    )
else:
    from curobo_tig_worker import (
        ARM_JOINT_NAMES, FINGER_JOINT_NAMES, CuroboBackend, _finite_json,
        _matrix, _positive_float, _tensor_json, _vector,
    )

PROTOCOL_VERSION = 1
SESSION_IDS = frozenset(f"{side}:{profile}" for side in ("left", "right")
                        for profile in ("approach", "transport"))
PROFILE_LIMITS = {"approach": (0.9, 0.9), "transport": (1.5, 2.6)}
WORLD_TYPES = {"cuboid": "cuboids", "mesh": "meshes"}


def canonical_hash(value: Any) -> str:
    """Hash exact JSON content, independent of object key ordering."""
    _finite_json(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def robot_topology(robot_cfg: dict[str, Any]) -> dict[str, Any]:
    """Remove exactly the two updateable robot-state fields from a copy."""
    result = deepcopy(robot_cfg)
    kin = result["kinematics"]
    kin.pop("lock_joints")
    kin["cspace"].pop("default_joint_position")
    return result


def world_topology(world: dict[str, Any]) -> dict[str, Any]:
    """Remove only obstacle poses; obstacle names and geometry remain exact."""
    return {kind: {name: {key: value for key, value in shape.items() if key != "pose"}
                   for name, shape in shapes.items()} for kind, shapes in world.items()}


def _pose(value: Any, name: str) -> list[float]:
    result = _vector(value, 7, name)
    if abs(math.sqrt(sum(x*x for x in result[3:])) - 1.0) > 1e-5:
        raise ValueError(f"{name} quaternion must be unit WXYZ")
    return result


def _validate_robot(robot_cfg: Any, *, payload_link: str | None = None) -> None:
    if not isinstance(robot_cfg, dict) or not isinstance(robot_cfg.get("kinematics"), dict):
        raise ValueError("robot_cfg requires a kinematics object")
    if robot_cfg.get("load_dynamics", False) is not False:
        raise ValueError("resident geometric sessions require load_dynamics=False")
    kin = robot_cfg["kinematics"]
    if set(kin.get("lock_joints", {})) != set(FINGER_JOINT_NAMES):
        raise ValueError("exactly the two independently measured fingers must be locked")
    for name in FINGER_JOINT_NAMES:
        value = _vector([kin["lock_joints"][name]], 1, f"locked {name}")[0]
        if not 0.0 <= value <= 0.04:
            raise ValueError("measured finger positions must be within [0,0.04] metres")
    cspace = kin.get("cspace", {})
    names = cspace.get("joint_names")
    if names != ARM_JOINT_NAMES + FINGER_JOINT_NAMES:
        raise ValueError("full cspace must preserve the explicit nine Panda joint names")
    default = _vector(cspace.get("default_joint_position"), 9, "default_joint_position")
    for name in FINGER_JOINT_NAMES:
        if abs(default[names.index(name)] - kin["lock_joints"][name]) > 1e-10:
            raise ValueError("default finger positions and measured locks disagree")
    # The empty-hand schema retains its ban. The payload schema validates
    # the exact reserved topology against its model without rewriting kin.
    for key in ("extra_links", "extra_collision_spheres"):
        if payload_link is None and kin.get(key):
            raise ValueError(f"{key} is unsupported in the empty-hand resident stage")
    if any("attach" in name.lower() or "payload" in name.lower()
           for name in kin.get("collision_link_names", []) if name != payload_link):
        raise ValueError("attachment/payload collision links are unsupported")


def _validate_urdf_path(robot_cfg: dict[str, Any]) -> Path:
    """Retain the init resource contract without parsing a second profile."""
    model = Path(robot_cfg["kinematics"].get("urdf_path", ""))
    if not model.is_absolute() or not model.is_file():
        raise ValueError("resident sessions require an absolute existing URDF")
    return model


def _validate_profile(session_id: str, robot_cfg: dict[str, Any]) -> None:
    """Check actual URDF-scaled arm velocity and per-joint acceleration limits."""
    velocity, acceleration = PROFILE_LIMITS[session_id.split(":")[1]]
    kin, cspace = robot_cfg["kinematics"], robot_cfg["kinematics"]["cspace"]
    acceleration_cfg = cspace.get("max_acceleration")
    if isinstance(acceleration_cfg, list):
        acceleration_values = _vector(acceleration_cfg, 9, "max_acceleration")
        if any(value <= 0 for value in acceleration_values):
            raise ValueError("max_acceleration must be positive for all nine joints")
    else:
        acceleration_values = [_positive_float(acceleration_cfg, "max_acceleration")] * 9
    if any(abs(value - acceleration) > 1e-10 for value in acceleration_values[:7]):
        raise ValueError(f"{session_id} requires acceleration {acceleration}")
    scales = _vector(cspace.get("velocity_scale"), 9, "velocity_scale")
    if any(scale <= 0 for scale in scales):
        raise ValueError("velocity_scale must be positive")
    model = _validate_urdf_path(robot_cfg)
    joints = {node.attrib["name"]: node for node in ET.parse(model).getroot().findall("joint")}
    for index, name in enumerate(ARM_JOINT_NAMES):
        limit = joints[name].find("limit")
        if limit is None:
            raise ValueError(f"actual URDF joint {name} has no velocity limit")
        actual_velocity = float(limit.attrib["velocity"]) * scales[index]
        if not math.isfinite(actual_velocity) or abs(actual_velocity - velocity) > 1e-9:
            raise ValueError(f"{session_id} joint {name} velocity {actual_velocity} != {velocity}")


def _validate_world(world: Any) -> None:
    if not isinstance(world, dict) or not set(world) <= set(WORLD_TYPES):
        raise ValueError("this stage accepts only existing cuboid and mesh obstacles")
    names: set[str] = set()
    for kind, shapes in world.items():
        if not isinstance(shapes, dict):
            raise ValueError(f"world.{kind} must be an obstacle-name mapping")
        for name, shape in shapes.items():
            if not isinstance(name, str) or not name or name in names or not isinstance(shape, dict):
                raise ValueError("world obstacle names must be unique nonempty strings")
            names.add(name)
            _pose(shape.get("pose"), f"world.{kind}.{name}.pose")
            if "collision" in shape and shape["collision"] is not True:
                raise ValueError("obstacle collision cannot be disabled")


def _validate_request_fields(request: Any, *, payload_link: str | None = None,
                             profile_validator: Callable[[str, dict[str, Any]], Any] | None = None,
                             scan_finite: bool = True) -> dict[str, Any]:
    """Non-mutating shared schema checks; options are fixed internal callers.

    The finite scan may be shared only within this same trusted call chain.
    No request field can disable a check, and no state is cached across calls.
    """
    if scan_finite:
        _finite_json(request)
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    version = request.get("protocol_version", PROTOCOL_VERSION)
    if isinstance(version, bool) or not isinstance(version, int) or version != PROTOCOL_VERSION:
        raise ValueError("unsupported resident-session protocol_version")
    ident = request.get("id")
    if isinstance(ident, bool) or not isinstance(ident, int) or ident < 0:
        raise ValueError("id must be a nonnegative integer")
    operation = request.get("op")
    allowed = {
        "init_session": {"session_id", "robot_cfg", "world", "seed", "interpolation_dt", "warmup"},
        "update_session": {"session_id", "robot_cfg", "world", "snapshot_id"},
        "fk": {"session_id", "positions", "snapshot_id"},
        "plan_pose": {"session_id", "snapshot_id", "start", "goal_pose", "max_attempts", "timeout_s"},
        "inspect_sessions": set(), "shutdown": set(),
    }
    if payload_link is not None:
        allowed["init_session"].add("payload_model")
        allowed["update_session"].add("payload_state")
    if operation not in allowed:
        raise ValueError("unknown resident operation")
    if set(request) - {"id", "op", "protocol_version"} - allowed[operation]:
        raise ValueError("request contains unsupported fields (payload/attachment is not implemented)")
    if operation in {"inspect_sessions", "shutdown"}:
        return request
    if request.get("session_id") not in SESSION_IDS:
        raise ValueError(f"session_id must be one of {sorted(SESSION_IDS)}")
    if operation in {"init_session", "update_session"}:
        _validate_robot(request.get("robot_cfg"), payload_link=payload_link)
        _validate_world(request.get("world"))
    if operation == "init_session":
        if request.get("warmup") is not True:
            raise ValueError("resident init requires exactly one warmup request")
        _positive_float(request.get("interpolation_dt", 0.02), "interpolation_dt")
        seed = request.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0,2**31)")
        checker = _validate_profile if profile_validator is None else profile_validator
        checker(request["session_id"], request["robot_cfg"])
    if operation in {"update_session", "plan_pose"} or "snapshot_id" in request:
        if not isinstance(request.get("snapshot_id"), str) or not request["snapshot_id"]:
            raise ValueError("snapshot_id must be a nonempty immutable caller identity")
    if operation == "fk":
        _matrix(request.get("positions"), 7, "positions")
    if operation == "plan_pose":
        _vector(request.get("start"), 7, "start")
        _pose(request.get("goal_pose"), "goal_pose")
        attempts = request.get("max_attempts", 4)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 4:
            raise ValueError("max_attempts must be an integer in [1,4]")
        _positive_float(request.get("timeout_s", 30.0), "timeout_s")
    return request


def validate_request(request: Any) -> dict[str, Any]:
    """Legacy empty-hand entry point with its original strict defaults."""
    return _validate_request_fields(request)


def _tensor_identity(tensor: Any) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "data_ptr": int(tensor.data_ptr()),
            "dtype": str(tensor.dtype)}


class ResidentBackend(CuroboBackend):
    """One never-rebuilt planner with public in-place state updates."""

    def _models(self) -> dict[str, Any]:
        self._require_configured()
        result = {"planner": self.planner.kinematics, "ik": self.planner.ik_solver.kinematics,
                  "trajopt": self.planner.trajopt_solver.kinematics}
        if self.planner.graph_planner is not None:
            result["graph"] = self.planner.graph_planner.kinematics
        return result

    def _storage_identity(self) -> dict[str, Any]:
        result: dict[str, Any] = {"planner_id": id(self.planner), "robot_models": {}, "world": {}}
        for name, model in self._models().items():
            params = model.config.kinematics_config
            result["robot_models"][name] = {
                "model_id": id(model), "params_id": id(params),
                "fixed_transforms": _tensor_identity(params.fixed_transforms),
                "link_spheres": _tensor_identity(params.link_spheres),
                "sphere_collision_mask": _tensor_identity(params.sphere_collision_mask)}
        data = self.planner.scene_collision_checker.data
        for kind, attr in WORLD_TYPES.items():
            storage = getattr(data, attr)
            if storage is not None:
                row = {"storage_id": id(storage), "capacity": int(storage.max_n),
                       "inv_pose": _tensor_identity(storage.inv_pose),
                       "names": deepcopy(storage.names)}
                if hasattr(storage, "mesh_ids"):
                    row["mesh_ids"] = _tensor_json(storage.mesh_ids)
                result["world"][kind] = row
        return result

    def _readback(self) -> dict[str, Any]:
        """Return actual native tensor contents, rather than echoed requests."""
        result: dict[str, Any] = {"robot_models": {}, "world": {}}
        for name, model in self._models().items():
            params = model.config.kinematics_config
            locked = params.lock_jointstate
            if locked is None:
                raise RuntimeError("native kinematics lost measured finger locks")
            result["robot_models"][name] = {
                "lock_joint_names": list(locked.joint_names),
                "lock_joint_positions": _tensor_json(locked.position),
                "default_joint_positions": _tensor_json(params.cspace.default_joint_position),
                "fixed_transforms_sha256": canonical_hash(_tensor_json(params.fixed_transforms)),
                "link_spheres_sha256": canonical_hash(_tensor_json(params.link_spheres)),
                "sphere_collision_mask_sha256": canonical_hash(_tensor_json(params.sphere_collision_mask)),
            }
        data = self.planner.scene_collision_checker.data
        for kind, attr in WORLD_TYPES.items():
            storage = getattr(data, attr)
            if storage is not None:
                names = storage.names[0]
                inv_poses = _tensor_json(storage.inv_pose[0, :, :7])
                enabled = _tensor_json(storage.enable[0])
                result["world"][kind] = {
                    name: {"inverse_pose_xyz_wxyz": inv_poses[index], "enabled": int(enabled[index])}
                    for index, name in enumerate(names) if name is not None}
        return result

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.planner is not None:
            raise RuntimeError("resident planner may be initialized only once")
        receipt = super().configure(request)
        self.initial_storage_identity = self._storage_identity()
        receipt["storage_identity"] = deepcopy(self.initial_storage_identity)
        receipt["native_readback"] = self._readback()
        return receipt

    def fk(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return TCP and actual post-lock collision spheres in robot-base frame.

        ``link_spheres[finger]`` has shape (N,K,4), with XYZ metres and radius
        metres. ``local_link_spheres`` has shape (K,4), preserving model order.
        These values are read from the same native model used by TrajOpt, so
        the caller can independently transform URDF finger spheres and compare.
        """
        result = super().fk(request)
        positions = _matrix(request["positions"], 7, "positions")
        query = self._JointState.from_position(self.tensor_args.to_device(positions),
                                               joint_names=self.joint_names)
        state = self.planner.compute_kinematics(query)
        spheres = state.get_link_spheres()
        assert len(spheres.shape) == 4 and tuple(spheres.shape[:2]) == (len(positions), 1)
        assert spheres.shape[-1] == 4
        params = self.planner.kinematics.config.kinematics_config
        actual, local, indices = {}, {}, {}
        for name in ("panda_leftfinger", "panda_rightfinger"):
            index = params.get_sphere_index_from_link_name(name)
            if len(index.shape) != 1 or index.shape[0] == 0:
                raise RuntimeError(f"actual native collision model has no {name} spheres")
            actual[name] = _tensor_json(spheres[:, 0, index, :])
            local[name] = _tensor_json(params.get_link_spheres(name))
            indices[name] = _tensor_json(index)
            assert len(actual[name]) == len(positions)
            for row in actual[name]:
                _matrix(row, 4, f"fk.{name}.spheres")
        self.torch.cuda.synchronize()
        result.update(link_spheres=actual, local_link_spheres=local,
                      link_sphere_indices=indices,
                      sphere_frame="selected_panda_link0",
                      collision_geometry_source="actual native KinematicsState.get_link_spheres")
        return result

    def update(self, robot_cfg: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
        """Refresh locked-finger kinematics and poses while preserving buffers."""
        from curobo.kinematics import KinematicsCfg

        self._require_configured()
        begin = time.perf_counter()
        fresh_cfg = KinematicsCfg.from_robot_yaml_file(
            {"robot_cfg": deepcopy(robot_cfg)}, device_cfg=self.tensor_args)
        fresh = fresh_cfg.kinematics_config
        updated_params: set[int] = set()
        for model in self._models().values():
            current = model.config.kinematics_config
            if id(current) in updated_params:
                continue
            for field_name in ("fixed_transforms", "link_spheres", "sphere_collision_mask",
                               "link_map", "joint_map", "joint_map_type"):
                if tuple(getattr(current, field_name).shape) != tuple(getattr(fresh, field_name).shape):
                    raise ValueError(f"kinematics {field_name} shape cannot change in place")
            for field_name in ("link_spheres", "sphere_collision_mask"):
                if not self.torch.equal(getattr(current, field_name), getattr(fresh, field_name)):
                    raise ValueError(f"kinematics {field_name} contents cannot change in place")
            model.update_kinematics_config(fresh)
            updated_params.add(id(current))
        for kind, shapes in world.items():
            for name, shape in shapes.items():
                pose = self._Pose.from_list(shape["pose"], device_cfg=self.tensor_args)
                self.planner.scene_collision_checker.update_obstacle_pose(name, pose)
        graph = self.planner.graph_planner
        if graph is not None:
            graph.reset_buffer()
        self.torch.cuda.synchronize()
        identity = self._storage_identity()
        if identity != self.initial_storage_identity:
            raise RuntimeError("resident planner/model/world storage identity changed during state update")
        self.lock_joints = deepcopy(robot_cfg["kinematics"]["lock_joints"])
        readback = self._readback()
        for model_name, measured in readback["robot_models"].items():
            values = measured["lock_joint_positions"]
            while len(values) == 1 and isinstance(values[0], list):
                values = values[0]
            names = measured["lock_joint_names"]
            if set(names) != set(FINGER_JOINT_NAMES) or len(values) != 2:
                raise RuntimeError(f"{model_name} returned malformed native lock joint state")
            if any(abs(values[names.index(name)] - self.lock_joints[name]) > 1e-7
                   for name in FINGER_JOINT_NAMES):
                raise RuntimeError(f"{model_name} did not apply both measured finger locks")
        return {"update_wall_s": time.perf_counter() - begin,
                "storage_identity": identity, "native_readback": readback,
                "updated_unique_kinematics_params": len(updated_params),
                "graph_nodes_cleared": graph is not None,
                "world_update_api": "scene_collision_checker.update_obstacle_pose",
                "kinematics_update_api": "KinematicsCfg.from_robot_yaml_file + Kinematics.update_kinematics_config"}


@dataclass
class Session:
    backend: Any
    robot_cfg: dict[str, Any]
    world: dict[str, Any]
    initialization: dict[str, Any]
    valid: bool = False
    snapshot_id: str | None = None
    version: int = 0
    update_count: int = 0
    plan_count: int = 0
    invalid_reason: str | None = "initialization requires a versioned update"
    used_snapshots: dict[str, str] = field(default_factory=dict)


class SessionManager:
    """CPU protocol/state ownership, independent of the actual CUDA backend."""

    def __init__(self, backend_factory: Callable[[], Any] = ResidentBackend) -> None:
        self.sessions: dict[str, Session] = {}
        self.backend_factory = backend_factory
        self.closed = False

    def invalidate(self, session_id: Any, reason: str) -> None:
        if isinstance(session_id, str) and session_id in self.sessions:
            session = self.sessions[session_id]
            session.valid = False
            session.invalid_reason = reason

    def _session(self, request: dict[str, Any], require_valid: bool = True) -> Session:
        if self.closed:
            raise RuntimeError("resident worker has shut down")
        session_id = request["session_id"]
        if session_id not in self.sessions:
            raise RuntimeError(f"session {session_id} has not been initialized")
        session = self.sessions[session_id]
        if require_valid and not session.valid:
            raise RuntimeError(f"session {session_id} is invalid: {session.invalid_reason}")
        if require_valid and "snapshot_id" in request and request["snapshot_id"] != session.snapshot_id:
            raise RuntimeError("request snapshot_id differs from the last successful session update")
        return session

    def _receipt(self, session_id: str) -> dict[str, Any]:
        session = self.sessions[session_id]
        backend = session.backend
        return {"protocol_version": PROTOCOL_VERSION, "session_id": session_id,
                "side": session_id.split(":")[0], "profile": session_id.split(":")[1],
                "valid": session.valid, "snapshot_id": session.snapshot_id, "version": session.version,
                "init_count": 1, "rebuild_count": 0, "warmup_count": 1,
                "update_count": session.update_count, "plan_count": session.plan_count,
                "planner_id": id(backend.planner), "backend_id": id(backend),
                "robot_cfg_sha256": canonical_hash(session.robot_cfg),
                "world_sha256": canonical_hash(session.world),
                "robot_topology_sha256": canonical_hash(robot_topology(session.robot_cfg)),
                "world_topology_sha256": canonical_hash(world_topology(session.world)),
                "locked_finger_positions": deepcopy(backend.lock_joints),
                "invalid_reason": session.invalid_reason}

    def init_session(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request["session_id"]
        if self.closed or session_id in self.sessions:
            raise RuntimeError("resident session cannot be reinitialized")
        backend = self.backend_factory()
        try:
            initialization = backend.configure(deepcopy(request))
        except Exception:
            backend.shutdown()
            raise
        self.sessions[session_id] = Session(backend, deepcopy(request["robot_cfg"]),
                                           deepcopy(request["world"]), initialization)
        return {**self._receipt(session_id), "initialization": initialization}

    def update_session(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session(request, require_valid=False)
        session_id = request["session_id"]
        self.invalidate(session_id, "update in progress")
        robot_cfg, world = request["robot_cfg"], request["world"]
        try:
            if robot_topology(robot_cfg) != robot_topology(session.robot_cfg):
                raise ValueError("robot topology, limits, sphere geometry or collision permissions changed")
            if world_topology(world) != world_topology(session.world):
                raise ValueError("world names, geometry, collision permissions or capacity changed")
            content_hash = canonical_hash({"robot_cfg": robot_cfg, "world": world})
            snapshot_id = request["snapshot_id"]
            if snapshot_id in session.used_snapshots and session.used_snapshots[snapshot_id] != content_hash:
                raise ValueError("a snapshot_id cannot be rebound to different physical content")
            native_receipt = session.backend.update(deepcopy(robot_cfg), deepcopy(world))
            session.robot_cfg, session.world = deepcopy(robot_cfg), deepcopy(world)
            session.snapshot_id, session.version = snapshot_id, session.version + 1
            session.used_snapshots[snapshot_id] = content_hash
            session.update_count += 1
            session.valid, session.invalid_reason = True, None
        except Exception as exc:
            self.invalidate(session_id, f"{type(exc).__name__}: {exc}")
            raise
        return {**self._receipt(session_id), "native_update": native_receipt}

    def fk(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session(request)
        return {**session.backend.fk(request), "session": self._receipt(request["session_id"])}

    def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session(request)
        result = session.backend.plan_pose(request)
        session.plan_count += 1
        return {**result, "session": self._receipt(request["session_id"])}

    def inspect_sessions(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"protocol_version": PROTOCOL_VERSION, "closed": self.closed,
                "session_count": len(self.sessions), "max_sessions": 4,
                "sessions": {name: self._receipt(name) for name in sorted(self.sessions)}}

    def shutdown(self, request: dict[str, Any]) -> dict[str, Any]:
        receipt = self.inspect_sessions(request)
        errors = []
        for session_id, session in self.sessions.items():
            try:
                session.backend.shutdown()
            except Exception as exc:
                errors.append({"session_id": session_id, "error": str(exc)})
            self.invalidate(session_id, "worker shut down")
        self.closed = True
        if errors:
            raise RuntimeError(f"resident shutdown failures: {errors}")
        return {"shutdown": True, "final_sessions": receipt["sessions"]}


def dispatch(request: Any, manager: SessionManager) -> dict[str, Any]:
    try:
        request = validate_request(request)
        with contextlib.redirect_stdout(sys.stderr):
            result = getattr(manager, request["op"])(request)
        response = {"id": request["id"], "ok": True, "result": result}
        _finite_json(response, "response")
        return response
    except Exception as exc:
        if isinstance(request, dict) and request.get("op") == "update_session":
            manager.invalidate(request.get("session_id"), f"{type(exc).__name__}: {exc}")
        raise


def serve(source: TextIO, sink: TextIO, manager: SessionManager) -> None:
    """Preserve request identities and strict finite JSON after every error."""
    for line in source:
        request: Any = None
        try:
            request = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")))
            response = dispatch(request, manager)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            ident = request.get("id") if isinstance(request, dict) else None
            if isinstance(ident, bool) or not isinstance(ident, int):
                ident = None
            response = {"id": ident, "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)}}
        sink.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
        sink.flush()
        if response.get("ok") and request["op"] == "shutdown":
            return
    # Broken input/parent exit cannot leave CUDA planners running indefinitely.
    if not manager.closed:
        manager.shutdown({})


def self_test() -> None:
    """CPU synthetic tests for versioning, topology, invalidation and framing."""
    assert "torch" not in sys.modules and "curobo" not in sys.modules

    class FakeBackend:
        def __init__(self) -> None:
            self.planner: Any = None
            self.lock_joints: dict[str, float] = {}
            self.configure_count = 0
            self.update_count = 0

        def configure(self, request: dict[str, Any]) -> dict[str, Any]:
            assert self.planner is None
            self.configure_count += 1
            self.planner = object()
            self.lock_joints = deepcopy(request["robot_cfg"]["kinematics"]["lock_joints"])
            return {"warmup_requested": True, "warmup_success": True}

        def update(self, robot_cfg: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
            if world["cuboid"]["table"]["pose"][0] == 999:
                raise RuntimeError("synthetic partial update failure")
            self.lock_joints = deepcopy(robot_cfg["kinematics"]["lock_joints"])
            self.update_count += 1
            return {"graph_nodes_cleared": True}

        def fk(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"positions": [[0., 0., 0.] for _ in request["positions"]]}

        def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"success": True, "position": [request["start"], request["start"]]}

        def shutdown(self) -> dict[str, bool]:
            self.planner = None
            return {"shutdown": True}

    def rejected(request: dict[str, Any], manager: SessionManager) -> None:
        try:
            dispatch(request, manager)
        except (ValueError, RuntimeError):
            return
        raise AssertionError(f"invalid request accepted: {request['op']}")

    with tempfile.TemporaryDirectory(prefix="tig-curobo-resident-cpu-") as directory:
        model = Path(directory) / "panda.urdf"
        model.write_text('<robot name="synthetic">' + ''.join(
            f'<joint name="{name}" type="revolute"><limit velocity="2.0"/></joint>'
            for name in ARM_JOINT_NAMES) + '</robot>')
        cfg = {"kinematics": {"urdf_path": str(model),
            "lock_joints": dict(zip(FINGER_JOINT_NAMES, [.037, .039])),
            "collision_link_names": ["panda_hand"], "collision_spheres": {"panda_hand": [{"radius": .02}]},
            "cspace": {"joint_names": ARM_JOINT_NAMES + FINGER_JOINT_NAMES,
                       "default_joint_position": [0.]*7 + [.037, .039],
                       "velocity_scale": [.45]*9, "max_acceleration": .9}}, "load_dynamics": False}
        world = {"cuboid": {"table": {"dims": [1., 1., .1], "pose": [0., 0., -.1, 1., 0., 0., 0.]}}, "mesh": {}}
        init = {"id": 0, "op": "init_session", "session_id": "left:approach", "robot_cfg": cfg,
                "world": world, "seed": 0, "interpolation_dt": .02, "warmup": True}
        manager = SessionManager(FakeBackend)
        dispatch(init, manager)
        plan = {"id": 1, "op": "plan_pose", "session_id": "left:approach", "snapshot_id": "s0",
                "start": [0.]*7, "goal_pose": [0., 0., .5, 1., 0., 0., 0.]}
        rejected(plan, manager)
        update = {"id": 2, "op": "update_session", "session_id": "left:approach",
                  "snapshot_id": "s0", "robot_cfg": deepcopy(cfg), "world": deepcopy(world)}
        before = id(manager.sessions["left:approach"].backend.planner)
        receipt = dispatch(update, manager)["result"]
        assert receipt["valid"] and receipt["version"] == 1
        assert receipt["init_count"] == receipt["warmup_count"] == 1 and receipt["rebuild_count"] == 0
        assert dispatch(plan, manager)["result"]["success"]
        rejected({**plan, "snapshot_id": "stale"}, manager)
        changed = deepcopy(update)
        changed["snapshot_id"] = "s1"
        changed["robot_cfg"]["kinematics"]["lock_joints"][FINGER_JOINT_NAMES[0]] = .031
        changed["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"][7] = .031
        changed["world"]["cuboid"]["table"]["pose"][0] = .1
        result = dispatch(changed, manager)["result"]
        assert result["version"] == 2 and result["snapshot_id"] == "s1"
        assert result["planner_id"] == before and result["update_count"] == 2
        assert result["locked_finger_positions"][FINGER_JOINT_NAMES[0]] == .031
        rejected({**changed, "snapshot_id": "s0"}, manager)
        rejected({**plan, "snapshot_id": "s1"}, manager)
        dispatch(changed, manager)
        for mutate in (
            lambda r: r.update(protocol_version=2),
            lambda r: r["robot_cfg"]["kinematics"]["cspace"].update(max_acceleration=2.),
            lambda r: r["robot_cfg"]["kinematics"]["collision_spheres"]["panda_hand"][0].update(radius=.03),
            lambda r: r["world"]["cuboid"]["table"].update(dims=[2., 1., .1]),
            lambda r: r["world"]["cuboid"].update(extra={"dims": [1., 1., .1], "pose": [0., 0., 0., 1., 0., 0., 0.]}),
            lambda r: r.update(payload={"mass": 1.}),
            lambda r: r["world"]["cuboid"]["table"]["pose"].__setitem__(0, 999),
        ):
            bad = deepcopy(changed)
            mutate(bad)
            rejected(bad, manager)
            rejected({**plan, "snapshot_id": "s1"}, manager)
            assert manager.sessions["left:approach"].snapshot_id == "s1"
            dispatch(changed, manager)
        rejected(init, manager)
        assert id(manager.sessions["left:approach"].backend.planner) == before
        for session_id in sorted(SESSION_IDS - {"left:approach"}):
            another = deepcopy(init)
            another["session_id"] = session_id
            velocity, acceleration = PROFILE_LIMITS[session_id.split(":")[1]]
            another["robot_cfg"]["kinematics"]["cspace"].update(
                velocity_scale=[velocity/2.]*7 + [.45]*2,
                max_acceleration=[acceleration]*7 + [.9]*2)
            dispatch(another, manager)
        inspected = dispatch({"id": 8, "op": "inspect_sessions"}, manager)["result"]
        assert inspected["session_count"] == 4
        assert len({row["planner_id"] for row in inspected["sessions"].values()}) == 4
        assert all(s.backend.configure_count == 1 for s in manager.sessions.values())
        stream = io.StringIO('{"id":9,"op":"unknown"}\n{"id":10,"op":"inspect_sessions"}\n'
                             '{"id":11,"op":"shutdown"}\n')
        sink = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            serve(stream, sink, manager)
        responses = [json.loads(line) for line in sink.getvalue().splitlines()]
        assert [row["id"] for row in responses] == [9, 10, 11]
        assert responses[0]["ok"] is False and responses[1]["ok"] and responses[2]["ok"]
        assert manager.closed and all(s.backend.planner is None for s in manager.sessions.values())
    assert "torch" not in sys.modules and "curobo" not in sys.modules
    print(json.dumps({"self_test": "PASS", "session_shape": [2, 2],
                      "position_shape": [1, 7], "full_cspace_shape": [9],
                      "checks": ["fixed_profile_limits", "four_distinct_resident_planners",
                                 "one_init_and_warmup", "versioned_finger_world_update",
                                 "topology_rejection", "failed_update_invalidates",
                                 "immutable_snapshot_ids", "strict_finite_json_lines",
                                 "no_cuda_import_or_environment_step"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with protocol, contextlib.redirect_stdout(sys.stderr):
        serve(sys.stdin, protocol, SessionManager())


if __name__ == "__main__":
    main()
