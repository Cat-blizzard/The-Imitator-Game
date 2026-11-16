#!/usr/bin/env python3
"""Versioned resident cuRobo geometric grasp/payload sessions for TIG.

No simulator is imported or advanced. The caller supplies measured fingers,
complete world snapshots and real-grasp evidence. Reserved payload spheres are
actor-local metres; attachment offsets and all poses use XYZ/WXYZ. The official
attachment manager updates geometry, never PhysX objects. Payload inertia is
validated and retained as evidence, but is not injected into V2 TimeOpt.

Only two collision exceptions exist: own fingers versus own target, and (during
externally checked initial liftoff) the carried object versus the exact source
table. Both use collision layers; no robot link or obstacle is globally hidden.
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
from typing import Any, Callable

if __package__:
    from .curobo_tig_session_worker import _validate_request_fields
    from .curobo_tig_worker import ARM_JOINT_NAMES, FINGER_JOINT_NAMES, _finite_json, _matrix, _tensor_json, _vector
    from .curobo_tig_session_worker import (
        ResidentBackend, SessionManager, canonical_hash, robot_topology, serve,
        validate_request as validate_empty_request, world_topology, _pose, _tensor_identity,
    )
else:
    from curobo_tig_session_worker import _validate_request_fields
    from curobo_tig_worker import ARM_JOINT_NAMES, FINGER_JOINT_NAMES, _finite_json, _matrix, _tensor_json, _vector
    from curobo_tig_session_worker import (
        ResidentBackend, SessionManager, canonical_hash, robot_topology, serve,
        validate_request as validate_empty_request, world_topology, _pose, _tensor_identity,
    )

PAYLOAD_LINK = "tig_payload"
FINGER_LINKS = ["panda_leftfinger", "panda_rightfinger"]
SOURCE_TABLE = "scene::table::shape_0"
PROTOCOL_VERSION = 1


def _names(value: Any, label: str) -> list[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(v, str) or not v for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} must be distinct nonempty names")
    return value


def _rotation(pose: list[float]) -> list[list[float]]:
    """Unit-WXYZ quaternion to an orthogonal 3x3 rotation matrix."""
    _, _, _, w, x, y, z = _pose(pose, "rigid pose")
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def transport_spheres(spheres: Any, current_tcp: list[float],
                      goal_tcp: list[float]) -> list[list[float]]:
    """Rigidly move native finger spheres to the exact requested TCP pose.

    This does not solve IK or change joint state. Fingers locked to their
    measured values have a fixed TCP-relative geometry for every IK solution.
    Input/output [N,4] are robot-base XYZ metres and radii metres.
    """
    rows = _matrix(spheres, 4, "native finger spheres")
    current, goal = _pose(current_tcp, "current_tcp"), _pose(goal_tcp, "goal_tcp")
    old_r, new_r = _rotation(current), _rotation(goal)
    result = []
    for row in rows:
        displacement = [row[i]-current[i] for i in range(3)]
        local = [sum(old_r[j][i]*displacement[j] for j in range(3)) for i in range(3)]
        position = [goal[i]+sum(new_r[i][j]*local[j] for j in range(3)) for i in range(3)]
        result.append([*position, row[3]])
    assert len(result) == len(rows) and all(len(row) == 4 for row in result)
    _finite_json(result)
    return result


def sphere_box_gaps(spheres: Any, box_pose: list[float], dimensions: Any) -> list[float]:
    """Exact oriented-cuboid signed distance minus sphere radius, in metres.

    Negative gap means overlap of the conservative sphere with the box. This
    witness concerns native approximate geometry, not physical mesh contact.
    """
    rows = _matrix(spheres, 4, "spheres")
    pose, size = _pose(box_pose, "box_pose"), _vector(dimensions, 3, "box dimensions")
    if any(value <= 0 for value in size):
        raise ValueError("box dimensions must be positive")
    rotation = _rotation(pose)
    result = []
    for row in rows:
        if row[3] <= 0:
            raise ValueError("sphere-box witnesses accept only active positive-radius spheres")
        displacement = [row[i]-pose[i] for i in range(3)]
        local = [sum(rotation[j][i]*displacement[j] for j in range(3)) for i in range(3)]
        extent = [abs(local[i])-size[i]/2 for i in range(3)]
        outside = math.sqrt(sum(max(value, 0.)**2 for value in extent))
        signed_distance = outside + min(max(extent), 0.)
        result.append(signed_distance-row[3])
    _finite_json(result)
    return result


def validate_model(model: Any, robot_cfg: dict[str, Any], world: dict[str, Any], *,
                   extra_fields: frozenset[str] = frozenset()) -> None:
    """Validate immutable reserved topology and exact target ownership on CPU."""
    required_model = {"link_name", "capacity", "target_obstacle_names", "target_actor_name", "finger_links"}
    optional_model = {"schema", "sphere_count", "spheres_actor_local", "geometry_sha256", "shape_coverage",
                      "sphere_pitch_m", "physical_inertial", "grasp_stable_steps"}
    if (not isinstance(model, dict) or not required_model <= set(model)
            or set(model) - required_model - optional_model - extra_fields):
        raise ValueError("payload_model has unexpected or missing fields")
    if model["link_name"] != PAYLOAD_LINK or model["finger_links"] != FINGER_LINKS:
        raise ValueError("payload/finger links must be the declared Panda physical links")
    grasp_steps = model.get("grasp_stable_steps", 5)
    if type(grasp_steps) is not int or grasp_steps < 1:
        raise ValueError("grasp_stable_steps must be a positive integer")
    capacity = model["capacity"]
    if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 32768:
        raise ValueError("payload capacity must be an integer in [1,32768]")
    actor = model["target_actor_name"]
    if not isinstance(actor, str) or not actor or "::" in actor:
        raise ValueError("target_actor_name must be one exact physical actor name")
    target_names = _names(model["target_obstacle_names"], "target_obstacle_names")
    actual_names = {name for shapes in world.values() for name in shapes}
    expected_names = {name for name in actual_names if name.startswith(f"scene::{actor}::shape_")}
    if set(target_names) != expected_names or not expected_names:
        raise ValueError("target shapes must exactly cover the declared own physical actor")
    kin = robot_cfg["kinematics"]
    links = kin.get("extra_links", {})
    if set(links) != {PAYLOAD_LINK}:
        raise ValueError("exactly one reserved payload link is supported")
    link = links[PAYLOAD_LINK]
    if (link.get("link_name") != PAYLOAD_LINK or link.get("parent_link_name") != "panda_hand_tcp"
            or link.get("joint_type") != "FIXED"
            or link.get("fixed_transform") != [0., 0., 0., 1., 0., 0., 0.]
            or link.get("child_link_name") is not None):
        raise ValueError("payload link must be a fixed identity child of the physical TCP")
    if (link.get("link_mass") != 0. or link.get("link_com", [0., 0., 0.]) != [0., 0., 0.]
            or link.get("link_inertia") != [0.] * 6):
        raise ValueError("reserved geometric payload cannot add undeclared fictitious inertia")
    if kin.get("extra_collision_spheres") != {PAYLOAD_LINK: capacity}:
        raise ValueError("payload sphere capacity must be explicitly reserved once")
    if kin.get("collision_link_names", []).count(PAYLOAD_LINK) != 1:
        raise ValueError("payload link must occur exactly once in collision links")
    ignores = kin.get("self_collision_ignore", {})
    declared = set(ignores.get(PAYLOAD_LINK, []))
    declared.update(name for name, ignored in ignores.items() if PAYLOAD_LINK in ignored)
    if declared != set(FINGER_LINKS):
        raise ValueError("payload self-collision exceptions must be exactly its own two fingers")


def validate_state(state: Any, model: dict[str, Any], robot_cfg: dict[str, Any], *,
                   source_names: tuple[str, ...] = (SOURCE_TABLE,)) -> None:
    """Check measured-state shapes and narrowly declared permissions on CPU."""
    if not isinstance(state, dict) or set(state) != {
            "contact_allowed", "source_support_allowed", "source_support_obstacle_names", "attachment"}:
        raise ValueError("payload_state has unexpected or missing fields")
    if any(type(state[k]) is not bool for k in ("contact_allowed", "source_support_allowed")):
        raise ValueError("collision permissions must be explicit booleans")
    expected_support = list(source_names) if state["source_support_allowed"] else []
    if state["source_support_obstacle_names"] != expected_support:
        raise ValueError("source-support permission is limited to the one exact tabletop")
    attachment = state["attachment"]
    if attachment is None:
        if state["source_support_allowed"]:
            raise ValueError("empty hands cannot request source-support permission")
        return
    if not state["contact_allowed"]:
        raise ValueError("attached payload requires explicit own finger contact permission")
    required = {"spheres", "object_pose_base_wxyz", "positions", "obstacle_names", "evidence_id",
                "physical_inertial", "tcp_to_object_wxyz", "grasp_confirmation"}
    if not isinstance(attachment, dict) or set(attachment) != required:
        raise ValueError("attachment has unexpected or missing fields")
    spheres = _matrix(attachment["spheres"], 4, "actor-local payload spheres")
    if not 1 <= len(spheres) <= model["capacity"] or any(row[3] <= 0 for row in spheres):
        raise ValueError("actual payload spheres require positive radii within reserved capacity")
    if "spheres_actor_local" in model and spheres != model["spheres_actor_local"]:
        raise ValueError("attached geometry differs from the frozen physical target coverage")
    _pose(attachment["object_pose_base_wxyz"], "object_pose_base_wxyz")
    _pose(attachment["tcp_to_object_wxyz"], "tcp_to_object_wxyz")
    positions = _vector(attachment["positions"], 7, "attachment.positions")
    measured = robot_cfg["kinematics"]["cspace"]["default_joint_position"][:7]
    if max(abs(a-b) for a, b in zip(positions, measured, strict=True)) > 1e-7:
        raise ValueError("attachment FK must use the same measured arm as this snapshot")
    if attachment["obstacle_names"] != model["target_obstacle_names"]:
        raise ValueError("attachment may disable only all exact own target shapes")
    if not isinstance(attachment["evidence_id"], str) or not attachment["evidence_id"]:
        raise ValueError("attachment requires immutable real-grasp evidence identity")
    confirmation = attachment["grasp_confirmation"]
    if not isinstance(confirmation, dict):
        raise ValueError("actual shared-executor grasp confirmation is required")
    required_stable, observed = confirmation.get("stable_steps_required"), confirmation.get("stable_steps_observed")
    grasp_steps = model.get("grasp_stable_steps", 5)
    if type(grasp_steps) is not int or grasp_steps < 1:
        raise ValueError("grasp_stable_steps must be a positive integer")
    if (isinstance(required_stable, bool) or not isinstance(required_stable, int) or required_stable < grasp_steps
            or not isinstance(observed, list) or len(observed) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < required_stable for v in observed)
            or not isinstance(confirmation.get("confirmation_report_sha256"), str)
            or not confirmation["confirmation_report_sha256"]):
        raise ValueError("both real grasps must pass the existing stability barrier")
    inertia = attachment["physical_inertial"]
    if not isinstance(inertia, dict) or not {"mass_kg", "com_actor_m", "inertia_actor_kgm2"} <= set(inertia):
        raise ValueError("physical inertial evidence is missing or malformed")
    if inertia.get("payload_inertia_applied_to_native_timeopt", False) is not False:
        raise ValueError("payload inertia has not been injected into native TimeOpt")
    mass = _vector([inertia["mass_kg"]], 1, "physical mass")[0]
    if mass <= 0:
        raise ValueError("actual payload mass must be positive")
    _vector(inertia["com_actor_m"], 3, "payload COM")
    matrix = _matrix(inertia["inertia_actor_kgm2"], 3, "payload inertia")
    if len(matrix) != 3 or any(matrix[i][i] < 0 for i in range(3)):
        raise ValueError("inertia must be a nonnegative symmetric 3x3 matrix")
    if any(abs(matrix[i][j]-matrix[j][i]) > 1e-10 for i in range(3) for j in range(3)):
        raise ValueError("payload inertia must be symmetric")
    for i in range(3):
        for j in range(i):
            if matrix[i][i]*matrix[j][j] - matrix[i][j]**2 < -1e-15:
                raise ValueError("payload inertia must be positive semidefinite")
    determinant = (matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]**2)
                   - matrix[0][1]*(matrix[0][1]*matrix[2][2]-matrix[1][2]*matrix[0][2])
                   + matrix[0][2]*(matrix[0][1]*matrix[1][2]-matrix[1][1]*matrix[0][2]))
    if determinant < -1e-18:
        raise ValueError("payload inertia must be positive semidefinite")


def _validate_payload_request(request: Any, *, profile_validator: Callable[[str, dict[str, Any]], Any] | None = None,
                              model_fields: frozenset[str] = frozenset(),
                              scan_finite: bool = True) -> dict[str, Any]:
    """Validate the original payload schema, sharing measured-state fields."""
    if scan_finite:
        _finite_json(request)
    if not isinstance(request, dict):
        raise ValueError("request must be a mapping")
    operation = request.get("op")
    _validate_request_fields(request, payload_link=PAYLOAD_LINK,
        profile_validator=profile_validator, scan_finite=False)
    if operation == "init_session":
        validate_model(request.get("payload_model"), request["robot_cfg"], request["world"],
                       extra_fields=model_fields)
    elif operation == "update_session" and "payload_state" not in request:
        raise ValueError("each measured update requires explicit complete payload_state")
    return request


def validate_request(request: Any) -> dict[str, Any]:
    """Legacy payload entry point with the original model/source defaults."""
    return _validate_payload_request(request)


class PayloadBackend(ResidentBackend):
    """Resident public-API geometric attachment, with native tensor readbacks."""

    def _storage_identity(self) -> dict[str, Any]:
        result = super()._storage_identity()
        for kind, attr in (("cuboid", "cuboids"), ("mesh", "meshes")):
            storage = getattr(self.planner.scene_collision_checker.data, attr)
            if storage is not None:
                result["world"][kind]["obstacle_layers"] = _tensor_identity(storage.obstacle_layers)
        return result

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        self.payload_model = deepcopy(request["payload_model"])
        self.payload_state: dict[str, Any] = {
            "contact_allowed": False, "source_support_allowed": False,
            "source_support_obstacle_names": [], "attachment": None}
        self.measured_world = deepcopy(request["world"])
        result = super().configure(request)
        result["payload_model"] = deepcopy(self.payload_model)
        result["timeopt_payload_inertia_applied"] = False
        return result

    def _failure_geometry(self, request: dict[str, Any]) -> dict[str, Any]:
        """Read-only native-FK witness at the requested finger goal geometry."""
        box = self.measured_world.get("cuboid", {}).get(SOURCE_TABLE)
        if box is None:
            return {"status": "SOURCE_TABLE_NOT_PRESENT"}
        q = self._JointState.from_position(self.tensor_args.to_device([request["start"]]),
                                          joint_names=self.joint_names)
        state = self.planner.compute_kinematics(q)
        tcp = state.tool_poses.get_link_pose(self.tool_frame)
        current_tcp = [*_tensor_json(tcp.position)[0], *_tensor_json(tcp.quaternion)[0]]
        all_spheres = state.get_link_spheres()
        params = self.planner.kinematics.config.kinematics_config
        native_world = self._payload_readback()["world"]
        table_state = native_world[SOURCE_TABLE]
        rows = []
        for name in FINGER_LINKS:
            indices = params.get_sphere_index_from_link_name(name)
            current = _tensor_json(all_spheres[0, 0, indices, :])
            goal = transport_spheres(current, current_tcp, request["goal_pose"])
            gaps = sphere_box_gaps(goal, box["pose"], box["dims"])
            minimum = min(range(len(gaps)), key=gaps.__getitem__)
            masks = _tensor_json(params.sphere_collision_mask.view(self.torch.int32)[0, indices])
            rows.append({"finger_link": name, "sphere_count": len(goal),
                         "minimum_signed_gap_m": gaps[minimum],
                         "minimum_local_sphere_index": minimum,
                         "minimum_global_sphere_index": int(_tensor_json(indices)[minimum]),
                         "minimum_goal_sphere_xyzr": goal[minimum],
                         "native_sphere_mask": masks[minimum],
                         "native_pair_collision_enabled": bool(table_state["enabled"] and
                             masks[minimum] & table_state["layers"]),
                         "all_goal_spheres_xyzr": goal, "all_signed_gaps_m": gaps})
        return {"status": "COMPUTED", "witness_type": "native locked-finger spheres at exact requested TCP; not a solved IK state",
                "scope": "conservative geometric overlap; not actual PhysX mesh contact",
                "source_obstacle": SOURCE_TABLE, "native_table_state": table_state,
                "table_pose_xyz_wxyz": box["pose"], "table_dimensions_m": box["dims"],
                "current_tcp_xyz_wxyz": current_tcp, "requested_tcp_xyz_wxyz": request["goal_pose"],
                "finger_goal_table": rows, "extra_planning_calls": 0,
                "collision_permissions_changed": False, "environment_steps": 0}

    def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
        result = super().plan_pose(request)
        if result.get("success") is not True:
            begin = time.perf_counter()
            try:
                result["failure_geometry"] = self._failure_geometry(request)
            except Exception as error:
                # A diagnostic error must not replace the original native failure.
                result["failure_geometry"] = {"status": "DIAGNOSTIC_FAILED",
                                               "error_type": type(error).__name__, "error": str(error)}
            result["failure_geometry_wall_s"] = time.perf_counter()-begin
        return result

    def _payload_readback(self) -> dict[str, Any]:
        link = self.payload_model["link_name"]
        models, local_spheres = {}, None
        for name, model in self._models().items():
            params = model.config.kinematics_config
            spheres = _tensor_json(params.get_link_spheres(link))
            indices = params.get_sphere_index_from_link_name(link)
            masks = params.sphere_collision_mask.view(self.torch.int32)[0, indices]
            models[name] = {
                "payload_local_spheres_sha256": canonical_hash(spheres),
                "payload_active_sphere_count": sum(row[3] > 0 for row in spheres),
                "payload_collision_masks": sorted(set(_tensor_json(masks))),
                "finger_collision_masks": {},
            }
            for finger in FINGER_LINKS:
                index = params.get_sphere_index_from_link_name(finger)
                models[name]["finger_collision_masks"][finger] = sorted(set(_tensor_json(
                    params.sphere_collision_mask.view(self.torch.int32)[0, index])))
            if local_spheres is None:
                local_spheres = spheres
            elif spheres != local_spheres:
                raise RuntimeError("planner/IK/TrajOpt/graph payload spheres disagree")
        world: dict[str, Any] = {}
        data = self.planner.scene_collision_checker.data
        for attr in ("cuboids", "meshes"):
            storage = getattr(data, attr)
            if storage is None:
                continue
            enabled = _tensor_json(storage.enable[0])
            layers = _tensor_json(storage.obstacle_layers.view(self.torch.int32)[0])
            for i, name in enumerate(storage.names[0]):
                if name is not None:
                    world[name] = {"enabled": bool(enabled[i]), "layers": layers[i]}
        return {"state": deepcopy(self.payload_state), "models": models,
                "world": world, "local_spheres": local_spheres,
                "timeopt_payload_inertia_applied": False,
                "timeopt_dynamics_scope": "official URDF robot only; payload inertial evidence is not injected",
                "attachment_api": "MotionPlanner.attachment_manager.update/detach + KinematicsParams.update_link_spheres"}

    def fk(self, request: dict[str, Any]) -> dict[str, Any]:
        result = super().fk(request)
        positions = _matrix(request["positions"], 7, "positions")
        query = self._JointState.from_position(self.tensor_args.to_device(positions), joint_names=self.joint_names)
        spheres = self.planner.compute_kinematics(query).get_link_spheres()
        params = self.planner.kinematics.config.kinematics_config
        index = params.get_sphere_index_from_link_name(PAYLOAD_LINK)
        actual = _tensor_json(spheres[:, 0, index, :])
        assert len(actual) == len(positions) and len(actual[0]) == self.payload_model["capacity"]
        result["link_spheres"][PAYLOAD_LINK] = actual
        result["local_link_spheres"][PAYLOAD_LINK] = _tensor_json(params.get_link_spheres(PAYLOAD_LINK))
        result["link_sphere_indices"][PAYLOAD_LINK] = _tensor_json(index)
        result["payload"] = self._payload_readback()
        return result

    def update(self, robot_cfg: dict[str, Any], world: dict[str, Any],
               state: dict[str, Any]) -> dict[str, Any]:
        """Refresh real state, attachment and masks without replacing storage.

        A fresh same-topology config recomputes both independent finger locks.
        Public in-place copying first preserves current dynamic payload/masks;
        the official attachment manager then computes new actor-to-TCP spheres.
        Those exact spheres propagate to every independent native model.
        """
        from curobo.kinematics import KinematicsCfg

        self._require_configured()
        begin = time.perf_counter()
        fresh = KinematicsCfg.from_robot_yaml_file(
            {"robot_cfg": deepcopy(robot_cfg)}, device_cfg=self.tensor_args).kinematics_config
        seen: set[int] = set()
        for model in self._models().values():
            current = model.config.kinematics_config
            if id(current) in seen:
                continue
            for field in ("fixed_transforms", "link_spheres", "sphere_collision_mask", "link_map", "joint_map", "joint_map_type"):
                if tuple(getattr(current, field).shape) != tuple(getattr(fresh, field).shape):
                    raise ValueError(f"resident {field} shape changed")
            selected = current.get_sphere_index_from_link_name(PAYLOAD_LINK)
            ordinary = self.torch.ones(current.link_spheres.shape[1], dtype=self.torch.bool,
                                       device=current.link_spheres.device)
            ordinary[selected] = False
            if not self.torch.equal(current.link_spheres[:, ordinary, :], fresh.link_spheres[:, ordinary, :]):
                raise ValueError("ordinary robot sphere geometry changed")
            fresh.link_spheres.copy_(current.link_spheres)
            fresh.sphere_collision_mask.copy_(current.sphere_collision_mask)
            model.update_kinematics_config(fresh)
            seen.add(id(current))
        for shapes in world.values():
            for name, shape in shapes.items():
                self.planner.scene_collision_checker.update_obstacle_pose(
                    name, self._Pose.from_list(shape["pose"], device_cfg=self.tensor_args))
        self.lock_joints = deepcopy(robot_cfg["kinematics"]["lock_joints"])
        attachment = state["attachment"]
        manager = self.planner.attachment_manager
        if attachment is None:
            manager.detach(link_name=PAYLOAD_LINK,
                           enable_obstacle_names=self.payload_model["target_obstacle_names"])
            # Explicit reset also covers independent IK/graph copies.
            for model in self._models().values():
                model.config.kinematics_config.reset_link_spheres(PAYLOAD_LINK)
        else:
            joint_state = self._JointState.from_position(
                self.tensor_args.to_device([attachment["positions"]]), joint_names=self.joint_names)
            offset = self._Pose.from_list(attachment["object_pose_base_wxyz"], device_cfg=self.tensor_args)
            manager.update(self.tensor_args.to_device(attachment["spheres"]), joint_state,
                           link_name=PAYLOAD_LINK, world_objects_pose_offset=offset)
            local = manager.kinematics_params.get_link_spheres(PAYLOAD_LINK).clone()
            for model in self._models().values():
                model.config.kinematics_config.update_link_spheres(PAYLOAD_LINK, local)
        data = self.planner.scene_collision_checker.data
        target_names = set(self.payload_model["target_obstacle_names"])
        for shapes in world.values():
            for name in shapes:
                data.set_obstacle_layers(name, 2 if name in target_names else (4 if name == SOURCE_TABLE else 1))
                self.planner.scene_collision_checker.enable_obstacle(name, enable=not (attachment is not None and name in target_names))
        for model in self._models().values():
            params = model.config.kinematics_config
            for link in robot_cfg["kinematics"]["collision_link_names"]:
                mask = 7
                if link in FINGER_LINKS and state["contact_allowed"]:
                    mask &= ~2
                if link == PAYLOAD_LINK and state["source_support_allowed"]:
                    mask &= ~4
                params.set_link_collision_mask(link, mask)
        if self.planner.graph_planner is not None:
            self.planner.graph_planner.reset_buffer()
        self.torch.cuda.synchronize()
        identity = self._storage_identity()
        if identity != self.initial_storage_identity:
            raise RuntimeError("resident payload planner/model/world storage changed")
        self.payload_state = deepcopy(state)
        self.measured_world = deepcopy(world)
        readback = self._readback()
        for model_name, values in readback["robot_models"].items():
            q = values["lock_joint_positions"]
            while len(q) == 1 and isinstance(q[0], list):
                q = q[0]
            names = values["lock_joint_names"]
            if any(abs(q[names.index(n)] - self.lock_joints[n]) > 1e-7 for n in FINGER_JOINT_NAMES):
                raise RuntimeError(f"{model_name} did not update both measured finger locks")
        payload = self._payload_readback()
        disabled = {n for n, row in payload["world"].items() if not row["enabled"]}
        if disabled != (target_names if attachment is not None else set()):
            raise RuntimeError("native world disabled a shape outside the exact own held actor")
        return {"update_wall_s": time.perf_counter()-begin, "storage_identity": identity,
                "native_readback": readback, "updated_unique_kinematics_params": len(seen),
                "graph_nodes_cleared": self.planner.graph_planner is not None,
                "world_update_api": "scene_collision_checker.update_obstacle_pose/enable_obstacle + data.set_obstacle_layers",
                "kinematics_update_api": "KinematicsCfg.from_robot_yaml_file + Kinematics.update_kinematics_config",
                "payload": payload}


class PayloadSessionManager(SessionManager):
    """Keep immutable payload topology and phase-versioned attachment ownership."""

    def __init__(self, backend_factory: Callable[[], Any] = PayloadBackend) -> None:
        super().__init__(backend_factory)
        self.models: dict[str, dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.support_revoked: set[str] = set()

    def _receipt(self, session_id: str) -> dict[str, Any]:
        result = super()._receipt(session_id)
        result["payload_model"] = deepcopy(self.models[session_id])
        result["payload_state"] = deepcopy(self.states[session_id])
        result["timeopt_payload_inertia_applied"] = False
        return result

    def init_session(self, request: dict[str, Any]) -> dict[str, Any]:
        name = request["session_id"]
        if name in self.models:
            raise RuntimeError("resident payload session cannot be reinitialized")
        self.models[name] = deepcopy(request["payload_model"])
        self.states[name] = {"contact_allowed": False, "source_support_allowed": False,
                             "source_support_obstacle_names": [], "attachment": None}
        return super().init_session(request)

    def update_session(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session(request, require_valid=False)
        name, side = request["session_id"], request["session_id"].split(":")[0]
        self.invalidate(name, "payload update in progress")
        robot, world, state = request["robot_cfg"], request["world"], request["payload_state"]
        try:
            validate_model(self.models[name], robot, world)
            validate_state(state, self.models[name], robot)
            if (state["attachment"] is not None
                    and state["attachment"]["grasp_confirmation"].get("physical_arm_index") != (0 if side == "left" else 1)):
                raise ValueError("grasp confirmation belongs to the other physical arm")
            if robot_topology(robot) != robot_topology(session.robot_cfg):
                raise ValueError("reserved robot topology/limits/spheres/permissions changed")
            if world_topology(world) != world_topology(session.world):
                raise ValueError("complete world topology/geometry changed")
            if state["source_support_allowed"] and side in self.support_revoked:
                raise ValueError("initial source support was already revoked for this grasp lifecycle")
            content = canonical_hash({"robot_cfg": robot, "world": world, "payload_state": state})
            token = request["snapshot_id"]
            if token in session.used_snapshots and session.used_snapshots[token] != content:
                raise ValueError("snapshot_id cannot be rebound to different physical or payload content")
            native = session.backend.update(deepcopy(robot), deepcopy(world), deepcopy(state))
            if state["attachment"] is not None and not state["source_support_allowed"]:
                self.support_revoked.add(side)
            # Only explicit detach clears the source-support lifecycle guard.
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


def dispatch(request: Any, manager: PayloadSessionManager) -> dict[str, Any]:
    try:
        request = validate_request(request)
        with contextlib.redirect_stdout(sys.stderr):
            result = getattr(manager, request["op"])(request)
        response = {"id": request["id"], "ok": True, "result": result}
        _finite_json(response)
        return response
    except Exception as error:
        if isinstance(request, dict) and request.get("op") == "update_session":
            manager.invalidate(request.get("session_id"), f"{type(error).__name__}: {error}")
        raise


def serve_payload(source: Any, sink: Any, manager: PayloadSessionManager) -> None:
    """Reuse robust old framing, substituting only the payload-aware dispatch."""
    if __package__:
        from . import curobo_tig_session_worker as resident
    else:
        import curobo_tig_session_worker as resident
    previous = resident.dispatch
    resident.dispatch = dispatch
    try:
        serve(source, sink, manager)
    finally:
        resident.dispatch = previous


def self_test() -> None:
    """Representative CPU-only finite/schema/versioning/layer boundary tests."""
    assert "torch" not in sys.modules and "curobo" not in sys.modules
    identity = [0., 0., 0., 1., 0., 0., 0.]
    translated = transport_spheres([[0., 0., .1, .02]], identity,
                                   [0., 0., -.08, 1., 0., 0., 0.])
    assert abs(translated[0][2]-.02) < 1e-15 and translated[0][3] == .02
    assert abs(sphere_box_gaps(translated, [0., 0., -.025, 1., 0., 0., 0.], [1., 1., .05])[0]) < 1e-15
    rotated = transport_spheres([[1., 0., 0., .01]], identity,
                               [0., 0., 0., math.sqrt(.5), 0., 0., math.sqrt(.5)])
    assert abs(rotated[0][0]) < 1e-15 and abs(rotated[0][1]-1.) < 1e-15
    assert abs(sphere_box_gaps([[0., 0., 0., .1]], identity, [2., 2., 2.])[0]+1.1) < 1e-15

    class FakeBackend:
        def __init__(self) -> None:
            self.planner: Any = None
            self.lock_joints: dict[str, float] = {}

        def configure(self, request: dict[str, Any]) -> dict[str, Any]:
            self.planner = object()
            self.lock_joints = deepcopy(request["robot_cfg"]["kinematics"]["lock_joints"])
            return {"warmup_success": True}

        def update(self, robot: dict[str, Any], world: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
            if world["cuboid"][SOURCE_TABLE]["pose"][0] == 999:
                raise RuntimeError("synthetic partial native failure")
            self.lock_joints = deepcopy(robot["kinematics"]["lock_joints"])
            return {"payload": deepcopy(state)}

        def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"success": True}

        def fk(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"positions": [[0., 0., 0.] for _ in request["positions"]]}

        def shutdown(self) -> dict[str, Any]:
            self.planner = None
            return {"shutdown": True}

    def rejected(request: dict[str, Any], manager: PayloadSessionManager) -> None:
        try:
            dispatch(request, manager)
        except (RuntimeError, ValueError, TypeError, KeyError):
            return
        raise AssertionError("malformed payload request accepted")

    with tempfile.TemporaryDirectory(prefix="tig-curobo-payload-cpu-") as directory:
        urdf = Path(directory)/"synthetic.urdf"
        urdf.write_text('<robot name="synthetic">'+''.join(
            f'<joint name="{n}" type="revolute"><limit velocity="2"/></joint>'
            for n in ARM_JOINT_NAMES)+'</robot>')
        model = {"link_name": PAYLOAD_LINK, "capacity": 2, "target_actor_name": "apple",
                 "target_obstacle_names": ["scene::apple::shape_0"], "finger_links": FINGER_LINKS}
        robot = {"load_dynamics": False, "kinematics": {
            "urdf_path": str(urdf), "lock_joints": dict(zip(FINGER_JOINT_NAMES, [.03, .031])),
            "collision_link_names": ["panda_hand", *FINGER_LINKS, PAYLOAD_LINK],
            "collision_spheres": {"panda_hand": [{"center": [0., 0., 0.], "radius": .01}]},
            "extra_collision_spheres": {PAYLOAD_LINK: 2},
            "extra_links": {PAYLOAD_LINK: {"link_name": PAYLOAD_LINK, "joint_name": "payload_fixed",
                "joint_type": "FIXED", "parent_link_name": "panda_hand_tcp",
                "fixed_transform": [0., 0., 0., 1., 0., 0., 0.],
                "link_mass": 0., "link_com": [0., 0., 0.], "link_inertia": [0.]*6}},
            "self_collision_ignore": {PAYLOAD_LINK: FINGER_LINKS},
            "cspace": {"joint_names": ARM_JOINT_NAMES+FINGER_JOINT_NAMES,
                "default_joint_position": [0.]*7+[.03, .031],
                "velocity_scale": [.45]*9, "max_acceleration": .9}}}
        world = {"cuboid": {name: {"dims": [.1, .1, .1], "pose": [0., 0., 0., 1., 0., 0., 0.]}
                             for name in [SOURCE_TABLE, "scene::apple::shape_0", "scene::banana::shape_0"]}, "mesh": {}}
        init = {"id": 0, "op": "init_session", "session_id": "left:approach", "robot_cfg": robot,
                "world": world, "payload_model": model, "warmup": True}
        manager = PayloadSessionManager(FakeBackend)
        dispatch(init, manager)
        free = {"contact_allowed": False, "source_support_allowed": False,
                "source_support_obstacle_names": [], "attachment": None}
        update = {"id": 1, "op": "update_session", "session_id": "left:approach",
                  "robot_cfg": robot, "world": world, "snapshot_id": "free", "payload_state": free}
        dispatch(update, manager)
        attached = {"contact_allowed": True, "source_support_allowed": True,
                    "source_support_obstacle_names": [SOURCE_TABLE], "attachment": {
                        "spheres": [[0., 0., 0., .02]], "object_pose_base_wxyz": [0., 0., 0., 1., 0., 0., 0.],
                        "positions": [0.]*7, "obstacle_names": model["target_obstacle_names"],
                        "evidence_id": "actual-close-report-sha256", "tcp_to_object_wxyz": [0., 0., 0., 1., 0., 0., 0.],
                        "grasp_confirmation": {"physical_arm_index": 0, "stable_steps_required": 5,
                                               "stable_steps_observed": [5, 5], "confirmation_report_sha256": "synthetic"},
                        "physical_inertial": {"mass_kg": .05, "com_actor_m": [0., 0., 0.],
                                              "inertia_actor_kgm2": [[.01, 0., 0.], [0., .01, 0.], [0., 0., .01]]}}}
        request = {**update, "snapshot_id": "lift", "payload_state": attached}
        receipt = dispatch(request, manager)["result"]
        assert receipt["payload_state"]["attachment"]["spheres"] == [[0., 0., 0., .02]]
        before_id = receipt["planner_id"]
        invalid = deepcopy(request)
        invalid["payload_state"]["attachment"]["obstacle_names"] = ["scene::banana::shape_0"]
        rejected(invalid, manager)
        assert not manager.sessions["left:approach"].valid
        dispatch(request, manager)
        for path, value in [("sphere", float("nan")), ("negative", -.1), ("capacity", .1)]:
            invalid = deepcopy(request)
            if path == "capacity":
                invalid["payload_state"]["attachment"]["spheres"] *= 3
            else:
                invalid["payload_state"]["attachment"]["spheres"][0][3] = value
            rejected(invalid, manager)
        invalid = deepcopy(update)
        invalid["payload_state"]["source_support_allowed"] = True
        invalid["payload_state"]["source_support_obstacle_names"] = [SOURCE_TABLE]
        rejected(invalid, manager)
        carry = deepcopy(attached)
        carry.update(source_support_allowed=False, source_support_obstacle_names=[])
        carried = {**update, "snapshot_id": "carry", "payload_state": carry}
        dispatch(carried, manager)
        rejected({**request, "snapshot_id": "illegal-reopen"}, manager)
        assert manager.sessions["left:approach"].backend.planner is not None
        dispatch(carried, manager)
        assert dispatch({"id": 3, "op": "plan_pose", "session_id": "left:approach",
                         "snapshot_id": "carry", "start": [0.]*7,
                         "goal_pose": [0., 0., .5, 1., 0., 0., 0.]}, manager)["result"]["success"]
        rejected({**carried, "snapshot_id": "lift"}, manager)
        dispatch(carried, manager)
        assert dispatch({"id": 4, "op": "inspect_sessions"}, manager)["result"]["sessions"]["left:approach"]["planner_id"] == before_id
        bad_init = deepcopy(init)
        bad_init["session_id"] = "right:approach"
        bad_init["robot_cfg"]["kinematics"]["self_collision_ignore"][PAYLOAD_LINK].append("panda_hand")
        rejected(bad_init, manager)
        stream, sink = io.StringIO('{"id":5,"op":"unknown"}\n{"id":6,"op":"shutdown"}\n'), io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            serve_payload(stream, sink, manager)
        responses = [json.loads(line) for line in sink.getvalue().splitlines()]
        assert responses[0]["ok"] is False and responses[1]["ok"] is True and manager.closed
    assert "torch" not in sys.modules and "curobo" not in sys.modules
    print(json.dumps({"self_test": "PASS", "spheres_shape": [1, 4], "positions_shape": [1, 7],
                      "checks": ["exact_target_ownership", "reserved_tcp_topology", "own_fingers_only",
                                 "finite_positive_spheres", "physical_inertia_shape", "version_identity",
                                 "failed_update_invalidates", "bounded_source_support_lifecycle",
                                 "persistent_planner_identity", "strict_ndjson", "no_cuda_or_env_step"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--serve", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with protocol, contextlib.redirect_stdout(sys.stderr):
        serve_payload(sys.stdin, protocol, PayloadSessionManager())


if __name__ == "__main__":
    main()
