"""Isolated cuRobo 0.9.0 worker for synchronous dual-Panda generation.

Run this file by path in the optional cuRobo environment. Only configure imports
CUDA libraries; importing this module needs Python alone. Each named session
owns one persistent planner and a fixed velocity/acceleration profile. Requests
and responses are JSON lines. Joint units are radians, distances metres, and
poses XYZ/WXYZ in that arm's base frame.

Native collision layers permit own-finger contact with the declared target and
payload contact with the source table during initial liftoff. The fixed base's
table mounting contact is allowed. The shared TIG checker still owns complete
paired-path acceptance and physical monitoring. Payload inertia is not injected
into TimeOpt. This worker never imports a simulator or advances physics.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import copy, deepcopy
import json
import math
import os
import random
import sys
import time
import traceback
from typing import Any, TextIO
import xml.etree.ElementTree as ET


ARM_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
FINGER_LINKS = ["panda_leftfinger", "panda_rightfinger"]
PAYLOAD_LINK = "tig_payload"
SOURCE_TABLE = "scene::table::shape_0"


def vector(value: Any, width: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{name} must have {width} entries")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(x) for x in value]


def pose(value: Any, name: str) -> list[float]:
    result = vector(value, 7, name)
    if abs(sum(x*x for x in result[3:])-1.) > 1e-5:
        raise ValueError(f"{name} requires a unit WXYZ quaternion")
    return result


def positive(value: Any, name: str) -> float:
    return_value = vector([value], 1, name)[0]
    if return_value <= 0:
        raise ValueError(f"{name} must be positive")
    return return_value


def select_cuda_device(torch: Any, requested: str) -> Any:
    """Select the current CUDA device before vendor configuration allocates."""
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError("cuRobo requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("cuRobo requires CUDA")
    torch.cuda.set_device(device)
    return device


def clean(value: Any) -> Any:
    """Encode native failure diagnostics, including infinite pose errors."""
    if hasattr(value, "detach"):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def single(value: Any) -> Any:
    value = clean(value)
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Expected one native result")
        value = value[0]
    return value


def matrix(value: Any, width: int, name: str) -> list[list[float]]:
    value = clean(value)
    while (isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
           and value[0] and isinstance(value[0][0], list)):
        value = value[0]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty matrix")
    return [vector(row, width, name) for row in value]


def trajectory_fields(trajectory: Any) -> dict[str, Any]:
    if trajectory is None:
        return {"positions": None, "times_s": None, "joint_names": ARM_NAMES}
    names = list(trajectory.joint_names)
    if len(names) != len(set(names)) or not set(ARM_NAMES).issubset(names):
        raise ValueError("Native trajectory lacks unique Panda arm names")
    raw = matrix(trajectory.position, len(names), "native trajectory")
    dt = positive(single(trajectory.dt), "native trajectory dt")
    fields = {"positions": [[row[names.index(n)] for n in ARM_NAMES] for row in raw],
              "times_s": [i*dt for i in range(len(raw))], "dt_s": dt, "joint_names": ARM_NAMES}
    for name in ("velocity", "acceleration"):
        value = getattr(trajectory, name, None)
        if value is None:
            raise ValueError(f"Successful native trajectory lacks {name}")
        rows = matrix(value, len(names), name)
        if len(rows) != len(raw):
            raise ValueError("Native trajectory derivative lengths differ")
        fields[name] = [[row[names.index(n)] for n in ARM_NAMES] for row in rows]
    return fields


def load_urdf(path: str) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Parse native velocity limits and the two independent prismatic fingers."""
    joints = {node.attrib["name"]: node for node in ET.parse(path).getroot().findall("joint")}
    limits = {name: positive(float(joints[name].find("limit").attrib["velocity"]), name)
              for name in ARM_NAMES}
    fingers = {}
    for name, child in zip(FINGER_NAMES, FINGER_LINKS):
        joint = joints[name]
        if (joint.attrib.get("type") != "prismatic" or joint.find("mimic") is not None
                or joint.find("parent").attrib["link"] != "panda_hand"
                or joint.find("child").attrib["link"] != child):
            raise ValueError("Worker requires the derived independent-finger Panda URDF")
        origin, axis = joint.find("origin"), joint.find("axis")
        xyz = vector([float(x) for x in origin.attrib.get("xyz", "0 0 0").split()], 3, "finger origin")
        roll, pitch, yaw = vector([float(x) for x in origin.attrib.get("rpy", "0 0 0").split()], 3, "finger RPY")
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rotation = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                    [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
        direction = vector([float(x) for x in axis.attrib["xyz"].split()], 3, "finger axis")
        if abs(sum(x*x for x in direction)-1.) > 1e-10:
            raise ValueError("Finger axis must be unit length")
        fingers[name] = {"child": child, "origin": xyz, "rotation": rotation, "axis": direction}
    return limits, fingers


def finger_transform(joint: dict[str, Any], position: float) -> list[list[float]]:
    q = vector([position], 1, "finger position")[0]
    rotation = joint["rotation"]
    translation = [joint["origin"][i]+q*sum(rotation[i][j]*joint["axis"][j] for j in range(3)) for i in range(3)]
    return [[*rotation[i], translation[i]] for i in range(3)]


def profile_config(robot: dict[str, Any], velocity: float, acceleration: float,
                   urdf_limits: dict[str, float]) -> dict[str, Any]:
    config = deepcopy(robot)
    kin = config["kinematics"]
    names = kin["cspace"]["joint_names"]
    if set(names) != set(ARM_NAMES+FINGER_NAMES) or len(names) != 9:
        raise ValueError("Expected nine distinct physical Panda joints")
    if set(kin["lock_joints"]) != set(FINGER_NAMES):
        raise ValueError("Both measured finger coordinates must be locked independently")
    vector([kin["lock_joints"][n] for n in FINGER_NAMES], 2, "finger locks")
    vector(kin["cspace"]["default_joint_position"], 9, "default joint positions")
    # Public v0.9.0 applies velocity_scale in the loader and model construction.
    kin["cspace"]["velocity_scale"] = [math.sqrt(velocity/urdf_limits[n]) if n in ARM_NAMES else 1. for n in names]
    kin["cspace"]["max_acceleration"] = acceleration
    config["load_dynamics"] = False
    return config


def robot_topology(robot: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(robot)
    kin = result["kinematics"]
    kin.pop("lock_joints", None)
    kin["cspace"].pop("default_joint_position", None)
    return result


def world_topology(world: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(world)
    for kind, shapes in result.items():
        if kind not in ("cuboid", "mesh") or not isinstance(shapes, dict):
            raise ValueError("World supports named cuboid and mesh obstacles")
        for shape in shapes.values():
            pose(shape.pop("pose"), "obstacle pose")
    return result


def empty_payload_state() -> dict[str, Any]:
    return {"contact_allowed": False, "source_support_allowed": False,
            "source_support_obstacle_names": [], "attachment": None}


def validate_payload_model(model: dict[str, Any], kin: dict[str, Any], world: dict[str, Any]) -> None:
    """A session may attach only its declared actor to the reserved TCP link."""
    capacity = model["capacity"]
    actor = model["target_actor_name"]
    actual = {n for shapes in world.values() for n in shapes
              if n.startswith(f"scene::{actor}::shape_")}
    if (not isinstance(actor, str) or not actor or "::" in actor or actor == "table"
            or not actual or set(model["target_obstacle_names"]) != actual):
        raise ValueError("Payload model must name all exact shapes of its own target actor")
    if (type(capacity) is not int or capacity < 1
            or model.get("link_name", PAYLOAD_LINK) != PAYLOAD_LINK
            or model.get("finger_links", FINGER_LINKS) != FINGER_LINKS
            or kin.get("extra_collision_spheres") != {PAYLOAD_LINK: capacity}
            or kin["collision_link_names"].count(PAYLOAD_LINK) != 1):
        raise ValueError("Payload model must match the explicitly reserved TCP sphere capacity")
    extra = kin.get("extra_links", {})
    link = extra.get(PAYLOAD_LINK, {})
    if (set(extra) != {PAYLOAD_LINK} or link.get("parent_link_name") != "panda_hand_tcp"
            or link.get("joint_type") != "FIXED"
            or link.get("fixed_transform") != [0., 0., 0., 1., 0., 0., 0.]):
        raise ValueError("Reserved payload must be fixed to the physical TCP")
    ignores = kin.get("self_collision_ignore", {})
    excluded = set(ignores.get(PAYLOAD_LINK, []))
    excluded.update(n for n, values in ignores.items() if PAYLOAD_LINK in values)
    if excluded != set(FINGER_LINKS):
        raise ValueError("Payload self-collision exceptions must be exactly its own fingers")
    if "spheres_actor_local" in model:
        spheres = matrix(model["spheres_actor_local"], 4, "configured payload spheres")
        if len(spheres) > capacity or any(row[3] <= 0 for row in spheres):
            raise ValueError("Configured payload geometry exceeds reserved capacity")


def collision_policy(model: dict[str, Any] | None, state: dict[str, Any], world: dict[str, Any],
                     links: list[str]) -> tuple[dict[str, int], dict[str, int], set[str]]:
    """Compute narrow layer masks without hiding peer targets or the table."""
    if any(type(state.get(n)) is not bool for n in ("contact_allowed", "source_support_allowed")):
        raise ValueError("Payload permissions must be explicit booleans")
    all_names = {n for shapes in world.values() for n in shapes}
    targets = set(model["target_obstacle_names"]) if model else set()
    if not targets <= all_names:
        raise ValueError("Declared target geometry is absent from the world")
    attachment = state.get("attachment")
    if (model is None and (attachment is not None or state["contact_allowed"] or state["source_support_allowed"])):
        raise ValueError("Target contact and attachment require a reserved payload model")
    sources = state.get("source_support_obstacle_names", [])
    if sources != ([SOURCE_TABLE] if state["source_support_allowed"] else []):
        raise ValueError("Source support is limited to the exact original table")
    if state["source_support_allowed"] and (attachment is None or SOURCE_TABLE not in all_names):
        raise ValueError("Source support requires a held object and the actual table")
    if attachment is not None:
        if not state["contact_allowed"] or set(attachment["obstacle_names"]) != targets:
            raise ValueError("Attachment may disable only its exact own target shapes")
        vector(attachment["positions"], 7, "attachment positions")
        pose(attachment["object_pose_base_wxyz"], "attachment object pose")
        spheres = matrix(attachment["spheres"], 4, "actor-local payload spheres")
        if len(spheres) > model["capacity"] or any(row[3] <= 0 for row in spheres):
            raise ValueError("Payload spheres must have positive radii within reserved capacity")
        if "spheres_actor_local" in model and spheres != model["spheres_actor_local"]:
            raise ValueError("Attached geometry differs from the configured target model")
    layers = {n: 2 if n in targets else 4 if n == SOURCE_TABLE else 1 for n in all_names}
    masks = {link: 7 for link in links}
    if "panda_link0" in masks:
        masks["panda_link0"] &= ~4  # fixed mounting contact only; self-collision remains enabled
    if state["contact_allowed"]:
        for link in FINGER_LINKS:
            masks[link] &= ~2
    if state["source_support_allowed"]:
        masks[PAYLOAD_LINK] &= ~4
    return layers, masks, targets if attachment is not None else set()


def model_limit_receipt(limits: Any, velocity: float, acceleration: float) -> dict[str, Any]:
    names = list(limits.joint_names)
    if names != ARM_NAMES:
        raise ValueError(f"Unexpected active joint order: {names}")
    receipt = {"joint_names": names}
    for name, requested in (("velocity", velocity), ("acceleration", acceleration)):
        rows = matrix(getattr(limits, name), 7, name)
        if len(rows) != 2 or any(not math.isclose(v, sign*requested, rel_tol=1e-5, abs_tol=1e-6)
                                for sign, row in zip((-1., 1.), rows) for v in row):
            raise RuntimeError(f"Native {name} limits differ from requested profile")
        receipt[name] = rows
    receipt.update(velocity_scale_adapter="curobo-0.9.0-double-scale-sqrt-v1", timeopt_default_limit_scale=.95)
    return receipt


class Session:
    def __init__(self) -> None:
        self.planner: Any = None
        self.ready = False
        self.snapshot_id = 0

    def models(self) -> list[Any]:
        models = [self.planner.kinematics, self.planner.ik_solver.kinematics, self.planner.trajopt_solver.kinematics]
        if self.planner.graph_planner is not None:
            models.append(self.planner.graph_planner.kinematics)
        return list({id(m.config.kinematics_config): m for m in models}.values())

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        import torch
        target_device = select_cuda_device(torch, request.get("device", "cuda:0"))
        import curobo
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg, JointState, Pose, GoalToolPose

        if str(curobo.__version__) != "0.9.0":
            raise RuntimeError(f"This optional backend supports cuRobo 0.9.0; found {curobo.__version__}")
        begin = time.perf_counter()
        self.torch, self.JointState, self.Pose, self.GoalToolPose = torch, JointState, Pose, GoalToolPose
        self.device = DeviceCfg(device=target_device, dtype=torch.float32)
        self.velocity = positive(request.get("velocity_limit", .9), "velocity limit")
        self.acceleration = positive(request.get("acceleration_limit", 1.2), "acceleration limit")
        seed = request.get("seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0,2**31)")
        robot = request["robot_cfg"]
        urdf_limits, self.fingers = load_urdf(robot["kinematics"]["urdf_path"])
        config = profile_config(robot, self.velocity, self.acceleration, urdf_limits)
        self.robot_structure, self.world_structure = robot_topology(robot), world_topology(request["world"])
        self.payload_model = deepcopy(request.get("payload_model"))
        if self.payload_model is not None:
            validate_payload_model(self.payload_model, config["kinematics"], request["world"])
        self.tool_frame = request.get("tool_frame", "panda_hand_tcp")
        config["kinematics"]["tool_frames"] = [self.tool_frame]
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cache = {k: max(len(request["world"].get(k, {})), 1) for k in ("cuboid", "mesh")}
        cfg = MotionPlannerCfg.create(robot={"robot_cfg": config}, scene_model=request["world"] or None,
            collision_cache=cache, device_cfg=self.device, random_seed=seed,
            interpolation_dt=positive(request.get("interpolation_dt", .05), "interpolation dt"),
            use_cuda_graph=True, self_collision_check=True,
            position_tolerance=.0005, orientation_tolerance=.001,
            graph_planner_algorithm="rrt", enable_trajopt=True, always_run_trajopt=True, enable_timeopt=True)
        self.planner = MotionPlanner(cfg)
        self.native_limits = model_limit_receipt(self.planner.kinematics.config.get_joint_limits(), self.velocity, self.acceleration)
        self.cached_params = {}
        for model in self.models():
            params = model.config.kinematics_config
            cached = copy(params)  # clone() rescales limits in this cuRobo version
            cached.fixed_transforms = params.fixed_transforms.clone()
            cached.lock_jointstate = params.lock_jointstate.clone()
            cached.cspace = copy(params.cspace)
            cached.cspace.default_joint_position = params.cspace.default_joint_position.clone()
            for name, joint in self.fingers.items():
                idx = params.link_name_to_idx_map[joint["child"]]
                actual = self.device.to_device(finger_transform(joint, robot["kinematics"]["lock_joints"][name]))
                if int(params.joint_map[idx]) != -1 or int(params.joint_map_type[idx]) != -1:
                    raise ValueError("Measured finger is not a native locked transform")
                if float(torch.max(torch.abs(params.fixed_transforms[idx]-actual)).item()) > 2e-6:
                    raise RuntimeError("Cached prismatic transform differs from native initialization")
            self.cached_params[id(params)] = cached
        receipt = self.update(request)
        warm_begin = time.perf_counter()
        warm = bool(self.planner.warmup(enable_graph=True, num_warmup_iterations=1))
        torch.cuda.synchronize()
        return {**receipt, "configured": True, "curobo_version": str(curobo.__version__),
                "native_model_limits": self.native_limits, "warmup_success": warm,
                "warmup_s": time.perf_counter()-warm_begin, "wall_s": time.perf_counter()-begin,
                "use_cuda_graph": True, "payload_inertia_applied_to_timeopt": False}

    def update(self, request: dict[str, Any]) -> dict[str, Any]:
        self.ready = False
        begin = time.perf_counter()
        robot, world = request["robot_cfg"], request["world"]
        if robot_topology(robot) != self.robot_structure or world_topology(world) != self.world_structure:
            raise ValueError("Robot/world geometry or profile changed; construct a new session")
        state = deepcopy(request.get("payload_state", empty_payload_state()))
        layers, masks, disabled = collision_policy(self.payload_model, state, world,
                                                   robot["kinematics"]["collision_link_names"])
        kin, defaults = robot["kinematics"], robot["kinematics"]["cspace"]
        locks = kin["lock_joints"]
        vector([locks[n] for n in FINGER_NAMES], 2, "measured finger locks")
        vector(defaults["default_joint_position"], len(defaults["joint_names"]), "measured default joints")
        if state["attachment"] is not None:
            measured = [defaults["default_joint_position"][defaults["joint_names"].index(n)] for n in ARM_NAMES]
            if any(abs(a-b) > 1e-6 for a, b in zip(measured, state["attachment"]["positions"])):
                raise ValueError("Attachment and measured arm positions differ")
        for model in self.models():
            params = model.config.kinematics_config
            cached = self.cached_params[id(params)]
            for name, joint in self.fingers.items():
                idx = params.link_name_to_idx_map[joint["child"]]
                cached.fixed_transforms[idx].copy_(self.device.to_device(finger_transform(joint, locks[name])))
            cached.lock_jointstate.position.copy_(self.device.to_device(
                [locks[n] for n in cached.lock_jointstate.joint_names]).reshape(cached.lock_jointstate.position.shape))
            cached.cspace.default_joint_position.copy_(self.device.to_device([
                defaults["default_joint_position"][defaults["joint_names"].index(n)]
                for n in cached.cspace.joint_names]).reshape(cached.cspace.default_joint_position.shape))
            model.update_kinematics_config(cached)
        checker = self.planner.scene_collision_checker
        for shapes in world.values():
            for name, shape in shapes.items():
                checker.update_obstacle_pose(name, self.Pose.from_list(shape["pose"], device_cfg=self.device))
                checker.data.set_obstacle_layers(name, layers[name])
                checker.enable_obstacle(name, enable=name not in disabled)
        if self.payload_model is not None:
            manager = self.planner.attachment_manager
            attachment = state["attachment"]
            if attachment is None:
                manager.detach(link_name=PAYLOAD_LINK)
                for model in self.models():
                    model.config.kinematics_config.reset_link_spheres(PAYLOAD_LINK)
            else:
                q = self.JointState.from_position(self.device.to_device([attachment["positions"]]), joint_names=ARM_NAMES)
                manager.update(self.device.to_device(attachment["spheres"]), q, link_name=PAYLOAD_LINK,
                    world_objects_pose_offset=self.Pose.from_list(attachment["object_pose_base_wxyz"], device_cfg=self.device))
                spheres = manager.kinematics_params.get_link_spheres(PAYLOAD_LINK).clone()
                for model in self.models():
                    model.config.kinematics_config.update_link_spheres(PAYLOAD_LINK, spheres)
        for model in self.models():
            params = model.config.kinematics_config
            for link, mask in masks.items():
                params.set_link_collision_mask(link, mask)
        if self.planner.graph_planner is not None:
            self.planner.graph_planner.reset_buffer()
        self.torch.cuda.synchronize()
        # Small actual readback verifies both independent locks without full tensor archives.
        readback = []
        for model in self.models():
            locked = model.config.kinematics_config.lock_jointstate
            values = clean(locked.position)
            while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
                values = values[0]
            if any(abs(values[list(locked.joint_names).index(n)]-locks[n]) > 1e-7 for n in FINGER_NAMES):
                raise RuntimeError("Native model failed to update both measured finger locks")
            readback.append(dict(zip(locked.joint_names, values)))
        self.snapshot_id += 1
        self.ready = True
        return {"updated": True, "snapshot_id": self.snapshot_id, "joint_names": ARM_NAMES,
                "locked_fingers": readback, "disabled_own_target_shapes": sorted(disabled),
                "collision_layers": layers, "link_collision_masks": masks,
                "wall_s": time.perf_counter()-begin}

    def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.ready or request.get("snapshot_id") != self.snapshot_id:
            raise RuntimeError("Planning requires the current successful update's snapshot_id")
        begin = time.perf_counter()
        positions = vector(request["start"], 7, "start")
        target = pose(request["goal_pose"], "goal pose")
        attempts = request.get("attempts", 2)
        if type(attempts) is not int or not 1 <= attempts <= 8:
            raise ValueError("attempts must be an integer from 1 to 8")
        start = self.JointState.from_position(self.device.to_device([positions]), joint_names=ARM_NAMES)
        goal = self.GoalToolPose.from_poses({self.tool_frame: self.Pose.from_list(target, device_cfg=self.device)},
                                          ordered_tool_frames=[self.tool_frame], num_goalset=1)
        self.torch.cuda.synchronize()
        solve_begin = time.perf_counter()
        native = self.planner.plan_pose(goal, start, max_attempts=attempts, enable_graph_attempt=0)
        self.torch.cuda.synchronize()
        success = native is not None and bool(single(native.success))
        result = {"success": success, "status": "SUCCESS" if success else "NO_SOLUTION",
                  "native_status": clean(getattr(native, "status", None)), "joint_names": ARM_NAMES,
                  "snapshot_id": self.snapshot_id, "solve_wall_s": time.perf_counter()-solve_begin,
                  "attempts": int(self.planner.last_planning_attempts),
                  "native_result_is_none": native is None,
                  "graph_calls": int(self.planner.last_graph_calls),
                  "graph_success": bool(self.planner.last_graph_success),
                  "graph_time_s": float(self.planner.last_graph_time)}
        if native is not None:
            for name in ("solve_time", "total_time", "trajectory_source", "position_error", "rotation_error", "feasible"):
                result[name] = clean(getattr(native, name, None))
        if success:
            result.update(trajectory_fields(native.get_interpolated_plan()))
        else:
            result.update(trajectory_fields(None))
        result["wall_s"] = time.perf_counter()-begin
        return result

    def shutdown(self) -> None:
        self.ready = False
        if self.planner is not None:
            self.planner.destroy()
            self.planner = None
            self.torch.cuda.synchronize()


class Worker:
    def __init__(self, session_factory: Any = Session) -> None:
        self.sessions: dict[str, Any] = {}
        self.session_factory = session_factory

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("Request must be an object")
        if request.get("op") == "shutdown":
            self.shutdown()
            return {"shutdown": True}
        op, name = request.get("op"), request.get("session_id")
        if op not in ("configure", "update", "plan_pose", "release"):
            raise ValueError("Unknown operation")
        if not isinstance(name, str) or not name:
            raise ValueError("session_id must be a nonempty string")
        if op == "configure":
            if name in self.sessions:
                raise RuntimeError("Session is already configured")
            if len(self.sessions) >= 4:
                raise RuntimeError("Worker supports two arms with two fixed profiles each")
            session = self.session_factory()
            try:
                result = session.configure(request)
            except BaseException:
                try:
                    session.shutdown()
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                raise
            self.sessions[name] = session
            return {**result, "session_id": name}
        if name not in self.sessions:
            raise RuntimeError("configure must succeed before update or planning")
        if op == "release":
            session = self.sessions.pop(name)
            session.shutdown()
            return {"released": True, "session_id": name}
        try:
            result = getattr(self.sessions[name], op)(request)
        except BaseException:
            if op == "update":
                # A partial paired refresh must never leave another old profile
                # available against a different physical snapshot.
                for session in self.sessions.values():
                    session.ready = False
            raise
        return {**result, "session_id": name}

    def shutdown(self) -> None:
        errors = []
        for session in self.sessions.values():
            try:
                session.shutdown()
            except Exception as error:
                errors.append(str(error))
        self.sessions.clear()
        if errors:
            raise RuntimeError("; ".join(errors))


def serve(source: TextIO, sink: TextIO, worker: Worker) -> None:
    for line in source:
        request: Any = None
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = worker.dispatch(request)
            response = {"id": request.get("id"), "ok": True, "result": clean(result)}
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"id": request.get("id") if isinstance(request, dict) else None, "ok": False,
                        "error": {"type": type(error).__name__, "message": str(error)}}
        sink.write(json.dumps(response, allow_nan=False)+"\n")
        sink.flush()
        if isinstance(request, dict) and request.get("op") == "shutdown":
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Serve JSON lines (the default)")
    parser.parse_args()
    # Capture both Python and native/vendor stdout on stderr. Keep a duplicate
    # of the original stdout exclusively for protocol responses.
    sink = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = Worker()
    try:
        serve(sys.stdin, sink, worker)
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            worker.shutdown()
        sink.close()


if __name__ == "__main__":
    main()
