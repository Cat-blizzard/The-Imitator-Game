"""Three fixed task payload snapshots, static sources and isolated sessions.

The source actor comes from the existing TaskAdapter and must be an actual
static rigid body. Its exact complete collision geometry and pose are frozen.
Only an already confirmed carried object can receive its temporary departure
exception; the shared exact source checker remains responsible for execution.
No physical pose, collision mask, joint state or environment tick is changed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .runtime_policy import RuntimePolicy, LEGACY_STRICT
from .curobo_backend import ROOT
from .curobo_payload import (CuroboPayloadSessions, FINGER_LINKS, PAYLOAD_LINK,
    SOURCE_TABLE, export_payload_snapshot, self_test as payload_self_test)
from .curobo_sessions import _SessionProcess, _digest, profile_config
from .multitask_adapters import ALIASES, TASK_IDS, get_task_adapter, task_execution_parameters

SOURCES = {TASK_IDS[0]: ("table", "table"), TASK_IDS[1]: ("table", "table"),
           TASK_IDS[2]: ("003_plate_1", "003_plate_2")}
SOURCE_FIELDS = {"task_id", "physical_arm_index", "source_actor_name", "source_obstacle_names",
                 "source_all_obstacle_names", "source_body_type", "source_is_kinematic",
                 "source_linear_velocity_m_s", "source_angular_velocity_rad_s",
                 "source_world_shapes", "source_geometry_sha256"}


def canonical_task(task_id: str) -> str:
    result = ALIASES.get(task_id, task_id)
    if result not in TASK_IDS:
        raise ValueError("Only the three existing task adapters are supported")
    return result


def task_profile_config(robot_cfg: dict[str, Any], task_id: str, profile: str) -> dict[str, Any]:
    """Set exact existing task limits from original URDF, without multiplying scales."""
    task = canonical_task(task_id)
    parameters = task_execution_parameters(task)
    result = profile_config(robot_cfg, profile)
    if profile == "transport":
        return result
    if profile != "approach":
        raise ValueError("Expected approach or transport fixed limit profile")
    velocity, acceleration = parameters["approach_vel_limit"], parameters["approach_acc_limit"]
    kin = result["kinematics"]
    names, cspace = kin["cspace"]["joint_names"], kin["cspace"]
    joints = {node.attrib["name"]: node for node in ET.parse(kin["urdf_path"]).getroot().findall("joint")}
    for name in (f"panda_joint{i}" for i in range(1, 8)):
        index = names.index(name)
        physical = float(joints[name].find("limit").attrib["velocity"])
        if not math.isfinite(physical) or not 0 < velocity <= physical:
            raise ValueError("Task limit exceeds the original physical URDF")
        cspace["velocity_scale"][index] = velocity/physical
        cspace["max_acceleration"][index] = acceleration
    json.dumps(result, allow_nan=False)
    return result


def _source_manifest(snapshot: dict[str, Any], task_id: str, arm_index: int,
                     actor_name: str, body_type: str, *, source_is_kinematic: bool = False,
                     linear_velocity_m_s: Any = (0.,0.,0.),
                     angular_velocity_rad_s: Any = (0.,0.,0.)) -> dict[str, Any]:
    """Pure exact-name/geometry manifest; accepts no arbitrary source actor."""
    task = canonical_task(task_id)
    if type(arm_index) is not int or arm_index not in (0, 1):
        raise ValueError("Canonical physical arm must be 0 or 1")
    if actor_name != SOURCES[task][arm_index] or type(source_is_kinematic) is not bool or not (
            body_type == "PhysxRigidStaticComponent" and not source_is_kinematic
            or body_type == "PhysxRigidDynamicComponent" and source_is_kinematic):
        raise ValueError("Source must be this fixed task arm's actual static or kinematic support actor")
    velocities = [np.asarray(value,dtype=float) for value in (linear_velocity_m_s,angular_velocity_rad_s)]
    if any(value.shape != (3,) or not np.isfinite(value).all() or np.any(np.abs(value)>1e-12) for value in velocities):
        raise ValueError("Fixed source requires measured zero linear and angular velocities")
    all_shapes = {name: {"kind": kind, **shape} for kind, group in snapshot["world"].items() for name, shape in group.items()}
    names = sorted(name for name in all_shapes if name.startswith(f"scene::{actor_name}::shape_"))
    if not names or (task != TASK_IDS[2] and SOURCE_TABLE not in names):
        raise ValueError("Source must retain every exact supported collision shape")
    if set(names) & set(snapshot["payload_model"]["target_obstacle_names"]):
        raise ValueError("Target and source collision shapes must be distinct")
    shapes = {name: all_shapes[name] for name in names}
    return {"task_id": task, "physical_arm_index": arm_index, "source_actor_name": actor_name,
            "source_obstacle_names": names if task == TASK_IDS[2] else [SOURCE_TABLE],
            "source_all_obstacle_names": names, "source_body_type": body_type,
            "source_is_kinematic":source_is_kinematic,
            "source_linear_velocity_m_s":velocities[0].tolist(),"source_angular_velocity_rad_s":velocities[1].tolist(),
            "source_world_shapes": copy.deepcopy(shapes), "source_geometry_sha256": _digest(shapes)}


def _validate_task_snapshot(snapshot: dict[str, Any], task_id: str, arm: int) -> None:
    model, state = snapshot["payload_model"], snapshot["payload_state"]
    expected = _source_manifest(snapshot, task_id, arm, model["source_actor_name"], model["source_body_type"],
        source_is_kinematic=model["source_is_kinematic"],linear_velocity_m_s=model["source_linear_velocity_m_s"],
        angular_velocity_rad_s=model["source_angular_velocity_rad_s"])
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError("Source geometry/pose/task/arm differs from its frozen actual manifest")
    if state["source_support_obstacle_names"] != (model["source_obstacle_names"] if state["source_support_allowed"] else []):
        raise ValueError("Source-support exception differs from exact own source shapes")


def export_task_payload_snapshot(env: Any, arm_index: int, target_actor: Any, *, task_id: str,
                                 source_actor: Any | None = None, contact_allowed: bool = False,
                                 grasp_evidence: dict[str, Any] | None = None,
                                 source_support_allowed: bool = False, sphere_pitch_m: float = .025,
                                 capacity: int | None = None,
                                 policy: RuntimePolicy = LEGACY_STRICT) -> dict[str, Any]:
    """Extend the established actual snapshot with its fixed actual static source.

    ``source_actor`` may be the existing adapter.initial_supports[arm] wrapper
    or exact canonical actor key. Supplied actors must resolve to that same
    native entity. Measured source body type, every shape and its full pose are
    frozen; subsequent session updates reject any source geometry/pose change.
    """
    task = canonical_task(task_id)
    raw = getattr(env, "unwrapped", env)
    adapter = get_task_adapter(task, raw)
    if type(arm_index) is not int or arm_index not in (0, 1):
        raise ValueError("Canonical physical arm must be 0 or 1")
    from .scene_collision import scene_entities
    import sapien.physx as physx
    entities = scene_entities(raw)

    def entity(value: Any) -> Any:
        if isinstance(value, str):
            return entities[value]
        objects = getattr(value, "_objs", None)
        if objects is not None:
            if len(objects) != 1:
                raise ValueError("Source must refer to one actual native entity")
            return objects[0]
        return value

    expected = entity(adapter.initial_supports[arm_index])
    selected = expected if source_actor is None else entity(source_actor)
    if selected != expected:
        raise ValueError("Supplied source differs from the existing task adapter support")
    keys = [key for key, candidate in entities.items() if candidate == selected]
    if len(keys) != 1:
        raise ValueError("Actual source must have exactly one canonical scene key")
    bodies = [component for component in selected.components if isinstance(component, physx.PhysxRigidBaseComponent)]
    if len(bodies) != 1:
        raise ValueError("Task source must have exactly one actual rigid body")
    body=bodies[0]
    if isinstance(body,physx.PhysxRigidDynamicComponent):
        kinematic=bool(body.kinematic)
        linear=np.asarray(body.linear_velocity,dtype=float)
        angular=np.asarray(body.angular_velocity,dtype=float)
        if not kinematic:
            raise ValueError("A dynamic nonkinematic source cannot receive fixed support permission")
    elif isinstance(body,physx.PhysxRigidStaticComponent):
        kinematic=False
        linear=np.zeros(3);angular=np.zeros(3)
    else:
        raise ValueError("Task source must be actual static or measured fixed kinematic rigid body")
    snapshot = export_payload_snapshot(env, arm_index, target_actor,
        contact_allowed=contact_allowed, grasp_evidence=grasp_evidence,
        source_support_allowed=source_support_allowed, sphere_pitch_m=sphere_pitch_m, capacity=capacity,
        policy=policy)
    manifest = _source_manifest(snapshot, task, arm_index, keys[0], type(body).__name__,
        source_is_kinematic=kinematic,linear_velocity_m_s=linear,angular_velocity_rad_s=angular)
    if len(manifest["source_all_obstacle_names"]) != len(body.get_collision_shapes()):
        raise ValueError("Source world coverage omitted an actual native collision shape")
    snapshot["payload_model"].update(manifest)
    snapshot["payload_state"]["source_support_obstacle_names"] = manifest["source_obstacle_names"] if source_support_allowed else []
    snapshot["coverage"]["payload"]["source"] = {**manifest,
        "actual_static_body_checked": True, "all_actual_source_shapes_covered": True,
        "external_exact_source_checker_required": source_support_allowed}
    _validate_task_snapshot(snapshot, task, arm_index)
    json.dumps(snapshot, allow_nan=False)
    return snapshot


def audit_task_native_state(snapshot: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Verify actual final native layers, enable flags and every robot-link mask."""
    model, state = snapshot["payload_model"], snapshot["payload_state"]
    _validate_task_snapshot(snapshot, model["task_id"], model["physical_arm_index"])
    native = receipt.get("native_update", receipt)
    payload = native["payload"]
    sources, targets = set(model["source_obstacle_names"]), set(model["target_obstacle_names"])
    expected_world = {name for group in snapshot["world"].values() for name in group}
    if set(payload["world"]) != expected_world:
        raise AssertionError("Native scene must retain every complete world obstacle")
    attached = state["attachment"] is not None
    for name, observed in payload["world"].items():
        if (observed["layers"] != (2 if name in targets else 4 if name in sources else 1)
                or observed["enabled"] != (not (attached and name in targets))):
            raise AssertionError(f"Wrong exact task world state: {name}")
    policy = payload["source_policy"]
    if set(policy["source_obstacle_names"]) != sources or policy["source_geometry_sha256"] != model["source_geometry_sha256"]:
        raise AssertionError("Native source policy differs from the frozen actual source")
    links = snapshot["robot_cfg"]["kinematics"]["collision_link_names"]
    for masks in policy["robot_link_collision_masks"].values():
        if set(masks) != set(links):
            raise AssertionError("Native source mask readback omitted robot links")
        for link, values in masks.items():
            expected = (3 if state["source_support_allowed"] else 7) if link == PAYLOAD_LINK else (
                5 if link in FINGER_LINKS and state["contact_allowed"] else 7)
            if values != [expected]:
                raise AssertionError(f"Native collision mask differs for {link}")
    return {"status": "passed", "source_actor_name": model["source_actor_name"],
            "source_obstacle_names": sorted(sources), "all_world_obstacles_checked": len(expected_world),
            "own_payload_source_exception": state["source_support_allowed"],
            "all_robot_links_check_source": True, "peer_source_and_table_retained": True}


class _TaskProcess(_SessionProcess):
    def __init__(self, gpu_uuid: str, log_path: Path, *, task_id: str, timeout_s: float = 180., python: Path | None = None) -> None:
        task = canonical_task(task_id)
        super().__init__(gpu_uuid, log_path, timeout_s=timeout_s, python=python,
            _worker="curobo_tig_task_payload_worker.py",
            _worker_args=("--serve", "--task-id", task))


class SelectedWorkerProcess(_SessionProcess):
    """The same task protocol with a caller-selected explicit worker source."""
    def __init__(self, gpu_uuid: str, log_path: Path, worker_script: Path, *,
                 task_id: str, timeout_s: float = 180., python: Path | None = None) -> None:
        task = canonical_task(task_id)
        super().__init__(gpu_uuid, log_path, timeout_s=timeout_s, python=python,
            _worker=worker_script.resolve(), _worker_args=("--serve", "--task-id", task))


class CuroboTaskPayloadSessions(CuroboPayloadSessions):
    def __init__(self, gpu_uuid: str, log_path: Path, *, task_id: str,
                 timeout_s: float = 180., python: Path | None = None, transport: Any | None = None) -> None:
        self.task_id = canonical_task(task_id)
        super().__init__(gpu_uuid, log_path, timeout_s=timeout_s, python=python,
            transport=transport if transport is not None else _TaskProcess(
                gpu_uuid, log_path, task_id=self.task_id, timeout_s=timeout_s, python=python))

    def _payload_pair(self, snapshots: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        pair = self._pair(snapshots)
        views = copy.deepcopy(pair)
        for arm, snapshot in enumerate(pair):
            _validate_task_snapshot(snapshot, self.task_id, arm)
            model = views[arm]["payload_model"]
            for key in SOURCE_FIELDS:
                model.pop(key, None)
            state = views[arm]["payload_state"]
            state["source_support_obstacle_names"] = [SOURCE_TABLE] if state["source_support_allowed"] else []
        CuroboPayloadSessions._payload_pair(views)
        return pair

    def initialize_pair(self, snapshots: tuple[dict[str, Any], dict[str, Any]], *, seed: int = 401,
                        interpolation_dt: float = .02, warmup: bool = True) -> list[dict[str, Any]]:
        if self._initialization_attempted:
            raise RuntimeError("Initialize fixed task sessions once per process")
        pair = self._payload_pair(snapshots)
        if any(snapshot["payload_state"]["attachment"] is not None for snapshot in pair):
            raise ValueError("Initialize reserved planners before actual grasping")
        if type(seed) is not int or seed < 0 or not math.isfinite(interpolation_dt) or interpolation_dt <= 0 or warmup is not True:
            raise ValueError("Task sessions require a valid seed, positive dt and one warmup")
        configs = {(arm, profile): task_profile_config(snapshot["robot_cfg"], self.task_id, profile)
                   for profile in ("approach", "transport") for arm, snapshot in enumerate(pair)}
        self._initialization_attempted = True
        self._models = tuple(copy.deepcopy(snapshot["payload_model"]) for snapshot in pair)
        results = []
        for profile in ("approach", "transport"):
            for arm, snapshot in enumerate(pair):
                session = self.session_id(arm, profile)
                self._tokens[session] = None
                results.append(self._rpc.request("init_session", session_id=session, robot_cfg=configs[arm, profile],
                    world=snapshot["world"], seed=seed, interpolation_dt=float(interpolation_dt), warmup=warmup,
                    payload_model=snapshot["payload_model"]))
        self._initialized = True
        return results

    def update_pair(self, snapshots: tuple[dict[str, Any], dict[str, Any]], *, profile: str, snapshot_id: str) -> list[dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Initialize every fixed task session before updating")
        sessions = [self.session_id(arm, profile) for arm in (0, 1)]
        self._current_snapshot_id = None
        for session in sessions:
            self._tokens[session] = None
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("A nonempty immutable physical snapshot ID is required")
        pair = self._payload_pair(snapshots)
        if self._models is None or any(snapshot["payload_model"] != self._models[arm] for arm, snapshot in enumerate(pair)):
            raise ValueError("Persistent task target/source geometry or inertial model changed")
        digest = _digest(pair)
        if snapshot_id in self._snapshot_digests and self._snapshot_digests[snapshot_id] != digest:
            raise ValueError("Snapshot ID cannot identify different task/source physical state")
        configs = [task_profile_config(snapshot["robot_cfg"], self.task_id, profile) for snapshot in pair]
        results = []
        for arm, snapshot in enumerate(pair):
            results.append(self._rpc.request("update_session", session_id=sessions[arm], robot_cfg=configs[arm],
                world=snapshot["world"], snapshot_id=snapshot_id, payload_state=snapshot["payload_state"]))
        self._snapshot_digests[snapshot_id] = digest
        for session in sessions:
            self._tokens[session] = snapshot_id
        self._current_snapshot_id = snapshot_id
        return results


def self_test() -> None:
    payload_self_test()
    from .curobo_scene import ARM_NAMES, FINGER_NAMES
    from .curobo_payload import reserve_payload_config
    from infrastructure.curobo_tig_task_payload_worker import TaskSessionManager, dispatch_task

    class FakeBackend:
        def __init__(self) -> None:
            self.planner: Any = None
            self.lock_joints: dict[str, float] = {}

        def configure(self, request: dict[str, Any]) -> dict[str, Any]:
            self.planner = object()
            self.lock_joints = copy.deepcopy(request["robot_cfg"]["kinematics"]["lock_joints"])
            return {"warmup_success": True}

        def update(self, robot: dict[str, Any], world: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
            if world["cuboid"]["scene::foreign::shape_0"]["pose"][0] == 999.:
                raise RuntimeError("synthetic partial right-arm update failure")
            self.lock_joints = copy.deepcopy(robot["kinematics"]["lock_joints"])
            return {"state": copy.deepcopy(state)}

        def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"success": True}

        def shutdown(self) -> dict[str, Any]:
            self.planner = None
            return {"shutdown": True}

    class FakeRPC:
        def __init__(self, task_id: str) -> None:
            self.manager = TaskSessionManager(task_id, backend_factory=FakeBackend)
            self.receipts: list[dict[str, Any]] = []

        def request(self, op: str, **payload: Any) -> dict[str, Any]:
            request = {"id": len(self.receipts), "op": op, **payload}
            self.receipts.append(copy.deepcopy(request))
            return dispatch_task(request, self.manager)["result"]

        def close(self) -> None:
            self.manager.shutdown({})

    protocol_cases = 0
    with tempfile.TemporaryDirectory(prefix="tig-task-client-cpu-") as directory:
        path = Path(directory)/"panda.urdf"
        names = list(ARM_NAMES+FINGER_NAMES)
        path.write_text('<robot name="test">'+''.join(f'<joint name="{name}"><limit velocity="2.5"/></joint>' for name in names)+'</robot>')
        cfg = {"kinematics": {"urdf_path": str(path), "tool_frames": ["panda_hand_tcp"],
            "collision_link_names": list(FINGER_LINKS),
            "collision_spheres": {name: [{"center": [0.,0.,0.],"radius":.005}] for name in FINGER_LINKS},
            "self_collision_ignore": {},
            "lock_joints": dict(zip(FINGER_NAMES, [.03,.031])),
            "cspace": {"joint_names": names, "default_joint_position": [0.]*7+[.03,.031],
                       "velocity_scale": [.2]*9, "max_acceleration": .9}}, "load_dynamics": False}
        for task in TASK_IDS:
            result = task_profile_config(cfg, task, "approach")
            parameters = task_execution_parameters(task)
            assert np.allclose(np.array(result["kinematics"]["cspace"]["velocity_scale"][:7])*2.5, parameters["approach_vel_limit"])
            assert result == task_profile_config(result, task, "approach")
            pair = []
            identity = [0.,0.,0.,1.,0.,0.,0.]
            shared_world = {"cuboid": {name: {"pose": identity.copy(), "dims": [.1]*3}
                for name in {SOURCE_TABLE, "scene::foreign::shape_0", "scene::target_0::shape_0", "scene::target_1::shape_0",
                             *[f"scene::{source}::shape_0" for source in SOURCES[task]]}}, "mesh": {}}
            for arm in (0, 1):
                actor = SOURCES[task][arm]
                source = f"scene::{actor}::shape_0"
                target = f"scene::target_{arm}::shape_0"
                identity = [0.,0.,0.,1.,0.,0.,0.]
                snapshot = {"world": {"cuboid": {name: {"pose": identity, "dims": [.1]*3}
                    for name in set((source,target,SOURCE_TABLE,"scene::foreign::shape_0"))}, "mesh": {}},
                    "payload_model": {"target_obstacle_names": [target]},
                    "payload_state": {"source_support_allowed": False,"source_support_obstacle_names":[]}}
                snapshot["payload_model"].update(_source_manifest(snapshot, task, arm, actor, "PhysxRigidStaticComponent"))
                _validate_task_snapshot(snapshot, task, arm)
                changed = copy.deepcopy(snapshot)
                changed["world"]["cuboid"][source]["pose"] = [.01,0.,0.,1.,0.,0.,0.]
                try:
                    _validate_task_snapshot(changed,task,arm)
                except ValueError:
                    pass
                else:
                    raise AssertionError("changed static source pose accepted")
                try:
                    _source_manifest(snapshot,task,arm,"foreign","PhysxRigidStaticComponent")
                except ValueError:
                    pass
                else:
                    raise AssertionError("foreign support actor accepted")
                protocol_snapshot = {"robot_cfg": reserve_payload_config(cfg, 2), "world": copy.deepcopy(shared_world),
                    "payload_model": {"link_name": PAYLOAD_LINK, "capacity": 2, "target_actor_name": f"target_{arm}",
                        "target_obstacle_names": [target], "finger_links": list(FINGER_LINKS), "sphere_count": 1,
                        "spheres_actor_local": [[0.,0.,0.,.005]]},
                    "payload_state": {"contact_allowed":False,"source_support_allowed":False,
                                      "source_support_obstacle_names":[],"attachment":None}}
                protocol_snapshot["payload_model"].update(_source_manifest(protocol_snapshot,task,arm,actor,"PhysxRigidStaticComponent"))
                pair.append(protocol_snapshot)
            rpc = FakeRPC(task)
            client = CuroboTaskPayloadSessions("GPU-synthetic",Path(directory)/"unused.log",task_id=task,transport=rpc)
            initialized = client.initialize_pair(tuple(pair))
            assert len(initialized) == 4 and len({row["planner_id"] for row in initialized}) == 4
            client.update_pair(tuple(pair),profile="approach",snapshot_id="initial")
            assert client.plan_pose(arm=0,profile="approach",snapshot_id="initial",start=[0.]*7,goal_pose=identity)["success"]
            broken = copy.deepcopy(pair)
            broken[1]["world"]["cuboid"]["scene::foreign::shape_0"]["pose"][0] = 999.
            try:
                client.update_pair(tuple(broken),profile="approach",snapshot_id="partial-failure")
            except RuntimeError:
                pass
            else:
                raise AssertionError("partial paired update failure disappeared")
            for arm in (0,1):
                try:
                    client.plan_pose(arm=arm,profile="approach",snapshot_id="initial",start=[0.]*7,goal_pose=identity)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("stale paired client remained authorized after partial update")
            assert not rpc.manager.sessions["right:approach"].valid
            client.update_pair(tuple(pair),profile="approach",snapshot_id="recovered")
            assert client.plan_pose(arm=1,profile="approach",snapshot_id="recovered",start=[0.]*7,goal_pose=identity)["success"]
            client.close()
            assert rpc.manager.closed
            protocol_cases += 1
    print(json.dumps({"task_client_self_test":"PASS","task_arm_shape":[3,2],"client_worker_protocol_cases":protocol_cases,
        "checks":["exact_task_profiles","repeated_profile_no_double_scaling","complete_static_source_manifest",
                  "static_pose_change_rejected","foreign_source_rejected","legacy_full_payload_self_test",
                  "four_resident_task_planners","paired_failure_invalidates_both_client_tokens","stale_plan_rejection_and_recovery"]}))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true",required=True)
    parser.parse_args()
    self_test()


if __name__ == "__main__":
    main()
