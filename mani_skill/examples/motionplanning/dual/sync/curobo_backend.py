"""Optional cuRobo candidate generation for the shared Panda executor.

The worker runs in a separately selected Python environment. This client only
reads physics: joint trajectories are radians/seconds in controller order,
poses are XYZ/WXYZ metres, and payload spheres are actor-local XYZ/radius.
The existing combined checker and executor own candidate acceptance and physics.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .planning_endpoint import POSITION_TOLERANCE_M, ROTATION_TOLERANCE_RAD, pose_error
from .trajectory_execution import as_numpy, bindings
from .trajectory_schedule import ArmTrajectory, build_hold_schedule


ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")
SOURCE_TABLE = "scene::table::shape_0"


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array


class CuroboProcess:
    """One owned JSONL worker; logs and optional CUDA dependencies stay isolated."""

    def __init__(self, *, python: str | None = None, device: str = "cuda:0",
                 cache_dir: str | None = None) -> None:
        interpreter = os.path.abspath(os.path.expanduser(
            python or os.environ.get("TIG_CUROBO_PYTHON", sys.executable)))
        if not Path(interpreter).is_file():
            raise FileNotFoundError("cuRobo Python was not found; set --curobo-python")
        if not isinstance(device, str) or not device.startswith("cuda:"):
            raise ValueError("cuRobo device must be a CUDA device such as cuda:0")
        self.device, self.sequence, self.failed = device, 0, False
        self.receipts: list[dict[str, Any]] = []
        self.log = tempfile.TemporaryFile(mode="w+b")
        self._read_buffer = bytearray()
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        if cache_dir is not None:
            root = Path(cache_dir).expanduser().absolute()
            for key, directory in (("WARP_CACHE_PATH", "warp"),
                                   ("TORCH_EXTENSIONS_DIR", "torch_extensions")):
                destination = root / directory
                destination.mkdir(parents=True, exist_ok=True)
                environment[key] = str(destination)
        try:
            self.process = subprocess.Popen(
                [interpreter, "-B", "-u", str(Path(__file__).with_name("curobo_worker.py")), "--serve"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding="utf-8", env=environment, bufsize=1)
        except Exception:
            self.log.close()
            raise

    def _diagnostic(self) -> str:
        self.log.flush()
        self.log.seek(0, os.SEEK_END)
        self.log.seek(max(0, self.log.tell() - 6000))
        return self.log.read().decode("utf-8", errors="replace")

    def request(self, op: str, *, timeout_s: float | None = None, **payload: Any) -> dict[str, Any]:
        if self.failed:
            raise RuntimeError("cuRobo worker is unavailable after a failed request")
        self.sequence += 1
        request = {"id": self.sequence, "op": op, **payload}
        encoded = json.dumps(request, allow_nan=False)
        timeout = timeout_s if timeout_s is not None else (1200. if op == "configure" else 180.)
        started = time.monotonic()
        assert self.process.stdin is not None and self.process.stdout is not None
        try:
            self.process.stdin.write(encoded + "\n")
            self.process.stdin.flush()
            deadline = started + timeout
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while b"\n" not in self._read_buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError(f"cuRobo {op} exceeded {timeout:g} seconds")
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError(f"cuRobo worker exited during {op}: {self.process.poll()}")
                    self._read_buffer.extend(chunk)
            line, _, rest = self._read_buffer.partition(b"\n")
            self._read_buffer[:] = rest
            response = json.loads(line)
            if not isinstance(response, dict) or response.get("id") != self.sequence:
                raise RuntimeError("cuRobo response does not match its request")
            if response.get("ok") is not True:
                raise RuntimeError(f"cuRobo {op}: {response.get('error', 'unknown error')}")
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("cuRobo response result must be an object")
            self.receipts.append({"op": op, "session_id": payload.get("session_id"),
                "wall_seconds": time.monotonic() - started, "ok": True})
            return result
        except Exception as exc:
            self.failed = True
            self.receipts.append({"op": op, "session_id": payload.get("session_id"),
                "wall_seconds": time.monotonic() - started, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"})
            raise RuntimeError(f"{exc}\n{self._diagnostic()}") from exc

    def close(self) -> None:
        if self.log.closed:
            return
        try:
            if self.process.poll() is None and not self.failed:
                try:
                    self.request("shutdown", timeout_s=10.)
                except Exception:
                    pass
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10.)
        finally:
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    pipe.close()
            self.log.close()


def _configuration_key(request: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed storage/solver identity; measured poses and payload fits may vary."""
    from .curobo_worker import robot_topology, world_topology
    model = request.get("payload_model")
    # Target identity and its fitted sphere values are validated dynamic state:
    # the same fixed TCP storage can serve either Panda's current target.
    payload = None if model is None else {key: deepcopy(model.get(key)) for key in (
        "link_name", "capacity")}
    return {"robot": robot_topology(request["robot_cfg"]),
        "world": world_topology(request["world"]), "payload": payload,
        **{key: request.get(key) for key in ("velocity_limit", "acceleration_limit",
            "seed", "device", "interpolation_dt", "tool_frame")}}


class CuroboSession:
    """Own two fixed-profile planners, serving the Panda arms sequentially.

    The first episode fixes the planner seed unless explicitly supplied. Scene,
    fingers and attachments are measured afresh each episode; geometry/storage
    incompatibility rebuilds the affected profile instead of trimming geometry.
    """

    def __init__(self, options: Mapping[str, Any] | None = None,
                 planner_seed: int | None = None) -> None:
        if planner_seed is not None and (type(planner_seed) is not int or not 0 <= planner_seed < 2**31):
            raise ValueError("planner_seed must be an integer in [0,2**31)")
        self.options = None if options is None else dict(options)
        self.planner_seed = planner_seed
        self.rpc: CuroboProcess | None = None
        self.configured: dict[str, dict[str, Any]] = {}
        self.active = False
        self.episode_count = 0

    def connect(self, options: Mapping[str, Any], episode_seed: int) -> CuroboProcess:
        if self.active:
            raise RuntimeError("A cuRobo session can run only one episode at a time")
        if self.options is None:
            self.options = dict(options)
        elif self.options != dict(options):
            raise ValueError("Shared cuRobo session options changed")
        if self.planner_seed is None:
            self.planner_seed = episode_seed
        if self.rpc is not None and self.rpc.failed:
            self.close()
        if self.rpc is None:
            self.rpc = CuroboProcess(**self.options)
        self.active = True
        return self.rpc

    def prepare(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        """Keep an existing buffer size when the new complete fit fits inside."""
        previous = self.configured.get(name)
        model = request.get("payload_model")
        if previous is None or model is None:
            return request
        capacity = (previous["request"].get("payload_model") or {}).get("capacity", 0)
        if capacity <= model["capacity"]:
            return request
        from .curobo_scene import resize_payload_config
        return {**request, "payload_model": {**model, "capacity": capacity},
            "robot_cfg": resize_payload_config(request["robot_cfg"], capacity)}

    def begin_episode(self, requests: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
        assert self.rpc is not None
        self.episode_count += 1
        self.rpc.request("begin_episode")
        receipts = []
        try:
            for name in list(self.configured):
                request = self.prepare(name, requests[name])
                if _configuration_key(request) != self.configured[name]["key"]:
                    self.rpc.request("release", session_id=name)
                    del self.configured[name]
                    continue
                # Refresh every retained profile, including currently inactive
                # transport models: no previous episode's attachment survives.
                result = self.rpc.request("update", session_id=name,
                    **{key: request[key] for key in ("robot_cfg", "world", "payload_model", "payload_state")})
                receipts.append({"session_id": name, "configuration_reused": True,
                                 "episode_reset": True, **result})
        except BaseException:
            self.close()
            raise
        return receipts

    def acquire(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        assert self.rpc is not None
        request = self.prepare(name, request)
        key = _configuration_key(request)
        if name in self.configured:
            if key == self.configured[name]["key"]:
                return {"configured": True, "configuration_reused": True,
                    "native_model_limits": deepcopy(self.configured[name]["receipt"].get("native_model_limits"))}
            self.rpc.request("release", session_id=name)
            del self.configured[name]
        try:
            result = self.rpc.request("configure", session_id=name, **request)
        except BaseException:
            self.close()
            raise
        self.configured[name] = {"key": key, "request": deepcopy(request), "receipt": deepcopy(result)}
        return {**result, "configuration_reused": False}

    def end_episode(self) -> None:
        self.active = False
        if self.rpc is not None and self.rpc.failed:
            self.close()

    def close(self) -> None:
        self.active = False
        try:
            if self.rpc is not None:
                self.rpc.close()
        finally:
            self.rpc = None
            self.configured.clear()


def checked_trajectory(result: Mapping[str, Any], measured_start: Any,
                       joint_names: tuple[str, ...], gripper: float, action_key: str,
                       *, velocity_limit: float, acceleration_limit: float) -> ArmTrajectory:
    """Map a successful named native trajectory to the common seconds clock.

    Native time parameterization is retained. The endpoint FK and the complete
    paired candidate are checked separately; sampled derivatives are not a
    continuous acceleration certificate or a measurement of physical torques.
    """
    if result.get("success") is not True:
        raise ValueError(f"cuRobo did not find a trajectory: {result.get('status')}")
    names = tuple(result.get("joint_names", ()))
    if len(names) != 7 or len(set(names)) != 7 or set(names) != set(joint_names):
        raise ValueError("cuRobo trajectory joint names differ from the controller")
    times = np.asarray(result.get("times_s"), dtype=float)
    if times.ndim != 1 or len(times) < 2 or times[0] != 0. or np.any(np.diff(times) <= 0):
        raise ValueError("cuRobo trajectory needs increasing seconds starting at zero")
    positions = _finite(result.get("positions"), (len(times), 7), "native positions")
    order = [names.index(name) for name in joint_names]
    positions = positions[:, order]
    if not np.allclose(positions[0], _finite(measured_start, (7,), "measured start"), atol=1e-6, rtol=0):
        raise ValueError("cuRobo trajectory does not start at the measured arm configuration")
    for key, limit in (("velocity", velocity_limit), ("acceleration", acceleration_limit)):
        derivative = _finite(result.get(key), (len(times), 7), f"native {key}")
        if float(np.abs(derivative).max()) > limit + 1e-5:
            raise ValueError(f"cuRobo sampled {key} exceeds the selected phase limit")
    return ArmTrajectory(action_key, joint_names, times, positions, gripper)


class CuroboBackend:
    """Fixed-profile arm planners for the shared executor.

    A standalone episode retires approach before transport (two arm residents).
    A batch retains one planner per profile (two residents), updating the full
    measured snapshot before each independent arm's candidate query.
    """

    def __init__(self, env: Any, solvers: list[Any], task: Any, *, seed: int | None,
                 options: Mapping[str, Any] | None, limits: dict[str, Any],
                 session: CuroboSession | None = None) -> None:
        self.raw, self.solvers, self.task = env.unwrapped, solvers, task
        options = dict(options or {})
        unsupported = set(options) - {"python", "device", "cache_dir"}
        if unsupported:
            raise ValueError(f"Unsupported cuRobo options: {sorted(unsupported)}")
        self.cache_dir = options.get("cache_dir")
        self.arms = bindings(self.raw, tuple(self.raw.agent.agents_dict))
        self.episode_seed, self.limits = int(seed or 0), limits
        self.session = session
        self.rpc = CuroboProcess(**options) if session is None else session.connect(options, self.episode_seed)
        self.seed = self.episode_seed if session is None else session.planner_seed
        self.request_start = len(self.rpc.receipts)
        self.serial = 0
        self.profile: str | None = None
        self.models: list[dict[str, Any]] = []
        self.initialization: list[dict[str, Any]] = []
        self.planning: list[dict[str, Any]] = []

    def _snapshot(self, arm_index: int, *, phase: str, closed: bool,
                  checker: Any) -> dict[str, Any]:
        from .curobo_scene import (_matrix, _pose, _shape_spheres, _entity_shapes, attached_shape_spheres,
            export_curobo_snapshot, reserve_payload_config)
        snapshot = export_curobo_snapshot(self.raw, arm_index, cache_dir=self.cache_dir)
        target = self.task.targets[arm_index]
        if len(self.models) <= arm_index:
            entity = target._objs[0]
            from .scene_collision import scene_entities
            actors = scene_entities(self.raw)
            names = [key for key, value in actors.items() if value == entity]
            if len(names) != 1:
                raise ValueError("Held target needs one canonical scene identity")
            import sapien.physx as physx
            bodies = [c for c in entity.components if isinstance(c, physx.PhysxRigidDynamicComponent)]
            if len(bodies) != 1:
                raise ValueError("Held target must have one dynamic rigid body")
            shapes = list(bodies[0].get_collision_shapes())
            spheres: list[list[float]] = []
            for shape in shapes:
                values, _ = _shape_spheres(shape, .0125, tight=True)
                spheres.extend([*s["center"], s["radius"]] for s in values)
            if not spheres:
                raise ValueError("Held target has no collision geometry")
            capacity = 64 * math.ceil(len(spheres) / 64)
            self.models.append({"link_name": "tig_payload", "capacity": capacity,
                "target_actor_name": names[0], "target_obstacle_names": [
                    f"scene::{names[0]}::shape_{i}" for i in range(len(shapes))],
                "finger_links": list(FINGER_LINKS), "sphere_count": len(spheres),
                "spheres_actor_local": spheres,
                "source_obstacle_names": [SOURCE_TABLE]})
        model = self.models[arm_index]
        if closed:
            agent = self.raw.agent.agents[arm_index]
            if not checker.attached or not bool(as_numpy(agent.is_grasping(target)).item()):
                raise ValueError("cuRobo payload attachment requires a current confirmed physical grasp")
        if closed and not model.get("cover_frozen_at_grasp", False):
            tcp_to_object = _pose(np.linalg.inv(_matrix(agent.tcp.pose)) @ _matrix(target.pose))
            shapes = _entity_shapes(target._objs[0])
            spheres, proofs = [], []
            for shape in shapes:
                values, proof = attached_shape_spheres(shape, .0125, tcp_to_object)
                spheres.extend([*s["center"], s["radius"]] for s in values)
                proofs.append(proof)
            if not spheres or len(shapes) != len(model["target_obstacle_names"]):
                raise ValueError("Held target geometry changed before payload refinement")
            model = deepcopy(model)
            model.update(capacity=64 * math.ceil(len(spheres) / 64), sphere_count=len(spheres),
                spheres_actor_local=spheres, cover_frozen_at_grasp=True,
                cover={"version": "full_solid_tcp_negative_z_2mm_v1",
                    "tcp_to_object_wxyz": tcp_to_object, "shape_proofs": proofs})
            self.models[arm_index] = model
        snapshot["robot_cfg"] = reserve_payload_config(snapshot["robot_cfg"], model["capacity"])
        snapshot["payload_model"] = deepcopy(model)
        # Candidate generation may leave release contact. The common checker
        # permits only existing, diminishing intersections and rejects reentry.
        state: dict[str, Any] = {"contact_allowed": phase in ("descend", "retreat") or closed,
            "attachment": None, "source_support_allowed": False,
            "source_support_obstacle_names": []}
        if closed:
            base = _matrix(agent.robot.pose)
            state["attachment"] = {"spheres": model["spheres_actor_local"],
                "object_pose_base_wxyz": _pose(np.linalg.inv(base) @ _matrix(target.pose)),
                "positions": snapshot["q_start"], "obstacle_names": model["target_obstacle_names"],
                "tcp_to_object_wxyz": _pose(np.linalg.inv(_matrix(agent.tcp.pose)) @ _matrix(target.pose))}
            from .physical_observation import measured_source_departure
            state["source_support_allowed"] = bool(
                phase == "lift_and_hold" and not measured_source_departure(checker)[arm_index])
            state["source_support_obstacle_names"] = [SOURCE_TABLE] if state["source_support_allowed"] else []
        snapshot["payload_state"] = state
        return snapshot

    def initialize(self, checker: Any) -> None:
        snapshots = [self._snapshot(i, phase="initialization", closed=False, checker=checker)
                     for i in (0, 1)]
        if self.session is not None:
            requests = {profile: self._request(profile, self._shared_snapshots(profile, snapshots)[0])
                for profile in ("approach", "global")}
            self.initialization.extend(self.session.begin_episode(requests))
        self._configure("approach", snapshots)

    def _shared_snapshots(self, profile: str, snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reserve the complete two-arm fits before allocating one profile."""
        from .curobo_scene import resize_payload_config
        capacity = max(snapshot["payload_model"]["capacity"] for snapshot in snapshots)
        previous = self.session.configured.get(profile)
        if previous is not None:
            capacity = max(capacity, previous["request"]["payload_model"]["capacity"])
        return [snapshot if snapshot["payload_model"]["capacity"] == capacity else {
            **snapshot, "payload_model": {**snapshot["payload_model"], "capacity": capacity},
            "robot_cfg": resize_payload_config(snapshot["robot_cfg"], capacity)}
            for snapshot in snapshots]

    def _request(self, profile: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        from .phase_kinematics import compact_phase_limits
        phase = "pregrasp" if profile == "approach" else "carry_lower"
        limits = compact_phase_limits(self.limits, phase)
        return {key: snapshot[key] for key in ("robot_cfg", "world", "payload_model", "payload_state")} | {
            "velocity_limit": limits["solver_velocity_limit_rad_s"],
            "acceleration_limit": limits["solver_acceleration_limit_rad_s2"],
            "device": self.rpc.device, "seed": self.seed, "interpolation_dt": .02}

    def _configure(self, profile: str, snapshots: list[dict[str, Any]]) -> None:
        if self.session is not None:
            snapshot = self._shared_snapshots(profile, snapshots)[0]
            result = self.session.acquire(profile, self._request(profile, snapshot))
            self.initialization.append({"session_id": profile, **result})
            self.profile = profile
            return
        for arm, snapshot in zip(("left", "right"), snapshots):
            name, request = f"{arm}:{profile}", self._request(profile, snapshot)
            result = self.rpc.request("configure", session_id=name, **request)
            self.initialization.append({"session_id": f"{arm}:{profile}", **result})
        self.profile = profile

    def plan(self, phase: str, poses: tuple[Any, Any], *, closed: bool,
             checker: Any, limits: dict[str, Any]) -> Any:
        from .curobo_scene import _matrix, _pose
        self.serial += 1
        snapshots = [self._snapshot(index, phase=phase, closed=closed, checker=checker) for index in (0, 1)]
        profile = "approach" if phase in ("pregrasp", "descend") else "global"
        if self.profile != profile:
            if self.profile != "approach" or profile != "global":
                raise ValueError("The task may transition from approach to transport only once")
            if getattr(self, "session", None) is None:
                for arm in ("left", "right"):
                    self.rpc.request("release", session_id=f"{arm}:approach")
            self._configure(profile, snapshots)
        if getattr(self, "session", None) is not None:
            snapshots = self._shared_snapshots(profile, snapshots)
        paths, audits = [], []
        call_started = time.monotonic()
        phase_report = {"name": phase, "status": "planning", "arms": audits}
        self.planning.append(phase_report)
        try:
            for index, (arm, binding, snapshot, goal) in enumerate(zip(
                    ("left", "right"), self.arms, snapshots, poses)):
                session = f"{arm}:{profile}"
                if getattr(self, "session", None) is not None:
                    session = profile
                    acquired = self.session.acquire(profile, self._request(profile, snapshot))
                    if acquired["configuration_reused"] is False:
                        self.initialization.append({"session_id": profile, "arm": arm, **acquired})
                arm_report: dict[str, Any] = {"session_id": session, "arm": arm}
                audits.append(arm_report)
                update = self.rpc.request("update", session_id=session,
                    robot_cfg=snapshot["robot_cfg"], world=snapshot["world"],
                    payload_model=snapshot["payload_model"], payload_state=snapshot["payload_state"],
                    reset_seed=getattr(self, "session", None) is not None)
                arm_report["update"] = update
                goal_base = _pose(np.linalg.inv(_matrix(binding.agent.robot.pose)) @ _matrix(goal))
                native = self.rpc.request("plan_pose", session_id=session, snapshot_id=update["snapshot_id"],
                    start=snapshot["q_start"], goal_pose=goal_base, attempts=2)
                # Preserve native failure details before trajectory decoding or
                # endpoint checks can raise and terminate the task.
                arm_report.update({key: deepcopy(native.get(key)) for key in (
                    "status", "native_status", "success", "native_result_is_none",
                    "graph_calls", "graph_success", "graph_time_s", "attempts",
                    "solve_wall_s", "position_error", "rotation_error", "feasible",
                    "trajectory_source")})
                trajectory = checked_trajectory(native, binding.arm_qpos(), binding.names,
                    -1. if closed else 1., binding.key,
                    velocity_limit=limits["nominal_velocity_limit_rad_s"],
                    acceleration_limit=limits["nominal_acceleration_limit_rad_s2"])
                solver, planner = self.solvers[index], self.solvers[index].planner
                measured = binding.full_qpos()
                active_names = [j.get_name() for j in binding.agent.robot.get_active_joints()]
                q = measured[[active_names.index(name) for name in planner.user_joint_names]]
                q[np.asarray(planner.move_group_joint_indices)] = trajectory.positions_rad[-1]
                planner.pinocchio_model.compute_forward_kinematics(q)
                actual = planner.pinocchio_model.get_link_pose(planner.link_name_2_idx[planner.move_group])
                error = pose_error(actual, goal_base)
                audit = {"status": "passed", "corrections": 0, "final_error": error,
                    "position_tolerance_m": POSITION_TOLERANCE_M,
                    "rotation_tolerance_rad": ROTATION_TOLERANCE_RAD}
                solver.last_endpoint_audit = audit
                arm_report.update(endpoint=audit, duration_s=float(trajectory.times_s[-1]))
                if error["position_m"] > POSITION_TOLERANCE_M or error["rotation_rad"] > ROTATION_TOLERANCE_RAD:
                    audit["status"] = "failed"
                    raise ValueError(f"cuRobo actual FK endpoint exceeds task tolerances: {error}")
                paths.append(trajectory)
            schedule = build_hold_schedule(*paths, control_timestep_s=self.raw.control_timestep)
            phase_report["status"] = "passed"
            return schedule
        except Exception as error:
            phase_report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            phase_report["wall_seconds"] = time.monotonic() - call_started

    def summary(self) -> dict[str, Any]:
        return {"version": "0.9.0", "initialization": self.initialization,
            "planning": self.planning, "requests": list(self.rpc.receipts[self.request_start:]),
            "episode_seed": self.episode_seed, "planner_seed": self.seed,
            "path_generator": "curobo", "mplib_path_fallback": False,
            "shared_geometry_checks": True, "joint_14d_planning": False,
            "resident_planners": 2 if self.session is None else len(self.session.configured),
            "phase_profiles_constructed_once": True, "cross_episode_reuse": self.session is not None,
            "shared_per_query_seed_reset": self.session is not None}

    def close(self) -> None:
        if self.session is None:
            self.rpc.close()
        else:
            self.session.end_episode()


def self_test() -> None:
    start = np.zeros(7)
    result = {"success": True, "joint_names": list(ARM_NAMES), "times_s": [0., .05],
        "positions": [start.tolist(), (start + .01).tolist()],
        "velocity": [[0.] * 7] * 2, "acceleration": [[0.] * 7] * 2}
    trajectory = checked_trajectory(result, start, ARM_NAMES, 1., "left",
        velocity_limit=1., acceleration_limit=1.)
    assert trajectory.positions_rad.shape == (2, 7)
    invalid = {**result, "success": False}
    try:
        checked_trajectory(invalid, start, ARM_NAMES, 1., "left", velocity_limit=1., acceleration_limit=1.)
    except ValueError:
        pass
    else:
        raise AssertionError("unsuccessful native path accepted")
    print("cuRobo client CPU check passed: positions (2,7), times (2,)")


if __name__ == "__main__":
    self_test()
