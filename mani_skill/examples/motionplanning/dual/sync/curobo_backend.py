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
    """Two resident arm planners; approach models retire before transport.

    Each fixed phase profile is constructed once. The workflow never returns to
    approach after grasp, so it need not keep both profiles on the GPU at once.
    """

    def __init__(self, env: Any, solvers: list[Any], task: Any, *, seed: int | None,
                 options: Mapping[str, Any] | None, limits: dict[str, Any]) -> None:
        self.raw, self.solvers, self.task = env.unwrapped, solvers, task
        options = dict(options or {})
        unsupported = set(options) - {"python", "device", "cache_dir"}
        if unsupported:
            raise ValueError(f"Unsupported cuRobo options: {sorted(unsupported)}")
        self.cache_dir = options.get("cache_dir")
        self.arms = bindings(self.raw, tuple(self.raw.agent.agents_dict))
        self.seed, self.limits = int(seed or 0), limits
        self.rpc = CuroboProcess(**options)
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
        if closed and not model.get("cover_frozen_at_grasp", False):
            agent = self.raw.agent.agents[arm_index]
            if not checker.attached or not bool(as_numpy(agent.is_grasping(target)).item()):
                raise ValueError("cuRobo payload refinement requires a confirmed physical grasp")
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
        state: dict[str, Any] = {"contact_allowed": phase == "descend" or closed,
            "attachment": None, "source_support_allowed": False,
            "source_support_obstacle_names": []}
        if closed:
            agent = self.raw.agent.agents[arm_index]
            if not checker.attached or not bool(as_numpy(agent.is_grasping(target)).item()):
                raise ValueError("cuRobo payload attachment requires a current confirmed physical grasp")
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
        self._configure("approach", snapshots)

    def _configure(self, profile: str, snapshots: list[dict[str, Any]]) -> None:
        from .phase_kinematics import compact_phase_limits
        phase = "pregrasp" if profile == "approach" else "carry_lower"
        limits = compact_phase_limits(self.limits, phase)
        for arm, snapshot in zip(("left", "right"), snapshots):
            result = self.rpc.request("configure", session_id=f"{arm}:{profile}",
                robot_cfg=snapshot["robot_cfg"], world=snapshot["world"],
                payload_model=snapshot["payload_model"], payload_state=snapshot["payload_state"],
                velocity_limit=limits["solver_velocity_limit_rad_s"],
                acceleration_limit=limits["solver_acceleration_limit_rad_s2"],
                device=self.rpc.device, seed=self.seed, interpolation_dt=.02)
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
            for arm in ("left", "right"):
                self.rpc.request("release", session_id=f"{arm}:approach")
            self._configure(profile, snapshots)
        paths, audits = [], []
        call_started = time.monotonic()
        phase_report = {"name": phase, "status": "planning", "arms": audits}
        self.planning.append(phase_report)
        try:
            for index, (arm, binding, snapshot, goal) in enumerate(zip(
                    ("left", "right"), self.arms, snapshots, poses)):
                session = f"{arm}:{profile}"
                arm_report: dict[str, Any] = {"session_id": session}
                audits.append(arm_report)
                update = self.rpc.request("update", session_id=session,
                    robot_cfg=snapshot["robot_cfg"], world=snapshot["world"],
                    payload_state=snapshot["payload_state"])
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
            "planning": self.planning, "requests": list(self.rpc.receipts),
            "path_generator": "curobo", "mplib_path_fallback": False,
            "shared_geometry_checks": True, "joint_14d_planning": False,
            "resident_planners": 2, "phase_profiles_constructed_once": True}

    def close(self) -> None:
        self.rpc.close()


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
