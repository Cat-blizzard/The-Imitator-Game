"""Persistent, phase-versioned cuRobo sessions for TIG's two physical arms.

Four native planners are initialized once: each arm has fixed approach and
transport limits. Updating a snapshot changes measured fingers/default state
and scene, never the limits cached inside trajectory optimization/TimeOpt.
The client owns no simulator and never advances physics. Joint values are
radians (finger positions metres), poses XYZ/WXYZ, and times seconds.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
from typing import Any, Protocol
import xml.etree.ElementTree as ET

import numpy as np

from .curobo_backend import CuroboProcess, ROOT, checked_array
from .curobo_scene import ARM_NAMES, FINGER_NAMES

PROFILE_LIMITS: dict[str, tuple[float, float]] = {
    "approach": (.9, .9), "transport": (1.5, 2.6),
}
ARMS = ("left", "right")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _positive_vector(value: Any, count: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(count, float(array))
    elif array.shape == (1,):
        array = np.full(count, float(array[0]))
    if array.shape != (count,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{label} must contain {count} finite positive limits")
    return array


def profile_config(robot_cfg: dict[str, Any], profile: str) -> dict[str, Any]:
    """Copy actual config and set seven arm limits; retain both finger limits.

    Input has the exporter's nine independently named joints. Velocity scaling
    is recomputed from the actual referenced URDF limits, by name. Output
    always has nine-element velocity/acceleration vectors and leaves every
    collision/kinematic/default/locked-joint parameter unchanged.
    """
    if profile not in PROFILE_LIMITS:
        raise ValueError(f"Unknown fixed limit profile: {profile!r}")
    result = copy.deepcopy(robot_cfg)
    kin = result["kinematics"]
    cspace = kin["cspace"]
    names = tuple(cspace["joint_names"])
    expected = set(ARM_NAMES + FINGER_NAMES)
    if len(names) != 9 or len(set(names)) != 9 or set(names) != expected:
        raise ValueError("Profiles require the actual seven arm and two independent finger joints")
    checked_array(cspace["default_joint_position"], (9,), "default configuration")
    locked = kin["lock_joints"]
    if set(locked) != set(FINGER_NAMES):
        raise ValueError("Exactly the two measured physical fingers must be locked")
    for name in FINGER_NAMES:
        value = float(locked[name])
        if (not math.isfinite(value)
                or abs(value - float(cspace["default_joint_position"][names.index(name)])) > 1e-9):
            raise ValueError("Finger lock/default values must describe the same measured state")
    velocity_scale = _positive_vector(cspace["velocity_scale"], 9, "velocity scales")
    acceleration = _positive_vector(cspace["max_acceleration"], 9, "accelerations")
    urdf = Path(kin["urdf_path"])
    if not urdf.is_absolute() or not urdf.is_file():
        raise ValueError("An existing absolute physical/derived URDF is required")
    joints = {node.attrib["name"]: node for node in ET.parse(urdf).getroot().findall("joint")}
    velocity, acceleration_limit = PROFILE_LIMITS[profile]
    for index, name in enumerate(names):
        node = joints.get(name)
        limit = None if node is None else node.find("limit")
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(f"URDF joint lacks velocity limit: {name}")
        native_limit = float(limit.attrib["velocity"])
        if not math.isfinite(native_limit) or native_limit <= 0:
            raise ValueError(f"Invalid URDF velocity limit: {name}")
        if name in ARM_NAMES:
            if velocity > native_limit:
                raise ValueError(f"Profile exceeds physical URDF velocity for {name}")
            velocity_scale[index] = velocity / native_limit
            acceleration[index] = acceleration_limit
    cspace["velocity_scale"] = velocity_scale.tolist()
    cspace["max_acceleration"] = acceleration.tolist()
    json.dumps(result, allow_nan=False)
    return result


class SessionTransport(Protocol):
    receipts: list[dict[str, Any]]

    def request(self, op: str, **payload: Any) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _SessionProcess(CuroboProcess):
    """NDJSON transport to the new worker, retaining existing owned shutdown."""
    def __init__(self, gpu_uuid: str, log_path: Path, *, timeout_s: float = 180.,
                 python: Path | None = None) -> None:
        python = python or ROOT / "infrastructure/envs/curobo-tig/bin/python"
        worker = ROOT / "infrastructure/curobo_tig_session_worker.py"
        if not python.is_file() or not worker.is_file():
            raise FileNotFoundError("Independent cuRobo environment/session worker is missing")
        if (not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-")
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("Explicit GPU UUID and finite positive request timeout are required")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(CUDA_VISIBLE_DEVICES=gpu_uuid, PYTHONNOUSERSITE="1",
                           PYTHONDONTWRITEBYTECODE="1")
        cache = ROOT / "infrastructure/tmp/curobo-v2-runtime-cache"
        for variable, subdirectory in (("CUDA_CACHE_PATH", "cuda"),
                                       ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                                       ("WARP_CACHE_PATH", "warp")):
            directory = cache / subdirectory
            directory.mkdir(parents=True, exist_ok=True)
            environment[variable] = str(directory)
        self._log = log_path.open("x")
        try:
            self.process = subprocess.Popen(
                [str(python), "-B", "-u", str(worker), "--serve"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                env=environment, cwd=ROOT, text=True, bufsize=1)
        except Exception:
            self._log.close()
            raise
        self.timeout_s, self.sequence, self.failed = float(timeout_s), 0, False
        self.receipts: list[dict[str, Any]] = []

    def request(self, op: str, **payload: Any) -> dict[str, Any]:
        if self.failed:
            raise RuntimeError("Session transport failed; close it before creating another")
        self.sequence += 1
        data = json.dumps({"id": self.sequence, "op": op, **payload}, allow_nan=False)
        before = time.monotonic()
        assert self.process.stdin is not None and self.process.stdout is not None
        timeout = max(self.timeout_s, 600.) if op == "init_session" else self.timeout_s
        try:
            self.process.stdin.write(data + "\n")
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                if not selector.select(timeout):
                    raise TimeoutError(f"cuRobo {op} exceeded {timeout}s; see worker log")
                line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"cuRobo worker exited during {op}: {self.process.poll()}")
            response = json.loads(line)
            if not isinstance(response, dict) or response.get("id") != self.sequence:
                raise RuntimeError("cuRobo session protocol response ID mismatch")
        except Exception as error:
            self.failed = True
            self.receipts.append({"id": self.sequence, "op": op, "ok": False,
                                  "wall_s": time.monotonic() - before,
                                  "error": repr(error), "timeout_s": timeout})
            raise
        self.receipts.append({"id": self.sequence, "op": op,
                              "wall_s": time.monotonic() - before,
                              "ok": response.get("ok"), "session_id": payload.get("session_id"),
                              "snapshot_id": payload.get("snapshot_id")})
        if response.get("ok") is not True:
            raise RuntimeError(f"cuRobo {op}: {response.get('error')}")
        if not isinstance(response.get("result"), dict):
            self.failed = True
            raise RuntimeError("cuRobo session result must be an object")
        return response["result"]


class CuroboSessions:
    """Four fixed planners, with explicit current-state tokens before queries.

    Paired snapshots must be captured without advancing the simulator between
    exports. Their physical token can be reused across profiles only when both
    complete input snapshots are identical. Any paired update failure clears
    both client tokens; partial updates cannot accidentally authorize a plan.
    """
    def __init__(self, gpu_uuid: str, log_path: Path, *, timeout_s: float = 180.,
                 python: Path | None = None, transport: SessionTransport | None = None) -> None:
        self._rpc: SessionTransport = transport if transport is not None else _SessionProcess(
            gpu_uuid, log_path, timeout_s=timeout_s, python=python)
        self._initialization_attempted = False
        self._initialized = False
        self._tokens: dict[str, str | None] = {}
        self._snapshot_digests: dict[str, str] = {}
        self._current_snapshot_id: str | None = None

    @property
    def process(self) -> subprocess.Popen[str]:
        return self._rpc.process  # type: ignore[attr-defined]

    @property
    def receipts(self) -> list[dict[str, Any]]:
        return self._rpc.receipts

    @property
    def current_snapshot_id(self) -> str | None:
        return self._current_snapshot_id

    @staticmethod
    def session_id(arm: int, profile: str) -> str:
        if isinstance(arm, bool) or not isinstance(arm, int) or arm not in (0, 1):
            raise ValueError("Canonical arm index must be 0=left or 1=right")
        if profile not in PROFILE_LIMITS:
            raise ValueError(f"Unknown fixed limit profile: {profile!r}")
        return f"{ARMS[arm]}:{profile}"

    @staticmethod
    def _pair(snapshots: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(snapshots, (tuple, list)) or len(snapshots) != 2:
            raise ValueError("Two complete snapshots in canonical left/right order are required")
        for snapshot in snapshots:
            if (not isinstance(snapshot, dict) or not isinstance(snapshot.get("robot_cfg"), dict)
                    or not isinstance(snapshot.get("world"), dict)):
                raise ValueError("Snapshot must contain robot_cfg and world dictionaries")
        _digest(snapshots)
        return snapshots[0], snapshots[1]

    def initialize_pair(self, snapshots: tuple[dict[str, Any], dict[str, Any]], *,
                        seed: int = 401, interpolation_dt: float = .02,
                        warmup: bool = True) -> list[dict[str, Any]]:
        if self._initialization_attempted:
            raise RuntimeError("Initialize each process only once; existing planners must be reused")
        snapshots = self._pair(snapshots)
        if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                or not math.isfinite(interpolation_dt) or interpolation_dt <= 0
                or not isinstance(warmup, bool)):
            raise ValueError("Seed, interpolation timestep and warmup flag are invalid")
        configs = {(arm, profile): profile_config(snapshot["robot_cfg"], profile)
                   for profile in PROFILE_LIMITS for arm, snapshot in enumerate(snapshots)}
        self._initialization_attempted = True
        results = []
        for profile in PROFILE_LIMITS:
            for arm, snapshot in enumerate(snapshots):
                session = self.session_id(arm, profile)
                self._tokens[session] = None
                results.append(self._rpc.request("init_session", session_id=session,
                    robot_cfg=configs[arm, profile], world=snapshot["world"], seed=seed,
                    interpolation_dt=float(interpolation_dt), warmup=warmup))
        self._initialized = True
        return results

    def update_pair(self, snapshots: tuple[dict[str, Any], dict[str, Any]], *,
                    profile: str, snapshot_id: str) -> list[dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Initialize all four fixed sessions before updating state")
        sessions = [self.session_id(arm, profile) for arm in (0, 1)]
        # A rejected attempted update cannot leave old state authorized locally.
        self._current_snapshot_id = None
        for session in sessions:
            self._tokens[session] = None
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("A nonempty physical snapshot version is required")
        snapshots = self._pair(snapshots)
        digest = _digest(snapshots)
        if snapshot_id in self._snapshot_digests and self._snapshot_digests[snapshot_id] != digest:
            raise ValueError("A physical snapshot token cannot be reused for different state")
        configs = [profile_config(snapshot["robot_cfg"], profile) for snapshot in snapshots]
        results = []
        for arm, snapshot in enumerate(snapshots):
            results.append(self._rpc.request("update_session", session_id=sessions[arm],
                robot_cfg=configs[arm], world=snapshot["world"], snapshot_id=snapshot_id))
        self._snapshot_digests[snapshot_id] = digest
        for session in sessions:
            self._tokens[session] = snapshot_id
        self._current_snapshot_id = snapshot_id
        return results

    def _ready(self, arm: int, profile: str, snapshot_id: str) -> str:
        session = self.session_id(arm, profile)
        if (not self._initialized or self._tokens.get(session) != snapshot_id
                or self._current_snapshot_id != snapshot_id or snapshot_id is None):
            raise RuntimeError(f"Session {session} has no successful update for snapshot {snapshot_id!r}")
        return session

    def fk(self, *, arm: int, profile: str, snapshot_id: str, positions: Any) -> dict[str, Any]:
        session = self._ready(arm, profile, snapshot_id)
        array = np.asarray(positions, dtype=np.float64)
        if array.shape == (7,):
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != 7 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("FK requires a nonempty finite seven-joint array[N,7]")
        return self._rpc.request("fk", session_id=session, snapshot_id=snapshot_id,
                                 positions=array.tolist())

    def plan_pose(self, *, arm: int, profile: str, snapshot_id: str, start: Any,
                  goal_pose: Any, max_attempts: int = 4) -> dict[str, Any]:
        session = self._ready(arm, profile, snapshot_id)
        start_array = checked_array(start, (7,), "measured planning start")
        goal = checked_array(goal_pose, (7,), "base-frame XYZ/WXYZ goal")
        if not np.isclose(np.linalg.norm(goal[3:]), 1., rtol=0, atol=1e-5):
            raise ValueError("Planning goal quaternion must be normalized")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("Maximum native attempts must be a positive integer")
        return self._rpc.request("plan_pose", session_id=session, snapshot_id=snapshot_id,
            start=start_array.tolist(), goal_pose=goal.tolist(), max_attempts=max_attempts)

    def inspect_sessions(self) -> dict[str, Any]:
        return self._rpc.request("inspect_sessions")

    def close(self) -> None:
        self._tokens.clear()
        self._initialized = False
        self._current_snapshot_id = None
        self._rpc.close()


def self_test() -> dict[str, Any]:
    """CPU-only profile, token, paired-failure and input-contract checks."""
    class FakeTransport:
        def __init__(self) -> None:
            self.receipts: list[dict[str, Any]] = []
            self.fail_right_update = False
            self.closed = False

        def request(self, op: str, **payload: Any) -> dict[str, Any]:
            self.receipts.append({"op": op, **copy.deepcopy(payload)})
            if (self.fail_right_update and op == "update_session"
                    and payload["session_id"].startswith("right:")):
                raise RuntimeError("synthetic partial pair failure")
            return {"op": op, **payload}

        def close(self) -> None:
            self.closed = True

    rejected = 0
    with tempfile.TemporaryDirectory(prefix="tig-curobo-session-cpu-") as directory:
        urdf = Path(directory) / "nine.urdf"
        names = tuple(reversed(ARM_NAMES + FINGER_NAMES))
        velocities = {name: 2. + index * .1 for index, name in enumerate(names)}
        urdf.write_text("<robot name='test'>" + "".join(
            f"<joint name='{name}'><limit velocity='{value}'/></joint>"
            for name, value in velocities.items()) + "</robot>")
        defaults = [(.021 if name == FINGER_NAMES[0] else .022) if name in FINGER_NAMES else 0.
                    for name in names]
        cfg = {"kinematics": {"urdf_path": str(urdf), "lock_joints":
            {FINGER_NAMES[0]: .021, FINGER_NAMES[1]: .022},
            "collision_spheres": {"finger": [{"center": [0., 0., 0.], "radius": .01}]},
            "cspace": {"joint_names": list(names), "default_joint_position": defaults,
                       "velocity_scale": [.2] * 9, "max_acceleration": .7}}, "load_dynamics": False}
        original = copy.deepcopy(cfg)
        for profile, limits in PROFILE_LIMITS.items():
            converted = profile_config(cfg, profile)
            space = converted["kinematics"]["cspace"]
            assert np.asarray(space["velocity_scale"]).shape == (9,)
            for index, name in enumerate(names):
                if name in ARM_NAMES:
                    assert math.isclose(space["velocity_scale"][index] * velocities[name], limits[0])
                    assert space["max_acceleration"][index] == limits[1]
                else:
                    assert space["velocity_scale"][index] == .2
                    assert space["max_acceleration"][index] == .7
            assert converted["kinematics"]["lock_joints"] == cfg["kinematics"]["lock_joints"]
        assert cfg == original
        snapshots = ({"robot_cfg": cfg, "world": {"cuboid": {}, "mesh": {}}},) * 2
        fake = FakeTransport()
        client = CuroboSessions("GPU-cpu-test-only", Path(directory) / "unused.log", transport=fake)
        client.initialize_pair(snapshots, warmup=False)
        assert len(fake.receipts) == 4
        assert len({item["session_id"] for item in fake.receipts}) == 4
        def reject(function: Any) -> None:
            nonlocal rejected
            try:
                function()
            except (ValueError, RuntimeError):
                rejected += 1
            else:
                raise AssertionError("Invalid operation was accepted")
        query = {"arm": 0, "profile": "approach", "snapshot_id": "first", "positions": [0.] * 7}
        reject(lambda: client.fk(**query))
        reject(lambda: client.initialize_pair(snapshots))
        client.update_pair(snapshots, profile="approach", snapshot_id="first")
        fk = client.fk(**query)
        assert np.asarray(fk["positions"]).shape == (1, 7)
        reject(lambda: client.fk(**{**query, "snapshot_id": "stale"}))
        reject(lambda: client.fk(**{**query, "positions": [float("nan")] * 7}))
        reject(lambda: client.plan_pose(arm=0, profile="approach", snapshot_id="first",
                                       start=[0.] * 7, goal_pose=[0.] * 7))
        client.update_pair(snapshots, profile="transport", snapshot_id="first")
        plan = client.plan_pose(arm=1, profile="transport", snapshot_id="first",
            start=[0.] * 7, goal_pose=[0., 0., 0., 1., 0., 0., 0.])
        assert plan["session_id"] == "right:transport"
        changed = copy.deepcopy(snapshots)
        changed[0]["world"]["cuboid"]["box"] = {"pose": [0., 0., 0., 1., 0., 0., 0.], "dims": [1., 1., 1.]}
        reject(lambda: client.update_pair(changed, profile="approach", snapshot_id="first"))
        reject(lambda: client.fk(**query))
        client.update_pair(snapshots, profile="approach", snapshot_id="second")
        reject(lambda: client.fk(arm=1, profile="transport", snapshot_id="first", positions=[0.] * 7))
        fake.fail_right_update = True
        reject(lambda: client.update_pair(snapshots, profile="approach", snapshot_id="third"))
        reject(lambda: client.fk(**{**query, "snapshot_id": "second"}))
        reject(lambda: client.fk(**{**query, "snapshot_id": "third"}))
        reject(lambda: profile_config(cfg, "unsupported"))
        bad = copy.deepcopy(cfg)
        bad["kinematics"]["cspace"]["velocity_scale"] = [float("inf")] * 9
        reject(lambda: profile_config(bad, "approach"))
        bad_names = copy.deepcopy(cfg)
        bad_names["kinematics"]["cspace"]["joint_names"] = ["duplicate"] * 9
        reject(lambda: profile_config(bad_names, "transport"))
        client.close()
        assert fake.closed
        assert sum(item["op"] == "init_session" for item in fake.receipts) == 4
    assert rejected == 14
    return {"status": "passed", "profile_shape": [9], "fk_shape": [1, 7],
            "fixed_sessions": 4, "rejected_cases": rejected, "gpu_initialized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), allow_nan=False))
