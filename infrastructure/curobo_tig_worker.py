#!/usr/bin/env python3
"""Isolated cuRobo V2 (official v0.9.0) planner for TIG; JSON lines.

This process owns one CUDA MotionPlanner instance. It neither imports TIG nor steps
an environment.  Joint arrays have seven columns in the returned joint-name
order; poses are in the exported robot base frame, [x,y,z,qw,qx,qy,qz].
Only genuine successful MotionPlanner results produce trajectories. The caller
must still validate the paired trajectory with TIG's full collision checker.

Run --self-test without CUDA/cuRobo imports, or --serve inside the dedicated
cuRobo environment.  Native/vendor stdout is redirected to stderr in --serve;
the original stdout descriptor carries JSON responses only.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, TextIO


DOF = 7
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, DOF + 1)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
PROTOCOL_VERSION = 1
CUROBO_COMMIT = "42498cf4e7d0d6fbd4ef9fbcec1243c00c888a83"


def _finite_json(value: Any, path: str = "request") -> None:
    """Reject non-JSON values and NaN/Inf recursively before native calls."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, list):
        for i, child in enumerate(value):
            _finite_json(child, f"{path}[{i}]")
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for key, child in value.items():
            _finite_json(child, f"{path}.{key}")
        return
    raise ValueError(f"{path} is not a JSON value")


def _vector(value: Any, width: int, name: str) -> list[float]:
    """Validate one finite physical vector, preserving its exact ordering."""
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{name} must have shape ({width},)")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError(f"{name} must contain real numbers")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _matrix(value: Any, width: int, name: str) -> list[list[float]]:
    """Validate nonempty (N,width) arrays without importing GPU libraries."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must have shape (N,{width}) with N>=1")
    return [_vector(row, width, f"{name}[{i}]") for i, row in enumerate(value)]


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def validate_request(request: Any) -> dict[str, Any]:
    """Validate protocol and shape invariants before any CUDA initialization."""
    _finite_json(request)
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    request_id = request.get("id")
    if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
        raise ValueError("id must be a nonnegative integer")
    operation = request.get("op")
    if operation not in {"configure", "fk", "plan_pose", "shutdown"}:
        raise ValueError("unknown operation")
    if operation == "configure":
        robot_cfg = request.get("robot_cfg")
        if not isinstance(robot_cfg, dict) or not isinstance(robot_cfg.get("kinematics"), dict):
            raise ValueError("robot_cfg must contain a kinematics object")
        if not isinstance(request.get("world"), dict):
            raise ValueError("world must be an object")
        if robot_cfg.get("load_dynamics", False) is not False:
            raise ValueError("this initial geometric integration requires load_dynamics=False")
        _positive_float(request.get("interpolation_dt", 0.02), "interpolation_dt")
        seed = request.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer in [0,2**31)")
        if not isinstance(request.get("warmup", False), bool):
            raise ValueError("warmup must be boolean")
    elif operation == "fk":
        _matrix(request.get("positions"), DOF, "positions")
    elif operation == "plan_pose":
        _vector(request.get("start"), DOF, "start")
        goal = _vector(request.get("goal_pose"), 7, "goal_pose")
        norm = math.sqrt(sum(x * x for x in goal[3:]))
        if abs(norm - 1.0) > 1e-5:
            raise ValueError("goal_pose quaternion must be unit wxyz")
        attempts = request.get("max_attempts", 4)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 4:
            raise ValueError("max_attempts must be an integer from 1 to 4")
        _positive_float(request.get("timeout_s", 30.0), "timeout_s")
    return request


def _tensor_json(value: Any) -> Any:
    """Copy a genuine result tensor to JSON; missing measurements stay null."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    _finite_json(value, "result")
    return value


def _diagnostic_json(value: Any) -> Any:
    """Preserve native failure diagnostics, encoding nonfinite values as text.

    Successful executable trajectories use _tensor_json and must be finite.
    Solver failures can legitimately report inf pose errors; they must remain
    plan failures in the protocol, rather than disappear as serialization errors.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, list):
        return [_diagnostic_json(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _diagnostic_json(v) for k, v in value.items()}
    _finite_json(value, "diagnostic")
    return value


def _single_value(value: Any, name: str) -> Any:
    """Unwrap singleton batch/seed axes; never choose from multiple results."""
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one value")
        value = value[0]
    return value


def _trajectory_matrix(value: Any, name: str, width: int = DOF) -> list[list[float]]:
    """Accept [H,D], [1,H,D] or [1,1,H,D] from the selected V2 result."""
    # JSON has no rank attribute. Count list axes from a nonempty first item,
    # then validate every row after removing singleton batch/seed axes.
    rank, item = 0, value
    while isinstance(item, list):
        if not item:
            raise ValueError(f"{name} must be nonempty")
        rank += 1
        item = item[0]
    if rank not in (2, 3, 4):
        raise ValueError(f"{name} must have rank 2, 3 or 4")
    for _ in range(rank - 2):
        if len(value) != 1:
            raise ValueError(f"{name} has multiple batches/seeds")
        value = value[0]
    return _matrix(value, width, name)


def extract_active_columns(native: dict[str, Any], locked: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate full nine-DOF physical output and select seven named arm columns.

    The two additional columns must be exactly the two configured locked
    fingers. Their positions must equal their individually measured lock
    values, and all returned derivatives must remain zero within 1e-6.
    Original full arrays are retained by the caller as native_trajectory.
    """
    names = native.get("joint_names")
    expected = ARM_JOINT_NAMES + FINGER_JOINT_NAMES
    if (not isinstance(names, list) or len(names) != 9
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != 9 or set(names) != set(expected)):
        raise ValueError(f"native trajectory must expose nine unique actual Panda joint names, got {names}")
    if set(locked) != set(FINGER_JOINT_NAMES):
        raise ValueError("native trajectory requires both individually configured finger locks")
    for name in FINGER_JOINT_NAMES:
        _vector([locked[name]], 1, f"lock {name}")
    arrays = {field: _matrix(native.get(field), 9, f"native.{field}")
              for field in ("position", "velocity", "acceleration", "jerk")}
    n = len(arrays["position"])
    if n < 2 or any(len(value) != n for value in arrays.values()):
        raise ValueError("native trajectory arrays must share a horizon with at least two samples")
    times = _vector(native.get("times_s"), n, "native.times_s")
    if times[0] != 0.0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("native timestamps must increase strictly from zero")
    dt = _positive_float(native.get("interpolation_dt"), "native.interpolation_dt")
    if any(abs(t - i * dt) > 1e-10 for i, t in enumerate(times)):
        raise ValueError("native times do not match the recorded interpolation timestep")
    fingers: dict[str, Any] = {}
    for name in FINGER_JOINT_NAMES:
        index = names.index(name)
        position_error = max(abs(row[index] - locked[name]) for row in arrays["position"])
        derivative_peaks = {field: max(abs(row[index]) for row in arrays[field])
                            for field in ("velocity", "acceleration", "jerk")}
        if position_error > 1e-6 or any(value > 1e-6 for value in derivative_peaks.values()):
            raise ValueError(f"native locked finger {name} moved: position_error={position_error}, derivatives={derivative_peaks}")
        fingers[name] = {"column": index, "configured_position": locked[name],
                         "max_position_error": position_error, "derivative_peaks": derivative_peaks}
    columns = [names.index(name) for name in ARM_JOINT_NAMES]
    selected = {field: [[row[index] for index in columns] for row in values]
                for field, values in arrays.items()}
    selected.update(joint_names=list(ARM_JOINT_NAMES), times_s=times, interpolation_dt=dt)
    return selected, {"native_joint_names": names, "selected_column_indices": columns,
                      "locked_fingers": fingers, "selection_by_name": True}


def _phase(label: str, begin: float) -> None:
    print(f"[cuRobo phase] {label}; elapsed_s={time.perf_counter() - begin:.6f}", file=sys.stderr, flush=True)


def _git_identity(source: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"module_path": str(source)}
    for label, args in (
        ("git_commit", ["rev-parse", "HEAD"]),
        ("git_status", ["status", "--porcelain"]),
    ):
        process = subprocess.run(
            ["git", "-C", str(source.parent), *args],
            check=False, capture_output=True, text=True, timeout=10,
        )
        result[label] = process.stdout.strip() if process.returncode == 0 else None
    return result


def _compare_package(installed: Path, official: Path) -> dict[str, Any]:
    """Verify installed runtime code/data against the pinned official source.

    Wheel generation creates _version.py; its measured hash is recorded
    separately. Bytecode is interpreter generated and is never source evidence.
    Missing optional tests/examples are permitted; public modules, _src modules
    and kernels, and runtime configuration files are required.
    """
    inventory: dict[str, str] = {}
    generated: dict[str, str] = {}
    mismatches: list[str] = []
    for path in sorted(installed.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(installed).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if relative == "_version.py":
            generated[relative] = digest
            continue
        inventory[relative] = digest
        original = official / relative
        if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != digest:
            mismatches.append(relative)
    required = []
    for original in official.rglob("*"):
        if not original.is_file() or "__pycache__" in original.parts:
            continue
        relative = original.relative_to(official).as_posix()
        if relative == "_version.py":
            continue
        if (
            ("/" not in relative and original.suffix == ".py")
            or (relative.startswith("_src/") and original.suffix in {".py", ".cu", ".cuh", ".h"})
            or relative.startswith("content/configs/")
        ):
            required.append(relative)
    missing = sorted(set(required) - inventory.keys())
    if mismatches or missing or not inventory:
        raise RuntimeError(f"installed cuRobo differs from official source: mismatches={mismatches[:10]}, missing={missing[:10]}, checked={len(inventory)}")
    manifest = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return {
        "installed_package_path": str(installed.resolve()),
        "official_package_path": str(official.resolve()),
        "installed_file_count": len(inventory),
        "required_runtime_file_count": len(required),
        "installed_inventory_sha256": hashlib.sha256(manifest).hexdigest(),
        "generated_files_sha256": generated,
        "all_compared_installed_bytes_match_official": True,
    }


def package_identity(module_file: Path) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent / "backends" / "curobo"
    source_module = source_root / "curobo" / "__init__.py"
    identity = _git_identity(source_module)
    identity["source_module_path"] = str(source_module)
    identity["module_path"] = str(module_file)
    if identity["git_commit"] != CUROBO_COMMIT or identity["git_status"] != "":
        raise RuntimeError(f"expected clean official v0.9.0 source {CUROBO_COMMIT}, got {identity}")
    identity["git_dirty"] = False
    identity["installed_source_comparison"] = _compare_package(module_file.parent, source_module.parent)
    return identity


def planner_kwargs(request: dict[str, Any], device_cfg: Any) -> dict[str, Any]:
    """One source of constructor arguments for runtime and installed-API tests."""
    return {
        "robot": {"robot_cfg": request["robot_cfg"]},
        "scene_model": request["world"], "device_cfg": device_cfg,
        "interpolation_dt": float(request.get("interpolation_dt", 0.02)),
        "use_cuda_graph": False, "position_tolerance": 0.001,
        "orientation_tolerance": 0.01, "self_collision_check": True,
        "random_seed": request.get("seed", 0),
        "graph_planner_algorithm": "rrt", "enable_trajopt": True,
        "always_run_trajopt": True,
        # Preserve V2's default TimeOpt. In v0.9.0 it constructs Dynamics
        # independently from actual URDF kinematics even with load_dynamics=False.
        "enable_timeopt": True,
    }


class CuroboBackend:
    """Lazy CUDA backend; construction and CPU protocol tests import no torch."""

    def __init__(self) -> None:
        self.planner: Any = None
        self.torch: Any = None
        self.tensor_args: Any = None
        self.joint_names: list[str] = []
        self._JointState: Any = None
        self._Pose: Any = None
        self._GoalToolPose: Any = None
        self.tool_frame = "panda_hand_tcp"
        self.lock_joints: dict[str, float] = {}

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        configure_begin = time.perf_counter()
        _phase("configure.start", configure_begin)
        # Imports deliberately live behind an explicit configure request.
        import torch
        import curobo
        import curobo._src.runtime as curobo_runtime
        import curobo.runtime as public_runtime
        import warp as wp

        cache_root = Path(__file__).resolve().parent / "tmp"
        runtime_cache = cache_root / "curobo-v2-runtime-cache"
        warp_cache = cache_root / "curobo-v2-warp-cache"
        runtime_cache.mkdir(parents=True, exist_ok=True)
        warp_cache.mkdir(parents=True, exist_ok=True)
        # The public runtime module re-exports scalar values; set the documented
        # backing runtime too, before importing the planners or creating objects.
        curobo_runtime.cache_dir = str(runtime_cache)
        public_runtime.cache_dir = str(runtime_cache)
        wp.config.kernel_cache_dir = str(warp_cache)

        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

        _phase("configure.imports.complete", configure_begin)

        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo planning requires an available CUDA device")
        identity = package_identity(Path(curobo.__file__).resolve())
        _phase("configure.source_identity.complete", configure_begin)
        locks = request["robot_cfg"]["kinematics"].get("lock_joints")
        if not isinstance(locks, dict) or set(locks) != set(FINGER_JOINT_NAMES):
            raise ValueError("configuration must individually lock both actual finger joints")
        self.lock_joints = {name: _vector([locks[name]], 1, name)[0]
                            for name in FINGER_JOINT_NAMES}
        if self.planner is not None:
            self.planner.destroy()
        self.planner = None
        torch.cuda.empty_cache()
        seed = request.get("seed", 0)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.tensor_args = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        self._JointState, self._Pose = JointState, Pose
        self._GoalToolPose = GoalToolPose
        torch.cuda.synchronize()
        begin = time.perf_counter()
        kwargs = planner_kwargs(request, self.tensor_args)
        inspect.signature(MotionPlannerCfg.create).bind(**kwargs)
        _phase("configure.MotionPlannerCfg.create.start", configure_begin)
        config = MotionPlannerCfg.create(**kwargs)
        _phase("configure.MotionPlannerCfg.create.complete", configure_begin)
        if (config.ik_solver_config.robot_config.dynamics is not None
                or config.trajopt_solver_config.robot_config.dynamics is not None
                or not config.trajopt_solver_config.enable_timeopt):
            raise RuntimeError("expected unloaded RobotCfg dynamics with default TimeOpt enabled")
        _phase("configure.MotionPlanner.construct.start", configure_begin)
        self.planner = MotionPlanner(config)
        _phase("configure.MotionPlanner.construct.complete", configure_begin)
        torch.cuda.synchronize()
        initialization_s = time.perf_counter() - begin
        self.joint_names = list(self.planner.joint_names)
        if self.joint_names != ARM_JOINT_NAMES:
            self.planner.destroy()
            self.planner = None
            raise ValueError(f"expected Panda controller joint order {ARM_JOINT_NAMES}, got {self.joint_names}")
        frames = list(self.planner.tool_frames)
        if frames != [self.tool_frame]:
            self.planner.destroy()
            self.planner = None
            raise ValueError(f"expected exactly TCP tool frame {[self.tool_frame]}, got {frames}")
        warmup_s = 0.0
        warmup_success = None
        if request.get("warmup", False):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            _phase("configure.warmup.start", configure_begin)
            warmup_success = bool(self.planner.warmup(enable_graph=True, num_warmup_iterations=1))
            torch.cuda.synchronize()
            warmup_s = time.perf_counter() - begin
            _phase("configure.warmup.complete", configure_begin)
        cfg_bytes = json.dumps(request["robot_cfg"], sort_keys=True, allow_nan=False).encode()
        world_bytes = json.dumps(request["world"], sort_keys=True, allow_nan=False).encode()
        return {
            "protocol_version": PROTOCOL_VERSION,
            "joint_names": self.joint_names,
            "ee_link": self.tool_frame,
            "tool_frames": frames,
            "locked_finger_positions": self.lock_joints,
            "initialization_s": initialization_s,
            "warmup_s": warmup_s,
            "warmup_requested": request.get("warmup", False),
            "warmup_success": warmup_success,
            "robot_cfg_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
            "world_sha256": hashlib.sha256(world_bytes).hexdigest(),
            "curobo_version": str(curobo.__version__),
            "source": identity,
            "python": sys.version,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "cuda_device": str(torch.cuda.get_device_name(0)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "use_cuda_graph": False,
            "self_collision_check": True,
            "position_threshold_m": 0.001,
            "rotation_threshold_rad": 0.01,
            "loaded_robot_dynamics": False,
            "enable_timeopt": True,
            "timeopt_dynamics_scope": "TimeOpt builds Dynamics from actual URDF kinematics independently; torque model not validated against PhysX",
            "graph_planner_algorithm": "rrt",
            "enable_trajopt": True,
            "always_run_trajopt": True,
            "runtime_torch_compile": bool(curobo_runtime.torch_compile),
            "runtime_torch_compile_slow": bool(curobo_runtime.torch_compile_slow),
            "runtime_torch_jit": bool(curobo_runtime.torch_jit),
            "cache_dirs": {"curobo": str(runtime_cache), "warp": str(warp_cache),
                           "cuda": os.environ.get("CUDA_CACHE_PATH"),
                           "torchinductor": os.environ.get("TORCHINDUCTOR_CACHE_DIR")},
            "request_timeout_scope": "caller_process_only; V2 plan_pose has no timeout argument",
            "diagnostic_nonfinite_encoding": "NaN/Infinity/-Infinity strings; executable trajectory must be finite",
        }

    def _require_configured(self) -> None:
        if self.planner is None:
            raise RuntimeError("configure must succeed before querying FK or planning")

    def fk(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require_configured()
        positions = _matrix(request["positions"], DOF, "positions")
        q = self.tensor_args.to_device(positions)
        assert tuple(q.shape) == (len(positions), DOF)
        begin = time.perf_counter()
        _phase("fk.start", begin)
        joint_state = self._JointState.from_position(q, joint_names=self.joint_names)
        state = self.planner.compute_kinematics(joint_state)
        self.torch.cuda.synchronize()
        if state.tool_poses is None or list(state.tool_frames) != [self.tool_frame]:
            raise RuntimeError("cuRobo FK did not return the configured TCP frame")
        assert tuple(state.tool_poses.position.shape) == (len(positions), 1, 1, 3)
        assert tuple(state.tool_poses.quaternion.shape) == (len(positions), 1, 1, 4)
        pose = state.tool_poses.get_link_pose(self.tool_frame)
        xyz = _tensor_json(pose.position)
        quat = _tensor_json(pose.quaternion)
        _matrix(xyz, 3, "fk.positions")
        _matrix(quat, 4, "fk.quaternions")
        if len(xyz) != len(positions) or len(quat) != len(positions):
            raise RuntimeError("cuRobo FK batch length changed")
        _phase("fk.complete", begin)
        return {
            "joint_names": self.joint_names,
            "positions": xyz,
            "quaternions": quat,
            "fk_wall_s": time.perf_counter() - begin,
        }

    def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
        self._require_configured()
        start = _vector(request["start"], DOF, "start")
        goal = _vector(request["goal_pose"], 7, "goal_pose")
        start_tensor = self.tensor_args.to_device([start])
        assert tuple(start_tensor.shape) == (1, DOF)
        joint_state = self._JointState.from_position(start_tensor, joint_names=self.joint_names)
        goal_pose = self._Pose.from_list(goal, device_cfg=self.tensor_args)
        goal_tools = self._GoalToolPose.from_poses(
            {self.tool_frame: goal_pose}, ordered_tool_frames=[self.tool_frame], num_goalset=1,
        )
        self.torch.cuda.synchronize()
        begin = time.perf_counter()
        _phase("plan_pose.start", begin)
        result = self.planner.plan_pose(
            goal_tools, joint_state, max_attempts=request.get("max_attempts", 4),
        )
        self.torch.cuda.synchronize()
        wall_s = time.perf_counter() - begin
        _phase("plan_pose.native.complete", begin)
        if result is None:
            return {"success": False, "status": "NO_SOLUTION",
                    "message": "MotionPlanner.plan_pose returned None",
                    "joint_names": self.joint_names, "planning_wall_s": wall_s,
                    "attempts": int(self.planner.last_planning_attempts),
                    "graph_time_s": float(self.planner.last_graph_time)}
        success_value = _single_value(_tensor_json(result.success), "success")
        if not isinstance(success_value, bool):
            raise RuntimeError("MotionPlanner success must be one boolean")
        success = success_value
        output: dict[str, Any] = {
            "success": success,
            "status": "SUCCESS" if success else "CONSTRAINT_OR_POSE_FAILURE",
            "message": "V2 TrajOptSolverResult.success",
            "joint_names": self.joint_names,
            "planning_wall_s": wall_s,
            "solver_time_s": float(result.solve_time),
            "total_time_s": float(result.total_time),
            "graph_time_s": float(self.planner.last_graph_time),
            "graph_calls": int(self.planner.last_graph_calls),
            "timeopt_time_s": float(result.timeopt_time),
            "attempts": int(self.planner.last_planning_attempts),
            "trajectory_source": result.trajectory_source,
            "position_error_m": _diagnostic_json(result.position_error),
            "rotation_error_rad": _diagnostic_json(result.rotation_error),
            "optimized_dt": _diagnostic_json(result.js_solution.dt) if result.js_solution is not None else None,
            "timeopt_duration_s": _diagnostic_json(result.timeopt_duration),
            "feasible": _diagnostic_json(result.feasible),
            "validation": _diagnostic_json(result.debug_info.get("validation", {})),
            "status_origin": "worker classification of native V2 success; V2 result has no status field",
        }
        if not success:
            return _diagnostic_json(output)
        trajectory = result.get_interpolated_plan()
        if trajectory is None:
            raise RuntimeError("successful MotionPlanner returned no interpolated trajectory")
        if trajectory.joint_names is None:
            raise RuntimeError("full native trajectory lacks explicit joint names")
        names = list(trajectory.joint_names)
        q = _trajectory_matrix(_tensor_json(trajectory.position), "trajectory.position", len(names))
        dt = _positive_float(_single_value(_tensor_json(trajectory.dt), "trajectory.dt"),
                             "interpolation_dt")
        if len(q) < 2:
            raise RuntimeError("successful MotionPlanner returned fewer than two samples")
        native: dict[str, Any] = {
            "joint_names": names, "position": q,
            "times_s": [i * dt for i in range(len(q))],
            "interpolation_dt": dt,
            "dt_tensor": _tensor_json(trajectory.dt),
            "position_tensor_shape": list(trajectory.position.shape),
        }
        for name in ("velocity", "acceleration", "jerk"):
            values = _trajectory_matrix(_tensor_json(getattr(trajectory, name)), f"trajectory.{name}", len(names))
            if len(values) != len(q):
                raise RuntimeError(f"cuRobo trajectory {name} length mismatch")
            native[name] = values
        output["native_trajectory"] = native
        try:
            selected, selection_audit = extract_active_columns(native, self.lock_joints)
        except ValueError as error:
            # Preserve the full physical output even when our execution
            # contract rejects it; the caller stores this before qualification.
            output.update(success=False, native_planner_success=True,
                          status="NATIVE_TRAJECTORY_REJECTED", message=str(error))
            _phase("plan_pose.trajectory_contract.rejected", begin)
            return output
        output.update(selected)
        output["native_column_selection"] = selection_audit
        output["start_error_rad"] = max(abs(a - b) for a, b in zip(selected["position"][0], start))
        _phase("plan_pose.trajectory_contract.complete", begin)
        _finite_json(output, "result")
        return output

    def shutdown(self) -> dict[str, bool]:
        if self.planner is not None:
            self.planner.destroy()
            self.planner = None
            self.torch.cuda.synchronize()
        return {"shutdown": True}


def dispatch(request: Any, backend: Any) -> dict[str, Any]:
    """Process one request; ok describes protocol execution, not plan success."""
    request = validate_request(request)
    if request["op"] == "shutdown":
        with contextlib.redirect_stdout(sys.stderr):
            result = backend.shutdown() if hasattr(backend, "shutdown") else {"shutdown": True}
    else:
        with contextlib.redirect_stdout(sys.stderr):
            result = getattr(backend, request["op"])(request)
    response = {"id": request["id"], "ok": True, "result": result}
    _finite_json(response, "response")
    return response


def serve(source: TextIO, sink: TextIO, backend: Any) -> None:
    """Serve independent JSON lines; errors retain their request id when valid."""
    for line in source:
        request: Any = None
        try:
            request = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {x}")))
            response = dispatch(request, backend)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            request_id = request.get("id") if isinstance(request, dict) else None
            if isinstance(request_id, bool) or not isinstance(request_id, int):
                request_id = None
            response = {
                "id": request_id, "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        sink.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
        sink.flush()
        if response.get("ok") and request["op"] == "shutdown":
            break


def self_test() -> None:
    """Exercise shape, finite, error isolation and JSON framing on CPU only."""
    assert "torch" not in sys.modules and "curobo" not in sys.modules
    assert _vector([0] * DOF, DOF, "q") == [0.0] * DOF
    assert len(_matrix([[0] * DOF], DOF, "q")) == 1
    q_boundary = [[0] * DOF, [1] * DOF]
    for value in (q_boundary, [q_boundary], [[q_boundary]]):
        assert _trajectory_matrix(value, "q") == q_boundary
    for value in ([q_boundary, q_boundary], [[q_boundary, q_boundary]], []):
        try:
            _trajectory_matrix(value, "q")
        except ValueError:
            pass
        else:
            raise AssertionError("multiple batches/seeds or empty trajectory must fail")
    assert _single_value([[True]], "success") is True
    assert _diagnostic_json({"error": [float("inf"), float("nan"), -float("inf")]}) == {
        "error": ["Infinity", "NaN", "-Infinity"]}
    shuffled = [FINGER_JOINT_NAMES[1], "panda_joint3", "panda_joint1", FINGER_JOINT_NAMES[0],
                "panda_joint7", "panda_joint4", "panda_joint2", "panda_joint6", "panda_joint5"]
    locks = {FINGER_JOINT_NAMES[0]: 0.037, FINGER_JOINT_NAMES[1]: 0.039}
    native = {"joint_names": shuffled, "times_s": [0.0, 0.02, 0.04], "interpolation_dt": 0.02}
    native["position"] = [[locks[name] if name in locks else ARM_JOINT_NAMES.index(name) * 0.01 + step * 0.001
                           for name in shuffled] for step in range(3)]
    for field in ("velocity", "acceleration", "jerk"):
        native[field] = [[0.0 if name in locks else 0.1 for name in shuffled] for _ in range(3)]
    active, mapping = extract_active_columns(native, locks)
    assert active["joint_names"] == ARM_JOINT_NAMES
    assert active["position"][0] == [i * 0.01 for i in range(DOF)]
    assert mapping["selected_column_indices"] == [shuffled.index(name) for name in ARM_JOINT_NAMES]
    assert native["joint_names"] == shuffled and len(native["position"][0]) == 9
    bad_natives = []
    for changed_names in ([*shuffled[:-1], shuffled[0]], [*shuffled[:-1], "unknown_joint"]):
        bad_natives.append({**native, "joint_names": changed_names})
    changed_positions = [row[:] for row in native["position"]]
    changed_positions[1][shuffled.index(FINGER_JOINT_NAMES[0])] += 2e-6
    bad_natives.append({**native, "position": changed_positions})
    changed_velocity = [row[:] for row in native["velocity"]]
    changed_velocity[1][shuffled.index(FINGER_JOINT_NAMES[1])] = 2e-6
    bad_natives.append({**native, "velocity": changed_velocity})
    bad_natives.append({**native, "times_s": [0.0, 0.02, 0.02]})
    for invalid_native in bad_natives:
        try:
            extract_active_columns(invalid_native, locks)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid physical full-DOF output must fail")
    with tempfile.TemporaryDirectory(prefix="tig-curobo-worker-selftest-") as directory:
        official, installed = Path(directory) / "official", Path(directory) / "installed"
        for target in (official, installed):
            (target / "_src").mkdir(parents=True)
            (target / "__init__.py").write_text("# synthetic source\n")
            (target / "_src" / "runtime.py").write_text("VALUE = 1\n")
        (installed / "_version.py").write_text("__version__ = 'synthetic'\n")
        identity = _compare_package(installed, official)
        assert identity["installed_file_count"] == 2
        assert "_version.py" in identity["generated_files_sha256"]
        (installed / "_src" / "runtime.py").write_text("VALUE = 2\n")
        try:
            _compare_package(installed, official)
        except RuntimeError:
            pass
        else:
            raise AssertionError("modified wheel source must be rejected")
        (installed / "_src" / "runtime.py").unlink()
        try:
            _compare_package(installed, official)
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing runtime source must be rejected")
    valid_goal = {"id": 1, "op": "plan_pose", "start": [0] * DOF,
                  "goal_pose": [0, 0, 0, 1, 0, 0, 0]}
    validate_request(valid_goal)
    rejected = [
        {**valid_goal, "id": True},
        {**valid_goal, "start": [0] * 6},
        {**valid_goal, "start": [False] * DOF},
        {**valid_goal, "start": [float("nan")] * DOF},
        {**valid_goal, "goal_pose": [0] * 7},
        {**valid_goal, "max_attempts": 5},
        {"id": 1, "op": "fk", "positions": []},
        {"id": 1, "op": "fk", "positions": [[0] * DOF, [0] * 6]},
        {"id": 1, "op": "configure", "robot_cfg": {}, "world": {}},
    ]
    for bad in rejected:
        try:
            validate_request(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid request was accepted: {bad}")
    backend = CuroboBackend()
    try:
        backend.fk({"positions": [[0] * DOF]})
    except RuntimeError:
        pass
    else:
        raise AssertionError("unconfigured FK must fail without initializing CUDA")

    class SyntheticBackend:
        def fk(self, request: dict[str, Any]) -> dict[str, Any]:
            print("synthetic vendor log: must not enter protocol")
            return {"positions": [[0, 0, 0] for _ in request["positions"]]}

        def plan_pose(self, request: dict[str, Any]) -> dict[str, Any]:
            return {"success": False, "status": "SYNTHETIC_FAILURE"}

    source = io.StringIO(
        '{"id":1,"op":"fk","positions":[[0,0,0,0,0,0,0]]}\n'
        '{"id":2,"op":"fk","positions":[]}\n'
        '{"id":3,"op":"fk","positions":[[NaN,0,0,0,0,0,0]]}\n'
        + json.dumps({**valid_goal, "id": 4}) + "\n"
        + '{"id":5,"op":"shutdown"}\n'
        + '{"id":6,"op":"shutdown"}\n'
    )
    sink, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(errors):
        serve(source, sink, SyntheticBackend())
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(responses) == 5
    assert [x["ok"] for x in responses] == [True, False, False, True, True]
    assert responses[3]["result"]["success"] is False
    assert "synthetic vendor log" in errors.getvalue()
    assert "synthetic vendor log" not in sink.getvalue()
    assert responses[0]["result"]["positions"] == [[0, 0, 0]]
    assert "torch" not in sys.modules and "curobo" not in sys.modules
    print(json.dumps({"self_test": "PASS", "dof": DOF,
                      "rejected_boundary_cases": len(rejected),
                      "native_dof": 9, "native_mapping_rejected_cases": len(bad_natives),
                      "json_responses": len(responses), "cuda_imported": False}))


def native_api_self_test() -> None:
    """Bind every invoked public signature against the installed V2 on CPU."""
    import torch
    import curobo
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU API check unexpectedly entered with CUDA initialized")
    request = {"robot_cfg": {"kinematics": {}, "load_dynamics": False}, "world": {}}
    kwargs = planner_kwargs(request, DeviceCfg(device=torch.device("cpu")))
    inspect.signature(MotionPlannerCfg.create).bind(**kwargs)
    inspect.signature(MotionPlanner.plan_pose).bind(object(), object(), object(), max_attempts=4)
    inspect.signature(MotionPlanner.compute_kinematics).bind(object(), object())
    inspect.signature(MotionPlanner.warmup).bind(object(), enable_graph=True, num_warmup_iterations=1)
    inspect.signature(Pose.from_list).bind([0, 0, 0, 1, 0, 0, 0], device_cfg=kwargs["device_cfg"])
    inspect.signature(GoalToolPose.from_poses).bind(
        {"panda_hand_tcp": object()}, ordered_tool_frames=["panda_hand_tcp"], num_goalset=1)
    inspect.signature(JointState.from_position).bind(object(), joint_names=ARM_JOINT_NAMES)
    identity = package_identity(Path(curobo.__file__).resolve())
    assert not torch.cuda.is_initialized()
    print(json.dumps({"native_api_self_test": "PASS", "curobo_version": curobo.__version__,
                      "bound_constructor_kwargs": sorted(kwargs),
                      "installed_source": identity, "cuda_initialized": False}, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-native-api", action="store_true",
                        help="also bind installed cuRobo signatures on CPU with --self-test")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.check_native_api:
            native_api_self_test()
        return
    if args.check_native_api:
        parser.error("--check-native-api requires --self-test")
    # Preserve a private descriptor for the protocol and route even C/CUDA
    # writes to fd 1 into stderr for the entire lifetime of the native backend.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with protocol, contextlib.redirect_stdout(sys.stderr):
        serve(sys.stdin, protocol, CuroboBackend())


if __name__ == "__main__":
    main()
