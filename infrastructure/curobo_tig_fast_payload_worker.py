#!/usr/bin/env python3
"""Cached, equivalent measured-state refresh for resident TIG cuRobo sessions.

Only two declared independent Panda prismatic joints may vary. Their original
URDF origin and axis are parsed once; locked fixed transforms, default arm
configuration and lock reconstruction values are copied through the existing
public Kinematics.update_kinematics_config API. Every collision sphere, limit,
SRDF pair, payload update, world update, readback and protocol check is retained.
No simulator module is imported or environment advanced.

Run --self-test without CUDA. --benchmark-snapshots reuses recorded states to
compare complete fast/previous updates on the SAME resident planner, including
native tensors, world state, FK and independent finger changes.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import copy, deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
import xml.etree.ElementTree as ET

from curobo_tig_payload_worker import (
    FINGER_LINKS, PAYLOAD_LINK, SOURCE_TABLE, PayloadBackend, PayloadSessionManager,
    _finite_json, _matrix, _tensor_json, _vector, canonical_hash, self_test as payload_self_test,
    serve_payload, validate_model, validate_state,
)
from curobo_tig_worker import FINGER_JOINT_NAMES


def _rpy_rotation(value: list[float]) -> list[list[float]]:
    roll, pitch, yaw = _vector(value, 3, "URDF origin RPY")
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
            [-sp, cp*sr, cp*cr]]


def load_finger_joints(path: Path) -> dict[str, dict[str, Any]]:
    """Parse exactly the supported URDF prismatic joints; keep source SHA."""
    if not path.is_absolute() or not path.is_file():
        raise ValueError("cached finger model requires an existing absolute URDF")
    tree = ET.parse(path).getroot()
    joints = {node.attrib["name"]: node for node in tree.findall("joint")}
    result = {}
    for name, expected_child in zip(FINGER_JOINT_NAMES, FINGER_LINKS, strict=True):
        node = joints[name]
        if node.attrib.get("type") != "prismatic" or node.find("mimic") is not None:
            raise ValueError("fast refresh supports only actual independent prismatic fingers")
        parent, child, origin, axis = node.find("parent"), node.find("child"), node.find("origin"), node.find("axis")
        if parent is None or child is None or parent.attrib.get("link") != "panda_hand" or child.attrib.get("link") != expected_child:
            raise ValueError("finger topology differs from the actual Panda model")
        if origin is None or axis is None:
            raise ValueError("finger origin and axis must be explicit in the actual URDF")
        xyz = _vector([float(x) for x in origin.attrib.get("xyz", "0 0 0").split()], 3, "URDF origin XYZ")
        rotation = _rpy_rotation([float(x) for x in origin.attrib.get("rpy", "0 0 0").split()])
        direction = _vector([float(x) for x in axis.attrib["xyz"].split()], 3, "URDF prismatic axis")
        if abs(sum(x*x for x in direction)-1.) > 1e-12:
            raise ValueError("URDF finger axis must have unit length")
        result[name] = {"child": expected_child, "origin_xyz": xyz, "rotation": rotation, "axis": direction}
    return result


def finger_transform(joint: dict[str, Any], position_m: float) -> list[list[float]]:
    """URDF T_parent_child(q)=T_origin @ Translation(axis*q), shape [3,4]."""
    q = _vector([position_m], 1, "measured finger position")[0]
    if not 0. <= q <= .04:
        raise ValueError("measured finger must lie in the actual [0,.04] metre limits")
    rotation, axis, origin = joint["rotation"], joint["axis"], joint["origin_xyz"]
    translation = [origin[i]+q*sum(rotation[i][j]*axis[j] for j in range(3)) for i in range(3)]
    result = [[*rotation[i], translation[i]] for i in range(3)]
    assert len(result) == 3 and all(len(row) == 4 for row in result)
    _finite_json(result)
    return result


class FastPayloadBackend(PayloadBackend):
    """Equivalent cached kinematics configuration with complete old receipts."""

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        result = super().configure(request)
        start = time.perf_counter()
        urdf = Path(request["robot_cfg"]["kinematics"]["urdf_path"])
        self.finger_urdf_path = urdf
        self.finger_joints = load_finger_joints(urdf)
        self.finger_urdf_sha256 = hashlib.sha256(urdf.read_bytes()).hexdigest()
        self.cached_params: dict[int, Any] = {}
        self.static_geometry: dict[int, tuple[Any, Any]] = {}
        self.static_limits: dict[int, dict[str, Any]] = {}
        self.initial_transform_error = 0.
        for model in self._models().values():
            params = model.config.kinematics_config
            if id(params) in self.cached_params:
                continue
            # KinematicsParams.clone() reconstructs the dataclass and rescales
            # limits in __post_init__. A shallow structural copy avoids that
            # extra scaling and clones exactly the three updateable tensors.
            cached = copy(params)
            cached.fixed_transforms = params.fixed_transforms.clone()
            cached.lock_jointstate = params.lock_jointstate.clone()
            cached.cspace = copy(params.cspace)
            cached.cspace.default_joint_position = params.cspace.default_joint_position.clone()
            for name, joint in self.finger_joints.items():
                index = params.link_name_to_idx_map[joint["child"]]
                if int(params.joint_map[index]) != -1 or int(params.joint_map_type[index]) != -1:
                    raise ValueError("native finger was not reduced to a locked fixed transform")
                measured = request["robot_cfg"]["kinematics"]["lock_joints"][name]
                analytic = self.tensor_args.to_device(finger_transform(joint, measured))
                error = float(self.torch.max(self.torch.abs(params.fixed_transforms[index]-analytic)).item())
                self.initial_transform_error = max(self.initial_transform_error, error)
                if error > 2e-6:
                    raise RuntimeError(f"cached URDF transform disagrees with official initialization: {name}: {error}")
            selected = params.get_sphere_index_from_link_name(PAYLOAD_LINK)
            ordinary = self.torch.ones(params.link_spheres.shape[1], dtype=self.torch.bool, device=params.link_spheres.device)
            ordinary[selected] = False
            self.static_geometry[id(params)] = ordinary, params.link_spheres[:, ordinary, :].clone()
            self.static_limits[id(params)] = {name: getattr(params.joint_limits, name).clone()
                for name in ("position", "velocity", "acceleration", "jerk")}
            self.cached_params[id(params)] = cached
        self.torch.cuda.synchronize()
        result["cached_state_update"] = {"enabled": True, "cache_build_wall_s": time.perf_counter()-start,
            "urdf_sha256": self.finger_urdf_sha256, "initial_native_transform_max_abs_error": self.initial_transform_error,
            "structural_cache_count": len(self.cached_params), "per_update_kinematics_config_rebuilds": 0,
            "new_planner_instances": 0, "limit_rescaling_performed": False}
        return result

    def _refresh_kinematics(self, robot_cfg: dict[str, Any]) -> int:
        locks = robot_cfg["kinematics"]["lock_joints"]
        default = robot_cfg["kinematics"]["cspace"]["default_joint_position"]
        full_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        seen: set[int] = set()
        for model in self._models().values():
            current = model.config.kinematics_config
            if id(current) in seen:
                continue
            cached = self.cached_params[id(current)]
            for name, joint in self.finger_joints.items():
                index = current.link_name_to_idx_map[joint["child"]]
                cached.fixed_transforms[index].copy_(self.tensor_args.to_device(finger_transform(joint, locks[name])))
            lock_values = self.tensor_args.to_device([locks[name] for name in cached.lock_jointstate.joint_names])
            cached.lock_jointstate.position.copy_(lock_values.reshape(cached.lock_jointstate.position.shape))
            defaults = self.tensor_args.to_device([default[full_names.index(name)] for name in cached.cspace.joint_names])
            cached.cspace.default_joint_position.copy_(defaults.reshape(cached.cspace.default_joint_position.shape))
            model.update_kinematics_config(cached)
            seen.add(id(current))
        return len(seen)

    def update(self, robot_cfg: dict[str, Any], world: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        self._require_configured()
        begin = previous = time.perf_counter()
        timings: dict[str, float] = {}

        def elapsed(label: str) -> None:
            nonlocal previous
            self.torch.cuda.synchronize()
            now = time.perf_counter()
            timings[label] = now-previous
            previous = now

        if hashlib.sha256(self.finger_urdf_path.read_bytes()).hexdigest() != self.finger_urdf_sha256:
            raise RuntimeError("cached original URDF contents changed after initialization")
        count = self._refresh_kinematics(robot_cfg)
        self.lock_joints = deepcopy(robot_cfg["kinematics"]["lock_joints"])
        elapsed("cached_kinematics")
        for shapes in world.values():
            for name, shape in shapes.items():
                self.planner.scene_collision_checker.update_obstacle_pose(
                    name, self._Pose.from_list(shape["pose"], device_cfg=self.tensor_args))
        elapsed("world_poses")
        attachment, manager = state["attachment"], self.planner.attachment_manager
        if attachment is None:
            manager.detach(link_name=PAYLOAD_LINK, enable_obstacle_names=self.payload_model["target_obstacle_names"])
            for model in self._models().values():
                model.config.kinematics_config.reset_link_spheres(PAYLOAD_LINK)
        else:
            joint_state = self._JointState.from_position(self.tensor_args.to_device([attachment["positions"]]), joint_names=self.joint_names)
            manager.update(self.tensor_args.to_device(attachment["spheres"]), joint_state, link_name=PAYLOAD_LINK,
                           world_objects_pose_offset=self._Pose.from_list(attachment["object_pose_base_wxyz"], device_cfg=self.tensor_args))
            local = manager.kinematics_params.get_link_spheres(PAYLOAD_LINK).clone()
            for model in self._models().values():
                model.config.kinematics_config.update_link_spheres(PAYLOAD_LINK, local)
        elapsed("payload_attachment")
        data = self.planner.scene_collision_checker.data
        targets = set(self.payload_model["target_obstacle_names"])
        for shapes in world.values():
            for name in shapes:
                data.set_obstacle_layers(name, 2 if name in targets else (4 if name == SOURCE_TABLE else 1))
                self.planner.scene_collision_checker.enable_obstacle(name, enable=not (attachment is not None and name in targets))
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
        elapsed("permissions_and_graph_reset")
        identity = self._storage_identity()
        if identity != self.initial_storage_identity:
            raise RuntimeError("fast update changed resident planner/model/world storage")
        for model in self._models().values():
            params = model.config.kinematics_config
            ordinary, geometry = self.static_geometry[id(params)]
            if not self.torch.equal(params.link_spheres[:, ordinary, :], geometry):
                raise RuntimeError("fast update changed ordinary robot collision geometry")
            if any(not self.torch.equal(getattr(params.joint_limits, name), expected)
                   for name, expected in self.static_limits[id(params)].items()):
                raise RuntimeError("fast update changed fixed profile joint limits")
        self.payload_state, self.measured_world = deepcopy(state), deepcopy(world)
        readback = self._readback()
        for name, values in readback["robot_models"].items():
            q, names = values["lock_joint_positions"], values["lock_joint_names"]
            while len(q) == 1 and isinstance(q[0], list):
                q = q[0]
            if any(abs(q[names.index(n)]-self.lock_joints[n]) > 1e-7 for n in FINGER_JOINT_NAMES):
                raise RuntimeError(f"{name} did not apply both actual finger locks")
        payload = self._payload_readback()
        disabled = {name for name, row in payload["world"].items() if not row["enabled"]}
        if disabled != (targets if attachment is not None else set()):
            raise RuntimeError("fast update disabled world geometry outside exact own held actor")
        elapsed("integrity_and_full_native_readback")
        return {"update_wall_s": time.perf_counter()-begin, "update_timing_s": timings,
                "storage_identity": identity, "native_readback": readback,
                "updated_unique_kinematics_params": count, "graph_nodes_cleared": self.planner.graph_planner is not None,
                "world_update_api": "scene_collision_checker.update_obstacle_pose/enable_obstacle + data.set_obstacle_layers",
                "kinematics_update_api": "cached URDF prismatic transforms + Kinematics.update_kinematics_config",
                "per_update_kinematics_config_rebuilds": 0, "urdf_sha256": self.finger_urdf_sha256,
                "initial_native_transform_max_abs_error": self.initial_transform_error, "payload": payload}


def self_test() -> None:
    payload_self_test()
    assert "torch" not in sys.modules and "curobo" not in sys.modules
    with tempfile.TemporaryDirectory(prefix="tig-fast-fingers-cpu-") as directory:
        path = Path(directory)/"panda.urdf"
        path.write_text('<robot name="synthetic">'+''.join(
            f'<joint name="{name}" type="prismatic"><parent link="panda_hand"/><child link="{child}"/>'
            f'<origin xyz="0 0 0.0584" rpy="0 0 0"/><axis xyz="0 {direction} 0"/></joint>'
            for name, child, direction in zip(FINGER_JOINT_NAMES, FINGER_LINKS, (1, -1), strict=True))+'</robot>')
        joints = load_finger_joints(path)
        for value in (0., .000001, .017, .04):
            left, right = [finger_transform(joints[name], value) for name in FINGER_JOINT_NAMES]
            assert len(left) == len(right) == 3 and all(len(row) == 4 for row in left+right)
            assert left[1][3] == value and right[1][3] == -value
            assert left[2][3] == right[2][3] == .0584
        # A rotated origin makes the axis transform essential.
        rotated = {"origin_xyz": [1., 2., 3.], "rotation": _rpy_rotation([0., 0., math.pi/2]), "axis": [1., 0., 0.]}
        result = finger_transform(rotated, .04)
        assert abs(result[0][3]-1.) < 1e-15 and abs(result[1][3]-2.04) < 1e-15
        for value in (-1e-10, .04000001, float("nan"), float("inf")):
            try:
                finger_transform(joints[FINGER_JOINT_NAMES[0]], value)
            except ValueError:
                continue
            raise AssertionError("invalid finger state accepted")
        path.write_text(path.read_text().replace('type="prismatic"', 'type="revolute"', 1))
        try:
            load_finger_joints(path)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported kinematic topology accepted")
    assert "torch" not in sys.modules and "curobo" not in sys.modules
    print(json.dumps({"fast_self_test": "PASS", "fixed_transform_shape": [2, 3, 4],
                      "checks": ["independent_prismatic_fingers", "zero_and_limit_positions", "rotated_URDF_origin",
                                 "nonfinite_and_out_of_range_rejection", "unsupported_topology_rejection",
                                 "inherited_full_payload_protocol_and_failure_tests", "no_GPU_import"]}))


def benchmark(snapshot_dir: Path, output: Path, gpu_uuid: str, arm: int) -> None:
    """Compare complete old/fast native updates on frozen actual-state inputs."""
    if not gpu_uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("benchmark requires matching explicit CUDA_VISIBLE_DEVICES UUID")
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"status": "running", "gpu_uuid": gpu_uuid, "arm_index": arm,
                              "environment_steps": 0, "simulator_imported": False, "stages": []}
    backend = FastPayloadBackend()

    def save() -> None:
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def native_state() -> dict[str, Any]:
        result = {}
        for name, model in backend._models().items():
            params = model.config.kinematics_config
            result[name] = {field: _tensor_json(getattr(params, field))
                for field in ("fixed_transforms", "link_spheres", "sphere_collision_mask")}
            result[name]["default_joint_position"] = _tensor_json(params.cspace.default_joint_position)
            result[name]["lock_joint_position"] = _tensor_json(params.lock_jointstate.position)
            result[name]["limits"] = {field: _tensor_json(getattr(params.joint_limits, field))
                                     for field in ("position", "velocity", "acceleration", "jerk")}
        return result

    def difference(a: Any, b: Any) -> float:
        if isinstance(a, dict):
            if set(a) != set(b):
                raise AssertionError("native comparison dictionary keys changed")
            return max([difference(a[k], b[k]) for k in a] or [0.])
        if isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                raise AssertionError("native comparison shape changed")
            return max([difference(x, y) for x, y in zip(a, b, strict=True)] or [0.])
        return abs(float(a)-float(b))

    try:
        initial_path = snapshot_dir/f"initial-arm-{arm}-snapshot.json"
        initial = json.loads(initial_path.read_text())
        report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("curobo_tig_payload_worker.py"),
                         Path(__file__).with_name("curobo_tig_session_worker.py"), Path(__file__).with_name("curobo_tig_worker.py"))}
        report["initialization"] = backend.configure({"robot_cfg": initial["robot_cfg"], "world": initial["world"],
            "payload_model": initial["payload_model"], "warmup": True, "seed": 401, "interpolation_dt": .02})
        save()
        for label in ("pregrasp", "descend", "lift_and_hold", "carry_orient_and_transport", "carry_lower", "released"):
            path = snapshot_dir/f"{label}-arm-{arm}-snapshot.json"
            snapshot = json.loads(path.read_text())
            validate_model(snapshot["payload_model"], snapshot["robot_cfg"], snapshot["world"])
            validate_state(snapshot["payload_state"], snapshot["payload_model"], snapshot["robot_cfg"])
            fast = backend.update(snapshot["robot_cfg"], snapshot["world"], snapshot["payload_state"])
            fast_state = native_state()
            fast_fk = backend.fk({"positions": [snapshot["q_start"]]})
            baseline = PayloadBackend.update(backend, snapshot["robot_cfg"], snapshot["world"], snapshot["payload_state"])
            baseline_state = native_state()
            baseline_fk = backend.fk({"positions": [snapshot["q_start"]]})
            error = difference(fast_state, baseline_state)
            fk_error = difference({k: fast_fk[k] for k in ("positions", "quaternions", "link_spheres")},
                                  {k: baseline_fk[k] for k in ("positions", "quaternions", "link_spheres")})
            row = {"phase": label, "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "fast_update_wall_s": fast["update_wall_s"], "fast_update_timing_s": fast["update_timing_s"],
                   "baseline_update_wall_s": baseline["update_wall_s"], "native_tensor_max_abs_error": error,
                   "native_fk_max_abs_error": fk_error,
                   "world_readback_identical": fast["payload"]["world"] == baseline["payload"]["world"],
                   "storage_identity_unchanged": fast["storage_identity"] == baseline["storage_identity"],
                   "complete_native_readback_preserved": True}
            report["stages"].append(row)
            (output/f"{label}-fast.json").write_text(json.dumps({"update": fast, "fk": fast_fk}, allow_nan=False)+"\n")
            (output/f"{label}-baseline.json").write_text(json.dumps({"update": baseline, "fk": baseline_fk}, allow_nan=False)+"\n")
            save()
            if error > 2e-6 or fk_error > 2e-6 or not row["world_readback_identical"] or not row["storage_identity_unchanged"]:
                raise AssertionError(f"fast/reference native equivalence failed: {label}: {row}")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        backend.shutdown()
        report["simulator_imported"] = any(name == "sapien" or name.startswith("mani_skill") for name in sys.modules)
        save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--serve", action="store_true")
    modes.add_argument("--benchmark-snapshots", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-uuid", default="")
    parser.add_argument("--arm", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.benchmark_snapshots is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required for benchmark")
        benchmark(args.benchmark_snapshots, args.output_dir, args.gpu_uuid, args.arm)
    else:
        protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with protocol, contextlib.redirect_stdout(sys.stderr):
            serve_payload(sys.stdin, protocol, PayloadSessionManager(backend_factory=FastPayloadBackend))


if __name__ == "__main__":
    main()
