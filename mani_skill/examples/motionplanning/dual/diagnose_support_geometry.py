"""Reconstruct archived banana/table geometry without advancing an environment.

Inputs: frozen per-seed H5 actor states (T+1,13), articulation states (T+1,31),
and original JSON reports. Actor state begins xyz,wxyz; articulated state begins
xyz,wxyz, linear/angular velocity, then nine qpos and nine qvel. Output is newly cooked asset
geometry at recorded poses, never a claim to recover unrecorded original meshes
or contacts. This module changes no planner, executor, gate, or archived file.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import sapien
import sapien.physx as physx
from mplib.pymp.collision_detection import fcl

from .scene_collision import _geometry
from .held_collision import _object_transform, _pair_result, _object_at, _transform
from .table_support_collision import NumericalTableSupportCollisionChecker

ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent
CHECKOUT = ROOT / "infrastructure/development/tig-dual-arm"
BATCH = ROOT / "infrastructure/validation/dual-arm/sync-compact-seeds-20261115T155130470922Z"
SEEDS = (1, 2, 4, 6, 9, 3)
TASK = "TwoRobotPickAppleBananaToBaskets-v1"
TABLE = "scene::table::shape_0"
PROBE_Z = (-1e-6, -1e-7, 1e-7, 1e-6)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def request_defaults() -> dict[str, Any]:
    """Binding exposes defaults in its signature, not readable instance fields."""
    signature = fcl.CollisionRequest.__init__.__doc__
    if "GST_LIBCCD" not in signature or "gjk_tolerance: float = 1e-06" not in signature:
        raise RuntimeError("Unexpected FCL default signature")
    return {"constructor_signature": signature, "defaults_from_binding_signature": {"gjk_solver_type": "GST_LIBCCD", "gjk_tolerance": 1e-6},
            "actual_query_parameters": {"num_max_contacts": 32, "enable_contact": True, "solver_and_tolerance_overridden": False}}


def pose(value: np.ndarray) -> sapien.Pose:
    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value[3:]), 1, atol=1e-6):
        raise ValueError("Expected finite xyz,wxyz pose with a unit quaternion")
    return sapien.Pose(value[:3], value[3:])


def vertices_receipt(shape: Any, actor_pose: sapien.Pose, obj: Any, table: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return independent world/table-local bounds for every cooked vertex."""
    if not isinstance(shape, physx.PhysxCollisionShapeConvexMesh):
        raise TypeError("Banana decomposition must contain only convex shapes")
    local = np.asarray(shape.vertices, dtype=np.float64) * np.asarray(shape.scale, dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 3 or len(local) < 4 or not np.isfinite(local).all():
        raise ValueError("Invalid convex vertex array")
    sf = np.asarray((actor_pose * shape.get_local_pose()).to_transformation_matrix(), dtype=float)
    ff = _object_transform(obj)
    tf = _object_transform(table)
    world_sapien = local @ sf[:3, :3].T + sf[:3, 3]
    world_fcl = local @ ff[:3, :3].T + ff[:3, 3]
    table_local = np.linalg.solve(tf[:3, :3], (world_fcl - tf[:3, 3]).T).T
    side = np.asarray(table.get_collision_geometry().side, dtype=float)
    assert np.allclose(tf[:3, 2], [0, 0, 1], rtol=0, atol=1e-6)
    top = float(tf[2, 3] + side[2] / 2)
    bounds = lambda a: [a.min(axis=0).tolist(), a.max(axis=0).tolist()]
    receipt = {"vertex_count": len(local), "scale": np.asarray(shape.scale).tolist(),
        "shape_local_pose_xyz_wxyz": np.r_[shape.get_local_pose().p, shape.get_local_pose().q].tolist(),
        "fcl_world_bounds_m": bounds(world_fcl), "sapien_world_bounds_m": bounds(world_sapien),
        "fcl_world_min_z_m": float(world_fcl[:, 2].min()), "fcl_world_max_z_m": float(world_fcl[:, 2].max()),
        "table_top_z_m": top, "minimum_vertex_minus_table_top_m": float(world_fcl[:, 2].min() - top),
        "vertex_penetration_into_infinite_top_halfspace_m": max(0., top - float(world_fcl[:, 2].min())),
        "table_local_xy_bounds_m": bounds(table_local[:, :2]), "table_half_xy_m": (side[:2] / 2).tolist(),
        "entire_vertex_footprint_within_table": bool(np.all(np.abs(table_local[:, :2]) <= side[:2] / 2)),
        "max_sapien_float32_vs_fcl_vertex_delta_m": float(np.abs(world_sapien - world_fcl).max()),
        "interpretation": "negative bottom gap establishes geometric plane penetration when the footprint lies inside the tabletop; small magnitude alone does not establish an FCL false positive"}
    return receipt, {"local_scaled_vertices": local, "triangles": np.asarray(shape.get_triangles(), dtype=np.int32),
                     "world_vertices_fcl": world_fcl, "world_vertices_sapien": world_sapien, "table_local_vertices": table_local}


def collision_object(shape: Any, actor_pose: sapien.Pose) -> Any:
    geometry, correction = _geometry(shape)
    composed = actor_pose * shape.get_local_pose() * correction
    return fcl.CollisionObject(geometry, np.asarray(composed.p, dtype=float), np.asarray(composed.q, dtype=float))


def query(first: Any, second: Any) -> dict[str, Any]:
    hit, contacts = _pair_result(first, second)
    diagnostics = []
    for c in contacts:
        normal = np.asarray(c["normal"], dtype=float)
        norm = float(np.linalg.norm(normal))
        diagnostics.append({"raw_normal_norm": norm, "normalized_normal_z": None if norm == 0 else float(normal[2] / norm),
                            "original_raw_abs_nz_ge_099": bool(abs(normal[2]) >= .99)})
    return {"collision": hit, "contacts": contacts, "contact_normal_diagnostics": diagnostics}


def checker(shapes: list[Any], actor_pose: sapien.Pose, objects: dict[str, Any], selected: list[str] | None = None) -> tuple[Any, dict[str, Any]]:
    """Call unchanged support methods on detached diagnostic bookkeeping only."""
    entity = sapien.Entity(); entity.pose = actor_pose
    names = [f"scene::banana_out::shape_{i}" for i in range(len(shapes))]
    c = NumericalTableSupportCollisionChecker()
    c.live = SimpleNamespace(_shape_refs={name: (entity, shape) for name, shape in zip(names, shapes)})
    c._target_names = frozenset(names if selected is None else selected)
    c._support_pairs = {}
    c._initial_object = _transform(np.r_[actor_pose.p, actor_pose.q])
    c.support_band_m, c.support_penetration_m, c.support_depth_slop_m = .003, .003, .0002
    c.support_released = False
    try:
        c._register_initial_support(objects)
        status, error = "registered", None
    except Exception as exc:
        status, error = "rejected", f"{type(exc).__name__}: {exc}"
    receipt = {"status": status, "error": error,
        "support_pairs": [{"target": a, "table": b, **copy.deepcopy(v)} for (a, b), v in c._support_pairs.items()],
        "numerical_top_support_receipts": copy.deepcopy(getattr(c, "numerical_top_support_receipts", []))}
    return c, receipt


def archived_states(seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import h5py
    directory = BATCH / f"seed-{seed:02d}"
    report = json.loads((directory / "report.json").read_text())
    ticks = [json.loads(line) for line in (directory / "ticks.jsonl").read_text().splitlines()]
    close = [t for t in ticks if t["phase"] == "close_and_confirm"]
    if not close: raise ValueError("No recorded close/attach boundary")
    attach = int(close[-1]["environment_step"])
    lift = next((int(t["environment_step"]) for t in ticks if t["phase"] == "lift_and_hold"), None)
    rows = []
    with h5py.File(directory / "trajectory.h5", "r") as h5:
        group = h5["traj_0/env_states"]
        for label, frame in (("coverage_anchor", 10), ("attach", attach), ("first_lift", lift)):
            if frame is None: continue
            actor = np.asarray(group["actors/banana_out"][frame]); table = np.asarray(group["actors/table"][frame])
            joints = [np.asarray(group[f"articulations/panda_wristcam-agent-{i}"][frame]) for i in (0, 1)]
            assert actor.shape == table.shape == (13,) and all(a.shape == (31,) for a in joints)
            rows.append({"label": label, "frame": frame, "banana_pose": actor[:7].tolist(), "table_pose": table[:7].tolist(),
                         "left_articulation_state": joints[0].tolist(), "right_articulation_state": joints[1].tolist(),
                         "left_qpos": joints[0][13:22].tolist(), "right_qpos": joints[1][13:22].tolist()})
    return report, rows


def run(output: Path) -> None:
    import gymnasium as gym
    import mani_skill.envs
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_l1_enabled, is_l2_enabled, is_l3_enabled
    protocol = json.loads((BATCH / "protocol.json").read_text())
    expected = protocol["source19_sha256"]
    source = {name: digest(HERE / name) for name in expected}
    if source != expected: raise RuntimeError("Frozen source19 differs before diagnosis")
    if any((is_l1_enabled(), is_l2_enabled(), is_l3_enabled())): raise RuntimeError("This diagnosis requires unchanged L0")
    output.mkdir(exist_ok=False)
    source_dir = output / "source"; source_dir.mkdir()
    for name in (*expected, Path(__file__).name): (source_dir / name).write_bytes((HERE / name).read_bytes())
    inputs = {str(path): digest(path) for seed in SEEDS for name in ("report.json", "trajectory.h5", "ticks.jsonl", "physics-substeps.jsonl")
              for path in [BATCH / f"seed-{seed:02d}" / name]}
    report: dict[str, Any] = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(), "seeds": list(SEEDS),
        "scope": "newly cooked diagnostic reconstruction of fixed archived poses, no actions or trajectory continuation",
        "original_cooked_geometry_was_recorded": False, "seed1_seed9_original_contact_normals_were_recorded": False,
        "source19_sha256": source, "runner_sha256": digest(Path(__file__)), "input_sha256_before": inputs,
        "task": TASK, "task_level": "L0", "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "step_guard": {"environment_step_calls": 0, "scene_step_calls": 0}, "reset_calls": 0, "actions_executed": 0,
        "physical_actor_state_restores": 0, "robot_state_restores": 0, "cases": [],
        "limitations": ["new cooking is not recovery of original cooked arrays", "query comparison uses archived contact anchors where available",
                        "FCL contact depth/normal and independent vertex-plane penetration are different measurements", "no threshold or runtime fix applied"]}
    report["fcl_request_defaults"] = request_defaults()
    write_json(output / "report.json", report)
    def forbidden_env(*args: Any, **kwargs: Any) -> Any:
        report["step_guard"]["environment_step_calls"] += 1
        raise RuntimeError("Zero-step diagnostic forbids BaseEnv.step")
    def forbidden_scene(*args: Any, **kwargs: Any) -> Any:
        report["step_guard"]["scene_step_calls"] += 1
        raise RuntimeError("Zero-step diagnostic forbids ManiSkillScene.step")
    env = None
    try:
        with patch.object(BaseEnv, "step", forbidden_env), patch.object(ManiSkillScene, "step", forbidden_scene):
            env = gym.make(TASK, obs_mode="none", reward_mode="none", control_mode="pd_joint_pos", sim_backend="cpu",
                           render_backend="gpu", render_mode="rgb_array", max_episode_steps=200)
            env.reset(seed=3); report["reset_calls"] += 1
            raw = env.unwrapped
            report["native_elapsed_steps_after_reset"] = int(raw.elapsed_steps[0])
            assert report["native_elapsed_steps_after_reset"] == 0
            shapes = list(raw.banana_out._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent).get_collision_shapes())
            table_body = raw.table_scene.table._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
            if table_body is None:
                table_body = next(c for c in raw.table_scene.table._objs[0].components if isinstance(c, physx.PhysxRigidBaseComponent))
            table_shapes = list(table_body.get_collision_shapes())
            if not shapes or not isinstance(table_shapes[0], physx.PhysxCollisionShapeBox): raise RuntimeError("Unexpected real task geometry")
            report["loaded_geometry"] = {"banana_convex_piece_count": len(shapes), "table_collision_shape_count": len(table_shapes),
                "table_top_half_size": np.asarray(table_shapes[0].half_size).tolist(),
                "table_top_local_pose": np.r_[table_shapes[0].get_local_pose().p, table_shapes[0].get_local_pose().q].tolist()}
            asset = Path(str(__import__("mani_skill").ASSET_DIR)) / "assets/mani_skill2_ycb/models/011_banana/collision.ply"
            report["banana_collision_asset"] = {"path": str(asset), "sha256": digest(asset)}
            for seed in SEEDS:
                original, states = archived_states(seed)
                case: dict[str, Any] = {"seed": seed, "original_status": original["status"], "original_error": original.get("error"),
                    "original_reason": original.get("reason"), "frames": [], "first_lift_available": any(r["label"] == "first_lift" for r in states),
                    "original_banana_support": original.get("attachment_initial_coverage", {}).get("support_checks", [{}, {}])[1]}
                initial_checker, captured_shape_local = None, None
                for row in states:
                    ap, tp = pose(row["banana_pose"]), pose(row["table_pose"])
                    objects = {TABLE: collision_object(table_shapes[0], tp)}
                    objects.update({f"scene::banana_out::shape_{i}": collision_object(shape, ap) for i, shape in enumerate(shapes)})
                    frame: dict[str, Any] = {**row, "pieces": [], "counterfactual_translation_probes": []}
                    for index, shape in enumerate(shapes):
                        name = f"scene::banana_out::shape_{index}"
                        geometry, arrays = vertices_receipt(shape, ap, objects[name], objects[TABLE])
                        np.savez(output / f"seed-{seed:02d}-{row['label']}-piece-{index:02d}.npz", **arrays)
                        _, isolated = checker(shapes, ap, objects, [name])
                        frame["pieces"].append({"name": name, **geometry, "banana_then_table": query(objects[name], objects[TABLE]),
                            "table_then_banana": query(objects[TABLE], objects[name]), "isolated_original_support_registration": isolated})
                    reconstructed, registered = checker(shapes, ap, objects)
                    frame["original_support_registration"] = registered
                    if row["label"] == "coverage_anchor":
                        saved = {s["name"]: s for s in original["initial_collision_coverage"]["shapes"] if "scene::banana_out::" in s["name"] or s["name"] == TABLE}
                        if set(saved) != set(objects): raise RuntimeError("Original coverage and reconstructed shape identities differ")
                        frame["coverage_anchor_transform_max_abs_error"] = {name: float(np.abs(_object_transform(obj) - np.asarray(pose(saved[name]["world_pose_wxyz"]).to_transformation_matrix(), dtype=float)).max()) for name, obj in objects.items()}
                    elif row["label"] == "attach":
                        initial_checker = reconstructed if registered["status"] == "registered" else None
                        captured_shape_local = {name: np.linalg.inv(_transform(row["banana_pose"])) @ _object_transform(obj)
                                                for name, obj in objects.items() if name != TABLE}
                        frame["diagnostic_captured_shape_local_T"] = {name: array.tolist() for name, array in captured_shape_local.items()}
                    elif row["label"] == "first_lift" and initial_checker is not None:
                        actual = _transform(row["banana_pose"])
                        touching = any(_pair_result(objects[a], objects[b])[0] for a, b in initial_checker._support_pairs)
                        dz = actual[2, 3] - initial_checker._initial_object[2, 3]
                        initial_checker.support_released = bool(initial_checker._support_pairs and (dz > initial_checker.support_band_m or not touching))
                        frame["support_refresh_reconstruction"] = {"inferred_support_released": initial_checker.support_released,
                            "original_first_lift_switch_was_recorded": False, "baseline_pair_still_touching_in_direct_live_geometry": touching,
                            "actor_vertical_displacement_m": float(dz), "interpretation": "reconstruction of old refresh rule, not an observed original switch"}
                        for piece in frame["pieces"]:
                            name = piece["name"]
                            recompose = _object_at(objects[name].get_collision_geometry(), actual @ captured_shape_local[name])
                            recomposed_geometry, arrays = vertices_receipt(shapes[int(name.rsplit("_", 1)[1])], ap, recompose, objects[TABLE])
                            np.savez(output / f"seed-{seed:02d}-{row['label']}-piece-{int(name.rsplit('_', 1)[1]):02d}-held-recomposition.npz", **arrays)
                            recomposed_pair = query(recompose, objects[TABLE])
                            piece["original_held_recomposition"] = {"geometry": recomposed_geometry,
                                "banana_then_table": recomposed_pair, "table_then_banana": query(objects[TABLE], recompose),
                                "max_transform_difference_from_direct_sapien": float(np.abs(_object_transform(recompose) - _object_transform(objects[name])).max())}
                            contacts = recomposed_pair["contacts"]
                            piece["original_baseline_pair_present"] = (piece["name"], TABLE) in initial_checker._support_pairs
                            piece["original_support_allowed"] = initial_checker._support_allowed(piece["name"], TABLE, actual, contacts)
                    if row["label"] != "coverage_anchor":
                        for dz in PROBE_Z:
                            shifted = sapien.Pose(np.asarray(ap.p) + [0, 0, dz], ap.q)
                            frame["counterfactual_translation_probes"].append({"label": "offline FCL-only actor translation; no physical actor changed", "delta_world_z_m": dz,
                                "pieces": [{"name": f"scene::banana_out::shape_{i}", "banana_then_table": query(collision_object(shape, shifted), objects[TABLE]),
                                            "table_then_banana": query(objects[TABLE], collision_object(shape, shifted))} for i, shape in enumerate(shapes)]})
                    case["frames"].append(frame)
                report["cases"].append(case)
                write_json(output / "report.json", report)
            report["native_elapsed_steps_after_queries"] = int(raw.elapsed_steps[0])
            assert report["native_elapsed_steps_after_queries"] == 0 and not any(report["step_guard"].values())
            env.close(); env = None
        report["input_sha256_after"] = {path: digest(Path(path)) for path in inputs}
        report["source19_sha256_after"] = {name: digest(HERE / name) for name in expected}
        if report["input_sha256_after"] != inputs or report["source19_sha256_after"] != source:
            raise RuntimeError("Source or input identity changed during diagnosis")
        report.update(status="completed", finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if env is not None: env.close()
        write_json(output / "report.json", report)


def self_test() -> dict[str, Any]:
    assert request_defaults()["defaults_from_binding_signature"]["gjk_tolerance"] == 1e-6
    material = physx.PhysxMaterial(.5, .5, 0.)
    cube = np.asarray([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)], dtype=np.float32)
    shape = physx.PhysxCollisionShapeConvexMesh(cube, [1., 2., 1.], material)
    table = fcl.CollisionObject(fcl.Box([1., 1., .1]), [0., 0., -.05])
    ap = sapien.Pose([0., 0., .0099])
    obj = collision_object(shape, ap)
    receipt, arrays = vertices_receipt(shape, ap, obj, table)
    assert receipt["entire_vertex_footprint_within_table"] and 9.99e-5 < receipt["vertex_penetration_into_infinite_top_halfspace_m"] < 1.001e-4
    assert arrays["world_vertices_fcl"].shape == (8, 3)
    assert query(obj, table)["collision"] and query(table, obj)["collision"]
    _, registration = checker([shape], ap, {TABLE: table, "scene::banana_out::shape_0": obj})
    assert registration["status"] == "registered" and len(registration["support_pairs"]) == 1
    above = collision_object(shape, sapien.Pose([0., 0., .02]))
    assert not query(above, table)["collision"]
    try: pose([0., 0., float("nan"), 1., 0., 0., 0.])
    except ValueError: pass
    else: raise AssertionError("Nonfinite archived pose accepted")
    return {"status": "passed", "gpu_used": False, "physics_steps": 0,
        "checks": ["actual binding request defaults read from signature", "actual SAPIEN convex scaled vertices vs FCL/table coordinates", "penetration and above-table separation", "both FCL query orders", "unchanged baseline registration", "malformed pose rejection"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    output = args.output_dir or ROOT / "infrastructure/validation/dual-arm" / datetime.now(timezone.utc).strftime("support-geometry-%Y%m%dT%H%M%S%fZ")
    run(output)
    print(json.dumps({"status": "completed", "output": str(output)})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
