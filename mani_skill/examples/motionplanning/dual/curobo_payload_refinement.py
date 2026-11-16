#!/usr/bin/env python3
"""Isolated CPU candidate for the three fixed TIG tasks' payload geometry.

This file is deliberately outside the runtime import tree. It accepts a complete
actual Task snapshot and returns a deep copy. Apples keep every value unchanged.
Pills retain the 25 mm grid except near the whole actor's +Y support plane:
orthogonally rotate +Y to +Z, use the frozen complete-solid LP cover with a 2 mm
plane envelope and at most three transverse subdivisions, then rotate back.
Fruits use the same frozen complete-solid algorithm with a 12.5 mm grid.

Contracts: shape vertices are finite [M,3] metres in each physical shape frame;
poses are XYZ/WXYZ; payload spheres are positive [N,4] actor-local XYZ/radius.
The original grid partitions each convex solid, six LP extrema enclose the full
cell intersection, and children partition the ORIGINAL cell. Orthogonal rotation
preserves coverage and radius. The 2 mm bound concerns ONE support plane, not a
global Euclidean/Hausdorff bound. No seed, hand pose, collision threshold, or
physical state participates in choosing or deleting spheres. Reserved inactive
slots change only in count. Every non-payload value and permission is preserved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "infrastructure/development/tig-dual-arm").is_dir())
DEV = ROOT / "infrastructure/development/tig-dual-arm"
for directory in (DEV, ROOT / "infrastructure"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mani_skill.examples.motionplanning.dual.curobo_payload import (  # noqa: E402
    PAYLOAD_LINK, tight_convex_spheres,
)
from mani_skill.examples.motionplanning.dual.curobo_task_payload import (  # noqa: E402
    _validate_task_snapshot, task_profile_config,
)
from curobo_tig_payload_worker import (  # noqa: E402
    canonical_hash, validate_model, validate_state,
)
from curobo_tig_task_payload_worker import (  # noqa: E402
    TASK_IDS, TaskSessionManager, dispatch_task, legacy_model, legacy_state,
    validate_task_request, validation_view,
)

RECIPE = "isolated-fixed-task-payload-refinement-v1"
BASE_POLICY = "full_convex_solid_cell_intersection_six_LP_tight_AABB_circumspheres_v1"
TARGETS = {TASK_IDS[0]: ("apple_out", "banana_out"),
           TASK_IDS[1]: ("080_pillbottle_1", "080_pillbottle_2"),
           TASK_IDS[2]: ("103_fruit_4", "103_fruit_5")}
Y_TO_Z = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
PILL_PADDING_M = .002
PILL_MAX_DEPTH = 3
EVIDENCE = ROOT / "infrastructure/validation/dual-arm/curobo-complete-20261116"


def _array(value: Any, columns: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}") from error
    if (result.ndim != 2 or result.shape[1] != columns or len(result) == 0
            or not np.isfinite(result).all()):
        raise ValueError(f"{name} must be finite nonempty [N,{columns}]")
    return result


def _pose_matrix(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if (pose.shape != (7,) or not np.isfinite(pose).all()
            or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6, rtol=0)):
        raise ValueError("Expected finite XYZ/WXYZ normalized pose[7]")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    result[:3, 3] = pose[:3]
    assert result.shape == (4, 4) and np.isfinite(result).all()
    return result


def _transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert transform.shape == (4, 4) and np.isfinite(transform).all()
    result = vertices @ transform[:3, :3].T + transform[:3, 3]
    assert result.shape == vertices.shape and np.isfinite(result).all()
    return result


def _components(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    model = snapshot["payload_model"]
    names, proofs = model["target_obstacle_names"], model["shape_coverage"]
    if (not isinstance(proofs, list) or len(proofs) != len(names)
            or any(not isinstance(proof, dict) for proof in proofs)
            or [proof.get("world_shape_name") for proof in proofs] != names):
        raise ValueError("Shape proofs must preserve every exact target shape and order")
    result = []
    for name, proof in zip(names, proofs, strict=True):
        if proof.get("complete_solid_coverage") is not True or proof.get("coverage_policy") != BASE_POLICY:
            raise ValueError("Only the existing complete convex-solid coverage is accepted")
        mesh = snapshot["world"].get("mesh", {}).get(name)
        if not isinstance(mesh, dict):
            raise ValueError("Each actual target must retain its complete collision mesh")
        vertices = _array(mesh.get("vertices"), 3, "actual collision vertices")
        faces = np.asarray(mesh.get("faces"))
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Actual target mesh needs in-bounds integer triangle indices")
        ConvexHull(vertices)
        transform = _pose_matrix(proof.get("shape_local_pose_wxyz"))
        actor_vertices = _transform(vertices, transform)
        result.append({"name": name, "proof": proof, "vertices": vertices,
                       "faces": faces,
                       "actor_vertices": actor_vertices, "transform": transform,
                       "mesh_sha256": canonical_hash({"vertices": mesh["vertices"],
                           "faces": mesh["faces"],
                           "shape_local_pose_wxyz": proof["shape_local_pose_wxyz"]})})
    return result


def _request(snapshot: dict[str, Any], operation: str, profile: str = "approach",
             token: str = "candidate-cpu") -> dict[str, Any]:
    model = snapshot["payload_model"]
    side = "left" if snapshot["arm_index"] == 0 else "right"
    request = {"id": 1, "op": operation, "session_id": f"{side}:{profile}",
               "robot_cfg": task_profile_config(snapshot["robot_cfg"], model["task_id"], profile),
               "world": deepcopy(snapshot["world"])}
    if operation == "init_session":
        request.update(payload_model=deepcopy(model), seed=401, interpolation_dt=.02, warmup=True)
    else:
        request.update(payload_state=deepcopy(snapshot["payload_state"]), snapshot_id=token)
    return request


def _validate_snapshot(snapshot: dict[str, Any]) -> tuple[str, int]:
    if not isinstance(snapshot, dict):
        raise ValueError("Expected a complete actual Task snapshot mapping")
    try:
        json.dumps(snapshot, allow_nan=False)
        model, state, robot = snapshot["payload_model"], snapshot["payload_state"], snapshot["robot_cfg"]
        coverage = snapshot.get("coverage")
        if (not isinstance(coverage, dict) or not isinstance(coverage.get("payload"), dict)
                or coverage["payload"].get("all_actual_target_shapes_covered") is not True):
            raise ValueError("Complete physical-target coverage metadata is required")
        task, arm = model["task_id"], snapshot["arm_index"]
        if task not in TARGETS or type(arm) is not int or arm not in (0, 1):
            raise ValueError("Only the three fixed task IDs and physical arms 0/1 are accepted")
        if model["target_actor_name"] != TARGETS[task][arm]:
            raise ValueError("Target actor differs from the fixed task/arm model")
        _validate_task_snapshot(snapshot, task, arm)
        _pose_matrix(snapshot["base_pose_wxyz"])
        _pose_matrix(snapshot["tcp_pose_base_wxyz"])
        q = np.asarray(snapshot["q_start"], dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected finite q_start[7]")
        spheres = _array(model["spheres_actor_local"], 4, "payload spheres")
        capacity = model["capacity"]
        if (type(capacity) is not int or not 1 <= len(spheres) <= capacity <= 32768
                or type(model["sphere_count"]) is not int or model["sphere_count"] != len(spheres)
                or np.any(spheres[:, 3] <= 0)):
            raise ValueError("Payload count/capacity/radii are inconsistent")
        if model["geometry_sha256"] != canonical_hash({"spheres": model["spheres_actor_local"],
                                                       "coverage": model["shape_coverage"]}):
            raise ValueError("Payload geometry hash is stale")
        kin = robot["kinematics"]
        if kin["collision_spheres"][PAYLOAD_LINK] != [
                {"center": [0., 0., 0.], "radius": -100.} for _ in range(capacity)]:
            raise ValueError("Reserved slots must be the unchanged inactive XYZ/radius entries")
        validate_model(legacy_model(model), robot, snapshot["world"])
        validate_state(legacy_state(state), legacy_model(model), robot)
        for operation in ("init_session", "update_session"):
            validate_task_request(_request(snapshot, operation), task)
        return task, arm
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed actual Task snapshot") from error


def _fixed_recipe(task: str) -> dict[str, Any]:
    """Return fresh immutable-in-use parameters, with no measured state fields."""
    if task not in (TASK_IDS[1], TASK_IDS[2]):
        raise ValueError("Only pill/fruit geometry uses the candidate cache")
    return {"recipe": RECIPE, "base_policy": BASE_POLICY, "task_id": task,
        "pitch_m": .025 if task == TASK_IDS[1] else .0125,
        "actor_to_cover_rotation": Y_TO_Z.tolist(),
        "support_plane_padding_m": PILL_PADDING_M,
        "maximum_refinement_depth": PILL_MAX_DEPTH if task == TASK_IDS[1] else 2}


def _geometry_uncached(components: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build actor-local [N,4] spheres; the fixed policy depends on geometry only."""
    task = spec["task_id"]
    cover_rotation = np.asarray(spec["actor_to_cover_rotation"], dtype=np.float64)
    padding, depth, pitch = (spec["support_plane_padding_m"],
                            spec["maximum_refinement_depth"], spec["pitch_m"])
    union_max_y = max(float(item["actor_vertices"][:, 1].max()) for item in components)
    rows, proofs = [], []
    for item in components:
        if task == TASK_IDS[1]:
            vertices = item["actor_vertices"] @ cover_rotation.T
            offset = union_max_y + padding - float(vertices[:, 2].max())
            spheres, proof = tight_convex_spheres(vertices, pitch,
                positive_z_padding_m=offset, max_depth=depth)
            spheres[:, :3] = spheres[:, :3] @ cover_rotation
            recipe = {"recipe": spec["recipe"], "task_id": task,
                "policy": "whole_actor_positive_Y_support_plane_envelope",
                "source_frame": "physical_actor_local", "output_frame": "physical_actor_local",
                "actor_to_cover_rotation": cover_rotation.tolist(), "normal_actor": [0., 1., 0.],
                "whole_actor_support_m": union_max_y, "support_plane_padding_m": padding,
                "absolute_support_limit_m": union_max_y + padding,
                "maximum_allowed_refinement_depth": depth,
                "subdivision_axes_actor": ["X", "Z"], "mesh_sha256": item["mesh_sha256"],
                "geometry_only": True, "seed_or_hand_dependent": False,
                "envelope_scope": "one support halfspace; not a global Hausdorff bound"}
            if np.max(spheres[:, 1] + spheres[:, 3]) > union_max_y + padding + 1e-12:
                raise ValueError("Fixed whole-actor +Y support-plane envelope was not met")
        else:
            spheres, proof = tight_convex_spheres(item["vertices"], pitch, max_depth=depth)
            spheres[:, :3] = _transform(spheres[:, :3], item["transform"])
            recipe = {"recipe": spec["recipe"], "task_id": task,
                "policy": "full_convex_solid_12_5mm_grid", "source_frame": "physical_shape_local",
                "output_frame": "physical_actor_local", "mesh_sha256": item["mesh_sha256"],
                "geometry_only": True, "seed_or_hand_dependent": False}
        proof.update(world_shape_name=item["name"],
            geometry_policy=item["proof"].get("geometry_policy"),
            shape_local_pose_wxyz=deepcopy(item["proof"]["shape_local_pose_wxyz"]),
            adaptive_rule_task_independent=True, candidate_refinement=recipe)
        rows.append(spheres)
        proofs.append(proof)
    result = np.vstack(rows)
    assert result.ndim == 2 and result.shape[1] == 4 and np.isfinite(result).all()
    assert np.all(result[:, 3] > 0)
    return result, proofs


def _geometry_cache_key(components: list[dict[str, Any]], spec: dict[str, Any]) -> str:
    """Canonical finite JSON of LOCAL geometry and fixed recipe, never state.

    Float64/int64 normalization, sorted mapping keys and compact separators give
    a reproducible UTF-8 SHA256 identity. Vertex/face ordering remains explicit.
    Shape/actor/world names, world poses, fingers, attachments and permissions are
    intentionally absent. Caller-dependent shape names are restored after lookup.
    """
    if (set(spec) != set(_fixed_recipe(TASK_IDS[1])) or spec["task_id"] not in TASK_IDS[1:]
            or not isinstance(spec["recipe"], str) or not spec["recipe"]
            or spec["base_policy"] != BASE_POLICY
            or type(spec["maximum_refinement_depth"]) is not int
            or not 0 <= spec["maximum_refinement_depth"] <= 8
            or not math.isfinite(spec["pitch_m"]) or spec["pitch_m"] <= 0
            or not math.isfinite(spec["support_plane_padding_m"]) or spec["support_plane_padding_m"] <= 0
            or not np.array_equal(np.asarray(spec["actor_to_cover_rotation"]), Y_TO_Z)):
        raise ValueError("Invalid pure geometry recipe")
    local = []
    for item in components:
        vertices = _array(item["vertices"], 3, "cache local vertices")
        faces = np.asarray(item["faces"])
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Cache requires actual in-bounds integer triangle indices")
        pose = np.asarray(item["proof"]["shape_local_pose_wxyz"], dtype=np.float64)
        _pose_matrix(pose)
        policy = item["proof"].get("geometry_policy")
        if not isinstance(policy, str) or not policy:
            raise ValueError("A scalar physical geometry provenance is required")
        local.append({"vertices": vertices.tolist(), "faces": faces.astype(np.int64).tolist(),
                      "shape_local_pose_wxyz": pose.tolist(), "geometry_policy": policy,
                      "mesh_sha256": item["mesh_sha256"]})
    return json.dumps({"schema": "actor-local-geometry-cache-v1", "recipe": spec,
                       "components": local}, sort_keys=True, allow_nan=False, separators=(",", ":"))


@lru_cache(maxsize=64)
def _cached_geometry(key: str) -> tuple[tuple[tuple[float, ...], ...], str,
                                        tuple[tuple[float, ...], ...]]:
    """Cache ONLY immutable positive local spheres/proof JSON and baseline balls."""
    data = json.loads(key)
    components, baseline = [], []
    for local in data["components"]:
        vertices = _array(local["vertices"], 3, "cached local vertices")
        transform = _pose_matrix(local["shape_local_pose_wxyz"])
        components.append({"name": "", "vertices": vertices,
            "actor_vertices": _transform(vertices, transform), "transform": transform,
            "mesh_sha256": local["mesh_sha256"], "proof": {
                "geometry_policy": local["geometry_policy"],
                "shape_local_pose_wxyz": local["shape_local_pose_wxyz"]}})
        original, _ = tight_convex_spheres(vertices, .025)
        original[:, :3] = _transform(original[:, :3], transform)
        baseline.append(original)
    spheres, proofs = _geometry_uncached(components, data["recipe"])
    return (tuple(tuple(float(v) for v in row) for row in spheres),
            json.dumps(proofs, sort_keys=True, allow_nan=False, separators=(",", ":")),
            tuple(tuple(float(v) for v in row) for row in np.vstack(baseline)))


def _geometry_bundle(components: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    """Every lookup returns fresh arrays/dicts; nothing mutable escapes the LRU."""
    key = _geometry_cache_key(components, spec)
    rows, proof_json, baseline = _cached_geometry(key)
    spheres = np.array(rows, dtype=np.float64, copy=True)
    proofs = json.loads(proof_json)
    for item, proof in zip(components, proofs, strict=True):
        proof["world_shape_name"] = item["name"]
        proof["shape_local_pose_wxyz"] = deepcopy(item["proof"]["shape_local_pose_wxyz"])
    return spheres, proofs, np.array(baseline, dtype=np.float64, copy=True)


def _geometry(components: list[dict[str, Any]], task: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    spheres, proofs, _ = _geometry_bundle(components, _fixed_recipe(task))
    return spheres, proofs


def _nonpayload_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(snapshot)
    model = result["payload_model"]
    for key in ("capacity", "sphere_count", "spheres_actor_local", "geometry_sha256", "shape_coverage", "sphere_pitch_m"):
        model.pop(key)
    kin = result["robot_cfg"]["kinematics"]
    kin["collision_spheres"].pop(PAYLOAD_LINK)
    kin["extra_collision_spheres"].pop(PAYLOAD_LINK)
    if result["payload_state"]["attachment"] is not None:
        result["payload_state"]["attachment"].pop("spheres")
    coverage = result["coverage"]["payload"]
    for key in ("reserved_sphere_capacity", "sphere_count", "payload_sphere_pitch_m",
                "payload_sphere_policy", "candidate_refinement"):
        coverage.pop(key, None)
    return result


def refine_task_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return a validated complete new snapshot; never mutate the input or runtime.

    Accept the frozen baseline payload geometry or this exact candidate's output.
    Stale sphere/proof/mesh identities fail closed. A candidate already refined by
    this recipe is idempotent. Actual state changes are carried through unchanged;
    initialized session model/topology guards remain responsible for rejecting a
    subsequently changed physical geometry or recipe.
    """
    task, _ = _validate_snapshot(snapshot)
    components = _components(snapshot)
    result = deepcopy(snapshot)
    if task == TASK_IDS[0]:
        assert result == snapshot and result is not snapshot
        return result
    model = snapshot["payload_model"]
    tagged = any("candidate_refinement" in proof for proof in model["shape_coverage"])
    spheres, proofs, baseline = _geometry_bundle(components, _fixed_recipe(task))
    if not tagged:
        if model["sphere_pitch_m"] != .025:
            raise ValueError("Expected the frozen 25mm baseline or this exact candidate")
        if not np.array_equal(baseline, np.asarray(model["spheres_actor_local"])):
            raise ValueError("Actual target mesh differs from the baseline frozen payload model")
    target = result["payload_model"]
    target.update(capacity=len(spheres), sphere_count=len(spheres), spheres_actor_local=spheres.tolist(),
        shape_coverage=proofs, sphere_pitch_m=.025 if task == TASK_IDS[1] else .0125,
        geometry_sha256=canonical_hash({"spheres": spheres.tolist(), "coverage": proofs}))
    kin = result["robot_cfg"]["kinematics"]
    kin["collision_spheres"][PAYLOAD_LINK] = [
        {"center": [0., 0., 0.], "radius": -100.} for _ in range(len(spheres))]
    kin["extra_collision_spheres"][PAYLOAD_LINK] = len(spheres)
    if result["payload_state"]["attachment"] is not None:
        result["payload_state"]["attachment"]["spheres"] = spheres.tolist()
    result["coverage"]["payload"].update(reserved_sphere_capacity=len(spheres), sphere_count=len(spheres),
        payload_sphere_pitch_m=target["sphere_pitch_m"], payload_sphere_policy=BASE_POLICY,
        candidate_refinement={"recipe": RECIPE, "isolated_candidate": True,
            "native_validation": "NOT_RUN", "geometry_only": True, "seed_or_hand_dependent": False})
    if tagged and result != snapshot:
        raise ValueError("Existing candidate recipe/geometry/proof identity is stale or malformed")
    assert _nonpayload_view(result) == _nonpayload_view(snapshot)
    _validate_snapshot(result)
    return result


def _body_coverage(components: list[dict[str, Any]], spheres: np.ndarray) -> dict[str, Any]:
    """Independent dense SOLID points, beyond the necessary vertex-only check."""
    count, maximum = 0, -math.inf
    rng = np.random.default_rng(90210)
    for item in components:
        vertices = item["actor_vertices"]
        hull = ConvexHull(vertices)
        axes = [np.linspace(vertices[:, axis].min(), vertices[:, axis].max(), 17) for axis in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        grid = grid[np.all(grid @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-12, axis=1)]
        weights = rng.exponential(size=(1024, len(vertices)))
        points = np.vstack((vertices, grid, weights @ vertices / weights.sum(axis=1, keepdims=True)))
        assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
        for start in range(0, len(points), 512):
            gaps = np.linalg.norm(points[start:start + 512, None] - spheres[None, :, :3], axis=2) - spheres[None, :, 3]
            maximum = max(maximum, float(gaps.min(axis=1).max()))
        count += len(points)
    assert count > 0 and maximum <= 1e-8, (count, maximum)
    return {"solid_points_tested": count, "maximum_uncovered_gap_m": maximum,
            "point_types": ["all source vertices", "17^3 bounding-grid interior points",
                            "1024 deterministic convex-combination interior points per shape"],
            "complete_volume_proof": "frozen six-LP grid-cell intersection cover; sampling supplements proof"}


class _CPUBackend:
    """Protocol-only double; provides no planning or physical grasp evidence."""
    def __init__(self) -> None:
        self.planner = object()
        self.lock_joints: dict[str, float] = {}

    def configure(self, request: dict[str, Any]) -> dict[str, Any]:
        self.lock_joints = deepcopy(request["robot_cfg"]["kinematics"]["lock_joints"])
        return {"cpu_protocol_double": True}

    def update(self, robot: dict[str, Any], world: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        self.lock_joints = deepcopy(robot["kinematics"]["lock_joints"])
        return {"cpu_protocol_double": True, "native_planning": "NOT_RUN"}

    def fk(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"cpu_protocol_double": True, "positions": deepcopy(request["positions"])}

    def shutdown(self) -> None:
        return None


def _synthetic_attachment(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Schema unit-test fixture, explicitly never an actual confirmed grasp."""
    result = deepcopy(snapshot)
    model = result["payload_model"]
    arm = result["arm_index"]
    result["payload_state"].update(contact_allowed=True, attachment={
        "spheres": deepcopy(model["spheres_actor_local"]),
        "object_pose_base_wxyz": [0., 0., 0., 1., 0., 0., 0.],
        "positions": deepcopy(result["q_start"]), "obstacle_names": deepcopy(model["target_obstacle_names"]),
        "evidence_id": "CPU_SYNTHETIC_NOT_AN_ACTUAL_GRASP", "physical_inertial": deepcopy(model["physical_inertial"]),
        "tcp_to_object_wxyz": [0., 0., 0., 1., 0., 0., 0.],
        "grasp_confirmation": {"stable_steps_required": 5, "stable_steps_observed": [5, 5],
            "physical_arm_index": arm, "confirmation_report_sha256": canonical_hash({"synthetic": True}),
            "synthetic_cpu_schema_fixture": True}})
    return result


def self_test() -> dict[str, Any]:
    """Actual six snapshots, four old goals, volume tests and strict CPU protocol."""
    started = time.perf_counter()
    snapshots, refined, records, rejected = {}, {}, [], []
    cache_checks = []
    _cached_geometry.cache_clear()

    def reject(label: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except (ValueError, RuntimeError):
            rejected.append(label)
        else:
            raise AssertionError(f"Negative test was accepted: {label}")

    for task in TASK_IDS:
        for arm in (0, 1):
            path = EVIDENCE / f"task-source-snapshots-dev02/{task}-arm-{arm}.json"
            original = json.loads(path.read_text())
            before = canonical_hash(original)
            start = time.perf_counter()
            cache_before = _cached_geometry.cache_info()
            new = refine_task_snapshot(original)
            cold_wall = time.perf_counter() - start
            cache_after_first = _cached_geometry.cache_info()
            assert canonical_hash(original) == before
            assert new is not original and _nonpayload_view(new) == _nonpayload_view(original)
            second_start = time.perf_counter()
            assert refine_task_snapshot(original) == new
            second_wall = time.perf_counter() - second_start
            cache_after_second = _cached_geometry.cache_info()
            if task != TASK_IDS[0]:
                assert cache_after_first.misses == cache_before.misses + 1
                assert cache_after_second.misses == cache_after_first.misses
                assert cache_after_second.hits == cache_after_first.hits + 1
            warm_start = time.perf_counter()
            assert refine_task_snapshot(new) == new
            idempotent_wall = time.perf_counter() - warm_start
            attached = _synthetic_attachment(original)
            attached_new = refine_task_snapshot(attached)
            assert attached_new["payload_state"]["attachment"]["spheres"] == new["payload_model"]["spheres_actor_local"]
            assert _nonpayload_view(attached_new) == _nonpayload_view(attached)
            components = _components(new)
            spheres = np.asarray(new["payload_model"]["spheres_actor_local"])
            points = _body_coverage(components, spheres)
            if task != TASK_IDS[0]:
                pristine = deepcopy(new)
                # Mutating either a complete returned snapshot or the direct
                # geometry API cannot pollute immutable cached values.
                new["payload_model"]["spheres_actor_local"][0][0] += .123
                new["payload_model"]["shape_coverage"][0]["candidate_refinement"]["recipe"] = "pollution"
                assert refine_task_snapshot(original) == pristine
                array, proof = _geometry(components, task)
                array[0, 0] += 99.
                proof[0]["candidate_refinement"]["recipe"] = "pollution"
                array2, proof2 = _geometry(components, task)
                assert np.array_equal(array2, spheres) and proof2 == pristine["payload_model"]["shape_coverage"]
                new = pristine
                hits_before_state = _cached_geometry.cache_info()
                # Change a peer WORLD pose; source manifest remains frozen.
                world_changed = deepcopy(original)
                world_changed["world"]["mesh"]["peer::panda_hand::shape_0"]["pose"][0] += .013
                # Change both measured finger locks and matching defaults.
                fingers_changed = deepcopy(original)
                finger_kin = fingers_changed["robot_cfg"]["kinematics"]
                for name, value in (("panda_finger_joint1", .031), ("panda_finger_joint2", .027)):
                    finger_kin["lock_joints"][name] = value
                    finger_kin["cspace"]["default_joint_position"][finger_kin["cspace"]["joint_names"].index(name)] = value
                attached_changed = _synthetic_attachment(original)
                attached_changed["payload_state"].update(source_support_allowed=True,
                    source_support_obstacle_names=deepcopy(original["payload_model"]["source_obstacle_names"]))
                for state_label, variant in (("world", world_changed), ("fingers", fingers_changed),
                                              ("attachment_and_support_permission", attached_changed)):
                    value = refine_task_snapshot(variant)
                    assert value["payload_model"] == pristine["payload_model"]
                    assert _nonpayload_view(value) == _nonpayload_view(variant)
                    assert value["world"] == variant["world"]
                    assert value["robot_cfg"]["kinematics"]["lock_joints"] == variant["robot_cfg"]["kinematics"]["lock_joints"]
                    assert value["payload_state"]["source_support_allowed"] == variant["payload_state"]["source_support_allowed"]
                    cache_checks.append(f"{task}/{arm}/{state_label}_fresh_state_preserved")
                info = _cached_geometry.cache_info()
                assert info.misses == hits_before_state.misses and info.hits == hits_before_state.hits + 3
                cache_checks.append(f"{task}/{arm}/returned_snapshot_array_and_proof_pollution_isolated")
            for profile in ("approach", "transport"):
                for view in (new, attached_new):
                    for operation in ("init_session", "update_session"):
                        request = _request(view, operation, profile)
                        validate_task_request(request, task)
                        # The Task function invokes the unchanged legacy validator
                        # using its own strictly checked normalized CPU view.
                        from curobo_tig_payload_worker import validate_request
                        validate_request(validation_view(request, task))
            if task == TASK_IDS[0]:
                assert new == original
            snapshots[task, arm], refined[task, arm] = original, new
            records.append({"task_id": task, "arm": arm, "input_path": str(path),
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "original_sphere_count": original["payload_model"]["sphere_count"],
                "candidate_sphere_count": len(spheres), "geometry_sha256": new["payload_model"]["geometry_sha256"],
                "apple_exact_identity": new == original if task == TASK_IDS[0] else None,
                "nonpayload_identity": "passed", "input_immutable": "passed",
                "idempotent": "passed", "cold_CPU_wall_s": cold_wall,
                "second_same_input_CPU_wall_s": second_wall,
                "geometry_cache_first_call_misses": cache_after_first.misses - cache_before.misses,
                "geometry_cache_second_call_hits": cache_after_second.hits - cache_after_first.hits,
                "idempotent_cached_CPU_wall_s": idempotent_wall, "body_coverage": points})

    # Small solids and rotated shape frames exercise dimension and partition boundaries.
    for vertices in (np.array([[x, y, z] for x in (0., .001) for y in (0., .001) for z in (0., .001)]),
                     np.array([[0., 0., 0.], [.03, 0., 0.], [0., .016, 0.], [0., 0., .02]])):
        component = {"name": "synthetic", "vertices": vertices,
            "faces": ConvexHull(vertices).simplices,
            "actor_vertices": vertices, "transform": np.eye(4), "mesh_sha256": canonical_hash(vertices.tolist()),
            "proof": {"geometry_policy": "synthetic_unit_test", "shape_local_pose_wxyz": [0.,0.,0.,1.,0.,0.,0.]}}
        synthetic_spheres, _ = _geometry([component], TASK_IDS[1])
        assert np.max(synthetic_spheres[:, 1] + synthetic_spheres[:, 3]) <= vertices[:, 1].max() + .002 + 1e-12
        _body_coverage([component], synthetic_spheres)

    # A real geometry change or recipe parameter change must MISS, even when
    # both inputs are otherwise valid and have the same task. Tiny solids make
    # these key-separation tests cheap, with independently checked new coverage.
    cache_component = deepcopy(component)
    spec = _fixed_recipe(TASK_IDS[1])
    original_key = _geometry_cache_key([cache_component], spec)
    initial, initial_proof, _ = _geometry_bundle([cache_component], spec)
    baseline_info = _cached_geometry.cache_info()
    start = time.perf_counter()
    repeated, repeated_proof, _ = _geometry_bundle([cache_component], spec)
    repeated_s = time.perf_counter() - start
    assert np.array_equal(initial, repeated) and initial_proof == repeated_proof
    assert _cached_geometry.cache_info().hits == baseline_info.hits + 1
    changed_component = deepcopy(cache_component)
    changed_component["vertices"] = cache_component["vertices"] * 1.031
    changed_component["actor_vertices"] = changed_component["vertices"].copy()
    changed_component["mesh_sha256"] = canonical_hash(changed_component["vertices"].tolist())
    changed_key = _geometry_cache_key([changed_component], spec)
    before_change = _cached_geometry.cache_info()
    changed_spheres, _, _ = _geometry_bundle([changed_component], spec)
    assert changed_key != original_key and _cached_geometry.cache_info().misses == before_change.misses + 1
    assert not np.array_equal(initial, changed_spheres)
    _body_coverage([changed_component], changed_spheres)
    recipe_changed = {**spec, "support_plane_padding_m": .0015}
    recipe_key = _geometry_cache_key([cache_component], recipe_changed)
    before_recipe = _cached_geometry.cache_info()
    recipe_spheres, recipe_proof, _ = _geometry_bundle([cache_component], recipe_changed)
    assert recipe_key != original_key and _cached_geometry.cache_info().misses == before_recipe.misses + 1
    assert recipe_proof[0]["candidate_refinement"]["support_plane_padding_m"] == .0015
    assert np.max(recipe_spheres[:, 1] + recipe_spheres[:, 3]) <= cache_component["vertices"][:, 1].max() + .0015 + 1e-12
    _body_coverage([cache_component], recipe_spheres)
    cache_checks.extend(["changed_actual_local_mesh_does_not_hit_old_geometry",
                         "changed_recipe_parameter_does_not_hit_old_geometry"])
    reject("cache_recipe_nonfinite", lambda: _geometry_cache_key([cache_component], {**spec, "pitch_m": math.nan}))
    reject("cache_recipe_state_injection", lambda: _geometry_cache_key([cache_component], {**spec, "world": {}}))

    # Actual saved object poses and old fixed grasp candidates, no new simulation.
    import h5py
    goal_results = []
    for task, directory in ((TASK_IDS[1], "pill-dev-021-seed-1"), (TASK_IDS[2], "fruits-dev-011-seed-4")):
        old = ROOT / "infrastructure/validation/dual-arm/m1-multitask-20261116/development" / directory
        report = json.loads((old / "report.json").read_text())
        geometry = json.loads((old / "task-geometry.json").read_text())
        with h5py.File(old / "trajectory.h5") as trajectory:
            for arm in (0, 1):
                raw, new = snapshots[task, arm], refined[task, arm]
                state_name = geometry["actors"][f"target{arm}"]["state_name"]
                actual_pose = np.asarray(trajectory[f"traj_0/env_states/actors/{state_name}"][10, :7])
                goal = report["grasp_tcp_poses"][arm][1]
                tcp_object = np.linalg.inv(_pose_matrix(goal)) @ _pose_matrix(actual_pose)
                peer = snapshots[task, 1-arm]
                tcp_hand_shape = np.linalg.inv(_pose_matrix(raw["base_pose_wxyz"]) @ _pose_matrix(raw["tcp_pose_base_wxyz"])) @ _pose_matrix(peer["base_pose_wxyz"]) @ _pose_matrix(peer["world"]["mesh"]["peer::panda_hand::shape_0"]["pose"])
                hand_shape = next(row for row in raw["coverage"]["active_robot_shapes"] if row["link"] == "panda_hand")
                tcp_hand = tcp_hand_shape @ np.linalg.inv(_pose_matrix(hand_shape["shape_local_pose_wxyz"]))
                hand = np.array([[*row["center"], row["radius"]] for row in raw["robot_cfg"]["kinematics"]["collision_spheres"]["panda_hand"]])
                hand[:, :3] = _transform(hand[:, :3], tcp_hand)
                gaps = []
                for view in (raw, new):
                    spheres = np.asarray(view["payload_model"]["spheres_actor_local"])
                    centers = _transform(spheres[:, :3], tcp_object)
                    gap = np.linalg.norm(centers[:, None] - hand[None, :, :3], axis=2) - spheres[:, None, 3] - hand[None, :, 3]
                    gaps.append(float(gap.min()))
                assert gaps[1] > 0, (task, arm, gaps)
                goal_results.append({"task_id": task, "arm": arm, "old_goal_directory": str(old),
                    "object_state_row": 10, "goal_xyz_wxyz": goal,
                    "baseline_minimum_payload_hand_gap_m": gaps[0],
                    "candidate_minimum_payload_hand_gap_m": gaps[1],
                    "scope": "saved actual object + old candidate TCP; not a new confirmed grasp"})

    raw = snapshots[TASK_IDS[1], 0]
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("unknown_task", lambda s: s["payload_model"].update(task_id="unknown")),
        ("wrong_arm", lambda s: s.update(arm_index=1)),
        ("foreign_target", lambda s: s["payload_model"].update(target_actor_name="foreign")),
        ("stale_model_hash", lambda s: s["payload_model"].update(geometry_sha256="bad")),
        ("bad_count", lambda s: s["payload_model"].update(sphere_count=1)),
        ("insufficient_capacity", lambda s: s["payload_model"].update(capacity=1)),
        ("active_reserved_slot", lambda s: s["robot_cfg"]["kinematics"]["collision_spheres"][PAYLOAD_LINK][0].update(radius=.001)),
        ("uncovered_shape", lambda s: s["payload_model"]["shape_coverage"][0].update(complete_solid_coverage=False)),
        ("wrong_shape_order", lambda s: s["payload_model"]["shape_coverage"].reverse()),
        ("forbidden_hand_ignore", lambda s: s["robot_cfg"]["kinematics"]["self_collision_ignore"][PAYLOAD_LINK].append("panda_hand")),
        ("foreign_support_permission", lambda s: s["payload_state"].update(source_support_obstacle_names=["foreign"])),
        ("nonfinite_q", lambda s: s["q_start"].__setitem__(0, math.nan)),
        ("malformed_coverage", lambda s: s.update(coverage=[])),
    ]
    first_name = raw["payload_model"]["target_obstacle_names"][0]
    mutations.extend([
        ("missing_mesh", lambda s: s["world"]["mesh"].pop(first_name)),
        ("nonfinite_mesh", lambda s: s["world"]["mesh"][first_name]["vertices"][0].__setitem__(0, math.inf)),
        ("bad_mesh_shape", lambda s: s["world"]["mesh"][first_name].update(vertices=[[0., 0.]])),
        ("bad_mesh_index", lambda s: s["world"]["mesh"][first_name]["faces"][0].__setitem__(0, 999999)),
        ("stale_mesh_identity", lambda s: s["world"]["mesh"][first_name]["vertices"][0].__setitem__(0, .5)),
        ("moving_frozen_source", lambda s: s["world"]["cuboid"]["scene::table::shape_0"]["pose"].__setitem__(0, .1)),
    ])
    for label, mutate in mutations:
        invalid = deepcopy(raw)
        mutate(invalid)
        reject(label, lambda invalid=invalid: refine_task_snapshot(invalid))
    reject("not_a_snapshot", lambda: refine_task_snapshot([]))
    reject("missing_snapshot_fields", lambda: refine_task_snapshot({}))
    attached_bad = _synthetic_attachment(raw)
    attached_bad["payload_state"]["attachment"]["spheres"][0][3] *= .5
    reject("attachment_model_mismatch", lambda: refine_task_snapshot(attached_bad))
    tagged_bad = deepcopy(refined[TASK_IDS[1], 0])
    tagged_bad["payload_model"]["shape_coverage"][0]["candidate_refinement"]["recipe"] = "wrong"
    tagged_bad["payload_model"]["geometry_sha256"] = canonical_hash({"spheres": tagged_bad["payload_model"]["spheres_actor_local"], "coverage": tagged_bad["payload_model"]["shape_coverage"]})
    reject("candidate_recipe_identity", lambda: refine_task_snapshot(tagged_bad))
    invalid_proof = deepcopy(raw)
    invalid_proof["payload_model"]["shape_coverage"][0] = 0
    invalid_proof["payload_model"]["geometry_sha256"] = canonical_hash({
        "spheres": invalid_proof["payload_model"]["spheres_actor_local"],
        "coverage": invalid_proof["payload_model"]["shape_coverage"]})
    reject("malformed_proof_mapping", lambda: refine_task_snapshot(invalid_proof))

    # Original manager enforces frozen topology, stale versions and invalidation.
    for task in TASK_IDS:
        manager = TaskSessionManager(task, _CPUBackend)
        for arm in (0, 1):
            for profile in ("approach", "transport"):
                view = refined[task, arm]
                dispatch_task(_request(view, "init_session", profile), manager)
                updated = dispatch_task(_request(view, "update_session", profile, "v1"), manager)["result"]
                assert updated["version"] == 1
                second = _request(view, "update_session", profile, "v2")
                dispatch_task(second, manager)
                stale = {"id": 2, "op": "fk", "session_id": second["session_id"], "snapshot_id": "v1", "positions": [view["q_start"]]}
                reject(f"{task}/{arm}/{profile}/stale_token", lambda stale=stale: dispatch_task(stale, manager))
                # Same token, different finger/contact state must invalidate.
                changed = deepcopy(second)
                changed["payload_state"]["contact_allowed"] = True
                reject(f"{task}/{arm}/{profile}/rebound_token", lambda changed=changed: dispatch_task(changed, manager))
                assert not manager.sessions[second["session_id"]].valid
                reject(f"{task}/{arm}/{profile}/invalid_after_failure", lambda: dispatch_task({**stale, "snapshot_id": "v2"}, manager))
                dispatch_task(_request(view, "update_session", profile, "recovered"), manager)
                if task != TASK_IDS[0]:
                    old_topology = _request(snapshots[task, arm], "update_session", profile, "old-topology")
                    reject(f"{task}/{arm}/{profile}/coarse_model_after_init", lambda old_topology=old_topology: dispatch_task(old_topology, manager))
        manager.shutdown({})
    assert "curobo" not in sys.modules
    torch = sys.modules.get("torch")
    assert torch is None or not torch.cuda.is_initialized()
    return {"status": "passed", "recipe": RECIPE, "isolated_candidate": True,
        "candidate_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "actual_snapshot_tests": records, "old_goal_geometry_tests": goal_results,
        "negative_tests_rejected": len(rejected), "negative_tests": rejected,
        "geometry_cache": {"storage": "bounded LRU64; immutable sphere tuples and proof JSON",
            "key_scope": "finite canonical local vertices, faces, shape-local pose, fixed recipe and version only",
            "world_fingers_attachment_permissions_cached": False,
            "key_sha256_example": hashlib.sha256(original_key.encode()).hexdigest(),
            "same_local_geometry_cached_lookup_wall_s": repeated_s,
            "checks": cache_checks, "cache_info": _cached_geometry.cache_info()._asdict()},
        "synthetic_attachment_scope": "CPU schema only; never actual-grasp evidence",
        "native_planner_initialized": False, "environment_step_calls": 0,
        "GPU_validation": "NOT_RUN", "wall_s": time.perf_counter() - started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), allow_nan=False, indent=2))
