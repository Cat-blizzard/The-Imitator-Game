"""Read-only, versioned shadow review of recorded stationary containers.

Never imported by production. Only exactly zero-area triangles may disappear.
Zero-area classification precedes deduplication and uses exact stored floats;
positive-area triangles, however small, remain subject to oriented face equality.
The original 10 micrometre vertex bound and transform/type checks are retained.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

VERSION = "stationary_container_geometry_equivalence_v2"
VERTEX_TOLERANCE_M = 1e-5
ROOT = Path(__file__).resolve().parents[1]
V1_RULE_SHA256 = "509d86cac122a3b682928f18d2911667e61a592ea843bfa4dfe1f15cecae3835"


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(value, root=ROOT):
    path = Path(value["path"])
    if not path.is_absolute():
        path = Path(root) / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
        raise ValueError("geometry_binding_hash_changed")
    return path


def exact_zero_faces(vertices, faces):
    """No area epsilon, norm-underflow, rounding or approximate vertex weld.

    An exact rational fallback resolves cancellation/underflow when floating
    cross products are zero. Fractions refer to the actual stored float values,
    not a higher-precision reconstruction of an upstream source model.
    """
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    result = np.zeros(len(triangles), dtype=bool)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cross = np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0])
    candidates = np.flatnonzero(np.all(cross == 0, axis=1) | ~np.isfinite(cross).all(axis=1))
    for i in candidates:
        a, b, c = triangles[i]
        if np.array_equal(a, b) or np.array_equal(a, c) or np.array_equal(b, c):
            result[i] = True
            continue
        a, b, c = [[Fraction.from_float(float(x)) for x in point] for point in (a, b, c)]
        u = [x-y for x, y in zip(b, a)]; v = [x-y for x, y in zip(c, a)]
        result[i] = all(u[j]*v[k] == u[k]*v[j] for j, k in ((1, 2), (2, 0), (0, 1)))
    return result


def _arrays(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1:] != (3,)
            or not len(faces) or faces.dtype.kind not in "iu" or faces.min() < 0
            or faces.max() >= len(vertices)):
        raise ValueError("invalid_triangle_mesh_arrays")
    return vertices, faces.astype(np.int64)


def _oriented_set(faces):
    # Cyclic rotation preserves orientation; sorting three indices does not.
    indices = (np.argmin(faces, axis=1)[:, None] + np.arange(3)) % 3
    return {tuple(x) for x in np.take_along_axis(faces, indices, axis=1)}


def compare_triangle_mesh(expected, actual):
    """Conservative equality of nondegenerate oriented triangles.

    Ambiguous/nonbijective nearest-vertex correspondence is rejected rather
    than merging nearby distinct vertices under the distance tolerance.
    """
    report = {"version": VERSION, "passed": False,
              "vertex_tolerance_m": VERTEX_TOLERANCE_M,
              "zero_area_policy": "exact_stored_coordinates_before_deduplication"}
    try:
        sv, sf = _arrays(expected.vertices, expected.faces)
        av, af = _arrays(actual["vertices"], actual["triangles"])
        sz, az = exact_zero_faces(sv, sf), exact_zero_faces(av, af)
        report.update(source_vertices=len(sv), native_vertices=len(av),
                      source_faces=len(sf), native_faces=len(af),
                      source_exact_zero_faces=int(sz.sum()), native_exact_zero_faces=int(az.sum()))
        su, si = np.unique(sv, axis=0, return_inverse=True)
        au, ai = np.unique(av, axis=0, return_inverse=True)
        distance, nearest = cKDTree(su).query(au)
        reverse = cKDTree(au).query(su)[0]
        error = float(max(distance.max(), reverse.max()))
        report["maximum_vertex_difference_m"] = error
        if error >= VERTEX_TOLERANCE_M:
            raise ValueError("container_native_vertex_difference")
        if np.array_equal(su, au):
            nearest = np.arange(len(su))
        if len(su) != len(au) or len(np.unique(nearest)) != len(su):
            raise ValueError("ambiguous_or_nonbijective_vertex_correspondence")
        if not np.any(~sz) or not np.any(~az):
            raise ValueError("no_nondegenerate_collision_surface")
        source = _oriented_set(si[sf[~sz]])
        native = _oriented_set(nearest[ai[af[~az]]])
        report.update(source_nondegenerate_oriented_faces=len(source),
                      native_nondegenerate_oriented_faces=len(native),
                      missing_nondegenerate_faces=len(source-native),
                      added_nondegenerate_faces=len(native-source))
        if source != native:
            raise ValueError("nondegenerate_oriented_triangle_difference")
        if not np.allclose(actual["scale"], [1, 1, 1]) or not np.allclose(actual["local_pose"], np.eye(4), atol=1e-6):
            raise ValueError("container_native_transform_difference")
        if actual["type"] != "PhysxCollisionShapeTriangleMesh":
            raise ValueError("container_cavity_must_not_use_convex_hull")
        report["passed"] = True
    except (ValueError, TypeError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report


def bound_geometry(support, recorded, root=ROOT):
    """Verify immutable identity before comparing any mesh representation."""
    if (support["id"] != recorded["id"] or support["container_descriptor"] != recorded["descriptor"]
            or recorded.get("static") is not True):
        raise ValueError("container_binding_mismatch")
    if len(recorded["shapes"]) != 1:
        raise ValueError("container_native_shape_count")
    descriptor = read(checked(support["container_descriptor"], root))
    expected = trimesh.load(checked(descriptor["collision"], root), force="mesh", process=False)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("z", support.get("yaw", 0.)).as_matrix()
    world[:3, 3] = support["position"]
    if not np.allclose(recorded["world_pose"], world, atol=1e-6):
        raise ValueError("container_native_pose_difference")
    result = compare_triangle_mesh(expected, recorded["shapes"][0])
    return {"id": support["id"], "source_id": descriptor["source_id"],
            "descriptor": support["container_descriptor"], "collision": descriptor["collision"], **result}, world, descriptor


def evaluate_execution(directory, case, root=ROOT):
    """Rerun original container and goal evaluators with one function substituted.

    A private temporary directory copies immutable evidence and the original
    retained physical audit. Original container/goal outputs are deliberately
    absent there, so both complete evaluators run and write only temporary
    outputs. The production files and loaded modules in other processes are
    unaffected. This function must not run concurrently in the same process.
    """
    from resimpair_pickplace import evaluator as original
    from resimpair_pickplace import requested_goal_audit
    if binding(original.__file__)["sha256"] != V1_RULE_SHA256:
        raise ValueError("original_container_rules_changed_requires_new_shadow_protocol")
    directory = Path(directory).resolve()
    goal = read(directory / "requested_goal_audit.json")
    old = read(directory / "container_audit.json")
    result = {"case": case["id"], "original_passed": goal["passed"],
              "original_container_passed": old["passed"], "candidate_passed": False,
              "original_goal_audit": binding(directory / "requested_goal_audit.json"),
              "original_container_audit": binding(directory / "container_audit.json"),
              "geometry": [], "errors": []}
    try:
        checked(goal["rule"], root)
        for bound in old["rule_bindings"].values():
            checked(bound, root)
        if read(checked(goal["native_audit"], root)) != old:
            raise ValueError("original_container_audit_binding_changed")
        retained = read(checked(old["retained_audit"], root))
        request = read(checked(case["source_request"], root))
        if request != read(checked(goal["source_request"], root)):
            raise ValueError("original_task_identity_changed")
        bundle = read(checked(retained["bundle"], root)); scene = bundle["scene"]
        if (bundle != read(directory / "execution_bundle.json") or bundle["case"] != case
                or scene != read(checked(case["scene"], root))
                or bundle["task_intent"] != read(checked(case["task"], root))):
            raise ValueError("execution_scene_or_task_binding_changed")
        native_record = binding(directory / "native_containers.json")
        recorded = read(checked(native_record, root))["containers"]
        if len(recorded) != len(scene["supports"]):
            raise ValueError("missing_native_container")
        for support, row in zip(scene["supports"], recorded):
            result["geometry"].append(bound_geometry(support, row, root)[0])

        def shadow_comparator(expected, actual):
            comparison = compare_triangle_mesh(expected, actual)
            if not comparison["passed"]:
                raise ValueError(comparison["error"])
            return comparison

        with tempfile.TemporaryDirectory(prefix="resimpair_container_v2_") as temporary:
            shadow = Path(temporary)
            for filename in ("audit.json", "execution_bundle.json", "native_containers.json"):
                # No symlink can redirect an original evaluator's write back
                # into production. Large meshes/measurements remain hash-bound
                # absolute read-only inputs referenced by the copied bundle.
                shutil.copyfile(directory / filename, shadow / filename)
            with patch.object(original, "compare_triangle_mesh", shadow_comparator):
                container = original.evaluate(shadow)
                metric = requested_goal_audit.evaluate(shadow, case)
        result.update(retained_native_passed=retained["passed"],
                      candidate_container_passed=container["passed"],
                      candidate_passed=metric["passed"],
                      recomputed_container_checks=container["checks"],
                      recomputed_container_errors=container["errors"],
                      recomputed_requested_goals=metric["goals"],
                      recomputed_requested_errors=metric["errors"],
                      original_metric_results_reproduced=metric["goals"] == goal["goals"],
                      original_rule_execution="original container.evaluate and requested_goal_audit.evaluate; only geometry comparator substituted in this process")
        result["native_record"] = native_record
        result["measurements"] = bundle["measurements"]
        # Outputs reused from production must remain byte-identical throughout.
        unchanged = [result["original_goal_audit"], result["original_container_audit"],
                     old["retained_audit"], retained["bundle"], bundle["measurements"], native_record,
                     case["scene"], case["task"], case["source_request"], goal["rule"],
                     *old["rule_bindings"].values()]
        for geometry in result["geometry"]:
            unchanged.extend([geometry["descriptor"], geometry["collision"]])
        for value in unchanged:
            checked(value, root)
        result["original_inputs_unchanged"] = True
        result["unchanged_bindings"] = unchanged
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["errors"].append(str(error))
        # A later integrity failure vetoes even an already computed pass.
        # Missing or changed evidence is unknown, never a geometry promotion.
        result["candidate_passed"] = None
        result["candidate_container_passed"] = None
        result["unknown"] = "shadow_reassessment_or_integrity_failed"
    return result


def reevaluate(roots, output, root=ROOT):
    """Snapshot completed cases; write a new report with exclusive creation."""
    snapshot = []
    for production in map(Path, roots):
        for cohort in sorted(production.glob("*_chunk_*/cohort.json")):
            for case in read(cohort)["cases"]:
                directory = cohort.parent / "executions" / case["id"]
                terminal = directory / "terminal.json"
                if terminal.exists():
                    snapshot.append((production, case, directory, binding(terminal)))
    rows = []
    for production, case, directory, terminal in snapshot:
        t = read(checked(terminal, root))
        if not t["physical_attempts"]:
            result = {"case": case["id"], "original_passed": False, "candidate_passed": False,
                      "preflight_rejection_unchanged": t.get("reason")}
        elif not (directory / "requested_goal_audit.json").exists():
            result = {"case": case["id"], "original_passed": None, "candidate_passed": None,
                      "unknown": "original_audit_not_yet_available"}
        else:
            result = evaluate_execution(directory, case, root)
        if result.get("errors") and result.get("candidate_passed") is True:
            # Defensive aggregation: an inconsistent per-case response cannot
            # contribute a passing numerator even if a caller is substituted.
            result["candidate_passed"] = None
            result["unknown"] = "inconsistent_shadow_pass_with_errors"
        rows.append({"production": str(production.resolve()), "terminal": terminal,
                     "source_request": case["source_request"],
                     "requested_task_id": case["requested_task_id"], **result})
    summary = {}
    for production in sorted({x["production"] for x in rows}):
        group = [x for x in rows if x["production"] == production]
        summary[production] = {"snapshot_rows": len(group),
            "original_passed": sum(x["original_passed"] is True for x in group),
            "candidate_passed": sum(x["candidate_passed"] is True for x in group),
            "promoted_shadow_only": [x["case"] for x in group if x["candidate_passed"] is True and x["original_passed"] is False],
            "regressed_shadow_only": [x["case"] for x in group if x["original_passed"] is True and x["candidate_passed"] is False],
            "unknown": sum(x["candidate_passed"] is None for x in group)}
    roots_by_name = {str(Path(p).resolve()): Path(p) for p in roots}
    fixed = {}
    for name, production in roots_by_name.items():
        protocol = read(production / "protocol.json")
        ids = set(protocol["qualification_cases"])
        group = [x for x in rows if x["production"] == name and x["case"] in ids]
        fixed[name] = {"protocol": binding(production / "protocol.json"), "declared_cases": len(ids),
                       "snapshot_rows": len(group),
                       "original_passed": sum(x["original_passed"] is True for x in group),
                       "candidate_passed": sum(x["candidate_passed"] is True for x in group)}
    value = {"version": VERSION, "implementation": binding(__file__), "created_unix": time.time(),
             "original_rule_sha256": V1_RULE_SHA256, "vertex_tolerance_m": VERTEX_TOLERANCE_M,
             "summary": summary, "fixed_qualification_comparison": fixed,
             "rows": rows, "original_statistics_modified": False,
             "physical_executions": 0, "api_calls": 0, "default_promoted": False,
             "scope": "Read-only shadow: original container and requested-goal evaluators run completely in a temporary evidence directory with only the geometry comparator substituted. The immutable original retained native audit is reused. No production output is rewritten.",
             "source_scope": "Equivalence to the bound collision asset; does not certify lossless export from its upstream GLB."}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = reevaluate(args.production, args.output, args.root)
    print(json.dumps({"version": VERSION, "summary": report["summary"], "output": str(args.output)}, ensure_ascii=False, indent=2))
