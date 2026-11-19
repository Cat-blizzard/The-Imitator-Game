"""Read-only mesh representation regressions; no simulator or paid calls."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import trimesh
from resimpair_tasks import container_geometry_equivalence_v2 as candidate

from resimpair_tasks.container_geometry_equivalence_v2 import (
    ROOT, binding, bound_geometry, compare_triangle_mesh, evaluate_execution, exact_zero_faces)


def mesh_and_shape():
    mesh = trimesh.Trimesh(vertices=[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    shape = {"vertices": mesh.vertices.tolist(), "triangles": mesh.faces.tolist(),
             "scale": [1., 1., 1.], "local_pose": np.eye(4).tolist(),
             "type": "PhysxCollisionShapeTriangleMesh"}
    return mesh, shape


def test_exact_zero_faces_can_be_removed_without_changing_surface():
    mesh, actual = mesh_and_shape()
    mesh.faces = np.vstack([mesh.faces, [0, 0, 1]])
    report = compare_triangle_mesh(mesh, actual)
    assert report["passed"] and report["source_exact_zero_faces"] == 1
    assert report["source_nondegenerate_oriented_faces"] == 2


def test_identical_coordinate_duplicates_and_cyclic_face_order_are_equivalent():
    mesh, actual = mesh_and_shape()
    actual["vertices"].append(actual["vertices"][0])
    actual["triangles"] = [[1, 2, 4], [2, 3, 0]]
    assert compare_triangle_mesh(mesh, actual)["passed"]


@pytest.mark.parametrize("change", ["missing", "added", "winding", "vertex", "scale", "transform", "type"])
def test_non_equivalent_geometry_still_fails(change):
    mesh, actual = mesh_and_shape()
    if change == "missing": actual["triangles"].pop()
    elif change == "added": actual["triangles"].append([0, 1, 3])
    elif change == "winding": actual["triangles"][0].reverse()
    elif change == "vertex": actual["vertices"][0][2] = 2e-5
    elif change == "scale": actual["scale"][0] = 1.01
    elif change == "transform": actual["local_pose"][0][3] = .01
    elif change == "type": actual["type"] = "PhysxCollisionShapeConvexMesh"
    assert compare_triangle_mesh(mesh, actual)["passed"] is False


@pytest.mark.parametrize("side", [1e-7, 1e-14, 1e-200])
def test_tiny_nonzero_face_is_never_filtered_even_when_cross_product_underflows(side):
    mesh, actual = mesh_and_shape()
    vertices = np.vstack([mesh.vertices, [[0., 0., 1.], [side, 0., 1.], [0., side, 1.]]])
    faces = np.vstack([mesh.faces, [4, 5, 6]])
    expected = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert not exact_zero_faces(vertices, faces).any()
    complete = deepcopy(actual); complete["vertices"] = vertices.tolist(); complete["triangles"] = faces.tolist()
    assert compare_triangle_mesh(expected, complete)["passed"]
    missing = deepcopy(complete); missing["triangles"].pop()
    report = compare_triangle_mesh(expected, missing)
    assert report["passed"] is False and report["source_exact_zero_faces"] == 0


def test_nearby_distinct_vertices_are_not_approximately_welded():
    mesh, actual = mesh_and_shape()
    vertices = np.vstack([mesh.vertices, [[1e-10, 0., 0.]]])
    expected = trimesh.Trimesh(vertices=vertices, faces=[[0, 4, 2], [0, 2, 3]], process=False)
    report = compare_triangle_mesh(expected, actual)
    assert report["passed"] is False and report["source_exact_zero_faces"] == 0


def test_geometry_identity_binding_mismatch_rejected_before_loading_geometry():
    support = {"id": "bowl", "container_descriptor": {"path": "expected", "sha256": "a"}}
    native = {"id": "bowl", "descriptor": {"path": "other", "sha256": "a"}, "static": True}
    with pytest.raises(ValueError, match="container_binding_mismatch"):
        bound_geometry(support, native)


@pytest.mark.parametrize("case", ["relocate_selected", "orient_selected", "arrange_group"])
def test_three_real_scene010_false_negatives_only_remove_exact_zero_faces(case):
    directory = ROOT / "runs/batch500_20261027/production_transport_v2/production_transport_v2_chunk_001/executions" / ("factory_scene_010_23e99d1c_" + case)
    if not (directory / "native_containers.json").exists():
        pytest.skip("Read-only recorded native evidence is not installed")
    old = json.loads((directory / "container_audit.json").read_text())
    assert not old["passed"] and "container_native_triangle_difference" in str(old["errors"])
    support = json.loads((directory / "execution_bundle.json").read_text())["scene"]["supports"][0]
    native = json.loads((directory / "native_containers.json").read_text())["containers"][0]
    report, _, _ = bound_geometry(support, native)
    assert report["passed"] and report["source_exact_zero_faces"] == 109
    assert report["native_exact_zero_faces"] == 0
    assert report["source_nondegenerate_oriented_faces"] == 34382
    assert report["maximum_vertex_difference_m"] < 1e-8


@pytest.mark.parametrize("case", ["000_3973d7e9_relocate_selected", "001_3d104d3b_relocate_selected",
                                  "002_f8ff83d2_contain_selected", "006_3530665a_relocate_selected"])
def test_previously_accepted_other_recorded_containers_remain_equivalent(case):
    directory = ROOT / "runs/batch500_20261027/production_seed_fix_r2/production_seed_fix_r2_chunk_000/executions" / ("factory_scene_" + case)
    if not (directory / "native_containers.json").exists():
        pytest.skip("Read-only recorded native evidence is not installed")
    assert json.loads((directory / "container_audit.json").read_text())["passed"]
    supports = json.loads((directory / "execution_bundle.json").read_text())["scene"]["supports"]
    native = json.loads((directory / "native_containers.json").read_text())["containers"]
    for support, row in zip(supports, native):
        assert bound_geometry(support, row)[0]["passed"]


@pytest.mark.parametrize("case", ["010_23e99d1c_relocate_selected", "010_23e99d1c_orient_selected",
                                  "010_23e99d1c_arrange_group", "003_83d683b0_relocate_selected"])
def test_complete_original_dependent_evaluators_rerun_without_touching_evidence(case):
    production = ROOT / "runs/batch500_20261027/production_transport_v2"
    case_id = "factory_scene_" + case
    cohorts = list(production.glob("*_chunk_*/cohort.json"))
    found = [(path.parent, c) for path in cohorts for c in json.loads(path.read_text())["cases"]
             if c["id"] == case_id]
    if not found or not (found[0][0] / "executions" / case_id / "container_audit.json").exists():
        pytest.skip("Read-only completed recorded evidence is not installed")
    cohort, request = found[0]; directory = cohort / "executions" / case_id
    names = ("audit.json", "container_audit.json", "requested_goal_audit.json", "execution_bundle.json",
             "native_containers.json", "measurements.npz", "terminal.json")
    before = {name: binding(directory / name) for name in names}
    report = evaluate_execution(directory, request)
    assert report["errors"] == []
    assert report["original_inputs_unchanged"]
    assert report["original_metric_results_reproduced"]
    assert before == {name: binding(directory / name) for name in names}
    # A genuinely failed original grasp/path remains failed even when its
    # container geometry and footprint are equivalent.
    assert report["candidate_passed"] == case.startswith("010_")
    if case.startswith("010_"):
        assert report["recomputed_container_checks"]["payload_inside_qualified_floor"]["passed"]
        assert all(g["passed"] for g in report["recomputed_requested_goals"])
    else:
        assert report["retained_native_passed"] is False


def test_late_binding_failure_vetoes_already_computed_pass(monkeypatch):
    cohort = ROOT / "runs/batch500_20261027/production_transport_v2/production_transport_v2_chunk_001"
    case_id = "factory_scene_010_23e99d1c_relocate_selected"
    if not (cohort / "executions" / case_id / "container_audit.json").exists():
        pytest.skip("Read-only completed recorded evidence is not installed")
    case = next(c for c in json.loads((cohort / "cohort.json").read_text())["cases"] if c["id"] == case_id)
    real_checked = candidate.checked
    calls = 0

    def fail_late(value, root=ROOT):
        nonlocal calls
        if Path(value["path"]).name == "native_containers.json":
            calls += 1
            if calls == 2:
                raise ValueError("geometry_binding_hash_changed_in_postcheck")
        return real_checked(value, root)

    monkeypatch.setattr(candidate, "checked", fail_late)
    report = evaluate_execution(cohort / "executions" / case_id, case)
    # Complete unchanged-rule evaluation did pass before the final hash veto.
    assert report["recomputed_requested_errors"] == []
    assert all(g["passed"] for g in report["recomputed_requested_goals"])
    assert calls == 2 and report["errors"]
    assert report["candidate_passed"] is None
    assert report["candidate_container_passed"] is None
    assert report["unknown"] == "shadow_reassessment_or_integrity_failed"


def test_summary_never_counts_pass_with_error(monkeypatch, tmp_path):
    production = tmp_path / "production"; cohort = production / "production_chunk_000"
    directory = cohort / "executions" / "case"; directory.mkdir(parents=True)
    case = {"id": "case", "source_request": {"path": "unused", "sha256": "unused"}, "requested_task_id": "task"}
    (cohort / "cohort.json").write_text(json.dumps({"cases": [case]}))
    (production / "protocol.json").write_text(json.dumps({"qualification_cases": ["case"]}))
    (directory / "terminal.json").write_text(json.dumps({"physical_attempts": 1}))
    (directory / "requested_goal_audit.json").write_text("{}")
    monkeypatch.setattr(candidate, "evaluate_execution", lambda *a: {
        "case": "case", "original_passed": False, "candidate_passed": True, "errors": ["invalid_binding"]})
    report = candidate.reevaluate([production], tmp_path / "new_report.json")
    summary = report["summary"][str(production)]
    assert summary["candidate_passed"] == 0 and summary["unknown"] == 1
    assert summary["promoted_shadow_only"] == []
