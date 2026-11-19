"""Representation and immutable claim tests; no native dynamics are run."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from resimpair_assets import raw_robocasa_native_v1 as v1
from resimpair_assets import raw_robocasa_native_v2 as v2
from resimpair_grasp.common import binding, read, resolve, write


def source(folder, articulated=False):
    folder.mkdir()
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.04)
    mesh.vertices[0] *= .2  # A real inward dent, not a mocked convexity flag.
    assert mesh.is_watertight and not mesh.is_convex
    mesh.export(folder / "part.obj", digits=12)
    (folder / "model.xml").write_text(
        '<mujoco><asset><mesh name="mesh" file="part.obj"/></asset><worldbody><body>'
        + ('<joint/>' if articulated else '') +
        '<geom type="mesh" mesh="mesh" group="1" contype="0" conaffinity="0"/>'
        '<geom type="mesh" mesh="mesh" group="0"/>'
        '</body></worldbody></mujoco>')
    return {"id": "robocasa:test/" + folder.name, "source": "robocasa", "name": "mug fixture",
        "path": str(folder / "model.xml"), "source_sha": binding(folder / "model.xml")["sha256"],
        "partition": "development"}


def rejected_v1(tmp_path, raw, registry):
    root = tmp_path / "v1"
    v1.prepare(root, [{"raw_row": raw}], maximum_new_sources=1, registry=registry)
    ref = read(root / "PROTOCOL.json")["jobs"][0]["job"]; path = resolve(ref)
    write(path.parent / "reservation.json", {"job": ref, "attempt": 1, "protocol": binding(root / "PROTOCOL.json")})
    result = v1.worker(root, path)
    assert result["status"] == "geometry_or_policy_failed" and result["recorded_settle_steps"] == 0
    v1.run(root)  # Only recover the completed format rejection, never launch native work.
    return ref


def test_author_mesh_convex_hull_is_explicit_measured_and_not_visual_substitution(tmp_path):
    raw = source(tmp_path / "source")
    receipt = v2.xml.inspect_raw_xml(raw)
    with pytest.raises(ValueError, match="convex"):
        v1.load_author_collision_parts(receipt)
    parts, proof = v2.load_author_convex_collision_parts(receipt)
    assert len(parts) == 1 and parts[0].is_convex
    assert proof["author_collision_grouping_preserved"] and not proof["new_decomposition"]
    part = proof["parts"][0]
    assert part["source_mesh"] == next(g for g in receipt["geoms"] if g["kind"] == "collision")["mesh_file"]
    assert part["original_convex_boolean_diagnostic_only"] is False
    assert part["convexification_delta"]["volume_difference_m3"] > 0
    assert part["convexification_delta"]["source_to_hull_max_probe_distance_m"] > .001
    assert part["convexification_delta"]["native_surface_gate_still_required"]
    assert not proof["native_acceptance_changed"]


def test_author_groups_and_XML_scale_transforms_remain_separate(tmp_path):
    raw = source(tmp_path / "source")
    root = ET.parse(raw["path"]).getroot()
    ET.SubElement(root.find("worldbody/body"), "geom", {"type": "mesh", "mesh": "mesh", "group": "0", "pos": ".2 0 0"})
    root.find("asset/mesh").set("scale", "2 3 4")
    ET.ElementTree(root).write(raw["path"]); raw["source_sha"] = binding(raw["path"])["sha256"]
    receipt = v2.xml.inspect_raw_xml(raw); parts, proof = v2.load_author_convex_collision_parts(receipt)
    assert len(parts) == 2
    assert np.allclose(parts[1].center_mass - parts[0].center_mass, [.2, 0, 0])
    assert np.allclose(parts[0].extents, [.16, .24, .32])
    assert proof["compound_properties"]["collision_shape_volume_sum_m3"] == pytest.approx(sum(p.volume for p in parts))
    assert np.allclose(proof["parts"][1]["source_mesh_to_XML_world"], [[2,0,0,.2],[0,3,0,0],[0,0,4,0],[0,0,0,1]])


def test_v2_default_reuses_v1_failure_without_charging_or_retry(tmp_path):
    raw = source(tmp_path / "source", articulated=True); registry = tmp_path / "registry"
    prior = rejected_v1(tmp_path, raw, registry)
    info = v2.prepare(tmp_path / "v2_reuse", [{"raw_row": raw}], maximum_new_sources=0, registry=registry)
    assert info["new_allocations"] == 0 and info["historical_allocation_count"] == 1
    protocol = v2._validate(tmp_path / "v2_reuse")
    assert protocol["jobs"][0]["job"] == prior and not protocol["jobs"][0]["owned"]
    assert v2.status(tmp_path / "v2_reuse")["counts"] == {"geometry_or_policy_failed": 1}


def test_explicit_method_comparison_is_new_policy_claim_with_old_attempt_retained(tmp_path):
    raw = source(tmp_path / "source", articulated=True); registry = tmp_path / "registry"
    prior = rejected_v1(tmp_path, raw, registry)
    comparison = {"source_id": raw["id"], "prior_job": prior, "reason": "declared per-part convex geometry interpretation"}
    root = tmp_path / "v2_comparison"
    info = v2.prepare(root, [{"raw_row": raw}], maximum_new_sources=1, workers=1, registry=registry,
                      method_comparisons=[comparison])
    assert info["new_allocations"] == 1 and info["cumulative_allocations_for_selected_sources_including_comparisons"] == 2
    protocol = v2._validate(root); new = protocol["jobs"][0]["job"]
    assert new != prior and protocol["historical_allocations_for_selected_sources"][0]["job"] == prior
    assert read(resolve(new))["geometry_policy"] == v2.VERSION
    continuation = v2.prepare(tmp_path / "continuation", [{"raw_row": raw}], maximum_new_sources=0, registry=registry)
    assert continuation["new_allocations"] == 0 and continuation["historical_allocation_count"] == 2
    assert v2._validate(tmp_path / "continuation")["jobs"][0]["job"] == new
    assert v2.status(tmp_path / "continuation")["counts"] == {"prior_campaign_not_submitted": 1}


def test_whole_population_accounting_keeps_previous_methods_and_allocates_only_new_sources(tmp_path):
    raw = source(tmp_path / "source", articulated=True); registry = tmp_path / "registry"
    prior = rejected_v1(tmp_path, raw, registry)
    v2.prepare(tmp_path / "comparison", [{"raw_row": raw}], maximum_new_sources=1, registry=registry,
               method_comparisons=[{"source_id": raw["id"], "prior_job": prior, "reason": "fixed method comparison"}])
    unseen = source(tmp_path / "unseen", articulated=True)
    result = v2.prepare(tmp_path / "population", [{"raw_row": raw}, {"raw_row": unseen}], maximum_new_sources=1, registry=registry)
    assert result["registered"] == 2 and result["new_allocations"] == 1
    assert result["historical_allocation_count"] == 2
    assert result["cumulative_allocations_for_selected_sources_including_comparisons"] == 3
    assert len(list((registry / "claims").glob('*.json'))) == 3


def test_undeclared_or_unfunded_method_comparison_rejects(tmp_path):
    raw = source(tmp_path / "source", articulated=True); registry = tmp_path / "registry"
    prior = rejected_v1(tmp_path, raw, registry)
    with pytest.raises(ValueError, match="reason"):
        v2.prepare(tmp_path / "missing_reason", [{"raw_row": raw}], maximum_new_sources=1, registry=registry,
                   method_comparisons=[{"source_id": raw["id"], "prior_job": prior}])
    with pytest.raises(ValueError, match="cumulative_allocation"):
        v2.prepare(tmp_path / "unfunded", [{"raw_row": raw}], maximum_new_sources=0, registry=registry,
                   method_comparisons=[{"source_id": raw["id"], "prior_job": prior, "reason": "declared"}])
    assert not (tmp_path / "missing_reason").exists() and not (tmp_path / "unfunded").exists()


def test_v2_worker_rejects_legacy_reused_job(tmp_path):
    raw = source(tmp_path / "source", articulated=True); registry = tmp_path / "registry"
    prior = rejected_v1(tmp_path, raw, registry)
    v2.prepare(tmp_path / "reuse", [{"raw_row": raw}], maximum_new_sources=0, registry=registry)
    with pytest.raises(ValueError, match="owned_frozen"):
        v2.worker(tmp_path / "reuse", resolve(prior))


def test_population_run_requires_bound_independent_release_before_any_submission(tmp_path):
    a, b = source(tmp_path / "a", articulated=True), source(tmp_path / "b", articulated=True)
    root = tmp_path / "population"
    v2.prepare(root, [{"raw_row": a}, {"raw_row": b}], maximum_new_sources=2, registry=tmp_path / "registry")
    protocol = v2._validate(root)
    assert protocol["independent_review_required_before_submission"]
    with pytest.raises(ValueError, match="independent_review_required"):
        v2.run(root)
    with pytest.raises(ValueError, match="independent_review_required"):
        v2.worker(root, resolve(protocol["jobs"][0]["job"]))
    assert not list(root.glob("jobs/*/reservation.json")) and not (root / "START.json").exists()
    evidence = tmp_path / "evidence.json"; write(evidence, {"checks": "fixture-only review proof"})
    gate = tmp_path / "gate.json"
    write(gate, {"schema": "raw_robocasa_native_v2_independent_release", "reviewer_role": "independent",
        "protocol": binding(root / "PROTOCOL.json"), "passed": True, "evidence": [binding(evidence)]})
    assert v2._release_gate(root, protocol, gate) == binding(gate)
    # Verify the pure gate only, deliberately do not call run with approval.
    assert not list(root.glob("jobs/*/reservation.json"))
