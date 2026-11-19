"""Asset geometry and leakage/factor integrity checks; CPU only, no simulator."""
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from data_generation_methods.human_to_robot_pipeline.household_assets import (
    conservative_collision, metric_transform,
)
from data_generation_methods.human_to_robot_pipeline.expand_household_cases import (
    rng_for,
)
from data_generation_methods.human_to_robot_pipeline.audit_household_coverage import (
    audit_attempt,camera_measurement,runtime_materials,
)
from data_generation_methods.human_to_robot_pipeline.repair_household_layouts import (
    bbox_margin_conflict,world_vertices,
)
from data_generation_methods.human_to_robot_pipeline.repair_household_material import split_glb
from data_generation_methods.r2h_pipeline.common import file_sha256

ARTIFACT=Path("/mnt/sdg/datasets/real_sim_human_pipeline/expansion_20261020e")


def test_baked_metric_transform_is_uniform_proper_and_bottom_supported():
    mesh=trimesh.creation.box(extents=[2.,5.,1.])
    mesh.apply_translation([11,-8,3])
    transform,record=metric_transform(mesh.vertices,.16,"flat_shortest_axis")
    baked=trimesh.transform_points(mesh.vertices,transform)
    assert np.ptp(baked,axis=0).max()==pytest.approx(.16)
    assert np.ptp(baked,axis=0).argmin()==2
    assert baked[:,2].min()==pytest.approx(0)
    np.testing.assert_allclose((baked.max(0)+baked.min(0))[:2]/2,0,atol=1e-15)
    linear=transform[:3,:3]
    np.testing.assert_allclose(linear.T@linear,np.eye(3)*record["uniform_scale_source_units_to_design_m"]**2,
                               atol=1e-15)
    assert np.linalg.det(linear)>0
    assert record["is_measured_real_world_scale"] is False


@pytest.mark.parametrize("vertices",[
    [[0,0,0],[1,1,float("nan")]],
    [[0,0,0],[1,1,0],[2,2,0]],
    [[0,0],[1,1]],
])
def test_bad_mesh_cannot_be_silently_scaled(vertices):
    with pytest.raises(ValueError):
        metric_transform(vertices,.1,"upright_y")


def test_collision_fallback_encloses_all_vertices():
    mesh=trimesh.creation.icosphere(subdivisions=3,radius=.04)
    mesh.apply_translation([.02,.03,.08])
    collision,kind=conservative_collision(mesh,max_hull_vertices=16)
    assert kind=="conservative_AABB_due_to_convex_vertex_budget"
    assert collision.is_watertight and collision.is_convex
    assert len(collision.vertices)==8
    assert (mesh.vertices>=collision.bounds[0]-1e-12).all()
    assert (mesh.vertices<=collision.bounds[1]+1e-12).all()


def test_independent_factor_rng_is_not_consumption_order_dependent():
    first=rng_for(2026302001,30).normal(size=8)
    rng_for(2026302001,20).normal(size=5000)
    np.testing.assert_array_equal(first,rng_for(2026302001,30).normal(size=8))
    assert not np.array_equal(first,rng_for(2026302001,10).normal(size=8))


@pytest.fixture(scope="module")
def real_catalog():
    path=ARTIFACT/"household_assets_v3/catalog.json"
    if not path.exists():
        pytest.skip("Local qualification artifacts are not shipped with source")
    return json.loads(path.read_text())


def test_real_assets_all_have_visual_collision_and_explicit_scale(real_catalog):
    assert len(real_catalog["assets"])==12
    assert not real_catalog["failed_candidates"]
    for asset in real_catalog["assets"]:
        assert asset["category"] not in {"bowl","apple"}
        assert asset["source_url"] and asset["license"] and asset["attribution"]
        assert asset["physics_qualified"] is False
        assert asset["training_admitted"] is False
        assert asset["runtime_scale_must_be_one"] is True
        for path,sha in asset["files"].items():
            assert file_sha256(path)==sha
        collision=trimesh.load(asset["collision"],force="mesh",process=False)
        visual=trimesh.load(asset["visual"],force="scene",process=False)
        assert collision.is_watertight and collision.is_convex
        assert np.isfinite(collision.vertices).all()
        np.testing.assert_allclose(collision.bounds,visual.bounds,atol=1e-6)
        assert not asset["transform"]["is_measured_real_world_scale"]
        assert 0.04<=max(asset["extents_m"])<=.30


@pytest.fixture(scope="module")
def real_layouts():
    path=ARTIFACT/"household_layouts_v2/candidate_manifest.json"
    if not path.exists():
        pytest.skip("Local qualification artifacts are not shipped with source")
    return json.loads(path.read_text())


def test_factor_groups_preserve_task_binding_and_nuisance_variables(real_layouts):
    assert real_layouts["new_robot_trajectories"]==0
    assert not real_layouts["simulation_executed"]
    assert not real_layouts["original_four_pilot_seeds_modified"]
    assert len(real_layouts["cases"])==24
    groups={}
    for case in real_layouts["cases"]:
        groups.setdefault(case["controlled_group_id"],[]).append(case)
        assert case["screen"]["coarse_screen_passed"]
        assert case["split"]==case["human_source_window"]["split"]=="train"
        assert not case["simulation_executed"] and not case["training_admitted"]
    assert len(groups)==4
    for cases in groups.values():
        baseline=next(c for c in cases if c["factor_design"]["clutter_count"]==0
                      and c["factor_design"]["camera_condition"]=="base"
                      and c["factor_design"]["appearance_condition"]=="native")
        for case in cases:
            for key in ["roles","seed","human_video","human_window_id","human_source_window"]:
                assert case[key]==baseline[key]
            if case["factor_design"]["camera_condition"]=="base":
                assert case["camera"]==baseline["camera"]
            for distractor in case["added_distractors"]:
                assert distractor["body_type"]=="static"
                assert distractor["must_remain_untouched"]
                assert distractor["visual"] in distractor["asset_files"]
                assert distractor["collision"] in distractor["asset_files"]
                assert distractor["no_runtime_scaling"]
        two=next(c for c in cases if len(c["added_distractors"])==2)
        four=next(c for c in cases if len(c["added_distractors"])==4)
        assert two["added_distractors"]==four["added_distractors"][:2]


def test_unimplemented_appearance_cannot_be_presented_as_effective_randomization(real_layouts):
    blocked=[]
    for case in real_layouts["cases"]:
        assert case["appearance"]["material_seed_consumed_by_current_adapter"] is False
        if case["factor_design"]["appearance_condition"]!="native":
            assert "explicit_appearance_override_not_implemented" in case["execution_gate"]["blockers"]
            assert case["appearance"]["explicit_surface_asset"]["runtime_override_implemented"] is False
            blocked.append(case)
    assert len(blocked)==8


def test_readable_categories_precede_low_readability_stapler(real_layouts):
    assert len(real_layouts["used_household_categories"])==12
    early=[c for c in real_layouts["cases"] if "g00_" in c["case_id"] or "g01_" in c["case_id"]]
    assert all(d["category"]!="stapler" for c in early for d in c["added_distractors"])
    g00=next(c for c in early if "g00_clutter4" in c["case_id"])
    assert {d["category"] for d in g00["added_distractors"]}=={
        "soda_can","water_bottle","tissue_box","soap_dispenser"}


def test_actual_camera_evidence_detects_mismatched_requested_camera():
    folder=Path("/mnt/sdg/datasets/real_sim_human_pipeline/strict_20261020e/layout2001_v2")
    if not folder.exists():pytest.skip("Archived physical episode absent")
    camera=json.loads((folder/"input.json").read_text())["proposal"]["camera"]
    measurement=camera_measurement(folder/"episode/qa_visibility.npz",camera)
    assert measurement["matches_proposal"]
    camera["eye"][0]+=.03
    assert not camera_measurement(folder/"episode/qa_visibility.npz",camera)["matches_proposal"]


def test_runtime_material_missing_is_unknown_not_passed():
    result=runtime_materials({})
    assert all(m["status"]=="unknown_missing_runtime_record" for m in result.values())


def test_actual_success_audit_binds_raw_camera_and_runtime_materials():
    folder=Path("/mnt/sdg/datasets/real_sim_human_pipeline/strict_20261020e/layout2001_v2")
    if not folder.exists():pytest.skip("Archived physical episode absent")
    proposal=json.loads((folder/"input.json").read_text())["proposal"]
    result=audit_attempt(folder,proposal)
    assert result["recorded_physical_pass"]
    assert result["coverage_evidence_complete"]
    assert not result["training_admitted"]


def test_native_obstacle_repair_preserves_task_and_removes_original_failure():
    path=ARTIFACT/"household_layouts_v3/proposals.json"
    if not path.exists():pytest.skip("Feedback repair artifacts absent")
    repaired=json.loads(path.read_text())
    original={c["case_id"]:c for c in json.loads(
        (ARTIFACT/"household_layouts_v2/proposals.json").read_text())["cases"]}
    changed=[]
    before_conflicts=0
    for case in repaired["cases"]:
        parent=original[case["repair"]["parent_case_id"]]
        for key in ("roles","camera","seed","active_arm","human_source_window","split"):
            assert case[key]==parent[key]
        bounds=np.asarray(case["native_scene_obstacles"][0]["bounds_world_m"])
        for before,after in zip(parent["added_distractors"],case["added_distractors"]):
            for key in ("visual","collision","asset_files","yaw_rad","category"):
                assert before[key]==after[key]
            old_v=world_vertices(before);v=world_vertices(after)
            before_conflicts+=int(bbox_margin_conflict(np.stack([old_v.min(0),old_v.max(0)]),bounds,.02))
            assert not bbox_margin_conflict(np.stack([v.min(0),v.max(0)]),bounds,.025)
        if case["repair"]["requires_new_attempt_for_changed_geometry"]:
            changed.append(case["case_id"])
    assert before_conflicts==3
    assert len(changed)==3
    assert set(changed)==set(repaired["changed_geometry_case_ids"])


def test_optional_matte_variant_does_not_modify_geometry_or_claim_renderer_bug():
    candidate=ARTIFACT/"milk_matte_candidate_v1"
    if not candidate.exists():pytest.skip("Optional appearance variant absent")
    before,before_binary=split_glb(ARTIFACT/"household_assets_v3/milk_carton/visual.glb")
    after,after_binary=split_glb(candidate/"visual.glb")
    assert before_binary==after_binary
    assert {k:v for k,v in before.items() if k!="materials"}=={
        k:v for k,v in after.items() if k!="materials"}
    assert before["materials"]!=after["materials"]
    asset=json.loads((candidate/"asset.json").read_text())
    assert asset["collision"]==str(ARTIFACT/"household_assets_v3/milk_carton/collision.obj")
    assert not asset["render_qualified"]
    quarantine=json.loads((candidate/"render_qa_quarantine.json").read_text())
    assert not quarantine["renderer_bug_fix_claimed"]
    assert quarantine["matte_variant_is_optional_engineering_choice"]
