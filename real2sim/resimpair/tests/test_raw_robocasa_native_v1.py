"""CPU-only geometry and real ledger checks; no native simulator is started."""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from resimpair_assets import raw_robocasa_native_v1 as intake
from resimpair_grasp.common import binding, read, resolve, write


def source(tmp_path, *, articulated=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=[.06, .08, .10])
    mesh.export(tmp_path / "mesh.obj", digits=12)
    (tmp_path / "model.xml").write_text(
        '<mujoco model="test"><asset><mesh name="m" file="mesh.obj"/>'
        '<material name="color" rgba="0.2 0.3 0.4 1"/></asset>'
        '<worldbody><body>' + ('<joint/>' if articulated else '') +
        '<geom type="mesh" mesh="m" group="1" contype="0" conaffinity="0" material="color"/>'
        '<geom type="mesh" mesh="m" group="0"/></body></worldbody></mujoco>')
    return {"id": "robocasa:test/" + tmp_path.name, "source": "robocasa", "name": "mug test",
        "path": str(tmp_path / "model.xml"), "source_sha": binding(tmp_path / "model.xml")["sha256"],
        "partition": "heldout"}


def test_authored_size_uses_unit_one_and_current_density_without_category_resize(tmp_path):
    raw = source(tmp_path / "source")
    dims = [.18, .07, .11]
    decision = intake.authored_size_decision(raw, dims, {"source_id": raw["id"]}, 42)
    assert decision["source_unit_to_m"] == 1
    assert decision["metric_obb_m"] == sorted(dims)
    assert decision["target_longest_m"] == .18
    assert decision["effective_density_kg_m3"] == 600
    assert "not_independently_measured" in decision["size_evidence"].values()
    policy = intake.retained_policy.metric_policy(decision, {"path": "declared", "sha256": "fixture"})
    assert policy["size"]["mode"] == "source_metric"
    assert policy["size"]["source_unit_to_m"] == 1 and policy["scale_factors"] == [1]
    with pytest.raises(ValueError, match="identity"):
        intake.authored_size_decision({**raw, "source": "objaverse"}, dims, {"source_id": raw["id"]}, 42)


def test_original_collision_volume_and_component_transform_are_preserved(tmp_path):
    raw = source(tmp_path / "source")
    receipt = intake.xml.inspect_raw_xml(raw)
    T = np.diag([2., 3., 4., 1.]); T[:3, 3] = [.11, -.15, .04]
    collision = next(g for g in receipt["geoms"] if g["kind"] == "collision")
    collision["mesh_to_world"] = T.tolist()
    parts, check = intake.load_author_collision_parts(receipt)
    assert len(parts) == 1
    assert np.allclose(parts[0].extents, [.12, .24, .4])
    assert np.allclose(parts[0].center_mass, T[:3, 3])
    assert check["compound_properties"]["collision_shape_volume_sum_m3"] == pytest.approx(.06 * .08 * .1 * 24)
    assert check["parts"][0]["source_mesh_to_XML_world"] == T.tolist()
    assert check["original_author_collisions_imported"] and not check["new_decomposition"]


def test_open_collision_is_rejected_without_visual_hull_substitution(tmp_path):
    raw = source(tmp_path / "source")
    mesh = trimesh.load(Path(raw["path"]).parent / "mesh.obj", force="mesh")
    mesh.update_faces(np.arange(len(mesh.faces) - 1)); mesh.export(Path(raw["path"]).parent / "mesh.obj")
    with pytest.raises(ValueError, match="author_collision_part_not_closed"):
        intake.load_author_collision_parts(intake.xml.inspect_raw_xml(raw))


def test_missing_author_collision_is_rejected(tmp_path):
    raw = source(tmp_path / "source")
    receipt = intake.xml.inspect_raw_xml(raw)
    receipt["geoms"] = [g for g in receipt["geoms"] if g["kind"] == "visual"]
    with pytest.raises(ValueError, match="explicit_author_collision"):
        intake.load_author_collision_parts(receipt)


def test_all_actual_input_bytes_are_frozen_without_mesh_loading(tmp_path):
    raw = source(tmp_path / "source")
    frozen = intake.freeze_source_inputs(raw)
    assert frozen["mesh_loads"] == 0 and frozen["declaration_status"] == "accepted"
    assert {Path(r["path"]).name for r in frozen["files"]} == {"model.xml", "mesh.obj"}
    with (Path(raw["path"]).parent / "mesh.obj").open("a") as out: out.write("\n# changed\n")
    assert intake.freeze_source_inputs(raw) != frozen


def test_registry_reuses_same_source_in_another_directory_without_new_allocation(tmp_path):
    raw = source(tmp_path / "source"); registry = tmp_path / "registry"
    first = intake.prepare(tmp_path / "first", [{"raw_row": raw}], maximum_new_sources=1, registry=registry)
    second = intake.prepare(tmp_path / "second", [{"raw_row": raw}], maximum_new_sources=0, registry=registry)
    assert first["new_allocations"] == 1 and second["new_allocations"] == 0
    assert second["reused_prior_allocations"] == 1
    assert intake._validate(tmp_path / "first")["jobs"][0]["job"] == intake._validate(tmp_path / "second")["jobs"][0]["job"]
    assert intake.status(tmp_path / "second")["counts"] == {"prior_campaign_not_submitted": 1}
    with pytest.raises(ValueError, match="must_resume"):
        intake.prepare(tmp_path / "first", [{"raw_row": raw}], maximum_new_sources=1, registry=registry)


def test_changed_dependency_cannot_reuse_or_replace_an_old_source_claim(tmp_path):
    raw = source(tmp_path / "source"); registry = tmp_path / "registry"
    intake.prepare(tmp_path / "first", [{"raw_row": raw}], maximum_new_sources=1, registry=registry)
    with (Path(raw["path"]).parent / "mesh.obj").open("a") as out: out.write("\n# changed\n")
    with pytest.raises(ValueError, match="dependencies_changed"):
        intake.prepare(tmp_path / "second", [{"raw_row": raw}], maximum_new_sources=1, registry=registry)
    assert not (tmp_path / "second").exists()


def test_unsubmitted_allocations_consume_budget_and_unknown_reservation_never_retries(tmp_path):
    raw = source(tmp_path / "source"); root = tmp_path / "campaign"
    intake.prepare(root, [{"raw_row": raw}], maximum_new_sources=1, registry=tmp_path / "registry")
    protocol = intake._validate(root); ref = protocol["jobs"][0]["job"]; folder = resolve(ref).parent
    write(folder / "reservation.json", {"job": ref, "attempt": 1, "protocol": binding(root / "PROTOCOL.json")})
    assert intake.status(root)["counts"] == {"unknown_reserved_no_terminal": 1}
    with pytest.raises(RuntimeError, match="requires_recovery_no_retry"):
        intake.run(root)
    assert not (folder / "worker_started.json").exists()


def test_invalid_format_is_retained_as_terminal_and_cannot_be_retried(tmp_path):
    raw = source(tmp_path / "source", articulated=True); root = tmp_path / "campaign"
    intake.prepare(root, [{"raw_row": raw}], maximum_new_sources=1, registry=tmp_path / "registry")
    protocol = intake._validate(root); path = resolve(protocol["jobs"][0]["job"])
    assert read(path)["frozen_inputs"]["declaration_status"] == "rejected"
    write(path.parent / "reservation.json", {"job": binding(path), "attempt": 1, "protocol": binding(root / "PROTOCOL.json")})
    result = intake.worker(root, path)
    assert result["status"] == "geometry_or_policy_failed" and result["recorded_settle_steps"] == 0
    assert result["GPU_jobs"] == 0 and result["robot_physical_executions"] == 0
    with pytest.raises(ValueError, match="no_retry"):
        intake.worker(root, path)
    assert intake.status(root)["counts"] == {"completed_awaiting_supervisor_receipt": 1}
    recovered = intake.run(root)
    assert recovered["counts"] == {"geometry_or_policy_failed": 1}
    assert read(path.parent / "terminal.json")["recovered_without_resubmission"]


@pytest.mark.parametrize("kwargs", [{"workers": 3}, {"source_timeout_seconds": 241}, {"wall_seconds": 0}])
def test_out_of_bound_resource_config_rejects_before_allocation(tmp_path, kwargs):
    raw = source(tmp_path / "source")
    with pytest.raises(ValueError, match="bounded_source"):
        intake.prepare(tmp_path / "campaign", [{"raw_row": raw}], maximum_new_sources=1,
                       registry=tmp_path / "registry", **kwargs)
    assert not (tmp_path / "campaign").exists()


def test_new_source_cannot_be_allocated_over_declared_limit(tmp_path):
    raw = source(tmp_path / "source")
    with pytest.raises(ValueError, match="cumulative_allocation"):
        intake.prepare(tmp_path / "campaign", [{"raw_row": raw}], maximum_new_sources=0,
                       registry=tmp_path / "registry")
    assert not (tmp_path / "campaign").exists()


def test_direct_compile_never_overwrites_existing_output(tmp_path):
    raw = source(tmp_path / "source"); output = tmp_path / "output"; output.mkdir()
    with pytest.raises(ValueError, match="fresh_RoboCasa_native_output"):
        intake.compile_raw(raw, output)
