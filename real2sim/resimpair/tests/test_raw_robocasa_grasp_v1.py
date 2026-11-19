"""Read-only real-native provenance plus isolated four-job preparation tests.

The recorded four native artifacts are integration fixtures. No solver, native
cooking, dynamics, GPU query, worker subprocess or model API is run here.
"""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from resimpair_assets import raw_robocasa_grasp_v1 as bridge
from resimpair_grasp.common import ROOT, read, resolve, binding

INDEX = ROOT / "reports/raw_asset_rollout_20261031/robocasa_xml_native_v2/SELECTED_EXISTING_NATIVE_CANDIDATES.json"
RELEASE = ROOT / "validation/raw_asset_rollout_20261031/robocasa_population_r1/RELEASE_GATE.json"
SOLVER = ROOT / "runs/raw_asset_rollout_20261031_v1/protocol.json"
pytestmark = pytest.mark.skipif(not INDEX.is_file() or not RELEASE.is_file(), reason="recorded XML-native integration artifacts required")


def prepared(tmp_path):
    root = tmp_path / "pilot"
    result = bridge.prepare(root, INDEX, RELEASE, SOLVER, registry=tmp_path / "registry")
    return root, result


def test_actual_XML_source_chain_matches_native_candidate_without_new_execution():
    value = read(INDEX)
    for entry in value["entries"]:
        proof = bridge.native_producer(entry, value["population_protocol"])
        assert proof["raw_row"]["source"] == "robocasa"
        assert proof["new_native_compilations"] == proof["new_CPU_dynamic_steps"] == 0
        assert proof["recorded_prior_CPU_settle_steps"] > 0


def test_changed_native_property_cannot_be_published_as_the_original_producer():
    value = read(INDEX); entry = deepcopy(value["entries"][0])
    entry["native_support"] = value["entries"][1]["native_support"]
    with pytest.raises(ValueError, match="actual_qualified_producer_output"):
        bridge.native_producer(entry, value["population_protocol"])


def test_prepare_freezes_four_jobs_and_keeps_original_XML_metadata_without_CPU_receipts(tmp_path):
    root, result = prepared(tmp_path); protocol = bridge.validate(root)
    assert result["registered_jobs"] == 4 and result["GPU_submitted"] == 0
    assert result["total_optimizer_candidates_if_all_jobs_run"] == 1024
    assert protocol["gpu_index"] == 3 and protocol["maximum_parallel_GPU_jobs"] == 1
    assert protocol["minimum_free_GPU_GiB"] == 12
    assert protocol["GPU_timeout_seconds"] == protocol["export_timeout_seconds"] == 120
    assert not list(root.glob("**/*reservation.json")) and not (root / "START.json").exists()
    assert not list(root.glob("cpu/*/*terminal.json")) and not list(root.glob("cpu/*/*result.json"))
    for spec in protocol["job_specs"]:
        mirror = read(resolve(spec["metadata_mirror"]))
        assert mirror["metadata_mirror_only"] and mirror["CPU_execution_claimed_here"] is False
        assert mirror["raw_row"] == spec["XML_native_producer"]["raw_row"]
    assert read(root / "native_cache_index.json")["entries"] == []


def test_moving_directory_does_not_erase_existing_geometry_allocation(tmp_path):
    root, _ = prepared(tmp_path)
    with pytest.raises(ValueError, match="existing_exact_geometry_grasp_allocation"):
        bridge.prepare(tmp_path / "another", INDEX, RELEASE, SOLVER, registry=tmp_path / "registry")
    assert not (tmp_path / "another").exists()
    assert len(list((tmp_path / "registry/claims").glob('*.json'))) == 4


def test_parent_GPU_ledger_release_is_required_before_any_submission_or_START(tmp_path):
    root, _ = prepared(tmp_path)
    with pytest.raises(ValueError, match="parent_GPU_ledger_review"):
        bridge.run(root)
    assert not list(root.glob("gpu/*/reservation.json")) and not (root / "START.json").exists()


def test_unrun_input_is_not_a_publishable_grasp_result(tmp_path):
    root, _ = prepared(tmp_path)
    with pytest.raises(ValueError, match="four_GPU_outcomes"):
        bridge.publish(root, tmp_path / "unused_baseline", tmp_path / "unused_receivers", tmp_path / "publication")
    assert not (tmp_path / "publication").exists()


def test_mutated_prepared_geometry_is_rejected(tmp_path):
    root, _ = prepared(tmp_path); path = next((root / "gpu").glob('*/job.json'))
    job = read(path); job["asset"] = read(INDEX)["entries"][1]["asset"]
    # Deliberately change a fixture, never production evidence.
    job["seed"] += 1; path.write_text(json.dumps(job))
    with pytest.raises(ValueError, match="prepared_GPU_job_or_geometry_changed"):
        bridge.validate(root)


def test_fixed_scope_and_existing_GPU_index_are_enforced(tmp_path):
    with pytest.raises(ValueError, match="shared_GPU_index"):
        bridge.prepare(tmp_path / "gpu4", INDEX, RELEASE, SOLVER, gpu_index=4, registry=tmp_path / "registry")
    assert not (tmp_path / "gpu4").exists()
