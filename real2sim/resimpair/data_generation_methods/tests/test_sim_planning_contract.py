from copy import deepcopy
import json
import numpy as np
import pytest

from data_generation_methods.human_to_robot_pipeline.planning_contract import (
    build_apple_contract, validate_contract, native_adapter_check, check_binding,
)
from data_generation_methods.human_to_robot_pipeline.structured_randomization import (
    propose, screen_layout, normalized_assets, segment_distance,
    generate,
)


@pytest.fixture
def contract():
    return build_apple_contract()


def test_real_contract_is_bound_and_valid(contract):
    assert validate_contract(contract)["valid"]


def test_loosening_hard_constraint_is_rejected(contract):
    contract["hard_rules"]["physical_grasp_release_support_required"] = False
    with pytest.raises(ValueError, match="Hard rules"):
        validate_contract(contract, False)


def test_semantic_mutation_cannot_be_silently_accepted(contract):
    contract["final_relation"] = "apple merely near the bowl"
    with pytest.raises(ValueError, match="reviewed source"):
        validate_contract(contract)


def test_controller_cannot_approve_itself(contract):
    contract["independent_verifier"] = deepcopy(contract["expert"])
    with pytest.raises(ValueError, match="own verifier"):
        validate_contract(contract, False)


def test_missing_file_binding_is_rejected():
    with pytest.raises(ValueError, match="Missing"):
        check_binding({"path": "none", "other": 1})


def test_current_adapter_rejects_silent_geometry_override(contract):
    result = native_adapter_check({"layout_mode": "explicit_proposed_geometry",
                                   "added_distractors": [{"instance": "x"}]}, contract)
    assert not result["executable_by_current_collector"]
    assert len(result["blockers"]) == 2


def test_existing_seed_only_adapter_stays_supported(contract):
    result = native_adapter_check({"layout_mode": "existing_seed_only"}, contract)
    assert result["executable_by_current_collector"]
    assert not result["simulation_executed"]


@pytest.fixture
def layout():
    return {
        "roles": {
            "source_container": {"center": [0., .145, 0.]},
            "destination_container": {"center": [0., -.145, 0.], "initially_empty": True},
            "selected_object": {"initial_relation": "inside_source_container"},
        }, "active_arm": 0, "added_distractors": [],
        "camera": {"eye": [-.5, 0., .73], "target": [0.,0.,.1], "fov_radians":1.3},
        "critical_event_probe_points": {"lift": [0.,.145,.2], "release":[0.,-.145,.08]},
    }


def test_clear_geometry_is_only_proxy_pass(layout):
    result = screen_layout(layout)
    assert result["coarse_screen_passed"] and not result["physical_success_verified"]
    assert "IK and complete robot swept collision" in result["still_required"]


def test_task_ambiguous_clutter_rejected(layout):
    layout["added_distractors"] = [{"category":"bowl", "center":[.17,.34,.03], "radius_m":.025}]
    assert "distractor_creates_task_ambiguity" in screen_layout(layout)["reasons"]


def test_empty_destination_invariant_is_hard(layout):
    layout["roles"]["destination_container"]["initially_empty"] = False
    assert not screen_layout(layout)["coarse_screen_passed"]


def test_path_blocking_clutter_rejected(layout):
    layout["added_distractors"] = [{"category":"pear", "center":[0.,0.,.03], "radius_m":.025}]
    assert "distractor_blocks_reserved_transfer_corridor" in screen_layout(layout)["reasons"]


def test_camera_looking_away_rejects_key_events(layout):
    layout["camera"]["target"] = [-1.,0.,.8]
    assert "critical_event_outside_camera" in screen_layout(layout)["reasons"]


def test_segment_distance_is_clamped_at_endpoints():
    assert segment_distance([2,0], [0,0], [1,0]) == 1.


def test_native_catalog_geometry_used_without_invented_size():
    row = {"uid":"a", "name":"pear", "adapter":{"bounds_m":[[-1,-2,0],[1,2,3]],
          "visual":"a.glb", "collision":"a.obj", "visual_sha256":"v", "collision_sha256":"c"}}
    result = normalized_assets([row])[0]
    assert result["bounds_rotated_m"] == row["adapter"]["bounds_m"]
    assert result["files"]["a.obj"] == "c"


def test_bounded_proposal_generator_preserves_case_list_and_native_arm(contract):
    with open("/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/fruit_adapters_v2/catalog.json") as source:
        catalog = json.load(source)["assets"]
    result = generate(contract, catalog, count=1)
    assert len(result["cases"]) == 3
    assert len({case["case_id"] for case in result["cases"]}) == 3
    for case in result["cases"]:
        selected = np.random.default_rng(case["seed"]+730001).choice(len(catalog), 2, replace=False)
        assert catalog[int(selected[case["active_arm"]])]["name"] == "apple"
        assert not case["simulation_executed"] and not case["training_admitted"]
