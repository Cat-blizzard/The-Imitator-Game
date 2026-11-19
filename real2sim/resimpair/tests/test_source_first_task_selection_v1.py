from copy import deepcopy

import numpy as np
import pytest

from resimpair_grasp.common import digest
from resimpair_pickplace.requested_tasks import Rejected, table_fit, check_separation
from resimpair_tasks import metric_selection_factory_v1 as retained
from resimpair_tasks import source_first_factory_adapter_v1 as adapter
from resimpair_tasks import source_first_task_selection_v1 as select


def population(lengths=(.05, .07, .09, .11, .14, .16, .19, .22), partition="development"):
    result = {}
    for i, length in enumerate(lengths):
        sid = "source_" + str(i)
        d = {"source_id": sid, "partition": partition, "dimensions_m": [.04, .04, length],
             "half_size": [.02, .02, length / 2]}
        e = {"variant_id": sid + "_cache", "asset": {"path": sid + ".json", "sha256": sid},
             "cache": {"path": sid + "_cache.json", "sha256": sid + "_cache"}}
        result[sid] = [(e, d)]
    return result


def container(capacity=.16, partition="development", source="bin", radius=.18):
    return ({"path": source + ".json", "sha256": source},
            {"source_id": source, "partition": partition, "name": "empty tray",
             "roles": {"stationary_container": True},
             "dimensions_m": [2 * radius / np.sqrt(2)] * 2 + [.03],
             "cavity": {"flat_floor_radius_m": capacity}})


def narrow_objects(n=8, radius=.035):
    p = population([.10] * n)
    for xs in p.values():
        xs[0][1]["half_size"][:2] = [radius / np.sqrt(2)] * 2
        xs[0][1]["dimensions_m"][:2] = [2 * radius / np.sqrt(2)] * 2
    return list(p[s][0] for s in sorted(p))


def test_source_uniform_identity_ignores_container_and_physical_results():
    p = population(); before = digest(p)
    chosen, receipt = select.choose_objects(p, 5, "development", 42)
    assert len({d["source_id"] for _, d in chosen}) == 5
    assert digest(p) == before
    altered = deepcopy(p)
    for xs in altered.values():
        xs[0][1]["last_physical_success"] = False
        xs[0][0]["physics_failure_count"] = 10000
    other, second = select.choose_objects(altered, 5, "development", 42)
    assert receipt == second
    assert [d["source_id"] for _, d in chosen] == [d["source_id"] for _, d in other]


def test_source_count_not_variant_count_determines_probability():
    p = population()
    expected, _ = select.choose_objects(p, 3, "development", 95)
    for i in range(20):
        e, d = deepcopy(p["source_0"][0]); e["variant_id"] = "extra_" + str(i)
        p["source_0"].append((e, d))
    actual, _ = select.choose_objects(p, 3, "development", 95)
    assert [d["source_id"] for _, d in actual] == [d["source_id"] for _, d in expected]


def test_weighted_distribution_is_declared_and_exact():
    chosen, receipt = select.choose_objects(population(), 8, "development", 2,
        mode="size_weighted", size_weights=[1, 1, 1, 1])
    assert receipt["selected_size_counts"] == {"size_0": 2, "size_1": 2, "size_2": 2, "size_3": 2}
    assert len(chosen) == 8
    with pytest.raises(Rejected, match="declared_size_bin_quota_unavailable"):
        select.choose_objects(population([.04] * 20), 8, "development", 2,
            mode="size_weighted", size_weights=[0, 0, 0, 1])


@pytest.mark.parametrize("changes", [{"mode": "source_uniform", "size_weights": [1, 1, 1, 1]},
                                    {"mode": "size_weighted", "size_weights": [1, -1, 1, 1]},
                                    {"mode": "unsupported"}])
def test_ambiguous_or_invalid_distribution_rejected(changes):
    with pytest.raises(ValueError):
        select.choose_objects(population(), 3, "development", 5, **changes)


def test_partitions_and_original_dimensions_are_preserved():
    p = population(); before = deepcopy(p)
    p["source_0"][0][1]["partition"] = "heldout_asset"
    with pytest.raises(Rejected, match="source_partition_or_identity_conflict"):
        select.choose_objects(p, 3, "development", 3)
    assert before["source_1"] == p["source_1"]


def test_more_targets_keep_all_objects_instead_of_selecting_small_replacements():
    group = narrow_objects(); before = digest(group)
    contract = deepcopy(select.DEFAULT_TASK_CONTRACT)
    contract["allowed_target_kinds"] = ["container"]
    plan = select.plan_targets(group, [container(.095, radius=.11)], "development", 9, contract=contract)
    assert plan["target_group_count"] == 4
    assert sorted(i for t in plan["targets"] for i in t["indices"]) == list(range(8))
    assert plan["removed_source_ids"] == [] and digest(group) == before
    assert all(t["packing"]["passed"] for t in plan["targets"])


def test_region_fallback_requires_explicit_contract_permission():
    group = narrow_objects(5, .06)
    plan = select.plan_targets(group, [], "development", 4)
    assert plan["target_kind_counts"] == {"table_region": plan["target_group_count"]}
    no_regions = deepcopy(select.DEFAULT_TASK_CONTRACT)
    no_regions["allowed_target_kinds"] = ["container"]
    with pytest.raises(Rejected, match="no_targets_for_fixed_object_selection") as error:
        select.plan_targets(group, [], "development", 4, contract=no_regions)
    assert error.value.details["selected_source_ids"] == [d["source_id"] for _, d in group]


def test_cross_split_and_operated_sources_cannot_be_receiving_assets():
    group = narrow_objects(3)
    no_regions = deepcopy(select.DEFAULT_TASK_CONTRACT); no_regions["allowed_target_kinds"] = ["container"]
    for bins in [[container(partition="heldout_asset")], [container(source=group[0][1]["source_id"])]]:
        with pytest.raises(Rejected, match="no_targets_for_fixed_object_selection"):
            select.plan_targets(group, bins, "development", 8, contract=no_regions)


def test_original_geometry_checks_and_table_goal_coordinates_are_used():
    group = narrow_objects(3, .035)
    before = digest(group)
    plan = select.plan_targets(group, [], "development", 18)
    spec, allocation = adapter.propose("fixture_scene", group, plan, 18, ({}, {}))
    assert digest(group) == before
    assert len(spec["objects"]) == 3 and len(spec["variant_choices"]) == 3
    full = next(t for t in spec["tasks"] if t["id"] == "distribute_all")
    assert sorted(p["object"] for p in full["placements"]) == ["object_0", "object_1", "object_2"]
    targets = {"object_" + str(i): (t, offset) for t in plan["targets"]
               for i, offset in zip(t["indices"], t["packing"]["offsets"])}
    occupied = []
    for obj in spec["objects"]:
        radius = select.sizes.footprint(group[int(obj["id"].split("_")[-1])])
        table_fit(obj["xy"], radius); check_separation(obj["xy"], radius, occupied)
        occupied.append((obj["id"], np.array(obj["xy"]), radius))
    for p in full["placements"]:
        target, offset = targets[p["object"]]
        assert np.allclose(p["target"]["xy"], np.array(target["xy"]) + offset)
    assert allocation["all_selected_objects_preserved"]


def test_factory_is_reused_and_schedule_cannot_reset_existing_output(tmp_path):
    assert adapter.coordinator is retained
    assert adapter.coordinator.factory.compile_requests is retained.factory.compile_requests
    with pytest.raises(ValueError, match="preserve_existing"):
        adapter.build(tmp_path / "bundle", tmp_path, schedule=[{"N": 3, "partition": "development", "seed": 2}])
    with pytest.raises(ValueError, match="one_to_twelve"):
        adapter.build(tmp_path / "bundle", tmp_path / "new", schedule=[{"N": 3, "partition": "development", "seed": 2}] * 13)


def test_legacy_callback_preserves_selection_and_compatibility_permissions():
    pool = population()
    bins = [container()]
    first, first_bins, _ = retained.source.capacity_select(pool, bins, 3, "development", 27)
    second, second_bins, receipt = select.legacy_capacity_select(pool, bins, 3, "development", 27)
    assert first == second and first_bins == second_bins
    assert len(receipt["selected_assets"]) == 3
    with pytest.raises(ValueError, match="explicit_two_container_contract"):
        select.capacity_select(pool, bins, 3, "development", 27,
                               task_capacity_contract=select.DEFAULT_TASK_CONTRACT)
