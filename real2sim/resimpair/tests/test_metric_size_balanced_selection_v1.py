"""Meaningful source/size/partition boundaries; no simulator or grasp generation."""
from copy import deepcopy
import json

import pytest

from resimpair_pickplace.requested_tasks import Rejected
from resimpair_tasks import metric_size_balanced_selection_v1 as candidate


def population(partition="development"):
    pool = {}
    for band, length in enumerate([.06, .10, .15, .22]):
        for number in range(3):
            source = f"source_{band}_{number}"
            dimensions = [.035 + number * .001, .04, length]
            descriptor = {"source_id": source, "partition": partition,
                "dimensions_m": dimensions, "half_size": [x / 2 for x in dimensions]}
            entry = {"variant_id": source + "_original", "relative_scale": 1.,
                "asset": {"path": source + ".json", "sha256": source},
                "cache": {"path": source + "_cache.json", "sha256": source + "_cache"}}
            pool[source] = [(entry, descriptor)]
    bins = [({"path": "bin.json", "sha256": "native_container"},
        {"source_id": "container", "partition": partition, "cavity": {"flat_floor_radius_m": .22}})]
    return pool, bins


@pytest.mark.parametrize("n", [3, 5, 8])
@pytest.mark.parametrize("partition", ["development", "heldout_asset"])
def test_exact_uniform_quota_source_identity_and_original_geometry(n, partition):
    pool, bins = population(partition)
    before = json.dumps([pool, bins], sort_keys=True)
    group, selected, receipt = candidate.capacity_select(pool, bins, n, partition, 19)
    assert len(group) == len({d["source_id"] for _, d in group}) == n
    assert receipt["requested_size_counts"] == receipt["selected_size_counts"]
    counts = list(receipt["selected_size_counts"].values())
    assert max(counts) - min(counts) <= 1
    assert all(d["partition"] == partition for _, d in group + selected)
    for entry, descriptor in group:
        assert (entry, descriptor) == pool[descriptor["source_id"]][0]
        assert entry["relative_scale"] == 1.
    for c, (_, descriptor) in enumerate(selected):
        assert candidate.slots([candidate.footprint(x) for x in group[c::2]],
                               descriptor["cavity"]["flat_floor_radius_m"])["passed"]
    assert json.dumps([pool, bins], sort_keys=True) == before
    assert receipt["runtime_rescaling"] is False and receipt["new_grasps"] == 0
    assert receipt["physics_success"] == "unmeasured"


def test_empty_large_band_is_rejected_without_small_object_fallback():
    pool, bins = population()
    pool = {s: xs for s, xs in pool.items() if not s.startswith("source_3_")}
    with pytest.raises(Rejected, match="declared_size_bin_quota_unavailable") as failure:
        candidate.capacity_select(pool, bins, 8, "development", 4)
    assert failure.value.details["size_bin"] == "size_3"
    assert failure.value.details["required"] == 2
    assert failure.value.details["selection"]["requested_size_counts"]["size_3"] == 2


def test_capacity_failure_retains_selected_assets_and_exact_quota():
    pool, bins = population()
    bins[0][1]["cavity"]["flat_floor_radius_m"] = .03
    before = deepcopy(pool)
    with pytest.raises(Rejected, match="no_container_for_declared_size_plan") as failure:
        candidate.capacity_select(pool, bins, 8, "development", 4)
    receipt = failure.value.details["selection"]
    assert receipt["selected_size_counts"] == receipt["requested_size_counts"]
    assert len(receipt["selected_assets"]) == 8
    assert pool == before
    assert all(x["required_floor_radius_m"] > x["maximum_available_floor_radius_m"]
               for x in receipt["group_capacity"])


def test_selection_is_seeded_and_containers_cannot_bias_object_choice():
    pool, bins = population()
    group, _, receipt = candidate.capacity_select(pool, bins, 8, "development", 13)
    larger = deepcopy(bins)
    larger[0][1]["cavity"]["flat_floor_radius_m"] = .45
    repeated, _, other = candidate.capacity_select(pool, larger, 8, "development", 13)
    assert repeated == group and other["selected_assets"] == receipt["selected_assets"]
    assert candidate.capacity_select(pool, bins, 8, "development", 13)[0] == group
    assert any(candidate.capacity_select(pool, bins, 8, "development", seed)[0] != group for seed in range(5))


def test_cross_partition_or_changed_identity_is_rejected():
    pool, bins = population()
    pool["source_0_0"][0][1]["partition"] = "heldout_asset"
    with pytest.raises(Rejected, match="partition_or_identity_conflict"):
        candidate.capacity_select(pool, bins, 3, "development", 1)


def test_same_source_cannot_fill_two_size_bands():
    pool, bins = population()
    original = pool["source_0_0"][0]
    larger = deepcopy(original)
    larger[0]["variant_id"] = "additional_bound_variant"
    larger[1]["dimensions_m"][2] = .20
    larger[1]["half_size"][2] = .10
    pool["source_0_0"].append(larger)
    group, _, _ = candidate.capacity_select(pool, bins, 8, "development", 5)
    assert len({d["source_id"] for _, d in group}) == 8


def test_large_tail_is_included_not_silently_filtered():
    pool, bins = population()
    for key in list(pool):
        if key.startswith("source_3_"):
            pool[key][0][1]["dimensions_m"][2] = .35
            pool[key][0][1]["half_size"][2] = .175
    _, _, receipt = candidate.capacity_select(pool, bins, 8, "development", 1)
    assert sum(x["longest_m"] == .35 for x in receipt["selected_assets"]) == 2
    assert candidate.describe_population(pool, bins)["partitions"][0]["size_bins"][-1]["unique_sources"] == 3


@pytest.mark.parametrize("edges", [[0, .1, .08, None], [.01, .1, None], [0, .1, .2], [0, float("nan"), None]])
def test_invalid_or_non_exhaustive_bin_plan_is_rejected(edges):
    with pytest.raises(ValueError, match="size_bin_edges"):
        candidate.size_plan(3, 1, size_bin_edges_m=edges)


def test_original_slots_and_propose_are_reused():
    assert candidate.slots is candidate.retained.slots
    assert candidate.propose is candidate.retained.propose
    assert candidate.footprint is candidate.retained.footprint
