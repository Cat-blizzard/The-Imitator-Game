"""Replay boundaries and exact factory inputs without physical execution."""
from copy import deepcopy

import pytest

from resimpair_grasp.common import binding, write
from resimpair_pickplace.requested_tasks import Rejected
from resimpair_tasks import metric_selection_factory_v1 as coordinator
from resimpair_tasks import metric_size_balanced_selection_v1 as selector


def fixture(tmp_path):
    pool = {}; assets = []
    for i in range(3):
        sid = "source_" + str(i)
        descriptor = {"source_id": sid, "partition": "development", "dimensions_m": [.04, .05, .09],
            "half_size": [.02, .025, .045]}
        write(tmp_path / (sid + ".json"), descriptor)
        write(tmp_path / (sid + "_cache.json"), {"test_fixture_only": True})
        entry = {"variant_id": sid + "_v1", "asset": binding(tmp_path / (sid + ".json")),
            "cache": binding(tmp_path / (sid + "_cache.json"))}
        pool[sid] = [(entry, descriptor)]
        assets.append({"source_id": sid, "variant_id": entry["variant_id"], "descriptor": entry["asset"],
            "cache": entry["cache"], "partition": "development", "dimensions_m": descriptor["dimensions_m"]})
    d = {"source_id": "bin", "partition": "development", "cavity": {"flat_floor_radius_m": .2}}
    write(tmp_path / "bin.json", d)
    bins = [(binding(tmp_path / "bin.json"), d)]
    value = {"passed": True, "assets": assets, "selection": {"original": True},
        "containers": [{"descriptor": bins[0][0], "source_id": "bin", "partition": "development", "floor_radius_m": .2}] * 2}
    return pool, bins, value


def test_exact_replay_and_original_rejections_need_no_selector(tmp_path):
    pool, bins, value = fixture(tmp_path)
    group, chosen, receipt = coordinator.replay_selection(value, pool, bins, partition="development", count=3)
    assert group == [pool[row["source_id"]][0] for row in value["assets"]]
    assert chosen == bins * 2 and receipt == value["selection"]
    with pytest.raises(Rejected, match="frozen_selection_rejected") as rejection:
        coordinator.replay_selection({"passed": False, "reason": "no_container"}, pool, bins,
                                     partition="development", count=3)
    assert rejection.value.details["original"]["reason"] == "no_container"


@pytest.mark.parametrize("field,value", [("dimensions_m", [.08, .1, .18]), ("partition", "heldout_asset"),
                                       ("cache", {"path": "other", "sha256": "different"})])
def test_replay_rejects_changed_size_split_or_pose_cache(tmp_path, field, value):
    pool, bins, saved = fixture(tmp_path)
    saved["assets"][0][field] = value
    with pytest.raises(ValueError, match="frozen_asset_cache_size_or_partition_changed"):
        coordinator.replay_selection(saved, pool, bins, partition="development", count=3)


def test_frozen_selector_seed_and_bundle_are_part_of_the_contract(tmp_path):
    schedule = [{"partition": "development", "N": 3, "seed": 5}]
    write(tmp_path / "bundle.json", {"test_fixture_only": True})
    protocol = {"bundle": binding(tmp_path / "bundle.json"), "cases": schedule,
        "candidate": binding(selector.__file__), "size_bin_edges_m": list(selector.DEFAULT_SIZE_BIN_EDGES_M)}
    write(tmp_path / "protocol.json", protocol)
    write(tmp_path / "results.json", {"protocol": binding(tmp_path / "protocol.json"), "rows": schedule})
    config = {"size_bin_edges_m": list(selector.DEFAULT_SIZE_BIN_EDGES_M)}
    assert coordinator._frozen_rows(tmp_path / "results.json", schedule, tmp_path / "bundle.json",
                                    selector.capacity_select, config) == schedule
    changed = deepcopy(schedule); changed[0]["seed"] += 1
    with pytest.raises(ValueError, match="frozen_selection_schedule_bundle_or_policy_changed"):
        coordinator._frozen_rows(tmp_path / "results.json", changed, tmp_path / "bundle.json",
                                 selector.capacity_select, config)


def test_bounded_schedule_and_existing_output_are_preserved(tmp_path):
    schedule = [{"partition": "development", "N": 3, "seed": 5}]
    with pytest.raises(ValueError, match="one_to_twelve"):
        coordinator.schedule_rows(schedule * 13)
    with pytest.raises(ValueError, match="preserve_existing"):
        coordinator.build(tmp_path / "no_bundle", tmp_path, schedule=schedule)


def test_compiler_is_retained_and_no_new_IK_controller_exists():
    from resimpair_tasks import metric_task_factory_v3
    assert coordinator.factory is metric_task_factory_v3
    assert coordinator.factory.compile_exact_scene is metric_task_factory_v3.compile_exact_scene
