"""Thin opt-in coordinator for selection callbacks and the retained factory.

An existing frozen selection can be replayed without selecting any new object
or container. The same entry point also accepts a file-backed selection policy
for new explicitly bounded preparations. It does not run IK, physics or APIs,
and does not register an execution allocation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected
from . import metric_population_sampler_v4 as source
from . import metric_task_factory_v3 as factory
from . import metric_size_balanced_selection_v1 as balanced

VERSION = "selection_callback_metric_factory_v1"


def schedule_rows(schedule):
    if not isinstance(schedule, list) or not 1 <= len(schedule) <= 12:
        raise ValueError("explicit_one_to_twelve_scene_schedule_required")
    result = deepcopy(schedule)
    for row in result:
        if (set(row) != {"partition", "N", "seed"} or not isinstance(row["partition"], str)
                or type(row["N"]) is not int or not 3 <= row["N"] <= 8 or type(row["seed"]) is not int):
            raise ValueError("explicit_partition_count_seed_required")
    return result


def replay_selection(value, pool, bins, *, partition, count):
    """Resolve exact selected variants; no fallback, RNG or pose re-generation."""
    if value.get("passed") is not True:
        raise Rejected("frozen_selection_rejected", original=deepcopy(value))
    by_variant = {(sid, e["variant_id"]): (e, d) for sid, variants in pool.items() for e, d in variants}
    group = []
    for row in value["assets"]:
        key = (row["source_id"], row["variant_id"])
        if key not in by_variant:
            raise ValueError("frozen_variant_not_in_exact_bundle")
        item = by_variant[key]; entry, descriptor = item
        if (entry["asset"] != row["descriptor"] or entry["cache"] != row["cache"]
                or descriptor["dimensions_m"] != row["dimensions_m"]
                or descriptor["partition"] != row["partition"] or row["partition"] != partition):
            raise ValueError("frozen_asset_cache_size_or_partition_changed")
        group.append(item)
    if len(group) != count or len({d["source_id"] for _, d in group}) != count:
        raise ValueError("frozen_distinct_object_count_changed")
    selected = []
    for row in value["containers"]:
        matches = [(ref, d) for ref, d in bins if ref == row["descriptor"]]
        if len(matches) != 1:
            raise ValueError("frozen_container_not_in_exact_bundle")
        ref, d = matches[0]
        if (d["source_id"] != row["source_id"] or d["partition"] != partition
                or d["cavity"]["flat_floor_radius_m"] != row["floor_radius_m"]):
            raise ValueError("frozen_container_identity_capacity_or_partition_changed")
        selected.append((ref, d))
    if len(selected) != 2:
        raise ValueError("exact_two_container_bindings_required")
    return group, selected, deepcopy(value["selection"])


def _frozen_rows(path, schedule, bundle, policy, config):
    result = read(path); protocol = read(resolve(result["protocol"]))
    if (protocol["bundle"] != binding(bundle) or protocol["cases"] != schedule
            or protocol["candidate"] != source.callback_binding(policy)["implementation"]
            or tuple(protocol["size_bin_edges_m"]) != tuple(config["size_bin_edges_m"])):
        raise ValueError("frozen_selection_schedule_bundle_or_policy_changed")
    if len(result["rows"]) != len(schedule):
        raise ValueError("frozen_selection_row_count_changed")
    for item, expected in zip(result["rows"], schedule):
        if {k: item[k] for k in ("partition", "N", "seed")} != expected:
            raise ValueError("frozen_selection_row_identity_changed")
    return result["rows"]


def build(bundle, output, *, schedule, selection_policy=balanced.capacity_select,
          selection_config=None, frozen_selections=None, object_pose_proposal=None):
    """Prepare at most 12 scenes through existing proposal and compiler tools.

    ``frozen_selections`` replays the candidate arm of a bound comparison. Each
    accepted selection gets one retained layout proposal and one factory call.
    Refusals, including layout failures, remain in the original denominator.
    """
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    schedule = schedule_rows(schedule)
    if output.exists():
        raise ValueError("preserve_existing_selection_factory_no_resample")
    config = deepcopy(selection_config or {"size_bin_edges_m": list(balanced.DEFAULT_SIZE_BIN_EDGES_M)})
    selector_ref = source.callback_binding(selection_policy)
    pose_ref = source.callback_binding(object_pose_proposal)
    frozen = _frozen_rows(frozen_selections, schedule, bundle, selection_policy, config) if frozen_selections else None
    b, entries, population, bins = source.admitted_pool(bundle)
    input_digest = digest([population, bins])
    contract = {"schema": VERSION, "bundle": binding(bundle), "schedule": schedule,
        "selection_policy": selector_ref, "selection_config": config,
        "frozen_selections": binding(frozen_selections) if frozen_selections else None,
        "object_pose_proposal": pose_ref, "implementation": binding(__file__),
        "retained_propose": binding(source.retained.__file__), "retained_factory": binding(factory.__file__),
        "retained_geometry_compiler": binding(factory.retained.__file__),
        "maximum_scene_compile_calls": len(schedule), "maximum_layout_calls_per_selected_group": 1,
        "source_or_container_reselection": False if frozen else "explicit_single_selector_call_per_schedule_row",
        "layout_policy": "retained finite placement loop; no changed seed or second layout call after failure",
        "new_physics": 0, "new_IK": 0, "new_GPU": 0, "model_API_calls": 0,
        "execution_allocation_registered": False, "default_promoted": False}
    write(output / "PROTOCOL.json", contract)
    static = factory.static_source_catalog(resolve(b["static_catalog"]), output / "static_inputs")
    rows = []; compile_calls = 0
    for index, setting in enumerate(schedule):
        folder = output / "cases" / f"{index:02d}"
        start = time.monotonic()
        row = {"index": index, **setting, "selection_passed": False, "layout_proposed": False,
            "factory_called": False, "compiled": False, "compiled_tasks": 0,
            "physical_success": "unmeasured", "new_IK": 0, "new_physics": 0}
        pool = {sid: variants for sid, variants in population.items()
                if variants[0][1]["partition"] == setting["partition"]}
        try:
            if frozen is not None:
                saved = frozen[index]["methods"]["size_first_v1"]
                write(folder / "original_selection.json", saved)
                group, selected_bins, receipt = replay_selection(saved, pool, bins,
                    partition=setting["partition"], count=setting["N"])
            else:
                group, selected_bins, receipt = selection_policy(pool, bins, setting["N"],
                    setting["partition"], setting["seed"], **config)
            row["selection_passed"] = True
            write(folder / "selection.json", receipt)
            scene_id = "selected_" + digest({"contract": digest(contract), "index": index, "selection": receipt})[:32]
            request, allocation = source.propose(scene_id, group, selected_bins, setting["seed"], static,
                scene_index=index, object_pose_proposal=object_pose_proposal)
            row["layout_proposed"] = True
            request["distractors"]["exclude_source_ids"] = sorted(d["source_id"] for _, d in group)
            write(folder / "capacity_and_layout.json", allocation)
            write(folder / "requests.json", {"schema": factory.VERSION, "scenes": [request]})
            # compile_requests calls the unchanged compile_exact_scene internally;
            # no redundant preflight compile, IK or copied compiler is needed.
            compile_calls += 1; row["factory_called"] = True
            summary = factory.compile_requests(bundle, folder / "requests.json", folder / "factory", setting["seed"])
            row.update(factory_summary=summary, compiled=summary["compiled_scenes"] == 1,
                       compiled_tasks=summary["compiled_tasks"])
            if not row["compiled"]:
                row["factory_rejections"] = read(folder / "factory/scenes.json")["scenes"]
            else:
                registered = read(folder / "factory/cohort.json")["cases"]
                scene = read(resolve(registered[0]["scene"]))
                operated = {d["source_id"]: (e, d) for e, d in group}
                actual = scene["metric_variants"]
                if set(v["source_id"] for v in actual.values()) != set(operated):
                    raise ValueError("compiled_object_source_set_changed")
                for instance, value in actual.items():
                    entry, descriptor = operated[value["source_id"]]
                    obj = next(o for o in scene["objects"] if o["id"] == instance)
                    if (value["asset"] != entry["asset"] or value["cache"] != entry["cache"]
                            or value["variant_id"] != entry["variant_id"] or obj["half_size"] != descriptor["half_size"]):
                        raise ValueError("compiled_asset_cache_or_size_changed")
                row["operated_source_size_cache_bindings_preserved"] = True
                row["scene"] = registered[0]["scene"]
                row["prepared_cohort"] = binding(folder / "factory/cohort.json")
        except (Rejected, ValueError, RuntimeError, KeyError) as error:
            row.update(reason=getattr(error, "code", type(error).__name__),
                       details=getattr(error, "details", {"message": str(error)}))
            # A failed post-compile binding check cannot remain labelled passed.
            row["compiled"] = False
        row["seconds"] = time.monotonic() - start
        write(folder / "RESULT.json", row); rows.append(row)
        print({"case": index, "compiled": row["compiled"], "reason": row.get("reason")}, flush=True)
    if digest([population, bins]) != input_digest:
        raise ValueError("coordinator_mutated_original_population")
    result = {"schema": VERSION, "protocol": binding(output / "PROTOCOL.json"), "rows": rows,
        "requested_scenes": len(schedule), "selection_passed": sum(r["selection_passed"] for r in rows),
        "layout_proposed": sum(r["layout_proposed"] for r in rows), "actual_scene_compile_calls": compile_calls,
        "compiled_scenes": sum(r["compiled"] for r in rows),
        "compiled_tasks": sum(r["compiled_tasks"] for r in rows if r["compiled"]),
        "all_selection_and_layout_rejections_retained": True, "original_input_population_unchanged": True,
        "physical_success": "unmeasured", "new_physics": 0, "new_IK": 0, "new_GPU": 0,
        "model_API_calls": 0, "default_promoted": False, "execution_allocation_registered": False}
    write(output / "RESULTS.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "output", "schedule", "frozen-selections"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(build(args.bundle, args.output, schedule=read(args.schedule), frozen_selections=args.frozen_selections))
