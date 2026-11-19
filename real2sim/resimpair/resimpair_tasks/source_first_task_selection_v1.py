"""Select sources before task capacity, with declared receiving-target choices.

The selected native dimensions and cuDex cache bindings never change. Task
grouping may change only inside the supplied contract; a failed group is not
replaced with smaller objects. This module reads no physical outcome records.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import product
import math
import re

import numpy as np

from resimpair_grasp.common import digest
from resimpair_pickplace.requested_tasks import Rejected
from . import metric_size_balanced_selection_v1 as sizes

VERSION = "source_first_task_selection_v1"
DEFAULT_SIZE_BIN_EDGES_M = sizes.DEFAULT_SIZE_BIN_EDGES_M
DEFAULT_TASK_CONTRACT = {
    "scene_object_count_range": [3, 8],
    "receiving_target_counts": [2, 3, 4],
    "allowed_target_kinds": ["container", "table_region"],
    "maximum_table_region_radius_m": .21,
    "receiving_lane_y_m": [-.85, .85],
    "target_gap_m": .035,
    "preserve_all_selected_objects": True,
    "prefer_containers": True,
}


def _validate_pool(pool, partition):
    for sid, variants in pool.items():
        if not variants:
            raise Rejected("empty_source_variant_list", source_id=sid)
        for entry, descriptor in variants:
            if descriptor["source_id"] != sid or descriptor["partition"] != partition:
                raise Rejected("source_partition_or_identity_conflict_in_pool")
            if not entry.get("variant_id") or not entry.get("asset") or not entry.get("cache"):
                raise ValueError("native_variant_descriptor_and_cache_bindings_required")
            sizes._longest((entry, descriptor)); sizes.footprint((entry, descriptor))


def _quotas(n, weights, bands, seed):
    weights = np.asarray(weights, float)
    if (weights.shape != (len(bands),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("nonnegative_explicit_size_weights_required")
    shares = weights / weights.sum() * n
    counts = np.floor(shares).astype(int)
    order = sorted(range(len(bands)), key=lambda i: (
        -(shares[i] - counts[i]), digest([seed, "size_quota_remainder", bands[i]["id"]])))
    for i in order[:n - int(counts.sum())]:
        counts[i] += 1
    return dict(zip([b["id"] for b in bands], counts.tolist()))


def choose_objects(pool, n, partition, seed, *, mode="source_uniform",
                   size_bin_edges_m=DEFAULT_SIZE_BIN_EDGES_M, size_weights=None):
    """Return exactly N source-distinct cached variants and an unfiltered receipt.

    A single variant is chosen per source before size binning. Additional cache
    variants do not give a source more sampling weight. Weighted mode specifies
    hard quotas; insufficient bins reject without changing their distribution.
    """
    if type(n) is not int or not 3 <= n <= 8 or type(seed) is not int:
        raise ValueError("three_to_eight_objects_and_integer_seed_required")
    _validate_pool(pool, partition)
    bands = sizes._bands(size_bin_edges_m)
    if mode not in {"source_uniform", "size_weighted"}:
        raise ValueError("explicit_supported_sampling_mode_required")
    if mode == "source_uniform" and size_weights is not None:
        raise ValueError("source_uniform_mode_cannot_silently_use_size_weights")
    chosen_variants = {sid: min(variants, key=lambda x: digest(
        [seed, "source_variant", sid, x[0]["variant_id"]])) for sid, variants in pool.items()}
    ordered = sorted(chosen_variants, key=lambda sid: digest([seed, "source_order", sid]))
    receipt = {"schema": VERSION, "mode": mode, "partition": partition, "seed": seed,
        "requested_object_count": n, "size_bin_edges_m": list(size_bin_edges_m),
        "size_weights": deepcopy(size_weights), "source_population": len(pool),
        "selection_uses_physical_success": False, "selection_uses_container_capacity": False,
        "runtime_rescaling": False, "new_grasps": 0, "physics_success": "unmeasured",
        "variant_policy": "one deterministic cached variant per source before size assignment"}
    if len(ordered) < n:
        raise Rejected("not_enough_distinct_admitted_sources_in_partition", selection=receipt)
    if mode == "source_uniform":
        selected = ordered[:n]
        receipt["requested_size_counts"] = None
    else:
        requested = _quotas(n, size_weights, bands, seed)
        selected = []
        receipt["requested_size_counts"] = requested
        available = {}
        for band in bands:
            candidates = [s for s in ordered if sizes._bin(sizes._longest(chosen_variants[s]), bands) == band["id"]]
            available[band["id"]] = len(candidates)
            if len(candidates) < requested[band["id"]]:
                receipt["available_sources_by_size_bin"] = available
                raise Rejected("declared_size_bin_quota_unavailable", size_bin=band["id"], selection=receipt)
            selected.extend(candidates[:requested[band["id"]]])
        selected.sort(key=lambda sid: digest([seed, "source_order", sid]))
        receipt["available_sources_by_size_bin"] = available
    group = [chosen_variants[s] for s in selected]
    receipt["selected_assets"] = [sizes._asset_receipt(x, bands) for x in group]
    receipt["selected_size_counts"] = dict(Counter(x["size_bin"] for x in receipt["selected_assets"]))
    receipt["selected_objects_sha256"] = digest(receipt["selected_assets"])
    return group, receipt


def task_contract(value):
    contract = deepcopy(DEFAULT_TASK_CONTRACT if value is None else value)
    if set(contract) != set(DEFAULT_TASK_CONTRACT):
        raise ValueError("complete_explicit_task_capacity_contract_required")
    counts = contract["receiving_target_counts"]
    if (contract["scene_object_count_range"] != [3, 8]
            or not isinstance(counts, list) or not counts or counts != sorted(set(counts))
            or any(type(k) is not int or not 2 <= k <= 4 for k in counts)
            or contract["preserve_all_selected_objects"] is not True
            or contract["prefer_containers"] is not True):
        raise ValueError("bounded_target_groups_and_preserved_objects_required")
    kinds = contract["allowed_target_kinds"]
    if not isinstance(kinds, list) or not kinds or len(set(kinds)) != len(kinds) or set(kinds) - {"container", "table_region"}:
        raise ValueError("explicit_container_or_table_region_permission_required")
    lane = np.asarray(contract["receiving_lane_y_m"], float)
    values = [contract["maximum_table_region_radius_m"], contract["target_gap_m"]]
    if (lane.shape != (2,) or not np.isfinite(lane).all() or not -.9 <= lane[0] < lane[1] <= .9
            or not np.isfinite(values).all() or not .012 < values[0] <= .21
            or not .025 < values[1] <= .10):
        raise ValueError("bounded_receiving_lane_and_original_clearance_required")
    return contract


def _partitions(indices, k):
    """Unlabelled set partitions, at most S(8,4)=1701 for any supported K."""
    if k == 1:
        yield [indices]; return
    if len(indices) < k:
        return
    first, rest = indices[0], indices[1:]
    for tail in _partitions(rest, k - 1):
        yield [[first], *tail]
    for tail in _partitions(rest, k):
        for position in range(len(tail)):
            yield [([first] + group if i == position else group) for i, group in enumerate(tail)]


def plan_targets(group, bins, partition, seed, *, contract=None, balanced_two_groups_only=False):
    """Assign the same objects to 2–4 permitted targets; no object resampling.

    Subsets use the unchanged center/line/ring packing test. Container choices
    use their real floor and outside dimensions. Table circles are explicit
    goals inside the declared receiving lane, not synthetic container assets.
    """
    contract = task_contract(contract)
    n = len(group)
    if not 3 <= n <= 8 or any(d["partition"] != partition for _, d in group):
        raise ValueError("one_partition_three_to_eight_selected_objects_required")
    excluded = {d["source_id"] for _, d in group}
    admitted = []
    for ref, descriptor in bins:
        if (descriptor["partition"] != partition or descriptor["source_id"] in excluded
                or not descriptor.get("roles", {}).get("stationary_container")
                or re.search(r"\b(soup|filled|water|cereal|fruit|food)\b", descriptor.get("name", ""), re.I)):
            continue
        admitted.append((ref, descriptor))
    radii = [sizes.footprint(x) for x in group]
    subset_options = {}
    for mask in range(1, 1 << n):
        indices = [i for i in range(n) if mask & (1 << i)]
        pack = sizes.slots([radii[i] for i in indices], 1.)
        required = max(float(np.linalg.norm(x)) + r for x, r in zip(pack["offsets"], pack["radii_m"])) + pack["required_boundary_margin_m"]
        options = []
        if "container" in contract["allowed_target_kinds"]:
            compatible = [(ref, d) for ref, d in admitted if sizes.slots(
                [radii[i] for i in indices], d["cavity"]["flat_floor_radius_m"])["passed"]]
            # Minimum fitting exterior reduces scene area; source choices are
            # already frozen and do not depend on this ordering.
            compatible.sort(key=lambda x: (float(np.linalg.norm(np.asarray(x[1]["dimensions_m"][:2]) / 2)),
                digest([seed, mask, x[1]["source_id"], x[0]["sha256"]])))
            if compatible:
                ref, descriptor = compatible[0]
                options.append({"kind": "container", "indices": indices, "descriptor": ref,
                    "source_id": descriptor["source_id"], "partition": partition,
                    "outside_radius_m": float(np.linalg.norm(np.asarray(descriptor["dimensions_m"][:2]) / 2)),
                    "floor_radius_m": descriptor["cavity"]["flat_floor_radius_m"],
                    "packing": sizes.slots([radii[i] for i in indices], descriptor["cavity"]["flat_floor_radius_m"])})
        region_radius = math.ceil((required + 1e-6) * 1000) / 1000
        if "table_region" in contract["allowed_target_kinds"] and region_radius <= contract["maximum_table_region_radius_m"]:
            options.append({"kind": "table_region", "indices": indices,
                "outside_radius_m": region_radius, "floor_radius_m": region_radius,
                "packing": sizes.slots([radii[i] for i in indices], region_radius),
                "visual_marker_status": "contract_coordinates_only_not_rendered_by_retained_compiler"})
        subset_options[mask] = options
    examined, best, best_key = 0, None, None
    lane = contract["receiving_lane_y_m"]
    for k in contract["receiving_target_counts"]:
        if k > n or (balanced_two_groups_only and k != 2):
            continue
        for partitioned in _partitions(list(range(n)), k):
            if balanced_two_groups_only and sorted(map(len, partitioned)) != [n // 2, (n + 1) // 2]:
                continue
            examined += 1
            options = [subset_options[sum(1 << i for i in indices)] for indices in partitioned]
            for targets in product(*options):
                used = {}; consistent = True
                for t in targets:
                    if t["kind"] == "container":
                        old = used.setdefault(t["source_id"], t["descriptor"])
                        consistent &= old == t["descriptor"]
                if not consistent:
                    continue
                span = sum(2 * t["outside_radius_m"] for t in targets) + (k - 1) * contract["target_gap_m"]
                if span > lane[1] - lane[0] or max(t["outside_radius_m"] for t in targets) >= .44:
                    continue
                key = (sum(t["kind"] == "table_region" for t in targets), k, span,
                       digest([seed, "target_assignment", targets]))
                if best_key is None or key < best_key:
                    best, best_key = deepcopy(targets), key
    if best is None:
        raise Rejected("no_targets_for_fixed_object_selection", contract=contract,
            selected_source_ids=[d["source_id"] for _, d in group], radii_m=radii,
            examined_group_partitions=examined, declared_object_count_preserved=n,
            subset_choice_limit="smallest fitting exterior container per subset; not an exhaustive container-combination proof")
    span = best_key[2]; cursor = (lane[0] + lane[1] - span) / 2
    for i, target in enumerate(best):
        radius = target["outside_radius_m"]
        target["id"] = "receiving_" + str(i)
        target["xy"] = [round(.46 - radius - .012, 7), round(cursor + radius, 7)]
        target["yaw"] = 0.
        cursor += 2 * radius + contract["target_gap_m"]
    return {"contract": contract, "targets": list(best), "examined_group_partitions": examined,
        "requested_scene_objects": n, "actual_scene_objects": n, "removed_source_ids": [],
        "target_group_count": len(best), "target_kind_counts": dict(Counter(t["kind"] for t in best)),
        "task_definition_changed_from_fixed_two_containers": len(best) != 2 or any(t["kind"] != "container" for t in best),
        "physical_success": "unmeasured", "goal_grouping_uses_physical_success": False,
        "container_search_limit": "smallest fitting exterior per subset; all supported object group partitions searched"}


def capacity_select(pool, bins, n, partition, seed, *, mode="source_uniform",
                    size_bin_edges_m=DEFAULT_SIZE_BIN_EDGES_M, size_weights=None, task_capacity_contract=None):
    """Compatibility callback for the unchanged two-container coordinator v1."""
    group, receipt = choose_objects(pool, n, partition, seed, mode=mode,
        size_bin_edges_m=size_bin_edges_m, size_weights=size_weights)
    contract = deepcopy(task_capacity_contract or DEFAULT_TASK_CONTRACT)
    if task_capacity_contract is not None and (
            contract.get("receiving_target_counts") != [2]
            or contract.get("allowed_target_kinds") != ["container"]):
        raise ValueError("compatibility_callback_requires_explicit_two_container_contract")
    contract["receiving_target_counts"] = [2]
    contract["allowed_target_kinds"] = ["container"]
    try:
        plan = plan_targets(group, bins, partition, seed, contract=contract, balanced_two_groups_only=True)
    except Rejected as error:
        raise Rejected(error.code, **error.details, selection=receipt) from error
    targets = sorted(plan["targets"], key=lambda t: -len(t["indices"]))
    groups = [[group[i] for i in t["indices"]] for t in targets]
    interleaved = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    by_binding = {digest(ref): (ref, d) for ref, d in bins}
    selected_bins = [by_binding[digest(t["descriptor"])] for t in targets]
    receipt["target_plan"] = plan
    return interleaved, selected_bins, receipt


def legacy_capacity_select(pool, bins, n, partition, seed, *, size_bin_edges_m=DEFAULT_SIZE_BIN_EDGES_M):
    """Named callback preserving v4 selection, accepting diagnostic bin config."""
    from .metric_selection_factory_v1 import source
    bands = sizes._bands(size_bin_edges_m)
    group, selected_bins, receipt = source.capacity_select(pool, bins, n, partition, seed)
    receipt = deepcopy(receipt)
    receipt["selected_assets"] = [sizes._asset_receipt(x, bands) for x in group]
    receipt["selected_size_counts"] = dict(Counter(x["size_bin"] for x in receipt["selected_assets"]))
    return group, selected_bins, receipt
