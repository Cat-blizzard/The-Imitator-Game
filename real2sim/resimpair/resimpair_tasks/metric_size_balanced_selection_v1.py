"""Opt-in size-stratified selection over immutable native grasp-cache entries.

The selection plan is independent of container capacity: first select an exact
number of distinct sources from each declared size bin, then find compatible
native containers using the retained slots rule. A missed quota or insufficient
capacity is a recorded rejection, never a replacement by smaller objects.
No geometry, source partition, grasp, mass or simulator state is changed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from resimpair_grasp.common import digest
from resimpair_pickplace.requested_tasks import Rejected
from . import metric_population_sampler_v3 as retained

VERSION = "metric_size_balanced_selection_v1"
# Exhaustive bands: >=18 cm includes the scarce >=26 cm tail; none are omitted.
DEFAULT_SIZE_BIN_EDGES_M = (0., .08, .12, .18, None)
footprint = retained.footprint
slots = retained.slots
propose = retained.propose


def _bands(edges):
    edges = tuple(edges)
    if (not 2 <= len(edges) <= 9 or edges[0] != 0 or edges[-1] is not None
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in edges[:-1])
            or any(a >= b for a, b in zip(edges[:-2], edges[1:-1]))):
        raise ValueError("ordered_exhaustive_size_bin_edges_required")
    return [{"id": f"size_{i}", "lower_m_inclusive": lo, "upper_m_exclusive": hi}
            for i, (lo, hi) in enumerate(zip(edges, edges[1:]))]


def _longest(item):
    dimensions = np.asarray(item[1]["dimensions_m"], float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("positive_native_descriptor_dimensions_required")
    return float(dimensions.max())


def _bin(length, bands):
    return next(b["id"] for b in bands if length >= b["lower_m_inclusive"]
                and (b["upper_m_exclusive"] is None or length < b["upper_m_exclusive"]))


def _buckets(pool, partition, bands):
    buckets = {b["id"]: defaultdict(list) for b in bands}
    for source, variants in pool.items():
        if not variants:
            raise Rejected("empty_source_variant_list", source_id=source)
        for item in variants:
            entry, descriptor = item
            if descriptor["source_id"] != source or descriptor["partition"] != partition:
                raise Rejected("source_partition_or_identity_conflict_in_pool")
            footprint(item)
            if not entry.get("variant_id"):
                raise ValueError("bound_variant_identity_required")
            buckets[_bin(_longest(item), bands)][source].append(item)
    return buckets


def _quantiles(values):
    return None if not values else dict(zip(
        ["min", "p10", "p25", "median", "p75", "p90", "max"],
        np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist()))


def describe_population(pool, bins, *, partition=None, size_bin_edges_m=DEFAULT_SIZE_BIN_EDGES_M):
    """Pure descriptor/slot diagnostic; dimensions are cached, not measured anew."""
    bands = _bands(size_bin_edges_m)
    parts = [partition] if partition else sorted({d["partition"] for xs in pool.values() for _, d in xs})
    rows = []
    for part in parts:
        selected = {s: xs for s, xs in pool.items() if xs and xs[0][1]["partition"] == part}
        buckets = _buckets(selected, part, bands)
        by_band = []
        for band in bands:
            entries = [item for xs in buckets[band["id"]].values() for item in xs]
            by_band.append({**band, "unique_sources": len(buckets[band["id"]]), "variants": len(entries),
                "longest_m": _quantiles([_longest(x) for x in entries]),
                "footprint_radius_m": _quantiles([footprint(x) for x in entries])})
        containers = [d for _, d in bins if d["partition"] == part]
        rows.append({"partition": part, "unique_sources": len(selected),
            "variants": sum(map(len, selected.values())), "size_bins": by_band,
            "container_sources": len({d["source_id"] for d in containers}),
            "container_variants": len(containers),
            "container_floor_radius_m": _quantiles([d["cavity"]["flat_floor_radius_m"] for d in containers])})
    return {"schema": VERSION, "partitions": rows, "dimensions_basis": "retained_native_descriptor",
        "source_can_appear_in_multiple_bins_if_multiple_cached_sizes_exist": True,
        "physics_success": "unmeasured", "geometry_changed": False}


def size_plan(n, seed, *, size_bin_edges_m=DEFAULT_SIZE_BIN_EDGES_M):
    if type(n) is not int or not 3 <= n <= 8:
        raise ValueError("three_to_eight_objects_required")
    if type(seed) is not int:
        raise ValueError("integer_recorded_seed_required")
    bands = _bands(size_bin_edges_m)
    order = sorted((b["id"] for b in bands), key=lambda b: digest([VERSION, seed, "quota", b]))
    counts = {b["id"]: n // len(bands) for b in bands}
    for key in order[:n % len(bands)]:
        counts[key] += 1
    return {"bins": bands, "requested_counts": counts, "seed": seed,
            "policy": "equal_per_bin_quotas; remainder_assigned_by_recorded_seed_hash"}


def _asset_receipt(item, bands):
    entry, d = item
    return {"source_id": d["source_id"], "variant_id": entry["variant_id"],
        "descriptor": entry.get("asset"), "cache": entry.get("cache"),
        "dimensions_m": list(d["dimensions_m"]), "longest_m": _longest(item),
        "footprint_radius_m": footprint(item), "size_bin": _bin(_longest(item), bands),
        "relative_scale": entry.get("relative_scale", d.get("size_provenance", {}).get("selected_relative_scale")),
        "partition": d["partition"]}


def capacity_select(pool, bins, n, partition, seed, *, size_bin_edges_m=DEFAULT_SIZE_BIN_EDGES_M):
    """Drop-in selector returning ``(interleaved_objects, two_bins, receipt)``.

    Bin quotas use descriptor longest edge; container packing uses the retained
    XY footprint. Source choices never consult physical outcomes or container
    sizes. Every selection is attempted once; callers must retain rejections.
    """
    plan = size_plan(n, seed, size_bin_edges_m=size_bin_edges_m)
    bands = plan["bins"]
    buckets = _buckets(pool, partition, bands)
    receipt = {"schema": VERSION, "partition": partition, "size_plan": plan,
        "available_sources_by_size_bin": {k: len(v) for k, v in buckets.items()},
        "requested_size_counts": dict(plan["requested_counts"]),
        "selected_size_counts": dict.fromkeys(plan["requested_counts"], 0),
        "source_selection_uses_container_capacity": False, "selection_uses_physics_success": False,
        "runtime_rescaling": False, "new_grasps": 0, "physics_success": "unmeasured"}
    chosen, used = [], set()
    # Rare bins go first solely to avoid source aliases spanning size bands;
    # the requested quotas themselves never depend on population availability.
    for band_id in sorted(buckets, key=lambda b: (len(buckets[b]), b)):
        need = plan["requested_counts"][band_id]
        sources = sorted(buckets[band_id], key=lambda s: digest([VERSION, seed, "source", band_id, s]))
        available = [s for s in sources if s not in used]
        if len(available) < need:
            raise Rejected("declared_size_bin_quota_unavailable", size_bin=band_id,
                required=need, available_distinct_sources=len(available), selection=receipt)
        for source in available[:need]:
            item = min(buckets[band_id][source],
                       key=lambda x: digest([VERSION, seed, "variant", source, x[0]["variant_id"]]))
            chosen.append(item); used.add(source)
            receipt["selected_size_counts"][band_id] += 1
    # Spread descending footprints between the same two interleaved groups
    # expected by retained.propose. No selected source/variant is substituted.
    group = sorted(chosen, key=lambda x: (-footprint(x), digest([seed, x[1]["source_id"]])))
    receipt["selected_assets"] = [_asset_receipt(item, bands) for item in group]
    receipt["group_sizes"] = [(n + 1) // 2, n // 2]
    receipt["group_assignment"] = "descending_XY_footprint_alternating_two_groups"
    receipt["group_capacity"] = []
    candidates = []
    for index in range(2):
        radii = [footprint(x) for x in group[index::2]]
        choices = [(ref, d) for ref, d in bins if d["partition"] == partition
                   and d["source_id"] not in used
                   and slots(radii, d["cavity"]["flat_floor_radius_m"])["passed"]]
        choices.sort(key=lambda x: digest([VERSION, seed, "container", index, x[1]["source_id"], x[0]["sha256"]]))
        candidates.append(choices)
        # The slots construction is independent of capacity; this derives its
        # required radius without changing its gaps or floor margin.
        probe = slots(radii, max(radii) * 20 + 1)
        required = max(radii) * 20 + 1 - probe["minimum_boundary_margin_m"] + probe["required_boundary_margin_m"]
        available = [d["cavity"]["flat_floor_radius_m"] for _, d in bins
                     if d["partition"] == partition and d["source_id"] not in used]
        receipt["group_capacity"].append({"group": index, "radii_m": radii,
            "required_floor_radius_m": required, "compatible_container_variants": len(choices),
            "maximum_available_floor_radius_m": max(available, default=None)})
    if not all(candidates):
        raise Rejected("no_container_for_declared_size_plan", selection=receipt,
            missing_groups=[i for i, choices in enumerate(candidates) if not choices])
    selected_bins = None
    for first in candidates[0]:
        second = next((x for x in candidates[1] if x[1]["source_id"] != first[1]["source_id"]
                       or x[0]["sha256"] == first[0]["sha256"]), None)
        if second is not None:
            selected_bins = [first, second]
            break
    if selected_bins is None:
        raise Rejected("no_consistent_container_pair_for_declared_size_plan", selection=receipt)
    receipt["selected_container_variants"] = [ref for ref, _ in selected_bins]
    receipt["selected_container_source_ids"] = [d["source_id"] for _, d in selected_bins]
    receipt["capacity_scope"] = "two_destination_subgroups; collect_all_not_required"
    receipt["all_requested_size_counts_met"] = receipt["requested_size_counts"] == receipt["selected_size_counts"]
    return group, selected_bins, receipt
