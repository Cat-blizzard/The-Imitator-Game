"""Thin request adapter for source-first target plans and the retained factory.

Only selection and request authoring are new. Compilation, native descriptor
loading, contact-independent geometry rules, task construction and clutter
placement all remain in metric_selection_factory_v1.factory.compile_requests.
No execution campaign, planner, physics worker or model client is called.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected, check_separation, table_fit
from . import metric_selection_factory_v1 as coordinator
from . import source_first_task_selection_v1 as selection

VERSION = "source_first_factory_adapter_v1"


def propose(identity, group, target_plan, seed, static):
    """Write original-schema tasks for every selected source and declared target.

    Goal regions reserve real tabletop area during initial placement. They are
    encoded as ordinary explicit table goals; visual markers are not claimed.
    """
    rng = np.random.default_rng(seed)
    targets = target_plan["targets"]
    occupied, containers, bindings = [], [], {}
    for target in targets:
        xy, radius = np.asarray(target["xy"], float), target["outside_radius_m"]
        table_fit(xy, radius); check_separation(xy, radius, occupied)
        occupied.append((target["id"], xy, radius))
        if target["kind"] == "container":
            containers.append({"id": target["id"], "source_id": target["source_id"],
                               "xy": xy.tolist(), "yaw": target["yaw"]})
            previous = bindings.setdefault(target["source_id"], target["descriptor"])
            if previous != target["descriptor"]:
                raise ValueError("one_exact_container_variant_per_source_required")
    objects, pose_receipts = [], []
    for index, (entry, descriptor) in enumerate(group):
        radius = selection.sizes.footprint((entry, descriptor))
        x_range = [-.46 + radius + .012, -.012 - radius]
        y_range = [-.90 + radius + .012, .90 - radius - .012]
        attempts = []
        if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
            raise Rejected("fixed_asset_exceeds_declared_initial_workspace", source_id=descriptor["source_id"])
        for proposal in range(128):
            xy = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
            try:
                table_fit(xy, radius); check_separation(xy, radius, occupied)
                break
            except Rejected as error:
                attempts.append({"proposal": proposal, "reason": error.code})
        else:
            raise Rejected("bounded_initial_object_layout_exhausted", object_index=index,
                           source_id=descriptor["source_id"], attempts=128, pose_attempts=attempts)
        oid = "object_" + str(index)
        occupied.append((oid, xy, radius))
        objects.append({"id": oid, "source_id": descriptor["source_id"],
                        "xy": np.round(xy, 7).tolist(), "yaw": float(rng.uniform(-np.pi, np.pi))})
        pose_receipts.append({"object": oid, "source_id": descriptor["source_id"],
            "accepted_proposal": proposal, "rejected_candidates": attempts})
    groups = []
    for target in targets:
        placements = []
        for index, offset in zip(target["indices"], target["packing"]["offsets"]):
            if target["kind"] == "container":
                goal = {"kind": "container", "container": target["id"], "offset_xy_m": offset}
            else:
                goal = {"kind": "table", "xy": (np.asarray(target["xy"]) + offset).tolist()}
            placements.append({"object": objects[index]["id"], "target": goal})
        groups.append(placements)
    tasks, seen = [], set()
    def append(name, placements):
        key = digest(placements)
        if key not in seen:
            tasks.append({"id": name, "placements": deepcopy(placements)}); seen.add(key)
    for i, group_placements in enumerate(groups[:2]):
        append("select_one_" + str(i), group_placements[:1])
    for i, group_placements in enumerate(groups):
        append("collect_group_" + str(i), group_placements)
    append("distribute_all", [p for placements in groups for p in placements])
    if not 1 <= len(tasks) <= 8:
        raise ValueError("retained_task_count_bound_exceeded")
    request = {"id": identity, "objects": objects, "containers": containers,
        "variant_choices": {d["source_id"]: e["variant_id"] for e, d in group},
        "container_variant_choices": bindings,
        "distractors": {"count": int(rng.integers(3, 6)), "max_dimension_m": .18,
            "catalog": static[0], "source_role_index": static[1],
            "exclude_source_ids": sorted(d["source_id"] for _, d in group)}, "tasks": tasks}
    allocation = {"target_plan": target_plan, "pose_proposals": pose_receipts,
        "initial_pose_workspace": "left half of original table; radius-aware; original fit/separation checks",
        "all_selected_objects_preserved": True, "requested_and_actual_objects": len(objects),
        "task_ids": [t["id"] for t in tasks], "new_IK": 0, "physical_success": "unmeasured"}
    return request, allocation


def _check_compiled(group, cohort):
    cases = read(cohort)["cases"]
    if not cases:
        raise ValueError("compiled_scene_has_no_task_cases")
    scene = read(resolve(cases[0]["scene"]))
    expected = {d["source_id"]: (e, d) for e, d in group}
    if set(v["source_id"] for v in scene["metric_variants"].values()) != set(expected):
        raise ValueError("compiled_source_set_changed")
    for oid, value in scene["metric_variants"].items():
        entry, descriptor = expected[value["source_id"]]
        obj = next(o for o in scene["objects"] if o["id"] == oid)
        if (value["asset"] != entry["asset"] or value["cache"] != entry["cache"]
                or value["variant_id"] != entry["variant_id"] or obj["half_size"] != descriptor["half_size"]
                or scene["partition"] != descriptor["partition"]):
            raise ValueError("compiled_source_size_cache_or_split_changed")
    return cases[0]["scene"]


def build(bundle, output, *, schedule, selection_config=None, task_capacity_contract=None):
    """At most 12 fixed selections and 12 original factory calls, without retry."""
    schedule = coordinator.schedule_rows(schedule)
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("preserve_existing_source_first_factory_no_resample")
    config = deepcopy(selection_config or {"mode": "source_uniform"})
    contract = selection.task_contract(task_capacity_contract)
    b, entries, population, bins = coordinator.source.admitted_pool(bundle)
    before = digest([population, bins])
    protocol = {"schema": VERSION, "bundle": binding(bundle), "schedule": schedule,
        "selection_config": config, "task_capacity_contract": contract,
        "implementation": binding(__file__), "selection": binding(selection.__file__),
        "retained_coordinator": binding(coordinator.__file__),
        "retained_factory": binding(coordinator.factory.__file__),
        "retained_compiler": binding(coordinator.factory.retained.__file__),
        "maximum_selections": len(schedule), "maximum_compilations": len(schedule),
        "same_attempt_object_reselection": False, "scene_count_reduction": False,
        "new_IK": 0, "new_physics": 0, "new_GPU": 0, "API_calls": 0,
        "default_promoted": False, "execution_allocation_registered": False}
    write(output / "PROTOCOL.json", protocol)
    static = coordinator.factory.static_source_catalog(resolve(b["static_catalog"]), output / "static_inputs")
    rows = []
    for i, setting in enumerate(schedule):
        folder = output / "cases" / f"{i:02d}"
        row = {"index": i, **setting, "source_selected": False, "targets_planned": False,
               "layout_proposed": False, "factory_called": False, "compiled": False,
               "compiled_tasks": 0, "physical_success": "unmeasured"}
        began = time.monotonic()
        try:
            pool = {s: xs for s, xs in population.items() if xs[0][1]["partition"] == setting["partition"]}
            stage_start = time.monotonic()
            group, receipt = selection.choose_objects(pool, setting["N"], setting["partition"], setting["seed"], **config)
            row["selection_seconds"] = time.monotonic() - stage_start
            write(folder / "selection.json", receipt)
            row.update(source_selected=True, selection=binding(folder / "selection.json"))
            stage_start = time.monotonic()
            plan = selection.plan_targets(group, bins, setting["partition"], setting["seed"], contract=contract)
            row["target_planning_seconds"] = time.monotonic() - stage_start
            write(folder / "target_plan.json", plan)
            row.update(targets_planned=True, target_plan=binding(folder / "target_plan.json"))
            identity = "source_first_" + digest([protocol, i, receipt])[:32]
            stage_start = time.monotonic()
            request, allocation = propose(identity, group, plan, setting["seed"], static)
            row["layout_seconds"] = time.monotonic() - stage_start
            write(folder / "allocation.json", allocation)
            write(folder / "requests.json", {"schema": coordinator.factory.VERSION, "scenes": [request]})
            row.update(layout_proposed=True, factory_called=True)
            stage_start = time.monotonic()
            summary = coordinator.factory.compile_requests(bundle, folder / "requests.json", folder / "factory", setting["seed"])
            row["compilation_seconds"] = time.monotonic() - stage_start
            row["factory_summary"] = summary
            if summary["compiled_scenes"] == 1:
                row.update(scene=_check_compiled(group, folder / "factory/cohort.json"),
                    compiled=True, compiled_tasks=summary["compiled_tasks"],
                    source_size_cache_partition_preserved=True, cohort=binding(folder / "factory/cohort.json"))
            else:
                row["factory_rejections"] = read(folder / "factory/scenes.json")["scenes"]
        except (Rejected, ValueError, RuntimeError, KeyError) as error:
            row.update(compiled=False, reason=getattr(error, "code", type(error).__name__),
                       details=getattr(error, "details", {"message": str(error)}))
        row["seconds"] = time.monotonic() - began
        write(folder / "RESULT.json", row); rows.append(row)
        print({"case": i, "compiled": row["compiled"], "reason": row.get("reason")}, flush=True)
    if digest([population, bins]) != before:
        raise ValueError("source_first_adapter_mutated_original_population")
    result = {"schema": VERSION, "protocol": binding(output / "PROTOCOL.json"), "rows": rows,
        "requested_scenes": len(rows), "source_selections": sum(r["source_selected"] for r in rows),
        "target_plans": sum(r["targets_planned"] for r in rows),
        "layout_proposals": sum(r["layout_proposed"] for r in rows),
        "actual_compile_calls": sum(r["factory_called"] for r in rows),
        "compiled_scenes": sum(r["compiled"] for r in rows),
        "compiled_tasks": sum(r["compiled_tasks"] for r in rows if r["compiled"]),
        "all_attempts_retained": True, "source_size_cache_partition_preserved": True,
        "new_IK": 0, "new_physics": 0, "new_GPU": 0, "API_calls": 0,
        "default_promoted": False, "execution_allocation_registered": False,
        "physical_success": "unmeasured"}
    write(output / "RESULTS.json", result)
    return result
