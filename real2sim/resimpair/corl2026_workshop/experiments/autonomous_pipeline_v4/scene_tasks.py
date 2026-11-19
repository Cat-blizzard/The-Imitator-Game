"""Deterministic scene -> candidate tasks, with conservative geometric screening.

This is a selector for existing pick/place adapters, NOT a motion feasibility
oracle or a general language-to-task generator. Rejections remain in the pool.
Selection happens before any execution and never uses successful outcomes.
"""
from itertools import combinations, permutations, product
import math

import numpy as np

from .core import digest, require
from .library import instantiate
from .simulation import targets

VERSION = "scene_task_selector_v1"


def footprint(obj, yaw=None):
    yaw = obj.get("yaw", 0.) if yaw is None else yaw
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    x, y = obj["half_size"][:2]
    return np.array([c * x + s * y, s * x + c * y])


def overlaps(a, ah, b, bh, margin=.012):
    return bool(np.all(np.abs(np.asarray(a)[:2] - np.asarray(b)[:2]) < ah + bh + margin))


def screen(scene, task):
    """Cheap preflight; actual grasp, contact, collision and goal gates stay separate."""
    objects = {o["id"]: o for o in scene["objects"]}
    supports = {s["id"]: s for s in scene["supports"]}
    reasons = []
    if any(op["primitive"] != "pick_place" for op in task["operations"]):
        reasons.append("unsupported_contact_adapter")
    if task["family"] == "stack" and any(objects[o]["shape"] != "box" for o in task["object_ids"]):
        reasons.append("mesh_stacking_not_qualified")
    for a, b in combinations(scene["objects"], 2):
        if overlaps(a["position"], footprint(a), b["position"], footprint(b)):
            reasons.append("initial_object_clearance:" + a["id"] + ":" + b["id"])
    locations = targets(scene, task)
    occupied = {o: np.array(v["position"]) for o, v in objects.items()}
    extents = {o: footprint(v) for o, v in objects.items()}
    for op in task["operations"]:
        oid, sid = op["object"], op["target"]["support"]
        obj, target = objects[oid], np.array(locations[oid])
        shape = footprint(obj, op["target"]["yaw"])
        if obj["shape"] == "convex_mesh" and "asset_descriptor" not in obj:
            reasons.append("missing_active_asset_descriptor:" + oid)
        if min(obj["half_size"][:2]) * 2 > .077 or obj["half_size"][2] * 2 < .04:
            reasons.append("unsupported_vertical_grasp_geometry:" + oid)
        root = np.array(scene["robot_roots"][op["arm"]])
        for stage, point in [("initial", occupied[oid]), ("target", target)]:
            if np.linalg.norm(point[:2] - root[:2]) > .76:
                reasons.append("conservative_workspace:" + oid + ":" + stage)
        if target[0] - shape[0] < -.50 or target[0] + shape[0] > .42 or abs(target[1]) + shape[1] > .60:
            reasons.append("table_extent:" + oid)
        if sid in supports:
            support = supports[sid]
            delta = np.abs(target[:2] - np.array(support["position"][:2]))
            if np.any(delta + shape + .008 > np.array(support["half_size"][:2])):
                reasons.append("support_capacity_xy:" + oid)
        elif sid in objects:
            if np.any(shape > footprint(objects[sid], 0.) + 1e-8):
                reasons.append("unstable_stack_footprint:" + oid)
        elif sid != "table":
            reasons.append("unsupported_support:" + sid)
        for other, point in occupied.items():
            if other not in {oid, sid} and overlaps(target, shape, point, extents[other]):
                reasons.append("target_object_clearance:" + oid + ":" + other)
        if sid == "table":
            for support in supports.values():
                if overlaps(target, shape, support["position"], footprint(support)):
                    reasons.append("table_target_overlaps_support:" + oid + ":" + support["id"])
        occupied[oid], extents[oid] = target, shape
    return {"eligible": not reasons, "reasons": sorted(set(reasons)),
            "scope": "static bounds/capacity only; no IK, swept-path or contact success claim"}


def enumerate_tasks(scene, required_object=None):
    movable = sorted(o["id"] for o in scene["objects"] if o["kind"] == "movable")
    supports = sorted(s["id"] for s in scene["supports"])
    trays = sorted(s["id"] for s in scene["supports"] if s["shape"] == "tray")
    platforms = sorted(s["id"] for s in scene["supports"] if s["shape"] == "platform")
    require(required_object is None or required_object in movable, "required_object_not_movable")
    proposals = []
    proposals += [("relocate", [o], [s]) for o, s in product(movable, supports)]
    for pair in combinations(movable, 2):
        proposals += [("pack", list(pair), [s]) for s in trays]
        proposals += [("sort", list(pair), list(ss)) for ss in permutations(trays, 2)]
        # Deterministic spatial order keeps the same-side assignment when possible.
        positions = {o["id"]: o["position"][1] for o in scene["objects"]}
        proposals.append(("arrange", sorted(pair, key=lambda o: (positions[o], o)), []))
        proposals += [("stack", list(order), [s]) for order in permutations(pair) for s in platforms]
    rows = []
    for family, objects, selected_supports in proposals:
        intent = {"family": family, "objects": objects, "supports": selected_supports}
        key = digest(intent)[:16]
        task_id = scene["id"] + "__" + family + "_" + key
        try:
            task = instantiate(family, task_id, scene, objects, selected_supports)
            result = screen(scene, task)
            if required_object is not None and required_object not in objects:
                result["eligible"] = False
                result["reasons"].append("pilot_requires_active_object:" + required_object)
            goal_key = digest({"targets": targets(scene, task), "precedence": task["required_operation_precedence"]})
        except ValueError as exc:
            task, goal_key = None, None
            result = {"eligible": False, "reasons": [str(exc)], "scope": "task construction rejected"}
        rows.append({"candidate_id": key, "intent": intent, "task": task, "goal_key": goal_key, "preflight": result})
    return sorted(rows, key=lambda r: r["candidate_id"])


def select(pool, seed, count=3):
    """Hash-ranked random draw, one per family and distinct goal; order invariant."""
    ranked = sorted((r for r in pool if r["preflight"]["eligible"]),
                    key=lambda r: digest({"seed": seed, "candidate": r["candidate_id"]}))
    chosen, families, goals = [], set(), set()
    for row in ranked:
        family = row["intent"]["family"]
        if family not in families and row["goal_key"] not in goals:
            chosen.append(row)
            families.add(family)
            goals.add(row["goal_key"])
        if len(chosen) == count:
            break
    require(len(chosen) == count, "insufficient_distinct_eligible_scene_tasks")
    return {"schema": VERSION, "seed": seed, "count": count, "selection": chosen,
            "replacement_after_execution_failure": False, "physical_feasibility": "unverified_until_execution"}
