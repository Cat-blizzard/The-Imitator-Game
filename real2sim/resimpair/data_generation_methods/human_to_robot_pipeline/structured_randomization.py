"""Task-preserving CPU scene proposals with conservative geometric screening.

These are *proposals*, not simulator rollouts. Proxy reach and line-of-sight tests
do not replace IK, swept collision checks, rendered segmentation or native QA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .planning_contract import bind, save_new, validate_contract, native_adapter_check


def normalized_assets(catalog):
    """Use existing factory adapter geometry; do not invent metric sizes."""
    result = []
    for source in catalog:
        if "adapter" in source:
            adapter = source["adapter"]
            result.append({
                "uid": source["uid"], "name": source["name"],
                "bounds_rotated_m": adapter["bounds_m"],
                "visual": adapter["visual"], "collision": adapter["collision"],
                "files": {adapter["visual"]: adapter["visual_sha256"],
                          adapter["collision"]: adapter["collision_sha256"]},
            })
        else:
            result.append(source)
    return result


def segment_distance(point, start, end):
    point, start, end = map(lambda x: np.asarray(x, dtype=float), (point, start, end))
    delta = end - start
    t = np.clip(np.dot(point - start, delta) / max(float(delta @ delta), 1e-12), 0, 1)
    return float(np.linalg.norm(point - (start + t * delta)))


def in_camera(point, camera, margin=.08):
    eye, target, point = map(np.asarray, (camera["eye"], camera["target"], point))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = point - eye
    depth = float(delta @ forward)
    half = depth * np.tan(camera["fov_radians"] / 2) * (1 - margin)
    return depth > .05 and abs(delta @ right) < half and abs(delta @ up) < half


def screen_layout(case):
    """No success predicate here: return rejection reasons and unknown evidence."""
    reasons = []
    source = np.array(case["roles"]["source_container"]["center"])
    target = np.array(case["roles"]["destination_container"]["center"])
    if np.linalg.norm((source-target)[:2]) < .22:
        reasons.append("containers_too_close")
    if case["roles"]["destination_container"].get("initially_empty") is not True:
        reasons.append("destination_not_empty")
    if case["roles"]["selected_object"].get("initial_relation") != "inside_source_container":
        reasons.append("initial_relation_invalid")
    obstacles = case.get("added_distractors", [])
    for item in obstacles:
        if item["category"] in ("apple", "bowl", "container"):
            reasons.append("distractor_creates_task_ambiguity")
        p = np.asarray(item["center"])
        if any(np.linalg.norm((p-c)[:2]) < .095 + item["radius_m"] + .03 for c in (source, target)):
            reasons.append("distractor_overlaps_task_geometry")
        if segment_distance(p[:2], source[:2], target[:2]) < item["radius_m"] + .10:
            reasons.append("distractor_blocks_reserved_transfer_corridor")
        if not (-.30 < p[0]-item["radius_m"] and p[0]+item["radius_m"] < .25
                and -.46 < p[1]-item["radius_m"] and p[1]+item["radius_m"] < .46):
            reasons.append("distractor_outside_conservative_table_region")
    for i, left in enumerate(obstacles):
        for right in obstacles[i+1:]:
            if np.linalg.norm(np.array(left["center"])-right["center"]) < left["radius_m"]+right["radius_m"]+.03:
                reasons.append("distractors_overlap")
    events = case["critical_event_probe_points"]
    base = np.array([-.6, -.35 if case["active_arm"] == 0 else .35, 0.])
    for point in events.values():
        if not .20 < np.linalg.norm(np.asarray(point)-base) < .83:
            reasons.append("outside_conservative_reach_envelope")
        if not in_camera(point, case["camera"]):
            reasons.append("critical_event_outside_camera")
        for obstacle in obstacles:
            if segment_distance(obstacle["center"], case["camera"]["eye"], point) < obstacle["radius_m"]+.015:
                reasons.append("critical_event_line_of_sight_blocked")
    return {
        "coarse_screen_passed": not reasons, "reasons": sorted(set(reasons)),
        "proxies": ["bounding-sphere initial clearance", "reserved transport corridor",
                    "conservative spherical reach envelope", "pinhole event-point frustum",
                    "distractor sphere versus camera-event ray"],
        "still_required": ["exact triangle collision", "IK and complete robot swept collision",
                           "native grasp and support physics", "rendered event visibility",
                           "human-robot task equivalence review"],
        "physical_success_verified": False,
    }


def propose(seed, level, assets, split="train", active_arm=None):
    rng = np.random.default_rng(seed)
    active = int(rng.integers(2)) if active_arm is None else int(active_arm)
    mid = np.r_[rng.uniform(-.065, .035), rng.uniform(-.05, .05), 0.]
    angle = float(rng.uniform(-.35, .35))
    separation = float(rng.uniform(.25, .35))
    direction = np.array([np.sin(angle), np.cos(angle), 0.]) * (1 if active else -1)
    source, target = mid+direction*separation/2, mid-direction*separation/2
    n = {"layout": 0, "clutter": 2, "clutter_camera": 4}[level]
    obstacles = []
    for i in range(n):
        asset = assets[int(rng.integers(len(assets)))]
        bounds = np.asarray(asset["bounds_rotated_m"], dtype=float)
        extents = bounds[1]-bounds[0]
        radius = float(np.linalg.norm(extents)/2)
        p = [float(rng.uniform(-.23, .18)), float(rng.uniform(-.39, .39)), float(extents[2]/2)]
        obstacles.append({"instance_id": f"distractor_{i}", "asset_uid": asset["uid"],
                          "category": asset["name"], "center": p, "radius_m": radius,
                          "yaw_rad": float(rng.uniform(-np.pi, np.pi)),
                          "visual": asset["visual"], "collision": asset["collision"],
                          "asset_files": asset["files"], "must_remain_untouched": True})
    camera = {"eye": [-.50, 0., .73], "target": [0., 0., .10], "fov_radians": 1.30,
              "fixed_during_episode": True}
    if level == "clutter_camera":
        camera["eye"] = (np.array(camera["eye"])+rng.uniform([-.06,-.08,-.04],[.04,.08,.07])).tolist()
        camera["fov_radians"] = float(rng.uniform(1.24, 1.38))
    return {
        "case_id": f"apple_structured_{level}_{seed}", "seed": seed, "split": split,
        "level": level, "layout_mode": "explicit_proposed_geometry", "active_arm": active,
        "roles": {
            "selected_object": {"category": "apple", "initial_relation": "inside_source_container"},
            "source_container": {"center": source.tolist(), "initially_occupied": True},
            "destination_container": {"center": target.tolist(), "initially_empty": True},
        },
        "added_distractors": obstacles, "camera": camera,
        "camera_mode": "explicit_proposed_camera",
        "appearance": {"wall_and_hdri_split": split, "material_seed": seed+70000001,
                       "sample_independently_of_goal": True, "asset_ids_pending_adapter_resolution": True},
        "critical_event_probe_points": {
            "initial_grasp": (source+[0,0,.08]).tolist(),
            "lift_clear": (source+[0,0,.22]).tolist(),
            "arrival": (target+[0,0,.22]).tolist(),
            "released_support": (target+[0,0,.08]).tolist(),
        },
        "planner_requirements": {
            "obstacle_instance_ids": ["source_container", "destination_container"]+
                                     [x["instance_id"] for x in obstacles],
            "include_attached_object_collision": True,
            "collision_mask_changes_after_failure_forbidden": True,
            "phase_labels_do_not_verify_contact": True,
        },
        "simulation_executed": False, "training_admitted": False,
    }


def generate(contract, assets, count=12, seed_base=2026102000, attempts_per_level=3000):
    validate_contract(contract)
    source_assets = normalized_assets(assets)
    assets = [a for a in source_assets if a["name"] not in ("apple", "bowl", "container")]
    if not assets or count <= 0 or attempts_per_level < count:
        raise ValueError("Missing admissible assets or invalid finite budget")
    selected, rejected = [], []
    windows = [h for h in contract["human_windows"] if h["split"] == "train"]
    if not windows:
        raise ValueError("No reviewed same-split human windows")
    for level_index, level in enumerate(("layout", "clutter", "clutter_camera")):
        accepted = 0
        for i in range(attempts_per_level):
            seed = seed_base + level_index*100000 + i
            chosen = np.random.default_rng(seed+730001).choice(len(source_assets), 2, replace=False)
            selected_arm = [arm for arm, index in enumerate(chosen) if source_assets[int(index)]["name"] == "apple"]
            if len(selected_arm) != 1:
                rejected.append({"seed": seed, "level": level, "reasons": ["native_seed_has_no_unique_apple"]})
                continue
            case = propose(seed, level, assets, active_arm=selected_arm[0])
            screen = screen_layout(case)
            if not screen["coarse_screen_passed"]:
                rejected.append({"case_id": case["case_id"], "seed": case["seed"],
                                 "level": level, "reasons": screen["reasons"]})
                continue
            case["screen"] = screen
            case["human_window_id"] = windows[accepted % len(windows)]["id"]
            case["human_video"] = windows[accepted % len(windows)]["video"]
            case["adapter_check"] = native_adapter_check(case, contract)
            selected.append(case)
            accepted += 1
            if accepted == count:
                break
        if accepted < count:
            raise RuntimeError(f"Finite {level} budget exhausted: {accepted}/{count}; do not relax constraints")
    return {"schema": "task_preserving_layout_proposals_v1", "cases": selected,
            "rejected_coarse_proposals": rejected, "requested_per_level": count,
            "simulation_executed": False, "training_admitted": False,
            "next_native_batch_gate": {
                "state": "blocked_pending_strict_path_adapter_qualification",
                "reason": "custom Screw planner does not reject its collision result; geometric validator is offline only",
                "required": ["log planner mode, world goal and full q-path",
                             "independent FK endpoint and full-path checks",
                             "per-configuration collision recheck with explicit contact permissions",
                             "small native qualification before scaling"],
            },
            "limitations": "Candidate geometry only; exact planner/physics/render adapter must be implemented and qualified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-level", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=2026102000)
    args = parser.parse_args()
    assets = normalized_assets(json.loads(args.assets.read_text())["assets"])
    for asset in assets:
        for path, digest in asset["files"].items():
            if bind(path)["sha256"] != digest:
                raise ValueError("Changed asset: "+path)
    report = generate(json.loads(args.contract.read_text()), assets, args.per_level, args.seed_base)
    report["contract"] = bind(args.contract)
    report["asset_catalog"] = bind(args.assets)
    save_new(args.output, report)
    print(json.dumps({"output": str(args.output), "proposals": len(report["cases"]),
                      "coarse_rejections": len(report["rejected_coarse_proposals"]),
                      "simulation_executed": False}))


if __name__ == "__main__":
    main()
