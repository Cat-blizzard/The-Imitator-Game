"""Task-aware asset roles and bounded reproducible random scenes."""
from copy import deepcopy
import numpy as np

from .common import read, resolve
from .assets import select
from corl2026_workshop.experiments.autonomous_pipeline_v4.library import make_scene


def item(binding, d, identity, xy, yaw, movable=False):
    return {"id": identity, "kind": "movable" if movable else "static_distractor", "shape": "catalog_mesh",
        "asset_descriptor": binding, "half_size": d["half_size"], "mass_kg": d["mass_kg"],
        "position": [float(xy[0]), float(xy[1]), d["half_size"][2] + .0005], "yaw": float(yaw),
        "rgba": [.5, .5, .5, 1.], "color_name": "original textured material",
        "semantic_description": f"the {d['name'].replace('_', ' ')} ({d['category']})"}


def sample(catalog, asset, cache, seed, layout):
    d = read(resolve(asset)); rng = np.random.default_rng(seed)
    identity = d["id"] + f"_L{layout}"
    scene = make_scene(identity, seed, nuisance=True, layout=False)
    scene.update(schema="resimpair_generic_mesh_scene_v1", scene_instance_id=identity,
        split_group=d["source_id"], partition=d["partition"], source_coverage="synthetic_only")
    arm = layout % 2
    sign = -1 if arm == 0 else 1
    xy = [rng.uniform(-.27, -.13), sign * rng.uniform(.22, .38)]
    goal = [rng.uniform(.015, .10), sign * rng.uniform(.22, .38)]
    yaw = float(rng.uniform(-np.pi, np.pi))
    scene["objects"] = [item(asset, d, "target_object", xy, yaw, True)]
    scene["grasp_caches"] = {"target_object": cache}
    scene["supports"] = [{"id": "goal_tray", "shape": "tray", "color_name": "orange", "rgba": [.7, .32, .12, 1.],
        "position": [*goal, 0.], "half_size": [.105, .105, .006], "top_z": .012, "wall_height": .025,
        "semantic_description": "the orange open shallow destination tray"}]
    # Exclude all manipulation and destination footprints. This is a sampling
    # prior, not a collision certificate: the actual arm and payload path is
    # checked against every scene collider before any command is executed.
    reserve_low = np.minimum(xy, goal) - np.array([.13, .15])
    reserve_high = np.maximum(xy, goal) + np.array([.13, .15])
    pools = [(b, v) for b, v in select(catalog, exclude={d["id"]}, max_dimension=.18)
             if v["partition"] in {d["partition"], "shared_distractor"}]
    rng.shuffle(pools)
    distinct, repeated, categories = [], [], set()
    for pair in pools:
        if pair[1]["category"] in categories:
            repeated.append(pair)
        else:
            categories.add(pair[1]["category"]); distinct.append(pair)
    pools = distinct + repeated
    count = min(int(rng.integers(3, 9)), len(pools))
    placed = []
    for b, v in pools:
        if len(placed) >= count: break
        radius = float(np.linalg.norm(v["half_size"][:2]))
        for _ in range(100):
            if rng.random() < .65:
                p = np.array([rng.uniform(.28, .38), rng.uniform(-.70, .70)])
            else:
                p = np.array([rng.uniform(-.35, .30), rng.choice([-1, 1]) * rng.uniform(.65, .78)])
            if np.all(p + radius >= reserve_low) and np.all(p - radius <= reserve_high): continue
            if any(np.linalg.norm(p - q) < radius + r + .025 for q, r in placed): continue
            if abs(p[1]) + radius > .90 or p[0] + radius > .46: continue
            scene["objects"].append(item(b, v, f"distractor_{len(placed)}", p, rng.uniform(-np.pi, np.pi)))
            placed.append((p, radius)); break
    if len(placed) < 3:
        raise ValueError("Could not place at least 3 distinct mesh distractors")
    scene["randomization"] = {"seed": seed, "xy_source_range_m": [[-.27, -.13], [.22, .38]],
        "yaw_range_rad": [-float(np.pi), float(np.pi)], "distractor_count": len(placed),
        "distractor_source_ids": [read(resolve(o["asset_descriptor"]))["source_id"] for o in scene["objects"][1:]],
        "distractor_scale_policy": "fixed metric descriptor; oversized objects excluded, never shrunk to force acceptance",
        "reserved_sweep_xy": [reserve_low.tolist(), reserve_high.tolist()],
        "collision_verification": "runtime dense full-arm/held-object mesh guard required"}
    # Multiple distinct goals share exactly the same initial physics scene.
    tasks = []
    for name, support, offset in [("tray_place", "goal_tray", [0., 0.]),
                                   ("table_relocate", "table", [-.20, sign * .08])]:
        # This second goal is in the central table region of the same arm's
        # workspace, sharing the initial scene and all object identities.
        tasks.append({"schema": "resimpair_task_intent_v4", "id": identity + "_" + name,
            "family": "relocate" if support != "table" else "arrange", "required_skills": ["pick_place"],
            "object_ids": ["target_object"], "goal": name.replace("_", " "),
            "operations": [{"id": "transfer_0", "primitive": "pick_place", "object": "target_object", "arm": arm,
                            "target": {"support": support, "offset_xy": offset, "yaw": yaw}}],
            "required_operation_precedence": [], "equivalence_groups": [],
            "incidental_controller_order_is_required": False, "human_hand_policy": "preserve_robot_body_side",
            "non_target_policy": "preserve_identity_and_pose", "release_precondition_control": True,
            "scope": "rigid_pick_place_only; no insertion, articulation or deformable claim"})
    return scene, tasks
