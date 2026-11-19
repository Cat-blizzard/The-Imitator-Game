"""Parameterized task and scene descriptions, independent of the source layout."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .core import digest, require

CAMERAS = {
    "robot_side": {"eye": [-.70, 0., .72], "target": [-.06, 0., .08],
                   "width": 512, "height": 512, "fov": 1.15},
    "opposite_high": {"eye": [.85, 0., 1.02], "target": [-.13, 0., .12],
                      "width": 512, "height": 512, "fov": 1.04},
}

TASK_TEMPLATES = {
    "relocate": {"required_skills": ["pick_place"], "goal": "specified object on specified support"},
    "pack": {"required_skills": ["pick_place"], "goal": "all selected objects inside one open tray"},
    "sort": {"required_skills": ["pick_place"], "goal": "objects in role-assigned trays"},
    "stack": {"required_skills": ["pick_place", "stack_on_support"], "goal": "a supported tower with specified bottom-to-top identities"},
    "arrange": {"required_skills": ["pick_place"], "goal": "objects at distinct specified table locations"},
}


def make_scene(scene_id, seed, *, nuisance=False, layout=False, appearance_config=None):
    rng = np.random.default_rng(seed)
    color_rng = np.random.default_rng(seed + 1000)
    scene = {"schema": "resimpair_tabletop_scene_v4", "id": scene_id, "seed": seed,
             "table_top_z": 0., "robot_roots": [[-.60, -.35, 0.], [-.60, .35, 0.]],
             "cameras": deepcopy(CAMERAS), "objects": [], "supports": [],
             "randomization": {"nuisance": nuisance, "task_layout": layout},
             "appearance": {"table_rgb": [.62, .55, .45], "background_rgb": [.77, .79, .80],
                            "ambient_rgb": [.35, .35, .35], "key_light_rgb": [1.5, 1.5, 1.5]}}
    positions = [[-.24, -.34, .0255], [-.24, .34, .0255], [-.32, -.02, .0255]]
    descriptors = [("red_block", "red", [.80, .10, .08, 1.]),
                   ("blue_block", "blue", [.08, .25, .82, 1.]),
                   ("yellow_block", "yellow", [.94, .70, .06, 1.])]
    for i, (name, color, rgba) in enumerate(descriptors):
        p = np.array(positions[i], dtype=float)
        if layout:
            p[:2] += rng.uniform(-.018, .018, 2)
        scene["objects"].append({"id": name, "kind": "movable", "shape": "box", "half_size": [.025, .025, .025],
                                 "color_name": color, "rgba": rgba, "position": p.tolist(),
                                 "yaw": float(rng.uniform(-.08, .08)) if layout else 0.,
                                 "mass_kg": .065, "semantic_description": f"the {color} cube"})
    supports = [("left_tray", "orange", [.69, .30, .12, 1.], [.08, -.25, 0.], "tray"),
                ("right_tray", "purple", [.46, .23, .64, 1.], [.08, .25, 0.], "tray"),
                ("platform", "gray", [.40, .42, .44, 1.], [.06, 0., 0.], "platform")]
    for name, color, rgba, position, shape in supports:
        p = np.array(position)
        if layout:
            p[:2] += rng.uniform(-.012, .012, 2)
        scene["supports"].append({"id": name, "shape": shape, "color_name": color,
                                  "rgba": rgba, "position": p.tolist(),
                                  "half_size": [.10, .075, .006] if shape == "tray" else [.065, .065, .006],
                                  "top_z": .012, "semantic_description": f"the {color} open shallow tray" if shape == "tray" else "the gray square platform"})
    clutter_count = 3 if nuisance else 2
    for i in range(clutter_count):
        position = [.29, -.42 + i * .40, .035]
        if nuisance:
            position[0] += float(rng.uniform(-.025, .035))
            position[1] += float(rng.uniform(-.04, .04))
        scene["objects"].append({"id": f"distractor_{i}", "kind": "static_distractor", "shape": "box",
                                 "half_size": [.028, .028, .035], "rgba": [.15, .48 + .08 * i, .27, 1.],
                                 "position": position, "yaw": float(rng.uniform(-.3, .3)), "mass_kg": 0.,
                                 "color_name": "green", "semantic_description": f"green background block {i + 1}"})
    if nuisance:
        scene["appearance"] = {"table_rgb": color_rng.uniform(.30, .70, 3).tolist(),
                               "background_rgb": color_rng.uniform(.55, .85, 3).tolist(),
                               "ambient_rgb": color_rng.uniform(.25, .50, 3).tolist(),
                               "key_light_rgb": color_rng.uniform(1.0, 1.8, 3).tolist()}
    if appearance_config is not None:
        from resimpair_tasks.appearance_v1 import sample
        scene = sample(scene, seed + 2000, config=appearance_config)
    return scene


def instantiate(family, task_id, scene, object_ids, support_ids=None):
    require(family in TASK_TEMPLATES, "Unsupported task family")
    objects = {o["id"]: o for o in scene["objects"]}
    supports = {o["id"]: o for o in scene["supports"]}
    require(object_ids and len(object_ids) == len(set(object_ids)), "Nonempty distinct target identities required")
    require(all(o in objects and objects[o]["kind"] == "movable" for o in object_ids), "Unknown movable target")
    require(family != "relocate" or len(object_ids) == 1, "relocate_requires_one_object")
    support_ids = list(support_ids or [])
    if family in {"relocate", "pack", "stack"}:
        require(len(support_ids) == 1 and support_ids[0] in supports, "One valid support required")
    if family == "sort":
        require(len(support_ids) == len(object_ids) and set(support_ids) <= set(supports), "Each object needs a valid tray")
    if family in {"pack", "sort"}:
        require(all(supports[s]["shape"] == "tray" for s in support_ids), "Packing and sorting require open trays")
    operations = []
    edges = []
    for index, oid in enumerate(object_ids):
        obj = objects[oid]
        target = {"support": "table", "offset_xy": [0., 0.], "yaw": 0.}
        if family in {"relocate", "pack", "sort"}:
            target["support"] = support_ids[index] if family == "sort" else support_ids[0]
            members = ([j for j, sid in enumerate(support_ids) if sid == target["support"]] if family == "sort"
                       else list(range(len(object_ids))) if family == "pack" else [index])
            count, slot = len(members), members.index(index)
            # Named supports define capacity; no silent overlap when object count grows.
            spacing = 2 * max(objects[o]["half_size"][0] for o in object_ids) + .018
            require(count * spacing <= 2 * supports[target["support"]]["half_size"][0] - .015,
                    "packing_capacity_exceeded")
            target["offset_xy"] = [(slot - (count - 1) / 2) * spacing, 0.]
        elif family == "stack":
            target["support"] = support_ids[0] if index == 0 else object_ids[index - 1]
            if index:
                edges.append([f"transfer_{index - 1}", f"transfer_{index}"])
        elif family == "arrange":
            target["offset_xy"] = [-.13, (index - (len(object_ids) - 1) / 2) * .17]
        operations.append({"id": f"transfer_{index}", "primitive": "pick_place", "object": oid,
                           "target": target, "arm": 0 if obj["position"][1] <= 0 else 1})
    return {"schema": "resimpair_task_intent_v4", "id": task_id, "family": family,
            "required_skills": TASK_TEMPLATES[family]["required_skills"], "object_ids": object_ids,
            "goal": TASK_TEMPLATES[family]["goal"], "operations": operations,
            "required_operation_precedence": edges, "equivalence_groups": [],
            "incidental_controller_order_is_required": False,
            "human_hand_policy": "preserve_robot_body_side",
            "non_target_policy": "preserve_identity_and_pose", "scope": "declared_composition_of_source_skills"}


def task_semantic_key(task):
    return digest({k: task[k] for k in ["family", "object_ids", "operations", "required_operation_precedence",
                                      "equivalence_groups", "human_hand_policy", "non_target_policy"]})
