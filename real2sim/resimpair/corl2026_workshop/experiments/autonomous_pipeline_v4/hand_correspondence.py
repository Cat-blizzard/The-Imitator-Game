"""Bind robot effectors to one operator's anatomical hands before generation.

This version supports the exported parallel Panda pair: local +X is forward
and local +Z is up. It is NOT a general humanoid retargeter. Ambiguous rigs or
unbound native channels fail closed instead of guessing from an arm name,
object position, wrist position, or a camera's image-left/image-right.
"""
from __future__ import annotations

import numpy as np

from .core import digest, require

VERSION = "resimpair_hand_correspondence_v1"
POLICY = "preserve_robot_body_side"
NATIVE_ARM_ARRAYS = {"grasp", "door_grasp", "robot_contact_impulses"}
HAND_ACTIONS = {"grasp", "lift", "arrive", "release", "withdraw", "articulate", "trigger", "insert"}


def body_frame(robot_roots):
    roots = np.asarray(robot_roots, dtype=float)
    require(roots.shape == (2, 4, 4) and np.isfinite(roots).all(), "hand_mapping_requires_two_finite_robot_roots")
    require(np.allclose(roots[:, 3], [0, 0, 0, 1], atol=1e-6), "invalid_hand_mapping_root_transform")
    rotations = roots[:, :3, :3]
    require(np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5)
            and np.allclose(np.linalg.det(rotations), 1, atol=1e-5), "hand_mapping_roots_must_be_proper_rotations")
    # A different robot convention or an opposed/tandem pair needs a separately
    # declared rig adapter, never an anatomical label inferred from array order.
    require(float(np.trace(rotations[0].T @ rotations[1])) > 1 + 2 * np.cos(np.deg2rad(5)),
            "hand_mapping_requires_cooriented_parallel_bases")
    forward = rotations[:, :, 0].mean(axis=0)
    forward /= np.linalg.norm(forward)
    up = rotations[:, :, 2].mean(axis=0)
    up -= forward * (up @ forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    center = roots[:, :3, 3].mean(axis=0)
    lateral = (roots[:, :3, 3] - center) @ right
    baseline = roots[1, :3, 3] - roots[0, :3, 3]
    require(np.min(np.abs(lateral)) > .05 and abs(baseline @ right) > .8 * np.linalg.norm(baseline),
            "hand_mapping_ambiguous_lateral_base_placement")
    return {"rig_convention": "parallel_panda_pair_local_x_forward_z_up_v1",
            "origin_world": center.tolist(), "forward_world": forward.tolist(),
            "up_world": up.tolist(), "right_world": right.tolist(),
            "robot_hands": ["right" if x > 0 else "left" for x in lateral],
            "lateral_offsets_m": lateral.tolist()}


def effector_indices(protocol):
    """Recover the native robot axis from contact/grasp bindings, not IDs."""
    result = {}
    for name, effector in protocol["effectors"].items():
        channels = sorted(set(effector["grasp_channels"].values()) | set(effector["contact_channels"].values()))
        indices = set()
        for channel in channels:
            spec = protocol["channels"][channel]
            selectors = spec["indices"]
            require(spec["array"] in NATIVE_ARM_ARRAYS and selectors and type(selectors[0]) is int,
                    "hand_mapping_unbound_native_robot_axis:" + channel)
            indices.add(selectors[0])
        require(len(indices) == 1, "hand_mapping_inconsistent_or_missing_effector_channels:" + name)
        result[name] = {"robot_index": indices.pop(), "native_channels": channels}
    require(len(result) == 2 and {v["robot_index"] for v in result.values()} == {0, 1},
            "hand_mapping_requires_bijective_native_effectors")
    return result


def resolve(protocol, robot_roots):
    frame = body_frame(robot_roots)
    bindings = effector_indices(protocol)
    for name, item in bindings.items():
        hand = frame["robot_hands"][item["robot_index"]]
        declared = protocol["effectors"][name]["human_hand"]
        require(declared in {"either", hand},
                f"hand_mapping_conflict:{name}:declared_{declared}:measured_{hand}")
        item["human_hand"] = hand
    return frame, bindings


def declare_from_scene(protocol, scene):
    """Planning-side declaration for the current fixed-orientation scene API."""
    positions = np.asarray(scene["robot_roots"], dtype=float)
    require(positions.shape == (2, 3), "unsupported_planning_robot_root_representation")
    roots = np.tile(np.eye(4), (2, 1, 1))
    roots[:, :3, 3] = positions
    _, bindings = resolve(protocol, roots)
    for name, item in bindings.items():
        protocol["effectors"][name]["human_hand"] = item["human_hand"]


def compile_correspondence(protocol, arrays, camera):
    require("robot_roots_world" in arrays, "hand_mapping_missing_native_robot_roots")
    roots = np.asarray(arrays["robot_roots_world"], dtype=float)
    frame, bindings = resolve(protocol, roots)
    intrinsic = np.asarray(camera["parameters"]["intrinsic_cv"], dtype=float)
    extrinsic = np.asarray(camera["parameters"]["extrinsic_cv"], dtype=float)
    require(intrinsic.shape == (3, 3) and extrinsic.shape in {(3, 4), (4, 4)}
            and np.isfinite(intrinsic).all() and np.isfinite(extrinsic).all(), "hand_mapping_invalid_camera")
    camera_points = roots[:, :3, 3] @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    require(np.all(camera_points[:, 2] > 1e-6), "hand_mapping_robot_origins_behind_camera")
    pixels = camera_points @ intrinsic.T
    xy = pixels[:, :2] / pixels[:, 2:3] / [camera["config"]["width"], camera["config"]["height"]]
    require(abs(xy[0, 0] - xy[1, 0]) > .03, "hand_mapping_camera_side_order_ambiguous")
    for item in bindings.values():
        i = item["robot_index"]
        item["initial_shoulder_image_side"] = "image_left" if xy[i, 0] < xy[1-i, 0] else "image_right"
    return {"schema": VERSION, "policy": POLICY, "body_frame": frame, "effectors": bindings,
            "robot_roots_sha256": digest(roots.tolist()),
            "camera_sha256": digest({"parameters": camera["parameters"], "config": camera["config"]}),
            "scope": "fixed parallel robot rig replaced by one operator facing the same workspace; preserve anatomical roles, allow natural human paths",
            "projection_scope": "initial shoulder ORDER follows robot-base order; no exact human joint positions or RGB-to-3D labels are inferred",
            "identity_rule": "track shoulder through elbow to wrist; crossing the image midline NEVER swaps a hand identity"}


def attach(generation, initial, correspondence):
    """One binding drives the anchor, events, continuous holds and review."""
    effectors = {name: {k: item[k] for k in ["human_hand", "robot_index", "initial_shoulder_image_side"]}
                 for name, item in correspondence["effectors"].items()}
    # Native channel names remain in the audit contract, not in model prompts.
    public = {k: v for k, v in correspondence.items()
              if k not in {"effectors", "body_frame", "robot_roots_sha256", "camera_sha256"}}
    public["effectors"] = effectors
    for info in (generation, initial):
        info["hand_correspondence"] = public
        info["effectors"] = effectors
    for event in generation["required_events"]:
        if event.get("effector") is not None:
            event["human_hand"] = effectors[event["effector"]]["human_hand"]
    for condition in generation.get("persistent_constraints", []):
        if condition.get("effector") is not None:
            condition["human_hand"] = effectors[condition["effector"]]["human_hand"]


def review_requirements(info):
    """Structured ownership evidence required in addition to event success."""
    if "hand_correspondence" not in info:
        return {}
    effects = info["hand_correspondence"]["effectors"]
    result = {}
    for event in info["required_events"]:
        name = event.get("effector")
        action = event.get("action", event.get("phase"))
        if name is not None and action in HAND_ACTIONS:
            result["event:" + event["id"]] = {**effects[name], "effector": name, "action": action,
                "entities": event.get("entities", [event.get("object")])}
    for name, state in info.get("terminal_hands", {}).items():
        if state["state"] != "free":
            result["terminal:" + name] = {**effects[name], "effector": name,
                "action": state["state"], "entities": [state["entity"]]}
    return result
