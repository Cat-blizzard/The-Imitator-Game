"""Camera-relative task identity, derived only from exported scene geometry.

Image positions are simulation projections, not annotations of generated people
or proof of visibility. Feature identity stays fixed when a held object rotates.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .core import require


def image_order(points, minimum_separation=.008):
    """Name an image axis only when every member is distinguishable on it."""
    if len(points) < 2:
        return None
    names = list(points)
    xy = np.asarray([points[name] for name in names], dtype=float)
    require(xy.shape == (len(names), 2) and np.isfinite(xy).all(), "invalid_image_points")
    axis = int(np.argmax(np.ptp(xy, axis=0)))
    order = np.argsort(xy[:, axis], kind="stable")
    if np.min(np.diff(xy[order, axis])) < minimum_separation:
        return None
    return {"relation": "left_to_right_in_initial_image" if axis == 0 else "top_to_bottom_in_initial_image",
            "members": [names[i] for i in order],
            "first_descriptor": "image-leftmost" if axis == 0 else "image-topmost",
            "last_descriptor": "image-rightmost" if axis == 0 else "image-bottommost"}


def transform_point(pose, local):
    pose, local = np.asarray(pose), np.asarray(local, dtype=float)
    require(local.shape == (3,) and np.isfinite(local).all(), "invalid_functional_point")
    if pose.shape == (4, 4):
        return (pose @ np.r_[local, 1.])[:3]
    require(pose.shape == (7,), "functional_point_requires_rigid_pose")
    return pose[:3] + Rotation.from_quat(pose[[4, 5, 6, 3]]).apply(local)


def functional_points(geometry):
    # The explicit form is mechanism independent. Adapters retain compatibility
    # with previously recorded geometry; no old protocol or observation changes.
    if "functional_points" in geometry:
        result = geometry["functional_points"]
    else:
        result = [{"id": f"socket_{i}", "kind": "socket", "center_local": point}
                  for i, point in enumerate(geometry.get("ports", []))]
        if "handle_center_local" in geometry:
            result.append({"id": "handle", "kind": "handle", "center_local": geometry["handle_center_local"]})
    require(len({p["id"] for p in result}) == len(result), "duplicate_functional_point_id")
    return result


def compile_grounding(protocol, channels, inventory, camera):
    from .compiler import project
    positions = {e["id"]: e["initial_image_xy_normalized"] for e in inventory if "initial_image_xy_normalized" in e}
    # Connected interaction components provide comparison groups without mixing
    # unrelated distractors into the ordinal identity of an identical-object row.
    neighbors = {}
    for event in protocol["events"]:
        ids = [i for i in event["entities"] if i in positions]
        for entity in ids:
            neighbors.setdefault(entity, set()).update(set(ids) - {entity})
    groups, remaining = [], set(neighbors)
    while remaining:
        todo, members = [min(remaining)], set()
        while todo:
            current = todo.pop()
            if current in members:
                continue
            members.add(current)
            todo.extend(neighbors[current] - members)
        remaining -= members
        order = image_order({i: positions[i] for i in sorted(members)})
        if order:
            groups.append(order)
    events = []
    for event in protocol["events"]:
        refs = []
        for entity in event["entities"]:
            ref = {"entity": entity}
            if entity in positions:
                ref["initial_image_xy_normalized"] = positions[entity]
            for group in groups:
                if entity in group["members"]:
                    index = group["members"].index(entity)
                    ref.update(comparison_group=group["members"], order=group["relation"], ordinal=index + 1)
                    if index == 0:
                        ref["endpoint"] = group["first_descriptor"]
                    elif index == len(group["members"]) - 1:
                        ref["endpoint"] = group["last_descriptor"]
            refs.append(ref)
        events.append({"event": event["id"], "entities": refs})
    features = []
    for entity in protocol["entities"]:
        points = functional_points(entity["geometry"])
        if not points:
            continue
        require("pose_channel" in entity, "functional_geometry_requires_entity_pose")
        poses = channels[entity["pose_channel"]]
        projected = []
        for point in points:
            projected.append({**point,
                "initial_image_xy_normalized": project(transform_point(poses[0], point["center_local"]), camera),
                "final_simulation_image_xy_normalized": project(transform_point(poses[-1], point["center_local"]), camera)})
        features.append({"entity": entity["id"], "points": projected,
                         "initial_order": image_order({p["id"]: p["initial_image_xy_normalized"] for p in projected})})
    selections = []
    for goal in protocol["goals"]:
        params = goal["parameters"]
        selected = params.get("selected_feature")
        if selected is not None:
            matches = [(e, p) for e in features if e["entity"] == selected["entity"]
                       for p in e["points"] if p["id"] == selected["feature_id"]]
        elif "selected_port_center_local" in params:
            # Old exported insertion contracts carry a geometric identity rather
            # than a feature ID. Resolve it only if the correspondence is unique.
            matches = [(e, p) for e in features if e["entity"] in goal["entities"]
                       for p in e["points"] if p["kind"] == "socket" and
                       np.allclose(p["center_local"], params["selected_port_center_local"], atol=1e-8, rtol=0)]
        else:
            continue
        require(len(matches) == 1, "selected_functional_feature_not_uniquely_resolved")
        entity, point = matches[0]
        selection = {"goal": goal["id"], "entity": entity["entity"], "feature_id": point["id"],
                     "initial_image_xy_normalized": point["initial_image_xy_normalized"],
                     "other_features": [p["id"] for p in entity["points"] if p["id"] != point["id"]],
                     "identity_rule": "track this same physical feature when the object moves or rotates; do not reselect by current image side"}
        order = entity["initial_order"]
        if order:
            selection.update(initial_order=order["relation"], initial_ordinal=order["members"].index(point["id"]) + 1)
        selections.append(selection)
    return {"initial_interaction_groups": groups, "event_entity_grounding": events,
            "functional_features": features, "selected_features": selections,
            "scope": "fixed simulation-camera projections; image order anchors initial identity, not human body coordinates or an occlusion guarantee"}
