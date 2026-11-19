"""Reusable protocol builders owned by the planning/export side.

They declare predicates from scene geometry and task intent, before observing
the execution. Generation and review never dispatch on these mechanisms.
"""
from __future__ import annotations

from .core import require
from .event_protocol import VERSION, validate


def const(value):
    return {"op": "constant", "value": value}


def signal(name):
    return {"op": "signal", "name": name}


def expr(operator, *args):
    return {"op": operator, "args": list(args)}


def component(value, index):
    return {"op": "component", "args": [value], "index": index}


def channel(plan, name, array, indices, kind, entities):
    require(name not in plan["channels"], "duplicate_planned_channel")
    plan["channels"][name] = {"array": array, "indices": indices, "kind": kind,
                              "entities": entities, "origin": "native_measurement"}
    return signal(name)


def condition(cid, relation, entities, predicate, **parameters):
    return {"id": cid, "relation": relation, "entities": entities, "predicate": predicate, "parameters": parameters}


def invariant(plan, cid, relation, entities, predicate, start="initial", until="terminal", include_end=True,
              visible=True, effector=None, **parameters):
    item = {**condition(cid, relation, entities, predicate, **parameters), "from": start, "until": until,
            "include_end": include_end, "human_visible_check": visible, "effector": effector}
    plan["constraints"].append(item)


def event(plan, eid, action, entities, predicate, effector=None, after=(), strict=(), terminal=False):
    plan["events"].append({"id": eid, "action": action, "entities": entities, "effector": effector,
                           "when": predicate, "selection": "terminal" if terminal else "first", "min_frames": 1,
                           "strictly_after": list(strict)})
    for before in after:
        plan["precedence"].append([before, eid])


def base(task_id, family, entities, skills, duration=8):
    return {"schema": VERSION, "task_id": task_id, "family": family, "required_skills": skills,
            "entities": entities, "effectors": {f"arm{i}": {"human_hand": "either", "grasp_channels": {},
                                                             "contact_channels": {}} for i in range(2)},
            "channels": {}, "events": [], "precedence": [], "initial_conditions": [], "goals": [],
            "constraints": [], "initial_hands": {f"arm{i}": {"state": "free", "entity": None} for i in range(2)},
            "terminal_hands": {f"arm{i}": {"state": "free", "entity": None} for i in range(2)},
            "equivalence_groups": [], "duration_seconds": duration, "terminal_window_frames": 20}


def rigid_channels(plan, object_ids):
    poses, grasps = {}, {}
    for i, oid in enumerate(object_ids):
        poses[oid] = channel(plan, oid + ".pose", "object_states", [i, {"slice": [0, 7]}], "pose_wxyz", [oid])
        for arm in range(2):
            name = f"arm{arm}.{oid}.grasp"
            grasps[arm, oid] = channel(plan, name, "grasp", [arm, i], "grasp", [oid])
            plan["effectors"][f"arm{arm}"]["grasp_channels"][oid] = name
    return poses, grasps


def unchanged(plan, oid, pose, *, contact=None):
    displacement = expr("norm", expr("sub", expr("position", pose), expr("position", expr("initial", pose))))
    rotation = expr("relative_angle", pose, expr("initial", pose))
    invariant(plan, oid + ".unchanged", "unchanged_pose", [oid],
              expr("all", expr("lt", displacement, const(.008)), expr("lt", rotation, const(.05))),
              position_tolerance_m=.008, rotation_tolerance_rad=.05)
    if contact is not None:
        invariant(plan, oid + ".untouched", "no_contact", [oid], contact)


def pick_place(scene, task, object_ids, support_ids, targets):
    """All relocation/packing/sorting/stacking/arranging share this builder."""
    selected = set(task["object_ids"])
    entities = [{"id": o["id"], "description": o["semantic_description"], "kind": o["shape"],
                 "role": "target" if o["id"] in selected else "unchanged",
                 "geometry": {"size_m": [2 * h for h in o["half_size"]], "color": o["color_name"]},
                 "pose_channel": o["id"] + ".pose"} for o in scene["objects"]]
    entities += [{"id": s["id"], "description": s["semantic_description"], "kind": s["shape"], "role": "support",
                  "geometry": {"size_m": [2 * h for h in s["half_size"]]},
                  "fixed_position_world": [s["position"][0], s["position"][1], s["top_z"]]} for s in scene["supports"]]
    entities.append({"id": "table", "description": "the tabletop", "kind": "plane", "role": "support", "geometry": {}})
    plan = base(task["id"], task["family"], entities, task["required_skills"], 8 if len(task["operations"]) <= 2 else 12)
    plan["equivalence_groups"] = task["equivalence_groups"]
    poses, grasps = rigid_channels(plan, object_ids)
    tcps = [channel(plan, f"arm{i}.tcp", "tcp_world", [i], "transform", []) for i in range(2)]
    cross = channel(plan, "robots.contact", "cross_robot_force_N", [], "contact_force", [])
    invariant(plan, "robot_collision", "no_contact", [], expr("lt", cross, const(.5)), visible=False)
    require(len({op["object"] for op in task["operations"]}) == len(task["operations"]), "repeated_transfer_needs_distinct_operation_windows")
    for op in task["operations"]:
        require(op["primitive"] == "pick_place", "planning_plugin_requires_pick_place")
        oid, arm, sid, prefix = op["object"], op["arm"], op["target"]["support"], op["id"]
        i = object_ids.index(oid)
        pos = expr("position", poses[oid])
        held = grasps[arm, oid]
        free = expr("not", expr("any", grasps[0, oid], grasps[1, oid]))
        support = channel(plan, oid + ".support", "support_impulses", [i, support_ids.index(sid)], "contact_impulse", [oid, sid])
        supported = expr("gt", support, const(1e-6))
        distance = expr("norm", expr("sub", pos, const(targets[oid])))
        arrived = expr("lt", distance, const(.015))
        lifted = expr("all", held, expr("gt", component(pos, 2), expr("add", component(expr("initial", pos), 2), const(.045))))
        clearance = expr("all", *[expr("gt", expr("norm", expr("sub", expr("position", tcp), pos)), const(.075)) for tcp in tcps])
        event(plan, prefix + ".grasp", "grasp", [oid], held, f"arm{arm}")
        event(plan, prefix + ".lift", "lift", [oid], lifted, f"arm{arm}", [prefix + ".grasp"])
        event(plan, prefix + ".arrive", "arrive", [oid, sid], arrived, f"arm{arm}",
              [prefix + ".lift"], [prefix + ".lift"])
        event(plan, prefix + ".release", "release", [oid], free, f"arm{arm}", [prefix + ".arrive"])
        event(plan, prefix + ".supported", "supported", [oid, sid], expr("all", supported, free, arrived),
              after=[prefix + ".release"])
        event(plan, prefix + ".withdraw", "withdraw", [oid], expr("all", clearance, free), f"arm{arm}", [prefix + ".release"])
        invariant(plan, prefix + ".held", "continuous_grasp", [oid], held,
                  start=prefix + ".lift", until=prefix + ".arrive", include_end=False, effector=f"arm{arm}")
        velocity = channel(plan, oid + ".velocity", "object_states", [i, {"slice": [7, 10]}], "linear_velocity", [oid])
        angular = channel(plan, oid + ".angular", "object_states", [i, {"slice": [10, 13]}], "angular_velocity", [oid])
        plan["goals"] += [condition(prefix + ".position", "at_position", [oid, sid], arrived,
                                     target_position_world=targets[oid], tolerance_m=.015),
                          condition(prefix + ".support", "supported_by", [oid, sid], expr("all", supported, free)),
                          condition(prefix + ".stable", "stable", [oid],
                                    expr("all", expr("lt", expr("norm", velocity), const(.025)),
                                         expr("lt", expr("norm", angular), const(.4)),
                                         expr("gt", expr("upright", poses[oid]), const(.95)))),
                          condition(prefix + ".clear", "clearance", [oid], clearance, minimum_m=.075)]
    for before, after in task["required_operation_precedence"]:
        plan["precedence"].append([before + ".supported", after + ".arrive"])
    for oid in set(object_ids) - selected:
        forces = [channel(plan, f"arm{arm}.{oid}.force", "robot_object_force_N", [arm, object_ids.index(oid)],
                          "contact_force", [oid]) for arm in range(2)]
        unchanged(plan, oid, poses[oid], contact=expr("all", *[expr("lt", f, const(.5)) for f in forces]))
    from .hand_correspondence import declare_from_scene
    declare_from_scene(plan, scene)
    validate(plan)
    return plan


def articulation(task_id, entities, object_ids, joint_entities, active_joint, active_arm, target_range):
    """A hinge or slider is a measured joint, never a loose object transfer."""
    plan = base(task_id, "articulation", entities, ["articulate"])
    poses, _ = rigid_channels(plan, object_ids)
    qs, gs = [], {}
    for j, oid in enumerate(joint_entities):
        qs.append(channel(plan, oid + ".joint", "cabinet_qpos", [j], "joint_position", [oid]))
        channel(plan, oid + ".pose", "door_world", [j], "transform", [oid])
        for arm in range(2):
            name = f"arm{arm}.{oid}.grasp"
            gs[arm, j] = channel(plan, name, "door_grasp", [arm, j], "grasp", [oid])
            plan["effectors"][f"arm{arm}"]["grasp_channels"][oid] = name
    oid, q = joint_entities[active_joint], qs[active_joint]
    held = gs[active_arm, active_joint]
    opened = expr("all", expr("ge", q, const(target_range[0])), expr("le", q, const(target_range[1])))
    event(plan, "handle.grasp", "grasp", [oid], held, f"arm{active_arm}")
    event(plan, "joint.motion", "articulate", [oid],
          expr("all", held, expr("gt", expr("abs", expr("sub", q, expr("initial", q))), const(.05))),
          f"arm{active_arm}", ["handle.grasp"])
    event(plan, "joint.goal", "articulate", [oid], opened, f"arm{active_arm}", ["joint.motion"])
    event(plan, "handle.release", "release", [oid], expr("not", held), f"arm{active_arm}", ["joint.goal"])
    invariant(plan, "hold_during_articulation", "continuous_grasp", [oid], held,
              start="joint.motion", until="joint.goal", include_end=True, effector=f"arm{active_arm}")
    velocity = channel(plan, oid + ".joint_velocity", "cabinet_qvel", [active_joint], "joint_velocity", [oid])
    plan["goals"] += [condition("open_joint", "joint_range", [oid], opened, range_rad=target_range),
                       condition("settled_joint", "stable", [oid], expr("lt", expr("abs", velocity), const(.02)))]
    for j, other in enumerate(joint_entities):
        plan["initial_conditions"].append(condition(other + ".closed", "joint_range", [other],
                                                    expr("lt", expr("abs", qs[j]), const(.02)), range_rad=[-.02, .02]))
        if j != active_joint:
            invariant(plan, other + ".closed", "joint_range", [other], expr("lt", expr("abs", qs[j]), const(.02)), range_rad=[-.02, .02])
    for oid, pose in poses.items():
        unchanged(plan, oid, pose)
    validate(plan)
    return plan


def insertion(task_id, entities, object_ids, port_center, receiver_face_z, tongue_tip_z, depth_range, lateral_tolerance):
    """Continuous holder grasp and relative interface geometry are explicit."""
    require(len(object_ids) == 2, "insertion_plugin_requires_receiver_and_insert")
    holder, plug = object_ids
    plan = base(task_id, "hold_align_insert", entities, ["pick_place", "insert"], 12)
    poses, grasps = rigid_channels(plan, object_ids)
    # The holder is native arm0; its anatomical side is resolved from the rig
    # geometry by the compiler, never from the holder/insert semantic role.
    plan["terminal_hands"]["arm0"] = {"state": "holding", "entity": holder}
    relative = expr("relative_position", poses[plug], poses[holder])
    lateral = expr("norm", expr("sub", expr("xy", relative), const(port_center[:2])))
    depth = expr("sub", const(receiver_face_z - tongue_tip_z), component(relative, 2))
    aligned = expr("all", expr("lt", lateral, const(lateral_tolerance)),
                   expr("ge", depth, const(depth_range[0])), expr("le", depth, const(depth_range[1])),
                   expr("lt", expr("relative_angle", poses[plug], poses[holder]), const(.08)))
    contact = channel(plan, "mating.contact", "interface_impulses", [], "contact_impulse", [holder, plug])
    event(plan, "receiver.grasp", "grasp", [holder], grasps[0, holder], "arm0")
    event(plan, "plug.grasp", "grasp", [plug], grasps[1, plug], "arm1", ["receiver.grasp"])
    event(plan, "plug.insert", "insert", [plug, holder],
          expr("all", aligned, expr("gt", contact, const(1e-5)), grasps[0, holder], grasps[1, plug]),
          "arm1", ["plug.grasp"])
    event(plan, "plug.release", "release", [plug], expr("not", grasps[1, plug]), "arm1", ["plug.insert"])
    invariant(plan, "receiver.continuously_held", "continuous_grasp", [holder], grasps[0, holder],
              start="plug.grasp", until="terminal", effector="arm0")
    plan["goals"] += [condition("mating_geometry", "inserted", [plug, holder], aligned,
                                selected_port_center_local=port_center, depth_range_m=depth_range,
                                lateral_tolerance_m=lateral_tolerance, receiver_face_z_m=receiver_face_z,
                                tongue_tip_z_m=tongue_tip_z),
                       condition("inserted_contact", "contact", [plug, holder], expr("gt", contact, const(1e-5)))]
    validate(plan)
    return plan


def passive_chain(task_id, entities, object_ids, selected_ids):
    plan = base(task_id, "push_passive_chain", entities, ["push"])
    poses, _ = rigid_channels(plan, object_ids)
    contacts = {}
    for oid in object_ids:
        for arm in range(2):
            contacts[arm, oid] = channel(plan, f"arm{arm}.{oid}.contact", "robot_contact_impulses",
                                          [arm, object_ids.index(oid)], "contact_impulse", [oid])
    first = selected_ids[0]
    event(plan, "chain.trigger", "trigger", [first], expr("gt", contacts[0, first], const(1e-5)), "arm0")
    previous = "chain.trigger"
    for index, (a, b) in enumerate(zip(selected_ids[:-1], selected_ids[1:])):
        hit = channel(plan, f"adjacent.{index}", "object_contact_impulses", [object_ids.index(a), object_ids.index(b)],
                      "contact_impulse", [a, b])
        eid = f"chain.propagate_{index}"
        event(plan, eid, "passive_contact", [a, b], expr("gt", hit, const(1e-5)), after=[previous])
        previous = eid
    for oid in object_ids:
        upright = expr("abs", expr("upright", poses[oid]))
        plan["initial_conditions"].append(condition(oid + ".upright", "upright", [oid], expr("gt", upright, const(.9))))
        if oid in selected_ids:
            plan["goals"].append(condition(oid + ".fallen", "toppled", [oid], expr("lt", upright, const(.45))))
        else:
            unchanged(plan, oid, poses[oid])
            plan["goals"].append(condition(oid + ".standing", "upright", [oid], expr("gt", upright, const(.9))))
        invariant(plan, oid + ".never_grasped", "no_grasp", [oid], expr("not", expr("any", *[
            signal(plan["effectors"][f"arm{arm}"]["grasp_channels"][oid]) for arm in range(2)])))
        for arm in range(2):
            if oid != first or arm != 0:
                invariant(plan, f"arm{arm}.{oid}.no_touch", "no_contact", [oid],
                          expr("le", contacts[arm, oid], const(1e-5)), effector=f"arm{arm}")
    validate(plan)
    return plan
