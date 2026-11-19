"""Indirect payload manipulation through a passive lever or removable support.

These recipes use native rigid-body contacts and passive joints. The solver
commands robot joints only; it never sets payload or mechanism state. Existing
event_protocol evaluates the predeclared object/joint/contact conditions.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math

import numpy as np

from .core import WORKSPACE, bind, require, write
from .functional_assets import geometry, preflight
from .library import CAMERAS
from .novel_catalog import box, joint, part
from . import planning_protocols as p

KINDS = ("lever_payload_lift", "support_gate_release")
VERSION = "indirect_payload_v1"
CONTROLLER_VERSION = "bounded_measured_contact_v3"
GATE_CONTACT_RULE_VERSION = "gate_fixture_no_contact_v2"
GATE_SUPPORT_RULE_VERSION = "gate_geometric_support_confirmation_v3"


def _box_world_max_x(pose, half_size):
    """World X envelope of a box from a native unit wxyz pose, in the DSL."""
    w, x, y, z = [p.component(pose, i) for i in range(3, 7)]
    twice = lambda expression: p.expr("mul", p.const(2.), expression)
    row = [p.expr("sub", p.const(1.), twice(p.expr("add", p.expr("mul", y, y), p.expr("mul", z, z)))),
           twice(p.expr("sub", p.expr("mul", x, y), p.expr("mul", w, z))),
           twice(p.expr("add", p.expr("mul", x, z), p.expr("mul", w, y)))]
    radii = [p.expr("mul", p.expr("abs", value), p.const(float(half))) for value, half in zip(row, half_size)]
    radius = p.expr("add", p.expr("add", radii[0], radii[1]), radii[2])
    return p.expr("add", p.component(p.expr("position", pose), 0), radius)


def _transformed_box_min_x(transform, center, half_size):
    """World X minimum of a declared box part from its native link transform."""
    row = [p.component(p.component(transform, column), 0) for column in range(3)]
    center_terms = [p.expr("mul", value, p.const(float(offset))) for value, offset in zip(row, center)]
    radii = [p.expr("mul", p.expr("abs", value), p.const(float(half))) for value, half in zip(row, half_size)]
    center_x = p.expr("add", p.component(p.expr("position", transform), 0),
                      p.expr("add", p.expr("add", center_terms[0], center_terms[1]), center_terms[2]))
    return p.expr("sub", center_x, p.expr("add", p.expr("add", radii[0], radii[1]), radii[2]))


@lru_cache(maxsize=1)
def _closed_panda_tool_envelope():
    """Read the actual hand/finger/camera collision shapes, without simulation."""
    from itertools import product
    import xml.etree.ElementTree as ET
    import trimesh
    from scipy.spatial.transform import Rotation
    urdf = WORKSPACE/"PolicyGen/mani_skill/assets/robots/panda/panda_v3.urdf"
    root = ET.parse(urdf).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    sources = {str(urdf): bind(urdf)}
    def origin(node):
        matrix = np.eye(4)
        if node is not None:
            matrix[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
            matrix[:3, :3] = Rotation.from_euler("xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        return matrix
    start = np.eye(4); start[2, 3] = -.1034
    pending, vertices = [("panda_hand", start)], []
    while pending:
        name, transform = pending.pop()
        for collision in links[name].findall("collision"):
            shape = collision.find("geometry")
            if shape.find("box") is not None:
                half = np.fromstring(shape.find("box").get("size"), sep=" ")/2
                points = np.array(list(product([-1., 1.], repeat=3)))*half
            else:
                mesh = shape.find("mesh")
                require(mesh is not None, "unsupported_panda_tool_collision_shape")
                path = urdf.parent/mesh.get("filename")
                sources[str(path)] = bind(path)
                points = trimesh.load(path, force="mesh").vertices
                points = points*np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            vertices.append(trimesh.transform_points(points, transform@origin(collision.find("origin"))))
        for joint_node in root.findall("joint"):
            if joint_node.find("parent").get("link") == name:
                require(joint_node.get("type") in {"fixed", "prismatic"}, "unsupported_panda_tool_joint")
                # Both finger displacements are zero for the closed pusher.
                pending.append((joint_node.find("child").get("link"), transform@origin(joint_node.find("origin"))))
    # Gate pushing uses this fixed downward grasp orientation.
    world_relative = np.concatenate(vertices)@np.diag([-1., 1., -1.])
    return {"bounds_relative_to_tcp_m": [world_relative.min(0).tolist(), world_relative.max(0).tolist()],
            "sources": list(sources.values()), "finger_state": "closed",
            "scope": "hand subtree including fingers and camera; whole arm checked by runtime collision planner"}


def gate_clearance_preflight(s, minimum_clearance_m=.005):
    """Exact translational swept AABBs; intended gate/payload contact is excluded."""
    require(s["id"] == KINDS[1], "gate_clearance_requires_gate_scene")
    protected = [o for o in s["objects"] if o["id"].startswith(("guide_", "collector_"))]
    tool = deepcopy(_closed_panda_tool_envelope())
    tool_lo, tool_hi = np.asarray(tool["bounds_relative_to_tcp_m"])
    checks = []
    for station in s["stations"]:
        spec = next(j for j in s["joints"] if j["id"] == station["joint"])
        require(spec["axis_world"] == [1., 0., 0.], "unsupported_gate_slide_orientation")
        pivot = np.asarray(station["pivot"])
        moving = []
        for index, piece in enumerate(spec["parts"]):
            center, half = np.asarray(piece["center"]), np.asarray(piece["half_size"])
            moving.append(("gate_part_"+str(index), pivot+center-half+[spec["limits"][0], 0., 0.],
                           pivot+center+half+[spec["limits"][1], 0., 0.]))
        handle = pivot+station["handle_local"]
        moving.append(("closed_panda_tool", handle+[-.065, 0., 0.]+tool_lo,
                       handle+[spec["limits"][1]-.03175, 0., 0.]+tool_hi))
        minimum = float("inf"); limiting = None
        for name, lower, upper in moving:
            for obj in protected:
                for pi, fixed in enumerate(obj["parts"]):
                    center = np.asarray(obj["position"])+fixed["center"]
                    half = np.asarray(fixed["half_size"])
                    distance = float(np.max(np.maximum(center-half-upper, lower-center-half)))
                    if distance < minimum:
                        minimum = distance; limiting = {"moving": name, "fixture": obj["id"], "part": pi}
        checks.append({"joint": station["joint"], "certified_separation_lower_bound_m": minimum,
                       "required_clearance_m": minimum_clearance_m, "slide_range_m": spec["limits"],
                       "limiting_pair": limiting})
        require(minimum >= minimum_clearance_m, "gate_fixture_blocks_declared_sweep:"+station["joint"])
    return {"method": "continuous_prismatic_swept_boxes", "checks": checks, "tool_envelope": tool,
            "scope": "mechanism and closed tool versus protected fixtures; whole arm requires native path checks"}


def lever_clearance_preflight(s, minimum_clearance_m=.002):
    """Conservative continuous-angle separation of the lever and its support.

    Box separating axes give a lower bound on distance at each sampled angle.
    Subtract the maximum corner travel to the nearest sample, so the positive
    bound also covers unsampled angles. Payload contact is intentionally not
    screened here. This checks the designed mechanism, not the robot path.
    """
    require(s["id"] == KINDS[0], "lever_clearance_requires_lever_scene")
    checks = []
    for station in s["stations"]:
        joint_spec = next(j for j in s["joints"] if j["id"] == station["joint"])
        support = next(o for o in s["objects"] if o["id"] == "fulcrum_"+str(station["arm"]))
        require(joint_spec["axis_world"] == [0., 1., 0.] and support["yaw"] == 0.,
                "unsupported_lever_clearance_orientation")
        angles = np.linspace(*joint_spec["limits"], 129)
        half_interval = float(angles[1]-angles[0])/2
        static_parts = support.get("parts", [part([0., 0., 0.], support["half_size"])])
        minimum = float("inf"); limiting = None
        for mi, moving in enumerate(joint_spec["parts"]):
            center, half = np.asarray(moving["center"]), np.asarray(moving["half_size"])
            travel_bound = float(np.linalg.norm(np.abs(center)+half))*half_interval
            for angle in angles:
                c, v = math.cos(angle), math.sin(angle)
                rotation = np.array([[c, 0., v], [0., 1., 0.], [-v, 0., c]])
                world = np.asarray(station["pivot"])+rotation@center
                # For these Y-axis boxes the remaining cross axes duplicate
                # a face normal; these five axes are the complete SAT set.
                axes = np.concatenate([np.eye(3), rotation[:, [0, 2]].T])
                moving_radius = np.abs(axes@rotation)@half
                for si, fixed in enumerate(static_parts):
                    fixed_world = np.asarray(support["position"])+np.asarray(fixed["center"])
                    fixed_radius = np.abs(axes)@np.asarray(fixed["half_size"])
                    separation = float(np.max(np.abs(axes@(world-fixed_world))-moving_radius-fixed_radius))
                    bound = separation-travel_bound
                    if bound < minimum:
                        minimum = bound
                        limiting = {"moving_part": mi, "support_part": si, "sample_angle_rad": float(angle)}
        checks.append({"joint": station["joint"], "support": support["id"],
                       "certified_separation_lower_bound_m": minimum,
                       "required_clearance_m": minimum_clearance_m,
                       "angle_range_rad": joint_spec["limits"], "samples": len(angles),
                       "unsampled_angles_covered_by_corner_travel_bound": True, "limiting_pair": limiting})
        require(minimum >= minimum_clearance_m, "lever_support_blocks_declared_arc:"+station["joint"])
    return {"method": "box_separating_axes_with_continuous_angle_travel_bound", "checks": checks}


def scene(kind, seed=2026102500):
    require(kind in KINDS, "unknown_contact_extension")
    rng = np.random.default_rng(seed)
    offset = rng.uniform([-.008, -.010], [.008, .010])
    s = {"schema": "resimpair_novel_scene_v1", "id": kind, "seed": int(seed),
         "contact_extension": VERSION, "table_top_z": 0.,
         "robot_roots": [[-.60, -.35, 0.], [-.60, .35, 0.]],
         "cameras": deepcopy(CAMERAS), "objects": [], "supports": [], "joints": [],
         "appearance": {"table_rgb": [.47, .45, .42], "background_rgb": [.70, .73, .75],
                        "ambient_rgb": [.38]*3, "key_light_rgb": [1.5]*3},
         "mechanism": "lever_transmitted_lift" if kind == KINDS[0] else "support_removal_gravity_release",
         "novelty_vs_previous": ("raise a separate payload through lever contact, not only change a joint angle"
                                 if kind == KINDS[0] else
                                 "withdraw supporting gate, releasing a separate payload under gravity into a collector"),
         "source_status": "designed_mechanism_probe_without_verified_real_video_window",
         "physics_scope": "native rigid bodies and passive joints; no state teleport or injected force",
         "asset_origin": "procedural functional geometry", "stations": []}
    for index, y in enumerate([-.24, .24]):
        color = ["red", "blue"][index]
        payload = f"payload_{index}"
        x, y = -.13+float(offset[0]), y+float(offset[1])
        if kind == KINDS[0]:
            mechanism = f"lifting_lever_{index}"
            pivot = [x, y, .12]
            # Positive joint rotation lowers the +X input pad and raises -X load.
            pieces = [part([0., 0., 0.], [.135, .025, .008]),
                      part([.105, 0., .012], [.024, .032, .012]),
                      part([-.105, 0., .012], [.030, .030, .004])]
            # A cradle prevents the payload sliding off while the lever tilts.
            for dx, dy, half in [(-.033, 0., [.004, .034, .026]),
                                 (.033, 0., [.004, .034, .026]),
                                 (0., -.033, [.030, .004, .026]),
                                 (0., .033, [.030, .004, .026])]:
                pieces.append(part([-.105+dx, dy, .038], half))
            # A solid block at z=.110 blocked the beam at q=.079 in native_r1.
            # Side posts leave the rotating beam's full swept volume clear.
            support = [part([0., 0., -.025], [.025, .044, .035]),
                       part([0., -.038, .035], [.009, .006, .025]),
                       part([0., .038, .035], [.009, .006, .025])]
            s["objects"].append(box(f"fulcrum_{index}", [x, y, .060], [.025, .044, .060], "dark",
                                    parts=support, description="fork support with side posts outside the lever sweep"))
            s["joints"].append(joint(mechanism, pivot, [0., 1., 0.], "revolute", [0., .44],
                                      pieces, "wood", mass=.10, friction=.08, damping=.025))
            pos = [x-.105, y, .1545]
            s["objects"].append(box(payload, pos, [.018]*3, color, movable=True, mass=.025,
                                    description=f"{color} payload in the output cradle of the lever"))
            station = {"joint": mechanism, "payload": payload, "pivot": pivot,
                       "input_local": [.105, 0., .024], "payload_initial": pos,
                       "minimum_lift_m": .025, "target_joint_rad": .36,
                       "joint_goal_range": [.25, .44], "arm": index}
        else:
            mechanism = f"support_gate_{index}"
            pivot = [x, y, .16]
            side = -1. if index == 0 else 1.
            handle = [-.095, side*.18, .025]
            pieces = [part([0., 0., 0.], [.09, .060, .008]),
                      part([-.085, side*.1125, .010], [.025, .0675, .012]),
                      part(handle, [.023, .026, .020])]
            s["joints"].append(joint(mechanism, pivot, [1., 0., 0.], "prismatic", [0., .16],
                                      pieces, "gray", mass=.06, friction=.20, damping=.15))
            cargo_x = x-.055
            pos = [cargo_x, y, .1865]
            s["objects"].append(box(payload, pos, [.018]*3, color, movable=True, mass=.025,
                                    description=f"{color} payload initially supported by the sliding gate"))
            # Sleeve bottom sits ABOVE gate top (.168), so the gate can slide
            # beneath it; the sleeve arrests horizontal drag but allows falling.
            sleeve = [part([sign*.028, 0., 0.], [.006, .034, .045]) for sign in [-1, 1]]
            sleeve += [part([0., sign*.028, 0.], [.022, .006, .045]) for sign in [-1, 1]]
            s["objects"].append(box(f"guide_{index}", [cargo_x, y, .220], [.034, .034, .045],
                                    "dark", parts=sleeve, description="fixed vertical guide above the removable support"))
            collector = f"collector_{index}"
            tray = [part([0., 0., 0.], [.070, .070, .008])]
            tray += [part([sign*.070, 0., .024], [.006, .076, .024]) for sign in [-1, 1]]
            tray += [part([0., sign*.070, .024], [.064, .006, .024]) for sign in [-1, 1]]
            s["objects"].append(box(collector, [cargo_x, y, .008], [.076, .076, .048], "green",
                                    parts=tray, description="fixed receiving tray underneath the support gate"))
            station = {"joint": mechanism, "payload": payload, "pivot": pivot,
                       "handle_local": handle, "payload_initial": pos, "collector": collector,
                       "collector_center_xy": [cargo_x, y], "target_slide_m": .135,
                       "minimum_drop_m": .08, "arm": index}
        s["stations"].append(station)
    for obj in s["objects"]:
        obj["geometry"] = geometry(obj)
    if kind == KINDS[0]:
        s["mechanism_geometry_revision"] = "swept_clearance_fork_support_v2"
        s["mechanism_clearance"] = lever_clearance_preflight(s)
    else:
        s["mechanism_geometry_revision"] = "outboard_gate_handle_v3"
        s["contact_acceptance_revision"] = GATE_CONTACT_RULE_VERSION
        s["support_acceptance_revision"] = GATE_SUPPORT_RULE_VERSION
        s["mechanism_clearance"] = gate_clearance_preflight(s)
    preflight(s)
    return s


def task(s, branch):
    require(s.get("contact_extension") == VERSION and branch in {0, 1, 2}, "invalid_contact_extension_task")
    selected = [branch] if branch < 2 else [0, 1]
    params = {"selected": selected}
    object_ids = [s["stations"][i][k] for i in selected for k in ["joint", "payload"]]
    return {"schema": "resimpair_task_intent_v4", "id": s["id"]+f"__seed{s['seed']}__b{branch}",
            "family": s["mechanism"], "scene_kind": s["id"], "required_skills": [s["mechanism"]],
            "object_ids": object_ids, "operations": [{"id": "indirect_payload_program", "primitive": s["mechanism"],
                                                       "parameters": params}], "planning_parameters": params,
            "branch": branch, "goal_description": s["novelty_vs_previous"],
            "required_operation_precedence": [], "equivalence_groups": [],
            "human_hand_policy": "preserve_robot_body_side", "non_target_policy": "preserve_identity_and_pose",
            "source_status": s["source_status"], "scope": "indirect contact probe; no certified real-source pairing"}


def protocol(s, t):
    from .event_protocol import validate
    from .hand_correspondence import declare_from_scene
    ids = [o["id"] for o in s["objects"]]
    joint_ids = [j["id"] for j in s["joints"]]
    all_ids = ids+joint_ids
    selected = set(t["object_ids"])
    entities = [{"id": o["id"], "description": o["semantic_description"], "kind": o["shape"],
                 "role": "target" if o["id"] in selected else "unchanged",
                 "geometry": o["geometry"], "pose_channel": o["id"]+".pose"} for o in s["objects"]]
    entities += [{"id": j["id"], "description": j["semantic_description"], "kind": j["joint_type"],
                  "role": "target" if j["id"] in selected else "unchanged",
                  "geometry": {k: j[k] for k in ["pivot_world", "axis_world", "limits", "parts"]},
                  "pose_channel": j["id"]+".pose"} for j in s["joints"]]
    plan = p.base(t["id"], t["family"], entities, t["required_skills"], 12)
    poses, grasps = p.rigid_channels(plan, ids)
    contacts = {}
    for arm in range(2):
        for i, oid in enumerate(all_ids):
            name = f"arm{arm}.{oid}.contact"
            contacts[arm, oid] = p.channel(plan, name, "robot_contact_impulses", [arm, i], "contact_impulse", [oid])
            plan["effectors"][f"arm{arm}"]["contact_channels"][oid] = name
    if s["id"] == KINDS[1]:
        # Apply to old and new geometry alike when re-evaluating a baseline;
        # absence of the scene revision marker must never bypass the rule.
        for arm in range(2):
            for oid in ids:
                if oid.startswith(("guide_", "collector_")):
                    p.invariant(plan, f"arm{arm}.{oid}.fixture_no_touch_v2", "no_contact", [oid],
                                p.expr("le", contacts[arm, oid], p.const(1e-5)), effector=f"arm{arm}",
                                ruleset_version=GATE_CONTACT_RULE_VERSION, maximum_contact_impulse=1e-5)
    qs, vs = {}, {}
    for i, j in enumerate(s["joints"]):
        oid = j["id"]
        qs[oid] = p.channel(plan, oid+".q", "cabinet_qpos", [i], "joint_position", [oid])
        vs[oid] = p.channel(plan, oid+".dq", "cabinet_qvel", [i], "joint_velocity", [oid])
        p.channel(plan, oid+".pose", "door_world", [i], "transform", [oid])
        rest = p.expr("lt", p.expr("abs", qs[oid]), p.const(.01))
        plan["initial_conditions"].append(p.condition(oid+".initial", "joint_range", [oid], rest))
        if oid not in selected:
            p.invariant(plan, oid+".unchanged", "joint_range", [oid], rest)
    for oid in ids:
        if oid not in selected:
            p.unchanged(plan, oid, poses[oid])
    cross = p.channel(plan, "robots.contact", "cross_robot_force_N", [], "contact_force", [])
    p.invariant(plan, "robot_collision", "no_contact", [], p.expr("lt", cross, p.const(.5)), visible=False)
    previous = None
    for index in t["planning_parameters"]["selected"]:
        station = s["stations"][index]
        oid, jid, arm = station["payload"], station["joint"], station["arm"]
        pos = p.expr("position", poses[oid]); z = p.component(pos, 2)
        start_z = p.component(p.expr("initial", pos), 2)
        supported = p.channel(plan, oid+".mechanism_contact", "object_contact_impulses",
                              [all_ids.index(oid), all_ids.index(jid)], "contact_impulse", [oid, jid])
        initial_pos = p.expr("lt", p.expr("norm", p.expr("sub", pos, p.const(station["payload_initial"]))), p.const(.008))
        plan["initial_conditions"].append(p.condition(oid+".initial_position", "at_position", [oid], initial_pos,
                                                      target_position_world=station["payload_initial"], tolerance_m=.008))
        # Initial contact must predate robot actuation. Recorded post-reset
        # contacts, rather than arbitrary start-frame labels, supply evidence.
        support_event = oid+".initial_support"
        p.event(plan, support_event, "supported", [oid, jid], p.expr("all", initial_pos,
                p.expr("gt", supported, p.const(1e-6)), p.expr("lt", p.expr("abs", qs[jid]), p.const(.01))))
        trigger = jid+".robot_trigger"
        after = [support_event]+([previous] if previous else [])
        p.event(plan, trigger, "trigger", [jid], p.expr("gt", contacts[arm, jid], p.const(1e-6)),
                f"arm{arm}", after=after, strict=[support_event])
        # Ordered event searches alone can rediscover an already-true state.
        # Check every earlier frame, including excursions that later return,
        # so spontaneous payload/joint motion cannot masquerade as actuation.
        displacement = p.expr("norm", p.expr("sub", pos, p.expr("initial", pos)))
        p.invariant(plan, oid+".at_rest_before_trigger", "at_position", [oid],
                    p.expr("lt", displacement, p.const(.008)), until=trigger, include_end=False,
                    position_tolerance_m=.008, reference="initial_native_position")
        p.invariant(plan, jid+".at_rest_before_trigger", "joint_range", [jid],
                    p.expr("lt", p.expr("abs", qs[jid]), p.const(.01)),
                    until=trigger, include_end=False, absolute_joint_tolerance=.01)
        for a in range(2):
            p.invariant(plan, f"arm{a}.{oid}.no_direct_touch", "no_contact", [oid],
                        p.expr("le", contacts[a, oid], p.const(1e-5)), effector=f"arm{a}")
        p.invariant(plan, oid+".never_grasped", "no_grasp", [oid],
                    p.expr("not", p.expr("any", grasps[0, oid], grasps[1, oid])))
        velocity = p.channel(plan, oid+".velocity", "object_states", [ids.index(oid), {"slice": [7, 10]}],
                             "linear_velocity", [oid])
        plan["goals"].append(p.condition(oid+".stable", "stable", [oid], p.expr("lt", p.expr("norm", velocity), p.const(.025))))
        if s["id"] == KINDS[0]:
            transfer = oid+".transmitted_contact"
            p.event(plan, transfer, "passive_contact", [jid, oid], p.expr("all",
                    p.expr("gt", supported, p.const(1e-6)), p.expr("gt", qs[jid], p.const(.12))),
                    after=[trigger], strict=[trigger])
            lifted = p.expr("gt", p.expr("sub", z, start_z), p.const(station["minimum_lift_m"]))
            p.invariant(plan, oid+".first_lift_after_transmission", "at_position", [oid, jid],
                        p.expr("not", lifted), until=transfer, include_end=True,
                        maximum_vertical_displacement_m=station["minimum_lift_m"],
                        cause_event=transfer, first_threshold_crossing="strictly_after_cause")
            previous = oid+".lifted"
            p.event(plan, previous, "lift", [oid, jid], lifted, after=[transfer], strict=[transfer])
            joint_goal = p.expr("all", p.expr("ge", qs[jid], p.const(station["joint_goal_range"][0])),
                                p.expr("le", qs[jid], p.const(station["joint_goal_range"][1])))
            plan["goals"] += [p.condition(oid+".raised", "at_position", [oid], lifted,
                                         minimum_vertical_displacement_m=station["minimum_lift_m"]),
                              p.condition(oid+".supported", "supported_by", [oid, jid], p.expr("gt", supported, p.const(1e-6))),
                              p.condition(jid+".raised_state", "joint_range", [jid], joint_goal,
                                          range_rad=station["joint_goal_range"])]
        else:
            support_rule = s.get("support_acceptance_revision")
            require(support_rule in {None, GATE_SUPPORT_RULE_VERSION}, "unknown_gate_support_acceptance_revision")
            moved = jid+".support_removed"
            support_parameters = {}
            if support_rule is None:
                # Explicit backward compatibility with native_r3. A review
                # using v3 must mark a separate scene copy and record that
                # override; loading an old scene never silently changes rules.
                p.event(plan, moved, "articulate", [jid], p.expr("gt", qs[jid], p.const(.065)),
                        f"arm{arm}", after=[trigger], strict=[trigger])
            else:
                payload_spec = next(obj for obj in s["objects"] if obj["id"] == oid)
                joint_spec = next(joint for joint in s["joints"] if joint["id"] == jid)
                require(payload_spec["shape"] == "box" and "parts" not in payload_spec,
                        "support_envelope_requires_declared_box_payload")
                require(joint_spec["joint_type"] == "prismatic" and joint_spec["axis_world"] == [1., 0., 0.],
                        "support_retraction_requires_positive_x_prismatic_gate")
                plate = joint_spec["parts"][0]
                edge = _transformed_box_min_x(p.signal(jid+".pose"), plate["center"], plate["half_size"])
                payload_max_x = _box_world_max_x(poses[oid], payload_spec["half_size"])
                support_lost = p.expr("all", p.expr("gt", edge, p.expr("initial", payload_max_x)),
                                     p.expr("gt", edge, payload_max_x), p.expr("le", supported, p.const(1e-6)))
                candidate = jid+".support_loss_candidate"
                p.event(plan, candidate, "articulate", [jid, oid], support_lost,
                        f"arm{arm}", after=[trigger], strict=[trigger])
                # min_frames marks a sustained window's FIRST frame. A separate
                # event records confirmation on its second frame, so later drop
                # evidence cannot be backdated to a future-confirmed cause.
                plan["events"][-1]["min_frames"] = 2
                p.event(plan, moved, "articulate", [jid, oid], support_lost,
                        f"arm{arm}", after=[candidate], strict=[candidate])
                support_parameters = {
                    "ruleset_version": GATE_SUPPORT_RULE_VERSION, "candidate_event": candidate,
                    "support_definition": "plate_min_x_exceeds_initial_and_current_payload_max_x_and_native_contact_lost",
                    "consecutive_candidate_frames": 2, "confirmation_frame": "first_frame_strictly_after_candidate",
                    "evidence_scope": "native sampled state/contact qualification; candidate and confirmation are not additional human task actions",
                    "maximum_support_contact_impulse": 1e-6,
                    "payload_half_size_m": payload_spec["half_size"], "plate_part_local": plate}
            fallen = oid+".gravity_drop"
            dropped = p.expr("gt", p.expr("sub", start_z, z), p.const(station["minimum_drop_m"]))
            p.invariant(plan, oid+".first_drop_after_support_removal", "at_position", [oid, jid],
                        p.expr("not", dropped), until=moved, include_end=True,
                        maximum_downward_displacement_m=station["minimum_drop_m"],
                        cause_event=moved, first_threshold_crossing="strictly_after_cause", **support_parameters)
            p.event(plan, fallen, "release", [oid, jid], dropped, after=[moved], strict=[moved])
            collector = station["collector"]
            landing = p.channel(plan, oid+".collector_contact", "object_contact_impulses",
                                [ids.index(oid), ids.index(collector)], "contact_impulse", [oid, collector])
            inside = p.expr("all", p.expr("lt", p.expr("norm", p.expr("sub", p.expr("xy", pos),
                            p.const(station["collector_center_xy"]))), p.const(.05)),
                            p.expr("lt", z, p.const(.065)), p.expr("gt", z, p.const(.025)))
            previous = oid+".collected"
            p.event(plan, previous, "supported", [oid, collector], p.expr("all", inside,
                    p.expr("gt", landing, p.const(1e-6))), after=[fallen], strict=[fallen])
            plan["goals"] += [p.condition(oid+".collector_region", "inside", [oid, collector], inside,
                                         center_xy_m=station["collector_center_xy"], radius_m=.05, height_range_m=[.025, .065]),
                              p.condition(oid+".landed", "supported_by", [oid, collector], p.expr("gt", landing, p.const(1e-6))),
                              p.condition(jid+".withdrawn", "joint_range", [jid], p.expr("gt", qs[jid], p.const(.10)),
                                          minimum_slide_m=.10)]
        plan["goals"].append(p.condition(jid+".settled", "stable", [jid], p.expr("lt", p.expr("abs", vs[jid]), p.const(.06))))
    declare_from_scene(plan, s)
    validate(plan)
    return plan


def _gate_fixture_contact(rec, start_frame=0):
    ids = [o["id"] for o in rec.unwrapped.scene_spec["objects"]]
    protected = [i for i, oid in enumerate(ids) if oid.startswith(("guide_", "collector_"))]
    for frame, row in enumerate(rec.rows[start_frame:], start_frame):
        values = np.asarray(row["robot_contact_impulses"])[:, protected]
        if np.any(values > 1e-5):
            arm, column = np.unravel_index(np.argmax(values), values.shape)
            return {"frame": frame, "arm": int(arm), "fixture": ids[protected[column]],
                    "impulse": float(values[arm, column])}
    return None


def _install_gate_fixture_guard(rec, solvers, observations):
    """Plan around fixtures and check whole-robot commands before executing.

    FCL updates only the planner's kinematic model. No simulation actor state
    is assigned. Interpolated commands are sampled at <= .015 rad per joint;
    native no-contact predicates remain the independent physical criterion.
    """
    import mplib
    from .simulation import array
    protected = [o for o in rec.unwrapped.scene_spec["objects"]
                 if o["id"].startswith(("guide_", "collector_"))]
    names = set()
    for solver in solvers:
        for obj in protected:
            for index, piece in enumerate(obj["parts"]):
                name = f"protected_{obj['id']}_{index}"
                names.add(name)
                shape = mplib.collision_detection.fcl.Box(2*(np.asarray(piece["half_size"])+.003))
                center = np.asarray(obj["position"])+piece["center"]
                collision = mplib.collision_detection.fcl.CollisionObject(shape, center, np.array([1., 0., 0., 0.]))
                solver.planner.planning_world.set_normal_object(name, collision)
        original = solver.follow_path
        def guarded(result, *args, _solver=solver, _original=original, **kwargs):
            measured = array(_solver.robot.qpos).reshape(9)
            report = {"type": "protected_fixture_command_preflight", "arm": _solver.multi_robot_id,
                      "stage": rec.stage, "frame": len(rec.rows)-1, "checked_configurations": 0,
                      "fixture_inflation_m": .003, "maximum_joint_step_rad": .015, "status": "checking"}
            observations.append(report)
            previous = measured[:7]
            with rec.performance.section("protected_fixture_whole_robot_command_checks"):
                for command in np.asarray(result["position"]):
                    count = max(1, int(np.ceil(np.max(np.abs(command-previous))/.015)))
                    for fraction in np.linspace(0., 1., count+1):
                        qpos = np.r_[previous+fraction*(command-previous), measured[7:]]
                        collisions = _solver.planner.check_for_env_collision(qpos=qpos)
                        hits = [{"object1": c.object_name1, "object2": c.object_name2,
                                 "link1": c.link_name1, "link2": c.link_name2}
                                for c in collisions if {c.object_name1, c.object_name2} & names]
                        report["checked_configurations"] += 1
                        if hits:
                            report.update(status="rejected", collisions=hits)
                            require(False, "gate_protected_fixture_command_collision:"+rec.stage)
                    previous = command
            report["status"] = "passed_command_checks"
            first_frame = len(rec.rows)
            result = _original(result, *args, **kwargs)
            violation = _gate_fixture_contact(rec, first_frame)
            report["native_contact_violation"] = violation
            require(violation is None, "gate_protected_fixture_native_contact:"+rec.stage)
            return result
        solver.follow_path = guarded


def solve(rec, t):
    observations = []
    status, failure = "running", None
    try:
        _solve_contact_program(rec, t, observations)
        status = "completed"
    except Exception as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Preserve feedback even if IK, collision checks, or bounded progress
        # raises. It remains diagnostic input, never acceptance evidence.
        write(rec.directory/"indirect_payload_controller_measurements.json", {
            "controller_version": CONTROLLER_VERSION, "status": status,
            "failure": failure, "observations": observations,
            "scope": "controller feedback only; independent event predicates determine qualification"})


def _solve_contact_program(rec, t, observations):
    import sapien
    from .simulation import array
    from .workbench_solvers import planners, move
    from .tool_motion import linear
    b = rec.unwrapped
    require(b.scene_spec.get("contact_extension") == VERSION, "extension_scene_missing")
    if t["scene_kind"] == KINDS[0]:
        lever_clearance_preflight(b.scene_spec)
    else:
        gate_clearance_preflight(b.scene_spec)
    ps = planners(rec, .35)
    if t["scene_kind"] == KINDS[1]:
        _install_gate_fixture_guard(rec, ps, observations)
    for index in t["planning_parameters"]["selected"]:
        station = b.scene_spec["stations"][index]
        arm = station["arm"]; pivot = np.asarray(station["pivot"])
        agent = b.agent.agents[arm]
        q = agent.build_grasp_pose(np.array([0., 0., -1.]), np.array([0., 1., 0.]), pivot).q
        rec.stage = "indirect:close_pusher"
        ps[arm].close_gripper(other_gripper_state=ps[1-arm].gripper_state, t=10)
        payload = b.objects[b.object_ids.index(station["payload"])]
        initial_z = float(array(payload.pose.p).reshape(3)[2])
        joint_index = next(i for i, j in enumerate(b.scene_spec["joints"]) if j["id"] == station["joint"])
        if t["scene_kind"] == KINDS[0]:
            radius = station["input_local"][0]
            above = pivot+np.array([radius, 0., .18])
            move(rec, ps, arm, sapien.Pose(above, q), "lever_payload:above", refine=6)
            linear(rec, ps, arm, sapien.Pose(pivot+[radius, 0., .045], q), "lever_payload:approach_pad")
            arrived = False
            stalls = 0
            for iteration in range(20):
                before = float(array(b.articulations[joint_index].qpos).reshape(-1)[0])
                # Follow the measured mechanism, never continue a blind arc
                # through a jam. The closed Panda fingers project about 10 mm
                # below TCP (native_r1 contact observations); an 8 mm TCP pad
                # clearance supplies only 2 mm of bounded contact overtravel.
                angle = min(before+.025, station["target_joint_rad"]+.04)
                c, v = math.cos(angle), math.sin(angle)
                point = pivot+np.array([[c, 0., v], [0., 1., 0.], [-v, 0., c]])@np.asarray(station["input_local"])
                point[2] += .008
                feedback = {"station": index, "iteration": iteration, "frame_before": len(rec.rows)-1,
                            "measured_angle_before_rad": before, "commanded_angle_rad": angle,
                            "desired_tcp_world": point.tolist(), "status": "command_pending"}
                observations.append(feedback)
                linear(rec, ps, arm, sapien.Pose(point, q), "lever_payload:press_pad")
                rec.hold(4, tuple(x.gripper_state for x in ps))
                joint_q = float(array(b.articulations[joint_index].qpos).reshape(-1)[0])
                lift = float(array(payload.pose.p).reshape(3)[2])-initial_z
                actual = array(agent.tcp.pose.p).reshape(3)
                feedback.update({"q": joint_q, "lift_m": lift, "frame": len(rec.rows)-1,
                                 "actual_tcp_world": actual.tolist(),
                                 "tcp_tracking_error_m": float(np.linalg.norm(actual-point)), "status": "measured"})
                if lift > station["minimum_lift_m"]+.004 and joint_q >= .29:
                    arrived = True
                    break
                stalls = stalls+1 if joint_q-before < .002 else 0
                feedback["consecutive_stalled_commands"] = stalls
                require(stalls < 3, "lever_no_measured_progress_within_three_small_commands")
            require(arrived, "lever_payload_not_lifted_within_bounded_press")
        else:
            handle = pivot+np.asarray(station["handle_local"])
            start = handle+[-.065, 0., 0.]
            move(rec, ps, arm, sapien.Pose(start+[0., 0., .17], q), "gate:above_handle", refine=6)
            linear(rec, ps, arm, sapien.Pose(start, q), "gate:behind_handle")
            arrived = False
            stalls = 0
            for iteration in range(12):
                before = float(array(b.articulations[joint_index].qpos).reshape(-1)[0])
                commanded = min(before+.020, station["target_slide_m"]+.015)
                # Native handle half-X=.023 and closed fingertip half-X=.00875.
                # Use the measured gate state, with finite 20 mm increments.
                point = handle+[commanded-.03175, 0., 0.]
                feedback = {"station": index, "iteration": iteration, "frame_before": len(rec.rows)-1,
                            "measured_slide_before_m": before, "commanded_slide_m": commanded,
                            "desired_tcp_world": point.tolist(), "status": "command_pending"}
                observations.append(feedback)
                first_frame = len(rec.rows)
                linear(rec, ps, arm, sapien.Pose(point, q), "gate:withdraw_support")
                rec.hold(4, tuple(x.gripper_state for x in ps))
                joint_q = float(array(b.articulations[joint_index].qpos).reshape(-1)[0])
                drop = initial_z-float(array(payload.pose.p).reshape(3)[2])
                actual = array(agent.tcp.pose.p).reshape(3)
                error = float(np.linalg.norm(actual-point))
                violation = _gate_fixture_contact(rec, first_frame)
                feedback.update({"q": joint_q, "drop_m": drop, "frame": len(rec.rows)-1,
                                 "actual_tcp_world": actual.tolist(), "tcp_tracking_error_m": error,
                                 "native_contact_violation": violation, "status": "measured"})
                require(violation is None, "gate_protected_fixture_native_contact")
                require(error <= .015, "gate_tcp_not_tracking_bounded_push")
                if joint_q >= .11 and drop >= station["minimum_drop_m"]:
                    arrived = True
                    break
                stalls = stalls+1 if joint_q-before < .001 else 0
                feedback["consecutive_stalled_commands"] = stalls
                require(stalls < 3, "gate_no_measured_progress_within_three_small_commands")
            require(arrived, "gate_payload_not_released_within_bounded_push")
        point = array(agent.tcp.pose.p).reshape(3)
        linear(rec, ps, arm, sapien.Pose(point+[0., 0., .15], q), "indirect:withdraw")
        ps[arm].open_gripper(other_gripper_state=ps[1-arm].gripper_state)
        rec.hold(35, tuple(x.gripper_state for x in ps))
    rec.stage = "indirect:final_settle"
    rec.hold(40, tuple(x.gripper_state for x in ps))
    if t["scene_kind"] == KINDS[1]:
        require(_gate_fixture_contact(rec) is None, "gate_protected_fixture_contact_during_execution")
