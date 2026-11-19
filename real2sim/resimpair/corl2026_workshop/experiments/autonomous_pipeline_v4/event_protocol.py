"""A declarative planning interface evaluated from native measurements.

There are no task-ID branches, executable expressions, controller-stage inputs,
handwritten event frames, or supplied success flags in this interpreter.
Mechanism-specific declarations belong to the planning/export boundary.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .core import identifier, require

VERSION = "resimpair_event_protocol_v1"
ACTIONS = {"grasp", "lift", "arrive", "release", "supported", "withdraw",
           "articulate", "trigger", "passive_contact", "insert", "terminal"}
RELATIONS = {"at_position", "supported_by", "inside", "joint_range", "unchanged_pose",
             "continuous_grasp", "no_grasp", "no_contact", "contact", "stable",
             "upright", "toppled", "inserted", "clearance", "held", "geometry_preserved"}
CHANNEL_SHAPES = {"grasp": (), "position": (3,), "pose_wxyz": (7,), "transform": (4, 4),
                  "joint_position": (), "joint_velocity": (), "linear_velocity": (3,),
                  "angular_velocity": (3,), "contact_impulse": (), "contact_force": ()}
UNARY = {"initial", "abs", "norm", "not", "position", "upright", "xy"}
BINARY = {"add", "sub", "mul", "lt", "le", "gt", "ge", "eq", "relative_position", "relative_angle"}

TEXT = {"type": "string", "minLength": 1}
STRINGS = {"type": "array", "uniqueItems": True, "items": TEXT}
EXPRESSION = {"type": "object", "required": ["op"]}


def obj(properties, required=None):
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties) if required is None else required}


CONDITION = obj({"id": TEXT, "relation": {"enum": sorted(RELATIONS)}, "entities": STRINGS,
                 "predicate": EXPRESSION, "parameters": {"type": "object"},
                 "effector": {"type": ["string", "null"]}},
                ["id", "relation", "entities", "predicate", "parameters"])
CONSTRAINT = obj({**CONDITION["properties"],
                  "from": TEXT, "until": TEXT, "include_end": {"type": "boolean"},
                  "human_visible_check": {"type": "boolean"}},
                 [*CONDITION["required"], "from", "until", "include_end", "human_visible_check"])
HAND = obj({"state": {"enum": ["free", "holding", "contacting"]},
            "entity": {"type": ["string", "null"]}})
SCHEMA = obj({
    "schema": {"const": VERSION}, "task_id": TEXT, "family": TEXT,
    "required_skills": {"type": "array", "minItems": 1, "uniqueItems": True, "items": TEXT},
    "entities": {"type": "array", "minItems": 1, "items": obj({
        "id": TEXT, "description": TEXT, "kind": TEXT,
        "role": {"enum": ["target", "unchanged", "support"]}, "geometry": {"type": "object"},
        "pose_channel": TEXT, "fixed_position_world": {"type": "array", "minItems": 3, "maxItems": 3,
                                                     "items": {"type": "number"}}},
        ["id", "description", "kind", "role", "geometry"])},
    "effectors": {"type": "object", "minProperties": 1, "additionalProperties": obj({
        "human_hand": {"enum": ["either", "left", "right"]},
        "grasp_channels": {"type": "object", "additionalProperties": TEXT},
        "contact_channels": {"type": "object", "additionalProperties": TEXT}})},
    "channels": {"type": "object", "minProperties": 1, "additionalProperties": obj({
        "array": TEXT, "indices": {"type": "array", "items": {"oneOf": [
            {"type": "integer", "minimum": 0},
            obj({"slice": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}})]}},
        "kind": {"enum": sorted(CHANNEL_SHAPES)}, "entities": STRINGS,
        "origin": {"const": "native_measurement"}})},
    "events": {"type": "array", "minItems": 1, "items": obj({
        "id": TEXT, "action": {"enum": sorted(ACTIONS)}, "entities": STRINGS,
        "effector": {"type": ["string", "null"]}, "when": EXPRESSION,
        "selection": {"enum": ["first", "terminal"]}, "min_frames": {"type": "integer", "minimum": 1},
        "strictly_after": STRINGS})},
    "precedence": {"type": "array", "uniqueItems": True, "items": {"type": "array", "items": TEXT,
                                                                     "minItems": 2, "maxItems": 2}},
    "initial_conditions": {"type": "array", "items": CONDITION},
    "goals": {"type": "array", "minItems": 1, "items": CONDITION},
    "constraints": {"type": "array", "items": CONSTRAINT},
    "initial_hands": {"type": "object", "additionalProperties": HAND},
    "terminal_hands": {"type": "object", "additionalProperties": HAND},
    "equivalence_groups": {"type": "array", "items": STRINGS},
    "duration_seconds": {"type": "integer", "minimum": 4, "maximum": 15},
    "terminal_window_frames": {"type": "integer", "minimum": 1},
})


def _expression(node, channel_ids, depth=0):
    require(isinstance(node, dict) and depth <= 24, "invalid_or_excessive_predicate_expression")
    op = node.get("op")
    if op == "signal":
        require(set(node) == {"op", "name"} and node["name"] in channel_ids, "unknown_measurement_channel")
    elif op == "constant":
        require(set(node) == {"op", "value"}, "invalid_constant")
        a = np.asarray(node["value"])
        require(a.ndim <= 1 and a.size <= 16 and a.dtype.kind in "biuf" and np.isfinite(a).all(), "invalid_constant")
    elif op == "component":
        require(set(node) == {"op", "args", "index"} and type(node["index"]) is int and node["index"] >= 0,
                "invalid_component")
        require(len(node["args"]) == 1, "invalid_expression_arity")
        _expression(node["args"][0], channel_ids, depth + 1)
    else:
        require(op in UNARY | BINARY | {"all", "any"} and set(node) == {"op", "args"}, "unsupported_predicate_operator")
        args = node["args"]
        require(isinstance(args, list) and 1 <= len(args) <= 64, "invalid_expression_arity")
        require(op not in UNARY or len(args) == 1, "invalid_expression_arity")
        require(op not in BINARY or len(args) == 2, "invalid_expression_arity")
        for arg in args:
            _expression(arg, channel_ids, depth + 1)


def validate(protocol):
    from jsonschema import validate as json_validate
    json_validate(protocol, SCHEMA)
    entities = {e["id"] for e in protocol["entities"]}
    require(len(entities) == len(protocol["entities"]), "duplicate_entity")
    effects, channels = protocol["effectors"], protocol["channels"]
    for name in entities | set(effects) | set(channels):
        identifier(name)
    for name, channel in channels.items():
        require(set(channel["entities"]) <= entities, "unknown_channel_entity")
        require(not any(x in channel["array"].lower() for x in ["stage", "success", "event", "label", "timestamp"]),
                "controller_annotations_are_not_measurements")
    for entity in protocol["entities"]:
        if entity["role"] == "target":
            require(bool(entity["geometry"]), "target_functional_geometry_required")
        if "pose_channel" in entity:
            require(entity["pose_channel"] in channels and
                    channels[entity["pose_channel"]]["kind"] in {"pose_wxyz", "transform", "position"}, "invalid_entity_pose")
    for effector in effects.values():
        for key, allowed in [("grasp_channels", {"grasp"}), ("contact_channels", {"contact_impulse", "contact_force"})]:
            require(set(effector[key]) <= entities, "unknown_hand_entity")
            require(all(c in channels and channels[c]["kind"] in allowed for c in effector[key].values()), "invalid_hand_channel")
    for endpoint in ["initial_hands", "terminal_hands"]:
        require(set(protocol[endpoint]) == set(effects), "each_effector_needs_endpoint_state")
        for key, hand in protocol[endpoint].items():
            require((hand["state"] == "free" and hand["entity"] is None) or hand["entity"] in entities, "invalid_hand_state")
            if hand["state"] != "free":
                field = "grasp_channels" if hand["state"] == "holding" else "contact_channels"
                require(hand["entity"] in effects[key][field], "endpoint_hand_state_lacks_measurement")
    events = {e["id"]: e for e in protocol["events"]}
    require(len(events) == len(protocol["events"]) and not {"initial", "terminal"} & set(events), "duplicate_or_reserved_event_id")
    for a, b in protocol["precedence"]:
        require(a in events and b in events and a != b, "invalid_event_edge")
    parents = {eid: set() for eid in events}
    for a, b in protocol["precedence"]:
        parents[b].add(a)
    for event in events.values():
        require(set(event["strictly_after"]) <= parents[event["id"]], "strict_order_must_be_declared_edge")
    ordered = []
    while len(ordered) < len(events):
        available = [eid for eid in events if eid not in ordered and parents[eid] <= set(ordered)]
        require(bool(available), "cyclic_event_graph")
        ordered.extend(available)
    groups = [protocol["events"], protocol["initial_conditions"], protocol["goals"], protocol["constraints"]]
    for group in groups:
        require(len(group) == len({x["id"] for x in group}), "duplicate_condition_id")
        for item in group:
            identifier(item["id"])
            require(set(item["entities"]) <= entities and item.get("effector") in {None, *effects}, "unknown_event_or_condition_entity")
            _expression(item.get("when", item.get("predicate")), set(channels))
    for constraint in protocol["constraints"]:
        require(constraint["from"] in {"initial", "terminal", *events} and
                constraint["until"] in {"initial", "terminal", *events}, "unknown_interval_endpoint")
    targets = {e["id"] for e in protocol["entities"] if e["role"] == "target"}
    require(targets <= {eid for goal in protocol["goals"] for eid in goal["entities"]}, "target_has_no_declared_goal")
    unchanged = {e["id"] for e in protocol["entities"] if e["role"] == "unchanged"}
    preserved = {eid for c in protocol["constraints"] if c["relation"] in {"unchanged_pose", "joint_range"}
                 and c["from"] == "initial" and c["until"] == "terminal" and c["include_end"] for eid in c["entities"]}
    require(unchanged <= preserved, "unchanged_entity_lacks_full_interval_constraint")
    return ordered, parents


def load_channels(protocol, arrays):
    n = len(arrays["timestamps_seconds"])
    channels = {}
    for name, spec in protocol["channels"].items():
        require(spec["array"] in arrays, "missing_native_measurement:" + spec["array"])
        raw = arrays[spec["array"]]
        require(raw.ndim > 0 and raw.shape[0] == n, "measurement_timeline_mismatch:" + name)
        indices = [slice(None)]
        for axis, index in enumerate(spec["indices"], 1):
            require(axis < raw.ndim, "measurement_selector_rank:" + name)
            if isinstance(index, int):
                require(index < raw.shape[axis], "measurement_selector_bounds:" + name)
                indices.append(index)
            else:
                start, stop = index["slice"]
                require(0 <= start < stop <= raw.shape[axis], "measurement_selector_bounds:" + name)
                indices.append(slice(start, stop))
        values = raw[tuple(indices)]
        require(values.shape == (n, *CHANNEL_SHAPES[spec["kind"]]), "measurement_channel_shape:" + name)
        require(values.dtype.kind in "biuf" and np.isfinite(values).all(), "nonfinite_or_nonnumeric_measurement:" + name)
        if spec["kind"] == "grasp":
            require(values.dtype == np.bool_, "grasp_must_be_native_boolean")
        if spec["kind"] == "pose_wxyz":
            require(np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1, atol=2e-4, rtol=0), "invalid_measured_quaternion")
        channels[name] = values
    return channels


def _rotation(pose):
    return Rotation.from_quat(np.asarray(pose)[..., [4, 5, 6, 3]])


def evaluate(node, channels):
    op = node["op"]
    if op == "signal":
        return channels[node["name"]]
    if op == "constant":
        return np.asarray(node["value"])
    args = [evaluate(arg, channels) for arg in node["args"]]
    a = args[0]
    if op == "initial":
        require(a.ndim >= 1, "initial_requires_timeline")
        return a[0]
    if op == "component":
        require(a.ndim >= 1 and node["index"] < a.shape[-1], "component_out_of_bounds")
        return a[..., node["index"]]
    if op == "position":
        return a[..., :3, 3] if a.ndim >= 2 and a.shape[-2:] == (4, 4) else a[..., :3]
    if op == "xy":
        require(a.ndim >= 1 and a.shape[-1] >= 2, "xy_requires_vector")
        return a[..., :2]
    if op == "upright":
        return _rotation(a).as_matrix()[..., 2, 2]
    if op == "abs":
        return np.abs(a)
    if op == "norm":
        require(a.ndim >= 1, "norm_requires_vector")
        return np.linalg.norm(a, axis=-1)
    if op == "not":
        require(a.dtype == np.bool_, "logical_predicate_requires_booleans")
        return ~a
    if op in {"all", "any"}:
        require(all(x.dtype == np.bool_ for x in args), "logical_predicate_requires_booleans")
        values = np.broadcast_arrays(*args)
        return (np.logical_and if op == "all" else np.logical_or).reduce(values)
    b = args[1]
    if op == "relative_position":
        return _rotation(b).inv().apply(a[..., :3] - b[..., :3])
    if op == "relative_angle":
        return (_rotation(b).inv() * _rotation(a)).magnitude()
    return {"add": np.add, "sub": np.subtract, "mul": np.multiply,
            "lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal, "eq": np.equal}[op](a, b)


def _mask(expression, channels, n):
    mask = evaluate(expression, channels)
    require(mask.dtype == np.bool_ and mask.shape == (n,), "predicate_must_produce_one_boolean_per_measured_frame")
    return mask


def _check(name, mask, start, end):
    require(0 <= start < end <= len(mask), "empty_or_reversed_constraint_interval:" + name)
    failures = np.flatnonzero(~mask[start:end])
    return {"id": name, "passed": not len(failures), "frame_start": start, "frame_end_inclusive": end - 1,
            "measured_frame_count": end - start, "first_failure_frame": int(start + failures[0]) if len(failures) else None}


def compile_events(protocol, arrays):
    ordered, parents = validate(protocol)
    times = arrays["timestamps_seconds"]
    require(times.ndim == 1 and len(times) >= 2 and np.isfinite(times).all() and (np.diff(times) > 0).all(), "invalid_native_clock")
    n = len(times)
    require(protocol["terminal_window_frames"] <= n, "terminal_window_exceeds_execution")
    channels = load_channels(protocol, arrays)
    declarations = {e["id"]: e for e in protocol["events"]}
    observed, frames = [], {"initial": 0, "terminal": n - 1}
    for eid in ordered:
        event = declarations[eid]
        start = max([frames[p] + int(p in event["strictly_after"]) for p in parents[eid]] or [0])
        mask = _mask(event["when"], channels, n)
        width = event["min_frames"]
        require(width <= n, "event_window_exceeds_execution")
        runs = np.convolve(mask.astype(int), np.ones(width, dtype=int), "valid") == width
        hits = np.flatnonzero(runs & (np.arange(len(runs)) >= start))
        if event["selection"] == "terminal":
            require(width == 1 and n - 1 in hits, "missing_terminal_event:" + eid)
            frame = n - 1
        else:
            require(len(hits) > 0, "missing_measured_event:" + eid)
            frame = int(hits[0])
        frames[eid] = frame
        observed.append({"id": eid, "phase": event["action"], "entities": event["entities"],
                         "effector": event["effector"], "frame": frame, "seconds": float(times[frame]),
                         "predicate": event["when"], "evidence_kind": "recomputed_native_measurements"})
    checks = []
    for kind, start, end in [("initial_conditions", 0, 1), ("goals", n - protocol["terminal_window_frames"], n)]:
        for condition in protocol[kind]:
            checks.append(_check(kind + ":" + condition["id"], _mask(condition["predicate"], channels, n), start, end))
    for condition in protocol["constraints"]:
        start, end = frames[condition["from"]], frames[condition["until"]] + int(condition["include_end"])
        checks.append(_check("constraint:" + condition["id"], _mask(condition["predicate"], channels, n), start, end))
    for endpoint, start, end in [("initial_hands", 0, 1), ("terminal_hands", n - protocol["terminal_window_frames"], n)]:
        for effector, hand in protocol[endpoint].items():
            spec = protocol["effectors"][effector]
            if hand["state"] == "free":
                evidence = list(spec["grasp_channels"].values())
                require(evidence, "free_hand_lacks_contact_measurements")
                mask = ~np.any([channels[c] for c in evidence], axis=0)
            elif hand["state"] == "holding":
                mask = channels[spec["grasp_channels"][hand["entity"]]]
            else:
                mask = channels[spec["contact_channels"][hand["entity"]]] > 1e-6
            checks.append(_check(endpoint + ":" + effector, mask, start, end))
    return {"events": observed, "required_event_precedence": protocol["precedence"],
            "qualification": {"passed": all(c["passed"] for c in checks), "checks": checks,
                              "scope": "declared predicates on every native measured frame; not an all-path proof"}}


def public_condition(condition):
    return {k: condition[k] for k in ["id", "relation", "entities", "parameters", "effector",
                                      "from", "until", "include_end", "human_visible_check"] if k in condition}
