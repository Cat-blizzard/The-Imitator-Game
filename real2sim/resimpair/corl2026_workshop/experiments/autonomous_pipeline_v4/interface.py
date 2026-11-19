"""The data required BEFORE planning; missing fields are errors, not requests for chat annotations."""
from __future__ import annotations

ARTIFACT = {"type": "object", "additionalProperties": False,
            "properties": {"path": {"type": "string", "minLength": 1},
                           "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
            "required": ["path", "sha256"]}

MEASUREMENTS = {
    "timestamps_seconds": {"shape": ["frames"], "units": "seconds", "origin": "native control clock"},
    "states": {"shape": ["frames", 18], "units": "radians or meters", "origin": "measured qpos"},
    "native_commands": {"shape": ["frames-1", 16], "units": "joint target and gripper command", "origin": "env.step inputs"},
    "tcp_world": {"shape": ["frames", 2, 4, 4], "units": "meters, homogeneous matrix", "origin": "measured link poses"},
    "object_states": {"shape": ["frames", "objects", 13], "units": "xyz m, wxyz quaternion, linear m/s, angular rad/s", "origin": "measured actor state"},
    "grasp": {"shape": ["frames", 2, "objects"], "units": "boolean", "origin": "native gripper contact predicate"},
    "support_impulses": {"shape": ["frames", "objects", "supports"], "units": "N s", "origin": "contacts accumulated within preceding control step"},
    "robot_object_force_N": {"shape": ["frames", 2, "objects"], "units": "N", "origin": "max native substep contact impulse / substep duration"},
    "cross_robot_force_N": {"shape": ["frames"], "units": "N", "origin": "max native substep contact impulse / substep duration"},
}

REQUIRED = {
    "schema": {"const": "resimpair_execution_bundle_v4"},
    "case": {"type": "object", "required": ["id", "scene", "task", "semantic_key", "partition"]},
    "task_intent": {"type": "object", "required": ["id", "family", "required_skills", "operations", "required_operation_precedence",
                                                    "equivalence_groups", "human_hand_policy", "non_target_policy"]},
    "scene": {"type": "object", "required": ["id", "seed", "objects", "supports", "cameras", "appearance", "robot_roots"]},
    "measurements": ARTIFACT, "planned_paths": ARTIFACT,
    "object_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}},
    "support_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string"}},
    "target_positions_world": {"type": "object", "minProperties": 1},
    "cameras": {"type": "object", "required": ["robot_side", "opposite_high"]},
    "native_execution": {"const": True}, "physical_execution_count": {"const": 1},
    "motion_planning_agent_participation": {"type": "boolean"},
    "manual_generation_semantics_supplied": {"const": False},
    "manual_review_observations_supplied": {"const": False},
    "planning_failure": {"type": ["object", "null"]}, "code": {"type": "object", "minProperties": 1},
}
BUNDLE_SCHEMA = {"type": "object", "properties": REQUIRED, "required": list(REQUIRED)}


def generic_bundle_schema():
    from copy import deepcopy
    schema = deepcopy(BUNDLE_SCHEMA)
    schema["properties"]["schema"] = {"const": "resimpair_execution_bundle_v5"}
    schema["properties"].update(event_protocol=ARTIFACT, planning_declaration=ARTIFACT)
    schema["required"] += ["event_protocol", "planning_declaration"]
    for name in ["object_ids", "support_ids", "target_positions_world"]:
        schema["required"].remove(name)
    return schema

CONSUMERS = {
    "source_model": ["entities", "critical_geometry", "initial_relations", "final_relations", "events", "observed_order",
                     "hand_roles", "hidden_state_observations", "window_assessment", "visible_goal", "transferable_skills", "uncertainties"],
    "planner": ["source transferable skill evidence", "declared task intent", "scene inventory and support geometry"],
    "compiler": ["task intent", "object/support IDs", "all native measurement fields", "two synchronized videos", "camera matrices"],
    "image_model": ["opposite_high initial RGB", "initial inventory and screen grounding", "one shared actor policy"],
    "video_model": ["approved initial human pixels", "goal relations", "persistent object descriptions", "necessary event graph",
                    "non-target invariants", "human hand policy", "duration and pacing"],
    "review_model": ["candidate pixels at explicit native frame indices", "same goal/object/event contract", "fixed per-check schema"],
    "terminal_diagnosis": ["failed/unknown checks", "source and request bindings", "model identity", "actual usage"],
}


def validate_bundle(bundle, arrays=None):
    from jsonschema import validate
    from .core import require
    if bundle.get("schema") == "resimpair_execution_bundle_v5":
        validate(bundle, generic_bundle_schema())
    else:
        validate(bundle, BUNDLE_SCHEMA)
    if arrays is not None:
        import numpy as np
        required = set(MEASUREMENTS) if bundle["schema"].endswith("v4") else {
            "timestamps_seconds", "states", "native_commands", "tcp_world", "robot_roots_world"}
        require(required <= set(arrays), "Planner omitted required measured fields: " + ",".join(sorted(required - set(arrays))))
        n = len(arrays["timestamps_seconds"])
        require(2 <= n <= 4001 and arrays["timestamps_seconds"].shape == (n,), "Invalid native timeline")
        require(np.all(np.diff(arrays["timestamps_seconds"]) > 0), "Nonmonotone native timeline")
        for key, shape in {"states": (n, 18), "native_commands": (n-1, 16), "tcp_world": (n, 2, 4, 4),
                           "robot_roots_world": (2, 4, 4)}.items():
            require(key in arrays and arrays[key].shape == shape and np.isfinite(arrays[key]).all(),
                    "Invalid native dimensions or values:" + key)
    require(set(bundle["cameras"]) == {"robot_side", "opposite_high"}, "Two fixed synchronized views required")
    for camera in bundle["cameras"].values():
        require(set(["video", "initial_image", "final_image", "parameters", "config", "fps", "frame_count", "synchronous_with_measurements"]) <= set(camera),
                "Camera export is incomplete")
        require(set(["extrinsic_cv", "cam2world_gl", "intrinsic_cv"]) <= set(camera["parameters"]), "Actual camera calibration required")


def publish(root):
    from pathlib import Path
    from .core import write, implementation
    output = Path(root) / "interfaces" / "v5"
    write(output / "planner_bundle.schema.json", generic_bundle_schema())
    write(output / "information_flow.json", {"measurements": MEASUREMENTS, "consumers": CONSUMERS,
          "missing_information_policy": "stop_with_typed_error_and_improve_shared_upstream_interface_in_next_release",
          "manual_semantic_patch_interface": None})
    from .event_protocol import SCHEMA
    write(output / "event_protocol.schema.json", SCHEMA)
    return str(output)
