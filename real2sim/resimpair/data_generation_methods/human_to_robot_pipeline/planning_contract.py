"""Human-grounded task contracts and a bounded interface for motion programs.

CPU-only. A valid contract describes what must be preserved; it is not evidence
that a trajectory satisfies it. The existing collectors remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol


WORKSPACE = Path(__file__).resolve().parents[2]
FACTORY = WORKSPACE / "PolicyGen/examples/baselines/hypericl_data_factory"
FROZEN_SPEC = Path("/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/generation/prepared_v1/human_grounded_task_spec.json")
TASK = "apple_bowl_transfer"
HARD_RULES = {
    "initial_apple_inside_source": True,
    "unique_initially_empty_destination": True,
    "destination_is_same_instance_until_end": True,
    "recording_object_pose_writes_forbidden": True,
    "recording_object_teleport_forbidden": True,
    "physical_grasp_release_support_required": True,
    "unselected_objects_untouched_required": True,
    "all_added_physical_objects_in_planner_world": True,
    "controller_phase_is_not_contact_ground_truth": True,
    "unknown_evidence_does_not_pass": True,
    "success_thresholds_may_not_be_relaxed": True,
}
FAILURE_CODES = [
    "human_task_incomplete", "ambiguous_role", "initial_relation_invalid",
    "asset_geometry_invalid", "initial_collision", "unreachable_goal",
    "planning_failed", "grasp_failed", "unexpected_contact", "unstable_release",
    "key_event_occluded", "timeline_invalid", "missing_evidence",
    "unsupported_randomization", "budget_exhausted",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def bind(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_binding(value):
    if not isinstance(value, dict) or not {"path", "sha256"} <= set(value):
        raise ValueError("Missing file binding")
    if sha(value["path"]) != value["sha256"]:
        raise ValueError("Bound file changed: " + value["path"])


def build_apple_contract(spec_path=FROZEN_SPEC):
    source = json.loads(Path(spec_path).read_text())
    task = next(t for t in source["tasks"] if t["id"] == TASK)
    stages = [
        ("approach", "selected_object", "grasp_pose plus local upward clearance",
         ["initial_apple_inside_source"], ["approach_collision_free"], ["approach"]),
        ("grasp", "selected_object", "collision-mesh grasp frame",
         ["approach_collision_free"], ["actual_grasp_established"], ["grasp_reach", "close"]),
        ("lift", "source_container", "above measured source rim",
         ["actual_grasp_established"], ["object_clear_of_source"], ["lift"]),
        ("transport", "destination_container", "above measured destination opening",
         ["object_clear_of_source"], ["object_above_destination"], ["above_bowl"]),
        ("release", "destination_container", "inside opening with measured rim clearance",
         ["object_above_destination"], ["object_released"], ["release_above_rim", "open"]),
        ("withdraw_and_settle", "destination_container", "retreat outside contact region",
         ["object_released"], ["supported_stable_inside", "unselected_untouched"],
         ["retreat", "return", "final_settle"]),
    ]
    return {
        "schema": "human_grounded_motion_contract_v1",
        "task_id": TASK, "human_spec": bind(spec_path),
        "human_windows": task["human_windows"],
        "semantic_scope": task["semantic_pair_type"],
        "initial_relation": task["initial"], "required_order": task["required_steps"],
        "final_relation": task["final"], "forbidden_changes": task["forbidden"],
        "roles": {
            "selected_object": {"category": "apple", "count": 1},
            "source_container": {"category": "bowl", "count": 1, "initially_occupied": True},
            "destination_container": {"category": "bowl", "count": 1, "initially_empty": True},
            "distractor": {"must_not_match": ["apple", "bowl"], "must_remain_untouched": True},
        },
        "coordinate_convention": {
            "length": "metres", "angle": "radians", "simulation_frame": "world",
            "goals": "object-relative frames resolved from current scene geometry",
            "quaternion": "wxyz only when native adapter requests a quaternion",
            "saved_actions": "existing query-local EEF14 at20Hz; no normalization change",
        },
        "stages": [{"id": name, "reference_role": role, "target_rule": target,
                    "preconditions": pre, "postconditions": post,
                    "existing_controller_phases": phases}
                   for name, role, target, pre, post, phases in stages],
        "hard_rules": HARD_RULES.copy(),
        "expert": bind(FACTORY / "real_bowl.py"),
        "independent_verifier": bind(FACTORY / "verify_bowl.py"),
        "verifier_entry": "examples.baselines.hypericl_data_factory.verify_bowl.verify",
        "native_adapter": "qualified_apple_bowl_v1",
        "native_capabilities": {
            "explicit_layout_override": False, "extra_physical_distractors": False,
            "existing_seed_randomization": True, "native_collision_replanning": True,
            "object_pose_and_contact_recording": True,
        },
        "required_evidence": [
            "source video hash and exact human task window", "seed and split",
            "asset hashes and collision geometry", "all attempted case outcomes",
            "per-frame states, native commands and EEF conversion",
            "actual grasp and contact records", "independent raw verifier",
            "actual key-event video visibility", "split and duplicate audit",
        ],
        "failure_codes": FAILURE_CODES,
        "proposal_is_training_approval": False,
    }


def validate_contract(contract, verify_files=True):
    if contract.get("schema") != "human_grounded_motion_contract_v1" or contract.get("task_id") != TASK:
        raise ValueError("Only the reviewed apple transfer contract is implemented")
    if contract.get("hard_rules") != HARD_RULES:
        raise ValueError("Hard rules changed or removed")
    if contract.get("proposal_is_training_approval") is not False:
        raise ValueError("Contract cannot approve training")
    if contract.get("expert", {}).get("path") == contract.get("independent_verifier", {}).get("path"):
        raise ValueError("Planner cannot be its own verifier")
    roles = contract.get("roles", {})
    for role in ("selected_object", "source_container", "destination_container"):
        if roles.get(role, {}).get("count") != 1:
            raise ValueError("Required semantic role is not unique")
    stages = contract.get("stages", [])
    if len(stages) != 6 or len({s.get("id") for s in stages}) != len(stages):
        raise ValueError("Missing or duplicated stage")
    for stage in stages:
        if (stage.get("reference_role") not in roles or not stage.get("target_rule")
                or not stage.get("preconditions") or not stage.get("postconditions")):
            raise ValueError("Every stage needs an object-relative goal and conditions")
    if not contract.get("human_windows"):
        raise ValueError("A task must be grounded in a complete human window")
    for window in contract["human_windows"]:
        a, b = window.get("window_frames", [None, None])
        if type(a) is not int or type(b) is not int or not 0 <= a < b:
            raise ValueError("Invalid human interval")
        if window.get("interval") != "start_inclusive_end_exclusive":
            raise ValueError("Ambiguous interval convention")
        if not all(window.get(k) for k in ("initial", "steps", "final", "review", "fps", "split")):
            raise ValueError("Incomplete human task provenance")
    if verify_files:
        for key in ("human_spec", "expert", "independent_verifier"):
            check_binding(contract[key])
        frozen = build_apple_contract(contract["human_spec"]["path"])
        # A VLM may propose edits separately; it cannot silently redefine the task.
        for key in ("roles", "initial_relation", "required_order", "final_relation",
                    "forbidden_changes", "human_windows", "stages", "native_capabilities"):
            if contract.get(key) != frozen[key]:
                raise ValueError("Contract differs from reviewed source: " + key)
        for window in contract["human_windows"]:
            check_binding(window["video"])
            check_binding(window["review"])
    return {"valid": True, "training_admitted": False,
            "scope": "Contract consistency, not trajectory success"}


class MotionProgram(Protocol):
    """Interface for a future expanded adapter; no simulator required to import."""
    def initialize(self, case: dict) -> dict:
        """Create scene once, settle; return measured role poses and asset IDs."""
    def plan_stage(self, stage: dict, measured_scene: dict) -> dict:
        """Resolve object-relative target; include all collider/attached meshes."""
    def execute_stage(self, plan: dict, step_budget: int) -> dict:
        """Only robot control; record raw state/contact/RGB/clock, return evidence."""
    def independent_verify(self, receipt_path: str) -> dict:
        """Call a separate immutable verifier; retain failures and unknowns."""


def native_adapter_check(case, contract):
    """Prevent a sampled proposal from silently falling back to the old layout."""
    validate_contract(contract, verify_files=False)
    blockers = []
    if case.get("layout_mode") != "existing_seed_only":
        blockers.append("explicit_layout_override_not_implemented")
    if case.get("added_distractors"):
        blockers.append("extra_distractor_collision_and_recording_not_implemented")
    if case.get("camera_mode", "existing_seed_only") != "existing_seed_only":
        blockers.append("explicit_camera_override_not_implemented")
    return {"adapter": contract["native_adapter"], "executable_by_current_collector": not blockers,
            "blockers": blockers, "simulation_executed": False,
            "training_admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=FROZEN_SPEC)
    args = parser.parse_args()
    contract = build_apple_contract(args.spec)
    validate_contract(contract)
    save_new(args.output, contract)
    print(json.dumps({"contract": str(args.output), "validated": True, "simulation_executed": False}))


if __name__ == "__main__":
    main()
