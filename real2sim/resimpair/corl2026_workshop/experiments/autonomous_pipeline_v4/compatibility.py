"""Read-only regression fixtures from retained native executions.

This module never registers a new execution or calls a provider. Historical
source annotations and human reviews are deliberately not consumed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from .core import bind, checked, implementation, read, require, write
from .event_protocol import compile_events
from .generic_compiler import generation_information
from . import planning_protocols as builders
from .prompts import anchor_prompt, video_prompt
from .review import anchor_checks, video_checks

HISTORICAL = Path("/mnt/sdg/datasets/real_sim_human_pipeline/five_task_probe_20261022_v1/human_generation_v3")
HISTORICAL_CASES = ["02_cabinet_open", "03_domino_cascade", "05_bimanual_insert"]


def load_current(bundle_path):
    bundle = read(bundle_path)
    with np.load(checked(bundle["measurements"]), allow_pickle=False) as z:
        arrays = dict(z)
    protocol = builders.pick_place(bundle["scene"], bundle["task_intent"], bundle["object_ids"],
                                    bundle["support_ids"], bundle["target_positions_world"])
    return protocol, arrays, bundle["cameras"]["opposite_high"], {"original_execution": bind(bundle_path),
                                                               "measurements": bundle["measurements"]}


def load_historical(case_id):
    require(case_id in HISTORICAL_CASES, "Unknown retained regression fixture")
    original = HISTORICAL / case_id / "task_contract.json"
    inputs = read(original)["input_bindings"]
    receipt = read(checked(inputs["receipt"]))
    require(bind(checked(inputs["arrays"]))["sha256"] == receipt["sha256"]["arrays.npz"], "Historical measurements changed")
    with np.load(checked(inputs["arrays"]), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "stage"}
    bindings = {"historical_locator": bind(original), "native_receipt": inputs["receipt"], "measurements": inputs["arrays"]}
    if "articulation" in inputs:
        with np.load(checked(inputs["articulation"]), allow_pickle=False) as z:
            arrays.update({k: z[k] for k in z.files if k != "stage"})
        bindings["articulation"] = inputs["articulation"]
    ids = [asset["name"] for asset in receipt["asset_specs"]]
    geometry_keys = {"primitive", "half_size_m", "outer_half_m", "socket_half_xy_m", "socket_floor_z_m",
                     "ports", "body_half_m", "tongue_half_m", "tongue_center_local", "bounds_rotated_m"}
    entities = [{"id": a["name"], "description": a["name"].replace("_", " "), "kind": a.get("primitive", "rigid_object"),
                 "role": "target", "geometry": {k: v for k, v in a.items() if k in geometry_keys},
                 "pose_channel": a["name"] + ".pose"} for a in receipt["asset_specs"]]
    if case_id == "02_cabinet_open":
        for entity in entities:
            entity["role"] = "unchanged"
        entities += [{"id": f"door_{j}", "description": f"hinged cabinet door {j}", "kind": "articulated_link",
                      "role": "target" if j == 1 else "unchanged", "geometry": {"joint_type": "revolute"},
                      "pose_channel": f"door_{j}.pose"} for j in range(2)]
        protocol = builders.articulation(case_id, entities, ids, ["door_0", "door_1"], 1, receipt["active_arm"], [1.2, 1.4])
    elif case_id == "05_bimanual_insert":
        receiver, plug = receipt["asset_specs"]
        tip = plug["tongue_center_local"][2] - plug["tongue_half_m"][2]
        protocol = builders.insertion(case_id, entities, ids, arrays["port_centers_local"][0].tolist(),
                                      receiver["outer_half_m"][2], tip, [.020, .028], .003)
    else:
        for entity in entities[5:]:
            entity["role"] = "unchanged"
        protocol = builders.passive_chain(case_id, entities, ids, ids[:5])
    old_camera = inputs["view"]["camera"]
    camera = {"config": {"width": old_camera["width"], "height": old_camera["height"]},
              "parameters": {"intrinsic_cv": old_camera["intrinsic_matrix"], "extrinsic_cv": old_camera["camera_from_world"]}}
    return protocol, arrays, camera, bindings


def _rejected(protocol, arrays):
    try:
        result = compile_events(protocol, arrays)
        return {"rejected": not result["qualification"]["passed"],
                "failed_checks": [c for c in result["qualification"]["checks"] if not c["passed"]]}
    except (ValueError, KeyError) as exc:
        return {"rejected": True, "reason": str(exc)}


def counterexamples(fixtures):
    cases = []
    plan, arrays, _, _ = fixtures["scene_A__pack"]
    compiled = compile_events(plan, arrays)
    first = {e["id"]: e["frame"] for e in compiled["events"]}
    changed = {k: v.copy() for k, v in arrays.items()}
    frame = (first["transfer_0.lift"] + first["transfer_0.arrive"]) // 2
    changed["grasp"][frame, 0, 0] = False
    cases.append({"case": "transport_grasp_gap", "mutated_frame": frame, **_rejected(plan, changed)})
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["object_states"][100, 1, 0] += .05
    cases.append({"case": "non_target_moves_midway", "mutated_frame": 100, **_rejected(plan, changed)})
    plan, arrays, _, _ = fixtures["05_bimanual_insert"]
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["grasp"][600, 0, 0] = False
    cases.append({"case": "support_hand_gap_inside_402_frame_interval", "mutated_frame": 600, **_rejected(plan, changed)})
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["object_states"][-10:, 1, 0] += .03
    cases.append({"case": "inserted_plug_moves_out_of_socket", **_rejected(plan, changed)})
    changed_plan = deepcopy(plan)
    changed_plan["terminal_hands"]["arm0"] = {"state": "free", "entity": None}
    cases.append({"case": "incorrect_both_hands_free_endpoint", **_rejected(changed_plan, arrays)})
    plan, arrays, _, _ = fixtures["03_domino_cascade"]
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["robot_contact_impulses"][215, 0, 2] = .1
    cases.append({"case": "hand_pushes_passive_chain_member", **_rejected(plan, changed)})
    plan, arrays, _, _ = fixtures["02_cabinet_open"]
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["cabinet_qpos"][300, 0] = .25
    cases.append({"case": "non_target_door_opens", **_rejected(plan, changed)})
    require(all(case["rejected"] for case in cases), "A physical counterexample escaped the generic interpreter")
    return cases


def replay(root, output):
    """Write an immutable audit; no paid calls, no simulated env.step."""
    root, output = Path(root), Path(output)
    fixtures = {}
    for handoff in sorted((root / "planner_handoffs").glob("*.json")):
        original = checked(read(handoff)["execution_bundle"])
        # The legacy bridge is specifically for previously completed v4 exports.
        if read(original)["schema"] == "resimpair_execution_bundle_v4":
            fixtures[handoff.stem] = load_current(original)
    fixtures.update({case: load_historical(case) for case in HISTORICAL_CASES})
    rows = []
    for case, (protocol, arrays, camera, bindings) in fixtures.items():
        compiled = compile_events(protocol, arrays)
        require(compiled["qualification"]["passed"], "Retained native fixture fails declared protocol:" + case)
        info, initial = generation_information(protocol, compiled, arrays, camera)
        contract = {"schema": "resimpair_protocol_replay_preview_v1", "generation_information": info,
                    "initial_scene_information": initial, "measured": compiled}
        directory = output / case
        write(directory / "protocol.json", protocol)
        write(directory / "measured.json", compiled)
        write(directory / "downstream_preview.json", {
            "contract": contract, "anchor_prompt": anchor_prompt(initial), "video_prompt": video_prompt(contract),
            "anchor_checks": anchor_checks(initial), "video_checks": video_checks(contract),
            "submitted_to_provider": False, "historical_source_annotations_consumed": False})
        rows.append({"case": case, "inputs": bindings, "protocol": bind(directory / "protocol.json"),
                     "measured": bind(directory / "measured.json"), "downstream_preview": bind(directory / "downstream_preview.json"),
                     "frames": len(arrays["timestamps_seconds"]), "events": len(compiled["events"]),
                     "passed_declared_predicates": True})
    result = {"schema": "resimpair_protocol_compatibility_audit_v1", "cases": rows,
              "counterexamples": counterexamples(fixtures), "implementation": implementation(),
              "physical_executions_started": 0, "provider_calls_started": 0,
              "historical_human_videos_rejudged": 0, "historical_manual_source_observations_used": False,
              "scope": "retrospective measurement compatibility and prompt preparation only; not new autonomous cross-mechanism generation"}
    write(output / "audit.json", result)
    return {"audit": str(output / "audit.json"), "compatible_native_executions": len(rows),
            "counterexamples_rejected": len(result["counterexamples"]), "new_provider_calls": 0, "new_simulations": 0}
