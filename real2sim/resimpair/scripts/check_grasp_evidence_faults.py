#!/usr/bin/env python3
"""Replay actual successful measurements with explicit evidence faults, CPU only."""
from pathlib import Path
import argparse
import copy
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resimpair_runtime import configure
configure()
from resimpair_grasp.common import binding, read, write
from resimpair_grasp.evaluator import geometry, bindings
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import checked, bind
from corl2026_workshop.experiments.autonomous_pipeline_v4.event_protocol import compile_events
from corl2026_workshop.experiments.autonomous_pipeline_v4.compiler import verify_native_kinematics, completed_contract
from corl2026_workshop.experiments.autonomous_pipeline_v4.generic_compiler import compile_bundle


def replay(directory, output):
    bundle = read(directory / "execution_bundle.json")
    protocol = read(checked(bundle["event_protocol"]))
    with np.load(checked(bundle["measurements"]), allow_pickle=False) as f:
        original = {k: f[k] for k in f.files}
    assert compile_events(protocol, original)["qualification"]["passed"]
    assert verify_native_kinematics(original)["passed"]
    assert geometry(directory / "native_geometry.json", bundle["scene"])["passed"]
    results = []
    for name in ["wrong_final_position", "missing_grasp", "robot_collision", "distractor_moved", "tcp_frame_error", "invalid_joint_command"]:
        a = {k: v.copy() for k, v in original.items()}
        if name == "wrong_final_position": a["object_states"][-20:, 0, 0] += .05
        if name == "missing_grasp": a["grasp"][:] = False
        if name == "robot_collision": a["cross_robot_force_N"][10] = 10.
        if name == "distractor_moved": a["object_states"][10, 1, 0] += .02
        if name == "tcp_frame_error": a["tcp_world"][:, 0, 0, 3] += .03
        if name == "invalid_joint_command": a["native_commands"][0, 0] = 8.
        try:
            result = verify_native_kinematics(a) if name in {"tcp_frame_error", "invalid_joint_command"} else compile_events(protocol, a)["qualification"]
            detected = not result["passed"]; reason = [c["id"] for c in result.get("checks", []) if not c["passed"]]
        except ValueError as exc:
            detected = True; reason = str(exc)
        results.append({"fault": name, "detected": detected, "reason": reason})
    native = copy.deepcopy(read(directory / "native_geometry.json"))
    native["objects"][0]["shapes"][0]["vertices"][0][0] += .02
    altered = output / "deliberately_changed_collider.json"; write(altered, native)
    try:
        geometry(altered, bundle["scene"]); detected = False
    except ValueError:
        detected = True
    results.append({"fault": "native_collider_changed", "detected": detected})
    # Invoke the retained completed-contract/media verifier on a deliberately
    # inconsistent camera packet, instead of treating a custom length check as
    # evidence that the actual downstream gate catches it.
    contract = compile_bundle(directory / "execution_bundle.json", {"mode": "synthetic_only", "sources": []})
    packets = {name: bind(directory / "audit_frames" / name / "packet.json") for name in bundle["cameras"]}
    positive = output / "unmodified_media"
    write(positive / "contract.json", contract)
    write(positive / "media_validation.json", {"contract": bind(positive / "contract.json"),
        "cameras": packets, "both_fully_decoded": True})
    completed_contract(positive / "contract.json")
    folder = output / "deliberately_changed_media"
    write(folder / "contract.json", contract)
    packet = read(checked(packets["opposite_high"]))
    packet["decoded_frames"] -= 1; packet["all_native_pts_seconds"].pop()
    write(folder / "incomplete_packet.json", packet)
    packets["opposite_high"] = bind(folder / "incomplete_packet.json")
    write(folder / "media_validation.json", {"contract": bind(folder / "contract.json"), "cameras": packets, "both_fully_decoded": True})
    try:
        completed_contract(folder / "contract.json"); detected = False
    except ValueError:
        detected = True
    results.append({"fault": "one_camera_missing_frame_in_packet", "detected": detected, "detector": "retained completed_contract"})
    report = {"source_bundle": binding(directory / "execution_bundle.json"), "rule_bindings": bindings(),
        "unmodified_positive_passed": True, "unmodified_media_passed": True,
        "faults": results, "all_detected": all(x["detected"] for x in results),
        "scope": "counterfactual faults in actual measurements; not new physical executions or production samples", "api_calls": 0}
    write(output / "report.json", report)
    assert report["all_detected"]
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--execution", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); print(json.dumps(replay(a.execution, a.output), indent=2))
