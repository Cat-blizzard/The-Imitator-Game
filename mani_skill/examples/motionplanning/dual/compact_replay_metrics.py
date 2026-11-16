"""Validate variable-length compact recordings without changing frozen baselines.

Actions are T by 16, actor states T+1 by 13, and Panda states T+1 by 31.
All native flags, actual commands, phase receipts and source hashes must agree.
The original pose, physical release and common 20-tick placement gates apply.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .replay_metrics import ACTION_KEYS, pose_errors
from .sync_replay_metrics import LEAVES, _array, _flag, _require, _rows, summarize_objects
from .task_targets import TASK
from .replay_support import ACTION_LIMITS


def load_compact_recording(directory: Path) -> dict[str, Any]:
    """Read a passed compact episode, checking original evidence in place."""
    directory = Path(directory).resolve()
    report = json.loads((directory / "report.json").read_text())
    metadata = json.loads((directory / "trajectory.json").read_text())
    ticks = [json.loads(line) for line in (directory / "ticks.jsonl").read_text().splitlines()]
    episodes, info = metadata.get("episodes", []), metadata.get("env_info", {})
    _require(len(episodes) == 1, "Expected one episode")
    episode, kwargs = episodes[0], info.get("env_kwargs", {})
    budget = report.get("max_episode_steps")
    _require(type(budget) is int and budget in (200, 400), "Expected explicit official or timing-diagnostic budget")
    _require(info.get("max_episode_steps") == budget and kwargs.get("max_episode_steps", budget) == budget,
             "Report and native TimeLimit metadata disagree")
    _require(report.get("status") == "passed" and report.get("mode") == "positive", "Passed positive recording required")
    _require(report.get("task") == info.get("env_id") == TASK, "Task identity mismatch")
    _require(kwargs.get("control_mode") == "pd_joint_pos" and kwargs.get("sim_backend") == "cpu"
             and report.get("backend", {}).get("physics") == "physx_cpu", "Control/physics configuration mismatch")
    _require(report.get("control_timestep_s") == .05, "Expected unchanged 20Hz control")
    _require(report.get("seed") == episode.get("episode_seed") == episode.get("reset_kwargs", {}).get("seed"),
             "Seed receipt mismatch")
    modes = episode.get("control_mode", {})
    _require(list(modes) == ACTION_KEYS and all(mode == "pd_joint_pos" for mode in modes.values()), "Action mapping mismatch")
    _require(report.get("recording", {}).get("action_key_order") == ACTION_KEYS, "Recording action mapping mismatch")
    with h5py.File(directory / "trajectory.h5") as handle:
        key = f"traj_{episode['episode_id']}"
        _require(list(handle) == [key], "Unexpected H5 episodes")
        group = handle[key]
        actions = group["actions"][:]
        count = len(actions)
        _require(actions.shape == (count, 16) and actions.dtype.kind == "f" and np.isfinite(actions).all()
                 and 0 < count <= budget, "Invalid action array or budget")
        _require(count == len(ticks) == report.get("total_environment_steps") == episode.get("elapsed_steps")
                 == report.get("recording", {}).get("recorded_steps"), "Step receipts disagree")
        arrays: dict[str, dict[str, np.ndarray]] = {}
        _require(set(group["env_states"]) == set(LEAVES), "State category mismatch")
        for category, names in LEAVES.items():
            _require(set(group["env_states"][category]) == names, "Seven-entity state identity mismatch")
            arrays[category] = {}
            for name in names:
                array = group[f"env_states/{category}/{name}"][:]
                width = 13 if category == "actors" else 31
                _require(array.shape == (count + 1, width) and array.dtype.kind == "f" and np.isfinite(array).all(),
                         "Invalid state shape or values")
                _require(np.all(np.abs(np.linalg.norm(array[:, 3:7], axis=1) - 1) <= 1e-3), "Nonunit state quaternion")
                arrays[category][name] = array
        flags = {name: group[name][:] for name in ("success", "terminated", "truncated")}
    _require(all(flag.shape == (count,) and flag.dtype.kind == "b" for flag in flags.values()), "Native flag shape/type mismatch")
    _require(not flags["truncated"].any() and not np.any(flags["terminated"] & ~flags["success"]),
             "Input contains timeout or non-success termination")
    states = [{category: {name: array[i].copy() for name, array in values.items()} for category, values in arrays.items()}
              for i in range(count + 1)]
    segments: list[str] = []
    counts: Counter[str] = Counter()
    for index, tick in enumerate(ticks):
        _require(tick.get("environment_step") == index + 1 and not _flag(tick.get("fault_injected"), "fault"),
                 "Nonsequential or fault-injected positive tick")
        phase = tick["phase"]
        counts[phase] += 1
        if not segments or segments[-1] != phase:
            segments.append(phase)
        for field in ("actions", "requested_actions"):
            _require(list(tick[field]) == ACTION_KEYS, "Tick action mapping mismatch")
            actual = np.concatenate([_array(tick[field][key], (8,), field) for key in ACTION_KEYS]).astype(actions.dtype)
            _require(np.array_equal(actual, actions[index]), "Tick command differs from H5 or positive command overridden")
        for flag, value in (("success", tick["official_evaluation"]["success"]),
                            ("terminated", tick["native_terminated"]), ("truncated", tick["native_truncated"])):
            _require(_flag(value, flag) == bool(flags[flag][index]), "H5 and tick native flags disagree")
        _require(_flag(tick["success_only_termination_continued"], "continued")
                 == bool(flags["success"][index] and flags["terminated"][index]), "Continuation receipt mismatch")
        for side, name in enumerate(("apple_out", "banana_out")):
            obj = tick["objects"][side]
            _require(obj["actor"] == name, "Physical object pairing changed")
            actor = states[index + 1]["actors"][name]
            measured = np.r_[_array(obj["pose"], (7,), "pose"), _array(obj["linear_velocity_m_s"], (3,), "velocity"),
                             _array(obj["angular_velocity_rad_s"], (3,), "angular velocity")]
            _require(np.array_equal(measured.astype(actor.dtype), actor), "Target telemetry differs from H5")
            robot = states[index + 1]["articulations"][f"panda_wristcam-agent-{side}"]
            _require(np.array_equal(np.asarray(tick[("left_qpos", "right_qpos")[side]], dtype=robot.dtype), robot[13:22]),
                     "Joint telemetry differs from H5")
            pose_errors(tick["tcp_poses"][side], tick["tcp_poses"][side])
            _flag(obj["is_grasping"], "grasp")
    expected = ["preparation", "pregrasp", "descend", "close_and_confirm", "lift_and_hold"]
    if report.get("execution_configuration", {}).get("route") == "merge-transport":
        expected.append("carry_raise")
    expected += ["carry_orient_and_transport", "carry_lower", "release", "retreat", "verify_placement"]
    _require(segments == expected, "Required physical phase sequence mismatch")
    open_phases = {"preparation", "pregrasp", "descend", "release", "retreat", "verify_placement"}
    for index, tick in enumerate(ticks):
        grip = 1 if tick["phase"] in open_phases else -1
        _require(np.all(actions[index, [7, 15]] == grip), "Phase gripper command mismatch")
    _require(counts["preparation"] == report["preparation_steps"] and counts["verify_placement"] == 20
             and flags["success"][-20:].all(), "Preparation or common final success window mismatch")
    stages = report["stages"]
    _require([stage["name"] for stage in stages] == segments[1:] and all(stage["status"] == "passed"
             and stage["environment_steps"] == counts[stage["name"]] for stage in stages), "Stage receipts mismatch")
    hashes = report["source_sha256"]
    _require(isinstance(hashes, dict) and {"validate_sync_compact.py", "dual_arm_execution.py", "dual_payload_collision.py"}
             <= set(hashes), "Missing critical execution source identities")
    for name, expected in hashes.items():
        _require(Path(name).name == name and name not in (".", "..") and len(expected) == 64, "Invalid source identity")
        _require(hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest() == expected, "Source snapshot hash mismatch")
    return {"actions": actions, "states": states, "ticks": ticks, "report": report, "metadata": metadata,
            "action_keys": list(ACTION_KEYS), "native_flags": flags, "source_sha256": hashes,
            "sync_receipt": {"action_steps": count, "state_frames": count + 1, "phase_counts": dict(counts)}}


def action_outcome(rows: list[dict[str, Any]], *, expected_steps: int) -> dict[str, Any]:
    """Apply bilateral physical gates to a predeclared variable-length episode."""
    _rows(rows)
    _require(type(expected_steps) is int and expected_steps > 0, "Expected positive source step count")
    ticks = rows[1:]
    release = [row for row in ticks if row["phase"] == "release"]
    retreat = [row for row in ticks if row["phase"] == "retreat"]
    placement = [row for row in ticks if row["phase"] == "verify_placement"]
    objects = summarize_objects(rows)
    rises = [float(retreat[-1]["objects"][i]["tcp_pose"][2] - release[-1]["objects"][i]["tcp_pose"][2])
             if release and retreat else 0. for i in range(2)]
    window = len(placement) == 20 and [r["frame"] for r in placement] == list(range(expected_steps - 19, expected_steps + 1))
    checks = {"all_source_actions_observed": len(ticks) == expected_steps,
              "both_grasp_sequences_match": all(_flag(obj["grasp_matches_reference"], "grasp match") for r in ticks for obj in r["objects"]),
              "both_released_open_last_three": len(release) >= 3 and all(_flag(obj["fingers_open"], "open")
                  and not _flag(obj["is_grasping"], "grasp") for r in release[-3:] for obj in r["objects"]),
              "both_retreat_rises_at_least_75mm": all(rise >= .075 for rise in rises),
              "phase_order": bool(release and retreat and placement and release[-1]["frame"] < retreat[0]["frame"]
                  and retreat[-1]["frame"] < placement[0]["frame"]),
              "common_final_twenty_tick_window": window,
              "both_placements_physically_valid": window and all(_flag(obj["placement_valid"], "valid")
                  and _flag(obj["physical_region"]["inside"], "inside") and _flag(obj["fingers_open"], "open")
                  and not _flag(obj["is_grasping"], "grasp")
                  and 0 <= obj["control_frame_linear_speed_m_s"] <= .03 and 0 <= obj["control_frame_angular_speed_rad_s"] <= .2
                  and 0 <= obj["window_position_drift_m"] <= .003 and 0 <= obj["window_rotation_drift_rad"] <= .03
                  and 0 <= r["max_raw_arm_speed_rad_s"] <= .05 for r in placement for obj in r["objects"]),
              "final_twenty_official_success": window and all(_flag(r["native_success"], "success") for r in placement),
              "no_truncation_or_failure": all(not _flag(r["native_truncated"], "truncated")
                  and not _flag(r.get("native_fail", False), "fail")
                  and (not _flag(r["native_terminated"], "terminated") or _flag(r["native_success"], "success")) for r in ticks),
              "object_and_tcp_pose_errors": all(obj[group][key] <= ACTION_LIMITS[key] for obj in objects["objects"]
                  for group in ("object_pose", "tcp_pose") for key in ("pose_translation_m", "pose_rotation_rad")),
              "no_measurement_errors": all(not row.get("error") for row in rows)}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "action_steps": len(ticks), "release_steps": len(release), "retreat_steps": len(retreat),
            "placement_steps": len(placement), "actual_retreat_tcp_rise_m": rises}


def self_test() -> dict[str, Any]:
    count, rows = 40, []
    for frame in range(count + 1):
        phase = "initial" if not frame else "release" if frame <= 3 else "retreat" if frame <= 20 else "verify_placement"
        objects = [{"pose_error": {"pose_translation_m": 0., "pose_rotation_rad": 0.},
                    "tcp_pose_error": None if not frame else {"pose_translation_m": 0., "pose_rotation_rad": 0.},
                    "tcp_pose": [0, 0, .1 if frame <= 3 else .18, 1, 0, 0, 0], "grasp_matches_reference": True,
                    "fingers_open": True, "is_grasping": False, "placement_valid": True, "physical_region": {"inside": True},
                    "control_frame_linear_speed_m_s": 0., "control_frame_angular_speed_rad_s": 0.,
                    "window_position_drift_m": 0., "window_rotation_drift_rad": 0.} for _ in range(2)]
        rows.append({"frame": frame, "phase": phase, "objects": objects, "max_raw_arm_speed_rad_s": 0.,
                     "native_success": frame > 20, "native_terminated": frame > 20, "native_truncated": False})
    assert action_outcome(rows, expected_steps=count)["status"] == "passed"
    assert action_outcome(rows[:-1], expected_steps=count)["status"] == "failed"
    for side in range(2):
        for field, value in (("is_grasping", True), ("fingers_open", False), ("placement_valid", False),
                             ("window_position_drift_m", .004), ("control_frame_angular_speed_rad_s", .21)):
            changed = copy.deepcopy(rows); changed[-1]["objects"][side][field] = value
            assert action_outcome(changed, expected_steps=count)["status"] == "failed"
    changed = copy.deepcopy(rows); changed[-1]["native_truncated"] = True
    assert action_outcome(changed, expected_steps=count)["status"] == "failed"
    changed = copy.deepcopy(rows); changed[-1]["max_raw_arm_speed_rad_s"] = .051
    assert action_outcome(changed, expected_steps=count)["status"] == "failed"
    return {"status": "passed", "checks": ["variable-length bilateral outcome", "partial replay rejected",
        "either-side release/placement failures rejected", "native truncation rejected", "raw arm speed gate retained"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(self_test(), indent=2))
    elif arguments.input_dir:
        print(json.dumps(load_compact_recording(arguments.input_dir)["sync_receipt"], indent=2))
    else:
        parser.error("Provide --self-test or --input-dir")
