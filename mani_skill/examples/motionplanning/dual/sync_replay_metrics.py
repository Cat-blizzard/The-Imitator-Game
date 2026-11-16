"""Read the frozen 323-step dual-object recording and audit replay evidence.

The original loader remains unchanged. State restoration is not contact replay;
``action_outcome`` accepts physical measurements from action replay only. Small
replay errors and actual placement are separate checks. Native success-only
termination continuation is an explicit 400-step engineering policy, not the
official default 200-step benchmark protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .replay_metrics import ACTION_KEYS, load_recording, pose_errors
from .replay_support import ACTION_LIMITS, STATE_LIMITS

ACTORS = ("apple_out", "banana_out")
ACTION_STEPS = 323
POSE_METRICS = ("pose_translation_m", "pose_rotation_rad")
PHASES = ("preparation", "pregrasp", "descend", "close_and_confirm", "lift_and_hold",
          "carry_raise", "carry_orient", "carry_above_baskets", "carry_lower", "release", "retreat", "verify_placement")
LEAVES = {"actors": {"apple_in", "apple_out", "banana_in", "banana_out", "table"},
          "articulations": {"panda_wristcam-agent-0", "panda_wristcam-agent-1"}}
DEFAULT_INPUT = Path(__file__).resolve().parents[6] / "validation/dual-arm/sync-positive-20261115T134421956872Z"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _flag(value: Any, label: str) -> bool:
    array = np.asarray(value)
    _require(array.size == 1 and array.dtype.kind in "biuf" and bool(np.isfinite(array).all())
             and bool(np.isin(array, [0, 1]).all()), f"{label}: expected one finite boolean")
    return bool(array.reshape(-1)[0])


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in "fiu", f"{label}: expected real numeric values")
    result = raw.astype(float)
    _require(result.shape == shape and bool(np.isfinite(result).all()), f"{label}: expected finite {shape}")
    return result


def _metric(value: Any, label: str) -> dict[str, float]:
    _require(isinstance(value, dict), f"{label}: missing pose error dictionary")
    result = {}
    for name in POSE_METRICS:
        number = value.get(name)
        _require(isinstance(number, (int, float, np.integer, np.floating)) and not isinstance(number, (bool, np.bool_))
                 and np.isfinite(number) and number >= 0, f"{label}/{name}: expected a finite nonnegative error")
        result[name] = float(number)
    return result


def _validate_sync(recording: dict[str, Any], flags: dict[str, np.ndarray]) -> dict[str, Any]:
    """Add dual-task validation after the original single-episode loader checks."""
    actions, states, ticks, report = (recording[key] for key in ("actions", "states", "ticks", "report"))
    count = len(actions)
    _require(count == ACTION_STEPS and len(states) == count + 1 and len(ticks) == count,
             "This fixed sync baseline requires 323 actions, 324 states and 323 ticks")
    _require(report.get("status") == "passed" and report.get("mode") == "positive", "A passed positive sync recording is required")
    _require(recording["action_keys"] == ACTION_KEYS, "Physical left/right action keys changed")
    _require(len(recording["source_sha256"]) == 18, "Expected the frozen baseline's 18 source snapshots")
    for state in states:
        _require(set(state) == set(LEAVES) and all(set(state[category]) == names for category, names in LEAVES.items()),
                 "Expected all seven canonical actor/articulation state leaves")
    for name in ("success", "terminated", "truncated"):
        _require(name in flags and flags[name].shape == (count,) and flags[name].dtype.kind == "b", f"Missing boolean native H5 flags: {name}")
    counts, segments = Counter(), []
    continued = 0
    for index, tick in enumerate(ticks):
        phase = tick.get("phase")
        _require(phase in PHASES, f"tick {index}: unknown sync phase")
        counts[phase] += 1
        if not segments or segments[-1] != phase: segments.append(phase)
        _require(not _flag(tick.get("fault_injected"), f"tick {index}/fault"), "Positive input contains fault injection")
        expected_grip = -1 if phase in ("close_and_confirm", "lift_and_hold", "carry_raise", "carry_orient", "carry_above_baskets", "carry_lower") else 1
        _require(bool(np.all(actions[index, [7, 15]] == expected_grip)), f"tick {index}: phase gripper commands differ")
        requested = np.concatenate([tick["requested_actions"][key] for key in ACTION_KEYS]).astype(actions.dtype)
        _require(bool(np.array_equal(requested, actions[index])), f"tick {index}: positive request was overridden")
        objects, tcps = tick.get("objects"), tick.get("tcp_poses")
        _require(isinstance(objects, list) and len(objects) == 2 and isinstance(tcps, list) and len(tcps) == 2,
                 f"tick {index}: expected paired objects and paired TCPs")
        for side, name in enumerate(ACTORS):
            obj = objects[side]
            _require(isinstance(obj, dict) and obj.get("actor") == name, f"tick {index}: physical target pairing changed")
            pose = _array(obj.get("pose"), (7,), f"tick {index}/{name}/pose")
            pose_errors(pose, pose); pose_errors(tcps[side], tcps[side])
            actor = states[index + 1]["actors"][name]
            _require(bool(np.array_equal(pose.astype(actor.dtype), actor[:7])), f"tick {index}/{name}: pose differs from H5 frame i+1")
            for field, section in (("linear_velocity_m_s", slice(7, 10)), ("angular_velocity_rad_s", slice(10, 13))):
                value = _array(obj.get(field), (3,), f"tick {index}/{name}/{field}")
                _require(bool(np.array_equal(value.astype(actor.dtype), actor[section])), f"tick {index}/{name}: native velocity differs from H5")
            _flag(obj.get("is_grasping"), f"tick {index}/{name}/grasp")
            forces = _array(obj.get("finger_contact_force_norms_N"), (2,), f"tick {index}/{name}/contact")
            _require(bool(np.all(forces >= 0)), "Contact-force norms cannot be negative")
            q = _array(tick.get("left_qpos" if side == 0 else "right_qpos"), (9,), f"tick {index}/qpos")
            articulation = states[index + 1]["articulations"][f"panda_wristcam-agent-{side}"]
            _require(bool(np.array_equal(q.astype(articulation.dtype), articulation[13:22])), f"tick {index}: qpos differs from H5")
        success = _flag(tick.get("official_evaluation", {}).get("success"), f"tick {index}/success")
        terminated = _flag(tick.get("native_terminated"), f"tick {index}/terminated")
        truncated = _flag(tick.get("native_truncated"), f"tick {index}/truncated")
        _require(success == bool(flags["success"][index]) and terminated == bool(flags["terminated"][index])
                 and truncated == bool(flags["truncated"][index]), f"tick {index}: raw flags differ from H5")
        _require(not truncated and (not terminated or success), "Input contains truncation or non-success termination")
        continuation = _flag(tick.get("success_only_termination_continued"), f"tick {index}/continued")
        _require(continuation == (terminated and success and not truncated), "Success-only continuation receipt mismatch")
        continued += int(continuation)
    _require(tuple(segments) == PHASES, "Sync phases must occur once in the original order")
    _require(counts["preparation"] == report.get("preparation_steps"), "Preparation step receipt mismatch")
    stages = report.get("stages", [])
    _require([stage.get("name") for stage in stages] == list(PHASES[1:]), "Stage order differs from source ticks")
    _require(all(stage.get("status") == "passed" and stage.get("environment_steps") == counts[stage["name"]] for stage in stages),
             "Stage status/count receipt differs from source ticks")
    _require(counts["verify_placement"] == 20 and bool(flags["success"][-20:].all()), "Input lacks the common 20-tick final-success window")
    _require(continued == report.get("success_only_termination_continuation_ticks"), "Continuation count receipt mismatch")
    return {"action_steps": count, "state_frames": len(states), "state_leaves": 7, "source_snapshots": 18,
            "phase_counts": dict(counts), "success_only_termination_continuation_ticks": continued,
            "native_flags_match_tick_receipts": True, "object_order": list(ACTORS), "tcp_order": ["left", "right"]}


def load_sync_recording(directory: Path) -> dict[str, Any]:
    """Return all original loader keys plus ``native_flags`` and ``sync_receipt``."""
    directory = Path(directory).resolve()
    recording = load_recording(directory)
    episode_id = recording["metadata"]["episodes"][0]["episode_id"]
    with h5py.File(directory / "trajectory.h5", "r") as handle:
        group = handle[f"traj_{episode_id}"]
        flags = {name: np.asarray(group[name][:]).copy() for name in ("success", "terminated", "truncated") if name in group}
    recording["sync_receipt"] = _validate_sync(recording, flags)
    recording["native_flags"] = flags
    return recording


def _rows(rows: list[dict[str, Any]]) -> None:
    _require(isinstance(rows, list) and bool(rows), "Measured replay rows are required")
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and type(row.get("frame")) is int and row["frame"] == index,
                 "Replay rows must cover frame zero and every following frame exactly once")
        _require(isinstance(row.get("objects"), (tuple, list)) and len(row["objects"]) == 2,
                 f"frame {index}: both measured objects are required")


def summarize_objects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Maxima for each object's pose and its physical arm TCP, in source order.

    Frame-zero TCP error may be None because the recording starts TCP telemetry
    after its first action. Every subsequent frame must have a TCP error.
    """
    _rows(rows)
    objects = []
    for side, name in enumerate(ACTORS):
        pose_values, tcp_values = [], []
        for row in rows:
            obj = row["objects"][side]
            pose_values.append(_metric(obj.get("pose_error"), f"frame {row['frame']}/{name}/pose"))
            tcp = obj.get("tcp_pose_error")
            if row["frame"] == 0 and tcp is None: continue
            tcp_values.append(_metric(tcp, f"frame {row['frame']}/{name}/tcp"))
        _require(bool(tcp_values), "At least one measured TCP-error frame is required")
        objects.append({"actor": name, "object_pose": {key: max(value[key] for value in pose_values) for key in POSE_METRICS},
                        "tcp_pose": {key: max(value[key] for value in tcp_values) for key in POSE_METRICS},
                        "object_sample_count": len(pose_values), "tcp_sample_count": len(tcp_values)})
    return {"sample_count": len(rows), "objects": objects}


def action_outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require both objects to reproduce the frozen release/retreat/place cycle."""
    _rows(rows)
    ticks = rows[1:]
    summary = summarize_objects(rows)
    release = [row for row in ticks if row.get("phase") == "release"]
    retreat = [row for row in ticks if row.get("phase") == "retreat"]
    placement = [row for row in ticks if row.get("phase") == "verify_placement"]
    mismatch = [[], []]
    valid_placement, rises = [[], []], []
    truncations, non_success_terminations = [], []
    for row in ticks:
        for side, obj in enumerate(row["objects"]):
            if not _flag(obj.get("grasp_matches_reference"), f"frame {row['frame']}/grasp_match"):
                mismatch[side].append(row["frame"])
        success = _flag(row.get("native_success"), f"frame {row['frame']}/success")
        terminated = _flag(row.get("native_terminated"), f"frame {row['frame']}/terminated")
        if _flag(row.get("native_truncated"), f"frame {row['frame']}/truncated"): truncations.append(row["frame"])
        if terminated and not success: non_success_terminations.append(row["frame"])
    release_valid = []
    for side in range(2):
        release_valid.append(len(release) >= 3 and all(_flag(row["objects"][side].get("fingers_open"), "release/open")
            and not _flag(row["objects"][side].get("is_grasping"), "release/grasp") for row in release[-3:]))
        rises.append(float(_array(retreat[-1]["objects"][side].get("tcp_pose"), (7,), "retreat TCP")[2]
                           - _array(release[-1]["objects"][side].get("tcp_pose"), (7,), "release TCP")[2]) if retreat and release else 0.)
        for row in placement:
            obj = row["objects"][side]
            values = {key: _array([obj.get(key)], (1,), f"placement/{key}")[0] for key in
                      ("control_frame_linear_speed_m_s", "control_frame_angular_speed_rad_s", "window_position_drift_m", "window_rotation_drift_rad")}
            valid = (_flag(obj.get("placement_valid"), "placement/valid")
                     and _flag(obj.get("fingers_open"), "placement/open")
                     and not _flag(obj.get("is_grasping"), "placement/grasp")
                     and _flag(obj.get("physical_region", {}).get("inside"), "placement/physical region")
                     and 0 <= values["control_frame_linear_speed_m_s"] <= .03
                     and 0 <= values["control_frame_angular_speed_rad_s"] <= .2
                     and 0 <= values["window_position_drift_m"] <= .003
                     and 0 <= values["window_rotation_drift_rad"] <= .03)
            if not valid: valid_placement[side].append(row["frame"])
    phase_order = bool(release and retreat and placement and release[-1]["frame"] < retreat[0]["frame"]
                       and retreat[-1]["frame"] < placement[0]["frame"])
    common_window = len(placement) == 20 and [row["frame"] for row in placement] == list(range(len(rows) - 20, len(rows)))
    checks = {"all_323_actions_observed": len(ticks) == ACTION_STEPS,
              "both_grasp_predicate_sequences_match": all(not frames for frames in mismatch),
              "both_released_open_last_three": all(release_valid),
              "both_retreat_rises_at_least_75mm": all(rise >= .075 for rise in rises),
              "release_then_retreat_then_placement": phase_order,
              "common_final_twenty_tick_window": common_window,
              "both_placements_physically_valid": common_window and all(not frames for frames in valid_placement),
              "final_twenty_official_success": common_window and all(_flag(row.get("native_success"), "final success") for row in placement),
              "no_truncation": not truncations,
              "only_success_terminations_continued": not non_success_terminations,
              "measurement_errors_absent": all(not row.get("error") for row in rows),
              "both_object_and_tcp_pose_errors_within_limits": all(
                  obj[group][key] <= ACTION_LIMITS[key] for obj in summary["objects"]
                  for group in ("object_pose", "tcp_pose") for key in POSE_METRICS)}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "action_steps": len(ticks), "release_steps": len(release), "retreat_steps": len(retreat),
            "placement_steps": len(placement), "actual_retreat_tcp_rise_m": rises,
            "grasp_mismatch_frames": mismatch, "invalid_placement_frames": valid_placement,
            "truncated_frames": truncations, "non_success_termination_frames": non_success_terminations}


def self_test() -> dict[str, Any]:
    """CPU-only synthetic bilateral outcomes plus the real read-only loader."""
    tests = []
    zero = {"pose_translation_m": 0., "pose_rotation_rad": 0.}
    rows = []
    for frame in range(ACTION_STEPS + 1):
        phase = "initial" if frame == 0 else "release" if 282 <= frame <= 284 else "retreat" if 285 <= frame <= 303 else "verify_placement" if frame >= 304 else "moving"
        objects = [{"pose_error": zero.copy(), "tcp_pose_error": None if frame == 0 else zero.copy(),
                    "tcp_pose": [0., 0., .18 if frame >= 285 else .1, 1., 0., 0., 0.],
                    "grasp_matches_reference": True, "is_grasping": False, "fingers_open": True,
                    "placement_valid": True, "physical_region": {"inside": True},
                    "control_frame_linear_speed_m_s": 0., "control_frame_angular_speed_rad_s": 0.,
                    "window_position_drift_m": 0., "window_rotation_drift_rad": 0.} for _ in ACTORS]
        rows.append({"frame": frame, "phase": phase, "objects": objects, "native_success": frame >= 304,
                     "native_terminated": frame >= 304, "native_truncated": False})
    assert action_outcome(rows)["status"] == "passed"
    summary = summarize_objects(rows)
    assert all(obj["object_sample_count"] == 324 and obj["tcp_sample_count"] == 323 for obj in summary["objects"])
    tests.append("complete 323-step dual outcome and both 324/323 pose/TCP sample counts")
    for side in (0, 1):
        for frame, field, value in ((323, "placement_valid", False), (323, "is_grasping", True),
                                     (284, "fingers_open", False), (1, "grasp_matches_reference", False),
                                     (323, "control_frame_linear_speed_m_s", .031), (323, "control_frame_angular_speed_rad_s", .201),
                                     (323, "window_position_drift_m", .0031), (323, "window_rotation_drift_rad", .031)):
            altered = copy.deepcopy(rows); altered[frame]["objects"][side][field] = value
            assert action_outcome(altered)["status"] == "failed", (side, field)
        altered = copy.deepcopy(rows); altered[303]["objects"][side]["tcp_pose"][2] = .17
        assert action_outcome(altered)["status"] == "failed"
        altered = copy.deepcopy(rows); altered[323]["objects"][side]["physical_region"]["inside"] = False
        assert action_outcome(altered)["status"] == "failed"
    tests.append("either side independently rejects grasp/release/retreat/region/FD/drift failures")
    for field, value in (("native_success", False), ("native_truncated", True)):
        altered = copy.deepcopy(rows); altered[-1][field] = value
        assert action_outcome(altered)["status"] == "failed"
    assert action_outcome(rows[:-1])["status"] == "failed"
    altered = copy.deepcopy(rows); altered[302]["phase"] = "verify_placement"
    assert action_outcome(altered)["status"] == "failed"
    tests.append("raw flag failures, partial episodes and noncommon windows cannot pass")
    pose = np.array([.1, .2, .3, 1., 0., 0., 0.]); opposite = pose.copy(); opposite[3:] *= -1
    assert pose_errors(pose, opposite) == zero
    altered = copy.deepcopy(rows); altered[-1]["objects"][1]["tcp_pose_error"]["pose_translation_m"] = .006
    assert action_outcome(altered)["status"] == "failed" and summarize_objects(altered)["objects"][1]["tcp_pose"]["pose_translation_m"] == .006
    altered = copy.deepcopy(rows); altered[-1]["objects"][0]["pose_error"]["pose_rotation_rad"] = .06
    assert action_outcome(altered)["status"] == "failed"
    tests.append("sign-invariant quaternion errors and independent bilateral object/TCP maxima")

    def reject(call: Any) -> None:
        try: call()
        except (ValueError, KeyError): return
        raise AssertionError("Malformed replay evidence was accepted")

    for field in ("pose_error", "tcp_pose_error"):
        altered = copy.deepcopy(rows); altered[1]["objects"][0][field] = None
        reject(lambda: summarize_objects(altered))
    altered = copy.deepcopy(rows); altered[1]["objects"][0]["pose_error"]["pose_translation_m"] = float("nan")
    reject(lambda: summarize_objects(altered))
    reject(lambda: summarize_objects(rows[1:]))
    tests.append("missing/nonfinite measurements and shifted frame sequences reject")
    recording = load_sync_recording(DEFAULT_INPUT)
    assert recording["sync_receipt"]["action_steps"] == 323 and recording["sync_receipt"]["state_leaves"] == 7
    bad = copy.deepcopy(recording); bad["ticks"][0]["objects"].reverse()
    reject(lambda: _validate_sync(bad, recording["native_flags"]))
    bad = copy.deepcopy(recording); bad["ticks"][0]["native_terminated"] = True
    reject(lambda: _validate_sync(bad, recording["native_flags"]))
    bad = copy.deepcopy(recording); bad["ticks"][0]["objects"][0]["pose"][0] += .001
    reject(lambda: _validate_sync(bad, recording["native_flags"]))
    tests.append("read-only frozen source loader and target/flag/H5 alignment corruption rejection")
    return {"status": "passed", "gpu_used": False, "native_physics_tested": False,
            "tests": tests, "check_count": len(tests), "source_receipt": recording["sync_receipt"],
            "state_limits": dict(STATE_LIMITS), "action_limits": dict(ACTION_LIMITS)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))
