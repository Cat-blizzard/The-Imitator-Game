"""CPU-only recording validation and state replay error measurements.

This loader supports the single-episode, two-Panda ``pd_joint_pos`` recording
contract used by the placement probe. It reads evidence without modifying it.
State playback restores stored state; action replay restores only its initial
state and then advances physics. These measurements do not establish contact
cache restoration, task success, or continuous collision safety.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np


TASK = "TwoRobotPickAppleBananaToBaskets-v1"
ACTION_KEYS = ["panda_wristcam-0", "panda_wristcam-1"]
STATE_WIDTHS = {"actors": 13, "articulations": 31}
METRICS = (
    "pose_translation_m", "pose_rotation_rad", "linear_velocity_m_s",
    "angular_velocity_rad_s", "arm_joint_max_rad", "finger_max_m",
    "joint_velocity_max", "arm_joint_velocity_max_rad_s",
    "finger_velocity_max_m_s",
)
DEFAULT_RECORDING = (
    Path(__file__).resolve().parents[6]
    / "validation/dual-arm/place-positive-20261115T124829247240Z"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_json(value: Any, path: str = "json") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_json(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_json(child, f"{path}/{index}")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"{path}: nonfinite number")


def _json(text: str, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label}: nonfinite JSON constant {value}")

    value = json.loads(text, parse_constant=reject_constant)
    _finite_json(value, label)
    return value


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    # Raw ManiSkill states may be torch tensors; no torch import is needed.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    raw = np.asarray(value)
    _require(raw.dtype.kind in "fiu", f"{label}: expected real numeric data")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape == (1,) + shape:
        arr = arr[0]
    _require(arr.shape == shape, f"{label}: expected {shape}, got {arr.shape}")
    _require(bool(np.isfinite(arr).all()), f"{label}: nonfinite data")
    return arr


def _quaternion(value: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    _require(math.isfinite(norm) and abs(norm - 1.0) <= 1e-3,
             f"{label}: quaternion must be unit length, got norm {norm}")
    return value / norm


def pose_errors(reference7: Any, actual7: Any) -> dict[str, float]:
    """Position norm and sign-invariant wxyz quaternion angular distance."""
    ref = _array(reference7, (7,), "reference pose")
    act = _array(actual7, (7,), "actual pose")
    qr = _quaternion(ref[3:7], "reference pose")
    qa = _quaternion(act[3:7], "actual pose")
    # Vector part of conjugate(qr) * qa avoids acos loss near zero angle.
    vector = qr[0] * qa[1:] - qa[0] * qr[1:] - np.cross(qr[1:], qa[1:])
    scalar = float(np.dot(qr, qa))
    result = {
        "pose_translation_m": float(np.linalg.norm(act[:3] - ref[:3])),
        "pose_rotation_rad": float(2.0 * math.atan2(float(np.linalg.norm(vector)), abs(scalar))),
    }
    _finite_json(result, "pose errors")
    return result


def _state_leaves(state: dict, label: str) -> dict[str, np.ndarray]:
    _require(isinstance(state, dict), f"{label}: expected state dictionary")
    _require(set(state) == set(STATE_WIDTHS),
             f"{label}: expected only actors/articulations, got {sorted(state)}")
    leaves = {}
    for category, width in STATE_WIDTHS.items():
        group = state[category]
        _require(isinstance(group, dict), f"{label}/{category}: expected dictionary")
        for name, value in group.items():
            _require(isinstance(name, str) and name != "", f"{label}: invalid entity name")
            key = f"{category}/{name}"
            arr = _array(value, (width,), f"{label}/{key}")
            _quaternion(arr[3:7], f"{label}/{key}")
            leaves[key] = arr
    _require(bool(leaves), f"{label}: empty state")
    return leaves


def _global_max(per_entity: dict[str, dict[str, float]]) -> dict[str, float]:
    return {metric: max((values.get(metric, 0.0) for values in per_entity.values()), default=0.0)
            for metric in METRICS}


def compare_states(reference: dict, actual: dict) -> dict:
    """Compare actor13 and Panda articulation31 states, batched or unbatched.

    Per-entity keys are ``actors/<name>`` and ``articulations/<name>``. Linear
    and angular velocity errors use vector norms; joint errors use max absolute
    differences. ``joint_velocity_max`` mixes the recorded joint velocity units,
    so arm/finger velocity maxima are also returned separately.
    """
    ref = _state_leaves(reference, "reference")
    act = _state_leaves(actual, "actual")
    _require(set(ref) == set(act), "reference/actual entity sets differ")
    errors = {}
    for key, r in ref.items():
        a = act[key]
        values = pose_errors(r[:7], a[:7])
        values.update({
            "linear_velocity_m_s": float(np.linalg.norm(a[7:10] - r[7:10])),
            "angular_velocity_rad_s": float(np.linalg.norm(a[10:13] - r[10:13])),
        })
        if key.startswith("articulations/"):
            values.update({
                "arm_joint_max_rad": float(np.max(np.abs(a[13:20] - r[13:20]))),
                "finger_max_m": float(np.max(np.abs(a[20:22] - r[20:22]))),
                "joint_velocity_max": float(np.max(np.abs(a[22:31] - r[22:31]))),
                "arm_joint_velocity_max_rad_s": float(np.max(np.abs(a[22:29] - r[22:29]))),
                "finger_velocity_max_m_s": float(np.max(np.abs(a[29:31] - r[29:31]))),
            })
        _finite_json(values, key)
        errors[key] = values
    return {"per_entity": errors, "global": _global_max(errors), "entity_count": len(errors)}


def summarize_errors(samples: list[dict]) -> dict:
    """Return per-entity and global maxima; an empty series has sample_count=0."""
    maxima: dict[str, dict[str, float]] = {}
    schema = None
    for index, sample in enumerate(samples):
        _require(isinstance(sample, dict) and isinstance(sample.get("per_entity"), dict),
                 f"error sample {index}: missing per_entity")
        entities = sample["per_entity"]
        current = {key: set(values) for key, values in entities.items()}
        if schema is None:
            schema = current
            maxima = {key: {metric: 0.0 for metric in values} for key, values in current.items()}
        _require(current == schema, f"error sample {index}: entity or metric schema differs")
        for key, values in entities.items():
            _require(bool(values), f"error sample {index}/{key}: empty metrics")
            for metric, value in values.items():
                _require(metric in METRICS and isinstance(value, (int, float))
                         and math.isfinite(value) and value >= 0,
                         f"error sample {index}/{key}/{metric}: invalid metric")
                maxima[key][metric] = max(maxima[key][metric], float(value))
    return {"per_entity": maxima, "global": _global_max(maxima),
            "entity_count": len(maxima), "sample_count": len(samples)}


def load_recording(directory: Path) -> dict:
    """Read and validate a single 400-step-budget CPU placement recording.

    The budget is metadata, not a claim of official 200-step benchmark success.
    No observations are required. Snapshot hashes are checked against the local
    ``source/`` files, never against potentially newer development source.
    """
    directory = Path(directory).resolve()
    report = _json((directory / "report.json").read_text(), "report.json")
    metadata = _json((directory / "trajectory.json").read_text(), "trajectory.json")
    ticks = [_json(line, f"ticks.jsonl:{index}")
             for index, line in enumerate((directory / "ticks.jsonl").read_text().splitlines(), 1)]
    _require(isinstance(report, dict) and isinstance(metadata, dict), "report/metadata must be dictionaries")
    episodes = metadata.get("episodes", [])
    _require(isinstance(episodes, list) and len(episodes) == 1, "expected exactly one metadata episode")
    episode = episodes[0]
    env_info = metadata.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {})
    _require(report.get("task") == env_info.get("env_id") == TASK, "task metadata mismatch")
    _require(report.get("max_episode_steps") == env_info.get("max_episode_steps") == 400,
             "expected matching 400-step budget metadata")
    _require(env_kwargs.get("max_episode_steps", 400) == 400, "env_kwargs episode budget mismatch")
    _require(env_kwargs.get("control_mode") == "pd_joint_pos", "unexpected environment control mode")
    _require(env_kwargs.get("sim_backend") == "cpu"
             and report.get("backend", {}).get("physics") == "physx_cpu", "physics backend mismatch")
    _require(report.get("control_timestep_s") == 0.05, "expected 0.05-second control timestep")
    modes = episode.get("control_mode", {})
    _require(isinstance(modes, dict) and list(modes) == ACTION_KEYS
             and all(value == "pd_joint_pos" for value in modes.values()), "episode action keys/control mode mismatch")
    recording = report.get("recording", {})
    action_keys = recording.get("action_key_order")
    _require(action_keys == ACTION_KEYS, "report recording action-key order mismatch")
    _require(report.get("seed") == episode.get("episode_seed")
             == episode.get("reset_kwargs", {}).get("seed"), "episode seed metadata mismatch")
    episode_id = episode.get("episode_id")
    _require(type(episode_id) is int and episode_id >= 0, "invalid episode_id")

    state_arrays = {}
    with h5py.File(directory / "trajectory.h5", "r") as handle:
        trajectory_name = f"traj_{episode_id}"
        _require(list(handle) == [trajectory_name], "expected exactly the metadata H5 trajectory")
        trajectory = handle[trajectory_name]
        _require("actions" in trajectory and isinstance(trajectory["actions"], h5py.Dataset), "missing H5 actions")
        actions = np.asarray(trajectory["actions"][:])
        _require(actions.ndim == 2 and actions.shape[1] == 16 and actions.dtype.kind == "f",
                 f"expected floating H5 actions (T,16), got {actions.shape}/{actions.dtype}")
        _require(bool(np.isfinite(actions).all()), "H5 actions are nonfinite")
        count = len(actions)
        _require(0 < count <= 400, "invalid action count for 400-step budget")
        _require(report.get("total_environment_steps") == recording.get("recorded_steps")
                 == episode.get("elapsed_steps") == count, "action/metadata step counts differ")
        _require(recording.get("actions_shape") == [count, 16], "report action shape mismatch")
        _require("env_states" in trajectory, "missing H5 env_states")
        states_group = trajectory["env_states"]
        _require(set(states_group) == set(STATE_WIDTHS), "unknown/missing H5 state category")
        for category, width in STATE_WIDTHS.items():
            _require(isinstance(states_group[category], h5py.Group), f"{category}: expected H5 group")
            for name, dataset in states_group[category].items():
                key = f"env_states/{category}/{name}"
                _require(isinstance(dataset, h5py.Dataset), f"{key}: expected H5 dataset")
                values = np.asarray(dataset[:])
                _require(values.shape == (count + 1, width) and values.dtype.kind == "f",
                         f"{key}: expected {(count + 1, width)} floating states, got {values.shape}/{values.dtype}")
                _require(bool(np.isfinite(values).all()), f"{key}: nonfinite states")
                norms = np.linalg.norm(values[:, 3:7].astype(np.float64), axis=1)
                _require(bool((np.abs(norms - 1.0) <= 1e-3).all()), f"{key}: nonunit state quaternion")
                state_arrays[key] = values
        _require(bool(state_arrays), "empty H5 states")
        _require(recording.get("state_lengths") == {key: count + 1 for key in state_arrays},
                 "report state lengths/keys do not match all H5 leaves")
        for key in ("success", "terminated", "truncated"):
            if key in trajectory:
                _require(isinstance(trajectory[key], h5py.Dataset)
                         and trajectory[key].shape == (count,), f"H5 {key}: action-time length mismatch")

    _require(len(ticks) == count, "tick/action lengths differ")
    for index, tick in enumerate(ticks):
        _require(isinstance(tick, dict) and tick.get("environment_step") == index + 1,
                 f"tick {index}: missing/nonsequential environment step")
        for field in ("actions", "requested_actions"):
            values = tick.get(field)
            _require(isinstance(values, dict) and list(values) == action_keys,
                     f"tick {index}/{field}: action-key order mismatch")
            for key in action_keys:
                _array(values[key], (8,), f"tick {index}/{field}/{key}")
        actual = np.concatenate([tick["actions"][key] for key in action_keys]).astype(actions.dtype)
        _require(bool(np.array_equal(actual, actions[index])), f"tick {index}: actual action differs from H5")

    source_sha256 = report.get("source_sha256", {})
    _require(isinstance(source_sha256, dict) and bool(source_sha256), "missing source SHA256 evidence")
    for filename, expected in source_sha256.items():
        _require(isinstance(filename, str) and Path(filename).name == filename and filename not in (".", ".."),
                 "source SHA256 keys must be basenames")
        _require(isinstance(expected, str) and len(expected) == 64
                 and all(character in "0123456789abcdef" for character in expected),
                 f"{filename}: invalid SHA256")
        snapshot = directory / "source" / filename
        actual_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        _require(actual_hash == expected, f"source/{filename}: snapshot SHA256 mismatch")

    states = []
    for index in range(count + 1):
        state = {category: {} for category in STATE_WIDTHS}
        for key, values in state_arrays.items():
            _, category, name = key.split("/", 2)
            state[category][name] = values[index].copy()
        states.append(state)
    return {"actions": actions, "states": states, "ticks": ticks, "report": report,
            "metadata": metadata, "action_keys": list(action_keys),
            "source_sha256": dict(source_sha256)}


def _self_test(recording_directory: Path) -> dict:
    tests = []
    pose = np.array([0., 0., 0., 1., 0., 0., 0.])
    actor = np.r_[pose, np.zeros(6)]
    articulation = np.r_[pose, np.zeros(24)]
    reference = {"actors": {"apple": actor}, "articulations": {"panda": articulation}}
    identical = {category: {name: values[None, :].copy() for name, values in group.items()}
                 for category, group in reference.items()}
    for group in identical.values():
        for values in group.values():
            values[:, 3:7] *= -1
    errors = compare_states(reference, identical)
    assert max(errors["global"].values()) == 0
    tests.append("quaternion_sign_and_batched_states")
    translated = np.array([3., 4., 0., math.cos(math.pi / 4), 0., 0., math.sin(math.pi / 4)])
    error = pose_errors(pose, translated)
    assert abs(error["pose_translation_m"] - 5.) < 1e-12
    assert abs(error["pose_rotation_rad"] - math.pi / 2) < 1e-12
    tests.append("known_pose_translation_rotation")
    changed = {category: {name: values.copy() for name, values in group.items()}
               for category, group in reference.items()}
    changed["articulations"]["panda"][[13, 20, 22, 29]] = [.01, .002, .03, .004]
    changed["actors"]["apple"][7:10] = [3., 4., 0.]
    changed["actors"]["apple"][10:13] = [0., 0., 2.]
    changed_errors = compare_states(reference, changed)
    joint = changed_errors["per_entity"]["articulations/panda"]
    assert joint["arm_joint_max_rad"] == .01 and joint["finger_max_m"] == .002
    assert joint["joint_velocity_max"] == .03 and joint["finger_velocity_max_m_s"] == .004
    assert changed_errors["global"]["linear_velocity_m_s"] == 5.
    assert changed_errors["global"]["angular_velocity_rad_s"] == 2.
    summary = summarize_errors([errors, changed_errors])
    assert summary["sample_count"] == 2 and summary["global"] == changed_errors["global"]
    json.dumps(summary, allow_nan=False)
    tests.append("state_layout_joint_velocity_and_summary")

    def rejected(function, label):
        try:
            function()
        except (ValueError, KeyError):
            tests.append(label)
        else:
            raise AssertionError(f"invalid fixture accepted: {label}")

    rejected(lambda: pose_errors(np.zeros(6), pose), "bad_pose_shape_rejected")
    rejected(lambda: pose_errors(np.r_[np.nan, pose[1:]], pose), "nonfinite_rejected")
    rejected(lambda: pose_errors(np.zeros(7), pose), "nonunit_quaternion_rejected")
    rejected(lambda: compare_states(reference, {"actors": {"apple": actor[:-1]}, "articulations": {"panda": articulation}}),
             "bad_actor_shape_rejected")
    rejected(lambda: summarize_errors([errors, {"per_entity": {}}]), "missing_entity_rejected")

    # Temporary files are disposable test fixtures, never recording evidence.
    with tempfile.TemporaryDirectory(prefix="tig-replay-metrics-selftest-") as temp:
        fixture = Path(temp)
        (fixture / "source").mkdir()
        snapshot = b"# disposable replay loader fixture\n"
        (fixture / "source/fixture.py").write_bytes(snapshot)
        actions = np.zeros((2, 16), dtype=np.float32)
        modes = {key: "pd_joint_pos" for key in ACTION_KEYS}
        metadata = {"env_info": {"env_id": TASK, "max_episode_steps": 400,
                                  "env_kwargs": {"control_mode": "pd_joint_pos", "sim_backend": "cpu"}},
                    "episodes": [{"episode_id": 0, "episode_seed": 0, "reset_kwargs": {"seed": 0},
                                  "elapsed_steps": 2, "control_mode": modes}]}
        report = {"task": TASK, "seed": 0, "max_episode_steps": 400,
                  "backend": {"physics": "physx_cpu"}, "control_timestep_s": .05,
                  "total_environment_steps": 2,
                  "source_sha256": {"fixture.py": hashlib.sha256(snapshot).hexdigest()},
                  "recording": {"action_key_order": ACTION_KEYS, "recorded_steps": 2,
                                "actions_shape": [2, 16],
                                "state_lengths": {f"env_states/{category}/{name}": 3
                                                  for category, group in reference.items() for name in group}}}
        fixture_ticks = [{"environment_step": i + 1, "actions": {key: [0.] * 8 for key in ACTION_KEYS},
                          "requested_actions": {key: [0.] * 8 for key in ACTION_KEYS}} for i in range(2)]
        (fixture / "trajectory.json").write_text(json.dumps(metadata))
        (fixture / "report.json").write_text(json.dumps(report))

        def write_ticks():
            (fixture / "ticks.jsonl").write_text("\n".join(json.dumps(tick) for tick in fixture_ticks) + "\n")

        def write_h5(drop_frame=False):
            with h5py.File(fixture / "trajectory.h5", "w") as handle:
                trajectory = handle.create_group("traj_0")
                trajectory.create_dataset("actions", data=actions)
                for category, group in reference.items():
                    for name, values in group.items():
                        length = 2 if drop_frame and category == "actors" else 3
                        trajectory.create_dataset(f"env_states/{category}/{name}", data=np.tile(values, (length, 1)))

        write_ticks()
        write_h5()
        assert len(load_recording(fixture)["states"]) == 3
        fixture_ticks[1]["environment_step"] = 3
        write_ticks()
        rejected(lambda: load_recording(fixture), "nonsequential_tick_rejected")
        fixture_ticks[1]["environment_step"] = 2
        fixture_ticks[1]["actions"][ACTION_KEYS[0]][0] = .01
        write_ticks()
        rejected(lambda: load_recording(fixture), "tick_h5_action_mismatch_rejected")
        fixture_ticks[1]["actions"][ACTION_KEYS[0]][0] = 0.
        write_ticks()
        write_h5(drop_frame=True)
        rejected(lambda: load_recording(fixture), "missing_state_frame_rejected")

    loaded = load_recording(recording_directory)
    assert len(loaded["states"]) == len(loaded["actions"]) + 1
    initial = loaded["states"][0]
    assert max(compare_states(initial, initial)["global"].values()) == 0
    tests.append("actual_h5_without_observations_and_snapshot_hashes")
    return {"status": "passed", "tests": tests, "recording": str(recording_directory.resolve()),
            "action_shape": list(loaded["actions"].shape), "state_frames": len(loaded["states"]),
            "entity_count": compare_states(initial, initial)["entity_count"],
            "source_snapshot_files": len(loaded["source_sha256"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test (this module performs no simulation)")
    print(json.dumps(_self_test(args.recording), indent=2, allow_nan=False))
