"""Shared replay sequencing, finite-state conversion and historical thresholds.

Recordings contain T finite float32 actions of shape (16,) and T+1 state
mappings. Action columns follow two ordered physical arm keys: seven arm joint
positions in radians plus one normalized OPEN/CLOSED gripper command per arm.
Batched state leaves have shape (1, D); conversion owns copies of each leaf.
Threshold names retain their original metre/radian/velocity units and values.
State replay restores each frame without stepping. Action replay restores frame
zero once and submits each original action once. This helper never makes an
environment, changes recorded input, or changes task acceptance policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Direct-script setup is test-only; normal package imports need no path changes.
if __name__ == "__main__" and not __package__:
    import mani_skill.examples.motionplanning.dual as _dual_package
    _dual_package.__path__ = [str(Path(__file__).parent), *_dual_package.__path__]
    __package__ = _dual_package.__name__

from .runtime_io import json_value
from .physical_observation import continue_success_only

STATE_LIMITS = {"pose_translation_m": 1e-6, "pose_rotation_rad": 1e-5,
                "linear_velocity_m_s": 1e-5, "angular_velocity_rad_s": 1e-4,
                "arm_joint_max_rad": 1e-6, "finger_max_m": 1e-7, "joint_velocity_max": 1e-5}


ACTION_LIMITS = {"pose_translation_m": .005, "pose_rotation_rad": .05,
                 "arm_joint_max_rad": .01, "finger_max_m": .002}


def split_action(row: Any, keys: list[str]) -> dict[str, np.ndarray]:
    values = np.asarray(row, dtype=np.float32)
    if len(keys) != 2 or len(set(keys)) != 2 or values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("Expected two distinct ordered keys and one finite 16-value action")
    if not np.isin(values[[7, 15]], [-1, 1]).all():
        raise ValueError("This fixed-gripper recording requires OPEN/CLOSED commands")
    return {key: values[index * 8:(index + 1) * 8].copy() for index, key in enumerate(keys)}


def replay_sequence(recording: dict[str, Any], mode: str, *,
                    restore: Callable[[int, dict[str, Any]], None],
                    step: Callable[[dict[str, np.ndarray]], None],
                    observe: Callable[[int], None]) -> dict[str, int]:
    """Keep state-setting and physical stepping separate, including frame zero."""
    if mode not in ("state", "action"):
        raise ValueError("Unknown replay mode")
    states, actions = recording["states"], recording["actions"]
    if len(states) != len(actions) + 1 or not len(actions):
        raise ValueError("Replay requires T actions and T+1 states")
    resets, steps = 0, 0
    if mode == "state":
        for index, state in enumerate(states):
            restore(index, state)
            resets += 1
            observe(index)
    else:
        restore(0, states[0])
        resets += 1
        observe(0)
        for index, action in enumerate(actions, start=1):
            step(split_action(action, recording["action_keys"]))
            steps += 1
            observe(index)
    return {"state_restore_calls": resets, "physical_environment_steps": steps,
            "observed_frames": len(states)}


def within_limits(maxima: dict[str, Any], limits: dict[str, float]) -> bool:
    return all(name in maxima and np.isfinite(maxima[name]) and 0 <= maxima[name] <= limit
               for name, limit in limits.items())


def unbatch_state(value: Any) -> Any:
    from .trajectory_execution import as_numpy
    if isinstance(value, dict): return {key: unbatch_state(v) for key, v in value.items()}
    array = as_numpy(value)
    if array.ndim != 2 or array.shape[0] != 1 or not np.isfinite(array).all():
        raise ValueError("Expected one finite batched recorded-state leaf")
    return array[0].copy()


def batch_state(value: Any) -> Any:
    return {key: batch_state(v) for key, v in value.items()} if isinstance(value, dict) else np.asarray(value)[None].copy()


def replay_signal(terminated: Any, truncated: Any, info: dict[str, Any]) -> dict[str, bool]:
    """Keep native signals and reject any non-success end of the episode."""
    def flag(value: Any) -> bool:
        a = np.asarray(json_value(value))
        if a.size != 1 or not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
            raise ValueError("Expected a single boolean signal")
        return bool(a.reshape(-1)[0])
    done, continued = continue_success_only(terminated, truncated, info)
    signal = {"success": flag(info.get("success", False)), "terminated": flag(terminated),
              "truncated": flag(truncated), "fail": flag(info.get("fail", False)),
              "success_only_termination_continued": continued}
    signal["unexpected_end"] = bool(done or signal["truncated"] or signal["fail"])
    return signal

def self_test() -> dict[str, Any]:
    """Check the extracted API on the CPU without an environment or GPU."""
    def rejects(function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        try:
            function(*args, **kwargs)
        except ValueError:
            return
        raise AssertionError(f"Expected ValueError from {function.__name__}")

    value = {"actors": {"a": np.arange(13, dtype=float)[None]},
             "articulations": {"r": np.arange(31, dtype=float)[None]}}
    unpacked = unbatch_state(value)
    restored = batch_state(unpacked)
    assert all(np.array_equal(value[c][n], restored[c][n]) for c in value for n in value[c])
    unpacked["actors"]["a"][0] = -2
    assert value["actors"]["a"][0, 0] == restored["actors"]["a"][0, 0] == 0
    rejects(unbatch_state, np.zeros(13))
    rejects(unbatch_state, np.zeros((2, 13)))
    rejects(unbatch_state, np.array([[np.nan]]))
    rejects(unbatch_state, np.array([[np.inf]]))
    assert unbatch_state({}) == batch_state({}) == {}
    assert np.array_equal(unbatch_state(np.empty((1, 0))), np.empty(0))

    assert replay_signal(True, False, {"success": True})["success_only_termination_continued"]
    assert not replay_signal(True, False, {"success": True})["unexpected_end"]
    assert replay_signal(True, False, {"success": True, "fail": True})["unexpected_end"]
    assert replay_signal(True, True, {"success": True})["unexpected_end"]
    assert replay_signal(True, False, {})["unexpected_end"]
    assert replay_signal(False, False, {"fail": True})["unexpected_end"]
    ordinary = replay_signal(np.array([False]), np.array([False]), {"success": np.array([True])})
    assert ordinary == {"success": True, "terminated": False, "truncated": False,
                        "fail": False, "success_only_termination_continued": False,
                        "unexpected_end": False}
    for invalid in (np.array([True, False]), np.array([]), np.nan, np.inf, 2, -1):
        rejects(replay_signal, invalid, False, {})
    rejects(replay_signal, False, False, {"success": np.array([True, False])})
    rejects(replay_signal, False, False, {"fail": 2})

    keys = ["left", "right"]
    action = np.r_[np.zeros(7), -1, np.ones(7), 1].astype(np.float32)
    decoded = split_action(action, keys)
    assert list(decoded) == keys and decoded["left"].shape == decoded["right"].shape == (8,)
    assert decoded["left"].dtype == decoded["right"].dtype == np.float32
    assert np.array_equal(np.r_[decoded["left"], decoded["right"]], action)
    decoded["left"][0] = .5
    assert action[0] == 0
    rejects(split_action, action, ["same", "same"])
    rejects(split_action, action[:-1], keys)
    invalid_gripper = action.copy(); invalid_gripper[7] = 0
    rejects(split_action, invalid_gripper, keys)
    nonfinite = action.copy(); nonfinite[0] = np.nan
    rejects(split_action, nonfinite, keys)

    recording = {"actions": np.stack([action, action]),
                 "states": [{"frame": i} for i in range(3)], "action_keys": keys}
    callbacks = dict(restore=lambda index, state: None, step=lambda row: None, observe=lambda index: None)
    for mode in ("state", "action"):
        setters, steps, observed, events = [], [], [], []
        def restore(index: int, state: dict[str, Any]) -> None:
            assert state is recording["states"][index]
            setters.append(index); events.append(("restore", index))
        def step(row: dict[str, np.ndarray]) -> None:
            assert np.array_equal(np.r_[row[keys[0]], row[keys[1]]], action)
            steps.append(row); events.append(("step", len(steps)))
        def observe(index: int) -> None:
            observed.append(index); events.append(("observe", index))
        result = replay_sequence(recording, mode, restore=restore, step=step, observe=observe)
        assert setters == ([0, 1, 2] if mode == "state" else [0])
        assert len(steps) == (0 if mode == "state" else 2) and observed == [0, 1, 2]
        assert result == {"state_restore_calls": len(setters), "physical_environment_steps": len(steps),
                          "observed_frames": 3}
        expected = ([("restore", i), ("observe", i)] for i in range(3)) if mode == "state" else ([("restore", 0), ("observe", 0)], [("step", 1), ("observe", 1)], [("step", 2), ("observe", 2)])
        assert events == [event for pair in expected for event in pair]
    rejects(replay_sequence, recording, "unknown", **callbacks)
    rejects(replay_sequence, {**recording, "states": recording["states"][:-1]}, "action", **callbacks)
    rejects(replay_sequence, {**recording, "states": [{}], "actions": []}, "state", **callbacks)

    for limits in (ACTION_LIMITS, STATE_LIMITS):
        assert within_limits({name: 0. for name in limits}, limits)
        assert within_limits(dict(limits), limits)
        metric, threshold = next(iter(limits.items()))
        for invalid in (-1., np.nan, np.inf, threshold * 2):
            assert not within_limits({**limits, metric: invalid}, limits)
        assert not within_limits({name: threshold for name, threshold in limits.items() if name != metric}, limits)
    return {"status": "passed", "checks": ["finite batched states and alias isolation",
        "native signal continuation with failure and timeout rejection", "ordered finite 16D action split",
        "state replay zero physics steps", "action replay initial-frame-only restoration",
        "T+1 frame and callback ordering", "unchanged inclusive finite replay thresholds"],
        "state_leaf_shapes": {"actor": [1, 13], "articulation": [1, 31]},
        "action_shape": [16], "physical_environment_steps": 0}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, allow_nan=False))
