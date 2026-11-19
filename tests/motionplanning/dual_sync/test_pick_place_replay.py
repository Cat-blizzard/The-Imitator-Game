"""Native action contracts, terminal-frame alignment and retained failures."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from mani_skill.examples.motionplanning.dual.sync import pick_place_replay as replay
from mani_skill.examples.motionplanning.dual.sync.runtime_policy import RuntimePolicy


def order(count):
    return [{"key": f"arm-{i}", "size": count+1,
             "arm_joint_names": [f"j{x}" for x in range(count)],
             "state_joint_names": [f"j{x}" for x in range(count+2)]} for i in range(2)]


def archive(tmp_path, count=6, terminal_frames=4):
    path = tmp_path / "trajectory.h5"
    contract = order(count)
    metadata = {
        "env_info": {"env_id": "DualArmPickPlace-v1"},
        "episodes": [{"episode_id": 0}],
        "embodiment_sync": {
            "task": "DualArmPickPlace-v1", "action_order": contract,
            "policy": RuntimePolicy().to_dict(), "control_freq": 20,
            "events": [{"step": i+1, "phase": "preparation" if i < 2 else "pregrasp"} for i in range(3)],
        },
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with h5py.File(path, "w") as handle:
        traj = handle.create_group("traj_0")
        traj.create_dataset("actions", data=np.zeros((3, 2*(count+1)), dtype=np.float32))
        for item in contract:
            traj.create_dataset(f'obs/agent/{item["key"]}/qpos', data=np.zeros((terminal_frames, count+2)))
        traj.create_dataset("env_states/actors/cube", data=np.zeros((4, 13)))
    return path


@pytest.mark.parametrize("count", [6, 7])
def test_native_width_and_live_joint_identity(tmp_path, count):
    path = archive(tmp_path, count)
    loaded = replay.load_archive(path)
    descriptions = loaded["original"]["action_order"]
    assert loaded["actions"].shape == (3, 14 if count == 6 else 16)
    arms = [SimpleNamespace(
        key=item["key"], names=tuple(item["arm_joint_names"]),
        all_names=tuple(item["state_joint_names"]),
        agent=SimpleNamespace(controller=SimpleNamespace(single_action_space=SimpleNamespace(shape=(count+1,)))),
    ) for item in descriptions]
    replay.validate_live_action_order(descriptions, arms)
    arms[0].all_names = tuple(reversed(arms[0].all_names))
    with pytest.raises(ValueError, match="state joint order"):
        replay.validate_live_action_order(descriptions, arms)
    with pytest.raises(ValueError, match="width"):
        replay.validate_action_order(descriptions, 16 if count == 6 else 14)


def test_missing_terminal_frame_is_rejected(tmp_path):
    path = archive(tmp_path, terminal_frames=3)
    with pytest.raises(ValueError, match=r"T\+1"):
        replay.load_archive(path)


def test_failed_archive_is_saved_without_creating_environment(tmp_path, monkeypatch):
    path = archive(tmp_path, terminal_frames=3)
    def should_not_make(*args, **kwargs):
        pytest.fail("An invalid archive must be rejected before environment construction")
    monkeypatch.setattr(replay.gym, "make", should_not_make)
    result = replay.replay(path, tmp_path / "failure")
    saved = json.loads((tmp_path / "failure" / "replay.json").read_text())
    assert saved == result
    assert result["passed"] is False
    assert result["restored_states"] == result["executed_actions"] == 0
    assert result["failures"][0]["reason"] == "replay_error"
    assert "T+1" in result["error"]


def test_state_difference_is_telemetry_without_state_writes():
    current = np.ones((1, 13), dtype=np.float32) * 4
    raw = SimpleNamespace(get_state_dict=lambda: {"actors": {"cube": current}})
    source = {"actors/cube": np.zeros((4, 13), dtype=np.float32)}
    assert replay._state_error(raw, source, 2) == 4
    source["actors/cube"] = np.zeros((4, 12))
    with pytest.raises(ValueError, match="shape"):
        replay._state_error(raw, source, 2)


def test_held_monitor_keeps_original_reference_and_rejects_loss():
    identity = np.eye(4)
    policy = RuntimePolicy()
    references = (identity.copy(), identity.copy())
    shifted = identity.copy()
    shifted[0, 3] = policy.position_slip_m * 2
    rows = replay.held_observations([identity]*2, [shifted, identity], references,
                                   [True, False], [0, 0], policy)
    assert rows[0]["violations"] == ["position_slip"]
    assert rows[1]["violations"] == []
    rows = replay.held_observations([identity]*2, [identity]*2, references,
                                   [True, False], [0, 1], policy)
    assert rows[1]["violations"] == ["grasp_lost"]
    assert np.array_equal(references[0], identity)  # No quality-reference reanchoring.
