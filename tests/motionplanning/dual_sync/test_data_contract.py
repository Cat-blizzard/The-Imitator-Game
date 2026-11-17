"""CPU coverage for recorded sampling rates and terminal row semantics."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from examples.baselines.act.train_act_single import _dataset_fps
from examples.baselines.lerobot_dataset import h5_to_lerobot as converter


@pytest.fixture
def trajectory(tmp_path):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5:
        traj = h5.create_group("traj_0")
        traj["actions"] = np.arange(32, dtype=np.float32).reshape(2, 16)
        for side in range(2):
            traj[f"obs/agent/panda_wristcam-{side}/qpos"] = np.arange(27, dtype=np.float32).reshape(3, 9) + side * 100
        traj["obs/sensor_data/zed2i/rgb"] = np.broadcast_to(
            np.arange(3, dtype=np.uint8)[:, None, None, None], (3, 8, 8, 3)
        )
    return path


def test_converter_terminal_policy_preserves_action_observation_pairs(trajectory):
    with h5py.File(trajectory) as h5:
        legacy = converter.load_trajectory_data(h5, "traj_0")
        data = converter.load_trajectory_data(h5, "traj_0", drop_terminal_frame=True)
    assert legacy["num_frames"] == 3
    assert data["num_frames"] == 2
    np.testing.assert_array_equal(data["qpos_left"], legacy["qpos_left"][:-1])
    np.testing.assert_array_equal(data["rgb_data"]["zed2i"], legacy["rgb_data"]["zed2i"][:-1])
    np.testing.assert_array_equal(data["actions"], legacy["actions"])
    assert data["qpos_right"][0, 0] == 100


def test_converter_rejects_misaligned_camera(trajectory):
    with h5py.File(trajectory, "a") as h5:
        del h5["traj_0/obs/sensor_data/zed2i/rgb"]
        h5["traj_0/obs/sensor_data/zed2i/rgb"] = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="T \\+ 1"):
            converter.load_trajectory_data(h5, "traj_0", drop_terminal_frame=True)


def test_converter_emits_two_real_rows_and_preserves_legacy_split(trajectory, tmp_path, monkeypatch):
    from examples.baselines.lerobot_dataset import lerobot_dataset

    frames = []
    meta = SimpleNamespace(info={}, splits={})

    def create(**kwargs):
        Path(kwargs["root"]).mkdir()
        meta.info["fps"] = kwargs["fps"]
        return SimpleNamespace(
            meta=meta, add_frame=frames.append, save_episode=lambda: None,
            finalize=lambda: None,
        )

    monkeypatch.setattr(lerobot_dataset.LeRobotDataset, "create", create)
    ok, _, _ = converter.process_h5_file(
        str(trajectory), str(tmp_path / "output"), fps=20, drop_terminal_frame=True
    )
    assert ok
    assert meta.info == {"fps": 20, "terminal_observation": "dropped"}
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[1]["action.qpos_gripper_actions"], np.arange(16, 32))
    assert meta.splits["train"] == []  # Existing 95% single-episode semantics stay unchanged.


def write_dataset(root, name, fps):
    info = root / name / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text(json.dumps({"fps": fps, "total_episodes": 1}))
    return {"repo_id": name, "root": name}


@pytest.mark.parametrize("fps", [20, 30])
def test_trainer_uses_recorded_fps(tmp_path, fps):
    config = tmp_path / "datasets.json"
    config.write_text(json.dumps([write_dataset(tmp_path, "data", fps)]))
    assert _dataset_fps(str(tmp_path), str(config)) == fps


def test_trainer_rejects_mixed_rates(tmp_path):
    config = tmp_path / "datasets.json"
    config.write_text(json.dumps([write_dataset(tmp_path, "first", 20), write_dataset(tmp_path, "second", 30)]))
    with pytest.raises(ValueError, match="one dataset sampling rate"):
        _dataset_fps(str(tmp_path), str(config))


def test_converter_does_not_reuse_wrong_terminal_or_fps(tmp_path):
    write_dataset(tmp_path, "data", 30)
    assert converter.check_dataset_complete(tmp_path / "data")[0]
    assert not converter.check_dataset_complete(tmp_path / "data", fps=20)[0]
    assert not converter.check_dataset_complete(tmp_path / "data", drop_terminal_frame=True)[0]
