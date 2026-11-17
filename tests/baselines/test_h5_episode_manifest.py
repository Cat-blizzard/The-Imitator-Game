"""Explicit episode selection, seed isolation and failed-conversion accounting."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from examples.baselines.lerobot_dataset import h5_to_lerobot as converter


def write_h5(path, episodes=1, offset=0):
    with h5py.File(path, "w") as h5:
        for episode in range(episodes):
            trajectory = h5.create_group(f"traj_{episode}")
            trajectory["actions"] = np.full((2, 16), episode + offset, dtype=np.float32)
            for side in range(2):
                trajectory[f"obs/agent/panda_wristcam-{side}/qpos"] = np.full(
                    (3, 9), episode + offset + side, dtype=np.float32
                )
            trajectory["obs/sensor_data/zed2i/rgb"] = np.full(
                (3, 8, 8, 3), episode + offset, dtype=np.uint8
            )


def write_manifest(tmp_path, entries=None, seed_splits=None):
    manifest = {
        "schema_version": 1,
        "seed_splits": seed_splits or {"train": [10, 11], "development": [20], "evaluation": [30]},
        "episodes": entries if entries is not None else [
            {"h5_path": "first.h5", "episode_id": 0, "seed": 10, "split": "train"},
            {"h5_path": "second.h5", "episode_id": 0, "seed": 20, "split": "development"},
        ],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def writer(monkeypatch):
    from examples.baselines.lerobot_dataset import lerobot_dataset

    class Writer:
        def __init__(self, **kwargs):
            # The real writer enriches video feature dictionaries in-place.
            for feature in kwargs["features"].values():
                if feature.get("dtype") == "video":
                    feature["info"] = {"video.fps": kwargs["fps"]}
            self.root = Path(kwargs["root"])
            self.root.mkdir(parents=True)
            self.meta = SimpleNamespace(info={"fps": kwargs["fps"], "total_episodes": 0, "total_frames": 0})
            self.buffer = []
            self.episodes = []
            self.finalized = False

        def add_frame(self, frame):
            self.buffer.append(frame)

        def save_episode(self):
            if len(self.episodes) == factory.fail_episode:
                raise RuntimeError("injected encoder failure")
            self.episodes.append(self.buffer)
            self.meta.info["total_episodes"] += 1
            self.meta.info["total_frames"] += len(self.buffer)
            self.meta.info["splits"] = {"train": f"0:{len(self.episodes)}"}
            self.buffer = []

        def finalize(self):
            self.finalized = True
            (self.root / "meta").mkdir(exist_ok=True)
            (self.root / "meta/info.json").write_text(json.dumps(self.meta.info))

    def factory(**kwargs):
        factory.instance = Writer(**kwargs)
        return factory.instance

    factory.fail_episode = None
    factory.instance = None
    monkeypatch.setattr(lerobot_dataset.LeRobotDataset, "create", factory)
    return factory


def test_manifest_combines_sources_with_fixed_splits(tmp_path, writer):
    write_h5(tmp_path / "first.h5", offset=1)
    write_h5(tmp_path / "second.h5", offset=5)
    path = write_manifest(tmp_path)
    ok, name, message = converter.process_episode_manifest(
        str(path), str(tmp_path / "converted"), "version1", fps=20, drop_terminal_frame=True
    )
    assert (ok, name) == (True, "version1"), message
    dataset_dir = tmp_path / "converted/version1"
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    assert info["splits"] == {"train": [0], "validation": [1]}
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 4
    records = json.loads((dataset_dir / "meta/source_episodes.json").read_text())
    assert records["fps"] == 20 and records["terminal_observation"] == "dropped"
    assert records["seed_splits"]["evaluation"] == [30]
    assert [(episode["episode_id"], episode["seed"], episode["split"])
            for episode in records["episodes"]] == [(0, 10, "train"), (1, 20, "development")]
    for record, source in zip(records["episodes"], ("first.h5", "second.h5")):
        assert not Path(record["source_path"]).is_absolute()
        assert (dataset_dir / record["source_path"]).resolve() == tmp_path / source
        assert record["source_episode_id"] == 0
    np.testing.assert_array_equal(writer.instance.episodes[1][0]["action.qpos_gripper_actions"], np.full(16, 5))
    assert json.loads((dataset_dir / ".complete").read_text())["episodes"] == 2


@pytest.mark.parametrize("mutation,match", [
    (lambda data: data["seed_splits"]["development"].append(10), "unique, disjoint"),
    (lambda data: data["episodes"][1].update(seed=30), "not assigned"),
    (lambda data: data["episodes"][1].update(seed=10, split="train"), "only once"),
    (lambda data: data["episodes"][1].update(h5_path="first.h5"), "only once"),
    (lambda data: data.update(episodes=[]), "at least one"),
    (lambda data: data["episodes"][0].update(episode_id=True), "nonnegative episode_id"),
])
def test_manifest_rejects_leakage_and_duplicate_sources(tmp_path, mutation, match):
    path = write_manifest(tmp_path)
    content = json.loads(path.read_text())
    mutation(content)
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=match):
        converter.load_episode_manifest(path)


def test_failed_episode_stops_conversion_without_false_completion(tmp_path, writer):
    write_h5(tmp_path / "source.h5", episodes=3)
    writer.fail_episode = 1
    ok, _, message = converter.process_h5_file(
        str(tmp_path / "source.h5"), str(tmp_path / "converted"), fps=20, drop_terminal_frame=True
    )
    assert not ok and "injected encoder failure" in message
    dataset_dir = tmp_path / "converted/source"
    assert not (dataset_dir / ".complete").exists()
    failure = json.loads((dataset_dir / ".conversion-failed.json").read_text())
    assert failure["saved_episodes"] == 1
    assert failure["failed_source"]["source_episode_id"] == 1
    assert len(writer.instance.episodes) == 1
    assert len(writer.instance.buffer) == 2  # No frames from the third trajectory were appended.
    assert writer.instance.finalized
    assert not converter.check_dataset_complete(dataset_dir, tmp_path / "source.h5")[0]


def test_manifest_never_replaces_existing_output(tmp_path, writer):
    path = write_manifest(tmp_path)
    dataset_dir = tmp_path / "converted/version1"
    dataset_dir.mkdir(parents=True)
    sentinel = dataset_dir / "keep"
    sentinel.write_text("existing dataset")
    ok, _, message = converter.process_episode_manifest(str(path), str(tmp_path / "converted"), "version1")
    assert not ok and "already exists" in message
    assert sentinel.read_text() == "existing dataset"
    assert writer.instance is None


def test_no_valid_episode_does_not_get_completion_marker(tmp_path, writer):
    write_h5(tmp_path / "empty.h5", episodes=0)
    ok, _, message = converter.process_h5_file(str(tmp_path / "empty.h5"), str(tmp_path / "converted"))
    assert not ok and "no trajectory" in message
    assert writer.instance is None
    assert not (tmp_path / "converted/empty/.complete").exists()


def test_force_cannot_remove_raw_source(tmp_path, writer):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    raw = source_dir / "source.h5"
    write_h5(raw)
    ok, _, message = converter.process_h5_file(str(raw), str(tmp_path), force=True)
    assert not ok and "must not contain the source H5" in message
    with h5py.File(raw) as h5:
        assert len(h5["traj_0/actions"]) == 2
    assert writer.instance is None


def test_cli_failed_conversion_returns_nonzero(tmp_path, monkeypatch):
    manifest = write_manifest(tmp_path, entries=[])
    monkeypatch.setattr("sys.argv", [
        "h5_to_lerobot", "--episode-manifest", str(manifest), "--output-dir", str(tmp_path / "output"),
        "--name", "version1", "--fps", "20", "--drop-terminal-frame", "--no-gpu",
    ])
    assert converter.main() == 1


def write_native_h5(path, robot, arm_width):
    state_width = arm_width+2
    with h5py.File(path, 'w') as source:
        group = source.create_group('traj_0')
        group['actions'] = np.arange(4*(arm_width+1), dtype=np.float32).reshape(2, -1)
        for side in range(2):
            group[f'obs/agent/{robot}-{side}/qpos'] = np.full((3, state_width), side+1, dtype=np.float32)
        group['obs/sensor_data/base_camera/rgb'] = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    order = [{"key": f"{robot}-{side}", "size": arm_width+1,
              "arm_joint_names": [f"j{i}" for i in range(arm_width)],
              "state_joint_names": [f"j{i}" for i in range(state_width)]} for side in range(2)]
    path.with_suffix('.json').write_text(json.dumps({
        'env_info': {'env_id': 'DualArmPickPlace-v1', 'env_kwargs': {'robot_uids': [robot]*2, 'shader_dir': 'default'}},
        'embodiment_sync': {'action_order': order, 'control_freq': 20}}))
    return order


@pytest.mark.parametrize('robot,arm_width', [('piper', 6), ('panda', 7)])
def test_native_dimensions_names_camera_and_terminal_alignment(tmp_path, robot, arm_width):
    path = tmp_path/'native.h5'
    order = write_native_h5(path, robot, arm_width)
    with h5py.File(path) as source:
        sample = converter.load_trajectory_data(source, 'traj_0', drop_terminal_frame=True)
    features = converter.build_lerobot_features(sample)
    assert sample['num_frames'] == 2
    assert sample['qpos_left'].shape == (2, arm_width+2)
    assert sample['actions'].shape == (2, 2*(arm_width+1))
    assert features['observation.qpos_gripper_states']['shape'] == (2*(arm_width+2),)
    assert features['action.qpos_gripper_actions']['shape'] == (2*(arm_width+1),)
    assert features['action.qpos_gripper_actions']['names'][arm_width+1] == f'{robot}-1/j0'
    normalized = [dict(item, arm_coordinate_unit='rad', gripper_coordinate_unit='m',
                       gripper_open_action=1.0, gripper_closed_action=-1.0) for item in order]
    assert sample['data_contract']['action_order'] == normalized
    assert sample['data_contract']['rendering'] == {'shader_dir': 'default', 'sensor_configs': {}}
    assert sample['rgb_data']['base_camera'].shape == (2, 8, 8, 3)


def test_converter_rejects_wrong_recorded_frequency_and_mixed_robots(tmp_path, writer):
    path = tmp_path/'native.h5'
    write_native_h5(path, 'piper', 6)
    ok, _, message = converter.process_h5_file(str(path), str(tmp_path/'out'), fps=30)
    assert not ok and 'control frequency' in message
    assert writer.instance is None


def test_optional_native_units_normalize_without_changing_sources(tmp_path, writer):
    paths = [tmp_path/'first.h5', tmp_path/'second.h5']
    for path in paths:
        write_native_h5(path, 'panda', 7)
    sidecar = paths[1].with_suffix('.json')
    explicit = json.loads(sidecar.read_text())
    for item in explicit['embodiment_sync']['action_order']:
        item.update(arm_coordinate_unit='rad', gripper_coordinate_unit='m',
                    gripper_open_action=1., gripper_closed_action=-1.)
    sidecar.write_text(json.dumps(explicit))
    original = sidecar.read_bytes()
    manifest = write_manifest(tmp_path)
    ok, _, message = converter.process_episode_manifest(str(manifest), str(tmp_path/'out'),
                                                        'same-panda', fps=20, drop_terminal_frame=True)
    assert ok, message
    assert len(writer.instance.episodes) == 2
    assert sidecar.read_bytes() == original
    for item in explicit['embodiment_sync']['action_order']:
        item.update(gripper_coordinate_unit='rad', gripper_open_action=0., gripper_closed_action=.81)
    sidecar.write_text(json.dumps(explicit))
    with h5py.File(paths[1]) as source:
        result = converter.native_data_contract(source, 16)['action_order'][0]
    assert result['gripper_coordinate_unit'] == 'rad'
    assert result['gripper_open_action'] == 0.
    assert result['gripper_closed_action'] == .81
