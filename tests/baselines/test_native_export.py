"""Read actual H5 archives and verify LeRobot rows, labels and source judgments."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import pytest

from imitator.dataset.native_export import ACTION_KEY, STATE_KEY, export_native, load_episode, recorded_contract


def archive(tmp_path, *, certified=False, state_per_arm=8, arm_dof=6, camera=True):
    path = tmp_path / "source.h5"
    T = 3
    order = [dict(key=side, size=arm_dof + 1,
                  arm_joint_names=[f"joint{i}" for i in range(arm_dof)],
                  state_joint_names=[f"joint{i}" for i in range(state_per_arm)],
                  arm_coordinate_unit="rad", gripper_coordinate_unit="m")
             for side in ("left", "right")]
    contract = dict(env_id="DualCubePlace-v1", action_order=order, robot_uids=["piper"] * 2,
                    control_mode="pd_joint_pos", control_freq=20, model_fingerprints=["a", "a"],
                    mounts=[[0] * 7, [0] * 7])
    qpos = np.arange((T + 1) * state_per_arm * 2, dtype=np.float32).reshape(T + 1, -1) * .01
    clean = np.arange(T * (arm_dof + 1) * 2, dtype=np.float32).reshape(T, -1) * .01
    executed = clean + .25 if certified else clean
    episode = dict(episode_id=0, episode_seed=17, success=True,
                   imsim=dict(certified=True, verdicts={"test": {"success": True}}, phys={"mass_scale": 1.1}))
    metadata = dict(env_info={"env_id": "DualCubePlace-v1", "env_kwargs": {}}, episodes=[episode])
    if certified:
        metadata["imsim"] = dict(native_data_contract=contract, contracts={"test": {"sha256": "frozen"}},
                                 layout=dict(obs_slices={"agent/left/qpos": [0, state_per_arm],
                                     "agent/right/qpos": [state_per_arm, state_per_arm * 2],
                                     "extra/object": [state_per_arm * 2, state_per_arm * 2 + 2]}))
    else:
        metadata["native_data_contract"] = contract
    with h5py.File(path, "w") as h5:
        group = h5.create_group("traj_0")
        group["actions"] = executed
        if certified:
            group["actions_clean"] = clean
            group["obs"] = np.column_stack([qpos, np.full((T + 1, 2), 9999, np.float32)])
        else:
            for i, spec in enumerate(order):
                group[f"obs/agent/{spec['key']}/qpos"] = qpos[:, i * state_per_arm:(i + 1) * state_per_arm]
        if camera:
            images = np.stack([np.full((32, 48, 3), i * 30, np.uint8) for i in range(T + 1)])
            group["rgb" if certified else "obs/sensor_data/zed2i/rgb"] = images
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path, metadata, contract, qpos, clean, executed


@pytest.mark.parametrize("state_per_arm,arm_dof", [(9, 7), (8, 6), (12, 6)])
def test_recordepisode_export_keeps_native_dimensions_and_timestamps(tmp_path, state_per_arm, arm_dof):
    from imitator.dataset.lerobot_dataset import LeRobotDataset

    source, metadata, contract, qpos, labels, executed = archive(
        tmp_path, state_per_arm=state_per_arm, arm_dof=arm_dof)
    result = export_native(source, tmp_path / "out", "native")
    dataset = LeRobotDataset(repo_id="native", root=tmp_path / "out/native", video_backend="pyav")
    np.testing.assert_array_equal(np.stack(dataset.hf_dataset[STATE_KEY]), qpos[:-1])
    np.testing.assert_array_equal(np.stack(dataset.hf_dataset[ACTION_KEY]), labels)
    np.testing.assert_allclose(np.array(dataset.hf_dataset["timestamp"]), np.arange(3) / 20, atol=1e-8)
    assert dataset.meta.info["native_data_contract"] == contract
    assert dataset.meta.info["terminal_observation"] == "dropped"
    assert dataset.meta.info["splits"]["train"] == []
    assert result["episodes"][0]["source_episode"] == metadata["episodes"][0]
    assert dataset[2]["observation.images.zed2i"].shape == (3, 32, 48)


def test_certified_flat_state_exports_only_qpos_and_clean_labels(tmp_path):
    from imitator.dataset.lerobot_dataset import LeRobotDataset

    source, metadata, contract, qpos, clean, executed = archive(tmp_path, certified=True)
    result = export_native(source, tmp_path / "out", "certified")
    dataset = LeRobotDataset(repo_id="certified", root=tmp_path / "out/certified", video_backend="pyav")
    np.testing.assert_array_equal(np.stack(dataset.hf_dataset[STATE_KEY]), qpos[:-1])
    np.testing.assert_array_equal(np.stack(dataset.hf_dataset[ACTION_KEY]), clean)
    assert not np.array_equal(clean, executed)
    assert dataset.meta.info["source_imsim"] == metadata["imsim"]
    assert result["episodes"][0]["label_key"] == "actions_clean"
    assert result["episodes"][0]["source_episode"]["imsim"]["certified"]


def test_missing_public_slice_never_falls_back_to_hidden_full_state(tmp_path):
    source, metadata, contract, *_ = archive(tmp_path, certified=True)
    del metadata["imsim"]["layout"]["obs_slices"]["agent/left/qpos"]
    with h5py.File(source) as h5:
        with pytest.raises(ValueError, match="full hidden state is never exported"):
            load_episode(h5["traj_0"], metadata, contract)


def test_missing_contract_and_multijoint_hands_are_explicitly_unsupported(tmp_path):
    source, metadata, contract, *_ = archive(tmp_path)
    unknown = deepcopy(metadata)
    del unknown["native_data_contract"]
    with pytest.raises(ValueError, match="provide --contract"):
        recorded_contract(unknown, 14)
    hand = deepcopy(contract)
    hand["action_order"][0]["size"] = 22
    with pytest.raises(ValueError, match="Allegro"):
        recorded_contract(dict(native_data_contract=hand), 29)


def test_converter_rejects_retiming_missing_rgb_and_existing_output(tmp_path):
    source, *_ = archive(tmp_path, camera=False)
    with pytest.raises(ValueError, match="control frequency"):
        export_native(source, tmp_path / "out", "wrong-fps", fps=30)
    with pytest.raises(ValueError, match="recorded RGB"):
        export_native(source, tmp_path / "out", "no-rgb")
    (tmp_path / "out/existing").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        export_native(source, tmp_path / "out", "existing")


@pytest.mark.parametrize("cloth,worlds,action_dim,supported", [
    (False, 1, 7, True), (True, 1, 7, False), (False, 2, 7, False), (False, 1, 22, False)])
def test_data_engine_preserves_layout_and_captures_only_supported_live_contracts(
        tmp_path, monkeypatch, cloth, worlds, action_dim, supported):
    from imitator.data import engine, store
    from imitator.embodiments import native

    (tmp_path / "shards").mkdir()
    raw = SimpleNamespace(spec=SimpleNamespace(id="DualCubePlace-v1", kwargs={}, max_episode_steps=200),
                          num_envs=worlds, control_freq=20, sim_freq=100,
                          imsim={"task": "DualCubePlace-v1", "embodiment": "dual_piper"})
    env = SimpleNamespace(unwrapped=raw)
    layout = store.Layout(["left", "right"], [action_dim] * 2, [(0, 6)] * 2, [list(range(6))] * 2)
    calls = []

    def contract(environment, cameras):
        calls.append(environment)
        return {"model_fingerprints": ["recorded-live-model"] * 2, "cameras": list(cameras)}

    monkeypatch.setattr(native, "native_contract", contract)
    engine._write_env_info(str(tmp_path), env, layout, has_cloth=cloth)
    saved = json.loads((tmp_path / "shards/env_info.json").read_text())
    assert saved["layout"] == store.jsonable(layout.to_json())
    assert saved["imsim_make"] == raw.imsim
    assert saved["native_export"]["status"] == ("supported" if supported else "unsupported")
    assert bool(calls) == supported
    if supported:
        assert saved["native_data_contract"]["model_fingerprints"] == ["recorded-live-model"] * 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
