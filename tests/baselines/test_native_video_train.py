"""Training configuration must preserve native dimensions and task pairing."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import ConcatDataset

from examples.baselines.act import train_act_native_video as training
from examples.baselines.act.native_video_data import NativeVideoDataset
from examples.baselines.act.train_act_native_video import (
    TrainingArgs, model_configuration, recorded_contract, robot_task_id, train, training_sampler,
)


def test_native_dimensions_own_the_policy_and_json_selects_video_encoder(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"state_dim": 18, "action_dim": 16,
                                "frozen_backbone_type": "dinov2_vitl14",
                                "hf_cache_dir": "/models/hf/hub"}))
    args = TrainingArgs(model_config=str(path), state_dim=24, action_dim=14,
                        cameras=["zed2i"], pred_horizon=16, video_backend="opencv")
    model = model_configuration(args)
    assert (model.state_dim, model.action_dim, model.pred_horizon) == (24, 14, 16)
    assert model.frozen_backbone_type == "dinov2_vitl14"
    assert model.hf_cache_dir == "/models/hf/hub"
    assert model.video_backend == "opencv" and model.input_mode == "video_only"


def test_task_pairing_uses_recorded_task_and_rejects_silent_multitask_mixing():
    dataset = SimpleNamespace(main_dataset=SimpleNamespace(meta=SimpleNamespace(
        info={"native_data_contract": {"env_id": "TwoRobotPickPillToRegions-v1"}})))
    assert robot_task_id([{"repo_id": "arx-pills"}], dataset) == "TwoRobotPickPillToRegions-v1"
    with pytest.raises(ValueError, match="one robot dataset/task"):
        robot_task_id([{"env_id": "task-a"}, {"env_id": "task-b"}], dataset)
    with pytest.raises(ValueError, match="recorded native task"):
        robot_task_id([{"env_id": "another-task"}], dataset)
    dataset.main_dataset.meta.info.clear()
    with pytest.raises(ValueError, match="identify its task"):
        robot_task_id([{"repo_id": "unidentified"}], dataset)


def test_training_refuses_to_overwrite_an_existing_run_before_loading_data(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("preserve this run\n")
    with pytest.raises(FileExistsError):
        train(TrainingArgs(output_dir=str(tmp_path), human_manifest="unused.json", cuda=False))
    assert (tmp_path / "metrics.jsonl").read_text() == "preserve this run\n"


class RobotTaskFixture:
    """Native rows/metadata with different per-task normalizer ranges."""

    def __init__(self, repo_id: str, env_id: str, length: int) -> None:
        self.repo_id, self.length = repo_id, length
        self.episodes = [0, 1]
        self.meta = SimpleNamespace(info={
            "fps": 20, "splits": {"train": [0, 1], "validation": [2]},
            "native_data_contract": {"env_id": env_id, "robot_uids": ["piper", "piper"],
                "action_order": [{"key": "piper-0", "size": 7}, {"key": "piper-1", "size": 7}],
                "control_freq": 20, "control_mode": "pd_joint_pos", "task_mounts": env_id},
        }, features={
            "state": {"shape": [16], "names": [f"qpos-{i}" for i in range(16)]},
            "action": {"shape": [14], "names": [f"action-{i}" for i in range(14)]},
            "observation.images.zed2i": {"shape": [3, 32, 32]},
        })

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {"view_1": torch.full((1, 3, 32, 32), index / max(self.length, 1)),
                "states": torch.zeros(1, 16), "actions": torch.zeros(4, 14),
                "action_is_pad": torch.tensor([False, False, True, True]), "repo_id": self.repo_id}


class RobotDatasetFixture:
    def __init__(self) -> None:
        self.sub_datasets = [RobotTaskFixture("pill", "pill-task", 2),
                             RobotTaskFixture("fruit", "fruit-task", 6)]
        self.main_dataset = self.sub_datasets[0]
        self.lerobot_dataset = ConcatDataset(self.sub_datasets)
        self.state_key, self.action_key = "state", "action"
        self.normalizer = SimpleNamespace(
            dataset_info={index: {"repo_id": sub.repo_id, "state_key": "state", "action_key": "action"}
                          for index, sub in enumerate(self.sub_datasets)},
            dataset_stats={0: {"action": {"q01": [0.0] * 14, "q99": [1.0] * 14}},
                           1: {"action": {"q01": [2.0] * 14, "q99": [3.0] * 14}}})

    def __len__(self) -> int:
        return len(self.lerobot_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        dataset_idx = 0 if index < 2 else 1
        sample = dict(self.lerobot_dataset[index])
        sample["dataset_idx"] = torch.tensor(dataset_idx)
        return sample


@pytest.fixture
def multi_inputs(tmp_path):
    """Two selected train datasets and two actual locally encoded human clips."""
    config_path = tmp_path / "datasets.json"
    config_path.write_text(json.dumps([{"repo_id": "pill", "env_id": "pill-task"},
                                       {"repo_id": "fruit", "env_id": "fruit-task"}]))
    path = tmp_path / "human.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    assert writer.isOpened()
    for value in [20, 100, 220]:
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()
    manifest_path = tmp_path / "humans.json"
    manifest_path.write_text(json.dumps({"schema_version": 2, "tasks": [
        {"task_id": task, "demos": [{"id": f"{task}-train", "repo_id": human,
          "episode_id": 0, "camera": "zed2i", "path": str(path), "role": "train", "num_frames": 3}]}
        for task, human in [("pill-task", "human_H31"), ("fruit-task", "human_H8")]]}))
    args = TrainingArgs(sub_config=str(config_path), human_manifest=str(manifest_path),
                        cameras=["zed2i"], image_size=[32, 32], state_dim=16, action_dim=14,
                        pred_horizon=4, video_backend="opencv", cuda=False, num_workers=0,
                        hidden_dim=32, nheads=4, dim_feedforward=64, enc_layers=1, dec_layers=1,
                        dropout=0.0, batch_size=2, max_updates=2, log_freq=1,
                        sampling_policy="equal_task",
                        output_dir=str(tmp_path / "training"))
    return args, RobotDatasetFixture()


def test_each_task_keeps_its_native_physics_and_train_only_normalizer(multi_inputs):
    args, base = multi_inputs
    contract = recorded_contract(args, base)
    assert "native" not in contract and "source_image_size" not in contract
    assert [record["env_id"] for record in contract["datasets"]] == ["pill-task", "fruit-task"]
    assert contract["datasets"][1]["native"]["task_mounts"] == "fruit-task"
    assert all(record["train_episodes"] == [0, 1] for record in contract["datasets"])
    assert all(record["normalizer_source"] == "selected_train_rows" for record in contract["datasets"])
    # Reordering config files cannot reorder the loaded dataset/stat identities.
    config_path = Path(args.sub_config)
    configs = json.loads(config_path.read_text())
    config_path.write_text(json.dumps(configs[::-1]))
    assert recorded_contract(args, base)["datasets"] == contract["datasets"]
    base.sub_datasets = base.sub_datasets[:1]
    single = recorded_contract(args, base)
    assert single["native"] == contract["datasets"][0]["native"]
    assert single["source_image_size"] == [32, 32]


@pytest.mark.parametrize("mutation,error", [
    (lambda info, features: info["native_data_contract"].update(env_id="wrong-task"), "recorded native task"),
    (lambda info, features: info["native_data_contract"].update(robot_uids=["panda", "panda"]), "same embodiment"),
    (lambda info, features: features["action"]["names"].reverse(), "ordering"),
    (lambda info, features: features["state"].update(shape=[18]), "dimensions"),
    (lambda info, features: features["observation.images.zed2i"].update(shape=[3, 64, 32]), "camera shape"),
    (lambda info, features: info.update(fps=30), "control frequency"),
])
def test_incompatible_task_contracts_fail_before_training(multi_inputs, mutation, error):
    args, base = multi_inputs
    mutation(base.sub_datasets[1].meta.info, base.sub_datasets[1].meta.features)
    with pytest.raises(ValueError, match=error):
        recorded_contract(args, base)


def test_equal_task_sampler_records_mass_not_frame_ratio_and_is_reproducible(multi_inputs):
    args, base = multi_inputs
    dataset = NativeVideoDataset(base, args.human_manifest, dataset_tasks={0: "pill-task", 1: "fruit-task"},
                                 image_size=[32, 32], video_backend="opencv")
    sampler, receipt = training_sampler(dataset, 51, "equal_task")
    assert sampler.weights[:2].sum() == pytest.approx(0.5)
    assert sampler.weights[2:].sum() == pytest.approx(0.5)
    assert receipt["task_probabilities"] == {"pill-task": 0.5, "fruit-task": 0.5}
    assert receipt["replacement"]
    control, _ = training_sampler(dataset, 51, "equal_task")
    first, second = list(sampler), list(control)
    assert first == second and len(first) == 8
    # Independent policy/dropout RNG cannot change the next epoch's sampler.
    torch.rand(100)
    assert list(sampler) == list(control)
    uniform, uniform_receipt = training_sampler(dataset, 51, "uniform_window")
    assert sorted(uniform) == list(range(8))
    assert uniform_receipt["task_probabilities"] == {"pill-task": 0.25, "fruit-task": 0.75}


def test_one_task_can_have_multiple_datasets_without_changing_its_total_mass(multi_inputs):
    args, base = multi_inputs
    base.sub_datasets.append(RobotTaskFixture("pill-extra", "pill-task", 4))
    base.lerobot_dataset = ConcatDataset(base.sub_datasets)
    dataset = NativeVideoDataset(base, args.human_manifest,
        dataset_tasks={0: "pill-task", 1: "fruit-task", 2: "pill-task"}, video_backend="opencv")
    sampler, receipt = training_sampler(dataset, 1, "equal_task")
    assert sampler.weights[:2].sum() + sampler.weights[8:].sum() == pytest.approx(0.5)
    assert sampler.weights[2:8].sum() == pytest.approx(0.5)
    assert receipt["task_probabilities"] == {"pill-task": 0.5, "fruit-task": 0.5}


def test_two_mode_real_cpu_optimizer_checkpoint_and_sampler_agree(multi_inputs, monkeypatch, tmp_path):
    """Run the real ACT optimizer; only pretrained video weights are a fixture."""
    from examples.baselines.act.act.detr_video import backbone as visual_backbone
    from examples.baselines.encoders.task_encoder.video_backbone import FrozenVideoBackbone

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(visual_backbone, "is_main_process", lambda: False)

    def load(encoder):
        if not encoder._loaded:
            encoder._backbone = nn.Linear(3, 1024, bias=False).requires_grad_(False).eval()
            encoder._loaded = True

    def encode(encoder, video):
        raw = encoder._backbone(video.mean(dim=(1, 3, 4)))
        return raw, raw.unsqueeze(1).expand(-1, 4, -1)

    monkeypatch.setattr(FrozenVideoBackbone, "_load_backbone", load)
    monkeypatch.setattr(FrozenVideoBackbone, "_encode_dino", encode)
    args, base = multi_inputs
    monkeypatch.setattr(training, "build_dataloader", lambda args: (None, base))
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"task_latent_dim": 16, "frozen_backbone_type": "dinov2_vitl14"}))
    args.model_config = str(model_path)
    summaries = []
    try:
        for mode in ("video", "constant"):
            args.conditioning_mode = mode
            args.output_dir = str(tmp_path / mode)
            train(args)
            output = tmp_path / mode
            summary = json.loads((output / "training_summary.json").read_text())
            checkpoint = torch.load(output / "checkpoint_step_000002.pt", weights_only=False)
            receipt = json.loads((output / "first_backward.json").read_text())
            assert summary["global_step"] == 2 and checkpoint["conditioning_mode"] == mode
            assert len(checkpoint["data_contract"]["datasets"]) == len(checkpoint["dataset_stats"]) == 2
            assert checkpoint["dataset_stats"][1]["stats"]["action"]["q01"] == [2.0] * 14
            assert checkpoint["data_contract"]["datasets"][1]["native"]["task_mounts"] == "fruit-task"
            assert sum(summary["actual_sampling"]["task_windows"].values()) == 4
            assert receipt["adapter_has_gradient"] == (mode == "video")
            assert (receipt["adapter_gradient_norm"] is None) == (mode == "constant")
            assert not receipt["frozen_backbone_has_gradient"]
            summaries.append(summary)
        assert summaries[0]["initial_policy_sha256"] == summaries[1]["initial_policy_sha256"]
        assert summaries[0]["actual_sampling"] == summaries[1]["actual_sampling"]
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
