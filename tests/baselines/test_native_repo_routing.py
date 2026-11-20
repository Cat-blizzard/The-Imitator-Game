"""Actual export and video batches keep dataset identity separate from task text."""
import json
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from examples.baselines.act.native_video_data import NativeVideoDataset
from imitator.dataset.lerobot_sim_dataset import LeRobotSimDataConfig, LeRobotSimDataset
from imitator.dataset.native_export import export_native
from tests.baselines.test_native_export import archive


@pytest.mark.parametrize("repos", [["piper-pill"], ["piper-pill-a", "piper-pill-b"]])
def test_export_repo_names_differ_from_task_without_breaking_real_video_batches(tmp_path, repos):
    source, _, _, *_ = archive(tmp_path)
    output = tmp_path / "datasets"
    for repo in repos:
        export_native(source, output, repo)
    config = tmp_path / "datasets.json"
    config.write_text(json.dumps([{"repo_id": repo, "root": repo, "train": [0]} for repo in repos]))
    base = LeRobotSimDataset(LeRobotSimDataConfig(
        root=str(output), dataset_file=str(config), cameras=["zed2i"], image_size=(16, 16),
        horizon=2, fps=20, video_backend="torchcodec", tolerance_s=.001,
        enable_augmentation=False, normalization_stats_source="train_rows"))
    video = tmp_path / "human.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (16, 16))
    assert writer.isOpened()
    for value in (20, 60, 100):
        writer.write(np.full((16, 16, 3), value, np.uint8))
    writer.release()
    task = "DualCubePlace-v1"
    manifest = {"schema_version": 1, "task_id": task, "demos": [
        {"id": "human-train", "repo_id": "human-fixture", "episode_id": 0,
         "camera": "zed2i", "path": str(video), "role": "train",
         "episode_frame_range": [0, 3], "frame_indices": [0, 1, 2]}]}
    paired = NativeVideoDataset(base, manifest, role="train", image_size=(16, 16),
                                video_backend="opencv", dataset_tasks={i: task for i in range(len(repos))})
    indices = [0] + list(base.lerobot_dataset.cumulative_sizes[:-1])
    batch = next(iter(DataLoader(Subset(paired, indices), batch_size=len(indices))))
    assert batch["repo_id"] == repos
    assert batch["dataset_idx"].tolist() == list(range(len(repos)))
    assert batch["task_name"] == [task] * len(repos)
    assert batch["states"].shape == (len(repos), 1, 16)
    assert batch["actions"].shape == (len(repos), 2, 14)
    assert batch["human_video"].shape == (len(repos), 3, 3, 16, 16)
    assert torch.isfinite(batch["states"]).all() and torch.isfinite(batch["actions"]).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
