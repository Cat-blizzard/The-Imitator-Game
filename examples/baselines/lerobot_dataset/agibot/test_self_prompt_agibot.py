#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.baselines.lerobot_dataset.lerobot_self_prompt_dataset import (  # noqa: E402
    SelfPromptRobotDataConfig,
    SelfPromptRobotDataset,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="/mnt/sdb/agi_data/data_lerobot_v3/agibotworld")
    p.add_argument("--repo-id", type=str, default="agibotworld")
    p.add_argument("--video-backend", type=str, default="torchcodec")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--no-normalize", action="store_true")
    args = p.parse_args()

    info_path = Path(args.root) / "meta" / "info.json"
    print("info:", info_path)
    info = json.loads(info_path.read_text())
    print("version:", info.get("codebase_version"))
    print("episodes:", info.get("total_episodes"))
    print("frames:", info.get("total_frames"))
    print("tasks:", info.get("total_tasks"))

    ds = SelfPromptRobotDataset(
        SelfPromptRobotDataConfig(
            root=args.root,
            repo_id=args.repo_id,
            video_backend=args.video_backend,
            enable_augmentation=not args.no_augmentation,
            normalize_state=not args.no_normalize,
            normalize_action=not args.no_normalize,
            debug=True,
        )
    )

    sample = ds[args.index]
    print("\n--- sample ---")
    print("task_id:", sample["task_id"])
    print("episode_uid:", sample["episode_uid"])
    print("dataset_idx:", sample["dataset_idx"].item())
    print("prompt_video:", tuple(sample["prompt_video"].shape))
    print("states:", tuple(sample["robot_obs"]["states"].shape))
    print("actions:", tuple(sample["robot_actions"].shape))
    for key, value in sample["robot_obs"].items():
        if key.startswith("view_"):
            print(f"{key}:", tuple(value.shape))

    assert sample["robot_obs"]["states"].shape[-1] == 16
    assert sample["robot_actions"].shape[-1] == 16
    assert sample["prompt_video"].shape[0] == 10
    assert torch.isfinite(sample["robot_actions"]).all()
    print("\nPASS")


if __name__ == "__main__":
    main()
