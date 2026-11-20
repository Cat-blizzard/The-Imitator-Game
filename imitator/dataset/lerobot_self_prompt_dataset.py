"""Same-episode robot-video prompt dataset for HyperICL pretraining.

One LeRobot episode supplies both sides of a training example:

    prompt video from the episode
    query observation at one timestep -> future action chunk

No human/robot task pairing file is required.  Compatibility aliases
``human_video`` and ``human_repo_id`` are emitted so the current HyperICL
collate/DINO-cache path can be reused while the codebase migrates to the more
generic prompt_* naming.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from imitator.dataset.lerobot_dataset import LeRobotDataset


DEFAULT_STATE_KEYS = (
    "observation.states.joint.position",
    "observation.states.effector.position",
)
DEFAULT_ACTION_KEYS = (
    "actions.joint.position",
    "actions.effector.position",
)


@dataclass
class SelfPromptRobotDataConfig:
    root: str = "/mnt/sdb/agi_data/data_lerobot_v3/agibotworld"
    repo_id: str = "agibotworld"
    episodes: Optional[List[int]] = None

    state_keys: List[str] = field(default_factory=lambda: list(DEFAULT_STATE_KEYS))
    action_keys: List[str] = field(default_factory=lambda: list(DEFAULT_ACTION_KEYS))

    cameras: List[str] = field(default_factory=lambda: ["head", "hand_left", "hand_right"])
    prompt_camera: str = "head"
    image_size: Tuple[int, int] = (224, 224)

    horizon: int = 16
    obs_horizon: int = 1
    fps: Optional[int] = None

    prompt_num_frames: int = 10
    prompt_sampling: str = "uniform_jitter"  # uniform | uniform_jitter
    prompt_max_jitter: int = 3
    prompt_query_gap: int = 16
    prompt_disjoint: bool = True

    video_backend: str = "torchcodec"
    tolerance_s: float = 1e-4
    enable_augmentation: bool = True

    normalize_state: bool = True
    normalize_action: bool = True
    normalization_method: str = "bounds_q99"  # bounds_q99 | bounds

    skip_prompt_video: bool = False
    require_full_action_horizon: bool = True
    debug: bool = False


class SelfPromptRobotDataset(Dataset):
    def __init__(self, config: Optional[SelfPromptRobotDataConfig] = None):
        super().__init__()
        self.config = config or SelfPromptRobotDataConfig()
        self.root = Path(self.config.root).expanduser().resolve()

        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing LeRobot meta/info.json: {info_path}")
        info = json.loads(info_path.read_text())
        if not str(info.get("codebase_version", "")).startswith("v3"):
            raise ValueError(
                f"Expected LeRobot v3.x, found {info.get('codebase_version')!r}"
            )

        self.fps = int(self.config.fps or info.get("fps", 30))
        if self.fps <= 0:
            raise ValueError(f"Invalid FPS: {self.fps}")

        obs_offsets = [
            i / self.fps for i in range(-(self.config.obs_horizon - 1), 1)
        ]
        action_offsets = [i / self.fps for i in range(self.config.horizon)]

        delta_timestamps: Dict[str, List[float]] = {}
        for key in self.config.state_keys:
            delta_timestamps[key] = obs_offsets
        for key in self.config.action_keys:
            delta_timestamps[key] = action_offsets
        for camera in self.config.cameras:
            delta_timestamps[f"observation.images.{camera}"] = obs_offsets

        self.dataset = LeRobotDataset(
            repo_id=self.config.repo_id,
            root=self.root,
            episodes=self.config.episodes,
            delta_timestamps=delta_timestamps,
            tolerance_s=self.config.tolerance_s,
            video_backend=self.config.video_backend,
        )

        self._validate_schema()
        self._strip_unused_visual_features()
        self._setup_transforms_human(self.config)

        # API expected by the existing DINO cache.
        self.human_dataset = self
        self.skip_human_video = bool(self.config.skip_prompt_video)

        self._episode_prompt_ids: Dict[int, str] = {}
        self._prompt_id_to_episode: Dict[str, int] = {}
        self.task_to_human_indices: Dict[str, Dict[str, Any]] = {}
        self._build_episode_prompt_registry()

        self.state_dim = sum(self._feature_dim(k) for k in self.config.state_keys)
        self.action_dim = sum(self._feature_dim(k) for k in self.config.action_keys)
        self.num_tasks = int(self.dataset.meta.total_tasks)

        if self.config.debug:
            print("[SelfPromptRobotDataset]", self.get_statistics())

    def _validate_schema(self) -> None:
        available = self.dataset.meta.features
        required = [
            *self.config.state_keys,
            *self.config.action_keys,
            *(f"observation.images.{x}" for x in self.config.cameras),
            f"observation.images.{self.config.prompt_camera}",
        ]
        missing = [key for key in required if key not in available]
        if missing:
            raise KeyError(
                f"Dataset schema is missing {missing}. Available={sorted(available)}"
            )

    def _strip_unused_visual_features(self) -> None:
        keep = {
            *(f"observation.images.{x}" for x in self.config.cameras),
            f"observation.images.{self.config.prompt_camera}",
        }
        for key in list(self.dataset.meta.features):
            spec = self.dataset.meta.features[key]
            if spec.get("dtype") in ("image", "video", "depth") and key not in keep:
                self.dataset.meta.features.pop(key)

        remove = [
            key
            for key in self.dataset.hf_dataset.column_names
            if key.startswith("observation.images.") and key not in keep
        ]
        if remove:
            self.dataset.hf_dataset = self.dataset.hf_dataset.remove_columns(remove)

    def _build_episode_prompt_registry(self) -> None:
        episodes = sorted(
            {
                int(x.item() if isinstance(x, torch.Tensor) else x)
                for x in self.dataset.hf_dataset["episode_index"]
            }
        )
        for ep in episodes:
            prompt_id = self._make_episode_uid(ep)
            self._episode_prompt_ids[ep] = prompt_id
            self._prompt_id_to_episode[prompt_id] = ep
            # dino_feature_cache.collect_training_repo_ids() traverses this map.
            self.task_to_human_indices[prompt_id] = {
                "num_episodes": 1,
                "task_name": prompt_id,
                "repos": [{"repo_id": prompt_id, "num_episodes": 1}],
            }

    def _feature_dim(self, key: str) -> int:
        shape = self.dataset.meta.features[key].get("shape", ())
        out = 1
        for d in shape:
            out *= int(d)
        return out

    def _setup_transforms_human(self, config: SelfPromptRobotDataConfig) -> None:
        """Compatibility hook also used by dino_feature_cache.py."""
        h, w = config.image_size
        if config.enable_augmentation:
            self.spatial_transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=10,
                        p=0.35,
                    ),
                    A.Resize(height=h, width=w, p=1.0),
                ]
            )
            self.rgb_transform = A.Compose(
                [
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.05,
                        p=0.5,
                    ),
                    A.Normalize(
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.spatial_transform = A.Compose([A.Resize(height=h, width=w, p=1.0)])
            self.rgb_transform = A.Compose(
                [
                    A.Normalize(
                        mean=[0.0, 0.0, 0.0],
                        std=[1.0, 1.0, 1.0],
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
            )

    def _process_video_tensor(self, video: torch.Tensor) -> torch.Tensor:
        if not isinstance(video, torch.Tensor):
            video = torch.as_tensor(video)
        if video.ndim == 3:
            video = video.unsqueeze(0)
        if video.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W], got {tuple(video.shape)}")

        processed = []
        for frame in video:
            frame = frame.detach().cpu()
            if frame.dtype == torch.uint8:
                image = frame.permute(1, 2, 0).numpy()
            else:
                image = (
                    frame.permute(1, 2, 0).numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
            image = self.spatial_transform(image=image)["image"]
            tensor = self.rgb_transform(image=image)["image"]
            processed.append(tensor)

        return (
            torch.stack(processed, dim=0)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    def _make_episode_uid(self, ep: int) -> str:
        return f"{self.config.repo_id}::ep_{int(ep):06d}"

    def _relative_index_from_absolute(self, absolute: int) -> int:
        mapping = getattr(self.dataset, "_absolute_to_relative_idx", None)
        if mapping is None:
            return int(absolute)
        return int(mapping[int(absolute)])

    def _episode_bounds(self, ep: int) -> Tuple[int, int]:
        row = self.dataset.meta.episodes[int(ep)]
        return int(row["dataset_from_index"]), int(row["dataset_to_index"])

    def _safe_query_relative_index(self, idx: int) -> Tuple[int, int, int]:
        row = self.dataset.hf_dataset[int(idx)]
        ep = int(row["episode_index"].item() if hasattr(row["episode_index"], "item") else row["episode_index"])
        absolute = int(row["index"].item() if hasattr(row["index"], "item") else row["index"])

        if not self.config.require_full_action_horizon:
            return int(idx), ep, absolute

        start, end = self._episode_bounds(ep)
        if end - start >= self.config.horizon:
            absolute = min(absolute, end - self.config.horizon)
            return self._relative_index_from_absolute(absolute), ep, absolute
        return int(idx), ep, absolute

    @staticmethod
    def _virtual_to_absolute(value: int, intervals: Sequence[Tuple[int, int]]) -> int:
        remaining = int(value)
        for lo, hi in intervals:
            size = hi - lo + 1
            if remaining < size:
                return lo + remaining
            remaining -= size
        return intervals[-1][1]

    def _sample_prompt_absolute_indices(
        self,
        ep: int,
        query_absolute: Optional[int],
    ) -> List[int]:
        start, end = self._episode_bounds(ep)
        last = end - 1
        intervals: List[Tuple[int, int]] = [(start, last)]

        if self.config.prompt_disjoint and query_absolute is not None:
            query_lo = query_absolute - (self.config.obs_horizon - 1)
            query_hi = query_absolute + (self.config.horizon - 1)
            forbidden_lo = query_lo - self.config.prompt_query_gap
            forbidden_hi = query_hi + self.config.prompt_query_gap
            intervals = []
            if start <= forbidden_lo - 1:
                intervals.append((start, min(last, forbidden_lo - 1)))
            if forbidden_hi + 1 <= last:
                intervals.append((max(start, forbidden_hi + 1), last))
            if not intervals:
                intervals = [(start, last)]

        total = sum(hi - lo + 1 for lo, hi in intervals)
        if total <= 0:
            raise RuntimeError(f"Episode {ep} has no frames")

        n = self.config.prompt_num_frames
        coordinates = np.linspace(0, total - 1, n).round().astype(np.int64)
        if self.config.prompt_sampling == "uniform_jitter" and total > n:
            max_jitter = min(
                self.config.prompt_max_jitter,
                max(0, total // max(2 * n, 1)),
            )
            if max_jitter > 0:
                coordinates += np.random.randint(-max_jitter, max_jitter + 1, size=n)
                coordinates = np.clip(coordinates, 0, total - 1)
                coordinates.sort()
        elif self.config.prompt_sampling != "uniform":
            raise ValueError(
                f"prompt_sampling must be uniform or uniform_jitter, got {self.config.prompt_sampling!r}"
            )

        return [
            self._virtual_to_absolute(int(x), intervals) for x in coordinates.tolist()
        ]

    def _timestamps_for_absolute_indices(self, absolute: Sequence[int]) -> List[float]:
        relative = [self._relative_index_from_absolute(x) for x in absolute]
        values = self.dataset.hf_dataset[relative]["timestamp"]
        return [float(x.item() if isinstance(x, torch.Tensor) else x) for x in values]

    def _decode_prompt_video(
        self,
        ep: int,
        query_absolute: Optional[int],
    ) -> torch.Tensor:
        absolute = self._sample_prompt_absolute_indices(ep, query_absolute)
        timestamps = self._timestamps_for_absolute_indices(absolute)
        key = f"observation.images.{self.config.prompt_camera}"
        frames = self.dataset._query_videos({key: timestamps}, int(ep))[key]
        return self._process_video_tensor(frames)

    @staticmethod
    def _as_temporal_tensor(value: Any) -> torch.Tensor:
        x = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        x = x.float()
        if x.ndim == 0:
            return x.reshape(1, 1)
        if x.ndim == 1:
            return x.unsqueeze(0)
        if x.ndim > 2:
            return x.flatten(start_dim=1)
        return x

    def _normalize_feature(self, x: torch.Tensor, key: str) -> torch.Tensor:
        stats = self.dataset.meta.stats.get(key) if self.dataset.meta.stats else None
        if not stats:
            return x

        if self.config.normalization_method == "bounds_q99":
            low, high = stats.get("q01"), stats.get("q99")
        elif self.config.normalization_method == "bounds":
            low, high = stats.get("min"), stats.get("max")
        else:
            raise ValueError(self.config.normalization_method)

        if low is None or high is None:
            low, high = stats.get("min"), stats.get("max")
        if low is None or high is None:
            return x

        low_t = torch.as_tensor(low, dtype=x.dtype).flatten()
        high_t = torch.as_tensor(high, dtype=x.dtype).flatten()
        if low_t.numel() != x.shape[-1] or high_t.numel() != x.shape[-1]:
            raise ValueError(
                f"Stats mismatch for {key}: feature={x.shape[-1]}, "
                f"low={low_t.numel()}, high={high_t.numel()}"
            )
        denom = high_t - low_t
        denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
        return 2.0 * (x - low_t) / denom - 1.0

    def _concat_features(
        self,
        item: Dict[str, Any],
        keys: Sequence[str],
        normalize: bool,
    ) -> torch.Tensor:
        parts = []
        for key in keys:
            x = self._as_temporal_tensor(item[key])
            if normalize:
                x = self._normalize_feature(x, key)
            parts.append(x)
        lengths = {x.shape[0] for x in parts}
        if len(lengths) != 1:
            raise ValueError(f"Temporal mismatch: {[tuple(x.shape) for x in parts]}")
        return torch.cat(parts, dim=-1)

    def _get_target_item(self, prompt_id: str, task_idx: int = None) -> Dict[str, Any]:
        """Interface consumed by the existing DINO feature-cache precompute."""
        del task_idx
        ep = self._prompt_id_to_episode[prompt_id]
        video = self._decode_prompt_video(ep, query_absolute=None)
        return {
            "video": video,
            "task": prompt_id,
            "task_id": torch.tensor(ep, dtype=torch.long),
            "episode_idx": torch.tensor(ep, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        relative, ep, absolute = self._safe_query_relative_index(int(idx))
        item = self.dataset[relative]

        state = self._concat_features(
            item, self.config.state_keys, self.config.normalize_state
        )
        actions = self._concat_features(
            item, self.config.action_keys, self.config.normalize_action
        )

        robot_obs: Dict[str, torch.Tensor] = {"states": state.float()}
        for view_idx, camera in enumerate(self.config.cameras, start=1):
            key = f"observation.images.{camera}"
            robot_obs[f"view_{view_idx}"] = self._process_video_tensor(item[key]).float()

        task_index_value = item.get("task_index", torch.tensor(0))
        task_index = int(
            task_index_value.item()
            if isinstance(task_index_value, torch.Tensor)
            else task_index_value
        )
        task_text = str(item.get("task", f"task_{task_index}"))
        prompt_id = self._episode_prompt_ids[ep]
        frame_value = item.get("frame_index", torch.tensor(-1))
        frame_index = int(frame_value.item() if isinstance(frame_value, torch.Tensor) else frame_value)

        result: Dict[str, Any] = {
            "robot_obs": robot_obs,
            "robot_actions": actions.float(),
            "dataset_idx": torch.tensor(task_index, dtype=torch.long),
            "task_index": torch.tensor(task_index, dtype=torch.long),
            "task_id": task_text,
            "robot_task_id": task_text,
            "episode_id": ep,
            "episode_uid": prompt_id,
            "frame_index": frame_index,
            "prompt_id": prompt_id,
            # Legacy name consumed by the current HyperICL collate/cache.
            "human_repo_id": prompt_id,
        }

        if not self.skip_human_video:
            prompt = self._decode_prompt_video(ep, absolute)
            result["prompt_video"] = prompt
            result["human_video"] = prompt

        return result

    def get_task_list(self) -> List[str]:
        return [str(x) for x in self.dataset.meta.tasks.index.tolist()]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "num_frames": len(self),
            "num_episodes": len(self._episode_prompt_ids),
            "num_tasks": self.num_tasks,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cameras": list(self.config.cameras),
            "prompt_camera": self.config.prompt_camera,
            "prompt_num_frames": self.config.prompt_num_frames,
            "prompt_disjoint": self.config.prompt_disjoint,
            "prompt_query_gap": self.config.prompt_query_gap,
        }


def collate_self_prompt_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "robot_actions": torch.stack([x["robot_actions"] for x in batch]),
        "dataset_idx": torch.stack([x["dataset_idx"] for x in batch]),
        "task_id": [x["task_id"] for x in batch],
        "episode_uid": [x["episode_uid"] for x in batch],
        "prompt_id": [x["prompt_id"] for x in batch],
        "human_repo_id": [x["prompt_id"] for x in batch],
    }
    obs_keys = batch[0]["robot_obs"].keys()
    out["robot_obs"] = {
        key: torch.stack([x["robot_obs"][key] for x in batch])
        for key in obs_keys
    }
    if "prompt_video" in batch[0]:
        out["prompt_video"] = torch.stack([x["prompt_video"] for x in batch])
        out["human_video"] = out["prompt_video"]
    return out
