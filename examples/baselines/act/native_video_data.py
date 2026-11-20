"""Pair native robot tasks with explicitly selected human video clips.

The robot dataset owns observation/action ordering, normalization and padding.
This wrapper only adds ``human_video`` (T, 3, H, W), float32 in [0, 1], and a
content-and-sampling identity. Human and robot frames are not time aligned.

Manifest v1 contains ``task_id`` and ``demos``. Each demo names ``id``, ``path``,
``repo_id``, ``episode_id``, ``camera``, ``role`` (train/development/evaluation),
and either ``frame_indices`` or ``num_frames``. Paths are relative to the JSON
file. For concatenated LeRobot videos, ``episode_frame_range: [start, end]``
sets the episode's inclusive/exclusive physical frame bounds. Explicit frame
indices are always global indices in the video file. Manifest v2 uses
``tasks: [{task_id, demos: [...]}]`` for explicit task-scoped pairing.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_ROLES = frozenset({"train", "development", "evaluation"})
Manifest = str | Path | Mapping[str, Any]


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_human_manifest(manifest: Manifest) -> dict[str, Any]:
    """Read/validate a manifest and resolve paths and actual file SHA256 values."""
    if isinstance(manifest, Mapping):
        payload = dict(manifest)
        directory = Path.cwd()
    else:
        path = Path(manifest).resolve()
        with path.open() as stream:
            payload = json.load(stream)
        directory = path.parent
    version = payload.get("schema_version")
    if version == 1:
        groups = [{"task_id": payload.get("task_id"), "demos": payload.get("demos")}]
    elif version == 2:
        groups = payload.get("tasks")
        if not isinstance(groups, list) or not groups:
            raise ValueError("Human manifest schema_version=2 requires a nonempty tasks list")
    else:
        raise ValueError("Human manifest requires schema_version=1 or schema_version=2")
    entries, tasks = [], set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("Each human task must be a mapping")
        task_id = group.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Human manifest requires a nonempty task_id")
        if task_id in tasks:
            raise ValueError(f"Duplicate human task_id: {task_id}")
        tasks.add(task_id)
        demos = group.get("demos")
        if not isinstance(demos, list) or not demos:
            raise ValueError("Human manifest requires a nonempty demos list per task")
        for entry in demos:
            if not isinstance(entry, Mapping):
                raise ValueError("Each human demo must be a mapping")
            if "task_id" in entry and entry["task_id"] != task_id:
                raise ValueError("Human demo task_id differs from its task group")
            entries.append(dict(entry, task_id=task_id))
    records, identifiers, hashes = [], set(), {}
    for entry in entries:
        record = dict(entry)
        for key in ("id", "repo_id", "camera", "path"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Human demo requires nonempty {key}")
        if record["id"] in identifiers:
            raise ValueError(f"Duplicate human demo id: {record['id']}")
        identifiers.add(record["id"])
        if record.get("role") not in _ROLES:
            raise ValueError(f"Human demo role must be one of {sorted(_ROLES)}")
        _positive_int(record.get("episode_id"), "episode_id", allow_zero=True)
        explicit, uniform = "frame_indices" in record, "num_frames" in record
        if explicit == uniform:
            raise ValueError("Specify exactly one of frame_indices and num_frames")
        if explicit:
            indices = record["frame_indices"]
            if not isinstance(indices, list) or not indices:
                raise ValueError("frame_indices must be a nonempty list")
            for value in indices:
                _positive_int(value, "frame index", allow_zero=True)
            if indices != sorted(indices):
                raise ValueError("frame_indices must be in temporal order")
        else:
            _positive_int(record["num_frames"], "num_frames")
        bounds = record.get("episode_frame_range")
        if bounds is not None:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError("episode_frame_range must be [start, end_exclusive]")
            start = _positive_int(bounds[0], "episode start", allow_zero=True)
            end = _positive_int(bounds[1], "episode end")
            if end <= start:
                raise ValueError("episode_frame_range end must exceed start")
            if explicit and any(index < start or index >= end for index in indices):
                raise ValueError("frame_indices lie outside episode_frame_range")
            record["episode_frame_range"] = [start, end]
        path = Path(record["path"]).expanduser()
        path = (path if path.is_absolute() else directory / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in hashes:
            hashes[path] = _sha256(path)
        actual_hash = hashes[path]
        if record.get("video_sha256", actual_hash) != actual_hash:
            raise ValueError(f"Human video SHA256 mismatch: {path}")
        record.update(path=str(path), video_sha256=actual_hash)
        records.append(record)
    normalized = dict(payload, demos=records)
    if version == 2:
        normalized["tasks"] = [dict(group, demos=[record for record in records
                                                if record["task_id"] == group["task_id"]])
                               for group in groups]
    return normalized


def _read_video(record: Mapping[str, Any], backend: str) -> tuple[torch.Tensor, list[int]]:
    """Decode exact requested file indices, without replacing failed frames."""
    path = str(record["path"])
    if backend == "torchcodec":
        from imitator.dataset.video_utils import VideoDecoderCache
        cache = VideoDecoderCache()
        try:
            decoder = cache.get_decoder(path)
            count = len(decoder)
            indices = _sample_indices(record, count)
            frames = decoder.get_frames_at(indices=indices).data
        finally:
            cache.clear()
    elif backend == "opencv":
        import cv2
        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open human video: {path}")
            indices = _sample_indices(record, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            decoded = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, image = capture.read()
                if not success:
                    raise RuntimeError(f"Cannot decode human video frame {index}: {path}")
                decoded.append(torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
            frames = torch.stack(decoded)
        finally:
            capture.release()
    else:
        raise ValueError("Human video backend must be torchcodec or opencv")
    if frames.ndim != 4 or frames.shape[:2] != (len(indices), 3):
        raise ValueError(f"Human RGB decoder returned unexpected shape {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise TypeError(f"Human decoder must return uint8 RGB, received {frames.dtype}")
    return frames, indices


def _sample_indices(record: Mapping[str, Any], count: int) -> list[int]:
    if count < 1:
        raise ValueError("Human video has no frames")
    start, end = record.get("episode_frame_range", (0, count))
    if end > count:
        raise ValueError(f"episode_frame_range ends beyond video length {count}")
    if "frame_indices" in record:
        indices = list(record["frame_indices"])
        if indices[-1] >= count:
            raise ValueError(f"Human frame index {indices[-1]} exceeds video length {count}")
    else:
        # Short episodes intentionally repeat physical indices, rather than
        # borrowing frames from their neighbouring episode in a shared file.
        indices = np.linspace(start, end - 1, record["num_frames"], dtype=np.int64).tolist()
    return indices


def _load_record(record: Mapping[str, Any], task_id: str, image_size: Sequence[int], backend: str) -> dict[str, Any]:
    if len(image_size) != 2 or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in image_size):
        raise ValueError("image_size must contain positive height and width")
    frames, indices = _read_video(record, backend)
    video = frames.to(torch.float32) / 255
    if tuple(video.shape[-2:]) != tuple(image_size):
        video = F.interpolate(video, size=tuple(image_size), mode="bilinear", align_corners=False)
    identity = {key: record[key] for key in ("video_sha256", "repo_id", "episode_id", "camera")}
    identity.update(frame_indices=indices, image_size=list(image_size))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata = dict(record, task_id=task_id, frame_indices=indices, image_size=list(image_size))
    return {"human_video": video.contiguous(), "human_demo_id": f"clip-sha256:{digest}", "metadata": metadata}


def load_human_demo(
    manifest: Manifest, *, role: str = "evaluation", demo_id: str | None = None,
    task_id: str | None = None,
    image_size: Sequence[int] = (224, 224), video_backend: str = "torchcodec",
) -> dict[str, Any]:
    """Load one explicitly scoped demo, rejecting ambiguous implicit selection."""
    payload = load_human_manifest(manifest)
    records = [record for record in payload["demos"] if record["role"] == role
               and (demo_id is None or record["id"] == demo_id)
               and (task_id is None or record["task_id"] == task_id)]
    if len(records) != 1:
        raise ValueError(f"Select exactly one human demo for role={role!r}, id={demo_id!r}; found {len(records)}")
    return _load_record(records[0], records[0]["task_id"], image_size, video_backend)


class NativeVideoDataset(Dataset):
    """Add human clips (T,3,H,W) using each robot sample's dataset identity.

    State/action tensors and train-only normalization remain owned by the base
    dataset. Within the identified task, clips cycle deterministically through
    its selected role. Robot task IDs are metadata, never policy inputs.
    """

    def __init__(
        self, base_dataset: Dataset, manifest: Manifest, *, role: str = "train",
        image_size: Sequence[int] = (224, 224), video_backend: str = "torchcodec",
        task_id: str | None = None, dataset_tasks: Mapping[int, str] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.manifest = load_human_manifest(manifest)
        if dataset_tasks is None:
            selected_task = task_id or self.manifest.get("task_id")
            if not selected_task:
                raise ValueError("Multitask human pairing requires explicit dataset_tasks")
            dataset_tasks = {0: selected_task}
        elif task_id is not None:
            raise ValueError("Specify task_id or dataset_tasks, not both")
        self.dataset_tasks = dict(dataset_tasks)
        if not self.dataset_tasks or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            or not isinstance(task, str) or not task
            for index, task in self.dataset_tasks.items()
        ):
            raise ValueError("dataset_tasks requires nonnegative dataset indices and nonempty task IDs")
        manifest_tasks = {record["task_id"] for record in self.manifest["demos"]}
        if not set(self.dataset_tasks.values()).issubset(manifest_tasks):
            raise ValueError("Human manifest task_id differs from the robot task")
        selected_tasks = set(self.dataset_tasks.values())
        records = [record for record in self.manifest["demos"]
                   if record["role"] == role and record["task_id"] in selected_tasks]
        if not records or any(not any(record["task_id"] == task for record in records)
                              for task in selected_tasks):
            raise ValueError(f"No human demonstrations for role {role!r} in one or more robot tasks")
        counts = {len(record["frame_indices"]) if "frame_indices" in record else record["num_frames"]
                  for record in records}
        if len(counts) != 1:
            raise ValueError("Human demos in one batch require the same sampled frame count")
        self.human_demos = [_load_record(record, record["task_id"], image_size, video_backend)
                            for record in records]
        self.task_demos = {task: [demo for demo in self.human_demos
                                  if demo["metadata"]["task_id"] == task] for task in selected_tasks}
        sub_datasets = getattr(base_dataset, "sub_datasets", [])
        self.dataset_repos = {index: str(sub.repo_id) for index, sub in enumerate(sub_datasets)}
        self.role = role

    def __getattr__(self, name: str) -> Any:
        base = self.__dict__.get("base_dataset")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base_dataset[index])
        dataset_idx = sample.get("dataset_idx")
        if dataset_idx is None:
            if len(self.dataset_tasks) != 1:
                raise ValueError("Multitask robot samples must identify their dataset_idx")
            dataset_idx = next(iter(self.dataset_tasks))
        if isinstance(dataset_idx, torch.Tensor):
            if dataset_idx.ndim != 0 or dataset_idx.dtype not in (torch.int32, torch.int64):
                raise ValueError("Robot dataset_idx must be a scalar integer")
            dataset_idx = int(dataset_idx)
        if isinstance(dataset_idx, bool) or not isinstance(dataset_idx, int) or dataset_idx not in self.dataset_tasks:
            raise ValueError(f"Unknown robot dataset_idx: {dataset_idx!r}")
        expected_repo = self.dataset_repos.get(dataset_idx)
        if expected_repo is not None and sample.get("repo_id") != expected_repo:
            raise ValueError("Robot sample repo_id differs from its actual dataset_idx")
        task = self.dataset_tasks[dataset_idx]
        demos = self.task_demos[task]
        # The index chooses only among this task's clips. It cannot select a
        # different task, including when the caller shuffles/reorders samples.
        demo = demos[index % len(demos)]
        sample.update(human_video=demo["human_video"], human_demo_id=demo["human_demo_id"],
                      native_sample_index=index)
        return sample


def _sanity_check() -> None:
    """Exercise real CPU video decoding and unchanged native tensor values."""
    import tempfile
    import cv2
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "demo.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
        assert writer.isOpened()
        for value in (0, 80, 160, 240):
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
        writer.release()
        manifest = {"schema_version": 1, "task_id": "pill", "demos": [
            {"id": "demo", "repo_id": "human_H31", "episode_id": 0, "camera": "zed2i", "path": str(path),
             "role": "train", "episode_frame_range": [1, 3], "num_frames": 3}]}
        sample = {"view_1": torch.zeros(1, 3, 16, 16), "states": torch.zeros(1, 16),
                  "actions": torch.zeros(2, 14), "action_is_pad": torch.tensor([False, True])}
        dataset = NativeVideoDataset([sample], manifest, image_size=(8, 8), video_backend="opencv")
        item = dataset[0]
        assert all(item[key] is value for key, value in sample.items())
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        assert batch["human_video"].shape == (1, 3, 3, 8, 8)
        assert torch.isfinite(batch["human_video"]).all()
        assert batch["human_video"].min() >= 0 and batch["human_video"].max() <= 1
        assert dataset.human_demos[0]["metadata"]["frame_indices"] == [1, 1, 2]
        print("native_video_data sanity:", {key: tuple(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)})


if __name__ == "__main__":
    _sanity_check()
