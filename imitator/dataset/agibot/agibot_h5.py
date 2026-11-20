#!/usr/bin/env python3
"""Convert local AgiBotWorld gripper data directly into this project's LeRobot v3.

Why this converter exists
-------------------------
the current any4lerobot AgiBot converter targets a newer upstream LeRobot API
(`DatasetWriter`, `feature_utils`, dataset aggregation).  The Imitator Game pins
LeRobot 0.5.x while maintaining its own v3-capable LeRobotDataset implementation
under ``imitator.dataset.lerobot_dataset``.

This file keeps any4lerobot's useful raw-data conventions but writes directly
through the project-maintained API:

    LeRobotDataset.create -> add_frame -> save episode metadata/parquet -> finalize

There is no Ray/DataTrove/new-LeRobot dependency and no temporary per-task
LeRobot datasets to aggregate.

The default output is tailored for SelfPromptRobotDataset:
  * RGB: head + hand_left + hand_right
  * state: joint.position(14) + effector.position(2) = 16D
  * action: joint.position(14) + effector.position(2) = 16D
  * gripper tasks only (dexhand/tactile tasks are skipped)

Raw MP4 files are never decoded/re-encoded.  Each source episode gets its own
LeRobot v3 video file.  ``--video-mode hardlink`` is very fast and does not
consume duplicate blocks when input/output share a filesystem; ``copy`` makes
an independent copy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Make `examples...` imports work even when this file is executed directly from
# imitator/dataset/agibot/.
AGIBOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGIBOT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imitator.dataset.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.compute_stats import compute_episode_stats  # noqa: E402
from lerobot.datasets.utils import write_info  # noqa: E402

# Some lerobot 0.5 builds keep get_video_info in video_utils only.  Prefer the
# project's implementation when the import above is unavailable at runtime.
from imitator.dataset.video_utils import (  # noqa: E402
    get_video_duration_in_s,
)

from agibot_utils.agibot_utils import (  # noqa: E402
    list_episode_ids,
    task_episode_ids,
    close_tar_cache,
    load_control_episode,
    load_task_info,
    task_instruction,
    task_numeric_id,
)
from agibot_utils.config import (  # noqa: E402
    DEFAULT_CAMERAS,
    NON_GRIPPER_TASK_IDS,
    is_gripper_task,
)
from agibot_utils.lerobot_utils import control_dims, generate_features  # noqa: E402


DEFAULT_FEATURE_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}


def _scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar(value[0])
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


class AgiBotLeRobotDataset(LeRobotDataset):
    """Project-local LeRobotDataset with external-MP4 episode writing."""

    def _nonvisual_user_features(self) -> dict[str, dict]:
        return {
            key: spec
            for key, spec in self.features.items()
            if key not in DEFAULT_FEATURE_KEYS
            and spec.get("dtype") not in ("video", "image", "depth")
        }

    def _validate_control_frame(self, frame: dict[str, Any]) -> None:
        if "task" not in frame:
            raise KeyError("Each frame must contain a 'task' string")

        expected = self._nonvisual_user_features()
        actual = {k for k in frame if k != "task" and k != "timestamp"}
        missing = set(expected) - actual
        extra = actual - set(expected)
        if missing or extra:
            raise ValueError(
                f"Frame feature mismatch. missing={sorted(missing)}, extra={sorted(extra)}"
            )

        for key, spec in expected.items():
            arr = np.asarray(frame[key])
            expected_shape = tuple(int(x) for x in spec.get("shape", ()))
            if tuple(arr.shape) != expected_shape:
                raise ValueError(
                    f"{key} has shape {arr.shape}, expected {expected_shape}"
                )

    def add_frame(self, frame: dict[str, Any]) -> None:
        """Add numerical AgiBot data; RGB is supplied once at episode save time."""
        frame = dict(frame)
        for key, value in list(frame.items()):
            if isinstance(value, torch.Tensor):
                frame[key] = value.detach().cpu().numpy()

        self._validate_control_frame(frame)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = int(self.episode_buffer["size"])
        timestamp = float(frame.pop("timestamp", frame_index / self.fps))
        task = str(frame.pop("task"))

        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        for key, value in frame.items():
            self.episode_buffer[key].append(np.asarray(value, dtype=np.float32))

        self.episode_buffer["size"] += 1

    def _materialize_video(
        self,
        source: Path,
        destination: Path,
        mode: str,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()

        if mode == "hardlink":
            try:
                os.link(source, destination)
                return
            except OSError as exc:
                print(
                    f"[video] hardlink failed ({exc}); falling back to copy: {source}"
                )
                shutil.copy2(source, destination)
                return

        if mode == "symlink":
            destination.symlink_to(source.resolve())
            return

        if mode == "copy":
            shutil.copy2(source, destination)
            return

        raise ValueError(f"Unknown video mode: {mode!r}")

    def _save_external_video(
        self,
        *,
        video_key: str,
        episode_index: int,
        source_path: Path,
        video_mode: str,
    ) -> dict[str, Any]:
        """Store one episode per MP4 shard; no concatenate/re-encode step."""
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        # v3 path templates use chunk/file indices.  A file per episode keeps
        # conversion transactional and is much cheaper than repeatedly rewriting
        # a growing concatenated MP4 shard.
        chunk_size = int(self.meta.chunks_size)
        chunk_idx = int(episode_index // chunk_size)
        file_idx = int(episode_index % chunk_size)

        dst = self.root / self.meta.video_path.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        self._materialize_video(source_path, dst, video_mode)

        duration = float(get_video_duration_in_s(dst))
        if duration <= 0:
            raise RuntimeError(f"Invalid video duration for {dst}: {duration}")

        # Populate codec/fps/video metadata after the first materialized file for
        # this feature.  update_video_info() is part of the project-maintained
        # LeRobotDatasetMetadata implementation.
        feature_info = self.meta.info["features"][video_key].get("info")
        if not feature_info:
            try:
                self.meta.update_video_info(video_key)
            except Exception:
                # In case the metadata helper assumes chunk-000/file-000, the
                # first destination episode is exactly that file. Re-raise with
                # context because loading videos later depends on this metadata.
                raise RuntimeError(
                    f"Failed to populate video metadata for {video_key} from {dst}"
                )
            write_info(self.meta.info, self.meta.root)

        return {
            "episode_index": int(episode_index),
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": 0.0,
            f"videos/{video_key}/to_timestamp": duration,
        }

    def save_external_episode(
        self,
        *,
        videos: dict[str, Path],
        source_task_id: str,
        source_episode_id: int,
        action_config: Any | None,
        video_mode: str,
    ) -> int:
        """Save current numerical buffer and attach raw AgiBot MP4s."""
        if self.episode_buffer is None or int(self.episode_buffer.get("size", 0)) <= 0:
            raise ValueError("Cannot save an empty episode")

        episode_buffer = self.episode_buffer
        episode_length = int(episode_buffer.pop("size"))
        tasks = list(episode_buffer.pop("task"))
        episode_tasks = list(dict.fromkeys(tasks))
        episode_index = int(_scalar(episode_buffer["episode_index"]))

        if episode_index != int(self.meta.total_episodes):
            raise RuntimeError(
                f"Destination episode index mismatch: buffer={episode_index}, "
                f"meta.total_episodes={self.meta.total_episodes}"
            )

        episode_buffer["index"] = np.arange(
            self.meta.total_frames,
            self.meta.total_frames + episode_length,
            dtype=np.int64,
        )
        episode_buffer["episode_index"] = np.full(
            (episode_length,), episode_index, dtype=np.int64
        )

        self.meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.asarray(
            [self.meta.get_task_index(task) for task in tasks], dtype=np.int64
        )

        numeric_features = self._nonvisual_user_features()
        for key in numeric_features:
            values = episode_buffer.get(key)
            if values is None or len(values) != episode_length:
                raise ValueError(
                    f"Episode feature {key} has {0 if values is None else len(values)} "
                    f"rows, expected {episode_length}"
                )
            episode_buffer[key] = np.stack(values).astype(np.float32, copy=False)

        # Stats are required by HyperICL normalization.  We intentionally do not
        # decode RGB just to compute image statistics.
        stats_input = {key: episode_buffer[key] for key in numeric_features}
        ep_stats = compute_episode_stats(stats_input, numeric_features)

        # Write frame-level parquet first. Video sources have already been
        # validated before this method is entered by the outer converter.
        ep_metadata = self._save_episode_data(episode_buffer)

        for video_key in self.meta.video_keys:
            if video_key not in videos:
                raise KeyError(f"Missing source video for {video_key}")
            video_meta = self._save_external_video(
                video_key=video_key,
                episode_index=episode_index,
                source_path=Path(videos[video_key]),
                video_mode=video_mode,
            )
            ep_metadata.update(video_meta)

        ep_metadata.update(
            {
                "source_task_id": str(source_task_id),
                "source_episode_id": int(source_episode_id),
                "source_action_config": json.dumps(
                    action_config, ensure_ascii=False, separators=(",", ":")
                )
                if action_config is not None
                else "",
            }
        )

        self.meta.save_episode(
            episode_index,
            episode_length,
            episode_tasks,
            ep_stats,
            ep_metadata,
        )

        # No temporary image directories were created; simply reset the buffer.
        self.episode_buffer = self.create_episode_buffer()
        return episode_index


class ConversionManifest:
    """Append-only source->destination mapping used for safe resume.

    Successfully written episodes also store source_task_id/source_episode_id in
    meta/episodes, so the manifest can be reconstructed after a lost sidecar.
    """

    def __init__(self, output_root: Path, dataset: AgiBotLeRobotDataset | None = None):
        self.path = output_root / "meta" / "agibot_conversion_manifest.jsonl"
        self.done: set[tuple[str, int]] = set()
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.done.add((str(row["task_id"]), int(row["episode_id"])))

        # Reconstruct/augment from canonical episode metadata when possible.
        if dataset is not None and dataset.meta.episodes is not None:
            try:
                for ep_idx in range(len(dataset.meta.episodes)):
                    row = dataset.meta.episodes[ep_idx]
                    if "source_task_id" in row and "source_episode_id" in row:
                        task_id = str(_scalar(row["source_task_id"]))
                        source_ep = int(_scalar(row["source_episode_id"]))
                        self.done.add((task_id, source_ep))
            except Exception as exc:
                print(f"[resume] could not reconstruct manifest from meta/episodes: {exc}")

    def contains(self, task_id: str, episode_id: int) -> bool:
        return (str(task_id), int(episode_id)) in self.done

    def append(
        self,
        *,
        task_id: str,
        episode_id: int,
        destination_episode: int,
        frames: int,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "task_id": str(task_id),
            "episode_id": int(episode_id),
            "destination_episode": int(destination_episode),
            "frames": int(frames),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done.add((str(task_id), int(episode_id)))


def _validate_source_video(path: Path) -> float:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing/empty MP4: {path}")
    duration = float(get_video_duration_in_s(path))
    if duration <= 0:
        raise RuntimeError(f"Unreadable/zero-duration MP4: {path}")
    return duration



def _load_task_info_authoritative(task_json: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """Load AgiBot task_info directly; task_info is authoritative for episode IDs.

    AgiBotWorld task_info/task_XXX.json is a JSON list whose rows contain
    ``episode_id``.  This intentionally does not inspect observations/<task>/
    because official releases may keep observations in range tar archives.
    """
    with task_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # Official Alpha/Beta format is a list.  Accept a couple of wrapper forms
    # defensively so local mirrors/preprocessing do not silently become empty.
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("episodes"), list):
            rows = payload["episodes"]
        elif isinstance(payload.get("data"), list):
            rows = payload["data"]
        else:
            raise ValueError(
                f"Unsupported task_info object in {task_json}; expected JSON list "
                f"or dict containing 'episodes'/'data'. keys={list(payload)[:20]}"
            )
    else:
        raise TypeError(
            f"Unsupported task_info root type in {task_json}: {type(payload).__name__}"
        )

    if not rows:
        raise ValueError(f"task_info is empty: {task_json}")

    cleaned: list[dict[str, Any]] = []
    episode_ids: list[int] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(
                f"task_info row {i} in {task_json} is {type(row).__name__}, expected dict"
            )
        if "episode_id" not in row:
            raise KeyError(
                f"task_info row {i} in {task_json} has no 'episode_id'; "
                f"keys={list(row)[:30]}"
            )
        try:
            eid = int(row["episode_id"])
        except Exception as exc:
            raise ValueError(
                f"Invalid episode_id at row {i} in {task_json}: {row.get('episode_id')!r}"
            ) from exc
        cleaned.append(row)
        episode_ids.append(eid)

    # Preserve one metadata row per episode; duplicates are almost certainly bad
    # source metadata, so fail loudly instead of silently dropping data.
    if len(set(episode_ids)) != len(episode_ids):
        seen=set(); dup=[]
        for eid in episode_ids:
            if eid in seen and eid not in dup:
                dup.append(eid)
            seen.add(eid)
        raise ValueError(f"Duplicate episode_id(s) in {task_json}: {dup[:20]}")

    order = np.argsort(np.asarray(episode_ids, dtype=np.int64))
    cleaned = [cleaned[int(i)] for i in order]
    episode_ids = [episode_ids[int(i)] for i in order]
    return cleaned, episode_ids


def _discover_task_jsons(src_root: Path, requested: set[str]) -> list[Path]:
    files = sorted((src_root / "task_info").glob("task_*.json"))
    if requested:
        files = [p for p in files if p.stem in requested]
    return files


def _inspect_local_dataset_for_resume(output_root: Path) -> tuple[int, list[Path]]:
    """Validate a local conversion output without invoking LeRobot's Hub fallback.

    The project-maintained LeRobotDatasetMetadata constructor falls back to
    Hugging Face whenever a local metadata file is missing.  That behavior is
    useful for normal Hub datasets, but wrong for this offline converter.
    Therefore we inspect the local files first and only call the constructor
    when the dataset is known to be complete enough for local resume.
    """
    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        return 0, [info_path]

    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Cannot read local info.json: {info_path}: {exc}") from exc

    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes == 0:
        # LeRobot.create() writes info.json immediately, while tasks/stats/episode
        # parquet files are only created after the first successful episode.
        # Thus this is a harmless empty shell from an interrupted/zero-episode run.
        return 0, []

    required = [
        output_root / "meta" / "tasks.parquet",
        output_root / "meta" / "stats.json",
    ]
    missing = [p for p in required if not p.exists()]

    episode_files = list((output_root / "meta" / "episodes").glob("**/*.parquet"))
    data_files = list((output_root / "data").glob("**/*.parquet"))
    if not episode_files:
        missing.append(output_root / "meta" / "episodes" / "<chunk>/file-*.parquet")
    if not data_files:
        missing.append(output_root / "data" / "<chunk>/file-*.parquet")

    return total_episodes, missing


def _open_or_create_dataset(
    *,
    output_root: Path,
    repo_id: str,
    features: dict,
    fps: int,
    video_backend: str,
    resume: bool,
    force: bool,
) -> tuple[AgiBotLeRobotDataset, bool]:
    info_path = output_root / "meta" / "info.json"

    if force and output_root.exists():
        print(f"[force] removing existing output: {output_root}")
        shutil.rmtree(output_root)

    if info_path.exists():
        if not resume:
            raise FileExistsError(
                f"Output already contains a LeRobot dataset: {output_root}\n"
                "Use --resume to append/skip converted source episodes, or --force to restart."
            )

        total_episodes, missing = _inspect_local_dataset_for_resume(output_root)

        if total_episodes == 0:
            # Typical case after the previous converter found zero raw episodes:
            # info.json exists, but tasks.parquet/stats/episodes do not.  Recreate
            # locally instead of letting LeRobot interpret this as a Hub dataset.
            print(
                f"[resume] found empty/incomplete zero-episode output; "
                f"recreating locally: {output_root}"
            )
            shutil.rmtree(output_root)
        else:
            if missing:
                missing_text = "\n  - ".join(str(p) for p in missing)
                raise RuntimeError(
                    "Cannot safely resume this local AgiBot conversion because "
                    "metadata/data files are missing. The converter will NOT "
                    "fall back to Hugging Face.\n"
                    f"Existing total_episodes={total_episodes}\n"
                    f"Missing:\n  - {missing_text}\n"
                    "Inspect the output or restart with --force."
                )

            # Local files have been validated, so the project LeRobot constructor
            # should not enter its download/Hub fallback path.
            ds = AgiBotLeRobotDataset(
                repo_id=repo_id,
                root=output_root,
                video_backend=video_backend,
            )
            ds.episode_buffer = ds.create_episode_buffer()
            return ds, True

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output exists but has no meta/info.json: {output_root}. "
            "Move it aside or use --force."
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    ds = AgiBotLeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type="a2d",
        features=features,
        use_videos=True,
        video_backend=video_backend,
        batch_encoding_size=1,
    )
    return ds, False


def convert(args: argparse.Namespace) -> None:
    src_root = args.src_path.expanduser().resolve()
    output_root = args.output_path.expanduser().resolve()
    cameras = list(dict.fromkeys(args.cameras))
    requested = set(args.task_ids or [])

    if not (src_root / "task_info").is_dir():
        raise FileNotFoundError(f"Missing {src_root / 'task_info'}")
    if not (src_root / "observations").is_dir():
        raise FileNotFoundError(f"Missing {src_root / 'observations'}")
    if not (src_root / "proprio_stats").is_dir():
        raise FileNotFoundError(f"Missing {src_root / 'proprio_stats'}")

    features = generate_features(cameras)
    state_dim, action_dim = control_dims()

    task_jsons = _discover_task_jsons(src_root, requested)
    if not task_jsons:
        raise RuntimeError("No matching task_info/task_*.json files found")

    gripper_jsons: list[Path] = []
    skipped_non_gripper: list[str] = []
    for path in task_jsons:
        if is_gripper_task(path.stem):
            gripper_jsons.append(path)
        else:
            skipped_non_gripper.append(path.stem)

    if skipped_non_gripper:
        print(
            "[schema] skipping non-gripper tasks (different action space): "
            + ", ".join(skipped_non_gripper)
        )

    print("=" * 80)
    print("AgiBot -> The-Imitator-Game LeRobot v3 direct converter")
    print(f"Source       : {src_root}")
    print(f"Output       : {output_root}")
    print(f"Repo ID      : {args.repo_id}")
    print(f"FPS          : {args.fps}")
    print(f"Cameras      : {cameras}")
    print(f"Control dims : state={state_dim}, action={action_dim}")
    print(f"Tasks        : {len(gripper_jsons)} gripper task(s)")
    print(f"Video mode   : {args.video_mode}")
    print(f"Resume       : {args.resume}")
    print("=" * 80)

    if args.dry_run:
        for path in gripper_jsons:
            numeric = task_numeric_id(path)
            rows, eps = _load_task_info_authoritative(path)
            if args.max_episodes_per_task is not None:
                eps = eps[: args.max_episodes_per_task]
            print(f"{path.stem}: {len(eps)} raw episode(s)")
        return

    dataset, resumed = _open_or_create_dataset(
        output_root=output_root,
        repo_id=args.repo_id,
        features=features,
        fps=args.fps,
        video_backend=args.video_backend,
        resume=args.resume,
        force=args.force,
    )

    manifest = ConversionManifest(output_root, dataset=dataset)
    if resumed:
        print(
            f"[resume] dataset episodes={dataset.meta.total_episodes}, "
            f"known source episodes={len(manifest.done)}"
        )
        if dataset.meta.total_episodes > 0 and not manifest.done:
            raise RuntimeError(
                "Cannot safely resume: existing dataset has episodes, but neither the "
                "conversion manifest nor meta/episodes contains source identities. "
                "Use --force to restart."
            )

    converted = 0
    skipped_done = 0
    skipped_bad = 0

    try:
        for task_json in gripper_jsons:
            task_stem = task_json.stem
            numeric_task = task_numeric_id(task_json)
            info_rows, episode_ids = _load_task_info_authoritative(task_json)
            info_by_episode = {int(row["episode_id"]): row for row in info_rows}
            instruction = task_instruction(task_stem, info_rows)

            print(
                f"[task_info] {task_json.name}: rows={len(info_rows)}, "
                f"first_ep={episode_ids[0]}, last_ep={episode_ids[-1]}"
            )
            if args.max_episodes_per_task is not None:
                episode_ids = episode_ids[: args.max_episodes_per_task]

            print(
                f"\n[{task_stem}] raw episodes={len(episode_ids)} | task={instruction}"
            )

            for source_ep in episode_ids:
                if manifest.contains(task_stem, source_ep):
                    skipped_done += 1
                    continue

                if source_ep not in info_by_episode:
                    print(
                        f"  [skip] episode {source_ep}: not present in {task_json.name}"
                    )
                    skipped_bad += 1
                    continue

                try:
                    frames, videos = load_control_episode(
                        src_root,
                        numeric_task,
                        source_ep,
                        cameras,
                    )

                    # Validate all MP4 containers before any parquet bytes for this
                    # episode are committed.
                    for video_path in videos.values():
                        _validate_source_video(Path(video_path))

                    for frame in frames:
                        item = dict(frame)
                        item["task"] = instruction
                        dataset.add_frame(item)

                except Exception as exc:
                    # Parsing/validation happens before any destination bytes for
                    # this episode are committed, so it is safe to skip a bad raw
                    # episode and continue.
                    dataset.episode_buffer = dataset.create_episode_buffer()
                    skipped_bad += 1
                    print(
                        f"  [skip-bad-source] {task_stem} episode={source_ep}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if args.fail_fast:
                        raise
                    continue

                # From this point onward we are mutating the final dataset. Any
                # exception is fatal: continuing could leave parquet/video/meta
                # indices inconsistent.
                label_info = info_by_episode[source_ep].get("label_info", {}) or {}
                action_config = label_info.get("action_config")
                try:
                    destination_ep = dataset.save_external_episode(
                        videos=videos,
                        source_task_id=task_stem,
                        source_episode_id=source_ep,
                        action_config=action_config,
                        video_mode=args.video_mode,
                    )
                    manifest.append(
                        task_id=task_stem,
                        episode_id=source_ep,
                        destination_episode=destination_ep,
                        frames=len(frames),
                    )
                except Exception as exc:
                    print(
                        f"  [FATAL write] {task_stem} episode={source_ep}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise

                converted += 1
                print(
                    f"  [ok] src_ep={source_ep} -> dst_ep={destination_ep} "
                    f"frames={len(frames)}"
                )

        dataset.finalize()

    finally:
        # finalize is idempotent enough for closing writers after exceptions; do
        # not hide the original exception if closing itself fails.
        try:
            dataset.finalize()
        except Exception as close_exc:
            print(f"[warning] finalize after conversion raised: {close_exc}")
        finally:
            close_tar_cache()

    print("\n" + "=" * 80)
    print("Conversion finished")
    print(f"new episodes      : {converted}")
    print(f"already converted : {skipped_done}")
    print(f"skipped/errors    : {skipped_bad}")
    print(f"total episodes    : {dataset.meta.total_episodes}")
    print(f"total frames      : {dataset.meta.total_frames}")
    print(f"output            : {output_root}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AgiBotWorld gripper data directly to the Imitator-Game LeRobot v3 format."
    )
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default="agibotworld")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=[],
        help="Optional task stems, e.g. task_398 task_410. Empty means all local gripper tasks.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="RGB cameras stored in the output. Default: head hand_left hand_right",
    )
    parser.add_argument(
        "--video-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How raw MP4s are materialized in the v3 output.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="torchcodec",
        help="Backend later used by the project LeRobot loader.",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Debug/smoke-test limit; omit for all episodes.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    if args.max_episodes_per_task is not None and args.max_episodes_per_task < 1:
        parser.error("--max-episodes-per-task must be >= 1")
    return args


if __name__ == "__main__":
    convert(parse_args())
