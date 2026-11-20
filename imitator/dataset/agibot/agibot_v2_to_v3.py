#!/usr/bin/env python3
"""
Convert existing AgiBotWorld LeRobot v2 task datasets into one local
The-Imitator-Game LeRobot v3 dataset.

Source layout (example):
  /mnt/sdb/agi_data/data_lerobot/agibotworld/
    task_398/
      meta/info.json
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.top_head/episode_000000.mp4
      ...

Target layout:
  /mnt/sdb/agi_data/data_lerobot_v3/agibotworld/
    meta/
    data/
    videos/

The old AgiBot v2 converter built vectors as:
  observation.state = [state_joint(14), state_effector(2), state_head(2), state_waist(2)]
  action            = [action_joint(14), action_effector(2), action_head(2),
                       action_waist(2), action_velocity(2)]

This converter keeps only the policy-control part and writes semantic v3 fields:
  observation.states.joint.position      [14]
  observation.states.effector.position    [2]
  actions.joint.position                 [14]
  actions.effector.position               [2]

Camera mapping:
  observation.images.top_head   -> observation.images.head
  observation.images.hand_left  -> observation.images.hand_left
  observation.images.hand_right -> observation.images.hand_right

Videos are hard-linked by default (copy fallback), never decoded/re-encoded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Make the project root importable even when this script is launched from agibot/.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = next(
    (p for p in _THIS.parents if (p / "examples" / "baselines" / "lerobot_dataset").is_dir()),
    None,
)
if _PROJECT_ROOT is None:
    raise RuntimeError("Could not locate The-Imitator-Game project root from script path")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from imitator.dataset.lerobot_dataset import LeRobotDataset  # noqa: E402
from imitator.dataset.video_utils import get_video_duration_in_s  # noqa: E402
from lerobot.datasets.compute_stats import compute_episode_stats  # noqa: E402
from lerobot.datasets.utils import write_info  # noqa: E402


V2_STATE_KEY = "observation.state"
V2_ACTION_KEY = "action"

V3_STATE_JOINT = "observation.states.joint.position"
V3_STATE_EFFECTOR = "observation.states.effector.position"
V3_ACTION_JOINT = "actions.joint.position"
V3_ACTION_EFFECTOR = "actions.effector.position"

CAMERA_MAP = {
    "head": "observation.images.top_head",
    "hand_left": "observation.images.hand_left",
    "hand_right": "observation.images.hand_right",
}

DEFAULT_GRIPPER_TASKS = [
    361, 362, 365, 366, 367, 368, 372, 374, 376, 377, 378, 380, 384,
    385, 388, 389, 390, 392, 398, 410, 414, 421, 422, 424, 440, 446,
]


def _parse_task_id(value: str | int) -> int:
    s = str(value)
    if s.startswith("task_"):
        s = s[5:]
    return int(s)


def _task_dir(root: Path, task_id: int) -> Path:
    return root / f"task_{task_id}"


def _read_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_task_text(task_info_root: Path | None, task_id: int) -> str:
    if task_info_root is not None:
        path = task_info_root / f"task_{task_id}.json"
        if path.exists():
            obj = _read_json(path)
            if isinstance(obj, list) and obj:
                row = obj[0]
            elif isinstance(obj, dict):
                row = obj
            else:
                row = {}
            name = str(row.get("task_name", f"task_{task_id}")).strip()
            scene = str(row.get("init_scene_text", "")).strip()
            return f"{name}. {scene}".strip() if scene else name
    return f"task_{task_id}"


def _load_v2_info(task_root: Path) -> dict:
    path = task_root / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing v2 info.json: {path}")
    info = _read_json(path)
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v2"):
        raise ValueError(f"Expected LeRobot v2 source at {task_root}, got {version!r}")
    for key, dim in ((V2_STATE_KEY, 20), (V2_ACTION_KEY, 22)):
        ft = info.get("features", {}).get(key)
        if ft is None:
            raise KeyError(f"{task_root}: missing feature {key!r}")
        shape = tuple(ft.get("shape", []))
        if shape != (dim,):
            raise ValueError(f"{task_root}: expected {key} shape {(dim,)}, got {shape}")
    return info


def _episode_index_from_parquet(path: Path) -> int:
    m = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not m:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(m.group(1))


def _iter_v2_episode_parquets(task_root: Path) -> list[tuple[int, Path]]:
    files = sorted((task_root / "data").glob("chunk-*/episode_*.parquet"))
    return [(_episode_index_from_parquet(p), p) for p in files]


def _resolve_v2_video(
    task_root: Path,
    info: dict,
    episode_index: int,
    source_video_key: str,
) -> Path:
    template = info.get("video_path")
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size

    if template:
        try:
            rel = template.format(
                episode_chunk=episode_chunk,
                chunk_index=episode_chunk,
                video_key=source_video_key,
                episode_index=episode_index,
            )
            candidate = task_root / rel
            if candidate.exists():
                return candidate
        except KeyError:
            pass

    # Robust fallback for slightly different v2 layouts.
    exact_name = f"episode_{episode_index:06d}.mp4"
    candidates = sorted((task_root / "videos").glob(f"**/{source_video_key}/{exact_name}"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Missing source video: task_root={task_root}, episode={episode_index}, key={source_video_key}"
    )


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy-fallback"


def _as_matrix(series: pd.Series, expected_dim: int, name: str) -> np.ndarray:
    values = [np.asarray(v, dtype=np.float32).reshape(-1) for v in series.tolist()]
    if not values:
        return np.empty((0, expected_dim), dtype=np.float32)
    bad = [i for i, v in enumerate(values) if v.shape != (expected_dim,)]
    if bad:
        i = bad[0]
        raise ValueError(f"{name}: row {i} has shape {values[i].shape}, expected {(expected_dim,)}")
    return np.stack(values, axis=0).astype(np.float32, copy=False)


def _v3_features(v2_info: dict, cameras: Iterable[str]) -> dict:
    features = {
        V3_STATE_JOINT: {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"joint_{i}" for i in range(14)],
        },
        V3_STATE_EFFECTOR: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_effector", "right_effector"],
        },
        V3_ACTION_JOINT: {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"joint_{i}" for i in range(14)],
        },
        V3_ACTION_EFFECTOR: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_effector", "right_effector"],
        },
    }

    src_features = v2_info.get("features", {})
    for cam in cameras:
        src_key = CAMERA_MAP[cam]
        src_ft = src_features.get(src_key)
        if src_ft is None:
            raise KeyError(f"Source dataset does not contain camera {src_key!r}")
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": tuple(src_ft["shape"]),
            "names": src_ft.get("names", ["height", "width", "channel"]),
        }
    return features


class AgiBotV2ToV3Dataset(LeRobotDataset):
    """Project LeRobotDataset with a direct MP4 ingestion path."""

    def _save_direct_video(
        self,
        video_key: str,
        episode_index: int,
        src_path: Path,
        mode: str,
    ) -> dict:
        # One physical MP4 per destination episode. This avoids concatenation and re-encoding.
        chunk_idx = episode_index // self.meta.chunks_size
        file_idx = episode_index % self.meta.chunks_size
        dst_path = self.root / self.meta.video_path.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        actual_mode = _link_or_copy(src_path, dst_path, mode)
        duration = float(get_video_duration_in_s(dst_path))

        if episode_index == 0:
            self.meta.update_video_info(video_key)
            write_info(self.meta.info, self.meta.root)

        return {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": 0.0,
            f"videos/{video_key}/to_timestamp": duration,
            f"videos/{video_key}/copy_mode": actual_mode,
        }

    def save_v2_episode(
        self,
        *,
        task: str,
        timestamps: np.ndarray,
        frame_indices: np.ndarray,
        state_joint: np.ndarray,
        state_effector: np.ndarray,
        action_joint: np.ndarray,
        action_effector: np.ndarray,
        videos: dict[str, Path],
        video_mode: str,
    ) -> int:
        n = int(len(timestamps))
        if n <= 0:
            raise ValueError("Cannot save an empty episode")

        arrays = {
            V3_STATE_JOINT: state_joint,
            V3_STATE_EFFECTOR: state_effector,
            V3_ACTION_JOINT: action_joint,
            V3_ACTION_EFFECTOR: action_effector,
        }
        for key, arr in arrays.items():
            if len(arr) != n:
                raise ValueError(f"Length mismatch for {key}: {len(arr)} != {n}")

        episode_index = int(self.meta.total_episodes)
        self.meta.save_episode_tasks([task])
        task_index = int(self.meta.get_task_index(task))

        episode_buffer = self.create_episode_buffer(episode_index=episode_index)
        episode_buffer["size"] = n
        episode_buffer["task"] = [task] * n
        episode_buffer["episode_index"] = episode_index
        episode_buffer["timestamp"] = [np.float32(x) for x in timestamps]
        episode_buffer["frame_index"] = [np.int64(x) for x in frame_indices]

        for key, arr in arrays.items():
            episode_buffer[key] = [row.astype(np.float32, copy=False) for row in arr]

        # Materialize the default/index columns exactly as the project v3 writer expects.
        episode_buffer["index"] = np.arange(
            self.meta.total_frames,
            self.meta.total_frames + n,
            dtype=np.int64,
        )
        episode_buffer["episode_index"] = np.full((n,), episode_index, dtype=np.int64)
        episode_buffer["task_index"] = np.full((n,), task_index, dtype=np.int64)
        episode_buffer["frame_index"] = np.asarray(episode_buffer["frame_index"], dtype=np.int64)
        episode_buffer["timestamp"] = np.asarray(episode_buffer["timestamp"], dtype=np.float32)
        for key in arrays:
            episode_buffer[key] = np.stack(episode_buffer[key]).astype(np.float32, copy=False)

        # v3 parquet intentionally excludes video columns; get_hf_features_from_features() drops them.
        ep_metadata = self._save_episode_data(episode_buffer)

        # Stats only for columns actually stored in parquet. No expensive video decoding.
        stat_features = {
            k: ft
            for k, ft in self.features.items()
            if ft["dtype"] not in ("video", "image")
        }
        stat_buffer = {k: episode_buffer[k] for k in stat_features}
        ep_stats = compute_episode_stats(stat_buffer, stat_features)

        for video_key, src_path in videos.items():
            ep_metadata.update(
                self._save_direct_video(
                    video_key=video_key,
                    episode_index=episode_index,
                    src_path=src_path,
                    mode=video_mode,
                )
            )

        self.meta.save_episode(
            episode_index=episode_index,
            episode_length=n,
            episode_tasks=[task],
            episode_stats=ep_stats,
            episode_metadata=ep_metadata,
        )
        self.episode_buffer = self.create_episode_buffer()
        return episode_index


def _manifest_path(output_root: Path) -> Path:
    return output_root / "meta" / "agibot_v2_to_v3_manifest.jsonl"


def _load_manifest(output_root: Path) -> set[tuple[int, int]]:
    path = _manifest_path(output_root)
    out: set[tuple[int, int]] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out.add((int(row["task_id"]), int(row["source_episode_index"])))
    return out


def _append_manifest(output_root: Path, row: dict) -> None:
    path = _manifest_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _validate_resume_root(root: Path) -> None:
    required = [
        root / "meta" / "info.json",
        root / "meta" / "tasks.parquet",
        root / "meta" / "stats.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if not list((root / "meta" / "episodes").glob("*/*.parquet")):
        missing.append(str(root / "meta" / "episodes" / "chunk-*/file-*.parquet"))
    if not list((root / "data").glob("*/*.parquet")):
        missing.append(str(root / "data" / "chunk-*/file-*.parquet"))
    if missing:
        raise RuntimeError(
            "Cannot safely resume incomplete local v3 dataset. Missing:\n  " + "\n  ".join(missing)
        )


def _create_or_resume_dataset(
    output_root: Path,
    repo_id: str,
    fps: int,
    features: dict,
    force: bool,
    resume: bool,
) -> tuple[AgiBotV2ToV3Dataset, bool]:
    if force and output_root.exists():
        print(f"[force] removing {output_root}")
        shutil.rmtree(output_root)

    info_path = output_root / "meta" / "info.json"
    if info_path.exists():
        if not resume:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --resume or --force."
            )
        info = _read_json(info_path)
        if int(info.get("total_episodes", 0)) == 0:
            print(f"[resume] removing zero-episode shell dataset: {output_root}")
            shutil.rmtree(output_root)
        else:
            _validate_resume_root(output_root)
            ds = AgiBotV2ToV3Dataset(repo_id=repo_id, root=output_root)
            return ds, True

    ds = AgiBotV2ToV3Dataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type="a2d",
        features=features,
        use_videos=True,
        metadata_buffer_size=1,  # every successful episode is immediately resumable
    )
    return ds, False


def convert(args: argparse.Namespace) -> None:
    v2_root = args.v2_root.resolve()
    output_root = args.output_root.resolve()
    task_info_root = args.task_info_root.resolve() if args.task_info_root else None

    if args.task_ids:
        task_ids = [_parse_task_id(x) for x in args.task_ids]
    else:
        discovered = sorted(
            _parse_task_id(p.name)
            for p in v2_root.glob("task_*")
            if p.is_dir() and re.fullmatch(r"task_\d+", p.name)
        )
        task_ids = [x for x in discovered if x != 475]

    if not task_ids:
        raise RuntimeError(f"No task_* datasets found under {v2_root}")

    cameras = list(args.cameras)
    for cam in cameras:
        if cam not in CAMERA_MAP:
            raise ValueError(f"Unsupported camera {cam!r}; choose from {sorted(CAMERA_MAP)}")

    # All selected tasks must share the source schema. Use first one to construct v3 features.
    first_info = _load_v2_info(_task_dir(v2_root, task_ids[0]))
    features = _v3_features(first_info, cameras)
    fps = int(args.fps or first_info.get("fps", 30))

    print("=" * 80)
    print("AgiBot LeRobot v2 -> The-Imitator-Game LeRobot v3")
    print(f"v2 source    : {v2_root}")
    print(f"v3 output    : {output_root}")
    print(f"repo id      : {args.repo_id}")
    print(f"tasks        : {len(task_ids)}")
    print(f"cameras      : {cameras}")
    print("state/action : 20/22 -> joint14 + effector2")
    print(f"video mode   : {args.video_mode}")
    print(f"resume       : {args.resume}")
    print("=" * 80)

    if args.dry_run:
        total = 0
        for task_id in task_ids:
            task_root = _task_dir(v2_root, task_id)
            info = _load_v2_info(task_root)
            eps = _iter_v2_episode_parquets(task_root)
            if args.max_episodes_per_task is not None:
                eps = eps[: args.max_episodes_per_task]
            print(
                f"task_{task_id}: episodes={len(eps)}, frames(meta)={info.get('total_frames')}, "
                f"fps={info.get('fps')}"
            )
            total += len(eps)
        print(f"dry-run total episodes: {total}")
        return

    dataset, resumed = _create_or_resume_dataset(
        output_root=output_root,
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        force=args.force,
        resume=args.resume,
    )
    converted = _load_manifest(output_root) if resumed else set()

    new_count = 0
    skipped_existing = 0
    errors = 0

    try:
        for task_id in task_ids:
            task_root = _task_dir(v2_root, task_id)
            info = _load_v2_info(task_root)
            task_fps = int(info.get("fps", fps))
            if task_fps != fps:
                raise ValueError(f"task_{task_id}: fps={task_fps} differs from target fps={fps}")

            episodes = _iter_v2_episode_parquets(task_root)
            if args.max_episodes_per_task is not None:
                episodes = episodes[: args.max_episodes_per_task]
            task_text = _load_task_text(task_info_root, task_id)

            print(f"\n[task_{task_id}] episodes={len(episodes)} | {task_text}")

            for src_ep, parquet_path in episodes:
                source_key = (task_id, src_ep)
                if source_key in converted:
                    skipped_existing += 1
                    continue

                try:
                    df = pd.read_parquet(parquet_path)
                    if V2_STATE_KEY not in df.columns or V2_ACTION_KEY not in df.columns:
                        raise KeyError(
                            f"Missing {V2_STATE_KEY!r}/{V2_ACTION_KEY!r} in {parquet_path}"
                        )

                    raw_state = _as_matrix(df[V2_STATE_KEY], 20, V2_STATE_KEY)
                    raw_action = _as_matrix(df[V2_ACTION_KEY], 22, V2_ACTION_KEY)
                    n = len(df)
                    if len(raw_state) != n or len(raw_action) != n:
                        raise ValueError("Frame count mismatch in parquet")

                    timestamps = (
                        df["timestamp"].to_numpy(dtype=np.float32)
                        if "timestamp" in df.columns
                        else np.arange(n, dtype=np.float32) / fps
                    )
                    frame_indices = (
                        df["frame_index"].to_numpy(dtype=np.int64)
                        if "frame_index" in df.columns
                        else np.arange(n, dtype=np.int64)
                    )

                    videos: dict[str, Path] = {}
                    for cam in cameras:
                        src_key = CAMERA_MAP[cam]
                        videos[f"observation.images.{cam}"] = _resolve_v2_video(
                            task_root=task_root,
                            info=info,
                            episode_index=src_ep,
                            source_video_key=src_key,
                        )

                    dst_ep = dataset.save_v2_episode(
                        task=task_text,
                        timestamps=timestamps,
                        frame_indices=frame_indices,
                        state_joint=raw_state[:, :14],
                        state_effector=raw_state[:, 14:16],
                        action_joint=raw_action[:, :14],
                        action_effector=raw_action[:, 14:16],
                        videos=videos,
                        video_mode=args.video_mode,
                    )

                    row = {
                        "task_id": task_id,
                        "source_episode_index": src_ep,
                        "destination_episode_index": dst_ep,
                        "frames": n,
                        "source_parquet": str(parquet_path),
                    }
                    _append_manifest(output_root, row)
                    converted.add(source_key)
                    new_count += 1
                    print(
                        f"  [ok] src=task_{task_id}/ep_{src_ep:06d} "
                        f"-> dst_ep={dst_ep:06d}, frames={n}"
                    )
                except Exception as exc:
                    errors += 1
                    print(
                        f"  [skip-error] task_{task_id} ep={src_ep}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if args.fail_fast:
                        raise
    finally:
        dataset.finalize()

    print("\n" + "=" * 80)
    print("Conversion finished")
    print(f"new episodes      : {new_count}")
    print(f"already converted : {skipped_existing}")
    print(f"errors            : {errors}")
    print(f"total episodes    : {dataset.meta.total_episodes}")
    print(f"total frames      : {dataset.meta.total_frames}")
    print(f"output            : {output_root}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--v2-root",
        type=Path,
        default=Path("/mnt/sdb/agi_data/data_lerobot/agibotworld"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/sdb/agi_data/data_lerobot_v3/agibotworld"),
    )
    p.add_argument("--repo-id", type=str, default="agibotworld")
    p.add_argument(
        "--task-info-root",
        type=Path,
        default=Path("/mnt/sdb/agi_data/task_info"),
    )
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument(
        "--cameras",
        nargs="+",
        default=["head", "hand_left", "hand_right"],
    )
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--video-mode", choices=["hardlink", "copy"], default="hardlink")
    p.add_argument("--max-episodes-per-task", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()
    if args.force and args.resume:
        p.error("--force and --resume are mutually exclusive")
    return args


if __name__ == "__main__":
    convert(parse_args())
