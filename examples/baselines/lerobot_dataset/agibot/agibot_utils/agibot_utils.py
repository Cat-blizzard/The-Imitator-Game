"""Raw AgiBotWorld episode reader for extracted or archived Alpha layouts.

Supported source layouts
------------------------
1. Extracted (what any4lerobot's loader expects):

    ROOT/observations/<task>/<episode>/videos/*.mp4
    ROOT/proprio_stats/<task>/<episode>/proprio_stats.h5

2. Official Hugging Face archive layout:

    ROOT/observations/<task>/<start>-<end>.tar
    ROOT/proprio_stats/<start>-<end>.tar

For archive inputs we materialize only the requested episode files into a cache;
we do not extract whole 40-50 GiB archives.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .config import ACTION_SPECS, CAMERA_SPECS, STATE_SPECS

_RANGE_RE = re.compile(r"^(\d+)-(\d+)\.tar$")
_TAR_CACHE: dict[Path, tuple[tarfile.TarFile, list[tarfile.TarInfo]]] = {}
_ARCHIVE_SCAN_CACHE: dict[Path, list[Path]] = {}


def load_task_info(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    rows.sort(key=lambda x: int(x["episode_id"]))
    return rows


def task_numeric_id(task_json: str | Path) -> str:
    """task_398.json -> '398'."""
    stem = Path(task_json).stem
    return stem.rsplit("_", 1)[-1]


def task_episode_ids(task_info: list[dict[str, Any]]) -> list[int]:
    """Use task_info as the authoritative episode list, matching current any4lerobot."""
    return sorted({int(row["episode_id"]) for row in task_info})


def list_episode_ids(src_root: str | Path, numeric_task_id: str | int) -> list[int]:
    """Legacy/extracted-layout discovery helper.

    Prefer :func:`task_episode_ids` in conversion code. This remains useful for
    diagnostics and extracted roots.
    """
    roots = [
        Path(src_root) / "observations" / str(numeric_task_id),
        Path(src_root) / "observations" / f"task_{numeric_task_id}",
    ]
    out: set[int] = set()
    for base in roots:
        if not base.exists():
            continue
        for p in base.iterdir():
            if p.is_dir() and p.name.isdigit():
                out.add(int(p.name))
    return sorted(out)


def task_instruction(task_stem: str, task_info: list[dict[str, Any]]) -> str:
    if not task_info:
        return task_stem
    first = task_info[0]
    name = str(first.get("task_name", "")).strip()
    scene = str(first.get("init_scene_text", "")).strip()
    desc = " | ".join(x for x in (name, scene) if x)
    return f"{task_stem}: {desc}" if desc else task_stem


def _all_tar_files(root: Path) -> list[Path]:
    """Return tar-like files below *root*, cached after the first scan.

    Different AgiBot download tools preserve the Hugging Face layout differently.
    Some keep ``observations/<task>/<range>.tar`` exactly, while others add an
    extra dataset/version directory.  Recursive discovery makes conversion
    independent of that packaging detail.
    """
    root = root.resolve()
    cached = _ARCHIVE_SCAN_CACHE.get(root)
    if cached is not None:
        return cached
    if not root.exists():
        out: list[Path] = []
    else:
        out = sorted(
            p for p in root.rglob("*")
            if p.is_file() and (
                p.name.endswith(".tar")
                or p.name.endswith(".tar.gz")
                or p.name.endswith(".tgz")
            )
        )
    _ARCHIVE_SCAN_CACHE[root] = out
    return out


def _parse_range_tar_name(path: Path) -> tuple[int, int] | None:
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    m = re.match(r"^(\d+)-(\d+)$", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _range_tar(directory: Path, episode_id: int, *, task_id: str | None = None) -> Path | None:
    """Find the smallest archive whose filename range covers ``episode_id``.

    First check the requested directory recursively.  If ``task_id`` is given,
    paths containing that task component are preferred over generic/global
    archives.  This supports both the official HF layout and mirrors that add
    extra nesting directories.
    """
    if not directory.exists():
        return None
    matches: list[tuple[int, int, int, Path]] = []
    task_tokens = {str(task_id), f"task_{task_id}"} if task_id is not None else set()
    for p in _all_tar_files(directory):
        parsed = _parse_range_tar_name(p)
        if parsed is None:
            continue
        lo, hi = parsed
        if lo <= int(episode_id) <= hi:
            parts = set(p.parts)
            task_penalty = 0 if (not task_tokens or parts.intersection(task_tokens)) else 1
            matches.append((task_penalty, hi - lo, lo, p))
    if not matches:
        return None
    matches.sort(key=lambda x: (x[0], x[1], x[2], str(x[3])))
    return matches[0][3]


def _candidate_nonrange_tars(directory: Path, *, task_id: str | None = None) -> list[Path]:
    """Fallback archives whose basename is not a numeric range.

    This is only used after normal range lookup fails.  Prefer archives whose
    path contains the task id, then shorter paths.
    """
    task_tokens = {str(task_id), f"task_{task_id}"} if task_id is not None else set()
    candidates = []
    for p in _all_tar_files(directory):
        if _parse_range_tar_name(p) is not None:
            continue
        parts = set(p.parts)
        penalty = 0 if (not task_tokens or parts.intersection(task_tokens)) else 1
        candidates.append((penalty, len(p.parts), str(p), p))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return [x[-1] for x in candidates]

def _open_tar(path: Path) -> tuple[tarfile.TarFile, list[tarfile.TarInfo]]:
    path = path.resolve()
    cached = _TAR_CACHE.get(path)
    if cached is not None:
        return cached
    # r:* supports plain .tar as well as mirrors that recompress to .tar.gz/.tgz.
    tf = tarfile.open(path, mode="r:*")
    members = tf.getmembers()
    _TAR_CACHE[path] = (tf, members)
    return tf, members


def close_tar_cache() -> None:
    for tf, _ in _TAR_CACHE.values():
        try:
            tf.close()
        except Exception:
            pass
    _TAR_CACHE.clear()
    _ARCHIVE_SCAN_CACHE.clear()


def _norm_member_name(name: str) -> str:
    return "/" + name.strip("./")


def _find_tar_member(path: Path, suffixes: list[str], required_parts: tuple[str, ...] = ()) -> tarfile.TarInfo:
    tf, members = _open_tar(path)
    normalized_suffixes = ["/" + s.strip("/") for s in suffixes]
    candidates: list[tarfile.TarInfo] = []
    for member in members:
        if not member.isfile():
            continue
        n = _norm_member_name(member.name)
        if required_parts and not all(f"/{part.strip('/')}" in n for part in required_parts):
            continue
        if any(n.endswith(sfx) for sfx in normalized_suffixes):
            candidates.append(member)
    if not candidates:
        raise FileNotFoundError(
            f"No matching member in archive {path}. suffixes={suffixes}, required_parts={required_parts}"
        )
    if len(candidates) > 1:
        # Prefer the shortest path (usually exactly <task>/<episode>/...).
        candidates.sort(key=lambda m: (len(Path(m.name).parts), m.name))
    return candidates[0]


def _extract_member(path: Path, member: tarfile.TarInfo, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size == member.size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tf, _ = _open_tar(path)
    fobj = tf.extractfile(member)
    if fobj is None:
        raise RuntimeError(f"Could not extract {member.name} from {path}")
    with fobj, tmp.open("wb") as out:
        shutil.copyfileobj(fobj, out, length=16 * 1024 * 1024)
    os.replace(tmp, dst)
    return dst


def _archive_cache_root(src_root: Path) -> Path:
    env = os.environ.get("AGIBOT_ARCHIVE_CACHE")
    return Path(env).expanduser().resolve() if env else src_root / ".agibot_archive_cache"


def _materialize_proprio(src_root: Path, task_id: str, episode_id: int) -> Path:
    extracted_candidates = [
        src_root / "proprio_stats" / task_id / str(episode_id) / "proprio_stats.h5",
        src_root / "proprio_stats" / f"task_{task_id}" / str(episode_id) / "proprio_stats.h5",
    ]
    for p in extracted_candidates:
        if p.exists():
            return p

    archive = _range_tar(src_root / "proprio_stats", episode_id, task_id=task_id)
    if archive is None:
        raise FileNotFoundError(
            f"No extracted proprio file and no covering .tar for task={task_id}, episode={episode_id}"
        )

    # Archive releases normally preserve <task>/<episode>/proprio_stats.h5.
    member = _find_tar_member(
        archive,
        suffixes=[
            f"{task_id}/{episode_id}/proprio_stats.h5",
            f"task_{task_id}/{episode_id}/proprio_stats.h5",
            # Some official range archives omit the task-id prefix because
            # episode IDs are globally unique.
            f"{episode_id}/proprio_stats.h5",
        ],
    )
    dst = _archive_cache_root(src_root) / "proprio_stats" / task_id / str(episode_id) / "proprio_stats.h5"
    return _extract_member(archive, member, dst)


def _resolve_raw_video(src_root: Path, numeric_task_id: str, episode_id: int, logical_camera: str) -> Path:
    if logical_camera not in CAMERA_SPECS:
        raise KeyError(f"Unknown camera: {logical_camera}")

    _, _, raw_stems = CAMERA_SPECS[logical_camera]
    observations_root = src_root / "observations"
    task_dirs = [
        observations_root / numeric_task_id,
        observations_root / f"task_{numeric_task_id}",
    ]

    # Prefer already-extracted layouts.  Besides the official
    # observations/<task>/<episode>/..., also accept mirrors that omit the task
    # level or insert extra directories.
    candidate_episode_dirs = [
        *(d / str(episode_id) for d in task_dirs),
        observations_root / str(episode_id),
    ]
    for direct in candidate_episode_dirs:
        if direct.is_dir():
            for stem in raw_stems:
                for candidate in (
                    direct / "videos" / f"{stem}_color.mp4",
                    direct / "videos" / f"{stem}.mp4",
                    direct / f"{stem}_color.mp4",
                    direct / f"{stem}.mp4",
                ):
                    if candidate.exists():
                        return candidate

    # Recursive extracted-directory fallback.  We search only for the requested
    # episode directory name, then validate an expected camera filename.
    if observations_root.exists():
        for ep_dir in observations_root.rglob(str(episode_id)):
            if not ep_dir.is_dir():
                continue
            parts = set(ep_dir.parts)
            if numeric_task_id not in parts and f"task_{numeric_task_id}" not in parts:
                # Episode IDs are globally unique in Alpha, but prefer task-scoped
                # directories and only use a task-less mirror as a fallback below.
                continue
            for stem in raw_stems:
                for candidate in (
                    ep_dir / "videos" / f"{stem}_color.mp4",
                    ep_dir / "videos" / f"{stem}.mp4",
                    ep_dir / f"{stem}_color.mp4",
                    ep_dir / f"{stem}.mp4",
                ):
                    if candidate.exists():
                        return candidate

    # Official HF layout is observations/<task>/<start>-<end>.tar.  Search the
    # whole observations subtree so mirrors with extra nesting also work.
    archive = _range_tar(observations_root, episode_id, task_id=numeric_task_id)

    suffixes: list[str] = []
    for stem in raw_stems:
        suffixes.extend([
            f"{numeric_task_id}/{episode_id}/videos/{stem}_color.mp4",
            f"{numeric_task_id}/{episode_id}/videos/{stem}.mp4",
            f"task_{numeric_task_id}/{episode_id}/videos/{stem}_color.mp4",
            f"task_{numeric_task_id}/{episode_id}/videos/{stem}.mp4",
            f"{episode_id}/videos/{stem}_color.mp4",
            f"{episode_id}/videos/{stem}.mp4",
            f"{episode_id}/{stem}_color.mp4",
            f"{episode_id}/{stem}.mp4",
        ])

    archives_to_try: list[Path] = []
    if archive is not None:
        archives_to_try.append(archive)
    # Some local mirrors rename the archive (e.g. task_398.tar).  Only inspect
    # these after the cheap numeric-range search failed.
    if archive is None:
        archives_to_try.extend(_candidate_nonrange_tars(observations_root, task_id=numeric_task_id))

    last_errors: list[str] = []
    for candidate_archive in archives_to_try:
        try:
            member = _find_tar_member(candidate_archive, suffixes=suffixes)
        except (FileNotFoundError, tarfile.TarError) as exc:
            last_errors.append(f"{candidate_archive}: {exc}")
            continue
        filename = Path(member.name).name
        dst = (
            _archive_cache_root(src_root)
            / "observations"
            / numeric_task_id
            / str(episode_id)
            / "videos"
            / filename
        )
        return _extract_member(candidate_archive, member, dst)

    standard_expected = observations_root / numeric_task_id / f"<range-containing-{episode_id}>.tar"
    nearby = []
    if observations_root.exists():
        # Keep diagnostics bounded; these names are enough to reveal local layout.
        for p in list(observations_root.rglob("*.tar"))[:12]:
            nearby.append(str(p))
    detail = "\n    ".join(nearby) if nearby else "<no .tar files found below observations>"
    extra = ("\nTried archives:\n    " + "\n    ".join(last_errors[-5:])) if last_errors else ""
    raise FileNotFoundError(
        f"Cannot locate video for task={numeric_task_id}, episode={episode_id}, camera={logical_camera}. "
        f"Expected official-style archive near {standard_expected}.\n"
        f"First local tar files found:\n    {detail}{extra}"
    )

def episode_video_paths(
    src_root: str | Path,
    numeric_task_id: str | int,
    episode_id: int,
    cameras: list[str] | tuple[str, ...],
) -> dict[str, Path]:
    root = Path(src_root)
    task_id = str(numeric_task_id)
    return {
        f"observation.images.{camera}": _resolve_raw_video(root, task_id, int(episode_id), camera)
        for camera in cameras
    }


def _require_h5_array(handle: h5py.File, path: str) -> np.ndarray:
    if path not in handle:
        raise KeyError(f"Missing HDF5 dataset {path!r} in {handle.filename}")
    return np.asarray(handle[path], dtype=np.float32)


def _align_action(
    handle: h5py.File,
    action: np.ndarray,
    *,
    index_path: str,
    fallback_state: np.ndarray,
    num_frames: int,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == len(expected_shape):
        action = action.reshape((-1, *expected_shape))
    if len(action) == num_frames:
        return action.astype(np.float32, copy=False)
    if len(action) > num_frames:
        raise ValueError(f"Action has more rows than state: {len(action)} > {num_frames}")
    if len(action) == 0:
        return np.zeros((num_frames, *expected_shape), dtype=np.float32)

    aligned = np.asarray(fallback_state, dtype=np.float32).copy()
    indices = np.asarray(handle[index_path]) if index_path in handle else np.empty((0,))
    indices = indices.reshape(-1).astype(np.int64, copy=False)
    if indices.size == len(action):
        if np.any(indices < 0) or np.any(indices >= num_frames):
            raise ValueError(
                f"Invalid sparse action indices in {index_path}: range=[{indices.min()}, {indices.max()}], frames={num_frames}"
            )
        aligned[indices] = action
        return aligned
    if len(action) == num_frames - 1 and indices.size == 0:
        aligned[: len(action)] = action
        return aligned
    raise ValueError(
        f"Cannot align action rows={len(action)} to frames={num_frames}; index rows={indices.size} at {index_path!r}"
    )


def load_control_episode(
    src_root: str | Path,
    numeric_task_id: str | int,
    episode_id: int,
    cameras: list[str] | tuple[str, ...],
) -> tuple[list[dict[str, np.ndarray]], dict[str, Path]]:
    root = Path(src_root)
    task_id = str(numeric_task_id)
    prop_path = _materialize_proprio(root, task_id, int(episode_id))

    states: dict[str, np.ndarray] = {}
    actions: dict[str, np.ndarray] = {}
    with h5py.File(prop_path, "r") as f:
        for out_key, spec in STATE_SPECS.items():
            arr = _require_h5_array(f, str(spec["h5_path"]))
            expected = tuple(spec["shape"])
            if tuple(arr.shape[1:]) != expected:
                raise ValueError(
                    f"State shape mismatch for {out_key}: got {arr.shape}, expected [T,{','.join(map(str, expected))}]"
                )
            states[out_key] = arr

        num_frames = len(next(iter(states.values())))
        if num_frames <= 0:
            raise ValueError("Episode has zero state frames")
        if any(len(v) != num_frames for v in states.values()):
            raise ValueError("State features have inconsistent frame counts: " + str({k: len(v) for k, v in states.items()}))

        for out_key, spec in ACTION_SPECS.items():
            raw = _require_h5_array(f, str(spec["h5_path"]))
            fallback = states[str(spec["fallback_state_key"])]
            actions[out_key] = _align_action(
                f,
                raw,
                index_path=str(spec["index_path"]),
                fallback_state=fallback,
                num_frames=num_frames,
                expected_shape=tuple(spec["shape"]),
            )

    frames: list[dict[str, np.ndarray]] = []
    for i in range(num_frames):
        frame: dict[str, np.ndarray] = {}
        for key, arr in states.items():
            frame[key] = np.asarray(arr[i], dtype=np.float32)
        for key, arr in actions.items():
            frame[key] = np.asarray(arr[i], dtype=np.float32)
        frames.append(frame)

    videos = episode_video_paths(root, task_id, int(episode_id), cameras)
    return frames, videos
