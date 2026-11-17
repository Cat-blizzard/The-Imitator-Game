#!/usr/bin/env python3
"""
🤖 H5 to LeRobot format conversion script V5 - GPU-aware scheduling + high parallelism

Core design:
  H5→LeRobot conversion is CPU+I/O intensive (numpy depth encoding, ffmpeg video encoding, parquet writes).
  The GPU is only occasionally used by the torch backend. Therefore:
  - --n-jobs controls the true number of parallel processes (can far exceed the GPU count)
  - GPUs are assigned via round-robin or memory-aware allocation, without limiting total parallelism
  - Each subprocess binds to a GPU via CUDA_VISIBLE_DEVICES

Usage examples:
  # 128 parallel processes, round-robin assignment across GPU 0/1/2
  python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
    --input demos --output-dir imitator_data --recursive \
    --gpu-ids 0 1 2 --gpu-mem-threshold 1024 --max-procs-per-gpu 10 \
    --n-jobs 128

  # CPU only (no GPU configured)
  python -m ... --input demos --output-dir imitator_data --n-jobs 64 --no-gpu

  # List pending files
  python -m ... --input demos --output-dir imitator_data --dry-run

  # Fine-grained GPU configuration
  python -m ... --input demos --output-dir imitator_data \
    --gpu-config "0:max_procs=20,mem_threshold=1024" \
                 "1:max_procs=20,mem_threshold=1024" \
    --n-jobs 128

H5 data format:
  - actions (N, 16): used directly as qpos_gripper_actions
  - panda_wristcam-0 qpos (N+1, 9): left arm [7 joints + 2 grippers]
  - panda_wristcam-1 qpos (N+1, 9): right arm [7 joints + 2 grippers]

LeRobot output format:
  - qpos_gripper_states (18): [left(9) + right(9)]
  - qpos_gripper_actions (16): uses H5 actions directly
"""

import os
import copy
import sys
import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback
import gc
import json
import time
import signal
import subprocess
import threading
from datetime import datetime
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import math
import multiprocessing
from dataclasses import dataclass

# GPU manager module lives in scripts/ (repo-root relative), not installed as a package.
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if _SCRIPTS_DIR.is_dir() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from gpu_manager import GPUManager, add_gpu_args, build_gpu_manager, get_all_gpus
    HAS_GPU_MANAGER = True
except ImportError:
    HAS_GPU_MANAGER = False


def get_proper_feature_names(feature_key: str, dim: int) -> List[str]:
    if "observation.images." in feature_key:
        if dim == 3:
            return ["channels", "height", "width"]
        return [f"dim_{i}" for i in range(dim)]
    if "qpos_gripper_states" in feature_key and dim == 18:
        return ([f"left_joint_{i}" for i in range(7)] +
                ["left_gripper1", "left_gripper2"] +
                [f"right_joint_{i}" for i in range(7)] +
                ["right_gripper1", "right_gripper2"])
    if "qpos_gripper_actions" in feature_key and dim == 16:
        return ([f"left_joint_{i}" for i in range(7)] + ["left_gripper1"] +
                [f"right_joint_{i}" for i in range(7)] + ["right_gripper1"])
    return [f"dim_{i}" for i in range(dim)]


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Depth encoding (identical to V4)                                ║
# ╚═══════════════════════════════════════════════════════════════╝

def rgb2hsv(rgb, *, output=None, ftype=np.float32):
    if output is None:
        output = np.empty_like(rgb)
    output[:] = 0
    h, s, v = np.split(output, 3, -1)
    h = h.squeeze(-1); s = s.squeeze(-1); v = v.squeeze(-1)
    rgb_amax = rgb.argmax(-1); rgb_max = rgb.max(-1); rgb_min = rgb.min(-1)
    r = (rgb_max - rgb_min).astype(ftype); ok = r > 0
    m = ok & (rgb_amax == 0); h[m] = 0 + (rgb[m, 1] - rgb[m, 2]) / r[m]
    m = ok & (rgb_amax == 1); h[m] = 2 + (rgb[m, 2] - rgb[m, 0]) / r[m]
    m = ok & (rgb_amax == 2); h[m] = 4 + (rgb[m, 0] - rgb[m, 1]) / r[m]
    h[:] *= 60; h[h < 0] += 360; s[ok] = r[ok] / rgb_max[ok]; v[:] = rgb_max
    return np.stack((h, s, v), -1)

def hsv2rgb(hsv, *, output=None):
    h, s, v = np.split(hsv, 3, axis=-1)
    h, s, v = h.squeeze(-1), s.squeeze(-1), v.squeeze(-1)
    h = h / 60; hi = np.floor(h).astype(int); f = h - hi
    p = v * (1 - s); q = v * (1 - s * f); t = v * (1 - s * (1 - f)); w = hi % 6
    if output is None:
        output = np.empty_like(hsv)
    m = w == 0; output[m, 0], output[m, 1], output[m, 2] = v[m], t[m], p[m]
    m = w == 1; output[m, 0], output[m, 1], output[m, 2] = q[m], v[m], p[m]
    m = w == 2; output[m, 0], output[m, 1], output[m, 2] = p[m], v[m], t[m]
    m = w == 3; output[m, 0], output[m, 1], output[m, 2] = p[m], q[m], v[m]
    m = w == 4; output[m, 0], output[m, 1], output[m, 2] = t[m], p[m], v[m]
    m = w == 5; output[m, 0], output[m, 1], output[m, 2] = v[m], p[m], q[m]
    return output

class EncoderOpts:
    def __init__(self, max_hue=300, qtype=np.uint8, ftype=np.float32,
                 err_depth=np.nan, err_rgb=(0., 0., 0.), min_v=0.1, min_s=0.1, use_lut=True):
        assert np.issubdtype(qtype, np.unsignedinteger)
        self.qinfo = np.iinfo(qtype); self.qtype = qtype; self.ftype = ftype
        self.max_hue = max_hue
        self.num_unique = int(self.max_hue / 60) * (2 ** self.qinfo.bits - 1) + 1
        self.bits = math.log(self.num_unique) / math.log(2)
        self.err_depth = err_depth
        self.err_rgb = np.array(err_rgb).astype(ftype)
        self.err_hsv = rgb2hsv(self.err_rgb[None]).squeeze().astype(ftype)
        self.min_v = min_v; self.min_s = min_s; self.use_lut = use_lut; self._enc_lut = None
    @property
    def enc_lut(self):
        if self.use_lut and self._enc_lut is None:
            self._enc_lut = _create_enc_lut(self)
        return self._enc_lut

_default_opts = EncoderOpts()

def encode(depth, *, output=None, sanitized=False, opts=None):
    opts = opts or _default_opts
    h = depth * opts.max_hue; s = np.ones_like(h); v = s
    if not sanitized:
        ok = np.isfinite(depth) & (depth >= 0) & (depth <= 1.0)
        h[~ok] = opts.err_hsv[0]; s[~ok] = opts.err_hsv[1]; v[~ok] = opts.err_hsv[2]
    return hsv2rgb(np.stack((h, s, v), -1), output=output)

def quantize(x, *, opts=None):
    opts = opts or _default_opts
    return np.round(np.iinfo(opts.qtype).max * x).astype(opts.qtype)

def encode_lut(depth, *, output=None, sanitized=True, opts=None):
    opts = opts or _default_opts
    if output is None:
        output = np.empty(depth.shape + (3,), dtype=opts.qtype)
    idx = np.round(depth * (opts.num_unique - 1)).astype(int)
    if sanitized:
        output[:] = opts.enc_lut[idx]
    else:
        ok = (idx >= 0) & (idx < opts.num_unique)
        output[:] = opts.err_rgb; output[ok] = opts.enc_lut[idx[ok]]
    return output

def _create_enc_lut(opts=None):
    opts = opts or _default_opts
    d = np.linspace(0, 1.0, opts.num_unique, dtype=opts.ftype)
    return quantize(encode(d, opts=opts), opts=opts)

def depth2rgb(d, zrange, *, output=None, sanitized=False, inv_depth=False, opts=None):
    opts = opts or _default_opts
    if inv_depth:
        zmin, zmax = 1 / zrange[1], 1 / zrange[0]; zrange = (zmin, zmax)
        with np.errstate(divide="ignore"): d = 1 / d
    d = (d - zrange[0]) / (zrange[1] - zrange[0])
    return (encode_lut(d, output=output, sanitized=sanitized, opts=opts) if opts.use_lut
            else encode(d, output=output, sanitized=sanitized, opts=opts))


def get_traj_names(h5_file):
    return sorted([k for k in h5_file.keys() if k.startswith('traj_')])

def native_data_contract(h5_file: h5py.File, action_width: int) -> dict:
    """Read native controller/joint order; old dual-Panda recordings remain supported."""
    sidecar = Path(h5_file.filename).with_suffix(".json")
    metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    sync = metadata.get("embodiment_sync", {})
    order = sync.get("action_order")
    if order is None:
        # The original TIG recorder flattens these two native controllers in this order.
        order = [{"key": f"panda_wristcam-{side}", "size": 8,
                  "arm_joint_names": [f"panda_joint{i}" for i in range(1, 8)],
                  "state_joint_names": [f"panda_joint{i}" for i in range(1, 8)]
                  + ["panda_finger_joint1", "panda_finger_joint2"]} for side in range(2)]
    if not isinstance(order, list) or len(order) != 2:
        raise ValueError("A recording must describe two native action controllers")
    for item in order:
        arm, state = item.get("arm_joint_names", []), item.get("state_joint_names", [])
        if (not isinstance(item.get("key"), str) or not arm or not state
                or len(set(arm)) != len(arm) or len(set(state)) != len(state)
                or not set(arm).issubset(state) or item.get("size") != len(arm)+1):
            raise ValueError("Invalid recorded native joint/action order")
    if len({item["key"] for item in order}) != 2 or sum(item["size"] for item in order) != action_width:
        raise ValueError("Recorded action order differs from the H5 action width")
    # Old Panda/Piper archives omitted these descriptive fields. Normalize only
    # their established defaults; explicit native radian gripper endpoints survive.
    order = [dict(item) for item in order]
    for item in order:
        item.setdefault("arm_coordinate_unit", "rad")
        item.setdefault("gripper_coordinate_unit", "m")
        item.setdefault("gripper_open_action", 1.0)
        item.setdefault("gripper_closed_action", -1.0)
    env = metadata.get("env_info", {})
    kwargs = env.get("env_kwargs", {})
    return {"action_order": order, "env_id": env.get("env_id"),
            "robot_uids": kwargs.get("robot_uids"), "control_mode": kwargs.get("control_mode", "pd_joint_pos"),
            "control_freq": sync.get("control_freq"),
            "rendering": {"shader_dir": kwargs.get("shader_dir"),
                          "sensor_configs": kwargs.get("sensor_configs", {})}}


def load_trajectory_data(h5_file: h5py.File, traj_name: str,
                         drop_terminal_frame: bool = False) -> dict:
    """T native actions and T+1 observations; supervised row i is (obs[i], action[i])."""
    traj = h5_file[traj_name]
    actions = traj['actions'][()].astype(np.float32)
    if actions.ndim != 2 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError(f"{traj_name}: expected nonempty finite (T, native action width) actions")
    num_frames = len(actions) + 1
    contract = native_data_contract(h5_file, actions.shape[1])
    positions = []
    for item in contract["action_order"]:
        qpos = traj['obs']['agent'][item["key"]]['qpos'][()].astype(np.float32)
        if qpos.shape != (num_frames, len(item["state_joint_names"])) or not np.isfinite(qpos).all():
            raise ValueError(f"{traj_name}: {item['key']} qpos must have shape (T + 1, named state width)")
        positions.append(qpos)
    sensor_data = traj['obs']['sensor_data']
    rgb_data, depth_data = {}, {}
    aliases = {'panda_wristcam_0_hand_camera': 'wristcam0',
               'panda_wristcam_1_hand_camera': 'wristcam1'}
    for h5_cam, cg in sensor_data.items():
        lr_cam = aliases.get(h5_cam, h5_cam)
        if 'rgb' in cg:
            rgb_data[lr_cam] = cg['rgb'][()]
        if 'depth' in cg:
            depth_data[lr_cam] = cg['depth'][()]
    for camera, frames in list(rgb_data.items()) + list(depth_data.items()):
        if len(frames) != num_frames:
            raise ValueError(f"{traj_name}: camera {camera} must contain T + 1 frames")
    if drop_terminal_frame:
        num_frames = len(actions)
        positions = [qpos[:-1] for qpos in positions]
        rgb_data = {key: frames[:-1] for key, frames in rgb_data.items()}
        depth_data = {key: frames[:-1] for key, frames in depth_data.items()}
    return {'actions': actions, 'qpos_left': positions[0], 'qpos_right': positions[1],
            'rgb_data': rgb_data, 'depth_data': depth_data, 'num_frames': num_frames,
            'data_contract': contract}

def process_depth_images(depth_data, verbose=False):
    processed = {}
    for cam_name, depth_imgs in depth_data.items():
        if verbose: print(f"    Processing {cam_name} depth: {depth_imgs.shape}")
        N, H, W, _ = depth_imgs.shape
        if "wrist" in cam_name.lower() or "wristcam" in cam_name.lower(): zrange = (0., 1.)
        elif "cam2" in cam_name.lower(): zrange = (0., 2.)
        else: zrange = (0., 3.)
        depth_rgb_list = []
        for i in range(N):
            depth_m = depth_imgs[i, :, :, 0].astype(np.float32) / 1000.0
            depth_rgb_list.append(depth2rgb(depth_m, zrange).transpose(2, 0, 1))
        processed[cam_name] = np.stack(depth_rgb_list, axis=0)
    return processed


def build_lerobot_features(sample):
    features = {}
    for cam_name, rgb in sample['rgb_data'].items():
        _, H, W, C = rgb.shape
        fk = f"observation.images.{cam_name}"
        features[fk] = {"dtype": "video", "shape": (C, H, W), "names": get_proper_feature_names(fk, C)}
    for cam_name in sample['depth_data']:
        _, C, H, W = sample['processed_depth'][cam_name].shape
        fk = f"observation.images.{cam_name}_depth"
        features[fk] = {"dtype": "video", "shape": (C, H, W), "names": get_proper_feature_names(fk, C)}
    order = sample["data_contract"]["action_order"]
    state_names = [f"{item['key']}/{joint}" for item in order for joint in item["state_joint_names"]]
    action_names = [f"{item['key']}/{joint}" for item in order
                    for joint in item["arm_joint_names"]+["gripper"]]
    fk = "observation.qpos_gripper_states"
    features[fk] = {"dtype": "float32", "shape": (len(state_names),), "names": state_names}
    fk = "action.qpos_gripper_actions"
    features[fk] = {"dtype": "float32", "shape": (len(action_names),), "names": action_names}
    return features

def get_h5_trajectory_count(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f: return len(get_traj_names(f))
    except: return 0

def check_dataset_complete(dataset_dir, h5_path=None, fps=None, drop_terminal_frame=None):
    if not dataset_dir.exists(): return False, "directory does not exist"
    if (dataset_dir / ".conversion-failed.json").exists():
        return False, "previous conversion failed"
    info_file = dataset_dir / "meta" / "info.json"
    if not info_file.exists(): return False, "missing meta/info.json"
    try:
        with open(info_file) as f: info = json.load(f)
        actual = info.get('total_episodes', 0)
    except: return False, "cannot read info.json"
    if actual == 0: return False, "episode=0"
    if fps is not None and info.get("fps") != fps:
        return False, "sampling rate differs from requested fps"
    if drop_terminal_frame is not None:
        expected = "dropped" if drop_terminal_frame else "duplicate_last_action"
        if info.get("terminal_observation", "duplicate_last_action") != expected:
            return False, "terminal observation policy differs from requested policy"
    if h5_path and h5_path.exists():
        expected = get_h5_trajectory_count(h5_path)
        if expected == 0: return True, f"assumed complete ({actual} eps)"
        if actual == expected:
            try:
                with open(dataset_dir / ".complete", 'w') as f:
                    f.write(json.dumps({"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "episodes": actual, "auto_verified": True}, indent=2))
            except: pass
            return True, f"complete ({actual}/{expected} eps)"
        return False, f"mismatch ({actual} vs {expected})"
    return True, f"assumed complete ({actual} eps)"

def cleanup_failed_dataset(dataset_dir, name):
    if dataset_dir.exists():
        print(f"  [{name}] 🧹 Cleaning up incomplete dataset")
        try:
            import shutil; shutil.rmtree(dataset_dir, ignore_errors=True)
        except Exception as e:
            print(f"  [{name}] ⚠️ {e}")


@dataclass(frozen=True)
class EpisodeSource:
    h5_path: Path
    episode_id: int
    seed: Optional[int] = None
    split: Optional[str] = None

    @property
    def trajectory(self) -> str:
        return f"traj_{self.episode_id}"


def load_episode_manifest(manifest_path: Path) -> Tuple[Dict[str, List[int]], List[EpisodeSource]]:
    """Select recorded episodes without changing their preassigned seed split.

    Schema: {"schema_version": 1, "seed_splits": {"train": [...],
    "development": [...], "evaluation": [...]}, "episodes": [{"h5_path":
    "relative/or/absolute.h5", "episode_id": 0, "seed": 123, "split": "train"}]}.
    Evaluation seeds are reserved for actual policy evaluation, never converted.
    """
    manifest_path = Path(manifest_path)
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("episode manifest requires schema_version=1")
    seed_splits = manifest.get("seed_splits", {})
    if set(seed_splits) != {"train", "development", "evaluation"}:
        raise ValueError("seed_splits must define train, development and evaluation")
    assigned = set()
    for split, seeds in seed_splits.items():
        if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds):
            raise ValueError(f"{split} seeds must be an integer list")
        if len(seeds) != len(set(seeds)) or assigned.intersection(seeds):
            raise ValueError("seed splits must contain unique, disjoint seeds")
        assigned.update(seeds)
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("episode manifest must select at least one episode")
    sources, seen_seeds, seen_sources = [], set(), set()
    for entry in entries:
        seed, episode_id, split = entry.get("seed"), entry.get("episode_id"), entry.get("split")
        if type(seed) is not int or type(episode_id) is not int or episode_id < 0:
            raise ValueError("each episode requires an integer seed and nonnegative episode_id")
        if split not in ("train", "development") or seed not in seed_splits[split]:
            raise ValueError(f"seed {seed} is not assigned to the requested training/development split")
        raw_path = entry.get("h5_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("each episode requires h5_path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        identity = (path, episode_id)
        if seed in seen_seeds or identity in seen_sources:
            raise ValueError("each seed and source episode may be selected only once")
        seen_seeds.add(seed)
        seen_sources.add(identity)
        sources.append(EpisodeSource(path, episode_id, seed, split))
    return seed_splits, sources


def _source_record(source: EpisodeSource, dataset_dir: Path, episode_id: int) -> dict:
    return {
        "episode_id": episode_id, "seed": source.seed, "split": source.split,
        "source_path": os.path.relpath(source.h5_path.resolve(), dataset_dir.resolve()),
        "source_episode_id": source.episode_id,
    }


def _convert_episodes(sources: List[EpisodeSource], dataset_dir: Path, dataset_name: str,
                      fps: int, drop_terminal_frame: bool, gpu_id: int,
                      seed_splits: Optional[Dict[str, List[int]]] = None) -> Tuple[bool, str, str]:
    """Use the original LeRobot writer for either one H5 or an explicit episode selection.

    An episode is counted only after save_episode succeeds. A failed write stops
    this conversion, so its pending images/frames cannot leak into another episode.
    Partial output is retained for diagnosis and is never marked complete.
    """
    from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset

    dataset, finalized, current_source = None, False, None
    converted = []
    try:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not sources:
            raise ValueError("no trajectory selected")
        features = None
        for source in sources:
            current_source = source
            with h5py.File(source.h5_path, "r") as h5_file:
                traj_data = load_trajectory_data(h5_file, source.trajectory, drop_terminal_frame)
            traj_data["processed_depth"] = process_depth_images(traj_data["depth_data"])
            source_features = build_lerobot_features(traj_data)
            native_contract = traj_data["data_contract"]
            if native_contract["control_freq"] is not None and not np.isclose(native_contract["control_freq"], fps):
                raise ValueError("Requested fps differs from the recorded control frequency")
            if dataset is None:
                features = source_features
                dataset_dir.parent.mkdir(parents=True, exist_ok=True)
                dataset = LeRobotDataset.create(
                    repo_id=dataset_name, fps=fps, root=str(dataset_dir),
                    features=copy.deepcopy(features), use_videos=True, video_backend="torchcodec",
                )
                dataset.meta.info["terminal_observation"] = (
                    "dropped" if drop_terminal_frame else "duplicate_last_action"
                )
                dataset.meta.info["native_data_contract"] = native_contract
            elif native_contract != dataset.meta.info["native_data_contract"]:
                raise ValueError("Selected episodes have different robot/controller contracts")
            elif source_features != features:
                raise ValueError(f"{source.trajectory}: camera/features differ from the first selected episode")
            qpos_states = np.concatenate([traj_data["qpos_left"], traj_data["qpos_right"]], axis=1)
            qpos_actions = traj_data["actions"]
            if not drop_terminal_frame:
                qpos_actions = np.vstack([qpos_actions, qpos_actions[-1:]])
            for i in range(traj_data["num_frames"]):
                frame = {"task": dataset_name}
                for camera, rgb in traj_data["rgb_data"].items():
                    frame[f"observation.images.{camera}"] = rgb[i].transpose(2, 0, 1)
                for camera, depth in traj_data["processed_depth"].items():
                    frame[f"observation.images.{camera}_depth"] = depth[i]
                frame["observation.qpos_gripper_states"] = qpos_states[i]
                frame["action.qpos_gripper_actions"] = qpos_actions[i]
                dataset.add_frame(frame)
            dataset.save_episode()
            converted.append(_source_record(source, dataset_dir, len(converted)))
            del traj_data
            gc.collect()

        total_episodes = len(converted)
        if seed_splits is None:
            # Preserve the original default split for ordinary --input use.
            shuffled = np.random.permutation(total_episodes).tolist()
            train_n = int(0.95 * total_episodes)
            splits = {"train": shuffled[:train_n], "test": shuffled[train_n:],
                      dataset_name: list(range(total_episodes))}
        else:
            splits = {
                "train": [entry["episode_id"] for entry in converted if entry["split"] == "train"],
                "validation": [entry["episode_id"] for entry in converted if entry["split"] == "development"],
            }
        dataset.meta.info["splits"] = splits
        dataset.finalize()
        finalized = True
        meta_dir = dataset_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        # meta.save_episode writes a default all-train split. Persist our final
        # assignment in info.json, rather than setting an unused meta.splits attribute.
        (meta_dir / "info.json").write_text(json.dumps(dataset.meta.info, indent=2) + "\n")
        if seed_splits is not None:
            (meta_dir / "source_episodes.json").write_text(json.dumps({
                "schema_version": 1, "fps": fps,
                "terminal_observation": dataset.meta.info["terminal_observation"],
                "seed_splits": seed_splits, "episodes": converted,
            }, indent=2) + "\n")
        (dataset_dir / ".complete").write_text(json.dumps({
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "episodes": total_episodes, "gpu_id": gpu_id, "fps": fps,
            "drop_terminal_frame": drop_terminal_frame,
        }, indent=2) + "\n")
        return True, dataset_name, f"success: {total_episodes} episodes"
    except Exception as error:
        message = f"error: {error}"
        if dataset is not None and not finalized:
            try:
                dataset.finalize()
            except Exception as close_error:
                message += f"; closing partial output: {close_error}"
        if dataset_dir.exists():
            (dataset_dir / ".complete").unlink(missing_ok=True)
            (dataset_dir / ".conversion-failed.json").write_text(json.dumps({
                "error": message, "saved_episodes": len(converted),
                "failed_source": _source_record(current_source, dataset_dir, len(converted))
                if current_source is not None else None,
            }, indent=2) + "\n")
        return False, dataset_name, message


def process_episode_manifest(manifest_path_str: str, output_dir_str: str,
                             dataset_name: str, fps: int = 30, gpu_id: int = -1,
                             drop_terminal_frame: bool = False) -> Tuple[bool, str, str]:
    """Write a new, immutable dataset version selected from one or more H5 files."""
    try:
        if not dataset_name or Path(dataset_name).name != dataset_name or dataset_name in (".", ".."):
            raise ValueError("manifest conversion requires a dataset --name without path components")
        seed_splits, sources = load_episode_manifest(Path(manifest_path_str))
        dataset_dir = Path(output_dir_str) / dataset_name
        if dataset_dir.exists():
            raise ValueError("output already exists; choose a new dataset version name")
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return _convert_episodes(sources, dataset_dir, dataset_name, fps,
                                 drop_terminal_frame, gpu_id, seed_splits)
    except Exception as error:
        return False, dataset_name, f"error: {error}"


def process_h5_file(h5_path_str: str, output_dir_str: str, fps: int = 30,
                    dataset_name: str = None, force: bool = False,
                    gpu_id: int = -1, drop_terminal_frame: bool = False) -> Tuple[bool, str, str]:
    """
    Process a single H5 file → LeRobot format.
    Designed to run independently inside a ProcessPoolExecutor worker.
    Arguments use str rather than Path for pickle compatibility.
    """
    h5_path = Path(h5_path_str)
    output_dir = Path(output_dir_str)

    try:
        if dataset_name is None:
            dataset_name = h5_path.stem
        if Path(dataset_name).name != dataset_name or dataset_name in (".", ".."):
            raise ValueError("dataset name must not contain path components")
        dataset_dir = output_dir / dataset_name
        if h5_path.resolve().is_relative_to(dataset_dir.resolve()):
            raise ValueError("output dataset directory must not contain the source H5")

        # Bind GPU
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            tag = f"GPU{gpu_id}"
        else:
            tag = "CPU"

        # Check completeness
        if not force:
            is_complete, reason = check_dataset_complete(
                dataset_dir, h5_path, fps=fps, drop_terminal_frame=drop_terminal_frame
            )
            if is_complete:
                return True, dataset_name, f"already processed: {reason}"
            if check_dataset_complete(dataset_dir, h5_path)[0]:
                return False, dataset_name, f"{reason}; choose a different output directory"

        # Explicit --force retains its original reprocessing meaning. Ordinary
        # retries also replace incomplete *derived* output; raw H5 files are read-only.
        if dataset_dir.exists():
            cleanup_failed_dataset(dataset_dir, dataset_name)

        with h5py.File(h5_path, "r") as h5_file:
            sources = [EpisodeSource(h5_path, int(name.removeprefix("traj_")))
                       for name in get_traj_names(h5_file)]
        print(f"  [{dataset_name}] [{tag}] Starting conversion...")
        return _convert_episodes(sources, dataset_dir, dataset_name, fps,
                                 drop_terminal_frame, gpu_id)
    except Exception as error:
        return False, dataset_name or h5_path.stem, f"error: {error}"


# ╔═══════════════════════════════════════════════════════════════╗
# ║  GPU assigner (Round-Robin + memory-aware)                        ║
# ╚═══════════════════════════════════════════════════════════════╝

class GPUAssigner:
    """
    Lightweight GPU assigner — does not limit parallelism, only assigns GPU IDs.

    Difference from GPUManager:
      GPUManager: acquire/release model, parallelism = total number of GPU slots
      GPUAssigner: only assigns, parallelism is controlled by --n-jobs

    Strategy:
      - round_robin: simple round robin (lowest overhead, recommended)
      - mem_aware: queries free VRAM and prefers idle cards
    """
    def __init__(self, gpu_ids: List[int], strategy: str = "round_robin",
                 mem_threshold_mib: int = 1024):
        self.gpu_ids = gpu_ids if gpu_ids else []
        self.strategy = strategy
        self.mem_threshold_mib = mem_threshold_mib
        self._counter = 0
        self._lock = Lock()

    def assign(self) -> int:
        if not self.gpu_ids:
            return -1
        with self._lock:
            if self.strategy == "round_robin":
                gid = self.gpu_ids[self._counter % len(self.gpu_ids)]
                self._counter += 1
                return gid
            else:
                # mem_aware fallback
                fallback = self.gpu_ids[self._counter % len(self.gpu_ids)]
                self._counter += 1
                try:
                    info = _query_nvidia_smi_quick()
                    best = [(info[g][1] - info[g][0], g) for g in self.gpu_ids
                            if g in info and (info[g][1] - info[g][0]) >= self.mem_threshold_mib]
                    if best:
                        best.sort(key=lambda x: -x[0])
                        return best[0][1]
                except: pass
                return fallback

    @property
    def info_str(self) -> str:
        if not self.gpu_ids: return "no GPU"
        return f"GPU {self.gpu_ids} ({self.strategy})"


def _query_nvidia_smi_quick():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10).decode()
        result = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                result[int(parts[0])] = (int(parts[1]), int(parts[2]))
        return result
    except: return {}


def _worker_fn(args_tuple):
    """ProcessPoolExecutor worker. Receives a tuple for map() compatibility."""
    h5_path_str, output_dir_str, fps, force, gpu_id, drop_terminal_frame = args_tuple
    return process_h5_file(h5_path_str, output_dir_str, fps=fps,
                           dataset_name=None, force=force, gpu_id=gpu_id, drop_terminal_frame=drop_terminal_frame)


class ConversionScheduler:

    def __init__(self, gpu_assigner: GPUAssigner, args):
        self.gpu_assigner = gpu_assigner
        self.args = args
        self._stop = False

    def collect_h5_files(self, input_dir: Path, output_dir: Path) -> List[Path]:
        if self.args.recursive:
            h5_files = sorted(list(input_dir.rglob("*.h5")))
        else:
            h5_files = sorted(list(input_dir.glob("*.h5")))
        if not h5_files:
            print(f"No H5 files found in {input_dir}")
            return []
        if not self.args.force:
            unprocessed = []
            for f in h5_files:
                ok, reason = check_dataset_complete(
                    output_dir / f.stem, f, fps=self.args.fps,
                    drop_terminal_frame=self.args.drop_terminal_frame
                )
                if not ok:
                    unprocessed.append(f)
                else:
                    print(f"  ✓ Skipping {f.stem}: {reason}")
            h5_files = unprocessed
        return h5_files

    def run(self, input_dir: Path, output_dir: Path) -> List[Tuple[bool, str, str]]:
        h5_files = self.collect_h5_files(input_dir, output_dir)
        if not h5_files:
            print("\nAll files already processed. Use --force to reprocess.")
            return []

        n_jobs = self.args.n_jobs
        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()

        # Do not exceed the number of files
        n_jobs = min(n_jobs, len(h5_files))

        # Memory limit
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            mem_per = self.args.mem_per_proc
            max_by_mem = max(1, int(avail_gb / mem_per))
            if n_jobs > max_by_mem:
                print(f"  ⚠️ Memory limit: {avail_gb:.0f}GB available / {mem_per}GB per process → "
                      f"limiting to {max_by_mem} parallel")
                n_jobs = max_by_mem
        except ImportError:
            pass

        print(f"\n{'='*70}")
        print(f"📊 Pending: {len(h5_files)} H5 files")
        print(f"🔧 Parallel processes: {n_jobs}")
        print(f"🖥️  GPU assignment: {self.gpu_assigner.info_str}")
        print(f"{'='*70}\n")

        # Pre-assign a GPU ID for each file (round-robin)
        task_args = []
        for h5_file in h5_files:
            gpu_id = self.gpu_assigner.assign()
            task_args.append((str(h5_file), str(output_dir), self.args.fps,
                              self.args.force, gpu_id, self.args.drop_terminal_frame))

        # Ctrl+C handling
        original_sigint = signal.getsignal(signal.SIGINT)
        def _sig(sig, frame):
            print("\n⚠️ Interrupted! Waiting for current tasks to finish...")
            self._stop = True
            signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGINT, _sig)

        results = []
        start_time = time.time()

        # ProcessPoolExecutor — true multi-process parallelism
        # spawn context avoids fork + CUDA conflicts
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
            future_to_file = {}
            for args_t, h5_file in zip(task_args, h5_files):
                if self._stop: break
                fut = executor.submit(_worker_fn, args_t)
                future_to_file[fut] = h5_file

            done_count = 0
            for fut in as_completed(future_to_file):
                h5_file = future_to_file[fut]
                try:
                    result = fut.result(timeout=self.args.task_timeout)
                    results.append(result)
                except Exception as e:
                    results.append((False, h5_file.stem, f"exception: {e}"))

                done_count += 1
                elapsed = time.time() - start_time
                rate = done_count / elapsed * 3600 if elapsed > 0 else 0

                # Print progress periodically
                interval = max(1, len(h5_files) // 20)
                if done_count % interval == 0 or done_count == len(h5_files):
                    ok_n = sum(1 for r in results if r[0])
                    fail_n = len(results) - ok_n
                    eta_min = (len(h5_files) - done_count) / (done_count / elapsed) / 60 if elapsed > 0 else 0
                    print(f"  📈 [{done_count}/{len(h5_files)}] "
                          f"{ok_n}✓ {fail_n}✗ "
                          f"| {rate:.0f}/h "
                          f"| elapsed {elapsed/60:.1f}min "
                          f"| ETA {eta_min:.1f}min")

        signal.signal(signal.SIGINT, original_sigint)
        return results


# ╔═══════════════════════════════════════════════════════════════╗
# ║  argparse GPU arguments (fallback when external gpu_manager is absent) ║
# ╚═══════════════════════════════════════════════════════════════╝

if not HAS_GPU_MANAGER:
    def add_gpu_args(parser):
        g = parser.add_argument_group("GPU configuration")
        g.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                        help="which GPUs to use")
        g.add_argument("--gpu-config", type=str, nargs="+", default=None,
                        help="fine-grained config (kept for compatibility; this version uses round-robin)")
        g.add_argument("--max-procs-per-gpu", type=int, default=2,
                        help="compatibility arg (actual parallelism is controlled by --n-jobs)")
        g.add_argument("--gpu-mem-threshold", type=int, default=4096,
                        help="VRAM threshold (MiB)")


def main():
    parser = argparse.ArgumentParser(
        description="🤖 H5→LeRobot V5 — GPU-aware scheduling + high parallelism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Basic arguments
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=str, help="input H5 file or directory")
    source.add_argument("--episode-manifest", type=str,
                        help="JSON selection of H5 episodes and fixed train/development/evaluation seed pools")
    parser.add_argument("--output-dir", type=str, required=True, help="output directory")
    parser.add_argument("--fps", type=int, default=30, help="frame rate (default 30)")
    parser.add_argument("--drop-terminal-frame", action="store_true",
                        help="Emit T training rows for T actions, omitting the terminal observation without an action. Default preserves T+1 rows and duplicates the final action.")
    parser.add_argument("--name", type=str, default=None, help="dataset name (single file)")
    parser.add_argument("--force", action="store_true", help="force reprocessing")
    parser.add_argument("--recursive", action="store_true", help="recursively search subdirectories")

    # Parallelism arguments
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="number of parallel processes (-1=auto, recommended=CPU cores×0.8)")
    parser.add_argument("--mem-per-proc", type=float, default=4.0,
                        help="estimated memory per process (GB); auto-limits parallelism (default 4.0)")
    parser.add_argument("--task-timeout", type=int, default=7200,
                        help="per-task timeout in seconds (default 7200)")

    # GPU arguments
    add_gpu_args(parser)
    parser.add_argument("--no-gpu", action="store_true", help="disable GPU")
    parser.add_argument("--gpu-strategy", type=str, default="round_robin",
                        choices=["round_robin", "mem_aware"],
                        help="GPU assignment strategy (default round_robin)")

    # Utilities
    parser.add_argument("--dry-run", action="store_true", help="only list pending files")

    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.episode_manifest and not args.name:
        parser.error("--episode-manifest requires --name for the new dataset version")
    if args.episode_manifest and args.force:
        parser.error("manifest outputs are immutable; choose a new --name instead of --force")

    input_path = Path(args.episode_manifest or args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Detect GPUs ──
    gpu_ids = []
    if not args.no_gpu:
        if args.gpu_ids:
            gpu_ids = args.gpu_ids
        else:
            info = _query_nvidia_smi_quick()
            gpu_ids = sorted(info.keys())
        if gpu_ids:
            info = _query_nvidia_smi_quick()
            print(f"\n🖥️  GPU ({len(gpu_ids)} detected):")
            for gid in gpu_ids:
                if gid in info:
                    used, total = info[gid]
                    free = total - used
                    bar_len = int(free / total * 20) if total > 0 else 0
                    bar = '█' * bar_len + '░' * (20 - bar_len)
                    print(f"    GPU{gid}: [{bar}] {free:,}MiB / {total:,}MiB")
        else:
            print("\n⚠️ No GPU detected")

    gpu_assigner = GPUAssigner(gpu_ids, args.gpu_strategy, args.gpu_mem_threshold)

    print(f"\n{'='*70}")
    print("🤖 H5 → LeRobot conversion V5 — GPU-aware scheduling + high parallelism")
    print(f"{'='*70}")
    print(f"  Input:     {input_path}")
    print(f"  Output:    {output_dir}")
    print(f"  FPS:       {args.fps}")
    print(f"  Processes: {args.n_jobs if args.n_jobs > 0 else 'auto'}")
    print(f"  Mem/proc:  {args.mem_per_proc} GB")
    print(f"  GPU:       {gpu_assigner.info_str}")
    print(f"  Recursive: {'yes' if args.recursive else 'no'}")
    print(f"  Force:     {'yes' if args.force else 'no'}")

    # System resource overview
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu_count = multiprocessing.cpu_count()
        print(f"  CPU cores: {cpu_count}")
        print(f"  RAM:       {mem.available/1024**3:.0f}GB available / {mem.total/1024**3:.0f}GB total")
        print(f"  Max async: ~{int(mem.available/1024**3/args.mem_per_proc)} (memory-bound)")
    except ImportError:
        pass
    print(f"{'='*70}")

    # ── Explicit selection across H5 files ──
    if args.episode_manifest:
        if args.dry_run:
            seed_splits, sources = load_episode_manifest(input_path)
            print(f"Selected: {len(sources)} episodes; reserved evaluation seeds: {len(seed_splits['evaluation'])}")
            for episode in sources:
                print(f"  {episode.split}: seed={episode.seed} {episode.h5_path}:{episode.trajectory}")
            return 0
        ok, _, message = process_episode_manifest(
            str(input_path), str(output_dir), args.name, fps=args.fps,
            gpu_id=gpu_assigner.assign(), drop_terminal_frame=args.drop_terminal_frame,
        )
        print(f"\nResult: {'✓' if ok else '✗'} {message}")
        return 0 if ok else 1

    # ── Single file ──
    if input_path.is_file():
        if input_path.suffix != '.h5':
            print(f"error: {input_path.suffix} is not supported")
            return 1
        gpu_id = gpu_assigner.assign()
        ok, name, msg = process_h5_file(str(input_path), str(output_dir),
                                        args.fps, args.name, args.force, gpu_id, args.drop_terminal_frame)
        print(f"\nResult: {'✓' if ok else '✗'} {msg}")
        return 0 if ok else 1

    # ── Directory ──
    if not input_path.is_dir():
        print(f"error: path does not exist {input_path}")
        return 1

    scheduler = ConversionScheduler(gpu_assigner, args)

    if args.dry_run:
        h5_files = scheduler.collect_h5_files(input_path, output_dir)
        total_size = sum(f.stat().st_size for f in h5_files) / 1024**3
        print(f"\nPending: {len(h5_files)} files ({total_size:.1f} GB)")
        for f in h5_files[:20]:
            print(f"  ○ {f.stem}  ({f.stat().st_size / 1024**3:.1f} GB)")
        if len(h5_files) > 20:
            print(f"  ... {len(h5_files) - 20} more")
        return

    results = scheduler.run(input_path, output_dir)

    if not results:
        print("\nNo files were processed.")
        return

    # ── Summary ──
    success_n = sum(1 for ok, _, msg in results if ok and "already processed" not in msg)
    skip_n = sum(1 for ok, _, msg in results if ok and "already processed" in msg)
    fail_n = len(results) - success_n - skip_n

    print(f"\n{'='*70}")
    print(f"📊 Processing complete")
    print(f"{'='*70}")
    print(f"  Total: {len(results)}  |  ✅Success: {success_n}  |  ⊙Skipped: {skip_n}  |  ❌Failed: {fail_n}")

    if fail_n > 0:
        print(f"\n❌ Failed:")
        for ok, name, msg in results:
            if not ok: print(f"    ✗ {name}: {msg[:200]}")

    if success_n > 0:
        print(f"\n✅ Success:")
        for ok, name, msg in results:
            if ok and "already processed" not in msg: print(f"    ✓ {name}: {msg}")

    print(f"\n{'='*70}")
    return 1 if fail_n else 0


if __name__ == "__main__":
    sys.exit(main())
