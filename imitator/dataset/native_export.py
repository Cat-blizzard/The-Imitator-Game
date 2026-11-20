"""Export recorded native robot trajectories to LeRobot without Panda dimensions.

RecordEpisode RGB/qpos archives and certified data-engine archives share this
entry. Certification metadata remains separate from the controller contract.
For noisy certified data, actions_clean are labels and actions remain replay
commands. Only public agent qpos is exported from a flat state observation.

    python -m imitator.dataset.native_export --input trajectory.h5 --output-dir data --name task
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


STATE_KEY = "observation.qpos_gripper_states"
ACTION_KEY = "action.qpos_gripper_actions"


def _validate_contract(contract: dict[str, Any], width: int) -> dict[str, Any]:
    result = deepcopy(contract)
    order = result.get("action_order", result.get("action_layout"))
    if not isinstance(order, list) or len(order) != 2:
        raise ValueError("Native export requires two explicitly recorded action controllers")
    for item in order:
        arm, state = item.get("arm_joint_names", []), item.get("state_joint_names", [])
        if (not item.get("key") or not arm or not state or len(set(state)) != len(state)
                or len(set(arm)) != len(arm) or not set(arm).issubset(state)):
            raise ValueError("Native export requires unique recorded arm/state joint names")
        if item.get("size") != len(arm) + 1:
            raise ValueError("Native export currently supports scalar parallel grippers, not Allegro hands")
    if len({item["key"] for item in order}) != 2 or sum(item["size"] for item in order) != width:
        raise ValueError("Recorded native action layout differs from the trajectory")
    if result.get("control_mode", "pd_joint_pos") != "pd_joint_pos":
        raise ValueError("Native joint-target export requires pd_joint_pos")
    result["action_order"] = order
    return result


def recorded_contract(metadata: dict[str, Any], width: int,
                      supplement: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read recorded identities; never derive old model settings from live assets."""
    imsim = metadata.get("imsim", {})
    contract = metadata.get("native_data_contract") or imsim.get("native_data_contract")
    if supplement is not None:
        if contract is not None and contract != supplement:
            raise ValueError("Supplement differs from the recorded native contract")
        contract = supplement
    if contract is not None:
        return _validate_contract(contract, width)
    motion = metadata.get("motion_planning", {})
    sync = metadata.get("embodiment_sync", {})
    order = sync.get("action_order", motion.get("native_action_layout"))
    env = metadata.get("env_info", {})
    kwargs = env.get("env_kwargs", {})
    if order is None:
        raise ValueError("Recording lacks native joint/controller identities; provide --contract from its collection")
    result = dict(action_order=order, env_id=env.get("env_id"),
                  robot_uids=kwargs.get("robot_uids", motion.get("robot_uids")),
                  control_mode=kwargs.get("control_mode", "pd_joint_pos"),
                  control_freq=sync.get("control_freq", motion.get("control_freq")))
    # Legacy profiles describe the model actually used, rather than the model
    # installed on the converting machine. Preserve them without interpreting.
    for field in ("task_profile", "robot_profile"):
        values = [ep.get("dual_sync", {}).get(field) for ep in metadata.get("episodes", [])]
        present = [value for value in values if value is not None]
        if present:
            if any(value != present[0] for value in present):
                raise ValueError(f"Recording contains inconsistent {field}")
            result[field] = present[0]
    return _validate_contract(result, width)


def _actions(group: h5py.Group, key: str, order: list[dict[str, Any]]) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Group):
        values = [node[item["key"]][()] for item in order]
        result = np.concatenate(values, axis=-1)
    else:
        result = node[()]
    result = np.asarray(result, np.float32)
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{key} must contain nonempty finite (T, A) values")
    return result


def _rgb_frames(group: h5py.Group, count: int, render_camera: str) -> dict[str, np.ndarray]:
    result = {}
    if "obs/sensor_data" in group:
        aliases = {"panda_wristcam_0_hand_camera": "wristcam0", "panda_wristcam_1_hand_camera": "wristcam1"}
        for name, sensor in group["obs/sensor_data"].items():
            if "rgb" in sensor:
                result[aliases.get(name, name)] = sensor["rgb"][()]
    elif "rgb" in group:
        result[render_camera] = group["rgb"][()]
    if not result:
        raise ValueError("Native visual export needs recorded RGB; render during collection")
    for name, frames in result.items():
        if frames.ndim != 4 or frames.shape[0] != count or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"Camera {name} must contain uint8 (T+1, H, W, 3) RGB")
    return result


def load_episode(group: h5py.Group, metadata: dict[str, Any], contract: dict[str, Any],
                 label_key: str | None = None, render_camera: str = "render") -> dict[str, Any]:
    """Read T public observation/action pairs, excluding the terminal observation."""
    order = contract["action_order"]
    executed = _actions(group, "actions", order)
    key = label_key or ("actions_clean" if "actions_clean" in group else "actions")
    labels = _actions(group, key, order)
    if labels.shape != executed.shape:
        raise ValueError("Clean labels and executed actions must have the same shape")
    count = len(executed) + 1
    qpos = []
    if "obs/agent" in group:
        for item in order:
            values = np.asarray(group[f"obs/agent/{item['key']}/qpos"][()], np.float32)
            qpos.append(values)
    else:
        if "obs" not in group or isinstance(group["obs"], h5py.Group):
            raise ValueError("No public qpos observation mapping is recorded")
        flat = np.asarray(group["obs"][()], np.float32)
        layout = metadata.get("imsim", {}).get("layout", {})
        slices = layout.get("obs_slices", {})
        for item in order:
            path = f"agent/{item['key']}/qpos"
            if path not in slices:
                raise ValueError(f"Certified layout lacks public qpos slice {path}; full hidden state is never exported")
            start, end = slices[path]
            if not 0 <= start < end <= flat.shape[-1]:
                raise ValueError(f"Invalid public qpos slice {path}")
            qpos.append(flat[:, start:end])
    for item, values in zip(order, qpos, strict=True):
        if values.shape != (count, len(item["state_joint_names"])) or not np.isfinite(values).all():
            raise ValueError(f"{item['key']}: expected finite (T+1, named public qpos width)")
    cameras = _rgb_frames(group, count, render_camera)
    return dict(states=np.concatenate(qpos, axis=-1)[:-1], labels=labels,
                executed=executed, label_key=key, rgb={k: v[:-1] for k, v in cameras.items()})


def export_native(input_path: str | Path, output_dir: str | Path, name: str,
                  episodes: Sequence[int] | None = None, fps: float | None = None,
                  contract: dict[str, Any] | None = None, label_key: str | None = None,
                  render_camera: str = "render", video_backend: str = "pyav") -> dict[str, Any]:
    """Export selected episodes, retaining source judgments as audit metadata."""
    from .lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import write_info

    source = Path(input_path)
    metadata = json.loads(source.with_suffix(".json").read_text())
    target = Path(output_dir) / name
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    with h5py.File(source, "r") as h5:
        names = sorted((key for key in h5 if key.startswith("traj_")), key=lambda k: int(k[5:]))
        if episodes is not None:
            selected = set(episodes)
            if len(selected) != len(episodes) or any(type(i) is not int or i < 0 for i in episodes):
                raise ValueError("Episode IDs must be unique nonnegative integers")
            names = [key for key in names if int(key[5:]) in selected]
            if len(names) != len(selected):
                raise ValueError("A selected source episode does not exist")
        if not names:
            raise ValueError("No trajectory episodes were selected")
        node = h5[names[0]]["actions"]
        width = sum(value.shape[-1] for value in node.values()) if isinstance(node, h5py.Group) else node.shape[-1]
        native = recorded_contract(metadata, width, contract)
        imsim = metadata.get("imsim", {})
        rate = native.get("control_freq") or imsim.get("control_freq")
        if rate is None:
            rate = fps
        if rate is None or not np.isfinite(rate) or rate <= 0:
            raise ValueError("A positive recorded control frequency or explicit --fps is required")
        if fps is not None and not np.isclose(fps, rate):
            raise ValueError("Export FPS must equal the recorded control frequency")
        native["control_freq"] = rate
        first = load_episode(h5[names[0]], metadata, native, label_key, render_camera)
        state_names = [f"{item['key']}/{joint}" for item in native["action_order"] for joint in item["state_joint_names"]]
        action_names = [f"{item['key']}/{joint}" for item in native["action_order"]
                        for joint in [*item["arm_joint_names"], "gripper"]]
        features = {
            STATE_KEY: dict(dtype="float32", shape=(len(state_names),), names=state_names),
            ACTION_KEY: dict(dtype="float32", shape=(len(action_names),), names=action_names),
        }
        for camera, frames in first["rgb"].items():
            features[f"observation.images.{camera}"] = dict(dtype="video", shape=(3, *frames.shape[1:3]),
                                                           names=["channels", "height", "width"])
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset.create(repo_id=name, fps=rate, root=target, features=features,
                                       use_videos=True, video_backend=video_backend,
                                       vcodec="h264", encoder_threads=1)
        manifest = []
        try:
            dataset.meta.info["native_data_contract"] = native
            dataset.meta.info["terminal_observation"] = "dropped"
            dataset.meta.info["source_imsim"] = imsim
            source_records = {int(ep.get("episode_id", i)): ep for i, ep in enumerate(metadata.get("episodes", []))}
            for index, key in enumerate(names):
                data = first if index == 0 else load_episode(h5[key], metadata, native, label_key, render_camera)
                if data["labels"].shape[1] != len(action_names) or data["states"].shape[1] != len(state_names):
                    raise ValueError("Episode dimensions differ from the recorded native layout")
                if data["rgb"].keys() != first["rgb"].keys():
                    raise ValueError("Selected episodes have different camera sets")
                for t in range(len(data["labels"])):
                    frame = {"task": native.get("env_id") or name, STATE_KEY: data["states"][t], ACTION_KEY: data["labels"][t]}
                    for camera, frames in data["rgb"].items():
                        frame[f"observation.images.{camera}"] = frames[t].transpose(2, 0, 1)
                    dataset.add_frame(frame)
                dataset.save_episode()
                source_id = int(key[5:])
                manifest.append(dict(episode_id=index, source_trajectory=key, source_episode=source_records.get(source_id),
                                     frames=len(data["labels"]), label_key=data["label_key"],
                                     executed_key="actions", terminal_observation="dropped"))
            # Conversion does not choose a scientific train/evaluation split.
            dataset.meta.info["splits"] = {"train": []}
            write_info(dataset.meta.info, dataset.meta.root)
        finally:
            dataset.finalize()
    digest = sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result = dict(source_h5=source.name, source_sha256=digest.hexdigest(),
                  dataset=name, episodes=manifest, fps=rate, native_data_contract=native,
                  source_imsim=imsim, certification_preserved=bool(imsim),
                  certification_policy="copy source facts without rejudging")
    (target / "native_episode_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _sanity_check() -> None:
    order = [dict(key=side, size=2, arm_joint_names=["joint"], state_joint_names=["joint", "finger"])
             for side in ("left", "right")]
    contract = dict(action_order=order, control_mode="pd_joint_pos", control_freq=20)
    metadata = dict(native_data_contract=contract, imsim={"layout": {"obs_slices": {
        "agent/left/qpos": [0, 2], "agent/right/qpos": [2, 4], "extra/object": [4, 5]}}})
    with h5py.File("native-export-sanity", "w", driver="core", backing_store=False) as h5:
        group = h5.create_group("traj_0")
        group["actions"] = np.ones((2, 4), np.float32)
        group["actions_clean"] = np.zeros((2, 4), np.float32)
        group["obs"] = np.array([[0, 1, 2, 3, 999], [4, 5, 6, 7, 999], [8, 9, 10, 11, 999]], np.float32)
        group["rgb"] = np.zeros((3, 8, 8, 3), np.uint8)
        data = load_episode(group, metadata, recorded_contract(metadata, 4))
    assert data["states"].shape == (2, 4) and data["labels"].shape == (2, 4)
    assert data["rgb"]["render"].shape == (2, 8, 8, 3)
    assert not data["labels"].any() and data["executed"].all()
    assert not (data["states"] == 999).any()
    print("native export sanity:", data["states"].shape, data["labels"].shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--contract", type=Path, help="Recorded native contract supplement for older archives")
    parser.add_argument("--label-key", choices=("actions", "actions_clean"))
    parser.add_argument("--render-camera", default="render")
    args = parser.parse_args()
    if args.self_test:
        _sanity_check()
        return
    if args.input is None or args.output_dir is None or args.name is None:
        parser.error("--input, --output-dir and --name are required for conversion")
    supplement = json.loads(args.contract.read_text()) if args.contract else None
    result = export_native(args.input, args.output_dir, args.name, args.episodes, args.fps,
                           supplement, args.label_key, args.render_camera)
    print(json.dumps({"dataset": result["dataset"], "episodes": len(result["episodes"]), "fps": result["fps"]}))


if __name__ == "__main__":
    main()
