"""Convert a completed dual-arm observation export to a local LeRobot dataset.

Only genuine pre-action rows (o_t, q_t, a_t), t=0..T-1, become training records.
The terminal frame is retained separately, never assigned a copied last action.
RGB uses lossless LeRobot image storage. --self-test performs a real CPU-only
LeRobot write/read roundtrip, including horizon padding and pixel comparisons.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np

from mani_skill.examples.motionplanning.dual.export_observations import (
    ACTION_NAMES, CAMERA, CONTROL_HZ, IMAGE_SHAPE, SCHEMA, STATE_NAMES,
    directory_identity, file_sha256, load_export, require, save_json,
)

IMAGE_KEY = f"observation.images.{CAMERA}"
STATE_KEY = "observation.qpos_gripper_states"
ACTION_KEY = "action.qpos_gripper_actions"
PADDING_KEY = ACTION_KEY + "_is_pad"
DATASET_SCHEMA = "tig-dual-arm-lerobot-v1"


def features() -> dict[str, Any]:
    """Existing TIG feature names, physical state18 and commanded action16."""
    return {IMAGE_KEY: {"dtype": "image", "shape": (3, 224, 224),
                       "names": ["channels", "height", "width"]},
            STATE_KEY: {"dtype": "float32", "shape": (18,), "names": STATE_NAMES},
            ACTION_KEY: {"dtype": "float32", "shape": (16,), "names": ACTION_NAMES}}


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_dataset(dataset_dir: Path, arrays: dict[str, np.ndarray], *,
                     repo_id: str, horizon: int = 16) -> dict[str, Any]:
    """Read real LeRobot records and chunks; compare first/middle/last labels.

    Padding comes from the vendored reader's episode boundaries. Clamped values
    are legal only where action_is_pad is True; they are not extra labels.
    """
    import torch
    from torch.utils.data import DataLoader
    from .lerobot_dataset import LeRobotDataset

    require(type(horizon) is int and horizon > 0, "A positive action horizon is required")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_dir, episodes=[0],
        delta_timestamps={ACTION_KEY: [i / CONTROL_HZ for i in range(horizon)]})
    count = len(arrays["actions"])
    require(len(dataset) == count and dataset.fps == CONTROL_HZ, "Dataset length/frequency differs")
    require(dataset.meta.info["splits"] == {"train": [0], "test": []},
            "Single-episode debug split must use all episode0 for training")
    require(dataset.meta.info["total_episodes"] == 1, "Expected exactly one episode")
    checks = []
    for index in sorted({0, count // 2, count - 1}):
        item = dataset[index]
        image = item[IMAGE_KEY]
        require(tuple(image.shape) == (3, 224, 224) and bool(torch.isfinite(image).all()),
                "Dataloader RGB shape/values differ")
        expected_image = arrays["rgb"][index].transpose(2, 0, 1)
        decoded = np.rint(image.numpy() * 255).astype(np.uint8)
        require(np.array_equal(decoded, expected_image), "Lossless RGB row/source frame mismatch")
        require(np.array_equal(item[STATE_KEY].numpy(), arrays["qpos"][index]),
                "Dataloader physical state differs from pre-action q_t")
        query = np.arange(index, index + horizon)
        expected_padding = query >= count
        expected_actions = arrays["actions"][np.minimum(query, count - 1)]
        require(tuple(item[ACTION_KEY].shape) == (horizon, 16)
                and np.array_equal(item[ACTION_KEY].numpy(), expected_actions),
                "Action chunk indexing differs from original actions")
        require(np.array_equal(item[PADDING_KEY].numpy(), expected_padding),
                "Reader failed to mark out-of-episode chunk padding")
        require(abs(float(item["timestamp"].item()) - index / CONTROL_HZ) < 1e-6,
                "LeRobot row timestamps differ from20Hz")
        checks.append({"dataset_row": index, "source_frame": index, "source_action": index,
                       "rgb_exact": True, "qpos_exact": True, "action_chunk_exact": True,
                       "timestamp_s": float(arrays["timestamps_s"][index]),
                       "padding": expected_padding.tolist(),
                       "rgb_sha256": array_digest(expected_image),
                       "qpos_sha256": array_digest(arrays["qpos"][index]),
                       "action_sha256": array_digest(arrays["actions"][index])})
    batch = next(iter(DataLoader(dataset, batch_size=min(8, count), shuffle=False, num_workers=0)))
    require(tuple(batch[IMAGE_KEY].shape[1:]) == (3, 224, 224)
            and tuple(batch[STATE_KEY].shape[1:]) == (18,)
            and tuple(batch[ACTION_KEY].shape[1:]) == (horizon, 16)
            and tuple(batch[PADDING_KEY].shape[1:]) == (horizon,), "Real batch dimensions differ")
    with np.load(dataset_dir / "meta/terminal-observation.npz") as terminal:
        require(set(terminal.files) == {"rgb", "qpos", "timestamp_s", "source_frame"}
                and np.array_equal(terminal["rgb"], arrays["rgb"][-1])
                and np.array_equal(terminal["qpos"], arrays["qpos"][-1])
                and int(terminal["source_frame"]) == count
                and float(terminal["timestamp_s"]) == float(arrays["timestamps_s"][-1]),
                "Terminal observation must be preserved independently without an action")
    return {"status": "passed", "frames": count, "fps": dataset.fps, "horizon": horizon,
            "split": dataset.meta.info["splits"], "checks": checks,
            "batch_shapes": {key: list(batch[key].shape) for key in
                (IMAGE_KEY, STATE_KEY, ACTION_KEY, PADDING_KEY)},
            "terminal_frame_retained_separately": True, "synthetic_terminal_action": False,
            "normalization_statistics_scope": f"all{count} genuine training rows; no test or terminal action"}


def convert(input_dir: Path, output_dir: Path, *, repo_id: str = "local/tig-pill-sync-seed401"
            ) -> dict[str, Any]:
    """Write a new local dataset and an explicit row-level provenance mapping."""
    from .lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import write_info

    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    require(not output_dir.exists(), "Output already exists; choose a fresh dataset directory")
    require(input_dir != output_dir and input_dir not in output_dir.parents,
            "Converted dataset must be outside the observation export")
    before = directory_identity(input_dir)
    manifest, arrays = load_export(input_dir)
    original = Path(manifest["source_dir"]).resolve()
    require(output_dir != original and original not in output_dir.parents,
            "Converted dataset must be outside the immutable original recording")
    count = len(arrays["actions"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = None
    report: dict[str, Any] = {"schema": DATASET_SCHEMA, "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "repo_id": repo_id,
        "source_export_dir": str(input_dir), "source_export_manifest_sha256": before["manifest.json"]["sha256"],
        "source_observations_sha256": before["observations.h5"]["sha256"],
        "source_recording_dir": manifest["source_dir"],
        "source_recording_h5_sha256": manifest["source_h5_sha256"],
        "source_report_sha256": manifest["source_report_sha256"], "task": manifest["task"],
        "seed": manifest["seed"], "control_mode": manifest["control_mode"], "fps": CONTROL_HZ,
        "camera": CAMERA, "image_storage": "lossless embedded LeRobot image",
        "state_dim": 18, "action_dim": 16, "training_rows": count, "source_state_frames": count + 1,
        "splits": {"train": [0], "test": []},
        "alignment": "row t contains source o_t,q_t,a_t; timestamp=t/20; no a_T",
        "terminal_observation": "meta/terminal-observation.npz",
        "row_mapping": "meta/source-frames.jsonl",
        "chunk_padding": "vendored LeRobotDataset action.qpos_gripper_actions_is_pad; offsets=i/20",
        "converter_sha256": file_sha256(Path(__file__)), "input_inventory_before": before}
    try:
        dataset = LeRobotDataset.create(repo_id=repo_id, root=output_dir, fps=CONTROL_HZ,
            features=features(), robot_type="dual_panda", use_videos=False)
        for index in range(count):
            # add_frame derives frame_index/fps; the installed validator does
            # not accept an explicit timestamp among user-provided features.
            dataset.add_frame({"task": manifest["task"],
                IMAGE_KEY: arrays["rgb"][index].transpose(2, 0, 1).copy(),
                STATE_KEY: arrays["qpos"][index].copy(), ACTION_KEY: arrays["actions"][index].copy()})
        dataset.save_episode()
        dataset.meta.info["splits"] = {"train": [0], "test": []}
        write_info(dataset.meta.info, dataset.root)
        dataset.finalize()
        dataset = None
        meta = output_dir / "meta"
        np.savez_compressed(meta / "terminal-observation.npz", rgb=arrays["rgb"][-1],
            qpos=arrays["qpos"][-1], timestamp_s=arrays["timestamps_s"][-1], source_frame=np.int64(count))
        with (meta / "source-frames.jsonl").open("w") as stream:
            for index in range(count):
                stream.write(json.dumps({"episode_index": 0, "dataset_row": index,
                    "source_frame": index, "source_action": index, "timestamp_s": index / CONTROL_HZ,
                    "source_observations_sha256": manifest["observations_h5_sha256"]}, sort_keys=True) + "\n")
        # Explicit train/test selection consumed by TIG's existing dataset-file adapter.
        save_json(meta / "dataset-selection.json", [{"repo_id": repo_id, "root": ".",
                                                   "train": "[0]", "test": "[]"}])
        report["reader_validation"] = validate_dataset(output_dir, arrays, repo_id=repo_id)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if dataset is not None:
            dataset.finalize()
        after = directory_identity(input_dir)
        report["input_export_unchanged"] = before == after
        report["input_inventory_after"] = after
        if before != after:
            report["status"] = "failed"
        if output_dir.exists():
            (output_dir / "meta").mkdir(exist_ok=True)
            save_json(output_dir / "meta/dual-arm-bridge.json", report)
        require(before == after, "Immutable observation export changed during conversion")
    return report


def self_test() -> dict[str, Any]:
    """Real CPU image/Parquet/reader/dataloader roundtrip on three source actions."""
    with tempfile.TemporaryDirectory(prefix="tig-dual-arm-lerobot-test-") as temporary:
        root = Path(temporary)
        source = root / "export"
        source.mkdir()
        count = 3
        rgb = np.zeros((count + 1,) + IMAGE_SHAPE, np.uint8)
        rgb[..., 0] = np.arange(count + 1, dtype=np.uint8)[:, None, None] * 40
        rgb[..., 1] = np.arange(224, dtype=np.uint8)[None, :, None]
        rgb[..., 2] = np.arange(224, dtype=np.uint8)[None, None, :]
        qpos = np.arange((count + 1) * 18, dtype=np.float32).reshape(count + 1, 18) / 100
        actions = np.arange(count * 16, dtype=np.float32).reshape(count, 16) / 100
        with h5py.File(source / "observations.h5", "w") as handle:
            handle.create_dataset("rgb", data=rgb)
            handle.create_dataset("qpos", data=qpos)
            handle.create_dataset("actions", data=actions)
            handle.create_dataset("timestamps_s", data=np.arange(count + 1, dtype=np.float64) / CONTROL_HZ)
        save_json(source / "manifest.json", {"schema": SCHEMA, "status": "passed", "control_hz": CONTROL_HZ,
            "camera": CAMERA, "control_mode": "pd_joint_pos", "action_steps": count,
            "state_frames": count + 1, "observations_h5_sha256": file_sha256(source / "observations.h5"),
            "source_dir": str(root / "synthetic-original"), "source_h5_sha256": "0" * 64,
            "source_report_sha256": "1" * 64, "task": "synthetic_dual_panda", "seed": 401})
        report = convert(source, root / "dataset", repo_id="local/dual-arm-synthetic-self-test")
        assert report["status"] == "passed" and report["input_export_unchanged"]
        final = report["reader_validation"]["checks"][-1]
        assert final["source_frame"] == 2 and final["padding"] == [False] + [True] * 15
        try:
            convert(source, root / "dataset")
        except ValueError:
            pass
        else:
            raise AssertionError("Converter overwrote an existing dataset")
        try:
            convert(source, root / "synthetic-original/derived")
        except ValueError:
            pass
        else:
            raise AssertionError("Converter accepted writes within the immutable original")
        return {"status": "passed", "validation": report["reader_validation"],
                "existing_output_refused": True, "original_recording_output_refused": True,
                "actual_native_trajectory": "not_run"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-id", default="local/tig-pill-sync-seed401")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        if not args.input_dir or not args.output_dir:
            parser.error("--input-dir and --output-dir are required")
        report = convert(args.input_dir, args.output_dir, repo_id=args.repo_id)
        print(json.dumps({key: report[key] for key in
                          ("status", "training_rows", "fps", "input_export_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
