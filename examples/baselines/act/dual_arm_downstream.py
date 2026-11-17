"""Run the single-episode dual-Panda data -> ACT -> closed-loop engineering check.

This is an interface smoke experiment, not a benchmark protocol. Training uses
all 171 real action rows (22 updates, batch eight, drop_last=False). Terminal
frame 172 is excluded from labels/statistics. The policy architecture is the
existing ACTSingleTaskAgent with its default ResNet18 and transformer settings.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time
import traceback
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dual_arm_adapter import (ACTION_KEY, CAMERA_KEY, FPS, HORIZON, STATE_KEY,
    make_normalizer, masked_l1, numpy_array, prepare_observation, sha256_file,
    split_action, train_statistics)


def save_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Path): return str(item)
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        if isinstance(item, torch.Tensor): return item.detach().cpu().tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=convert) + "\n")


def guard_output_directory(output_dir: Path, export_dir: Path, *,
                           dataset_dirs: tuple[Path, ...] = (),
                           manifest: dict[str, Any] | None = None) -> Path:
    """Reject writes beneath any immutable input, including symlink aliases."""
    export_dir = export_dir.resolve()
    if manifest is None:
        manifest = json.loads((export_dir / "manifest.json").read_text())
    protected = [("original recording", Path(manifest["source_dir"]).resolve()),
                 ("observation export", export_dir)]
    protected.extend(("LeRobot dataset", directory.resolve()) for directory in dataset_dirs)
    output_dir = output_dir.resolve()
    for label, directory in protected:
        if output_dir == directory or directory in output_dir.parents:
            raise ValueError(f"Output must be outside the immutable {label}: {directory}")
    return output_dir


def authenticated_source_metadata(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Check frozen source identities before reading evaluation reset inputs."""
    source_dir = Path(manifest["source_dir"]).resolve()
    for filename, identity_key in (("trajectory.json", "source_metadata_sha256"),
                                   ("trajectory.h5", "source_h5_sha256")):
        if sha256_file(source_dir / filename) != manifest[identity_key]:
            raise ValueError(f"Original recording {filename} differs from the observation export")
    return source_dir, json.loads((source_dir / "trajectory.json").read_text())


def source_code_identity() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name("dual_arm_adapter.py"),
             Path(__file__).with_name("train_act_single.py"),
             Path(__file__).parent / "act/detr_video/detr_vae.py"]
    return {str(path.resolve()): sha256_file(path) for path in paths}


class BridgeDataset(Dataset):
    """Real vendored LeRobot reader with explicit 20 Hz horizons and tail mask."""

    def __init__(self, dataset_dir: Path, export_dir: Path, *, stats: dict[str, Any] | None = None):
        from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset
        from mani_skill.examples.motionplanning.dual.export_observations import load_export
        self.dataset_dir, self.export_dir = dataset_dir.resolve(), export_dir.resolve()
        self.source_manifest, self.export_arrays = load_export(self.export_dir)
        conversion_manifest = json.loads((self.dataset_dir / "meta/dual-arm-bridge.json").read_text())
        if conversion_manifest.get("status") != "passed" or conversion_manifest.get("source_observations_sha256") != self.source_manifest["observations_h5_sha256"]:
            raise ValueError("Converted dataset provenance differs from the authenticated observation export")
        self.repo_id = conversion_manifest["repo_id"]
        self.raw = LeRobotDataset(repo_id=self.repo_id, root=self.dataset_dir,
            episodes=[0], delta_timestamps={ACTION_KEY: [index / FPS for index in range(HORIZON)]},
            tolerance_s=1e-4, video_backend="torchcodec")
        if self.raw.fps != FPS or len(self.raw) != 171:
            raise ValueError("First bridge requires exactly 171 train rows at 20 Hz")
        if self.raw.meta.info["splits"] != {"train": [0], "test": []}:
            raise ValueError("Single-episode debug split must contain only train episode0")
        self.states = torch.stack([self.raw.hf_dataset[index][STATE_KEY] for index in range(len(self.raw))]).float()
        self.actions = torch.stack([self.raw.hf_dataset[index][ACTION_KEY] for index in range(len(self.raw))]).float()
        if self.states.shape != (171, 18) or self.actions.shape != (171, 16):
            raise ValueError("LeRobot physical qpos/action dimensions differ from source")
        if not np.array_equal(self.states.numpy(), self.export_arrays["qpos"][:-1]) or not np.array_equal(
                self.actions.numpy(), self.export_arrays["actions"]):
            raise ValueError("LeRobot state/action rows do not exactly match the exported source")
        self.stats = train_statistics(self.states, self.actions) if stats is None else stats
        self.normalizer = make_normalizer(self.stats, self.repo_id)
        self.audit = self.audit_alignment()

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.raw[index]
        obs = prepare_observation(sample[CAMERA_KEY], sample[STATE_KEY], self.normalizer)
        actions = sample[ACTION_KEY].float()
        is_pad = sample[ACTION_KEY + "_is_pad"].bool()
        if actions.shape != (HORIZON, 16) or is_pad.shape != (HORIZON,):
            raise ValueError("LeRobot action horizon or tail mask differs")
        return {"robot_obs": obs,
                "robot_actions": self.normalizer.normalize_action(actions, 0, method="bounds_q99"),
                "is_pad": is_pad, "sample_index": index}

    def audit_alignment(self) -> dict[str, Any]:
        rows = []
        for index in (0, 85, 170):
            sample = self.raw[index]
            expected = prepare_observation(self.export_arrays["rgb"][index],
                                           self.export_arrays["qpos"][index], self.normalizer)
            actual = prepare_observation(sample[CAMERA_KEY], sample[STATE_KEY], self.normalizer)
            if not all(torch.equal(actual[key], expected[key]) for key in actual):
                raise ValueError(f"Source/LeRobot observation mismatch at frame {index}")
            indices = np.minimum(np.arange(index, index + HORIZON), 170)
            padding = torch.arange(index, index + HORIZON) >= 171
            if not np.array_equal(sample[ACTION_KEY].numpy(), self.export_arrays["actions"][indices]) or not torch.equal(
                    sample[ACTION_KEY + "_is_pad"], padding):
                raise ValueError(f"Action chunk or terminal padding mismatch at frame {index}")
            timestamp = float(sample["timestamp"].reshape(-1)[0])
            if not np.isclose(timestamp, index / FPS, rtol=0, atol=5e-7):
                raise ValueError("Sample timestamp differs from 20 Hz source")
            rows.append({"index": index, "timestamp_s": timestamp,
                         "rgb_exact": True, "qpos_exact": True, "action_chunk_exact": True,
                         "valid_actions": int((~padding).sum()), "padding_mask_exact": True})
        return {"status": "passed", "camera": "zed2i", "fps": FPS, "train_episodes": [0],
                "test_episodes": [], "real_action_rows": len(self), "terminal_label_rows": 0,
                "train_statistics_rows": 171, "normalization": "shared ActionNormalizer bounds_q99",
                "augmentation": False, "source_rows": rows,
                "state_layout": "physical left qpos[9], physical right qpos[9]",
                "action_layout": "left arm[7], left gripper[1], right arm[7], right gripper[1]"}


def _fixed_batch(dataset: BridgeDataset) -> dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate
    return default_collate([dataset[index] for index in (0, 85, 170)])


def _training_args(output_dir: Path) -> Any:
    from .train_act_single import TrainingArgs
    return TrainingArgs(seed=1, cameras=["zed2i"], include_depth=False, single_arm=False,
        image_size=[224, 224], state_dim=18, action_dim=16, obs_encoder_type="resnet18",
        obs_horizon=1, pred_horizon=16, total_epochs=1, batch_size=8, num_workers=0,
        output_dir=str(output_dir))


def _build_agent(args: Any, device: torch.device) -> Any:
    from .train_act_single import ACTSingleTaskAgent
    if args.obs_encoder_type != "resnet18":
        raise ValueError("The bridge must retain the planned ResNet18 observation encoder")
    # The official encoder loads these pretrained weights. Requiring the cache
    # avoids an unrecorded network download or a silent random-backbone fallback.
    weight_file = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    if not weight_file.is_file():
        raise FileNotFoundError(f"Pretrained ResNet18 cache is required: {weight_file}")
    return ACTSingleTaskAgent(args, device).to(device)


def _predict(agent: Any, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    agent.eval()
    with torch.no_grad():
        result = agent.get_action(batch["robot_obs"]["rgb"].to(device), batch["robot_obs"]["state"].to(device))
    if result.shape != (3, HORIZON, 16) or not torch.isfinite(result).all():
        raise ValueError("Invalid fixed-batch ACT inference")
    return result.cpu()


def _model_digest(agent: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(agent.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check(args: argparse.Namespace) -> None:
    args.output_dir = guard_output_directory(args.output_dir, args.export_dir,
                                            dataset_dirs=(args.dataset_dir,))
    dataset = BridgeDataset(args.dataset_dir, args.export_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "dataset-check.json", {**dataset.audit,
        "source_manifest": dataset.source_manifest, "train_statistics": dataset.stats,
        "source_code": source_code_identity()})
    print(json.dumps(dataset.audit, ensure_ascii=False))


def train(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    args.output_dir = guard_output_directory(args.output_dir, args.export_dir,
                                            dataset_dirs=(args.dataset_dir,))
    if (args.output_dir / "checkpoint_epoch_0001.pt").exists():
        raise FileExistsError("Training will not overwrite an existing checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = _training_args(args.output_dir)
    random.seed(training.seed); np.random.seed(training.seed); torch.manual_seed(training.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    dataset = BridgeDataset(args.dataset_dir, args.export_dir)
    save_json(args.output_dir / "dataset-check.json", dataset.audit)
    generator = torch.Generator().manual_seed(training.seed)
    loader = DataLoader(dataset, batch_size=training.batch_size, shuffle=True, num_workers=0,
                        drop_last=False, generator=generator)
    agent = _build_agent(training, device)
    backbone = agent.obs_encoder.encoder.backbone if hasattr(agent.obs_encoder, "encoder") else None
    if backbone is None:
        # ObservationEncoder exposes backbone inside the selected encoder.
        backbone = getattr(agent.obs_encoder, "backbone", None)
    backbone_params = list(backbone.parameters()) if backbone is not None else []
    backbone_ids = {id(parameter) for parameter in backbone_params}
    other_params = [parameter for parameter in agent.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": training.lr_backbone},
        {"params": other_params, "lr": training.lr}], weight_decay=training.weight_decay)
    before = {key: value.detach().cpu().clone() for key, value in agent.named_parameters() if value.requires_grad}
    model_before = _model_digest(agent)
    agent.train()
    steps: list[dict[str, Any]] = []
    for index, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context:
            loss = agent.compute_loss(batch)
        if not torch.isfinite(loss):
            raise ValueError("ACT training loss is nonfinite")
        loss.backward()
        gradients = [parameter.grad for parameter in agent.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
            raise ValueError("ACT training gradients are missing or nonfinite")
        norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), training.grad_clip)
        optimizer.step()
        row = {"optimizer_step": index, "batch_size": int(batch["robot_actions"].shape[0]),
               "sample_indices": batch["sample_index"].tolist(), "loss": float(loss.detach()),
               "gradient_norm_before_clip": float(norm), "finite_gradients": True,
               "valid_action_tokens": int((~batch["is_pad"]).sum())}
        steps.append(row); print(json.dumps(row), flush=True)
    seen = [value for row in steps for value in row["sample_indices"]]
    if len(steps) != 22 or sorted(seen) != list(range(171)):
        raise RuntimeError("One epoch must update on all 171 samples exactly once")
    changed = [key for key, value in agent.named_parameters() if key in before and not torch.equal(before[key], value.detach().cpu())]
    if not changed:
        raise RuntimeError("Optimizer did not change any trainable parameter")
    batch = _fixed_batch(dataset)
    prediction = _predict(agent, batch, device)
    checkpoint = args.output_dir / "checkpoint_epoch_0001.pt"
    torch.save({"schema": "tig-dual-arm-act-bridge-v1", "epoch": 1, "global_step": len(steps),
        "agent_state_dict": agent.state_dict(), "optimizer": optimizer.state_dict(),
        "args": asdict(training), "dataset_stats": [{"dataset_idx": 0, "repo_id": dataset.repo_id,
            "state_key": STATE_KEY, "action_key": ACTION_KEY, "single_arm": False, "stats": dataset.stats}],
        "bridge_normalizer": {"method": "bounds_q99", "stats": dataset.stats, "repo_id": dataset.repo_id,
            "train_episodes": [0], "row_count": 171, "excludes_terminal": True},
        "dataset_info": [{"repo_id": dataset.repo_id, "root": str(dataset.dataset_dir)}],
        "export_dir": str(dataset.export_dir), "source_manifest": dataset.source_manifest,
        "source_code": source_code_identity(), "fixed_batch_indices": [0, 85, 170],
        "fixed_batch_prediction": prediction, "training_steps": steps}, checkpoint)
    report = {"status": "passed", "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "device": str(device), "args": asdict(training), "epochs": 1, "optimizer_steps": len(steps),
        "samples_seen": len(seen), "drop_last": False, "finite_gradients": True,
        "changed_parameter_tensors": len(changed), "changed_parameter_names": changed,
        "model_state_sha256_before": model_before, "model_state_sha256_after": _model_digest(agent),
        "training_steps": steps, "source_manifest": dataset.source_manifest,
        "dataset_alignment": dataset.audit, "wall_time_s": time.perf_counter() - started,
        "pretrained_resnet18_sha256": sha256_file(Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"),
        "scope": "single-episode interface training; no held-out performance claim"}
    save_json(args.output_dir / "training-report.json", report)
    print(json.dumps({"status": "passed", "optimizer_steps": len(steps), "checkpoint": str(checkpoint)}))


def load_checkpoint(path: Path, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    from .train_act_single import TrainingArgs
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != "tig-dual-arm-act-bridge-v1":
        raise ValueError("Expected the dedicated bridge checkpoint schema")
    training = TrainingArgs(**checkpoint["args"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    agent = _build_agent(training, device)
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    agent.eval()
    return agent, training, checkpoint


def reload_check(args: argparse.Namespace) -> None:
    args.output_dir = guard_output_directory(args.output_dir, args.export_dir,
                                            dataset_dirs=(args.dataset_dir,))
    device = torch.device(args.device)
    agent, _, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = BridgeDataset(args.dataset_dir, args.export_dir, stats=checkpoint["bridge_normalizer"]["stats"])
    if dataset.source_manifest != checkpoint["source_manifest"]:
        raise ValueError("Reload dataset source identity differs from checkpoint")
    actual = _predict(agent, _fixed_batch(dataset), device)
    expected = checkpoint["fixed_batch_prediction"].cpu()
    exact = torch.equal(actual, expected)
    if not exact:
        raise ValueError(f"Strict reload inference changed: max error {float((actual-expected).abs().max())}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "reload-report.json", {"status": "passed", "strict_state_dict": True,
        "same_input_prediction_exact": exact, "prediction_shape": list(actual.shape),
        "max_abs_error": float((actual-expected).abs().max()), "batch_indices": [0, 85, 170],
        "checkpoint_sha256": sha256_file(args.checkpoint), "dataset_alignment": dataset.audit})
    print(json.dumps({"status": "passed", "strict_reload": True, "prediction_exact": exact}))


def _write_tree(group: h5py.Group, state: dict[str, Any]) -> None:
    for key, value in state.items():
        if isinstance(value, dict): _write_tree(group.create_group(key), value)
        else: group.create_dataset(key, data=numpy_array(value))


def evaluate(args: argparse.Namespace) -> None:
    """One initial restoration; every later observation comes from real physics."""
    args.output_dir = guard_output_directory(args.output_dir, args.export_dir)
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 -- register task IDs
    from mani_skill.examples.motionplanning.dual.export_observations import (
        load_export, load_initial_state, extract_policy_observation)
    from mani_skill.examples.motionplanning.dual.validate_curobo_complete_replay import restore_initial_fixed_roots
    from mani_skill.examples.motionplanning.dual.trajectory_execution import bindings
    if (args.output_dir / "rollout.h5").exists():
        raise FileExistsError("Evaluation will not overwrite an existing physical rollout")
    device = torch.device(args.device)
    agent, _, checkpoint = load_checkpoint(args.checkpoint, device)
    manifest, arrays = load_export(args.export_dir)
    if manifest != checkpoint["source_manifest"]:
        raise ValueError("Evaluation source identity differs from checkpoint")
    args.output_dir = guard_output_directory(args.output_dir, args.export_dir,
        dataset_dirs=tuple(Path(entry["root"]) for entry in checkpoint["dataset_info"]),
        manifest=manifest)
    source_dir, metadata = authenticated_source_metadata(manifest)
    normalizer = make_normalizer(checkpoint["bridge_normalizer"]["stats"], checkpoint["bridge_normalizer"]["repo_id"])
    task = metadata["env_info"]["env_id"]
    kwargs = dict(manifest["config"]["environment_kwargs_export"])
    kwargs.update(obs_mode="rgb", max_episode_steps=200)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = None
    report: dict[str, Any] = {"schema": "tig-dual-arm-act-rollout-v1", "status": "running",
        "task": task, "control_mode": "pd_joint_pos", "fps": FPS, "max_episode_steps": 200,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_manifest": manifest, "source_code": source_code_identity(),
        "policy": "receding horizon: predict16, execute first action, observe again",
        "expert_fallback": False, "state_correction_frames": [0], "steps": []}
    rgbs, qposes, states, actions, normalized_predictions, denormalized_predictions = [], [], [], [], [], []
    started = time.perf_counter()
    try:
        env = gym.make(task, **kwargs)
        env.reset(**metadata["episodes"][0]["reset_kwargs"])
        raw = env.unwrapped
        if raw.gpu_sim_enabled or raw.num_envs != 1 or not np.isclose(raw.control_timestep, .05):
            raise ValueError("Evaluation requires one CPU-physics environment at 20 Hz")
        report["initial_restore"] = restore_initial_fixed_roots(raw, load_initial_state(source_dir))
        keys = list(raw.agent.agents_dict)
        bindings(raw, tuple(keys))  # shared absolute joint-controller contract
        report["action_keys"] = keys
        report["env_kwargs"] = kwargs
        camera_parameters = {key: numpy_array(value).tolist()
                             for key, value in raw.get_sensor_params()["zed2i"].items()}
        if camera_parameters != manifest["camera_parameters"]:
            raise ValueError("Online zed2i camera parameters differ from exported observations")
        report["camera_parameters_exact"] = True
        def observe() -> dict[str, torch.Tensor]:
            rgb, qpos = extract_policy_observation(raw)
            rgbs.append(numpy_array(rgb).copy()); qposes.append(numpy_array(qpos).copy())
            states.append({category: {name: numpy_array(value).copy() for name, value in entries.items()}
                           for category, entries in raw.get_state_dict().items()})
            return prepare_observation(rgb, qpos, normalizer)
        observation = observe()
        offline = prepare_observation(arrays["rgb"][0], arrays["qpos"][0], normalizer)
        qpos_equal = torch.equal(observation["state"], offline["state"])
        rgb_error = float((observation["rgb"] - offline["rgb"]).abs().max())
        report["offline_online_initial_alignment"] = {"state_exact": qpos_equal,
            "rgb_exact": torch.equal(observation["rgb"], offline["rgb"]), "rgb_max_abs_error": rgb_error,
            "same_adapter": True, "compared_camera": "zed2i"}
        if not qpos_equal:
            raise ValueError("Offline/online initial physical qpos differs")
        success_any = False
        for step in range(200):
            with torch.no_grad():
                chunk = agent.get_action(observation["rgb"].unsqueeze(0).to(device),
                                         observation["state"].unsqueeze(0).to(device))
            if chunk.shape != (1, HORIZON, 16) or not torch.isfinite(chunk).all():
                raise ValueError("ACT predicted an invalid action chunk")
            normalized = chunk[0, 0].float()
            denormalized = normalizer.denormalize_action(normalized, 0, method="bounds_q99").cpu().numpy()
            requested = split_action(denormalized, keys)
            # Apply only the environment's documented finite Box bounds; record
            # pre/post values so clipping cannot hide model outputs.
            applied = {key: np.clip(requested[key], raw.action_space[key].low, raw.action_space[key].high).astype(np.float32)
                       for key in keys}
            _, reward, terminated, truncated, info = env.step(applied)
            normalized_predictions.append(normalized.cpu().numpy())
            denormalized_predictions.append(denormalized.copy())
            actions.append(np.concatenate([applied[key] for key in keys]))
            evaluation = raw.evaluate()
            official = info.get("success", evaluation.get("success", False))
            success = bool(numpy_array(official).reshape(-1)[0]); success_any |= success
            row = {"step": step + 1, "success": success,
                "terminated": bool(numpy_array(terminated).reshape(-1)[0]),
                "truncated": bool(numpy_array(truncated).reshape(-1)[0]),
                "reward": float(numpy_array(reward).reshape(-1)[0]),
                "clipped_action_components": int(np.count_nonzero(actions[-1] != denormalized)),
                "official_evaluation": {key: numpy_array(value).tolist() for key, value in evaluation.items()}}
            report["steps"].append(row)
            observation = observe()
            if row["terminated"] or row["truncated"]: break
        report.update(status="passed", chain_complete=True, actual_steps=len(actions),
            official_success_any=success_any, official_success_final=report["steps"][-1]["success"],
            timed_out=report["steps"][-1]["truncated"] or len(actions) >= 200,
            learning_success_claim=False,
            task_result="success" if report["steps"][-1]["success"] else "not_completed")
    except BaseException as exc:
        report.update(status="failed", chain_complete=False, actual_steps=len(actions),
                      error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["wall_time_s"] = time.perf_counter() - started
        if rgbs:
            with h5py.File(args.output_dir / "rollout.h5", "x") as handle:
                handle.create_dataset("rgb", data=np.stack(rgbs), compression="gzip")
                handle.create_dataset("qpos", data=np.stack(qposes))
                handle.create_dataset("actions", data=np.asarray(actions, dtype=np.float32).reshape(-1, 16))
                handle.create_dataset("normalized_predictions", data=np.asarray(normalized_predictions, dtype=np.float32).reshape(-1, 16))
                handle.create_dataset("denormalized_predictions", data=np.asarray(denormalized_predictions, dtype=np.float32).reshape(-1, 16))
                handle.create_dataset("timestamps_s", data=np.arange(len(rgbs), dtype=np.float64) / FPS)
                for index, state in enumerate(states): _write_tree(handle.create_group(f"env_states/{index:04d}"), state)
            report["rollout_h5_sha256"] = sha256_file(args.output_dir / "rollout.h5")
        save_json(args.output_dir / "rollout-report.json", report)
        if env is not None: env.close()
    print(json.dumps({key: report[key] for key in ("status", "chain_complete", "actual_steps", "official_success_final", "timed_out")}))


def guard_self_test() -> dict[str, Any]:
    """CPU temporary-directory checks; no model, simulator, or source-data writes."""
    with tempfile.TemporaryDirectory(prefix="tig-act-guard-test-") as temporary:
        root = Path(temporary)
        source, export, dataset = (root / name for name in ("source", "export", "dataset"))
        for directory in (source, export, dataset):
            directory.mkdir()
        metadata_bytes = b'{"env_info": {"env_id": "synthetic-only"}}\n'
        h5_bytes = b"synthetic source identity only; no simulator"
        (source / "trajectory.json").write_bytes(metadata_bytes)
        (source / "trajectory.h5").write_bytes(h5_bytes)
        manifest = {"source_dir": str(source),
            "source_metadata_sha256": sha256_file(source / "trajectory.json"),
            "source_h5_sha256": sha256_file(source / "trajectory.h5")}
        save_json(export / "manifest.json", manifest)
        legal_output = root / "new-output"
        assert guard_output_directory(legal_output, export, dataset_dirs=(dataset,)) == legal_output
        rejected = 0
        for index, directory in enumerate((source, export, dataset)):
            alias = root / f"input-alias-{index}"
            alias.symlink_to(directory, target_is_directory=True)
            for output in (directory, directory / "child", alias, alias / "child"):
                try:
                    guard_output_directory(output, export, dataset_dirs=(dataset,))
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError(f"Accepted writes beneath an immutable input: {output}")
        # Resolve protected input aliases as well as output aliases.
        for aliased_export, aliased_dataset, output in (
                (root / "input-alias-1", dataset, export / "child"),
                (export, root / "input-alias-2", dataset / "child")):
            try:
                guard_output_directory(output, aliased_export,
                                       dataset_dirs=(aliased_dataset,))
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("Accepted a protected input supplied through a symlink")
        verified_source, metadata = authenticated_source_metadata(manifest)
        assert verified_source == source and metadata["env_info"]["env_id"] == "synthetic-only"
        rejected_hashes = 0
        for filename, original in (("trajectory.json", metadata_bytes), ("trajectory.h5", h5_bytes)):
            path = source / filename
            path.write_bytes(original + b"changed")
            try:
                authenticated_source_metadata(manifest)
            except ValueError:
                rejected_hashes += 1
            else:
                raise AssertionError(f"Accepted a modified source file: {filename}")
            finally:
                path.write_bytes(original)
        assert not legal_output.exists() and rejected == 14 and rejected_hashes == 2
        return {"status": "passed", "legal_sibling_output": True,
            "protected_output_paths_rejected": rejected, "modified_source_hashes_rejected": rejected_hashes,
            "cuda_initialized": torch.cuda.is_initialized()}


def self_test() -> None:
    """CPU-only default ACT masking/adapter test; use the frozen local vision cache."""
    from .dual_arm_adapter import self_test as adapter_test
    from .train_act_single import _ACTCfg, TrainingArgs
    from .act.detr_video.detr_vae import DETRVAE, build_encoder, build_transformer
    adapter_test()
    guard_report = guard_self_test()
    torch.set_num_threads(1); torch.manual_seed(7)
    training = TrainingArgs()
    cfg = _ACTCfg(training, training.hidden_dim, HORIZON, 16, training.hidden_dim)
    model = DETRVAE(None, build_transformer(cfg), build_encoder(cfg), training.hidden_dim,
                    16, HORIZON, video_feature_dim=1).eval()
    obs, action = torch.randn(1, 256), torch.randn(1, HORIZON, 16)
    pad = torch.arange(HORIZON).unsqueeze(0) >= 1
    changed = action.clone(); changed[:, 1:] = 999
    torch.manual_seed(10); output, posterior = model(obs, action, torch.zeros(1, 1), is_pad=pad)
    torch.manual_seed(10); changed_output, changed_posterior = model(obs, changed, torch.zeros(1, 1), is_pad=pad)
    assert torch.equal(output, changed_output)
    assert all(torch.equal(a, b) for a, b in zip(posterior, changed_posterior))
    assert torch.equal(masked_l1(output, action, pad), masked_l1(changed_output, changed, pad))
    torch.manual_seed(11); unmasked, _ = model(obs, action, torch.zeros(1, 1))
    torch.manual_seed(11); false_mask, _ = model(obs, action, torch.zeros(1, 1), is_pad=torch.zeros_like(pad))
    assert torch.equal(unmasked, false_mask)
    predicted, _ = model(obs, actions=None, video_feature=torch.zeros(1, 1))
    assert predicted.shape == (1, HORIZON, 16) and torch.isfinite(predicted).all()
    # Exercise the actual baseline constructor and obs_proj wiring as well as
    # its compute_loss interface; no optimizer or simulator is launched here.
    agent = _build_agent(training, torch.device("cpu")).eval()
    batch = {"robot_obs": {"rgb": torch.zeros(1, 3, 224, 224), "state": torch.zeros(1, 18)},
             "robot_actions": action, "is_pad": pad}
    torch.manual_seed(12); baseline_loss = agent.compute_loss(batch)
    batch["robot_actions"] = changed
    torch.manual_seed(12); changed_loss = agent.compute_loss(batch)
    assert torch.equal(baseline_loss, changed_loss)
    full_output = agent.get_action(batch["robot_obs"]["rgb"], batch["robot_obs"]["state"])
    assert full_output.shape == (1, HORIZON, 16) and torch.isfinite(full_output).all()
    print(json.dumps({"status": "passed", "decoder_shape": list(predicted.shape),
        "padding_invariant_posterior_prediction_loss": True, "no_mask_compatibility": True,
        "actual_resnet18_agent_forward_and_masked_loss": True,
        "output_and_source_guards": guard_report,
        "cuda_initialized": torch.cuda.is_initialized()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--guard-self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")
    for name in ("check", "train", "reload", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--export-dir", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        if name in ("check", "train", "reload"):
            command.add_argument("--dataset-dir", type=Path, required=True)
        if name in ("reload", "evaluate"):
            command.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.guard_self_test: print(json.dumps(guard_self_test()))
    elif args.self_test: self_test()
    elif args.command: {"check": check, "train": train, "reload": reload_check, "evaluate": evaluate}[args.command](args)
    else: parser.error("Choose a stage, --self-test, or --guard-self-test")


if __name__ == "__main__":
    main()
