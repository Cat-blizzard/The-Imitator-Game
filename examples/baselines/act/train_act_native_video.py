"""Train video-conditioned ACT using recorded native robot observations/actions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler, WeightedRandomSampler
import tyro

from examples.baselines.act.train_act_single import (
    TrainingArgs as SingleTrainingArgs, build_dataloader,
)
from examples.baselines.act.train_act_imitator import TrainingArgs as ImitatorTrainingArgs
from examples.baselines.act.native_video_data import NativeVideoDataset
from examples.baselines.act.native_video_model import NativeVideoACTAgent


@dataclass
class TrainingArgs(SingleTrainingArgs):
    human_manifest: str = ""
    model_config: str = ""  # Optional overrides for the existing imitator model.
    save_freq_updates: int = 3000
    conditioning_mode: str = "video"  # "constant" replaces the human condition with zeros.
    sampling_policy: str = "uniform_window"  # Explicit "equal_task" balances multitask windows.


def model_configuration(args: TrainingArgs) -> ImitatorTrainingArgs:
    """Keep native state/action widths while using the existing imitator network."""
    overrides = (json.loads(Path(args.model_config).read_text(encoding="utf-8"))
                 if args.model_config else {})
    model_args = ImitatorTrainingArgs(**overrides)
    for name in ("state_dim", "action_dim", "pred_horizon", "obs_horizon", "hidden_dim",
                 "dim_feedforward", "nheads", "enc_layers", "dec_layers", "dropout",
                 "pre_norm", "kl_weight", "lr_backbone", "cameras", "include_depth"):
        setattr(model_args, name, getattr(args, name))
    model_args.image_size = tuple(args.image_size)
    model_args.backbone = args.obs_encoder_type
    model_args.input_mode = "video_only"
    model_args.state_type = args.state_type
    model_args.single_arm = args.single_arm
    model_args.video_backend = args.video_backend
    return model_args


def robot_task_id(configs: list[dict[str, Any]], dataset: Any) -> str:
    """Legacy single-task helper; multitask training uses per-dataset records."""
    if len(configs) != 1:
        raise ValueError("This helper requires one robot dataset/task; use recorded_contract for multitask")
    native = dataset.main_dataset.meta.info.get("native_data_contract") or {}
    if configs[0].get("env_id") and native.get("env_id") and configs[0]["env_id"] != native["env_id"]:
        raise ValueError("Configured robot task differs from its recorded native task")
    task_id = configs[0].get("env_id") or native.get("env_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("The robot dataset must identify its task to pair a human demonstration")
    return task_id


def recorded_contract(args: TrainingArgs, dataset: Any) -> dict[str, Any]:
    """Record each native dataset separately and validate common policy tensors.

    States have width S and actions width A; each camera stores CHW RGB. Task
    physics remain in that task's ``native`` record and are never substituted
    with the first dataset's environment contract.
    """
    configs = json.loads(Path(args.sub_config).read_text(encoding="utf-8"))
    config_by_repo = {config["repo_id"]: config for config in configs}
    if len(config_by_repo) != len(configs):
        raise ValueError("Native dataset repo_id entries must be unique")
    datasets = getattr(dataset, "sub_datasets", None)
    if datasets is None:
        datasets = [dataset.main_dataset]
    if not datasets:
        raise ValueError("No native robot datasets")
    records, common_signature = [], None
    for index, sub in enumerate(datasets):
        if sub.repo_id not in config_by_repo:
            raise ValueError(f"No dataset configuration for recorded repo_id {sub.repo_id!r}")
        config = config_by_repo[sub.repo_id]
        native = sub.meta.info.get("native_data_contract") or {}
        env_id = config.get("env_id") or native.get("env_id")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("The robot dataset must identify its task to pair a human demonstration")
        if config.get("env_id") and native.get("env_id") and config["env_id"] != native["env_id"]:
            raise ValueError("Configured robot task differs from its recorded native task")
        if len(datasets) > 1 and (not native.get("robot_uids") or not native.get("action_order")):
            raise ValueError("Multitask datasets require recorded native embodiment and joint ordering")
        state_feature = sub.meta.features[dataset.state_key]
        action_feature = sub.meta.features[dataset.action_key]
        divisor = 2 if args.single_arm else 1
        state_dim, action_dim = state_feature["shape"][0] // divisor, action_feature["shape"][0] // divisor
        if (state_dim, action_dim) != (args.state_dim, args.action_dim):
            raise ValueError(f"Native state/action dimensions differ for {sub.repo_id}")
        rate = sub.meta.info["fps"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"Invalid dataset fps {rate!r} in {sub.repo_id}")
        if native.get("control_freq", rate) != rate or native.get("control_mode", "pd_joint_pos") != "pd_joint_pos":
            raise ValueError("Native control frequency/mode differs from recorded ACT data")
        camera_features = [sub.meta.features[f"observation.images.{camera}"] for camera in args.cameras]
        if any(feature["shape"][0] != 3 for feature in camera_features):
            raise ValueError("Native cameras must contain CHW RGB images")
        signature = {
            "fps": rate, "state_shape": state_feature["shape"], "action_shape": action_feature["shape"],
            "state_names": state_feature.get("names"), "action_names": action_feature.get("names"),
            "camera_shapes": [feature["shape"] for feature in camera_features],
            "robot_uids": native.get("robot_uids"), "action_order": native.get("action_order"),
            "control_mode": native.get("control_mode", "pd_joint_pos"),
        }
        if common_signature is not None and signature != common_signature:
            raise ValueError("Multitask ACT requires the same embodiment, state/action ordering, camera shape and FPS")
        common_signature = signature
        train_episodes = config.get("train", sub.meta.info.get("splits", {}).get("train"))
        if train_episodes is None:
            raise ValueError(f"No training split for native dataset {sub.repo_id}")
        records.append({
            "dataset_idx": index, "repo_id": sub.repo_id, "env_id": env_id, "native": native,
            "source_image_size": list(camera_features[0]["shape"][-2:]), "fps": rate,
            "state_dim": state_dim, "action_dim": action_dim, "cameras": list(args.cameras),
            "train_episodes": train_episodes, "selected_episodes": list(sub.episodes),
            "normalizer_source": "selected_train_rows",
        })
    contract = {
        "schema_version": 2, "datasets": records, "fps": records[0]["fps"],
        "cameras": list(args.cameras), "image_size": list(args.image_size),
        "rgb": "float32_CHW_0_1", "state_dim": args.state_dim,
        "action_dim": args.action_dim, "control_mode": "pd_joint_pos",
        "robot_order": "left_then_right", "normalization_method": args.normalization_method,
        "normalization_min_range": args.normalization_min_range, "split": args.split,
    }
    if len(records) == 1:
        # Preserve the legacy single-task evaluator/checkpoint contract.
        contract.update(native=records[0]["native"], source_image_size=records[0]["source_image_size"])
    return contract


def training_sampler(dataset: NativeVideoDataset, seed: int, policy: str) -> tuple[Sampler[int], dict[str, Any]]:
    """Select windows with explicit expected task mass and a separate RNG stream.

    Equal-task sampling uses replacement and weights of shape (N,), with each
    task assigned total probability 1/K irrespective of its trajectory length.
    """
    generator = torch.Generator().manual_seed(seed)
    sub_datasets = getattr(dataset.base_dataset, "sub_datasets", None)
    lengths = ([len(sub) for sub in sub_datasets] if sub_datasets is not None else [len(dataset)])
    if (sum(lengths) != len(dataset) or set(dataset.dataset_tasks) != set(range(len(lengths)))
            or any(length < 1 for length in lengths)):
        raise ValueError("Sampler requires one nonempty concatenated segment per native dataset")
    task_lengths: dict[str, int] = {}
    for index, length in enumerate(lengths):
        task = dataset.dataset_tasks[index]
        task_lengths[task] = task_lengths.get(task, 0) + length
    if policy == "equal_task":
        weights = torch.cat([torch.full((length,), 1.0 / (len(task_lengths) * task_lengths[dataset.dataset_tasks[index]]),
                                        dtype=torch.float64) for index, length in enumerate(lengths)])
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)
        task_probabilities = {task: 1.0 / len(task_lengths) for task in task_lengths}
        dataset_probabilities = [length / (len(task_lengths) * task_lengths[dataset.dataset_tasks[index]])
                                 for index, length in enumerate(lengths)]
    elif policy == "uniform_window":
        sampler = RandomSampler(dataset, generator=generator)
        task_probabilities = {task: length / len(dataset) for task, length in task_lengths.items()}
        dataset_probabilities = [length / len(dataset) for length in lengths]
    else:
        raise ValueError("sampling_policy must be equal_task or uniform_window")
    receipt = {
        "policy": policy, "seed": seed, "samples_per_epoch": len(dataset),
        "replacement": policy == "equal_task", "task_probabilities": task_probabilities,
        "datasets": [{"dataset_idx": index, "task_id": dataset.dataset_tasks[index],
                      "windows": length, "probability": dataset_probabilities[index]}
                     for index, length in enumerate(lengths)],
    }
    return sampler, receipt


def first_backward_receipt(agent: NativeVideoACTAgent, conditioning_mode: str) -> dict[str, Any]:
    """Report actual adapter gradients; the constant-condition control has none."""
    encoder = agent.task_encoder.backbone
    adapter_gradients = [parameter.grad.detach().float().square().sum()
                         for parameter in encoder.cls_adapter.parameters() if parameter.grad is not None]
    return {
        "conditioning_mode": conditioning_mode,
        "adapter_has_gradient": bool(adapter_gradients),
        "adapter_gradient_norm": (float(torch.stack(adapter_gradients).sum().sqrt())
                                  if adapter_gradients else None),
        "frozen_backbone_has_gradient": any(parameter.grad is not None for parameter in encoder._backbone.parameters()),
    }


def initial_policy_sha256(agent: NativeVideoACTAgent) -> str:
    """Hash non-video-encoder parameter/buffer values before optimizer updates."""
    digest = hashlib.sha256()
    for name, tensor in sorted(agent.state_dict().items()):
        if name.startswith("task_encoder."):
            continue
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train(args: TrainingArgs) -> None:
    """Optimize the train split; human clips provide conditions, robot actions labels."""
    if args.conditioning_mode not in {"video", "constant"}:
        raise ValueError("conditioning_mode must be video or constant")
    if args.sampling_policy not in {"equal_task", "uniform_window"}:
        raise ValueError("sampling_policy must be equal_task or uniform_window")
    if args.split != "train" or not args.human_manifest:
        raise ValueError("Native video ACT requires --split train and --human-manifest")
    if (args.max_updates is not None and args.max_updates < 1) or args.total_epochs < 1 or args.save_freq_updates < 1:
        raise ValueError("Training and checkpoint budgets must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise FileExistsError(f"Training output already contains a run: {output}")

    _, base_dataset = build_dataloader(args)
    contract = recorded_contract(args, base_dataset)
    dataset_tasks = {record["dataset_idx"]: record["env_id"] for record in contract["datasets"]}
    dataset = NativeVideoDataset(base_dataset, args.human_manifest, role="train",
                                 image_size=tuple(args.image_size), video_backend=args.video_backend,
                                 dataset_tasks=dataset_tasks)
    sampler, sampling_receipt = training_sampler(dataset, args.seed, args.sampling_policy)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False)
    if not len(loader):
        raise ValueError("No native training samples")
    model_args = model_configuration(args)
    agent = NativeVideoACTAgent(model_args, device, conditioning_mode=args.conditioning_mode)
    initial_policy_sha = initial_policy_sha256(agent)
    backbone_params, other_params = [], []
    for name, parameter in agent.named_parameters():
        if parameter.requires_grad:
            (backbone_params if name.startswith("model.backbones.") else other_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dataset_stats = [{"dataset_idx": idx, **base_dataset.normalizer.dataset_info.get(idx, {}),
                      "stats": base_dataset.normalizer.dataset_stats[idx]}
                     for idx in sorted(base_dataset.normalizer.dataset_stats)]
    single_names = {field.name for field in fields(SingleTrainingArgs)}
    single_args = {key: value for key, value in asdict(args).items() if key in single_names}
    configuration = {"training_args": asdict(args), "model_args": asdict(model_args),
                     "data_contract": contract, "human_manifest": dataset.manifest,
                     "conditioning_mode": args.conditioning_mode, "sampling": sampling_receipt,
                     "initial_policy_sha256": initial_policy_sha,
                     "parameters": {"total": sum(p.numel() for p in agent.parameters()),
                                    "trainable": sum(p.numel() for p in agent.parameters() if p.requires_grad)},
                     "human_train_demonstrations": [item["metadata"] for item in dataset.human_demos]}
    config_text = json.dumps(configuration, indent=2, allow_nan=False)
    (output / "configuration.json").write_text(config_text + "\n", encoding="utf-8")
    config_sha = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    epoch_budget = math.ceil(args.max_updates / len(loader)) if args.max_updates else args.total_epochs
    update_budget = args.max_updates or epoch_budget * len(loader)
    print(f"Native video ACT: {len(dataset)} samples, {update_budget} updates, "
          f"{len(dataset.human_demos)} real human clips", flush=True)
    global_step, checkpoint = 0, None
    sampled_task_windows = {task: 0 for task in dataset_tasks.values()}
    sampled_index_digest = hashlib.sha256()
    started = time.monotonic()
    agent.train()
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
        for epoch in range(1, epoch_budget + 1):
            for batch in loader:
                indices = batch["native_sample_index"].detach().cpu().numpy().astype("<i8", copy=False)
                sampled_index_digest.update(indices.tobytes())
                for dataset_idx in batch["dataset_idx"].tolist():
                    sampled_task_windows[dataset_tasks[dataset_idx]] += 1
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = agent.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at update {global_step + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if global_step == 0:
                    gradient_receipt = first_backward_receipt(agent, args.conditioning_mode)
                    (output / "first_backward.json").write_text(
                        json.dumps(gradient_receipt, indent=2) + "\n", encoding="utf-8")
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                row = {"global_step": global_step, "epoch": epoch, "loss": float(loss.detach()),
                       **agent.last_loss_terms, "grad_norm": float(grad_norm),
                       "sampled_task_windows": dict(sampled_task_windows),
                       "elapsed_seconds": time.monotonic() - started}
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
                if global_step % args.log_freq == 0 or global_step == update_budget:
                    metrics.flush()
                    print(json.dumps(row), flush=True)
                if global_step % args.save_freq_updates == 0 or global_step == update_budget:
                    checkpoint = output / f"checkpoint_step_{global_step:06d}.pt"
                    torch.save({"policy_type": "native_video_act", "args": single_args,
                        "model_args": asdict(model_args), "training_args": asdict(args),
                        "agent_state_dict": agent.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "global_step": global_step, "data_contract": contract,
                        "dataset_stats": dataset_stats, "human_manifest": dataset.manifest,
                        "conditioning_mode": args.conditioning_mode, "sampling": sampling_receipt,
                        "initial_policy_sha256": initial_policy_sha,
                        "actual_sampling": {"task_windows": dict(sampled_task_windows),
                                            "window_index_sequence_sha256": sampled_index_digest.hexdigest()},
                        "data_sampler_generator_state": sampler.generator.get_state(),
                        "human_train_demonstrations": configuration["human_train_demonstrations"],
                        "configuration_sha256": config_sha}, checkpoint)
                    print(f"Saved {checkpoint}", flush=True)
                if global_step >= update_budget:
                    break
            if global_step >= update_budget:
                break
    summary = {"policy_type": "native_video_act", "global_step": global_step, "epoch": epoch,
               "samples": len(dataset), "human_train_clips": len(dataset.human_demos),
               "conditioning_mode": args.conditioning_mode, "sampling": sampling_receipt,
               "initial_policy_sha256": initial_policy_sha,
               "actual_sampling": {"task_windows": sampled_task_windows,
                                   "window_index_sequence_sha256": sampled_index_digest.hexdigest()},
               "last_loss": row, "checkpoint": str(checkpoint.resolve()),
               "configuration_sha256": config_sha}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def _sanity_check() -> None:
    """Exercise real CPU video pairing, balanced sampling and gradient receipts."""
    import tempfile
    from types import SimpleNamespace

    import cv2
    from torch.utils.data import Dataset, TensorDataset

    class Rows(Dataset):
        def __init__(self) -> None:
            self.sub_datasets = [TensorDataset(torch.zeros(2, 1)), TensorDataset(torch.zeros(6, 1))]
            for index, sub in enumerate(self.sub_datasets):
                sub.repo_id = f"robot-{index}"

        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int) -> dict[str, Any]:
            dataset_idx = int(index >= 2)
            return {"dataset_idx": torch.tensor(dataset_idx), "repo_id": f"robot-{dataset_idx}",
                    "states": torch.zeros(1, 16), "actions": torch.zeros(4, 14)}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "human.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
        assert writer.isOpened()
        for value in (0, 255):
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
        writer.release()
        manifest = {"schema_version": 2, "tasks": [
            {"task_id": f"task-{index}", "demos": [{"id": f"human-{index}", "repo_id": f"human-{index}",
                "episode_id": 0, "camera": "zed2i", "path": str(path), "role": "train", "num_frames": 2}]}
            for index in range(2)]}
        dataset = NativeVideoDataset(Rows(), manifest, dataset_tasks={0: "task-0", 1: "task-1"},
                                     image_size=(8, 8), video_backend="opencv")
        sampler, receipt = training_sampler(dataset, 7, "equal_task")
        expected, _ = training_sampler(dataset, 7, "equal_task")
        assert list(sampler) == list(expected)
        assert torch.isclose(sampler.weights[:2].sum(), sampler.weights[2:].sum())
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        assert batch["human_video"].shape == (1, 2, 3, 8, 8)
        adapter = torch.nn.Linear(2, 2)
        adapter(torch.zeros(1, 2)).sum().backward()
        fixture = SimpleNamespace(task_encoder=SimpleNamespace(backbone=SimpleNamespace(
            cls_adapter=adapter, _backbone=torch.nn.Linear(2, 2).requires_grad_(False))))
        gradient = first_backward_receipt(fixture, "video")
        assert gradient["adapter_has_gradient"] and not gradient["frozen_backbone_has_gradient"]
        print("native_video_train sanity:", {"video_shape": list(batch["human_video"].shape),
                                              "sampling": receipt, "gradient": gradient})


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--self-test"]:
        _sanity_check()
    else:
        train(tyro.cli(TrainingArgs))
