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
from torch.utils.data import DataLoader
import tyro

from examples.baselines.act.train_act_single import (
    TrainingArgs as SingleTrainingArgs, build_dataloader, _dataset_fps,
)
from examples.baselines.act.train_act_imitator import TrainingArgs as ImitatorTrainingArgs
from examples.baselines.act.native_video_data import NativeVideoDataset
from examples.baselines.act.native_video_model import NativeVideoACTAgent


@dataclass
class TrainingArgs(SingleTrainingArgs):
    human_manifest: str = ""
    model_config: str = ""  # Optional overrides for the existing imitator model.
    save_freq_updates: int = 3000


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


def recorded_contract(args: TrainingArgs, dataset: Any) -> dict[str, Any]:
    """Persist the same observation/action and normalization contract as native ACT."""
    return {
        "fps": _dataset_fps(args.sim_root, args.sub_config),
        "cameras": list(args.cameras), "image_size": list(args.image_size),
        "rgb": "float32_CHW_0_1", "state_dim": args.state_dim,
        "action_dim": args.action_dim, "control_mode": "pd_joint_pos",
        "robot_order": "left_then_right", "normalization_method": args.normalization_method,
        "normalization_min_range": args.normalization_min_range, "split": args.split,
        "native": dataset.main_dataset.meta.info.get("native_data_contract"),
        "source_image_size": list(dataset.main_dataset.meta.features[
            f"observation.images.{args.cameras[0]}"]["shape"][-2:]),
    }


def robot_task_id(configs: list[dict[str, Any]], dataset: Any) -> str:
    """Bind this single-task bridge to the actual native robot recording."""
    if len(configs) != 1:
        raise ValueError("Native video ACT requires one robot dataset/task per run")
    native = dataset.main_dataset.meta.info.get("native_data_contract") or {}
    if configs[0].get("env_id") and native.get("env_id") and configs[0]["env_id"] != native["env_id"]:
        raise ValueError("Configured robot task differs from its recorded native task")
    task_id = configs[0].get("env_id") or native.get("env_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("The robot dataset must identify its task to pair a human demonstration")
    return task_id


def train(args: TrainingArgs) -> None:
    """Optimize the train split; human clips provide conditions, robot actions labels."""
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
    configs = json.loads(Path(args.sub_config).read_text(encoding="utf-8"))
    task_id = robot_task_id(configs, base_dataset)
    dataset = NativeVideoDataset(base_dataset, args.human_manifest, role="train",
                                 image_size=tuple(args.image_size), video_backend=args.video_backend,
                                 task_id=task_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False)
    if not len(loader):
        raise ValueError("No native training samples")
    model_args = model_configuration(args)
    agent = NativeVideoACTAgent(model_args, device)
    backbone_params, other_params = [], []
    for name, parameter in agent.named_parameters():
        if parameter.requires_grad:
            (backbone_params if name.startswith("model.backbones.") else other_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    contract = recorded_contract(args, base_dataset)
    dataset_stats = [{"dataset_idx": idx, **base_dataset.normalizer.dataset_info.get(idx, {}),
                      "stats": base_dataset.normalizer.dataset_stats[idx]}
                     for idx in sorted(base_dataset.normalizer.dataset_stats)]
    single_names = {field.name for field in fields(SingleTrainingArgs)}
    single_args = {key: value for key, value in asdict(args).items() if key in single_names}
    configuration = {"training_args": asdict(args), "model_args": asdict(model_args),
                     "data_contract": contract, "human_manifest": dataset.manifest,
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
    started = time.monotonic()
    agent.train()
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
        for epoch in range(1, epoch_budget + 1):
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = agent.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at update {global_step + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if global_step == 0:
                    encoder = agent.task_encoder.backbone
                    gradient_receipt = {
                        "adapter_gradient_norm": float(sum(
                            p.grad.detach().float().square().sum() for p in encoder.cls_adapter.parameters()
                            if p.grad is not None).sqrt()),
                        "frozen_backbone_has_gradient": any(
                            p.grad is not None for p in encoder._backbone.parameters()),
                    }
                    (output / "first_backward.json").write_text(
                        json.dumps(gradient_receipt, indent=2) + "\n", encoding="utf-8")
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                row = {"global_step": global_step, "epoch": epoch, "loss": float(loss.detach()),
                       **agent.last_loss_terms, "grad_norm": float(grad_norm),
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
                        "human_train_demonstrations": configuration["human_train_demonstrations"],
                        "configuration_sha256": config_sha}, checkpoint)
                    print(f"Saved {checkpoint}", flush=True)
                if global_step >= update_budget:
                    break
            if global_step >= update_budget:
                break
    summary = {"policy_type": "native_video_act", "global_step": global_step, "epoch": epoch,
               "samples": len(dataset), "human_train_clips": len(dataset.human_demos),
               "last_loss": row, "checkpoint": str(checkpoint.resolve()),
               "configuration_sha256": config_sha}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    train(tyro.cli(TrainingArgs))
