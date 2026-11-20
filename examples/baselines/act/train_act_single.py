"""
Single-Task ACT Training (LeRobot)
====================================
No task encoder, no human dataset.
DETRVAE conditioned on projected obs_latent + state_latent. A dummy zero
vector of dim 1 is passed as video_feature so the existing video-conditioned
DETRVAE stays unmodified; this constant feature carries no task information.

Architecture:
  obs_encoder   →  obs_feat   (obs_latent_dim)
  state_encoder →  state_feat (state_latent_dim)
  [obs_feat ; state_feat]  →  obs_proj (hidden_dim)  →  DETRVAE
  video_feature = zeros(B, 1)

Usage:
  python train_act_single.py \
    --sub-config config/sub_configs/L0_TwoRobotCleanCup-v1.json \
    --sim-root   /data/sim \
    --output-dir runs/act_single_task/L0_CleanCup
"""

ALGO_NAME = "ACTSingleTask"

import json
import os, random, time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro

from imitator.dataset.lerobot_dataloader import (
    LeRobotDataConfig,
    build_lerobot_dataset,
)
from examples.baselines.encoders.obs_encoder import ObservationEncoder, ObsEncoderConfig
from examples.baselines.encoders.state_encoder import StateEncoder, StateEncoderConfig
from imitator.eval.native_io import masked_l1
from examples.baselines.act.act.detr_video.detr_vae import (
    DETRVAE,
    build_encoder,
    build_transformer,
)


# =============================================================================
# Training arguments
# =============================================================================

@dataclass
class TrainingArgs:
    """Training configuration for single-task ACT."""

    # ── Experiment ────────────────────────────────────────────────────────────
    exp_name:           Optional[str] = None
    seed:               int  = 1
    cuda:               bool = True
    track:              bool = False
    wandb_project_name: str  = "ACTSingleTask"
    wandb_entity:       Optional[str] = None

    # ── Data ──────────────────────────────────────────────────────────────────
    sub_config:    str  = ""
    sim_root:      str  = "./data/sim"
    state_type:    str  = "qpos"
    cameras:       List[str] = field(default_factory=lambda: ["cam1", "cam2"])
    include_depth: bool = False
    single_arm:    bool = False
    image_size:    List[int] = field(default_factory=lambda: [224, 224])
    split: str = "train"
    enable_augmentation: bool = True
    normalization_method: str = "bounds_q99"
    normalization_min_range: float = 0.0
    normalization_stats_source: str = "metadata"
    video_backend: str = "torchcodec"
    tolerance_s: float = 0.05

    # ── Model dims ────────────────────────────────────────────────────────────
    state_dim:           int = 18
    action_dim:          int = 16
    obs_latent_dim:      int = 256
    state_latent_dim:    int = 64
    obs_encoder_type:    str = "resnet18"
    obs_freeze_backbone: bool = False
    obs_finetune_layers: int = 2
    state_type_enc:      str = "mlp"
    state_hidden_dim:    int = 128
    state_num_layers:    int = 2
    state_num_heads:     int = 4

    # ── ACT / DETRVAE ─────────────────────────────────────────────────────────
    pred_horizon:       int   = 16
    obs_horizon:        int   = 1
    kl_weight:          float = 10.0
    hidden_dim:         int   = 256
    dim_feedforward:    int   = 2048
    nheads:             int   = 8
    enc_layers:         int   = 4
    dec_layers:         int   = 7
    dropout:            float = 0.1
    pre_norm:           bool  = False
    # Backbone LR separate from main LR
    lr_backbone:        float = 1e-5

    # ── Training ──────────────────────────────────────────────────────────────
    max_updates:       Optional[int] = None  # Explicit update budget; otherwise retain epochs.
    total_epochs:      int   = 30        # number of full passes over the dataset
    save_freq_epochs:  int   = 5         # save a checkpoint every N epochs
    batch_size:        int   = 32
    lr:                float = 1e-4
    weight_decay:      float = 1e-4
    grad_clip:         float = 1.0
    num_workers:       int   = 4
    log_freq:          int   = 100       # log every N gradient steps (within epoch)

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "runs/act_single_task"


# =============================================================================
# Minimal DETRVAE config shim
# =============================================================================

class _ACTCfg:
    """Minimal namespace that build_transformer / build_encoder expect."""
    def __init__(self, args: TrainingArgs, hidden_dim: int, num_queries: int,
                 action_dim: int, state_dim: int):
        # transformer
        self.hidden_dim      = hidden_dim
        self.dim_feedforward = args.dim_feedforward
        self.nheads          = args.nheads
        self.enc_layers      = args.enc_layers
        self.dec_layers      = args.dec_layers
        self.dropout         = args.dropout
        self.pre_norm        = args.pre_norm
        self.num_queries     = num_queries
        # encoder (VAE)
        self.latent_dim      = hidden_dim
        self.action_dim      = action_dim
        self.state_dim       = state_dim


# =============================================================================
# Agent
# =============================================================================

class ACTSingleTaskAgent(nn.Module):
    """ACT agent — single-task, no task encoder."""

    def __init__(self, args: TrainingArgs, device: torch.device):
        super().__init__()
        self.args    = args
        self.device  = device
        self.kl_weight   = args.kl_weight
        self.pred_horizon = args.pred_horizon
        self.obs_horizon = args.obs_horizon

        # ── Obs encoder ───────────────────────────────────────────────────────
        self.obs_encoder = ObservationEncoder(ObsEncoderConfig(
            encoder_type=args.obs_encoder_type,
            image_size=args.image_size[0],
            output_dim=args.obs_latent_dim,
            hidden_dim=args.obs_latent_dim,
            freeze_backbone=args.obs_freeze_backbone,
            finetune_layers=args.obs_finetune_layers,
        )).to(device)

        # ── State encoder ─────────────────────────────────────────────────────
        self.state_encoder = StateEncoder(StateEncoderConfig(
            state_type=args.state_type,
            state_dim=args.state_dim,
            num_frames=args.obs_horizon,
            encoder_type=args.state_type_enc,
            hidden_dim=args.state_hidden_dim,
            output_dim=args.state_latent_dim,
            num_layers=args.state_num_layers,
            num_heads=args.state_num_heads,
        )).to(device)

        obs_cond_dim = args.obs_latent_dim + args.state_latent_dim

        # ── Project obs_cond → DETRVAE hidden_dim ────────────────────────────
        self.obs_proj = nn.Linear(obs_cond_dim, args.hidden_dim).to(device)

        # ── DETRVAE ───────────────────────────────────────────────────────────
        # video_feature_dim=1 with a zero vector → constant task conditioning
        act_cfg = _ACTCfg(
            args,
            hidden_dim=args.hidden_dim,
            num_queries=args.pred_horizon,
            action_dim=args.action_dim,
            state_dim=args.hidden_dim,
        )
        self.model = DETRVAE(
            backbones=None,
            transformer=build_transformer(act_cfg),
            encoder=build_encoder(act_cfg),
            state_dim=args.hidden_dim,
            action_dim=args.action_dim,
            num_queries=args.pred_horizon,
            video_feature_dim=1,             # dummy 1-D task feature (always zero)
        ).to(device)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  ACTSingleTaskAgent: {param_count/1e6:.1f}M params | "
              f"obs_cond_dim={obs_cond_dim}  hidden_dim={args.hidden_dim}")

    # ------------------------------------------------------------------
    def _encode_obs(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        rgb  : (B, C, H, W) or (B, T, C, H, W)
        state: (B, D)        or (B, T, D)
        """
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(1)       # (B, C, H, W) → (B, 1, C, H, W)
        if state.dim() == 2:
            state = state.unsqueeze(1)   # (B, D) → (B, 1, D)

        T = min(state.shape[1], self.obs_horizon)

        obs_in = rgb[:, :T]
        if T == 1:
            obs_in = obs_in.squeeze(1)   # (B, 1, C, H, W) → (B, C, H, W)

        obs_feat,   _ = self.obs_encoder(obs_in)           # (B, obs_latent_dim)
        state_feat, _ = self.state_encoder(state[:, :T])   # (B, state_latent_dim)
        return self.obs_proj(torch.cat([obs_feat, state_feat], dim=-1))

    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        robot_obs = batch.get("robot_obs", batch)
        rgb    = robot_obs.get("rgb",   robot_obs.get("view_1")).to(self.device)
        state  = robot_obs.get("state", robot_obs.get("states")).to(self.device)
        action = batch.get("robot_actions", batch.get("actions", batch.get("action"))).to(self.device)   # (B, pred_horizon, action_dim)

        B = rgb.shape[0]
        obs_for_model = self._encode_obs(rgb, state)  # (B, hidden_dim)
        video_feature = torch.zeros(B, 1, device=self.device)

        is_pad = batch.get("action_is_pad", batch.get("actions_is_pad"))
        if is_pad is not None:
            is_pad = is_pad.to(self.device, dtype=torch.bool)
        a_hat, (mu, log_var) = self.model(
            obs_for_model, actions=action, video_feature=video_feature, action_is_pad=is_pad
        )

        # L1 reconstruction + KL
        l1_loss = F.l1_loss(a_hat, action) if is_pad is None else masked_l1(a_hat, action, is_pad)
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        self.last_loss_terms = {"l1": float(l1_loss.detach()), "kl": float(kl_loss.detach())}
        return l1_loss + self.kl_weight * kl_loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """→ (B, pred_horizon, action_dim) in normalized space"""
        B = rgb.shape[0]
        obs_for_model = self._encode_obs(rgb, state)
        video_feature = torch.zeros(B, 1, device=self.device)
        a_hat, _ = self.model(obs_for_model, actions=None, video_feature=video_feature)
        return a_hat


# =============================================================================
# Dataset / DataLoader
# =============================================================================

def _auto_detect_cameras(sim_root, dataset_file, fallback):
    try:
        import json as _j, os as _o
        from imitator.dataset.lerobot_dataset import LeRobotDataset as _D
        with open(dataset_file) as _f:
            _c = _j.load(_f)[0]
        _probe = _D(repo_id=_c["repo_id"],
                    root=_o.path.join(sim_root, _c.get("root", _c["repo_id"])),
                    delta_timestamps=None)
        _cams = [k.replace("observation.images.", "")
                 for k, v in _probe.meta.features.items()
                 if k.startswith("observation.images.") and not k.endswith("_depth")
                 and v.get("dtype") in ("image", "video")]
        if _cams:
            if set(_cams) != set(fallback):
                print(f"  📷 Auto-detected cameras: {_cams}  (config had: {fallback})")
            return _cams
    except Exception as _e:
        print(f"  ⚠️  Camera auto-detect failed ({_e}); using {fallback}")
    return fallback


def _dataset_fps(sim_root: str, dataset_file: str) -> float:
    """Use the recorded sampling rate for action-chunk timestamp offsets."""
    with open(dataset_file) as stream:
        configs = json.load(stream)
    if not configs:
        raise ValueError("The dataset configuration is empty")
    rates = set()
    for config in configs:
        dataset_root = Path(sim_root) / config.get("root", config["repo_id"])
        with (dataset_root / "meta" / "info.json").open() as stream:
            rate = json.load(stream)["fps"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
            raise ValueError(f"Invalid dataset fps {rate!r} in {dataset_root}")
        rates.add(rate)
    if len(rates) != 1:
        raise ValueError(f"Single-task ACT requires one dataset sampling rate, got {sorted(rates)}")
    return rates.pop()


def build_dataloader(args: TrainingArgs, *, normalization_stats=None):
    native = args.normalization_stats_source == "train_rows"
    cameras = list(args.cameras) if native else _auto_detect_cameras(args.sim_root, args.sub_config, args.cameras)
    if native and (len(cameras) != 1 or args.include_depth):
        raise ValueError("Native ACT consumes one explicitly selected RGB camera")
    cfg = LeRobotDataConfig(
        source_type="sim",
        root=args.sim_root,
        split=args.split,
        dataset_file=args.sub_config,
        cameras=cameras,
        include_depth=args.include_depth,
        image_size=tuple(args.image_size),
        state_type=args.state_type,
        single_arm=args.single_arm,
        horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        fps=_dataset_fps(args.sim_root, args.sub_config),
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
        enable_augmentation=args.enable_augmentation,
        normalization_method=args.normalization_method,
        normalization_min_range=args.normalization_min_range,
        normalization_stats=normalization_stats,
        normalization_stats_source=args.normalization_stats_source,
    )
    dataset = build_lerobot_dataset(cfg)
    if native:
        if dataset.state_dim != args.state_dim or dataset.action_dim != args.action_dim:
            raise ValueError("Configured ACT state/action dimensions differ from the dataset")
    print(f"  Dataset size: {len(dataset)} samples")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.split == "train",
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.normalization_stats_source == "metadata" and args.split == "train",
    )
    return loader, dataset


# =============================================================================
# Training loop
# =============================================================================

def train(args: TrainingArgs):
    if args.split != "train":
        raise ValueError("Optimizer updates require --split train")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device     = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name   = args.exp_name or f"{ALGO_NAME}_{Path(args.sub_config).stem}_{args.seed}"
    writer     = SummaryWriter(str(output_dir / "tb" / run_name))

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   name=run_name, config=vars(args), sync_tensorboard=True)

    print("\n📂 Loading dataset...")
    dataloader, dataset = build_dataloader(args)

    print("\n🤖 Building agent...")
    agent = ACTSingleTaskAgent(args, device)

    # ── Two param groups: backbone (lower LR) vs rest ─────────────────────────
    backbone_params = list(agent.obs_encoder.backbone.parameters()) \
                      if hasattr(agent.obs_encoder, "backbone") else []
    backbone_ids    = {id(p) for p in backbone_params}
    other_params    = [p for p in agent.parameters() if id(p) not in backbone_ids]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": other_params,    "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    losses = defaultdict(list)

    steps_per_epoch = len(dataloader)
    if not steps_per_epoch or args.total_epochs < 1 or (args.max_updates is not None and args.max_updates < 1):
        raise ValueError("Training requires a nonempty loader and a positive update/epoch budget")
    epoch_budget = ((args.max_updates + steps_per_epoch - 1) // steps_per_epoch
                    if args.max_updates is not None else args.total_epochs)
    print(f"\n🚀 Training {run_name} for {epoch_budget} epochs "
          f"({steps_per_epoch} steps/epoch)\n")
    agent.train()
    global_step = 0

    for epoch in tqdm(range(1, epoch_budget + 1), desc="Epochs"):
        epoch_losses: List[float] = []

        for batch in dataloader:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = agent.compute_loss(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            losses["loss"].append(loss_val)
            epoch_losses.append(loss_val)

            if global_step % args.log_freq == 0:
                avg = np.mean(losses["loss"]); losses["loss"].clear()
                writer.add_scalar("train/loss", avg, global_step)
                tqdm.write(f"[E{epoch:03d}/{epoch_budget} "
                           f"S{global_step:>8}] loss={avg:.5f}")

            if args.max_updates is not None and global_step >= args.max_updates:
                break

        # ── End-of-epoch bookkeeping ──────────────────────────────────────────
        epoch_avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        writer.add_scalar("train/epoch_loss", epoch_avg, epoch)
        tqdm.write(f"── Epoch {epoch:03d}/{epoch_budget} avg_loss={epoch_avg:.5f}")

        if epoch % args.save_freq_epochs == 0 or epoch == epoch_budget:
            ckpt = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch":            epoch,
                "global_step":      global_step,
                "agent_state_dict": agent.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "args":             vars(args),
                "data_contract": {
                    "fps": _dataset_fps(args.sim_root, args.sub_config),
                    "cameras": list(args.cameras), "image_size": list(args.image_size),
                    "state_dim": args.state_dim, "action_dim": args.action_dim,
                    "normalization_method": args.normalization_method,
                    "normalization_min_range": args.normalization_min_range,
                    "normalization_stats_source": args.normalization_stats_source,
                    "native": dataset.main_dataset.meta.info.get("native_data_contract"),
                },
                "dataset_stats":    [
                    {
                        "dataset_idx": idx,
                        **dataset.normalizer.dataset_info.get(idx, {}),
                        "stats": dataset.normalizer.dataset_stats.get(idx, {}),
                    }
                    for idx in sorted(dataset.normalizer.dataset_stats.keys())
                ],
                "dataset_info":     [{
                    "repo_id": ds.repo_id if hasattr(ds, "repo_id") else "",
                    "root":    str(ds.root),
                } for ds in dataset.sub_datasets] if hasattr(dataset, "sub_datasets") else [],
            }, ckpt)
            print(f"💾 Saved: {ckpt}")

    writer.close()
    if args.track:
        import wandb; wandb.finish()
    print("\n✅ Training complete.")


if __name__ == "__main__":
    train(tyro.cli(TrainingArgs))
