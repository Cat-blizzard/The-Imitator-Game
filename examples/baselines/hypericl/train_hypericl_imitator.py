"""Train the HyperICL agent on The Imitator Game (same dataset contract as train_template_imitator.py).

Stages (run one per invocation, chain with --init-from):

    A  base policy        scratch: FM-DiT theta_0 + obs encoders     | vla: domain LoRA on the frozen VLA, then merged
    B  expert bank        per-task LoRA experts on top of frozen theta_0 (parameter / function teachers)
    C  video encoder      E: future prediction + progress alignment (+ domain factorisation if robot_video is given)
    D  compiler           T + D with the full loss set (one-shot HyperICL)
    E  closed-loop + TTT  feedback pass, imagined feedback, latent TTT unrolled

Examples
    # from-scratch version
    python -m examples.baselines.hypericl.train_hypericl_imitator --target scratch --stage A --exp-name s_A
    python -m examples.baselines.hypericl.train_hypericl_imitator --target scratch --stage B --init-from runs/s_A/checkpoints/final_model.pt --exp-name s_B
    python -m examples.baselines.hypericl.train_hypericl_imitator --target scratch --stage C --init-from runs/s_B/checkpoints/final_model.pt --exp-name s_C
    python -m examples.baselines.hypericl.train_hypericl_imitator --target scratch --stage D --init-from runs/s_C/checkpoints/final_model.pt --exp-name s_D
    python -m examples.baselines.hypericl.train_hypericl_imitator --target scratch --stage E --init-from runs/s_D/checkpoints/final_model.pt --exp-name s_E --ttt-steps-train 1 --p-img 0.3

    # pretrained-VLA version (identical commands with --target vla --vla-profile lerobot_smolvla)

The script prints a trainable-parameter breakdown at start-up; use --dit-dim/--dit-depth (scratch)
and --rank/--vla-max-layers/--n-atoms (vla) to match the two budgets (see README.md).
"""
from __future__ import annotations

import inspect
import json
import os
import random
import time
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset, PairedDatasetConfig, get_collate_fn)

from .agent import HyperICLAgent
from .config import HyperICLArgs


def _init_dataloader_worker(worker_id: int, torch_threads: int = 1):
    """Keep each worker lightweight and deterministically seeded.

    Video decoding libraries may internally use CPU threads; with many DataLoader
    workers that can lead to severe oversubscription. Limit PyTorch/OpenCV threads
    inside each worker process.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass


def build_dataloader(args: HyperICLArgs) -> DataLoader:
    skip_human_video = bool(args.skip_human_video_ab and args.stage in ("A", "B"))

    dataset = HumanSimPairedDataset(PairedDatasetConfig(
        human_root=args.human_root,
        sim_root=args.sim_root,
        task_mapping_file=args.task_mapping_file,
        human_dataset_file=args.human_dataset_file,
        sim_dataset_file=args.sim_dataset_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        split="train",
        cameras=args.cameras,
        include_depth=args.include_depth,
        image_size=args.image_size,
        num_frames=args.frozen_backbone_num_frames,
        horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        state_type=args.state_type,
        single_arm=args.single_arm,
        fps=args.fps,
        video_backend=args.video_backend,
        input_mode=args.input_mode,
        include_first_frame=False,
        skip_human_video=skip_human_video,
    ))

    kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=get_collate_fn(args.input_mode),
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = max(1, args.prefetch_factor)
        kwargs["worker_init_fn"] = partial(
            _init_dataloader_worker, torch_threads=args.worker_torch_threads
        )

        mp_context = args.multiprocessing_context.strip().lower()
        if mp_context not in ("", "none", "default"):
            kwargs["multiprocessing_context"] = mp_context

        # Newer PyTorch versions support in_order=False, which prevents one slow
        # worker from blocking ready batches from other workers. This directly
        # helps the "every N workers/steps there is a stall" pattern.
        if "in_order" in inspect.signature(DataLoader).parameters:
            kwargs["in_order"] = args.dataloader_in_order

    print(
        "[dataloader] "
        f"workers={args.num_workers}, prefetch={args.prefetch_factor if args.num_workers > 0 else 0}, "
        f"persistent={args.persistent_workers if args.num_workers > 0 else False}, "
        f"pin_memory={args.pin_memory}, skip_human_video={skip_human_video}, "
        f"in_order={kwargs.get('in_order', 'unsupported')}"
    )
    return DataLoader(**kwargs)


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, agent: HyperICLAgent, optimizer, args: HyperICLArgs, epoch: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ck = {"epoch": epoch, "args": vars(args), "stage": agent.stage,
          "agent_state_dict": agent.state_dict_for_ckpt(),
          "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
          "task_registry": agent.registry.ids,
          "bank_ready": agent.bank_ready or agent.stage == "B"}
    if args.target == "vla" and agent.stage == "A":            # domain LoRA -> dense, merged on load
        ck["domain_lora_dense"] = {k: v[0].cpu() for k, v in agent.bank.dense().items()}
    elif getattr(agent, "_domain_dense", None) is not None:      # carry theta_0's merged delta forward
        ck["domain_lora_dense"] = agent._domain_dense
    torch.save(ck, path)
    print(f"Saved checkpoint: {path}")


def load_into(agent: HyperICLAgent, ckpt_path: str, device) -> Dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    own = agent.state_dict()
    sd, skipped = {}, []
    for k, v in ck["agent_state_dict"].items():
        if k in own and own[k].shape == v.shape:
            sd[k] = v
        else:
            skipped.append(k)
    missing, unexpected = agent.load_state_dict(sd, strict=False)
    if skipped:
        print(f"[load] skipped {len(skipped)} keys with shape mismatch / unknown (e.g. {skipped[:3]})")
    agent.registry.ids = list(ck.get("task_registry", []))
    agent.bank_ready = bool(ck.get("bank_ready", False))
    if ck.get("domain_lora_dense") and agent.args.target == "vla" and agent.stage != "A":
        agent.target.merge_domain_dense(ck["domain_lora_dense"])
        agent._domain_dense = ck["domain_lora_dense"]
    agent.obs_proj_ema.load_state_dict(agent.obs_proj.state_dict())
    agent.to(device)
    return ck



# --------------------------------------------------------------------------- #
# experiment tracking
# --------------------------------------------------------------------------- #
def init_wandb(args: HyperICLArgs, run_name: str, log_dir: Path):
    """Create a W&B run only when --track is enabled.

    wandb stays an optional dependency: normal TensorBoard-only training does not
    import it at all.
    """
    if not args.track:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking was enabled with --track, but `wandb` is not installed. "
            "Install it with `pip install wandb`, then run `wandb login` once "
            "(or use --wandb-mode offline)."
        ) from exc

    if args.wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(
            f"invalid --wandb-mode={args.wandb_mode!r}; "
            "expected one of: online, offline, disabled"
        )

    # Environment variable takes priority, so `export WANDB_MODE=offline` works
    # even if the dataclass default is changed later.
    wandb_mode = os.environ.get("WANDB_MODE", args.wandb_mode).strip().lower()
    if wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError(f"invalid W&B mode: {wandb_mode!r}")

    # Keep stage/target as default tags so runs are easy to filter in the UI.
    tags = list(dict.fromkeys([args.target, f"stage-{args.stage}", *args.wandb_tags]))
    run = wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        tags=tags,
        job_type=f"train-stage-{args.stage}",
        config=vars(args),
        dir=str(log_dir.resolve()),
        mode=wandb_mode,
    )
    return run


def scalar_metrics(out: Dict) -> Dict[str, float]:
    """Detach scalar tensors/numbers from compute_loss() for logging."""
    metrics: Dict[str, float] = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            if v.numel() != 1:
                continue
            v = v.detach().item()
        try:
            metrics[f"train/{k}"] = float(v)
        except (TypeError, ValueError):
            continue
    return metrics


# --------------------------------------------------------------------------- #
def main():
    args = tyro.cli(HyperICLArgs)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    run_name = args.exp_name or f"hypericl_{args.target}_{args.stage}__seed_{args.seed}__{int(time.time())}"
    log_dir = Path("runs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    wandb_run = init_wandb(args, run_name, log_dir)

    dataloader = build_dataloader(args)
    agent = HyperICLAgent(args, device)
    if args.init_from:
        load_into(agent, args.init_from, device)
        agent.set_stage(args.stage)                       # re-apply freezing after loading
    if args.stage in ("D", "E") and args.init_dict_from_bank and agent.bank_ready:
        with torch.no_grad():
            ev = agent.bank.explained_variance(args.n_atoms)
            if ev:
                print(f"[expert bank] mean explained variance by {args.n_atoms} atoms: {np.mean(list(ev.values())):.3f} "
                      f"(M0 gate: >= 0.70)")
            agent.compiler.D.init_atoms_from_bank(agent.target, agent.bank)
        print("[expert bank] manifold atoms initialised from expert SVD")
    elif args.stage in ("D", "E") and not agent.bank_ready:
        print("[warning] no expert bank loaded -> parameter/function teacher losses are OFF (init from a stage-B/C ckpt)")

    # ------------------------------------------------------------------
    # Frozen DINO patch-token cache (C/D/E only)
    # ------------------------------------------------------------------
    if (
        args.stage in ("C", "D", "E")
        and args.dino_cache_enabled
    ):
        from .modules.dino_feature_cache import setup_dino_feature_cache

        dino_cache = setup_dino_feature_cache(
            agent.tokenizer,
            dataloader,
            device,
            args,
        )
        agent.set_dino_feature_cache(dino_cache)

        if args.dino_cache_skip_human_video:
            # DataLoader workers are created lazily on first iteration. We have not
            # iterated the training loader yet, so changing the dataset here is seen
            # by all workers even with persistent_workers=True.
            dataloader.dataset.skip_human_video = True

            # The paired mapping has already been built. Drop the heavy human dataset
            # object so worker processes do not carry video decoder state they never use.
            if hasattr(dataloader.dataset, "human_dataset"):
                dataloader.dataset.human_dataset = None

            print(
                "[DinoCache] training human-video decode disabled: "
                "batches now carry human_repo_id only."
            )

    rep = agent.trainable_report()
    print("trainable parameters (M):", json.dumps({k: round(v, 3) for k, v in rep.items()}, ensure_ascii=False))
    writer.add_text("trainable_params", json.dumps(rep))
    if wandb_run is not None:
        for k, v in rep.items():
            wandb_run.summary[f"params/{k}"] = float(v)

    params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.total_epochs * len(dataloader)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, max(total_steps, 1))

    global_step = 0
    for epoch in range(args.total_epochs):
        agent.train()
        agent.tokenizer.eval()
        if args.target == "vla":
            agent.target.policy.eval()                    # frozen VLA stays in eval mode (dropout / norm)

        pbar = tqdm(dataloader, desc=f"[{args.target}/{args.stage}] epoch {epoch}")
        previous_step_end = time.perf_counter()
        wait_sum = 0.0
        wait_max = 0.0
        wait_count = 0

        for batch in pbar:
            # Time spent waiting for the next batch. If this periodically spikes,
            # the GPU is starving on DataLoader / storage rather than model compute.
            data_wait = time.perf_counter() - previous_step_end
            wait_sum += data_wait
            wait_max = max(wait_max, data_wait)
            wait_count += 1

            out = agent.compute_loss(batch)
            loss = out["loss"]
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                previous_step_end = time.perf_counter()
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in params if p.grad is not None], args.grad_clip
            )
            optimizer.step(); sched.step()
            agent.update_ema()

            if global_step % args.log_freq == 0:
                metrics = scalar_metrics(out)
                metrics["train/lr"] = float(optimizer.param_groups[0]["lr"])
                metrics["train/grad_norm"] = float(grad_norm)
                metrics["train/epoch"] = float(epoch)

                if args.profile_dataloader and wait_count > 0:
                    metrics["perf/data_wait_last_s"] = float(data_wait)
                    metrics["perf/data_wait_avg_s"] = float(wait_sum / wait_count)
                    metrics["perf/data_wait_max_s"] = float(wait_max)

                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=global_step)

                postfix = {
                    k: f"{float(v):.4f}"
                    for k, v in out.items()
                    if k in ("loss", "fm", "param", "func")
                }
                if args.profile_dataloader and wait_count > 0:
                    postfix["wait_avg"] = f"{wait_sum / wait_count:.3f}s"
                    postfix["wait_max"] = f"{wait_max:.3f}s"
                pbar.set_postfix(postfix)

                wait_sum = 0.0
                wait_max = 0.0
                wait_count = 0

            global_step += 1
            previous_step_end = time.perf_counter()

        if epoch > 0 and epoch % args.save_epoch_freq == 0:
            save_checkpoint(str(log_dir / "checkpoints" / f"epoch_{epoch}.pt"), agent, optimizer, args, epoch)

    final_ckpt = log_dir / "checkpoints" / "final_model.pt"
    save_checkpoint(str(final_ckpt), agent, optimizer, args, args.total_epochs)
    writer.close()
    if wandb_run is not None:
        wandb_run.summary["train/global_steps"] = global_step
        wandb_run.summary["checkpoint/final"] = str(final_ckpt)
        wandb_run.finish()


if __name__ == "__main__":
    main()
