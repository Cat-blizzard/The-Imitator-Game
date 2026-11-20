"""Official video-conditioned ACT with native joint data and explicit padding.

Robot inputs are RGB ``(B,1,3,H,W)``, native normalized qpos ``(B,1,S)``
and normalized action chunks ``(B,Q,A)``. Human inputs are sampled RGB clips
``(B,T,3,H,W)`` in [0,1]. A demonstration ID identifies the file content and
sampled frames, rather than only its human task/repository.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from examples.baselines.act.single_task_io import masked_l1
from examples.baselines.act.train_act_imitator import ACTAgent, TrainingArgs
from examples.baselines.encoders.task_encoder.frozen_backbone_cache import (
    CachedFrozenBackbone, RawFeatureCache,
)


class _MemoryRawFeatureCache(RawFeatureCache):
    """Use the official raw-feature cache interface without a shared disk cache."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, torch.Tensor]] = {}
        self._dirty = 0
        self._path = "<native-video in-memory cache>"

    def flush(self) -> None:
        self._dirty = 0


class NativeVideoACTAgent(ACTAgent):
    """Keep the official visual ACT and frozen video encoder architecture.

    Only frozen *raw* video features are cached. Every training loss runs the
    trainable task adapter and task normalization again, retaining gradients.
    Evaluation caches the current adapted task latent after set_demonstration.
    """

    def __init__(self, model_args: TrainingArgs, device: torch.device, *,
                 conditioning_mode: str = "video") -> None:
        if conditioning_mode not in {"video", "constant"}:
            raise ValueError("conditioning_mode must be video or constant")
        if (model_args.obs_horizon != 1 or model_args.include_depth
                or len(model_args.cameras) != 1):
            raise ValueError("Native video ACT requires one RGB camera and obs_horizon=1")
        if (model_args.input_mode != "video_only"
                or model_args.task_encoder_type != "frozen_backbone"
                or model_args.frozen_backbone_lora_rank != 0):
            raise ValueError("Native video ACT requires video_only and a frozen backbone without LoRA")
        super().__init__(model_args, torch.device(device), dataloader=None)
        self.conditioning_mode = conditioning_mode
        self.task_latent_dim = model_args.task_latent_dim
        # Load before optimizer/state_dict creation: lazy loading can replace
        # adapter dimensions and introduce frozen backbone state-dict keys.
        self.task_encoder._load_backbone()
        self.task_encoder.to(self.device)
        self.task_encoder._backbone.eval()
        self.task_encoder = CachedFrozenBackbone(self.task_encoder, _MemoryRawFeatureCache())
        self.last_loss_terms: dict[str, float] = {}
        self._demonstration_summary: dict[str, Any] | None = None

    def train(self, mode: bool = True) -> NativeVideoACTAgent:
        super().train(mode)
        encoder = self.task_encoder
        if isinstance(encoder, CachedFrozenBackbone):
            encoder.backbone._backbone.eval()
        if mode:
            self.clear_cache()
            self._demonstration_summary = None
        return self

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True,
                        assign: bool = False) -> Any:
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        # Restored frozen weights invalidate raw features computed beforehand.
        self.task_encoder._raw_cache._store.clear()
        self.clear_cache()
        self._demonstration_summary = None
        return result

    def _task_z(self, video: torch.Tensor, demo_ids: Sequence[str]) -> torch.Tensor:
        if (video.ndim != 5 or video.shape[2] != 3
                or len(demo_ids) != video.shape[0]
                or not all(isinstance(key, str) and key for key in demo_ids)):
            raise ValueError("Expected human_video (B,T,3,H,W) and one nonempty demo ID per clip")
        if not video.is_floating_point() or not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError("Human RGB clips must be finite floating point in [0,1]")
        if self.conditioning_mode == "constant":
            # Same visual ACT and dataset routing, with the human condition
            # removed. No clip pixels/identity enter the policy in this ablation.
            return torch.zeros(video.shape[0], self.task_latent_dim, device=self.device)
        cache = self.task_encoder._raw_cache
        # Deduplicate cold clips in a robot minibatch; repeated windows share
        # their human demonstration but never share a detached adapter output.
        missing = {key: index for index, key in enumerate(demo_ids) if not cache.is_cached(key)}
        if missing:
            backbone = self.task_encoder.backbone
            indices = torch.tensor(list(missing.values()), device=video.device)
            clips = video.index_select(0, indices).to(self.device, non_blocking=True)
            dispatch = {"dino": backbone._encode_dino, "clip_hf": backbone._encode_clip_hf,
                        "siglip": backbone._encode_siglip, "videomae": backbone._encode_videomae}
            with torch.no_grad():
                raw_cls, raw_seq = dispatch[backbone._type](clips)
            cache.put_batch(list(missing), raw_cls, raw_seq)
        encoded = self.task_encoder.encode(human_video=None, human_vl_ids=list(demo_ids))
        return self.ftask_norm(encoded["z"])

    def _native_observation(self, rgb: torch.Tensor, state: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb.ndim == 5 and rgb.shape[1] == 1:
            rgb = rgb[:, 0]
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        if (rgb.ndim != 4 or rgb.shape[1] != 3 or state.ndim != 2
                or state.shape != (rgb.shape[0], self.state_dim)):
            raise ValueError("Expected native RGB (B,3,H,W) and state (B,state_dim), optionally with one time axis")
        return self._preprocess_obs_for_model({
            "rgb": rgb.to(self.device, dtype=torch.float32, non_blocking=True),
            "state": state.to(self.device, dtype=torch.float32, non_blocking=True),
        })

    def compute_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Masked L1 plus the official KL sum over its 32 posterior coordinates."""
        obs = self._native_observation(batch["view_1"], batch["states"])
        action = batch["actions"].to(self.device, dtype=torch.float32, non_blocking=True)
        is_pad = batch["action_is_pad"].to(self.device, dtype=torch.bool, non_blocking=True)
        task_z = self._task_z(batch["human_video"], batch["human_demo_id"])
        prediction, (mu, logvar) = self.model(
            obs, actions=action, video_feature=task_z, action_is_pad=is_pad,
        )
        l1 = masked_l1(prediction.float(), action, is_pad)
        mu, logvar = mu.float(), logvar.float()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(1).mean()
        self.last_loss_terms = {"l1": float(l1.detach()), "kl": float(kl.detach())}
        return l1 + self.kl_weight * kl

    @torch.no_grad()
    def set_demonstration(self, video: torch.Tensor, demo_id: str) -> dict[str, Any]:
        """Select one real human clip; subsequent get_action calls reuse its z."""
        self.eval()
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.shape[0] != 1:
            raise ValueError("Evaluation selects exactly one human demonstration")
        self._cached_task_z = self._task_z(video, [demo_id]).detach()
        cpu_z = self._cached_task_z.float().cpu().contiguous()
        self._demonstration_summary = {
            "human_demo_id": demo_id,
            "conditioning_mode": self.conditioning_mode,
            "task_z_sha256": hashlib.sha256(cpu_z.numpy().tobytes()).hexdigest(),
            "task_z_norm": float(cpu_z.norm()),
            "task_z_shape": list(cpu_z.shape),
        }
        return dict(self._demonstration_summary)

    @torch.no_grad()
    def get_action(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Predict normalized native action chunks (B,Q,A), using zero CVAE latent."""
        if self._cached_task_z is None:
            raise RuntimeError("Call set_demonstration before native video ACT inference")
        obs = self._native_observation(rgb, state)
        task_z = self._cached_task_z.expand(obs["state"].shape[0], -1)
        action, _ = self.model(obs, actions=None, video_feature=task_z)
        return action


if __name__ == "__main__":
    # Uses the real pretrained video encoder; its weights must be available.
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--frozen-backbone-type", default="dinov2_vitb14")
    options = parser.parse_args()
    torch.manual_seed(11)
    args = TrainingArgs(hidden_dim=32, nheads=4, dim_feedforward=64,
                        enc_layers=1, dec_layers=1, dropout=0.0,
                        task_latent_dim=16, pred_horizon=3, state_dim=16,
                        action_dim=14, frozen_backbone_type=options.frozen_backbone_type,
                        hf_cache_dir=options.hf_cache_dir, frozen_backbone_num_frames=2)
    agent = NativeVideoACTAgent(args, torch.device(options.device))
    batch = {"view_1": torch.zeros(1, 1, 3, 64, 64), "states": torch.zeros(1, 1, 16),
             "actions": torch.zeros(1, 3, 14), "action_is_pad": torch.tensor([[False, True, True]]),
             "human_video": torch.rand(1, 2, 3, 224, 224), "human_demo_id": ["synthetic-self-test-11"]}
    loss = agent.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert agent.task_encoder.backbone.cls_adapter[0].weight.grad.abs().sum() > 0
    summary = agent.set_demonstration(batch["human_video"], batch["human_demo_id"][0])
    actions = agent.get_action(batch["view_1"], batch["states"])
    assert actions.shape == (1, 3, 14) and torch.isfinite(actions).all()
    print({"loss": float(loss.detach()), "actions": list(actions.shape), "conditioning": summary})
