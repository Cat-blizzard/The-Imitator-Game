"""Frozen per-frame tokenizer for human (and robot) video.

Backends:
  * template: requires token-level output from the benchmark backbone.
  * dinov2_hf: HuggingFace DINOv2, e.g. facebook/dinov2-large.
  * dinov2_hub: original torch.hub DINOv2 path.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def format_video(video: torch.Tensor) -> torch.Tensor:
    """-> float [B, N, C, H, W] in [0,1]."""
    if video.dim() == 4:
        video = video.unsqueeze(1)
    if video.shape[-1] in (3, 4):
        video = video.permute(0, 1, 4, 2, 3)
    video = video[:, :, :3].float()
    if video.max() > 1.5:
        video = video / 255.0
    return video


class FrozenFrameTokenizer(nn.Module):
    def __init__(
        self,
        backend: str = "template",
        template_backbone: Optional[nn.Module] = None,
        seq_patches: int = 32,
        feat_dim: int = 1024,
        hub_repo: str = "facebookresearch/dinov2",
        hub_name: str = "dinov2_vitl14",
        image_size: int = 224,
        hf_model_name: str = "facebook/dinov2-large",
        hf_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        assert backend in ("template", "dinov2_hf", "dinov2_hub"), backend

        self.backend = backend
        self.seq_patches = seq_patches
        self.feat_dim = feat_dim
        self.image_size = image_size
        self._pool_hw = self._pool_grid(seq_patches)

        if backend == "template":
            assert template_backbone is not None, (
                "pass the benchmark backbone for backend='template'"
            )
            self.backbone = template_backbone

        elif backend == "dinov2_hf":
            from transformers import AutoModel

            print(
                f"[FrozenFrameTokenizer] Loading HF DINOv2: {hf_model_name} "
                f"(cache={hf_cache_dir})"
            )
            self.backbone = AutoModel.from_pretrained(
                hf_model_name,
                cache_dir=hf_cache_dir,
            )
            hidden_size = int(self.backbone.config.hidden_size)
            if hidden_size != self.feat_dim:
                raise ValueError(
                    f"HuggingFace DINOv2 hidden_size={hidden_size}, "
                    f"but feat_dim={self.feat_dim}. Set feat_dim={hidden_size}."
                )
            patch_size = getattr(self.backbone.config, "patch_size", 14)
            print(
                f"[FrozenFrameTokenizer] DINOv2 ready: hidden_size={hidden_size}, "
                f"patch_size={patch_size}, pooled_patches={seq_patches}"
            )

        else:
            self.backbone = torch.hub.load(hub_repo, hub_name)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    @staticmethod
    def _pool_grid(n: int):
        h = int(math.sqrt(n))
        while n % h:
            h -= 1
        return (h, n // h)

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        vl_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Return per-frame tokens [B, N, P, F]."""
        if self.backend == "template":
            return self._template_tokens(video, vl_ids)

        video = format_video(video)
        if self.backend == "dinov2_hf":
            return self._dinov2_hf_tokens(video)
        return self._dinov2_hub_tokens(video)

    def _template_tokens(self, video, vl_ids):
        out = self.backbone.encode(
            human_video=(
                video.to(next(self.backbone.parameters()).device)
                if video is not None else None
            ),
            human_vl_ids=vl_ids,
        )

        tok = None
        for k in ("tokens", "patch_tokens", "frame_tokens"):
            if isinstance(out, dict) and k in out and out[k] is not None:
                tok = out[k]
                break

        if tok is None:
            z_dim = None
            if isinstance(out, dict) and out.get("z") is not None:
                z_dim = out["z"].shape[-1]
            raise RuntimeError(
                "template video backbone returned no token-level output. "
                "HyperICL stage C/D/E requires [B,N,P,F] tokens. "
                f"Pooled z dim={z_dim}, feat_dim={self.feat_dim}. "
                "Use tokenizer_backend='dinov2_hf' or expose frame_tokens/patch_tokens "
                "from video_backbone.py."
            )

        if tok.dim() == 3:
            n = getattr(self.backbone, "num_sampled_frames", None) or 1
            if tok.shape[1] % n != 0:
                raise RuntimeError(
                    f"Cannot reshape token sequence {tuple(tok.shape)} into {n} frames."
                )
            tok = tok.view(tok.shape[0], n, -1, tok.shape[-1])

        if tok.dim() != 4:
            raise RuntimeError(
                f"Expected template tokens [B,N,P,F], got {tuple(tok.shape)}"
            )

        if tok.shape[-1] != self.feat_dim:
            raise RuntimeError(
                f"Template token dim={tok.shape[-1]}, feat_dim={self.feat_dim}."
            )

        return tok.float()

    def _dinov2_hf_tokens(self, video: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = video.shape
        x = video.reshape(B * N, C, H, W)

        if (H, W) != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        mean = _IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
        std = _IMAGENET_STD.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        outputs = self.backbone(pixel_values=x, return_dict=True)
        hidden = outputs.last_hidden_state

        patch_size = getattr(self.backbone.config, "patch_size", 14)
        if isinstance(patch_size, (tuple, list)):
            ph, pw = int(patch_size[0]), int(patch_size[1])
        else:
            ph = pw = int(patch_size)

        gh = self.image_size // ph
        gw = self.image_size // pw
        num_patches = gh * gw

        if hidden.shape[1] < num_patches + 1:
            raise RuntimeError(
                f"DINOv2 returned {hidden.shape[1]} tokens, expected at least "
                f"{num_patches + 1}."
            )

        # last num_patches tokens = spatial patch tokens; skips CLS/register tokens
        feats = hidden[:, -num_patches:, :]

        if feats.shape[-1] != self.feat_dim:
            raise RuntimeError(
                f"DINOv2 token dim={feats.shape[-1]}, feat_dim={self.feat_dim}."
            )

        feats = feats.transpose(1, 2).reshape(
            B * N, self.feat_dim, gh, gw
        )
        feats = F.adaptive_avg_pool2d(
            feats,
            self._pool_hw,
        ).flatten(2).transpose(1, 2).contiguous()

        return feats.reshape(
            B, N, feats.shape[1], feats.shape[2]
        ).float()

    def _dinov2_hub_tokens(self, video: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = video.shape
        x = video.reshape(B * N, C, H, W)

        if (H, W) != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        x = (
            x - _IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
        ) / _IMAGENET_STD.to(device=x.device, dtype=x.dtype)

        feats = self.backbone.forward_features(x)["x_norm_patchtokens"]
        g = int(math.sqrt(feats.shape[1]))
        feats = feats.transpose(1, 2).reshape(B * N, -1, g, g)
        feats = F.adaptive_avg_pool2d(
            feats,
            self._pool_hw,
        ).flatten(2).transpose(1, 2)

        return feats.reshape(
            B, N, feats.shape[1], feats.shape[2]
        ).float()
