"""TargetPolicy: the interface the HyperICL agent needs from the policy whose parameters it generates.

Two implementations with identical semantics:
    targets.scratch_target.ScratchTarget       FM-DiT trained from scratch (theta_0 = stage A)
    targets.pretrained_target.PretrainedTarget  frozen pretrained VLA/WAM (lerobot SmolVLA / pi0, GR00T...)
                                                with generated LoRA injected into its action expert
The agent, the encoder E, the compiler (T, D, TTT), the expert bank and every loss are shared.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..modules.lora_dit import LayerSpec, LoRALinear

PDict = Dict[str, Dict[str, torch.Tensor]]


class TargetPolicy(nn.Module):
    act_dim: int
    horizon: int
    n_blocks: int

    # ---- structure ----------------------------------------------------- #
    def layer_specs(self) -> List[LayerSpec]:
        return [m.spec for m in self.lora_layers()]

    def lora_layers(self) -> List[LoRALinear]:
        raise NotImplementedError

    def layer(self, name: str) -> LoRALinear:
        for m in self.lora_layers():
            if m.spec.name == name:
                return m
        raise KeyError(name)

    @torch.no_grad()
    def base_norms(self) -> Dict[str, torch.Tensor]:
        return {m.spec.name: m.base.weight.float().pow(2).sum() for m in self.lora_layers()}

    # ---- training / inference ----------------------------------------- #
    def loss_and_velocity(self, P: Optional[PDict], robot_obs: Dict, actions: torch.Tensor, h: torch.Tensor,
                          noise: Optional[torch.Tensor] = None, u: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """-> (fm_loss, v_pred [B,T,A] or None, x0_hat [B,T,A] or None, noise, u).
        `noise`/`u` are reused across student / teacher / second-clip calls so that the
        function-space losses compare the SAME (x_t, t)."""
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, P: Optional[PDict], robot_obs: Dict, h: torch.Tensor) -> torch.Tensor:
        """-> actions [B, horizon, act_dim] (normalised action space of the dataset)."""
        raise NotImplementedError

    # ---- stage freezing -------------------------------------------------- #
    def set_stage(self, stage: str, train_base: bool):
        """train_base=True only in stage A (scratch) / domain-LoRA stage A (vla)."""
        raise NotImplementedError

    def trainable_summary(self) -> Dict[str, int]:
        return {"target": sum(p.numel() for p in self.parameters() if p.requires_grad)}
