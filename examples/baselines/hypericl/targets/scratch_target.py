"""ScratchTarget: FM-DiT trained from scratch. Conditioning = robot latent h_t (from the benchmark's
observation/state encoders, owned by the agent).  theta_0 is trained in stage A and frozen afterwards;
the manifold atoms inside each LoRALinear remain trainable in stages D/E (they belong to the hypernet)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..modules.heads import FlowMatching
from ..modules.lora_dit import LoRADiT, LoRALinear
from .base import PDict, TargetPolicy


class ScratchTarget(TargetPolicy):
    def __init__(self, act_dim: int, h_dim: int, horizon: int, dim: int, depth: int, nhead: int,
                 rank: int, ada_rank: int, n_atoms: int, sample_steps: int):
        super().__init__()
        self.act_dim, self.horizon = act_dim, horizon
        self.dit = LoRADiT(act_dim, h_dim, horizon, dim, depth, nhead, 4, 64, rank, ada_rank, n_atoms)
        self.head = FlowMatching(sample_steps)
        self.n_blocks = self.dit.n_blocks

    def lora_layers(self) -> List[LoRALinear]:
        return self.dit.lora_layers()

    def loss_and_velocity(self, P, robot_obs, actions, h, noise=None, u=None):
        noise = torch.randn_like(actions) if noise is None else noise
        u = torch.rand(actions.shape[0], device=actions.device) if u is None else u
        x_t, tgt, t_in = self.head.train_pair(actions, noise, u)
        v = self.dit(x_t, t_in, h, P)
        x0_hat = self.head.x0_from_v(x_t, u, v)
        return F.mse_loss(v, tgt), v, x0_hat, noise, u

    @torch.no_grad()
    def sample(self, P, robot_obs, h):
        fn = lambda x, t: self.dit(x, t, h, P)
        return self.head.sample(fn, (h.shape[0], self.horizon, self.act_dim), h.device)

    def set_stage(self, stage: str, train_base: bool):
        for m in self.dit.lora_layers():
            m.base.weight.requires_grad_(train_base)
            if m.base.bias is not None:
                m.base.bias.requires_grad_(train_base)
            trainable_atoms = stage in ("D", "E")
            m.atom_P.requires_grad_(trainable_atoms)
            m.atom_Q.requires_grad_(trainable_atoms)
        for n, p in self.dit.named_parameters():
            if "atom_" not in n and ".base." not in n:          # t_fc0/t_fc1/pos (structural)
                p.requires_grad_(train_base)
