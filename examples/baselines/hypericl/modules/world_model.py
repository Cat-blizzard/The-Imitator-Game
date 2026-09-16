"""Latent world model W: (h_t, a_{t:t+H}) -> h_{t+H}.  Never renders RGB.

Uses (see DESIGN §4.6):
  * L_world  : predict the EMA-target robot latent one chunk ahead (needs `robot_next_obs` in the batch)
  * L_effect : EffectHead(h_start, h_end_imagined) should match the human effect token z_eff
  * imagined feedback during training: replace the real next observation with W(h, x0_hat) with
    probability p_img so that T learns to correct S under self-induced drift.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LatentDynamics(nn.Module):
    def __init__(self, h_dim: int, act_dim: int, horizon: int, hidden: int = 512):
        super().__init__()
        self.act_enc = nn.Sequential(nn.Conv1d(act_dim, 64, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(64, 128, 5, padding=2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.net = nn.Sequential(nn.Linear(h_dim + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, h_dim))

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """h [B,dh], a [B,T,act] -> h_next [B,dh] (residual)."""
        return h + self.net(torch.cat([h, self.act_enc(a.transpose(1, 2))], dim=-1))


class EffectHead(nn.Module):
    def __init__(self, h_dim: int, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * h_dim, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, h0: torch.Tensor, h1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h0, h1, h1 - h0], dim=-1))
