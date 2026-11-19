"""Block-level pseudo-forward/backward refiner for HyperICL.

This module borrows the useful idea from the original learned-optimizer hypernet:
use target-network topology to propagate a forward-like signal and a backward-like
credit signal before changing parameters.  Unlike the original implementation it
NEVER encodes/decodes full target weights.  It refines HyperICL's compact block
skill state S [B, L, d], preserving the expert-manifold + residual-LoRA decoder.

For the scratch DiT, block order is:
    input -> DiT block 1 -> ... -> DiT block depth -> final
so causal / reverse-causal attention is an explicit approximation of target
forward / backward topology.
"""
from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttnFFN(nn.Module):
    def __init__(self, d: int, nhead: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.ln1(x)
        y, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        x = x + y
        return x + self.ff(self.ln2(x))


class _CrossAttn(nn.Module):
    def __init__(self, d: int, nhead: int):
        super().__init__()
        self.q_ln = nn.LayerNorm(d)
        self.kv_ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.out_ln = nn.LayerNorm(d)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(
            self.q_ln(q), self.kv_ln(kv), self.kv_ln(kv), need_weights=False
        )
        return self.out_ln(q + y)


class BlockHyperOptRefiner(nn.Module):
    """Refine layer/block skill tokens using learned pseudo-forward/backward signals.

    Identity at initialization: the final delta projection is zero initialized.
    This lets an old C/D checkpoint be loaded safely; before training this module,
    enabling it does not perturb S.
    """

    def __init__(
        self,
        d: int,
        n_blocks: int,
        nhead: int = 4,
        num_steps: int = 2,
        layers: int = 1,
        lr_max: float = 0.10,
        dropout: float = 0.0,
    ):
        super().__init__()
        del dropout
        self.d = int(d)
        self.n_blocks = int(n_blocks)
        self.num_steps = int(num_steps)
        self.lr_max = float(lr_max)

        self.block_emb = nn.Parameter(torch.randn(1, n_blocks, d) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(max(num_steps, 1), d) * 0.02)
        self.in_ln = nn.LayerNorm(d)
        self.human_ca = _CrossAttn(d, nhead)
        self.fb_ca = _CrossAttn(d, nhead)

        self.fwd_layers = nn.ModuleList([_AttnFFN(d, nhead) for _ in range(layers)])
        self.bwd_layers = nn.ModuleList([_AttnFFN(d, nhead) for _ in range(layers)])

        # pseudo-gradient update.  Last layer zero-init => exact identity at start.
        self.delta_net = nn.Sequential(
            nn.LayerNorm(4 * d),
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

        self.lr_head = nn.Linear(d, 1)
        nn.init.zeros_(self.lr_head.weight)
        nn.init.constant_(self.lr_head.bias, -2.0)

        # Non-persistent diagnostics.
        self.last_stats: Dict[str, float] = {}

    @staticmethod
    def _fwd_mask(L: int, device) -> torch.Tensor:
        # Query i can attend to blocks <= i.
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    @staticmethod
    def _bwd_mask(L: int, device) -> torch.Tensor:
        # Query i can attend to blocks >= i (reverse-topological propagation).
        return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=-1)

    def forward(
        self,
        S: torch.Tensor,
        human_tokens: torch.Tensor,
        feedback_tokens: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        B, L, d = S.shape
        if L != self.n_blocks or d != self.d:
            raise ValueError(
                f"BlockHyperOptRefiner expected [B,{self.n_blocks},{self.d}], got {tuple(S.shape)}"
            )
        K = self.num_steps if steps is None else int(steps)
        if K <= 0:
            return S

        fmask = self._fwd_mask(L, S.device)
        bmask = self._bwd_mask(L, S.device)
        S0 = S
        lr_acc = []

        for it in range(K):
            x = self.in_ln(S + self.block_emb)
            hctx = self.human_ca(x, human_tokens)
            if feedback_tokens is not None and feedback_tokens.shape[1] > 0:
                hctx = self.fb_ca(hctx, feedback_tokens)

            fwd = hctx + self.step_emb[min(it, self.step_emb.shape[0] - 1)][None, None]
            for layer in self.fwd_layers:
                fwd = layer(fwd, fmask)

            bwd = hctx + self.step_emb[min(it, self.step_emb.shape[0] - 1)][None, None]
            for layer in self.bwd_layers:
                bwd = layer(bwd, bmask)

            pseudo_grad = self.delta_net(torch.cat([x, hctx, fwd, bwd], dim=-1))
            lr = torch.sigmoid(self.lr_head(hctx)) * self.lr_max
            S = S + lr * torch.tanh(pseudo_grad)
            lr_acc.append(lr.detach())

        with torch.no_grad():
            lr_cat = torch.cat(lr_acc, dim=1) if lr_acc else torch.zeros(1, device=S.device)
            self.last_stats = {
                "refiner_delta_rms": float((S - S0).float().pow(2).mean().sqrt().item()),
                "refiner_lr_mean": float(lr_cat.float().mean().item()),
                "refiner_lr_max": float(lr_cat.float().max().item()),
            }
        return S
