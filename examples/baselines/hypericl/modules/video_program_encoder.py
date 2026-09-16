"""VideoProgramEncoder E  (replaces the mean-pooled task encoder).

    tokens [B, N, P, F]  ->  {
        z_eff   [B, d]        effect token: what changed between first and last frame
        Z_evt   [B, K, d]     K event / phase tokens
        evt_mass[B, K, N]     where in time each event lives (attention mass over frames)
        Z_prog  [B, N, d_p]   per-frame progress tokens (L2-normalised, shared with the robot side)
        z_emb   [B, d_e]      embodiment / domain token (kept, NOT aligned across domains)
        frame_g [B, N, d]     per-frame summary used by the future-prediction head
    }

Also owns:
    future_loss(frame_g, mod)  multi-horizon latent future prediction (V-JEPA style, no RGB).
                               `mod` = optional (shift, scale) from the skill state S -> this is
                               the test-time objective of Latent TTT (the same S must explain
                               the human dynamics).
    domain_losses(...)         factorised domain alignment: GRL on the invariant part, explicit
                               domain classification on z_emb, orthogonality between them.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


class _QueryPool(nn.Module):
    """n_q learned queries cross-attend over a token set."""

    def __init__(self, d: int, n_q: int, nhead: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_q, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln_q, self.ln_kv = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, kv: torch.Tensor, need_weights: bool = False):
        q = self.ln_q(self.q.expand(kv.shape[0], -1, -1))
        out, w = self.attn(q, self.ln_kv(kv), self.ln_kv(kv), need_weights=need_weights,
                           average_attn_weights=True)
        return out, w


class VideoProgramEncoder(nn.Module):
    def __init__(self, feat_dim: int, d: int = 256, d_prog: int = 64, d_emb: int = 32, K: int = 4,
                 n_pool: int = 4, layers: int = 4, nhead: int = 4, max_frames: int = 64,
                 future_deltas: Tuple[int, ...] = (1, 2, 4), n_domains: int = 2):
        super().__init__()
        self.d, self.d_prog, self.d_emb, self.K, self.n_pool = d, d_prog, d_emb, K, n_pool
        self.future_deltas = future_deltas
        self.in_proj = nn.Linear(feat_dim, d)
        self.spatial_pool = _QueryPool(d, n_pool, nhead)
        self.frame_pos = nn.Parameter(torch.randn(1, max_frames, 1, d) * 0.02)
        self.slot_pos = nn.Parameter(torch.randn(1, 1, n_pool, d) * 0.02)
        self.domain_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_domains)])   # per-domain adapter
        enc_layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(enc_layer, layers)
        # read-out heads
        self.eff_mlp = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.evt_pool = _QueryPool(d, K, nhead)
        self.prog_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_prog))
        self.emb_pool = _QueryPool(d, 1, nhead)
        self.emb_proj = nn.Linear(d, d_emb)
        # future-prediction head F_dyn (per-delta embedding, AdaLN-modulated by S during TTT)
        self.delta_emb = nn.Embedding(max(future_deltas) + 1, d)
        self.fut_ln = nn.LayerNorm(d, elementwise_affine=False)
        self.fut_mlp = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        # domain heads
        self.dom_inv = nn.Sequential(nn.Linear(d + d_prog, 128), nn.GELU(), nn.Linear(128, n_domains))
        self.dom_emb = nn.Sequential(nn.Linear(d_emb, 64), nn.GELU(), nn.Linear(64, n_domains))

    # ------------------------------------------------------------------ #
    def forward(self, tokens: torch.Tensor, domain: int = 0) -> Dict[str, torch.Tensor]:
        B, N, P, Fd = tokens.shape
        x = self.in_proj(tokens.reshape(B * N, P, Fd))                       # [B*N, P, d]
        pooled, _ = self.spatial_pool(x)                                    # [B*N, n_pool, d]
        pooled = pooled.view(B, N, self.n_pool, self.d)
        pooled = pooled + self.frame_pos[:, :N] + self.slot_pos
        pooled = self.domain_ln[domain](pooled)
        H = self.temporal(pooled.reshape(B, N * self.n_pool, self.d))       # [B, N*n_pool, d]
        frame_g = H.view(B, N, self.n_pool, self.d).mean(2)                # [B, N, d]

        g0, g1 = frame_g[:, 0], frame_g[:, -1]
        z_eff = self.eff_mlp(torch.cat([g0, g1, g1 - g0], dim=-1))
        Z_evt, w = self.evt_pool(H, need_weights=True)                      # w: [B, K, N*n_pool]
        evt_mass = w.view(B, self.K, N, self.n_pool).sum(-1)               # [B, K, N]
        evt_mass = evt_mass / evt_mass.sum(-1, keepdim=True).clamp_min(1e-6)
        Z_prog = F.normalize(self.prog_proj(frame_g), dim=-1)              # [B, N, d_p]
        z_emb, _ = self.emb_pool(H)
        z_emb = self.emb_proj(z_emb[:, 0])
        return dict(z_eff=z_eff, Z_evt=Z_evt, evt_mass=evt_mass, Z_prog=Z_prog, z_emb=z_emb,
                    frame_g=frame_g, H=H)

    # ------------------------------------------------------------------ #
    def future_loss(self, frame_g: torch.Tensor,
                    mod: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Multi-horizon latent future prediction. mod=(shift[B,d], scale[B,d]) for TTT."""
        B, N, d = frame_g.shape
        total, n = frame_g.new_zeros(()), 0
        for delta in self.future_deltas:
            if N <= delta:
                continue
            src, tgt = frame_g[:, :-delta], frame_g[:, delta:].detach()
            h = self.fut_ln(src)
            if mod is not None:
                shift, scale = mod
                h = h * (1 + scale[:, None]) + shift[:, None]
            de = self.delta_emb(torch.full((B, N - delta), delta, dtype=torch.long, device=frame_g.device))
            pred = self.fut_mlp(torch.cat([h, de], dim=-1))
            total = total + F.mse_loss(pred, tgt)
            n += 1
        return total / max(n, 1)

    def domain_losses(self, enc: Dict[str, torch.Tensor], domain: int, lam: float = 1.0):
        inv = torch.cat([enc["z_eff"], enc["Z_prog"].mean(1)], dim=-1)
        tgt = torch.full((inv.shape[0],), domain, dtype=torch.long, device=inv.device)
        l_conf = F.cross_entropy(self.dom_inv(grad_reverse(inv, lam)), tgt)     # invariant part: confuse
        l_emb = F.cross_entropy(self.dom_emb(enc["z_emb"]), tgt)               # embodiment part: predict
        z_inv = F.normalize(enc["z_eff"], dim=-1)
        z_e = F.normalize(enc["z_emb"], dim=-1)
        # orthogonality between (projected) invariant and embodiment codes
        l_orth = (z_inv[:, : z_e.shape[-1]] * z_e).sum(-1).pow(2).mean()
        return l_conf, l_emb, l_orth

    def phase_anchors(self, enc: Dict[str, torch.Tensor], uniform: bool = False) -> torch.Tensor:
        """rho [B, K, d_p]: progress anchor of each event (attention mass over frames)."""
        Z_prog, mass = enc["Z_prog"], enc["evt_mass"]
        if uniform:
            B, K, N = mass.shape
            idx = torch.arange(N, device=mass.device)
            seg = (idx * K) // N
            mass = F.one_hot(seg, K).float().t()[None].expand(B, -1, -1)
            mass = mass / mass.sum(-1, keepdim=True)
        return F.normalize(torch.einsum("bkn,bnd->bkd", mass, Z_prog), dim=-1)
