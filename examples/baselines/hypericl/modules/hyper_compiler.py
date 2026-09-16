"""HyperCompiler = ContextTransformer T  +  HyperDecoder D  +  LatentTTT P_phi.

    Z_H  = E(V_H)                                          (video_program_encoder)
    S0   = T(None, human_tokens(Z_H), None)                one-shot skill state      [B, L, d]
    S0'  = TTT(S0, Z_H, K)                                 optional, human-video self-supervision
    S_c  = T(S_{c-1}, human_tokens, feedback_{<=c})        closed-loop recurrence (constant cost)
    P    = D(S_c, alpha_c)                                 per-task parameter deltas for LoRADiT

Design notes (see DESIGN_H2R_HYPERNET.md):
  * S is the fast state: history is compressed into L skill tokens (one per target block),
    like RoboTTT's fast weights, but explicitly decoded into policy parameters.
  * D starts at the base policy: coefficient heads / residual heads are zero-init and the
    gate starts at sigmoid(-4). HypoGen's zero-init residual principle is baked in.
  * Phase (event) structure only touches AdaLN biases: b = b^G + sum_k alpha_k b^k.
  * P_phi (global attention over [grad tokens, S, human tokens]) plays the role that the
    pseudo-gradient block played in HyLaP, but it now preconditions a REAL self-supervised
    gradient on the human video.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_dit import LayerSpec, LoRADiT, LoRALinear


# --------------------------------------------------------------------------- #
# feedback tokens r_c (the red tokens in the sketch)
# --------------------------------------------------------------------------- #
class FeedbackTokenizer(nn.Module):
    def __init__(self, h_dim: int, d_prog: int, act_dim: int, horizon: int, d: int,
                 use_prev_action: bool = False, max_chunks: int = 512):
        super().__init__()
        self.use_prev_action, self.horizon, self.act_dim = use_prev_action, horizon, act_dim
        self.act_enc = nn.Sequential(nn.Conv1d(act_dim, 64, 5, padding=2), nn.GELU(),
                                     nn.Conv1d(64, 64, 5, padding=2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.chunk_pos = nn.Embedding(max_chunks, d)
        in_dim = h_dim + d_prog + (64 if use_prev_action else 0)
        self.mlp = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, h: torch.Tensor, p_R: torch.Tensor, chunk_idx: torch.Tensor,
                a_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [h, p_R]
        if self.use_prev_action:
            a_prev = (torch.zeros(h.shape[0], self.act_dim, self.horizon, device=h.device)
                      if a_prev is None else a_prev.transpose(1, 2))
            parts.append(self.act_enc(a_prev))
        return self.mlp(torch.cat(parts, dim=-1)) + self.chunk_pos(chunk_idx.clamp(max=self.chunk_pos.num_embeddings - 1))


# --------------------------------------------------------------------------- #
# ContextTransformer T
# --------------------------------------------------------------------------- #
class ContextTransformer(nn.Module):
    def __init__(self, d: int, L: int, K: int, d_prog: int, layers: int = 4, nhead: int = 4, M_fb: int = 8):
        super().__init__()
        self.d, self.L, self.K, self.M_fb = d, L, K, M_fb
        self.e_robot = nn.Parameter(torch.randn(1, 1, d) * 0.02)         # target-embodiment token
        self.S_init = nn.Parameter(torch.randn(1, L, d) * 0.02)          # skill queries at c=0
        self.prog_proj = nn.Linear(d_prog, d)
        self.type_emb = nn.Embedding(5, d)                               # robot/eff/evt/prog/fb/skill types
        self.skill_type = nn.Parameter(torch.zeros(1, 1, d))
        enc_layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc_layer, layers)
        self.out_ln = nn.LayerNorm(d)
        self.fb_gate = nn.Parameter(torch.zeros(1, 1, d))                # zero-init: feedback starts silent

    def human_tokens(self, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, 1+1+K+N, d]; computed once per episode and cached."""
        B = enc["z_eff"].shape[0]
        te = self.type_emb.weight
        toks = [self.e_robot.expand(B, -1, -1) + te[0],
                enc["z_eff"][:, None] + te[1],
                enc["Z_evt"] + te[2],
                self.prog_proj(enc["Z_prog"]) + te[3]]
        return torch.cat(toks, dim=1)

    def _mask(self, n_h: int, n_f: int, device) -> torch.Tensor:
        n = n_h + n_f + self.L
        m = torch.ones(n, n, dtype=torch.bool, device=device)          # True = blocked
        m[:n_h, :n_h] = False                                           # human sees human
        for i in range(n_f):                                            # feedback: human + earlier fb (causal)
            m[n_h + i, :n_h] = False
            m[n_h + i, n_h: n_h + i + 1] = False
        m[n_h + n_f:, :] = False                                        # skill tokens see everything
        return m

    def forward(self, S_prev: Optional[torch.Tensor], human_tokens: torch.Tensor,
                fb_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = human_tokens.shape[0]
        skill_in = (self.S_init.expand(B, -1, -1) if S_prev is None else S_prev) + self.skill_type
        parts = [human_tokens]
        n_f = 0
        if fb_tokens is not None and fb_tokens.shape[1] > 0:
            fb_tokens = fb_tokens[:, -self.M_fb:] + self.type_emb.weight[4]
            parts.append(fb_tokens)
            n_f = fb_tokens.shape[1]
        parts.append(skill_in)
        x = torch.cat(parts, dim=1)
        out = self.tr(x, mask=self._mask(human_tokens.shape[1], n_f, x.device))
        S_out = self.out_ln(out[:, -self.L:])
        if S_prev is None:
            return S_out
        return S_prev + self.fb_gate * S_out                            # gated recurrent update


# --------------------------------------------------------------------------- #
# HyperDecoder D
# --------------------------------------------------------------------------- #
class HyperDecoder(nn.Module):
    def __init__(self, specs: List[LayerSpec], d: int, K: int, n_atoms: int = 16, trunk_dim: int = 64,
                 gate_init: float = -4.0, use_manifold: bool = True, use_residual: bool = True,
                 share_res_heads: bool = True):
        super().__init__()
        self.specs, self.d, self.K, self.n_atoms = specs, d, K, n_atoms
        self.use_manifold, self.use_residual, self.share_res_heads = use_manifold, use_residual, share_res_heads
        self.layer_emb = nn.Parameter(torch.randn(len(specs), d) * 0.02)   # tells shared heads which layer
        self.trunk = nn.Sequential(nn.Linear(d, trunk_dim), nn.GELU())
        self.c_head, self.res_head, self.gate_head, self.bias_head = (nn.ModuleDict(), nn.ModuleDict(),
                                                                       nn.ModuleDict(), nn.ModuleDict())
        self._res_key = {}
        for s in specs:
            key = s.name.replace(".", "__")
            ch = nn.Linear(d, n_atoms); nn.init.zeros_(ch.weight); nn.init.zeros_(ch.bias)
            self.c_head[key] = ch
            # residual generator: one head per (in,out,rank) shape class when shared (Text-to-LoRA style),
            # keeps D small even for a 18-layer VLA action expert; per-layer heads otherwise.
            rkey = f"shape_{s.in_f}_{s.out_f}_{s.rank}" if share_res_heads else key
            self._res_key[s.name] = rkey
            if rkey not in self.res_head:
                rh = nn.Linear(trunk_dim, s.rank * (s.in_f + s.out_f))
                nn.init.zeros_(rh.weight); nn.init.zeros_(rh.bias)      # zero-init residual (HypoGen)
                self.res_head[rkey] = rh
            gh = nn.Linear(d, 1); nn.init.zeros_(gh.weight); nn.init.constant_(gh.bias, gate_init)
            self.gate_head[key] = gh
            if s.gen_bias:      # (global + K phase) AdaLN bias deltas, generated from the low-dim trunk
                bh = nn.Linear(trunk_dim, s.out_f * (1 + K)); nn.init.zeros_(bh.weight); nn.init.zeros_(bh.bias)
                self.bias_head[key] = bh
        self.layer_energy = nn.Linear(d, 1)                              # per-block gate for L_layer

    # ---- per-task decode --------------------------------------------- #
    def forward(self, S: torch.Tensor, alpha: Optional[torch.Tensor] = None,
                inv: Optional[torch.Tensor] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """S [U,L,d]; alpha [B,K] (per sample) ; inv [B] -> P with per-sample tensors.
        If inv is None, U == B and alpha is per task."""
        U = S.shape[0]
        P = {}
        for j, s in enumerate(self.specs):
            key = s.name.replace(".", "__")
            m = S[:, s.block] + self.layer_emb[j]
            tr = self.trunk(m)                                           # [U, trunk_dim]
            entry = {}
            if self.use_manifold:
                entry["c"] = self.c_head[key](m)                         # [U, M]
            if self.use_residual:
                ab = self.res_head[self._res_key[s.name]](tr)
                A = ab[:, : s.rank * s.in_f].reshape(U, s.rank, s.in_f)
                B = ab[:, s.rank * s.in_f:].reshape(U, s.out_f, s.rank)
                entry["A"], entry["B"] = A, B
                entry["g"] = torch.sigmoid(self.gate_head[key](m))       # [U, 1]
            if s.gen_bias:
                bias = self.bias_head[key](tr).view(U, 1 + self.K, s.out_f)
                entry["_bias_all"] = bias                                 # mixed with alpha below
            P[s.name] = entry
        if inv is not None:
            P = {n: {k: v[inv] for k, v in e.items()} for n, e in P.items()}
        # phase mixing of AdaLN biases: db = b^G + sum_k alpha_k b^k
        for n, e in P.items():
            if "_bias_all" in e:
                bias_all = e.pop("_bias_all")
                db = bias_all[:, 0]
                if alpha is not None:
                    db = db + torch.einsum("bk,bkd->bd", alpha, bias_all[:, 1:])
                e["db"] = db
        return P

    # ---- dense deltas for the parameter-space teacher losses ---------- #
    def dense_delta_layer(self, P_task: Dict[str, Dict[str, torch.Tensor]], name: str, target) -> Optional[torch.Tensor]:
        """Dense generated delta of ONE layer, [U,out,in]. target exposes .layer(name) -> LoRALinear."""
        e = P_task[name]
        layer = target.layer(name)
        dW = None
        if "c" in e:
            dW = layer.manifold_dense(e["c"])
        if "A" in e:
            r = LoRALinear.lora_dense(e["A"], e["B"], e["g"], layer.spec.rank)
            dW = r if dW is None else dW + r
        return dW

    def dense_deltas(self, P_task, target) -> Dict[str, torch.Tensor]:
        """All layers at once (fine for the scratch DiT; prefer the streaming path for a VLA)."""
        return {s.name: self.dense_delta_layer(P_task, s.name, target) for s in self.specs}

    def block_gates(self, S: torch.Tensor) -> torch.Tensor:
        return self.layer_energy(S).squeeze(-1)                          # [U, L] logits

    @torch.no_grad()
    def init_atoms_from_bank(self, target, bank):
        """SVD-initialise the atoms of every layer from the expert bank (streamed layer by layer)."""
        for s in self.specs:
            layer = target.layer(s.name)
            D = bank.dense_layer(s.name)                                  # [N, out, in]
            N = D.shape[0]
            flat = D.reshape(N, -1)
            try:
                Uu, Ss, Vh = torch.linalg.svd(flat, full_matrices=False)     # Vh: [min(N,oi), out*in]
            except RuntimeError:
                continue
            M = min(self.n_atoms, Vh.shape[0])
            for m in range(M):
                comp = (Vh[m] * Ss[m] / math.sqrt(N)).reshape(s.out_f, s.in_f)
                Pu, Ps, Pv = torch.linalg.svd(comp, full_matrices=False)     # rank-r truncation
                r = s.rank
                layer.atom_P[m] = (Pu[:, :r] * Ps[:r].sqrt()).t()             # [r, out]
                layer.atom_Q[m] = Pv[:r].t() * Ps[:r].sqrt()                  # [in, r]


# --------------------------------------------------------------------------- #
# Latent TTT preconditioner P_phi  (HyLaP's global pseudo-gradient block, re-purposed)
# --------------------------------------------------------------------------- #
class LatentTTT(nn.Module):
    def __init__(self, d: int, L: int, nhead: int = 4, layers: int = 1, lr_max: float = 0.05):
        super().__init__()
        self.L, self.lr_max = L, lr_max
        self.grad_ln = nn.LayerNorm(d)
        self.type_emb = nn.Embedding(3, d)
        enc_layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc_layer, layers)
        self.delta = nn.Linear(d, d); nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        self.lr_head = nn.Linear(d, 1); nn.init.zeros_(self.lr_head.weight); nn.init.constant_(self.lr_head.bias, -2.0)
        # heads that make L_SSL depend on S
        self.vid_mod = nn.Linear(d, 2 * d); nn.init.zeros_(self.vid_mod.weight); nn.init.zeros_(self.vid_mod.bias)
        self.mem_dec = nn.Linear(d, d)

    def ssl_loss(self, S: torch.Tensor, enc: Dict[str, torch.Tensor], E) -> torch.Tensor:
        """Human-video self-supervision that depends on S: modulated future prediction + memory recon."""
        pooled = S.mean(1)
        shift, scale = self.vid_mod(pooled).chunk(2, -1)
        l_fut = E.future_loss(enc["frame_g"], mod=(shift, scale))
        l_mem = F.mse_loss(self.mem_dec(pooled), enc["z_eff"].detach())
        return l_fut + l_mem

    def step(self, S: torch.Tensor, grad: torch.Tensor, human_tokens: torch.Tensor) -> torch.Tensor:
        te = self.type_emb.weight
        g = self.grad_ln(grad) + te[0]
        x = torch.cat([g, S + te[1], human_tokens + te[2]], dim=1)
        out = self.tr(x)[:, self.L: 2 * self.L]
        dS = self.delta(out)                                             # zero-init -> no-op at start
        lr = torch.sigmoid(self.lr_head(out)) * self.lr_max                # per-block learned lr
        return S - lr * (grad + dS)


# --------------------------------------------------------------------------- #
# wrapper
# --------------------------------------------------------------------------- #
class HyperCompiler(nn.Module):
    def __init__(self, specs: List[LayerSpec], n_blocks: int, d: int, K: int, d_prog: int, h_dim: int,
                 act_dim: int, horizon: int, n_atoms: int = 16, T_layers: int = 4, nhead: int = 4, M_fb: int = 8,
                 use_manifold: bool = True, use_residual: bool = True, use_prev_action: bool = False,
                 share_res_heads: bool = True, trunk_dim: int = 64):
        super().__init__()
        self.L = n_blocks
        self.T = ContextTransformer(d, self.L, K, d_prog, T_layers, nhead, M_fb)
        self.D = HyperDecoder(specs, d, K, n_atoms, trunk_dim, use_manifold=use_manifold,
                              use_residual=use_residual, share_res_heads=share_res_heads)
        self.ttt = LatentTTT(d, self.L, nhead)
        self.fb = FeedbackTokenizer(h_dim, d_prog, act_dim, horizon, d, use_prev_action)

    def compile(self, human_tokens: torch.Tensor) -> torch.Tensor:
        return self.T(None, human_tokens, None)

    def step(self, S: torch.Tensor, human_tokens: torch.Tensor, fb_tokens: torch.Tensor) -> torch.Tensor:
        return self.T(S, human_tokens, fb_tokens)

    def run_ttt(self, S: torch.Tensor, enc: Dict[str, torch.Tensor], human_tokens: torch.Tensor, E,
                K: int, create_graph: bool = False) -> torch.Tensor:
        for _ in range(K):
            with torch.enable_grad():
                S_ = S if S.requires_grad else S.detach().requires_grad_(True)
                loss = self.ttt.ssl_loss(S_, enc, E)
                grad, = torch.autograd.grad(loss, S_, create_graph=create_graph)
            if create_graph:                                   # meta-training: outer loss flows through the update
                S = self.ttt.step(S_, grad, human_tokens)
            else:                                              # deployment: first-order, no graph kept
                with torch.no_grad():
                    S = self.ttt.step(S_.detach(), grad.detach(), human_tokens)
        return S

    def decode(self, S: torch.Tensor, alpha: Optional[torch.Tensor], inv: Optional[torch.Tensor]):
        return self.D(S, alpha, inv)
