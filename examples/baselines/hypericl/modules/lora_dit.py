"""FM-DiT target policy whose Linear layers accept per-sample parameter deltas.

Every generated layer computes

    y = W0 x + b0                                   (slow weights, theta_0)
      + sum_m c_m P_m Q_m^T x                       (manifold part: M rank-r atoms, per-sample coeffs c)
      + g * B A x / sqrt(r)                         (zero-init residual LoRA, per-sample A, B, gate g)
      + db                                          (per-sample bias delta; AdaLN layers only)

Delta tensors arrive in a dict `P[name] = {"c": [B,M], "A": [B,r,in], "B": [B,out,r], "g": [B,1], "db": [B,out]}`
(any entry may be missing / None). With `P=None` the network is exactly the base policy.
Atoms (P_m, Q_m) live in the layer (initialised from the expert bank by `HyperDecoder`).
No `functional_call` / `vmap`: per-sample deltas are plain einsums, so SDPA is safe.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Active-delta context: lets LoRALinear layers injected into a *pretrained* policy
# (whose forward signature we cannot change) pick up per-sample deltas.
# --------------------------------------------------------------------------- #
_ACTIVE_P: Dict[str, object] = {"P": None}


@contextlib.contextmanager
def lora_context(P: Optional[Dict[str, Dict[str, torch.Tensor]]]):
    prev = _ACTIVE_P["P"]
    _ACTIVE_P["P"] = P
    try:
        yield
    finally:
        _ACTIVE_P["P"] = prev


@dataclass
class LayerSpec:
    name: str
    in_f: int
    out_f: int
    rank: int
    block: int          # which skill token drives it
    gen_bias: bool      # AdaLN layers: also generate a (global + per-phase) bias delta


class LoRALinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, name: str, rank: int, block: int,
                 gen_bias: bool = False, n_atoms: int = 16, base: Optional[nn.Linear] = None):
        super().__init__()
        self.base = base if base is not None else nn.Linear(in_f, out_f)
        self.spec = LayerSpec(name, in_f, out_f, rank, block, gen_bias)
        self.n_atoms = n_atoms
        # manifold atoms P_m [M, r, out], Q_m [M, in, r]  (small random; overwritten from expert SVD)
        self.atom_P = nn.Parameter(torch.randn(n_atoms, rank, out_f) * 0.02)
        self.atom_Q = nn.Parameter(torch.randn(n_atoms, in_f, rank) * 0.02)

    @classmethod
    def from_linear(cls, linear: nn.Linear, name: str, rank: int, block: int,
                    gen_bias: bool = False, n_atoms: int = 16) -> "LoRALinear":
        """Wrap an existing (pretrained, frozen) nn.Linear in place."""
        # atoms stay small-random: the coefficient heads are zero-initialised, so the layer is exactly
        # the base policy at start while gradients still flow into both c and the atoms.
        return cls(linear.in_features, linear.out_features, name, rank, block, gen_bias, n_atoms, base=linear)

    @torch.no_grad()
    def merge_delta_into_base(self, dW: torch.Tensor, db: Optional[torch.Tensor] = None):
        """Fold a dense delta [out,in] (and bias [out]) into the slow weights (stage A of the VLA version)."""
        self.base.weight.add_(dW.to(self.base.weight.dtype))
        if db is not None and self.base.bias is not None:
            self.base.bias.add_(db.to(self.base.bias.dtype))

    # ---- dense views used by the parameter-space losses ---------------- #
    def manifold_dense(self, c: torch.Tensor) -> torch.Tensor:
        """c [U, M] -> dW [U, out, in]."""
        atoms = torch.einsum("mro,mir->moi", self.atom_P, self.atom_Q)        # [M, out, in]
        return torch.einsum("um,moi->uoi", c, atoms)

    @staticmethod
    def lora_dense(A: torch.Tensor, B: torch.Tensor, g: torch.Tensor, rank: int) -> torch.Tensor:
        return g[:, :, None] * torch.einsum("uor,uri->uoi", B, A) / math.sqrt(rank)

    # ---- forward -------------------------------------------------------- #
    def forward(self, x: torch.Tensor, P: Optional[Dict[str, Dict[str, torch.Tensor]]] = None) -> torch.Tensor:
        y = self.base(x)
        if P is None:
            P = _ACTIVE_P["P"]
        if P is None or self.spec.name not in P:
            return y
        p = P[self.spec.name]
        squeeze = x.dim() == 2
        # delta path runs in fp32 (atoms / generated deltas are fp32) and is cast back to the base dtype,
        # so bf16 pretrained experts and the fp32 scratch DiT share this code.
        xx = (x.unsqueeze(1) if squeeze else x).float()                       # [B, T, in]
        delta = None
        c = p.get("c")
        if c is not None:
            M, r = self.atom_Q.shape[0], self.atom_Q.shape[2]
            h = xx @ self.atom_Q.permute(1, 0, 2).reshape(self.spec.in_f, M * r)   # [B,T,M*r]
            h = h.view(xx.shape[0], xx.shape[1], M, r) * c.float()[:, None, :, None]
            delta = torch.einsum("btmr,mro->bto", h, self.atom_P)
        A, Bm = p.get("A"), p.get("B")
        if A is not None and Bm is not None:
            g = p.get("g")
            h = torch.einsum("bti,bri->btr", xx, A.float())
            h = torch.einsum("btr,bor->bto", h, Bm.float()) / math.sqrt(self.spec.rank)
            if g is not None:
                h = h * g.float()[:, None, :]
            delta = h if delta is None else delta + h
        db = p.get("db")
        if db is not None:
            db = db.float()[:, None, :]
            delta = db if delta is None else delta + db
        if delta is None:
            return y
        delta = delta.squeeze(1) if squeeze else delta
        return y + delta.to(y.dtype)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
        e = t[:, None].float() * f[None, :]
        return torch.cat((e.sin(), e.cos()), dim=-1)


def _mod(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_ratio, rank, ada_rank, block_idx, n_atoms):
        super().__init__()
        self.dim, self.nhead = dim, nhead
        p = f"b{block_idx}"
        mk = lambda name, i, o, r, gb=False: LoRALinear(i, o, f"{p}.{name}", r, block_idx, gb, n_atoms)
        self.q, self.k, self.v, self.o = (mk("q", dim, dim, rank), mk("k", dim, dim, rank),
                                          mk("v", dim, dim, rank), mk("o", dim, dim, rank))
        self.fc1, self.fc2 = mk("fc1", dim, dim * mlp_ratio, rank), mk("fc2", dim * mlp_ratio, dim, rank)
        self.ada = mk("ada", dim, 6 * dim, ada_rank, gb=True)

    def forward(self, h, cs, P):
        B, T, D = h.shape
        Hh, hd = self.nhead, D // self.nhead
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(cs, P).chunk(6, -1)
        y = _mod(F.layer_norm(h, (D,)), sh1, sc1)
        sp = lambda z: z.reshape(B, T, Hh, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(self.q(y, P)), sp(self.k(y, P)), sp(self.v(y, P)))
        o = o.transpose(1, 2).reshape(B, T, D)
        h = h + g1.unsqueeze(1) * self.o(o, P)
        y = _mod(F.layer_norm(h, (D,)), sh2, sc2)
        y = self.fc2(F.gelu(self.fc1(y, P)), P)
        return h + g2.unsqueeze(1) * y


class LoRADiT(nn.Module):
    """Blocks (skill tokens): 0 = input, 1..depth = DiT blocks, depth+1 = final."""

    def __init__(self, act_dim: int, cond_dim: int, horizon: int = 16, dim: int = 256, depth: int = 6,
                 nhead: int = 4, mlp_ratio: int = 4, dsed: int = 64, rank: int = 8, ada_rank: int = 16,
                 n_atoms: int = 16):
        super().__init__()
        self.act_dim, self.dim, self.depth, self.horizon = act_dim, dim, depth, horizon
        self.time_pos = SinusoidalPosEmb(dsed)
        self.t_fc0, self.t_fc1 = nn.Linear(dsed, dim), nn.Linear(dim, dim)          # structural, not generated
        self.pos = nn.Parameter(torch.randn(1, horizon, dim) * 0.02)
        self.x_embed = LoRALinear(act_dim, dim, "in.x_embed", rank, 0, False, n_atoms)
        self.c_proj = LoRALinear(dim + cond_dim, dim, "in.c_proj", rank, 0, False, n_atoms)
        self.blocks = nn.ModuleList([DiTBlock(dim, nhead, mlp_ratio, rank, ada_rank, i + 1, n_atoms)
                                     for i in range(depth)])
        self.f_ada = LoRALinear(dim, 2 * dim, "fin.ada", ada_rank, depth + 1, True, n_atoms)
        self.f_out = LoRALinear(dim, act_dim, "fin.out", ada_rank, depth + 1, False, n_atoms)
        self.n_blocks = depth + 2

    # ---- introspection used by the hypernet ---------------------------- #
    def lora_layers(self) -> List[LoRALinear]:
        return [m for m in self.modules() if isinstance(m, LoRALinear)]

    def layer_specs(self) -> List[LayerSpec]:
        return [m.spec for m in self.lora_layers()]

    def layer(self, name: str) -> LoRALinear:
        for m in self.lora_layers():
            if m.spec.name == name:
                return m
        raise KeyError(name)

    # ---- forward -------------------------------------------------------- #
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                P: Optional[Dict] = None) -> torch.Tensor:
        """x [B,T,act], t [B] (scaled), cond [B,cond_dim] -> v [B,T,act]."""
        B, T, _ = x.shape
        c = self.t_fc1(F.silu(self.t_fc0(self.time_pos(t))))
        c = self.c_proj(torch.cat([c, cond], dim=-1), P)
        cs = F.silu(c)
        h = self.x_embed(x, P) + self.pos[:, :T]
        for blk in self.blocks:
            h = blk(h, cs, P)
        sh, sc = self.f_ada(cs, P).chunk(2, -1)
        h = _mod(F.layer_norm(h, (self.dim,)), sh, sc)
        return self.f_out(h, P)


def expand_P(P: Dict[str, Dict[str, torch.Tensor]], inv: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
    """Per-task [U,...] deltas -> per-sample [B,...] via inv[B]."""
    return {name: {k: (v[inv] if v is not None else None) for k, v in d.items()} for name, d in P.items()}
