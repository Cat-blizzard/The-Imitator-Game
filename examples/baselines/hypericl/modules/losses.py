"""All auxiliary losses of HyperICL (formulas in DESIGN_H2R_HYPERNET.md §5)."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# progress alignment
# --------------------------------------------------------------------------- #
def _pdist(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """u [B,N,d], v [B,M,d] -> squared distances [B,N,M]."""
    return (u[:, :, None] - v[:, None]).pow(2).sum(-1)


def tcc_loss(u: torch.Tensor, v: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Temporal cycle consistency (cycle-back regression) between two sequences on the shared
    progress manifold. u [B,N,d], v [B,M,d]. Returns normalised MSE of cycled-back index."""
    B, N, _ = u.shape
    d_uv = _pdist(u, v)                                  # [B,N,M]
    a = torch.softmax(-d_uv / temp, dim=-1)              # soft NN in v
    v_t = torch.einsum("bnm,bmd->bnd", a, v)             # [B,N,d]
    d_back = _pdist(v_t, u)                              # [B,N,N]
    b = torch.softmax(-d_back / temp, dim=-1)
    idx = torch.arange(N, device=u.device, dtype=u.dtype)
    mu = (b * idx).sum(-1)                               # [B,N]
    return ((mu - idx[None]) / N).pow(2).mean()


def soft_dtw(u: torch.Tensor, v: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """Soft-DTW (Cuturi & Blondel) with a plain O(NM) loop; fine for N,M <= 64."""
    D = _pdist(u, v)                                     # [B,N,M]
    B, N, M = D.shape
    inf = torch.full((B,), float("inf"), device=D.device, dtype=D.dtype)
    R = [[inf] * (M + 1) for _ in range(N + 1)]
    R[0][0] = torch.zeros(B, device=D.device, dtype=D.dtype)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            prev = torch.stack([R[i - 1][j - 1], R[i - 1][j], R[i][j - 1]], dim=-1)
            R[i][j] = D[:, i - 1, j - 1] - gamma * torch.logsumexp(-prev / gamma, dim=-1)
    return (R[N][M] / (N + M)).mean()


def order_loss(p: torch.Tensor, margin: float = 0.1, n_triplets: int = 32) -> torch.Tensor:
    """Temporal order margin on a progress sequence p [B,N,d]: d(i,j) + m < d(i,k) for i<j<k."""
    B, N, _ = p.shape
    if N < 3:
        return p.new_zeros(())
    i = torch.randint(0, N - 2, (n_triplets,), device=p.device)
    j = i + 1 + (torch.rand(n_triplets, device=p.device) * (N - 2 - i)).long()
    k = j + 1 + (torch.rand(n_triplets, device=p.device) * (N - 1 - j)).long()
    pi, pj, pk = p[:, i], p[:, j], p[:, k]
    d_ij = (pi - pj).pow(2).sum(-1)
    d_ik = (pi - pk).pow(2).sum(-1)
    return F.relu(margin + d_ij - d_ik).mean()


def robot_progress_loss(p_R: torch.Tensor, Z_prog: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    """p_R [B,d]; Z_prog [B,N,d] (human, same task); progress [B] in [0,1] -> pull p_R to the
    human progress token at the same normalised time (linear interpolation)."""
    N = Z_prog.shape[1]
    pos = progress.clamp(0, 1) * (N - 1)
    lo = pos.floor().long().clamp(max=N - 1)
    hi = (lo + 1).clamp(max=N - 1)
    w = (pos - lo.float())[:, None]
    tgt = (1 - w) * Z_prog[torch.arange(len(lo)), lo] + w * Z_prog[torch.arange(len(hi)), hi]
    return (p_R - F.normalize(tgt, dim=-1).detach()).pow(2).sum(-1).mean()


# --------------------------------------------------------------------------- #
# parameter-space teacher losses (on dense products dW = B A, gauge-free)
# --------------------------------------------------------------------------- #
def streaming_param_stats(names: List[str], gen_fn, tea_fn=None, base_norms: Optional[Dict[str, torch.Tensor]] = None,
                          eps: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Layer-by-layer accumulation (never materialises all dense deltas at once — a VLA expert has
    ~70 layers of up to 1M numbers each). gen_fn(name) -> [U,out,in] generated delta (with grad);
    tea_fn(name) -> [U,out,in] teacher delta (no grad) or None.
    Returns: rel, cos, mag (teacher losses), e_gen/e_tea [U, n_layers] energies, dw_reg (relative
    delta norm), dw2 [U,U] pairwise squared distances (for the geometry loss)."""
    rel, cos, mag, e_gen, e_tea, dw_reg, used_tea = [], [], [], [], [], [], []
    dw2 = None
    for n in names:
        dW = gen_fn(n)
        if dW is None:
            continue
        g_f = dW.flatten(1)
        e_gen.append(g_f.norm(dim=-1))
        if base_norms is not None and n in base_norms:
            dw_reg.append(g_f.pow(2).sum(-1).mean() / (base_norms[n] + eps))
        d2 = torch.cdist(g_f, g_f).pow(2)
        dw2 = d2 if dw2 is None else dw2 + d2
        if tea_fn is not None:
            t = tea_fn(n)
            if t is not None:
                t_f = t.detach().flatten(1)
                rel.append(((g_f - t_f).pow(2).sum(-1) / (t_f.pow(2).sum(-1) + eps)).mean())
                cos.append((1 - F.cosine_similarity(g_f, t_f, dim=-1, eps=eps)).mean())
                mag.append((torch.log(g_f.norm(dim=-1) + eps) - torch.log(t_f.norm(dim=-1) + eps)).abs().mean())
                e_tea.append(t_f.norm(dim=-1))
                used_tea.append(n)
        del dW, g_f
    out = {"dw2": dw2, "used_tea": used_tea}
    out["rel"] = torch.stack(rel).mean() if rel else None
    out["cos"] = torch.stack(cos).mean() if cos else None
    out["mag"] = torch.stack(mag).mean() if mag else None
    out["e_gen"] = torch.stack(e_gen, -1) if e_gen else None
    out["e_tea"] = torch.stack(e_tea, -1) if e_tea else None
    out["dw_reg"] = torch.stack(dw_reg).mean() if dw_reg else None
    return out


def geometry_loss_from_dw2(z: torch.Tensor, dw2: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    U = z.shape[0]
    if dw2 is None or U < 3:
        return None
    dz = torch.cdist(z, z)
    dw = dw2.clamp_min(0).sqrt()
    iu = torch.triu_indices(U, U, 1, device=z.device)
    dz, dw = dz[iu[0], iu[1]], dw[iu[0], iu[1]]
    dz = (dz - dz.mean()) / (dz.std() + 1e-6)
    dw = (dw - dw.mean()) / (dw.std() + 1e-6)
    return (dz - dw).abs().mean()


def param_losses(gen: Dict[str, torch.Tensor], teacher: Dict[str, torch.Tensor], eps: float = 1e-6):
    """gen/teacher: name -> [U,out,in]. Returns (rel_frob, 1-cos, log-mag, layer energies)."""
    rel, cos, mag, e_gen, e_tea = [], [], [], [], []
    for n, dW in gen.items():
        if dW is None or n not in teacher:
            continue
        t = teacher[n].detach()
        g_f, t_f = dW.flatten(1), t.flatten(1)
        t_n = t_f.pow(2).sum(-1)
        rel.append(((g_f - t_f).pow(2).sum(-1) / (t_n + eps)).mean())
        cos.append((1 - F.cosine_similarity(g_f, t_f, dim=-1, eps=eps)).mean())
        mag.append((torch.log(g_f.norm(dim=-1) + eps) - torch.log(t_f.norm(dim=-1) + eps)).abs().mean())
        e_gen.append(g_f.norm(dim=-1)); e_tea.append(t_f.norm(dim=-1))
    if not rel:
        z = torch.zeros(())
        return z, z, z, None, None
    return (torch.stack(rel).mean(), torch.stack(cos).mean(), torch.stack(mag).mean(),
            torch.stack(e_gen, -1), torch.stack(e_tea, -1))        # energies [U, n_layers]


def layer_energy_kl(gate_logits: torch.Tensor, teacher_energy: torch.Tensor, block_of_layer: List[int]) -> torch.Tensor:
    """KL( s* || softmax(gates) ) where s* is the teacher's per-block energy share."""
    U, L = gate_logits.shape
    s = torch.zeros(U, L, device=gate_logits.device)
    for j, b in enumerate(block_of_layer):
        s[:, b] += teacher_energy[:, j]
    s = s / s.sum(-1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(F.log_softmax(gate_logits, -1), s.detach(), reduction="batchmean")


def delta_norm_reg(gen: Dict[str, torch.Tensor], base_norms: Dict[str, torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    vals = [dW.flatten(1).pow(2).sum(-1).mean() / (base_norms[n] + eps) for n, dW in gen.items() if dW is not None]
    return torch.stack(vals).mean() if vals else torch.zeros(())


def atom_ortho_reg(layers) -> torch.Tensor:
    vals = []
    for layer in layers:
        Q = layer.atom_Q                                   # [M, in, r]
        Qf = F.normalize(Q.flatten(1), dim=-1)             # [M, in*r]
        G = Qf @ Qf.t()
        vals.append((G - torch.eye(G.shape[0], device=G.device)).pow(2).mean())
    return torch.stack(vals).mean() if vals else torch.zeros(())


def skill_smooth_reg(S: torch.Tensor) -> torch.Tensor:
    return (S[:, 1:] - S[:, :-1]).pow(2).mean()


def geometry_loss(z: torch.Tensor, gen: Dict[str, torch.Tensor]) -> torch.Tensor:
    """|d_Z(i,j) - d_W(i,j)| over task pairs, both distance matrices standardised (U>=3)."""
    U = z.shape[0]
    if U < 3:
        return z.new_zeros(())
    dz = torch.cdist(z, z)
    dw2 = None
    for dW in gen.values():                       # accumulate per layer (no giant concatenation)
        if dW is None:
            continue
        f = dW.flatten(1)
        d2 = torch.cdist(f, f).pow(2)
        dw2 = d2 if dw2 is None else dw2 + d2
    if dw2 is None:
        return z.new_zeros(())
    dw = dw2.clamp_min(0).sqrt()
    iu = torch.triu_indices(U, U, 1, device=z.device)
    dz, dw = dz[iu[0], iu[1]], dw[iu[0], iu[1]]
    dz = (dz - dz.mean()) / (dz.std() + 1e-6)
    dw = (dw - dw.mean()) / (dw.std() + 1e-6)
    return (dz - dw).abs().mean()
