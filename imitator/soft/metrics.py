"""Cloth metrics shared by the envs and the trace recorder (torch, batched over worlds).

footprint_area: convex-hull area of the top-down projection, computed on the GPU as the area of the polygon circumscribed
by the support lines in M equally spaced directions (exact for polygons whose edge normals are among the directions,
otherwise a slight over-estimate; measured vs scipy: < 0.1 % on towel states, < 0.2 % on random clouds for M = 720,
tests/imitator/test_soft.py). Used for the env's per-step `info`; the contract verdict uses veritask's scipy ConvexHull on
the recorded vertices (contract_eval.py).
"""
from __future__ import annotations

import math

import torch

_DIRS: dict = {}


def _dirs(m: int, device) -> torch.Tensor:
    key = (m, str(device))
    if key not in _DIRS:
        th = torch.arange(m, device=device, dtype=torch.float32) * (2 * math.pi / m)
        _DIRS[key] = torch.stack([torch.cos(th), torch.sin(th)], 1)
    return _DIRS[key]


def footprint_area(xy: torch.Tensor, m: int = 720) -> torch.Tensor:
    """xy: (N, V, 2) -> (N,) circumscribed-polygon area of the convex hull."""
    d = _dirs(m, xy.device)                              # (M, 2)
    h = (xy.float() @ d.T).amax(dim=1)                   # (N, M) support function
    h1 = torch.roll(h, -1, dims=1)
    d1 = torch.roll(d, -1, dims=0)
    s = math.sin(2 * math.pi / m)
    x = (h * d1[:, 1] - h1 * d[:, 1]) / s                # intersection of support lines k and k+1
    y = (h1 * d[:, 0] - h * d1[:, 0]) / s
    return 0.5 * (x * torch.roll(y, -1, 1) - torch.roll(x, -1, 1) * y).sum(1).abs()


def corner_distance(verts: torch.Tensor, a: int, c: int) -> torch.Tensor:
    return torch.linalg.norm(verts[:, a, :2] - verts[:, c, :2], dim=-1)


def frac_above(verts: torch.Tensor, height: float = 0.05, ground_z: float = 0.0) -> torch.Tensor:
    return ((verts[..., 2] - ground_z) > height).float().mean(-1)


def speed_quantile(vel: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    return torch.quantile(torch.linalg.norm(vel.float(), dim=-1), q, dim=1)
