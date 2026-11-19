"""Flow-matching action head (the only head used by the main model).

    train_pair(x0, noise, u) -> (x_t, target, t_in)      u ~ U[0,1)
    sample(fn, shape, device, noise)  -> x0 (Euler, `steps` steps)
    x0_from_v(x_t, u, v)              -> one-step x0 estimate (differentiable; used by the
                                         world-model "imagined feedback" path)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class FlowMatching:
    name = "fm"

    def __init__(self, steps: int = 10, t_scale: float = 1000.0):
        self.steps, self.t_scale = steps, t_scale
        self.loss_fn = F.mse_loss

    def train_pair(self, x0: torch.Tensor, noise: torch.Tensor, u: torch.Tensor):
        tt = u.view(-1, *([1] * (x0.dim() - 1)))
        x_t = (1.0 - tt) * x0 + tt * noise
        return x_t, noise - x0, u * self.t_scale

    def x0_from_v(self, x_t: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x_t = (1-u) x0 + u eps,  v = eps - x0  =>  x0 = x_t - u v
        tt = u.view(-1, *([1] * (x_t.dim() - 1)))
        return x_t - tt * v

    @torch.no_grad()
    def sample(self, fn, shape, device, noise: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.randn(shape, device=device) if noise is None else noise.clone()
        dt = 1.0 / self.steps
        for i in reversed(range(self.steps)):
            t = torch.full((shape[0],), (i + 1) * dt, device=device)
            x = x - dt * fn(x, t * self.t_scale)
        return x
