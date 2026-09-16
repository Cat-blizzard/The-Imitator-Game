"""Stage B: robot expert bank.

One LoRA (+ AdaLN bias) set per training task, trained jointly with the base policy frozen
(`P = bank(task_idx)`), no per-task dataset filtering needed. After training it provides:

    bank.dense()            name -> [N_task, out, in]   parameter teacher  (DESIGN §5.2)
    bank(task_idx)          P dict                       function teacher   (DESIGN §5.3, L_func)
    bank.energies()         [N_task, n_layers]           layer-energy teacher (L_layer)
    HyperDecoder.init_atoms_from_bank(dit, bank.dense())  manifold dictionary init (§4.4)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .modules.lora_dit import LayerSpec, LoRADiT


class TaskLoRABank(nn.Module):
    def __init__(self, specs: List[LayerSpec], n_tasks: int):
        super().__init__()
        self.specs, self.n_tasks = specs, n_tasks
        self.A, self.B, self.db = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        for s in specs:
            key = s.name.replace(".", "__")
            A = torch.empty(n_tasks, s.rank, s.in_f)
            nn.init.kaiming_uniform_(A.view(-1, s.in_f), a=math.sqrt(5))
            self.A[key] = nn.Parameter(A)
            self.B[key] = nn.Parameter(torch.zeros(n_tasks, s.out_f, s.rank))
            if s.gen_bias:
                self.db[key] = nn.Parameter(torch.zeros(n_tasks, s.out_f))

    def forward(self, task_idx: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        P = {}
        for s in self.specs:
            key = s.name.replace(".", "__")
            e = {"A": self.A[key][task_idx], "B": self.B[key][task_idx]}
            if key in self.db:
                e["db"] = self.db[key][task_idx]
            P[s.name] = e
        return P

    def _spec(self, name: str) -> LayerSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def dense_layer(self, name: str, task_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dense expert delta of ONE layer: [N or len(task_idx), out, in] (detached)."""
        s = self._spec(name)
        key = name.replace(".", "__")
        A, B = self.A[key], self.B[key]
        if task_idx is not None:
            A, B = A[task_idx], B[task_idx]
        return (torch.einsum("nor,nri->noi", B, A) / math.sqrt(s.rank)).detach()

    @torch.no_grad()
    def dense(self) -> Dict[str, torch.Tensor]:
        return {s.name: self.dense_layer(s.name) for s in self.specs}

    @torch.no_grad()
    def energies(self) -> torch.Tensor:
        return torch.stack([self.dense_layer(s.name).flatten(1).norm(dim=-1) for s in self.specs], dim=-1)

    @torch.no_grad()
    def explained_variance(self, n_atoms: int) -> Dict[str, float]:
        """Diagnostic for the M0 gate: fraction of variance captured by the first n_atoms SVs per layer."""
        out = {}
        for s in self.specs:
            name, D = s.name, self.dense_layer(s.name)
            flat = D.reshape(D.shape[0], -1)
            try:
                S = torch.linalg.svdvals(flat)
            except RuntimeError:
                continue
            out[name] = float((S[:n_atoms] ** 2).sum() / (S ** 2).sum().clamp_min(1e-12))
        return out
