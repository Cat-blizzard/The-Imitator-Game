"""Evaluation semantics: how a per-step success signal is turned into an episode verdict.

The same predicate gives different scores under different semantics. Our audits found that several widely used
benchmarks credit an episode if the predicate holds at *any single step* (and stop the episode there), so a
transient state (object held inside a region, momentary contact) counts as task completion. This module makes the
semantics explicit so one recorded rollout can be rescored under several of them.

Modes (strings accepted by `parse_mode`):
  any          true if the signal is true at any step                  (LIBERO, RoboTwin 2.0 official scoring)
  final        true if the signal is true at the last step
  window:K     true if the signal is true at each of the last K steps   (settled end state)
  persist:K    true if the signal is true for K consecutive steps at some point (sustained, not transient)
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

# Official scoring semantics observed in source (see research/results/e5/E5_REPORT.md).
BENCHMARK_SEMANTICS = {
    "libero": dict(mode="any", granularity="control step (20 Hz)",
                   evidence="LIBERO@8f1084e lifelong/metric.py:147 `dones[k] = dones[k] or done[k]`; "
                            "openpi / OpenVLA-OFT LIBERO scripts break at the first done"),
    "robotwin2": dict(mode="any", granularity="physics substep",
                      evidence="RoboTwin@ea8b211 envs/_base_task.py:1727-1731 check_success() after every scene.step(); "
                               "first True sets eval_success and take_action returns"),
}


def as_bool_array(trace: Sequence) -> np.ndarray:
    a = np.asarray(trace)
    if a.dtype != bool:
        a = a.astype(float) > 0.5
    return a.reshape(-1)


def first_true(trace: Sequence) -> int | None:
    a = as_bool_array(trace)
    idx = np.flatnonzero(a)
    return int(idx[0]) if idx.size else None


def longest_run(trace: Sequence) -> int:
    a = as_bool_array(trace); best = cur = 0
    for v in a:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return best


def any_step(trace) -> bool:
    return bool(as_bool_array(trace).any())


def final(trace) -> bool:
    a = as_bool_array(trace)
    return bool(a[-1]) if a.size else False


def final_window(trace, k: int) -> bool:
    a = as_bool_array(trace)
    return bool(a.size >= k and a[-k:].all())


def persist(trace, k: int) -> bool:
    return longest_run(trace) >= k


def parse_mode(mode: str):
    """'any' | 'final' | 'window:K' | 'persist:K'  ->  callable(trace) -> bool"""
    name, _, arg = mode.partition(":")
    if name == "any":
        return any_step
    if name == "final":
        return final
    if name in ("window", "persist"):
        if not arg.isdigit() or int(arg) < 1:
            raise ValueError(f"mode '{mode}' needs a positive integer, e.g. {name}:10")
        k = int(arg)
        return (lambda tr: final_window(tr, k)) if name == "window" else (lambda tr: persist(tr, k))
    raise ValueError(f"unknown semantics mode '{mode}'")


def rescore(trace, modes: Iterable[str] = ("any", "final", "window:10", "persist:20")) -> dict[str, bool]:
    return {m: parse_mode(m)(trace) for m in modes}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rescore_many(traces: Sequence[Sequence], modes: Iterable[str] = ("any", "final", "window:10", "persist:20"),
                 reference: str = "any") -> dict:
    """Success rate (with Wilson 95% CI) of a set of episodes under each mode, plus how many episodes the
    reference mode credits that each other mode does not ('over_credit')."""
    modes = list(modes)
    if reference not in modes:
        modes = [reference] + modes
    per_ep = [rescore(t, modes) for t in traces]
    n = len(per_ep); out = {"n": n, "modes": {}}
    for m in modes:
        k = sum(e[m] for e in per_ep)
        over = sum(e[reference] and not e[m] for e in per_ep)
        out["modes"][m] = dict(successes=k, rate=k / n if n else None, ci95=wilson(k, n),
                               over_credit_by_reference=over)
    return out
