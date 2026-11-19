"""Sim-to-real agreement statistics for policy evaluation.

- rank agreement between simulated and real success rates across policies (Kendall tau, Spearman rho)
- MMRV (mean maximum rank violation, as used by SIMPLER): for each policy, the largest real-performance gap among
  pairs whose sim ranking is inverted; averaged over policies. 0 = sim never mis-ranks policies.
- prediction-powered inference (PPI) for a policy's real success rate: combine many sim rollouts with a few paired
  sim+real rollouts to debias the sim estimate, with a normal-approximation confidence interval.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rank_agreement(sim: dict[str, float], real: dict[str, float]) -> dict:
    keys = sorted(set(sim) & set(real))
    s = np.array([sim[k] for k in keys]); r = np.array([real[k] for k in keys])
    tau = stats.kendalltau(s, r).statistic if len(keys) > 1 else float("nan")
    rho = stats.spearmanr(s, r).statistic if len(keys) > 1 else float("nan")
    return dict(n_policies=len(keys), kendall_tau=float(tau), spearman_rho=float(rho), mmrv=mmrv(sim, real))


def mmrv(sim: dict[str, float], real: dict[str, float]) -> float:
    keys = sorted(set(sim) & set(real))
    if len(keys) < 2:
        return float("nan")
    worst = []
    for i in keys:
        v = 0.0
        for j in keys:
            if i == j:
                continue
            if (sim[i] < sim[j]) != (real[i] < real[j]):   # sim ordering disagrees with real ordering
                v = max(v, abs(real[i] - real[j]))
        worst.append(v)
    return float(np.mean(worst))


def ppi_mean(sim_unlabeled: np.ndarray, sim_paired: np.ndarray, real_paired: np.ndarray, alpha: float = 0.05) -> dict:
    """Prediction-powered estimate of the real success rate.

    sim_unlabeled: verifier outcomes (0/1 or scores) on N sim-only rollouts
    sim_paired, real_paired: outcomes on n paired rollouts evaluated both in sim and on the real robot
    estimate = mean(sim_unlabeled) + mean(real_paired - sim_paired)   (bias 'rectifier')
    """
    f_u = np.asarray(sim_unlabeled, float); f_p = np.asarray(sim_paired, float); y_p = np.asarray(real_paired, float)
    N, n = len(f_u), len(y_p)
    rect = y_p - f_p
    est = f_u.mean() + rect.mean()
    se = np.sqrt(f_u.var(ddof=1) / N + rect.var(ddof=1) / n) if N > 1 and n > 1 else float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    real_only_se = np.sqrt(y_p.var(ddof=1) / n) if n > 1 else float("nan")
    return dict(estimate=float(est), ci=(float(est - z * se), float(est + z * se)), se=float(se),
                real_only_estimate=float(y_p.mean()) if n else float("nan"), real_only_se=float(real_only_se),
                sim_bias=float(-rect.mean()) if n else float("nan"), N_sim=N, n_paired=n)
