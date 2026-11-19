"""Credibility of a verifier, measured against behaviours whose ground truth is known.

Inputs are episodes: each has the verifier's verdict, the ground-truth outcome (from privileged physics or a
stronger reference), and optionally a continuous score / margin. The headline numbers:
  - false_success_rate: P(verdict=True | ground truth=False)  -> 'can it be fooled?'
  - false_failure_rate: P(verdict=False | ground truth=True)
  - counterexample_pass_rate: fraction of designed counterexamples the verifier accepts (must be 0)
  - margin statistics: how far expert vs counterexample distributions sit from the threshold (fragility)
  - cross-engine agreement: does the verdict for the same behaviour agree across physics engines?
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Episode:
    behaviour: str
    verdict: bool | None          # what the verifier under test says
    truth: bool | None            # ground truth
    score: float | None = None    # continuous score from the verifier (higher = more 'success'), optional
    margin: float | None = None   # signed distance of the ground-truth metric to its threshold, optional
    engine: str = ""
    seed: int | None = None


def confusion(eps: list[Episode]) -> dict:
    v = [(e.verdict, e.truth) for e in eps if e.verdict is not None and e.truth is not None]
    tp = sum(a and b for a, b in v); tn = sum((not a) and (not b) for a, b in v)
    fp = sum(a and (not b) for a, b in v); fn = sum((not a) and b for a, b in v)
    pos, neg = tp + fn, tn + fp
    tpr = tp / pos if pos else float("nan"); tnr = tn / neg if neg else float("nan")
    return dict(n=len(v), tp=tp, tn=tn, fp=fp, fn=fn,
                false_success_rate=fp / neg if neg else float("nan"),
                false_failure_rate=fn / pos if pos else float("nan"),
                balanced_accuracy=np.nanmean([tpr, tnr]) if pos or neg else float("nan"),
                unknown=sum(e.verdict is None for e in eps))


def counterexample_pass_rate(eps: list[Episode]) -> dict:
    """Per behaviour: fraction of episodes the verifier accepted. Counterexamples should be ~0, controls ~1."""
    out = {}
    for b in sorted({e.behaviour for e in eps}):
        xs = [e.verdict for e in eps if e.behaviour == b and e.verdict is not None]
        out[b] = dict(n=len(xs), accepted=int(sum(xs)), rate=(sum(xs) / len(xs)) if xs else float("nan"))
    return out


def by_margin(eps: list[Episode], edges=(0.0, 0.01, 0.02, 0.04, 0.08, np.inf)) -> list[dict]:
    """Verifier accuracy as a function of |margin| (distance of the true outcome from the decision boundary).
    Near-miss episodes (small |margin|) are where weak verifiers fail."""
    rows = []
    m = np.array([abs(e.margin) if e.margin is not None else np.nan for e in eps])
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [e for e, mm in zip(eps, m) if lo <= mm < hi]
        c = confusion(sel)
        rows.append(dict(abs_margin_lo=lo, abs_margin_hi=hi, n=c["n"], balanced_accuracy=c["balanced_accuracy"],
                         false_success_rate=c["false_success_rate"]))
    return rows


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Probability a random true-success episode scores higher than a random failure (ties count 1/2)."""
    p, n = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    if not len(p) or not len(n):
        return float("nan")
    gt = (p[:, None] > n[None]).mean(); eq = (p[:, None] == n[None]).mean()
    return float(gt + 0.5 * eq)


def cross_engine_agreement(verdicts: dict[str, dict[str, bool]]) -> dict:
    """verdicts[behaviour][engine] -> bool. Fraction of behaviours on which all engines agree, and the list of
    disagreements (these behaviours are 'sim-uncertain' and should not be scored without a real-world check)."""
    agree, dis = 0, []
    for b, per in verdicts.items():
        vals = set(per.values())
        if len(vals) == 1:
            agree += 1
        else:
            dis.append(b)
    n = len(verdicts)
    return dict(n=n, agreement=agree / n if n else float("nan"), sim_uncertain=dis)
