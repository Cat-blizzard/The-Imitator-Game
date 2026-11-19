"""Evaluation protocol v2, simulator-independent part (pure numpy).

Why v2 (defects found by the E10 red team, research/results/e10_redteam/E10_REPORT.md):
  1. contact sampling   v1 read contacts once per control step, at the last physics substep; short contacts were missed.
                        v2 signals are ACCUMULATED over all physics substeps of the control step (the adapter does this;
                        see adapters/maniskill_v2.py).
  2. window mean        v1 end-state terms used trace.final (mean of the last K steps); a late violation could hide.
                        v2 end-state terms use trace.window_max / trace.window_min / trace.last and are read ONLY from
                        the settle phase, so every settle step must satisfy the bound.
  3. settle             v1 declared settle_seconds but nobody ran it. In v2 the EVALUATOR runs the settle phase after the
                        policy's last action (robot holds its last command, no new actions) and marks those steps in the
                        'phase' signal (0 = policy, 1 = settle). A trace without a complete settle phase is not judged
                        (verdict unknown).
  4. released           v1 used "not is_grasping" (a force test). v2: no robot-object contact AND finger-object gap >
                        RELEASE_GAP_M, at EVERY step of the settle phase (ReleaseRule below).

Term semantics under v2
  end-state metrics  trace.window_max, trace.window_min, trace.last   -> computed on the settle phase only; an explicit
                     args.window must not exceed the number of settle steps (default: the whole settle phase)
  event metrics      trace.min, trace.max, trace.any, trace.count_true -> computed on the whole episode (policy + settle)
  trace.final        not allowed in a v2 contract (it is a mean)

Everything here works on plain trace dicts (signal -> per-step array), so saved .npz traces can be re-scored offline
with other parameters (see research/scripts/e11_rescore.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..outcomes.metrics import measure
from .base import BENCH_SIGNAL

PROTOCOL = "v2"
PHASE_SIGNAL = "phase"                 # per-step: 0 = policy phase, 1 = evaluator-run settle phase
RELEASE_SIGNAL = "released_gap"        # per-step release flag recorded by the adapter (same rule as ReleaseRule)
RELEASE_GAP_M = 0.005                  # finger-object gap that counts as "let go" [m]
RELEASE_MAX_IMPULSE_NS = 0.0           # robot-object accumulated contact impulse allowed while "released" [N*s]
CONTACT_IMPULSE_NS = 5e-4              # a control step counts as "contact" if the impulse accumulated over all of its
                                       # physics substeps reaches this [N*s] (= 0.05 N, v1's force floor, for one 10 ms substep)
END_STATE_METRICS = ("trace.window_max", "trace.window_min", "trace.last")
EVENT_METRICS = ("trace.min", "trace.max", "trace.any", "trace.count_true", "trace.sum")
FORBIDDEN_METRICS = ("trace.final",)


class ProtocolError(ValueError):
    """The contract cannot be evaluated under protocol v2 as written."""


@dataclass
class ReleaseRule:
    """released(step) = every contact signal <= max_impulse_Ns AND every gap signal > gap_m.
    contact_signals: accumulated robot-object impulse per control step [N*s] (all robot links).
    gap_signals:     smallest finger-object distance seen in any physics substep of the control step [m]."""
    contact_signals: list[str] = field(default_factory=list)
    gap_signals: list[str] = field(default_factory=list)
    gap_m: float = RELEASE_GAP_M
    max_impulse_Ns: float = RELEASE_MAX_IMPULSE_NS

    def per_step(self, traces: dict) -> np.ndarray | None:
        sigs = list(self.contact_signals) + list(self.gap_signals)
        if not sigs or any(s not in traces or len(traces[s]) == 0 for s in sigs):
            return None
        ok = np.ones(len(traces[sigs[0]]), bool)
        for s in self.contact_signals:
            ok &= np.asarray(traces[s], float) <= self.max_impulse_Ns
        for s in self.gap_signals:
            ok &= np.asarray(traces[s], float) > self.gap_m
        return ok

    def step_value(self, values: dict) -> float:
        """Same rule on one step's values (used by the recorder while stepping)."""
        return float(all(values[s] <= self.max_impulse_Ns for s in self.contact_signals)
                     and all(values[s] > self.gap_m for s in self.gap_signals))

    def to_dict(self) -> dict:
        return dict(contact_signals=list(self.contact_signals), gap_signals=list(self.gap_signals), gap_m=self.gap_m,
                    max_impulse_Ns=self.max_impulse_Ns)


def phase_counts(traces: dict) -> tuple[int, int]:
    """(policy steps, settle steps). The settle phase is the trailing run of phase == 1."""
    ph = traces.get(PHASE_SIGNAL)
    if ph is None:
        n = max((len(v) for v in traces.values()), default=0)
        return n, 0
    ph = np.asarray(ph, float) > 0.5
    n_settle = 0
    for x in ph[::-1]:
        if not x:
            break
        n_settle += 1
    return len(ph) - n_settle, n_settle


def settle_steps(contract, control_freq: float) -> int:
    return int(round(float(contract.settle_seconds) * float(control_freq)))


def check_contract_v2(contract, control_freq: float = 20.0, release: ReleaseRule | None = None) -> list[str]:
    """Problems that make a contract unusable under protocol v2 (empty list = fine)."""
    p = []
    n = settle_steps(contract, control_freq)
    if n < 1:
        p.append("settle_seconds gives no settle step: protocol v2 needs an evaluator-run settle phase")
    for i, t in enumerate(contract.outcome):
        if t.metric in FORBIDDEN_METRICS:
            p.append(f"outcome[{i}] uses {t.metric} (a mean over the window); use trace.window_max / trace.window_min")
        elif t.metric in END_STATE_METRICS:
            w = t.args.get("window")
            if w is not None and int(w) > n:
                p.append(f"outcome[{i}] {t.key}: window {w} is longer than the settle phase ({n} steps); end-state terms "
                         f"may only read the settle phase")
        elif t.metric.startswith("trace.") and t.metric not in EVENT_METRICS:
            p.append(f"outcome[{i}] uses {t.metric}, which protocol v2 does not classify (end-state or event)")
    if contract.require_released and release is None:
        p.append("require_released is true but no ReleaseRule was given")
    if release is not None:   # an explicit gap term in the contract must agree with the rule (one number, written once)
        for i, t in enumerate(contract.outcome):
            if t.args.get("signal") in release.gap_signals and abs(float(t.threshold) - release.gap_m) > 1e-12:
                p.append(f"outcome[{i}] {t.key}: threshold {t.threshold} differs from the release rule's gap_m {release.gap_m}")
    return p


def measurements_v2(contract, traces: dict, n_policy: int, n_settle: int, extra: dict | None = None) -> dict[str, float | None]:
    m: dict[str, Any] = dict(extra or {})
    settle = {k: np.asarray(v, float)[n_policy:n_policy + n_settle] for k, v in traces.items()}
    for t in contract.outcome:
        if not t.metric.startswith("trace.") or t.key in m:
            continue
        sig = t.args.get("signal")
        if sig not in traces or len(traces[sig]) == 0:
            m[t.key] = None; continue
        if t.metric in END_STATE_METRICS:
            args = dict(t.args); args["window"] = int(args.get("window") or n_settle)
            m[t.key] = measure(t.metric, traces=settle, **args) if n_settle else None
        else:
            m[t.key] = measure(t.metric, traces=traces, **t.args)
    return m


def bench_flags_v2(traces: dict, n_policy: int) -> dict[str, Any]:
    """The benchmark's own success flag: at the policy's last step ('final'), at any policy step ('any'), and at the
    end of the settle phase (reported separately; the benchmark itself has no settle phase)."""
    bs = traces.get(BENCH_SIGNAL)
    if bs is None or len(bs) == 0 or n_policy == 0:
        return dict(bench_success_final=None, bench_success_any=None, bench_success_after_settle=None)
    bs = np.asarray(bs, float) > 0.5
    return dict(bench_success_final=bool(bs[n_policy - 1]), bench_success_any=bool(bs[:n_policy].any()),
                bench_success_after_settle=bool(bs[-1]))


def evaluate_traces_v2(contract, traces: dict, release: ReleaseRule | None = None, control_freq: float = 20.0,
                       extra_measurements: dict | None = None, strict: bool = True):
    """(contract verdict, benchmark flags, protocol info) for one rollout recorded under protocol v2.

    The verdict is 'unknown' (success None) when the trace has no complete settle phase. With strict=True a contract
    that violates the protocol (check_contract_v2) raises ProtocolError instead of being evaluated loosely."""
    problems = check_contract_v2(contract, control_freq, release)
    if problems and strict:
        raise ProtocolError(f"{contract.task_id}: " + "; ".join(problems))
    n_policy, n_settle = phase_counts(traces)
    need = settle_steps(contract, control_freq)
    info = dict(protocol=PROTOCOL, n_policy=n_policy, n_settle=n_settle, settle_steps_required=need, problems=problems,
                release_rule=release.to_dict() if release else None)
    bench = bench_flags_v2(traces, n_policy)
    if n_settle < need or n_policy == 0:
        info["unknown_reason"] = f"settle phase has {n_settle} steps, contract needs {need}"
        v = contract.verdict({}, released=None)
        return v, bench, info
    # judge exactly the required settle steps (a longer recorded settle is truncated to the first `need` steps after
    # the policy ended, so the verdict does not depend on how long the recorder kept going)
    traces = {k: np.asarray(v, float)[:n_policy + need] for k, v in traces.items()}
    m = measurements_v2(contract, traces, n_policy, need, extra_measurements)
    released = None
    if contract.require_released and release is not None:
        ps = release.per_step(traces)
        if ps is not None:
            rs = ps[n_policy:n_policy + need]
            released = bool(rs.all())
            info["released_steps"] = int(rs.sum())
    info["n_settle"] = need
    return contract.verdict(m, released=released), bench, info
