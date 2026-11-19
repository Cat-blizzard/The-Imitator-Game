"""Simulator-independent pieces shared by all adapters.

An adapter (ManiSkill/SAPIEN, robosuite/LIBERO, RoboTwin, Newton, real robot logs) only has to produce *traces*:
a dict  signal name -> per-step list of floats  (privileged state probes, contact flags, the benchmark's own
success flag under the reserved name 'bench_success'). Everything else (measurements for trace.* terms, the
'released' condition, the contract verdict, rescoring under different semantics) is computed here, so the same
contract gives the same verdict whichever adapter recorded the rollout.

Trace files (.npz): one array per signal, plus a JSON string array '__meta__' (task, seed, behaviour, adapter...).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..outcomes.metrics import measure

BENCH_SIGNAL = "bench_success"


def measurements_from_traces(contract, traces: dict, extra: dict | None = None) -> dict[str, float]:
    """Measurements for every trace.* term of the contract; non-trace terms must come in `extra` (keyed by term.key)."""
    m = dict(extra or {})
    for t in contract.outcome:
        if t.metric.startswith("trace.") and t.key not in m:
            sig = t.args.get("signal")
            m[t.key] = measure(t.metric, traces=traces, **t.args) if sig in traces and len(traces[sig]) else None
    return m


def released_from_trace(trace, window: int = 10) -> bool | None:
    """'released' is true if the release signal is on (>0.5) for the majority of the last `window` steps."""
    if trace is None or len(trace) == 0:
        return None
    return bool(np.mean(np.asarray(trace, float)[-window:]) > 0.5)


def bench_flags(traces: dict) -> dict[str, Any]:
    bs = traces.get(BENCH_SIGNAL)
    if bs is None or len(bs) == 0:
        return dict(bench_success_final=None, bench_success_any=None)
    return dict(bench_success_final=bool(np.asarray(bs)[-1]), bench_success_any=bool(np.any(bs)))


def evaluate_traces(contract, traces: dict, released_signal: str | None = "released",
                    extra_measurements: dict | None = None, release_window: int = 10,
                    protocol: str | None = None, release=None, control_freq: float = 20.0, hold: str | None = None):
    """(contract verdict, benchmark's own flags) from one recorded rollout.

    Dispatches on the evaluation protocol (argument, else contract.protocol, default "v1"). Under "v2" the trace
    must contain the evaluator-run settle phase ('phase' signal); `release` is a ReleaseRule or its kwargs (default:
    contract.release); the flags dict additionally carries 'protocol_info'. See adapters/protocol_v2.py.
    Under "v3" (adapters/protocol_v3.py) `hold` ('A' / 'B') overrides the contract's settle.hold and `release` (a dict)
    overrides the contract's release block."""
    proto = protocol or getattr(contract, "protocol", "v1")
    if proto == "v3":
        from .protocol_v3 import evaluate_traces_v3
        v, bench, info = evaluate_traces_v3(contract, traces, hold=hold, control_freq=control_freq,
                                            extra_measurements=extra_measurements, release=release)
        return v, dict(bench, protocol_info=info)
    if proto == "v2":
        from .protocol_v2 import ReleaseRule, evaluate_traces_v2
        rule = release if release is not None else (getattr(contract, "release", None) or None)
        if isinstance(rule, dict):
            rule = ReleaseRule(**rule)
        v, bench, info = evaluate_traces_v2(contract, traces, rule, control_freq, extra_measurements)
        return v, dict(bench, protocol_info=info)
    m = measurements_from_traces(contract, traces, extra_measurements)
    released = released_from_trace(traces.get(released_signal), release_window) if released_signal else None
    return contract.verdict(m, released=released), bench_flags(traces)


# ------------------------------------------------------------------------------------------------ trace files
def save_traces(path: str, traces: dict, meta: dict | None = None) -> None:
    arrays = {k: np.asarray(v, dtype=float) for k, v in traces.items()}
    if "__meta__" in arrays:
        raise ValueError("'__meta__' is reserved")
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta or {}, ensure_ascii=False)), **arrays)


def load_traces(path: str) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
    return {k: z[k] for k in z.files if k != "__meta__"}, meta
