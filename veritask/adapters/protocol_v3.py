"""Evaluation protocol v3, simulator-independent part (pure numpy).

Why v3 (second red-team round E13 against protocol v2, research/results/e13_redteam2/E13_REPORT.md; adjudicated by the
parent in adjudication.json): v2 made 0 false accepts but 25 false rejects on 132 decided episodes, from
  - a fixed 5 mm finger-object gap as "released": the fully open Panda gripper is only 3.7-4.1 mm from cup 1 (C23);
  - release judged as end-state proximity of the fingers only: a hand that let go and came back to rest 3 mm away
    failed (C25, J25) while a hand body 1-2 mm away passed (C29);
  - a window starting at the policy's last action in which every step must pass (C28, P23);
  - an impulse rule that counted a vertical fingertip load on the cube as a push (P24);
  - settle() refusing pd_joint_delta_pos (no verdict) and REPLAYING pd_joint_vel commands (the robot keeps moving).

Definitions (parent's defaults, 2026-11-14; switches kept for the alternatives)
  hold A   after the policy's last action the EVALUATOR holds the robot where it stopped: controller-level hold of the
           joint positions reached at the policy's last step (zero drive-velocity targets), grippers hold their last
           commanded opening; the same in every control mode (adapters/maniskill_v3.py).
  hold B   "the robot must have come to rest by itself": the evaluator replays the policy's last action unchanged and,
           in addition, requires robot joint speeds <= settle.robot_speed_max at every step of the judged window.
  settle   the hold runs for at most settle.max_seconds; the scene must become QUIESCENT (for settle.quiet_steps
           consecutive control steps: every quiet signal <= its bound, every quiet displacement group moved <= its bound,
           robot joint speed <= robot_speed_max); the JUDGED WINDOW is the settle.window_seconds that follow. No
           quiescence early enough for the window to fit inside max_seconds -> reject. End-state terms
           (END_STATE_METRICS) read only the judged window; event terms read the episode up to the window's end.
  release  at every step of the judged window: no robot-object contact impulse (any link, all substeps) AND the object
           is not enclosed by a gripper: the enclosure clearance signal (sum of the two fingers' clearances to the
           object, recorded only when part of the object lies between the finger pads; 1 m otherwise) > the contract's
           release.enclosure_margin_m. Proximity without contact and without enclosure is released, whatever the link.

The window position is recomputed here from the recorded traces and the contract's settle block (the recorder uses the
same function online only to know when to stop), so a verdict depends on the trace file and the contract alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..outcomes.metrics import measure
from .base import BENCH_SIGNAL

PROTOCOL = "v3"
PHASE_SIGNAL = "phase"                  # 0 = policy, 1 = evaluator-run settle (same signal as protocol v2)
HOLD_MODES = ("A", "B")
NOT_ENCLOSED_M = 1.0                    # value of an enclosure-clearance signal when nothing lies between the finger pads
END_STATE_METRICS = ("trace.window_max", "trace.window_min", "trace.last", "trace.window_disp")
EVENT_METRICS = ("trace.min", "trace.max", "trace.any", "trace.count_true", "trace.sum")
FORBIDDEN_METRICS = ("trace.final",)
SETTLE_KEYS = {"hold", "max_seconds", "window_seconds", "quiet_steps", "quiet_max", "quiet_disp", "robot_speed_signal",
               "robot_speed_max"}


class ProtocolError(ValueError):
    """The contract cannot be evaluated under protocol v3 as written."""


# ------------------------------------------------------------------------------------------------ settle config
@dataclass
class SettleConfig:
    hold: str = "A"
    max_seconds: float = 3.0
    window_seconds: float = 1.0
    quiet_steps: int = 10
    quiet_max: dict[str, float] = field(default_factory=dict)      # signal -> bound at each of the quiet steps
    quiet_disp: list[dict] = field(default_factory=list)           # [{"signals": [x, y, z], "max": m}] over the quiet steps
    robot_speed_signal: str = "robot_qvel_max"
    robot_speed_max: float = 0.01

    @staticmethod
    def from_contract(contract, hold: str | None = None) -> "SettleConfig":
        s = dict(getattr(contract, "settle", None) or {})
        cfg = SettleConfig(**{k: v for k, v in s.items() if k in SETTLE_KEYS})
        cfg.quiet_max = dict(cfg.quiet_max); cfg.quiet_disp = [dict(g) for g in cfg.quiet_disp]
        if hold is not None:
            cfg.hold = hold
        if cfg.hold not in HOLD_MODES:
            raise ProtocolError(f"settle.hold '{cfg.hold}' not one of {HOLD_MODES}")
        return cfg

    def steps(self, control_freq: float) -> tuple[int, int, int]:
        """(quiet steps K, window steps W, cap N) in control steps"""
        return int(self.quiet_steps), int(round(self.window_seconds * control_freq)), int(round(self.max_seconds * control_freq))

    def signals(self) -> list[str]:
        return sorted(set(self.quiet_max) | {s for g in self.quiet_disp for s in g["signals"]} | {self.robot_speed_signal})


def _disp(X: np.ndarray) -> float:
    """largest distance of any row of X from its first row (X: steps x dims)"""
    return float(np.max(np.linalg.norm(X - X[0], axis=1))) if len(X) else 0.0


def quiet_mask(traces: dict, cfg: SettleConfig, n_policy: int) -> np.ndarray | None:
    """Per settle step (index 0 = first settle step): 1 if the per-step bounds hold (quiet signals, robot speed)."""
    n = len(traces[PHASE_SIGNAL]) - n_policy
    if n <= 0:
        return np.zeros(0, bool)
    ok = np.ones(n, bool)
    for s, b in list(cfg.quiet_max.items()) + [(cfg.robot_speed_signal, cfg.robot_speed_max)]:
        if s not in traces:
            return None
        ok &= np.asarray(traces[s], float)[n_policy:] <= float(b)
    return ok


def find_window(traces: dict, cfg: SettleConfig, n_policy: int, control_freq: float = 20.0) -> dict[str, Any]:
    """Locate the judged window in a recorded settle phase.

    Returns dict(status=..., quiet_at=..., window=(first, last) absolute step indices) where status is
      'ok'            quiescence at settle step q (steps q-K+1..q quiet), window = the W steps after it, all recorded
      'incomplete'    no window yet, but the recorded settle is shorter than the cap (keep stepping / verdict unknown)
      'not_quiescent' no quiescence early enough for the window to fit inside the cap -> reject."""
    K, W, N = cfg.steps(control_freq)
    n_settle = len(traces[PHASE_SIGNAL]) - n_policy
    m = quiet_mask(traces, cfg, n_policy)
    if m is None:
        return dict(status="missing_signal", missing=[s for s in cfg.signals() if s not in traces])
    groups = [np.stack([np.asarray(traces[s], float)[n_policy:] for s in g["signals"]], axis=1) for g in cfg.quiet_disp]
    last_q = N - W - 1                         # latest settle index at which quiescence still leaves room for the window
    for q in range(K - 1, min(n_settle, last_q + 1)):
        if not m[q - K + 1:q + 1].all():
            continue
        if any(_disp(G[q - K + 1:q + 1]) > float(g["max"]) for G, g in zip(groups, cfg.quiet_disp)):
            continue
        first, last = n_policy + q + 1, n_policy + q + W
        if q + W >= n_settle:
            return dict(status="incomplete", quiet_at=n_policy + q, window=(first, last))
        return dict(status="ok", quiet_at=n_policy + q, window=(first, last))
    if n_settle >= last_q + 1:                 # every admissible quiescence point has been examined
        return dict(status="not_quiescent")
    return dict(status="incomplete")


# ------------------------------------------------------------------------------------------------ release
@dataclass
class ReleaseRuleV3:
    """released(step) = every contact signal <= max_impulse_Ns AND every enclosure-clearance signal > enclosure_margin_m
    (AND every gap signal > gap_m, for compatibility with v2-style rules)."""
    contact_signals: list[str] = field(default_factory=list)
    enclosure_signals: list[str] = field(default_factory=list)
    enclosure_margin_m: float = 0.002
    max_impulse_Ns: float = 0.0
    gap_signals: list[str] = field(default_factory=list)
    gap_m: float = 0.005

    @staticmethod
    def from_dict(d: dict) -> "ReleaseRuleV3":
        keys = {"contact_signals", "enclosure_signals", "enclosure_margin_m", "max_impulse_Ns", "gap_signals", "gap_m"}
        return ReleaseRuleV3(**{k: v for k, v in (d or {}).items() if k in keys})

    def per_step(self, traces: dict) -> np.ndarray | None:
        sigs = list(self.contact_signals) + list(self.enclosure_signals) + list(self.gap_signals)
        if not sigs or any(s not in traces or len(traces[s]) == 0 for s in sigs):
            return None
        ok = np.ones(len(traces[sigs[0]]), bool)
        for s in self.contact_signals:
            ok &= np.asarray(traces[s], float) <= self.max_impulse_Ns
        for s in self.enclosure_signals:
            ok &= np.asarray(traces[s], float) > self.enclosure_margin_m
        for s in self.gap_signals:
            ok &= np.asarray(traces[s], float) > self.gap_m
        return ok

    def to_dict(self) -> dict:
        return dict(contact_signals=list(self.contact_signals), enclosure_signals=list(self.enclosure_signals),
                    enclosure_margin_m=self.enclosure_margin_m, max_impulse_Ns=self.max_impulse_Ns,
                    gap_signals=list(self.gap_signals), gap_m=self.gap_m)


# ------------------------------------------------------------------------------------------------ checks
def phase_counts(traces: dict) -> tuple[int, int]:
    """(policy steps, settle steps); the settle phase is the trailing run of phase == 1."""
    ph = traces.get(PHASE_SIGNAL)
    if ph is None:
        return max((len(v) for v in traces.values()), default=0), 0
    ph = np.asarray(ph, float) > 0.5
    n = 0
    for x in ph[::-1]:
        if not x:
            break
        n += 1
    return len(ph) - n, n


def check_contract_v3(contract, control_freq: float = 20.0) -> list[str]:
    """Problems that make a contract unusable under protocol v3 (empty list = fine)."""
    p = []
    try:
        cfg = SettleConfig.from_contract(contract)
    except (TypeError, ProtocolError) as e:
        return [f"settle block: {e}"]
    K, W, N = cfg.steps(control_freq)
    if K < 1 or W < 1:
        p.append("settle.quiet_steps and settle.window_seconds must give at least one step each")
    if K + W > N:
        p.append(f"settle: quiet_steps ({K}) + window ({W}) exceed the cap max_seconds ({N} steps)")
    for g in cfg.quiet_disp:
        if not g.get("signals") or "max" not in g:
            p.append("settle.quiet_disp entries need 'signals' and 'max'")
    for i, t in enumerate(contract.outcome):
        if t.metric in FORBIDDEN_METRICS:
            p.append(f"outcome[{i}] uses {t.metric} (a mean); use trace.window_max / window_min")
        elif t.metric in END_STATE_METRICS:
            w = t.args.get("window")
            if w is not None and int(w) != W:
                p.append(f"outcome[{i}] {t.key}: window {w} differs from the judged window ({W} steps)")
        elif t.metric.startswith("trace.") and t.metric not in EVENT_METRICS:
            p.append(f"outcome[{i}] uses {t.metric}, which protocol v3 does not classify (end-state or event)")
    rel = getattr(contract, "release", None) or {}
    if contract.require_released:
        if not rel.get("contact_signals"):
            p.append("require_released: the release block needs contact_signals (no robot-object contact)")
        if not rel.get("enclosure_signals"):
            p.append("require_released: the release block needs enclosure_signals (object not enclosed by a gripper)")
    return p


# ------------------------------------------------------------------------------------------------ evaluation
def bench_flags(traces: dict, n_policy: int) -> dict[str, Any]:
    bs = traces.get(BENCH_SIGNAL)
    if bs is None or len(bs) == 0 or n_policy == 0:
        return dict(bench_success_final=None, bench_success_any=None, bench_success_after_settle=None)
    bs = np.asarray(bs, float) > 0.5
    return dict(bench_success_final=bool(bs[n_policy - 1]), bench_success_any=bool(bs[:n_policy].any()),
                bench_success_after_settle=bool(bs[-1]))


def evaluate_traces_v3(contract, traces: dict, hold: str | None = None, control_freq: float = 20.0,
                       extra_measurements: dict | None = None, release: dict | ReleaseRuleV3 | None = None,
                       strict: bool = True):
    """(verdict, benchmark flags, protocol info) for one rollout recorded under protocol v3.

    hold: 'A' or 'B' (default: the contract's settle.hold). Under 'B' robot joint speed must also stay <= the bound at
    every window step. release: override of the contract's release block (default: contract.release)."""
    problems = check_contract_v3(contract, control_freq)
    if problems and strict:
        raise ProtocolError(f"{contract.task_id}: " + "; ".join(problems))
    cfg = SettleConfig.from_contract(contract, hold)
    n_policy, n_settle = phase_counts(traces)
    info: dict[str, Any] = dict(protocol=PROTOCOL, hold=cfg.hold, n_policy=n_policy, n_settle=n_settle, problems=problems)
    bench = bench_flags(traces, n_policy)
    win = find_window(traces, cfg, n_policy, control_freq) if n_settle else dict(status="incomplete")
    info["window_status"] = win["status"]
    if win["status"] != "ok":
        v = contract.verdict({}, released=None)
        if win["status"] == "not_quiescent":
            K, W, N = cfg.steps(control_freq)
            v.update(success=False, binding_term="protocol.quiescence",
                     protocol_reject=f"the scene did not become quiescent within {cfg.max_seconds} s "
                                     f"({K} quiet steps + a {W}-step window inside {N} steps)")
        else:
            info["unknown_reason"] = win.get("missing") or "settle phase too short to locate the judged window"
        return v, bench, info
    first, last = win["window"]
    info.update(quiet_at=win["quiet_at"], window=[first, last], settle_steps_used=last + 1 - n_policy)
    W = last + 1 - first
    upto = {k: np.asarray(v, float)[:last + 1] for k, v in traces.items()}       # episode up to the end of the window
    wtr = {k: v[first:] for k, v in upto.items()}                               # the judged window
    m: dict[str, Any] = dict(extra_measurements or {})
    for t in contract.outcome:
        if not t.metric.startswith("trace.") or t.key in m:
            continue
        sig = t.args.get("signal")
        if sig not in traces or len(traces[sig]) == 0:
            m[t.key] = None; continue
        args = dict(t.args)
        if t.metric in END_STATE_METRICS:
            args["window"] = W
            if t.metric == "trace.window_disp" and any(s not in traces for s in args.get("signals", [sig])):
                m[t.key] = None; continue
            m[t.key] = measure(t.metric, traces=wtr, **args)
        else:
            m[t.key] = measure(t.metric, traces=upto, **args)
    released = None
    if contract.require_released:
        rule = release if isinstance(release, ReleaseRuleV3) else ReleaseRuleV3.from_dict(release or contract.release)
        ps = rule.per_step(upto)
        if ps is not None:
            released = bool(ps[first:].all())
            info.update(released_steps=int(ps[first:].sum()), release_rule=rule.to_dict())
    v = contract.verdict(m, released=released)
    if cfg.hold == "B":
        rs = np.asarray(traces.get(cfg.robot_speed_signal, []), float)
        at_rest = bool(len(rs) > last and (rs[first:last + 1] <= cfg.robot_speed_max).all())
        info["robot_at_rest_in_window"] = at_rest
        v["robot_at_rest"] = at_rest
        if not at_rest and v["success"] is not None:
            v["success"] = False
            v["binding_term"] = "protocol.robot_at_rest"
    return v, bench, info


# ================================================================================================ D2 additions (appended)
# 1. Topology trace metrics (outcomes/topology.py) are classified like the window / event metrics above: '*_window_min'
#    reads the judged window (end-state), '*_min' the episode up to the window's end (event).
from ..outcomes.topology import TRACE_END_STATE as _TOPO_END_STATE, TRACE_EVENT as _TOPO_EVENT  # noqa: E402

END_STATE_METRICS = END_STATE_METRICS + _TOPO_END_STATE
EVENT_METRICS = EVENT_METRICS + _TOPO_EVENT

# 2. Proof test (contract schema 0.4, block "proof_test"): an optional stage AFTER the judged window and the retreat.
#    The evaluator moves the robot out of the workspace, then runs each declared test from the same post-retreat state:
#    a short settle (stage 0), a standard disturbance applied to a scene object by the evaluator (stage 1: lift and
#    tilt a container, shake a hook, push a top object, pull along the extraction axis, shake the table), and a rest
#    (stage 2). Each test is judged on its own traces: end-state terms on the last window_seconds of the test, event
#    terms on stages 1-2. A simulator binding records every test's traces under keys 'pt:<test>:<signal>' (plus
#    'pt:<test>:pt_stage'), so the proof verdict is recomputable from the trace file and the contract alone.
#    Verdicts: main = evaluate_traces_v3 (unchanged); proof = evaluate_proof_tests; with proof = combine_with_proof.
PT_PREFIX = "pt:"
PT_STAGE_SIGNAL = "pt_stage"
PT_STAGES = {"settle": 0, "disturb": 1, "rest": 2}
PT_DEFAULTS = dict(robot="clear", pre_settle_seconds=0.5, window_seconds=1.0)


def pt_key(test: str, signal: str) -> str:
    return f"{PT_PREFIX}{test}:{signal}"


def proof_test_block(contract) -> dict:
    """the contract's proof_test block with defaults filled in ({} when the contract declares none)"""
    b = dict(getattr(contract, "proof_test", None) or {})
    if not b:
        return {}
    out = dict(PT_DEFAULTS, **b)
    out["tests"] = [dict(t, params=dict(t.get("params") or {}), rest_seconds=float(t.get("rest_seconds", 1.0)))
                    for t in b.get("tests", [])]
    return out


def split_proof_traces(traces: dict) -> tuple[dict, dict]:
    """(traces without 'pt:' keys, {test name: {signal: array}})"""
    main, per = {}, {}
    for k, v in traces.items():
        if k.startswith(PT_PREFIX):
            name, sig = k[len(PT_PREFIX):].split(":", 1)
            per.setdefault(name, {})[sig] = np.asarray(v, float)
        else:
            main[k] = v
    return main, per


def check_proof_test(contract, control_freq: float = 20.0) -> list[str]:
    """Problems that make the proof_test block unusable (empty list = fine or no block)."""
    from ..spec.contract import validate_proof_test
    b = getattr(contract, "proof_test", None) or {}
    if not b:
        return []
    p = [x for x in validate_proof_test(b, getattr(contract, "schema_version", "0.4"), contract.protocol)
         if not x.startswith("warning:")]
    blk = proof_test_block(contract)
    W = int(round(float(blk["window_seconds"]) * control_freq))
    for t in blk["tests"]:
        if int(round(t["rest_seconds"] * control_freq)) < W:
            p.append(f"proof_test {t.get('name')}: rest_seconds shorter than the judged window")
        for term in t.get("terms", []):
            m, w = term.get("metric"), (term.get("args") or {}).get("window")
            if m in END_STATE_METRICS and w is not None and int(w) != W:
                p.append(f"proof_test {t.get('name')}: {m} window {w} differs from the proof window ({W} steps)")
            elif isinstance(m, str) and m.startswith("trace.") and m not in END_STATE_METRICS + EVENT_METRICS:
                p.append(f"proof_test {t.get('name')}: {m} is neither an end-state nor an event metric")
    return p


def _term_obj(d: dict):
    from ..spec.contract import OutcomeTerm
    return OutcomeTerm(**d)


def evaluate_proof_tests(contract, traces: dict, control_freq: float = 20.0) -> dict[str, Any]:
    """Proof verdict from the 'pt:' traces of one rollout.

    Returns dict(declared, success, tests=[dict(name, kind, target, success, terms, margins, binding_term, n_steps,
    status)], binding_test). success is None when a declared test has no traces or a term cannot be measured (unknown,
    never silently true or false)."""
    blk = proof_test_block(contract)
    if not blk:
        return dict(declared=False, success=None, tests=[], binding_test=None)
    _, per = split_proof_traces(traces)
    W = int(round(float(blk["window_seconds"]) * control_freq))
    out, ok, unknown = [], True, False
    for t in blk["tests"]:
        name = t["name"]
        tr = per.get(name)
        res = dict(name=name, kind=t.get("kind"), target=t.get("target"), success=None, terms={}, margins={},
                   binding_term=None, n_steps=0, status="missing")
        if not tr or PT_STAGE_SIGNAL not in tr:
            unknown = True; out.append(res); continue
        st = tr[PT_STAGE_SIGNAL]
        res["n_steps"] = int(len(st))
        idx = np.nonzero(st >= PT_STAGES["disturb"])[0]
        if not len(idx) or len(st) - idx[0] < W:
            res["status"] = "too_short"; unknown = True; out.append(res); continue
        evt = {k: v[idx[0]:] for k, v in tr.items()}
        win = {k: v[-W:] for k, v in tr.items()}
        t_ok, t_unknown, worst = True, False, None
        for d in t.get("terms", []):
            term = _term_obj(d)
            sig = term.args.get("signal")
            if sig not in tr:
                res["terms"][term.key] = None; t_unknown = True; continue
            args = dict(term.args)
            if term.metric in END_STATE_METRICS:
                args["window"] = W
                v = measure(term.metric, traces=win, **args)
            else:
                v = measure(term.metric, traces=evt, **args)
            h = term.holds(float(v))
            res["terms"][term.key] = float(v)
            res["margins"][term.key] = term.norm_margin(float(v))
            t_ok &= h
            if worst is None or term.norm_margin(float(v)) < res["margins"][worst]:
                worst = term.key
        res["binding_term"] = worst
        res["status"] = "ok"
        res["success"] = None if t_unknown else bool(t_ok)
        unknown |= t_unknown
        ok &= bool(t_ok) if not t_unknown else True
        out.append(res)
    decided_fail = any(r["success"] is False for r in out)
    success = False if decided_fail else (None if unknown else bool(ok))
    binding = min((r for r in out if r["margins"]), key=lambda r: min(r["margins"].values()), default={}).get("name")
    return dict(declared=True, success=success, tests=out, binding_test=binding)


def combine_with_proof(main_success, proof: dict | None):
    """verdict of 'contract + proof test': False if either is False, None if either is unknown, else True. Without a
    declared proof test this is the main verdict."""
    if not proof or not proof.get("declared"):
        return main_success
    p = proof.get("success")
    if main_success is False or p is False:
        return False
    if main_success is None or p is None:
        return None
    return True
