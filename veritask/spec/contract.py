"""Task contract: a declarative description of a physical task whose completion can be verified.

A contract states *what physical outcome must hold at the end*, not how to get there. From one contract we derive
(1) the success verifier, (2) the counterexample suite it must reject, (3) the quantities a real-robot setup must
measure so the same verdict can be computed in sim and on hardware.

Contracts are plain JSON (see tasks/*.json; machine-readable schema in spec/contract.schema.json). This module
loads, validates, saves and evaluates them.

Schema history
  0.1  (2026-11-12) first versioned schema.
  0.2  (2026-11-12) optional "protocol" ("v1" default | "v2": evaluator-run settle phase, end-state terms must hold
       at every settle step, see adapters/protocol_v2.py) and optional "release" block (contact_signals, gap_signals,
       gap_m, max_impulse_Ns) so the release rule is written in the contract, not only in the simulator binding.
       0.2 is a superset of 0.1; files without "schema_version" are read as the current version.
  0.3  (2026-11-14) protocol "v3" (adapters/protocol_v3.py): evaluator hold (A: robot held where it stopped; B: robot
       must come to rest by itself), adaptive settle with a quiescence condition and a judged window, release by
       contact + enclosure. New optional "settle" block (hold, max_seconds, window_seconds, quiet_steps, quiet_max,
       quiet_disp, robot_speed_signal, robot_speed_max) and release keys enclosure_signals / enclosure_margin_m.
       0.3 is a superset of 0.2.
  0.4  (2026-11-16, card D2) optional "proof_test" block: evaluator-run standard disturbances applied to scene objects
       (not through the robot) after release and retreat, each followed by a re-judgement on its own terms (see
       adapters/protocol_v3.py, section "proof test"); topology metrics (outcomes/topology.py). 0.4 is a superset of
       0.3; contracts without a proof_test block are unchanged.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

SCHEMA_VERSION = "0.4"
SUPPORTED_SCHEMA_VERSIONS = ("0.1", "0.2", "0.3", "0.4")
PROTOCOLS = ("v1", "v2", "v3")

OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
}


class ContractError(ValueError):
    """A contract file is structurally invalid (cannot be evaluated as written)."""


@dataclass
class OutcomeTerm:
    """One measurable condition on the final state, e.g. fold distance dAC < 0.08 m."""
    metric: str                 # name in the outcome registry, e.g. "cloth.corner_distance"
    op: str                     # one of OPS
    threshold: float
    units: str = ""
    args: dict[str, Any] = field(default_factory=dict)   # metric arguments, e.g. corner vertex ids
    real_measurement: str = ""  # how the same quantity is measured on hardware (camera / scale / ...)

    def holds(self, value: float) -> bool:
        return OPS[self.op](value, self.threshold)

    def margin(self, value: float) -> float:
        """Signed distance to the threshold; positive = satisfied. Same units as the metric."""
        return (self.threshold - value) if self.op in ("<", "<=") else (value - self.threshold)

    @property
    def key(self) -> str:
        """Measurement key: the metric name, qualified by the trace signal for trace.* metrics."""
        sig = self.args.get("signal")
        return f"{self.metric}:{sig}" if sig else self.metric

    def norm_margin(self, value: float) -> float:
        """Margin relative to the threshold magnitude, so terms with different units can be compared."""
        return self.margin(value) / (abs(self.threshold) if self.threshold else 1.0)


@dataclass
class Counterexample:
    """A behaviour the verifier MUST reject (expected=False) or accept (expected=True, for controls)."""
    name: str
    description: str
    expected: bool = False
    generator: str = ""         # adapter-specific behaviour id


@dataclass
class TaskContract:
    task_id: str
    instruction: str
    outcome: list[OutcomeTerm]
    require_released: bool = True        # the robot must not be holding the object at evaluation time
    settle_seconds: float = 1.0          # the EVALUATOR runs a settle phase of this length after the policy ends
                                         # (robot holds its last command, no new actions); end-state windows are
                                         # read from that phase (E10: a late but correct finish must not fail)
    counterexamples: list[Counterexample] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)  # e.g. {"benchmark": "imitator", "task": "_049", "human_video": ...}
    notes: str = ""
    schema_version: str = SCHEMA_VERSION
    protocol: str = "v1"                 # evaluation protocol; "v2" = adapters/protocol_v2.py, "v3" = protocol_v3.py
    release: dict[str, Any] = field(default_factory=dict)   # v2/v3 release rule (ReleaseRule kwargs); {} = from the binding
    settle: dict[str, Any] = field(default_factory=dict)    # v3 settle block (protocol_v3.SettleConfig kwargs)
    proof_test: dict[str, Any] = field(default_factory=dict)  # 0.4: optional proof-test stage ({} = none)

    # ------------------------------------------------------------------ io
    @staticmethod
    def from_dict(d: dict, validate: bool = True, strict: bool = False) -> "TaskContract":
        """Build a contract from a parsed JSON dict. With validate=True, structural problems raise ContractError;
        soft problems (e.g. a metric name not in the local registry) are emitted as warnings, or raised if strict."""
        if validate:
            problems = validate_dict(d)
            errors = [p for p in problems if not p.startswith("warning:")]
            soft = [p for p in problems if p.startswith("warning:")]
            if errors or (strict and soft):
                raise ContractError(f"{d.get('task_id', '<no task_id>')}: " + "; ".join(errors + (soft if strict else [])))
            for p in soft:
                warnings.warn(f"{d.get('task_id')}: {p}", stacklevel=3)
        return TaskContract(
            task_id=d["task_id"], instruction=d["instruction"],
            outcome=[OutcomeTerm(**t) for t in d["outcome"]],
            require_released=d.get("require_released", True), settle_seconds=d.get("settle_seconds", 1.0),
            counterexamples=[Counterexample(**c) for c in d.get("counterexamples", [])],
            source=d.get("source", {}), notes=d.get("notes", ""),
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            protocol=str(d.get("protocol", "v1")), release=dict(d.get("release") or {}),
            settle=dict(d.get("settle") or {}), proof_test=dict(d.get("proof_test") or {}))

    @staticmethod
    def load(path: str, validate: bool = True, strict: bool = False) -> "TaskContract":
        with open(path, encoding="utf-8") as f:
            return TaskContract.from_dict(json.load(f), validate=validate, strict=strict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d.get("proof_test"):
            d.pop("proof_test", None)       # 0.4 field: omitted when empty, so older contracts round-trip unchanged
        # schema_version first for readability
        return {"schema_version": d.pop("schema_version"), **d}

    def save(self, path: str) -> None:
        problems = [p for p in validate_dict(self.to_dict()) if not p.startswith("warning:")]
        if problems:
            raise ContractError(f"{self.task_id}: refusing to save an invalid contract: " + "; ".join(problems))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, ensure_ascii=False)

    def validate(self) -> list[str]:
        """Problems with this contract ('warning:'-prefixed entries are soft). Empty list = valid."""
        return validate_dict(self.to_dict())

    # ------------------------------------------------------------------ verdict
    def verdict(self, measurements: dict[str, float], released: bool | None = None) -> dict[str, Any]:
        """Evaluate the contract on measured end-state quantities (from sim privileged state or real sensors).

        Returns the boolean verdict plus per-term margins, so callers can see *how close* an episode was.
        Missing measurements make the verdict 'unknown' rather than silently False/True.
        """
        terms, ok, unknown = [], True, False
        for t in self.outcome:
            if t.key not in measurements or measurements[t.key] is None:
                terms.append(dict(metric=t.key, value=None, holds=None, margin=None, norm_margin=None)); unknown = True; continue
            v = float(measurements[t.key]); h = t.holds(v)
            terms.append(dict(metric=t.key, value=v, holds=h, margin=t.margin(v), norm_margin=t.norm_margin(v)))
            ok &= h
        if self.require_released:
            if released is None:
                unknown = True
            else:
                ok &= bool(released)
        return dict(success=None if unknown else bool(ok), terms=terms, released=released,
                    min_norm_margin=min((x["norm_margin"] for x in terms if x.get("norm_margin") is not None), default=None),
                    binding_term=min((x for x in terms if x.get("norm_margin") is not None), key=lambda x: x["norm_margin"], default={}).get("metric"))


# ---------------------------------------------------------------------------------------------------- validation
_TERM_KEYS = {"metric", "op", "threshold", "units", "args", "real_measurement"}
_CEX_KEYS = {"name", "description", "expected", "generator"}
_TOP_KEYS = {"schema_version", "task_id", "instruction", "outcome", "require_released", "settle_seconds",
             "counterexamples", "source", "notes", "protocol", "release", "settle", "proof_test"}
_RELEASE_KEYS = {"contact_signals", "gap_signals", "gap_m", "max_impulse_Ns", "enclosure_signals", "enclosure_margin_m"}
_SETTLE_KEYS = {"hold", "max_seconds", "window_seconds", "quiet_steps", "quiet_max", "quiet_disp", "robot_speed_signal",
                "robot_speed_max"}


def _registry() -> dict:
    try:
        from ..outcomes.metrics import REGISTRY
        return REGISTRY
    except Exception:          # metrics module unavailable (e.g. numpy missing): skip the registry check
        return {}


def validate_dict(d: Any) -> list[str]:
    """Pure-Python structural validation (no jsonschema dependency). Mirrors spec/contract.schema.json.

    Errors make a contract unevaluable; 'warning:' entries flag things that are legal but suspicious
    (unknown metric names, missing real-world measurement description, no counterexamples)."""
    p: list[str] = []
    if not isinstance(d, dict):
        return ["contract must be a JSON object"]
    for k in ("task_id", "instruction", "outcome"):
        if k not in d:
            p.append(f"missing required field '{k}'")
    for k in set(d) - _TOP_KEYS:
        p.append(f"warning: unknown top-level field '{k}'")
    sv = str(d.get("schema_version", SCHEMA_VERSION))
    if sv not in SUPPORTED_SCHEMA_VERSIONS:
        p.append(f"unsupported schema_version '{sv}' (supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)})")
    proto = d.get("protocol", "v1")
    if proto not in PROTOCOLS:
        p.append(f"protocol '{proto}' not one of {PROTOCOLS}")
    rel = d.get("release", {})
    if not isinstance(rel, dict):
        p.append("release must be an object"); rel = {}
    for k in set(rel) - _RELEASE_KEYS:
        p.append(f"release: unknown field '{k}'")
    for k in ("contact_signals", "gap_signals", "enclosure_signals"):
        if k in rel and not (isinstance(rel[k], list) and all(isinstance(x, str) for x in rel[k])):
            p.append(f"release.{k} must be a list of signal names")
    for k in ("gap_m", "max_impulse_Ns", "enclosure_margin_m"):
        if k in rel and (not isinstance(rel[k], (int, float)) or isinstance(rel[k], bool) or rel[k] < 0):
            p.append(f"release.{k} must be a non-negative number")
    if proto == "v2" and d.get("require_released", True) and not rel:
        p.append("warning: protocol v2 with require_released but no 'release' block: the release rule must come from "
                 "the simulator binding")
    if proto == "v3" and d.get("require_released", True) and not (rel.get("contact_signals") and rel.get("enclosure_signals")):
        p.append("protocol v3 with require_released needs a release block with contact_signals and enclosure_signals")
    st = d.get("settle", {})
    if not isinstance(st, dict):
        p.append("settle must be an object"); st = {}
    for k in set(st) - _SETTLE_KEYS:
        p.append(f"settle: unknown field '{k}'")
    if st and proto != "v3":
        p.append("warning: a 'settle' block is only read by protocol v3")
    if proto == "v3" and not st:
        p.append("warning: protocol v3 without a 'settle' block: protocol defaults apply (hold A, 3 s cap, 1 s window, "
                 "no quiet signals except robot speed)")
    if "hold" in st and st["hold"] not in ("A", "B"):
        p.append("settle.hold must be 'A' or 'B'")
    for k in ("max_seconds", "window_seconds", "robot_speed_max"):
        if k in st and (not isinstance(st[k], (int, float)) or isinstance(st[k], bool) or st[k] <= 0):
            p.append(f"settle.{k} must be a positive number")
    if "quiet_steps" in st and (not isinstance(st["quiet_steps"], int) or isinstance(st["quiet_steps"], bool) or st["quiet_steps"] < 1):
        p.append("settle.quiet_steps must be a positive integer")
    if "quiet_max" in st and not (isinstance(st["quiet_max"], dict)
                                  and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in st["quiet_max"].values())):
        p.append("settle.quiet_max must map signal names to numbers")
    if "quiet_disp" in st and not (isinstance(st["quiet_disp"], list) and all(
            isinstance(g, dict) and isinstance(g.get("signals"), list) and g["signals"] and isinstance(g.get("max"), (int, float))
            for g in st["quiet_disp"])):
        p.append("settle.quiet_disp must be a list of {signals: [...], max: number}")
    if "task_id" in d and (not isinstance(d["task_id"], str) or not d["task_id"].strip()):
        p.append("task_id must be a non-empty string")
    if "instruction" in d and not isinstance(d["instruction"], str):
        p.append("instruction must be a string")
    for k, typ in (("require_released", bool), ("notes", str), ("source", dict)):
        if k in d and not isinstance(d[k], typ):
            p.append(f"{k} must be {typ.__name__}")
    if "settle_seconds" in d and (not isinstance(d["settle_seconds"], (int, float)) or isinstance(d["settle_seconds"], bool)
                                  or d["settle_seconds"] < 0):
        p.append("settle_seconds must be a non-negative number")

    outcome = d.get("outcome")
    if "outcome" in d:
        if not isinstance(outcome, list) or not outcome:
            p.append("outcome must be a non-empty list of terms")
            outcome = []
    reg, keys = _registry(), []
    for i, t in enumerate(outcome or []):
        where = f"outcome[{i}]"
        if not isinstance(t, dict):
            p.append(f"{where} must be an object"); continue
        for k in ("metric", "op", "threshold"):
            if k not in t:
                p.append(f"{where}: missing '{k}'")
        for k in set(t) - _TERM_KEYS:
            p.append(f"{where}: unknown field '{k}'")
        if t.get("op") is not None and t.get("op") not in OPS:
            p.append(f"{where}: op '{t.get('op')}' not one of {sorted(OPS)}")
        thr = t.get("threshold")
        if "threshold" in t and (not isinstance(thr, (int, float)) or isinstance(thr, bool) or not math.isfinite(thr)):
            p.append(f"{where}: threshold must be a finite number")
        args = t.get("args", {})
        if not isinstance(args, dict):
            p.append(f"{where}: args must be an object"); args = {}
        m = t.get("metric")
        if isinstance(m, str):
            if m.startswith("trace.") and "signal" not in args:
                p.append(f"{where}: trace metric '{m}' needs args.signal")
            if reg and m not in reg:
                p.append(f"warning: {where}: metric '{m}' is not in the local outcome registry")
            keys.append(f"{m}:{args['signal']}" if args.get("signal") else m)
        elif "metric" in t:
            p.append(f"{where}: metric must be a string")
        if m == "trace.final" and proto in ("v2", "v3"):
            p.append(f"{where}: trace.final (a mean) is not allowed under protocol {proto}; use trace.window_max / window_min")
        elif m == "trace.final" and t.get("op") in OPS:
            p.append(f"warning: {where}: trace.final averages the window, so a violation in the last few steps can be hidden; "
                     f"use trace.window_max (with < / <=) or trace.window_min (with > / >=) for end-state conditions")
        if not t.get("real_measurement"):
            p.append(f"warning: {where}: no real_measurement (how is this measured on hardware?)")
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        p.append(f"duplicate measurement keys {dup}: two terms would read the same measurement; "
                 f"give trace terms distinct signals or merge them")

    if "proof_test" in d:
        p += validate_proof_test(d.get("proof_test"), sv, proto, reg)

    cex = d.get("counterexamples", [])
    if not isinstance(cex, list):
        p.append("counterexamples must be a list"); cex = []
    names = []
    for i, c in enumerate(cex):
        where = f"counterexamples[{i}]"
        if not isinstance(c, dict):
            p.append(f"{where} must be an object"); continue
        for k in ("name", "description"):
            if k not in c:
                p.append(f"{where}: missing '{k}'")
        for k in set(c) - _CEX_KEYS:
            p.append(f"{where}: unknown field '{k}'")
        if "expected" in c and not isinstance(c["expected"], bool):
            p.append(f"{where}: expected must be true/false")
        names.append(c.get("name"))
    if len(names) != len(set(names)):
        p.append("counterexample names must be unique")
    if "outcome" in d and not cex:
        p.append("warning: no counterexamples: the contract has never been shown to reject anything")
    elif cex and not any(isinstance(c, dict) and c.get("expected") is False for c in cex):
        p.append("warning: no counterexample with expected=false")
    return p


# ---------------------------------------------------------------------------------------------------- proof test (0.4)
# Appended for card D2. The block is evaluated by adapters/protocol_v3.py (evaluate_proof_tests) on traces recorded by a
# simulator binding (imitator/verify/proof_test.py); structure only here.
PROOF_TEST_KINDS = {
    # kind:        required numeric params (all other params are free-form and passed to the binding)
    "lift_tilt":   ("lift_m", "tilt_deg", "lift_seconds", "tilt_seconds", "hold_seconds"),
    "shake":       ("amplitude_m", "freq_hz", "duration_seconds"),
    "push":        ("delta_v_mps",),
    "pull":        ("force_N", "duration_seconds"),
    "table_shake": ("accel_mps2", "freq_hz", "duration_seconds"),
}
_PT_KEYS = {"robot", "pre_settle_seconds", "window_seconds", "tests", "notes"}
_PT_TEST_KEYS = {"name", "kind", "target", "params", "rest_seconds", "terms", "description"}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_proof_test(pt: Any, schema_version: str = SCHEMA_VERSION, protocol: str = "v3", reg: dict | None = None
                        ) -> list[str]:
    """Problems with a proof_test block ('warning:'-prefixed entries are soft)."""
    p: list[str] = []
    if not isinstance(pt, dict):
        return ["proof_test must be an object"]
    if not pt:
        return p
    if schema_version in ("0.1", "0.2", "0.3"):
        p.append(f"warning: proof_test is a schema-0.4 field (contract declares {schema_version})")
    if protocol != "v3":
        p.append("proof_test needs protocol v3 (it runs after the v3 judged window and the retreat)")
    for k in set(pt) - _PT_KEYS:
        p.append(f"proof_test: unknown field '{k}'")
    if pt.get("robot", "clear") != "clear":
        p.append("proof_test.robot must be 'clear' (the evaluator moves the robot out of the workspace before the tests)")
    for k in ("pre_settle_seconds", "window_seconds"):
        if k in pt and (not _num(pt[k]) or pt[k] < 0 or (k == "window_seconds" and pt[k] <= 0)):
            p.append(f"proof_test.{k} must be a {'positive' if k == 'window_seconds' else 'non-negative'} number")
    tests = pt.get("tests")
    if not isinstance(tests, list) or not tests:
        return p + ["proof_test.tests must be a non-empty list"]
    names = []
    for i, t in enumerate(tests):
        w = f"proof_test.tests[{i}]"
        if not isinstance(t, dict):
            p.append(f"{w} must be an object"); continue
        for k in set(t) - _PT_TEST_KEYS:
            p.append(f"{w}: unknown field '{k}'")
        name = t.get("name")
        if not isinstance(name, str) or not name or ":" in name:
            p.append(f"{w}: name must be a non-empty string without ':'")
        names.append(name)
        kind = t.get("kind")
        if kind not in PROOF_TEST_KINDS:
            p.append(f"{w}: kind '{kind}' not one of {sorted(PROOF_TEST_KINDS)}")
        if not isinstance(t.get("target"), str) or not t.get("target"):
            p.append(f"{w}: target must name a scene object")
        params = t.get("params", {})
        if not isinstance(params, dict):
            p.append(f"{w}: params must be an object"); params = {}
        for k in PROOF_TEST_KINDS.get(kind, ()):
            if not _num(params.get(k)):
                p.append(f"{w}: params.{k} must be a number")
        if "rest_seconds" in t and (not _num(t["rest_seconds"]) or t["rest_seconds"] <= 0):
            p.append(f"{w}: rest_seconds must be a positive number")
        terms = t.get("terms")
        if not isinstance(terms, list) or not terms:
            p.append(f"{w}: terms must be a non-empty list of outcome terms"); continue
        keys = []
        for j, term in enumerate(terms):
            ww = f"{w}.terms[{j}]"
            if not isinstance(term, dict):
                p.append(f"{ww} must be an object"); continue
            for k in ("metric", "op", "threshold"):
                if k not in term:
                    p.append(f"{ww}: missing '{k}'")
            for k in set(term) - _TERM_KEYS:
                p.append(f"{ww}: unknown field '{k}'")
            if term.get("op") is not None and term.get("op") not in OPS:
                p.append(f"{ww}: op '{term.get('op')}' not one of {sorted(OPS)}")
            if "threshold" in term and not _num(term["threshold"]):
                p.append(f"{ww}: threshold must be a finite number")
            a = term.get("args", {})
            if not isinstance(a, dict):
                p.append(f"{ww}: args must be an object"); a = {}
            m = term.get("metric")
            if not isinstance(m, str):
                continue
            if not m.startswith("trace."):
                p.append(f"{ww}: proof-test terms must be trace metrics (judged on the test's own traces)")
            elif "signal" not in a:
                p.append(f"{ww}: trace metric '{m}' needs args.signal")
            if m == "trace.final":
                p.append(f"{ww}: trace.final (a mean) is not allowed; use window_max / window_min")
            if reg and m not in reg:
                p.append(f"warning: {ww}: metric '{m}' is not in the local outcome registry")
            keys.append(f"{m}:{a['signal']}" if a.get("signal") else m)
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            p.append(f"{w}: duplicate measurement keys {dup}")
    if len([n for n in names if n is not None]) != len(set(n for n in names if n is not None)):
        p.append("proof_test test names must be unique")
    return p
