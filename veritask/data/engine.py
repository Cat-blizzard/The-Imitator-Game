"""Data engine v0: perturb -> correct -> certify.

A *source* rollout (scripted expert, or a degraded "weak" policy) produces restorable *snapshots*. *Perturbers*
propose small programs of primitive operations that knock the scene off the nominal path. *Correctors* (layered:
replay the expert's remaining stages, stage-aware replanning driven by the contract, GPU motion generation) try to
recover. A *certifier* judges every resulting episode with the task's physical outcome contract AND records the
benchmark's own success flag, so the ledger can count false recoveries (bench says success, contract says fail).

Everything in this module is simulator-agnostic (numpy only). Simulator bindings implement `Session` (see
data/maniskill/session.py); the protocols below are what the engine relies on.

Conventions
- One *episode* = source prefix (reset .. snapshot) + perturbation segment + correction segment + settle hold.
  Certification is on the whole episode, exactly what an evaluator would see; segment ids are stored per step so a
  learner can keep only the corrective part (segment 2).
- One *attempt* = (snapshot, perturbation, corrector). Every corrector sees the same post-perturbation state
  (the engine snapshots after the perturbation and restores it for each corrector), so layers are paired.
"""
from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

SEG_PREFIX, SEG_PERTURB, SEG_CORRECT, SEG_SETTLE = 0, 1, 2, 3


# ================================================================================================= data types
@dataclass
class Snapshot:
    """Restorable simulator state plus metadata.

    task, seed   : task id (contract task_id) and reset seed
    step         : control steps since reset at which the snapshot was taken
    stage        : stage-graph label the source policy was about to execute ("done" = after the last stage)
    source       : which rollout produced it ("expert", "weak", "expert_press", ... or "<src>+<perturbation>")
    sim_state    : simulator state as nested dicts of numpy arrays (ManiSkill env.get_state_dict())
    task_state   : task-internal state that is not part of the sim state but changes the benchmark flag or the scene
                   (e.g. PressJuicer's latched juicer_ever_pressed, poses of static actors)
    ctl          : executor state (per-arm commanded joint targets and gripper commands)
    traces       : TraceRecorder traces of the episode so far (certification is over the whole episode)
    log          : per-step episode log so far (actions, qpos, object poses, segment ids, stage ids)
    ctx          : skill context computed by earlier stages (e.g. the grasp pose chosen before descending)
    meta         : free-form (source success flags, weak-policy noise, perturbation spec, ...)
    """
    task: str
    seed: int
    step: int
    stage: str
    source: str
    sim_state: dict = field(default_factory=dict, repr=False)
    task_state: dict = field(default_factory=dict, repr=False)
    ctl: dict = field(default_factory=dict, repr=False)
    traces: dict = field(default_factory=dict, repr=False)
    log: dict = field(default_factory=dict, repr=False)
    ctx: dict = field(default_factory=dict, repr=False)
    meta: dict = field(default_factory=dict)

    def fork(self) -> "Snapshot":
        return copy.deepcopy(self)

    @property
    def id(self) -> str:
        return f"{self.task}/s{self.seed}/{self.source}/t{self.step}/{self.stage}"


# Primitive operations a perturbation program may use, with argument bounds. A Session implements each op as a
# method of the same name. Programs proposed by a model are validated against this table before they touch the sim
# ("AI proposes, physics disposes"): unknown ops or out-of-range arguments are rejected, never clipped silently.
PRIMITIVES: dict[str, dict[str, tuple[float, float]]] = {
    "hold":         {"steps": (1, 80)},
    "set_gripper":  {"arm": (0, 1), "cmd": (-1, 1), "steps": (1, 40)},
    "joint_noise":  {"arm": (0, 1), "sigma": (0.0, 0.3), "steps": (1, 40), "seed": (0, 2**31 - 1)},
    "shove":        {"vx": (-1.5, 1.5), "vy": (-1.5, 1.5), "vz": (-0.5, 1.0), "wx": (-10, 10), "wy": (-10, 10),
                     "wz": (-10, 10), "settle": (0, 60)},
    "offset_pose":  {"dx": (-0.12, 0.12), "dy": (-0.12, 0.12), "dz": (-0.02, 0.05), "dyaw_deg": (-45, 45),
                     "settle": (0, 60)},
    "move_tcp_rel": {"arm": (0, 1), "dx": (-0.15, 0.15), "dy": (-0.15, 0.15), "dz": (-0.15, 0.15)},
}
OBJECT_OPS = {"shove", "offset_pose"}  # these take an extra string argument 'obj'


def validate_ops(ops: Sequence[dict], objects: Iterable[str] | None = None) -> list[str]:
    """Return a list of problems (empty = valid). Each op is {"op": name, "args": {...}}."""
    errs, objects = [], (set(objects) if objects is not None else None)
    for i, o in enumerate(ops):
        name, args = o.get("op"), dict(o.get("args", {}))
        if name not in PRIMITIVES:
            errs.append(f"op[{i}]: unknown primitive {name!r}"); continue
        if name in OBJECT_OPS:
            obj = args.pop("obj", None)
            if not isinstance(obj, str):
                errs.append(f"op[{i}] {name}: missing 'obj'")
            elif objects is not None and obj not in objects:
                errs.append(f"op[{i}] {name}: unknown object {obj!r} (have {sorted(objects)})")
        for k, v in args.items():
            if k not in PRIMITIVES[name]:
                errs.append(f"op[{i}] {name}: unknown argument {k!r}"); continue
            lo, hi = PRIMITIVES[name][k]
            if not isinstance(v, (int, float)) or not (lo <= v <= hi):
                errs.append(f"op[{i}] {name}.{k}={v!r} outside [{lo}, {hi}]")
    return errs


@dataclass
class Perturbation:
    """Something applicable to the env state: a short program of PRIMITIVES ops.

    kind   : 'joint_noise' | 'shove' | 'drop' | 'target_offset' | 'weak_policy' | 'program:<name>'
    family : 'action' | 'object' | 'weak' | 'semantic'
    ops    : [{"op": "shove", "args": {"obj": "cup_1", "vx": 0.4, ...}}, ...] executed in order
    params : the sampled parameters (for the JSONL index)
    An empty ops list is the identity (used by the weak-policy source: its own failure state is the perturbation).
    """
    kind: str
    family: str
    ops: list[dict] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def apply(self, sim: "Session") -> dict:
        errs = validate_ops(self.ops, sim.objects())
        if errs:
            raise ValueError("invalid perturbation program: " + "; ".join(errs))
        t0, s0 = time.time(), sim.step_count
        for o in self.ops:
            getattr(sim, o["op"])(**o.get("args", {}))
        return dict(steps=sim.step_count - s0, wall_s=round(time.time() - t0, 3))

    def spec(self) -> dict:
        return dict(kind=self.kind, family=self.family, params=self.params, ops=self.ops)


@dataclass
class Trajectory:
    """What a corrector executed. The per-step arrays live in the session log (segment SEG_CORRECT); this object
    carries the provenance: layer, stages run, every motion-planning query with backend/status/latency."""
    corrector: str
    layer: str
    start_step: int
    end_step: int
    stages: list[str] = field(default_factory=list)
    plan_calls: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return self.end_step - self.start_step


@dataclass
class Verdict:
    """Certifier output. category: 'certified' (contract pass and bench pass), 'contract_only' (contract pass,
    bench fail; also certified), 'false_recovery' (bench pass, contract fail), 'both_fail', 'unknown' (a contract
    measurement was missing). `certified` = contract_success is True."""
    contract_success: bool | None
    bench_success: bool | None
    bench_success_any: bool | None
    released: bool | None
    terms: list[dict]
    min_norm_margin: float | None
    binding_term: str | None

    @property
    def certified(self) -> bool:
        return self.contract_success is True

    @property
    def category(self) -> str:
        c, b = self.contract_success, self.bench_success
        if c is None:
            return "unknown"
        if c:
            return "certified" if b else "contract_only"
        return "false_recovery" if b else "both_fail"

    def to_json(self) -> dict:
        return dict(category=self.category, contract_success=self.contract_success, bench_success=self.bench_success,
                    bench_success_any=self.bench_success_any, released=self.released,
                    min_norm_margin=None if self.min_norm_margin is None else round(self.min_norm_margin, 5),
                    binding_term=self.binding_term,
                    terms=[dict(metric=t["metric"], value=None if t["value"] is None else round(float(t["value"]), 5),
                                holds=t["holds"], norm_margin=None if t.get("norm_margin") is None else round(t["norm_margin"], 4))
                           for t in self.terms])


# ================================================================================================= protocols
@runtime_checkable
class Session(Protocol):
    """Simulator binding used by the engine, perturbers and correctors (implemented in data/maniskill/session.py).
    Besides the methods below it implements every PRIMITIVES op as a method of the same name."""
    task_id: str
    step_count: int
    traces: dict

    def reset(self, seed: int) -> None: ...
    def snapshot(self, stage: str, source: str, **meta) -> Snapshot: ...
    def restore(self, snap: Snapshot) -> None: ...
    def objects(self) -> list[str]: ...
    def holding(self) -> dict[int, str | None]: ...
    def settle(self, steps: int) -> None: ...
    def episode_arrays(self) -> dict[str, np.ndarray]: ...


@runtime_checkable
class Perturber(Protocol):
    name: str

    def propose(self, snapshot: Snapshot, rng: np.random.Generator) -> list[Perturbation]: ...


@runtime_checkable
class Corrector(Protocol):
    """correct() executes in `env` (a Session restored to the post-perturbation state) and returns the Trajectory,
    or None if it could not (planning failure) or would not (contract says the episode is unrecoverable).
    The reason for None is left in `last_failure` ('abstain:...' or 'fail:...')."""
    name: str
    layer: str
    last_failure: str | None

    def correct(self, env: Session, snapshot: Snapshot, stage: str | None) -> Trajectory | None: ...


@runtime_checkable
class Certifier(Protocol):
    def certify(self, trace: dict) -> Verdict: ...


@runtime_checkable
class Source(Protocol):
    """Produces snapshots for one seed. A weak-policy source also returns the identity perturbation to use."""
    name: str

    def rollout(self, env: Session, seed: int, rng: np.random.Generator) -> list[Snapshot]: ...


# ================================================================================================= certifier
class ContractCertifier:
    """Wraps a TaskContract plus the benchmark's own success flag recorded in the same traces.

    trace: dict of per-step signals (TraceRecorder.traces). Contract trace.* terms are measured with the outcome
    metric registry; 'released' follows TraceRecorder.evaluate (mean of the last 10 steps of `released_signal`
    > 0.5); the bench flag is `bench_signal` at the final step (and 'any' over the episode, for latching benches).
    """

    def __init__(self, contract, released_signal: str | None = "released", bench_signal: str = "bench_success"):
        self.contract, self.released_signal, self.bench_signal = contract, released_signal, bench_signal

    def measurements(self, trace: dict) -> dict[str, float]:
        from ..outcomes.metrics import measure
        return {t.key: measure(t.metric, traces=trace, **t.args) for t in self.contract.outcome if t.metric.startswith("trace.")}

    def certify(self, trace: dict, extra_measurements: dict | None = None) -> Verdict:
        m = self.measurements(trace); m.update(extra_measurements or {})
        rel = trace.get(self.released_signal) if self.released_signal else None
        released = bool(np.mean(rel[-10:]) > 0.5) if rel is not None and len(rel) else None
        v = self.contract.verdict(m, released=released)
        bs = trace.get(self.bench_signal)
        bench = bool(bs[-1]) if bs is not None and len(bs) else None
        bench_any = bool(np.any(bs)) if bs is not None and len(bs) else None
        return Verdict(contract_success=v["success"], bench_success=bench, bench_success_any=bench_any,
                       released=v["released"], terms=v["terms"], min_norm_margin=v["min_norm_margin"],
                       binding_term=v["binding_term"])


# ================================================================================================= progress graph
def term_monotonicity(term) -> str:
    """How a contract term can evolve as the episode continues, derived from the metric and the comparison only.

    'latch_true'  : once satisfied it stays satisfied (trace.min <=, trace.max >=, trace.any / count_true >=)
    'latch_false' : once violated it can never be satisfied again (count_true <= / <, trace.max <=, trace.min >=)
    'state'       : depends on the current state (trace.final, end-state metrics)
    The corrector uses this to (a) skip stages whose latched terms already hold and (b) abstain early when an
    invariant is already violated (no correction can certify the episode, so do not spend planning time on it).
    """
    m, op = term.metric, term.op
    up = op in (">", ">=")
    if m in ("trace.count_true", "trace.any", "trace.max"):   # non-decreasing over time
        return "latch_true" if up else "latch_false"
    if m == "trace.min":                                       # non-increasing over time
        return "latch_false" if up else "latch_true"
    return "state"


def contract_progress(contract, traces: dict) -> dict[str, dict]:
    """Evaluate every trace.* contract term on the episode *so far*, as a progress signal.

    trace.final terms use the latest value (window=1) instead of the settled-window mean, so the result reflects
    the current state. Returns {term_key: {value, holds, mono, dead}} where dead = violated and latch_false.
    """
    from ..outcomes.metrics import measure
    out = {}
    for t in contract.outcome:
        if not t.metric.startswith("trace.") or not traces.get(t.args.get("signal"), None):
            continue
        args = dict(t.args)
        if t.metric == "trace.final":
            args["window"] = 1
        v = measure(t.metric, traces=traces, **args); h = t.holds(v); mono = term_monotonicity(t)
        out[t.key] = dict(value=v, holds=bool(h), mono=mono, dead=(mono == "latch_false" and not h))
    return out


@dataclass
class ProgressNode:
    """One node of a task's progress graph. A node is met when its contract term holds (term_key, evaluated by
    contract_progress with the contract's own threshold) and/or its auxiliary predicate holds (aux: name of a
    boolean the task computes from privileged state, e.g. 'peg_grasped'). `skill` = stage to resume from when this
    is the earliest unmet node."""
    name: str
    skill: str
    term_key: str | None = None
    aux: str | None = None
    negate_aux: bool = False
    or_aux: str | None = None   # node also counts as met while this predicate holds (e.g. tilt is moot while held)

    def met(self, progress: dict, aux: dict) -> bool | None:
        if self.or_aux is not None and aux.get(self.or_aux):
            return True
        ok = True
        if self.term_key is not None:
            if self.term_key not in progress:
                return None
            ok &= progress[self.term_key]["holds"]
        if self.aux is not None:
            a = bool(aux.get(self.aux))
            ok &= (not a) if self.negate_aux else a
        return bool(ok)


@dataclass
class ProgressGraph:
    """Ordered list of ProgressNodes derived from the contract (+ a few auxiliary predicates).

    infer() returns, in this order of precedence:
      abstain   if a latch_false contract term is already violated (no continuation can certify the episode)
      'done'    if every trace.* contract term holds now and every goal_aux predicate holds
      <skill>   of the earliest unmet node (a skill name starting with 'no_skill' = no skill available)
      'settle'  if all nodes are met but the goal is not yet (e.g. objects still moving)
    """
    nodes: list[ProgressNode]
    goal_aux: list[str] = field(default_factory=list)

    def infer(self, contract, traces: dict, aux: dict) -> dict:
        prog = contract_progress(contract, traces)
        dead = [k for k, p in prog.items() if p["dead"]]
        if dead:
            return dict(stage=None, abstain=True, reason="invariant violated: " + ",".join(dead), progress=prog, unmet=[])
        n_trace = sum(t.metric.startswith("trace.") for t in contract.outcome)
        goal = len(prog) == n_trace and all(p["holds"] for p in prog.values()) and all(aux.get(a) for a in self.goal_aux)
        unmet = [n.name for n in self.nodes if not n.met(prog, aux)]
        if goal:
            return dict(stage="done", node=None, abstain=False, reason="", progress=prog, unmet=unmet)
        first = next((n for n in self.nodes if not n.met(prog, aux)), None)
        return dict(stage=first.skill if first else "settle", node=first.name if first else None, abstain=False,
                    reason="", progress=prog, unmet=unmet)


# ================================================================================================= ledger
LEDGER_FIELDS = ("attempts", "corrector_fail", "abstain", "executed", "bench_accept", "certified", "false_recovery",
                 "contract_only", "both_fail", "unknown")


class Ledger:
    """Counts per (task, perturbation kind, corrector) and overall.

    attempts       : (snapshot, perturbation, corrector) triples tried
    corrector_fail : corrector returned None for a planning/execution reason
    abstain        : corrector returned None because the contract says the episode is unrecoverable
    executed       : corrector returned a trajectory (episode certified/judged)
    bench_accept   : benchmark success flag at the end of the episode (what a bench-gated pipeline would keep)
    certified      : contract success (what this engine keeps)
    false_recovery : bench success AND contract fail (bench-only accepts)
    contract_only  : contract success AND bench fail
    """

    def __init__(self):
        self.rows: list[dict] = []
        self.c = defaultdict(lambda: {k: 0 for k in LEDGER_FIELDS} | dict(wall_s=0.0))

    def add(self, key: tuple, outcome: str, category: str | None, bench: bool | None, wall_s: float):
        for k in (key, ("ALL",) + key[:1], ("ALL",)):
            d = self.c[k]; d["attempts"] += 1; d["wall_s"] += wall_s
            if outcome == "abstain":
                d["abstain"] += 1
            elif outcome == "fail":
                d["corrector_fail"] += 1
            else:
                d["executed"] += 1; d["bench_accept"] += int(bool(bench))
                if category == "certified":
                    d["certified"] += 1
                elif category == "contract_only":
                    d["certified"] += 1; d["contract_only"] += 1
                elif category == "false_recovery":
                    d["false_recovery"] += 1
                elif category == "both_fail":
                    d["both_fail"] += 1
                else:
                    d["unknown"] += 1

    def summary(self) -> dict:
        out = {}
        for k, d in self.c.items():
            n, ex = d["attempts"], d["executed"]
            out["|".join(map(str, k))] = dict(d, wall_s=round(d["wall_s"], 1),
                                              certified_rate=round(d["certified"] / n, 4) if n else None,
                                              bench_rate=round(d["bench_accept"] / n, 4) if n else None,
                                              false_recovery_rate_of_bench=round(d["false_recovery"] / d["bench_accept"], 4) if d["bench_accept"] else None,
                                              wall_s_per_certified=round(d["wall_s"] / d["certified"], 2) if d["certified"] else None)
        return out


# ================================================================================================= engine
class DataEngine:
    """Orchestrates source rollout -> snapshots -> perturb -> correct -> certify -> write.

    env         : Session (one simulator instance; the engine resets/restores it)
    sources     : list of Source (expert / weak policy); each returns snapshots for a seed
    perturbers  : list of Perturber; applied to expert snapshots (weak-policy snapshots carry their own identity
                  perturbation in snapshot.meta['perturbation'])
    correctors  : list of Corrector; each attempt is run once per corrector from the same post-perturbation state
    certifier   : ContractCertifier
    writer      : EpisodeWriter-like object with write(env, snap, pert, traj, verdict, row) -> path | None
    settle_steps: identical hold appended after every corrector (so trace.final windows see a settled scene)
    snapshot_filter(snap) -> bool : which snapshots to keep
    per_perturber: proposals sampled per perturber per seed (over all expert snapshots of that seed)
    """

    def __init__(self, env: Session, sources, perturbers, correctors, certifier, writer=None, settle_steps: int = 25,
                 snapshot_filter: Callable[[Snapshot], bool] | None = None, per_perturber: int = 2,
                 log: Callable[[dict], None] | None = None):
        self.env, self.sources, self.perturbers, self.correctors = env, sources, perturbers, correctors
        self.certifier, self.writer, self.settle_steps = certifier, writer, settle_steps
        self.snapshot_filter = snapshot_filter or (lambda s: True)
        self.per_perturber = per_perturber
        self.ledger, self.log = Ledger(), log or (lambda r: None)
        self.timing = defaultdict(float)

    # ------------------------------------------------------------------------------------------ helpers
    def _attempts_for(self, snaps: list[Snapshot], rng) -> list[tuple[Snapshot, Perturbation]]:
        """Weak-policy snapshots carry their own (identity) perturbation. For expert snapshots, each perturber
        proposes over all snapshots and `per_perturber` proposals are sampled per seed (balanced design)."""
        out = []
        for s in snaps:
            if "perturbation" in s.meta:
                p = s.meta["perturbation"]
                out.append((s, p if isinstance(p, Perturbation) else Perturbation(**p)))
        plain = [s for s in snaps if "perturbation" not in s.meta]
        for pz in self.perturbers:
            props = [(s, p) for s in plain for p in pz.propose(s, rng)]
            if props:
                idx = rng.permutation(len(props))[: self.per_perturber]
                out += [props[i] for i in sorted(idx)]
        return out

    def run_seed(self, seed: int, rng: np.random.Generator) -> list[dict]:
        rows = []
        for src in self.sources:
            t0 = time.time()
            snaps = [s for s in src.rollout(self.env, seed, rng) if self.snapshot_filter(s)]
            self.timing[f"source:{src.name}"] += time.time() - t0
            last = dict(getattr(src, "last", {}) or {})
            self.log(dict(kind="source", task=self.env.task_id, n_snapshots=len(snaps), wall_s=round(time.time() - t0, 2), **last))
            for snap, pert in self._attempts_for(snaps, rng):
                rows += self.run_attempt(snap, pert)
        return rows

    def run_attempt(self, snap: Snapshot, pert: Perturbation) -> list[dict]:
        env, rows = self.env, []
        t0 = time.time()
        env.restore(snap)
        env.begin_segment(SEG_PERTURB)
        try:
            pinfo = pert.apply(env)
        except Exception as e:  # invalid program or sim error: count once per corrector as corrector-independent failure
            pinfo = dict(error=repr(e)[:300])
        post = env.snapshot(stage=snap.stage, source=f"{snap.source}+{pert.kind}", parent=snap.id)
        t_pert = time.time() - t0
        for corr in self.correctors:
            t1 = time.time()
            env.restore(post)
            env.begin_segment(SEG_CORRECT)
            traj, err = None, None
            if "error" not in pinfo:
                try:
                    traj = corr.correct(env, post, snap.stage)
                except Exception as e:
                    import traceback
                    err = traceback.format_exc()[-600:]
                    corr.last_failure = "fail:exception"
            reason = None if traj is not None else (corr.last_failure or ("fail:perturbation_error" if "error" in pinfo else "fail:unknown"))
            outcome = "executed" if traj is not None else ("abstain" if reason.startswith("abstain") else "fail")
            verdict = None
            if traj is not None:
                env.begin_segment(SEG_SETTLE)
                env.settle(self.settle_steps)
                verdict = self.certifier.certify(env.traces)
            wall = time.time() - t1 + t_pert
            row = dict(kind="attempt", task=snap.task, seed=snap.seed, snapshot=snap.id, source=snap.source, stage=snap.stage,
                       snap_step=snap.step, perturbation=pert.spec(), perturb_info=pinfo, corrector=corr.name,
                       layer=corr.layer, outcome=outcome, reason=reason, error=err,
                       verdict=verdict.to_json() if verdict else None,
                       correction=None if traj is None else dict(stages=traj.stages, n_steps=traj.n_steps,
                                                                  n_plan_calls=len(traj.plan_calls),
                                                                  plan_fail=sum(not c.get("ok", True) for c in traj.plan_calls),
                                                                  plan_ms=round(sum(c.get("ms", 0) for c in traj.plan_calls), 1),
                                                                  info=traj.info),
                       episode_steps=env.step_count, wall_s=round(wall, 3))
            if self.writer is not None and traj is not None:
                row["file"] = self.writer.write(env, snap, pert, traj, verdict, row)
            row["plan_calls"] = traj.plan_calls if traj is not None else getattr(corr, "last_plan_calls", [])
            self.ledger.add((snap.task, pert.kind, corr.name), outcome, verdict.category if verdict else None,
                            verdict.bench_success if verdict else None, wall)
            self.log(row); rows.append(row)
        return rows

    def run(self, seeds: Sequence[int], rng_seed: int = 0) -> Ledger:
        for seed in seeds:
            self.run_seed(seed, np.random.default_rng([rng_seed, seed]))
        return self.ledger
