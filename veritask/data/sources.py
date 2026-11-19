"""Snapshot sources: the scripted expert and a degraded 'weak' policy (scripted expert with noisy waypoints and
random early/late gripper actions). Both run a task's stage plan through Session.run_plan(); snapshots are taken at
stage boundaries (before each stage) and after the last stage ('done').

Session.run_plan(plan, start_stage=None, ctx=None, noise=None, rng=None, strict=False, on_boundary=None,
                 until_segment_end=False) -> dict(ok, stages, failed_stage, faults)
is provided by the simulator binding (data/maniskill/session.py); stage functions live with the task.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .engine import Perturbation, Snapshot


@dataclass
class ExpertSource:
    """Nominal scripted expert; every stage boundary becomes a snapshot (stage = stage about to run)."""
    plan: str = "expert"
    name: str = "expert"
    settle_steps: int = 25
    certifier: object = None
    last: dict = field(default_factory=dict)

    def rollout(self, env, seed: int, rng) -> list[Snapshot]:
        env.reset(seed)
        snaps = []
        res = env.run_plan(self.plan, on_boundary=lambda st: snaps.append(env.snapshot(stage=st, source=self.name)))
        snaps.append(env.snapshot(stage="done", source=self.name))
        env.settle(self.settle_steps)
        v = self.certifier.certify(env.traces) if self.certifier else None
        self.last = dict(seed=seed, source=self.name, plan=self.plan, ok=res["ok"], steps=env.step_count,
                         verdict=v.to_json() if v else None, plan_fail=sum(not c.get("ok", True) for c in env.plan_calls))
        for s in snaps:
            s.meta["source_verdict"] = None if v is None else v.category
        return snaps


@dataclass
class WeakPolicySource:
    """Scripted expert with noise: waypoint position noise (std `waypoint_sigma` m on grasp/place/press targets),
    random early gripper actions (close before reaching / open while carrying, prob p_early) and late ones
    (close after starting to lift / retreat before opening, prob p_late). The rollout's own failure states become
    snapshots: if the rollout fails the contract (only_failures=True), up to `max_snaps` snapshots are returned:
    the final state plus boundary snapshots taken after the first injected fault, each tagged with the identity
    perturbation kind 'weak_policy' (the weak policy's own deviation is the perturbation)."""
    plan: str = "expert"
    name: str = "weak"
    waypoint_sigma: float = 0.012
    p_early: float = 0.25
    p_late: float = 0.25
    only_failures: bool = True
    max_snaps: int = 2
    settle_steps: int = 25
    certifier: object = None
    extra: dict = field(default_factory=dict)   # task-specific fault probabilities, e.g. {"p_shortcut": 0.5}
    last: dict = field(default_factory=dict)

    def noise(self) -> dict:
        return dict(waypoint_sigma=self.waypoint_sigma, p_early=self.p_early, p_late=self.p_late, **self.extra)

    def rollout(self, env, seed: int, rng) -> list[Snapshot]:
        env.reset(seed)
        snaps = []
        res = env.run_plan(self.plan, noise=self.noise(), rng=rng,
                           on_boundary=lambda st: snaps.append(env.snapshot(stage=st, source=self.name)))
        env.settle(self.settle_steps)
        final = env.snapshot(stage="done", source=self.name)
        v = self.certifier.certify(env.traces) if self.certifier else None
        faults = res.get("faults", [])
        self.last = dict(seed=seed, source=self.name, plan=self.plan, ok=res["ok"], steps=env.step_count, faults=faults,
                         verdict=v.to_json() if v else None)
        if self.only_failures and v is not None and v.contract_success:
            return []
        first = min((f["step"] for f in faults), default=None)
        mids = [s for s in snaps if first is not None and s.step > first]
        pick = [final]
        if mids:
            k = min(self.max_snaps - 1, len(mids))
            pick = [mids[i] for i in sorted(rng.choice(len(mids), k, replace=False))] + pick
        pert = Perturbation("weak_policy", "weak", [], params=dict(noise=self.noise(), faults=faults,
                                                                     source_verdict=None if v is None else v.category))
        for s in pick:
            s.meta["perturbation"] = pert
            s.meta["source_verdict"] = None if v is None else v.category
        return pick
