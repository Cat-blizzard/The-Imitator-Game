"""Layered correctors.

L0  NullCorrector               hold still (baseline: what the perturbation alone does to bench and contract)
L1  ReplayStagesCorrector       resume the scripted expert at the snapshot's stage label, fresh mplib plans from the
                                perturbed state, planning failures ignored like the official solution scripts
                                (the SimPT-style "resume from the progress snapshot" baseline)
L2  StageAwareCorrector         infer the stage from the contract (ProgressGraph over contract terms + privileged
                                probes), abstain if a latch_false term is already violated, resume from the
                                earliest unmet node, run to the end of that plan segment, re-infer; strict planning
L3  CuroboStageAwareCorrector   L2 with cuRobo (GPU, collision-aware) as the motion backend
--  SamplingMPCCorrector        Newton sampling-MPC for soft bodies (interface stub)

Session requirements (beyond engine.Session): run_plan(...), progress() -> ProgressGraph.infer(...) dict,
hold(n), set_backend(name) -> previous name, plan_calls (list of planning-query records).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .engine import Snapshot, Trajectory


class _Base:
    name = "base"; layer = "L?"

    def __init__(self):
        self.last_failure = None; self.last_plan_calls = []

    def _begin(self, env):
        self.last_failure = None; self._s0 = env.step_count; self._c0 = len(env.plan_calls)

    def _traj(self, env, stages, **info) -> Trajectory:
        calls = env.plan_calls[self._c0:]
        self.last_plan_calls = calls
        return Trajectory(corrector=self.name, layer=self.layer, start_step=self._s0, end_step=env.step_count,
                          stages=list(stages), plan_calls=list(calls), info=info)

    def _fail(self, env, reason):
        self.last_failure = reason; self.last_plan_calls = env.plan_calls[self._c0:]
        return None


class NullCorrector(_Base):
    name, layer = "L0_hold", "L0"

    def correct(self, env, snapshot: Snapshot, stage):
        self._begin(env)
        return self._traj(env, [])


class ReplayStagesCorrector(_Base):
    """L1: resume plan `plan` at `stage` (or at its first stage if the label is not in this plan)."""
    layer = "L1"

    def __init__(self, plan: str = "expert", name: str | None = None):
        super().__init__(); self.plan = plan; self.name = name or f"L1_replay_{plan}"

    def correct(self, env, snapshot: Snapshot, stage):
        self._begin(env)
        stages = env.plan_stages(self.plan)
        if stage == "done":
            return self._traj(env, [], note="snapshot after the last stage: nothing to resume")
        start = stage if stage in stages else stages[0]
        res = env.run_plan(self.plan, start_stage=start, ctx=dict(snapshot.ctx), strict=False)
        return self._traj(env, res["stages"], start=start, plan_failed_stages=res.get("failed", []))


class StageAwareCorrector(_Base):
    """L2: contract-driven stage inference + strict replanning, up to `max_rounds` infer/execute rounds."""
    layer = "L2"
    backend = "mplib"

    def __init__(self, plan: str = "recover", max_rounds: int = 4, settle_first: int = 10, name: str | None = None):
        super().__init__(); self.plan, self.max_rounds, self.settle_first = plan, max_rounds, settle_first
        self.name = name or f"{self.layer}_stage_{self.backend}"

    def correct(self, env, snapshot: Snapshot, stage):
        self._begin(env)
        prev = env.set_backend(self.backend)
        try:
            return self._run(env)
        finally:
            env.set_backend(prev)

    def _run(self, env):
        ran, rounds = [], []
        if self.settle_first:  # let transients die out (a shoved cup rocking reads as tilted) before reading the contract
            env.hold(self.settle_first); ran.append("settle_first")
        for r in range(self.max_rounds):
            info = env.progress()
            rounds.append(dict(stage=info.get("stage"), node=info.get("node"), unmet=info.get("unmet"), reason=info.get("reason")))
            if info["abstain"]:
                return self._fail(env, "abstain:" + info["reason"])
            st = info["stage"]
            if st == "done":
                return self._traj(env, ran, rounds=rounds)
            if st is None or st.startswith("no_skill"):
                return self._fail(env, f"fail:{st or 'no_skill'}:{info.get('node')}")
            if st == "settle":
                env.hold(15); ran.append("settle"); continue
            res = env.run_plan(self.plan, start_stage=st, ctx={}, strict=True, until_segment_end=True)
            ran += res["stages"]
            if not res["ok"]:
                rounds[-1]["plan_failed_at"] = res.get("failed_stage")
        info = env.progress()
        if info["stage"] == "done":
            return self._traj(env, ran, rounds=rounds)
        return self._fail(env, f"fail:not_done_after_{self.max_rounds}_rounds:{info.get('node') or info.get('reason')}")


class CuroboStageAwareCorrector(StageAwareCorrector):
    """L3: identical stage logic to L2; every motion query goes to the cuRobo planning server (separate venv,
    see data/curobo_server.py) instead of mplib."""
    layer = "L3"
    backend = "curobo"


class SamplingMPCCorrector(_Base):
    """Interface stub for soft-body correction with Newton sampling-based MPC (MPPI/CEM over end-effector action
    sequences, rolled out in parallel Newton worlds cloned from the snapshot, cost = negative contract margin).
    Not implemented in v0."""
    name, layer = "L4_sampling_mpc", "L4"

    def __init__(self, horizon: int = 20, samples: int = 256, iters: int = 4):
        super().__init__(); self.horizon, self.samples, self.iters = horizon, samples, iters

    def correct(self, env, snapshot: Snapshot, stage):
        raise NotImplementedError("SamplingMPCCorrector: Newton soft-body sampling MPC is a v1 item")
