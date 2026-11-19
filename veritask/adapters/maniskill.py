"""ManiSkill / SAPIEN adapter: record privileged-state traces during any episode and evaluate a TaskContract.

Usage:
    probes = {"knob_qpos": lambda u: float(u.partnet_link.joint.qpos[0]),
              "released":  lambda u: float(not u.left_agent.is_grasping(u.knob)[0])}
    env = TraceRecorder(gym.make(...), probes)
    ... run any policy / scripted behaviour through env.step ...
    verdict = env.evaluate(contract, metric_args={"cloth.corner_distance": {...}})
The benchmark's own info['success'] is recorded too (signal 'bench_success') so both verdicts come from the
same rollout.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import evaluate_traces


class TraceRecorder(gym.Wrapper):
    def __init__(self, env, probes: dict):
        super().__init__(env); self.probes = probes; self.traces = {}

    def reset(self, **kw):
        out = self.env.reset(**kw); self.traces = {k: [] for k in list(self.probes) + ["bench_success"]}
        return out

    def step(self, action):
        out = self.env.step(action); info = out[-1]; u = self.env.unwrapped
        for k, fn in self.probes.items():
            self.traces[k].append(fn(u))
        s = info.get("success")
        self.traces["bench_success"].append(float(s[0]) if s is not None and hasattr(s, "__len__") else float(bool(s)))
        return out

    def set_traces(self, traces: dict) -> None:
        """Replace the recorded traces (e.g. when restoring a data-engine snapshot). Copies the lists so later
        steps do not mutate the caller's snapshot."""
        self.traces = {k: list(v) for k, v in traces.items()}
        for k in list(self.probes) + ["bench_success"]:
            self.traces.setdefault(k, [])

    def settle(self, seconds: float, hold_action, control_freq: float | None = None) -> int:
        """Evaluator-side settle phase: after the policy is done, step the env with `hold_action` (an action array,
        or a callable(env) -> action that keeps the robot at its last command) for `seconds`, recording traces.
        Contracts read their end-state windows from this phase, so the result does not depend on whether the
        policy happened to stop early enough (E10 C4) and late undoing is visible (E10 P1). Returns steps run."""
        u = self.env.unwrapped
        hz = control_freq or getattr(u, "control_freq", None) or 20
        n = int(round(seconds * hz))
        for _ in range(n):
            a = hold_action(self.env) if callable(hold_action) else hold_action
            self.step(a)
        return n

    def evaluate(self, contract, released_signal: str | None = "released", extra_measurements: dict | None = None):
        """Contract verdict from recorded traces (trace.* terms need args['signal']) plus any extra end-state
        measurements keyed by term.key. Also returns the benchmark's own success flag from the same rollout.
        Shared logic lives in adapters/base.py (a missing signal now gives an 'unknown' verdict, not a KeyError)."""
        return evaluate_traces(contract, self.traces, released_signal, extra_measurements)
