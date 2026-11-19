"""Benchmark adapter interface for the metamorphic tests.

An adapter wraps ONE task environment of a benchmark and exposes what the test items need:

  required
    rollout(ep)                 run or replay the original episode from its initial state; per-step predicate trace
    goto(ep, ro, t)             put the simulator in the state after control step t of that episode (t=-1: initial)
    replay(ep, ro, freeze, dt_scale)   re-execute the episode's actions open-loop from the initial state, optionally
                                with the goal objects frozen at their initial pose (M6) or a scaled timestep (M7)
    step(action)                one control step through the benchmark's own action interface
    success()                   the benchmark's OWN success predicate on the current state
    object_poses()              {name: [x, y, z, qw, qx, qy, qz]} of the task objects
    tcp_positions()             {arm: [x, y, z]} of every end effector
    set_hold_reference()        remember the current arm configuration / end-effector pose as the hold target
    hold_action(gripper)        action that keeps the arm at the hold reference; gripper 'keep' (last command) or 'open'
    tcp_action(targets)         action that moves every TCP toward its target position, gripper open, orientation kept
  optional (return None / False when unsupported; the row then says so)
    contact_begin() / contact_end()   arm-object contacts accumulated over every physics sub-step in between
    nudge(names, dv)            add a velocity change dv (m/s, world frame) to the named objects' centre of mass
    policy_action()             next action of the episode's policy (M4); None if the episode has no policy we own
    idle_action()               M0 action (default: hold the initial arm target with the initial gripper command)

Phase methods (phase_hold, phase_release, phase_retreat, phase_nudge, phase_policy, phase_idle) are implemented
here from the step-level primitives; adapters whose benchmark only offers macro actions (e.g. RoboTwin's planner
moves) override them.  Every phase returns a PhaseLog.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .protocol import Protocol


@dataclass
class PhaseLog:
    n_steps: int = 0
    success: bool | None = None          # predicate at the end of the phase
    success_any: bool = False            # predicate true at any step of the phase
    success_frac: float | None = None    # share of the phase's steps with the predicate true
    contact_task: bool | None = None     # any robot link touched a task object during the phase
    contact_other: bool | None = None    # ... touched another movable (non-robot, non-static) body
    contact_detail: list = field(default_factory=list)   # [(robot link, body), ...] (deduplicated, capped)
    reached: bool | None = None          # retreat: every TCP within tolerance of its lift target
    tcp_rise_m: float | None = None      # retreat: smallest TCP rise over the arms
    disp: dict = field(default_factory=dict)   # object displacement (m) from the phase start
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contact_detail"] = [list(x) for x in self.contact_detail[:20]]
        return d


@dataclass
class Rollout:
    """Result of running / replaying the original episode."""
    success: list                         # predicate after each control step (index t = after step t)
    official: bool                        # the benchmark's official verdict for this episode
    official_semantics: str               # 'any' | 'end'
    actions: list | None = None           # open-loop actions (for goto by replay, M6, M7)
    meta: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.success)

    @property
    def t_first(self) -> int | None:
        for i, s in enumerate(self.success):
            if s:
                return i
        return None


def _disp(p0: dict, p1: dict) -> dict:
    out = {}
    for k in p0:
        if k in p1:
            out[k] = round(float(np.linalg.norm(np.asarray(p1[k][:3]) - np.asarray(p0[k][:3]))), 5)
    return out


class Adapter:
    benchmark: str = "?"
    control_freq: float = 20.0
    has_policy: bool = False

    # ------------------------------------------------------------------ required
    def rollout(self, ep) -> Rollout:
        raise NotImplementedError

    def goto(self, ep, ro: Rollout, t: int) -> None:
        raise NotImplementedError

    def replay(self, ep, ro: Rollout, freeze: bool = False, dt_scale: float = 1.0) -> dict:
        """-> dict(success=[...per step], frozen=[names], dt=..., error=None)"""
        raise NotImplementedError

    def step(self, action) -> None:
        raise NotImplementedError

    def success(self) -> bool:
        raise NotImplementedError

    def object_poses(self) -> dict:
        raise NotImplementedError

    def tcp_positions(self) -> dict:
        raise NotImplementedError

    def set_hold_reference(self) -> None:
        raise NotImplementedError

    def hold_action(self, gripper: str = "keep"):
        raise NotImplementedError

    def tcp_action(self, targets: dict):
        raise NotImplementedError

    # ------------------------------------------------------------------ optional
    def horizon(self, ep, ro: Rollout) -> int:
        """M0 length in control steps: the episode end (default: the original episode's length)."""
        return ro.n_steps

    def goal_objects(self) -> list:
        return list(self.object_poses())

    def nudge_objects(self) -> list:
        return self.goal_objects()

    def contact_begin(self) -> bool:
        return False

    def contact_end(self) -> dict | None:
        return None

    def nudge(self, names: list, dv) -> bool:
        return False

    def policy_action(self):
        return None

    def idle_action(self):
        """M0 action: hold the reference taken at the initial state; gripper open (its initial command in every
        benchmark tested here; adapters override if not)."""
        return self.hold_action("open")

    def reset_initial(self, ep, ro: Rollout) -> None:
        self.goto(ep, ro, -1)

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ phases
    def _run(self, n: int, action_fn, log: PhaseLog, trace: list) -> None:
        for _ in range(n):
            self.step(action_fn())
            trace.append(bool(self.success()))
        log.n_steps += n

    @staticmethod
    def _close(log: PhaseLog, trace: list) -> PhaseLog:
        if trace:
            log.success = trace[-1]
            log.success_any = any(trace)
            log.success_frac = round(sum(trace) / len(trace), 4)
        return log

    def _contacts_into(self, log: PhaseLog, started: bool) -> None:
        if not started:
            return
        c = self.contact_end()
        if c is None:
            return
        log.contact_task = bool(c.get("task"))
        log.contact_other = bool(c.get("other"))
        log.contact_detail = sorted({tuple(x) for x in c.get("detail", [])})

    def phase_hold(self, proto: Protocol, gripper: str = "keep") -> PhaseLog:
        log, tr = PhaseLog(), []
        p0 = self.object_poses()
        self.set_hold_reference()
        started = self.contact_begin()
        n = proto.steps(proto.hold_s if gripper == "keep" else proto.release_s, self.control_freq)
        self._run(n, lambda: self.hold_action(gripper), log, tr)
        self._contacts_into(log, started)
        log.disp = _disp(p0, self.object_poses())
        return self._close(log, tr)

    def phase_release(self, proto: Protocol) -> PhaseLog:
        return self.phase_hold(proto, gripper="open")

    def phase_retreat(self, proto: Protocol) -> PhaseLog:
        """Lift every TCP by retreat_height_m along world +z with the gripper open (servo reference moving at
        retreat_speed_mps, capped at retreat_timeout_s), then hold retreat_wait_s with the gripper open.
        Contacts are recorded over the lift and the wait."""
        log, tr = PhaseLog(), []
        p0 = self.object_poses()
        start = {k: np.asarray(v, float) for k, v in self.tcp_positions().items()}
        goal = {k: v + np.array([0.0, 0.0, proto.retreat_height_m]) for k, v in start.items()}
        started = self.contact_begin()
        n_max = proto.steps(proto.retreat_timeout_s, self.control_freq)
        n_ramp = proto.steps(proto.retreat_height_m / proto.retreat_speed_mps, self.control_freq)

        def at_goal():
            cur = self.tcp_positions()
            return all(np.linalg.norm(np.asarray(cur[a]) - goal[a]) <= proto.retreat_tol_m for a in goal)

        k = 0
        while k < n_max:
            frac = min(1.0, (k + 1) / n_ramp)
            self.step(self.tcp_action({a: start[a] + frac * (goal[a] - start[a]) for a in start}))
            tr.append(bool(self.success()))
            k += 1
            if frac >= 1.0 and at_goal():
                break
        log.n_steps = k
        cur = self.tcp_positions()
        log.reached = bool(at_goal())
        log.tcp_rise_m = round(float(min(np.asarray(cur[a])[2] - start[a][2] for a in start)), 4)
        log.extra["lift_steps"] = k
        self.set_hold_reference()
        self._run(proto.steps(proto.retreat_wait_s, self.control_freq), lambda: self.hold_action("open"), log, tr)
        self._contacts_into(log, started)
        log.disp = _disp(p0, self.object_poses())
        return self._close(log, tr)

    def phase_nudge(self, proto: Protocol, direction) -> PhaseLog:
        log, tr = PhaseLog(), []
        p0 = self.object_poses()
        names = [n for n in self.nudge_objects() if n in p0]
        dv = np.asarray(direction, float) * float(proto.nudge_dv_mps)
        ok = bool(names) and self.nudge(names, dv)
        log.extra.update(nudged=names if ok else [], dv=[round(float(x), 4) for x in dv], supported=bool(ok))
        if not ok:
            log.error = "nudge unsupported" if names else "no nudge objects"
            return log
        self.set_hold_reference()
        started = self.contact_begin()
        self._run(proto.steps(proto.nudge_wait_s, self.control_freq), lambda: self.hold_action("open"), log, tr)
        self._contacts_into(log, started)
        log.disp = _disp(p0, self.object_poses())
        return self._close(log, tr)

    def phase_policy(self, proto: Protocol) -> PhaseLog:
        log, tr = PhaseLog(), []
        if not self.has_policy:
            log.error = "no policy"
            return log
        p0 = self.object_poses()
        self._run(proto.steps(proto.continue_s, self.control_freq), self.policy_action, log, tr)
        log.disp = _disp(p0, self.object_poses())
        return self._close(log, tr)

    def phase_idle(self, n_steps: int) -> PhaseLog:
        """M0: from the current (initial) state hold the arm with the idle action for n_steps."""
        log, tr = PhaseLog(), []
        self.set_hold_reference()
        self._run(n_steps, self.idle_action, log, tr)
        return self._close(log, tr)
