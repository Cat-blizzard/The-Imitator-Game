"""ManiSkill / SAPIEN adapter for evaluation protocol v2 (see adapters/protocol_v2.py for the semantics).

What this recorder adds to adapters/maniskill.TraceRecorder
  * Substep-accumulated contact signals. BaseEnv._step_action runs sim_freq / control_freq PhysX substeps per control
    step and calls self._after_simulation_step() after each one. We wrap that method ON THE ENV INSTANCE (no ManiSkill
    file is touched; scene.step is left alone) and read scene.px.get_contacts() after every substep. Per named pair
    (links A, bodies B) and per control step we record
        <pair>_impulse_Ns      sum over substeps of sum over (a, b) of |pairwise impulse(a, b)|        [N*s]
        <pair>_contact_acc     1 if <pair>_impulse_Ns >= contact_impulse_Ns, else 0
        <pair>_n_sub           number of substeps in which the pair exchanged a non-zero impulse        (diagnostic)
        <pair>_max_sub_force_N largest single-substep force = impulse / dt                              (diagnostic)
        <pair>_last_sub_force_N force at the last substep = what protocol v1 sampled                    (diagnostic)
        <pair>_gap_m           smallest contact-point separation seen in any substep, clipped to [GAP_FLOOR, gap_far];
                               gap_far (= sum of the two shapes' PhysX contact offsets) means "no contact point reported,
                               i.e. farther than PhysX's contact range"                                 (if gap=True)
        <pair>_impulse_cum_Ns  running total of <pair>_impulse_Ns over the episode                      (if cumulative=True)
    The per-substep pairwise impulse is computed exactly like ManiSkillScene.get_pairwise_contact_impulses on CPU
    (vector sum of the point impulses of all contacts between the two bodies); verify_against_scene_api() checks this
    on live substeps.
  * A gap/contact-based release flag 'released_gap' (protocol_v2.ReleaseRule).
  * An evaluator-run settle phase: settle() holds the LAST action the policy sent (absolute joint-position targets),
    for contract.settle_seconds, and marks these steps with phase = 1. evaluate_v2() reads end-state terms only there.

Limits
  * CPU PhysX only (px.get_contacts()); num_envs = 1.
  * settle() with hold_action=None needs an absolute-target control mode (pd_joint_pos, pd_ee_pose ...). For delta
    control modes repeating the last action is NOT holding; pass hold_action explicitly.
  * The gap is PhysX's contact-point separation, available only inside the contact range (default 2 x 0.02 m);
    release_gap_m must be smaller than that range (checked at reset).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .base import save_traces
from .maniskill import TraceRecorder
from .protocol_v2 import (CONTACT_IMPULSE_NS, PHASE_SIGNAL, RELEASE_SIGNAL, ReleaseRule, evaluate_traces_v2, settle_steps)

GAP_FLOOR_M = -0.01   # separations are clipped below at -1 cm (deep penetration carries no extra information)


class Pair:
    """A named contact pair: a(u) -> list of ManiSkill Actors/Links on side A, b(u) -> list on side B."""

    def __init__(self, a: Callable, b: Callable, gap: bool = False, cumulative: bool = False):
        self.a, self.b, self.gap, self.cumulative = a, b, gap, cumulative


def _entity(x):
    return x._bodies[0].entity


def _contact_range(objs_a, objs_b) -> float:
    """Smallest (contact offset of a shape on A) + (contact offset of a shape on B): beyond this PhysX reports nothing."""
    def off(objs):
        offs = [float(s.contact_offset) for o in objs for s in o._bodies[0].get_collision_shapes()]
        if not offs:
            raise ValueError("a gap pair needs collision shapes on both sides")
        return min(offs)
    return off(objs_a) + off(objs_b)


class SubstepContacts:
    """Accumulates pairwise contact impulses and minimum separations over the physics substeps of one control step."""

    def __init__(self, pairs: dict[str, Pair]):
        self.pairs = pairs
        self.sets: dict[str, tuple[set, set]] = {}
        self.gap_far: dict[str, float] = {}
        self.dt = None
        self.n_substeps_seen = 0
        self._begin()

    def resolve(self, u) -> None:
        """Look the entities up again (ManiSkill rebuilds actors on a reconfiguring reset)."""
        self.sets, self.gap_far = {}, {}
        for name, p in self.pairs.items():
            la, lb = list(p.a(u)), list(p.b(u))
            self.sets[name] = ({_entity(x) for x in la}, {_entity(x) for x in lb})
            self.gap_far[name] = _contact_range(la, lb) if p.gap else float("nan")
        self.dt = float(u.scene.px.timestep)
        self._begin()

    def _begin(self) -> None:
        self.imp = {k: 0.0 for k in self.pairs}
        self.nsub = {k: 0 for k in self.pairs}
        self.maxsub = {k: 0.0 for k in self.pairs}
        self.lastsub = {k: 0.0 for k in self.pairs}
        self.sep = {k: None for k in self.pairs}
        self.n_in_step = 0

    def substep_impulses(self, contacts) -> tuple[dict[str, float], dict[str, float | None]]:
        """One substep: per pair, sum over (a, b) body pairs of |vector sum of point impulses|, and min separation."""
        vec: dict[str, dict] = {k: {} for k in self.pairs}
        sep: dict[str, float | None] = {k: None for k in self.pairs}
        for c in contacts:
            e0, e1 = c.bodies[0].entity, c.bodies[1].entity
            for name, (A, B) in self.sets.items():
                if e0 in A and e1 in B:
                    key, sign = (e0, e1), 1.0
                elif e1 in A and e0 in B:
                    key, sign = (e1, e0), -1.0
                else:
                    continue
                tot = np.zeros(3)
                for pt in c.points:
                    tot += pt.impulse
                    s = float(pt.separation)
                    if sep[name] is None or s < sep[name]:
                        sep[name] = s
                d = vec[name]
                d[key] = d.get(key, 0.0) + sign * tot
        return {k: float(sum(np.linalg.norm(v) for v in d.values())) for k, d in vec.items()}, sep

    def on_substep(self, u) -> None:
        if not self.sets:
            return
        imp, sep = self.substep_impulses(u.scene.px.get_contacts())
        self.n_in_step += 1; self.n_substeps_seen += 1
        for k in self.pairs:
            j = imp[k]
            self.imp[k] += j
            self.nsub[k] += int(j > 0.0)
            self.maxsub[k] = max(self.maxsub[k], j / self.dt)
            self.lastsub[k] = j / self.dt
            if sep[k] is not None and (self.sep[k] is None or sep[k] < self.sep[k]):
                self.sep[k] = sep[k]

    def end_step(self) -> dict[str, float]:
        out = {}
        for k, p in self.pairs.items():
            out[f"{k}_impulse_Ns"] = self.imp[k]
            out[f"{k}_n_sub"] = float(self.nsub[k])
            out[f"{k}_max_sub_force_N"] = self.maxsub[k]
            out[f"{k}_last_sub_force_N"] = self.lastsub[k]
            if p.gap:
                far = self.gap_far[k]
                out[f"{k}_gap_m"] = far if self.sep[k] is None else float(np.clip(self.sep[k], GAP_FLOOR_M, far))
        n = self.n_in_step
        self._begin()
        out["_n_substeps"] = float(n)
        return out


class TraceRecorderV2(TraceRecorder):
    """TraceRecorder + substep-accumulated contact pairs + gap-based release + evaluator-run settle (protocol v2).

    probes   per-control-step probes as in TraceRecorder (may include the v1 probes, so v1 and v2 verdicts can be
             computed from the same rollout)
    pairs    name -> Pair
    release  protocol_v2.ReleaseRule over this recorder's signals (or None when the contract does not require release)
    """

    def __init__(self, env, probes: dict, pairs: dict[str, Pair] | None = None, release: ReleaseRule | None = None,
                 contact_impulse_Ns: float = CONTACT_IMPULSE_NS):
        super().__init__(env, probes)
        self.pairs = dict(pairs or {})
        self.release = release
        self.contact_impulse_Ns = float(contact_impulse_Ns)
        self.acc = SubstepContacts(self.pairs)
        self.last_action = None
        self._settling = False
        self._hooked_env = None
        self.n_settle = 0

    # ---------------------------------------------------------------- substep hook
    def _install_hook(self) -> None:
        u = self.env.unwrapped
        if self._hooked_env is u:
            return
        orig = u._after_simulation_step   # the task's own hook (bound method), kept and called first

        def after_simulation_step():
            orig()
            self.acc.on_substep(u)
        u._after_simulation_step = after_simulation_step   # instance attribute: shadows the class method for this env only
        self._hooked_env = u

    def v2_signal_names(self) -> list[str]:
        names = []
        for k, p in self.pairs.items():
            names += [f"{k}_impulse_Ns", f"{k}_contact_acc", f"{k}_n_sub", f"{k}_max_sub_force_N", f"{k}_last_sub_force_N"]
            if p.gap:
                names.append(f"{k}_gap_m")
            if p.cumulative:
                names.append(f"{k}_impulse_cum_Ns")
        if self.release is not None:
            names.append(RELEASE_SIGNAL)
        return names + [PHASE_SIGNAL]

    # ---------------------------------------------------------------- gym API
    def reset(self, **kw):
        out = super().reset(**kw)
        u = self.env.unwrapped
        self._install_hook()
        self.acc.resolve(u)
        if self.release is not None:
            for g in self.release.gap_signals:
                far = self.acc.gap_far[g[:-len("_gap_m")]]
                if not self.release.gap_m < far:
                    raise ValueError(f"release gap {self.release.gap_m} m is outside PhysX's contact range {far} m for {g}")
        for k in self.v2_signal_names():
            self.traces[k] = []
        self.last_action, self._settling, self.n_settle = None, False, 0
        return out

    def set_traces(self, traces: dict) -> None:
        super().set_traces(traces)
        for k in self.v2_signal_names():
            self.traces.setdefault(k, [])

    def step(self, action):
        self.acc._begin()
        out = super().step(action)   # env.step -> substep hook fires sim_freq / control_freq times -> v1-style probes
        vals = self.acc.end_step()
        n_sub = vals.pop("_n_substeps")
        u = self.env.unwrapped
        if self.pairs and int(n_sub) != int(u._sim_steps_per_control):
            raise RuntimeError(f"substep hook saw {int(n_sub)} substeps, expected {u._sim_steps_per_control}")
        for k, p in self.pairs.items():
            j = vals[f"{k}_impulse_Ns"]
            vals[f"{k}_contact_acc"] = float(j >= self.contact_impulse_Ns)
            if p.cumulative:   # running total kept in the trace itself, so snapshot / restore of traces stays consistent
                prev = self.traces[f"{k}_impulse_cum_Ns"]
                vals[f"{k}_impulse_cum_Ns"] = (prev[-1] if len(prev) else 0.0) + j
        if self.release is not None:
            vals[RELEASE_SIGNAL] = self.release.step_value(vals)
        vals[PHASE_SIGNAL] = 1.0 if self._settling else 0.0
        for k in self.v2_signal_names():
            self.traces[k].append(float(vals[k]))
        if not self._settling:
            self.last_action = _copy_action(action)
        return out

    # ---------------------------------------------------------------- protocol v2
    def control_modes(self) -> list[str]:
        u = self.env.unwrapped
        cm = getattr(u, "control_mode", None)
        if isinstance(cm, dict):
            return [str(v) for v in cm.values()]
        return [str(cm)]

    def settle(self, seconds: float, hold_action=None, control_freq: float | None = None) -> int:
        """Evaluator-run settle: hold the policy's last command for `seconds`. Returns the number of settle steps."""
        if hold_action is None:
            if self.last_action is None:
                raise RuntimeError("settle() before any policy step: there is no last command to hold")
            if any("delta" in m for m in self.control_modes()):
                raise ValueError(f"control mode {self.control_modes()} is relative: repeating the last action does not hold "
                                 f"the robot; pass hold_action")
            hold_action = self.last_action
        self._settling = True
        try:
            n = super().settle(seconds, hold_action, control_freq)
        finally:
            self._settling = False
        self.n_settle += n
        return n

    def run_settle(self, contract, hold_action=None) -> int:
        """Run exactly the settle phase the contract asks for (contract.settle_seconds at the env's control rate)."""
        u = self.env.unwrapped
        return self.settle(float(contract.settle_seconds), hold_action, float(u.control_freq))

    def evaluate_v2(self, contract, extra_measurements: dict | None = None, strict: bool = True):
        u = self.env.unwrapped
        return evaluate_traces_v2(contract, self.traces, release=self.release, control_freq=float(u.control_freq),
                                  extra_measurements=extra_measurements, strict=strict)

    def save(self, path: str, meta: dict | None = None) -> None:
        u = self.env.unwrapped
        m = dict(protocol="v2", adapter="maniskill_v2", control_freq=float(u.control_freq), sim_freq=float(u.sim_freq),
                 contact_impulse_Ns=self.contact_impulse_Ns, release_rule=self.release.to_dict() if self.release else None,
                 gap_far_m={k: v for k, v in self.acc.gap_far.items() if v == v}, n_settle=self.n_settle)
        m.update(meta or {})
        save_traces(path, self.traces, meta=m)

    # ---------------------------------------------------------------- verification
    def verify_against_scene_api(self, pair_objs: list[tuple]) -> float:
        """Max |our pairwise impulse - |scene.get_pairwise_contact_impulses(a, b)|| over the given (a, b) struct pairs,
        on the CURRENT contact list. Call it from a substep hook or right after a step."""
        u = self.env.unwrapped
        contacts = u.scene.px.get_contacts()
        worst = 0.0
        for a, b in pair_objs:
            chk = SubstepContacts({"x": Pair(lambda _u, a=a: [a], lambda _u, b=b: [b])})
            chk.sets = {"x": ({_entity(a)}, {_entity(b)})}
            ours = chk.substep_impulses(contacts)[0]["x"]
            api = float(np.linalg.norm(u.scene.get_pairwise_contact_impulses(a, b)[0].cpu().numpy()))
            worst = max(worst, abs(ours - api))
        return worst


def _copy_action(a):
    if isinstance(a, dict):
        return type(a)((k, _copy_action(v)) for k, v in a.items())
    try:
        import torch
        if torch.is_tensor(a):
            return a.clone()
    except ImportError:
        pass
    return np.array(a, copy=True)
