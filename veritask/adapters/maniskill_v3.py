"""ManiSkill / SAPIEN recorder for evaluation protocol v3 (semantics in adapters/protocol_v3.py).

TraceRecorderV3 = TraceRecorderV2 (substep-accumulated contact pairs, v2 gap / release signals, so v1 and v2 verdicts can
still be computed from the same rollout) plus:
  * in-plane impulse per pair (PairV3(inplane=True)): <pair>_impulse_xy_Ns (per step, |xy| of the vector impulse summed over
    the pair's bodies, per substep) and <pair>_impulse_xy_cum_Ns. Diagnostic only: PhysX contact-point impulses are
    normal impulses (measured tangential part < 1e-7 N*s), so friction drags do not show up in it.
  * PushTracker: displacement of an object caused by robot contact. A "push episode" starts at the first physics substep
    in which the robot touches the object and lasts while it touches, then while the object keeps coasting, until the
    object is at rest (speed <= rest_speed) or another pusher (e.g. the peg) touches it. The episode's NET in-plane
    displacement is added to a running total. Signals: <name>_disp_m (running total incl. the open episode; use trace.max),
    plus its state (<name>_total_m, _active, _start_x, _start_y) so snapshots/restores of the trace stay consistent.
  * Enclosure: per control step and object group, the smallest total clearance (finger 1 + finger 2, PhysX contact-point
    separations clipped at 0) among grippers with part of the object between their finger pads (inter-finger slab in the
    hand frame, from the finger collision boxes); protocol_v3.NOT_ENCLOSED_M when no gripper has the object between its
    pads or a finger is beyond PhysX's contact range. The margin is applied by the evaluator (contract release block).
  * settle_v3(contract, hold): hold 'A' = switch every arm to pd_joint_pos and hold the joint positions reached at the
    policy's last step (drive-velocity targets zeroed; gripper keeps its last commanded opening), in any control mode;
    hold 'B' = replay the policy's last action unchanged. Steps until protocol_v3.find_window reports a complete judged
    window or that quiescence can no longer come early enough (cap settle.max_seconds). The original control modes are
    restored at the next reset.
  * continue_hold(n): more steps under the same hold (aftermath; callers restore the judged traces afterwards).
"""
from __future__ import annotations

import numpy as np

from .maniskill_v2 import Pair, SubstepContacts, TraceRecorderV2, _entity
from .protocol_v3 import NOT_ENCLOSED_M, SettleConfig, evaluate_traces_v3, find_window


class PairV3(Pair):
    def __init__(self, a, b, gap: bool = False, cumulative: bool = False, inplane: bool = False):
        super().__init__(a, b, gap=gap, cumulative=cumulative)
        self.inplane = inplane


class PushTracker:
    def __init__(self, name: str, pair: str, obj, others=(), rest_speed: float = 0.005):
        self.name, self.pair, self.obj, self.others, self.rest_speed = name, pair, obj, tuple(others), float(rest_speed)
        self.total, self.active, self.start, self.p_prev, self.step_max = 0.0, False, None, None, 0.0

    def signal_names(self):
        n = self.name
        return [f"{n}_disp_m", f"{n}_total_m", f"{n}_active", f"{n}_start_x", f"{n}_start_y"]

    def sync(self, u, traces: dict):
        """state from the last recorded step (so a restored trace restores the tracker), position from the sim"""
        n = self.name
        tr = traces.get(f"{n}_total_m")
        if tr is not None and len(tr):
            self.total = float(tr[-1]); self.active = bool(traces[f"{n}_active"][-1] > 0.5)
            self.start = np.array([traces[f"{n}_start_x"][-1], traces[f"{n}_start_y"][-1]]) if self.active else None
        else:
            self.total, self.active, self.start = 0.0, False, None
        self.p_prev = self._p(u); self.step_max = self.running(self.p_prev)

    def _p(self, u):
        return self.obj(u).pose.p[0, :2].cpu().numpy().astype(float)

    def running(self, p) -> float:
        return self.total + (float(np.linalg.norm(p - self.start)) if self.active else 0.0)

    def on_substep(self, u, imp: dict):
        p = self._p(u)
        if imp.get(self.pair, 0.0) > 0.0:
            if not self.active:
                self.active, self.start = True, self.p_prev.copy()
        elif self.active:
            speed = float(np.linalg.norm(self.obj(u).linear_velocity[0].cpu().numpy()))
            if any(imp.get(o, 0.0) > 0.0 for o in self.others) or speed <= self.rest_speed:
                self.total += float(np.linalg.norm(p - self.start)); self.active, self.start = False, None
        self.step_max = max(self.step_max, self.running(p))
        self.p_prev = p

    def end_step(self) -> dict:
        n = self.name
        st = self.start if self.active else np.zeros(2)
        out = {f"{n}_disp_m": self.step_max, f"{n}_total_m": self.total, f"{n}_active": float(self.active),
               f"{n}_start_x": float(st[0]), f"{n}_start_y": float(st[1])}
        self.step_max = self.running(self.p_prev)
        return out


class SubstepContactsV3(SubstepContacts):
    """SubstepContacts + per-pair in-plane impulse + push trackers."""

    def __init__(self, pairs: dict, trackers=()):
        self.trackers = list(trackers)
        super().__init__(pairs)

    def _begin(self) -> None:
        super()._begin()
        self.imp_xy = {k: 0.0 for k in self.pairs}

    def substep_vectors(self, contacts):
        """per pair: (sum over body pairs of |vector impulse|, min separation, vector impulse on side A summed over pairs)"""
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
        norms = {k: float(sum(np.linalg.norm(v) for v in d.values())) for k, d in vec.items()}
        sums = {k: (np.sum(list(d.values()), axis=0) if d else np.zeros(3)) for k, d in vec.items()}
        return norms, sep, sums

    def on_substep(self, u) -> None:
        if not self.sets:
            return
        imp, sep, sums = self.substep_vectors(u.scene.px.get_contacts())
        self.n_in_step += 1; self.n_substeps_seen += 1
        for k, p in self.pairs.items():
            j = imp[k]
            self.imp[k] += j
            self.nsub[k] += int(j > 0.0)
            self.maxsub[k] = max(self.maxsub[k], j / self.dt)
            self.lastsub[k] = j / self.dt
            if sep[k] is not None and (self.sep[k] is None or sep[k] < self.sep[k]):
                self.sep[k] = sep[k]
            if getattr(p, "inplane", False):
                self.imp_xy[k] += float(np.linalg.norm(np.asarray(sums[k])[:2]))
        for t in self.trackers:
            t.on_substep(u, imp)

    def end_step(self) -> dict:
        xy = dict(self.imp_xy)
        out = super().end_step()          # resets the per-step accumulators (calls _begin)
        for k, p in self.pairs.items():
            if getattr(p, "inplane", False):
                out[f"{k}_impulse_xy_Ns"] = xy[k]
        for t in self.trackers:
            out.update(t.end_step())
        return out


# ------------------------------------------------------------------------------------------------ enclosure
def _agents(u):
    return list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent]


class Enclosure:
    """Smallest total finger clearance to the objects among grippers that have part of an object between their finger
    pads (see module docstring). objs(u) -> list of Actors / Links."""

    def __init__(self, objs, n_centroids: bool = True):
        self.objs = objs
        self._key, self._fingers, self._pts = None, None, None

    def _cache(self, u):
        from .maniskill_v2_tasks import collision_parts_local
        objs = self.objs(u)
        key = tuple(id(o._objs[0]) for o in objs) + tuple(id(ag.robot._objs[0]) for ag in _agents(u))
        if key != self._key:
            self._fingers = [[np.vstack(collision_parts_local(f)) for f in (ag.finger1_link, ag.finger2_link)] for ag in _agents(u)]
            pts = []
            for o in objs:
                from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
                ms = get_component_meshes(o._bodies[0])
                pts.append(np.vstack([np.vstack([m.vertices, m.triangles_center]) for m in ms]).astype(float))
            self._pts, self._key = pts, key
        return objs

    @staticmethod
    def _T(obj):
        return obj.pose.to_transformation_matrix()[0].cpu().numpy().astype(float)

    def __call__(self, u) -> float:
        objs = self._cache(u)
        contacts = u.scene.px.get_contacts()
        best = NOT_ENCLOSED_M
        for ag, fverts in zip(_agents(u), self._fingers):
            Th = np.linalg.inv(self._T(ag.robot.links_map["panda_hand"]) if "panda_hand" in ag.robot.links_map else self._T(ag.tcp))
            F = []
            for f, V in zip((ag.finger1_link, ag.finger2_link), fverts):
                T = Th @ self._T(f)
                F.append((T[:3, :3] @ V.T).T + T[:3, 3])
            hi, lo = (F[0], F[1]) if F[0][:, 1].mean() > F[1][:, 1].mean() else (F[1], F[0])
            y_hi, y_lo = hi[:, 1].min(), lo[:, 1].max()
            if y_hi - y_lo <= 0:
                continue                        # closed gripper: nothing can lie between the pads
            x0, x1 = max(F[0][:, 0].min(), F[1][:, 0].min()), min(F[0][:, 0].max(), F[1][:, 0].max())
            z0, z1 = max(F[0][:, 2].min(), F[1][:, 2].min()), min(F[0][:, 2].max(), F[1][:, 2].max())
            fents = [_entity(ag.finger1_link), _entity(ag.finger2_link)]
            for o, P in zip(objs, self._pts):
                T = Th @ self._T(o)
                Q = (T[:3, :3] @ P.T).T + T[:3, 3]
                between = ((Q[:, 0] >= x0) & (Q[:, 0] <= x1) & (Q[:, 2] >= z0) & (Q[:, 2] <= z1)
                           & (Q[:, 1] > y_lo) & (Q[:, 1] < y_hi)).any()
                if not between:
                    continue
                oe = _entity(o); seps = []
                for fe in fents:
                    s = None
                    for c in contacts:
                        e0, e1 = c.bodies[0].entity, c.bodies[1].entity
                        if (e0 == fe and e1 == oe) or (e1 == fe and e0 == oe):
                            for pt in c.points:
                                s = float(pt.separation) if s is None else min(s, float(pt.separation))
                    seps.append(s)
                if any(s is None for s in seps):
                    continue                    # a finger beyond PhysX's contact range: not enclosed
                best = min(best, sum(max(s, 0.0) for s in seps))
        return float(best)


# ------------------------------------------------------------------------------------------------ recorder
class TraceRecorderV3(TraceRecorderV2):
    def __init__(self, env, probes: dict, pairs: dict | None = None, release=None, trackers=(), **kw):
        super().__init__(env, probes, pairs=pairs, release=release, **kw)
        self.trackers = list(trackers)
        self.acc = SubstepContactsV3(self.pairs, self.trackers)
        self.hold_action = None
        self.hold_mode = None
        self._orig_modes = None
        self.window_info = None

    def v2_signal_names(self) -> list[str]:
        names = super().v2_signal_names()
        extra = []
        for k, p in self.pairs.items():
            if getattr(p, "inplane", False):
                extra += [f"{k}_impulse_xy_Ns", f"{k}_impulse_xy_cum_Ns"]
        for t in self.trackers:
            extra += t.signal_names()
        return names[:-1] + extra + names[-1:]          # keep 'phase' last

    def reset(self, **kw):
        self._restore_modes()
        out = super().reset(**kw)
        self.hold_action, self.hold_mode, self.window_info = None, None, None
        return out

    def step(self, action):
        u = self.env.unwrapped
        for t in self.trackers:
            if self.traces:
                t.sync(u, self.traces)
        xy_prev = {}
        for k, p in self.pairs.items():
            if getattr(p, "inplane", False):
                prev = self.traces.get(f"{k}_impulse_xy_cum_Ns")
                xy_prev[k] = float(prev[-1]) if prev is not None and len(prev) else 0.0
        # TraceRecorderV2.step appends every name in v2_signal_names(); the xy cumulative values are filled in by
        # wrapping end_step once
        acc = self.acc
        orig_end = acc.end_step

        def end_step():
            out = orig_end()
            for k, v in xy_prev.items():
                out[f"{k}_impulse_xy_cum_Ns"] = v + out[f"{k}_impulse_xy_Ns"]
            return out
        acc.end_step = end_step
        try:
            return super().step(action)
        finally:
            acc.end_step = orig_end

    # ---------------------------------------------------------------- hold
    def _restore_modes(self):
        if self._orig_modes is not None:
            u = self.env.unwrapped
            try:
                if hasattr(u.agent, "agents"):
                    u.agent.set_control_mode(list(self._orig_modes))
                else:
                    u.agent.set_control_mode(self._orig_modes[0])
            except Exception:
                pass
            self._orig_modes = None

    def hold_action_A(self):
        """Switch every arm to a pd_joint_pos hold of the joint positions reached now; grippers keep their last commanded
        opening. Returns the action to repeat."""
        import torch
        from collections import OrderedDict
        u = self.env.unwrapped
        agents = _agents(u); multi = hasattr(u.agent, "agents")
        last = self.last_action
        if last is None:
            raise RuntimeError("hold A before any policy step")
        keys = list(u.action_space.spaces.keys()) if multi else [None]
        grips = []
        for ag, k in zip(agents, keys):
            a = np.asarray(last[k] if multi else last, dtype=np.float64).reshape(-1)
            cc = ag.controller
            if hasattr(cc, "action_mapping") and "gripper" in cc.action_mapping:
                s, e = cc.action_mapping["gripper"]; grips.append(a[s:e])
            else:
                grips.append(np.zeros(0))
        self._orig_modes = [ag.control_mode for ag in agents]
        if any("pd_joint_pos" not in ag.supported_control_modes for ag in agents):
            raise RuntimeError("hold A needs a pd_joint_pos controller")
        if multi:
            u.agent.set_control_mode(["pd_joint_pos"] * len(agents))
        else:
            u.agent.set_control_mode("pd_joint_pos")
        acts = []
        for ag, g in zip(agents, grips):
            ag.controller.reset()               # a controller created on the fly has no targets yet (ManiSkill 3.0.1 mimic gripper)
            arm = ag.controller.controllers["arm"]
            arm.articulation.set_joint_drive_velocity_targets(torch.zeros((1, len(arm.joints))), arm.joints, arm.active_joint_indices)
            q = arm.qpos[0].cpu().numpy().astype(np.float32)
            acts.append(np.hstack([q, np.asarray(g, np.float32)]).astype(np.float32))
        if multi:
            return OrderedDict((k, a) for k, a in zip(keys, acts))
        return acts[0]

    def settle_v3(self, contract, hold: str | None = None) -> dict:
        u = self.env.unwrapped
        cfg = SettleConfig.from_contract(contract, hold)
        hz = float(u.control_freq)
        K, W, N = cfg.steps(hz)
        n_policy = len(self.traces["phase"])
        act = self.hold_action_A() if cfg.hold == "A" else self.last_action
        self.hold_action, self.hold_mode = act, cfg.hold
        st = dict(status="incomplete")
        self._settling = True
        try:
            for _ in range(N):
                self.step(act)
                st = find_window(self.traces, cfg, n_policy, hz)
                if st["status"] in ("ok", "not_quiescent", "missing_signal"):
                    break
        finally:
            self._settling = False
        self.window_info = dict(st, n_policy=n_policy, hold=cfg.hold)
        return self.window_info

    def continue_hold(self, n: int) -> None:
        """aftermath: n more steps under the evaluator's hold (callers keep or restore the judged traces)"""
        self._settling = True
        try:
            for _ in range(int(n)):
                self.step(self.hold_action)
        finally:
            self._settling = False

    def evaluate_v3(self, contract, hold: str | None = None, extra_measurements: dict | None = None, strict: bool = True):
        u = self.env.unwrapped
        return evaluate_traces_v3(contract, self.traces, hold=hold or self.hold_mode, control_freq=float(u.control_freq),
                                  extra_measurements=extra_measurements, strict=strict)

    def save(self, path: str, meta: dict | None = None) -> None:
        m = dict(protocol="v3", adapter="maniskill_v3", hold=self.hold_mode, window_info=self.window_info)
        m.update(meta or {})
        super().save(path, meta=m)
