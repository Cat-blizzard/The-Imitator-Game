"""Opt-in actuator calibration overrides for imsim embodiments (card F1: embodiment credibility cards).

Nothing here changes the defaults in registry.py. A calibration is applied only on request:

    env = imitator.make("DualCubePlace-v1", embodiment="dual_piper", embodiment_calibration="f1")
    # or, for any ManiSkill env whose agents are imsim agents:
    from imitator.embodiments import calibration
    env = calibration.wrap(env, "f1")

What a calibration file (<name>.json in this directory) holds, per arm record (registry.ARMS name):
  arm.joints            arm joint names (order of the arrays below)
  arm.stiffness/damping/force_limit   PhysX joint-drive gains for the arm joints (replace the record's 1e3 / 1e2 / ...)
  arm.action_delay_s    pure delay of the arm drive targets [s] (per joint)
  arm.prefilter_tau_s   time constant of a first-order low-pass on the (delayed) arm drive targets [s] (0 = none)
  arm.gravity           false: keep ManiSkill's gravity-free robot links (default); true: re-enable gravity on the arm
                        links (the drive then sags like a real servo; CPU sim only)
plus provenance (datasets, licences, fit settings, held-out RMSE). The gripper is left at the record's defaults (gripper
and contact calibration need a real robot: card level 3).

How it is applied (mode-agnostic): the delay + filter act on the absolute joint targets that the arm sub-controller sends
to PhysX (set_drive_targets), so pd_joint_pos, delta and end-effector modes are all delayed / filtered the same way, once
per control step (dt = 1 / control_freq; per sim step if the controller interpolates). The filter state is reset to the
current arm joint positions whenever the controller is reset (every env.reset). CPU PhysX only: drive gains cannot be
changed after the GPU sim is initialised, so wrap() refuses GPU envs.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "imsim.embodiment_calibration/1"


def available() -> list[str]:
    return sorted(f[:-5] for f in os.listdir(HERE) if f.endswith(".json"))


def load(name_or_dict) -> dict:
    """calibration dict from a name in this directory ('f1'), a path to a .json file, or a dict (returned as is)"""
    if isinstance(name_or_dict, dict):
        cal = name_or_dict
    else:
        p = name_or_dict if str(name_or_dict).endswith(".json") else os.path.join(HERE, f"{name_or_dict}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no embodiment calibration {name_or_dict!r} (available: {available()})")
        with open(p) as f:
            cal = json.load(f)
    if cal.get("schema") != SCHEMA:
        raise ValueError(f"calibration schema {cal.get('schema')!r} != {SCHEMA!r}")
    return cal


def arm_entry(cal: dict, arm_name: str) -> dict | None:
    e = cal.get("arms", {}).get(arm_name)
    return e if e and e.get("arm") else None


# ------------------------------------------------------------------------------------------------ the pre-filter
class TargetFilter:
    """per-joint pure delay (fractional, linear interpolation between buffered targets) + first-order low-pass,
    stepped once per drive-target update. Works on torch tensors of shape (num_envs, n)."""

    def __init__(self, delay_s, tau_s, dt: float, n: int):
        self.dt = float(dt)
        self.delay = np.broadcast_to(np.asarray(delay_s, float), (n,)).copy()
        self.tau = np.broadcast_to(np.asarray(tau_s, float), (n,)).copy()
        self.d_steps = np.maximum(self.delay / self.dt, 0.0)
        self.n_buf = int(math.floor(self.d_steps.max())) + 2
        self.alpha = np.where(self.tau > 1e-9, 1.0 - np.exp(-self.dt / np.maximum(self.tau, 1e-9)), 1.0)
        self.buf: deque | None = None
        self.y = None

    def reset(self, q):
        """q: current arm joint positions (torch, (num_envs, n)); the robot holds q until delayed commands arrive"""
        self.buf = deque([q.clone() for _ in range(self.n_buf)], maxlen=self.n_buf)
        self.y = q.clone()

    def __call__(self, target):
        import torch
        if self.buf is None:
            self.reset(target)
        self.buf.appendleft(target.clone())          # buf[0] = newest
        n0 = np.floor(self.d_steps).astype(int)
        f = torch.as_tensor(self.d_steps - n0, dtype=target.dtype, device=target.device)
        cols = []
        for j in range(target.shape[-1]):
            a = self.buf[min(n0[j], self.n_buf - 1)][..., j]
            b = self.buf[min(n0[j] + 1, self.n_buf - 1)][..., j]
            cols.append((1 - f[j]) * a + f[j] * b)
        delayed = torch.stack(cols, -1)
        al = torch.as_tensor(self.alpha, dtype=target.dtype, device=target.device)
        self.y = self.y + al * (delayed - self.y)
        return self.y


# ------------------------------------------------------------------------------------------------ apply to agents
def _agents_of(env):
    base = env.unwrapped
    ag = base.agent
    return list(getattr(ag, "agents", [ag]))


def _arm_controller(agent):
    ctrl = agent.controller
    subs = getattr(ctrl, "controllers", None)
    if subs is None:
        return ctrl
    return subs.get("arm")


def apply_to_agent(agent, entry: dict, control_dt: float, sim_dt: float) -> dict:
    """set the calibrated drive gains on the agent's arm joints and patch its arm controller (idempotent)"""
    arm = _arm_controller(agent)
    if arm is None:
        raise ValueError(f"agent {agent.uid}: no 'arm' controller")
    e = entry["arm"]
    names = [j.name for j in arm.joints]
    if list(e["joints"]) != names:
        raise ValueError(f"calibration joints {e['joints']} != controller joints {names}")
    n = len(names)
    K = np.broadcast_to(np.asarray(e["stiffness"], float), (n,))
    D = np.broadcast_to(np.asarray(e["damping"], float), (n,))
    F = np.broadcast_to(np.asarray(e["force_limit"], float), (n,))
    if getattr(arm, "_f1_patched", None) is not entry:
        arm.config.stiffness, arm.config.damping, arm.config.force_limit = K.tolist(), D.tolist(), F.tolist()
        arm.set_drive_property()
        if e.get("gravity", False):
            if agent.scene.gpu_sim_enabled:
                raise RuntimeError("gravity re-enabling is only possible before GPU init (use the CPU sim)")
            joint_children = {j.child_link.name for j in arm.joints if j.child_link is not None}
            for link in agent.robot.links:
                if link.name in _downstream_links(agent, joint_children):
                    link.disable_gravity = False
        dt = sim_dt if getattr(arm.config, "interpolate", False) else control_dt
        filt = TargetFilter(e.get("action_delay_s", 0.0), e.get("prefilter_tau_s", 0.0), dt, n)
        orig_set = arm.__class__.set_drive_targets
        orig_reset = arm.__class__.reset

        def set_drive_targets(targets, _arm=arm, _f=filt, _orig=orig_set):
            return _orig(_arm, _f(targets))

        def reset(_arm=arm, _f=filt, _orig=orig_reset):
            _orig(_arm)
            _f.reset(_arm.qpos.clone())

        arm.set_drive_targets = set_drive_targets
        arm.reset = reset
        arm._f1_filter = filt
        arm._f1_patched = entry
        filt.reset(arm.qpos.clone())
    return dict(arm=agent.record.name if hasattr(agent, "record") else agent.uid, joints=names, stiffness=K.tolist(),
                damping=D.tolist(), force_limit=F.tolist(), action_delay_s=list(np.broadcast_to(
                    np.asarray(e.get("action_delay_s", 0.0), float), (n,))), prefilter_tau_s=list(np.broadcast_to(
                    np.asarray(e.get("prefilter_tau_s", 0.0), float), (n,))), gravity=bool(e.get("gravity", False)))


def _downstream_links(agent, roots: set) -> set:
    """names of the arm links (children of the arm joints) and everything below them"""
    out = set(roots)
    changed = True
    joints = agent.robot.get_joints() if hasattr(agent.robot, "get_joints") else agent.robot.joints
    while changed:
        changed = False
        for j in joints:
            p = getattr(j, "parent_link", None)
            c = getattr(j, "child_link", None)
            if p is not None and c is not None and p.name in out and c.name not in out:
                out.add(c.name)
                changed = True
    return out


def apply(env, name_or_dict="f1") -> list[dict]:
    """apply a calibration to every imsim agent of an already created env (CPU sim). Arms without an entry keep their
    defaults. Returns one summary per agent."""
    from .. import registry as reg
    cal = load(name_or_dict)
    base = env.unwrapped
    if getattr(base, "gpu_sim_enabled", False) or getattr(base.scene, "gpu_sim_enabled", False):
        raise NotImplementedError("embodiment calibration overrides are CPU-sim only (PhysX GPU drive gains are fixed at "
                                  "GPU init); use sim='cpu'")
    control_dt = 1.0 / float(base.control_freq)
    sim_dt = 1.0 / float(base.sim_freq)
    out = []
    for agent in _agents_of(env):
        try:
            rec = reg.record_of_agent(agent)
        except KeyError:
            out.append(dict(agent=agent.uid, applied=False, reason="not an imsim agent"))
            continue
        entry = arm_entry(cal, rec.name)
        if entry is None:
            out.append(dict(agent=agent.uid, arm=rec.name, applied=False, reason=f"no '{rec.name}' entry in calibration"))
            continue
        s = apply_to_agent(agent, entry, control_dt, sim_dt)
        s.update(agent=agent.uid, applied=True, level=entry.get("level"))
        out.append(s)
    return out


def wrap(env, name_or_dict="f1"):
    """gym wrapper that (re-)applies the calibration after every reset (agents and controllers are rebuilt when the
    env reconfigures) and before every step (a control-mode switch creates a new controller)."""
    import gymnasium as gym

    cal = load(name_or_dict)

    class EmbodimentCalibration(gym.Wrapper):
        def __init__(self, e):
            super().__init__(e)
            self.calibration = cal
            self.calibration_summary = apply(e, cal)
            missing = [s for s in self.calibration_summary if not s.get("applied")]
            if missing:
                warnings.warn(f"embodiment calibration {cal.get('name')!r} not applied to: {missing}")

        def _ensure(self):
            for agent in _agents_of(self.env):
                arm = _arm_controller(agent)
                if arm is not None and getattr(arm, "_f1_patched", None) is None:
                    self.calibration_summary = apply(self.env, cal)
                    return

        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            self.calibration_summary = apply(self.env, cal)
            return obs, info

        def step(self, action):
            self._ensure()
            return self.env.step(action)

    return EmbodimentCalibration(env)
