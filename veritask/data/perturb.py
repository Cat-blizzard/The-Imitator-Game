"""Perturbers: propose small programs of primitive ops (engine.PRIMITIVES) from a snapshot.

(a) JointNoisePerturber      DART-style action noise: a random-walk excursion of the commanded joint targets
(b) ShovePerturber           object slide / shove (initial linear + angular velocity, then let physics run)
    DropPerturber            open the holding gripper for k steps mid-grasp, then restore the command
    TargetOffsetPerturber    small pose offset of the target (goal region / tray / appliance)
(c) weak policy              see sources.WeakPolicySource (its own failure states become snapshots)
(d) ProgramPerturber         semantic perturbations as JSON programs (hand-written or model-proposed);
    AIProgramProposer        interface for model-proposed programs through veritask.ai.backends (v0: prompt + parser
                             only; generate() raises NotImplementedError)

Perturbers only read snapshot.meta, which the session fills at snapshot time:
  meta['objects']      movable object names            meta['holding']  {arm: object name or None}
  meta['targets']      target object names that may be offset now (empty when something rests on the target)
  meta['active_arms']  arms that are moving in the current stage
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .engine import Perturbation, Snapshot, validate_ops


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


@dataclass
class JointNoisePerturber:
    """DART-style noise injection. The commanded joint targets of an active arm follow a Gaussian random walk for
    `steps` control steps (final excursion std = sigma rad per joint), gripper command unchanged."""
    sigmas: tuple = (0.05, 0.10, 0.15)
    steps: tuple = (8, 16)
    name: str = "joint_noise"

    def propose(self, snapshot: Snapshot, rng) -> list[Perturbation]:
        arms = snapshot.meta.get("active_arms") or [0]
        out = []
        for s in self.sigmas:
            a = int(rng.choice(arms)); k = int(rng.integers(self.steps[0], self.steps[1] + 1))
            out.append(Perturbation("joint_noise", "action", [dict(op="joint_noise", args=dict(arm=a, sigma=float(s), steps=k, seed=int(rng.integers(2**31 - 1))))],
                                    params=dict(arm=a, sigma=float(s), steps=k)))
        return out


@dataclass
class ShovePerturber:
    """Slide/shove a movable object: horizontal speed in [vmin, vmax] m/s in a random direction, optional spin about
    z, then `settle` steps with the arms holding their targets. Objects currently held are skipped unless
    include_held (a shove of a held object mostly tests the grasp)."""
    vmin: float = 0.25
    vmax: float = 0.8
    wz_max: float = 4.0
    settle: int = 15
    include_held: bool = False
    name: str = "shove"

    def propose(self, snapshot: Snapshot, rng) -> list[Perturbation]:
        held = {o for o in snapshot.meta.get("holding", {}).values() if o}
        out = []
        for obj in snapshot.meta.get("objects", []):
            if obj in held and not self.include_held:
                continue
            v, th = _u(rng, self.vmin, self.vmax), _u(rng, -np.pi, np.pi); wz = _u(rng, -self.wz_max, self.wz_max)
            args = dict(obj=obj, vx=round(v * np.cos(th), 4), vy=round(v * np.sin(th), 4), vz=0.0, wx=0.0, wy=0.0, wz=round(wz, 3), settle=self.settle)
            out.append(Perturbation("shove", "object", [dict(op="shove", args=args)], params=dict(obj=obj, speed=round(v, 3), heading_deg=round(np.degrees(th), 1), wz=round(wz, 3))))
        return out


@dataclass
class DropPerturber:
    """Drop mid-grasp: open the gripper of an arm that is holding an object for k steps, then restore the previous
    gripper command, then let the object settle."""
    k_range: tuple = (4, 10)
    settle: int = 15
    name: str = "drop"

    def propose(self, snapshot: Snapshot, rng) -> list[Perturbation]:
        out = []
        for arm, obj in snapshot.meta.get("holding", {}).items():
            if not obj:
                continue
            arm = int(arm); k = int(rng.integers(self.k_range[0], self.k_range[1] + 1))
            prev = float(snapshot.ctl.get("grip", [-1.0, -1.0])[arm]) if snapshot.ctl else -1.0
            ops = [dict(op="set_gripper", args=dict(arm=arm, cmd=1.0, steps=k)),
                   dict(op="set_gripper", args=dict(arm=arm, cmd=prev, steps=6)),
                   dict(op="hold", args=dict(steps=self.settle))]
            out.append(Perturbation("drop", "object", ops, params=dict(arm=arm, obj=obj, open_steps=k)))
        return out


@dataclass
class TargetOffsetPerturber:
    """Small pose offset of a target object (goal region / tray / appliance): |dxy| in [dmin, dmax] m in a random
    direction plus a yaw offset in [-yaw, yaw] deg. Only proposed when the session marks the target as free."""
    dmin: float = 0.02
    dmax: float = 0.06
    yaw_deg: float = 15.0
    settle: int = 5
    name: str = "target_offset"

    def propose(self, snapshot: Snapshot, rng) -> list[Perturbation]:
        out = []
        for obj in snapshot.meta.get("targets", []):
            d, th = _u(rng, self.dmin, self.dmax), _u(rng, -np.pi, np.pi); yaw = _u(rng, -self.yaw_deg, self.yaw_deg)
            args = dict(obj=obj, dx=round(d * np.cos(th), 4), dy=round(d * np.sin(th), 4), dz=0.0, dyaw_deg=round(yaw, 2), settle=self.settle)
            out.append(Perturbation("target_offset", "object", [dict(op="offset_pose", args=args)], params=dict(obj=obj, dist=round(d, 4), heading_deg=round(np.degrees(th), 1), dyaw_deg=round(yaw, 2))))
        return out


# ============================================================================================ (d) semantic programs
@dataclass
class ProgramPerturber:
    """Semantic perturbations written as JSON programs (by a person or proposed by a model).

    File format: {"programs": [{"name", "description", "task": task_id or "*",
                                "applies_if": {"stage_in": [...], "holding": true|false|null},
                                "ops": [{"op": ..., "args": {...}}]}]}
    Argument placeholders resolved from the snapshot: "$held" -> name of a held object, "$held_arm" -> its arm,
    "$active_arm" -> first active arm, "$target" -> first offsettable target, "$object:<i>" -> i-th movable object.
    Every program is validated against engine.PRIMITIVES before it is applied.
    """
    path: str
    name: str = "program"
    programs: list = field(default_factory=list)

    def __post_init__(self):
        self.programs = json.load(open(self.path))["programs"]

    @staticmethod
    def _resolve(v, snap: Snapshot):
        if not (isinstance(v, str) and v.startswith("$")):
            return v
        held = [(int(a), o) for a, o in snap.meta.get("holding", {}).items() if o]
        if v == "$held":
            return held[0][1] if held else None
        if v == "$held_arm":
            return held[0][0] if held else None
        if v == "$active_arm":
            return (snap.meta.get("active_arms") or [0])[0]
        if v == "$target":
            t = snap.meta.get("targets", []); return t[0] if t else None
        if v.startswith("$object:"):
            objs = snap.meta.get("objects", []); i = int(v.split(":")[1]); return objs[i] if i < len(objs) else None
        return None

    def propose(self, snapshot: Snapshot, rng) -> list[Perturbation]:
        out = []
        for p in self.programs:
            if p.get("task", "*") not in ("*", snapshot.task):
                continue
            cond = p.get("applies_if", {})
            if cond.get("stage_in") and snapshot.stage not in cond["stage_in"]:
                continue
            holding = any(snapshot.meta.get("holding", {}).values())
            if cond.get("holding") is not None and bool(cond["holding"]) != holding:
                continue
            ops = [dict(op=o["op"], args={k: self._resolve(v, snapshot) for k, v in o.get("args", {}).items()}) for o in p["ops"]]
            if any(v is None for o in ops for v in o["args"].values()):
                continue
            if validate_ops(ops, snapshot.meta.get("objects", []) + snapshot.meta.get("all_targets", [])):
                continue
            out.append(Perturbation(f"program:{p['name']}", "semantic", ops, params=dict(name=p["name"], description=p.get("description", ""))))
        return out


PROPOSE_PROMPT = """You are red-teaming a robot data engine. Task instruction: "{instruction}".
The physical success contract is: {contract}.
Current scene (privileged state): {scene}.
Propose {n} *semantic* perturbations: realistic things that could go wrong at this moment (a person bumps the
object, the grasp slips, the target is moved, ...) that a recovery planner should be able to fix.
Write each as a program of these primitive operations only (argument ranges are hard limits):
{primitives}
Objects you may name in 'obj': {objects}.
Return JSON only: {{"programs": [{{"name": "...", "description": "...", "ops": [{{"op": "...", "args": {{...}}}}]}}]}}"""


class AIProgramProposer:
    """Interface for model-proposed semantic perturbations (v0: not executed in E7).

    backend: name for veritask.ai.backends.make ('agent' | 'claude' | 'local' | 'openai'); kw passed through.
    build_prompt() and parse() are implemented and tested; generate() is the model call and is left unimplemented
    for v0 (write programs to a JSON file and use ProgramPerturber). Proposed programs must pass validate_ops and
    are judged like any other perturbation: AI proposes, physics disposes.
    """

    def __init__(self, backend: str = "agent", **kw):
        self.backend_name, self.kw = backend, kw

    @staticmethod
    def build_prompt(instruction: str, contract_json: dict, scene: dict, objects: list[str], n: int = 3) -> str:
        from .engine import PRIMITIVES
        return PROPOSE_PROMPT.format(instruction=instruction, contract=json.dumps(contract_json.get("outcome", []))[:2000],
                                     scene=json.dumps(scene)[:2000], n=n, primitives=json.dumps(PRIMITIVES), objects=objects)

    @staticmethod
    def parse(text: str, objects: list[str]) -> tuple[list[dict], list[str]]:
        """-> (valid programs, rejection messages)."""
        from ..ai.loop import _json_block
        d = _json_block(text) or {}
        ok, rej = [], []
        for p in d.get("programs", []):
            errs = validate_ops(p.get("ops", []), objects)
            (rej.append(f"{p.get('name')}: {'; '.join(errs)}") if errs else ok.append(p))
        return ok, rej

    def generate(self, instruction: str, contract_json: dict, scene: dict, objects: list[str], n: int = 3):
        # from ..ai.backends import make; model = make(self.backend_name, **self.kw); text = model.chat(prompt)
        raise NotImplementedError("v0: interface only. Put programs in a JSON file and use ProgramPerturber.")
