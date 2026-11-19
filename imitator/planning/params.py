"""Parameter spaces of the scripted experts (A2: expert auto-tuning), and the hook that lets a tuned parameter set replace
an expert's defaults.

Every space lists 5-15 continuous parameters with bounds. The DEFAULTS are the values of the frozen experts (E29b /
E27 / E29a scripts) and reproduce them exactly: an expert called with params=None (or with the defaults) executes the
same actions as before the parameterisation. Integer parameters (step counts) are rounded; boolean switches are
continuous in [0, 1] and read as `>= 0.5`.

    from imitator.planning import params as P
    P.space_for("PlugCharger-v1", "panda").defaults()
    P.expert_callable("PlugCharger-v1", "panda", {"insert_step": 0.0008})(env)      # one expert episode on `env`
    P.resolve("PlugCharger-v1", None)                                                 # defaults (+ adopted tuned set)

Tuned parameter sets live in tuned_params.json next to this file: {"<space key>@<embodiment>": {"params": {...},
"adopted": bool, "source": ..., "date": ...}} (a bare "<space key>" entry would apply to every embodiment of the space;
A2 writes per-embodiment entries only). An ADOPTED set replaces the defaults for everybody who runs the expert on that
embodiment (an expert revision, logged in research/results/a2_expert_tuning/STATUS.md); a non-adopted set is only used
when asked for explicitly (params=P.tuned(key, emb=...) or IMSIM_EXPERT_PARAMS=tuned). IMSIM_EXPERT_PARAMS=frozen forces
the frozen defaults everywhere. Experts find their embodiment through env.unwrapped.imsim['embodiment'] (imitator.make).
The TaskSpec entries in tasks.py are not touched: experts read their parameters through resolve().
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TUNED_PATH = os.path.join(HERE, "tuned_params.json")


@dataclass(frozen=True)
class Param:
    name: str
    default: float
    lo: float
    hi: float
    integer: bool = False
    unit: str = ""
    doc: str = ""
    switch: bool = False             # boolean switch (on = value >= 0.5; default off); searched with a wider spread


@dataclass
class Space:
    key: str                         # registry key, e.g. "PlugCharger-v1" or "DualCubePlace-v1:hand"
    task: str
    expert: str                      # 'module:function'; called as f(env, params=<dict>, **kwargs)
    params: tuple
    kwargs: dict = field(default_factory=dict)
    kinds: tuple | None = None       # embodiment kinds ('arm_gripper', 'arm_hand') this space applies to; None = all
    embodiments: tuple | None = None # or explicit embodiment names; None = all
    notes: str = ""

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def lo(self) -> np.ndarray:
        return np.array([p.lo for p in self.params], float)

    def hi(self) -> np.ndarray:
        return np.array([p.hi for p in self.params], float)

    def clip(self, d: dict) -> dict:
        out = {}
        for p in self.params:
            v = float(np.clip(float(d.get(p.name, p.default)), p.lo, p.hi))
            out[p.name] = int(round(v)) if p.integer else v
        return out

    def to_unit(self, d: dict) -> np.ndarray:
        """parameters -> [0, 1]^n (bounds-normalised)"""
        x = np.array([float(d.get(p.name, p.default)) for p in self.params])
        return (x - self.lo()) / (self.hi() - self.lo())

    def from_unit(self, z) -> dict:
        x = self.lo() + np.clip(np.asarray(z, float), 0.0, 1.0) * (self.hi() - self.lo())
        return self.clip({p.name: v for p, v in zip(self.params, x)})

    def describe(self) -> list[dict]:
        return [dict(name=p.name, default=p.default, lo=p.lo, hi=p.hi, integer=p.integer, switch=p.switch, unit=p.unit,
                     doc=p.doc) for p in self.params]


SPACES: dict[str, Space] = {}


def register(space: Space) -> Space:
    for p in space.params:
        if not (p.lo <= p.default <= p.hi):
            raise ValueError(f"{space.key}.{p.name}: default {p.default} outside [{p.lo}, {p.hi}]")
    if len(space.params) > 15:
        raise ValueError(f"{space.key}: {len(space.params)} parameters (A2 card: 5-15)")
    SPACES[space.key] = space
    return space


def _kind(emb: str) -> str:
    if emb in (None, "panda", "panda_cloth"):
        return "arm_gripper"
    from ..embodiments import dual as D
    return D.CONFIGS[emb].records[0].kind


def space_for(task: str, emb: str | None = None) -> Space | None:
    """the parameter space that applies to (task, embodiment), or None"""
    cands = [s for s in SPACES.values() if s.task == task]
    for s in cands:
        if s.embodiments is not None and emb not in s.embodiments:
            continue
        if s.kinds is not None and _kind(emb) not in s.kinds:
            continue
        return s
    return None


# ------------------------------------------------------------------------------------------------ tuned sets
def _tuned_file() -> dict:
    if os.path.exists(TUNED_PATH):
        with open(TUNED_PATH) as f:
            return json.load(f)
    return {}


def tuned(key: str, adopted_only: bool = False, emb: str | None = None) -> dict | None:
    """the tuned set of space `key` for embodiment `emb` ('<key>@<emb>' entry, else a bare '<key>' entry)"""
    f = _tuned_file()
    e = (f.get(f"{key}@{emb}") if emb else None) or f.get(key)
    if not e or (adopted_only and not e.get("adopted")):
        return None
    return dict(e["params"])


def _active_tuned(key: str, emb: str | None = None) -> dict | None:
    mode = os.environ.get("IMSIM_EXPERT_PARAMS", "adopted")
    if mode == "frozen":
        return None
    return tuned(key, adopted_only=(mode != "tuned"), emb=emb)


def emb_of(env) -> str | None:
    """the imsim embodiment name of an env (imitator.make records it in env.unwrapped.imsim)"""
    try:
        return (getattr(getattr(env, "unwrapped", env), "imsim", None) or {}).get("embodiment")
    except Exception:                                                   # noqa: BLE001
        return None


def resolve(key: str, overrides: dict | None = None, emb: str | None = None) -> dict:
    """defaults <- adopted tuned set of (key, emb) (unless IMSIM_EXPERT_PARAMS=frozen) <- explicit overrides; unknown
    names raise"""
    s = SPACES[key]
    d = s.defaults()
    t = _active_tuned(key, emb)
    if t:
        d.update({k: v for k, v in t.items() if k in d})
    if overrides:
        bad = set(overrides) - set(d)
        if bad:
            raise KeyError(f"{key}: unknown expert parameters {sorted(bad)}")
        d.update(overrides)
    return s.clip(d)


def expert_callable(task: str, emb: str, params: dict | None = None, variant: str | None = None):
    """callable(env) -> log running the task's expert with `params` (None: defaults + adopted tuned set); tasks without
    a registered space run their TaskSpec expert script unchanged"""
    from ..tasks import _ref, get_task
    s = space_for(task, emb)
    if s is None:
        spec = get_task(task)
        if params:
            raise KeyError(f"{task} ({emb}): no parameter space registered")
        return spec.script(variant or spec.data_expert)
    f = _ref(s.expert)
    kw = dict(s.kwargs)
    if variant is not None:
        kw[_variant_kw(s)] = variant
    p = resolve(s.key, params, emb)
    return lambda env: f(env, params=p, **kw)


def _variant_kw(s: Space) -> str:
    for k in ("variant", "behaviour", "mode"):
        if k in s.kwargs:
            return k
    return "variant"


def expert_variant(key: str) -> str:
    """the variant / mode / behaviour name of the space's positive control (the tuned expert)"""
    s = SPACES[key]
    return s.kwargs.get(_variant_kw(s), "expert")


def pv(params: dict | None, key: str, env=None, variant: str | None = None) -> dict:
    """inside an expert: the full parameter dict (resolve() for the env's embodiment if the caller passed None or a
    partial dict). Tuned sets apply to the positive control only: a counterexample variant (variant given and not the
    space's expert variant) keeps the frozen parameters, so tuning never changes the designed counterexamples (they
    share code with the expert; A2 found that tuned parameters can turn one into a genuine success)."""
    if params is not None and set(params) >= set(SPACES[key].names):
        return params
    emb = emb_of(env) if env is not None else None
    if variant is not None and variant != expert_variant(key):
        emb = None
    return resolve(key, params, emb)


def I(x) -> int:                       # noqa: E743  (integer parameter)
    return int(round(float(x)))


def B(x) -> bool:                      # boolean switch parameter
    return float(x) >= 0.5


# ================================================================================================ spaces
_P = Param

register(Space(
    key="PlugCharger-v1", task="PlugCharger-v1", expert="imitator.planning.panda_ms:plug_charger",
    kwargs=dict(variant="expert"),
    notes="E29b expert v1.1 (final_v11): ManiSkill solution + E27 refinement + 1 mm insertion increments + "
          "align-at-contact + spiral search",
    params=(
        _P("vel", 0.5, 0.25, 0.9, doc="planner joint velocity / acceleration scale of the transit moves"),
        _P("grasp_tilt_deg", 15.0, 0.0, 25.0, unit="deg", doc="grasp pitch about the TCP y axis (ManiSkill: 15)"),
        _P("reach_dz", 0.05, 0.03, 0.10, unit="m", doc="pre-grasp distance above the grasp pose"),
        _P("refine_tol", 3e-4, 1e-4, 1e-3, unit="m", doc="pre-insertion refinement tolerance"),
        _P("refine_wait", 6, 3, 14, integer=True, unit="steps", doc="hold steps after each refinement move"),
        _P("insert_vel", 0.3, 0.1, 0.6, doc="joint velocity / acceleration limit of the insertion increments"),
        _P("insert_step", 0.001, 0.0005, 0.003, unit="m", doc="insertion increment along the goal axis"),
        _P("insert_wait", 3, 0, 8, integer=True, unit="steps", doc="hold steps after each increment"),
        _P("stall_m", 0.002, 0.001, 0.004, unit="m", doc="lag behind the commanded depth that counts as a stall"),
        _P("spiral_r", 0.0012, 0.0006, 0.002, unit="m", doc="largest spiral-search offset (4 rings)"),
        _P("fix_wait", 8, 3, 15, integer=True, unit="steps", doc="hold steps before an align-at-contact measurement"),
        _P("post_wait", 5, 0, 15, integer=True, unit="steps", doc="hold steps after the insertion"),
        _P("release_steps", 8, 4, 16, integer=True, unit="steps", doc="gripper-open steps"),
        _P("back_m", 0.05, 0.02, 0.08, unit="m", doc="back-off along the insertion axis after the release"),
        _P("pre_wait", 0, 0, 20, integer=True, unit="steps", doc="still steps at the pre-grasp pose (CONTACT)"),
    )))

register(Space(
    key="PullCubeTool-v1", task="PullCubeTool-v1", expert="imitator.planning.panda_ms:pull_cube_tool",
    kwargs=dict(variant="expert"),
    notes="E29b expert v1 (= E27's port of ManiSkill's solution); 'radial' and 'line' are the v1.1 structural "
          "variants as continuous switches (default off = v1)",
    params=(
        _P("vel", 0.75, 0.4, 1.0, doc="planner joint velocity / acceleration scale"),
        _P("grasp_x", 0.03, 0.01, 0.10, unit="m", doc="grasp point on the handle, from its free end"),
        _P("reach_dz", 0.05, 0.03, 0.10, unit="m", doc="pre-grasp distance above the handle grasp"),
        _P("lift_dz", 0.15, 0.06, 0.25, unit="m", doc="tool lift after the grasp"),
        _P("approach_dz", 0.10, 0.04, 0.15, unit="m", doc="height of the approach pose above the hook pose"),
        _P("hook_behind", 0.06, 0.035, 0.09, unit="m", doc="hook inner face beyond the cube centre"),
        _P("hook_side", 0.067, 0.05, 0.085, unit="m", doc="handle line offset to the side of the cube"),
        _P("tool_z", 0.03, 0.026, 0.045, unit="m", doc="tool origin height during the pull (half height 0.025)"),
        _P("pull_m", 0.35, 0.25, 0.40, unit="m", doc="pull length"),
        _P("radial", 0.0, 0.0, 1.0, doc="switch: tool aligned with / pulled along the base -> cube direction (v1.1)", switch=True),
        _P("line", 0.0, 0.0, 1.0, doc="switch: hook lowering and pull as segmented straight lines without RRT (v1.1)", switch=True),
        _P("post_wait", 3, 0, 10, integer=True, unit="steps", doc="hold steps after the pull"),
        _P("pre_wait", 0, 0, 20, integer=True, unit="steps", doc="still steps at the pre-grasp pose (CONTACT)"),
        _P("hook_wait", 0, 0, 20, integer=True, unit="steps", doc="still steps above the cube before lowering"),
    )))

_DCP_COMMON = (
    _P("vel", 0.9, 0.4, 1.0, doc="planner joint velocity / acceleration limit of the acting arm"),
    _P("pre_wait", 0, 0, 20, integer=True, unit="steps", doc="still steps at the pre-grasp pose before the final approach"),
    _P("h_scale", 1.0, 0.6, 1.6, doc="scale of the pre-grasp / lift / above heights (frozen: 10, 7, 5 cm, tried in order)"),
    _P("close_steps", 25, 10, 45, integer=True, unit="steps", doc="gripper / hand closing duration"),
    _P("place_dz", 0.006, 0.0, 0.02, unit="m", doc="cube bottom height above the table at the release"),
    _P("place_wait", 5, 0, 20, integer=True, unit="steps", doc="hold steps at the place pose before opening"),
    _P("corr_max", 0.06, 0.02, 0.12, unit="m", doc="largest in-hand shift that is corrected for at the place"),
    _P("reaim", 0.0, 0.0, 1.0, doc="switch: re-measure the cube above the target and re-aim the place (A2)", switch=True),
    _P("release_steps", 20, 8, 40, integer=True, unit="steps", doc="opening duration"),
    _P("retreat_scale", 1.0, 0.5, 2.0, doc="retreat height after the release, x the waypoint height"),
)
register(Space(
    key="DualCubePlace-v1:gripper", task="DualCubePlace-v1", expert="imitator.planning.scripted:pick_place",
    kwargs=dict(variant="release"), kinds=("arm_gripper",),
    notes="E19/E23 scripted pick-and-place (parallel grippers), phases by E29a",
    params=_DCP_COMMON + (
        _P("grasp_dz", 0.004, 0.0, 0.012, unit="m", doc="fingertip clearance above the table at the grasp"),
        _P("close_cmd", 1.0, 0.8, 1.0, doc="gripper close command (fraction of the closed command)"),
    )))
register(Space(
    key="DualCubePlace-v1:hand", task="DualCubePlace-v1", expert="imitator.planning.scripted:pick_place",
    kwargs=dict(variant="release"), kinds=("arm_hand",),
    notes="E19/E23 scripted pick-and-place with the Allegro power grasp (fixed synergy)",
    params=_DCP_COMMON + (
        _P("hand_along", 0.0, -0.02, 0.02, unit="m", doc="palm TCP offset from the cube centre along the finger axis"),
        _P("hand_below", 0.0, -0.015, 0.015, unit="m", doc="palm TCP offset along the palm normal (into the cube +)"),
        _P("close_cmd", 1.0, 0.8, 1.2, doc="hand synergy value of the grasp (1 = the record's closed synergy)"),
    )))

register(Space(
    key="DualDrawer-v1", task="DualDrawer-v1", expert="imitator.planning.dual_drawer:run", kwargs=dict(variant="expert"),
    notes="E29b fork B2 expert: left arm holds the cabinet's side knob, right arm pulls the drawer handle",
    params=(
        _P("pull_m", 0.10, 0.085, 0.12, unit="m", doc="drawer pull length (the contract needs >= 8 cm open)"),
        _P("pull_speed", 0.04, 0.015, 0.07, unit="m/s", doc="pull speed"),
        _P("knob_close", 1.0, 0.8, 1.0, doc="holder (left) gripper close command"),
        _P("handle_close", 1.0, 0.8, 1.0, doc="puller (right) gripper close command"),
        _P("close_steps", 20, 10, 40, integer=True, unit="steps", doc="closing duration (both grasps)"),
        _P("tip_margin", 0.004, 0.0, 0.012, unit="m", doc="fingertip clearance above the knob / handle plate"),
        _P("pre_scale", 1.0, 0.6, 1.5, doc="pre-grasp distances x (frozen 6 / 4 / 3 cm, tried in order)"),
        _P("post_wait", 5, 0, 20, integer=True, unit="steps", doc="hold steps after the pull"),
        _P("release_steps", 20, 8, 40, integer=True, unit="steps", doc="opening duration (each arm)"),
        _P("release_wait", 0, 0, 15, integer=True, unit="steps", doc="hold steps after opening, before the retreat"),
        _P("back_m", 0.07, 0.03, 0.10, unit="m", doc="retreat distance along -z_tcp"),
    )))

register(Space(
    key="DualFoldTowel-v1", task="DualFoldTowel-v1", expert="imitator.planning.fold:plan_batch", kwargs=dict(),
    notes="E29b fork C fold timeline (planning/fold.py EXPERT); run through research/scripts/a2_cloth.py",
    params=(
        _P("up", 0.10, 0.06, 0.14, unit="m", doc="carry height"),
        _P("low_rel", 0.006, 0.0, 0.02, unit="m", doc="release height relative to the grasp height"),
        _P("retreat", 0.12, 0.08, 0.16, unit="m", doc="final retreat height"),
        _P("d_close", 0.5, 0.3, 1.0, unit="s", doc="close duration"),
        _P("d_lift", 1.0, 0.6, 1.6, unit="s", doc="lift duration"),
        _P("d_carry", 1.2, 0.8, 2.0, unit="s", doc="carry duration"),
        _P("d_lower", 1.6, 1.0, 2.4, unit="s", doc="lowering duration"),
        _P("d_wait", 0.5, 0.0, 1.2, unit="s", doc="still time before opening"),
        _P("d_open", 0.8, 0.4, 1.4, unit="s", doc="opening duration"),
        _P("d_post", 0.5, 0.0, 1.2, unit="s", doc="still time after opening"),
        _P("back", 0.04, 0.0, 0.07, unit="m", doc="back-off along -z_tcp before rising"),
        _P("inset_grasp", 0.025, 0.015, 0.04, unit="m", doc="grasp point inset from the far corner"),
        _P("inset_release", 0.03, 0.01, 0.05, unit="m", doc="release point inset from the near corner"),
        _P("grasp_dz", 0.0, -0.002, 0.006, unit="m", doc="grasp height offset (fingertips 2 mm above the table + this)"),
    )))

register(Space(
    key="FoldTowel-v1", task="FoldTowel-v1", expert="imitator.planning.towel_params:run_towel_p", kwargs=dict(mode="expert"),
    notes="E20 / E29a Panda towel fold (imitator.soft.expert.run_towel: towel_waypoints, read cloth pose); run through "
          "research/scripts/a2_cloth.py",
    params=(
        _P("z_grasp", 0.011, 0.008, 0.016, unit="m", doc="TCP height at the grasp"),
        _P("up", 0.10, 0.06, 0.14, unit="m", doc="carry height"),
        _P("low", 0.035, 0.02, 0.06, unit="m", doc="release height"),
        _P("retreat", 0.12, 0.08, 0.16, unit="m", doc="retreat height"),
        _P("speed", 1.0, 0.6, 1.4, doc="timeline speed factor (segment durations / speed)"),
        _P("t_close", 1.0, 0.5, 1.6, unit="s", doc="close duration"),
        _P("t_open", 1.0, 0.5, 1.6, unit="s", doc="open duration"),
        _P("approach_s", 0.6, 0.3, 1.2, unit="s", doc="transit from the start pose to the first waypoint"),
        _P("settle", 0.0, 0.0, 1.0, unit="s", doc="still time at the end of the timeline"),
        _P("inset_grasp", 0.025, 0.015, 0.04, unit="m", doc="grasp point inset from corner A"),
        _P("inset_release", 0.03, 0.01, 0.05, unit="m", doc="release point inset from corner C"),
    )))

register(Space(
    key="StackCube-v1", task="StackCube-v1", expert="imitator.planning.panda_ms:stack_cube", kwargs=dict(variant="expert"),
    notes="E29b Panda expert (ManiSkill solution + lower onto B + release + retreat); H2 task",
    params=(
        _P("vel", 0.9, 0.4, 1.0, doc="planner joint velocity / acceleration scale"),
        _P("reach_dz", 0.05, 0.03, 0.10, unit="m", doc="pre-grasp distance above the grasp"),
        _P("pre_wait", 0, 0, 20, integer=True, unit="steps", doc="still steps at the pre-grasp pose (CONTACT)"),
        _P("grasp_depth", 0.025, 0.012, 0.035, unit="m", doc="TCP depth below the cube's top face at the grasp"),
        _P("close_steps", 6, 4, 15, integer=True, unit="steps", doc="closing duration"),
        _P("lift_dz", 0.10, 0.05, 0.15, unit="m", doc="lift height"),
        _P("align_dz", 0.05, 0.02, 0.08, unit="m", doc="height above the place pose before lowering"),
        _P("align_wait", 0, 0, 20, integer=True, unit="steps", doc="still steps above B before lowering"),
        _P("place_dz", 0.003, 0.0, 0.01, unit="m", doc="A's bottom above B's top when lowered"),
        _P("place_wait", 3, 0, 15, integer=True, unit="steps", doc="still steps before opening"),
        _P("reaim", 0.0, 0.0, 1.0, doc="switch: re-centre A over B (closed loop) before opening", switch=True),
        _P("release_steps", 6, 4, 15, integer=True, unit="steps", doc="opening duration"),
        _P("up_m", 0.10, 0.05, 0.15, unit="m", doc="retreat height"),
    )))

register(Space(
    key="PushCube-v1", task="PushCube-v1", expert="imitator.planning.panda_ms:push_cube", kwargs=dict(variant="expert"),
    notes="E29b Panda expert (ManiSkill solution + back-off + retreat); H2 task",
    params=(
        _P("vel", 0.9, 0.4, 1.0, doc="planner joint velocity / acceleration scale"),
        _P("start_back", 0.05, 0.03, 0.08, unit="m", doc="TCP start distance behind the cube centre"),
        _P("push_dz", 0.0, -0.01, 0.012, unit="m", doc="TCP height offset from the cube centre during the push"),
        _P("pre_wait", 0, 0, 20, integer=True, unit="steps", doc="still steps behind the cube before the push (CONTACT)"),
        _P("push_end", 0.12, 0.06, 0.14, unit="m", doc="push end: TCP this far behind the goal centre"),
        _P("push_segs", 1, 1, 4, integer=True, doc="push segments; between segments the line is re-centred on the cube"),
        _P("back_m", 0.03, 0.01, 0.06, unit="m", doc="back-off after the push"),
        _P("up_m", 0.10, 0.05, 0.15, unit="m", doc="retreat height"),
    )))
