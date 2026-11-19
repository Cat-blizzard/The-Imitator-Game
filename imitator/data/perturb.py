"""Composable perturbations for imitator.generate: action noise on the arm joint targets + physical uncertainty at reset.

PerturbSpec components (all independent; presets below):
  noise     'none' | 'white' (i.i.d. per step) | 'ar1' (x_t = rho x_{t-1} + e_t, stationary covariance = the phase's
            covariance, e_t ~ N(0, C (1 - rho^2)))
  cov       'iso' (sd^2 I per arm) | 'devmodel' (per arm and phase covariance + lag-1 rho of a learner's deviation model,
            imitator.data.devmodel; E26 'directed DART')
  rho       AR(1) coefficient; None = the deviation model's per-phase rho; always clipped to [0, rho_max] (E26: 0.95)
  gating    'all' (CONTACT phases noised with variance x contact_scale; E26: 0.25) | 'free_only' (no noise in CONTACT,
            the AR state is reset to 0 there; E12 DART and E27 --free_only)
  scale     global std scale s (multiplies sd / the deviation-model std); None = the deviation model's calibrated s_t
            (imitator.calibrate_scale), 1 for iso
  phys      PhysSpec or None: reset physical uncertainty, applied generically to every dynamic non-robot actor of the
            ManiSkill scene (pose +-pose_xy_m in x and y, yaw +-yaw_deg, friction x U(friction), mass x U(mass)); cloth
            per `cloth` (see PhysSpec)
Noise goes only on the ARM joint targets (pd_joint_pos), never on the gripper / hand command; the labels stored by the
engine are the clean expert actions. Phases come from the expert's set_phase annotations (imitator.data.phase).

Presets (PRESETS / preset(name, **overrides)):
  mp            plain expert (E26 mp_fresh)
  dart          E12 DART: white N(0, 0.03^2) rad on free transits only (E26 dart_hi)
  dart_dir      E26 directed DART: AR(1) noise, per-phase covariance of a deviation model x s_t^2, CONTACT variance x 0.25,
                + physical uncertainty (+-1 cm, +-10 deg, friction and mass x U(0.7, 1.3)); needs devmodel=...
  phys          plain expert + physical uncertainty (E27 phys_only)
  E27 ablations dart_hi_phys (dart + phys), dart_dir_nophys, dart_dir_white (rho = 0, same stationary covariance),
                dart_dir_freeonly (no noise in CONTACT)
  aliases       mp_fresh = mp, dart_hi = dart, phys_only = phys
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .phase import CONTACT, FREE

NOISES = ("none", "white", "ar1")
COVS = ("iso", "devmodel")
GATINGS = ("all", "free_only")
CLOTH_MODES = ("skip", "pose", "pose_mass")


@dataclass
class PhysSpec:
    """Reset physical uncertainty (E26 dart_dir (iii)). Rigid actors: every dynamic actor in u.scene.actors except the
    task's TaskSpec.phys_exclude (robots are articulations, never touched; kinematic / static actors such as tables and
    visual target markers are not 'movable' and are left alone). Mass and friction are rescaled per episode on CPU
    PhysX only and restored before the next episode (GPU PhysX: pose only, recorded in the draw).
    cloth: 'skip' (towel untouched), 'pose' (rigid offset of every world's towel: +-pose_xy_m, +-yaw_deg about its
    centre; only when the task's expert reads the cloth pose, TaskSpec.cloth_pose_aware, else skipped and logged),
    'pose_mass' (+ particle mass x U(mass) per world). Cloth friction is not perturbed: Newton's body-particle friction
    (soft_contact_mu) is one value for all worlds."""
    pose_xy_m: float = 0.01
    yaw_deg: float = 10.0
    friction: tuple | None = (0.7, 1.3)
    mass: tuple | None = (0.7, 1.3)
    cloth: str = "pose"

    def __post_init__(self):
        if self.cloth not in CLOTH_MODES:
            raise ValueError(f"PhysSpec.cloth must be one of {CLOTH_MODES}")
        self.friction = tuple(self.friction) if self.friction is not None else None
        self.mass = tuple(self.mass) if self.mass is not None else None


@dataclass
class PerturbSpec:
    name: str = "custom"
    noise: str = "none"
    cov: str = "iso"
    sd: float = 0.0
    devmodel: str | dict | None = None
    rho: float | None = None
    rho_max: float = 0.95
    gating: str = "all"
    contact_scale: float = 0.25
    scale: float | None = None
    phys: PhysSpec | None = None
    seed_salt: int = 29

    def __post_init__(self):
        if self.noise not in NOISES:
            raise ValueError(f"noise must be one of {NOISES}")
        if self.cov not in COVS:
            raise ValueError(f"cov must be one of {COVS}")
        if self.gating not in GATINGS:
            raise ValueError(f"gating must be one of {GATINGS}")
        if isinstance(self.phys, dict):
            self.phys = PhysSpec(**self.phys)
        if self.noise != "none" and self.cov == "devmodel" and self.devmodel is None:
            raise ValueError(f"{self.name}: cov='devmodel' needs devmodel=<path or dict> (imitator.fit_deviation_model)")
        if self.noise == "ar1" and self.rho is None and self.cov == "iso":
            raise ValueError("noise='ar1' with cov='iso' needs an explicit rho")

    # ------------------------------------------------------------------ helpers
    @property
    def has_noise(self) -> bool:
        return self.noise != "none" and (self.cov == "devmodel" or self.sd > 0)

    def replace(self, **kw) -> "PerturbSpec":
        d = copy.deepcopy(self)
        for k, v in kw.items():
            if k not in {f.name for f in fields(self)}:
                raise TypeError(f"PerturbSpec has no field {k!r}")
            setattr(d, k, v)
        d.__post_init__()
        return d

    def to_dict(self) -> dict:
        d = asdict(self)
        if isinstance(self.devmodel, dict):
            d["devmodel"] = "<inline deviation model>"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PerturbSpec":
        d = dict(d)
        if d.get("phys") is not None and not isinstance(d["phys"], PhysSpec):
            d["phys"] = PhysSpec(**d["phys"])
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})

    def describe(self) -> str:
        parts = []
        if self.has_noise:
            c = f"iso sd {self.sd:g} rad" if self.cov == "iso" else "deviation-model covariance"
            r = "" if self.noise == "white" else (f", rho {self.rho:g}" if self.rho is not None else ", per-phase rho")
            g = "FREE phases only" if self.gating == "free_only" else f"all phases (CONTACT variance x {self.contact_scale:g})"
            s = "" if self.scale in (None, 1.0) and self.cov == "iso" else (
                f", s = {self.scale:g}" if self.scale is not None else ", s = calibrated s_t")
            parts.append(f"{self.noise} noise on arm joint targets: {c}{r}{s}; {g}")
        if self.phys is not None:
            p = self.phys
            parts.append(f"reset physical uncertainty: rigid movable actors pose +-{p.pose_xy_m * 100:g} cm, yaw "
                         f"+-{p.yaw_deg:g} deg{', friction x U' + str(p.friction) if p.friction else ''}"
                         f"{', mass x U' + str(p.mass) if p.mass else ''}; cloth: {p.cloth}"
                         f"{' (same pose ranges)' if p.cloth == 'pose' else ' (same pose ranges, particle mass)' if p.cloth == 'pose_mass' else ''}")
        return "; ".join(parts) or "none (plain expert)"


def _p(name, **kw) -> PerturbSpec:
    return PerturbSpec(name=name, **kw)


_DART = dict(noise="white", cov="iso", sd=0.03, gating="free_only")
_DIR = dict(noise="ar1", cov="devmodel", gating="all", contact_scale=0.25, scale=None)
_PRESET_KW = {
    "mp": dict(),
    "dart": dict(_DART),
    "dart_dir": dict(_DIR, phys="default"),
    "phys": dict(phys="default"),
    "dart_hi_phys": dict(_DART, phys="default"),
    "dart_dir_nophys": dict(_DIR),
    "dart_dir_white": dict(_DIR, noise="white", phys="default"),
    "dart_dir_freeonly": dict(_DIR, gating="free_only", phys="default"),
}
ALIASES = {"mp_fresh": "mp", "dart_hi": "dart", "phys_only": "phys"}
PRESETS = tuple(_PRESET_KW) + tuple(ALIASES)
NEEDS_DEVMODEL = {k for k, v in _PRESET_KW.items() if v.get("cov") == "devmodel"}


def preset(name: str, **overrides) -> PerturbSpec:
    """PerturbSpec of a named preset; keyword overrides (e.g. devmodel=..., scale=..., phys=PhysSpec(cloth='skip'))"""
    base = ALIASES.get(name, name)
    if base not in _PRESET_KW:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    kw = dict(_PRESET_KW[base])
    if kw.get("phys") == "default":
        kw["phys"] = PhysSpec()
    if base in NEEDS_DEVMODEL and "devmodel" not in overrides:
        kw["devmodel"] = "<required>"
    kw.update(overrides)
    if kw.get("devmodel") == "<required>":
        raise ValueError(f"preset {name!r} needs devmodel=<path or dict> (imitator.fit_deviation_model / "
                         f"`python -m imitator devmodel`)")
    return PerturbSpec(name=name, **kw)


def as_spec(perturb, **overrides) -> PerturbSpec:
    if perturb is None:
        return preset("mp")
    if isinstance(perturb, PerturbSpec):
        return perturb.replace(**overrides) if overrides else perturb
    if isinstance(perturb, str):
        return preset(perturb, **overrides)
    if isinstance(perturb, dict):
        return PerturbSpec.from_dict({**perturb, **overrides})
    raise TypeError(f"perturb must be a PerturbSpec, preset name or dict, got {type(perturb)}")


# ================================================================================================ action noise
class NoiseProcess:
    """Per-episode noise state for every arm. offsets(phases) -> list (per arm) of a joint-target offset or None.
    Matches E26 DartDirSession._step: rho = 0 for 'white' else clip(rho(arm, phase), 0, rho_max); variance scale
    contact_scale in CONTACT (gating 'all'); innovation ~ N(0, C * scale_phase * (1 - rho^2)); x = rho x + innovation.
    gating 'free_only': in CONTACT the state is reset to 0 and no offset is applied."""

    def __init__(self, spec: PerturbSpec, dofs: list[int], devmodel=None):
        self.spec, self.dofs = spec, [int(d) for d in dofs]
        self.dev = devmodel
        if spec.has_noise and spec.cov == "devmodel":
            if devmodel is None:
                from .devmodel import DeviationModel
                self.dev = DeviationModel.load(spec.devmodel)
            for arm, d in enumerate(self.dofs):
                for ph in (FREE, CONTACT):
                    n = self.dev.cov(arm, ph).shape[0]
                    if n != d:
                        raise ValueError(f"deviation model arm {arm} ({ph}) is {n}-D, the embodiment's arm {arm} has {d} "
                                         f"joints (fit a model for this embodiment)")
        s = spec.scale
        if s is None:
            s = float(self.dev.scale) if (self.dev is not None and spec.cov == "devmodel") else 1.0
        self.s = float(s)
        self._chol = {}
        self.rng = np.random.default_rng(0)
        self.x = [np.zeros(d) for d in self.dofs]

    def reset(self, rng):
        self.rng = rng
        self.x = [np.zeros(d) for d in self.dofs]

    def cov(self, arm: int, phase: str) -> np.ndarray:
        if self.spec.cov == "iso":
            return np.eye(self.dofs[arm]) * (self.spec.sd * self.s) ** 2
        return np.asarray(self.dev.cov(arm, phase), float) * self.s ** 2

    def rho(self, arm: int, phase: str) -> float:
        if self.spec.noise == "white":
            return 0.0
        r = self.spec.rho if self.spec.rho is not None else self.dev.rho(arm, phase)
        return float(np.clip(r, 0.0, self.spec.rho_max))

    def _L(self, arm, phase):
        k = (arm, phase)
        if k not in self._chol:
            C = self.cov(arm, phase)
            C = 0.5 * (C + C.T)
            w, V = np.linalg.eigh(C)                      # PSD square root (robust for near-singular covariances)
            self._chol[k] = V * np.sqrt(np.clip(w, 0.0, None))
        return self._chol[k]

    def offsets(self, phases) -> list:
        out = []
        if not self.spec.has_noise:
            return [None] * len(self.dofs)
        for arm, ph in enumerate(phases):
            if ph == CONTACT and self.spec.gating == "free_only":
                self.x[arm] = np.zeros(self.dofs[arm])
                out.append(None)
                continue
            rho = self.rho(arm, ph)
            var = (self.spec.contact_scale if ph == CONTACT else 1.0) * (1.0 - rho ** 2)
            e = self._L(arm, ph) @ self.rng.standard_normal(self.dofs[arm]) * np.sqrt(var)
            self.x[arm] = rho * self.x[arm] + e
            out.append(self.x[arm].copy())
        return out


# ================================================================================================ physical uncertainty
def _yaw_quat(a: float) -> np.ndarray:
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def movable_actors(u, exclude=()) -> list:
    """(name, Actor) of every dynamic non-robot actor of the scene, sorted by name (draw order is part of the seed)"""
    out = []
    for name, a in sorted(getattr(u.scene, "actors", {}).items()):
        if name in exclude or getattr(a, "px_body_type", None) != "dynamic":
            continue
        out.append((name, a))
    return out


def _orig(u, name, a):
    """cached original (mass, [(static, dynamic, restitution) per collision shape]) of an actor's CPU body"""
    cache = u.__dict__.setdefault("_imsim_phys_orig", {})
    body = a._bodies[0]
    key = (name, id(body))
    if key not in cache:
        mats = [sh.get_physical_material() for sh in body.get_collision_shapes()]
        cache[key] = (float(body.mass), [(m.static_friction, m.dynamic_friction, m.restitution) for m in mats])
    return body, cache[key]


def restore_physics(u, exclude=()) -> None:
    """undo the previous episode's mass / friction scaling (actors persist across resets)"""
    import sapien
    if getattr(u, "gpu_sim_enabled", False):
        return
    for name, a in movable_actors(u, exclude):
        try:
            body, (m0, mats) = _orig(u, name, a)
            body.mass = m0
            for sh, (sf, df, r) in zip(body.get_collision_shapes(), mats):
                sh.set_physical_material(sapien.physx.PhysxMaterial(sf, df, r))
        except Exception:                                              # noqa: BLE001
            pass


def apply_physical_uncertainty(u, rng, phys: PhysSpec, exclude=()) -> dict:
    """draw and apply the reset physical uncertainty to every movable actor (call right after env.reset; CPU sim:
    single env). Returns the draw {actor: {dx, dy, dyaw_deg, mass_scale, friction_scale}}."""
    import sapien
    import torch
    from mani_skill.utils.structs.pose import Pose
    out = {}
    gpu = bool(getattr(u, "gpu_sim_enabled", False))
    for name, a in movable_actors(u, exclude):
        d = {}
        dx, dy = (float(v) for v in rng.uniform(-phys.pose_xy_m, phys.pose_xy_m, 2))
        dyaw = float(rng.uniform(-phys.yaw_deg, phys.yaw_deg))
        fm = float(rng.uniform(*phys.mass)) if phys.mass else None
        ff = float(rng.uniform(*phys.friction)) if phys.friction else None
        try:
            p = a.pose.p.detach().cpu().numpy().astype(np.float64)
            q = a.pose.q.detach().cpu().numpy().astype(np.float64)
            p[:, 0] += dx; p[:, 1] += dy
            q = np.stack([_qmul(_yaw_quat(np.radians(dyaw)), qq) for qq in q])
            a.set_pose(Pose.create_from_pq(torch.as_tensor(p, dtype=torch.float32, device=u.device),
                                           torch.as_tensor(q, dtype=torch.float32, device=u.device)))
            d.update(dx=dx, dy=dy, dyaw_deg=dyaw)
        except Exception as e:                                          # noqa: BLE001
            d.update(pose_error=repr(e)[:160])
        if gpu:
            d.update(mass_scale=None, friction_scale=None, note="GPU PhysX: mass / friction not rescaled")
        else:
            try:
                body, (m0, mats) = _orig(u, name, a)
                if fm is not None:
                    body.mass = m0 * fm
                if ff is not None:
                    for sh, (sf, df, r) in zip(body.get_collision_shapes(), mats):
                        sh.set_physical_material(sapien.physx.PhysxMaterial(sf * ff, df * ff, r))
                d.update(mass_scale=fm, friction_scale=ff, mass_kg=m0 * (fm or 1.0))
            except Exception as e:                                      # noqa: BLE001
                d.update(phys_error=repr(e)[:160])
        out[name] = d
    if gpu and out:
        sc = u.scene
        sc._gpu_apply_all()
        sc.px.gpu_update_articulation_kinematics()
        sc._gpu_fetch_all()
    return out


def apply_cloth_uncertainty(u, rngs: list, phys: PhysSpec, pose_aware: bool) -> list[dict]:
    """per world: rigid offset of the towel (xy +-pose_xy_m, yaw +-yaw_deg about its centroid) and, for 'pose_mass',
    particle mass x U(mass). rngs: one generator per world. Applied right after env.reset (the towel is flat at rest)."""
    if phys is None or phys.cloth == "skip" or getattr(u, "cloth", None) is None:
        return [dict(cloth="skip")] * u.num_envs
    if not pose_aware:
        return [dict(cloth="skipped: the task's expert does not read the cloth pose (TaskSpec.cloth_pose_aware)")] * u.num_envs
    import warp as wp
    cl = u.cloth
    n, ppw = cl.n, cl.ppw
    draws = []
    q = cl.s0.particle_q.numpy().reshape(n, ppw, 3)
    mass = cl.model.particle_mass.numpy().reshape(n, ppw)
    for w in range(n):
        r = rngs[w]
        dx, dy = (float(v) for v in r.uniform(-phys.pose_xy_m, phys.pose_xy_m, 2))
        dyaw = float(r.uniform(-phys.yaw_deg, phys.yaw_deg))
        fm = float(r.uniform(*phys.mass)) if (phys.cloth == "pose_mass" and phys.mass) else None
        c = q[w, :, :2].mean(0)
        a = np.radians(dyaw)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        q[w, :, :2] = (q[w, :, :2] - c) @ R.T + c + [dx, dy]
        if fm is not None:
            mass[w] *= fm
        draws.append(dict(cloth=phys.cloth, dx=dx, dy=dy, dyaw_deg=dyaw, mass_scale=fm))
    qa = q.reshape(-1, 3).astype(np.float32)
    for s in (cl.s0, cl.s1):
        s.particle_q.assign(wp.array(qa, dtype=wp.vec3, device=cl.device))
    if phys.cloth == "pose_mass":
        m = mass.reshape(-1).astype(np.float32)
        cl.model.particle_mass.assign(m)
        cl.model.particle_inv_mass.assign(np.where(m > 0, 1.0 / np.maximum(m, 1e-12), 0.0).astype(np.float32))
    try:                                   # cold-start VBD's contact / history state at the moved rest pose
        wm = np.ones(n + 1, bool); wm[n] = False
        cl.solver.reset(cl.s0, world_mask=wp.array(wm, dtype=wp.bool, device=cl.device))
    except Exception:                                                   # noqa: BLE001
        pass
    cl._signals()
    return draws


def spec_json(spec: PerturbSpec) -> str:
    return json.dumps(spec.to_dict(), sort_keys=True)
