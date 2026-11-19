"""Deviation model of a learner (E26 'directed DART'), ported from research/scripts/e26_devmodel.py and
veritask/data/osg.py (DeviationModel, match_deviation) without importing them.

Procedure (E26 section 1, prereg_amend2):
  measurement  a learner (e.g. a policy trained on mp data) is rolled out nominally from the start state of each of the
               first N certified expert episodes; every policy step is matched to the nearest step of that episode's
               expert path (arm joint positions of ALL arms jointly, monotone window [j - 5, j + 40] around the last
               match j); deviation = q_policy - q_expert per arm
  phases       the matched expert step's phase per arm (imitator.data.phase annotations stored with the expert data;
               FREE = free transit, CONTACT = everything else)
  statistics   per arm and phase: covariance without the 10 % largest deviations (failure excursions; the untrimmed
               covariance is reported), lag-1 autocorrelation = median over runs of the mean over joints of the
               correlation between consecutive same-phase deviations
  calibration  s_t = the largest of 1 / 0.5 / 0.25 / 0.125 / 0.0625 whose 12-seed dart_dir pilot certifies >= 50 %
               (calibrate_scale)
JSON layout (E26-compatible): {task, embodiment, model: {arm: {free|contact: {cov, rho, n}}}, stats, scale, calibration}.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .phase import CONTACT, FREE

PHASES = (FREE, CONTACT)
DEFAULT_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)


class DeviationModel:
    """per arm and phase: covariance (d x d) and lag-1 autocorrelation rho of the learner's joint deviations; `scale`
    is the calibrated std scale s_t (applied by the noise process, not folded into cov)"""

    def __init__(self, model: dict, scale: float = 1.0, meta: dict | None = None):
        self.d = {int(a): {ph: dict(cov=np.asarray(v["cov"], float), rho=float(v["rho"]), n=int(v.get("n", 0)))
                           for ph, v in m.items()} for a, m in model.items()}
        self.scale = float(scale if scale is not None else 1.0)
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, src, scale: float | None = None) -> "DeviationModel":
        if isinstance(src, DeviationModel):
            return src
        raw = src if isinstance(src, dict) else json.load(open(src))
        sc = raw.get("scale", 1.0) if scale is None else scale
        meta = {k: v for k, v in raw.items() if k not in ("model", "scale")}
        if not isinstance(src, dict):
            meta["path"] = os.path.abspath(src)
        return cls(raw["model"], sc, meta)

    def to_json(self) -> dict:
        model = {str(a): {ph: dict(cov=np.asarray(v["cov"]).tolist(), rho=v["rho"], n=v["n"]) for ph, v in m.items()}
                 for a, m in self.d.items()}
        out = {k: v for k, v in self.meta.items() if k != "path"}
        out.update(model=model, scale=self.scale)
        return out

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)
        return path

    @classmethod
    def isotropic(cls, dofs, sd: dict, rho: dict, meta: dict | None = None, scale: float = 1.0) -> "DeviationModel":
        """iso model: sd / rho = {'free': x, 'contact': y} (e.g. transferred from another embodiment's fit)"""
        model = {a: {ph: dict(cov=np.eye(d) * float(sd[ph]) ** 2, rho=float(rho[ph]), n=0) for ph in PHASES}
                 for a, d in enumerate(dofs)}
        return cls(model, scale, {"kind": "isotropic", **(meta or {})})

    @classmethod
    def transferred_isotropic(cls, src: "DeviationModel", dofs, meta: dict | None = None) -> "DeviationModel":
        """an isotropic model for another embodiment from a fitted one: per phase, iso sd = sqrt of the n-weighted mean
        of trace(C) / d over the source arms, rho = the n-weighted mean of the arms' rho (arms with n = 0 ignored).
        The calibrated scale is NOT transferred (calibrate it on the target embodiment)."""
        sd, rho = {}, {}
        for ph in PHASES:
            w = [(src._m(a, ph)["n"], src.iso_sd(a, ph) ** 2, src.rho(a, ph)) for a in sorted(src.d)]
            w = [x for x in w if x[0] > 0] or [(1, src.iso_sd(a, ph) ** 2, src.rho(a, ph)) for a in sorted(src.d)]
            n = sum(x[0] for x in w)
            sd[ph] = float(np.sqrt(sum(x[0] * x[1] for x in w) / n))
            rho[ph] = float(sum(x[0] * x[2] for x in w) / n)
        m = dict(meta or {}, kind="isotropic, transferred", source=src.meta.get("path") or src.meta.get("task"),
                 source_embodiment=src.meta.get("embodiment"), iso_sd=sd, iso_rho=rho)
        return cls.isotropic(dofs, sd, rho, meta=m, scale=1.0)

    # ------------------------------------------------------------------ access
    def _m(self, arm: int, phase: str) -> dict:
        m = self.d.get(int(arm)) or self.d[min(self.d)]
        return m.get(phase) or m[FREE]

    def cov(self, arm: int, phase: str) -> np.ndarray:
        """the measured covariance (WITHOUT the calibrated scale; the noise process applies scale^2)"""
        return self._m(arm, phase)["cov"]

    def rho(self, arm: int, phase: str) -> float:
        return self._m(arm, phase)["rho"]

    def iso_sd(self, arm: int, phase: str) -> float:
        C = self.cov(arm, phase)
        return float(np.sqrt(np.trace(C) / C.shape[0]))

    def summary(self) -> dict:
        return {f"arm{a}_{ph}": dict(iso_sd_rad=self.iso_sd(a, ph), rho=self.rho(a, ph), n=self._m(a, ph)["n"],
                                     dof=int(self.cov(a, ph).shape[0])) for a in sorted(self.d) for ph in PHASES}


# ================================================================================================ fitting
def match_deviation(q_pol: np.ndarray, q_exp: np.ndarray, back: int = 5, fwd: int = 40):
    """E26 alignment: every policy step is matched to the nearest expert step (all arms jointly) within the monotone
    window [j - back, j + fwd] around the last match j. Returns (deviation (T, D), matched expert index (T,))."""
    q_pol, q_exp = np.asarray(q_pol, float), np.asarray(q_exp, float)
    dev, idx, j = [], [], 0
    for t in range(len(q_pol)):
        lo, hi = max(0, j - back), min(len(q_exp), j + fwd + 1)
        k = lo + int(np.argmin(np.linalg.norm(q_exp[lo:hi] - q_pol[t], axis=1)))
        j = max(j, k)
        dev.append(q_pol[t] - q_exp[k]); idx.append(k)
    return np.asarray(dev), np.asarray(idx, int)


def rollout_deviation(q_pol, q_exp, phase_exp, dofs) -> dict:
    """one rollout -> dict(dev (T, D), phase [[per step] per arm]) with phase codes 0 = FREE, 1 = CONTACT (or str)"""
    dev, k = match_deviation(q_pol, q_exp)
    ph = np.asarray(phase_exp)
    if ph.ndim == 1:
        ph = np.repeat(ph[:, None], len(dofs), 1)
    names = []
    for arm in range(len(dofs)):
        col = ph[np.clip(k, 0, len(ph) - 1), arm]
        names.append([(FREE if (c == 0 or c == FREE) else CONTACT) for c in col])
    return dict(dev=dev, phase=names, dofs=list(dofs))


def fit_from_rollouts(rollouts: list[dict], trim: float = 0.1, meta: dict | None = None) -> DeviationModel:
    """rollouts: [dict(dev (T, sum dofs), phase [[str per step] per arm], dofs)] -> DeviationModel (E26 statistics)"""
    if not rollouts:
        raise ValueError("no rollouts")
    dofs = list(rollouts[0]["dofs"])
    offs = np.r_[0, np.cumsum(dofs)]
    model, stats = {}, {}
    for arm, d in enumerate(dofs):
        model[arm] = {}
        for phase in PHASES:
            X, rho = [], []
            for r in rollouts:
                dv = np.asarray(r["dev"])[:, offs[arm]:offs[arm + 1]]
                m = np.asarray([p == phase for p in r["phase"][arm]])
                X.append(dv[m])
                idx = np.flatnonzero(m[:-1] & m[1:])
                if len(idx) > 5:
                    a0, a1 = dv[idx], dv[idx + 1]
                    c = [np.corrcoef(a0[:, j], a1[:, j])[0, 1] for j in range(d)
                         if a0[:, j].std() > 1e-9 and a1[:, j].std() > 1e-9]
                    if c:
                        rho.append(float(np.mean(c)))
            X = np.concatenate(X) if X else np.zeros((0, d))
            nrm = np.linalg.norm(X, axis=1) if len(X) else np.zeros(0)
            keep = nrm <= np.quantile(nrm, 1.0 - trim) if len(X) > 20 else np.ones(len(X), bool)
            cov_full = np.cov(X.T) if len(X) > 20 else np.eye(d) * 1e-6
            cov = np.cov(X[keep].T) if keep.sum() > 20 else cov_full
            cov = np.atleast_2d(cov)
            model[arm][phase] = dict(cov=cov, rho=float(np.median(rho)) if rho else 0.0, n=int(len(X)))
            stats[f"arm{arm}_{phase}"] = dict(
                n=int(len(X)), iso_sd_rad=float(np.sqrt(np.trace(cov) / d)),
                iso_sd_full_rad=float(np.sqrt(np.trace(np.atleast_2d(cov_full)) / d)), rho=model[arm][phase]["rho"],
                n_rho_runs=len(rho),
                norm_quantiles_rad={str(q): float(np.quantile(nrm, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)} if len(X) else None,
                per_joint_sd=np.sqrt(np.diag(cov)).round(5).tolist(), mean=X.mean(0).round(5).tolist() if len(X) else None)
    m = dict(meta or {})
    m.update(stats=stats, trim=trim, n_rollouts=len(rollouts), phase_rule="imitator.data.phase annotations of the matched "
             "expert step (FREE = free transit, else CONTACT)", alignment="monotone window [j-5, j+40], all arms jointly")
    return DeviationModel(model, 1.0, m)


# ================================================================================================ policy rollouts
def _rollout_worker(job):
    wid, task, emb, episodes, policy, max_factor, root, env_kwargs = job
    from .engine import worker_setup
    worker_setup()
    from ..api import make
    from ..tasks import get_task
    from . import store
    from .policy import as_policy
    spec = get_task(task)
    env = make(task, emb, sim="cpu", render="none", obs="state", **(env_kwargs or {}))
    u = env.unwrapped
    lay = store.Layout.from_env(env)
    pol = as_policy(policy, lay)
    ds = store.open_dataset(root)
    out = []
    for ep in episodes:
        t0 = time.time()
        arr = ds.arrays(ep["_index"], keys=("arm_qpos", "phase", "env_states"))
        q_exp, ph_exp = arr["arm_qpos"][1:], arr["phase"]
        T = len(ph_exp)
        obs = store.reset_to_episode(env, spec, ep, arr.get("env_states"))
        if hasattr(pol, "reset"):
            pol.reset(seed=int(ep["episode_seed"]))
        q_pol, succ = [], []
        n = int(round(max_factor * T)) + 20
        for _ in range(n):
            a = pol(obs)
            obs, _, _, _, info = env.step(lay.unflatten_action(a))
            obs = store.flat_obs(obs)
            q_pol.append(lay.arm_qpos(u))
            s_ = info.get("success", False)
            succ.append(bool(s_[0]) if hasattr(s_, "__len__") else bool(s_))
        r = rollout_deviation(np.asarray(q_pol), q_exp, ph_exp, lay.arm_dofs)
        r.update(seed=int(ep["episode_seed"]), steps=n, expert_steps=T, seconds=round(time.time() - t0, 1),
                 bench_success_last=bool(succ[-1]), bench_success_any=bool(any(succ)))
        r["dev"] = r["dev"].tolist()
        out.append(r)
        print(f"[devmodel w{wid}] seed {ep['episode_seed']}: {n} steps, |dev| median "
              f"{float(np.median(np.linalg.norm(np.asarray(r['dev']), axis=1))):.4f} rad, {time.time() - t0:.1f}s", flush=True)
    env.close()
    return out


def policy_rollouts(task: str, embodiment: str | None, policy, expert: str, n: int = 20, workers: int = 1,
                    max_factor: float = 1.3, env_kwargs: dict | None = None) -> list[dict]:
    """roll `policy` out from the start state of the first n certified episodes of the expert dataset (seed order)"""
    import multiprocessing as mp
    from . import store
    from ..tasks import get_task
    spec = get_task(task)
    emb = embodiment or spec.default_embodiment
    ds = store.open_dataset(expert)
    eps = sorted(ds.episodes, key=lambda e: int(e["episode_seed"]))[:n]
    if not eps:
        raise ValueError(f"{expert}: no certified episodes")
    if callable(policy) and not isinstance(policy, str):
        return _rollout_worker((0, task, emb, eps, policy, max_factor, ds.root, env_kwargs))
    W = max(1, min(int(workers), len(eps)))
    jobs = [(w, task, emb, eps[w::W], policy, max_factor, ds.root, env_kwargs) for w in range(W)]
    from .engine import pool_env
    with pool_env(cuda=False), mp.get_context("spawn").Pool(W) as pool:
        res = pool.map(_rollout_worker, jobs, chunksize=1)
    out = [r for rs in res for r in rs]
    return sorted(out, key=lambda r: r["seed"])


def fit_deviation_model(task: str, embodiment: str | None = None, rollouts: list | None = None, policy=None,
                        expert: str | None = None, n: int = 20, workers: int = 1, out: str | None = None,
                        trim: float = 0.1, max_factor: float = 1.3, env_kwargs: dict | None = None) -> DeviationModel:
    """E26 deviation model. Either `rollouts` (list of dict(dev, phase, dofs) or dict(q_policy, q_expert, phase_expert,
    dofs)) or `policy` (callable(obs) -> action, or 'module:callable') + `expert` (dataset dir with certified expert
    episodes; their seeds / start states are replayed). Writes `out` (JSON) if given."""
    from ..tasks import get_task
    spec = get_task(task)
    emb = embodiment or spec.default_embodiment
    t0 = time.time()
    meta = dict(task=spec.name, embodiment=emb, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if rollouts is None:
        if policy is None or expert is None:
            raise ValueError("fit_deviation_model needs rollouts=... or policy=... and expert=<dataset dir>")
        rollouts = policy_rollouts(spec.name, emb, policy, expert, n=n, workers=workers, max_factor=max_factor,
                                   env_kwargs=env_kwargs)
        meta.update(policy=policy if isinstance(policy, str) else repr(policy)[:200], expert=os.path.abspath(expert),
                    n_demos=len(rollouts), max_factor=max_factor,
                    rollout_seeds=[r["seed"] for r in rollouts],
                    learner_bench_success_last=sum(bool(r.get("bench_success_last")) for r in rollouts),
                    learner_bench_note="benchmark flag at the last rollout step (context only, not a contract verdict)")
    else:
        rr = []
        for r in rollouts:
            if "dev" in r:
                rr.append(r)
            else:
                rr.append(rollout_deviation(r["q_policy"], r["q_expert"], r["phase_expert"], r["dofs"]))
        rollouts = rr
        meta.update(n_demos=len(rollouts), source="rollouts given")
    dm = fit_from_rollouts(rollouts, trim=trim, meta=meta)
    dm.meta["wall_s"] = round(time.time() - t0, 1)
    if out:
        dm.save(out)
        dm.meta["path"] = os.path.abspath(out)
    return dm


def calibrate_scale(task: str, embodiment: str | None = None, devmodel=None, scales=DEFAULT_SCALES, n_seeds: int = 12,
                    seeds=None, threshold: float = 0.5, preset: str = "dart_dir", workers: int = 1,
                    out: str | None = None, write: bool = True, **gen_kw) -> dict:
    """E26 rule: s_t = the largest scale whose `n_seeds`-seed pilot (preset `preset` at that scale) certifies at least
    `threshold`. Scales are tried from the largest down and the search stops at the first that passes. The chosen scale
    and the per-scale counts are written into the deviation-model JSON (write=True and devmodel is a path).
    Returns dict(scale, table, passed)."""
    from .engine import generate
    from .perturb import preset as P
    if devmodel is None:
        raise ValueError("calibrate_scale needs devmodel=<path or DeviationModel>")
    dm = DeviationModel.load(devmodel)
    path = devmodel if isinstance(devmodel, str) else dm.meta.get("path")
    seeds = list(seeds) if seeds is not None else list(range(900_000, 900_000 + int(n_seeds)))
    out = out or os.path.join(os.path.dirname(os.path.abspath(path)) if path else ".", "calibration")
    table, chosen = {}, None
    for s in sorted(scales, reverse=True):
        spec = P(preset, devmodel=dm.to_json(), scale=float(s))
        res = generate(task, embodiment, seeds=seeds, perturb=spec, out=os.path.join(out, f"s{s:g}"), workers=workers,
                       card=False, overwrite=True, **gen_kw)
        k, nn = res["certified"], res["attempted"]
        table[f"{s:g}"] = dict(certified=k, n=nn, rate=k / max(nn, 1))
        print(f"[calibrate] {task} {embodiment} s = {s:g}: {k}/{nn} certified", flush=True)
        if nn and k / nn >= threshold:
            chosen = float(s)
            break
    rule = (f"largest of {sorted(scales, reverse=True)} with >= {threshold:.0%} of a {len(seeds)}-seed {preset} pilot "
            f"certified (E26 prereg_amend2)")
    res = dict(scale=chosen if chosen is not None else float(min(scales)), passed=chosen is not None, table=table,
               rule=rule, seeds=seeds)
    if write and path:
        raw = json.load(open(path))
        raw["scale"] = res["scale"]
        raw["calibration"] = dict(table=table, rule=rule, passed=res["passed"], seeds=[seeds[0], seeds[-1]],
                                  preset=preset, at=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(raw, open(path, "w"), indent=1)
    return res
