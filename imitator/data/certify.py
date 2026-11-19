"""imitator.certify(): judge an external (or imsim) dataset in ManiSkill3 trajectory HDF5 format with a task's contract.

    rep = imitator.certify("path/to/trajectory.h5", "DualCubePlace-v1")                  # default + extra contracts
    rep = imitator.certify(dataset_dir, "PickCube-v1", contract="imsim_pick_cube_v1", workers=8)

Per episode: build the task's env (CPU PhysX, the dataset's control mode) under the task's trace recorder, reset with the
stored seed / reset options, re-apply the stored physical-uncertainty draw (imsim datasets: mass / friction), set the
stored initial state env_states[0] when present, step the stored (executed) actions, compare every replayed env state
with the stored env_states[t] (actor positions [m], articulation joint positions [rad]), then run the evaluator's settle
(+ aftermath) and judge the default contract and the extra contracts exactly as imitator.generate does.
A replay DIVERGES when an actor's position error exceeds `tol_m` (default 1 cm) or a joint position error exceeds
`tol_rad` (0.05 rad) at any step; its verdict is still reported but it is not counted as certified-by-replay. Replays
with any error above `exact_tol` (10 um / 1e-5 rad) are counted as INEXACT (reported, not rejected): on CPU PhysX a
fully stored episode replays exactly, so a small error points to unstored randomisation (E29a external demo: friction
x 0.3 / mass x 3 not stored -> 1.4 mm) or another simulator / backend.
Actions: a flat (T, A) array (FlattenActionSpaceWrapper order, imsim datasets) or a ManiSkill dict group (one dataset per
agent, RecordEpisode with a multi-agent env). Cloth tasks: the towel is not part of env_states; the stored cloth pose
draw is re-applied and only the robot state is compared (sim per the dataset, num_envs = 1).
Output: <out>.json (per-episode rows + summary) and <out>.md.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np

from . import store


def _mode(cm):
    """control mode of an episode row (multi-agent envs store a dict agent -> mode; imitator.make takes one string)"""
    if isinstance(cm, dict):
        v = sorted(set(cm.values()))
        return v[0] if len(v) == 1 else None
    return cm


def _actions(arr: dict, lay: store.Layout) -> list:
    a = arr["actions"]
    if isinstance(a, dict):
        T = len(next(iter(a.values())))
        keys = lay.keys if lay.keys != [None] else list(a.keys())
        return [{k: np.asarray(a[k][t], np.float32) for k in keys} for t in range(T)]
    a = np.asarray(a, np.float32)
    return [lay.unflatten_action(a[t]) for t in range(len(a))]


def _state_error(u, ref: dict) -> tuple[float, float]:
    """(max actor position error [m], max articulation qpos error [rad]) between the env and a stored state dict"""
    cur = store.state_dict_np(u)
    ep, eq = 0.0, 0.0
    for name, s in (ref.get("actors") or {}).items():
        c = (cur.get("actors") or {}).get(name)
        if c is not None:
            ep = max(ep, float(np.linalg.norm(np.asarray(c)[:3] - np.asarray(s)[:3])))
    for name, s in (ref.get("articulations") or {}).items():
        c = (cur.get("articulations") or {}).get(name)
        if c is not None:
            s, c = np.asarray(s), np.asarray(c)
            n = (len(s) - 13) // 2
            if n > 0:
                eq = max(eq, float(np.abs(c[13:13 + n] - s[13:13 + n]).max()))
            ep = max(ep, float(np.linalg.norm(c[:3] - s[:3])))
    return ep, eq


def _worker(job):
    wid, task, emb, path, idxs, cnames, opts = job
    from .engine import certify_verdicts, judge_rigid, verdict_row, worker_setup
    worker_setup(cuda=False)
    from ..api import make
    from ..tasks import _ref, get_task
    spec = get_task(task)
    ds = store.open_dataset(path)
    cm = opts.get("control_mode") or "pd_joint_pos"
    env = make(spec.name, emb, sim="cpu", render="none", obs="state", control_mode=cm, **(opts.get("env_kwargs") or {}))
    u = env.unwrapped
    rec = _ref(spec.recorder)(env)
    lay = ds.layout() or store.Layout.from_env(env)
    contracts = [spec.load_contract(c) for c in cnames]
    derives = {c.task_id: (_ref(spec.derive)(c) if spec.derive else None) for c in contracts}
    rows = []
    for i in idxs:
        ep = ds.episodes[i]
        t0 = time.time()
        row = dict(index=i, episode_id=ep.get("episode_id"), episode_seed=ep.get("episode_seed"))
        judged, err = {}, None
        try:
            arr = ds.arrays(i)
            states = arr.get("env_states")
            acts = _actions(arr, lay)
            store.reset_to_episode(rec, spec, ep, states if opts.get("use_first_state", True) else None,
                                   apply_phys=opts.get("apply_phys", True))
            ep_err, eq_err, first_div = 0.0, 0.0, None
            for t, a in enumerate(acts):
                rec.step(a)
                if states is not None:
                    try:
                        ref = store.index_tree(states, t + 1)
                    except Exception:                                   # noqa: BLE001
                        ref = None
                    if ref is not None:
                        a_e, q_e = _state_error(u, ref)
                        ep_err, eq_err = max(ep_err, a_e), max(eq_err, q_e)
                        if first_div is None and (a_e > opts["tol_m"] or q_e > opts["tol_rad"]):
                            first_div = t + 1
            row.update(steps=len(acts), max_pos_err_m=ep_err if states is not None else None,
                       max_qpos_err_rad=eq_err if states is not None else None, first_divergence_step=first_div,
                       diverged=bool(first_div is not None), states_compared=states is not None)
            judged = judge_rigid(spec, rec, contracts, derives)
        except Exception as e:                                          # noqa: BLE001
            import traceback
            err = repr(e)
            row["error"] = traceback.format_exc()[-1200:]
        cert = certify_verdicts(judged, [c.task_id for c in contracts], None, err)
        row.update(cert, verdicts={cid: verdict_row(j) for cid, j in judged.items()},
                   bench_final=((judged.get(contracts[0].task_id) or {}).get("bench") or {}).get("bench_success_final"),
                   stored_certified=(ep.get("imsim") or {}).get("certified"), seconds=round(time.time() - t0, 2))
        row["certified_by_replay"] = bool(row["certified"] and not row.get("diverged"))
        rows.append(row)
        if opts.get("verbose", True):
            print(f"[certify w{wid}] ep {i} seed {ep.get('episode_seed')}: certified={row['certified']} "
                  f"diverged={row.get('diverged')} {row.get('rejection_cause') or ''} {row['seconds']}s", flush=True)
    ds.close()
    env.close()
    return rows


def _soft_worker(job):
    wid, task, emb, path, idxs, cnames, opts = job
    from .engine import certify_verdicts, verdict_row, worker_setup
    worker_setup(cuda=True)
    from ..api import make
    from ..soft_eval import ClothTraceRecorder, _settle, judge_world
    from ..tasks import get_task
    from .perturb import PhysSpec, apply_cloth_uncertainty
    spec = get_task(task)
    ds = store.open_dataset(path)
    sim = opts.get("sim") or "cpu"
    env = make(spec.name, emb, num_envs=1, sim=sim, render="none", obs="state")
    u = env.unwrapped
    lay = ds.layout() or store.Layout.from_env(env)
    contracts = [spec.load_contract(c) for c in cnames]
    rows = []
    for i in idxs:
        ep = ds.episodes[i]
        t0 = time.time()
        row = dict(index=i, episode_id=ep.get("episode_id"), episode_seed=ep.get("episode_seed"))
        judged, err = {}, None
        try:
            arr = ds.arrays(i)
            states = arr.get("env_states")
            acts = _actions(arr, lay)
            env.reset(seed=int(ep["episode_seed"]))
            d = (ep.get("imsim") or {}).get("phys") or {}
            if d.get("dx") is not None:
                class _R:                                              # replays the stored draw
                    def __init__(self, d):
                        self.v = [d["dx"], d["dy"], d["dyaw_deg"]] + ([d["mass_scale"]] if d.get("mass_scale") else [])
                        self.k = 0

                    def uniform(self, lo, hi, size=None):
                        if size:
                            out = np.array(self.v[self.k:self.k + size]); self.k += size; return out
                        out = self.v[self.k]; self.k += 1; return out
                apply_cloth_uncertainty(u, [_R(d)], PhysSpec(cloth=d.get("cloth", "pose"), mass=(1, 1)), True)
            rec = ClothTraceRecorder(1, u.control_freq)
            rec.set_init(u.cloth.particle_q())
            if states is not None:
                u.set_state_dict(store._to_torch_tree(store.index_tree(states, 0), u.device))
            eq_err, first_div = 0.0, None
            for t, a in enumerate(acts):
                act = a if isinstance(a, dict) else a[None]
                import torch
                _, _, _, _, info = env.step(torch.as_tensor(act, device=u.device) if not isinstance(act, dict) else act)
                rec.record(u.cloth_signals(), u.cloth.particle_q(), info["success"], 0)
                if states is not None:
                    _, q_e = _state_error(u, store.index_tree(states, t + 1))
                    eq_err = max(eq_err, q_e)
                    if first_div is None and q_e > opts["tol_rad"]:
                        first_div = t + 1
            _settle(env, rec, contracts[0])
            tr = rec.traces(0)
            judged = {c.task_id: judge_world(c, tr, u.control_freq) for c in contracts}
            row.update(steps=len(acts), max_qpos_err_rad=eq_err if states is not None else None,
                       first_divergence_step=first_div, diverged=bool(first_div is not None),
                       states_compared=states is not None, note="cloth not in env_states: robot state compared only")
        except Exception as e:                                          # noqa: BLE001
            import traceback
            err = repr(e)
            row["error"] = traceback.format_exc()[-1200:]
        cert = certify_verdicts(judged, [c.task_id for c in contracts], None, err)
        row.update(cert, verdicts={cid: verdict_row(j) for cid, j in judged.items()},
                   stored_certified=(ep.get("imsim") or {}).get("certified"), seconds=round(time.time() - t0, 2))
        row["certified_by_replay"] = bool(row["certified"] and not row.get("diverged"))
        rows.append(row)
        if opts.get("verbose", True):
            print(f"[certify-soft w{wid}] ep {i} seed {ep.get('episode_seed')}: certified={row['certified']} "
                  f"diverged={row.get('diverged')} {row.get('rejection_cause') or ''} {row['seconds']}s", flush=True)
    ds.close()
    env.close()
    return rows


def certify(dataset: str, task: str, contract: str | None = None, embodiment: str | None = None,
            extra_contracts=None, workers: int = 1, out: str | None = None, tol_m: float = 0.01,
            tol_rad: float = 0.05, exact_tol: float = 1e-5, max_episodes: int | None = None, use_first_state: bool = True,
            apply_phys: bool = True, env_kwargs: dict | None = None, verbose: bool = True) -> dict:
    """Replay and judge every episode of a ManiSkill3 trajectory dataset (see module docstring). Returns the summary
    (n, certified_by_replay, diverged, agreement with stored labels, causes) and writes <out>.json / <out>.md."""
    import multiprocessing as mp
    from ..tasks import get_task
    spec = get_task(task)
    ds = store.open_dataset(dataset)
    meta = ds.meta
    kw = (ds.env_info or {}).get("env_kwargs") or {}
    emb = embodiment or meta.get("embodiment") or kw.get("dual_config") or spec.default_embodiment
    if emb not in spec.embodiments:
        raise ValueError(f"{task}: embodiment {emb!r} not supported (supported: {list(spec.embodiments)})")
    c0 = contract or spec.default_contract
    extras = list(extra_contracts if extra_contracts is not None else (spec.extra_contracts if contract is None else ()))
    cnames = [c0] + [c for c in extras if spec.contract_path(c) != spec.contract_path(c0)]
    modes = Counter(_mode(e.get("control_mode")) for e in ds.episodes)
    cm = modes.most_common(1)[0][0] if modes else None
    n = len(ds.episodes) if max_episodes is None else min(len(ds.episodes), int(max_episodes))
    idx = list(range(n))
    ds.close()
    cid0 = spec.load_contract(c0).task_id
    out = out or os.path.join(ds.root, f"certify_{cid0}")
    opts = dict(control_mode=cm or "pd_joint_pos", tol_m=float(tol_m), tol_rad=float(tol_rad), env_kwargs=env_kwargs,
                use_first_state=use_first_state, apply_phys=apply_phys, verbose=verbose, sim=meta.get("sim"))
    W = max(1, min(int(workers), n)) if n else 1
    t0 = time.time()
    fn = _soft_worker if spec.has_cloth else _worker
    if spec.has_cloth:
        opts["sim"] = "gpu" if (meta.get("num_envs") or 1) > 1 else "cpu"
    jobs = [(w, spec.name, emb, ds.h5_path, idx[w::W], cnames, opts) for w in range(W)]
    if n == 0:
        rows = []
    else:
        from .engine import pool_env
        with pool_env(cuda=spec.has_cloth), mp.get_context("spawn").Pool(W) as pool:
            rows = [r for rs in pool.map(fn, jobs, chunksize=1) for r in rs]
    rows.sort(key=lambda r: r["index"])
    wall = time.time() - t0
    cert = sum(r["certified_by_replay"] for r in rows)
    stored = [r for r in rows if r.get("stored_certified") is not None]
    summary = dict(dataset=os.path.abspath(ds.h5_path), task=spec.name, embodiment=emb, contracts=cnames, n=len(rows),
                   certified=sum(bool(r["certified"]) for r in rows), certified_by_replay=cert,
                   rate=cert / max(len(rows), 1), diverged=sum(bool(r.get("diverged")) for r in rows),
                   errors=sum(1 for r in rows if r.get("error")),
                   causes=dict(Counter(r["rejection_cause"] for r in rows if not r["certified"]).most_common()),
                   stored_labels=len(stored),
                   agreement_with_stored=(sum(bool(r["certified"]) == bool(r["stored_certified"]) for r in stored) / len(stored))
                   if stored else None,
                   inexact=sum(1 for r in rows if (r.get("max_pos_err_m") or 0.0) > exact_tol or
                               (r.get("max_qpos_err_rad") or 0.0) > exact_tol),
                   exact_tol=exact_tol,
                   max_pos_err_m=max([r.get("max_pos_err_m") or 0.0 for r in rows], default=None),
                   max_qpos_err_rad=max([r.get("max_qpos_err_rad") or 0.0 for r in rows], default=None),
                   tol_m=tol_m, tol_rad=tol_rad, control_mode=opts["control_mode"], wall_s=round(wall, 1), workers=W,
                   created=time.strftime("%Y-%m-%d %H:%M:%S"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(store.jsonable(dict(summary=summary, episodes=rows)), open(out + ".json", "w"), indent=1)
    with open(out + ".md", "w") as f:
        f.write(render_md(summary, rows))
    if verbose:
        print(json.dumps({k: v for k, v in summary.items() if k != "dataset"}), flush=True)
    return dict(summary, out=out, rows=rows)


def render_md(s: dict, rows: list) -> str:
    L = [f"# Certification report: {os.path.basename(os.path.dirname(s['dataset']))}", "",
         f"- dataset: `{s['dataset']}`", f"- task / embodiment: {s['task']} / {s['embodiment']}",
         f"- contracts: {', '.join(s['contracts'])}",
         f"- episodes: {s['n']}; certified (all contracts accept): {s['certified']}; certified and not diverged: "
         f"{s['certified_by_replay']} ({s['rate']:.0%})",
         f"- diverged replays: {s['diverged']} (actor position error > {s['tol_m']} m or joint error > {s['tol_rad']} rad); "
         f"max errors {s['max_pos_err_m']} m / {s['max_qpos_err_rad']} rad",
         f"- replays that are not exact (an actor or joint error > {s['exact_tol']:g}; CPU PhysX replays of a fully stored "
         f"episode are exact, so this flags unstored randomisation or a different simulator): {s['inexact']}",
         f"- errors: {s['errors']}",
         f"- agreement with stored certification labels: {s['agreement_with_stored']} ({s['stored_labels']} labelled)",
         f"- control mode: {s['control_mode']}; wall {s['wall_s']} s with {s['workers']} workers", "",
         "## Rejection causes", "", "| cause | episodes |", "|---|---|"]
    for k, v in s["causes"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Episodes", "", "| # | seed | certified | diverged | first divergence step | max pos err [m] | cause |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        pe = r.get("max_pos_err_m")
        L.append(f"| {r['index']} | {r.get('episode_seed')} | {r['certified']} | {r.get('diverged')} | "
                 f"{r.get('first_divergence_step')} | {'' if pe is None else f'{pe:.4f}'} | {r.get('rejection_cause') or ''} |")
    return "\n".join(L) + "\n"
