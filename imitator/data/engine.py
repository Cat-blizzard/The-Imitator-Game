"""imitator.generate(): contract-certified datasets for any registered imsim task, with composable perturbations.

    ds = imitator.generate("DualCubePlace-v1", "dual_ur5_wsg", seeds=range(50), perturb="dart", out=dir, workers=8)
    ds = imitator.generate("DualCubePlace-v1", seeds=range(50), perturb=imitator.PerturbSpec(noise="ar1", cov="iso", sd=0.01,
                        rho=0.9, gating="all", phys=imitator.PhysSpec()), out=dir)
    ds = imitator.generate("FoldTowel-v1", seeds=range(50), perturb="phys", out=dir, num_envs=25)   # cloth: batched worlds

Per episode (rigid tasks; CPU PhysX, one env per worker process, `workers` processes):
  1. reset(seed, TaskSpec.reset_options(seed)); clear the phase annotations; restore the previous episode's physics;
  2. reset physical uncertainty (PerturbSpec.phys) on every dynamic non-robot actor; the draw is stored;
  3. the task's expert (TaskSpec.data_script, else scripts[data_expert]) drives a PerturbingEnv: before every env.step
     the expert's phase annotation of each arm is read, the noise process adds its offset to that arm's joint targets
     (never to the gripper), the executed action goes to the task's trace recorder, and executed action, clean action
     (label), phases, state observation, env state, arm joint positions (and optionally RGB) are recorded;
  4. certification: the evaluator's settle (+ retreat aftermath) and the verdict of the task's default contract plus its
     extra_contracts (TaskSpec.judge or v31.judge_rec, exactly as imitator.evaluate); kept only if every contract accepts
     and the expert reported status 'ok';
  5. storage: imitator.data.store (ManiSkill3 trajectory HDF5 + JSON + per-episode imsim metadata), dataset card.
Cloth tasks (TaskSpec.has_cloth): Newton needs CUDA, so a worker runs `num_envs` worlds at once (sim 'gpu', or 'cpu'
PhysX + CUDA cloth with num_envs = 1); every world is one episode (own seed, own noise, own cloth pose draw); verdicts
per world through soft_eval's recorder / settle / judge (protocol v3).
Seeds: noise and physical draws come from numpy generators seeded by (seed_salt, episode seed, stream), so an
episode is reproducible from its seed alone; the env itself is reset with the episode seed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, OrderedDict

import numpy as np

from .phase import CONTACT, FREE, get_phase, reset_phase
from .perturb import (NoiseProcess, PerturbSpec, apply_cloth_uncertainty, apply_physical_uncertainty, as_spec,
                      restore_physics)
from . import store

from ..compat import sapien_home
from ..paths import out_dir

DEFAULT_OUT = out_dir("datasets")
GENERATOR = "imitator.data.engine v1 (E29a)"
SOFT_SCRIPTS = {"FoldTowel-v1": ("imitator.soft.expert:run_towel", dict(mode="expert"), True),
                "DualFoldTowel-v1": ("imitator.data.engine:fold_driver", dict(behaviour="expert"), False)}


# ================================================================================================ process setup
def worker_setup(cuda: bool = False, threads: int = 1):
    """worker process: one BLAS / torch thread, no CUDA unless needed, HOME inside the project (SAPIEN caches)"""
    if not cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    if sapien_home():
        os.environ["HOME"] = sapien_home()
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:                                                   # noqa: BLE001
        pass
    import warnings
    warnings.filterwarnings("ignore")


class pool_env:
    """context: environment inherited by spawned worker processes (one BLAS / OpenMP thread each, HOME inside the
    project, optionally no CUDA); worker_setup() repeats it inside the worker, but numpy reads the thread caps at
    import, i.e. before the worker function runs"""

    def __init__(self, cuda: bool = False):
        self.cuda = cuda
        self.saved = {}

    def __enter__(self):
        new = {v: "1" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        if not self.cuda:
            new["CUDA_VISIBLE_DEVICES"] = ""
        if sapien_home():
            new["HOME"] = sapien_home()
        for k, v in new.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def code_hash() -> dict:
    """sha256 over the imsim package sources (sorted relative paths + contents) and over the data engine alone"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h_all, h_data = hashlib.sha256(), hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            if f.endswith((".py", ".json")):
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, root)
                b = open(p, "rb").read()
                h_all.update(rel.encode() + b"\0" + b)
                if rel.startswith("data" + os.sep):
                    h_data.update(rel.encode() + b"\0" + b)
    return dict(imsim=h_all.hexdigest()[:16], imsim_data=h_data.hexdigest()[:16])


def file_sha256(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def expert_params_tag() -> str:
    """which expert parameter sets were active: IMSIM_EXPERT_PARAMS mode + hash of planning/tuned_params.json (A2)"""
    import hashlib
    mode = os.environ.get("IMSIM_EXPERT_PARAMS", "adopted")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planning", "tuned_params.json")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12] if os.path.exists(path) else "none"
    return f"{mode}:{h}"


def expert_of(spec):
    """(callable(env) -> log, name, cloth_pose_aware) of the task's data-engine expert"""
    from ..tasks import _ref
    if spec.data_script:
        f = _ref(spec.data_script)
        return (lambda env: f(env)), spec.data_script, bool(spec.cloth_pose_aware)
    if spec.has_cloth and spec.name in SOFT_SCRIPTS:
        fn, kw, aware = SOFT_SCRIPTS[spec.name]
        f = _ref(fn)
        return (lambda env: f(env, **kw)), fn, bool(spec.cloth_pose_aware or aware)
    name = spec.behaviour(spec.data_expert)
    return spec.script(name), name, bool(spec.cloth_pose_aware)


def fold_driver(env, behaviour: str = "expert"):
    """DualFoldTowel-v1 expert for the data engine: planning/fold.plan_batch (E29b's planner; no phase annotation,
    nominal towel corners) stepped through env.step"""
    import torch
    from ..planning.fold import plan_batch
    u = env.unwrapped
    n = u.num_envs
    Q, G, info = plan_batch(u, [behaviour] * n)
    keys = list(u.single_action_space.spaces.keys())
    for k in range(len(Q[0])):
        act = {keys[i]: torch.as_tensor(np.concatenate([Q[i][k], G[i][k]], -1), device=u.device) for i in (0, 1)}
        if n == 1:
            act = {kk: v[0] for kk, v in act.items()}
        env.step(act)
    return dict(status="ok", steps=len(Q[0]))


# ================================================================================================ perturbing env
class PerturbingEnv:
    """Between the expert and the recorder: reads the phase annotation of every arm, adds the noise process's offset to
    the arm joint targets of the executed action (gripper untouched), steps the wrapped env with it and records
    executed / clean actions, phases, observations, env states, arm joint positions, flags (and RGB) per step.
    Batched envs (num_envs worlds): one noise process per world; phases are shared (one expert timeline).
    Without noise the expert's action object is passed through unchanged."""

    def __init__(self, env, layout: store.Layout, noises: list, rgb=None, post_step=None, record_states: bool = True):
        self.env = env
        self.unwrapped = env.unwrapped
        self.layout = layout
        self.noises = noises
        self.rgb = rgb
        self.post_step = post_step
        self.record_states = record_states
        self.N = int(getattr(self.unwrapped, "num_envs", 1))
        self.buf = None
        self.n_steps = 0
        self.arm_flat = [[o + j for j in range(s, e)] for o, (s, e) in zip(layout.offsets, layout.arm_slices)]
        self.has_noise = any(n.spec.has_noise for n in noises)

    # ------------------------------------------------------------------ episode
    def begin(self, obs0, rngs: list):
        u = self.unwrapped
        for n, r in zip(self.noises, rngs):
            n.reset(r)
        b = dict(actions=[], actions_clean=[], phase=[], rewards=[], success=[], terminated=[], truncated=[], fail=[],
                 obs=[self._obs(obs0)], arm_qpos=[self.layout.arm_qpos_batch(u)])
        if self.record_states:
            b["env_states"] = [store.state_dict_np(u, "all")]
        if self.rgb is not None:
            b["rgb"] = [self.rgb()]
        self.buf = b
        self.n_steps = 0

    def _obs(self, obs):
        if isinstance(obs, dict):
            obs = obs.get("state", obs)
        a = obs.detach().cpu().numpy() if hasattr(obs, "detach") else np.asarray(obs)
        a = np.asarray(a, np.float32)
        return a[None] if a.ndim == 1 else a

    def phases(self) -> list[str]:
        return [get_phase(self, i) for i in range(self.layout.n_arms)]

    # ------------------------------------------------------------------ step
    def _env_action(self, exe: np.ndarray, like):
        """executed flat actions (N, A) -> the action structure the expert used"""
        import torch
        lay = self.layout
        if isinstance(like, dict):
            out = OrderedDict()
            for k, o, d in zip(lay.keys, lay.offsets, lay.dims):
                x = exe[:, o:o + d]
                ref = like[k]
                if hasattr(ref, "detach"):
                    t = torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
                    out[k] = t if ref.ndim == 2 else t[0]
                else:
                    out[k] = x.astype(np.asarray(ref).dtype) if np.asarray(ref).ndim == 2 else x[0].astype(np.float32)
            return out
        if hasattr(like, "detach"):
            t = torch.as_tensor(exe, dtype=like.dtype, device=like.device)
            return t if like.ndim == 2 else t[0]
        return exe if np.asarray(like).ndim == 2 else exe[0]

    def step(self, action):
        u = self.unwrapped
        clean = self.layout.flatten_batch(action)
        if clean.shape[0] == 1 and self.N > 1:
            clean = np.repeat(clean, self.N, 0)
        ph = self.phases()
        exe = clean.copy()
        noised = False
        if self.has_noise:
            for w, npr in enumerate(self.noises):
                for arm, off in enumerate(npr.offsets(ph)):
                    if off is not None:
                        exe[w, self.arm_flat[arm]] += off.astype(np.float32)
                        noised = True
        act = self._env_action(exe, action) if noised else action
        out = self.env.step(act)
        obs, rew, term, trunc, info = out
        if self.post_step is not None:
            self.post_step(out)
        b = self.buf
        if b is not None:
            b["actions"].append(exe)
            b["actions_clean"].append(clean)
            b["phase"].append(np.array([0 if p == FREE else 1 for p in ph], np.uint8))
            b["obs"].append(self._obs(obs))
            b["arm_qpos"].append(self.layout.arm_qpos_batch(u))
            b["rewards"].append(_vec(rew, self.N, np.float32))
            b["success"].append(_vec(info.get("success", False), self.N, bool))
            b["fail"].append(_vec(info.get("fail", False), self.N, bool))
            b["terminated"].append(_vec(term, self.N, bool))
            b["truncated"].append(_vec(trunc, self.N, bool))
            if self.record_states:
                b["env_states"].append(store.state_dict_np(u, "all"))
            if self.rgb is not None:
                b["rgb"].append(self.rgb())
        self.n_steps += 1
        return out

    def arrays(self, w: int = 0) -> dict:
        """recorded arrays of world w in the store layout"""
        b = self.buf
        T = len(b["actions"])
        A = self.layout.act_dim
        out = dict(actions=np.stack([x[w] for x in b["actions"]]).astype(np.float32) if T else np.zeros((0, A), np.float32),
                   actions_clean=np.stack([x[w] for x in b["actions_clean"]]).astype(np.float32) if T else np.zeros((0, A), np.float32),
                   phase=np.stack(b["phase"]).astype(np.uint8) if T else np.zeros((0, self.layout.n_arms), np.uint8),
                   terminated=np.array([x[w] for x in b["terminated"]], bool),
                   truncated=np.array([x[w] for x in b["truncated"]], bool),
                   success=np.array([x[w] for x in b["success"]], bool),
                   rewards=np.array([x[w] for x in b["rewards"]], np.float32),
                   obs=np.stack([x[w] for x in b["obs"]]).astype(np.float32),
                   arm_qpos=np.stack([x[w] for x in b["arm_qpos"]]).astype(np.float32))
        if any(x[w] for x in b["fail"]):
            out["fail"] = np.array([x[w] for x in b["fail"]], bool)
        if "env_states" in b:
            out["env_states"] = store.stack_dicts([store.index_tree(s, w) for s in b["env_states"]])
        if "rgb" in b:
            out["rgb"] = np.stack([x[w] for x in b["rgb"]]).astype(np.uint8)
        return out

    def noise_stats(self, w: int = 0) -> dict:
        b = self.buf
        if not b["actions"]:
            return {}
        d = np.stack([x[w] for x in b["actions"]]) - np.stack([x[w] for x in b["actions_clean"]])
        ph = np.stack(b["phase"])
        out = {}
        for arm, idx in enumerate(self.arm_flat):
            da = d[:, idx]
            on = np.abs(da).max(1) > 0
            out[f"arm{arm}"] = dict(rms_rad=float(np.sqrt((da ** 2).mean())), max_abs_rad=float(np.abs(da).max()),
                                    free_frac=float((ph[:, arm] == 0).mean()), noised_frac=float(on.mean()),
                                    rms_noised_rad=float(np.sqrt((da[on] ** 2).mean())) if on.any() else 0.0)
        g = self.layout.gripper_dims_flat()
        out["gripper_max_abs"] = float(np.abs(d[:, g]).max()) if g else 0.0
        return out

    def __getattr__(self, name):
        return getattr(self.env, name)


def _vec(x, n, dtype):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=dtype).reshape(-1)
    if a.size == 1 and n > 1:
        a = np.repeat(a, n)
    return a


def rgb_grabber(env, size=None):
    """env.render() of every world as (N, H, W, 3) uint8 (SAPIEN human camera on CPU / lavapipe, Newton camera),
    resized (nearest) to size = (W, H) when given"""
    def grab():
        img = env.render()
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 3:
            img = img[None]
        img = img[..., :3]
        if size is not None and (img.shape[2], img.shape[1]) != tuple(size):
            yi = np.linspace(0, img.shape[1] - 1, size[1]).round().astype(int)
            xi = np.linspace(0, img.shape[2] - 1, size[0]).round().astype(int)
            img = img[:, yi][:, :, xi]
        return img.astype(np.uint8)
    return grab


def _make_rigid(spec, emb, rgb: bool, render, env_kwargs, rgb_size):
    from ..api import make
    kw = dict(env_kwargs or {})
    if rgb and "render_size" not in kw:
        try:                                    # imsim tabletop envs render the human camera at render_size
            return make(spec.name, emb, sim="cpu", render=render, obs="state", render_size=tuple(rgb_size), **kw)
        except TypeError:
            pass
    return make(spec.name, emb, sim="cpu", render=(render if rgb else "none"), obs="state", **kw)


# ================================================================================================ verdicts
def _plan_cause(status) -> str:
    s = str(status)
    return "plan:" + s.split(":")[0].strip().replace(" ", "_")[:40]


def protocol_reason(j: dict) -> str:
    """short name of a protocol rejection (window status when known, else the start of the message)"""
    ws = (j.get("info") or {}).get("window_status") or (j.get("window") or {}).get("status")
    if ws and ws != "ok":
        return str(ws)
    t = str(j.get("protocol_reject")).lower()
    for key, name in (("quiescent", "not_quiescent"), ("missing", "missing_signal"), ("released", "not_released"),
                      ("incomplete", "incomplete")):
        if key in t:
            return name
    return t[:40].replace(" ", "_")


def contract_cause(cid: str, j: dict) -> str | None:
    if j.get("success") is True:
        return None
    if j.get("protocol_reject"):
        return f"{cid}:protocol:{protocol_reason(j)}"
    if j.get("success") is None:
        ws = (j.get("info") or {}).get("window_status") or (j.get("window") or {}).get("status") or "unknown"
        return f"{cid}:unknown:{ws}"
    if j.get("released") is False:
        return f"{cid}:not_released"
    return f"{cid}:{j.get('binding_term') or 'failed'}"


def verdict_row(j: dict) -> dict:
    return dict(success=j.get("success"), released=j.get("released"), binding=j.get("binding_term"),
                protocol_reject=j.get("protocol_reject"), margin=j.get("margin"), margins=j.get("margins"),
                terms=j.get("terms"), window_status=(j.get("info") or {}).get("window_status"),
                aftermath=(j.get("aftermath") or {}).get("value") if isinstance(j.get("aftermath"), dict) else None)


def judge_rigid(spec, rec, contracts, derives) -> dict:
    """settle (+ aftermath) + verdicts of all contracts on the recorder's traces (as harness.evaluate)"""
    from ..tasks import _ref
    from ..verify import v31
    u = rec.env.unwrapped
    C0 = contracts[0]
    if spec.judge:
        j0 = _ref(spec.judge)(rec, C0, derive=derives[C0.task_id])
    else:
        obj, half = getattr(u, spec.object_attr), getattr(u, spec.half_attr)
        bases = u.dual.mount_poses()
        j0 = v31.judge_rec(rec, C0, obj, half, bases, derive=derives[C0.task_id])
    out = {C0.task_id: j0}
    for c in contracts[1:]:
        ex_m = dict(j0.get("extra") or {}) if v31.retreat_cfg(c) is not None else {}
        out[c.task_id] = v31.judge_traces(c, rec.traces, float(u.control_freq), ex_m, derive=derives[c.task_id])
    return out


def certify_verdicts(judged: dict, contract_ids: list, plan_status=None, error=None) -> dict:
    causes = []
    if error is not None:
        causes.append("error:" + str(error).split("(")[0][:60])
    if plan_status not in (None, "ok"):
        causes.append(_plan_cause(plan_status))
    for cid in contract_ids:
        j = judged.get(cid)
        if j is None:
            if error is None:
                causes.append(f"{cid}:not_judged")
            continue
        c = contract_cause(cid, j)
        if c:
            causes.append(c)
    return dict(certified=not causes, rejection_cause=causes[0] if causes else None, causes=causes)


# ================================================================================================ rigid worker
def _rng(spec_p: PerturbSpec, seed: int, stream: int):
    return np.random.default_rng([int(spec_p.seed_salt), int(seed) & 0x7FFFFFFF, int(stream)])


def _write_env_info(out: str, env, lay: store.Layout):
    """the first worker to get here writes shards/env_info.json (ManiSkill env_info + layouts) for the dataset JSON"""
    p = os.path.join(out, "shards", "env_info.json")
    if os.path.exists(p):
        return
    u = env.unwrapped
    kw = dict(getattr(u.spec, "kwargs", {}) or {}) if getattr(u, "spec", None) is not None else {}
    info = dict(env_info=dict(env_id=u.spec.id if getattr(u, "spec", None) is not None else None,
                              env_kwargs=store.jsonable(kw),
                              max_episode_steps=getattr(u.spec, "max_episode_steps", None) if getattr(u, "spec", None) else None),
                layout=lay.to_json(), control_freq=float(u.control_freq), sim_freq=float(u.sim_freq),
                imsim_make=store.jsonable(getattr(u, "imsim", {})))
    tmp = p + f".{os.getpid()}"
    json.dump(store.jsonable(info), open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def _ep_meta(seed, reset_kwargs, control_mode, T, success_last, im: dict) -> dict:
    return dict(episode_seed=int(seed), reset_kwargs=reset_kwargs, control_mode=control_mode, elapsed_steps=int(T),
                success=bool(success_last), imsim=im)


def _rigid_worker(job):
    wid, task, emb, seeds, pspec, out, opts = job
    worker_setup(cuda=False)
    from ..api import make
    from ..tasks import _ref, get_task
    from ..verify import v31  # noqa: F401
    from .devmodel import DeviationModel
    spec = get_task(task)
    perturb = PerturbSpec.from_dict(pspec)
    rgb = bool(opts.get("rgb"))
    env = _make_rigid(spec, emb, rgb, opts.get("render"), opts.get("env_kwargs"), opts.get("rgb_size") or (128, 128))
    u = env.unwrapped
    rec = _ref(spec.recorder)(env)
    lay = store.Layout.from_env(env)
    dm = DeviationModel.load(perturb.devmodel) if (perturb.has_noise and perturb.cov == "devmodel") else None
    noise = NoiseProcess(perturb, lay.arm_dofs, dm)
    pert = PerturbingEnv(rec, lay, [noise], rgb=rgb_grabber(env, opts.get("rgb_size") or (128, 128)) if rgb else None,
                         record_states=not opts.get("no_states"))
    expert, expert_name, _ = expert_of(spec)
    contracts = [spec.load_contract(c) for c in opts["contracts"]]
    derives = {c.task_id: (_ref(spec.derive)(c) if spec.derive else None) for c in contracts}
    excl = tuple(spec.phys_exclude or ())
    writer = store.ShardWriter(out, wid, keep_rejected=bool(opts.get("keep_rejected")))
    _write_env_info(out, env, lay)
    do_cert = bool(opts.get("certify", True))
    n_done = 0
    for seed in seeds:
        t0 = time.time()
        im = dict(perturb=perturb.name, worker=wid, expert=expert_name, expert_params=expert_params_tag())
        arrays, T, succ_last = None, 0, False
        ropts = spec.reset_options(seed) if spec.reset_options else None
        reset_kwargs = dict(seed=int(seed), options=ropts)
        judged, log, err = {}, {}, None
        try:
            rec.reset_out = rec.reset(seed=int(seed), options=ropts)
            reset_phase(env)
            restore_physics(u, excl)
            draw = apply_physical_uncertainty(u, _rng(perturb, seed, 1), perturb.phys, excl) if perturb.phys else {}
            im["phys"] = draw
            obs0 = u.get_obs() if draw else rec.reset_out[0]
            pert.begin(obs0, [_rng(perturb, seed, 2)])
            log = expert(pert) or {}
            T = pert.n_steps
            arrays = pert.arrays(0)
            succ_last = bool(arrays["success"][-1]) if T else False
            im["noise"] = pert.noise_stats(0)
            if T == 0:
                raise RuntimeError("the expert made no env.step")
            if do_cert:
                judged = judge_rigid(spec, rec, contracts, derives)
            n_judge = len(rec.traces.get("phase", [])) - T
            im["steps"] = dict(policy=T, judged_total=len(rec.traces.get("phase", [])), settle=n_judge)
        except Exception as e:                                          # noqa: BLE001
            import traceback
            err = repr(e)
            im["error"] = traceback.format_exc()[-1500:]
        cert = certify_verdicts(judged, [c.task_id for c in contracts] if do_cert else [], log.get("status"), err)
        j0 = judged.get(contracts[0].task_id) or {}
        im.update(cert, split="data" if cert["certified"] else ("error" if err else "rejected"),
                  verdicts={cid: verdict_row(j) for cid, j in judged.items()},
                  bench=dict(final=(j0.get("bench") or {}).get("bench_success_final"),
                             any=(j0.get("bench") or {}).get("bench_success_any"),
                             after_settle=(j0.get("bench") or {}).get("bench_success_after_settle")),
                  plan={k: v for k, v in log.items() if k not in ("final_targets",)},
                  seconds=round(time.time() - t0, 2),
                  start=dict(how="reset(**reset_kwargs); physical draw (mass / friction); set_state_dict(env_states[0])"))
        writer.write(f"s{int(seed)}", arrays, _ep_meta(seed, reset_kwargs, lay.control_mode, T, succ_last, im))
        n_done += 1
        if opts.get("verbose", True):
            print(f"[w{wid}] {task} {emb} {perturb.name} seed {seed}: {im['split']} "
                  f"{im['rejection_cause'] or ''} T={T} {time.time() - t0:.1f}s", flush=True)
    writer.close()
    env.close()
    return n_done


# ================================================================================================ cloth worker
def _soft_worker(job):
    wid, task, emb, seeds, pspec, out, opts = job
    worker_setup(cuda=True)
    import torch
    from ..api import make
    from ..soft_eval import ClothTraceRecorder, _settle, judge_world
    from ..tasks import get_task
    from .devmodel import DeviationModel
    spec = get_task(task)
    perturb = PerturbSpec.from_dict(pspec)
    B = int(opts.get("num_envs") or 1)
    sim = opts.get("sim") or ("gpu" if B > 1 else "cpu")
    rgb = bool(opts.get("rgb"))
    env = make(spec.name, emb, num_envs=B, sim=sim, render="newton" if rgb else "none", obs="state",
               **(opts.get("env_kwargs") or {}))
    u = env.unwrapped
    lay = store.Layout.from_env(env)
    dm = DeviationModel.load(perturb.devmodel) if (perturb.has_noise and perturb.cov == "devmodel") else None
    noises = [NoiseProcess(perturb, lay.arm_dofs, dm) for _ in range(B)]
    expert, expert_name, pose_aware = expert_of(spec)
    C = spec.load_contract(opts["contracts"][0])
    extra = [spec.load_contract(c) for c in opts["contracts"][1:]]
    writer = store.ShardWriter(out, wid, keep_rejected=bool(opts.get("keep_rejected")))
    _write_env_info(out, env, lay)
    do_cert = bool(opts.get("certify", True))
    holder = {}

    def post(outp):
        holder["rec"].record(u.cloth_signals(), u.cloth.particle_q(), outp[-1]["success"], 0)

    pert = PerturbingEnv(env, lay, noises, rgb=rgb_grabber(env) if rgb else None, post_step=post,
                         record_states=not opts.get("no_states"))
    batches = [seeds[i:i + B] for i in range(0, len(seeds), B)]
    for bs in batches:
        t0 = time.time()
        ws = list(bs) + [bs[-1]] * (B - len(bs))          # pad the last batch (padded worlds are not written)
        err, judged_w, log = None, [None] * B, {}
        draws = [{} for _ in range(B)]
        pert.buf, pert.n_steps, T = None, 0, 0
        holder.pop("rec", None)
        try:
            obs, _ = env.reset(seed=[int(s) for s in ws] if B > 1 else int(ws[0]))
            reset_phase(env)
            draws = apply_cloth_uncertainty(u, [_rng(perturb, s, 1) for s in ws], perturb.phys, pose_aware) \
                if perturb.phys else [{} for _ in range(B)]
            if perturb.phys:
                obs = u.get_obs()
            rec = ClothTraceRecorder(B, u.control_freq)
            rec.set_init(u.cloth.particle_q())
            holder["rec"] = rec
            pert.begin(obs, [_rng(perturb, s, 2) for s in ws])
            log = expert(pert) or {}
            T = pert.n_steps
            if do_cert:
                _settle(env, rec, C)
            for w in range(B if do_cert else 0):
                tr = rec.traces(w)
                jw = {C.task_id: judge_world(C, tr, u.control_freq)}
                for c in extra:
                    jw[c.task_id] = judge_world(c, tr, u.control_freq)
                judged_w[w] = jw
        except Exception as e:                                          # noqa: BLE001
            import traceback
            err = repr(e)
            log = dict(status="error", error=traceback.format_exc()[-1500:])
            T = pert.n_steps
        dt = time.time() - t0
        for w, seed in enumerate(bs):
            im = dict(perturb=perturb.name, worker=wid, expert=expert_name, world=w, batch_worlds=B, sim=sim,
                      phys=draws[w] if w < len(draws) else {})
            arrays = pert.arrays(w) if (pert.buf is not None and T) else None
            j = judged_w[w] or {}
            cert = certify_verdicts(j, ([C.task_id] + [c.task_id for c in extra]) if do_cert else [], log.get("status"), err)
            j0 = j.get(C.task_id) or {}
            im.update(cert, split="data" if cert["certified"] else ("error" if err else "rejected"),
                      verdicts={cid: verdict_row(v) for cid, v in j.items()},
                      bench=dict(final=(j0.get("bench") or {}).get("bench_success_final"),
                                 any=(j0.get("bench") or {}).get("bench_success_any"),
                                 after_settle=(j0.get("bench") or {}).get("bench_success_after_settle")),
                      plan={k: v for k, v in log.items() if k in ("status", "steps", "error")},
                      noise=pert.noise_stats(w) if arrays is not None else {},
                      steps=dict(policy=T, judged_total=len(holder["rec"].rows) if "rec" in holder else None),
                      seconds=round(dt / len(bs), 2), batch_seconds=round(dt, 2),
                      start=dict(how="reset(seed); cloth pose draw (imitator.phys) applied to the flat towel; "
                                     "env_states[0] for the robot"))
            succ_last = bool(arrays["success"][-1]) if arrays is not None and T else False
            meta = _ep_meta(seed, dict(seed=int(seed)), lay.control_mode, T, succ_last, im)
            meta["_world"] = w
            writer.write(f"s{int(seed)}", arrays, meta)
        if opts.get("verbose", True):
            c = Counter((judged_w[w] or {}).get(C.task_id, {}).get("success") for w in range(len(bs)))
            print(f"[w{wid}] {task} {perturb.name} seeds {bs[0]}..{bs[-1]} ({B} worlds, {sim}): verdicts "
                  f"{dict(c)} T={T} {dt:.1f}s {err or ''}", flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    writer.close()
    env.close()
    return len(seeds)


# ================================================================================================ API
def generate(task: str, embodiment: str | None = None, seeds=range(10), perturb=None, certify: bool = True,
             out: str | None = None, workers: int = 1, contract: str | None = None, extra_contracts=None,
             rgb: bool = False, render: str | None = None, rgb_size=(128, 128), keep_rejected: bool = False,
             num_envs: int | None = None,
             sim: str | None = None, env_kwargs: dict | None = None, card: bool = True, verbose: bool = True,
             overwrite: bool = False, tag: str = "") -> dict:
    """Generate a contract-certified dataset (see module docstring). Returns a summary dict (out, attempted, certified,
    rate, causes, wall_s, episodes_per_hour, ...). certify=False skips the settle / aftermath / verdicts and keeps every
    episode whose expert reported status 'ok' (the dataset is then NOT contract-certified; dataset.json and the card
    say so)."""
    import multiprocessing as mp
    from ..tasks import get_task
    spec = get_task(task)
    emb = embodiment or spec.default_embodiment
    if emb not in spec.embodiments:
        raise ValueError(f"{task}: embodiment {emb!r} not supported (supported: {list(spec.embodiments)})")
    p = as_spec(perturb)
    if p.has_noise and p.cov == "devmodel" and not isinstance(p.devmodel, dict):
        from .devmodel import DeviationModel
        p = p.replace(devmodel=DeviationModel.load(p.devmodel).to_json() | {"path": os.path.abspath(p.devmodel)})
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds")
    out = out or os.path.join(DEFAULT_OUT, spec.name, emb, p.name + (f"_{tag}" if tag else ""))
    if os.path.exists(os.path.join(out, "trajectory.json")) and not overwrite:
        raise FileExistsError(f"{out} already holds a dataset (overwrite=True to replace it)")
    if os.path.isdir(os.path.join(out, "shards")):
        import shutil
        shutil.rmtree(os.path.join(out, "shards"))
    os.makedirs(out, exist_ok=True)
    c0 = contract or spec.default_contract
    extras = list(extra_contracts if extra_contracts is not None else spec.extra_contracts)
    cnames = [c0] + [c for c in extras if spec.contract_path(c) != spec.contract_path(c0)]
    opts = dict(out=out, contracts=cnames, rgb=rgb, render=render, rgb_size=tuple(rgb_size), keep_rejected=keep_rejected,
                certify=certify,
                env_kwargs=env_kwargs or {}, verbose=verbose, num_envs=num_envs, sim=sim)
    t0 = time.time()
    if spec.has_cloth:
        B = int(num_envs or min(len(seeds), 16))
        opts["num_envs"] = B
        W = max(1, min(int(workers), int(np.ceil(len(seeds) / B))))
        chunks = [seeds[w::W] for w in range(W)]
        fn = _soft_worker
    else:
        W = max(1, min(int(workers), len(seeds)))
        chunks = [seeds[w::W] for w in range(W)]
        fn = _rigid_worker
    jobs = [(w, spec.name, emb, chunks[w], p.to_dict() | {"devmodel": p.devmodel}, out, opts) for w in range(W)]
    if W == 1 and not spec.has_cloth and os.environ.get("IMSIM_INPROCESS") == "1":
        fn(jobs[0])                                    # debugging: run in this process
    else:
        with pool_env(cuda=spec.has_cloth), mp.get_context("spawn").Pool(W) as pool:
            pool.map(fn, jobs, chunksize=1)
    wall = time.time() - t0
    meta = dataset_meta(spec, emb, p, cnames, seeds, workers=W, wall=wall, opts=opts, certify=certify)
    m = store.merge(out, meta, keep_rejected=opts["keep_rejected"])
    rows = m["rows"]
    att = len(rows)
    cert = sum(1 for r in rows if r["imsim"]["certified"])
    causes = Counter(r["imsim"]["rejection_cause"] for r in rows if not r["imsim"]["certified"])
    summary = dict(task=spec.name, embodiment=emb, perturb=p.name, out=out, attempted=att, certified=cert,
                   rate=cert / max(att, 1), errors=sum(1 for r in rows if r["imsim"]["split"] == "error"),
                   causes=dict(causes.most_common()), wall_s=round(wall, 1), workers=W,
                   episodes_per_hour=round(att / max(wall, 1e-9) * 3600, 1),
                   certified_per_hour=round(cert / max(wall, 1e-9) * 3600, 1),
                   cpu_s_per_episode=round(float(np.mean([r["imsim"].get("seconds") or 0 for r in rows])), 2) if rows else None)
    meta.update(summary=summary)
    js = json.load(open(os.path.join(out, "dataset.json")))
    js.update(summary=summary)
    json.dump(store.jsonable(js), open(os.path.join(out, "dataset.json"), "w"), indent=1)
    tj = json.load(open(os.path.join(out, "trajectory.json")))
    tj["imsim"]["summary"] = summary
    json.dump(store.jsonable(tj), open(os.path.join(out, "trajectory.json"), "w"), indent=1)
    if card:
        from .card import write_card
        summary["card"] = write_card(out)["md"]
    if verbose:
        print(json.dumps({k: v for k, v in summary.items() if k != "out"}), flush=True)
    return summary


def dataset_meta(spec, emb, p: PerturbSpec, cnames, seeds, workers, wall, opts, certify=True) -> dict:
    """dataset-level metadata (env_info for ManiSkill replays, layouts, contracts, code hash)"""
    import mani_skill
    lay_json, env_info, extra = None, dict(env_id=spec.env_id, env_kwargs={}, max_episode_steps=None), {}
    p_info = os.path.join(opts["out"], "shards", "env_info.json")
    if os.path.exists(p_info):
        ei = json.load(open(p_info))
        env_info, lay_json = ei["env_info"], ei["layout"]
        extra = {k: ei.get(k) for k in ("control_freq", "sim_freq", "imsim_make")}
    dev = None
    if isinstance(p.devmodel, dict):
        dev = dict(path=p.devmodel.get("path"), scale_used=p.scale if p.scale is not None else p.devmodel.get("scale"),
                   calibration=p.devmodel.get("calibration"), task=p.devmodel.get("task"),
                   embodiment=p.devmodel.get("embodiment"), kind=p.devmodel.get("kind"))
    return dict(env_info=env_info, commit_info=dict(mani_skill=mani_skill.__version__),
                source_type="motionplanning", source_desc=f"imsim scripted expert ({spec.name}); {p.describe()}",
                task=spec.name, embodiment=emb, generator=GENERATOR, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                perturb=p.to_dict(), perturb_text=p.describe(), devmodel=dev, certify=certify,
                contracts={json.load(open(spec.contract_path(c)))["task_id"]: dict(name=c, sha256=file_sha256(spec.contract_path(c)))
                           for c in cnames},
                keep_rule="every contract accepts and the expert status is 'ok'" if certify else "none (certify=False)",
                code_hash=code_hash(), layout=lay_json, **extra, seeds=[min(seeds), max(seeds), len(seeds)], workers=workers,
                wall_s=round(wall, 1), rgb=bool(opts.get("rgb")), num_envs=opts.get("num_envs"),
                python=sys.version.split()[0])
