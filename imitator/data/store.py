"""Dataset storage of the imsim data engine: ManiSkill3 trajectory HDF5 + JSON, plus per-episode imsim metadata.

<out>/
  trajectory.h5         certified episodes, ManiSkill3 layout: group traj_<episode_id> with
                          actions        (T, A) float32  EXECUTED actions (what env.step received; replay uses these),
                                                         flat: the agents' actions concatenated in action-space key
                                                         order (mani_skill.utils.wrappers.FlattenActionSpaceWrapper)
                          terminated / truncated / success (T,) bool, rewards (T,) float32
                          obs            (T+1, D) float32 flat state observation (obs_mode 'state'; obs[0] after reset)
                          env_states     {actors: {name: (T+1, 13)}, articulations: {name: (T+1, 13 + 2 dof)}}
                        imsim extensions in the same group:
                          actions_clean  (T, A) float32  LABELS: the clean expert actions (noise-free targets)
                          phase          (T, n_arms) uint8  0 = FREE, 1 = CONTACT (imitator.data.phase, per step and arm)
                          arm_qpos       (T+1, sum arm dof) float32  arm joint positions (deviation models)
                          rgb            (T+1, H, W, 3) uint8 (optional, gzip; render path of imitator.make)
  trajectory.json       ManiSkill3 JSON: env_info {env_id, env_kwargs, max_episode_steps}, commit_info, source_type,
                        source_desc, episodes [{episode_id, episode_seed, reset_kwargs, control_mode, elapsed_steps,
                        success (benchmark flag at the last step, ManiSkill semantics), imsim: {...}}], imsim {dataset}
  rejected.h5           optional (keep_rejected=True): rejected episodes, same layout, keys rej_<n>
  attempts.jsonl        every attempted episode (certified, rejected, error) with its imsim metadata
  dataset.json          dataset metadata (task, embodiment, layouts, perturbation spec, contracts + sha256, code hash,
                        counts, throughput)
  card.md / card.json   dataset card (imitator.data.card)
The imsim episode block holds: certified, split, rejection_cause, causes, verdicts {contract: success, released,
binding, margin, margins, window_status, protocol_reject}, bench {final, any, after_settle}, phys (the physical
uncertainty draw), plan (expert log), noise statistics, steps, seconds, and `start` (how to reproduce the start state).
Replay: env.reset(**reset_kwargs), then env.set_state_dict(env_states[0]) and the physical draw's mass / friction
(store.reset_to_episode does both), then step `actions`.
"""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict

import numpy as np

FORMAT = "imsim-traj-v1"
LABEL_KEY = "actions_clean"
PHASE_CODES = {"free": 0, "contact": 1}


# ================================================================================================ layouts
def _agents(u):
    return list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent]


def flat_obs(obs) -> np.ndarray:
    """flat state vector (D,) of env 0 from a ManiSkill obs (tensor / array / dict with 'state')"""
    if isinstance(obs, dict):
        obs = obs.get("state", obs)
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    a = np.asarray(obs, np.float32)
    return a[0] if a.ndim == 2 else a


class Layout:
    """action / observation layout of an env (one world): action keys and dims, arm and gripper dims per agent, arm
    qpos indices, obs key slices"""

    def __init__(self, keys, dims, arm_slices, arm_qidx, obs_slices=None, n_obs=None, control_mode=None):
        self.keys = list(keys)                         # action-space keys (None for a single agent with a Box space)
        self.dims = [int(d) for d in dims]
        self.arm_slices = [tuple(int(v) for v in s) for s in arm_slices]   # (start, end) within each agent's action
        self.arm_qidx = [list(map(int, q)) for q in arm_qidx]              # arm joint indices in each robot's qpos
        self.obs_slices = dict(obs_slices or {})
        self.n_obs = n_obs
        self.control_mode = control_mode

    @property
    def n_arms(self) -> int:
        return len(self.dims)

    @property
    def arm_dofs(self) -> list[int]:
        return [e - s for s, e in self.arm_slices]

    @property
    def act_dim(self) -> int:
        return int(sum(self.dims))

    @property
    def offsets(self) -> list[int]:
        return list(np.r_[0, np.cumsum(self.dims)][:-1].astype(int))

    def arm_dims_flat(self) -> list[int]:
        """flat action indices of all arm joint targets"""
        return [o + j for o, (s, e) in zip(self.offsets, self.arm_slices) for j in range(s, e)]

    def gripper_dims_flat(self) -> list[int]:
        arm = set(self.arm_dims_flat())
        return [i for i in range(self.act_dim) if i not in arm]

    @classmethod
    def from_env(cls, env) -> "Layout":
        u = env.unwrapped
        sp = getattr(u, "single_action_space", u.action_space)
        ags = _agents(u)
        if hasattr(sp, "spaces"):
            keys = list(sp.spaces.keys())
            dims = [int(np.prod(sp.spaces[k].shape)) for k in keys]
        else:
            keys, dims = [None], [int(np.prod(sp.shape))]
        arm_slices, arm_qidx = [], []
        for ag, d in zip(ags, dims):
            cc = ag.controller
            amap = getattr(cc, "action_mapping", None) or {}
            if "arm" in amap:
                arm_slices.append(tuple(amap["arm"]))
            else:
                arm_slices.append((0, int(ag.record.dof_arm) if hasattr(ag, "record") else d))
            ctrl = getattr(cc, "controllers", {}).get("arm") if hasattr(cc, "controllers") else None
            if ctrl is not None:
                arm_qidx.append([int(i) for i in np.asarray(ctrl.active_joint_indices.cpu() if hasattr(
                    ctrl.active_joint_indices, "cpu") else ctrl.active_joint_indices).reshape(-1)])
            else:
                names = list(getattr(ag.record, "arm_joints", []))
                arm_qidx.append([i for i, j in enumerate(ag.robot.active_joints) if j.name in names])
        obs_slices, n_obs = obs_layout(u)
        return cls(keys, dims, arm_slices, arm_qidx, obs_slices, n_obs, getattr(u, "control_mode", None))

    def to_json(self) -> dict:
        return dict(keys=self.keys, dims=self.dims, arm_slices=self.arm_slices, arm_qidx=self.arm_qidx,
                    arm_dims_flat=self.arm_dims_flat(), gripper_dims_flat=self.gripper_dims_flat(),
                    obs_slices={k: list(v) for k, v in self.obs_slices.items()}, n_obs=self.n_obs,
                    control_mode=self.control_mode)

    @classmethod
    def from_json(cls, d: dict) -> "Layout":
        return cls(d["keys"], d["dims"], d["arm_slices"], d["arm_qidx"], {k: tuple(v) for k, v in d.get("obs_slices", {}).items()},
                   d.get("n_obs"), d.get("control_mode"))

    # ------------------------------------------------------------------ conversions
    def flatten_action(self, a) -> np.ndarray:
        """env action (dict / tensor / array, batch 1 or none) -> flat (A,) float32 of env 0"""
        def one(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            x = np.asarray(x, np.float32)
            return x[0] if x.ndim == 2 else x
        if isinstance(a, dict):
            return np.concatenate([one(a[k]).reshape(-1) for k in self.keys]).astype(np.float32)
        return one(a).reshape(-1).astype(np.float32)

    def flatten_batch(self, a) -> np.ndarray:
        """batched env action -> (N, A) float32"""
        def one(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            x = np.asarray(x, np.float32)
            return x[None] if x.ndim == 1 else x
        if isinstance(a, dict):
            return np.concatenate([one(a[k]) for k in self.keys], 1)
        return one(a)

    def unflatten_action(self, flat):
        flat = np.asarray(flat, np.float32).reshape(-1)
        if self.keys == [None]:
            return flat
        out, o = OrderedDict(), 0
        for k, d in zip(self.keys, self.dims):
            out[k] = flat[o:o + d].copy(); o += d
        return out

    def arm_qpos(self, u, env_idx: int = 0) -> np.ndarray:
        q = []
        for ag, idx in zip(_agents(u), self.arm_qidx):
            qq = ag.robot.get_qpos()[env_idx].detach().cpu().numpy()
            q.append(qq[idx])
        return np.concatenate(q).astype(np.float32)

    def arm_qpos_batch(self, u) -> np.ndarray:
        return np.concatenate([ag.robot.get_qpos().detach().cpu().numpy()[:, idx]
                               for ag, idx in zip(_agents(u), self.arm_qidx)], 1).astype(np.float32)


def obs_layout(u) -> tuple[dict, int | None]:
    """key path -> (start, end) of the flat 'state' observation (same order as mani_skill common.flatten_state_dict)"""
    try:
        info = u.get_info()
        d = u._get_obs_state_dict(info)
    except Exception:                                                   # noqa: BLE001
        return {}, None
    out, pos = {}, [0]

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}/{k}" if path else str(k))
            return
        n = 1
        if hasattr(x, "shape"):
            sh = tuple(x.shape)
            n = int(np.prod(sh[1:])) if len(sh) >= 2 else 1
        out[path] = (pos[0], pos[0] + n)
        pos[0] += n
    walk(d, "")
    return out, pos[0]


# ================================================================================================ env state helpers
def state_dict_np(u, env_idx=None) -> dict:
    """env.get_state_dict() as numpy (env 0, or all envs when env_idx is 'all')"""
    sd = u.get_state_dict()

    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        a = a.astype(np.float32)
        if env_idx == "all":
            return a
        return a[0] if a.ndim >= 2 else a
    return conv(sd)


def stack_dicts(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {}
    for k, v in rows[0].items():
        if isinstance(v, dict):
            out[k] = stack_dicts([r[k] for r in rows])
        else:
            out[k] = np.stack([r[k] for r in rows]).astype(np.float32)
    return out


def index_tree(x, i):
    if isinstance(x, dict):
        return {k: index_tree(v, i) for k, v in x.items()}
    return x[i]


def _to_torch_tree(x, device):
    import torch
    if isinstance(x, dict):
        return {k: _to_torch_tree(v, device) for k, v in x.items()}
    t = torch.as_tensor(np.asarray(x, np.float32), device=device)
    return t[None] if t.ndim == 1 else t


def reset_to_episode(env, spec, ep: dict, env_states=None, apply_phys: bool = True):
    """reset `env` (CPU, 1 env) to an episode's start: reset(**reset_kwargs), the physical draw's mass / friction
    scaling (imsim datasets), then set_state_dict(env_states[0]) when stored. Returns the flat obs."""
    from .perturb import restore_physics
    u = env.unwrapped
    rk = dict(ep.get("reset_kwargs") or {})
    seed = rk.pop("seed", ep.get("episode_seed"))
    if isinstance(seed, list):
        seed = seed[0]
    opts = rk.pop("options", None)
    obs, _ = env.reset(seed=None if seed is None else int(seed), options=opts)
    im = ep.get("imsim") or {}
    excl = tuple(getattr(spec, "phys_exclude", ()) or ())
    restore_physics(u, excl)
    if apply_phys and im.get("phys"):
        apply_draw(u, im["phys"], excl)
    if env_states is not None:
        s0 = index_tree(env_states, 0)
        u.set_state_dict(_to_torch_tree(s0, u.device))
        obs = u.get_obs()
    return flat_obs(obs)


def apply_draw(u, draw: dict, exclude=()) -> None:
    """re-apply the mass / friction scaling of a stored physical-uncertainty draw (poses come from env_states[0])"""
    import sapien
    from .perturb import _orig, movable_actors
    if getattr(u, "gpu_sim_enabled", False):
        return
    acts = dict(movable_actors(u, exclude))
    for name, d in (draw or {}).items():
        a = acts.get(name)
        if a is None or not isinstance(d, dict):
            continue
        body, (m0, mats) = _orig(u, name, a)
        if d.get("mass_scale") is not None:
            body.mass = m0 * float(d["mass_scale"])
        if d.get("friction_scale") is not None:
            ff = float(d["friction_scale"])
            for sh, (sf, df, r) in zip(body.get_collision_shapes(), mats):
                sh.set_physical_material(sapien.physx.PhysxMaterial(sf * ff, df * ff, r))


# ================================================================================================ writing
def _put(g, key: str, data, compress: bool = False):
    if isinstance(data, dict):
        sub = g.create_group(key, track_order=True)
        for k, v in data.items():
            _put(sub, k, v, compress or key == "rgb")
        return
    a = np.asarray(data)
    if key in ("rgb", "depth", "seg") or (compress and a.dtype == np.uint8):
        g.create_dataset(key, data=a, dtype=a.dtype, compression="gzip", compression_opts=5)
    elif a.ndim >= 1 and a.size > 4096:
        g.create_dataset(key, data=a, dtype=a.dtype, compression="gzip", compression_opts=3)
    else:
        g.create_dataset(key, data=a, dtype=a.dtype)


class ShardWriter:
    """one per worker: shard_<w>.h5 (group per attempted episode that produced arrays) + shard_<w>.jsonl (metadata)"""

    def __init__(self, root: str, wid: int, keep_rejected: bool = False):
        import h5py
        self.dir = os.path.join(root, "shards")
        os.makedirs(self.dir, exist_ok=True)
        self.h5_path = os.path.join(self.dir, f"shard_{wid:03d}.h5")
        self.jl_path = os.path.join(self.dir, f"shard_{wid:03d}.jsonl")
        self.h5 = h5py.File(self.h5_path, "a")
        self.keep_rejected = keep_rejected

    def write(self, key: str, arrays: dict | None, meta: dict):
        keep_arrays = arrays is not None and (meta["imsim"]["certified"] or self.keep_rejected)
        if keep_arrays:
            if key in self.h5:
                del self.h5[key]
            g = self.h5.create_group(key, track_order=True)
            for k, v in arrays.items():
                _put(g, k, v)
            self.h5.flush()
        meta = dict(meta, _shard=os.path.basename(self.h5_path), _key=key if keep_arrays else None)
        with open(self.jl_path, "a") as f:
            f.write(json.dumps(jsonable(meta)) + "\n")

    def close(self):
        self.h5.close()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "detach"):
        return jsonable(x.detach().cpu().numpy())
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else (1e30 if x > 0 else -1e30)
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    return repr(x)


def merge(root: str, dataset_meta: dict, keep_rejected: bool = False, remove_shards: bool = True) -> dict:
    """shards -> trajectory.h5 / trajectory.json (certified, seed order, traj_0..), rejected.h5, attempts.jsonl"""
    import h5py
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "shards", "shard_*.jsonl"))):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    rows.sort(key=lambda r: (int(r.get("episode_seed", 0)), r.get("_world", 0)))
    srcs = {}
    traj_path = os.path.join(root, "trajectory.h5")
    rej_path = os.path.join(root, "rejected.h5")
    for p in (traj_path, rej_path):
        if os.path.exists(p):
            os.remove(p)
    out = h5py.File(traj_path, "w")
    rej = h5py.File(rej_path, "w") if keep_rejected else None
    episodes, k, kr = [], 0, 0
    for r in rows:
        im = r["imsim"]
        if r.get("_key") is None:
            continue
        shard = os.path.join(root, "shards", r["_shard"])
        if shard not in srcs:
            srcs[shard] = h5py.File(shard, "r")
        if im["certified"]:
            r["episode_id"] = k
            srcs[shard].copy(srcs[shard][r["_key"]], out, name=f"traj_{k}")
            im["traj"] = f"traj_{k}"
            k += 1
        elif rej is not None:
            srcs[shard].copy(srcs[shard][r["_key"]], rej, name=f"rej_{kr}")
            im["traj"] = f"rejected.h5:rej_{kr}"
            kr += 1
    for f in srcs.values():
        f.close()
    out.close()
    if rej is not None:
        rej.close()
    for r in rows:
        r.pop("_shard", None); r.pop("_key", None)
        if r["imsim"].get("certified"):
            episodes.append({k2: v for k2, v in r.items() if not k2.startswith("_")})
    with open(os.path.join(root, "attempts.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(jsonable(r)) + "\n")
    ms = dict(dataset_meta)
    tj = dict(env_info=ms.pop("env_info"), commit_info=ms.pop("commit_info", {}),
              source_type=ms.get("source_type", "motionplanning"), source_desc=ms.get("source_desc", ""),
              episodes=episodes, imsim=dict(ms, format=FORMAT, label_key=LABEL_KEY, phase_codes=PHASE_CODES))
    with open(os.path.join(root, "trajectory.json"), "w") as f:
        json.dump(jsonable(tj), f, indent=1)
    with open(os.path.join(root, "dataset.json"), "w") as f:
        json.dump(jsonable(dict(tj["imsim"], env_info=tj["env_info"], n_certified=len(episodes))), f, indent=1)
    if remove_shards:
        for p in glob.glob(os.path.join(root, "shards", "*")):
            os.remove(p)
        try:
            os.rmdir(os.path.join(root, "shards"))
        except OSError:
            pass
    return dict(rows=rows, episodes=episodes)


# ================================================================================================ reading
class Dataset:
    """a ManiSkill3 trajectory dataset (ours or external): .episodes (JSON rows, '_index' = position), .meta (imsim
    block or {}), .arrays(i) -> dict of numpy arrays of traj_<episode_id>"""

    def __init__(self, path: str):
        if os.path.isdir(path):
            h5 = os.path.join(path, "trajectory.h5")
        else:
            h5 = path
        js = os.path.splitext(h5)[0] + ".json"
        if not os.path.exists(h5):
            raise FileNotFoundError(h5)
        self.root = os.path.dirname(os.path.abspath(h5))
        self.h5_path, self.json_path = h5, js
        self.json = json.load(open(js)) if os.path.exists(js) else dict(episodes=[], env_info={})
        self.meta = self.json.get("imsim") or {}
        self.env_info = self.json.get("env_info") or {}
        eps = self.json.get("episodes") or []
        if not eps:                                            # an h5 without JSON: one row per traj group
            import h5py
            with h5py.File(h5, "r") as f:
                eps = [dict(episode_id=int(k.split("_")[1]), episode_seed=None, reset_kwargs={}) for k in f.keys()
                       if k.startswith("traj_")]
        self.episodes = [dict(e, _index=i) for i, e in enumerate(eps)]
        self._f = None

    def _file(self):
        import h5py
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def group_name(self, i: int) -> str:
        return f"traj_{self.episodes[i]['episode_id']}"

    def arrays(self, i: int, keys=None) -> dict:
        g = self._file()[self.group_name(i)]

        def read(x):
            import h5py
            if isinstance(x, h5py.Group):
                return {k: read(v) for k, v in x.items()}
            return x[()]
        return {k: read(g[k]) for k in g.keys() if keys is None or k in keys}

    def layout(self) -> Layout | None:
        lay = self.meta.get("layout")
        return Layout.from_json(lay) if lay else None

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


def open_dataset(path: str) -> Dataset:
    return Dataset(path)


def load_episodes(path: str, labels: bool = True, obs_keys=None) -> list[dict]:
    """[{obs (T, D), act (T, A), phase (T, n_arms), meta}] with obs[t] the observation before action t. act = the
    clean labels (labels=True, default) or the executed actions."""
    ds = open_dataset(path)
    out = []
    for e in ds.episodes:
        a = ds.arrays(e["_index"], keys=("obs", "actions", LABEL_KEY, "phase"))
        act = a.get(LABEL_KEY) if labels and LABEL_KEY in a else a["actions"]
        obs = np.asarray(a["obs"], np.float32)
        out.append(dict(obs=obs[:-1], act=np.asarray(act, np.float32), phase=a.get("phase"), meta=e))
    ds.close()
    return out
