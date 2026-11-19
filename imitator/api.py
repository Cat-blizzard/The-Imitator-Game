"""imitator.make(): one entry point for every imsim task and embodiment (ManiSkill 3 gym envs).

    env = imitator.make("DualHandover-v1", embodiment="dual_ur5_wsg", num_envs=1, sim="cpu", render=None, obs="state")

sim     'cpu'  PhysX CPU (ManiSkill: one env per process); physics 100 Hz
        'gpu'  PhysX GPU (num_envs >= 1); physics rate per embodiment from E19's stress test (DualConfig.gpu_sim_freq:
               ARX X5 200 Hz, PiPER 500 Hz, others 100 Hz); HOME is pointed at the project copy of the PhysX GPU
               library first (imitator.compat.ensure_sapien_home) so SAPIEN never writes to ~/.sapien
render  None   automatic: 'newton' when the task has cloth (ManiSkill cameras cannot see the Newton towel) or when no
               Vulkan device is visible at all; with GPU physics 'sapien_gpu' if a CUDA-visible Vulkan device exists, else
               'newton'; with CPU physics 'sapien_gpu' if a CUDA-visible Vulkan device exists, else 'sapien_cpu' (SAPIEN
               on lavapipe, device 'pci:0')
        'none' | 'sapien_cpu' | 'sapien_gpu' | 'newton'   force a backend
obs     'state' (flat state vector) | 'rgb' (state + camera image: ManiSkill sensor 'front_camera' for SAPIEN backends;
        obs['sensor_data']['newton_camera']['rgb'] for the Newton camera) | any ManiSkill obs_mode string
The returned env is the ManiSkill env (wrapped by gymnasium's TimeLimit / order enforcing as gym.make does);
env.unwrapped.imsim = dict(task, embodiment, sim, render, obs, ...).
embodiment_calibration=None (default) | 'f1' | path.json: opt-in calibrated actuators (imitator/embodiments/calibration).
tactile=None (default) | 'T0' | 'T1' | 'T2': opt-in tactile pads (imitator/sensors/tactile.py) -> obs['tactile'] (a flat state
        obs becomes {'state': obs, 'tactile': ...}); tactile_kw=dict(layout='12x32' | '16x16', ...) -> TactileSensor.
"""
from __future__ import annotations

import functools

from . import compat
from .tasks import get_task

RENDERS = ("none", "sapien_cpu", "sapien_gpu", "newton")


@functools.lru_cache(maxsize=1)
def vulkan_info() -> dict:
    """{'any': a Vulkan device SAPIEN can render with (lavapipe counts), 'cuda': a CUDA-visible Vulkan device}"""
    out = dict(any=False, cuda=False, summary="")
    try:
        import sapien
        try:
            out["summary"] = sapien.render.get_device_summary()
        except Exception:
            pass
        for name, key in (("cuda", "cuda"), ("pci:0", "any")):
            try:
                if sapien.Device(name).can_render():
                    out[key] = True
            except Exception:
                pass
        out["any"] = out["any"] or out["cuda"]
    except Exception as e:  # pragma: no cover
        out["error"] = repr(e)
    return out


def choose_render(has_cloth: bool, sim: str, render: str | None) -> str:
    if render is not None:
        if render not in RENDERS:
            raise ValueError(f"render must be None or one of {RENDERS}")
        return render
    vk = vulkan_info()
    if has_cloth:
        return "newton"
    if sim == "gpu":
        return "sapien_gpu" if vk["cuda"] else "newton"
    if vk["cuda"]:
        return "sapien_gpu"
    return "sapien_cpu" if vk["any"] else "newton"


def list_embodiments(detail: bool = False):
    from .embodiments import dual as D
    from .embodiments import registry as reg
    names = sorted(D.CONFIGS) + ["panda_cloth"]
    if not detail:
        return names
    out = {}
    for n, c in D.CONFIGS.items():
        l, r = c.records
        out[n] = dict(label=c.label, left=l.name, right=r.name, kind=l.kind, dof=[l.action_dim, r.action_dim],
                      separation_m=c.separation, gpu_sim_freq=c.gpu_sim_freq, license=l.license, source=l.source)
    out["panda_cloth"] = dict(label="Franka Panda (single arm, E20 cloth agent)", kind="arm_gripper", gpu_sim_freq=100)
    return out


def make(task: str, embodiment: str | None = None, num_envs: int = 1, sim: str = "cpu", render: str | None = None,
         obs: str = "state", control_mode: str = "pd_joint_pos", **kw):
    import importlib

    import gymnasium as gym
    spec = get_task(task)
    emb = embodiment or spec.default_embodiment
    if emb not in spec.embodiments:
        raise ValueError(f"{task}: embodiment {emb!r} not supported (supported: {list(spec.embodiments)})")
    if sim not in ("cpu", "gpu"):
        raise ValueError("sim must be 'cpu' or 'gpu'")
    if sim == "cpu" and num_envs != 1:
        raise ValueError("ManiSkill's CPU physics runs one env per process; use sim='gpu' for num_envs > 1")
    backend = choose_render(spec.has_cloth, sim, render)
    compat.patch_render_backend()
    importlib.import_module(spec.module)
    from .embodiments import agents  # noqa: F401  (registers the imsim agents)
    if sim == "gpu" or spec.gpu_only_cloth:
        compat.ensure_sapien_home()
    sim_backend = "physx_cpu" if sim == "cpu" else "physx_cuda"
    emb_calibration = kw.pop("embodiment_calibration", None)  # F1 opt-in (embodiments/calibration); off by default
    tactile = kw.pop("tactile", None)                          # E1 opt-in (sensors/tactile); off by default
    tactile_kw = kw.pop("tactile_kw", None) or {}
    if emb_calibration and sim != "cpu":
        raise ValueError("embodiment_calibration needs sim='cpu' (PhysX GPU drive gains are fixed at GPU init)")
    args = dict(spec.env_defaults)
    args.update(kw)
    if spec.name == "FoldTowel-v1":
        args.setdefault("robot_uids", "panda_cloth")
    else:
        args["dual_config"] = emb
        if sim == "gpu" and "sim_config" not in args:
            from .embodiments import dual as D
            f = D.CONFIGS[emb].gpu_sim_freq
            if f != 100:
                args["sim_config"] = dict(sim_freq=f)
    obs_mode = {"state": "state", "rgb": "rgb"}.get(obs, obs)
    newton_cam = backend == "newton"
    if spec.has_cloth:
        if newton_cam:
            args.setdefault("rgb_source", "newton")
        obs_mode = "state_dict" if obs == "rgb" else obs_mode
        render_backend = "none" if backend in ("none", "newton") else ("pci:0" if backend == "sapien_cpu" else "cuda")
    else:
        if newton_cam and obs == "rgb":
            obs_mode = "state_dict"
        render_backend = {"none": "none", "newton": "none", "sapien_cpu": "pci:0", "sapien_gpu": "cuda"}[backend]
    render_mode = None if backend == "none" else "rgb_array"
    env = gym.make(spec.env_id, num_envs=num_envs, sim_backend=sim_backend, render_backend=render_backend,
                   render_mode=render_mode, obs_mode=obs_mode, control_mode=control_mode, **args)
    if newton_cam and not spec.has_cloth:
        from .render.newton_mirror import NewtonRigidCameraWrapper
        env = NewtonRigidCameraWrapper(env, add_to_obs=(obs == "rgb"))
    env.unwrapped.imsim = dict(task=spec.name, embodiment=emb, sim=sim, render=backend, obs=obs, num_envs=num_envs,
                               sim_backend=sim_backend, render_backend=render_backend, obs_mode=obs_mode)
    if emb_calibration:  # F1: calibrated actuator overrides (drive gains + action delay / low-pass), CPU sim only
        from .embodiments import calibration as _calibration
        env = _calibration.wrap(env, emb_calibration)
        env.unwrapped.imsim["embodiment_calibration"] = emb_calibration
    if tactile:  # E1: tactile pads on the embodiment's fingers (default pads: imitator/sensors/pads.py)
        from .sensors.tactile import TactileWrapper
        env = TactileWrapper(env, level=tactile, **tactile_kw)
        env.unwrapped.imsim["tactile"] = dict(level=env.level, pads=None, **{k: v for k, v in tactile_kw.items()
                                                                             if isinstance(v, (str, int, float, bool))})
    return env


# ================================================================================================ data engine (E29a)
def generate(task: str, embodiment: str | None = None, seeds=range(10), perturb=None, certify: bool = True,
             out: str | None = None, workers: int = 1, **kw) -> dict:
    """Contract-certified dataset generation (imitator.data.engine.generate): perturb = PerturbSpec | preset name
    ('mp', 'dart', 'dart_dir', 'phys', 'dart_hi_phys', 'dart_dir_nophys', 'dart_dir_white', 'dart_dir_freeonly') | dict.
    Returns the summary (out, attempted, certified, rate, causes, episodes_per_hour, card)."""
    from .data.engine import generate as _g
    return _g(task, embodiment, seeds=seeds, perturb=perturb, certify=certify, out=out, workers=workers, **kw)


def certify(dataset: str, task: str, contract: str | None = None, **kw) -> dict:
    """Replay a ManiSkill3 trajectory dataset (stored seed + initial state) under the task's recorder, judge every
    episode with the contract(s) and report diverging replays (imitator.data.certify.certify)."""
    from .data.certify import certify as _c
    return _c(dataset, task, contract=contract, **kw)


def fit_deviation_model(task: str, embodiment: str | None = None, rollouts=None, policy=None, **kw):
    """E26 deviation model from learner rollouts (or a policy rolled out from an expert dataset's start states)"""
    from .data.devmodel import fit_deviation_model as _f
    return _f(task, embodiment, rollouts=rollouts, policy=policy, **kw)


def calibrate_scale(task: str, embodiment: str | None = None, devmodel=None, **kw) -> dict:
    """E26 s_t calibration: largest of 1/0.5/0.25/0.125/0.0625 whose 12-seed dart_dir pilot certifies >= 50 %"""
    from .data.devmodel import calibrate_scale as _c
    return _c(task, embodiment, devmodel=devmodel, **kw)


def dataset_card(dataset_dir: str) -> dict:
    from .data.card import write_card
    return write_card(dataset_dir)
