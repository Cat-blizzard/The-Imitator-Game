"""FoldTowel-v1: imsim's ManiSkill-compatible diagonal towel fold with a Newton (VBD) towel. Route A of E20
(research/results/e20_soft/E20_REPORT.md), the route recommended for the platform:

  robot     ManiSkill 3 / SAPIEN Panda ('panda_cloth': stock Panda whose gripper close target stops at the cloth layer),
            PhysX on the CPU or the GPU, every ManiSkill controller (pd_joint_pos, pd_ee_pose, ...) and wrapper works.
  towel     Newton 1.7 VBD, 25x25 vertices, one per parallel env (imitator.soft.cloth.NewtonClothWorlds); the hand and
            finger links are copied into Newton every control step as kinematic colliders (one-way coupling; the
            neglected cloth->robot reaction is measured: info['robot_cloth_impulse'], info['robot_cloth_fmax']).
  success   info['success'] = the towel_fold_diagonal terms on the current step (corner distance < 8 cm, footprint < 0.70,
            <= 2 % of the towel above 5 cm, released). The contract verdict (protocol v3, evaluator-run hold-A settle)
            is computed by run_contract_episode() below / imitator.soft.contract_eval from per-step traces.
  rendering GPUs without graphics capability (A100/H100 class, or containers without NVIDIA graphics): ManiSkill's GPU
            renderer needs a CUDA-visible Vulkan device, so use rgb_source='newton' (Warp ray-cast camera on the
            CUDA device, robot visual meshes mirrored into the Newton model; obs['sensor_data']['newton_camera']['rgb']
            with obs_mode='state_dict', and env.render()). With physx_cpu + a Vulkan device (lavapipe 'pci:0' works),
            SAPIEN cameras render the robot but not the towel; imitator.soft.render_sync.SapienMirror shows the towel
            mesh in SAPIEN renders (CPU).

    import gymnasium as gym, imitator.envs.fold_towel
    env = gym.make("FoldTowel-v1", num_envs=16, obs_mode="state_dict", control_mode="pd_joint_pos",
                   sim_backend="physx_cuda", render_backend="none", rgb_source="newton")

The all-Newton alternative (route B: Panda in MuJoCo-Warp) is imitator.soft.route_b.FoldTowelNewtonEnv; make_env(route="B")
returns it with the same step/reset/info conventions.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mani_skill.utils.registration import register_env  # noqa: E402

from imitator.soft.cloth import ClothSolverSpec, TowelSpec  # noqa: E402,F401
from imitator.soft.route_a import FoldTowelBridgeEnv, scripted_joint_action  # noqa: E402


@register_env("FoldTowel-v1", max_episode_steps=200)
class FoldTowelEnv(FoldTowelBridgeEnv):
    """Diagonal towel fold (Imitator _049_fold_towel, physical version), single Panda, Newton VBD towel."""

    def __init__(self, *args, rgb_source: str = "none", camera_res: int = 128, **kwargs):
        self.rgb_source = rgb_source
        if rgb_source == "newton":
            kwargs.setdefault("newton_camera", dict(width=camera_res, height=camera_res))
        super().__init__(*args, **kwargs)

    def get_obs(self, info=None, unflattened: bool = False):
        obs = super().get_obs(info, unflattened=unflattened) if unflattened else super().get_obs(info)
        if self.newton_camera is not None and isinstance(obs, dict) and self.obs_mode == "state_dict":
            obs.setdefault("sensor_data", {})["newton_camera"] = {"rgb": self.newton_camera.render_all()}
        return obs

    def render(self):
        if self.newton_camera is not None and self.render_mode in (None, "rgb_array", "sensors"):
            return self.newton_camera.render_all()
        return super().render()


def make_env(route: str = "A", num_envs: int = 1, **kw):
    """Route A: gym.make('FoldTowel-v1', ...) (ManiSkill env); route B: the all-Newton env (same reset/step/info)."""
    if route == "A":
        import gymnasium as gym
        kw.setdefault("obs_mode", "state")
        kw.setdefault("control_mode", "pd_joint_pos")
        kw.setdefault("sim_backend", "physx_cuda")
        kw.setdefault("render_backend", "none")
        return gym.make("FoldTowel-v1", num_envs=num_envs, **kw)
    from imitator.soft.route_b import FoldTowelNewtonEnv
    return FoldTowelNewtonEnv(num_envs=num_envs, **kw)


def run_contract_episode(env, behaviours, settle_steps: int = 60, seed: int = 0):
    """Run one scripted episode per env (behaviours: list of expert / no_grip / drop_midway / drag_on_ground, one per
    env), the evaluator's hold-A settle phase, and return the veritask verdicts (v3 binding + frozen v1 contract)."""
    from imitator.soft.contract_eval import TraceRecorder, load_contracts
    from imitator.soft.expert import joint_space_plan, towel_waypoints, total_time
    u = env.unwrapped
    n = u.num_envs
    assert len(behaviours) == n
    dt = 1.0 / u.control_freq
    T = max(total_time(towel_waypoints(b)) for b in set(behaviours))
    n_policy = int(np.ceil(T / dt))
    plans = {b: joint_space_plan(b, center=u.towel_spec.center, control_dt=dt, n_steps=n_policy) for b in set(behaviours)}
    Q = torch.tensor(np.stack([plans[b][0] for b in behaviours], 1))
    G = torch.tensor(np.stack([plans[b][1] for b in behaviours], 1))
    env.reset(seed=seed)
    rec = TraceRecorder(n, u.control_freq)
    rec.set_init(u.cloth.particle_q().cpu().numpy())
    for k in range(n_policy):
        env.step(scripted_joint_action(u, Q[k], G[k]))
        rec.record(u.cloth_signals(), u.cloth.particle_q(), 0)
    u.hold_robot()
    for _ in range(settle_steps):
        env.step(None)
        rec.record(u.cloth_signals(), u.cloth.particle_q(), 1)
    c3, c1 = load_contracts()
    return rec.evaluate(c3, c1), rec
