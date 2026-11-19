"""Single-arm replay of logged joint commands in imsim (card F1). Used to fit and to validate calibration files.

    from imitator.embodiments.calibration.replay import Replayer
    rp = Replayer("piper", control_freq=30, sim_freq=120)
    rp.set_params(dict(stiffness=[...], damping=[...], force_limit=[...], action_delay_s=[...], prefilter_tau_s=[...]))
    q_sim = rp.run(u, y0, v0, u_hist=u_before)          # (T, n) arm joint trajectory, q_sim[0] = y0

Scene: one imsim arm at the origin, NO ground or table (free-space replay; logged trajectories come from robots
mounted at unknown heights), PhysX CPU, gravity-free robot links unless the parameters say gravity=True (ManiSkill
default: balance_passive_force disables link gravity). The drive targets go through calibration.apply_to_agent, i.e.
the same delay / first-order filter code the imitator.make(..., embodiment_calibration=...) hook uses.
"""
from __future__ import annotations

import numpy as np

_REGISTERED = False


def _register():
    global _REGISTERED
    if _REGISTERED:
        return
    import sapien
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils.registration import register_env

    @register_env("ImsimF1Replay-v1", max_episode_steps=10_000_000)
    class ImsimF1Replay(BaseEnv):
        SUPPORTED_REWARD_MODES = ["none"]

        def __init__(self, *args, robot_uids="imsim_piper", **kwargs):
            super().__init__(*args, robot_uids=robot_uids, **kwargs)

        @property
        def _default_sensor_configs(self):
            return []

        @property
        def _default_human_render_camera_configs(self):
            return []

        def _load_agent(self, options: dict):
            super()._load_agent(options, sapien.Pose())

        def _load_scene(self, options: dict):
            pass

        def _initialize_episode(self, env_idx, options: dict):
            pass

        def evaluate(self):
            return {}

        def _get_obs_extra(self, info: dict):
            return dict()

    _REGISTERED = True


class Replayer:
    def __init__(self, arm: str, control_freq: float = 30, sim_freq: float | None = None, substeps: int = 4):
        import gymnasium as gym
        import torch  # noqa: F401

        from .. import agents as _ag  # noqa: F401  (registers the imsim agents)
        from .. import registry as reg
        _register()
        self.rec = reg.ARMS[arm]
        cf = float(control_freq)
        sf = float(sim_freq) if sim_freq else cf * substeps
        if abs(sf / cf - round(sf / cf)) > 1e-9:
            raise ValueError("sim_freq must be a multiple of control_freq")
        self.env = gym.make("ImsimF1Replay-v1", robot_uids=self.rec.agent_uid, obs_mode="none", reward_mode="none",
                            control_mode="pd_joint_pos", sim_backend="physx_cpu", render_backend="none",
                            sim_config=dict(sim_freq=int(round(sf)), control_freq=int(round(cf))))
        self.env.reset(seed=0)
        self.base = self.env.unwrapped
        self.agent = self.base.agent
        self.arm_ctrl = self.agent.controller.controllers["arm"]
        self.arm_names = [j.name for j in self.arm_ctrl.joints]
        self.active = [j.name for j in self.agent.robot.active_joints]
        self.arm_idx = [self.active.index(n) for n in self.arm_names]
        self.n = len(self.arm_names)
        self.control_dt = 1.0 / cf
        self.sim_dt = 1.0 / sf
        self.default = self.default_params()
        self.params = None
        self.gripper_hold = np.asarray(self.rec.gripper_cmd_open, np.float32) if self.rec.kind == "arm_gripper" else \
            np.zeros(0, np.float32)
        self.set_params({})                              # imsim defaults (identity delay / filter)

    # ------------------------------------------------------------------------------------------ parameters
    def default_params(self) -> dict:
        c = self.arm_ctrl.config
        n = len(self.arm_ctrl.joints)
        b = lambda v: np.broadcast_to(np.asarray(v, float), (n,)).tolist()  # noqa: E731
        return dict(joints=[j.name for j in self.arm_ctrl.joints], stiffness=b(c.stiffness), damping=b(c.damping),
                    force_limit=b(c.force_limit), action_delay_s=[0.0] * n, prefilter_tau_s=[0.0] * n, gravity=False)

    def set_params(self, p: dict):
        """p: dict with stiffness / damping / force_limit / action_delay_s / prefilter_tau_s (per joint) [+ gravity]"""
        from . import apply_to_agent
        full = dict(self.default)
        full.update({k: v for k, v in p.items() if v is not None})
        full["joints"] = self.arm_names
        if full.get("gravity") and not (self.params or {}).get("gravity"):
            pass
        if (self.params or {}).get("gravity") and not full.get("gravity"):
            for link in self.agent.robot.links:          # back to ManiSkill's gravity-free links
                link.disable_gravity = True
        self.arm_ctrl._f1_patched = None                 # force re-application with the new parameters
        self._unpatch()
        apply_to_agent(self.agent, dict(arm=full), self.control_dt, self.sim_dt)
        self.params = full

    def _unpatch(self):
        for attr in ("set_drive_targets", "reset"):
            if attr in self.arm_ctrl.__dict__:
                del self.arm_ctrl.__dict__[attr]

    # ------------------------------------------------------------------------------------------ replay
    def run(self, u: np.ndarray, y0: np.ndarray, v0: np.ndarray | None = None, u_hist: np.ndarray | None = None,
            grip: np.ndarray | None = None) -> np.ndarray:
        """u (T, n) commanded arm joint targets [rad]; y0, v0 initial measured position / velocity; u_hist (H, n) the
        commands BEFORE the segment (oldest first) to pre-roll the delay / filter state; grip (T, g) gripper part of
        the pd_joint_pos action (default: hold open). Returns q (T, n) with q[0] = y0, q[k+1] = after applying u[k]."""
        import torch
        T = len(u)
        rob = self.agent.robot
        q = rob.get_qpos().clone()
        q[0, self.arm_idx] = torch.as_tensor(np.asarray(y0, np.float32))
        qd = torch.zeros_like(q)
        if v0 is not None:
            qd[0, self.arm_idx] = torch.as_tensor(np.asarray(v0, np.float32))
        rob.set_qpos(q)
        rob.set_qvel(qd)
        self.agent.controller.reset()                    # arm filter state <- current qpos
        filt = self.arm_ctrl._f1_filter
        if u_hist is not None and len(u_hist):           # pre-roll delay buffer and low-pass state on past commands
            h = torch.as_tensor(np.asarray(u_hist, np.float32))[:, None, :]
            filt.reset(h[0].clone())
            for k in range(len(h)):
                filt(h[k])
        out = np.empty((T, self.n))
        out[0] = y0
        g = self.gripper_hold if grip is None else None
        steps = self.base._sim_steps_per_control
        scene = self.base.scene
        for k in range(T - 1):
            gk = g if g is not None else np.asarray(grip[k], np.float32)
            a = np.concatenate([np.asarray(u[k], np.float32), gk])[None]
            self.agent.set_action(torch.as_tensor(a))
            for _ in range(steps):
                self.agent.before_simulation_step()
                scene.step()
            out[k + 1] = rob.get_qpos()[0, self.arm_idx].numpy()
        return out

    def close(self):
        self.env.close()
