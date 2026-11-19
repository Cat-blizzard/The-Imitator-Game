"""Imitator-fork (The-Imitator-Game, a ManiSkill 3 fork with dual-Panda tasks) adapter for the metamorphic tests.

Same mechanics as adapters/maniskill.py (CPU PhysX, recorded actions replayed from reset(seed), IK retreat, sub-step
contact hook, kinematic freeze, half-step env), extended to two arms:
  actions   OrderedDict {'panda_wristcam-0': [q(7), grip], 'panda_wristcam-1': [...]} (pd_joint_pos), as the fork's
            dual motion planner sends them
  hold      each arm keeps the arm target of its last action; gripper keep / open (+1)
  tcp move  both TCPs lifted (ManiSkill CPU IK per arm)
  freeze    actors kinematic; the juicer articulation's joints reset to their initial qpos after every sub-step
Official verdict: success_once (the fork's eval template reports success_once); the predicates of the juicer and the
kettle tasks are themselves latched (juicer_ever_pressed / kettle_ever_poured), which this adapter does not alter;
goto restores env.get_state_dict() AND these latch attributes (and kettle_up_axis) as they were after step t.
M6/M7 replays run in dedicated envs reconfigured before every replay (as in adapters/maniskill.py).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..adapter import Rollout
from .maniskill import GRIP_OPEN, ManiSkillAdapter, _np

TASKS = {
    "TwoRobotPlaceCupPlate-v1": dict(objs=["cup_1", "cup_2", "tray"], freeze=["cup_1", "cup_2"], nudge=["cup_1", "cup_2"],
                                     solution="_036_place_cup_plate"),
    "TwoRobotPressJuicer-v1": dict(objs=["partnet_link"], freeze=[], freeze_art=["juicer"], nudge=[],
                                   solution="_050_press_juicer"),
    "TwoRobotPourKettle-v1": dict(objs=["kettle", "cup"], freeze=["kettle"], nudge=["kettle"],
                                  solution="_020_pour_kettle"),
}


class ImitatorAdapter(ManiSkillAdapter):
    benchmark = "imitator"

    def __init__(self, env_id: str, contact_eps_n: float = 0.05):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        import torch
        torch.set_num_threads(1)
        self._gym, self._torch = gym, torch
        self.env_id, self.source = env_id, "official_mp"
        cfg = TASKS[env_id]
        self.cfg = cfg
        self.objs, self.freeze_names, self.nudge_names = cfg["objs"], cfg["freeze"], cfg["nudge"]
        self.control_mode, self.obs_mode = "pd_joint_pos", "none"
        self.env = self._make(1.0)
        self.u = self.env.unwrapped
        self.env.reset(seed=0)
        self.control_freq = float(self.u.control_freq)
        self.sim_freq = float(self.u.sim_freq)
        self._half = None
        self._rep = None
        self._snaps = []
        self.contact_eps = contact_eps_n
        self._mon = None
        self._freeze_art = None
        self.u._after_simulation_step = self._after_sub
        self.agents = list(self.u.agent.agents)
        self.uids = [f"panda_wristcam-{i}" for i in range(len(self.agents))]
        self._build_maps()
        from mani_skill.utils import gym_utils
        self.max_steps = int(gym_utils.find_max_episode_steps_value(self.env))
        self._kins = {}
        self._ref = None
        self._last_action = None
        self._tcp_q0 = None
        self.has_policy = False
        self.net = None

    def _make(self, scale: float):
        # the fork's table scene adds visual entities that need a render system: render device = lavapipe ('pci:0',
        # as in research/scripts/e4_*.py); nothing is rendered unless render_mode is set
        kw = dict(obs_mode="none", control_mode="pd_joint_pos", render_mode=None, sim_backend="physx_cpu",
                  render_backend="pci:0")
        if scale != 1.0:
            kw["sim_config"] = dict(sim_freq=int(round(self.sim_freq / scale)), control_freq=int(self.control_freq))
        return self._gym.make(self.env_id, **kw)

    def _build_maps(self):
        import sapien
        u = self.u
        self.robot_ent = {}
        for i, ag in enumerate(self.agents):
            for l in ag.robot.links:
                for c in l._objs:
                    self.robot_ent[c.entity] = f"{i}:{l.name}"
        robot_arts = {id(ag.robot) for ag in self.agents}
        self.obj_ent = {}
        for name, a in u.scene.actors.items():
            for b in a._bodies:
                if isinstance(b, sapien.physx.PhysxRigidDynamicComponent) and not b.kinematic:
                    self.obj_ent[b.entity] = name
        for name, art in u.scene.articulations.items():
            if id(art) in robot_arts or any(art is ag.robot for ag in self.agents):
                continue
            for l in art.links:
                for c in l._objs:
                    self.obj_ent[c.entity] = f"{name}/{l.name}"
        self.task_actor_names = set()
        for n in self.objs + self.cfg.get("freeze_art", []):
            a = getattr(self.u, n, None)
            if a is not None and hasattr(a, "name"):
                self.task_actor_names.add(a.name)
        if "freeze_art" in self.cfg:
            self.task_actor_names |= {v for v in self.obj_ent.values() if v.startswith("juicer")}

    # ------------------------------------------------------------------ contacts / freeze hook
    def _after_sub(self):
        if self._freeze_art is not None:
            env_u, arts = self._freeze_art
            for art, root0, q0 in arts:
                self._pin(art, root0, q0)
        super()._after_sub()

    def _pin(self, art, root0, q0):
        art.set_root_pose(root0)
        z = self._torch.zeros((1, 3))
        art.set_root_linear_velocity(z); art.set_root_angular_velocity(z)
        art.set_qpos(q0); art.set_qvel(self._torch.zeros_like(q0))

    # ------------------------------------------------------------------ state
    def object_poses(self) -> dict:
        out = {}
        for n in self.objs:
            a = getattr(self.u, n, None)
            if a is None:
                continue
            p = a.pose
            out[n] = np.concatenate([_np(p.p).reshape(-1)[:3], _np(p.q).reshape(-1)[:4]])
        if "freeze_art" in self.cfg:
            q = _np(self.u.partnet_link.joint.qpos).reshape(-1)
            out["juicer_knob_qpos"] = np.array([float(q[0]), 0, 0, 1, 0, 0, 0])
        return out

    def grasped(self) -> dict:
        out = {}
        for n in self.objs:
            a = getattr(self.u, n, None)
            if a is None or n == "tray":
                continue
            try:
                out[n] = bool(any(bool(_np(ag.is_grasping(a)).reshape(-1)[0]) for ag in self.agents))
            except Exception:
                pass
        return out

    def tcp_positions(self) -> dict:
        return {self.uids[i]: _np(ag.tcp.pose.p).reshape(-1)[:3].astype(float) for i, ag in enumerate(self.agents)}

    def _qpos_i(self, i):
        return _np(self.agents[i].robot.get_qpos()).reshape(-1).astype(float)

    # ------------------------------------------------------------------ actions
    def step(self, action):
        a = OrderedDict((k, np.asarray(v, dtype=np.float32).reshape(-1)) for k, v in action.items())
        self._last_action = OrderedDict((k, v.copy()) for k, v in a.items())
        self.env.step(a)

    def set_hold_reference(self) -> None:
        if self._last_action is not None:
            self._ref = OrderedDict((k, v[:-1].astype(float).copy()) for k, v in self._last_action.items())
            self._grip = OrderedDict((k, float(v[-1])) for k, v in self._last_action.items())
        else:
            self._ref = OrderedDict((self.uids[i], self._qpos_i(i)[:7]) for i in range(len(self.agents)))
            self._grip = OrderedDict((k, GRIP_OPEN) for k in self.uids)
        self._tcp_q0 = None

    def hold_action(self, gripper: str = "keep"):
        return OrderedDict((k, np.concatenate([self._ref[k], [self._grip[k] if gripper == "keep" else GRIP_OPEN]]))
                           for k in self.uids)

    def _kin(self, i):
        if i not in self._kins:
            import torch
            from mani_skill.agents.controllers.utils.kinematics import Kinematics
            ag = self.agents[i]
            self._kins[i] = Kinematics(ag.urdf_path, ag.tcp.name, ag.robot, torch.arange(7, dtype=torch.int32))
        return self._kins[i]

    def tcp_action(self, targets: dict):
        from mani_skill.utils.structs.pose import Pose
        if self._tcp_q0 is None:
            self._tcp_q0 = {k: _np(self.agents[i].tcp.pose.q).reshape(-1)[:4].copy() for i, k in enumerate(self.uids)}
        out = OrderedDict()
        for i, k in enumerate(self.uids):
            ag = self.agents[i]
            tgt_w = Pose.create_from_pq(self._torch.tensor(np.asarray(targets[k], np.float32)[None]),
                                        self._torch.tensor(self._tcp_q0[k][None].astype(np.float32)))
            res = self._kin(i).compute_ik(ag.robot.pose.inv() * tgt_w, ag.robot.get_qpos())
            q_arm = self._ref[k] if res is None else _np(res).reshape(-1)[:7]
            if res is None:
                self._ik_fail = getattr(self, "_ik_fail", 0) + 1
            out[k] = np.concatenate([q_arm, [GRIP_OPEN]])
        return out

    # ------------------------------------------------------------------ episodes
    def _reset(self, ep, env=None):
        env = env or self.env
        out = env.reset(seed=int(ep["seed"]))
        if env is self.env:
            self._last_action = None
        return out

    def rollout(self, ep) -> Rollout:
        import importlib
        sol = importlib.import_module(f"mani_skill.examples.motionplanning.dual.solutions.{self.cfg['solution']}")
        ad = self
        gym = self._gym

        class Rec(gym.Wrapper):
            def __init__(s, env):
                super().__init__(env); s.actions, s.succ = [], []
            def step(s, a):
                out = s.env.step(a)
                s.actions.append(OrderedDict((k, np.asarray(v, np.float32).reshape(-1).copy()) for k, v in a.items()))
                s.succ.append(ad.success())
                ad._snaps.append(ad._state())
                return out
        self._snaps = []
        rec = Rec(self.env)
        res = sol.solve(rec, seed=int(ep["seed"]), debug=False, vis=False)
        planned = not (isinstance(res, int) and res == -1)
        self._last_action = rec.actions[-1] if rec.actions else None
        return Rollout(success=list(rec.succ), official=bool(any(rec.succ)), official_semantics="any",
                       actions=rec.actions,
                       meta=dict(planned=planned, replay_is_original=False, goto="state_dict",
                                 success_at_end=bool(rec.succ and rec.succ[-1]),
                                 end_poses={k: v.tolist() for k, v in self.object_poses().items()},
                                 max_episode_steps=self.max_steps))

    LATCHES = ("juicer_ever_pressed", "kettle_ever_poured", "kettle_up_axis")

    def _extra_state(self):
        return {k: getattr(self.u, k).clone() for k in self.LATCHES if hasattr(self.u, k)}

    def _set_extra_state(self, extra):
        for k, v in extra.items():
            setattr(self.u, k, v.clone())

    def replay(self, ep, ro: Rollout, freeze: bool = False, dt_scale: float = 1.0) -> dict:
        env = self._replay_env(dt_scale)
        env.reset(seed=int(ep["seed"]), options={"reconfigure": True})
        env.reset(seed=int(ep["seed"]))
        u = env.unwrapped
        frozen = self._set_kinematic(env, self.freeze_names, True) if freeze else []
        if freeze and self.cfg.get("freeze_art"):
            arts = []
            for n in self.cfg["freeze_art"]:
                art = getattr(u, n, None)
                if art is not None:
                    arts.append((art, art.pose, art.get_qpos().clone())); frozen.append(n)
            self._freeze_art = (u, arts)
            u._after_simulation_step = self._after_sub_replay
        succ = []
        try:
            for a in ro.actions:
                env.step(OrderedDict((k, np.asarray(v, np.float32)) for k, v in a.items()))
                succ.append(bool(_np(u.evaluate()["success"]).reshape(-1)[0]))
        finally:
            if freeze:
                self._set_kinematic(env, self.freeze_names, False)
                self._freeze_art = None
        return dict(success=succ, frozen=frozen, dt=1.0 / float(u.sim_freq))

    def _after_sub_replay(self):
        """sub-step hook of the replay envs: only the articulation freeze (no contact monitor)"""
        if self._freeze_art is not None:
            _, arts = self._freeze_art
            for art, root0, q0 in arts:
                self._pin(art, root0, q0)

    def nudge(self, names, dv) -> bool:
        if not names:
            return False
        return super().nudge(names, dv)
