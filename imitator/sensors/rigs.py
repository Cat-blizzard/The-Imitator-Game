"""Tactile test rig (ManiSkill env, CPU or GPU PhysX, no agent): ImsimTactilePressRig-v1.

One pad link (box; top face = the pad) fixed to a fixed base, and FREE indenter bodies resting on it under gravity
(sphere / cylinder "edge" / box "flat" / box with an off-centre COM / convex file), so the load and its line of action
are known exactly: Fn = m g, CoP = COM projection (statically determinate). Used by the E1 validation and the tests.

    import gymnasium as gym, imitator.sensors.rigs as rigs
    rig = dict(pad=dict(W=0.016, L=0.016, t=0.004), indenters=[dict(kind="sphere", r=0.01, mass=0.5, xy=(0, 0))])
    env = gym.make("ImsimTactilePressRig-v1", num_envs=1, sim_backend="physx_cpu", rig=rig, render_backend="none")
    sensor = TactileSensor(env, "T2", pads=[rigs.press_pad_spec(rig)], layout="16x16")
"""
from __future__ import annotations

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SimConfig

from .pads import PadSpec

PAD_Z = 0.10      # pad top surface height


def press_pad_spec(rig: dict) -> PadSpec:
    p = rig.get("pad", {})
    W, L, t = p.get("W", 0.024), p.get("L", 0.064), p.get("t", 0.004)
    return PadSpec(name="pad", link="pad", origin=(0.0, 0.0, t / 2), R=((1, 0, 0), (0, 1, 0), (0, 0, 1)), size=(W, L),
                   articulation="rig", stiffness=p.get("k"))


def _yaw_q(yaw):
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


@register_env("ImsimTactilePressRig-v1", max_episode_steps=1000000)
class TactilePressRig(BaseEnv):
    SUPPORTED_ROBOTS = ["none"]

    def __init__(self, *args, rig: dict | None = None, robot_uids="none", **kw):
        self.rig = dict(rig or {})
        super().__init__(*args, robot_uids=robot_uids, **kw)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=int(self.rig.get("sim_freq", 500)), control_freq=int(self.rig.get("control_freq", 50)))

    def _load_agent(self, options, *a, **k):
        super()._load_agent(options, *a, **k)

    def _load_scene(self, options: dict):
        p = self.rig.get("pad", {})
        W, L, t = p.get("W", 0.024), p.get("L", 0.064), p.get("t", 0.004)
        ab = self.scene.create_articulation_builder()
        ab.set_name("rig")
        base = ab.create_link_builder()
        base.set_name("base")
        base.add_box_collision(sapien.Pose([0, 0, -t - 0.01]), half_size=[W / 2, L / 2, 0.005])
        pad = ab.create_link_builder(base)
        pad.set_name("pad")
        pad.set_joint_name("pad_joint")
        pad.set_joint_properties("fixed", [], sapien.Pose([0, 0, -t / 2]), sapien.Pose())
        mat = sapien.physx.PhysxMaterial(p.get("mu", 0.8), p.get("mu", 0.8), 0.0)
        pad.add_box_collision(half_size=[W / 2, L / 2, t / 2], material=mat)
        ab.initial_pose = sapien.Pose([0, 0, PAD_Z])
        self.rig_art = ab.build(fix_root_link=True)
        self.indenters = []
        for i, d in enumerate(self.rig.get("indenters", [])):
            b = self.scene.create_actor_builder()
            kind = d["kind"]
            m = float(d.get("mass", 0.5))
            mat = sapien.physx.PhysxMaterial(d.get("mu", 0.8), d.get("mu", 0.8), 0.0)
            if kind == "sphere":
                b.add_sphere_collision(radius=d["r"], material=mat)
            elif kind == "cylinder":                  # axis along local x; lies on the pad
                b.add_cylinder_collision(radius=d["r"], half_length=d["hl"], material=mat)
            elif kind == "box":
                b.add_box_collision(half_size=d["half"], material=mat)
            elif kind == "convex":
                b.add_convex_collision_from_file(d["file"], scale=d.get("scale", [1, 1, 1]), material=mat)
            com = sapien.Pose(list(d.get("com_offset", [0, 0, 0])))
            b.set_mass_and_inertia(m, com, d.get("inertia", [1e-4, 1e-4, 1e-4]))
            b.initial_pose = sapien.Pose(self._rest_pos(d), _yaw_q(d.get("yaw", 0.0)))
            self.indenters.append(b.build(name=f"ind{i}"))

    def _rest_pos(self, d):
        kind = d["kind"]
        z = {"sphere": lambda: d["r"], "cylinder": lambda: d["r"], "box": lambda: d["half"][2],
             "convex": lambda: d.get("rest_z", 0.0)}[kind]()
        xy = d.get("xy", (0.0, 0.0))
        return [xy[0], xy[1], PAD_Z + z + d.get("lift", 0.0)]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        for a, d in zip(self.indenters, self.rig.get("indenters", [])):
            from mani_skill.utils.structs.pose import Pose
            n = len(env_idx)
            p = torch.tensor(self._rest_pos(d), dtype=torch.float32).repeat(n, 1)
            q = torch.tensor(_yaw_q(d.get("yaw", 0.0)), dtype=torch.float32).repeat(n, 1)
            a.set_pose(Pose.create_from_pq(p, q))
            a.set_linear_velocity(torch.zeros((n, 3)))
            a.set_angular_velocity(torch.zeros((n, 3)))

    def evaluate(self):
        return {}

    def _get_obs_agent(self):
        return {}

    def get_state_dict(self):                     # ManiSkill 3.0.1 assumes an agent here
        return self.scene.get_sim_state()

    def set_state_dict(self, state, env_idx=None):
        self.scene.set_sim_state(state, env_idx)

    def _get_obs_extra(self, info: dict):
        return {}

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
