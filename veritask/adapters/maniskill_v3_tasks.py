"""Protocol-v3 task bindings for the three rigid ManiSkill tasks.

Each v3 binding extends its v2 binding (adapters/maniskill_v2_tasks.py): every v1 / v2 signal is still recorded, so the
v1 and v2 contracts can be judged on the same rollout. Added per task:
  all      robot_qvel_max          largest ARM joint speed over the last control step, from joint positions [rad/s]
  juicer   knob_qvel_abs           |dial joint velocity| [rad/s]
           knob_enclosure_clearance_m   protocol_v3 enclosure clearance (maniskill_v3.Enclosure) for the dial
  cups     cup{1,2}_{x,y,z}        cup origin (bottom centre) position [m]
           cups_enclosure_clearance_m   enclosure clearance, both grippers x both cups
  poke     cube_{x,y,z}, peg_{x,y,z}, peg_speed
           robot_cube_impulse_xy_Ns / _cum_Ns   in-plane robot-cube impulse (diagnostic; PhysX impulses are normal-only)
           robot_cube_push_disp_m  net cube displacement caused by robot contact (maniskill_v3.PushTracker: while any robot
                                   link touches the cube and while it keeps coasting afterwards, until it is at rest or the
                                   peg touches it); trace.max of it is the PokeCube v3 robot-cube term
"""
from __future__ import annotations

import numpy as np

from .maniskill_v2_tasks import PlaceCupPlateV2, PokeCubeV2, PressJuicerV2, _all_robot_links
from .maniskill_v3 import Enclosure, PairV3, PushTracker, TraceRecorderV3

PUSH_REST_SPEED = 0.005      # a push episode ends when the cube is slower than this [m/s] (or the peg touches it)


def _arm_joint_idx(ag):
    grip = set(getattr(ag, "gripper_joint_names", []) or [])
    return np.array([i for i, j in enumerate(ag.robot.get_active_joints()) if j.name not in grip], dtype=int)


class RobotSpeed:
    """largest ARM joint speed of every robot over the last control step, from joint POSITIONS:
    max_j |q_j(t) - q_j(t-1)| / control_dt [rad/s]. Not PhysX's reported joint velocity: it stays non-zero while nothing
    moves when the robot presses on something through a held object (0.012 rad/s for a static arm holding the peg against
    the cube; fingers closed on an object 0.017-0.043 m/s; E15 development runs). Gripper finger joints are excluded.
    The first step after a reset (or any jump in the env's step counter, e.g. a restored snapshot) uses PhysX's velocity."""

    def __init__(self):
        self.prev = None          # (id(env), elapsed_steps, q)

    def __call__(self, u) -> float:
        agents = list(u.agent.agents) if hasattr(u.agent, "agents") else [u.agent]
        q = np.concatenate([ag.robot.get_qpos()[0].cpu().numpy()[_arm_joint_idx(ag)] for ag in agents]).astype(float)
        k = int(np.asarray(u.elapsed_steps).reshape(-1)[0]) if hasattr(u, "elapsed_steps") else None
        p = self.prev
        self.prev = (id(u), k, q)
        if p is not None and p[0] == id(u) and k is not None and p[1] == k - 1 and len(p[2]) == len(q):
            return float(np.max(np.abs(q - p[2])) / float(u.control_timestep))
        return float(max(np.abs(ag.robot.get_qvel()[0].cpu().numpy()[_arm_joint_idx(ag)]).max() for ag in agents))


def _xyz(name, i):
    return lambda u: float(getattr(u, name).pose.p[0, i])


class _V3:
    def trackers(self):
        return ()

    def recorder(self, env, v1_probes: dict | None = None) -> TraceRecorderV3:
        probes = dict(v1_probes or {})
        probes.update(self.probes())
        return TraceRecorderV3(env, probes, pairs=self.pairs(), release=self.release(), trackers=self.trackers())


class PressJuicerV3(_V3, PressJuicerV2):
    task_id = "press_juicer_v3"

    def probes(self):
        p = super().probes()
        enc = Enclosure(lambda u: [u.partnet_link])
        p.update(knob_qvel_abs=lambda u: float(abs(u.partnet_link.joint.qvel.flatten()[0])), robot_qvel_max=RobotSpeed(),
                 knob_enclosure_clearance_m=enc)
        return p


class PlaceCupPlateV3(_V3, PlaceCupPlateV2):
    task_id = "place_cup_plate_v3"

    def probes(self):
        p = super().probes()
        for c in ("cup_1", "cup_2"):
            for i, ax in enumerate("xyz"):
                p[f"{c.replace('_', '')}_{ax}"] = _xyz(c, i)
        p.update(robot_qvel_max=RobotSpeed(), cups_enclosure_clearance_m=Enclosure(lambda u: [u.cup_1, u.cup_2]))
        return p


class PokeCubeV3(_V3, PokeCubeV2):
    task_id = "poke_cube_v3"

    def probes(self):
        p = super().probes()
        for o in ("cube", "peg"):
            for i, ax in enumerate("xyz"):
                p[f"{o}_{ax}"] = _xyz(o, i)
        p.update(peg_speed=lambda u: float(np.linalg.norm(u.peg.linear_velocity[0].cpu().numpy())), robot_qvel_max=RobotSpeed())
        return p

    def pairs(self):
        cube = lambda u: [u.cube]  # noqa: E731
        return {"robot_cube": PairV3(_all_robot_links, cube, gap=True, cumulative=True, inplane=True),
                "peg_cube": PairV3(lambda u: [u.peg], cube)}

    def trackers(self):
        return (PushTracker("robot_cube_push", "robot_cube", obj=lambda u: u.cube, others=("peg_cube",), rest_speed=PUSH_REST_SPEED),)


BINDINGS_V3 = {b.task_id: b for b in (PressJuicerV3, PlaceCupPlateV3, PokeCubeV3)}
