"""A2: the FoldTowel-v1 (Panda, Newton towel) expert of imitator.soft.expert.run_towel with its timeline parameters exposed
(imitator.planning.params 'FoldTowel-v1'). With the default parameters it executes the same joint targets as run_towel
(same towel_waypoints, same towel_points insets, same IK and phases); soft/expert.py itself is not changed."""
from __future__ import annotations

import numpy as np


def run_towel_p(env, mode: str = "expert", params: dict | None = None, read_pose: bool = True) -> dict:
    """FoldTowel-v1 expert driving env.step for every world; returns a log dict (as soft.expert.run_towel)"""
    import torch
    from . import params as PR
    from ..data.phase import set_phase
    from ..soft.expert import interp_waypoints, phase_at, segment_phases, total_time, towel_points, towel_waypoints
    from ..soft.panda_ik import joint_trajectory
    from ..soft.route_a import scripted_joint_action
    p = PR.pv(params, "FoldTowel-v1", env, mode)
    u = env.unwrapped
    n = u.num_envs
    dt = 1.0 / float(u.control_freq)
    base = u.agent.robot.pose.p[0].detach().cpu().numpy().astype(float)
    q = u.cloth.particle_q().detach().cpu().numpy().astype(float)
    tcp = u.agent.tcp_pose.p.detach().cpu().numpy().astype(float)
    res = u.towel_spec.res
    approach_s = p["approach_s"]
    wkw = dict(z_grasp=p["z_grasp"], up=p["up"], low=p["low"], retreat=p["retreat"], settle=p["settle"],
               speed=p["speed"], t_close=p["t_close"], t_open=p["t_open"])
    cache, plans, err = {}, [], 0.0
    for w in range(n):
        if read_pose:
            A, C = towel_points(q[w], res, inset_grasp=p["inset_grasp"], inset_release=p["inset_release"])
        else:
            wp0 = towel_waypoints(mode, half=u.towel_spec.half_size, center=u.towel_spec.center, **wkw)
            A, C = np.array(wp0[0][:2]), np.array(wp0[-1][:2])
        key = tuple(np.round(np.r_[A, C, tcp[w]], 6))
        if key not in cache:
            wps = towel_waypoints(mode, half=u.towel_spec.half_size, center=u.towel_spec.center, A=A, C=C, **wkw)
            if approach_s > 0:
                wps = [(*tcp[w], 0, approach_s)] + wps
            ph = segment_phases(wps, approach_s > 0)
            T = int(np.ceil(total_time(wps) / dt))
            tgt = np.array([interp_waypoints(wps, (k + 1) * dt) for k in range(T)])
            qj, e = joint_trajectory(tgt[:, :3], base_xyz=base)
            cache[key] = (qj.astype(np.float32), tgt[:, 3].astype(np.float32),
                          [phase_at(wps, ph, (k + 1) * dt) for k in range(T)], e, A, C)
        plans.append(cache[key])
        err = max(err, cache[key][3])
    T = max(len(pl[0]) for pl in plans)
    Q = torch.tensor(np.stack([np.vstack([pl[0], np.repeat(pl[0][-1:], T - len(pl[0]), 0)]) for pl in plans], 1))
    G = torch.tensor(np.stack([np.r_[pl[1], np.repeat(pl[1][-1:], T - len(pl[1]))] for pl in plans], 1))
    phases = plans[0][2] + [plans[0][2][-1]] * (T - len(plans[0][2]))
    for k in range(T):
        set_phase(env, phases[k])
        env.step(scripted_joint_action(u, Q[k], G[k]))
    return dict(status="ok" if err < 1e-3 else f"ik: max error {err:.4f}", steps=T, mode=mode, max_ik_err=float(err),
                free_steps=int(sum(ph == "free" for ph in phases)), A=[pl[4].round(4).tolist() for pl in plans][:4],
                C=[pl[5].round(4).tolist() for pl in plans][:4])
