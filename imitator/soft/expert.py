"""Scripted towel-fold behaviours in TCP space (shared by both routes and the kinematic-gripper probe).

Waypoints follow research/scripts/newton_towel_fold_probe.py (the earlier two-way gantry probe) so results can be
compared: grasp 2.5 cm inside corner A, lift to 10 cm, carry over a point 3 cm inside corner C, lower to 3.5 cm, open,
retreat. Behaviours (the contract's counterexamples): expert, no_grip (never closes), drop_midway (opens halfway through
the carry), drag_on_ground (grips and drags A to C without lifting).
Waypoint tuple: (x, y, z, grip 0=open 1=closed, duration of the segment that STARTS at this waypoint [s]).

run_towel(env, mode) is the same expert as an env driver (FoldTowel-v1, any number of worlds) for imitator.generate: it reads
each world's towel corners from the cloth (privileged state, so towel pose offsets are followed), adds a FREE approach
segment from the arm's current TCP to the first waypoint (instead of a one-step jump), and annotates phases
(imitator.data.phase): FREE for that approach and for the retreat after the release, CONTACT from the descent to the grasp
until the gripper has opened.
"""
from __future__ import annotations

import numpy as np

BEHAVIOURS = ("expert", "no_grip", "drop_midway", "drag_on_ground")
TCP_IN_HAND = 0.1034            # panda_hand_tcp joint offset (panda_v2.urdf)
FINGER_IN_HAND = 0.0584         # panda_finger_joint1/2 origin z


def towel_waypoints(mode: str, half: float = 0.15, center=(0.0, 0.0), z_grasp: float = 0.011, up: float = 0.10,
                    low: float = 0.035, retreat: float = 0.12, settle: float = 0.0, speed: float = 1.0,
                    t_close: float = 1.0, t_open: float = 1.0, A=None, C=None):
    """t_close / t_open: the arm waits this long while the gripper closes / opens (the earlier probe used 0.5 / 0.4 s
    with an ideal gripper; the PhysX and MuJoCo Panda grippers need ~0.8 s for the full 4 cm stroke, and a release that
    happens while the arm already moves on is not a release at that place). A / C (xy) override the grasp and release
    points derived from `center` / `half` (run_towel passes the points read from the cloth)."""
    cx, cy = center
    A = np.array([cx - half + 0.025, cy - half + 0.025]) if A is None else np.asarray(A, float)[:2]
    C = np.array([cx + half - 0.03, cy + half - 0.03]) if C is None else np.asarray(C, float)[:2]
    s = 1.0 / speed
    g = 0 if mode == "no_grip" else 1
    w = [(*A, 0.08, 0, 0.4 * s), (*A, z_grasp, 0, 0.5 * s), (*A, z_grasp, g, t_close * s)]
    if mode == "drag_on_ground":
        w += [(*C, z_grasp, 1, 2.0 * s), (*C, z_grasp, 0, t_open * s), (*C, retreat, 0, 0.5 * s)]
    else:
        mid = (A + C) / 2
        w += [(*A, up, g, 0.8 * s)]
        if mode == "drop_midway":
            w += [(*mid, up, g, 0.8 * s), (*mid, up, 0, t_open * s), (*C, up, 0, 0.8 * s)]
        else:
            w += [(*C, up, g, 1.6 * s)]
        w += [(*C, low, g, 0.6 * s), (*C, low, 0, t_open * s), (*C, retreat, 0, 0.5 * s)]
    w += [(*C, retreat, 0, max(settle, 1e-3))]
    if mode not in BEHAVIOURS:
        raise ValueError(mode)
    return w


def interp_waypoints(wps, t: float):
    """(x, y, z, grip) at time t: position linear between waypoints; grip switches at the START of a segment."""
    acc = 0.0
    for i, w in enumerate(wps):
        last = i == len(wps) - 1
        if t < acc + w[4] or last:
            if last:
                return np.array(w[:4], float)
            nxt = wps[i + 1]
            u = min(max((t - acc) / w[4], 0.0), 1.0)
            p = np.array(w[:3]) * (1 - u) + np.array(nxt[:3]) * u
            return np.array([*p, _grip_at(w, nxt, u)])
        acc += w[4]
    return np.array(wps[-1][:4], float)


def _grip_at(w, nxt, u):
    # a closing / opening segment (same position, grip changes) ramps the grip over the segment
    if w[3] != nxt[3] and np.allclose(w[:3], nxt[:3]):
        return w[3] + (nxt[3] - w[3]) * u
    return w[3]


def total_time(wps) -> float:
    return float(sum(w[4] for w in wps))


def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


TCP_DOWN_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])   # hand z axis pointing down, fingers closing along world y


def panda_gripper_link_poses(tcp, q_finger: float, quat_wxyz=TCP_DOWN_WXYZ):
    """Poses [x y z qw qx qy qz] of panda_hand, panda_leftfinger, panda_rightfinger for a TCP position/orientation and
    finger joint position q (each finger; gap between the pads = 2q)."""
    R = _rot(quat_wxyz)
    hand = np.asarray(tcp, float) - R @ np.array([0.0, 0.0, TCP_IN_HAND])
    lf = hand + R @ np.array([0.0, q_finger, FINGER_IN_HAND])
    rf = hand + R @ np.array([0.0, -q_finger, FINGER_IN_HAND])
    return np.stack([np.r_[p, quat_wxyz] for p in (hand, lf, rf)]).astype(np.float32)


def joint_space_plan(mode: str, center=(0.0, 0.0), control_dt: float = 0.05, n_steps: int | None = None,
                     base_xyz=(-0.615, 0.0, 0.0)):
    """Per control step k (target for the end of step k): arm joint targets (n, 7) by IK and grip (n,) in [0, 1].
    Padded with the last target up to n_steps."""
    from .panda_ik import joint_trajectory
    wps = towel_waypoints(mode, center=center)
    n = int(np.ceil(total_time(wps) / control_dt))
    tgt = np.array([interp_waypoints(wps, (k + 1) * control_dt) for k in range(n)])
    q, err = joint_trajectory(tgt[:, :3], base_xyz=base_xyz)
    assert err < 1e-3, f"IK failed for {mode}: {err}"
    g = tgt[:, 3]
    if n_steps is not None and n_steps > n:
        q = np.vstack([q, np.repeat(q[-1:], n_steps - n, 0)])
        g = np.r_[g, np.repeat(g[-1:], n_steps - n)]
        tgt = np.vstack([tgt, np.repeat(tgt[-1:], n_steps - n, 0)])
    return q.astype(np.float32), g.astype(np.float32), tgt


# ------------------------------------------------------------------------------------------------ data-engine driver
def towel_points(q, res: int, inset_grasp: float = 0.025, inset_release: float = 0.03):
    """grasp point A (inset from corner 0 along both towel edges) and release point C (inset from the opposite corner),
    xy, from one world's particle positions (V, 3) of a (res+1)^2 grid"""
    r = res + 1
    p00, px, py, pxy = q[0, :2], q[r - 1, :2], q[r * r - r, :2], q[r * r - 1, :2]
    ex = (px - p00) / (np.linalg.norm(px - p00) + 1e-12)
    ey = (py - p00) / (np.linalg.norm(py - p00) + 1e-12)
    return p00 + inset_grasp * (ex + ey), pxy - inset_release * (ex + ey)


def segment_phases(wps, approach: bool) -> list[str]:
    """phase of every waypoint segment: FREE for the approach segment (if prepended) and from the start of the final
    ascent to the retreat height (after the release) on; CONTACT otherwise"""
    from ..data.phase import CONTACT, FREE
    z_ret = wps[-1][2]
    first_ret = next(i for i, w in enumerate(wps) if abs(w[2] - z_ret) < 1e-9 and i > 0)
    out = []
    for i in range(len(wps)):
        if (approach and i == 0) or i >= first_ret - 1:
            out.append(FREE)
        else:
            out.append(CONTACT)
    return out


def phase_at(wps, phases, t: float) -> str:
    acc = 0.0
    for i, w in enumerate(wps):
        if t < acc + w[4] or i == len(wps) - 1:
            return phases[i]
        acc += w[4]
    return phases[-1]


def run_towel(env, mode: str = "expert", approach_s: float = 0.6, read_pose: bool = True) -> dict:
    """FoldTowel-v1 expert driving env.step for every world (see the module docstring); returns a log dict"""
    import torch
    from ..data.phase import set_phase
    from .panda_ik import joint_trajectory
    from .route_a import scripted_joint_action
    u = env.unwrapped
    n = u.num_envs
    dt = 1.0 / float(u.control_freq)
    base = u.agent.robot.pose.p[0].detach().cpu().numpy().astype(float)
    q = u.cloth.particle_q().detach().cpu().numpy().astype(float)
    tcp = u.agent.tcp_pose.p.detach().cpu().numpy().astype(float)
    res = u.towel_spec.res
    cache, plans, err = {}, [], 0.0
    for w in range(n):
        if read_pose:
            A, C = towel_points(q[w], res)
        else:
            wp0 = towel_waypoints(mode, half=u.towel_spec.half_size, center=u.towel_spec.center)
            A, C = np.array(wp0[0][:2]), np.array(wp0[-1][:2])
        key = tuple(np.round(np.r_[A, C, tcp[w]], 6))
        if key not in cache:
            wps = towel_waypoints(mode, half=u.towel_spec.half_size, center=u.towel_spec.center, A=A, C=C)
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
    T = len(plans[0][0])
    Q = torch.tensor(np.stack([p[0] for p in plans], 1))
    G = torch.tensor(np.stack([p[1] for p in plans], 1))
    phases = plans[0][2]
    for k in range(T):
        set_phase(env, phases[k])
        env.step(scripted_joint_action(u, Q[k], G[k]))
    return dict(status="ok" if err < 1e-3 else f"ik: max error {err:.4f}", steps=T, mode=mode, max_ik_err=float(err),
                free_steps=int(sum(p == "free" for p in phases)), A=[p[4].round(4).tolist() for p in plans][:4],
                C=[p[5].round(4).tolist() for p in plans][:4])
