"""Scripted dual-arm towel half fold (DualFoldTowel-v1) for any parallel-gripper dual configuration, and the contract's
designed failures. Waypoints follow E20 (imitator.soft.expert.towel_waypoints) per arm: grasp 2.5 cm inside the far corner
(fingertips 2 mm above the table), close to the cloth-close command, lift, carry over a point inside the near corner,
lower, wait, open, retreat. Both arms run the same timeline (mirrored in y). The expert's timing / heights are in
EXPERT (E29b values; E23 used up=0.10, low=0.035, t_wait=0, t_open=1.0, t_retreat=0.5, inset_release=0.03).
Behaviours: expert, no_grip (never closes), drop_midway (opens halfway through the carry), drag_on_ground (grips and
drags the corner to the target without lifting).
Orientation: closing axis along world y or x, approach tilt 0 / 0.3 / 0.55 rad (approach leaning away from the base, so
small arms reach the far corner low); the first (axis, tilt) whose whole TCP path is IK-trackable (pinocchio CLIK) is used.
Approach from home: mplib RRT to the first waypoint (table + other arm as obstacles).

Phases (imitator.data.phase, E29b): every waypoint segment carries a phase tag (6th tuple element): the transit from home to
the first waypoint (8 cm above the corner) is FREE; the descent to the corner, the close, lift, carry, lower, wait and
open are CONTACT; everything after the gripper has opened (retreat; for drop_midway the rest of the motion after the
mid-air opening) is FREE. plan_batch returns per-step phase codes per arm (info['_phases'], 1 = CONTACT) and the cloth
runner (imitator.soft_eval.run_fold_batch) calls set_phase(env, ..., arm=i) before every env.step (world 0's behaviour when
the worlds of a batch run different behaviours; the per-world codes are kept on env.unwrapped._imsim_phase_worlds).
"""
from __future__ import annotations

import numpy as np
import sapien
from transforms3d.quaternions import mat2quat

from .mplib_planner import ArmPlanner, arm_points
from .scripted import UP, frame, tcp_extent
from ..soft.expert import interp_waypoints, total_time

BEHAVIOURS = ("expert", "no_grip", "drop_midway", "drag_on_ground")
# waypoint shapes of every behaviour name (the cover task adds short_fold = expert shape with an env-chosen short release
# point, hold_cloth = expert without opening / retreat)
SHAPES = dict(expert="expert", no_grip="no_grip", drop_midway="drop_midway", drag_on_ground="drag_on_ground",
              short_fold="expert", hold_cloth="hold_cloth", hold_up="hold_cloth", overshoot="expert")
IDLE_MODES = ("no_op",)          # every arm stays at its home configuration
TILTS = (0.0, 0.3, 0.55)
F, C_ = 0, 1                     # phase codes: FREE, CONTACT
# E29b expert parameters (heights [m], segment durations [s]); the counterexamples use the same values. The E23 values
# were: up 0.10, low 0.035, d_desc 0.4, d_close 0.5, d_lift 1.0, d_carry 0.8, d_lower 1.6, d_wait 0, d_open 0.6,
# d_retreat 1.0, d_hold 0.5 (soft.expert.towel_waypoints semantics: a waypoint's duration is the segment that STARTS at it).
# E29b changes (expert pilots on the v2 draft / frozen v2, 20 worlds per config): release with the fingertips 6 mm above
# the grasp height (low_rel; E23 released at TCP z 3.5 cm, below the UR5's grasp height), hold still 0.5 s before
# opening (d_wait), open over 0.8 s, stay 0.5 s (d_post), back off 4 cm along -z_tcp (a tilted PiPER gripper retracted
# straight up dragged the released corner along), carry over 1.2 s. Final pilots: UR5 20/20, ARX 19/20, PiPER 20/20.
EXPERT = dict(up=0.10, low=0.035, retreat=0.12, d_desc=0.4, d_close=0.5, d_lift=1.0, d_carry=1.2, d_lower=1.6,
              d_wait=0.5, d_open=0.8, d_retreat=1.0, d_hold=0.5, d_drag=1.0, inset_grasp=0.025, inset_release=0.03,
              low_rel=0.006,     # low_rel: release height relative to the grasp height (TCP z), overrides `low`
              d_post=0.5, back=0.04)   # still 0.5 s after opening; back off 4 cm along -z_tcp before rising


def fold_waypoints(mode: str, A, C, z_grasp: float, up: float = 0.10, low: float = 0.035, retreat: float = 0.12,
                   d_desc: float = 0.4, d_close: float = 0.5, d_lift: float = 1.0, d_carry: float = 0.8,
                   d_lower: float = 1.6, d_wait: float = 0.0, d_open: float = 0.6, d_retreat: float = 1.0,
                   d_hold: float = 0.5, d_drag: float = 1.0, d_post: float = 0.0, back_dir=None, back: float = 0.0,
                   **_):
    """waypoints (x, y, z, grip, duration of the segment that starts here, phase) with grasp point A and release point
    C (xy); a segment between two waypoints at the same position with different grip ramps the gripper.
    d_post: still time after the opening; back_dir / back: the retreat first backs the TCP off by `back` along the
    unit vector back_dir (the approach axis reversed, -z_tcp; E29b: a tilted gripper retracted straight up dragged the
    released corner along on PiPER), then rises to `retreat` (the E23 retreat: straight up)."""
    A, C = np.asarray(A, float), np.asarray(C, float)
    if mode not in SHAPES:
        raise ValueError(mode)
    mode = SHAPES[mode]
    g = 0 if mode == "no_grip" else 1
    w = [(*A, 0.08, 0, d_desc, C_), (*A, z_grasp, 0, d_close, C_)]                 # descend, close
    if mode == "drag_on_ground":
        w += [(*A, z_grasp, g, d_drag, C_),                                         # drag along the table to C
              (*C, z_grasp, g, d_open, C_),                                         # open
              (*C, z_grasp, 0, d_retreat, F), (*C, retreat, 0, d_hold, F), (*C, retreat, 0, 1e-3, F)]
        return w
    mid = (A + C) / 2
    w += [(*A, z_grasp, g, d_lift, C_)]                                             # lift
    if mode == "drop_midway":
        w += [(*A, up, g, d_carry / 2, C_), (*mid, up, g, d_open, C_),              # half the carry, open in mid-air
              (*mid, up, 0, d_carry / 2, F), (*C, up, 0, d_lower, F), (*C, low, 0, d_wait + d_open, F)]
    else:
        w += [(*A, up, g, d_carry, C_), (*C, up, g, d_lower, C_)]                   # carry, lower
        if mode == "hold_cloth":                                                    # never opens, stays
            w += [(*C, low, g, d_wait + d_open + d_retreat + d_hold, C_), (*C, low, g, 1e-3, C_)]
            return w
        if d_wait > 0:
            w += [(*C, low, g, d_wait, C_)]                                         # hold still before opening
        w += [(*C, low, g, d_open, C_)]                                             # open
    if d_post > 0:
        w += [(*C, low, 0, d_post, F)]
    if back > 0 and back_dir is not None:
        b = np.array([C[0], C[1], low]) + back * np.asarray(back_dir, float)
        w += [(*C, low, 0, d_retreat / 2, F), (b[0], b[1], max(b[2], low), 0, d_retreat / 2, F)]
        w += [(b[0], b[1], max(retreat, b[2]), 0, d_hold, F), (b[0], b[1], max(retreat, b[2]), 0, 1e-3, F)]
        return w
    w += [(*C, low, 0, d_retreat, F), (*C, retreat, 0, d_hold, F), (*C, retreat, 0, 1e-3, F)]
    return w


def phase_at(wps, t: float) -> int:
    """phase tag of the segment that contains time t (same segment logic as soft.expert.interp_waypoints)"""
    acc = 0.0
    for i, w in enumerate(wps):
        if t < acc + w[4] or i == len(wps) - 1:
            return int(w[5]) if len(w) > 5 else C_
        acc += w[4]
    return int(wps[-1][5]) if len(wps[-1]) > 5 else C_


def corner_points(u, side: str, inset_grasp: float = 0.025, inset_release: float = 0.03, world: int = 0):
    """(grasp xy, release xy) for the left ('L': FL -> NL) or right ('R': FR -> NR) arm, from the ACTUAL corner vertices
    of world 0 (E29b, for data engines that move the towel; equal to the nominal towel pose after a reset): grasp = the
    far corner moved inset_grasp inwards along both towel edges, release = the near corner moved inset_release inwards"""
    t = u.towel_spec
    cx, cy = t.center
    H = t.half_size
    sy = 1.0 if side == "L" else -1.0
    far = np.array([cx + H - inset_grasp, cy + sy * (H - inset_grasp)])
    near = np.array([cx - H + inset_release, cy + sy * (H - inset_release)])
    ci = getattr(u, "corner_idx", None)
    if getattr(u, "cloth", None) is None or ci is None:
        return far, near
    q = u.cloth.particle_q()[world].detach().cpu().numpy().astype(float)[:, :2]
    if not np.isfinite(q).all():
        return far, near
    P = {k: q[i] for k, i in ci.items()}
    ex = (P["FL"] + P["FR"] - P["NL"] - P["NR"]) / 2; ex /= np.linalg.norm(ex) + 1e-12        # near -> far
    ey = (P["FL"] + P["NL"] - P["FR"] - P["NR"]) / 2; ey /= np.linalg.norm(ey) + 1e-12        # right -> left
    if side == "L":
        return P["FL"] - inset_grasp * (ex + ey), P["NL"] + inset_release * (ex - ey)
    return P["FR"] - inset_grasp * (ex - ey), P["NR"] + inset_release * (ex + ey)


def _orientations(base_xy, grasp_xy):
    radial = np.array([grasp_xy[0] - base_xy[0], grasp_xy[1] - base_xy[1], 0.0])
    radial /= np.linalg.norm(radial) + 1e-12
    out = []
    for y in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
        d = np.cross(y, UP)
        if d @ radial < 0:
            d = -d
        for tilt in TILTS:
            out.append((tilt, frame(y, -UP * np.cos(tilt) + d * np.sin(tilt))))
    return out


def task_points(u, i: int, mode: str = "expert", kw: dict | None = None, world: int = 0):
    """(grasp xy, release xy) of arm i; envs may override with u.fold_points(i, EXPERT, mode) (e.g. the cover task);
    kw: the expert parameters in use (A2: insets)"""
    kw = EXPERT if kw is None else kw
    if hasattr(u, "fold_points"):
        return u.fold_points(i, kw, mode)
    return corner_points(u, "L" if i == 0 else "R", kw["inset_grasp"], kw["inset_release"], world=world)


def plan_arm(u, i: int, mode: str, dt: float, params: dict | None = None, world: int = 0):
    """(approach path (k, dof), fold path (n, dof), grip (n,), info) for arm i (0 = left, 1 = right); info['phase'] =
    per-step phase codes of the fold path. params (A2, imitator.planning.params 'DualFoldTowel-v1'): overrides of EXPERT
    (+ grasp_dz); None = the frozen values"""
    ag = u.agent.agents[i]
    if mode in IDLE_MODES or i in getattr(u, "idle_arms", ()):         # this arm does not take part: hold home, FREE
        idx = [k for k, j in enumerate(ag.robot.active_joints) if j.name in ag.record.arm_joints]
        q = ag.robot.get_qpos()[0].cpu().numpy()[idx][None].astype(np.float64)
        return q, q.copy(), np.zeros(1), dict(idle=True, n=1, phase=np.array([F], np.int8))
    base = u.dual.mount_poses()[i]
    planner = ArmPlanner(ag, base, control_dt=dt)
    planner.set_obstacle_points(arm_points(u.agent.agents[1 - i]))
    kw = dict(EXPERT)
    if (getattr(u, "imsim", None) or {}).get("task") == "DualFoldTowel-v1":
        from . import params as PR                          # A2: defaults (= EXPERT) <- adopted tuned set <- params
        kw.update(PR.pv(params, "DualFoldTowel-v1", u, mode))
    elif params:
        kw.update(params)
    A, C = task_points(u, i, mode, kw, world=world)
    ext = tcp_extent(ag)
    q_home = planner.qpos()
    why = []
    if hasattr(u, "fold_waypoint_overrides"):
        kw.update(u.fold_waypoint_overrides())
    for tilt, R in _orientations(base.p, A):
        zg = ext * np.cos(tilt) + 0.015 * np.sin(tilt) + 0.002 + float(kw.get("grasp_dz", 0.0))
        kw_t = dict(kw)
        if kw_t.get("low_rel") is not None:
            kw_t["low"] = zg + float(kw_t["low_rel"])
        kw_t["back_dir"] = -R[:, 2]                     # approach axis reversed (retreat along it first)
        wps = fold_waypoints(mode, A, C, zg, **kw_t)
        n = int(np.ceil(total_time(wps) / dt))
        tgt = np.array([interp_waypoints(wps, (k + 1) * dt) for k in range(n)])
        ph = np.array([phase_at(wps, (k + 1) * dt) for k in range(n)], np.int8)
        ok, sols = planner.ik(sapien.Pose(tgt[0, :3], mat2quat(R)), n_init=20)
        if not ok:
            why.append(f"tilt {tilt}: first waypoint IK"); continue
        best = None
        for s0 in sols[:6]:
            path, err = planner.track(tgt[:, :3], R, s0)
            if path is not None:
                best = (s0, path); break
            why.append(f"tilt {tilt}: {err}")
        if best is None:
            continue
        s0, path = best
        q_goal = np.asarray(s0)[planner.arm_idx]
        res = planner.plan_qpos(q_goal)
        if res["status"] != "Success":
            k = 30
            a = q_home[planner.arm_idx]
            appr = np.array([a + (q_goal - a) * (j + 1) / k for j in range(k)])
        else:
            appr = res["position"]
        return appr, path, tgt[:, 3], dict(tilt=tilt, R=R.tolist(), z_grasp=zg, A=A.tolist(), C=C.tolist(), n=n,
                                           phase=ph)
    raise RuntimeError(f"{u.dual.name} arm {i} {mode}: no IK-trackable fold path ({why[-4:]})")


def plan_batch(u, behaviours: list[str], dt: float | None = None, params: dict | None = None):
    """per env (behaviour) and arm: joint targets (T, N, dof) and gripper commands (T, N, gdim) for the whole policy
    phase; all behaviours padded to the same length (last target held). info['_phases'][i] = (T, N) phase codes of arm i
    (0 = FREE, 1 = CONTACT; the approach from home is FREE, the padding keeps the last code)."""
    dt = dt or float(u.control_timestep)
    plans = {}
    for b in sorted(set(behaviours)):
        arms = [plan_arm(u, i, b, dt, params) for i in (0, 1)]
        k = max(len(a[0]) for a in arms)
        out = []
        for i, (appr, path, grip, info) in enumerate(arms):
            pad = np.repeat(appr[-1:], k - len(appr), 0) if len(appr) < k else np.zeros((0, appr.shape[1]))
            q = np.vstack([appr, pad, path])
            g = np.concatenate([np.zeros(k), grip])
            ph = np.concatenate([np.full(k, F, np.int8), info["phase"]])
            out.append((q, g, info, ph))
        plans[b] = out
    T = max(len(p[0][0]) for p in plans.values())
    Q, G, P = [], [], []
    for i in (0, 1):
        ag = u.agent.agents[i]
        r = ag.record
        o, c = float(r.gripper_cmd_open[0]), float(u.cloth_close[i])
        qs, gs, ps = [], [], []
        for b in behaviours:
            q, g, _, ph = plans[b][i]
            if len(q) < T:
                q = np.vstack([q, np.repeat(q[-1:], T - len(q), 0)]); g = np.concatenate([g, np.repeat(g[-1:], T - len(g))])
                ph = np.concatenate([ph, np.repeat(ph[-1:], T - len(ph))])
            qs.append(q); gs.append((1 - g) * o + g * c); ps.append(ph)
        Q.append(np.stack(qs, 1).astype(np.float32)); G.append(np.stack(gs, 1)[..., None].astype(np.float32))
        P.append(np.stack(ps, 1))
    info = {b: [p[2] for p in plans[b]] for b in plans}
    for b in info:
        for a in info[b]:
            a.pop("phase", None)
    info["_phases"] = P
    return Q, G, info


def run(env, behaviour: str = "expert"):
    """drive one behaviour on every env of the batch (used by imitator.evaluate through imitator.soft_eval)"""
    from ..soft_eval import run_fold_batch
    return run_fold_batch(env, [behaviour] * env.unwrapped.num_envs)
