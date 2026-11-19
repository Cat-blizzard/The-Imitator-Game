"""Scripted (mplib) pick-and-place for any dual configuration, plus the counterexample variants.

pick_place(env, variant="release" | "hold" | "cage")
  The arm on the cube's side picks the cube and puts it in the target square:
    home -> pre-grasp (RRTConnect, the other arm as a point-cloud obstacle) -> grasp (screw) -> close -> lift (screw)
    -> above target (screw, RRT fallback) -> place (screw) -> [variant]
      release: open fully, retreat along the approach axis, back to home (RRT)
      hold:    the cube is pressed down onto the table (2 mm) and the gripper / hand stays closed on it
      cage:    open to ~1 mm beyond the cube width (parallel grippers; hands: synergy 0.65), then nothing
  Grasp orientation and waypoint heights come from a small search (cube-face yaw x approach tilt x height) that keeps
  every waypoint IK-feasible for the arm (small arms such as PiPER cannot point straight down above ~0.1 m).
  Parallel grippers: top-down grasp, closing axis normal to two cube faces, fingertips >= 4 mm above the table.
  Hands: palm-down power grasp with the record's fixed synergy (HandRecord.synergy), fingers along a cube-face normal.
no_op(env, n): both arms hold home.
The env passed in may be wrapped (e.g. by verify.smoke.make_recorder); actions go through env.step.
Phases (imitator.data.phase, read by the data engine; they never change the simulation): the acting arm is FREE for the
first step, the move to the pre-grasp pose and the retreat / return home after the release, CONTACT from the final
approach to the grasp until the gripper has opened (grasp, lift, carry, place, release; the whole hold / cage variants).
The idle arm is never annotated, so it counts as CONTACT (E26 D2: an arm is FREE only while it executes a free move).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import sapien
from transforms3d.quaternions import mat2quat, quat2mat

from ..data.phase import CONTACT, FREE, set_phase
from .mplib_planner import ArmPlanner, arm_points

UP = np.array([0.0, 0.0, 1.0])


class DualExecutor:
    def __init__(self, env):
        self.env = env
        self.u = env.unwrapped
        self.agents = list(self.u.agent.agents)
        self.keys = list(self.u.single_action_space.spaces.keys()) if hasattr(self.u, "single_action_space") else \
            list(self.u.action_space.spaces.keys())
        self.targets = []
        for ag in self.agents:
            a = ag.home_action().copy()
            a[:ag.record.dof_arm] = ag.robot.get_qpos()[0].cpu().numpy()[
                [i for i, j in enumerate(ag.robot.active_joints) if j.name in ag.record.arm_joints]]
            self.targets.append(a)
        self.n = 0
        self.log = []

    def action(self):
        return OrderedDict((k, t.copy()) for k, t in zip(self.keys, self.targets))

    def phase(self, i: int | None, phase: str):
        """annotate the stage that follows for arm i (None: all arms); see imitator.data.phase"""
        set_phase(self.env, phase, arm=i)

    def step(self, n: int = 1):
        for _ in range(n):
            self.env.step(self.action())
            self.n += 1

    def follow(self, i: int, positions: np.ndarray, settle: int = 3):
        nd = self.agents[i].record.dof_arm
        for q in positions:
            self.targets[i][:nd] = q
            self.step()
        self.step(settle)

    def set_gripper(self, i: int, closed: float, n: int = 20):
        ag = self.agents[i]
        self.targets[i][ag.record.dof_arm:] = ag.gripper_action(closed)
        self.step(n)

    def set_gripper_raw(self, i: int, g, n: int = 20):
        ag = self.agents[i]
        self.targets[i][ag.record.dof_arm:] = np.asarray(g, np.float32)
        self.step(n)


# ------------------------------------------------------------------------------------------------ geometry helpers
def tcp_extent(ag) -> float:
    """how far the grasp links reach beyond the TCP along its approach axis (z), gripper open"""
    from ..embodiments.urdf_tools import load_urdf
    r = ag.record
    m = load_urdf(r.urdf_path)
    Ts = m.fk(r.home_qpos())
    Ti = np.linalg.inv(Ts[r.tcp_link])
    z = []
    for l in r.grasp_links:
        P = m.collision_points(l)
        if len(P):
            T = Ti @ Ts[l]
            z.append(((T[:3, :3] @ P.T).T + T[:3, 3])[:, 2].max())
    return float(max(z)) if z else 0.0


def frame(y_axis, z_axis) -> np.ndarray:
    z = np.asarray(z_axis, float); z /= np.linalg.norm(z)
    y = np.asarray(y_axis, float); y = y - (y @ z) * z; y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


def pose(p, R) -> sapien.Pose:
    return sapien.Pose(np.asarray(p, float), mat2quat(R))


def cube_yaw(q) -> float:
    R = quat2mat(np.asarray(q, float))
    # the cube axis closest to horizontal-x defines the face yaw (mod pi/2)
    a = R[:, 0] if abs(R[2, 0]) < 0.7 else R[:, 1]
    return float(np.arctan2(a[1], a[0]))


def rot_angle(R1, R2) -> float:
    c = (np.trace(R1.T @ R2) - 1) / 2
    return float(np.arccos(np.clip(c, -1, 1)))


# ------------------------------------------------------------------------------------------------ plan search
HAND_GRASP = dict(along=0.0, below=0.0)          # palm_tcp offset from the cube centre (fingers dir, palm normal)
TILTS = (0.0, 0.3, 0.55)                          # approach tilts tried [rad] (away from the arm's base)


def grasp_candidates(ag, planner: ArmPlanner, cube_p, cube_q, target_p, half: float, base_p, p: dict | None = None):
    """yield (score, dict(R, grasp, pre, lift, above, place, retreat)) for IK-feasible candidates (best first);
    p: A2 expert parameters (h_scale, grasp_dz, hand_along, hand_below, place_dz, retreat_scale; None = frozen values)"""
    p = p or {}
    h_scale = p.get("h_scale", 1.0)
    grasp_dz = p.get("grasp_dz", 0.004)
    along, below = p.get("hand_along", HAND_GRASP["along"]), p.get("hand_below", HAND_GRASP["below"])
    place_dz = p.get("place_dz", 0.006)
    retreat_scale = p.get("retreat_scale", 1.0)
    r = ag.record
    psi = cube_yaw(cube_q)
    ext = tcp_extent(ag) if r.kind == "arm_gripper" else 0.0
    R_now = planner.fk_tcp(planner.qpos())[:3, :3]
    radial = np.array([cube_p[0] - base_p[0], cube_p[1] - base_p[1], 0.0]); radial /= np.linalg.norm(radial) + 1e-9
    cands = []
    for k in range(4):
        a = psi + k * np.pi / 2
        y = np.array([np.cos(a), np.sin(a), 0.0])
        d = np.cross(y, UP)                                  # horizontal, perpendicular to the closing / finger axis
        if d @ radial < 0:
            d = -d
        for tilt in TILTS:
            z = -UP * np.cos(tilt) + d * np.sin(tilt)
            R = frame(y, z)
            if r.kind == "arm_gripper":
                zc = max(half, ext * np.cos(tilt) + 0.015 * np.sin(tilt) + grasp_dz)
                g = np.array([cube_p[0], cube_p[1], zc])
            else:
                g = np.array(cube_p, float) + along * y + below * (-z)
            dz_place = g[2] - cube_p[2] + half + place_dz        # tcp height above table when the cube is 6 mm up
            for h in ((0.10, 0.07, 0.05) if h_scale == 1.0 else tuple(x * h_scale for x in (0.10, 0.07, 0.05))):
                hr = h if retreat_scale == 1.0 else h * retreat_scale
                wp = dict(R=R, grasp=g, pre=g - z * h, lift=g + UP * h,
                          above=np.array([target_p[0], target_p[1], dz_place + h]),
                          place=np.array([target_p[0], target_p[1], dz_place]),
                          retreat=np.array([target_p[0], target_p[1], dz_place]) + UP * hr, h=h, tilt=tilt, yaw=a)
                if r.kind == "arm_gripper":
                    score = tilt * 2.0 + (0.10 - h) * 5.0 + rot_angle(R_now, R) * 0.3
                else:      # hands: a tilted palm keeps the thumb off the table; low waypoints keep the arm compact
                    score = (0.55 - tilt) * 2.0 + abs(h - 0.05) * 5.0 + rot_angle(R_now, R) * 0.3
                cands.append((score, wp))
    cands.sort(key=lambda c: c[0])
    for score, wp in cands:
        ok = True
        q_seed = None
        for key in ("pre", "grasp", "lift", "above", "place", "retreat"):
            good, sol = planner.ik(pose(wp[key], wp["R"]), q_start=q_seed, n_init=10)
            if not good and q_seed is not None:                    # retry from the current configuration
                good, sol = planner.ik(pose(wp[key], wp["R"]), q_start=None, n_init=10)
            if not good:
                ok = False
                break
            q_seed = sol[0]
        if ok:
            yield score, wp


def _go(ex: DualExecutor, i: int, planner: ArmPlanner, target: sapien.Pose, method: str, attach: bool = False,
        fallback: bool = True):
    """method: 'rrt' (collision-aware), 'screw' (collision-aware straight line, RRT fallback), 'line' (cartesian
    straight line, no collision check; screw then RRT as fallback)"""
    if method == "line":
        res = planner.cartesian(target)
        if res["status"] != "Success" and fallback:
            res = planner.plan(target, method="screw", attach=attach)
    else:
        res = planner.plan(target, method=method, attach=attach)
    if res["status"] != "Success" and fallback and method in ("screw", "line"):
        res = planner.plan(target, method="rrt", attach=attach)
    if res["status"] != "Success" and fallback and method == "screw":
        res = planner.joint_interp(target)                    # IK + joint-space line (no path collision check)
    if res["status"] != "Success" and fallback and method == "screw":
        res = planner.cartesian(target)                       # last resort: straight line without collision check
    if res["status"] != "Success":
        return res["status"]
    ex.follow(i, res["position"])
    return "ok"


def pick_place(env, variant: str = "release", arm: int | None = None, max_candidates: int = 6, home_after: bool = True,
               verbose: bool = False, on_event=None, params: dict | None = None) -> dict:
    """run the scripted policy on the (reset) env; returns a log dict (status, arm, waypoints, steps).
    params (A2, imitator.planning.params 'DualCubePlace-v1:gripper' / ':hand' by the acting arm's kind): planner speed,
    waypoint heights, grasp offsets, close command / duration, place height and wait, in-hand correction limit, re-aim
    switch, release duration, retreat height; None = the frozen values (+ an adopted tuned set)."""
    from . import params as PR
    u = env.unwrapped
    ex = DualExecutor(env)
    ev = on_event or (lambda name: None)
    cube_p = u.cube.pose.p[0].cpu().numpy().astype(float)
    cube_q = u.cube.pose.q[0].cpu().numpy().astype(float)
    tgt = u.target_pos[0].cpu().numpy().astype(float)
    half = float(u.cube_half)
    i = arm if arm is not None else (0 if cube_p[1] > 0 else 1)
    other = 1 - i
    ag = ex.agents[i]
    key = "DualCubePlace-v1:hand" if ag.record.kind == "arm_hand" else "DualCubePlace-v1:gripper"
    p = PR.pv(params, key, env, variant)
    base = u.dual.mount_poses()[i]
    if p["vel"] == 0.9:
        planner = ArmPlanner(ag, base, control_dt=float(u.control_timestep))
    else:
        planner = ArmPlanner(ag, base, control_dt=float(u.control_timestep), vel=p["vel"], acc=p["vel"])
    planner.set_obstacle_points(arm_points(ex.agents[other]))
    log = dict(arm=i, variant=variant, status="no_candidate", tried=0)
    ex.phase(None, CONTACT)                                  # clear stale annotations: every arm CONTACT ...
    ex.phase(i, FREE)                                        # ... except the acting arm's transit to the pre-grasp pose
    ex.step(1)                                               # the policy always acts at least once (protocol v3 hold A)
    for n, (score, wp) in enumerate(grasp_candidates(ag, planner, cube_p, cube_q, tgt, half, base.p, p)):
        if n >= max_candidates:
            break
        log["tried"] = n + 1
        R = wp["R"]
        ex.phase(i, FREE)
        st = _go(ex, i, planner, pose(wp["pre"], R), "rrt")
        if st != "ok":
            log["status"] = f"pre: {st}"; continue
        ex.phase(i, CONTACT)                                 # final approach, grasp, carry, place, release
        if p["pre_wait"]:
            ex.step(p["pre_wait"])                           # A2: still at the pre-grasp pose before the final approach
        st = _go(ex, i, planner, pose(wp["grasp"], R), "line")
        if st != "ok":
            log["status"] = f"grasp: {st}"; continue          # next candidate (the planner starts from where the arm is)
        ex.set_gripper(i, p["close_cmd"], n=p["close_steps"])
        ev("grasped")
        # planning-only proxy of the grasped cube, shrunk so the fingers wrapped around it are not "in collision" with it
        shrink = 0.012 if ag.record.kind == "arm_gripper" else 0.024
        planner.attach_box([2 * half - shrink] * 3, sapien.Pose(np.linalg.inv(R) @ (cube_p - wp["grasp"])))
        st = _go(ex, i, planner, pose(wp["lift"], R), "line", attach=True)
        if st != "ok":
            log["status"] = f"lift: {st}"; break
        ev("lifted")
        # the cube may have shifted in the hand while closing: aim the PLACE so the cube (not the TCP) lands on the
        # target (privileged cube pose; the scripted expert is allowed to use it)
        d = u.cube.pose.p[0].cpu().numpy().astype(float) - ag.tcp.pose.p[0].cpu().numpy().astype(float)
        nominal = cube_p - wp["grasp"]
        corr = (d - nominal)[:2]
        log["cube_shift_in_tcp_m"] = (wp["R"].T @ (d - nominal)).round(4).tolist()   # where the grasp moved the cube
        if np.linalg.norm(corr) < p["corr_max"]:
            for k_ in ("above", "place", "retreat"):
                wp[k_] = wp[k_].copy(); wp[k_][:2] -= corr
            log["place_correction_m"] = float(np.linalg.norm(corr))
        st = _go(ex, i, planner, pose(wp["above"], R), "screw", attach=ag.record.kind == "arm_gripper")
        if st != "ok":
            log["status"] = f"transport: {st}"; break
        if PR.B(p["reaim"]):
            # A2 switch: re-measure where the cube hangs in the hand above the target and shift the place pose so that
            # the cube centre (not the TCP) is lowered onto the target centre
            c_now = u.cube.pose.p[0].cpu().numpy().astype(float)
            e_xy = (c_now - tgt)[:2]
            if np.linalg.norm(e_xy) < 0.08:
                for k_ in ("place", "retreat"):
                    wp[k_] = wp[k_].copy(); wp[k_][:2] -= e_xy
                log["reaim_m"] = float(np.linalg.norm(e_xy))
        st = _go(ex, i, planner, pose(wp["place"], R), "line", attach=True)
        if st != "ok":
            log["status"] = f"place: {st}"; break
        ex.step(p["place_wait"])
        ev("placed")
        log.update(grasped_at_place=bool(ag.is_grasping(u.cube)[0]), h=wp["h"], tilt=wp["tilt"])
        if variant == "release":
            ex.set_gripper(i, 0.0, n=p["release_steps"])
            ex.phase(i, FREE)                                # retreat and return home: nothing in hand
            st = _go(ex, i, planner, pose(wp["retreat"], R), "line")
            if st != "ok":
                log["status"] = f"retreat: {st}"; break
            if home_after:
                res = planner.plan_qpos(np.asarray(ag.record.home_arm_qpos, float))
                if res["status"] == "Success":
                    ex.follow(i, res["position"])
        elif variant == "hold":
            # set the cube down ON the table (2 mm press) and keep the gripper / hand closed: only the release rule
            # (contact + enclosure) can tell this apart from a placement
            down = pose(wp["place"] - UP * 0.008, R)
            res = planner.cartesian(down)
            if res["status"] == "Success":
                ex.follow(i, res["position"], settle=10)
        elif variant == "cage":
            cage(ex, i, half)
        ev("done")
        log["status"] = "ok"
        break
    log["steps"] = ex.n
    log["final_targets"] = [t.tolist() for t in ex.targets]
    return log


def cage(ex: DualExecutor, i: int, half: float, extra: float = 0.0012):
    """open just enough that the pads clear the cube by ~extra/2 per side (no contact, still enclosing)"""
    ag = ex.agents[i]
    r = ag.record
    if r.kind == "arm_hand":
        ex.set_gripper(i, 0.65, n=20)
        return
    # bisection on the command, using the measured opening (geometric) after a short settle
    lo, hi = 1.0, 0.0                                        # closed .. open (fraction "closed")
    want = 2 * half + extra
    for _ in range(7):
        mid = (lo + hi) / 2
        ex.set_gripper(i, mid, n=8)
        o = float(ag.measured_opening()[0])
        if o < want:
            lo = mid
        else:
            hi = mid
    ex.set_gripper(i, hi, n=15)


def lift_probe(env, arm: int, last_action=None, dz: float = 0.05) -> dict:
    """Ground-truth aftermath (AFTER the verdict): raise the acting arm's TCP by dz with the gripper / hand command
    unchanged and report how far the cube came along. held = cube rose >= 60 % of dz."""
    u = env.unwrapped
    ex = DualExecutor(env)
    ag = ex.agents[arm]
    if last_action is not None:
        for i, (k, a) in enumerate(zip(ex.keys, ex.agents)):
            la = np.asarray(last_action[k], np.float32).reshape(-1)
            ex.targets[i][a.record.dof_arm:] = la[a.record.dof_arm:]
    z0 = float(u.cube.pose.p[0, 2])
    planner = ArmPlanner(ag, u.dual.mount_poses()[arm], control_dt=float(u.control_timestep), table=False)
    T = ag.tcp.pose.to_transformation_matrix()[0].cpu().numpy()
    res = planner.cartesian(sapien.Pose(T[:3, 3] + UP * dz, mat2quat(T[:3, :3])), speed=0.05)
    if res["status"] != "Success":
        return dict(status=res["status"], cube_rise_m=None, held=None)
    ex.follow(arm, res["position"], settle=10)
    rise = float(u.cube.pose.p[0, 2]) - z0
    return dict(status="ok", cube_rise_m=rise, held=bool(rise >= 0.6 * dz))


def no_op(env, n: int = 20) -> dict:
    ex = DualExecutor(env)
    ex.step(n)
    return dict(status="ok", steps=ex.n, variant="no_op")
