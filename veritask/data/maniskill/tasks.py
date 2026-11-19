"""Task bindings for the data engine: probes (ported from research/scripts/e4_*.py), objects, stage plans (ported
from the E1 scripted experts / official Imitator solutions), progress graphs over the contract terms, weak-policy
noise hooks, task-internal state for snapshots, and a coarse collision world for cuRobo.

Plans are lists of Stage(name, fn, segment); fn(session, strict) -> bool runs one motion or gripper action.
Stage functions read targets from session.ctx when present (L1 resumes with the source's context) and recompute
them from the current privileged state otherwise (L2 / L3 start with an empty context).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..engine import ProgressGraph, ProgressNode
from .session import CLOSED, CONTACT_FORCE_N, OPEN, P, pose7

TASKS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tasks")   # veritask/tasks


@dataclass
class Stage:
    name: str
    fn: Callable
    segment: str = "main"


def _quat_mat(q):
    from scipy.spatial.transform import Rotation as R
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _rot(actor):
    return _quat_mat(actor.pose.q[0].cpu().numpy())


def _T(pose):
    T = np.eye(4); T[:3, :3] = _quat_mat(pose.q); T[:3, 3] = pose.p; return T


def _pose_from_T(T):
    import sapien
    from transforms3d.quaternions import mat2quat
    return sapien.Pose(T[:3, 3], mat2quat(T[:3, :3]))


def _last(sess, sig, default=0.0):
    tr = sess.traces.get(sig) or []
    return tr[-1] if tr else default


class _TaskBase:
    task_id = env_id = ""
    n_arms = 1
    movable: list = []
    target_names: list = []
    log_objects: list = []
    plans: dict = {}
    graph: ProgressGraph = None

    def make_env(self, render: bool = False):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  (registers tasks)
        kw = {}
        if render:  # low-res frames from the human render camera on the CPU Vulkan device (lavapipe)
            from mani_skill.envs.utils.system import backend as _be
            _orig = _be.parse_backend_device_id   # mani_skill 3.0.1 splits "pci:0"; keep it intact (as in E1)
            _be.parse_backend_device_id = lambda b: (b, None) if isinstance(b, str) and b.startswith("pci:") else _orig(b)
            kw = dict(human_render_camera_configs=dict(width=128, height=128))
        return gym.make(self.env_id, obs_mode="state", control_mode="pd_joint_pos", render_mode="rgb_array" if render else None,
                        sim_backend="cpu", render_backend="pci:0" if render else "none", num_envs=1, **kw)

    def actor(self, sess, name):
        return getattr(sess.u, name)

    def object_pose7(self, sess, name):
        return self.actor(sess, name).pose.raw_pose[0].cpu().numpy().astype(np.float32)

    def pre_reset(self, sess):
        return None

    def on_reset(self, sess, pre):
        for i in range(sess.n_arms):
            sess.task_state[f"init_tcp_{i}"] = pose7(sess.tcp(i))

    def get_task_state(self, sess):
        return dict(sess.task_state)

    def set_task_state(self, sess, ts):
        sess.task_state = dict(ts)

    def after_offset(self, sess, obj):
        pass

    def free_targets(self, sess):
        return list(self.target_names)

    def active_arms(self, stage):
        return [0]

    def held_object(self, sess, arm):
        ag = sess.agents[arm]
        for o in self.movable:
            if bool(ag.is_grasping(self.actor(sess, o))[0]):
                return o
        return None

    def world_cuboids(self, sess) -> list[dict]:
        """Coarse static collision world for cuRobo (world frame): name, dims [m], pose [x y z qw qx qy qz]."""
        return []


# =================================================================================================== PokeCube
class PokeCube(_TaskBase):
    """ManiSkill 3.0.1 PokeCube-v1 (venv msup). Expert = E1 peg_poke (grasp peg 3 cm from its tail, lift 3 mm,
    line the peg head up 3 cm behind the cube, push the cube centre onto the goal centre, hold), generalised to any
    peg yaw and push direction so it can also serve as the recovery skill."""
    task_id, env_id, n_arms = "poke_cube_v1", "PokeCube-v1", 1
    rrt_fallback = True  # E1 peg_poke: screw, then RRTConnect if screw fails
    contract_file = f"{TASKS}/poke_cube_v1.json"
    movable = ["cube", "peg"]
    target_names = ["goal_region"]
    log_objects = ["cube", "peg", "goal_region"]

    def probes(self, C):
        S = {}

        def rc(u):
            S["rc"] = C.force_sum(list(u.agent.robot.links), u.cube); return S["rc"]

        def pc(u):
            S["pc"] = C.force_sum([u.peg], u.cube); return S["pc"]
        return {
            "n_contacts": C.refresh,  # first: caches this step's contact list
            "cube_goal_xy": lambda u: float(np.linalg.norm((u.cube.pose.p[0, :2] - u.goal_region.pose.p[0, :2]).cpu().numpy())),
            "robot_cube_force_N": rc,
            "robot_cube_contact": lambda u: float(S["rc"] > CONTACT_FORCE_N),
            "peg_cube_force_N": pc,
            "peg_cube_contact": lambda u: float(S["pc"] > CONTACT_FORCE_N),
        }

    def aux(self, sess):
        return dict(peg_grasped=bool(sess.u.agent.is_grasping(sess.u.peg)[0]))

    def held_object(self, sess, arm):
        return "peg" if bool(sess.u.agent.is_grasping(sess.u.peg)[0]) else None

    def world_cuboids(self, sess):
        # table top at z = 0 in front of the robot (base at x = -0.615 is left out so link0 does not start in collision)
        return [dict(name="table", dims=[1.2, 1.6, 0.1], pose=[0.05, 0.0, -0.05, 1, 0, 0, 0])]


def _peg_axis(u):
    a = _rot(u.peg)[:, 0].copy(); a[2] = 0.0; return a / np.linalg.norm(a)


def _tcp_for_peg(s, center_xy, z, yaw):
    """TCP pose that puts the grasped peg's centre at (center_xy, z) with its axis at world yaw `yaw`, using the
    current in-hand offset (rigid grasp assumed)."""
    u = s.u; T_tcp = _T(s.tcp(0)); T_peg = _T(u.peg.pose.sp)
    T_off = np.linalg.inv(T_tcp) @ T_peg
    a = _peg_axis(u); dpsi = yaw - np.arctan2(a[1], a[0])
    Rz = np.array([[np.cos(dpsi), -np.sin(dpsi), 0], [np.sin(dpsi), np.cos(dpsi), 0], [0, 0, 1]])
    T_des = np.eye(4); T_des[:3, :3] = Rz @ T_peg[:3, :3]; T_des[:3, 3] = [center_xy[0], center_xy[1], z]
    return _pose_from_T(T_des @ np.linalg.inv(T_off))


def _push_dir(s):
    u = s.u; d = (u.goal_region.pose.p[0, :2] - u.cube.pose.p[0, :2]).cpu().numpy()
    n = np.linalg.norm(d)
    return np.array([1.0, 0.0]) if n < 0.01 else d / n


def _pk_grasp_pose(s):
    import sapien  # noqa: F401
    if "grasp" in s.ctx:
        return P(s.ctx["grasp"])
    u = s.u; a = _peg_axis(u)
    c = s.jitter(np.asarray(u.peg.pose.sp.p) - 0.09 * a)  # 3 cm from the tail
    closing = np.cross([0, 0, 1.0], a); closing /= np.linalg.norm(closing)
    g = u.agent.build_grasp_pose(np.array([0, 0, -1.0]), closing, c)
    s.ctx["grasp"] = pose7(g)
    return g


def pk_open(s, strict):
    s.set_gripper(0, OPEN); return True


def pk_pregrasp(s, strict):
    import sapien
    g = _pk_grasp_pose(s)
    ok = s.move_to(0, g * sapien.Pose([0, 0, -0.08]), strict)
    if s.fault("early"):
        s.set_gripper(0, CLOSED)  # weak: closes before descending onto the peg
    return ok


def pk_grasp(s, strict):
    return s.move_to(0, _pk_grasp_pose(s), strict, linear=True)


def pk_close(s, strict):
    import sapien
    if s.fault("late"):  # weak: starts lifting before closing
        t = s.tcp(0); s.move_to(0, sapien.Pose(np.asarray(t.p) + [0, 0, 0.03], t.q))
    s.set_gripper(0, CLOSED); return True


def pk_lift(s, strict):
    import sapien
    g = P(s.ctx["grasp"]) if "grasp" in s.ctx else s.tcp(0)
    return s.move_to(0, sapien.Pose(np.asarray(g.p) + [0, 0, 0.003], g.q), strict, linear=True)


def _pk_align(s, strict, up):
    u = s.u; d = _push_dir(s); s.ctx["d"] = d.tolist()
    cube = u.cube.pose.p[0, :2].cpu().numpy()
    head = cube - d * (u.cube_half_size + 0.03)
    centre = s.jitter(np.r_[head - d * u.peg_half_length, 0.0], z_scale=0.0)[:2] if up == 0 else head - d * u.peg_half_length
    pose = _tcp_for_peg(s, centre, u.peg_half_width + 0.003 + up, np.arctan2(d[1], d[0]))
    ok = s.move_to(0, pose, strict, linear=(up == 0))
    if up == 0 and s.fault("early"):
        s.set_gripper(0, OPEN)  # weak: lets go of the tool before poking
    return ok


def pk_align_up(s, strict):
    return _pk_align(s, strict, 0.05)


def pk_align(s, strict):
    return _pk_align(s, strict, 0.0)


def pk_push(s, strict, overshoot=0.0):
    u = s.u; d = np.asarray(s.ctx["d"]) if "d" in s.ctx else _push_dir(s)
    goal = u.goal_region.pose.p[0, :2].cpu().numpy()
    centre = s.jitter(np.r_[goal - d * (u.cube_half_size + u.peg_half_length - overshoot), 0.0], z_scale=0.0)[:2]
    return s.move_to(0, _tcp_for_peg(s, centre, u.peg_half_width + 0.003, np.arctan2(d[1], d[0])), strict, linear=True)


def pk_push_recover(s, strict):
    """recovery push ends 6 mm past the goal centre so the peg head really touches the cube even when the cube
    already sits on the goal (cube shoved onto the goal without a poke)."""
    return pk_push(s, strict, overshoot=0.006)


def pk_hold(s, strict):
    s.hold(20); return True


def pk_maybe_shortcut(s, strict):
    """weak policy 'shortcut' fault (prob noise['p_shortcut']): ignore the tool and push the cube onto the goal with
    the closed, 45-deg-tilted gripper (the E1 gripper_push counterexample), then abandon the plan."""
    import sapien
    from transforms3d.euler import euler2quat
    from transforms3d.quaternions import qmult
    if not s.fault("shortcut"):
        return True
    u = s.u; cube = u.cube.pose.p[0].cpu().numpy(); goal = u.goal_region.pose.p[0].cpu().numpy()
    q = qmult(euler2quat(0, np.deg2rad(-45.0), 0), s.tcp(0).q); zc = 0.022
    s.set_gripper(0, CLOSED)
    s.move_to(0, sapien.Pose([cube[0] - 0.07, cube[1], 0.08], q))
    s.move_to(0, sapien.Pose([cube[0] - 0.07, cube[1], zc], q))
    x_goal = goal[0] - u.cube_half_size - 0.012
    for dx in np.arange(0.0, 0.12, 0.01):  # farthest reachable push target on the line (as E1)
        path = s.plan(0, sapien.Pose([x_goal - dx, goal[1], zc], q))
        if path is not None and len(path) <= 120:
            s.follow(0, path); break
    t = np.asarray(s.tcp(0).p)
    s.move_to(0, sapien.Pose(t + [-0.03, 0, 0], q)); s.move_to(0, sapien.Pose(t + [-0.03, 0, 0.04], q))
    s.ctx["abandon_plan"] = True
    return True


PokeCube.plans = {
    "weak_shortcut": [Stage("p_maybe_shortcut", pk_maybe_shortcut), Stage("p_open", pk_open), Stage("p_pregrasp", pk_pregrasp),
                      Stage("p_grasp", pk_grasp), Stage("p_close", pk_close), Stage("p_lift", pk_lift), Stage("p_align", pk_align),
                      Stage("p_push", pk_push), Stage("p_hold", pk_hold)],
    "expert": [Stage("p_open", pk_open), Stage("p_pregrasp", pk_pregrasp), Stage("p_grasp", pk_grasp), Stage("p_close", pk_close),
               Stage("p_lift", pk_lift), Stage("p_align", pk_align), Stage("p_push", pk_push), Stage("p_hold", pk_hold)],
    "recover": [Stage("p_open", pk_open), Stage("p_pregrasp", pk_pregrasp), Stage("p_grasp", pk_grasp), Stage("p_close", pk_close),
                Stage("p_lift", pk_lift), Stage("p_align_up", pk_align_up), Stage("p_align", pk_align), Stage("p_push", pk_push_recover),
                Stage("p_hold", pk_hold)],
}
# progress graph: contract terms (cube at goal, poked by the peg; robot-cube contact is a latch_false invariant ->
# abstain) + one auxiliary predicate (peg in hand) that decides between "regrasp the tool" and "re-poke"
PokeCube.graph = ProgressGraph([
    ProgressNode("peg_grasped", "p_open", aux="peg_grasped"),
    ProgressNode("cube_at_goal", "p_align_up", term_key="trace.final:cube_goal_xy"),
    ProgressNode("peg_poked", "p_align_up", term_key="trace.count_true:peg_cube_contact"),
])


# =================================================================================================== PlaceCupPlate
class PlaceCupPlate(_TaskBase):
    """Imitator TwoRobotPlaceCupPlate-v1 (venv ms2). Expert = official _036 solution steps (E1 re-implementation):
    left arm places cup_1, right arm places cup_2 on the (static) tray."""
    task_id, env_id, n_arms = "place_cup_plate_v1", "TwoRobotPlaceCupPlate-v1", 2
    contract_file = f"{TASKS}/place_cup_plate_v1.json"
    movable = ["cup_1", "cup_2"]
    target_names = ["tray"]
    log_objects = ["cup_1", "cup_2", "tray"]

    def make_env(self, render: bool = False):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        kw = dict(human_render_camera_configs=dict(width=128, height=128)) if render else {}
        return gym.make(self.env_id, obs_mode="state", control_mode="pd_joint_pos", render_mode="rgb_array" if render else None,
                        render_backend="pci:0", sim_backend="cpu", num_envs=1, **kw)

    def probes(self, C):
        from ...outcomes.metrics import measure
        S, CACHE = {}, {}

        def geom(u):
            key = id(u.tray._objs[0])
            if CACHE.get("key") != key:
                b = u.tray.get_first_collision_mesh(to_world_frame=False).bounds
                corners = np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]])
                h = [float(c.get_first_collision_mesh(to_world_frame=False).bounds[1, 1]) for c in (u.cup_1, u.cup_2)]
                CACHE.update(key=key, corners=corners, h=h)
            return CACHE

        def tilt(a):
            return measure("rigid.tilt_deg", up_axis_world=_rot(a)[:, 1])  # model up = local +y

        def on_tray(u, i):
            g = geom(u); cup = (u.cup_1, u.cup_2)[i - 1]
            tp = u.tray.pose.p[0].cpu().numpy(); cw = tp + (_rot(u.tray) @ g["corners"].T).T
            lo, hi = cw.min(0), cw.max(0)
            c = cup.pose.p[0].cpu().numpy() + _rot(cup) @ np.array([0, g["h"][i - 1] / 2, 0])
            inside = bool(lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1])
            f = C.force_sum([cup], u.tray)
            return float(inside and f > CONTACT_FORCE_N)

        def released(u):
            g = False
            for ag in (u.left_agent, u.right_agent):
                for cup in (u.cup_1, u.cup_2):
                    g |= bool(ag.is_grasping(cup)[0])
            return float(not g)
        return {
            "n_contacts": C.refresh,
            "cup1_tilt_deg": lambda u: tilt(u.cup_1),
            "cup2_tilt_deg": lambda u: tilt(u.cup_2),
            "cup1_on_tray": lambda u: on_tray(u, 1),
            "cup2_on_tray": lambda u: on_tray(u, 2),
            "cup_max_speed": lambda u: float(max(np.linalg.norm(u.cup_1.linear_velocity[0].cpu().numpy()),
                                                 np.linalg.norm(u.cup_2.linear_velocity[0].cpu().numpy()))),
            "cup1_tray_xy": lambda u: float(np.linalg.norm((u.cup_1.pose.p[0, :2] - u.tray.pose.p[0, :2]).cpu().numpy())),
            "cup2_tray_xy": lambda u: float(np.linalg.norm((u.cup_2.pose.p[0, :2] - u.tray.pose.p[0, :2]).cpu().numpy())),
            "released": released,
        }

    def on_reset(self, sess, pre):
        super().on_reset(sess, pre)
        sess.task_state["tray"] = pose7(sess.u.tray.pose.sp)  # static actor: not part of get_state_dict()

    def set_task_state(self, sess, ts):
        sess.task_state = dict(ts)
        if "tray" in ts:
            sess.u.tray.set_pose(P(ts["tray"]))

    def after_offset(self, sess, obj):
        if obj == "tray":
            sess.task_state["tray"] = pose7(sess.u.tray.pose.sp)

    def free_targets(self, sess):
        busy = any(_last(sess, f"cup{i}_on_tray") > 0 or _last(sess, f"cup{i}_tray_xy", 1.0) < 0.13 for i in (1, 2))
        return [] if busy else ["tray"]

    def active_arms(self, stage):
        return [0] if stage.startswith("c1") else [1] if stage.startswith("c2") else [0, 1]

    def held_object(self, sess, arm):
        ag = sess.agents[arm]
        for o in ("cup_1", "cup_2"):
            if bool(ag.is_grasping(getattr(sess.u, o))[0]):
                return o
        return None

    def aux(self, sess):
        u = sess.u; out = {}
        for i, cup in ((1, u.cup_1), (2, u.cup_2)):
            held = bool(sess.agents[i - 1].is_grasping(cup)[0])
            anyheld = any(bool(ag.is_grasping(cup)[0]) for ag in sess.agents)
            out[f"cup{i}_held"] = held
            # skill precondition (not a contract term): a cup leaning < 45 deg can still be regrasped top-down and is
            # re-seated by the grasp; a cup lying on its side needs a righting skill we do not have
            out[f"cup{i}_held_or_graspable"] = held or _last(sess, f"cup{i}_tilt_deg", 0.0) < 45.0
            out[f"cup{i}_held_or_on_tray"] = held or _last(sess, f"cup{i}_on_tray") > 0
            out[f"cup{i}_released"] = not anyheld
        out["released"] = out["cup1_released"] and out["cup2_released"]
        return out

    def world_cuboids(self, sess):
        t = sess.task_state.get("tray")
        cubs = [dict(name="table", dims=[1.0, 1.5, 0.1], pose=[0.0, 0.0, -0.05, 1, 0, 0, 0])]
        if t:
            cubs.append(dict(name="tray", dims=[0.22, 0.30, 0.03], pose=[t[0], t[1], 0.015, 1, 0, 0, 0]))
        return cubs


def _cup_stages(i):
    import sapien
    arm = i - 1
    dy, d_grasp, d_place = (0.07, -0.08, -0.06) if i == 1 else (-0.07, -0.04, -0.02)
    seg = f"cup{i}"

    def cup(s):
        return s.u.cup_1 if i == 1 else s.u.cup_2

    def grasp(s):
        if f"g{i}" in s.ctx:
            return P(s.ctx[f"g{i}"])
        appr = np.array([0, 0, -1.0])
        closing = _quat_mat(s.task_state["init_tcp_1"][3:])[:, 1]  # official: right TCP y-axis (at its initial pose)
        closing = closing - (closing @ appr) * appr; closing /= np.linalg.norm(closing)
        g = s.agents[arm].build_grasp_pose(appr, closing, s.jitter(cup(s).pose.sp.p))
        s.ctx[f"g{i}"] = pose7(g); return g

    def goal(s, g):
        if f"goal{i}" not in s.ctx:
            s.ctx[f"goal{i}"] = pose7(sapien.Pose(s.u.tray.pose.sp.p, g.q) * sapien.Pose([0, 0, -0.05]))
        return P(s.ctx[f"goal{i}"])

    def reach3(s):
        if f"r3{i}" not in s.ctx:
            s.ctx[f"r3{i}"] = pose7(goal(s, grasp(s)) * sapien.Pose([0, dy, -0.2]))
        return P(s.ctx[f"r3{i}"])

    def c_open(s, strict):
        if s.grip[arm] != OPEN:
            s.set_gripper(arm, OPEN)
        return True

    def c_approach(s, strict):
        ok = s.move_to(arm, grasp(s) * sapien.Pose([0, 0, -0.15]), strict)
        if s.fault("early"):
            s.set_gripper(arm, CLOSED)  # weak: closes before descending
        return ok

    def c_descend(s, strict):
        return s.move_to(arm, grasp(s) * sapien.Pose([0, 0, d_grasp]), strict, linear=True)

    def c_close(s, strict):
        if s.fault("late"):  # weak: rises 6 cm before closing
            s.move_to(arm, grasp(s) * sapien.Pose([0, 0, d_grasp - 0.06]))
        s.set_gripper(arm, CLOSED); return True

    def c_lift(s, strict):
        return s.move_to(arm, grasp(s) * sapien.Pose([0, 0, -0.2]), strict, linear=True)

    def c_transport(s, strict):
        ok = s.move_to(arm, reach3(s), strict)
        if s.fault("early"):
            s.set_gripper(arm, OPEN)  # weak: opens 20 cm above the tray
        return ok

    def c_lower(s, strict):
        tgt = goal(s, grasp(s)) * sapien.Pose([0, dy, d_place])
        return s.move_to(arm, sapien.Pose(s.jitter(tgt.p, scale=1.5, z_scale=0.0), tgt.q), strict, linear=True)

    def c_release(s, strict):
        if s.fault("late"):
            s.move_to(arm, reach3(s))  # weak: retreats before opening (cup lifted, then dropped)
        s.set_gripper(arm, OPEN); return True

    def c_retreat(s, strict):
        return s.move_to(arm, reach3(s), strict)

    def c_home(s, strict):
        return s.move_to(arm, P(s.task_state[f"init_tcp_{arm}"]), strict)

    return [Stage(f"c{i}_{n}", f, seg) for n, f in (("open", c_open), ("approach", c_approach), ("descend", c_descend),
                                                     ("close", c_close), ("lift", c_lift), ("transport", c_transport),
                                                     ("lower", c_lower), ("release", c_release), ("retreat", c_retreat),
                                                     ("home", c_home))]


def _f_hold(s, strict):
    s.hold(30); return True


PlaceCupPlate.plans = {"expert": _cup_stages(1) + _cup_stages(2) + [Stage("f_hold", _f_hold, "final")]}
PlaceCupPlate.plans["recover"] = PlaceCupPlate.plans["expert"]
PlaceCupPlate.graph = ProgressGraph(
    [n for i in (1, 2) for n in (
        ProgressNode(f"cup{i}_upright", "no_skill_righting", term_key=f"trace.final:cup{i}_tilt_deg", or_aux=f"cup{i}_held_or_graspable"),
        ProgressNode(f"cup{i}_held_or_on_tray", f"c{i}_open", aux=f"cup{i}_held_or_on_tray"),
        ProgressNode(f"cup{i}_on_tray", f"c{i}_transport", term_key=f"trace.final:cup{i}_on_tray"),
        ProgressNode(f"cup{i}_released", f"c{i}_release", aux=f"cup{i}_released"))],
    goal_aux=["released"])


# =================================================================================================== PressJuicer
class PressJuicer(_TaskBase):
    """Imitator TwoRobotPressJuicer-v1 (venv ms2). Two plans:
    'official' = _050 solution (close, waypoint1, waypoint2, squeeze, back, home, open): never presses the knob (E1/E4)
    'press'    = E1/E4 press_knob (closed gripper tilted 45 deg, fingertips on the knob edge, 1.5 cm below its top,
                 hold, back off): the behaviour the contract accepts.
    The official waypoints are fixed at reset (juicer offset applied once); they do not follow later juicer moves."""
    task_id, env_id, n_arms = "press_juicer_v1", "TwoRobotPressJuicer-v1", 2
    contract_file = f"{TASKS}/press_juicer_v1.json"
    movable: list = []
    target_names = ["juicer"]
    log_objects = ["juicer", "knob"]

    make_env = PlaceCupPlate.make_env

    def actor(self, sess, name):
        return sess.u.partnet_link if name == "knob" else getattr(sess.u, name)

    def object_pose7(self, sess, name):
        return self.actor(sess, name).pose.raw_pose[0].cpu().numpy().astype(np.float32)

    def probes(self, C):
        S = {}

        def robot_links(u):
            return list(u.left_agent.robot.links) + list(u.right_agent.robot.links)

        def kf(u):
            S["kf"] = C.force_sum(robot_links(u), u.partnet_link); return S["kf"]
        return {
            "n_contacts": C.refresh,
            "knob_qpos": lambda u: float(u.partnet_link.joint.qpos.flatten()[0]),
            "robot_knob_force_N": kf,
            "robot_knob_contact": lambda u: float(S["kf"] > CONTACT_FORCE_N),
            "released": lambda u: float(S["kf"] <= CONTACT_FORCE_N),
            "tcp_lid_dist": lambda u: float(np.linalg.norm((u.left_agent.tcp.pose.p - u.partnet_link_positions())[0].cpu().numpy())),
        }

    def on_reset(self, sess, pre):
        super().on_reset(sess, pre)
        u = sess.u  # the env reconfigures on every reset (random juicer model), so reset-time waypoints are fresh
        sess.task_state.update(wp1=pose7(u.waypoint1), wp2=pose7(u.waypoint2), ever_pressed=False, juicer=pose7(u.juicer.pose.sp))

    def get_task_state(self, sess):
        ts = dict(sess.task_state); ts["ever_pressed"] = bool(sess.u.juicer_ever_pressed[0]); return ts

    def set_task_state(self, sess, ts):
        sess.task_state = dict(ts)
        sess.u.juicer_ever_pressed[:] = bool(ts.get("ever_pressed", False))

    def after_offset(self, sess, obj):
        sess.task_state["juicer"] = pose7(sess.u.juicer.pose.sp)

    def free_targets(self, sess):
        return [] if _last(sess, "robot_knob_contact") > 0 else ["juicer"]

    def held_object(self, sess, arm):
        return None

    def aux(self, sess):
        return dict(released_now=_last(sess, "released", 1.0) > 0)

    def world_cuboids(self, sess):
        return [dict(name="table", dims=[1.0, 1.5, 0.1], pose=[0.0, 0.0, -0.05, 1, 0, 0, 0])]


def _o_move(key):
    def f(s, strict):
        p = P(s.task_state[key] if key != "home" else s.task_state["init_tcp_0"])
        import sapien
        return s.move_to(0, sapien.Pose(s.jitter(p.p) if key != "home" else p.p, p.q), strict)
    return f


def _o_close(s, strict):
    s.set_gripper(0, CLOSED); return True


def _o_squeeze(s, strict):
    for _ in range(4):
        s.set_gripper(0, CLOSED)
    return True


def _o_open(s, strict):
    s.set_gripper(0, OPEN); s.set_gripper(1, OPEN); return True


def _lever(s):
    u = s.u
    m = u.partnet_links_meshes[0][0].copy()
    m.apply_transform(u.partnet_link.pose.to_transformation_matrix()[0].cpu().numpy())
    return m.bounds


def _press_ctx(s):
    from transforms3d.euler import euler2quat
    from transforms3d.quaternions import qmult
    if "press" not in s.ctx:
        bmin, bmax = _lever(s)
        s.ctx["press"] = [float(bmin[0]) - 0.005, float(bmax[1]) - 0.008, float(bmax[2])]
        s.ctx["qt"] = list(qmult(euler2quat(0, np.deg2rad(-45), 0), s.task_state["init_tcp_0"][3:]))
    return s.ctx["press"], np.asarray(s.ctx["qt"])


def _k_pre(s, strict):
    import sapien
    (px, py, top), qt = _press_ctx(s)
    return s.move_to(0, sapien.Pose([px - 0.03, py, top + 0.05], qt), strict)


def _k_touch(s, strict):
    import sapien
    (px, py, top), qt = _press_ctx(s)
    xy = s.jitter([px, py, 0.0], scale=0.6, z_scale=0.0)[:2]
    s.ctx["press_xy"] = [float(xy[0]), float(xy[1])]
    return s.move_to(0, sapien.Pose([xy[0], xy[1], top + 0.01], qt), strict, linear=True)


def _k_press(s, strict):
    import sapien
    (px, py, top), qt = _press_ctx(s)
    x, y = s.ctx.get("press_xy", [px, py])
    if s.fault("early"):
        return True  # weak: backs off without pressing
    depth = 0.015
    if s.noise:
        depth = float(np.clip(0.015 + s.rng.normal(0, 0.008), 0.0, 0.025))
        s.faults.append(dict(kind="press_depth", stage=s.stage, step=s.step_count, depth=round(depth, 4)))
    return s.move_to(0, sapien.Pose([x, y, top - depth], qt), strict, linear=True)


def _k_hold(s, strict):
    s.hold(10); return True


def _k_back(s, strict):
    return _k_pre(s, strict)


def _k_rest(s, strict):
    s.hold(10); return True


PressJuicer.plans = {
    "official": [Stage("o_close", _o_close), Stage("o_wp1", _o_move("wp1")), Stage("o_wp2", _o_move("wp2")),
                 Stage("o_squeeze", _o_squeeze), Stage("o_wp2b", _o_move("wp2")), Stage("o_wp1b", _o_move("wp1")),
                 Stage("o_home", _o_move("home")), Stage("o_open", _o_open)],
    "press": [Stage("k_close", _o_close), Stage("k_pre", _k_pre), Stage("k_touch", _k_touch), Stage("k_press", _k_press),
              Stage("k_hold", _k_hold), Stage("k_back", _k_back), Stage("k_rest", _k_rest)],
}
PressJuicer.plans["recover"] = PressJuicer.plans["press"]
PressJuicer.plans["expert"] = PressJuicer.plans["press"]
PressJuicer.graph = ProgressGraph([
    ProgressNode("knob_pressed", "k_close", term_key="trace.min:knob_qpos"),
    ProgressNode("robot_pressed", "k_close", term_key="trace.count_true:robot_knob_contact"),
    ProgressNode("released", "k_back", aux="released_now"),
], goal_aux=["released_now"])

TASK_CLASSES = {c.task_id: c for c in (PokeCube, PlaceCupPlate, PressJuicer)}
