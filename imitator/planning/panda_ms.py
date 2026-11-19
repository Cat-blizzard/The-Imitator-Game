"""Scripted experts and designed counterexample behaviours for the ManiSkill 3.0.1 Panda tasks (E29b), with phase
annotation (imitator.data.phase: FREE = free-space transit with nothing grasped, CONTACT = final approach, grasp, carry,
place, push, release ...).

The experts follow ManiSkill's own motion-planning solutions (mani_skill/examples/motionplanning/panda/solutions) with
small changes that are noted per task (lift before carrying, a retreat after the release, a pre-grasp search); the
counterexamples are variations of the same scripts (names = the contract's counterexamples). Every behaviour is
f(env, variant=...) -> dict(status=...), where env is the (recorded) env, already reset by the harness.
"""
from __future__ import annotations

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from ..data.phase import CONTACT, FREE, set_phase

FINGER_LENGTH = 0.025


class PandaExec:
    """ManiSkill's PandaArmMotionPlanningSolver driven through the recorded env, with phases and a few helpers"""

    def __init__(self, env, vel: float = 0.9, acc: float = 0.9):
        from .ms_planner.panda import PandaArmMotionPlanningSolver
        self.env, self.u = env, env.unwrapped
        self.pl = PandaArmMotionPlanningSolver(env, debug=False, vis=False, base_pose=self.u.agent.robot.pose,
                                               visualize_target_grasp_pose=False, print_env_info=False,
                                               joint_vel_limits=vel, joint_acc_limits=acc)
        self.steps = 0
        self.failed = []

    # ---------------------------------------------------------------- phases / motion
    def phase(self, p):
        set_phase(self.env, p)

    def _at(self, pose: sapien.Pose, tol_m: float = 0.002, tol_deg: float = 2.0) -> bool:
        t = self.tcp()
        c = abs(float(np.dot(np.asarray(t.q, float), np.asarray(pose.q, float))))
        return bool(np.linalg.norm(t.p - pose.p) < tol_m and 2 * np.degrees(np.arccos(min(1.0, c))) < tol_deg)

    def _plan(self, fn, pose: sapien.Pose, **kw):
        """A2 expert revision R2 (2026-11-17): a degenerate (zero-length) plan raises inside ManiSkill / mplib ('obs'
        unbound in follow_path for an empty path; 'Fail to parameterize path' in TOPP) and crashed the episode; now: 0
        (nothing to do) if the TCP is already at the target (2 mm / 2 deg), else -1 (planning failure)"""
        try:
            return fn(pose, **kw)
        except (UnboundLocalError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "parameterize" not in str(e):
                raise
            return 0 if self._at(pose) else -1

    def move(self, pose: sapien.Pose, phase=CONTACT, refine: int = 0, name: str = "") -> bool:
        self.phase(phase)
        res = self._plan(self.pl.move_to_pose_with_screw, pose, refine_steps=refine)
        if res == -1:
            res = self._plan(self.pl.move_to_pose_with_RRTConnect, pose, refine_steps=refine)
        if res == -1:
            self.failed.append(name or "move")
            return False
        return True

    def line(self, pose: sapien.Pose, phase=CONTACT, seg: float = 0.03, name: str = "") -> bool:
        """straight TCP line to `pose` (orientation of `pose`) in screw segments of <= seg metres, NO RRT fallback (an RRT
        detour would sweep the held tool / object through the scene); a failing segment is retried in halves once"""
        self.phase(phase)
        p0 = self.tcp().p
        n = max(1, int(np.ceil(np.linalg.norm(pose.p - p0) / seg)))
        for k in range(1, n + 1):
            tgt = sapien.Pose(p0 + (pose.p - p0) * k / n, pose.q)
            if self._plan(self.pl.move_to_pose_with_screw, tgt) != -1:
                continue
            # near the reach limit a few cm need large joint motions (screw plan refused): 5 mm screw sub-steps, then CLIK
            a = self.tcp().p
            m = max(1, int(np.ceil(np.linalg.norm(tgt.p - a) / 0.005)))
            for j in range(1, m + 1):
                sub = sapien.Pose(a + (tgt.p - a) * j / m, pose.q)
                if self._plan(self.pl.move_to_pose_with_screw, sub) == -1 and not self.clik(sub):
                    self.failed.append(name or "line")
                    return False
        return True

    def clik(self, pose: sapien.Pose, step: float = 0.004) -> bool:
        """straight TCP line by incremental damped IK (pinocchio CLIK on the planner's model, one waypoint per control
        step at <= `step` m), no collision check: the fallback when mplib's screw plan fails near the reach limit"""
        pl = self.pl.planner
        pin = pl.pinocchio_model
        idx = pl.link_name_2_idx[pl.move_group]
        q = self.pl.robot.get_qpos()[0].cpu().numpy().astype(np.float64).copy()     # measured (consistent with tcp())
        Tb = np.linalg.inv(self.u.agent.robot.pose.sp.to_transformation_matrix())
        p0, p1 = self.tcp().p, np.asarray(pose.p, float)
        n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
        lims = np.concatenate(pin.get_joint_limits())
        path = []
        from transforms3d.quaternions import mat2quat
        for k in range(1, n + 1):
            T = sapien.Pose(p0 + (p1 - p0) * k / n, pose.q).to_transformation_matrix()
            G = Tb @ T
            # a joint at its limit (e.g. the elbow fully stretched at the reach limit) makes the damped IK pick the
            # stretched branch: start the iteration 0.05 rad inside the limits
            guess = q.copy()
            guess[:7] = np.clip(q[:7], lims[:7, 0] + 0.05, lims[:7, 1] - 0.05)
            res = pin.compute_IK_CLIK(idx, np.concatenate([G[:3, 3], mat2quat(G[:3, :3])]), guess, [])
            qn, err = np.asarray(res[0]), float(np.linalg.norm(res[2]))
            if err > 3e-3 or (qn < lims[:, 0] - 1e-4).any() or (qn > lims[:, 1] + 1e-4).any() or np.abs(qn - q)[:7].max() > 0.3:
                return False
            q = qn
            path.append(q[:7].copy())
        self.pl.follow_path(dict(position=np.array(path)))
        return True

    def can_reach(self, pose: sapien.Pose) -> bool:
        return self._plan(self.pl.move_to_pose_with_screw, pose, dry_run=True) != -1

    def grip(self, state: float, t: int = 6, phase=CONTACT):
        self.phase(phase)
        if state >= 0:
            self.pl.open_gripper(t=t, gripper_state=state)
        else:
            self.pl.close_gripper(t=t, gripper_state=state)

    def wait(self, t: int, phase=CONTACT):
        """hold the current arm configuration and gripper command for t steps"""
        self.phase(phase)
        q = self.pl.robot.get_qpos()[0, :7].cpu().numpy()
        for _ in range(t):
            self.env.step(np.hstack([q, self.pl.gripper_state]))

    def tcp(self) -> sapien.Pose:
        return self.u.agent.tcp.pose.sp

    def retreat_up(self, dz: float = 0.10, phase=FREE) -> bool:
        """straight up from the current TCP pose (after a release)"""
        p = self.tcp()
        return self.move(sapien.Pose(p.p + np.array([0, 0, dz]), p.q), phase=phase, name="retreat")

    def log(self, ok: bool, variant: str, **kw) -> dict:
        st = "ok" if ok and not self.failed else ("plan_failed: " + ",".join(self.failed) if self.failed else "failed")
        return dict(status=st, variant=variant, arm=0, steps=self.pl.elapsed_steps, **kw)


def top_grasp(ex: PandaExec, actor, angles=(0.0,), center=None, depth: float = FINGER_LENGTH):
    """ManiSkill's OBB top-down grasp (compute_grasp_info_by_obb), the first yaw offset among `angles` that is
    reachable by a screw motion"""
    from ..compat import grasp_utils
    compute_grasp_info_by_obb, get_actor_obb = grasp_utils().compute_grasp_info_by_obb, grasp_utils().get_actor_obb
    u = ex.u
    obb = get_actor_obb(actor)
    approaching = np.array([0, 0, -1])
    target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    gi = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=depth)
    g = u.agent.build_grasp_pose(approaching, gi["closing"], gi["center"] if center is None else center)
    for a in angles:
        g2 = g * sapien.Pose(q=euler2quat(0, 0, a))
        if ex.can_reach(g2):
            return g2
    return g


def yaw_angles(base=0.0):
    a = np.arange(0, np.pi * 2 / 3, np.pi / 2) + base
    a = np.repeat(a, 2)
    a[1::2] *= -1
    return tuple(a)


def pick(ex: PandaExec, actor, lift: float = 0.10, angles=None, grip: float = -1, reach_dz: float = 0.05,
         depth: float = FINGER_LENGTH, pre_wait: int = 0, close_t: int = 6) -> sapien.Pose | None:
    """reach (FREE) -> [pre_wait steps still at the pre-grasp pose, CONTACT: start of the final approach] -> grasp
    (CONTACT) -> close -> lift; returns the lift pose (None on a planning failure). The keyword defaults are the frozen
    values (A2 parameters)."""
    g = top_grasp(ex, actor, angles=angles or yaw_angles(), depth=depth)
    if not ex.move(g * sapien.Pose([0, 0, -reach_dz]), FREE, name="reach"):
        return None
    if pre_wait:
        ex.wait(pre_wait, CONTACT)
    if not ex.move(g, CONTACT, name="grasp"):
        return None
    ex.grip(grip, t=close_t)
    lp = sapien.Pose([0, 0, lift]) * g
    if not ex.move(lp, CONTACT, name="lift"):
        return None
    return lp


def carry_offset(ex: PandaExec, actor, goal_p, lift_pose: sapien.Pose) -> sapien.Pose:
    """TCP pose that puts the held actor's centre at goal_p (keeps the lift orientation)"""
    off = np.asarray(goal_p, float) - actor.pose.sp.p
    return sapien.Pose(lift_pose.p + off, lift_pose.q)


def cage_open(ex: PandaExec, width: float | None, extra: float = 0.0015, t: int = 10):
    """open the Panda gripper so that the pads clear an object of `width` by ~extra/2 per side (no contact, still
    enclosing): gripper command in [-1, 1] maps linearly to the finger target -0.01 .. 0.04 m. width=None: the
    current measured finger opening (the gripper is closed on the object)"""
    if width is None:
        width = float(ex.u.agent.robot.get_qpos()[0, -2:].sum())
    target = (width + extra) / 2.0                       # per finger
    cmd = (target + 0.01) / 0.05 * 2 - 1
    ex.grip(float(np.clip(cmd, -1, 1)), t=t)


def face_yaw(actor) -> float:
    """yaw of a box's face normal about world z, modulo pi/2 (in [-pi/4, pi/4))"""
    R = actor.pose.sp.to_transformation_matrix()[:3, :3]
    a = R[:, 0] if abs(R[2, 0]) < 0.7 else R[:, 1]
    y = float(np.arctan2(a[1], a[0]))
    return (y + np.pi / 4) % (np.pi / 2) - np.pi / 4


def align_held_yaw(ex: PandaExec, held, yaw_to: float, name: str = "yaw") -> bool:
    """rotate the TCP about world z (in place) so that the held box's faces get the yaw `yaw_to` (mod pi/2)"""
    d = (yaw_to - face_yaw(held) + np.pi / 4) % (np.pi / 2) - np.pi / 4
    if abs(d) < np.radians(1.0):
        return True
    p = ex.tcp()
    q = sapien.Pose(q=euler2quat(0, 0, d)) * sapien.Pose(q=p.q)
    return ex.move(sapien.Pose(p.p, q.q), CONTACT, name=name)


def no_op(env, variant: str = "no_op", n: int = 20) -> dict:
    ex = PandaExec(env)
    ex.wait(n, FREE)
    return ex.log(True, variant)


# ================================================================================================ PickCube
def pick_cube(env, variant: str = "expert") -> dict:
    """expert: grasp, lift 10 cm, carry the cube centre to the goal, hold still (ManiSkill's solution moves from the
    grasp straight to the goal; the lift keeps a low goal from being reached by dragging the cube on the table).
    drop_at_goal: carry to the goal; the policy's last action is the gripper-open command.
    place_under_goal: put the cube on the table under the goal, open, retreat.
    near_miss: hold the cube 4 cm (horizontally) from the goal."""
    if variant == "no_op":
        return no_op(env)
    ex = PandaExec(env)
    u = ex.u
    lp = pick(ex, u.cube, lift=0.10)
    if lp is None:
        return ex.log(False, variant)
    goal = u.goal_site.pose.sp.p.copy()
    if variant == "near_miss":
        d = goal[:2] - u.agent.robot.pose.sp.p[:2]
        d = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)          # sideways from the base->goal ray
        goal[:2] += 0.04 * d
    if variant == "place_under_goal":
        goal[2] = float(u.cube_half_size) + 0.004
    ok = ex.move(carry_offset(ex, u.cube, goal, lp), CONTACT, name="carry")
    if ok:      # re-aim once from the actual in-hand pose
        ok = ex.move(carry_offset(ex, u.cube, goal, ex.tcp()), CONTACT, name="aim")
    if not ok:
        return ex.log(False, variant)
    if variant == "drop_at_goal":
        ex.grip(1, t=1)                     # the opening command is the policy's last action
    elif variant == "place_under_goal":
        ex.grip(1)
        ex.retreat_up(0.10)
    else:
        ex.wait(5)
    return ex.log(True, variant)


# ================================================================================================ StackCube
def stack_cube(env, variant: str = "expert", params: dict | None = None) -> dict:
    """expert: ManiSkill's solution (grasp A, lift 10 cm, align over B, lower until A rests on B) + open fully, retreat
    10 cm up (the solution opens and stops; lowering onto B replaces its release from the align height).
    hold_on_top: A put on B, gripper stays closed. cage: A put on B, gripper opened ~1.5 mm beyond the cube width
    (no contact, still around A). press_on_top: A released on B, then the closed gripper presses down on A's top face.
    beside: A put on the table next to B (touching), released, retreat.
    params (A2, imitator.planning.params 'StackCube-v1'): None = the frozen values (+ an adopted tuned set).
    v1.2 (A2 revision R1): the grasp of A is verified and retried (see below)."""
    if variant == "no_op":
        return no_op(env)
    from . import params as PR
    p = PR.pv(params, "StackCube-v1", env, variant)
    ex = PandaExec(env, vel=p["vel"], acc=p["vel"])
    u = ex.u
    h = float(u.cube_half_size[2])
    # A2 expert revision R1 (v1.2, 2026-11-17): verified grasp (planning/panda_ms_rev.pick_verified: is_grasping after
    # closing, up to 2 re-grasps, 'plan_failed: grasp' instead of carrying on with an empty gripper)
    from .panda_ms_rev import pick_verified
    lp, why = pick_verified(ex, u.cubeA, lift=p["lift_dz"], reach_dz=p["reach_dz"], depth=p["grasp_depth"],
                            pre_wait=p["pre_wait"], close_t=p["close_steps"], retries=2, retry_wait=10)
    if lp is None:
        return dict(ex.log(False, variant), status=f"plan_failed: {why}")
    B = u.cubeB.pose.sp.p.copy()
    if variant == "beside":
        d = u.cubeA.pose.sp.p[:2] - B[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        goal = np.array([B[0] + d[0] * (2 * h + 0.004), B[1] + d[1] * (2 * h + 0.004), h + 0.003])
    else:
        goal = B + np.array([0, 0, 2 * h + p["place_dz"]])
    above = goal + np.array([0, 0, p["align_dz"]])
    ok = ex.move(carry_offset(ex, u.cubeA, above, lp), CONTACT, name="align")
    if ok and p["align_wait"]:
        ex.wait(p["align_wait"])
    ok = ok and ex.move(carry_offset(ex, u.cubeA, goal, ex.tcp()), CONTACT, name="lower")
    if not ok:
        return ex.log(False, variant)
    ex.wait(p["place_wait"])
    if PR.B(p["reaim"]) and variant != "beside":
        # A2 switch: closed-loop correction before the release: A's centre measured against B's top centre; if it is
        # off by 6 mm - 2.5 cm horizontally, lift 1.5 cm, shift by the error, lower again
        e = (u.cubeA.pose.sp.p - (u.cubeB.pose.sp.p + np.array([0, 0, 2 * h])))[:2]
        if 0.006 < np.linalg.norm(e) < 0.025:
            t = ex.tcp()
            ok = ex.move(sapien.Pose(t.p + np.array([0, 0, 0.015]), t.q), CONTACT, name="reaim_up")
            ok = ok and ex.move(sapien.Pose(t.p + np.array([-e[0], -e[1], 0.0]), t.q), CONTACT, name="reaim_down")
            ex.wait(p["place_wait"])
            if not ok:
                return ex.log(False, variant)
    if variant == "hold_on_top":
        ex.wait(5)
    elif variant == "cage":
        cage_open(ex, 2 * h)
    else:
        ex.grip(1, t=p["release_steps"])
        ok = ex.retreat_up(p["up_m"])
        if ok and variant == "press_on_top":
            ex.grip(-1)
            top = u.cubeA.pose.sp.p + np.array([0, 0, h])
            p = ex.tcp()
            ok = ex.move(sapien.Pose([top[0], top[1], top[2] + 0.03], p.q), CONTACT, name="press_above")
            ok = ok and ex.move(sapien.Pose([top[0], top[1], top[2] - 0.004], p.q), CONTACT, name="press")
            ex.wait(5)
    return ex.log(ok, variant)


# ================================================================================================ PushCube / PullCube
def _push_like(env, variant: str, direction: float, params: dict | None = None, key: str | None = None) -> dict:
    """direction +1 = push along +x (PushCube), -1 = pull towards the robot along -x (PullCube; the fingers hook the
    cube's far face). expert: ManiSkill's solution (closed gripper behind the cube, straight line to the goal) + a
    retreat straight up. carry: pick the cube up, place it at the goal, release, retreat. *_and_stay: the expert's
    motion, the gripper stays against the cube. short: stop 3 cm before the goal region's edge. hold_above (PullCube):
    pick the cube and hold it 5 cm above the goal centre."""
    if variant == "no_op":
        return no_op(env)
    from . import params as PR
    p = PR.pv(params, key, env, variant) if key is not None else dict(vel=0.9, start_back=0.05, push_dz=0.0, pre_wait=0, push_end=None,
                                                         push_segs=1, back_m=0.03, up_m=0.10)
    ex = PandaExec(env, vel=p["vel"], acc=p["vel"])
    u = ex.u
    h = float(u.cube_half_size)
    goal = u.goal_region.pose.sp.p.copy()
    R = float(u.goal_radius)
    if variant in ("carry", "hold_above"):
        lp = pick(ex, u.obj, lift=0.10)
        if lp is None:
            return ex.log(False, variant)
        tgt = np.array([goal[0], goal[1], h + 0.004]) if variant == "carry" else np.array([goal[0], goal[1], h + 0.05])
        ok = ex.move(carry_offset(ex, u.obj, tgt + np.array([0, 0, 0.04]), lp), CONTACT, name="carry")
        ok = ok and ex.move(carry_offset(ex, u.obj, tgt, ex.tcp()), CONTACT, name="lower")
        if ok and variant == "carry":
            ex.grip(1)
            ok = ex.retreat_up(0.10)
        elif ok:
            ex.wait(5)
        return ex.log(ok, variant)
    ex.grip(-1, phase=FREE)
    q = ex.tcp().q
    c = u.obj.pose.sp.p
    start = sapien.Pose(p=c + np.array([-direction * p["start_back"], 0, p["push_dz"]]), q=q)
    ok = ex.move(start, FREE, name="reach")
    if not ok:
        return ex.log(False, variant)
    if p["pre_wait"]:
        ex.wait(p["pre_wait"], CONTACT)                  # still behind the cube before the push (A2 parameter)
    if direction > 0:   # ManiSkill: TCP 12 cm behind the goal centre (cube centre ~ 7 cm behind it)
        end = goal + np.array([-(0.12 if p["push_end"] is None else p["push_end"]), 0, 0])
    else:               # ManiSkill: TCP 5 cm beyond the goal centre (cube centre ~ at the goal centre)
        end = goal + np.array([0.05, 0, 0])
    if variant == "short":
        # cube centre ends 3 cm outside the goal disc: shift the end point back along the push direction
        cube_end_nominal = end[0] + direction * 0.05 if direction > 0 else end[0] - 0.05
        want = goal[0] - direction * (R + 0.03)
        end = end + np.array([want - cube_end_nominal, 0, 0])
    end = np.array([end[0], end[1], start.p[2]])
    nseg = int(p["push_segs"])
    if nseg <= 1:
        ok = ex.move(sapien.Pose(p=end, q=q), CONTACT, name="push")
    else:
        # A2: the push in nseg straight segments; before each one the lateral offset of the TCP from the line through
        # the cube centre (along the push direction) is corrected (closed loop on the cube's measured position)
        ok = True
        t0 = ex.tcp().p
        for j in range(1, nseg + 1):
            tgt = t0 + (end - t0) * j / nseg
            cy = u.obj.pose.sp.p[1]
            tgt = np.array([tgt[0], cy, tgt[2]])
            ok = ex.move(sapien.Pose(p=tgt, q=q), CONTACT, name="push") and ok
    if not ok:
        return ex.log(False, variant)
    # back off from the cube along the push direction's opposite, then up (FREE: nothing grasped)
    pt = ex.tcp().p
    ok = ex.move(sapien.Pose(pt - np.array([direction * p["back_m"], 0, 0]), q), FREE, name="back")
    ok = ok and ex.retreat_up(p["up_m"])
    if ok and variant in ("push_and_press", "pull_and_press"):
        # the robot keeps loading the cube at the end: closed fingertips pressed 4 mm down onto its top face
        top = u.obj.pose.sp.p + np.array([0, 0, h])
        ok = ex.move(sapien.Pose([top[0], top[1], top[2] + 0.03], q), CONTACT, name="press_above")
        ok = ok and ex.move(sapien.Pose([top[0], top[1], top[2] - 0.004], q), CONTACT, name="press")
        ex.wait(5)
    return ex.log(ok, variant)


def push_cube(env, variant: str = "expert", params: dict | None = None) -> dict:
    return _push_like(env, variant, +1.0, params, key="PushCube-v1")


def pull_cube(env, variant: str = "expert") -> dict:
    return _push_like(env, variant, -1.0)


# ================================================================================================ PlaceSphere
def place_sphere(env, variant: str = "expert") -> dict:
    """expert: ManiSkill's solution (grasp the sphere, lift 10 cm, align over the bin) + lower until the sphere is 3 mm
    above the bin floor, open, retreat. hold_in_bin: lowered into the bin, gripper stays closed. cage: lowered into the
    bin, gripper opened ~1.5 mm beyond the sphere diameter. beside: put on the table next to the bin, released."""
    if variant == "no_op":
        return no_op(env)
    ex = PandaExec(env)
    u = ex.u
    r = float(u.radius)
    lp = pick(ex, u.obj, lift=0.10, angles=yaw_angles(np.pi / 4))
    if lp is None:
        return ex.log(False, variant)
    binp = u.bin.pose.sp.p
    floor = binp[2] + float(u.block_half_size[0])          # bin floor top
    if variant == "beside":
        d = u.obj.pose.sp.p[:2] - binp[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        goal = np.array([binp[0] + d[0] * 0.06, binp[1] + d[1] * 0.06, r + 0.003])
    else:
        goal = np.array([binp[0], binp[1], floor + r + 0.003])
    ok = ex.move(carry_offset(ex, u.obj, goal + np.array([0, 0, 0.05]), lp), CONTACT, name="align")
    ok = ok and ex.move(carry_offset(ex, u.obj, goal, ex.tcp()), CONTACT, name="lower")
    if not ok:
        return ex.log(False, variant)
    ex.wait(3)
    if variant == "hold_in_bin":
        ex.wait(5)
    elif variant == "cage":
        cage_open(ex, 2 * r)
    else:
        ex.grip(1)
        ok = ex.retreat_up(0.10)
    return ex.log(ok, variant)


# ================================================================================================ StackPyramid
def stack_pyramid(env, variant: str = "expert") -> dict:
    """expert (ours; ManiSkill's solution carries A to 0.8 x B's absolute position, which is not a placement next to B):
    pick A, put it on the table touching B on the side facing C... (side chosen along the B->A direction), release,
    retreat; pick C, lower it onto the middle of the A-B pair, release, retreat.
    hold_top: as the expert, but the gripper stays closed on C. c_on_a_only: C put on A, 8 mm off A's centre away from B
    (it does not rest on B).
    pair_only: A put next to B, C left on the table. cage_top: C lowered onto the pair, gripper opened ~1.5 mm beyond C."""
    if variant == "no_op":
        return no_op(env)
    ex = PandaExec(env)
    u = ex.u
    h = float(u.cube_half_size[2])
    lp = pick(ex, u.cubeA, lift=0.10)
    if lp is None:
        return ex.log(False, variant)
    B = u.cubeB.pose.sp.p.copy()
    # A goes face to face with B: along the face normal of B closest to the B -> A direction, A's yaw matched to B's
    yb = face_yaw(u.cubeB)
    if not align_held_yaw(ex, u.cubeA, yb, "A_yaw"):
        return ex.log(False, variant)
    # the fingers must not come between A and B: d is one of B's face normals perpendicular to the closing axis
    yc = ex.tcp().to_transformation_matrix()[:3, 1][:2]
    yc = yc / (np.linalg.norm(yc) + 1e-9)
    v = u.cubeA.pose.sp.p[:2] - B[:2]
    normals = [np.array([np.cos(yb + k * np.pi / 2), np.sin(yb + k * np.pi / 2)]) for k in range(4)]
    normals = [n for n in normals if abs(float(n @ yc)) < 0.5]
    d = max(normals, key=lambda n: float(n @ v))
    gA = np.array([B[0] + d[0] * (2 * h + 0.003), B[1] + d[1] * (2 * h + 0.003), h + 0.003])
    ok = ex.move(carry_offset(ex, u.cubeA, gA + np.array([0, 0, 0.05]), ex.tcp()), CONTACT, name="A_align")
    ok = ok and ex.move(carry_offset(ex, u.cubeA, gA, ex.tcp()), CONTACT, name="A_lower")
    if not ok:
        return ex.log(False, variant)
    ex.grip(1)
    if not ex.retreat_up(0.10):
        return ex.log(False, variant)
    if variant == "pair_only":
        return ex.log(True, variant)
    lp = pick(ex, u.cubeC, lift=0.12)
    if lp is None:
        return ex.log(False, variant)
    A, B = u.cubeA.pose.sp.p, u.cubeB.pose.sp.p
    if variant == "c_on_a_only":        # on A, shifted 8 mm away from B so that C cannot touch B (still stable on A)
        away = (A - B) / (np.linalg.norm(A - B) + 1e-9)
        mid = A + away * 0.008
    else:
        mid = (A + B) / 2
    gC = np.array([mid[0], mid[1], mid[2] + 2 * h + 0.003])
    if not align_held_yaw(ex, u.cubeC, face_yaw(u.cubeB), "C_yaw"):
        return ex.log(False, variant)
    ok = ex.move(carry_offset(ex, u.cubeC, gC + np.array([0, 0, 0.05]), ex.tcp()), CONTACT, name="C_align")
    ok = ok and ex.move(carry_offset(ex, u.cubeC, gC, ex.tcp()), CONTACT, name="C_lower")
    if not ok:
        return ex.log(False, variant)
    ex.wait(3)
    if variant == "hold_top":
        ex.wait(5)
    elif variant == "cage_top":
        cage_open(ex, 2 * h)
    else:
        ex.grip(1)
        ok = ex.retreat_up(0.10)
    return ex.log(ok, variant)


# ================================================================================================ LiftPegUpright
def lift_peg_upright(env, variant: str = "expert") -> dict:
    """expert: ManiSkill's solution (grasp 10 cm from the peg centre, partial close -0.6, lift 30 cm, rotate the hand by
    36 deg about its y axis... then lower 10 cm) changed to place the peg: after rotating, the peg's lower end is put on
    the table (bottom 3 mm up), opened, retreat. hold_upright: same, gripper stays closed. tilted_release: the hand
    rotation is stopped 25 deg short of upright, the peg is lowered until it touches the table and released (it falls).
    lift_flat: the peg is lifted 10 cm without rotating and held."""
    if variant == "no_op":
        return no_op(env)
    ex = PandaExec(env, vel=0.75, acc=0.75)
    u = ex.u
    from ..compat import grasp_utils
    compute_grasp_info_by_obb, get_actor_obb = grasp_utils().compute_grasp_info_by_obb, grasp_utils().get_actor_obb
    obb = get_actor_obb(u.peg)
    approaching = np.array([0, 0, -1])
    target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    gi = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    g = u.agent.build_grasp_pose(approaching, gi["closing"], gi["center"]) * sapien.Pose([0.10, 0, 0])
    if not ex.move(g * sapien.Pose([0, 0, -0.05]), FREE, name="reach"):
        return ex.log(False, variant)
    if not ex.move(g, CONTACT, name="grasp"):
        return ex.log(False, variant)
    ex.grip(-0.6)
    if variant == "lift_flat":
        ok = ex.move(sapien.Pose([0, 0, 0.10]) * g, CONTACT, name="lift")
        ex.wait(5)
        return ex.log(ok, variant)
    lp = sapien.Pose([0, 0, 0.30]) * g
    if not ex.move(lp, CONTACT, name="lift"):
        return ex.log(False, variant)
    # find the hand rotation that makes the peg axis vertical (search, as the in-hand pose varies)
    best = None
    for th in np.linspace(np.pi / 10 * 0.6, np.pi / 10 * 1.6, 11):
        for s in (1, -1):
            pose = lp * sapien.Pose(q=np.array([np.cos(s * th), 0, np.sin(s * th), 0]))
            Tp = u.peg.pose.sp.to_transformation_matrix()
            Tt = ex.tcp().to_transformation_matrix()
            Tn = pose.to_transformation_matrix() @ np.linalg.inv(Tt) @ Tp
            tilt = np.degrees(np.arccos(np.clip(abs(Tn[2, 0]), 0, 1)))
            if best is None or tilt < best[0]:
                best = (tilt, pose, Tn)
    tilt0, pose, Tn = best
    if variant == "tilted_release":     # stop 25 deg short of upright: rotate back about the hand's y axis
        for th in np.linspace(0.0, 0.8, 41):
            for s in (1, -1):
                p2 = pose * sapien.Pose(q=np.array([np.cos(s * th / 2), 0, np.sin(s * th / 2), 0]))
                Tn2 = p2.to_transformation_matrix() @ np.linalg.inv(ex.tcp().to_transformation_matrix()) @ \
                    u.peg.pose.sp.to_transformation_matrix()
                t2 = np.degrees(np.arccos(np.clip(abs(Tn2[2, 0]), 0, 1)))
                if abs(t2 - 25.0) < 2.0:
                    pose, Tn = p2, Tn2
                    break
            else:
                continue
            break
    if not ex.move(pose, CONTACT, name="rotate"):
        return ex.log(False, variant)
    # lower until the peg's lowest point is 3 mm above the table
    L, W = float(u.peg_half_length), float(u.peg_half_width)
    Tp = u.peg.pose.sp.to_transformation_matrix()
    corners = np.array([[sx * L, sy * W, sz * W] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    zmin = float((corners @ Tp[:3, :3].T + Tp[:3, 3])[:, 2].min())
    dz = zmin - 0.003
    p = ex.tcp()
    if not ex.move(sapien.Pose(p.p - np.array([0, 0, dz]), p.q), CONTACT, name="lower"):
        return ex.log(False, variant)
    ex.wait(3)
    if variant == "hold_upright":
        ex.wait(5)
        return ex.log(True, variant)
    ex.grip(1)
    ok = True
    if variant == "expert":
        # back the hand off along its approach axis, then up
        T = ex.tcp().to_transformation_matrix()
        ok = ex.move(sapien.Pose(T[:3, 3] - T[:3, 2] * 0.06, ex.tcp().q), FREE, name="back")
        ok = ok and ex.retreat_up(0.08)
    else:
        ex.wait(5)
    return ex.log(ok, variant, upright_tilt_planned_deg=float(tilt0))


# ================================================================================================ PlugCharger
def _plug_goal(u) -> sapien.Pose:
    """the charger's plugged pose from the CURRENT receptacle pose (ManiSkill: goal = receptacle * Rz(pi))"""
    return u.receptacle.pose.sp * sapien.Pose(q=euler2quat(0, 0, np.pi))


def _tcp_for_charger(ex: PandaExec, target: sapien.Pose) -> sapien.Pose:
    return target * ex.u.charger.pose.sp.inv() * ex.tcp()


def plug_charger(env, variant: str = "expert", params: dict | None = None) -> dict:
    """expert: ManiSkill's solution (OBB grasp of the charger base tilted 15 deg, align 5 cm in front of the receptacle,
    insert along the goal axis) with E27's refinement (the pre-insertion command is corrected by the charger's actual
    pose error; here up to 12 rounds to 0.3 mm), then (ours) a slow straight insertion in 1 mm increments (joint speed
    limit 0.3); on a stall (prongs on the receptacle face): v1.1 aligns at contact (static lateral error measured with the
    charger backed off 1 mm, up to 6 corrections), then a spiral search of lateral offsets (0.3-1.2 mm, re-refining every
    8 tries); finally release + back off 5 cm along the insertion axis + up. (Final eval 'final' used v1 = without the
    align-at-contact step; 'final_v11' uses v1.1, contract unchanged.) hold_plugged: inserted, gripper stays closed.
    release_short: insertion stopped 1.5 cm short, released. cage_plugged: inserted, gripper opened ~1.5 mm beyond the
    grasped width. no_op.
    params (A2, imitator.planning.params 'PlugCharger-v1'): the numbers above are the defaults; None = defaults (+ an
    adopted tuned set)."""
    if variant == "no_op":
        return no_op(env)
    import trimesh
    from ..compat import grasp_utils
    compute_grasp_info_by_obb = grasp_utils().compute_grasp_info_by_obb
    from . import params as PR
    p = PR.pv(params, "PlugCharger-v1", env, variant)
    ex = PandaExec(env, vel=p["vel"], acc=p["vel"])
    u = ex.u
    bp = u.charger_base_pose.sp
    obb = trimesh.primitives.Box(extents=np.array(u._base_size) * 2, transform=bp.to_transformation_matrix())
    gi = compute_grasp_info_by_obb(obb, approaching=np.array([0, 0, -1.0]),
                                   target_closing=u.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1], depth=FINGER_LENGTH)
    g = u.agent.build_grasp_pose(np.array([0, 0, -1.0]), gi["closing"], gi["center"]) * \
        sapien.Pose(q=euler2quat(0, np.deg2rad(p["grasp_tilt_deg"]), 0))
    if not ex.move(g * sapien.Pose([0, 0, -p["reach_dz"]]), FREE, name="reach"):
        return ex.log(False, variant)
    if p["pre_wait"]:
        ex.wait(p["pre_wait"], CONTACT)                  # A2: still at the pre-grasp pose before the final approach
    if not ex.move(g, CONTACT, name="grasp"):
        return ex.log(False, variant)
    ex.grip(-1)
    pre = _tcp_for_charger(ex, _plug_goal(u) * sapien.Pose([-0.05, 0, 0]))
    if not ex.move(pre, CONTACT, name="align"):
        return ex.log(False, variant)
    pl = ex.pl
    lim_v, lim_a = pl.planner.joint_vel_limits.copy(), pl.planner.joint_acc_limits.copy()
    pl.planner.joint_vel_limits = np.minimum(lim_v, p["insert_vel"])
    pl.planner.joint_acc_limits = np.minimum(lim_a, p["insert_vel"])

    def local_err():
        g = _plug_goal(u)
        return g.to_transformation_matrix()[:3, :3].T @ (u.charger.pose.sp.p - g.p)

    def refine(back: float, rounds: int, tol: float, cmd):
        """E27 c_refine at `back` metres in front of the plugged pose: the command is corrected by the charger's pose
        error, 5-6 hold steps per round, >= 3 rounds, until the error is below tol"""
        e = None
        for i in range(rounds):
            target = _plug_goal(u) * sapien.Pose([-back, 0, 0])
            cmd = target * u.charger.pose.sp.inv() * cmd
            ex.move(cmd, CONTACT, name="refine")
            ex.wait(p["refine_wait"])
            e = float(np.linalg.norm(u.charger.pose.sp.p - target.p))
            if i >= 2 and e < tol:
                break
        return cmd, e

    errs = []
    pre, e = refine(0.05, 12, p["refine_tol"], pre)   # E27: 6 rounds to 0.5 mm; ours: up to 12 rounds to 0.3 mm
    errs.append(round(e, 5))
    # ours (ManiSkill / E27 move the 5 cm in one screw motion and jam ~1.6 cm short = prong length, with 10-15 deg tilt):
    # from the refined pre-insertion command, a straight insertion in 1 mm increments with 3 hold steps each (the
    # refined command already compensates the arm's steady-state sag, which is ~constant over the 5 cm translation)
    depth = 0.05 - (0.015 if variant == "release_short" else 0.0)
    axis = _plug_goal(u).to_transformation_matrix()[:3, 0]
    ok, retries = True, 0
    step = float(p["insert_step"])
    n_inc = int(round(depth / step))
    inv_step = 1.0 / step
    gR = _plug_goal(u).to_transformation_matrix()[:3, :3]
    lateral = np.zeros(3)
    base_lateral = np.zeros(3)
    contact_fix = 0
    radii = tuple(round(p["spiral_r"] * f, 10) for f in (0.25, 0.5, 0.75, 1.0))
    spiral = [(r * np.cos(a), r * np.sin(a)) for r in radii for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    k = 1
    while k <= n_inc:
        ok = ex.move(sapien.Pose(axis * (step * k) + lateral) * pre, CONTACT, name="insert") and ok
        ex.wait(p["insert_wait"])
        ax = 0.05 + float(local_err()[0])                 # inserted depth of the charger from the pre-insertion point
        if ax < step * k - p["stall_m"] and contact_fix < 6:
            # stalled (the prongs hit the receptacle face): align at contact first (expert v1.1, E29b): back off 1 mm,
            # hold 8 steps, measure the charger's lateral error w.r.t. the goal axis (static) and shift the command by
            # it (accumulated), up to 6 times
            contact_fix += 1
            k = max(1, int(ax * inv_step) - 1)
            ex.move(sapien.Pose(axis * (step * k) + lateral) * pre, CONTACT, name="backoff")
            ex.wait(p["fix_wait"])
            le = local_err()
            lateral = lateral - gR @ np.array([0.0, le[1], le[2]])
            base_lateral = lateral.copy()
            continue
        if ax < step * k - p["stall_m"] and retries < len(spiral):
            # still stalled: spiral search over lateral offsets in the goal frame around the contact-aligned offset,
            # backing off 1 mm before each try (classic peg-in-hole search); every 8 tries the pre-insertion command is
            # refined again 3 mm in front of the stall point (the charger may have shifted in the hand)
            if retries and retries % 8 == 0:
                back = 0.05 - max(0.0, ax - 0.003)
                pre, e = refine(back, 12, p["refine_tol"], sapien.Pose(axis * max(0.0, ax - 0.003)) * pre)
                pre = sapien.Pose(-axis * max(0.0, ax - 0.003)) * pre
            dy, dz = spiral[retries]
            retries += 1
            lateral = base_lateral + gR @ np.array([0.0, dy, dz])
            k = max(1, int(ax * inv_step) - 1)
            continue
        k += 1
    errs.append(dict(retries=retries, final_axial_mm=round(-float(local_err()[0]) * 1000, 2)))
    pl.planner.joint_vel_limits, pl.planner.joint_acc_limits = lim_v, lim_a
    if not ok:
        return ex.log(False, variant, refine_err=errs)
    ex.wait(p["post_wait"])
    if variant == "hold_plugged":
        ex.wait(5)
        return ex.log(True, variant, refine_err=errs)
    if variant == "cage_plugged":
        cage_open(ex, None)
        return ex.log(True, variant, refine_err=errs)
    ex.grip(1, t=p["release_steps"])
    t = ex.tcp()
    ok = ex.move(sapien.Pose(t.p - axis * p["back_m"], t.q), FREE, name="back")
    ok = ok and ex.retreat_up(0.06)
    return ex.log(ok, variant, refine_err=errs)


# ================================================================================================ E27's held-out tasks
# Ported from research/scripts/e27_tasks.py (E27's stage plans after ManiSkill's motion-planning solutions; that file is
# not edited) to PandaExec, with phases; same geometry, offsets and refinement rules.
def slow_line(ex: PandaExec, pose: sapien.Pose, vel: float = 0.3, name: str = "slow") -> bool:
    """E27 slow_move: straight (screw) move planned from the COMMANDED arm joints (the controller's steady-state offset
    then stays ~constant along the path) with joint velocity / acceleration limits <= vel"""
    pl = ex.pl
    q = pl.robot.get_qpos()[0].cpu().numpy().astype(np.float64).copy()
    t = getattr(ex.u.agent.controller.controllers["arm"], "_target_qpos", None)
    if t is not None:
        q[:7] = t[0].cpu().numpy()
    lim_v, lim_a = pl.planner.joint_vel_limits.copy(), pl.planner.joint_acc_limits.copy()
    pl.planner.joint_vel_limits = np.minimum(lim_v, vel)
    pl.planner.joint_acc_limits = np.minimum(lim_a, vel)
    try:
        res = pl.planner.plan_screw(np.concatenate([pose.p, pose.q]), q, time_step=ex.u.control_timestep)
    finally:
        pl.planner.joint_vel_limits, pl.planner.joint_acc_limits = lim_v, lim_a
    if res["status"] != "Success":
        ex.failed.append(name)
        return False
    ex.phase(CONTACT)
    pl.follow_path(res)
    return True


def _peg_goal(u, dx: float) -> sapien.Pose:
    """ManiSkill's goal_pose (peg pose with its head at the hole centre) shifted by dx along the peg axis"""
    return u.goal_pose.sp * sapien.Pose([dx, 0, 0])


def _peg_grasp(ex: PandaExec) -> sapien.Pose:
    """E27 peg_grasp: top grasp 2.5 cm below the peg's top face, max(5 cm, L/2 + 1 cm) from the centre towards the tail"""
    u = ex.u
    T = u.peg.pose.sp.to_transformation_matrix()
    Rp, pp = T[:3, :3], T[:3, 3]
    L, hz = float(u.peg_half_sizes[0, 0]), float(u.peg_half_sizes[0, 2])
    closing = Rp[:, 1] - Rp[2, 1] * np.array([0, 0, 1.0])
    closing /= np.linalg.norm(closing)
    if closing @ ex.tcp().to_transformation_matrix()[:3, 1] < 0:
        closing = -closing
    centre = pp - Rp[:, 0] * max(0.05, L / 2 + 0.01)
    centre[2] = pp[2] + hz - min(0.025, hz)
    return u.agent.build_grasp_pose(np.array([0, 0, -1.0]), closing, centre)


def _peg_axis_err(u) -> float:
    """lateral offsets of the peg head and of the peg centre from the hole axis (larger of the two) [m]"""
    Th = u.box_hole_pose.sp.to_transformation_matrix()
    Tp = u.peg.pose.sp.to_transformation_matrix()
    L = float(u.peg_half_sizes[0, 0])
    head = Tp[:3, 3] + Tp[:3, 0] * L
    return max(float(np.linalg.norm((Th[:3, :3].T @ (head - Th[:3, 3]))[1:])),
               float(np.linalg.norm((Th[:3, :3].T @ (Tp[:3, 3] - Th[:3, 3]))[1:])))


def peg_insertion_side(env, variant: str = "expert") -> dict:
    """expert (E27's port of ManiSkill's solution): top grasp near the tail, align the peg 1 cm in front of the box
    face, refine the commanded pre-insertion pose by the peg's pose error (up to 6 rounds, 5 hold steps each, until head
    and centre are within 1 mm of the hole axis; no correction without the peg in hand = E27's no-grasp guard), slow
    straight insertion (joint speed <= 0.3) until the head is 3 cm past the box centre, open, retreat 8 cm up.
    hold_inserted: inserted, gripper stays closed. cage_inserted: inserted, gripper opened ~1.5 mm beyond the grasped
    width. partial_insert: insertion stopped with the head 4 cm before the box centre, released, retreat.
    release_at_entrance: released at the refined pre-insertion pose (head 1 cm outside the hole), retreat. no_op."""
    if variant == "no_op":
        return no_op(env)
    ex = PandaExec(env)
    u = ex.u
    L = float(u.peg_half_sizes[0, 0])
    ex.grip(1, phase=FREE)
    g = _peg_grasp(ex)
    if not ex.move(g * sapien.Pose([0, 0, -0.05]), FREE, name="reach"):
        return ex.log(False, variant)
    if not ex.move(g, CONTACT, name="grasp"):
        return ex.log(False, variant)
    ex.grip(-1)
    cmd = _peg_goal(u, -0.01 - L) * u.peg.pose.sp.inv() * ex.tcp()
    if not ex.move(cmd, CONTACT, name="align"):
        return ex.log(False, variant)
    errs = []
    for i in range(6):
        if not bool(u.agent.is_grasping(u.peg)[0]):
            ex.failed.append("refine: peg not grasped")
            return ex.log(False, variant, refine_err=errs)
        cmd = _peg_goal(u, -0.01 - L) * u.peg.pose.sp.inv() * cmd
        ex.move(cmd, CONTACT, name="refine")
        ex.wait(5)
        errs.append(round(_peg_axis_err(u), 5))
        if i >= 2 and errs[-1] < 0.001:
            break
    if variant != "release_at_entrance":
        axis = u.goal_pose.sp.to_transformation_matrix()[:3, 0]
        depth = 0.01 + L + (0.03 if variant != "partial_insert" else -0.04)
        if not slow_line(ex, sapien.Pose(axis * depth) * cmd, name="insert"):
            return ex.log(False, variant, refine_err=errs)
        ex.wait(3)
    if variant == "hold_inserted":
        ex.wait(10)
        return ex.log(True, variant, refine_err=errs)
    if variant == "cage_inserted":
        cage_open(ex, None)
        return ex.log(True, variant, refine_err=errs)
    ex.grip(1)
    ok = ex.retreat_up(0.08)
    return ex.log(ok, variant, refine_err=errs)


def _tool_grasp(ex: PandaExec, grasp_x: float = 0.03) -> sapien.Pose:
    """E27 tool_grasp: top grasp on the handle, 3 cm from its free end (the solution: tool origin + 2 cm)"""
    u = ex.u
    T = u.l_shape_tool.pose.sp.to_transformation_matrix()
    Rt, tp = T[:3, :3], T[:3, 3]
    closing = Rt[:, 1] - Rt[2, 1] * np.array([0, 0, 1.0])
    closing /= np.linalg.norm(closing)
    if closing @ ex.tcp().to_transformation_matrix()[:3, 1] < 0:
        closing = -closing
    return u.agent.build_grasp_pose(np.array([0, 0, -1.0]), closing, tp + Rt[:, 0] * grasp_x)


def _radial(u) -> np.ndarray:
    """unit vector (xy) from the robot base to the cube"""
    b = u.agent.robot.get_links()[0].pose.sp.p
    c = u.cube.pose.sp.p
    r = np.array([c[0] - b[0], c[1] - b[1], 0.0])
    return r / (np.linalg.norm(r) + 1e-9)


def _hook_tcp(ex: PandaExec, dz: float, dy: float = 0.0, behind: float = 0.06, side: float = 0.067, tool_z: float = 0.03,
              radial: bool = False) -> sapien.Pose:
    """E27 hook_tcp: TCP pose that puts the tool, aligned with world x, behind the cube (hook inner face 6 cm beyond
    the cube centre, handle line 6.7 cm to the side; ManiSkill's offsets), tool origin 3 cm above the table (+ dz),
    with the current in-hand transform. radial (A2 switch = the v1.1 variant): the tool's x axis along the robot
    base -> cube direction instead of world x."""
    u = ex.u
    cp = u.cube.pose.sp.p
    hl, hk = float(u.handle_length), float(u.hook_length)
    if not radial:
        target = sapien.Pose([cp[0] + behind - (hl - hk), cp[1] - side + dy, tool_z + dz], [1, 0, 0, 0])
    else:
        r = _radial(u)
        n = np.array([-r[1], r[0], 0.0])
        org = cp[:2] + r[:2] * (behind - (hl - hk)) - n[:2] * (side - dy)
        target = sapien.Pose([org[0], org[1], tool_z + dz], euler2quat(0, 0, float(np.arctan2(r[1], r[0]))))
    return target * u.l_shape_tool.pose.sp.inv() * ex.tcp()


def pull_cube_tool(env, variant: str = "expert", params: dict | None = None) -> dict:
    """expert (E27's port of ManiSkill's solution): grasp the handle, lift 15 cm, move the hook above and behind the
    cube, lower it (tool origin 3 cm above the table), pull the TCP 35 cm towards the robot, open, retreat 8 cm up.
    hold_tool: the pull, gripper stays closed on the tool. cage_tool: the pull, gripper opened ~1.5 mm beyond the
    handle width. pull_short: the pull stops when the cube is ~0.65 m from the robot base (success needs < 0.6 m),
    released, retreat. pull_by_gripper: the tool is not used; the closed gripper reaches behind the cube and drags it to
    0.5 m from the base (bypasses the required tool), backs off and up. no_op.
    params (A2, imitator.planning.params 'PullCubeTool-v1'): defaults = the frozen v1 expert; the switches 'radial' (tool
    along the base -> cube direction, radial pull) and 'line' (hook lowering and pull as segmented straight lines without
    the RRT fallback; a lowering stopped early still pulls when the hook is >= 1 cm below the cube top) are the v1.1
    variants (off by default)."""
    if variant == "no_op":
        return no_op(env)
    from . import params as PR
    p = PR.pv(params, "PullCubeTool-v1", env, variant)
    radial, line = PR.B(p["radial"]), PR.B(p["line"])
    ex = PandaExec(env, vel=p["vel"], acc=p["vel"])
    u = ex.u
    base = u.agent.robot.get_links()[0].pose.sp.p
    if variant == "pull_by_gripper":
        ex.grip(-1, phase=FREE)
        q = ex.tcp().q
        c = u.cube.pose.sp.p
        start = np.array([c[0] + 0.05, c[1], 0.02])
        ok = ex.move(sapien.Pose(start + np.array([0, 0, 0.08]), q), FREE, name="above")
        ok = ok and ex.move(sapien.Pose(start, q), CONTACT, name="down")
        if not ok:
            return ex.log(False, variant)
        dy = c[1] - base[1]
        x_end = base[0] + np.sqrt(max(0.5 ** 2 - dy ** 2, 0.0)) + 0.05
        ok = ex.move(sapien.Pose([x_end, c[1], 0.02], q), CONTACT, name="drag")
        p_ = ex.tcp().p
        ok = ok and ex.move(sapien.Pose(p_ + np.array([0.03, 0, 0]), q), FREE, name="back")
        ok = ok and ex.retreat_up(0.10)
        return ex.log(ok, variant)
    hk = dict(behind=p["hook_behind"], side=p["hook_side"], tool_z=p["tool_z"], radial=radial)
    ex.grip(1, phase=FREE)
    g = _tool_grasp(ex, p["grasp_x"])
    if not ex.move(g * sapien.Pose([0, 0, -p["reach_dz"]]), FREE, name="reach"):
        return ex.log(False, variant)
    if p["pre_wait"]:
        ex.wait(p["pre_wait"], CONTACT)                  # A2: still at the pre-grasp pose before the final approach
    if not ex.move(g, CONTACT, name="grasp"):
        return ex.log(False, variant)
    ex.grip(-1)
    t = ex.tcp()
    if not ex.move(sapien.Pose(t.p + np.array([0, 0, p["lift_dz"]]), t.q), CONTACT, name="lift"):
        return ex.log(False, variant)
    if not ex.move(_hook_tcp(ex, p["approach_dz"], **hk), CONTACT, name="approach"):
        return ex.log(False, variant)
    if p["hook_wait"]:
        ex.wait(p["hook_wait"])                          # A2: still above the cube before lowering the hook
    if line:
        if not ex.line(_hook_tcp(ex, 0.0, **hk), CONTACT, name="hook"):
            # lowering stopped early (reach limit): pull anyway if the hook's lower face is >= 1 cm below the cube top
            tool_bottom = float(u.l_shape_tool.pose.sp.p[2]) - float(u.height) / 2
            if tool_bottom > float(u.cube.pose.sp.p[2]) + float(u.cube_half_size) - 0.01:
                return ex.log(False, variant)
            ex.failed.remove("hook")
    elif not ex.move(_hook_tcp(ex, 0.0, **hk), CONTACT, name="hook"):
        return ex.log(False, variant)
    pull = p["pull_m"]
    if variant == "pull_short":
        c = u.cube.pose.sp.p
        if radial:
            pull = float(np.clip(np.linalg.norm(c[:2] - base[:2]) - 0.65, 0.0, p["pull_m"]))
        else:
            dy = c[1] - base[1]
            x_goal = base[0] + np.sqrt(max(0.65 ** 2 - dy ** 2, 0.0))
            pull = float(np.clip(c[0] - x_goal, 0.0, p["pull_m"]))
    t = ex.tcp()
    direction = _radial(u) if radial else np.array([1.0, 0, 0])
    goal = sapien.Pose(t.p - direction * pull, t.q)
    if pull > 0 and not (ex.line(goal, CONTACT, name="pull") if line else ex.move(goal, CONTACT, name="pull")):
        return ex.log(False, variant)
    ex.wait(p["post_wait"])
    if variant == "hold_tool":
        ex.wait(10)
        return ex.log(True, variant)
    if variant == "cage_tool":
        cage_open(ex, None)
        return ex.log(True, variant)
    ex.grip(1)
    ok = ex.retreat_up(0.08)
    return ex.log(ok, variant)
