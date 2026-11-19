"""Batched scripted pick-and-place for the dual cube task on GPU PhysX (N parallel envs).

The E19 policy (imitator.planning.scripted.pick_place) segment by segment, synchronised across envs: before each segment
every env plans its own path from its CURRENT simulated state (mplib / pinocchio on the CPU, one ArmPlanner per arm with
env_idx switched per env), then all envs execute their paths in lockstep (shorter paths hold their last target). The
acting arm per env is the arm on the cube's side; the other arm holds home. An env whose planning fails holds its pose for
the rest of the episode and is reported as not executed.
Variants: release (positive control), hold, cage, no_op - as in E19.
"""
from __future__ import annotations

import numpy as np
import sapien
import torch

from .mplib_planner import ArmPlanner, arm_points
from .scripted import UP, grasp_candidates, pose


class BatchExecutor:
    def __init__(self, env):
        self.env = env
        self.u = u = env.unwrapped
        self.N = u.num_envs
        self.agents = list(u.agent.agents)
        self.keys = list(u.single_action_space.spaces.keys())
        self.targets = []
        for ag in self.agents:
            a = np.tile(ag.home_action(), (self.N, 1)).astype(np.float32)
            idx = [i for i, j in enumerate(ag.robot.active_joints) if j.name in ag.record.arm_joints]
            a[:, :ag.record.dof_arm] = ag.robot.get_qpos()[:, idx].cpu().numpy()
            self.targets.append(a)
        self.n = 0

    def action(self):
        dev = self.u.device
        return {k: torch.as_tensor(t, device=dev) for k, t in zip(self.keys, self.targets)}

    def step(self, n: int = 1):
        for _ in range(n):
            self.env.step(self.action())
            self.n += 1

    def follow(self, paths: dict, settle: int = 3):
        """paths: env -> (arm, positions (k, dof))"""
        if not paths:
            return
        n = max(len(p) for _, p in paths.values())
        for k in range(n):
            for e, (i, P) in paths.items():
                self.targets[i][e, :self.agents[i].record.dof_arm] = P[min(k, len(P) - 1)]
            self.step()
        self.step(settle)

    def set_gripper(self, arm_of: dict, closed, n: int = 20):
        """arm_of: env -> arm; closed: float or env -> float"""
        for e, i in arm_of.items():
            c = closed[e] if isinstance(closed, dict) else closed
            ag = self.agents[i]
            self.targets[i][e, ag.record.dof_arm:] = ag.gripper_action(c)
        self.step(n)


def plan_path(planner: ArmPlanner, target: sapien.Pose, method: str, attach: bool = False):
    """E19 _go's fallback chain, planning only: returns (positions or None, status)"""
    if method == "line":
        res = planner.cartesian(target)
        if res["status"] != "Success":
            res = planner.plan(target, method="screw", attach=attach)
    else:
        res = planner.plan(target, method=method, attach=attach)
    if res["status"] != "Success" and method in ("screw", "line"):
        res = planner.plan(target, method="rrt", attach=attach)
    if res["status"] != "Success" and method == "screw":
        res = planner.joint_interp(target)
    if res["status"] != "Success" and method == "screw":
        res = planner.cartesian(target)
    return (res["position"], "ok") if res["status"] == "Success" else (None, res["status"])


def pick_place_batched(env, variant: str = "release", max_candidates: int = 6) -> list[dict]:
    u = env.unwrapped
    ex = BatchExecutor(env)
    N = ex.N
    bases = u.dual.mount_poses()
    planners = [ArmPlanner(ag, b, control_dt=float(u.control_timestep)) for ag, b in zip(ex.agents, bases)]
    half = float(u.cube_half)
    cube_p0 = u.cube.pose.p.cpu().numpy().astype(float)
    cube_q0 = u.cube.pose.q.cpu().numpy().astype(float)
    tgt = u.target_pos.cpu().numpy().astype(float)
    arm = {e: (0 if cube_p0[e, 1] > 0 else 1) for e in range(N)}
    logs = [dict(arm=arm[e], variant=variant, status="?") for e in range(N)]
    alive = set(range(N))
    ex.step(1)
    if variant == "no_op":
        ex.step(20)
        for lg in logs:
            lg["status"] = "ok"; lg["steps"] = ex.n
        return logs

    def pl(e):
        p = planners[arm[e]]
        p.env_idx = e
        return p

    def fail(e, msg):
        logs[e]["status"] = msg
        alive.discard(e)

    # pre-grasp + grasp (planned open loop from the current state, first candidate whose both segments plan)
    wps, pre_paths, grasp_paths = {}, {}, {}
    for e in sorted(alive):
        p = pl(e)
        p.set_obstacle_points(arm_points(ex.agents[1 - arm[e]], env_idx=e))
        found = False
        for n, (score, wp) in enumerate(grasp_candidates(ex.agents[arm[e]], p, cube_p0[e], cube_q0[e], tgt[e], half,
                                                          bases[arm[e]].p)):
            if n >= max_candidates:
                break
            P1, st = plan_path(p, pose(wp["pre"], wp["R"]), "rrt")
            if P1 is None:
                continue
            q_full = p.qpos(); q_full[p.arm_idx] = P1[-1]
            res = p.cartesian(pose(wp["grasp"], wp["R"]), q_start=q_full)
            if res["status"] != "Success":
                continue
            wps[e], pre_paths[e], grasp_paths[e] = wp, (arm[e], P1), (arm[e], res["position"])
            logs[e].update(tilt=wp["tilt"], h=wp["h"], tried=n + 1)
            found = True
            break
        if not found:
            fail(e, "no_candidate")
    ex.follow(pre_paths)
    ex.follow({e: grasp_paths[e] for e in alive})
    ex.set_gripper({e: arm[e] for e in alive}, 1.0, n=25)
    # attach a planning proxy and lift
    shrink = {e: (0.012 if ex.agents[arm[e]].record.kind == "arm_gripper" else 0.024) for e in alive}
    paths = {}
    for e in sorted(alive):
        p = pl(e); wp = wps[e]
        p.attach_box([2 * half - shrink[e]] * 3, sapien.Pose(np.linalg.inv(wp["R"]) @ (cube_p0[e] - wp["grasp"])))
        P, st = plan_path(p, pose(wp["lift"], wp["R"]), "line", attach=True)
        if P is None:
            fail(e, f"lift: {st}")
        else:
            paths[e] = (arm[e], P)
    ex.follow(paths)
    # place correction (privileged cube pose) and transport
    cp = u.cube.pose.p.cpu().numpy().astype(float)
    paths = {}
    for e in sorted(alive):
        ag = ex.agents[arm[e]]; wp = wps[e]
        d = cp[e] - ag.tcp.pose.p[e].cpu().numpy().astype(float)
        corr = (d - (cube_p0[e] - wp["grasp"]))[:2]
        if np.linalg.norm(corr) < 0.06:
            for key in ("above", "place", "retreat"):
                wp[key] = wp[key].copy(); wp[key][:2] -= corr
            logs[e]["place_correction_m"] = float(np.linalg.norm(corr))
        p = pl(e)
        p.set_obstacle_points(arm_points(ex.agents[1 - arm[e]], env_idx=e))
        P, st = plan_path(p, pose(wp["above"], wp["R"]), "screw", attach=ag.record.kind == "arm_gripper")
        if P is None:
            fail(e, f"transport: {st}")
        else:
            paths[e] = (arm[e], P)
    ex.follow(paths)
    paths = {}
    for e in sorted(alive):
        P, st = plan_path(pl(e), pose(wps[e]["place"], wps[e]["R"]), "line", attach=True)
        if P is None:
            fail(e, f"place: {st}")
        else:
            paths[e] = (arm[e], P)
    ex.follow(paths)
    ex.step(5)
    for e in alive:
        logs[e]["grasped_at_place"] = bool(ex.agents[arm[e]].is_grasping(u.cube)[e])
    if variant == "release":
        ex.set_gripper({e: arm[e] for e in alive}, 0.0, n=20)
        paths = {}
        for e in sorted(alive):
            P, st = plan_path(pl(e), pose(wps[e]["retreat"], wps[e]["R"]), "line")
            if P is None:
                fail(e, f"retreat: {st}")
            else:
                paths[e] = (arm[e], P)
        ex.follow(paths)
        paths = {}
        for e in sorted(alive):
            res = pl(e).plan_qpos(np.asarray(ex.agents[arm[e]].record.home_arm_qpos, float))
            if res["status"] == "Success":
                paths[e] = (arm[e], res["position"])
        ex.follow(paths)
    elif variant == "hold":
        paths = {}
        for e in sorted(alive):
            res = pl(e).cartesian(pose(wps[e]["place"] - UP * 0.008, wps[e]["R"]))
            if res["status"] == "Success":
                paths[e] = (arm[e], res["position"])
        ex.follow(paths, settle=10)
    elif variant == "cage":
        live = {e: arm[e] for e in alive}
        hand = {e for e, i in live.items() if ex.agents[i].record.kind == "arm_hand"}
        if hand:
            ex.set_gripper({e: live[e] for e in hand}, 0.65, n=20)
        grip = {e: i for e, i in live.items() if e not in hand}
        lo = {e: 1.0 for e in grip}; hi = {e: 0.0 for e in grip}
        want = 2 * half + 0.0012
        for _ in range(7):
            mid = {e: (lo[e] + hi[e]) / 2 for e in grip}
            ex.set_gripper(grip, mid, n=8)
            for e, i in grip.items():
                o = float(ex.agents[i].measured_opening()[e])
                if o < want:
                    lo[e] = mid[e]
                else:
                    hi[e] = mid[e]
        ex.set_gripper(grip, hi, n=15)
    for e in alive:
        logs[e]["status"] = "ok"
    for lg in logs:
        lg["steps"] = ex.n
    return logs
