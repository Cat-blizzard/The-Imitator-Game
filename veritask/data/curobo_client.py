"""Client side of the cuRobo motion backend (runs inside a ManiSkill venv; the planner itself runs in the separate
cuRobo venv, see curobo_server.py). Converts world-frame TCP goals and the task's coarse collision world into the
robot base frame and returns a joint path sampled at the control dt, like MplibBackend.plan()."""
from __future__ import annotations

import os
import time

import numpy as np


def _slerp7(a, b, s):
    from scipy.spatial.transform import Rotation as R, Slerp
    qa, qb = a[3:], b[3:]
    sl = Slerp([0, 1], R.from_quat([[qa[1], qa[2], qa[3], qa[0]], [qb[1], qb[2], qb[3], qb[0]]]))
    q = sl([s]).as_quat()[0]
    return list(np.asarray(a[:3]) + s * (np.asarray(b[:3]) - np.asarray(a[:3]))) + [q[3], q[0], q[1], q[2]]


class CuroboBackend:
    name = "curobo"

    def __init__(self, sess, arm: int, address: str | None, n_path: int = 6):
        from multiprocessing.connection import Client
        if not address:
            raise RuntimeError("cuRobo backend requested but no server address (start veritask/data/curobo_server.py)")
        self.sess, self.arm, self.n_path = sess, arm, n_path
        self.conn = Client(address, family="AF_UNIX", authkey=os.environ.get("VERITASK_CUROBO_KEY", "veritask-e7").encode())
        self.dt = float(sess.u.control_timestep)

    def plan(self, pose, q_full, rrt_fallback=False, linear=False):
        from .maniskill.session import P, pose7
        s = self.sess; base_inv = s.agents[self.arm].robot.pose.sp.inv()
        goal_b = pose7(base_inv * pose)
        world = [dict(name=c["name"], dims=c["dims"], pose=pose7(base_inv * P(c["pose"]))) for c in s.task.world_cuboids(s)]
        req = dict(op="plan", q_start=[float(v) for v in q_full[:7]], goal=goal_b, world=world, dt=self.dt, vmax=s.vel)
        if linear:
            cur = pose7(base_inv * s.tcp(self.arm))
            req["path"] = [_slerp7(cur, goal_b, k / self.n_path) for k in range(1, self.n_path + 1)]
        t = time.time(); self.conn.send(req); r = self.conn.recv(); rtt = (time.time() - t) * 1000
        ok = bool(r.get("ok"))
        return (np.asarray(r["positions"]) if ok else None), dict(
            backend="curobo", method="cartesian" if linear else "pose", ok=ok, status=r.get("status"), ms=r.get("ms"),
            rtt_ms=round(rtt, 2), n=len(r["positions"]) if ok else 0)
