"""ManiSkill / SAPIEN binding of the data-engine Session (CPU PhysX, num_envs=1).

- Traces: veritask.adapters.maniskill.TraceRecorder with the E4 probes (data/maniskill/tasks.py), so the
  certifier sees exactly the signals the E4 contracts were validated on.
- Control: pd_joint_pos; the session keeps per-arm commanded joint targets + gripper commands (OPEN=1, CLOSED=-1)
  and builds the (single-arm vector or dual-arm dict) action itself. Idle arms hold their last commanded target.
- Motion: per-arm backends. 'mplib' = the venv's PandaArmMotionPlanningSolver planner (screw, retried once as in
  the official scripts; RRTConnect fallback when strict), 'curobo' = data/curobo_client.CuroboBackend.
  Every planning query is appended to plan_calls (q_start, goal, base pose, status, latency) for the mplib/cuRobo
  comparison.
- Snapshots: env.get_state_dict() (actors + articulations) as numpy, plus task_state (latched benchmark flags,
  static actor poses, reset-time waypoints), controller targets, traces and the episode log so far.
"""
from __future__ import annotations

import copy
import time
from collections import OrderedDict

import numpy as np

from ..engine import SEG_PREFIX, Snapshot

OPEN, CLOSED = 1.0, -1.0
CONTACT_FORCE_N = 0.05  # same threshold as E4 (research/scripts/e4_common.py)


# ------------------------------------------------------------------------------------------------ helpers
def to_np(x):
    import torch
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    return x.detach().cpu().numpy().copy() if torch.is_tensor(x) else np.asarray(x).copy()


def to_torch(x):
    import torch
    if isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    return torch.as_tensor(np.asarray(x))


def pose7(p) -> list:
    return [float(v) for v in list(p.p) + list(p.q)]


def P(x):
    import sapien
    return sapien.Pose(np.asarray(x[:3], float), np.asarray(x[3:], float))


class Contacts:
    """Port of research/scripts/e4_common.Contacts: caches the CPU contact list of the current control step
    (refresh() must be the first probe); pairwise force = |impulse| / dt summed over links."""

    def __init__(self):
        self.contacts = []; self.dt = None

    def refresh(self, u):
        self.contacts = u.scene.px.get_contacts(); self.dt = float(u.scene.px.timestep)
        return float(len(self.contacts))

    def impulse(self, a, b):
        from mani_skill.utils import sapien_utils
        return float(np.linalg.norm(sapien_utils.get_pairwise_contact_impulse(self.contacts, a._bodies[0].entity, b._bodies[0].entity)))

    def force_sum(self, links_a, b):
        return sum(self.impulse(l, b) for l in links_a) / self.dt


# ------------------------------------------------------------------------------------------------ motion backends
class MplibBackend:
    name = "mplib"

    def __init__(self, sess, arm: int, vel: float = 0.9):
        from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver as S
        ag = sess.agents[arm]
        kw = dict(debug=False, vis=False, base_pose=ag.robot.pose, visualize_target_grasp_pose=False, print_env_info=False,
                  joint_vel_limits=vel, joint_acc_limits=vel)
        if sess.n_arms > 1:
            kw["multi_robot_id"] = arm
        self.solver = S(sess.env, **kw); self.planner = self.solver.planner; self.dt = float(sess.u.control_timestep)

    def plan(self, pose, q_full, rrt_fallback=False, linear=False):
        goal = np.concatenate([pose.p, pose.q]); t = time.time(); method = "screw"
        r = self.planner.plan_screw(goal, q_full, time_step=self.dt, use_point_cloud=False)
        if r["status"] != "Success":  # the official solver retries screw once
            r = self.planner.plan_screw(goal, q_full, time_step=self.dt, use_point_cloud=False)
        if r["status"] != "Success" and rrt_fallback:
            method = "screw+rrt"
            try:
                r = self.planner.plan_qpos_to_pose(goal, q_full, time_step=self.dt, use_point_cloud=False, wrt_world=True)
            except RuntimeError as e:  # mplib TOPP can raise "Fail to parameterize path"
                r = dict(status=f"exception:{e}", position=[])
        ok = r["status"] == "Success"
        return (np.asarray(r["position"]) if ok else None), dict(backend="mplib", method=method, ok=ok, status=str(r["status"]),
                                                                  ms=round((time.time() - t) * 1000, 2), n=int(len(r["position"])) if ok else 0)


# ------------------------------------------------------------------------------------------------ session
class ManiSkillSession:
    def __init__(self, task, vel: float = 0.9, curobo_address: str | None = None, render_every: int = 0):
        """render_every > 0 stores a low-res frame (the task's human render camera, CPU lavapipe) every k control steps
        as observation.image in the episode npz. Off by default (CPU rendering is slow on this host)."""
        from ...adapters.maniskill import TraceRecorder
        from ...spec.contract import TaskContract
        self.task, self.task_id, self.vel = task, task.task_id, vel
        self.contract = TaskContract.load(task.contract_file)
        self.C = Contacts()
        self.render_every = int(render_every)
        base = task.make_env(render=self.render_every > 0)
        self.rec = TraceRecorder(base, task.probes(self.C))
        self.env = self.rec
        self.u = base.unwrapped
        self.n_arms = task.n_arms
        self.backends: dict = {}
        self.backend_name = "mplib"
        self.curobo_address = curobo_address
        self.plan_calls: list[dict] = []
        self.seed, self.step_count, self.stage, self.segment = None, 0, "start", SEG_PREFIX
        self.ctx, self.noise, self.rng, self.faults = {}, None, None, []
        self.task_state: dict = {}

    # ---------------------------------------------------------------- basics
    @property
    def agents(self):
        return list(self.u.agent.agents) if self.n_arms > 1 else [self.u.agent]

    @property
    def traces(self):
        return self.rec.traces

    def qpos(self, arm):
        return self.agents[arm].robot.get_qpos()[0].cpu().numpy()

    def tcp(self, arm):
        return self.agents[arm].tcp.pose.sp

    def reset(self, seed: int):
        pre = self.task.pre_reset(self)
        self.rec.reset(seed=seed)
        self.seed, self.step_count, self.stage, self.segment = seed, 0, "start", SEG_PREFIX
        self.ctx, self.noise, self.rng, self.faults, self.plan_calls = {}, None, None, [], []
        self.task_state = {}
        self.task.on_reset(self, pre)
        self.targets = [self.qpos(i)[:7].copy() for i in range(self.n_arms)]
        self.grip = [OPEN] * self.n_arms
        self.log = dict(action=[], qpos=[], segment=[], stage=[], obj={k: [] for k in self.task.log_objects})
        self.stage_names: list[str] = []
        if not self.backends:
            for i in range(self.n_arms):
                self.backends[("mplib", i)] = MplibBackend(self, i, self.vel)

    def _action(self):
        if self.n_arms == 1:
            return np.hstack([self.targets[0], self.grip[0]])
        keys = list(self.u.action_space.spaces.keys())
        return OrderedDict((k, np.hstack([self.targets[i], self.grip[i]])) for i, k in enumerate(keys))

    def _step(self):
        self.env.step(self._action())
        self.step_count += 1
        lg = self.log
        lg["action"].append(np.hstack([np.hstack([t, g]) for t, g in zip(self.targets, self.grip)]).astype(np.float32))
        lg["qpos"].append(np.hstack([self.qpos(i) for i in range(self.n_arms)]).astype(np.float32))
        lg["segment"].append(self.segment)
        if self.render_every and (self.step_count - 1) % self.render_every == 0:
            img = self.env.render()
            img = img[0].cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)[0]
            lg.setdefault("image", []).append(img.astype(np.uint8)); lg.setdefault("image_step", []).append(self.step_count - 1)
        if self.stage not in self.stage_names:
            self.stage_names.append(self.stage)
        lg["stage"].append(self.stage_names.index(self.stage))
        for k in self.task.log_objects:
            lg["obj"][k].append(self.task.object_pose7(self, k))

    def begin_segment(self, seg: int):
        self.segment = seg

    def hold(self, steps: int):
        for _ in range(int(steps)):
            self._step()

    def settle(self, steps: int):
        self.stage = "settle"; self.hold(steps)

    def follow(self, arm: int, positions):
        for q in positions:
            self.targets[arm] = np.asarray(q[:7], float); self._step()

    # ---------------------------------------------------------------- motion
    def set_backend(self, name: str) -> str:
        prev, self.backend_name = self.backend_name, name
        if name == "curobo":
            for i in range(self.n_arms):
                if ("curobo", i) not in self.backends:
                    from ..curobo_client import CuroboBackend
                    self.backends[("curobo", i)] = CuroboBackend(self, i, self.curobo_address)
        return prev

    def plan(self, arm: int, pose, strict: bool = False, linear: bool = False):
        """linear=True marks contact moves (grasp descent, push, press, place); mplib's screw plans are straight
        lines anyway, cuRobo then plans through Cartesian waypoints instead of a free-space path."""
        q = self.qpos(arm)
        rrt = strict or getattr(self.task, "rrt_fallback", False)
        path, rec = self.backends[(self.backend_name, arm)].plan(pose, q, rrt_fallback=rrt, linear=linear)
        rec.update(task=self.task_id, seed=self.seed, arm=arm, stage=self.stage, step=self.step_count, goal=pose7(pose), linear=linear,
                   q_start=[round(float(v), 6) for v in q[:7]], base=pose7(self.agents[arm].robot.pose.sp))
        self.plan_calls.append(rec)
        return path

    def move_to(self, arm: int, pose, strict: bool = False, linear: bool = False) -> bool:
        path = self.plan(arm, pose, strict=strict, linear=linear)
        if path is None:
            return False
        self.follow(arm, path)
        return True

    # ---------------------------------------------------------------- weak-policy noise
    def jitter(self, xyz, scale=1.0, z_scale=0.3):
        xyz = np.asarray(xyz, float).copy()
        if self.noise and self.rng is not None and self.noise.get("waypoint_sigma", 0) > 0:
            s = self.noise["waypoint_sigma"] * scale
            d = self.rng.normal(0, s, 3) * np.array([1, 1, z_scale])
            self.faults.append(dict(kind="waypoint", stage=self.stage, step=self.step_count, offset=np.round(d, 4).tolist()))
            xyz += d
        return xyz

    def fault(self, kind: str) -> bool:
        """kind in ('early', 'late'); fires with the weak policy's probability."""
        if self.noise and self.rng is not None and self.rng.random() < self.noise.get(f"p_{kind}", 0):
            self.faults.append(dict(kind=f"{kind}_gripper", stage=self.stage, step=self.step_count)); return True
        return False

    # ---------------------------------------------------------------- plans
    def plan_stages(self, plan: str) -> list[str]:
        return [s.name for s in self.task.plans[plan]]

    def run_plan(self, plan, start_stage=None, ctx=None, noise=None, rng=None, strict=False, on_boundary=None,
                 until_segment_end=False) -> dict:
        stages = self.task.plans[plan]
        names = [s.name for s in stages]
        i0 = names.index(start_stage) if start_stage else 0
        seg = stages[i0].segment
        if ctx is not None:
            self.ctx = ctx
        self.noise, self.rng = noise, rng
        ran, failed = [], []
        for st in stages[i0:]:
            if until_segment_end and st.segment != seg:
                break
            if self.ctx.get("abandon_plan"):  # a weak-policy shortcut gave up on the plan
                break
            if on_boundary is not None:
                on_boundary(st.name)
            self.stage = st.name
            ok = bool(st.fn(self, strict))
            ran.append(st.name)
            if not ok:
                failed.append(st.name)
                if strict:
                    break
        self.noise, self.rng = None, None
        return dict(ok=not failed, stages=ran, failed=failed, failed_stage=failed[0] if failed else None, faults=list(self.faults))

    def progress(self) -> dict:
        return self.task.graph.infer(self.contract, self.traces, self.task.aux(self))

    # ---------------------------------------------------------------- perturbation primitives (engine.PRIMITIVES)
    def objects(self) -> list[str]:
        return list(self.task.movable) + list(self.task.target_names)

    def holding(self) -> dict:
        return {i: self.task.held_object(self, i) for i in range(self.n_arms)}

    def set_gripper(self, arm, cmd, steps=6):
        arm = int(arm); self.targets[arm] = self.qpos(arm)[:7].copy(); self.grip[arm] = float(cmd); self.hold(steps)

    def joint_noise(self, arm, sigma, steps, seed):
        arm = int(arm); r = np.random.default_rng(int(seed))
        lim = self.agents[arm].robot.get_qlimits()[0, :7].cpu().numpy()
        q = self.targets[arm].copy(); inc = float(sigma) / np.sqrt(int(steps))
        for _ in range(int(steps)):
            q = np.clip(q + r.normal(0, inc, 7), lim[:, 0], lim[:, 1]); self.targets[arm] = q; self._step()

    def shove(self, obj, vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0, settle=15):
        import torch
        a = self.task.actor(self, obj)
        a.set_linear_velocity(torch.tensor([[vx, vy, vz]], dtype=torch.float32))
        a.set_angular_velocity(torch.tensor([[wx, wy, wz]], dtype=torch.float32))
        self.hold(settle)

    def offset_pose(self, obj, dx=0.0, dy=0.0, dz=0.0, dyaw_deg=0.0, settle=5):
        import sapien
        from transforms3d.euler import euler2quat
        from transforms3d.quaternions import qmult
        a = self.task.actor(self, obj); p = a.pose.sp
        newp = sapien.Pose(np.asarray(p.p) + np.array([dx, dy, dz]), qmult(euler2quat(0, 0, np.deg2rad(dyaw_deg)), p.q))
        (a.set_root_pose if hasattr(a, "set_root_pose") else a.set_pose)(newp)
        self.task.after_offset(self, obj)
        self.hold(settle)

    def move_tcp_rel(self, arm, dx=0.0, dy=0.0, dz=0.0):
        import sapien
        t = self.tcp(int(arm)); self.move_to(int(arm), sapien.Pose(np.asarray(t.p) + np.array([dx, dy, dz]), t.q))

    # ---------------------------------------------------------------- snapshots
    def snapshot(self, stage: str, source: str, **meta) -> Snapshot:
        m = dict(objects=list(self.task.movable), holding=self.holding(), targets=self.task.free_targets(self),
                 all_targets=list(self.task.target_names), active_arms=self.task.active_arms(stage))
        m.update(meta)
        lg = {k: list(v) for k, v in self.log.items() if k != "obj"}  # (lists of arrays: shallow copies are enough)
        lg["obj"] = {k: list(v) for k, v in self.log["obj"].items()}
        lg["stage_names"] = list(self.stage_names)
        return Snapshot(task=self.task_id, seed=self.seed, step=self.step_count, stage=stage, source=source,
                        sim_state=to_np(self.u.get_state_dict()), task_state=copy.deepcopy(self.task.get_task_state(self)),
                        ctl=dict(targets=[t.copy() for t in self.targets], grip=list(self.grip)),
                        traces=copy.deepcopy(self.rec.traces), log=lg, ctx=copy.deepcopy(self.ctx), meta=m)

    def restore(self, snap: Snapshot):
        st = {k: v for k, v in snap.sim_state.items() if k in ("actors", "articulations")}
        self.u.set_state_dict(to_torch(st))
        self.task.set_task_state(self, copy.deepcopy(snap.task_state))
        self.targets = [np.asarray(t, float).copy() for t in snap.ctl["targets"]]; self.grip = list(snap.ctl["grip"])
        self.rec.traces = copy.deepcopy(snap.traces)
        lg = snap.log
        self.log = {k: list(v) for k, v in lg.items() if k not in ("obj", "stage_names")}
        self.log["obj"] = {k: list(v) for k, v in lg["obj"].items()}
        self.stage_names = list(lg["stage_names"])
        self.step_count, self.seed, self.stage, self.ctx = snap.step, snap.seed, snap.stage, copy.deepcopy(snap.ctx)
        self.segment = SEG_PREFIX

    def episode_arrays(self) -> dict:
        lg = self.log
        out = {"action": np.asarray(lg["action"], np.float32), "observation.state": np.asarray(lg["qpos"], np.float32),
               "segment": np.asarray(lg["segment"], np.int8), "stage": np.asarray(lg["stage"], np.int16),
               "stage_names": np.asarray(self.stage_names)}
        for k, v in lg["obj"].items():
            out[f"obj.{k}"] = np.asarray(v, np.float32)
        for k, v in self.rec.traces.items():
            out[f"trace.{k}"] = np.asarray(v, np.float32)
        if lg.get("image"):
            out["observation.image"] = np.stack(lg["image"]); out["observation.image_step"] = np.asarray(lg["image_step"])
        return out
