"""Batched (GPU PhysX, N envs) recorder for the dual cube task under protocol v3 / v3.1.

Same signals as the CPU recorders (imitator.verify.smoke / v31), per env:
  cube_x / _y / _z, cube_speed, cube_target_margin_m, cube_bottom_dz_m
  robot_cube_impulse_Ns (+ _cum_Ns)  |impulse| between the cube and every collision link of both robots, summed over the
                                     links and over all physics substeps of the control step. GPU: one PhysX contact-pair
                                     impulse query (all links x all envs) read after every substep (the per-pair net
                                     impulse vector, the GPU counterpart of the CPU per-body-pair vector sum)
  cube_enclosure_clearance_m        v3 finger-side rule, cube_enclosure_all_m  v3.1 all-link rule (imitator.verify.v31)
  robot_qvel_max                    largest arm-joint speed of both robots from joint positions over the control step
  bench_success, phase
Hold A: the batched executor is in pd_joint_pos; settle() sends the arm joint positions reached at the policy's last step
plus the last gripper command (the same hold as veritask's hold_action_A, which switches every arm to pd_joint_pos).
"""
from __future__ import annotations

import numpy as np
import torch

from . import v31

NOT_ENCLOSED_M = v31.NOT_ENCLOSED_M


class BatchedRecorder:
    def __init__(self, env, obj_attr: str = "cube", half_attr: str = "cube_half"):
        self.env = env
        self.u = u = env.unwrapped
        self.N = u.num_envs
        self.obj_attr, self.half_attr = obj_attr, half_attr
        self.agents = list(u.agent.agents)
        self.links = [l for ag in self.agents for l in ag.robot.links if l._bodies[0].get_collision_shapes()]
        self.arm_idx = [[i for i, j in enumerate(ag.robot.active_joints) if j.name in ag.record.arm_joints] for ag in self.agents]
        self.need = [v31.needed_links(ag) for ag in self.agents]
        self.query = None
        self._hooked = False
        self._settling = False
        self.traces: list[dict] = []
        self.timing = dict(substep_query_s=0.0, probe_s=0.0, sim_s=0.0)

    @property
    def obj(self):
        return getattr(self.u, self.obj_attr)

    # ---------------------------------------------------------------- contact query
    def _build_query(self):
        px = self.u.scene.px
        cube = self.obj
        pairs = [(l._bodies[e], cube._bodies[e]) for l in self.links for e in range(self.N)]
        self.query = px.gpu_create_contact_pair_impulse_query(pairs)

    def _install(self):
        if self._hooked:
            return
        u = self.u
        orig = u._after_simulation_step

        def hook():
            orig()
            self._on_substep()
        u._after_simulation_step = hook
        self._hooked = True

    def _on_substep(self):
        px = self.u.scene.px
        px.gpu_query_contact_pair_impulses(self.query)
        imp = self.query.cuda_impulses.torch().reshape(len(self.links), self.N, 3)
        self._acc += torch.linalg.norm(imp, dim=-1).sum(0)

    # ---------------------------------------------------------------- gym-like API
    def reset(self, seed=None, options=None):
        out = self.env.reset(seed=seed, options=options)
        if self.query is None:
            self._build_query()
        self._install()
        self.traces = [dict() for _ in range(self.N)]
        self._q_prev = [ag.robot.get_qpos().clone() for ag in self.agents]
        self.last_action = None
        return out

    def step(self, action):
        import time
        self._acc = torch.zeros(self.N, device=self.u.device)
        t0 = time.time()
        out = self.env.step(action)
        t1 = time.time()
        info = out[-1]
        self._record(info)
        self.timing["sim_s"] += t1 - t0
        self.timing["probe_s"] += time.time() - t1
        if not self._settling:
            self.last_action = {k: v.clone() if hasattr(v, "clone") else np.array(v, copy=True) for k, v in action.items()}
        return out

    def _record(self, info):
        u, N = self.u, self.N
        obj = self.obj
        half = getattr(u, self.half_attr)
        p = obj.pose.p
        T = obj.pose.to_transformation_matrix().cpu().numpy().astype(float)
        speed = torch.linalg.norm(obj.linear_velocity, dim=-1)
        d = (p[:, :2] - u.target_pos[:, :2]).abs()
        margin = u.dual.target_half - d.max(-1).values
        bottom = u.cube_bottom_z()
        v = torch.zeros(N, device=u.device)
        for k, ag in enumerate(self.agents):
            q = ag.robot.get_qpos()
            v = torch.maximum(v, (q[:, self.arm_idx[k]] - self._q_prev[k][:, self.arm_idx[k]]).abs().amax(1) / float(u.control_timestep))
            self._q_prev[k] = q.clone()
        # link transforms of all envs, one host copy per link
        Ts = []
        for ag, names in zip(self.agents, self.need):
            lm = ag.robot.links_map
            Ts.append({n: lm[n].pose.to_transformation_matrix().cpu().numpy().astype(float) for n in names if n in lm})
        tcp = [ag.tcp.pose.p.cpu().numpy().astype(float) for ag in self.agents]
        cols = dict(cube_x=p[:, 0], cube_y=p[:, 1], cube_z=p[:, 2], cube_speed=speed, cube_target_margin_m=margin,
                    cube_bottom_dz_m=bottom, robot_qvel_max=v, robot_cube_impulse_Ns=self._acc,
                    bench_success=info["success"].float())
        host = {k: t.detach().float().cpu().numpy().astype(float) for k, t in cols.items()}
        phase = 1.0 if self._settling else 0.0
        for e in range(N):
            tr = self.traces[e]
            for k, a in host.items():
                tr.setdefault(k, []).append(float(a[e]))
            cum = tr.get("robot_cube_impulse_cum_Ns")
            tr.setdefault("robot_cube_impulse_cum_Ns", []).append((cum[-1] if cum else 0.0) + float(host["robot_cube_impulse_Ns"][e]))
            Te = [{n: M[e] for n, M in Td.items()} for Td in Ts]
            v3 = NOT_ENCLOSED_M
            for k, (ag, Td) in enumerate(zip(self.agents, Te)):
                if v31.too_far(ag.record, tcp[k][e], T[e], half):
                    continue
                if ag.record.pad_links:
                    c = v31.pad_clearance(ag.record, Td, T[e], half)
                else:
                    c = v31.hand_clearance(ag.record, Td, T[e], half, with_palm=False)
                if c is not None:
                    v3 = min(v3, c)
            tr.setdefault("cube_enclosure_clearance_m", []).append(float(v3))
            tr.setdefault("cube_enclosure_all_m", []).append(v31.enclosure_all_value(self.agents, Te, T[e], half,
                                                                                    tcp_ps=[tcp[k][e] for k in range(len(self.agents))]))
            tr.setdefault("phase", []).append(phase)

    def n_steps(self) -> int:
        return len(self.traces[0].get("phase", []))
