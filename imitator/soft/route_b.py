"""Route B: everything in Newton. Panda simulated by Newton's SolverMuJoCo (MuJoCo-Warp backend, from the same
panda_v2.urdf ManiSkill uses), towel by NewtonClothWorlds (VBD), wrapped in a ManiSkill-like gym API.

Two Newton models are stepped in lockstep in one process on one GPU: the robot model (MuJoCo-Warp; robot + table plane)
and the cloth model (VBD; towel + kinematic copies of the hand/finger links). Each control step the robot advances
`robot_substeps` MuJoCo steps, then its link poses are copied (GPU, no host round trip) into the cloth model, exactly as
route A copies SAPIEN link poses: the coupling (one-way, kinematic, friction grasp) is identical in both routes, so the
comparison isolates the robot simulator and the renderer. Newton's own two-way path (SolverCoupledProxy, experimental)
is not used (issue #4698: proxy wrenches ~100x too large with VBD).
Robot gravity is off (ideal gravity compensation), as ManiSkill does for its PD-controlled arms (links have
disable_gravity=True in BaseAgent); joint drives: stiffness 1e3, damping 1e2 (ManiSkill Panda arm/gripper values);
gripper close target clamped at cloth_q_close (as PandaCloth in route A).
Control modes: pd_joint_pos (7 absolute arm targets + gripper in [-1, 1]); pd_ee_pose (TCP xyz + euler XYZ in the robot
base frame + gripper; solved each step by Newton's batched GPU IK, newton.ik).
"""
from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import newton

from .cloth import ClothSolverSpec, NewtonClothWorlds, TowelSpec
from .colliders import PANDA_COLLIDERS, PANDA_PADS
from .metrics import footprint_area, speed_quantile
from .panda_ik import PANDA_REST, PANDA_URDF

RELEASE_IMPULSE_NS = 5e-4
ENCLOSURE_MARGIN_M = 0.002


@wp.kernel
def _gather_link_poses(body_q: wp.array[wp.transform], idx: wp.array[int], bpw: int, k: int,
                       out: wp.array2d[float]):
    """out[w*k + j] = [p, q_wxyz] of body idx[j] of world w"""
    t = wp.tid()
    w = t // k
    j = t - w * k
    tf = body_q[w * bpw + idx[j]]
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)
    out[t, 0] = p[0]; out[t, 1] = p[1]; out[t, 2] = p[2]
    out[t, 3] = q[3]; out[t, 4] = q[0]; out[t, 5] = q[1]; out[t, 6] = q[2]


class NewtonPandaWorlds:
    """N Panda arms (one per world), MuJoCo-Warp, joint position drives."""

    def __init__(self, num_worlds: int, base=(-0.615, 0.0, 0.0), control_dt: float = 0.05, substeps: int = 10,
                 q_close: float = 0.002, arm_ke: float = 1e3, arm_kd: float = 1e2, finger_ke: float = 1e3,
                 finger_kd: float = 1e2, device: str = "cuda:0", urdf: str = PANDA_URDF, use_graph: bool = True):
        self.n, self.control_dt, self.substeps = int(num_worlds), control_dt, substeps
        self.q_close, self.device = q_close, wp.get_device(device)
        b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        b.add_urdf(urdf, xform=wp.transform(wp.vec3(*base), wp.quat_identity()), floating=False, hide_visuals=True,
                   enable_self_collisions=False, collapse_fixed_joints=False)
        assert b.joint_dof_count == 9, b.joint_dof_count
        for i in range(9):
            b.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)
            b.joint_target_ke[i] = arm_ke if i < 7 else finger_ke
            b.joint_target_kd[i] = arm_kd if i < 7 else finger_kd
        q0 = np.r_[PANDA_REST, 0.04, 0.04]
        for i in range(9):
            b.joint_q[i] = float(q0[i])
        self.bpw = b.body_count
        labels = list(getattr(b, "body_label", None) or getattr(b, "body_key"))
        self.body_names = [l.split("/")[-1] for l in labels]
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.replicate(b, world_count=self.n)
        scene.add_ground_plane()                      # the table top (z = 0); the robot base stands on it
        with wp.ScopedDevice(self.device):
            self.model = scene.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=True, njmax=200, nconmax=100)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.ctrl = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.s0)
        self.q_rest = torch.tensor(q0, dtype=torch.float32, device="cuda")
        self.target = wp.to_torch(self.ctrl.joint_target_q).view(self.n, 9)
        self.target[:] = self.q_rest
        self.use_graph = use_graph and self.device.is_cuda
        self.graph = None
        self._idx_cache: dict = {}

    def _sub(self):
        dt = self.control_dt / self.substeps
        for _ in range(self.substeps):
            self.s0.clear_forces()
            self.solver.step(self.s0, self.s1, self.ctrl, None, dt)
            self.s0, self.s1 = self.s1, self.s0
        if self.substeps % 2 == 1:
            for a in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(self.s1, a), getattr(self.s0, a))
            self.s0, self.s1 = self.s1, self.s0

    def step(self):
        if self.use_graph:
            if self.graph is None:
                self._sub()
                wp.synchronize_device(self.device)
                with wp.ScopedCapture(device=self.device, force_module_load=False) as cap:
                    self._sub()
                self.graph = cap.graph
            else:
                wp.capture_launch(self.graph)
        else:
            self._sub()

    def qpos(self) -> torch.Tensor:
        return wp.to_torch(self.s0.joint_q).view(self.n, 9)

    def qvel(self) -> torch.Tensor:
        return wp.to_torch(self.s0.joint_qd).view(self.n, 9)

    def set_qpos(self, q: torch.Tensor, mask: torch.Tensor):
        for s in (self.s0, self.s1):
            jq = wp.to_torch(s.joint_q).view(self.n, 9); jqd = wp.to_torch(s.joint_qd).view(self.n, 9)
            jq[mask] = q[mask]; jqd[mask] = 0.0
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        wp.copy(self.s1.body_q, self.s0.body_q)

    def link_poses(self, names: list[str]) -> wp.array:
        key = tuple(names)
        if key not in self._idx_cache:
            idx = wp.array([self.body_names.index(nm) for nm in names], dtype=int, device=self.device)
            self._idx_cache[key] = (idx, wp.zeros((self.n * len(names), 7), dtype=float, device=self.device))
        idx, out = self._idx_cache[key]
        wp.launch(_gather_link_poses, dim=self.n * len(names), inputs=[self.s0.body_q, idx, self.bpw, len(names)],
                  outputs=[out], device=self.device)
        return out


class FoldTowelNewtonEnv:
    """ManiSkill-like vectorised env (gymnasium API, batched torch tensors) for the diagonal towel fold, route B."""

    metadata = {"render_modes": ["rgb_array"]}
    towel_center = (-0.05, 0.0)

    def __init__(self, num_envs: int = 1, control_mode: str = "pd_joint_pos", obs_mode: str = "state",
                 towel: TowelSpec | None = None, cloth_solver: ClothSolverSpec | None = None, control_freq: int = 20,
                 robot_substeps: int = 10, max_episode_steps: int = 200, render_mode: str | None = None,
                 camera=None, device: str = "cuda:0"):
        t0 = time.time()
        self.num_envs, self.control_mode, self.obs_mode = num_envs, control_mode, obs_mode
        self.control_freq = control_freq
        self.device = torch.device(device)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.towel_spec = towel or TowelSpec(center=self.towel_center)
        self.cloth_spec = cloth_solver or ClothSolverSpec()
        self.robot = NewtonPandaWorlds(num_envs, control_dt=1.0 / control_freq, substeps=robot_substeps, device=device)
        colliders = PANDA_COLLIDERS
        if camera:                       # robot visual meshes go into the cloth model as visual-only kinematic bodies
            from .render_sync import panda_render_links
            colliders = panda_render_links(PANDA_COLLIDERS)
        self.link_names = [c.link for c in colliders]
        self.cloth = NewtonClothWorlds(num_envs, colliders, PANDA_PADS, self.towel_spec, self.cloth_spec,
                                       control_dt=1.0 / control_freq, device=device)
        self.camera = None
        if camera:
            from .render_sync import NewtonCamera
            self.camera = NewtonCamera(self.cloth, **(camera if isinstance(camera, dict) else {}))
        self._ik = None
        self._elapsed = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._last_qpos = None
        self._robot_qvel_max = torch.zeros(num_envs, device=self.device)
        self._init_area = None
        nact = 8 if control_mode == "pd_joint_pos" else 7
        self.single_action_space = gym.spaces.Box(-np.inf, np.inf, (nact,), np.float32)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (num_envs, nact), np.float32)
        self.build_s = time.time() - t0

    # ------------------------------------------------------------------ gym API
    @property
    def unwrapped(self):
        return self

    def reset(self, seed: int | None = None, options: dict | None = None):
        options = options or {}
        env_idx = options.get("env_idx")
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if env_idx is None:
            mask[:] = True
        else:
            mask[torch.as_tensor(env_idx, device=self.device)] = True
        q = self.robot.q_rest.repeat(self.num_envs, 1)
        self.robot.set_qpos(q, mask)
        self.robot.target[mask] = q[mask]
        self.cloth.reset(mask.cpu().numpy().astype(np.int32))
        self.cloth.set_link_poses(self.robot.link_poses(self.link_names), first=True)
        self.cloth._signals()
        qq = self.cloth.particle_q()
        self._init_area = footprint_area(qq[..., :2]).clone()
        self._elapsed[mask] = 0
        self._last_qpos = self.robot.qpos().clone()
        self._robot_qvel_max = torch.zeros(self.num_envs, device=self.device)
        info = self.evaluate()
        return self.get_obs(info), info

    def _apply_action(self, action: torch.Tensor):
        a = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
        g = (a[:, -1].clamp(-1, 1) + 1) / 2 * (0.04 - self.robot.q_close) + self.robot.q_close
        if self.control_mode == "pd_joint_pos":
            qa = a[:, :7]
        elif self.control_mode == "pd_ee_pose":
            qa = self._solve_ik(a[:, :6])
        else:
            raise ValueError(self.control_mode)
        self.robot.target[:, :7] = qa
        self.robot.target[:, 7] = g
        self.robot.target[:, 8] = g

    def step(self, action):
        if action is not None:
            self._apply_action(action)
        self.robot.step()
        self.cloth.set_link_poses(self.robot.link_poses(self.link_names))
        if self.cloth_spec.grasp_mode == "attach":
            self.cloth.set_grip(self.robot.qpos()[:, 7] < self.robot.q_close + 0.0015)
        self.cloth.step()
        q = self.robot.qpos()
        self._robot_qvel_max = (q[:, :7] - self._last_qpos[:, :7]).abs().amax(1) * self.control_freq
        self._last_qpos = q.clone()
        self._elapsed += 1
        info = self.evaluate()
        obs = self.get_obs(info)
        rew = 1.0 - torch.tanh(5.0 * info["towel_corner_dist"]) + info["success"].float()
        terminated = info["success"].clone()
        truncated = self._elapsed >= self.max_episode_steps
        return obs, rew, terminated, truncated, info

    def hold_robot(self):
        """Protocol v3 hold A: arm targets := joint positions reached; gripper keeps its last target."""
        self.robot.target[:, :7] = self.robot.qpos()[:, :7]

    # ------------------------------------------------------------------ task
    def cloth_signals(self) -> dict[str, torch.Tensor]:
        s = {k: v.to(self.device) for k, v in self.cloth.signals().items()}
        qq = self.cloth.particle_q()
        s["towel_footprint_ratio"] = footprint_area(qq[..., :2]) / self._init_area
        s["cloth_speed_p95"] = speed_quantile(self.cloth.particle_qd())
        s["robot_qvel_max"] = self._robot_qvel_max
        tcp = wp.to_torch(self.robot.link_poses(["panda_hand_tcp"])).view(self.num_envs, 7)
        s["tcp_x"], s["tcp_y"], s["tcp_z"] = tcp[:, 0].clone(), tcp[:, 1].clone(), tcp[:, 2].clone()
        s["finger_q"] = self.robot.qpos()[:, 7].clone()
        s["released"] = ((s["robot_cloth_impulse"] <= RELEASE_IMPULSE_NS)
                         & (s["finger_cloth_enclosure"] > ENCLOSURE_MARGIN_M)).float()
        return s

    def evaluate(self) -> dict:
        s = self.cloth_signals()
        folded = (s["towel_corner_dist"] < 0.08) & (s["towel_footprint_ratio"] < 0.70) & (s["towel_frac_above_5cm"] <= 0.02)
        success = folded & (s["released"] > 0.5) & (s["cloth_nonfinite"] == 0)
        return dict(success=success, folded=folded, fail=s["cloth_nonfinite"] > 0, elapsed_steps=self._elapsed.clone(), **s)

    def get_obs(self, info: dict | None = None) -> dict:
        r = self.towel_spec.res + 1
        idx = [0, r - 1, r * r - r, r * r - 1, (r * r) // 2]
        qq = self.cloth.particle_q()
        tcp = wp.to_torch(self.robot.link_poses(["panda_hand_tcp"])).view(self.num_envs, 7)
        obs = dict(agent=dict(qpos=self.robot.qpos().clone(), qvel=self.robot.qvel().clone()),
                   extra=dict(tcp_pose=tcp.clone(), cloth_keypoints=qq[:, idx, :].reshape(self.num_envs, -1).clone()))
        if self.camera is not None and self.obs_mode in ("rgb", "rgbd"):
            obs["sensor_data"] = self.camera.render_all(self)
        return obs

    def render(self):
        if self.camera is None:
            return None
        return self.camera.render_all(self)

    def close(self):
        pass

    # ------------------------------------------------------------------ IK for pd_ee_pose
    def _solve_ik(self, pose6: torch.Tensor) -> torch.Tensor:
        import newton.ik as ik
        if self._ik is None:
            b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            b.add_urdf(PANDA_URDF, floating=False, hide_visuals=True, enable_self_collisions=False,
                       collapse_fixed_joints=False)
            m = b.finalize(device=str(self.robot.device))
            names = [l.split("/")[-1] for l in (getattr(b, "body_label", None) or getattr(b, "body_key"))]
            tcp = names.index("panda_hand_tcp")
            n = self.num_envs
            pos = ik.IKObjectivePosition(link_index=tcp, link_offset=wp.vec3(0.0, 0.0, 0.0),
                                         target_positions=wp.zeros(n, dtype=wp.vec3, device=m.device))
            rot = ik.IKObjectiveRotation(link_index=tcp, link_offset_rotation=wp.quat_identity(),
                                         target_rotations=wp.zeros(n, dtype=wp.vec4, device=m.device))
            lim = ik.IKObjectiveJointLimit(joint_limit_lower=m.joint_limit_lower, joint_limit_upper=m.joint_limit_upper,
                                           weight=10.0)
            q = wp.zeros((n, m.joint_coord_count), dtype=float, device=m.device)
            solver = ik.IKSolver(model=m, n_problems=n, objectives=[pos, rot, lim], lambda_initial=0.1,
                                 jacobian_mode=ik.IKJacobianType.ANALYTIC)
            self._ik = (m, pos, rot, q, solver)
        m, pos, rot, q, solver = self._ik
        from scipy.spatial.transform import Rotation
        p = pose6[:, :3].detach().cpu().numpy()
        quat = Rotation.from_euler("XYZ", pose6[:, 3:6].detach().cpu().numpy()).as_quat()   # xyzw
        pos.target_positions.assign(p.astype(np.float32))
        rot.target_rotations.assign(quat.astype(np.float32))
        wp.to_torch(q)[:] = self.robot.qpos()
        solver.step(q, q, iterations=16)
        return wp.to_torch(q)[:, :7].clone()


def scripted_joint_action(env: FoldTowelNewtonEnv, q_arm: torch.Tensor, grip: torch.Tensor) -> torch.Tensor:
    return torch.cat([q_arm.to(env.device).float(), (1.0 - 2.0 * grip.to(env.device).float())[:, None]], dim=1)
