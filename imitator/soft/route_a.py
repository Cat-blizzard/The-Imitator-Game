"""Route A: ManiSkill 3 / SAPIEN (PhysX, CPU or GPU sim) robot + Newton (VBD) towel, one-way bridged.

Per control step (env.step):
  1. ManiSkill applies the action and runs its PhysX substeps (robot, table) as usual;
  2. the bridge reads the poses of the robot links listed in `colliders` (hand, two fingers) from SAPIEN (GPU buffers, no
     host copy on the GPU backend) and hands them to NewtonClothWorlds, which advances the towels of all envs one control
     step with those links as kinematic colliders (no force back to PhysX; the neglected cloth->robot impulse is reported);
  3. evaluate() reads the per-world cloth signals.
Grasping is contact friction (default) or the documented attachment rule (ClothSolverSpec.grasp_mode='attach'); the
PhysX fingers stop at PandaCloth.cloth_q_close where the pinched cloth layer would stop them.
Rendering: ManiSkill's GPU-sim renderer needs a CUDA-visible Vulkan device (absent here); see render_sync.py for the
CPU (lavapipe) SAPIEN mirror that shows the cloth mesh, and the Newton Warp camera.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

import mani_skill.envs.sapien_env as _se
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.system import backend as _bk
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from ..envs.stock_table import StockTableSceneBuilder as TableSceneBuilder

from . import maniskill_agent  # noqa: F401  (registers 'panda_cloth')
from .cloth import ClothSolverSpec, NewtonClothWorlds, TowelSpec
from .colliders import PANDA_COLLIDERS, PANDA_PADS
from .metrics import footprint_area, speed_quantile

RELEASE_IMPULSE_NS = 5e-4      # robot-cloth contact impulse per control step that still counts as "no contact" [N s]
ENCLOSURE_MARGIN_M = 0.002


def _patch_pci_render_backend():
    """ManiSkill 3.0.1 splits render_backend on ':' so 'pci:0' (the lavapipe device on hosts without NVIDIA graphics)
    cannot be selected; wrap its parser at runtime (no source change)."""
    if getattr(_se, "_imsim_pci_patch", False):
        return
    orig = _bk.parse_sim_and_render_backend

    def patched(sim_backend, render_backend):
        if isinstance(render_backend, str) and render_backend.startswith("pci"):
            info = orig(sim_backend, "none")
            info.render_device = sapien.Device(render_backend)
            info.render_backend = "sapien_pci"
            return info
        return orig(sim_backend, render_backend)

    _se.parse_sim_and_render_backend = patched
    _se._imsim_pci_patch = True


_patch_pci_render_backend()


@register_env("FoldTowelBridge-v1", max_episode_steps=200)
class FoldTowelBridgeEnv(BaseEnv):
    """Diagonal towel fold, single Panda; towel simulated by Newton VBD (route A)."""

    SUPPORTED_ROBOTS = ["panda_cloth"]
    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]
    towel_center = (-0.05, 0.0)
    arm_dofs = 7

    def __init__(self, *args, robot_uids="panda_cloth", towel: TowelSpec | None = None,
                 cloth_solver: ClothSolverSpec | None = None, robot_init_qpos_noise: float = 0.0,
                 newton_camera: bool | dict = False, **kwargs):
        self.newton_camera_cfg = newton_camera
        self.newton_camera = None
        self.towel_spec = towel or TowelSpec(center=self.towel_center)
        self.cloth_spec = cloth_solver or ClothSolverSpec()
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cloth: NewtonClothWorlds | None = None
        self._last_qpos = None
        self._robot_qvel_max = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------ scene
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.35, -0.45, 0.45], target=[-0.05, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.45, -0.55, 0.55], target=[-0.05, 0.0, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        colliders = PANDA_COLLIDERS
        if self.newton_camera_cfg:       # optional GPU renderer: robot visual meshes mirrored into the cloth model
            from .render_sync import panda_render_links
            colliders = panda_render_links(PANDA_COLLIDERS)
        if self.cloth is None:
            dev = self.device if self.device.type == "cuda" else torch.device("cuda:0")
            self.cloth = NewtonClothWorlds(self.num_envs, colliders, PANDA_PADS, self.towel_spec, self.cloth_spec,
                                           control_dt=1.0 / self.control_freq, device=str(dev))
            if self.newton_camera_cfg:
                from .render_sync import NewtonCamera
                cfg = self.newton_camera_cfg if isinstance(self.newton_camera_cfg, dict) else {}
                self.newton_camera = NewtonCamera(self.cloth, **cfg)
        self._links = [self.agent.robot.links_map[c.link] for c in colliders]
        self._init_area = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)
        # TableSceneBuilder only knows the stock uids; put panda_cloth in the Panda 'rest' keyframe like PickCube does
        b = len(env_idx)
        qpos = torch.tensor(self.agent.keyframes["rest"].qpos, dtype=torch.float32, device=self.device).repeat(b, 1)
        if self.robot_init_qpos_noise > 0:
            qpos[:, :-2] += torch.randn_like(qpos[:, :-2]) * self.robot_init_qpos_noise
        qpos[:, -2:] = 0.04
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        mask = np.zeros(self.num_envs, np.int32)
        mask[env_idx.cpu().numpy()] = 1
        self.cloth.reset(mask)

    # ------------------------------------------------------------------ bridge
    def link_poses(self) -> torch.Tensor:
        """(N, K, 7) [p, q_wxyz] of the collider links in each env's frame."""
        return torch.stack([l.pose.raw_pose for l in self._links], dim=1).to("cuda:0")

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.cloth.set_link_poses(self.link_poses(), first=True)
        self.cloth._signals()
        q = self.cloth.particle_q()
        self._init_area = footprint_area(q[..., :2]).clone()
        self._last_qpos = self.agent.robot.get_qpos().clone()
        self._robot_qvel_max = torch.zeros(self.num_envs, device=self.device)
        info = self.get_info()
        obs = self.get_obs(info)
        return obs, info

    def _step_action(self, action):
        action = super()._step_action(action)
        self.cloth.set_link_poses(self.link_poses())
        if self.cloth_spec.grasp_mode == "attach":   # attachment rule input: fingers physically closed on the cloth
            self.cloth.set_grip(self.agent.robot.get_qpos()[:, 7] < self.agent.cloth_q_close + 0.0015)
        self.cloth.step()
        qpos = self.agent.robot.get_qpos()
        if self._last_qpos is not None:
            self._robot_qvel_max = ((qpos[:, : self.arm_dofs] - self._last_qpos[:, : self.arm_dofs]).abs().amax(1)
                                    * self.control_freq)
        self._last_qpos = qpos.clone()
        return action

    def hold_robot(self):
        """Protocol v3 hold A: hold the arm at the joint positions it has reached, keep the gripper's last target."""
        ctrl = self.agent.controller
        arm = ctrl.controllers["arm"]
        q = self.agent.robot.get_qpos()[:, arm.active_joint_indices]
        arm._target_qpos = q.clone()
        arm.set_drive_targets(q)
        if self.gpu_sim_enabled:
            self.scene.px.gpu_apply_articulation_target_position()

    # ------------------------------------------------------------------ task
    def cloth_signals(self) -> dict[str, torch.Tensor]:
        s = {k: v.to(self.device) for k, v in self.cloth.signals().items()}
        q = self.cloth.particle_q()
        if self._init_area is not None:
            s["towel_footprint_ratio"] = (footprint_area(q[..., :2]) / self._init_area).to(self.device)
        s["cloth_speed_p95"] = speed_quantile(self.cloth.particle_qd()).to(self.device)
        s["robot_qvel_max"] = (self._robot_qvel_max if self._robot_qvel_max is not None
                               else torch.zeros(self.num_envs, device=self.device))
        tcp = self.agent.tcp_pose.p
        s["tcp_x"], s["tcp_y"], s["tcp_z"] = tcp[:, 0], tcp[:, 1], tcp[:, 2]
        s["finger_q"] = self.agent.robot.get_qpos()[:, 7]
        s["released"] = ((s["robot_cloth_impulse"] <= RELEASE_IMPULSE_NS)
                         & (s["finger_cloth_enclosure"] > ENCLOSURE_MARGIN_M)).float()
        return s

    def evaluate(self) -> dict:
        if self.cloth is None:
            return {}
        s = self.cloth_signals()
        foot = s.get("towel_footprint_ratio", torch.ones(self.num_envs, device=self.device))
        folded = (s["towel_corner_dist"] < 0.08) & (foot < 0.70) & (s["towel_frac_above_5cm"] <= 0.02)
        success = folded & (s["released"] > 0.5) & (s["cloth_nonfinite"] == 0)
        return dict(success=success, folded=folded, fail=s["cloth_nonfinite"] > 0, **s)

    def _get_obs_extra(self, info: dict):
        q = self.cloth.particle_q().to(self.device)
        r = self.towel_spec.res + 1
        idx = [0, r - 1, r * r - r, r * r - 1, (r * r) // 2]          # four corners + centre
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(cloth_keypoints=q[:, idx, :].reshape(self.num_envs, -1))
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return 1.0 - torch.tanh(5.0 * info["towel_corner_dist"]) + info["success"].float()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 2.0


def scripted_action(env: FoldTowelBridgeEnv, tcp_xyz: torch.Tensor, grip: torch.Tensor) -> torch.Tensor:
    """pd_ee_pose action for a TCP position (env frame), gripper pointing down, grip in [0, 1] (1 = closed)."""
    base = env.agent.robot.pose.p                        # (N, 3) robot root in the env frame (identity rotation)
    n = tcp_xyz.shape[0]
    a = torch.zeros(n, 7, device=env.device)
    a[:, :3] = tcp_xyz.to(env.device) - base
    a[:, 3] = np.pi                                       # euler XYZ (pi, 0, 0): hand z axis down
    a[:, 6] = 1.0 - 2.0 * grip.to(env.device)
    return a


def scripted_joint_action(env: FoldTowelBridgeEnv, q_arm: torch.Tensor, grip: torch.Tensor) -> torch.Tensor:
    """pd_joint_pos action: absolute arm joint targets + gripper (grip 1 = closed -> -1 = PandaCloth.cloth_q_close)."""
    return torch.cat([q_arm.to(env.device).float(), (1.0 - 2.0 * grip.to(env.device).float())[:, None]], dim=1)
