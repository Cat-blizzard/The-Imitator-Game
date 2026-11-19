"""DualFoldTowel-v1: dual-arm half fold of a Newton (VBD) towel, any parallel-gripper dual configuration (E20 route A
bridge: ManiSkill / SAPIEN PhysX robots -> Newton cloth, one-way, per control step).

  towel     30 x 30 cm, 25 x 25 vertices, 0.15 kg (E20 TowelSpec), centred at towel_center on the table (z = 0)
            corners (world frame, x forward, y left): NR = near-right (vertex 0), FR = far-right (vertex res),
            NL = near-left (vertex res*(res+1)), FL = far-left (last vertex)
  task      the left arm grasps the far-left corner FL and folds it onto the near-left corner NL, the right arm FR onto
            NR (fold line x = towel centre), both release
  robots    DualConfig mount (imitator.embodiments.dual); every gripper link is a kinematic Newton collider (oriented
            bounding box of its URDF collision geometry, imitator.soft.embodiment_colliders); the PhysX fingers stop at the
            cloth layer: the gripper 'cloth close' command leaves a 4 mm pad gap (cloth_close_command)
  signals   cloth_signals(): per world, corner distances towel_dist_L = |FL - NL|_xy, towel_dist_R = |FR - NR|_xy, corner
            positions, towel_lift_L / _R = highest vertex of the 3 x 3 vertex patch at the grasped corner, towel_frac_above_5cm,
            robot_cloth_impulse (all colliders) and robot_cloth_impulse_0 / _1 (left / right arm), finger_cloth_enclosure
            (smallest over both grippers) and _0 / _1, cloth_speed_p95 (VBD particle velocities = position change over the
            last VBD substep, 1/600 s), cloth_speed_p95_pos / cloth_speed_max_pos (E29b: from particle POSITIONS over the
            control step, the v3.2 rule), towel_cx / _cy / _cz (centroid of all particles, E29b), robot_qvel_max (both
            arms), released
  success   benchmark-style flag (E20 FoldTowel style, one step): both corner distances < 8 cm, footprint < 0.70, <= 2 %
            above 5 cm, released, no NaN. It does not check that the corners were lifted (a drag can pass).
  render    rgb_source='newton': Newton Warp ray-cast camera over the cloth model with every robot link's visual mesh
            mirrored (GPU, no Vulkan needed); obs['sensor_data']['newton_camera']['rgb'] with obs_mode='state_dict'.
Newton runs on CUDA even when the robots use PhysX CPU (num_envs = 1).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from ..embodiments import agents as _agents  # noqa: F401  (registers the imsim agents)
from ..embodiments import dual as D
from ..soft import route_a as _ra  # noqa: F401  (render-backend parser patch)
from ..soft.cloth import ClothSolverSpec, NewtonClothWorlds, TowelSpec
from ..soft.colliders import Collider
from ..soft.embodiment_colliders import cloth_close_command, embodiment_colliders
from ..soft.metrics import footprint_area, speed_quantile

RELEASE_IMPULSE_NS = 5e-4
ENCLOSURE_MARGIN_M = 0.002


def _cloth_close(rec, gap: float) -> float:
    """gripper action that leaves `gap` between the pads: analytic for prismatic grippers (cloth_close_command); for
    linkage grippers (xArm6 + Robotiq) from the measured free-space calibration imsim/verify/grip_calib.json (E29b:
    opening vs the PD target of the first gripper joint, which for the Robotiq is the action value itself)"""
    try:
        return cloth_close_command(rec, gap)
    except ValueError:
        from ..verify import v32
        c = v32.grip_calib().get(rec.name)
        if not c:
            raise
        t, o = np.asarray(c["target"], float), np.asarray(c["opening"], float)
        i = np.argsort(o)
        return float(np.interp(gap, o[i], t[i]))


@register_env("DualFoldTowel-v1", max_episode_steps=800)
class DualFoldTowelEnv(BaseEnv):
    SUPPORTED_ROBOTS = [c.robot_uids for c in D.CONFIGS.values()]
    SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]

    def __init__(self, *args, dual_config: str = "dual_ur5_wsg", towel: TowelSpec | None = None,
                 cloth_solver: ClothSolverSpec | None = None, towel_center=(0.34, 0.0), rgb_source: str = "none",
                 camera_res: int = 128, cloth_gap: float = 0.004, camera_eye=(0.95, -0.55, 0.75),
                 camera_target=(0.30, 0.0, 0.0), camera_fovy: float = np.pi / 2, **kwargs):
        self.dual = D.CONFIGS[dual_config] if isinstance(dual_config, str) else dual_config
        self.towel_spec = towel or TowelSpec(center=tuple(towel_center))
        self.cloth_spec = cloth_solver or ClothSolverSpec()
        self.rgb_source, self.camera_res = rgb_source, int(camera_res)
        self.camera_eye, self.camera_target = tuple(camera_eye), tuple(camera_target)
        self.camera_fovy = float(camera_fovy)
        self.cloth: NewtonClothWorlds | None = None
        self.newton_camera = None
        self.base_table_links = [D.base_links_on_table(r, self.dual.base_z) for r in self.dual.records]
        self.cloth_gap = float(cloth_gap)
        self.cloth_close = [_cloth_close(r, cloth_gap) for r in self.dual.records]
        self._last_qpos = None
        self._robot_qvel_max = None
        self._init_area = None
        res = self.towel_spec.res
        self.corner_idx = dict(NR=0, FR=res, NL=res * (res + 1), FL=(res + 1) ** 2 - 1)
        super().__init__(*args, robot_uids=self.dual.robot_uids, **kwargs)

    # ---------------------------------------------------------------- config
    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2 ** 24, max_rigid_patch_count=2 ** 18,
                                                           max_rigid_contact_count=2 ** 20))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.95, 0.0, 0.65], target=[0.30, 0.0, 0.05])
        return [CameraConfig("front_camera", pose, 128, 128, np.pi / 2.5, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=list(self.camera_eye), target=list(self.camera_target))
        return CameraConfig("render_camera", pose, 512, 512, 0.95, 0.01, 100)

    # ---------------------------------------------------------------- scene
    def _load_agent(self, options: dict):
        super()._load_agent(options, self.dual.mount_poses())

    def _load_scene(self, options: dict):
        b = self.scene.create_actor_builder()
        hx, hy = (D.TABLE_X[1] - D.TABLE_X[0]) / 2, (D.TABLE_Y[1] - D.TABLE_Y[0]) / 2
        cx, cy = (D.TABLE_X[1] + D.TABLE_X[0]) / 2, (D.TABLE_Y[1] + D.TABLE_Y[0]) / 2
        half = [hx, hy, D.TABLE_THICK / 2]
        b.add_box_collision(half_size=half)
        b.add_box_visual(half_size=half, material=sapien.render.RenderMaterial(base_color=[0.78, 0.70, 0.60, 1]))
        b.initial_pose = sapien.Pose([cx, cy, D.TABLE_TOP_Z - D.TABLE_THICK / 2])
        self.table = b.build_kinematic(name="table")
        ags = list(self.agent.agents)
        self.group2_remaps = D.make_group2_bits_disjoint(ags)
        for ag, links in zip(ags, self.base_table_links):
            D.filter_base_table(ag, self.table, links)
        colliders, pads, groups, links = [], [], [], []
        for k, (ag, prefix) in enumerate(zip(ags, ("L:", "R:"))):
            cols, pp = embodiment_colliders(ag.record, prefix)
            for c in cols:
                colliders.append(c); groups.append(k); links.append(ag.robot.links_map[c.link[len(prefix):]])
            pads.append(pp)
        if self.rgb_source == "newton":         # visual-only mirror of every robot link for the Newton camera
            from ..render.meshes import robot_meshes
            by = {c.link: c for c in colliders}
            for k, (ag, prefix) in enumerate(zip(ags, ("L:", "R:"))):
                vis = robot_meshes(ag.record.urdf_path, [l.name for l in ag.robot.get_links()], max_faces=3000)
                for l in ag.robot.get_links():
                    key = prefix + l.name
                    if key in by:
                        by[key].visual = vis.get(l.name, [])
                    elif vis.get(l.name):
                        c = Collider(key, [], visual=vis[l.name])
                        colliders.append(c); groups.append(-1); links.append(l)
        for c, src in self._extra_cloth_colliders():   # E29b: static scene objects in the cloth model (group -1)
            colliders.append(c); groups.append(-1); links.append(src)
        self._links = links
        if self.cloth is None:
            dev = self.device if self.device.type == "cuda" else torch.device("cuda:0")
            self.cloth = NewtonClothWorlds(self.num_envs, colliders, pads, self.towel_spec, self.cloth_spec,
                                           control_dt=1.0 / self.control_freq, device=str(dev), collider_groups=groups)
            if self.rgb_source == "newton":
                from ..soft.render_sync import NewtonCamera
                self.newton_camera = NewtonCamera(self.cloth, width=self.camera_res, height=self.camera_res,
                                                  fovy=self.camera_fovy, eye=self.camera_eye, target=self.camera_target)

    def _extra_cloth_colliders(self) -> list:
        """[(Collider, SAPIEN actor / link whose pose drives it)] added to the cloth model after the robot links (not part
        of any arm's contact group); none for the fold"""
        return []

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            for ag, pose in zip(self.agent.agents, self.dual.mount_poses()):
                q = torch.as_tensor(ag.home_qpos_vector(), dtype=torch.float32).repeat(b, 1)
                ag.reset(q)
                ag.robot.set_pose(Pose.create_from_pq(torch.as_tensor(pose.p, dtype=torch.float32).repeat(b, 1),
                                                      torch.as_tensor(pose.q, dtype=torch.float32).repeat(b, 1)))
        mask = np.zeros(self.num_envs, np.int32)
        mask[env_idx.cpu().numpy()] = 1
        self.cloth.reset(mask)

    # ---------------------------------------------------------------- bridge
    def link_poses(self) -> torch.Tensor:
        return torch.stack([l.pose.raw_pose for l in self._links], dim=1).to("cuda:0")

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.cloth.set_link_poses(self.link_poses(), first=True)
        self.cloth._signals()
        q = self.cloth.particle_q()
        self._init_area = footprint_area(q[..., :2]).clone()
        self._last_qpos = [ag.robot.get_qpos().clone() for ag in self.agent.agents]
        self._robot_qvel_max = torch.zeros(self.num_envs, device=self.device)
        self._cloth_speed_pos = torch.zeros((2, self.num_envs), device=self.device)
        info = self.get_info()
        obs = self.get_obs(info)
        return obs, info

    def _step_action(self, action):
        action = super()._step_action(action)
        self.cloth.set_link_poses(self.link_poses())
        if self.cloth_spec.grasp_mode == "attach":   # E29b: attachment rule input (first pad pair = left gripper closed)
            ag = self.agent.agents[0]
            self.cloth.set_grip(ag.measured_opening() < self.cloth_gap + 0.0015)
        q_prev = self.cloth.particle_q().clone()
        self.cloth.step()
        # E29b (v3.2 for cloth): particle speed from POSITIONS over the control step (p95 and max over the particles)
        sp = torch.linalg.norm((self.cloth.particle_q() - q_prev).float(), dim=-1) * float(self.control_freq)
        sp = torch.nan_to_num(sp, nan=1e3, posinf=1e3)
        self._cloth_speed_pos = torch.stack([torch.quantile(sp, 0.95, dim=1), sp.amax(1)]).to(self.device)
        v = torch.zeros(self.num_envs, device=self.device)
        for k, ag in enumerate(self.agent.agents):
            q = ag.robot.get_qpos()
            idx = [i for i, j in enumerate(ag.robot.active_joints) if j.name in ag.record.arm_joints]
            if self._last_qpos is not None:
                v = torch.maximum(v, (q[:, idx] - self._last_qpos[k][:, idx]).abs().amax(1) * self.control_freq)
            if self._last_qpos is not None:
                self._last_qpos[k] = q.clone()
        self._robot_qvel_max = v
        return action

    def hold_robot(self):
        """protocol v3 hold A: both arms held at the joint positions reached now, grippers keep their last targets"""
        for ag in self.agent.agents:
            arm = ag.controller.controllers["arm"]
            q = ag.robot.get_qpos()[:, arm.active_joint_indices]
            arm._target_qpos = q.clone()
            arm.set_drive_targets(q)
        if self.gpu_sim_enabled:
            self.scene.px.gpu_apply_articulation_target_position()

    # ---------------------------------------------------------------- task
    def cloth_signals(self) -> dict[str, torch.Tensor]:
        s = {k: v.to(self.device) for k, v in self.cloth.signals().items()}
        q = self.cloth.particle_q().to(self.device)
        ci = self.corner_idx
        for n, i in ci.items():
            for a, ax in enumerate("xyz"):
                s[f"towel_{n}_{ax}"] = q[:, i, a]
        dxy = lambda a, b: torch.linalg.norm(q[:, ci[a], :2] - q[:, ci[b], :2], dim=-1)  # noqa: E731
        s["towel_dist_L"], s["towel_dist_R"] = dxy("FL", "NL"), dxy("FR", "NR")
        cen = q.mean(1)                                               # E29b: towel centroid (mean particle position)
        s["towel_cx"], s["towel_cy"], s["towel_cz"] = cen[:, 0], cen[:, 1], cen[:, 2]
        r = self.towel_spec.res
        V = q.reshape(self.num_envs, r + 1, r + 1, 3)               # [iy, ix]
        s["towel_lift_L"] = V[:, -3:, -3:, 2].amax((1, 2))          # FL patch (iy, ix high)
        s["towel_lift_R"] = V[:, :3, -3:, 2].amax((1, 2))           # FR patch (iy low, ix high)
        if self._init_area is not None:
            s["towel_footprint_ratio_gpu"] = (footprint_area(q[..., :2]) / self._init_area.to(self.device))
        s["cloth_speed_p95"] = speed_quantile(self.cloth.particle_qd()).to(self.device)
        csp = getattr(self, "_cloth_speed_pos", None)
        if csp is None:
            csp = torch.zeros((2, self.num_envs), device=self.device)
        s["cloth_speed_p95_pos"], s["cloth_speed_max_pos"] = csp[0], csp[1]
        s["robot_qvel_max"] = (self._robot_qvel_max if self._robot_qvel_max is not None
                               else torch.zeros(self.num_envs, device=self.device))
        s["released"] = ((s["robot_cloth_impulse"] <= RELEASE_IMPULSE_NS)
                         & (s["finger_cloth_enclosure"] > ENCLOSURE_MARGIN_M)).float()
        return s

    def evaluate(self) -> dict:
        if self.cloth is None:
            return {}
        s = self.cloth_signals()
        foot = s.get("towel_footprint_ratio_gpu", torch.ones(self.num_envs, device=self.device))
        folded = ((s["towel_dist_L"] < 0.08) & (s["towel_dist_R"] < 0.08) & (foot < 0.70)
                  & (s["towel_frac_above_5cm"] <= 0.02))
        success = folded & (s["released"] > 0.5) & (s["cloth_nonfinite"] == 0)
        return dict(success=success, folded=folded, fail=s["cloth_nonfinite"] > 0)

    def _get_obs_extra(self, info: dict):
        obs = {f"tcp_pose_{i}": ag.tcp.pose.raw_pose for i, ag in enumerate(self.agent.agents)}
        if "state" in self.obs_mode:
            q = self.cloth.particle_q().to(self.device)
            idx = list(self.corner_idx.values()) + [((self.towel_spec.res + 1) ** 2) // 2]
            obs["cloth_keypoints"] = q[:, idx, :].reshape(self.num_envs, -1)
        return obs

    def get_obs(self, info=None, unflattened: bool = False):
        obs = super().get_obs(info, unflattened=unflattened) if unflattened else super().get_obs(info)
        if self.newton_camera is not None and isinstance(obs, dict) and self.obs_mode == "state_dict":
            obs.setdefault("sensor_data", {})["newton_camera"] = {"rgb": self.newton_camera.render_all()}
        return obs

    def render(self):
        if self.newton_camera is not None and self.render_mode in (None, "rgb_array", "sensors"):
            return self.newton_camera.render_all()
        return super().render()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        s = self.cloth_signals()
        return 2.0 - torch.tanh(5.0 * s["towel_dist_L"]) - torch.tanh(5.0 * s["towel_dist_R"]) + info["success"].float()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 3.0
