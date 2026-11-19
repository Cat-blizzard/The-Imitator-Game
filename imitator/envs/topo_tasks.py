"""Card D2 tasks: two rigid tasks whose success is a topological relation, not a distance (single-arm ManiSkill Panda).

RingOnPeg-v1   a torus ring (core radius 4.5 cm, tube radius 7.5 mm; 16 capsules, 25 g) lies flat on the table; a 6 cm
               tall peg (radius 1.5 cm, kinematic, on the table) stands nearby. Success: the ring is around the peg (the
               peg passes through the ring's hole), released, the robot away.
MugOnHook-v1   a mug (outer radius 3.5 cm, height 9 cm, wall 5 mm; round handle: core radius 1.8 cm, tube radius 4 mm,
               hole radius 1.4 cm; angular damping 2.5/s) stands on the table; a rack (kinematic: base plate, 30 cm post,
               a 10 cm hook peg of radius 5 mm sticking out of the post towards -x at 20 cm height, tilted 5 deg up)
               stands to the robot's left. Success:
               the mug hangs from the hook by its handle (the hook passes through the handle's hole), released, the
               robot away.

Each env's evaluate() is the task's BENCHMARK-STYLE DISTANCE PREDICATE (the card's predicate (a), written before any
result, see research/results/d2_topology/prereg.json), evaluated by the harness at the policy's last step (bench column):
  RingOnPeg  robosuite 1.4.0 NutAssembly, verbatim thresholds (environments/manipulation/nut_assembly.py: on_peg
             l.366-379, _check_success l.626-646): |x_ring - x_peg| < 3 cm and |y_ring - y_peg| < 3 cm and
             z_ring < table + 5 cm (ring frame origin = ring centre, peg frame origin = peg foot), and the hand away:
             r_reach = 1 - tanh(10 |tcp - ring centre|) < 0.6 (i.e. |tcp - ring centre| > 4.24 cm).
  MugOnHook  RoboTwin hanging_mug.check_success (envs/hanging_mug.py l.81-88): handle point within 2 cm of the rack's
             middle point in x and in y (middle = mean of the rack origin and the rack's functional point; here the
             rack origin is the post foot and the functional point the hook tip, so the middle is the hook's midpoint in
             x-y), handle point higher than table + 12 cm (RoboTwin: 0.86 m with its table top at 0.74 m), and the
             gripper commanded open (RoboTwin reads the commanded gripper value; here the gripper PD target >= 39 mm).
The topology contracts (imitator/contracts/imsim_ring_on_peg_v1.json, imsim_mug_on_hook_v1.json) do not use these.

Geometry constants are class attributes so experts, recorders and contract builders read the same numbers.
"""
from __future__ import annotations

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from .stock_table import StockTableSceneBuilder as TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from .panda_ms import MAX_STEPS, _ImsimPanda


def _mat(rgba):
    return sapien.render.RenderMaterial(base_color=list(rgba))


def _ring_parts(R: float, a: float, n: int, plane: str = "xy", centre=(0.0, 0.0, 0.0)):
    """n capsules (pose, radius, half_length) approximating a torus of core radius R, tube radius a; the capsule ends
    meet on the core circle. plane 'xy': ring axis = local z; 'xz': ring axis = local y"""
    out = []
    c = np.asarray(centre, float)
    for k in range(n):
        th = 2 * np.pi * k / n
        rr = R * np.cos(np.pi / n)
        if plane == "xy":
            p = c + [rr * np.cos(th), rr * np.sin(th), 0.0]
            q = euler2quat(0, 0, th + np.pi / 2)              # capsule axis (local x) -> tangent
        else:                                                   # circle in the x-z plane, tangent (-sin, 0, cos)
            p = c + [rr * np.cos(th), 0.0, rr * np.sin(th)]
            q = euler2quat(0, -th - np.pi / 2, 0)
        out.append((sapien.Pose(p, q), a, R * np.sin(np.pi / n)))
    return out


class _TopoBase(_ImsimPanda, BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    TABLE_Z = 0.0

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, dual_config: str = "panda", **kwargs):
        self._imsim_init(dual_config)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _rng(self):
        return self._episode_rng

    def tcp_p(self) -> np.ndarray:
        return self.agent.tcp.pose.p[0].cpu().numpy().astype(float)

    def grip_target(self) -> float:
        c = self.agent.controller.controllers.get("gripper")
        t = getattr(c, "_target_qpos", None)
        return float(t[0, 0]) if t is not None else float(self.agent.robot.get_qpos()[0, -1])

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)

    def compute_dense_reward(self, obs, action, info: dict):
        return info["success"].float()

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs, action, info)


# ================================================================================================ RingOnPeg
class RingOnPegEnv(_TopoBase):
    """Put the ring over the peg, let go, move the hand away."""
    IMSIM_OBJ, IMSIM_HALF = "ring", 0.0525
    RING_R, RING_A, RING_N, RING_DENSITY = 0.045, 0.0075, 16, 500.0
    PEG_R, PEG_H = 0.015, 0.06
    PEG_XY = ((0.02, 0.10), (-0.10, 0.10))         # peg foot x range, y range
    RING_XY = ((-0.15, -0.06), (-0.14, 0.14))      # ring centre x range, y range
    MIN_SEP = 0.04                                  # min gap between the ring's outer edge and the peg surface at reset
    # predicate (a): robosuite 1.4.0 NutAssembly (verbatim thresholds)
    BENCH_XY_TOL, BENCH_Z_MAX, BENCH_REACH_MAX = 0.03, 0.05, 0.6

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.25, -0.45, 0.32], target=[-0.02, 0.0, 0.03])
        return CameraConfig("render_camera", pose, 512, 512, 0.9, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        b = self.scene.create_actor_builder()
        for pose, r, hl in _ring_parts(self.RING_R, self.RING_A, self.RING_N):
            b.add_capsule_collision(pose, radius=r, half_length=hl, density=self.RING_DENSITY)
            b.add_capsule_visual(pose, radius=r, half_length=hl, material=_mat([0.85, 0.35, 0.05, 1]))
        b.initial_pose = sapien.Pose([-0.1, 0, self.RING_A + 0.001])
        self.ring = b.build(name="ring")
        b = self.scene.create_actor_builder()
        cyl = sapien.Pose([0, 0, self.PEG_H / 2], euler2quat(0, -np.pi / 2, 0))     # cylinder axis (local x) -> z
        b.add_cylinder_collision(cyl, radius=self.PEG_R, half_length=self.PEG_H / 2)
        b.add_cylinder_visual(cyl, radius=self.PEG_R, half_length=self.PEG_H / 2, material=_mat([0.35, 0.35, 0.4, 1]))
        b.initial_pose = sapien.Pose([0.05, 0, 0])
        self.peg = b.build_kinematic(name="peg")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            rng = self._rng()
            (px0, px1), (py0, py1) = self.PEG_XY
            (rx0, rx1), (ry0, ry1) = self.RING_XY
            while True:
                peg = np.array([rng.uniform(px0, px1), rng.uniform(py0, py1)])
                ring = np.array([rng.uniform(rx0, rx1), rng.uniform(ry0, ry1)])
                if np.linalg.norm(peg - ring) >= self.RING_R + self.RING_A + self.PEG_R + self.MIN_SEP:
                    break
            yaw = rng.uniform(-np.pi, np.pi)
            self.peg.set_pose(Pose.create_from_pq(torch.tensor([[peg[0], peg[1], 0.0]]), torch.tensor([[1.0, 0, 0, 0]])))
            q = euler2quat(0, 0, yaw)
            self.ring.set_pose(Pose.create_from_pq(torch.tensor([[ring[0], ring[1], self.RING_A + 0.0005]]),
                                                   torch.tensor([list(q)], dtype=torch.float32)))

    # -------------------------------------------------------------------- predicate (a)
    def bench_terms(self) -> dict:
        r = self.ring.pose.p[0].cpu().numpy().astype(float)
        p = self.peg.pose.p[0].cpu().numpy().astype(float)
        on_peg = abs(r[0] - p[0]) < self.BENCH_XY_TOL and abs(r[1] - p[1]) < self.BENCH_XY_TOL and \
            r[2] < self.TABLE_Z + self.BENCH_Z_MAX
        r_reach = 1.0 - np.tanh(10.0 * float(np.linalg.norm(self.tcp_p() - r)))
        return dict(on_peg=bool(on_peg), r_reach=float(r_reach), success=bool(on_peg and r_reach < self.BENCH_REACH_MAX))

    def evaluate(self):
        t = self.bench_terms()
        return dict(success=torch.tensor([t["success"]]), on_peg=torch.tensor([t["on_peg"]]),
                    r_reach=torch.tensor([t["r_reach"]]))


# ================================================================================================ MugOnHook
class MugOnHookEnv(_TopoBase):
    """Hang the mug on the hook by its handle, let go, move the hand away."""
    IMSIM_OBJ, IMSIM_HALF = "mug", 0.045
    MUG_R, MUG_WALL, MUG_H, MUG_BOTTOM, MUG_N, MUG_DENSITY = 0.035, 0.005, 0.09, 0.006, 16, 800.0
    HANDLE_R, HANDLE_A, HANDLE_Z, HANDLE_N = 0.018, 0.004, 0.05, 12
    HANDLE_X = MUG_R + HANDLE_R - 0.002            # handle ring centre along the mug's local +x (in the mug frame)
    POST_HALF, POST_H, BASE_HALF, BASE_T = 0.01, 0.30, 0.05, 0.01
    HOOK_Z, HOOK_L, HOOK_R, HOOK_TILT_DEG = 0.20, 0.10, 0.005, 5.0
    RACK_XY = ((0.08, 0.14), (0.17, 0.23))         # clear of the Panda's hand at its start pose (TCP ~ (0.01, 0.04, 0.18))
    MUG_XY = ((-0.13, -0.05), (-0.20, -0.06))
    MUG_ANGULAR_DAMPING = 2.5                       # stands in for the handle-hook pivot friction PhysX does not model
                                                    # (rolling contact): a hung mug comes to rest within 4.5-7.5 s
    # predicate (a): RoboTwin hanging_mug.check_success (same structure and thresholds)
    BENCH_XY_TOL, BENCH_Z_MIN, BENCH_GRIP_OPEN = 0.02, 0.12, 0.039

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.30, -0.55, 0.45], target=[0.04, 0.0, 0.14])
        return CameraConfig("render_camera", pose, 512, 512, 0.9, 0.01, 100)

    # -------------------------------------------------------------------- geometry in the rack / mug frames
    @classmethod
    def hook_dir(cls) -> np.ndarray:
        a = np.radians(cls.HOOK_TILT_DEG)
        return np.array([-np.cos(a), 0.0, np.sin(a)])

    @classmethod
    def hook_root(cls) -> np.ndarray:
        return np.array([-cls.POST_HALF, 0.0, cls.HOOK_Z])

    @classmethod
    def hook_tip(cls) -> np.ndarray:
        """centre of the hook capsule's tip end-cap (the hook's free end is HOOK_R beyond it)"""
        return cls.hook_root() + cls.HOOK_L * cls.hook_dir()

    @classmethod
    def hook_polyline(cls) -> list:
        """the hook as a curve in the rack frame, from the fixed end (inside the post) to the free end (tip surface)"""
        return [[0.0, 0.0, cls.HOOK_Z], cls.hook_root().tolist(), (cls.hook_tip() + cls.HOOK_R * cls.hook_dir()).tolist()]

    @classmethod
    def handle_centre(cls) -> np.ndarray:
        return np.array([cls.HANDLE_X, 0.0, cls.HANDLE_Z])

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        # ---- mug (frame: bottom centre, z up; handle towards local +x, handle hole axis = local y)
        b = self.scene.create_actor_builder()
        white = _mat([0.92, 0.92, 0.88, 1]); red = _mat([0.75, 0.12, 0.1, 1])
        bot = sapien.Pose([0, 0, self.MUG_BOTTOM / 2], euler2quat(0, -np.pi / 2, 0))
        b.add_cylinder_collision(bot, radius=self.MUG_R, half_length=self.MUG_BOTTOM / 2, density=self.MUG_DENSITY)
        b.add_cylinder_visual(bot, radius=self.MUG_R, half_length=self.MUG_BOTTOM / 2, material=white)
        hw = self.MUG_R * np.tan(np.pi / self.MUG_N) * 1.05
        for k in range(self.MUG_N):
            th = 2 * np.pi * k / self.MUG_N
            rr = self.MUG_R - self.MUG_WALL / 2
            pose = sapien.Pose([rr * np.cos(th), rr * np.sin(th), self.MUG_H / 2], euler2quat(0, 0, th))
            hs = [self.MUG_WALL / 2, hw, self.MUG_H / 2]
            b.add_box_collision(pose, half_size=hs, density=self.MUG_DENSITY)
            b.add_box_visual(pose, half_size=hs, material=white)
        for pose, r, hl in _ring_parts(self.HANDLE_R, self.HANDLE_A, self.HANDLE_N, plane="xz",
                                       centre=self.handle_centre()):
            b.add_capsule_collision(pose, radius=r, half_length=hl, density=self.MUG_DENSITY)
            b.add_capsule_visual(pose, radius=r, half_length=hl, material=red)
        b.initial_pose = sapien.Pose([-0.1, -0.1, 0.001])
        self.mug = b.build(name="mug")
        for body in self.mug._bodies:
            body.angular_damping = self.MUG_ANGULAR_DAMPING
        # ---- rack (frame: post foot on the table)
        b = self.scene.create_actor_builder()
        wood = _mat([0.55, 0.38, 0.22, 1]); metal = _mat([0.3, 0.3, 0.32, 1])
        base = sapien.Pose([0, 0, self.BASE_T / 2])
        b.add_box_collision(base, half_size=[self.BASE_HALF, self.BASE_HALF, self.BASE_T / 2])
        b.add_box_visual(base, half_size=[self.BASE_HALF, self.BASE_HALF, self.BASE_T / 2], material=wood)
        post = sapien.Pose([0, 0, self.BASE_T + self.POST_H / 2])
        b.add_box_collision(post, half_size=[self.POST_HALF, self.POST_HALF, self.POST_H / 2])
        b.add_box_visual(post, half_size=[self.POST_HALF, self.POST_HALF, self.POST_H / 2], material=wood)
        a = np.radians(self.HOOK_TILT_DEG)
        hook = sapien.Pose(self.hook_root() + self.HOOK_L / 2 * self.hook_dir(), euler2quat(0, np.pi + a, 0))
        b.add_capsule_collision(hook, radius=self.HOOK_R, half_length=self.HOOK_L / 2)
        b.add_capsule_visual(hook, radius=self.HOOK_R, half_length=self.HOOK_L / 2, material=metal)
        b.initial_pose = sapien.Pose([0.13, 0.06, 0])
        self.rack = b.build_kinematic(name="rack")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            rng = self._rng()
            (ax0, ax1), (ay0, ay1) = self.RACK_XY
            (mx0, mx1), (my0, my1) = self.MUG_XY
            rack = np.array([rng.uniform(ax0, ax1), rng.uniform(ay0, ay1)])
            mug = np.array([rng.uniform(mx0, mx1), rng.uniform(my0, my1)])
            yaw = rng.uniform(-np.pi, np.pi)
            self.rack.set_pose(Pose.create_from_pq(torch.tensor([[rack[0], rack[1], 0.0]]), torch.tensor([[1.0, 0, 0, 0]])))
            q = euler2quat(0, 0, yaw)
            self.mug.set_pose(Pose.create_from_pq(torch.tensor([[mug[0], mug[1], 0.0005]]),
                                                  torch.tensor([list(q)], dtype=torch.float32)))

    # -------------------------------------------------------------------- world quantities
    def handle_world(self) -> np.ndarray:
        T = self.mug.pose.sp.to_transformation_matrix()
        return T[:3, :3] @ self.handle_centre() + T[:3, 3]

    def rack_T(self) -> np.ndarray:
        return self.rack.pose.sp.to_transformation_matrix()

    def bench_terms(self) -> dict:
        T = self.rack_T()
        tip = T[:3, :3] @ self.hook_tip() + T[:3, 3]
        middle = (T[:3, 3] + tip) / 2
        h = self.handle_world()
        xy_ok = abs(h[0] - middle[0]) < self.BENCH_XY_TOL and abs(h[1] - middle[1]) < self.BENCH_XY_TOL
        z_ok = h[2] > self.TABLE_Z + self.BENCH_Z_MIN
        grip_open = self.grip_target() >= self.BENCH_GRIP_OPEN
        return dict(xy_ok=bool(xy_ok), z_ok=bool(z_ok), grip_open=bool(grip_open),
                    success=bool(xy_ok and z_ok and grip_open))

    def evaluate(self):
        t = self.bench_terms()
        return dict(success=torch.tensor([t["success"]]), xy_ok=torch.tensor([t["xy_ok"]]),
                    z_ok=torch.tensor([t["z_ok"]]), grip_open=torch.tensor([t["grip_open"]]))


def _register(cls, name):
    W = type(f"Imsim{name}Env", (cls,), {})
    register_env(f"Imsim{name}-v1", max_episode_steps=MAX_STEPS)(W)
    return W


ImsimRingOnPegEnv = _register(RingOnPegEnv, "RingOnPeg")
ImsimMugOnHookEnv = _register(MugOnHookEnv, "MugOnHook")
